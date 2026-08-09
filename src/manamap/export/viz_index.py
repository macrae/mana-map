"""Step 14: The two artifacts the discovery front door runs on.

The map used to be the front door: 34,322 points and a request that you already
know where to look. Discovery inverts that — land on one card, click to reveal
its neighbours, grow a graph. That only feels right if the click is *instant*,
and instant rules out both artifacts the browser currently leans on:

    projection_2d.json      12.9 MB raw    2.90 MB gzipped
    embeddings_ability.bin  16.8 MB raw   15.54 MB gzipped   <- float32, incompressible

The embedding matrix is the killer. It is needed on the very first click a new
visitor makes, it compresses by 7%, and scanning it costs 4.4M multiply-adds on
the main thread per branch. Three relation types read live would have meant ~48 MB
of lazy fetches and an `await` inside every gesture.

So this step precomputes what branching actually needs — the top few neighbours
per card, per relation — into two small files:

    viz_index.json    ~0.4 MB gzipped   pick a card, filter, resolve a name to a row
    neighbours.bin    see below         branch synchronously, no await, no matrix

Deliberately absent from `viz_index`: oracle text, type line, mana cost, P/T. The
Scryfall card image already shows all of it, so a local copy is weight the user
never sees. That is also why the landing can paint from a name alone.

**The one rule this file's output imposes on its readers: the neighbour lists are
stored pre-sorted and must never be re-sorted client-side.** Similarities here are
quantised, and this embedding is a narrow cone — median pairwise cosine 0.714 with
a 1st-to-50th spread of 0.03. Re-sorting by a lossy value changes the top-10 for
roughly two thirds of cards, which would look like a model regression rather than
a precision artefact. Order is the payload; the values are for edge length only.
"""

import hashlib
import json
import struct

import numpy as np
import pandas as pd

from manamap.analysis.common import top_k_similar
from manamap.config import (
    ABILITY_EMBEDDINGS_PATH,
    CARD_ROLES_PATH,
    NEIGHBOURS_BIN_PATH,
    NEIGHBOURS_FORMAT_VERSION,
    NEIGHBOURS_HEADER_BYTES,
    NEIGHBOURS_K_OBSOLETE,
    NEIGHBOURS_K_SIMILAR,
    NEIGHBOURS_K_SYNERGY,
    NEIGHBOURS_MAGIC,
    NEIGHBOURS_NONE,
    NEIGHBOURS_NO_REASON,
    OBSOLESCENCE_INDEX_PATH,
    OUTPUT_CSV_PATH,
    SYNERGY_GRAPH_PATH,
    SYNERGY_RULES,
    VIZ_INDEX_PATH,
)


def build_name_index(names):
    """{name: row}. First printing wins, matching every other name lookup here.

    51 names duplicate across the corpus. Which row a duplicate resolves to only
    matters for consistency, not correctness — the rows carry identical oracle
    data — so this mirrors `analysis/eval_embeddings.resolve_groups`.

    NOT the same function as `analysis/common.build_name_index`, which is
    LAST-write-wins. Same name, opposite tie-break, on purpose: that one feeds
    positional embedding lookups, this one feeds the browser. Do not unify.
    """
    index = {}
    for row, name in enumerate(names):
        index.setdefault(name, row)
    return index


def build_viz_index(df, roles_by_name):
    """The slim per-card record: everything needed to pick, filter and resolve.

    `rarity` is here because `MM.categoryColor` reads it; role tags are here to
    answer "why is this card next to that one" without another fetch. Oracle text
    is *not* here on purpose — the card image already carries it.
    """
    records = []
    for row in df.itertuples(index=False):
        rec = {
            "n": row.name,
            "s": row.supertype if isinstance(row.supertype, str) else "Unknown",
            "c": row.primary_color if isinstance(row.primary_color, str) else "Colorless",
            "r": row.rarity if isinstance(row.rarity, str) else "",
            "m": 0.0 if pd.isna(row.cmc) else float(row.cmc),
        }
        tags = roles_by_name.get(row.name)
        if tags:
            rec["g"] = tags
        records.append(rec)
    return records


def _pad(rows, k):
    """Fixed-width slot array; unused slots carry the sentinel."""
    out = list(rows[:k])
    return out + [NEIGHBOURS_NONE] * (k - len(out)), len(out)


def build_tables(df, embeddings, synergy, obsolescence, name_index):
    """Per-card neighbour rows for all three relations, plus true counts.

    Similarity is computed here rather than read from a graph; synergy and
    obsolescence are name-keyed graphs, so partner names resolve through
    `name_index` and anything unresolvable is simply dropped — a partner naming a
    card that is not in this corpus is a stale graph, not a fatal error.
    """
    n = len(df)
    names = df["name"].tolist()
    reason_index = {label: i for i, (_, _, label) in enumerate(SYNERGY_RULES)}
    if len(reason_index) >= NEIGHBOURS_NO_REASON:
        raise SystemExit(
            f"{len(reason_index)} synergy reasons no longer fit in a uint8 slot "
            f"(sentinel is {NEIGHBOURS_NO_REASON}). Widen the block or drop the sentinel."
        )

    sim_idx = np.full((n, NEIGHBOURS_K_SIMILAR), NEIGHBOURS_NONE, dtype=np.uint16)
    sim_raw = np.zeros((n, NEIGHBOURS_K_SIMILAR), dtype=np.float32)
    syn_idx = np.full((n, NEIGHBOURS_K_SYNERGY), NEIGHBOURS_NONE, dtype=np.uint16)
    syn_reason = np.full((n, NEIGHBOURS_K_SYNERGY), NEIGHBOURS_NO_REASON, dtype=np.uint8)
    obs_idx = np.full((n, NEIGHBOURS_K_OBSOLETE), NEIGHBOURS_NONE, dtype=np.uint16)
    counts = np.zeros((n, 3), dtype=np.uint8)

    for row in range(n):
        top = top_k_similar(embeddings, row, NEIGHBOURS_K_SIMILAR)
        for slot, (other, score) in enumerate(top):
            sim_idx[row, slot] = other
            sim_raw[row, slot] = score
        counts[row, 0] = len(top)

        name = names[row]

        partners = [p for p in (synergy.get(name) or [])
                    if p.get("partner") in name_index]
        resolved = [name_index[p["partner"]] for p in partners]
        slots, k = _pad(resolved, NEIGHBOURS_K_SYNERGY)
        syn_idx[row] = slots
        counts[row, 1] = k
        # The FIRST reason only. A partner can match several rules, but an edge label
        # gets one line — and the full list is still in synergy_graph.json for anything
        # that wants it. Rules are in a fixed order, so "first" is deterministic.
        for slot, p in enumerate(partners[:NEIGHBOURS_K_SYNERGY]):
            labels = p.get("synergies") or []
            if labels and labels[0] in reason_index:
                syn_reason[row, slot] = reason_index[labels[0]]

        entry = obsolescence.get(name)
        better = entry.get("obsoleted_by", []) if entry else []
        resolved = [name_index[b["name"]] for b in better if b.get("name") in name_index]
        slots, k = _pad(resolved, NEIGHBOURS_K_OBSOLETE)
        obs_idx[row] = slots
        counts[row, 2] = k

        if row and row % 5000 == 0:
            print(f"    {row:,}/{n:,} cards…")

    # Quantise similarity to uint8 over the observed range, which is carried in the
    # header so the file stays self-describing. Measured max error 0.00027 cosine —
    # invisible in an edge length, and it halves the block's gzipped size (0.61 ->
    # 0.27 MB) because uint16 quantisation is nearly incompressible noise. These
    # values are for edge length ONLY; ordering lives in the array order.
    lo = float(sim_raw[counts[:, 0] > 0].min())
    hi = float(sim_raw.max())
    span = (hi - lo) or 1.0
    sim_val = np.round((sim_raw - lo) / span * 255).clip(0, 255).astype(np.uint8)
    return sim_idx, sim_val, syn_idx, syn_reason, obs_idx, counts, lo, hi


def pack(n, digest, sim_idx, sim_val, syn_idx, syn_reason, obs_idx, counts,
         sim_lo, sim_hi):
    """Header, then every uint16 block, then every uint8 block. Little-endian.

    The ordering is load-bearing, not tidiness. A `Uint16Array` view on an odd byte
    offset throws in JS — at page load, nowhere near the cause. Header is padded to
    64 bytes and all uint16 blocks are contiguous after it, so alignment holds for
    ANY k values. Interleaving the uint8 similarity block between uint16 blocks
    would work today only because K_SIMILAR happens to be even; changing it to an
    odd number would break the file silently.
    """
    vocab = json.dumps([label for _, _, label in SYNERGY_RULES],
                       separators=(",", ":")).encode("utf-8")
    header = struct.pack(
        "<4sIIHHHH32sffI",
        NEIGHBOURS_MAGIC,
        NEIGHBOURS_FORMAT_VERSION,
        n,
        NEIGHBOURS_K_SIMILAR,
        NEIGHBOURS_K_SYNERGY,
        NEIGHBOURS_K_OBSOLETE,
        0,
        digest,
        sim_lo,
        sim_hi,
        len(vocab),
    )
    header = header.ljust(NEIGHBOURS_HEADER_BYTES, b"\0")
    assert len(header) == NEIGHBOURS_HEADER_BYTES

    return b"".join([
        header,
        sim_idx.astype("<u2").tobytes(),
        syn_idx.astype("<u2").tobytes(),
        obs_idx.astype("<u2").tobytes(),
        sim_val.astype(np.uint8).tobytes(),
        syn_reason.astype(np.uint8).tobytes(),
        counts.tobytes(),
        # The reason vocabulary rides along rather than becoming a third fetch. The
        # header was already padded to 64 bytes, so its length fits in the spare four
        # and the file stays self-describing.
        vocab,
    ])


def embeddings_digest(path):
    """sha256 of the embedding matrix these neighbours were derived from.

    A stale derived artifact still parses and still returns confident, wrong
    answers — which is exactly the failure the DATA_VERSION comment in
    mana-map.js already memorialises. The digest lets a test say so out loud.
    """
    return hashlib.sha256(path.read_bytes()).digest()


def main():
    print("Loading inputs...")
    df = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
    embeddings = np.load(ABILITY_EMBEDDINGS_PATH)
    if len(embeddings) != len(df):
        raise SystemExit(
            f"index alignment broken: {len(embeddings):,} embeddings vs {len(df):,} cards. "
            f"Re-run the pipeline from the changed step onward."
        )

    with open(SYNERGY_GRAPH_PATH, encoding="utf-8") as fh:
        synergy = json.load(fh)
    with open(OBSOLESCENCE_INDEX_PATH, encoding="utf-8") as fh:
        obsolescence = json.load(fh)
    roles_by_name = {}
    if CARD_ROLES_PATH.exists():
        with open(CARD_ROLES_PATH, encoding="utf-8") as fh:
            roles_by_name = json.load(fh)["roles"]

    names = df["name"].tolist()
    name_index = build_name_index(names)
    print(f"  {len(df):,} cards · {len(synergy):,} synergy entries · "
          f"{len(obsolescence):,} obsolescence entries")

    print("\nWriting viz_index.json...")
    records = build_viz_index(df, roles_by_name)
    with open(VIZ_INDEX_PATH, "w", encoding="utf-8") as fh:
        json.dump(records, fh, separators=(",", ":"), ensure_ascii=False)
    print(f"  {VIZ_INDEX_PATH} — {VIZ_INDEX_PATH.stat().st_size / 1048576:.2f} MB")

    print("\nBuilding neighbour tables...")
    # Unpacked by name, not by index. This was `counts = tables[4]` with a `*tables`
    # splat, so adding the reason block silently shifted `counts` onto `obs_idx` and the
    # coverage report claimed 100% for every relation — wrong numbers, no error.
    (sim_idx, sim_val, syn_idx, syn_reason, obs_idx, counts, sim_lo, sim_hi) = build_tables(
        df, embeddings, synergy, obsolescence, name_index
    )

    blob = pack(len(df), embeddings_digest(ABILITY_EMBEDDINGS_PATH),
                sim_idx, sim_val, syn_idx, syn_reason, obs_idx, counts, sim_lo, sim_hi)
    NEIGHBOURS_BIN_PATH.write_bytes(blob)
    print(f"  {NEIGHBOURS_BIN_PATH} — {len(blob) / 1048576:.2f} MB")

    have_syn = int((counts[:, 1] > 0).sum())
    have_obs = int((counts[:, 2] > 0).sum())
    neither = int(((counts[:, 1] == 0) & (counts[:, 2] == 0)).sum())
    print(f"\n  similar:     {len(df):,} cards (100.0%)")
    print(f"  synergy:     {have_syn:,} cards ({have_syn / len(df) * 100:.1f}%)")
    print(f"  obsolescence:{have_obs:,} cards ({have_obs / len(df) * 100:.1f}%)")
    print(f"  only similar:{neither:,} cards ({neither / len(df) * 100:.1f}%) "
          f"— the UI must state this rather than offer dead buttons")


if __name__ == "__main__":
    main()
