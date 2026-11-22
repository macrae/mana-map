"""Step 13b: the INTERACTION space — cards alike by the events they consume and produce, and the
directed relation of one card feeding another (2026-11-22; docs/research/engine-model.md §4).

    manamap interaction          # the pipeline step: card_events.json, then the matrix

Two relations, kept apart on purpose:

  * **alike(A, B)** — cosine over IDF-weighted signatures. Two death payoffs are alike;
    a death payoff and a sacrifice outlet are not. It is a SPACE (`spaces.INTERACTION`,
    `data/embeddings_interaction.npy`, 128-d, corpus order) so every kernel that reads
    a space — `top_k_similar`, `centroid`, `deck_map.cosine_distances` — reads it.
  * **feeds(A → B)** — the overlap of what A PRODUCES with what B CONSUMES, over event
    kinds. Directed and a FUNCTION, never a space: Ashnod's Altar feeds Blood Artist,
    Blood Artist does not feed Ashnod's Altar.

A card's SIGNATURE is a bag of tokens read by `pilot.ability_graph` from its Forge
script, the parser the engine graph uses:

    C:<EVENT>  C:<EVENT>:<actor>        what it consumes (a trigger) — coarse and fine
    K:<STOCK>                            a cost that eats a stock (BODIES, HAND, LIFE)
    P:<EVENT>  P:<EVENT>:<actor>        what it produces (explicit abilities only)
    S:<EVENT>                            an event it scales (a doubler, an anthem)
    L:<IN>><OUT>                         one per ability: the conversion it performs

Implicit productions (every nonland is cast, every permanent enters) are NOT in the
signature: they are what every card does, and IDF would weigh them to nothing anyway.

`data/card_events.json` is TRACKED — `{meta, cards{name: {sig[], reader, partial}}}` —
because Forge lives outside the repo: a fresh clone without Forge still builds the
matrix from it. The matrix is gitignored and rebuilt in seconds. Zero LLM calls.
"""
import json
import math
from collections import Counter

import numpy as np

from manamap import config

CARD_EVENTS_PATH = config.DATA_DIR / "card_events.json"
INTERACTION_EMBEDDINGS_PATH = config.DATA_DIR / "embeddings_interaction.npy"
#: A token must appear on at least this many cards to be a dimension: a token on one
#: card says only "this card", which is what its name already says.
MIN_DF = 3
DIM = 128
SEED = 0


# ── signatures ───────────────────────────────────────────────────────────────

def signature(parsed):
    """The token bag of one `ability_graph.card_abilities` row, sorted."""
    toks = []
    for a in parsed.get("abilities") or []:
        if a.get("kind") in ("implicit",):
            continue
        c = a.get("consumes")
        ins = []
        if c:
            toks += [f"C:{c['event']}", f"C:{c['event']}:{c.get('actor') or 'any'}"]
            ins.append(c["event"])
        for k in a.get("costs") or []:
            if k.get("stock"):
                toks.append(f"K:{k['stock']}")
                ins.append(k["stock"])
        outs = []
        for p in a.get("produces") or []:
            toks += [f"P:{p['event']}", f"P:{p['event']}:{p.get('actor') or 'any'}"]
            outs.append(p["event"])
        for k in a.get("costs") or []:
            if k.get("event") in ("DIES", "DISCARD") and k.get("stock"):
                toks.append(f"P:{k['event']}")             # a sacrifice cost produces the death
                outs.append(k["event"])
        sc = a.get("scales")
        if sc:
            toks.append(f"S:{sc['event']}")
            outs.append(f"x{sc['event']}")
        for i in ins or ["CAST" if a.get("kind") == "spell" else "ACT"]:
            for o in outs:
                toks.append(f"L:{i}>{o}")
    return sorted(toks)


def kinds(sig, side):
    """The coarse event kinds on one side ('P' produces, 'C' consumes) of a signature."""
    out = Counter()
    for t in sig:
        parts = t.split(":")
        if parts[0] == side and len(parts) == 2:
            out[parts[1]] += 1
        if side == "C" and parts[0] == "K":
            out[{"BODIES": "DIES", "HAND": "DISCARD", "LIFE": "LIFE_LOSS"}.get(parts[1], parts[1])] += 1
    return out


def feeds(sig_a, sig_b):
    """feeds(A → B) in [0, 1]: cosine of A's produced kinds with B's consumed kinds."""
    pa, cb = kinds(sig_a, "P"), kinds(sig_b, "C")
    num = sum(pa[k] * cb[k] for k in pa if k in cb)
    if not num:
        return 0.0
    den = math.sqrt(sum(v * v for v in pa.values())) * math.sqrt(sum(v * v for v in cb.values()))
    return round(num / den, 4)


# ── card_events.json ─────────────────────────────────────────────────────────

def corpus_names():
    from manamap.pilot.card_pool import load_frame
    frame = load_frame()
    if frame is None:
        raise SystemExit("cards.csv is absent — run `manamap extract` first.")
    return frame["name"].tolist()


def build_card_events(names=None):
    """Parse every corpus card's Forge script (one zip open) into its signature."""
    from manamap.pilot import ability_graph as ag
    from manamap.sim import forge_cards
    if not forge_cards.installed():
        raise SystemExit("Forge's card scripts are not installed at FORGE_HOME — card_events.json "
                         "is built from them; the committed copy stands.")
    names = list(dict.fromkeys(names if names is not None else corpus_names()))
    texts = forge_cards.scripts(names)
    cards = {}
    readers = Counter()
    for n in names:
        parsed = ag.card_abilities(n, texts.get(n), None)
        readers[parsed["reader"]] += 1
        row = {"sig": signature(parsed), "reader": parsed["reader"]}
        if parsed.get("partial"):
            row["partial"] = True
        cards[n] = row
    return {"meta": {"ability_graph_version": config.ABILITY_GRAPH_VERSION,
                     "forge_digest": forge_cards.cardsfolder_digest(),
                     "cards": len(cards), "readers": dict(sorted(readers.items()))},
            "cards": cards}


def load_card_events():
    if not CARD_EVENTS_PATH.exists():
        return None
    return json.loads(CARD_EVENTS_PATH.read_text(encoding="utf-8"))


# ── the matrix ───────────────────────────────────────────────────────────────

def tfidf(sigs, min_df=MIN_DF):
    """(matrix rows = sigs, vocabulary) with w = sqrt(tf) · idf, idf = log((N+1)/(df+1)) + 1,
    tokens under `min_df` dropped, rows L2-normalised (an empty row stays zero)."""
    from scipy import sparse
    df = Counter(t for s in sigs for t in set(s))
    vocab = sorted(t for t, d in df.items() if d >= min_df)
    col = {t: i for i, t in enumerate(vocab)}
    n = len(sigs)
    idf = np.array([math.log((n + 1) / (df[t] + 1)) + 1 for t in vocab], dtype=np.float64)
    rows, cols, vals = [], [], []
    for r, s in enumerate(sigs):
        for t, tf in Counter(s).items():
            if t in col:
                rows.append(r)
                cols.append(col[t])
                vals.append(math.sqrt(tf) * idf[col[t]])
    m = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)), dtype=np.float64)
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    m = sparse.diags(1.0 / norms) @ m
    return m, vocab


def alike_matrix(events=None, names=None, min_df=None):
    """The 128-d interaction embedding for every corpus ROW (printings share their
    name's vector), by TF-IDF then a seeded truncated SVD, rows L2-normalised. A card
    with an empty signature is a zero row, and the second return value names them."""
    from sklearn.decomposition import TruncatedSVD
    events = events or load_card_events()
    if events is None:
        raise SystemExit(f"{CARD_EVENTS_PATH.name} is absent — `manamap interaction` builds it (needs Forge).")
    names = names if names is not None else corpus_names()
    uniq = list(dict.fromkeys(names))
    sigs = [(events["cards"].get(n) or {}).get("sig") or [] for n in uniq]
    m, vocab = tfidf(sigs, min_df=MIN_DF if min_df is None else min_df)
    if m.shape[1] < 2:
        raise SystemExit(f"the signature vocabulary has {m.shape[1]} token(s) — too few to embed")
    k = min(DIM, m.shape[1] - 1, m.shape[0] - 1)
    z = TruncatedSVD(n_components=k, random_state=SEED, algorithm="randomized", n_iter=7).fit_transform(m)
    if k < DIM:
        z = np.hstack([z, np.zeros((z.shape[0], DIM - k))])
    norms = np.linalg.norm(z, axis=1)
    empty = [uniq[i] for i in np.where(norms == 0)[0]]
    norms[norms == 0] = 1.0
    z = (z / norms[:, None]).astype(np.float32)
    by_name = {n: i for i, n in enumerate(uniq)}
    return z[[by_name[n] for n in names]], {"vocab": len(vocab), "empty": empty, "unique": len(uniq)}


def main(args=None):
    if getattr(args, "matrix_only", False) is False:
        try:
            events = build_card_events()
            CARD_EVENTS_PATH.write_text(json.dumps(events, ensure_ascii=False, sort_keys=True,
                                                   separators=(",", ":")) + "\n", encoding="utf-8")
            print(f"  wrote {CARD_EVENTS_PATH.name}: {events['meta']['cards']} cards, readers {events['meta']['readers']}")
        except SystemExit as exc:
            if load_card_events() is None:
                raise
            print(f"  {exc}")
    z, info = alike_matrix()
    np.save(INTERACTION_EMBEDDINGS_PATH, z)
    print(f"  wrote {INTERACTION_EMBEDDINGS_PATH.name}: {z.shape[0]} rows × {z.shape[1]}, vocabulary "
          f"{info['vocab']} tokens (df ≥ {MIN_DF}), {len(info['empty'])} of {info['unique']} names with no "
          f"signature (zero rows: vanilla, keyword-only, or unread)")
    return 0
