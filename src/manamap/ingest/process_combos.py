"""Step 8: Process raw Commander Spellbook data into the combo artifacts.

Three files, because they have three different audiences:

- `combo_graph.json` — `{"partners": {name: [names]}, "meta": {"source_timestamp"}}`,
  the partner adjacency map. It was the only thing the viz deck builder read
  (`graph.partners[name]`); that builder is deleted, so today it is read by
  `analysis/synergy.py` (to exclude known partners) and by `config.py`, which
  uses it as the invalidation proxy for its larger sibling. Kept small on the
  old grounds anyway.

  MEASURED AND NOT KEPT (2026-08-24): Spellbook carries a `description` on
  **every** variant — 83,261 of 83,261 — a numbered walk-through of how the
  combo works, median 405 characters. Writing it into `combo_details.json`
  costs **+36.4 MB on a 25.7 MB tracked file**, and `data/` carries no LFS
  because Pages serves pointers. Gzipped as an id-keyed sidecar it is 4.4 MB
  (8.4x, the same ratio the raw dumps get), which is affordable — but at the
  time of measuring it had no reader: the browser's combo prose comes from the
  per-deck stack artifacts it already fetches, and those cover 50 of 50
  published lines. The consumer that WOULD justify it is `engine_facts`, whose
  contained-combo lines are handed to `deck-engineer` with no explanation of
  how each one works. Do it when that agent is the one asking, key the sidecar
  by Spellbook `id` and never by position — an index-aligned sidecar is the
  `projection[i] == cards.csv[i]` hazard with none of the pipeline discipline
  that keeps that one honest.
- `combo_details.json` — the full combo records plus a card→combo index. Read
  by Python and by agents, never by the browser. This is where the power-level
  signal lives: Spellbook tags every variant with a bracket letter, which is
  what lets `pilot/bracket.py` compute a deck's bracket floor. Every record
  carries Spellbook's `id` (2026-10-08), so a sidecar or a page can name a
  combo without naming its position; `meta.source` is the dump's own
  `{timestamp, version}` from `.combos-meta.json`, None when the sidecar
  predates the bulk route.
- `combo_index.json` — the browser-sized cut (`build_combo_index`): per card,
  the TRUE totals (`n`, `inf`) and the top `COMBO_INDEX_PER_CARD` combos by
  popularity, pointing into a compact row list that holds only the combos some
  card's top list names. A card page can say "in 212 combos, 180 of them
  infinite, here are the twelve people run" without the 25 MB file.

The raw dump is read through `raw_variants`, which takes BOTH shapes step 7 has
written: the bulk file's `{timestamp, version, variants, aliases}` (the default
since 2026-10-08) and the paged route's bare list. Neither is migrated.

The graph stays format-agnostic by design — Commander-banned combos are kept
and flagged (`banned: true`), not dropped. Filtering happens at consumption.
Spellbook's `status` is NOT filtered on either: the 2026-04 dump is 83,261 of
83,261 `OK`, and the 1,375 commander-illegal variants are exactly the `B`
bracket tag, which already carries its flag. Any other status is counted and
printed so a bulk file that starts carrying drafts is seen, not silently kept.
"""

import json
from collections import Counter, defaultdict

import pandas as pd

from manamap.ingest.common import open_dump
from manamap.config import (
    COMBO_BANNED_TAG,
    COMBO_BRACKET_TAGS,
    COMBO_DETAILS_PATH,
    COMBO_GRAPH_PATH,
    COMBO_INDEX_PATH,
    COMBO_INDEX_PER_CARD,
    COMBOS_META_PATH,
    COMBOS_RAW_PATH,
    OUTPUT_CSV_PATH,
)

#: Spellbook's "this variant is live" status; anything else is reported, not dropped.
STATUS_OK = "OK"
#: Same predicate as `pilot.deck_combos.is_infinite` (prefix `pilot.deck_combos.INFINITE_PREFIX`),
#: copied rather than imported because `ingest/` sits below `pilot/` — a test
#: holds the two to the same answer.
INFINITE_PREFIX = "infinite"


def raw_variants(doc):
    """The variant list out of whichever dump shape step 7 wrote.

    The bulk file is `{timestamp, version, variants, aliases}`; the paged route
    writes the list itself. Anything else is a malformed dump, and says so.
    """
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict) and isinstance(doc.get("variants"), list):
        return doc["variants"]
    raise ValueError("combos dump is neither a variant list nor {variants: [...]}")


def load_source_meta(path=None):
    """`{timestamp, version}` of the dump, from the step-7 sidecar; None when unknown."""
    path = COMBOS_META_PATH if path is None else path
    try:
        raw = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    return {"timestamp": raw.get("timestamp"), "version": raw.get("version")}


def status_counts(combos):
    """How many variants carry each `status` — printed by step 8, never filtered on."""
    return Counter(combo.get("status") for combo in combos)


def load_known_cards(csv_path):
    """Load set of known card names from cards.csv."""
    df = pd.read_csv(csv_path, usecols=["name"])
    return set(df["name"].dropna().str.strip())


def extract_card_names(combo):
    """Extract card names from a combo variant's 'uses' array."""
    uses = combo.get("uses", [])
    names = []
    for use in uses:
        card = use.get("card", {})
        name = card.get("name", "").strip()
        if name:
            names.append(name)
    return names


def extract_color_identity(combo):
    """Extract color identity string from combo."""
    identity = combo.get("identity", "")
    if isinstance(identity, str):
        return identity.upper()
    return ""


def extract_produces(combo):
    """Extract what a combo produces from the 'produces' array."""
    produces = combo.get("produces", [])
    results = []
    for prod in produces:
        feature = prod.get("feature", {})
        name = feature.get("name", "").strip()
        if name:
            results.append(name)
    return results


def extract_bracket(combo):
    """Map Spellbook's bracket letter onto the WotC ladder.

    Returns (bracket, banned). `bracket` is None for the banned tag and for any
    letter we don't recognize — an unknown letter must not silently read as
    bracket 1, or a new Spellbook tag would quietly under-report a deck's floor.
    """
    tag = combo.get("bracketTag")
    if tag == COMBO_BANNED_TAG:
        return None, True
    return COMBO_BRACKET_TAGS.get(tag), False


def is_infinite(record):
    """Does a processed combo record produce an unbounded loop? (`pilot.deck_combos.is_infinite`)"""
    return any(str(p).lower().startswith(INFINITE_PREFIX) for p in record.get("produces", []))


def build_combo_graph(combos, known_cards):
    """Build partners adjacency map and combo detail list.

    Only includes combos where ALL cards exist in our dataset.
    """
    partners = defaultdict(set)
    combo_list = []

    for combo in combos:
        card_names = extract_card_names(combo)
        if len(card_names) < 2:
            continue

        # Check all cards exist in our dataset
        if not all(name in known_cards for name in card_names):
            continue

        # Build partner adjacency (every card partners with every other card)
        for i, name in enumerate(card_names):
            for j, other in enumerate(card_names):
                if i != j:
                    partners[name].add(other)

        # Build combo detail record
        ci = extract_color_identity(combo)
        produces = extract_produces(combo)
        bracket, banned = extract_bracket(combo)

        record = {
            "id": combo.get("id"),
            "cards": card_names,
            "produces": produces,
            "ci": ci,
            "bracket": bracket,
            "mana_value_needed": combo.get("manaValueNeeded"),
            "popularity": combo.get("popularity"),
        }
        if banned:
            record["banned"] = True
        combo_list.append(record)

    # Convert sets to sorted lists for JSON serialization
    partners_dict = {k: sorted(v) for k, v in partners.items()}

    return partners_dict, combo_list


def build_card_index(combo_list):
    """Card name → sorted list of indices into combo_list.

    Without this a builder linear-scans 83K combos per candidate card; with it
    the "what does this deck contain" question is a dict lookup per card.
    """
    index = defaultdict(set)
    for i, combo in enumerate(combo_list):
        for name in combo["cards"]:
            index[name].add(i)
    return {k: sorted(v) for k, v in index.items()}


def bracket_summary(combo_list):
    """Count combos per bracket for the details meta block."""
    counts = defaultdict(int)
    for combo in combo_list:
        key = "banned" if combo.get("banned") else str(combo["bracket"])
        counts[key] += 1
    return dict(sorted(counts.items()))


def _popularity_rank(record):
    """Most popular first, Spellbook id as the tiebreak: two runs, one order."""
    return (-(record.get("popularity") or 0), str(record.get("id") or ""))


def build_combo_index(combos, per_card=COMBO_INDEX_PER_CARD, source_timestamp=None):
    """The browser-sized index over processed combo records.

    `combos` is `build_combo_graph`'s detail list. Per card: `n` and `inf` are
    the TRUE totals over every combo it is in; `top` is at most `per_card`
    positions into the returned `combos` rows, ordered by (popularity desc, id
    asc). The rows hold only combos some `top` names, sorted by id, as
    `[id, [names], infinite 0/1, bracket | null, mana_value_needed]`. A banned
    combo keeps its null bracket; the flag lives in `combo_details.json`.
    """
    memberships = defaultdict(list)  # name -> positions in `combos`, each once
    for pos, record in enumerate(combos):
        for name in dict.fromkeys(record["cards"]):
            memberships[name].append(pos)

    tops = {}
    selected = set()
    for name, positions in memberships.items():
        top = sorted(positions, key=lambda pos: _popularity_rank(combos[pos]))[:per_card]
        tops[name] = top
        selected.update(top)

    ordered = sorted(selected, key=lambda pos: (str(combos[pos].get("id") or ""), pos))
    row_of = {pos: row for row, pos in enumerate(ordered)}
    rows = [
        [
            combos[pos].get("id"),
            list(combos[pos]["cards"]),
            1 if is_infinite(combos[pos]) else 0,
            combos[pos].get("bracket"),
            combos[pos].get("mana_value_needed"),
        ]
        for pos in ordered
    ]
    by_card = {
        name: {
            "n": len(positions),
            "inf": sum(1 for pos in positions if is_infinite(combos[pos])),
            "top": [row_of[pos] for pos in tops[name]],
        }
        for name, positions in sorted(memberships.items())
    }
    return {
        "meta": {
            "source_timestamp": source_timestamp,
            "per_card": per_card,
            "combos": len(rows),
            "indexed": len(by_card),
        },
        "combos": rows,
        "by_card": by_card,
    }


def _write_json(path, doc):
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Wrote {path} ({size_mb:.1f} MB)")


def main():
    print("Loading raw combos...")
    with open_dump(COMBOS_RAW_PATH, "rt") as f:
        combos = raw_variants(json.load(f))
    source = load_source_meta()
    print(f"  {len(combos):,} raw combo variants "
          f"(source timestamp {source['timestamp']}, version {source['version']})")
    statuses = status_counts(combos)
    not_ok = {k: v for k, v in statuses.items() if k != STATUS_OK}
    if not_ok:
        print(f"  WARNING: {sum(not_ok.values()):,} variants are not status {STATUS_OK!r}: "
              f"{dict(sorted(not_ok.items(), key=str))} — kept, not filtered (see module doc)")

    print("Loading known cards from cards.csv...")
    known_cards = load_known_cards(OUTPUT_CSV_PATH)
    print(f"  {len(known_cards):,} known cards")

    print("Building combo graph...")
    partners, combo_list = build_combo_graph(combos, known_cards)
    by_card = build_card_index(combo_list)
    summary = bracket_summary(combo_list)

    print(f"  {len(partners):,} cards with combo partners")
    print(f"  {len(combo_list):,} valid combos (all cards in dataset)")
    print(f"  bracket distribution: {summary}")

    _write_json(COMBO_GRAPH_PATH, {
        "partners": partners,
        "meta": {"source_timestamp": source["timestamp"]},
    })

    _write_json(COMBO_DETAILS_PATH, {
        "combos": combo_list,
        "by_card": by_card,
        "meta": {"combo_count": len(combo_list), "brackets": summary, "source": source},
    })

    print("Building combo index...")
    index = build_combo_index(combo_list, source_timestamp=source["timestamp"])
    print(f"  {index['meta']['indexed']:,} cards indexed over {index['meta']['combos']:,} combos "
          f"(top {index['meta']['per_card']} per card)")
    _write_json(COMBO_INDEX_PATH, index)


if __name__ == "__main__":
    main()
