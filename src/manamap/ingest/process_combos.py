"""Step 8: Process raw Commander Spellbook data into the combo artifacts.

Two files, because they have two different audiences:

- `combo_graph.json` — the partner adjacency map. Small, and the only thing the
  viz deck builder reads (`graph.partners[name]`). It is fetched and parsed on
  the browser's main thread, so nothing else belongs in it.
- `combo_details.json` — the full combo records plus a card→combo index. Read
  by Python and by agents, never by the browser. This is where the power-level
  signal lives: Spellbook tags every variant with a bracket letter, which is
  what lets `pilot/bracket.py` compute a deck's bracket floor.

The graph stays format-agnostic by design — Commander-banned combos are kept
and flagged (`banned: true`), not dropped. Filtering happens at consumption.
"""

import json
from collections import defaultdict

import pandas as pd

from manamap.config import (
    COMBO_BANNED_TAG,
    COMBO_BRACKET_TAGS,
    COMBO_DETAILS_PATH,
    COMBO_GRAPH_PATH,
    COMBOS_RAW_PATH,
    OUTPUT_CSV_PATH,
)


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


def main():
    print("Loading raw combos...")
    with open(COMBOS_RAW_PATH, "r") as f:
        combos = json.load(f)
    print(f"  {len(combos):,} raw combo variants")

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

    with open(COMBO_GRAPH_PATH, "w") as f:
        json.dump({"partners": partners}, f, separators=(",", ":"))
    size_mb = COMBO_GRAPH_PATH.stat().st_size / (1024 * 1024)
    print(f"  Wrote {COMBO_GRAPH_PATH} ({size_mb:.1f} MB)")

    details = {
        "combos": combo_list,
        "by_card": by_card,
        "meta": {"combo_count": len(combo_list), "brackets": summary},
    }
    with open(COMBO_DETAILS_PATH, "w") as f:
        json.dump(details, f, separators=(",", ":"))
    size_mb = COMBO_DETAILS_PATH.stat().st_size / (1024 * 1024)
    print(f"  Wrote {COMBO_DETAILS_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
