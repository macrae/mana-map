"""Step 8: Process raw Commander Spellbook data into combo_graph.json."""

import json
from collections import defaultdict

import pandas as pd

from config import COMBO_GRAPH_PATH, COMBOS_RAW_PATH, OUTPUT_CSV_PATH


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

        combo_list.append({
            "cards": card_names,
            "produces": produces,
            "ci": ci,
        })

    # Convert sets to sorted lists for JSON serialization
    partners_dict = {k: sorted(v) for k, v in partners.items()}

    return partners_dict, combo_list


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

    graph = {
        "partners": partners,
        "combos": combo_list,
    }

    print(f"  {len(partners):,} cards with combo partners")
    print(f"  {len(combo_list):,} valid combos (all cards in dataset)")

    with open(COMBO_GRAPH_PATH, "w") as f:
        json.dump(graph, f, separators=(",", ":"))

    size_mb = COMBO_GRAPH_PATH.stat().st_size / (1024 * 1024)
    print(f"  Wrote {COMBO_GRAPH_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
