"""Step 11: Detect power creep / obsolescence — find strictly-better printings."""

import json

import numpy as np
import pandas as pd

from config import (
    ABILITY_EMBEDDINGS_PATH,
    EMBEDDINGS_PATH,
    MECHANICAL_TAG_NAMES,
    OBSOLESCENCE_INDEX_PATH,
    OUTPUT_CSV_PATH,
)


def parse_stat(val):
    """Parse a power/toughness value. Returns float or None for '*' and similar."""
    if val is None or pd.isna(val):
        return None
    val = str(val).strip()
    if val == "" or val == "*":
        return None
    if val.startswith("+") or val.startswith("-"):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def parse_color_requirement(mana_cost):
    """Parse mana cost into color requirements dict.

    Returns dict of color -> pip count, e.g. '{2}{W}{W}' -> {'W': 2}.
    """
    if not mana_cost or pd.isna(mana_cost):
        return {}
    import re
    tokens = re.findall(r'\{([^}]+)\}', str(mana_cost))
    pips = {}
    for t in tokens:
        if t in "WUBRG":
            pips[t] = pips.get(t, 0) + 1
        elif "/" in t:
            # Hybrid mana — count as 0.5 of each color
            for part in t.split("/"):
                if part in "WUBRG":
                    pips[part] = pips.get(part, 0) + 0.5
    return pips


def color_requirement_subset(cost_a, cost_b):
    """Check if B's color requirements are the same or easier than A's.

    Returns True if B needs the same or fewer pips of each color.
    """
    pips_a = parse_color_requirement(cost_a)
    pips_b = parse_color_requirement(cost_b)
    # B must not require any color that A doesn't
    for color in pips_b:
        if pips_b[color] > pips_a.get(color, 0):
            return False
    return True


def parse_tags_set(tags_str):
    """Parse mechanical_tags string into a set."""
    if not tags_str or pd.isna(tags_str):
        return set()
    return {t.strip() for t in str(tags_str).split(",") if t.strip()}


def find_strictly_better(df):
    """Find strictly-better replacements for each card.

    Card B is strictly better than Card A if:
    1. Same supertype
    2. B.cmc <= A.cmc
    3. Same or easier color requirement
    4. B has all of A's mechanical tags (superset)
    5. Same or better power/toughness (for creatures)
    6. B was released after A (newer)

    Returns dict mapping card_name -> list of {name, advantages, released_at}.
    """
    # Pre-process data
    records = []
    for i, row in df.iterrows():
        tags = parse_tags_set(row.get("mechanical_tags", ""))
        records.append({
            "idx": i,
            "name": row["name"],
            "supertype": row["supertype"],
            "cmc": float(row["cmc"]) if pd.notna(row["cmc"]) else 0.0,
            "mana_cost": row.get("mana_cost", ""),
            "power": parse_stat(row.get("power")),
            "toughness": parse_stat(row.get("toughness")),
            "tags": tags,
            "released_at": str(row.get("released_at", "")),
            "edhrec_rank": row.get("edhrec_rank"),
        })

    # Group by supertype for efficiency
    by_supertype = {}
    for rec in records:
        st = rec["supertype"]
        if st not in by_supertype:
            by_supertype[st] = []
        by_supertype[st].append(rec)

    obsolescence = {}

    for st, group in by_supertype.items():
        if st in ("Land", "Unknown"):
            continue

        for a in group:
            if not a["tags"]:
                continue  # Skip vanilla cards (no tags to compare)
            if not a["mana_cost"] or pd.isna(a["mana_cost"]):
                continue  # Skip cards with no mana cost (augments, tokens)

            strictly_better = []

            for b in group:
                if b["name"] == a["name"]:
                    continue
                if not b["mana_cost"] or pd.isna(b["mana_cost"]):
                    continue  # Skip cards with no mana cost

                # 1. Same supertype (already grouped)
                # 2. B.cmc <= A.cmc
                if b["cmc"] > a["cmc"]:
                    continue

                # 3. Same or easier color requirement
                if not color_requirement_subset(a["mana_cost"], b["mana_cost"]):
                    continue

                # 4. B has all of A's tags (superset)
                if not a["tags"].issubset(b["tags"]):
                    continue

                # 5. Same or better stats (creatures)
                if a["power"] is not None and b["power"] is not None:
                    if b["power"] < a["power"]:
                        continue
                    if b["toughness"] is not None and a["toughness"] is not None:
                        if b["toughness"] < a["toughness"]:
                            continue

                # 6. B must be at least as good overall — needs at least one advantage
                advantages = []
                if b["cmc"] < a["cmc"]:
                    advantages.append("Lower CMC")
                if b["power"] is not None and a["power"] is not None and b["power"] > a["power"]:
                    advantages.append("Better Power")
                if b["toughness"] is not None and a["toughness"] is not None and b["toughness"] > a["toughness"]:
                    advantages.append("Better Toughness")
                extra_tags = b["tags"] - a["tags"]
                if extra_tags:
                    tag_labels = sorted(extra_tags)[:3]
                    advantages.append("Additional: " + ", ".join(tag_labels))

                if not advantages:
                    continue  # B is identical, not strictly better

                # 7. B was released after A (newer)
                if b["released_at"] and a["released_at"] and b["released_at"] <= a["released_at"]:
                    continue

                strictly_better.append({
                    "name": b["name"],
                    "advantages": advantages,
                    "released_at": b["released_at"],
                })

            if strictly_better:
                # Sort by number of advantages (most advantages first), limit to 5
                strictly_better.sort(key=lambda x: -len(x["advantages"]))
                obsolescence[a["name"]] = {
                    "obsoleted_by": strictly_better[:5],
                }

    return obsolescence


def compute_soft_obsolescence(df, embeddings=None):
    """Compute soft obsolescence score for cards that have strictly-better replacements.

    Uses ability-embedding similarity as a proxy.
    This is a lightweight scoring — returns dict of card_name -> float [0,1].
    """
    # Placeholder: soft obsolescence can be computed later with ability embeddings
    return {}


def main():
    print("Loading cards...")
    df = pd.read_csv(OUTPUT_CSV_PATH)
    print(f"  {len(df):,} cards")

    print("Finding strictly-better replacements...")
    obsolescence = find_strictly_better(df)

    print(f"  {len(obsolescence):,} cards have strictly-better replacements")

    # Sample some results
    if obsolescence:
        samples = list(obsolescence.items())[:5]
        for name, data in samples:
            replacements = [r["name"] for r in data["obsoleted_by"][:3]]
            print(f"    {name} -> {', '.join(replacements)}")

    with open(OBSOLESCENCE_INDEX_PATH, "w") as f:
        json.dump(obsolescence, f, separators=(",", ":"))

    size_mb = OBSOLESCENCE_INDEX_PATH.stat().st_size / (1024 * 1024)
    print(f"  Wrote {OBSOLESCENCE_INDEX_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
