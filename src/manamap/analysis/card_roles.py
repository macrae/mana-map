"""Step 13: Classify every card by the job it does in a deck.

`mechanical_tags.py` answers "what is this card like" — a retrieval vocabulary
that clusters cards by flavour of effect. Deckbuilding asks a different
question: "what job does this card do in a 99", and the answers have to be
countable. You cannot say "this deck needs three more pieces of interaction"
from a tag set where `ramp` covers Sol Ring, Llanowar Elves, Rampant Growth and
Dark Ritual alike.

Two rules keep this honest:

- **The type line disambiguates mana.** An artifact that taps for mana is a
  rock, a creature is a dork, an instant is a ritual. They cost differently,
  they die differently, and a curve model that conflates them is wrong.
- **Unclassified is a reported number, not a silent bucket.** The output meta
  block carries coverage, and `ROLE_COVERAGE_TARGET` is asserted in the test
  suite — a classifier that quietly covers 70% would make the slot filler
  guess while looking confident.

Fully deterministic: regex over oracle text plus type-line rules, no model, no
randomness. Same cards.csv always produces the same card_roles.json.
"""

import json
import re

import pandas as pd

from manamap.config import (
    CARD_ROLES_PATH,
    OUTPUT_CSV_PATH,
    ROLE_BODY_FALLBACK,
    ROLE_COST_REDUCTION,
    ROLE_LAND_PATTERNS,
    ROLE_LAND_RAMP,
    ROLE_MANA_BY_SUPERTYPE,
    ROLE_MANA_SOURCE,
    ROLE_NAMES,
    ROLE_PATTERNS,
)

_COMPILED_ROLES = {
    role: re.compile(pattern, re.IGNORECASE) for role, pattern in ROLE_PATTERNS.items()
}
_COMPILED_LAND_ROLES = {
    role: re.compile(pattern, re.IGNORECASE) for role, pattern in ROLE_LAND_PATTERNS.items()
}
_MANA_SOURCE_RE = re.compile(ROLE_MANA_SOURCE, re.IGNORECASE)
_LAND_RAMP_RE = re.compile(ROLE_LAND_RAMP, re.IGNORECASE)
_COST_REDUCTION_RE = re.compile(ROLE_COST_REDUCTION, re.IGNORECASE)
_BASIC_LAND_RE = re.compile(r"\bBasic\b.*\bLand\b", re.IGNORECASE)
_PRODUCES_TWO_COLORS_RE = re.compile(r"add (?:\{[WUBRG]\}|one mana).{0,40}?or", re.IGNORECASE)


def classify_land(type_line, text):
    """Land quality roles. A land never carries a spell role."""
    roles = set()
    if _BASIC_LAND_RE.search(type_line):
        roles.add("land:basic")
    for role, pattern in _COMPILED_LAND_ROLES.items():
        if pattern.search(text):
            roles.add(role)
    if "Land //" in type_line or "// Land" in type_line:
        roles.add("land:mdfc")
    # An untapped dual is the thing every mana base actually wants: more than
    # one color, no tapped clause.
    if _PRODUCES_TWO_COLORS_RE.search(text) and "land:tapped" not in roles:
        roles.add("land:untapped-dual")
    return roles


def classify_spell(supertype, text):
    """Non-land roles: pattern matches plus type-disambiguated mana production."""
    roles = {role for role, pattern in _COMPILED_ROLES.items() if pattern.search(text)}

    if _LAND_RAMP_RE.search(text):
        roles.add("ramp:land")
    if _MANA_SOURCE_RE.search(text):
        mana_role = ROLE_MANA_BY_SUPERTYPE.get(supertype)
        if mana_role:
            roles.add(mana_role)
    if _COST_REDUCTION_RE.search(text):
        roles.add("ramp:cost-reduction")
    if supertype == "Creature":
        roles.add(ROLE_BODY_FALLBACK)

    # A sweeper is not also spot removal — the narrower claim would let a
    # builder count Wrath of God toward its single-target interaction budget.
    if "removal:sweeper" in roles:
        roles.discard("removal:spot")
    # "search your library for a card" also matches the narrow pattern.
    if "tutor:unrestricted" in roles:
        roles.discard("tutor:narrow")
    return roles


def classify_row(row):
    """Roles for one card row. Returns a sorted list."""
    type_line = str(row.get("type_line", "") or "")
    supertype = str(row.get("supertype", "") or "")
    text = str(row.get("oracle_text", "") or "")
    keywords = str(row.get("keywords", "") or "")
    if keywords:
        text += " " + keywords

    if supertype == "Land" or "Land" in type_line.split(" // ")[0]:
        return sorted(classify_land(type_line, text))
    return sorted(classify_spell(supertype, text))


def build_roles(df):
    """Classify every row. Returns (roles_by_name, meta)."""
    roles_by_name = {}
    counts = {role: 0 for role in ROLE_NAMES}
    unclassified = []

    for row in df.to_dict("records"):
        roles = classify_row(row)
        name = row["name"]
        if roles:
            roles_by_name[name] = roles
            for role in roles:
                counts[role] = counts.get(role, 0) + 1
        elif row.get("legal_commander") == "legal":
            unclassified.append(name)

    # Coverage is measured over the population the builder actually draws from:
    # Commander-legal cards. Counting the whole corpus would flatter us with
    # cards no deck can play. `specific` drops the body fallback, because
    # labelling every creature "threat:body" is not classification.
    eligible = df[df["legal_commander"] == "legal"]
    covered = sum(1 for n in eligible["name"] if n in roles_by_name)
    specific = sum(
        1 for n in eligible["name"]
        if [r for r in roles_by_name.get(n, []) if r != ROLE_BODY_FALLBACK]
    )
    total = len(eligible)
    meta = {
        "card_count": len(df),
        "commander_legal_count": int(total),
        "classified_count": covered,
        "specific_count": specific,
        "coverage": round(covered / total, 4) if total else 0.0,
        "specific_coverage": round(specific / total, 4) if total else 0.0,
        "unclassified_sample": sorted(unclassified)[:25],
        "role_counts": dict(sorted(counts.items())),
    }
    return roles_by_name, meta


def main():
    print("Loading cards.csv...")
    df = pd.read_csv(OUTPUT_CSV_PATH)
    print(f"  {len(df):,} cards")

    print("Classifying deckbuilding roles...")
    roles_by_name, meta = build_roles(df)
    print(f"  {len(roles_by_name):,} cards with at least one role")
    print(f"  coverage (Commander-legal): {meta['coverage']:.1%}"
          f"  |  excluding the body fallback: {meta['specific_coverage']:.1%}")

    top = sorted(meta["role_counts"].items(), key=lambda kv: -kv[1])[:8]
    print(f"  most common: {', '.join(f'{r} {c:,}' for r, c in top)}")

    doc = {"roles": roles_by_name, "meta": meta}
    with open(CARD_ROLES_PATH, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")

    size_mb = CARD_ROLES_PATH.stat().st_size / (1024 * 1024)
    print(f"  Wrote {CARD_ROLES_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
