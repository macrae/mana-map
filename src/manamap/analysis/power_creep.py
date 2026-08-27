"""Step 11: Detect power creep / obsolescence — find strictly-better printings."""

import json
import re

import numpy as np
import pandas as pd

from manamap.analysis.common import (
    creature_types, load_first_embeddings, parse_tag_set)
from manamap.config import (
    ABILITY_EMBEDDINGS_PATH,
    OBSOLESCENCE_INDEX_PATH,
    OBSOLESCENCE_MAX_REPLACEMENTS,
    OBSOLESCENCE_MIN_TAGS,
    OBSOLESCENCE_SIMILARITY_THRESHOLD,
    OBSOLESCENCE_SINGLE_TAG_THRESHOLD,
    OUTPUT_CSV_PATH,
    DEFAULT_TAG_VALENCE,
    RESTRICTION_PATTERNS,
    TAG_VALENCE,
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


# Public name kept for backwards compatibility; implementation is shared.
parse_tags_set = parse_tag_set


# ── What the comparison could not read, and now must ─────────────────────

_RESTRICTION_RE = {k: re.compile(v, re.IGNORECASE)
                   for k, v in RESTRICTION_PATTERNS.items()}
_ACTIVATION_RE = re.compile(r"(^|\n)([^:\n]{0,60}?):", re.MULTILINE)
_GENERIC_RE = re.compile(r"\{(\d+)\}")
_COLOURED_RE = re.compile(r"\{[WUBRGC]\}")
_PHYREXIAN_RE = re.compile(r"\{[WUBRG]/P\}")


def _rank(value):
    import math
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(value)


#: A TRIBAL GATE, which is the failure the caveat is actually about and which the
#: pattern dict cannot express — it needs the corpus's type list, not a regex.
#:
#: `MECHANICAL_TAGS["death_trigger"]` is `when .* dies`. Boggart Mischief reads
#: "Whenever a Goblin you control dies"; Bastion of Remembrance reads "Whenever
#: another creature you control dies". The `.*` swallows the difference, the tag
#: sets come out identical, and the index offered the Goblin card as a
#: replacement in a box with almost no Goblins.
#:
#: The subject sits between the trigger word and the verb, so capture THAT
#: window and look for a real type in it. Validated against the corpus type list
#: rather than a capital-letter rule: "Treasures you control" is not a tribe, and
#: the first cut of the sibling check in `assess` reported "needs Treasuress".
_TRIGGER_SUBJECT_RE = re.compile(
    r"\b(?:whenever|when|if)\s+((?:another\s+|each\s+|a\s+|an\s+)?[^,.]{0,40}?)"
    r"\s+(?:you control\s+)?(?:dies|enters|attacks|deals)",
    re.IGNORECASE)


def tribal_gates(text, types):
    """Creature types this card's triggers are gated on."""
    body = text if isinstance(text, str) else ""
    found = set()
    for match in _TRIGGER_SUBJECT_RE.finditer(body):
        for word in match.group(1).split():
            token = word.strip("s,.")
            if token in types:
                found.add(token)
            elif word.strip(",.") in types:
                found.add(word.strip(",."))
    return found


def restrictions(text):
    """Which restriction classes this card's text carries."""
    body = text if isinstance(text, str) else ""
    return {name for name, pat in _RESTRICTION_RE.items() if pat.search(body)}


def activation_mana(text):
    """Cheapest mana in an activation cost, or None if there is no ability.

    Bartolome del Presidio's sacrifice ability is FREE; the index's proposed
    upgrade charges {2} for the same effect and the difference was reported as
    `Better Toughness`. Of pairs where both cards have an activated ability,
    29.9% made it MORE expensive.
    """
    best = None
    for match in _ACTIVATION_RE.finditer(text if isinstance(text, str) else ""):
        cost = match.group(2)
        if "{" not in cost and "Sacrifice" not in cost and "Discard" not in cost:
            continue
        total = len(_COLOURED_RE.findall(cost)) + sum(
            int(g) for g in _GENERIC_RE.findall(cost))
        best = total if best is None else min(best, total)
    return best


def effective_cost(cmc, mana_cost):
    """Mana value, minus the Phyrexian pips you can pay with life.

    `preprocess.parse_mana_pips` counts `{U/P}` as a full blue pip and says so
    deliberately — that is the honest reading for CASTABILITY, where the pip is
    payable with mana. It is the wrong reading for a COST COMPARISON, where the
    point is that Dismember is a one-mana spell. The same one-concept-two-
    questions split as `cast_pips` vs `manabase.count_pips`.

    Understated, never overstated: a Phyrexian pip is counted as free, which
    makes the ORIGINAL look cheaper and so makes a replacement harder to claim.
    `manabase.land_colors`' asymmetry — overstating is what buys a downgrade.
    """
    # NaN is a float and NaN is TRUTHY, so `mana_cost or ""` does not catch a
    # missing cell — the trap the pandas-facing code in this module already
    # guards with `pd.isna` at every other read.
    cost = mana_cost if isinstance(mana_cost, str) else ""
    return max(0.0, float(cmc) - len(_PHYREXIAN_RE.findall(cost)))


def valence(tags):
    """Split a tag difference into what it GIVES and what it CHARGES."""
    gains, costs, context = [], [], []
    for tag in sorted(tags):
        where = {"gain": gains, "cost": costs}.get(
            TAG_VALENCE.get(tag, DEFAULT_TAG_VALENCE), context)
        where.append(tag)
    return gains, costs, context


#: HOW WEIGHTS WERE SET. Each factor is a failure class the audit MEASURED, and
#: its weight is calibrated against the four cases `pool_facts` documents plus the
#: ones sampling found — Boggart Mischief, Prognostic Sphinx, Bartolome del
#: Presidio, Leaden Myr must all score low; a genuine cheaper-strict-superset must
#: score high. `tests/test_power_creep.py` asserts those anchors, so a weight
#: cannot be tuned until the fixtures stop meaning anything.
#:
#: MULTIPLICATIVE, not additive: two problems compound rather than averaging out.
#: A card that both narrows the trigger AND charges more for the ability is not
#: half as bad as one that does either.
OBSOLESCENCE_PENALTIES = {
    # A trigger narrowed to a creature type is the sharpest narrowing there is —
    # Boggart Mischief drains only on a Goblin, in a box with almost no Goblins.
    "tribal_gate": 0.30,
    # A gate the original does not have: timing, conditional, an extra cost.
    "restriction": 0.55,
    # The same ability, but you pay for it. Bartolome's is free.
    "ability_costs_more": 0.45,
    # A tag that is a PRICE, reported by the old schema as an advantage.
    "cost_tag": 0.75,
    # Weak evidence and weighted as such: rank is ordinal, non-linear, and the
    # gap between #18,000 and #18,200 means nothing.
    "played_less": 0.90,
}


def obsolescence_strength(a, b, gains, costs, act_a, act_b):
    """How strongly does B outclass A? 0.0 = not at all, 1.0 = strictly better.

    A SCORE RATHER THAN A GATE, because the honest answer is a degree. The first
    cut of this repair used hard gates and they were right about the big classes
    and wrong in shape: a gate throws away every near-miss, and the coarse
    restriction classes CANCEL — Frightshroud Courier and Arnim Zola both carry
    `conditional`, so the set difference is empty and a real narrowing passes
    silently. A score degrades where a gate flips.

    It also puts the judgement where it belongs. This module can measure a
    difference; whether the difference is worth taking depends on a deck it
    cannot see. So: publish the measure, let the pilot set the line.
    """
    # The ceiling is how much cheaper and bigger B is — a strict superset that
    # costs the same scores lower than one that costs two less, because "you
    # could swap this" is a weaker claim than "this is free value".
    cheaper = a["effective_cost"] - b["effective_cost"]
    base = 0.45
    base += min(0.30, 0.12 * max(0.0, cheaper))
    base += min(0.15, 0.05 * len([g for g in gains if "more" in g]))
    base += min(0.10, 0.04 * len([g for g in gains if g not in ("more power",
                                                               "more toughness")]))
    score = min(1.0, base)

    for _ in (b["tribes"] - a["tribes"]):
        score *= OBSOLESCENCE_PENALTIES["tribal_gate"]
    for _ in (b["restrictions"] - a["restrictions"]):
        score *= OBSOLESCENCE_PENALTIES["restriction"]
    if act_a is not None and act_b is not None and act_b > act_a:
        score *= OBSOLESCENCE_PENALTIES["ability_costs_more"]
    for _ in costs:
        score *= OBSOLESCENCE_PENALTIES["cost_tag"]
    ra, rb = _rank(a["edhrec_rank"]), _rank(b["edhrec_rank"])
    if ra is not None and rb is not None and rb > ra:
        score *= OBSOLESCENCE_PENALTIES["played_less"]
    return round(score, 3)


def find_strictly_better(df, ability_embeddings=None, legal=None, similarity_threshold=None,
                         min_tags=None, single_tag_threshold=None):
    """Find strictly-better replacements for each card.

    Card B is strictly better than Card A if:
    1. Same supertype
    2. B.cmc <= A.cmc
    3. Same or easier color requirement
    4. Cosine similarity >= threshold in ability embedding space (if embeddings provided)
       Uses tiered thresholds: single_tag_threshold for 1-tag cards, similarity_threshold for 2+
    5. B has all of A's mechanical tags (superset)
    6. Same or better power/toughness (for creatures)
    7. At least one concrete advantage
    8. B was released after A (newer)

    Args:
        df: DataFrame with card data
        ability_embeddings: Optional (N, 128) array of ability embeddings for similarity gate
        similarity_threshold: Min cosine similarity for 2+-tag cards (default from config)
        min_tags: Min mechanical tags required for a card to be compared (default from config)
        single_tag_threshold: Min cosine similarity for 1-tag cards (default from config)

    Returns dict mapping card_name -> {obsoleted_by: [...], ...}.
    """
    if similarity_threshold is None:
        similarity_threshold = OBSOLESCENCE_SIMILARITY_THRESHOLD
    if single_tag_threshold is None:
        single_tag_threshold = OBSOLESCENCE_SINGLE_TAG_THRESHOLD
    if min_tags is None:
        min_tags = OBSOLESCENCE_MIN_TAGS

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
            # NEVER LOADED BEFORE. The module read tags, cost, stats and a date
            # and never opened the oracle text — which is the whole
            # structural-versus-functional gap in one line.
            "text": str(row.get("oracle_text") or ""),
            "legal": legal.get(row["name"]) if legal else None,
        })
    # PRECOMPUTED PER CARD, NOT PER PAIR. Both depend only on the card, and the
    # comparison is O(n^2) within each supertype group — running them inline
    # turned a 30-second step into one that had not finished in ten minutes.
    # The vectorised similarity gate above exists for exactly this reason.
    types = creature_types(df)
    for rec in records:
        rec["restrictions"] = restrictions(rec["text"])
        rec["tribes"] = tribal_gates(rec["text"], types)
        rec["activation"] = activation_mana(rec["text"])
        rec["effective_cost"] = effective_cost(rec["cmc"], rec["mana_cost"])

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

        # Pre-compute similarity matrix for this supertype group if embeddings available
        sim_matrix = None
        if ability_embeddings is not None:
            group_indices = [rec["idx"] for rec in group]
            group_embs = ability_embeddings[group_indices]
            # L2 normalize (should already be normalized, but be safe)
            norms = np.linalg.norm(group_embs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            group_embs = group_embs / norms
            # Batch cosine similarity via matrix multiply
            sim_matrix = group_embs @ group_embs.T

        for i_local, a in enumerate(group):
            if len(a["tags"]) < min_tags:
                continue  # Skip cards with too few tags
            if not a["mana_cost"] or pd.isna(a["mana_cost"]):
                continue  # Skip cards with no mana cost (augments, tokens)

            strictly_better = []

            for j_local, b in enumerate(group):
                if b["name"] == a["name"]:
                    continue
                if not b["mana_cost"] or pd.isna(b["mana_cost"]):
                    continue  # Skip cards with no mana cost

                # Similarity gate: tiered threshold based on tag count
                sim_score = None
                if sim_matrix is not None:
                    sim_score = float(sim_matrix[i_local, j_local])
                    effective_threshold = single_tag_threshold if len(a["tags"]) == 1 else similarity_threshold
                    if sim_score < effective_threshold:
                        continue

                # 1. Same supertype (already grouped)

                # LEGALITY, which was never checked. The index is
                # format-agnostic and offered acorn and Vintage-only cards as
                # replacements for Commander staples — 8.2% of published pairs.
                # `banned` is refused with `not_legal`, since neither is a card
                # you may sleeve.
                if b["legal"] is not None and b["legal"] != "legal":
                    continue

                # 2. B costs no more — with Phyrexian read as the discount it is.
                if b["effective_cost"] > a["effective_cost"]:
                    continue

                # RESTRICTIONS, TRIBAL GATES AND ABILITY COST ARE SCORED, NOT
                # GATED. They are the classes the audit measured — 22.9% of the
                # old index added a restriction, 29.9% of ability-pairs made the
                # ability more expensive — but a hard gate is the wrong SHAPE
                # for them: it throws away every near-miss, and the coarse
                # classes cancel (two cards both carrying `conditional` differ
                # by nothing, whatever the two conditions say). They feed
                # `obsolescence_strength` below instead, and the pilot picks the
                # line. What stays a gate is what makes a pair incomparable at
                # all: supertype, similarity, legality, and cost.
                act_a, act_b = a["activation"], b["activation"]

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

                # 6. THE SYMMETRIC DIFF, replacing a one-sided advantage list.
                #
                # `advantages` reported every tag B had and A lacked as a gain,
                # whatever it meant — so `Additional: discard` was published as
                # a reason to make the swap, and 15.5% of pairs counted a price
                # as a gain. A tag has no sign; `TAG_VALENCE` gives it one, and
                # `context` is the honest default: whether `tokens` is a payoff
                # or noise depends on the deck, so it is reported as a
                # DIFFERENCE and never as an advantage.
                gains, costs = [], []
                cheaper = a["effective_cost"] - b["effective_cost"]
                if cheaper > 0:
                    gains.append(f"costs {cheaper:g} less")
                if b["power"] is not None and a["power"] is not None and b["power"] > a["power"]:
                    gains.append("more power")
                if b["toughness"] is not None and a["toughness"] is not None and b["toughness"] > a["toughness"]:
                    gains.append("more toughness")
                if act_a is not None and act_b is not None and act_b < act_a:
                    gains.append(f"its ability costs {act_a - act_b:g} less")
                tag_gain, tag_cost, tag_context = valence(b["tags"] - a["tags"])
                gains += tag_gain
                costs += tag_cost

                if not gains:
                    continue  # nothing to compare on

                # 7. B was released after A (newer)
                if b["released_at"] and a["released_at"] and b["released_at"] <= a["released_at"]:
                    continue

                # RANK IS REPORTED, NEVER A FILTER. Storm Crow is genuinely
                # outclassed by a card nobody plays; that is structurally true
                # and useless, and the pilot is the one who should see the
                # number and decide. 30.8% of published replacements were played
                # LESS than the card they claimed to outclass.
                rank_a, rank_b = _rank(a["edhrec_rank"]), _rank(b["edhrec_rank"])
                strength = obsolescence_strength(a, b, gains, costs, act_a, act_b)
                entry = {
                    # 0.0 = these are two different cards that happen to sort
                    # near each other; 1.0 = strictly better, cheaper, no
                    # strings. Everything between is a judgement this module
                    # cannot make and does not pretend to.
                    "strength": strength,
                    "name": b["name"],
                    "gains": gains,
                    "costs": costs,
                    "also_differs": tag_context,
                    "narrows": sorted((b["tribes"] - a["tribes"])
                                      | (b["restrictions"] - a["restrictions"])),
                    "released_at": b["released_at"],
                    "edhrec_rank": rank_b,
                    "played_more": (None if rank_a is None or rank_b is None
                                    else rank_b < rank_a),
                }
                if sim_score is not None:
                    entry["similarity"] = round(sim_score, 4)
                strictly_better.append(entry)

            if strictly_better:
                # Sort by similarity (desc), then by number of advantages (desc)
                strictly_better.sort(
                    # BY STRENGTH, not by similarity. Similarity says "these
                    # two cards are alike"; strength says "and this one is
                    # better", which is the question the panel is answering.
                    key=lambda x: (-x["strength"], -x.get("similarity", 0))
                )
                obsolescence[a["name"]] = {
                    # NOT `obsoleted_by`. "Obsoleted" is a verdict about all
                    # contexts; this data supports a COMPARISON and the pilot
                    # supplies the context. `close.py`'s doctrine one module
                    # over: it retrieves, it does not score.
                    "compare_with": strictly_better[:OBSOLESCENCE_MAX_REPLACEMENTS],
                }

    return obsolescence


def main():
    print("Loading cards...")
    df = pd.read_csv(OUTPUT_CSV_PATH)
    print(f"  {len(df):,} cards")

    # Load ability embeddings for similarity gate
    ability_embeddings, _ = load_first_embeddings(ABILITY_EMBEDDINGS_PATH)
    if ability_embeddings is not None:
        print(f"  Loaded ability embeddings: {ability_embeddings.shape}")
    else:
        print("  Warning: ability embeddings not found, running without similarity gate")

    # LEGALITY, from the corpus column the index never read. `analysis/` must
    # not import `pilot/` (pilot imports analysis), so the column is read here
    # rather than through `card_pool.legality` — with the same combining rule
    # that module measured: any legal printing wins, `banned` checked first,
    # because first-printing-wins reported Savage Lands as illegal off an `fmsc`
    # promo and failed two decks on their own tracked lists.
    legal = None
    if "legal_commander" in df.columns:
        legal = {}
        for name, status in zip(df["name"], df["legal_commander"].fillna("not_legal")):
            prev = legal.get(name)
            if prev == "banned":
                continue
            if status == "banned" or prev is None or status == "legal":
                legal[name] = status
        print(f"  Legality: {sum(1 for v in legal.values() if v == 'legal'):,} "
              f"commander-legal of {len(legal):,}")

    print("Comparing cards that do the same job...")
    obsolescence = find_strictly_better(df, ability_embeddings=ability_embeddings,
                                        legal=legal)

    pairs = sum(len(v["compare_with"]) for v in obsolescence.values())
    print(f"  {len(obsolescence):,} cards have a comparable alternative "
          f"({pairs:,} pair(s))")

    if obsolescence:
        for name, data in list(obsolescence.items())[:5]:
            first = data["compare_with"][0]
            gains = ", ".join(first["gains"][:2])
            costs = (" · costs: " + ", ".join(first["costs"])) if first["costs"] else ""
            print(f"    {name} -> {first['name']}  ({gains}{costs})")

    with open(OBSOLESCENCE_INDEX_PATH, "w") as f:
        json.dump(obsolescence, f, separators=(",", ":"))

    size_mb = OBSOLESCENCE_INDEX_PATH.stat().st_size / (1024 * 1024)
    print(f"  Wrote {OBSOLESCENCE_INDEX_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
