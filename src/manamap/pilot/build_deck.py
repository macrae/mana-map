"""Pilot: build a legal, tier-conditioned 99 from a commander and a brief.

This is the deterministic half of Deck Building v2, and it stands alone on
purpose: it produces a complete, legal, goldfishable deck with no agent
involvement at all. That gives the agent layer a baseline to be measured
against, keeps the expensive path optional, and means a cache miss degrades to
a worse deck rather than no deck.

The order matters. Hard constraints are applied to the *pool*, before anything
is scored, so a Bracket 2 build never sees a Game Changer and cannot be talked
into one. Then slots are filled by role budget, and only then does the bracket
check run again — because two cards that are each individually fine can combine
into a line that isn't, and catching that is the difference between a filter
and an actual tier guarantee.

Every slot keeps its runners-up. A build plan is a strong starting point, not a
verdict, and a slot whose second choice was 0.01 behind should say so.

Deterministic: same brief, same artifacts, same plan, byte for byte.
"""

import json
import math

import numpy as np
import pandas as pd

from manamap.analysis.common import (
    build_name_index,
    color_identity_mask,
    parse_color_identity,
    parse_tag_set,
)
from manamap.config import (
    ABILITY_EMBEDDINGS_PATH,
    BRACKET_DEFAULT,
    BRACKET_MAX,
    BRACKETS,
    CARD_ROLES_PATH,
    COMBO_DETAILS_PATH,
    DECK_BUILD_ALTERNATES,
    DECK_BUILD_EDHREC_BY_BRACKET,
    DECK_BUILD_MAX_BRACKET_PASSES,
    DECK_BUILD_WEIGHTS,
    DECK_CURVE_SWEET_SPOT,
    DECK_CURVE_TOLERANCE,
    DECK_ROLE_BUDGET,
    DECK_ROLE_GROUPS,
    EMBEDDINGS_PATH,
    OUTPUT_CSV_PATH,
    SYNERGY_GRAPH_PATH,
    SYNERGY_RULES,
)
from manamap.pilot import bracket as bracket_mod
from manamap.pilot import manabase
from manamap.pilot.common import deck_dir

BASIC_LANDS = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}


class BriefError(ValueError):
    """The brief is unusable — the caller must fix it, not build around it."""


def load_brief(slug):
    """Load and sanity-check data/decks/<slug>/brief.json."""
    path = deck_dir(slug) / "brief.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — author it first. Minimum: "
            f'{{"slug": "{slug}", "commander": "<name>", "bracket": {BRACKET_DEFAULT}}}'
        )
    with open(path) as f:
        brief = json.load(f)
    if not brief.get("commander"):
        raise BriefError("brief.json has no commander")
    target = brief.get("bracket", BRACKET_DEFAULT)
    if target not in BRACKETS:
        raise BriefError(f"bracket must be 1-{BRACKET_MAX}, got {target!r}")
    brief["bracket"] = target
    brief.setdefault("must_include", [])
    brief.setdefault("must_exclude", [])
    return brief


def commander_identity(row):
    """Colour identity a commander licenses, plus a legality check.

    Rejects anything that can't actually be a commander. Planeswalkers and
    other non-creatures qualify only via explicit "can be your commander" text.
    """
    type_line = str(row.get("type_line", "") or "")
    text = str(row.get("oracle_text", "") or "")
    front = type_line.split(" // ")[0]
    is_legendary_creature = "Legendary" in front and "Creature" in front
    if not is_legendary_creature and "can be your commander" not in text.lower():
        raise BriefError(
            f"{row['name']} is not a legal commander "
            f"(needs to be a legendary creature, or say 'can be your commander')"
        )
    if row.get("legal_commander") != "legal":
        raise BriefError(f"{row['name']} is not legal in Commander")
    return parse_color_identity(row.get("color_identity", ""))


def role_group(roles):
    """Which budget line a card's roles satisfy, most specific first."""
    for group, members in DECK_ROLE_GROUPS.items():
        if any(r in members for r in roles):
            return group
    return "flex"


def candidate_pool(df, identity, target_bracket, brief):
    """Legal, in-identity, bracket-safe cards. Hard filters only, before scoring."""
    mask = color_identity_mask(df, identity)
    mask &= (df["legal_commander"] == "legal").to_numpy()
    mask &= (df["name"] != brief["commander"]).to_numpy()
    # Un-set sticker sheets carry legal_commander == "legal" in Scryfall's data
    # (48 rows), have real mechanical tags, and therefore score — the pool
    # surfaced Familiar Beeble Mascot as a wincon before this. They are not
    # castable cards. Any consumer filtering on legality alone hits this.
    mask &= ~df["type_line"].fillna("").str.contains("Stickers", regex=False).to_numpy()

    excluded = set(brief.get("must_exclude", []))
    if excluded:
        mask &= ~df["name"].isin(excluded).to_numpy()

    # The bracket's card-level restrictions. Applied to the pool so the
    # scorer is never in a position to want something it can't have.
    limits = BRACKETS[target_bracket]
    if limits["game_changers"] == 0:
        mask &= ~df["game_changer"].to_numpy()
    if not limits["mass_land_denial"]:
        from manamap.config import MASS_LAND_DENIAL
        mask &= ~df["name"].isin(MASS_LAND_DENIAL).to_numpy()

    return df[mask]


def synergy_affinity(card_tags, deck_tags):
    """Rule-based complementarity between one card and the whole deck.

    The synergy graph is a top-10 shortlist per card, which makes it a
    retrieval aid and not a scoring function — you cannot ask it "how well does
    X fit deck D". So the rules are applied directly against the deck's tag
    union instead.
    """
    score = 0
    for tag_a, tag_b, _label in SYNERGY_RULES:
        if tag_a in card_tags and tag_b in deck_tags:
            score += 1
        if tag_b in card_tags and tag_a in deck_tags:
            score += 1
    return score


def edhrec_component(rank, max_rank):
    """Log-scaled popularity. A linear rank transform says rank 1 and rank 100
    are nearly identical, which is the opposite of true."""
    if rank is None or (isinstance(rank, float) and math.isnan(rank)):
        return 0.3
    return max(0.0, 1.0 - math.log1p(rank) / math.log1p(max_rank))


def castability(mana_cost):
    """How demanding this cost is on the mana base.

    Colour *legality* is already guaranteed by the pool filter, so scoring it
    would be a constant. What still varies is intensity: {B}{B}{B} is a real
    cost that {2}{B} isn't.
    """
    pips = manabase.count_pips(mana_cost)
    heaviest = max(pips.values()) if pips else 0
    if heaviest <= 1:
        return 1.0
    return max(0.0, 1.0 - (heaviest - 1) * 0.25)


def curve_fit(cmc):
    """Reward cards near the sweet spot, taper hard past it."""
    if cmc is None or (isinstance(cmc, float) and math.isnan(cmc)):
        return 0.5
    distance = max(0.0, float(cmc) - DECK_CURVE_SWEET_SPOT)
    return max(0.0, 1.0 - distance / DECK_CURVE_TOLERANCE)


def score_candidates(pool, embeddings, name_index, commander_name, identity,
                     deck_tags, combo_partners, target_bracket, deck_names=None):
    """Score every candidate against the deck as it currently stands."""
    deck_names = set(deck_names or {commander_name})
    weights = DECK_BUILD_WEIGHTS
    edhrec_scale = DECK_BUILD_EDHREC_BY_BRACKET[target_bracket]
    commander_vec = embeddings[name_index[commander_name]]
    max_rank = float(pool["edhrec_rank"].max() or 1.0)
    identity = set(identity) or set("WUBRG")

    scores = []
    for row in pool.itertuples(index=False):
        idx = name_index.get(row.name)
        similarity = float(embeddings[idx] @ commander_vec) if idx is not None else 0.0
        similarity = max(0.0, similarity)

        tags = parse_tag_set(getattr(row, "mechanical_tags", ""))
        synergy = min(synergy_affinity(tags, deck_tags) / 6.0, 1.0)
        combo = min(len(combo_partners.get(row.name, set()) & deck_names) / 3.0, 1.0)
        edhrec = edhrec_component(getattr(row, "edhrec_rank", None), max_rank) * edhrec_scale
        cast = castability(getattr(row, "mana_cost", ""))
        curve = curve_fit(getattr(row, "cmc", None))

        total = (
            weights["similarity"] * similarity
            + weights["synergy"] * synergy
            + weights["combo"] * combo
            + weights["curve"] * curve
            + weights["edhrec"] * edhrec
            + weights["castability"] * cast
        )
        scores.append({
            "name": row.name,
            "score": total,
            "components": {
                "similarity": round(similarity, 4),
                "synergy": round(synergy, 4),
                "combo": round(combo, 4),
                "curve": round(curve, 4),
                "edhrec": round(edhrec, 4),
                "castability": round(cast, 4),
            },
        })
    # Sort by score then name so ties are stable across runs.
    scores.sort(key=lambda s: (-s["score"], s["name"]))
    return scores


def fill_slots(scored, roles, budget, must_include):
    """Fill each budget line with the best-scoring cards that satisfy it.

    Returns (slots, leftovers). Each slot carries its runners-up so the plan
    reads as a starting point rather than a verdict.
    """
    by_group = {group: [] for group in budget}
    for entry in scored:
        by_group.setdefault(role_group(roles.get(entry["name"], [])), []).append(entry)

    taken = set()
    slots = []

    # Must-includes claim their slot first, whatever their score.
    for name in must_include:
        if name in taken:
            continue
        group = role_group(roles.get(name, []))
        slots.append({"name": name, "role": group, "reason": "must_include", "alternates": []})
        taken.add(name)

    for group, count in budget.items():
        if group == "lands":
            continue
        filled = sum(1 for s in slots if s["role"] == group)
        pool = [e for e in by_group.get(group, []) if e["name"] not in taken]
        # Flex absorbs anything left over, so it draws from every group.
        if group == "flex":
            pool = [e for e in scored if e["name"] not in taken]
        for entry in pool[: max(0, count - filled)]:
            alternates = [
                {"name": alt["name"], "delta": round(entry["score"] - alt["score"], 4)}
                for alt in pool[pool.index(entry) + 1: pool.index(entry) + 1 + DECK_BUILD_ALTERNATES]
                if alt["name"] not in taken
            ]
            slots.append({
                "name": entry["name"],
                "role": group,
                "score": round(entry["score"], 4),
                "components": entry["components"],
                "alternates": alternates,
            })
            taken.add(entry["name"])

    return slots, taken


def enforce_bracket(slots, scored, roles, card_flags, details, commanders, target):
    """Swap out cards until the computed floor fits the target.

    Two cards each individually legal for a bracket can combine into a line
    that isn't. This is where that gets caught — a pool filter alone cannot.
    Bounded, and it fails loudly rather than shipping an off-bracket deck.
    """
    cut = []
    by_name = {e["name"]: e for e in scored}

    for _ in range(DECK_BUILD_MAX_BRACKET_PASSES):
        names = [s["name"] for s in slots]
        report = bracket_mod.assess(names, card_flags, roles, details, commanders)
        if report["floor"] <= target:
            return slots, cut, report

        offenders = bracket_mod.offending_cards(report, target)
        if not offenders:
            return slots, cut, report

        # Cut the most-implicated card, breaking ties toward the cheapest slot.
        taken = set(names)
        offender_names = [o["name"] for o in offenders]
        victim = min(
            offender_names,
            key=lambda n: (by_name.get(n, {}).get("score", 0.0), n),
        )
        slot = next(s for s in slots if s["name"] == victim)
        replacement = next(
            (e for e in scored
             if e["name"] not in taken and role_group(roles.get(e["name"], [])) == slot["role"]),
            None,
        )
        reason = next(o["reasons"] for o in offenders if o["name"] == victim)
        cut.append({
            "name": victim,
            "reasons": reason,
            "replaced_by": replacement["name"] if replacement else None,
        })
        slots.remove(slot)
        if replacement:
            slots.append({
                "name": replacement["name"],
                "role": slot["role"],
                "score": round(replacement["score"], 4),
                "components": replacement["components"],
                "alternates": [],
            })

    names = [s["name"] for s in slots]
    report = bracket_mod.assess(names, card_flags, roles, details, commanders)
    raise BriefError(
        f"could not reach bracket {target} in {DECK_BUILD_MAX_BRACKET_PASSES} passes "
        f"(floor is still {report['floor']}) — the brief may be asking for a commander "
        f"whose best cards are out of bracket"
    )


def build(slug):
    """Build a deck plan for `slug`. Returns the plan dict."""
    brief = load_brief(slug)
    target = brief["bracket"]

    df = pd.read_csv(OUTPUT_CSV_PATH)
    with open(CARD_ROLES_PATH) as f:
        roles = json.load(f)["roles"]
    with open(COMBO_DETAILS_PATH) as f:
        details = json.load(f)
    with open(SYNERGY_GRAPH_PATH) as f:
        json.load(f)  # presence check; scoring uses the rules directly

    matches = df[df["name"] == brief["commander"]]
    if matches.empty:
        raise BriefError(f"commander {brief['commander']!r} is not in cards.csv")
    commander = matches.iloc[0].to_dict()
    identity = commander_identity(commander)

    embeddings = np.load(
        ABILITY_EMBEDDINGS_PATH if ABILITY_EMBEDDINGS_PATH.exists() else EMBEDDINGS_PATH
    )
    name_index = build_name_index(df)

    pool = candidate_pool(df, identity, target, brief)
    card_flags = {
        row.name_: {"game_changer": bool(row.game_changer), "legal_commander": row.legal_commander}
        for row in df.rename(columns={"name": "name_"}).itertuples(index=False)
    }
    combo_partners = {
        name: {c for i in idxs for c in details["combos"][i]["cards"] if c != name}
        for name, idxs in details["by_card"].items()
    }

    spell_pool = pool[pool["supertype"] != "Land"]
    deck_tags = parse_tag_set(commander.get("mechanical_tags", ""))
    scored = score_candidates(
        spell_pool, embeddings, name_index, commander["name"],
        identity, deck_tags, combo_partners, target,
    )

    budget = dict(DECK_ROLE_BUDGET)
    slots, _ = fill_slots(scored, roles, budget, brief["must_include"])
    slots, cut, report = enforce_bracket(
        slots, scored, roles, card_flags, details, {commander["name"]}, target
    )

    # Mana base last: it needs the spells it has to cast.
    spell_rows = df[df["name"].isin([s["name"] for s in slots])].to_dict("records")
    land_pool = pool[(pool["supertype"] == "Land") & (~pool["type_line"].str.contains("Basic", na=False))]
    basics = {
        colour: df[df["name"] == BASIC_LANDS[colour]].iloc[0].to_dict()
        for colour in identity
        if not df[df["name"] == BASIC_LANDS[colour]].empty
    }
    lands, mana_diag = manabase.build(
        spell_rows, land_pool.to_dict("records"), budget["lands"], basics
    )

    plan = {
        "slug": slug,
        "commander": commander["name"],
        "color_identity": sorted(identity),
        "bracket": {
            "target": target,
            "target_name": BRACKETS[target]["name"],
            "computed_floor": report["floor"],
            "drivers": report["drivers"],
            "within_target": report["floor"] <= target,
        },
        "role_budget": budget,
        "role_budget_grounding": "provisional — pending strategy:deckbuilding.ratios",
        "slots": sorted(slots, key=lambda s: (s["role"], s["name"])),
        "lands": sorted({land["name"] for land in lands}),
        "land_counts": _land_counts(lands),
        "manabase": mana_diag,
        "cut_for_bracket": cut,
        "notes": report["notes"],
        "generated_by": "manamap pilot build-deck (deterministic; no agent involvement)",
    }
    return plan


def _land_counts(lands):
    counts = {}
    for land in lands:
        counts[land["name"]] = counts.get(land["name"], 0) + 1
    return dict(sorted(counts.items()))


# Split cards are written in full on a decklist ("Fire // Ice"); every other
# multi-face layout is written as its front face ("Mosswood Dreadknight", not
# "Mosswood Dreadknight // Dread Whispers"). cards.csv and the graphs use the
# joined form as their key, so the translation happens here, at the boundary.
FULL_NAME_LAYOUTS = {"split"}


def decklist_name(name, layout):
    """The name a decklist — and Scryfall's collection endpoint — expects."""
    if " // " not in name or layout in FULL_NAME_LAYOUTS:
        return name
    return name.split(" // ")[0]


def decklist_text(plan, layouts=None):
    """Render the plan as a decklist.txt that fetch-deck can parse."""
    layouts = layouts or {}

    def render(name):
        return decklist_name(name, layouts.get(name, ""))

    lines = [f"1 {render(plan['commander'])} *CMDR*"]
    for slot in sorted(plan["slots"], key=lambda s: s["name"]):
        lines.append(f"1 {render(slot['name'])}")
    for name, count in plan["land_counts"].items():
        lines.append(f"{count} {render(name)}")
    return "\n".join(lines) + "\n"


def main(args):
    plan = build(args.slug)
    base = deck_dir(args.slug)

    out = base / "build_plan.json"
    with open(out, "w") as f:
        json.dump(plan, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    total = len(plan["slots"]) + sum(plan["land_counts"].values()) + 1
    print(f"Wrote {out}")
    print(f"  {plan['commander']} — {''.join(plan['color_identity']) or 'C'} — "
          f"{total} cards")
    print(f"  bracket target {plan['bracket']['target']} "
          f"({plan['bracket']['target_name']}), computed floor "
          f"{plan['bracket']['computed_floor']}")
    if plan["cut_for_bracket"]:
        print(f"  cut for bracket: "
              f"{', '.join(c['name'] for c in plan['cut_for_bracket'])}")
    short = plan["manabase"]["shortfalls"]
    if short:
        print(f"  mana base shortfalls: {short}")

    if getattr(args, "write_decklist", False):
        layouts = pd.read_csv(OUTPUT_CSV_PATH, usecols=["name", "layout"])
        layouts = dict(zip(layouts["name"], layouts["layout"]))
        path = base / "decklist.txt"
        path.write_text(decklist_text(plan, layouts), encoding="utf-8")
        print(f"  Wrote {path}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot build-deck <slug>`.")
