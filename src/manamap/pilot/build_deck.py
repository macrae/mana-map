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
    DECK_BUILD_ALTERNATES,
    DECK_BUILD_COMBO_COMPLETIONS,
    DECK_BUILD_EDHREC_BY_BRACKET,
    DECK_BUILD_MAX_BRACKET_PASSES,
    DECK_BUILD_WEIGHTS,
    DECK_CURVE_SWEET_SPOT,
    DECK_CURVE_TOLERANCE,
    DECK_ROLE_BUDGET,
    DECKS_DIR,
    DECK_ROLE_GROUPS,
    EMBEDDINGS_PATH,
    SYNERGY_GRAPH_PATH,
    SYNERGY_RULES,
)
from manamap.pilot import bracket as bracket_mod
from manamap.pilot import manabase
from manamap.pilot.card_pool import card_flags as corpus_card_flags
from manamap.pilot.card_pool import load_frame
from manamap.pilot.common import (
    commander_rejection,
    deck_dir,
    load_card_roles,
    load_combo_details,
)

BASIC_LANDS = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}


class BriefError(ValueError):
    """The brief is unusable — the caller must fix it, not build around it."""


def scaffold_brief(slug, commander, library=(), theme=None, bracket=None,
                   pool_files=()):
    """Write `brief.json` for a new deck. PRD-v1 §7.4 — the start of a build-out.

    The library goes in as `must_include`: those are the cards the pilot
    deliberately kept, and `must_include` is exactly the promise that they are in
    the 99. Everything else is the builder's to choose.

    THE DECK ARRIVES ON THE BENCH, NOT SLEEVED, and nothing here pretends
    otherwise — no `paper` block is written. §7.4 says a new deck lands at
    v0.1.0, and 0.x is the version of a list that exists only digitally;
    reaching 1.0.0 is the act of sleeving it, which only the pilot can do.
    """
    # `DECKS_DIR / slug`, not `deck_dir(slug)`: that helper is a READER and
    # refuses a directory that does not exist yet, which is correct for every
    # other caller and exactly wrong for the one creating a deck.
    base = DECKS_DIR / slug
    path = base / "brief.json"
    if path.exists():
        raise SystemExit(f"{path} already exists — edit it, or pick another slug")
    base.mkdir(parents=True, exist_ok=True)

    brief = {"slug": slug, "commander": commander,
             "bracket": bracket or BRACKET_DEFAULT,
             "must_include": list(library), "must_exclude": []}
    if theme:
        brief["theme"] = theme
    if pool_files:
        brief["pool_files"] = list(pool_files)
    path.write_text(json.dumps(brief, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path, brief


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
    brief["_pool"] = resolve_pool(brief)
    return brief


def deck_printings(brief, names):
    """The owned printing for each name in the deck, when the brief names a pool.

    A build from a physical collection has to name the physical card. Without this
    the decklist carries bare names, `fetch-deck` takes whatever Scryfall considers
    default, and the manual illustrates and credits cards the pilot does not own.
    """
    files = brief.get("pool_files") or []
    if not files:
        return {}
    from manamap.pilot.pool_facts import collect_paths, load_cards, pool_printings

    owned = pool_printings(collect_paths(files, []), load_cards())
    return {n: owned[n] for n in names if n in owned}


def resolve_pool(brief):
    """The names a build may draw from, or None for the whole format.

    Two ways to say it, because they answer different questions. `pool` is an
    explicit name list. `pool_files` points at decklists — your collection as
    exported from paper — and is parsed by the same reader `pool-facts` uses, so
    a box analysed and a box built from can never disagree about what is in it.

    Basics are deliberately NOT constrained (see `build`): you always own more
    Swamps, and a paper pool that happens to list four of them should not cap
    the mana base at four.
    """
    names = set(brief.get("pool") or [])
    files = brief.get("pool_files") or []
    if files:
        # Imported lazily: pool_facts imports this module for `role_group`, so a
        # module-level import here would be a cycle.
        from manamap.pilot.pool_facts import collect_paths, load_cards, read_sources

        per_file, unresolved = read_sources(collect_paths(files, []), load_cards())
        if unresolved:
            raise BriefError(
                f"{len(unresolved)} pool card(s) did not resolve against cards.csv: "
                + ", ".join(sorted(unresolved)[:5])
            )
        for counts in per_file.values():
            names.update(counts)
    if not names:
        return None
    if brief["commander"] not in names:
        raise BriefError(
            f"the pool does not contain the commander ({brief['commander']})"
        )
    missing = [n for n in brief["must_include"] if n not in names]
    if missing:
        raise BriefError(f"must_include names outside the pool: {', '.join(missing)}")
    return names


def commander_identity(row):
    """Colour identity a commander licenses, plus a legality check.

    Rejects anything that can't actually be a commander. Planeswalkers and
    other non-creatures qualify only via explicit "can be your commander" text.
    """
    reason = commander_rejection(row)
    if reason:
        raise BriefError(f"{row['name']} is {reason}")
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

    # Build from a physical collection rather than the format. A hard filter on
    # the pool, alongside the others, so the scorer never wants what you cannot
    # sleeve — the same reason the bracket limits are applied here.
    if brief.get("_pool") is not None:
        mask &= df["name"].isin(brief["_pool"]).to_numpy()

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
        # Clamped: deck scoring against a centroid wants "how much does this belong",
        # so anti-correlated reads as zero rather than as a penalty large enough to
        # swamp the other five factors. Retrieval paths (analysis/common, synergy,
        # power_creep) deliberately do NOT clamp — they need the true ordering.
        # This is the ONLY deck scorer in the repo; a second one in the browser
        # diverged from these weights on five of six factors.
        # Retrieval paths (analysis/common.py, synergy, power_creep) deliberately do NOT
        # clamp — they need the true ordering. Consistent by role, enforced nowhere.
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


# The cited curve, as a share of the nonland deck. `DECK_AXIS_TARGETS["curve"]`
# quotes strategy:deckbuilding.curve — "puts the modal mana value at 2, with 15.7
# two-drops and 15.4 three-drops in the average deck and only ~1.5 cards at mana
# value 8+" — and `deck_audit` has always MEASURED against it while the builder never
# read it. These are that quote's numbers over a ~64-card nonland deck, with the
# unstated middle filled monotonically down from the mode and the 8+ tail set by the
# "~1.5" clause. Shares rather than counts so a different nonland total scales.
CURVE_SHARE = {0: 0.02, 1: 0.14, 2: 0.245, 3: 0.24, 4: 0.16, 5: 0.10, 6: 0.06,
               7: 0.02, 8: 0.015}


def curve_targets(nonland_total):
    """`{mana value bucket: how many cards belong there}`, 8 meaning "8 or more".

    The builder scored each card independently and took the top N per role, and
    `curve_fit` is a flat plateau to mana value 3 then a monotone decline — so the
    top N were ALWAYS cheap. Measured on the first kinnan baseline: 64 nonland cards
    with curve {0:1, 1:11, 2:28, 3:24} and NOTHING above 3, 29 of them mana dorks. A
    legal deck that ramps into nothing. N independent maxima are not a distribution;
    this is the distribution.
    """
    raw = {mv: share * nonland_total for mv, share in CURVE_SHARE.items()}
    out = {mv: int(v) for mv, v in raw.items()}
    # Hand the rounding remainder to the buckets that lost the most to truncation,
    # so the totals match exactly and the mode is never starved by rounding.
    short = nonland_total - sum(out.values())
    for mv in sorted(raw, key=lambda m: raw[m] - out[m], reverse=True)[:max(0, short)]:
        out[mv] += 1
    return out


def curve_bucket(cmc):
    """A card's curve bucket; 8 is "8 or more", matching the cited tail."""
    if cmc is None or (isinstance(cmc, float) and math.isnan(cmc)):
        return 0
    return min(int(float(cmc)), 8)


def role_budget_for(brief, roles):
    """The role budget to fill to, and where it came from. §7.3.

    `DECK_ROLE_BUDGET` is ONE FLAT BUDGET for every deck. Its own comment in
    `config.py` calls it PROVISIONAL, `upgrade_facts` printed its shortfalls as
    "Context, not evidence", and a stax deck and a voltron deck plainly do not
    want the same eight numbers.

    When the brief names a `theme`, the budget is measured instead: the role
    histogram of that archetype's average deck on EDHREC, collapsed through
    `role_group` — THE SAME mapping `fill_slots` uses, so the budget and the
    filling cannot disagree about what counts as ramp.

    It is SHAPED, not copied. The measured histogram is renormalised onto the
    provisional budget's total, because that total is what the rest of the
    builder is sized against (land counts, the curve quota, the bracket pass);
    taking EDHREC's raw counts would change the deck's size as a side effect of
    choosing a style. What the theme decides is the SHAPE — how the same number
    of slots is divided — which is the part that actually differs.

    Falls back with a reason rather than an exception. A theme that cannot be
    fetched must not stop a build: the flat budget is what every deck here was
    built on until now.
    """
    theme = (brief or {}).get("theme")
    if not theme:
        return dict(DECK_ROLE_BUDGET), "provisional — pending strategy:deckbuilding.ratios"

    try:
        from manamap.pilot.archetypes import role_template

        template = role_template(brief["commander"], theme, roles)
    except Exception as exc:
        return (dict(DECK_ROLE_BUDGET),
                f"provisional — could not read the {theme!r} template ({exc})")

    # Collapse the fine roles onto the budget's coarse lines through the
    # builder's own mapping.
    #
    # LANDS ARE EXCLUDED FIRST. `role_group` has no land line, so every
    # `land:basic` / `land:tapped` / `land:utility` fell into `flex` — 32 of
    # voltron's 135 role-copies — inflating flex and deflating every real line
    # in the same stroke. Lands are budgeted separately by `land_counts`, so
    # counting them here was double-counting the mana base into the spells.
    measured = {}
    for role, count in template["roles"].items():
        if role.startswith("land:"):
            continue
        measured[role_group([role])] = measured.get(role_group([role]), 0) + count

    spells = {k: v for k, v in DECK_ROLE_BUDGET.items() if k != "lands"}
    total = sum(spells.values())
    pool = sum(v for k, v in measured.items() if k != "lands") or 1
    budget = {"lands": DECK_ROLE_BUDGET["lands"]}
    for line in spells:
        budget[line] = max(0, round(measured.get(line, 0) * total / pool))

    # Renormalising by rounding does not land on the total; `flex` absorbs the
    # remainder because that is already what it does for overflow and shortfall.
    budget["flex"] = max(0, budget.get("flex", 0) + total - sum(
        v for k, v in budget.items() if k != "lands"))
    return budget, (f"measured — {template['theme']} role histogram over "
                    f"{template['deck_size']} cards (EDHREC), shaped to the "
                    f"provisional budget's total")


def legal_must_includes(names, identity, frame):
    """Split the must-includes into those the commander can legally play, and
    those it cannot.

    MUST-INCLUDES WERE NEVER IDENTITY-CHECKED. `fill_slots` pins them "whatever
    their score" — before any budget line is consulted and before the candidate
    pool's identity filter has any say, because they never pass through it. That
    was survivable while a brief was hand-authored by someone who knew the
    commander. It stopped being survivable the moment `must_include` came from
    the LIBRARY, which holds whatever you kept while exploring 34,890 cards.

    Measured on the report that found it: a Zur (WUB) build pinned `Grolnok, the
    Omnivore` — mono-green — and the deck came out at 88 cards, because the mana
    base then tried to support a colour the commander does not license and gave
    up twelve lands short. One illegal card, and the failure surfaced as a
    missing mana base.

    DROPPED AND REPORTED, not silently removed and not fatal. The pilot
    deliberately kept these cards, so the plan says which ones could not come
    and why; building an illegal deck is worse than building a smaller library.
    """
    if not names:
        return [], []
    ident = {c for c in (identity or [])}
    by_name = dict(zip(frame["name"], frame["color_identity"]))
    keep, illegal = [], []
    for name in names:
        raw = by_name.get(name)
        if raw is None:                       # not in the corpus; another check owns that
            keep.append(name)
            continue
        # `parse_color_identity`, not an ad-hoc split: a colourless card's cell is
        # pandas NaN, and `str(nan)` is the string 'nan' — truthy, and outside every
        # identity, so EVERY colourless card was reported illegal. Sol Ring, the
        # artifact lands and Darksteel Citadel could not be pinned at all.
        card = parse_color_identity(raw)
        outside = card - ident
        if outside:
            illegal.append({"name": name, "identity": sorted(card),
                            "outside": sorted(outside)})
        else:
            keep.append(name)
    return keep, illegal


def fill_slots(scored, roles, budget, must_include, cmcs=None):
    """Fill each budget line with the best-scoring cards that satisfy it.

    Returns (slots, taken, effective_budget). Each slot carries its runners-up so
    the plan reads as a starting point rather than a verdict.

    `effective_budget` is what was actually filled to, which is not always what
    was asked for: must-includes are pinned before any budget line is consulted,
    so a brief can force a group over its allowance. It is returned rather than
    recomputed from the slots because `validate_build._validate_budget` compares
    the two, and a budget derived from the slots it is meant to check would make
    that comparison vacuous.
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

    def count_in(group):
        return sum(1 for s in slots if s["role"] == group)

    # The curve is a SHAPE, filled across every role rather than inside one. Role
    # quotas and mana-value quotas are crossed: a card is taken in score order unless
    # its bucket is already full, and a bucket that cannot be filled from a role's
    # candidates simply is not — `flex` and the remainder pass absorb it, exactly as
    # they already absorb role overflow and shortfall.
    nonland_total = sum(v for k, v in budget.items() if k != "lands")
    curve_left = curve_targets(nonland_total) if cmcs else {}
    for slot in slots:                      # must-includes already spent their bucket
        b = curve_bucket((cmcs or {}).get(slot["name"]))
        if b in curve_left:
            curve_left[b] -= 1

    def take(group, want, respect_curve=True):
        """Fill `want` more slots from `group`, best score first.

        `respect_curve=False` is the second pass: once a group's shape-fitting
        candidates are exhausted, a legal deck of the right SIZE beats a perfect
        curve, so the remainder is filled on score alone.
        """
        if want <= 0:
            return
        available = ([e for e in scored if e["name"] not in taken] if group == "flex"
                     else [e for e in by_group.get(group, []) if e["name"] not in taken])
        placed = 0
        for i, entry in enumerate(available):
            if placed >= want:
                break
            bucket = curve_bucket((cmcs or {}).get(entry["name"]))
            if respect_curve and curve_left and curve_left.get(bucket, 0) <= 0:
                continue
            alternates = [
                {"name": alt["name"], "delta": round(entry["score"] - alt["score"], 4)}
                for alt in available[i + 1: i + 1 + DECK_BUILD_ALTERNATES]
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
            if bucket in curve_left:
                curve_left[bucket] -= 1
            placed += 1

    # `flex` is the slack line and is filled LAST, because it is the only group
    # that can absorb the other two failure modes. Both are real and both shipped
    # a wrong-size plan before they were caught:
    #
    #   OVERFLOW — must-includes are pinned before any budget is consulted, so a
    #   brief can push a group past its allowance. A 23-card pin list put `wincon`
    #   at 4 against a budget of 3 and produced a 101-card plan.
    #
    #   SHORTFALL — a group can run out of candidates. Mono-black held only 2
    #   cards the taxonomy calls `sweeper` against a budget of 3, and the plan
    #   came out at 99. Nothing gave the missing slot back.
    #
    # Filling non-flex groups first, then giving flex whatever is left of the
    # non-land total, handles both without special-casing either.
    for group, count in budget.items():
        if group in ("lands", "flex"):
            continue
        take(group, count - count_in(group))
        # A role whose shape-fitting candidates ran out still owes its slots.
        take(group, count - count_in(group), respect_curve=False)

    effective = dict(budget)
    for group in budget:
        if group not in ("lands", "flex"):
            effective[group] = count_in(group)

    if "flex" in budget:
        nonland_target = sum(v for k, v in budget.items() if k != "lands")
        placed = sum(effective[g] for g in effective if g not in ("lands", "flex"))
        take("flex", (nonland_target - placed) - count_in("flex"))
        take("flex", (nonland_target - placed) - count_in("flex"), respect_curve=False)
        effective["flex"] = count_in("flex")

    return slots, taken, effective


def complete_combos(slots, scored, roles, details, seed_names, limit=DECK_BUILD_COMBO_COMPLETIONS):
    """Swap in the ONE missing card of a combo line the deck already half-holds.

    `score_candidates` scores every card ONCE, against `{commander}`, so its combo
    term only ever asks "does this combo with the commander" — never "with anything
    already chosen" — and a single partner earns at most `combo_weight × 1/3`. The
    first kinnan baseline held **23 Kinnan combo partners and not one completion**:
    the halves that also scored well on similarity and curve got in, and the cards
    that would have finished a line did not, because nothing in the pipeline knows a
    pair is a pair.

    Reads real LINES, not `combo_partners`. That map is flat co-occurrence, so
    "partners with something present" is true of a hundred cards the moment the
    commander is on the list — it cannot tell a completion from a coincidence. A
    completion here means: some line in `combo_details` has exactly one card the deck
    does not hold, and this is it.

    Bounded like `enforce_bracket`, and for the same reason — a deck that is only
    combo pieces is a different failure from one that has none. The victim is the
    lowest-scoring slot of the completion's own role, so the role budget survives;
    a completion is ALLOWED to score lower than what it replaces, because raw score
    is exactly the thing that cannot see the pair.
    """
    present = {s["name"] for s in slots} | set(seed_names)
    by_name = {e["name"]: e for e in scored}
    combos, by_card = details.get("combos") or [], details.get("by_card") or {}
    completed, protected = [], set(seed_names)
    for _ in range(limit):
        # Every line reachable from what the deck already holds.
        best = None
        seen = set()
        for held in sorted(present):
            for idx in by_card.get(held, []):
                if idx in seen:
                    continue
                seen.add(idx)
                line = combos[idx] if idx < len(combos) else None
                if not line:
                    continue
                missing = [c for c in line["cards"] if c not in present]
                if len(missing) != 1:
                    continue
                cand = by_name.get(missing[0])
                if cand is None:            # out of pool, identity or bracket
                    continue
                if best is None or cand["score"] > best[0]["score"]:
                    best = (cand, line)
        if best is None:
            return slots, completed
        entry, line = best
        group = role_group(roles.get(entry["name"], []))
        victims = [s for s in slots if s.get("reason") != "must_include"
                   and s["role"] == group and s["name"] not in protected]
        if not victims:
            return slots, completed
        victim = min(victims, key=lambda s: (s.get("score", 0.0), s["name"]))
        slots = [s for s in slots if s is not victim]
        slots.append({"name": entry["name"], "role": group,
                      "score": round(entry["score"], 4), "components": entry["components"],
                      "alternates": [], "reason": "combo_completion"})
        present.discard(victim["name"])
        present.add(entry["name"])
        protected.add(entry["name"])
        completed.append({"added": entry["name"], "replaced": victim["name"],
                          "line": line["cards"], "produces": line.get("produces")})
    return slots, completed


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

    df = load_frame()
    roles = load_card_roles()
    details = load_combo_details()
    if not SYNERGY_GRAPH_PATH.exists():  # presence check; scoring uses the rules directly
        raise BriefError(f"{SYNERGY_GRAPH_PATH} missing — run `manamap synergy` first")

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
    # Aliased on import: `card_flags` is also the parameter name in
    # `enforce_bracket` and `bracket.assess`, and a bare import would let a
    # call site silently pass the FUNCTION where a dict is expected.
    flags = corpus_card_flags()
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

    budget, grounding = role_budget_for(brief, roles)
    cmcs = dict(zip(spell_pool["name"], spell_pool["cmc"]))
    keep, illegal = legal_must_includes(brief["must_include"], identity, df)
    slots, taken, effective_budget = fill_slots(
        scored, roles, budget, keep, cmcs=cmcs)
    slots, completed = complete_combos(
        slots, scored, roles, details, {commander["name"]})
    slots, cut, report = enforce_bracket(
        slots, scored, roles, flags, details, {commander["name"]}, target
    )

    # Mana base last: it needs the spells it has to cast.
    # The name-join must dedupe: card names collide with Mystery Booster playtest
    # cards ("Pick Your Poison" exists as MKM's {G} sorcery AND a {3}{B}{G} cmb2
    # playtest card), and a ghost row here invents pip requirements in colours
    # the deck cannot produce. Playtest doubles are never commander-legal, so
    # keeping legal rows first and deduping by name always picks the real card.
    slot_names = [s["name"] for s in slots]
    spell_rows = (
        df[df["name"].isin(slot_names)]
        .sort_values("legal_commander", key=lambda s: s != "legal")
        .drop_duplicates("name")
        .to_dict("records")
    )
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
        "role_budget": effective_budget,
        "role_budget_target": budget,
        # Non-empty when must_include forced a group past its allowance. Recorded
        # rather than smoothed over: a deviation the brief caused should be
        # visible to whoever reads the plan.
        "role_budget_deviation": {
            g: {"target": budget[g], "actual": effective_budget[g]}
            for g in budget if effective_budget.get(g) != budget[g]
        },
        "role_budget_grounding": grounding,
        # Kept cards the commander cannot legally play. Named rather than
        # dropped in silence — the pilot chose them on purpose.
        "must_include_illegal": illegal,
        # Recorded so `validate-build` can re-derive the pool and prove every
        # non-basic name was actually owned. A build that silently reached
        # outside the collection is the one failure a paper pilot cannot use.
        # Only for the cards in this deck — the whole 764-entry map would bloat the
        # plan and say nothing about the build.
        "printings": deck_printings(brief, [commander["name"]]
                                    + [s["name"] for s in slots]
                                    + list(_land_counts(lands))),
        "pool": (
            None if brief.get("_pool") is None
            else {"files": brief.get("pool_files") or [], "size": len(brief["_pool"])}
        ),
        "slots": sorted(slots, key=lambda s: (s["role"], s["name"])),
        "lands": sorted({land["name"] for land in lands}),
        "land_counts": _land_counts(lands),
        "manabase": mana_diag,
        "cut_for_bracket": cut,
        # What the one-shot scorer could not see: a card that FINISHES a line the
        # deck already half-holds. Surfaced like `cut_for_bracket`, because a swap
        # the plan made for a reason the score does not show must be readable.
        "combo_completions": completed,
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

    # The PRINTING, not just the card. A deck built from a physical collection is a
    # list of specific objects: this art, this set, this foil. Writing bare names let
    # Scryfall pick a default for every one of them — a Sol Ring came back as Marvel
    # Super Heroes Commander — and Featured Artist credits artists per printing, so
    # the manual ended up crediting artists who painted none of the cards you own.
    printings = plan.get("printings") or {}

    def render(name):
        base = decklist_name(name, layouts.get(name, ""))
        pr = printings.get(name)
        if not pr or not pr.get("set"):
            return base
        suffix = f" ({pr['set'].upper()})"
        if pr.get("collector_number"):
            suffix += f" {pr['collector_number']}"
        if pr.get("foil"):
            suffix += " *F*"
        return base + suffix

    lines = [f"1 {render(plan['commander'])} *CMDR*"]
    for slot in sorted(plan["slots"], key=lambda s: s["name"]):
        lines.append(f"1 {render(slot['name'])}")
    for name, count in plan["land_counts"].items():
        lines.append(f"{count} {render(name)}")
    return "\n".join(lines) + "\n"


# Keys the deck-architect / deck-critic loop merges into build_plan.json. The
# deterministic builder never produces them, so a re-materialisation must carry
# them forward — the builder only rewrites what it owns. (This is the fix for
# the two-writer bug that silently erased
# hapatra's critic block: build-deck ran after the agent merge and dropped it.)
AGENT_PLAN_KEYS = (
    "archetype", "gameplan", "role_budget_citations", "swaps",
    "engines", "keep", "gaps", "critic",
)


def merge_agent_keys(plan, existing):
    """Carry agent-merged keys from an existing plan into a fresh build.

    `role_budget` is special: the deterministic builder emits a provisional
    budget, but if the existing plan carries an agent-cited one
    (`role_budget_citations` present), the cited budget + its grounding travel
    with the citations as an atomic set — citations for an overwritten budget
    would be worse than either version alone.
    """
    for key in AGENT_PLAN_KEYS:
        if key not in plan and key in existing:
            plan[key] = existing[key]
    if "role_budget_citations" in existing:
        for key in ("role_budget", "role_budget_grounding"):
            if key in existing:
                plan[key] = existing[key]
    return plan


def main(args):
    plan = build(args.slug)
    base = deck_dir(args.slug)

    out = base / "build_plan.json"
    if out.exists():
        with open(out) as f:
            plan = merge_agent_keys(plan, json.load(f))
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
        frame = load_frame()          # already parsed by build(); free here
        layouts = dict(zip(frame["name"], frame["layout"]))
        path = base / "decklist.txt"
        path.write_text(decklist_text(plan, layouts), encoding="utf-8")
        print(f"  Wrote {path}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot build-deck <slug>`.")
