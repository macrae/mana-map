"""Pilot: build a mana base from the deck's actual colour requirements.

The question a mana base answers is "how many sources of each colour do I need
to cast my spells on curve", and that is a hypergeometric problem, not a
heuristic. This module computes the answer directly:

    P(at least `pips` sources among the cards seen by turn T) >= target

so the numbers are derived arithmetic (◆ evidence, reproducible) rather than a
table quoted from memory. `manamap pilot build-deck` folds the diagnostics into
the plan so the critic can check the mana base against the spells.

Two things it fixes about the browser prototype it replaces. That one scored
lands as `colors.size * 10`, so any "add one mana of any colour" land outscored
everything and got taken first regardless of what the deck actually needed; and
it decremented every colour a land covered, so a five-colour land satisfied
five requirements at once, the loop terminated early, and the tail filled with
basics of a single colour.

Deterministic: same deck, same pool, same mana base.
"""

import math
import re

from manamap.pilot import formats
from manamap.analysis.common import WUBRG
from manamap.pilot.common import front_field, is_land

# Cards seen by turn T on the play: the opening seven plus one per turn after
# the first. Commander is singleton and games are long, but the mana base has
# to work on curve, so we size against the early turns.
OPENING_HAND = 7
SOURCE_TARGET_PROBABILITY = 0.90
# DERIVED, not written down. This was `99` — a second literal for `100 - 1`,
# and a format where that arithmetic differs (a commander that starts on the
# battlefield, a format with none at all) would have needed both changed and
# would have got one. The name stays because the maths reads better with it.
DECK_SIZE_AFTER_COMMANDER = formats.DEFAULT.library_size
MAX_CASTING_TURN = 6

# Colour requirements are planned against turn 3 at the earliest. Sizing to the
# literal earliest castable turn is arithmetically correct and practically
# useless: a single {G}{G} two-drop would demand 40 green sources, which no
# 36-land base can hold, so every deck would report a permanent shortfall.
# Commander is slow enough that turn 3 is the honest planning horizon, and the
# achieved on-curve probability is reported per colour either way.
MIN_PLANNING_TURN = 3

# Size the base against the pip weight a meaningful share of a colour's cards
# actually demand, not the single heaviest card. One {B}{B}{B} bomb in an
# otherwise single-pip deck would otherwise demand 48 sources — more than the
# whole land slot — and the greedy selector would never converge. The heaviest
# requirement is still reported so the cost of casting it is visible.
PIP_WEIGHT_QUORUM = 0.20

_PIP_RE = re.compile(r"\{([^}]+)\}")


def cards_seen(turn, on_the_play=True):
    """Cards seen by the start of `turn`."""
    draws = turn - 1 if on_the_play else turn
    return OPENING_HAND + max(0, draws)


def hypergeometric_at_least(successes_in_deck, draws, wanted, deck_size=DECK_SIZE_AFTER_COMMANDER):
    """P(at least `wanted` successes) when drawing `draws` from `deck_size`."""
    if wanted <= 0:
        return 1.0
    if successes_in_deck < wanted or draws < wanted:
        return 0.0
    draws = min(draws, deck_size)
    total = math.comb(deck_size, draws)
    misses = sum(
        math.comb(successes_in_deck, i) * math.comb(deck_size - successes_in_deck, draws - i)
        for i in range(0, wanted)
        if draws - i <= deck_size - successes_in_deck
    )
    return 1.0 - misses / total


def sources_needed(pips, turn, target=SOURCE_TARGET_PROBABILITY, on_the_play=True):
    """Fewest sources that hit `target` probability of `pips` by `turn`.

    Returns the deck size if even a full deck of sources can't get there, which
    only happens for absurd requirements and is reported rather than silently
    clamped.
    """
    if pips <= 0:
        return 0
    draws = cards_seen(turn, on_the_play)
    for k in range(pips, DECK_SIZE_AFTER_COMMANDER + 1):
        if hypergeometric_at_least(k, draws, pips) >= target:
            return k
    return DECK_SIZE_AFTER_COMMANDER


def count_pips(mana_cost):
    """Coloured pip counts for one mana cost.

    Hybrid pips count half to each side: {W/U} is genuinely castable off
    either, so charging both colours a full pip would over-build the base.
    """
    counts = {c: 0.0 for c in WUBRG}
    if not isinstance(mana_cost, str):
        return counts  # lands and NaN cells have no pips
    for symbol in _PIP_RE.findall(mana_cost):
        inner = symbol.upper()
        if inner in counts:
            counts[inner] += 1.0
            continue
        parts = [p for p in inner.split("/") if p in counts]
        if parts:
            share = 1.0 / len(parts)
            for part in parts:
                counts[part] += share
    return counts


def pip_requirements(cards):
    """Per-colour requirement across the deck's spells.

    Returns {colour: {"total_pips", "max_pips", "cards", "earliest_turn"}}.
    `max_pips` drives the source target — a base that can cast every single-pip
    spell but never {G}{G} is not a working base.
    """
    reqs = {
        c: {"total_pips": 0.0, "max_pips": 0, "cards": 0, "earliest_turn": None,
            "weights": []}
        for c in WUBRG
    }
    for card in cards:
        if is_land(card):
            continue
        # NOT card["mana_cost"]: Scryfall leaves it empty on transform/MDFC layouts
        # and puts the cost on each face, so every double-faced spell counted zero
        # pips. See `common.front_field`.
        counts = count_pips(front_field(card, "mana_cost"))
        turn = max(1, min(int(card.get("cmc") or 1), MAX_CASTING_TURN))
        for colour, pips in counts.items():
            if pips <= 0:
                continue
            entry = reqs[colour]
            entry["total_pips"] += pips
            entry["cards"] += 1
            whole = int(math.ceil(pips))
            entry["weights"].append(whole)
            if whole > entry["max_pips"]:
                entry["max_pips"] = whole
            if entry["earliest_turn"] is None or turn < entry["earliest_turn"]:
                entry["earliest_turn"] = turn

    for entry in reqs.values():
        entry["effective_pips"] = effective_pips(entry["weights"])
        del entry["weights"]
    return {c: r for c, r in reqs.items() if r["cards"]}


def effective_pips(weights):
    """Heaviest pip requirement that at least PIP_WEIGHT_QUORUM of cards share."""
    if not weights:
        return 0
    threshold = max(1, math.ceil(len(weights) * PIP_WEIGHT_QUORUM))
    for weight in sorted(set(weights), reverse=True):
        if sum(1 for w in weights if w >= weight) >= threshold:
            return weight
    return min(weights)


def source_targets(requirements):
    """Sources needed per colour, from the deck's own requirements."""
    targets = {}
    for colour, req in requirements.items():
        turn = req["earliest_turn"] or MAX_CASTING_TURN
        # Size against the heaviest requirement, no earlier than the turn a
        # spell of that weight can be cast, and never inside the planning
        # horizon — see MIN_PLANNING_TURN.
        turn = max(turn, req["effective_pips"], MIN_PLANNING_TURN)
        targets[colour] = sources_needed(req["effective_pips"], turn)
    return targets


def achieved_probability(requirements, sources):
    """Actual on-curve probability per colour for the base we built.

    A shortfall against the target is information, not a failure — this is the
    number that says how much it costs.
    """
    result = {}
    for colour, req in requirements.items():
        turn = max(req["earliest_turn"] or MAX_CASTING_TURN, req["max_pips"], MIN_PLANNING_TURN)
        result[colour] = round(
            hypergeometric_at_least(
                sources.get(colour, 0), cards_seen(turn), req["effective_pips"]), 3
        )
    return result


RESTRICTED_MANA = "spend this mana only"


def land_colors(card):
    """Which colours a land can produce *for general purposes*.

    "Add one mana of any color" is only a five-colour source when the mana is
    unrestricted. Haven of the Spirit Dragon says "Spend this mana only to cast
    a Dragon creature spell", so in a Vampire deck it taps for {C} and nothing
    else — counting it as five sources overstates a mana base badly, and it is
    the kind of card a greedy selector reaches for precisely because it looks
    like it covers everything.

    Conditional sources are deliberately counted at their *unconditional* value
    only. That understates a card like Cavern of Souls in a tribal deck, where
    the restriction is nearly free — understating a source is recoverable,
    overstating one produces a deck that cannot cast its spells.
    """
    text = str(card.get("oracle_text", "") or "")
    type_line = str(card.get("type_line", "") or "")
    produced = set()
    for basic, colour in (("Plains", "W"), ("Island", "U"), ("Swamp", "B"),
                          ("Mountain", "R"), ("Forest", "G")):
        if basic in type_line:
            produced.add(colour)
    lowered = text.lower()
    restricted = RESTRICTED_MANA in lowered
    if "any color" in lowered and not restricted:
        produced.update(WUBRG)
    # Every coloured symbol in an "add …" clause counts, not just the first:
    # "Add {R} or {W}" produces both. Clauses stop at the sentence, so a
    # restriction sentence after it never widens the count.
    for clause in re.findall(r"add[^.\n]*", lowered):
        for symbol in re.findall(r"\{([wubrg])\}", clause):
            produced.add(symbol.upper())
    if not produced and not restricted:
        produced.update(c for c in str(card.get("color_identity", "")) if c in WUBRG)
    return produced


def enters_tapped(card):
    """Does this land's text mention entering tapped AT ALL?

    A deliberate superset: the land selector uses it as a tapped BUDGET, where
    counting a conditional land against the budget is the conservative choice.
    Do not use it to measure tempo — see `enters_tapped_unconditionally`.
    """
    text = str(card.get("oracle_text", "") or "").lower()
    return "enters tapped" in text or "enters the battlefield tapped" in text


def enters_tapped_unconditionally(card):
    """Does it ALWAYS enter tapped, with no escape and no condition?

    The substring test above is a superset and reading it as a tempo cost
    overstates badly. Of edgar-vampires' sixteen matches, three are
    unconditional; three are shocks where entering tapped is a choice you make
    at two life; and ten carry an "unless" clause — three of those being
    "unless you have two or more opponents", which in Commander is ALWAYS true,
    so they are untapped duals being counted as taplands.

    `strategy:deckbuilding.mana-base` prices taplands as a real drawback
    ("around eight is comfortable in a slow deck, near zero in an aggressive
    one"), and that clearly means lands that actually enter tapped. Sixteen
    against a cap of eight is a finding; three is not.
    """
    text = str(card.get("oracle_text", "") or "").lower()
    if not enters_tapped(card):
        return False
    if "unless" in text:
        return False
    # Shocklands: "you may pay 2 life. If you don't, it enters tapped."
    if "you may pay" in text:
        return False
    return True


def land_quality(card):
    """Tiebreak among lands that close the same gap.

    Without this the selector picks alphabetically among everything that taps
    for the right colours, which is how a two-colour deck ends up running
    Abstergo Entertainment ahead of Bayou. EDHREC rank is a weak signal in
    general but a good one for lands, where popularity really does track
    quality. Unranked lands sort last.
    """
    rank = card.get("edhrec_rank")
    if rank is None or (isinstance(rank, float) and rank != rank):
        return 0.0
    return 1.0 / (1.0 + float(rank))


def select_lands(pool, targets, slots, tapped_budget=None):
    """Choose `slots` lands that best close the gap to `targets`.

    Greedy on marginal coverage: each pick is scored by how much of the
    *remaining* shortfall it closes, so a five-colour land in a two-colour deck
    scores exactly as much as the two colours it actually helps. Ties break on
    entering untapped, then on how little irrelevant colour it carries, then on
    quality, then on name for determinism.
    """
    if tapped_budget is None:
        tapped_budget = max(0, slots // 3)

    remaining = dict(targets)
    chosen = []
    tapped_used = 0
    available = sorted(pool, key=lambda c: c["name"])

    while len(chosen) < slots and available:
        best, best_score = None, None
        for card in available:
            colours = land_colors(card) & set(remaining)
            gain = sum(1 for c in colours if remaining.get(c, 0) > 0)
            if gain == 0:
                continue
            tapped = enters_tapped(card)
            if tapped and tapped_used >= tapped_budget:
                continue
            score = (
                gain,
                0 if tapped else 1,
                -len(land_colors(card) - set(targets)),
                land_quality(card),
            )
            if best_score is None or score > best_score:
                best, best_score = card, score
        if best is None:
            break
        chosen.append(best)
        available.remove(best)
        if enters_tapped(best):
            tapped_used += 1
        for colour in land_colors(best) & set(remaining):
            remaining[colour] = max(0, remaining[colour] - 1)

    return chosen, remaining, tapped_used


def fill_with_basics(chosen, basics, targets, slots, sources):
    """Top up to `slots` with basics, always for the colour furthest behind."""
    if not basics:
        return chosen
    while len(chosen) < slots:
        shortfall = {c: targets[c] - sources.get(c, 0) for c in targets}
        colour = max(shortfall, key=lambda c: (shortfall[c], c))
        basic = basics.get(colour)
        if basic is None:
            break
        chosen.append(dict(basic))
        sources[colour] = sources.get(colour, 0) + 1
    return chosen


def count_sources(lands, colours):
    counts = {c: 0 for c in colours}
    for land in lands:
        for colour in land_colors(land) & set(colours):
            counts[colour] += 1
    return counts


def build(spells, pool, slots, basics=None):
    """Build a mana base. Returns (lands, diagnostics).

    `spells` are the non-land cards, `pool` the legal nonbasic lands, `basics`
    a {colour: card} map for topping up.
    """
    requirements = pip_requirements(spells)
    targets = source_targets(requirements)

    chosen, _, tapped_used = select_lands(pool, targets, slots)
    sources = count_sources(chosen, targets)
    chosen = fill_with_basics(chosen, basics or {}, targets, slots, sources)
    sources = count_sources(chosen, targets)

    shortfalls = {c: targets[c] - sources.get(c, 0) for c in targets if sources.get(c, 0) < targets[c]}
    diagnostics = {
        "slots": slots,
        # Stamped so a later edit to the spell list is detectable: diagnostics
        # computed against a deck you no longer run are worse than none.
        "spell_slots": len(spells),
        "lands_selected": len(chosen),
        "requirements": {
            c: {"effective_pips": r["effective_pips"], "max_pips": r["max_pips"],
                "cards": r["cards"], "earliest_turn": r["earliest_turn"],
                "total_pips": round(r["total_pips"], 1)}
            for c, r in requirements.items()
        },
        "source_targets": targets,
        "sources": sources,
        "shortfalls": shortfalls,
        "on_curve_probability": achieved_probability(requirements, sources),
        "enters_tapped": tapped_used,
        "method": (
            f"hypergeometric, P(pips by earliest castable turn) >= "
            f"{SOURCE_TARGET_PROBABILITY:.0%} over {DECK_SIZE_AFTER_COMMANDER} cards"
        ),
    }
    return chosen, diagnostics
