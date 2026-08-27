"""Pilot: goldfish simulation — seeded Monte Carlo resource-development metrics.

Tier-2 (data-derived) evidence for the bench. The model simulates
resource development, NOT full games; its assumptions are stated in the output
artifact and rendered in the manual. Deterministic: same seed and deck produce
byte-identical metrics.

Model assumptions (v1):
- Multiplayer Commander: every player draws on each of their turns, turn 1 included.
- Mulligan rule: keep a 7 with 2-5 lands; otherwise redraw a fresh 7 (up to 2
  redraws), keeping the last hand regardless. No bottoming.
- One land played per turn when available.
- Persistent mana producers ("{T}: Add ...") are cast greedily after the
  commander and contribute their mana starting the following turn.
- The commander is cast on the first turn it is affordable (highest priority).
- Bodies-by-turn casts creature/token cards greedily by cost with leftover
  mana, counting the card itself (if a creature) plus tokens parsed from
  "create ... token" text. Crude by design: no interactions, no haste math.
- Cost reducers, rituals, and card draw beyond one per turn are NOT modeled;
  estimates are therefore conservative for decks that use them.
"""

import contextlib
import json
import pathlib
import random
import re

from manamap import console
from manamap.config import (
    GOLDFISH_ITERATIONS,
    GOLDFISH_MAX_MULLIGANS,
    GOLDFISH_MAX_TURN,
    GOLDFISH_MULLIGAN_MAX_LANDS,
    GOLDFISH_MULLIGAN_MIN_LANDS,
    GOLDFISH_OPPONENT_LIFE,
    GOLDFISH_SEED,
)
from manamap.pilot import manabase
from manamap.pilot.common import (
    deck_dir, deck_file, front_field, load_deck_cards)

#: THE MODEL'S OWN IDENTITY, SO STALENESS BECOMES DECIDABLE.
#:
#: Every artifact here stamped the DECK (`decklist_sha256`), the seed, the
#: iteration count and the turn limit — and nothing identified the model that
#: produced the figures. So a number computed today and one computed before a
#: model fix were indistinguishable, and when the fleet was regenerated after
#: the mana-rock and colour fixes it left **39 stale figures in authored prose
#: across four decks** with `validate-diagnosis`, `validate-strategic-frame` and
#: `validate-tutor-guide` all passing. The decklist sha had not moved, so
#: nothing could tell.
#:
#: A sha over THIS FILE's bytes. The same trick `tests/conftest.py:unchanged()`
#: and `pilot/agent_cache.py` already use, and deliberately not a hand-kept
#: integer: a version somebody has to remember to bump is one that will not be.
#: It is coarse on purpose — a comment edit bumps it, which costs a regeneration
#: nobody needed. The alternative is a curated list of "model-facing" lines,
#: which is exactly the judgement call that goes wrong silently.
def model_version():
    """First 12 hex of a sha256 over the simulator's source."""
    import hashlib
    return hashlib.sha256(
        pathlib.Path(__file__).read_bytes()).hexdigest()[:12]


MODEL_ASSUMPTIONS = [
    "Simulates resource development, not full games (no interaction, no removal).",
    "Draw every turn including turn 1 (multiplayer Commander).",
    "Mulligan: keep 7-card hands with 2-5 lands; up to 2 fresh redraws, keep the last.",
    "One land drop per turn when available.",
    "Mana rocks ('{T}: Add') contribute from the turn after they are cast.",
    "Commander cast on first affordable turn (highest spending priority).",
    "Bodies count = creatures cast + tokens parsed from 'create ... token' text.",
    "Target assembly counts cards DRAWN by a turn (cast cards still count).",
    "Unrestricted tutors ('search your library for a card') are modeled as "
    "wildcards: a tutor that has been drawn and is affordable fills ONE missing "
    "any_of group. Consumed once, mana paid, and a tutor that puts the card on "
    "top of the library costs a turn. Reported as the *_assisted figures; the "
    "unassisted figures beside them exclude tutors entirely.",
    # "CONSERVATIVE" IS TRUE OF SPEED AND FALSE OF COMPARABILITY, and the old
    # one-word claim hid the difference. Drawing one card a turn understates
    # every deck's speed, which is safe; but the size of the understatement
    # scales with how much draw a deck RUNS — heliod 16 cards, gishath 7 — so
    # the bias is not neutral ACROSS decks, and `benchmark` ranks decks.
    #
    # Modelling it was measured and refused: of 146 draw cards on this fleet,
    # **5 are unconditional**. The other 141 are triggers, activations and
    # costs — Rhystic Study needs opponents, Yawgmoth needs a sac outlet and a
    # creature. A model that saw the 5 would cover 3.4% of the axis while
    # letting this list claim draw was modelled, which is worse than the
    # refusal.
    "STATIC cost reduction IS modeled: a commander's eminence from the command "
    "zone (live from turn one, unremovable) and typed reducers once they are on "
    "the battlefield. It pays GENERIC only and is floored at the coloured pip "
    "count, because a discount can never pay a pip. A reduction that SCALES "
    "with a board state (Animar, Rakdos, Hamza) is refused rather than counted "
    "flat, and cost reduction on artifacts, noncreature spells or a colour is "
    "not modeled at all.",
    "Rituals are not modeled (conservative).",
    "Extra card draw is NOT modeled: one card per turn, always. That "
    "understates every deck, but by an amount proportional to how much draw it "
    "runs — so it is conservative WITHIN a deck and not neutral BETWEEN them. "
    "`meta.card_advantage` reports how much of each list this hides. Card "
    "advantage is measured nowhere in this suite.",
]

# Appended only for a deck that opts in. Stating "Treasures are modelled" on a
# deck where they are not is worse than saying nothing — and keeping them out
# keeps every non-opted deck's artifact byte-identical, which is the whole point
# of the flag.
TREASURE_ASSUMPTIONS = [
    "Treasures are a one-shot STOCKPILE, not a mana rock: spent only when lands "
    "and rocks fall short, and gone once broken. Reported separately from "
    "mean_available_mana_by_turn, which still means repeatable mana per turn.",
    "Only Treasure triggers a goldfish can see are modeled — upkeep, landfall "
    "and cast (recurring), Saga chapters, plus enters-the-battlefield (once). "
    "Combat- and opponent-gated sources produce NOTHING here, because this model "
    "has no combat and no opponents; they are named in "
    "meta.treasure_sources_not_modelled so a low hoard figure is legible.",
]

# Appended only for a deck that opts in, same contract as TREASURE_ASSUMPTIONS.
COMBAT_ASSUMPTIONS = [
    "COMBAT: one opponent at 40 life who does nothing — no blockers, no removal, "
    "no interaction. This is a goldfish in the literal sense, so `kill_turn` is "
    "the turn an UNOPPOSED board would finish ONE seat, not a win rate.",
    "Attacks with every creature that is not summoning-sick; haste is read from "
    "the type line. There is nothing to block, so nothing is ever held back — in "
    "a real four-player game you would keep blockers, which makes this an "
    "optimistic clock and a pessimistic board.",
    "Attack triggers, combat-damage triggers and additional combat phases are "
    "modelled, which is what makes Treasure sources gated on combat produce here "
    "when they produce nothing without this flag. Effects the parser cannot read "
    "are named in meta.combat_effects_not_modelled.",
    "Bodies count CREATURES only under this flag. Without it a Treasure token "
    "scores as a creature, which inflates `mean_bodies_by_turn` for any deck "
    "that makes non-creature tokens.",
]

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "x": 0,
}

# Token types that are NOT creatures. `_TOKEN_RE` matches "create ... token"
# generically, so without this list a Treasure scores as a body — measured on
# ur-dragon, that was 37% of the reported turn-six board.
_NONCREATURE_TOKENS = ("treasure", "clue", "food", "blood", "gold", "powerstone",
                       "map", "incubator", "junk", "shard")

_TOKEN_RE = re.compile(r"create (\w+)(?: [\w/+-]+)* tokens?", re.IGNORECASE)
# A TAP-FOR-MANA ABILITY, WRITTEN THREE WAYS, AND THE FIRST CUT SAW ONE.
# The old pattern was `\{T\}: Add ((?:\{[WUBRGC0-9]\})+)` — an explicit symbol
# list and nothing else. So `{T}: Add {C}{C}` parsed and `{T}: Add one mana of
# any color` did not, which is **Arcane Signet, Birds of Paradise, Relic of
# Legends and Sanctum Weaver**: 71 of the fleet's 110 tap-for-mana cards, 65%,
# reading zero. ur-dragon's model could see 2 of its 11 non-land mana, and
# turn-seven mana came out ~19% low on every deck measured.
#
# It is NOT a stated assumption — the module's assumption list says rituals and
# cost reducers are unmodelled and says nothing about rocks, because nobody
# knew. A silent half-working regex is the most expensive kind: it produces a
# number, the number is plausible, and it is wrong by a fifth.
#
# `[^:\n]*` lets a cost precede the tap (`{1}, {T}: Add …`) while refusing to
# cross a colon or a line, so a `{T}` in one ability cannot bind to an `: Add`
# in another. The `{T}` requirement is load-bearing and stays: Phyrexian Altar
# is `Sacrifice a creature: Add one mana`, which is not free repeatable mana and
# must not be counted as a rock.
_TAP_ADD_RE = re.compile(r"\{T\}[^:\n]*: ?Add ([^.\n]+)", re.IGNORECASE)
#: A COST THAT CONSUMES THE PERMANENT IS NOT A RATE. Widening the pattern to
#: catch "Add one mana of any color" also caught **Jeweled Lotus**, whose
#: ability reads `{T}, Sacrifice this artifact: Add three mana` — the model
#: would have collected three mana from it every turn, forever. Same for
#: Kaleidostone, Lotus Bloom and Transmogrant Altar. `produced_mana` answers
#: "per turn, repeatably", so an ability that eats its own source, exiles or
#: discards is worth zero here and belongs to a one-shot channel that does not
#: exist. A mana cost in the activation ({1}, {T}: …) is fine — filters are
#: real rocks.
_CONSUMING_COST = re.compile(r"sacrifice|exile|discard", re.IGNORECASE)
#: Mana that can only be spent on some things. See the meta note in `run`.
_RESTRICTED_MANA_RE = re.compile(r"Spend this mana only", re.IGNORECASE)
#: Any card that draws beyond the draw step. Counted, never modelled.
_DRAW_RE = re.compile(r"\bdraw (a|two|three|four|X|that many) card", re.I)
#: Written-out quantities. `X` is board-dependent (Sanctum Weaver counts
#: enchantments, Selvala reads a power), so it takes the conservative 1 — the
#: same call `treasure_profile` makes for "for each" and "equal to".
_MANA_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "x": 1}

# Deliberately the SAME pattern as ROLE_PATTERNS["tutor:unrestricted"] in
# config.py. A second definition of "what is a tutor" would let the sim and the
# role histogram disagree about the same 99, which is the class of bug this repo
# has paid for before. Narrow tutors ("search your library for a LAND card") are
# excluded on purpose — they cannot fetch a missing combo half.
_TUTOR_RE = re.compile(r"search your library for a card", re.IGNORECASE)
# Vampiric Tutor and Insatiable Avarice fetch to the TOP of the library, not to
# hand: the card arrives on the next draw, so the wildcard lands a turn later.
# The printed wording is "put that card on top." with no "of your library", so
# match the bare phrase — an "on top of" pattern silently matches neither.
_TUTOR_TO_TOP_RE = re.compile(r"\bon top\b", re.IGNORECASE)
# Spree/modal tutors ("+ {2} — Search your library for a card") charge the mode
# cost ON TOP of the card's mana value. Insatiable Avarice is cmc 1 but cannot
# tutor for less than 3, and billing it at 1 would overstate how early the
# wildcard is live.
_TUTOR_MODE_COST_RE = re.compile(r"\+\s*\{(\d+)\}[^\n]{0,4}—[^\n]{0,40}search your library for a card",
                                 re.IGNORECASE)
# Diabolic Intent's additional cost. A tutor you cannot pay for is not a wildcard.
_TUTOR_SAC_RE = re.compile(r"as an additional cost.{0,40}sacrifice a creature",
                           re.IGNORECASE | re.DOTALL)


# ── Treasures ─────────────────────────────────────────────────────────────
#
# A Treasure is NOT a mana rock and modelling it as one is the whole trap: a
# rock produces every turn forever, a Treasure produces once and is gone. The
# stockpile below is spent only when lands and rocks come up short, which is
# both how it is played and what makes a hoard-counting payoff measurable.
#
# **Only triggers this simulation can honestly see are modelled.** There is no
# combat here and there are no opponents, so "whenever this creature deals
# combat damage to a player" (Old Gnawbone, Cavern-Hoard Dragon) and "whenever
# an opponent draws a card" (Smothering Tithe) produce NOTHING — and that is a
# finding rather than a shortcoming. Measured across the fleet, 16 of the 19
# Treasure sources in the nine decks are combat- or opponent-gated; a naive
# "create a Treasure token" match would hand eight decks free mana they never
# get, turning a deliberately conservative model optimistic. Unmodelled sources
# are NAMED in the output so a low number is legible instead of mysterious.
_TREASURE_RE = re.compile(r"creates?\s+(?:[\w\-]+\s+)*?Treasure tokens?", re.IGNORECASE)
_TREASURE_N_RE = re.compile(
    r"creates?\s+(a|an|one|two|three|four|five|X|\d+)\s+(?:[\w\-]+\s+)*?Treasure",
    re.IGNORECASE)
# Recurring, and free at the point of use.
_TRE_UPKEEP_RE   = re.compile(r"at the beginning of your upkeep", re.IGNORECASE)
_TRE_LANDFALL_RE = re.compile(r"whenever a land you control enters|landfall", re.IGNORECASE)
_TRE_CAST_RE     = re.compile(r"whenever you cast", re.IGNORECASE)
# A Saga adds a lore counter "after your draw step" every turn, so a Saga whose
# chapters make Treasures IS a recurring engine — The Misty Mountains Cold makes
# one on each of four chapters. Modelled as recurring and NOT sacrificed at IV:
# the chapter payout is the mana question, and the 6/6 Dragon it converts into is
# a body the bodies series would need to know about. Slightly generous after turn
# four on that one card, and stated here rather than hidden.
_TRE_SAGA_RE     = re.compile(r"Enchantment — Saga|add a lore counter", re.IGNORECASE)
# One-shot, on resolution.
_TRE_ETB_RE      = re.compile(r"when (?:this creature|this artifact|[A-Z][\w' ,-]{2,30}) enters",
                              re.IGNORECASE)
# TWO KINDS OF MULTIPLIER, AND CONFLATING THEM IS WRONG IN BOTH DIRECTIONS.
# Xorn and Jolene ADD one Treasure to every Treasure event; Anointed Procession,
# Parallel Lives, Doubling Season and Mondrak DOUBLE whatever the event makes.
# They coincide only when the event makes exactly one, which is why an additive
# stand-in for doubling reads almost right and is not.
#
# These are PUBLIC because `assess._MULTIPLIER` is the other reader of the same
# concept and the two had diverged silently: this module matched one wording and
# assess matched five, so the goldfish priced 2 of the 8 multipliers ur-dragon's
# treasure branch DECLARES and counted the other 6 as drawn-and-inert. That is
# the `front_field` defect one subsystem over — two halves of one idea drifting
# because nothing made them share a definition. They live here rather than in
# config because this module owns the Treasure model; config owns the frozen,
# model-facing vocabulary and adding to it invalidates a trained net.
TREASURE_BONUS_RE = re.compile(r"instead create those tokens plus an additional Treasure",
                               re.IGNORECASE)
#: "it creates twice that many of those tokens instead" and Mondrak's inversion
#: of the same sentence. Deliberately keyed on TOKENS — Panharmonicon doubles
#: ETB TRIGGERS and Academy Manufactor converts Clue/Food events into Treasure
#: ones; both are real multipliers for a deck and neither is this one, so they
#: stay blind and get NAMED rather than folded in where they would read as right.
TOKEN_DOUBLER_RE = re.compile(
    r"creates twice that many of those tokens|twice that many of those tokens are created|(?:twice|three times) that many (?:of those )?(?:creature )?tokens are created",
    re.IGNORECASE)
#: Kept so the old private name still resolves for anything reading it.
_TRE_EXTRA_RE = TREASURE_BONUS_RE


def treasure_profile(card):
    """How this card makes Treasures, and whether a goldfish can see it.

    Returns `(per_event, trigger)` where trigger is one of `upkeep`,
    `landfall`, `cast` (recurring), `etb` (once), or `unmodelled`.
    A card with no Treasure text returns `(0, None)`.
    """
    text = card.get("oracle_text") or ""
    if not _TREASURE_RE.search(text):
        return 0, None
    match = _TREASURE_N_RE.search(text)
    word = (match.group(1).lower() if match else "a")
    # "X Treasures" is opponent- or board-dependent every time it appears in
    # this corpus, so it is counted as one rather than guessed at.
    count = int(word) if word.isdigit() else _NUMBER_WORDS.get(word, 1) or 1
    saga = _TRE_SAGA_RE.search(card.get("type_line", "") or "") or _TRE_SAGA_RE.search(text)
    if saga:
        return count, "upkeep"
    for trigger, pattern in (("upkeep", _TRE_UPKEEP_RE),
                             ("landfall", _TRE_LANDFALL_RE),
                             ("cast", _TRE_CAST_RE),
                             ("etb", _TRE_ETB_RE)):
        if pattern.search(text):
            return count, trigger
    return count, "unmodelled"


# ── Combat ────────────────────────────────────────────────────────────────
#
# OPT-IN, for exactly the reason `model_treasures` is: switching combat on
# changes `mean_bodies_by_turn` for every deck that makes non-creature tokens
# (all nine of them), and those figures are quoted in published prose on five
# decks and in one `engine.json` carrying a critic verdict. A deck opts in when
# it is next re-baselined deliberately.
#
# The discipline is the same as the Treasure model's: model only what can be
# read honestly, and NAME what cannot. What this buys is the class of card the
# resource model priced at exactly zero — attack triggers (Savage Ventmaw's
# {R}{R}{R}{G}{G}{G}, Old Gnawbone's Treasures, Smaug's ping), additional combat
# phases (Scourge of the Throne, Aggravated Assault), and therefore the
# combat-gated Treasure sources that `treasure_profile` returns `unmodelled` for.
# On ur-dragon that was nine of fourteen sources and both halves of the deck's
# only verified win line.

_HASTE_RE = re.compile(r"\bhaste\b", re.IGNORECASE)
# "create a 1/1 red Dragon creature token", "create two 2/2 ... tokens"
_TOKEN_PT_RE = re.compile(r"create (\w+) ([\dX]+)/([\dX]+)([^.]*?)tokens?", re.IGNORECASE)
_ATTACKS_RE = re.compile(
    r"whenever you attack\b|whenever (?:this creature|[A-Z][\w' ,-]{2,30}|one or more [\w ]+ you control) attacks",
    re.IGNORECASE)
_COMBAT_DMG_RE = re.compile(
    r"whenever (?:this creature|[A-Z][\w' ,-]{2,30}) deals combat damage to a player",
    re.IGNORECASE)
_EXTRA_COMBAT_RE = re.compile(r"additional combat phase", re.IGNORECASE)
# An ACTIVATED extra combat (Aggravated Assault's {3}{R}{R}), as opposed to a
# triggered one (Scourge of the Throne). The cost decides whether it is a free
# repeat button or one you have to buy every turn.
# Sentence-crossing on PURPOSE. Aggravated Assault reads "{3}{R}{R}: Untap all
# creatures you control. After this main phase, there is an additional combat
# phase" — a `[^.]` bound stops at that period, the cost never binds, and the
# deck's only verified win line silently becomes unmodelled. Caught by test.
_ACTIVATED_COMBAT_RE = re.compile(
    r"((?:\{[WUBRGC0-9]\})+)\s*:.{0,160}?additional combat phase",
    re.IGNORECASE | re.DOTALL)
_DMG_EQUAL_TREASURE_RE = re.compile(
    r"deals damage equal to the number of Treasures", re.IGNORECASE)


def _mana_pips(cost_string):
    """How much generic-equivalent mana a '{3}{R}{R}' style string costs."""
    total = 0
    for sym in re.findall(r"\{([WUBRGC0-9])\}", cost_string or ""):
        total += int(sym) if sym.isdigit() else 1
    return total


def _stat(value):
    """Power/toughness as an int; '*' and None become 0 (conservative)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def creature_body_count(card):
    """Bodies counting CREATURES only — a Treasure token is not a blocker.

    This is `body_count` with the non-creature tokens removed. It is reached
    only under `model_combat` so that a deck which has not opted in keeps the
    number it published.
    """
    text = card.get("oracle_text", "") or ""
    bodies = 1 if "Creature" in card.get("type_line", "") else 0
    for match in _TOKEN_RE.finditer(text):
        clause = text[match.start():match.end() + 60].lower()
        if any(k in clause for k in _NONCREATURE_TOKENS):
            continue
        word = match.group(1).lower()
        bodies += int(word) if word.isdigit() else _NUMBER_WORDS.get(word, 1)
    return bodies


def combat_profile(card):
    """What this card does once there is a combat step.

    Returns a dict the simulation reads directly. `unreadable` is set when the
    card clearly has a combat trigger whose EFFECT the parser cannot price —
    those are surfaced in the metrics rather than silently scoring zero.
    """
    text = card.get("oracle_text", "") or ""
    type_line = card.get("type_line", "") or ""
    is_creature = "Creature" in type_line

    profile = {
        "is_creature": is_creature,
        "power": _stat(card.get("power")) if is_creature else 0,
        "haste": bool(_HASTE_RE.search(text)),
        "token_power": 0,
        "token_bodies": 0,
        "attack_mana": 0,
        "attack_treasure": 0,
        "attack_draw": 0,
        "attack_damage": 0,
        "attack_token_power": 0,
        "attack_token_bodies": 0,
        "damage_scales_with_treasure": False,
        "extra_combat_free": False,
        "extra_combat_cost": None,
        "unreadable": None,
    }

    # Creature tokens this card makes, with their power.
    for match in _TOKEN_PT_RE.finditer(text):
        tail = (match.group(4) or "").lower()
        if any(k in tail for k in _NONCREATURE_TOKENS):
            continue
        word = match.group(1).lower()
        count = int(word) if word.isdigit() else _NUMBER_WORDS.get(word, 1)
        profile["token_bodies"] += count
        profile["token_power"] += count * _stat(match.group(2))

    combat_trigger = _ATTACKS_RE.search(text) or _COMBAT_DMG_RE.search(text)
    if combat_trigger:
        window = text[combat_trigger.start():combat_trigger.start() + 220]
        # `_TAP_ADD_RE` now captures the whole clause rather than a symbol run,
        # so route it through the one parser instead of counting pips here —
        # two readers of one pattern is the divergence this file has paid for.
        got = produced_mana(window)
        if not got:
            plain = re.search(r"add ((?:\{[WUBRGC0-9]\})+)", window, re.IGNORECASE)
            got = _mana_pips(plain.group(1)) if plain else 0
        if got:
            profile["attack_mana"] = got
        if re.search(r"treasure token", window, re.IGNORECASE):
            n = _TREASURE_N_RE.search(window)
            word = (n.group(1).lower() if n else "a")
            # "for each" / "equal to" counts are board-dependent; one is the
            # conservative read, matching what `treasure_profile` does.
            profile["attack_treasure"] = int(word) if word.isdigit() else \
                _NUMBER_WORDS.get(word, 1) or 1
        drawn = re.search(r"draw (\w+) cards?", window, re.IGNORECASE)
        if drawn:
            word = drawn.group(1).lower()
            profile["attack_draw"] = int(word) if word.isdigit() else \
                _NUMBER_WORDS.get(word, 1)
        if _DMG_EQUAL_TREASURE_RE.search(window):
            profile["damage_scales_with_treasure"] = True
        # Direct damage on attack (Drakuseth). Only the FIRST "deals N damage"
        # is counted: the follow-on clauses ("and 3 damage to each of up to two
        # other targets") usually point at creatures, and this model has none to
        # point at, so crediting them to the opponent's face would invent reach.
        fixed = re.search(r"deals (\d+) damage", window, re.IGNORECASE)
        if fixed:
            profile["attack_damage"] = int(fixed.group(1))
        # Creature tokens made on attack (Utvara Hellkite). Counted ONCE per
        # combat even where the trigger is per-attacker, for the same reason.
        for tok in _TOKEN_PT_RE.finditer(window):
            if any(k in (tok.group(4) or "").lower() for k in _NONCREATURE_TOKENS):
                continue
            word = tok.group(1).lower()
            count = int(word) if word.isdigit() else _NUMBER_WORDS.get(word, 1)
            profile["attack_token_bodies"] += count
            profile["attack_token_power"] += count * _stat(tok.group(2))
        if not any((profile["attack_mana"], profile["attack_treasure"],
                    profile["attack_draw"], profile["damage_scales_with_treasure"],
                    profile["attack_damage"], profile["attack_token_bodies"],
                    # Checked against the FULL text, not the window: Scourge of
                    # the Throne's reminder clause pushes "additional combat
                    # phase" past 220 characters, and flagging a card whose
                    # effect IS modelled makes the not-modelled list a liar.
                    _EXTRA_COMBAT_RE.search(text))):
            profile["unreadable"] = card.get("name")

    if _EXTRA_COMBAT_RE.search(text):
        activated = _ACTIVATED_COMBAT_RE.search(text)
        if activated:
            profile["extra_combat_cost"] = _mana_pips(activated.group(1))
        elif combat_trigger:
            profile["extra_combat_free"] = True
        else:
            # AN EXTRA COMBAT THIS MODEL CANNOT PLACE, AND IT WAS SILENT.
            # Neither activated (no mana cost binds) nor triggered on an attack:
            # a one-shot spell ("After this main phase, there is an additional
            # combat phase"), or a permanent keyed on being BLOCKED, on exert,
            # on landfall, or on a loyalty ability. The model has no channel for
            # any of those, which is a boundary rather than a bug — but it fell
            # through both branches and set nothing, so the card contributed
            # nothing to the clock AND appeared in no not-modelled list.
            #
            # Corpus-wide that is 32 cards; on this fleet it is ONE
            # (goblin-storm's Great Train Heist). Naming it is what keeps a low
            # kill figure legible, the same contract
            # `treasure_sources_not_modelled` keeps.
            profile["unreadable"] = card.get("name")

    return profile


#: THE MODEL WAS COLOURLESS, AND `mana_analysis` HAS ALWAYS SAID IT MATTERED.
#: `spend()` took a scalar, so a five-colour Ur-Dragon and a mono-green Radagast
#: with the same land count had identical curves. Measured against the closed
#: form one module over, the BINDING colour is available on curve 56%-99% of the
#: time across the fleet (median 82%; ur-dragon's black is 56%) — and the
#: simulation was casting at 100%. That is the largest single accuracy gap in
#: the resource model, and it ran OPPOSITE to the rock blindness above, so the
#: two partly cancelled and the total stayed plausible while both halves were
#: wrong.
#: A PIP, FOR CASTING. `manabase.count_pips` answers a different question and
#: answers it correctly: it half-charges a hybrid to each side, which is right
#: for SIZING a base (a {W/U} spell really is castable off either, so charging
#: both a full pip over-builds). For CASTABILITY a hybrid is one pip payable two
#: ways, and half a pip is not a thing you can pay. Same split as `bodies` vs
#: `creature_bodies` two functions down — one concept, two questions, and
#: forcing them into one reader is how this file has been bitten before.
_CAST_PIP_RE = re.compile(r"\{([^}]+)\}")


def cast_pips(mana_cost):
    """One entry per coloured pip: the set of colours that can pay it."""
    out = []
    for symbol in _CAST_PIP_RE.findall(mana_cost or ""):
        inner = symbol.upper()
        # {2/W} is payable with two generic OR one white; a goldfish that has
        # the mana always has the two, so it never constrains a colour.
        if any(ch.isdigit() for ch in inner):
            continue
        colours = frozenset(c for c in inner.split("/") if c in "WUBRG")
        if colours:
            out.append(colours)
    return out


def can_pay(pips, sources, wildcards=0):
    """Can these coloured pips be paid from these sources?

    `pips` is a list of colour-sets (one per pip), `sources` one colour-set per
    untapped producer, `wildcards` the Treasures, which make any colour.
    GREEDY, MOST-CONSTRAINED PIP FIRST, and each pip takes the source with the
    FEWEST colours that can pay it — the standard assignment heuristic, exact at
    these sizes (a Commander cost is a handful of pips against a dozen sources).
    """
    if not pips:
        return True
    used = [False] * len(sources)
    def supply(pip):
        return sum(1 for c in sources if c & pip)
    for pip in sorted(pips, key=supply):
        best = -1
        for i, c in enumerate(sources):
            if used[i] or not (c & pip):
                continue
            if best < 0 or len(c) < len(sources[best]):
                best = i
        if best >= 0:
            used[best] = True
        elif wildcards > 0:
            wildcards -= 1
        else:
            return False
    return True


def produced_mana(oracle_text):
    """Mana a persistent '{T}: Add ...' producer yields per turn (0 if none)."""
    match = _TAP_ADD_RE.search(oracle_text or "")
    if not match:
        return 0
    # The activation cost is everything from `{T}` back to the clause start.
    cost = (oracle_text or "")[max(0, match.start() - 40):match.start()
                               + match.group(0).index(":")]
    if _CONSUMING_COST.search(cost):
        return 0
    body = match.group(1)
    # ALTERNATIVES ARE A CHOICE, NOT A SUM. `Add {R}, {G}, or {W}` is ONE mana
    # and counting the symbols gives three; `Add {U} or {C}{U}` is two, not
    # three. So take the LARGEST CONSECUTIVE RUN rather than the total — which
    # is also exactly what the old narrow pattern did by accident, since
    # `(?:\{..\})+` stopped at the first comma. Widening the match without
    # this reintroduced the bug as an overcount on every dual-choice rock.
    runs = [len(re.findall(r"\{[WUBRGC0-9]\}", r))
            for r in re.findall(r"(?:\{[WUBRGC0-9]\})+", body)]
    if runs:
        return max(runs)
    word = re.match(r"\s*(one|two|three|four|five|X)\b", body, re.IGNORECASE)
    return _MANA_WORDS[word.group(1).lower()] if word else 0


def body_count(card):
    """Bodies this card contributes when cast: itself (if creature) + tokens."""
    bodies = 1 if "Creature" in card.get("type_line", "") else 0
    for word in _TOKEN_RE.findall(card.get("oracle_text", "") or ""):
        bodies += _NUMBER_WORDS.get(word.lower(), 1 if not word.isdigit() else int(word))
    return bodies


#: STATIC COST REDUCTION — `<Type> spells you cast cost {N} less to cast`.
#:
#: THE MODEL SAID "cost reducers are not modeled (conservative)" AND FOR THIS
#: FLEET THAT IS NOT CONSERVATIVE, IT IS WRONG ABOUT THE THESIS. The Ur-Dragon's
#: eminence takes {1} off every Dragon spell from the COMMAND ZONE — always on,
#: from turn one, unremovable — which is 22 of its 24 creatures and takes the
#: mean Dragon from 5.73 to 4.73. Four more reducers sit in the 99 and read as
#: vanilla bodies. A mv5 Dragon with eminence and one Dragonlord's Servant costs
#: 3 in paper and 5 in the model, so every figure about when a threat lands was
#: measured in a world where the commander's ability does not exist.
#:
#: Same class as the two already in the record: the model could not see 65% of
#: the fleet's mana rocks, and the mana model was colourless.
#:
#: The subtype capture is CASE-SENSITIVE and deliberate: `Dragon` is a creature
#: type and `dragon` in flavour prose is not.
_COST_REDUCTION_RE = re.compile(
    r"(?P<other>other )?(?P<what>[A-Z][a-z]+) spells"
    r"(?: you cast)?(?P<chosen> of the chosen type)? cost "
    r"\{(?P<amount>\d)\} less to cast")

#: What a reduction applies to when the card says "the chosen type". A real
#: player names the type they built around, so the model resolves it to the
#: deck's most common creature subtype — stated, because it is a choice the
#: model is making on the pilot's behalf.
CHOSEN_TYPE = "\x00chosen"


#: A REDUCTION THAT SCALES IS NOT A RATE, and the corpus sweep is what caught it.
#: `Creature spells you cast cost {1} less to cast FOR EACH …` — Hamza counts
#: +1/+1 counters, Animar counts counters on itself and starts at zero, and
#: Rakdos counts life the opponents lost this turn, which in a solitaire model
#: is zero forever. The regex stops at "to cast" and would have reported a flat
#: 1 for all three: a plausible number that is wrong, which is the Jeweled Lotus
#: failure exactly. Refused, and named as unmodelled rather than counted.
_SCALING_REDUCTION_RE = re.compile(r"\A for each\b")

#: WHAT A REDUCTION CAN APPLY TO, checked against the corpus's own creature
#: types rather than a word list. The sweep found the regex capturing
#: `Noncreature` (7 cards), `Artifact` (7), `Equipment` (5), `Enchantment` (4)
#: and six colour words (14) — none of them creature subtypes, all of them
#: silently matching nothing once tested against a card's subtypes. A silent
#: half-working matcher is the most expensive kind, so these are REFUSED here
#: instead of quietly reducing nothing. Artifact and noncreature reduction is
#: real and simply not modelled; saying so is the difference.
_ALL_CREATURES = ("Creature",)


def cost_reduction(card, creature_types=None):
    """`(amount, applies_to, excludes_self)` for a static reducer, else None.

    `applies_to` is a creature subtype, `CHOSEN_TYPE`, or None for every
    creature spell. GENERIC ONLY — a reduction can never pay a coloured pip,
    which is why the caller floors the result at the pip count rather than at
    zero.
    """
    text = card.get("oracle_text") or ""
    got = _COST_REDUCTION_RE.search(text)
    if not got:
        return None
    if _SCALING_REDUCTION_RE.match(text[got.end():]):
        return None
    what = got.group("what")
    if got.group("chosen"):
        applies = CHOSEN_TYPE
    elif what in _ALL_CREATURES:
        applies = None
    elif creature_types is not None and what not in creature_types:
        return None
    elif creature_types is None and what in _NOT_A_CREATURE_TYPE:
        return None
    else:
        applies = what
    return (int(got.group("amount")), applies, bool(got.group("other")))


def _corpus_creature_types():
    """The corpus's own creature types, memoised.

    `analysis.common.creature_types` is the one scan, shared with `assess` and
    `power_creep`, so the triage that WARNS a pilot and the model that PRICES
    their deck cannot drift on what a tribe is. Returns None when there is no
    corpus, and `cost_reduction` then falls back to its own refusal list — a
    unit test with no data behind it still rejects the right words.
    """
    global _CREATURE_TYPES_CACHE
    if _CREATURE_TYPES_CACHE is _UNSET:
        try:
            from manamap.analysis import common as _acommon
            from manamap.pilot import card_pool
            _CREATURE_TYPES_CACHE = _acommon.creature_types(card_pool.load_frame())
        except Exception:                      # pragma: no cover - defensive
            _CREATURE_TYPES_CACHE = None
    return _CREATURE_TYPES_CACHE or None


_UNSET = object()
_CREATURE_TYPES_CACHE = _UNSET


def reduced_cost(card, reductions, chosen=None):
    """What this spell costs with these static reducers in play.

    GENERIC ONLY, floored at the coloured pip count. A cost reduction can never
    pay a coloured pip — `{4}{W}{U}{B}{R}{G}` with three reducers out is still
    five mana, not two — and flooring at zero instead would have made a
    five-colour commander look castable off two lands.
    """
    total = 0
    for amount, applies, excludes_self in reductions:
        # `is_commander` is NOT a key `classify` emits — the commander is never
        # in the library — so it stays a `.get`. Everything else is subscripted,
        # because `test_every_signal_the_model_sets_is_read_by_something` counts
        # a subscript as the proof a flag is acted on and a `.get` would let a
        # key look read while nothing used it.
        if excludes_self and card.get("is_commander"):
            continue
        if applies is None:
            if card["is_creature"]:
                total += amount
        elif applies is CHOSEN_TYPE:
            if chosen and chosen in card["subtypes"]:
                total += amount
        elif applies in card["subtypes"]:
            total += amount
    return max(len(card["pips"]), int(card["cmc"]) - total)


#: The fallback when no corpus is loaded — the classes the sweep actually found,
#: so a unit test with no corpus behind it still refuses the right words.
_NOT_A_CREATURE_TYPE = frozenset({
    "Noncreature", "Artifact", "Equipment", "Enchantment", "Aura", "Vehicle",
    "Lair", "Lesson", "Arcane", "Historic", "Legendary", "Commander",
    "Planeswalker", "Multicolored", "Colorless",
    "White", "Blue", "Black", "Red", "Green",
})


def subtypes_of(type_line):
    """The subtypes after the em dash. `Legendary Creature — Dragon` -> {Dragon}."""
    if "\u2014" not in (type_line or ""):
        return frozenset()
    tail = type_line.split("\u2014", 1)[1].split("//")[0]
    return frozenset(w for w in tail.split() if w[:1].isupper())


def chosen_type_for(cards):
    """The creature subtype a player would name. Most common wins; ties by name.

    Deterministic, because everything in this model must be: a tie broken by
    dict order would make two identical decks measure differently.
    """
    counts = {}
    for c in cards:
        if "Creature" not in (c.get("type_line") or ""):
            continue
        for t in subtypes_of(c.get("type_line")):
            counts[t] = counts.get(t, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def classify(card):
    """Return a compact sim-card dict for one physical copy."""
    type_line = card.get("type_line", "")
    text = card.get("oracle_text") or ""
    is_land = "Land" in type_line and "Creature" not in type_line.split("//")[0]
    is_tutor = bool(not is_land and _TUTOR_RE.search(text))
    mode_cost = _TUTOR_MODE_COST_RE.search(text) if is_tutor else None
    return {
        "name": card["name"],
        "is_land": is_land,
        "cmc": int(card.get("cmc") or 0),
        # What it actually costs to USE the tutor mode, which is what decides
        # when the wildcard comes online.
        "tutor_cmc": int(card.get("cmc") or 0) + (int(mode_cost.group(1)) if mode_cost else 0),
        "produces": 0 if "Land" in type_line else produced_mana(card.get("oracle_text")),
        "bodies": 0 if "Land" in type_line else body_count(card),
        # Creature-only body count and the combat profile ride along always;
        # they are READ only under `model_combat`, so a non-opted deck is
        # byte-identical and this stays a pure widening of the sim card.
        "creature_bodies": 0 if "Land" in type_line else creature_body_count(card),
        "combat": combat_profile(card),
        "tutor": is_tutor,
        # A top-of-library tutor delivers on the next draw step, not this turn.
        "tutor_delay": 1 if is_tutor and _TUTOR_TO_TOP_RE.search(text) else 0,
        "tutor_needs_body": bool(is_tutor and _TUTOR_SAC_RE.search(text)),
        "treasure_n": 0 if is_land else treasure_profile(card)[0],
        "treasure_trigger": None if is_land else treasure_profile(card)[1],
        # Xorn makes no Treasure of its own; it adds one to every event.
        # WHAT IT PRODUCES and WHAT IT COSTS, in colours. Both ride along
        # always and are READ only under `model_colors`, so the colourless path
        # stays byte-identical — the `creature_bodies` rule. `land_colors` is
        # `manabase`'s and is deliberately restriction-aware: Haven of the
        # Spirit Dragon taps for {C} in a Vampire deck, and counting it as five
        # sources is how a mana base comes out looking fine and cannot cast its
        # spells.
        "colors": frozenset(manabase.land_colors(card)),
        "pips": cast_pips(front_field(card, "mana_cost") or ""),
        "treasure_bonus": bool(TREASURE_BONUS_RE.search(text)),
        # Anointed Procession et al. make none either, and DOUBLE every event.
        # Rides on the card always and is read only under `model_treasures`, so
        # a non-opted deck stays byte-identical — the `creature_bodies` rule.
        "treasure_doubler": bool(TOKEN_DOUBLER_RE.search(text)),
        # Static cost reduction, and what this card IS so a reduction can be
        # tested against it. Both ride along always and are read only when a
        # reducer is actually in play, so a deck with none stays byte-identical.
        "reduces": cost_reduction(card, _corpus_creature_types()),
        "subtypes": subtypes_of(type_line),
        "is_creature": "Creature" in type_line,
    }


def build_library(doc):
    """Expand the main deck (minus commanders) into per-copy sim cards."""
    library = []
    commanders = []
    for card in doc["cards"]:
        if card.get("is_commander"):
            commanders.append(card)
            continue
        library.extend([classify(card)] * card.get("quantity", 1))
    return library, commanders


def keepable(hand):
    lands = sum(1 for c in hand if c["is_land"])
    return GOLDFISH_MULLIGAN_MIN_LANDS <= lands <= GOLDFISH_MULLIGAN_MAX_LANDS


def _target_met(target, names_in_hand, commander_cast, tutors=0):
    """Is this target assembled, allowing `tutors` wildcards to fill holes?

    `tutors` is applied per target independently — each target is a separate
    counterfactual ("could this have been assembled by now"), exactly as the
    unassisted metric already treats them. It is NOT a shared pool drained
    across targets, which would make one target's rate depend on the order the
    others happen to be listed in.
    """
    if target.get("commander") and not commander_cast:
        return False
    unmet = sum(1 for need in target["need"]
                if not any(name in names_in_hand for name in need["any_of"]))
    return unmet <= tutors


def simulate_once(rng, library, commander_cmc, targets, max_turn,
                  model_treasures=False, model_combat=False,
                  model_colors=False, commander_pips=None,
                  command_zone_reduction=(), chosen_type=None,
                  commander_subtypes=frozenset()):
    """One goldfish iteration. Returns a per-iteration result dict."""
    deck = library[:]
    rng.shuffle(deck)

    hand = deck[:7]
    deck = deck[7:]
    # Captured BEFORE the mulligan loop rebinds `hand`. The two populations
    # answer different questions and must not be conflated: the first seven is
    # what the keep rule is applied to, the kept hand is what you actually play.
    # Reporting the kept hand as "opening" made the distribution nearly
    # invariant to deck composition — every deck looks ~99% healthy at 2-5
    # lands, because that is the keep rule restating itself.
    first_seven_lands = sum(1 for c in hand if c["is_land"])

    mulligans = 0
    while not keepable(hand) and mulligans < GOLDFISH_MAX_MULLIGANS:
        mulligans += 1
        deck = library[:]
        rng.shuffle(deck)
        hand = deck[:7]
        deck = deck[7:]

    kept_hand_lands = sum(1 for c in hand if c["is_land"])
    seen = {c["name"] for c in hand}

    lands_in_play = 0
    rock_production = 0
    # EMINENCE IS ON FROM TURN ONE. It works from the command zone, so it is
    # live before the commander is ever cast and cannot be removed — which is
    # exactly why leaving it unmodelled mispriced the whole deck rather than
    # just its late game. Reducers cast from hand are appended as they land.
    reductions = list(command_zone_reduction)
    commander_card = {"is_commander": True, "is_creature": True,
                      "subtypes": commander_subtypes, "cmc": commander_cmc,
                      "pips": commander_pips or ()}
    # One colour-set per untapped producer, parallel to the two counts above.
    # Only maintained under `model_colors`; empty otherwise, so `can_pay` is
    # never reached and the colourless arithmetic is untouched.
    sources = []
    treasures = 0                 # a STOCKPILE: each one is spendable once
    treasure_engines = []         # (per_event, trigger) for modelled sources in play
    treasure_bonus = 0            # Xorn-style +N per creation event
    # Procession-style xN. Multiplicative and applied AFTER the additive bonus,
    # which is the order a player would choose: replacement effects on one event
    # are ordered by the affected player, and (n + 1) x 2 beats n x 2 + 1. A
    # goldfish assumes the pilot takes the better line.
    treasure_multiplier = 1
    treasures_by_turn = []
    treasure_online_by_turn = []
    commander_turn = None
    land_hits = []
    # STALL: a turn on which NOTHING IN HAND COULD BE CAST AT ALL.
    #
    # THE OBVIOUS DEFINITION IS WRONG HERE AND WRONG BY A LOT. "A turn on which
    # nothing was cast" measures the MODEL, not the deck: this is a resource
    # simulation, so it casts rocks, tutors, extra-combat permanents and bodies
    # and never casts a wipe, a counterspell or a targeted removal spell. Scored
    # that way ur-dragon shows 6.4 dead turns in 10 while its hand grows to
    # eleven cards — which is a description of what the model declines to
    # represent, not of the deck stalling.
    #
    # So the question asked is CASTABILITY: with the mana this turn produced,
    # was there any nonland card in hand you could legally have cast? That needs
    # only mana value and available mana, so it is true of cards the model would
    # never pick up, and it is the honest reading of the PRD's "no legal play".
    #
    # A LAND DROP IS NOT A PLAY. A turn spent playing a land and casting nothing
    # is exactly the turn this measures.
    stall_by_turn = []          # True on a turn with nothing castable
    hand_size_by_turn = []      # a stall with cards left is a mana problem;
                                # a stall with an empty hand is a draw problem
    mana_by_turn = []
    bodies_cum = 0
    bodies_by_turn = []
    # Combat state. `battlefield` holds one entry per creature that can attack:
    # (power, turn_it_arrived, has_haste). Kept only under model_combat so the
    # resource-only path allocates nothing extra.
    battlefield = []
    combat_engines = []           # per-attack triggers of creatures in play
    extra_combat_free = 0         # Scourge-style, one additional phase
    extra_combat_costs = []       # Aggravated Assault-style, buy each time
    opponent_life = GOLDFISH_OPPONENT_LIFE
    kill_turn = None
    damage_by_turn = []
    board_power_by_turn = []
    target_turns = [None] * len(targets)
    target_turns_unassisted = [None] * len(targets)
    tutor_ready_turns = []

    for turn in range(1, max_turn + 1):
        if deck:
            drawn = deck.pop(0)
            hand.append(drawn)
            seen.add(drawn["name"])

        land_index = next((i for i, c in enumerate(hand) if c["is_land"]), None)
        if land_index is not None:
            played = hand.pop(land_index)
            lands_in_play += 1
            if model_colors:
                sources.append(played["colors"])
            land_hits.append(True)
        else:
            land_hits.append(False)

        # Recurring Treasure engines already in play fire before you spend.
        # `landfall` only pays out on a turn a land actually entered, which is
        # what makes Tireless Provisioner worth less than an upkeep trigger.
        for per_event, trigger in treasure_engines:
            if trigger == "landfall" and not land_hits[-1]:
                continue
            if trigger in ("upkeep", "landfall"):
                treasures += (per_event + treasure_bonus) * treasure_multiplier

        pool = lands_in_play + rock_production
        # Reported WITHOUT the stockpile, so this series keeps meaning exactly
        # what it has always meant: repeatable mana per turn. Treasures are a
        # one-shot reserve and get their own series.
        mana_by_turn.append(pool)
        treasures_by_turn.append(treasures)
        treasure_online_by_turn.append(bool(treasure_engines))

        def spend(cost, pips=None):
            """Pay from lands and rocks first, then break Treasures.

            Under `model_colors` the colour requirement is checked too, against
            the sources actually in play, with Treasures as wildcards. Refusing
            here is the whole point: it is the turn the deck has the mana and
            not the colour, which every figure in this model used to ignore.
            """
            nonlocal pool, treasures
            if pool + treasures < cost:
                return False
            if model_colors and pips and not can_pay(pips, sources, treasures):
                return False
            if cost <= pool:
                pool -= cost
            else:
                treasures -= cost - pool
                pool = 0
            return True

        # A REDUCER ON THE BATTLEFIELD DOES CUT THE COMMANDER'S COST. Eminence
        # says "OTHER Dragon spells", so it never pays for itself — but
        # Dragonlord's Servant takes {1} off The Ur-Dragon like any other Dragon
        # spell, and a nine-drop commander is exactly where that matters.
        if commander_turn is None and spend(
                reduced_cost(commander_card, reductions, chosen_type),
                commander_pips):
            commander_turn = turn

        # A COST REDUCER IS NEITHER A ROCK, A TUTOR NOR A BODY — the third card
        # to fall through this hole, after Aggravated Assault and Primal Vigor.
        # Urza's Incubator and Herald's Horn are artifacts with `produces` 0 and
        # `bodies` 0, so every existing loop skips them and they would sit in
        # hand for ten turns while being the deck's stated curve fixer. Cast
        # BEFORE anything else affordable, because a reducer's whole value is
        # what it makes the rest of the turn cost.
        for card in sorted((c for c in hand if c["reduces"] and c["bodies"] == 0
                            and c["produces"] == 0),
                           key=lambda c: c["cmc"]):
            if spend(reduced_cost(card, reductions, chosen_type), card["pips"]):
                reductions.append(card["reduces"])
                hand.remove(card)

        # Cast rocks cheapest-first; they produce starting next turn.
        for card in sorted((c for c in hand if c["produces"] > 0), key=lambda c: c["cmc"]):
            if spend(reduced_cost(card, reductions, chosen_type), card["pips"]):
                if card["reduces"]:
                    reductions.append(card["reduces"])
                rock_production += card["produces"]
                if model_colors:
                    # A rock adds as many sources as it makes mana.
                    sources.extend([card["colors"] or frozenset()] * card["produces"])
                hand.remove(card)

        # Cast tutors before bodies: a tutor is a setup spell, and it competes
        # for the same mana. Previously tutors had bodies=0 and produces=0, so
        # they were never cast at all and their mana silently went to creatures.
        for card in sorted((c for c in hand if c["tutor"]), key=lambda c: c["tutor_cmc"]):
            if card["tutor_needs_body"] and bodies_cum < 1:
                continue
            if not spend(card["tutor_cmc"], card["pips"]):
                continue
            hand.remove(card)
            tutor_ready_turns.append(turn + card["tutor_delay"])

        # A permanent that grants an ADDITIONAL COMBAT PHASE is neither a rock,
        # a tutor nor a body, so the resource-only loop never cast it at all --
        # Aggravated Assault sat in hand for ten turns while being the deck's
        # only verified win line. Bought before creatures, because it is the
        # thing the creatures are for.
        if model_combat:
            for card in sorted((c for c in hand
                                if not c["is_land"] and c["bodies"] == 0
                                and (c["combat"]["extra_combat_cost"] is not None
                                     or c["combat"]["extra_combat_free"])),
                               key=lambda c: c["cmc"]):
                if spend(card["cmc"], card["pips"]):
                    hand.remove(card)
                    if card["combat"]["extra_combat_cost"] is not None:
                        extra_combat_costs.append(card["combat"]["extra_combat_cost"])
                    else:
                        extra_combat_free += 1

        # A TREASURE MULTIPLIER IS NEITHER A ROCK, A TUTOR NOR A BODY — the
        # same hole Aggravated Assault fell through two loops up. `bodies` is
        # the model's proxy for "is this worth casting", and it happens to be 1
        # for Anointed Procession, Parallel Lives and Doubling Season (their
        # text reads as token creation) and 0 for Primal Vigor, which is the
        # identical card. So Primal Vigor sat in hand for ten turns while
        # carrying the flag that says it changes what the deck produces, and a
        # candidate sweep read it as byte-identical to a card that does nothing.
        # A flag the model set is a claim the model must act on.
        if model_treasures:
            for card in sorted((c for c in hand if not c["is_land"]
                                and c["bodies"] == 0 and c["produces"] == 0
                                and not c["tutor"]
                                and (c["treasure_doubler"] or c["treasure_bonus"])),
                               key=lambda c: c["cmc"]):
                if not spend(card["cmc"], card["pips"]):
                    continue
                hand.remove(card)
                if card["treasure_doubler"]:
                    treasure_multiplier *= 2
                if card["treasure_bonus"]:
                    treasure_bonus += 1

        # Spend what's left on bodies, cheapest-first.
        for card in sorted((c for c in hand if c["bodies"] > 0),
                           key=lambda c: reduced_cost(c, reductions, chosen_type)):
            if spend(reduced_cost(card, reductions, chosen_type), card["pips"]):
                bodies_cum += card["creature_bodies"] if model_combat else card["bodies"]
                hand.remove(card)
                # Dragonlord's Servant and Dragonspeaker Shaman are bodies that
                # also reduce; from here on they pay for every Dragon behind them.
                if card["reduces"]:
                    reductions.append(card["reduces"])
                if model_combat:
                    combat = card["combat"]
                    if combat["is_creature"]:
                        battlefield.append((combat["power"], turn, combat["haste"]))
                    # Creature tokens arrive with summoning sickness too, and
                    # they arrive on the turn their maker resolved.
                    if combat["token_bodies"]:
                        each = combat["token_power"] // max(combat["token_bodies"], 1)
                        for _ in range(combat["token_bodies"]):
                            battlefield.append((each, turn, False))
                    if combat["extra_combat_free"]:
                        extra_combat_free += 1
                    if combat["extra_combat_cost"] is not None:
                        extra_combat_costs.append(combat["extra_combat_cost"])
                    if any((combat["attack_mana"], combat["attack_treasure"],
                            combat["attack_draw"], combat["damage_scales_with_treasure"],
                            combat["attack_damage"], combat["attack_token_bodies"])):
                        combat_engines.append(combat)
                # Casting it turns its Treasure engine on for later turns, and
                # an ETB or cast trigger pays out immediately.
                if not model_treasures:
                    pass
                elif card["treasure_bonus"]:
                    treasure_bonus += 1
                if model_treasures and card["treasure_doubler"]:
                    # Two doublers is x4, not x3 — each replaces the other's
                    # output, which is why this compounds rather than sums.
                    treasure_multiplier *= 2
                if not model_treasures:
                    pass
                elif card["treasure_trigger"] in ("upkeep", "landfall"):
                    treasure_engines.append((card["treasure_n"], card["treasure_trigger"]))
                elif card["treasure_trigger"] in ("etb", "cast"):
                    treasures += ((card["treasure_n"] + treasure_bonus)
                                  * treasure_multiplier)
        bodies_by_turn.append(bodies_cum)

        # ── Combat step ────────────────────────────────────────────────────
        # Nothing blocks, so every creature that can attack does. Each combat
        # phase fires the attack triggers again, which is what makes an
        # additional combat phase worth more than its own power.
        if model_combat:
            attackers = [p for p, arrived, haste in battlefield
                         if haste or arrived < turn]
            swing = sum(attackers)
            phases = 1 + extra_combat_free
            # Buy as many extra combats as the leftover mana allows, cheapest
            # first. `pool` is what survived the main phase.
            for cost in sorted(extra_combat_costs):
                while pool + treasures >= cost:
                    if not spend(cost):
                        break
                    phases += 1
                    if phases > 20:       # runaway guard; an infinite is a win
                        break
                if phases > 20:
                    break

            dealt = 0
            for _ in range(phases):
                if not attackers:
                    break
                bonus = treasures if any(
                    e["damage_scales_with_treasure"] for e in combat_engines) else 0
                dealt += swing + bonus
                for engine in combat_engines:
                    pool += engine["attack_mana"]
                    dealt += engine["attack_damage"]
                    if model_treasures:
                        treasures += engine["attack_treasure"]
                    for _ in range(engine["attack_draw"]):
                        if deck:
                            extra = deck.pop(0)
                            hand.append(extra)
                            seen.add(extra["name"])
                    # Tokens made mid-combat are summoning-sick, so they swell
                    # the board for NEXT turn rather than this swing.
                    if engine["attack_token_bodies"]:
                        each = (engine["attack_token_power"]
                                // max(engine["attack_token_bodies"], 1))
                        for _ in range(engine["attack_token_bodies"]):
                            battlefield.append((each, turn, False))
                        bodies_cum += engine["attack_token_bodies"]
            opponent_life -= dealt
            damage_by_turn.append(dealt)
            board_power_by_turn.append(sum(p for p, _, _ in battlefield))
            if kill_turn is None and opponent_life <= 0:
                kill_turn = turn

        # Measured against the turn's FULL mana — lands, rocks and the
        # Treasure stockpile — because a Treasure you are holding is mana you
        # could have spent. `pool` has already been drawn down by the main
        # phase, so this asks what was reachable at the START of it.
        available = lands_in_play + rock_production + treasures
        stall_by_turn.append(not any(
            (not c["is_land"]) and c["cmc"] <= available for c in hand))
        hand_size_by_turn.append(len(hand))

        tutors = sum(1 for t in tutor_ready_turns if t <= turn)
        for i, target in enumerate(targets):
            commander_cast = commander_turn is not None
            if target_turns[i] is None and _target_met(target, seen, commander_cast, tutors):
                target_turns[i] = turn
            if target_turns_unassisted[i] is None and _target_met(target, seen, commander_cast):
                target_turns_unassisted[i] = turn

    return {
        "first_seven_lands": first_seven_lands,
        "kept_hand_lands": kept_hand_lands,
        "mulligans": mulligans,
        "land_hits": land_hits,
        "stall_by_turn": stall_by_turn,
        "hand_size_by_turn": hand_size_by_turn,
        "mana_by_turn": mana_by_turn,
        "commander_turn": commander_turn,
        "bodies_by_turn": bodies_by_turn,
        "target_turns": target_turns,
        "target_turns_unassisted": target_turns_unassisted,
        "treasures_by_turn": treasures_by_turn,
        "treasure_online_by_turn": treasure_online_by_turn,
        "damage_by_turn": damage_by_turn,
        "board_power_by_turn": board_power_by_turn,
        "kill_turn": kill_turn,
    }


def _round(x):
    return round(x, 3)


def aggregate(results, targets, max_turn, model_treasures=False, model_combat=False):
    n = len(results)
    turns = list(range(1, max_turn + 1))

    commander_turns = [r["commander_turn"] for r in results]
    cast_counts = {}
    for t in commander_turns:
        key = str(t) if t is not None else "not_by_max_turn"
        cast_counts[key] = cast_counts.get(key, 0) + 1
    cast_values = sorted(t for t in commander_turns if t is not None)

    target_stats = []
    for i, target in enumerate(targets):
        def _rates(key):
            got = sorted(r[key][i] for r in results if r[key][i] is not None)
            return {
                "assembled_rate": _round(len(got) / n),
                "mean_turn": _round(sum(got) / len(got)) if got else None,
                "by_turn_6_rate": _round(sum(1 for t in got if t <= 6) / n),
            }
        # The unassisted figures keep the historical key names, so every
        # existing consumer and every published figure still means what it
        # meant. Tutor-assisted estimates sit beside them under _assisted.
        assisted = _rates("target_turns")
        stats = {"label": target["label"], **_rates("target_turns_unassisted")}
        stats.update({
            "assembled_rate_assisted": assisted["assembled_rate"],
            "mean_turn_assisted": assisted["mean_turn"],
            "by_turn_6_rate_assisted": assisted["by_turn_6_rate"],
        })
        target_stats.append(stats)

    def _histogram(key):
        counts = {}
        for r in results:
            bucket = str(r[key])
            counts[bucket] = counts.get(bucket, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: int(kv[0])))

    return {
        "iterations": n,
        "opening_hand": {
            # Two distributions, deliberately both reported. `first_seven` is
            # the deck's real land distribution and moves when you change the
            # mana base. `kept_hand` is that distribution after the keep rule
            # has filtered it, so it sits near 100% inside the keep window for
            # every deck — informative about the mulligan rule, useless as a
            # fitness signal. The single `land_histogram` key this replaces
            # carried the second while being read as the first.
            "first_seven_land_histogram": _histogram("first_seven_lands"),
            "kept_hand_land_histogram": _histogram("kept_hand_lands"),
            "keep_first_seven_rate": _round(sum(1 for r in results if r["mulligans"] == 0) / n),
            "mean_mulligans": _round(sum(r["mulligans"] for r in results) / n),
        },
        "land_drop_hit_rate_by_turn": {
            str(t): _round(sum(1 for r in results if r["land_hits"][t - 1]) / n) for t in turns
        },
        "mean_available_mana_by_turn": {
            str(t): _round(sum(r["mana_by_turn"][t - 1] for r in results) / n) for t in turns
        },
        "commander": {
            "cast_turn_histogram": cast_counts,
            "mean_cast_turn": _round(sum(cast_values) / len(cast_values)) if cast_values else None,
            "median_cast_turn": cast_values[len(cast_values) // 2] if cast_values else None,
            "cast_by_turn_6_rate": _round(sum(1 for t in cast_values if t <= 6) / n),
        },
        "mean_bodies_by_turn": {
            str(t): _round(sum(r["bodies_by_turn"][t - 1] for r in results) / n) for t in turns
        },
        # A Treasure is a one-shot reserve, so it is reported SEPARATELY from
        # `mean_available_mana_by_turn` rather than folded into it. Folding it in
        # would have changed what that series has always meant — repeatable mana
        # per turn — and it is quoted in published prose across the fleet.
        **({} if not model_treasures else {"treasure": {
            "mean_treasures_in_hoard_by_turn": {
                str(t): _round(sum(r["treasures_by_turn"][t - 1] for r in results) / n)
                for t in turns
            },
            "engine_online_rate_by_turn": {
                str(t): _round(sum(1 for r in results if r["treasure_online_by_turn"][t - 1]) / n)
                for t in turns
            },
        }}),
        # The clock. `kill_turn` is the turn an UNOPPOSED board would finish one
        # 40-life seat — deliberately not a win rate, because nothing here models
        # blockers, removal or three opponents. Reported beside board power so a
        # fast clock built out of 2/2s is legible as such.
        **({} if not model_combat else {"combat": (lambda kills: {
            "mean_board_power_by_turn": {
                str(t): _round(sum(r["board_power_by_turn"][t - 1] for r in results) / n)
                for t in turns
            },
            "mean_damage_by_turn": {
                str(t): _round(sum(r["damage_by_turn"][t - 1] for r in results) / n)
                for t in turns
            },
            "kill_turn_histogram": dict(sorted(
                ((str(k), sum(1 for r in results if r["kill_turn"] == k))
                 for k in {r["kill_turn"] for r in results} if k is not None),
                key=lambda kv: int(kv[0]))),
            "mean_kill_turn": _round(sum(kills) / len(kills)) if kills else None,
            "median_kill_turn": kills[len(kills) // 2] if kills else None,
            "kill_by_turn_rate": {
                str(t): _round(sum(1 for k in kills if k <= t) / n) for t in turns
            },
            "no_kill_by_max_turn_rate": _round((n - len(kills)) / n),
        })(sorted(r["kill_turn"] for r in results if r["kill_turn"] is not None))}),
        "targets": target_stats,
    }


class _Silent:
    """A progress sink for a run inside a sweep, which draws its own."""

    def advance(self, n=1):
        pass


def run(slug, iterations=None, seed=None, max_turn=None,
        model_treasures=None, model_combat=None, with_results=False, branch=None,
        doc=None, quiet=False, targets_override=None, model_colors=None):
    """Run the goldfish simulation for a deck. Returns the metrics document.

    `model_treasures` and `model_combat` default to None, meaning READ THE
    DECK'S DECLARATION — the opt-in described below, and the behaviour every
    tracked `goldfish_metrics.json` was produced under. An explicit bool
    OVERRIDES it, which exists for one caller: the benchmark.

    A benchmark cannot read per-deck flags. Exactly one deck of twelve opts into
    combat today, so aggregating the fleet's own metrics would rank a deck that
    was measured with a kill clock against eleven that were not — the
    "uncontrolled output cannot be aggregated" failure the PRD names. The
    benchmark therefore runs its OWN uniform configuration and never writes to
    the deck's tracked file.
    """
    iterations = iterations or GOLDFISH_ITERATIONS
    seed = GOLDFISH_SEED if seed is None else seed
    max_turn = max_turn or GOLDFISH_MAX_TURN

    # `doc` lets a caller measure a list that is not on disk — one card
    # substituted, to find out what that card actually does. It changes nothing
    # about the model; it only skips the read.
    doc = doc if doc is not None else load_deck_cards(slug, branch)
    library, commanders = build_library(doc)
    if not commanders:
        raise SystemExit(f"No commander flagged in {slug}/cards.json")
    commander_cmc = int(commanders[0].get("cmc") or 0)

    # A branch inherits the deck's ENGINE DECLARATION unless it writes its own:
    # nobody authors a second targets file to try a candidate list, and measuring
    # a branch against no declaration would report a different deck rather than a
    # different list.
    targets_path = deck_file(slug, "goldfish_targets.json", branch)
    targets = []
    # OPT-IN, and for the same reason `OPTIONAL_DEPARTMENTS` existed: a model
    # that changes every deck's numbers at once cannot be landed on one deck
    # first. Measured before choosing this — turning it on fleet-wide moves
    # three decks' published figures, and gishath's `mean_cast_turn` alone
    # (7.969 -> 7.912) is quoted SIXTEEN times across seven tracked artifacts
    # including agent-authored prose and an `engine.json` carrying a critic
    # verdict. Silently invalidating that is the "confident and wrong" failure
    # this project exists to avoid, so a deck opts in when it is next
    # re-baselined deliberately.
    #
    # With the model off the treasure keys are ABSENT rather than zeroed, so the
    # six unaffected decks stay byte-identical and nothing needs regenerating.
    # Remove this flag once every deck has been re-baselined; a permanently
    # optional model is one nobody committed to.
    declared_treasures = False
    declared_combat = False
    # Bound before the branch: a deck with no declaration still has colours,
    # and reading it only inside the `if` made every declaration-less deck
    # (which is the benchmark's whole fleet path) raise UnboundLocalError.
    targets_doc = {}
    if targets_path.exists():
        with open(targets_path) as f:
            targets_doc = json.load(f)
        # AN OVERRIDE MUST REACH THE SIMULATION, NOT JUST THE REPORT. The first
        # cut passed a modified declaration to the reporting layer while this
        # loop still read the file — so `target_turns` was indexed by the FILE's
        # targets and the override changed nothing. The tell was eight different
        # candidates returning the identical 0.501.
        targets = targets_override if targets_override is not None else targets_doc["targets"]
        declared_treasures = bool(targets_doc.get("model_treasures"))
        declared_combat = bool(targets_doc.get("model_combat"))
        # A target member not in the deck can never be drawn — it silently
        # deflates the assembly rate (a target naming a card ur-dragon had moved
        # out once cost it a wrong "cost reducer drawn" figure). Warn loudly; the
        # fix is authored, so this stays a warning rather than a hard error.
        main_names = {c.get("name") for c in doc.get("cards", [])}
        for target in targets:
            for group in target.get("need", []):
                ghosts = [n for n in group.get("any_of", []) if n not in main_names]
                if ghosts:
                    if not quiet:
                        print(f"  WARNING target '{target.get('label', '?')}' names "
                          f"cards not in the maindeck (can never be drawn): "
                          f"{', '.join(ghosts)}")

    # Name every Treasure source the model CANNOT see. Silence here would make a
    # low hoard figure look like a modelling bug instead of a fact about the
    # deck — and the fact is usually load-bearing: ur-dragon's four Treasure
    # makers are all combat-triggered, so a goldfish that never attacks reports
    # zero and is right to.
    # An explicit argument OVERRIDES the declaration; None means read it. The
    # benchmark is the one caller that overrides, because uniform conditions are
    # what makes decks comparable at all.
    model_treasures = declared_treasures if model_treasures is None else bool(model_treasures)
    model_combat = declared_combat if model_combat is None else bool(model_combat)
    # COLOUR IS NOT OPTIONAL THE WAY TREASURES ARE. Every deck has colours and a
    # colourless mana model is simply wrong; the flag exists so the change can
    # be measured against the old behaviour and against `mana_analysis`'s
    # closed form, not so a deck can decline to have colours.
    declared_colors = bool((targets_doc or {}).get("model_colors", True))
    model_colors = declared_colors if model_colors is None else bool(model_colors)
    commander_pips = cast_pips(
        front_field(commanders[0], "mana_cost") or "") if commanders else []

    # THE COMMAND ZONE IS A SOURCE OF STATIC EFFECTS, and this is the first one
    # the model reads. Eminence is live from turn one whether or not the
    # commander is ever cast, and it cannot be answered — the single most
    # load-bearing fact about a deck built on it.
    creature_types = _corpus_creature_types()
    chosen_type = chosen_type_for(doc["cards"])
    command_zone_reduction = []
    commander_subtypes = frozenset()
    for c in commanders:
        got = cost_reduction(c, creature_types)
        if got:
            command_zone_reduction.append(got)
        commander_subtypes |= subtypes_of(c.get("type_line") or "")

    # A CARD IS BLIND ONLY IF EVERY CHANNEL IS BLIND. This list was built from
    # `treasure_profile` alone while the model has three ways to see a Treasure:
    # the trigger table, `treasure_bonus` (an adder — Xorn, Jolene),
    # `treasure_doubler` (Procession, Mondrak) and `combat.attack_treasure` (Goldspan, Old Gnawbone, Ragavan) once
    # `model_combat` is on. Reported from one channel it named nineteen sources
    # on ur-dragon's treasure branch as invisible when six of them were being
    # simulated — and the whole point of the list is that a low hoard figure
    # should be LEGIBLE, so over-reporting it is the same failure as omitting it.
    def _blind(c):
        # Computed from the card, not read off it: these are raw cards.json
        # entries, and the derived fields only exist on a built library entry.
        if treasure_profile(c)[1] != "unmodelled":
            return False
        text = c.get("oracle_text") or ""
        if TREASURE_BONUS_RE.search(text):
            return False                       # an adder: Xorn, Jolene
        if TOKEN_DOUBLER_RE.search(text):
            return False                       # a doubler: Procession, Mondrak
        if model_combat and combat_profile(c).get("attack_treasure"):
            return False                       # Goldspan, Old Gnawbone, Ragavan
        return True

    unmodelled = sorted({
        c["name"] for c in doc.get("cards", [])
        if not c.get("is_commander") and _blind(c)
    })
    # HOW MUCH OF THIS LIST THE DRAW ASSUMPTION HIDES. Not a list of names:
    # unlike a Treasure blind spot there is no figure here to make legible —
    # card advantage is measured nowhere — so what a reader needs is the SIZE
    # of the gap, which is what makes two decks' figures comparable or not.
    draw_cards = sum(
        c.get("quantity", 1) for c in doc.get("cards", [])
        if not c.get("is_commander") and _DRAW_RE.search(c.get("oracle_text") or ""))
    restricted = sorted({
        f"{c['name']} ({produced_mana(c.get('oracle_text'))})"
        for c in doc.get("cards", [])
        if "Land" not in (c.get("type_line") or "")
        and produced_mana(c.get("oracle_text"))
        and _RESTRICTED_MANA_RE.search(c.get("oracle_text") or "")
    })
    if not model_treasures:
        visible = sorted({
            c["name"] for c in doc.get("cards", [])
            if not c.get("is_commander")
            and treasure_profile(c)[1] in ("upkeep", "landfall", "cast", "etb")
        })
        if visible:
            print(f"  WARNING {slug} has {len(visible)} Treasure source(s) this model "
                  f"CAN simulate and `model_treasures` is not set in "
                  f"goldfish_targets.json, so they are ignored: {', '.join(visible)}")

    # Same contract as the Treasure warning above, one layer out: a combat
    # trigger whose EFFECT the parser cannot price scores zero, and a zero
    # nobody is told about reads as a fact about the deck.
    combat_unreadable = sorted({
        c["combat"]["unreadable"] for c in library if c["combat"]["unreadable"]
    }) if model_combat else []

    rng = random.Random(seed)
    # The loop is a list comprehension no longer, because 10,000 silent
    # simulations look identical to a hang. The comprehension is otherwise
    # unchanged — same rng, same order, same seed, so the RESULT is
    # bit-identical and `tests/test_pilot_goldfish.py`'s determinism assertions
    # hold. Progress is drawn on stderr and nothing here reads it back.
    results = []
    # A sweep runs this dozens of times; a progress bar per run is noise, and the
    # sweep draws its own.
    ctx = (contextlib.nullcontext(_Silent()) if quiet
           else console.task(f"Goldfishing {slug}", total=iterations, unit="sims"))
    with ctx as t:
        for _ in range(iterations):
            results.append(
                simulate_once(rng, library, commander_cmc, targets, max_turn,
                              command_zone_reduction=command_zone_reduction,
                              chosen_type=chosen_type,
                              commander_subtypes=commander_subtypes,
                              model_treasures=model_treasures,
                              model_combat=model_combat,
                              model_colors=model_colors,
                              commander_pips=commander_pips))
            t.advance()

    return {
        "meta": {
            "deck": slug,
            "decklist_sha256": doc.get("decklist_sha256"),
            "seed": seed,
            "model_version": model_version(),
            "iterations": iterations,
            "max_turn": max_turn,
            "commander": commanders[0]["name"],
            "commander_cmc": commander_cmc,
            "model_assumptions": MODEL_ASSUMPTIONS + (
                TREASURE_ASSUMPTIONS if model_treasures else []) + (
                COMBAT_ASSUMPTIONS if model_combat else []),
            # RESTRICTED MANA IS COUNTED AS FREE, AND THE READER SHOULD KNOW.
            # `spend()` is a scalar, so it cannot represent "only to cast
            # Dragon spells". Delighted Halfling's legendary-only mana is very
            # nearly free in a Commander deck; Throne of Eldraine's four is not.
            # Same contract as the Treasure blind spots: the assumption is
            # NAMED rather than silently made or silently dropped.
            "card_advantage": {
                "cards_that_draw": draw_cards,
                "modelled": 0,
                "why": "one card per turn, always — see model_assumptions. The "
                       "understatement is proportional to this count, so two "
                       "decks with different counts are not directly comparable "
                       "on any speed figure.",
            },
            **({"restricted_mana_counted_as_free": restricted} if restricted else {}),
            **({"treasure_sources_not_modelled": unmodelled} if model_treasures else {}),
            **({"combat_effects_not_modelled": combat_unreadable}
               if model_combat and combat_unreadable else {}),
        },
        "metrics": aggregate(results, targets, max_turn, model_treasures, model_combat),
        # OPT-IN, and default off so the returned document is byte-identical
        # to every tracked `goldfish_metrics.json`. Two tests compare `run`'s
        # output against the artifact directly, and they caught this the first
        # time it was unconditional — which is exactly what they are for.
        #
        # The benchmark needs a SPREAD and `aggregate` reports means, so it asks
        # for the rows rather than the shared artifact growing a stdev key to
        # serve one caller.
        **({"_results": results} if with_results else {}),
    }


def main(args):
    branch = getattr(args, "branch", None)
    # BRANCHED WRITE, UN-BRANCHED READ — the mirror of the defect
    # `resolve_out_path` documents, and it silently filed the CHAMPION's
    # measurement under the branch's name for as long as branches have existed.
    # On ur-dragon's treasure branch that understated the turn-10 hoard 5.29 ->
    # 1.32, a factor of four, in a file whose own `meta.decklist_sha256` said
    # which list it had really measured. Nothing read that field.
    doc = run(args.slug, branch=branch)
    out = deck_dir(args.slug, branch) / "goldfish_metrics.json"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    commander = doc["meta"]["commander"]
    stats = doc["metrics"]["commander"]
    print(
        f"Wrote {out}\n  {commander}: mean cast turn {stats['mean_cast_turn']}, "
        f"cast by turn 6 in {stats['cast_by_turn_6_rate']:.0%} of games"
    )
    for target in doc["metrics"]["targets"]:
        print(f"  {target['label']}: by turn 6 in {target['by_turn_6_rate']:.0%} of games")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot goldfish <slug>`.")
