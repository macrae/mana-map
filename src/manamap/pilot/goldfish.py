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

import json
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
from manamap.pilot.common import deck_dir, load_deck_cards

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
    "Cost reducers, rituals, and extra card draw are not modeled (conservative).",
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
_TAP_ADD_RE = re.compile(r"\{T\}: Add ((?:\{[WUBRGC0-9]\})+)")

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
# Xorn: "instead create those tokens plus an additional Treasure token."
_TRE_EXTRA_RE = re.compile(r"instead create those tokens plus an additional Treasure",
                           re.IGNORECASE)


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
    r"whenever (?:this creature|[A-Z][\w' ,-]{2,30}|one or more [\w ]+ you control) attacks",
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
        mana = _TAP_ADD_RE.search(window) or re.search(
            r"add ((?:\{[WUBRGC0-9]\})+)", window, re.IGNORECASE)
        if mana:
            profile["attack_mana"] = _mana_pips(mana.group(1))
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

    return profile


def produced_mana(oracle_text):
    """Mana a persistent '{T}: Add ...' producer yields per turn (0 if none)."""
    match = _TAP_ADD_RE.search(oracle_text or "")
    if not match:
        return 0
    return len(re.findall(r"\{[WUBRGC0-9]\}", match.group(1)))


def body_count(card):
    """Bodies this card contributes when cast: itself (if creature) + tokens."""
    bodies = 1 if "Creature" in card.get("type_line", "") else 0
    for word in _TOKEN_RE.findall(card.get("oracle_text", "") or ""):
        bodies += _NUMBER_WORDS.get(word.lower(), 1 if not word.isdigit() else int(word))
    return bodies


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
        "treasure_bonus": bool(_TRE_EXTRA_RE.search(text)),
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
                  model_treasures=False, model_combat=False):
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
    treasures = 0                 # a STOCKPILE: each one is spendable once
    treasure_engines = []         # (per_event, trigger) for modelled sources in play
    treasure_bonus = 0            # Xorn-style +N per creation event
    treasures_by_turn = []
    treasure_online_by_turn = []
    commander_turn = None
    land_hits = []
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
            hand.pop(land_index)
            lands_in_play += 1
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
                treasures += per_event + treasure_bonus

        pool = lands_in_play + rock_production
        # Reported WITHOUT the stockpile, so this series keeps meaning exactly
        # what it has always meant: repeatable mana per turn. Treasures are a
        # one-shot reserve and get their own series.
        mana_by_turn.append(pool)
        treasures_by_turn.append(treasures)
        treasure_online_by_turn.append(bool(treasure_engines))

        def spend(cost):
            """Pay from lands and rocks first, then break Treasures. Returns True."""
            nonlocal pool, treasures
            if pool + treasures < cost:
                return False
            if cost <= pool:
                pool -= cost
            else:
                treasures -= cost - pool
                pool = 0
            return True

        if commander_turn is None and spend(commander_cmc):
            commander_turn = turn

        # Cast rocks cheapest-first; they produce starting next turn.
        for card in sorted((c for c in hand if c["produces"] > 0), key=lambda c: c["cmc"]):
            if spend(card["cmc"]):
                rock_production += card["produces"]
                hand.remove(card)

        # Cast tutors before bodies: a tutor is a setup spell, and it competes
        # for the same mana. Previously tutors had bodies=0 and produces=0, so
        # they were never cast at all and their mana silently went to creatures.
        for card in sorted((c for c in hand if c["tutor"]), key=lambda c: c["tutor_cmc"]):
            if card["tutor_needs_body"] and bodies_cum < 1:
                continue
            if not spend(card["tutor_cmc"]):
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
                if spend(card["cmc"]):
                    hand.remove(card)
                    if card["combat"]["extra_combat_cost"] is not None:
                        extra_combat_costs.append(card["combat"]["extra_combat_cost"])
                    else:
                        extra_combat_free += 1

        # Spend what's left on bodies, cheapest-first.
        for card in sorted((c for c in hand if c["bodies"] > 0), key=lambda c: c["cmc"]):
            if spend(card["cmc"]):
                bodies_cum += card["creature_bodies"] if model_combat else card["bodies"]
                hand.remove(card)
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
                if not model_treasures:
                    pass
                elif card["treasure_trigger"] in ("upkeep", "landfall"):
                    treasure_engines.append((card["treasure_n"], card["treasure_trigger"]))
                elif card["treasure_trigger"] in ("etb", "cast"):
                    treasures += card["treasure_n"] + treasure_bonus
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


def run(slug, iterations=None, seed=None, max_turn=None):
    """Run the goldfish simulation for a deck. Returns the metrics document."""
    iterations = iterations or GOLDFISH_ITERATIONS
    seed = GOLDFISH_SEED if seed is None else seed
    max_turn = max_turn or GOLDFISH_MAX_TURN

    doc = load_deck_cards(slug)
    library, commanders = build_library(doc)
    if not commanders:
        raise SystemExit(f"No commander flagged in {slug}/cards.json")
    commander_cmc = int(commanders[0].get("cmc") or 0)

    targets_path = deck_dir(slug) / "goldfish_targets.json"
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
    model_treasures = False
    model_combat = False
    if targets_path.exists():
        with open(targets_path) as f:
            targets_doc = json.load(f)
        targets = targets_doc["targets"]
        model_treasures = bool(targets_doc.get("model_treasures"))
        model_combat = bool(targets_doc.get("model_combat"))
        # A target member not in the deck can never be drawn — it silently
        # deflates the assembly rate (a target naming a card ur-dragon had moved
        # out once cost it a wrong "cost reducer drawn" figure). Warn loudly; the
        # fix is authored, so this stays a warning rather than a hard error.
        main_names = {c.get("name") for c in doc.get("cards", [])}
        for target in targets:
            for group in target.get("need", []):
                ghosts = [n for n in group.get("any_of", []) if n not in main_names]
                if ghosts:
                    print(f"  WARNING target '{target.get('label', '?')}' names "
                          f"cards not in the maindeck (can never be drawn): "
                          f"{', '.join(ghosts)}")

    # Name every Treasure source the model CANNOT see. Silence here would make a
    # low hoard figure look like a modelling bug instead of a fact about the
    # deck — and the fact is usually load-bearing: ur-dragon's four Treasure
    # makers are all combat-triggered, so a goldfish that never attacks reports
    # zero and is right to.
    unmodelled = sorted({
        c["name"] for c in doc.get("cards", [])
        if not c.get("is_commander") and treasure_profile(c)[1] == "unmodelled"
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
    with console.task(f"Goldfishing {slug}", total=iterations, unit="sims") as t:
        for _ in range(iterations):
            results.append(
                simulate_once(rng, library, commander_cmc, targets, max_turn,
                              model_treasures=model_treasures, model_combat=model_combat))
            t.advance()

    return {
        "meta": {
            "deck": slug,
            "decklist_sha256": doc.get("decklist_sha256"),
            "seed": seed,
            "iterations": iterations,
            "max_turn": max_turn,
            "commander": commanders[0]["name"],
            "commander_cmc": commander_cmc,
            "model_assumptions": MODEL_ASSUMPTIONS + (
                TREASURE_ASSUMPTIONS if model_treasures else []) + (
                COMBAT_ASSUMPTIONS if model_combat else []),
            **({"treasure_sources_not_modelled": unmodelled} if model_treasures else {}),
            **({"combat_effects_not_modelled": combat_unreadable}
               if model_combat and combat_unreadable else {}),
        },
        "metrics": aggregate(results, targets, max_turn, model_treasures, model_combat),
    }


def main(args):
    doc = run(args.slug)
    out = deck_dir(args.slug) / "goldfish_metrics.json"
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
