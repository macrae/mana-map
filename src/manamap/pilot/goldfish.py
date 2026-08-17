"""Pilot: goldfish simulation — seeded Monte Carlo resource-development metrics.

Tier-2 (data-derived) evidence for the pilot's manual. The model simulates
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

from manamap.config import (
    GOLDFISH_ITERATIONS,
    GOLDFISH_MAX_MULLIGANS,
    GOLDFISH_MAX_TURN,
    GOLDFISH_MULLIGAN_MAX_LANDS,
    GOLDFISH_MULLIGAN_MIN_LANDS,
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

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "x": 0,
}

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
                  model_treasures=False):
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

        # Spend what's left on bodies, cheapest-first.
        for card in sorted((c for c in hand if c["bodies"] > 0), key=lambda c: c["cmc"]):
            if spend(card["cmc"]):
                bodies_cum += card["bodies"]
                hand.remove(card)
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
    }


def _round(x):
    return round(x, 3)


def aggregate(results, targets, max_turn, model_treasures=False):
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
    if targets_path.exists():
        with open(targets_path) as f:
            targets_doc = json.load(f)
        targets = targets_doc["targets"]
        model_treasures = bool(targets_doc.get("model_treasures"))
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

    rng = random.Random(seed)
    results = [
        simulate_once(rng, library, commander_cmc, targets, max_turn,
                      model_treasures=model_treasures)
        for _ in range(iterations)
    ]

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
                TREASURE_ASSUMPTIONS if model_treasures else []),
            **({"treasure_sources_not_modelled": unmodelled} if model_treasures else {}),
        },
        "metrics": aggregate(results, targets, max_turn, model_treasures),
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
