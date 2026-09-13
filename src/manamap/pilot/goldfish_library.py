"""THE LIBRARY: a decklist becomes a shuffled pile of CLASSIFIED cards.

MOVED OUT OF `goldfish.py` ON 2026-09-13, verbatim. Four functions and one
authored constant, and they are the seam between the card readers and the game:
`classify` runs every profile in `goldfish_profiles` once per card and returns
the flat dict the turn loop reads, `build_library` expands copies into that pile,
`keepable` is the mulligan rule, and `_target_met` asks whether a declared
engine component has been assembled.

`classify`'s output is the model's ONTOLOGY — the set of things the simulator can
see about a card — and `test_metric_hygiene` sweeps its emitted keys to assert
that nothing is computed and never read.
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
    GOLDFISH_POISON_TO_LOSE,
    GOLDFISH_SEED,
)
from manamap.pilot import manabase
from manamap.pilot.common import (
    deck_dir, deck_file, front_field, load_deck_cards)

from manamap.pilot.goldfish_profiles import (
    GOLDFISH_MULLIGAN_MAX_LANDS,
    GOLDFISH_MULLIGAN_MIN_LANDS,
    TOKEN_DOUBLER_RE,
    TREASURE_BONUS_RE,
    _ATTACK_ENABLER_RE,
    _EERIE_RE,
    _SCALING_COLOR_MANA_RE,
    _TUTOR_MODE_COST_RE,
    _TUTOR_SAC_RE,
    _TUTOR_TO_BATTLEFIELD_RE,
    _TUTOR_TO_TOP_RE,
    _corpus_creature_types,
    _land_mana_bonus,
    body_count,
    cast_pips,
    cast_token_profile,
    combat_profile,
    cost_reduction,
    creature_body_count,
    death_profile,
    devotion_gate,
    drain_profile,
    draw_profile,
    event_payoffs,
    front_field,
    is_tutor,
    manabase,
    produced_mana,
    room_profile,
    sac_outlet_profile,
    subtypes_of,
    token_doubler,
    treasure_profile,
)


#: A PILOT DOES NOT CAST STROKE OF GENIUS FOR X=1. Scryfall counts {X} as
#: zero, so an X spell's `cmc` is its FIXED part and every cheapest-first
#: casting loop in this module would fire it on the turn that part is
#: affordable — Stroke of Genius on turn three for X=0, drawing nothing and
#: burning the card. The floor is authored and it is the only authored number
#: in this channel; it is stated in `model_assumptions` for that reason.
X_DRAW_MIN = 2


def classify(card, pool=None):
    """Return a compact sim-card dict for one physical copy.

    `pool` is the deck's lands, and it exists for ONE card class: a fetchland's
    colours are a property of the deck, not of the card (`manabase.land_colors`).
    Without it every fetch is a colourless land that never produces anything.
    """
    type_line = card.get("type_line", "")
    text = card.get("oracle_text") or ""
    is_land = "Land" in type_line and "Creature" not in type_line.split("//")[0]
    is_tutor_card = bool(not is_land and is_tutor(card))
    mode_cost = _TUTOR_MODE_COST_RE.search(text) if is_tutor_card else None
    # None for every card that is not a Room, so this is a pure widening: a deck
    # with no Room takes the same `cmc` and the same `pips` it always did, and
    # its output is byte-identical.
    room = room_profile(card)
    if room:
        # EVERY TEXT-DERIVED PROFILE BELOW NOW READS ONE DOOR. Rebinding `card`
        # and `text` here rather than patching each profile call means bodies,
        # draw, drain, combat and the rest are all attributed correctly by
        # construction, and a profile added later inherits the fix for free.
        # The TYPE LINE is deliberately left whole: the permanent really is an
        # Enchantment whichever door is open.
        card = dict(card, oracle_text=room["entry_text"])
        text = room["entry_text"]
        # What the OTHER door does, applied when it is unlocked and not before.
        # Same parsers, different half — no new effect vocabulary is invented,
        # and the corpus sweep says that is enough: of the 26 Rooms with a "when
        # you unlock this door" clause the effects are tokens (5), counters (4),
        # recursion (4) and draw (2), all of which these already read.
        _unlocked = dict(card, oracle_text=room["unlock_text"])
        room["on_unlock"] = {
            "bodies": body_count(_unlocked),
            "creature_bodies": creature_body_count(_unlocked),
            "draw": draw_profile(_unlocked),
            "drain": dict(drain_profile(_unlocked),
                          eerie=bool(_EERIE_RE.search(room["unlock_text"]))),
            "combat": combat_profile(_unlocked),
        }
    return {
        "name": card["name"],
        "is_land": is_land,
        # A ROOM COSTS ONE DOOR, NOT BOTH. `cmc` is what the model SPENDS, and
        # it doubles as the animated body's size — which is correct for a Room
        # with one door open, per CR 709.5. `room["full_mv"]` takes over once
        # both are.
        "cmc": room["entry_cost"] if room else int(card.get("cmc") or 0),
        "room": room,
        # The eerie flag lives on the DRAIN PROFILE and not here, because the
        # scorer holds profiles rather than cards — and a duplicate on the card
        # was set and never read, which `test_metric_hygiene` caught by name.
        # A flag the model sets is a claim the model must act on.
        # A GOD IS NOT A CREATURE BELOW ITS DEVOTION THRESHOLD. None when the
        # card carries no such clause, which is 20 of the 23 enchantment
        # creatures in the list this was written for.
        "devotion_gate": devotion_gate(card),
        # CARRIED FOR THE COMMANDER'S ATTACK TUTOR, which filters on the printed
        # type. Nothing else in this model reads a type line at simulation time —
        # every other question is answered here, at classify time — so this key
        # exists for one caller and says so. Without it the filter matched the
        # empty string and the tutor silently never fired.
        "type_line": type_line,
        # What it actually costs to USE the tutor mode, which is what decides
        # when the wildcard comes online.
        "tutor_cmc": int(card.get("cmc") or 0) + (int(mode_cost.group(1)) if mode_cost else 0),
        # A SCALING DORK PRODUCES AT LEAST ONE. Without this it never reaches
        # the rock loop at all, which is how it came to read as zero.
        "produces": 0 if "Land" in type_line else (
            produced_mana(card.get("oracle_text"), type_line)
            or (1 if _SCALING_COLOR_MANA_RE.search(text) else 0)),
        "land_mana_bonus": 0 if "Land" in type_line else _land_mana_bonus(text),
        "bodies": 0 if "Land" in type_line else body_count(card),
        # Creature-only body count and the combat profile ride along always;
        # they are READ only under `model_combat`, so a non-opted deck is
        # byte-identical and this stays a pure widening of the sim card.
        "creature_bodies": 0 if "Land" in type_line else creature_body_count(card),
        "combat": combat_profile(card),
        # ON EVERY CARD, not only the commander. Read under `model_combat` /
        # `model_draw` like the rest, so a deck that opts into neither is
        # byte-identical.
        "cast_token": cast_token_profile(card),
        # The scorer holds a drain profile, not the card, so the flag that
        # decides whether an unlock re-fires this payoff has to travel with it.
        "drain": dict(drain_profile(card), eerie=bool(_EERIE_RE.search(text))),
        "attack_enabler": bool(_ATTACK_ENABLER_RE.search(
            card.get("oracle_text", "") or "")),
        "draw": draw_profile(card),
        # Discard- and draw-triggered payoffs, read on every card and acted on
        # under `model_discard` only.
        "event": event_payoffs(card),
        "death": death_profile(card),
        "token_doubler": token_doubler(card),
        "sac_outlet": sac_outlet_profile(card),
        "tutor": is_tutor_card,
        # A top-of-library tutor delivers on the next draw step, not this turn.
        "tutor_delay": 1 if is_tutor_card and _TUTOR_TO_TOP_RE.search(text) else 0,
        "tutor_needs_body": bool(is_tutor_card and _TUTOR_SAC_RE.search(text)),
        # The searched creature ENTERS rather than going to hand; the type it
        # may be, or "" for any creature.
        "tutor_to_battlefield": (
            (_TUTOR_TO_BATTLEFIELD_RE.search(text).group(1) or "")
            if is_tutor_card and _TUTOR_TO_BATTLEFIELD_RE.search(text) else None),
        "treasure_n": 0 if is_land else treasure_profile(card)[0],
        "treasure_trigger": None if is_land else treasure_profile(card)[1],
        # Xorn makes no Treasure of its own; it adds one to every event.
        # WHAT IT PRODUCES and WHAT IT COSTS, in colours. Both ride along
        # always and are READ only under `model_colors`, so the colourless path
        # stays byte-identical — the `creature_bodies` rule.
        #
        # THE TURN LOOP BROKE THAT RULE AND THIS COMMENT WAS THE TELL (#35,
        # fixed 2026-09-13). `sources` — the colours on the battlefield — was
        # APPENDED to only under the flag, and `scales_with_colors` producers
        # read it in BOTH arms to size their own output. So with the flag off
        # they found an empty list and made one mana instead of up to five: the
        # flag added a castability penalty and unlocked a production bonus at
        # once, and on a five-colour deck the bonus won. `sources` is tracked
        # unconditionally now and read only under the flag, which is what this
        # paragraph always claimed. `land_colors` is
        # `manabase`'s and is deliberately restriction-aware: Haven of the
        # Spirit Dragon taps for {C} in a Vampire deck, and counting it as five
        # sources is how a mana base comes out looking fine and cannot cast its
        # spells.
        # `pool` resolves a fetchland against what it can actually go and get.
        # A non-land is unaffected: `fetch_profile` gates on the type line.
        "colors": frozenset(manabase.land_colors(card, pool=pool)),
        # THE PIPS MUST COME FROM THE DOOR THE COST CAME FROM. `front_field`
        # always answers with the left half, which is the wrong half on the six
        # Rooms whose back door is cheaper.
        "pips": room["entry_pips"] if room
                else cast_pips(front_field(card, "mana_cost") or ""),
        "treasure_bonus": bool(TREASURE_BONUS_RE.search(text)),
        # Anointed Procession et al. make none either, and DOUBLE every event.
        # Rides on the card always and is read only under `model_treasures`, so
        # a non-opted deck stays byte-identical — the `creature_bodies` rule.
        "treasure_doubler": bool(TOKEN_DOUBLER_RE.search(text)),
        # Static cost reduction, and what this card IS so a reduction can be
        # tested against it. Both ride along always and are read only when a
        # reducer is actually in play, so a deck with none stays byte-identical.
        "reduces": cost_reduction(card, _corpus_creature_types()),
        # Priced at CAST TIME from the colours actually in play, so it enters
        # the rock loop (`produces > 0`) and its real output is computed there.
        "scales_with_colors": bool(_SCALING_COLOR_MANA_RE.search(text)),
        "subtypes": subtypes_of(type_line, text),
        "is_creature": "Creature" in type_line,
    }


def build_library(doc):
    """Expand the main deck (minus commanders) into per-copy sim cards."""
    library = []
    commanders = []
    # The fetch pool is every land in the list, commander included — a fetch
    # searches the LIBRARY, and what it may find does not depend on which zone
    # the search was started from.
    pool = [c for c in doc["cards"] if "Land" in str(c.get("type_line") or "")]
    for card in doc["cards"]:
        if card.get("is_commander"):
            commanders.append(card)
            continue
        library.extend([classify(card, pool=pool)] * card.get("quantity", 1))
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
    # THE HOTTEST LINE IN THE SIMULATION. Profiled on edgar at 2,000 games,
    # this function and its two generator expressions were 0.797s of a 2.088s
    # loop — 38% — over 290,127 calls, because the `any_of` scan is rebuilt in
    # Python on every call for every unmet need on every turn of every game.
    #
    # The `any_of` list is CONSTANT for the whole run, so the set is built once
    # per need and cached on it. `names_in_hand` is already a set (`seen`), so
    # `isdisjoint` is a C-level intersection test. Same predicate, same result:
    # a need is unmet exactly when none of its names has been seen.
    #
    # Cached on the need dict rather than threaded through the signature
    # because the raw need dicts never reach the artifact — `target_stats`
    # takes only `target["label"]` — so there is nothing for a private key to
    # leak into.
    unmet = 0
    for need in target["need"]:
        names = need.get("_any_of_set")
        if names is None:
            names = need["_any_of_set"] = frozenset(need["any_of"])
        if names.isdisjoint(names_in_hand):
            unmet += 1
            # Counting past the tutor budget cannot change the answer.
            if unmet > tutors:
                return False
    return True
