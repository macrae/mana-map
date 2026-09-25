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
    GOLDFISH_POISON_TO_LOSE,
    GOLDFISH_SEED,
)
from manamap.pilot import manabase
from manamap.pilot.common import (
    deck_dir, deck_file, front_field, load_deck_cards)


# ── The card readers, moved to `goldfish_profiles` on 2026-09-13 ──────────
#
# Re-exported rather than referenced through the module, so every caller
# that reaches for `goldfish.<name>` — six modules and thirty test files —
# keeps working unchanged. The split is about where the code LIVES, and a
# rename would make a 5,800-line move unreviewable.
from manamap.pilot.goldfish_profiles import (  # noqa: F401
    CHOSEN_TYPE,
    commander_copies_spells,
    ETB_CHAIN_LIMIT,
    GOLDFISH_ITERATIONS,
    GOLDFISH_MAX_MULLIGANS,
    GOLDFISH_MAX_TURN,
    GOLDFISH_MULLIGAN_MAX_LANDS,
    GOLDFISH_MULLIGAN_MIN_LANDS,
    GOLDFISH_OPPONENT_LIFE,
    GOLDFISH_POISON_TO_LOSE,
    GOLDFISH_SEED,
    TOKEN_DOUBLER_RE,
    TREASURE_BONUS_RE,
    WHEEL_MIN_HAND,
    _ACTIVATED_COMBAT_RE,
    _ACTIVATION_COST_RE,
    _ALL_CREATURES,
    _ARRIVAL_DRAW_RE,
    _ARRIVAL_GAIN_RE,
    _ATTACKS_RE,
    _ATTACK_ENABLER_RE,
    _ATTACK_PING_EACH_RE,
    _ATTACK_TOKEN_SCALES_RE,
    _CAST_DAMAGE_POWER_RE,
    _CAST_DRAW_GATES,
    _CAST_DRAW_MV_RE,
    _CAST_DRAW_PAY_RE,
    _CAST_DRAW_RE,
    _CAST_PIP_RE,
    _CAST_TOKEN_RE,
    _CAST_TOKEN_SCALES_RE,
    _CAST_TOKEN_TYPE_GATES,
    _CHANGELING_RE,
    _COMBAT_DMG_RE,
    _CONSTELLATION_DRAIN_RE,
    _CONSTELLATION_GAIN_RE,
    _CONSTELLATION_TOKEN_RE,
    _CONSUMING_COST,
    _COSTED_SAC_OUTLET_RE,
    _COST_REDUCTION_RE,
    _CREATURE_TYPES_CACHE,
    _DAMAGE_DOUBLER_RE,
    _DEATH_DRAIN_RE,
    _DEATH_DRAW_RE,
    _DEATH_TREASURE_RE,
    _DEATH_TRIGGER_RE,
    _DEVOTION_COLOURS,
    _DEVOTION_GATE_RE,
    _DEVOTION_WORDS,
    _DMG_EQUAL_TREASURE_RE,
    _DRAIN_EQUAL_RE,
    _DRAIN_FIXED_RE,
    _DRAW_ADDITIONAL_COST_RE,
    _DRAW_CONDITIONAL_RE,
    _DRAW_GREATEST_POWER_RE,
    _DRAW_ONCE_RE,
    _DRAW_POWER_MAX_RE,
    _DRAW_POWER_MIN_RE,
    _DRAW_QUALIFIER_OK_RE,
    _DRAW_RE,
    _DRAW_THEN_DISCARD_RE,
    _DRAW_WORDS,
    _EERIE_RE,
    _ENCHANTMENT_ENTERS,
    _ENTERS_HASTE_RE,
    _ETB_CHOSEN_TYPE_COPY_RE,
    _ETB_CLAUSE_END_RE,
    _ETB_CLAUSE_HEAD_RE,
    _ETB_COPY_NONLEGENDARY_RE,
    _ETB_COPY_RE,
    _ETB_DMG_COUNT_RE,
    _ETB_DMG_FIXED_RE,
    _ETB_DMG_POWER_RE,
    _ETB_DRAW_PER_TYPE_RE,
    _ETB_DRAW_RE,
    _ETB_ENGINE_FIELDS,
    _ETB_LIFE_LOSS_RE,
    _ETB_NONTOKEN_RE,
    _ETB_OPTIONAL_COST_RE,
    _ETB_SUBJECT_RE,
    _ETB_THIS_TURN_RE,
    _ETB_TRIGGER_RE,
    _EVENT_COUNTER_RE,
    _EVENT_DAMAGE_RE,
    _EVENT_DRAW_RE,
    _EVENT_TOKEN_RE,
    _EVENT_TRIGGER_RE,
    _EXTRA_COMBAT_RE,
    _FLYING_KW_RE,
    _FREE_SAC_OUTLET_RE,
    _GENERIC_PIP_RE,
    _GRANTED_AWAY,
    _GRANT_CLASS,
    _GRANT_CONDITIONAL_RE,
    _GRANT_TO_SELF,
    _HASTE_RE,
    _INFECT_KW_RE,
    _IRREGULAR,
    _KEYWORD_NOT_GRANTED,
    _LAND_MANA_BONUS_RE,
    _LIFELINK_GRANT_TYPE_RE,
    _LIFELINK_NOT_SELF_RE,
    _LIFELINK_RE,
    _LIFE_WORDS,
    _MANA_WORDS,
    _MASS_ANIMATE_ALWAYS_RE,
    _MASS_ANIMATE_RE,
    _NONCREATURE_TOKENS,
    _NOT_A_CREATURE_TYPE,
    _NUMBER_WORDS,
    _OPPONENT_DEATH_GAIN_RE,
    _PT_RE,
    _RECURRING_DRAIN_RE,
    _RECURRING_DRAW_RE,
    _RECURRING_GAIN_RE,
    _RESTRICTED_MANA_RE,
    _RIOT_TEAM_RE,
    _ROOM_TYPE_RE,
    _SCALING_COLOR_MANA_RE,
    _SCALING_REDUCTION_RE,
    _SCRY_THEN_DRAW_RE,
    _SELF_DOUBLE_STRIKE_RE,
    _SPELL_DAMAGE_POWER_RE,
    _SPELL_DRAW_RE,
    _SYMMETRIC_DRAIN_GAIN_ETB_RE,
    _SYMMETRIC_DRAIN_GAIN_PER_TYPE_RE,
    _SYMMETRIC_DRAIN_GAIN_RE,
    _TAP_ADD_RE,
    _TEAM_COUNTER_ETB_RE,
    _TEAM_COUNTER_ON_TYPE_RE,
    _TEAM_DOUBLE_STRIKE_RE,
    _TEMPORARY_EFFECT_RE,
    _TEAM_HASTE_RE,
    _TEAM_POWER_DOUBLE_RE,
    _TOKEN_CLAUSE_HEAD_RE,
    _TOKEN_CREATED_TRIGGER_RE,
    _TOKEN_DOUBLER_RE,
    _TOKEN_PT_RE,
    _TOKEN_RE,
    _TOXIC_KW_RE,
    _TREASURE_N_RE,
    _TREASURE_RE,
    _TRE_CAST_RE,
    _TRE_ETB_RE,
    _TRE_EXTRA_RE,
    _TRE_LANDFALL_RE,
    _TRE_SAGA_RE,
    _TRE_UPKEEP_RE,
    _TUTOR_LAND_RE,
    _TUTOR_MODE_COST_RE,
    _TUTOR_RE,
    _TUTOR_SAC_RE,
    _TUTOR_TO_BATTLEFIELD_RE,
    _TUTOR_TO_TOP_RE,
    _TYPED_HASTE_RE,
    _TYPED_TUTOR_RE,
    _UNSET,
    _UPKEEP_REVEAL_RE,
    _WHEEL_EXCLUDED,
    _WHEEL_RE,
    _X_DRAW_ALT_COST_RE,
    _X_DRAW_DISCARD_ONE_RE,
    _X_DRAW_DISCARD_X_RE,
    _X_DRAW_RE,
    _X_SCALES_RE,
    _X_SUBJECTS,
    _corpus_creature_types,
    _etb_clause,
    _grant_is_to_self,
    _inside_activation,
    _land_mana_bonus,
    _life_amount,
    _mana_pips,
    _sentence_around,
    _singular,
    _stat,
    _static_grant,
    _strip_foreign_grants,
    body_count,
    can_pay,
    cast_pips,
    cast_token_profile,
    chosen_type_for,
    combat_profile,
    console,
    contextlib,
    cost_reduction,
    creature_body_count,
    death_profile,
    deck_dir,
    deck_file,
    devotion_gate,
    devotion_of,
    drain_profile,
    draw_profile,
    event_payoffs,
    front_field,
    has_event_payoff,
    is_death_engine,
    is_etb_engine,
    is_tutor,
    json,
    load_deck_cards,
    mana_value,
    manabase,
    pathlib,
    produced_mana,
    random,
    re,
    reduced_cost,
    room_profile,
    sac_outlet_profile,
    subtypes_of,
    team_haste_grant,
    token_doubler,
    treasure_profile,
)

# ── The game, moved out on 2026-09-13 ────────────────────────────────────────
#
# `goldfish.py` was 5,830 lines: the card readers were two thirds of it and the
# turn loop most of the rest, so a reader looking for `run` scrolled past every
# regex in the project first. Three modules now, and this one keeps the CLI, the
# aggregation and the declaration handling.
#
# Re-exported so every caller that reaches for `goldfish.<name>` keeps working —
# the split is about where code LIVES. Proved by re-measuring all 39 tracked
# metric files and diffing them byte for byte.
from manamap.pilot.goldfish_library import (  # noqa: F401
    X_DRAW_MIN, _target_met, build_library, classify, keepable,
)
from manamap.pilot.goldfish_turn import simulate_once  # noqa: F401


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
class DeclarationError(ValueError):
    """A deck's `goldfish_targets.json` declares something the model cannot use.

    `run()` RAISED `SystemExit` FOR THIS, and `run()` is called IN PROCESS by
    `benchmark`, `diagnostic`, `calibrate` and `deck_branch` — so one deck's
    malformed declaration killed whatever command was running. Mid-`regen`, that
    is a fleet sweep abandoned at deck three with a message about a file nobody
    asked about.

    A caller decides now: `main` converts this to `SystemExit` so the terminal
    behaviour is unchanged, and `regen` reports the deck and carries on.
    """


#: THE SIMULATOR IS FOUR FILES NOW, so the stamp is over all four.
#:
#: It hashed `__file__` alone, which was correct while the simulator WAS one
#: file. After the 2026-09-13 split, a change to a card reader in
#: `goldfish_profiles` or to the turn loop in `goldfish_turn` would not have
#: moved the version at all — every derived artifact would have read as current
#: while the model underneath it changed. That is the exact failure this stamp
#: exists to prevent, and the split would have created it.
_MODEL_FILES = ("goldfish.py", "goldfish_profiles.py", "goldfish_library.py",
                "goldfish_turn.py")


def model_version():
    """First 12 hex of a sha256 over the simulator's source — ALL of it.

    Coarse on purpose: a comment edit bumps it, which costs a regeneration
    nobody needed. The alternative is a curated list of "model-facing" lines,
    which is exactly the judgement call that goes wrong silently.

    Sorted by filename so the stamp does not depend on directory order.
    """
    import hashlib

    here = pathlib.Path(__file__).parent
    sha = hashlib.sha256()
    for name in sorted(_MODEL_FILES):
        sha.update(name.encode())
        sha.update((here / name).read_bytes())
    return sha.hexdigest()[:12]



MODEL_ASSUMPTIONS = [
    "Simulates resource development, not full games (no interaction, no removal).",
    "Draw every turn including turn 1 (multiplayer Commander).",
    "Mulligan: keep 7-card hands with 2-5 lands; up to 2 fresh redraws, keep the last.",
    "One land drop per turn when available.",
    # MEASURED, NOT ASSUMED. A `candidates` sweep of twelve lands against
    # ur-dragon returned exactly TWO distinct readings: 45.304 for every
    # five-colour land and 44.027 for every restricted one. Grand Coliseum,
    # which always enters tapped, read identically to Forbidden Orchard,
    # which never does — the byte-identical tell this repo already uses to
    # catch a flag nothing acts on. The loop plays the FIRST land in hand
    # and credits its colours the same turn, so there is no tapped state to
    # act on. Modelling it would slow every deck's early turns and restate
    # every published figure on the fleet, so it is named here rather than
    # changed quietly.
    "LANDS ENTER UNTAPPED, ALWAYS, and the one in hand longest is the one "
    "played. A tapland costs nothing here and no land is ever chosen over "
    "another, so this model CANNOT rank two lands that make the same "
    "colours — `mana-analysis` and `mana-fit` are what answer a land "
    "question, and they are deterministic for exactly this reason.",
    "Mana rocks ('{T}: Add') contribute from the turn after they are cast.",
    "Commander cast on first affordable turn (highest spending priority).",
    "Bodies count = creatures cast + tokens parsed from 'create ... token' text.",
    "Target assembly counts cards DRAWN by a turn (cast cards still count).",
    "X-SPELL DRAW IS PAID FOR OUT OF WHAT IS LEFT. An {X} instant or sorcery "
    "that draws X is cast AFTER every other loop has spent what it wanted, "
    "with the whole remaining pool, and only when X would be at least "
    f"{X_DRAW_MIN} — a pilot does not cast Stroke of Genius for one card, and "
    "every cheapest-first loop in this module would have, because Scryfall "
    "counts {X} as zero and the card's `cmc` is only its fixed part. That "
    "floor is the one authored number in this channel. Casting last makes the "
    "figure a FLOOR in the other direction too: the spell only ever gets mana "
    "nothing else asked for. Corpus sweep: 33 X-cost instants and sorceries "
    "that draw X, 28 credited and 5 skipped — a split card whose cmc is both "
    "halves, an alternative cost under which X is 0, and three whose X is "
    "bounded by a discard or by a graveyard this model does not have.",
    "CARD DRAW IS ONE A TURN UNLESS `model_draw` IS SET. With the flag on, four "
    "channels are modelled — a card's own ETB draw, an instant or sorcery that "
    "draws, an upkeep trigger, and an ARRIVAL trigger that draws when other "
    "creatures or tokens enter (Welcoming Vampire, Caretaker's Talent), which "
    "rides the same single door the ETB damage payoffs use. Activated draw "
    "(needs a spending policy this model has no opinion on), X-based draw "
    "(board-dependent), and death- or attack-triggered draw (no deaths, and "
    "attacks only under `model_combat`) are NOT modelled and the cards are "
    "named in `meta.card_advantage.draw_not_modelled`. A conditional trigger — "
    "'if you control an artifact', 'you may draw' — is treated as unmodelled "
    "rather than assumed to be on: 39 of the 348 ETB-draw cards in the corpus "
    "carry one.",
    "AN ARRIVAL-DRAW ENGINE NEVER SEES ITS OWN ARRIVAL. Welcoming Vampire is a "
    "2/3 that draws 'whenever one or more OTHER creatures you control with "
    "power 2 or less enter'; registered before its own entry it passed its own "
    "gate and drew a card it does not draw, overstating edgar-vampires by 22% "
    "at turn eight. The deferral applies to every card and not only the ones "
    "worded 'other', which understates Tocasia's Welcome and two relatives by "
    "one draw on the turn they land.",
    "ARRIVAL DRAW IS SMALLER WITHOUT `model_combat`, and legitimately so. The "
    "channel rides the one door onto the battlefield, which `model_combat` also "
    "uses to spawn token copies and ETB-payoff tokens — those are real "
    "additional arrivals that do not exist in the resource-only model. Measured "
    "on edgar-vampires: 1.264 extra cards by turn ten with both flags against "
    "1.106 with draw alone. Before 2026-08-28 the gap was 1.264 against 0.323, "
    "because every call to that door sat inside `if model_combat:` and a "
    "draw-only deck silently lost three quarters of its arrival draws.",
    "HELD-UP INTERACTION is reported as two series and both are FLOORS. This "
    "model casts everything it can afford every turn, so `interaction_castable` "
    "is what a pilot developing at full speed has left over, not what a pilot "
    "choosing to hold up two mana would have. A low figure is therefore a real "
    "finding and a high one is unambiguous good news.",
    "Tutors are modeled as wildcards: a CAST tutor that fetches to hand or the "
    "top — including a TYPED one (Worldly Tutor, Sarkhan's Triumph), which the "
    "old literal 'a card' match missed entirely — fills ONE missing any_of "
    "group. Consumed once, mana paid, and a tutor that puts the card on top of "
    "the library costs a turn. An ACTIVATED, ETB or death-triggered tutor does "
    "not count, because this model only knows 'drawn and affordable'; nor does "
    "a land fetch, which is ramp and has no channel here yet. Reported as the "
    "*_assisted figures; the unassisted figures beside them exclude tutors "
    "entirely.",
    "ENTERS-THE-BATTLEFIELD PAYOFFS are modelled: damage equal to the entering "
    "creature's power (Terror of the Peaks), X damage where X counts your board "
    "(Scourge of Valkas, Dragon Tempest), a token (Lathliss) and a token COPY "
    "of what entered (Miirym). Every arrival fires them — cast, token or copy — "
    "and a copy is itself an arrival, so they compound, which is the deck "
    "working rather than a bug. A payoff worded 'another NONTOKEN' does not "
    "re-trigger on its own tokens, which is the brake the rules already had. A "
    "chain is capped at 12 deep. X counts the whole board rather than one "
    "subtype: exact where the creatures are Dragons, generous otherwise.",
    "The COMMANDER joins the battlefield when cast, attacks, and fires its own "
    "triggers. It used to be a mana sink and a flag — cast, then dropped — so a "
    "10/10 flier contributed no power and never swung. Commander tax, death and "
    "recasting are still not modelled: it is cast once and it stays.",
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
    "Only Treasure triggers a goldfish can see are modeled — upkeep and landfall "
    "are RECURRING, Saga chapters likewise; a `cast` or enters-the-battlefield "
    "trigger pays out ONCE, when its own source resolves. "
    "Combat- and opponent-gated sources produce NOTHING here, because this model "
    "has no combat and no opponents; they are named in "
    "meta.treasure_sources_not_modelled so a low hoard figure is legible.",
]

# Appended only for a deck that opts in, same contract as TREASURE_ASSUMPTIONS.
DISCARD_ASSUMPTIONS = [
    "BLOOD: a Blood token is created on an ETB, a per-opponent ETB (three, the "
    "pod) or a spell's resolution; Blood made by CONNECTING in combat is read "
    "and NOT credited, because a blocker can prevent it and this model has no "
    "blockers. AT MOST ONE IS CRACKED PER TURN, before the casting loops, and "
    "only with a card in hand, since discarding is part of the cost. One a turn "
    "is an authored floor: a pilot holding three with mana to spare cracks "
    "more, so this understates. Cracking one is a DISCARD and a DRAW, which the "
    "commander payoffs below charge for, and it sacrifices an ARTIFACT, which "
    "the artifact-sacrifice payoffs charge for.",
    "ACTIVATED DRAW: a draw you BUY -- '{1}, {T}, Sacrifice this artifact: Draw "
    "a card' -- is paid for out of whatever mana the casting loops did not "
    "want, and a cost that eats its own source fires once. Energy and {X} "
    "activation costs are REFUSED and named, never priced at zero.",
    "DISCARD: a wheel discards the WHOLE hand, lands included, and is cast only "
    "when the hand holds at most three nonland cards or a discard payoff is on "
    "the battlefield; a loot pitches lands beyond the next drop first, then the "
    "most expensive nonland. A discarded card stays 'seen' for target assembly "
    "(assembly is 'drawn by', and it was). Wheel of Misfortune is excluded "
    "(conditional). Activated loots and rummages, cycling and madness are NOT "
    "modelled and are named in meta.draw_not_modelled. Discard- and draw-"
    "triggered payoffs pay a flat amount per event, outside the combat gate; "
    "+1/+1 counters they put on their own body are added to the swing while the "
    "commander is out, an approximation. Payoffs that DRAW on a draw are not "
    "read: an uncapped loop would draw the library and call it steam.",
    "DISCARD: a wheel a PERMANENT carries is activated from the battlefield "
    "under the same hand-size gate as a wheel that is cast, paying its mana "
    "cost every time. It fires at the earliest on the turn AFTER the permanent "
    "lands, which is the summoning sickness of the tap in its cost, and a cost "
    "that sacrifices its own source fires once and takes the body off the "
    "board. An ability activated from the graveyard (Runehorn Hellkite) is a "
    "zone this model does not have and is refused rather than fired.",
    "DISCARD: THE OPPONENT'S HALF OF A WHEEL IS NOW COUNTED, AND ONLY THE TAX "
    "ON IT. A symmetrical wheel refills the opponent too, so `whenever an "
    "opponent draws a card` payoffs (Razorkin Needlehead, Scrawling Crawler, "
    "Mind's Eye, Consecrated Sphinx) fire on their draw step and on every card "
    "a wheel deals them. ONE SEAT, the same convention the damage pillar uses "
    "-- Brallin's '1 damage to EACH opponent' is counted once against a single "
    "40-life opponent -- so a real three-opponent table pays roughly three "
    "times this and every figure here is a FLOOR. WHAT IS STILL NOT MODELLED "
    "IS THE COST: the opponent draws those cards and never gets to use them, "
    "because this model has no opponent turn, no opponent board and no "
    "interaction. So the tax is priced and the risk it offsets is not. A "
    "trigger on an opponent CASTING (Rhystic Study) is a different family and "
    "stays unread.",
    "DISCARD: A WHEEL THAT DRAWS 'CARDS EQUAL TO THE GREATEST NUMBER A PLAYER "
    "DISCARDED' IS A FLOOR HERE, AND THE FLOOR IS LOW. This model has no "
    "opponents holding cards, so Jace's Archivist and Windfall draw what OUR "
    "hand held -- and the gate only lets them fire when our hand is thin, "
    "which is exactly the board on which a real table pays them seven. The "
    "figure is not corrected, because the correction would be an authored "
    "opponent hand size driving a headline; read it as a lower bound and "
    "settle the question in Forge.",
]

COMBAT_ASSUMPTIONS = [
    "COMBAT: one opponent at 40 life who does nothing — no blockers, no removal, "
    "no interaction. This is a goldfish in the literal sense, so `kill_turn` is "
    "the turn an UNOPPOSED board would finish ONE seat, not a win rate.",
    "Attacks with every creature that is not summoning-sick; haste is read from "
    "the ORACLE TEXT (this line said 'type line' and was simply wrong). There is nothing to block, so nothing is ever held back — in "
    "a real four-player game you would keep blockers, which makes this an "
    "optimistic clock and a pessimistic board.",
    "A COMMANDER'S COMBAT-DAMAGE REVEAL IS DECLARED PER DECK "
    "(`model_commander_combat_reveal`, 2026-09-11, Gishath): when the commander "
    "is on the battlefield and the deck attacks, with the declared connect rate "
    "it reveals its damage's worth of cards, every creature of the named type "
    "among them enters through the one door (ETB payoffs, engines, reductions "
    "all register; nothing that needs a CAST fires), the rest go to the bottom. "
    "The rate is Forge's, named in the record; this model has no blockers.",
    "AN ENTRY TRIGGER THAT NAMES A TYPE FIRES ON THAT TYPE (2026-09-11): Dragon "
    "Tempest no longer fires on a mana dork; Molten Echoes copies the chosen "
    "type. Tokens carry no type line here and pass the gate. A CAST fires what "
    "listens for a cast, at every door the commander included: draw on a "
    "spell of mana value N or more (Up the Beanstalk), draw for a paid mana on "
    "a creature cast (Lifecrafter's Bestiary, paid when the pool has it), "
    "damage on casting a creature of power N or more (Sarkhan's Unsealing, "
    "counted as noncombat damage to the one opponent). A spell that has a "
    "creature deal its power to each opponent (Chandra's Ignition) deals the "
    "biggest body's power once; the wipe half has nothing to hit. A tutor "
    "that puts a creature ONTO THE BATTLEFIELD (Savage Order) sacrifices the "
    "smallest 4-power nontoken body and fetches the highest-power creature "
    "of the named type the library holds -- a policy, stated. Evasion for "
    "the commander (Majestic Heliopterus) is Forge's to read: this model "
    "has no blockers and the connect rate is declared.",
    "A LAND-MANA BONUS (Mirari's Wake and three others) adds one mana per land "
    "from the turn after it lands; DRAW EQUAL TO THE GREATEST POWER resolves "
    "against the board at cast and is held while the board is empty; DRAW A "
    "CARD FOR EACH <type> ON ENTRY counts the board it joins; an attack token "
    "AS BIG AS THE BEST OTHER ATTACKER (Ghalta and Mavren) takes the second-"
    "largest swing. NOT modelled and named: Etali, Primal Storm (casting other "
    "players' cards is outside this model) and draw equal to combat damage "
    "dealt by one creature (Hunter's Insight).",
    "TEAM HASTE IS MODELLED (2026-09-11). A permanent that says 'creatures you "
    "control have haste' lets every creature attack the turn it lands, from "
    "the turn the grant itself lands; a typed grant (Karrthus: Dragons) reads "
    "the creature's type line; Dragon Tempest's grant reads keyword flying; "
    "riot is read as always choosing haste. TOKENS carry no type line and no "
    "flying here, so a typed or flying grant does not reach them. NOT read: a "
    "grant from the graveyard (Anger), an activated one (Crashing Drawbridge), "
    "one conditional on the board ('as long as'), and the lands whose mana "
    "carries haste (Hall of the Bandit Lord, Arena of Glory) -- this model "
    "taps no particular land. Before this a granted creature waited a turn "
    "and its haste enabler was read as nothing.",
    "Attack triggers, combat-damage triggers and additional combat phases are "
    "modelled, which is what makes Treasure sources gated on combat produce here "
    "when they produce nothing without this flag. Effects the parser cannot read "
    "are named in meta.combat_effects_not_modelled.",
    "DAMAGE MULTIPLICATION is modelled and stacks MULTIPLICATIVELY as the rules "
    "do: doubling the power, swinging twice and doubling the damage dealt is "
    "eight times, not four. Three wordings are read — a replacement effect on "
    "damage you deal (Twinflame Tyrant), a granted double strike (Atarka) and a "
    "power doubling (Thrakkus). A grant worded 'each Dragon you control' is "
    "treated as applying to the WHOLE team: exact in a deck whose attackers are "
    "Dragons, generous in one where they are not. A creature's own double strike "
    "multiplies only itself. Board power is unaffected — a double-striker is not "
    "a bigger creature.",
    "Bodies count CREATURES only under this flag. Without it a Treasure token "
    "scores as a creature, which inflates `mean_bodies_by_turn` for any deck "
    "that makes non-creature tokens.",
    "LIFE LOSS ON ARRIVAL COUNTS AS DAMAGE. 'Each opponent loses 1 life' "
    "(Corpse Knight) and 'deals 1 damage to each opponent' (Impact Tremors) are "
    "the same event, the same cadence and the same number against one opponent "
    "at 40 life, and only the second was priced until 2026-08-28. They are NOT "
    "the same in the rules — lifelink, damage prevention and 'whenever an "
    "opponent loses life' all separate them — and this model has none of those. "
    "Nine corpus cards read through this channel, plus Mirkwood Bats through a "
    "token-creation channel of its own.",
    "DEATH-TRIGGERED DRAIN IS NOT MODELLED AND IS THE LARGEST KNOWN GAP HERE. "
    "Blood Artist, Cruel Celebrant, Zulaport Cutthroat, Bastion of Remembrance "
    "and Elas il-Kor all key on a creature DYING, and nothing dies in this "
    "model: no blockers, no removal, no sacrifice outlets. Their contribution "
    "is ABSENT, not zero, and a deck whose kill runs through them is understated "
    "here by however much that line is worth. `simulate` against a real pod is "
    "where that number lives.",
]











def _both_online(results, turns, n):
    """When are BOTH commanders on the battlefield — the partner deck's ignition.

    ABSENT RATHER THAN ZERO for a game that never gets there: `mean_turn` is
    computed over the games that DID, and `never_rate` carries the rest. A mean
    that quietly counts a never-cast game as turn 10 would make a deck that
    assembles half the time look like one that assembles late.
    """
    both = [max(r["commander_turn"], r["partner_turn"])
            for r in results
            if r["commander_turn"] is not None and r["partner_turn"] is not None]
    got = len(both)
    out = {
        "both_online_by_turn": {
            str(t): _round(sum(1 for b in both if b <= t) / n) for t in turns},
        "both_online_by_turn_6_rate": _round(sum(1 for b in both if b <= 6) / n),
        "never_both_online_rate": _round((n - got) / n),
    }
    if got:
        out["mean_both_online_turn"] = _round(sum(both) / got)
        out["median_both_online_turn"] = sorted(both)[got // 2]
    return out


def _round(x):
    return round(x, 3)


def aggregate(results, targets, max_turn, model_treasures=False,
              model_combat=False, model_draw=False, model_sacrifice=False,
              model_drain=False, attack_tutor_rate=None, model_discard=False,
              partner=False, commander_reveal=None):
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
            # THE STRICTER THRESHOLD, beside the loose one it does not replace.
            "keep_can_act_by_t3_rate": _round(
                sum(1 for r in results if r["keep_can_act_by_t3"]) / n),
            "mean_mulligans": _round(sum(r["mulligans"] for r in results) / n),
        },
        # THE COMMANDER'S ATTACK TUTOR. It is a CEILING (see the model note
        # where it fires): a goldfish commander always attacks, a real one
        # attacks when it is safe to.
        #
        # ABSENT MEANS ABSENT. A deck that declares no tutor gets no block at
        # all rather than one reading `declared: false` — the same rule the
        # commander-damage block keeps, and it also keeps every other deck's
        # artifact free of a key about a thing it does not have.
        **({"commander_reveal": {
            # THE DECLARED REVEAL, reported the way the tutor is: an output of
            # the simulation, with the declared rate and its source beside it.
            "declared": True,
            "mean_fired": _round(sum(r["reveal_fired"] for r in results) / n),
            "games_it_fired": sum(1 for r in results if r["reveal_fired"]),
            "mean_bodies": _round(sum(r["reveal_bodies"] for r in results) / n),
            "rate": commander_reveal,
            "ceiling_note": "this model has no blockers; the rate is Forge's, not this model's",
        }} if commander_reveal else {}),
        **({"attack_tutor": {
            # DERIVED from the rows rather than passed in: a result carries
            # `attack_tutor_fired`, and a deck that never declared one has the
            # key at 0 in every game. `declared` is therefore "it fired at least
            # once somewhere", which is the honest thing this function can see.
            "declared": any(r["attack_tutor_fired"] for r in results),
            "mean_fired": _round(sum(r["attack_tutor_fired"] for r in results) / n),
            "games_it_fired": sum(1 for r in results if r["attack_tutor_fired"]),
            # THE RATE TRAVELS WITH THE FIGURE. A reader who sees "0.877 fires
            # per game" must be able to find out where that came from without
            # leaving the record.
            **({"rate": attack_tutor_rate} if attack_tutor_rate else {}),
            # HOW OFTEN THE DECK ACHIEVED THE FREE ATTACK. This is the number
            # "make the commander swing more often" is actually about, and it is
            # an OUTPUT of the simulation rather than a declared rate — it moves
            # when the decklist moves, which the rate cannot.
            "enabled_share": _round(
                sum(r["tutor_enabled_turns"] for r in results)
                / max(1, sum(r["tutor_eligible_turns"] for r in results))),
            "basis": ("fires at the DECLARED RATE from the turn after the "
                      "commander lands, pulling the highest-mana-value match "
                      "from the library onto the battlefield. The rate is "
                      "measured from a sim run and named in `rate.source`. It "
                      "used to fire EVERY turn, which was the engine's ceiling "
                      "reported as its output: 5.70 fires a game against the "
                      "1.22 Forge resolved over 60 games with piloting "
                      "confirmed comparable."),
        }} if any(r["attack_tutor_fired"] for r in results) else {}),
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
        # THE SERIES THIS MODEL EXISTED WITHOUT FOR A YEAR. Cumulative cards
        # drawn BEYOND the one-a-turn draw step, so it is zero for a deck with
        # no readable draw and the difference between two lists is the whole
        # of it. Absent, not zero, when the deck has not opted in.
        # HOW MANY TOKENS THE POLICY ATE. Reported so the reader can see the
        # size of the assumption rather than only its consequence: a run showing
        # 30 sacrifices by turn ten is a very different claim from one showing 3.
        **({"mean_sacrifices_by_turn": {
            str(t): _round(sum(r["sacrifices_by_turn"][t - 1] for r in results) / n)
            for t in turns},
            "sac_cap_hit_rate": _round(
                sum(1 for r in results if r.get("sac_cap_hits")) / n)}
           if model_sacrifice else {}),
        **({"drain": {
            "mean_drain_by_turn": {
                str(i): _round(sum(r["drain_by_turn"][i - 1] for r in results) / n)
                for i in turns if all(len(r["drain_by_turn"]) >= i for r in results)},
            "mean_cumulative_drain_by_turn": {
                str(i): _round(sum(sum(r["drain_by_turn"][:i]) for r in results) / n)
                for i in turns if all(len(r["drain_by_turn"]) >= i for r in results)},
            "share_of_games_draining_by_turn_10": _round(
                sum(1 for r in results if sum(r["drain_by_turn"])) / n),
           }} if model_drain else {}),
        **({"mean_extra_cards_drawn_by_turn": {
            str(t): _round(sum(r["drawn_extra_by_turn"][t - 1] for r in results) / n)
            for t in turns}} if model_draw else {}),
        # THE DISCARD CHANNEL. Cumulative cards discarded, and the damage the
        # discard/draw payoffs dealt, by turn. Absent when the deck did not opt
        # in -- not zero.
        **({"discard": {
            "mean_cards_discarded_by_turn": {
                str(t): _round(sum(r["discarded_by_turn"][t - 1] for r in results) / n)
                for t in turns},
            "mean_event_damage_by_turn": {
                str(t): _round(sum(r["event_damage_by_turn"][t - 1] for r in results) / n)
                for t in turns},
            "mean_cumulative_event_damage_by_turn": {
                str(t): _round(sum(sum(r["event_damage_by_turn"][:t]) for r in results) / n)
                for t in turns},
            # HOW BIG THE PAIR HAS GOT. The commanders' own +1/+1 counters,
            # cumulative — Brallin one per discard, Shabraz one per draw, so a
            # seven-card wheel is fourteen counters between them. The
            # simulation was already computing this and folding it into the
            # attack step; it was never a figure anybody could read. Their
            # OWN counters only: every other per-event counter payoff in the
            # deck (Chasm Skulker is the loud one) is in the swing and not
            # here, because the question is how large the commanders are.
            "mean_commander_counters_by_turn": {
                str(t): _round(sum(r["commander_counters_by_turn"][t - 1] for r in results) / n)
                for t in turns},
        }} if model_discard else {}),
        # THE PARTNER, cast on its own curve. `commander` above stays the first
        # commander's for every reader; this is the second's.
        #
        # AND THE JOINT, WHICH IS THE ONE THAT MATTERS ON A PARTNER DECK. Two
        # marginal means do not give it: a game that lands Brallin on turn 3
        # and Shabraz on turn 9 reads well on both rows above and has the
        # engine off until turn 9. On sharknado each commander taxes a
        # different half of a wheel — Brallin the discard, Shabraz the draw —
        # so the deck does not start when one is cast, it starts when the
        # second is. `both_online_by_turn` is that curve; the rest of this
        # block describes it.
        **({"partner": dict({
            "cast_by_turn_6_rate": _round(sum(1 for r in results if r["partner_turn"] is not None
                                               and r["partner_turn"] <= 6) / n),
            "mean_cast_turn": _round(sum(r["partner_turn"] for r in results if r["partner_turn"] is not None)
                                     / max(1, sum(1 for r in results if r["partner_turn"] is not None))),
        }, **_both_online(results, turns, n))} if partner else {}),
        # HELD-UP INTERACTION, both halves. A low `castable` against a high
        # `in_hand` is a MANA problem and not a drawing problem, which is the
        # distinction the pilot's own diagnosis turned on.
        "interaction_in_hand_by_turn": {
            str(t): _round(sum(r["interaction_in_hand_by_turn"][t - 1] for r in results) / n)
            for t in turns},
        "interaction_castable_by_turn": {
            str(t): _round(sum(r["interaction_castable_by_turn"][t - 1] for r in results) / n)
            for t in turns},
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
            # THE SECOND CLOCK. Counters given per turn and the share of games
            # the poison clock ended rather than the life one; on a deck with
            # no infect or toxic source both are exactly zero, which is a
            # measurement — the sources were counted and there were none.
            "mean_poison_by_turn": {
                str(t): _round(sum(r["poison_by_turn"][t - 1] for r in results) / n)
                for t in turns
            },
            "kill_by_poison_rate": _round(
                sum(1 for r in results if r["kill_by"] == "poison") / n),
        })(sorted(r["kill_turn"] for r in results if r["kill_turn"] is not None))}),
        "targets": target_stats,
    }


class _Silent:
    """A progress sink for a run inside a sweep, which draws its own."""

    def advance(self, n=1):
        pass


#: The figures the band is reported on, as PATHS INTO THIS MODULE'S OWN metrics
#: document. Headline speed, board and damage only — a declared commander
#: ability moves those and not the mana or the opening hand, and a band on every
#: row would bury the four that matter.
#:
#: NOT `candidates.OBJECTIVE_AXES`, which maps the DIAGNOSTIC's shape
#: (`output.kill_by_turn`) and not the goldfish's (`combat.kill_by_turn_rate`).
#: Reusing it returned an empty band in silence — every lookup missed and the
#: block rendered with no rows, which reads as "no difference" rather than
#: "nothing was read".
BAND_ROWS = {
    "kill_by_8":     ("combat", "kill_by_turn_rate", "8"),
    "kill_by_10":    ("combat", "kill_by_turn_rate", "10"),
    "board_power_6": ("combat", "mean_board_power_by_turn", "6"),
    "damage_10":     ("combat", "mean_damage_by_turn", "10"),
}


def run(slug, iterations=None, seed=None, max_turn=None,
        model_treasures=None, model_combat=None, model_draw=None,
        model_sacrifice=None, with_results=False, branch=None, model_discard=None,
        doc=None, quiet=False, targets_override=None, model_colors=None,
        _band=True, _targets_doc=None):
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
        # A DECK FILE PROBLEM, not a reason to end the process.
        raise DeclarationError(f"No commander flagged in {slug}/cards.json")
    commander_cmc = int(commanders[0].get("cmc") or 0)

    # A branch inherits the deck's ENGINE DECLARATION unless it writes its own:
    # nobody authors a second targets file to try a candidate list, and measuring
    # a branch against no declaration would report a different deck rather than a
    # different list.
    # RESOLVED ONLY WHEN IT IS GOING TO BE READ. `_targets_doc` exists so a
    # caller can supply the declaration instead of a file, and `doc` so it can
    # supply the list — but this line ran regardless, so a caller supplying both
    # still needed a deck directory on disk to exist. `card-value` supplies both
    # and its unit tests supply neither, which is how that surfaced.
    targets_path = None
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
    declared_draw = False
    declared_sacrifice = False
    # AND DRAIN, which was bound only INSIDE the `targets_path.exists()` branch
    # while its four siblings were bound outside — the comment below explains
    # why they were lifted and drain was missed. Latent until something skipped
    # the file read: a deck with no goldfish_targets.json raised
    # UnboundLocalError, and so did the band's floor run, which supplies the
    # declaration directly instead of reading it.
    declared_drain = False
    declared_discard = False        # the same lesson, one flag later
    declared_deaths = None
    # Bound before the branch: a deck with no declaration still has colours,
    # and reading it only inside the `if` made every declaration-less deck
    # (which is the benchmark's whole fleet path) raise UnboundLocalError.
    targets_doc = {}
    # `_targets_doc` REPLACES the file read, flags and all. `targets_override`
    # only ever replaced the `targets` LIST, so it could not switch a declared
    # commander ability off — which is the one thing the band run needs.
    if _targets_doc is not None:
        targets_doc = _targets_doc
    else:
        targets_path = deck_file(slug, "goldfish_targets.json", branch)
        if targets_path.exists():
            with open(targets_path) as f:
                targets_doc = json.load(f)

    # UNCONDITIONAL ON THE DOCUMENT, NOT ON WHERE IT CAME FROM. The first cut of
    # the band put this body inside the `elif`, so a caller supplying the
    # declaration directly skipped every read in it — and `model_combat` /
    # `model_draw` fell back to False. The floor run was then measuring a
    # DIFFERENT MODEL rather than the same model with one ability off, which is
    # the whole claim the band makes. It read kill_by_8 0.102 against a true
    # floor of 0.219: the band would have blamed the ability for a gap that was
    # mostly combat being switched off.
    if targets_doc:
        # AN OVERRIDE MUST REACH THE SIMULATION, NOT JUST THE REPORT. The first
        # cut passed a modified declaration to the reporting layer while this
        # loop still read the file — so `target_turns` was indexed by the FILE's
        # targets and the override changed nothing. The tell was eight different
        # candidates returning the identical 0.501.
        targets = targets_override if targets_override is not None else targets_doc["targets"]
        declared_treasures = bool(targets_doc.get("model_treasures"))
        declared_combat = bool(targets_doc.get("model_combat"))
        declared_draw = bool(targets_doc.get("model_draw"))
        declared_sacrifice = bool(targets_doc.get("model_sacrifice"))
        declared_drain = bool(targets_doc.get("model_drain"))
        declared_discard = bool(targets_doc.get("model_discard"))
        # THE RATE MUST NAME WHERE IT CAME FROM. A death rate somebody invented
        # driving a damage figure is the deleted engine lift; a rate read off a
        # Forge run on this deck is evidence. `source` is REQUIRED, and the
        # value lands in the record's assumptions so a reader can go and check.
        declared_deaths = targets_doc.get("model_deaths") or None
        if declared_deaths:
            missing = [k for k in ("own_per_turn", "opponent_per_turn", "source")
                       if k not in declared_deaths]
            if missing:
                raise DeclarationError(
                    f"model_deaths is missing {', '.join(missing)} — a death rate "
                    f"without a `source` naming the run it was measured from is "
                    f"an authored number driving a damage figure, which is the "
                    f"failure `engine_online` was deleted for.")
            declared_deaths = {
                "own_per_turn": float(declared_deaths["own_per_turn"]),
                "opponent_per_turn": float(declared_deaths["opponent_per_turn"]),
                "source": str(declared_deaths["source"]),
            }
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
    # THE COMMANDER'S ATTACK TUTOR, declared per deck because it is one
    # commander's text rather than a rule of the format. Shape:
    #     "model_commander_attack_tutor": {"type": "Enchantment", "max_mv": 3}
    # `type` is matched as a SUBSTRING of the type line, so "Enchantment" also
    # matches an enchantment CREATURE — which is the point for Zur, whose whole
    # plan is fetching bodies. Absent means absent: no other deck's figures move.
    attack_tutor = targets_doc.get("model_commander_attack_tutor") or None
    if attack_tutor:
        # `fires_per_turn` AND `source` ARE REQUIRED. Without a rate this model
        # fired the tutor EVERY turn after the commander landed, and Forge — 60
        # games, piloting confirmed comparable — resolved Zur's search 1.22
        # times a game against the 5.70 that produced. A commander does not
        # attack every turn, and pretending otherwise inflated every figure
        # built on the tutor. `source` names the run the rate was read off, the
        # same contract `model_deaths` keeps.
        missing = [k for k in ("fires_per_turn", "fires_per_turn_when_enabled",
                              "source") if k not in attack_tutor]
        if missing:
            raise DeclarationError(
                f"model_commander_attack_tutor is missing {', '.join(missing)} — "
                f"an attack rate without a source is an authored number driving "
                f"every figure the tutor touches. Measure it from a sim run "
                f"(searches resolved / games / attack-window turns) and name it.")
        attack_tutor = {"type": str(attack_tutor.get("type") or "Enchantment"),
                        "max_mv": int(attack_tutor.get("max_mv", 3)),
                        "fires_per_turn": float(attack_tutor["fires_per_turn"]),
                        "fires_per_turn_when_enabled": float(
                            attack_tutor["fires_per_turn_when_enabled"]),
                        "source": str(attack_tutor["source"])}
    # THE COMMANDER'S COMBAT-DAMAGE REVEAL, declared per deck (Gishath, Sun's
    # Avatar is the one card). Shape:
    #     "model_commander_combat_reveal": {"type": "Dinosaur",
    #         "connects_per_attack": 0.70, "source": "<run>: attacks vs triggers"}
    # The rate is REQUIRED with its source, for the reason the attack tutor's
    # is: this model has no blockers and would connect every time.
    commander_reveal = targets_doc.get("model_commander_combat_reveal") or None
    if commander_reveal:
        missing = [k for k in ("type", "connects_per_attack", "source") if k not in commander_reveal]
        if missing:
            raise DeclarationError(
                f"model_commander_combat_reveal is missing {', '.join(missing)} — "
                f"a connect rate without a source is an authored number driving "
                f"every body the reveal puts onto the battlefield. Measure it from "
                f"a sim run (attacks assigned vs triggers resolved) and name it.")
        commander_reveal = {"type": str(commander_reveal["type"]),
                            "connects_per_attack": float(commander_reveal["connects_per_attack"]),
                            "source": str(commander_reveal["source"])}
    model_combat = declared_combat if model_combat is None else bool(model_combat)
    model_draw = declared_draw if model_draw is None else bool(model_draw)
    model_sacrifice = (declared_sacrifice if model_sacrifice is None
                       else bool(model_sacrifice))
    model_drain = declared_drain
    model_discard = declared_discard if model_discard is None else bool(model_discard)
    model_deaths = declared_deaths
    if model_deaths and not model_drain and not quiet:
        print("  WARNING model_deaths is set without model_drain: death triggers "
              "feed the drain channel, so nothing will be counted.")
    # LOUD, NOT SILENT. The drain half of this model is DAMAGE, and damage only
    # exists under `model_combat`. A deck that sets one flag and not the other
    # would otherwise get the draw and the Treasures and silently lose the
    # payoff it turned the flag on for — which is exactly the defect the arrival
    # channel shipped with and had to be found by measurement.
    if model_sacrifice and not model_combat and not quiet:
        print("  WARNING model_sacrifice is set without model_combat: death "
              "DRAIN is damage and there is no damage series without combat, "
              "so only the draw and Treasure halves will be read.")
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
    # The commander's own combat profile — it is a creature like any other and
    # was the only one the loop never put on the battlefield.
    commander_combat = combat_profile(commanders[0]) if commanders else None
    # A STATIC GRANT THE COMMANDER MAKES. Zur, Eternal Schemer gives every
    # ENCHANTMENT CREATURE deathtouch, lifelink and hexproof — and the lifelink
    # half is this deck's clock, because three cards turn life gained into life
    # lost. Read from the commander's own text; absent commanders and commanders
    # without a grant contribute nothing.
    commander_grants_lifelink_to = (
        drain_profile(commanders[0])["grants_lifelink_to"] if commanders else None)
    commander_event = event_payoffs(commanders[0]) if commanders else None
    # THE SECOND COMMANDER OF A PARTNER PAIR, as its own bundle. Everything
    # below this line that says `commanders[0]` is the first commander on
    # purpose: `commander_mean_cast_turn` stays the first's for every reader.
    partner = None
    if len(commanders) > 1:
        _p = commanders[1]
        partner = {"name": _p["name"], "cmc": int(_p.get("cmc") or 0),
                   "pips": cast_pips(front_field(_p, "mana_cost") or ""),
                   "type_line": _p.get("type_line") or "",
                   "subtypes": subtypes_of(_p.get("type_line") or "", _p.get("oracle_text") or ""),
                   "combat": combat_profile(_p), "event": event_payoffs(_p),
                   "grants_lifelink_to": drain_profile(_p)["grants_lifelink_to"]}
    # DECLARED, and required to name a cost and a scope. A commander ability
    # that only one card in the corpus has cannot be pattern-matched honestly;
    # it is the same contract `model_commander_attack_tutor` kept.
    commander_animate = targets_doc.get("model_commander_animate") or None
    if commander_animate:
        missing = [k for k in ("cost", "scope") if k not in commander_animate]
        if missing:
            raise DeclarationError(
                f"model_commander_animate is missing {', '.join(missing)}")
        commander_animate = {"cost": int(commander_animate["cost"]),
                             "scope": str(commander_animate["scope"]),
                             "exclude": str(commander_animate.get("exclude", "Aura"))}
    # THE COMMANDER COPIES A SPELL ACROSS YOUR OWN BOARD. Zada, Hedron Grinder
    # is the only card in the corpus with the ability, so it is declared per
    # deck like `model_commander_animate` — but unlike the attack tutor and the
    # combat reveal, IT REQUIRES NO AUTHORED RATE, and that is the point. The
    # copy count is the number of other creatures you control, which this model
    # already measures; there is no fires-per-turn to write down and therefore
    # no authored number driving the headline.
    #
    # THE FLAG IS CHECKED AGAINST THE COMMANDER'S OWN TEXT. A deck that sets it
    # on a commander without the ability gets an error rather than a silent
    # multiplier on every cantrip it casts.
    commander_copy = bool(targets_doc.get("model_commander_copy"))
    if commander_copy:
        if not commanders:
            raise DeclarationError(
                "model_commander_copy is set but the deck has no commander.")
        if not any(commander_copies_spells(c) for c in commanders):
            raise DeclarationError(
                f"model_commander_copy is set but "
                f"{commanders[0].get('name')!r} has no ability that copies a "
                f"single-target spell for each other creature you control. One "
                f"card in the corpus has it (Zada, Hedron Grinder); Agrus Kos, "
                f"Eternal Soldier copies only for Warriors and Soldiers, a "
                f"typed subset this model does not track.")
        if not model_draw and not quiet:
            print("  WARNING model_commander_copy is set without model_draw: "
                  "what the copies multiply is the spell's DRAW, so nothing "
                  "will be counted.")
    creature_types = _corpus_creature_types()
    chosen_type = chosen_type_for(doc["cards"])
    command_zone_reduction = []
    commander_subtypes = frozenset()
    commander_cast_token = None
    for c in commanders:
        got = cost_reduction(c, creature_types)
        if got:
            command_zone_reduction.append(got)
        commander_subtypes |= subtypes_of(c.get("type_line") or "",
                                          c.get("oracle_text") or "")
        commander_cast_token = commander_cast_token or cast_token_profile(c)

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
    # WHAT COUNTS AS INTERACTION IS DECK_AUDIT'S QUESTION AND IT ALREADY OWNS
    # THE ANSWER. `SUITE_ROLES` is removal + sweepers + protection + stax — the
    # wider "interactive suite" rather than the removal count, because the two
    # cards the pilot's log names are Deflecting Swat and Teferi's Protection
    # and both are protection. Imported lazily: deck_audit reaches for goldfish
    # figures and a module-level import would close the loop.
    interaction_names = frozenset()
    draw_unmodelled = []
    try:
        from manamap.pilot.common import load_card_roles
        from manamap.pilot.deck_audit import SUITE_ROLES
        roles = load_card_roles()
        suite = set(SUITE_ROLES)
        interaction_names = frozenset(
            c["name"] for c in doc.get("cards", [])
            if not c.get("is_commander") and suite & set(roles.get(c["name"], [])))
    except Exception:
        # A missing card_roles.json is not a reason to lose the whole run; the
        # series simply reports against an empty set and says so below.
        interaction_names = frozenset()
    for c in doc.get("cards", []):
        if c.get("is_commander"):
            continue
        d = draw_profile(dict(c, oracle_text=c.get("oracle_text") or ""))
        if d["unmodelled"]:
            draw_unmodelled.append(d["unmodelled"])
    restricted = sorted({
        f"{c['name']} ({produced_mana(c.get('oracle_text'), c.get('type_line'))})"
        for c in doc.get("cards", [])
        if "Land" not in (c.get("type_line") or "")
        and produced_mana(c.get("oracle_text"), c.get("type_line"))
        and _RESTRICTED_MANA_RE.search(c.get("oracle_text") or "")
    })
    if not model_treasures:
        visible = sorted({
            c["name"] for c in doc.get("cards", [])
            if not c.get("is_commander")
            and treasure_profile(c)[1] in ("upkeep", "landfall", "cast", "etb")
        })
        # `quiet` GATES IT, because a programmatic caller runs this function
        # once per card. `card-value` measures a 100-card deck by re-running
        # the whole simulation with each card blanked in turn, so an ungated
        # warning here printed 101 identical lines and buried the ranking it
        # was called to produce. The warning still fires on every run a person
        # asked for, which is every run that is not `quiet`.
        if visible and not quiet:
            print(f"  WARNING {slug} has {len(visible)} Treasure source(s) this model "
                  f"CAN simulate and `model_treasures` is not set in "
                  f"goldfish_targets.json, so they are ignored: {', '.join(visible)}")

    # Same contract as the Treasure warning above, one layer out: a combat
    # trigger whose EFFECT the parser cannot price scores zero, and a zero
    # nobody is told about reads as a fact about the deck.
    combat_unreadable = sorted({
        c["combat"]["unreadable"] for c in library if c["combat"]["unreadable"]
    }) if model_combat else []

    # THE DRAIN FIGURE IS A FLOOR AND HAS TO SAY SO. A death drain has no event
    # here (nothing dies), so Bastion of Remembrance and The Meathook Massacre
    # contribute zero — and a reader with no list of names cannot tell a deck
    # whose drain is small from one whose drain is unread. Same contract as
    # `draw_not_modelled` and `combat_effects_not_modelled`.
    # A CARD IS ONLY UNMODELLED IF NO CHANNEL READS IT. Both of these were
    # death triggers, so switching `model_deaths` on moves them out of this list
    # — and leaving them in would have told a reader the figure was a floor
    # because of the very cards it had just started counting.
    drain_unmodelled = sorted({
        c["drain"]["unmodelled"] for c in library
        if c["drain"]["unmodelled"]
        and not (model_deaths and (c["death"]["death_drain"]
                                   or c["death"]["gain_on_opponent_death"]))
    }) if model_drain else []

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
                              commander_combat=commander_combat,
                              commander_grants_lifelink_to=commander_grants_lifelink_to,
                              commander_animate=commander_animate,
                              command_zone_reduction=command_zone_reduction,
                              chosen_type=chosen_type,
                              commander_subtypes=commander_subtypes,
                              commander_cast_token=commander_cast_token,
                              attack_tutor=attack_tutor,
                              commander_reveal=commander_reveal if model_combat else None,
                              model_treasures=model_treasures,
                              model_combat=model_combat,
                              model_draw=model_draw,
                              model_sacrifice=model_sacrifice,
                              model_drain=model_drain,
                              model_deaths=model_deaths,
                              commander_copy=commander_copy,
                              interaction_names=interaction_names,
                              model_colors=model_colors,
                              commander_pips=commander_pips,
                              model_discard=model_discard, partner=partner,
                              commander_event=commander_event))
            t.advance()

    _metrics = aggregate(results, targets, max_turn, model_treasures,
                         model_combat, model_draw, model_sacrifice,
                         model_drain, attack_tutor, model_discard=model_discard,
                         partner=bool(partner),
                         commander_reveal=commander_reveal if model_combat else None)
    _band_doc = _ability_band(slug, branch, targets_doc, _metrics, iterations,
                              seed, max_turn, model_treasures, model_combat,
                              model_draw, model_sacrifice,
                              model_colors) if _band else None

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
            **({"partner": partner["name"]} if partner else {}),
            "model_assumptions": MODEL_ASSUMPTIONS + (
                TREASURE_ASSUMPTIONS if model_treasures else []) + (
                COMBAT_ASSUMPTIONS if model_combat else []) + (
                DISCARD_ASSUMPTIONS if model_discard else []),
            # RESTRICTED MANA IS COUNTED AS FREE, AND THE READER SHOULD KNOW.
            # `spend()` is a scalar, so it cannot represent "only to cast
            # Dragon spells". Delighted Halfling's legendary-only mana is very
            # nearly free in a Commander deck; Throne of Eldraine's four is not.
            # Same contract as the Treasure blind spots: the assumption is
            # NAMED rather than silently made or silently dropped.
            "card_advantage": {
                "cards_that_draw": draw_cards,
                "modelled": (draw_cards - len(draw_unmodelled)) if model_draw else 0,
                "why": ("ETB, spell, upkeep and arrival draw are modelled; "
                        "activated, X-based, death- and attack-triggered draw "
                        "are not, and the cards are named in "
                        "`draw_not_modelled`. Read the two numbers together: a "
                        "deck whose count is twelve and whose modelled figure "
                        "is one has almost no UNCONDITIONAL card advantage, "
                        "which is a finding about the deck."
                        if model_draw else
                        "one card per turn, always — see model_assumptions. "
                        "The understatement is proportional to this count, so "
                        "two decks with different counts are not directly "
                        "comparable on any speed figure."),
                **({"draw_not_modelled": sorted(draw_unmodelled)}
                   if model_draw and draw_unmodelled else {}),
            },
            # WHAT THE HELD-UP SERIES WAS MEASURED AGAINST. An empty set would
            # make both series read a flat zero, which is indistinguishable from
            # a deck that runs no interaction.
            "interaction_suite_counted": sorted(interaction_names),
            **({"restricted_mana_counted_as_free": restricted} if restricted else {}),
            **({"treasure_sources_not_modelled": unmodelled} if model_treasures else {}),
            **({"combat_effects_not_modelled": combat_unreadable}
               if model_combat and combat_unreadable else {}),
            **({"drain_not_modelled": drain_unmodelled}
               if model_drain and drain_unmodelled else {}),
            **({"death_rate": model_deaths} if model_deaths else {}),
        },
        "metrics": _metrics,
        # THE BAND. See `_ability_band`. Absent unless the deck declares an
        # ability the simulation cannot confirm.
        **({"commander_ability_band": _band_doc} if _band_doc else {}),
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


#: A declared commander ability the SIM CANNOT CONFIRM.
#: `model_commander_animate` and `model_commander_attack_tutor` are AUTHORED —
#: a human writes them into `goldfish_targets.json` because one card in the
#: corpus has the ability and no pattern can find it. The model then applies
#: them every turn it can afford to, and reports one number.
_DECLARED_ABILITIES = ("model_commander_animate", "model_commander_attack_tutor")


def _band_value(metrics, axis):
    """One BAND_ROWS figure out of a metrics document, or None."""
    block, key, turn = BAND_ROWS.get(axis, (None, None, None))
    if not block:
        return None
    got = (metrics.get(block) or {}).get(key)
    if turn and isinstance(got, dict):
        got = got.get(turn)
    return got if isinstance(got, (int, float)) else None


def _ability_band(slug, branch, targets_doc, ceiling, iterations, seed, max_turn,
                  model_treasures, model_combat, model_draw, model_sacrifice,
                  model_colors):
    """The same deck WITHOUT its declared commander abilities — the floor.

    WHY A BAND AND NOT A NUMBER. zur-enchantress declares
    `model_commander_animate` for Zur, Eternal Schemer's "{1}{W}: target non-Aura
    enchantment becomes a creature with power and toughness equal to its mana
    value". The goldfish activates it every turn it can afford. Forge's AI
    activated it FIVE TIMES ACROSS 119 GAMES — 5% and 3% of the two runs that
    played the right commander — because its evaluator cannot price a benefit
    with no immediate board change.

        kill_by_8      0.381 with the ability, 0.219 without
        kill_by_10     0.888 with, 0.745 without
        board_power_6  7.168 with, 6.313 without

    Forty-three percent of the headline kill figure came from an ability that
    fires in one game in twenty at a real table. TWENTY-FOUR zur branches were
    graded on `kill_by_8` before anybody measured that.

    So a deck that declares one gets a CEILING and a FLOOR instead of a figure
    that looks like a measurement. The truth is between them and neither
    instrument reaches it — which a reader can act on, where 0.381 alone is not.

    ABSENT when nothing is declared. A zero-width band on an ordinary commander
    is noise on every other deck's page, and this repo's rule is that a figure
    nobody measured must be missing rather than reported.

    THE FLOOR RUN IS NOT FREE: it doubles the deck's simulation time, ~4s to ~8s
    at ten thousand games. It runs only for the decks that declare an ability —
    two of ten today.
    """
    declared = [k for k in _DECLARED_ABILITIES if (targets_doc or {}).get(k)]
    if not declared:
        return None
    stripped = {k: v for k, v in (targets_doc or {}).items() if k not in declared}
    try:
        floor = run(slug, iterations=iterations, seed=seed, max_turn=max_turn,
                    model_treasures=model_treasures, model_combat=model_combat,
                    model_draw=model_draw, model_sacrifice=model_sacrifice,
                    branch=branch, quiet=True, model_colors=model_colors,
                    _band=False, _targets_doc=stripped)
    except Exception as exc:                       # pragma: no cover - defensive
        return {"abilities": declared,
                "unavailable": f"{exc.__class__.__name__} — the floor run failed"}
    rows = {}
    for axis in BAND_ROWS:  # dict iteration: the keys, in declaration order
        hi = _band_value(ceiling, axis)
        lo = _band_value(floor["metrics"], axis)
        if hi is None or lo is None:
            continue
        rows[axis] = {"ceiling": hi, "floor": lo, "owed_to_the_ability": round(hi - lo, 4)}
    return {
        "abilities": declared,
        "ceiling_is": ("the ability fires whenever the deck can afford it, which "
                       "is what this model does"),
        "floor_is": "the same 99 with the ability switched off",
        "why": ("Forge's AI fired zur's animate in 5% of games. Neither end is "
                "the truth; read the pair. A branch graded on one end is graded "
                "on an assumption."),
        "rows": rows,
    }


def _coverage_preflight(slug, branch):
    """Say what this model cannot see BEFORE it spends ten thousand games.

    Every expensive fidelity surprise on this bench — eminence, the token
    doublers, the fetchlands — was found after the run. Imported lazily because
    `model_coverage` imports this module.
    """
    try:
        from manamap.pilot import model_coverage

        line = model_coverage.headline(model_coverage.analyze(slug, branch))
    except Exception:                              # noqa: BLE001 - never block
        return
    if line:
        print(f"  {line}")


def main(args):
    branch = getattr(args, "branch", None)
    _coverage_preflight(args.slug, branch)
    # BRANCHED WRITE, UN-BRANCHED READ — the mirror of the defect
    # `resolve_out_path` documents, and it silently filed the CHAMPION's
    # measurement under the branch's name for as long as branches have existed.
    # On ur-dragon's treasure branch that understated the turn-10 hoard 5.29 ->
    # 1.32, a factor of four, in a file whose own `meta.decklist_sha256` said
    # which list it had really measured. Nothing read that field.
    # THE TERMINAL STILL EXITS; ONLY THE LIBRARY STOPPED DECIDING. `run` raised
    # `SystemExit` on a malformed declaration, and four commands call it in
    # process — so one deck's bad `goldfish_targets.json` killed a fleet sweep
    # at deck three.
    try:
        doc = run(args.slug, branch=branch)
    except DeclarationError as bad:
        raise SystemExit(str(bad))
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
    _print_band(doc.get("commander_ability_band"))


def _print_band(band):
    """Say the headline is a BAND, at the point the headline is printed.

    A reader who has to open the JSON to learn that `kill_by_8` rests on an
    ability Forge fires in one game in twenty will not open the JSON. Twenty-four
    zur branches were graded on the ceiling before anybody measured the floor.
    """
    if not band:
        return
    print(f"\n  BAND — this deck declares {', '.join(band['abilities'])}, which "
          f"this model applies\n  every turn it can afford. Read the pair; the "
          f"table is somewhere between.")
    if band.get("unavailable"):
        print(f"    floor unavailable: {band['unavailable']}")
        return
    print(f"    {'':<14}{'ceiling':>9}{'floor':>9}{'owed':>9}")
    for axis, row in band["rows"].items():
        print(f"    {axis:<14}{row['ceiling']:>9.3f}{row['floor']:>9.3f}"
              f"{row['owed_to_the_ability']:>+9.3f}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot goldfish <slug>`.")
