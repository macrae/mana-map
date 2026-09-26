"""ONE GAME. `simulate_once` — the goldfish's turn loop, moved out verbatim.

MOVED OUT OF `goldfish.py` ON 2026-09-13. It is 1,999 lines in one function
with 239 assigned locals and ten nested closures, and that is not tidied here:
this is a MOVE, proved by a byte-identical fleet re-measurement, and a move plus
a tidy is a change nobody can review. Decomposing the function itself is its own
task with its own proof.

WHAT IT IS. Deal seven, mulligan by `keepable`, then `max_turn` turns of: untap,
upkeep triggers, draw, land, cast by channel (rocks, tutors, bodies, payoffs),
combat, end step. Every figure the bench quotes about resource development comes
out of here, seeded, so two runs of the same list at the same seed are the same
games.

WHY IT IS THIS SHAPE. The channels are the model's whole epistemology: a card is
cast only if some casting loop SELECTS it, and a card matching no channel sits
in hand for ten turns while its profile says exactly what it would have done.
That failure was found five times in one session and only caught as a class on
the fourth — see `model_coverage.never_cast` and `docs/gotchas-bench.md`.
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
    ETB_CHAIN_LIMIT,
    GOLDFISH_MAX_MULLIGANS,
    GOLDFISH_OPPONENT_LIFE,
    GOLDFISH_POISON_TO_LOSE,
    WHEEL_MIN_HAND,
    can_pay,
    devotion_of,
    has_event_payoff,
    is_death_engine,
    is_etb_engine,
    reduced_cost,
)
from manamap.pilot.goldfish_library import X_DRAW_MIN, _target_met, keepable


def simulate_once(rng, library, commander_cmc, targets, max_turn,
                  model_treasures=False, model_combat=False, model_draw=False,
                  model_sacrifice=False, model_drain=False, model_deaths=None,
                  model_colors=False, commander_pips=None,
                  command_zone_reduction=(), chosen_type=None,
                  commander_subtypes=frozenset(), commander_combat=None,
                  commander_grants_lifelink_to=None,
                  commander_animate=None,
                  commander_cast_token=None, interaction_names=frozenset(),
                  attack_tutor=None, model_discard=False, partner=None,
                  commander_event=None, commander_reveal=None,
                  commander_copy=False):
    """One goldfish iteration. Returns a per-iteration result dict.

    `partner` is the second commander of a Partner pair as
    `{cmc, pips, combat, event, grants_lifelink_to}`; it is cast on its own
    curve after the first and arrives through the same door."""
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

    attack_tutor_fired = 0
    # THE SPELL COUNT. There was none before this, which is why STORM —
    # a mechanic defined entirely by the count — was unreadable, and why
    # per-cast damage and magecraft had nothing to fire on.
    spells_cast_total = 0
    per_cast_engines = []      # Guttersnipe: damage on a CAST, never a copy
    magecraft_engines = []     # Storm-Kiln Artist: fires on cast OR COPY
    cast_damage_engines = []       # Sarkhan's Unsealing: damage on a big creature cast
    reveal_fired = 0               # the declared combat-damage reveal (Gishath)
    reveal_bodies = 0
    land_mana_bonus = 0            # extra mana per land tapped (Mirari's Wake)
    attack_tutor_debt = 0.0
    attack_enabler_out = False
    tutor_enabled_turns = 0
    tutor_eligible_turns = 0
    # DEVOTION, and the gods waiting on it. `battlefield_pips` is one pip list
    # per nonland permanent in play; `pending_gods` holds the gods that have
    # RESOLVED but are not creatures yet. A god below its threshold is an
    # enchantment: no power, no attack, no block. It still contributes its own
    # pips to devotion, which is what lets a second god switch a first one on.
    battlefield_pips = []
    pending_gods = []
    mulligans = 0
    while not keepable(hand) and mulligans < GOLDFISH_MAX_MULLIGANS:
        mulligans += 1
        deck = library[:]
        rng.shuffle(deck)
        hand = deck[:7]
        deck = deck[7:]

    kept_hand_lands = sum(1 for c in hand if c["is_land"])
    # A STRICTER KEEP TEST, REPORTED AND NEVER ENFORCED. `keepable` asks only
    # "2-5 lands", which is the rule this model mulligans by and changing it
    # would restate every figure on every deck. This asks the question the
    # pilot's log actually raises about a two-land keep going fifth: COULD THIS
    # HAND HAVE DONE SOMETHING BY TURN THREE — is there a nonland card whose
    # cost the lands in this hand can reach by then?
    #
    # Lands in hand are capped at 3 because you get three land drops by turn 3,
    # and no draws are assumed: it is a property of the SEVEN CARDS KEPT, not a
    # forecast. That makes it a floor, like every other figure here.
    _reach = min(kept_hand_lands, 3)
    keep_can_act_by_t3 = any(
        (not c["is_land"]) and c["cmc"] <= _reach for c in hand)
    seen = {c["name"] for c in hand}

    lands_in_play = 0
    rock_production = 0
    # EMINENCE IS ON FROM TURN ONE. It works from the command zone, so it is
    # live before the commander is ever cast and cannot be removed — which is
    # exactly why leaving it unmodelled mispriced the whole deck rather than
    # just its late game. Reducers cast from hand are appended as they land.
    reductions = list(command_zone_reduction)
    team_damage_multiplier = 1
    etb_engines = []              # payoffs in play that fire on a creature entering
    # DRAW. `draw_engines` holds permanents that keep drawing — an upkeep
    # trigger or an arrival trigger; `drawn_extra_by_turn` is the series the
    # whole model was built for, and it is CUMULATIVE EXTRA cards: the one-a-turn
    # draw step is not in it, because every deck gets that and a series that
    # includes it hides the difference it exists to show.
    # INTERACTION HELD UP. Two series, because they answer different halves of
    # the same failure and the pilot's own diagnosis turned on which half it was:
    # "Deflecting Swat and Teferi's Protection sat in my hand uncast for the
    # entire game ... my mana was spent casting vampires, so there was never
    # open mana to hold up."
    #   _in_hand    — you drew it. A draw problem if this is low.
    #   _castable   — you drew it AND the turn ended with enough mana unspent to
    #                 cast it. A MANA problem if this is low while the first is
    #                 high, which is the structural claim the log makes.
    #
    # MEASURED AT THE END OF THE MAIN PHASE, which is the moment the decision is
    # actually made: you commit your mana to the board or you keep it up, and
    # you choose before combat. Extra combat phases are paid for BELOW this
    # point and attack triggers add mana below it too, so neither is counted —
    # on a deck with Aggravated Assault the float measured here is larger than
    # what survives the turn. edgar-vampires has no extra-combat effect, so the
    # figure is exact for it.
    #
    # AND IT IS A FLOOR AGAINST THE SPENDING POLICY: the model casts everything
    # it can afford, cheapest-first, every turn. A real pilot holding up two
    # mana would score higher. So a LOW figure says "this deck cannot afford to
    # hold up interaction while developing at full speed", which is the question
    # asked, and a high one is unambiguous good news.
    interaction_in_hand_by_turn = []
    interaction_castable_by_turn = []
    # SACRIFICE. `death_engines` are the payoffs in play; `free_sac_outlet` is
    # whether anything can convert a token for nothing. Both are needed — a
    # drain with no outlet never fires and an outlet with no payoff is a worse
    # board.
    # TOKEN DOUBLERS, MULTIPLICATIVE. Two doublers is x4 and not x3 — each
    # replaces the other's output, exactly as `treasure_multiplier` documents
    # for the Treasure side. Three doublers is x8, which is what this deck's
    # engine brief means by "the doublers turn one mint into four".
    token_multiplier = 1
    death_engines = []
    free_sac_outlet = False
    sacrifices = 0
    sacrifices_by_turn = []
    sac_cap_hits = 0
    draw_engines = []
    # THE WHEELS THAT ARE ALREADY ON THE TABLE. Held apart from `draw_engines`
    # because this one needs the BODY as well as the profile: a cost that says
    # "Sacrifice this creature" takes the body with it, and a draw profile has
    # no way to point at the battlefield entry it belongs to. One dict per
    # permanent: {"draw", "power", "is_creature"}.
    wheel_engines = []
    # A DRAW YOU BUY, and the stockpile of one-shot ones.
    #
    # `bought_draws` IS NOT `draw_engines` TWENTY LINES UP, and the first cut of
    # this channel named it that and silently shadowed it — `draw_engines` holds
    # permanents that draw on their own every upkeep, this holds permanents that
    # draw when you PAY. The collision emptied the recurring list at boot and
    # the model died on the first card that had one (`KeyError: recurring_draw`,
    # because the entries are shaped differently too). Worth the long name.
    #
    # `blood` is a count of Blood tokens, shaped like `treasures` above because
    # it is the same kind of thing — a resource that sits on the board until it
    # is spent, once.
    bought_draws = []
    blood = 0
    blood_by_turn = []
    # Permanents that pay when an artifact you control is sacrificed, and the
    # count of those sacrifices this turn. Reset per turn like `drawn_this_turn`,
    # because the payoff is per EVENT and the turn is the accounting period.
    artifact_sac_payoff_permanents = []
    artifact_sacs_this_turn = 0
    # CUMULATIVE, and returned: a channel whose events nobody can count is a
    # channel nobody can tell apart from a channel that never fires. This is
    # what showed the first placement of the Blood crack was inert.
    artifact_sacs = 0
    artifact_sacs_by_turn = []
    # EVERY DRAW AFTER THE DRAW STEP, TIMES THIS. Multiplicative across sources
    # the way the damage multiplier is, because the rules are: two Teferi's
    # Ageless Insights really do draw four.
    draw_multiplier = 1

    def _register_wheel(card):
        """ONE DOOR FOR A WHEEL ARRIVING ON THE BATTLEFIELD.

        Called from every loop that puts a permanent into play, for the same
        reason `_commander_arrives` exists: three separate registration sites
        for `draw_engines` is already one too many, and a fourth channel added
        at two of the three is how a card comes to work on some turns.
        """
        nonlocal blood, draw_multiplier
        if not (model_draw and model_discard):
            return
        # BLOOD IS MADE ON ARRIVAL, so it is credited here rather than in a
        # loop of its own — the docstring above says three registration sites
        # is already one too many, and a fourth channel added at two of them is
        # how a card comes to work on some turns and not others.
        #
        # `combat` and `unmodelled` triggers are deliberately NOT credited: a
        # Blood made by connecting is one a blocker can prevent and this model
        # has no blockers, which is the same reason the combat pillar is opt-in.
        # SUBSCRIPTED, NOT `.get`, and that is not style. `classify` emits
        # "blood" for every card, and `test_every_signal_the_model_sets_is_read_
        # by_something` is an AST sweep for SUBSCRIPTS — a `.get` is invisible
        # to it, so the channel would read as a flag the model sets and nothing
        # acts on, which is the exact failure that test exists to catch.
        _bn, _bt = card["blood"]
        if _bt in ("etb", "per_opponent", "spell"):
            blood += _bn
        if card["draw"]["draw_multiplier"] > 1:
            draw_multiplier *= card["draw"]["draw_multiplier"]
        if any(v for k, v in card["artifact_sac"].items() if k != "unmodelled"):
            artifact_sac_payoff_permanents.append(card["artifact_sac"])
        if card["draw"]["activated_draw"]:
            bought_draws.append({
                "draw": card["draw"],
                "power": card["combat"]["power"],
                "is_creature": card["combat"]["is_creature"]})
        if not card["draw"]["activated_wheel"]:
            return
        wheel_engines.append({
            "draw": card["draw"],
            "power": card["combat"]["power"],
            "is_creature": card["combat"]["is_creature"]})
    drawn_extra = 0
    drawn_extra_by_turn = []
    # THE DISCARD CHANNEL. `discarded` counts cards that left hand by a wheel
    # or a loot; `drawn_this_turn` counts EVERY draw this turn, the draw step
    # included, for the per-draw and second-draw payoffs. Under model_discard
    # only; a deck that does not opt in sees none of this.
    discarded = 0
    discarded_by_turn = []
    drawn_this_turn = 0
    event_payoff_permanents = []
    event_damage_by_turn = []
    counter_power = 0             # +1/+1 counters the payoffs put on their bodies
    # THE COMMANDERS' OWN SHARE OF THAT, tracked apart because it is the
    # question this deck is built to ask. Brallin takes a counter per DISCARD
    # and Shabraz one per DRAW, so a wheel pays them twice and the pair is a
    # clock the goldfish was computing and throwing away: `counter_power` was
    # folded into the swing at the attack step and never reported, so "how big
    # are they by turn six" had no answer. Identity, not equality — Chasm
    # Skulker's profile is byte-identical to Shabraz's and is not a commander.
    commander_counters = 0
    commander_counters_by_turn = []
    commander_event_objs = []
    partner_turn = None
    arrival_draw_used = set()     # ids of `once each turn` engines, per turn
    etb_damage = 0                # noncombat damage dealt this turn by those
    etb_chain_hits = 0            # times the chain guard stopped a cascade
    bodies_cum_bump = [0]         # tokens spawned by an ETB payoff, counted once
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
    # (power, turn_it_arrived, has_haste, own_damage_multiplier) — the fourth is
    # 2 for a double-striker and 1 otherwise, and is SEPARATE from the board-wide
    # `team_damage_multiplier` because the two have different scopes and stack.
    # Kept only under model_combat so the resource-only path allocates nothing.
    battlefield = []
    combat_engines = []           # per-attack triggers of creatures in play
    extra_combat_free = 0         # Scourge-style, one additional phase
    extra_combat_costs = []       # Aggravated Assault-style, buy each time
    opponent_life = GOLDFISH_OPPONENT_LIFE
    kill_turn = None
    # POISON is a second, independent clock on the same seat: ten counters and
    # the game ends whatever the life total says. `kill_by` records which clock
    # fired so a deck whose kills are poison is legible as one.
    opponent_poison = 0
    poison_by_turn = []
    kill_by = None
    # THE DRAIN PILLAR. `drain_permanents` is every profile on the battlefield
    # that gains life, drains, or pays off on gaining; `lifelink_power` is the
    # power of lifelink CREATURES, accumulated and never removed because nothing
    # dies here. Arrival counters are reset each turn.
    # Types some card in THIS deck counts ("X is the number of Shrines you
    # control"). A permanent of such a type is worth casting for the count
    # alone.
    scaled_types = {c["drain"]["scales_with"] for c in library
                    if c["drain"]["scales_with"]}
    scaled_types |= {c["combat"]["team_counters_scale_type"] for c in library
                     if c["combat"]["team_counters_scale_type"]}
    # A STANDING +N/+N ON EVERY CREATURE. Counters do not wear off, so this is a
    # bonus applied to the whole board and to everything that joins it later —
    # not a pump. `team_anthem_on_type` is the second half of Southern Air
    # Temple: one more counter on everything each time another Shrine lands.
    team_anthem = 0
    team_anthem_on_type = []
    # Who has been granted haste by permanents in play: "all", "nontoken",
    # "flying" or a creature type. Read at the attack step only.
    haste_grants = []
    drain_permanents = []
    # Permanents that pay when another of a named type lands ("whenever another
    # Shrine you control enters"). A one-shot ETB and a per-type trigger are
    # DIFFERENT from a per-turn one, and reading all three as recurring is what
    # inflated the drain figure before 2026-09-05.
    per_type_watchers = []
    # Cards that mint a creature token every time an enchantment enters.
    enchantment_token_engines = []
    # CAST-TOKEN ENGINES FROM THE 99, not just the commander. `cast_token_profile`
    # was computed for the commander alone, so Sigil of the Empty Throne parsed
    # correctly and was never asked for.
    cast_token_engines = []
    # DEATHS. `death_engines` already exists for the sacrifice channel; these
    # accumulate FRACTIONAL deaths so a measured rate like 0.187 per turn fires
    # a trigger every fifth or sixth turn rather than never.
    own_death_debt = 0.0
    opponent_death_debt = 0.0
    death_drains = []          # profiles that fire when OUR creature dies
    opponent_death_gains = []  # profiles that gain when THEIRS does
    lifelink_power = 0
    drain_by_turn = []
    # Type lines of nonland permanents on the battlefield, so a card whose X is
    # "the number of Shrines you control" can be given the real count.
    battlefield_types = []
    # Mana value of each nonland permanent, index-aligned with battlefield_types.
    # `model_commander_animate` turns a non-Aura enchantment into a body whose
    # power IS its mana value, so the two have to travel together.
    battlefield_mv = []
    # Index-aligned with `battlefield_mv`: the Room profile for a half-open Room,
    # None for everything else. A Room's mana value CHANGES on the battlefield
    # when its second door opens, which no other permanent in the game does.
    battlefield_rooms = []
    # Types a static grant gives lifelink to ("Enchantment creatures you control
    # have … lifelink"). A creature already counted for its own lifelink is not
    # counted twice.
    lifelink_granted_types = set()
    animated_idx = set()
    # Type line of each battlefield creature, index-aligned with `battlefield`.
    creature_types = []
    creature_flying = []          # index-aligned with `battlefield`, like the types
    mass_animate_threshold = 0
    damage_by_turn = []
    board_power_by_turn = []
    target_turns = [None] * len(targets)
    target_turns_unassisted = [None] * len(targets)
    tutor_ready_turns = []

    for turn in range(1, max_turn + 1):
        # A PUMP LASTS ONE TURN. `team_anthem` is permanent and a pump must
        # never join it: a one-turn grant read as a permanent one is this
        # file's documented trap (a saga back face read as a damage doubler,
        # and cutting it measured as a LOSS). Both reset here, every turn.
        spells_cast_this_turn = 0   # STORM reads this: copies = the count BEFORE it
        turn_pump = 0        # applies to EVERY attacker this turn
        turn_double_strike = False   # a spell granted it; ONE turn
        turn_power_mult = 1          # a spell doubled power; ONE turn
        spell_each_opponent = 0      # Chandra's Ignition, resolved at the swing
        flat_pump = 0        # a single-target pump with no copy ability
        enchantments_entered = 0
        # Rooms whose SECOND door opened this turn. Separate from
        # `enchantments_entered` because only Eerie reads it — see _EERIE_RE.
        rooms_unlocked = 0
        creatures_entered_this_turn = 0
        deaths_drained = 0      # damage from OUR creatures dying, this turn
        deaths_gained = 0       # life from THEIRS dying, this turn
        etb_drained = 0         # one-shot and per-type drain, this turn
        etb_gained = 0
        drawn_this_turn = 0
        # THE OPPONENT'S DRAWS, so the tax on them has an event to fire on.
        # ONE SEAT, the same convention the damage pillar already uses: Brallin
        # says "1 damage to EACH opponent" and this model counts it once
        # against a single 40-life opponent. So one draw step a turn, and a
        # wheel refills that one seat. A real table has three, which makes
        # every figure downstream of this a FLOOR and is stated as one.
        opponent_draws_this_turn = 1
        artifact_sacs_this_turn = 0
        if deck:
            drawn = deck.pop(0)
            hand.append(drawn)
            seen.add(drawn["name"])
            drawn_this_turn = 1

        land_index = next((i for i, c in enumerate(hand) if c["is_land"]), None)
        if land_index is not None:
            played = hand.pop(land_index)
            lands_in_play += 1
            # TRACKED UNCONDITIONALLY. `sources` was maintained only under
            # `model_colors`, and a colour-scaling producer READS it to size its
            # own output — so the flag was not a constraint at all. See the note
            # at the `scales_with_colors` branch below.
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

        etb_damage = 0
        bodies_cum_bump[0] = 0
        arrival_draw_used = set()

        def draw_n(n):
            """Take n off the top. The deck running out is a real outcome and
            is not an error: a goldfish that decks itself has answered the
            question about steam more loudly than any rate could.

            THE DRAW DOUBLER IS APPLIED HERE AND NOWHERE ELSE, which is what
            makes "except the first one you draw in each of your draw steps"
            exact: the draw step takes its card straight off the deck a hundred
            lines up and never calls this, so the one draw the card does not
            double is the one draw that cannot reach this multiplication.
            """
            nonlocal drawn_extra, drawn_this_turn
            for _ in range(int(n) * draw_multiplier):
                if not deck:
                    return
                got = deck.pop(0)
                hand.append(got)
                seen.add(got["name"])
                drawn_extra += 1
                drawn_this_turn += 1

        def discard_n(n, everything=False, shuffled=False):
            """Put n cards from hand into the yard. AUTHORED POLICY, stated in
            MODEL_ASSUMPTIONS: a wheel discards the whole hand, lands included;
            a loot pitches lands beyond the next drop first, then the most
            expensive nonland. A discarded card stays in `seen` -- assembly is
            "drawn by", and it was."""
            nonlocal discarded
            if everything:
                if not shuffled:
                    discarded += len(hand)
                hand.clear()
                return
            for _ in range(int(n)):
                if not hand:
                    return
                lands = [c for c in hand if c["is_land"]]
                if len(lands) > 1:
                    hand.remove(lands[-1])
                else:
                    pick = max((c for c in hand if not c["is_land"]),
                               key=lambda c: c["cmc"], default=hand[0])
                    hand.remove(pick)
                discarded += 1

        # Recurring draw engines already in play fire in the upkeep, BEFORE the
        # mana is spent, so a card drawn this way is castable this turn.
        if model_draw:
            for eng in draw_engines:
                if eng["recurring_draw"]:
                    draw_n(eng["recurring_draw"])

        pool = lands_in_play * (1 + land_mana_bonus) + rock_production
        # Reported WITHOUT the stockpile, so this series keeps meaning exactly
        # what it has always meant: repeatable mana per turn. Treasures are a
        # one-shot reserve and get their own series.
        mana_by_turn.append(pool)
        treasures_by_turn.append(treasures)
        treasure_online_by_turn.append(bool(treasure_engines))

        def creature_entered(power, arrived, haste=False, mult=1, depth=0,
                             toughness=1,
                             is_token=False, is_legendary=False, type_line="",
                             infect=False, toxic=0, flying=False):
            """ONE DOOR ONTO THE BATTLEFIELD, so every payoff fires every time.

            Casting a creature, a token being made and a copy being made are the
            same event to Terror of the Peaks — the model used to have three
            separate `battlefield.append` sites and no payoff at any of them.

            IT RECURSES ON PURPOSE. Miirym's copy is a Dragon entering, which
            fires Scourge and Tempest again and raises X for the next one; that
            compounding IS the deck. `ETB_CHAIN_LIMIT` stops it and the depth is
            reported, because a loop that terminates silently cannot be told
            from one that never ran.
            """
            nonlocal etb_damage, etb_chain_hits
            # The sixth field is the poison pair (infect, toxic). It rides IN
            # the entry rather than in a parallel list, because the sacrifice
            # site below rebuilds `battlefield` and a parallel list would not
            # follow it — `creature_types` already does not.
            # THE SEVENTH FIELD IS TOUGHNESS, and it rides IN the entry for the
            # same reason the poison pair does: the sacrifice site rebuilds
            # `battlefield`, and a parallel list would not follow it —
            # `creature_types` already does not and that drift is a known
            # defect. Every unpack below takes `*_` so the eighth field, when
            # it comes, breaks nothing.
            battlefield.append((power, arrived, haste, mult, is_token,
                                (infect, toxic), toughness))
            # INDEX-ALIGNED WITH `battlefield`, appended at the same one door, so
            # the two can never drift the way the zip that preceded this did.
            creature_types.append(type_line)
            creature_flying.append(flying)
            # BODIES INTO CARDS, on the same door the damage payoffs use.
            # The power condition is honoured in both directions: Welcoming
            # Vampire ("power 2 or less") draws off a 1/1 token, Garruk's
            # Uprising ("power 4 or greater") must not.
            if model_draw:
                for i, eng in enumerate(draw_engines):
                    n = eng["arrival_draw"]
                    if not n:
                        continue
                    lo, hi = eng["arrival_power_min"], eng["arrival_power_max"]
                    if lo is not None and power < lo:
                        continue
                    if hi is not None and power > hi:
                        continue
                    if eng["arrival_draw_once"]:
                        if i in arrival_draw_used:
                            continue
                        arrival_draw_used.add(i)
                    draw_n(n)
            if not model_combat or depth >= ETB_CHAIN_LIMIT:
                if depth >= ETB_CHAIN_LIMIT:
                    etb_chain_hits += 1
                return
            spawned = []
            for eng in etb_engines:
                # "another NONTOKEN Dragon you control enters" — a token copy
                # does not re-trigger the thing that made it. This is the brake
                # the rules already had, and without it the board compounds
                # without bound.
                if is_token and eng["etb_nontoken_only"]:
                    continue
                # A TYPED TRIGGER FIRES ON ITS TYPE. Tokens carry no type line
                # here and pass, so Lathliss's Dragons still fire Tempest.
                gate = eng["etb_type_gate"]
                if gate and not is_token:
                    want = chosen_type if gate == "chosen" else gate
                    if not want or want not in type_line:
                        continue
                if eng["etb_damage_self_power"]:
                    etb_damage += power
                if eng["etb_damage_fixed"]:
                    etb_damage += eng["etb_damage_fixed"]
                if eng["etb_life_loss_fixed"]:
                    # Corpse Knight. Same event, same cadence and same number as
                    # Impact Tremors above; it read as zero until the payload
                    # regex learned the second way of wording it.
                    etb_damage += eng["etb_life_loss_fixed"]
                if is_token and eng["token_created_life_loss"]:
                    # Mirkwood Bats, and only on CREATURE tokens, because those
                    # are the only tokens this model makes. Bats also fires on a
                    # Blood token and on a sacrifice, and neither exists here.
                    etb_damage += eng["token_created_life_loss"]
                if eng["etb_damage_count"]:
                    # X is "the number of Dragons you control". The board is
                    # counted whole rather than by subtype — exact in a deck
                    # whose creatures are Dragons, generous otherwise, and
                    # stated in model_assumptions.
                    etb_damage += len(battlefield)
                if eng["etb_copy"]:
                    # A COPY IS LEGENDARY UNLESS THE CARD STRIPS IT. Miirym says
                    # "except the token isn't legendary" and is played for
                    # exactly that; Flameshadow Conjuring does not, so a copy of
                    # a legendary creature dies to the legend rule before it
                    # does anything — and 12 of this deck's creatures are
                    # legendary.
                    if is_legendary and eng["etb_copy_keeps_legendary"]:
                        continue
                    # And it usually charges. "You may pay {R}" is a cost, not a
                    # formality: firing it free reported 130.91 damage at turn
                    # ten against a 56.43 baseline.
                    if eng["etb_copy_cost"] and not spend(eng["etb_copy_cost"]):
                        continue
                    spawned.append((power, haste, mult))
                elif eng["etb_token_bodies"]:
                    each = eng["etb_token_power"] // max(eng["etb_token_bodies"], 1)
                    # DOUBLED HERE TOO. A payoff that makes a token on arrival
                    # is a token-creation event like any other, and a doubler
                    # that missed this site would double the printed token
                    # makers and not the engine's own.
                    for _ in range(eng["etb_token_bodies"] * token_multiplier):
                        spawned.append((each, False, 1))
            for pw, hs, mt in spawned:
                bodies_cum_bump[0] += 1
                creature_entered(pw, arrived, hs, mt, depth + 1, is_token=True)

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

        def _commander_arrives(cprof, cevent):
            """ONE DOOR FOR A COMMANDER'S ARRIVAL, used by both of a pair."""
            nonlocal team_damage_multiplier, extra_combat_free
            if model_combat and cprof and cprof["team_haste"]:
                haste_grants.append(cprof["team_haste"])
            if model_combat and cprof and cprof["is_creature"]:
                creature_entered(cprof["power"], turn, cprof["haste"],
                                 2 if cprof["double_strike"] else 1,
                                 is_legendary=True, type_line=cprof["type_line"],
                                 flying=cprof["flying"],
                                 infect=cprof["infect"], toxic=cprof["toxic"],
                                 toughness=cprof["toughness"])
                if cprof["team_damage_multiplier"] > 1:
                    team_damage_multiplier *= cprof["team_damage_multiplier"]
                if any((cprof["attack_mana"], cprof["attack_damage"],
                        cprof["attack_treasure"], cprof["attack_draw"],
                        cprof["attack_token_bodies"],
                        cprof["attack_ping_per_attacker"])):
                    combat_engines.append(cprof)
                if cprof["extra_combat_free"]:
                    extra_combat_free += 1
                if cprof["extra_combat_cost"] is not None:
                    extra_combat_costs.append(cprof["extra_combat_cost"])
            if model_discard and has_event_payoff(cevent):
                event_payoff_permanents.append(cevent)
                commander_event_objs.append(cevent)

        def _cast_triggers(cmc, tl, power, is_creature):
            """What fires on the CAST itself, whatever door the card goes
            through: the mana-value and paid cast-draws (Up the Beanstalk,
            Lifecrafter's Bestiary) and power-gated cast damage (Sarkhan's
            Unsealing). Engines already in play only."""
            nonlocal pool, etb_damage
            if model_draw:
                for _eng in draw_engines:
                    if not _eng["cast_draw"]:
                        continue
                    hit = ((_eng["cast_draw_gate_mv"] and cmc >= _eng["cast_draw_gate_mv"])
                           or (_eng["cast_draw_cost"] and _eng["cast_draw_gate"]
                               and _eng["cast_draw_gate"] in tl))
                    if not hit:
                        continue
                    if _eng["cast_draw_cost"]:
                        if pool < _eng["cast_draw_cost"]:
                            continue
                        pool -= _eng["cast_draw_cost"]
                    draw_n(_eng["cast_draw"])
            if model_combat and is_creature:
                for _eng in cast_damage_engines:
                    if power >= _eng["cast_damage_power_min"]:
                        etb_damage += _eng["cast_damage"]

        def _magecraft(events):
            """Fire magecraft `events` times. CAST OR COPY — that wording is the
            whole reason this is separate from `per_cast_engines`: with Zada out
            and six other bodies, one {R} cantrip is SEVEN magecraft triggers,
            and Storm-Kiln Artist turns each into a Treasure."""
            nonlocal treasures
            if events <= 0 or not model_treasures:
                return
            for _eng in magecraft_engines:
                treasures += _eng["magecraft_treasure"] * events

        def _note_cast(card):
            """EVERY cast goes through here: the card leaves hand, the spell
            count moves, and the cast-triggered payoffs fire.

            There are twelve cast sites in this function and storm's whole value
            is the count, so a site that removes a card from hand without coming
            through here silently undercounts it. A structural test greps this
            source for `hand.remove` and fails if the call is not this helper —
            that is the only thing that keeps it correct as the file grows.
            """
            nonlocal spells_cast_this_turn, spells_cast_total, etb_damage
            hand.remove(card)
            spells_cast_this_turn += 1
            spells_cast_total += 1
            tl = card.get("type_line") or ""
            is_spell = "Instant" in tl or "Sorcery" in tl
            is_creature = "Creature" in tl
            # PER-CAST DAMAGE FIRES ON A CAST AND NEVER ON A COPY. "Whenever you
            # cast" is not "cast or copy": Guttersnipe, Firebrand Archer and
            # Kessig Flamebreather all say cast, so Zada's copies do not feed
            # them. That is a rules fact, not a modelling shortcut.
            if model_combat:
                for _eng in per_cast_engines:
                    gate = _eng["per_cast_damage_gate"]
                    if gate == "an instant or sorcery" and not is_spell:
                        continue
                    if gate == "a noncreature" and is_creature:
                        continue
                    etb_damage += _eng["per_cast_damage"]
            if is_spell:
                _magecraft(1)

        def _free_creature_enters(card):
            """A CREATURE PUT ONTO THE BATTLEFIELD WITHOUT BEING CAST (the
            declared reveal). The same door and the same registrations as a
            cast body, minus what a cast is: no cast-triggered tokens, no
            cast-draw, no spend. Devotion-gated gods are held like any other."""
            nonlocal bodies_cum, team_damage_multiplier, extra_combat_free
            nonlocal lifelink_power, token_multiplier, treasures
            combat = card["combat"]
            seen.add(card["name"])
            bodies_cum += card["creature_bodies"] if model_combat else card["bodies"]
            if card["reduces"]:
                reductions.append(card["reduces"])
            tl = card.get("type_line") or ""
            battlefield_pips.append(card["pips"])
            battlefield_types.append(tl)
            battlefield_mv.append(card["cmc"])
            battlefield_rooms.append(None)
            if model_draw:
                draw_n(card["draw"]["etb_draw"])
                if card["draw"]["etb_draw_per_type"]:
                    draw_n(sum(1 for tl_ in battlefield_types[:-1]
                               if card["draw"]["etb_draw_per_type"] in tl_))
            if card["drain"]["lifelink"] and combat["is_creature"]:
                lifelink_power += combat["power"]
            if card["drain"]["grants_lifelink_to"]:
                lifelink_granted_types.add(card["drain"]["grants_lifelink_to"])
            if model_combat and is_etb_engine(combat):
                etb_engines.append(combat)
            if combat["is_creature"]:
                if card["devotion_gate"]:
                    pending_gods.append((card, combat))
                else:
                    creature_entered(combat["power"], turn, combat["haste"],
                                     2 if combat["double_strike"] else 1,
                                     is_legendary="Legendary" in tl, type_line=tl,
                                     infect=combat["infect"], toxic=combat["toxic"],
                                     flying=combat["flying"],
                                     toughness=combat["toughness"])
            if combat["token_bodies"]:
                each = combat["token_power"] // max(combat["token_bodies"], 1)
                for _ in range(combat["token_bodies"] * token_multiplier):
                    creature_entered(each, turn, False, 1, is_token=True)
            if model_combat:
                if combat["team_haste"]:
                    haste_grants.append(combat["team_haste"])
                if combat["team_damage_multiplier"] > 1:
                    team_damage_multiplier *= combat["team_damage_multiplier"]
                if combat["extra_combat_free"]:
                    extra_combat_free += 1
                if combat["extra_combat_cost"] is not None:
                    extra_combat_costs.append(combat["extra_combat_cost"])
                if any((combat["attack_mana"], combat["attack_treasure"],
                        combat["attack_draw"], combat["damage_scales_with_treasure"],
                        combat["attack_damage"], combat["attack_token_bodies"],
                        combat["attack_ping_per_attacker"])):
                    combat_engines.append(combat)
            if model_draw and any(card["draw"][k] for k in
                                  ("recurring_draw", "arrival_draw", "cast_draw")):
                draw_engines.append(card["draw"])
            _register_wheel(card)
            if card["token_doubler"]:
                token_multiplier *= 2
            if model_treasures:
                if card["treasure_trigger"] in ("upkeep", "landfall"):
                    treasure_engines.append((card["treasure_n"], card["treasure_trigger"]))
                elif card["treasure_trigger"] == "etb":
                    treasures += (card["treasure_n"] + treasure_bonus) * treasure_multiplier

        # A REDUCER ON THE BATTLEFIELD DOES CUT THE COMMANDER'S COST. Eminence
        # says "OTHER Dragon spells", so it never pays for itself — but
        # Dragonlord's Servant takes {1} off The Ur-Dragon like any other Dragon
        # spell, and a nine-drop commander is exactly where that matters.
        if commander_turn is None and spend(
                reduced_cost(commander_card, reductions, chosen_type),
                commander_pips):
            commander_turn = turn
            # The commander is a nonland permanent and its own pips count
            # toward devotion — Zur is {1}{W}{U}{B}, one each of three colours.
            battlefield_pips.append(commander_pips or [])
            battlefield_types.append(commander_card.get("type_line") or "")
            battlefield_mv.append(commander_cmc)
            battlefield_rooms.append(None)
            # THE COMMANDER IS A STUB HERE, not a classified card — it carries
            # pips, cmc and subtypes and nothing else — so a static grant it
            # makes has to be threaded in explicitly, the same way its combat
            # profile already is.
            if commander_grants_lifelink_to:
                lifelink_granted_types.add(commander_grants_lifelink_to)
            _cast_triggers(commander_cmc, commander_card.get("type_line") or "",
                           commander_combat["power"] if commander_combat else 0,
                           bool(commander_combat and commander_combat["is_creature"]))
            # THE COMMANDER USED TO BE CAST AND THEN DROPPED. It set this flag,
            # spent the mana, and never joined the battlefield — so a 10/10
            # flier contributed no power, never attacked, and fired none of its
            # own triggers. On The Ur-Dragon that is an entire stated win
            # condition (commander damage) measured as zero, and it is also why
            # Hellkite Courser, whose whole text is "put a commander onto the
            # battlefield", read as a vanilla body.
            #
            # Commander tax, death and recasting stay out of scope and stay
            # named: it is cast once, it stays, which is the same generous
            # direction the rest of this model takes.
            _commander_arrives(commander_combat, commander_event)

        # THE PARTNER, cast on its own curve after the first commander and
        # arriving through the same door. Brallin, Skyshark Rider was never
        # cast, never on the battlefield and not in the library before this:
        # `commanders[0]` was the only commander the model knew, so half of a
        # Partner pair -- the discard half of sharknado -- scored zero however
        # well its trigger parsed.
        if partner and partner_turn is None and spend(
                reduced_cost({"is_commander": True, "is_creature": True,
                              "subtypes": partner.get("subtypes", frozenset()),
                              "cmc": partner["cmc"], "pips": partner["pips"]},
                             reductions, chosen_type),
                partner["pips"]):
            partner_turn = turn
            battlefield_pips.append(partner["pips"] or [])
            battlefield_types.append(partner.get("type_line") or "")
            battlefield_mv.append(partner["cmc"])
            battlefield_rooms.append(None)
            if partner.get("grants_lifelink_to"):
                lifelink_granted_types.add(partner["grants_lifelink_to"])
            _commander_arrives(partner["combat"], partner.get("event"))

        # THE COMMANDER'S OWN ATTACK TUTOR — an APPROXIMATION, declared per deck.
        #
        # Zur reads "whenever Zur attacks, search your library for an enchantment
        # card with mana value 3 or less, put it ONTO THE BATTLEFIELD". That is
        # the deck's entire engine and this model could not see it: every figure
        # for that deck counted only cards it DREW, so the axis the win condition
        # rides was measured as nothing. Leaving it unmodelled is not neutral —
        # it is a systematic understatement of exactly one deck's plan.
        #
        # WHAT IS MODELLED, AND HOW GENEROUSLY. From the turn AFTER the commander
        # lands (it must survive to attack), the best matching card is pulled
        # from the library and enough mana is added to the pool to pay for it, so
        # the normal casting loop resolves it through the ordinary path and every
        # arrival, ETB and body channel fires exactly as it would for a cast card.
        #
        # THIS IS OPTIMISTIC AND THE DIRECTION IS KNOWN. A goldfish has no
        # blockers, so the commander always attacks; at a real table it attacks
        # when the pilot judges it safe, and Forge's AI would not attack at all
        # (measured: Zur attacked in 17-47% of games where a human attacks every
        # turn, because the trigger fires on ATTACK and not on connect). So this
        # is a CEILING on the engine, not an estimate of it — and the ceiling is
        # the useful bound, because before this the floor was zero and nothing
        # else was available.
        #
        # "Best" is the highest mana value that fits the filter, which is the
        # crude part: a real pilot fetches for the board, not for the curve. It
        # is stated rather than hidden, and it is the same rule for every list
        # being compared.
        # THE COMMANDER DOES NOT ATTACK EVERY TURN, and modelling it as though
        # it does was worth a factor of nearly five.
        #
        # This fired once per turn from the turn after the commander landed.
        # Forge, 60 games on the standard pod with piloting confirmed
        # COMPARABLE, resolved Zur's search 73 times — 1.22 per game against
        # the 5.70 this model was reporting. The attack window is about 5.25 own
        # turns (games run ~10.3 own turns, first attack ~5.05), so the measured
        # rate is 0.232 fires per turn in that window.
        #
        # The reason is not a Forge quirk. Attacking with a 1/4 into a developed
        # board is a real decision with a real cost, and neither the AI nor a
        # person makes it every turn. Reconnaissance is the card that would
        # change the answer and it was discarded in all three games it appeared.
        #
        # SOURCED LIKE `model_deaths`: a rate driving a figure must name where it
        # was measured, or it is the deleted engine lift wearing a new hat.
        if attack_tutor and commander_turn is not None and turn > commander_turn:
            tutor_eligible_turns += 1
            if attack_enabler_out:
                tutor_enabled_turns += 1
            attack_tutor_debt += (attack_tutor["fires_per_turn_when_enabled"]
                                  if attack_enabler_out
                                  else attack_tutor["fires_per_turn"])
        while (attack_tutor and attack_tutor_debt >= 1.0
               and commander_turn is not None and turn > commander_turn):
            attack_tutor_debt -= 1.0
            # THE FIRST FETCH IS THE ENABLER, and modelling it as "take the
            # biggest thing" was wrong about the only decision this engine makes.
            #
            # A pilot whose commander has just attacked into an open board does
            # not go and get a four-power body. They get the one-mana Aura that
            # makes every FUTURE attack free, because the trigger is worth more
            # than anything it can fetch. Aqueous Form costs {U} and turns a
            # once-a-game trigger into a once-a-turn one.
            #
            # So: while no enabler is out, prefer one — cheapest, because it has
            # to be cast this turn to matter. After that, revert to the biggest
            # legal card, which is the right greed once attacking is free.
            match = None
            want_enabler = not attack_enabler_out
            for i, cand in enumerate(deck):
                if cand["cmc"] > attack_tutor["max_mv"]:
                    continue
                tl = cand["type_line"] or ""
                if attack_tutor["type"] not in tl:
                    continue
                if want_enabler:
                    if not cand["attack_enabler"]:
                        continue
                    if match is None or cand["cmc"] < deck[match]["cmc"]:
                        match = i
                elif match is None or cand["cmc"] > deck[match]["cmc"]:
                    match = i
            if match is None and want_enabler:
                for i, cand in enumerate(deck):
                    if cand["cmc"] > attack_tutor["max_mv"]:
                        continue
                    if attack_tutor["type"] not in (cand["type_line"] or ""):
                        continue
                    if match is None or cand["cmc"] > deck[match]["cmc"]:
                        match = i
            if match is not None:
                fetched = deck.pop(match)
                hand.append(fetched)
                pool += reduced_cost(fetched, reductions, chosen_type)
                attack_tutor_fired += 1

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
                _note_cast(card)

        # AN ETB PAYOFF THAT IS NOT A BODY falls through every other loop —
        # Dragon Tempest is an enchantment with `bodies` 0 and `produces` 0, so
        # it sat in hand for ten turns while being half of the deck's stated win
        # condition. Fourth card to find this hole, after Aggravated Assault,
        # Primal Vigor and the cost reducers. Cast early: its whole value is
        # what the creatures behind it are worth.
        if model_combat:
            for card in sorted((c for c in hand
                                if c["bodies"] == 0 and c["produces"] == 0
                                and not c["is_land"] and not c["tutor"]
                                and any((c["combat"]["etb_damage_self_power"],
                                         c["combat"]["etb_damage_count"],
                                         c["combat"]["etb_damage_fixed"],
                                         c["combat"]["etb_token_bodies"],
                                         c["combat"]["etb_copy"],
                                         # A DAMAGE DOUBLER THAT IS NOT A BODY
                                         # fell through here too — Gratuitous
                                         # Violence and Dictate of the Twin Gods
                                         # are enchantments, read correctly and
                                         # never cast.
                                         c["combat"]["team_damage_multiplier"] > 1,
                                         # A HASTE ENABLER IS AN ENGINE PERMANENT:
                                         # Fervor has no body, and without this
                                         # line it is read and never cast.
                                         c["combat"]["team_haste"],
                                         c["combat"]["cast_damage"],
                                         # Chandra's Ignition needs a body to aim.
                                         c["combat"]["spell_damage_greatest_power"] and battlefield))),
                               key=lambda c: c["cmc"]):
                if spend(reduced_cost(card, reductions, chosen_type), card["pips"]):
                    if is_etb_engine(card["combat"]):
                        etb_engines.append(card["combat"])
                    if card["combat"]["team_haste"]:
                        haste_grants.append(card["combat"]["team_haste"])
                    if card["combat"]["cast_damage"]:
                        cast_damage_engines.append(card["combat"])
                    if card["combat"]["spell_damage_greatest_power"]:
                        # THE BIGGEST BODY'S POWER, once, to the one opponent;
                        # the wipe half has nothing to hit here.
                        etb_damage += max((p + team_anthem for p, *_ in battlefield), default=0)
                    if card["combat"]["team_damage_multiplier"] > 1:
                        team_damage_multiplier *= card["combat"]["team_damage_multiplier"]
                    _note_cast(card)

        # Cast rocks cheapest-first; they produce starting next turn.
        for card in sorted((c for c in hand if c["produces"] > 0 or c["land_mana_bonus"]),
                           key=lambda c: c["cmc"]):
            if spend(reduced_cost(card, reductions, chosen_type), card["pips"]):
                if card["reduces"]:
                    reductions.append(card["reduces"])
                # A LAND-MANA BONUS IS A ROCK THAT PAYS PER LAND, from next turn.
                land_mana_bonus += card["land_mana_bonus"]
                made, colors = card["produces"], card["colors"] or frozenset()
                if card["scales_with_colors"]:
                    # SNAPSHOT AT CAST, and deliberately the conservative end of
                    # the range: it counts the colours on the board the turn it
                    # resolves and never grows, so a Faeburrow Elder cast on two
                    # colours and living to see five is UNDERSTATED. Understating
                    # is recoverable; overstating is how a mana base comes out
                    # looking fine and cannot cast its spells.
                    #
                    # AND IT READS `sources`, WHICH IS WHY THEY ARE NOW TRACKED
                    # UNCONDITIONALLY (#35, 2026-09-13). `sources` used to be
                    # appended to only under `model_colors`, so with the flag OFF
                    # this branch found an EMPTY list and every colour-scaling
                    # producer fell to `max(1, 0)` = one mana. Bloom Tender and
                    # Faeburrow Elder made 1 blind and up to 5 under the flag.
                    #
                    # So `model_colors` added a castability PENALTY and unlocked
                    # a production BONUS at the same time, and on a five-colour
                    # deck the bonus won: turning on a CONSTRAINT made ur-dragon
                    # read BETTER — `[0.219, 0.319]`, a move of +0.100 against an
                    # SE of 0.022. `test_a_mono_colour_deck_is_barely_affected_
                    # and_a_five_colour_one_is` caught it because a
                    # constraint-only flag cannot raise a cast rate.
                    colors = frozenset().union(*sources) if sources else frozenset()
                    made = max(1, min(len(colors), 5))
                rock_production += made
                # A rock adds as many sources as it makes mana — tracked always,
                # for the same reason the land above is.
                sources.extend([colors] * made)
                _note_cast(card)

        # Cast tutors before bodies: a tutor is a setup spell, and it competes
        # for the same mana. Previously tutors had bodies=0 and produces=0, so
        # they were never cast at all and their mana silently went to creatures.
        for card in sorted((c for c in hand if c["tutor"]), key=lambda c: c["tutor_cmc"]):
            if card["tutor_needs_body"] and bodies_cum < 1:
                continue
            if card["tutor_to_battlefield"] is not None and (model_combat or model_draw):
                # SAVAGE ORDER: sacrifice a 4-power body, the best creature of
                # the named type enters from the library. Held until a
                # nontoken body of power 4+ is on the board, and until the
                # library holds one to fetch.
                _fodder = [i for i, (p, _a, _h, _m, _tok, *_) in enumerate(battlefield)
                           if not _tok and p >= 4]
                if card["tutor_needs_body"] and not _fodder:
                    continue
                _want = card["tutor_to_battlefield"]
                _pick = max((c for c in deck if "Creature" in (c.get("type_line") or "")
                             and (not _want or _want in (c.get("type_line") or ""))),
                            key=lambda c: c["combat"]["power"], default=None)
                if _pick is None:
                    continue
                if not spend(card["tutor_cmc"], card["pips"]):
                    continue
                _note_cast(card)
                if card["tutor_needs_body"]:
                    _i = min(_fodder, key=lambda i: battlefield[i][0])
                    battlefield.pop(_i); creature_types.pop(_i); creature_flying.pop(_i)
                deck.remove(_pick)
                _free_creature_enters(_pick)
                tutor_ready_turns.append(turn)
                continue
            if not spend(card["tutor_cmc"], card["pips"]):
                continue
            _note_cast(card)
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
                    _note_cast(card)
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
                _note_cast(card)
                if card["treasure_doubler"]:
                    treasure_multiplier *= 2
                if card["treasure_bonus"]:
                    treasure_bonus += 1

        # THE WHEEL THAT IS ALREADY ON THE TABLE, AND WHEELS AGAIN NEXT TURN.
        #
        # Jace's Archivist reads `{U}, {T}: Each player discards their hand,
        # then draws cards equal to the greatest number a player discarded` —
        # every turn, forever, for one blue mana. The model saw a 2/2 Vedalken
        # Wizard with no text, because `wheel_draws` is credited on an Instant
        # or a Sorcery only and a permanent's copy of that sentence is an
        # ACTIVATED ability. Three of sharknado's twelve wheel-shaped cards were
        # invisible this way on a deck whose whole plan is wheeling.
        #
        # FIRED HERE, and the position is doing three jobs:
        #
        #   * BEFORE the draw spells, so what a wheel finds is castable on the
        #     turn it is found. Wheel, then spend, is the line a pilot takes.
        #   * AFTER `pool` is set, because unlike an upkeep trigger this one
        #     costs mana and has to compete for it like everything else.
        #   * BEFORE the loops that put permanents into play, which is what
        #     gives a `{T}` ability its summoning sickness for free: a wheel
        #     joins `wheel_engines` inside this turn's casting loops, which run
        #     below, so the earliest it can fire is the turn after it lands. No
        #     tapped state is tracked and none is needed.
        #
        # THE AUTHORED GATE IS THE SPELL'S, minus its off-by-one. `WHEEL_MIN_HAND`
        # counts the nonlands that would be thrown away; the spell subtracts
        # itself from that count because it is in the hand being emptied, and
        # this one is on the battlefield. Without the gate the Archivist wheels
        # away a good hand every turn from turn four, forever, and reports it
        # as card advantage.
        # ── CRACK A BLOOD ─────────────────────────────────────────────────
        #
        # BEFORE THE CASTING LOOPS, WHICH IS WHERE THE DECISION ACTUALLY SITS.
        # The first cut put this last, on whatever mana nothing else wanted, by
        # analogy with the X spells — and that made the channel INERT: a
        # goldfish spends its whole pool casting, so `spend(1)` failed almost
        # every turn and the tokens sat on the board uncracked. Measured at the
        # time: 49 of 300 games had Blood standing at end of turn and virtually
        # none was ever spent. A resource the model can never spend is a
        # resource the model cannot price.
        #
        # ONE PER TURN, AND THAT IS AN AUTHORED NUMBER. A pilot cracks a Blood
        # to smooth a draw, not to empty the board; cracking every token the
        # pool could afford would spend a turn-eight deck's whole mana on
        # rummaging. One is the conservative reading — it understates a pilot
        # holding three with mana to spare, which is the direction every other
        # choice in this file takes — and it is stated in MODEL_ASSUMPTIONS
        # rather than buried here.
        #
        # A BLOOD CANNOT BE CRACKED WITH AN EMPTY HAND: discarding is part of
        # the cost, and a cost you cannot pay is an ability you cannot activate.
        # Without the guard the model draws a free card off every token whenever
        # it is hellbent — which is exactly when a wheel deck usually is.
        if model_draw and model_discard and blood > 0 and hand and spend(1, []):
            blood -= 1
            artifact_sacs_this_turn += 1        # the token IS an artifact
            artifact_sacs += 1
            discard_n(1)
            draw_n(1)

        if model_draw and model_discard and wheel_engines:
            for _we in list(wheel_engines):
                _wp = _we["draw"]
                if not (sum(1 for c in hand if not c["is_land"]) <= WHEEL_MIN_HAND
                        or event_payoff_permanents):
                    continue
                if not spend(_wp["activated_wheel_cost"],
                             _wp["activated_wheel_pips"]):
                    continue
                _held = len(hand)
                discard_n(0, everything=True,
                          shuffled=_wp["activated_wheel_shuffles"])
                _an = _held if _wp["activated_wheel"] < 0 else _wp["activated_wheel"]
                draw_n(_an)
                opponent_draws_this_turn += _an   # symmetrical, as above
                if not _wp["activated_wheel_once"]:
                    continue
                wheel_engines.remove(_we)
                # A COST THAT EATS ITS OWN SOURCE TAKES THE BODY WITH IT.
                # Magus of the Wheel and Whirlpool Warrior both say "Sacrifice
                # this creature", and leaving a 3/3 on the board after it has
                # been sacrificed is the over-credit that the sacrifice channel
                # already learned to avoid. Matched on power against a
                # NON-TOKEN entry — the same shape `tutor_needs_body` uses two
                # loops up, and the closest a profile can get to pointing at
                # its own battlefield row.
                if not (_wp["activated_wheel_sacs_self"] and _we["is_creature"]):
                    continue
                _own = [i for i, (_p, _a, _h, _m, _tok, *_) in enumerate(battlefield)
                        if _p == _we["power"] and not _tok]
                if _own:
                    _i = _own[0]
                    battlefield.pop(_i)
                    creature_types.pop(_i)
                    creature_flying.pop(_i)

        # A DRAW SPELL IS NEITHER A ROCK, A TUTOR, A BODY NOR AN ETB PAYOFF —
        # the fifth card to fall through every loop here, after Aggravated
        # Assault, Primal Vigor, the cost reducers and Dragon Tempest. Night's
        # Whisper is a sorcery with `bodies` 0 and `produces` 0, so it sat in
        # hand for ten turns while being the only unconditional card advantage
        # in edgar-vampires. Cast BEFORE bodies, cheapest first: a cantrip you
        # cast first can find the body, and one you cast last cannot.
        if model_draw:
            for card in sorted((c for c in hand if not c["is_land"]
                                and c["bodies"] == 0 and c["produces"] == 0
                                and not c["tutor"]
                                # cast_draw IS IN THE PREDICATE, in the same
                                # commit as the channel. A card the model reads
                                # and never casts is the failure this file has
                                # documented six times: Mesa Enchantress has no
                                # body, makes no mana and tutors nothing, so
                                # without this line she sits in hand for ten
                                # turns while her profile says what she'd draw.
                                and any((c["draw"]["spell_draw"],
                                         c["draw"]["etb_draw"],
                                         c["draw"]["recurring_draw"],
                                         c["draw"]["arrival_draw"],
                                         c["draw"]["cast_draw"],
                                         c["draw"]["spell_draw_greatest_power"],
                                         # A WHEEL IS A DRAW SPELL, and is in the
                                         # predicate in the same commit as the
                                         # channel -- cast under model_discard
                                         # only, since a wheel that draws seven
                                         # and discards nothing is the
                                         # over-credit the flag exists to prevent.
                                         model_discard and c["draw"]["wheel_draws"],
                                         # AND THE WHEEL THAT IS A PERMANENT.
                                         # Every card in that family so far is
                                         # a creature and is cast by the bodies
                                         # loop below, so this line is doing
                                         # nothing today -- it is here because
                                         # the next Memory Jar will have no
                                         # body, and the rule is that the
                                         # casting predicate ships in the same
                                         # commit as the channel rather than in
                                         # the session that notices.
                                         model_discard and c["draw"]["activated_wheel"],
                                         # AND THE TWO CHANNELS ADDED WITH THIS
                                         # SENTENCE. A rock that cashes itself
                                         # in for a card, and anything that
                                         # makes Blood. Both need only to reach
                                         # the battlefield: `_register_wheel`
                                         # picks them up from there, and
                                         # `never_cast` was taught the same
                                         # pair in the same commit. Unlike the
                                         # line above, these are NOT all
                                         # creatures -- Blood Fountain, Sanguine
                                         # Statuette and Mind Stone have no body
                                         # at all, so without this they are
                                         # read perfectly and never played.
                                         # A DRAW YOU BUY AND A DRAW DOUBLER ARE
                                         # DRAW CARDS, gated on `model_draw`
                                         # ALONE. Requiring `model_discard` too
                                         # was wrong and the fleet guard said so
                                         # within the hour: heliod runs
                                         # Alhammarret's Archive and Teferi's
                                         # Ageless Insight with model_draw ON and
                                         # model_discard OFF, so the model read
                                         # both doublers, priced both, and no
                                         # casting loop would ever put either on
                                         # the table.
                                         model_draw
                                         and (c["draw"]["activated_draw"]
                                              or c["draw"]["draw_multiplier"] > 1),
                                         # BLOOD NEEDS BOTH HALVES and keeps the
                                         # pair: cracking one is a discard AND a
                                         # draw, and a deck that has not opted
                                         # into discard cannot be paid for it.
                                         model_discard and model_draw
                                         and (c["blood"][1]
                                              in ("etb", "per_opponent", "spell")
                                              or any(v for k, v in c["artifact_sac"].items()
                                                     if k != "unmodelled"))))),
                               key=lambda c: reduced_cost(c, reductions, chosen_type)):
                # THE LOOP WALKS A SNAPSHOT OF THE HAND, and a wheel cast two
                # iterations ago emptied it: a card that has since been
                # discarded is not here to cast.
                if card not in hand:
                    continue
                if card["draw"]["wheel_draws"] and not (
                        sum(1 for c in hand if not c["is_land"]) - 1 <= WHEEL_MIN_HAND
                        or event_payoff_permanents):
                    continue          # the hand is worth more than seven fresh cards
                if card["draw"]["spell_draw_greatest_power"] and not battlefield:
                    continue          # draws nothing with no creature; held
                if not spend(reduced_cost(card, reductions, chosen_type), card["pips"]):
                    continue
                _note_cast(card)
                if card["draw"]["wheel_draws"]:
                    held = len(hand)
                    discard_n(0, everything=True, shuffled=card["draw"]["wheel_shuffles"])
                    _n = held if card["draw"]["wheel_draws"] < 0 else card["draw"]["wheel_draws"]
                    draw_n(_n)
                    # A WHEEL IS SYMMETRICAL AND THIS IS THE HALF THE MODEL
                    # NEVER SAW. "Each player draws seven" refills the opponent
                    # too, which is the cost of every wheel in this deck and
                    # the reason the tax cards exist. A "that many" wheel draws
                    # each player their OWN discard count and this model does
                    # not track the opponent's hand, so ours stands in for it.
                    opponent_draws_this_turn += _n
                    continue
                draw_n(card["draw"]["spell_draw"] + card["draw"]["etb_draw"])
                # THE COMMANDER COPIES IT FOR EACH OTHER CREATURE YOU CONTROL.
                # Zada, Hedron Grinder's whole deck: a {R} cantrip targeting
                # only her draws ONE card, and with six other bodies out it
                # draws SEVEN. `battlefield` includes the commander herself, so
                # the copy count is len(battlefield) - 1 and a lone commander
                # multiplies by nothing, which is correct.
                #
                # WHAT THIS DOES NOT MODEL, named rather than left implied:
                # the model has no removal, so once cast she stays, and the real
                # card is a 3-mana 3/3 that dies to everything. Every figure
                # this channel moves is therefore a CEILING on the games where
                # she survives, not a forecast. `model_combat` carries the pump
                # and evasion halves of the same copies; this arm is the draw.
                if (commander_copy and commander_turn is not None
                        and card["copy_fodder"]
                        and card["draw"]["spell_draw"]):
                    others = max(len(battlefield) - 1, 0)
                    if others:
                        draw_n(card["draw"]["spell_draw"] * others)
                if card["draw"]["spell_draw_greatest_power"]:
                    # Resolved against the board at cast; the anthem rides on
                    # every body the way it does at the attack step.
                    draw_n(max((p + team_anthem for p, *_ in battlefield), default=0))
                if model_discard and card["draw"]["spell_discard"]:
                    discard_n(card["draw"]["spell_discard"])
                if (card["draw"]["recurring_draw"] or card["draw"]["arrival_draw"]
                        or card["draw"]["cast_draw"]):
                    draw_engines.append(card["draw"])
                _register_wheel(card)

        # STORM. A copy for each spell cast BEFORE it this turn — so the spell
        # count IS the card, and until this commit there was no count to read.
        # Cast LAST in the main phase, because every earlier cast raises the
        # copy count and a storm spell cast first copies nothing.
        #
        # THE PAYOFF SHAPES a red deck actually has: a copy that deals damage
        # (Grapeshot) and a copy that makes bodies (Empty the Warrens). A storm
        # spell whose effect is neither still COUNTS as a cast and its copies do
        # nothing — understating, the documented safe direction.
        #
        # MAGECRAFT FIRES ON EVERY COPY, which is why the count matters twice:
        # Storm-Kiln Artist turns a five-copy Grapeshot into five more Treasures.
        for card in sorted((c for c in hand if c["spell_count"]["storm"]),
                           key=lambda c: -reduced_cost(c, reductions, chosen_type)):
            if card not in hand:
                continue
            sc = card["spell_count"]
            if not (sc["storm_damage"] or sc["storm_token_bodies"]):
                continue          # nothing this model can price; left in hand
            if not spend(reduced_cost(card, reductions, chosen_type), card["pips"]):
                continue
            copies = spells_cast_this_turn      # the count BEFORE this cast
            _note_cast(card)                    # ...which this then increments
            total = 1 + copies
            if model_combat and sc["storm_damage"]:
                etb_damage += sc["storm_damage"] * total
            if sc["storm_token_bodies"]:
                each = sc["storm_token_power"] // max(sc["storm_token_bodies"], 1)
                for _ in range(sc["storm_token_bodies"] * total * token_multiplier):
                    battlefield.append((each, turn, False, 1, True, (0, 0), 1))
                    creature_types.append("Creature — Goblin")
                    creature_flying.append(False)
                    bodies_cum += 1
            _magecraft(copies)                  # the cast itself fired already

        # A PUMP SPELL IS NOT A BODY, A DRAW, A TUTOR OR A ROCK — so it fell
        # through every loop above, which is why Haze of Rage and every other
        # pump in the corpus contributed nothing to any damage figure. THE
        # CASTING PREDICATE SHIPS IN THE SAME COMMIT AS THE CHANNEL; this file
        # has documented six cards read perfectly and never played.
        #
        # Cast LAST in the main phase and only with attackers already out: a
        # pump on an empty board does nothing, and a pilot holds it until the
        # swing. Most expensive first, because the biggest pump is the one you
        # want when the mana is there.
        if model_combat and battlefield:
            for card in sorted((c for c in hand if not c["is_land"]
                                and (c["combat"]["spell_pump_single"]
                                     or c["combat"]["spell_pump_team"]
                                     or c["combat"]["spell_counters"]
                                     or c["combat"]["spell_double_strike"]
                                     or c["combat"]["spell_power_multiplier"] > 1
                                     or c["combat"]["spell_power_to_each_opponent"]
                                     or c["combat"]["spell_extra_combat"])),
                               key=lambda c: -reduced_cost(c, reductions, chosen_type)):
                if card not in hand:
                    continue
                if not spend(reduced_cost(card, reductions, chosen_type), card["pips"]):
                    continue
                _note_cast(card)
                team = card["combat"]["spell_pump_team"]
                single = card["combat"]["spell_pump_single"]
                # THE TREASURES THE SPELL MAKES, credited on exactly the same
                # multiplication as the pump. Reckless Ransacking is {1}{R} for
                # +3/+2 AND a Treasure; copied across eight creatures that is
                # eight Treasures, which is what pays for the follow-up. Gated on
                # `model_treasures` like every other token of its kind.
                if model_treasures and card["combat"]["spell_treasure"]:
                    _tre = card["combat"]["spell_treasure"]
                    _mult = 1
                    if (commander_copy and commander_turn is not None
                            and card["copy_fodder"] and len(battlefield) > 1):
                        _mult = len(battlefield)      # the original plus one per other
                    treasures += _tre * _mult
                # PERMANENT +1/+1 COUNTERS, applied to the board itself rather
                # than to `turn_pump`, because they do not expire. Copied by a
                # Zada-style ability every creature gets its own counter — which
                # is why the smaller permanent number beats the bigger temporary
                # one over ten turns.
                if card["combat"]["spell_counters"]:
                    _n = card["combat"]["spell_counters"]
                    if (commander_copy and commander_turn is not None
                            and card["copy_fodder"] and len(battlefield) > 1):
                        for _i, _c in enumerate(battlefield):
                            battlefield[_i] = (_c[0] + _n,) + _c[1:]
                    elif battlefield:
                        battlefield[0] = (battlefield[0][0] + _n,) + battlefield[0][1:]
                # THE FOUR THAT ACTUALLY KILL. Each is ONE TURN and each is
                # multiplied by the copy ability — which for double strike and
                # power doubling means the whole BOARD, not one creature.
                _cbt = card["combat"]
                _copied = (commander_copy and commander_turn is not None
                           and card["copy_fodder"] and len(battlefield) > 1)
                if _cbt["spell_double_strike"] and (_copied or battlefield):
                    # Board-wide only when copied; on one creature it is already
                    # inside that creature's own `mult`, which this model applies
                    # per-attacker — so the uncopied case is deliberately not
                    # credited rather than credited to everybody.
                    if _copied:
                        turn_double_strike = True
                if _cbt["spell_power_multiplier"] > 1 and _copied:
                    turn_power_mult = max(turn_power_mult,
                                          _cbt["spell_power_multiplier"])
                if _cbt["spell_power_to_each_opponent"]:
                    # Every creature deals its own power to each opponent. This
                    # model tracks ONE seat, so one seat's worth is the sum of the
                    # board's power — once if uncopied, once per creature if not.
                    _base = sum(_c[0] + team_anthem for _c in battlefield)
                    spell_each_opponent += _base if _copied else (
                        battlefield[0][0] + team_anthem if battlefield else 0)
                if _cbt["spell_extra_combat"]:
                    # NOT MULTIPLIED BY THE COPY COUNT, and the first version was.
                    # This model has NO TAPPED STATE: attackers are chosen by
                    # `haste or arrived < turn`, so every combat phase swings the
                    # FULL board. Seize the Day untaps ONE creature per copy, so
                    # eight copies are eight extra combats of one attacker each —
                    # crediting eight full-board swings read 74.01 damage @10
                    # against a baseline of 21.36, a 3.5x that is the model's
                    # missing untap rule and not the card. One extra combat is the
                    # same generous-but-bounded convention `extra_combat_free`
                    # already uses for Aggravated Assault.
                    extra_combat_free += _cbt["spell_extra_combat"]
                if team:
                    turn_pump += team
                elif single:
                    # COPIED, IT IS A TEAM PUMP FOR THE TURN. Zada targets only
                    # herself, the copies target every other creature, so every
                    # body gets +N — which is how a go-wide cantrip deck kills.
                    # Uncopied it pumps exactly one creature, and that is a flat
                    # addition to the swing rather than a board-wide bonus.
                    if (commander_copy and commander_turn is not None
                            and card["copy_fodder"] and len(battlefield) > 1):
                        turn_pump += single
                    else:
                        flat_pump += single

        # A PERMANENT THAT ONLY DRAINS WAS NEVER CAST AT ALL.
        #
        # Every casting loop above this one selects on a channel: cards that
        # draw, cards that ramp, cards that make Treasure, and below, cards with
        # bodies. A card with none of those — Sanctum of Stone Fangs, Northern
        # Air Temple: not creatures, no draw, no mana — matched no loop and sat
        # in hand for ten turns.
        #
        # That is why zur-enchantress's two Shrines measured as EXACTLY nothing
        # when they were added: not because two Shrines are weak, but because
        # the model never put them on the battlefield. A synthetic library of
        # twenty of them drained 0.
        #
        # Cheapest-first, like every other loop here, and each arm guarded by
        # its own flag so a deck that has not opted in is byte-identical.
        #
        # THE FLEET AUDIT THAT FOUND THE REST. Sweeping every deck for cards
        # that feed an ACTIVE channel and match no casting predicate returned
        # four, on two decks, and both classes are here:
        #
        #   edgar-vampires  Ashnod's Altar, Altar of Dementia — free sacrifice
        #       outlets with no body. `free_sac_outlet` is set inside the BODIES
        #       loop, so an outlet that is not a creature could never register:
        #       the deck's engine ran on 2 of its 4 outlets and its redundancy
        #       read as half what it is.
        #
        #   zur-enchantress  Steel of the Godhead, Sheltered by Ghosts — Auras
        #       that GRANT lifelink. They feed the drain channel and matched
        #       nothing.
        def _engine_permanent(c):
            if c["is_land"] or c["bodies"] > 0:
                return False
            if model_drain and (any(c["drain"][k] for k in (
                    "payoff_equal", "payoff_fixed", "gain_recurring",
                    "gain_per_enchantment", "gain_per_creature",
                    "drain_recurring", "drain_per_enchantment"))
                    or c["drain"]["lifelink"]):
                return True
            # A DEATH ENGINE IS A PERMANENT TOO, and this is the third time
            # this exact gap has bitten: a card gains a modelled ability and the
            # CASTING predicate is not taught about it, so the model reads it
            # perfectly and never puts it on the table. The Meathook Massacre
            # went straight back into the never-cast bucket the moment its death
            # triggers started working.
            if model_deaths and (c["death"]["death_drain"]
                                 or c["death"]["gain_on_opponent_death"]):
                return True
            # AN ATTACK ENABLER IS A PERMANENT TOO — the FOURTH time this gap
            # has bitten, and the worst of them, because it was a DEADLOCK. Four
            # of zur-enchantress's six enablers matched no casting loop,
            # including both one-mana ones, so the only route to an enabler was
            # the tutor — which needs an attack, which needs an enabler. The
            # model had made the deck unable to start its own engine.
            if attack_tutor and c["attack_enabler"]:
                return True
            # MASS ANIMATION IS A BODY ENGINE, and a card that stands the whole
            # board up must be castable or it does nothing at all — the sixth
            # time this class has bitten.
            if model_combat and c["combat"]["mass_animate_threshold"]:
                return True
            # A DISCARD OR DRAW PAYOFF WITH NO BODY -- Improbable Alliance is
            # an enchantment that draws nothing and makes no mana -- is an
            # engine permanent, in the predicate in the same commit as the
            # channel, the eighth time this class has bitten.
            if model_discard and has_event_payoff(c.get("event")):
                return True
            # A CAST-TOKEN ENGINE IS A BODY ENGINE. Sigil of the Empty Throne
            # has no body of its own, makes no mana and draws nothing, so
            # without this it sits in hand while its profile says what it would
            # have minted — the seventh time this class has bitten.
            if model_combat and c["cast_token"]:
                return True
            # A PERMANENT WHOSE VALUE IS BEING COUNTED BY SOMETHING ELSE.
            # Sanctum of Tranquil Light does almost nothing on its own — its job
            # is to be a SHRINE, so that the two cards reading "X is the number
            # of Shrines you control" see a bigger number. It feeds no channel,
            # so nothing would ever cast it, so the count it exists to raise
            # stayed low and the whole package measured worse than it is.
            #
            # Derived from the deck, not declared: if any card here scales with
            # a type, a permanent of that type is worth playing.
            if model_drain and scaled_types and not c["is_land"]:
                tl = c["type_line"] or ""
                if any(s in tl for s in scaled_types):
                    return True
            return bool(model_sacrifice and c["sac_outlet"])

        if model_drain or model_sacrifice or model_deaths or attack_tutor:
            for card in sorted((c for c in hand if _engine_permanent(c)),
                               key=lambda c: reduced_cost(c, reductions, chosen_type)):
                if not spend(reduced_cost(card, reductions, chosen_type),
                             card["pips"]):
                    continue
                _note_cast(card)
                battlefield_pips.append(card["pips"])
                battlefield_types.append(card.get("type_line") or "")
                # THREE PARALLEL LISTS AND THIS LOOP FED TWO OF THEM. Every
                # permanent cast here — a drain payoff, a sac outlet, a death
                # engine, an attack enabler — grew `battlefield_types` while
                # `battlefield_mv` stood still, so the animate scan's
                # `zip(battlefield_types, battlefield_mv)` paired a type line
                # with ANOTHER card's mana value and then truncated at the
                # shorter list, hiding every later permanent from animation
                # entirely. It bites exactly the deck that has both halves:
                # zur-enchantress runs model_drain AND model_commander_animate.
                battlefield_mv.append(card["cmc"])
                battlefield_rooms.append(dict(card["room"], open=False)
                                         if card["room"] else None)
                if "Enchantment" in (card.get("type_line") or ""):
                    enchantments_entered += 1
                # CAST-TRIGGERED BODIES, same door and the same rule: engines
                # already in play only. `gate_kind` decides what is matched —
                # a CARD TYPE against the type line (Sigil: any enchantment) or
                # a creature SUBTYPE against the card's subtypes (eminence:
                # Vampire). Matching a card type against subtypes is what made
                # this channel invisible.
                if model_combat or model_draw:   # `arrivals_matter` is bound at the other door only
                    _tl2 = card.get("type_line") or ""
                    for _eng in cast_token_engines:
                        if (_eng["subtype"] in _tl2 if _eng["gate_kind"] == "type"
                                else _eng["subtype"] in card["subtypes"]):
                            _n = _eng["bodies"] * token_multiplier
                            for _ in range(_n):
                                creature_entered(_eng["power"], turn, False, 1,
                                                 is_token=True)
                            bodies_cum += _n
                if card["cast_token"]:
                    cast_token_engines.append(card["cast_token"])
                # CAST-TRIGGERED DRAW fires here because in this model a
                # permanent spell is CAST and ENTERS in the same step, so
                # this door is every cast of one. Engines already in play
                # only — a card does not trigger itself, and the registration
                # below happens after this loop for exactly that reason.
                if model_draw:
                    _tl = card.get("type_line") or ""
                    for _eng in draw_engines:
                        # gate first: a mana-value gate has no type gate
                        if (_eng["cast_draw_gate"] and not _eng["cast_draw_cost"]
                                and _eng["cast_draw"] and _eng["cast_draw_gate"] in _tl):
                            draw_n(_eng["cast_draw"])
                _cast_triggers(card["cmc"], card.get("type_line") or "",
                               card["combat"]["power"] if card["combat"]["is_creature"] else 0,
                               card["combat"]["is_creature"])
                if model_discard and has_event_payoff(card.get("event")):
                    event_payoff_permanents.append(card["event"])
                if model_drain:
                    if any(card["drain"][k] for k in (
                            "payoff_equal", "payoff_fixed", "gain_recurring",
                            "gain_per_enchantment", "gain_per_creature",
                            "drain_recurring", "drain_per_enchantment")):
                        drain_permanents.append(card["drain"])
                    # AN AURA GRANTS LIFELINK TO ONE CREATURE, so it is worth
                    # the power of the body you would put it on — the biggest
                    # one — and nothing at all with an empty board. Capped by
                    # total board power so two Auras cannot credit more life
                    # than the whole team can deal.
                    if card["drain"]["lifelink"] and battlefield:
                        best = max(p for p, *_ in battlefield)
                        total = sum(p for p, *_ in battlefield)
                        lifelink_power = min(lifelink_power + best, total)
                if model_sacrifice:
                    if is_death_engine(card["death"]):
                        death_engines.append(card["death"])
                    if card["sac_outlet"] == "free":
                        free_sac_outlet = True
                if model_deaths:
                    if card["death"]["death_drain"]:
                        death_drains.append(card["death"])
                    if card["death"]["gain_on_opponent_death"]:
                        opponent_death_gains.append(card["death"])
                if card["attack_enabler"]:
                    attack_enabler_out = True
                if card["combat"]["mass_animate_threshold"]:
                    # THE EASIEST THRESHOLD WINS, not the last one drawn. Two
                    # cards in the corpus mass-animate and they do it
                    # independently: Opalescence at 1 (unconditional) and
                    # Starfield of Nyx at 5. Either one being satisfied stands
                    # the board up, so holding both must not be WORSE than
                    # holding the unconditional one alone — which is exactly
                    # what a plain assignment did, silently, whenever Starfield
                    # happened to resolve second.
                    _mat = card["combat"]["mass_animate_threshold"]
                    mass_animate_threshold = (
                        _mat if not mass_animate_threshold
                        else min(mass_animate_threshold, _mat))
                if model_drain:
                    d_ = card["drain"]
                    # PAYS ONCE, ON ENTRY — scaled by its own named type if it
                    # names one, counted on the board it is joining.
                    if d_["drain_etb"] or d_["gain_etb"]:
                        x_ = 1
                        if d_["scales_with"]:
                            x_ = max(1, sum(1 for tl_ in battlefield_types
                                            if d_["scales_with"] in tl_))
                        etb_drained += d_["drain_etb"] * x_
                        etb_gained += d_["gain_etb"] * x_
                    # And every permanent of a named type that lands afterwards
                    # pays the ones already out.
                    for prof_ in per_type_watchers:
                        if prof_["per_type"] in (card.get("type_line") or ""):
                            etb_drained += prof_["drain_per_type"]
                            etb_gained += prof_["gain_per_type"]
                    if d_["per_type"]:
                        per_type_watchers.append(d_)
                # THE SPELL-COUNT ENGINES, registered where every other engine is.
                # A card read perfectly and never registered is this file's
                # documented failure; both registries are populated here.
                sc_ = card["spell_count"]
                if sc_["per_cast_damage"]:
                    per_cast_engines.append(sc_)
                if sc_["magecraft_treasure"]:
                    magecraft_engines.append(sc_)
                cb_ = card["combat"]
                if model_combat and cb_["team_haste"]:
                    haste_grants.append(cb_["team_haste"])
                if cb_["team_counters_etb"]:
                    n_ = cb_["team_counters_etb"]
                    if n_ < 0:            # X = the count of its own named type
                        ty_ = cb_["team_counters_scale_type"] or ""
                        n_ = sum(1 for tl_ in battlefield_types if ty_ and ty_ in tl_)
                    team_anthem += n_
                if cb_["team_counters_scale_type"]:
                    team_anthem_on_type.append(
                        (cb_["team_counters_scale_type"], cb_["team_counters_per_type"]))
                for ty_, per_ in team_anthem_on_type:
                    if ty_ in (card.get("type_line") or "") and cb_ is not card["combat"]:
                        team_anthem += per_

        # Spend what's left on bodies, cheapest-first.
        for card in sorted((c for c in hand if c["bodies"] > 0),
                           key=lambda c: reduced_cost(c, reductions, chosen_type)):
            if spend(reduced_cost(card, reductions, chosen_type), card["pips"]):
                bodies_cum += card["creature_bodies"] if model_combat else card["bodies"]
                _note_cast(card)
                # A BODY THAT ALSO DRAWS is the shape the pilot's Edgar refactor
                # is built on: "a vampire that draws is better than a sorcery
                # that draws — same effect, plus a body, plus an eminence
                # trigger".
                #
                # THE ENGINE IS REGISTERED AFTER ITS OWN ARRIVAL, and the first
                # cut of this registered it before. Welcoming Vampire is a 2/3
                # that draws "whenever one or more OTHER creatures you control
                # with power 2 or less enter" — its own power is 2, so it passed
                # its own gate and drew a card it does not draw. Deferred, and
                # appended below once `creature_entered` has fired.
                #
                # It is deferred for EVERY card and not only the ones worded
                # "other", which understates Tocasia's Welcome and its two
                # relatives by one draw apiece on the turn they land. That is
                # the direction every other choice in this file takes.
                pending_draw_engine = None
                if model_draw:
                    draw_n(card["draw"]["etb_draw"])
                    if card["draw"]["etb_draw_per_type"]:
                        # "for each OTHER Dinosaur": counted on the board it joins.
                        draw_n(sum(1 for tl_ in battlefield_types
                                   if card["draw"]["etb_draw_per_type"] in tl_))
                    if (card["draw"]["recurring_draw"]
                            or card["draw"]["arrival_draw"]
                            or card["draw"]["cast_draw"]):
                        pending_draw_engine = card["draw"]
                # Dragonlord's Servant and Dragonspeaker Shaman are bodies that
                # also reduce; from here on they pay for every Dragon behind them.
                if card["reduces"]:
                    reductions.append(card["reduces"])
                combat = card["combat"]
                # THE DOOR OPENS FOR EITHER MODEL, and it used to open only for
                # combat. `creature_entered` is where the arrival-draw channel
                # lives (Welcoming Vampire, Caretaker's Talent, Tocasia's
                # Welcome), and every call to it sat inside `if model_combat:` —
                # so a deck opting into `model_draw` ALONE lost three quarters
                # of its arrival draws and reported the smaller number without
                # saying anything. Measured on edgar-vampires: 1.264 extra cards
                # by turn ten with both flags, 0.323 with draw alone.
                #
                # A deck with `model_combat` on is byte-identical either way —
                # the disjunction is already true — and a deck with neither flag
                # never reaches here at all.
                arrivals_matter = model_combat or model_draw
                if not card["is_land"]:
                    battlefield_pips.append(card["pips"])
                    battlefield_types.append(card.get("type_line") or "")
                    battlefield_mv.append(card["cmc"])
                    battlefield_rooms.append(dict(card["room"], open=False)
                                             if card["room"] else None)
                    if "Enchantment" in (card.get("type_line") or ""):
                        enchantments_entered += 1
                    # CAST-TRIGGERED BODIES, same door and the same rule: engines
                    # already in play only. `gate_kind` decides what is matched —
                    # a CARD TYPE against the type line (Sigil: any enchantment) or
                    # a creature SUBTYPE against the card's subtypes (eminence:
                    # Vampire). Matching a card type against subtypes is what made
                    # this channel invisible.
                    if arrivals_matter:
                        _tl2 = card.get("type_line") or ""
                        for _eng in cast_token_engines:
                            if (_eng["subtype"] in _tl2 if _eng["gate_kind"] == "type"
                                    else _eng["subtype"] in card["subtypes"]):
                                _n = _eng["bodies"] * token_multiplier
                                for _ in range(_n):
                                    creature_entered(_eng["power"], turn, False, 1,
                                                     is_token=True)
                                bodies_cum += _n
                    if card["cast_token"]:
                        cast_token_engines.append(card["cast_token"])
                    # CAST-TRIGGERED DRAW fires here because in this model a
                    # permanent spell is CAST and ENTERS in the same step, so
                    # this door is every cast of one. Engines already in play
                    # only — a card does not trigger itself, and the registration
                    # below happens after this loop for exactly that reason.
                    if model_draw:
                        _tl = card.get("type_line") or ""
                        for _eng in draw_engines:
                            # gate first: a mana-value gate has no type gate
                            if (_eng["cast_draw_gate"] and not _eng["cast_draw_cost"]
                                    and _eng["cast_draw"] and _eng["cast_draw_gate"] in _tl):
                                draw_n(_eng["cast_draw"])
                    _cast_triggers(card["cmc"], card.get("type_line") or "",
                                   combat["power"] if combat["is_creature"] else 0,
                                   combat["is_creature"])
                    if any(card["drain"][k] for k in
                           ("payoff_equal", "payoff_fixed", "gain_recurring",
                            "gain_per_enchantment", "gain_per_creature",
                            "drain_recurring", "drain_per_enchantment")):
                        drain_permanents.append(card["drain"])
                    if model_discard and has_event_payoff(card.get("event")):
                        event_payoff_permanents.append(card["event"])
                    if card["drain"]["lifelink"] and combat["is_creature"]:
                        lifelink_power += combat["power"]
                    if card["drain"]["grants_lifelink_to"]:
                        lifelink_granted_types.add(
                            card["drain"]["grants_lifelink_to"])
                if model_combat and combat["enchantment_token_bodies"]:
                    enchantment_token_engines.append(combat)
                if model_combat:
                    # REGISTERED BEFORE IT ENTERS, and that is correct for the
                    # printed wording: Scourge of Valkas says "whenever THIS
                    # CREATURE or another Dragon you control enters", so it does
                    # see itself. Terror of the Peaks says "another", and its
                    # own entry deals nothing because the damage is the
                    # ENTERING creature's power and it is not another creature.
                    if is_etb_engine(combat):
                        etb_engines.append(combat)
                if arrivals_matter and combat["is_creature"]:
                    # A GOD RESOLVES AS AN ENCHANTMENT. It is held here and
                    # joins the battlefield on the turn its devotion is met,
                    # which may be this turn (its own pips count) or never.
                    if card["devotion_gate"]:
                        pending_gods.append((card, combat))
                    else:
                        creature_entered(
                            combat["power"], turn, combat["haste"],
                            2 if combat["double_strike"] else 1,
                            is_legendary="Legendary" in (card.get("type_line") or ""),
                            type_line=card.get("type_line") or "",
                            infect=combat["infect"], toxic=combat["toxic"],
                            flying=combat["flying"])
                        creatures_entered_this_turn += 1
                # EMINENCE MINTS ITS TOKEN ON THE CAST, from the command zone,
                # whether or not the commander has ever been cast. "Another"
                # is why the commander's own arrival does not trigger it — it
                # is not in this loop.
                if (arrivals_matter and commander_cast_token
                        and commander_cast_token["subtype"] in card["subtypes"]):
                    minted = commander_cast_token["bodies"] * token_multiplier
                    for _ in range(minted):
                        creature_entered(commander_cast_token["power"], turn,
                                         False, 1, is_token=True)
                    bodies_cum += minted
                # Creature tokens arrive with summoning sickness too, and they
                # arrive on the turn their maker resolved.
                if arrivals_matter and combat["token_bodies"]:
                    each = combat["token_power"] // max(combat["token_bodies"], 1)
                    for _ in range(combat["token_bodies"] * token_multiplier):
                        creature_entered(each, turn, False, 1, is_token=True)
                if model_combat:
                    if combat["team_damage_multiplier"] > 1:
                        team_damage_multiplier *= combat["team_damage_multiplier"]
                    if combat["extra_combat_free"]:
                        extra_combat_free += 1
                    if combat["extra_combat_cost"] is not None:
                        extra_combat_costs.append(combat["extra_combat_cost"])
                    if any((combat["attack_mana"], combat["attack_treasure"],
                            combat["attack_draw"], combat["damage_scales_with_treasure"],
                            combat["attack_damage"], combat["attack_token_bodies"],
                            combat["attack_ping_per_attacker"])):
                        combat_engines.append(combat)
                if pending_draw_engine is not None:
                    draw_engines.append(pending_draw_engine)
                _register_wheel(card)
                if card["token_doubler"]:
                    token_multiplier *= 2
                if model_sacrifice:
                    if is_death_engine(card["death"]):
                        death_engines.append(card["death"])
                    if card["sac_outlet"] == "free":
                        free_sac_outlet = True
                if model_deaths:
                    if card["death"]["death_drain"]:
                        death_drains.append(card["death"])
                    if card["death"]["gain_on_opponent_death"]:
                        opponent_death_gains.append(card["death"])
                if card["attack_enabler"]:
                    attack_enabler_out = True
                if card["combat"]["mass_animate_threshold"]:
                    # THE EASIEST THRESHOLD WINS, not the last one drawn. Two
                    # cards in the corpus mass-animate and they do it
                    # independently: Opalescence at 1 (unconditional) and
                    # Starfield of Nyx at 5. Either one being satisfied stands
                    # the board up, so holding both must not be WORSE than
                    # holding the unconditional one alone — which is exactly
                    # what a plain assignment did, silently, whenever Starfield
                    # happened to resolve second.
                    _mat = card["combat"]["mass_animate_threshold"]
                    mass_animate_threshold = (
                        _mat if not mass_animate_threshold
                        else min(mass_animate_threshold, _mat))
                if model_drain:
                    d_ = card["drain"]
                    # PAYS ONCE, ON ENTRY — scaled by its own named type if it
                    # names one, counted on the board it is joining.
                    if d_["drain_etb"] or d_["gain_etb"]:
                        x_ = 1
                        if d_["scales_with"]:
                            x_ = max(1, sum(1 for tl_ in battlefield_types
                                            if d_["scales_with"] in tl_))
                        etb_drained += d_["drain_etb"] * x_
                        etb_gained += d_["gain_etb"] * x_
                    # And every permanent of a named type that lands afterwards
                    # pays the ones already out.
                    for prof_ in per_type_watchers:
                        if prof_["per_type"] in (card.get("type_line") or ""):
                            etb_drained += prof_["drain_per_type"]
                            etb_gained += prof_["gain_per_type"]
                    if d_["per_type"]:
                        per_type_watchers.append(d_)
                # THE SPELL-COUNT ENGINES, registered where every other engine is.
                # A card read perfectly and never registered is this file's
                # documented failure; both registries are populated here.
                sc_ = card["spell_count"]
                if sc_["per_cast_damage"]:
                    per_cast_engines.append(sc_)
                if sc_["magecraft_treasure"]:
                    magecraft_engines.append(sc_)
                cb_ = card["combat"]
                if model_combat and cb_["team_haste"]:
                    haste_grants.append(cb_["team_haste"])
                if cb_["team_counters_etb"]:
                    n_ = cb_["team_counters_etb"]
                    if n_ < 0:            # X = the count of its own named type
                        ty_ = cb_["team_counters_scale_type"] or ""
                        n_ = sum(1 for tl_ in battlefield_types if ty_ and ty_ in tl_)
                    team_anthem += n_
                if cb_["team_counters_scale_type"]:
                    team_anthem_on_type.append(
                        (cb_["team_counters_scale_type"], cb_["team_counters_per_type"]))
                for ty_, per_ in team_anthem_on_type:
                    if ty_ in (card.get("type_line") or "") and cb_ is not card["combat"]:
                        team_anthem += per_
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
        # ── X SPELLS: the mana sink, cast LAST with whatever is left ─────
        #
        # Deliberately after every other loop, because that is how the card is
        # played and because it makes the channel CONSERVATIVE: an X spell here
        # can only ever spend mana nothing else wanted. A loop placed earlier
        # would have eaten the whole pool and starved the board.
        #
        # `cmc` is the fixed part (Scryfall counts {X} as zero) and
        # `x_draw_multiplier` is how many {X} symbols the cost carries, so
        # {X}{X}{U}{U} buys one card per two mana. Largest X first: with two in
        # hand the deck casts the one that draws more, and the second is left
        # for a later turn rather than cast for nothing.
        if model_draw:
            while True:
                _avail = pool + treasures
                _best, _bx = None, 0
                for _c in hand:
                    _m = _c["draw"]["x_draw_multiplier"]
                    if not _m:
                        continue
                    _x = (_avail - _c["cmc"]) // _m
                    if _x > _bx:
                        _best, _bx = _c, _x
                if _best is None or _bx < X_DRAW_MIN:
                    break
                _cost = _best["cmc"] + _bx * _best["draw"]["x_draw_multiplier"]
                if not spend(_cost, _best["pips"]):
                    break
                _note_cast(_best)
                draw_n(max(0, _bx - _best["draw"]["x_draw_discard"]))

        # ── BLOOD, AND THE DRAWS YOU BUY ────────────────────────────────
        #
        # LAST, ON WHAT IS LEFT, for the same reason the X spells above are:
        # a loop placed earlier eats the pool and starves the board, and
        # spending only the leftovers makes the channel CONSERVATIVE. A real
        # pilot cracks a Blood early to dig for a land; this model cannot, and
        # understates in the direction every other choice in this file does.
        #
        # WHY BLOOD IS WORTH A CHANNEL AT ALL, since it draws no cards on net:
        # one leaves the hand and one arrives, so the card economy is flat and
        # the EVENTS are the whole point. A deck whose commanders charge for a
        # discard and for a draw is paid twice for every token it cracks, which
        # is why this arrived for sharknado and not for anybody else.
        if model_draw and model_discard:
            for _de in list(bought_draws):
                _dp = _de["draw"]
                if _dp["activated_draw_discards"] and not hand:
                    continue
                if not spend(_dp["activated_draw_cost"], _dp["activated_draw_pips"]):
                    continue
                if _dp["activated_draw_discards"]:
                    discard_n(_dp["activated_draw_discards"])
                draw_n(_dp["activated_draw"])
                if _dp["activated_draw_sacs_self"] and not _de["is_creature"]:
                    # Mind Stone, Commander's Sphere, a Blood token: the cost
                    # ate an ARTIFACT, which is the event Jaws charges for.
                    artifact_sacs_this_turn += 1
                    artifact_sacs += 1
                if not _dp["activated_draw_once"]:
                    continue
                bought_draws.remove(_de)
                # A COST THAT EATS ITS OWN SOURCE TAKES THE BODY WITH IT --
                # the same correction the wheel loop above carries, and for
                # the same reason: Living Lectern says "Sacrifice this
                # creature" and leaving its body on the board is an over-credit
                # the sacrifice channel already learned to avoid.
                if not (_dp["activated_draw_sacs_self"] and _de["is_creature"]):
                    continue
                _own = [i for i, (_p, _a, _h, _m, _tok, *_) in enumerate(battlefield)
                        if _p == _de["power"] and not _tok]
                if _own:
                    _i = _own[0]
                    battlefield.pop(_i)
                    creature_types.pop(_i)
                    creature_flying.pop(_i)

        # THE DISCARD AND DRAW PAYOFFS, paid on the turn's events. Applied
        # OUTSIDE the combat gate, the way drain is, so Brallin's ping counts
        # under model_discard alone rather than dying silently inside
        # `if model_combat:` -- and multiplied by the team damage multiplier
        # explicitly, because it is damage. Counters accumulate as power the
        # swing adds while the commander is out (an approximation, stated).
        _evt_dmg = 0
        # WHAT THE SACRIFICES PAID. Folded into `_evt_dmg` so it is multiplied
        # by the team damage multiplier and checked against the kill exactly
        # like Brallin's ping, which is the same kind of number. Counters go to
        # `counter_power` the same way, and a draw off a sacrifice terminates by
        # construction: cracking a token is not caused by drawing.
        if model_discard and model_draw and artifact_sac_payoff_permanents \
                and artifact_sacs_this_turn:
            for _a in artifact_sac_payoff_permanents:
                _evt_dmg += _a["per_artifact_sac_damage"] * artifact_sacs_this_turn
                counter_power += _a["per_artifact_sac_counter"] * artifact_sacs_this_turn
                if _a["per_artifact_sac_draw"]:
                    draw_n(_a["per_artifact_sac_draw"] * artifact_sacs_this_turn)
        if model_discard and event_payoff_permanents:
            _disc_t = discarded - (discarded_by_turn[-1] if discarded_by_turn else 0)
            for _e in event_payoff_permanents:
                _evt_dmg += _e["per_discard_damage"] * _disc_t
                _evt_dmg += _e["per_draw_damage"] * drawn_this_turn
                # THE TAX ON THEIR HALF. Their draws are the event; our draw
                # off it is safe where the same field on our own draws is
                # refused as a Curiosity loop, because our draws do not cause
                # theirs.
                _evt_dmg += _e["per_opponent_draw_damage"] * opponent_draws_this_turn
                if _e["per_opponent_draw_our_draw"] and opponent_draws_this_turn:
                    draw_n(_e["per_opponent_draw_our_draw"] * opponent_draws_this_turn)
                if opponent_draws_this_turn >= 2:
                    _evt_dmg += _e["opponent_second_draw_damage"]
                    if _e["opponent_second_draw_our_draw"]:
                        draw_n(_e["opponent_second_draw_our_draw"])
                _ctr = (_e["per_discard_counter"] * _disc_t
                        + _e["per_draw_counter"] * drawn_this_turn)
                counter_power += _ctr
                if any(_e is _c for _c in commander_event_objs):
                    commander_counters += _ctr
                if drawn_this_turn >= 2:
                    _evt_dmg += _e["second_draw_damage"]
                    if _e["second_draw_token_power"]:
                        creature_entered(_e["second_draw_token_power"], turn, False, 1, is_token=True)
                        bodies_cum += 1
                for _pw, _n in ((_e["per_discard_token_power"], _disc_t),
                                (_e["per_draw_token_power"], drawn_this_turn)):
                    for _ in range(_n if _pw else 0):
                        creature_entered(_pw, turn, False, 1, is_token=True)
                        bodies_cum += 1
                if _e["per_discard_draw"] and _disc_t:
                    draw_n(_e["per_discard_draw"] * _disc_t)
            _evt_dmg *= team_damage_multiplier
            if _evt_dmg:
                opponent_life -= _evt_dmg
                if kill_turn is None and opponent_life <= 0:
                    kill_turn = turn
                    kill_by = "life"
        event_damage_by_turn.append(_evt_dmg)
        blood_by_turn.append(blood)
        artifact_sacs_by_turn.append(artifact_sacs)
        discarded_by_turn.append(discarded)
        # CUMULATIVE, like `discarded_by_turn` beside it: the size the pair has
        # reached, not what they gained this turn.
        commander_counters_by_turn.append(commander_counters)

        bodies_cum += bodies_cum_bump[0]
        bodies_by_turn.append(bodies_cum)
        drawn_extra_by_turn.append(drawn_extra)
        held = [c for c in hand if c["name"] in interaction_names]
        interaction_in_hand_by_turn.append(bool(held))
        interaction_castable_by_turn.append(
            any(c["cmc"] <= pool + treasures for c in held))

        # ── Combat step ────────────────────────────────────────────────────
        # Nothing blocks, so every creature that can attack does. Each combat
        # phase fires the attack triggers again, which is what makes an
        # additional combat phase worth more than its own power.
        if model_combat:
            # DOUBLE STRIKE IS PER CREATURE; the team multiplier is per board.
            # Kept apart because they have different scopes and stack: a
            # double-striker under Twinflame Tyrant deals its power four times.
            # THE ANTHEM IS PER CREATURE AND IS PART OF ITS POWER, so it
            # rides inside the double-strike multiplier exactly as printed power
            # does. A +6/+6 team on a double-striker swings twelve extra.
            # A GRANT REACHES A CREATURE THAT ARRIVED THIS TURN. `haste` on
            # the entry is the creature's own keyword; the grants are read
            # here, against the type line and flying that ride beside it.
            def _granted(tl_, fl_, tok_):
                for g_ in haste_grants:
                    if (g_ == "all" or (g_ == "nontoken" and not tok_)
                            or (g_ == "flying" and fl_)
                            or (g_ not in ("all", "nontoken", "flying") and g_ in tl_)):
                        return True
                return False
            attackers = [(((p + team_anthem + turn_pump) * turn_power_mult)
                          * (mult if mult > 1 else (2 if turn_double_strike else 1)), pz)
                         for (p, arrived, haste, mult, _tok, pz, *_), tl_, fl_
                         in zip(battlefield, creature_types, creature_flying)
                         if haste or arrived < turn
                         or (haste_grants and _granted(tl_, fl_, _tok))]
            # TWO CLOCKS FROM ONE SWING. An infect attacker's damage is poison
            # (702.90b) and never touches the life total; a toxic attacker
            # deals its damage AND adds N counters on connecting, which with
            # no blockers is every attack. Double strike rides in `mult` for
            # both, as printed.
            swing = sum(d for d, pz in attackers if not pz[0])
            # An uncopied single-target pump landed on ONE attacker.
            if flat_pump and any(not pz[0] for _d, pz in attackers):
                swing += flat_pump
            # Chandra's Ignition is NOT combat damage — it is dealt on resolution,
            # so it does not need an attacker and is not doubled by double strike.
            if spell_each_opponent:
                swing += spell_each_opponent
                spell_each_opponent = 0
            if counter_power and attackers and commander_turn is not None and turn > commander_turn:
                swing += counter_power
            swing_poison = (sum(d for d, pz in attackers if pz[0])
                            + sum(pz[1] for _d, pz in attackers))
            # The per-attacker ping is dealt BY the attacker, so an infect
            # attacker's ping is a counter and a plain one's is damage.
            ping = sum(e["attack_ping_per_attacker"] for e in combat_engines)
            ping_poison = ping * sum(1 for _d, pz in attackers if pz[0])
            ping_life = ping * sum(1 for _d, pz in attackers if not pz[0])
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
            poisoned = 0
            for _ in range(phases):
                if not attackers:
                    break
                bonus = treasures if any(
                    e["damage_scales_with_treasure"] for e in combat_engines) else 0
                dealt += swing + bonus + ping_life
                poisoned += swing_poison + ping_poison
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
                        if engine["attack_token_scales"]:
                            # As big as the best OTHER attacker (Ghalta and
                            # Mavren): the second-largest swing on the board.
                            _ranked = sorted((p for p, _pz in attackers), reverse=True)
                            each = _ranked[1] if len(_ranked) > 1 else 0
                        for _ in range(engine["attack_token_bodies"]):
                            creature_entered(each, turn, False, 1, is_token=True)
                        bodies_cum += engine["attack_token_bodies"]
                # THE DECLARED COMBAT-DAMAGE REVEAL (Gishath, Sun's Avatar): the
                # commander connects for its damage, that many cards are
                # revealed, every creature of the named type among them enters
                # through the one door, the rest go to the bottom. The connect
                # rate is MEASURED from a Forge run and named in the record;
                # this model has no blockers, so its own rate would be 1.0.
                if (commander_reveal and commander_turn is not None and attackers
                        and commander_combat and commander_combat["is_creature"]
                        and rng.random() < commander_reveal["connects_per_attack"]):
                    _n_reveal = ((commander_combat["power"] + team_anthem)
                                 * (2 if commander_combat["double_strike"] else 1))
                    _top, _rest = deck[:_n_reveal], []
                    del deck[:_n_reveal]
                    for _c in _top:
                        _tl = _c.get("type_line") or ""
                        if commander_reveal["type"] in _tl and "Creature" in _tl:
                            _free_creature_enters(_c)
                            reveal_bodies += 1
                        else:
                            _rest.append(_c)
                    rng.shuffle(_rest)
                    deck.extend(_rest)
                    reveal_fired += 1
            # THE REPLACEMENT EFFECT APPLIES LAST, to everything this deck
            # dealt — combat swings and the attack triggers alike, because
            # Twinflame Tyrant says "a source you control" and an attack
            # trigger is one.
            # ETB damage is NONCOMBAT and already happened this main phase, so
            # it is added before the multiplier rather than per combat phase —
            # Twinflame Tyrant says "a source you control", and a Terror trigger
            # is one, but it fires once per creature and not once per swing.
            # SACRIFICE, AFTER THE SWING IS SNAPSHOTTED. `attackers` was taken
            # above, so a token converted here has already attacked — this is a
            # conversion, not a trade against this turn's combat.
            if model_sacrifice and free_sac_outlet and death_engines:
                # THE GUARD IS ABOUT RE-ENTRANCY, NOT BOARD WIDTH (#34,
                # 2026-09-13). It was `n_sac >= SAC_LIMIT_PER_TURN`, which
                # STOPPED CONVERTING at twenty tokens — and twenty tokens is a
                # board Edgar genuinely reaches on turn ten under eminence with
                # Anointed Procession and Mondrak. Measured: lift the cap and
                # the hit rate goes 0.007 -> 0, with the per-turn series
                # unchanged, because nothing was ever runaway. The guard was
                # truncating a real board and calling it a loop.
                #
                # What its docstring names — "a death payoff that makes a token
                # is a loop" — is a genuine hazard and none of today's payoffs
                # do it: they drain, draw and make Treasure, none of which is a
                # battlefield entry. So the sweep iterates a SNAPSHOT of a list
                # it never appends to, and terminates structurally.
                #
                # `board_before` is the assertion that keeps it that way: if a
                # death payoff ever starts creating a creature, the battlefield
                # grows during the sweep and THAT is the runaway, caught by the
                # thing it actually is rather than by a number somebody tuned.
                board_before = len(battlefield)
                kept, kept_idx, n_sac = [], [], 0
                for idx_, entry in enumerate(battlefield):
                    is_tok = entry[4]
                    if not is_tok:
                        kept.append(entry)
                        kept_idx.append(idx_)
                        continue
                    n_sac += 1
                    for eng in death_engines:
                        # Life loss is damage here for the same reason the
                        # arrival channel says so: one opponent, 40 life.
                        etb_damage += eng["death_drain"]
                        draw_n(eng["death_draw"])
                        if model_treasures:
                            treasures += eng["death_treasure"]
                if len(battlefield) > board_before:
                    # A death payoff put something onto the battlefield while we
                    # were sacrificing. That is the loop the guard is named for.
                    sac_cap_hits += 1
                battlefield[:] = kept
                # THE PARALLEL LISTS FOLLOW THE REBUILD. They did not until
                # 2026-09-11, so after a sacrifice the typed lifelink grant
                # read another creature's type line, and the haste grants
                # below would have done the same.
                creature_types[:] = [creature_types[i] for i in kept_idx]
                creature_flying[:] = [creature_flying[i] for i in kept_idx]
                sacrifices += n_sac
            dealt += etb_damage
            dealt *= team_damage_multiplier
            opponent_life -= dealt
            damage_by_turn.append(dealt)
            # A damage doubler doubles the damage a source deals, and infect
            # converts what is dealt, so the multiplier applies to counters too.
            poisoned *= team_damage_multiplier
            opponent_poison += poisoned
            poison_by_turn.append(poisoned)
            # BOARD POWER IS ACTUAL POWER. A double-striker is not a bigger
            # creature, so the multiplier belongs to the damage series and
            # never to this one.
            # A TOKEN PER ENCHANTMENT THAT LANDED THIS TURN. Summoning-sick
            # like any other arrival, so it swells NEXT turn's swing — the same
            # call the attack-trigger tokens above already make.
            for eng in enchantment_token_engines:
                for _ in range(eng["enchantment_token_bodies"] * enchantments_entered):
                    creature_entered(eng["enchantment_token_power"], turn,
                                     False, 1, is_token=True)
                    bodies_cum += 1

            # A GOD SWITCHES ON when devotion reaches its threshold, and it is
            # checked AFTER the turn's permanents have resolved because they are
            # what moves devotion. It arrives with summoning sickness like any
            # other creature: `creature_entered` stamps this turn.
            if pending_gods:
                still = []
                for card, combat in pending_gods:
                    gate = card["devotion_gate"]
                    if devotion_of(battlefield_pips, gate["colors"]) >= gate["threshold"]:
                        creature_entered(
                            combat["power"], turn, combat["haste"],
                            2 if combat["double_strike"] else 1, is_legendary=True,
                            type_line=card.get("type_line") or "",
                            flying=combat["flying"])
                        if model_combat and combat["team_haste"]:
                            haste_grants.append(combat["team_haste"])
                    else:
                        still.append((card, combat))
                pending_gods[:] = still

            board_power_by_turn.append(
                sum(p for p, *_ in battlefield)
                + team_anthem * len(battlefield))
            if kill_turn is None and (opponent_life <= 0
                                      or opponent_poison >= GOLDFISH_POISON_TO_LOSE):
                kill_turn = turn
                kill_by = "life" if opponent_life <= 0 else "poison"

        # ── the commander animates an enchantment ───────────────────────────
        #
        # "{1}{W}: Target non-Aura enchantment you control becomes a creature in
        # addition to its other types and has base power and base toughness each
        # equal to its mana value." UNIQUE IN THE CORPUS — one card — so it is
        # DECLARED per deck like the attack tutor was, not pattern-matched.
        #
        # An animated permanent has been under your control since the turn began,
        # so it is NOT summoning sick and can attack the same turn. Biggest mana
        # value first, because power is mana value here and the ability is
        # repeatable but mana-limited.
        # ── unlocking a Room ────────────────────────────────────────────────
        #
        # CR 709.5e: paying a locked half's mana cost is a SPECIAL ACTION, taken
        # at sorcery speed in your own main phase with an empty stack — so it
        # belongs here, after the turn's casting and before animation, and it
        # spends from the same pool.
        #
        # It buys two things a card in hand does not. The permanent's mana value
        # becomes both doors combined (CR 709.5 + 202.3d), which is what the
        # animate scan below reads to set base power; and it FULLY UNLOCKS a
        # Room, which re-fires every Eerie payoff for no card from hand. Cheapest
        # door first, because opening two small ones beats opening one large one
        # when the Eerie trigger is the point.
        for _i, _room in sorted(
                ((i, r) for i, r in enumerate(battlefield_rooms)
                 if r and not r["open"]),
                key=lambda ir: ir[1]["unlock_cost"]):
            if not spend(_room["unlock_cost"], _room["unlock_pips"]):
                continue
            _room["open"] = True
            battlefield_mv[_i] = _room["full_mv"]
            rooms_unlocked += 1

            # WHAT THE DOOR ACTUALLY DOES, applied here and nowhere earlier.
            # 26 of the 30 Rooms in the corpus carry a "When you unlock this
            # door" clause and none of it was read: the model paid the unlock
            # cost and got a bigger animation target for it, which made an
            # 8-mana investment look like a body and nothing else.
            _on = _room.get("on_unlock") or {}
            # The same token convention the cast path uses: `token_power` is the
            # TOTAL across bodies, so a 6/6 Demon is (6, 1) and not (6, 6). Doing
            # this by hand was worth catching -- `combat_profile(card)["power"]`
            # is the CARD's power, which is 0 for a Room, so a 6/6 would have
            # entered as a 1/1.
            _uc = _on.get("combat") or {}
            if (model_combat or model_draw) and _uc.get("token_bodies"):
                _each = _uc["token_power"] // max(_uc["token_bodies"], 1)
                for _ in range(_uc["token_bodies"] * token_multiplier):
                    creature_entered(_each, turn, False, 1, is_token=True)
                    bodies_cum += 1
            _ud = _on.get("drain")
            if model_drain and _ud and any(
                    _ud[k] for k in ("payoff_equal", "payoff_fixed",
                                     "gain_recurring", "gain_per_enchantment",
                                     "gain_per_creature", "drain_recurring",
                                     "drain_per_enchantment")):
                drain_permanents.append(_ud)
            _udr = _on.get("draw") or {}
            if model_draw and _udr.get("etb_draw"):
                drawn_extra += _udr["etb_draw"]

        if commander_animate and commander_turn is not None:
            cost = commander_animate["cost"]
            while pool >= cost:
                best, best_mv = -1, 0
                for i, (tl_, mv_) in enumerate(zip(battlefield_types, battlefield_mv)):
                    if i in animated_idx:
                        continue
                    if commander_animate["scope"] not in tl_:
                        continue
                    if commander_animate.get("exclude", "Aura") in tl_:
                        continue
                    if "Creature" in tl_:      # already a body
                        continue
                    if mv_ > best_mv:
                        best, best_mv = i, mv_
                if best < 0 or best_mv <= 0:
                    break
                pool -= cost
                animated_idx.add(best)
                # "becomes a creature IN ADDITION TO ITS OTHER TYPES" — so an
                # animated enchantment is now an ENCHANTMENT CREATURE, and the
                # commander's own grant of deathtouch, lifelink and hexproof
                # covers it. That is the synergy, and it only works if the type
                # line travels with the body.
                creature_entered(int(best_mv), turn - 1, False, 1,
                                 type_line=battlefield_types[best] + " Creature")

        # ── mass animation ──────────────────────────────────────────────────
        #
        # Starfield of Nyx: once five or more enchantments are out, every OTHER
        # non-Aura enchantment is a creature with power equal to its mana value.
        # Free, static, and board-wide — the commander's {1}{W} without the
        # mana. Applied after the turn's permanents have resolved, because they
        # are what crosses the threshold.
        if model_combat and mass_animate_threshold:
            ench = sum(1 for tl_ in battlefield_types if "Enchantment" in tl_)
            if ench >= mass_animate_threshold:
                for i, (tl_, mv_) in enumerate(zip(battlefield_types, battlefield_mv)):
                    if i in animated_idx or "Aura" in tl_ or "Creature" in tl_:
                        continue
                    if "Enchantment" not in tl_ or mv_ <= 0:
                        continue
                    animated_idx.add(i)
                    creature_entered(int(mv_), turn - 1, False, 1,
                                     type_line=tl_ + " Creature")

        # ── deaths ──────────────────────────────────────────────────────────
        #
        # NOTHING DIED IN THIS SIMULATION UNTIL NOW, and that blanked two cards
        # in zur-enchantress outright: The Meathook Massacre and Bastion of
        # Remembrance, both named in `meta.drain_not_modelled`. In a deck with
        # eleven token-makers feeding them, that is not a rounding error.
        #
        # THE RATE IS MEASURED, NOT AUTHORED. `model_deaths` carries a
        # per-own-turn rate for our creatures and for the opponents', and a
        # `source` naming the Forge run it was read off. That distinction is the
        # whole design: a figure computed from a rate somebody invented is the
        # deleted engine lift again, where three defensible declarations of one
        # list gave +0.007, -0.036 and +0.014 on the same 10,000 games.
        #
        # AND THE CREATURES ACTUALLY LEAVE. Firing the drain without removing
        # the body would hand the deck free damage and no cost, which is worse
        # than not modelling it at all — the board is already overstated here by
        # never losing anything. Weakest first: chump blockers and tokens die
        # before real threats.
        if model_deaths and turn > 1:
            own_death_debt += model_deaths["own_per_turn"]
            opponent_death_debt += model_deaths["opponent_per_turn"]
            n_own = int(own_death_debt)
            own_death_debt -= n_own
            n_opp = int(opponent_death_debt)
            opponent_death_debt -= n_opp

            for _ in range(n_own):
                if not battlefield:
                    break
                # The weakest body dies -- the first of the lowest power, which
                # is what the stable sort-then-pop this replaces removed -- and
                # the parallel lists lose the same index.
                weakest = min(range(len(battlefield)), key=lambda k: battlefield[k][0])
                battlefield.pop(weakest)
                creature_types.pop(weakest)
                creature_flying.pop(weakest)
                bodies_cum = max(0, bodies_cum - 1)
                for prof in death_drains:
                    deaths_drained += prof["death_drain"]
            for _ in range(n_opp):
                for prof in opponent_death_gains:
                    deaths_gained += prof["gain_on_opponent_death"]

        # ── the drain pillar ────────────────────────────────────────────────
        #
        # Runs OUTSIDE the combat block, because a drain deck kills without ever
        # attacking and gating this on `model_combat` would reproduce the bug it
        # exists to fix.
        #
        # EVENTS AND TOTAL ARE TRACKED SEPARATELY, and they have to be: "target
        # opponent loses that much life" (Vito) scales with the AMOUNT gained,
        # while "each opponent loses 1 life" (Marauding Blight-Priest) fires once
        # per GAIN EVENT whatever the amount. Aggregating a turn into one event
        # would understate the second by however many times you gained.
        if model_drain:
            def _x_for(d):
                """X, as a real count of what the card names. 1 when the subject
                is one this model cannot count — never 0, because the card does
                do something."""
                if not d["scales_with"]:
                    return 1
                return max(1, sum(1 for tl in battlefield_types
                                  if d["scales_with"] in tl))

            gain_total = gain_events = 0
            for d in drain_permanents:
                # EERIE FIRES ON AN UNLOCK AND CONSTELLATION DOES NOT. Both read
                # "whenever an enchantment you control enters"; only Eerie adds
                # "and whenever you fully unlock a Room". Crediting the unlock to
                # every per-enchantment payoff would over-pay the plain
                # constellation cards, which is the whole reason the flag exists.
                _events = enchantments_entered + (
                    rooms_unlocked if d.get("eerie") else 0)
                if d["gain_recurring"]:
                    gain_total += d["gain_recurring"] * _x_for(d); gain_events += 1
                if d["gain_per_enchantment"] and _events:
                    gain_total += d["gain_per_enchantment"] * _events
                    gain_events += _events
                if d["gain_per_creature"] and creatures_entered_this_turn:
                    gain_total += d["gain_per_creature"] * creatures_entered_this_turn
                    gain_events += creatures_entered_this_turn
            # Lifelink gains what those creatures DEALT, so it is capped by the
            # damage actually dealt this turn — one event, because the combat
            # model resolves a swing as a single number.
            # A death that gains life is a GAIN EVENT like any other, so it
            # feeds Vito and Marauding Blight-Priest exactly as a Shrine tick
            # does. That chain is the reason the Meathook's third ability is
            # worth anything here.
            if deaths_gained:
                gain_total += deaths_gained
                gain_events += 1
            if etb_gained:
                gain_total += etb_gained
                gain_events += 1
            # A GRANT COVERS THE WHOLE TYPE, and it is worth far more than any
            # one lifelink creature: Zur, Eternal Schemer gives it to EVERY
            # enchantment creature, and this deck runs twenty-odd. Summed over
            # the board rather than accumulated on arrival, because a grant that
            # lands later covers everything already out.
            # ZIPPED TWO DIFFERENT LISTS. `battlefield_types` holds every
            # nonland permanent; `battlefield` holds only creatures. Pairing
            # them positionally matched a creature's power against an unrelated
            # permanent's type line, so the grant figure was computed from
            # garbage. The type line now rides WITH the creature.
            granted = 0
            if lifelink_granted_types:
                for tl_, (pw_, *_) in zip(creature_types, battlefield):
                    if any(ty_ in tl_ for ty_ in lifelink_granted_types):
                        granted += pw_
            effective_lifelink = max(lifelink_power, granted)
            if effective_lifelink and damage_by_turn:
                linked = min(effective_lifelink, damage_by_turn[-1])
                if linked > 0:
                    gain_total += linked; gain_events += 1

            drained = deaths_drained + etb_drained
            for d in drain_permanents:
                drained += d["drain_recurring"] * _x_for(d)
                drained += d["drain_per_enchantment"] * (
                    enchantments_entered + (rooms_unlocked if d.get("eerie") else 0))
                if d["payoff_equal"]:
                    drained += gain_total
                drained += d["payoff_fixed"] * gain_events
            drain_by_turn.append(drained)
            if drained:
                opponent_life -= drained
                if kill_turn is None and opponent_life <= 0:
                    kill_turn = turn
                    kill_by = "life"

        # Measured against the turn's FULL mana — lands, rocks and the
        # Treasure stockpile — because a Treasure you are holding is mana you
        # could have spent. `pool` has already been drawn down by the main
        # phase, so this asks what was reachable at the START of it.
        available = lands_in_play * (1 + land_mana_bonus) + rock_production + treasures
        stall_by_turn.append(not any(
            (not c["is_land"]) and c["cmc"] <= available for c in hand))
        hand_size_by_turn.append(len(hand))
        # AT THE END OF THE TURN, because the sacrifice step runs inside combat.
        # This was recorded BEFORE the combat step, so every entry held the
        # total as it stood at the START of its turn — the series was shifted by
        # one and the LAST turn's sacrifices were never recorded at all. That is
        # how a game could report `sac_cap_hits: 1` beside
        # `sacrifices_by_turn: [0,0,0,0,0,0,0,0,0,0]`: the board went wide and
        # converted on turn ten, and the only field that could have shown it had
        # already been written (#34).
        sacrifices_by_turn.append(sacrifices)

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
        "keep_can_act_by_t3": keep_can_act_by_t3,
        "mulligans": mulligans,
        "attack_tutor_fired": attack_tutor_fired,
        "reveal_fired": reveal_fired,
        "reveal_bodies": reveal_bodies,
        "tutor_enabled_turns": tutor_enabled_turns,
        "tutor_eligible_turns": tutor_eligible_turns,
        "land_hits": land_hits,
        "stall_by_turn": stall_by_turn,
        "hand_size_by_turn": hand_size_by_turn,
        "mana_by_turn": mana_by_turn,
        "commander_turn": commander_turn,
        "commander_counters_by_turn": commander_counters_by_turn,
        # The stockpile STANDING at the end of each turn, not what was made:
        # the same reading `treasures_by_turn` takes, so the two can be read
        # beside each other without a footnote.
        "blood_by_turn": blood_by_turn,
        "artifact_sacs_by_turn": artifact_sacs_by_turn,
        "bodies_by_turn": bodies_by_turn,
        "drawn_extra_by_turn": drawn_extra_by_turn,
        "discarded_by_turn": discarded_by_turn,
        "event_damage_by_turn": event_damage_by_turn,
        "partner_turn": partner_turn,
        "sacrifices_by_turn": sacrifices_by_turn,
        "sac_cap_hits": sac_cap_hits,
        "interaction_in_hand_by_turn": interaction_in_hand_by_turn,
        "interaction_castable_by_turn": interaction_castable_by_turn,
        "target_turns": target_turns,
        "target_turns_unassisted": target_turns_unassisted,
        "treasures_by_turn": treasures_by_turn,
        "treasure_online_by_turn": treasure_online_by_turn,
        "damage_by_turn": damage_by_turn,
        "board_power_by_turn": board_power_by_turn,
        "kill_turn": kill_turn,
        "poison_by_turn": poison_by_turn,
        "kill_by": kill_by,
        "drain_by_turn": drain_by_turn,
    }
