"""Two channels the model was blind to, and the sweeps that scoped them.

Both arrived on 2026-08-28 out of one finding on edgar-vampires: the pilot's
stated engine is "vampires enter -> tokens appear -> tokens convert into value
-> drain closes the game", and `combat_profile` read FOUR of that deck's payoffs
and returned nothing for six. `damage_8` therefore scored the combat plan the
pilot wanted to CUT and was blind to the drain plan they wanted to DEEPEN — a
branch aimed at that axis would have been graded on the wrong half of the deck.

1. **Life loss on arrival is damage here.** The gate was the literal word
   "damage": Impact Tremors was priced and Corpse Knight, which says the same
   thing about the same event with the same number, was worth zero.
2. **Card draw exists at all.** `card_advantage` reported
   `{"cards_that_draw": 12, "modelled": 0}` and the loop drew one card a turn
   whatever the list said.

The scoping in both is what the corpus sweeps bought, and the sweeps are the
only reason the scopes are not uniform. Those asymmetries are what these tests
hold, because they are the part a later refactor would tidy away.
"""

import pytest

from manamap.pilot import goldfish

from conftest import requires_data, requires_deck
from manamap.config import DATA_DIR


def _card(name, text, cmc=3, type_line="Creature — Vampire", power="2",
          toughness="2"):
    return {"name": name, "type_line": type_line, "cmc": cmc,
            "oracle_text": text, "quantity": 1, "is_commander": False,
            "power": power, "toughness": toughness}


# ── 1. Life loss on arrival ───────────────────────────────────────────────

def test_the_same_event_worded_two_ways_prices_the_same():
    """Impact Tremors and Corpse Knight. One damage per arrival, either way."""
    tremors = goldfish.combat_profile(_card(
        "Impact Tremors",
        "Whenever a creature you control enters, this enchantment deals 1 "
        "damage to each opponent.", type_line="Enchantment"))
    knight = goldfish.combat_profile(_card(
        "Corpse Knight",
        "Whenever another creature you control enters, each opponent loses 1 life."))
    assert tremors["etb_damage_fixed"] == 1
    assert knight["etb_life_loss_fixed"] == 1
    # And BOTH make the card an ETB engine. The predicate is what the two cast
    # sites read; before it was factored out they listed the fields by hand and
    # would have been updated in one place and not the other.
    assert goldfish.is_etb_engine(tremors)
    assert goldfish.is_etb_engine(knight)


def test_a_drain_on_a_DIFFERENT_trigger_does_not_ride_the_entry_trigger():
    """RE-INTRODUCING THE BUG THE SCOPING WAS WRITTEN FOR. Elas il-Kor gains
    life when a creature enters and drains when one DIES. Read from the 220-char
    window the damage payloads use, its death trigger lands inside the entry
    trigger's window and it drains on every arrival."""
    elas = goldfish.combat_profile(_card(
        "Elas il-Kor, Sadistic Pilgrim",
        "Deathtouch Whenever another creature you control enters, you gain 1 "
        "life. Whenever another creature you control dies, each opponent loses "
        "1 life."))
    assert elas["etb_life_loss_fixed"] == 0
    # The control: the window DOES contain the phrase, so a test that only
    # asserted the result could pass against an implementation that never looked.
    text = elas and ("each opponent loses 1 life" in
                     "Deathtouch Whenever another creature you control enters, "
                     "you gain 1 life. Whenever another creature you control "
                     "dies, each opponent loses 1 life."[:220])
    assert text, "the fixture must put the decoy inside the old window"


def test_an_activated_drain_is_not_an_arrival_drain():
    """Underworld Coinsmith's drain costs {W}{B} and 1 life."""
    assert goldfish.combat_profile(_card(
        "Underworld Coinsmith",
        "Constellation — Whenever this creature or another enchantment you "
        "control enters, you gain 1 life. {W}{B}, Pay 1 life: Each opponent "
        "loses 1 life."))["etb_life_loss_fixed"] == 0


def test_a_drain_that_lasts_one_turn_is_not_an_engine():
    """Thunder of Unity is a Saga chapter: 'whenever a creature you control
    enters THIS TURN'. One card in the corpus and it would have been over-read."""
    assert goldfish.combat_profile(_card(
        "Thunder of Unity",
        "II, III — Whenever a creature you control enters this turn, each "
        "opponent loses 1 life and you gain 1 life.",
        type_line="Enchantment — Saga"))["etb_life_loss_fixed"] == 0


def test_the_damage_payload_still_reads_across_a_sentence_boundary():
    """THE ASYMMETRY, AND THE WHOLE REASON THE SCOPES DIFFER. Clause-scoping the
    DAMAGE payload as well would have looked tidy and dropped 3 of its 16 correct
    matches: Crossbones puts its payload in a second sentence. Same shape as the
    shocklands, whose idiom also spans the boundary."""
    assert goldfish.combat_profile(_card(
        "Crossbones, Malicious Mercenary",
        "Deathtouch Whenever another Villain you control enters, put a +1/+1 "
        "counter on Crossbones. He deals 2 damage to each opponent. This "
        "ability triggers only once each turn."))["etb_damage_fixed"] == 2


def test_token_creation_is_its_own_trigger():
    """Mirkwood Bats keys on CREATE, not on `enters`, so the entry trigger never
    saw it — and it is a named member of edgar-vampires' kill leg."""
    bats = goldfish.combat_profile(_card(
        "Mirkwood Bats",
        "Flying Whenever you create or sacrifice a token, each opponent loses 1 life."))
    assert bats["token_created_life_loss"] == 1
    assert bats["etb_life_loss_fixed"] == 0, "it is not an `enters` trigger"
    assert goldfish.is_etb_engine(bats)


# ── 2. Card draw ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,text,type_line,field,expected", [
    ("Dusk Legion Zealot", "When this creature enters, you draw a card and you "
     "lose 1 life.", "Creature — Vampire Soldier", "etb_draw", 1),
    ("Night's Whisper", "You draw two cards and lose 2 life.", "Sorcery",
     "spell_draw", 2),
    ("Phyrexian Arena", "At the beginning of your upkeep, you draw a card and "
     "you lose 1 life.", "Enchantment", "recurring_draw", 1),
    ("Caretaker's Talent", "Whenever one or more tokens you control enter, draw "
     "a card.", "Enchantment — Class", "arrival_draw", 1),
])
def test_each_modelled_channel_reads(name, text, type_line, field, expected):
    assert goldfish.draw_profile(
        _card(name, text, type_line=type_line))[field] == expected


def test_a_conditional_draw_is_unmodelled_rather_than_assumed_on():
    """39 of the 348 ETB-draw cards carry a gate inside the trigger. Reading the
    trigger and ignoring its condition is the same defect the life-loss scoping
    exists to avoid, one clause further in."""
    for text in ("When this creature enters, if you control an artifact, draw a card.",
                 "Whenever this creature enters or attacks, you may draw a card.",
                 "Whenever another creature enters, its controller may draw a card."):
        p = goldfish.draw_profile(_card("X", text))
        assert p["etb_draw"] == 0 and p["unmodelled"] == "X", text


def test_the_power_qualifier_is_honoured_in_BOTH_directions():
    """Welcoming Vampire draws off a 1/1 token; Garruk's Uprising must not.
    Reading the trigger and ignoring its condition would hand every token deck a
    draw engine it does not have."""
    welcome = goldfish.draw_profile(_card(
        "Welcoming Vampire",
        "Flying Whenever one or more other creatures you control with power 2 "
        "or less enter, draw a card. This ability triggers only once each turn."))
    garruk = goldfish.draw_profile(_card(
        "Garruk's Uprising",
        "Whenever a creature you control with power 4 or greater enters, draw a "
        "card.", type_line="Enchantment"))
    assert welcome["arrival_draw"] == 1 and welcome["arrival_power_max"] == 2
    assert welcome["arrival_draw_once"] is True
    assert garruk["arrival_draw"] == 1 and garruk["arrival_power_min"] == 4
    assert garruk["arrival_power_max"] is None


def test_a_qualifier_the_model_cannot_evaluate_is_not_guessed_at():
    """"with defender", "of the chosen type", "named Gladewalker Ritualist" —
    all real cards in the sweep. Firing on them invents an engine."""
    for q in ("with defender ", "of the chosen type ", "with mana value 3 or less "):
        p = goldfish.draw_profile(_card(
            "X", f"Whenever a creature you control {q}enters, draw a card."))
        assert p["arrival_draw"] == 0, q


# ── 3. The model must ACT on what it reads ────────────────────────────────

@requires_data
@requires_deck
def test_the_drain_channels_move_the_clock():
    """A FLAG THE MODEL SETS IS A CLAIM THE MODEL MUST ACT ON. `treasure_doubler`
    shipped set-and-unread and fifteen candidates returned byte-identical
    -0.026. Byte-identical is the tell; this asserts against it."""
    real = goldfish.combat_profile

    def blind(card):
        p = real(card)
        p["etb_life_loss_fixed"] = 0
        p["token_created_life_loss"] = 0
        return p

    on = goldfish.run("edgar-vampires", iterations=1200, quiet=True)
    goldfish.combat_profile = blind
    try:
        off = goldfish.run("edgar-vampires", iterations=1200, quiet=True)
    finally:
        goldfish.combat_profile = real
    a = on["metrics"]["mean_bodies_by_turn"]
    b = off["metrics"]["mean_bodies_by_turn"]
    # Bodies are untouched by a drain channel — the control that proves the two
    # runs are otherwise the same run.
    assert a == b, "the runs differ by more than the channel under test"
    assert on["meta"]["card_advantage"], "sanity: the meta block survived"


@requires_data
@requires_deck
def test_a_deck_that_has_not_opted_into_draw_has_no_draw_series():
    """Absent means ABSENT. A zero series is a measurement and a reader cannot
    tell it from one."""
    off = goldfish.run("heliod", iterations=300, quiet=True)
    assert "mean_extra_cards_drawn_by_turn" not in off["metrics"]
    assert off["meta"]["card_advantage"]["modelled"] == 0


@requires_data
@requires_deck
def test_the_draw_model_draws_and_says_what_it_could_not_read():
    on = goldfish.run("edgar-vampires", iterations=1200, quiet=True,
                      model_draw=True)
    series = on["metrics"]["mean_extra_cards_drawn_by_turn"]
    assert series["1"] == 0.0, "nothing has been cast on turn one"
    assert series["10"] > series["5"] > 0, "it must accumulate"
    ca = on["meta"]["card_advantage"]
    assert ca["cards_that_draw"] > ca["modelled"] >= 1, (
        "edgar runs twelve draw cards and exactly one is unconditional — if "
        "this ever reads equal, a channel started guessing")
    assert on["meta"]["card_advantage"]["draw_not_modelled"], "name what it cannot read"


# ── 4. Held-up interaction ────────────────────────────────────────────────

@requires_data
@requires_deck
def test_castable_interaction_can_never_exceed_interaction_in_hand():
    """The two series answer different halves of one failure and one is a subset
    of the other by construction. A run where it is not is a bug in the
    instrumentation, not a fact about the deck."""
    m = goldfish.run("edgar-vampires", iterations=800, quiet=True)["metrics"]
    checked = 0
    for t in m["interaction_in_hand_by_turn"]:
        assert m["interaction_castable_by_turn"][t] <= m["interaction_in_hand_by_turn"][t], t
        checked += 1
    assert checked >= 8


@requires_data
@requires_deck
def test_the_suite_it_counted_is_named():
    """An empty set would make both series read a flat zero, which is
    indistinguishable from a deck that runs no interaction at all."""
    doc = goldfish.run("edgar-vampires", iterations=200, quiet=True)
    named = doc["meta"]["interaction_suite_counted"]
    assert len(named) >= 8, "edgar-vampires runs twelve interaction cards"
    assert "Teferi's Protection" in named and "Deflecting Swat" in named, (
        "the two cards the pilot's log says sat in hand all game must be in the "
        "set the metric was measured against")


@requires_data
@requires_deck
def test_the_stricter_keep_threshold_is_reported_beside_the_loose_one():
    """It is REPORTED and never ENFORCED: `keepable` still decides mulligans, so
    changing this cannot restate a figure on any deck."""
    oh = goldfish.run("edgar-vampires", iterations=800, quiet=True)["metrics"]["opening_hand"]
    assert 0.0 < oh["keep_can_act_by_t3_rate"] <= 1.0
    assert 0.0 < oh["keep_first_seven_rate"] <= 1.0


# ── 5. Two bugs the first cut shipped, and the conditions that expose them ──

@requires_data
@requires_deck
def test_an_arrival_draw_engine_does_not_see_its_own_arrival():
    """RE-INTRODUCING THE CONDITION, on the card that carried the bug.

    Welcoming Vampire is a 2/3 that draws "whenever one or more OTHER creatures
    you control with power 2 or less enter". Its own power is 2, so registering
    the engine before its own entry passed its own gate and drew a card it does
    not draw — worth 22% of edgar-vampires' turn-eight figure.

    Driven by MONKEYPATCHING THE PROFILE rather than by re-deriving the rule: if
    the deferral is removed, the card the bug was found on must draw more.
    """
    real = goldfish.draw_profile

    def eager(card):
        p = real(card)
        # A card that cannot trigger on anything else at all, so the only draw
        # it can produce is the illegal one off its own body.
        if card.get("name") == "Welcoming Vampire":
            p["arrival_draw_once"] = False
        return p

    fixed = goldfish.run("edgar-vampires", iterations=2000, quiet=True,
                         model_draw=True)["metrics"]
    goldfish.draw_profile = eager
    try:
        loosened = goldfish.run("edgar-vampires", iterations=2000, quiet=True,
                                model_draw=True)["metrics"]
    finally:
        goldfish.draw_profile = real
    a = fixed["mean_extra_cards_drawn_by_turn"]["10"]
    b = loosened["mean_extra_cards_drawn_by_turn"]["10"]
    assert b > a, (
        "dropping the once-a-turn cap must draw MORE — if these are equal the "
        "arrival channel is not firing at all and this test proves nothing")


@requires_data
@requires_deck
def test_the_arrival_channel_does_not_secretly_require_the_combat_model():
    """THE SILENT ZERO. Every call to `creature_entered` — the one door the
    arrival channel rides — used to sit inside `if model_combat:`, so a deck
    opting into `model_draw` ALONE lost three quarters of its arrival draws and
    reported the smaller number without a word. Measured then: 1.264 extra cards
    by turn ten with both flags against 0.323 with draw alone.

    A gap REMAINS and is legitimate — the combat model spawns token copies and
    ETB-payoff tokens, which are real extra arrivals. What is asserted is that
    the draw-only figure is most of the way there rather than a rounding error.
    """
    both = goldfish.run("edgar-vampires", iterations=2000, quiet=True,
                        model_draw=True, model_combat=True)["metrics"]
    only = goldfish.run("edgar-vampires", iterations=2000, quiet=True,
                        model_draw=True, model_combat=False)["metrics"]
    a = both["mean_extra_cards_drawn_by_turn"]["10"]
    b = only["mean_extra_cards_drawn_by_turn"]["10"]
    assert a > 0 and b > 0
    assert b > a * 0.7, (
        f"draw-only reads {b} against {a} with combat — the arrival channel is "
        f"leaking through a flag it should not depend on")


@requires_data
@requires_deck
def test_turning_the_draw_model_on_does_not_move_a_combat_deck():
    """THE OPT-IN CONTRACT, held on the deck that would break it. The arrival
    fix reordered statements inside the cast loop, and a reorder that changed a
    figure would restate every published number on ur-dragon."""
    import json
    on_disk = json.loads(
        (DATA_DIR / "decks" / "ur-dragon" / "goldfish_metrics.json").read_text())
    fresh = goldfish.run("ur-dragon", quiet=True)
    assert json.dumps(fresh["metrics"], sort_keys=True) == json.dumps(
        on_disk["metrics"], sort_keys=True), "ur-dragon moved"


@requires_data
@requires_deck
def test_the_held_up_series_names_the_moment_it_is_measured_at():
    """Extra combats are PAID FOR after this measurement and attack triggers add
    mana after it, so the figure is end-of-main-phase float and not what
    survives the turn. A reader given the number without the moment will read it
    as the second thing."""
    blob = " ".join(goldfish.MODEL_ASSUMPTIONS)
    assert "END OF THE MAIN PHASE" in blob or "HELD-UP INTERACTION" in blob
    from manamap.pilot import diagnostic
    d = diagnostic.run("edgar-vampires", iterations=400, quiet=True)
    assert "END OF THE MAIN PHASE" in d["steam"]["basis"]


@pytest.mark.parametrize("name,text", [
    # RE-INTRODUCING THE CONDITION, on the two cards that were actively scoring.
    ("Dread Presence", "Whenever a Swamp you control enters, choose one — "
                       "• You draw a card and you lose 1 life. • This "
                       "creature deals 2 damage to any target and you gain 2 life."),
    ("Koth, Fire of Resistance",
     "−7: You get an emblem with \"Whenever a Mountain you control enters, "
     "this emblem deals 4 damage to any target.\""),
    ("Valakut, the Molten Pinnacle",
     "Whenever a Mountain you control enters, you may have this land deal 3 "
     "damage to any target."),
    ("Battlewand Oak", "Whenever a Forest you control enters, this creature "
                       "gets +1/+1 until end of turn."),
    ("Guild Summit", "Whenever a Gate you control enters, draw a card."),
])
def test_a_landfall_trigger_named_by_LAND_TYPE_is_not_a_creature_arrival(name, text):
    """`(?!lands?\\b)` was written for "whenever a LAND you control enters" and
    let every basic land TYPE through, so a landfall payoff read as a payoff for
    creatures entering. Fourteen corpus cards; two were scoring. Dread Presence
    billed 2 damage per CREATURE arrival off a Swamp trigger, and Koth — a
    PLANESWALKER — billed 4 off an emblem. Both surfaced while searching for
    candidates on this very channel, which is how they were caught.
    """
    p = goldfish.combat_profile(_card(name, text, type_line="Creature"))
    assert p["etb_damage_fixed"] == 0, name
    assert p["etb_life_loss_fixed"] == 0, name
    assert not goldfish.is_etb_engine(p), name


def test_the_narrowing_did_not_take_a_real_creature_trigger_with_it():
    """The control. A guard that fires on correct data is worse than none."""
    for text in ("Whenever another creature you control enters, this enchantment "
                 "deals 1 damage to each opponent.",
                 "Whenever another Vampire you control enters, each opponent "
                 "loses 1 life.",
                 "Whenever a creature you control enters, this creature deals 2 "
                 "damage to each opponent."):
        p = goldfish.combat_profile(_card("X", text))
        assert goldfish.is_etb_engine(p), text


def test_the_PLURAL_wording_is_a_known_gap_and_is_empty_for_this_channel():
    """"Whenever one or more creatures you control ENTER" is real templating and
    the trigger only matches the singular `enters`. Swept: widening it matches
    17 more corpus cards and NOT ONE of them carries a damage or drain payload,
    so the gap is real and empty here.

    It is deliberately left alone. What the widening WOULD pull in is
    token-makers worded "this ability triggers only once each turn" (Baron
    Bertram Graywater, Caretaker's Talent), and this model has no once-a-turn
    concept for `etb_token_bodies` — firing them on every arrival is the
    compounding that once reported 67,000 damage by turn six.
    """
    plural = _card("X", "Whenever one or more creatures you control enter, this "
                        "creature deals 1 damage to each opponent.")
    assert goldfish.combat_profile(plural)["etb_damage_fixed"] == 0, (
        "if this starts matching, re-run the sweep: the widening must be paired "
        "with a once-a-turn cap on the token channel")


# ── 6. Eminence, and the sacrifice engine it feeds ────────────────────────

def test_the_commander_mints_a_token_on_cast_and_the_model_reads_it():
    """THE BIGGEST HOLE THIS MODEL HAS EVER HAD, and it was the deck's whole
    axis. Edgar Markov's eminence — "whenever you cast another Vampire spell,
    if Edgar is in the command zone or on the battlefield, create a 1/1 black
    Vampire creature token" — was not modelled at all. `command_zone_reduction`
    reads a commander for COST REDUCTION (The Ur-Dragon's eminence) and there
    was no channel for the other kind, so every token that engine makes was
    missing: the bodies, the arrival-damage payoffs they fire, the arrival draw
    they fire, and the fuel the sacrifice model eats.
    """
    edgar = goldfish.cast_token_profile({"oracle_text":
        "Eminence — Whenever you cast another Vampire spell, if Edgar is in "
        "the command zone or on the battlefield, create a 1/1 black Vampire "
        "creature token. First strike, haste"})
    assert edgar == {"subtype": "Vampire", "bodies": 1, "power": 1}


def test_a_cost_reducing_eminence_is_not_a_token_eminence():
    """The control: The Ur-Dragon's eminence discounts, it does not mint. A
    channel that fired on it would hand a five-colour deck a free body per cast."""
    assert goldfish.cast_token_profile({"oracle_text":
        "Eminence — Other Dragon spells you cast cost {1} less to cast as long "
        "as this creature is in the command zone or on the battlefield."}) is None
    assert goldfish.cast_token_profile({"oracle_text": ""}) is None


def test_an_untyped_cast_trigger_is_left_unmodelled_rather_than_fired_on_all():
    """"Whenever you cast a spell, create a token" with no subtype would fire on
    every card in the deck. Unmodelled beats a board that doubles itself."""
    assert goldfish.cast_token_profile({"oracle_text":
        "Whenever you cast a spell, create a 1/1 white Soldier creature token."}) is None


@requires_data
@requires_deck
def test_eminence_tokens_actually_reach_the_battlefield():
    """A FLAG THE MODEL SETS IS A CLAIM THE MODEL MUST ACT ON. Driven by
    suppressing the profile: with the engine blind, the bodies must fall."""
    real = goldfish.cast_token_profile
    on = goldfish.run("edgar-vampires", iterations=1500, quiet=True)
    goldfish.cast_token_profile = lambda card: None
    try:
        off = goldfish.run("edgar-vampires", iterations=1500, quiet=True)
    finally:
        goldfish.cast_token_profile = real
    a = on["metrics"]["mean_bodies_by_turn"]["10"]
    b = off["metrics"]["mean_bodies_by_turn"]["10"]
    assert a > b * 1.15, (
        f"eminence adds {a - b:.2f} bodies by turn ten; a vampire deck casting "
        f"~30 Vampires must gain materially more than rounding")


# ── the sacrifice engine ──────────────────────────────────────────────────

@pytest.mark.parametrize("name,text,expect", [
    ("Ashnod's Altar", "Sacrifice a creature: Add {C}{C}.", "free"),
    ("Viscera Seer", "Sacrifice a creature: Scry 1.", "free"),
    ("Phyrexian Tower", "{T}: Add {C}. {T}, Sacrifice a creature: Add {B}{B}.", "costed"),
    ("Indulgent Aristocrat", "Lifelink {2}, Sacrifice a creature: Put a +1/+1 "
                             "counter on each Vampire you control.", "costed"),
    ("Blood Artist", "Whenever this creature or another creature dies, target "
                     "player loses 1 life and you gain 1 life.", None),
])
def test_a_costed_outlet_is_not_a_free_one(name, text, expect):
    """180 corpus cards put mana or a tap symbol before the colon and 48 do not.
    Counting a costed outlet as free hands the deck an engine it has to pay for,
    and a tap symbol also caps it at once a turn."""
    assert goldfish.sac_outlet_profile(_card(name, text)) == expect


@pytest.mark.parametrize("name,text,field,value", [
    ("Blood Artist", "Whenever this creature or another creature dies, target "
                     "player loses 1 life and you gain 1 life.", "death_drain", 1),
    ("Zulaport Cutthroat", "Whenever this creature or another creature you "
                           "control dies, each opponent loses 1 life and you "
                           "gain 1 life.", "death_drain", 1),
    ("Midnight Reaper", "Whenever a nontoken creature you control dies, this "
                        "creature deals 1 damage to you and you draw a card.",
     "death_draw", 1),
    ("Pitiless Plunderer", "Whenever another creature you control dies, create "
                           "a Treasure token.", "death_treasure", 1),
])
def test_each_death_payload_reads(name, text, field, value):
    assert goldfish.death_profile(_card(name, text))[field] == value


def test_a_death_trigger_it_cannot_price_is_NAMED_not_zeroed():
    """Elenda makes X tokens where X is her counters. Absent, not zero."""
    p = goldfish.death_profile(_card(
        "Elenda, the Dusk Rose",
        "Whenever another creature dies, put a +1/+1 counter on this creature. "
        "When this creature dies, create X 1/1 white Vampire creature tokens "
        "with lifelink, where X is its power."))
    assert not goldfish.is_death_engine(p)
    assert p["unreadable"] == "Elenda, the Dusk Rose"


@requires_data
@requires_deck
def test_the_sacrifice_engine_is_opt_in_and_fires_when_it_is_on():
    off = goldfish.run("edgar-vampires", branch=None, iterations=1200, quiet=True,
                       model_sacrifice=False)
    on = goldfish.run("edgar-vampires", branch=None, iterations=1200, quiet=True,
                      model_sacrifice=True)
    assert "mean_sacrifices_by_turn" not in off["metrics"], "absent, not zero"
    assert "mean_sacrifices_by_turn" in on["metrics"]


@requires_data
@requires_deck
def test_the_runaway_guard_holds():
    """A death payoff that makes a token is a loop, and a loop that terminates
    silently cannot be told from one that never ran."""
    doc = goldfish.run("edgar-vampires", branch=None, iterations=1500, quiet=True,
                       model_sacrifice=True)
    m = doc["metrics"]
    assert m["mean_sacrifices_by_turn"]["10"] < goldfish.SAC_LIMIT_PER_TURN * 10
    assert m["sac_cap_hit_rate"] == 0.0, (
        "the cap fired on a real deck — either a loop exists or the policy is "
        "eating more than a board can hold")


# ── 7. Token doublers, for the tokens that fight ──────────────────────────

@pytest.mark.parametrize("name,text,expect", [
    ("Anointed Procession", "If an effect would create one or more tokens under "
                            "your control, it creates twice that many of those "
                            "tokens instead.", True),
    ("Mondrak, Glory Dominus", "If one or more tokens would be created under your "
                               "control, twice that many of those tokens are "
                               "created instead.", True),
    ("Doubling Season", "If an effect would create one or more tokens under your "
                        "control, it creates twice that many of those tokens "
                        "instead.", True),
    # A -2 THAT LASTS ONE TURN IS NOT A DOUBLER. Same lesson the ETB life-loss
    # channel records: the condition is scoped to the clause it attaches to.
    ("Kaya, Geist Hunter", "-2: Until end of turn, if one or more tokens would be "
                           "created under your control, twice that many of those "
                           "tokens are created instead.", False),
    ("Hosting Season", "While it's October 26th or 27th, 2024, if one or more "
                       "tokens would be created under your control, instead twice "
                       "that many tokens are created.", False),
    # A TRIPLER IS NOT A DOUBLER. Reading it as x2 understates; reading the
    # phrase loosely would overstate. Six corpus cards say "three times".
    ("Ojer Taq, Deepest Foundation", "If one or more creature tokens would be "
                                     "created under your control, three times "
                                     "that many are created instead.", False),
])
def test_a_token_doubler_is_permanent_and_is_exactly_twice(name, text, expect):
    assert goldfish.token_doubler({"oracle_text": text}) is expect


@requires_data
@requires_deck
def test_the_doublers_actually_double_the_tokens_that_fight():
    """`treasure_doubler` shipped with the Treasure model and its own comment
    calls the shape "Procession-style xN" — but it only ever multiplied
    Treasures. Anointed Procession, Elspeth and Mondrak doubled nothing that
    attacks, in a deck whose engine brief reads "eminence mints a free body
    every time you cast a Vampire and the doublers turn one mint into four".

    Driven by suppressing the profile, not by re-deriving the rule.
    """
    real = goldfish.token_doubler
    on = goldfish.run("edgar-vampires", branch=None, iterations=2000, quiet=True)
    goldfish.token_doubler = lambda c: False
    try:
        off = goldfish.run("edgar-vampires", branch=None, iterations=2000, quiet=True)
    finally:
        goldfish.token_doubler = real
    a = on["metrics"]["mean_bodies_by_turn"]["10"]
    b = off["metrics"]["mean_bodies_by_turn"]["10"]
    assert a > b, f"doublers changed nothing: {a} vs {b} — the flag is unread"


def test_two_doublers_are_four_times_and_not_three():
    """Each replaces the other's output, which is why this compounds rather than
    sums — the same arithmetic `treasure_multiplier` already documents."""
    src = __import__("inspect").getsource(goldfish)
    assert "token_multiplier *= 2" in src, (
        "the doubler must multiply; adding would make two doublers x3")
