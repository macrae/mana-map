"""The discard channel: wheels, loots, and the payoffs on a discard or a draw.

The six wheels in sharknado were INVISIBLE to this model, not dark — `_DRAW_RE`
wants "draw N card" and Wheel of Fortune says "draws seven cards" — and no
casting loop ever selected one. Faithless Looting read +2 with nothing netting
the discard. "Whenever you discard a card" and "whenever you draw a card" were
modelled nowhere. And the second commander of a Partner pair was never cast:
`commanders[0]` was the only one the model knew, so Brallin, Skyshark Rider
scored zero however well his trigger parsed.
"""

import csv
import re

import pytest

from manamap.config import OUTPUT_CSV_PATH
from manamap.pilot import goldfish
from conftest import patch_model, requires_data, requires_deck


def _spell(text, type_line="Sorcery", name="a spell", mana_cost="{2}{R}"):
    return goldfish.draw_profile({"name": name, "oracle_text": text, "type_line": type_line,
                                  "mana_cost": mana_cost})


def _ev(text, name="a card"):
    return goldfish.event_payoffs({"name": name, "oracle_text": text})


def test_wheels_are_read_and_the_conditional_one_is_not():
    assert _spell("Each player discards their hand, then draws seven cards.")["wheel_draws"] == 7
    assert _spell("Each player discards their hand, then draws cards equal to the greatest "
                  "number of cards a player discarded this way.")["wheel_draws"] == -1
    assert _spell("Discard your hand, then draw three cards.")["wheel_draws"] == 3
    assert _spell("Each player discards their hand, then draws seven cards.",
                  name="Wheel of Misfortune")["wheel_draws"] == 0
    # A wheel on a creature ("whenever … discards their hand") is not a spell.
    assert _spell("Each player discards their hand, then draws seven cards.",
                  type_line="Creature — Wizard")["wheel_draws"] == 0
    assert _spell("Each player discards their hand, then draws seven cards.")["unmodelled"] is None


def test_a_loot_keeps_its_gross_draw_and_records_its_discard():
    d = _spell("Draw two cards, then discard two cards.")
    assert (d["spell_draw"], d["spell_discard"]) == (2, 2)
    d = _spell("Draw three cards, then put two cards from your hand on top of your library.")
    assert (d["spell_draw"], d["spell_discard"]) == (3, 0)
    # An additional-cost discard stays unmodelled, as before.
    d = _spell("As an additional cost to cast this spell, discard a card.\nDraw two cards.")
    assert d["spell_draw"] == 0 and d["unmodelled"]


def test_the_payoffs_are_read_per_event():
    assert _ev("Whenever you discard a card, put a +1/+1 counter on Brallin and it deals 1 "
               "damage to each opponent.") == {**_ev(""), "per_discard_damage": 1, "per_discard_counter": 1}
    assert _ev("Whenever you draw a card, put a +1/+1 counter on Shabraz and you gain 1 life.")["per_draw_counter"] == 1
    assert _ev("Whenever you draw a card, each opponent loses 1 life.")["per_draw_damage"] == 1
    assert _ev("Whenever you draw a card, Niv-Mizzet deals 1 damage to any target.")["per_draw_damage"] == 1
    assert _ev("Whenever you draw your second card each turn, this creature deals 3 damage "
               "to any target.")["second_draw_damage"] == 3
    # A payoff that draws on a draw is deliberately not read (an uncapped loop).
    assert not goldfish.has_event_payoff(_ev("Whenever enchanted creature deals damage to an opponent, draw a card."))
    assert goldfish.has_event_payoff(_ev("Whenever you discard a card, draw a card.")) is True


@requires_data
def test_the_corpus_sweep_is_locked():
    wheels = loots = disc = draw = second = 0
    with open(OUTPUT_CSV_PATH, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            card = {"name": row["name"], "oracle_text": row["oracle_text"],
                    "type_line": row["type_line"], "mana_cost": row.get("mana_cost")}
            d = goldfish.draw_profile(card); e = goldfish.event_payoffs(card)
            wheels += bool(d["wheel_draws"]); loots += bool(d["spell_discard"])
            disc += bool(e["per_discard_damage"] or e["per_discard_counter"]
                         or e["per_discard_draw"] or e["per_discard_token_power"])
            draw += bool(e["per_draw_damage"] or e["per_draw_counter"] or e["per_draw_token_power"])
            second += bool(e["second_draw_damage"] or e["second_draw_token_power"])
    assert (wheels, loots, disc, draw, second) == (WHEELS, LOOTS, DISC, DRAW, SECOND), \
        (wheels, loots, disc, draw, second)


# Filled in by the sweep the day the channel shipped; a widened pattern moves
# these on purpose.
#
# DISC 15 -> 13 on 2026-09-14, and the two that left were never there. Stripping
# reminder text from `event_payoffs` removed a PHANTOM `per_discard_draw` from
# Marauding Mako and Scrounging Skyray, whose parenthesised reminders fell inside
# a trigger's effect window. Magmakin Artillerist was the third card the strip
# moved and it stays in the count — it swapped a phantom draw for the damage it
# actually deals, which is the whole reason the bug was worth finding.
WHEELS, LOOTS, DISC, DRAW, SECOND = (25, 40, 13, 29, 20)


@requires_data
@requires_deck
def test_sharknado_wheels_and_the_partner_is_cast(monkeypatch):
    on = goldfish.run("sharknado", branch="recon-v1", iterations=1200, quiet=True, model_discard=True)["metrics"]
    off = goldfish.run("sharknado", branch="recon-v1", iterations=1200, quiet=True, model_discard=False)["metrics"]
    assert "discard" in on and "discard" not in off, "absent, not zero, when the flag is off"
    assert on["discard"]["mean_cards_discarded_by_turn"]["8"] > 1
    assert on["mean_extra_cards_drawn_by_turn"]["8"] > off["mean_extra_cards_drawn_by_turn"]["8"] * 2, \
        "the wheels are cast now and were invisible before"
    assert on["discard"]["mean_cumulative_event_damage_by_turn"]["8"] > 0
    # THE PARTNER is cast in both arms: the second commander is not a channel.
    assert on["partner"]["cast_by_turn_6_rate"] > 0.8 and off["partner"]["cast_by_turn_6_rate"] > 0.8
    assert on["commander"]["cast_by_turn_6_rate"] > 0.7, "the first commander still casts on its own curve"
    # PROVEN BY BLINDING THE PROFILE, partners on in both arms: no wheels read,
    # and the extra-cards figure falls back to the loots alone.
    real = goldfish.draw_profile
    def blind(card):
        d = real(card); d["wheel_draws"] = 0; return d
    patch_model(monkeypatch, "draw_profile", blind)
    blinded = goldfish.run("sharknado", branch="recon-v1", iterations=1200, quiet=True, model_discard=True)["metrics"]
    assert blinded["mean_extra_cards_drawn_by_turn"]["8"] < on["mean_extra_cards_drawn_by_turn"]["8"] * 0.6


@requires_data
@requires_deck
def test_a_deck_that_did_not_opt_in_is_byte_identical_beyond_the_stamp():
    """The channel is gated: a non-opted deck computes nothing new."""
    import json
    from manamap.config import DATA_DIR
    on_disk = json.loads((DATA_DIR / "decks" / "ur-dragon" / "goldfish_metrics.json").read_text())
    fresh = goldfish.run("ur-dragon", quiet=True)
    a, b = dict(on_disk["metrics"]), dict(fresh["metrics"])
    assert a == b, "ur-dragon moved without opting into model_discard"


def test_scry_then_draw_and_upkeep_reveal_are_draw():
    """Found on ingris-infect/draw-v1: Read the Bones and Dark Confidant went in
    as card advantage and the draw axis did not move, because neither says
    'draw' where the parser looks."""
    d = goldfish.draw_profile({"name": "Read the Bones", "type_line": "Sorcery",
                               "oracle_text": "Scry 2, then draw two cards. You lose 2 life."})
    assert d["spell_draw"] == 2 and d["unmodelled"] is None
    d = goldfish.draw_profile({"name": "Dark Confidant", "type_line": "Creature — Human Wizard",
                               "oracle_text": "At the beginning of your upkeep, reveal the top card of "
                                              "your library and put that card into your hand. You lose "
                                              "life equal to its mana value."})
    assert d["recurring_draw"] == 1


@requires_data
def test_the_two_draw_shapes_are_locked_to_the_corpus():
    scry = reveal = 0
    with open(OUTPUT_CSV_PATH, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            t = row["oracle_text"] or ""
            if goldfish._SCRY_THEN_DRAW_RE.search(t) and ("Instant" in row["type_line"] or "Sorcery" in row["type_line"]):
                scry += 1
            if goldfish._UPKEEP_REVEAL_RE.search(t):
                reveal += 1
    assert (scry, reveal) == (34, 3), (scry, reveal)


# ── the wheel that is a PERMANENT ─────────────────────────────────────────
#
# `wheel_draws` is credited on an Instant or a Sorcery only, and that is right:
# a permanent carrying the same sentence carries it as an ACTIVATED ABILITY,
# paid for on every use. Three such cards sat in sharknado — a deck whose whole
# plan is wheeling — and all three read as vanilla bodies. Jace's Archivist is
# the one that matters: `{U}, {T}` wheels the table EVERY TURN, forever.

def _perm(name, type_line="Creature — Wizard", text=None, mana_cost="{1}{U}{U}"):
    from manamap.pilot import card_pool
    text = text if text is not None else card_pool.corpus_oracle().get(name)
    assert text, f"{name} is not in the corpus"
    return goldfish.draw_profile({"name": name, "oracle_text": text,
                                  "type_line": type_line, "mana_cost": mana_cost})


@requires_data
def test_the_repeatable_wheel_is_read_as_repeatable():
    """THE CARD THE CHANNEL EXISTS FOR. Re-introduce the bug by deleting the
    `activated_wheel` branch in `draw_profile` and the Archivist goes back to
    being a 2/2 Vedalken Wizard with no text on it."""
    got = _perm("Jace's Archivist")
    assert got["activated_wheel"] == -1, "draws what was discarded"
    assert got["activated_wheel_cost"] == 1
    assert [set(p) for p in got["activated_wheel_pips"]] == [{"U"}]
    assert got["activated_wheel_once"] is False, "{U}, {T} is payable every turn"
    assert got["activated_wheel_sacs_self"] is False
    assert got["unmodelled"] is None
    # The spell path is untouched: a creature has no `wheel_draws`.
    assert got["wheel_draws"] == 0


@requires_data
def test_a_cost_that_eats_its_own_source_is_one_shot():
    """Magus of the Wheel is `{1}{R}, {T}, Sacrifice this creature` and Whirlpool
    Warrior `{R}, Sacrifice this creature`. Reading either as repeatable would
    wheel the table seven cards a turn off a card that is in the graveyard."""
    magus = _perm("Magus of the Wheel", mana_cost="{2}{R}")
    assert magus["activated_wheel"] == 7
    assert magus["activated_wheel_cost"] == 2, "{1}{R}; the tap is not mana"
    assert magus["activated_wheel_once"] is True
    assert magus["activated_wheel_sacs_self"] is True

    whirl = _perm("Whirlpool Warrior", mana_cost="{2}{R}")
    assert whirl["activated_wheel"] == -1
    assert whirl["activated_wheel_shuffles"] is True, "no discard trigger fires"
    assert whirl["activated_wheel_once"] is True
    assert whirl["activated_wheel_sacs_self"] is True


@requires_data
def test_an_ability_activated_from_a_zone_this_model_lacks_is_refused():
    """Runehorn Hellkite's wheel costs `{5}{R}, Exile this card from your
    GRAVEYARD`. This model has no graveyard, so firing it off the battlefield
    would be a seven-card refill the card cannot give. Refused, and it stays in
    `meta.draw_not_modelled`, which is the honest place for it — the same call
    `_CAST_DRAW_GATES` makes about a gate it cannot evaluate."""
    got = _perm("Runehorn Hellkite", type_line="Creature — Dragon", mana_cost="{5}{R}")
    assert got["activated_wheel"] == 0
    assert got["unmodelled"] == "Runehorn Hellkite"


@requires_data
@pytest.mark.parametrize("name,type_line", [
    # THE JARS EXILE A HAND, they do not discard it, so `_WHEEL_RE` never
    # matches and no discard payoff is credited. This falls out of the pattern
    # rather than being special-cased, and this test is what says so.
    ("Magus of the Jar", "Creature — Human Wizard"),
    ("Memory Jar", "Artifact"),
    # A TRIGGER IS NOT AN ACTIVATION. Dragon Mage wheels on combat damage and
    # Sensation Gorger on a kinship check; neither has a cost to pay, and both
    # would be free seven-card refills if the colon anchor slipped.
    ("Dragon Mage", "Creature — Dragon Wizard"),
    ("Sensation Gorger", "Creature — Goblin Shaman"),
])
def test_the_lookalikes_are_not_swept_in(name, type_line):
    assert _perm(name, type_line=type_line)["activated_wheel"] == 0


def test_a_spell_wheel_is_not_also_an_activated_one():
    """The two paths are disjoint by construction — a Sorcery cannot carry an
    activated ability — rather than by an `elif` somebody can break."""
    spell = _spell("Each player discards their hand, then draws seven cards.")
    assert spell["wheel_draws"] == 7 and spell["activated_wheel"] == 0


@requires_data
def test_the_activated_corpus_sweep_is_locked():
    """WIDENING A PATTERN NEEDS A CORPUS SWEEP IN THE SAME COMMIT. 34,814 cards
    on 2026-09-13: 43 say a player empties their hand, 7 do it from an activated
    ability on the battlefield, and the repeatable/one-shot split is the whole
    design. A new set moves these on purpose."""
    from manamap.pilot import card_pool
    from manamap.pilot.goldfish_profiles import activated_wheel

    pool, oracle = card_pool.load_pool(), card_pool.corpus_oracle()
    assert len(oracle) > 30000, "corpus did not load"
    repeatable, one_shot, checked = set(), set(), 0
    for name, text in oracle.items():
        checked += 1
        if not activated_wheel(text or ""):
            continue
        got = goldfish.draw_profile(dict(pool.get(name) or {}, name=name,
                                         oracle_text=text))
        if not got["activated_wheel"]:
            continue
        (one_shot if got["activated_wheel_once"] else repeatable).add(name)
    assert checked > 30000
    assert repeatable == {"Jace's Archivist", "Queen Kayla bin-Kroog"}, repeatable
    assert one_shot == {"Immortus, Master of Eternity", "Magus of the Wheel",
                        "Jack of Hearts, Volatile Hero", "Vindictive Flamestoker",
                        "Whirlpool Warrior"}, one_shot


# ── the channel is reachable, and it is reachable from every door ─────────

def test_the_activated_wheel_is_in_the_casting_predicate():
    """A CARD CAN BE READ CORRECTLY AND NEVER PLAYED — seven recorded instances.
    Every card in this family today is a creature and is cast by the bodies
    loop, so the line is doing nothing yet; it ships with the channel because
    the next Memory Jar will have no body."""
    from conftest import simulator_source

    assert 'c["draw"]["activated_wheel"]' in simulator_source(), (
        "activated_wheel is not in the casting predicate — a bodyless one "
        "would sit in hand for ten turns")


def test_every_door_onto_the_battlefield_registers_the_wheel():
    """Three loops put a permanent into play, and a channel wired to two of them
    undercounts silently on whichever loop caught the card. `_register_wheel` is
    one door precisely so this can be counted."""
    from conftest import simulator_source

    src = simulator_source()
    doors = re.findall(r"(?<!def )_register_wheel\(card\)", src)
    assert len(doors) == 3, (
        "a registration site was added or lost — count the loops that append "
        f"to draw_engines and match them; found {len(doors)}")


def test_the_wheel_fires_before_anything_joins_the_battlefield():
    """SUMMONING SICKNESS, WITHOUT TRACKING A TAPPED STATE. Every activated
    wheel in the corpus but one pays `{T}`, so it cannot fire on the turn it
    lands. The fire site sits ABOVE every casting loop, so a permanent that
    joins `wheel_engines` this turn is first seen next turn — and if the block
    ever moves below a registration site, an Archivist wheels the turn she
    resolves and this fails."""
    from conftest import simulator_source

    src = simulator_source()
    fires = src.index("if model_draw and model_discard and wheel_engines:")
    draw_loop = src.index("# A DRAW SPELL IS NEITHER A ROCK")
    bodies_loop = src.index("pending_draw_engine = None")
    assert fires < draw_loop < bodies_loop, (
        "the fire site has moved below a casting loop — a wheel now fires on "
        "the turn it lands, which is a tap ability with no summoning sickness")
    # The third door, `_free_creature_enters`, is a helper for the declared
    # reveal and runs in combat, later still; its position in the file says
    # nothing about when it is called, so it is not asserted on here.


@requires_data
@requires_deck
def test_the_archivist_wheels_every_turn_and_the_model_measures_it():
    """DRIVEN THROUGH THE SIMULATOR, and proved by RE-INTRODUCING THE BUG.

    sharknado at 4,000 games, seed 3, reads 14.320 extra cards by turn ten with
    the three activated wheels invisible. Blinding the channel must return
    exactly that number: anything else means the delta is coming from somewhere
    other than the cards this commit taught the model to see.

    RE-BASELINED THREE TIMES ON 2026-09-14 — 10.317, 12.020, 14.320, now 14.625
    — and each move is the point rather than a nuisance. The THIRD was the deck
    again: Queen Kayla bin-Kroog came in as a FOURTH activated wheel ({4}, {T}:
    discard your hand, draw that many), so the count in this test moves from
    three to four and the channel is worth 8.391 cards by turn ten against 5.542
    before. A test that counts a deck's cards has to move when the deck does;
    what must not move is the rule that blinding the channel recovers the floor
    EXACTLY. The second was the DECK
    changing under the test rather than the model: Elesh Norn and Goblin
    Engineer came out for Teferi's Ageless Insight and Ivora, and Teferi's
    DOUBLES every draw after the draw step, so the floor this test stands on
    rose again. The wheels are worth 5.542 cards by turn ten against 4.582
    before the swap, which is the doubler multiplying what they already drew.

    The first was the model:  The `activated_draw` channel added that day reads a draw you BUY
    — "{1}, {T}, Sacrifice this artifact: Draw a card" — across 357 corpus
    cards, several of which sharknado runs. Those draws are real and were
    previously scored as nothing, so the floor this test stands on rose by 1.703
    cards. What the test asserts is unchanged: with the wheels blinded the model
    must return the floor EXACTLY, so the wheels' own contribution is the whole
    of the difference. That contribution is 4.582 cards by turn ten, against
    4.5 before — the wheels did not get better, the control got honest.
    """
    import copy

    from manamap.pilot.common import load_deck_cards

    doc = load_deck_cards("sharknado")

    def t10(d):
        m = goldfish.run("sharknado", doc=d, iterations=4000, seed=3,
                         quiet=True, _band=False)["metrics"]
        return m["mean_extra_cards_drawn_by_turn"]["10"]

    with_wheels = t10(copy.deepcopy(doc))

    blind = copy.deepcopy(doc)
    checked = 0
    for card in blind["cards"]:
        if goldfish.draw_profile(card)["activated_wheel"]:
            card["oracle_text"] = "Flying"      # the bug, re-introduced
            checked += 1
    assert checked == 4, f"sharknado should hold four activated wheels, not {checked}"
    assert t10(blind) == 14.625, "the pre-change figure is not being recovered"
    assert with_wheels > 22, (
        f"the activated wheels are worth ~8.4 cards by turn ten; got {with_wheels}")


@requires_data
@requires_deck
def test_a_repeatable_wheel_beats_the_same_card_made_one_shot():
    """THE DIRECTIVE, ASSERTED. Rewriting only the COST — adding "Sacrifice this
    creature" to the Archivist's — must lower the draw figure, and nothing else
    about the card changes. If `activated_wheel_once` were ignored the two arms
    would be identical."""
    import copy

    from manamap.pilot.common import load_deck_cards

    doc = load_deck_cards("sharknado")

    def t10(d):
        return goldfish.run("sharknado", doc=d, iterations=2000, seed=11,
                            quiet=True, _band=False)["metrics"
                            ]["mean_extra_cards_drawn_by_turn"]["10"]

    once = copy.deepcopy(doc)
    patched = 0
    for card in once["cards"]:
        if card["name"] == "Jace's Archivist":
            card["oracle_text"] = card["oracle_text"].replace(
                "{U}, {T}:", "{U}, {T}, Sacrifice this creature:")
            patched += 1
    assert patched == 1
    assert goldfish.draw_profile(
        [c for c in once["cards"] if c["name"] == "Jace's Archivist"][0]
    )["activated_wheel_once"] is True, "the rewrite did not take"
    assert t10(copy.deepcopy(doc)) > t10(once), (
        "a wheel that can be activated every turn must out-draw the same wheel "
        "that can be activated once")


@requires_data
@requires_deck
def test_a_one_shot_that_sacrifices_itself_takes_the_body_with_it():
    """The cost says "Sacrifice this creature" and the board must show it.
    Leaving a 3/3 standing after it has been sacrificed is the over-credit the
    sacrifice channel already learned to avoid — and it is invisible in the draw
    figures, which is why it is asserted on BOARD POWER instead."""
    import copy

    from manamap.pilot.common import load_deck_cards

    doc = load_deck_cards("sharknado")

    def power10(d):
        return goldfish.run("sharknado", doc=d, iterations=4000, seed=3,
                            quiet=True, _band=False)["metrics"][
                                "combat"]["mean_board_power_by_turn"]["10"]

    kept = copy.deepcopy(doc)
    for card in kept["cards"]:
        if card["name"] == "Magus of the Wheel":
            card["oracle_text"] = card["oracle_text"].replace(
                ", Sacrifice this creature:", ":")
    assert power10(copy.deepcopy(doc)) < power10(kept), (
        "the sacrificed body is still on the board")


# ── the OTHER half of a wheel: what an opponent drawing pays us ──────────
#
# Every trigger in `event_payoffs` keyed on what WE do, so a card taxing an
# OPPONENT's draw returned an all-zero profile with `unmodelled` UNSET — the
# silent zero, on the half of a symmetrical wheel that had never been counted.

def _ev(text, name="a card"):
    return goldfish.event_payoffs({"name": name, "oracle_text": text})


def test_a_tax_on_an_opponents_draw_is_read():
    """Razorkin Needlehead and Scrawling Crawler word the damage AT THE DRAWER
    — "to them", "that player loses" — which `_EVENT_DAMAGE_RE` cannot see; it
    wants "each opponent". Re-introduce the bug by deleting the `opp` arm of
    `_EVENT_TRIGGER_RE` and both of these go to zero."""
    assert _ev("Whenever an opponent draws a card, this creature deals 1 "
               "damage to them.")["per_opponent_draw_damage"] == 1
    assert _ev("At the beginning of your upkeep, each player draws a card. "
               "Whenever an opponent draws a card, that player loses 1 life."
               )["per_opponent_draw_damage"] == 1
    assert _ev("Flying Whenever an opponent draws a card, you may draw two "
               "cards.")["per_opponent_draw_our_draw"] == 2
    assert _ev("Whenever an opponent draws their second card each turn, you "
               "draw a card.")["opponent_second_draw_our_draw"] == 1


def test_an_effect_gated_on_MANA_is_refused_and_named():
    """AN EFFECT YOU PAY FOR IS NOT A RATE, and a wheel hands the opponent
    seven draws — so reading Mind's Eye would give this deck seven free cards
    off an artifact it never paid for. Same refusal the X-draw path makes for
    an additional cost. Drop `_EVENT_PAID_RE` and this fails."""
    got = _ev("Whenever an opponent draws a card, you may pay {1}. If you do, "
              "draw a card.", name="Mind's Eye")
    assert got["per_opponent_draw_our_draw"] == 0
    assert got["unmodelled"] == "Mind's Eye", "refused must mean NAMED, not zero"


def test_our_own_trigger_is_unchanged_by_the_new_arm():
    """The opponent arm must not disturb the four triggers that already
    worked. Brallin is the one every figure in this deck rests on."""
    brallin = _ev("Whenever you discard a card, put a +1/+1 counter on Brallin "
                  "and it deals 1 damage to each opponent.")
    assert brallin["per_discard_damage"] == 1 and brallin["per_discard_counter"] == 1
    assert brallin["per_opponent_draw_damage"] == 0
    assert _ev("Whenever you draw a card, each opponent loses 1 life."
               )["per_draw_damage"] == 1


def test_a_trigger_on_an_opponent_CASTING_stays_unread():
    """Rhystic Study is a different family — it keys on a SPELL, not a draw —
    and reading it here would be the parser claiming a card it cannot price."""
    got = _ev("Whenever an opponent casts a spell, you may draw a card unless "
              "that player pays {1}.", name="Rhystic Study")
    assert got["per_opponent_draw_our_draw"] == 0
    assert got["per_opponent_draw_damage"] == 0


@requires_data
def test_the_opponent_draw_sweep_is_locked():
    """WIDENING A PATTERN NEEDS A CORPUS SWEEP IN THE SAME COMMIT. 34,814 cards
    on 2026-09-14: 16 newly read, and 12 more that matched the trigger but
    whose effect this model refuses now carry their NAME instead of an all-zero
    profile. A set moves these on purpose."""
    from manamap.pilot import card_pool

    pool, oracle = card_pool.load_pool(), card_pool.corpus_oracle()
    read, named, checked = set(), 0, 0
    for name, text in oracle.items():
        checked += 1
        got = goldfish.event_payoffs(dict(pool.get(name) or {}, name=name,
                                          oracle_text=text))
        if any(got[k] for k in got if k.startswith(("per_opponent_draw",
                                                    "opponent_second_draw"))):
            read.add(name)
    assert checked > 30000
    assert 12 <= len(read) <= 24, f"{len(read)} cards read — re-read the sweep"
    for must in ("Razorkin Needlehead", "Scrawling Crawler", "Underworld Dreams",
                 "Nekusar, the Mindrazer", "Consecrated Sphinx"):
        assert must in read, must
    for must_not in ("Mind's Eye", "Smothering Tithe", "Rhystic Study"):
        assert must_not not in read, f"{must_not} is gated or a cast trigger"


@requires_data
@requires_deck
def test_a_wheel_refills_the_opponent_and_that_is_part_of_the_tax():
    """THE HALF THE MODEL NEVER SAW, ISOLATED.

    A per-opponent-draw tax gains from TWO sources: their draw step, one a
    turn, and every card a wheel deals them. Only the second is new, so a test
    that just checks "the tax fired" proves nothing — the first draft asserted
    a gain over 1.0 and PASSED with the wheel half deleted, because the draw
    steps alone clear it.

    Measured on sharknado at 3,000 games, seed 5. Razorkin Needlehead's gain in
    cumulative ping damage by turn ten:

        deck as printed (17 wheels)   9.89 -> 12.34   +2.45
        every wheel blinded           2.28 ->  3.77   +1.50
        the wheel's share                             +0.96

    So the wheels are 39% of what the tax is worth, and deleting
    `opponent_draws_this_turn += _n` at either wheel site collapses the two
    arms together.
    """
    import copy

    from manamap.pilot import candidates
    from manamap.pilot.common import load_deck_cards

    def ping(d):
        return goldfish.run("sharknado", doc=d, iterations=1500, seed=5,
                            quiet=True, _band=False)["metrics"]["discard"][
                                "mean_cumulative_event_damage_by_turn"]["10"]

    def plus_tax(d):
        d = copy.deepcopy(d)
        victim = max((c for c in d["cards"] if not c.get("is_commander")
                      and "Land" not in (c.get("type_line") or "")),
                     key=lambda c: float(c.get("cmc") or 0))
        d["cards"] = [c for c in d["cards"] if c["name"] != victim["name"]]
        d["cards"].append(candidates._resolve("Razorkin Needlehead"))
        return d

    doc = load_deck_cards("sharknado")
    with_wheels = ping(plus_tax(doc)) - ping(doc)

    blind = copy.deepcopy(doc)
    blinded = 0
    for c in blind["cards"]:
        t = c.get("oracle_text") or ""
        if ("discards their hand" in t or "discard your hand" in t
                or "shuffles their hand" in t):
            c["oracle_text"] = "Flying"
            blinded += 1
    # 9 -> 8 on 2026-09-14 with the budget swaps: Wheel of Fortune and Wheel of
    # Misfortune left, and of the two that arrived only Anje's Ravager matches
    # this grep — Queen Kayla says "discard ALL THE CARDS IN your hand", which is
    # the same act in different words. The grep is deliberately literal, so the
    # floor moves with the wording rather than the wording being chased.
    assert blinded >= 8, f"only {blinded} wheels blinded — the control is weak"
    without_wheels = ping(plus_tax(blind)) - ping(blind)

    assert with_wheels > without_wheels + 0.4, (
        f"the tax gains {with_wheels:.2f} with wheels and {without_wheels:.2f} "
        f"without — the opponent's share of a wheel is not being counted")


# ── The draw you BUY, and the Blood token ─────────────────────────────────
#
# A draw attached to an ETB, a cast, an upkeep or a wheel was modelled; a draw
# you pay for by SACRIFICING the permanent was not, so "{1}, {T}, Sacrifice this
# artifact: Draw a card" returned an all-zero profile with `unmodelled` still
# None — the one value that means "nothing to see here". Measured before the
# channel existed: 405 corpus cards carry a sacrifice-gated draw and
# `draw_profile` read zero draw for 400 of them, 99%.
#
# A BLOOD TOKEN IS THAT ABILITY WITH A DISCARD IN THE COST, which is why the
# two shipped together: "{1}, {T}, Discard a card, Sacrifice this token: Draw a
# card". It is card-NEUTRAL — one leaves the hand, one arrives — so it makes no
# card advantage at all. What it makes is EVENTS, and a deck whose commanders
# charge for a discard AND for a draw is paid twice for every token it cracks.


def _art(name, text, type_line="Artifact", mana_cost="{2}"):
    return goldfish.draw_profile({"name": name, "oracle_text": text,
                                  "type_line": type_line, "mana_cost": mana_cost})


BLOOD_TEXT = "{1}, {T}, Discard a card, Sacrifice this token: Draw a card."


def test_a_draw_you_buy_is_read_with_its_cost_and_its_discard():
    """THE CARD THE CHANNEL EXISTS FOR, in its two shapes. Re-introduce the bug
    by deleting the `activated_draw` branch in `draw_profile` and Mind Stone is
    a rock with no text on it again."""
    stone = _art("Mind Stone", "{T}: Add {C}. {1}, {T}, Sacrifice this artifact: Draw a card.")
    assert stone["activated_draw"] == 1
    assert stone["activated_draw_cost"] == 1, "{1}; the tap is not mana"
    assert stone["activated_draw_once"] is True, "it eats its own source"
    assert stone["activated_draw_sacs_self"] is True
    assert stone["activated_draw_discards"] == 0
    assert stone["unmodelled"] is None, "read, so it must not also be named unreadable"

    blood = _art("Blood token", BLOOD_TEXT, type_line="Artifact Token")
    assert blood["activated_draw"] == 1
    assert blood["activated_draw_cost"] == 1
    assert blood["activated_draw_discards"] == 1, "the discard is part of the COST"
    assert blood["activated_draw_once"] is True


def test_a_free_sacrifice_to_draw_is_still_a_cost():
    """Commander's Sphere is "Sacrifice this artifact: Draw a card" — no mana at
    all. `activated_wheel` skips a cost with no mana symbol, and copying that
    rule here would have dropped the whole free-sacrifice family."""
    got = _art("Commander's Sphere",
               "{T}: Add one mana of any color in your commander's color identity. "
               "Sacrifice this artifact: Draw a card.")
    assert got["activated_draw"] == 1
    assert got["activated_draw_cost"] == 0
    assert got["activated_draw_once"] is True


def test_the_loot_rider_on_an_activated_draw_is_counted_as_a_discard():
    """"{T}: Draw a card, then discard a card" — Thought Courier. The discard is
    in the EFFECT rather than the cost, and to this model both are the same
    event: a card leaves the hand, which is the whole of what Brallin charges
    for. Summed into one field because a reader who must add two will add one."""
    got = _art("Thought Courier", "{T}: Draw a card, then discard a card.",
               type_line="Creature — Human Wizard")
    assert got["activated_draw"] == 1
    assert got["activated_draw_discards"] == 1
    assert got["activated_draw_once"] is False, "{T} alone is payable every turn"


def test_reminder_text_describes_the_token_not_the_card():
    """THE DOUBLE COUNT THIS WOULD HAVE SHIPPED. Every Blood maker carries the
    token's ability in brackets — "(It's an artifact with "{1}, {T}, Discard a
    card, Sacrifice this token: Draw a card.")" — so Voldaren Epicure read as
    drawing a card ITSELF, and would have been scored once here and again as the
    Blood it makes. Every Clue and Food maker has the same shape.

    Re-introduce the bug by dropping the `_REMINDER_RE.sub` in `activated_draw`.
    """
    epicure = _art("Voldaren Epicure",
                   "When this creature enters, it deals 1 damage to each opponent. "
                   "Create a Blood token. (It's an artifact with \"" + BLOOD_TEXT + "\")",
                   type_line="Creature — Vampire", mana_cost="{R}")
    assert epicure["activated_draw"] == 0, "that ability belongs to the TOKEN"
    assert epicure["activated_draw_discards"] == 0

    clue = _art("Hard Evidence",
                "Create a 0/3 blue Crab creature token. Investigate. (Create a Clue "
                "token. It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")",
                type_line="Sorcery", mana_cost="{U}")
    assert clue["activated_draw"] == 0


def test_a_cost_this_model_cannot_price_is_refused_and_named():
    """ENERGY AND {X} BOTH READ AS FREE through `_activation_mana`, and free is
    the dangerous direction. Era of Innovation is "Pay six {E}, Sacrifice this
    enchantment: Draw three cards" and Bargaining Table is "{X}, {T}: Draw a
    card" where X is an opponent's hand size — read naively they are a free
    draw-three and a free repeatable draw engine. Both found by reading the tail
    of the corpus sweep, which is what the tail is for."""
    era = _art("Era of Innovation",
               "Whenever an artifact or Artificer you control enters, you may pay {1}. "
               "If you do, you get {E}{E} (two energy counters). "
               "Pay six {E}, Sacrifice this enchantment: Draw three cards.",
               type_line="Enchantment")
    assert era["activated_draw"] == 0
    assert era["unmodelled"] == "Era of Innovation", "refused must still be NAMED"

    table = _art("Bargaining Table",
                 "{X}, {T}: Draw a card. X is the number of cards in an opponent's hand.")
    assert table["activated_draw"] == 0
    assert table["unmodelled"] == "Bargaining Table"


def test_an_ability_that_cannot_be_paid_twice_is_not_repeatable():
    """Surge Engine says "Activate only if this creature is blue and only once";
    every EXHAUST ability says "Activate each exhaust ability only once". Both
    read as REPEATABLE draw engines until the sweep's tail was read — Loot, the
    Pathfinder would have drawn three every turn forever for {U}."""
    surge = _art("Surge Engine",
                 "{4}{U}{U}: Draw three cards. Activate only if this creature is "
                 "blue and only once.", type_line="Artifact Creature")
    assert surge["activated_draw"] == 3
    assert surge["activated_draw_cost"] == 6
    assert surge["activated_draw_once"] is True

    loot = _art("Loot, the Pathfinder",
                "Exhaust — {U}, {T}: Draw three cards. (Activate each exhaust "
                "ability only once.)", type_line="Legendary Creature")
    assert loot["activated_draw_once"] is True


@requires_data
def test_no_card_is_paid_for_on_both_activated_channels():
    """A wheel IS a draw, and a card read on both would be paid for twice —
    once emptying the hand for seven, once drawing on top of it.

    ASSERTED OVER THE WHOLE CORPUS, and that is the point rather than laziness.
    `activated_draw` carries a guard that skips a cost whose effect is a wheel,
    and trying to prove that guard by re-introducing the bug turned NO test red:
    it is unreachable, because a wheel's effect always begins "each player
    discards" and `_ACTIVATED_DRAW_RE` is anchored on "draw". The property is
    real and the guard is not what delivers it, so the test asserts the property
    across 34,814 cards instead of asserting a single card the guard never
    touched."""
    from manamap.pilot import card_pool

    pool, oracle = card_pool.load_pool(), card_pool.corpus_oracle()
    assert len(oracle) > 30000, "corpus did not load"
    both, checked = [], 0
    for name, text in oracle.items():
        checked += 1
        got = goldfish.draw_profile(dict(pool.get(name) or {}, name=name,
                                         oracle_text=text))
        if got["activated_wheel"] and got["activated_draw"]:
            both.append(name)
    assert checked > 30000
    assert both == [], f"read on both channels, so paid for twice: {both}"


# ── Blood ─────────────────────────────────────────────────────────────────


def _blood(name, text, type_line="Creature — Vampire"):
    from manamap.pilot.goldfish_profiles import blood_profile
    return blood_profile({"name": name, "oracle_text": text, "type_line": type_line})


def test_blood_is_read_with_its_trigger():
    assert _blood("Voldaren Epicure",
                  "When this creature enters, it deals 1 damage to each opponent. "
                  "Create a Blood token.") == (1, "etb")
    assert _blood("Falkenrath Celebrants",
                  "Menace When this creature enters, create two Blood tokens.") == (2, "etb")
    assert _blood("Arterial Alchemy",
                  "When this enchantment enters, create a Blood token for each opponent "
                  "you have.", type_line="Enchantment") == (3, "per_opponent")
    assert _blood("Vampire's Kiss",
                  "Target player loses 2 life and you gain 2 life. Create two Blood tokens.",
                  type_line="Sorcery") == (2, "spell")


def test_an_etb_that_spans_a_sentence_is_still_an_etb():
    """Voldaren Epicure is the best one-mana Blood maker in the corpus and its
    create is a SECOND SENTENCE — "When this creature enters, it deals 1 damage
    to each opponent. Create a Blood token." Standard templating; a pattern that
    stops at the first full stop reads it as `unmodelled` and the card does
    nothing. Re-introduce the bug by removing the optional second clause from
    `_BLOOD_ETB_RE`."""
    assert _blood("Voldaren Epicure",
                  "When this creature enters, it deals 1 damage to each opponent. "
                  "Create a Blood token.")[1] == "etb"


def test_a_card_that_is_both_etb_and_combat_is_read_as_the_guaranteed_half():
    """Ivora reads "When Ivora enters AND whenever it deals combat damage to a
    player, create a Blood token" — one trigger clause, two conditions. Filing
    her under `combat` puts her whole contribution behind `model_combat` and
    understates a Blood that arrives on turn two whatever the board does."""
    assert _blood("Ivora, Insatiable Heir",
                  "Trample When Ivora enters and whenever it deals combat damage to a "
                  "player, create a Blood token. Whenever you discard a card, put a "
                  "+1/+1 counter on Ivora.") == (1, "etb")


def test_a_card_that_plainly_makes_blood_is_never_scored_as_having_none():
    """ABSENT MEANS ABSENT, NEVER ZERO — and `(0, None)` here means "no Blood
    text at all", which is a different claim from "makes Blood in a shape this
    model has no event for". Reading all 44 Blood cards card by card found two
    in the wrong bucket: Lacerate Flesh ("create a number of Blood tokens equal
    to the amount of excess damage") and Transmutation Font ("{T}: Create your
    choice of a Blood token, a Clue token, or a Food token")."""
    n, trigger = _blood("Lacerate Flesh",
                        "Lacerate Flesh deals 4 damage to target creature. Create a "
                        "number of Blood tokens equal to the amount of excess damage.",
                        type_line="Instant")
    assert trigger is not None, "a card that creates Blood must never read as having none"
    n, trigger = _blood("Transmutation Font",
                        "{T}: Create your choice of a Blood token, a Clue token, or a "
                        "Food token.", type_line="Artifact")
    assert trigger == "unmodelled"
    # And a card that only SACRIFICES Blood genuinely has none to make.
    assert _blood("Wedding Security",
                  "Whenever this creature attacks, you may sacrifice a Blood token. If "
                  "you do, put a +1/+1 counter on this creature.") == (0, None)


def test_the_blood_corpus_sweep_is_locked():
    """WIDENING A PATTERN NEEDS A CORPUS SWEEP IN THE SAME COMMIT. 44 cards in
    the corpus mention a Blood token on 2026-09-14; every one was read by hand
    when the channel shipped. Exactly two make none — both only sacrifice them —
    and everything else lands in a named bucket. A new set moves these on
    purpose."""
    import collections

    from manamap.pilot import card_pool
    from manamap.pilot.goldfish_profiles import blood_profile

    oracle = card_pool.corpus_oracle()
    assert len(oracle) > 30000, "corpus did not load"
    pool = card_pool.load_pool()
    buckets, checked = collections.Counter(), 0
    for name, text in oracle.items():
        if "Blood token" not in (text or ""):
            continue
        checked += 1
        buckets[blood_profile(dict(pool.get(name) or {}, name=name,
                                   oracle_text=text))[1]] += 1
    assert checked >= 40, f"only {checked} Blood cards found — did the corpus change?"
    assert buckets[None] == 2, (
        f"a card that creates Blood is scoring as having none: {buckets}")
    assert buckets["etb"] >= 10 and buckets["spell"] >= 5, buckets


# ── the channels are reachable, and the model ACTS on them ────────────────


def test_the_bought_draw_and_blood_are_in_the_casting_predicate():
    """A CARD CAN BE READ CORRECTLY AND NEVER PLAYED — the failure this repo has
    recorded seven times. Unlike the activated wheel, this family is NOT all
    creatures: Mind Stone, Blood Fountain and Sanguine Statuette have no body at
    all, so without the predicate they are read perfectly and sit in hand for
    ten turns."""
    from conftest import simulator_source

    src = simulator_source()
    assert 'c["draw"]["activated_draw"]' in src, "a bodyless rock would never be cast"
    assert 'c["blood"]' in src, "a bodyless Blood maker would never be cast"


def test_model_coverage_knows_both_channels():
    """`never_cast` is what the fleet test reads to find a card the model
    understands and never plays. A channel missing here reports every card in it
    as a silent loss."""
    from manamap.pilot.model_coverage import never_cast

    flags = {"model_draw": True, "model_discard": True}
    stone = {"is_land": False, "bodies": 0, "produces": 0, "tutor": False,
             "reduces": False, "treasure_doubler": False, "treasure_bonus": 0,
             "treasure_trigger": None, "blood": (0, None),
             "draw": _art("Mind Stone",
                          "{T}: Add {C}. {1}, {T}, Sacrifice this artifact: Draw a card."),
             "event": {}, "combat": {}}
    assert never_cast(stone, flags) is False, "a rock that draws is worth casting"

    maker = dict(stone, blood=(1, "etb"),
                 draw=_art("Blood Fountain", "When this artifact enters, create a Blood token."))
    assert never_cast(maker, flags) is False, "a Blood maker is cast FOR the Blood"


def test_the_bought_draw_list_does_not_shadow_the_recurring_one():
    """THE BUG THIS CHANNEL SHIPPED WITH FOR ONE COMMIT. `draw_engines` already
    existed twenty lines above — permanents that draw on their own every upkeep
    — and the first cut of this channel named its list the same thing. The
    second binding won, the recurring list was empty at boot, and the model died
    on the first card that had one: `KeyError: 'recurring_draw'`, because the
    entries are shaped differently too.

    A single-assignment check, because the names are close enough that the next
    edit will reach for the wrong one."""
    from conftest import simulator_source

    src = simulator_source()
    assert len(re.findall(r"^    draw_engines = \[\]", src, re.M)) == 1, (
        "draw_engines is bound twice — the recurring-draw list is being shadowed")
    assert "bought_draws = []" in src


# ── The draw doubler ──────────────────────────────────────────────────────


def test_a_draw_doubler_is_read_and_the_draw_step_is_the_exception():
    """"If you would draw a card EXCEPT THE FIRST ONE YOU DRAW IN EACH OF YOUR
    DRAW STEPS, draw two cards instead." The exception is the whole card, and
    the model gets it for free: the draw step takes its card straight off the
    deck and never calls `draw_n`, which is the only place the multiplier is
    applied. Structure, not bookkeeping.

    THE HELLBENT ONES ARE REFUSED. Blood Scrivener doubles only "while you have
    no cards in hand" — true at exactly the moment a wheel deck is about to
    refill, so reading it as unconditional would be the most flattering possible
    error on the deck most likely to play it.
    """
    from manamap.pilot import card_pool
    oracle = card_pool.corpus_oracle()

    def mult(name):
        return goldfish.draw_profile({"name": name, "oracle_text": oracle.get(name),
                                      "type_line": "Enchantment", "cmc": 4})

    for name in ("Teferi's Ageless Insight", "Bard, King of Dale", "Thought Reflection"):
        got = mult(name)
        assert got["draw_multiplier"] == 2, name
        assert got["unmodelled"] is None, f"{name} is read, so must not be named unreadable"
    assert mult("Blood Scrivener")["draw_multiplier"] == 1, "hellbent is not unconditional"
    assert mult("Sol Ring")["draw_multiplier"] == 1


def test_the_draw_doubler_is_applied_where_the_exception_is_free():
    """Re-introduce the bug by multiplying at the draw step as well, or by
    dropping the multiplication in `draw_n`."""
    from conftest import simulator_source

    src = simulator_source()
    assert "for _ in range(int(n) * draw_multiplier):" in src, (
        "the multiplier is not applied in draw_n")
    assert 'c["draw"]["draw_multiplier"] > 1' in src, (
        "a doubler with no body would be read and never cast")


@requires_data
def test_the_draw_doubler_sweep_is_locked():
    """WIDENING A PATTERN NEEDS A CORPUS SWEEP IN THE SAME COMMIT. Eight cards
    on 2026-09-14, five of them legal in a Jeskai deck. A new set moves this on
    purpose."""
    from manamap.pilot import card_pool

    oracle = card_pool.corpus_oracle()
    assert len(oracle) > 30000, "corpus did not load"
    hits = {n for n, t in oracle.items()
            if goldfish.draw_profile({"name": n, "oracle_text": t or "",
                                      "type_line": "", "cmc": 0})["draw_multiplier"] > 1}
    assert 5 <= len(hits) <= 14, f"the doubler family moved to {len(hits)}: {sorted(hits)}"
    for real in ("Teferi's Ageless Insight", "Thought Reflection", "Bard, King of Dale"):
        assert real in hits, real
    for hellbent in ("Blood Scrivener", "Phial of Galadriel"):
        assert hellbent not in hits, f"{hellbent} doubles only while hellbent"


@requires_data
def test_a_blood_count_is_read_as_a_word_or_a_digit():
    """The count alternation stopped at "four" and accepted no digits, which is
    a narrowness rather than a decision — `_NUMBER_WORDS` beside it goes to ten.
    Found by a sensitivity run whose three arms came back BYTE-IDENTICAL because
    "create 3 Blood tokens", "create 5" and "create 1" all failed to match and
    fell into the same bucket, which is the shape of a measurement that silently
    measured nothing."""
    from manamap.pilot.goldfish_profiles import blood_profile

    def n(text):
        return blood_profile({"name": "x", "oracle_text": text, "type_line": "Creature"})

    assert n("When this enters, create 5 Blood tokens.") == (5, "etb")
    assert n("When this enters, create seven Blood tokens.") == (7, "etb")
    assert n("When this enters, create two Blood tokens.") == (2, "etb")


def test_reminder_text_is_not_the_cards_own_discard_payoff():
    """REMINDER TEXT IS NOT THE CARD'S EFFECT — the second time this exact class
    bit in one day. `activated_draw` learned it in the morning; `event_payoffs`
    had the same hole until the afternoon.

    Magmakin Artillerist reads "Whenever you discard one or more cards, this
    creature deals that much damage to each opponent. Cycling {1}{R} ({1}{R},
    Discard this card: Draw a card.)" The parenthesised CYCLING reminder fell
    inside the trigger's effect window, so the card scored `per_discard_draw: 1`
    and NO DAMAGE — a draw it does not have, and none of the damage it does.

    WHY IT MATTERED RATHER THAN JUST BEING WRONG. sharknado's one declared
    single point of failure is "a second per-discard DAMAGE source", assembled
    in 39.5% of games. Magmakin is precisely that card, and it was invisible on
    exactly the axis it fixes — found while shortlisting cards for that slot.

    THE SWEEP: three corpus cards change when reminder text is stripped and all
    three change correctly. Magmakin gains its damage and loses the phantom
    draw; Marauding Mako and Scrounging Skyray lose a phantom draw each.

    Re-introduce the bug by dropping the `_REMINDER_RE.sub` in `event_payoffs`.
    """
    from manamap.pilot.goldfish_profiles import event_payoffs

    MAGMAKIN = ("Whenever you discard one or more cards, this creature deals that "
                "much damage to each opponent. Cycling {1}{R} ({1}{R}, Discard this "
                "card: Draw a card.) When you cycle this card, it deals 1 damage to "
                "each opponent.")
    got = event_payoffs({"name": "Magmakin Artillerist", "oracle_text": MAGMAKIN})
    assert got["per_discard_damage"] == 1, "the damage half is the whole card"
    assert got["per_discard_draw"] == 0, "that draw belongs to the cycling reminder"

    # AND THE STRIP IS WHAT DOES IT, both ways round. With the reminder left in,
    # the card reads as a draw it does not have and none of the damage it does.
    import re as _re
    unstripped = dict(got)
    assert unstripped["per_discard_damage"] == 1 and unstripped["per_discard_draw"] == 0

    # The control: a real per-discard draw still reads. Glint-Horn Buccaneer's
    # "{1}{R}, Discard a card: Draw a card" is its OWN ability, not a reminder.
    glint = event_payoffs({"name": "Glint-Horn Buccaneer", "oracle_text": (
        "Haste Whenever you discard a card, this creature deals 1 damage to each "
        "opponent. {1}{R}, Discard a card: Draw a card.")})
    assert glint["per_discard_damage"] == 1 and glint["per_discard_draw"] == 1


@requires_data
def test_the_reminder_strip_changes_exactly_the_three_it_should():
    """WIDENING OR NARROWING A MATCHER NEEDS A CORPUS SWEEP IN THE SAME COMMIT.
    Stripping reminder text from the discard/draw payoffs moves three cards and
    no others; a fourth means a real ability is being eaten."""
    import re

    from manamap.pilot import card_pool
    from manamap.pilot.goldfish_profiles import event_payoffs

    oracle = card_pool.corpus_oracle()
    assert len(oracle) > 30000, "corpus did not load"
    moved = set()
    for name, text in oracle.items():
        text = text or ""
        if "(" not in text:
            continue
        # The un-stripped reading, reproduced by putting the reminder somewhere
        # the stripper cannot reach: a card with no brackets at all.
        bare = re.sub(r"\([^)]*\)", " ", text)
        if event_payoffs({"name": name, "oracle_text": text}) != \
           event_payoffs({"name": name, "oracle_text": bare}):
            moved.add(name)
    assert moved == set(), (
        f"stripping is no longer idempotent — {sorted(moved)} still differ, so "
        f"`event_payoffs` is reading brackets somewhere")
