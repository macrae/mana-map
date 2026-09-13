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
WHEELS, LOOTS, DISC, DRAW, SECOND = (25, 40, 15, 29, 20)


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

    sharknado at 4,000 games, seed 3, read 10.317 extra cards by turn ten with
    the three activated wheels invisible. Blinding the channel must return
    exactly that number: anything else means the delta is coming from somewhere
    other than the cards this commit taught the model to see.
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
    assert checked == 3, f"sharknado should hold three activated wheels, not {checked}"
    assert t10(blind) == 10.317, "the pre-change figure is not being recovered"
    assert with_wheels > 14, (
        f"the activated wheels are worth ~4.5 cards by turn ten; got {with_wheels}")


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
