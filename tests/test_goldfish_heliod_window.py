"""Heliod, the Warped Eclipse: the flip, the flash, the discount, and the window (2026-11-23).

The pilot: "teach the goldfish Heliod's discount... play it at flash after forcing massive
draw and discounting the generic mana cost... getting Heliod out and tutoring for it are
critical." Under `model_commander_flash_discount` the commander flips for {3}{U/P}; then the
main phase holds its mana while Approach is reachable, and the last opponent's end step is the
window: find Approach, force each-player draws until it costs its {W}, cast it; after the first
cast, Dig Through Time / Scroll Rack / Sensei's Top bring it back and it wins.
"""

import json

import pytest

from manamap.pilot import goldfish, model_coverage
from manamap.pilot import goldfish_library as gl
from manamap.pilot.common import deck_file
from manamap.pilot.goldfish import DeclarationError
from manamap.pilot.goldfish_profiles import window_profile

from conftest import requires_deck


def _c(name, text, type_line, cmc=1, mana_cost="{1}"):
    return {"name": name, "oracle_text": text, "type_line": type_line, "cmc": cmc,
            "mana_cost": mana_cost, "colors": [], "color_identity": [], "keywords": []}


MINE = _c("Howling Mine", "At the beginning of each player's draw step, if this artifact is "
          "untapped, that player draws an additional card.", "Artifact", 2, "{2}")
FRUITION = _c("Forced Fruition", "Whenever an opponent casts a spell, that player draws seven "
              "cards.", "Enchantment", 6, "{4}{U}{U}")
FOLIO = _c("Folio of Fancies", "Players have no maximum hand size.\n{X}{X}, {T}: Each player "
           "draws X cards.", "Artifact — Book", 2, "{1}{U}")
SKEINS = _c("Vision Skeins", "Each player draws two cards.", "Instant", 2, "{1}{U}")
RACK = _c("Scroll Rack", "{1}, {T}: Exile any number of cards from your hand face down. Put "
          "that many cards from the top of your library into your hand. Then look at the "
          "exiled cards and put them on top of your library in any order.", "Artifact", 2, "{2}")
TOP = _c("Sensei's Divining Top", "{1}: Look at the top three cards of your library, then put "
         "them back in any order.\n{T}: Draw a card, then put this artifact on top of its "
         "owner's library.", "Artifact", 1, "{1}")
DIG = _c("Dig Through Time", "Delve\nLook at the top seven cards of your library. Put two of "
         "them into your hand and the rest on the bottom of your library in any order.",
         "Instant", 8, "{6}{U}{U}")
MYSTICAL = _c("Mystical Tutor", "Search your library for an instant or sorcery card, reveal "
              "it, then shuffle and put that card on top.", "Instant", 1, "{U}")
PLANS = _c("Long-Term Plans", "Search your library for a card, then shuffle and put that card "
           "third from the top.", "Instant", 3, "{2}{U}")


def test_every_window_shape_reads_from_the_text():
    assert window_profile(MINE)["static_extra"] == 1
    assert window_profile(FRUITION)["fruition"] == 7
    assert window_profile(FOLIO)["folio"]
    assert window_profile(SKEINS)["spell_each_draw"] == 2
    assert window_profile(RACK)["scroll_rack"]
    assert window_profile(TOP)["top"]
    d = window_profile(DIG)
    assert d["dig"] == 7 and d["delve"]
    assert window_profile(MYSTICAL)["tutor_dest"] == "top"
    assert window_profile(PLANS)["tutor_dest"] == "third"


def test_a_spell_that_draws_only_for_us_is_not_a_table_draw():
    brainstorm = _c("Brainstorm", "Draw three cards, then put two cards from your hand on top "
                    "of your library in any order.", "Instant", 1, "{U}")
    assert not window_profile(brainstorm)["spell_each_draw"]


def test_the_window_cards_are_read_and_cast_only_where_heliod_is_declared():
    flags_on = {"model_commander_flash_discount": True, "model_approach": True}
    for card in (MINE, FRUITION, FOLIO, SKEINS, RACK, DIG, MATH, STROKE, CHARM):
        prof = gl.classify(dict(card))
        assert "flash_window" in model_coverage.channels_for(prof), card["name"]
        assert model_coverage.never_cast(dict(prof), flags_on) is False, card["name"]
    assert model_coverage.CHANNELS["flash_window"] == "model_commander_flash_discount"


@requires_deck
def test_the_declaration_refuses_a_commander_that_is_not_heliod():
    t = json.loads(deck_file("meren-recursion", "goldfish_targets.json").read_text())
    with pytest.raises(DeclarationError):
        goldfish.run("meren-recursion", iterations=10, seed=7, quiet=True, _band=False,
                     _targets_doc=dict(t, model_commander_flash_discount=True))


@requires_deck
def test_heliod_flips_and_wins_faster_in_the_window_than_without_it():
    """PROVED BY SWITCHING IT OFF: with the declaration Heliod flips, nearly every Approach win
    comes in the window, and the deck wins by turn ten more often than without it."""
    t = json.loads(deck_file("heliod", "goldfish_targets.json").read_text())
    base = dict(t, model_approach=True)
    on = goldfish.run("heliod", iterations=1500, seed=7, quiet=True, _band=False,
                      with_results=True, _targets_doc=dict(base, model_commander_flash_discount=True))
    off = goldfish.run("heliod", iterations=1500, seed=7, quiet=True, _band=False,
                       with_results=True, _targets_doc=dict(base, model_commander_flash_discount=False))
    rows_on, rows_off = on["_results"], off["_results"]
    assert sum(1 for r in rows_on if r["heliod_flip_turn"]) > len(rows_on) // 2
    assert not any(r["heliod_flip_turn"] for r in rows_off)
    assert any(r["approach_window_win"] for r in rows_on)
    assert not any(r["approach_window_win"] for r in rows_off)
    assert on["metrics"]["approach"]["win_by_turn"]["10"] > off["metrics"]["approach"]["win_by_turn"]["10"]


MATH = _c("Mathemagics", "Target player draws 2ˣ cards. (2⁰ = 1, 2¹ = 2, 2² = 4, 2³ = 8, 2⁴ = "
          "16, 2⁵ = 32, and so on.)", "Sorcery", 2, "{X}{X}{U}{U}")
STROKE = _c("Stroke of Genius", "Target player draws X cards.", "Instant", 3, "{X}{2}{U}")
CHARM = _c("Archmage's Charm", "Choose one —\n• Counter target spell.\n• Target player draws "
           "two cards.\n• Gain control of target nonland permanent with mana value 1 or less.",
           "Instant", 3, "{U}{U}{U}")


def test_the_target_player_draws_read_as_an_opponent_made_to_draw():
    assert window_profile(MATH)["target_draw"] == "pow2"
    assert window_profile(STROKE)["target_draw"] == "x"
    assert window_profile(CHARM)["target_draw"] == 2
    assert window_profile(SKEINS)["target_draw"] is None


@requires_deck
def test_the_discount_on_their_step_is_measured_after_the_flip_and_absent_before():
    m = goldfish.run("heliod", iterations=1500, seed=7, quiet=True, _band=False,
                     with_results=True)
    h = m["metrics"]["heliod_window"]
    assert h["mean_base_discount_by_turn"]["8"] >= 1          # their own draw step at least
    assert h["median_reachable_discount_by_turn"]["8"] >= 0
    for r in m["_results"]:
        if r["heliod_flip_turn"] is None:
            assert all(x is None for x in r["window_base_by_turn"]), "absent, never zero"
        else:
            assert all(x is None for x in r["window_base_by_turn"][:r["heliod_flip_turn"] - 1])


@requires_deck
def test_tithe_pays_treasure_off_the_window_s_forced_draws(monkeypatch):
    """PAIRED: the same games with Tithe always paid out against never paid out -- the
    Treasure is window mana, so the deck wins by Approach at least as often, and more."""
    from manamap.pilot import goldfish_turn
    monkeypatch.setattr(goldfish_turn, "TITHE_PAY_RATE", 0.0)
    always = goldfish.run("heliod", iterations=1500, seed=7, quiet=True, _band=False)["metrics"]
    monkeypatch.setattr(goldfish_turn, "TITHE_PAY_RATE", 1.01)
    never = goldfish.run("heliod", iterations=1500, seed=7, quiet=True, _band=False)["metrics"]
    assert always["approach"]["win_by_turn"]["10"] > never["approach"]["win_by_turn"]["10"]


@requires_deck
def test_a_fractional_fruition_rate_is_sampled_not_a_crash(monkeypatch):
    """OPPONENT_CASTS_PER_TURN is 1.264 (measured); a fractional Forced Fruition rate is one
    spell plus a second at the fraction, as Rhystic Study's is -- never a float card count."""
    monkeypatch.setattr(goldfish, "FRUITION_OPPONENT_SPELLS_PER_TURN", 1.264)
    m = goldfish.run("heliod", branch="b3-v1", iterations=600, seed=7, quiet=True, _band=False)
    assert m["metrics"]["approach"]["win_by_turn"]["10"] > 0


@requires_deck
def test_the_engine_s_output_is_measured_and_its_opponent_draw_payoffs_pay(monkeypatch):
    """The group-hug engine is reported per turn after the flip, and the window's forced draws
    pay "whenever an opponent draws" payoffs. PROVED BY REMOVING THEM: blank Consecrated
    Sphinx's draw on the sleeved list and the cards we draw on their turns fall."""
    from manamap.pilot import goldfish_library as gl
    on = goldfish.run("heliod", iterations=1500, seed=7, quiet=True, _band=False)["metrics"]
    h = on["heliod_window"]
    for key in ("spells", "mv", "saved", "drawn", "damage"):
        assert h[f"engine_{key}_by_turn"]["10"] is not None
    assert h["engine_saved_by_turn"]["10"] > 0 and h["engine_mv_by_turn"]["10"] > 0
    real = gl.event_payoffs
    monkeypatch.setattr(gl, "event_payoffs",
                        lambda card: {**real(card), "per_opponent_draw_our_draw": 0})
    off = goldfish.run("heliod", iterations=1500, seed=7, quiet=True, _band=False)["metrics"]
    assert off["heliod_window"]["engine_drawn_by_turn"]["10"] < h["engine_drawn_by_turn"]["10"]


@requires_deck
def test_life_kills_are_split_by_source_and_never_exceed_the_life_kills():
    doc = goldfish.run("heliod", branch="b3-v1", iterations=1500, seed=7, quiet=True,
                       _band=False, with_results=True)
    src = doc["metrics"]["heliod_window"]["life_kill_source_by_turn_10"]
    assert set(src) <= {"combat", "drain", "window"}
    life = sum(1 for r in doc["_results"] if r["kill_by"] == "life" and r["kill_turn"] <= 10)
    assert abs(sum(src.values()) - life / len(doc["_results"])) < 0.002


@requires_deck
def test_the_crawlers_drain_in_the_window_and_go_when_their_payoffs_go(monkeypatch):
    """PROVED BY REMOVING THEM: blank the per-draw damage (Psychosis Crawler) and the
    per-opponent-draw damage (Scrawling Crawler, now paid on Forced Fruition's sevens too) and
    the window's damage on b3-v1 falls."""
    from manamap.pilot import goldfish_library as gl
    on = goldfish.run("heliod", branch="b3-v1", iterations=1500, seed=7, quiet=True,
                      _band=False)["metrics"]["heliod_window"]["engine_damage_by_turn"]["10"]
    real = gl.event_payoffs
    monkeypatch.setattr(gl, "event_payoffs", lambda card: {
        **real(card), "per_draw_damage": 0, "per_opponent_draw_damage": 0})
    off = goldfish.run("heliod", branch="b3-v1", iterations=1500, seed=7, quiet=True,
                       _band=False)["metrics"]["heliod_window"]["engine_damage_by_turn"]["10"]
    assert on > off
