"""Approach of the Second Sun as a kill the goldfish plays (2026-11-23, Heliod b3-v1).

The pilot: "how fast do we play approach of the second sun?" -- unopposed. Under the deck's
`model_approach` declaration the turn loop casts it the moment the mana pays, puts the first
cast seventh from the top, wins on the second cast from hand, and a tutor takes it first.
"""

import json

from manamap.pilot import goldfish, model_coverage
from manamap.pilot.common import deck_file
from manamap.pilot.goldfish_profiles import approach_profile

from conftest import requires_deck

APPROACH = {"name": "Approach of the Second Sun", "type_line": "Sorcery", "oracle_text": (
    "If this spell was cast from your hand and you've cast another spell named Approach of "
    "the Second Sun this game, you win the game. Otherwise, put Approach into its owner's "
    "library seventh from the top and you gain 7 life.")}


def test_the_approach_shape_reads_from_the_text():
    assert approach_profile(APPROACH)


def test_another_alternate_win_is_not_the_approach_shape():
    """Laboratory Maniac wins too, but nothing goes seventh from the top: a different card."""
    maniac = {"name": "Laboratory Maniac", "type_line": "Creature — Human Wizard",
              "oracle_text": "If you would draw a card while your library has no cards in it, "
                             "you win the game instead."}
    assert not approach_profile(maniac)


def test_coverage_reads_it_on_its_channel_and_the_cast_loop_only_where_declared():
    from manamap.pilot import goldfish_library as gl
    prof = gl.classify(dict(APPROACH, cmc=7, mana_cost="{6}{W}", colors=["W"],
                            color_identity=["W"], keywords=[]))
    assert "approach" in model_coverage.channels_for(prof)
    assert model_coverage.CHANNELS["approach"] == "model_approach"
    assert model_coverage.never_cast(dict(prof), {"model_approach": True}) is False
    assert model_coverage.never_cast(dict(prof), {}) is True


@requires_deck
def test_heliod_wins_by_approach_only_with_the_channel_declared():
    """PROVED BY SWITCHING IT OFF: with the flag the deck wins some games by Approach and
    records the first cast; without it, none."""
    t = json.loads(deck_file("heliod", "goldfish_targets.json").read_text())
    on = goldfish.run("heliod", iterations=1500, seed=7, quiet=True, _band=False,
                      _targets_doc=dict(t, model_approach=True))["metrics"]
    off = goldfish.run("heliod", iterations=1500, seed=7, quiet=True, _band=False,
                       _targets_doc=dict(t, model_approach=False))["metrics"]
    assert (on.get("kill_by_approach_rate") or 0) > 0
    assert on["approach"]["first_cast_by_turn"]["10"] > on["approach"]["win_by_turn"]["10"]
    assert "approach" not in off and not off.get("kill_by_approach_rate")
