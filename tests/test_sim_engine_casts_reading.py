"""The print-time reading over `engine_casts`: was the deck's engine ever cast?

Mirrors `test_sim_pilot_quality`: read from the record, never the logs; the
verdict is withheld under MIN_GAMES; an old record reads None (not measured);
the engine set comes from an authored file and is applied at print time only.
"""

import copy
import inspect

import pytest

from manamap.sim import engine_casts as ec
from conftest import requires_deck

REC = {
    "seats": [{"slug": "sharky", "commander": ["Shabraz, the Skyshark", "Brallin, Skyshark Rider"]}],
    "engine_casts": {"seat": "sharky", "games": 60, "turns": 576, "kept_hand_mean": 6.97,
                     "by_card": {"Commit": {"cast": 12, "activated": 0, "discarded": 0},
                                 "Commit // Memory": {"cast": 0, "activated": 0, "discarded": 4},
                                 "Windfall": {"cast": 0, "activated": 0, "discarded": 3},
                                 "Wheel of Fortune": {"cast": 1, "activated": 0, "discarded": 1},
                                 "Lonely Sandbar": {"cast": 0, "activated": 8, "discarded": 11},
                                 "Brallin, Skyshark Rider": {"cast": 70, "activated": 3, "discarded": 0}}},
}
NAMES = {"Commit // Memory", "Windfall", "Wheel of Fortune", "Brallin, Skyshark Rider",
         "Magus of the Wheel", "Lonely Sandbar"}


def test_it_reads_the_record_not_the_logs():
    src = inspect.getsource(ec)
    assert ".log" not in src and "logs" not in src.replace("catalog", "")


def test_an_old_record_is_not_measured():
    assert ec.from_record({"seats": [], "games": [{}]}) is None


def test_faces_merge_and_lands_are_excluded_and_the_line_is_per_card():
    q = ec.from_record(REC, NAMES)
    assert q["by_card"]["Commit // Memory"]["cast"] == 12, "the cast face and the discard name are one card"
    assert q["by_card"]["Commit // Memory"]["discarded"] == 4
    never = [r["card"] for r in q["never_cast"]]
    assert "Windfall" in never, "discarded three times, never played: held and passed over"
    assert "Magus of the Wheel" in never, "never seen in the log, expected ~10 natural draws"
    assert "Wheel of Fortune" not in never, "one cast is a cast"
    assert "Commit // Memory" not in never
    assert q["expected_natural_draws_per_card"] == pytest.approx(60 * (6.97 + 9.6) / 98, abs=0.1), \
        "a partner deck has a 98-card library"
    assert q["engine"] is None and q["covered"] is None
    assert "no engine declaration" in q["reading"]


def test_the_verdict_needs_games_and_a_declaration():
    few = copy.deepcopy(REC); few["engine_casts"]["games"] = 5
    q = ec.from_record(few, NAMES)
    assert q["covered"] is None and "too few" in q["reading"]
    eng = {"cards": ["Windfall", "Wheel of Fortune"], "scaffolded": False, "sources": ["engine.json"]}
    q = ec.from_record(REC, NAMES, eng)
    assert q["covered"] is False and q["engine"]["never_cast"] == ["Windfall"]
    assert "NEVER CAST" in q["reading"]
    # Proven by re-introducing the play: cast Windfall once and the verdict flips.
    played = copy.deepcopy(REC); played["engine_casts"]["by_card"]["Windfall"]["cast"] = 1
    assert ec.from_record(played, NAMES, eng)["covered"] is True
    # A scaffolded declaration gives no set-level share, and says why.
    q = ec.from_record(REC, NAMES, {**eng, "scaffolded": True})
    assert q["engine"]["played_share"] is None and "scaffold" in q["engine"]["why"]


def test_render_names_the_verdict():
    lines = ec.render(ec.from_record(REC, NAMES, {"cards": ["Windfall"], "scaffolded": False, "sources": ["x"]}))
    assert any("ENGINE NEVER CAST" in l for l in lines)
    assert any("Windfall (discarded x3)" in l for l in lines)


@requires_deck
def test_the_day_it_was_built_for():
    """sharknado@recon-v1, 60 games at standard-v3: the wheels were never cast,
    and the record says so without a hand-parsed count."""
    import glob, json
    from manamap.pilot.common import load_deck_cards
    paths = sorted(glob.glob("data/decks/sharknado/branches/recon-v1/sim/*n60*.json"))
    if not paths:
        pytest.skip("the record is not on this checkout")
    rec = json.load(open(paths[-1]))
    q = ec.from_record(rec, ec.nonland_names(load_deck_cards("sharknado", "recon-v1")),
                       ec.engine_set("sharknado", "recon-v1"))
    never = [r["card"] for r in q["never_cast"]]
    assert "Windfall" in never and "Magus of the Wheel" in never
    assert q["by_card"]["Wheel of Fortune"]["cast"] == 1
    assert q["covered"] is False
    assert not any(n in never for n in ("Swamp", "Island", "Mystic Monastery")), "lands are played, not cast"
