"""`scenario-ab`: a slice A/B from one spec, and the `--check` Sean OKs (PRD v2 Step 5)."""
import json

import pytest

from manamap.pilot import scenario_ab as sab
from manamap.sim import slice_state as ss

BOARD = {"turn": 6, "phase": "precombat main", "active_seat": "you",
         "seats": [{"seat": "you", "life": 40, "board": ["Brallin, Skyshark Rider", "Island"],
                    "hand": ["Lightning Greaves", "Windfall"]},
                   {"seat": "opp1", "life": 40, "board": ["Plains"],
                    "hand": {"known": ["Swords to Plowshares"], "unknown": 1}}]}
SPEC = {"seats": {"you": "shark", "opp1": "angels"}, "board": BOARD,
        "arms": {"greaves": [], "signet": [{"seat": "you", "zone": "hand",
                                            "out": "Lightning Greaves", "in": "Arcane Signet"}]},
        "primary": "commander_out", "seeds": 4, "question": "Greaves or Signet?"}


def write(tmp_path, spec):
    p = tmp_path / "scenario.json"
    p.write_text(json.dumps(spec))
    return p


def test_an_arm_is_edits_to_the_board_and_the_board_is_untouched(tmp_path):
    spec, board, arms, seeds = sab.load_spec(write(tmp_path, SPEC))
    assert arms["greaves"]["seats"][0]["hand"] == ["Lightning Greaves", "Windfall"]
    assert arms["signet"]["seats"][0]["hand"] == ["Windfall", "Arcane Signet"]
    assert board["seats"][0]["hand"] == ["Lightning Greaves", "Windfall"]
    assert seeds == 4 and board["version"] == 2


def test_an_edit_into_a_partly_known_hand_lands_in_its_known_cards():
    out = sab.apply_edits(BOARD, [{"seat": "opp1", "zone": "hand", "out": "Swords to Plowshares",
                                   "in": "Path to Exile"}])
    assert out["seats"][1]["hand"] == {"known": ["Path to Exile"], "unknown": 1}


@pytest.mark.parametrize("edit, why", [
    ({"seat": "you", "zone": "hand", "out": "Sol Ring"}, "not in you's hand"),
    ({"seat": "opp9", "zone": "hand", "in": "Sol Ring"}, "does not have"),
    ({"seat": "you", "zone": "library", "in": "Sol Ring"}, "is not one of"),
])
def test_an_edit_that_cannot_apply_is_a_refusal(edit, why):
    with pytest.raises(SystemExit, match=why):
        sab.apply_edits(BOARD, [edit])


def test_a_spec_needs_two_arms_a_primary_and_enough_seeds(tmp_path):
    for bad, why in ((dict(SPEC, arms={"a": []}), "exactly two arms"),
                     ({k: v for k, v in SPEC.items() if k != "primary"}, "needs `primary`"),
                     (dict(SPEC, seeds=1), "seeds must be")):
        with pytest.raises(SystemExit, match=why):
            sab.load_spec(write(tmp_path, bad))


def test_check_shows_each_arms_board_per_seat_and_plays_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ss, "_deck_copies", lambda slug: (
        (["Brallin, Skyshark Rider", "Lightning Greaves", "Arcane Signet", "Windfall"] + ["Island"] * 8,
         ["Brallin, Skyshark Rider"]) if slug == "shark" else
        (["Giada, Font of Hope", "Swords to Plowshares", "Path to Exile"] + ["Plains"] * 8,
         ["Giada, Font of Hope"])))
    from manamap.pilot import card_pool
    monkeypatch.setattr(card_pool, "corpus_names", lambda: {
        "Brallin, Skyshark Rider", "Lightning Greaves", "Arcane Signet", "Windfall", "Island",
        "Giada, Font of Hope", "Swords to Plowshares", "Path to Exile", "Plains"})
    from manamap.sim import slice_ab
    monkeypatch.setattr(slice_ab, "compare", lambda *a, **k: pytest.fail("--check must not play"))

    class Args:
        spec, check, as_json = str(write(tmp_path, SPEC)), True, False
    sab.main(Args)
    out = capsys.readouterr().out
    assert "QUESTION  Greaves or Signet?" in out and "primary commander_out" in out
    assert out.count("you (shark) life 40") == 2 and "Arcane Signet, Windfall" not in out
    assert "hand:        Windfall, Arcane Signet" in out
    assert "not played" in out
