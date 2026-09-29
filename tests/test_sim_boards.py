"""The board finder and the lift-freshness gate.

Pinned: each criterion names a cut and a SHAPE, recurrence counts games (with
a Wilson interval), the exemplar lifts through the bridge with the finder's
provenance stamped, and a committed lifted scenario that no longer matches a
fresh lift of its own cut FAILS while unresolved and NOTES once checker-passed.
"""

import json
import pathlib

import pytest

from manamap.sim import boards
from manamap.sim import bridge
from manamap.sim import forge
from manamap.pilot import validate_lift

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "forge" / "two-seat-one-game.log"
SLUG = "radagast"        # the fixture's seat 0 is Ai(1)-radagast


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A deck directory holding one run whose single log is the fixture."""
    base = tmp_path / "decks" / SLUG
    sim = base / "sim" / "logs" / "fixture-run"
    sim.mkdir(parents=True)
    (base / "decklist.txt").write_text("1 Radagast of Rhosgobel *CMDR*\n1 Forest\n")
    (base / "stacks").mkdir()
    text = FIXTURE.read_text(encoding="utf-8", errors="replace")
    (sim / "part-00.log").write_text(text)
    opp = tmp_path / "decks" / "edgar-vampires"
    opp.mkdir()
    (opp / "decklist.txt").write_text("1 Edgar Markov *CMDR*\n1 Swamp\n")
    outcome = forge.parse_outcomes(text)[0]
    rec = {"run_id": "fixture-run", "slug": SLUG, "at": "2026-09-30",
           "seats": [{"slug": SLUG, "forge_name": "radagast", "decklist_sha256": "a" * 64,
                      "commander": ["Radagast of Rhosgobel"]},
                     {"slug": "edgar-vampires", "forge_name": "edgar-vampires",
                      "decklist_sha256": "b" * 64, "commander": ["Edgar Markov"]}],
           "games_completed": 1, "games_requested": 1,
           "outcomes": [{"winner": outcome["winner"], "won_by": outcome["won_by"], "draw": False,
                         "round": outcome["round"], "global_turn": outcome["global_turn"],
                         "truncated": False, "log": "part-00.log", "seed": 7, "game_in_job": 1,
                         "seat_order": [SLUG, "edgar-vampires"]}]}
    (base / "sim" / "fixture-run.json").write_text(json.dumps(rec))
    monkeypatch.setattr("manamap.config.DECKS_DIR", tmp_path / "decks")
    monkeypatch.setattr(forge, "_out_dir", lambda slug: base / "sim")
    monkeypatch.setattr(boards, "_out_dir", lambda slug: base / "sim")
    monkeypatch.setattr(bridge, "seat_dir", lambda slug: tmp_path / "decks" / slug)
    monkeypatch.setattr(bridge, "deck_dir", lambda slug, branch=None: base)
    monkeypatch.setattr(validate_lift, "deck_dir", lambda slug, branch=None: base)
    return base


def test_buckets_make_one_shape_of_neighbouring_boards():
    assert boards.bodies_bucket(0) == "0-2" and boards.bodies_bucket(5) == "3-5"
    assert boards.bodies_bucket(8) == "6-8" and boards.bodies_bucket(30) == "9+"
    assert boards.turn_bucket(8) == "early (≤8)" and boards.turn_bucket(17) == "late (17+)"


def test_every_criterion_names_a_cut_and_a_shape_and_recurrence_counts_games(run):
    seen = 0
    for crit in boards.CRITERIA:
        found = boards.find(SLUG, "fixture-run", crit, {"n": 2})
        assert found["games"] == 1 and found["definition"]
        for row in found["shapes"]:
            seen += 1
            ex = row["exemplar"]
            assert ex["turn"] >= 1 and ex["phase"] and "shape" in row
            assert row["games"] == 1 and row["of"] == 1 and row["ci95"][0] is not None
            assert ex["replay"] == "-n 1 -s 7"
    assert seen >= 3, "the fixture game should hit widest, modal and first-attack at least"
    with pytest.raises(SystemExit):
        boards.find(SLUG, "fixture-run", "nope")


def test_widest_is_the_own_turn_with_the_most_bodies(run):
    found = boards.find(SLUG, "fixture-run", "widest")
    assert len(found["shapes"]) == 1
    ex = found["shapes"][0]["exemplar"]
    # the exemplar's bodies equal the max bodies over every own precombat main
    game = boards._games_of(SLUG, "fixture-run")[1][0][2]
    ours = next(s for s in game["seats"] if s.endswith("-radagast"))
    best = 0
    for g in boards._own_turns(game, ours):
        states, _ = boards._cut_state(game, g, "precombat main", None, {})
        best = max(best, boards.bs.snapshot(states[ours])["bodies"])
    assert ex["detail"]["bodies"] == best and best > 0


def test_lifting_the_shortlist_stamps_the_finders_provenance_and_the_bridge_version(run):
    found = boards.find(SLUG, "fixture-run", "first-attack")
    (path, doc), = boards.lift_shortlist(SLUG, "fixture-run", found, top=1, to_stack=True)
    assert path.name.startswith("001-sim-g1-t")
    src = doc["scenario"]["source"]
    f = doc["scenario"]["extras"]["finder"]
    assert f["criterion"] == "first-attack" and f["recurrence"] == {"games": 1, "of": 1, "ci95": found["shapes"][0]["ci95"]}
    assert src["lift_sha"] == validate_lift.lift_sha(doc) and src["bridge_sha"] == validate_lift.bridge_sha()


def test_a_committed_lift_that_no_longer_matches_fails_while_unresolved_and_notes_once_passed(run):
    """THE 010-VS-012 CASE: delete one opponent creature from the committed
    copy and the gate names the card and the seat. Re-introduce by comparing
    nothing and it says OK."""
    found = boards.find(SLUG, "fixture-run", "widest")
    (path, doc), = boards.lift_shortlist(SLUG, "fixture-run", found, top=1, to_stack=True)
    errs, notes = validate_lift.check(doc)
    assert errs == [] and notes == [], "a fresh lift equals itself"
    other = next(s for s in doc["scenario"]["seats"] if s["seat"] != "you")
    victim = next((b for b in other["board"] if b.get("pt")), None) or other["board"][0]
    other["board"].remove(victim)
    errs, notes = validate_lift.check(doc)
    assert errs and victim["name"] in errs[0] and other["seat"] in errs[0] and "MISSING" in errs[0]
    doc["checker"] = {"verdict": "pass"}
    errs, notes = validate_lift.check(doc)
    assert errs == [] and notes and "checker pass" in notes[0] and victim["name"] in notes[0]
    doc["checker"] = {"verdict": "fail"}
    errs, notes = validate_lift.check(doc)
    assert errs == [] and "checker fail" in notes[0], "a loop that gave up is not re-litigated"
    # no logs here: only the bridge version can be checked
    doc["scenario"]["source"]["log"] = "gone.log"
    doc["scenario"]["source"]["bridge_sha"] = "000000000000"
    errs, notes = validate_lift.check(doc)
    assert errs == [] and notes and "bridge" in notes[0]


def test_the_module_never_writes_a_state_the_finder_stores():
    """Recurrence is DERIVED from the logs each time; the finder stores nothing."""
    import inspect
    src = inspect.getsource(boards)
    assert "write_text" not in src.replace("lift_shortlist", ""), "the finder writes only through bridge.lift"
