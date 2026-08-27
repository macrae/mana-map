"""The controlled experiment: two versions, same table, one artifact with the delta.

Pinned: an arm resolves from a version ref or `working` and an A/A is refused with the
reason; the id moves with either arm's list, the table, N and the seed; the delta reads
both arms' aggregates and says whether the win-rate intervals overlap; and the
same-seeds-are-not-paired-games honesty line rides in the assumptions."""

import hashlib
import json
import subprocess

import pytest

from manamap.pilot import deck_history as dh
from manamap.sim import experiment as ex
from conftest import ROOT

SLUG = "xdeck"
V1 = "1 Radagast of Rhosgobel *CMDR*\n1 Craterhoof Behemoth\n30 Forest\n"
V2 = V1.replace("Craterhoof Behemoth", "Hornet Queen")


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(root), "PATH": "/usr/bin:/bin:/usr/local/bin"})


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path
    deck = root / "data" / "decks" / SLUG
    deck.mkdir(parents=True)
    opp = root / "data" / "opponents" / "rival"
    opp.mkdir(parents=True)
    opp.joinpath("decklist.txt").write_text("1 Edgar Markov *CMDR*\n1 Swamp\n")
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", root / "data" / "decks")
    monkeypatch.setattr("manamap.sim.forge.DECKS_DIR", root / "data" / "decks")
    monkeypatch.setattr(dh, "_REPO_ROOT", root)
    _git(root, "init", "-q")
    (deck / "decklist.txt").write_text(V1)
    _git(root, "add", "."); _git(root, "commit", "-q", "-m", "v1")
    (deck / "decklist.txt").write_text(V2)
    _git(root, "add", "."); _git(root, "commit", "-q", "-m", "v2: Hornet Queen for the Hoof")
    return deck


def test_an_arm_resolves_from_a_version_or_the_working_copy(repo):
    a = ex.resolve_arm(SLUG, "V1")
    assert "Craterhoof Behemoth" in a["decklist_text"] and a["label"].startswith("V1")
    (repo / "decklist.txt").write_text(V2 + "1 Sol Ring\n")
    w = ex.resolve_arm(SLUG, "working")
    assert "Sol Ring" in w["decklist_text"] and w["ref"] == "working"
    with pytest.raises(SystemExit):
        ex.resolve_arm(SLUG, "V9")


def test_an_a_a_is_refused_with_the_reason(repo, monkeypatch):
    with pytest.raises(SystemExit) as e:
        ex.run(SLUG, "V2", "working", ["rival"], games=2)
    assert "noise floor" in str(e.value), "V2 IS the working copy here — same list, same sha"


def test_the_id_moves_with_either_arm_the_table_n_and_seed(repo):
    a, b = ex.resolve_arm(SLUG, "V1"), ex.resolve_arm(SLUG, "V2")
    base = ex.experiment_id(SLUG, a, b, ["rival"], 20, 7)
    assert base.startswith("v1-vs-v2-x-rival-n20-") and base.endswith("-s7")
    assert ex.experiment_id(SLUG, a, b, ["rival"], 21, 7) != base
    assert ex.experiment_id(SLUG, a, b, ["rival"], 20, 8) != base
    (repo.parent.parent / "opponents" / "rival" / "decklist.txt").write_text("1 Edgar Markov *CMDR*\n2 Swamp\n")
    assert ex.experiment_id(SLUG, a, b, ["rival"], 20, 7) != base, "the table is part of the id"


def _analysis(win, n, dmg, share):
    lo_hi = [round(max(0, win - .2), 3), round(min(1, win + .2), 3)]
    return {"games": n, "seats": {SLUG: {
        "win_rate": win, "win_rate_ci95": lo_hi, "wins": int(win * n),
        "eliminated_turn": {"mean": 30.0, "n": n},
        "combat_damage_dealt_to_players": {"mean": dmg, "n": n},
        "combat_damage_taken": {"mean": 20.0, "n": n},
        "first_attack_turn": {"mean": 7.0, "n": n},
        "tokens": {"token_damage_share": {"mean": share, "n": n},
                   "tokens_observed": {"mean": 1.0, "n": n},
                   "token_resolutions": {"mean": 2.0, "n": n}}}}}


def test_the_delta_puts_an_interval_on_the_difference_not_on_each_arm():
    """The fix. `intervals_overlap` is GONE, not deprecated — it named the overlap
    fallacy, and a key left in the artifact re-invites the error it was removed
    for. Every figure now carries an interval on the difference, its N and the
    method that produced it, and no figure carries a bare `diff`."""
    d = ex.delta(_analysis(0.1, 20, 10.0, 0.0), _analysis(0.2, 20, 30.0, 0.19), SLUG)
    w = d["win_rate"]
    assert w["a"] == 0.1 and w["b"] == 0.2 and w["diff"] == 0.1
    assert "intervals_overlap" not in w
    assert w["ci95_diff"][0] < 0 < w["ci95_diff"][1], "0.1 vs 0.2 at n=20 cannot be called"
    assert w["excludes_zero"] is False
    assert "Newcombe" in w["method"]
    for name, _ in ex.DELTA_KEYS:
        assert "n_a" in d[name] and "method" in d[name], f"{name} travels without its N"


def test_a_real_win_rate_difference_is_called():
    d = ex.delta(_analysis(0.05, 200, 1, 0), _analysis(0.60, 200, 1, 0), SLUG)
    assert d["win_rate"]["excludes_zero"] is True
    assert "EXCLUDES zero" in d["reading"]


def test_the_reading_never_calls_an_uninformative_result_no_effect():
    """The old string said an overlap meant "the difference is noise". At n=20 the
    experiment cannot detect anything under about a 40-point swing, and saying so
    is a different claim from saying there is no difference."""
    d = ex.delta(_analysis(0.1, 20, 10.0, 0.0), _analysis(0.2, 20, 30.0, 0.19), SLUG)
    assert "CONTAINS zero" in d["reading"]
    assert "not evidence of no effect" in d["reading"]
    assert "noise" not in d["reading"]


def test_the_power_block_says_what_could_have_been_detected():
    d = ex.delta(_analysis(0.0, 12, 1, 0), _analysis(0.0, 12, 1, 0), SLUG)
    p = d["power"]
    assert p["primary_endpoint"] == "win_rate" and p["n_per_arm"] == 12
    assert p["minimum_detectable_rate_b"] == 0.415, (
        "at twelve games an arm, arm B must win five of twelve to be called")
    assert p["games_per_arm_to_detect_0.10"] is not None


def test_only_the_win_rate_is_pre_registered():
    """Eleven figures at alpha=0.05 means roughly one interval in two experiments
    excludes zero by chance. The other ten are descriptive and must not carry a
    verdict of their own."""
    d = ex.delta(_analysis(0.1, 20, 10.0, 0.0), _analysis(0.2, 20, 30.0, 0.19), SLUG)
    assert d["power"]["primary_endpoint"] == ex.PRIMARY_ENDPOINT
    assert ex.PRIMARY_ENDPOINT == "win_rate"


def test_a_figure_with_no_per_game_values_says_so_rather_than_implying_precision():
    """An older artifact whose logs are gone re-derives to a bare difference. An
    absent interval must never read as a narrow one."""
    d = ex.delta(_analysis(0.1, 20, 10.0, 0.0), _analysis(0.2, 20, 30.0, 0.19), SLUG)
    m = d["combat_damage_dealt_to_players"]
    assert m["diff"] == 20.0
    assert m.get("ci95_diff") is None
    assert "no per-game values" in m["method"]


def test_the_tracked_artifact_carries_the_honesty_lines():
    doc = json.load(open("data/decks/radagast/experiments/v1-vs-v5-x-giada-angels-vito-n10-a50f20db-s577662891.json")) \
        if (ROOT / "data/decks/radagast/experiments").is_dir() else None
    if doc is None:
        pytest.skip("no tracked experiment (fresh clone)")
    assert any("NOT PAIRED" in a.upper() for a in doc["assumptions"])
    assert any("SEEDED" in a for a in doc["assumptions"])
    assert doc["arms"]["a"]["decklist_text"] and doc["arms"]["b"]["decklist_text"], \
        "the arms' lists ride in the artifact so the gitignored logs are exactly regenerable"
    w = doc["delta"]["win_rate"]
    assert "intervals_overlap" not in w, "the overlap fallacy must not survive in the artifact"
    assert w["ci95_diff"] and "Newcombe" in w["method"]
    assert doc["delta"]["power"]["minimum_detectable_difference"] is not None, (
        "an experiment that reports no difference must say what it could have found")


def test_the_delta_carries_commander_damage_per_defender():
    """The A/B is the tool for judging a change to a commander-damage deck, and it was
    the one path in the sim stack blind to commander damage — `experiment.py` called
    `analyze_logs` without commander names while `run`, `analyze` and `validate_sim`
    all passed them. Measured on kianne: the win rate moved 0.0 -> 0.083 (noise at
    n=12) while games reaching 21 on one defender moved 0 -> 2, which is the figure
    the change was actually aimed at.

    `max_on_one_defender` and `dealt_total` are BOTH reported because they answer
    different questions: 60 damage spread across three seats wins nothing.
    """
    keys = dict(ex.DELTA_KEYS)
    assert keys["commander_damage_max_on_one_defender"] == \
        ("commander_damage", "max_on_one_defender", "mean")
    assert keys["commander_damage_dealt_total"] == ("commander_damage", "dealt_total", "mean")
    assert keys["commander_damage_games_reaching_21"] == ("commander_damage", "games_reaching_21")

    def seat(maxd, total, reaching):
        a = _analysis(0.1, 12, 10.0, 0.0)
        a["seats"][SLUG]["commander_damage"] = {
            "commander": ["Kianne, Corrupted Memory"],
            "dealt_total": {"mean": total, "n": 12},
            "max_on_one_defender": {"mean": maxd, "n": 12},
            "games_reaching_21": reaching}
        return a

    d = ex.delta(seat(2.25, 2.5, 0), seat(17.42, 21.92, 2), SLUG)
    assert d["commander_damage_max_on_one_defender"]["diff"] == 15.17
    assert d["commander_damage_games_reaching_21"]["a"] == 0
    assert d["commander_damage_games_reaching_21"]["b"] == 2


def test_an_arm_with_no_commander_damage_block_reports_none_not_zero():
    """An arm whose commander could not be identified must not read as 'dealt 0' —
    the same absent-rather-than-zeroed contract the parser holds."""
    d = ex.delta(_analysis(0.1, 12, 10.0, 0.0), _analysis(0.2, 12, 30.0, 0.1), SLUG)
    cd = d["commander_damage_max_on_one_defender"]
    assert cd["a"] is None and cd["b"] is None and cd["diff"] is None
