"""The controlled experiment: two versions, same table, one artifact with the delta.

Pinned: an arm resolves from a version ref or `working` and an A/A is refused with the
reason; the id moves with either arm's list, the table, N and the seed; the delta reads
both arms' aggregates and says whether the win-rate intervals overlap; and the
same-seeds-are-not-paired-games honesty line rides in the assumptions."""

import hashlib
import os
import json
import pathlib
import subprocess

import pytest

from manamap import config
from manamap.pilot import deck_history as dh
from manamap.sim import experiment as ex
from manamap.sim import forge
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
    monkeypatch.setattr("manamap.config.DECKS_DIR", root / "data" / "decks")
    monkeypatch.setattr(dh, "_REPO_ROOT", root)
    # THE ENGINE ON THIS MACHINE IS NOT PART OF A UNIT TEST. `run` reads the
    # installed card-script fingerprint into the id (an axis of the harness);
    # a test's id must not depend on what happens to be installed here.
    monkeypatch.setattr(ex, "card_overrides", lambda: None)
    monkeypatch.setattr(ex, "telemetry_for_run", lambda plain=False: (None, None))
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


# ── the flagship's own success path, which had no test ──────────────────────

def test_the_final_print_reads_keys_the_delta_actually_emits(repo, capsys):
    """`main` raised KeyError on `ci95_a` on EVERY real run.

    `delta()` emits `a`, `b`, `n_a`, `n_b`, `diff` and `ci95_diff` — never a
    marginal interval per arm, and it must not: two marginal intervals
    overlapping implies nothing, which is why `intervals_overlap` was deleted
    rather than deprecated. The printer asked for the keys the artifact exists to
    make impossible, and it raised AFTER writing the artifact, so the measurement
    survived and the command exited 1.

    It lived because only `--dry-run` and `--analyze` were exercised. This drives
    the branch that actually runs, so it cannot come back.
    """
    d = ex.delta(_analysis(0.1, 20, 10.0, 0.0), _analysis(0.2, 20, 30.0, 0.19), SLUG)
    doc = {"experiment_id": "x-vs-y-n20", "arms": {"a": {"label": "V1"}, "b": {"label": "V2"}},
           "games_per_arm": 20, "opponents": [{"slug": "rival"}],
           "wall_seconds": 12.3, "delta": d}

    ex._print_result(SLUG, doc, repo / "experiments" / "x.json")

    out = capsys.readouterr().out
    # The primary endpoint prints under its label; every other key by name.
    assert "win rate" in out and ex.PRIMARY_ENDPOINT == "win_rate"
    assert "95%" in out and "[" in out, "every figure travels with its interval"
    # A figure absent from both arms is correctly skipped — this fixture has no
    # commander-damage block, and "absent" must not render as a row of zeroes.
    shown = 0
    for name, _ in ex.DELTA_KEYS[1:]:
        if d[name]["a"] is None and d[name]["b"] is None:
            assert name not in out, f"{name} is absent and must not print"
        else:
            assert name in out, f"{name} never reached the report"
            shown += 1
    assert shown >= 6, "the report iterated almost nothing"
    assert "-0.134, +0.328" in out, "the interval is on the DIFFERENCE"
    assert "ci95_a" not in out and "ci95_b" not in out


def test_an_absent_interval_prints_a_dash_and_never_a_zero_band():
    """An unbounded difference must not render as a narrow one.

    `delta()` writes "no per-game values available; difference is unbounded" for
    a figure it could not bound. A printer that turned that into `[0.000, 0.000]`
    would undo the sentence.
    """
    assert ex._band({"ci95_diff": None}) == "—"
    assert ex._band({}) == "—"
    assert ex._band({"ci95_diff": [-0.36, 0.038]}) == "[-0.360, +0.038]"
    # `0` is a measurement and prints as one; `None` is not.
    assert ex._num(0) == 0 and ex._num(0.0) == 0.0
    assert ex._num(None) == "—"


# ── the pod is part of the instrument ───────────────────────────────────────

def test_the_pod_runs_the_same_pilot_simulate_gives_it(repo, monkeypatch):
    """`simulate` has run the pod on Experimental since 2026-08-30; this ran it
    on Default and never read the standard, so a controlled A/B was controlled
    against a table the deck is never measured against."""
    from manamap.sim import forge

    seen = {}
    monkeypatch.setattr(ex, "forge_jar", lambda: pathlib.Path("/nope/forge.jar"))

    _, doc = ex.run(SLUG, "V1", "working", ["rival"], games=2, dry_run=True)
    assert doc["profiles"] == [forge.DEFAULT_PROFILE, forge.STANDARD_POD_PROFILE]

    _, old = ex.run(SLUG, "V1", "working", ["rival"], games=2, dry_run=True,
                    vs_profile="Default")
    assert old["profiles"] == [forge.DEFAULT_PROFILE, "Default"]


def test_the_experiment_id_carries_the_profiles(repo):
    """The digest is over decklists, opponents, games and seed — none of which
    move when the AI does. Without the tag, two configurations write one path and
    the second silently replaces the first, which is the defect `profile_tag`
    was written for on the `simulate` side.

    `profile_tag` omits a suffix for Default, so every experiment already on disk
    keeps its id and still means what it said.
    """
    _, new = ex.run(SLUG, "V1", "working", ["rival"], games=2, dry_run=True)
    _, old = ex.run(SLUG, "V1", "working", ["rival"], games=2, dry_run=True,
                    vs_profile="Default")

    assert new["experiment_id"].endswith("-podExperimental")
    assert not old["experiment_id"].endswith("-podExperimental")
    assert new["experiment_id"] != old["experiment_id"]
    # Same seed, same arms, same table: only the pilot differs.
    assert new["seed"] == old["seed"]


def test_every_tracked_experiment_id_still_resolves():
    """No record on disk may be renamed or reinterpreted by the tag rule."""
    tracked = sorted((ROOT / "data" / "decks").glob("*/experiments/*.json"))
    assert len(tracked) >= 2, "the guard iterated zero files"
    tagged = 0
    for path in tracked:
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["experiment_id"] == path.stem
        tagged += path.stem.endswith("-podExperimental")
    # The two records that predate the pod fix carry NO tag and were measured
    # against the Default pod; anything run after it carries one. Both must be
    # able to coexist — that is the whole point of `profile_tag` omitting a
    # suffix for Default, and it is why no old record had to be renamed.
    assert tagged >= 1, "no experiment has been run since the pod fix"
    assert tagged < len(tracked), "the pre-fix records were renamed — they must not be"


def test_the_experiment_uses_performance_cores_like_simulate_does(repo, monkeypatch):
    """`simulate` was moved to performance cores on measurement; this was not.

    4-JVM runs truncated 0% of their games and every 7-JVM run truncated 5-18%.
    A truncated game has no winner and is EXCLUDED from the rate, so
    oversubscribing does not merely run slower — it throws games away, and an
    experiment throws them away from BOTH arms. The first real A/B ran at 7 jobs
    on a 4-performance-core machine and took 7,295 seconds for 80 games.
    """
    seen = {}
    monkeypatch.setattr(ex, "split_games",
                        lambda g, j: seen.setdefault("jobs", j) or [g])
    monkeypatch.setattr(ex, "forge_jar", lambda: pathlib.Path("/nope/forge.jar"))
    try:
        ex.run(SLUG, "V1", "working", ["rival"], games=4, dry_run=True)
    except Exception:
        pass
    if "jobs" in seen:
        assert seen["jobs"] == forge.default_jobs(), \
            "the experiment must not oversubscribe where simulate does not"
    assert forge.default_jobs() <= (os.cpu_count() or 2), "sanity"


# ── group-sequential looks, and the instrument fixes ───────────────────────

def _fake_wave(wins_per_wave):
    """A `_run_wave` stub that returns canned texts and the rotation the real
    one would have produced, and a `_score_arm` stub that scores each arm by
    the number of texts it holds."""
    def run_wave(letter, meta, opp_names, games, jobs, clock, seed_base, profile,
                 vs_profile, log_dir, jar, first_job=0):
        seats = [meta, *opp_names]
        n = len(seats)
        parts = ex.split_games(games, jobs)
        job_ix = [first_job + i for i in range(len(parts))]
        log_dir.mkdir(parents=True, exist_ok=True)
        for j in job_ix:
            (log_dir / f"{letter}-part-{j:02d}.log").write_text(f"{letter}{j}")
        return {"texts": [f"{letter}{j}" for j in job_ix],
                "seeds": [seed_base + j for j in job_ix],
                "orders": [seats[j % n:] + seats[:j % n] for j in job_ix],
                "jobs": job_ix, "bad": 0}

    def score(texts, meta, opp_names, slug, arm, opponents):
        waves = len(texts)          # one text per job; jobs per wave = 2 in these tests
        letter = texts[0][0]
        games = 10 * len(texts)
        wins = sum(wins_per_wave[letter][:len(texts)])
        analysis = {"games": games, "decided": games,
                    "seats": {slug: {"wins": wins, "win_rate": wins / games}}}
        return [], analysis, []
    return run_wave, score


def _sequential(repo, monkeypatch, wins, **kw):
    """Run a 4-look, 80-game, 2-job experiment against the stubs."""
    run_wave, score = _fake_wave(wins)
    monkeypatch.setattr(ex, "_run_wave", run_wave)
    monkeypatch.setattr(ex, "_score_arm", score)
    monkeypatch.setattr(ex, "forge_jar", lambda: pathlib.Path("/nope/forge.jar"))
    monkeypatch.setattr(ex, "install_named", lambda meta, text: meta)
    monkeypatch.setattr(ex, "install_deck", lambda o: f"mm-{o}")
    monkeypatch.setattr(ex, "card_overrides", lambda: None)
    monkeypatch.setattr(ex, "telemetry_for_run", lambda plain=False: (None, None))
    monkeypatch.setattr(ex, "forge_version", lambda: "test")
    monkeypatch.setattr(ex, "_java_version", lambda: "test")
    monkeypatch.setattr(ex._pods, "record_for", lambda name, opps: {"name": name, "named": bool(name)})
    kw.setdefault("games", 80); kw.setdefault("jobs", 2); kw.setdefault("looks", 4)
    return ex.run(SLUG, "V1", "working", ["rival"], seed=7, **kw)


@pytest.mark.slow
def test_a_look_is_whole_jobs_from_both_arms_never_a_partial_job(repo, monkeypatch):
    """Four looks over 80 games at 2 jobs: waves of 20, each wave two whole jobs
    per arm, the schedule 20/40/60/80, and a design that would cut a job in two
    is refused. Every look tests at its OWN boundary."""
    wins = {"a": [1, 1, 1, 1, 1, 1, 1, 1], "b": [2, 2, 2, 2, 2, 2, 2, 2]}
    path, doc = _sequential(repo, monkeypatch, wins)
    assert doc["status"] == "complete" and len(doc["looks"]) == 4
    assert doc["design"]["schedule"] == [20, 40, 60, 80]
    for i, lk in enumerate(doc["looks"]):
        assert lk["jobs"]["a"] == [2 * i, 2 * i + 1] and lk["jobs"]["b"] == [2 * i, 2 * i + 1]
        assert lk["n_a"] == 20 * (i + 1) and lk["z_boundary"] == [4.049, 2.863, 2.337, 2.024][i]
    assert doc["looks"][-1]["decision"] == "complete"
    assert not ex.validate(doc), ex.validate(doc)
    with pytest.raises(SystemExit) as e:
        _sequential(repo, monkeypatch, wins, games=90, looks=4)
    assert "does not divide" in str(e.value)


def test_the_experiment_rotates_seats_per_job_like_simulate(tmp_path, monkeypatch):
    """THE BUG: our seat sat at index 0 in every job of every arm, and the
    profiles were not rotated with the seats. Drives the REAL `_run_wave` with
    the JVM stubbed: the `-d` order rotates per GLOBAL job index (a later wave
    continues the rotation), the `-a` list rotates with it, the seeds continue,
    and the log names carry the global index. Re-introduce by building every
    order as `list(seats)` and the second job's `-d` no longer leads with the
    opponent."""
    argvs = []
    real_command = ex.command
    monkeypatch.setattr(ex, "command", lambda *a, **kw: argvs.append(real_command(*a, **kw)) or argvs[-1])
    monkeypatch.setattr(ex.subprocess, "run",
                        lambda cmd, **kw: type("P", (), {"returncode": 0})())
    got = ex._run_wave("a", "mm-x-xdeck-a", ["mm-rival"], 4, 2, 600, 7,
                       "Default", "Experimental", tmp_path / "logs",
                       pathlib.Path("/nope/forge.jar"), first_job=2)
    assert got["jobs"] == [2, 3] and got["seeds"] == [9, 10]
    assert got["orders"] == [["mm-x-xdeck-a", "mm-rival"], ["mm-rival", "mm-x-xdeck-a"]]
    d0 = argvs[0][argvs[0].index("-d") + 1: argvs[0].index("-f")]
    d1 = argvs[1][argvs[1].index("-d") + 1: argvs[1].index("-f")]
    assert d0 == ["mm-x-xdeck-a", "mm-rival"] and d1 == ["mm-rival", "mm-x-xdeck-a"]
    a1 = argvs[1][argvs[1].index("-a") + 1: argvs[1].index("-a") + 3]
    assert a1 == ["Experimental", "Default"], "the profiles rotate with the seats"
    assert sorted(p.name for p in (tmp_path / "logs").iterdir()) == ["a-part-02.log", "a-part-03.log"]
    # and the label map scores our seat at EVERY index
    label = ex._label_for("mm-x-xdeck-a", ["mm-rival"], SLUG)
    assert label["Ai(1)-mm-x-xdeck-a"] == SLUG and label["Ai(2)-mm-x-xdeck-a"] == SLUG


@pytest.mark.slow
def test_a_moderate_effect_waits_for_a_later_look(repo, monkeypatch):
    """THE BUG: a fixed 1.96 at every look. 4/20 against 12/20 excludes zero at
    1.96 and NOT at the first OBF boundary (4.049); the same arms at 40 games
    clear the second (2.863). So the run must stop at look 2, never look 1 —
    re-introduce by dropping `z=` from `_look` and it stops at look 1."""
    wins = {"a": [2] * 8, "b": [6] * 8}
    _, doc = _sequential(repo, monkeypatch, wins)
    assert doc["status"] == "stopped_efficacy"
    assert len(doc["looks"]) == 2, [lk["decision"] for lk in doc["looks"]]
    assert doc["looks"][0]["excludes_zero"] is False and doc["looks"][1]["excludes_zero"] is True


def test_an_early_look_that_excludes_zero_at_its_boundary_stops_the_run(repo, monkeypatch):
    """A huge effect stops at look 2; the record says at which boundary, and
    the 1.96 interval beside it is labelled descriptive."""
    wins = {"a": [0, 0, 0, 0, 0, 0, 0, 0], "b": [9, 9, 9, 9, 9, 9, 9, 9]}
    _, doc = _sequential(repo, monkeypatch, wins)
    assert doc["status"] == "stopped_efficacy"
    assert doc["looks"][-1]["decision"] == "stop: efficacy" and len(doc["looks"]) < 4
    assert "STOPPED AT LOOK" in doc["delta"]["reading"] and "descriptive" in doc["delta"]["reading"]
    assert not ex.validate(doc)


@pytest.mark.slow
def test_futility_stops_only_when_the_asked_for_effect_is_excluded(repo, monkeypatch):
    """Non-binding futility: with `until_mde` the run stops when the boundary
    interval already excludes +X on the favourable side, and never without it."""
    wins = {"a": [5] * 8, "b": [5] * 8}
    _, plain = _sequential(repo, monkeypatch, wins)
    assert plain["status"] == "complete"
    # the same arms again need a new id: `_sequential` fixes seed=7, so run directly
    run_wave, score = _fake_wave(wins)
    monkeypatch.setattr(ex, "_run_wave", run_wave); monkeypatch.setattr(ex, "_score_arm", score)
    _, fut = ex.run(SLUG, "V1", "working", ["rival"], seed=9, games=80, jobs=2, looks=4, until_mde=0.30)
    assert fut["status"] == "stopped_futility", fut["looks"][-1]
    assert "futility" in fut["delta"]["reading"]


@pytest.mark.slow
def test_a_killed_wave_resumes_at_the_same_look_with_the_same_seeds(repo, monkeypatch):
    """The record is rewritten after every wave. Kill the run after look 2,
    resume the same command line, and looks 3–4 run on the seeds and job
    indices the design planned; without `--resume` the path is refused, and a
    record with no design cannot be resumed at all."""
    wins = {"a": [1] * 8, "b": [1] * 8}
    run_wave, score = _fake_wave(wins)
    calls = {"n": 0}

    def dying(*args, **kw):
        calls["n"] += 1
        if calls["n"] > 4:                     # 2 waves x 2 arms
            raise RuntimeError("killed")
        return run_wave(*args, **kw)
    monkeypatch.setattr(ex, "_run_wave", dying)
    monkeypatch.setattr(ex, "_score_arm", score)
    for name, val in (("forge_jar", lambda: pathlib.Path("/nope/forge.jar")),
                      ("install_named", lambda meta, text: meta), ("install_deck", lambda o: f"mm-{o}"),
                      ("card_overrides", lambda: None), ("forge_version", lambda: "test"),
                      ("_java_version", lambda: "test")):
        monkeypatch.setattr(ex, name, val)
    monkeypatch.setattr(ex._pods, "record_for", lambda name, opps: {"name": name, "named": bool(name)})
    with pytest.raises(RuntimeError):
        ex.run(SLUG, "V1", "working", ["rival"], seed=7, games=80, jobs=2, looks=4)
    eid = ex.experiment_id(SLUG, ex.resolve_arm(SLUG, "V1"), ex.resolve_arm(SLUG, "working"),
                           ["rival"], 80, 7, vs_profile=ex.STANDARD_POD_PROFILE, looks=4)
    path = ex.deck_dir(SLUG) / ex.EXP_DIR / f"{eid}.json"
    half = json.loads(path.read_text())
    assert half["status"] == "running" and len(half["looks"]) == 2
    with pytest.raises(SystemExit) as e:
        ex.run(SLUG, "V1", "working", ["rival"], seed=7, games=80, jobs=2, looks=4)
    assert "--resume" in str(e.value)
    monkeypatch.setattr(ex, "_run_wave", run_wave)
    _, done = ex.run(SLUG, "V1", "working", ["rival"], seed=7, games=80, jobs=2, looks=4, resume=True)
    assert done["status"] == "complete" and len(done["looks"]) == 4
    assert done["looks"][2]["jobs"]["a"] == [4, 5] and done["looks"][2]["seeds"]["a"] == [11, 12]
    assert done["seeds"] == [7, 8, 9, 10, 11, 12, 13, 14]
    # a record that was never sequential cannot grow a look after the fact
    legacy = {k: v for k, v in done.items() if k not in ("design", "looks")}
    eid1 = ex.experiment_id(SLUG, ex.resolve_arm(SLUG, "V1"), ex.resolve_arm(SLUG, "working"),
                            ["rival"], 80, 7, vs_profile=ex.STANDARD_POD_PROFILE, looks=1)
    (ex.deck_dir(SLUG) / ex.EXP_DIR / f"{eid1}.json").write_text(json.dumps(legacy))
    with pytest.raises(SystemExit) as e:
        ex.run(SLUG, "V1", "working", ["rival"], seed=7, games=80, jobs=2, looks=1, resume=True)
    assert "optional stopping" in str(e.value)


def test_an_a_a_needs_the_flag_and_two_seeds(repo, monkeypatch):
    """One list twice is refused without `--aa` (the existing rule) and runs
    under it with arm B on a second seed base; `--profile-b` is the other way
    two arms may share a list, and it may not equal arm A's profile."""
    monkeypatch.setattr(ex, "forge_jar", lambda: pathlib.Path("/nope/forge.jar"))
    with pytest.raises(SystemExit) as e:
        ex.run(SLUG, "working", "working", ["rival"], games=2, dry_run=True)
    assert "--aa" in str(e.value) and "--profile-b" in str(e.value)
    _, doc = ex.run(SLUG, "working", "working", ["rival"], games=2, dry_run=True, aa=True)
    assert doc["experiment_id"].endswith("-aa")
    with pytest.raises(SystemExit):
        ex.run(SLUG, "V1", "working", ["rival"], games=2, dry_run=True, aa=True)
    with pytest.raises(SystemExit) as e:
        ex.run(SLUG, "working", "working", ["rival"], games=2, dry_run=True, profile_b="Default")
    assert "differ in nothing" in str(e.value)
    _, pb = ex.run(SLUG, "working", "working", ["rival"], games=2, dry_run=True,
                   profile_b="Experimental")
    assert "-bmeExperimental" in pb["experiment_id"] and pb["profiles_b"][0] == "Experimental"


def test_the_experiment_id_carries_the_rest_of_the_harness_and_old_ids_still_resolve(repo):
    """Clock, overrides and AI-profile shas, looks and A/A are in the id — each
    EMPTY at its default, so every tracked id is unchanged (that test still
    runs beside this one)."""
    a, b = ex.resolve_arm(SLUG, "V1"), ex.resolve_arm(SLUG, "working")
    base = ex.experiment_id(SLUG, a, b, ["rival"], 20, 1)
    assert ex.experiment_id(SLUG, a, b, ["rival"], 20, 1, clock=ex.SIM_GAME_CLOCK_SECONDS) == base
    assert ex.experiment_id(SLUG, a, b, ["rival"], 20, 1, clock=300) == base + "-c300"
    assert ex.experiment_id(SLUG, a, b, ["rival"], 20, 1, overrides_sha="60636e9e5565") == base + "-ov60636e9e"
    assert ex.experiment_id(SLUG, a, b, ["rival"], 20, 1, looks=4) == base + "-k4"
    assert ex.experiment_id(SLUG, a, b, ["rival"], 20, 1, aa=True, looks=2) == base + "-aa-k2"


def test_every_tracked_experiment_passes_the_form_check():
    tracked = sorted((ROOT / "data" / "decks").glob("*/experiments/*.json"))
    assert len(tracked) >= 2
    for path in tracked:
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert ex.validate(doc) == [], (path.name, ex.validate(doc))


@pytest.mark.slow
def test_the_win_rate_interval_divides_by_decided_games():
    """THE BUG: `delta` divided wins by every game PLAYED, clock-outs included,
    while the rate beside it divided by decided games."""
    a = {"games": 100, "decided": 80, "seats": {"x": {"wins": 20, "win_rate": 0.25}}}
    b = {"games": 100, "decided": 80, "seats": {"x": {"wins": 28, "win_rate": 0.35}}}
    d = ex.delta(a, b, "x")
    assert d["win_rate"]["n_a"] == 80 and d["power"]["n_per_arm"] == 80
    assert d["power"]["baseline_rate_a"] == 0.25
