"""The one-heavy-job guard (`manamap.jobs`): what counts as a job, and the hook's verdict.

The session that took the Mac to load 47 (2026-10-29) ran `make prepush` from the main
thread and from subagents at once, and stopped tasks left `train-ability` running.
"""
import io
import json

import pytest

from manamap import jobs

HEAVY = [
    "make prepush",
    "cd /repo && make prepush > log 2>&1; echo $?",
    "make -j2 test-browser",
    "uptime && make prepush",
    "(make test)",
    ".venv/bin/pytest -q",
    ".venv/bin/pytest -m 'browser and not serial_only' -n 4",
    "python -u -m pytest tests",
    "for s in a b; do .venv/bin/manamap pilot regen --slug $s; done",
    "export PYTHONUNBUFFERED=1 && .venv/bin/manamap train-ability >> log",
    "nohup .venv/bin/manamap pilot simulate edgar --games 200 &",
    ".venv/bin/python -m manamap.check_scope prepush",
    "if true; then make regression; fi",
]

LIGHT = [
    ".venv/bin/pytest -n0 -q -k foo",
    ".venv/bin/pytest -o addopts= -n0 tests/test_x.py",
    "grep 'make prepush' log",
    "tail -f prepush.log | grep -E pytest",
    "git commit -m 'manamap run, then make prepush'",
    ".venv/bin/manamap pilot deck-info sharknado",
    ".venv/bin/manamap serve",
    "make jobs",
    "ps aux | grep pytest",
    "echo make test",
    ".venv/bin/python -c 'import pytest'",
]


@pytest.mark.parametrize("cmd", HEAVY)
def test_a_suite_or_a_long_run_is_heavy(cmd):
    assert jobs.is_heavy_command(cmd)


@pytest.mark.parametrize("cmd", LIGHT)
def test_a_word_on_the_line_is_not_a_job(cmd):
    assert not jobs.is_heavy_command(cmd)


@pytest.mark.parametrize("command,kind", [
    ("/repo/.venv/bin/python3 .venv/bin/manamap train-ability", "heavy"),
    ("/usr/bin/make prepush", "heavy"),
    ("/repo/.venv/bin/python -m manamap.check_scope prepush", "heavy"),
    ("/repo/.venv/bin/python -c import sys;exec(eval(sys.stdin.readline()))", "worker"),
    ("/x/chrome-headless-shell --headless=new --remote-debugging-pipe", "browser"),
    ("python3 -m http.server 8001", "server"),
    (".venv/bin/python3 .venv/bin/manamap serve", "server"),
    # The shell wrapping a job is not a second job; the `make` beside it is the job.
    ("/bin/zsh -c source snap.sh && make prepush", None),
    ("grep -E pytest", None),
])
def test_a_running_process_is_judged_by_its_program(command, kind):
    assert jobs.classify(command) == kind


def _row(pid, kind, ppid=500):
    return {"pid": pid, "ppid": ppid, "cpu": 180.0, "etime": "08:05:52", "kind": kind,
            "command": "python3 .venv/bin/manamap train-ability", "orphan": ppid == 1}


def _hook(cmd, running, monkeypatch, capsys):
    monkeypatch.setattr(jobs, "processes", lambda: running)
    code = jobs.hook(io.StringIO(json.dumps({"tool_input": {"command": cmd}})))
    return code, capsys.readouterr().err


def test_a_second_heavy_job_is_refused_and_names_the_first(monkeypatch, capsys):
    code, err = _hook("make prepush", [_row(17140, "heavy", ppid=1)], monkeypatch, capsys)
    assert code == 2
    assert "17140" in err and "ORPHAN" in err and "make reap" in err
    assert jobs.OVERRIDE in err


def test_a_light_command_beside_a_job_passes(monkeypatch, capsys):
    code, _ = _hook(".venv/bin/pytest -n0 -k one", [_row(1, "worker")], monkeypatch, capsys)
    assert code == 0


@pytest.mark.parametrize("running", [[], [_row(9, "browser"), _row(10, "server")]])
def test_a_heavy_job_on_a_quiet_machine_passes(running, monkeypatch, capsys):
    """A browser or a server is not a suite: only heavy jobs and xdist workers block."""
    assert _hook("make prepush", running, monkeypatch, capsys)[0] == 0


def test_the_override_lets_a_deliberate_second_job_through(monkeypatch, capsys):
    code, _ = _hook(f"{jobs.OVERRIDE} make test", [_row(1, "heavy")], monkeypatch, capsys)
    assert code == 0


def test_a_broken_guard_fails_open(monkeypatch, capsys):
    def boom():
        raise OSError("ps is gone")
    monkeypatch.setattr(jobs, "processes", boom)
    assert jobs.hook(io.StringIO(json.dumps({"tool_input": {"command": "make prepush"}}))) == 0
    assert jobs.hook(io.StringIO("not json")) == 0


def test_reap_kills_orphans_only(monkeypatch):
    sent = []
    monkeypatch.setattr(jobs.os, "kill", lambda pid, sig: sent.append(pid))
    rows = [_row(11, "heavy", ppid=1), _row(12, "worker", ppid=11), _row(13, "server", ppid=1)]
    assert jobs.reap(rows) == [11, 13]
    assert sent == [11, 13]
