"""The console layer: what it is allowed to draw, and where.

The contract is small and the whole module exists to keep it. Every test here
is one clause of it, and the clauses were chosen because each one, broken,
produces a failure that is either invisible (progress on stdout, inside JSON
somebody pipes to `jq`) or dishonest (a percentage nobody measured).
"""

import json
import os
import pty
import re
import subprocess
import sys

import pytest

from manamap import console

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test decides plainness for itself; none inherits the runner's."""
    monkeypatch.delenv("MANAMAP_PLAIN", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)


# ── Rule 1: stdout is the answer, stderr is the theatre ────────────────────


def test_progress_never_touches_stdout(capsys):
    """THE contract. A bar on stdout ends up inside the JSON somebody pipes.

    Captured output is not a TTY, so this exercises the plain path — which is
    exactly the path a pipe takes, and therefore the one that matters.
    """
    with console.task("Simulating", total=4, unit="games") as t:
        for _ in range(4):
            t.advance()
    with console.task("Thinking") as t:
        t.state("resolving printings")

    out, err = capsys.readouterr()
    assert out == "", f"the console wrote {out!r} to stdout"
    assert "Simulating" in err and "resolving printings" in err, (
        "nothing reached stderr either — the test is not exercising the layer")


def test_err_goes_to_stderr(capsys):
    console.err("a message")
    out, err = capsys.readouterr()
    assert out == ""
    assert err.strip() == "a message"


# ── Rule 2: degrade when nobody is watching a terminal ─────────────────────


def test_captured_output_is_plain(capsys):
    """`capsys` replaces stderr with a non-TTY, which is what a pipe looks like."""
    assert console.is_plain() is True


@pytest.mark.parametrize("var", ["MANAMAP_PLAIN", "NO_COLOR"])
def test_the_escape_hatches(monkeypatch, var):
    monkeypatch.setenv(var, "1")
    assert console.is_plain() is True


def test_plain_output_carries_no_ansi(capsys):
    """A CI log full of escape codes is worse than no progress at all."""
    with console.task("Working", total=3, unit="items") as t:
        for _ in range(3):
            t.advance()
    _, err = capsys.readouterr()
    assert ANSI.search(err) is None, f"ANSI leaked into plain output: {err!r}"
    assert "\r" not in err, "a carriage return in a plain log"


# ── Rule 3: never fake a percentage ────────────────────────────────────────


def test_work_of_unknown_size_has_no_percentage(capsys):
    """The honesty rule, and it is enforced by shape rather than by discipline.

    A bar reading 94% when the code has not measured anything is the same class
    of claim as a figure printed without its interval. There is deliberately no
    API that produces one: omit `total` and `percent` is None, permanently.
    """
    with console.task("Querying EDHREC") as t:
        assert t.total is None
        assert t.percent is None
        t.advance()
        t.state("still going")
        assert t.percent is None, "a percentage appeared from nowhere"


def test_a_known_total_gives_a_real_fraction(capsys):
    with console.task("Auditing", total=16, unit="axes") as t:
        assert t.percent == 0.0
        for _ in range(8):
            t.advance()
        assert t.percent == 0.5
        for _ in range(8):
            t.advance()
        assert t.percent == 1.0


def test_overshooting_does_not_exceed_one(capsys):
    """Forge reports per-JVM; a miscount must not print 130%."""
    with console.task("Simulating", total=10, unit="games") as t:
        t.advance(25)
        assert t.percent == 1.0


# ── Rule 4: removable without moving a figure ──────────────────────────────


def test_plain_mode_still_yields_a_working_task(capsys):
    """`--plain` disables drawing, never the code around it.

    The failure this prevents: a caller wrapped in `with task(...)` that works
    on a terminal and raises in CI, which is the worst possible split because
    only one of the two is watched.
    """
    os.environ["MANAMAP_PLAIN"] = "1"
    try:
        with console.task("Working", total=2, unit="things") as t:
            t.advance()
            t.state("halfway")
            t.advance()
            assert t.done == 2
            assert t.percent == 1.0
    finally:
        del os.environ["MANAMAP_PLAIN"]


def test_set_plain_reaches_subprocesses(monkeypatch):
    """A spawned step must not disagree with its parent about drawing.

    `simulate` shells out to seven JVMs and the pipeline runs steps as children;
    a module-global flag would be invisible to all of them.
    """
    monkeypatch.delenv("MANAMAP_PLAIN", raising=False)
    console.set_plain(True)
    assert os.environ["MANAMAP_PLAIN"] == "1"
    console.set_plain(False)
    assert "MANAMAP_PLAIN" not in os.environ


# ── Rule 5: it must not become the bottleneck ──────────────────────────────


def test_a_fast_loop_does_not_emit_a_line_per_tick(capsys):
    """10,000 sims logging each sim writes 10,000 lines to say one thing."""
    with console.task("Goldfishing", total=10_000, unit="sims") as t:
        for _ in range(10_000):
            t.advance()
    _, err = capsys.readouterr()
    lines = [ln for ln in err.splitlines() if ln.strip()]
    assert len(lines) <= 12, f"{len(lines)} lines for one task"
    assert "10000/10000" in err, "the final state must always be printed"


# ── The gate that matters as the layer spreads ─────────────────────────────


def _run(args, tty):
    """Run a CLI command with stdout captured, stderr on a TTY or not.

    A pty is the only honest way to test the TTY branch: `isatty()` is asked of
    the real file descriptor, and no amount of monkeypatching inside the process
    reproduces what a terminal does to a child process.
    """
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}
    exe = os.path.join(ROOT, ".venv", "bin", "python")
    if not os.path.exists(exe):
        exe = sys.executable
    cmd = [exe, "-m", "manamap.cli"] + args
    if not tty:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
        return p.stdout
    # stderr gets a pty; stdout stays a pipe, so the two are told apart exactly
    # as they would be by `command --json > file` in a terminal.
    primary, replica = pty.openpty()
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=replica,
                           text=True, env=env, timeout=180)
        return p.stdout
    finally:
        os.close(primary)
        os.close(replica)


@pytest.mark.parametrize("args", [["pilot", "deck-version", "--help"]])
def test_json_output_is_identical_with_and_without_a_tty(args):
    """One assertion covering the whole stdout/stderr split.

    It is written now, before `console` is wired into any command, ON PURPOSE:
    a gate added after the thing it guards has already spread is a gate that
    ratifies whatever it finds. This one currently passes trivially and starts
    doing real work the moment G2 puts a progress bar inside a `--json`
    command — which is precisely when the bug it catches becomes possible.
    """
    assert _run(args, tty=False) == _run(args, tty=True)


def test_the_tty_probe_actually_distinguishes_the_two_cases():
    """Prove the harness above can tell a TTY from a pipe.

    Without this, `test_json_output_is_identical...` passes when the pty setup
    is broken and both runs are secretly plain — a green test asserting nothing,
    which is the failure mode this repo keeps finding in its own history.
    """
    probe = ("import sys; from manamap import console; "
             "print(console.is_plain())")
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}
    env.pop("MANAMAP_PLAIN", None)
    env.pop("NO_COLOR", None)
    exe = os.path.join(ROOT, ".venv", "bin", "python")
    if not os.path.exists(exe):
        exe = sys.executable

    piped = subprocess.run([exe, "-c", probe], capture_output=True, text=True,
                           env=env, timeout=120).stdout.strip()
    primary, replica = pty.openpty()
    try:
        on_tty = subprocess.run([exe, "-c", probe], stdout=subprocess.PIPE,
                                stderr=replica, text=True, env=env,
                                timeout=120).stdout.strip()
    finally:
        os.close(primary)
        os.close(replica)

    assert piped == "True", "a piped stderr must read as plain"
    assert on_tty == "False", "a pty stderr must read as a terminal"
