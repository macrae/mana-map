"""The feedback contract, asserted rather than described.

`console.py` has stated five rules since it was written. Only seven modules
imported it — roughly 96% of commands were bare `print()`, about a quarter of
those hand-rolled aligned columns, and none used colour. A contract nothing
checks is a style guide.

The rule that matters most, and the one asserted hardest here:

    stdout is the ANSWER; stderr is the theatre.

Every progress bar, spinner and state message goes to stderr so that
`--json | jq` stays byte-clean. A bar that leaks into stdout does not look
broken — it produces a JSON parse error somewhere downstream, days later, in
whatever consumed it.

The second rule with teeth: NEVER FAKE A PERCENTAGE. Work of unknown size gets a
spinner and a state message, never a creeping bar. It is enforced structurally
rather than by review — no `total` means `percent` returns None, and there is no
code path that invents one.
"""

import io
import json
import subprocess
import sys

import pytest

from manamap import console


# ── the split ─────────────────────────────────────────────────────────────

def test_a_bar_writes_nothing_to_stdout(capsys):
    """The whole contract in one assertion."""
    with console.task("working", total=3, unit="things") as bar:
        for _ in range(3):
            bar.advance(1, state="a step")
    captured = capsys.readouterr()
    assert captured.out == "", f"a progress bar reached stdout: {captured.out!r}"


def test_json_output_stays_parseable_with_a_bar_running():
    """The failure this prevents is not a broken-looking terminal — it is a JSON
    parse error days later in whatever consumed the payload.

    Driven through a real subprocess because that is the only way to see what a
    pipe actually receives; `capsys` would not catch a bar written to fd 1
    directly.
    """
    code = (
        "import json, sys;"
        "from manamap import console;"
        "ctx = console.task('working', total=2, unit='x');"
        "bar = ctx.__enter__();"
        "bar.advance(1, state='one'); bar.advance(1);"
        "ctx.__exit__(None, None, None);"
        "print(json.dumps({'answer': 42}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == {"answer": 42}, (
        f"stdout was not clean JSON: {out.stdout!r}")


# ── never fake a percentage ───────────────────────────────────────────────

def test_work_of_unknown_size_has_no_percentage():
    """A creeping bar on work nobody has counted is a lie with a progress
    indicator on it. Structural: no total, no percent."""
    with console.task("thinking") as bar:
        bar.state("still going")
        assert bar.percent is None
        assert bar.total is None


def test_work_of_known_size_reports_a_real_fraction():
    with console.task("working", total=4, unit="steps") as bar:
        bar.advance(1)
        assert bar.percent == pytest.approx(0.25)
        bar.advance(3)
        assert bar.percent == pytest.approx(1.0)


# ── one vocabulary ────────────────────────────────────────────────────────

def test_the_state_glyphs_do_not_collide_with_the_evidence_tiers():
    """✓ means RULES-VERIFIED and nothing else. A deck whose file merely exists
    has not been verified by a rules checker, and a glyph that says otherwise is
    the one claim this bench refuses to let anything make."""
    assert set(console.STATES.values()).isdisjoint(set(console.TIERS.values()))
    assert len(set(console.STATES.values())) == len(console.STATES), "glyphs collide"


def test_colour_is_named_for_meaning_not_appearance():
    """`paint(x, "bad")` survives a palette change; `paint(x, "red")` does not."""
    for meaning in ("good", "warn", "bad", "dim", "key"):
        console.paint("x", meaning)
    with pytest.raises(ValueError, match="not one of"):
        console.paint("x", "red")


def test_colour_disappears_when_the_output_is_not_a_terminal(monkeypatch):
    """`NO_COLOR`, a pipe, or `--plain`. An escape sequence in a captured log is
    noise; in a `--json` payload it is a bug."""
    monkeypatch.setenv("MANAMAP_PLAIN", "1")
    assert console.paint("4 of 6 gates", "warn") == "4 of 6 gates"


def test_there_is_no_hand_rolled_ansi_anywhere_in_the_source():
    """One place applies colour. An escape sequence written by hand is one that
    ends up in a payload eventually — this repo has zero, and that is worth
    keeping true rather than rediscovering."""
    import re
    from pathlib import Path

    from manamap import config

    offenders = []
    for path in sorted((config._REPO_ROOT / "src").rglob("*.py")):
        if path.name == "console.py":
            continue
        text = path.read_text(errors="replace")
        if re.search(r"\\033\[|\\x1b\[", text):
            offenders.append(str(path.relative_to(config._REPO_ROOT)))
    assert not offenders, f"raw ANSI outside console.py: {offenders}"


# ── the commands that carry the daily load ────────────────────────────────

#: Commands that take more than three seconds and are run by hand often enough
#: that silence is a real cost. NOT all 107 — ten carry the load, and the
#: decorator makes the rest cheap whenever they matter.
NARRATING = [
    ("manamap/pilot/net_change.py", "two 10,000-game diagnostic arms, ~15s"),
    ("manamap/pilot/benchmark.py", "four measures x 10,000 sims x N decks"),
    ("manamap/pilot/goldfish.py", "10,000 seeded games, twice when a band is declared"),
    ("manamap/pilot/candidates.py", "2,000 sims per candidate card"),
    ("manamap/pilot/autobuild.py", "six stages"),
    ("manamap/pilot/build_corpus.py", "embedding hundreds of chunks"),
]


@pytest.mark.parametrize("module,why", NARRATING, ids=lambda v: v.split("/")[-1])
def test_a_long_command_says_it_is_alive(module, why):
    """A command silent for fifteen seconds is indistinguishable from one that
    has hung. `net-change` was exactly that: 62 bare prints, none of them until
    both arms had finished."""
    from manamap import config

    source = (config._REPO_ROOT / "src" / module).read_text()
    assert "console.task(" in source, (
        f"{module} runs {why} with no progress indicator")
