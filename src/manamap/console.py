"""The terminal is a surface, and this is the only module that draws on it.

The bench is used from the command line far more than from a browser: 66 pilot
subcommands, 18 top-level ones, and that is where the analysis, the searching
and the prototyping happen. It was also silent while it worked. Measured before
this module existed: **two of eighteen long-running operations reported any
progress at all**. `simulate` runs 45–62 minutes across seven JVMs and printed
nothing until it finished, which is indistinguishable from a hung process.

Five rules, and each of them is load-bearing rather than stylistic.

**1. stdout is the ANSWER; stderr is the theatre.** Every spinner, bar and state
message goes to stderr. `manamap pilot deck-facts heliod --json | jq` has to
keep working byte-for-byte, and a progress bar on stdout would end up inside the
JSON. This is the rule the rest of the module exists to keep, and
`tests/test_console.py` asserts it by capturing both streams separately.

**2. TTY-aware, and it degrades to plain lines.** Piped, redirected, or under
CI, the bars become ordinary one-line-per-event log output. CI runs `make test`;
a log full of `\\r` and ANSI escapes is worse than no progress at all. The
browser suite already taught this repo what environment-sensitive output costs.

**3. NEVER FAKE A PERCENTAGE.** Determinate work — N games, N cards, N axes —
gets a bar with real counts. Work whose size is unknown gets a spinner and a
*state message* ("querying EDHREC", "resolving 47 printings"), never a bar
creeping toward a number nobody measured. A bar reading 94% when the code does
not know is the same class of claim as a figure printed without its interval,
and this repo already refuses the second one. `task()` enforces it: pass no
`total` and you cannot get a percentage.

**4. Presentation must be removable without moving a figure.** `--plain` or
`MANAMAP_PLAIN=1` disables the whole layer. Nothing here returns a value that
computation reads; the module writes and nothing else. That is the same split
`deck_facts` keeps with its renderers, one layer out.

**5. It must not become the bottleneck.** Refresh is capped. A bar that redraws
once per game across seven JVMs is a bar measuring itself.

`rich` is an explicit dependency in `pyproject.toml`. It was present in the
environment for months only because `typer` happened to require it — a
transitive dependency is not an available one, and this module would have
started failing the day something dropped typer.
"""

import os
import sys
import time

# Imported lazily inside `_rich()`. `rich` costs ~50 ms to import and most
# commands are far shorter than that; a plain run must not pay for a renderer it
# is not going to use.
_RICH = None
_RICH_TRIED = False

#: Evidence tiers, as one glyph each. The SAME three the manual and the deck
#: page use — a tier that reads ✓ in the browser and "verified" in the terminal
#: is two vocabularies for one contract.
TIERS = {"verified": "✓", "derived": "◆", "coaching": "★"}


def _rich():
    """The rich namespace, or None. Absence is survivable and is not an error."""
    global _RICH, _RICH_TRIED
    if not _RICH_TRIED:
        _RICH_TRIED = True
        try:
            from rich.console import Console
            from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                                       SpinnerColumn, TextColumn, TimeElapsedColumn,
                                       TimeRemainingColumn)
            _RICH = {
                "Console": Console, "Progress": Progress, "BarColumn": BarColumn,
                "TextColumn": TextColumn, "SpinnerColumn": SpinnerColumn,
                "MofNCompleteColumn": MofNCompleteColumn,
                "TimeElapsedColumn": TimeElapsedColumn,
                "TimeRemainingColumn": TimeRemainingColumn,
            }
        except ImportError:
            _RICH = None
    return _RICH


def is_plain():
    """Should this run draw anything at all?

    Four ways to end up plain, and the last one is the important one: a run whose
    stderr is not a terminal is being captured — by a pipe, a file, a CI log or a
    test — and none of those readers want a carriage return.
    """
    if os.environ.get("MANAMAP_PLAIN"):
        return True
    if os.environ.get("NO_COLOR"):        # the no-color.org convention
        return True
    if _rich() is None:
        return True
    return not sys.stderr.isatty()


def set_plain(on=True):
    """Force plain output for the rest of the process — the `--plain` flag.

    Implemented through the environment rather than a module global so that a
    subprocess (Forge's JVMs, a spawned pipeline step) inherits the decision
    instead of quietly disagreeing with its parent.
    """
    if on:
        os.environ["MANAMAP_PLAIN"] = "1"
    else:
        os.environ.pop("MANAMAP_PLAIN", None)


def err(message):
    """One line to stderr. The floor every other helper is built on."""
    print(message, file=sys.stderr, flush=True)


class _Task:
    """A unit of work in progress.

    Determinate when it was given a `total`, indeterminate when it was not, and
    the distinction is not cosmetic: `percent` is None for the second kind and
    there is no code path that invents one.
    """

    def __init__(self, label, total=None, unit="", plain=True, progress=None, task_id=None):
        self.label = label
        self.total = total
        self.unit = unit
        self.done = 0
        self._plain = plain
        self._progress = progress
        self._task_id = task_id
        self._started = time.monotonic()
        # Plain mode prints milestones rather than every tick: a 10,000-sim run
        # that logs each sim writes 10,000 lines into a CI log to say one thing.
        self._last_print = 0.0

    @property
    def percent(self):
        """A real fraction, or None. Never a guess — see rule 3."""
        if not self.total:
            return None
        return min(1.0, self.done / self.total)

    @property
    def elapsed(self):
        return time.monotonic() - self._started

    def advance(self, n=1, state=None):
        """Move the work forward, optionally saying what it is doing now."""
        self.done += n
        if state is not None:
            self.state(state)
            return
        if self._progress is not None:
            self._progress.update(self._task_id, completed=self.done)
        elif self.total:
            self._plain_tick()

    def state(self, message):
        """Say what is happening. The only progress an indeterminate task has."""
        if self._progress is not None:
            self._progress.update(self._task_id, description=f"{self.label} · {message}",
                                  completed=self.done)
        else:
            err(f"  {self.label}: {message}")

    def _plain_tick(self):
        """Milestones only: every 10% of a known total, at most once a second."""
        now = time.monotonic()
        if now - self._last_print < 1.0 and self.done < self.total:
            return
        step = max(1, self.total // 10)
        if self.done % step and self.done < self.total:
            return
        self._last_print = now
        err(f"  {self.label}: {self.done}/{self.total} {self.unit}".rstrip())


class _NullCtx:
    """A task that draws nothing, for the plain path. Same API, no output."""

    def __init__(self, task):
        self.task = task

    def __enter__(self):
        return self.task

    def __exit__(self, *exc):
        return False


def task(label, total=None, unit=""):
    """Work in progress, as a context manager.

        with task("Simulating", total=100, unit="games") as t:
            for game in games:
                t.advance()

        with task("Querying EDHREC") as t:      # size unknown -> spinner
            t.state("fetching average deck")

    Pass a `total` and you get a bar with real counts; omit it and you get a
    spinner. There is deliberately no third option.
    """
    plain = is_plain()
    if plain:
        t = _Task(label, total, unit, plain=True)
        if total:
            err(f"{label}: {total} {unit}".rstrip())
        else:
            err(f"{label}…")
        return _NullCtx(t)
    return _RichTask(label, total, unit)


class _RichTask:
    """The drawing path. Everything here is presentation and nothing else."""

    def __init__(self, label, total, unit):
        self.label = label
        self.total = total
        self.unit = unit
        self._progress = None
        self._task = None

    def __enter__(self):
        r = _rich()
        console = r["Console"](stderr=True)          # rule 1, in one argument
        if self.total:
            columns = [
                r["TextColumn"]("[bold]{task.description}"),
                r["BarColumn"](bar_width=32),
                r["MofNCompleteColumn"](),
                r["TextColumn"](self.unit),
                r["TimeElapsedColumn"](),
                r["TimeRemainingColumn"](),
            ]
        else:
            # No bar and no percentage for work of unknown size. A spinner is an
            # honest "still going"; a bar would be a claim about how far.
            columns = [
                r["SpinnerColumn"](),
                r["TextColumn"]("[bold]{task.description}"),
                r["TimeElapsedColumn"](),
            ]
        self._progress = r["Progress"](
            *columns, console=console, transient=False,
            refresh_per_second=8,                    # rule 5
        )
        self._progress.start()
        self._task = self._progress.add_task(self.label, total=self.total)
        return _Task(self.label, self.total, self.unit, plain=False,
                     progress=self._progress, task_id=self._task)

    def __exit__(self, *exc):
        if self._progress is not None:
            self._progress.stop()
        return False
