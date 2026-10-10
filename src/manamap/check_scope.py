"""How much of the suite a change has to pass before a push — decided from the diff.

A DECKLIST IS PACKAGING, NOT CODE (2026-10-06). `make prepush` ran both tiers on
every push, so a branch edit to one deck — no line of `src/` moved since the last
full green run — waited fifteen minutes for 2,500 unit tests and every other
deck's freshness checks, none of which it could break. The pilot named the cost:
the workbench had become slow and clunky. So the scope follows what changed.

THE AREAS (2026-10-09). "Code" was one bucket, and the bucket was the whole suite:
a push cycle was ~25 minutes — unit twice, regression's two halves, the browser
suite, all in a row — and it ran eight times in a day, once because an UNTRACKED
handoff note counted as a code change. Now every changed path lands in an area,
each area names the tiers that can see it, a mixed diff runs the union, and the
two long tiers run side by side:

    docs     docs/**, README, CLAUDE.md, any *.md       -> the doc guards       ~10 s
    decks    data/decks/<slug>/**, manuals/p/<slug>.html -> that deck's checks   ~1 min
    data     the rest of data/ (corpus, pods, overrides) -> the regression tier  ~7 min
    viz      viz/**, tests/test_viz_*, conftest_viz.py   -> unit + browser       ~8 min
    agents   .claude/** (charters, skills)               -> unit + doc guards    ~1 min
    plugin   tools/claude-plugins/**                      -> `claude plugin test`  ~5 s
    python   src/manamap/**, tests/**, tools/*.py          -> unit + isolated +
                                                            regression           ~10 min
    full     Makefile, pyproject, conftest.py, config.py,
             CI, and any path not listed above            -> everything          ~15 min

Only TRACKED changes count: commits ahead of the push base (the upstream, or
`origin/main` for a branch that has none), what is staged, and tracked files
modified in the working tree. An untracked file is not something a push carries.

When both the regression tier and the browser suite are in the plan they run
CONCURRENTLY, each output captured to a file and printed in order at the end,
the browser at `-n 2` so the pair fits an 8-core machine (docs/testing.md).

Never less safe than it looks: a path this file does not recognise is FULL, no
push base means FULL, `make prepush-full` is FULL unconditionally, and CI runs the
whole suite on every push regardless — this only decides what blocks the pilot
locally.

    python -m manamap.check_scope plan [--base REV]   what prepush would run, and why
    python -m manamap.check_scope prepush [--base REV]   run it
    python -m manamap.check_scope full                 run everything (`make prepush-full`)
    python -m manamap.check_scope deck <slug>...       the deck-scoped check, no git
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DECKS = REPO / "data" / "decks"

_DECK_PATH = re.compile(r"^data/decks/([a-z0-9][a-z0-9-]*)/")
_MANUAL_PATH = re.compile(r"^manuals/p/([a-z0-9][a-z0-9-]*)\.html$")
#: Generated FROM the decks, so a deck edit legitimately moves them; their own
#: freshness tests name no single deck and are kept in the deck scope.
_DECK_DERIVED = {"data/decks/index.json", "manuals/page.css"}
_VIZ_TEST = re.compile(r"^tests/(test_viz_[^/]*\.py|conftest_viz\.py)$")
#: The harness itself, or a constant every producer reads: nothing is scoped
#: below a change here, because every tier can see it.
_FULL = re.compile(r"^(Makefile|pyproject\.toml|setup\.(py|cfg)|tests/conftest\.py"
                   r"|tests/report_plugin\.py|tests/repo_tree\.py|src/manamap/config\.py"
                   r"|\.github/.*|\.mcp\.json)$")

AREAS = ("full", "python", "viz", "data", "decks", "agents", "plugin", "docs")

DOC_GUARDS = ["tests/test_docs_counts.py", "tests/test_docs_section_count.py"]
PLUGIN_DIR = "tools/claude-plugins/job-band"

#: The tests ABOUT deck artifacts: every tracked file's validator and freshness,
#: branches, the manifest, version lists, the dossier, and the commit protocol
#: (a decklist and its version list never in one commit). A decklist edit can
#: break these and nothing else in `src/`-free territory — the fleet-wide
#: calibrations re-derive constants from code and wait for CI. Measured: the
#: whole regression tier minus other decks took 5:13 for Edgar; this set ~1:45.
DECK_TESTS = ["tests/test_pilot_tracked_artifacts_validate.py",
              "tests/test_pilot_artifact_freshness.py",
              "tests/test_pilot_deck_branch.py", "tests/test_pilot_validate_branch.py",
              "tests/test_pilot_deck_manifest.py", "tests/test_pilot_deck_versions.py",
              "tests/test_pilot_deck_status.py", "tests/test_pilot_deck_info.py",
              "tests/test_pilot_commit_protocol.py"]

#: What each area's change can break — the tiers that can SEE it. The unit tier
#: reads no tracked data (`make test-unit-isolated` proves it), so data and deck
#: edits skip it; the browser suite renders `viz/` and nothing else re-tests a
#: page; the regression tier is where a corpus or pilot change shows.
AREA_TIERS = {
    "docs": ["docs"],
    "decks": ["decks", "docs"],
    "data": ["regression"],
    "viz": ["unit", "browser"],
    "agents": ["unit", "docs"],
    "plugin": ["plugin"],
    "python": ["unit", "unit-isolated", "regression"],
}
#: Everything `make prepush-full` runs: both tiers, the isolation proof and the
#: browser suite. The plugin's tests join only when the plugin changed.
FULL_TIERS = ["unit", "unit-isolated", "regression", "browser"]
#: Cheap and fast-failing first; the two long tiers last, side by side.
TIER_ORDER = ["docs", "plugin", "unit", "unit-isolated", "decks", "regression", "browser"]
#: The pair that CAN run concurrently when both are in a plan — opt-in with
#: `MANAMAP_PREPUSH_PAIR=1`. Measured 2026-10-09 on the 8-core Mac: paired, the
#: browser suite at two workers took 15 min beside regression (7 min alone at
#: four), regression slowed from 7 to 9.4, and the contention flaked a
#: canvas-timing test — 16.6 min wall against ~21 serial, for a flake. Serial is
#: the default; the pair stays for a machine with the cores to spare.
CONCURRENT = ("regression", "browser")
PAIR_ENV = "MANAMAP_PREPUSH_PAIR"


def pair_enabled():
    return os.environ.get(PAIR_ENV) == "1"
#: A tier that is a subset of others, dropped when they are all present: the deck
#: checks and the doc guards are regression- and unit-tier tests of named files.
_SUBSUMED = {"decks": {"unit", "regression"}, "docs": {"unit", "regression"}}
#: The pytest runs each tier makes — the band's `total` for a prepush row, which
#: counts its finished pytest children (`_finished_children`).
PYTEST_RUNS = {"docs": 1, "plugin": 0, "unit": 1, "unit-isolated": 1, "decks": 1,
               "regression": 2, "browser": 2}
#: The browser suite's workers when it shares the machine with the regression
#: tier (`-n auto`, every core): four Chromiums beside eight pytest workers
#: oversubscribe an 8-core Mac. `make test-browser` alone keeps its four.
BROWSER_WORKERS_PAIRED = 2


def _git(*args):
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return r.returncode, r.stdout


def push_base():
    """The commit a push is measured from: the upstream, or `origin/main` for a
    branch that has none yet (a push would create it from there). None when the
    repo has neither — and None means FULL."""
    rc, out = _git("rev-parse", "--abbrev-ref", "@{u}")
    if rc == 0 and out.strip():
        return out.strip()
    rc, _ = _git("rev-parse", "--verify", "-q", "origin/main")
    return "origin/main" if rc == 0 else None


def _nul_split(out):
    return [p for p in out.split("\0") if p]


def changed_paths(base=None):
    """The TRACKED paths a push would carry: commits ahead of the base, the index,
    and tracked files modified in the working tree. Never an untracked file — a
    handoff note nobody committed forced the full suite onto a data-only commit
    (2026-10-09). None when there is no base to diff against."""
    base = base or push_base()
    if base is None:
        return None
    paths = set()
    for args in ((f"{base}...HEAD",), ("--cached",), ()):
        rc, out = _git("diff", "--name-only", "-z", *args)
        if rc == 0:
            paths.update(_nul_split(out))
    return sorted(paths)


def area_of(path):
    """The area one changed path belongs to (see the module docstring)."""
    if _FULL.match(path):
        return "full"
    if _DECK_PATH.match(path) or _MANUAL_PATH.match(path) or path in _DECK_DERIVED:
        return "decks"
    if path.startswith("data/"):
        return "data"
    if path.startswith("viz/") or _VIZ_TEST.match(path):
        return "viz"
    if path.startswith(".claude/"):
        return "agents"
    if path.startswith("tools/claude-plugins/"):
        return "plugin"
    if path.startswith("src/manamap/") or path.startswith("tests/") \
            or (path.startswith("tools/") and path.endswith(".py")):
        return "python"
    if path.startswith("docs/") or path.endswith(".md"):
        return "docs"
    return "full"


def _deck_of(path):
    m = _DECK_PATH.match(path) or _MANUAL_PATH.match(path)
    return m.group(1) if m else None


def tiers_for(areas):
    """The tiers a set of areas needs, in run order, subsumed ones dropped."""
    if "full" in areas:
        tiers = set(FULL_TIERS) | {t for a in areas if a != "full" for t in AREA_TIERS[a]}
    else:
        tiers = {t for a in areas for t in AREA_TIERS[a]}
    tiers -= {t for t, by in _SUBSUMED.items() if by <= tiers}
    return [t for t in TIER_ORDER if t in tiers]


def classify(paths):
    """`paths` -> a plan: {"scope", "areas": {area: [paths]}, "tiers", "decks", "why"}.

    `scope` is the areas joined ("python+viz"), "full", or "none"; "decks" lists
    the decks whose files moved (the deck tier deselects every other)."""
    if paths is None:
        return _plan("full", {}, [], "no push base to diff against")
    if not paths:
        return _plan("none", {}, [], "nothing to push")
    areas = {}
    for p in paths:
        areas.setdefault(area_of(p), []).append(p)
    decks = sorted({d for p in areas.get("decks", []) if (d := _deck_of(p))})
    tiers = tiers_for(set(areas))
    shown = "; ".join(f"{a}: {_some(ps)}" for a in AREAS if (ps := areas.get(a)))
    if "full" in areas:
        return _plan("full", areas, decks, f"the whole suite — {shown}", tiers)
    named = sorted(areas)
    # An area the others already cover — drop it and no tier moves — adds nothing
    # to the name: a decklist beside a pilot edit is a "python" push.
    covered = {a for a in named if len(named) > 1
               and tiers_for(set(named) - {a}) == tiers}
    scope = "+".join(a for a in named if a not in covered) or named[0]
    if scope == "decks" and not decks:
        scope, shown = "decks", "only fleet-derived files changed"
    return _plan(scope, areas, decks, shown, tiers)


def _some(paths, n=3):
    return ", ".join(paths[:n]) + (f" (+{len(paths) - n} more)" if len(paths) > n else "")


def _plan(scope, areas, decks, why, tiers=None):
    if tiers is None:
        tiers = tiers_for({scope}) if scope in AREA_TIERS or scope == "full" else []
    return {"scope": scope, "areas": areas, "tiers": tiers, "decks": decks, "why": why}


def all_slugs():
    return sorted(d.name for d in DECKS.iterdir() if d.is_dir())


def deck_keyword(decks, every=None):
    """A `-k` expression that drops the tests parametrized on OTHER decks and keeps
    everything else — the changed decks' cases and every check that names none."""
    others = [s for s in (every or all_slugs()) if s not in set(decks)]
    return " and ".join(f"not {s}" for s in others) if others else ""


def plan_tiers(plan):
    """A plan's tiers; derived from its scope for a plan built by hand."""
    if "tiers" in plan:
        return plan["tiers"]
    return tiers_for({plan["scope"]}) if plan["scope"] != "none" else []


def tier_command(tier, plan, paired=False):
    """The command one tier runs (a list, from the repo root). The Makefile is the
    one home of each tier's definition; this composes them."""
    py = sys.executable
    if tier == "unit":
        return ["make", "test"]
    if tier == "unit-isolated":
        return ["make", "test-unit-isolated"]
    if tier == "regression":
        # The fleet regen is left to CI unless everything was asked for.
        return ["make", "regression" if plan.get("everything") else "regression-prepush"]
    if tier == "browser":
        return ["make", "test-browser"] + (
            [f"BROWSER_WORKERS={BROWSER_WORKERS_PAIRED}"] if paired else [])
    if tier == "docs":
        return [py, "-m", "pytest", "-o", "addopts=", "-n0", "-q", *DOC_GUARDS]
    if tier == "plugin":
        return ["claude", "plugin", "test", PLUGIN_DIR]
    if tier == "decks":
        k = deck_keyword(plan.get("decks", []))
        # `-m ""`: these files hold both tiers' deck tests; `not regen` keeps the
        # in-place fleet regen (~6 min, every deck) for the full suite and CI.
        return [py, "-m", "pytest", "-m", "not regen", "-n", "auto", "-o", "addopts=", "-q",
                *(["-k", k] if k else []), *DECK_TESTS]
    raise ValueError(f"unknown tier {tier!r}")


def steps(plan):
    """The plan's commands as ordered GROUPS; the commands in one group run at
    the same time. Only the regression/browser pair ever shares a group."""
    tiers = plan_tiers(plan)
    paired = pair_enabled() and all(t in tiers for t in CONCURRENT)
    out, pair = [], []
    for tier in tiers:
        cmd = tier_command(tier, plan, paired=paired and tier in CONCURRENT)
        if paired and tier in CONCURRENT:
            pair.append(cmd)
            if len(pair) == len(CONCURRENT):
                out.append(pair)
        else:
            out.append([cmd])
    return out


def commands(plan):
    """The commands a plan runs, flattened in order (lists, run from the repo root)."""
    return [cmd for group in steps(plan) for cmd in group]


def pytest_runs(plan):
    tiers = plan_tiers(plan)
    # `regression-prepush` is one pytest run; `make regression` is two.
    skipped_regen = "regression" in tiers and not plan.get("everything")
    return sum(PYTEST_RUNS[t] for t in tiers) - skipped_regen


def _finished_children(job_id, seen=None):
    """The band's count for prepush: its child pytest runs that have finished.

    `seen` carries the ids already counted, because a finished row does not stay on
    disk: a new job prunes finished files older than ten minutes (`progress`), and
    a full prepush's regression half outlives its unit runs by more than that. The
    first live run counted 2 of 4 for exactly that reason (2026-10-08)."""
    from manamap import progress
    seen = set() if seen is None else seen
    for f in progress.DIR.glob("*.json"):
        try:
            doc = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if doc.get("parent") == job_id and doc.get("state") != "running":
            seen.add(doc.get("id") or f.stem)
    return len(seen)


def run(plan):
    """Run the plan's commands as one `prepush` row on the job band, its pytest runs
    nested under it (they inherit `MANAMAP_JOB_PARENT`, `manamap.progress`) — two
    at once while the regression/browser pair runs."""
    from manamap.progress import Progress
    total = pytest_runs(plan)
    with Progress(f"prepush {plan['scope']}", total=total or None, unit="runs",
                  name="prepush") as p:
        done = set()
        p.counter = lambda: _finished_children(p.id, done)
        rc = _run(plan)
        if rc:
            p.advance(failed=1)
        return rc


def _show(cmd):
    return " ".join(cmd if len(" ".join(cmd)) < 160 else cmd[:8] + ["…"])


def run_concurrently(cmds, cwd=REPO):
    """Run `cmds` at the same time, each output captured to its own file and
    printed in order once every one has finished, under a header naming the
    command and its exit status. The exit status is the first non-zero one.

    Captured rather than interleaved: two pytest progress lines on one terminal
    are unreadable, and the band already shows both running (each is a child of
    the prepush row). Nothing is lost — a failure's output is printed whole."""
    with tempfile.TemporaryDirectory(prefix="prepush-") as tmp:
        procs = []
        for i, cmd in enumerate(cmds):
            out = open(Path(tmp) / f"{i}.log", "w+b")
            procs.append((cmd, out, subprocess.Popen(cmd, cwd=cwd, stdout=out,
                                                     stderr=subprocess.STDOUT)))
        rcs = []
        for cmd, out, proc in procs:
            rc = proc.wait()
            rcs.append(rc)
        for (cmd, out, _), rc in zip(procs, rcs):
            out.flush()
            out.seek(0)
            print(f"\n───── {_show(cmd)}  (exit {rc}) ─────")
            sys.stdout.write(out.read().decode("utf-8", errors="replace"))
            sys.stdout.flush()
            out.close()
    return next((rc for rc in rcs if rc not in (0, 5)), 0)


def _run(plan):
    print(f"CHECK SCOPE — {plan['scope'].upper()}: {plan['why']}")
    tiers = plan_tiers(plan)
    if plan["scope"] != "full":
        print(f"  tiers: {', '.join(tiers) or '(none)'} "
              "(CI runs the full suite on every push; this is what blocks locally)")
    for group in steps(plan):
        try:
            if len(group) == 1:
                print("  $ " + _show(group[0]))
                rc = subprocess.run(group[0], cwd=REPO).returncode
            else:
                print("  $ " + "   ∥   ".join(_show(c) for c in group)
                      + "   (concurrently; each output printed whole when both finish)")
                rc = run_concurrently(group)
        except FileNotFoundError as e:       # `claude` not on PATH, say
            print(f"CHECK SCOPE — FAILED: {e.filename or e} is not installed")
            return 127
        if rc not in (0, 5):     # 5: nothing selected, e.g. a deck with no tests
            print(f"CHECK SCOPE — FAILED (exit {rc}): fix it before pushing")
            return rc
    print(f"CHECK SCOPE — PASSED ({plan['scope']})")
    return 0


def _printable(plan):
    return {**plan, "commands": [" ".join(c) for c in commands(plan)],
            "concurrent": [[" ".join(c) for c in g] for g in steps(plan) if len(g) > 1]}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    base = None
    if "--base" in argv:
        i = argv.index("--base")
        base = argv[i + 1]
        del argv[i:i + 2]
    cmd = argv[0] if argv else "plan"
    if cmd == "deck":
        slugs = argv[1:]
        unknown = [s for s in slugs if s not in all_slugs()]
        if not slugs or unknown:
            raise SystemExit(f"check-deck needs deck slugs; unknown: {unknown or '(none given)'}")
        return run({"scope": "decks", "decks": slugs, "why": f"asked for {', '.join(slugs)}"})
    if cmd == "full":
        return run({**_plan("full", {}, [], "asked for everything (make prepush-full)"),
                    "everything": True})
    plan = classify(changed_paths(base))
    if cmd == "plan":
        print(json.dumps(_printable(plan), indent=1))
        return 0
    if cmd == "prepush":
        return run(plan)
    raise SystemExit(f"unknown command {cmd!r}: plan | prepush | full | deck <slug>... "
                     "[--base REV]")


if __name__ == "__main__":
    sys.exit(main())
