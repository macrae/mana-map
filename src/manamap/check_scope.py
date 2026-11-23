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
    decks    data/decks/<slug>/** -> that deck's checks   ~1 min
    data     the rest of data/ (corpus, pods, overrides) -> the regression tier  ~15 min
    viz      viz/**, tests/test_viz_*, conftest_viz.py   -> the core + the viz unit
                                                            files + browser      ~5 min
    agents   .claude/** (charters, skills)               -> unit + doc guards    ~1 min
    plugin   tools/claude-plugins/**                      -> `claude plugin test`  ~5 s
    python   src/manamap/**, tests/**, tools/*.py          -> THE MINIMUM PATH: the core,
                                                            then the unit and regression
                                                            tests the change reaches
                                                            (`manamap.test_impact`), the
                                                            fleet loops on the changed +
                                                            representative decks   ~1-4 min
    harness  Makefile, pyproject, conftest.py, core_contracts, report_plugin, repo_tree,
             .mcp.json                                    -> core + unit + isolated ~3 min
    ci       .github/**                                   -> the doc guards       ~10 s
    full     any path not listed above                    -> everything          ~22 min

Only TRACKED changes count: commits ahead of the push base (the upstream, or
`origin/main` for a branch that has none), what is staged, and tracked files
modified in the working tree. An untracked file is not something a push carries.

When both the regression tier and the browser suite are in the plan they run
CONCURRENTLY, each output captured to a file and printed in order at the end,
the browser at `-n 2` so the pair fits an 8-core machine (docs/testing.md).

Never less safe than it looks: a path this file does not recognise is FULL, no
push base means FULL, a python change the selector cannot place runs each tier whole,
`make prepush-full` is FULL unconditionally, and CI runs the whole fleet with the regen
NIGHTLY (2026-11-22) — this only decides what blocks the pilot per push.

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
#: Generated FROM the decks, so a deck edit legitimately moves them; their own
#: freshness tests name no single deck and are kept in the deck scope.
_DECK_DERIVED = {"data/decks/index.json"}
_VIZ_TEST = re.compile(r"^tests/(test_viz_[^/]*\.py|conftest_viz\.py)$")
#: The harness itself, or a constant every producer reads: nothing is scoped
#: below a change here, because every tier can see it.
#: THE HARNESS (2026-11-22): what decides how tests are collected and run. It used to be
#: `full` — a one-line Makefile edit ran every tier, ~22 min — but a harness edit can
#: change WHICH tests run and how they are tiered, not what the product does: the whole
#: unit tier plus its isolation proof is the check that can see it, and the nightly
#: full run covers the rest. `config.py` moved to `python`: `test_impact` sends it to the
#: whole tier anyway, since every producer reads its constants at import.
_HARNESS = re.compile(r"^(Makefile|pyproject\.toml|setup\.(py|cfg)|tests/conftest\.py"
                      r"|tests/core_contracts\.py"
                      r"|tests/report_plugin\.py|tests/repo_tree\.py|\.mcp\.json)$")
_CI = re.compile(r"^\.github/")

AREAS = ("full", "harness", "python", "viz", "data", "decks", "agents", "plugin", "ci", "docs")

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
    "harness": ["smoke", "unit", "unit-isolated"],
    "ci": ["docs"],
    "docs": ["docs"],
    "decks": ["decks", "docs"],
    "data": ["regression"],
    "viz": ["smoke", "unit", "browser"],
    "agents": ["unit", "docs"],
    "plugin": ["plugin"],
    # No isolation proof for a source change (2026-10-29): it re-ran the whole unit
    # tier (~1 min, on the critical path) to prove what only a HARNESS change can
    # break, and CI runs it on every push. `full` keeps it.
    # THE MINIMUM PATH (2026-11-22): the core, then the unit and regression tests the
    # change can reach (`manamap.test_impact` — a measured map, the import graph behind
    # it). A change the selector cannot place runs each tier whole.
    "python": ["smoke", "unit", "regression"],
}
#: Everything `make prepush-full` runs: both tiers, the isolation proof and the
#: browser suite. The plugin's tests join only when the plugin changed.
FULL_TIERS = ["unit", "unit-isolated", "regression", "browser"]
#: Cheap and fast-failing first; the two long tiers last, side by side.
TIER_ORDER = ["docs", "plugin", "smoke", "unit", "unit-isolated", "decks", "regression", "browser"]
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
#: The smoke run is part of every unit run that is NOT narrowed; `tier_command` drops it
#: then, so a whole-tier run never pays for the core twice.
#: The pytest runs each tier makes — the band's `total` for a prepush row, which
#: counts its finished pytest children (`_finished_children`).
PYTEST_RUNS = {"docs": 1, "plugin": 0, "smoke": 1, "unit": 1, "unit-isolated": 1, "decks": 1,
               "regression": 2, "browser": 1}
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
    if _HARNESS.match(path):
        return "harness"
    if _CI.match(path):
        return "ci"
    if _DECK_PATH.match(path) or path in _DECK_DERIVED:
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
    m = _DECK_PATH.match(path)
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
    selected = select_tests(areas)
    if selected is not None:
        # A NARROWED run no longer holds the named-file tiers it subsumed: the deck checks
        # and the doc guards come back whenever their area moved.
        wanted = {t for a in areas for t in AREA_TIERS.get(a, ()) if t in ("docs", "decks")}
        tiers = [t for t in TIER_ORDER if t in set(tiers) | wanted]
    shown = "; ".join(f"{a}: {_some(ps)}" for a in AREAS if (ps := areas.get(a)))
    if "full" in areas:
        return _plan("full", areas, decks, f"the whole suite — {shown}", tiers)
    if selected is not None:
        shown += f" — {len(selected)} test file(s) reach it"
    named = sorted(areas)
    # An area the others already cover — drop it and no tier moves — adds nothing
    # to the name: a decklist beside a pilot edit is a "python" push.
    covered = {a for a in named if len(named) > 1
               and tiers_for(set(named) - {a}) == tiers}
    scope = "+".join(a for a in named if a not in covered) or named[0]
    if scope == "decks" and not decks:
        scope, shown = "decks", "only fleet-derived files changed"
    plan = _plan(scope, areas, decks, shown, tiers)
    plan["selected"] = selected
    return plan


def _viz_unit_files():
    return sorted(str(p.relative_to(REPO)) for p in (REPO / "tests").glob("test_viz_*.py"))


def select_tests(areas):
    """The test files a change reaches, or None for "run each tier whole". `python`
    narrows through `manamap.test_impact`; a `viz` edit adds the `test_viz_*` files
    (their unit-tier tests: cache busts, page specs — the browser tier renders the rest).
    A harness edit, a data edit, or a path the selector cannot place returns None."""
    narrowable = {"python", "viz", "docs", "agents", "decks", "ci"}
    if set(areas) - narrowable or not ({"python", "viz"} & set(areas)):
        return None
    out = set()
    if "python" in areas:
        from manamap import test_impact
        got = test_impact.affected_tests(areas["python"])
        if got is None:
            return None
        out |= set(got)
    if "viz" in areas:
        out |= set(_viz_unit_files())
    return sorted(out)


def _some(paths, n=3):
    return ", ".join(paths[:n]) + (f" (+{len(paths) - n} more)" if len(paths) > n else "")


def _plan(scope, areas, decks, why, tiers=None):
    if tiers is None:
        tiers = tiers_for({scope}) if scope in AREA_TIERS or scope == "full" else []
    return {"scope": scope, "areas": areas, "tiers": tiers, "decks": decks, "why": why}


#: The decks a narrowed regression run keeps beside the ones the push moved: the two
#: with the most branches, stacks, channels and logged games, so a change to how decks
#: are measured meets both a typal swarm and a draw-and-discard voltron before the push.
REPRESENTATIVE_DECKS = ("sharknado", "edgar-vampires")


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
    selected = plan.get("selected")
    if tier == "smoke":
        return ["make", "smoke"]
    if tier == "unit":
        if selected is not None:
            return [py, "-m", "pytest", "-m", "unit", *selected] if selected else None
        return ["make", "test"]
    if tier == "unit-isolated":
        return ["make", "test-unit-isolated"]
    if tier == "regression":
        # The fleet regen is left to CI unless everything was asked for.
        if selected is not None and not plan.get("everything"):
            if not selected:
                return None
            # THE FLEET LOOPS, SCOPED (2026-11-22): a narrowed run re-simulates only the
            # decks the push moved plus the representatives; every deck runs nightly.
            keep = sorted(set(plan.get("decks") or []) | set(REPRESENTATIVE_DECKS))
            k = deck_keyword(keep)
            return [py, "-m", "pytest", "-m", "regression and not regen",
                    "-o", "addopts=-q -n auto --dist worksteal", *(["-k", k] if k else []), *selected]
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
    narrowed = plan.get("selected") is not None
    for tier in tiers:
        if tier == "smoke" and not narrowed and "unit" in tiers and plan.get("scope") != "harness":
            continue                                   # a whole unit run already holds the core
        cmd = tier_command(tier, plan, paired=paired and tier in CONCURRENT)
        if cmd is None:
            continue                                   # a narrowed tier that selected nothing
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


#: Tiers that read the composed status files (`restamp`'s targets) and every deck
#: directory on disk; the preflight below runs only when one of them will.
_READS_DECKS = ("regression", "decks")


def untracked_decks():
    """Deck directories with NO tracked file. The manifest, the fleet file and every
    fleet test read the disk, not git, so an uncommitted brew passes locally and
    breaks CI — or, moved aside mid-gate, breaks it locally (black-waltz, 2026-11-12)."""
    rc, out = _git("ls-files", "-z", "--", "data/decks")
    if rc:
        return []
    tracked = {p.split("/")[2] for p in _nul_split(out) if p.count("/") >= 3}
    return sorted(d.name for d in DECKS.iterdir() if d.is_dir() and d.name not in tracked)


def unflagged_cards():
    """`[(slug, card, stem)]`: live-deck cards carrying Forge's `AI:RemoveDeck:All` that
    `unflag.txt` does not name — the regression tier's fleet gate
    (`test_no_live_deck_carries_a_removedeck_all_card_the_list_does_not_unflag`), asked
    in seconds. Meren V23 (2026-11-13) added Blasting Station and Ghoulcaller Gisa and
    the gate said so thirteen minutes in. Empty when the Forge scripts are not
    installed, exactly as that test skips."""
    from manamap.sim import forge_cards, forge_pilot
    if not forge_cards.installed():
        return []
    return forge_pilot.unflag_candidates()


def preflight(plan):
    """Fail in about a minute on what used to fail fifteen minutes in. Returns an exit
    code: 0 to go on."""
    if not any(t in _READS_DECKS for t in plan_tiers(plan)):
        return 0
    loose = untracked_decks()
    if loose:
        print("CHECK SCOPE — PREFLIGHT: untracked deck directories "
              f"{', '.join(loose)} — every fleet test reads them from disk. Commit "
              "them (`git add data/decks/<slug>`) or move them out of data/decks/.")
        return 3
    flagged = unflagged_cards()
    if flagged:
        for slug, name, stem in flagged:
            print(f"    {slug}: {name}  ({stem})")
        print("CHECK SCOPE — PREFLIGHT: live-deck card(s) Forge's AI will never cast "
              "(AI:RemoveDeck:All) that data/forge_overrides/unflag.txt does not name — add "
              "each stem to unflag.txt, run `manamap pilot forge-install --generate`, commit "
              "the list and the new override scripts")
        return 4
    print("  $ manamap pilot restamp --check   (preflight: the composed status files)")
    rc = subprocess.run([sys.executable, "-m", "manamap.cli", "pilot", "restamp", "--check"],
                        cwd=REPO).returncode
    if rc:
        print("CHECK SCOPE — PREFLIGHT FAILED: run `manamap pilot restamp`, commit, retry")
        return rc
    # The fleet's gates, moved here from the commit gate (2026-11-22): every tracked
    # artifact's validator over every live deck — a push-time question, about a minute.
    print("  $ manamap pilot deck-status --all --live   (preflight: every deck's gates)")
    rc = subprocess.run([sys.executable, "-m", "manamap.cli", "pilot", "deck-status", "--all", "--live"],
                        cwd=REPO, capture_output=True, text=True)
    if rc.returncode:
        print(rc.stdout[-2000:] + rc.stderr[-1000:])
        print("CHECK SCOPE — PREFLIGHT FAILED: a deck's gate fails (`manamap pilot deck-status --all`)")
        return rc.returncode
    # AN OPEN BRANCH STAGED AFTER ITS MEASUREMENT (2026-11-12): threat-v1 took three
    # swaps after `net-change --write`, so its report described a list no longer on disk,
    # and the tracked-artifact gate said so fifteen minutes into the regression tier.
    from manamap.pilot.restamp import targets
    for slug, branch in targets("net_change.json"):
        if not branch:
            continue
        rc = subprocess.run([sys.executable, "-m", "manamap.cli", "pilot",
                             "validate-net-change", slug, "--branch", branch],
                            cwd=REPO, capture_output=True, text=True)
        if rc.returncode:
            print(rc.stdout + rc.stderr)
            print(f"CHECK SCOPE — PREFLIGHT FAILED: {slug}@{branch} was staged after its "
                  f"measurement — `manamap pilot net-change {slug} --branch {branch} --write`")
            return rc.returncode
    return 0


def _run(plan):
    print(f"CHECK SCOPE — {plan['scope'].upper()}: {plan['why']}")
    rc = preflight(plan)
    if rc:
        return rc
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


def working_tree_paths():
    """What the NEXT commit would carry: staged and unstaged tracked changes against HEAD."""
    paths = set()
    for args in (("--cached",), ()):
        rc, out = _git("diff", "--name-only", "-z", *args)
        if rc == 0:
            paths.update(_nul_split(out))
    return sorted(paths)


def run_precommit():
    """THE COMMIT GATE, narrowed (2026-11-22): the core always; then the unit tests the
    working tree's python changes reach, or the whole unit tier when a change cannot be
    placed (a harness file, config.py, a viz edit). Fleet checks stay with `prepush`."""
    paths = working_tree_paths()
    areas = {}
    for p in paths:
        areas.setdefault(area_of(p), []).append(p)
    selected = select_tests(areas) if ({"python", "viz"} & set(areas)) else None
    py = sys.executable
    if not paths or set(areas) <= {"docs", "agents", "ci", "decks", "data", "plugin"}:
        cmds = [["make", "smoke"]]
        why = "no python change in the working tree: the core only"
    elif selected is None:
        cmds = [["make", "test"]]
        why = "a change the selector cannot narrow: the whole unit tier"
    else:
        cmds = [["make", "smoke"]] + ([[py, "-m", "pytest", "-m", "unit", *selected]] if selected else [])
        why = f"the core + {len(selected)} test file(s) the change reaches"
    # NO `restamp --check` HERE (2026-11-22): it recomposes all 34 status files (~45 s,
    # twice the tests a one-line edit selects). A stale composed file is not a bug at
    # commit time, and the prepush preflight runs it before anything leaves the machine.
    print(f"PRECOMMIT — {why}")
    for c in cmds:
        print("  $ " + _show(c))
        rc = subprocess.run(c, cwd=REPO).returncode
        if rc not in (0, 5):
            print(f"PRECOMMIT — FAILED (exit {rc})")
            return rc
    print("PRECOMMIT — PASSED")
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
    if cmd == "precommit":
        return run_precommit()
    if cmd == "keyword":
        # The `-k` that keeps the decks this push moved and the representatives
        # (`make regression-scoped`, CI per push). Prints nothing to keep every deck.
        moved = classify(changed_paths(base)).get("decks") or []
        print(deck_keyword(sorted(set(moved) | set(REPRESENTATIVE_DECKS))))
        return 0
    plan = classify(changed_paths(base))
    if cmd == "plan":
        print(json.dumps(_printable(plan), indent=1))
        return 0
    if cmd == "prepush":
        return run(plan)
    raise SystemExit(f"unknown command {cmd!r}: plan | precommit | keyword | prepush | full | deck <slug>... "
                     "[--base REV]")


if __name__ == "__main__":
    sys.exit(main())
