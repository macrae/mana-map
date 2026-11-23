"""`check_scope`: the suite a push must pass, decided from its diff.

The point is that a change stops paying for tiers it cannot break — and the risk
is the opposite error, a change slipping through on a path that cannot see it.
So the area table is held both ways: what each area runs, and what it must not
drop. `classify` and `tiers_for` are pure (a list of paths in, a plan out); the
git reading and the runner are tested against stubs.
"""
import json
import os
import re

import pytest

from manamap import check_scope as cs

PY = cs.sys.executable


@pytest.fixture(autouse=True)
def _no_preflight(request, monkeypatch):
    """The runner tests stub `subprocess.run` for the tiers; the preflight (its git
    read, its restamp) is its own tests' subject, so it is off everywhere else."""
    if not any(w in request.node.name for w in ("preflight", "untracked")):
        monkeypatch.setattr(cs, "preflight", lambda plan: 0)


def _tiers(*paths):
    return cs.classify(list(paths))["tiers"]


# ── the areas ────────────────────────────────────────────────────────────

@pytest.fixture
def two_files(monkeypatch):
    """The selector stubbed to two test files: these tests are about the PLAN, and the
    measured map under `.pytest_cache` is local state a clean checkout does not have."""
    from manamap import test_impact
    monkeypatch.setattr(test_impact, "affected_tests",
                        lambda paths: ["tests/test_a.py", "tests/test_b.py"])


def test_docs_alone_run_the_doc_guards_and_nothing_else():
    plan = cs.classify(["docs/testing.md", "README.md", "CLAUDE.md", "docs/history/x.md"])
    assert plan["scope"] == "docs" and plan["tiers"] == ["docs"]
    assert cs.commands(plan) == [[PY, "-m", "pytest", "-o", "addopts=", "-n0", "-q",
                                  *cs.DOC_GUARDS]]


def test_deck_data_alone_is_scoped_to_those_decks():
    plan = cs.classify(["data/decks/edgar-vampires/branches/mardu-combo-v1/decklist.txt",
                        "data/decks/edgar-vampires/info.json", "data/decks/heliod/info.json",
                        "data/decks/index.json", "docs/testing.md"])
    assert plan["scope"] == "decks" and plan["decks"] == ["edgar-vampires", "heliod"]
    assert plan["tiers"] == ["docs", "decks"]


def test_the_manifest_alone_keeps_the_checks_that_name_no_deck():
    plan = cs.classify(["data/decks/index.json"])
    assert plan == {**plan, "scope": "decks", "decks": []}
    k = cs.deck_keyword([], every=["edgar-vampires", "heliod"])
    assert k == "not edgar-vampires and not heliod"


def test_the_deck_scope_drops_only_the_other_decks():
    k = cs.deck_keyword(["edgar-vampires"], every=["edgar-vampires", "heliod", "ur-dragon"])
    assert k == "not heliod and not ur-dragon"
    cmds = cs.commands({"scope": "decks", "decks": ["edgar-vampires"], "why": ""})
    pytest_cmd = next(c for c in cmds if "-k" in c)
    assert "not regen" in pytest_cmd
    assert set(cs.DECK_TESTS) <= set(pytest_cmd)
    assert all(__import__("pathlib").Path(cs.REPO / f).is_file() for f in cs.DECK_TESTS)
    assert "edgar-vampires" not in pytest_cmd[pytest_cmd.index("-k") + 1]


def test_viz_runs_the_core_its_own_unit_files_and_the_browser_suite():
    """A JS edit cannot move a Python test: the core, the `test_viz_*` unit files (cache
    busts, page specs) and the browser suite — no whole unit tier, no regression."""
    for p in ("viz/js/shell.js", "viz/deck.html", "viz/style.css",
              "tests/test_viz_shell.py", "tests/conftest_viz.py"):
        assert cs.area_of(p) == "viz", p
    plan = cs.classify(["viz/js/shell.js", "tests/test_viz_shell.py"])
    assert plan["scope"] == "viz" and plan["tiers"] == ["smoke", "unit", "browser"]
    cmds = cs.commands(plan)
    assert cmds[0] == ["make", "smoke"] and cmds[-1] == ["make", "test-browser"]
    unit = cmds[1]
    assert unit[:5][-2:] == ["-m", "unit"] and all(f.startswith("tests/test_viz_") for f in unit[5:])


def test_python_runs_the_core_then_only_the_tests_it_reaches(two_files):
    """THE MINIMUM PATH (2026-11-22): a source edit runs the core, then the unit and
    regression tests the change can reach — not 3,700 tests and fifteen minutes."""
    plan = cs.classify(["src/manamap/pilot/net_change.py", "tests/test_pilot_net_change.py"])
    assert plan["scope"] == "python" and plan["selected"] == ["tests/test_a.py", "tests/test_b.py"]
    assert plan["tiers"] == ["smoke", "unit", "regression"]
    cmds = cs.commands(plan)
    assert cmds[0] == ["make", "smoke"]
    assert cmds[1][-3:] == ["unit", "tests/test_a.py", "tests/test_b.py"]
    assert "regression and not regen" in cmds[2] and cmds[2][-2:] == ["tests/test_a.py", "tests/test_b.py"]


def test_a_change_the_selector_cannot_place_runs_each_tier_whole(monkeypatch):
    from manamap import test_impact
    monkeypatch.setattr(test_impact, "affected_tests", lambda paths: None)
    plan = cs.classify(["src/manamap/config.py"])
    assert plan["selected"] is None
    assert cs.commands(plan) == [["make", "test"], ["make", "regression-prepush"]]
    plan = cs.classify(["tools/docs_index_sizes.py"])
    assert cs.commands(plan) == [["make", "test"], ["make", "regression-prepush"]]


def test_a_narrowed_tier_that_selects_nothing_runs_nothing(monkeypatch):
    from manamap import test_impact
    monkeypatch.setattr(test_impact, "affected_tests", lambda paths: [])
    assert cs.commands(cs.classify(["src/manamap/pilot/net_change.py"])) == [["make", "smoke"]]


def test_prepush_leaves_the_fleet_regen_to_the_nightly_run_and_prepush_full_runs_it(monkeypatch):
    """The fleet regen is the nightly run's (and `make prepush-full`'s), serial, 7-15
    minutes; a scoped push skips it. Asking for everything still runs it."""
    from manamap import test_impact
    monkeypatch.setattr(test_impact, "affected_tests", lambda paths: None)
    plan = cs.classify(["src/manamap/pilot/net_change.py"])
    assert ["make", "regression-prepush"] in cs.commands(plan)
    everything = {**plan, "everything": True}
    assert ["make", "regression"] in cs.commands(everything)
    recipe = (cs.REPO / "Makefile").read_text()
    target = re.search(r"^regression-prepush:.*\n((?:\t.*\n)+)", recipe, re.M)
    assert target and "REGRESSION_PARALLEL" in target.group(1)
    assert "REGRESSION_SERIAL" not in target.group(1)


def test_fleet_data_runs_the_regression_tier():
    for p in ("data/card_roles.json", "data/forge_overrides/unflag.txt",
              "data/pods/standard-v3.json", "data/forge_overrides/README.md"):
        assert cs.area_of(p) == "data", p
    assert _tiers("data/card_roles.json") == ["regression"]


def test_a_charter_runs_the_unit_tier_and_the_doc_guards():
    plan = cs.classify([".claude/agents/deck-doctor.md", ".claude/skills/jarvis/SKILL.md"])
    assert plan["scope"] == "agents" and plan["tiers"] == ["docs", "unit"]


def test_the_plugin_runs_its_own_tests_only():
    plan = cs.classify(["tools/claude-plugins/job-band/hooks/register.tsx"])
    assert plan["scope"] == "plugin"
    assert cs.commands(plan) == [["claude", "plugin", "test", cs.PLUGIN_DIR]]


def test_the_harness_runs_the_unit_tier_and_its_isolation_proof_and_an_unknown_path_is_full():
    """A harness edit changes WHICH tests run and how they are tiered — the whole unit tier
    and its isolation proof are what can see that; the nightly run covers the fleet. A
    path this module has never heard of is still not a reason to run less."""
    for p in ("Makefile", "pyproject.toml", "tests/conftest.py", "tests/report_plugin.py",
              ".mcp.json"):
        plan = cs.classify([p])
        assert plan["scope"] == "harness", (p, plan)
        assert plan["tiers"] == ["smoke", "unit", "unit-isolated"], p
    assert cs.classify([".github/workflows/test.yml"])["scope"] == "ci"
    for p in ("something/new.bin", "manuals/index.html"):
        assert cs.classify([p])["scope"] == "full", p
    assert cs.classify(None)["scope"] == "full"
    assert cs.classify([])["scope"] == "none" and cs.commands(cs.classify([])) == []


def test_full_is_everything_serial_by_default_and_prepush_full_runs_it(monkeypatch):
    monkeypatch.delenv(cs.PAIR_ENV, raising=False)
    plan = cs.classify(["something/new.bin"])
    assert cs.steps(plan) == [[["make", "test"]], [["make", "test-unit-isolated"]],
                              [["make", "regression-prepush"]], [["make", "test-browser"]]]
    assert cs.pytest_runs(plan) == 4
    # The pair is opt-in: measured once, it cost a flake for four minutes.
    monkeypatch.setenv(cs.PAIR_ENV, "1")
    assert cs.steps(plan)[-1] == [["make", "regression-prepush"],
                                  ["make", "test-browser", f"BROWSER_WORKERS={cs.BROWSER_WORKERS_PAIRED}"]]
    recipe = (cs.REPO / "Makefile").read_text()
    assert re.search(r"^prepush-full:.*\n\t.*manamap\.check_scope full", recipe, re.M)
    # A slept laptop stalls a prepush mid-browser-tier: both are held awake.
    for target in ("prepush", "prepush-full"):
        assert re.search(rf"^{target}:.*\n\t\$\(KEEPAWAKE\) ", recipe, re.M), target
    assert "BROWSER_WORKERS ?= 4" in recipe and "-n $(BROWSER_WORKERS)" in recipe


# ── a mixed diff is the union ────────────────────────────────────────────

def test_a_mixed_diff_runs_the_union_of_its_areas(two_files):
    plan = cs.classify(["viz/js/shell.js", "src/manamap/pilot/net_change.py"])
    assert plan["scope"] == "python+viz"
    assert plan["tiers"] == ["smoke", "unit", "regression", "browser"]
    assert sorted(plan["areas"]) == ["python", "viz"]
    assert {"tests/test_a.py", "tests/test_b.py"} <= set(plan["selected"])
    assert any(f.startswith("tests/test_viz_") for f in plan["selected"])
    plan = cs.classify(["docs/viz.md", "viz/js/shell.js"])
    assert plan["scope"] == "docs+viz" and plan["tiers"] == ["docs", "smoke", "unit", "browser"]
    plan = cs.classify(["something/new.bin", "tools/claude-plugins/job-band/hooks/register.tsx"])
    assert plan["scope"] == "full" and "plugin" in plan["tiers"]


def test_a_narrowed_run_keeps_the_deck_checks_and_doc_guards_it_used_to_subsume(two_files):
    """A narrowed unit or regression run may not hold the deck checks or the doc guards,
    so a python change beside a decklist and a doc still runs both."""
    plan = cs.classify(["data/decks/heliod/decklist.txt", "src/manamap/pilot/goldfish.py",
                        "docs/pilot.md"])
    assert plan["decks"] == ["heliod"]
    assert plan["tiers"] == ["docs", "smoke", "unit", "decks", "regression"]
    assert "python" in plan["why"] and "decks" in plan["why"] and "docs" in plan["why"]


def test_a_plan_built_by_hand_derives_its_tiers_from_its_scope():
    assert cs.plan_tiers({"scope": "decks", "decks": ["heliod"], "why": ""}) == ["docs", "decks"]
    assert cs.plan_tiers({"scope": "full", "decks": [], "why": ""}) == cs.FULL_TIERS
    assert cs.plan_tiers({"scope": "none"}) == []


# ── only tracked changes count ───────────────────────────────────────────

def test_untracked_files_are_not_something_a_push_carries(monkeypatch):
    """`CHECK SCOPE — FULL: code changed: NEXT_SESSION.md` — an untracked handoff
    note forced the whole suite onto a data-only commit (2026-10-09). The scope is
    what the push would change: commits ahead of the base, the index, and tracked
    files modified in the tree. `git status`'s `??` lines are never read."""
    calls = []

    def fake_git(*args):
        calls.append(args)
        if args[:2] == ("rev-parse", "--abbrev-ref"):
            return 0, "origin/main\n"
        if args[:3] == ("diff", "--name-only", "-z"):
            rest = args[3:]
            if rest == ("origin/main...HEAD",):
                return 0, "data/decks/heliod/prices.json\0"
            if rest == ("--cached",):
                return 0, "docs/testing.md\0"
            if rest == ():
                return 0, "data/decks/heliod/decklist.txt\0"
        if args[0] == "status":
            raise AssertionError("status --porcelain lists untracked files; never read it")
        raise AssertionError(args)

    monkeypatch.setattr(cs, "_git", fake_git)
    paths = cs.changed_paths()
    assert paths == ["data/decks/heliod/decklist.txt", "data/decks/heliod/prices.json",
                     "docs/testing.md"]
    assert cs.classify(paths)["scope"] == "decks"
    assert not any(a[0] == "status" for a in calls)


def test_a_branch_with_no_upstream_is_measured_from_origin_main(monkeypatch):
    answers = {("rev-parse", "--abbrev-ref", "@{u}"): (128, ""),
               ("rev-parse", "--verify", "-q", "origin/main"): (0, "abc\n")}
    monkeypatch.setattr(cs, "_git", lambda *a: answers.get(a, (0, "")))
    assert cs.push_base() == "origin/main"
    answers[("rev-parse", "--verify", "-q", "origin/main")] = (1, "")
    assert cs.push_base() is None and cs.changed_paths() is None


def test_an_explicit_base_plans_the_working_tree_alone(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(cs, "_git", lambda *a: (seen.append(a), (0, "viz/x.js\0"))[1])
    assert cs.main(["plan", "--base", "HEAD"]) == 0
    assert ("diff", "--name-only", "-z", "HEAD...HEAD") in seen
    assert not any(a[:2] == ("rev-parse", "--abbrev-ref") for a in seen)
    out = json.loads(capsys.readouterr().out)
    assert out["scope"] == "viz"
    assert out["commands"][0] == "make smoke" and out["commands"][-1] == "make test-browser"
    assert out["concurrent"] == []


# ── the concurrent pair ──────────────────────────────────────────────────

class _FakeProc:
    def __init__(self, cmd, stdout, rc, text):
        stdout.write(text.encode())
        self.rc = rc

    def wait(self):
        return self.rc


def test_the_concurrent_runner_prints_both_outputs_in_order_and_fails_if_either_fails(
        monkeypatch, capsys):
    """Two pytest progress lines on one terminal are unreadable, so each command's
    output is captured and printed whole, in the order the commands were given,
    under a header with its exit status; a failure in either fails the pair."""
    rcs = {"regression": 0, "test-browser": 1}

    def fake_popen(cmd, cwd=None, stdout=None, stderr=None):
        return _FakeProc(cmd, stdout, rcs[cmd[1]], f"output of {cmd[1]}\n")

    monkeypatch.setattr(cs.subprocess, "Popen", fake_popen)
    rc = cs.run_concurrently([["make", "regression"], ["make", "test-browser"]])
    out = capsys.readouterr().out
    assert rc == 1
    assert out.index("make regression  (exit 0)") < out.index("output of regression") \
        < out.index("make test-browser  (exit 1)") < out.index("output of test-browser")
    rcs["test-browser"] = 0
    assert cs.run_concurrently([["make", "regression"], ["make", "test-browser"]]) == 0
    rcs["regression"] = 2
    assert cs.run_concurrently([["make", "regression"], ["make", "test-browser"]]) == 2


def test_a_full_run_launches_the_pair_together_when_asked_and_the_rest_one_at_a_time(monkeypatch):
    monkeypatch.setenv(cs.PAIR_ENV, "1")
    monkeypatch.setattr(cs, "preflight", lambda plan: 0)
    launched, ran = [], []
    monkeypatch.setattr(cs.subprocess, "Popen",
                        lambda cmd, **kw: (launched.append(cmd),
                                           _FakeProc(cmd, kw["stdout"], 0, ""))[1])
    monkeypatch.setattr(cs.subprocess, "run",
                        lambda cmd, cwd=None: (ran.append(cmd), type("R", (), {"returncode": 0}))[1])
    assert cs._run(cs.classify(["something/new.bin"])) == 0
    assert ran == [["make", "test"], ["make", "test-unit-isolated"]]
    assert launched == [["make", "regression-prepush"],
                        ["make", "test-browser", "BROWSER_WORKERS=2"]]


def test_a_missing_executable_fails_with_a_sentence(monkeypatch, capsys):
    def missing(cmd, cwd=None):
        raise FileNotFoundError(2, "No such file", cmd[0])
    monkeypatch.setattr(cs.subprocess, "run", missing)
    assert cs._run(cs.classify(["tools/claude-plugins/job-band/hooks/register.tsx"])) == 127
    assert "claude is not installed" in capsys.readouterr().out


# ── the job band ─────────────────────────────────────────────────────────

def test_prepush_is_one_band_row_and_its_pytest_runs_nest_under_it(monkeypatch, capsys):
    """THE JOB GRAPH: prepush writes its own row, and each command it runs inherits
    MANAMAP_JOB_PARENT — so the band draws the tiers under it, and prepush's count is
    its finished children. A failing command fails the row."""
    from manamap import progress

    def fake(cmd, cwd=None):
        child = progress.Progress("pytest x", name=f"pytest{len(ran)}")
        ran.append(child.parent)
        child.start().finish()
        return type("R", (), {"returncode": 1 if len(ran) == 2 else 0})()

    ran = []
    monkeypatch.delenv(progress.PARENT_ENV, raising=False)
    monkeypatch.setattr(cs.subprocess, "run", fake)
    assert cs.run({"scope": "decks", "decks": ["heliod"], "why": "t"}) == 1
    row = json.loads((progress.DIR / f"prepush-{os.getpid()}.json").read_text())
    assert ran == [row["id"], row["id"]] and row["label"] == "prepush decks"
    assert (row["done"], row["total"], row["state"]) == (2, 2, "failed")
    assert cs._finished_children(row["id"]) == 2


def test_prepush_keeps_counting_a_child_whose_file_was_pruned():
    """A full prepush outlives its unit runs by more than the ten minutes a finished
    row stays on disk; the first live run read 2 of 4 because of it."""
    from manamap import progress
    progress.DIR.mkdir(parents=True, exist_ok=True)
    f = progress.DIR / "pytest-1.json"
    f.write_text(json.dumps({"id": "pytest-1", "parent": "prepush-9", "state": "passed"}))
    seen = set()
    assert cs._finished_children("prepush-9", seen) == 1
    f.unlink()
    (progress.DIR / "pytest-2.json").write_text(
        json.dumps({"id": "pytest-2", "parent": "prepush-9", "state": "passed"}))
    assert cs._finished_children("prepush-9", seen) == 2


def test_the_pair_runs_under_the_prepush_row_two_children_at_once(monkeypatch, two_files):
    """Both halves of the pair start while `MANAMAP_JOB_PARENT` names the prepush
    row, so the band draws them side by side beneath it."""
    from manamap import progress
    parents = []

    def fake_popen(cmd, cwd=None, stdout=None, stderr=None):
        parents.append(os.environ.get(progress.PARENT_ENV))
        return _FakeProc(cmd, stdout, 0, "")

    monkeypatch.delenv(progress.PARENT_ENV, raising=False)
    monkeypatch.setenv(cs.PAIR_ENV, "1")
    monkeypatch.setattr(cs, "preflight", lambda plan: 0)
    monkeypatch.setattr(cs.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cs.subprocess, "run",
                        lambda cmd, cwd=None: type("R", (), {"returncode": 0})())
    assert cs.run(cs.classify(["src/manamap/pilot/x.py", "viz/x.js"])) == 0
    assert parents == [f"prepush-{os.getpid()}"] * 2


# ── the preflight (2026-11-12) ───────────────────────────────────────────
# Three gate runs failed fifteen minutes in on things a minute could have shown: nine
# stale info.json after a goldfish change, and an untracked brew the manifest named.

def test_an_untracked_deck_directory_is_named(tmp_path, monkeypatch):
    decks = tmp_path / "decks"
    (decks / "kept").mkdir(parents=True)
    (decks / "brew").mkdir()
    monkeypatch.setattr(cs, "DECKS", decks)
    monkeypatch.setattr(cs, "_git", lambda *a: (0, "data/decks/kept/decklist.txt\0"))
    assert cs.untracked_decks() == ["brew"]


def test_the_preflight_refuses_an_untracked_deck_before_any_tier(tmp_path, monkeypatch):
    """Proved by re-introducing the bug: return 0 from `preflight` and this fails."""
    monkeypatch.setattr(cs, "untracked_decks", lambda: ["brew"])
    ran = []
    monkeypatch.setattr(cs.subprocess, "run", lambda *a, **k: ran.append(a))
    assert cs.preflight({"scope": "decks", "tiers": ["decks"]}) == 3
    assert ran == [], "no restamp, no tier: the refusal comes first"


def test_the_preflight_refuses_a_card_forge_will_never_cast(monkeypatch):
    """Meren V23, 2026-11-13: Blasting Station and Ghoulcaller Gisa carry
    AI:RemoveDeck:All, and the fleet gate said so thirteen minutes into the regression
    tier. Proved by re-introducing the bug: drop the `unflagged_cards` check from
    `preflight` and this reaches restamp instead."""
    monkeypatch.setattr(cs, "untracked_decks", lambda: [])
    monkeypatch.setattr(cs, "unflagged_cards",
                        lambda: [("meren-recursion", "Blasting Station", "blasting_station")])
    ran = []
    monkeypatch.setattr(cs.subprocess, "run", lambda *a, **k: ran.append(a))
    assert cs.preflight({"scope": "decks", "tiers": ["decks"]}) == 4
    assert ran == [], "no restamp, no tier: the refusal comes first"


def test_unflagged_cards_is_empty_without_the_forge_scripts(monkeypatch):
    """CI has no Forge: the preflight must skip there, exactly as the fleet test does."""
    from manamap.sim import forge_cards, forge_pilot
    monkeypatch.setattr(forge_cards, "installed", lambda: False)
    monkeypatch.setattr(forge_pilot, "unflag_candidates",
                        lambda: (_ for _ in ()).throw(AssertionError("must not be asked")))
    assert cs.unflagged_cards() == []


def test_the_preflight_runs_restamp_check_and_stops_on_it(monkeypatch):
    monkeypatch.setattr(cs, "untracked_decks", lambda: [])
    monkeypatch.setattr(cs, "unflagged_cards", lambda: [])
    seen = []

    class Done:
        returncode = 1
    monkeypatch.setattr(cs.subprocess, "run", lambda cmd, **k: (seen.append(cmd), Done)[1])
    assert cs.preflight({"scope": "full", "tiers": ["unit", "regression"]}) == 1
    assert seen and seen[0][-2:] == ["restamp", "--check"]


def test_a_push_that_reads_no_deck_skips_the_preflight(monkeypatch):
    monkeypatch.setattr(cs, "untracked_decks", lambda: ["brew"])
    assert cs.preflight({"scope": "docs", "tiers": ["docs"]}) == 0


def test_the_preflight_refuses_a_branch_staged_after_its_measurement(monkeypatch):
    """threat-v1, 2026-11-12. Proved by re-introducing the bug: drop the branch loop
    from `preflight` and this returns 0."""
    monkeypatch.setattr(cs, "untracked_decks", lambda: [])
    monkeypatch.setattr(cs, "unflagged_cards", lambda: [])

    class Done:
        def __init__(self, rc):
            self.returncode, self.stdout, self.stderr = rc, "", ""
    calls = []

    def fake_run(cmd, **k):
        calls.append(cmd)
        return Done(1 if "validate-net-change" in cmd else 0)
    monkeypatch.setattr(cs.subprocess, "run", fake_run)
    from manamap.pilot import restamp
    monkeypatch.setattr(restamp, "targets", lambda artifact: [("deck", None), ("deck", "b1")])
    assert cs.preflight({"scope": "decks", "tiers": ["decks"]}) == 1
    assert any("validate-net-change" in c and "b1" in c for c in calls)


def test_a_narrowed_regression_run_keeps_the_changed_decks_and_the_representatives(two_files, monkeypatch):
    """Per-deck fleet loops (a fresh net-change per branch, 100-350 s each) run on the decks
    the push moved and the representatives; the nightly run takes the whole fleet."""
    monkeypatch.setattr(cs, "all_slugs", lambda: ["edgar-vampires", "heliod", "sharknado", "gishath"])
    plan = cs.classify(["src/manamap/pilot/net_change.py", "data/decks/heliod/decklist.txt"])
    reg = next(c for c in cs.commands(plan) if "regression and not regen" in c)
    k = reg[reg.index("-k") + 1]
    assert k == "not gishath", k
    for slug in cs.REPRESENTATIVE_DECKS:
        assert (cs.REPO / "data" / "decks" / slug).is_dir(), slug


def test_a_core_list_edit_runs_the_core_and_the_unit_tier_and_nothing_wider(two_files):
    """`core_contracts.py` only moves the `core` mark; a 16-minute prepush for a list edit
    was the harness rule over-reaching."""
    plan = cs.classify(["tests/core_contracts.py"])
    assert plan["scope"] == "core" and plan["tiers"] == ["smoke", "unit"]
    assert cs.commands(plan) == [["make", "smoke"], ["make", "test"]]
    plan = cs.classify(["tests/core_contracts.py", "src/manamap/pilot/bracket.py"])
    cmds = cs.commands(plan)
    assert ["make", "test"] in cmds and any("regression and not regen" in c for c in cmds)
    reg = next(c for c in cmds if "regression and not regen" in c)
    assert reg[-2:] == ["tests/test_a.py", "tests/test_b.py"], "the python change still narrows regression"
