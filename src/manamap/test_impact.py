"""Which tests can a change reach? A MEASURED map first, the import graph behind it.

    python -m manamap.test_impact [PATH …]      # the test files a set of changed paths reaches
    python -m manamap.test_impact --build COVERAGE_FILE   # write the measured map (make impact-map)

TWO MAPS, AND WHY THE MEASURED ONE LEADS (2026-11-22). The static import graph below is
honest and nearly useless here: `pilot/common.py` lazily imports `deck_edit`, `goldfish`
and `agent_cache`, and `deck_edit` names every rebuild step by string, so on paper a
`bracket.py` edit could reach 275 of 311 test files. Coverage with a per-test context
records which source files each test ACTUALLY executes; measured on the unit tier, the
same edit reaches 2 (`ability_graph` 7, `deck_edit` 7, `goldfish_turn` 16, `common` 107).
`make impact-map` builds it in about two minutes into `.pytest_cache/manamap/impact-map.json`.

The selection is the union, per changed source file, of: the tests the measured map saw
executing it; or, when the map has never seen the file (new, or only run at import), the
tests whose static import closure reaches it. `config.py` is the exception that returns
None (the whole tier): its constants run at import, outside any test's context, and every
producer reads them. A changed test file always selects itself and any test importing it.

THE MINIMUM PATH (2026-11-22, docs/testing.md). A one-line edit to `src/manamap/pilot/
bracket.py` ran all ~3,700 unit tests and the whole regression half (~15 min), because
the `python` area was all or nothing. This answers the narrower question — which test
files import, directly or transitively, a module that changed — and `check_scope`
runs those plus the core (`tests/core_contracts.py`), which runs on every change.

AN EDGE IS ANYTHING THAT LOADS A MODULE, and the two kinds the import statement misses
are counted on purpose, because a missed edge is not a red test, it is a STALE PASS:

  * a module-path STRING (`"manamap.pilot.poh"`) — `registry.py` dispatches every
    subcommand by string, and the rebuild chains in `deck_edit`, `deck_branch` and
    `regen` name their steps the same way;
  * a test that DRIVES THE CLI (`subprocess … manamap …`, `run_pilot_step`, `cli.main`)
    reaches every subcommand, so it depends on `manamap.pilot.registry` and runs on any
    change.

A test that imports another test module (`from test_pilot_engine_graph import …`)
inherits that module's edges; the shared harness (`conftest`, `conftest_viz`) does not —
a harness edit runs the whole unit tier, and a module reached ONLY through a conftest
fixture is the one gap, covered by the nightly full run. `config.py` is reached by nearly everything, and is
treated like any module. A changed path this cannot place (a `.json` under `src/`, a
helper outside `tests/`) returns None: the caller runs the whole tier.
"""
import ast
import re
import sys
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
TESTS = REPO / "tests"

_MODULE_STR = re.compile(r"^manamap(\.[A-Za-z_][\w]*)+$")
_CLI_DRIVE = re.compile(r"""(["']manamap["']\s*,\s*["']pilot["']|\.venv/bin/manamap|run_pilot_step|cli\.main\(|"""
                        r"""\[\s*sys\.executable\s*,\s*["']-m["']\s*,\s*["']manamap)""")
#: Test-side helpers that are imported like modules by test files.
_TEST_HELPERS = ("conftest", "conftest_viz", "core_contracts", "repo_tree", "report_plugin")


def module_of(path):
    """`src/manamap/pilot/bracket.py` -> `manamap.pilot.bracket`; None outside the package."""
    p = Path(path)
    try:
        rel = (p if p.is_absolute() else REPO / p).resolve().relative_to(SRC)
    except ValueError:
        return None
    if rel.suffix != ".py":
        return None
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _refs(path, package):
    """Every `manamap.*` module and every test module a file names: imports (absolute and
    relative, top level or inside a function), and module-path string literals."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set(), set()
    mods, tests = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                (mods if a.name.startswith("manamap") else tests).add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[:len(parts) - node.level + 1])
                dotted = f"{base}.{node.module}" if node.module else base
            else:
                dotted = node.module or ""
            if dotted.startswith("manamap"):
                mods.add(dotted)
                mods.update(f"{dotted}.{a.name}" for a in node.names)
            else:
                tests.add(dotted)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and _MODULE_STR.match(node.value):
            mods.add(node.value)
    return mods, {t for t in tests if t.startswith("test_") or t in _TEST_HELPERS}


@lru_cache(maxsize=1)
def _graph():
    """{module: set(modules it reaches directly)} over src/manamap, names that are not
    modules (a function imported by name) dropped."""
    files = {module_of(p): p for p in (SRC / "manamap").rglob("*.py")}
    files.pop(None, None)
    known = set(files)
    graph = {}
    for mod, path in files.items():
        pkg = mod if path.name == "__init__.py" else mod.rsplit(".", 1)[0]
        refs, _ = _refs(path, pkg)
        graph[mod] = {r for r in refs if r in known and r != mod}
    return graph


def closure(modules):
    """Every module the given ones reach, themselves included."""
    graph = _graph()
    seen, queue = set(), [m for m in modules if m in graph]
    while queue:
        m = queue.pop()
        if m in seen:
            continue
        seen.add(m)
        queue.extend(graph.get(m, ()))
    return seen


@lru_cache(maxsize=1)
def _test_map():
    """{test file name: set(modules it reaches)} for every tests/test_*.py."""
    direct, test_edges = {}, {}
    for p in sorted(TESTS.glob("*.py")):
        mods, tests = _refs(p, "tests")
        text = p.read_text(encoding="utf-8", errors="replace")
        if _CLI_DRIVE.search(text):
            mods |= {"manamap.pilot.registry", "manamap.cli"}
        direct[p.stem] = {m for m in mods if m in _graph()}
        test_edges[p.stem] = tests
    out = {}
    for stem in direct:
        if not stem.startswith("test_"):
            continue
        mods, seen, queue = set(direct[stem]), {stem}, list(test_edges[stem])
        while queue:                                   # a test module imported by another
            t = queue.pop()
            # THE SHARED HARNESS DOES NOT PROPAGATE. `from conftest import requires_corpus`
            # made a test inherit every module conftest reaches (~200) and a bracket edit
            # selected 275 of 311 files. A harness edit runs the whole unit tier anyway;
            # a module reached ONLY through a conftest fixture is the documented gap the
            # nightly full run covers.
            if t in seen or t not in direct or not t.startswith("test_"):
                continue
            seen.add(t)
            mods |= direct[t]
            queue.extend(test_edges.get(t, ()))
        out[f"{stem}.py"] = closure(mods)
    return out


MAP_PATH = REPO / ".pytest_cache" / "manamap" / "impact-map.json"
#: Source files whose effect is not visible in any test's measured lines — constants read
#: at import by every producer. A change here runs the whole tier.
WHOLE_TIER = {"src/manamap/config.py", "src/manamap/__init__.py"}


def build_map(coverage_file, out=MAP_PATH):
    """A coverage data file recorded with `--cov-context=test` -> {src path: [test files]}."""
    import json
    import subprocess
    import coverage
    data = coverage.CoverageData(basename=str(coverage_file))
    data.read()
    files = {}
    for f in data.measured_files():
        rel = "src/" + f.split("/src/", 1)[1] if "/src/" in f else f
        tests = sorted({c.split("::", 1)[0] for cs in data.contexts_by_lineno(f).values()
                        for c in cs if c and "::" in c})
        if tests:
            files[rel] = tests
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    doc = {"built_at_commit": head, "contexts": len(data.measured_contexts()), "files": files}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=0, sort_keys=True) + "\n", encoding="utf-8")
    return doc


def load_map(path=MAP_PATH):
    import json
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def affected_tests(paths, measured="auto"):
    """The test files (`tests/test_*.py`) a set of changed repo paths can reach, sorted,
    or None when a path cannot be placed and the whole tier must run."""
    tmap = None
    mmap = load_map() if measured == "auto" else measured
    mfiles = (mmap or {}).get("files") or {}
    changed_mods, changed_tests, measured_hit = set(), set(), set()
    for raw in paths:
        p = str(raw)
        if p in WHOLE_TIER:
            return None
        if p.startswith("src/manamap/") and p in mfiles:
            measured_hit.update(Path(t).name for t in mfiles[p])
            continue
        if p.startswith("src/manamap/"):
            m = module_of(p)
            if m is None:
                return None                             # a non-Python file under the package
            changed_mods.add(m)
        elif p.startswith("tests/"):
            name = Path(p).name
            if name.startswith("test_") and name.endswith(".py"):
                changed_tests.add(name)
            elif Path(p).stem in _TEST_HELPERS or p.startswith("tests/fixtures"):
                return None                             # a shared helper: every test may see it
            else:
                return None
        else:
            return None
    # A test that imports a changed TEST module runs too.
    importers = set()
    if changed_tests:
        stems = {Path(t).stem for t in changed_tests}
        for p in TESTS.glob("test_*.py"):
            _, tests = _refs(p, "tests")
            if tests & stems:
                importers.add(p.name)
    hit = set()
    if changed_mods:                                    # files the measured map has never seen
        tmap = _test_map()
        hit = {t for t, mods in tmap.items() if mods & changed_mods}
    return sorted(f"tests/{t}" for t in hit | measured_hit | changed_tests | importers
                  if (TESTS / t).exists())


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--build"]:
        doc = build_map(argv[1])
        print(f"wrote {MAP_PATH.relative_to(REPO)}: {len(doc['files'])} source files, "
              f"{doc['contexts']} test contexts, at {doc['built_at_commit'][:10]}")
        return 0
    got = affected_tests(argv)
    if got is None:
        print("ALL — a changed path cannot be placed; run the whole tier")
        return 0
    print("\n".join(got) if got else "(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
