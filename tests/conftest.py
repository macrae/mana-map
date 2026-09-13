"""Shared pytest fixtures and markers for the mana-map test suite."""

import contextlib
import functools
import hashlib
import http.server
import socket
import threading
from pathlib import Path

import pytest


from manamap import config

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "manamap"

# Marker for tests that need generated data/ artifacts (run the pipeline first).
# Gate on embeddings.npy — the last artifact of the train/embed stage — so a
# partially-populated data/ dir still skips cleanly.
requires_data = pytest.mark.skipif(
    not config.EMBEDDINGS_PATH.exists(),
    reason="requires generated data/ artifacts (run `manamap run` first)",
)

# Pilot subsystem artifact gates (same pattern).
requires_rules = pytest.mark.skipif(
    not config.RULES_INDEX_PATH.exists(),
    reason="requires the rules DB (run `manamap pilot download-rules && manamap pilot build-rules-db`)",
)

requires_deck = pytest.mark.skipif(
    not (config.DECKS_DIR / "goblin-storm" / "cards.json").exists(),
    reason="requires a fetched deck (run `manamap pilot fetch-deck goblin-storm`)",
)

requires_strategy = pytest.mark.skipif(
    not config.STRATEGY_INDEX_PATH.exists(),
    reason="requires the strategy DB (run `manamap pilot build-strategy-db`)",
)

def _a_branch(slug):
    """The name of any branch on this deck, or None.

    A BRANCH IS A PILOT ARTIFACT, NOT A FIXTURE. Sixteen tests hardcoded
    `ur-dragon/treasure-v2` and all sixteen failed the day the pilot deleted it
    — correctly deleted, the treasure refactor was measured and abandoned. A
    suite that requires one particular candidate 99 to exist forever is a suite
    that punishes the pilot for using the tool. These tests need A branch, not
    THAT branch, so they take whichever one is there and skip when there is
    none.
    """
    root = config.DECKS_DIR / slug / "branches"
    if not root.is_dir():
        return None
    got = sorted(d.name for d in root.iterdir()
                 if (d / "decklist.txt").exists())
    if not got:
        return None

    # A MERGED BRANCH IS NOT A BRANCH-SHAPED FIXTURE. Once merged, its list IS
    # the deck's list, so its diff is +0 -0 by definition — and the tests that
    # take "whichever branch exists" are all about the DIFF: that both sides
    # balance in copies, that every row carries its count, that the bill and the
    # diff cannot disagree. Handed a merged branch they assert those things
    # about nothing and fail on an empty set.
    #
    # Found when `eminence-v3` merged as v1.0.2 and sorted first alphabetically.
    def merged(name):
        doc = config.DECKS_DIR / slug / "branches" / name / "branch.json"
        try:
            import json
            return bool(json.loads(doc.read_text()).get("merged"))
        except Exception:                       # noqa: BLE001 - absent or unreadable
            return False

    live = [name for name in got if not merged(name)]

    def staged(name):
        doc = config.DECKS_DIR / slug / "branches" / name / "branch.json"
        try:
            import json
            return len(json.loads(doc.read_text()).get("staged") or [])
        except Exception:                       # noqa: BLE001
            return 0

    # THE BRANCH WITH THE MOST STAGED SWAPS. The first name alphabetically was
    # final-v1 (2026-09-11), whose one staged swap puts a card BACK that the
    # deck already runs, so the pairing tests had nothing to pair.
    return max(live or got, key=staged)


#: The deck the branch-shaped tests measure against, and any branch it has.
BRANCH_SLUG = "ur-dragon"
A_BRANCH = _a_branch(BRANCH_SLUG)

requires_branch = pytest.mark.skipif(
    A_BRANCH is None,
    reason=(f"requires a branch on {BRANCH_SLUG} "
            f"(`manamap pilot deck-branch {BRANCH_SLUG} new <name> --from <file> "
            f"--objective \"<measure> <op> <number>\"`)"),
)

requires_roles = pytest.mark.skipif(
    not config.CARD_ROLES_PATH.exists(),
    reason="requires the role index (run `manamap card-roles`)",
)


@pytest.fixture(scope="session")
def data_dir():
    """The resolved data directory (honors MANAMAP_DATA_DIR)."""
    return config.DATA_DIR


# ── The regenerate-and-compare cache ────────────────────────────────────
#
# Four test files do the same thing: recompute an artifact from files in this
# repo and assert it matches the tracked copy. `test_pilot_artifact_freshness`
# alone re-runs 90,000 seeded goldfish games — 20.5 s, every run, to re-derive
# nine files that only change when someone edits a decklist or the simulator.
# That work is a PURE FUNCTION of files on disk, so if no input moved, the answer
# cannot have.
#
# **This repo's standing rule is: never record a pass to make a board green.** A
# test cache is that rule's exact shape one level down, so five properties keep it
# from becoming somewhere false confidence can live:
#
#   1. The key covers the SOURCE of the code under test, not only its data. Edit
#      `goldfish.py` and all nine deck cases re-run.
#   2. Recorded only on PASS (`pytest_runtest_makereport` below). A failing test
#      never seeds the cache, so a red run cannot be cached green.
#   3. Stored in `.pytest_cache/`, which is gitignored. It cannot be committed,
#      pushed or pulled — a hit can never travel to another machine, and CI checks
#      out fresh and therefore runs everything with no flag required.
#   4. `--no-test-cache` forces full execution. Use it before trusting a release.
#   5. Hits are COUNTED AND PRINTED. A twelve-second run that says nothing about
#      what it skipped is the failure mode; this one says `27 cached`.
#
# It deliberately does NOT try to discover inputs automatically. A test names the
# files it depends on, because a wrong automatic answer here is invisible.

_UNCHANGED_KEYS = {}                 # nodeid -> key, pending a passing result
_CACHE_PREFIX = "manamap/unchanged"
CACHED_SKIP = "cached: unchanged since the last passing run"


def pytest_addoption(parser):
    parser.addoption(
        "--no-test-cache", action="store_true", default=False,
        help="ignore the regenerate-and-compare cache; run every test for real.")


# Directory contents that can never change a deterministic artifact, and that
# change constantly. `.agent-out/` is gitignored agent scratch; `.agent-cache.json`
# moves whenever anyone runs `cache-record`.
_DIGEST_SKIP_DIRS = {"__pycache__", ".agent-out", ".pytest_cache"}
_DIGEST_SKIP_FILES = {".agent-cache.json", ".DS_Store"}
_TREE_MEMO = {}
_FILE_MEMO = {}


def _file_digest(path):
    """sha256 of one file, memoised on `(mtime_ns, size)` for the run.

    Necessary rather than tidy: the validator cases name `synergy_graph.json`
    (27 MB) and `combo_details.json` (25 MB), and 63 parametrized cases each
    hashing 60 MB would cost more than the tests they are meant to skip. Same
    `(mtime_ns, size)` discipline as `pilot/common.load_json_memo`.
    """
    try:
        stat = path.stat()
    except OSError:
        return "absent"
    sig = (stat.st_mtime_ns, stat.st_size)
    hit = _FILE_MEMO.get(path)
    if hit is not None and hit[0] == sig:
        return hit[1]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _FILE_MEMO[path] = (sig, digest)
    return digest


def _tree_digest(directory):
    """A key over every file under `directory`, memoised per run.

    THE MEMO HAS NO SIGNATURE CHECK, unlike `_file_digest`'s `(mtime_ns, size)`,
    and that is an assumption rather than an oversight: **a tree named as an
    input must not change during a run.** It holds today because the only tests
    that rewrite tracked artifacts (`test_pilot_artifact_freshness._roundtrip`)
    restore them from a backup before the next case reads them, and because
    `unchanged()` takes its key at the START of a test, before any rewrite.

    If a test ever leaves a named tree changed, every later digest of that tree
    in the same process is stale and the cache records a key for a state that no
    longer exists. Clear `_TREE_MEMO` if you write such a test.
    """
    hit = _TREE_MEMO.get(directory)
    if hit is not None:
        return hit
    sha = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if set(path.parts) & _DIGEST_SKIP_DIRS or path.name in _DIGEST_SKIP_FILES:
            continue
        sha.update(path.relative_to(directory).as_posix().encode())
        sha.update(path.read_bytes())
    digest = sha.hexdigest()
    _TREE_MEMO[directory] = digest
    return digest


def _module_file(dotted):
    """`manamap.pilot.goldfish` -> its file, or None if it is not ours."""
    if not (dotted == "manamap" or dotted.startswith("manamap.")):
        return None
    rel = dotted.split(".")[1:]
    base = SRC.joinpath(*rel) if rel else SRC
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _imports_in(path, package):
    """Every `manamap.*` module this FILE imports, at any depth.

    `ast.walk`, not a scan of the module body: this package imports lazily
    almost everywhere — `pipeline.STEPS`, `registry`'s dispatch, and ~40
    function-local `from manamap.config import …` — and a closure that only saw
    top-level imports would miss most of the graph, which is the failure mode
    this whole mechanism has to avoid.
    """
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):              # pragma: no cover - defensive
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                      # `from . import x`, `from .x import y`
                parts = package.split(".")
                base = ".".join(parts[:len(parts) - node.level + 1])
                dotted = f"{base}.{node.module}" if node.module else base
            else:
                dotted = node.module or ""
            out.add(dotted)
            out.update(f"{dotted}.{alias.name}" for alias in node.names)
    return {d for d in out if d == "manamap" or d.startswith("manamap.")}


def module_closure(*modules):
    """Every file under `src/manamap/` that these modules can reach, transitively.

    WHY THIS EXISTS. Three test files keyed the regenerate-and-compare cache on
    `SRC` — the whole source tree — so an edit anywhere under `src/manamap/`
    re-ran 279 heavy freshness cases, most of them 10,000-game goldfish runs.

    WHY IT IS DERIVED AND NOT A LIST. The version before `SRC` DID name the
    inputs by hand, and its comment asserted the closure was "checked rather
    than assumed". It was wrong in NINE modules across three subpackages, and
    the failure of a missed edge is not a red test — it is a stale PASS. So the
    closure is computed from the syntax tree, and two controls in
    `tests/test_conftest_cache.py` hold it to reality: one asserts it covers
    every `manamap.*` module a real producer run actually imports, and one
    re-introduces the bug by touching a transitive dependency.

    `config.py` is always included: it is the one module every producer reads
    and the one whose constants change behaviour without an import edge moving.
    """
    seen, queue = set(), []
    for module in modules:
        path = Path(getattr(module, "__file__", "") or "")
        if path.is_file():
            queue.append((path, getattr(module, "__name__", "manamap")))
    always = _module_file("manamap.config")
    if always:
        queue.append((always, "manamap.config"))

    while queue:
        path, package = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for dotted in _imports_in(path, package):
            found = _module_file(dotted)
            if found and found not in seen:
                queue.append((found, dotted))
    return tuple(sorted(seen))


def _digest(paths):
    """One key over every named file, or every file under a named directory.

    Directories are accepted, and are how the callers stay HONEST rather than
    clever. Naming a producer's exact inputs means tracing its transitive imports
    by hand, and a missed edge does not fail — it silently serves a stale pass,
    which is the one outcome this cache must never have. `src/manamap/pilot` in
    one argument over-invalidates (any pilot edit re-runs every freshness case)
    and cannot be wrong. Over-invalidation costs 20 s during pilot work, which is
    exactly when you want those tests running anyway.
    """
    sha = hashlib.sha256()
    for path in paths:
        path = Path(path)
        sha.update(str(path).encode())
        if path.is_dir():
            sha.update(_tree_digest(path).encode())
        else:
            sha.update(_file_digest(path).encode())
    return sha.hexdigest()


def _cache_key(nodeid):
    return f"{_CACHE_PREFIX}/{nodeid.replace('/', '__').replace('::', '--')}"


def simulator_source():
    """The goldfish's source, ALL FOUR FILES, concatenated.

    Several tests assert on the simulator's source TEXT — that a channel is
    reachable from a casting loop, that a trigger set matches what the turn loop
    branches on. That is a deliberate pattern here: it is how a test proves a
    flag is acted on rather than merely set, which is a class of bug this repo
    has shipped five times.

    They read `pilot/goldfish.py`. The 2026-09-13 split moved the card readers
    and the turn loop into their own modules, so those greps found nothing and
    every one of them went green-by-absence — an assertion looking for an idiom
    in a file that no longer contains any idioms at all.

    One reader, derived from `goldfish._MODEL_FILES`, so a fifth module joins
    automatically.
    """
    from manamap.pilot import goldfish

    here = Path(goldfish.__file__).parent
    return "\n".join((here / name).read_text(encoding="utf-8")
                      for name in sorted(goldfish._MODEL_FILES))


def patch_model(monkeypatch, name, value):
    """Patch a simulator name EVERYWHERE it is bound, and assert it landed.

    THE SPLIT MADE A RE-EXPORT NOT A PATCH POINT. `goldfish.py` re-exports the
    card readers so every caller that reads `goldfish.draw_profile` keeps
    working — but `classify` resolves `draw_profile` in `goldfish_library`'s
    OWN namespace, so `goldfish.draw_profile = blind` changed a name nothing
    called. The affected tests did not error; they compared a channel against
    itself and reported "3.708 against 3.708 blind", which reads exactly like
    the channel not firing.

    Same defect as the deck root in #31, one layer in: a `from`-import is a
    COPY, and which modules hold a copy is an implementation detail a test
    should not have to track. This sets the name on every simulator module that
    has it and FAILS if that is none — so a rename cannot make a patch quietly
    do nothing.
    """
    import importlib

    from manamap.pilot import goldfish

    landed = []
    for file_name in goldfish._MODEL_FILES:
        module = importlib.import_module(
            "manamap.pilot." + file_name[:-len(".py")])
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)
            landed.append(module.__name__.rsplit(".", 1)[-1])
    assert landed, (
        f"{name!r} is bound in no simulator module — the patch would have done "
        f"nothing, which is how a channel comes to be compared against itself")
    return landed


def _cache_of(config):
    """pytest's cache, or None when the cacheprovider plugin is not loaded.

    `getattr`, not `config.cache` plus a `None` check — the check WAS there and
    it was one line too late. With `-p no:cacheprovider` the attribute does not
    exist at all, so reading it raised `AttributeError: 'Config' object has no
    attribute 'cache'` before anything could look at the value, and every one of
    the ~500 `unchanged` cases died on collection-clean, run-broken. Running
    without the plugin is a normal thing to want (a fresh, no-side-effects run),
    and `make test-fresh` should not be the only way to get it.

    No cache means no hit and no record, which is exactly `--no-test-cache`.
    """
    return getattr(config, "cache", None)


@pytest.fixture
def unchanged(request):
    """Skip when every named input is byte-identical to the last passing run.

    Call it FIRST in a test, with every file the test's answer depends on —
    the inputs, the expected artifact, and the module that produces it:

        unchanged(deck / "cards.json", SRC / "pilot/goldfish.py",
                  SRC / "config.py", deck / "goldfish_metrics.json")

    Naming too many files costs a needless re-run. Naming too few is the real
    error and the reason this is explicit: the test would then pass from cache
    after a change it was written to catch.
    """
    def check(*paths):
        key = _digest(paths)
        if request.config.getoption("--no-test-cache"):
            return
        cache = _cache_of(request.config)
        if cache is not None and cache.get(_cache_key(request.node.nodeid), None) == key:
            pytest.skip(CACHED_SKIP)
        _UNCHANGED_KEYS[request.node.nodeid] = key
    return check


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Record a key only once the test has actually passed."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    key = _UNCHANGED_KEYS.pop(item.nodeid, None)
    if key is None or not report.passed:
        return
    cache = _cache_of(item.config)
    if cache is not None:
        cache.set(_cache_key(item.nodeid), key)


def pytest_terminal_summary(terminalreporter):
    """Say how many tests were served from the cache — never let it be silent."""
    cached = sum(1 for rep in terminalreporter.stats.get("skipped", [])
                 if CACHED_SKIP in str(getattr(rep, "longrepr", "")))
    if cached:
        terminalreporter.write_line(
            f"{cached} test(s) served from the regenerate-and-compare cache "
            f"(unchanged inputs). `--no-test-cache` runs them for real.")


# ── Browser fixtures ────────────────────────────────────────────────────
#
# Session-scoped and defined HERE rather than in conftest_viz.py, because a fixture
# imported into two test modules is registered twice — which meant two concurrent
# `sync_playwright()` contexts and a full-run-only failure that neither file showed alone.
# Playwright itself is imported lazily inside `browser`, so non-browser runs pay nothing.

def _free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D102 - silence per-request logging
        pass


@pytest.fixture(scope="session")
def viz_server() -> str:
    """Serve the repo root; yield the base URL."""
    port = _free_port()
    handler = functools.partial(_QuietHandler, directory=str(ROOT))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture(scope="session")
def browser():
    playwright = pytest.importorskip(
        "playwright.sync_api",
        reason="browser tests need playwright: pip install playwright && playwright install chromium",
    )
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            yield browser
        finally:
            browser.close()


# Discovery is the landing now, so every existing fixture asks for the map explicitly.
# A test about rendering 34,322 points should say so rather than rely on what boot
# happens to produce — and it documents the change for whoever reads these next.
EXPLORE = "?mode=explore"


