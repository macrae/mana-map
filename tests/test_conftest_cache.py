"""The regenerate-and-compare cache's own seams.

`tests/conftest.py` holds a cache that skips a test when every input it names is
byte-identical to the last passing run. It is the reason `make test` is five
minutes rather than nine, and it is also the one place in this repo where a
mistake can make a board green — so its seams get tests like anything else.
"""

import pytest

from conftest import _TREE_MEMO, _cache_of, _digest, requires_data


class _NoCacheConfig:
    """A `Config` as it looks under `-p no:cacheprovider`: no `cache` at all."""


class _WithCache:
    cache = "a real cache object"


def test_the_accessor_survives_a_missing_cacheprovider():
    """`-p no:cacheprovider` used to kill every `unchanged` case.

    The guard existed — `if cache is not None` — and sat one line BELOW
    `cache = config.cache`, which is the line that raised. Collection worked, so
    the breakage only appeared when a test actually ran.

    This is testable from inside pytest where `-p no:cacheprovider` is not: the
    plugin is loaded for this very session, so the only way to exercise the
    absence is to hand the accessor a config that lacks the attribute.
    """
    assert _cache_of(_NoCacheConfig()) is None
    assert _cache_of(_WithCache()) == "a real cache object"


def test_the_digest_changes_when_a_named_file_changes(tmp_path):
    """The whole cache rests on this: same inputs, same key; any input moves and
    the key moves with it."""
    one, two = tmp_path / "a.txt", tmp_path / "b.txt"
    one.write_text("alpha", encoding="utf-8")
    two.write_text("beta", encoding="utf-8")

    before = _digest([one, two])
    assert _digest([one, two]) == before, "the digest is not stable"

    two.write_text("beta!", encoding="utf-8")
    assert _digest([one, two]) != before, "a changed file did not move the key"


def test_the_digest_covers_a_whole_directory(tmp_path):
    """Directories are how callers stay honest rather than clever — naming a
    tree cannot miss a file inside it."""
    tree = tmp_path / "deck"
    (tree / "branches" / "x").mkdir(parents=True)
    (tree / "cards.json").write_text("{}", encoding="utf-8")
    before = _digest([tree])

    # A file added DEEP in the tree must move the key. This is the property a
    # per-deck key relies on — and the one that says nothing whatever about
    # ANOTHER deck's tree, which is #49.
    (tree / "branches" / "x" / "net_change.json").write_text("{}", encoding="utf-8")
    # `_tree_digest` memoises per run WITHOUT a file signature, on the stated
    # assumption that a named tree does not change mid-run. Clearing it is how
    # this test asks the function its question rather than the memo's.
    _TREE_MEMO.clear()
    assert _digest([tree]) != before, "a new file in a named tree did not move the key"


def test_an_absent_file_is_a_stable_key_rather_than_an_error(tmp_path):
    """A named input that does not exist yet must not raise — a fresh clone has
    none of the generated artifacts."""
    missing = tmp_path / "not-here.json"
    assert _digest([missing]) == _digest([missing])
    missing.write_text("now it is", encoding="utf-8")
    assert _digest([missing]) != _digest([tmp_path / "still-not-here.json"])


# ── Process-global caches, and the one function that drops them ─────────────

def test_every_process_cache_is_cleared_by_clear_memo():
    """A CACHE `clear_memo` DOES NOT KNOW ABOUT IS A LEAK ACROSS TESTS.

    It said it "mirrors `_SHA_MEMO.clear()`" and did not clear `_SHA_MEMO`.
    Three caches outlived it — `agent_cache._SHA_MEMO`, `card_refs`'s
    `lru_cache` and `goldfish._CREATURE_TYPES_CACHE` — and the tell was three
    test files reaching into `agent_cache` to clear it by hand, a workaround for
    the function whose job is to make that unnecessary.

    IT POPULATES EVERY CACHE IT FINDS BEFORE CLEARING. The first version of this
    only asserted the caches were empty afterwards, and passed with the bug
    deliberately re-introduced — because nothing in that process had put
    anything in `_SHA_MEMO`. A test that cannot fail when the defect is present
    is testing itself. Planting a sentinel in each one is what makes the
    assertion mean anything.

    DERIVED, not a list: it walks the package, so adding a cache fails this
    until `clear_memo` learns it. A cache that survives a `monkeypatch`
    teardown answers a later test with an earlier test's tree (#31) — quietly,
    and in an order that changes under `-n auto`.
    """
    import importlib
    import pkgutil

    import manamap.pilot
    from manamap.pilot import common

    found = []
    for info in pkgutil.iter_modules(manamap.pilot.__path__):
        module = importlib.import_module(f"manamap.pilot.{info.name}")
        for name in dir(module):
            if not (name.endswith("_MEMO") or name.endswith("_CACHE")):
                continue
            value = getattr(module, name)
            if isinstance(value, dict):
                found.append((f"{info.name}.{name}", value))

    assert len(found) >= 5, (
        f"only {len(found)} module-level dict caches found — did they move, or "
        f"did the naming convention change?")

    for _label, cache in found:
        cache["__planted_by_the_test__"] = "a value from a previous test's tree"

    common.clear_memo()

    leaked = [label for label, cache in found if cache]
    assert not leaked, (
        "clear_memo() left a process cache populated:\n  " + "\n  ".join(leaked)
        + "\n(add it to common.clear_memo)")


def test_clear_memo_resets_the_sentinel_caches_too():
    """Two caches are not dicts and cannot be found by the sweep above.

    `goldfish._CREATURE_TYPES_CACHE` is a SENTINEL — `_UNSET` means "not looked
    up", and `None` is a real answer (no corpus behind it). Clearing it to
    `None` would pin the fallback for the rest of the process rather than
    dropping the memo. `card_refs.ambiguous_tokens` is an `lru_cache`.
    """
    from manamap.pilot import card_refs, common, goldfish

    goldfish._CREATURE_TYPES_CACHE = {"Vampire"}
    common.clear_memo()
    assert goldfish._CREATURE_TYPES_CACHE is goldfish._UNSET, (
        "cleared to a real value rather than to 'not looked up yet'")
    assert card_refs.ambiguous_tokens.cache_info().currsize == 0


# ── The import closure: the two controls that make it safe to narrow ────────

def test_the_closure_covers_every_module_a_real_producer_imports():
    """CONTROL 1, AND THE ONE THAT DECIDES WHETHER NARROWING IS SAFE AT ALL.

    The cache key for the freshness tests used to be the whole source tree,
    because the version before it hand-traced the inputs, asserted the list was
    "checked rather than assumed", and was wrong in NINE modules across three
    subpackages. A missed edge does not go red — it serves a stale PASS.

    So the closure is computed from the syntax tree, and this holds it to
    reality: RUN the producers, then assert every `manamap.*` file that actually
    got imported is in the closure. It catches `importlib`, `__import__` and
    anything else a parser cannot see.

    IT RUNS IN A CLEAN INTERPRETER. The first version read `sys.modules` in the
    pytest process and failed — not because the closure was wrong, but because
    by then a hundred other test modules had imported half the package, so it
    was measuring the session rather than the producers. A control contaminated
    by everything that ran before it cannot answer this question.

    If this ever fails for real, the honest move is to widen the key back to
    `SRC`, not to add the missing module by hand.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    from conftest import ROOT, SRC, module_closure
    from manamap.pilot import (bracket, deck_info, diagnostic, goldfish,
                               mana_analysis, net_change)

    closure = set(module_closure(bracket, deck_info, diagnostic, goldfish,
                                 mana_analysis, net_change))

    probe = r"""
import argparse, contextlib, io, json, sys
from manamap.pilot import deck_info, goldfish, mana_analysis
with contextlib.redirect_stdout(io.StringIO()):
    goldfish.run("goblin-storm", iterations=20, seed=1)
    mana_analysis.main(argparse.Namespace(slug="goblin-storm", write=False,
                                          as_json=False, branch=None, out=None))
    deck_info.compose("goblin-storm", verify=False)
files = sorted({getattr(m, "__file__", "") or ""
                for n, m in sys.modules.items() if n.startswith("manamap")})
print(json.dumps([f for f in files if f]))
"""
    done = subprocess.run([sys.executable, "-c", probe], cwd=ROOT,
                          capture_output=True, text=True)
    assert done.returncode == 0, f"the probe failed:\n{done.stderr[-2000:]}"
    imported = {Path(f) for f in json.loads(done.stdout.splitlines()[-1])
                if str(f).startswith(str(SRC))}

    assert len(imported) >= 20, (
        f"only {len(imported)} manamap modules were imported — did the "
        f"producers stop running?")
    missing = sorted(str(p.relative_to(SRC)) for p in imported - closure)
    assert not missing, (
        "the AST closure MISSES modules a real run imports, so the cache could "
        "serve a stale pass:\n  " + "\n  ".join(missing)
        + "\n(widen the key back to SRC rather than adding these by hand)")


def test_the_closure_actually_narrows_and_excludes_what_it_should():
    """CONTROL 3. A closure that covers everything is `SRC` with extra steps."""
    from conftest import SRC, module_closure
    from manamap.pilot import goldfish

    closure = set(module_closure(goldfish))
    everything = set(SRC.rglob("*.py"))
    assert closure < everything, "the closure is not a strict subset of the tree"
    assert len(closure) < len(everything) * 0.6, (
        f"the closure is {len(closure)} of {len(everything)} files — barely a "
        f"narrowing, so the added machinery is not paying for itself")

    # Things a goldfish run provably cannot reach. Each is a whole subtree that
    # used to re-run 279 heavy cases when anyone touched it.
    for never in ("sven/loop.py", "training/train_vae.py", "export/reduce.py",
                  "analysis/eval_embeddings.py"):
        assert SRC / never not in closure, f"{never} should not be in a goldfish closure"


def test_the_closure_sees_an_import_written_inside_a_function():
    """The lazy-import idiom is everywhere in this package — `pipeline.STEPS`,
    `registry`'s dispatch, ~40 function-local `from manamap.config import …` —
    so a closure that only read top-level imports would miss most of the graph
    while looking complete."""
    from conftest import _imports_in

    src = '''
def later():
    from manamap.pilot import deck_branch
    import manamap.sim.forge
    return deck_branch, manamap.sim.forge
'''
    probe = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "probe.py"
    probe.write_text(src, encoding="utf-8")
    found = _imports_in(probe, "manamap.pilot.probe")
    assert "manamap.pilot.deck_branch" in found, "a function-level import was missed"
    assert "manamap.sim.forge" in found, "a function-level plain import was missed"


@requires_data
def test_touching_a_dependency_moves_the_key_and_touching_a_stranger_does_not():
    """CONTROL 2 — the bug, re-introduced, in BOTH directions.

    A narrowed key is only worth having if it still invalidates on everything
    the producer reads. Asserting that alone would pass for a key that
    invalidates on everything, so the second half is the real one: a module the
    producer provably cannot reach must NOT move it.

    Measured 2026-09-12 on `test_pilot_artifact_freshness.CODE`:

        pilot/common.py           moves it   (a transitive dependency)
        config.py                 moves it   (always in the closure)
        sven/llm.py               does not   (nothing reaches it)
        training/train_vae.py     does not
        pilot/build_manual.py     does not   (the frozen magazine renderer)
    """
    import test_pilot_artifact_freshness as freshness
    from conftest import _FILE_MEMO, _TREE_MEMO, ROOT, _digest

    def key():
        # The digests memoise per run; this asks the files, not the memo.
        _TREE_MEMO.clear()
        _FILE_MEMO.clear()
        return _digest(freshness.CODE)

    base = key()
    cases = {"src/manamap/pilot/common.py": True,
             "src/manamap/config.py": True,
             "src/manamap/sven/llm.py": False,
             "src/manamap/training/train_vae.py": False,
             "src/manamap/pilot/build_manual.py": False}
    wrong, checked = [], 0
    for name, should_move in cases.items():
        path = ROOT / name
        if not path.exists():                    # a module may legitimately go
            continue
        checked += 1
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n# touched by a test\n")
            moved = key() != base
        finally:
            path.write_bytes(original)
        if moved != should_move:
            wrong.append(f"{name}: expected the key to "
                         f"{'move' if should_move else 'hold'}, it "
                         f"{'moved' if moved else 'held'}")
    assert checked >= 4, f"only {checked} probes ran"
    assert not wrong, "the cache key is wrong about its inputs:\n  " + "\n  ".join(wrong)
    assert key() == base, "a probe did not restore its file"
