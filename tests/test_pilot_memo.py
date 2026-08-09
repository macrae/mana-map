"""`mtime_memo`: one memo discipline, and the bug that motivated extracting it.

Five modules hand-rolled the same "parse this artifact once per process" cache
and two got it wrong in the same way — gating on truthiness (`if not _MEMO`)
rather than on the file's (mtime_ns, size). A truthiness gate is correct for a
CLI process that exits in seconds and wrong everywhere else: in pytest one
process regenerates `cards.csv` and every later assertion in that session is
answered from the pre-edit copy, silently, looking like a stale fixture.

These tests pin the discipline rather than any one caller, because the next
hand-rolled memo is the one that will get it wrong again.
"""

import json

import pytest

from manamap.pilot.common import _MTIME_MEMO, clear_memo, mtime_memo


@pytest.fixture(autouse=True)
def _clean():
    clear_memo()
    yield
    clear_memo()


def test_a_second_call_does_not_rebuild(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text('{"v": 1}')
    calls = []

    def build():
        calls.append(1)
        return json.loads(path.read_text())

    assert mtime_memo(path, "t", build) == {"v": 1}
    assert mtime_memo(path, "t", build) == {"v": 1}
    assert len(calls) == 1, "the whole point is one parse per process"


def test_a_rewrite_is_noticed(tmp_path):
    """The defect the two unkeyed memos had.

    `if not _MEMO: build()` passes this file's first assertion and fails this
    one — it keeps serving the pre-edit value for the life of the process.
    """
    path = tmp_path / "artifact.json"
    path.write_text('{"v": 1}')
    build = lambda: json.loads(path.read_text())          # noqa: E731

    assert mtime_memo(path, "t", build) == {"v": 1}
    # A rewrite of a DIFFERENT size — the cheap signal. Same-size rewrites are
    # caught by mtime_ns, which is why the key is a pair.
    path.write_text('{"v": 22}')
    assert mtime_memo(path, "t", build) == {"v": 22}


def test_two_builders_over_one_file_do_not_collide(tmp_path):
    """`cards.csv` legitimately backs both a name set and an oracle map.

    This is why the key is explicit rather than derived from the build
    function: `__qualname__` is `<lambda>` for both of these.
    """
    path = tmp_path / "cards.csv"
    path.write_text("name\nSol Ring\n")
    names = mtime_memo(path, "names", lambda: {"Sol Ring"})
    oracle = mtime_memo(path, "oracle", lambda: {"Sol Ring": "{T}: Add {C}{C}."})
    assert names == {"Sol Ring"}
    assert oracle == {"Sol Ring": "{T}: Add {C}{C}."}


def test_a_missing_file_returns_absent_and_caches_nothing(tmp_path):
    """A fresh clone that later generates the artifact must pick it up.

    Caching the absence would mean a process that ran before `manamap extract`
    keeps answering "no corpus" after the corpus exists.
    """
    path = tmp_path / "not-yet.json"
    assert mtime_memo(path, "t", lambda: "built", absent={}) == {}
    assert not _MTIME_MEMO
    path.write_text("{}")
    assert mtime_memo(path, "t", lambda: "built", absent={}) == "built"


def test_clear_memo_drops_mtime_entries(tmp_path):
    """Registered with the shared teardown, not a private atexit hook.

    Each hand-rolled copy had to remember its own `atexit.register(...clear)`;
    two of the five did. One registry means a new memo cannot forget.
    """
    path = tmp_path / "a.json"
    path.write_text("{}")
    mtime_memo(path, "t", lambda: "x")
    assert _MTIME_MEMO
    clear_memo()
    assert not _MTIME_MEMO


def test_the_real_corpus_readers_are_all_keyed():
    """The four converted callers, by the key they register under.

    A regression here means someone re-hand-rolled a memo; the names are
    module-scoped so two readers of one file stay distinct.
    """
    import inspect

    from manamap.pilot import bracket, pool_facts, validate_diagnosis, validate_stack
    for module, key in ((bracket, "bracket:flags"),
                        (pool_facts, "pool_facts:cards"),
                        (validate_stack, "validate_stack:names"),
                        (validate_diagnosis, "validate_diagnosis:oracle")):
        source = inspect.getsource(module)
        assert f'"{key}"' in source, (
            f"{module.__name__} no longer registers {key} — if it grew its own "
            f"memo again, key it on (mtime_ns, size) via mtime_memo")
