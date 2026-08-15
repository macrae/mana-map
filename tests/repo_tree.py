"""One pruned walk of the repository, shared by the documentation guards.

Two doc tests needed to answer "does a file with this bare name exist anywhere in
the repo", and both answered it by globbing the whole tree per question:

    ROOT.glob(f"**/{ref}")          # test_docs_section_count.py, 179 times
    ROOT.rglob(name)                # test_docs_counts.py, 7 times

The tree is 38,669 files and **37,653 of them are inside `.venv`** — 1.4 GB of
site-packages that neither test could ever want. The first call site then threw
the `.venv` hits away with a substring check, having already paid to find them.
Together that was ~6.9 million directory operations per test run and **8.7
seconds**, which made the documentation guards the second most expensive file in
the fast suite after 90,000 goldfish simulations.

The fix is not a cache, it is not walking the wrong thing: one pruned walk,
memoised at module scope, and both questions become dict lookups. Measured on
this machine: 8.71 s -> 0.09 s, with both tests asserting exactly what they
asserted before.

`PRUNED` is deliberately a small closed set rather than "anything gitignored".
Reading `.gitignore` would exclude `data/`'s tracked exceptions and, worse, would
make the guards' answer depend on a file the guards are supposed to be
independent of.
"""

import functools
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directory names never descended into. `data/` is here because it holds ~150 MB
# of generated artifacts and no source file a doc would name in backticks; if a
# doc ever needs to claim `data/foo.py` exists, it will carry a slash and take
# `(ROOT / ref).exists()` instead of this index.
PRUNED = {".venv", ".git", "node_modules", "__pycache__", ".pytest_cache",
          ".ruff_cache", ".mypy_cache", "data", "venv", "env", ".tox"}


@functools.lru_cache(maxsize=1)
def _index():
    """`{basename: (relative paths,)}` for every file outside PRUNED."""
    found = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in PRUNED]
        base = Path(dirpath)
        for name in filenames:
            found.setdefault(name, []).append(
                (base / name).relative_to(ROOT).as_posix())
    return {k: tuple(v) for k, v in found.items()}


def paths_named(basename):
    """Every repo-relative path whose filename is `basename`, `.venv` excluded."""
    return _index().get(basename, ())


def exists_anywhere(basename):
    """Is there a file with this bare name somewhere in the source tree?"""
    return bool(paths_named(basename))
