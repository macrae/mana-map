"""The rebuild's pool moves no byte: serial and parallel write the same deck.

`deck_edit.rebuild` runs the goldfish chain, deck-combos, diagnose and benchmark
side by side in a spawn pool (`REBUILD_JOBS`), on the argument that none reads
another's output and each seeds its own generator (`PARALLEL_STAGES`). The unit
tier proves the pool's ORDERING with a probe producer; this proves the CLAIM on a
real deck and the real corpus: meren-recursion copied twice into tmp data dirs,
rebuilt once with `REBUILD_JOBS=1` and once with the pool, every file in the
deck directory compared byte for byte.

Each rebuild runs in its own subprocess with `MANAMAP_DATA_DIR` at its tmp tree,
so the spawned workers (which re-import `config`) see the same tree as the
parent. The tree is the real data dir's entries SYMLINKED read-only, except
`decks/` (one copied deck + the manifest) and `cache/` (empty, so no write
reaches the real one). `fetch-deck` is stubbed to keep the deck's existing
`cards.json` — no network. The try warm is off: it writes only the cache.
The two subprocesses run concurrently, so the test costs one serial rebuild.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import manamap
from manamap import config
from manamap.pilot import deck_edit

SLUG = "meren-recursion"
#: Local state whose bytes name a pid or a time by design — not a rebuild output.
SKIP = {deck_edit.LOCK, deck_edit.REBUILD_LOCK}

DRIVER = """
import json, sys
from manamap.pilot import deck_edit, fetch_deck
fetch_deck.main = lambda args: None          # keep the deck's own cards.json
deck_edit.REBUILD_JOBS = int(sys.argv[2])
r = deck_edit.rebuild(sys.argv[1], warm=False)
print("RESULT " + json.dumps({"failures": r["failures"], "ran": r["ran"],
                               "behind": r.get("behind")}))
"""

pytestmark = [
    pytest.mark.regression,
    pytest.mark.skipif(not (config.DATA_DIR / "cards.csv").exists()
                       or not (config.DECKS_DIR / SLUG / "cards.json").exists(),
                       reason=f"needs the corpus and the tracked {SLUG} deck"),
]


def _tree(root, side):
    data = root / side / "data"
    data.mkdir(parents=True)
    for entry in config.DATA_DIR.iterdir():
        if entry.name in ("decks", "cache"):
            continue
        (data / entry.name).symlink_to(entry)
    (data / "cache").mkdir()
    decks = data / "decks"
    decks.mkdir()
    shutil.copytree(config.DECKS_DIR / SLUG, decks / SLUG, symlinks=False,
                    ignore=shutil.ignore_patterns(*SKIP))
    shutil.copy2(config.DECKS_DIR / "index.json", decks / "index.json")
    return data


def _files(deck):
    return {str(p.relative_to(deck)): p.read_bytes()
            for p in sorted(deck.rglob("*")) if p.is_file() and p.name not in SKIP}


def test_the_rebuild_pool_writes_the_serial_runs_bytes(tmp_path):
    # The subprocess imports the SAME package this test did (a worktree's src,
    # not whatever the venv's editable install points at).
    pkg_src = str(Path(manamap.__file__).resolve().parents[1])
    pypath = os.pathsep.join(p for p in (pkg_src, os.environ.get("PYTHONPATH")) if p)
    runs = {}
    for side, jobs in (("serial", 1), ("pool", deck_edit.REBUILD_JOBS)):
        data = _tree(tmp_path, side)
        env = dict(os.environ, MANAMAP_DATA_DIR=str(data), MANAMAP_NO_PROGRESS="1",
                   PYTHONPATH=pypath)
        runs[side] = (data, subprocess.Popen(
            [sys.executable, "-c", DRIVER, SLUG, str(jobs)], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    files = {}
    for side, (data, proc) in runs.items():
        out, err = proc.communicate(timeout=300)
        assert proc.returncode == 0, f"{side} rebuild failed:\n{err[-3000:]}"
        line = [ln for ln in out.splitlines() if ln.startswith("RESULT ")]
        assert line, f"{side}: no result\n{out[-2000:]}\n{err[-2000:]}"
        res = json.loads(line[-1][len("RESULT "):])
        assert res["failures"] == [] and not res["behind"], (side, res)
        assert "goldfish" in res["ran"], (side, res)
        files[side] = _files(data / "decks" / SLUG)

    assert len(files["serial"]) >= 10, sorted(files["serial"])
    assert sorted(files["serial"]) == sorted(files["pool"])
    differ = [n for n in files["serial"] if files["serial"][n] != files["pool"][n]]
    assert differ == [], f"the pool moved bytes in: {differ}"
