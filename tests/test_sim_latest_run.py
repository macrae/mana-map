"""`list_runs` is ordered by DATE, so `[-1]` is the latest run.

It sorted by FILENAME, and a run id begins with the opponents:

    giada-angels-vs-baylen-tokens-vs-abaddon-n120-...   2026-09-07, 120 games
    giada-angels-vs-vito-vs-baylen-tokens-n20-...       2026-08-26,  20 games

'b' precedes 'v', so the August run sorted last and the dossier reported it as
`latest`: 0.118 over 20 games, while a 120-game run reading 0.252 sat beside it
unread. Six times the sample and more than twice the rate, hidden by an
alphabet — and nothing about the output gave it away, because it showed a real
run with a real interval.

Change the pod and the ordering changes with it. That is the shape of the bug:
a figure selected by something that has nothing to do with the question.
"""

import json

import pytest

from manamap.sim import forge


def _write(tmp_path, name, at, games):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"run_id": name, "at": at, "games_completed": games,
                             "slug": "x", "seats": [{"slug": "x"}]}))
    return p


@pytest.fixture
def runs(tmp_path, monkeypatch):
    d = tmp_path / "decks" / "x" / "sim"
    d.mkdir(parents=True)
    monkeypatch.setattr(forge, "_out_dir", lambda slug: d)
    return d


def test_the_latest_run_is_the_newest_not_the_last_alphabetically(runs):
    """THE EXACT PAIR, with the real dates and sizes."""
    _write(runs, "giada-angels-vs-baylen-tokens-vs-abaddon-n120-a", "2026-09-07", 120)
    _write(runs, "giada-angels-vs-vito-vs-baylen-tokens-n20-b", "2026-08-26", 20)
    got = forge.list_runs("x")
    assert [r["games_completed"] for r in got] == [20, 120], "oldest first"
    assert got[-1]["games_completed"] == 120, (
        "the dossier takes [-1] as the latest; alphabetically that was the "
        "August 20-game run")


def test_same_day_runs_break_the_tie_on_mtime(runs):
    """`at` is a DATE with no clock, so two runs on one day tie on it. The file's
    write time is the only thing left that orders them."""
    import os
    import time
    a = _write(runs, "aaa-n10", "2026-09-07", 10)
    b = _write(runs, "zzz-n20", "2026-09-07", 20)
    # make the ALPHABETICALLY-FIRST file the newer one
    now = time.time()
    os.utime(b, (now - 500, now - 500))
    os.utime(a, (now, now))
    assert forge.list_runs("x")[-1]["run_id"] == "aaa-n10"


def test_an_undated_record_sorts_as_OLD(runs):
    """A record written before `at` existed is old, not new. Sorting it last
    would make the least-known run the headline."""
    _write(runs, "zzz-modern", "2026-09-07", 100)
    p = runs / "aaa-ancient.json"
    p.write_text(json.dumps({"run_id": "aaa-ancient", "games_completed": 5,
                             "slug": "x", "seats": [{"slug": "x"}]}))
    got = forge.list_runs("x")
    assert got[0]["run_id"] == "aaa-ancient"
    assert got[-1]["run_id"] == "zzz-modern"


def test_no_runs_is_an_empty_list_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(forge, "_out_dir", lambda slug: tmp_path / "nope")
    assert forge.list_runs("x") == []
