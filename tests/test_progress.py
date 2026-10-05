"""`manamap.progress`: the live file the job band draws. What it has to get right
is the one thing a terminal cannot say — that a job STOPPED — so the heartbeat
is tested apart from the count, and the log counter on the boundary case that
would make a running simulation read short.
"""
import json
import re
import time

from manamap import progress


def _read(p):
    return json.loads(p.path.read_text())


def test_the_heartbeat_moves_while_the_count_does_not(tmp_path):
    """A 5-minute test, a 20-minute Forge job: no progress for a long time is
    normal. Only a heartbeat that STOPS means the job died or the machine slept."""
    p = progress.Progress("pytest unit", total=10, unit="tests",
                          heartbeat=0.05, directory=tmp_path).start()
    first = _read(p)["updated_at"]
    time.sleep(0.3)
    second = _read(p)
    p.finish()
    assert second["done"] == 0 and second["state"] == "running"
    assert second["updated_at"] > first


def test_a_finish_says_whether_it_passed(tmp_path):
    with progress.Progress("regen", total=3, unit="targets", directory=tmp_path) as p:
        p.advance()
        p.advance(failed=1)
    doc = _read(p)
    assert (doc["done"], doc["failed"], doc["state"]) == (2, 1, "failed")
    ok = progress.Progress("regen", total=1, directory=tmp_path, name="other").start()
    ok.advance()
    ok.finish()
    assert _read(ok)["state"] == "passed"


def test_a_counter_is_polled_on_every_beat_and_at_the_end(tmp_path):
    seen = iter(range(100))
    p = progress.Progress("simulate x", total=100, unit="games", heartbeat=0.05,
                          counter=lambda: next(seen), directory=tmp_path).start()
    time.sleep(0.3)
    p.finish()
    assert _read(p)["done"] >= 3


def test_the_log_counter_reads_only_what_is_new_and_never_splits_a_line(tmp_path):
    """Forge writes a game's result line in one go, but a read can land in the
    middle of it; counting the half line twice, or not at all, would make a
    running simulation read wrong."""
    done = re.compile(r"^Game Result: Game (\d+) ended in ", re.M)
    log = tmp_path / "part-00.log"
    count = progress.LogCounter(done, str(tmp_path / "part-*.log"))
    log.write_text("Turn: Turn 1\nGame Result: Game 1 ended in a win\nGame Res")
    assert count() == 1
    with open(log, "a") as f:
        f.write("ult: Game 2 ended in a draw\n")
    assert count() == 2
    (tmp_path / "part-01.log").write_text("Game Result: Game 1 ended in a loss\n")
    assert count() == 3
    assert count() == 3, "a second read of unchanged files counted again"


def test_the_off_switch_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("MANAMAP_NO_PROGRESS", "1")
    with progress.Progress("regen", total=1, directory=tmp_path) as p:
        p.advance()
    assert not list(tmp_path.iterdir())
