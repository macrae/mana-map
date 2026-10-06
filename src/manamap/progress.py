"""Live progress for a long job: `<repo>/.progress/<name>-<pid>.json`.

The file the Claude Code `job-band` mod draws above the prompt — a bar, the
elapsed time, an ETA — and the one thing a terminal cannot say on its own: that
a job has STOPPED. `updated_at` is a heartbeat, refreshed every few seconds by a
daemon thread whether or not the count moved, so a heartbeat that stops means
the process died or the machine slept. The first three-tier test report idled
into sleep for 1h50m with nothing on screen to say so; that is the failure this
exists to make visible.

    with Progress("regen", total=115, unit="targets") as p:
        for target in targets:
            ...
            p.advance()

`counter=` replaces `advance` for a job whose progress lives somewhere else
(`simulate` counts finished games in its Forge logs): it is polled on every
heartbeat. Written atomically (temp file + rename); a write that fails (a
read-only checkout) is dropped, never raised — progress must not break a run.
`MANAMAP_NO_PROGRESS=1` turns every writer off.

Never part of a result: nothing reads these files but the band, they are
gitignored, and a finished job's file is pruned after a day.
"""

import json
import os
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DIR = REPO / ".progress"
HEARTBEAT_S = 5.0
#: A finished job's file is pruned this long after it last wrote (the band
#: hides it after two minutes; the file only feeds the band). It was a day, and
#: a day of pytest runs piled up as cruft on the band (2026-10-06).
PRUNE_AFTER_S = 600


def enabled():
    return not os.environ.get("MANAMAP_NO_PROGRESS")


class Progress:
    def __init__(self, label, total=None, unit="", name=None, counter=None,
                 heartbeat=HEARTBEAT_S, directory=None):
        self.label = label
        self.total = total
        self.unit = unit
        self.counter = counter
        self.heartbeat = heartbeat
        self.dir = Path(directory) if directory else DIR
        self.path = self.dir / f"{name or label.split()[0]}-{os.getpid()}.json"
        self.done = 0
        self.failed = 0
        self.detail = ""
        self.state = "running"
        self.started = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    # ── the writer's side ───────────────────────────────────────────────

    def advance(self, n=1, failed=0, detail=None):
        with self._lock:
            self.done += n
            self.failed += failed
            if detail is not None:
                self.detail = detail

    def set(self, done=None, total=None, detail=None):
        with self._lock:
            if done is not None:
                self.done = done
            if total is not None:
                self.total = total
            if detail is not None:
                self.detail = detail

    def start(self):
        if not enabled():
            return self
        self._prune()
        self.write()
        self._thread = threading.Thread(target=self._beat, daemon=True,
                                        name=f"progress-{self.label}")
        self._thread.start()
        return self

    def finish(self, ok=True):
        self._stop.set()
        if self.counter is not None:
            self._poll()
        self.state = "passed" if ok and not self.failed else "failed"
        if enabled():
            self.write()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.finish(ok=exc_type is None)
        return False

    # ── internals ───────────────────────────────────────────────────────

    def payload(self):
        with self._lock:
            return {"label": self.label, "done": self.done, "total": self.total,
                    "unit": self.unit, "failed": self.failed, "state": self.state,
                    "started_at": self.started, "updated_at": time.time(),
                    "detail": self.detail or f"pid {os.getpid()}"}

    def write(self):
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.payload()))
            os.replace(tmp, self.path)
        except OSError:
            pass

    def _poll(self):
        try:
            self.set(done=self.counter())
        except Exception:                      # noqa: BLE001 - never break a run
            pass

    def _beat(self):
        while not self._stop.wait(self.heartbeat):
            if self.counter is not None:
                self._poll()
            self.write()

    def _prune(self):
        for old in self.dir.glob("*.json"):
            try:
                if old.resolve() == self.path.resolve():
                    continue
                if time.time() - old.stat().st_mtime > PRUNE_AFTER_S or _dead(old):
                    old.unlink()
            except OSError:
                pass


def _pid_of(path):
    """`<name>-<pid>.json` -> pid, or None."""
    tail = path.stem.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _dead(path):
    """A file that still says RUNNING but whose process is gone: a killed run,
    or a crash. It would otherwise sit on the band as NO HEARTBEAT for half an
    hour. A live process with a stale heartbeat (a machine that slept) is NOT
    dead and stays — that is the warning the band exists to give."""
    try:
        if json.loads(path.read_text()).get("state") != "running":
            return False
    except (OSError, ValueError):
        return False
    pid = _pid_of(path)
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False          # alive, someone else's
    return False


class LogCounter:
    """Count a pattern across growing log files, reading only what is new.

    `simulate`'s Forge logs grow to tens of megabytes; re-reading them on every
    heartbeat would cost more than the progress is worth. Each file keeps its
    offset, and a match split across a read boundary is caught by carrying the
    unfinished last line into the next read.
    """

    def __init__(self, pattern, glob_pattern):
        self.pattern = pattern
        self.glob = glob_pattern
        self._state = {}                         # path -> (offset, count, tail)

    def __call__(self):
        total = 0
        for path in sorted(Path(self.glob).parent.glob(Path(self.glob).name)):
            offset, count, tail = self._state.get(path, (0, 0, ""))
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    f.seek(offset)
                    chunk = f.read()
                    offset = f.tell()
            except OSError:
                chunk = ""
            text = tail + chunk
            complete, _, tail = text.rpartition("\n")
            count += len(self.pattern.findall(complete + "\n")) if complete else 0
            self._state[path] = (offset, count, tail)
            total += count
        return total
