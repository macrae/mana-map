"""One heavy job at a time, and nothing left behind.

Session 0ada19cc (2026-10-23 -> 10-29) took the Mac to a load average of 47 on eight
cores: `make prepush` (the regression tier's `-n auto` beside the browser suite's
four Chromiums) ran from the main thread and from subagents AT ONCE, while a retrain
whose `train-ability` normally takes ~4 min crawled for 8 h behind them. And a
stopped background task is not a dead process tree: `TaskStop` ended the shell, and
`train-ability` (PID 17140) kept running until it was killed by hand. xdist workers do not
carry the word "pytest" on their command line, so a `ps | grep pytest` cannot see
them either.

Three entry points, stdlib only, so the hook runs as a plain script in ~20 ms:

  python -I src/manamap/jobs.py hook   # PreToolUse on Bash (.claude/settings.json):
                                       # refuses a heavy command while another heavy
                                       # job runs. Fires for subagents too.
  make jobs                            # what is running, orphans marked
  make reap                            # kill the ORPHANS (parent is launchd), only

Override, for a deliberate second job: put `MANAMAP_ALLOW_CONCURRENT=1` in the command.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time

OVERRIDE = "MANAMAP_ALLOW_CONCURRENT=1"

# The make targets that start a suite or a fleet-wide rebuild.
_MAKE_TARGETS = re.compile(
    r"^(?:prepush(?:-full)?|test(?:-unit-isolated|-browser|-fresh|-report|-all)?"
    r"|regression(?:-prepush)?|integration|check|manuals|demo)$")
# The card pipeline's compute steps, and the pilot's long runs.
_PIPELINE = {"run", "train", "train-ability", "train-vae", "embed", "reduce", "export",
             "synergy", "power-creep", "cluster-regions", "eval-embeddings",
             "eval-obsolescence", "vae-cache", "span-cache", "project-space"}
_PILOT = {"regen", "simulate", "experiment", "scenario-ab"}
# The prepush runner and the pipeline module, run as `python -m`.
_MODULES = {"manamap.check_scope", "manamap.pipeline"}
# A pytest run that fans out. `-n0` / `-n 0` is one process: a single test, not a job.
_SERIAL = re.compile(r"(?:^|\s)-n\s*0(?:\s|$)|-p\s+no:xdist")
# xdist's workers: execnet bootstraps each one with this line, and nothing else.
_WORKER = "import sys;exec(eval(sys.stdin.readline()))"
_HEADLESS = re.compile(r"headless_shell|chrome-headless|--headless")
_HTTP = re.compile(r"\bhttp\.server\b|\bmanamap\s+serve\b")
# Words that run the next word as the command: wrappers, and the shell keywords a
# loop or an `if` puts in front of one.
_PREFIXES = {"time", "nohup", "exec", "caffeinate", "env", "command", "do", "then", "else",
             "{", "!"}


def _is_heavy_argv(argv: list[str]) -> bool:
    """Is this argv — a program and its arguments — a heavy job? Judged by the
    program at command position, never by words elsewhere: `grep "make prepush"`,
    `tail -f prepush.log` and a commit message naming `manamap run` are not jobs."""
    while argv and (re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[0]) or argv[0] in _PREFIXES
                    or argv[0].startswith("-")):
        argv = argv[1:]
    if not argv:
        return False
    prog, rest = os.path.basename(argv[0]), argv[1:]
    line = " ".join(argv)
    if prog in ("make", "gmake"):
        return any(_MAKE_TARGETS.match(a) for a in rest if not a.startswith("-"))
    if prog.startswith("python"):
        while rest and rest[0].startswith("-") and rest[0] not in ("-m", "-c"):
            rest = rest[1:]                                   # -u, -I, -B …
        if len(rest) >= 2 and rest[0] == "-m":
            if rest[1] in _MODULES:
                return True
            if rest[1] == "pytest":
                return not _SERIAL.search(line)
        if not rest or rest[0].startswith("-"):
            return False
        prog, rest = os.path.basename(rest[0]), rest[1:]  # a script: .venv/bin/manamap
    if prog == "pytest":
        return not _SERIAL.search(line)
    if prog == "manamap":
        words = [a for a in rest if not a.startswith("-")]
        if words and words[0] in _PIPELINE:
            return True
        return len(words) >= 2 and words[0] == "pilot" and words[1] in _PILOT
    return False


def is_heavy_command(cmd: str) -> bool:
    """Would this Bash command start a heavy job? Judged per shell segment, so
    `pytest -n0 one; pytest` is heavy and `cd repo && make prepush` is too."""
    for seg in re.split(r"&&|\|\||;|\||\n|\(|\)|`|\$\(", cmd):
        if _is_heavy_argv(seg.split()):
            return True
    return False


def classify(command: str) -> str | None:
    """What a RUNNING process is: heavy / worker / browser / server, or None.
    The shell wrapping a `make prepush` is not a second job: only the argv's own
    program counts, and the `make` beside the shell is the job."""
    if _WORKER in command:
        return "worker"
    if _is_heavy_argv(command.split()):
        return "heavy"
    if _HEADLESS.search(command):
        return "browser"
    if _HTTP.search(command):
        return "server"
    return None


def processes() -> list[dict]:
    out = subprocess.run(["ps", "-Ao", "pid=,ppid=,pcpu=,etime=,command="],
                         capture_output=True, text=True, check=True).stdout
    me = {os.getpid(), os.getppid()}
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid, ppid = int(parts[0]), int(parts[1])
        kind = classify(parts[4])
        if kind and pid not in me:
            rows.append({"pid": pid, "ppid": ppid, "cpu": float(parts[2]),
                         "etime": parts[3], "kind": kind, "command": parts[4],
                         "orphan": ppid == 1})
    return rows


def _running_jobs(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["kind"] in ("heavy", "worker")]


def hook(stdin=sys.stdin) -> int:
    """PreToolUse: exit 2 (the call is refused, stderr goes to the model) when a
    heavy command would start beside a running heavy job. Anything unexpected
    fails OPEN: a broken guard must never block the shell."""
    try:
        cmd = (json.load(stdin).get("tool_input") or {}).get("command") or ""
        if OVERRIDE in cmd or not is_heavy_command(cmd):
            return 0
        running = _running_jobs(processes())
    except Exception:
        return 0
    if not running:
        return 0
    load = os.getloadavg()[0]
    lines = [f"REFUSED: a heavy job is already running (load {load:.1f} on "
             f"{os.cpu_count()} cores). One heavy job at a time: wait for it, or stop "
             f"it and run `make reap` so nothing it started is left behind."]
    for r in running[:8]:
        tag = " ORPHAN" if r["orphan"] else ""
        lines.append(f"  {r['pid']:>6} {r['cpu']:5.1f}% {r['etime']:>11} "
                     f"{r['kind']}{tag}  {r['command'][:110]}")
    lines.append(f"Deliberately concurrent? Put {OVERRIDE} in the command.")
    print("\n".join(lines), file=sys.stderr)
    return 2


def report(rows: list[dict]) -> str:
    if not rows:
        return "no suite, pipeline, worker, headless browser or server process is running"
    lines = [f"load {os.getloadavg()[0]:.1f} on {os.cpu_count()} cores"]
    for r in sorted(rows, key=lambda r: (r["kind"], r["pid"])):
        tag = "ORPHAN " if r["orphan"] else "       "
        lines.append(f"{r['pid']:>6} {r['cpu']:5.1f}% {r['etime']:>11} {tag}"
                     f"{r['kind']:<7} {r['command'][:120]}")
    return "\n".join(lines)


def reap(rows: list[dict]) -> list[int]:
    """SIGTERM every orphan (its parent died and launchd adopted it). A process
    with a live parent belongs to someone — a terminal, a running suite — and is
    left alone."""
    killed = []
    for r in rows:
        if r["orphan"]:
            try:
                os.kill(r["pid"], signal.SIGTERM)
                killed.append(r["pid"])
            except ProcessLookupError:
                pass
    return killed


def main(argv: list[str]) -> int:
    mode = argv[0] if argv else "list"
    if mode == "hook":
        return hook()
    rows = processes()
    print(report(rows))
    if mode == "reap":
        # Killing an orphaned `make` orphans its pytest, and that its workers: sweep
        # until a pass finds nothing new.
        killed: list[int] = []
        for _ in range(5):
            fresh = [p for p in reap(processes()) if p not in killed]
            if not fresh:
                break
            killed += fresh
            time.sleep(1)
        print(f"reaped {len(killed)} orphan(s): {killed}" if killed else "no orphans")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
