"""Pilot: the gate on `data/queue.jsonl` — replay every line through the same
transition check `apply` uses, so a hand edit cannot leave the queue in a state
the loop could never have reached (a second challenge round, a result before a
promote, an id skipped). A fleet file, so it takes no slug.
"""
from manamap.pilot import queue
from manamap.pilot.common import report_errors


def validate(lines):
    errors, seen = [], []
    for n, e in enumerate(lines, 1):
        why = queue.check(e, seen)
        if why:
            errors.append(f"line {n} ({e.get('id') or e.get('of') or e.get('kind')}): {why}")
        seen.append(e)
    return errors


def main(args=None):
    if not queue.PATH.exists():
        print(f"no {queue.PATH.name} — the queue is empty (absent means absent)")
        return
    lines = queue.read()
    report_errors(f"{queue.PATH.name}", validate(lines))
    print(f"OK   {queue.PATH.name}: {len(lines)} line(s), {len(queue.items(lines))} item(s)")
