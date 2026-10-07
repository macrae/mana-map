"""Pilot: how the sub-agents are doing against their response-time targets.

The job band writes one line to `.progress/sla-log.jsonl` each time an agent with a
target (`sla_s:` in its charter's frontmatter) finishes. This reads it back: runs,
median, slowest, and how many missed, per agent. A repeat miss is the PRD's signal
to trim or split the agent. Gitignored and local, like the rest of `.progress/`.
"""
import json
import statistics

from manamap import config

LOG = config.DATA_DIR.parent / ".progress" / "sla-log.jsonl"


def rows(path=LOG):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue                      # a line caught mid-write is not a run
    return out


def summary(runs):
    by = {}
    for r in runs:
        by.setdefault(r.get("type", "?"), []).append(r)
    out = []
    for agent, rs in sorted(by.items()):
        secs = [float(r.get("elapsed_s") or 0) for r in rs]
        out.append({"agent": agent, "runs": len(rs), "sla_s": rs[-1].get("sla_s"),
                    "median_s": round(statistics.median(secs), 1), "max_s": round(max(secs), 1),
                    "missed": sum(1 for r in rs if r.get("missed")),
                    "last": rs[-1].get("at")})
    return out


def main(args=None):
    runs = rows()
    if not runs:
        print(f"no agent runs logged yet — the job band writes {LOG.name} when an agent "
              "with an sla_s target finishes")
        return
    print(f"{'agent':18} {'runs':>4} {'target':>7} {'median':>7} {'slowest':>8} {'missed':>7}")
    for s in summary(runs):
        print(f"{s['agent']:18} {s['runs']:>4} {str(s['sla_s']) + 's':>7} {str(s['median_s']) + 's':>7} "
              f"{str(s['max_s']) + 's':>8} {s['missed']:>3}/{s['runs']:<3}")
