"""Pilot: how the sub-agents are doing against their response-time targets.

The job band writes one line to `.progress/sla-log.jsonl` each time an agent with a
target (`sla_s:` in its charter's frontmatter) finishes. This reads it back: runs,
median, slowest, and how many missed, per agent. A repeat miss is the PRD's signal
to trim or split the agent. Gitignored and local, like the rest of `.progress/`.
"""
import json
import pathlib
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


BAND = config.DATA_DIR.parent / "tools" / "claude-plugins" / "job-band"
INSTALLED = pathlib.Path.home() / ".claude" / "plugins" / "installed_plugins.json"


def band_drift(installed=INSTALLED, repo=BAND):
    """Why the running band may not be the repo's, or None when they match.

    Claude Code runs a CACHED copy of a plugin, keyed on its version: an edit to
    `register.tsx` without a version bump never reaches the band that is running.
    That is how the SLA log stayed empty through a day of agent runs (2026-10-07):
    the code that writes it was in the repo and not in the cache.
    """
    try:
        doc = json.loads(installed.read_text())
    except (OSError, json.JSONDecodeError):
        return None                       # not installed here: nothing to compare
    rows = (doc.get("plugins") or doc).get("job-band@mana-map") or []
    if not rows:
        return None
    cached = pathlib.Path(rows[-1].get("installPath", "")) / "hooks" / "register.tsx"
    mine = repo / "hooks" / "register.tsx"
    if not cached.exists() or not mine.exists():
        return None
    if cached.read_bytes() != mine.read_bytes():
        return (f"the job band running ({rows[-1].get('version')}) is not the repo's — bump "
                "`version` in tools/claude-plugins/job-band/.claude-plugin/plugin.json, then "
                "`claude plugin marketplace update mana-map && claude plugin update "
                "job-band@mana-map` and restart Claude Code")
    return None


LATENCY = LOG.with_name("latency-log.jsonl")
#: The PRD's "time to first response from Jarvis".
FIRST_RESPONSE_TARGET_S = 2.0


def latency(path=LATENCY):
    """`{n, median_s, p90_s, over}` of submit-to-first-chunk times, or None."""
    secs = sorted(float(r["first_response_s"]) for r in rows(path) if "first_response_s" in r)
    if not secs:
        return None
    return {"n": len(secs), "median_s": round(statistics.median(secs), 2),
            "p90_s": secs[min(len(secs) - 1, int(0.9 * len(secs)))],
            "over": sum(1 for s in secs if s > FIRST_RESPONSE_TARGET_S)}


#: What the log cannot hold, said where its rows are read (found 2026-10-09: one row
#: after two days of trials, every one of them a charter pasted into general-purpose).
UNTIMED = ("only a run spawned BY ITS TYPE is timed — a charter pasted into a general-purpose "
           "agent is not, and a charter added this session needs a restart to be spawnable")


def main(args=None):
    drift = band_drift()
    if drift:
        print(f"WARNING: {drift}")
    lat = latency()
    if lat:
        print(f"first response: median {lat['median_s']}s, p90 {lat['p90_s']}s over {lat['n']} prompt(s); "
              f"{lat['over']} over the {FIRST_RESPONSE_TARGET_S:g}s target")
    runs = rows()
    if not runs:
        print(f"no agent runs logged yet — the job band writes {LOG.name} when an agent "
              "with an sla_s target finishes")
        print(UNTIMED)
        return
    print(f"{'agent':18} {'runs':>4} {'target':>7} {'median':>7} {'slowest':>8} {'missed':>7}")
    for s in summary(runs):
        print(f"{s['agent']:18} {s['runs']:>4} {str(s['sla_s']) + 's':>7} {str(s['median_s']) + 's':>7} "
              f"{str(s['max_s']) + 's':>8} {s['missed']:>3}/{s['runs']:<3}")
    print(UNTIMED)
