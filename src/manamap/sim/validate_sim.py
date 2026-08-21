"""Simulation: form-check a run record, and prove its analysis is what its logs say.

A run record is tier ◆ SEEDED (runs made with `-s`, replayable byte for byte) or
◆ SAMPLED (the first runs, made before the seed flag was known). Either way the parser
over the logs is deterministic, so the one thing a validator CAN prove is that the tracked
`analysis` block is exactly what `sim.parse` derives from the logs — where the logs exist.
Where they do not (a fresh clone; they are gitignored), the record is form-checked: the
keys a consumer relies on, counts that agree with each other, every seat's decklist sha
present, and the SEEDED/SAMPLED assumption that matches whether seeds are recorded. A record that cannot be re-derived is still evidence
of a run; it just cannot be re-proven here, and the OK line says so.
"""

import re

from manamap.config import SIM_DIR
from manamap.pilot.common import deck_dir, load_json, report_errors
from manamap.sim import parse as sim_parse
from manamap.sim.forge import _seat_label, record_commanders

REQUIRED = {"run_id", "slug", "at", "engine", "seats", "games_requested", "games_completed",
            "summary", "outcomes", "analysis", "assumptions"}
_SHA = re.compile(r"^[0-9a-f]{64}$")


def validate(rec, slug, logs_text=None):
    errors = []
    missing = REQUIRED - set(rec)
    if missing:
        return [f"missing keys {sorted(missing)}"]
    if rec["slug"] != slug:
        errors.append(f"slug {rec['slug']!r} != {slug!r}")
    seats = rec.get("seats") or []
    if not seats or seats[0].get("slug") != slug:
        errors.append("seats[0] must be this deck")
    for i, s in enumerate(seats):
        if not _SHA.match(str(s.get("decklist_sha256") or "")):
            errors.append(f"seats[{i}] ({s.get('slug')}): decklist_sha256 is not a sha256")
    n = rec["games_completed"]
    if n != len(rec["outcomes"]):
        errors.append(f"games_completed {n} != {len(rec['outcomes'])} outcomes")
    if rec["analysis"].get("games") != n:
        errors.append(f"analysis.games {rec['analysis'].get('games')} != games_completed {n}")
    wins = rec["summary"].get("wins") or {}
    if sum(wins.values()) + rec["summary"].get("draws", 0) != n:
        errors.append(f"summary.wins {wins} + draws {rec['summary'].get('draws')} != {n}")
    a_wins = {k: v.get("wins") for k, v in (rec["analysis"].get("seats") or {}).items()}
    if any(a_wins.get(k) != v for k, v in wins.items()):
        errors.append(f"analysis per-seat wins {a_wins} disagree with summary.wins {wins}")
    for s, d in (rec["analysis"].get("seats") or {}).items():
        lo, hi = (d.get("win_rate_ci95") or [None, None])
        if lo is not None and not (0 <= lo <= (d.get("win_rate") or 0) <= hi <= 1):
            errors.append(f"analysis.seats[{s}]: win_rate {d.get('win_rate')} outside ci95 [{lo}, {hi}]")
    seeded = bool(rec.get("seeds"))
    want = "SEEDED" if seeded else "SAMPLED"
    if not any(want in str(x) for x in rec["assumptions"]):
        errors.append(f"assumptions must state {want} — the record "
                      f"{'carries seeds' if seeded else 'has no seeds'}")
    if seeded and len(rec["seeds"]) != rec.get("jobs"):
        errors.append(f"{len(rec['seeds'])} seeds for {rec.get('jobs')} jobs")
    if logs_text:
        label = _seat_label([s["forge_name"] for s in seats])
        _, derived = sim_parse.analyze_logs(logs_text, label, record_commanders(rec))
        if derived != rec["analysis"]:
            keys = [k for k in set(derived) | set(rec["analysis"]) if derived.get(k) != rec["analysis"].get(k)]
            errors.append(f"analysis does not match what the logs derive (differs at {sorted(keys)}) — "
                          f"`simulate {slug} --analyze {rec['run_id']}` rewrites it")
    return errors


def main(args):
    slug = args.slug
    base = deck_dir(slug) / SIM_DIR
    paths = sorted(base.glob("*.json")) if base.is_dir() else []
    if not paths:
        print(f"OK   {slug} — no simulation runs")
        return
    errors, reproven = [], 0
    for p in paths:
        rec = load_json(p)
        logs = sorted((base / "logs" / p.stem).glob("part-*.log"))
        texts = [l.read_text(encoding="utf-8", errors="replace") for l in logs] or None
        errs = validate(rec, slug, texts)
        errors += [f"{p.name}: {e}" for e in errs]
        reproven += bool(texts)
    report_errors(f"{slug} simulation runs", errors,
                  ok_line=f"OK   {slug} — {len(paths)} run(s), {reproven} re-derived from logs"
                          f"{'' if reproven == len(paths) else ' (the rest form-checked; logs are gitignored)'}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-sim <slug>`.")
