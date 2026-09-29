"""THE LIFT-FRESHNESS GATE: does a committed lifted scenario still match a fresh
lift of the same cut?

CLAUDE.md named the gap: "nothing compares a lifted scenario against a fresh
lift of the same cut, which is the freshness test every other tracked artifact
has". It cost a real artifact — goblin-storm 010 was resolved against a board
that omitted seat 2's Lord of Extinction, and 012 had to supersede it by hand
after a bridge fix; radagast 008 was proven with 7 of our creatures where a
re-lift gives 11.

The comparison is over the CANONICAL board: per seat, life, the commander's
zone, and the sorted (name, tapped, pt, token) of every board entry — the
facts a resolution reasons from. Hand estimates and notes are not compared.

FAIL when the artifact has NO CHECKER VERDICT yet and the fresh lift differs:
it is work in progress and the board it will be argued from is wrong. NOTE
once the loop has finished on it, pass OR fail: finished work is not condemned
by a later bridge fix — `validate_stack` gives `unknown_cards` the same
treatment — but the drift is said out loud, which is what 010 lacked, and a
`fail` the loop gave up on is not re-litigated by a gate.

Where the logs are absent (a clone, another machine) the fresh lift cannot be
made; then `source.bridge_sha`, stamped at lift time, says whether the bridge
has moved since, which is the one thing that can be known without the log.
"""

import hashlib
import json

from manamap.pilot.common import deck_dir, load_json, report_errors


def canonical(doc):
    """The board a resolution reasons from, in a form two lifts can be diffed on."""
    out = {}
    for s in ((doc.get("scenario") or {}).get("seats") or []):
        cm = s.get("commander") or {}
        board = sorted(((b.get("name") or ""), bool(b.get("tapped")), (b.get("pt") or ""),
                        bool(b.get("token")))
                       for b in (s.get("board") or []))
        out[s.get("seat")] = {"life": s.get("life"), "commander_zone": cm.get("zone"),
                              "board": board}
    return out


def lift_sha(doc):
    return hashlib.sha256(json.dumps(canonical(doc), sort_keys=True).encode()).hexdigest()[:12]


def bridge_sha():
    import inspect
    from manamap.sim import bridge
    return hashlib.sha256(inspect.getsource(bridge).encode()).hexdigest()[:12]


def diff(old, new):
    """Human-readable differences between two canonical boards."""
    lines = []
    for seat in sorted(set(old) | set(new)):
        a, b = old.get(seat), new.get(seat)
        if a is None or b is None:
            lines.append(f"{seat}: present in {'the committed' if a else 'the fresh'} lift only")
            continue
        if a["life"] != b["life"]:
            lines.append(f"{seat}: life {a['life']} committed, {b['life']} fresh")
        if a["commander_zone"] != b["commander_zone"]:
            lines.append(f"{seat}: commander in {a['commander_zone']!r} committed, "
                         f"{b['commander_zone']!r} fresh")
        ca, cb = collections_counter(a["board"]), collections_counter(b["board"])
        for item, n in (cb - ca).items():
            lines.append(f"{seat}: the fresh lift has {n} more {item[0]!r}"
                         + (f" ({item[2]})" if item[2] else "") + " — MISSING from the committed board")
        for item, n in (ca - cb).items():
            lines.append(f"{seat}: the committed board has {n} {item[0]!r} the fresh lift does not")
    return lines


def collections_counter(items):
    import collections
    return collections.Counter(items)


def fresh_lift(doc):
    """Re-lift the committed cut, or None when the logs are not here."""
    from manamap.sim import bridge
    from manamap.sim import parse as sim_parse
    from manamap.sim.forge import _out_dir, _seat_label
    sc = doc.get("scenario") or {}
    src = sc.get("source") or {}
    slug = doc.get("slug") or doc.get("deck")
    if not (src.get("run_id") and slug):
        return None
    base = _out_dir(slug)
    rec = load_json(base / f"{src['run_id']}.json")
    if not rec:
        return None
    log = base / "logs" / src["run_id"] / (src.get("log") or "")
    if not log.exists():
        return None
    games = sim_parse.parse_games(log.read_text(encoding="utf-8", errors="replace"))
    gij = src.get("game_in_job") or 1
    if gij > len(games):
        return None
    cut = src.get("cut") or {}
    label = _seat_label([s["forge_name"] for s in rec["seats"]])
    step_text = cut.get("step") or cut.get("phase")
    return bridge.build_scenario(slug, rec, src["game"], cut["turn"], step_text, games[gij - 1], label)


def check(doc):
    """(errors, notes) for one artifact. Only a lifted v2 scenario is checked."""
    sc = doc.get("scenario") or {}
    src = sc.get("source") or {}
    if sc.get("version") != 2 or not src.get("run_id"):
        return [], []
    verdict = (doc.get("checker") or {}).get("verdict")
    finished = verdict in ("pass", "fail")
    fresh = fresh_lift(doc)
    if fresh is None:
        notes = []
        if src.get("bridge_sha") and src["bridge_sha"] != bridge_sha():
            notes.append(f"lifted under bridge {src['bridge_sha']} and the bridge is "
                         f"{bridge_sha()} now — re-lift where the logs are to see whether "
                         f"the board moved")
        elif not src.get("bridge_sha"):
            notes.append("lifted before the bridge stamped its version; the logs are not "
                         "here, so whether the board still matches cannot be told")
        return [], notes
    lines = diff(canonical(doc), canonical(fresh))
    if not lines:
        return [], []
    head = (f"a fresh lift of {src['run_id'][:40]} game {src.get('game')} turn "
            f"{(src.get('cut') or {}).get('turn')} differs from the committed board:")
    body = [head] + ["  " + l for l in lines[:12]] + (["  …"] if len(lines) > 12 else [])
    if finished:
        return [], [f"checker {verdict}, and " + " ".join(body)]
    return ["\n".join(body) + "\n  — unresolved, so the board it will be argued from is "
            "wrong; re-lift (`sim-scenario … --stack`) and supersede this one"], []


def main(args):
    slug = args.slug
    base = deck_dir(slug)
    paths = sorted((base / "stacks").glob("*.json"))
    if getattr(args, "stack", None):
        paths = [p for p in paths if p.name.startswith(f"{args.stack}-")]
    errors, checked = [], 0
    for p in paths:
        doc = load_json(p) or {}
        errs, notes = check(doc)
        if not ((doc.get("scenario") or {}).get("source") or {}).get("run_id"):
            continue
        checked += 1
        for n in notes:
            print(f"  ! {p.name}: {n}")
        errors += [f"{p.name}: {e}" for e in errs]
        if not errs and not notes:
            print(f"OK   {p.name} (the fresh lift matches)")
    if not checked:
        print(f"{slug}: no lifted scenario under stacks/")
        return
    report_errors(f"{slug}: lifted scenarios", errors)
    print(f"{slug}: {checked} lifted scenario(s) checked")
