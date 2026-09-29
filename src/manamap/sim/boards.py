"""THE BOARD FINDER: which moments in a run are worth lifting, and how often they recur.

`sim-scenario` lifts a board by hand — `--game G --turn T` — and the two
boards proven so far (goblin-storm 011 "the modal board", 012 "the widest")
were chosen by the author reading logs. A board is worth a procedure only if
it RECURS: one game is an anecdote. So every criterion here names a CUT (game,
global turn, CR step) and a SHAPE (what must be the same for two hits to count
as one kind of moment), and the shortlist is ranked by how many games reached
that shape, with a Wilson interval on the share.

Every figure a criterion reads is the board series' estimate — the bridge's
reconstruction from resolve lines — so a criterion inherits its floors: a body
that entered without being cast is seen only when it acts, printed power
ignores counters and anthems, summoning sickness is unknown. `lethal-missed`
says so in its own row.

`--lift` hands each exemplar to `bridge.lift` with the finder's provenance in
`extras.finder`, which is what a later handbook proposal cites.
"""

import collections
import json
import pathlib

from manamap.pilot.common import deck_dir, load_json
from manamap.sim import board_series as bs
from manamap.sim import parse as sim_parse
from manamap.sim.bridge import reconstruct
from manamap.sim.forge import _out_dir, deck_meta_name, record_commanders
from manamap.sim.parse import wilson

#: Bodies bucketed so "6 or 7 creatures" is one shape, not two.
def bodies_bucket(n):
    return "0-2" if n <= 2 else "3-5" if n <= 5 else "6-8" if n <= 8 else "9+"


def turn_bucket(t):
    return "early (≤8)" if t <= 8 else "mid (9-16)" if t <= 16 else "late (17+)"


def _front(v):
    if isinstance(v, (set, list, tuple)):
        v = sorted(v)[0] if v else None
    return v.split(" // ")[0].strip() if isinstance(v, str) else v


def _games_of(slug, run_id):
    """(record, [(game_index, outcome, game)]) — every game of the run the logs hold."""
    base = _out_dir(slug)
    rec = load_json(base / f"{run_id}.json")
    if not rec:
        raise SystemExit(f"{slug}: no run {run_id!r} under {base}")
    logdir = base / "logs" / run_id
    if not logdir.is_dir():
        raise SystemExit(f"{logdir} is missing — logs are gitignored and exist only where the "
                         f"run was made")
    parsed, rows = {}, []
    for i, out in enumerate(rec["outcomes"], 1):
        name = out.get("log")
        if name not in parsed:
            p = logdir / name
            parsed[name] = (sim_parse.parse_games(p.read_text(encoding="utf-8", errors="replace"))
                            if p.exists() else [])
        gij = out.get("game_in_job") or 1
        if gij <= len(parsed[name]):
            rows.append((i, out, parsed[name][gij - 1]))
    return rec, rows


def _cut_state(game, turn, phase, step, cmd):
    states, notes, _a, _p, _s = reconstruct(game, turn, phase, step, cmd)
    return states, notes


# ── the criteria ──────────────────────────────────────────────────────────────
#
# Each takes (game, ours_label, cmd, facts_for_seat, params) and returns a list
# of hits: {turn, phase, step, shape: {...}, detail: {...}}.

def _own_turns(game, ours):
    return bs.own_turns(game).get(ours, [])


def _acted(game, ours, turn):
    """Did our seat cast or activate anything on this global turn?"""
    return any(ev.get("seat") == ours and ev.get("turn") == turn and ev.get("kind") in ("cast", "activated")
               for ev in game["events"])


def _attacked(game, ours, turn):
    return any(ev.get("seat") == ours and ev.get("turn") == turn and ev.get("kind") == "attack"
               for ev in game["events"])


def crit_widest(game, ours, cmd, facts, params):
    best = None
    for g in _own_turns(game, ours):
        states, _ = _cut_state(game, g, "precombat main", None, cmd)
        snap = bs.snapshot(states[ours])
        if best is None or snap["bodies"] > best[1]["bodies"]:
            best = (g, snap)
    if not best:
        return []
    g, snap = best
    return [{"turn": g, "phase": "precombat main", "step": None,
             "shape": {"bodies": bodies_bucket(snap["bodies"]),
                       "commander": snap["commander_on_battlefield"]},
             "detail": snap}]


def crit_modal(game, ours, cmd, facts, params):
    lo, hi = params.get("from_turn", 4), params.get("to_turn", 10)
    rows = []
    for k, g in enumerate(_own_turns(game, ours), 1):
        if not lo <= k <= hi:
            continue
        states, _ = _cut_state(game, g, "precombat main", None, cmd)
        snap = bs.snapshot(states[ours])
        rows.append({"turn": g, "phase": "precombat main", "step": None,
                     "shape": {"bodies": snap["bodies"], "lands": snap["lands"],
                               "commander": snap["commander_on_battlefield"]},
                     "detail": snap})
    return rows


def crit_death(game, ours, cmd, facts, params):
    et = (facts or {}).get("eliminated_turn")
    if not et:
        return []
    before = [g for g in _own_turns(game, ours) if g < et]
    if not before:
        return []
    g = before[-1]
    states, _ = _cut_state(game, g, "precombat main", None, cmd)
    snap = bs.snapshot(states[ours])
    return [{"turn": g, "phase": "precombat main", "step": None,
             "shape": {"how": (facts or {}).get("eliminated_how"),
                       "by": (facts or {}).get("eliminated_by"),
                       "turn": turn_bucket(et)},
             "detail": dict(snap, eliminated_turn=et)}]


def crit_held(game, ours, cmd, facts, params):
    n = params.get("n", 4)
    hits = []
    for g in _own_turns(game, ours):
        if _acted(game, ours, g):
            continue
        states, _ = _cut_state(game, g, "ending", "cleanup", cmd)
        snap = bs.snapshot(states[ours])
        if snap["open_lands"] >= n:
            est = states[ours]["kept"] + states[ours]["draw_steps"] + states[ours]["drawn"] \
                - states[ours]["lands_n"] - states[ours]["cast_n"] - states[ours]["discards"]
            hits.append({"turn": g, "phase": "ending", "step": "cleanup",
                         "shape": {"open_lands": "4-5" if snap["open_lands"] <= 5 else "6+",
                                   "hand_estimate": "0-2" if est <= 2 else "3-5" if est <= 5 else "6+"},
                         "detail": dict(snap, hand_estimate=max(0, est))})
    return hits


def crit_pre_wipe(game, ours, cmd, facts, params):
    fact = {"per_seat": facts_all(game, cmd)}
    hits = []
    for w in sim_parse.wipes(fact, ours):
        t = w["turn"]
        states, _ = _cut_state(game, t, "beginning", "untap", cmd)
        snap = bs.snapshot(states[ours])
        hits.append({"turn": t, "phase": "beginning", "step": "untap",
                     "shape": {"bodies": bodies_bucket(snap["bodies"]),
                               "lost": "0-2" if w["permanents_lost_mine"] <= 2 else "3-5"
                               if w["permanents_lost_mine"] <= 5 else "6+"},
                     "detail": dict(snap, wipe=w)})
    return hits


def crit_lethal_missed(game, ours, cmd, facts, params):
    hits = []
    for g in _own_turns(game, ours):
        states, _ = _cut_state(game, g, "precombat main", None, cmd)
        me = states[ours]
        power = 0
        for p in list(me["perms"].values()) + list(me["tokens"].values()):
            pt = p.get("pt")
            if pt and not p.get("tapped"):
                pw = str(pt).split("/")[0].strip()
                if pw.isdigit():
                    power += int(pw)
        if power <= 0:
            continue
        for seat, st in states.items():
            if seat == ours or st["life"] <= 0:
                continue
            if st["life"] <= power and not _attacked(game, ours, g):
                hits.append({"turn": g, "phase": "precombat main", "step": None,
                             "shape": {"target": seat.split("-", 1)[1] if "-" in seat else seat,
                                       "margin": "exact" if power - st["life"] < 3 else "wide"},
                             "detail": {"printed_power_untapped": power, "target_life": st["life"],
                                        "note": "printed power of untapped bodies, summoning "
                                                "sickness and blockers unknown — a floor on "
                                                "lethal, never a proof of it"}})
    return hits


def crit_first_attack(game, ours, cmd, facts, params):
    fa = (facts or {}).get("first_attack_turn")
    if not fa:
        return []
    states, _ = _cut_state(game, fa, "combat", "beginning of combat", cmd)
    snap = bs.snapshot(states[ours])
    return [{"turn": fa, "phase": "combat", "step": "beginning of combat",
             "shape": {"bodies": bodies_bucket(snap["bodies"]), "turn": turn_bucket(fa)},
             "detail": snap}]


CRITERIA = {
    "widest": (crit_widest, "our precombat main of the own turn with the most bodies"),
    "modal": (crit_modal, "every own turn 4-10 at precombat main; the shape is the exact "
                          "(bodies, lands, commander) tuple, so the most common one is the modal board"),
    "death": (crit_death, "our last own precombat main before elimination"),
    "held": (crit_held, "a cleanup of our own turn with >= N lands untapped and nothing cast "
                        "or activated that turn — a held hand"),
    "pre-wipe": (crit_pre_wipe, "the start of a turn on which the table lost a wipe's worth of permanents"),
    "lethal-missed": (crit_lethal_missed, "our precombat main where an opponent's life is at or "
                                          "under our untapped printed power and we did not attack"),
    "first-attack": (crit_first_attack, "the beginning of combat of our first attack"),
}


def facts_all(game, cmd):
    """`game_facts` per seat, memoised per game object."""
    key = id(game)
    cache = facts_all.__dict__.setdefault("_cache", {})
    if key not in cache:
        cache[key] = sim_parse.game_facts(game, cmd)["per_seat"]
    return cache[key]


def find(slug, run_id, criterion, params=None):
    """The shortlist: shapes ranked by recurrence, each with an exemplar cut."""
    params = params or {}
    if criterion not in CRITERIA:
        raise SystemExit(f"unknown criterion {criterion!r}; one of {sorted(CRITERIA)}")
    fn, definition = CRITERIA[criterion]
    rec, rows = _games_of(slug, run_id)
    cmd_full = record_commanders(rec)
    cmd = {k: _front(v) for k, v in cmd_full.items()}
    ours_meta = deck_meta_name(slug)
    by_shape = collections.OrderedDict()
    games_seen = 0
    for game_index, outcome, game in rows:
        ours = next((s for s in game["seats"] if s.endswith(f"-{ours_meta}")), None)
        if ours is None:
            continue
        games_seen += 1
        facts = facts_all(game, cmd).get(ours) or {}
        hits = fn(game, ours, cmd, facts, params)
        seen_here = set()
        for h in hits:
            key = json.dumps(h["shape"], sort_keys=True)
            row = by_shape.setdefault(key, {"shape": h["shape"], "games": set(), "hits": 0,
                                             "exemplar": None})
            row["hits"] += 1
            row["games"].add(game_index)
            ex = {"game": game_index, "turn": h["turn"], "phase": h["phase"], "step": h["step"],
                  "outcome": {"winner": outcome.get("winner"), "round": outcome.get("round")},
                  "replay": (f"-n {outcome.get('game_in_job')} -s {outcome.get('seed')}"
                             if outcome.get("seed") else None),
                  "detail": h["detail"]}
            # the exemplar is the first hit, or for `widest` the widest of them
            if row["exemplar"] is None or (criterion == "widest" and
                                           h["detail"].get("bodies", 0) > row["exemplar"]["detail"].get("bodies", 0)):
                row["exemplar"] = ex
    shapes = []
    for key, row in by_shape.items():
        k = len(row["games"])
        lo, hi = wilson(k, games_seen) if games_seen else (None, None)
        shapes.append({"shape": row["shape"], "games": k, "of": games_seen,
                       "share": round(k / games_seen, 3) if games_seen else None,
                       "ci95": [lo, hi], "hits": row["hits"], "exemplar": row["exemplar"]})
    shapes.sort(key=lambda r: (-r["games"], -r["hits"]))
    return {"slug": slug, "run_id": run_id, "criterion": criterion, "params": params,
            "definition": definition, "games": games_seen, "shapes": shapes,
            "limits": ["every figure is the board series' ESTIMATE — see board_series.LIMITS",
                       "recurrence counts GAMES with at least one hit of a shape; a Wilson "
                       "interval on that share is the honesty on 'this happens a lot'"]}


def lift_shortlist(slug, run_id, found, top=1, to_stack=False):
    """Lift the top shapes' exemplars, stamping the finder's provenance."""
    from manamap.sim import bridge
    out = []
    for rank, row in enumerate(found["shapes"][:top], 1):
        ex = row["exemplar"]
        step_text = ex["step"] or ex["phase"]
        path, doc = bridge.lift(slug, run_id, ex["game"], ex["turn"], step_text, to_stack=to_stack,
                                extras={"finder": {"criterion": found["criterion"],
                                                   "params": found["params"], "shape": row["shape"],
                                                   "recurrence": {"games": row["games"], "of": row["of"],
                                                                  "ci95": row["ci95"]},
                                                   "rank": rank}})
        out.append((path, doc))
    return out


def _fmt(found):
    lines = [f"BOARDS — {found['slug']} · {found['run_id'][:60]}",
             f"  {found['criterion']}: {found['definition']}",
             f"  {found['games']} game(s) read; {len(found['shapes'])} shape(s)\n"]
    for i, r in enumerate(found["shapes"][:12], 1):
        ex = r["exemplar"]
        ci = r["ci95"]
        lines.append(f"  {i:2}. {json.dumps(r['shape'])}")
        lines.append(f"      in {r['games']} of {r['of']} games ({r['share']:.0%}, ci95 "
                     f"[{ci[0]}, {ci[1]}]), {r['hits']} hit(s)")
        lines.append(f"      exemplar: game {ex['game']} turn {ex['turn']} {ex['step'] or ex['phase']}"
                     + (f" · replay {ex['replay']}" if ex.get("replay") else "")
                     + f" · winner {ex['outcome'].get('winner')}")
    if len(found["shapes"]) > 12:
        lines.append(f"  … {len(found['shapes']) - 12} more")
    lines.append("")
    for l in found["limits"]:
        lines.append(f"  · {l}")
    return "\n".join(lines)


def main(args):
    slug, run_id = args.slug, args.run
    params = {}
    if getattr(args, "n", None) is not None:
        params["n"] = args.n
    found = find(slug, run_id, args.criterion, params)
    if getattr(args, "as_json", False):
        print(json.dumps(found, indent=2, ensure_ascii=False))
    else:
        print(_fmt(found))
    if getattr(args, "lift", False) or getattr(args, "stack", False):
        for path, doc in lift_shortlist(slug, run_id, found, top=getattr(args, "top", 1) or 1,
                                        to_stack=getattr(args, "stack", False)):
            print(f"  lifted -> {path}")
