"""Pilot: the workbench view — one deck, one screen, and what to do next.

WHY THIS EXISTS. Seven commands each answer a piece of "where is this deck":
`deck-status` (is it finished, is it stale), `deck-version` (which list is this),
`deck-notes` (what happened at the table), `prescribe --list` (what was asked and
answered), `deck-audit` (is it any good), `goldfish` (how fast), `bracket-check`
(how strong). A pilot sitting down to tinker ran four of them and held the join in
their head. This is the join, composed from those modules and the tracked
artifacts — it computes nothing new, so it can never disagree with the command
that owns each figure.

THE `next` BLOCK IS THE POINT. A dashboard tells you the state; a workbench tells
you the move. Every suggestion here is DERIVED from a condition that is true right
now (un-debriefed games → `/debrief`; an uncommitted working list → commit it so
the log can stamp a version; a stale stage → the command that regenerates it; no
games logged → play it), and each names the command. None of it is judgment about
the deck — that is the doctor's, behind `/prescribe`.

Read-only and computed on demand. `--json` is the shape a future UI reads; the human
print is the same dict laid out.

`--write` puts that shape on disk as `info.json` so the deck page can fetch it, and
that artifact IS committed — unlike `deck-facts`, which stays uncommitted because
nothing reads it but an agent standing right there. Two consequences, both handled:
it is staleness-gated like `mana_analysis.json` (regenerate after a decklist change or
the test fails), and it OMITS the version block. Versions are derived from a git walk,
so a committed copy is one commit behind forever — the deck page gets them from a
deploy-time `versions.json` instead, which CI can build because `deck_versions` needs
only git while `deck_audit` needs the gitignored corpus.
"""

import json

from manamap.pilot import deck_facts as facts_mod
from manamap.pilot import deck_status as status_mod
from manamap.pilot import deck_versions as versions_mod
from manamap.pilot.common import UNPLAYABLE_STATUSES, deck_dir, deck_lifecycle, load_json
from manamap.pilot.deck_notes import annotations, read_log
from manamap.pilot.prescribe import list_all as prescriptions_of
from manamap.sim.experiment import list_all as experiments_of
from manamap.sim.forge import list_runs as sim_runs


def _pct(x):
    return None if x is None else round(100 * float(x))


def _goldfish(base):
    doc = load_json(base / "goldfish_metrics.json")
    if not doc:
        return None
    m = doc.get("metrics") or {}
    cmd = m.get("commander") or {}
    out = {"seed": (doc.get("meta") or {}).get("seed"),
           "iterations": m.get("iterations"),
           "commander_mean_cast_turn": cmd.get("mean_cast_turn"),
           "commander_cast_by_turn_6_pct": _pct(cmd.get("cast_by_turn_6_rate")),
           "targets": [{"label": t.get("label"), "by_turn_6_pct": _pct(t.get("by_turn_6_rate"))}
                       for t in (m.get("targets") or [])]}
    # The clock, when the deck opted into `model_combat`. Absent otherwise, which
    # is every deck that has not been deliberately re-baselined — see the opt-in
    # contract in `goldfish.py`.
    #
    # THIS BLOCK HAD NEVER EXECUTED. It asked for `kill_by_turn_8_pct`,
    # `kill_turn_distribution` and `never_by_turn_10_pct`; the producer emits
    # `kill_by_turn_rate`, `kill_turn_histogram` and `no_kill_by_max_turn_rate`.
    # Not one key matched, so the first deck to opt in would have written
    # `"combat": {}` into a tracked `info.json` and rendered nothing — dead code
    # that no test could catch, because no artifact has ever carried a combat
    # block for it to run against.
    combat = m.get("combat") or {}
    if combat:
        max_turn = (doc.get("meta") or {}).get("max_turn")
        by_turn = combat.get("kill_by_turn_rate") or {}
        out["combat"] = {
            "mean_kill_turn": combat.get("mean_kill_turn"),
            "median_kill_turn": combat.get("median_kill_turn"),
            "kill_by_turn_6_pct": _pct(by_turn.get("6")),
            "kill_by_turn_8_pct": _pct(by_turn.get("8")),
            # Named from the artifact rather than hardcoded: the producer's rate is
            # over GOLDFISH_MAX_TURN, and a key saying "by_turn_10" would quietly
            # lie the moment that constant moved.
            "max_turn": max_turn,
            "no_kill_by_max_turn_pct": _pct(combat.get("no_kill_by_max_turn_rate")),
            "kill_turn_histogram": combat.get("kill_turn_histogram"),
        }
    return out


def _binding_axes(slug):
    """The audit's under/over axes — what the measurement says binds. Never a score."""
    try:
        from manamap.pilot import deck_audit
        doc = deck_audit.analyze(slug)
    except Exception:                      # noqa: BLE001 — an optional panel, never a gate
        return None
    axes = doc.get("axes") or []
    arch = doc.get("archetype")
    if isinstance(arch, dict):              # detect_archetype returns its evidence too
        arch = arch.get("archetype")
    return {"archetype": arch,
            "under": [a["axis"] for a in axes if a.get("verdict") == "under"],
            "over": [a["axis"] for a in axes if a.get("verdict") == "over"],
            "stale": [k for k, v in (doc.get("freshness") or {}).items()
                      if isinstance(v, dict) and v.get("state") not in (None, "current")]}


def _open_questions(base):
    out = []
    for name, key in (("engine.json", "engine"), ("diagnosis.json", "diagnosis")):
        doc = load_json(base / name) or {}
        for q in doc.get("open_questions") or []:
            out.append({"from": key, "question": q.get("question"),
                        "settled_by": q.get("settled_by")})
    for eid, note in sorted((load_json(base / "log_annotations.json") or {}).get("entries", {}).items()):
        for q in note.get("open_questions") or []:
            out.append({"from": f"log:{eid}", "question": q.get("question"),
                        "settled_by": q.get("settled_by")})
    return out


def compose(slug):
    base = deck_dir(slug)
    facts = facts_mod.analyze(slug)
    counts = facts.get("counts") or {}
    identity = sorted({c for v in (facts.get("colours") or {}).values()
                       for c in (v.get("card") or [])}, key="WUBRG".index)
    rows = status_mod.status(slug, validate=True)
    vdoc = versions_mod.report(slug)
    log = read_log(slug)
    done = annotations(slug)
    rx = prescriptions_of(slug)
    bracket = load_json(base / "bracket_report.json") or {}
    engine = load_json(base / "engine.json") or {}
    diag = load_json(base / "diagnosis.json") or {}

    cur = next((v for v in vdoc["versions"] if v["version"] == vdoc["current_version"]), None)
    record = {"games": len(log),
              "win": sum(1 for e in log if e.get("result") == "win"),
              "loss": sum(1 for e in log if e.get("result") == "loss"),
              "draw": sum(1 for e in log if e.get("result") == "draw"),
              "last_played": max((e["at"][:10] for e in log), default=None),
              "undebriefed": [e["id"] for e in log if e["id"] not in done]}
    lines = engine.get("lines") or []
    life = deck_lifecycle(slug)
    info = {
        "slug": slug,
        "commander": facts.get("commander"),
        "lifecycle": ({"status": life[0], "headline": life[1], "body": life[2]}
                      if life else None),
        "colour_identity": identity,
        "size": counts.get("copies"), "lands": counts.get("land_copies"),
        "version": {"current": vdoc["current_version"], "of": len(vdoc["versions"]),
                    "date": cur["first_date"] if cur else None,
                    "tags": cur["tags"] if cur else [],
                    "uncommitted": vdoc["current_version"] is None and bool(vdoc["versions"])},
        "status": {"complete": sum(1 for r in rows if r["state"] == "present"),
                   "of": len(rows),
                   "stale": [r["stage"] for r in rows if r["state"] == "STALE"],
                   "invalid": [r["stage"] for r in rows if r["state"] == "INVALID"],
                   "missing": [r["stage"] for r in rows if r["state"] == "missing"]},
        "bracket": {"floor": bracket.get("floor"), "floor_name": bracket.get("floor_name"),
                    "target": bracket.get("target"),
                    "within_target": bracket.get("within_target")} if bracket else None,
        "record": record,
        "goldfish": _goldfish(base),
        "engine": {"thesis": engine.get("thesis"),
                   "critic": (engine.get("critic") or {}).get("verdict"),
                   "lines": len(lines),
                   "verified_lines": sum(1 for l in lines if l.get("verified_by"))} if engine else None,
        "diagnosis": {"verdict": diag.get("verdict"),
                      "skeptic": (diag.get("skeptic") or {}).get("verdict"),
                      "stale": diag.get("as_of_decklist_sha256") not in
                      (None, (load_json(base / "cards.json") or {}).get("decklist_sha256"))}
        if diag else None,
        "audit": _binding_axes(slug),
        "prescriptions": {"count": len(rx),
                          "answered": sum(1 for p in rx if "add_candidates" in p),
                          "latest": ({"id": rx[-1]["id"], "prompt": rx[-1]["prompt"],
                                      "adds": len(rx[-1].get("add_candidates") or []),
                                      "cuts": len(rx[-1].get("cut_candidates") or []),
                                      "skeptic": (rx[-1].get("skeptic") or {}).get("verdict")}
                                     if rx else None)},
        "open_questions": _open_questions(base),
        "simulation": _simulation(slug),
        "experiments": _experiments(slug),
    }
    info["next"] = _next(info)
    return info


def _simulation(slug):
    runs = sim_runs(slug)
    if not runs:
        return None
    r = runs[-1]
    me = (r.get("analysis") or {}).get("seats", {}).get(slug, {})
    tok = me.get("tokens") or {}
    return {"runs": len(runs), "latest": r["run_id"], "at": r.get("at"),
            "games": r.get("games_completed"),
            "vs": [s["slug"] for s in r.get("seats", [])[1:]],
            "win_rate": me.get("win_rate"), "win_rate_ci95": me.get("win_rate_ci95"),
            "eliminated_by": me.get("eliminated_by"),
            "mean_round": (r.get("summary") or {}).get("mean_round"),
            "token_damage_share": (tok.get("token_damage_share") or {}).get("mean"),
            "tokens_observed": (tok.get("tokens_observed") or {}).get("mean")}


def _experiments(slug):
    docs = experiments_of(slug)
    if not docs:
        return None
    d = docs[-1]
    w = d["delta"]["win_rate"]
    power = d["delta"].get("power") or {}
    # `overlap` is gone rather than deprecated. It named the overlap fallacy —
    # two marginal intervals overlapping says nothing about their difference —
    # and leaving the key in the artifact re-invites the error it was removed
    # for. What replaces it is the interval on the DIFFERENCE, and the effect
    # size this experiment could have detected at all.
    return {"count": len(docs), "latest": {
        "question": d["question"], "at": d["at"],
        "games_per_arm": d["games_per_arm"], "win_a": w["a"], "win_b": w["b"],
        "win_diff_ci95": w.get("ci95_diff"),
        "differs": w.get("excludes_zero"),
        "minimum_detectable_difference": power.get("minimum_detectable_difference"),
        "reading": d["delta"]["reading"]}}


def _next(info):
    """Each suggestion is derived from a condition true right now and names the
    command. No judgment about the deck lives here."""
    nxt = []
    slug = info["slug"]
    # A deck that has been pulled apart cannot be shuffled, so every suggestion
    # that ends in "play it" or "measure it before you play it" is an
    # instruction the pilot cannot follow. Say the status instead, and say what
    # is being withheld — a silently shorter list reads as "nothing to do here",
    # which is a different claim. What survives is everything that still works
    # on a published record: a failing gate, a stale artifact, an open rules
    # question. `superseded` is NOT in this set — that list is still sleeved.
    closed = (info["lifecycle"] or {}).get("status") in UNPLAYABLE_STATUSES
    if closed:
        nxt.append(f"{info['lifecycle']['headline'].lower()} — the play/measure loop is "
                   f"closed for this deck ({info['lifecycle']['status']}); its artifacts "
                   f"stay as published. Suggestions to log a game, simulate or run an "
                   f"experiment are withheld")
    if info["status"]["invalid"]:
        nxt.append(f"{len(info['status']['invalid'])} artifact(s) fail their own gate "
                   f"({', '.join(info['status']['invalid'])}) — `deck-status {slug}` names them; "
                   f"fix before anything else reads them")
    if info["status"]["stale"]:
        nxt.append(f"stale: {', '.join(info['status']['stale'])} — regenerate against the "
                   f"current list (`/publish-deck` sequences it)")
    if info["version"]["uncommitted"]:
        nxt.append("decklist.txt differs from every committed version — commit it so the "
                   "log can stamp games against a version (`deck-version` will then show it)")
    if info["record"]["games"] == 0 and not closed:
        nxt.append(f"nothing in the captain's log — play it, then "
                   f"`deck-notes {slug} add \"…\" --result win|loss --opponents N`")
    elif info["record"]["undebriefed"]:
        ids = info["record"]["undebriefed"]
        nxt.append(f"{len(ids)} logged game(s) not yet debriefed ({', '.join(ids)}) — `/debrief {slug}`")
    if info["prescriptions"]["count"] > info["prescriptions"]["answered"]:
        nxt.append(f"{info['prescriptions']['count'] - info['prescriptions']['answered']} open "
                   f"prescription(s) — `/prescribe {slug}` runs the doctor ⇄ skeptic loop")
    if info["diagnosis"] and info["diagnosis"]["stale"]:
        nxt.append(f"diagnosis.json describes an older list — `/diagnose-deck {slug}`")
    if info["engine"] and info["engine"]["critic"] != "pass":
        nxt.append(f"engine.json's critic verdict is {info['engine']['critic']!r} — `/analyze-engine {slug}`")
    routed = [q for q in info["open_questions"] if q.get("settled_by")]
    if routed:
        by = {}
        for q in routed:
            by.setdefault(q["settled_by"], 0)
            by[q["settled_by"]] += 1
        nxt.append("open questions routed: " + ", ".join(f"{k} ×{v}" for k, v in sorted(by.items())))
    if info["simulation"] is None and not closed:
        nxt.append(f"no simulation runs — `simulate {slug} --vs <opp> [--vs …] --games N` "
                   f"(Forge; ◆ seeded)")
    elif info["experiments"] is None and info["version"]["of"] > 1 and not closed:
        nxt.append(f"{info['version']['of']} versions and no experiment — "
                   f"`experiment {slug} --a V<n> --b working --vs <pod> --games N` measures a swap")
    if info["bracket"] and info["bracket"].get("within_target") is False:
        nxt.append(f"bracket floor {info['bracket']['floor']} exceeds target "
                   f"{info['bracket']['target']} — `bracket-check {slug}` names the drivers")
    if not nxt:
        nxt.append("nothing outstanding — ask the doctor something (`/prescribe`) or play it")
    elif closed and len(nxt) == 1:
        nxt.append("nothing else outstanding — this deck's record is complete as it stands")
    return nxt


def _print(info):
    ci = "".join(info["colour_identity"]) or "C"
    print(f"WORKBENCH — {info['slug']} · {' / '.join(info['commander'] or [])} · {ci} · "
          f"{info['size']} cards ({info['lands']} lands)")
    if info["lifecycle"]:
        print(f"  ⚑ {info['lifecycle']['headline']} — {info['lifecycle']['body']}")
    print()
    v = info["version"]
    vline = (f"V{v['current']} of {v['of']} · {v['date']}" if v["current"]
             else (f"uncommitted working list ({v['of']} committed)" if v["of"] else "no git history"))
    if v["tags"]:
        vline += f" · [{', '.join(v['tags'])}]"
    print(f"  version    {vline}")
    s = info["status"]
    sline = f"{s['complete']}/{s['of']} stages"
    if s["stale"]:
        sline += f" · STALE: {', '.join(s['stale'])}"
    if s["invalid"]:
        sline += f" · INVALID: {', '.join(s['invalid'])}"
    if info["bracket"]:
        b = info["bracket"]
        sline += (f" · bracket floor {b['floor']}" + (f" ({b['floor_name']})" if b.get("floor_name") else "")
                  + (f" · target {b['target']} {'✓' if b.get('within_target') else '✗'}"
                     if b.get("target") else ""))
    print(f"  status     {sline}")
    r = info["record"]
    rline = (f"{r['games']} game(s) · {r['win']}W {r['loss']}L" + (f" {r['draw']}D" if r["draw"] else "")
             + (f" · last {r['last_played']}" if r["last_played"] else "")
             + (f" · {len(r['undebriefed'])} un-debriefed" if r["undebriefed"] else ""))
    print(f"  record     {rline}")
    g = info["goldfish"]
    if g:
        gl = f"commander mean t{g['commander_mean_cast_turn']} ({g['commander_cast_by_turn_6_pct']}% by t6)"
        if g.get("combat"):
            gl += f" · combat: {g['combat']}"
        print(f"  goldfish   {gl}")
        for t in g["targets"][:3]:
            print(f"             {t['by_turn_6_pct']}% by t6  {str(t['label'])[:70]}")
    if info["engine"]:
        e = info["engine"]
        print(f"  engine     {e['verified_lines']}/{e['lines']} lines verified · critic {e['critic']}")
        print(f"             {str(e['thesis'])[:100]}")
    if info["audit"]:
        a = info["audit"]
        print(f"  audit      {a.get('archetype')} · under: {', '.join(a['under']) or '—'} · "
              f"over: {', '.join(a['over']) or '—'}")
    if info["diagnosis"]:
        d = info["diagnosis"]
        print(f"  diagnosis  skeptic {d['skeptic']}{' · STALE' if d['stale'] else ''}")
        print(f"             {str(d['verdict'])[:100]}")
    sm = info["simulation"]
    if sm:
        print(f"  simulated  {sm['runs']} run(s) · latest {sm['games']} games vs {', '.join(sm['vs'])} · "
              f"win {sm['win_rate']} ci95 {sm['win_rate_ci95']} · mean round {sm['mean_round']} · "
              f"eliminated by {sm['eliminated_by']} · token dmg share {sm['token_damage_share']}")
    xp = info["experiments"]
    if xp:
        l = xp["latest"]
        print(f"  tested     {xp['count']} experiment(s) · latest {l['question'][:70]}")
        print(f"             win {l['win_a']} → {l['win_b']} over {l['games_per_arm']}/arm · {l['reading'][:80]}")
    p = info["prescriptions"]
    if p["count"]:
        lt = p["latest"]
        print(f"  asked      {p['count']} prescription(s), {p['answered']} answered · latest "
              f"{lt['id']}: {lt['prompt'][:60]} → {lt['adds']} add / {lt['cuts']} cut")
    if info["open_questions"]:
        print(f"  questions  {len(info['open_questions'])} open")
        for q in info["open_questions"][:5]:
            print(f"             [{q['from']} → {q['settled_by']}] {str(q['question'])[:80]}")
    print("\n  NEXT")
    for n in info["next"]:
        print(f"   · {n}")


# Everything except the version block, which cannot be committed accurately.
def fetchable(info):
    """`info` minus what a committed artifact would misreport.

    `version` is derived by walking git, and the commit that changes `decklist.txt`
    receives its sha AFTER anything written in the same commit — so a committed copy
    names the previous version forever. It is dropped rather than frozen, and the
    page reads `versions.json` (built at deploy time) instead. A wrong version number
    is worse than an absent one: the log stamps games against it.
    """
    out = {k: v for k, v in info.items() if k != "version"}
    out["_note"] = ("Written by `deck-info --write`. The version block is deliberately "
                    "absent: it is a git walk, and a committed copy is one commit "
                    "behind forever. The deck page reads versions.json instead.")
    return out


def main(args):
    info = compose(args.slug)
    if getattr(args, "write", False):
        path = deck_dir(args.slug) / "info.json"
        path.write_text(json.dumps(fetchable(info), indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote {path}")
        return
    if getattr(args, "as_json", False):
        print(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        _print(info)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot deck-info <slug>`.")
