"""CAN FORGE'S AI PILOT EVERY SEAT AT THIS TABLE? (2026-10-21)

The pilot, after the finisher-v1 audit: "it has to be able to pilot every single deck
that is playing... we cannot continue this until our forge simulation is working." A
run where any seat cannot play its plan measures a different table — counts and rates
alike describe the AI's refusals, not the decks — so this is the gate `simulate --pod`
reads before the games, and the reading written beside the pod after them.

Two halves, per seat, OURS AND EVERY OPPONENT:

STATIC (no games): every nonland card in the seat's list — is there a script at all;
is it still `AI:RemoveDeck:All` after the repo's overrides (the AI never casts such a
card proactively); which `NeedsToPlay` / `NeedsToPlayVar` gates it carries (listed, not
failed — a gate is a rule, and the hints are how the pilot writes one); the floors the
seat's hints file names (a card the engine has no logic for, kept on purpose).

DYNAMIC (`--run`, a pod run with its logs on disk): from the logs, for every seat —
did the commander resolve, which cards were never cast or held while castable (the
corpus gives every seat its mana values since today), which of the cards the pod names
as the seat's `engine` were cast; for our seat the wheel timing.

THE DEFINITION is written into the audit so the pilot can tighten it, and a seat that
fails it is fixed or the run does not happen. `--anyway` runs a screen and the record
says so, the `cast_proofs` precedent.
"""

import datetime
import json
import re
from pathlib import Path

from manamap import config

SINCE = "2026-10-21"

#: THE DEFINITION — stated on every audit. Tighten here, never in a reader.
DEFINITION = {
    "static": [
        "no nonland card in the seat's list is `AI:RemoveDeck:All` after the repo's overrides",
        "every card in the list has a Forge script",
    ],
    "dynamic": [
        f"the commander resolved in at least {80}% of the games the seat played",
        "no card the pod names as the seat's `engine` was never cast or held while castable",
        "our seat: wheels fired with a commander out at least 90% of the time, and at "
        "least one wheel a game (a wheel deck only)",
    ],
    "floors": "a card named in the seat's hints as a floor is reported and never fails the seat",
}
COMMANDER_RESOLVED_MIN = 0.80
WHEELS_WITH_COMMANDER_MIN = 0.90
HELD_TURNS = 2             # engine_casts.HELD_CASTABLE_TURNS, restated for a seat with no record
NEVER_CAST_IN_HAND_GAMES = 2

_FLAG = re.compile(r"^AI:RemoveDeck:(\w+)", re.M)
_GATE = re.compile(r"^SVar:(NeedsToPlay(?:Var)?):(.+)$", re.M)


def audit_path(pod):
    from manamap.sim import pods
    return pods.PODS_DIR / f"{pod}.audit.json"


def _seat_doc(slug):
    """The seat's cards with type lines: cards.json for ours, the corpus for an opponent."""
    from manamap.sim import forge
    doc = forge._deck_doc(slug)
    if doc is None:
        doc = forge._corpus_doc(slug)
    return doc


def _floors(slug):
    """Cards the seat's hints file names as floors — engine has no logic, kept on purpose."""
    from manamap.sim import forge
    try:
        home = forge.seat_home(slug)
    except Exception:
        return []
    p = Path(home) / "forge_hints.json"
    if not p.is_file():
        return []
    try:
        return list((json.loads(p.read_text(encoding="utf-8")) or {}).get("floors") or [])
    except Exception:                                # pragma: no cover - defensive
        return []


def static_seat(slug):
    """The no-games half for one seat."""
    from manamap.sim import forge, forge_cards
    doc = _seat_doc(slug)
    names = [c["name"] for c in (doc or {}).get("cards") or []] if doc else forge._seat_names(slug)
    lands = {c["name"] for c in (doc or {}).get("cards") or [] if "Land" in str(c.get("type_line") or "")}
    flagged, unfound, gated = [], [], {}
    for name in names:
        if name in lands:
            continue
        text = forge_cards.script(name, installed_copy=True)
        if text is None:
            unfound.append(name)
            continue
        m = _FLAG.search(text)
        if m and m.group(1) == "All":
            flagged.append(name)
        g = [f"{k}:{v.strip()}" for k, v in _GATE.findall(text)]
        if g:
            gated[name] = g
    floors = _floors(slug)
    return {"cards": len(names), "nonland": len(names) - len(lands),
            "flagged": sorted(flagged), "unfound": sorted(unfound),
            "gated": dict(sorted(gated.items())), "floors": floors,
            "source": (doc or {}).get("source", "cards.json") if doc else "decklist only",
            "clean": not flagged and not unfound}


def _find_run(run_id):
    hits = list(config.DECKS_DIR.glob(f"*/sim/{run_id}.json")) + \
        list(config.DECKS_DIR.glob(f"*/branches/*/sim/{run_id}.json"))
    if not hits:
        raise SystemExit(f"no run record named {run_id!r} under {config.DECKS_DIR}")
    return hits[0]


def _logs_for(path, rec):
    log_dir = path.parent / "logs" / path.stem
    if not log_dir.is_dir():
        return []
    return [p.read_text(encoding="utf-8", errors="replace") for p in sorted(log_dir.glob("part-*.log"))]


def dynamic_seats(rec, texts, engine_by_slug=None):
    """The games half, every seat, from a run's logs. {} when the logs are gone."""
    from manamap.sim import forge, parse as sim_parse
    if not texts:
        return {}
    seats = rec.get("seats") or []
    label = forge._seat_label([s["forge_name"] for s in seats])
    slugs = [s["slug"] for s in seats]
    cmc, kws = {}, {}
    for s in slugs:
        doc = forge.played_doc(s, rec) if s == slugs[0] else None
        cmc.update(forge.cmc_map(s, doc))
        kws.update(forge.keyword_map(s, doc))
    facts, analysis = sim_parse.analyze_logs(texts, label, forge.record_commanders(rec), cmc, kws)
    out = {}
    for s in slugs:
        meta = forge.deck_meta_name(s)
        ec = sim_parse.engine_casts(facts, label, meta)
        by = ec.get("by_card") or {}
        held = sorted(n for n, r in by.items()
                      if (r.get("castable_uncast") or 0) >= HELD_TURNS
                      and (r.get("cast", 0) + r.get("activated", 0)) <= 1)
        never = sorted(n for n, r in by.items()
                       if (r.get("cast", 0) + r.get("activated", 0)) == 0
                       and (r.get("in_hand_games") or 0) >= NEVER_CAST_IN_HAND_GAMES)
        ca = ((analysis.get("seats") or {}).get(s) or {}).get("commander_access") or {}
        engine = sorted((engine_by_slug or {}).get(s) or [])
        engine_rows = {n: {"cast": by.get(n, {}).get("cast", 0) + by.get(n, {}).get("activated", 0),
                           "held": n in held, "never_cast": n in never} for n in engine}
        out[s] = {"games": ec.get("games", 0),
                  "commander_resolved_rate": ca.get("resolved_rate"),
                  "commander_resolved_games": ca.get("games_resolved"),
                  "held_while_castable": held, "never_cast": never,
                  "engine": engine_rows}
    wt = rec.get("wheel_timing")
    if wt:
        out[slugs[0]]["wheel_timing"] = {
            "wheels_per_game": (wt["wheels_cast"]["mean_ci"] or {}).get("mean"),
            "with_commander": wt["wheels_with_commander"].get("rate"),
            "first_wheel_turn": (wt["first_wheel_turn"]["mean_ci"] or {}).get("mean")}
    return out


def _verdict(seat, static, dynamic, ours):
    why = []
    if static["flagged"]:
        why.append(f"{len(static['flagged'])} card(s) the AI will not cast: {', '.join(static['flagged'][:4])}"
                   + (" …" if len(static["flagged"]) > 4 else ""))
    if static["unfound"]:
        why.append(f"{len(static['unfound'])} card(s) without a script: {', '.join(static['unfound'][:4])}")
    if dynamic:
        r = dynamic.get("commander_resolved_rate")
        if r is not None and r < COMMANDER_RESOLVED_MIN:
            why.append(f"the commander resolved in {r:.0%} of games (< {COMMANDER_RESOLVED_MIN:.0%})")
        floors = set(static.get("floors") or [])
        bad = [n for n, row in (dynamic.get("engine") or {}).items()
               if (row["never_cast"] or row["held"]) and n not in floors]
        if bad:
            why.append("engine card(s) never cast or held: " + ", ".join(bad))
        wt = dynamic.get("wheel_timing")
        if ours and wt and wt.get("with_commander") is not None:
            if wt["with_commander"] < WHEELS_WITH_COMMANDER_MIN:
                why.append(f"wheels fired with a commander out {wt['with_commander']:.0%} of the time "
                           f"(< {WHEELS_WITH_COMMANDER_MIN:.0%})")
            if not wt.get("wheels_per_game"):
                why.append("no wheel was cast")
    return ("PILOTABLE" if not why else "NOT"), why


def audit(pod, run_id=None):
    """The whole table. `run_id` adds the games half from that run's logs."""
    from manamap.sim import forge, pods
    doc = pods.load(pod)
    engine_by_slug = {s["slug"]: s.get("engine") for s in doc["seats"] if s.get("engine")}
    rec, texts, ours = None, [], None
    if run_id:
        path = _find_run(run_id)
        rec = json.loads(path.read_text(encoding="utf-8"))
        texts = _logs_for(path, rec)
        ours = (rec.get("seats") or [{}])[0].get("slug")
    dyn = dynamic_seats(rec, texts, engine_by_slug) if rec else {}
    seat_slugs = ([ours] if ours else []) + [s["slug"] for s in doc["seats"]]
    rows = []
    for slug in seat_slugs:
        st = static_seat(slug)
        dy = dyn.get(slug)
        verdict, why = _verdict(slug, st, dy, ours=(slug == ours))
        rows.append({"slug": slug, "ours": slug == ours, "static": st, "dynamic": dy,
                     "verdict": verdict, "why": why})
    # THE SAME SHA A RUN RECORD CARRIES (`card_overrides.sha`), so the gate compares
    # like with like; `overrides_tag()` is the run-id suffix and is "" at the default.
    harness = {"overrides": (forge.card_overrides() or {}).get("sha"),
               "telemetry": forge.telemetry_tag()}
    return {"pod": pod, "at": datetime.date.today().isoformat(), "since": SINCE,
            "harness": harness,
            "run": {"id": run_id, "games": (rec or {}).get("summary", {}).get("decided"),
                    "logs_read": len(texts)} if run_id else None,
            "definition": DEFINITION,
            "seats": rows,
            "verdict": "PILOTABLE" if all(r["verdict"] == "PILOTABLE" for r in rows) else "NOT",
            "limits": [
                "STATIC reads the installed scripts: a card that is castable but that the AI "
                "holds for its own reasons shows only in the DYNAMIC half, which needs a run.",
                "DYNAMIC castable counts are lands-only (colours ignored), the engine_casts floor.",
                "An opponent seat is an EDHREC average list unless its decklist was pasted over.",
            ]}


def validate(doc):
    errors = []
    for k in ("pod", "at", "since", "harness", "definition", "seats", "verdict", "limits"):
        if k not in doc:
            errors.append(f"seat audit: missing {k}")
    if errors:
        return errors
    if doc["verdict"] not in ("PILOTABLE", "NOT"):
        errors.append(f"seat audit: verdict {doc['verdict']!r}")
    for r in doc["seats"]:
        if r.get("verdict") not in ("PILOTABLE", "NOT"):
            errors.append(f"seat audit: seat {r.get('slug')} verdict {r.get('verdict')!r}")
        if r.get("verdict") == "NOT" and not r.get("why"):
            errors.append(f"seat audit: seat {r.get('slug')} fails with no reason")
        st = r.get("static") or {}
        if st.get("clean") != (not st.get("flagged") and not st.get("unfound")):
            errors.append(f"seat audit: seat {r.get('slug')} `clean` disagrees with its lists")
    want = "PILOTABLE" if all(r.get("verdict") == "PILOTABLE" for r in doc["seats"]) else "NOT"
    if doc["verdict"] != want:
        errors.append("seat audit: the table's verdict disagrees with its seats")
    return errors


def load(pod):
    p = audit_path(pod)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def gate(pod, anyway=False):
    """Before `simulate --pod`: the audit must exist under THIS harness and read
    PILOTABLE, or the run is refused; `--anyway` runs a screen and returns the
    summary the record carries."""
    from manamap.sim import forge
    doc = load(pod)
    ov = (forge.card_overrides() or {}).get("sha")
    if doc is None:
        msg = (f"{pod}: no seat audit — `manamap pilot forge-seat-audit {pod} --write` says "
               f"whether the AI can pilot every seat; a run without it measures a table "
               f"nobody checked")
    elif (doc.get("harness") or {}).get("overrides") != ov:
        msg = (f"{pod}: the seat audit was written under overrides "
               f"{(doc.get('harness') or {}).get('overrides')} and the engine carries {ov} — "
               f"re-run `forge-seat-audit {pod} --write`")
    elif doc.get("verdict") != "PILOTABLE":
        bad = [f"{r['slug']} ({'; '.join(r['why'])})" for r in doc["seats"] if r["verdict"] != "PILOTABLE"]
        msg = f"{pod}: the AI cannot pilot every seat — " + " · ".join(bad)
    else:
        return {"verdict": "PILOTABLE", "at": doc["at"], "harness": doc["harness"], "anyway": False}
    if not anyway:
        raise SystemExit(msg + "\n  `--anyway` runs it as a SCREEN and the record says so")
    return {"verdict": (doc or {}).get("verdict", "none"), "at": (doc or {}).get("at"),
            "harness": (doc or {}).get("harness"), "anyway": True, "why": msg}


def _print(doc):
    print(f"SEAT AUDIT — {doc['pod']} · {doc['at']} · overrides {doc['harness']['overrides']}"
          + (f" · run {doc['run']['id']} ({doc['run']['games']} games)" if doc.get("run") else " · static only"))
    for r in doc["seats"]:
        st = r["static"]
        mark = "✓" if r["verdict"] == "PILOTABLE" else "✗"
        print(f"  {mark} {r['slug']:22} {'ours' if r['ours'] else 'opp ':4} "
              f"{st['nonland']:3} nonland · flagged {len(st['flagged'])} · unfound {len(st['unfound'])}"
              f" · gated {len(st['gated'])}" + (f" · floors {len(st['floors'])}" if st["floors"] else ""))
        for w in r["why"]:
            print(f"      ! {w}")
        dy = r.get("dynamic")
        if dy:
            cr = dy.get("commander_resolved_rate")
            print(f"      commander resolved {cr if cr is None else f'{cr:.0%}'} · held {len(dy['held_while_castable'])}"
                  f" · never cast {len(dy['never_cast'])}"
                  + (f" · wheels {dy['wheel_timing']['wheels_per_game']}/game, "
                     f"{dy['wheel_timing']['with_commander']:.0%} with a commander" if dy.get("wheel_timing") else ""))
            if dy["held_while_castable"]:
                print(f"      held: {', '.join(dy['held_while_castable'][:8])}")
    print(f"\n  {doc['verdict']} — " + ("every seat can be piloted under the definition above"
                                         if doc["verdict"] == "PILOTABLE" else
                                         "fix the seats named, or the run does not happen"))


def main(args):
    pod = args.pod
    doc = audit(pod, run_id=getattr(args, "run", None))
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2, ensure_ascii=False))
    else:
        _print(doc)
    if getattr(args, "write", False):
        errs = validate(doc)
        if errs:
            raise SystemExit("\n".join(errs))
        audit_path(pod).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  wrote {audit_path(pod)}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot forge-seat-audit <pod>`.")
