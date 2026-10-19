"""Pilot: the deck rating — ★ coaching over ◆ inputs, within a bracket, 0–100.

WHAT THIS IS. A scorecard: every component is a measured figure the bench
already publishes (the uniform-harness benchmark, the cited audit axes, the
pilot's own record and survey) mapped to 0–100 by its placement within the
deck's bracket group and by an absolute anchor quoted from the strategy
companion, weighted by the table below, plus ONE judgment component an agent
fills (`deck-rater`) and an adversarial verifier attacks (`rating-skeptic`).
The result is a score, a grade, and a rank within the bracket — printed with
its group and its n, and claiming no significance.

WHAT THIS IS NOT, AND WHY THE LABEL. The bench has refused a composite deck
score at the ◆ tier three times — `benchmark.py` keeps `score: None` ("a
formula picked before the distribution is a guess wearing a number"),
`diagnostic.py` ranks nothing against another deck (speed spans 400x and ranks
a combo deck last for not attacking), `calibrate.py` refuses a verdict below
`MIN_DECKS` decks × `MIN_GAMES` games — and nothing here overturns that. The
rating is ★: `tier: "★"`, `RATING_LABEL` on the artifact, every input figure
traced to its file with its interval and n, and the composite never gates
anything. `rating --calibrate` checks it against outcomes with the same
refusal.

ABSENT MEANS ABSENT. A component with no usable input is WITHHELD — named in
`withheld[]` with its reason — and the weights are renormalised over what is
present. Never zero. Today `table_record`, `pilot_feel` and `forge` are
withheld on every deck, and the skeleton says so; the value grows with logged
games and surveys.

THE JUDGMENT IS A WEIGHTED COMPONENT, NOT A ±ADDER: a 0–100 sub-score at a
fixed 0.20 weight is bounded by construction (±20 of the composite at most),
visible beside the measured components, and renormalises with them. ITS NEUTRAL
POINT IS `code_score`, NOT 50 (2026-10-19): a judgment equal to the measured
score leaves the composite where the components put it; 50 would drag every
deck above 50 down and every deck below it up, which the first fleet pass
showed. The skeptic attacks the number against the evidence; the code bounds
its reach.

THE SKELETON IS RECOMPUTED, THE JUDGMENT IS CARRIED. `--write` rebuilds every
code component and keeps an existing judgment only when it was written
against this same list and this same spec (`judgment.against`); otherwise the
judgment is dropped into `withheld[]` with the sha reason. The captain's-log
rule: the agent writes prose, never re-derives a figure.
"""

import hashlib
import json
import statistics

from manamap import config
from manamap.pilot import calibrate, deck_audit
from manamap.pilot import fleet_analytics as fa
from manamap.pilot.common import deck_dir, load_json, measured_sha
from manamap.pilot.deck_notes import read_log
from manamap.sim.stats import wilson_bounds

RATING_FILE = "rating.json"
RATING_SPEC_VERSION = 1
RATING_MIN_GROUP = 3           # below this, within-bracket placement is withheld
RATING_MIN_LOG_GAMES = 5       # table_record needs this many logged games
RATING_MIN_SURVEY_GAMES = 3    # pilot_feel needs this many surveyed games
RATING_ARGUMENT_MAX_WORDS = 250
RATING_LABEL = ("This rating is ★ coaching: a labelled judgment over ◆ measured inputs. "
                "It ranks within a bracket of {n} decks and claims no significance.")
TIER = "★"

#: The weights, summing to 1.0 over the full set; renormalised over what is
#: present. `forge` is 0.00 by the pilot's rule (2026-10-18): not a power read
#: until the harness work is done. `judgment` is the agent's, fixed at 0.20.
WEIGHTS = {
    "speed": 0.15, "mana": 0.10, "resilience": 0.15, "construction": 0.15,
    "table_record": 0.15, "pilot_feel": 0.10, "forge": 0.00, "judgment": 0.20,
}
GRADES = (("A", 80), ("B", 65), ("C", 50), ("D", 35), ("F", 0))
FAIR_SHARE = 0.25              # a four-seat table's null, `sim/pods.calibration`

#: The audit axes the construction component reads: those with a band from the
#: strategy companion. `colour-sources` is deliberately out — its figure is a
#: deficit count, not a count against a band — as are the informational axes.
CONSTRUCTION_AXES = ("mana-base", "ramp", "card-advantage", "interaction", "sweepers",
                     "protection", "threat-density")

#: The bracket's expected turn, as the benchmark can measure it (6, 8, 10).
EXPECTED_KILL_TURN = {1: 10, 2: 10, 3: 8, 4: 6, 5: 6}
SPEED_ANCHOR_RATE = 0.5        # kill by the bracket's turn in half the games scores 100

LIMITS_FIXED = [
    "A rating is ★ coaching over ◆ inputs; the score never gates. The floor "
    "(bracket_report.json) is the only bracket gate.",
    f"calibrate.py refuses a verdict below {calibrate.MIN_DECKS} decks with "
    f"{calibrate.MIN_GAMES} games each; no figure here is checked against outcomes yet "
    "(`rating --calibrate` reports the refusal with today's n).",
    "Goldfish kill is one column, never the rule: the benchmark is an unopposed ceiling "
    "with no blockers and no removal, and a combo or control deck reads slow for not attacking.",
    "Forge is not a power read until the harness work is done (the pilot's rule, "
    "2026-10-18): its component ships at weight 0.00 and is shown only for a run on the "
    "current list.",
    "DeckCheck's Threat Index is shown in `context` and never scored.",
    "Diagnostic engine figures are read against each deck's own declaration and are never "
    "ranked across decks; they are evidence for the argument, not a component.",
]


# ── Primitives ───────────────────────────────────────────────────────────


def spec_sha():
    """The scorecard's version: every constant that moves a number, hashed."""
    spec = {"version": RATING_SPEC_VERSION, "weights": WEIGHTS, "grades": GRADES,
            "min_group": RATING_MIN_GROUP, "min_log_games": RATING_MIN_LOG_GAMES,
            "min_survey_games": RATING_MIN_SURVEY_GAMES, "fair_share": FAIR_SHARE,
            "construction_axes": CONSTRUCTION_AXES, "expected_kill_turn": EXPECTED_KILL_TURN,
            "speed_anchor_rate": SPEED_ANCHOR_RATE, "label": RATING_LABEL}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def band_score(value, low, high):
    """100 inside `[low, high]`; outside, 100·max(0, 1 − d/span) with `span =
    max(high − low, 2)`; a floor-only band scores 100 above it. Monotone, so an
    interval maps by evaluating its two ends."""
    if value is None or (low is None and high is None):
        return None
    if high is None:
        if value >= low:
            return 100.0
        return clamp(100.0 * max(0.0, 1.0 - (low - value) / max(low, 1e-9)))
    if low is None:
        if value <= high:
            return 100.0
        return clamp(100.0 * max(0.0, 1.0 - (value - high) / max(high, 1e-9)))
    span = max(high - low, 2)
    if low <= value <= high:
        return 100.0
    d = (low - value) if value < low else (value - high)
    return clamp(100.0 * max(0.0, 1.0 - d / span))


def placement(values, slug, higher_is_better=True):
    """`(percentile 0-100, ties_within_ci)` within the group, ties averaged;
    `(None, None)` below `RATING_MIN_GROUP` or when the deck has no value."""
    vals = {s: v for s, v in values.items() if v is not None}
    if slug not in vals or len(vals) < RATING_MIN_GROUP:
        return None, None
    mine = vals[slug]
    others = [v for s, v in vals.items() if s != slug]
    below = sum(1 for v in others if ((v < mine) if higher_is_better else (v > mine)))
    equal = sum(1 for v in others if v == mine)
    rank_from_bottom = below + equal / 2.0
    return round(100.0 * rank_from_bottom / (len(vals) - 1), 2), None


def grade_of(score):
    for g, floor in GRADES:
        if score >= floor:
            return g
    return "F"


# ── Figures per deck ─────────────────────────────────────────────────────


def _fleet():
    doc = fa.load()
    if not doc:
        raise SystemExit(f"no {config.FLEET_ANALYTICS_PATH.name} — `manamap pilot fleet-analytics --write` first")
    return doc


def _row(fleet, slug):
    for r in fleet["decks"]:
        if r["slug"] == slug:
            return r
    raise SystemExit(f"{slug} is not in the fleet artifact — `manamap pilot fleet-analytics --write`")


def group_for(fleet, slug):
    """`(key, source, members)` — the bracket group the deck competes in. The
    fleet is the SLEEVED decks: a bench or brewing deck is refused here."""
    r = _row(fleet, slug)
    if not r.get("member"):
        raise SystemExit(f"{slug} is not a fleet member (sleeved, bracketed, Commander, not apart) — "
                         f"the rating ranks within the sleeved fleet; sleeve it (`deck-version {slug} paper`) first")
    b = r["bracket"]
    key = None
    if b.get("group") and b["group"] != "unbracketed":
        key = int(b["group"][1:])
    members = (fleet["groups"].get(b["group"]) or {}).get("members") or []
    return key, b.get("group_basis"), members


def _input(figure, cell, source=None):
    """One traced input: the figure's value, interval, n, source and sha."""
    return {"figure": figure, "value": cell.get("value"), "ci95": cell.get("ci95"),
            "n": cell.get("n"), "source": source or cell.get("source"),
            "decklist_sha256": cell.get("decklist_sha256")}


def _audit(slug, cache):
    if slug not in cache:
        try:
            cache[slug] = deck_audit.analyze(slug)
        except Exception as exc:  # noqa: BLE001 — an audit that cannot run is an absence
            cache[slug] = {"_error": f"{type(exc).__name__}: {exc}"}
    return cache[slug]


def _axis(audit, name):
    for a in (audit or {}).get("axes") or []:
        if a.get("axis") == name:
            return a
    return None


def _record(slug):
    log = read_log(slug)
    games = [e for e in log if e.get("result") in ("win", "loss", "draw")]
    wins = sum(1 for e in games if e["result"] == "win")
    opp = [1.0 / (e["opponents"] + 1) for e in games if isinstance(e.get("opponents"), int) and e["opponents"] > 0]
    par = sum(opp) / len(opp) if opp else FAIR_SHARE
    return {"games": len(games), "wins": wins, "par": round(par, 4)}


def _component(name, *, sub, sub_ci, place, anchor, method, inputs, citations=()):
    return {"name": name, "weight": WEIGHTS[name], "weight_effective": None,
            "sub_score": round(sub, 2), "sub_ci": [round(sub_ci[0], 2), round(sub_ci[1], 2)],
            "placement": None if place is None else round(place, 2),
            "anchor": None if anchor is None else round(anchor, 2),
            "method": method, "inputs": inputs, "citations": list(citations)}


def _combine(place, anchor, anchor_lo=None, anchor_hi=None):
    """Half placement, half anchor when both exist; the one present otherwise."""
    if place is not None and anchor is not None:
        lo = 0.5 * place + 0.5 * (anchor if anchor_lo is None else anchor_lo)
        hi = 0.5 * place + 0.5 * (anchor if anchor_hi is None else anchor_hi)
        return 0.5 * place + 0.5 * anchor, (min(lo, hi), max(lo, hi)), "0.5·placement + 0.5·anchor"
    if place is not None:
        return place, (place, place), "placement only (no anchor)"
    if anchor is not None:
        lo = anchor if anchor_lo is None else anchor_lo
        hi = anchor if anchor_hi is None else anchor_hi
        return anchor, (min(lo, hi), max(lo, hi)), "anchor only (placement withheld)"
    return None, None, None


def components(slug, fleet, audit_cache=None):
    """Every code component for `slug`: `(components[], withheld[], context)`."""
    audit_cache = audit_cache if audit_cache is not None else {}
    r = _row(fleet, slug)
    key, _basis, members = group_for(fleet, slug)
    rows = {m: _row(fleet, m) for m in members}
    comps, withheld = [], []
    bench, diag = r["benchmark"], r["diagnostic"]
    audit = _audit(slug, audit_cache)

    def grp(block, k):
        return {m: fa._measure_value(rows[m], (block, k)) for m in members}

    # speed ─────────────────────────────────────────────────────────────
    k8 = None if fa.is_absent(bench) else bench.get("kill_by_turn_8")
    if fa.is_fig(k8):
        place, _ = placement(grp("benchmark", "kill_by_turn_8"), slug, True)
        t = EXPECTED_KILL_TURN.get(key, 8)
        kt = bench.get(f"kill_by_turn_{t}")
        anchor = anchor_lo = anchor_hi = None
        if fa.is_fig(kt):
            anchor = clamp(100.0 * kt["value"] / SPEED_ANCHOR_RATE)
            if kt.get("ci95"):
                anchor_lo = clamp(100.0 * kt["ci95"][0] / SPEED_ANCHOR_RATE)
                anchor_hi = clamp(100.0 * kt["ci95"][1] / SPEED_ANCHOR_RATE)
        sub, ci, how = _combine(place, anchor, anchor_lo, anchor_hi)
        if sub is not None:
            comps.append(_component(
                "speed", sub=sub, sub_ci=ci, place=place, anchor=anchor,
                method=(f"{how}: placement of kill_by_turn_8 within bracket {key} (n={len(members)}); "
                        f"anchor kill_by_turn_{t} / {SPEED_ANCHOR_RATE} for a bracket-{key} clock "
                        f"(config.BRACKETS expected turns); Wilson 95% from the benchmark's n"),
                inputs=[_input("kill_by_turn_8", k8)] + ([_input(f"kill_by_turn_{t}", kt)] if fa.is_fig(kt) and t != 8 else []),
                citations=[{"kind": "strategy", "ref": "strategy:deckbuilding.power-level",
                            "quote": "the sharpest line between them is expected game length"}]))
        else:
            withheld.append({"component": "speed", "reason": "no placement and no anchor"})
    else:
        withheld.append({"component": "speed", "reason": "no benchmark kill_by_turn_8 figure"})

    # mana ──────────────────────────────────────────────────────────────
    mld = None if fa.is_absent(bench) else bench.get("missed_land_drop_rate")
    cons = _axis(audit, "consistency")
    if fa.is_fig(mld) or cons:
        place = placement(grp("benchmark", "missed_land_drop_rate"), slug, False)[0] if fa.is_fig(mld) else None
        anchor = None
        cites, inputs = [], []
        if fa.is_fig(mld):
            inputs.append(_input("missed_land_drop_rate", mld))
        if cons and cons.get("measured", {}).get("value") is not None:
            tv = cons["target"]
            anchor = band_score(cons["measured"]["value"], tv.get("low"), tv.get("high"))
            inputs.append({"figure": "audit.consistency", "value": cons["measured"]["value"], "ci95": None,
                           "n": None, "source": "deck-audit#consistency (" + cons["measured"].get("how", "") + ")",
                           "decklist_sha256": measured_sha(slug)})
            cites.append({"kind": "strategy", "ref": tv.get("source"), "quote": tv.get("quote")})
        sub, ci, how = _combine(place, anchor)
        if sub is not None:
            comps.append(_component(
                "mana", sub=sub, sub_ci=ci, place=place, anchor=anchor,
                method=f"{how}: placement of missed_land_drop_rate (lower is better) within bracket {key}; "
                       f"anchor band_score(turn-3 land-drop rate, low 0.74) from the audit's consistency axis",
                inputs=inputs, citations=cites))
        else:
            withheld.append({"component": "mana", "reason": "no placement and no anchor"})
    else:
        withheld.append({"component": "mana", "reason": "no benchmark mana figure and no audit"})

    # resilience ────────────────────────────────────────────────────────
    ans = None if fa.is_absent(bench) else bench.get("answer_cards")
    breadth = _axis(audit, "interaction-breadth")
    if fa.is_fig(ans) or breadth:
        place = placement(grp("benchmark", "answer_cards"), slug, True)[0] if fa.is_fig(ans) else None
        parts, inputs, cites = [], [], []
        if fa.is_fig(ans):
            parts.append(band_score(ans["value"], 15, 20))
            inputs.append(_input("answer_cards", ans))
            cites.append({"kind": "strategy", "ref": "strategy:deckbuilding.interaction-suite",
                          "quote": "interactive suite"})
        if breadth and breadth.get("measured", {}).get("value") is not None:
            classes = breadth["measured"]["value"]
            parts.append(100.0 * classes / 5.0)
            inputs.append({"figure": "audit.interaction-breadth", "value": classes, "ci95": None, "n": None,
                           "source": "deck-audit#interaction-breadth (classes answered of 5)",
                           "decklist_sha256": measured_sha(slug)})
        anchor = sum(parts) / len(parts) if parts else None
        sub, ci, how = _combine(place, anchor)
        if sub is not None:
            comps.append(_component(
                "resilience", sub=sub, sub_ci=ci, place=place, anchor=anchor,
                method=f"{how}: placement of answer_cards within bracket {key}; anchor = mean of "
                       f"band_score(answer_cards, 15, 20) and classes_answered/5·100 — CAPACITY, not performance",
                inputs=inputs, citations=cites))
        else:
            withheld.append({"component": "resilience", "reason": "no placement and no anchor"})
    else:
        withheld.append({"component": "resilience", "reason": "no benchmark answer figure and no audit"})

    # construction ──────────────────────────────────────────────────────
    scores, inputs, cites, dropped = [], [], [], []
    for name in CONSTRUCTION_AXES:
        a = _axis(audit, name)
        v = (a or {}).get("measured", {}).get("value")
        t = (a or {}).get("target") or {}
        if a is None or v is None or (t.get("low") is None and t.get("high") is None):
            dropped.append(name)
            continue
        scores.append(band_score(v, t.get("low"), t.get("high")))
        inputs.append({"figure": f"audit.{name}", "value": v, "ci95": None, "n": None,
                       "source": f"deck-audit#{name} (verdict {a.get('verdict')})",
                       "decklist_sha256": measured_sha(slug)})
        if t.get("source") and t.get("quote"):
            cites.append({"kind": "strategy", "ref": t["source"], "quote": t["quote"]})
    if scores:
        sub = sum(scores) / len(scores)
        comps.append(_component(
            "construction", sub=sub, sub_ci=(sub, sub), place=None, anchor=sub,
            method="anchor only: mean band_score over the audit axes with a cited band ("
                   + ", ".join(n for n in CONSTRUCTION_AXES if n not in dropped) + ")"
                   + (f"; dropped (no band or unreadable): {', '.join(dropped)}" if dropped else "")
                   + ". Counts, no interval; an authored target is not a sample, so no placement",
            inputs=inputs, citations=cites))
    else:
        withheld.append({"component": "construction", "reason": "no audit axis with a band could be read"})

    # table_record ──────────────────────────────────────────────────────
    recs = {m: _record(m) for m in members}
    mine = recs.get(slug) or _record(slug)
    if mine["games"] >= RATING_MIN_LOG_GAMES:
        wr = mine["wins"] / mine["games"]
        lo, hi = wilson_bounds(mine["wins"], mine["games"])
        edge = {m: ((x["wins"] / x["games"]) - x["par"]) if x["games"] >= RATING_MIN_LOG_GAMES else None
                for m, x in recs.items()}
        place = placement(edge, slug, True)[0]
        anchor = clamp(50.0 + 100.0 * (wr - mine["par"]))
        sub, ci, how = _combine(place, anchor, clamp(50.0 + 100.0 * (lo - mine["par"])),
                                clamp(50.0 + 100.0 * (hi - mine["par"])))
        comps.append(_component(
            "table_record", sub=sub, sub_ci=ci, place=place, anchor=anchor,
            method=f"{how}: win rate over par (par = mean 1/(opponents+1) = {mine['par']}) on every logged "
                   f"game, any version; Wilson 95% at n={mine['games']}",
            inputs=[{"figure": "log.win_rate", "value": round(wr, 4), "ci95": [round(lo, 4), round(hi, 4)],
                     "n": mine["games"], "source": "log.jsonl#result", "decklist_sha256": None}],
            citations=[{"kind": "log", "ref": "log.jsonl", "quote": f"{mine['wins']} wins in {mine['games']} games"}]))
    else:
        withheld.append({"component": "table_record",
                         "reason": f"{mine['games']} game(s) logged; {RATING_MIN_LOG_GAMES} needed"})

    # pilot_feel ────────────────────────────────────────────────────────
    sv = r["survey"]
    if not fa.is_absent(sv) and sv.get("games_surveyed", 0) >= RATING_MIN_SURVEY_GAMES:
        means = [sv[k]["mean"] for k in ("power", "resilience", "consistency") if sv.get(k)]
        if means:
            mean = sum(means) / len(means)
            anchor = clamp(100.0 * (mean - 1.0) / 4.0)
            feel_group = {}
            for m in members:
                s2 = rows[m]["survey"]
                ms = [s2[k]["mean"] for k in ("power", "resilience", "consistency")
                      if not fa.is_absent(s2) and s2.get(k)] if not fa.is_absent(s2) else []
                feel_group[m] = (sum(ms) / len(ms)) if ms and s2.get("games_surveyed", 0) >= RATING_MIN_SURVEY_GAMES else None
            place = placement(feel_group, slug, True)[0]
            sub, ci, how = _combine(place, anchor)
            comps.append(_component(
                "pilot_feel", sub=sub, sub_ci=ci, place=place, anchor=anchor,
                method=f"{how}: the pilot's own 1-5 means of power, resilience and consistency, "
                       f"(mean − 1)/4·100; AUTHORED, n={sv['games_surveyed']} surveyed games",
                inputs=[{"figure": f"survey.{k}", "value": sv[k]["mean"], "ci95": None, "n": sv[k]["n"],
                         "source": "log.jsonl#feel", "decklist_sha256": None}
                        for k in ("power", "resilience", "consistency") if sv.get(k)],
                citations=[{"kind": "log", "ref": "log.jsonl", "quote": "feel"}]))
        else:
            withheld.append({"component": "pilot_feel", "reason": "no power/resilience/consistency scores"})
    else:
        n = 0 if fa.is_absent(sv) else sv.get("games_surveyed", 0)
        withheld.append({"component": "pilot_feel",
                         "reason": f"{n} surveyed game(s); {RATING_MIN_SURVEY_GAMES} needed"})

    # forge ─────────────────────────────────────────────────────────────
    tb = r["table"]
    wrf = None if fa.is_absent(tb) else tb.get("win_rate")
    if fa.is_fig(wrf) and not wrf.get("stale"):
        anchor = clamp(50.0 + 100.0 * (wrf["value"] - FAIR_SHARE))
        lo, hi = (wrf.get("ci95") or [wrf["value"], wrf["value"]])
        sub, ci, how = _combine(None, anchor, clamp(50.0 + 100.0 * (lo - FAIR_SHARE)),
                                clamp(50.0 + 100.0 * (hi - FAIR_SHARE)))
        comps.append(_component(
            "forge", sub=sub, sub_ci=ci, place=None, anchor=anchor,
            method=f"{how}: Forge win rate over a fair share ({FAIR_SHARE}) at the standard table, on the "
                   f"CURRENT list; weight {WEIGHTS['forge']} by the pilot's rule (2026-10-18)",
            inputs=[_input("forge.win_rate", wrf)]))
    else:
        withheld.append({"component": "forge",
                         "reason": ("no Forge run on the current list" + (" (latest run describes an older list)"
                                                                           if fa.is_fig(wrf) else ""))})

    context = {"dti": (None if fa.is_absent(r["dti"]) else
                       {"score": r["dti"]["score"]["value"], "max_score": r["dti"]["max_score"],
                        "bracket": r["dti"]["bracket"], "as_of": r["dti"]["as_of"],
                        "stale_list": r["dti"]["stale_list"]}),
               "floor_vs_target": f"floor {r['bracket']['floor']}, target {r['bracket']['target']}"}
    return comps, withheld, context


# ── The document ─────────────────────────────────────────────────────────


def _finish(doc, judgment=None):
    """Weights, composite, interval, grade — from the components present."""
    comps = [c for c in doc["components"]]
    present = {c["name"]: c for c in comps}
    j = judgment
    names = list(present) + (["judgment"] if j else [])
    total = sum(WEIGHTS[n] for n in names)
    code_total = sum(WEIGHTS[n] for n in present) or 1.0
    for c in comps:
        c["weight_effective"] = round(WEIGHTS[c["name"]] / total, 4) if total else None
    code = sum(WEIGHTS[c["name"]] * c["sub_score"] for c in comps) / code_total if comps else None
    if j:
        j["weight"] = WEIGHTS["judgment"]
        j["weight_effective"] = round(WEIGHTS["judgment"] / total, 4)
    score = None
    lo = hi = 0.0
    if total and (comps or j):
        score = (sum(WEIGHTS[c["name"]] * c["sub_score"] for c in comps)
                 + (WEIGHTS["judgment"] * j["sub_score"] if j else 0.0)) / total
        lo = (sum(WEIGHTS[c["name"]] * c["sub_ci"][0] for c in comps)
              + (WEIGHTS["judgment"] * j["sub_score"] if j else 0.0)) / total
        hi = (sum(WEIGHTS[c["name"]] * c["sub_ci"][1] for c in comps)
              + (WEIGHTS["judgment"] * j["sub_score"] if j else 0.0)) / total
    doc["code_score"] = None if code is None else round(code, 1)
    doc["score"] = None if score is None else round(score, 1)
    doc["score_ci"] = None if score is None else [round(lo, 1), round(hi, 1)]
    doc["grade"] = None if score is None else grade_of(score)
    if score is not None:
        bounds = [f for _g, f in GRADES if lo < f <= hi]
        doc["resolution"] = (f"the interval spans {len(bounds)} grade boundar{'y' if len(bounds) == 1 else 'ies'}"
                             + (f" ({', '.join(str(b) for b in bounds)})" if bounds else "")
                             + "; measured components carry their inputs' intervals, the judgment none")
    else:
        doc["resolution"] = "no component present"
    doc["judgment"] = j
    return doc


def skeleton(slug, fleet=None, audit_cache=None):
    fleet = fleet or _fleet()
    key, basis, members = group_for(fleet, slug)
    r = _row(fleet, slug)
    comps, withheld, context = components(slug, fleet, audit_cache)
    withheld = withheld + [{"component": "judgment", "reason": "no deck-rater pass yet"}]
    doc = {
        "slug": slug, "tier": TIER, "label": RATING_LABEL.format(n=len(members)),
        "bracket": {"key": key, "source": basis, "floor": r["bracket"]["floor"],
                    "floor_name": r["bracket"]["floor_name"],
                    "within_target": (r["bracket"]["target"] is None or r["bracket"]["floor"] is None
                                      or r["bracket"]["floor"] <= r["bracket"]["target"])},
        "components": comps, "withheld": withheld,
        "limits": list(LIMITS_FIXED) + [f"rank over n={len(members)} decks; no significance claim"],
        "context": context,
        "decklist_sha256": measured_sha(slug),
        "fleet_sha": hashlib.sha256(config.FLEET_ANALYTICS_PATH.read_bytes()).hexdigest()[:12],
        "model_version": spec_sha(), "spec_version": RATING_SPEC_VERSION,
        "open_questions": [], "iterations": 0, "verifier": None,
    }
    _finish(doc, None)
    doc["rank"] = None     # filled by `ranked`, which needs the group's scores
    return doc


def carry_forward(doc, previous):
    """Keep an existing judgment / verifier / open questions when they were
    written against this list and this spec; otherwise drop them into
    `withheld[]` with the reason."""
    if not previous:
        return doc
    j = previous.get("judgment")
    against = (j or {}).get("against") or {}
    if j and against.get("decklist_sha256") == doc["decklist_sha256"] and against.get("spec_sha") == doc["model_version"]:
        doc["withheld"] = [w for w in doc["withheld"] if w["component"] != "judgment"]
        doc["open_questions"] = previous.get("open_questions") or []
        doc["iterations"] = previous.get("iterations") or 0
        doc["verifier"] = previous.get("verifier")
        _finish(doc, dict(j))
    elif j:
        why = ("judgment written against " + str(against.get("decklist_sha256", ""))[:12]
               + (" (another spec)" if against.get("spec_sha") != doc["model_version"] else ""))
        doc["withheld"] = [w for w in doc["withheld"] if w["component"] != "judgment"] \
            + [{"component": "judgment", "reason": why}]
    return doc


def member_score(slug, fleet, audit_cache):
    """A group member's final score: its tracked rating when it describes the
    current list, else its skeleton's code score. `(score, basis)`."""
    tracked = load_json(deck_dir(slug) / RATING_FILE)
    if tracked and tracked.get("decklist_sha256") == measured_sha(slug) and tracked.get("score") is not None:
        return tracked["score"], "rating.json"
    sk = skeleton(slug, fleet, audit_cache)
    return sk["code_score"], "skeleton code_score"


def ranked(doc, fleet=None, audit_cache=None):
    """Fill `rank` from the group's final scores."""
    fleet = fleet or _fleet()
    audit_cache = audit_cache if audit_cache is not None else {}
    _key, _basis, members = group_for(fleet, doc["slug"])
    scores, basis = {}, {}
    for m in members:
        if m == doc["slug"]:
            scores[m] = doc["score"]
            basis[m] = "this document"
        else:
            scores[m], basis[m] = member_score(m, fleet, audit_cache)
    ordered = sorted((m for m in members if scores[m] is not None), key=lambda m: (-scores[m], m))
    pos = ordered.index(doc["slug"]) + 1 if doc["slug"] in ordered else None
    ties = None
    if pos is not None and doc.get("score_ci"):
        lo, hi = doc["score_ci"]
        ties = any(m != doc["slug"] and scores[m] is not None and lo <= scores[m] <= hi for m in ordered)
    doc["rank"] = {"position": pos, "of": len(ordered), "group": ordered, "ties_within_ci": ties,
                   "basis": {m: basis[m] for m in ordered}, "scores": {m: scores[m] for m in ordered}}
    return doc


def build(slug, fleet=None, audit_cache=None, previous=None):
    fleet = fleet or _fleet()
    audit_cache = audit_cache if audit_cache is not None else {}
    doc = skeleton(slug, fleet, audit_cache)
    doc = carry_forward(doc, previous if previous is not None else load_json(deck_dir(slug) / RATING_FILE))
    return ranked(doc, fleet, audit_cache)


def dumps(doc):
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def write(slug, doc):
    p = deck_dir(slug) / RATING_FILE
    p.write_text(dumps(doc), encoding="utf-8")
    return p


def leaderboard(fleet=None):
    fleet = fleet or _fleet()
    audit_cache = {}
    out = {}
    for g, spec in fleet["groups"].items():
        rows = []
        for m in spec["members"]:
            doc = build(m, fleet, audit_cache)
            rows.append(doc)
        rows.sort(key=lambda d: (-(d["score"] or 0), d["slug"]))
        out[g] = rows
    return out


def calibrate_rating(fleet=None):
    """Spearman of the score against outcomes, with calibrate's refusal."""
    fleet = fleet or _fleet()
    board = leaderboard(fleet)
    rows = []
    for g, docs in board.items():
        for d in docs:
            r = _row(fleet, d["slug"])
            rec = _record(d["slug"])
            sv = r["survey"]
            rows.append({"slug": d["slug"], "bracket": g, "score": d["score"], "code_score": d["code_score"],
                         "pilot_win_rate": (rec["wins"] / rec["games"]) if rec["games"] >= calibrate.MIN_GAMES else None,
                         "pilot_games": rec["games"],
                         "survey_power": (sv.get("power") or {}).get("mean") if not fa.is_absent(sv) else None,
                         "dti": None if fa.is_absent(r["dti"]) else r["dti"]["score"]["value"]})
    cells = {}
    for name, key in (("score_vs_pilot_win_rate", "pilot_win_rate"), ("score_vs_survey_power", "survey_power"),
                      ("score_vs_dti", "dti")):
        pairs = [(x["score"], x[key]) for x in rows if x["score"] is not None and x[key] is not None]
        cells[name] = fa._spearman_or_refusal(pairs, name, "more rated decks carrying the outcome")
    return {"rows": rows, "cells": cells, "min_decks": calibrate.MIN_DECKS, "min_games": calibrate.MIN_GAMES}


def _print_doc(doc):
    b = doc["bracket"]
    rk = doc.get("rank") or {}
    print(f"RATING — {doc['slug']}   {doc['tier']} coaching")
    print(f"  {doc['grade'] or '—'} in Bracket {b['key']} ({rk.get('position')} of {rk.get('of')})   "
          f"score {doc['score']}  [{doc['score_ci'][0] if doc['score_ci'] else '—'}, "
          f"{doc['score_ci'][1] if doc['score_ci'] else '—'}]   code {doc['code_score']}")
    print(f"  {doc['label']}")
    for c in doc["components"]:
        print(f"    {c['name']:14} {c['sub_score']:6.1f}  [{c['sub_ci'][0]:5.1f}, {c['sub_ci'][1]:5.1f}]  "
              f"w {c['weight']:.2f} → {c['weight_effective']:.3f}   {c['method'][:70]}")
    if doc.get("judgment"):
        j = doc["judgment"]
        print(f"    {'judgment':14} {j['sub_score']:6.1f}  w {j['weight']:.2f} → {j['weight_effective']:.3f}   "
              f"{len(j.get('citations') or [])} citation(s)")
    for w in doc["withheld"]:
        print(f"    withheld {w['component']:12} — {w['reason']}")
    print(f"  {doc['resolution']}")
    if rk.get("group"):
        print("  group: " + ", ".join(f"{m} {rk['scores'][m]}" for m in rk["group"]))


def main(args):
    slug = getattr(args, "slug", None)
    if getattr(args, "calibrate", False):
        cal = calibrate_rating()
        if getattr(args, "as_json", False):
            print(json.dumps(cal, indent=1)); return
        print(f"RATING CALIBRATION — {len(cal['rows'])} rated decks "
              f"(refusal below {cal['min_decks']} decks with {cal['min_games']} games)")
        for name, c in cal["cells"].items():
            print(f"  {name:26} {c['verdict']}"
                  + (f" (n={c['n']}) — {c['why']}" if c["verdict"] != "measured"
                     else f" rho={c['rho']} n={c['n']} critical={c['critical_rho']}"))
        return
    if getattr(args, "all_decks", False) or not slug:
        board = leaderboard()
        if getattr(args, "as_json", False):
            print(json.dumps(board, indent=1)); return
        print(f"RATINGS — ★ coaching over ◆ inputs; within bracket; no significance claim\n")
        for g, docs in board.items():
            print(f"  {g}  (n={len(docs)})")
            for d in docs:
                j = "judged" if d.get("judgment") else "skeleton"
                print(f"    {d['grade'] or '—'}  {d['slug']:18} {str(d['score']):>6}  "
                      f"[{d['score_ci'][0] if d['score_ci'] else '—'}, {d['score_ci'][1] if d['score_ci'] else '—'}]  "
                      f"{j}; withheld: {', '.join(w['component'] for w in d['withheld']) or '—'}")
        return
    doc = build(slug)
    if getattr(args, "write", False):
        p = write(slug, doc)
        print(f"wrote {p}")
        _print_doc(doc)
        return
    if getattr(args, "as_json", False):
        print(dumps(doc), end=""); return
    _print_doc(doc)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot rating <slug> [--write] | --all | --calibrate`.")
