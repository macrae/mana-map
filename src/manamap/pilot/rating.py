"""Pilot: the deck rating — ★ coaching over ◆ inputs, within a bracket, 0–100.

WHAT THIS IS. A scorecard: every component is a measured figure the bench
already publishes (the uniform-harness benchmark, the cited audit axes,
DeckCheck's dated reading, the pilot's own record and survey) mapped to 0–100
by its placement within the deck's bracket group and by an absolute anchor
quoted from the strategy companion, weighted by a DECLARED PRIOR, plus ONE
judgment component an agent fills (`deck-rater`) and an adversarial verifier
attacks (`rating-skeptic`). The result is a score with a 90% interval, a grade
with its probabilities, and a rank within the bracket with the probability of
each position — printed with its group and its n, and claiming no significance.

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

VERSION 2 (2026-10-19): THE WEIGHTS ARE A PRIOR, THE INTERVALS ARE REAL, AND
DISAGREEMENT IS A FIGURE. The first fleet pass put heliod at D 49.9 [49.9,
49.9] while DeckCheck read it the fleet's highest, and the pilot called that an
anti-signal the rating should measure. Now `WEIGHTS` is the mean of a
Dirichlet (`WEIGHT_CONCENTRATION`), the inputs are drawn from their posteriors,
a placement is shrunk toward its anchor by the group's size (`PLACEMENT_SHRINK_K`
— at n=3 a percentile is a three-point scale), the DTI is a component at its
prior weight, a win the goldfish cannot see halves speed's weight by a
mechanical rule (`alt_win_cards`), and `concordance` ranks the deck by every
instrument and says how far they disagree — a divergent deck prints its grade
as a range. `rating_mc.py` holds the machinery; every prior is named there and
in `uncertainty`.

ABSENT MEANS ABSENT. A component with no usable input is WITHHELD — named in
`withheld[]` with its reason — and the weights are renormalised over what is
present. Never zero. `table_record`, `pilot_feel` and `forge` are withheld on
most decks today, and the skeleton says so; the value grows with logged games
and surveys.

THE JUDGMENT IS A WEIGHTED COMPONENT, NOT A ±ADDER: a 0–100 sub-score at a
fixed 0.20 weight is bounded by construction, visible beside the measured
components, and renormalises with them. ITS NEUTRAL POINT IS `code_score`,
NOT 50: a judgment equal to the measured score leaves the composite where the
components put it; 50 would drag every deck above 50 down and every deck below
it up, which the first fleet pass showed. The skeptic attacks the number
against the evidence; the code bounds its reach.

THE SKELETON IS RECOMPUTED, THE JUDGMENT IS CARRIED. `--write` rebuilds every
code component and keeps an existing judgment only when it was written
against this same list and this same spec (`judgment.against`); otherwise the
judgment is dropped into `withheld[]` with the sha reason. The captain's-log
rule: the agent writes prose, never re-derives a figure.
"""

import hashlib
import json
import random
import re
import statistics

from manamap import config
from manamap.pilot import calibrate, deck_audit, dti as dti_mod, rating_mc as mc
from manamap.pilot import fleet_analytics as fa
from manamap.pilot.common import deck_dir, load_json, measured_sha
from manamap.pilot.deck_notes import read_log
from manamap.sim.stats import wilson_bounds

RATING_FILE = "rating.json"
RATING_SPEC_VERSION = 2
RATING_MIN_GROUP = 3           # below this, within-bracket placement is withheld
RATING_MIN_LOG_GAMES = 5       # table_record needs this many logged games
RATING_MIN_SURVEY_GAMES = 3    # pilot_feel needs this many surveyed games
RATING_ARGUMENT_MAX_WORDS = 250
RATING_LABEL = ("This rating is ★ coaching: a labelled judgment over ◆ measured inputs. "
                "It ranks within a bracket of {n} decks and claims no significance.")
TIER = "★"

#: THE PRIOR MEAN of the weights, summing to 1.0 over the full set and
#: renormalised over what is present. `forge` is 0.00 by the pilot's rule
#: (2026-10-18): not a power read until the harness work is done, and a rule
#: rather than a prior — it is excluded from the Dirichlet draw. `dti` entered
#: at 0.15 on 2026-10-19 (the pilot's call); the goldfish's share (speed + mana)
#: fell from 0.25 to 0.20. `judgment` is the agent's, fixed at 0.20.
WEIGHTS = {
    "speed": 0.12, "mana": 0.08, "resilience": 0.12, "construction": 0.13, "dti": 0.15,
    "table_record": 0.12, "pilot_feel": 0.08, "forge": 0.00, "judgment": 0.20,
}
GRADES = (("A", 80), ("B", 65), ("C", 50), ("D", 35), ("F", 0))
GRADE_LETTERS = tuple(g for g, _f in GRADES)
FAIR_SHARE = 0.25              # a four-seat table's null, `sim/pods.calibration`

#: The uncertainty machinery's constants — every one a DECLARED PRIOR, not an
#: estimate (known-issues §27). All are hashed into `spec_sha`.
WEIGHT_CONCENTRATION = 20.0    # κ: Dirichlet(κ·WEIGHTS); 20 is "fairly sure, not certain"
RATING_MC_K = 2000             # draws per group
RATING_MC_SEED = 20261019
RATING_MC_LEVEL = 0.90         # the printed interval: [q05, q95]
PLACEMENT_SHRINK_K = 4         # pseudo-comparisons shrinking a placement toward its anchor
JUDGMENT_SD = 5.0              # the judgment's draw sd, on the 0–100 scale
DTI_PRIOR_READ_SD = 4.0        # one DeckCheck reading's prior sd on the 0–100 anchor (two rubric steps), until test-retest
SPEED_VALIDITY_ALT_WIN = 0.5   # speed's weight multiplier when the engine's win is not damage
ALT_WIN_RE = re.compile(r"you win the game|each opponent loses the game|target opponent loses the game",
                        re.IGNORECASE)
CONCORDANCE_MIXED_AT = 0.34    # spread of instrument ranks /(n−1) at which a deck reads mixed
CONCORDANCE_DIVERGENT_AT = 0.67
KENDALL_EXACT_MAX = 50_000
KENDALL_PERM_ITERATIONS = 5000

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
    "with no blockers and no removal, and a combo or control deck reads slow for not attacking; "
    f"a win the goldfish cannot see (`alt_win_cards`) multiplies speed's weight by {SPEED_VALIDITY_ALT_WIN}.",
    "Forge is not a power read until the harness work is done (the pilot's rule, "
    "2026-10-18): its component ships at weight 0.00 and is shown only for a run on the "
    "current list.",
    f"DeckCheck's Threat Index is a component at prior weight {WEIGHTS['dti']} (2026-10-19): one "
    f"LLM's dated read of card quality, withheld when it describes another list; its interval is a "
    f"declared prior sd of {DTI_PRIOR_READ_SD} on the anchor until three decks carry two readings on one list; "
    f"its score places within the bracket, its twelve rubric grades anchor.",
    "Diagnostic engine figures are read against each deck's own declaration and are never "
    "ranked across decks; they are evidence for the argument, not a component.",
    f"The weights are a Dirichlet prior (κ = {WEIGHT_CONCENTRATION}) and the inputs are drawn from "
    f"their posteriors: `score` is the median of {RATING_MC_K} seeded draws, `score_ci` its "
    f"{int(RATING_MC_LEVEL * 100)}% interval, and every sd in `uncertainty` is declared, not estimated.",
    f"A placement within a group of n is a (n−1)-point scale and earns the share λ = (n−1)/(n−1+"
    f"{PLACEMENT_SHRINK_K}) of its component, the rest going to the cited anchor; at n=3 that is one third.",
    "A divergent `concordance` — the deck's instruments rank it far apart — prints the grade as a "
    "range, never a letter, and the judgment must address every outlier instrument.",
]


# ── Primitives ───────────────────────────────────────────────────────────


def spec_sha():
    """The scorecard's version: every constant that moves a number, hashed."""
    spec = {"version": RATING_SPEC_VERSION, "weights": WEIGHTS, "grades": GRADES,
            "min_group": RATING_MIN_GROUP, "min_log_games": RATING_MIN_LOG_GAMES,
            "min_survey_games": RATING_MIN_SURVEY_GAMES, "fair_share": FAIR_SHARE,
            "construction_axes": CONSTRUCTION_AXES, "expected_kill_turn": EXPECTED_KILL_TURN,
            "speed_anchor_rate": SPEED_ANCHOR_RATE, "label": RATING_LABEL,
            "kappa": WEIGHT_CONCENTRATION, "mc_k": RATING_MC_K, "mc_seed": RATING_MC_SEED,
            "mc_level": RATING_MC_LEVEL, "placement_shrink_k": PLACEMENT_SHRINK_K,
            "judgment_sd": JUDGMENT_SD, "dti_prior_read_sd": DTI_PRIOR_READ_SD,
            "speed_validity_alt_win": SPEED_VALIDITY_ALT_WIN, "alt_win_re": ALT_WIN_RE.pattern,
            "concordance": [CONCORDANCE_MIXED_AT, CONCORDANCE_DIVERGENT_AT],
            "kendall": [KENDALL_EXACT_MAX, KENDALL_PERM_ITERATIONS]}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()[:12]


clamp = mc.clamp


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
    """`(percentile 0-100, n)` within the group, ties averaged; `(None, None)`
    below `RATING_MIN_GROUP` or when the deck has no value."""
    vals = {s: v for s, v in values.items() if v is not None}
    if slug not in vals or len(vals) < RATING_MIN_GROUP:
        return None, None
    mine = vals[slug]
    others = [v for s, v in vals.items() if s != slug]
    below = sum(1 for v in others if ((v < mine) if higher_is_better else (v > mine)))
    equal = sum(1 for v in others if v == mine)
    rank_from_bottom = below + equal / 2.0
    return round(100.0 * rank_from_bottom / (len(vals) - 1), 2), len(vals)


def placement_weight(n):
    return mc.placement_weight(n, PLACEMENT_SHRINK_K)


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


def alt_win_cards(slug):
    """The cards in the engine's `wincon` stage whose oracle text wins the game
    without damage — the goldfish measures combat damage and cannot see them.
    `(names, why)`: `why` says "no engine.json" when the rule could not run."""
    eng = load_json(deck_dir(slug) / "engine.json")
    if not eng:
        return [], "no engine.json"
    declared = set()
    for s in eng.get("stages") or []:
        if s.get("stage") == "wincon":
            declared |= set(s.get("cards") or [])
    if not declared:
        return [], "engine.json declares no wincon stage"
    cards = (load_json(deck_dir(slug) / "cards.json") or {}).get("cards") or []
    hits = []
    for c in cards:
        if c.get("name") not in declared:
            continue
        text = c.get("oracle_text") or " ".join((f.get("oracle_text") or "") for f in c.get("card_faces") or [])
        if ALT_WIN_RE.search(text or ""):
            hits.append(c["name"])
    return sorted(hits), None


def _component(name, *, sub, place, anchor, method, inputs, citations=(), validity=1.0):
    return {"name": name, "weight": WEIGHTS[name], "weight_validity": validity, "weight_effective": None,
            "sub_score": round(sub, 2), "sub_ci": [round(sub, 2), round(sub, 2)],
            "placement": None if place is None else round(place, 2),
            "anchor": None if anchor is None else round(anchor, 2),
            "method": method, "inputs": inputs, "citations": list(citations)}


def _combine(place, n, anchor):
    """`(sub, how, lam)`: λ(n)·placement + (1−λ)·anchor when both exist; the one present otherwise."""
    if place is not None and anchor is not None:
        lam = placement_weight(n)
        return lam * place + (1.0 - lam) * anchor, f"{lam:.3f}·placement + {1 - lam:.3f}·anchor (λ at n={n})", lam
    if place is not None:
        return place, "placement only (no anchor)", 1.0
    if anchor is not None:
        return anchor, "anchor only (placement withheld)", 0.0
    return None, None, None


def _recipe(place, n, lam, anchor):
    """The draw recipe `rating_mc.draw_component` reads; `anchor` is a kind dict or None."""
    return {"place": place, "bin": (50.0 / (n - 1)) if (place is not None and n and n > 1) else 0.0,
            "lam": lam, "anchor": anchor}


def _rate_anchor(cell, a, b):
    """A rate anchor's recipe: p ~ Beta(k+1, n−k+1) with k = value·n, mapped a + b·p."""
    n = cell.get("n")
    if not n:
        return {"kind": "fixed", "value": clamp(a + b * cell["value"])}
    return {"kind": "rate", "k": int(round(cell["value"] * n)), "n": int(n), "a": a, "b": b}


#: DeckCheck's rubric grades on the 0–100 anchor scale. Their headline score
#: tops out at 96 for a cEDH list and reads 28–46 across this fleet, so
#: 100·score/96 would drag every bracket-3 and -4 deck toward F; their twelve
#: per-axis grades are the read of card quality the anchor wants, and the
#: score keeps its job as the PLACEMENT within the bracket.
DTI_GRADE_POINTS = {"S": 100.0, "A": 75.0, "B": 50.0, "C": 25.0, "F": 0.0}


def dti_anchor(reading):
    """The mean of a reading's twelve rubric grades on 0–100, or None."""
    grades = [DTI_GRADE_POINTS.get(v) for v in (reading.get("benchmarks") or {}).values()]
    grades = [g for g in grades if g is not None]
    return (sum(grades) / len(grades)) if grades else None


def _dti_readings(slug, sha):
    """The anchors of every reading on this list, latest last."""
    doc = dti_mod.load(slug)
    if not doc:
        return []
    out = []
    for r in doc.get("readings") or []:
        if r.get("decklist_sha256") == sha:
            a = dti_anchor(r)
            if a is not None:
                out.append(a)
    return out


def components(slug, fleet, audit_cache=None):
    """Every code component for `slug`: `(components[], withheld[], context, recipes)`.
    `recipes` is `{name: recipe}` for the simulation; it is recomputed, never stored."""
    audit_cache = audit_cache if audit_cache is not None else {}
    r = _row(fleet, slug)
    key, _basis, members = group_for(fleet, slug)
    rows = {m: _row(fleet, m) for m in members}
    comps, withheld, recipes, limits = [], [], {}, []
    bench = r["benchmark"]
    audit = _audit(slug, audit_cache)

    def grp(block, k):
        return {m: fa._measure_value(rows[m], (block, k)) for m in members}

    # speed ─────────────────────────────────────────────────────────────
    k8 = None if fa.is_absent(bench) else bench.get("kill_by_turn_8")
    if fa.is_fig(k8):
        place, n = placement(grp("benchmark", "kill_by_turn_8"), slug, True)
        t = EXPECTED_KILL_TURN.get(key, 8)
        kt = bench.get(f"kill_by_turn_{t}")
        anchor = anchor_recipe = None
        if fa.is_fig(kt):
            anchor = clamp(100.0 * kt["value"] / SPEED_ANCHOR_RATE)
            anchor_recipe = _rate_anchor(kt, 0.0, 100.0 / SPEED_ANCHOR_RATE)
        sub, how, lam = _combine(place, n, anchor)
        if sub is not None:
            alt, why = alt_win_cards(slug)
            validity = SPEED_VALIDITY_ALT_WIN if alt else 1.0
            if alt:
                limits.append(f"speed: the goldfish measures combat damage and cannot see {', '.join(alt)} "
                              f"(engine.json wincon; oracle text wins the game); its prior weight is multiplied "
                              f"by {SPEED_VALIDITY_ALT_WIN} (SPEED_VALIDITY_ALT_WIN)")
            elif why:
                limits.append(f"speed validity unchecked: {why}")
            comps.append(_component(
                "speed", sub=sub, place=place, anchor=anchor, validity=validity,
                method=(f"{how}: placement of kill_by_turn_8 within bracket {key} (n={len(members)}); "
                        f"anchor kill_by_turn_{t} / {SPEED_ANCHOR_RATE} for a bracket-{key} clock "
                        f"(config.BRACKETS expected turns); Beta posterior from the benchmark's n"
                        + (f"; weight × {validity} — a win the goldfish cannot see ({', '.join(alt)})" if alt else "")),
                inputs=[_input("kill_by_turn_8", k8)] + ([_input(f"kill_by_turn_{t}", kt)] if fa.is_fig(kt) and t != 8 else []),
                citations=[{"kind": "strategy", "ref": "strategy:deckbuilding.power-level",
                            "quote": "the sharpest line between them is expected game length"}]))
            recipes["speed"] = _recipe(place, n, lam, anchor_recipe)
        else:
            withheld.append({"component": "speed", "reason": "no placement and no anchor"})
    else:
        withheld.append({"component": "speed", "reason": "no benchmark kill_by_turn_8 figure"})

    # mana ──────────────────────────────────────────────────────────────
    mld = None if fa.is_absent(bench) else bench.get("missed_land_drop_rate")
    cons = _axis(audit, "consistency")
    if fa.is_fig(mld) or cons:
        place, n = placement(grp("benchmark", "missed_land_drop_rate"), slug, False) if fa.is_fig(mld) else (None, None)
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
        sub, how, lam = _combine(place, n, anchor)
        if sub is not None:
            comps.append(_component(
                "mana", sub=sub, place=place, anchor=anchor,
                method=f"{how}: placement of missed_land_drop_rate (lower is better) within bracket {key}; "
                       f"anchor band_score(turn-3 land-drop rate, low 0.74) from the audit's consistency axis",
                inputs=inputs, citations=cites))
            recipes["mana"] = _recipe(place, n, lam, None if anchor is None else {"kind": "fixed", "value": anchor})
        else:
            withheld.append({"component": "mana", "reason": "no placement and no anchor"})
    else:
        withheld.append({"component": "mana", "reason": "no benchmark mana figure and no audit"})

    # resilience ────────────────────────────────────────────────────────
    ans = None if fa.is_absent(bench) else bench.get("answer_cards")
    breadth = _axis(audit, "interaction-breadth")
    if fa.is_fig(ans) or breadth:
        place, n = placement(grp("benchmark", "answer_cards"), slug, True) if fa.is_fig(ans) else (None, None)
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
        sub, how, lam = _combine(place, n, anchor)
        if sub is not None:
            comps.append(_component(
                "resilience", sub=sub, place=place, anchor=anchor,
                method=f"{how}: placement of answer_cards within bracket {key}; anchor = mean of "
                       f"band_score(answer_cards, 15, 20) and classes_answered/5·100 — CAPACITY, not performance",
                inputs=inputs, citations=cites))
            recipes["resilience"] = _recipe(place, n, lam, None if anchor is None else {"kind": "fixed", "value": anchor})
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
            "construction", sub=sub, place=None, anchor=sub,
            method="anchor only: mean band_score over the audit axes with a cited band ("
                   + ", ".join(n for n in CONSTRUCTION_AXES if n not in dropped) + ")"
                   + (f"; dropped (no band or unreadable): {', '.join(dropped)}" if dropped else "")
                   + ". Counts, no interval; an authored target is not a sample, so no placement",
            inputs=inputs, citations=cites))
        recipes["construction"] = _recipe(None, None, 0.0, {"kind": "fixed", "value": sub})
    else:
        withheld.append({"component": "construction", "reason": "no audit axis with a band could be read"})

    # dti ───────────────────────────────────────────────────────────────
    dti = r["dti"]
    reads = [] if fa.is_absent(dti) else _dti_readings(slug, measured_sha(slug))
    if fa.is_absent(dti):
        withheld.append({"component": "dti", "reason": "no DeckCheck reading (deckcheck.json)"})
    elif dti.get("stale_list") or not reads:
        withheld.append({"component": "dti", "reason": "the DeckCheck reading describes another list — "
                                                       f"`deckcheck {slug} --write` after a rescan"})
    else:
        sc, mx = dti["score"], dti.get("max_score") or dti_mod.MAX_SCORE
        place, n = placement(grp("dti", "score"), slug, True)
        anchor = clamp(reads[-1])
        if len(reads) >= 2:
            sd = statistics.stdev(reads) / (len(reads) ** 0.5)
            how_sd = f"sd of the mean over {len(reads)} readings on this list"
        else:
            sd = DTI_PRIOR_READ_SD
            how_sd = f"DTI_PRIOR_READ_SD = {DTI_PRIOR_READ_SD}, a declared prior (one reading on this list)"
        sub, how, lam = _combine(place, n, anchor)
        comps.append(_component(
            "dti", sub=sub, place=place, anchor=anchor,
            method=f"{how}: placement of DeckCheck's score (of {mx}) within bracket {key}; anchor = the mean of "
                   f"its twelve rubric grades (S 100, A 75, B 50, C 25, F 0); draw Normal(anchor, {how_sd}); "
                   f"an LLM's dated read of card quality, never ours",
            inputs=[_input("dti.score", sc),
                    {"figure": "dti.rubric_mean", "value": round(anchor, 2), "ci95": None, "n": len(reads),
                     "source": "deckcheck.json#readings[-1].benchmarks", "decklist_sha256": sc.get("decklist_sha256")}],
            citations=[{"kind": "skeleton", "ref": "skeleton:dti", "quote": "DeckCheck Threat Index"}]))
        recipes["dti"] = _recipe(place, n, lam, {"kind": "normal", "mean": anchor, "sd": sd})

    # table_record ──────────────────────────────────────────────────────
    recs = {m: _record(m) for m in members}
    mine = recs.get(slug) or _record(slug)
    if mine["games"] >= RATING_MIN_LOG_GAMES:
        wr = mine["wins"] / mine["games"]
        lo, hi = wilson_bounds(mine["wins"], mine["games"])
        edge = {m: ((x["wins"] / x["games"]) - x["par"]) if x["games"] >= RATING_MIN_LOG_GAMES else None
                for m, x in recs.items()}
        place, n = placement(edge, slug, True)
        anchor = clamp(50.0 + 100.0 * (wr - mine["par"]))
        sub, how, lam = _combine(place, n, anchor)
        comps.append(_component(
            "table_record", sub=sub, place=place, anchor=anchor,
            method=f"{how}: win rate over par (par = mean 1/(opponents+1) = {mine['par']}) on every logged "
                   f"game, any version; Beta posterior at n={mine['games']}",
            inputs=[{"figure": "log.win_rate", "value": round(wr, 4), "ci95": [round(lo, 4), round(hi, 4)],
                     "n": mine["games"], "source": "log.jsonl#result", "decklist_sha256": None}],
            citations=[{"kind": "log", "ref": "log.jsonl", "quote": f"{mine['wins']} wins in {mine['games']} games"}]))
        recipes["table_record"] = _recipe(place, n, lam, {"kind": "rate", "k": mine["wins"], "n": mine["games"],
                                                           "a": 50.0 - 100.0 * mine["par"], "b": 100.0})
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
            place, n = placement(feel_group, slug, True)
            sub, how, lam = _combine(place, n, anchor)
            comps.append(_component(
                "pilot_feel", sub=sub, place=place, anchor=anchor,
                method=f"{how}: the pilot's own 1-5 means of power, resilience and consistency, "
                       f"(mean − 1)/4·100; AUTHORED, n={sv['games_surveyed']} surveyed games",
                inputs=[{"figure": f"survey.{k}", "value": sv[k]["mean"], "ci95": None, "n": sv[k]["n"],
                         "source": "log.jsonl#feel", "decklist_sha256": None}
                        for k in ("power", "resilience", "consistency") if sv.get(k)],
                citations=[{"kind": "log", "ref": "log.jsonl", "quote": "feel"}]))
            recipes["pilot_feel"] = _recipe(place, n, lam, {"kind": "fixed", "value": anchor})
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
        sub, how, lam = _combine(None, None, anchor)
        comps.append(_component(
            "forge", sub=sub, place=None, anchor=anchor,
            method=f"{how}: Forge win rate over a fair share ({FAIR_SHARE}) at the standard table, on the "
                   f"CURRENT list; weight {WEIGHTS['forge']} by the pilot's rule (2026-10-18)",
            inputs=[_input("forge.win_rate", wrf)]))
        recipes["forge"] = _recipe(None, None, 0.0, {"kind": "fixed", "value": anchor})
    else:
        withheld.append({"component": "forge",
                         "reason": ("no Forge run on the current list" + (" (latest run describes an older list)"
                                                                           if fa.is_fig(wrf) else ""))})

    context = {"dti": (None if fa.is_absent(r["dti"]) else
                       {"score": r["dti"]["score"]["value"], "max_score": r["dti"]["max_score"],
                        "bracket": r["dti"]["bracket"], "as_of": r["dti"]["as_of"],
                        "stale_list": r["dti"]["stale_list"]}),
               "floor_vs_target": f"floor {r['bracket']['floor']}, target {r['bracket']['target']}",
               "limits": limits}
    return comps, withheld, context, recipes


# ── The document ─────────────────────────────────────────────────────────


def _entries(doc, judgment):
    """`[(name, weight × validity, sub)]` — the deterministic point score's parts."""
    out = [(c["name"], WEIGHTS[c["name"]] * c.get("weight_validity", 1.0), c["sub_score"]) for c in doc["components"]]
    if judgment:
        out.append(("judgment", WEIGHTS["judgment"], float(judgment["sub_score"])))
    return out


def _finish(doc, judgment=None):
    """Weights, the deterministic point score and code score, the provisional
    grade, the closed-form sensitivity. `ranked` then overlays the simulation."""
    comps = doc["components"]
    j = judgment
    entries = _entries(doc, j)
    total = sum(w for _n, w, _s in entries)
    code_entries = [(n, w, s) for n, w, s in entries if n != "judgment"]
    code_total = sum(w for _n, w, _s in code_entries)
    for c in comps:
        c["weight_effective"] = round(WEIGHTS[c["name"]] * c.get("weight_validity", 1.0) / total, 4) if total else None
    code = (sum(w * s for _n, w, s in code_entries) / code_total) if code_total else None
    if j:
        j["weight"] = WEIGHTS["judgment"]
        j["weight_effective"] = round(WEIGHTS["judgment"] / total, 4) if total else None
    point = (sum(w * s for _n, w, s in entries) / total) if total else None
    doc["code_score"] = None if code is None else round(code, 1)
    doc["score_point"] = None if point is None else round(point, 1)
    doc["score"] = doc["score_point"]
    doc["score_ci"] = None if point is None else [doc["score_point"], doc["score_point"]]
    doc["grade"] = None if point is None else grade_of(point)
    doc["grade_range"] = None if point is None else [doc["grade"], doc["grade"]]
    doc["grade_display"] = doc["grade"]
    doc["grade_probabilities"] = None
    doc["uncertainty"] = None
    doc["sensitivity"] = mc.sensitivity(entries, GRADES, grade_of) if entries else {"leave_one_out": {}, "flips": []}
    doc["concordance"] = None
    doc["resolution"] = "point score only; `ranked` draws the interval" if point is not None else "no component present"
    doc["judgment"] = j
    return doc


def skeleton(slug, fleet=None, audit_cache=None):
    fleet = fleet or _fleet()
    key, basis, members = group_for(fleet, slug)
    r = _row(fleet, slug)
    comps, withheld, context, _recipes = components(slug, fleet, audit_cache)
    withheld = withheld + [{"component": "judgment", "reason": "no deck-rater pass yet"}]
    per_deck = context.pop("limits")
    doc = {
        "slug": slug, "tier": TIER, "label": RATING_LABEL.format(n=len(members)),
        "bracket": {"key": key, "source": basis, "floor": r["bracket"]["floor"],
                    "floor_name": r["bracket"]["floor_name"],
                    "within_target": (r["bracket"]["target"] is None or r["bracket"]["floor"] is None
                                      or r["bracket"]["floor"] <= r["bracket"]["target"])},
        "components": comps, "withheld": withheld,
        "limits": list(LIMITS_FIXED) + per_deck + [f"rank over n={len(members)} decks; no significance claim"],
        "context": context,
        "decklist_sha256": measured_sha(slug),
        "fleet_sha": hashlib.sha256(config.FLEET_ANALYTICS_PATH.read_bytes()).hexdigest()[:12],
        "model_version": spec_sha(), "spec_version": RATING_SPEC_VERSION,
        "open_questions": [], "iterations": 0, "verifier": None,
    }
    _finish(doc, None)
    doc["rank"] = None     # filled by `ranked`, which needs the group's scores
    return doc


def judgment_current(j, slug):
    against = (j or {}).get("against") or {}
    return bool(j) and against.get("decklist_sha256") == measured_sha(slug) and against.get("spec_sha") == spec_sha()


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


def _member(slug, fleet, audit_cache, own=None):
    """A group member as the simulation sees it: its components' recipes,
    validity multipliers, sub-scores, and its judgment — this document's own
    when `own` is given, else the tracked rating's when it describes the
    current list and spec. `basis` says which."""
    comps, _withheld, _context, recipes = components(slug, fleet, audit_cache)
    if own is not None:
        j, basis = own.get("judgment"), "this document"
    else:
        tracked = load_json(deck_dir(slug) / RATING_FILE)
        j = (tracked or {}).get("judgment")
        if judgment_current(j, slug):
            basis = "rating.json judgment"
        else:
            j, basis = None, "skeleton"
    subs = {c["name"]: c["sub_score"] for c in comps if WEIGHTS.get(c["name"], 0) > 0}
    if j:
        subs["judgment"] = float(j["sub_score"])
    return {"slug": slug, "comps": recipes, "validity": {c["name"]: c.get("weight_validity", 1.0) for c in comps},
            "judgment": None if not j else float(j["sub_score"]), "subs": subs, "basis": basis}


def ranked(doc, fleet=None, audit_cache=None):
    """The group simulation: `score`, `score_ci`, the grade's probabilities and
    range, each component's interval, `rank` with its acceptability, and
    `concordance`."""
    fleet = fleet or _fleet()
    audit_cache = audit_cache if audit_cache is not None else {}
    _key, _basis, members = group_for(fleet, doc["slug"])
    prepared = [_member(m, fleet, audit_cache, own=doc if m == doc["slug"] else None) for m in sorted(members)]
    rng = random.Random(RATING_MC_SEED)
    draws, ranks = mc.simulate(prepared, rng, RATING_MC_K, WEIGHT_CONCENTRATION, WEIGHTS, JUDGMENT_SD)
    medians = {}
    for m in prepared:
        med, _lo, _hi = mc.interval(draws[m["slug"]]["scores"], RATING_MC_LEVEL)
        medians[m["slug"]] = None if med is None else round(med, 1)
    me = draws[doc["slug"]]
    med, lo, hi = mc.interval(me["scores"], RATING_MC_LEVEL)
    if med is not None:
        doc["score"] = round(med, 1)
        doc["score_ci"] = [round(lo, 1), round(hi, 1)]
        doc["grade"] = grade_of(doc["score"])
        doc["grade_range"] = [grade_of(lo), grade_of(hi)]
        doc["grade_probabilities"] = {g: round(p, 3) for g, p in
                                      mc.grade_probabilities(me["scores"], grade_of, GRADE_LETTERS).items()}
        for c in doc["components"]:
            xs = me["comps"].get(c["name"])
            if xs:
                _m, clo, chi = mc.interval(xs, RATING_MC_LEVEL)
                c["sub_ci"] = [round(clo, 2), round(chi, 2)]
        if doc.get("judgment") and me["judgment"]:
            _m, jlo, jhi = mc.interval(me["judgment"], RATING_MC_LEVEL)
            doc["judgment"]["sub_ci"] = [round(jlo, 2), round(jhi, 2)]
    doc["uncertainty"] = {
        "method": "Monte Carlo over a Dirichlet weight prior and the inputs' posteriors, joint within the group",
        "k": RATING_MC_K, "seed": RATING_MC_SEED, "kappa": WEIGHT_CONCENTRATION, "level": RATING_MC_LEVEL,
        "judgment_sd": JUDGMENT_SD, "placement_bin": "uniform over ±50/(n−1) percentile points",
        "dti_prior_read_sd": DTI_PRIOR_READ_SD,
        "priors": "every sd here is declared, not estimated (docs/known-issues.md §27)"}
    ordered = sorted((m for m in medians if medians[m] is not None), key=lambda m: (-medians[m], m))
    pos = ordered.index(doc["slug"]) + 1 if doc["slug"] in ordered else None
    doc["rank"] = {"position": pos, "of": len(ordered), "group": ordered,
                   "acceptability": mc.rank_acceptability(ranks[doc["slug"]], len(ordered)),
                   "basis": {m["slug"]: m["basis"] for m in prepared if m["slug"] in ordered},
                   "scores": {m: medians[m] for m in ordered}}
    doc["rank"]["acceptability"] = {k: round(v, 3) for k, v in doc["rank"]["acceptability"].items()}
    doc["concordance"] = mc.concordance({m["slug"]: m["subs"] for m in prepared}, doc["slug"],
                                        min_group=RATING_MIN_GROUP, mixed_at=CONCORDANCE_MIXED_AT,
                                        divergent_at=CONCORDANCE_DIVERGENT_AT, seed=RATING_MC_SEED,
                                        exact_max=KENDALL_EXACT_MAX, iterations=KENDALL_PERM_ITERATIONS)
    gr = doc.get("grade_range")
    if doc["concordance"]["level"] == "divergent" and gr and gr[0] != gr[1]:
        doc["grade_display"] = f"{gr[0]}–{gr[1]}"
    else:
        doc["grade_display"] = doc["grade"]
    if doc.get("score_ci"):
        lo, hi = doc["score_ci"]
        bounds = [f for _g, f in GRADES if lo < f <= hi]
        doc["resolution"] = (f"the {int(RATING_MC_LEVEL * 100)}% interval spans {len(bounds)} grade "
                             f"boundar{'y' if len(bounds) == 1 else 'ies'}"
                             + (f" ({', '.join(str(b) for b in bounds)})" if bounds else "")
                             + f"; concordance {doc['concordance']['level']}")
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
    """Spearman of the score against outcomes, with calibrate's refusal. The
    DTI cell uses the score WITHOUT the dti component (it is an input now)."""
    fleet = fleet or _fleet()
    board = leaderboard(fleet)
    rows = []
    for g, docs in board.items():
        for d in docs:
            r = _row(fleet, d["slug"])
            rec = _record(d["slug"])
            sv = r["survey"]
            loo = ((d.get("sensitivity") or {}).get("leave_one_out") or {}).get("dti")
            rows.append({"slug": d["slug"], "bracket": g, "score": d["score"], "code_score": d["code_score"],
                         "score_without_dti": loo["score"] if loo else d["score_point"],
                         "pilot_win_rate": (rec["wins"] / rec["games"]) if rec["games"] >= calibrate.MIN_GAMES else None,
                         "pilot_games": rec["games"],
                         "survey_power": (sv.get("power") or {}).get("mean") if not fa.is_absent(sv) else None,
                         "dti": None if fa.is_absent(r["dti"]) else r["dti"]["score"]["value"]})
    cells = {}
    for name, xkey, ykey in (("score_vs_pilot_win_rate", "score", "pilot_win_rate"),
                             ("score_vs_survey_power", "score", "survey_power"),
                             ("score_without_dti_vs_dti", "score_without_dti", "dti")):
        pairs = [(x[xkey], x[ykey]) for x in rows if x[xkey] is not None and x[ykey] is not None]
        cells[name] = fa._spearman_or_refusal(pairs, name, "more rated decks carrying the outcome")
    return {"rows": rows, "cells": cells, "min_decks": calibrate.MIN_DECKS, "min_games": calibrate.MIN_GAMES}


def _print_doc(doc):
    b = doc["bracket"]
    rk = doc.get("rank") or {}
    cc = doc.get("concordance") or {}
    print(f"RATING — {doc['slug']}   {doc['tier']} coaching")
    print(f"  {doc.get('grade_display') or '—'} in Bracket {b['key']} ({rk.get('position')} of {rk.get('of')})   "
          f"score {doc['score']}  [{doc['score_ci'][0] if doc['score_ci'] else '—'}, "
          f"{doc['score_ci'][1] if doc['score_ci'] else '—'}]   point {doc.get('score_point')}   code {doc['code_score']}")
    print(f"  {doc['label']}")
    for c in doc["components"]:
        v = c.get("weight_validity", 1.0)
        print(f"    {c['name']:14} {c['sub_score']:6.1f}  [{c['sub_ci'][0]:5.1f}, {c['sub_ci'][1]:5.1f}]  "
              f"w {c['weight']:.2f}{'×' + str(v) if v != 1.0 else ''} → {c['weight_effective']:.3f}   {c['method'][:60]}")
    if doc.get("judgment"):
        j = doc["judgment"]
        print(f"    {'judgment':14} {j['sub_score']:6.1f}  w {j['weight']:.2f} → {j['weight_effective']:.3f}   "
              f"{len(j.get('citations') or [])} citation(s)")
    for w in doc["withheld"]:
        print(f"    withheld {w['component']:12} — {w['reason']}")
    print(f"  {doc['resolution']}")
    if doc.get("grade_probabilities"):
        print("  grades: " + "  ".join(f"{g} {p:.0%}" for g, p in doc["grade_probabilities"].items() if p))
    if rk.get("acceptability"):
        print("  rank:   " + "  ".join(f"P({p}) {v:.0%}" for p, v in rk["acceptability"].items()))
    if cc.get("instruments"):
        print(f"  concordance {cc['level']} (spread {cc['spread']}): "
              + ", ".join(f"{i} #{int(r) if r == int(r) else r}" for i, r in cc["instruments"].items())
              + (f"; outliers {', '.join(o['instrument'] for o in cc['outliers'])}" if cc.get("outliers") else "")
              + (f"; group W {cc['group']['kendall_w']} p {cc['group']['p']}" if cc.get("group") else ""))
    sens = doc.get("sensitivity") or {}
    if sens.get("flips"):
        f = sens["flips"][0]
        print(f"  nearest flip: {f['component']} share {f['share_now']} → {f['share_at_flip']} "
              f"({f['direction']}) makes it {f['grade_at_flip']}")
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
                p1 = ((d.get("rank") or {}).get("acceptability") or {}).get("1")
                cc = (d.get("concordance") or {}).get("level")
                print(f"    {d.get('grade_display') or '—':5} {d['slug']:18} {str(d['score']):>6}  "
                      f"[{d['score_ci'][0] if d['score_ci'] else '—'}, {d['score_ci'][1] if d['score_ci'] else '—'}]  "
                      f"P(1st) {p1:.0%}  {cc}; {j}; withheld: {', '.join(w['component'] for w in d['withheld']) or '—'}"
                      if p1 is not None else
                      f"    {d.get('grade_display') or '—':5} {d['slug']:18} {str(d['score']):>6}  {cc}; {j}")
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
