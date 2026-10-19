"""Pilot: the rating's uncertainty, sensitivity and concordance (Rating v2, 2026-10-19).

WHY THIS EXISTS. The first fleet rating printed heliod as D 49.9 with an interval of
[49.9, 49.9] while DeckCheck's Threat Index read it the fleet's highest. Three things
hid in that number: the weights were one vector nobody had estimated, a placement
within a group of three was a 0/50/100 scale given half of every component, and the
printed interval summed the inputs' interval ends and so ignored both. The pilot's
read — "the large delta is an anti-signal between metrics" — is a measurement the
rating should make itself. This module makes it.

THE WEIGHTS ARE A PRIOR. `rating.WEIGHTS` is the mean of a Dirichlet with
concentration `WEIGHT_CONCENTRATION`; `simulate` draws K weight vectors per bracket
group, SHARED across the group's members within a draw so the ranks are joint, and
renormalises each member's draw over the components it can measure (the aggregation
property: a Dirichlet renormalised over a subset is the Dirichlet over that subset's
alphas, so one prior serves every withheld pattern). Forge at weight 0.00 is a rule,
not a prior, and stays out of the draw.

THE INPUTS ARE DRAWN, NOT FIXED. A rate anchor is a Beta posterior from its Wilson k
and n; a placement is uniform over its percentile bin (±50/(n−1): a rank is
interval-censored to its step, and at n=3 that is ±25 points — the honest width of a
three-point scale); the DTI is Normal around its reading; the judgment is Normal with
`JUDGMENT_SD`; bands and counts are fixed. Every one of those sds is a DECLARED PRIOR,
named in `uncertainty`, not an estimate — `docs/known-issues.md` §27.

WHAT COMES OUT. Per deck: the score's median and its 90% interval, the probability
of each grade, the probability of each rank within the group (SMAA's rank
acceptability), each component's 90% interval; `sensitivity`, in closed form, says
what the score is without each component and at what weight share the grade flips;
`concordance` ranks the members by every instrument and reports how far this deck's
instruments disagree, with Kendall's W over the group and its permutation p. A
divergent deck prints its grade as a range, never a letter.

Seeded (`RATING_MC_SEED`), one generator per group, fixed draw order, so a run is
byte-identical on one machine. `gammavariate` and `gauss` call libm, so another
platform can differ by an ulp; the validator recomputes where it validates.
"""

import math
import statistics

from manamap.sim import stats


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def placement_weight(n, shrink_k):
    """λ(n) = (n−1)/(n−1+K): the share of a component a within-group placement
    earns at group size n. A placement is a proportion estimated from n−1 pairwise
    comparisons; shrinking it toward the cited anchor with K pseudo-comparisons is
    the Beta-binomial posterior-mean form. n=3 → 1/3 at K=4; n=5 → 1/2; n=9 → 2/3."""
    if n is None or n < 2:
        return 0.0
    return (n - 1) / (n - 1 + shrink_k)


# ── Draws ────────────────────────────────────────────────────────────────


def draw_component(rng, recipe):
    """One draw of a component's 0–100 sub-score from its recipe:
    `{"anchor": {"kind": fixed|rate|normal, …} | None, "place": float | None,
      "bin": float, "lam": float}`. The draw order is fixed by the recipe's shape."""
    anchor = None
    a = recipe.get("anchor")
    if a is not None:
        if a["kind"] == "fixed":
            anchor = a["value"]
        elif a["kind"] == "rate":
            p = stats.beta_rate(rng, a["k"], a["n"])
            anchor = clamp(a["a"] + a["b"] * p)
        elif a["kind"] == "normal":
            anchor = clamp(rng.gauss(a["mean"], a["sd"]))
        else:
            raise ValueError(f"unknown anchor kind {a['kind']!r}")
    place = None
    if recipe.get("place") is not None:
        b = recipe["bin"]
        place = clamp(rng.uniform(recipe["place"] - b, recipe["place"] + b))
    lam = recipe["lam"]
    if place is not None and anchor is not None:
        return lam * place + (1.0 - lam) * anchor
    return place if place is not None else anchor


def simulate(members, rng, k, kappa, weights, judgment_sd):
    """The group simulation. `members` is a list, in the order to draw, of
    `{"slug", "comps": {name: recipe}, "validity": {name: multiplier},
    "judgment": sub_score | None}`. Returns `(draws, ranks)`: per slug the
    score draws, each component's draws and the judgment's; and per slug the
    rank it took in each draw."""
    names = [n for n, w in weights.items() if w > 0]
    alphas = [kappa * weights[n] for n in names]
    draws = {m["slug"]: {"scores": [], "comps": {n: [] for n in m["comps"] if weights.get(n, 0) > 0},
                         "judgment": []} for m in members}
    ranks = {m["slug"]: [] for m in members}
    for _ in range(k):
        w = dict(zip(names, stats.dirichlet(rng, alphas)))
        this = {}
        for m in members:
            num = den = 0.0
            for name in sorted(m["comps"]):
                if weights.get(name, 0) <= 0:
                    continue
                s = draw_component(rng, m["comps"][name])
                draws[m["slug"]]["comps"][name].append(s)
                wv = w[name] * m["validity"].get(name, 1.0)
                num += wv * s
                den += wv
            if m["judgment"] is not None:
                js = clamp(rng.gauss(m["judgment"], judgment_sd))
                draws[m["slug"]]["judgment"].append(js)
                num += w["judgment"] * js
                den += w["judgment"]
            sc = (num / den) if den > 0 else None
            draws[m["slug"]]["scores"].append(sc)
            this[m["slug"]] = sc
        order = sorted((s for s in this if this[s] is not None), key=lambda s: (-this[s], s))
        for pos, s in enumerate(order, 1):
            ranks[s].append(pos)
    return draws, ranks


def interval(xs, level):
    """`(median, lo, hi)` of a list of draws at `level` (0.90 → q05, q95)."""
    ys = sorted(x for x in xs if x is not None)
    if not ys:
        return None, None, None
    tail = (1.0 - level) / 2.0
    return statistics.median(ys), stats.quantile(ys, tail), stats.quantile(ys, 1.0 - tail)


def grade_probabilities(scores, grade_of, letters):
    xs = [x for x in scores if x is not None]
    counts = {g: 0 for g in letters}
    for x in xs:
        counts[grade_of(x)] += 1
    return {g: (counts[g] / len(xs) if xs else None) for g in letters}


def rank_acceptability(ranks, of):
    if not ranks:
        return {}
    return {str(p): sum(1 for r in ranks if r == p) / len(ranks) for p in range(1, of + 1)}


# ── Sensitivity, closed form ────────────────────────────────────────────


def sensitivity(entries, grades, grade_of):
    """`entries`: `[(name, weight, sub_score)]` with the weights already
    multiplied by validity — the deterministic point score's parts. The score
    is linear in each component's SHARE u with the rest renormalised,
    `S(u) = u·s_i + (1−u)·S_rest`, so the share at which it meets a grade
    boundary B is `u* = (B − S_rest)/(s_i − S_rest)` — exact, no search.
    Returns `{"leave_one_out": {name: {score, grade}}, "flips": [{component,
    share_now, share_at_flip, boundary, grade_at_flip, direction}]}` with one
    flip per component, the nearest in-range boundary, sorted by distance."""
    total = sum(w for _n, w, _s in entries)
    if total <= 0:
        return {"leave_one_out": {}, "flips": []}
    loo, flips = {}, []
    for name, w, s in entries:
        rest_w = total - w
        rest_num = sum(w2 * s2 for n2, w2, s2 in entries if n2 != name)
        if rest_w <= 0:
            continue
        s_rest = rest_num / rest_w
        loo[name] = {"score": round(s_rest, 1), "grade": grade_of(s_rest)}
        share = w / total
        if abs(s - s_rest) < 1e-9:
            continue
        best = None
        for _g, floor in grades:
            if floor <= 0:
                continue
            u = (floor - s_rest) / (s - s_rest)
            if not 0.0 < u < 1.0:
                continue
            rises = (u > share) == (s > s_rest)
            cand = {"component": name, "share_now": round(share, 3), "share_at_flip": round(u, 3),
                    "boundary": floor, "grade_at_flip": grade_of(floor) if rises else grade_of(floor - 0.05),
                    "direction": "raise" if u > share else "lower"}
            if best is None or abs(u - share) < abs(best["share_at_flip"] - share):
                best = cand
        if best is not None:
            flips.append(best)
    flips.sort(key=lambda f: (abs(f["share_at_flip"] - f["share_now"]), f["component"]))
    return {"leave_one_out": loo, "flips": flips}


# ── Concordance ─────────────────────────────────────────────────────────


def concordance(subs, slug, *, min_group, mixed_at, divergent_at, seed, exact_max, iterations):
    """`subs`: `{member: {instrument: sub_score}}` — every component present on
    each member plus `judgment` when it has one. Each instrument carried by at
    least `min_group` members ranks the members (1 best). For `slug`: its rank
    under each instrument, the spread `(max − min)/(n − 1)`, a level, the
    outlier instruments (the smaller side of its median rank; both sides on a
    tie), and the group's Kendall W with its permutation p."""
    members = sorted(subs)
    n = len(members)
    if n < min_group:
        return {"level": "withheld", "why": f"group of {n} below {min_group}", "instruments": {},
                "of": n, "spread": None, "outliers": [], "group": None}
    instruments = sorted({i for m in members for i in subs[m]})
    rankings = {}
    for inst in instruments:
        vals = {m: subs[m].get(inst) for m in members}
        if sum(1 for v in vals.values() if v is not None) < min_group:
            continue
        rankings[inst] = stats.rank_with_ties(vals, higher_is_better=True)
    mine = {inst: r[slug] for inst, r in rankings.items() if slug in r}
    # The judgment is ranked and shown, and it counts toward the group's W, but
    # the spread and the outlier sides are over the MEASURED instruments: the
    # judgment is the thing being asked to explain the disagreement, so it
    # cannot be its own outlier (and with it in, a three-against-three tie at
    # n=3 listed every instrument).
    measured = {i: r for i, r in mine.items() if i != "judgment"}
    if len(measured) < 2:
        return {"level": "withheld", "why": "fewer than two instruments rank this deck", "instruments": mine,
                "of": n, "spread": None, "outliers": [], "group": None}
    spread = (max(measured.values()) - min(measured.values())) / (n - 1)
    level = "divergent" if spread >= divergent_at else ("mixed" if spread >= mixed_at else "concordant")
    outliers = []
    if level != "concordant":
        med = statistics.median(measured.values())
        better = sorted(i for i, r in measured.items() if r < med)
        worse = sorted(i for i, r in measured.items() if r > med)
        sides = []
        if better and (not worse or len(better) <= len(worse)):
            sides.append(("better", better))
        if worse and (not better or len(worse) <= len(better)):
            sides.append(("worse", worse))
        for side, names in sides:
            outliers += [{"instrument": i, "rank": measured[i], "side": side} for i in names]
    full = [r for r in rankings.values() if len(r) == n]
    group = None
    if len(full) >= 2:
        w = stats.kendall_w(full)
        p, method = stats.kendall_w_p(full, seed=seed, exact_max=exact_max, iterations=iterations)
        group = {"kendall_w": None if w is None else round(w, 3), "p": None if p is None else round(p, 4),
                 "p_method": method, "m": len(full), "n": n}
    return {"level": level, "instruments": {i: mine[i] for i in sorted(mine)}, "of": n,
            "spread": round(spread, 3), "outliers": outliers, "group": group}
