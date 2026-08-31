"""Small-sample statistics for the experiment harness, hand-implemented.

WHY THERE IS NO SCIPY HERE. Everything below is closed-form, enumerable, or a
thirty-row table. `pyproject.toml` pins Python to 3.10 against torch, `numpy<2`
and `sentence-transformers<4` — a graph fragile enough that adding a dependency
for two quantile functions is a poor trade. The better reason is the repo's own
claim: every figure is arithmetic a reader can re-derive. A sixty-line module
with its own tests keeps that promise in a way an import does not.

WHAT THIS REPLACES. `experiment.delta()` compared eleven figures and tested ONE,
by asking whether two 95% intervals overlapped — and then reading an overlap as
"the difference is noise". That is the overlap fallacy: non-overlap does imply a
difference at the 0.05 level, but overlap implies nothing at all, because two
intervals can overlap while the interval on their DIFFERENCE excludes zero. The
fix is to stop comparing intervals and put an interval on the difference itself,
which is also the quantity anyone actually wants.

THE POWER FUNCTIONS ARE THE POINT. At the sizes these experiments run — ten or
twenty games an arm — almost nothing is detectable, and an artifact that reports
"no significant difference" without saying what it COULD have detected is
telling the reader half a fact. `mde_proportion` answers the other half exactly,
by enumerating the two-binomial grid rather than leaning on a normal
approximation that is wrong at n=12 and wrong in a particular way at the p=0
boundary — which is where the one real experiment on disk actually sits, arm A
having won 0 of 12.
"""

import functools
import math
import random

# Two-sided 97.5% quantiles of Student's t, by degrees of freedom. A table
# because it is auditable: every value here can be checked against any printed
# table in a minute, which is not true of a series expansion. Beyond 30 the
# normal quantile is within 0.4%, and these samples never get there.
T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
Z975 = 1.96


def t_crit(df):
    """The 97.5% two-sided critical value for `df` degrees of freedom."""
    if df < 1:
        return None
    return T975.get(int(df), Z975) if df <= 30 else Z975


def wilson_bounds(k, n, z=Z975):
    """Wilson score interval, UNROUNDED. `(None, None)` when n == 0.

    `parse.wilson` is the rounded, reporting-facing version and delegates here so
    there is one implementation. Newcombe's interval below is built out of these
    bounds, and rounding them to three places first puts that error straight into
    the difference — small, but there is no reason to accept it.
    """
    if not n:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def diff_proportions(k_a, n_a, k_b, n_b, z=Z975):
    """95% interval for (p_b - p_a): Newcombe (1998) method 10.

    Built out of the two Wilson intervals rather than from a pooled normal
    approximation, which is what gives it usable coverage at these sample sizes
    and at the boundary. It is six lines and it is the whole fix.
    """
    if not n_a or not n_b:
        return None
    p_a, p_b = k_a / n_a, k_b / n_b
    l1, u1 = wilson_bounds(k_a, n_a, z)
    l2, u2 = wilson_bounds(k_b, n_b, z)
    d = p_b - p_a
    lower = d - math.sqrt((p_b - l2) ** 2 + (u1 - p_a) ** 2)
    upper = d + math.sqrt((u2 - p_b) ** 2 + (p_a - l1) ** 2)
    return {"diff": round(d, 4), "ci95": [round(max(-1.0, lower), 4),
                                          round(min(1.0, upper), 4)],
            "excludes_zero": lower > 0 or upper < 0,
            "method": "Newcombe score interval on the difference of proportions"}


def _mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def diff_means(xs, ys):
    """95% interval for (mean(ys) - mean(xs)): Welch, with a t critical value.

    Welch rather than pooled because the arms are two different decklists and
    there is no reason to assume they have the same variance — the whole point of
    the experiment is that one of them behaves differently. The t quantile rather
    than 1.96 because at n=10 per arm the normal is anticonservative by about 12%
    on the half-width, which is exactly the range where a reader would be misled.
    """
    xs = [x for x in xs if x is not None]
    ys = [y for y in ys if y is not None]
    if len(xs) < 2 or len(ys) < 2:
        return None
    ma, sa = _mean_sd(xs)
    mb, sb = _mean_sd(ys)
    va, vb = sa ** 2 / len(xs), sb ** 2 / len(ys)
    se = math.sqrt(va + vb)
    d = mb - ma
    if se == 0:
        # Both arms constant. A zero-width interval is the honest answer and the
        # only one available; saying "no difference detected" would be a claim
        # about power that nothing here supports.
        return {"diff": round(d, 4), "ci95": [round(d, 4), round(d, 4)],
                "excludes_zero": d != 0, "df": None,
                "method": "both arms constant; no variance to estimate"}
    df = (va + vb) ** 2 / (va ** 2 / (len(xs) - 1) + vb ** 2 / (len(ys) - 1))
    half = t_crit(df) * se
    return {"diff": round(d, 4), "ci95": [round(d - half, 4), round(d + half, 4)],
            "excludes_zero": (d - half) > 0 or (d + half) < 0,
            "df": round(df, 1),
            "method": "Welch t interval on the difference of means"}


def permutation_p(xs, ys, seed=0, iterations=10000):
    """Two-sided permutation p for a difference of means. Seeded, so it replays.

    Assumes nothing about the distribution, which matters here: the samples that
    motivated this work look like `0 0 0 0 0 0 0 0 0 0 31 178`, and a t interval
    on that is a true number describing no game.
    """
    xs = [x for x in xs if x is not None]
    ys = [y for y in ys if y is not None]
    if len(xs) < 2 or len(ys) < 2:
        return None
    observed = abs(sum(ys) / len(ys) - sum(xs) / len(xs))
    pool = xs + ys
    n = len(xs)
    rng = random.Random(seed)
    hits = 0
    for _ in range(iterations):
        rng.shuffle(pool)
        a, b = pool[:n], pool[n:]
        if abs(sum(b) / len(b) - sum(a) / len(a)) >= observed - 1e-12:
            hits += 1
    # The +1s are Phipson & Smyth: a permutation p of exactly 0 is not a
    # possible truth, it is a statement that the sample was too small to produce
    # a more extreme rearrangement.
    return round((hits + 1) / (iterations + 1), 4)


def diff_medians(xs, ys, seed=0, iterations=10000):
    """Percentile-bootstrap interval for the difference of medians. Seeded."""
    xs = [x for x in xs if x is not None]
    ys = [y for y in ys if y is not None]
    if len(xs) < 2 or len(ys) < 2:
        return None

    def med(v):
        o = sorted(v)
        m = len(o) // 2
        return o[m] if len(o) % 2 else (o[m - 1] + o[m]) / 2

    rng = random.Random(seed)
    d = med(ys) - med(xs)
    draws = []
    for _ in range(iterations):
        a = [xs[rng.randrange(len(xs))] for _ in range(len(xs))]
        b = [ys[rng.randrange(len(ys))] for _ in range(len(ys))]
        draws.append(med(b) - med(a))
    draws.sort()
    lo = draws[int(0.025 * iterations)]
    hi = draws[min(int(0.975 * iterations), iterations - 1)]
    return {"diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": lo > 0 or hi < 0,
            "method": f"percentile bootstrap, {iterations} resamples, seed {seed}"}


# ── Power: what could this experiment have found? ───────────────────────────

#: MEMOISED BECAUSE THE SWEEP ASKS FOR THE SAME VECTOR TWO HUNDRED TIMES.
#: `mde_proportion` walks p_b from p_a to 1.0 in 0.005 steps, calling
#: `power_for` at each step — and p_a and n_a never move, so the arm-A vector is
#: recomputed identically every time. Profiled on the worst tracked case
#: (edgar-vampires@bloodline `net-change`): 990,066 `math.comb` calls, 27.4s of
#: a 54.2s build. Pure function, so the cache is exact rather than approximate.
@functools.lru_cache(maxsize=200_000)
def _binom_pmf(k, n, p):
    if p <= 0:
        return 1.0 if k == 0 else 0.0
    if p >= 1:
        return 1.0 if k == n else 0.0
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


# MEASURED, because it bears on how the result may be reported: Newcombe's
# method is NOT uniformly conservative. Type I error at p_b == p_a runs 0.009 to
# 0.064 over p in {0.1 … 0.7} and n in {10 … 50}, worst at p=0.5, n=12 — the
# sawtooth of a discrete distribution, whose attainable rates step past 0.05
# rather than landing on it. Mean coverage is close to nominal, which is what the
# method promises; a per-point guarantee is not. `tests/test_sim_stats.py` pins
# the range. It is one more reason only `win_rate` gets a verdict.


def _significant_grid(n_a, n_b, z=Z975):
    """Which (k_a, k_b) outcomes the test would call a difference.

    Depends ONLY on the counts, never on the true rates — so it is computed once
    and reused for every candidate rate below. That is what keeps exact power
    cheap enough to do by enumeration instead of approximating it.
    """
    return [[diff_proportions(ka, n_a, kb, n_b, z)["excludes_zero"]
             for kb in range(n_b + 1)] for ka in range(n_a + 1)]


def power_for(p_a, p_b, n_a, n_b, grid=None, z=Z975):
    """Exact probability that the test calls a difference, given true rates.

    No normal approximation anywhere: this sums the actual two-binomial
    distribution over the outcomes that would be called significant. It is
    therefore correct at n=12 and correct at p=0, which the textbook formula is
    not — and p_a = 0 is where the one real experiment on disk sits.
    """
    grid = grid if grid is not None else _significant_grid(n_a, n_b, z)
    # HOIST THE ARM-B VECTOR. It does not depend on `ka`, and computing it in
    # the inner loop evaluated the same n_b+1 terms once per surviving ka — the
    # quadratic factor behind the profile above. The accumulation below is left
    # in its original order, term for term, so the sum is bit-identical.
    pb_vec = [_binom_pmf(kb, n_b, p_b) for kb in range(n_b + 1)]
    total = 0.0
    for ka in range(n_a + 1):
        pa = _binom_pmf(ka, n_a, p_a)
        if pa < 1e-15:
            continue
        row = grid[ka]
        for kb in range(n_b + 1):
            if row[kb]:
                total += pa * pb_vec[kb]
    return total


def mde_proportion(p_a, n_a, n_b=None, target_power=0.8, z=Z975, step=0.005):
    """The smallest rate for arm B this experiment could reliably detect.

    Returns `{minimum_detectable_rate_b, minimum_detectable_difference,
    achieved_power}`, or None if even a certainty (p_b = 1) would not reach the
    target — which is itself worth reporting rather than hiding.
    """
    n_b = n_a if n_b is None else n_b
    grid = _significant_grid(n_a, n_b, z)
    p = p_a
    while p <= 1.0 + 1e-9:
        power = power_for(p_a, min(p, 1.0), n_a, n_b, grid, z)
        if power >= target_power:
            return {"minimum_detectable_rate_b": round(min(p, 1.0), 4),
                    "minimum_detectable_difference": round(min(p, 1.0) - p_a, 4),
                    "achieved_power": round(power, 4)}
        p += step
    return None


def games_for_difference(p_a, difference, target_power=0.8, z=Z975, max_n=1000):
    """Games per arm needed to detect a given difference. None beyond `max_n`.

    Doubling until it passes, then a binary search — the grid is O(n^2) to build,
    so a linear scan to four hundred is minutes and this is milliseconds.
    """
    p_b = p_a + difference
    if not 0 <= p_b <= 1:
        return None

    def ok(n):
        return power_for(p_a, p_b, n, n, _significant_grid(n, n, z), z) >= target_power

    n = 8
    while n <= max_n and not ok(n):
        n *= 2
    if n > max_n:
        return None
    lo, hi = n // 2, n
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi
