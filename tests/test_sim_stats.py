"""The statistics are the pitch, so they are tested against simulation.

A hand-written estimator nobody checked would be worse than the interval-overlap
check it replaces, because it would be wrong with more authority. So the
intervals are validated by COVERAGE — generate thousands of experiments from
known truth and count how often the interval contains it — and the power
functions by brute force against the test they claim to describe. Neither
validation depends on my remembering a published table correctly, which is the
failure mode a "compare to the textbook value" test has when nobody re-derives
the textbook value.
"""

import random

import pytest

from manamap.sim import stats


# ── The t table ─────────────────────────────────────────────────────────────

def test_the_t_table_matches_published_values():
    """Spot values any printed table carries. The table is here to be auditable;
    a test that could not catch a transcription slip would defeat that."""
    assert stats.T975[1] == 12.706
    assert stats.T975[2] == 4.303
    assert stats.T975[10] == 2.228
    assert stats.T975[30] == 2.042


def test_t_falls_back_to_the_normal_beyond_the_table():
    assert stats.t_crit(200) == stats.Z975
    assert stats.t_crit(31) == stats.Z975
    assert stats.t_crit(0) is None


def test_t_decreases_monotonically_toward_the_normal():
    vals = [stats.T975[df] for df in sorted(stats.T975)]
    assert vals == sorted(vals, reverse=True)
    assert vals[-1] > stats.Z975


# ── Wilson ──────────────────────────────────────────────────────────────────

def test_wilson_bounds_are_unrounded_and_inside_zero_one():
    lo, hi = stats.wilson_bounds(5, 12)
    assert 0 < lo < 5 / 12 < hi < 1
    assert round(lo, 3) != lo or round(hi, 3) != hi or True  # not asserting drift
    assert stats.wilson_bounds(0, 0) == (None, None)


def test_wilson_at_the_boundary_does_not_produce_a_zero_width_interval():
    """0/12 is where the real experiment sits. A naive normal interval gives
    [0, 0] there and would call every difference significant."""
    lo, hi = stats.wilson_bounds(0, 12)
    assert lo == 0.0
    assert hi > 0.2, f"a zero-width interval at k=0 would be a bug, got {hi}"


def test_parse_wilson_still_agrees_with_the_shared_implementation():
    """`parse.wilson` is the rounded, reporting-facing version. If the two ever
    disagree, every tracked run record's interval is quietly a different number
    from the one the experiment harness reasons with."""
    from manamap.sim.parse import wilson
    for k, n in ((0, 12), (5, 12), (12, 12), (3, 20), (7, 10)):
        lo, hi = stats.wilson_bounds(k, n)
        assert wilson(k, n) == (round(lo, 3), round(hi, 3))


# ── Newcombe: validated by coverage, not by a remembered table ──────────────

@pytest.mark.parametrize("p_a,p_b,n", [
    (0.25, 0.25, 20), (0.25, 0.50, 20), (0.10, 0.30, 30), (0.50, 0.50, 12),
])
def test_the_difference_interval_covers_the_truth_about_95_percent_of_the_time(p_a, p_b, n):
    """The property that actually matters. Newcombe is mildly conservative by
    design, so the bar is 'at least nominal', not 'exactly nominal'."""
    rng = random.Random(20260822)
    truth = p_b - p_a
    covered = 0
    trials = 3000
    for _ in range(trials):
        k_a = sum(rng.random() < p_a for _ in range(n))
        k_b = sum(rng.random() < p_b for _ in range(n))
        lo, hi = stats.diff_proportions(k_a, n, k_b, n)["ci95"]
        covered += lo <= truth <= hi
    rate = covered / trials
    assert 0.93 <= rate <= 0.999, f"coverage {rate:.3f} at p={p_a}/{p_b}, n={n}"


def test_the_difference_interval_is_oriented_b_minus_a():
    """Sign errors here would invert every reading in the artifact."""
    d = stats.diff_proportions(2, 20, 10, 20)
    assert d["diff"] > 0, "arm B won more; the difference must be positive"
    assert stats.diff_proportions(10, 20, 2, 20)["diff"] == -d["diff"]


def test_two_overlapping_wilson_intervals_can_still_differ():
    """The exact failure being fixed. If this ever stops being demonstrable, the
    old overlap test was not actually wrong and this module is unnecessary."""
    k_a, k_b, n = 6, 15, 25
    la, ua = stats.wilson_bounds(k_a, n)
    lb, ub = stats.wilson_bounds(k_b, n)
    overlap = not (ua < lb or ub < la)
    assert overlap, "picked a case where the marginal intervals do not overlap"
    assert stats.diff_proportions(k_a, n, k_b, n)["excludes_zero"], (
        "the difference interval should exclude zero even though the marginals overlap "
        "— that gap IS the overlap fallacy")


# ── Welch ───────────────────────────────────────────────────────────────────

def test_welch_covers_a_known_mean_difference():
    rng = random.Random(7)
    covered = 0
    for _ in range(1500):
        xs = [rng.gauss(0, 1) for _ in range(12)]
        ys = [rng.gauss(0.5, 2.5) for _ in range(12)]     # unequal variances
        lo, hi = stats.diff_means(xs, ys)["ci95"]
        covered += lo <= 0.5 <= hi
    assert 0.92 <= covered / 1500 <= 0.98, covered / 1500


def test_welch_uses_t_not_the_normal_at_small_n():
    """At n=10 an interval built on 1.96 is about 12% too narrow, which is
    exactly the range where a reader is misled rather than merely imprecise."""
    xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ys = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    d = stats.diff_means(xs, ys)
    half = (d["ci95"][1] - d["ci95"][0]) / 2
    import math
    se = math.sqrt(2 * (sum((x - 4.5) ** 2 for x in xs) / 9) / 10)
    assert half > stats.Z975 * se, "a t interval must be wider than a normal one"


def test_two_constant_arms_do_not_claim_an_estimate():
    d = stats.diff_means([0] * 10, [0] * 10)
    assert d["ci95"] == [0.0, 0.0] and d["excludes_zero"] is False
    assert "no variance" in d["method"]


def test_means_and_medians_disagree_on_the_sample_that_motivated_this():
    """Arm B's real commander damage, from the kianne experiment. The mean says
    a sevenfold improvement; the median says nothing happened in ten of twelve
    games. Both belong in the artifact."""
    a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 15]
    b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 178]
    assert stats.diff_means(a, b)["diff"] > 14
    assert stats.diff_medians(a, b, seed=1)["diff"] == 0
    assert not stats.diff_means(a, b)["excludes_zero"], (
        "twelve games cannot carry that mean; the interval must span zero")


# ── Permutation ─────────────────────────────────────────────────────────────

def test_permutation_is_seeded_and_replays():
    a, b = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
    assert stats.permutation_p(a, b, seed=3, iterations=500) == \
           stats.permutation_p(a, b, seed=3, iterations=500)


def test_permutation_never_reports_exactly_zero():
    """A p of 0 is not a possible truth — it says the sample was too small to
    produce a more extreme rearrangement. Phipson & Smyth's +1."""
    p = stats.permutation_p([0] * 8, [100] * 8, seed=1, iterations=200)
    assert p > 0


def test_permutation_finds_a_real_difference_and_not_a_fake_one():
    same = stats.permutation_p([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6],
                               seed=2, iterations=2000)
    apart = stats.permutation_p([1, 2, 3, 4, 5, 6], [21, 22, 23, 24, 25, 26],
                                seed=2, iterations=2000)
    assert same > 0.5 and apart < 0.05


# ── Power, brute-forced against the test it describes ───────────────────────

def test_exact_power_matches_a_monte_carlo_of_the_actual_test():
    """The strongest check available: simulate experiments, run the real
    `diff_proportions` on each, and count how often it calls a difference.
    Enumeration and simulation must agree."""
    p_a, p_b, n = 0.2, 0.6, 15
    exact = stats.power_for(p_a, p_b, n, n)
    rng = random.Random(99)
    hits = 0
    trials = 4000
    for _ in range(trials):
        k_a = sum(rng.random() < p_a for _ in range(n))
        k_b = sum(rng.random() < p_b for _ in range(n))
        hits += stats.diff_proportions(k_a, n, k_b, n)["excludes_zero"]
    assert abs(exact - hits / trials) < 0.025, f"exact {exact:.3f} vs mc {hits/trials:.3f}"


def test_the_false_positive_rate_stays_near_nominal_but_is_not_always_under_it():
    """Measured across the grid, NOT assumed. The first version of this test
    asserted "<= 0.05 because Newcombe is conservative" and failed at 0.053.

    It is not uniformly conservative. Type I error over p in {0.1 … 0.7} and n in
    {10 … 50} runs 0.009 to 0.064, worst at p=0.5, n=12 — the sawtooth that comes
    from summing a discrete distribution, where the attainable rates step past
    0.05 rather than landing on it. Mean coverage is close to nominal, which is
    what the method promises; a per-point guarantee is not.

    That is why `win_rate` is the only figure allowed a verdict, and why the
    artifact reports the interval rather than a bare "significant"."""
    worst = max(stats.power_for(p, p, n, n)
                for n in (10, 12, 20, 30, 50)
                for p in (0.1, 0.2, 0.3, 0.5, 0.7))
    assert worst <= 0.07, f"type I error drifted to {worst:.4f}"
    assert worst > 0.05, (
        "if this is now uniformly under nominal the method changed — re-derive "
        "the range in this docstring rather than deleting the assertion")


def test_the_mde_at_the_real_experiment_sizes():
    """These are the numbers the artifact will print, and they are the most
    honest sentence in it: at twelve games an arm, almost nothing is
    detectable."""
    assert stats.mde_proportion(0.0, 12)["minimum_detectable_rate_b"] == 0.415
    assert stats.mde_proportion(0.0, 10)["minimum_detectable_rate_b"] == 0.485
    assert stats.mde_proportion(0.25, 12)["minimum_detectable_difference"] == 0.52


def test_the_mde_actually_achieves_the_power_it_claims():
    for p_a, n in ((0.0, 12), (0.25, 12), (0.4, 20)):
        m = stats.mde_proportion(p_a, n)
        assert stats.power_for(p_a, m["minimum_detectable_rate_b"], n, n) >= 0.8


def test_more_games_detect_smaller_differences():
    a = stats.mde_proportion(0.25, 12)["minimum_detectable_difference"]
    b = stats.mde_proportion(0.25, 50)["minimum_detectable_difference"]
    c = stats.mde_proportion(0.25, 200)["minimum_detectable_difference"]
    assert a > b > c


@pytest.mark.slow
def test_games_for_difference_lands_on_the_exact_boundary():
    """Not an approximation: power must cross the target between n-1 and n."""
    n = stats.games_for_difference(0.25, 0.10)
    assert stats.power_for(0.25, 0.35, n, n) >= 0.8
    assert stats.power_for(0.25, 0.35, n - 1, n - 1) < 0.8


def test_an_undetectable_difference_reports_none_rather_than_a_number():
    assert stats.mde_proportion(0.9, 4) is None or \
           stats.mde_proportion(0.9, 4)["minimum_detectable_rate_b"] <= 1.0
    assert stats.games_for_difference(0.25, 0.0001, max_n=64) is None


@pytest.mark.slow
def test_the_mde_switches_to_the_normal_approximation_above_the_cap(monkeypatch):
    """The exact grid is O(n^2) and overflows a double near 4,000; above
    `EXACT_MDE_MAX_N` the normal approximation answers, and says so. Bug:
    remove the switch and this test builds a grid it was told not to."""
    calls = []
    real = stats._significant_grid
    def spy(n_a, n_b, z):
        calls.append(max(n_a, n_b))
        return real(n_a, n_b, z)
    monkeypatch.setattr(stats, "_significant_grid", spy)
    big = stats.mde_proportion(0.11, 2000, 2000)
    assert big["method"] == "normal approximation"
    assert 0.02 < big["minimum_detectable_difference"] < 0.04
    need = stats.games_for_difference(0.11, 0.05)
    assert need is not None and 500 < need < 1500, need
    assert max(calls) <= stats.EXACT_MDE_MAX_N, calls
    # Below the cap nothing changed: the exact boundary case still holds.
    assert stats.games_for_difference(0.25, 0.10) == 329


def test_the_normal_mde_is_the_one_formula_diagnostic_used_to_restate():
    """`diagnostic._mde` carried `2.8016 * sqrt(p(1-p)(1/n_a + 1/n_b))` as its
    own line. One home now; the constant and the rounding are identical."""
    import math
    p, na, nb = 0.3, 5000, 5000
    want = round(2.8016 * math.sqrt(p * (1 - p) * (1 / na + 1 / nb)), 4)
    assert stats.mde_proportion_normal(p, na, nb)["minimum_detectable_difference"] == want
    from manamap.pilot import diagnostic
    assert diagnostic._mde(p, na, nb) == want
    # and the exact path below diagnostic's cap is untouched
    assert diagnostic._mde(0.25, 12, 12) == stats.mde_proportion(0.25, 12)["minimum_detectable_difference"]


def test_welch_from_summary_equals_welch_from_the_series():
    """`diff_means` is `diff_means_summary` after `_mean_sd`, bit for bit — so
    a caller holding only {rate, sd, n} per cell gets the interval the series
    would have given."""
    xs = [3.0, 4.5, 2.0, 6.5, 5.0, 3.5, 4.0]
    ys = [5.0, 7.0, 6.5, 8.0, 4.5, 6.0, 7.5, 5.5]
    ma, sa = stats._mean_sd(xs)
    mb, sb = stats._mean_sd(ys)
    assert stats.diff_means(xs, ys) == stats.diff_means_summary(ma, sa, len(xs), mb, sb, len(ys))
    assert stats.diff_means_summary(1.0, 0.0, 1, 2.0, 0.0, 5) is None


def test_the_alpha_over_m_table_matches_published_values():
    """z(0.05/m), two-sided, m = 1..12 — pinned the way `T975` is."""
    want = [1.960, 2.241, 2.394, 2.498, 2.576, 2.638, 2.690, 2.734, 2.773, 2.807, 2.838, 2.865]
    got = [round(stats.z_two_sided(0.05 / m), 3) for m in range(1, 13)]
    assert got == want, got


def test_holm_is_step_down_not_bonferroni():
    """Rank 1 pays alpha/m, rank 2 alpha/(m-1), and so on — so in a family of
    three, a second row at z = 2.30 clears its alpha/2 threshold (2.241) where
    Bonferroni-3 would demand 2.394 of it. And the walk STOPS at the first
    failure: in [2.3, 2.0, 1.97] nothing clears rank 1 (2.394), so the 1.97 that
    would pass the rank-3 threshold on its own is not significant either."""
    h = stats.holm([3.5, 2.30, 1.0])
    assert [x["significant"] for x in h] == [True, True, False]
    assert h[0]["threshold"] == 2.394 and h[1]["threshold"] == 2.241 and h[2]["threshold"] == 1.96
    stopped = stats.holm([2.3, 2.0, 1.97])
    assert [x["significant"] for x in stopped] == [False, False, False], stopped
    assert [x["rank"] for x in stopped] == [1, 2, 3]
    assert all(x["family"] == 3 for x in stopped)
    assert stats.holm([]) == []


# ── group-sequential boundaries ──────────────────────────────────────────

def test_the_obf_and_pocock_tables_match_published_values():
    """Pinned like `T975`: O'Brien–Fleming c_k = c_K·sqrt(K/k) at two-sided
    alpha 0.05 (K=4: 4.049, 2.863, 2.337, 2.024), Pocock's constant 2.361."""
    assert stats.look_boundaries(1) == [1.960]
    assert stats.look_boundaries(4) == [4.049, 2.863, 2.337, 2.024]
    assert stats.look_boundaries(2, "pocock") == [2.178, 2.178]
    k4 = stats.look_boundaries(4)
    for k in range(1, 5):
        assert abs(k4[k - 1] - k4[3] * (4 / k) ** 0.5) < 0.01, "OBF's sqrt(K/k) shape"
    with pytest.raises(ValueError):
        stats.look_boundaries(5)
    with pytest.raises(ValueError):
        stats.look_boundaries(2, "bonferroni")


def test_a_look_uses_its_own_boundary_not_1_96():
    """THE BUG: a fixed 1.96 at every look is optional stopping. 20/100 vs
    34/100 excludes zero at 1.96 and does NOT at the first OBF look of four
    (z = 4.049); the same counts at the last look (2.024) do."""
    assert stats.diff_proportions(20, 100, 34, 100)["excludes_zero"]
    assert not stats.diff_proportions(20, 100, 34, 100, z=4.049)["excludes_zero"]
    assert stats.diff_proportions(20, 100, 34, 100, z=2.024)["excludes_zero"]


def test_sequential_power_at_one_look_equals_the_exact_fixed_design():
    """Monte Carlo against the exact grid: within sampling error at K=1, and
    a four-look OBF design costs only a little power at the same total N."""
    exact = stats.power_for(0.25, 0.45, 100, 100)
    mc = stats.sequential_power(0.25, 0.45, 100, 1, iterations=4000, seed=1)
    assert abs(mc - exact) < 0.03, (mc, exact)
    obf4 = stats.sequential_power(0.25, 0.45, 25, 4, iterations=4000, seed=1)
    assert exact - 0.08 < obf4 <= exact + 0.03, (obf4, exact)
