"""The candidate pool as an eval axis, and the interval on the difference.

Step 15 ranked every golden card against all 34,890 cards and reported the bare
difference between two spaces. Nothing in the product ranks that way — commander
search ranks against 79 candidates, Find Similar shows 12, `build-deck` ranks
within a colour identity and a pool — and a bare difference is not a finding.

MEASURED 2026-08-31, 28 test groups, paired bootstrap over groups:

    distractors   function    text      gap   95% CI on the difference
            100      0.964   0.819   +0.145   [+0.053, +0.235]  excludes 0
            500      0.794   0.629   +0.165   [+0.052, +0.289]  excludes 0
           2000      0.562   0.446   +0.115   [-0.018, +0.255]
          10000      0.363   0.311   +0.052   [-0.045, +0.162]
         34,890      0.227   0.240   -0.013   [-0.083, +0.058]  SPANS ZERO

So the `-0.012` that named issue #12 is a TIE, and at product pool sizes the
trained space wins at an interval excluding zero.

These tests gate the INSTRUMENT, not the reading. The numbers above will move
with any retrain and are not asserted here; what is asserted is that the
instrument cannot go back to answering the question the wrong way.
"""

import numpy as np
import pytest

from manamap.analysis import eval_embeddings as E
from conftest import requires_data


# ── the bootstrap ──


def test_identical_inputs_give_a_zero_gap_that_spans_zero():
    a = np.array([0.1, 0.5, 0.9, 0.3])
    stat = E.paired_bootstrap(a, a.copy())
    assert stat["gap"] == 0.0
    assert stat["lo"] <= 0.0 <= stat["hi"]
    assert not stat["excludes_zero"]


def test_a_large_consistent_difference_excludes_zero():
    a = np.array([0.9, 0.8, 0.95, 0.85, 0.9, 0.88])
    b = a - 0.4
    stat = E.paired_bootstrap(a, b)
    assert stat["gap"] == pytest.approx(0.4, abs=1e-9)
    assert stat["excludes_zero"] and stat["lo"] > 0


def test_a_small_noisy_difference_does_not_exclude_zero():
    """THE CASE THE EVAL EXISTED WITHOUT. `-0.012 recall@10` was printed as
    "Training is destroying information" for months. A difference that flips
    sign across the sample must not read as a finding."""
    rng = np.random.default_rng(0)
    a = rng.random(28)
    b = a + rng.normal(0, 0.3, 28)          # same centre, plenty of scatter
    assert not E.paired_bootstrap(a, b)["excludes_zero"]


def test_the_bootstrap_is_paired():
    """Both spaces are scored on the IDENTICAL groups, so the variance that
    matters is between groups, not between two independent samples. Shuffling
    one side destroys the pairing and must widen the interval."""
    rng = np.random.default_rng(7)
    a = rng.random(40)
    b = a - 0.15                            # a constant, perfectly paired offset
    paired = E.paired_bootstrap(a, b)
    shuffled = E.paired_bootstrap(a, rng.permutation(b))
    assert paired["hi"] - paired["lo"] < shuffled["hi"] - shuffled["lo"]
    # …and the perfectly-paired case has essentially no spread at all.
    assert paired["hi"] - paired["lo"] < 1e-9


def test_an_empty_or_ragged_sample_reports_nothing_rather_than_guessing():
    assert E.paired_bootstrap([], [])["n"] == 0
    assert not E.paired_bootstrap([0.1], [0.2, 0.3])["excludes_zero"]


# ── the pool ──


def _groups(n_groups=6, per=4, offset=0):
    return [{"id": f"g{i}", "split": "test",
             "rows": list(range(offset + i * per, offset + (i + 1) * per))}
            for i in range(n_groups)]


def _planted(n=400, per=4, n_groups=6, dim=8, seed=3):
    """A space where each group's rows genuinely cluster, so recall is nonzero."""
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, dim))
    for g in range(n_groups):
        centre = rng.normal(size=dim) * 5
        emb[g * per:(g + 1) * per] = centre + rng.normal(size=(per, dim)) * 0.01
    return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)


def test_recall_is_reported_per_group_not_per_query():
    """Queries inside a group ask about the same targets, so they are not
    independent. Pooling them would inflate the sample fourfold here and shrink
    every interval built on it."""
    emb = _planted()
    out = E.recall_by_group(emb, _groups(), np.arange(len(emb)), pool=None)
    assert len(out) == 6, "one number per GROUP"
    assert out.max() > 0, "the planted clusters should be findable"


def test_every_group_survives_every_pool_size():
    """THE DESIGN THAT MAKES THE COMPARISON HONEST, and the bug it replaced.

    Candidates are each group's own targets PLUS N distractors. The obvious
    alternative — restrict to the top-N most played and keep only groups that
    fit inside it — changes WHICH GROUPS QUALIFY as the pool narrows, so a
    selection effect reads as a pool effect. Tried first on the real data: it
    gave a clean monotonic +0.200 at pool 500, and holding the groups genuinely
    constant collapsed the sample to 5 test / 2 dev groups whose splits then
    disagreed in SIGN.
    """
    emb = _planted()
    order = np.arange(len(emb))
    groups = _groups(offset=300)            # rows far outside a small pool
    counts = {pool: len(E.recall_by_group(emb, groups, order, pool=pool))
              for pool in (10, 50, 400, None)}
    assert len(set(counts.values())) == 1, f"group count moved with the pool: {counts}"
    assert set(counts.values()) == {6}


def test_a_narrower_pool_cannot_lower_recall_for_the_same_space():
    """Removing distractors can only remove competitors for a top-k slot. If a
    narrower pool ever scored WORSE, the pool is not being applied as
    distractors and the axis means something else."""
    emb = _planted()
    order = np.arange(len(emb))
    groups = _groups()
    wide = E.recall_by_group(emb, groups, order, pool=None).mean()
    narrow = E.recall_by_group(emb, groups, order, pool=20).mean()
    assert narrow >= wide - 1e-12


def test_the_pool_axis_covers_more_than_the_whole_corpus():
    """A single-entry axis would be the old behaviour wearing a new name."""
    from manamap.config import EVAL_POOL_SIZES

    assert len(EVAL_POOL_SIZES) >= 3
    assert None in EVAL_POOL_SIZES, "the corpus-wide figure stays, for continuity"
    finite = [p for p in EVAL_POOL_SIZES if p is not None]
    assert min(finite) <= 500, "must reach the pool sizes the product actually uses"


def test_unranked_cards_sort_last():
    """`edhrec_rank` is missing for ~8% of the corpus, and those are by
    construction the cards nobody plays."""
    import pandas as pd

    frame = pd.DataFrame({"edhrec_rank": [500.0, np.nan, 1.0, np.nan, 50.0]})
    order = E.playability_order(frame)
    assert list(order[:3]) == [2, 4, 0]
    assert set(order[3:]) == {1, 3}


# ── the real corpus ──


@requires_data
def test_the_report_carries_an_interval_on_every_gap():
    """The instrument, not the reading: whatever the numbers say, each row must
    carry an interval on the DIFFERENCE. That is the rule the eval broke."""
    text, curve = E.pool_section()
    if not text:
        pytest.skip("embedding artifacts not built")
    assert "95% CI on the difference" in text
    assert text.count("[") >= len(curve), "every pool row needs its interval"
    for pool, spaces in curve.items():
        stat = E.paired_bootstrap(spaces["function (ability)"],
                                  spaces["text baseline (frozen MiniLM)"])
        assert stat["n"] >= 20, f"pool {pool} scored only {stat['n']} groups"
        assert stat["lo"] <= stat["gap"] <= stat["hi"]


@requires_data
def test_the_verdict_never_calls_an_overlapping_interval_a_finding():
    """Re-introducing the bug: the old code printed "Training is destroying
    information" off a bare `-0.012`. Any wording that declares a winner while
    the interval spans zero is the same defect."""
    import contextlib
    import io as _io

    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        E.main()
    out = buf.getvalue()
    if "CANDIDATE POOL" not in out:
        pytest.skip("embedding artifacts not built")
    _text, curve = E.pool_section()
    stat = E.paired_bootstrap(curve[None]["function (ability)"],
                              curve[None]["text baseline (frozen MiniLM)"])
    if not stat["excludes_zero"]:
        assert "INDISTINGUISHABLE" in out
        assert "destroying information" not in out
