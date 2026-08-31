"""Does the embedding actually represent similarity?

The gap this fills: every other embedding test in this repo is structural.
`test_find_similar.py` asserts L2 norms, `.bin`/`.npy` fidelity, byte sizes and
2D-vs-128D divergence — all of which pass against a randomly initialized model.
So the suite was fully green while `Doubling Season`'s nearest neighbours were
arbitrary green enchantments, and it had been for a long time.

Two kinds of assertion here, on purpose:

- **Regression floors** pass today and fail if a change makes a space worse.
  They are set from measured values, a little below them, so ordinary run-to-run
  variation does not trip them.
- **Ship gates** encode where this is going, not where it is. While a gate is unmet
  it carries `xfail(strict=True)`: the suite stays green, the goal stays visible in
  the output, and when the work lands the test XPASSes — which `strict` turns into a
  failure, so the marker cannot be left on and the achievement cannot be silently
  pocketed. Two of the three gates have now been met and are ordinary assertions.
  The third (neighbour spread) is still `xfail` and its threshold was deliberately
  NOT lowered to match what the retrain achieved.
"""

import json

import numpy as np
import pytest

from conftest import requires_data

from manamap.analysis import eval_embeddings
from manamap.config import SIMILARITY_GOLDEN_PATH

# Measured on the shipped artifacts, test split (see docs/architecture.md).
# Floors sit below the measurement, not at it — this catches breakage, not noise.
MEASURED = {
    "layout (color+type)": {"recall@10": 0.090, "effective_dim": 3.20},
    "function (ability)": {"recall@10": 0.245, "effective_dim": 27.87},
    "text baseline (frozen MiniLM)": {"recall@10": 0.244, "effective_dim": 50.41},
}
FLOOR_TOLERANCE = 0.8

# A space using fewer than this many of its 128 dimensions has collapsed. Set while
# the trained spaces were at 3.05 and 5.97; the retrain moved the function space to
# 27.87, so this now passes. Left where it was rather than raised to hug the result —
# a floor that tracks the current number stops being a floor.
MIN_EFFECTIVE_DIM = 25.0


@pytest.fixture(scope="module")
def golden():
    with open(SIMILARITY_GOLDEN_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def metrics():
    results, _ = eval_embeddings.collect()
    return results


# ── the golden set itself ───────────────────────────────────────────────


def test_golden_set_has_both_splits(golden):
    """A held-out split is the whole point.

    The dev groups were used while diagnosing the collapse, so any decision taken
    in response to those numbers is fitted to them. Conclusions get drawn from the
    test split; if it ever empties out, the headline number becomes self-graded.
    """
    splits = [g["split"] for g in golden["groups"]]
    assert splits.count("test") >= 20, "the held-out split has been eroded"
    assert splits.count("dev") >= 5


def test_golden_groups_are_usable(golden):
    for group in golden["groups"]:
        assert len(group["cards"]) >= 2, f"{group['id']} cannot measure recall alone"
        assert len(set(group["cards"])) == len(group["cards"]), f"{group['id']} repeats a card"


def test_golden_set_records_its_independence(golden):
    """Guards intent, cheaply.

    The set may not be derived from mechanical tags, roles, the synergy graph or
    combo data, because the training objective mines positives from exactly those —
    an eval built from them would measure whether training memorised its own
    supervision. That constraint lives in a comment, so this asserts the comment.
    """
    comment = " ".join(golden["_comment"]).lower()
    assert "hand-authored" in comment
    assert "independen" in comment


@requires_data
def test_every_golden_card_still_exists():
    """The set is tracked and hand-edited; cards.csv is regenerated.

    A renamed card must surface here rather than silently shrinking the evaluation.
    """
    import pandas as pd

    from manamap.config import OUTPUT_CSV_PATH

    names = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)["name"].tolist()
    _, missing = eval_embeddings.resolve_groups(eval_embeddings.load_golden(), names)
    assert missing == [], f"golden cards no longer in cards.csv: {missing}"


# ── the metrics compute at all ──────────────────────────────────────────


@requires_data
def test_every_pinned_space_is_measured(metrics):
    """Every space in `MEASURED` must still be measurable — extras are allowed.

    This was `== set(MEASURED)`, which is the right assertion when the set of
    artifacts is closed and the wrong one the moment a SHADOW artifact exists.
    The plan builds each new space beside the old ones and cuts over only if the
    eval says so, so `embeddings_function_vae.npy` appearing is the design
    working, not an artifact going missing.

    Containment keeps the guarantee that matters — a pinned space that vanishes
    or gets renamed still fails — and drops only the ability to notice a new one,
    which is now an expected event rather than a surprising one.
    """
    absent = set(MEASURED) - set(metrics)
    assert not absent, f"an embedding artifact went missing: {sorted(absent)}"
    for name, m in metrics.items():
        assert m["recall"]["test"]["queries"] > 50, f"{name}: too few queries to trust"


@requires_data
def test_effective_dimensionality_detects_collapse(metrics):
    """The metric has to be able to tell the two apart, or it is not a metric.

    Frozen MiniLM uses ~81 of its 384 dimensions; the trained spaces use 3 and 6 of
    128. If this assertion ever gets close, the participation ratio has stopped
    discriminating and the collapse gate below is meaningless.
    """
    text = metrics["text baseline (frozen MiniLM)"]["effective_dim"]
    layout = metrics["layout (color+type)"]["effective_dim"]
    assert text > 10 * layout


@requires_data
def test_neighbour_spread_is_reported(metrics):
    """A near-zero spread means the top-50 ordering is float noise, not ranking."""
    assert metrics["layout (color+type)"]["neighbour_spread"] < 0.01
    assert metrics["text baseline (frozen MiniLM)"]["neighbour_spread"] > 0.05


# ── regression floors (pass today) ──────────────────────────────────────


@requires_data
@pytest.mark.parametrize("space", sorted(MEASURED))
def test_no_space_regresses(metrics, space):
    got = metrics[space]["recall"]["test"]["recall@10"]
    floor = MEASURED[space]["recall@10"] * FLOOR_TOLERANCE
    assert got >= floor, f"{space} recall@10 fell to {got:.3f}, floor {floor:.3f}"


@requires_data
def test_text_baseline_stays_the_bar_to_beat(metrics):
    """Pin the baseline itself.

    Phase 1 changes the embedding text, which moves this number — deliberately. It
    should move UP; a drop means the text change made the input worse and the whole
    comparison has shifted under the retrain.
    """
    text = metrics["text baseline (frozen MiniLM)"]["recall"]["test"]["recall@10"]
    assert text >= MEASURED["text baseline (frozen MiniLM)"]["recall@10"] * FLOOR_TOLERANCE


# ── ship gates (xfail today, by design) ─────────────────────────────────


@requires_data
def test_function_space_beats_the_frozen_text_it_is_built_from(metrics):
    """The ship gate — asserted on MEDIAN RANK, not recall@10, and that is the point.

    A trained embedding that loses to its own frozen input is destroying structure
    rather than adding it. The retrain fixed that, against the model it replaced:
    recall@10 0.093 -> 0.245, median rank 995 -> 78.

    Against the *frozen text* baseline, though, recall@10 is a **tie** — 0.245 versus
    0.244. That +0.001 across ~160 queries is noise, and a gate a coin flip can pass
    is not a gate, so it is deliberately not what this asserts. The improvement that
    is real and large is median rank (124 -> 78, a 37% cut), which also uses every
    query rather than thresholding at a top-10 cutoff.

    Recall@10 still gets a floor, one-sided: the function space is allowed to tie the
    baseline, never to fall behind the input it is built from.
    """
    function = metrics["function (ability)"]["recall"]["test"]
    baseline = metrics["text baseline (frozen MiniLM)"]["recall"]["test"]
    assert function["median_rank"] < baseline["median_rank"] * 0.8, (
        f"median rank {function['median_rank']:.0f} vs baseline "
        f"{baseline['median_rank']:.0f} — the depth win is gone"
    )
    assert function["recall@10"] >= baseline["recall@10"] * 0.95, (
        "the function space has fallen behind the frozen text it is built from"
    )


@requires_data
def test_function_space_is_not_collapsed(metrics):
    """Collapse was the mechanism behind the recall gap, and it fails earlier than
    recall does. FIXED: 5.97 -> 27.87 effective dimensions.

    The old model's triplet loss hit zero within a few epochs, having learned to
    separate its labels and nothing else — every card within a label landed on the
    same point, so the ranking among them was numerical noise. In-batch InfoNCE keeps
    producing gradient after the easy pairs are solved, which is what reopened the
    space.
    """
    assert metrics["function (ability)"]["effective_dim"] > MIN_EFFECTIVE_DIM


@requires_data
@pytest.mark.xfail(strict=True, reason="improved 0.0236 -> 0.0315 by the retrain, but "
                                       "still short of 0.05 — the top-50 are tighter "
                                       "than they should be")
def test_function_space_ranks_its_neighbours(metrics):
    """The one gate the retrain did NOT clear.

    Left failing rather than moved to meet what was achieved. A threshold edited down
    to match the result is not a threshold, and this one is still saying something
    true: the top-50 neighbours are closer together than a well-spread space would put
    them, so their ordering carries less information than it should.
    """
    assert metrics["function (ability)"]["neighbour_spread"] > 0.05


# ── the probes a human would check ──────────────────────────────────────


@requires_data
def test_the_probe_that_started_this():
    """One named case, because a table of averages hides the felt problem — and it
    localises the defect more precisely than the aggregates do.

    `Doubling Season` should sit beside the other token doublers. Measured today:

        layout   → Gift of the Woods, Super Strength, Naturalize the Phyresis, …
        function → Concordant Crossroads, Far Out, …, Parallel Lives (5th), Primal Vigor

    The garbage is the *layout* space, which is what the default map searches for
    Find Similar. The function space is imperfect but already finds real doublers.
    That asymmetry is the argument for decoupling similarity from the displayed map:
    the data to answer this question already exists, and the viz asks the wrong
    artifact for it.

    Asserted as a comparison rather than as "layout is broken", so it stays true and
    keeps its meaning after the retrain.
    """
    import pandas as pd

    from manamap.analysis.common import top_k_similar
    from manamap.config import ABILITY_EMBEDDINGS_PATH, EMBEDDINGS_PATH, OUTPUT_CSV_PATH

    doublers = {"Parallel Lives", "Anointed Procession", "Primal Vigor",
                "Mondrak, Glory Dominus"}
    names = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)["name"].tolist()
    row = names.index("Doubling Season")

    def hits(path):
        embeddings = np.load(path)
        found = {names[i] for i, _ in top_k_similar(embeddings, row, k=10)}
        return len(found & doublers)

    assert hits(ABILITY_EMBEDDINGS_PATH) > hits(EMBEDDINGS_PATH), (
        "the function space no longer beats the layout space on the probe that "
        "motivated decoupling"
    )
