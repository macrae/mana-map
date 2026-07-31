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
- **Ship gates** are `xfail(strict=True)` — they encode where this is going, not
  where it is. They fail today, which is correct and documented. When the retrain
  lands they XPASS, and `strict=True` turns that into a suite failure, so nobody
  can fix the model and leave the goal marked as unmet.
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
    "layout (color+type)": {"recall@10": 0.044, "effective_dim": 3.05},
    "function (ability)": {"recall@10": 0.093, "effective_dim": 5.97},
    "text baseline (frozen MiniLM)": {"recall@10": 0.187, "effective_dim": 81.04},
}
FLOOR_TOLERANCE = 0.8

# A space using fewer than this many of its 128 dimensions has collapsed. Today's
# trained spaces are at 3.05 and 5.97, so this is the number the retrain must move.
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
def test_all_three_spaces_are_measured(metrics):
    assert set(metrics) == set(MEASURED), "an embedding artifact went missing"
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
@pytest.mark.xfail(strict=True, reason="the defect this work exists to fix: the trained "
                                       "function space scores 0.093 against the frozen text "
                                       "it is built from at 0.187")
def test_function_space_beats_the_frozen_text_it_is_built_from(metrics):
    """The ship gate for the retrain.

    A trained embedding that loses to its own frozen input is not adding structure,
    it is destroying it. If the retrain cannot clear this bar, the honest outcome is
    to ship the text embedding as the function space and say so.
    """
    function = metrics["function (ability)"]["recall"]["test"]["recall@10"]
    baseline = metrics["text baseline (frozen MiniLM)"]["recall"]["test"]["recall@10"]
    assert function > baseline


@requires_data
@pytest.mark.xfail(strict=True, reason="the function space uses 5.97 of its 128 dimensions")
def test_function_space_is_not_collapsed(metrics):
    """Collapse is the mechanism behind the recall gap, and it fails earlier.

    A model whose triplet loss hits zero at epoch 3 has learned to separate its
    labels and nothing else; every card within a label lands on the same point, and
    the ranking among them is numerical noise. That shows up here long before it
    shows up in recall.
    """
    assert metrics["function (ability)"]["effective_dim"] > MIN_EFFECTIVE_DIM


@requires_data
@pytest.mark.xfail(strict=True, reason="the top-50 neighbours are a numerical tie "
                                       "(gap 0.0236)")
def test_function_space_ranks_its_neighbours(metrics):
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
