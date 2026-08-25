"""Spike S1's instrument, and the two ways it could quietly lie.

The result this eval produced is decisive enough to change the build order, so
the instrument itself needs gating. Two failure modes matter more than the rest:
a leak that makes any embedding look good, and a type control that claims an
improvement it did not make.
"""

import random

import numpy as np
import pytest

from conftest import requires_data
from manamap.analysis import eval_commander_search as ecs

# `requires_data` is a skipif, not a marker — the whole module needs the
# embeddings, and the fixtures below add the two gates it cannot cover: the
# corpus CSV and the frozen EDHREC pool.
pytestmark = [requires_data]


@pytest.fixture(scope="module")
def corpus():
    try:
        return ecs.load_corpus()
    except FileNotFoundError:
        pytest.skip("cards.csv not built — `manamap extract`")


@pytest.fixture(scope="module")
def pool():
    try:
        return ecs.load_pool()
    except SystemExit:
        pytest.skip("no frozen commander pool — `manamap eval-commander-search --refresh`")


def test_the_pool_is_frozen_and_says_why(pool):
    """Ground truth that moves cannot tell a model change from a metagame change."""
    import json
    doc = json.loads(ecs.POOL_PATH.read_text(encoding="utf-8"))
    assert doc["commanders"], "an empty pool would make every metric vacuous"
    blob = " ".join(doc["_comment"]).lower()
    assert "frozen" in blob or "fetched once" in blob


def test_basic_lands_are_excluded_but_utility_lands_are_not(corpus):
    """§6.1 step 2. Basics carry no signal and would drag every centroid toward
    one point; specialty lands carry real signal and must survive."""
    _, by_name, _ = corpus
    rows = ecs._rows(["Plains", "Island", "Command Tower", "Sol Ring"], by_name)
    kept = {n for n in ("Command Tower", "Sol Ring") if by_name.get(n) in rows}
    assert kept == {"Command Tower", "Sol Ring"}
    assert by_name.get("Plains") not in rows
    assert by_name.get("Island") not in rows


def test_an_artifact_creature_counts_as_a_creature():
    """Composition control needs ONE type per card, and the order is a judgement:
    a body is a body first, and a land that also does something is a land."""
    assert ecs._primary_type("Artifact Creature — Golem") == "Creature"
    assert ecs._primary_type("Legendary Land") == "Land"
    assert ecs._primary_type("Instant") == "Instant"


# ── The leak, which is the failure that would invalidate the whole result ──


def test_the_seed_is_held_out_of_its_own_reference(corpus, pool):
    """WITHOUT the hold-out, the true commander contains its own query.

    This is the bug that would report near-perfect accuracy for an embedding
    that had learned nothing, so it is proven by construction: run the eval on
    RANDOM vectors. With the hold-out, random must score at chance. If it scores
    well, the answer is leaking through the deck membership rather than through
    the embedding.
    """
    names, by_name, types = corpus
    rng = np.random.default_rng(0)
    rand = rng.normal(size=(len(names), 64)).astype("float32")
    rand /= np.maximum(np.linalg.norm(rand, axis=1, keepdims=True), 1e-8)

    m = ecs.evaluate(rand, pool, by_name, types, controlled=False,
                     rng=random.Random(1))
    assert m["queries"] > 20, "too few queries for the assertion to mean anything"
    # Chance is 1/candidates. Allow generous headroom for a small sample — the
    # bug this catches produces top1 near 1.0, not near 0.1.
    assert m["top1"] < 0.15, (
        f"random vectors scored top1={m['top1']:.3f} — the seed is leaking into "
        f"its own reference centroid")
    assert m["mrr"] < 0.25, f"random vectors scored MRR={m['mrr']:.3f}"


def test_a_real_embedding_beats_random_by_a_wide_margin(corpus, pool):
    """The other half of the pair: the eval must be able to SEE signal.

    A leak-free eval that also cannot detect a good embedding is just a broken
    eval with a clean conscience.
    """
    from manamap.config import TEXT_EMBEDDINGS_PATH
    try:
        emb = ecs._normalized(TEXT_EMBEDDINGS_PATH)
    except FileNotFoundError:
        pytest.skip("text embeddings not built — `manamap preprocess`")
    names, by_name, types = corpus
    m = ecs.evaluate(emb, pool, by_name, types, controlled=True, rng=random.Random(1))
    assert m["top1"] > 0.3, f"top1={m['top1']:.3f} — the eval cannot see signal"
    assert m["top1"] > 10 * m["random_top1"]


# ── Type control: an improvement, or a claim? ──────────────────────────────


def test_type_control_is_measured_rather_than_assumed(corpus, pool):
    """§6.1 step 6 argues the reference centroid must be restricted to the seed's
    type composition, or the ranking measures deck COMPOSITION rather than deck
    IDENTITY. That is a claim, and this is the measurement.

    MEASURED OVER REPEATS, and that is the whole point of this test existing in
    this form. On a SINGLE draw type control looked like +12.7 points for the
    text space and +7.6 for the function space; over ten draws it is +3.7 and
    +0.0 respectively, with overlapping ranges. The first version of this test
    asserted the single-draw gain and failed on a different seed within the
    hour, which is the correct outcome for a test pinned to noise.

    So what is asserted is the durable half: control does not HURT. The gain
    itself is not established at this pool size and the module prints the range
    so a reader can see that for themselves.
    """
    from manamap.config import TEXT_EMBEDDINGS_PATH
    try:
        emb = ecs._normalized(TEXT_EMBEDDINGS_PATH)
    except FileNotFoundError:
        pytest.skip("text embeddings not built")
    names, by_name, types = corpus
    off = ecs.repeated(emb, pool, by_name, types, False, repeats=4)
    on = ecs.repeated(emb, pool, by_name, types, True, repeats=4)
    assert on["top1"] >= off["top1"] - 0.03, (
        f"type control made ranking WORSE across repeats: "
        f"{off['top1']:.3f} -> {on['top1']:.3f}. §6.1 step 6's claim needs "
        f"re-arguing, not re-fitting.")


def test_type_control_never_returns_an_empty_reference(corpus):
    """A deck with none of the seed's types must still be rankable — badly, but
    rankable. Returning nothing would drop candidates from the pool silently and
    make the denominator disagree with the header."""
    _, _, types = corpus
    rows = [i for i, t in enumerate(types) if t == "Creature"][:20]
    picked = ecs.type_controlled_rows(rows, types, {"Instant": 1.0}, random.Random(0))
    assert picked, "an all-creature deck against an all-instant seed returned nothing"


def test_the_text_baseline_still_beats_the_trained_space(corpus, pool):
    """The finding that reorders the build, asserted so a retrain has to move it.

    Over ten draws: text top-1 0.584 [0.52-0.67] against the trained function
    space's 0.410 [0.30-0.47] — ranges that do NOT overlap, so the gap survives
    resampling. It confirms at the DECISION level what `eval-embeddings` showed
    at the card level (0.232 vs 0.244 recall@10), and much more loudly: 17
    points of top-1 rather than one point of recall.

    This test is expected to FAIL the day Track A2 succeeds, and that failure is
    the deliverable. Flip it then; do not weaken it now.
    """
    from manamap.config import ABILITY_EMBEDDINGS_PATH, TEXT_EMBEDDINGS_PATH
    try:
        fn = ecs._normalized(ABILITY_EMBEDDINGS_PATH)
        tx = ecs._normalized(TEXT_EMBEDDINGS_PATH)
    except FileNotFoundError:
        pytest.skip("embeddings not built")
    names, by_name, types = corpus
    f = ecs.repeated(fn, pool, by_name, types, True, repeats=4)
    t = ecs.repeated(tx, pool, by_name, types, True, repeats=4)
    assert t["top1"] > f["top1"], (
        f"the trained space now BEATS the text baseline ({f['top1']:.3f} vs "
        f"{t['top1']:.3f}) — Track A2 has landed. Update this test and the "
        f"finding it guards rather than deleting it.")
