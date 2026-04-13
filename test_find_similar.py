"""
Tests for Find Similar Cards — verifies that 128D cosine similarity
produces meaningfully different results from 2D Euclidean distance,
and that the JS implementation matches the Python reference.
"""

import json
import struct
import numpy as np
import pytest
from pathlib import Path

DATA_DIR = Path("data")
EMBED_DIM = 128


@pytest.fixture(scope="module")
def embeddings_npy():
    """Load the numpy embeddings (ground truth)."""
    return np.load(DATA_DIR / "embeddings.npy")


@pytest.fixture(scope="module")
def embeddings_bin():
    """Load embeddings.bin as Float32Array (same format JS uses)."""
    raw = (DATA_DIR / "embeddings.bin").read_bytes()
    arr = np.frombuffer(raw, dtype=np.float32)
    n_cards = len(arr) // EMBED_DIM
    return arr.reshape(n_cards, EMBED_DIM)


@pytest.fixture(scope="module")
def projection():
    """Load 2D projection data."""
    with open(DATA_DIR / "projection_2d.json") as f:
        return json.load(f)


# ── Data integrity tests ──


def test_embeddings_bin_matches_npy(embeddings_npy, embeddings_bin):
    """embeddings.bin must be a faithful Float32 export of embeddings.npy."""
    assert embeddings_bin.shape == embeddings_npy.shape, (
        f"Shape mismatch: bin={embeddings_bin.shape}, npy={embeddings_npy.shape}"
    )
    np.testing.assert_allclose(embeddings_bin, embeddings_npy.astype(np.float32), atol=1e-6)


def test_projection_count_matches_embeddings(embeddings_npy, projection):
    """projection_2d.json must have the same number of cards as embeddings."""
    assert len(projection) == embeddings_npy.shape[0], (
        f"Projection has {len(projection)} cards, embeddings has {embeddings_npy.shape[0]}"
    )


def test_embeddings_are_l2_normalized(embeddings_npy):
    """Model output embeddings should be L2-normalized (norm ≈ 1)."""
    norms = np.linalg.norm(embeddings_npy, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4,
        err_msg="Embeddings should be L2-normalized")


# ── Cosine similarity correctness ──


def cosine_similarity(a, b):
    """Reference cosine similarity matching the JS implementation."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def test_cosine_similarity_identical_vectors():
    """Identical vectors should have cosine similarity of 1."""
    v = np.random.randn(EMBED_DIM).astype(np.float32)
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    """Orthogonal vectors should have cosine similarity of 0."""
    a = np.zeros(EMBED_DIM, dtype=np.float32)
    b = np.zeros(EMBED_DIM, dtype=np.float32)
    a[0] = 1.0
    b[1] = 1.0
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_opposite_vectors():
    """Opposite vectors should have cosine similarity of -1."""
    v = np.random.randn(EMBED_DIM).astype(np.float32)
    assert abs(cosine_similarity(v, -v) + 1.0) < 1e-6


def test_cosine_similarity_on_real_embeddings(embeddings_bin):
    """Cosine similarity on actual card embeddings should be in [-1, 1]."""
    for _ in range(100):
        i, j = np.random.choice(len(embeddings_bin), 2, replace=False)
        sim = cosine_similarity(embeddings_bin[i], embeddings_bin[j])
        assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6, f"sim={sim} out of range"


# ── 128D cosine vs 2D Euclidean produce different rankings ──


def find_similar_2d(ref_idx, projection, top_k=20):
    """Find similar by 2D Euclidean distance (old method)."""
    ref = projection[ref_idx]
    rx, ry = ref["x"], ref["y"]
    dists = []
    for i, d in enumerate(projection):
        if i == ref_idx:
            continue
        dist = np.sqrt((d["x"] - rx) ** 2 + (d["y"] - ry) ** 2)
        dists.append((i, dist))
    dists.sort(key=lambda x: x[1])
    return [idx for idx, _ in dists[:top_k]]


def find_similar_128d(ref_idx, embeddings, top_k=20):
    """Find similar by 128D cosine similarity (new method)."""
    ref = embeddings[ref_idx]
    sims = []
    for i in range(len(embeddings)):
        if i == ref_idx:
            continue
        sim = cosine_similarity(ref, embeddings[i])
        sims.append((i, sim))
    sims.sort(key=lambda x: -x[1])
    return [idx for idx, _ in sims[:top_k]]


def test_2d_vs_128d_differ_for_sample_cards(embeddings_bin, projection):
    """128D cosine similarity top-20 must differ from 2D Euclidean top-20
    for at least some cards — proving the embedding method adds value."""
    np.random.seed(42)
    test_indices = np.random.choice(len(embeddings_bin), 10, replace=False)

    differ_count = 0
    for ref_idx in test_indices:
        top20_2d = set(find_similar_2d(ref_idx, projection))
        top20_128d = set(find_similar_128d(ref_idx, embeddings_bin))
        overlap = top20_2d & top20_128d
        if len(overlap) < 20:
            differ_count += 1

    assert differ_count > 0, (
        "128D and 2D methods produced identical top-20 for all 10 test cards — "
        "the embedding similarity should differ from the 2D projection"
    )


def test_128d_vs_2d_overlap_statistics(embeddings_bin, projection):
    """Measure how much 2D and 128D top-20 overlap across many cards.
    We expect partial overlap (PaCMAP preserves SOME structure) but not 100%."""
    np.random.seed(123)
    test_indices = np.random.choice(len(embeddings_bin), 50, replace=False)

    overlaps = []
    for ref_idx in test_indices:
        top20_2d = set(find_similar_2d(ref_idx, projection))
        top20_128d = set(find_similar_128d(ref_idx, embeddings_bin))
        overlap = len(top20_2d & top20_128d)
        overlaps.append(overlap)

    mean_overlap = np.mean(overlaps)
    # PaCMAP is good at local preservation, so some overlap is expected.
    # But it shouldn't be 100% for all cards.
    assert mean_overlap < 20, (
        f"Mean overlap is {mean_overlap}/20 — methods are essentially identical"
    )
    # Print stats for visibility
    print(f"\n  2D vs 128D overlap stats (n=50): "
          f"mean={mean_overlap:.1f}/20, min={min(overlaps)}, max={max(overlaps)}")


def test_128d_selects_different_cards_than_2d(embeddings_bin, projection):
    """The 128D method should select meaningfully different cards than 2D —
    on average at least 5 of the top-20 should differ.
    (Cards may still be spatially close in 2D since PaCMAP preserves
    local structure, but they should be DIFFERENT cards.)"""
    np.random.seed(77)
    test_indices = np.random.choice(len(embeddings_bin), 30, replace=False)

    unique_to_128d = []
    for ref_idx in test_indices:
        top20_2d = set(find_similar_2d(ref_idx, projection))
        top20_128d = set(find_similar_128d(ref_idx, embeddings_bin))
        unique_to_128d.append(len(top20_128d - top20_2d))

    mean_unique = np.mean(unique_to_128d)
    assert mean_unique >= 5, (
        f"128D method only finds {mean_unique:.1f} unique cards per query vs 2D — "
        "expected at least 5 different cards on average"
    )


# ── JS/binary format consistency ──


def test_embeddings_bin_is_raw_float32(embeddings_npy):
    """embeddings.bin should be raw Float32 with no header (JS reads it directly)."""
    raw = (DATA_DIR / "embeddings.bin").read_bytes()
    expected_size = embeddings_npy.shape[0] * embeddings_npy.shape[1] * 4
    assert len(raw) == expected_size, (
        f"embeddings.bin is {len(raw)} bytes, expected {expected_size} "
        f"({embeddings_npy.shape[0]} cards × {embeddings_npy.shape[1]} dims × 4 bytes)"
    )

    # Spot-check first card
    first_card = struct.unpack(f"<{EMBED_DIM}f", raw[:EMBED_DIM * 4])
    np.testing.assert_allclose(
        first_card, embeddings_npy[0].astype(np.float32), atol=1e-6,
        err_msg="First card in .bin doesn't match .npy"
    )


def test_js_index_alignment(embeddings_bin, projection):
    """Card indices in projection_2d.json must align with embeddings.bin rows.
    Verify by checking that the 'n' (name) field exists and index count matches."""
    assert len(projection) == len(embeddings_bin), (
        f"Index mismatch: projection has {len(projection)} entries, "
        f"embeddings has {len(embeddings_bin)} rows"
    )
    # Every projection entry should have 'n' (name) and 'x','y' fields
    for i in [0, 100, len(projection) - 1]:
        d = projection[i]
        assert "n" in d, f"projection[{i}] missing 'n' field"
        assert "x" in d, f"projection[{i}] missing 'x' field"
        assert "y" in d, f"projection[{i}] missing 'y' field"
