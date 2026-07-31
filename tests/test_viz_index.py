"""The discovery artifacts: `viz_index.json` and `neighbours.bin`.

These exist so the browser can land on a card and branch from it without fetching
the 12.9 MB projection or the 16.8 MB embedding matrix — 2.4 MB gzipped against
18.4 MB. That only holds if the tables say exactly what the embeddings say, so
most of this file is a round-trip: decode the binary, compare against a live
`top_k_similar` over `embeddings_ability.npy`.

The rest guards the two ways a precomputed artifact goes quietly wrong:

- **Staleness.** A table built from an older embedding still parses and still
  returns confident, wrong neighbours. The header carries a digest of the matrix
  it was derived from; `test_digest_matches_the_live_embeddings` is the alarm.
- **Re-sorting.** The stored similarities are quantised. Ordering lives in the
  array order and nothing may re-derive it — see the module docstring on
  `export/viz_index.py` for why (the space is a narrow cone; lossy values reorder
  the top-10 for most cards).
"""

import gzip
import json
import struct

import numpy as np
import pytest
from conftest import requires_data

from manamap.analysis.common import top_k_similar
from manamap.config import (
    ABILITY_EMBEDDINGS_PATH,
    NEIGHBOURS_BIN_PATH,
    NEIGHBOURS_FORMAT_VERSION,
    NEIGHBOURS_HEADER_BYTES,
    NEIGHBOURS_K_OBSOLETE,
    NEIGHBOURS_K_SIMILAR,
    NEIGHBOURS_K_SYNERGY,
    NEIGHBOURS_MAGIC,
    NEIGHBOURS_NONE,
    OUTPUT_CSV_PATH,
    VIZ_INDEX_PATH,
)
from manamap.export import viz_index as vi

requires_viz_index = pytest.mark.skipif(
    not NEIGHBOURS_BIN_PATH.exists(),
    reason="requires the discovery artifacts (run `manamap viz-index`)",
)

pytestmark = [requires_data, requires_viz_index]


def decode(blob):
    """Reference decoder — the shape the JS reader must mirror.

    Kept in the tests rather than in `export/viz_index.py` on purpose: the writer
    should not be able to satisfy the reader by construction. If this and the
    writer drift, that is exactly the bug worth failing on.
    """
    # unpack_from, not unpack: the fields occupy 60 bytes and the header is padded to
    # 64 so the uint16 blocks land aligned. `unpack` demands an exact-size buffer.
    magic, version, n, ks, ky, ko, _, digest, lo, hi = struct.unpack_from(
        "<4sIIHHHH32sff", blob, 0
    )
    assert magic == NEIGHBOURS_MAGIC
    off = NEIGHBOURS_HEADER_BYTES

    def u16(k):
        nonlocal off
        a = np.frombuffer(blob, dtype="<u2", count=n * k, offset=off).reshape(n, k)
        off += n * k * 2
        return a

    sim_idx, syn_idx, obs_idx = u16(ks), u16(ky), u16(ko)
    sim_val = np.frombuffer(blob, dtype=np.uint8, count=n * ks, offset=off).reshape(n, ks)
    off += n * ks
    counts = np.frombuffer(blob, dtype=np.uint8, count=n * 3, offset=off).reshape(n, 3)
    return {
        "version": version, "n": n, "digest": digest, "lo": lo, "hi": hi,
        "sim_idx": sim_idx, "sim_val": sim_val, "syn_idx": syn_idx,
        "obs_idx": obs_idx, "counts": counts,
    }


@pytest.fixture(scope="module")
def table():
    return decode(NEIGHBOURS_BIN_PATH.read_bytes())


@pytest.fixture(scope="module")
def index():
    with open(VIZ_INDEX_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# ── the binary decodes at all ───────────────────────────────────────────


def test_header_is_wellformed(table):
    assert table["version"] == NEIGHBOURS_FORMAT_VERSION
    assert table["lo"] < table["hi"], "similarity range must be non-degenerate"


def test_uint16_blocks_are_two_aligned():
    """A misaligned `Uint16Array` view throws in JS at page load, far from the cause.

    Header is 64 bytes and every uint16 block is contiguous after it, so this holds
    for any k. Interleaving the uint8 block between them would pass today only
    because K_SIMILAR is even.
    """
    assert NEIGHBOURS_HEADER_BYTES % 2 == 0
    for k in (NEIGHBOURS_K_SIMILAR, NEIGHBOURS_K_SYNERGY, NEIGHBOURS_K_OBSOLETE):
        assert (NEIGHBOURS_HEADER_BYTES + 34322 * k * 2) % 2 == 0


def test_row_ids_fit_the_sentinel(table):
    """34,322 < 65,535, which is what makes uint16 ids and 0xFFFF-as-empty safe."""
    assert table["n"] < NEIGHBOURS_NONE
    real = table["sim_idx"][table["sim_idx"] != NEIGHBOURS_NONE]
    assert real.max() < table["n"]


# ── it says what the embeddings say ─────────────────────────────────────


@requires_data
def test_similar_neighbours_match_the_live_embeddings(table):
    """The round-trip. Anything else here is bookkeeping; this is the artifact."""
    embeddings = np.load(ABILITY_EMBEDDINGS_PATH)
    rng = np.random.default_rng(0)
    for row in rng.choice(table["n"], 120, replace=False):
        expected = [i for i, _ in top_k_similar(embeddings, int(row), NEIGHBOURS_K_SIMILAR)]
        stored = [int(i) for i in table["sim_idx"][row] if i != NEIGHBOURS_NONE]
        assert stored == expected, f"row {row} disagrees with the live embeddings"


@requires_data
def test_stored_order_is_descending_by_true_similarity(table):
    """The pre-sorted guarantee, checked against real cosines rather than the
    quantised values — because the quantised values are exactly what a client must
    not sort by."""
    embeddings = np.load(ABILITY_EMBEDDINGS_PATH)
    rng = np.random.default_rng(1)
    for row in rng.choice(table["n"], 60, replace=False):
        rows = [int(i) for i in table["sim_idx"][row] if i != NEIGHBOURS_NONE]
        sims = [float(embeddings[row] @ embeddings[r]) for r in rows]
        assert sims == sorted(sims, reverse=True), f"row {row} is not pre-sorted"


@requires_data
def test_quantised_similarity_is_accurate_enough_for_edge_length(table):
    """uint8 over the header's range. Used for edge length only; the tolerance is
    what justifies halving the block instead of storing uint16."""
    embeddings = np.load(ABILITY_EMBEDDINGS_PATH)
    lo, hi = table["lo"], table["hi"]
    rng = np.random.default_rng(2)
    worst = 0.0
    for row in rng.choice(table["n"], 60, replace=False):
        for slot, other in enumerate(table["sim_idx"][row]):
            if other == NEIGHBOURS_NONE:
                continue
            decoded = table["sim_val"][row, slot] / 255.0 * (hi - lo) + lo
            worst = max(worst, abs(decoded - float(embeddings[row] @ embeddings[other])))
    assert worst < 0.01, f"quantisation error {worst:.5f} is too large for edge length"


@requires_data
def test_digest_matches_the_live_embeddings(table):
    """Staleness alarm.

    A neighbour table built from a previous model parses cleanly and answers
    confidently — the failure mode the DATA_VERSION comment in mana-map.js already
    memorialises. Retraining without regenerating this file must fail loudly here.
    """
    assert table["digest"] == vi.embeddings_digest(ABILITY_EMBEDDINGS_PATH), (
        "neighbours.bin was built from a different embedding matrix — "
        "re-run `manamap viz-index`"
    )


# ── counts describe reality, so the UI can promise honestly ─────────────


def test_counts_agree_with_the_slots(table):
    for name, key, k in (("similar", "sim_idx", NEIGHBOURS_K_SIMILAR),
                         ("synergy", "syn_idx", NEIGHBOURS_K_SYNERGY),
                         ("obsolete", "obs_idx", NEIGHBOURS_K_OBSOLETE)):
        col = {"similar": 0, "synergy": 1, "obsolete": 2}[name]
        filled = (table[key] != NEIGHBOURS_NONE).sum(axis=1)
        assert np.array_equal(filled, table["counts"][:, col]), f"{name} counts lie"


def test_relation_coverage_matches_the_source_graphs(table):
    """The numbers the UI has to be honest about: everything has similar, three
    quarters have synergy, a fifth have an obsolescence entry, and a quarter have
    nothing but similar."""
    counts = table["counts"]
    n = table["n"]
    assert (counts[:, 0] > 0).all(), "every card must have similar neighbours"
    assert 0.70 < (counts[:, 1] > 0).sum() / n < 0.80
    assert 0.18 < (counts[:, 2] > 0).sum() / n < 0.28
    only_similar = ((counts[:, 1] == 0) & (counts[:, 2] == 0)).sum() / n
    assert 0.20 < only_similar < 0.28


def test_synergy_is_a_capped_list_not_a_ranking(table):
    """Every card with any synergy has exactly 10 (min = median = max). The UI must
    not imply the first partner is the best one — the graph is rule-based."""
    have = table["counts"][:, 1]
    assert set(np.unique(have[have > 0]).tolist()) == {NEIGHBOURS_K_SYNERGY}


# ── viz_index ───────────────────────────────────────────────────────────


def test_viz_index_is_row_aligned(index):
    """Fifth member of the index invariant. Positional, like everything else."""
    import pandas as pd

    names = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)["name"].tolist()
    assert len(index) == len(names)
    assert [r["n"] for r in index[:200]] == names[:200]
    assert [r["n"] for r in index[-200:]] == names[-200:]


def test_viz_index_carries_what_the_landing_needs(index):
    """Enough to pick, filter and colour — and no oracle text, because the Scryfall
    card image already shows it and a local copy is weight nobody sees."""
    rec = index[0]
    assert set(rec) <= {"n", "s", "c", "r", "m", "g"}
    for key in ("n", "s", "c", "m"):
        assert key in rec
    assert not any("o" in r or "t" in r for r in index[:500]), "oracle text leaked in"


def test_viz_index_supports_the_coarse_filters(index):
    """Type, colour and CMC are the filters the landing offers, so all three must be
    populated rather than mostly blank."""
    assert sum(1 for r in index if r["s"] == "Creature") > 15000
    assert len({r["c"] for r in index}) >= 7
    assert sum(1 for r in index if r["m"] > 0) > 25000


def test_boot_payload_stays_small():
    """The whole point. If this regresses, discovery has stopped being cheaper than
    the thing it replaced."""
    total = sum(
        len(gzip.compress(p.read_bytes(), 6))
        for p in (VIZ_INDEX_PATH, NEIGHBOURS_BIN_PATH)
    )
    assert total < 3 * 1024 * 1024, f"boot payload grew to {total / 1048576:.2f} MB gzipped"
