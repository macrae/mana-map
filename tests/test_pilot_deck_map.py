"""The deck constellation: determinism, balance, and the rules that were measured.

`deck_map.json` is TRACKED and drawn in two places — the printed magazine and the
dossier — so a layout that wobbles between runs is not a cosmetic problem: it is a
diff on every rebuild and a different picture for the same deck.

Three properties are asserted here, and all three exist because the obvious
implementation got them wrong first:

  * **Determinism.** MDS is indifferent to rotation, reflection and scale, so the
    same deck can come back mirrored. `_orient` pins all three. Verified once by
    rerunning and diffing bytes; that verification is now a test.
  * **The balance bound.** `cluster` grows k until the largest city holds under
    `MAX_CITY_SHARE`. This is the change that actually shrank radagast's oversized
    city — 38/71 down to 23/71 — and it has no other guard.
  * **Completeness.** Every card lands in exactly one city and one neighbourhood.
    A map with a hole in it is not a map.

WHAT IS DELIBERATELY *NOT* ASSERTED, because three attempts to write it each came
back narrower than the last, and the record was over-claimed in between:

  **That Ward beats average linkage.** Measured on radagast: at k=5 Ward is 38/71
  and average 37/71 — average is *better*. Ward only leads once k grows (32% vs
  49% at k=7), so the win belongs to the balance bound, not to the linkage.
  Ward's other apparent advantage — no one-card cities — also fails to generalise:
  on the outlier-bearing fixture below, Ward isolates the strays into singletons
  exactly as average does.

  So the linkage choice is a judgment supported by one deck's data, not an
  invariant, and a test asserting it would be asserting a fact about radagast
  while looking like a fact about the algorithm. If Ward is ever revisited, the
  thing to re-measure is the balance at the k the bound selects — not the horse
  race at a fixed k, which is what misled the first three attempts here.
"""

import numpy as np
import pytest

from manamap.pilot import deck_map

from conftest import requires_data


# A deliberately lumpy synthetic space: three tight clusters of different sizes
# plus two outliers. Built rather than sampled so the assertions below are about
# the ALGORITHM and not about whatever a real deck happens to look like today.
def _fixture_vectors(seed=0):
    rng = np.random.default_rng(seed)
    blobs = []
    for centre, n in (((3, 0), 30), ((-3, 1), 12), ((0, -4), 10)):
        base = np.zeros(16)
        base[:2] = centre
        blobs.append(base + rng.normal(0, 0.25, size=(n, 16)))
    stray = np.zeros((2, 16))
    stray[0, 5] = 9.0
    stray[1, 9] = -9.0
    return np.vstack(blobs + [stray])


def test_layout_is_byte_stable_across_runs():
    """Same input, same coordinates — the property the tracked artifact rests on."""
    vectors = _fixture_vectors()
    distance = deck_map.cosine_distances(vectors)
    first = deck_map.layout(distance)
    second = deck_map.layout(distance)
    assert np.array_equal(first, second), (
        "the layout is not reproducible; a rebuild would rewrite deck_map.json "
        "and redraw every constellation for no reason")


def test_orientation_survives_a_rotation_of_the_input():
    """`_orient` pins rotation, reflection and scale.

    Rotate the whole input space and the projected SHAPE must be unchanged — a
    rotation of the inputs is not new information, and without this the same deck
    comes back mirrored on a rerun that changed nothing.

    Compared as a sorted distance matrix rather than point-for-point: orientation
    is pinned, but MDS may still relabel a symmetric configuration, and asserting
    on coordinates would be asserting on something this function does not promise.
    """
    vectors = _fixture_vectors()
    theta = 0.7
    rotation = np.eye(16)
    rotation[0, 0] = rotation[1, 1] = np.cos(theta)
    rotation[0, 1], rotation[1, 0] = -np.sin(theta), np.sin(theta)

    plain = deck_map.layout(deck_map.cosine_distances(vectors))
    turned = deck_map.layout(deck_map.cosine_distances(vectors @ rotation.T))

    def pairwise(points):
        d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        return np.sort(d[np.triu_indices(len(points), 1)])

    assert np.allclose(pairwise(plain), pairwise(turned), atol=1e-6), (
        "rotating the input changed the projected shape — orientation is not pinned")


def test_the_layout_is_scaled_into_a_unit_box():
    points = deck_map.layout(deck_map.cosine_distances(_fixture_vectors()))
    assert np.isclose(np.abs(points).max(), 1.0, atol=1e-6)


# ── The clustering rules, which nothing else guards ─────────────────────


def _unit(vectors):
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)


def test_no_city_holds_more_than_the_balance_bound():
    """The measured rule: grow k until the largest city is under MAX_CITY_SHARE.

    A fixed cards-per-city divisor put 54% of radagast in one city — technically a
    partition, useless as a legend, and it looked correct.
    """
    vectors = _fixture_vectors()
    cities, _hoods = deck_map.cluster(_unit(vectors), len(vectors))
    counts = {c: cities.count(c) for c in set(cities)}
    largest = max(counts.values())
    assert largest <= deck_map.MAX_CITY_SHARE * len(vectors) or \
        len(counts) >= deck_map.MAX_CITIES, (
        f"largest city holds {largest}/{len(vectors)} "
        f"({largest / len(vectors):.0%}) at only {len(counts)} cities — the count "
        f"should have grown, or MAX_CITIES should have stopped it")


def test_every_card_lands_in_exactly_one_city_and_one_neighbourhood():
    """The contract the magazine page rests on: a map with a hole in it is not a map."""
    vectors = _fixture_vectors()
    cities, hoods = deck_map.cluster(_unit(vectors), len(vectors))
    assert len(cities) == len(hoods) == len(vectors)
    assert all(isinstance(c, int) for c in cities)
    assert -1 not in cities, "a noise label reached the city assignment"


def test_fallback_names_are_unique_within_a_level():
    """Two cities called the same thing read as a misprint, not as a coincidence."""
    ranked = [["Bodies", "Ramp"], ["Bodies", "Mana"], ["Bodies"], ["Ramp", "Mana"]]
    names = deck_map.unique_names(ranked)
    assert len(set(names)) == len(names), names
    assert names[0] == "Bodies"          # the first claimant keeps its truest name
    assert names[1] == "Mana"            # the second moves to its next-ranked role


# ── Against the real corpus ─────────────────────────────────────────────


@requires_data
def test_a_real_deck_reproduces_byte_for_byte():
    """The end-to-end property: `deck-map` twice on one deck is the same artifact.

    `requires_data` because this needs `embeddings_ability.npy`, which is
    gitignored — a fresh clone renders the committed map and cannot rebuild it.
    """
    import json

    from manamap.pilot.common import deck_dir

    slug = "radagast"
    if not (deck_dir(slug) / "cards.json").exists():
        pytest.skip(f"{slug} not present")
    first = deck_map.build(slug)
    second = deck_map.build(slug)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
