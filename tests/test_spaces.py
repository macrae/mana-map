"""The embedding-space registry: one home, and the artifacts that travel with it."""

import numpy as np
import pytest

from manamap import spaces


def test_no_two_spaces_claim_the_same_artifact():
    """THE BUG THIS WAS WRITTEN FOR HAPPENED WHILE WRITING THE REGISTRY.

    The first version derived every artifact name from a `_<slug>` stem, which
    invented `embeddings_layout.bin` and `projection_2d_layout.json` alongside the
    `embeddings.bin` and `projection_2d.json` the layout space has ALREADY had on
    disk since the pipeline was written. Two names for one artifact means one of
    them is stale the moment the other is rebuilt.
    """
    owned = {}
    for slug, space in spaces.SPACES.items():
        for kind in ("bin", "projection", "regions", "neighbours"):
            path = getattr(space, kind)
            if path is None:
                continue
            assert path not in owned, (
                f"{slug} and {owned[path]} both claim {path.name}")
            owned[path] = slug
    assert len(owned) >= 11


def test_the_incumbent_keeps_the_filenames_it_already_has():
    """A deployed page fetches these by name. Renaming them is a data migration
    for no gain, and `neighbours.bin` in particular is in a live URL."""
    from manamap import config

    function = spaces.get("function")
    assert function.bin == config.ABILITY_EMBEDDINGS_BIN_PATH
    assert function.projection == config.ABILITY_PROJECTION_PATH
    assert function.regions == config.REGIONS_ABILITY_PATH
    assert function.neighbours == config.NEIGHBOURS_BIN_PATH

    layout = spaces.get("layout")
    assert layout.bin.name == "embeddings.bin"
    assert layout.projection.name == "projection_2d.json"
    assert layout.regions == config.REGIONS_DEFAULT_PATH


def test_the_default_is_the_incumbent():
    """CardBERT loses functional similarity at every pool size measured, so it is
    offered rather than imposed. If this ever changes it should change HERE, once,
    with a measurement — not by a caller passing a different string."""
    assert spaces.DEFAULT == "function"
    assert spaces.get() is spaces.get("function")
    assert spaces.get(None) is spaces.get("function")


def test_only_128_dimension_spaces_are_browsable():
    """`viz/js/mana-map.js:63` hardcodes `EMBED_DIM = 128` and the .bin is
    HEADERLESS with no validation, so a 384-d file parses as plausible garbage
    rather than failing. `text` must never reach the frontend."""
    assert "text" not in spaces.BROWSABLE
    assert "layout" not in spaces.BROWSABLE, (
        "the layout space knows only colour and type — its neighbours are "
        "arbitrary same-colour cards")
    for slug in spaces.BROWSABLE:
        assert spaces.get(slug).bin is not None, f"{slug} has nothing to serve"


def test_a_browsable_space_owns_every_artifact_the_frontend_fetches():
    for slug in spaces.BROWSABLE:
        space = spaces.get(slug)
        for kind in ("bin", "projection", "regions", "neighbours"):
            assert getattr(space, kind) is not None, f"{slug} is missing {kind}"


def test_an_unknown_slug_names_the_known_ones():
    with pytest.raises(KeyError, match="unknown embedding space"):
        spaces.get("funktion")


def test_choices_are_derived_not_duplicated():
    """`pilot/registry.py` carried a hand-copied literal of these slugs, so a
    rename would have left the flag accepting a slug nothing could resolve."""
    assert spaces.choices() == sorted(spaces.SPACES)
    for slug in spaces.choices():
        assert spaces.get(slug).slug == slug


def test_every_space_explains_itself():
    """A registry entry with no note is a path with a nickname."""
    for slug, space in spaces.SPACES.items():
        assert len(space.note) > 40, f"{slug} has no usable note"
        assert space.label, slug


@pytest.mark.skipif(not spaces.get("cardbert").exists(),
                    reason="cardbert matrix not built")
def test_load_returns_unit_rows():
    """The JS treats a dot product AS a cosine and never renormalises
    (`viz/js/mana-map.js:1417-1418`), so a space that reaches the browser
    unnormalised makes every similarity wrong without erroring."""
    matrix = spaces.load("cardbert")
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


@pytest.mark.skipif(not spaces.get("function").exists(),
                    reason="function matrix not built")
def test_browsable_spaces_agree_on_row_count():
    """Row index is the join key across `cards.csv`, the projection, the regions
    membership and the .bin. Two browsable spaces of different lengths would put
    a card's coordinates and its neighbours on different cards."""
    lengths = {slug: np.load(spaces.get(slug).npy, mmap_mode="r").shape[0]
               for slug in spaces.BROWSABLE if spaces.get(slug).exists()}
    assert len(set(lengths.values())) == 1, lengths
