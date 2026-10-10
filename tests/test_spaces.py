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


def test_the_pages_use_cardbert_and_the_bench_keeps_function():
    """THE PILOT'S DECISION, 2026-10-10: the pages draw CardBERT and answer
    similarity out of it, with no picker; the bench's default does not move.
    Two constants because they are two decisions — collapsing them would move
    every `manamap pilot` command along with the Atlas."""
    assert spaces.UI == "cardbert"
    assert spaces.BROWSABLE == (spaces.UI,)
    assert spaces.DEFAULT == "function"
    assert spaces.UI != spaces.DEFAULT


def test_the_frontend_mirrors_the_ui_space():
    """`mana-map.js` holds a JS copy of the registry; a silent duplicate is how
    the two sides drift. The one space and the one map it boots must be `UI`."""
    import re

    from manamap import config

    src = (config.VIZ_DIR / "js" / "mana-map.js").read_text()
    assert re.search(r"const currentSpace = '" + spaces.UI + "'", src)
    assert re.search(r"let currentMap = '" + spaces.UI + "'", src)


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


# ── the tracked viz artifacts ───────────────────────────────────────────


@pytest.mark.parametrize("slug", sorted(spaces.BROWSABLE))
def test_a_browsable_space_ships_all_four_artifacts(slug):
    """GitHub Pages serves `data/` as static files and a browser cannot run the
    pipeline, so a browsable space that is missing one artifact works locally and
    404s in production — the worst kind of difference between the two.

    There is no LFS here on purpose: Pages would serve the pointer file.
    """
    space = spaces.get(slug)
    for kind in ("bin", "projection", "regions", "neighbours"):
        path = getattr(space, kind)
        assert path.exists(), f"{slug} is missing {kind} ({path.name})"


@pytest.mark.parametrize("slug", sorted(spaces.BROWSABLE))
def test_the_shipped_binary_is_unit_norm_and_the_right_length(slug):
    """The .bin is HEADERLESS — nothing in it records rows or dims, so a
    truncated or wrong-dimension file parses fine and every offset is silently
    wrong. And the JS treats a dot product AS a cosine
    (`viz/js/mana-map.js:1417`), so non-unit rows make every similarity wrong
    without erroring. This is the only place either can be caught.
    """
    import pandas as pd

    from manamap.config import OUTPUT_CSV_PATH

    if not OUTPUT_CSV_PATH.exists():
        pytest.skip("corpus not built")
    cards = len(pd.read_csv(OUTPUT_CSV_PATH, low_memory=False))
    raw = spaces.get(slug).bin.read_bytes()
    assert len(raw) == cards * 128 * 4, (
        f"{slug}.bin is {len(raw)} bytes, expected {cards} x 128 x 4")
    matrix = np.frombuffer(raw, "<f4").reshape(cards, 128)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4)


@pytest.mark.parametrize("slug", sorted(spaces.BROWSABLE))
def test_neighbours_carry_the_digest_of_their_own_matrix(slug):
    """A neighbours file built from a DIFFERENT matrix than the .bin beside it
    answers "what is similar" out of one space while the coordinates come from
    another, and nothing at runtime checks it — the header digest is the only
    record of which matrix produced these tables."""
    import struct

    from manamap.export.viz_index import embeddings_digest

    space = spaces.get(slug)
    if not space.npy.exists():
        pytest.skip(f"{slug} matrix is gitignored and not built here")
    header = space.neighbours.read_bytes()[:52]
    magic, _, _, _, _, _, _, digest = struct.unpack("<4sIIHHHH32s", header)
    assert magic == b"MMNB"
    assert digest == embeddings_digest(space.npy), (
        f"{slug}/neighbours was built from a different matrix than {space.npy.name}")


@pytest.mark.parametrize("slug", sorted(spaces.BROWSABLE))
def test_the_served_outclassed_by_block_matches_the_obsolescence_index(slug):
    """THE BUTTON'S COUNT AND THE LIST UNDER IT MUST BE ONE ANSWER.

    The card panel's "Outclassed by N" count and Discover's obsolete branch read
    the served neighbour table; the comparison list under the button reads
    `obsolescence_index.json`. Obsolescence is not space-dependent (see this
    module's docstring), so every browsable space's table must carry exactly the
    index's `compare_with` rows. Found when the pages moved to CardBERT
    (2026-10-10): `neighbours_cardbert.bin` had been built from an older index and
    disagreed on 2,157 cards — The Elder Dragon War's button said 1 over a list of 2.
    The fix is a rebuild (`manamap viz-index --space cardbert`), never a looser test.
    """
    import json
    import struct

    from manamap.config import OBSOLESCENCE_INDEX_PATH, VIZ_INDEX_PATH
    from manamap.export.viz_index import build_name_index

    space = spaces.get(slug)
    buf = space.neighbours.read_bytes()
    n, = struct.unpack_from("<I", buf, 8)
    ks, ky, ko = struct.unpack_from("<HHH", buf, 12)
    off = 64 + n * ks * 2 + n * ky * 2
    obs = np.frombuffer(buf, np.uint16, n * ko, off).reshape(n, ko)

    names = [r["n"] for r in json.loads(VIZ_INDEX_PATH.read_text(encoding="utf-8"))]
    assert len(names) == n
    name_index = build_name_index(names)
    index = json.loads(OBSOLESCENCE_INDEX_PATH.read_text(encoding="utf-8"))
    wrong = []
    checked = 0
    for row, name in enumerate(names):
        entry = index.get(name) or {}
        better = entry.get("compare_with") or entry.get("obsoleted_by") or []
        want = [name_index[b["name"]] for b in better if b.get("name") in name_index][:ko]
        got = [int(r) for r in obs[row] if r != 0xFFFF]
        checked += bool(want)
        if got != want:
            wrong.append(name)
    assert checked >= 5000, f"only {checked} cards carry an outclassed-by entry"
    assert not wrong, (
        f"{slug}: {len(wrong):,} cards' outclassed-by rows disagree with "
        f"obsolescence_index.json (e.g. {wrong[:3]}) — rebuild with "
        f"`manamap viz-index --space {slug}`")
