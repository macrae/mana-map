"""The embedding spaces, and every artifact derived from each one.

## WHY THIS EXISTS

Three registries described the same set of spaces and none of them knew about the
others:

    analysis/commander_search.py   SPACES = {"text": …, "function": …}   slugs
    analysis/eval_embeddings.py    spaces_on_disk() -> {label: path}     labels
    pilot/registry.py:642-648      choices=["text", "function", "layout"] literal

Adding a fourth space meant editing all three, and the third was a hand-copied
duplicate of the first that nothing checked. This module is the one home; the
others alias it or derive from it.

## WHAT A SPACE OWNS

A space is not just a matrix — it is a matrix plus the four artifacts built from
it, and they must travel together or the frontend reads neighbours from one space
and coordinates from another:

    npy          the embedding matrix, corpus order (gitignored, regenerable)
    bin          L2-NORMALISED contiguous float32 for the browser
    projection   the 2D layout PaCMAP produced from it
    regions      the clusters found in that projection
    neighbours   precomputed top-k, with the matrix sha in its header

## THE DEFAULT DOES NOT MOVE

`function` stays the default everywhere, and its artifacts keep their existing
unsuffixed filenames, so adding a space moves nothing that already works. That is
not caution for its own sake — CardBERT LOSES functional similarity at every pool
size measured (−0.275 at pool 500, interval excluding zero) while winning theme
(+0.151) and centroid headroom (0.976 against 0.019). It is a trade, so it is
offered rather than imposed.

## OBSOLESCENCE IS DELIBERATELY NOT SELECTABLE

`power_creep` gates on `OBSOLESCENCE_SIMILARITY_THRESHOLD = 0.75`, and that
number is calibrated to one space's cosine distribution. Measured over 40,000
random pairs:

    function (ability)   mean +0.719    31.8175% of pairs above 0.75
    cardbert             mean -0.002     0.4175% of pairs above 0.75

0.75 sits barely above the MEAN in one space and at the 99.6th percentile in the
other. Making obsolescence selectable without recalibrating the thresholds per
space would silently empty the index while looking like it worked.
"""

from dataclasses import dataclass
from pathlib import Path

from manamap import config

#: The slug used when a caller does not choose. Every default in the repo points
#: here rather than naming a string, so this is the one line that moves if the
#: measurement ever justifies moving it.
DEFAULT = "function"


@dataclass(frozen=True)
class Space:
    """One embedding space and the artifacts derived from it."""

    slug: str
    label: str
    npy: Path
    #: `None` means the artifact is not built for this space — `text` exports
    #: nothing, and `layout` has no neighbours file.
    bin: Path | None
    projection: Path | None
    regions: Path | None
    neighbours: Path | None
    note: str = ""

    def exists(self):
        return self.npy.exists()


#: THE EXISTING ARTIFACTS KEEP THEIR EXISTING NAMES. Both `function` and `layout`
#: already have derived files on disk under names a deployed page fetches, so a
#: rename would be a data migration for nothing. Only a NEW space gets the
#: `_<slug>` suffix, which is why this is a table rather than a rule — the first
#: version derived every name from a stem and quietly invented
#: `embeddings_layout.bin` alongside the `embeddings.bin` that already existed.
SPACES = {
    "function": Space(
        slug="function", label="function (ability)",
        npy=config.ABILITY_EMBEDDINGS_PATH,
        bin=config.ABILITY_EMBEDDINGS_BIN_PATH,
        projection=config.ABILITY_PROJECTION_PATH,
        regions=config.REGIONS_ABILITY_PATH,
        neighbours=config.NEIGHBOURS_BIN_PATH,
        note="the incumbent. Positives mined from role and tag regexes, so "
             "function is what it was built for and tribe is what it discards."),
    "cardbert": Space(
        slug="cardbert", label="cardbert (masked fields)",
        npy=config.DATA_DIR / "embeddings_cardbert.npy",
        bin=config.DATA_DIR / "embeddings_cardbert.bin",
        projection=config.DATA_DIR / "projection_2d_cardbert.json",
        regions=config.DATA_DIR / "regions_cardbert.json",
        neighbours=config.DATA_DIR / "neighbours_cardbert.bin",
        note="masked-field imputation over 73 typed fields and 6 text spans. "
             "Wins theme and centroid headroom, loses functional similarity."),
    "layout": Space(
        slug="layout", label="layout (color+type)",
        npy=config.EMBEDDINGS_PATH,
        bin=config.DATA_DIR / "embeddings.bin",
        projection=config.DATA_DIR / "projection_2d.json",
        regions=config.REGIONS_DEFAULT_PATH,
        neighbours=None,
        note="colour and type only, 3.89 of 128 dimensions in use. Feeds the "
             "default map; NEVER a similarity source — it knows only colour and "
             "type, so its neighbours are arbitrary same-colour cards."),
    "text": Space(
        slug="text", label="text baseline (frozen MiniLM)",
        npy=config.TEXT_EMBEDDINGS_PATH,
        bin=None, projection=None, regions=None, neighbours=None,
        note="frozen sentence vectors, 384-d and untrained. The measured default "
             "for commander search; nothing is exported from it, and its 384 "
             "dims would misparse against the frontend's hardcoded 128."),
}

#: Spaces a browser may be pointed at. `text` is 384-d and the frontend's
#: `EMBED_DIM` is a hardcoded 128 (`viz/js/mana-map.js:63`) with NO validation on
#: a headerless .bin — a 384-d file would parse as plausible garbage rather than
#: fail. `layout` is excluded for a different reason: it knows only colour and
#: type, so asking it for neighbours returns arbitrary same-colour cards.
BROWSABLE = ("function", "cardbert")


def get(slug=None):
    """A `Space` by slug. `None` means the default."""
    slug = DEFAULT if slug is None else slug
    if slug not in SPACES:
        raise KeyError(
            f"unknown embedding space {slug!r} — known: {sorted(SPACES)}")
    return SPACES[slug]


def choices():
    """Slugs, for an argparse `choices=`.

    Derived rather than duplicated: `pilot/registry.py` carried a hand-copied
    literal of these, so a rename would have left the flag accepting a slug the
    registry no longer knew.
    """
    return sorted(SPACES)


def on_disk():
    """Every space whose matrix is actually present."""
    return {slug: space for slug, space in SPACES.items() if space.exists()}


def load(slug=None):
    """The matrix, L2-NORMALISED.

    Delegates to `commander_search.normalized` rather than repeating it — one
    normalisation, one place. Every browser consumer assumes unit rows because
    the JS treats a dot product as a cosine and never renormalises
    (`viz/js/mana-map.js:1417-1418`).
    """
    from manamap.analysis.commander_search import normalized

    return normalized(get(slug).npy)
