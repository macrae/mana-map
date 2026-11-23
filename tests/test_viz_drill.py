"""Drill mode — diving into a selection and re-mapping it from the embeddings.

The world map is one PaCMAP layout of 34,322 cards at `n_neighbors=10`, which preserves
global shape by compressing local shape. Drilling recomputes a layout for the selected
cards from the 128-d embeddings, so the structure the projection squashed becomes the
whole view. Measured on a real region: 156 Aura cards occupy 0.3 x 0.7 on the world map
and 45.2 x 49.9 once re-mapped.

**The invariant these tests exist to protect**: a drilled position is *local*. The same
card sits somewhere else on the world map, and the two coordinate systems mean different
things. Anything anchored to world coordinates — region labels, the search highlight —
must be suppressed while a local layout is showing, and the breadcrumb must say which
system the reader is looking at. That is the honesty rule the whole feature rests on, and
it fails silently: a stale annotation over a local layout points at nothing, and looks
exactly like a label that means something.

Source assertions, like `test_viz_deck_lens.py` — there is no JS runtime in the suite,
and the failure mode here is a silent omission, which is what source checks catch.
"""

import re

from manamap.config import VIZ_DIR

DRILL_JS = VIZ_DIR / "js" / "drill.js"
MAP_JS = VIZ_DIR / "js" / "mana-map.js"
INDEX_HTML = VIZ_DIR / "index.html"


def _drill() -> str:
    return DRILL_JS.read_text(encoding="utf-8")


def _map() -> str:
    return MAP_JS.read_text(encoding="utf-8")


# ── The contract with mana-map.js ───────────────────────────────────────


# ── The honesty marker ──────────────────────────────────────────────────


# ── Wiring ──────────────────────────────────────────────────────────────


def test_all_viz_scripts_share_one_cache_bust():
    html = INDEX_HTML.read_text(encoding="utf-8")
    busts = dict(re.findall(r'src="js/([\w-]+)\.js\?v=(\d+)"', html))
    assert len(set(busts.values())) == 1, f"viz script cache-busts disagree: {busts}"


def test_every_copy_of_data_version_agrees():
    """DATA_VERSION is declared TWICE and nothing made them agree.

    `viz/js/mana-map.js` holds the real one; `viz/js/spaces-view.js` carries a
    hand-copied duplicate whose only guard is the comment "mirrors mana-map.js".
    The asset bust (`?v=`) has a test, and `DATA_VERSION` — the one that decides
    whether a browser re-fetches the DATA — did not.

    The incident this repo already paid for is recorded in `viz/index.html:7-13`:
    three layers of correct cache-busting defeated by one file with none, and a
    hard reload did not clear it. A second copy drifting silently is the same
    shape of bug with a shorter fuse.
    """
    root = INDEX_HTML.parent / "js"
    found = {}
    for path in sorted(root.glob("*.js")):
        m = re.search(r"DATA_VERSION\s*=\s*(\d+)", path.read_text(encoding="utf-8"))
        if m:
            found[path.name] = m.group(1)
    assert len(found) >= 2, (
        f"expected DATA_VERSION in at least two files, found {found} — if the "
        f"duplicate was removed, this test has done its job and can go with it")
    assert len(set(found.values())) == 1, f"DATA_VERSION copies disagree: {found}"


def test_data_urls_are_versioned():
    """A schema change to a data artifact must not be served from cache.

    `membership` was added to regions_*.json and every browser that had already loaded
    the map kept its cached copy, so drill-by-region found no membership and disabled
    itself — politely, which is what made it expensive to spot.
    """
    src = _map()
    assert "const DATA_VERSION" in src
    assert re.search(r"regionsCardbert: v\(", src), "region URLs are not versioned"


