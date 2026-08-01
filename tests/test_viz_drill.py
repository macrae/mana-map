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


def test_drill_exposes_the_contract_render_calls():
    src = _drill()
    for member in ("isActive", "hidesWorld", "getOverlayTraces",
                   "getDimmedIndices", "getContourSource"):
        assert re.search(rf"^\s+{member},$", src, re.M), f"Drill.{member} not exported"


def test_render_hides_the_world_rather_than_dimming_it():
    """Dimming would leave both coordinate systems on screen at once.

    The deck overlays dim because they highlight a subset *in place*. Drill moves the
    points, so the world must go away entirely — a dimmed world card at its global
    position beside a drilled card at its local position is two different claims about
    what a position means, rendered identically.
    """
    src = _map()
    assert "window.Drill.hidesWorld()" in src
    assert "visible: drilling ? false : true" in src


def test_world_anchored_layers_are_suppressed_while_drilling():
    src = _map()
    # Region labels were Plotly annotations rebuilt per render; they are DOM buttons now,
    # and `refreshCanvasLabels` is what decides whether any exist. The invariant is the
    # same — nothing anchored to world centroids may render over a local layout.
    assert "if (!data || !showRegionLabels) { mapCanvas.setAnnotations([]); return; }" in src, (
        "region labels are anchored to world centroids and must not render over a "
        "local layout"
    )
    assert "searchTerm.length >= 2 && !drilling" in src, (
        "the search highlight plots world coordinates and must not render over a "
        "local layout"
    )


def test_selection_highlight_follows_the_active_coordinate_system():
    """The gold selection ring is drawn after every render, from `allData[i].x`.

    While drilling that is a world coordinate painted over a local layout — a marker
    pointing at nothing, in the one colour the map uses to mean "this is the card you
    are looking at". Found by eye in a screenshot after the source checks passed, which
    is why the position lookup is now shared rather than repeated inline.
    """
    src = _map()
    # One shared lookup, used by both the 8-card stack and browse mode — two copies
    # would drift, and only one of them would be fixed the next time this is found.
    assert "function cardPosition(idx)" in src
    assert "window.Drill.localPosition(idx)" in src
    assert "const posOf = cardPosition;" in src
    # A card outside the drilled subset has no local position and must be dropped,
    # not defaulted back to its world coordinate.
    assert ".filter(c => posOf(c.idx))" in src
    assert ".filter(i => posOf(i))" in src, "browse mode must drop unpositioned cards too"

    drill_src = _drill()
    assert "function localPosition(rowIdx)" in drill_src
    assert "return i === -1 ? null" in drill_src, (
        "localPosition must return null for a card outside the subset"
    )


def test_contours_rebin_over_the_drilled_subset():
    src = _map()
    assert "drilling ? window.Drill.getContourSource() : filtered" in src


def test_status_line_reports_the_local_count():
    """`34,322 cards shown` is false while drilling — those cards are not on screen."""
    src = _map()
    assert "local layout from the 128-dim embeddings" in src


# ── The honesty marker ──────────────────────────────────────────────────


def test_breadcrumb_always_declares_the_local_layout():
    """Every rendered drill state carries the marker, including mid-flight.

    `renderBar` has exactly three branches — hidden, the pending offer, and an active
    drill — and only the active one may omit... nothing. It must carry `drill-local`.
    """
    src = _drill()
    assert "drill-local" in src
    active_branch = src[src.index("const crumbs = ["):src.index("window.Drill = {")]
    assert "drill-local" in active_branch, (
        "the active-drill branch of renderBar must state that positions are local"
    )


def test_truncation_is_announced_not_silent():
    """A cap that hides how much it dropped reads as 'this is everything'."""
    src = _drill()
    assert "truncatedFrom" in src
    assert "drill-warn" in src
    assert "showing " in src


def test_hidden_tab_still_lands_on_a_settled_layout():
    """requestAnimationFrame does not fire in a background tab.

    Without a fallback, switching tabs mid-flight freezes the points at meaningless
    intermediate positions forever — the callback that would schedule the next frame
    never runs, so it cannot recover on its own.
    """
    src = _drill()
    assert "visibilitychange" in src
    assert "function finishNow()" in src
    assert "if (document.hidden)" in src


# ── Wiring ──────────────────────────────────────────────────────────────


def test_index_loads_drill_before_the_modes_that_call_it():
    html = INDEX_HTML.read_text(encoding="utf-8")
    order = re.findall(r'src="js/([\w-]+)\.js\?v=\d+"', html)
    assert "drill" in order, "drill.js not loaded by viz/index.html"
    assert order.index("mana-map") < order.index("drill"), (
        "drill.js reads MM at load time and must come after mana-map.js"
    )
    assert 'id="drillBar"' in html, "the breadcrumb has nowhere to render"


def test_all_viz_scripts_share_one_cache_bust():
    html = INDEX_HTML.read_text(encoding="utf-8")
    busts = dict(re.findall(r'src="js/([\w-]+)\.js\?v=(\d+)"', html))
    assert len(set(busts.values())) == 1, f"viz script cache-busts disagree: {busts}"


def test_data_urls_are_versioned():
    """A schema change to a data artifact must not be served from cache.

    `membership` was added to regions_*.json and every browser that had already loaded
    the map kept its cached copy, so drill-by-region found no membership and disabled
    itself — politely, which is what made it expensive to spot.
    """
    src = _map()
    assert "const DATA_VERSION" in src
    assert re.search(r"regionsDefault: v\(", src), "region URLs are not versioned"


def test_escape_pops_the_drill_before_anything_else():
    src = _map()
    escape_block = src[src.index("if (e.key === 'Escape')"):]
    escape_block = escape_block[:escape_block.index("return;")]
    assert "window.Drill.isActive()" in escape_block
    assert escape_block.index("window.Drill") < escape_block.index("DeckBuilder")
