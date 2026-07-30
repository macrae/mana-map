"""The card viewer — browsing a multi-card selection without losing your place.

The old panel put the open card's detail at the top and the list of selected cards
underneath it. Choosing a different card meant scrolling down past the whole card to
reach the list, clicking, and then scrolling back up to look at what you picked. With
eight cards selected that is a scroll on every single change.

The accordion inverts it: the list *is* the panel, and the open card expands inside the
row you clicked, so the point of interaction and the thing it reveals stay in the same
place. A sticky header keeps the arrows and the close button reachable from anywhere in
the list.

These are source assertions, like the other viz suites — there is no JS runtime here.
Behaviour was verified in a browser: the open row lands at 89px from the panel top on
every path (click, arrows, arrow keys, number keys, adding a card from the map).
"""

import re

from manamap.config import VIZ_DIR

MAP_JS = VIZ_DIR / "js" / "mana-map.js"
MAP_CSS = VIZ_DIR / "css" / "mana-map.css"


def _js() -> str:
    return MAP_JS.read_text(encoding="utf-8")


def _css() -> str:
    return MAP_CSS.read_text(encoding="utf-8")


def _viewer_fn() -> str:
    src = _js()
    start = src.index("function updateViewerPanel()")
    return src[start:src.index("function scrollActiveRowIntoView()")]


def test_detail_renders_inside_the_open_row_not_above_the_list():
    """The whole point: the card expands where you clicked, not somewhere else."""
    body = _viewer_fn()
    assert "acc-body" in body
    # The detail builder must be called inside the per-row loop, guarded by isActive —
    # not once, above the list.
    assert re.search(r"if \(isActive\) \{\s*\n\s*html \+= '<div class=\"acc-body\">' \+ buildCardDetailHtml",
                     body), "the card detail is not rendered inside the active row"


def test_only_the_open_row_carries_a_body():
    """Rendering all eight bodies would restore the scrolling problem, just inline."""
    body = _viewer_fn()
    acc_block = body[body.index("<div class=\"accordion\">"):body.index("keyboard-hint")]
    assert acc_block.count("buildCardDetailHtml") == 1


def test_every_path_reveals_the_open_row():
    """Scrolling lives in updateViewerPanel, so no caller can forget it.

    It was originally only in bringToTop, which meant clicking a row scrolled correctly
    but selecting a new card from the map did not.
    """
    js = _js()
    body = _viewer_fn()
    assert "scrollActiveRowIntoView();" in body, (
        "updateViewerPanel must reveal the open row itself"
    )
    # And bringToTop must not do it a second time.
    bring = js[js.index("function bringToTop("):js.index("// ── Viewer Panel ──")]
    assert "scrollActiveRowIntoView" not in bring


def test_arrows_and_arrow_keys_share_one_implementation():
    """Two cycle implementations would drift — one wrapping, one clamping."""
    js = _js()
    assert "function cycleSelection(delta)" in js
    assert "cyclePrev: () => cycleSelection(-1)" in js
    assert "cycleNext: () => cycleSelection(1)" in js
    # The key handler delegates rather than recomputing an index.
    assert "e.preventDefault();\n      cycleSelection(-1);" in js
    assert "e.preventDefault();\n      cycleSelection(1);" in js


def test_cycling_wraps_in_both_directions():
    js = _js()
    fn = js[js.index("function cycleSelection(delta)"):]
    fn = fn[:fn.index("\n  }")]
    assert "% n + n) % n" in fn, "cycleSelection must wrap rather than clamp"


def test_header_is_sticky_and_bleeds_across_the_panel_padding():
    """A sticky header sized to the content box leaves gutters either side.

    `.detail-inner` has 16px padding, so without the negative side margins the scrolling
    list shows through beside the header. Measured before the fix: 16px left, 22px right.
    """
    css = _css()
    header = css[css.index(".viewer-header {"):]
    header = header[:header.index("}")]
    assert "position: sticky" in header
    assert "margin: -16px -16px" in header, "header background does not cover the padding"
    assert "padding: 16px 16px" in header


def test_scroll_container_is_positioned_for_offsettop_maths():
    """scrollActiveRowIntoView measures row.offsetTop against .detail-inner."""
    css = _css()
    inner = css[css.index(".detail-inner {"):]
    inner = inner[:inner.index("}")]
    assert "position: relative" in inner


def test_neighbour_images_are_preloaded():
    """Each card image is a Scryfall round-trip; without this every arrow press shows
    a beat of empty grey, which is most of what made browsing feel slow."""
    js = _js()
    assert "function preloadNeighbourImages()" in js
    assert "preloadNeighbourImages();" in _viewer_fn()
    fn = js[js.index("function preloadNeighbourImages()"):]
    fn = fn[:fn.index("\n  }")]
    assert "[-1, 1]" in fn, "preload should cover both neighbours, not just the next"


# ── Browse mode (selections too big for the accordion) ──────────────────


def _browse_fn(name: str) -> str:
    js = _js()
    start = js.index(f"function {name}(")
    return js[start:js.index("\n  }", start)]


def test_box_select_keeps_the_whole_set_not_an_arbitrary_eight():
    """`plotly_selected` returns points grouped by trace — colour groups in palette
    order (G, R, Colorless, U, B, W, Multicolor), then cards.csv row order within each.
    Taking the first 8 meant boxing a mixed cluster and getting eight green cards in
    Scryfall dump order: not a sample of the selection, an artifact of trace construction.
    """
    js = _js()
    start = js.index("on('plotly_selected'")
    handler = js[start:start + 2500]
    assert "enterBrowse(all, 'Selection')" in handler
    assert "MAX_SELECTED} of ${total}" not in handler, "the arbitrary-8 truncation is back"


def test_browse_order_uses_the_embeddings_not_the_projection():
    """Screen distance is the projection's compromise. An ordering exists to say
    something the picture does not already."""
    fn = _browse_fn("orderByCentroidDistance")
    assert "EMBED_DIM" in fn and "embeddings[" in fn
    assert ".x" not in fn and ".y" not in fn, "ordering must not use 2D positions"


def test_browse_order_is_furthest_first():
    fn = _browse_fn("orderByCentroidDistance")
    assert "sort((a, b) => b.d - a.d)" in fn, "must sort descending — least typical first"


def test_centroid_is_renormalised_before_scoring():
    """Rows are L2-normalised at export so a dot product is a cosine, but the *mean* of
    unit vectors is not itself a unit vector — without renormalising, the distances are
    scaled by the centroid's length and the ordering silently depends on how tightly
    clustered the selection happens to be."""
    fn = _browse_fn("orderByCentroidDistance")
    assert "Math.sqrt(norm)" in fn
    assert "centroid[i] /= norm" in fn


def test_browse_marker_moves_without_rebuilding_the_selection():
    """Rebuilding the highlight per arrow press was a deleteTraces + addTraces of the
    whole selection — measured 197 ms per step on a 3,434-card browse."""
    js = _js()
    assert "function moveBrowseMarker()" in js
    assert "if (!moveBrowseMarker()) updateSelectionHighlight();" in js, (
        "cycling must try the fast path and fall back, not always rebuild"
    )
    fn = _browse_fn("moveBrowseMarker")
    assert "_isBrowseCurrent" in fn, "find the marker by flag, not by trace name"
    assert "Plotly.restyle" in fn


def test_browse_mode_is_exclusive_with_the_card_stack():
    """Two 'current card' markers on the plot at once would be two different claims."""
    js = _js()
    assert "browseSet = null;" in _browse_fn("addToSelection")
    assert "browseSet = null;" in _browse_fn("clearSelection")
    enter = _browse_fn("enterBrowse")
    assert "selectedCards = [];" in enter


def test_browse_panel_has_no_list_and_explains_its_order():
    """A list of 400 names is a wall, not navigation — so the ordering has to carry the
    meaning the list would have, which means saying what it is."""
    js = _js()
    panel = js[js.index("function renderBrowsePanel()"):js.index("// Put the open row's header")]
    assert "acc-row" not in panel and "accordion" not in panel
    assert "browse-order-label" in panel
    assert "least typical" in panel


def test_open_card_image_is_not_lazy():
    """The only card image rendered is the open one, and it is scrolled into view as it
    appears — deferring it just adds a beat of grey."""
    js = _js()
    detail = js[js.index("function buildCardDetailHtml("):]
    detail = detail[:detail.index("function updateViewerPanel()")]
    # Comment lines mention the attribute by name; only the emitted markup matters.
    emitted = "\n".join(ln for ln in detail.splitlines() if not ln.strip().startswith("//"))
    assert 'loading="lazy"' not in emitted
