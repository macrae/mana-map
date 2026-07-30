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


def test_open_card_image_is_not_lazy():
    """The only card image rendered is the open one, and it is scrolled into view as it
    appears — deferring it just adds a beat of grey."""
    js = _js()
    detail = js[js.index("function buildCardDetailHtml("):]
    detail = detail[:detail.index("function updateViewerPanel()")]
    # Comment lines mention the attribute by name; only the emitted markup matters.
    emitted = "\n".join(ln for ln in detail.splitlines() if not ln.strip().startswith("//"))
    assert 'loading="lazy"' not in emitted
