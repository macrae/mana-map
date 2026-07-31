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
    """Two cycle implementations would drift — one wrapping, one clamping.

    Asserts the delegation, not the exact source shape: the previous version matched
    literal indentation and broke the moment the key handler was rewritten to fix the
    browse-mode gate, even though the invariant it cared about was untouched.
    """
    js = _js()
    assert "function cycleSelection(delta)" in js
    assert "cyclePrev: () => cycleSelection(-1)" in js
    assert "cycleNext: () => cycleSelection(1)" in js
    # The key handler must delegate rather than recompute an index of its own.
    handler = js[js.index("document.addEventListener('keydown'"):]
    handler = handler[:handler.index("// ── Shift+Drag Box Select ──")]
    assert "cycleSelection(" in handler, "arrow keys no longer delegate to cycleSelection"
    assert "topCardIndex +" not in handler, "the key handler is recomputing an index itself"


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


# ── Performance invariants ──────────────────────────────────────────────
#
# The map renders 34,322 WebGL points. Every one of these was measured, and each
# regressed silently before it was found — nothing threw, the app just got slower.


def test_no_trace_builds_hover_text_while_hover_is_off():
    """Every trace sets `hoverinfo: 'none'` and nothing reads `trace.text`.

    Building it anyway meant ~34,000 escHtml calls per render — four chained global
    regexes on each of three fields — for strings that were thrown away. Measured at
    37 ms of a 90 ms render; removing it took render to 30 ms.

    If hover is ever turned on, build the text in the hover callback for the one point
    under the cursor, not for all 34K up front.
    """
    for path in sorted((VIZ_DIR / "js").glob("*.js")):
        offenders = [
            ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if "buildHoverTextMinimal(" in ln
            and not ln.strip().startswith(("//", "*"))
            and "function buildHoverTextMinimal" not in ln
        ]
        assert not offenders, (
            f"{path.name} still builds hover text for a trace that never shows it: "
            f"{offenders[:2]}"
        )


def test_render_draws_the_selection_in_its_single_react():
    """`Plotly.react` replaces the trace list, so it dropped the highlight and
    `updateSelectionHighlight()` added it straight back — an extra addTraces of the whole
    selection on every render. With a 15,000-card browse that is a full trace rebuild on
    every pan, filter and panel open. One react now draws everything."""
    js = _js()
    assert "function buildSelectionTraces()" in js
    assert "traces.push(...buildSelectionTraces());" in js
    # render() registers the plot event handlers inside its first-run branch, and the
    # box-select handler legitimately calls updateSelectionHighlight for the <=8 path.
    # What must not come back is the unconditional re-add at the END of render.
    start = js.index("function render()")
    render = js[start:js.index("\n  function ", start + 10)]
    tail = render[render.rindex("Plotly.react("):]
    assert "updateSelectionHighlight();" not in tail, (
        "render() must not re-add the highlight it already drew"
    )
    assert "Re-apply selection highlight after render" not in js


def test_browse_highlight_is_cached_by_set_identity():
    """Only the marker moves as you cycle; the set does not. Without the key check the
    whole selection was torn down and rebuilt per press (197 ms at 3,434 cards)."""
    js = _js()
    assert "function browseHighlightKey()" in js
    assert "key === _highlightKey" in js
    key = js[js.index("function browseHighlightKey()"):]
    key = key[:key.index("\n  }")]
    # Drilling changes the coordinate system, so a cached world-space highlight must not
    # survive into a local layout.
    assert "drilling ? 'local' : 'world'" in key


def test_box_select_has_one_destination():
    """It used to build the 8-card stack, render the panel and rebuild the plot
    highlight — then, for a big box, throw all of it away and do it again as browse."""
    js = _js()
    start = js.index("on('plotly_selected'")
    handler = js[start:start + 2000]
    assert handler.count("updateViewerPanel();") <= 1, "box-select renders the panel twice"
    assert "return;" in handler.split("enterBrowse")[1][:200], (
        "the browse path must return rather than falling through to the 8-card path"
    )


# ── Similarity is not the displayed map ─────────────────────────────────


def _mana_map_src():
    return (VIZ_DIR / "js" / "mana-map.js").read_text(encoding="utf-8")


def test_similarity_does_not_follow_the_displayed_map():
    """The defect that made Find Similar look broken.

    `loadEmbeddings` used to read `MAP_CONFIGS[currentMap].embeddings`, so on the
    colour/type map "similar" meant "same colour and type" — measured, that space
    uses 3.05 of its 128 dimensions and scores 0.044 recall@10 against known
    functional equivalents. The projection is a picture; similarity is a question,
    and they want different spaces.

    Structural, so a source assertion is the right tool: the guarantee is that this
    lookup is *absent*, which no amount of clicking would reveal until the numbers
    were already wrong.
    """
    src = _mana_map_src()
    assert "SIMILARITY_EMBEDDINGS" in src
    assert "embeddingsCache[currentMap]" not in src, (
        "similarity is keyed on the displayed map again"
    )


def test_only_one_knn_implementation_remains():
    """`findSimilarCards` hand-rolled a second full scan for years while a comment on
    `nearestTo` claimed the two had been consolidated. They had not — and the two
    disagreed on filtering."""
    src = _mana_map_src()
    assert "function cosineSimilarity" not in src, "the duplicate scorer is back"
    assert src.count("nearestTo(") >= 2, "callers should route through nearestTo"


def test_nearest_excludes_duplicate_names():
    """cards.csv carries 51 duplicate names, so self-exclusion alone let a card return
    its own twin at cosine 1.0 — true, and useless."""
    src = _mana_map_src()
    assert "allData[j].n === selfName" in src
