"""Behavioural tests — the ones that drive a real browser and look at what rendered.

WHY THIS FILE EXISTS. Every other viz test is `Path.read_text()` plus substring matching.
They assert that certain strings appear in certain files. On 2026-07-30 a perf commit
deleted a variable declaration and left the property that referenced it, so
`drill.js:getOverlayTraces()` threw `ReferenceError: text is not defined` on every render
while drilling. Drill mode was completely dead: no points, no breadcrumb.

**All 13 tests in `test_viz_drill.py` passed.**

The strings were all still there. That is the failure mode of source-assertion tests, and
it is why these exist. The gate on this file is concrete: `test_drill_renders_its_layout`
must FAIL against that broken revision and pass after the one-line fix. If a future test
here cannot fail for a real reason, it is decoration.

Run: `.venv/bin/python -m pytest tests/test_viz_behaviour.py`
Needs: `pip install playwright && playwright install chromium` (skips cleanly without).
"""

from __future__ import annotations

import json

import pytest

# F401: `page`, `discover_page` and `canvas_page` are pytest FIXTURES. They are
# used by name in test signatures, never called here, so every unused-import
# check reports all three — and removing them takes the whole browser suite
# down with an unrelated-looking "fixture not found".
from conftest_viz import await_projection  # noqa: F401
from conftest_viz import (  # noqa: F401
    BOOT_TIMEOUT_MS, canvas_page, corpus_count, discover_page, page, still_page,
)

pytestmark = pytest.mark.browser

# Region labels are placed by a RENDER PASS, not by boot, so "the page is ready"
# does not imply "a label exists". Four tests reached for one anyway — two clicked
# it (`null.click()`, which reads as a broken selector rather than as a race) and
# two measured the list. All four were green for months and all four fell over the
# first time the suite got busier, which is the whole signature of a latent flake:
# it measures the machine, and the machine got slower.
#
# Paste this at the top of an `async` evaluate body that touches `.map-label`. It
# is inlined at each site rather than spliced in from a constant, because these
# bodies are plain triple-quoted strings and a `" + NAME + "` in the middle of one
# lands in the page as literal JavaScript text:
#
#     for (let i = 0; i < 200 && !document.querySelector('.map-label'); i++) {
#         await new Promise(r => setTimeout(r, 50));
#     }
#
# It resolves as soon as one label is placed and gives up after ~10 s, so a
# genuine "no labels were ever placed" still fails on its own assertion instead of
# hanging until the test timeout.


# ── Boot ────────────────────────────────────────────────────────────────


def test_map_boots_clean(page, corpus_count):
    assert page.evaluate("MM.allData.length") == corpus_count
    traces = page.evaluate("MM.mapRenderer.layers.length")
    assert traces >= 6, "base scatter did not render"
    assert page.js_errors == [], f"console/page errors at boot: {page.js_errors}"


def test_canvas_is_not_blurry_on_retina(page):
    """Plotly handles devicePixelRatio for us today. When the renderer is replaced this
    becomes the assertion that keeps text and points crisp — see the migration plan."""
    ratio = page.evaluate("window.devicePixelRatio")
    assert ratio >= 1
    # The plot fills its flex cell; a zero-size plot renders nothing and looks like a bug.
    box = page.evaluate(
        "(() => { const r = document.getElementById('plot').getBoundingClientRect();"
        " return {w: Math.round(r.width), h: Math.round(r.height)}; })()")
    assert box["w"] > 400 and box["h"] > 300, f"plot has no area: {box}"


# ── Drill: the test that would have caught the regression ───────────────


def test_drill_renders_its_layout(page):
    """The regression gate.

    Against the broken revision: `drillTraces` is 0, the breadcrumb is hidden, and a
    ReferenceError lands in `page.js_errors`. All three assertions fail.
    """
    result = page.evaluate("""async () => {
        const rd = await MM.getRegionData();
        const reg = rd.regions.find(r => r.level === 1 && r.count > 100);
        await Drill.enterRegion(reg.id);
        await new Promise(r => setTimeout(r, 2500));
        const gd = document.getElementById('plot');
        const dt = MM.mapRenderer.layers.find(t => t._isDrill);
        let moved = 0;
        if (dt) {
            for (let i = 0; i < dt.customdata.length; i++) {
                const w = MM.allData[dt.customdata[i]];
                if (Math.hypot(dt.x[i] - w.x, dt.y[i] - w.y) > 0.05) moved++;
            }
        }
        return {
            region: reg.label,
            expected: reg.count,
            drillTraces: MM.mapRenderer.layers.filter(t => t._isDrill).length,
            points: dt ? dt.x.length : 0,
            movedFraction: dt ? moved / dt.customdata.length : 0,
            barVisible: document.getElementById('drillBar').style.display !== 'none',
            active: Drill.isActive(),
        };
    }""")

    assert page.js_errors == [], f"drill threw: {page.js_errors}"
    assert result["drillTraces"] == 1, "the drill layout did not render"
    assert result["points"] == result["expected"], (
        f"drilled {result['points']} of {result['expected']} cards in {result['region']}")
    assert result["movedFraction"] > 0.9, (
        "points did not move — the layout was never recomputed from the embeddings")
    assert result["barVisible"], "no breadcrumb, so nothing says the coordinates are local"


def test_drill_back_restores_the_world(page):
    result = page.evaluate("""async () => {
        const rd = await MM.getRegionData();
        const reg = rd.regions.find(r => r.level === 1 && r.count > 100);
        await Drill.enterRegion(reg.id);
        await new Promise(r => setTimeout(r, 2000));
        Drill.back();
        await new Promise(r => setTimeout(r, 900));
        const layers = MM.mapRenderer.layers;
        return {
            active: Drill.isActive(),
            drillTraces: layers.filter(t => t._isDrill).length,
            hiddenBaseTraces: layers.filter(t => t.visible === false).length,
            // Region labels are DOM buttons now, not layout annotations.
            annotations: document.querySelectorAll('.map-label').length,
            barHidden: document.getElementById('drillBar').style.display === 'none',
        };
    }""")
    assert page.js_errors == []
    assert not result["active"] and result["drillTraces"] == 0
    assert result["hiddenBaseTraces"] == 0, "the world stayed hidden"
    assert result["annotations"] > 0, "region labels did not come back"
    assert result["barHidden"]


# ── Selection ───────────────────────────────────────────────────────────


def test_small_selection_uses_the_accordion(page):
    result = page.evaluate("""async () => {
        ['Sol Ring', 'Lightning Bolt', 'Counterspell', 'Grave Pact'].forEach(n => MM.selectByName(n));
        await new Promise(r => setTimeout(r, 600));
        const inner = document.getElementById('detailInner');
        const active = inner.querySelector('.acc-row.active');
        const ir = inner.getBoundingClientRect(), rr = active.getBoundingClientRect();
        return {
            rows: inner.querySelectorAll('.acc-row').length,
            bodies: inner.querySelectorAll('.acc-body').length,
            rowTopInPanel: Math.round(rr.top - ir.top),
            hasArrows: !!inner.querySelector('.viewer-arrow'),
        };
    }""")
    assert page.js_errors == []
    assert result["rows"] == 4
    assert result["bodies"] == 1, "more than one card body is open"
    assert 0 <= result["rowTopInPanel"] <= 140, (
        f"the open row is not near the top of the panel ({result['rowTopInPanel']}px)")
    assert result["hasArrows"]


def test_big_selection_enters_browse_and_orders_by_embedding(page):
    """Browse must keep the WHOLE set and order it, not truncate to an arbitrary 8."""
    result = page.evaluate("""async () => {
        const rd = await MM.getRegionData();
        const reg = rd.regions.find(r => r.level === 0 && r.count > 800);
        const cid = parseInt(reg.id.split('_')[1], 10);
        const rows = [];
        rd.membership.l0.forEach((v, i) => { if (v === cid) rows.push(i); });
        await MM.enterBrowse(rows, reg.label);
        await new Promise(r => setTimeout(r, 1200));
        const gd = document.getElementById('plot');
        const bs = MM.browseSet;
        return {
            expected: rows.length,
            held: bs.indices.length,
            marker: MM.mapRenderer.layers.filter(t => t._isBrowseCurrent).length,
            setTrace: MM.mapRenderer.layers.filter(t => t.name && t.name.startsWith('Selection')).length,
            noAccordion: !document.querySelector('.acc-row'),
            first: MM.allData[bs.indices[0]].n,
            last: MM.allData[bs.indices[bs.indices.length - 1]].n,
        };
    }""")
    assert page.js_errors == []
    assert result["held"] == result["expected"], "browse truncated the selection"
    assert result["marker"] == 1 and result["setTrace"] == 1
    assert result["noAccordion"], "browse must not render the list"
    assert result["first"] != result["last"]


def test_browse_cycling_moves_the_marker(page):
    result = page.evaluate("""async () => {
        const rows = []; for (let i = 0; i < 400; i++) rows.push(i * 7 % 34322);
        await MM.enterBrowse(rows, 'T');
        await new Promise(r => setTimeout(r, 900));
        const gd = document.getElementById('plot');
        const snap = () => { const t = MM.mapRenderer.layers.find(x => x._isBrowseCurrent);
                             return {x: t.x[0], y: t.y[0], cd: t.customdata[0]}; };
        const a = snap(); MM.cycleNext(); await new Promise(r => setTimeout(r, 300));
        const b = snap(); MM.cyclePrev(); await new Promise(r => setTimeout(r, 300));
        const c = snap();
        return {a, b, c, count: document.querySelector('.viewer-count').textContent};
    }""")
    assert page.js_errors == []
    assert result["a"]["cd"] != result["b"]["cd"], "next did not change the card"
    assert result["c"]["cd"] == result["a"]["cd"], "prev did not return to the start"


# ── The camera bug ──────────────────────────────────────────────────────
#
# `tests/test_viz_camera.py` used to guard this with three source assertions, and was
# retired with Plotly rather than ported. Its subject no longer exists: `Plotly.react`
# replaced layout wholesale, so a layout with no explicit axis range silently reset the
# viewport to autorange, and `render()` had to read the live range and write it back
# (`keepX`/`keepY`) on every pass. Filtering and zooming were mutually destructive
# without it, and it was invisible because nothing failed — measured in a browser: zoom
# to a span of 20.5, call `MM.render()`, read 116.6. The canvas never rebuilds a layout;
# `transform` is the only thing that moves, so the hazard left with the renderer.
#
# What survives is the invariant, and it was always better tested here than by grep:
# the camera must survive a re-render, EXCEPT when the coordinates themselves change.


def test_filtering_does_not_reset_the_zoom(page):
    """Before the fix: zoom to a span of 20.5, toggle a filter, get 116.6."""
    result = page.evaluate("""async () => {
        const span = () => { const c = MM.mapRenderer.getCamera();
                             return Math.abs(c.x[1] - c.x[0]); };
        MM.mapRenderer.setCamera({x: [-5, 5], y: [-5, 5]});
        await new Promise(r => setTimeout(r, 300));
        const zoomed = span();
        document.querySelectorAll('#toggles button')[1].click();
        await new Promise(r => setTimeout(r, 700));
        const afterToggle = span();
        const s = document.getElementById('search');
        s.value = 'goblin'; s.dispatchEvent(new Event('input'));
        await new Promise(r => setTimeout(r, 800));
        return {zoomed, afterToggle, afterSearch: span()};
    }""")
    assert page.js_errors == []
    assert abs(result["afterToggle"] - result["zoomed"]) < 0.5, "a filter toggle reset the camera"
    assert abs(result["afterSearch"] - result["zoomed"]) < 0.5, "a search reset the camera"


def test_map_switch_refits_the_camera(page):
    """The one case that SHOULD autorange — the coordinates themselves change."""
    result = page.evaluate("""async () => {
        const span = () => { const c = MM.mapRenderer.getCamera();
                             return Math.abs(c.x[1] - c.x[0]); };
        // Start on the OTHER map: the app boots on Abilities, so switching to Abilities
        // is a no-op that returns before doing anything and refits nothing.
        const ms = document.getElementById('mapSelect');
        ms.value = 'default'; ms.dispatchEvent(new Event('change'));
        await new Promise(r => setTimeout(r, 8000));
        MM.mapRenderer.setCamera({x: [-5, 5], y: [-5, 5]});
        await new Promise(r => setTimeout(r, 300));
        const zoomed = span();
        ms.value = 'ability'; ms.dispatchEvent(new Event('change'));
        await new Promise(r => setTimeout(r, 12000));
        return {zoomed, after: span(), status: document.getElementById('status').textContent};
    }""")
    assert page.js_errors == []
    assert result["after"] > result["zoomed"] * 2, "the map switch kept a stale camera"
    assert "Abilities" in result["status"]


# ── Modes ───────────────────────────────────────────────────────────────


def test_the_map_view_lights_a_deck(page):
    """The old Deck Lens, now a view inside Build.

    Build opens on the GRAPH, so the atlas overlay is not drawn until you ask for it —
    hence the explicit `setView('map')`. That is the behaviour change, not a regression:
    the map answers "where does this sit" and the graph answers "what is next to what".
    """
    result = page.evaluate("""async () => {
        document.getElementById('modeSelect').value = 'build';
        MM.setMode('build');
        await new Promise(r => setTimeout(r, 4000));
        await Build.select('edgar-vampires');
        await new Promise(r => setTimeout(r, 6000));
        Build.setView('map');
        await new Promise(r => setTimeout(r, 2500));
        const gd = document.getElementById('plot');
        const base = MM.mapRenderer.layers.find(t => !t._isDeckOverlay && t.marker);
        return {
            overlays: MM.mapRenderer.layers.filter(t => t._isDeckOverlay).length,
            commander: MM.mapRenderer.layers.filter(t => t.name === 'Commander').length,
            dimScalar: typeof base.marker.opacity,
            deckName: (document.querySelector('.lens-title') || {}).textContent,
            status: document.getElementById('status').textContent,
        };
    }""")
    assert page.js_errors == []
    assert result["overlays"] >= 10, "the deck did not light up"
    assert result["commander"] == 1
    assert result["dimScalar"] == "number", "Deck Lens regressed to per-point opacity"
    assert result["deckName"] == "EDGAR MARKOV"


def test_mode_switches_are_exclusive(page):
    result = page.evaluate("""async () => {
        const out = {};
        document.getElementById('modeSelect').value = 'build'; MM.setMode('build');
        await new Promise(r => setTimeout(r, 3000));
        out.deckPanelOpen = document.getElementById('deckPanel').classList.contains('open');
        document.getElementById('modeSelect').value = 'explore'; MM.setMode('explore');
        await new Promise(r => setTimeout(r, 900));
        out.afterExplore = document.getElementById('deckPanel').classList.contains('open');
        out.overlays = MM.mapRenderer.layers
            .filter(t => t._isDeckOverlay).length;
        return out;
    }""")
    assert page.js_errors == []
    assert result["deckPanelOpen"]
    assert not result["afterExplore"], "the deck panel survived a mode switch"
    assert result["overlays"] == 0, "deck overlays survived a mode switch"


# ── Perf baseline ───────────────────────────────────────────────────────


def test_render_stays_under_budget(page):
    """A ceiling, not a benchmark — generous enough for CI, tight enough to catch the
    class of regression this project keeps hitting: work done per render that nothing
    looks at. Measured on a warm dev machine: ~30 ms empty, ~36 ms with a big browse."""
    result = page.evaluate("""async () => {
        const t = (f, n) => { const xs = [];
            for (let i = 0; i < n; i++) { const a = performance.now(); f(); xs.push(performance.now() - a); }
            xs.sort((p, q) => p - q); return xs[(n / 2) | 0]; };
        for (let i = 0; i < 3; i++) MM.render();          // warm
        const empty = t(() => MM.render(), 7);
        const rows = []; for (let i = 0; i < 8000; i++) rows.push(i * 3 % 34322);
        await MM.enterBrowse(rows, 'Perf');
        await new Promise(r => setTimeout(r, 900));
        return {empty, browse: t(() => MM.render(), 7), cycle: t(() => MM.cycleNext(), 7)};
    }""")
    assert page.js_errors == []
    assert result["empty"] < 200, f"empty render {result['empty']:.0f}ms"
    assert result["browse"] < 300, f"render with 8k browse {result['browse']:.0f}ms"
    assert result["cycle"] < 150, f"arrow press {result['cycle']:.0f}ms"


def test_render_draws_everything_in_one_pass(page):
    """`render()` must draw everything, including the selection highlight, in one pass.

    When `render()` left the highlight to `updateSelectionHighlight()`, every pan and
    filter did an extra add/delete of the whole selection — a full rebuild of a
    15,000-point trace on each one.

    The invariant outlived the renderer; only the call being counted changed. This
    counted `Plotly.react` / `addTraces` / `deleteTraces` / `restyle`; the canvas
    equivalents are `setLayers` (draw everything) and `updateLayerBy` (move one layer
    without touching the other 34,322), which is precisely the distinction the original
    was protecting.
    """
    result = page.evaluate("""async () => {
        const rows = []; for (let i = 0; i < 5000; i++) rows.push(i * 5 % 34322);
        await MM.enterBrowse(rows, 'Calls');
        await new Promise(r => setTimeout(r, 900));
        const counts = {setLayers: 0, updateLayerBy: 0};
        const r = MM.mapRenderer, orig = {};
        for (const k of ['setLayers', 'updateLayerBy']) {
            orig[k] = r[k].bind(r);
            r[k] = function (...a) { counts[k]++; return orig[k](...a); };
        }
        MM.render();
        const perRender = {...counts};
        counts.setLayers = counts.updateLayerBy = 0;
        MM.cycleNext();
        const perCycle = {...counts};
        for (const k of ['setLayers', 'updateLayerBy']) r[k] = orig[k];
        return {perRender, perCycle};
    }""")
    assert page.js_errors == []
    assert result["perRender"] == {"setLayers": 1, "updateLayerBy": 0}, (
        f"render() is not a single full draw: {result['perRender']}")
    assert result["perCycle"] == {"setLayers": 0, "updateLayerBy": 1}, (
        f"an arrow press should move one layer, not redraw: {result['perCycle']}")


# ── The graph (Discover) ────────────────────────────────────────────────
#
# These were "The Walk" — a fifth mode that was Discover with different chrome (two
# behaviours and a status string, across four `chrome ===` reads in force.js). It was
# deleted; its panel's keepers moved into Discovery's. The engine under test is unchanged,
# so these tests keep their assertions and change only how they get to the graph.


def _walk(page, seed_js, settle=9000):
    return page.evaluate("""async ([seedJs, settle]) => {
        const rows = await (new Function('return (async () => {' + seedJs + '})()'))();
        document.getElementById('modeSelect').value = 'discover';
        MM.setMode('discover');
        await new Promise(r => setTimeout(r, 200));
        await Force.enter(rows, 'Test');
        await new Promise(r => setTimeout(r, settle));
        Force.freeze();
        const b = Force.bbox();
        return {
            seeded: rows.length,
            nodes: Force.nodeCount, links: Force.linkCount,
            bbox: {w: b.w, h: b.h},
            linkStats: Force.linkStats(),
            canvas: !!document.getElementById('forceCanvas'),
            plotHidden: document.getElementById('plot').classList.contains('force-mode'),
        };
    }""", [seed_js, settle])


DECK_SEED = """
    const deck = await (await fetch('../data/decks/edgar-vampires/cards.json')).json();
    const names = new Set(deck.cards.map(c => c.name));
    const rows = []; MM.allData.forEach((d, i) => { if (names.has(d.n)) rows.push(i); });
    return rows;
"""


def test_walk_builds_a_graph_that_spreads(page):
    r = _walk(page, DECK_SEED)
    assert page.js_errors == [], f"the walk threw: {page.js_errors}"
    assert r["canvas"] and r["plotHidden"]
    # NOT a hardcoded count. This seeds from edgar-vampires/cards.json, and the
    # deck is a real object that moves: the v1.0.0 paper check-in took it from 97
    # distinct names to 96 and turned this red for a change it was never about.
    # The property is that every seeded row becomes a node; the total is whatever
    # the deck happens to be. The floor guards the case that would make the rest
    # of the assertions vacuous — a seed that matched almost nothing.
    assert r["nodes"] == r["seeded"], (r["nodes"], r["seeded"])
    assert r["seeded"] >= 90, f"only {r['seeded']} of the deck matched the atlas"
    assert r["links"] > r["nodes"], "every card should carry links"
    # The layout must actually resolve. A collapsed graph reads as an empty canvas, which
    # is what happened when nodes were seeded at identical world positions — some regions
    # really are degenerate (the White Sorceries filament is 187 cards at 0.1 x 0.0).
    assert r["bbox"]["w"] > 50 and r["bbox"]["h"] > 50, f"graph collapsed: {r['bbox']}"


def test_link_length_is_the_embedding_distance(page):
    """Chord distance on a unit sphere is bounded by [0, 2]. Screen distance is not, so
    this fails immediately if the layout is ever fed 2-D positions instead."""
    r = _walk(page, DECK_SEED, settle=3000)
    st = r["linkStats"]
    assert st is not None and st["n"] > 0
    assert 0 <= st["min"] < st["max"] <= 2.0, f"link d out of chord range: {st}"
    assert st["mean"] < 1.0, f"a k-nearest graph should be mostly short links: {st}"


def test_branching_grows_the_graph_and_records_the_walk(page):
    """The walk is built from the deck's LIVE contents, never from card names.

    This test used to name Edgar Markov, Sorin and Exquisite Blood. Two of the
    three left the deck, so two of the three branches became no-ops and the graph
    stopped growing — `assert 103 > 103`, reported as a rendering regression when
    it was a decklist that had moved. `test_pilot_validate_stack` carries the same
    lesson about the same card: a test that hardcodes a decklist is testing the
    decklist, not the predicate.
    """
    r = page.evaluate("""async () => {
        const deck = await (await fetch('../data/decks/edgar-vampires/cards.json')).json();
        const names = new Set(deck.cards.map(c => c.name));
        const rows = []; MM.allData.forEach((d, i) => { if (names.has(d.n)) rows.push(i); });
        document.getElementById('modeSelect').value = 'discover'; MM.setMode('discover');
        await new Promise(r => setTimeout(r, 200));
        await Force.enter(rows, 'Test');
        await new Promise(r => setTimeout(r, 2500));
        const before = Force.nodeCount;
        const steps = [];
        // Three cards the deck really runs, spread across the list so their
        // neighbourhoods do not collapse onto each other.
        const walk = [rows[0], rows[Math.floor(rows.length / 2)], rows[rows.length - 1]];
        for (const i of walk) {
            Force.focusCard(i);
            await new Promise(r => setTimeout(r, 700));
            steps.push(Force.nodeCount);
        }
        // A card that is not on the graph must be a no-op, not a crash and not a
        // phantom trail entry.
        const absent = MM.allData.findIndex(d => d.n === 'Black Lotus');
        Force.focusCard(absent);
        await new Promise(r => setTimeout(r, 300));
        Force.freeze();
        return {before, steps, walk: walk.length,
                trail: Force.trailLength, afterAbsent: Force.nodeCount};
    }""")
    assert page.js_errors == []
    assert r["walk"] == 3, "the deck must supply three distinct cards to walk"
    assert r["steps"][0] == r["before"] + 6, "a branch should pull in BRANCH_K neighbours"
    assert r["steps"][2] > r["steps"][0], "the graph must keep growing as you walk"
    assert r["trail"] == 3, "each distinct card visited should be recorded on the trail"
    assert r["afterAbsent"] == r["steps"][2], (
        "focusing a card that is not on the graph must change nothing")


def test_leaving_the_walk_restores_the_map(page):
    r = page.evaluate("""async () => {
        const rows = []; for (let i = 0; i < 60; i++) rows.push(i * 37 % 34322);
        document.getElementById('modeSelect').value = 'discover'; MM.setMode('discover');
        await new Promise(r => setTimeout(r, 200));
        await Force.enter(rows, 'Test');
        await new Promise(r => setTimeout(r, 1500));
        document.getElementById('modeSelect').value = 'explore'; MM.setMode('explore');
        await new Promise(r => setTimeout(r, 900));
        const gd = document.getElementById('plot');
        const cv = document.getElementById('forceCanvas');
        return {
            active: Force.isActive(),
            forceMode: gd.classList.contains('force-mode'),
            // Hidden by the CSS class, NOT by an inline style. Asserting the inline
            // style was asserting the bug: it survived re-entry and left the canvas 0x0.
            canvasComputedHidden: getComputedStyle(cv).display === 'none',
            canvasInline: cv.style.display || '',
            plotTraces: MM.mapRenderer.layers.length,
        };
    }""")
    assert page.js_errors == []
    assert not r["active"] and not r["forceMode"]
    assert r["canvasComputedHidden"], "the walk canvas is still visible over the map"
    assert r["canvasInline"] == "", "visibility must come from the class, not an inline style"
    assert r["plotTraces"] >= 6, "the map did not come back"


def test_the_panel_always_offers_somewhere_to_go(page):
    """Somewhere to start must always be one click away, and an empty graph must never
    render as a "0 CARDS / 0 LINKS" scoreboard.

    This used to be The Walk's `renderEmptyState`, which was the only place a REGION could
    become a graph — `Drill.enterRegion` re-embeds a region into the atlas and never seeds
    the graph. Deleting the mode without porting the region list would have deleted the
    capability, so the list now lives in Discovery's panel and this test guards it there.
    """
    r = page.evaluate("""async () => {
        document.getElementById('modeSelect').value = 'discover';
        MM.setMode('discover');
        await new Promise(r => setTimeout(r, 3500));
        const el = document.getElementById('deckInner');
        return {
            decks: el.querySelectorAll('#dcDeck option').length - 1,   // minus the prompt
            regions: el.querySelectorAll('[onclick^="Force.walkRegion"]').length,
            saysZeroCards: (el.innerText || '').indexOf('0 CARDS') !== -1,
        };
    }""")
    # Derived from the manifest, never hardcoded: this asserted a literal 7 and broke
    # the moment an eighth deck was built. A count of a growing artifact is a
    # maintenance tax that teaches nothing — what matters is that the picker offers
    # EVERY loadable deck, published or not.
    import json
    from manamap.config import DECKS_DIR
    expected = len(json.loads((DECKS_DIR / "index.json").read_text())["decks"])

    assert page.js_errors == []
    assert r["decks"] == expected, "every loadable deck should be one click from a graph"
    assert r["regions"] > 0, "regions come from the HDBSCAN membership"
    assert not r["saysZeroCards"], "an empty graph rendered the dead-end scoreboard"


def test_walking_a_deck_from_the_empty_state(page):
    r = page.evaluate("""async () => {
        document.getElementById('modeSelect').value = 'discover';
        MM.setMode('discover');
        await new Promise(r => setTimeout(r, 2500));
        await Discovery.loadDeck('goblin-storm');
        await new Promise(r => setTimeout(r, 9000));
        Force.freeze();
        const b = Force.bbox();
        return {nodes: Force.nodeCount, links: Force.linkCount, w: b.w, h: b.h};
    }""")
    assert page.js_errors == []
    assert r["nodes"] > 50 and r["links"] > r["nodes"]
    assert r["w"] > 50 and r["h"] > 50, "the deck's graph collapsed"


def test_walk_survives_a_round_trip_through_explore(page):
    """Leaving for the map and coming back must not lose the graph.

    Two separate faults met here. `exit()` set an inline `display: none` on the canvas
    that nothing ever cleared, so re-entry rebuilt the graph correctly into a 0x0 hidden
    element — 79 nodes, 156 links, the right status line, and a blank screen. And the
    graph was discarded on exit, so a walk you spent minutes growing had to be rebuilt
    just because you glanced at the map.
    """
    r = page.evaluate("""async () => {
        const setMode = m => { document.getElementById('modeSelect').value = m; MM.setMode(m); };
        const cv = () => document.getElementById('forceCanvas');
        const ink = () => {
            const c = cv();
            if (!c || !c.width) return 0;
            const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
            let l = 0, n = 0;
            for (let i = 3; i < d.length; i += 4 * 30) { n++; if (d[i] > 20) l++; }
            return 100 * l / n;
        };

        setMode('discover');
        await new Promise(r => setTimeout(r, 2500));
        await Discovery.loadDeck('goblin-storm');
        await new Promise(r => setTimeout(r, 9000));
        const i = MM.allData.findIndex(d => d.n === 'Past in Flames');
        if (i >= 0) Force.focusCard(i);
        await new Promise(r => setTimeout(r, 2000));
        const before = {nodes: Force.nodeCount, trail: Force.trailLength,
                        w: cv().clientWidth, ink: ink()};

        setMode('explore');
        await new Promise(r => setTimeout(r, 1200));
        const explore = {traces: MM.mapRenderer.layers.length,
                         inlineDisplay: cv().style.display || ''};

        setMode('discover');
        await new Promise(r => setTimeout(r, 2500));
        const after = {nodes: Force.nodeCount, trail: Force.trailLength,
                       w: cv().clientWidth, ink: ink()};
        return {before, explore, after};
    }""")
    assert page.js_errors == []
    assert r["explore"]["traces"] >= 6, "the map did not come back"
    assert r["explore"]["inlineDisplay"] == "", (
        "exit() must not set an inline display — the CSS class owns visibility, and an "
        "inline hide survives re-entry")
    assert r["after"]["nodes"] == r["before"]["nodes"], "the graph was discarded on exit"
    assert r["after"]["trail"] == r["before"]["trail"], "the trail was discarded on exit"
    assert r["after"]["w"] > 400, f"canvas came back at {r['after']['w']}px wide"
    assert r["after"]["ink"] > 0.5, "the graph is not being drawn after re-entry"


def test_the_walk_panel_is_actually_on_screen(page):
    """Rendering the menu is not the same as being able to click it.

    The earlier tests asserted the buttons existed in the DOM and clicked them
    programmatically — which passes even when the panel is collapsed to 1px and the
    buttons are laid out past the right edge of the window, unreachable by a real mouse.
    This asserts geometry and hit-testing instead.
    """
    r = page.evaluate("""async () => {
        document.getElementById('modeSelect').value = 'discover';
        MM.setMode('discover');
        await new Promise(r => setTimeout(r, 2500));
        const panel = document.getElementById('deckPanel');
        // The region seeds live behind a disclosure now — they are a way of
        // starting somewhere ELSE, and nine equal-weight buttons above the card
        // were most of why the landing read as heavy. This test is about
        // GEOMETRY and hit-testing (are the buttons laid out off-screen?), not
        // about whether a disclosure is open, so it opens the one it needs.
        for (const d of document.querySelectorAll('#deckInner details')) d.open = true;
        await new Promise(r => setTimeout(r, 120));
        const btn = document.querySelector('[onclick^="Force.walkRegion"]');
        // …and scrolls it into view. `elementFromPoint` only sees the VIEWPORT,
        // and the region list sits well down a scrolling panel now that the
        // card leads. The invariant here is that the buttons are not laid out
        // past the right edge — horizontal geometry — not that everything in a
        // scrolling panel is visible at once, which no panel can promise.
        btn.scrollIntoView({block: 'center'});
        await new Promise(r => setTimeout(r, 120));
        const b = btn.getBoundingClientRect();
        const cx = Math.round(b.left + b.width / 2), cy = Math.round(b.top + b.height / 2);
        const hit = document.elementFromPoint(cx, cy);
        return {
            panelWidth: Math.round(panel.getBoundingClientRect().width),
            buttonRight: Math.round(b.right),
            viewportWidth: document.documentElement.clientWidth,
            reachable: !!hit && (hit === btn || btn.contains(hit)),
        };
    }""")
    assert page.js_errors == []
    assert r["panelWidth"] > 300, f"the panel is collapsed to {r['panelWidth']}px"
    assert r["buttonRight"] <= r["viewportWidth"], "the buttons are laid out off-screen"
    assert r["reachable"], "a real click at the button's centre does not land on it"


def test_a_walk_in_progress_can_be_restarted(page):
    """Restoring the graph on re-entry made the graph a one-way door: the deck menu used
    to appear only when the graph was empty, so the first set you picked was the only set
    you could ever pick. `Start over` is the way back — and the deck picker is now a
    persistent `<select>`, so loading a second deck no longer needs an empty graph."""
    r = page.evaluate("""async () => {
        const setMode = m => { document.getElementById('modeSelect').value = m; MM.setMode(m); };
        setMode('discover');
        await new Promise(r => setTimeout(r, 2500));
        await Discovery.loadDeck('goblin-storm');
        await new Promise(r => setTimeout(r, 6000));
        const walking = {nodes: Force.nodeCount,
                         hasStartOver: !!document.querySelector('[onclick^="Discovery.newGraph"]')};
        document.querySelector('[onclick^="Discovery.newGraph"]').click();
        await new Promise(r => setTimeout(r, 2500));
        const reset = {nodes: Force.nodeCount};
        await Discovery.loadDeck('heliod');
        await new Promise(r => setTimeout(r, 6000));
        return {walking, reset, switched: Force.nodeCount};
    }""")
    assert page.js_errors == []
    assert r["walking"]["nodes"] > 50
    assert r["walking"]["hasStartOver"], "no way back from a graph in progress"
    # `Start over` reseeds on a fresh landing card rather than leaving an empty canvas,
    # because the deck and region pickers are always in the panel now — there is no empty
    # state left to return to.
    assert r["reset"]["nodes"] == 1, f"Start over left {r['reset']['nodes']} nodes"
    assert r["switched"] > 50 and r["switched"] != r["walking"]["nodes"], "could not pick a different set"


# ── Navigation: arrows, hover, and the card in the graph ────────────────
#
# Every test here presses REAL KEYS. The previous browse test called `MM.cycleNext()`
# directly, which is why it never noticed that the arrow keys were dead in browse mode:
# the handler bailed on `selectedCards.length === 0` and `enterBrowse` empties that array.
# Only the on-screen buttons worked, while the panel's own hint said "← → browse".


def _key(page, name):
    page.evaluate("k => document.dispatchEvent("
                  "new KeyboardEvent('keydown', {key: k, bubbles: true}))", name)


def test_one_card_plus_arrow_walks_its_neighbourhood(page):
    """Selecting a card and pressing → was a no-op: `cycleSelection` returned early below
    two selected cards. It now seeds the card's k nearest and steps into them."""
    page.evaluate("MM.selectByName('Zada, Hedron Grinder')")
    page.wait_for_timeout(700)
    assert page.evaluate("!!MM.browseSet") is False, "selection should not start as a browse"

    _key(page, "ArrowRight")
    # Seeding the neighbourhood awaits the embedding matrix, so this is a real
    # async wait rather than a repaint — but it is still a CONDITION. `pos > 0`
    # is the step itself; waiting only for `browseSet` to exist would race the
    # arrow that moves into it.
    page.wait_for_function(
        "() => MM.browseSet && MM.browseSet.indices"
        "      && MM.browseSet.indices.length > 1 && MM.browseSet.pos > 0",
        timeout=30_000)

    r = page.evaluate("""async () => {
        const bs = MM.browseSet;
        // Recompute the true nearest independently, so this asserts the ordering is the
        // model's and not just self-consistent.
        const emb = await MM.getEmbeddings(), dim = MM.EMBED_DIM;
        const a = bs.anchor, oa = a * dim;
        let bestI = -1, best = -2;
        for (let j = 0; j < MM.allData.length; j++) {
            if (j === a) continue;
            const oj = j * dim; let d = 0;
            for (let i = 0; i < dim; i++) d += emb[oa + i] * emb[oj + i];
            if (d > best) { best = d; bestI = j; }
        }
        return {anchor: MM.allData[bs.anchor].n, pos: bs.pos, n: bs.indices.length,
                showing: MM.allData[bs.indices[bs.pos]].n,
                trueNearest: MM.allData[bestI].n};
    }""")
    assert page.js_errors == []
    assert r["anchor"] == "Zada, Hedron Grinder", "the anchor should be the card you picked"
    assert r["pos"] == 1, "the first press should already move, not just seed"
    assert r["showing"] == r["trueNearest"], (
        f"stepped to {r['showing']}, nearest is {r['trueNearest']}")

    _key(page, "ArrowLeft")
    page.wait_for_timeout(600)
    back = page.evaluate("({pos: MM.browseSet.pos, "
                         "showing: MM.allData[MM.browseSet.indices[MM.browseSet.pos]].n})")
    assert back["pos"] == 0 and back["showing"] == "Zada, Hedron Grinder"


def test_enter_reanchors_the_neighbourhood(page):
    page.evaluate("MM.selectByName('Zada, Hedron Grinder')")
    page.wait_for_timeout(700)
    _key(page, "ArrowRight")
    # Seeding the neighbourhood awaits the embedding matrix, so this is a real
    # async wait rather than a repaint — but it is still a CONDITION. `pos > 0`
    # is the step itself; waiting only for `browseSet` to exist would race the
    # arrow that moves into it.
    page.wait_for_function(
        "() => MM.browseSet && MM.browseSet.indices"
        "      && MM.browseSet.indices.length > 1 && MM.browseSet.pos > 0",
        timeout=30_000)
    was = page.evaluate("MM.allData[MM.browseSet.indices[MM.browseSet.pos]].n")
    _key(page, "Enter")
    page.wait_for_timeout(3000)
    r = page.evaluate("({anchor: MM.allData[MM.browseSet.anchor].n, pos: MM.browseSet.pos})")
    assert page.js_errors == []
    assert r["anchor"] == was, "Enter should re-anchor to the card you walked to"
    assert r["pos"] == 0, "a fresh neighbourhood starts on its anchor"


def test_arrow_keys_drive_browse_mode(page):
    """The regression the suite was blind to — real keys, not MM.cycleNext()."""
    page.evaluate("""async () => {
        const rows = []; for (let i = 0; i < 300; i++) rows.push(i * 11 % 34322);
        await MM.enterBrowse(rows, 'T');
    }""")
    page.wait_for_timeout(1500)
    start = page.evaluate("MM.browseSet.pos")
    _key(page, "ArrowRight")
    page.wait_for_timeout(400)
    fwd = page.evaluate("MM.browseSet.pos")
    _key(page, "ArrowLeft")
    page.wait_for_timeout(400)
    back = page.evaluate("MM.browseSet.pos")
    assert page.js_errors == []
    assert fwd != start, "ArrowRight did nothing in browse mode"
    assert back == start, "ArrowLeft did not return"


def test_hover_shows_a_card_image_at_the_cursor(page):
    """Hovering a point shows that card's art at the cursor.

    This drove Plotly's `plotly_hover` through `_fullLayout.xaxis.d2p`; the canvas emits
    its own `hover` off a quadtree pick, and `dataToPixel` is the d2p equivalent. The
    subtle part is unchanged and is why the assertion reads the way it does: aiming at a
    card's pixel does NOT guarantee that card. The pick takes the nearest point within a
    radius over 34,322 of them, so a denser neighbour a pixel away wins — asking for Sol
    Ring's coordinates once returned Krark-Clan Ironworks. The invariant is that the popup
    shows whatever was actually hovered, not that it shows Sol Ring.
    """
    r = page.evaluate("""async () => {
        const host = document.getElementById('plot');
        const cv = MM.mapRenderer.canvas;
        const rect = host.getBoundingClientRect();
        const i = MM.allData.findIndex(d => d.n === 'Sol Ring'), d = MM.allData[i];
        const [px, py] = MM.mapRenderer.dataToPixel(d.x, d.y);
        let hoveredRow = null;
        MM.mapRenderer.on('hover', e => { if (hoveredRow === null) hoveredRow = e.row; });
        cv.dispatchEvent(new MouseEvent('mousemove',
            {bubbles: true, clientX: rect.left + px, clientY: rect.top + py}));
        // Wait for OUR OWN code to set the popup's src, not for a fixed 700 ms and
        // not for the image to arrive. The `src` attribute is written by the hover
        // handler after the 180 ms open delay, so it is a DOM condition we control;
        // under `-n 4` the fixed wait expired first and read an empty string, which
        // failed as "popup shows , but Sol Ring was hovered". Whether Scryfall then
        // serves the bytes is deliberately not asserted anywhere.
        let p = null, img = null;
        for (let i = 0; i < 100; i++) {
            p = document.querySelector('.card-popup');
            img = p && p.querySelector('img');
            if (p && p.style.display === 'block' && img && img.getAttribute('src')) break;
            await new Promise(r => setTimeout(r, 50));
        }
        const pr = p ? p.getBoundingClientRect() : null;
        return {
            visible: !!p && p.style.display === 'block',
            hoveredName: hoveredRow === null ? null : MM.allData[hoveredRow].n,
            src: img ? decodeURIComponent(img.getAttribute('src') || '') : '',
            pointerEvents: p ? getComputedStyle(p).pointerEvents : null,
            insideRight: pr ? pr.right <= rect.right + 1 : false,
            insideTop: pr ? pr.top >= rect.top - 1 : false,
        };
    }""")
    assert page.js_errors == []
    assert r["visible"], "no hover popup appeared"
    assert r["hoveredName"], "the canvas hover event never fired"
    assert r["hoveredName"] in r["src"], (
        f"popup shows {r['src'][:80]}, but {r['hoveredName']} was hovered")
    # Without this the popup sits under its own cursor and steals the hover from the point
    # that summoned it, flickering forever.
    assert r["pointerEvents"] == "none"
    assert r["insideRight"] and r["insideTop"], "popup escaped the plot area"

    page.evaluate("MM.hideCardPopup()")
    page.wait_for_timeout(300)
    assert page.evaluate("document.querySelector('.card-popup').style.display") == "none"


def test_the_walk_shows_the_card_it_pinned(page):
    """force mode hides #detailPanel, so the old "Open the card →" button pushed the card
    into an invisible element — nothing appeared, and it then popped open on leaving the
    Walk. The card now renders in the walk's own panel, from the same builder Explore uses.
    """
    r = page.evaluate("""async () => {
        // Every wait here was a fixed timer (2500 / 8000 / 2000 ms) and the whole
        // test measured the machine rather than the behaviour: under `-n 4` the
        // 97-card deck load ran past its 8 s and the panel was read before it had
        // been written. Wait for the condition, never for a clock.
        const until = async (fn, tries) => {
            for (let i = 0; i < (tries || 200); i++) {
                if (fn()) return true;
                await new Promise(r => setTimeout(r, 50));
            }
            return false;
        };
        document.getElementById('modeSelect').value = 'discover'; MM.setMode('discover');
        await until(() => MM.mode === 'discover' && Force.nodeCount > 0);
        await Discovery.loadDeck('goblin-storm');
        await until(() => Force.nodeCount > 50);
        const i = MM.allData.findIndex(d => d.n === 'Past in Flames');
        Force.focusCard(i);
        await until(() => document
            .querySelector('#deckInner .detail-card-image img'));
        const el = document.getElementById('deckInner');
        const img = el.querySelector('.detail-card-image img');
        const txt = (el.innerText || '');
        return {
            hasImage: !!img,
            src: img ? decodeURIComponent(img.getAttribute('src') || '') : '',
            hasOracle: txt.indexOf('ORACLE TEXT') !== -1,
            detailHidden: document.getElementById('detailPanel').style.display === 'none',
        };
    }""")
    assert page.js_errors == []
    assert r["hasImage"], "the walk panel shows no card image"
    assert "Past in Flames" in r["src"]
    assert r["hasOracle"], "the walk panel shows an image but none of the card's text"
    assert r["detailHidden"], "two panels open would cut the canvas in half"


# ── Capping a set without biasing it ────────────────────────────────────


def test_the_drill_button_reports_what_it_would_do(page, corpus_count):
    """It used to read only "Drill ⤓". With no filters that meant "re-map all 34,322
    cards", which the cap truncated to an arbitrary 2,000 — a cross-section of the entire
    universe that flew in from everywhere and settled into an incoherent pile."""
    r = page.evaluate("""async () => {
        const btn = document.getElementById('drillFiltered');
        const wide = {label: btn.textContent, disabled: btn.classList.contains('is-disabled')};
        const el = document.getElementById('status');
        // WAIT FOR THE STATUS TO CHANGE, do not sleep and hope. Under `-n 4` the
        // machine is running four Chromiums and a fixed 700 ms read the PREVIOUS
        // status ("34,890 cards shown"), so the refusal assertion failed while the
        // refusal had happened correctly — a timer measuring load, not behaviour.
        const was = el.textContent;
        btn.click();
        for (let i = 0; i < 60 && el.textContent === was; i++) {
            await new Promise(r => setTimeout(r, 50));
        }
        const refused = {active: Drill.isActive(), status: el.textContent};

        // Narrow to a supertype small enough to drill.
        document.querySelectorAll('#toggles button').forEach(b => {
            if (b.textContent !== 'Battle' && b.classList.contains('active')) b.click();
        });
        await new Promise(r => setTimeout(r, 1200));
        const narrow = {label: btn.textContent, disabled: btn.classList.contains('is-disabled')};
        btn.click();
        await new Promise(r => setTimeout(r, 4000));
        return {wide, refused, narrow, drilled: Drill.isActive()};
    }""")
    assert page.js_errors == []
    assert f"{corpus_count:,}" in r["wide"]["label"], "the button does not state its size"
    assert r["wide"]["disabled"], "the button looks live when it would refuse"
    assert not r["refused"]["active"], "drilling the whole map should be refused"
    assert "too many" in r["refused"]["status"], "refusal must explain itself"
    assert not r["narrow"]["disabled"] and "Drill 3" in r["narrow"]["label"]
    assert r["drilled"], "a drillable set did not drill"


def test_capping_samples_evenly_rather_than_taking_a_prefix(page):
    """`slice(0, N)` takes the first N rows in cards.csv order — Scryfall's export order —
    so a truncated drill of a 3,434-card region showed whichever cards happened to be
    exported first. The breadcrumb said "showing 2000 of 3434", honest about the count and
    silent about the bias."""
    r = page.evaluate("""() => {
        const src = [];
        for (let i = 0; i < 100; i++) src.push(i);
        const s = Drill.sampleEvenly(src, 10);
        return {
            sample: s,
            small: Drill.sampleEvenly([1, 2, 3], 10),
            spansTheSet: s[s.length - 1] > 80,
            isNotAPrefix: s[1] !== 1,
        };
    }""")
    assert page.js_errors == []
    assert len(r["sample"]) == 10
    assert r["isNotAPrefix"], "capping still takes a prefix"
    assert r["spansTheSet"], "the sample does not reach the end of the set"
    assert r["small"] == [1, 2, 3], "a set under the cap must pass through untouched"


# ── The canvas renderer (?renderer=canvas) ──────────────────────────────
#
# Phase 2 of the Plotly migration. Both renderers are live at once so they can be compared
# on identical data; these assert the canvas path draws, picks and stays within budget.


def _ink_strength(page):
    """Sampled ALPHA: how much is drawn, and how strongly.

    Alpha and not luminance, and this is not a detail. The canvas has a transparent
    background — the dark page shows through — so a point drawn at 0.09 alpha lands as
    (full colour, alpha 23), and `getImageData` returns colour UN-premultiplied. Read RGB
    and a dimmed map looks identical to a lit one; the dimming lives entirely in the alpha
    channel. Measured on a legend spotlight: RGB luminance moved 6.88 -> 6.85 while the
    composited image lost 63% of its bright pixels.

    `lit` counts anything drawn at all (the `_ink` question: did it draw); `solid` counts
    pixels drawn at close to full strength, which is what separates spotlit from muted.
    """
    return page.evaluate("""() => {
        const c = document.querySelector('.map-canvas');
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        let lit = 0, solid = 0, n = 0;
        for (let i = 3; i < d.length; i += 4 * 30) {
            n++;
            if (d[i] > 10) lit++;
            if (d[i] > 150) solid++;
        }
        return {lit: 100 * lit / n, solid: 100 * solid / n};
    }""")


def _ink(page):
    """Percentage of sampled pixels that are not transparent — "did anything draw"."""
    return page.evaluate("""() => {
        const c = document.querySelector('.map-canvas');
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        let lit = 0, n = 0;
        for (let i = 3; i < d.length; i += 4 * 30) { n++; if (d[i] > 10) lit++; }
        return 100 * lit / n;
    }""")


def test_canvas_renderer_draws_the_map_without_plotly(canvas_page, corpus_count):
    r = canvas_page.evaluate("""() => ({
        canvas: !!document.querySelector('.map-canvas'),
        plotlyDrew: !!document.querySelector('#plot .plot-container'),
        cards: MM.allData.length,
    })""")
    assert canvas_page.js_errors == [], f"canvas renderer threw: {canvas_page.js_errors}"
    assert r["canvas"] and not r["plotlyDrew"], "Plotly still drew under ?renderer=canvas"
    assert r["cards"] == corpus_count
    assert _ink(canvas_page) > 0.5, "the canvas is blank"


def test_canvas_redraws_when_the_filter_changes(canvas_page):
    """setLayers draws synchronously rather than through rAF — a filter is a discrete
    state change, and rAF does not fire in a hidden tab at all."""
    before = _ink(canvas_page)
    canvas_page.evaluate("document.querySelectorAll('#toggles button')[0].click()")
    # Wait for the CONDITION, not for a change and not for 900 ms. Waiting for
    # "the status differs from what it was" looks like the same thing and is not:
    # under `-n 4` the text captured before the click was itself a mid-boot
    # string, so the boot finishing satisfied the wait and the filtered read
    # happened before the filter had repainted. Both spellings of the expected
    # status are accepted for the same reason the assertion below accepts both.
    canvas_page.wait_for_function(
        "() => { const t = document.getElementById('status').textContent;"
        "        return t.includes('15,') || t.includes('cards shown'); }",
        # Generous because it is waiting on a real repaint of 34,890 points while
        # three other Chromiums share the machine. 10 s passed in isolation and
        # timed out one run in two under `-n 4`; the condition was right and the
        # budget was measuring the contention.
        timeout=30000)
    after = _ink(canvas_page)
    status = canvas_page.evaluate("document.getElementById('status').textContent")
    assert canvas_page.js_errors == []
    assert after < before, "turning off Creatures did not remove ink"
    assert "15," in status or "cards shown" in status


def test_canvas_click_selects_the_card_under_the_pointer(canvas_page):
    r = canvas_page.evaluate("""async () => {
        const c = document.querySelector('.map-canvas');
        const rect = c.getBoundingClientRect();
        // Aim at a real card via the renderer's own projection. Guessing a pixel depends
        // on the viewport size and lands on empty space as often as not.
        const i = MM.allData.findIndex(d => d.n === 'Sol Ring');
        const p = MM.mapRenderer.dataToPixel(MM.allData[i].x, MM.allData[i].y);
        c.dispatchEvent(new MouseEvent('click',
            {bubbles: true, clientX: rect.left + p[0], clientY: rect.top + p[1]}));
        await new Promise(r => setTimeout(r, 800));
        const rows = MM.selectedRows();
        return {n: rows.length, name: rows.length ? MM.allData[rows[0]].n : null,
                aimedAt: MM.allData[i].n};
    }""")
    assert canvas_page.js_errors == []
    assert r["n"] == 1, "a click on the canvas selected nothing"
    # Not necessarily the card aimed at: the quadtree returns the nearest within the pick
    # radius, and the map is dense. What must hold is that a click resolves to a real card.
    assert r["name"], "selected a row with no card behind it"


@pytest.mark.serial_only
def test_canvas_render_beats_the_plotly_budget(canvas_page):
    """Plotly's render measured ~30 ms on this data. The canvas path must not be slower —
    the quadtree is cached across renders because rebuilding it is 23.5 ms and setLayers
    runs on every filter and keystroke.

    **`serial_only`: a wall-clock budget cannot be asserted under contention.** With
    `-n 4` this machine is running four Chromiums and the median render measured
    41 ms — the renderer was not slower, the CPU was busier. Deselected by the
    `-m browser` default so the parallel run stays meaningful; run
    `pytest -m "browser and serial_only"` to check the budget itself.
    """
    ms = canvas_page.evaluate("""() => {
        const t = [];
        for (let i = 0; i < 9; i++) { const a = performance.now(); MM.render(); t.push(performance.now() - a); }
        t.sort((x, y) => x - y);
        return t[4];
    }""")
    assert canvas_page.js_errors == []
    assert ms < 30, f"canvas render {ms:.0f}ms — no faster than Plotly"


def test_both_renderers_agree_on_the_data(page, canvas_page):
    """The A/B the strangler exists for: same cards, same filtered counts, either path."""
    a = page.evaluate("({cards: MM.allData.length, "
                      "shown: MM.allData.filter(d => MM.passesFilters(d)).length})")
    b = canvas_page.evaluate("({cards: MM.allData.length, "
                             "shown: MM.allData.filter(d => MM.passesFilters(d)).length})")
    assert a == b, f"renderers disagree: plotly={a} canvas={b}"


# ── Phase 3: everything Plotly still owned ──────────────────────────────


def test_canvas_region_labels_are_real_dom(canvas_page):
    """Plotly drew these as layout annotations: a relayout to change one, no transition
    (the crossfade was an rgba() alpha rebuilt on a 150 ms debounce, so they popped), and
    no click target — clicking a region needed a 30-line hit-test against anchors."""
    r = canvas_page.evaluate("""async () => {
        for (let i = 0; i < 200 && !document.querySelector('.map-label'); i++) {
            await new Promise(r => setTimeout(r, 50));
        }
        const els = document.querySelectorAll('.map-label');
        const first = els[0];
        return {
            count: els.length,
            text: first ? first.textContent : null,
            isButton: first ? first.tagName === 'BUTTON' : false,
            // Read the authored rule, not the computed one: this fixture injects
            // `transition: none` globally, because Chrome throttles transitions in a
            // backgrounded page and every duration assertion would be a lie.
            authoredTransition: [...document.styleSheets]
                .flatMap(s => { try { return [...s.cssRules]; } catch (e) { return []; } })
                .some(r => r.selectorText === '.map-label' && r.style.transition),
            positioned: first ? first.style.transform.indexOf('translate') !== -1 : false,
        };
    }""")
    assert canvas_page.js_errors == []
    assert r["count"] > 5, "no region labels on the canvas renderer"
    assert r["isButton"], "a label must be a real click target, not painted text"
    assert r["authoredTransition"], "the L0/L1 crossfade should be a CSS transition"
    assert r["positioned"] and r["text"]


def test_canvas_has_its_own_legend(canvas_page):
    r = canvas_page.evaluate("""() => ({
        present: !!document.getElementById('mapLegend'),
        rows: document.querySelectorAll('.map-legend-row').length,
        text: (document.getElementById('mapLegend') || {}).innerText || '',
    })""")
    assert canvas_page.js_errors == []
    assert r["present"] and r["rows"] >= 6, "no legend under the canvas renderer"
    # Named after whatever the colour mode groups by. Pinned to Primary Color rather than
    # asserting the shipped default, so recolouring the map is not a legend regression.
    canvas_page.evaluate("""() => {
        const s = document.getElementById('colorBy');
        s.value = 'color'; s.dispatchEvent(new Event('change'));
    }""")
    canvas_page.wait_for_timeout(1200)
    text = canvas_page.evaluate("() => document.getElementById('mapLegend').innerText")
    assert "Multicolor" in text or "Colorless" in text


def test_canvas_draws_density_contours(canvas_page):
    """d3-contourDensity replaces histogram2dcontour. Plotly auto-binned to whatever
    extent it was handed, which is why its levels were never comparable between filters.

    Measured ZOOMED IN, where the atmospheric halo is switched off by design (see
    `auraLevel` in render/canvas.js). At the fitted view the halo already covers ~36% of
    the canvas and the contours draw over the same clusters, so switching them on moved
    total ink by 2.8 points and no ratio test could survive — not because the contours had
    stopped drawing, but because the measure had saturated. Zoomed in the halo is absent
    and the same toggle is a 7.6x change.
    """
    canvas_page.evaluate(
        """() => {
            const d = MM.allData[100];
            MM.mapRenderer.setCamera({x: [d.x - 6, d.x + 6], y: [d.y - 3.6, d.y + 3.6]});
        }"""
    )
    # VERIFY THE CAMERA TOOK, and set it again if a late render refit it.
    #
    # The fixture waits for `getCamera()` to answer, which means `baseFit` exists —
    # necessary but not sufficient. Selecting the map kicks off a render that ends in
    # a fit, and under `-n 4` that render can land AFTER the camera move above,
    # resetting it. The test then measured the fitted view, where the halo covers a
    # third of the canvas, and read 38.7 ink instead of 3.8.
    #
    # This is a bounded retry with a settle between attempts, which is NOT the thing
    # that failed earlier: putting `setCamera` inside a `wait_for_function` predicate
    # re-applied the zoom every animation frame and left the camera pinned at the fit.
    for _ in range(10):
        canvas_page.wait_for_timeout(400)
        span = canvas_page.evaluate(
            "() => { const c = MM.mapRenderer.getCamera();"
            "        return c ? Math.abs(c.x[1] - c.x[0]) : 1e9; }")
        if span < 20:
            break
        canvas_page.evaluate(
            """() => {
                const d = MM.allData[100];
                MM.mapRenderer.setCamera({x: [d.x - 6, d.x + 6], y: [d.y - 3.6, d.y + 3.6]});
            }""")
    else:
        pytest.fail(f"the camera never stayed zoomed in (span {span:.1f}) — "
                    f"a render keeps refitting it")

    canvas_page.wait_for_timeout(800)
    before = _ink(canvas_page)
    canvas_page.evaluate("document.getElementById('toggleContours').click()")
    canvas_page.wait_for_timeout(1800)
    during = _ink(canvas_page)
    canvas_page.evaluate("document.getElementById('toggleContours').click()")
    canvas_page.wait_for_timeout(1000)
    after = _ink(canvas_page)
    assert canvas_page.js_errors == []
    assert during > before * 3, f"Topo added almost no ink ({before:.1f} -> {during:.1f})"
    assert after < during / 2, "Topo did not turn off"


def test_canvas_box_select_uses_the_quadtree(canvas_page):
    """The 138 ms operation. Measured at 4.5 ms here over 22,161 caught points."""
    r = canvas_page.evaluate("""async () => {
        const c = document.querySelector('.map-canvas');
        const rect = c.getBoundingClientRect();
        document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Shift', bubbles: true}));
        await new Promise(r => setTimeout(r, 200));
        c.dispatchEvent(new MouseEvent('mousedown',
            {bubbles: true, clientX: rect.left + 400, clientY: rect.top + 200}));
        window.dispatchEvent(new MouseEvent('mousemove',
            {bubbles: true, clientX: rect.left + 800, clientY: rect.top + 450}));
        await new Promise(r => setTimeout(r, 200));
        window.dispatchEvent(new MouseEvent('mouseup',
            {bubbles: true, clientX: rect.left + 800, clientY: rect.top + 450}));
        await new Promise(r => setTimeout(r, 2500));
        document.dispatchEvent(new KeyboardEvent('keyup', {key: 'Shift', bubbles: true}));
        return {browse: !!MM.browseSet,
                held: MM.browseSet ? MM.browseSet.indices.length : MM.selectedRows().length};
    }""")
    assert canvas_page.js_errors == []
    assert r["held"] > 100, f"the marquee caught {r['held']} cards"
    assert r["browse"], "a large box-select should enter browse mode"


def test_the_map_view_and_drill_run_on_canvas(canvas_page):
    """`focusLine` frames a verified line on the atlas, so it needs the MAP view — Build
    opens on the graph. Drill pushes 90 frames through updateLayerBy rather than
    rebuilding every layer per frame."""
    r = canvas_page.evaluate("""async () => {
        document.getElementById('modeSelect').value = 'build'; MM.setMode('build');
        await new Promise(r => setTimeout(r, 4000));
        await Build.select('edgar-vampires');
        await new Promise(r => setTimeout(r, 6000));
        Build.setView('map');
        await new Promise(r => setTimeout(r, 2500));
        const lens = {deck: (document.querySelector('.lens-title') || {}).textContent};

        const span = () => { const c = MM.mapRenderer.getCamera();
                             return Math.abs(c.x[1] - c.x[0]); };
        const before = span();
        Build.focusLine(0);
        await new Promise(r => setTimeout(r, 1200));
        const zoomed = span();

        document.getElementById('modeSelect').value = 'explore'; MM.setMode('explore');
        await new Promise(r => setTimeout(r, 1200));
        const rd = await MM.getRegionData();
        const reg = rd.regions.find(r => r.level === 1 && r.count > 120 && r.count < 300);
        await Drill.enterRegion(reg.id);
        await new Promise(r => setTimeout(r, 4000));
        const drilled = {active: Drill.isActive(),
                         bar: document.getElementById('drillBar').style.display !== 'none'};
        Drill.back();
        await new Promise(r => setTimeout(r, 1200));
        return {lens, before, zoomed, drilled, backOk: !Drill.isActive(),
                labels: document.querySelectorAll('.map-label').length};
    }""")
    assert canvas_page.js_errors == []
    assert r["lens"]["deck"] == "EDGAR MARKOV", "Deck Lens did not load on canvas"
    assert r["zoomed"] < r["before"], "focusLine did not move the camera"
    assert r["drilled"]["active"] and r["drilled"]["bar"], "drill did not run on canvas"
    assert r["backOk"] and r["labels"] > 5, "leaving drill did not restore the map"


# ── Discovery: the front door ───────────────────────────────────────────


def test_landing_paints_a_card_without_the_projection(discover_page):
    """The reframe's whole claim.

    Boot used to block on 12.9 MB of projection before a single pixel appeared, and the
    first click then needed 16.8 MB of incompressible float32 on top. The landing now
    renders from viz_index alone — this asserts the card is *there* while allData may
    still be empty, which is the only way to prove the dependency is gone.
    """
    r = discover_page.evaluate("""() => ({
        mode: document.getElementById('modeSelect').value,
        card: Discovery.index[Discovery.current].n,
        panel: !!document.querySelector('.discover-filters'),
        cardInPanel: document.getElementById('deckInner').textContent
                       .includes('Craterhoof Behemoth'),
        stuckPopup: (() => {
            const p = document.querySelector('.card-popup');
            return !!p && p.style.display === 'block';
        })(),
        nodes: Force.nodeCount,
    })""")
    assert discover_page.js_errors == []
    assert r["mode"] == "discover"
    assert r["card"] == "Craterhoof Behemoth", "?card= deep link was not honoured"
    assert r["panel"], "the discovery controls did not render"
    assert r["nodes"] == 1, "the landing card should be the graph's single seed"
    assert r["cardInPanel"], "the landing card must be readable in the panel"
    assert not r["stuckPopup"], (
        "a card is floating over the graph with nothing hovered — the popup is for hover "
        "only, and a persistent one cannot be dismissed and covers the points behind it"
    )


def test_relation_counts_are_stated_before_any_click(discover_page):
    """23.6% of cards have nothing but similar. A button that turns out to do nothing
    reads as broken; a button labelled 0 reads as a fact about the card. The counts are
    precomputed, so there is no excuse for finding out after the click."""
    r = discover_page.evaluate(r"""() => {
        const btns = [...document.querySelectorAll('.discover-rel')];
        return {
            labels: btns.map(b => b.textContent.replace(/\s+/g, ' ').trim()),
            disabled: btns.map(b => b.disabled),
            counts: Discovery.counts(Discovery.current),
        };
    }""")
    assert discover_page.js_errors == []
    assert any("Similar" in l for l in r["labels"])
    assert r["counts"]["similar"] > 0
    # Every button states its number, and an empty relation is disabled rather than absent.
    for label, disabled in zip(r["labels"], r["disabled"]):
        n = int(label.split()[-1])
        assert disabled == (n == 0), f"{label!r} disabled={disabled} disagrees with its count"


def test_a_card_with_no_synergy_says_zero_rather_than_lying(discover_page):
    """Doubling Season is genuinely absent from the synergy graph — a real coverage hole.
    Surfacing it as `Synergy 0` is the honest rendering of that."""
    r = discover_page.evaluate("""async () => {
        Discovery.show(Discovery.rowByName('Doubling Season'));
        await new Promise(r => setTimeout(r, 300));
        const btns = [...document.querySelectorAll('.discover-rel')];
        return {
            counts: Discovery.counts(Discovery.current),
            synergyBtn: btns.find(b => b.textContent.includes('Synergy')).disabled,
        };
    }""")
    assert discover_page.js_errors == []
    assert r["counts"]["synergy"] == 0
    assert r["synergyBtn"] is True, "an empty relation must be disabled, not clickable"


def test_branching_is_synchronous(discover_page):
    """The reason the table exists. An await inside a click is what makes a graph feel
    laggy instead of physical; this asserts the branch completes within one call."""
    r = discover_page.evaluate("""() => {
        const before = Force.nodeCount;
        const t = performance.now();
        Force.branchByRow(Discovery.current, 'similar');
        const ms = performance.now() - t;
        // Read the count immediately — no await, no timeout. If branching were async
        // the nodes would not exist yet on this line.
        return {ms: ms, before: before, after: Force.nodeCount};
    }""")
    assert discover_page.js_errors == []
    assert r["after"] > r["before"], "branch did not add nodes synchronously"
    assert r["ms"] < 100, f"a branch took {r['ms']:.0f} ms — is something awaiting?"


def test_the_graph_is_not_a_pure_tree(discover_page):
    """The defect a single-seed start exposed.

    `branchFrom` skipped every neighbour already on the graph and only ever added
    parent->child edges, so from one seed there were no cycles and no cross-links — two
    near-duplicates reached down different branches would sit far apart with nothing
    between them, contradicting the file's own thesis of reading adjacency. Branching now
    also links to cards already present.
    """
    r = discover_page.evaluate("""async () => {
        Force.branchByRow(Discovery.current, 'similar');
        await new Promise(r => setTimeout(r, 200));
        const rows = Discovery.neighbours(Discovery.current, 'similar').map(n => n.row);
        for (const row of rows.slice(0, 4)) Force.branchByRow(row, 'similar');
        await new Promise(r => setTimeout(r, 300));
        return {nodes: Force.nodeCount, links: Force.linkCount};
    }""")
    assert discover_page.js_errors == []
    assert r["links"] > r["nodes"] - 1, (
        f"{r['nodes']} nodes / {r['links']} links is still a tree — cross-linking is gone"
    )


def test_the_pin_beats_the_hover(discover_page):
    """"Click a card to open its details" used to evaporate the moment the cursor moved
    off the node: the panel read `hovered || pinned`, so it flicked to whatever you
    happened to be passing over. Hover is a preview; the pin is where you are."""
    r = discover_page.evaluate("""async () => {
        Force.branchByRow(Discovery.current, 'similar');
        await new Promise(r => setTimeout(r, 400));
        document.getElementById('modeSelect').value = 'discover';
        MM.setMode('discover');
        await new Promise(r => setTimeout(r, 300));
        const pinnedName = Discovery.index[Discovery.current].n;
        const c = document.getElementById('forceCanvas');
        const rect = c.getBoundingClientRect();
        // Sweep the cursor across empty canvas — nothing should steal the panel.
        c.dispatchEvent(new MouseEvent('mousemove',
            {bubbles: true, clientX: rect.left + 5, clientY: rect.top + 5}));
        await new Promise(r => setTimeout(r, 300));
        return {panel: document.getElementById('deckInner').textContent, expect: pinnedName};
    }""")
    assert discover_page.js_errors == []
    assert r["expect"] in r["panel"], "the pinned card left the panel when the cursor moved"


def test_feeling_lucky_changes_the_card(discover_page):
    r = discover_page.evaluate("""async () => {
        const first = Discovery.current;
        const seen = new Set([first]);
        for (let i = 0; i < 6; i++) {
            Discovery.reroll();
            await new Promise(r => setTimeout(r, 120));
            seen.add(Discovery.current);
        }
        return {distinct: seen.size, nodes: Force.nodeCount};
    }""")
    assert discover_page.js_errors == []
    assert r["distinct"] > 2, "Feeling lucky kept returning the same card"
    assert r["nodes"] == 1, "a re-roll should reset the graph to the new single card"


def test_filters_narrow_the_pick(discover_page):
    r = discover_page.evaluate("""async () => {
        Discovery.onFilter('supertype', 'Land');
        await new Promise(r => setTimeout(r, 200));
        const land = Discovery.index[Discovery.current].s;
        Discovery.onFilter('supertype', '');
        Discovery.onFilter('color', 'R');
        await new Promise(r => setTimeout(r, 200));
        return {land: land, colour: Discovery.index[Discovery.current].c,
                pool: Discovery.poolSize()};
    }""")
    assert discover_page.js_errors == []
    assert r["land"] == "Land"
    assert r["colour"] == "R"
    assert r["pool"] > 100


def test_the_atlas_is_still_one_click_away(discover_page):
    """Discovery is the front door, not a replacement — the 34,322-point map still has
    to be reachable, and it still backs Deck Lens."""
    # Reads MM.allData, which this fixture deliberately does not wait for.
    await_projection(discover_page)
    r = discover_page.evaluate("""async () => {
        document.getElementById('modeSelect').value = 'explore';
        MM.setMode('explore');
        await new Promise(r => setTimeout(r, 2500));
        const gd = document.getElementById('plot');
        return {traces: MM.mapRenderer.layers.length, rows: MM.allData.length};
    }""")
    assert discover_page.js_errors == []
    assert r["rows"] > 30000, "the projection never loaded behind the landing"
    assert r["traces"] > 0, "switching to Explore did not draw the map"


# ── Slice 3: the tray, import, and the hand-off ─────────────────────────


def test_a_real_moxfield_export_imports_completely(discover_page):
    """The end-to-end claim: paste a decklist, get your deck as a graph.

    Checked against a deck the CLI already ingested, so `cards.json` is an independent
    answer to what that list contains. Resolution goes through `viz_index`, NOT
    `data/decks/index.json` — Deck Lens refuses any slug it does not already know, and an
    imported deck has no slug and never will.
    """
    r = discover_page.evaluate("""async () => {
        const text = await (await fetch('../data/decks/edgar-vampires/decklist.txt')).text();
        const doc = await (await fetch('../data/decks/edgar-vampires/cards.json')).json();
        const res = Discovery.importText(text);
        await new Promise(r => setTimeout(r, 3000));
        const expected = new Set(doc.cards.map(c => c.name));
        const got = new Set(Discovery.library.names());
        const missingFromGraph = [...expected].filter(n => !got.has(n));
        return {
            resolved: res.resolved,
            unresolved: res.missing,
            nodes: Force.nodeCount,
            commander: res.commander >= 0 ? Discovery.index[res.commander].n : null,
            missingFromGraph: missingFromGraph.slice(0, 10),
            missingCount: missingFromGraph.length,
        };
    }""")
    assert discover_page.js_errors == []
    assert r["unresolved"] == [], f"names the index could not resolve: {r['unresolved']}"
    assert r["commander"] == "Edgar Markov", "the commander was not identified"
    assert r["missingCount"] == 0, f"cards in the deck but not the graph: {r['missingFromGraph']}"
    assert r["nodes"] == r["resolved"], (
        f"{r['nodes']} nodes from {r['resolved']} cards — the import grew the deck, which "
        f"is what focusCard did before pinCard existed"
    )


def test_import_keeps_the_discovery_chrome(discover_page):
    """The panel must stay Discovery's after an import, or the tray and export vanish.

    `renderPanel` runs on every reheat, and the graph reheats on every branch — so a
    walk-chrome panel would wipe the controls seconds after the import landed.
    """
    r = discover_page.evaluate("""async () => {
        const text = await (await fetch('../data/decks/goblin-storm/decklist.txt')).text();
        Discovery.importText(text);
        await new Promise(r => setTimeout(r, 2500));
        Force.renderPanel();
        return {
            header: document.querySelector('#deckInner h2').textContent,
            tray: !!document.querySelector('.discover-tray'),
        };
    }""")
    assert discover_page.js_errors == []
    assert r["header"] == "Discover"
    assert r["tray"], "the tray controls were erased by a walk-chrome render"


def test_the_library_is_its_own_thing(discover_page):
    r = discover_page.evaluate("""async () => {
        const row = Discovery.current;
        Discovery.library.toggle(row);
        const added = Discovery.library.has(row);
        Discovery.library.toggle(row);
        const removed = !Discovery.library.has(row);
        Discovery.library.toggle(row);
        Discovery.library.toggle(Discovery.rowByName('Sol Ring'));
        await new Promise(r => setTimeout(r, 150));
        const two = Discovery.library.list.length;
        Discovery.library.clear();
        return {added: added, removed: removed, two: two, cleared: Discovery.library.list.length};
    }""")
    assert discover_page.js_errors == []
    assert r["added"] and r["removed"]
    assert r["two"] == 2 and r["cleared"] == 0


def test_the_brief_is_the_hand_off_not_a_backend(discover_page):
    """There is no server and this does not add one. The pilot loop is 6-10 serial
    subagent spawns costing ~330k-1.7M tokens; a static page cannot run it. So Build emits
    a brief for a human to run in Claude Code, and says so.

    An imported list carries its own commander (`*CMDR*` in the decklist), so this also
    covers the path where the commander is NOT chosen by hand — the brief must still come
    out in the shape `load_brief` reads.
    """
    r = discover_page.evaluate("""async () => {
        const text = await (await fetch('../data/decks/heliod/decklist.txt')).text();
        const res = Discovery.importText(text);
        await new Promise(r => setTimeout(r, 2500));
        const b = Discovery.brief();
        return {resolved: res.resolved, importedCommander: res.commander,
                commander: b.commander, slug: b.slug, bracket: b.bracket,
                mustInclude: b.must_include.length, poolSize: b._manamap.pool_size,
                next: b.next_step};
    }""")
    assert discover_page.js_errors == []
    assert r["resolved"] > 50
    assert r["poolSize"] > 50, "the pool is the graph you are holding"
    assert r["mustInclude"] > 50, "importing fills the tray, which is what must_include is"
    assert r["bracket"] in range(1, 6)
    # The decklist names its own commander, so the brief should not need one chosen.
    if r["importedCommander"] is not None and r["importedCommander"] >= 0:
        assert r["commander"], "an imported commander did not reach the brief"
        assert r["slug"] != "untitled"
    assert "Claude Code" in r["next"], "the brief must say where it gets run"


def test_import_does_not_go_through_deck_lens(discover_page):
    """Deck Lens hard-refuses any slug absent from the CLI-built manifest, and an imported
    deck has no slug. Asserting the manifest is untouched keeps a future change from
    quietly routing imports through a door that cannot open for them."""
    r = discover_page.evaluate("""async () => {
        const text = await (await fetch('../data/decks/sisay/decklist.txt')).text();
        const res = Discovery.importText(text);
        await new Promise(r => setTimeout(r, 2500));
        return {resolved: res.resolved, mode: document.getElementById('modeSelect').value,
                nodes: Force.nodeCount};
    }""")
    assert discover_page.js_errors == []
    assert r["resolved"] > 50
    assert r["mode"] == "discover", "import switched modes — Deck Lens cannot host this"
    assert r["nodes"] == r["resolved"]


def test_the_hover_card_appears_and_leaves(discover_page):
    """The floating card is hover-only: it opens over the point you rest on, and it goes
    away when you are not on one.

    A persistent version shipped briefly — pinned card art that survived re-rolls and
    clicks, could not be dismissed, and hid the points behind it. Two card displays
    existed at once and only one of them knew how to disappear.

    Note the rests. `showCardPopup` has a 180 ms delay, so a sweep that keeps moving
    cancels the timer on every step and the popup never opens — which is exactly how this
    test failed first time, looking like a broken feature rather than a hurried cursor.
    """
    r = discover_page.evaluate("""async () => {
        Force.branchByRow(Discovery.current, 'similar');
        await new Promise(r => setTimeout(r, 600));
        Force.fit();
        await new Promise(r => setTimeout(r, 700));

        const c = document.getElementById('forceCanvas');
        const rect = c.getBoundingClientRect();
        const move = (x, y) => c.dispatchEvent(new MouseEvent('mousemove',
            {bubbles: true, clientX: rect.left + x, clientY: rect.top + y}));
        const shown = () => {
            const p = document.querySelector('.card-popup');
            return !!p && p.style.display === 'block';
        };

        const before = shown();

        // pick() sets cursor:pointer over a node — cheaper than guessing coordinates.
        let at = null;
        outer:
        for (let x = 0; x < rect.width; x += 4) {
            for (let y = 0; y < rect.height; y += 4) {
                move(x, y);
                if (c.style.cursor === 'pointer') { at = [x, y]; break outer; }
            }
        }
        if (!at) return {before: before, found: false};

        move(at[0], at[1]);
        await new Promise(r => setTimeout(r, 450));
        const opened = shown();

        move(2, 2);
        await new Promise(r => setTimeout(r, 250));
        const afterOff = shown();

        move(at[0], at[1]);
        await new Promise(r => setTimeout(r, 450));
        c.dispatchEvent(new MouseEvent('mouseleave', {bubbles: true}));
        await new Promise(r => setTimeout(r, 250));

        return {before: before, found: true, opened: opened,
                afterOff: afterOff, afterLeave: shown()};
    }""")
    assert discover_page.js_errors == []
    assert r["found"], "no node under the cursor anywhere — the graph did not draw"
    assert not r["before"], "a card was floating before anything was hovered"
    assert r["opened"], "resting on a node did not open the card"
    assert not r["afterOff"], "the card stayed after the cursor moved off the node"
    assert not r["afterLeave"], (
        "the card stayed after the cursor left the canvas — mouseleave fires no mousemove, "
        "so nothing else would ever clear it"
    )


def test_a_sample_of_cards_stays_named(discover_page):
    """Names you can read without touching anything.

    Labelling all 500 nodes is an unreadable smear; labelling only the hovered one means
    the graph says nothing until you interact with it. So a bounded sample is placed
    greedily and rejected on collision — which is why it thins out when the graph is dense
    and fills back in as you zoom, with no zoom logic of its own.

    Canvas text cannot be queried, so this asserts the count the draw actually placed.
    """
    r = discover_page.evaluate("""async () => {
        const grow = async () => {
            Force.branchByRow(Discovery.current, 'similar');
            await new Promise(r => setTimeout(r, 300));
            for (const nb of Discovery.neighbours(Discovery.current, 'similar').slice(0, 4)) {
                Force.branchByRow(nb.row, 'similar');
            }
            await new Promise(r => setTimeout(r, 900));
        };
        const single = Force.labelCount;      // the landing: one card
        await grow();
        Force.fit();
        await new Promise(r => setTimeout(r, 800));
        return {single: single, many: Force.labelCount,
                nodes: Force.nodeCount, cap: Force.LABEL_MAX};
    }""")
    assert discover_page.js_errors == []
    assert r["nodes"] > 15, "the graph did not grow enough to test label thinning"
    assert r["many"] > 3, "almost nothing is named — the graph reads as anonymous dots"
    assert r["many"] <= r["cap"], f"{r['many']} labels exceeds the {r['cap']} cap"
    assert r["many"] < r["nodes"], (
        "every node is labelled — at 500 nodes that is a smear, not a sample"
    )


def test_the_hover_card_stays_inside_the_frame(discover_page):
    """Reported: hovering a point near the foot of the page ran the card off-screen.

    The cause was that `positionPopup` measured the popup the instant the <img> was
    inserted — before the network returned anything — so the height read ~0 and the bottom
    clamp had nothing to clamp. The CSS now reserves the card's aspect ratio and the
    fallback height is explicit.
    """
    r = discover_page.evaluate("""async () => {
        const host = document.getElementById('plot');
        const rect = host.getBoundingClientRect();
        const row = Discovery.current;
        const out = [];
        const spots = [['bottom edge', rect.bottom - 4], ['near bottom', rect.bottom - 40],
                       ['middle', rect.top + rect.height / 2], ['top edge', rect.top + 4]];
        for (const [label, cy] of spots) {
            MM.hideCardPopup();
            MM.showCardPopup(row, rect.left + 300, cy);
            // Wait for the IMAGE, not for a timer. A fixed delay makes this a test of
            // whether Scryfall answered in 420 ms — it failed intermittently in full runs
            // and passed alone, which is the signature of timing an external fetch. The
            // clamp is what is under test, and it can only be judged once the popup has
            // its real height.
            //
            // Wait for the popup to be VISIBLE, not merely present. The element is
            // created once and reused, so after the first spot it is already in the DOM
            // while still hidden behind the 180 ms hover delay — polling for existence
            // returns instantly and measures a display:none box as 0x0.
            let p = null;
            for (let t = 0; t < 60; t++) {
                const el = document.querySelector('.card-popup');
                if (el && el.style.display === 'block') { p = el; break; }
                await new Promise(r => setTimeout(r, 50));
            }
            if (!p) return [{at: label, missing: true}];
            const img = p.querySelector('img');
            if (img && !img.complete) {
                await new Promise(res => {
                    img.addEventListener('load', res, {once: true});
                    img.addEventListener('error', res, {once: true});
                    setTimeout(res, 4000);
                });
            }
            await new Promise(r => setTimeout(r, 120));
            const pr = p.getBoundingClientRect();
            out.push({
                at: label, height: Math.round(pr.height),
                // Scryfall can fail or rate-limit under a full-suite run, and the popup
                // then collapses to the card's name — a legitimately short box. The clamp
                // still has to work for it; its HEIGHT just is not the card's.
                //
                // The class goes on the POPUP itself: the onerror handler does
                // `this.parentElement.classList.add('card-popup-failed')`, where `this` is
                // the <img> and the parent is `.card-popup`. Checking with querySelector
                // looks for a descendant and never matches — which is why this test kept
                // failing on the height assertion with `failed` reading false.
                failed: p.classList.contains('card-popup-failed'),
                insideFrame: pr.bottom <= rect.bottom + 1 && pr.top >= rect.top - 1,
                insideViewport: pr.bottom <= window.innerHeight + 1 && pr.top >= 0,
            });
        }
        MM.hideCardPopup();
        return out;
    }""")
    assert discover_page.js_errors == []
    for case in r:
        assert case["insideFrame"], f"the card escaped the plot frame at the {case['at']}"
        assert case["insideViewport"], f"the card ran off the page at the {case['at']}"
        # The clamp assertions above are the invariant and always apply. This one is
        # about the CSS reserving the card's box before the image arrives, so it only
        # means anything when a card is actually being shown.
        if not case.get("failed"):
            assert case["height"] > 200, (
                f"the popup measured {case['height']}px at the {case['at']} — an unloaded "
                f"image measuring ~0 is exactly what defeated the clamp"
            )


def test_a_click_survives_a_shaky_hand(discover_page):
    """Reported: "some points don't expand the first time, then work if I click again".

    Not latency and not rendering — the click event was being **swallowed**. d3-drag's
    `clickDistance` defaults to 0, so ANY pointer movement between mousedown and mouseup
    makes d3 install a capture-phase suppressor that eats the following `click`. One pixel
    of hand tremor and the card does not expand; the next click happens to be steadier and
    "fixes" it.

    Measured on this page with the default: 0px jitter delivered the click, 1px and 3px
    swallowed it. With a 6px tap tolerance, 0px and 3px deliver and 12px is correctly read
    as a drag.

    The tolerance is the whole point of the test, so it drives the real drag behaviour
    rather than asserting a constant.
    """
    r = discover_page.evaluate("""async () => {
        Force.branchByRow(Discovery.current, 'similar');
        await new Promise(r => setTimeout(r, 600));
        Force.fit();
        await new Promise(r => setTimeout(r, 800));

        const c = document.getElementById('forceCanvas');
        const nodeAt = () => {
            const r = c.getBoundingClientRect();
            for (let x = 0; x < r.width; x += 3) {
                for (let y = 0; y < r.height; y += 3) {
                    c.dispatchEvent(new MouseEvent('mousemove',
                        {bubbles: true, clientX: r.left + x, clientY: r.top + y}));
                    if (c.style.cursor === 'pointer') return [r.left + x, r.top + y];
                }
            }
            return null;
        };
        // press, jitter, release, click — the shape of a real click by a real hand
        const tap = async (jitter) => {
            const at = nodeAt();
            if (!at) return null;
            let clicks = 0;
            const count = () => { clicks++; };
            c.addEventListener('click', count, true);
            const o = (x, y) => ({bubbles: true, clientX: x, clientY: y,
                                  view: window, button: 0});
            c.dispatchEvent(new MouseEvent('mousedown', o(at[0], at[1])));
            window.dispatchEvent(new MouseEvent('mousemove', o(at[0] + jitter, at[1])));
            window.dispatchEvent(new MouseEvent('mouseup', o(at[0] + jitter, at[1])));
            c.dispatchEvent(new MouseEvent('click', o(at[0] + jitter, at[1])));
            await new Promise(r => setTimeout(r, 120));
            c.removeEventListener('click', count, true);
            return clicks;
        };
        return {steady: await tap(0), tremor: await tap(3), drag: await tap(14)};
    }""")
    assert discover_page.js_errors == []
    assert r["steady"] == 1, "a perfectly still click did not register at all"
    assert r["tremor"] == 1, (
        "a 3px hand tremor swallowed the click — clickDistance is back at d3's default of "
        "0, and cards will intermittently refuse to expand"
    )
    assert r["drag"] == 0, "a 14px drag should be a fling, not a click"


def test_a_click_falls_back_to_the_highlighted_card(discover_page):
    """The simulation keeps running after a branch, so a node can drift out from under the
    cursor between press and release. The card the UI is highlighting is the one the user
    aimed at, so the click handler uses it when the hit test comes up empty."""
    src = discover_page.evaluate(
        "() => [...document.scripts].map(s => s.src).find(s => s.includes('force.js'))")
    body = discover_page.evaluate("""async (url) => (await (await fetch(url)).text())""", src)
    assert "|| hovered" in body, "the click handler no longer falls back to the hovered node"


def test_clicking_a_card_opens_it_in_the_panel(discover_page):
    """The graph says what is selected; the panel follows.

    Clicking used to branch the graph and leave the panel on the landing card — so the
    relation counts described a card you were no longer looking at, and "+ Keep this card"
    put the wrong one in the tray.

    `Discovery.focus` is deliberately not `show`: `show` reseeds the graph, and opening a
    card you walked to must not throw away the walk that got you there.
    """
    r = discover_page.evaluate("""async () => {
        const landing = Discovery.index[Discovery.current].n;
        Force.branchByRow(Discovery.current, 'similar');
        await new Promise(r => setTimeout(r, 700));
        Force.fit();
        await new Promise(r => setTimeout(r, 800));
        const nodesAfterBranch = Force.nodeCount;

        const c = document.getElementById('forceCanvas');
        const rect = c.getBoundingClientRect();
        const title = () => {
            const el = document.querySelector('#deckInner .lens-title');
            return el ? el.textContent : null;
        };

        let clicked = null;
        outer:
        for (let x = 0; x < rect.width; x += 3) {
            for (let y = 0; y < rect.height; y += 3) {
                c.dispatchEvent(new MouseEvent('mousemove',
                    {bubbles: true, clientX: rect.left + x, clientY: rect.top + y}));
                if (c.style.cursor !== 'pointer') continue;
                const before = Discovery.current;
                c.dispatchEvent(new MouseEvent('click',
                    {bubbles: true, clientX: rect.left + x, clientY: rect.top + y}));
                await new Promise(r => setTimeout(r, 250));
                if (Discovery.current !== before) {
                    clicked = Discovery.index[Discovery.current].n;
                    break outer;
                }
            }
        }

        const keep = document.querySelector('.discover-keep');
        const libBefore = Discovery.library.list.length;
        if (keep) keep.click();
        await new Promise(r => setTimeout(r, 200));

        return {
            landing: landing, clicked: clicked, panel: title(),
            counts: Discovery.counts(Discovery.current),
            libBefore: libBefore, libAfter: Discovery.library.list.length,
            trayNames: Discovery.library.names(),
            keptTheRightCard: Discovery.library.names().includes(clicked),
            graphKept: Force.nodeCount >= nodesAfterBranch,
        };
    }""")
    assert discover_page.js_errors == []
    assert r["clicked"], "clicking a node never changed the selected card"
    assert r["clicked"] != r["landing"], "the click selected the landing card again"
    assert r["panel"] == r["clicked"], (
        f"panel shows {r['panel']!r} after clicking {r['clicked']!r} — the relation counts "
        f"and Keep button would describe the wrong card"
    )
    assert r["counts"]["similar"] > 0, "the panel is not showing the clicked card's relations"
    assert r["libAfter"] == r["libBefore"] + 1
    assert r["keptTheRightCard"], f"the tray got {r['trayNames']} instead of the clicked card"
    assert r["graphKept"], "opening a card discarded the walk that reached it"


def test_a_checked_in_deck_loads_with_its_commander(discover_page):
    """Load one of the published decks by slug and walk out from it.

    Distinct from a pasted import in the way that matters: the manifest carries a KNOWN
    commander, so it is ringed and centred rather than inferred from a `*CMDR*` marker.
    """
    r = discover_page.evaluate("""async () => {
        const res = await Discovery.loadDeck('goblin-storm');
        await new Promise(r => setTimeout(r, 3000));
        return {
            res: res,
            commanderName: res && res.commander >= 0
                ? Discovery.index[res.commander].n : null,
            panel: (document.querySelector('#deckInner .lens-title') || {}).textContent,
            membership: Force.membership(),
            tray: Discovery.library.list.length,
            decks: Discovery.decks.map(d => d.slug),
        };
    }""")
    assert discover_page.js_errors == []
    assert "goblin-storm" in r["decks"] and len(r["decks"]) >= 7
    assert r["res"]["missing"] == [], f"unresolved deck cards: {r['res']['missing']}"
    assert r["commanderName"] == "Zada, Hedron Grinder"
    assert r["panel"] == "Zada, Hedron Grinder", "the panel did not open on the commander"
    assert r["membership"]["commander"] == 1, "exactly one card should be ringed as commander"
    assert r["membership"]["deck"] > 50
    assert r["membership"]["explored"] == 0, "a freshly loaded deck has nothing explored yet"
    assert r["tray"] == r["membership"]["deck"], "the loaded deck should fill the tray"


def test_cards_you_brought_look_different_from_cards_you_found(discover_page):
    """The point of loading a deck: you can see what is yours and what you wandered into.

    Deck membership drives radius, fill opacity, the white ring, the commander's gold ring
    and — the part that makes the deck readable as a structure — warm heavy edges between
    two deck cards versus thin cool ones for everything discovered.
    """
    r = discover_page.evaluate("""async () => {
        await Discovery.loadDeck('goblin-storm');
        await new Promise(r => setTimeout(r, 3000));
        const before = Force.membership();
        for (const row of Discovery.library.list.slice(0, 5)) {
            Force.branchByRow(row, 'similar');
            await new Promise(r => setTimeout(r, 120));
        }
        await new Promise(r => setTimeout(r, 1200));
        return {before: before, after: Force.membership()};
    }""")
    assert discover_page.js_errors == []
    a = r["after"]
    assert a["explored"] > 10, "branching from deck cards pulled in nothing"
    assert a["deck"] == r["before"]["deck"], "exploring changed the deck's own membership"
    # Deck edges are a fixed set; every link added by exploring is NOT a deck edge, which
    # is exactly what makes the two readable apart.
    assert a["deckLinks"] == r["before"]["deckLinks"], "an explored edge was drawn as a deck edge"
    assert a["links"] > a["deckLinks"], "no exploration edges were added"


def test_loading_a_deck_replaces_the_previous_one(discover_page):
    """Two decks at once would be an unreadable pile, and the membership flags would be
    ambiguous — a card in both decks could only be one colour."""
    r = discover_page.evaluate("""async () => {
        await Discovery.loadDeck('goblin-storm');
        await new Promise(r => setTimeout(r, 2500));
        const first = {m: Force.membership(),
                       cmdr: Discovery.index[Discovery.current].n};
        await Discovery.loadDeck('heliod');
        await new Promise(r => setTimeout(r, 3000));
        return {first: first, second: {m: Force.membership(),
                                       cmdr: Discovery.index[Discovery.current].n}};
    }""")
    assert discover_page.js_errors == []
    assert r["first"]["cmdr"] == "Zada, Hedron Grinder"
    assert r["second"]["cmdr"] != r["first"]["cmdr"], "the second deck did not take over"
    assert r["second"]["m"]["commander"] == 1, "two commanders are ringed at once"
    assert r["second"]["m"]["explored"] == 0, "the previous deck leaked in as explored cards"


def test_a_loaded_deck_arrives_already_arranged(discover_page):
    """Loading a deck used to be a spectacle: a hundred nodes seeded at scaled world
    coordinates appeared as a distorted smear, collapsed inward over several seconds, and
    re-framed itself fourteen times on a 550 ms timer while the user could do nothing.

    `enter` now pre-settles the layout with synchronous `sim.tick()` calls, which advance
    the simulation WITHOUT dispatching tick events — so nothing draws until it is done and
    the graph arrives arranged.
    """
    r = discover_page.evaluate("""async () => {
        const t0 = performance.now();
        await Discovery.loadDeck('goblin-storm');
        const ms = performance.now() - t0;
        const bb = () => JSON.stringify(Force.bbox());
        const atArrival = bb();
        await new Promise(r => setTimeout(r, 1500));
        return {ms: ms, stillAfterwards: bb() === atArrival, nodes: Force.nodeCount};
    }""")
    assert discover_page.js_errors == []
    assert r["nodes"] > 50
    assert r["ms"] < 3000, f"the deck took {r['ms']:.0f} ms to arrange"
    assert r["stillAfterwards"], (
        "the layout kept moving after it arrived — the intro is animating again"
    )


def test_your_zoom_survives_the_graph_moving(discover_page):
    """Reported: zooming while the graph was still moving zoomed back out.

    A settle-time `fitToGraph` was overwriting the transform mid-gesture. Auto-fit is now
    a suggestion — `fitToGraph(animate, auto=true)` returns early once any real pan, zoom
    or drag has happened. `sourceEvent` is what separates a gesture from a programmatic
    transform.
    """
    r = discover_page.evaluate("""async () => {
        await Discovery.loadDeck('goblin-storm');
        await new Promise(r => setTimeout(r, 1200));
        const c = document.getElementById('forceCanvas');
        const rect = c.getBoundingClientRect();
        // Reheat, then zoom into the moving graph — the exact reported sequence.
        Force.branchByRow(Discovery.current, 'similar');
        const before = d3.zoomTransform(c).k;
        c.dispatchEvent(new WheelEvent('wheel', {bubbles: true, cancelable: true,
            clientX: rect.left + 400, clientY: rect.top + 300, deltaY: -300}));
        const zoomed = d3.zoomTransform(c).k;
        await new Promise(r => setTimeout(r, 3000));   // through the settle and any fit
        return {before: before, zoomed: zoomed, after: d3.zoomTransform(c).k};
    }""")
    assert discover_page.js_errors == []
    assert r["zoomed"] != r["before"], "the wheel gesture did not zoom at all"
    assert abs(r["after"] - r["zoomed"]) < 1e-6, (
        f"the camera was stolen back: zoomed to {r['zoomed']:.3f}, ended at {r['after']:.3f}"
    )


def test_the_graph_stops_moving_promptly_after_a_branch(discover_page):
    """New cards should fly out, find their place, and stop.

    alphaDecay was 0.015 — an ~8 second settle — chosen so the initial layout could be
    watched arranging itself. That layout no longer animates, so all the low decay bought
    was a graph drifting under the cursor long after anything interesting had happened.
    """
    r = discover_page.evaluate("""async () => {
        Force.branchByRow(Discovery.current, 'similar');
        const bb = () => JSON.stringify(Force.bbox());
        let last = bb(), still = 0, elapsed = 0, moved = false;
        while (elapsed < 6000 && still < 400) {
            await new Promise(r => setTimeout(r, 100));
            elapsed += 100;
            const now = bb();
            if (now === last) { still += 100; } else { still = 0; last = now; moved = true; }
        }
        return {settleMs: elapsed - still, settled: still >= 400, moved: moved};
    }""")
    assert discover_page.js_errors == []
    assert r["moved"], "branching did not animate at all — the graph is frozen"
    assert r["settled"], "the graph never came to rest within 6 s"
    assert r["settleMs"] < 3000, f"took {r['settleMs']} ms to settle after a branch"


# ── synergy reasons ─────────────────────────────────────────────────────


def test_synergy_neighbours_carry_their_reason(discover_page):
    """The reason is what makes a synergy edge worth drawing: it says WHY two cards are
    connected, not merely that a rule fired.

    These strings were already being computed and thrown away — the old Find Synergies
    wrote them into a Plotly trace and then set `hoverinfo: 'none'`, so nobody ever saw
    one. They now ride in `neighbours.bin` as a uint8 code plus an appended vocabulary,
    so branching stays synchronous.
    """
    r = discover_page.evaluate("""async () => {
        const row = Discovery.rowByName("Ashnod's Altar");
        Discovery.show(row);
        await new Promise(r => setTimeout(r, 400));
        const syn = Discovery.neighbours(row, 'synergy');
        return {
            count: syn.length,
            allHaveReasons: syn.every(s => !!s.reason),
            sample: syn.slice(0, 3).map(s => ({card: Discovery.index[s.row].n,
                                               reason: s.reason})),
            vocab: Discovery.table.reasons.length,
            similarHasNoReason: !Discovery.neighbours(row, 'similar')[0].reason,
        };
    }""")
    assert discover_page.js_errors == []
    assert r["count"] > 5
    assert r["vocab"] == 24, "the reason vocabulary did not travel with the file"
    assert r["allHaveReasons"], f"a synergy partner arrived with no reason: {r['sample']}"
    assert r["similarHasNoReason"], "similarity is not a rule match and has no reason"
    reasons = {s["reason"] for s in r["sample"]}
    assert any("Sac" in x or "Token" in x for x in reasons), f"unexpected reasons: {reasons}"


def test_synergy_edges_are_labelled_in_the_graph(discover_page):
    """The reasons reach the canvas, subject to the same collision discipline as node
    labels so they can never sit on top of a card name.

    **Wait for the layout to settle before measuring.** Reading these counters mid-settle
    reports almost nothing — the nodes are still clustered, so every label collides — and
    it looks exactly like a broken feature. That cost several minutes of chasing a
    non-bug, hence the explicit settle here.
    """
    r = discover_page.evaluate("""async () => {
        const row = Discovery.rowByName("Ashnod's Altar");
        Discovery.show(row);
        await new Promise(r => setTimeout(r, 400));
        Force.branchByRow(row, 'synergy');
        // Settle: alphaDecay 0.08 is ~1.3 s, and a cramped graph collides every label.
        const bb = () => JSON.stringify(Force.bbox());
        let last = bb(), still = 0, elapsed = 0;
        while (elapsed < 6000 && still < 500) {
            await new Promise(r => setTimeout(r, 100));
            elapsed += 100;
            const now = bb();
            if (now === last) still += 100; else { still = 0; last = now; }
        }
        const t = d3.zoomTransform(document.getElementById('forceCanvas'));
        const b = Force.bbox();
        return {edgeLabels: Force.edgeLabelCount, nodeLabels: Force.labelCount,
                nodes: Force.nodeCount,
                spreadPx: Math.round((b.maxX - b.minX) * t.k)};
    }""")
    assert discover_page.js_errors == []
    assert r["spreadPx"] > 150, (
        f"the graph is only {r['spreadPx']}px wide — still settling, so any label count "
        f"below is meaningless"
    )
    assert r["edgeLabels"] > 0, "no synergy edge was labelled with its reason"
    assert r["nodeLabels"] > 0, "edge labels crowded out every card name"


def test_obsolescence_fills_in_outside_explore(discover_page):
    """"Obsoleted By" and its advantage badges used to be invisible in three of the five
    panels that render card detail.

    The index was fetched in exactly one place — inside `updateViewerPanel` — and patched
    only that panel's open card, so the browse panel, The Walk and Discover drew a
    placeholder nothing ever filled. `buildObsolescenceHtml` now triggers the load itself
    and a single patcher fills every placeholder on the page.
    """
    r = discover_page.evaluate("""async () => {
        // A card that definitely has a replacement, opened in Discover with no prior
        // explore selection to have warmed the index.
        Discovery.show(Discovery.rowByName('Storm Crow'));
        await new Promise(r => setTimeout(r, 2500));
        const section = document.querySelector('.obsolescence-section');
        return {
            filled: !!section,
            hasAdvantageBadge: !!document.querySelector('.obsolescence-badge'),
            stillPlaceholder: !!document.querySelector('.obsolescence-placeholder'),
            text: section ? section.textContent.slice(0, 120) : null,
        };
    }""")
    assert discover_page.js_errors == []
    assert r["filled"], "Obsoleted By never rendered in Discover — the index never loaded"
    assert r["hasAdvantageBadge"], "the advantage strings (Lower CMC, ...) are still hidden"
    assert not r["stillPlaceholder"], "a placeholder was left unpatched"


# ── one relation mechanism, every panel ─────────────────────────────────


def test_the_old_find_buttons_are_gone(discover_page):
    """Find Similar Cards / Find Synergies were broken four ways at once.

    They took no card argument and read `selectedCards`, so they were silent no-ops in
    Discover and in the browse panel (both clear it), acted on the *wrong card* in The
    Walk while drawing onto a hidden Plotly surface, and threw outright under
    `?renderer=canvas` where `#plot` has no `.data`. Nothing tested any of it.
    """
    r = discover_page.evaluate("""async () => {
        const src = [...document.scripts].map(s => s.src).find(s => s.includes('mana-map.js'));
        const body = await (await fetch(src)).text();
        return {
            buttons: document.querySelectorAll('.btn-similar, .btn-synergy').length,
            findSimilar: typeof MM.findSimilar,
            findSynergies: typeof MM.findSynergies,
            relate: typeof MM.relate,
            sourceHasHandlers: /function findSimilarCards|function findSynergyCards/.test(body),
            sourceHasTraceFlag: body.includes('similarTrace'),
        };
    }""")
    assert discover_page.js_errors == []
    assert r["buttons"] == 0, "the old buttons still render"
    assert r["findSimilar"] == "undefined" and r["findSynergies"] == "undefined"
    assert r["relate"] == "function", "nothing replaced them"
    assert not r["sourceHasHandlers"], "the handlers are unreferenced but still present"
    assert not r["sourceHasTraceFlag"], "the highlight-trace machinery survives"


def test_relations_render_in_every_panel(discover_page):
    """The unification. The buttons live in `buildCardDetailHtml` now, so every panel that
    shows a card offers the same control — which is what makes deleting the old pair a
    merge rather than the removal of a feature from explore."""
    r = discover_page.evaluate("""async () => {
        const count = () => {
            const p = document.querySelector('.deck-panel.open, #detailPanel');
            return document.querySelectorAll('.discover-rel').length;
        };
        const seen = {};
        seen.discover = count();

        document.getElementById('modeSelect').value = 'explore';
        MM.setMode('explore');
        await new Promise(r => setTimeout(r, 3000));
        MM.selectByName("Ashnod's Altar");
        await new Promise(r => setTimeout(r, 1200));
        seen.explore = count();

        // browse panel — a different renderer again, and one of the silent no-ops
        MM.enterNeighbourhood(Discovery.rowByName("Ashnod's Altar"));
        await new Promise(r => setTimeout(r, 1500));
        seen.browse = count();
        return seen;
    }""")
    assert discover_page.js_errors == []
    for panel, n in r.items():
        assert n >= 3, f"the {panel} panel shows {n} relation controls, expected 3"


def test_a_relation_always_grows_the_graph(discover_page):
    """One button, one behaviour, everywhere: it grows what you are holding.

    This control has been rewritten twice, and the history is the point. It first FORKED —
    graph modes grew the graph, Explore opened a linear browse set — on the reasoning that
    a scatter plot cannot grow. One control meaning two things is why the atlas felt dead.
    Then it CARRIED YOU OUT: clicking a relation in Explore switched to Discover. Better,
    but it still meant the map could only ever hand you off, never respond.

    Now it grows the graph wherever you are. From Explore that means the constellation and
    its arcs appear on the map and you stay put; from the graph it branches as before. What
    is invariant, and all this test asserts, is that the graph is strictly larger and
    nothing was thrown away.
    """
    r = discover_page.evaluate("""async () => {
        const row = Discovery.rowByName("Ashnod's Altar");

        Discovery.show(row);
        await new Promise(r => setTimeout(r, 500));
        const before = Session.size();
        MM.relate(row, 'synergy');
        await new Promise(r => setTimeout(r, 1200));
        const fromDiscover = {grew: Session.size() > before, held: Session.rows().slice()};

        // From the atlas: same control, same direction of travel — bigger, never smaller.
        document.getElementById('modeSelect').value = 'explore';
        MM.setMode('explore');
        await new Promise(r => setTimeout(r, 3000));
        const beforeExplore = Session.size();
        MM.relate(row, 'similar');
        await new Promise(r => setTimeout(r, 1500));
        const after = Session.rows();
        return {
            fromDiscover: {grew: fromDiscover.grew},
            mode: document.getElementById('modeSelect').value,
            grewAgain: Session.size() > beforeExplore,
            keptEverything: fromDiscover.held.every(x => after.indexOf(x) !== -1),
            stillHasIt: Session.has(row),
            noBrowseSet: !MM.browseSet,
        };
    }""")
    assert discover_page.js_errors == []
    assert r["fromDiscover"]["grew"], "a relation in Discover did not grow the graph"
    assert r["mode"] == "explore", "growing from the atlas left the atlas"
    assert r["grewAgain"], "a relation in Explore did not grow the graph"
    assert r["keptEverything"], "growing from the atlas dropped cards already held"
    assert r["stillHasIt"] and r["noBrowseSet"]


def test_the_library_follows_the_card_not_the_mode(discover_page):
    """Keeping something you found in the atlas is the same act as keeping something you
    walked to, so the control lives in the shared card HTML rather than only in Discover."""
    r = discover_page.evaluate("""async () => {
        document.getElementById('modeSelect').value = 'explore';
        MM.setMode('explore');
        await new Promise(r => setTimeout(r, 3000));
        MM.selectByName("Ashnod's Altar");
        await new Promise(r => setTimeout(r, 1000));
        const keep = document.querySelector('.discover-keep');
        const before = Discovery.library.list.length;
        if (keep) keep.click();
        await new Promise(r => setTimeout(r, 400));
        return {
            hadButton: !!keep,
            before: before,
            after: Discovery.library.list.length,
            label: (document.querySelector('.discover-keep') || {}).textContent,
            kept: Discovery.library.names().includes("Ashnod's Altar"),
        };
    }""")
    assert discover_page.js_errors == []
    assert r["hadButton"], "no Keep control in the atlas — the library is still Discover-only"
    assert r["after"] == r["before"] + 1 and r["kept"]
    assert "In library" in (r["label"] or ""), "the control did not reflect the new state"


def test_relations_survive_the_canvas_renderer(canvas_page):
    """The old path threw here: `Plotly.addTraces` on a div with no `.data`, swallowed as
    an unhandled rejection so nothing drew and nothing complained."""
    r = canvas_page.evaluate("""async () => {
        await new Promise(r => setTimeout(r, 500));
        const row = MM.allData.findIndex(d => d.n === "Ashnod's Altar");
        MM.selectByName("Ashnod's Altar");
        await new Promise(r => setTimeout(r, 800));
        const before = Session.size();
        MM.relate(row, 'similar');
        await new Promise(r => setTimeout(r, 1800));
        return {mode: document.getElementById('modeSelect').value,
                grew: Session.size() > before, seeded: Session.has(row),
                arcs: (MM.mapRenderer.layers.find(l => l.mode === 'edges') || {edges: []})
                        .edges.length};
    }""")
    assert canvas_page.js_errors == [], f"the canvas path threw: {canvas_page.js_errors}"
    # Growing now happens in place, so the atlas keeps the user AND draws the arcs.
    assert r["mode"] == "explore" and r["seeded"] and r["grew"]
    assert r["arcs"] > 0, "the canvas path grew the graph but drew no arcs"


# ── Explore as an orientation lens ──────────────────────────────────────


def test_leaving_a_graph_for_the_atlas_shows_where_it_sits(discover_page):
    """Explore stopped being a workspace.

    The graph encodes **adjacency and has no absolute position** — force.js says so in its
    own header — so "where does this sit in card space" is the one question it structurally
    cannot answer. Entering Explore from a graph now lights up exactly those cards and dims
    the other 34,000, which gives the atlas a job it is uniquely good at instead of being a
    second, worse place to work.
    """
    r = discover_page.evaluate("""async () => {
        MM.relate(Discovery.current, 'synergy');
        await new Promise(r => setTimeout(r, 1400));
        const built = Force.nodeCount;

        document.getElementById('modeSelect').value = 'explore';
        MM.setMode('explore');
        await new Promise(r => setTimeout(r, 4000));
        // Arriving in Explore no longer auto-orients — a clean atlas is the entry state,
        // by design. The lens is engaged from INSIDE Explore, which is what `MM.relate`
        // does here.
        MM.relate(Session.rows()[0], 'synergy');
        await new Promise(r => setTimeout(r, 1500));
        return {
            // Re-read AFTER engaging the lens: the relate above grows the graph, and the
            // invariant under test is "the lens lights the whole graph", not "the graph
            // never changed".
            built: Force.nodeCount,
            grownFrom: built,
            active: !!MM.orientation,
            // Membership is read live from Session now rather than snapshotted into
            // `orientation.rows`, and the anchor is `Session.focus`. The lens object
            // holds only whether the lens is on.
            rows: MM.orientation ? Session.size() : 0,
            anchored: Session.focus >= 0,
            status: document.getElementById('status').textContent,
        };
    }""")
    assert discover_page.js_errors == []
    assert r["built"] > 1
    assert r["active"], "entering Explore from a graph did not light it up"
    assert r["rows"] == r["built"], (
        f"{r['rows']} cards highlighted from a {r['built']}-card graph"
    )
    assert r["anchored"], "the card you were on should be marked in the atlas"
    assert "cards shown" not in r["status"], (
        "the status is still the bare corpus count — you came to see YOUR cards"
    )


def test_escape_returns_the_whole_atlas(discover_page):
    """The lens is a view, not a trap: Esc gives the full map back before it touches the
    selection."""
    r = discover_page.evaluate("""async () => {
        MM.relate(Discovery.current, 'similar');
        await new Promise(r => setTimeout(r, 1200));
        document.getElementById('modeSelect').value = 'explore';
        MM.setMode('explore');
        await new Promise(r => setTimeout(r, 3500));
        // Arriving in Explore no longer auto-orients — a clean atlas is the entry state,
        // by design. The lens is engaged from INSIDE Explore, which is what `MM.relate`
        // does here.
        MM.relate(Session.rows()[0], 'similar');
        await new Promise(r => setTimeout(r, 1500));
        const on = !!MM.orientation;
        MM.clearOrientation();
        await new Promise(r => setTimeout(r, 800));
        return {on: on, off: !MM.orientation,
                status: document.getElementById('status').textContent};
    }""")
    assert discover_page.js_errors == []
    assert r["on"] and r["off"]
    assert "cards shown" in r["status"], "clearing the lens did not restore the atlas status"


def test_region_labels_do_not_pile_on_each_other(canvas_page):
    """The inconsistency that made the atlas feel noisier than the graph.

    `force.js` has rejected colliding node labels since the graph shipped; the map renderer
    emitted every region label unconditionally, so "White Creatures — Flyers — ETB (West)"
    sat straight across "Green Creatures — ETB — Tramplers (East)" and neither was readable.

    Collision is evaluated in PIXELS during positioning, not in `setAnnotations`: the
    annotations carry world coordinates, and comparing those to pixel widths is a units
    error that rejects nearly everything. Doing it at position time also makes it
    zoom-responsive.
    """
    r = canvas_page.evaluate("""async () => {

        for (let i = 0; i < 200 && !document.querySelector('.map-label'); i++) {
            await new Promise(r => setTimeout(r, 50));
        }
        await new Promise(r => setTimeout(r, 2500));
        const all = [...document.querySelectorAll('.map-label')];
        const shown = all.filter(e => e.style.display !== 'none');
        const boxes = shown.map(e => e.getBoundingClientRect());
        let overlaps = 0;
        for (let i = 0; i < boxes.length; i++) {
            for (let j = i + 1; j < boxes.length; j++) {
                const a = boxes[i], b = boxes[j];
                if (a.left < b.right && a.right > b.left &&
                    a.top < b.bottom && a.bottom > b.top) overlaps++;
            }
        }
        return {inDom: all.length, visible: shown.length, overlaps: overlaps};
    }""")
    assert canvas_page.js_errors == []
    assert r["inDom"] > 5, "no region labels rendered at all"
    assert r["visible"] > 3, f"only {r['visible']} labels survived — the filter is too harsh"
    assert r["overlaps"] == 0, f"{r['overlaps']} pairs of region labels overlap"


def test_growing_from_the_atlas_never_destroys_the_graph(discover_page):
    """Growing must not be able to delete.

    `MM.relate` used to call `Discovery.show(row)` for any card not already on the walk,
    and `show` calls `Force.newWalk(true)`, which empties `nodes`, `links` and `trail`.
    So building a graph, switching to Explore, and clicking a relation on some card you
    had not walked to threw the whole graph away without a word. `discovery.js` carries a
    comment warning about that exact hazard on `focus()`; the Explore path took the
    destructive branch anyway.

    The seed-vs-adopt rule is the fix: reseed only when there is nothing to lose.
    """
    r = discover_page.evaluate("""async () => {
        // Build a real graph by branching a few times.
        Discovery.show(Discovery.rowByName("Ashnod's Altar"));
        await new Promise(r => setTimeout(r, 600));
        for (let i = 0; i < 3; i++) {
            const rows = Force.rows();
            MM.relate(rows[rows.length - 1], 'similar');
            await new Promise(r => setTimeout(r, 500));
        }
        const before = Force.rows().slice();
        if (before.length < 8) return {tooSmall: before.length};

        // Find a card that is NOT on the graph — the destructive case.
        let outsider = -1;
        for (let i = 0; i < Discovery.index.length && outsider < 0; i += 97) {
            if (!Force.hasRow(i) && Discovery.counts(i).similar > 0) outsider = i;
        }

        // Go to the atlas and grow from it, exactly as a user would.
        document.getElementById('modeSelect').value = 'explore';
        MM.setMode('explore');
        await new Promise(r => setTimeout(r, 3000));
        MM.relate(outsider, 'similar');
        await new Promise(r => setTimeout(r, 1500));

        const after = Force.rows();
        const kept = before.filter(x => after.indexOf(x) !== -1);
        return {
            before: before.length, after: after.length,
            kept: kept.length, lost: before.length - kept.length,
            outsiderJoined: Force.hasRow(outsider),
            grew: after.length > before.length,
        };
    }""")
    assert discover_page.js_errors == []
    assert not r.get("tooSmall"), f"could not build a graph to test with: {r}"
    assert r["lost"] == 0, (
        f"growing from the atlas destroyed {r['lost']} of {r['before']} nodes"
    )
    assert r["outsiderJoined"], "the card grown from did not join the graph"
    assert r["grew"], "the graph did not grow"


def test_an_empty_graph_still_seeds_from_the_atlas(discover_page):
    """The other half of the rule: with nothing on the walk there is nothing to lose, so
    a relation clicked in Explore must still seed rather than no-op."""
    r = discover_page.evaluate("""async () => {
        Force.newWalk(true);
        await new Promise(r => setTimeout(r, 300));
        document.getElementById('modeSelect').value = 'explore';
        MM.setMode('explore');
        await new Promise(r => setTimeout(r, 3000));
        const row = Discovery.rowByName("Ashnod's Altar");
        MM.relate(row, 'similar');
        await new Promise(r => setTimeout(r, 1500));
        return {nodes: Force.nodeCount, seeded: Force.hasRow(row)};
    }""")
    assert discover_page.js_errors == []
    assert r["seeded"] and r["nodes"] > 1, f"an empty graph did not seed: {r}"


# ── Typed edges on the atlas ────────────────────────────────────────────


def test_the_atlas_draws_typed_edges(page):
    """An edge on the map must be able to say WHAT it is.

    What the atlas had was a `lines` layer: one flattened polyline with a single colour
    for the whole layer, excluded from the quadtree so it could never be hovered. That is
    enough for the Deck Lens's verified-line edges and structurally unable to say that
    this edge is a synergy and that one is an obsolescence — `force.js` owned the only
    real edge model and kept it to itself.

    Asserted by reading pixels rather than by inspecting the layer list, because "the
    layer is present" is exactly the kind of claim that passes while nothing is drawn.
    The three inks come from `Stage.INK`, so the graph and the map now agree on what a
    relation looks like.
    """
    r = page.evaluate("""async () => {
        const row = MM.allData.findIndex(d => d.n === "Ashnod's Altar");
        const a = MM.allData[row];
        const edges = [];
        for (const rel of ['similar', 'synergy', 'obsolete']) {
            for (const nb of Discovery.neighbours(row, rel)) {
                const t = MM.allData[nb.row];
                if (!t) continue;
                edges.push({source: [a.x, a.y], target: [t.x, t.y],
                            rel: nb.relation, d: 1 - (nb.sim || 0.5)});
            }
        }
        const layers = MM.mapRenderer.layers.slice();
        layers.push({mode: 'edges', name: 'relations', edges: edges,
                     curve: 0.12, line: {width: 1.4}, opacity: 0.9});
        MM.mapRenderer.setLayers(layers);
        await new Promise(r => setTimeout(r, 700));

        const cv = MM.mapRenderer.canvas;
        const px = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
        let violet = 0, red = 0;
        for (let i = 0; i < px.length; i += 4) {
            const R = px[i], G = px[i + 1], B = px[i + 2];
            if (R > 140 && R < 200 && G > 95 && G < 145 && B > 185) violet++;
            else if (R > 185 && G > 95 && G < 145 && B > 95 && B < 145) red++;
        }
        return {
            edges: edges.length,
            byRel: edges.reduce((m, e) => (m[e.rel] = (m[e.rel] || 0) + 1, m), {}),
            violet: violet, red: red,
            // An edge layer carries no customdata, so it must stay out of the quadtree
            // and out of the legend — a swatch for it would be a dot standing for a line.
            legendRows: document.querySelectorAll('.map-legend-row').length,
            markerLayers: MM.mapRenderer.layers.filter(l => l.mode !== 'edges'
                          && l.mode !== 'lines' && l.name).length,
            picksACard: MM.mapRenderer.pointCount > 0,
        };
    }""")
    assert page.js_errors == []
    assert r["byRel"].get("synergy", 0) > 0 and r["byRel"].get("similar", 0) > 0
    assert r["violet"] > 20, f"no synergy-inked edge pixels on the map ({r['violet']})"
    assert r["red"] > 0, f"no obsolescence-inked edge pixels ({r['red']})"
    # One row per MARKER layer. Was pinned at 7, the size of the colour palette, which made
    # recolouring the map by supertype look like an edge layer had leaked into the legend.
    assert r["legendRows"] == r["markerLayers"], (
        f"legend has {r['legendRows']} rows for {r['markerLayers']} marker layers — "
        "an edge layer took one")
    assert r["picksACard"], "the quadtree lost its points to the edge layer"


def test_the_orientation_lens_is_live_not_a_snapshot(discover_page):
    """Explore must show the graph you have, not the one you had when you arrived.

    `orientation` used to hold `{rows: Set, label, anchor}` copied out of `Force.rows()`
    at the moment you entered Explore. Anything that changed the graph afterwards was
    invisible until you left and came back — the atlas was a photograph of your walk. That
    is a large part of why it felt inert next to Discover, and no amount of styling would
    have fixed it.

    Membership now reads through `Session` on every render, so growing the graph while
    Explore is showing lights up more of the map.
    """
    r = discover_page.evaluate("""async () => {
        for (let i = 0; i < 3; i++) {
            const rows = Session.rows();
            MM.relate(rows[rows.length - 1], 'similar');
            await new Promise(r => setTimeout(r, 500));
        }
        document.getElementById('modeSelect').value = 'explore';
        MM.setMode('explore');
        await new Promise(r => setTimeout(r, 3500));
        // Arriving in Explore no longer auto-orients — a clean atlas is the entry state,
        // by design. Engage the lens from INSIDE Explore, which is the whole point: what
        // this test measures is that it then tracks the graph rather than freezing it.
        MM.relate(Session.rows()[0], 'similar');
        await new Promise(r => setTimeout(r, 1500));

        const lit = () => {
            const l = MM.mapRenderer.layers.find(
                t => t._isOrientation && t.name && t.name.indexOf('On your graph') === 0);
            return l ? l.x.length : 0;
        };
        const before = lit();
        const beforeSize = Session.size();

        // Grow while Explore is on screen. A snapshot cannot move.
        Session.grow(Session.rows()[0], 'synergy');
        await new Promise(r => setTimeout(r, 600));
        MM.render();
        await new Promise(r => setTimeout(r, 600));
        return {before: before, after: lit(),
                beforeSize: beforeSize, afterSize: Session.size()};
    }""")
    assert discover_page.js_errors == []
    assert r["before"] > 1, f"the lens lit nothing on entry ({r['before']})"
    assert r["afterSize"] > r["beforeSize"], "the graph did not actually grow"
    assert r["after"] > r["before"], (
        f"the lens stayed at {r['before']} while the graph went to {r['afterSize']} — "
        f"it is still a snapshot"
    )


# ── Phase 3: the constellation on the atlas ─────────────────────────────


def test_explore_grows_in_place_and_draws_the_arcs(page):
    """Clicking a relation in Explore grows the constellation THERE.

    The atlas used to be able only to hand you off. First it forked (a relation opened a
    linear browse set, on the reasoning that a scatter plot cannot grow); then it switched
    modes and carried you into the walk. Both meant the same thing: the map could not
    respond to you. Now the card and its relations join the graph and the edges are drawn
    where those cards actually live, so you see reach and position at once — the one thing
    the force layout structurally cannot show.
    """
    r = page.evaluate("""async () => {
        // Pinned to the colour+type map. Which relations earn an arc is per-map and
        // MEASURED (`MAP_ARC_RELATIONS`): this map draws similarity, the ability map
        // deliberately draws none. Inheriting the map from the boot default made that
        // implicit, so changing the default turned a correct renderer into a red test.
        const pinSel = document.getElementById('mapSelect');
        pinSel.value = 'default'; pinSel.dispatchEvent(new Event('change'));
        await new Promise(r => setTimeout(r, 9000));
        const edgeLayer = () => MM.mapRenderer.layers.find(l => l.mode === 'edges');
        const row = MM.allData.findIndex(d => d.n === "Ashnod's Altar");
        MM.selectByName("Ashnod's Altar");
        await new Promise(r => setTimeout(r, 700));
        const before = Session.size();
        MM.relate(row, 'similar');
        await new Promise(r => setTimeout(r, 1500));
        const el = edgeLayer();
        return {
            mode: document.getElementById('modeSelect').value,
            grew: Session.size() > before,
            arcs: el ? el.edges.length : 0,
            rels: el ? Array.from(new Set(el.edges.map(e => e.rel))) : [],
            // Arcs must terminate at the cards' REAL atlas positions — that is the whole
            // claim the picture is making.
            anchored: el ? el.edges.every(e => {
                const t = MM.allData.find(d => Math.abs(d.x - e.target[0]) < 1e-6 &&
                                               Math.abs(d.y - e.target[1]) < 1e-6);
                return !!t;
            }) : false,
        };
    }""")
    assert page.js_errors == []
    assert r["mode"] == "explore", "growing from the atlas left the atlas"
    assert r["grew"], "the graph did not grow"
    assert r["arcs"] > 0, "no relation arcs were drawn on the map"
    assert r["rels"] == ["similar"]
    assert r["anchored"], "an arc did not end at a real card position"


def test_synergy_is_never_drawn_as_an_atlas_arc(page):
    """Measured, not chosen.

    Median synergy edge length is 0.95x a random pair on the default map and 1.04x on the
    ability map — indistinguishable from noise in world space. That is correct rather than
    broken: synergy is *complementary*, so partners belong in different regions by
    construction (blink finds an ETB creature). It is orthogonal to every 2-D projection
    here, so drawing it as an arc would be drawing a random line and calling it structure.

    The partners still join the graph. The affordance is the force layout, where adjacency
    IS the geometry, and the status line says so.
    """
    r = page.evaluate("""async () => {
        const edgeLayer = () => MM.mapRenderer.layers.find(l => l.mode === 'edges');
        const row = MM.allData.findIndex(d => d.n === "Ashnod's Altar");
        MM.relate(row, 'similar');
        await new Promise(r => setTimeout(r, 1200));
        const arcsBefore = edgeLayer() ? edgeLayer().edges.length : 0;
        MM.relate(row, 'synergy');
        await new Promise(r => setTimeout(r, 1500));
        const el = edgeLayer();
        return {
            synergyLinks: Session.links().filter(l => l.rel === 'synergy').length,
            arcsBefore: arcsBefore,
            arcsAfter: el ? el.edges.length : 0,
            arcRels: el ? Array.from(new Set(el.edges.map(e => e.rel))) : [],
            status: document.getElementById('status').textContent,
        };
    }""")
    assert page.js_errors == []
    assert r["synergyLinks"] > 0, "synergy did not add links to the graph at all"
    assert "synergy" not in r["arcRels"], "a synergy edge was drawn on the map"
    assert r["arcsAfter"] == r["arcsBefore"], "the synergy branch added arcs"
    assert "graph" in r["status"], f"the status did not point at the graph: {r['status']}"


def test_the_ability_map_draws_no_similarity_arcs(page):
    """On the ability map a card's similar neighbours are 0.27u apart on a 71u map — 97% of
    them inside 5% of the atlas. An arc there is a single pixel pretending to be
    information, so none is drawn and the status points at drill, which already exists and
    is the honest answer to "these are all on top of each other"."""
    r = page.evaluate("""async () => {
        // Pinned to the colour+type map. Which relations earn an arc is per-map and
        // MEASURED (`MAP_ARC_RELATIONS`): this map draws similarity, the ability map
        // deliberately draws none. Inheriting the map from the boot default made that
        // implicit, so changing the default turned a correct renderer into a red test.
        const pinSel = document.getElementById('mapSelect');
        pinSel.value = 'default'; pinSel.dispatchEvent(new Event('change'));
        await new Promise(r => setTimeout(r, 9000));
        const edgeLayer = () => MM.mapRenderer.layers.find(l => l.mode === 'edges');
        const row = MM.allData.findIndex(d => d.n === "Ashnod's Altar");
        MM.relate(row, 'similar');
        await new Promise(r => setTimeout(r, 1200));
        const onDefault = edgeLayer() ? edgeLayer().edges.length : 0;

        const ms = document.getElementById('mapSelect');
        ms.value = 'ability'; ms.dispatchEvent(new Event('change'));
        await new Promise(r => setTimeout(r, 15000));
        const held = Session.size();
        MM.relate(row, 'similar');
        await new Promise(r => setTimeout(r, 1500));
        const el = edgeLayer();
        return {onDefault: onDefault, onAbility: el ? el.edges.length : 0,
                held: held, stillHeld: Session.size(),
                status: document.getElementById('status').textContent};
    }""")
    assert page.js_errors == []
    assert r["onDefault"] > 0, "the default map drew no arcs to compare against"
    assert r["onAbility"] == 0, f"the ability map drew {r['onAbility']} similarity arcs"
    assert r["stillHeld"] >= r["held"], "the map switch lost the graph"
    assert "drill" in r["status"].lower(), f"no drill affordance offered: {r['status']}"


def test_only_one_surface_draws_per_mode(page):
    """Discover must not leave the atlas drawing underneath it.

    The mode CSS named `.js-plotly-plot`, `.plot-container` and `.svg-container` — Plotly
    elements that stopped existing when Plotly was deleted. Nothing then hid the canvas
    that replaced them, so in the graph modes the 34,322-point atlas kept drawing under the
    graph with its region labels and legend still on screen. Both views at once.

    It survived a mode-by-mode check because `.map-canvas` is created lazily on the first
    Explore render: boot straight into Discover and there is nothing to leak. The bug only
    appears once you have visited Explore and come back, which is the ordinary way to use
    it — so this test goes through Explore FIRST on purpose.
    """
    r = page.evaluate("""async () => {
        const plot = document.getElementById('plot');
        const shown = sel => {
            const el = plot.querySelector(sel);
            return !!el && getComputedStyle(el).display !== 'none';
        };
        const look = mode => ({
            mode: mode,
            map: shown('.map-canvas'),
            force: shown('.force-canvas'),
            labels: Array.from(document.querySelectorAll('.map-label'))
                         .filter(e => e.offsetParent !== null).length,
        });
        // Explore first, so `.map-canvas` exists before the graph modes are entered.
        const out = [look('explore')];
        for (const m of ['discover', 'explore']) {
            document.getElementById('modeSelect').value = m;
            MM.setMode(m);
            await new Promise(r => setTimeout(r, 2800));
            out.push(look(m));
        }
        return out;
    }""")
    assert page.js_errors == []
    by = {row["mode"]: row for row in r}
    for m in ("discover",):
        assert not by[m]["map"], f"the atlas was still drawing in {m}"
        assert by[m]["labels"] == 0, f"{by[m]['labels']} region labels survived into {m}"
        assert by[m]["force"], f"the graph canvas was not drawing in {m}"
    # And coming back restores it — hiding must not cost the atlas its state.
    back = r[-1]
    assert back["map"] and not back["force"], "returning to Explore did not restore the map"
    assert back["labels"] > 0, "region labels did not come back with the atlas"


def test_clicking_a_cluster_label_zooms_and_filters(page):
    """A label click is a camera move, not a re-layout.

    It used to run DRILL, which is a different thing wearing the same gesture: drill
    re-embeds the subset from the 128-d vectors with stress majorization, so the points fly
    out of their world positions over 90 frames and land somewhere new. Informative when
    you asked for local structure; disorienting when you clicked a label expecting to look
    closer. It also left the map uninteractable, because the drill animation pushes new
    coordinates through `updateLayerBy` while the quadtree still holds the world positions
    it was built from — so every hit-test was against where the cards used to be.

    Now the camera frames the region's real extent and the map draws only its members. The
    positions never move, so picking keeps working, which is what this asserts last. Drill
    is still reachable from the toolbar and from box-select, where a re-layout is an
    explicit request.
    """
    r = page.evaluate("""async () => {
        const span = () => {
            const c = MM.mapRenderer.getCamera();
            return Math.abs(c.x[1] - c.x[0]);
        };
        const points = () => MM.mapRenderer.layers
            .filter(l => l.customdata && l.mode !== 'edges')
            .reduce((n, l) => n + l.x.length, 0);

        const before = {span: span(), points: points()};
        // Label placement is a render pass, not part of boot: under `-n 4` this
        // ran before the first pass finished and threw on `null.click()`, which
        // reads as a broken selector rather than as a race. The camera waits
        // below were already fixed; this one was simply the earlier of the two.
        let label = null;
        for (let i = 0; i < 200 && !label; i++) {
            label = document.querySelector('.map-label');
            if (!label) await new Promise(r => setTimeout(r, 50));
        }
        if (!label) return {error: 'no .map-label was ever placed'};
        label.click();
        // Wait for the FOCUS and for the camera transition to settle, rather than
        // for 2500 ms — under `-n 4` that expired mid-transition and the span
        // assertion failed on a camera still on its way in.
        for (let i = 0; i < 80 && !MM.regionFocus; i++) {
            await new Promise(r => setTimeout(r, 50));
        }
        let last = -1;
        for (let i = 0; i < 80 && Math.abs(span() - last) > 0.01; i++) {
            last = span();
            await new Promise(r => setTimeout(r, 50));
        }

        const members = MM.regionFocus ? MM.regionFocus.rows.size : 0;
        // Picking must still work: same points, same coordinates, closer camera.
        const row = MM.regionFocus ? Array.from(MM.regionFocus.rows)[0] : -1;
        const d = MM.allData[row];
        const p = MM.mapRenderer.dataToPixel(d.x, d.y);
        const picked = MM.mapRenderer.pick(p[0], p[1], 12);

        const after = {span: span(), points: points(), members: members,
                       picked: picked === row,
                       drilling: typeof Drill !== 'undefined' && Drill.isActive()};

        document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
        for (let i = 0; i < 60 && MM.regionFocus; i++) {
            await new Promise(r => setTimeout(r, 50));
        }
        last = -1;
        for (let i = 0; i < 60 && Math.abs(span() - last) > 0.01; i++) {
            last = span();
            await new Promise(r => setTimeout(r, 50));
        }
        return {before: before, after: after,
                escaped: {span: span(), points: points(), focused: !!MM.regionFocus}};
    }""")
    assert page.js_errors == []
    assert not r.get("error"), r.get("error")
    assert r["after"]["span"] < r["before"]["span"] / 2, "the camera did not zoom to the region"
    # SPOTLIT, not filtered: every point stays drawn and the non-members recede. Removing
    # them left a cluster alone in a void, which answers none of the questions the atlas
    # exists for. `test_focusing_a_region_dims_the_map_instead_of_erasing_it` asserts the
    # ink; this asserts the point count is untouched.
    assert r["after"]["members"] > 0, "no region was focused"
    assert r["after"]["points"] == r["before"]["points"], (
        f"focusing dropped points ({r['before']['points']} -> {r['after']['points']}) — "
        f"non-members must dim, not disappear"
    )
    assert not r["after"]["drilling"], "a label click started a drill"
    assert r["after"]["picked"], "the map stopped hit-testing after focusing a region"
    assert not r["escaped"]["focused"]
    assert r["escaped"]["points"] == r["before"]["points"], "Escape did not restore the map"
    assert abs(r["escaped"]["span"] - r["before"]["span"]) < 1, "Escape did not restore the camera"


def test_a_settled_drill_is_still_clickable(page):
    """After a drill settles, the map must hit-test against where the cards ARE.

    The quadtree signature is deliberately cheap — layer lengths plus endpoint ids —
    because a rebuild costs 23.5 ms and `setLayers` runs on every filter and search
    keystroke. The cost of cheap is that it cannot see positions move, and a drill moves
    every position: it mutates coordinates in place through `updateLayerBy` for 90 frames
    while every field in the signature stays identical. So the tree went on answering with
    the world-seeded positions the drill started from, and hovering or clicking a drilled
    card hit whatever used to be at those coordinates, or nothing.

    `reindex()` is the explicit "positions moved" signal, called once at settle and never
    per frame.
    """
    r = page.evaluate("""async () => {
        const rd = await MM.getRegionData();
        const reg = rd.regions.find(r => r.level === 1 && r.count > 100 && r.count < 600);
        await Drill.enterRegion(reg.id);
        await new Promise(r => setTimeout(r, 4000));   // the 90-frame settle
        const layer = MM.mapRenderer.layers.find(l => l._isDrill);
        if (!layer) return {noLayer: true};
        let tried = 0, hits = 0;
        for (let i = 0; i < Math.min(12, layer.x.length); i++) {
            const p = MM.mapRenderer.dataToPixel(layer.x[i], layer.y[i]);
            tried++;
            if (MM.mapRenderer.pick(p[0], p[1], 14) === layer.customdata[i]) hits++;
        }
        return {drilling: Drill.isActive(), points: layer.x.length,
                tried: tried, hits: hits};
    }""")
    assert page.js_errors == []
    assert not r.get("noLayer"), "the drill layer never appeared"
    assert r["drilling"] and r["points"] > 50
    assert r["hits"] == r["tried"] > 0, (
        f"only {r['hits']}/{r['tried']} drilled cards were pickable — the quadtree is "
        f"still holding pre-drill positions"
    )


def test_selecting_a_card_does_not_leave_the_region(page):
    """Clicking a point while a region is focused must not move the camera.

    `clearSelection()` was doing two jobs. Every plain click runs "replace the selection"
    first, and that function also carried the Escape peel chain — so clicking a point while
    a region was focused cleared the focus and refit the camera, and the map zoomed out from
    under you as you selected a card. The `orientation` branch had done the same thing for
    longer and less visibly: clicking a point while the lens was on just turned the lens
    off.

    They are two jobs and are now two functions. `clearSelection` clears the selection;
    `escapeOnce` peels one layer, and only the Escape key calls it.
    """
    r = page.evaluate("""async () => {
        const span = () => {
            const c = MM.mapRenderer.getCamera();
            return Math.abs(c.x[1] - c.x[0]);
        };

        for (let i = 0; i < 200 && !document.querySelector('.map-label'); i++) {
            await new Promise(r => setTimeout(r, 50));
        }
        const label = document.querySelector('.map-label');
        if (!label) return {error: 'no .map-label was ever placed'};
        label.click();
        await new Promise(r => setTimeout(r, 2500));
        const points = () => MM.mapRenderer.layers
            .filter(l => l.customdata && l.mode !== 'edges')
            .reduce((n, l) => n + l.x.length, 0);
        const zoomed = {span: span(), region: !!MM.regionFocus, points: points()};

        const row = Array.from(MM.regionFocus.rows)[0];
        const d = MM.allData[row];
        const p = MM.mapRenderer.dataToPixel(d.x, d.y);
        const rect = document.getElementById('plot').getBoundingClientRect();
        MM.mapRenderer.canvas.dispatchEvent(new MouseEvent('click',
            {bubbles: true, clientX: rect.left + p[0], clientY: rect.top + p[1]}));
        await new Promise(r => setTimeout(r, 1200));
        const picked = {span: span(), region: !!MM.regionFocus, points: points(),
                        card: (document.querySelector('#detailInner h2') || {}).textContent};

        // Escape still peels the region — the behaviour that was borrowed, not deleted.
        document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
        await new Promise(r => setTimeout(r, 1200));
        return {zoomed: zoomed, picked: picked,
                escaped: {span: span(), region: !!MM.regionFocus}};
    }""")
    assert page.js_errors == []
    assert not r.get("error"), r.get("error")
    assert r["zoomed"]["region"] and r["zoomed"]["span"] < 40
    assert r["picked"]["region"], "selecting a card cleared the region focus"
    # Deliberately NOT an equality check on the span. Opening the detail panel narrows
    # `#plot` at this viewport, so the same zoom transform covers less world width — two
    # viewports, not two cameras. (Verified in a wider window where the panel overlays
    # instead of pushing: the span is identical to three decimals.) The bug was a reset to
    # the global view, so that is what this asserts.
    assert r["picked"]["span"] < r["zoomed"]["span"] * 1.5, (
        f"the camera zoomed out on select: {r['zoomed']['span']:.2f} -> "
        f"{r['picked']['span']:.2f}"
    )
    assert r["picked"]["points"] == r["zoomed"]["points"] + 1, (
        "selecting a card changed which cards are drawn (+1 is the highlight layer)"
    )
    assert r["picked"]["card"], "no card was actually selected"
    assert not r["escaped"]["region"] and r["escaped"]["span"] > r["picked"]["span"] * 2, (
        "Escape no longer peels the region"
    )


# ── Build: one mode for a set of cards ──────────────────────────────────


def test_build_opens_on_the_graph_and_keeps_it_across_views(page):
    """Build opens on the graph, and switching to the map and back must not cost you it.

    `Force.enter` with an explicit seed takes the rebuild path, so seeding unconditionally
    on every view change threw away everything branched to — measured at six explored cards
    lost on one round trip. Same hazard the relation buttons had, one file further on.
    `seedGraph` restores instead when the graph already holds the deck.
    """
    r = page.evaluate("""async () => {
        document.getElementById('modeSelect').value = 'build';
        MM.setMode('build');
        await new Promise(r => setTimeout(r, 3000));
        await Build.select('edgar-vampires');
        await new Promise(r => setTimeout(r, 9000));
        const seeded = {view: Build.view, nodes: Force.nodeCount,
                        m: Force.membership()};

        MM.relate(Force.rows()[5], 'similar');
        await new Promise(r => setTimeout(r, 2500));
        const grown = Force.nodeCount;

        Build.setView('map');
        await new Promise(r => setTimeout(r, 2500));
        const onMap = {view: Build.view,
                       forceMode: document.getElementById('plot').classList.contains('force-mode'),
                       layers: MM.mapRenderer.layers.length};
        Build.setView('graph');
        await new Promise(r => setTimeout(r, 3000));
        return {seeded, grown, onMap,
                back: {nodes: Force.nodeCount, m: Force.membership()}};
    }""")
    assert page.js_errors == []
    assert r["seeded"]["view"] == "graph", "Build did not open on the graph"
    assert r["seeded"]["nodes"] > 50
    assert r["seeded"]["m"]["commander"] == 1, "the commander is not ringed"
    assert r["seeded"]["m"]["explored"] == 0, "a freshly loaded deck has nothing explored yet"
    assert r["grown"] > r["seeded"]["nodes"], "branching did not grow the graph"
    # The map view is the old Deck Lens: the atlas with the deck lit.
    assert not r["onMap"]["forceMode"] and r["onMap"]["layers"] > 5
    assert r["back"]["nodes"] >= r["grown"], (
        f"the view round trip lost cards: {r['grown']} -> {r['back']['nodes']}"
    )
    assert r["back"]["m"]["explored"] > 0, "the cards you found did not survive the round trip"


def test_the_cards_you_brought_are_inked_differently(page):
    """A pasted pool must look like a pool, not like something you wandered into.

    `Discovery.importText` did not pass `opts.deck`, so a pasted list — which is how a bulk
    pool arrives — got a pinned commander and none of the visual language: no gold ring, no
    deck ink, no warm deck edges. `loadDeck` passed it and a paste did not, which is the
    exact path a 300-card pool takes.
    """
    r = page.evaluate("""async () => {
        const d = await (await fetch('../data/decks/heliod/cards.json')).json();
        const text = d.cards
                            .map(c => (c.quantity || 1) + ' ' + c.name).join('\\n');
        document.getElementById('modeSelect').value = 'discover';
        MM.setMode('discover');
        await new Promise(r => setTimeout(r, 2000));
        const distinct = new Set(d.cards.map(c => c.name)).size;
        const res = Discovery.importText(text);
        await new Promise(r => setTimeout(r, 9000));
        const imported = Force.membership();
        MM.relate(Force.rows()[3], 'similar');
        await new Promise(r => setTimeout(r, 2500));
        return {resolved: res.resolved, missing: res.missing.length, distinct: distinct,
                imported: imported, after: Force.membership()};
    }""")
    assert page.js_errors == []
    # A card is a POSITION on the graph, so eleven Islands are one node — `importText`
    # dedupes and the quantity rides along in the panel. Compare against distinct names,
    # not the 100-card total, or this asserts the copies rule backwards.
    assert r["missing"] == 0, "a published deck should resolve completely"
    assert r["resolved"] == r["distinct"], (
        f"{r['resolved']} rows for {r['distinct']} distinct names"
    )
    assert r["imported"]["deck"] == r["resolved"], (
        "the pasted cards were not marked as brought — opts.deck was not passed"
    )
    assert r["imported"]["explored"] == 0
    assert r["after"]["explored"] > 0, "branching added nothing"
    assert r["after"]["deck"] == r["imported"]["deck"], (
        "cards you found were counted as cards you brought"
    )


def test_the_commander_is_one_card_and_changing_it_keeps_the_graph(page):
    """One card, read from Session by the ring, the colour identity and the brief.

    Offered on legendary creatures only, because that is the rule rather than a
    preference. `Force.setCommander` re-inks rather than re-seeding — changing your mind
    about a commander must not cost you the graph.
    """
    r = page.evaluate("""async () => {
        document.getElementById('modeSelect').value = 'discover';
        MM.setMode('discover');
        await new Promise(r => setTimeout(r, 2000));
        const edgar = MM.allData.findIndex(d => d.n === 'Edgar Markov');
        const bolt = MM.allData.findIndex(d => d.n === 'Lightning Bolt');

        Discovery.show(edgar);
        await new Promise(r => setTimeout(r, 1200));
        const onLegend = document.getElementById('deckInner').innerHTML.indexOf('Set as commander') !== -1;

        for (let i = 0; i < 2; i++) {
            MM.relate(Force.rows()[Force.nodeCount - 1], 'similar');
            await new Promise(r => setTimeout(r, 700));
        }
        const before = Force.nodeCount;
        MM.setCommander(edgar);
        await new Promise(r => setTimeout(r, 1200));
        const set = {isCommander: Session.commander === edgar,
                     ringed: Force.membership().commander,
                     nodes: Force.nodeCount};

        Discovery.focus(bolt);
        await new Promise(r => setTimeout(r, 900));
        const onSpell = document.getElementById('deckInner').innerHTML.indexOf('Set as commander') !== -1;
        return {onLegend, onSpell, before, set};
    }""")
    assert page.js_errors == []
    assert r["onLegend"], "no way to name a legendary creature as commander"
    assert not r["onSpell"], "a non-creature was offered as a commander"
    assert r["set"]["isCommander"] and r["set"]["ringed"] == 1
    assert r["set"]["nodes"] >= r["before"], "naming a commander rebuilt the graph"


def test_the_panel_belongs_to_the_mode_not_the_engine(page):
    """One graph engine, two owners, and the engine has to ask which.

    When The Walk was deleted, `Force.renderPanel` was collapsed to "Discovery always owns
    the panel". That was true while Discover was the only thing seeding the graph. It
    stopped being true the moment Build seeded the same engine: every reheat, branch and
    pin repainted Build's roles, curve and verified lines with Discover's landing controls.

    Three call sites had inherited the same assumption — `renderPanel`, `Force.pinCard` /
    `branchFrom`, and `MM.relate` — all reaching for `Discovery.focus`, which *renders*.
    Noting which card is open (`Discovery.setCurrent`) and deciding who draws the panel are
    two different jobs.
    """
    r = page.evaluate("""async () => {
        const heading = () => (document.querySelector('#deckInner h2') || {}).textContent;
        document.getElementById('modeSelect').value = 'build';
        MM.setMode('build');
        await new Promise(r => setTimeout(r, 3000));
        await Build.select('edgar-vampires');
        await new Promise(r => setTimeout(r, 9000));
        const loaded = {panel: heading(), nodes: Force.nodeCount};

        // Branching is what reheats the graph, which is what repainted the panel.
        MM.relate(Force.rows()[5], 'similar');
        await new Promise(r => setTimeout(r, 2500));
        const branched = {panel: heading(), nodes: Force.nodeCount};

        document.getElementById('modeSelect').value = 'discover';
        MM.setMode('discover');
        await new Promise(r => setTimeout(r, 3500));
        const discover = {panel: heading(), nodes: Force.nodeCount,
                          deck: Force.membership().deck};
        return {loaded, branched, discover};
    }""")
    assert page.js_errors == []
    assert r["loaded"]["panel"] == "Build", "Build did not own its own panel"
    assert r["branched"]["panel"] == "Build", (
        "branching handed Build's panel to Discover — the engine assumed an owner"
    )
    assert r["branched"]["nodes"] > r["loaded"]["nodes"], "the branch did not grow the graph"
    assert r["discover"]["panel"] == "Discover"


def test_leaving_build_hands_the_canvas_back(page):
    """Build owns its graph; a walk you grew is yours.

    Everywhere else the rule is "growing must never be able to delete" — a walk survives a
    round trip through Explore. A loaded DECK is not that: it is Build's subject, and
    leaving it behind meant Discover opened on someone else's 97-card deck with the landing
    card buried in it.

    Explore is exempt on purpose. It is a LENS on whatever graph is current, not a
    workspace, so clearing on the way there would empty the very thing it exists to show.
    """
    r = page.evaluate("""async () => {
        document.getElementById('modeSelect').value = 'build';
        MM.setMode('build');
        await new Promise(r => setTimeout(r, 3000));
        await Build.select('edgar-vampires');
        await new Promise(r => setTimeout(r, 9000));
        const inBuild = Force.nodeCount;

        document.getElementById('modeSelect').value = 'discover';
        MM.setMode('discover');
        await new Promise(r => setTimeout(r, 3500));
        const discover = {nodes: Force.nodeCount, deck: Force.membership().deck};

        // Now grow a walk of your own, and check Explore does NOT take it away.
        for (let i = 0; i < 2; i++) {
            MM.relate(Force.rows()[Force.nodeCount - 1], 'similar');
            await new Promise(r => setTimeout(r, 700));
        }
        const mine = Force.nodeCount;
        document.getElementById('modeSelect').value = 'explore';
        MM.setMode('explore');
        await new Promise(r => setTimeout(r, 3000));
        return {inBuild, discover, mine, throughExplore: Session.size()};
    }""")
    assert page.js_errors == []
    assert r["inBuild"] > 50
    assert r["discover"]["deck"] == 0, "Build's deck followed you into Discover"
    assert r["discover"]["nodes"] <= 2, (
        f"Discover opened on {r['discover']['nodes']} nodes — it should be a landing card"
    )
    assert r["mine"] > r["discover"]["nodes"], "could not grow a walk after leaving Build"
    assert r["throughExplore"] >= r["mine"], (
        "Explore cleared the walk — it is a lens, not a workspace"
    )


def test_the_brief_is_the_schema_build_deck_reads(discover_page):
    """The export must BE `brief.json`, not a description of one.

    It used to emit `{generated_by, card_count, cards, commander_candidates, next_step}`,
    which is none of the shape `pilot/build_deck.py:load_brief` requires — so every export
    had to be hand-translated before the loop could run, and the browser's own answers
    (which card IS the commander, what you kept versus what you found) were thrown away
    and re-derived by the agent.

    Two rules come from the Python side and are honoured rather than guessed: colour
    identity is DERIVED from the commander and never authored, so it rides in the
    provenance block the builder ignores; and budget is unsupported because prices are
    stripped from the card data.

    Round-tripped for real: `load_brief` accepts this document unchanged, tolerates the
    `_manamap` and `next_step` extras, and its derived identity for Edgar Markov ({B,R,W})
    matches what the browser emits.
    """
    # Reads MM.allData, which this fixture deliberately does not wait for.
    await_projection(discover_page)
    r = discover_page.evaluate("""async () => {
        const edgar = MM.allData.findIndex(d => d.n === 'Edgar Markov');

        // With no commander the brief must refuse rather than emit an unusable document.
        const blocked = Discovery.brief();

        Discovery.show(edgar);
        await new Promise(r => setTimeout(r, 1200));
        MM.setCommander(edgar);
        await new Promise(r => setTimeout(r, 600));
        for (let i = 0; i < 3; i++) {
            MM.relate(Force.rows()[Force.nodeCount - 1], 'similar');
            await new Promise(r => setTimeout(r, 700));
        }
        Force.rows().filter(r => r !== edgar).slice(0, 6).forEach(function (r) {
            if (!Session.library.has(r)) Session.library.toggle(r);
        });
        const b = Discovery.brief();
        return {
            blocked: {commander: blocked.commander, hasBlocked: !!blocked._manamap.blocked},
            keys: Object.keys(b),
            slug: b.slug, commander: b.commander, bracket: b.bracket,
            mustInclude: b.must_include, mustExclude: b.must_exclude,
            ci: b._manamap.colour_identity,
            budget: b._manamap.budget,
            sources: Array.from(new Set(b._manamap.pool.map(p => p.source))).sort(),
            commanderInMustInclude: b.must_include.indexOf('Edgar Markov') !== -1,
        };
    }""")
    assert discover_page.js_errors == []

    # Blocked without a commander — `build-deck` cannot start without one.
    assert r["blocked"]["commander"] is None and r["blocked"]["hasBlocked"]

    # The three keys `load_brief` requires, in the shape it requires them.
    for key in ("slug", "commander", "bracket", "must_include", "must_exclude"):
        assert key in r["keys"], f"brief is missing {key} — load_brief will reject it"
    assert r["slug"] == "edgar-markov", f"slug not derived from the commander: {r['slug']}"
    assert r["commander"] == "Edgar Markov"
    assert r["bracket"] in range(1, 6), "bracket must be 1-5"
    assert isinstance(r["mustInclude"], list) and isinstance(r["mustExclude"], list)
    assert len(r["mustInclude"]) == 6, "the library is what must_include means"
    assert not r["commanderInMustInclude"], (
        "the commander occupies its own slot and must not also be in must_include"
    )

    # Derived, informational, and correct — Edgar Markov is Mardu.
    assert sorted(r["ci"]) == ["B", "R", "W"]
    assert "unsupported" in r["budget"], "budget must say it is unsupported, not guess"

    # What you brought vs what you found: the graph knows, so the agent should not guess.
    assert r["sources"] == ["found", "kept"], f"pool provenance lost: {r['sources']}"


def test_a_box_select_can_become_a_graph(page):
    """A lassoed set is a set of cards, and a set of cards can be walked.

    This was `Force.seedFrom()`, reachable only by entering The Walk while a selection was
    live. Deleting that mode did not delete the capability — it made it *unreachable*:
    `seedFrom` still worked and nothing called it, which is the quietest kind of
    regression. The browse panel offers it now.
    """
    r = page.evaluate("""async () => {
        const rows = [];
        for (let i = 0; i < MM.allData.length; i += 430) rows.push(i);
        await MM.enterBrowse(rows.slice(0, 60), 'Test lasso');
        await new Promise(r => setTimeout(r, 1800));
        const offered = !!document.querySelector('[onclick^="MM.growFromBrowse"]');

        MM.growFromBrowse();
        await new Promise(r => setTimeout(r, 9000));
        return {offered: offered, mode: MM.mode, nodes: Force.nodeCount,
                links: Force.linkCount,
                status: document.getElementById('status').textContent};
    }""")
    assert page.js_errors == []
    assert r["offered"], "a browse set offers no way to grow a graph from it"
    assert r["mode"] == "discover", "growing did not carry you to where growing happens"
    assert r["nodes"] == 60, f"seeded {r['nodes']} of 60 lassoed cards"
    assert r["links"] > 0, "a multi-seed graph should have intra-set links"
    assert "Test lasso" in r["status"], "the graph forgot where it came from"


def test_naming_a_commander_refreshes_build(page):
    """`MM.setCommander` calls `Build.onCommanderChange()` behind a `&&` guard, and the
    guard was hiding the fact that the method did not exist — so naming a commander in
    Build changed the ring and left the panel's legality read stale. Colour identity is
    derived from the commander, so the panel has to follow it."""
    r = page.evaluate("""async () => {
        document.getElementById('modeSelect').value = 'build';
        MM.setMode('build');
        await new Promise(r => setTimeout(r, 3000));
        await Build.select('heliod');
        await new Promise(r => setTimeout(r, 9000));
        return {defined: typeof Build.onCommanderChange,
                sections: document.querySelectorAll('#deckInner .deck-section-title').length};
    }""")
    assert page.js_errors == []
    assert r["defined"] == "function", (
        "Build.onCommanderChange is called by MM.setCommander but never defined"
    )
    assert r["sections"] >= 5, "Build's panel did not render its analysis"


def test_fit_frames_a_small_graph_instead_of_leaving_it_tiny(discover_page):
    """`Fit` must fill the canvas, and the cap must not stop it short.

    The zoom-in was capped at 1.6 on the reasoning that "a 59-unit-wide graph blown up to
    fill a 1439px canvas is not more readable, just bigger". That is wrong here, because
    node radius and label text are drawn in SCREEN space (`r / transform.k`) — zooming in
    enlarges nothing, it only spreads the nodes apart, which is exactly what a label needs.

    It bites hardest on the commonest state in Discover: a landing card plus one branch.
    Measured at 7 nodes spanning 20.7x26 world units — filling the canvas wants k≈19, the
    cap allowed 1.6, so the graph rendered ~33px wide and every label collided. The cap is
    now the zoom behaviour's own ceiling, so a fit may go wherever a drag could.
    """
    r = discover_page.evaluate("""async () => {
        Discovery.show(Discovery.rowByName('Sol Ring'));
        await new Promise(r => setTimeout(r, 900));
        MM.relate(Force.rows()[0], 'similar');
        await new Promise(r => setTimeout(r, 4000));
        // Freeze BEFORE measuring: the simulation keeps expanding the bbox, and reading it
        // mid-settle is how this was misdiagnosed twice.
        Force.freeze();
        await new Promise(r => setTimeout(r, 400));
        const b = Force.bbox();
        const cv = document.getElementById('forceCanvas');
        const wanted = Math.min(cv.clientWidth / (b.w * 1.18),
                                cv.clientHeight / (b.h * 1.18));
        Force.fit();
        await new Promise(r => setTimeout(r, 1200));
        return {nodes: Force.nodeCount, bw: b.w, bh: b.h,
                wanted: wanted, labels: Force.labelCount,
                onScreenWidthAtOldCap: b.w * 1.6};
    }""")
    assert discover_page.js_errors == []
    assert 2 < r["nodes"] < 30, "this test is about a SMALL graph"
    assert r["wanted"] > 1.6, (
        "this graph is not small enough to exercise the cap — pick a smaller one"
    )
    # Not a fixed pixel count: branching is random, so the bbox varies run to run and an
    # exact threshold is a flake waiting to happen. What is invariant is the RATIO — the
    # old cap framed the graph at a small fraction of what filling the canvas wants.
    assert r["wanted"] / 1.6 > 3, (
        f"the old cap was {r['wanted'] / 1.6:.1f}x too small here; this graph does not "
        f"demonstrate the problem"
    )
    assert r["labels"] >= 3, (
        f"only {r['labels']} labels placed — the fit is still leaving the graph too small "
        f"for any of them to clear each other"
    )


def test_both_panels_draw_the_same_card_header(page):
    """One header, two panels. There were two headers and they had drifted.

    The browse panel's copy lost the loyalty and defense branches and the in-deck control,
    so the SAME planeswalker showed `Loyalty: 4` when selected and nothing when browsed.
    Neither omission was a decision — they were copies that stopped being copied.
    """
    r = page.evaluate("""async () => {
        const q = () => (document.querySelector('#detailInner .viewer-quickstats') || {}).textContent;
        const nHeaders = () => document.querySelectorAll('#detailInner .viewer-header').length;

        MM.selectByName('Teferi, Hero of Dominaria');
        await new Promise(r => setTimeout(r, 1000));
        const selected = {stats: q(), headers: nHeaders()};

        const pw = MM.allData.findIndex(d => d.n === 'Teferi, Hero of Dominaria');
        await MM.enterBrowse([pw, 10, 20, 30, 40], 'Header check');
        await new Promise(r => setTimeout(r, 2000));
        // enterBrowse orders by distance from the set's centroid, so walk to the card
        // rather than assuming it is first — comparing two different cards proves nothing.
        let guard = 0;
        while (MM.browseSet.indices[MM.browseSet.pos] !== pw && guard++ < 10) {
            MM.cycleNext();
            await new Promise(r => setTimeout(r, 300));
        }
        await new Promise(r => setTimeout(r, 600));
        return {
            selected: selected,
            browsed: {stats: q(), headers: nHeaders(),
                      card: MM.allData[MM.browseSet.indices[MM.browseSet.pos]].n,
                      inDeckControl: !!document.querySelector(
                          '#detailInner .btn-add-deck, #detailInner .in-deck-badge')},
        };
    }""")
    assert page.js_errors == []
    assert r["browsed"]["card"] == "Teferi, Hero of Dominaria", "did not reach the card"
    assert "Loyalty" in r["selected"]["stats"], "the selection panel lost loyalty"
    assert r["browsed"]["stats"] == r["selected"]["stats"], (
        f"the two panels disagree about the same card: "
        f"{r['selected']['stats']!r} vs {r['browsed']['stats']!r}"
    )
    assert r["browsed"]["inDeckControl"], "the browse panel still has no in-deck control"
    for where in ("selected", "browsed"):
        assert r[where]["headers"] == 1, f"{where} rendered {r[where]['headers']} headers"


# ── The dossier's issue link ─────────────────────────────────────────────


def _dossier(browser, viz_server, slug):
    page = browser.new_page()
    page.goto(f"{viz_server}/viz/deck.html?deck={slug}")
    page.wait_for_timeout(2500)
    return page


def test_no_live_surface_links_into_the_magazine(browser, viz_server):
    """The magazine is not a product any more, so nothing may invite a pilot in.

    These two surfaces used to carry an Archive link each — the dossier's
    `#issueLink` to `../manuals/<slug>.html` and the workbench's to the
    newsstand. The magazine renderer is NOT deleted and the issues stay on
    disk: they are the record of what was published, and `issue.json` still
    carries the deck's `status`, which `deck_lifecycle` reads. What went is the
    invitation.

    Asserted in a browser rather than by grepping the HTML, because the
    dossier's link was built by `deck-view.js` at render time — a source
    assertion would have passed on a page that still grew one.
    """
    import json
    from manamap.config import DECKS_DIR

    manifest = json.loads((DECKS_DIR / "index.json").read_text())
    slug = manifest["decks"][0]["slug"]

    page = _dossier(browser, viz_server, slug)
    try:
        assert page.query_selector("#issueLink") is None
        hrefs = page.eval_on_selector_all(
            ".head-links a", "els => els.map(e => e.getAttribute('href'))")
        assert not [h for h in hrefs if h and "manuals/" in h and "manuals/p/" not in h], hrefs
    finally:
        page.close()

    page = browser.new_page()
    page.goto(f"{viz_server}/viz/workbench.html")
    page.wait_for_timeout(2500)
    try:
        hrefs = page.eval_on_selector_all(
            "a", "els => els.map(e => e.getAttribute('href'))")
        assert not [h for h in hrefs if h and "manuals/" in h and "manuals/p/" not in h], hrefs
    finally:
        page.close()


def test_the_dossier_hides_a_manual_link_it_cannot_serve(browser, viz_server):
    """A link that 404s is worse than a link that is not there.

    The rule the deleted issue-link tests existed for, moved to the artifact it
    now governs. A deck is loadable as soon as it has a `cards.json`; its
    Pilot's Manual comes later, and `has.page` on the manifest entry is what
    says whether one exists. Reading `d.has` instead of `d.entry.has` once hid
    this link on every deck that had a manual — silently, since a hidden link
    raises nothing.
    """
    import json
    from manamap.config import DECKS_DIR

    manifest = json.loads((DECKS_DIR / "index.json").read_text())
    with_page = [d for d in manifest["decks"]
                 if (d.get("has") or {}).get("page")]
    without = [d for d in manifest["decks"]
               if not (d.get("has") or {}).get("page")]
    if not with_page:
        pytest.skip("no deck in the manifest has a compact page")

    slug = with_page[0]["slug"]
    page = _dossier(browser, viz_server, slug)
    try:
        assert page.is_visible("#manualLink") is True
        assert page.get_attribute("#manualLink", "href") == f"../manuals/p/{slug}.html"
    finally:
        page.close()

    if without:
        page = _dossier(browser, viz_server, without[0]["slug"])
        try:
            assert page.is_visible("#manualLink") is False
            assert page.get_attribute("#manualLink", "href") is None
        finally:
            page.close()


# ── The verified-line spotlight ──────────────────────────────────────────
#
# `Build.focusLine` existed and was wired to the sidebar click, but all it did was
# `MM.mapRenderer.setCamera(...)`. Build defaults to the GRAPH, where the map canvas is
# `display:none` — so the click panned a hidden canvas and changed a status string. Every
# source-assertion test passed: the handler was there, the markup was there, the function
# ran. Nothing was visible. That is why these read state and pixels.

_INK = """
() => {
  const c = document.querySelector('canvas.force-canvas');
  if (!c) return null;
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let lit = 0, sum = 0, green = 0;
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], g = d[i+1], b = d[i+2], a = d[i+3];
    const v = r + g + b;
    if (a > 8 && v > 40) { lit++; sum += v; }
    // The #4CAF50 emphasis ink: green clearly dominant and bright.
    if (a > 60 && g > 110 && g > r + 40 && g > b + 40) green++;
  }
  return { lit, mean: lit ? sum / lit : 0, green };
}
"""


def _build_page(browser, viz_server, slug):
    from conftest_viz import _boot
    page = _boot(browser, viz_server, f"?deck={slug}")
    page.wait_for_function("() => window.Force && Force.nodeCount > 0", timeout=BOOT_TIMEOUT_MS)
    page.wait_for_timeout(1500)
    return page


def _a_deck_with_a_drawable_line():
    import json
    from manamap.config import DECKS_DIR
    manifest = json.loads((DECKS_DIR / "index.json").read_text())
    for d in manifest["decks"]:
        if any(len(v) >= 2 for v in (d.get("stack_cards") or {}).values()):
            return d["slug"]
    return None


def test_verified_lines_become_edges_in_the_graph(browser, viz_server):
    """The graph's links come only from embedding similarity, so two combo pieces that
    are not near-neighbours had no edge at all. A verified line is a claim about which
    cards talk to each other; the graph must be able to say it."""
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck with a drawable verified line")
    page = _build_page(browser, viz_server, slug)
    try:
        assert page.js_errors == []
        assert page.evaluate("Force.verifiedLinkCount") > 0
    finally:
        page.close()


def test_clicking_a_verified_line_spotlights_it(browser, viz_server):
    """Click the line: its rows go under a spotlight and the row marks itself."""
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck with a drawable verified line")
    page = _build_page(browser, viz_server, slug)
    try:
        assert page.evaluate("Build.activeLine") == -1
        page.click(".lens-line")
        page.wait_for_timeout(900)

        assert page.evaluate("Build.activeLine") == 0
        assert page.evaluate("Force.activeLine") is not None
        assert page.evaluate("Force.lineRowCount") >= 2
        assert page.evaluate(
            "document.querySelector('.lens-line').classList.contains('is-on')")
        assert page.js_errors == []
    finally:
        page.close()


# Green ink in a tight box around the spotlit line's OWN cards — the edge and both
# rings are inside it, the rest of the graph is not. See the test below for why the
# whole canvas cannot answer this question.
_NEAR_LINE = """
(rows) => {
  const c = document.querySelector('canvas.force-canvas');
  const W = c.width, H = c.height, dpr = W / c.clientWidth;
  const ns = Force.screenNodes().filter(n => rows.indexOf(n.row) !== -1);
  if (ns.length < 2) return null;
  const xs = ns.map(n => n.x * dpr), ys = ns.map(n => n.y * dpr), pad = 14 * dpr;
  const x0 = Math.max(0, Math.min(...xs) - pad), x1 = Math.min(W - 1, Math.max(...xs) + pad);
  const y0 = Math.max(0, Math.min(...ys) - pad), y1 = Math.min(H - 1, Math.max(...ys) + pad);
  const d = c.getContext('2d').getImageData(0, 0, W, H).data;
  let n = 0;
  for (let y = Math.round(y0); y <= Math.round(y1); y++)
    for (let x = Math.round(x0); x <= Math.round(x1); x++) {
      const i = (y * W + x) * 4, r = d[i], g = d[i+1], b = d[i+2], a = d[i+3];
      if (a > 60 && g > 110 && g > r + 40 && g > b + 40) n++;
    }
  return n;
}
"""


def test_the_spotlight_actually_dims_the_canvas(browser, viz_server):
    """Pixels, not state. A layer being present passes while nothing draws.

    Compared at the SAME camera — clearing does not refit — so the difference is the
    dimming and nothing else.

    **The green ink is measured near the LINE, not across the canvas, and the
    difference is the whole point of this test.** Clearing the spotlight does two
    things at once: it rests the line you were looking at, and it un-mutes every
    OTHER verified line. On goblin-storm those cancel almost exactly — 868 green px
    spotlit against 833 cleared, a ratio of 0.96 that reads as "deselecting does
    nothing" — while the line's own box goes 1024 -> 301. The bounding boxes say it
    plainly: spotlit, green occupies 47x294 px; cleared, it is spread across
    591x687. A whole-canvas count cannot tell "this line stopped shouting" from
    "eleven other lines started whispering", and for a while it reported the second
    as a failure of the first.

    Threshold 0.52, measured on both sides rather than fitted to one. Healthy
    dimming over four runs: 0.204 / 0.255 / 0.422 / 0.439. With the resting ink
    disabled (`Stage.INK.verifiedQuiet = Stage.INK.verified`, the bug this guards):
    0.589 / 0.601 / 0.617 / 0.695. That simulated bug is only PARTIAL — the resting
    state also halves the stroke weight (2.4 -> 1.4 in `weightOf`), which the patch
    leaves intact — so a real regression in both would sit near 1.0 and this has
    more headroom than the numbers suggest. The spread inside each group is
    d3-force layout non-determinism: how many other quiet edges happen to cross the
    box.
    """
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck with a drawable verified line")
    page = _build_page(browser, viz_server, slug)
    try:
        page.click(".lens-line")
        page.wait_for_timeout(1500)          # includes the 450ms fit transition
        spotlit = page.evaluate(_INK)
        rows = page.evaluate("() => Force.spotlitRows")
        near_lit = page.evaluate(_NEAR_LINE, rows)

        page.click(".lens-line")             # clear; camera stays put
        page.wait_for_timeout(900)
        cleared = page.evaluate(_INK)
        # Same rows, same camera — only the styling moved.
        near_off = page.evaluate(_NEAR_LINE, rows)

        assert len(rows) >= 2, rows
        assert near_lit and near_off is not None

        assert spotlit and cleared
        # Lifting the spotlight brings the rest of the deck back: more ink, brighter ink.
        assert cleared["lit"] > spotlit["lit"] * 1.15, (spotlit, cleared)
        assert cleared["mean"] > spotlit["mean"] * 1.05, (spotlit, cleared)
        # And the line itself must stop shouting — measured WHERE THE LINE IS.
        assert near_off < near_lit * 0.52, (near_lit, near_off, rows)
        assert page.js_errors == []
    finally:
        page.close()


def test_clicking_the_same_line_again_clears_it(browser, viz_server):
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck with a drawable verified line")
    page = _build_page(browser, viz_server, slug)
    try:
        page.click(".lens-line"); page.wait_for_timeout(800)
        assert page.evaluate("Force.lineRowCount") >= 2
        page.click(".lens-line"); page.wait_for_timeout(800)
        assert page.evaluate("Build.activeLine") == -1
        assert page.evaluate("Force.lineRowCount") == 0
        assert not page.evaluate(
            "document.querySelector('.lens-line').classList.contains('is-on')")
        assert page.js_errors == []
    finally:
        page.close()


def test_escape_peels_the_spotlight_first(browser, viz_server):
    """Escape peels outermost-first, and a line is the innermost thing you are inside."""
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck with a drawable verified line")
    page = _build_page(browser, viz_server, slug)
    try:
        page.click(".lens-line"); page.wait_for_timeout(800)
        assert page.evaluate("Build.activeLine") == 0
        page.keyboard.press("Escape"); page.wait_for_timeout(600)
        assert page.evaluate("Build.activeLine") == -1
        assert page.evaluate("Force.lineRowCount") == 0
        assert page.js_errors == []
    finally:
        page.close()


def test_escape_reframes_the_whole_deck(browser, viz_server):
    """Escape means "get me back out".

    Dropping the spotlight without moving the camera left you zoomed into two cards
    with no way back except a manual pan.
    """
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck with a drawable verified line")
    page = _build_page(browser, viz_server, slug)
    try:
        page.click(".lens-line")
        page.wait_for_timeout(1600)          # the 450ms fit transition, settled
        k_line = page.evaluate(
            "() => d3.zoomTransform(document.querySelector('canvas.force-canvas')).k")

        page.keyboard.press("Escape")
        page.wait_for_timeout(1400)
        k_deck = page.evaluate(
            "() => d3.zoomTransform(document.querySelector('canvas.force-canvas')).k")

        assert page.evaluate("Build.activeLine") == -1
        # Framing four cards is a closer camera than framing the whole deck.
        assert k_deck < k_line, (k_line, k_deck)
        assert page.js_errors == []
    finally:
        page.close()


# ── Nodes must not overlap on screen ─────────────────────────────────────


def test_graph_nodes_do_not_overlap_on_screen(browser, viz_server):
    """A node is drawn at a SCREEN-constant radius; d3's collide works in WORLD units.

    With a fixed world radius the on-screen gap depends on whatever zoom the graph was
    fitted at. Measured on a 78-node deck at k=0.505, `radius(n => n.r + 3)` gave 9 world
    units = **4.5 screen px** between nodes drawn **12 px wide** — they overlapped, which
    shrinks each node's pickable region because `pick` awards the hover to the nearest
    centre.

    THIS ASSERTION DISCRIMINATES AND A REACHABILITY ONE DOES NOT. Checked against the
    broken revision: every node still had *some* winning cursor position (2 px of overlap
    narrows a target without removing it), so "every node is reachable" passed on both and
    would have been decoration. The minimum gap is the number that moves: −2.4 px broken,
    +8.6 px fixed.
    """
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck to load")
    page = _build_page(browser, viz_server, slug)
    try:
        r = page.evaluate("""() => {
            const ns = Force.screenNodes();
            let minGap = Infinity;
            for (const a of ns) for (const b of ns) {
              if (a === b) continue;
              const g = Math.hypot(a.x - b.x, a.y - b.y) - (a.r + b.r);
              if (g < minGap) minGap = g;
            }
            return { n: ns.length, minGap };
        }""")
        assert r["n"] > 10, "need a real deck-sized graph for this to mean anything"
        assert r["minGap"] > 0, (
            f"nodes overlap on screen by {-r['minGap']:.1f}px — collide radius is not "
            f"tracking the zoom")
        assert page.js_errors == []
    finally:
        page.close()


# ── Click to select, double-click to expand ──────────────────────────────


def _a_node_position(page):
    """A node comfortably inside the canvas, in page coordinates."""
    return page.evaluate("""() => {
        const c = document.querySelector('canvas.force-canvas');
        const r = c.getBoundingClientRect();
        const ns = Force.screenNodes();
        const n = ns.find(n => n.x > 80 && n.y > 80 && n.x < r.width - 80 && n.y < r.height - 80)
                  || ns[0];
        return { x: r.left + n.x, y: r.top + n.y, name: n.name };
    }""")


def _nodes_outside_viewport(page):
    return page.evaluate("""() => {
        const c = document.querySelector('canvas.force-canvas');
        const r = c.getBoundingClientRect();
        return Force.screenNodes()
            .filter(n => n.x < -4 || n.y < -4 || n.x > r.width + 4 || n.y > r.height + 4).length;
    }""")


def test_single_click_selects_without_growing_the_graph(browser, viz_server):
    """Expanding on a single click meant there was no way to just READ a card:
    every look cost six new nodes and a re-layout."""
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck to load")
    page = _build_page(browser, viz_server, slug)
    try:
        before = page.evaluate("({n: Force.nodeCount, t: Force.trailLength})")
        pos = _a_node_position(page)
        page.mouse.click(pos["x"], pos["y"])
        page.wait_for_timeout(900)

        assert page.evaluate("Force.nodeCount") == before["n"], "a click grew the graph"
        # The breadcrumb records where you WENT, not everything you looked at.
        assert page.evaluate("Force.trailLength") == before["t"], "a click joined the trail"
        assert page.js_errors == []
    finally:
        page.close()


def test_double_click_expands(browser, viz_server):
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck to load")
    page = _build_page(browser, viz_server, slug)
    try:
        before = page.evaluate("Force.nodeCount")
        pos = _a_node_position(page)
        page.mouse.dblclick(pos["x"], pos["y"])
        page.wait_for_timeout(3000)
        assert page.evaluate("Force.nodeCount") > before
        assert page.js_errors == []
    finally:
        page.close()


def test_a_click_does_not_claim_the_camera_but_a_drag_does(browser, viz_server):
    """`drag.on('start')` fires on mousedown over a node, BEFORE any movement, and it
    used to set `userAdjusted`. One click therefore disabled every auto-fit for the rest
    of the session — which is why an expansion's neighbours went off screen and never
    came back. Ownership belongs to movement, not to pressing the button."""
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck to load")
    page = _build_page(browser, viz_server, slug)
    try:
        pos = _a_node_position(page)
        page.mouse.click(pos["x"], pos["y"])
        page.wait_for_timeout(500)
        assert page.evaluate("Force.cameraOwnedByUser") is False, "a click claimed the camera"

        page.mouse.move(pos["x"], pos["y"])
        page.mouse.down()
        page.mouse.move(pos["x"] + 120, pos["y"] + 90, steps=8)
        page.mouse.up()
        page.wait_for_timeout(500)
        assert page.evaluate("Force.cameraOwnedByUser") is True, "a real drag must own it"
        assert page.js_errors == []
    finally:
        page.close()


def test_expansion_keeps_every_card_on_screen(browser, viz_server):
    """The regression this whole change exists for.

    The only refit after a branch was the simulation's `end` handler ~1.3 s later, and
    with the camera wrongly marked user-owned it never ran at all. On the Discover
    landing — fitted to ONE card at the k=12 ceiling — six neighbours arrived entirely
    outside the viewport and stayed there.
    """
    from conftest_viz import _boot
    page = _boot(browser, viz_server, "?card=Craterhoof%20Behemoth")
    page.wait_for_function("() => window.Force && Force.nodeCount > 0", timeout=BOOT_TIMEOUT_MS)
    page.wait_for_timeout(1500)
    try:
        pos = _a_node_position(page)
        page.mouse.dblclick(pos["x"], pos["y"])
        page.wait_for_timeout(2600)

        assert page.evaluate("Force.nodeCount") > 1, "nothing expanded"
        assert page.evaluate("Force.followCount") > 0, "the follow camera never ran"
        assert _nodes_outside_viewport(page) == 0
        assert page.js_errors == []
    finally:
        page.close()


def test_zoom_to_the_deck_works_in_graph_view(browser, viz_server):
    """The button called `zoomToDeck`, which only drives the MAP camera — and Build
    defaults to the graph, where the map canvas is display:none."""
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck to load")
    page = _build_page(browser, viz_server, slug)
    try:
        pos = _a_node_position(page)
        page.mouse.move(pos["x"], pos["y"])
        page.mouse.wheel(0, -600)          # take the camera somewhere else
        page.wait_for_timeout(600)
        moved = page.evaluate("() => d3.zoomTransform(document.querySelector('canvas.force-canvas')).k")

        page.evaluate("Build.fitDeck()")
        page.wait_for_timeout(1200)
        framed = page.evaluate("() => d3.zoomTransform(document.querySelector('canvas.force-canvas')).k")

        assert framed != moved, "zoom-to-the-deck did not move the graph camera"
        assert _nodes_outside_viewport(page) == 0
        assert page.js_errors == []
    finally:
        page.close()


def test_focusing_a_region_dims_the_map_instead_of_erasing_it(browser, viz_server):
    """A region only means something against its neighbours.

    `visible()` used to exclude every non-member, so clicking a region left a cluster
    floating in a void with no way to tell where on the map it sat.

    Measured in a PATCH OF CANVAS THAT CONTAINS ONLY NON-MEMBERS, and that is the whole
    design of this test. Measuring the full canvas does not discriminate: the focused
    region keeps its own points at full strength, and with the halo they carry enough ink
    that the totals still clear any sane threshold even when every non-member has been
    erased — verified by setting the unlit alpha to 0, where the whole-canvas version
    passed happily. A patch with no members in it can only be lit by the cards this test
    is about.

    The camera is also restored before measuring: `focusRegion` frames the region, so the
    two readings would otherwise differ by zoom as well as by dimming, and zoom changes
    the ink on its own through the atmospheric halo.
    """
    from conftest_viz import _boot
    page = _boot(browser, viz_server, "?mode=explore")
    # `getCamera() != null` is the renderer's OWN readiness answer — non-null means
    # the canvas and the base fit both exist, which is exactly the state a 4 s sleep
    # was approximating. `canvas_page` waits on the same probe and documents why.
    page.wait_for_function(
        "() => document.querySelector('.map-canvas') && MM.mapRenderer"
        "      && MM.mapRenderer.getCamera() !== null", timeout=30_000)

    # Alpha, not luminance: the canvas background is transparent, so `getImageData`
    # returns colour un-premultiplied and a point at 0.09 alpha reads as full-brightness
    # RGB. All of the dimming lives in the alpha channel.
    patch_ink = """(box) => {
      const c = document.querySelector('.map-canvas');
      const d = c.getContext('2d').getImageData(box.x, box.y, box.w, box.h).data;
      let lit = 0, solid = 0;
      for (let i = 3; i < d.length; i += 4) { if (d[i] > 8) lit++; if (d[i] > 150) solid++; }
      return {lit: lit, solid: solid};
    }"""
    try:
        # Pick a patch that is dense with cards and holds no member of the region we are
        # about to focus. Chosen from the data, not hardcoded, so a re-cluster or a new
        # projection cannot quietly point it at empty space.
        box = page.evaluate(
            """() => {
                const rows = MM.regionRows ? MM.regionRows('l0_0') : null;
                return {rows: rows ? rows.length : 0};
            }"""
        )
        page.evaluate("MM.focusRegion('l0_0')")
        page.wait_for_timeout(2000)
        members = page.evaluate("() => [...MM.regionFocus.rows]")
        page.evaluate("MM.clearRegionFocus()")
        page.wait_for_timeout(1500)

        box = page.evaluate(
            """(members) => {
                const mem = new Set(members);
                const S = 120;                       // patch size in CSS pixels
                const c = document.querySelector('.map-canvas');
                const W = c.clientWidth, H = c.clientHeight;
                // Bin every drawn card into S-sized cells, counting members separately.
                const cells = new Map();
                for (let i = 0; i < MM.allData.length; i++) {
                    const d = MM.allData[i];
                    const p = MM.mapRenderer.dataToPixel(d.x, d.y);
                    if (!p || p[0] < 0 || p[1] < 0 || p[0] >= W || p[1] >= H) continue;
                    const key = Math.floor(p[0] / S) + ',' + Math.floor(p[1] / S);
                    let cell = cells.get(key);
                    if (!cell) { cell = {n: 0, mem: 0}; cells.set(key, cell); }
                    cell.n++;
                    if (mem.has(i)) cell.mem++;
                }
                // The densest cell with NO members in it.
                let best = null, bestKey = null;
                cells.forEach((cell, key) => {
                    if (cell.mem > 0) return;
                    if (!best || cell.n > best.n) { best = cell; bestKey = key; }
                });
                if (!best) return null;
                const [cx, cy] = bestKey.split(',').map(Number);
                return {x: cx * S, y: cy * S, w: S, h: S, cards: best.n};
            }""",
            members,
        )
        assert box and box["cards"] > 200, f"no dense member-free patch found: {box}"

        before = page.evaluate(patch_ink, box)
        assert before["lit"] > 500, f"the chosen patch was not drawn: {before}"

        cam = page.evaluate("() => MM.mapRenderer.getCamera()")
        page.evaluate("MM.focusRegion('l0_0')")
        page.wait_for_timeout(2200)
        page.evaluate("(c) => MM.mapRenderer.setCamera(c)", cam)
        page.wait_for_timeout(1200)
        after = page.evaluate(patch_ink, box)

        # Dimmed...
        assert after["solid"] < before["solid"] * 0.5, (
            f"focusing a region did not dim the cards outside it "
            f"({before['solid']} -> {after['solid']} px at full strength)")
        # ...but still THERE. This is the load-bearing half: these pixels are exclusively
        # non-members, so if the region focus erased them this patch goes black.
        assert after["lit"] > before["lit"] * 0.5, (
            f"non-members vanished from a patch that contains only them "
            f"({after['lit']} of {before['lit']} px) — orientation is lost")
        assert page.js_errors == []
    finally:
        page.close()


def test_regions_are_named_three_levels_deep(browser, viz_server):
    """Countries, states and neighbourhoods, each with a hand-authored or inherited name
    rather than the machine label."""
    import json
    from manamap.config import DATA_DIR

    doc = json.loads((DATA_DIR / "regions_default.json").read_text())
    levels = {r["level"] for r in doc["regions"]}
    assert levels == {0, 1, 2}, levels
    # Every L0/L1 carries the authored name, with the machine one kept alongside.
    for r in doc["regions"]:
        if r["level"] <= 1:
            assert r.get("mechanical"), f"{r['id']} lost its mechanical label"
            assert r["label"] != r["mechanical"], f"{r['id']} kept the machine name"


@pytest.mark.browser
def test_explore_boots_on_the_ability_map(page):
    """The shipping default is the ability map, and `currentMap` alone does not deliver it.

    The boot fetch was hardcoded to `MAP_CONFIGS.default.projection`, so flipping the
    default would have left `currentMap` reporting 'ability' while `allData` held the
    colour+type coordinates — every position wrong, nothing on screen to say so, and the
    selector still reading Abilities.
    """
    assert page.evaluate("() => MM.currentMap") == "ability"
    assert page.evaluate("() => document.getElementById('mapSelect').value") == "ability"
    # The projection actually loaded is the ability one. Compared against the OTHER map's
    # coordinates for the same row: identical values would mean the wrong file was fetched.
    same = page.evaluate("""async () => {
      const before = MM.allData.slice(0, 40).map(d => [d.x, d.y]);
      document.getElementById('mapSelect').value = 'default';
      document.getElementById('mapSelect').dispatchEvent(new Event('change'));
      await new Promise(r => setTimeout(r, 2500));
      const after = MM.allData.slice(0, 40).map(d => [d.x, d.y]);
      return before.filter((p, i) => p[0] === after[i][0] && p[1] === after[i][1]).length;
    }""")
    assert same < 5, f"{same}/40 points identical across maps — the boot fetched one map twice"


@pytest.mark.browser
def test_switching_maps_reindexes_the_hit_test(page):
    """`applyProjection` moves every point in place, and the quadtree cannot tell.

    Its signature is layer lengths plus endpoint ids — all identical across a map switch,
    the same 34,322 cards in the same groups — so it never rebuilds on its own. The rebuild
    also has to happen AFTER `render()` installs the new layers: `buildTree` copies
    coordinates out of the layer arrays, so reindexing first rebuilds from the outgoing
    ones and `setLayers` then skips its own rebuild against that unchanged signature. The
    stale positions survive the very call meant to remove them.

    Asserted through `pick` directly rather than through a hover, because the hover path
    adds interference this invariant has nothing to do with: at the whole-map fit
    neighbouring cards are sub-pixel apart (measured: `pick(845.1, 653.7)` and
    `pick(845, 654)` return different rows), and region labels are real DOM buttons layered
    over the canvas, so a card under one cannot be hovered at all. Framing each card first
    removes the density ambiguity; `test_hovering_names_the_card_under_the_cursor` covers
    the hover pipeline itself.
    """
    rows = [100, 9000, 20000, 31000]

    def picks_itself():
        return page.evaluate(
            """(rows) => rows.map(row => {
                const d = MM.allData[row];
                // Frame the card so it owns its pixels — otherwise the answer is a
                // statement about point density, not about the index.
                MM.mapRenderer.setCamera({x: [d.x - 1.2, d.x + 1.2], y: [d.y - 0.8, d.y + 0.8]});
                const px = MM.mapRenderer.dataToPixel(d.x, d.y);
                return [row, MM.mapRenderer.pick(px[0], px[1])];
            })""",
            rows,
        )

    assert [r for r, got in picks_itself() if r != got] == [], "hit test wrong before any switch"

    # ability -> default -> ability. Switching to the map you are already on is a no-op and
    # proves nothing; the bug needs the coordinates to actually change.
    for target in ("default", "ability"):
        page.evaluate(
            """(m) => {
                const s = document.getElementById('mapSelect');
                s.value = m;
                s.dispatchEvent(new Event('change'));
            }""",
            target,
        )
        page.wait_for_timeout(3000)
        assert page.evaluate("() => MM.currentMap") == target
        wrong = [(r, got) for r, got in picks_itself() if r != got]
        assert not wrong, (
            f"after switching to the {target} map, rows {wrong} no longer hit-test to "
            "themselves — the quadtree still holds the previous map's positions"
        )
    assert page.js_errors == []


@pytest.mark.browser
def test_hovering_names_the_card_under_the_cursor(page):
    """The popup must name whatever `pick` reports at the cursor.

    Driven by the real mouse on purpose: a synthetic `MouseEvent` leaves `offsetX`/`offsetY`
    at 0 and the hover handler picks on exactly those, so a synthetic probe reports the card
    at the canvas origin and passes on completely broken code. Asserted against `pick` at
    the same coordinates rather than against a chosen row, so point density cannot make a
    correct popup look wrong.
    """
    # STOP THE MAP FIRST. The atlas drifts — `proj()` adds a time term and
    # `unproj()` inverts it exactly — so `pick(x, y)` is a function of the CLOCK
    # as well as the cursor. This test hovers, waits 600 ms for the popup, and
    # then asks `pick` the same question; on a moving field those are two
    # different questions and the answers legitimately differ. It was a latent
    # flake for exactly that reason, and was verified to fail with and without
    # the change that first surfaced it.
    #
    # Freezing motion makes `pick` time-invariant, which turns this back into
    # the geometry assertion it is supposed to be. Another wait would only move
    # the odds.
    page.evaluate("() => MM.mapRenderer.setMotion(false)")
    page.wait_for_timeout(900)          # let the 700 ms motion ramp settle to rest
    box = page.evaluate(
        """() => {
            const r = document.querySelector('.map-canvas').getBoundingClientRect();
            return {l: r.left, t: r.top, w: r.width, h: r.height};
        }"""
    )
    checked = 0
    for fx, fy in ((0.42, 0.45), (0.55, 0.60), (0.62, 0.38)):
        x = round(box["l"] + box["w"] * fx)
        y = round(box["t"] + box["h"] * fy)
        # Park in the corner first so the hover genuinely re-fires rather than being
        # deduped by `hoverRow === row`.
        page.mouse.move(round(box["l"] + 6), round(box["t"] + 6))
        page.wait_for_timeout(150)
        page.mouse.move(x, y)
        page.wait_for_timeout(600)
        state = page.evaluate(
            """([x, y]) => {
                const c = document.querySelector('.map-canvas');
                const r = c.getBoundingClientRect();
                const top = document.elementFromPoint(x, y);
                const row = MM.mapRenderer.pick(x - r.left, y - r.top);
                const e = document.querySelector('.card-popup');
                const img = e && e.style.display !== 'none' ? e.querySelector('img') : null;
                return {
                    onCanvas: !!top && top.classList.contains('map-canvas'),
                    expected: row == null ? null : MM.cardRecord(row).n,
                    shown: img ? img.getAttribute('alt') : null,
                };
            }""",
            [x, y],
        )
        # Region labels are DOM buttons over the canvas; a sample that lands on one is
        # testing the label, not the hover.
        if not state["onCanvas"] or state["expected"] is None:
            continue
        checked += 1
        assert state["shown"] == state["expected"], (
            f"cursor at ({x}, {y}) is over {state['expected']!r} but the popup shows "
            f"{state['shown']!r}"
        )
    assert checked, "every sample landed on a region label — nothing was actually tested"
    assert page.js_errors == []


@pytest.mark.browser
def test_clicking_a_legend_row_spotlights_that_group(page):
    """The legend is a control, not a caption.

    Asserted on drawn ALPHA (see `_ink_strength`): the canvas background is transparent, so
    dimming a group changes the alpha channel and leaves `getImageData`'s un-premultiplied
    RGB untouched — an RGB probe reports a spotlight as no change at all.

    Two assertions, because a spotlight has two halves. `solid` must collapse, or nothing
    was dimmed; `lit` must hold, or the surroundings were erased rather than muted — which
    is the failure this map already made once with region focus, and the reason you could
    not tell where anything was.
    """
    page.wait_for_timeout(800)
    before = _ink_strength(page)

    page.evaluate(
        """() => {
            const row = [...document.querySelectorAll('.map-legend-row')]
                .find(r => r.dataset.key === 'Planeswalker');
            row.click();
        }"""
    )
    page.wait_for_timeout(1000)
    during = _ink_strength(page)

    assert page.evaluate("() => MM.legendFocus && MM.legendFocus.key") == "Planeswalker"
    assert page.evaluate("() => document.querySelectorAll('.map-legend-row.is-active').length") == 1
    # Planeswalkers are 1.0% of the corpus, so spotlighting them takes nearly all the
    # full-strength ink out of the map.
    assert during["solid"] < before["solid"] * 0.3, (
        f"legend focus barely dimmed anything ({before['solid']:.2f}% -> "
        f"{during['solid']:.2f}% drawn at full strength)"
    )
    # ...and the rest is still on screen. Muted, not erased.
    assert during["lit"] > before["lit"] * 0.8, (
        f"the muted points vanished ({before['lit']:.2f}% -> {during['lit']:.2f}% drawn)"
    )

    page.keyboard.press("Escape")
    page.wait_for_timeout(900)
    assert page.evaluate("() => MM.legendFocus") is None
    assert page.evaluate("() => document.querySelectorAll('.map-legend-row.is-active').length") == 0
    after = _ink_strength(page)
    assert after["solid"] > before["solid"] * 0.8, "Escape did not restore the map"
    assert page.js_errors == []


@pytest.mark.browser
def test_entering_explore_shows_the_whole_map(discover_page):
    """Arriving in Explore is not asking for the lens.

    It used to auto-orient on a non-empty tray, so walking a few cards in Discover and
    switching opened the atlas with almost all of it at 8% alpha and the camera somewhere
    else — which reads as a rendering fault, not as a lens.
    """
    # Reads MM.allData, which this fixture deliberately does not wait for.
    await_projection(discover_page)
    page = discover_page
    # Hold something, so the old auto-orient path would have fired.
    # A grown graph is what Session.size() counts, and it is what the old auto-orient
    # branch keyed on.
    page.evaluate("() => MM.relate(Discovery.current, 'similar')")
    page.wait_for_timeout(400)

    page.evaluate(
        """() => {
            const s = document.getElementById('modeSelect');
            s.value = 'explore';
            s.dispatchEvent(new Event('change'));
        }"""
    )
    # The mode switch loads the projection and refits. Wait for the renderer to say
    # it has a fit rather than for a number that was tuned on one machine.
    page.wait_for_function(
        "() => MM.mode === 'explore' && MM.allData && MM.allData.length > 0"
        "      && MM.mapRenderer && MM.mapRenderer.getCamera() !== null",
        timeout=30_000)

    assert page.evaluate("() => MM.orientation") is None, "Explore auto-oriented on entry"
    assert page.evaluate("() => MM.regionFocus") is None
    assert page.evaluate("() => MM.legendFocus") is None
    # The camera shows the whole map, not a corner of it.
    span = page.evaluate(
        """() => {
            const cam = MM.mapRenderer.getCamera();
            return Math.abs(cam.x[1] - cam.x[0]);
        }"""
    )
    assert span > 40, f"Explore opened zoomed in (camera span {span:.1f})"
    assert page.js_errors == []


# ── Ambient motion ──────────────────────────────────────────────────────
#
# The atlas drifts at altitude: a slow differential sway that reads as orbital motion.
# Every test here exists because the obvious implementation of that is wrong in a way
# nothing else in the suite would catch — motion in the DATA rather than in the
# projection would leave the quadtree answering with stale positions, and a motion that
# accumulated would pull PaCMAP neighbours apart while looking perfectly nice.


def _excursion_track(page, seconds, step_ms=500):
    """Peak distance, in px, between where cards are drawn and where they are stored.

    Sampled through `dataToPixel`, which is what every overlay and every click aims
    through — so this measures the thing users actually interact with rather than an
    internal the renderer could be lying about.
    """
    return page.evaluate(
        """async ([seconds, stepMs]) => {
            const R = MM.mapRenderer;
            const rows = [];
            for (let i = 0; i < MM.allData.length; i += 997) rows.push(i);
            R.setMotion(false);
            await new Promise(r => setTimeout(r, 1200));   // let it ease home
            const home = rows.map(i => R.dataToPixel(MM.allData[i].x, MM.allData[i].y));
            R.setMotion(true);
            const track = [];
            for (let s = 0; s * stepMs < seconds * 1000; s++) {
                await new Promise(r => setTimeout(r, stepMs));
                let m = 0;
                rows.forEach((i, j) => {
                    const p = R.dataToPixel(MM.allData[i].x, MM.allData[i].y);
                    m = Math.max(m, Math.hypot(p[0] - home[j][0], p[1] - home[j][1]));
                });
                track.push(m);
            }
            return track;
        }""",
        [seconds, step_ms],
    )


def test_the_atlas_drifts_at_altitude_and_holds_still_up_close(page):
    """Motion is atmosphere, and rides the same altitude ramp the halo does.

    Zoomed in, Explore has to converge on Discover and Build — and a card you are trying
    to click must not be moving under the cursor. The ramp is what buys both.
    """
    at_fit = max(_excursion_track(page, seconds=6))
    assert at_fit > 3, f"the map never moved at the fit (peak {at_fit:.1f}px)"

    up_close = page.evaluate(
        """async () => {
            const R = MM.mapRenderer;
            const d = MM.allData[MM.allData.findIndex(c => c.n === 'Sol Ring')];
            R.setCamera({x: [d.x - 1.5, d.x + 1.5], y: [d.y - 1.0, d.y + 1.0]});
            await new Promise(r => setTimeout(r, 1200));
            const rows = [];
            for (let i = 0; i < MM.allData.length; i += 997) rows.push(i);
            const snap = () => rows.map(i => R.dataToPixel(MM.allData[i].x, MM.allData[i].y));
            const a = snap();
            await new Promise(r => setTimeout(r, 2000));
            const b = snap();
            let m = 0;
            for (let i = 0; i < a.length; i++) {
                m = Math.max(m, Math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1]));
            }
            return {level: R.motionLevel, movedPx: m};
        }"""
    )
    assert up_close["level"] == 0, "ambient motion is still running zoomed in"
    assert up_close["movedPx"] == 0, f"points moved {up_close['movedPx']:.2f}px up close"
    assert page.js_errors == []


def test_the_drift_is_bounded_and_returns(page):
    """THE load-bearing test: this is a sway, not a rotation.

    A galaxy that genuinely rotates winds up — inner orbits lap outer ones and, on a map
    whose entire content is "what is near what", cards PaCMAP placed side by side end up
    a quarter of the map apart. The picture would be lying, slowly, and it would still
    look good the whole time. So the excursion is capped and the field comes home.
    """
    track = _excursion_track(page, seconds=16)
    peak = max(track)
    # Generous either side of the ~32px measured on a 900px viewport: the point is the
    # ORDER OF MAGNITUDE — single-digit percent of the viewport, never a lap.
    assert 3 < peak < 90, f"excursion {peak:.1f}px is not a slight drift: {track}"
    # And it must come back. A monotonic rotation only ever grows away from home until it
    # has gone most of the way round; a sway retreats from its own peak.
    at_peak = track.index(peak)
    after = track[at_peak:]
    assert min(after) < peak * 0.8, (
        f"the drift never retreated from its peak — it is accumulating: {track}")
    assert page.js_errors == []


def test_a_click_still_lands_on_the_card_it_aimed_at_while_the_map_moves(page):
    """The reason the motion lives in the projection and not in the data.

    `pick` inverts the swirl exactly — it is a rotation about a fixed centre, so radius is
    preserved and the angle is recoverable from where the point landed. Hit-testing against
    stored positions while drawing somewhere else would be off by tens of pixels at the
    fit, which in this corpus is a different card entirely, and would look like a flaky
    map rather than a coordinate bug.
    """
    hits = page.evaluate(
        """async () => {
            const R = MM.mapRenderer;
            const names = ['Sol Ring', 'Llanowar Elves', 'Counterspell',
                           'Craterhoof Behemoth', 'Wrath of God'];
            const out = [];
            for (const nm of names) {
                const i = MM.allData.findIndex(d => d.n === nm);
                if (i < 0) continue;
                const p = R.dataToPixel(MM.allData[i].x, MM.allData[i].y);
                out.push({aim: nm, exact: R.pick(p[0], p[1], 3) === i});
                await new Promise(r => setTimeout(r, 700));   // sample several phases
            }
            return out;
        }"""
    )
    assert hits, "no probe cards resolved"
    missed = [h["aim"] for h in hits if not h["exact"]]
    assert not missed, f"the pick inverse is off while the map drifts: {missed}"
    assert page.js_errors == []


def test_motion_stops_when_the_map_is_not_the_thing_on_screen(page):
    """`#plot` is shared with the force graph and stays visible in Discover and Build; only
    `.map-canvas` is hidden. Checking the host rather than the canvas animated 34,890
    points into a surface nobody could see, in the two modes that do not use it."""
    r = page.evaluate(
        """async () => {
            MM.setMode('discover');
            await new Promise(r => setTimeout(r, 900));
            const hidden = document.querySelector('.map-canvas').offsetParent === null;
            const off = MM.mapRenderer.motionLevel;
            MM.setMode('explore');
            await new Promise(r => setTimeout(r, 1200));
            return {hidden, off, back: MM.mapRenderer.motionLevel};
        }"""
    )
    assert r["hidden"], "the map canvas was still displayed in Discover"
    assert r["off"] == 0, "the ambient ticker kept running behind the graph modes"
    assert r["back"] > 0, "motion did not resume on returning to Explore"
    assert page.js_errors == []


def test_the_motion_toggle_reports_the_renderer_not_the_markup(page):
    r = page.evaluate(
        """async () => {
            const btn = document.getElementById('toggleMotion');
            const on = {motion: MM.mapRenderer.motion,
                        active: btn.classList.contains('active')};
            btn.click();
            await new Promise(r => setTimeout(r, 1400));
            return {on, off: {motion: MM.mapRenderer.motion,
                              active: btn.classList.contains('active'),
                              level: MM.mapRenderer.motionLevel}};
        }"""
    )
    assert r["on"] == {"motion": True, "active": True}
    assert r["off"]["motion"] is False and r["off"]["active"] is False
    assert r["off"]["level"] == 0, "the map kept drifting after motion was switched off"
    assert page.js_errors == []


def test_reduced_motion_is_honoured_at_boot(still_page):
    """An ambient drift is precisely what `prefers-reduced-motion` is about, so it is the
    DEFAULT that changes — not a hard override. Someone who asks for it explicitly still
    gets it, and the toggle has to show which state they are actually in."""
    r = still_page.evaluate(
        """async () => {
            const R = MM.mapRenderer, btn = document.getElementById('toggleMotion');
            const rows = [];
            for (let i = 0; i < MM.allData.length; i += 997) rows.push(i);
            const snap = () => rows.map(i => R.dataToPixel(MM.allData[i].x, MM.allData[i].y));
            const a = snap();
            await new Promise(r => setTimeout(r, 2000));
            const b = snap();
            let moved = 0;
            for (let i = 0; i < a.length; i++) {
                moved = Math.max(moved, Math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1]));
            }
            // Snapshot the BOOTED state before opting in. Reading it after the click
            // measures the click instead of the default, and passes either way.
            const booted = {motion: R.motion, active: btn.classList.contains('active')};
            btn.click();
            await new Promise(r => setTimeout(r, 1500));
            return {...booted, movedPx: moved, afterOptIn: R.motion};
        }"""
    )
    assert r["motion"] is False, "reduced-motion was ignored"
    assert r["active"] is False, "the toggle claimed motion was on while nothing moved"
    assert r["movedPx"] == 0, f"the map drifted under reduced-motion ({r['movedPx']:.2f}px)"
    assert r["afterOptIn"] is True, "an explicit opt-in was refused"
    assert still_page.js_errors == []


def test_box_select_catches_what_is_drawn_inside_the_marquee(page):
    """A rotation maps a rectangle to something that is not a rectangle, so the stored
    positions inside a screen box stop being an axis-aligned range. Pruning is padded by
    the largest displacement in play and membership is decided by projecting each candidate
    forward — otherwise the marquee quietly catches the wrong cards at the edges."""
    r = page.evaluate(
        """() => {
            const R = MM.mapRenderer;
            const c = document.querySelector('.map-canvas');
            const w = c.clientWidth, h = c.clientHeight;
            const box = [w * 0.35, h * 0.35, w * 0.65, h * 0.65];
            const rows = R.pickRect(box[0], box[1], box[2], box[3]);
            // Everything returned must actually be drawn inside the marquee.
            const stray = rows.filter(i => {
                const p = R.dataToPixel(MM.allData[i].x, MM.allData[i].y);
                return p[0] < box[0] - 1 || p[0] > box[2] + 1
                    || p[1] < box[1] - 1 || p[1] > box[3] + 1;
            });
            return {n: rows.length, stray: stray.length};
        }"""
    )
    assert r["n"] > 50, f"the marquee caught almost nothing ({r['n']})"
    assert r["stray"] == 0, f"{r['stray']} caught cards are drawn outside the marquee"
    assert page.js_errors == []


# ── The stack theatre (the magazine's one interactive component) ─────────
#
# Everything else in a manual is static HTML, so this is the only place in the
# magazine where a source-assertion test would be actively misleading: the markup
# can be perfectly correct while the CSS selects the wrong element. It already
# was — the rail's "RESOLVE" label sat inside the tab list, so every generated
# `:nth-child(I)` rule landed one tab early and step 4 rendered with tab 3 lit.
# The markup was right; the mechanism was off by one. These read computed style.


def _theatre(browser, viz_server, slug="radagast"):
    page = browser.new_page()
    page.goto(f"{viz_server}/manuals/{slug}.html")
    page.wait_for_selector(".theatre", timeout=15000)
    return page


def test_the_theatre_step_and_its_tab_and_plate_agree(browser, viz_server):
    """Clicking step N lights tab N, shows note N, and brings plate N forward."""
    page = _theatre(browser, viz_server)
    try:
        tabs = page.query_selector_all("#th-003 .th-tab")
        assert len(tabs) >= 6, "radagast 003 has eight steps"
        for n in (1, 4, 6):
            page.click(f"#th-003 .th-tab:nth-of-type({n})")
            page.wait_for_timeout(500)
            notes = page.eval_on_selector_all(
                "#th-003 .th-note", "e=>e.map(x=>getComputedStyle(x).display)")
            shown = [i for i, d in enumerate(notes) if d != "none"]
            assert shown == [n - 1], f"step {n} shows note(s) {shown}"
            lit = page.eval_on_selector_all(
                "#th-003 .th-tab",
                "e=>e.map(x=>getComputedStyle(x).backgroundColor)")
            # The active tab is the burst yellow; every other one is the muted
            # translucent fill. Comparing against its OWN siblings rather than a
            # literal colour keeps this alive through a palette change.
            assert lit.count(lit[n - 1]) == 1, f"tab {n} is not uniquely lit"
            # The front plate is the only one at full opacity.
            op = page.eval_on_selector_all(
                "#th-003 .th-plate", "e=>e.map(x=>+getComputedStyle(x).opacity)")
            assert op[n - 1] == max(op) and op[n - 1] > 0.9
            assert sum(1 for v in op if v > 0.9) == 1, op
    finally:
        page.close()


def test_the_theatre_opens_on_a_valid_view_with_no_script(browser, viz_server):
    """Step 1 is `checked` in the markup, so the page is never a blank stage —
    and the manual carries no script to make it one either."""
    page = _theatre(browser, viz_server)
    try:
        assert page.eval_on_selector_all("script", "e=>e.length") == 0
        notes = page.eval_on_selector_all(
            "#th-001 .th-note", "e=>e.map(x=>getComputedStyle(x).display)")
        assert [i for i, d in enumerate(notes) if d != "none"] == [0]
    finally:
        page.close()


def test_the_theatre_prints_every_step(browser, viz_server):
    """A printed page showing step 1 and hiding seven is a page missing the
    proof. In print the stage becomes an illustration and the record prints."""
    page = _theatre(browser, viz_server)
    try:
        page.emulate_media(media="print")
        page.wait_for_timeout(300)
        notes = page.eval_on_selector_all(
            "#th-003 .th-note", "e=>e.map(x=>getComputedStyle(x).display)")
        assert all(d != "none" for d in notes), notes
        assert page.eval_on_selector(
            "#th-003 .th-railwrap", "e=>getComputedStyle(e).display") == "none"
    finally:
        page.close()


def test_hovering_a_plate_lifts_it(browser, viz_server):
    """The one interaction that needs no click, and the reason the stack reads
    as an object rather than a diagram.

    The point is found rather than assumed. Playwright hovers an element's
    CENTRE, and the centre of every back plate is under the front one — the first
    version of this test hovered a covered pixel, got no `:hover`, and reported
    that the lift was broken when it was not. Probing for a pixel where the plate
    is genuinely the topmost element also checks the thing that makes the fan a
    fan: if no back plate has an exposed pixel, the stack is just one card.
    """
    page = _theatre(browser, viz_server)
    try:
        plate = "#th-003 .th-plate:nth-of-type(8)"
        page.eval_on_selector(plate, "e=>e.scrollIntoView({block:'center'})")
        page.wait_for_timeout(400)
        spot = page.eval_on_selector(plate, """el => {
          const r = el.getBoundingClientRect();
          for (let fy = 0.06; fy < 0.95; fy += 0.08)
            for (let fx = 0.06; fx < 0.95; fx += 0.08) {
              const x = r.left + r.width * fx, y = r.top + r.height * fy;
              const hit = document.elementFromPoint(x, y);
              if (hit && el.contains(hit)) return {x, y};
            }
          return null;
        }""")
        assert spot, "no pixel of the last plate is reachable — the fan is flat"
        before = page.eval_on_selector(plate, "e=>e.getBoundingClientRect().width")
        page.mouse.move(spot["x"], spot["y"])
        page.wait_for_timeout(700)
        after = page.eval_on_selector(plate, "e=>e.getBoundingClientRect().width")
        # Lifting is a translateZ under perspective, so it reads as growth.
        assert after > before * 1.05, (before, after)
    finally:
        page.close()


def test_the_case_index_scans_closed_and_holds_the_record_open(browser, viz_server):
    """Judge's Desk is a list you run your eye down and open ONE of.

    "Shrinks to verdicts" and "may not truncate a citation" are both binding, and
    they only look contradictory if "shrinks" means "holds less". What shrinks is
    the footprint: one row per case closed, the complete verbatim record inside.
    """
    page = browser.new_page()
    page.goto(f"{viz_server}/manuals/radagast.html")
    page.wait_for_selector(".dossier", timeout=15000)
    try:
        rows = page.query_selector_all("#judges-desk .case-row")
        assert len(rows) >= 5, "radagast publishes seven cases"
        # Closed by default, and cheap: a case row is one line, not a header block.
        assert page.eval_on_selector_all(
            "#judges-desk details.dossier", "e=>e.filter(d=>d.open).length") == 0
        heights = page.eval_on_selector_all(
            "#judges-desk .case-row", "e=>e.map(r=>r.getBoundingClientRect().height)")
        assert max(heights) < 90, f"case rows are not one-liners: {heights}"

        # The record is there, in full, the moment one is opened.
        before = page.eval_on_selector(
            "#judges-desk", "e=>e.getBoundingClientRect().height")
        page.eval_on_selector("#judges-desk details.dossier", "d=>{d.open=true}")
        page.wait_for_timeout(300)
        after = page.eval_on_selector(
            "#judges-desk", "e=>e.getBoundingClientRect().height")
        assert after > before * 2, (before, after)
        opened = page.eval_on_selector(
            "#judges-desk details.dossier[open]", "d=>d.innerText")
        assert "CR " in opened, "an opened case shows no citations"
    finally:
        page.close()


def test_the_kill_points_at_the_proof_instead_of_reprinting_it(browser, viz_server):
    """The theatre shipped printing every citation inline, which put the identical
    120 quotes into both departments. The walkthrough keeps action and effect; the
    rules live in one place and The Kill links to them."""
    page = browser.new_page()
    page.goto(f"{viz_server}/manuals/radagast.html")
    page.wait_for_selector(".theatre", timeout=15000)
    try:
        # A prose MENTION of a rule is legitimate and is what the renderer's
        # evidence links exist for (STYLEv3 8.4) — a caption may say "CR 302.6"
        # and become a link to the case. What may not appear here is a citation
        # BLOCK: the rule number set beside its verbatim quote, which is the
        # appendix's job and was being printed in both places.
        assert page.query_selector_all("#the-kill .cite") == []
        assert page.query_selector_all("#judges-desk .cite") != []
        # `innerText` reflects text-transform, and this label is set in caps.
        kill = page.eval_on_selector("#the-kill", "e=>e.innerText").lower()
        assert "citations on the record" in kill
        assert page.query_selector("#the-kill a.dossier-pointer") is not None
    finally:
        page.close()


# ── The Workbench landing ────────────────────────────────────────────────
#
# The front door answers ONE question before any other — which decks can I play
# tonight — and it is the only question in the repo that no artifact derives,
# because it is a fact about cardboard. These drive the renderer against a
# STUBBED manifest rather than a locked deck: `deck_versions.json` is tracked and
# scanned by `build-index`, so locking one in a test races every other test that
# reads the manifest. Routing the fetch tests the rendering, which is the part
# that can break.


def _workbench(browser, viz_server, decks, infos=None):
    page = browser.new_page()
    import json as _json
    page.route("**/data/decks/index.json*", lambda route: route.fulfill(
        status=200, content_type="application/json",
        body=_json.dumps({"decks": decks})))
    page.route("**/data/decks/*/info.json", lambda route: route.fulfill(
        status=200, content_type="application/json",
        body=_json.dumps((infos or {}).get(route.request.url.split("/")[-2], {}))))
    page.goto(f"{viz_server}/viz/workbench.html")
    page.wait_for_timeout(1200)
    return page


def _deck(slug, **kw):
    base = {"slug": slug, "deck_name": slug.upper(), "commander": "Someone",
            "image": None, "status": None, "verified": 0, "decisions": 0,
            "sim_runs": [], "experiments": [], "prescriptions": [],
            "locked": False, "paper": None, "published": True, "has": {}}
    base.update(kw)
    return base


def test_the_workbench_splits_locked_from_the_bench(browser, viz_server):
    """The split IS the page. A deck that exists only as JSON and a deck you can
    put on a table are different objects, and the old picker — one line of text
    links — could not tell you which was which."""
    page = _workbench(browser, viz_server, [
        _deck("sleeved", locked=True,
              paper={"version": 6, "in_sync": True, "versions_behind": 0, "drift": None}),
        _deck("onbench"),
        _deck("dead", status=["broken-down", "BROKEN DOWN FOR PARTS", "…"]),
    ])
    try:
        heads = page.eval_on_selector_all(
            ".wb-rack h2", "els => els.map(e => e.textContent.trim().split(' ')[0])")
        assert heads == ["Sleeved", "On", "History"], heads
        racks = page.eval_on_selector_all(
            ".wb-rack", "els => els.map(e => e.querySelectorAll('.wb-card').length)")
        assert racks == [1, 1, 1], racks
        assert page.eval_on_selector_all(
            ".wb-rack:first-of-type .wb-card h3", "els => els.map(e => e.textContent)") \
            == ["SLEEVED"]
    finally:
        page.close()


def test_a_drifted_lock_says_what_to_pull_and_add(browser, viz_server):
    """The two sides are the physical instruction. A lock that only said
    "drifted" would send the pilot to go and diff it by hand."""
    page = _workbench(browser, viz_server, [
        _deck("drifted", locked=True, paper={
            "version": 5, "in_sync": False, "versions_behind": 1,
            "drift": {"pull": ["a", "b", "c"], "add": ["d"]}}),
    ])
    try:
        chips = page.eval_on_selector_all(
            ".wb-chip", "els => els.map(e => e.textContent.trim())")
        assert any("V5" in c and "1 behind" in c.lower() for c in chips), chips
        assert any("pull 3" in c.lower() and "add 1" in c.lower() for c in chips), chips
        warn = page.eval_on_selector_all(".wb-chip.wb-warn", "els => els.length")
        assert warn == 2, "drift must be coloured as a warning, not as neutral chrome"
    finally:
        page.close()


def test_an_in_sync_lock_reads_as_ok_not_as_a_warning(browser, viz_server):
    page = _workbench(browser, viz_server, [
        _deck("level", locked=True,
              paper={"version": 6, "in_sync": True, "versions_behind": 0, "drift": None}),
    ])
    try:
        assert page.eval_on_selector_all(".wb-chip.wb-ok", "els => els.length") == 1
        assert page.eval_on_selector_all(".wb-chip.wb-warn", "els => els.length") == 0
    finally:
        page.close()


def test_an_unresolvable_lock_is_not_reported_as_in_sync(browser, viz_server):
    """`in_sync` is tri-state — true, false, or null when the lock names a version
    git no longer carries. Null must not read as fine."""
    page = _workbench(browser, viz_server, [
        _deck("ghost", locked=True,
              paper={"version": 9, "in_sync": None, "unresolved": True, "drift": None}),
    ])
    try:
        chips = page.eval_on_selector_all(
            ".wb-chip", "els => els.map(e => e.textContent.trim())")
        assert any("not in git" in c for c in chips), chips
        assert page.eval_on_selector_all(".wb-chip.wb-ok", "els => els.length") == 0
    finally:
        page.close()


def test_the_empty_state_names_the_command_that_fixes_it(browser, viz_server):
    """Every deck is unlocked today, so this is the state the page actually opens
    in — and a rack with no explanation reads as a bug rather than as a fact."""
    page = _workbench(browser, viz_server, [_deck("onbench")])
    try:
        assert page.is_visible(".wb-empty")
        assert "deck-version" in page.text_content(".wb-empty")
    finally:
        page.close()


def test_every_card_offers_all_three_destinations_by_name(browser, viz_server):
    """Three named links, and NOTHING implicit.

    The card used to be one big invisible link to the dossier. The rule behind
    that — "a card that opens two different things depending on where you click
    is the interaction bug the atlas already fixed once" — is obeyed here by
    deleting the invisible target rather than adding a second one.

    `deck.html?deck=<slug>` and `index.html?deck=<slug>` are both inbound
    contracts other surfaces already rely on: the dossier deep-links from every
    manual, and the map reads `?deck=` on entry to land in Build with the deck
    loaded rather than on an unfiltered 34,890-card atlas.
    """
    page = _workbench(browser, viz_server, [_deck("alpha", has={"page": True})])
    try:
        hrefs = page.eval_on_selector_all(
            ".wb-links a", "els => els.map(e => e.getAttribute('href'))")
        assert hrefs == ["../manuals/p/alpha.html",
                         "deck.html?deck=alpha",
                         "index.html?deck=alpha"], hrefs
        # The title is the manual, because that is what a pilot opens a deck to
        # read — but it is a LABELLED link, not a hidden hit area.
        assert page.get_attribute(".wb-title", "href") == "../manuals/p/alpha.html"
        assert page.query_selector_all(".wb-hit") == []
    finally:
        page.close()


def test_the_manual_link_appears_only_when_there_is_a_manual(browser, viz_server):
    """A card is ONE hit area to the deck page, with the manual as its own small
    link inside it. A card that opens two different things depending on where you
    click is the interaction bug the atlas already fixed once. And a link to a
    page that does not exist is worse than no link, so it is hidden rather than
    dead — the same rule the dossier's issue link follows."""
    page = _workbench(browser, viz_server, [
        _deck("withpage", has={"page": True}),
        _deck("nopage", has={"page": False}),
    ])
    try:
        links = page.eval_on_selector_all(
            ".wb-links a", "els => els.map(e => e.getAttribute('href'))")
        # withpage gets three; nopage gets two, and the missing one is the
        # manual rather than a dead link to a page that is not there.
        assert links == ["../manuals/p/withpage.html",
                         "deck.html?deck=withpage",
                         "index.html?deck=withpage",
                         "deck.html?deck=nopage",
                         "index.html?deck=nopage"], links
        # A deck with no manual does not get its title linked either.
        assert page.eval_on_selector_all(".wb-title", "els => els.length") == 1
    finally:
        page.close()


def test_a_dead_deck_shows_its_headline_and_is_dimmed(browser, viz_server):
    page = _workbench(browser, viz_server, [
        _deck("gone", status=["retired", "RETIRED", "kept as published"]),
    ])
    try:
        assert page.text_content(".wb-dead").strip() == "RETIRED"
        opacity = page.eval_on_selector(".wb-card", "e => getComputedStyle(e).opacity")
        assert float(opacity) < 0.8, f"a retired deck should read as history, got {opacity}"
    finally:
        page.close()


def test_the_games_chip_reads_the_key_deck_info_actually_writes(browser, viz_server):
    """A logged game must reach the front door.

    `deck-info` writes the record under `info.record` (deck_info.py:137-143);
    `info.status` is the STAGE-COUNT block (deck_info.py:157-161). The workbench
    read `info.status.games`, so the chip was dead code that had never rendered
    once — and by the time it was found, edgar-vampires and ur-dragon each had a
    real logged game the landing page structurally could not show.

    The stub below is the true shape, copied from a committed `info.json`. That
    is the whole point: a test written against the shape the READER expected
    would have passed while the page stayed blank.
    """
    page = _workbench(browser, viz_server, [_deck("played"), _deck("unplayed")], infos={
        "played": {"record": {"games": 3, "win": 1, "loss": 2, "draw": 0,
                              "last_played": "2026-08-22", "undebriefed": []},
                   "status": {"complete": 14, "of": 17, "stale": [], "invalid": [],
                              "missing": []}},
        "unplayed": {"record": {"games": 0, "win": 0, "loss": 0, "draw": 0,
                                "last_played": None, "undebriefed": []},
                     "status": {"complete": 4, "of": 17, "stale": [], "invalid": [],
                                "missing": []}},
    })
    try:
        # `innerText` reflects text-transform, and chips are set in caps.
        chips = page.eval_on_selector_all(
            ".wb-card", "els => els.map(e => e.innerText.toLowerCase())")
        assert len(chips) == 2, chips
        assert "3 games" in chips[0] and "1–2" in chips[0], chips[0]
        # A deck with no games says nothing rather than "0 games · 0–0": an
        # absence is not a result, and the empty state is the rack's job.
        assert "game" not in chips[1].lower(), chips[1]
    finally:
        page.close()


def test_the_fleet_table_reads_the_next_field_nothing_else_reads(browser, viz_server):
    """`info.next` is the derived to-do list and the workbench ignored it.

    `deck-info` already builds a per-deck to-do list where every line names the
    command that would settle it (deck_info.py:254-315), commits it to
    `info.json`, and the front door fetched the file and read one other key.
    The table's last column is that field. If this ever stops rendering, the
    page has gone back to telling you what a deck IS rather than what to do
    with it.
    """
    page = _workbench(browser, viz_server, [
        _deck("alpha", has={"page": True}),
    ], infos={"alpha": {
        "record": {"games": 0, "win": 0, "loss": 0},
        "status": {"complete": 9, "of": 17, "stale": [], "invalid": [], "missing": []},
        "next": ["no games logged — `manamap pilot deck-notes alpha add`"],
    }})
    try:
        page.click('[data-view="table"]')
        page.wait_for_selector(".wb-table")
        assert "no games logged" in page.eval_on_selector(".t-next", "e => e.innerText")
        # stages come from `status`, which is a different block from `record` —
        # the exact confusion that hid the games chip for weeks.
        body = page.eval_on_selector(".wb-table tbody", "e => e.innerText")
        assert "9/17" in body, body
    finally:
        page.close()


def test_each_fleet_sort_puts_the_deck_that_needs_you_first(browser, viz_server):
    """A sort is only useful if it ANSWERS its label.

    Four sorts, four questions. Each is asserted by which deck lands in row
    one, because that is the whole claim a sort makes — not that it ordered
    something, but that the top of the list is where to look.
    """
    decks = [_deck("quiet", has={"page": True}),
             _deck("busy", has={"page": True}),
             _deck("broken", has={"page": True})]
    infos = {
        # played recently, everything else fine
        "quiet": {"record": {"games": 4, "win": 2, "loss": 2, "last_played": "2026-08-23"},
                  "status": {"complete": 17, "of": 17, "stale": [], "invalid": [], "missing": []},
                  "engine": {"critic": "pass", "lines": 9, "verified_lines": 9},
                  "prescriptions": {"count": 0, "answered": 0}, "open_questions": []},
        # never played
        "busy": {"record": {"games": 0, "win": 0, "loss": 0, "last_played": None},
                 "status": {"complete": 17, "of": 17, "stale": [], "invalid": [], "missing": []},
                 "engine": {"critic": "pass", "lines": 9, "verified_lines": 9},
                 "prescriptions": {"count": 6, "answered": 0},
                 "open_questions": [{}, {}, {}]},
        # played, but its evidence is failing and half-built
        "broken": {"record": {"games": 9, "win": 1, "loss": 8, "last_played": "2026-08-01"},
                   "status": {"complete": 4, "of": 17, "stale": ["engine"],
                              "invalid": ["diagnosis"], "missing": []},
                   "engine": {"critic": "fail", "lines": 9, "verified_lines": 1},
                   "diagnosis": {"skeptic": "fail", "stale": True},
                   "prescriptions": {"count": 0, "answered": 0}, "open_questions": []},
    }
    page = _workbench(browser, viz_server, decks, infos=infos)

    def first_row():
        return page.eval_on_selector(".wb-table tbody tr th", "e => e.innerText").lower()

    try:
        page.click('[data-view="table"]')
        page.wait_for_selector(".wb-table")
        for sort, expected in (("played", "quiet"),        # most recent date
                               ("logs", "busy"),           # zero games
                               ("analysis", "broken"),     # failing gates + missing stages
                               ("optimisations", "busy")): # six unanswered prescriptions
            page.click('[data-sort="' + sort + '"]')
            page.wait_for_timeout(150)
            assert expected in first_row(), (sort, first_row())
    finally:
        page.close()


def test_the_fleet_sort_is_in_the_url_so_it_can_be_sent(browser, viz_server):
    """A view worth reaching twice is worth addressing.

    Same reason `?deck=` and `?mode=` are params on the map. `replaceState`,
    not `pushState` — flipping a sort is not a navigation and must not stack
    up in the back button.
    """
    page = _workbench(browser, viz_server, [_deck("alpha", has={"page": True})])
    try:
        page.click('[data-view="table"]')
        page.click('[data-sort="analysis"]')
        page.wait_for_timeout(150)
        url = page.evaluate("location.search")
        assert "view=table" in url and "sort=analysis" in url, url
        # and it survives a reload, which is what "can be sent" means
        page.reload()
        page.wait_for_selector(".wb-table")
        assert page.eval_on_selector('[data-sort="analysis"]',
                                     "e => e.className").find("is-on") >= 0
    finally:
        page.close()


# ── Bars as controls ─────────────────────────────────────────────────────


def _lit_pixels(page):
    """Pixels above a brightness threshold on the force canvas.

    Three measures were tried before this one and the first two are the
    lesson. Summing EVERY pixel moved 3% in the wrong direction while two
    thirds of the nodes were visibly dimmed — 96 dots are a rounding error
    against the ground. Sampling small discs at `Force.screenNodes()` was no
    better: a 12x12 box around a 6px node is mostly background either way, so
    the ratio compressed to nothing. Counting lit pixels canvas-wide is coarse
    but it is the one that actually moves.
    """
    return page.evaluate("""() => {
        const c = document.getElementById('forceCanvas');
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        let n = 0;
        for (let i = 0; i < d.length; i += 4) {
            if (Math.max(d[i], d[i+1], d[i+2]) > 110) n++;
        }
        return n;
    }""")


def test_a_group_spotlight_actually_dims_the_graph(browser, viz_server):
    """A bar you can read but cannot click is a legend that forgot it was one.

    THE THRESHOLD IS MEASURED ON BOTH SIDES, because one chosen only from
    passing runs says nothing about what it catches. On edgar-vampires, against
    a base of 31,313 lit pixels:

        spotlight EVERY row (a no-op)      105%   <- every node gains a ring
        spotlight 1/7 of the deck           84%
        spotlight 1/20                      82%
        released                           100%   exact, to the pixel

    So 90% sits between a real spotlight and a spotlight that does nothing —
    and note the no-op reads ABOVE base rather than at it, which is why "did
    the number go down" had to be checked rather than assumed.

    Finding this also fixed two real rendering bugs. The node fill was dimmed
    and then `globalAlpha` was reset to 1 before the rim, so in Build — where
    every deck card carries a white rim — a spotlight left 96 bright outlines
    on screen; `setLine` had shipped with the same defect. And a group
    spotlight left every edge at full strength, so the graph stayed a bright
    web while the nodes receded.
    """
    page = _build_page(browser, viz_server, "edgar-vampires")
    try:
        assert page.js_errors == [], page.js_errors
        base = _lit_pixels(page)

        # The SMALLEST group on the curve. The first is `Creature`, and edgar
        # is a creature deck — spotlighting it dims almost nothing, so the
        # measurement would be about the deck rather than about the feature.
        key = page.evaluate(r"""() => {
            const segs = [...document.querySelectorAll('.curve-seg[data-group]')];
            const n = {};
            for (const s of segs) {
                const m = /\u00d7 (\d+)/.exec(s.getAttribute('title') || '');
                if (m) n[s.getAttribute('data-group')] =
                    (n[s.getAttribute('data-group')] || 0) + Number(m[1]);
            }
            return Object.keys(n).sort((a, b) => n[a] - n[b])[0];
        }""")
        assert key, "no curve segment rendered"

        page.query_selector('.curve-seg[data-group="%s"]' % key).click()
        page.wait_for_timeout(800)
        assert page.js_errors == [], page.js_errors
        assert page.evaluate("Build.focusedGroup && Build.focusedGroup.key") == key
        during = _lit_pixels(page)
        assert during < base * 0.9, (base, during, key)

        # Clicking the same segment releases it, and the graph comes back.
        page.query_selector('.curve-seg[data-group="%s"]' % key).click()
        page.wait_for_timeout(700)
        assert page.evaluate("Build.focusedGroup") is None
        after = _lit_pixels(page)
        assert after > during * 1.05, (during, after)
    finally:
        page.close()


def test_a_role_bar_switches_the_overlay_because_it_is_always_about_roles(browser, viz_server):
    """Clicking "ramp" while the map is coloured by supertype must mean
    "colour by role AND light up ramp" — not "light up a supertype called
    ramp", which is nothing at all.

    `role` is also the one grouping whose data is NOT in the boot payload
    (`card_roles.json`, 0.39 MB gz, loaded on selection), so this exercises the
    `ensure()` path: without it every card colours 'unclassified' and reads as
    a broken roles file rather than an absent one.
    """
    page = _build_page(browser, viz_server, "edgar-vampires")
    try:
        assert page.evaluate("MM.grouping") != "role"
        row = page.query_selector(".lens-bar-row[data-role]")
        assert row is not None, "no role bar rendered"
        family = row.get_attribute("data-role")
        row.click()
        page.wait_for_function("() => window.Build && Build.focusedGroup",
                               timeout=BOOT_TIMEOUT_MS)
        page.wait_for_timeout(900)
        assert page.js_errors == [], page.js_errors
        assert page.evaluate("MM.grouping") == "role"
        assert page.evaluate("Build.focusedGroup.key") == family
        # The select is the other control for the same state; two controls
        # disagreeing about one value is how a legend ends up lying.
        assert page.eval_on_selector("#colorBy", "e => e.value") == "role"
        # And the spotlight really reached the graph, not just the panel.
        assert page.evaluate("Force.nodeCount") > 0
    finally:
        page.close()


def test_a_group_spotlight_and_a_line_spotlight_are_not_the_same_thing(browser, viz_server):
    """Two answers to "show me", and holding both means neither is legible.

    They are also deliberately different in scope: a LINE is a claim about
    edges, so it mutes every edge that is not part of it; a GROUP is a claim
    about nodes, and muting the deck's verified lines while you look at its
    ramp would hide the thing that makes the graph worth reading. Taking one
    must put the other down.
    """
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck with a drawable verified line")
    page = _build_page(browser, viz_server, slug)
    try:
        seg = page.query_selector(".curve-seg[data-group]")
        assert seg is not None, "no curve segment rendered"
        seg.click()
        page.wait_for_timeout(600)
        assert page.evaluate("Build.focusedGroup") is not None

        page.evaluate("Build.focusLine(0)")
        page.wait_for_timeout(600)
        assert page.evaluate("Build.focusedGroup") is None, \
            "taking a line must put the group down"
        assert page.js_errors == [], page.js_errors
    finally:
        page.close()


def test_engine_lines_give_verified_edges_a_direction(browser, viz_server):
    """A clique has no arrows; `engine.json` is the one artifact that knows.

    A verified line becomes a clique over the cards its stack names, so
    `{source, target}` is whichever order the pair was built in — drawing an
    arrow on that would be array order wearing a claim. The engine model
    declares each line as `from -> to` across two of eight stages with a
    `carries` noun, written by an engineer and attacked by a critic.

    So only the pairs that actually SPAN the two stages get an arrowhead. A
    pair sitting wholly inside one stage stays undirected, which is honest:
    for that pair the direction genuinely is not known.
    """
    page = _build_page(browser, viz_server, "ur-dragon")
    try:
        assert page.js_errors == [], page.js_errors
        links = page.evaluate("Force.links()")
        directed = [l for l in links if l.get("dir")]
        assert directed, "no verified edge picked up a direction from engine.json"
        # Every directed edge names what it moves — that is the whole point of
        # the arrow, and an unlabelled one would just be a prettier line.
        assert all(l.get("carries") for l in directed), directed
        # And most edges stay undirected: the arrow is the exception, earned.
        assert len(directed) < len(links)
    finally:
        page.close()


def test_a_stack_carrying_two_engine_lines_keeps_both_nouns(browser, viz_server):
    """ur-dragon's stack 002 is cited by two lines — `bodies` and `triggers`.

    Taking the first match would silently drop half of what that board proves.
    They agree on direction, so the nouns are joined; if two lines citing one
    stack DISAGREED on direction, no arrowhead is drawn at all, because a pair
    pointing two ways is a pair whose direction is not a fact.
    """
    page = _build_page(browser, viz_server, "ur-dragon")
    try:
        carries = page.evaluate(
            "[...new Set(Force.links().filter(l => l.dir).map(l => l.carries))]")
        assert any(" · " in (c or "") for c in carries), carries
    finally:
        page.close()


# ── The open line explains itself ──────────────────────────────────────────
#
# `buildEdges` kept a stack's `title` and dropped everything else, so the panel
# could NAME a verified line and never say what it does. Every word it needed
# was already on the wire: Build fetches whole stack documents, and 50 of 50
# published stacks carry `resolution.final_state.summary`.


def test_the_open_line_shows_its_prose(browser, viz_server):
    """Clicking a line explains it, and closing it takes the explanation away."""
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck with a drawable verified line")
    page = _build_page(browser, viz_server, slug)
    try:
        assert page.locator(".lens-line-prose").count() == 0, (
            "prose before anything was opened")

        page.click(".lens-line")
        page.wait_for_timeout(800)

        # Exactly ONE block, under the row that is open. The median summary is
        # 838 characters and the longest is 4,337; rendering every line's prose
        # at once turns the list you choose from into a wall of text.
        assert page.locator(".lens-line-prose").count() == 1
        assert page.locator(".lens-line.is-on .lens-line-prose").count() == 1

        text = page.locator(".lens-line-prose").inner_text()
        assert len(text) > 200, f"prose block is only {len(text)} chars"

        page.click(".lens-line")
        page.wait_for_timeout(600)
        assert page.locator(".lens-line-prose").count() == 0
        assert page.js_errors == []
    finally:
        page.close()


def test_the_prose_is_the_artifact_verbatim(browser, viz_server):
    """A checker read these words.

    The panel may choose which of the three sources to print and may leave one
    out — it must never re-word one. Summarising a resolution in the browser
    puts a ✓ over prose no checker saw, which is the same mistake as editing a
    resolution's step text to fix a stale cross-reference.
    """
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck with a drawable verified line")
    page = _build_page(browser, viz_server, slug)
    try:
        page.click(".lens-line")
        page.wait_for_timeout(800)
        shown = page.locator(".lens-line-prose").inner_text()

        # Pull the same line's artifact prose out of Build's own state, then
        # assert the rendered block contains it whole.
        src = page.evaluate("Build.__lineProse ? Build.__lineProse(0) : null")
        assert src, "Build did not expose the open line's prose"
        for key in ("note", "answer", "summary"):
            val = src.get(key)
            if not val:
                continue
            # innerText collapses the source's newlines; compare a run that has
            # none rather than weakening the assertion to a prefix.
            probe = max(val.split("\n"), key=len).strip()
            assert probe and probe in shown, f"{key} was not printed verbatim"
        assert page.js_errors == []
    finally:
        page.close()


def test_the_fade_only_promises_more_when_there_is_more(browser, viz_server):
    """The fade at the bottom of the prose block means "keep scrolling".

    It has to come off in the two states where that is false: scrolled to the
    bottom, and a block shorter than the 340px cap — which never fires a scroll
    event at all, so it cannot be settled by the scroll handler. One line in the
    fleet is that short (goblin-storm 002, 877 characters), and a fade over the
    end of a complete text is a lie about there being more.
    """
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck with a drawable verified line")
    page = _build_page(browser, viz_server, slug)
    try:
        page.click(".lens-line")
        page.wait_for_timeout(800)
        state = page.evaluate("""() => {
            const el = document.querySelector('.lens-line-prose');
            const over = el.scrollHeight > el.clientHeight + 2;
            const top = { over, end: el.classList.contains('is-end') };
            el.scrollTop = el.scrollHeight;
            el.dispatchEvent(new Event('scroll'));
            return { top, end: el.classList.contains('is-end') };
        }""")
        # Overflowing: faded at the top, not at the bottom. Not overflowing:
        # never faded. Both are "the fade is on iff there is more below".
        assert state["top"]["end"] == (not state["top"]["over"])
        assert state["end"] is True, "fade survived scrolling to the end"
        assert page.js_errors == []
    finally:
        page.close()


# ── Starting a walk from cards you name ────────────────────────────────────


def test_a_comma_stays_inside_a_card_name(discover_page):
    """3,222 of 34,890 card names contain a comma — 9.2%, and they are the
    legendary creatures somebody would actually seed a walk with.

    Splitting on commas turns the commonest input into two cards that do not
    exist, and it fails silently: the graph comes up short with no explanation.
    So the ENUMERATION marker separates items, never a bare comma.
    """
    page = discover_page
    got = page.evaluate("""() => {
        const P = Discovery.parseSeedNames;
        return {
            one:   P('Miirym, Sentinel Wyrm').rows.length,
            two:   P('1) Miirym, Sentinel Wyrm, 2) Sol Ring').rows.length,
            lines: P('Miirym, Sentinel Wyrm\\nSol Ring').rows.length,
            // The pilot's own shorthand.
            short: P('1) sol ring, 2) lightning bolt').rows.length,
        };
    }""")
    assert got["one"] == 1, "a comma inside a name split it in two"
    assert got["two"] == 2, "a numbered marker did not separate two items"
    assert got["lines"] == 2
    assert got["short"] == 2
    assert page.js_errors == []


def test_a_marker_inside_a_name_is_not_a_separator(discover_page):
    """Eight corpus names carry an enumeration-looking marker inside them —
    "Vault 87: Forced Evolution" and its five Fallout siblings.

    A naive /\\d+[).:]/ cuts those in half. The marker only separates where an
    item can BEGIN: at the start, after a newline, or after a comma.
    """
    page = discover_page
    got = page.evaluate(
        "() => Discovery.parseSeedNames('Vault 87: Forced Evolution').rows.length")
    assert got == 1, "an enumeration-like marker inside a card name split it"
    assert page.js_errors == []


def test_naming_cards_never_deletes_the_walk(discover_page):
    """GROWING MUST NEVER BE ABLE TO DELETE.

    `Force.enter([row])` REBUILDS — it replaces the graph with that one card.
    Two callers have shipped this bug already, each destroying a walk silently.
    So "Add to walk" adopts, and only "Start here" reseeds; both are buttons
    that say which one they are.
    """
    page = discover_page
    got = page.evaluate("""async () => {
        // Build a walk worth losing.
        await Discovery.seedFromRows(
            Discovery.parseSeedNames('Sol Ring\\nLightning Bolt').rows, 'seed', {});
        Force.branchByRow(Force.rows()[0], 'similar');
        await new Promise(r => setTimeout(r, 900));
        const before = Force.rows().slice();

        // ADD — every card that was there must still be there.
        await Discovery.seedFromRows(
            Discovery.parseSeedNames('Birds of Paradise').rows, 'add', {replace: false});
        const after = new Set(Force.rows());
        const grew = { before: before.length, after: after.size,
                       survived: before.every(r => after.has(r)) };

        // REPLACE — the explicit request, and it does replace.
        await Discovery.seedFromRows(
            Discovery.parseSeedNames('Sol Ring').rows, 'fresh', {});
        return { grew, replaced: Force.nodeCount };
    }""")
    assert got["grew"]["before"] > 2, "the walk under test never grew"
    assert got["grew"]["survived"], "adding a named card deleted part of the walk"
    assert got["grew"]["after"] == got["grew"]["before"] + 1
    assert got["replaced"] == 1, "Start here did not start over"
    assert page.js_errors == []


def test_an_unresolved_name_is_reported_not_dropped(discover_page):
    """A typo that quietly yields a one-card walk instead of two is
    indistinguishable from the feature not working."""
    page = discover_page
    got = page.evaluate(
        "() => Discovery.parseSeedNames('Sol Ring\\nNot A Real Card')")
    assert len(got["rows"]) == 1
    assert got["missing"] == ["Not A Real Card"]
    assert page.js_errors == []


def test_a_walk_can_be_linked_to(browser, viz_server):
    """`?cards=` is the shareable form, and it seeds ONCE.

    Seeding twice — show() then a reseed — draws the first card, throws it away
    and draws the set, which reads as a flicker on the first frame the pilot
    ever sees.
    """
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        page.goto(f"{viz_server}/viz/index.html"
                  "?cards=1)%20Miirym,%20Sentinel%20Wyrm,%202)%20Sol%20Ring")
        page.wait_for_function(
            "() => window.Force && Force.nodeCount >= 2", timeout=30000)
        assert page.evaluate("MM.mode") == "discover"
        assert page.evaluate("Force.nodeCount") == 2
        assert errors == []
    finally:
        page.close()


# ── The library persists ───────────────────────────────────────────────────
#
# PRD §7.1: the library is "the connective tissue for the whole flow" and must
# survive navigation between surfaces. It previously did not — `Session.tray`
# was an in-memory array that died on every reload.


def test_the_library_survives_a_reload(discover_page, viz_server):
    """The whole point of persisting it."""
    page = discover_page
    kept = page.evaluate("""async () => {
        const rows = ['Sol Ring', 'Command Tower', 'Rhystic Study']
            .map(Discovery.rowByName).filter(r => r >= 0);
        rows.forEach(r => { if (!Session.library.has(r)) Session.library.toggle(r); });
        return Discovery.library.names();
    }""")
    assert len(kept) == 3

    page.reload()
    page.wait_for_function(
        "() => window.Discovery && Discovery.isReady() && Discovery.current >= 0",
        timeout=30000)
    back = page.evaluate("Discovery.library.names()")
    assert sorted(back) == sorted(kept), f"library did not survive: {back}"
    assert page.js_errors == []


def test_the_library_stores_names_not_row_indices(discover_page):
    """THE bug that deleted the previous attempt at this.

    `localStorage['manamap-deck']` stored raw positional row indices with no
    schema version. A Scryfall refresh reorders `cards.csv`, every index shifts,
    and a saved deck silently reinterprets as DIFFERENT CARDS — no error, no
    warning, a plausible wrong answer.

    Asserted against the stored bytes rather than the behaviour, because the
    behaviour only diverges after a corpus refresh and by then it is too late.
    """
    page = discover_page
    raw = page.evaluate("""() => {
        Session.library.clear();
        const r = Discovery.rowByName('Sol Ring');
        Session.library.toggle(r);
        return {stored: localStorage.getItem('manamap-library'), row: r};
    }""")
    doc = json.loads(raw["stored"])
    assert doc["v"] == 1, "no schema version — the exact omission that sank the last one"
    assert doc["cards"] == ["Sol Ring"], f"stored {doc['cards']!r}, expected names"
    assert raw["row"] not in doc["cards"], "a row index reached the store"


def test_a_card_that_left_the_corpus_is_reported_not_dropped(discover_page):
    """A library that quietly comes back two cards short is indistinguishable
    from one that came back whole. `rowOf` answers -1 for an unknown name — the
    range check the deleted version did not have."""
    page = discover_page
    report = page.evaluate("""() => {
        localStorage.setItem('manamap-library', JSON.stringify({
            v: 1, corpus: 'whatever',
            cards: ['Sol Ring', 'A Card That Was Never Printed', 'Command Tower'],
        }));
        return Session.useCards({
            nameOf: r => (MM.cardRecord(r) || {}).n || null,
            rowOf: Discovery.rowByName,
            fingerprint: () => 'now',
        });
    }""")
    assert report["restored"] == 2
    assert report["missing"] == ["A Card That Was Never Printed"]
    assert report["corpusChanged"] is True, "a changed fingerprint must be visible"
    assert page.js_errors == []


def test_an_unknown_schema_is_ignored_rather_than_guessed_at(discover_page):
    """A newer build's data must survive an older build reading it. Upgrading a
    shape you do not know is how you turn someone's saved work into garbage."""
    page = discover_page
    out = page.evaluate("""() => {
        localStorage.setItem('manamap-library', JSON.stringify({
            v: 99, cards: ['Sol Ring'], somethingNew: true}));
        const report = Session.useCards({
            nameOf: r => (MM.cardRecord(r) || {}).n || null,
            rowOf: Discovery.rowByName, fingerprint: () => 'now'});
        return {report: report, stillThere: localStorage.getItem('manamap-library')};
    }""")
    assert out["report"]["restored"] == 0
    assert out["report"]["schema"] == 99
    assert "somethingNew" in out["stillThere"], "the unknown document was destroyed"
    assert page.js_errors == []


# ── The shell: one strip, three surfaces ───────────────────────────────────


def _shell_page(browser, viz_server, path):
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{viz_server}{path}")
    page.wait_for_selector("#shell .shell-nav", timeout=BOOT_TIMEOUT_MS)
    page.js_errors = errors
    return page


@pytest.mark.parametrize("path,here", [
    ("/viz/workbench.html", "Workbench"),
    ("/viz/index.html", "Atlas"),
])
def test_the_shell_marks_where_you_are(browser, viz_server, path, here):
    """The surface you are on is not a link.

    A nav that offers to take you where you already are teaches the reader the
    nav is decorative — and it must be MARKED, not merely unlinked, because an
    unlinked word beside links reads as broken rather than as current.
    """
    page = _shell_page(browser, viz_server, path)
    try:
        current = page.locator("#shell .shell-here").first.inner_text()
        assert current.strip().lower() == here.lower()
        hrefs = page.eval_on_selector_all(
            "#shell .shell-nav a", "els => els.map(e => e.getAttribute('href'))")
        assert hrefs, "the shell offered no way out"
        assert not any(path.endswith(h) for h in hrefs), \
            f"the current surface is still a link: {hrefs}"
        assert page.js_errors == []
    finally:
        page.close()


def test_the_shell_needs_neither_session_nor_the_atlas(browser, viz_server):
    """It lives on `workbench.html`, which loads neither.

    A shell that needed them would either pull 0.56 MB of card index onto a page
    that draws no cards, or exist on one page and stop being a shell.
    """
    page = _shell_page(browser, viz_server, "/viz/workbench.html")
    try:
        state = page.evaluate("""() => ({
            session: !!window.Session, mm: !!window.MM, shell: !!window.Shell,
        })""")
        assert state["shell"] is True
        assert state["session"] is False and state["mm"] is False
        assert page.js_errors == []
    finally:
        page.close()


def test_the_library_count_crosses_surfaces(browser, viz_server):
    """The count is readable with no corpus BECAUSE the library stores names.

    Had it stored row indices — the form that got the previous attempt deleted —
    a page would have to load the whole index to say "4". That is the quieter
    second argument for names, and this test is where it is asserted.
    """
    page = _shell_page(browser, viz_server, "/viz/workbench.html")
    try:
        shown = page.evaluate("""() => {
            localStorage.setItem('manamap-library', JSON.stringify({
                v: 1, corpus: null,
                cards: ['Sol Ring', 'Rhystic Study', 'Command Tower', 'Cyclonic Rift'],
            }));
            Shell.refresh();
            return document.querySelector('.shell-library').innerText;
        }""")
        assert "4" in shown, f"shell shows {shown!r}"
        assert page.js_errors == []
    finally:
        page.close()


def test_the_count_is_never_one_behind(discover_page):
    """SAVE BEFORE EMIT, and this test exists because it was the other way.

    The shell reads the count from `localStorage`, so emitting first meant it
    read the PREVIOUS save: four cards in, three on the strip, permanently one
    behind. It looked right at a glance — plausible number, wrong by one — and
    is invisible unless you compare the strip against the store, which is what
    this does.
    """
    page = discover_page
    r = page.evaluate("""() => {
        Session.library.clear();
        const rows = ['Sol Ring', 'Rhystic Study', 'Command Tower', 'Cyclonic Rift']
            .map(Discovery.rowByName).filter(x => x >= 0);
        rows.forEach(x => Session.library.toggle(x));
        const stored = JSON.parse(localStorage.getItem('manamap-library')).cards;
        return {stored: stored.length,
                shown: document.querySelector('.shell-library').innerText};
    }""")
    assert r["stored"] == 4
    assert r["shown"].strip().startswith("4"), (
        f"the strip says {r['shown']!r} while {r['stored']} are stored — "
        f"save() must run before emit()")
    assert page.js_errors == []


# ── Harvesting from a reference deck (PRD §6.1 steps 9-10) ─────────────────


def test_a_reference_deck_is_not_loaded_as_your_deck(browser, viz_server, tmp_path):
    """THE distinction step 10 rests on.

    A reference deck is somebody else's list, fetched by the CLI because the
    page cannot reach EDHREC. Loading it with `opts.deck` would ring a
    commander, ink the cards as yours and put all eighty in your library — the
    opposite of harvesting a few out of another brew. It seeds the graph and
    nothing else; the library only grows when you Keep something.
    """
    import json as _json
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    ref_dir = root / "data" / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    slug = "zz-test-reference"
    (ref_dir / f"{slug}.json").write_text(_json.dumps({
        "slug": slug, "commander": "Test Commander",
        "cards": ["Sol Ring", "Arcane Signet", "Command Tower",
                  "Rhystic Study", "Cyclonic Rift"],
        "unresolved": [], "source": "test",
    }))
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        page.goto(f"{viz_server}/viz/index.html?ref={slug}")
        page.wait_for_function("() => window.Force && Force.nodeCount >= 4",
                               timeout=BOOT_TIMEOUT_MS)
        state = page.evaluate("""() => {
            Session.library.clear();
            return {nodes: Force.nodeCount, library: Session.library.size,
                    commander: Session.commander};
        }""")
        assert state["nodes"] >= 4, "the reference deck did not seed the graph"
        assert state["library"] == 0, (
            "loading a reference deck put its cards in your library — that is "
            "loading it AS your deck, not harvesting from it")
        assert state["commander"] == -1, "a reference deck claimed the commander ring"

        # …and harvesting one card is exactly one card.
        grew = page.evaluate("""() => {
            const before = Session.library.size;
            Session.library.toggle(Force.rows()[0]);
            return {before: before, after: Session.library.size};
        }""")
        assert grew["after"] == grew["before"] + 1
        assert errors == []
    finally:
        page.close()
        (ref_dir / f"{slug}.json").unlink(missing_ok=True)


def test_a_missing_reference_deck_says_so(browser, viz_server):
    """`data/reference/` is gitignored and local-only, so a deployed page will
    never have one. "Nothing happened" is the wrong way to learn that — the
    static build must degrade VISIBLY, not silently (PRD §5.2)."""
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        page.goto(f"{viz_server}/viz/index.html?ref=zz-does-not-exist")
        page.wait_for_function(
            "() => (document.getElementById('status')||{}).innerText"
            "      && document.getElementById('status').innerText.includes('reference')",
            timeout=BOOT_TIMEOUT_MS)
        status = page.inner_text("#status")
        assert "commander-search" in status, (
            f"the failure did not name the command that fixes it: {status!r}")
        assert errors == []
    finally:
        page.close()


# ── The toolbar is mode-aware ──────────────────────────────────────────────


def test_the_toolbar_shows_only_what_the_mode_uses(page):
    """It carried the same 17 controls in every mode.

    Discover is a graph of ONE CARD and was showing nine type-filter chips, a
    density-contour toggle and "Color by" — none of which act on anything
    there. Roughly half the surface was inert at any moment, which is most of
    why the atlas felt heavy. Measured after: Discover 4, Build 14, Explore 19.
    """
    counts = {}
    for mode in ("discover", "build", "explore"):
        page.evaluate(f"MM.setMode('{mode}')")
        page.wait_for_timeout(400)
        counts[mode] = page.evaluate("""() => [...document.querySelectorAll(
            '.toolbar select, .toolbar input, .toolbar button')]
            .filter(e => e.offsetParent !== null).length""")
    assert counts["discover"] < counts["build"] < counts["explore"], counts
    assert counts["discover"] <= 6, (
        f"Discover shows {counts['discover']} controls — it needs the mode tabs "
        f"and a search box, and nothing that acts on a map it is not showing")
    assert page.js_errors == []


def test_the_mode_select_stays_the_one_source_of_truth(page):
    """The tabs are a VIEW of `modeSelect`, never a second answer.

    `?mode=` applies through it and the browser suite reads it; two places
    holding "which mode is current" is the bug this repo keeps undoing.
    """
    page.evaluate("document.querySelector('.mode-tab[data-mode=\\'build\\']').click()")
    page.wait_for_timeout(500)
    assert page.evaluate("document.getElementById('modeSelect').value") == "build"
    assert page.evaluate("MM.mode") == "build"
    marked = page.eval_on_selector_all(
        ".mode-tab.is-on", "els => els.map(e => e.getAttribute('data-mode'))")
    assert marked == ["build"], marked
    assert page.js_errors == []


def test_the_landing_leads_with_the_card_and_its_relations(discover_page):
    """The card used to render NINTH, under sixteen equal-weight buttons.

    The landing is a card and what you can do with it; everything else is a way
    of choosing a different card, which is a smaller question and now looks
    like one. Asserted by ORDER on screen, not by DOM order — the fix is a flex
    `order`, so reading the markup would pass while the page looked unchanged.
    """
    page = discover_page
    pos = page.evaluate("""() => {
        const y = s => { const e = document.querySelector(s);
                         return e ? e.getBoundingClientRect().top : null; };
        return {card: y('#deckInner .detail-card-image'),
                relations: y('.discover-relations'),
                keep: y('.discover-keep'),
                details: y('#deckInner .detail-section'),
                more: y('#deckInner details.discover-more')};
    }""")
    assert pos["card"] is not None and pos["relations"] is not None
    assert pos["card"] < pos["relations"] < pos["keep"], pos
    assert pos["relations"] < pos["details"], (
        "the oracle text is between the card and the action again — every word "
        "of it is already legible on the card image above")
    assert pos["keep"] < pos["more"], "the ways to start elsewhere outrank Keep"
    assert page.js_errors == []


def test_the_card_is_not_clipped_by_its_art_crop_frame(discover_page):
    """`.detail-card-image` is a 200px art-crop window with `overflow:hidden` —
    right for a hover preview, wrong for a landing that leads with the card.

    Capping the IMG alone overflowed that window and sliced the card through its
    rules text, which reads as broken rather than as cropped. And removing the
    frame's `min-height` then let it collapse to nothing, because making the
    panel a column flex container had quietly made every child shrinkable.
    """
    page = discover_page
    # Wait on the IMAGE's own readiness, not on a timer. `discover_page` waits
    # for `Discovery.isReady()`, which is about the card INDEX; the art comes
    # from Scryfall afterwards, and an unloaded <img> with `height:auto` measures
    # 0. A sleep here would be the sixth member of this repo's flake family —
    # every one a wait on the machine rather than on the behaviour.
    page.wait_for_function(
        """() => { const i = document.querySelector('#deckInner .detail-card-image img');
                   return i && i.complete && i.naturalHeight > 0; }""",
        timeout=BOOT_TIMEOUT_MS)
    box = page.evaluate("""() => {
        const w = document.querySelector('#deckInner .detail-card-image');
        const i = w.querySelector('img');
        return {wrap: w.getBoundingClientRect().height,
                img: i.getBoundingClientRect().height,
                natural: i.naturalHeight};
    }""")
    assert box["img"] > 250, f"the card is only {box['img']}px tall"
    assert box["wrap"] >= box["img"] - 2, (
        f"the frame ({box['wrap']}px) is shorter than the card ({box['img']}px) "
        f"— it is clipping again")
    assert page.js_errors == []


# ── The local bridge, seen from the page ───────────────────────────────────


def test_agent_affordances_are_absent_not_broken_without_a_server(browser, viz_server):
    """The PRD's rule for the static build, and the answer to `CLAUDE.md`'s
    objection that a local bridge makes two products.

    It does make two, and the page is the thing that says which one you are
    holding. The browser suite's `viz_server` is a plain static server with no
    `/api`, which is exactly the deployed shape — so this test runs in the
    condition it is about rather than simulating it.
    """
    # `_build_page` is the helper every other Build test uses — it waits for the
    # graph to have nodes, which is the signal that a deck actually loaded.
    # Hand-rolling the wait here read "Loading artifacts…" and reported a
    # missing message that had simply not been written yet.
    page = _build_page(browser, viz_server, "heliod")
    errors = page.js_errors
    try:
        page.wait_for_function("() => window.Api && Api.probed", timeout=BOOT_TIMEOUT_MS)
        state = page.evaluate("() => ({ready: Api.ready, reason: Api.reason})")
        assert state["ready"] is False
        assert "manamap serve" in (state["reason"] or ""), state
        body = page.inner_text("#deckInner")
        assert "manamap serve" in body, (
            "the static build did not say what is missing — absent is fine, "
            "silent is not")
        assert page.query_selector("#bdAsk") is None, (
            "an Ask box was offered with no server behind it")
        assert errors == [], errors
    finally:
        page.close()


def test_the_probe_happens_once(browser, viz_server):
    """A page on Pages would otherwise spend its life issuing failing requests
    to an origin that has no server."""
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    try:
        calls = []
        page.on("request", lambda r: calls.append(r.url) if "/api/" in r.url else None)
        page.goto(f"{viz_server}/viz/index.html")
        page.wait_for_function("() => window.Api && Api.probed", timeout=BOOT_TIMEOUT_MS)
        page.wait_for_timeout(1500)
        page.evaluate("() => Api.probe()")
        page.evaluate("() => Api.probe()")
        page.wait_for_timeout(400)
        assert len(calls) <= 2, f"{len(calls)} API requests: {calls}"
    finally:
        page.close()


def test_the_new_deck_form_never_declines_in_silence(browser, viz_server):
    """"I tried to build a Standard deck and nothing happened."

    `newDeckBuild` began `if (!newDeck.slug) return;` — a silent early return,
    which is exactly that report. A control that declines has to say why, or it
    reads as broken software rather than as a limit.

    Driven through the real functions with the API stubbed, because the point is
    the DECLINING path and a live server would build a deck instead.
    """
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        page.goto(f"{viz_server}/viz/index.html?mode=build")
        page.wait_for_function("() => window.Build && window.Api && Api.probed",
                               timeout=BOOT_TIMEOUT_MS)
        said = page.evaluate("""async () => {
            // Stub the API as present, with the real shape `formats` returns.
            Object.defineProperty(Api, 'ready', {get: () => true, configurable: true});
            Api.call = function (name) {
                if (name === 'formats') return Promise.resolve({formats: [
                    {key: 'commander', name: 'Commander', deck_size: 100,
                     exact_size: true, singleton: true, commanders: 1,
                     buildable: true},
                    {key: 'standard', name: 'Standard', deck_size: 60,
                     exact_size: false, singleton: false, commanders: 0,
                     buildable: false}]});
                return Promise.resolve({});
            };
            Build.newDeck();
            await new Promise(r => setTimeout(r, 300));
            const offered = [...document.querySelectorAll('#ndFmt option')]
                .map(o => o.value);
            Build.newDeckBuild();                 // no slug — the reported case
            await new Promise(r => setTimeout(r, 150));
            const complaint = (document.querySelector('.lens-line-prose') || {}).innerText;
            const notes = [...document.querySelectorAll('.lens-note')]
                .map(e => e.textContent).join(' | ');
            return {offered, complaint, notes};
        }""")
        assert said["complaint"], "Build declined silently — the reported bug"
        assert "slug" in said["complaint"].lower()
        # A menu is a promise; it may only offer what it can keep.
        assert said["offered"] == ["commander"], said["offered"]
        assert "not built yet" in said["notes"], (
            "the unbuildable formats vanished without explanation, which is a "
            "different way of being silent")
        assert errors == [], errors
    finally:
        page.close()


def test_typing_a_slug_does_not_create_a_deck_per_keystroke(browser, viz_server):
    """Typing "zur-enchantress" left `zur`, `zur-en`, `zur-enchan` and
    `zur-enchantress` on the bench — three of them junk.

    The slug names a DIRECTORY. Every other field can autosave noisily because
    it edits a draft that already exists; this one decides WHICH draft that is,
    so it commits on `change` (blur or enter) rather than on `input`.
    """
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        page.goto(f"{viz_server}/viz/index.html?mode=build")
        page.wait_for_function("() => window.Build && window.Api && Api.probed",
                               timeout=BOOT_TIMEOUT_MS)
        saved = page.evaluate("""async () => {
            const calls = [];
            Object.defineProperty(Api, 'ready', {get: () => true, configurable: true});
            Api.call = function (name, payload) {
                if (name === 'build/save') calls.push(payload.slug);
                if (name === 'formats') return Promise.resolve({formats: [
                    {key: 'commander', name: 'Commander', deck_size: 100,
                     exact_size: true, singleton: true, commanders: 1,
                     buildable: true}]});
                return Promise.resolve({slug: payload && payload.slug});
            };
            Build.newDeck();
            await new Promise(r => setTimeout(r, 300));
            const box = document.getElementById('ndSlug');
            // Type it, character by character, exactly as a person does.
            for (const ch of 'zur-ench') {
                box.value += ch;
                box.dispatchEvent(new Event('input', {bubbles: true}));
                await new Promise(r => setTimeout(r, 90));
            }
            await new Promise(r => setTimeout(r, 900));   // past the debounce
            const duringTyping = calls.length;
            box.dispatchEvent(new Event('change', {bubbles: true}));   // blur
            await new Promise(r => setTimeout(r, 900));
            return {duringTyping, after: calls.slice()};
        }""")
        assert saved["duringTyping"] == 0, (
            f"typing saved {saved['duringTyping']} draft(s) — one directory per "
            f"prefix is exactly the reported bug")
        assert saved["after"] == ["zur-ench"], saved["after"]
        assert errors == [], errors
    finally:
        page.close()
