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

import pytest

from conftest_viz import (  # noqa: F401
    BOOT_TIMEOUT_MS, browser, canvas_page, page, viz_server,
)

pytestmark = pytest.mark.browser


# ── Boot ────────────────────────────────────────────────────────────────


def test_map_boots_clean(page):
    assert page.evaluate("MM.allData.length") == 34322
    traces = page.evaluate("document.getElementById('plot').data.length")
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
        const dt = (gd.data || []).find(t => t._isDrill);
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
            drillTraces: (gd.data || []).filter(t => t._isDrill).length,
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
        const gd = document.getElementById('plot');
        return {
            active: Drill.isActive(),
            drillTraces: (gd.data || []).filter(t => t._isDrill).length,
            hiddenBaseTraces: (gd.data || []).filter(t => t.visible === false).length,
            annotations: (gd._fullLayout.annotations || []).length,
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
            marker: (gd.data || []).filter(t => t._isBrowseCurrent).length,
            setTrace: (gd.data || []).filter(t => t.name && t.name.startsWith('Selection')).length,
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
        const snap = () => { const t = gd.data.find(x => x._isBrowseCurrent);
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


def test_filtering_does_not_reset_the_zoom(page):
    """Before the fix: zoom to a span of 20.5, toggle a filter, get 116.6."""
    result = page.evaluate("""async () => {
        const gd = document.getElementById('plot');
        const span = () => { const r = gd._fullLayout.xaxis.range;
                             return Math.abs(r[1] - r[0]); };
        await Plotly.relayout('plot', {'xaxis.range': [-5, 5], 'yaxis.range': [-5, 5]});
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
        const gd = document.getElementById('plot');
        const span = () => { const r = gd._fullLayout.xaxis.range;
                             return Math.abs(r[1] - r[0]); };
        await Plotly.relayout('plot', {'xaxis.range': [-5, 5], 'yaxis.range': [-5, 5]});
        const zoomed = span();
        const ms = document.getElementById('mapSelect');
        ms.value = 'ability'; ms.dispatchEvent(new Event('change'));
        await new Promise(r => setTimeout(r, 12000));
        return {zoomed, after: span(), status: document.getElementById('status').textContent};
    }""")
    assert page.js_errors == []
    assert result["after"] > result["zoomed"] * 2, "the map switch kept a stale camera"
    assert "Abilities" in result["status"]


# ── Modes ───────────────────────────────────────────────────────────────


def test_deck_lens_lights_a_deck(page):
    result = page.evaluate("""async () => {
        document.getElementById('modeSelect').value = 'deck';
        MM.setMode('deck');
        await new Promise(r => setTimeout(r, 4000));
        await DeckMap.select('edgar-vampires');
        await new Promise(r => setTimeout(r, 2000));
        const gd = document.getElementById('plot');
        const base = gd.data.find(t => !t._isDeckOverlay && t.marker);
        return {
            overlays: gd.data.filter(t => t._isDeckOverlay).length,
            commander: gd.data.filter(t => t.name === 'Commander').length,
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
        document.getElementById('modeSelect').value = 'deck'; MM.setMode('deck');
        await new Promise(r => setTimeout(r, 3000));
        out.deckPanelOpen = document.getElementById('deckPanel').classList.contains('open');
        document.getElementById('modeSelect').value = 'explore'; MM.setMode('explore');
        await new Promise(r => setTimeout(r, 900));
        out.afterExplore = document.getElementById('deckPanel').classList.contains('open');
        out.overlays = document.getElementById('plot').data
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


def test_render_makes_one_plotly_call(page):
    """`react` must draw everything, including the selection highlight.

    When `render()` left the highlight to `updateSelectionHighlight()`, every pan and
    filter did an extra add/delete of the whole selection — a full rebuild of a
    15,000-point trace on each one.
    """
    result = page.evaluate("""async () => {
        const rows = []; for (let i = 0; i < 5000; i++) rows.push(i * 5 % 34322);
        await MM.enterBrowse(rows, 'Calls');
        await new Promise(r => setTimeout(r, 900));
        const counts = {add: 0, del: 0, react: 0, restyle: 0};
        const orig = {};
        for (const k of ['addTraces', 'deleteTraces', 'react', 'restyle']) {
            orig[k] = Plotly[k].bind(Plotly);
            Plotly[k] = function (...a) {
                counts[k === 'addTraces' ? 'add' : k === 'deleteTraces' ? 'del' : k]++;
                return orig[k](...a);
            };
        }
        MM.render();
        const perRender = {...counts};
        counts.add = counts.del = counts.react = counts.restyle = 0;
        MM.cycleNext();
        const perCycle = {...counts};
        for (const k of ['addTraces', 'deleteTraces', 'react', 'restyle']) Plotly[k] = orig[k];
        return {perRender, perCycle};
    }""")
    assert page.js_errors == []
    assert result["perRender"] == {"add": 0, "del": 0, "react": 1, "restyle": 0}, (
        f"render() is not a single react: {result['perRender']}")
    assert result["perCycle"]["restyle"] == 1 and result["perCycle"]["add"] == 0, (
        f"an arrow press should be one restyle: {result['perCycle']}")


# ── The Walk (force mode) ───────────────────────────────────────────────


def _walk(page, seed_js, settle=9000):
    return page.evaluate("""async ([seedJs, settle]) => {
        const rows = await (new Function('return (async () => {' + seedJs + '})()'))();
        document.getElementById('modeSelect').value = 'force';
        MM.setMode('force');
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
    const names = new Set(deck.cards.filter(c => !c.is_sideboard).map(c => c.name));
    const rows = []; MM.allData.forEach((d, i) => { if (names.has(d.n)) rows.push(i); });
    return rows;
"""


def test_walk_builds_a_graph_that_spreads(page):
    r = _walk(page, DECK_SEED)
    assert page.js_errors == [], f"the walk threw: {page.js_errors}"
    assert r["canvas"] and r["plotHidden"]
    assert r["nodes"] == r["seeded"] == 97
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
    r = page.evaluate("""async () => {
        const deck = await (await fetch('../data/decks/edgar-vampires/cards.json')).json();
        const names = new Set(deck.cards.filter(c => !c.is_sideboard).map(c => c.name));
        const rows = []; MM.allData.forEach((d, i) => { if (names.has(d.n)) rows.push(i); });
        document.getElementById('modeSelect').value = 'force'; MM.setMode('force');
        await new Promise(r => setTimeout(r, 200));
        await Force.enter(rows, 'Test');
        await new Promise(r => setTimeout(r, 2500));
        const before = Force.nodeCount;
        const steps = [];
        for (const name of ['Edgar Markov', 'Sorin, Imperious Bloodlord', 'Exquisite Blood']) {
            const i = MM.allData.findIndex(d => d.n === name);
            Force.focusCard(i);
            await new Promise(r => setTimeout(r, 700));
            steps.push(Force.nodeCount);
        }
        // A card that is not on the graph must be a no-op, not a crash and not a
        // phantom trail entry. Bloodline Keeper is in Edgar's SIDEBOARD, which is
        // exactly the kind of near-miss that finds this.
        const absent = MM.allData.findIndex(d => d.n === 'Black Lotus');
        Force.focusCard(absent);
        await new Promise(r => setTimeout(r, 300));
        Force.freeze();
        return {before, steps, trail: Force.trailLength, afterAbsent: Force.nodeCount};
    }""")
    assert page.js_errors == []
    assert r["steps"][0] == r["before"] + 6, "a branch should pull in BRANCH_K neighbours"
    assert r["steps"][2] > r["steps"][0], "the graph must keep growing as you walk"
    assert r["trail"] == 3, "each distinct card visited should be recorded on the trail"
    assert r["afterAbsent"] == r["steps"][2], (
        "focusing a card that is not on the graph must change nothing")


def test_leaving_the_walk_restores_the_map(page):
    r = page.evaluate("""async () => {
        const rows = []; for (let i = 0; i < 60; i++) rows.push(i * 37 % 34322);
        document.getElementById('modeSelect').value = 'force'; MM.setMode('force');
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
            plotTraces: gd.data.length,
        };
    }""")
    assert page.js_errors == []
    assert not r["active"] and not r["forceMode"]
    assert r["canvasComputedHidden"], "the walk canvas is still visible over the map"
    assert r["canvasInline"] == "", "visibility must come from the class, not an inline style"
    assert r["plotTraces"] >= 6, "the Plotly map did not come back"


def test_walk_with_nothing_selected_offers_somewhere_to_go(page):
    """An empty graph must never render as a "0 CARDS / 0 LINKS" scoreboard.

    Routed inside `renderPanel` rather than at each call site, so a future caller cannot
    reintroduce the dead end — which is how it got there the first time.
    """
    r = page.evaluate("""async () => {
        document.getElementById('modeSelect').value = 'force';
        MM.setMode('force');
        await new Promise(r => setTimeout(r, 2500));
        const el = document.getElementById('deckInner');
        return {
            decks: el.querySelectorAll('[onclick^="Force.walkDeck"]').length,
            regions: el.querySelectorAll('[onclick^="Force.walkRegion"]').length,
            saysZeroCards: (el.innerText || '').indexOf('0 CARDS') !== -1,
        };
    }""")
    assert page.js_errors == []
    assert r["decks"] == 7, "every published deck should be one click from a walk"
    assert r["regions"] > 0, "regions come from the HDBSCAN membership"
    assert not r["saysZeroCards"], "an empty graph rendered the dead-end scoreboard"


def test_walking_a_deck_from_the_empty_state(page):
    r = page.evaluate("""async () => {
        document.getElementById('modeSelect').value = 'force';
        MM.setMode('force');
        await new Promise(r => setTimeout(r, 2500));
        document.querySelector('[onclick^="Force.walkDeck"]').click();
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

        setMode('force');
        await new Promise(r => setTimeout(r, 2500));
        document.querySelector('[onclick^="Force.walkDeck"]').click();
        await new Promise(r => setTimeout(r, 9000));
        const i = MM.allData.findIndex(d => d.n === 'Past in Flames');
        if (i >= 0) Force.focusCard(i);
        await new Promise(r => setTimeout(r, 2000));
        const before = {nodes: Force.nodeCount, trail: Force.trailLength,
                        w: cv().clientWidth, ink: ink()};

        setMode('explore');
        await new Promise(r => setTimeout(r, 1200));
        const explore = {traces: document.getElementById('plot').data.length,
                         inlineDisplay: cv().style.display || ''};

        setMode('force');
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
        document.getElementById('modeSelect').value = 'force';
        MM.setMode('force');
        await new Promise(r => setTimeout(r, 2500));
        const panel = document.getElementById('deckPanel');
        const btn = document.querySelector('[onclick^="Force.walkDeck"]');
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
    """Restoring the graph on re-entry made the walk a one-way door: the deck menu only
    appears when the graph is empty, so the first set you picked was the only set you
    could ever pick. `New walk` is the way back."""
    r = page.evaluate("""async () => {
        const setMode = m => { document.getElementById('modeSelect').value = m; MM.setMode(m); };
        setMode('force');
        await new Promise(r => setTimeout(r, 2500));
        await Force.walkDeck('goblin-storm', 'GOBLIN STORM');
        await new Promise(r => setTimeout(r, 6000));
        const walking = {nodes: Force.nodeCount,
                         menu: document.querySelectorAll('[onclick^="Force.walkDeck"]').length,
                         hasNewWalk: !!document.querySelector('[onclick^="Force.newWalk"]')};
        document.querySelector('[onclick^="Force.newWalk"]').click();
        await new Promise(r => setTimeout(r, 2500));
        const reset = {nodes: Force.nodeCount,
                       menu: document.querySelectorAll('[onclick^="Force.walkDeck"]').length};
        await Force.walkDeck('heliod', 'HELIOD');
        await new Promise(r => setTimeout(r, 6000));
        return {walking, reset, switched: Force.nodeCount};
    }""")
    assert page.js_errors == []
    assert r["walking"]["nodes"] > 50 and r["walking"]["menu"] == 0
    assert r["walking"]["hasNewWalk"], "no way back to the menu from a walk in progress"
    assert r["reset"]["nodes"] == 0 and r["reset"]["menu"] == 7, "New walk did not restore the menu"
    assert r["switched"] > 50 and r["switched"] != r["walking"]["nodes"], "could not pick a different set"


# ── Navigation: arrows, hover, and the card in The Walk ─────────────────
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
    page.wait_for_timeout(3000)

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
    page.wait_for_timeout(3000)
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
    """`plotly_hover` fires even though every trace sets `hoverinfo: 'none'` — verified in
    a browser before this was built ('none' hides the label, 'skip' kills the event)."""
    r = page.evaluate("""async () => {
        const gd = document.getElementById('plot'), fl = gd._fullLayout;
        const rect = gd.getBoundingClientRect();
        const i = MM.allData.findIndex(d => d.n === 'Sol Ring'), d = MM.allData[i];
        const px = fl.xaxis.d2p(d.x) + fl._size.l, py = fl.yaxis.d2p(d.y) + fl._size.t;
        const drag = gd.querySelector('.nsewdrag') || gd;
        // Record what Plotly says it hovered. Aiming at a card's pixel does NOT guarantee
        // that card: hovermode is 'closest' over 34,322 points, so a denser neighbour a
        // pixel away wins — asking for Sol Ring's coordinates returned Krark-Clan
        // Ironworks. The invariant is that the popup shows whatever was hovered.
        let hoveredRow = null;
        gd.on('plotly_hover', e => {
            if (hoveredRow === null && e.points && e.points[0]) hoveredRow = e.points[0].customdata;
        });
        drag.dispatchEvent(new MouseEvent('mousemove',
            {bubbles: true, clientX: rect.left + px, clientY: rect.top + py}));
        await new Promise(r => setTimeout(r, 700));
        const p = document.querySelector('.card-popup');
        const img = p && p.querySelector('img');
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
    assert r["hoveredName"], "plotly_hover never fired — has hoverinfo been set to 'skip'?"
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
        document.getElementById('modeSelect').value = 'force'; MM.setMode('force');
        await new Promise(r => setTimeout(r, 2500));
        document.querySelector('[onclick^="Force.walkDeck"]').click();
        await new Promise(r => setTimeout(r, 8000));
        const i = MM.allData.findIndex(d => d.n === 'Past in Flames');
        Force.focusCard(i);
        await new Promise(r => setTimeout(r, 2000));
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


def test_the_drill_button_reports_what_it_would_do(page):
    """It used to read only "Drill ⤓". With no filters that meant "re-map all 34,322
    cards", which the cap truncated to an arbitrary 2,000 — a cross-section of the entire
    universe that flew in from everywhere and settled into an incoherent pile."""
    r = page.evaluate("""async () => {
        const btn = document.getElementById('drillFiltered');
        const wide = {label: btn.textContent, disabled: btn.classList.contains('is-disabled')};
        btn.click();
        await new Promise(r => setTimeout(r, 700));
        const refused = {active: Drill.isActive(),
                         status: document.getElementById('status').textContent};

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
    assert "34,322" in r["wide"]["label"], "the button does not state its size"
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


def _ink(page):
    """Percentage of sampled pixels that are not transparent — "did anything draw"."""
    return page.evaluate("""() => {
        const c = document.querySelector('.map-canvas');
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        let lit = 0, n = 0;
        for (let i = 3; i < d.length; i += 4 * 30) { n++; if (d[i] > 10) lit++; }
        return 100 * lit / n;
    }""")


def test_canvas_renderer_draws_the_map_without_plotly(canvas_page):
    r = canvas_page.evaluate("""() => ({
        canvas: !!document.querySelector('.map-canvas'),
        plotlyDrew: !!document.querySelector('#plot .plot-container'),
        cards: MM.allData.length,
    })""")
    assert canvas_page.js_errors == [], f"canvas renderer threw: {canvas_page.js_errors}"
    assert r["canvas"] and not r["plotlyDrew"], "Plotly still drew under ?renderer=canvas"
    assert r["cards"] == 34322
    assert _ink(canvas_page) > 0.5, "the canvas is blank"


def test_canvas_redraws_when_the_filter_changes(canvas_page):
    """setLayers draws synchronously rather than through rAF — a filter is a discrete
    state change, and rAF does not fire in a hidden tab at all."""
    before = _ink(canvas_page)
    canvas_page.evaluate("document.querySelectorAll('#toggles button')[0].click()")
    canvas_page.wait_for_timeout(900)
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


def test_canvas_render_beats_the_plotly_budget(canvas_page):
    """Plotly's render measured ~30 ms on this data. The canvas path must not be slower —
    the quadtree is cached across renders because rebuilding it is 23.5 ms and setLayers
    runs on every filter and keystroke."""
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
