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
    BOOT_TIMEOUT_MS, canvas_page, discover_page, page,
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


# ── Phase 3: everything Plotly still owned ──────────────────────────────


def test_canvas_region_labels_are_real_dom(canvas_page):
    """Plotly drew these as layout annotations: a relayout to change one, no transition
    (the crossfade was an rgba() alpha rebuilt on a 150 ms debounce, so they popped), and
    no click target — clicking a region needed a 30-line hit-test against anchors."""
    r = canvas_page.evaluate("""() => {
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
    assert "Multicolor" in r["text"] or "Colorless" in r["text"]


def test_canvas_draws_density_contours(canvas_page):
    """d3-contourDensity replaces histogram2dcontour. Plotly auto-binned to whatever
    extent it was handed, which is why its levels were never comparable between filters."""
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


def test_deck_lens_and_drill_run_on_canvas(canvas_page):
    """Deck Lens dims with a per-point opacity array — the last thing the canvas renderer
    could not draw. Drill pushes 90 frames through updateLayerBy rather than rebuilding
    every layer per frame."""
    r = canvas_page.evaluate("""async () => {
        document.getElementById('modeSelect').value = 'deck'; MM.setMode('deck');
        await new Promise(r => setTimeout(r, 4000));
        await DeckMap.select('edgar-vampires');
        await new Promise(r => setTimeout(r, 2500));
        const lens = {deck: (document.querySelector('.lens-title') || {}).textContent};

        const span = () => { const c = MM.mapRenderer.getCamera();
                             return Math.abs(c.x[1] - c.x[0]); };
        const before = span();
        DeckMap.focusLine(0);
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
        document.getElementById('modeSelect').value = 'force';
        MM.setMode('force');
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
    r = discover_page.evaluate("""async () => {
        document.getElementById('modeSelect').value = 'explore';
        MM.setMode('explore');
        await new Promise(r => setTimeout(r, 2500));
        const gd = document.getElementById('plot');
        return {traces: (gd.data || []).length, rows: MM.allData.length};
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
        const got = new Set(Discovery.tray.names());
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


def test_the_tray_is_its_own_thing(discover_page):
    r = discover_page.evaluate("""async () => {
        const row = Discovery.current;
        Discovery.tray.toggle(row);
        const added = Discovery.tray.has(row);
        Discovery.tray.toggle(row);
        const removed = !Discovery.tray.has(row);
        Discovery.tray.toggle(row);
        Discovery.tray.toggle(Discovery.rowByName('Sol Ring'));
        await new Promise(r => setTimeout(r, 150));
        const two = Discovery.tray.list.length;
        Discovery.tray.clear();
        return {added: added, removed: removed, two: two, cleared: Discovery.tray.list.length};
    }""")
    assert discover_page.js_errors == []
    assert r["added"] and r["removed"]
    assert r["two"] == 2 and r["cleared"] == 0


def test_the_brief_is_the_hand_off_not_a_backend(discover_page):
    """There is no server and this does not add one. The pilot loop is 6-10 serial
    subagent spawns costing ~330k-1.7M tokens; a static page cannot run it. So the tray
    emits a brief for a human to paste into Claude Code, and says so."""
    r = discover_page.evaluate("""async () => {
        const text = await (await fetch('../data/decks/heliod/decklist.txt')).text();
        Discovery.importText(text);
        await new Promise(r => setTimeout(r, 2500));
        return Discovery.brief();
    }""")
    assert discover_page.js_errors == []
    assert r["card_count"] > 50
    assert len(r["cards"]) == r["card_count"]
    assert r["commander_candidates"], "a brief with no commander candidates is not useful"
    assert "Claude Code" in r["next_step"], "the brief must say where it gets run"


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
            await new Promise(r => setTimeout(r, 420));
            const p = document.querySelector('.card-popup');
            const pr = p.getBoundingClientRect();
            out.push({
                at: label, height: Math.round(pr.height),
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
        assert case["height"] > 200, (
            f"the popup measured {case['height']}px at the {case['at']} — an unloaded image "
            f"measuring ~0 is exactly what defeated the clamp"
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
        const trayBefore = Discovery.tray.list.length;
        if (keep) keep.click();
        await new Promise(r => setTimeout(r, 200));

        return {
            landing: landing, clicked: clicked, panel: title(),
            counts: Discovery.counts(Discovery.current),
            trayBefore: trayBefore, trayAfter: Discovery.tray.list.length,
            trayNames: Discovery.tray.names(),
            keptTheRightCard: Discovery.tray.names().includes(clicked),
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
    assert r["trayAfter"] == r["trayBefore"] + 1
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
            tray: Discovery.tray.list.length,
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
        for (const row of Discovery.tray.list.slice(0, 5)) {
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
