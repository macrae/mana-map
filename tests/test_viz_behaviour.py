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

from conftest_viz import BOOT_TIMEOUT_MS, browser, page, viz_server  # noqa: F401

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
