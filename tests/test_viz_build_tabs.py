"""Build's side panel as tabs — P1 of the pilot's Atlas review, 2026-10-28.

The panel was one scroll of sixteen sections. It is now a pinned strip (picker, name,
badges, tab bar) over four tabs — Change / Candidates / Deck / Numbers — with the card
as an overlay on the active tab, View and Show in a popover on the canvas, and the
goldfish profile grouped by axis with no single call line. Every test drives the real
page; the manifest is routed where a rung matters, and `/api/**` is the fake from
`test_viz_deck_edit` (a catch-all refuses anything unmocked), so nothing here writes.

Each was proven by putting the old behaviour back (docs/viz.md, "Build mode: the panel
is tabs"): a rung-blind default, `?tab=` ignored, no sessionStorage write, Escape
clearing the line before the card, an off-tab tray change re-rendering the panel, the
`⇒ CALL` line restored, `BranchRows.live` returning every branch, the settings bar
hidden in the map view, and the branch form requiring an objective.
"""

from __future__ import annotations

import json

import pytest

from conftest_viz import BOOT_TIMEOUT_MS
from manamap.config import DECKS_DIR
from test_viz_deck_edit import (ADD, BENCH, CUT, SLEEVED, _card, _open, _until,  # noqa: F401
                                _wait_mode)

pytestmark = pytest.mark.browser


def _tab(page):
    return page.evaluate("""() => ({tab: Build.tab,
        on: [...document.querySelectorAll('#deckInner .bt-tab.is-on')].map(b => b.dataset.btTab)})""")


def _wait_deck(page, slug):
    page.wait_for_function(
        "s => window.Build && Build.deckSlug === s && document.querySelector('#deckInner .bt-tabs')",
        arg=slug, timeout=BOOT_TIMEOUT_MS)


# ── which tab, and remembering it ─────────────────────────────────────────


def test_the_default_tab_is_the_rungs_question(browser, viz_server):
    """Sleeved opens on Change (it is opened to be changed, through a branch); a bench
    deck opens on Deck. Fresh contexts, so no remembered tab interferes."""
    for slug, want in ((SLEEVED, "change"), (BENCH, "deck")):
        page, _ = _open(browser, viz_server, f"?mode=build&deck={slug}")
        try:
            _wait_deck(page, slug)
            got = _tab(page)
            assert got == {"tab": want, "on": [want]}, (slug, got)
            assert page.query_selector(f"#deckInner .bt-body[data-tab='{want}']") is not None
            assert page.js_errors == []
        finally:
            page.close()


def test_a_tab_deep_link_opens_that_tab(browser, viz_server):
    page, _ = _open(browser, viz_server, f"?mode=build&deck={BENCH}&tab=numbers")
    try:
        _wait_deck(page, BENCH)
        assert _tab(page) == {"tab": "numbers", "on": ["numbers"]}
        body = page.locator("#deckInner .bt-body")
        assert body.locator(".lens-stats").count() == 1 and body.locator(".mana-curve").count() == 1
        # Only the active tab is rendered: the role budget lives on Deck.
        assert body.locator(".lens-bar-row").count() == 0
        assert page.js_errors == []
    finally:
        page.close()


def test_the_last_tab_is_remembered_per_deck_and_reported_to_page_state(browser, viz_server):
    page, _ = _open(browser, viz_server, f"?mode=build&deck={BENCH}")
    try:
        _wait_deck(page, BENCH)
        page.click("#deckInner .bt-tab[data-bt-tab='candidates']")
        assert _tab(page)["tab"] == "candidates"
        assert page.evaluate("() => PageState.collect().panel") == "candidates"
        assert page.evaluate(f"() => sessionStorage.getItem('mm.buildTab.{BENCH}')") == "candidates"
        page.reload()
        _wait_deck(page, BENCH)
        assert _tab(page)["tab"] == "candidates", "the tab did not survive a reload"
        # Per deck: another deck opens on its own default, and coming back restores ours.
        page.evaluate(f"() => Build.select('{SLEEVED}')")
        _wait_deck(page, SLEEVED)
        assert _tab(page)["tab"] == "change"
        page.evaluate(f"() => Build.select('{BENCH}')")
        _wait_deck(page, BENCH)
        assert _tab(page)["tab"] == "candidates"
        assert page.js_errors == []
    finally:
        page.close()


# ── the card overlay, and Escape's order ──────────────────────────────────


def _a_deck_with_a_drawable_line():
    manifest = json.loads((DECKS_DIR / "index.json").read_text())
    for d in manifest["decks"]:
        if any(len(v) >= 2 for v in (d.get("stack_cards") or {}).values()):
            return d["slug"]
    return None


def _node(page):
    return page.evaluate("""() => {
        const c = document.querySelector('canvas.force-canvas');
        const r = c.getBoundingClientRect();
        const ns = Force.screenNodes();
        const n = ns.find(n => n.x > 80 && n.y > 80 && n.x < r.width - 80 && n.y < r.height - 80)
                  || ns[0];
        return { x: r.left + n.x, y: r.top + n.y, name: n.name };
    }""")


def test_a_clicked_card_overlays_the_tab_and_escape_peels_it_first(browser, viz_server):
    """A real click on a node opens the card over the Deck tab; Escape closes the card
    and leaves the spotlit line alone; the second Escape clears the line."""
    from conftest_viz import _boot
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck with a drawable verified line")
    page = _boot(browser, viz_server, f"?deck={slug}&tab=deck")
    try:
        page.wait_for_function("() => window.Force && Force.nodeCount > 0", timeout=BOOT_TIMEOUT_MS)
        page.wait_for_timeout(1500)
        assert page.evaluate("() => Build.cardOpen") is False, "the overlay opened on the seed's pin"
        page.click("#deckInner .lens-line:not(.lens-line-nodraw)")
        line = page.evaluate("() => Build.activeLine")
        assert line != -1
        page.evaluate("() => document.getElementById('deckInner').scrollTop = 120")
        before = page.evaluate("() => document.getElementById('deckInner').scrollTop")
        page.wait_for_timeout(1500)       # the spotlight fits; let the nodes settle
        pos = _node(page)
        page.mouse.click(pos["x"], pos["y"])
        page.wait_for_selector("#deckInner #buildCard", timeout=5000)
        r = page.evaluate("""() => ({open: Build.cardOpen, tab: Build.tab,
            lines: document.querySelectorAll('#deckInner .lens-line').length,
            title: (document.querySelector('#buildCard .viewer-header, #buildCard .detail-name, #buildCard h3')
                    || document.querySelector('#buildCard')).textContent})""")
        assert r["open"] and r["tab"] == "deck" and r["lines"] == 0, r
        assert pos["name"] in r["title"], (pos["name"], r["title"][:200])

        page.keyboard.press("Escape")
        page.wait_for_function("() => !document.querySelector('#deckInner #buildCard')", timeout=5000)
        assert page.evaluate("() => Build.activeLine") == line, "Escape cleared the line before the card"
        assert page.evaluate("() => document.getElementById('deckInner').scrollTop") == before
        page.keyboard.press("Escape")
        page.wait_for_function("() => Build.activeLine === -1", timeout=5000)

        # × does the same as Escape; choosing a tab closes the card too.
        page.evaluate("() => Build.openCard(Force.rows()[0])")
        page.click("#deckInner .bt-card-close")
        assert page.evaluate("() => Build.cardOpen") is False
        page.evaluate("() => Build.openCard(Force.rows()[0])")
        page.click("#deckInner .bt-tab[data-bt-tab='numbers']")
        assert page.evaluate("() => Build.cardOpen") is False and _tab(page)["tab"] == "numbers"
        assert page.js_errors == []
    finally:
        page.close()


# ── the Change tab ────────────────────────────────────────────────────────


def test_off_the_change_tab_a_tray_change_repaints_only_the_count(browser, viz_server):
    """The tray grows from the detail panel and from the card overlay while Deck is
    showing: the tab says "Change · N", and neither the tab body nor the overlay is
    re-rendered out from under the click."""
    page, st = _open(browser, viz_server, f"?mode=build&deck={BENCH}", api_slug=BENCH)
    try:
        _wait_mode(page, BENCH, "edit")
        page.wait_for_function("() => DeckEdit.baseSha", timeout=10000)
        assert _tab(page)["tab"] == "deck"
        # The graph repaints the panel while it seeds (Force.renderPanel asks the mode);
        # the probe goes on after that, so only the tray could take it off.
        page.wait_for_function("() => window.Force && Force.nodeCount > 0", timeout=BOOT_TIMEOUT_MS)
        page.wait_for_timeout(1500)
        page.evaluate("() => { document.querySelector('#deckInner .bt-body').__probe = 1; }")
        _card(page, CUT)
        page.click("#detailInner .de-card [data-de-act='cut']")
        page.wait_for_function("() => document.getElementById('btChangeN').textContent === ' · 1'",
                               timeout=5000)
        assert page.evaluate("() => document.querySelector('#deckInner .bt-body').__probe") == 1, (
            "an off-tab tray change re-rendered the panel")
        # A job landing (history, apply, save) paints everything it owns; with no tray on
        # screen that fell back to re-rendering Build's whole panel.
        page.evaluate("() => DeckEdit.__paintAll()")
        assert page.evaluate("() => document.querySelector('#deckInner .bt-body').__probe") == 1, (
            "a landed job re-rendered the panel off the Change tab")

        # The overlay: Add from the card that is open over the tab.
        row = page.evaluate("n => MM.allData.findIndex(d => d.n === n)", ADD)
        page.evaluate("r => Build.openCard(r)", row)
        page.wait_for_selector("#deckInner #buildCard .de-card [data-de-act='add']", timeout=5000)
        page.evaluate("() => { document.getElementById('buildCard').__probe = 2; }")
        page.click("#deckInner #buildCard .de-card [data-de-act='add']")
        page.wait_for_function("() => document.getElementById('btChangeN').textContent === ' · 2'",
                               timeout=5000)
        assert page.evaluate("() => (document.getElementById('buildCard') || {}).__probe") == 2, (
            "the Add on the overlay re-rendered the card away")
        page.evaluate("() => DeckEdit.__paintAll()")
        assert page.evaluate("() => (document.getElementById('buildCard') || {}).__probe") == 2, (
            "a landed job re-rendered the card away")
        assert "+ in the tray" in page.inner_text("#deckInner #buildCard .de-card")

        # The Change tab, when shown, has both rows.
        page.click("#deckInner .bt-tab[data-bt-tab='change']")
        rows = page.inner_text("#deRows")
        assert f"− {CUT}" in rows and f"+ {ADD}" in rows, rows
        assert page.js_errors == []
    finally:
        page.close()


def test_the_goldfish_profile_is_grouped_by_axis_with_no_call_line(browser, viz_server):
    page, st = _open(browser, viz_server, f"?mode=build&deck={BENCH}&tab=change", api_slug=BENCH)
    try:
        _wait_mode(page, BENCH, "edit")
        page.wait_for_function("() => DeckEdit.baseSha", timeout=10000)
        # Before any row: one line on how to add a swap, and no actions or save form.
        assert page.query_selector("#deRows .de-empty") is not None
        assert page.query_selector("#deActions .de-apply") is None
        assert page.query_selector("#deForm .de-save") is None

        _card(page, CUT)
        page.click("#detailInner .de-card [data-de-act='cut']")
        _card(page, ADD)
        page.click("#detailInner .de-card [data-de-act='add']")
        page.wait_for_function(
            "() => document.querySelector('#dePreview .de-preview:not(.is-stale) .de-gf-group')",
            timeout=10000)
        g = page.evaluate("""() => Object.fromEntries([...document.querySelectorAll('#dePreview .de-gf-group')]
            .map(e => [e.dataset.verdict, {n: e.querySelector('.de-gf-head span').textContent,
                       rows: [...e.querySelectorAll('.de-gf-m')].map(td => td.textContent),
                       text: e.textContent}]))""")
        assert list(g) == ["better", "worse", "noise"], g
        assert g["better"]["rows"] == ["kill_by_8"] and g["better"]["n"] == "1", g
        assert "[+0.011, +0.053]" in g["better"]["text"], g
        assert g["worse"]["rows"] == [] and "none" in g["worse"]["text"], g
        assert g["noise"]["rows"] == ["stall"], g
        pv = page.evaluate("() => document.getElementById('dePreview').textContent")
        assert page.query_selector("#dePreview .de-call") is None and "⇒" not in pv, pv
        assert "no blockers" in pv and "every card in and out is seen" in pv, pv
        # Once the tray has rows, the actions are offered.
        assert page.query_selector("#deActions .de-apply") is not None
        assert page.js_errors == []
    finally:
        page.close()


def _info_with_branches(slug):
    info = json.loads((DECKS_DIR / slug / "info.json").read_text())
    info["branches"] = [
        {"name": "live-v1", "state": "OPEN", "summary": {"title": "A live one", "blurb": "Two in."},
         "verdict": {"state": "do not merge", "rose": 2, "fell": 1, "no_call": 7},
         "objective": {"axis": "kill_by_8", "op": ">=", "value": 0.3},
         "objective_grade": {"state": "not met", "reading": 0.2657}},
        {"name": "free-v1", "state": "OPEN", "objective": None, "verdict": None},
        {"name": "old-v1", "state": "MERGED", "verdict": {"rose": 1, "fell": 0, "no_call": 9}},
    ]
    return json.dumps(info)


def test_the_change_tab_lists_the_decks_live_branches(browser, viz_server):
    """Open and proposed only, each linking to its page, with what was measured and
    the guard in its row — and never the report's merge word as the headline."""
    body = _info_with_branches(BENCH)
    page, _ = _open(browser, viz_server, f"?mode=build&deck={BENCH}&tab=change", api_slug=BENCH)
    try:
        page.route(f"**/data/decks/{BENCH}/info.json*", lambda route: route.fulfill(
            status=200, content_type="application/json", body=body))
        page.reload()
        _wait_mode(page, BENCH, "edit")
        page.wait_for_selector("#deBranches .wb-branch-row", timeout=10000)
        rows = page.evaluate("""() => [...document.querySelectorAll('#deBranches .wb-branch-row')].map(r => ({
            href: r.querySelector('.wb-br-name').getAttribute('href'),
            profile: (r.querySelector('.wb-br-profile') || {}).textContent || ''}))""")
        assert [r["href"] for r in rows] == [f"branch.html?deck={BENCH}&branch=live-v1",
                                             f"branch.html?deck={BENCH}&branch=free-v1"], rows
        assert rows[0]["profile"] == ("2 better · 1 worse · 7 no call · "
                                      "guard kill_by_8 >= 0.3: not met (0.266)"), rows
        assert rows[1]["profile"] == "not measured yet", rows
        assert "do not merge" not in page.inner_text("#deBranches")
        # The rows come before the actions and the history, after the profile.
        order = page.evaluate("""() => ['dePreview', 'deBranches', 'deActions', 'deHistory']
            .map(id => document.getElementById(id).getBoundingClientRect().top)""")
        assert order == sorted(order), order
        assert page.js_errors == []
    finally:
        page.close()


def test_a_branch_opens_with_no_guard(browser, viz_server):
    """The objective is optional: the form defaults to "no guard", and Start a branch
    posts no `objective` at all. Picking an axis still sends the guard."""
    page, st = _open(browser, viz_server, f"?mode=build&deck={SLEEVED}", api_slug=SLEEVED)
    try:
        _wait_mode(page, SLEEVED, "branch")
        assert _tab(page)["tab"] == "change"
        # Nothing to act on yet: no Start button, no form.
        assert page.query_selector("#deActions .de-start-branch") is None
        assert page.query_selector("[data-de-field=branch]") is None
        _card(page, "Arcane Denial")
        page.click("#detailInner .de-card [data-de-act='cut']")
        _card(page, ADD)
        page.click("#detailInner .de-card [data-de-act='add']")
        page.wait_for_selector("#deForm [data-de-field=axis]", timeout=10000)
        page.wait_for_function("() => [...document.querySelectorAll('#deForm [data-de-field=axis] option')]"
                               ".some(o => o.value === 'kill_by_8')", timeout=10000)
        sel = page.locator("#deForm [data-de-field=axis]")
        assert sel.input_value() == "" and "no guard" in sel.inner_text()
        form = page.inner_text("#deForm")
        assert "optional" in form and "graded on this one objective" not in form, form
        page.fill("[data-de-field=branch]", "free-v1")
        page.click("#deActions .de-start-branch")
        _until(page, lambda: st["posted"].get("branch/new"))
        posted = st["posted"]["branch/new"][0]
        assert "objective" not in posted and posted["name"] == "free-v1", posted
        assert page.js_errors == []
    finally:
        page.close()


# ── View and Show, on the canvas ──────────────────────────────────────────


def test_view_and_show_live_in_a_popover_on_the_canvas(browser, viz_server):
    page, _ = _open(browser, viz_server, f"?mode=build&deck={SLEEVED}")
    try:
        _wait_deck(page, SLEEVED)
        assert page.evaluate("() => typeof Build.ask") == "undefined", "Ask the bench is still exported"
        assert page.query_selector("#deckInner .lens-toggle") is None, "Show toggles still in the panel"
        assert page.query_selector("#deckInner #bdAsk") is None
        page.wait_for_selector("#buildBar:not([hidden]) .bt-settings-btn", timeout=10000)
        bar, plot = (page.locator(s).bounding_box() for s in ("#buildBar", "#plot"))
        assert bar["y"] - plot["y"] < 40 and plot["x"] + plot["width"] - (bar["x"] + bar["width"]) < 160, (bar, plot)
        page.click("#buildBar .bt-settings-btn")
        page.wait_for_selector("#buildSettings", timeout=3000)
        assert page.locator("#buildSettings .lens-toggle").count() == 4
        page.click("#buildSettings .lens-btn:text-is('Map')")
        page.wait_for_function("() => Build.view === 'map'", timeout=5000)
        # The way back to the graph must survive the graph bar hiding with the graph.
        assert page.evaluate("() => document.getElementById('buildBar').offsetParent !== null")
        page.click("#buildBar .bt-settings-btn") if page.query_selector("#buildSettings") is None else None
        page.click("#buildSettings .lens-btn:text-is('Graph')")
        page.wait_for_function("() => Build.view === 'graph'", timeout=5000)
        # A click outside closes it.
        if page.query_selector("#buildSettings") is None:
            page.click("#buildBar .bt-settings-btn")
        page.click("#deckInner .lens-title")
        page.wait_for_function("() => !document.getElementById('buildSettings')", timeout=3000)
        # Leaving Build takes the bar with it.
        page.evaluate("() => { document.getElementById('modeSelect').value = 'explore'; MM.setMode('explore'); }")
        assert page.evaluate("() => document.getElementById('buildBar').hidden")
        assert page.js_errors == []
    finally:
        page.close()
