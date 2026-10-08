"""The page-state beacon, wired into the pages (PRD v2 Step 7, 2026-10-08).

`viz/js/page-state.js` polls ONE describer per page and POSTs the snapshot to
serve's `page/state`, so Jarvis can resolve "this card" and "this deck". These
tests drive the real pages: a card focused by a REAL mouse click (Playwright's
hit-tested click, never `el.click()` in script) must come back as the snapshot's
`focus`, and the deck page must name its deck.

Two servers, two contracts:
  * the suite's plain static `viz_server` (the deployed shape) — the probe finds
    no `/api`, so NOTHING is POSTed and no console error appears;
  * serve's own `Handler`, started here on a free port with
    `page_state.PATH` monkeypatched to a tmp file — the POST lands, and the real
    `.progress/page/state.json` is never touched.

Proof the Atlas test can fail: with the `PageState.register` call in
`mana-map.js` removed, `collect()` carries no `mode` and no `focus`, and
`test_a_clicked_card_on_the_atlas_is_the_snapshots_focus` and
`test_a_clicked_card_in_build_is_the_focus_and_the_deck_is_named` both fail.
"""
from __future__ import annotations

import functools
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from conftest import ROOT, _free_port
from conftest_viz import BOOT_TIMEOUT_MS, _boot, _record, canvas_page  # noqa: F401

pytestmark = pytest.mark.browser

COLLECT = "() => PageState.collect()"


def _open(browser, base, path):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list[str] = []
    add = _record(errors)
    page.on("pageerror", lambda e: add(e))
    page.on("console", lambda m: add(m.text) if m.type == "error" else None)
    posts: list[str] = []
    page.on("request", lambda r: posts.append(r.url)
            if r.method == "POST" and "/api/page/state" in r.url else None)
    page.js_errors = errors
    page.page_state_posts = posts
    page.goto(f"{base}{path}")
    return page


def test_a_clicked_card_on_the_atlas_is_the_snapshots_focus(canvas_page):
    """Explore: a real click on a point selects it; the snapshot names that card,
    lists it in `selected`, and says the mode."""
    page = canvas_page
    pos = page.evaluate("""() => {
        const host = document.getElementById('plot').getBoundingClientRect();
        const i = MM.allData.findIndex(d => d.n === 'Sol Ring'), d = MM.allData[i];
        const [px, py] = MM.mapRenderer.dataToPixel(d.x, d.y);
        return { x: host.left + px, y: host.top + py };
    }""")
    page.mouse.click(pos["x"], pos["y"])
    # The pick takes the nearest point under the cursor (the field drifts), so the
    # invariant is "the snapshot names WHATEVER the click selected", not Sol Ring.
    page.wait_for_function("() => MM.selectedRows().length > 0", timeout=10_000)
    clicked = page.evaluate("() => MM.allData[MM.selectedRows()[0]].n")
    snap = page.evaluate(COLLECT)
    assert snap["page"] == "atlas"
    assert snap.get("mode") == "explore", snap
    assert snap.get("focus") == clicked, snap
    assert clicked in snap.get("selected", []), snap
    assert page.js_errors == []


def test_a_clicked_card_in_build_is_the_focus_and_the_deck_is_named(browser, viz_server):
    """Build: the deck in the graph is `deck`, and a real click on a node makes
    that card the focus (the force graph routes it through Session's focus)."""
    page = _boot(browser, viz_server, "?deck=sharknado")
    try:
        page.wait_for_function("() => window.Force && Force.nodeCount > 0", timeout=BOOT_TIMEOUT_MS)
        page.wait_for_function("() => window.Build && Build.deckSlug === 'sharknado'",
                               timeout=BOOT_TIMEOUT_MS)
        page.wait_for_timeout(1500)   # the layout settles; a node mid-flight is a miss
        pos = page.evaluate("""() => {
            const c = document.querySelector('canvas.force-canvas');
            const r = c.getBoundingClientRect();
            const ns = Force.screenNodes();
            const n = ns.find(n => n.x > 80 && n.y > 80 && n.x < r.width - 80 && n.y < r.height - 80)
                      || ns[0];
            return { x: r.left + n.x, y: r.top + n.y, name: n.name };
        }""")
        page.mouse.click(pos["x"], pos["y"])
        page.wait_for_function(
            "name => { const s = PageState.collect(); return s.focus === name; }",
            arg=pos["name"], timeout=10_000)
        snap = page.evaluate(COLLECT)
        assert snap.get("mode") == "build", snap
        assert snap.get("deck") == "sharknado", snap
        assert snap.get("view") in ("graph", "map"), snap
        assert page.js_errors == []
    finally:
        page.close()


def test_the_deck_page_names_its_deck_and_posts_nothing_without_serve(browser, viz_server):
    """Plain static (the deployed shape): the snapshot is right, and the beacon
    stays silent — no POST, no console error — because the probe found no API."""
    page = _open(browser, viz_server, "/viz/deck.html?deck=sharknado")
    try:
        page.wait_for_selector("#panels section", timeout=15_000)
        page.wait_for_function("() => window.Api && Api.probed", timeout=10_000)
        snap = page.evaluate(COLLECT)
        assert snap["page"] == "deck"
        assert snap.get("deck") == "sharknado", snap
        page.wait_for_timeout(3500)          # two poll periods
        assert page.page_state_posts == []
        assert page.js_errors == []
    finally:
        page.close()


@pytest.mark.parametrize("path, expect", [
    ("/viz/workbench.html?view=table&sort=played", {"page": "workbench", "view": "table"}),
    ("/viz/branch.html?deck=sharknado&branch=nope", {"page": "branch", "deck": "sharknado",
                                                     "branch": "nope"}),
    ("/viz/library.html", {"page": "library"}),
    ("/viz/spaces.html", {"page": "spaces", "view": "identity"}),
])
def test_every_other_page_registers_a_describer(browser, viz_server, path, expect):
    page = _open(browser, viz_server, path)
    try:
        page.wait_for_load_state("load")
        snap = page.evaluate(COLLECT)
        for k, v in expect.items():
            assert snap.get(k) == v, (k, snap)
        assert page.js_errors == []
    finally:
        page.close()


@pytest.fixture
def serve_server(tmp_path, monkeypatch):
    """serve's own Handler over the repo root, with page/state writing to tmp."""
    from manamap import serve
    from manamap.pilot import page_state

    state = tmp_path / "page" / "state.json"
    monkeypatch.setattr(page_state, "PATH", state)
    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port),
                                functools.partial(serve.Handler, directory=str(ROOT)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}", state
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_under_serve_the_snapshot_lands_in_page_state(browser, serve_server):
    base, state = serve_server
    page = _open(browser, base, "/viz/deck.html?deck=sharknado")
    try:
        page.wait_for_selector("#panels section", timeout=15_000)
        page.wait_for_function("() => PageState.collect().deck === 'sharknado'", timeout=10_000)
        for _ in range(40):
            if state.exists():
                tabs = json.loads(state.read_text())["tabs"]
                if any(t.get("deck") == "sharknado" for t in tabs.values()):
                    break
            page.wait_for_timeout(250)
        assert state.exists(), "nothing was POSTed to page/state"
        rows = list(json.loads(state.read_text())["tabs"].values())
        assert any(r.get("page") == "deck" and r.get("deck") == "sharknado" for r in rows), rows
        assert page.page_state_posts, "the request was never seen"
        assert page.js_errors == []
    finally:
        page.close()
