"""The Atlas's Build-mode REVIEW GRID: its keyboard and its write safety.

An audit on 2026-10-08 (Playwright, 1440x900, sharknado) found the grid writing on
browser shortcuts (Cmd+W posted `verdict: watching` before the tab closed), posting a
double `w` twice, rebuilding itself on every j/k (art blanked and re-queued, open
Oracle text closed), scrolling to the top after every verdict, and going deaf after a
click on the map. Every test here drives REAL keyboard and mouse events.

**Nothing here may reach a real `serve`.** `watch/mark` writes the tracked
`data/decks/sharknado/watchlist.json`; every test installs `_mock_api` before the grid
opens, and its catch-all answers any other `/api/` call itself, so a missed route fails
the test instead of writing the file.
"""

from __future__ import annotations

import json

import pytest

from conftest_viz import page  # noqa: F401  (fixture)

pytestmark = pytest.mark.browser

TILE = "#candGrid .cg-tile"


def _mock_api(page, hold=False):
    """Mock serve. Returns (posted, held): every `watch/mark` body, and — with
    `hold=True` — the routes not yet answered, as (route, body) pairs."""
    state: dict[str, dict] = {}
    posted: list[dict] = []
    held: list = []

    def answer(body):
        card = state.setdefault(body["card"], {"name": body["card"], "verdict": "unreviewed",
                                               "note": None})
        if "verdict" in body:
            card["verdict"] = body["verdict"]
        if "note" in body:
            card["note"] = body["note"]
        return json.dumps({"result": {"slug": body["slug"], "set": body["set"],
                                      "card": dict(card, at="2026-10-08T12:00:00-07:00")}})

    def mark(route):
        body = json.loads(route.request.post_data or "{}")
        posted.append(body)
        if hold:
            held.append((route, answer(body)))
        else:
            route.fulfill(status=200, content_type="application/json", body=answer(body))

    # Registered FIRST so it matches LAST: anything the specific routes below miss is
    # refused here rather than reaching a real server.
    page.route("**/api/**", lambda route: route.fulfill(
        status=500, content_type="application/json", body='{"error": "unmocked in test"}'))
    page.route("**/api/health", lambda route: route.fulfill(
        status=200, content_type="application/json", body='{"result": {"ok": true}}'))
    page.route("**/api/", lambda route: route.fulfill(
        status=200, content_type="application/json", body='{"commands": ["watch/mark"]}'))
    page.route("**/api/watch/mark", mark)
    page.evaluate("() => Api.refresh()")
    return posted, held


def _release(held):
    while held:
        route, body = held.pop(0)
        route.fulfill(status=200, content_type="application/json", body=body)


def _open_grid(page):
    page.evaluate("""() => {
        document.getElementById('modeSelect').value = 'build';
        MM.setMode('build');
    }""")
    page.wait_for_function("() => MM.mode === 'build'", timeout=10000)
    page.evaluate("() => Build.select('sharknado')")
    page.wait_for_function(
        "() => document.querySelectorAll('#candGrid .cg-tile [data-verdict]').length > 0",
        timeout=30000)
    page.wait_for_timeout(400)      # the settle-and-refit after the grid opens


def _click_tile(page, i):
    # The card's reason text: a plain part of the tile, so the click focuses the tile.
    page.locator(TILE).nth(i).locator(".cg-why").click(timeout=3000)


def _focus(page):
    return page.evaluate("""() => {
        const ae = document.activeElement;
        const tiles = Array.from(document.querySelectorAll('#candGrid .cg-tile'));
        return {card: Build.gridCard, tag: ae ? ae.tagName : null,
                active: ae && ae.classList && ae.classList.contains('cg-tile')
                        ? ae.getAttribute('data-card') : null,
                index: tiles.findIndex(t => t.classList.contains('is-focus')),
                progress: (document.querySelector('#candGrid .cg-progress') || {}).textContent};
    }""")


def test_a_modified_key_never_writes(page):
    """Cmd+W / Ctrl+P are the browser's: they closed the tab or opened print AFTER the
    grid had posted a verdict. A plain `w` afterwards is the positive control — it
    proves the keys were reaching the grid at all."""
    posted, _ = _mock_api(page)
    _open_grid(page)
    _click_tile(page, 0)
    assert _focus(page)["active"], "the click did not focus a tile"
    for chord in ("Meta+w", "Control+w", "Meta+p", "Control+p", "Alt+w", "Meta+u"):
        page.keyboard.press(chord)
    page.wait_for_timeout(600)
    assert posted == [], f"a modified key wrote a verdict: {posted}"
    page.keyboard.press("w")
    page.wait_for_function("() => document.querySelector('#candGrid .cg-progress')"
                           ".textContent.startsWith('1 /')", timeout=5000)
    assert len(posted) == 1 and posted[0]["verdict"] == "watching"
    assert page.js_errors == []


def test_a_double_w_posts_once(page):
    """The second `w` arrives while the first is still on the wire: it is dropped."""
    posted, held = _mock_api(page, hold=True)
    _open_grid(page)
    _click_tile(page, 0)
    name = _focus(page)["card"]
    page.keyboard.press("w")
    page.keyboard.press("w")
    page.keyboard.press("p")
    page.wait_for_timeout(600)
    assert len(posted) == 1, f"one card, one pending write — got {posted}"
    _release(held)
    page.wait_for_function(f"() => !Build.__gridCards().includes({json.dumps(name)})",
                           timeout=5000)
    page.wait_for_timeout(300)
    assert len(posted) == 1 and posted[0]["card"] == name
    assert page.js_errors == []


def test_navigation_does_not_rebuild_the_grid(page):
    """j/k/arrows move the highlight; they must not re-create tiles. An open Oracle
    text stays open and every <img> is the SAME element (so nothing is re-fetched)."""
    _mock_api(page)
    _open_grid(page)
    first = page.locator(TILE).nth(0)
    first.locator("details.cg-oracle summary").click(timeout=3000)
    page.evaluate("""() => {
        document.querySelectorAll('#candGrid img.cg-art').forEach((im, i) => { im.__probe = i + 1; });
        document.querySelector('#candGrid .cg-tiles').__probe = 1;
    }""")
    names = page.evaluate("() => Build.__gridCards()")
    page.keyboard.press("j")
    page.keyboard.press("j")
    page.keyboard.press("k")
    f = _focus(page)
    assert f["card"] == names[1] and f["active"] == names[1], f
    cols = page.evaluate("""() => {
        const t = Array.from(document.querySelectorAll('#candGrid .cg-tile'));
        const top = Math.round(t[0].getBoundingClientRect().top);
        return t.filter(x => Math.abs(Math.round(x.getBoundingClientRect().top) - top) <= 2).length;
    }""")
    page.keyboard.press("ArrowDown")
    assert _focus(page)["card"] == names[1 + cols], "ArrowDown did not move one row"
    page.keyboard.press("ArrowUp")
    assert _focus(page)["card"] == names[1]
    if cols > 1:
        page.keyboard.press("h")
        assert _focus(page)["card"] == names[0]
        page.keyboard.press("ArrowRight")
        assert _focus(page)["card"] == names[1]
    after = page.evaluate("""() => {
        const imgs = Array.from(document.querySelectorAll('#candGrid img.cg-art'));
        return {open: document.querySelector('#candGrid .cg-tile details.cg-oracle').open,
                fresh: imgs.filter(im => !im.__probe).length, n: imgs.length,
                tiles: document.querySelector('#candGrid .cg-tiles').__probe === 1};
    }""")
    assert after["open"], "j/k closed the open Oracle text — the grid was re-rendered"
    assert after["tiles"] and after["fresh"] == 0, f"tiles were re-created: {after}"
    assert page.js_errors == []


def test_a_verdict_advances_to_the_next_unreviewed_card_in_view(page):
    """Under Unreviewed: focus moves to the next card, which is on screen, and the
    tiles keep their scroll. Under All: the next UNREVIEWED card, skipping one already
    watched."""
    posted, _ = _mock_api(page)
    _open_grid(page)
    _click_tile(page, 0)
    scroll = 0
    for _ in range(14):
        page.keyboard.press("j")
        scroll = page.evaluate("() => document.querySelector('#candGrid .cg-tiles').scrollTop")
        if scroll > 0:
            break
    assert scroll > 0, "the grid never scrolled — the test cannot see a reset"
    names = page.evaluate("() => Build.__gridCards()")
    i = _focus(page)["index"]
    page.keyboard.press("w")
    page.wait_for_function(f"() => !Build.__gridCards().includes({json.dumps(names[i])})",
                           timeout=5000)
    page.wait_for_timeout(200)
    f = _focus(page)
    assert f["active"] == names[i + 1] == f["card"], f
    view = page.evaluate("""() => {
        const box = document.querySelector('#candGrid .cg-tiles');
        const b = box.getBoundingClientRect(), t = document.activeElement.getBoundingClientRect();
        return {scroll: box.scrollTop, visible: t.top >= b.top - 1 && t.bottom <= b.bottom + 1};
    }""")
    assert view["scroll"] > 0, "the verdict scrolled the tiles back to the top"
    assert view["visible"], "the next card is focused but off screen"

    # All: mark the card just BEFORE the watched one; the next unreviewed skips it.
    page.locator('#candGrid .cg-filter[data-filter="all"]').click()
    order = page.evaluate("() => Build.__gridCards()")
    w = order.index(names[i])
    assert w >= 1
    _click_tile(page, w - 1)
    page.keyboard.press("w")
    page.wait_for_function("() => document.querySelector('#candGrid .cg-progress')"
                           ".textContent.startsWith('2 /')", timeout=5000)
    page.wait_for_timeout(200)
    f = _focus(page)
    assert f["active"] == order[w + 1], f"under All the focus did not skip a reviewed card: {f}"
    assert [p["card"] for p in posted] == [names[i], order[w - 1]]
    assert page.js_errors == []


def test_keys_work_after_a_click_on_the_map(page):
    """One click on the graph leaves focus on <body>; the grid's keys still answer."""
    posted, _ = _mock_api(page)
    _open_grid(page)
    _click_tile(page, 0)
    names = page.evaluate("() => Build.__gridCards()")
    box = page.locator("#plot").bounding_box()
    page.mouse.click(box["x"] + 12, box["y"] + box["height"] - 12)
    ae = page.evaluate("() => document.activeElement.closest('#candGrid') ? 'grid'"
                       " : document.activeElement.tagName")
    assert ae != "grid", "the map click left focus in the grid — the test proves nothing"
    page.keyboard.press("j")
    assert _focus(page)["card"] == names[1], "j did nothing after a click on the map"
    page.keyboard.press("w")
    page.wait_for_function("() => document.querySelector('#candGrid .cg-progress')"
                           ".textContent.startsWith('1 /')", timeout=5000)
    assert [p["card"] for p in posted] == [names[1]]
    assert page.js_errors == []


def test_u_undoes_and_the_header_counts(page):
    posted, _ = _mock_api(page)
    _open_grid(page)
    page.locator('#candGrid .cg-filter[data-filter="all"]').click()
    total = len(page.evaluate("() => Build.__gridCards()"))
    _click_tile(page, 0)
    assert _focus(page)["progress"].startswith(f"0 / {total} reviewed · card 1 of {total}")
    page.keyboard.press("w")
    page.wait_for_function("() => document.querySelector('#candGrid .cg-progress')"
                           ".textContent.startsWith('1 /')", timeout=5000)
    assert _focus(page)["index"] == 1
    page.keyboard.press("k")
    page.keyboard.press("u")
    page.wait_for_function("() => document.querySelector('#candGrid .cg-progress')"
                           ".textContent.startsWith('0 /')", timeout=5000)
    f = _focus(page)
    assert f["index"] == 0, "undo moved the focus off the card it corrected"
    assert [p.get("verdict") for p in posted] == ["watching", "unreviewed"]
    assert page.evaluate(f"() => document.querySelectorAll('{TILE}')[0].classList"
                         ".contains('v-unreviewed')")
    page.keyboard.press("?")
    assert page.evaluate("() => !document.querySelector('#candGrid .cg-keys').hidden")
    assert page.js_errors == []


def test_a_note_and_a_click_elsewhere_are_written_one_after_the_other(page):
    """The note saves on blur; the click that caused the blur marks another card. Both
    must arrive — in order, never concurrently — and the note text must survive."""
    posted, held = _mock_api(page, hold=True)
    _open_grid(page)
    names = page.evaluate("() => Build.__gridCards()")
    _click_tile(page, 0)
    page.keyboard.press("n")
    page.wait_for_function("() => document.activeElement.tagName === 'TEXTAREA'", timeout=3000)
    page.keyboard.type("worth a look")
    page.keyboard.press("w")        # typed into the note, never a verdict
    page.locator(TILE).nth(1).locator('[data-verdict="watching"]').click(timeout=3000)
    page.wait_for_timeout(500)
    assert len(posted) == 1 and posted[0]["note"] == "worth a lookw", posted
    _release(held)
    for _ in range(40):
        if len(posted) >= 2:
            break
        page.wait_for_timeout(100)
    _release(held)
    assert [(p["card"], p.get("verdict")) for p in posted] == [(names[0], None),
                                                               (names[1], "watching")]
    page.wait_for_function(f"() => !Build.__gridCards().includes({json.dumps(names[1])})",
                           timeout=5000)
    note = page.evaluate("""n => {
        const t = Array.from(document.querySelectorAll('#candGrid .cg-tile'))
            .find(x => x.getAttribute('data-card') === n);
        const p = t && t.querySelector('.cg-note');
        return p ? p.textContent : null;
    }""", names[0])
    assert note == "worth a lookw"
    assert page.js_errors == []
