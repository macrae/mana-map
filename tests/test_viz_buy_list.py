"""The buy list for Mana Pool on `branch.html` (Area B4, 2026-10-09).

A REAL RENDER, like `test_the_branch_page_renders_the_decision`: the surface a
purchase is pasted from gets a browser or it gets nothing. Two shapes — with a
fake local API (the button copies the server's text and opens the shop) and
without one (the static <pre> plus a plain Copy).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from conftest_viz import BOOT_TIMEOUT_MS, _record  # noqa: F401

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _a_branch_with_a_bill():
    """The first tracked branch whose net_change.json bill has a BUY row — the
    block only renders when there is something to buy. None skips, never fails:
    a branch is a pilot artifact, not a fixture."""
    for nc in sorted((ROOT / "data" / "decks").glob("*/branches/*/net_change.json")):
        try:
            bill = json.loads(nc.read_text(encoding="utf-8")).get("bill") or {}
        except (OSError, ValueError):
            continue
        buys = [r for r in bill.get("cards") or [] if r.get("state") == "buy"]
        if buys:
            return nc.parent.parent.parent.name, nc.parent.name, len(buys)
    return None


_BILLED = _a_branch_with_a_bill()
requires_bill = pytest.mark.skipif(_BILLED is None, reason="no tracked branch has a BUY row")

_BUY_FIXTURE = {"ok": True, "command": "branch/buy-list",
                "result": {"text": "1 Sol Ring\n1 Rhystic Study", "count": 2,
                           "buy_cents": None, "as_of": None}}
_MANAPOOL = "https://manapool.com/add-deck"

_STUBS = """
window.__opened = null; window.__copied = null;
window.open = function (u) { window.__opened = u; return null; };
Object.defineProperty(navigator, 'clipboard', {
  configurable: true,
  value: { writeText: function (t) { window.__copied = t; return Promise.resolve(); } }
});
"""


def _buy_page(browser, viz_server, slug, branch, api_calls=None):
    """`_branch_page` with the clipboard and `window.open` stubbed, and — when
    `api_calls` is a list — a fake local API: health, the command list, and
    `branch/buy-list` answered from a fixture while its request bodies are
    recorded."""
    page = browser.new_page(viewport={"width": 1440, "height": 1200})
    errors: list[str] = []
    add = _record(errors)
    page.on("pageerror", lambda e: add(e))
    page.on("console",
            lambda m: add(m.text)
            if m.type == "error" and "Failed to load resource" not in m.text
            else None)
    page.add_init_script(_STUBS)
    if api_calls is not None:
        def _json(route, body, status=200):
            route.fulfill(status=status, content_type="application/json",
                          body=json.dumps(body))
        page.route("**/api/page/state", lambda r: _json(r, {"ok": True, "result": {}}))
        page.route("**/api/health",
                   lambda r: _json(r, {"ok": True, "command": "health",
                                       "result": {"ok": True, "api": 1}}))
        page.route("**/api/", lambda r: _json(r, {"commands": ["branch/buy-list", "health"]}))

        def buy(route):
            api_calls.append(route.request.post_data_json)
            _json(route, _BUY_FIXTURE)
        page.route("**/api/branch/buy-list*", buy)
    page.goto(f"{viz_server}/viz/branch.html?deck={slug}&branch={branch}")
    page.wait_for_timeout(2200)
    page.js_errors = errors
    return page


@requires_bill
@pytest.mark.browser
def test_copy_for_mana_pool_copies_the_servers_text_and_opens_the_shop(browser, viz_server):
    """With a local server the button asks `branch/buy-list` for the text, puts
    it on the clipboard and opens Mana Pool — paste only, no URL prefill — and
    the exact-printings box travels with the request."""
    slug, branch, _ = _BILLED
    calls = []
    page = _buy_page(browser, viz_server, slug, branch, api_calls=calls)
    try:
        page.wait_for_selector('#buyList [data-act="buy-copy"]', timeout=BOOT_TIMEOUT_MS)
        assert "paste there, choose exact printings" in page.inner_text("#buyList").lower()
        page.click('#buyList [data-act="buy-copy"]')
        page.wait_for_function("() => window.__opened !== null", timeout=BOOT_TIMEOUT_MS)
        assert page.evaluate("() => window.__copied") == _BUY_FIXTURE["result"]["text"]
        assert page.evaluate("() => window.__opened") == _MANAPOOL
        assert calls and calls[0] == {"slug": slug, "branch": branch, "exact": False}

        page.evaluate("() => { window.__opened = null; }")
        page.check("#buyExact")
        page.click('#buyList [data-act="buy-copy"]')
        page.wait_for_function("() => window.__opened !== null", timeout=BOOT_TIMEOUT_MS)
        assert len(calls) == 2 and calls[1]["exact"] is True
        assert not page.js_errors, page.js_errors
    finally:
        page.close()


@requires_bill
@pytest.mark.browser
def test_the_static_site_shows_the_buy_list_and_copies_it(browser, viz_server):
    """No server: the list is computed from the bill's BUY rows and the branch's
    cards.json, shown in a <pre>, one line per purchase, and a plain Copy puts
    exactly that text on the clipboard. Exact printings re-render it."""
    slug, branch, n_buy = _BILLED
    page = _buy_page(browser, viz_server, slug, branch)
    try:
        page.wait_for_selector("#buyList pre.buylist-text", timeout=BOOT_TIMEOUT_MS)
        plain = page.inner_text("#buyList pre.buylist-text")
        assert len(plain.splitlines()) == n_buy
        page.click('#buyList [data-act="buy-copy"]')
        page.wait_for_function("() => window.__copied !== null", timeout=BOOT_TIMEOUT_MS)
        assert page.evaluate("() => window.__copied") == plain
        assert page.evaluate("() => window.__opened") is None, "no shop tab without a server"
        page.check("#buyExact")
        exact = page.inner_text("#buyList pre.buylist-text")
        assert len(exact.splitlines()) == n_buy
        # The line shape is `check_in.decklist_line`'s, including the foil marker.
        assert page.evaluate(
            "() => Branch.__buyLine({name: 'X', quantity: 2, set: 'slx', "
            "collector_number: '7', foil: true}, true)") == "2 X (SLX) 7 *F*"
        assert page.evaluate(
            "() => Branch.__buyLine({name: 'X', quantity: 1, set: 'slx', "
            "collector_number: '7'}, false)") == "1 X"
        assert not page.js_errors, page.js_errors
    finally:
        page.close()
