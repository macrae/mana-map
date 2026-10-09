"""Prices as dated evidence on the two pages that quote them (Area B2, 2026-10-09).

The deck page's cover carries one Price line — the total from `prices.json` with its
source and `as_of` — and the Build card panel's deck block carries one price row per
card from the same file. Both are ROUTED IN at the URL the page fetches, like the
combos fixture, so the assertions do not move when `prices --write` re-runs against a
new day's feed; the absent case routes a 404 and asserts there is no line at all,
because a cover that printed a live figure would be the one number with no date on it.
"""
import json

import pytest

from conftest_viz import page  # noqa: F401  (fixture)
from test_viz_card_panel import _open, _route_scryfall
from test_viz_deck_combos import _new_page
from test_viz_review_grid import _mock_api, _open_grid

pytestmark = pytest.mark.browser

PRICES = {
    "slug": "sharknado", "branch": None, "as_of": "2026-10-09", "source": "manapool",
    "currency": "USD", "decklist_sha256": "0" * 64,
    "cards": {
        "Arcane Signet": {"scryfall_id": "a" * 32, "printing": "(C20) 232", "nm_cents": 420,
                          "lp_cents": 380, "foil_cents": 900, "quantity": 1, "note": None,
                          "url": "https://manapool.com/card/c20/232/arcane-signet"},
        "Rhystic Study": {"scryfall_id": "b" * 32, "printing": "(J22) 114", "nm_cents": 6959,
                          "lp_cents": None, "foil_cents": None, "quantity": 1, "note": None,
                          "url": None},
    },
    "total_cents": 420 + 6959, "total_nm_cents": 420 + 6959,
    "missing": ["Island"],
}


def test_the_cover_quotes_the_total_with_its_source_and_date(browser, viz_server):
    page, errors = _new_page(browser, viz_server, "sharknado",
                             [("prices.json", json.dumps(PRICES), 200)])
    try:
        assert errors == []
        book = page.locator("#panel-cover .cov-book")
        assert book.count() == 1
        # `inner_text` is uppercased by the book's CSS; read the DOM text.
        dts = book.locator("dt").evaluate_all("els => els.map(e => e.textContent)")
        assert "Price" in dts, dts
        line = page.locator("#panel-cover .cov-price").inner_text()
        assert line.startswith("≈ $74"), line
        assert "(Manapool, 2026-10-09; 1 unpriced)" in line, line
        # Beside Marks, in the same book.
        assert dts.index("Price") == dts.index("Marks") + 1, dts
    finally:
        page.close()


def test_no_prices_file_means_no_price_line(browser, viz_server):
    page, errors = _new_page(browser, viz_server, "sharknado",
                             [("prices.json", "", 404)])
    try:
        assert errors == []
        assert page.locator("#panel-cover .cov-price").count() == 0
        assert "Price" not in page.locator("#panel-cover .cov-book dt").evaluate_all(
            "els => els.map(e => e.textContent)")
    finally:
        page.close()


def test_the_build_panel_shows_the_card_price_from_the_same_file(page):
    """The deck block's price row: NM, the foil beside it, the source and the date —
    directly after the colour identity row, and absent for an unpriced card."""
    page.route("**/data/decks/sharknado/prices.json*",
               lambda route: route.fulfill(status=200, content_type="application/json",
                                           body=json.dumps(PRICES)))
    _route_scryfall(page)
    _mock_api(page)
    _open_grid(page)
    _open(page, "Arcane Signet")
    ctx = "#detailInner .deck-ctx"
    page.wait_for_selector(ctx, timeout=5000)
    row = page.locator(f"{ctx} .deck-ctx-price")
    assert row.count() == 1
    text = row.inner_text()
    assert "$4.20" in text and "(foil $9.00)" in text, text
    assert "Manapool, 2026-10-09" in text, text
    keys = page.eval_on_selector_all(f"{ctx} .deck-ctx-row .deck-ctx-k",
                                     "els => els.map(e => e.textContent)")
    assert keys.index("price") == keys.index("colour identity") + 1, keys
    assert page.evaluate("() => Build.cardContext(MM.allData.findIndex(d => d.n === "
                         "'Arcane Signet')).price") == {
        "nm_cents": 420, "foil_cents": 900, "source": "manapool", "as_of": "2026-10-09"}
    # A card in the 99 the file does not price: no row, and `price` is null.
    _open(page, "Sol Ring")
    page.wait_for_selector(ctx, timeout=5000)
    assert page.locator(f"{ctx} .deck-ctx-price").count() == 0
    assert page.evaluate("() => Build.cardContext(MM.allData.findIndex(d => d.n === "
                         "'Sol Ring')).price") is None
    assert page.js_errors == []
