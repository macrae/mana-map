"""Choosing a printing from the page, and drawing the one the pilot sleeves.

Three claims, each driven in a real browser with nothing reaching a server or
Scryfall: the deck block's printing row opens a strip of thumbnails from
`printings` and a click POSTs `printing/set` with the chosen set, collector number
and foil, after which the row reads the reloaded `cards.json`; `Shell.cardImageUrl`
returns a `cards.json` entry's own image (its face for a DFC, `art_crop` when
asked) and falls back to the by-name URL without one; and the deck page's hover
art for a card in the deck is the `cards.json` image rather than Scryfall's
default for the name.

`_mock_api` is written here rather than borrowed from the review grid: that one
advertises only `watch/mark`, and `Api.has('printing/set')` is what gates the
change button.
"""

from __future__ import annotations

import json

import pytest

from conftest_viz import (CARD_PNG, page, serve_card_images_locally,  # noqa: F401
                          serve_scryfall_prints_locally, serve_watchlist_fixture)
from manamap.config import DECKS_DIR

pytestmark = pytest.mark.browser

SLUG = "sharknado"
NAME = "Arcane Signet"


def _cards():
    return json.loads((DECKS_DIR / SLUG / "cards.json").read_text(encoding="utf-8"))


def _mock_api(page, cards_doc):
    """Mock serve: `printing/set` records every body and the next `cards.json` the
    page fetches carries the printing it chose, so the reload is observable."""
    posted: list[dict] = []
    state = {"doc": cards_doc}

    def set_printing(route):
        body = json.loads(route.request.post_data or "{}")
        posted.append(body)
        doc = json.loads(json.dumps(state["doc"]))
        for c in doc["cards"]:
            if c["name"] == body["card"]:
                c["set"] = body["set"]
                c["collector_number"] = body["collector_number"]
                c["foil"] = bool(body.get("foil"))
                c["set_name"] = "Secret Lair Drop" if body["set"] == "sld" else c.get("set_name")
                c["image"] = "https://cards.scryfall.io/normal/front/5/1/51d00000-0000-4000-8000-000000001234.jpg"
        state["doc"] = doc
        route.fulfill(status=200, content_type="application/json", body=json.dumps({
            "ok": True, "command": "printing/set",
            "result": {"changed": True, "ran": ["fetch-deck", "goldfish", "mana-analysis"],
                       "line": f"1 {body['card']} ({body['set'].upper()}) {body['collector_number']}"
                               + (" *F*" if body.get("foil") else ""),
                       "slug": body["slug"], "card": body["card"]}}))

    page.route("**/api/**", lambda route: route.fulfill(
        status=500, content_type="application/json", body='{"error": "unmocked in test"}'))
    page.route("**/api/health", lambda route: route.fulfill(
        status=200, content_type="application/json", body='{"result": {"ok": true}}'))
    page.route("**/api/", lambda route: route.fulfill(
        status=200, content_type="application/json",
        body='{"commands": ["watch/mark", "printings", "printing/set"]}'))
    page.route("**/api/printing/set", set_printing)
    page.route(f"**/data/decks/{SLUG}/cards.json*", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(state["doc"])))
    page.evaluate("() => Api.refresh()")
    page.wait_for_function("() => Api.probed && Api.has('printing/set')", timeout=5000)
    return posted


def _open_build(page):
    serve_watchlist_fixture(page)
    page.evaluate("""() => { document.getElementById('modeSelect').value = 'build';
                             MM.setMode('build'); }""")
    page.wait_for_function("() => MM.mode === 'build'", timeout=10000)
    page.evaluate(f"() => Build.select('{SLUG}')")
    page.wait_for_function("() => Build.deckSlug === '%s' && Build.deckCard('%s')" % (SLUG, NAME),
                           timeout=30000)


def _open_card(page, name):
    page.evaluate("n => { MM.closeDetail(); MM.selectByName(n); }", name)
    page.wait_for_selector("#detailInner .deck-ctx .deck-ctx-printing", timeout=10000)


def test_the_strip_renders_a_click_posts_and_the_row_follows_the_reload(page):
    cards = _cards()
    before = next(c for c in cards["cards"] if c["name"] == NAME)
    serve_card_images_locally(page)
    page.route("https://cards.scryfall.io/**", lambda route: route.fulfill(
        status=200, content_type="image/png", body=CARD_PNG))
    # AFTER `_mock_api`: Playwright tries the most recently registered route first,
    # so the catch-all `**/api/**` refusal must be older than the printings answer.
    posted = _mock_api(page, cards)
    projected = serve_scryfall_prints_locally(page)
    _open_build(page)
    _open_card(page, NAME)
    row = "#detailInner .deck-ctx .deck-ctx-printing"
    text = page.inner_text(row)
    assert f"({before['set'].upper()}) {before['collector_number']}" in text, text
    assert before["set_name"] in text
    # The panel's image is the sleeved printing's, not Scryfall's default by name —
    # at the panel's `large` version (2026-10-10), the same CDN path one size up.
    src = page.get_attribute("#detailInner .detail-card-image img", "src")
    assert src == before["image"].replace("/normal/", "/large/"), src

    page.click(f"{row} .deck-ctx-change")
    page.wait_for_selector(f"{row} .printing-thumb", timeout=5000)
    thumbs = page.eval_on_selector_all(
        f"{row} .printing-thumb",
        "els => els.map(e => ({set: e.dataset.set, cn: e.dataset.cn, cur: e.classList.contains('is-current'),"
        " img: (e.querySelector('img') || {}).getAttribute ? e.querySelector('img').getAttribute('src') : null}))")
    assert [(t["set"], t["cn"]) for t in thumbs] == [(p["set"], p["collector_number"]) for p in projected]
    assert all(t["img"] == p["image"] for t, p in zip(thumbs, projected))
    assert not any(t["cur"] for t in thumbs), "none of the fixture's printings is the sleeved one"
    assert page.is_checked(f"{row} .printing-foil input") is bool(before.get("foil"))

    page.check(f"{row} .printing-foil input")
    page.click(f"{row} .printing-thumb[data-set='sld']")
    page.wait_for_function(
        "() => (document.querySelector('#detailInner .deck-ctx-print') || {}).textContent"
        ".indexOf('(SLD) 1234') === 0", timeout=10000)
    assert posted == [{"slug": SLUG, "card": NAME, "set": "sld", "collector_number": "1234",
                       "foil": True}], posted
    after = page.inner_text(row)
    assert "(SLD) 1234" in after and "Secret Lair Drop" in after and "foil" in after, after
    # The reloaded row is what Build now holds, and the status names the line written.
    assert page.evaluate(f"() => Build.deckCard('{NAME}').set") == "sld"
    assert "(SLD) 1234 *F*" in page.inner_text("#status")
    assert page.js_errors == []


def test_a_card_not_in_the_deck_has_no_printing_row(page):
    serve_card_images_locally(page)
    _mock_api(page, _cards())
    _open_build(page)
    page.evaluate("n => { MM.closeDetail(); MM.selectByName(n); }", "Lightning Bolt")
    page.wait_for_selector("#detailInner .deck-ctx", timeout=10000)
    assert page.query_selector("#detailInner .deck-ctx-printing") is None
    assert page.js_errors == []


def test_card_image_url_prefers_the_entry_and_falls_back_by_name(page):
    r = page.evaluate("""() => {
      const dfc = {name: 'A // B', image: 'https://cards.scryfall.io/normal/front/x.jpg',
                   art_crop: 'https://cards.scryfall.io/art_crop/front/x.jpg',
                   card_faces: [{name: 'A', image: 'https://cards.scryfall.io/normal/front/x.jpg'},
                                {name: 'B', image: 'https://cards.scryfall.io/normal/back/x.jpg'}]};
      const split = {name: 'Fire // Ice', image: 'https://cards.scryfall.io/normal/front/s.jpg',
                     card_faces: [{name: 'Fire', image: null}, {name: 'Ice', image: null}]};
      return {
        plain: Shell.cardImageUrl('A // B', 'normal', dfc),
        art: Shell.cardImageUrl('A // B', 'art_crop', dfc),
        back: Shell.cardImageUrl('B', 'normal', dfc),
        splitFace: Shell.cardImageUrl('Ice', 'normal', split),
        bare: Shell.cardImageUrl('Sol Ring', 'small'),
        empty: Shell.cardImageUrl('Sol Ring', 'normal', {name: 'Sol Ring'}),
        viaMM: MM.cardImageUrl('A // B', 'normal', dfc),
      };
    }""")
    assert r["plain"] == "https://cards.scryfall.io/normal/front/x.jpg"
    assert r["art"] == "https://cards.scryfall.io/art_crop/front/x.jpg"
    assert r["back"] == "https://cards.scryfall.io/normal/back/x.jpg", "a face by its name"
    assert r["splitFace"] == "https://cards.scryfall.io/normal/front/s.jpg", "a split card's one image"
    assert r["bare"] == "https://api.scryfall.com/cards/named?exact=Sol%20Ring&format=image&version=small"
    assert r["empty"] == "https://api.scryfall.com/cards/named?exact=Sol%20Ring&format=image&version=normal", (
        "an entry with no image falls back by name, never to nothing")
    assert r["viaMM"] == r["plain"]
    assert page.js_errors == []


def test_a_larger_version_upsizes_the_sleeved_printing(page):
    """The card panel asks for `large` and the magnifier for `png` (2026-10-10). The
    entry holds Scryfall's `normal`; the version is a PATH segment on Scryfall's
    CDN, so the sleeved printing must upsize in place — not fall back to the
    by-name default printing, and not stay at `normal` under a `large` label."""
    r = page.evaluate("""() => {
      const dfc = {name: 'A // B', image: 'https://cards.scryfall.io/normal/front/x.jpg?1700',
                   card_faces: [{name: 'A', image: 'https://cards.scryfall.io/normal/front/x.jpg?1700'},
                                {name: 'B', image: 'https://cards.scryfall.io/normal/back/x.jpg?1700'}]};
      const odd = {name: 'C', image: 'https://example.org/c.jpg'};
      return {
        large: Shell.cardImageUrl('A // B', 'large', dfc),
        back: Shell.cardImageUrl('B', 'large', dfc),
        png: Shell.cardImageUrl('A // B', 'png', dfc),
        normal: Shell.cardImageUrl('A // B', 'normal', dfc),
        odd: Shell.cardImageUrl('C', 'large', odd),
        byName: Shell.cardImageUrl('Sol Ring', 'large'),
      };
    }""")
    assert r["large"] == "https://cards.scryfall.io/large/front/x.jpg?1700"
    assert r["back"] == "https://cards.scryfall.io/large/back/x.jpg?1700", "the back face upsizes too"
    assert r["png"] == "https://cards.scryfall.io/png/front/x.png?1700"
    assert r["normal"] == "https://cards.scryfall.io/normal/front/x.jpg?1700", "normal is untouched"
    assert r["odd"] == "https://example.org/c.jpg", "a non-Scryfall URL is never rewritten"
    assert r["byName"].endswith("&version=large")
    assert page.js_errors == []


def test_the_deck_page_draws_the_cards_json_image(browser, viz_server):
    """The dossier's hover art for a card in the deck comes from `cards.json`
    (`cards.scryfall.io/...`), which is the sleeved printing, not from
    `cards/named?exact=`, which is Scryfall's default for the name."""
    cards = _cards()
    by_name = {c["name"]: c for c in cards["cards"]}
    pg = browser.new_page()
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    serve_card_images_locally(pg)
    pg.route("https://cards.scryfall.io/**", lambda route: route.fulfill(
        status=200, content_type="image/png", body=CARD_PNG))
    try:
        pg.goto(f"{viz_server}/viz/deck.html?deck={SLUG}")
        # `attached`, not visible: a pop is `display:none` until its link is hovered.
        pg.wait_for_selector("#panel-context a.cardref img.card-pop", state="attached",
                             timeout=15000)
        pops = pg.eval_on_selector_all(
            "#panel-context a.cardref",
            "els => els.map(a => ({name: a.getAttribute('data-card') || a.textContent.trim(),"
            " src: (a.querySelector('img.card-pop') || {}).getAttribute"
            " ? a.querySelector('img.card-pop').getAttribute('data-src') : null}))")
        in_deck = [p for p in pops if p["name"] in by_name and p["src"]]
        assert len(in_deck) >= 20, f"too few deck cards with hover art: {len(in_deck)}"
        wrong = [p for p in in_deck if p["src"] != by_name[p["name"]]["image"]]
        assert wrong == [], wrong[:3]
        assert not any("cards/named" in p["src"] for p in in_deck)
        assert errors == []
    finally:
        pg.close()
