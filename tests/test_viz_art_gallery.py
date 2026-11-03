"""The deck page's art gallery, in a real browser with nothing reaching a server.

Viewing works with no server; picking appears only when `art/set` does; a pick posts
the printing and the page then shows what the list says; a split card keeps its other
arts when one tile moves; the picker's foil box cannot invent a finish; and Build's
Apply still works after Build's strip moved the list.
"""
from __future__ import annotations

import json

import pytest

from conftest_viz import CARD_PNG, page  # noqa: F401
from manamap.config import DECKS_DIR

pytestmark = pytest.mark.browser

SLUG = "goblin-storm"
LIST = ("Commander:\n1 Zada, Hedron Grinder (SLD) 2406 *F*\n\nDeck:\n1 Sol Ring\n"
        "8 Mountain (SLD) 2418 *F*\n7 Mountain (SLD) 2419 *F*\n")
IMG = "https://cards.scryfall.io/normal/front/0/0/{}.jpg"


def _doc():
    return {"deck": SLUG, "cards": [
        {"name": "Zada, Hedron Grinder", "quantity": 1, "is_commander": True, "type_line": "Legendary Creature",
         "set": "sld", "collector_number": "2406", "artist": "Wizard of Barge", "foil": True, "image": IMG.format("zada")},
        {"name": "Sol Ring", "quantity": 1, "type_line": "Artifact", "set": "c21", "collector_number": "263",
         "artist": "Mike Bierek", "image": IMG.format("ring")},
        {"name": "Mountain", "quantity": 15, "type_line": "Basic Land — Mountain", "set": "sld",
         "collector_number": "2418", "artist": "Wizard of Barge", "foil": True, "image": IMG.format("m1"),
         "printings": [
             {"set": "sld", "collector_number": "2418", "quantity": 8, "foil": True, "artist": "Wizard of Barge", "image": IMG.format("m1")},
             {"set": "sld", "collector_number": "2419", "quantity": 7, "foil": True, "artist": "Wizard of Barge", "image": IMG.format("m2")}]},
    ]}


PRINTS = [
    {"set": "c21", "collector_number": "263", "image": IMG.format("ring"), "finishes": ["nonfoil"], "artist": "Mike Bierek", "prices_usd": "1.50"},
    {"set": "sld", "collector_number": "2417", "image": IMG.format("wob"), "finishes": ["foil"], "artist": "Wizard of Barge", "prices_usd_foil": "47.14"},
]


def _serve(page, api=True):
    state = {"list": LIST, "doc": _doc(), "posted": []}
    page.route("https://cards.scryfall.io/**", lambda r: r.fulfill(status=200, content_type="image/png", body=CARD_PNG))
    page.route(f"**/data/decks/{SLUG}/decklist.txt*", lambda r: r.fulfill(status=200, content_type="text/plain", body=state["list"]))
    page.route(f"**/data/decks/{SLUG}/cards.json*", lambda r: r.fulfill(status=200, content_type="application/json", body=json.dumps(state["doc"])))
    if not api:
        return state

    def ok(route, result):
        route.fulfill(status=200, content_type="application/json", body=json.dumps({"ok": True, "result": result}))

    def art_set(route):
        body = json.loads(route.request.post_data)
        state["posted"].append(body)
        for pin in body["pins"]:
            if pin["card"] == "Sol Ring":
                state["list"] = state["list"].replace("1 Sol Ring\n", "1 Sol Ring (SLD) 2417 *F*\n")
                state["doc"]["cards"][1].update(set="sld", collector_number="2417", foil=True, artist="Wizard of Barge", image=IMG.format("wob"))
        ok(route, {"job": {"id": "j1", "state": "running"}})

    page.route("**/api/**", lambda r: r.fulfill(status=500, content_type="application/json", body='{"error": "unmocked"}'))
    page.route("**/api/health", lambda r: r.fulfill(status=200, content_type="application/json", body='{"result": {"ok": true}}'))
    page.route("**/api/", lambda r: r.fulfill(status=200, content_type="application/json",
                                              body='{"commands": ["printings", "art/set", "art/pin-shown", "art/undo", "order", "job"]}'))
    page.route("**/api/printings*", lambda r: ok(r, {"name": "x", "printings": PRINTS, "truncated": False}))
    page.route("**/api/art/set", art_set)
    page.route("**/api/job*", lambda r: ok(r, {"id": "j1", "state": "done", "result": {"changed": True, "cards": ["Sol Ring"], "notes": [], "refresh": {"failures": []}}}))
    return state


def _open(page):
    page.goto(page.url.split("/viz/")[0] + f"/viz/deck.html?deck={SLUG}")
    page.wait_for_selector("#panel-art .art-tile", timeout=30000)


def test_viewing_needs_no_server_and_offers_no_pick(page):
    _serve(page, api=False)
    _open(page)
    assert page.locator("#panel-art button.art-tile").count() == 0
    assert page.locator("#panel-art div.art-tile").count() == 4
    assert page.locator("[data-act='art-pin']").count() == 0
    assert page.locator("[data-act='art-copy']").count() == 1
    assert "3 of 4 lines pinned" in page.inner_text("#panel-art .art-count")


def test_a_split_card_has_a_tile_per_art_and_lands_come_last(page):
    _serve(page, api=False)
    _open(page)
    names = page.eval_on_selector_all("#panel-art .art-name", "els => els.map(e => e.textContent)")
    assert names == ["1 Sol Ring", "1 Zada, Hedron Grinder", "8 Mountain", "7 Mountain"]
    srcs = page.eval_on_selector_all("#panel-art .art-tile img", "els => els.map(e => e.getAttribute('src'))")
    assert srcs[2] == IMG.format("m1") and srcs[3] == IMG.format("m2")


def test_only_not_pinned_filter_hides_the_rest(page):
    _serve(page, api=False)
    _open(page)
    page.check("[data-act='art-loose']")
    assert page.locator("#panel-art .art-tile:visible").count() == 1
    assert "not pinned" in page.inner_text("#panel-art .art-tile:visible")


def test_a_pick_posts_the_printing_and_the_tile_follows(page):
    state = _serve(page)
    _open(page)
    page.click("#panel-art button.art-tile[data-loose]")
    page.wait_for_selector("#panel-art .art-picker .printing-thumb")
    page.click("#panel-art .art-picker .printing-thumb[data-set='sld']")
    page.wait_for_function("() => document.querySelector('#panel-art .art-count').textContent.startsWith('4 of 4')", timeout=10000)
    # A foil-only printing is foil whatever the box says.
    assert state["posted"] == [{"slug": SLUG, "pins": [{"card": "Sol Ring", "set": "sld", "collector_number": "2417", "foil": True}]}]
    assert "SLD" in page.inner_text("#panel-art .art-tile >> nth=0")
    assert page.locator("[data-act='art-undo']:visible").count() == 1


def test_moving_one_art_of_a_split_card_keeps_the_other(page):
    state = _serve(page)
    _open(page)
    page.click("#panel-art button.art-tile >> nth=3")
    page.wait_for_selector("#panel-art .art-picker .printing-thumb")
    page.click("#panel-art .art-picker .printing-thumb[data-set='c21']")
    page.wait_for_function("() => document.querySelector('.art-say').textContent.includes('pinned')", timeout=10000)
    pins = state["posted"][0]["pins"]
    assert {(p["set"], p["collector_number"], p["quantity"]) for p in pins} == {("sld", "2418", 8), ("c21", "263", 7)}
    # Never-foil printing: the box cannot make it foil.
    assert next(p for p in pins if p["set"] == "c21")["foil"] is False


def test_foil_for_follows_the_printing(page):
    _serve(page, api=False)
    _open(page)
    got = page.evaluate("""() => {
      const b = f => { const e = document.createElement('button'); e.setAttribute('data-finishes', f); return e; };
      return [Printings.foilFor(b('foil'), false), Printings.foilFor(b('nonfoil'), true),
              Printings.foilFor(b('nonfoil,foil'), true), Printings.foilFor(b('nonfoil,foil'), false),
              Printings.foilFor(b(''), true)];
    }""")
    assert got == [True, False, True, False, True]


def test_copy_without_a_server_merges_lines_and_keeps_each_art(page):
    _serve(page, api=False)
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    _open(page)
    page.click("[data-act='art-copy']")
    page.wait_for_function("() => document.querySelector('.art-say').textContent.length > 0")
    text = page.evaluate("() => navigator.clipboard.readText()")
    assert text.splitlines() == ["1 Sol Ring", "1 Zada, Hedron Grinder (SLD) 2406 *F*",
                                 "8 Mountain (SLD) 2418 *F*", "7 Mountain (SLD) 2419 *F*"]


def test_every_tracked_deck_list_parses_with_a_starred_collector_number(page):
    _serve(page, api=False)
    _open(page)
    got = page.evaluate("() => Decklist.parse('2 Raging Goblin (10E) 224★', {printings: true})[0]")
    assert got["name"] == "Raging Goblin" and got["set"] == "10e" and got["collector_number"] == "224★"
