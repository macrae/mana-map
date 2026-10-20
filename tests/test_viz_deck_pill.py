"""The "In these decks" pill on the Atlas's card panel (2026-10-20).

The pilot: "would be cool to see cards in discover display if they are in any
sleeved/brewing decks." The deck manifest carries every name and face in each
list (`cards`, MANIFEST_VERSION 12); `discovery.js` hands it to `MM.setDecks`
when it loads, and `quickStatsHtml` — drawn everywhere a card is named — appends
one linked pill per deck whose list holds the card, sleeved first, bench and
brewing after, archived and broken-down decks never. The page computes nothing:
membership is an exact `includes` against the manifest's names.
"""
from __future__ import annotations

import json

import pytest

from conftest_viz import page  # noqa: F401  (fixture)
from manamap.config import DATA_DIR
from test_viz_card_panel import _open, _route_scryfall

pytestmark = pytest.mark.browser


def _a_card_on_the_map(page):
    return page.evaluate("() => MM.allData.find(d => d.n && !/\\/\\//.test(d.n)).n")


def test_a_card_in_sleeved_and_brewing_decks_gets_linked_pills_and_an_archived_deck_none(page):
    """Three decks hold the card: one sleeved, one brewing, one broken down
    (stage null). Two pills, sleeved first, each linking its deck page with the
    rung in its title; the archived deck draws nothing; a sleeved deck without
    the card draws nothing. Re-introduce the bug (drop `isWorkableDeck`) and the
    archived deck appears."""
    _route_scryfall(page)
    name = _a_card_on_the_map(page)
    page.evaluate("""name => MM.setDecks([
        {slug: 'brew', deck_name: 'The Brew', stage: 'dev', cards: [name, 'Other']},
        {slug: 'dead', deck_name: 'Dead Deck', stage: null, status: ['broken-down', 'X', 'y'], cards: [name]},
        {slug: 'sleeved', deck_name: 'Sleeved One', stage: 'sleeved', cards: [name]},
        {slug: 'without', deck_name: 'Without', stage: 'sleeved', cards: ['Other']}])""", name)
    _open(page, name)
    r = page.evaluate("""() => {
        const inner = document.getElementById('detailInner');
        const pills = Array.from(inner.querySelectorAll('.viewer-quickstats .deck-pill'));
        return pills.map(p => ({text: p.textContent, href: p.getAttribute('href'), title: p.title})); }""")
    assert [p["text"] for p in r] == ["Sleeved One", "The Brew"], r
    assert r[0]["href"] == "deck.html?deck=sleeved" and r[0]["title"] == "Sleeved"
    assert r[1]["href"] == "deck.html?deck=brew" and r[1]["title"] == "Brewing"
    # No manifest, no pill — absent, not "in no deck".
    page.evaluate("() => MM.setDecks([])")
    _open(page, name)
    assert page.evaluate("() => document.querySelectorAll('#detailInner .deck-pill').length") == 0


def test_the_manifest_is_handed_over_at_boot(page):
    """`discovery.js` loads the manifest in `ready()` and must pass it to
    `MM.setDecks`; without that the pill renders only after something else
    calls it. Read against the tracked manifest: a card in a sleeved deck's
    `cards` draws that deck's pill with no test-side `setDecks`."""
    manifest = json.loads((DATA_DIR / "decks" / "index.json").read_text(encoding="utf-8"))
    sleeved = [d for d in manifest["decks"] if d.get("stage") == "sleeved" and d.get("cards")]
    if not sleeved:
        pytest.skip("no sleeved deck with a card list in the tracked manifest")
    deck = sleeved[0]
    name = page.evaluate("cards => { const have = new Set(MM.allData.map(d => d.n)); return cards.find(c => have.has(c)) || null; }",
                         deck["cards"])
    assert name, "none of the deck's cards is on the map"
    _route_scryfall(page)
    page.wait_for_function("() => MM.deckPillHtml && MM.deckPillHtml(%s) !== ''" % json.dumps(name), timeout=10000)
    _open(page, name)
    hrefs = page.evaluate("() => Array.from(document.querySelectorAll('#detailInner .viewer-quickstats .deck-pill')).map(a => a.getAttribute('href'))")
    assert "deck.html?deck=" + deck["slug"] in hrefs, hrefs
