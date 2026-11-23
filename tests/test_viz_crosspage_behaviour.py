"""Browser behaviour: contracts across pages.

THE BEHAVIOUR CORE (2026-11-22). Split from `test_viz_behaviour.py` (272 tests) and cut
to what a page must DO: boot without a JS error, read and write the right files,
land deep links, and show exactly what the artifacts hold. Wording, layout, pixels,
timing and JS source text are not tested here — the pages change fast and those
tests broke on every change. A new test asserts a behaviour, never a phrase.
"""
from __future__ import annotations
import base64
import json
import pathlib
import re
import pytest
from conftest import A_BRANCH, requires_branch, requires_deck
from manamap.config import SYNERGY_RULES
from conftest_viz import await_projection  # noqa: F401
from conftest_viz import (  # noqa: F401
    BOOT_TIMEOUT_MS, _record, canvas_page, corpus_count, discover_page, page,
    idle, serve_watchlist_fixture, still_page,
)
pytestmark = pytest.mark.browser


def _dossier(browser, viz_server, slug):
    page = browser.new_page()
    page.goto(f"{viz_server}/viz/deck.html?deck={slug}")
    idle(page)
    return page


def test_no_live_surface_links_into_the_magazine(browser, viz_server):
    """The magazine is not a product any more, so nothing may invite a pilot in.

    These two surfaces used to carry an Archive link each — the dossier's
    `#issueLink` to `../manuals/<slug>.html` and the workbench's to the
    newsstand. The invitation went first; `443cf6b7` then deleted the renderer
    and every page it had written, so those hrefs would now 404 as well as
    mislead. The assertion is unchanged and still the right one — it is about
    what a live surface OFFERS, which is why it kept passing across the deletion
    while six tests that FETCHED those pages went red and nobody noticed.

    `manuals/p/` is excluded deliberately: the Pilot's Operating Handbook lives
    there, it is live, and the dossier links to it on purpose.

    Asserted in a browser rather than by grepping the HTML, because the
    dossier's link was built by `deck-view.js` at render time — a source
    assertion would have passed on a page that still grew one.
    """
    import json
    from manamap.config import DECKS_DIR

    manifest = json.loads((DECKS_DIR / "index.json").read_text())
    slug = manifest["decks"][0]["slug"]

    page = _dossier(browser, viz_server, slug)
    try:
        assert page.query_selector("#issueLink") is None
        hrefs = page.eval_on_selector_all(
            ".head-links a", "els => els.map(e => e.getAttribute('href'))")
        assert not [h for h in hrefs if h and "manuals/" in h], hrefs
    finally:
        page.close()

    page = browser.new_page()
    page.goto(f"{viz_server}/viz/workbench.html")
    idle(page)
    try:
        hrefs = page.eval_on_selector_all(
            "a", "els => els.map(e => e.getAttribute('href'))")
        assert not [h for h in hrefs if h and "manuals/" in h], hrefs
    finally:
        page.close()


def _workbench(browser, viz_server, decks, infos=None, ratings=None):
    page = browser.new_page()
    # A rating is fetched only where the manifest's `has.rating` says one exists
    # (2026-10-20); the route answers the slugs given and 404s the rest.
    page.route("**/data/decks/*/rating.json",
               lambda route: route.fulfill(status=200, content_type="application/json",
                                           body=_json.dumps((ratings or {})[route.request.url.split("/")[-2]]))
               if (ratings or {}).get(route.request.url.split("/")[-2]) is not None
               else route.fulfill(status=404, body=""))
    import json as _json
    page.route("**/data/decks/index.json*", lambda route: route.fulfill(
        status=200, content_type="application/json",
        body=_json.dumps({"decks": decks})))
    page.route("**/data/decks/*/info.json", lambda route: route.fulfill(
        status=200, content_type="application/json",
        body=_json.dumps((infos or {}).get(route.request.url.split("/")[-2], {}))))
    page.goto(f"{viz_server}/viz/workbench.html")
    idle(page)
    return page


def _deck(slug, **kw):
    base = {"slug": slug, "deck_name": slug.upper(), "commander": "Someone",
            "image": None, "status": None, "verified": 0, "decisions": 0,
            "sim_runs": [], "experiments": [], "prescriptions": [],
            "locked": False, "paper": None, "published": True, "has": {}}
    base.update(kw)
    return base


def test_every_card_offers_all_three_destinations_by_name(browser, viz_server):
    """Three named links, and NOTHING implicit.

    The card used to be one big invisible link to the dossier. The rule behind
    that — "a card that opens two different things depending on where you click
    is the interaction bug the atlas already fixed once" — is obeyed here by
    deleting the invisible target rather than adding a second one.

    `deck.html?deck=<slug>` and `index.html?deck=<slug>` are both inbound
    contracts other surfaces already rely on: the dossier deep-links from every
    manual, and the map reads `?deck=` on entry to land in Build with the deck
    loaded rather than on an unfiltered 34,890-card atlas.
    """
    page = _workbench(browser, viz_server, [_deck("alpha", has={"page": True})])
    try:
        hrefs = page.eval_on_selector_all(
            ".wb-links a", "els => els.map(e => e.getAttribute('href'))")
        assert hrefs == ["deck.html?deck=alpha", "index.html?deck=alpha"], hrefs
        # The title is the deck page (PRD v2: the Deck Context leads it; it was the
        # Pilot's Manual until 2026-10-07) — a LABELLED link, not a hidden hit area.
        assert page.get_attribute(".wb-title", "href") == "deck.html?deck=alpha"
        assert page.query_selector_all(".wb-hit") == []
    finally:
        page.close()


def _shell_page(browser, viz_server, path):
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{viz_server}{path}")
    page.wait_for_selector("#shell .shell-nav", timeout=BOOT_TIMEOUT_MS)
    page.js_errors = errors
    return page


def test_the_library_count_crosses_surfaces(browser, viz_server):
    """The count is readable with no corpus BECAUSE the library stores names.

    Had it stored row indices — the form that got the previous attempt deleted —
    a page would have to load the whole index to say "4". That is the quieter
    second argument for names, and this test is where it is asserted.
    """
    page = _shell_page(browser, viz_server, "/viz/workbench.html")
    try:
        # Seed the store and load the page against it, which is what a pilot
        # arriving from the Atlas actually does. Asserted through the rendered
        # strip rather than through whichever object holds the list, so it stays
        # a statement about the product and not about the plumbing.
        page.evaluate("""() => {
            localStorage.setItem('manamap-library', JSON.stringify({
                v: 1, corpus: null,
                cards: ['Sol Ring', 'Rhystic Study', 'Command Tower', 'Cyclonic Rift'],
            }));
            location.reload();
        }""")
        page.wait_for_function("() => document.querySelector('.shell-library')",
                               timeout=20000)
        shown = page.evaluate("() => document.querySelector('.shell-library').innerText")
        assert "4" in shown, f"shell shows {shown!r}"
        assert page.js_errors == []
    finally:
        page.close()
