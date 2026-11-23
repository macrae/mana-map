"""Browser behaviour: Curate (viz/library.html).

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


CURATE_STORE = {
    "v": 2, "corpus": None, "active": "Keep",
    "zones": [
        {"name": "Keep", "cards": ["Sol Ring", "Rhystic Study"]},
        {"name": "Cuts", "cards": ["Command Tower", "Cyclonic Rift"]},
    ],
}


# A store for the SORT AND FILTER tests, whose cards are chosen so that every
# facet under test separates them and no two share a sort key:
#
#   Sol Ring        Artifact     Colorless  MV 1   ramp:rock, utility:activated
#   Lightning Bolt  Instant      R          MV 1   removal:damage
#   Rhystic Study   Enchantment  U          MV 3   draw:engine
#   Llanowar Elves  Creature     G          MV 1   ramp:dork, threat:body, …
#   Command Tower   Land         Colorless  MV 0   land:fixing, …
#   Nicol Bolas     — NOT A REAL CARD, and deliberately so: it is the card the
#                     corpus cannot resolve, which is the bucket that must stay
#                     visible rather than silently vanishing from a filter.
#
# `Order kept` is the insertion order below, which is NOT alphabetical and not
# sorted by any facet — so a sort test that passes is testing the sort.
CURATE_FACETS = {
    "v": 2, "corpus": None, "active": "Keep",
    "zones": [
        {"name": "Keep", "cards": [
            "Sol Ring", "Lightning Bolt", "Rhystic Study",
            "Nicol Bolas, the Card That Is Not"]},
        {"name": "Cuts", "cards": ["Llanowar Elves", "Command Tower"]},
    ],
}


def _curate(browser, viz_server, store=None):
    """The page, loaded against a seeded library."""
    import json as _json

    page = browser.new_page(viewport={"width": 1400, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{viz_server}/viz/library.html")
    page.evaluate(
        "doc => { localStorage.setItem('manamap-library', JSON.stringify(doc));"
        "         location.reload(); }",
        store or CURATE_STORE)
    page.wait_for_function("() => window.Curate && document.querySelector('.cur-zone')",
                           timeout=BOOT_TIMEOUT_MS)
    page.js_errors = errors
    return page


def _store(page):
    return page.evaluate(
        "() => JSON.parse(localStorage.getItem('manamap-library'))")


def test_curate_shows_every_pile_at_once_which_the_drawer_cannot(browser, viz_server):
    """THE REASON THE PAGE EXISTS. `shell.js:350` filters the drawer to the
    active zone — `shown = all.filter(e => e.zone === active)` — so the drawer
    can never show two piles, and reading across piles was the request. Each
    tile carries its own pile's badge, because a mixed grid with no badge is
    four cards from nowhere."""
    page = _curate(browser, viz_server)
    try:
        assert page.evaluate("() => Curate.rail") == "all"
        assert sorted(page.evaluate("() => Curate.shown")) == [
            "Command Tower", "Cyclonic Rift", "Rhystic Study", "Sol Ring"]
        # VISIBLE badges, not badge ELEMENTS. Every tile carries a badge node at
        # all times because the tile survives a re-sort and the node is what
        # gets relabelled; whether it SHOWS is the actual claim, and asserting
        # on `offsetParent` also catches a badge that is present and displayed
        # where an element count would not.
        badges = page.eval_on_selector_all(
            ".cur-badge", "els => els.filter(e => e.offsetParent)"
                          "            .map(e => e.textContent)")
        assert sorted(set(badges)) == ["Cuts", "Keep"], (
            f"the All view must badge each card with its pile; got {badges}")

        # And filtering to one pile really narrows it.
        page.evaluate("() => Curate.setRail('Cuts')")
        assert sorted(page.evaluate("() => Curate.shown")) == [
            "Command Tower", "Cyclonic Rift"]
        assert page.eval_on_selector_all(
            ".cur-badge", "els => els.filter(e => e.offsetParent).length") == 0, (
            "a single-pile view badges nothing — every card is in that pile")
        assert page.js_errors == []
    finally:
        page.close()


def test_a_bulk_move_relocates_every_selected_card_and_persists(browser, viz_server):
    """Driven through the real button and the real select, and asserted against
    `localStorage` rather than the page's own state — the page can believe
    anything; the store is what survives a reload."""
    page = _curate(browser, viz_server)
    try:
        page.evaluate("() => Curate.setRail('Keep')")
        page.click('[data-act="all"]')
        assert page.evaluate("() => Curate.picked.length") == 2
        page.select_option("#cur-move", "Cuts")

        zones = {z["name"]: z["cards"] for z in _store(page)["zones"]}
        assert zones["Keep"] == [], f"Keep still holds {zones['Keep']}"
        assert sorted(zones["Cuts"]) == [
            "Command Tower", "Cyclonic Rift", "Rhystic Study", "Sol Ring"]
        assert page.evaluate("() => Curate.picked.length") == 0, (
            "a completed bulk action leaves nothing selected")
        assert page.js_errors == []
    finally:
        page.close()


def test_a_bulk_remove_drops_them_and_survives_a_reload(browser, viz_server):
    """The point of the page is cutting cards, so the assertion that matters is
    that they are still gone after a reload."""
    page = _curate(browser, viz_server)
    try:
        page.on("dialog", lambda d: d.accept())
        page.evaluate("() => Curate.setRail('Cuts')")
        page.click('[data-act="all"]')
        page.click('[data-act="remove"]')
        page.wait_for_function("() => Curate.picked.length === 0",
                               timeout=BOOT_TIMEOUT_MS)

        names = [n for z in _store(page)["zones"] for n in z["cards"]]
        assert sorted(names) == ["Rhystic Study", "Sol Ring"]
        page.reload()
        page.wait_for_function("() => window.Curate", timeout=BOOT_TIMEOUT_MS)
        assert sorted(page.evaluate("() => Curate.shown")) == [
            "Rhystic Study", "Sol Ring"], "the removal did not survive a reload"
        assert page.js_errors == []
    finally:
        page.close()


def test_the_page_is_usable_before_the_card_index_lands(browser, viz_server):
    """THE FETCH IS AN UPGRADE, NOT A GATE — and this is the assertion that
    keeps it one. With `viz_index.json` failing outright, the grid, the piles,
    the name box and the three factless sorts all still work; only the facet
    controls go, and they say why rather than vanishing.
    """
    import json as _json

    page = browser.new_page(viewport={"width": 1400, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("**/viz_index.json*", lambda route: route.abort())
    page.goto(f"{viz_server}/viz/library.html")
    page.evaluate(
        "doc => { localStorage.setItem('manamap-library', JSON.stringify(doc));"
        "         location.reload(); }",
        CURATE_FACETS)
    try:
        page.wait_for_function("() => window.Curate && document.querySelector('.cur-zone')",
                               timeout=BOOT_TIMEOUT_MS)
        page.wait_for_function("() => Curate.facts === 'failed'", timeout=BOOT_TIMEOUT_MS)

        page.evaluate("() => Curate.setRail('all')")
        assert len(page.evaluate("() => Curate.shown")) == 6
        page.evaluate("() => Curate.setSort('name')")
        shown = page.evaluate("() => Curate.shown")
        assert shown == sorted(shown), shown

        # The sorts that need facts are DISABLED AND PRESENT, not missing: a
        # control that appears when a fetch returns is a control nobody finds.
        opts = page.evaluate("""() => [...document.querySelectorAll('#cur-sort option')]
            .map(o => [o.value, o.disabled])""")
        assert dict(opts)["mv"] is True, opts
        assert dict(opts)["name"] is False, opts

        rail = page.eval_on_selector("#cur-rail", "e => e.innerText")
        assert "unavailable" in rail.lower(), rail
        assert page.query_selector(".cur-chip") is None
        assert errors == []
    finally:
        page.close()
