"""Browser behaviour: the workbench (viz/workbench.html).

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


def test_a_dead_deck_is_archived_even_when_it_still_holds_a_paper_lock(
        browser, viz_server):
    """THE BUG THIS WAS WRITTEN FOR, and it was live on the pilot's own bench.

    `locked` was filtered over the WHOLE deck list before `status` was consulted,
    so a deck carrying both landed in "Sleeved — you can play these tonight" and
    could never reach the archive. yawgmoth-swarm sat there for weeks after it
    was broken down for parts: the one screen whose entire job is answering what
    can I play tonight, naming a box of nothing. That is `deck-info` telling the
    pilot to go and play hapatra (common.py:452-462), one surface later.

    `set_lifecycle` now withdraws the lock when a deck is archived, so this state
    should not reach the wire — but `deck_versions.json` is AUTHORED, a hand edit
    can still create it, and a page must not need a command to have been run
    correctly in order to be right about cardboard.

    Prove it by reverting `render`'s filter order: this fails, the rest pass.
    """
    page = _workbench(browser, viz_server, [
        _deck("zombie", locked=True,
              paper={"version": 5, "release": "v2.0.0", "in_sync": True,
                     "versions_behind": 0, "drift": None},
              status=["broken-down", "BROKEN DOWN FOR PARTS", "..."]),
    ])
    try:
        heads = page.eval_on_selector_all(
            ".wb-rack h2", "els => els.map(e => e.textContent.trim().split(' ')[0])")
        assert heads == ["Archive"], heads
        assert page.text_content(".wb-dead").strip() == "BROKEN DOWN FOR PARTS"
        # The contradiction is SHOWN, not dropped. Somebody has to resolve it,
        # and they cannot resolve what the page hid.
        assert "v2.0.0" in page.text_content(".wb-stamp")
    finally:
        page.close()


def test_every_live_branch_is_a_git_style_row_on_the_card(browser, viz_server):
    """P0 (2026-10-20): "link to and highlight feature branches (like git), with
    easy access to recommended card swaps, measured impact, net change report".
    A live branch is a ROW under the chips — name linked to its page, the state
    badge, the objective and its grade, the verdict, the swap count, the headline
    deltas `review()` put on `info.json`, and four links: swaps, measured, net
    change, the deck page's rack. A decided branch (merged, rejected, withdrawn)
    is not shown; it lives in the deck page's fold."""
    page = _workbench(browser, viz_server, [_deck("alpha")], infos={
        "alpha": {"branches": [
            {"name": "mean-v2", "state": "OPEN", "opened": "2026-10-20", "add": 13, "out": 13,
             "summary": {"title": "The meanest bracket-four Edgar", "blurb": "Four denial Vampires, a third drain loop and cheaper casts.", "by": "jarvis", "at": "2026-10-20"},
             "objective": {"axis": "extra_cards_8", "op": ">=", "value": 1.65},
             "objective_grade": {"state": "met", "reading": 1.78},
             "verdict": {"state": "a trade", "rose": 5, "fell": 2, "no_call": 3},
             "impact": {"state": "a trade", "rose": ["damage @T10"], "fell": ["stall, 2 in a row"],
                        "headline": [{"measure": "damage @T10", "delta": 1.94, "ci95_diff": [1.80, 2.08],
                                      "mde": 0.3, "verdict": "better", "unit": "mean", "better_is": "higher"},
                                     {"measure": "killed by T10", "delta": 0.057, "ci95_diff": [0.048, 0.067],
                                      "mde": 0.01, "verdict": "better", "unit": "rate", "better_is": "higher"}]}},
            {"name": "old", "state": "REJECTED"},
            {"name": "done", "state": "MERGED"}]}})
    try:
        rows = page.locator(".wb-card .wb-branch-row")
        assert rows.count() == 1, "one live branch, two decided ones hidden"
        text = rows.first.text_content()
        # IN WORDS (2026-10-20): Jarvis's title leads, the slug sits beside it,
        # the guard and the verdict are sentences, the deltas say "up"/"down".
        for bit in ("The meanest bracket-four Edgar", "mean-v2", "OPEN", "Four denial Vampires"):
            assert bit in text, bit
        # WORDS ONLY (the pilot, 2026-10-20: the figures were "messy, illegible,
        # takes up space"): no guard, no verdict, no swap count, no delta here —
        # the branch page carries them and the links reach it.
        for gone in ("extra_cards_8", "1.65", "a trade", "13 cards", "damage by turn ten", "+1.94", "pts"):
            assert gone not in text, gone
        assert rows.first.locator(".wb-delta").count() == 0
        assert rows.first.locator("a.wb-br-name").get_attribute("href") == "branch.html?deck=alpha&branch=mean-v2"
        # Scoped to the card: the Branches rack above carries the same row.
        hrefs = [a.get_attribute("href") for a in rows.first.locator(".wb-br-links a").all()]
        # The rack's copy of a branch WITHOUT a summary says so rather than inventing one.
        assert page.locator(".wb-br-unsummarised").count() == 0
        # The call leads the links (2026-11-22); the goldfish and the cards follow.
        assert hrefs == ["branch.html?deck=alpha&branch=mean-v2#study",
                         "branch.html?deck=alpha&branch=mean-v2#swaps",
                         "branch.html?deck=alpha&branch=mean-v2#measured",
                         "branch.html?deck=alpha&branch=mean-v2#net-change",
                         "deck.html?deck=alpha#panel-branches"]
        assert rows.first.locator(".wb-br-study").count() == 1
        assert page.locator(".wb-chips a.wb-branch").count() == 0, "the chip is gone; the block replaced it"
    finally:
        page.close()


def test_the_bracket_and_the_grade_and_the_record_are_chips_on_the_card(browser, viz_server):
    """"A bracket badge, and W/L badge" (2026-10-20). The bracket chip is the
    TARGET the deck is built for, floor and DeckCheck's read in its title, amber
    when the floor outgrew the target; the ★ grade comes from rating.json only
    where the manifest's `has.rating` says one exists; the record leads with
    W–L. All read off the artifacts; the page computes nothing."""
    page = _workbench(browser, viz_server,
                      [_deck("alpha", has={"rating": True}), _deck("beta")],
                      infos={"alpha": {"bracket": {"floor": 4, "floor_name": "Optimized", "target": 4, "within_target": True},
                                       "deckcheck": {"score": 46, "max_score": 96, "bracket": 4},
                                       "record": {"games": 6, "win": 0, "loss": 5, "draw": 1, "last_played": "2026-09-29"}},
                             "beta": {"bracket": {"floor": 3, "floor_name": "Upgraded", "target": 2, "within_target": False},
                                      "record": {"games": 3, "win": 2, "loss": 1}}},
                      ratings={"alpha": {"grade": "C", "grade_display": "C", "score": 61.8,
                                         "rank": {"position": 2, "of": 2}, "bracket": {"key": "4"}}})
    try:
        cards = page.locator(".wb-card")
        a = cards.nth(0)
        assert a.locator(".wb-chip.wb-bracket").text_content() == "B4"
        assert "DeckCheck B4 46/96" in a.locator(".wb-chip.wb-bracket").get_attribute("title")
        assert a.locator(".wb-chip.wb-star").text_content() == "★ C"
        assert "2 of 2" in a.locator(".wb-chip.wb-star").get_attribute("title")
        assert a.locator(".wb-chip.wb-record b").text_content() == "0–5–1"
        assert "6 games" in a.locator(".wb-chip.wb-record").text_content()
        b = cards.nth(1)
        assert b.locator(".wb-chip.wb-bracket").text_content() == "B2"
        assert "wb-warn" in b.locator(".wb-chip.wb-bracket").get_attribute("class"), "the floor outgrew the target"
        assert b.locator(".wb-chip.wb-star").count() == 0, "no rating on file, no chip"
        assert b.locator(".wb-chip.wb-record b").text_content() == "2–1"
    finally:
        page.close()


def test_the_fleet_controls_are_absent_without_a_local_server(browser, viz_server):
    """THE PROPERTY THAT MAKES A WRITE LAYER SAFE TO PUT ON A PUBLIC PAGE.

    `manamap serve` has an `/api` the deployed GitHub Pages site does not, so
    Archive and Delete cannot work there. The repo's standing rule for that gap
    (`shell.js:consider`, `deck-view.js:130`) is ABSENT, never
    present-and-broken: a disabled Delete button on a page with no server is a
    promise the page cannot keep, and a live-looking one is worse.

    `viz_server` is a plain static server with no `/api`, which is exactly the
    deployed shape — so this asserts the real condition rather than a stub of it.
    """
    page = _workbench(browser, viz_server, [
        _deck("alpha", deletable=True, undeletable_because=[]),
        _deck("beta", status=["retired", "RETIRED", "..."]),
    ])
    try:
        assert page.eval_on_selector_all("[data-act]", "els => els.length") == 0
        text = page.text_content("#racks")
        for word in ("Archive</button>", "Delete", "Restore"):
            assert word not in page.inner_html("#racks"), word
        # ...and the page is otherwise entirely itself.
        assert "ALPHA" in text and "BETA" in text
    finally:
        page.close()


def test_the_workbench_splits_locked_from_the_bench(browser, viz_server):
    """The split IS the page. A deck that exists only as JSON and a deck you can
    put on a table are different objects, and the old picker — one line of text
    links — could not tell you which was which."""
    page = _workbench(browser, viz_server, [
        _deck("sleeved", locked=True,
              paper={"version": 6, "in_sync": True, "versions_behind": 0, "drift": None}),
        _deck("onbench"),
        _deck("dead", status=["broken-down", "BROKEN DOWN FOR PARTS", "…"]),
    ])
    try:
        heads = page.eval_on_selector_all(
            ".wb-rack h2", "els => els.map(e => e.textContent.trim().split(' ')[0])")
        assert heads == ["Sleeved", "On", "Archive"], heads
        racks = page.eval_on_selector_all(
            ".wb-rack", "els => els.map(e => e.querySelectorAll('.wb-card').length)")
        assert racks == [1, 1, 1], racks
        # INDEXED, not `:first-of-type` — that pseudo-class is per ELEMENT TYPE,
        # and the archive is a <details> while the others are <section>, so it
        # matched the first of each and quietly returned two racks' cards.
        assert page.eval_on_selector_all(
            ".wb-rack", "els => [...els[0].querySelectorAll('.wb-card h3')]"
                        ".map(e => e.textContent)") == ["SLEEVED"]
    finally:
        page.close()


def test_an_unresolvable_lock_is_not_reported_as_in_sync(browser, viz_server):
    """`in_sync` is tri-state — true, false, or null when the lock names a version
    git no longer carries. Null must not read as fine."""
    page = _workbench(browser, viz_server, [
        _deck("ghost", locked=True,
              paper={"version": 9, "in_sync": None, "unresolved": True, "drift": None}),
    ])
    try:
        stamp = page.text_content(".wb-stamp")
        assert "not in git" in stamp, stamp
        # AND NO PIN. The pin asserts this exact 99 is in sleeves and level with
        # the repo; a lock naming a version git cannot find has not established
        # the second half, so it must not wear the glyph that claims both.
        assert page.eval_on_selector_all(".wb-stamp.is-ok", "els => els.length") == 0
        assert page.eval_on_selector_all(".wb-pin", "els => els.length") == 0
    finally:
        page.close()


def test_the_fleet_sort_is_in_the_url_so_it_can_be_sent(browser, viz_server):
    """A view worth reaching twice is worth addressing.

    Same reason `?deck=` and `?mode=` are params on the map. `replaceState`,
    not `pushState` — flipping a sort is not a navigation and must not stack
    up in the back button.
    """
    page = _workbench(browser, viz_server, [_deck("alpha", has={"page": True})])
    try:
        page.click('[data-view="table"]')
        page.click('[data-sort="analysis"]')
        page.wait_for_timeout(150)
        url = page.evaluate("location.search")
        assert "view=table" in url and "sort=analysis" in url, url
        # and it survives a reload, which is what "can be sent" means
        page.reload()
        page.wait_for_selector(".wb-table")
        assert page.eval_on_selector('[data-sort="analysis"]',
                                     "e => e.className").find("is-on") >= 0
    finally:
        page.close()
