"""The deck page, rebuilt for the pilot (2026-11-22).

Measured on sharknado before the rebuild: 72 screens, 33 panels, nothing folded,
and the handbook's authored half — the mulligan, the turns, the emergencies, the
table read — not on the page at all (it rendered only under `manuals/p/`, which
PRD v2 unlinked). The page is five chapters now: one screen open, the handbook
next, everything else behind a title with a count. Every test here asserts
behaviour the page owes a reader, never a pinned string or a pixel figure.

The prose and the departed-card fixtures are ROUTED IN at the URLs the page
fetches, as the combos tests do, so the assertions do not move when the agents
rewrite a deck's notes.
"""
import json

import pytest

from test_viz_deck_combos import _new_page
from test_viz_deck_context import _manifest, _open

pytestmark = pytest.mark.browser

SLUG = "sharknado"

PROSE = {
    "decklist_sha256_prefix": "0" * 12,
    "how_it_wins": "Land the shark and feed it cards until it is lethal in the air.\n\n"
                   "WHAT HAS TO BE TRUE. He survives the turn he lands.",
    "mulligan": "Keep two lands and a rock with a play for turn two.\n\n"
                "Ship five lands and a wheel.",
    "threat_assessment": "KILL ON SIGHT. Krenko, before your own momentum play.",
    "matchups": "OLIVER, KRENKO. Answer it first.\n\nTOM, PURPHOROS. Answer it too.",
    "combo_lines": {},
}

PROCEDURES = {
    "decklist_sha256_prefix": "0" * 12,
    "emergency": [
        {"condition": "removal", "condition_text": "a spell is pointed at the shark",
         "indications": ["a seat passes with open mana"], "immediate": ["Swat it for free"],
         "subsequent": ["recast him behind Boots"], "notes": "", "grounded_in": ["log:001"]},
        {"condition": "wipe", "condition_text": "a sweeper is coming",
         "indications": ["five mana open"], "immediate": ["hold the second commander"],
         "subsequent": [], "notes": "", "grounded_in": []},
    ],
    "normal": {
        "preflight": {"keep": ["two lands and a rock"], "ship": ["five lands"], "note": ""},
        "startup": {"steps": ["rock on two"]},
        "assembly": {"steps": ["shark on four", "Boots the same turn"]},
        "cruise": {"steps": ["draw, swing, hold protection"]},
        "closing": {"steps": ["count double strike in the air"]},
    },
    "handling": {"optics": ["you look slow until the counters come"], "reveal": [],
                 "alliances": [], "targets": ["Krenko first"]},
}


def _sleeved_with_prose():
    rows = [d["slug"] for d in _manifest()
            if d.get("paper") and (d.get("has") or {}).get("manual_prose")
            and (d.get("has") or {}).get("poh_procedures")]
    assert rows, "no sleeved deck in the manifest has prose and procedures"
    return rows[0]


def _open_panels(page):
    """Panels a reader sees open: plain ones, and folds whose <details> is open."""
    return page.evaluate("""Array.from(document.querySelectorAll('section[id^=panel-]')).filter(p => {
        const d = p.querySelector(':scope > details');
        return !p.classList.contains('fold') || (d && d.open);
    }).map(p => p.id)""")


def test_the_brief_fits_in_two_screens_and_the_rest_folds(browser, viz_server):
    """One screen says what the deck is and how it wins; the chapter that holds it
    is under two viewports on a sleeved deck with every artifact, and most of
    the page sits behind a title. The Not-yet strip is not content and is not
    counted."""
    slug = _sleeved_with_prose()
    page, errors = _open(browser, viz_server, slug, unfold=False)
    try:
        brief, vh = page.evaluate(
            "[document.getElementById('ch-brief').getBoundingClientRect().height, window.innerHeight]")
        assert brief < 2 * vh, f"the brief is {brief:.0f}px on a {vh}px viewport"
        opened = [p for p in _open_panels(page) if p not in ("panel-coverage", "panel-status")]
        folded = page.locator("section.panel.fold").count()
        assert len(opened) <= 8, opened
        assert folded >= 10, folded
        # The chapters, in reading order, each reachable from the sticky nav.
        chapters = page.locator(".chapter").evaluate_all("els => els.map(e => e.id)")
        assert chapters[:2] == ["ch-brief", "ch-play"] and chapters[-2] == "ch-instruments", chapters
        assert errors == []
    finally:
        page.close()


def test_a_deep_link_opens_its_fold(browser, viz_server):
    """A hash to a folded panel opens it, or the link lands on a title with nothing
    under it. Another fold stays closed."""
    slug = _sleeved_with_prose()
    page = browser.new_page()
    try:
        page.goto(f"{viz_server}/viz/deck.html?deck={slug}#panel-emergencies")
        page.wait_for_selector("#panel-emergencies", timeout=15000)
        assert page.evaluate("document.querySelector('#panel-emergencies > details').open") is True
        assert page.evaluate("document.querySelector('#panel-art > details').open") is False
        # An in-page jump does the same without a reload.
        page.evaluate("location.hash = '#panel-art'")
        page.wait_for_function("document.querySelector('#panel-art > details').open === true", timeout=5000)
    finally:
        page.close()


def test_the_handbook_renders_from_the_prose_files(browser, viz_server):
    """The mulligan, the turns, the emergencies, the table and how it wins come
    from `manual_prose.json` and `poh_procedures.json` — the files the pilot-notes
    and poh-procedures agents write — one fold per phase and per condition, the
    paragraph leads set as headings."""
    page, errors = _new_page(browser, viz_server, SLUG, [
        ("manual_prose.json", json.dumps(PROSE), 200),
        ("poh_procedures.json", json.dumps(PROCEDURES), 200),
    ])
    try:
        assert errors == []
        assert "feed it cards" in page.text_content("#panel-wins")
        assert "Keep two lands" in page.text_content("#panel-mulligan")
        assert "five lands" in page.text_content("#panel-mulligan .mull-cols")
        assert page.locator("#panel-procedures details.case").count() == 4
        assert page.locator("#panel-emergencies details.case").count() == 2
        assert "2 condition" in page.text_content("#panel-emergencies .fold-note")
        assert page.locator("#panel-table-read b.lead").count() >= 3   # the threat lead and two seats
        assert "2 seat" in page.text_content("#panel-table-read .fold-note")
        # The brief carries the first keep rule and points at the mulligan.
        assert "two lands and a rock" in page.text_content("#panel-wins .keep-line")
        # The handbook sits in the Play-it chapter, ahead of the cards.
        ids = page.locator("#panels section").evaluate_all("els => els.map(e => e.id)")
        assert ids.index("panel-mulligan") < ids.index("panel-decklist") < ids.index("panel-goldfish")
    finally:
        page.close()


def test_a_lesson_naming_a_card_that_left_is_marked_where_it_stands(browser, viz_server):
    """The read coaches from older lists on purpose; a lesson that names a card the
    list no longer holds says so inline, with the version it left at — the join of
    `versions.json`'s `out` lists against the `cards.json` names the page holds."""
    sha = "0" * 64
    versions = {"slug": SLUG, "current_version": 2, "working_decklist_sha256": sha, "tags": {},
                "versions": [
                    {"version": 1, "sha": "a" * 12, "first_sha": "a" * 12, "date": "2026-09-01",
                     "first_date": "2026-09-01", "subject": "first", "decklist_sha256": "1" * 64,
                     "decklist_sha256s": ["1" * 64], "size": 100, "in": [], "out": [],
                     "games": 0, "record": {"win": 0, "loss": 0, "draw": 0}, "log_ids": [], "tags": []},
                    {"version": 2, "sha": "b" * 12, "first_sha": "b" * 12, "date": "2026-09-02",
                     "first_date": "2026-09-02", "subject": "cut the wall", "decklist_sha256": sha,
                     "decklist_sha256s": [sha], "size": 100, "in": ["Island"],
                     "out": ["Angelic Wall"], "games": 0,
                     "record": {"win": 0, "loss": 0, "draw": 0}, "log_ids": [], "tags": []}]}
    log = {"slug": SLUG, "nights": {},
           "read_meta": {"as_of_version": 2, "as_of_sha": sha, "games_considered": [],
                         "versions_spanned": [], "unmatched": []},
           "read": {"what_it_does": "x", "how_it_plays": "y", "what_changed": "z",
                    "mindful_of": ["Angelic Wall held the ground while the shark grew (001).",
                                   "Cast the commander on curve (002)."]}}
    page, errors = _new_page(browser, viz_server, SLUG, [
        ("versions.json", json.dumps(versions), 200),
        ("captains_log.json", json.dumps(log), 200),
    ], unfold=False)
    try:
        assert errors == []
        marked = page.locator("#panel-lessons .cut-card")
        assert marked.count() == 1
        assert "Angelic Wall" in marked.text_content()
        assert "V2" in marked.locator(".chip.cut").text_content()
        assert page.locator("#panel-lessons li.lesson").count() == 2
        # The lesson that names only cards still in the list is not marked.
        assert page.locator("#panel-lessons li.lesson").nth(1).locator(".cut-card").count() == 0
    finally:
        page.close()


def test_every_prose_block_says_which_list_it_was_written_for(browser, viz_server):
    """Each handbook block carries deck-info's stamp — written for this list, or
    written for V<n> while the list is V<m> — so a reader can tell coaching about
    the deck in the sleeves from coaching about one that is gone."""
    slug = _sleeved_with_prose()
    page, errors = _open(browser, viz_server, slug)
    try:
        for pid in ("wins", "mulligan", "table-read"):
            chips = page.locator(f"#panel-{pid} .chip.stamp-ok, #panel-{pid} .chip.stale")
            assert chips.count() >= 1, pid
        if page.locator("#panel-lessons li.lesson").count():
            assert page.locator("#panel-lessons .chip.stamp-ok, #panel-lessons .chip.stale").count() >= 1
        assert errors == []
    finally:
        page.close()


def test_no_instrument_figure_sits_on_the_cover(browser, viz_server):
    """The cover's heads are the bracket (the floor and the dated DTI), the record
    on this list and the latest study. The goldfish's rate lives with the goldfish,
    in Instruments, which is the last chapter."""
    slug = _sleeved_with_prose()
    page, errors = _open(browser, viz_server, slug)
    try:
        heads = page.text_content("#panel-cover .cov-nums").lower()
        assert "goldfish" not in heads and "by t6" not in heads, heads
        labels = page.locator("#panel-cover .cov-k").all_text_contents()
        assert any("bracket" in x.lower() for x in labels), labels
        assert page.locator("#ch-instruments #panel-goldfish").count() == 1
        assert errors == []
    finally:
        page.close()
