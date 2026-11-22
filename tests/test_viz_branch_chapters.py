"""The branch page and the workbench, rebuilt for the pilot (2026-11-22).

Measured on sharknado@jaws-v1 before: 18 screens, 15 panels, nothing folded but
the goldfish (which was open), the glossary alone four screens. The page is four
chapters now — the call, the cards, the instruments, the trail — the study and
the proposal plain, everything else behind its title with a count. The workbench
card leads with the deck in one sentence and the latest study, before any chip.
Behaviour only: no pinned string, no pixel figure.
"""
import json

import pytest

from manamap.config import DECKS_DIR
from conftest_viz import idle

pytestmark = pytest.mark.browser


def _manifest():
    return json.loads((DECKS_DIR / "index.json").read_text())["decks"]


def _a_branch():
    """A deck with a branch on file, from the tracked manifest; skip when none."""
    for d in _manifest():
        for b in d.get("branches") or []:
            if (DECKS_DIR / d["slug"] / "branches" / b["name"] / "branch.json").exists():
                return d["slug"], b["name"]
    pytest.skip("no tracked deck has a branch")


def _branch(browser, viz_server, slug, name, hash_=""):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{viz_server}/viz/branch.html?deck={slug}&branch={name}{hash_}")
    page.wait_for_selector("#ch-call:not([hidden]) #study", timeout=20000)
    idle(page)
    return page, errors


def test_the_call_leads_and_the_instruments_follow_the_cards(browser, viz_server):
    slug, name = _a_branch()
    page, errors = _branch(browser, viz_server, slug, name)
    try:
        assert errors == []
        chapters = page.locator(".chapter:not([hidden])").evaluate_all("els => els.map(e => e.id)")
        assert chapters[0] == "ch-call", chapters
        if "ch-instruments" in chapters and "ch-cards" in chapters:
            assert chapters.index("ch-cards") < chapters.index("ch-instruments")
        # The study is the first thing in the call, whatever order the fetches landed in.
        first = page.evaluate("document.querySelector('#ch-call .panels').firstElementChild.id")
        assert first == "study"
        assert page.locator("#ch-call #summary").count() == 1
        # The nav is the chapters, every link landing on one that is drawn.
        links = page.locator("#branchNav a").evaluate_all("els => els.map(a => a.getAttribute('href'))")
        assert links and all(page.locator(h + ":not([hidden])").count() == 1 for h in links), links
    finally:
        page.close()


def test_past_the_call_every_panel_folds_behind_a_title_with_a_count(browser, viz_server):
    slug, name = _a_branch()
    page, errors = _branch(browser, viz_server, slug, name)
    try:
        folded = page.locator("section.panel.fold")
        assert folded.count() >= 4, "the trail and the instruments fold"
        opened = page.evaluate(
            "Array.from(document.querySelectorAll('section.panel.fold > details')).filter(d => d.open).length")
        assert opened == 0, "nothing past the call opens on its own"
        # The goldfish is one instrument among the folds, closed, and says what it saw.
        if page.locator("#measured").count():
            assert page.evaluate("document.querySelector('#measured').open") is False
            assert page.locator("#ch-instruments #measured").count() == 1
        # A fold's line is a count, not a figure: the bill says what is to buy.
        if page.locator("#bill").count():
            note = page.text_content("#bill .fold-note")
            assert "to buy" in note, note
        assert errors == []
    finally:
        page.close()


def test_a_deep_link_opens_its_fold_on_the_branch_page(browser, viz_server):
    slug, name = _a_branch()
    page, errors = _branch(browser, viz_server, slug, name, "#definitions")
    try:
        if page.locator("#definitions").count() == 0:
            pytest.skip("no report on this branch, so no definitions")
        assert page.evaluate("document.querySelector('#definitions > details').open") is True
        assert page.evaluate("document.querySelector('#trail > details').open") is False
        page.evaluate("location.hash = '#trail'")
        page.wait_for_function("document.querySelector('#trail > details').open === true", timeout=5000)
    finally:
        page.close()


def test_the_workbench_card_leads_with_the_deck_in_one_sentence_and_its_latest_call(browser, viz_server):
    """`info.lead` is the Context's first sentence, `info.studies[0]` the latest
    call: the card says what the deck IS and what was last decided before any
    instrument's chip, and the sim/experiment tallies are gone from it."""
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{viz_server}/viz/workbench.html")
    page.wait_for_selector(".wb-card", timeout=20000)
    idle(page)
    try:
        checked = 0
        for d in _manifest():
            info_path = DECKS_DIR / d["slug"] / "info.json"
            if not info_path.exists():
                continue
            info = json.loads(info_path.read_text())
            card = page.locator(f".wb-card:has(a.wb-title[href='deck.html?deck={d['slug']}'])").first
            if not card.count():
                continue
            if info.get("lead"):
                assert card.locator(".wb-lead").text_content().strip() == info["lead"]
                checked += 1
            if info.get("studies"):
                s = info["studies"][0]
                assert s["id"] in card.locator(".wb-study").text_content()
            chips = " ".join(card.locator(".wb-chip").all_text_contents()).lower()
            assert " sim" not in chips and "experiment" not in chips
        assert checked >= 3
        # Every branch row carries the call, and its links lead with it.
        rows = page.locator(".wb-branchlist .wb-branch-row")
        if rows.count():
            assert rows.first.locator(".wb-br-study").count() == 1
            assert rows.first.locator(".wb-br-links a").first.get_attribute("href").endswith("#study")
        assert errors == []
    finally:
        page.close()
