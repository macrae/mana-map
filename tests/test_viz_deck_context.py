"""The deck page renders the Deck Context (PRD v2 Step 3, 2026-10-07).

Browser tests over the tracked fleet: `viz/js/context-md.js` draws CONTEXT.md as
the dossier's first panel, card links get the page's hover art, and Cards by role
filters by name, colour, type and cost. Interactions go through Playwright
locators (real hit-testing), never `el.click()` in script — the Atlas review grid
shipped unclickable because its test skipped hit-testing.
"""
import json

import pytest

from manamap.config import DECKS_DIR

pytestmark = pytest.mark.browser


def _manifest():
    return json.loads((DECKS_DIR / "index.json").read_text())["decks"]


def _open(browser, viz_server, slug):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{viz_server}/viz/deck.html?deck={slug}")
    page.wait_for_selector("#panels section", timeout=15000)
    return page, errors


def _with_context():
    slugs = [d["slug"] for d in _manifest() if (d.get("has") or {}).get("context")]
    assert slugs, "no deck in the manifest has a Deck Context"
    return slugs


def test_every_deck_page_renders_without_a_script_error(browser, viz_server):
    """One uncaught error and the dossier renders NOTHING. sharknado's page was
    blank from 2026-09-29 to 2026-10-07: `signed()` lived only in branch-view.js
    and its decision ledger was the only one that called it."""
    checked = 0
    for d in _manifest():
        page, errors = _open(browser, viz_server, d["slug"])
        try:
            assert errors == [], f"{d['slug']}: {errors}"
            checked += 1
        finally:
            page.close()
    assert checked >= 8


def test_the_context_is_the_first_panel_with_card_links_and_hover_art(browser, viz_server):
    slug = _with_context()[0]
    page, errors = _open(browser, viz_server, slug)
    try:
        first = page.locator("#panels section").first
        assert first.get_attribute("id") == "panel-context"
        refs = page.locator("#panel-context a.cardref")
        assert refs.count() >= 50
        assert refs.first.get_attribute("href").startswith("index.html?cards=")
        refs.first.hover()
        page.wait_for_timeout(300)
        pop = refs.first.locator("img.card-pop")
        assert pop.get_attribute("src"), "hovering did not promote the preview's data-src"
        assert page.locator("#panel-context [data-section='cards-by-role']").count() == 1
        assert errors == []
    finally:
        page.close()


def test_cards_by_role_filters_by_type_name_colour_and_cost(browser, viz_server):
    page, _ = _open(browser, viz_server, "sharknado")
    try:
        bar = page.locator("#panel-context .ctx-filter")
        count = bar.locator(".ctx-count")
        hidden = "document.querySelectorAll('#panel-context li[hidden], #panel-context p[hidden]').length"
        assert page.evaluate(hidden) == 0

        bar.locator("select[data-f=type]").select_option("Land")
        lands = int(count.inner_text().split()[0])
        assert lands >= 30, f"only {lands} lands matched; the role written as a paragraph was skipped"
        assert page.evaluate(hidden) > 0

        bar.locator("select[data-f=type]").select_option("")
        bar.locator("input[data-f=name]").fill("windfall")
        assert count.inner_text().startswith("1 ")
        visible = page.locator("#panel-context a.cardref:not(.ctx-dim)").evaluate_all(
            "els => els.filter(e => e.closest('[hidden]') === null).map(e => e.dataset.card)")
        assert "Windfall" in visible

        bar.locator("input[data-f=name]").fill("")
        bar.locator("select[data-f=colour]").select_option("B")       # a Jeskai deck
        assert count.inner_text().startswith("0 ")
        bar.locator("select[data-f=colour]").select_option("")
        bar.locator("select[data-f=cost]").select_option("6")
        assert int(count.inner_text().split()[0]) >= 1

        bar.locator("select[data-f=cost]").select_option("")
        assert count.inner_text() == "" and page.evaluate(hidden) == 0
    finally:
        page.close()


def test_the_renderer_escapes_everything_it_does_not_mean(browser, viz_server):
    """Text is escaped before markup is added back; only an http(s) or same-site
    link survives as a link."""
    page, _ = _open(browser, viz_server, "sharknado")
    try:
        html = page.evaluate("""ContextMD.render(
            '## Head <img src=x onerror=alert(1)>\\n\\n' +
            'A <script>alert(1)</script> [bad](javascript:alert(1)) ' +
            '[ok](https://manamap.seanmacrae.com/viz/deck.html?deck=x) **b** _i_ `c`\\n' +
            '- [Sol Ring](https://manamap.seanmacrae.com/viz/index.html?cards=Sol+Ring)', {})""")
        assert "<script" not in html and "<img src=x" not in html and "javascript:" not in html
        assert 'href="deck.html?deck=x"' in html
        assert 'data-card="Sol Ring"' in html
        assert "<strong>b</strong>" in html and "<em>i</em>" in html and "<code>c</code>" in html
    finally:
        page.close()
