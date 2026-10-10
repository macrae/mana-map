"""The pages know the format (Area C7, 2026-10-09): a 60-card deck is not a Commander deck.

The fixture is the bench's first real constructed deck, `data/decks/elves/` — Modern
mono-green Elves, no commander, no sideboard. Every figure the assertions compare
against is read from its tracked files at test time (the main-deck copy count, the
audit's `not_measured` axes, the manifest's `image`), never hard-coded, so a re-run of
the chain on a changed list moves the expectation with it.

What is asserted, surface by surface:

- **Build**: the header stat is `N / 60+` (count AND rule — a minimum), no node wears
  the commander ring, the "grey out what you cannot play" lens dims by the MODERN
  legality column and not by colour (a red card is legal in a green Modern deck), a
  card in the list says "In the 60" and its facts line leads "Modern: legal", "Set as
  commander" is not offered, and Session refuses a commander outright.
- **Sideboard**: routed in (elves has none) — listed as "Sideboard (N)" and kept out of
  the copy count and off the graph.
- **Deck page**: "The 60", the n/a line (never the `absent()` call to action) for the
  goldfish, the vitals, the bracket and the table, and the audit's not-measured axes.
- **Workbench**: the elves card reads `Modern · G · 60 cards` under its own art.
- **Discover**: a pasted 60-card list with no commander leaves the format alone and
  shows the hint; picking Modern exports a brief with `format` and no `commander`.
"""

from __future__ import annotations

import json

import pytest

from conftest_viz import page, discover_page  # noqa: F401  (fixtures)
from manamap.config import DECKS_DIR

pytestmark = pytest.mark.browser

SLUG = "elves"


def _cards():
    return json.loads((DECKS_DIR / SLUG / "cards.json").read_text(encoding="utf-8"))


def _main_copies():
    return sum(c.get("quantity", 1) for c in _cards()["cards"])


def _entry():
    doc = json.loads((DECKS_DIR / "index.json").read_text(encoding="utf-8"))
    return next(d for d in doc["decks"] if d["slug"] == SLUG)


def _open_build(page, slug=SLUG):
    page.evaluate("""() => { document.getElementById('modeSelect').value = 'build';
                             MM.setMode('build'); }""")
    page.wait_for_function("() => MM.mode === 'build'", timeout=10000)
    page.evaluate("s => Build.select(s)", slug)
    page.wait_for_function("s => Build.deckSlug === s && document.querySelector('#deckInner .lens-stats')",
                           arg=slug, timeout=30000)


def test_the_fixture_is_a_modern_deck_with_no_commander():
    """Guard the guard: every assertion below assumes this shape."""
    assert _cards()["format"] == "modern"
    e = _entry()
    assert e["format"] == "modern" and e["commander"] is None and e["image"]
    assert _main_copies() >= 60


def test_build_reads_the_format_from_the_deck(page):
    _open_build(page)
    n = _main_copies()
    stat = page.inner_text("#deckInner .lens-stats .lens-stat:first-child .lens-stat-n")
    assert stat == f"{n} / 60+", stat
    assert "Modern" in page.inner_text("#deckInner .lens-sub")
    assert page.evaluate("() => Build.deckFormat()") == "modern"
    assert page.evaluate("() => Session.format") == "modern"
    # Session is the one answer, and a format with no slot refuses a commander.
    refused = page.evaluate("""() => {
        const r = MM.allData.findIndex(d => d.n === 'Edgar Markov');
        return { ok: Session.setCommander(r), now: Session.commander };
    }""")
    assert refused == {"ok": False, "now": -1}, refused
    # No node on the graph wears the commander ring.
    page.wait_for_function("() => window.Force && Force.nodeCount > 0", timeout=15000)
    assert page.evaluate("() => Force.membership().commander") == 0
    assert page.js_errors == []


def test_the_illegal_lens_dims_by_modern_legality_not_by_colour(page):
    _open_build(page)
    page.evaluate("() => Build.toggle('showIllegal', true)")
    got = page.evaluate("""() => {
        const dim = Build.getDimmedIndices();
        const notModern = MM.allData.reduce((n, d) =>
            n + (!d.f || !d.f.split(',').includes('modern') ? 1 : 0), 0);
        const at = n => MM.allData.findIndex(d => d.n === n);
        return { size: dim.size, notModern: notModern,
                 bolt: dim.has(at('Lightning Bolt')),       // red, Modern-legal
                 arbor: dim.has(at('Arbor Elf')) };
    }""")
    assert got["size"] == got["notModern"] and got["size"] > 0, got
    assert got["bolt"] is False, "a red card is legal in a green Modern deck"
    assert got["arbor"] is False
    assert page.js_errors == []


def test_an_elves_card_reads_modern_legal_and_in_the_60(page):
    _open_build(page)
    name = "Arbor Elf"
    qty = next(c["quantity"] for c in _cards()["cards"] if c["name"] == name)
    page.evaluate("n => { MM.closeDetail(); MM.selectByName(n); }", name)
    page.wait_for_selector("#detailInner .deck-ctx", timeout=10000)
    page.wait_for_function("() => /Modern: legal/.test("
                           "(document.querySelector('#detailInner .detail-facts') || {}).textContent || '')",
                           timeout=15000)
    ctx = page.inner_text("#detailInner .deck-ctx")
    assert f"In the 60 ×{qty}" in ctx and "99" not in ctx, ctx
    assert "has no identity rule" in ctx, ctx
    assert page.js_errors == []


def test_set_as_commander_is_not_offered_in_a_format_without_one(page):
    _open_build(page)
    page.wait_for_function("() => window.Discovery && Discovery.isReady()", timeout=30000)
    page.evaluate("n => { MM.closeDetail(); MM.selectByName(n); }", "Edgar Markov")
    page.wait_for_selector("#detailInner .detail-actions", timeout=10000)
    assert "Set as commander" not in page.inner_text("#detailInner .detail-actions")
    # And it comes back with a Commander deck: the button is the FORMAT's, not gone.
    page.evaluate("() => Session.setFormat('commander')")
    page.evaluate("n => { MM.closeDetail(); MM.selectByName(n); }", "Edgar Markov")
    page.wait_for_selector("#detailInner .detail-actions", timeout=10000)
    assert "Set as commander" in page.inner_text("#detailInner .detail-actions")
    assert page.js_errors == []


def test_a_sideboard_is_listed_and_kept_out_of_the_deck(page):
    doc = _cards()
    side = [{"name": "Lightning Bolt", "quantity": 3}, {"name": "Thoughtseize", "quantity": 2}]
    doc["sideboard"] = side
    body = json.dumps(doc)
    page.route(f"**/data/decks/{SLUG}/cards.json*",
               lambda route: route.fulfill(status=200, content_type="application/json", body=body))
    _open_build(page)
    n = _main_copies()
    sect = page.locator("#deckInner .deck-sideboard")
    assert sect.count() == 1
    assert "Sideboard (5)" in sect.text_content()
    assert page.inner_text("#deckInner .lens-stats .lens-stat:first-child .lens-stat-n") == f"{n} / 60+"
    page.wait_for_function("() => window.Force && Force.nodeCount > 0", timeout=15000)
    on_graph = page.evaluate("""() => ['Lightning Bolt', 'Thoughtseize'].map(n =>
        Force.hasRow(MM.allData.findIndex(d => d.n === n)))""")
    assert on_graph == [False, False]
    assert page.js_errors == []


def test_the_deck_page_says_the_60_and_not_measured(browser, viz_server):
    page = browser.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{viz_server}/viz/deck.html?deck={SLUG}")
    page.wait_for_selector("#panels section", timeout=15000)
    try:
        text = page.inner_text("#panels")
        assert "The 60" in text and "The 99" not in text
        for pid in ("goldfish", "vitals", "bracket", "table"):
            line = page.locator(f"#panel-{pid} .na-line")
            assert line.count() == 1, f"#panel-{pid} has no n/a line"
            assert line.inner_text().startswith("Not measured for Modern:"), line.inner_text()
            # Never the call to action: no command to copy, no button to press.
            assert page.locator(f"#panel-{pid} .todo-cmd, #panel-{pid} .todo-run").count() == 0
        assert "Commander-only" in page.inner_text("#panel-goldfish")
        assert "Commander by T6" not in page.inner_text("#panel-cover")
        assert "Modern" in page.inner_text("#panel-cover .cov-sub")
        nm = json.loads((DECKS_DIR / SLUG / "audit.json").read_text())["not_measured"]
        items = page.locator("#panel-audit .audit-nm li").all_inner_texts()
        assert len(items) == len(nm) and len(nm) > 0
        for axis, why in nm.items():
            assert any(i.startswith(axis) and why in i for i in items), (axis, items)
        assert errors == []
    finally:
        page.close()


def test_the_workbench_card_says_modern(browser, viz_server):
    page = browser.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{viz_server}/viz/workbench.html")
    page.wait_for_selector(f".wb-card a.wb-title[href='deck.html?deck={SLUG}']",
                           state="attached", timeout=15000)
    try:
        got = page.evaluate("""s => {
            const a = document.querySelector(".wb-card a.wb-title[href='deck.html?deck=" + s + "']");
            const card = a.closest('.wb-card');
            return { sub: card.querySelector('.wb-sub').textContent,
                     art: (card.querySelector('img.wb-art') || {}).src || null };
        }""", SLUG)
        n = json.loads((DECKS_DIR / SLUG / "info.json").read_text())["size"]
        assert got["sub"] == f"Modern · G · {n} cards", got
        assert got["art"] == _entry()["image"], got
        assert errors == []
    finally:
        page.close()


def test_a_pasted_sixty_asks_for_a_format_and_the_brief_follows(discover_page):
    page = discover_page
    text = "\n".join(f"{c['quantity']} {c['name']}" for c in _cards()["cards"])
    page.evaluate("t => Discovery.importText(t)", text)
    page.wait_for_function("() => !!Discovery.formatHint", timeout=10000)
    assert page.evaluate("() => Session.format") == "commander", "never guessed"
    page.wait_for_selector("#deckInner .discover-format-hint", timeout=10000)
    assert "no commander — pick a format" in page.inner_text("#deckInner .discover-format-hint")
    page.select_option("#dcFmt", "modern")
    page.wait_for_function("() => Session.format === 'modern'", timeout=5000)
    brief = page.evaluate("() => Discovery.brief()")
    assert brief["format"] == "modern"
    assert "commander" not in brief
    assert "blocked" not in brief["_manamap"]
    assert len(brief["must_include"]) <= 60
    assert page.js_errors == []
