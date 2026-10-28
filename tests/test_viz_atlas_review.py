"""The pilot's UI review of the Atlas, 2026-10-28 — the P0 quick wins, driven in a browser.

Live on `?card=Windfall` the relation buttons and Keep sat under twelve combo rows, the
first of them B1 lines in colours no deck at the table plays; on `?deck=sharknado` a
review grid with every card judged opened across 40% of the map reading "Nothing here",
and the deck header named the commander and a count and nothing else. Each test below
drives the real page and was proven by putting the old behaviour back (docs/viz.md).

Nothing here writes: the manifest, the watch list and `info.json` are routed fixtures
where a test needs a state the tracked files may not be in, and every `/api/` call the
Build tests could make is answered by a mock whose catch-all refuses the rest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest_viz import (WATCHLIST_FIXTURE, _record, _wait_for_boot, page,  # noqa: F401
                          serve_card_images_locally, serve_watchlist_fixture)

pytestmark = pytest.mark.browser

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CARD = "Windfall"


def _combo_index():
    return json.loads((DATA / "combo_index.json").read_text(encoding="utf-8"))


def _manifest():
    return json.loads((DATA / "decks" / "index.json").read_text(encoding="utf-8"))


# ── 1. Discover: the verbs under the card, the combos folded, your decks first ──


def _fixture_deck(index):
    """A one-deck manifest built from the index's OWN lines for Windfall, so the test
    survives a Spellbook refresh: the LAST line with two or more partners, made one
    card short — the sort has to carry it from the bottom to the top."""
    entry = index["by_card"][CARD]
    lines = [(pos, index["combos"][i]) for pos, i in enumerate(entry["top"])]
    multi = [(pos, c) for pos, c in lines if len([n for n in c[1] if n != CARD]) >= 2]
    assert multi, "no Windfall line with two partners left to build the fixture from"
    pos, combo = multi[-1]
    assert pos > 0, "the chosen line already leads; the fixture would prove nothing"
    partners = [n for n in combo[1] if n != CARD]
    deck = {"slug": "fixture-wheels", "deck_name": "Fixture Wheels", "commander": "Nobody",
            "format": "commander", "stage": "sleeved", "has": {}, "branches": [],
            "cards": [CARD] + partners[:-1]}
    return {"decks": [deck], "drafts": []}, combo[0], partners[-1]


@pytest.fixture
def windfall(browser, viz_server):
    manifest, line_id, missing = _fixture_deck(_combo_index())
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list[str] = []
    add = _record(errors)
    page.on("pageerror", lambda e: add(e))
    page.on("console", lambda m: add(m.text) if m.type == "error" else None)
    serve_card_images_locally(page)
    page.route("**/data/decks/index.json*", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(manifest)))
    page.goto(f"{viz_server}/viz/index.html?card={CARD}")
    page.add_style_tag(content="*, *::before, *::after {"
                               " transition: none !important; animation: none !important; }")
    page.js_errors = errors
    _wait_for_boot(page, "() => window.Discovery && Discovery.isReady() && Discovery.current >= 0",
                   "discovery to land on Windfall")
    # The tier needs colour identity, which is the projection's; the panel re-renders
    # once when it lands, so wait for both and for the lazy combo index's lines.
    _wait_for_boot(page, "() => window.MM && MM.allData && MM.allData.length > 0",
                   "the projection to arrive behind the landing")
    page.wait_for_function("() => document.querySelectorAll("
                           "'#deckInner .combo-lines > .combo-line.is-index').length > 3",
                           timeout=60000)
    page.line_id, page.missing = line_id, missing
    try:
        yield page
    finally:
        page.close()


def test_the_relations_and_keep_sit_under_the_card_above_the_combos(windfall):
    """The mode's verbs were a full panel-scroll down, under Windfall's combo rows."""
    r = windfall.evaluate("""() => {
        const inner = document.getElementById('deckInner');
        const top = s => { const e = inner.querySelector(s);
                           return e ? e.getBoundingClientRect().top : null; };
        const img = inner.querySelector('.detail-card-image');
        return {img: img ? img.getBoundingClientRect().bottom : null,
                rel: top('.discover-relations'), keep: top('.detail-actions .discover-keep'),
                combos: top('.combo-block'), text: top('.detail-text, .detail-face, .detail-type'),
                fold: window.innerHeight};
    }""")
    assert None not in (r["img"], r["rel"], r["keep"], r["combos"]), r
    assert r["img"] <= r["rel"] < r["keep"] < r["combos"], (
        f"the relations and Keep are not directly under the card, above the combos: {r}")
    assert windfall.js_errors == []


def test_combos_show_three_rows_then_all_on_request(windfall):
    q = """() => {
        const ls = [...document.querySelectorAll('#deckInner .combo-lines > .combo-line')];
        const more = document.querySelector('#deckInner .combo-more');
        return {all: ls.length, visible: ls.filter(e => e.offsetParent !== null).length,
                more: more ? more.textContent : null};
    }"""
    before = windfall.evaluate(q)
    assert before["all"] > 3, before
    assert before["visible"] == 3, f"the block is not folded to three rows: {before}"
    assert before["more"] == f"show all {before['all']}", before
    windfall.click("#deckInner .combo-more")
    after = windfall.evaluate(q)
    assert after["visible"] == after["all"], after
    assert after["more"] == "show 3", after
    windfall.click("#deckInner .combo-more")
    assert windfall.evaluate(q)["visible"] == 3
    assert windfall.js_errors == []


def test_combos_lead_with_your_decks_and_keep_the_index_order_within_a_tier(windfall):
    """The fixture deck holds Windfall and all but one card of the index's last
    multi-partner line: that line must lead, saying which card it is short, and the
    tiers must never go backwards — while inside a tier the index's own order stands."""
    index = _combo_index()
    top = [index["combos"][i][0] for i in index["by_card"][CARD]["top"]]
    r = windfall.evaluate("""() => [...document.querySelectorAll(
        '#deckInner .combo-lines > .combo-line.is-index')].map(e => ({
            id: e.getAttribute('data-id'), tier: +e.getAttribute('data-tier'),
            tag: (e.querySelector('.combo-tag') || {}).textContent || ''}))""")
    assert r, "no index lines drew"
    assert r[0]["id"] == windfall.line_id, f"the line one card short does not lead: {r[:3]}"
    assert r[0]["tag"] == f"one card short in Fixture Wheels: {windfall.missing}", r[0]
    tiers = [x["tier"] for x in r]
    assert tiers == sorted(tiers), f"a tier went backwards: {tiers}"
    assert set(tiers) >= {1, 3}, f"the fixture should reach tier 1 and tier 3: {tiers}"
    for t in set(tiers):
        ids = [x["id"] for x in r if x["tier"] == t]
        assert ids == [i for i in top if i in ids], f"tier {t} re-ordered the index: {ids}"
    assert windfall.js_errors == []


def test_an_empty_relation_is_disabled_and_says_why(windfall):
    """Windfall has no rule-based synergy partners. "Synergy 0" read as an answer."""
    r = windfall.evaluate("""() => {
        const b = document.querySelector('#deckInner .discover-rel[data-rel="synergy"]');
        return {disabled: b.disabled, count: b.querySelector('.discover-count').textContent,
                title: b.title, onclick: b.getAttribute('onclick')};
    }""")
    assert r["disabled"] and r["onclick"] is None, r
    assert r["count"] == "none", r
    assert "No rule-based synergy partners" in r["title"], r
    assert windfall.js_errors == []


# ── 3 + 4. Build: the review grid's collapsed state, the deck header's badges ──


def _mock_api(page):
    """Refuse every `/api/` call but the probe — nothing here may write."""
    posted: list = []
    page.route("**/api/**", lambda route: (posted.append(route.request.url), route.fulfill(
        status=500, content_type="application/json", body='{"error": "unmocked in test"}')))
    page.route("**/api/health", lambda route: route.fulfill(
        status=200, content_type="application/json", body='{"result": {"ok": true}}'))
    page.route("**/api/", lambda route: route.fulfill(
        status=200, content_type="application/json", body='{"commands": ["watch/mark"]}'))
    page.route("**/api/watch/mark", lambda route: (posted.append(route.request.post_data),
                                                   route.fulfill(status=500, body="{}")))
    return posted


def _all_reviewed_watchlist():
    doc = json.loads(WATCHLIST_FIXTURE.read_text(encoding="utf-8"))
    for c in doc["sets"][0]["cards"]:
        c["verdict"] = "pass"
    return doc


def _open_sharknado(page, watchlist=None, routes=()):
    if watchlist is None:
        serve_watchlist_fixture(page)
    else:
        body = json.dumps(watchlist)
        page.route("**/data/decks/sharknado/watchlist.json*", lambda route: route.fulfill(
            status=200, content_type="application/json", body=body))
    for file, body in routes:
        page.route(f"**/data/decks/sharknado/{file}*",
                   (lambda b: lambda route: route.fulfill(
                       status=404 if b is None else 200, content_type="application/json",
                       body="{}" if b is None else json.dumps(b)))(body))
    page.evaluate("() => Api.refresh()")
    page.evaluate("""() => { document.getElementById('modeSelect').value = 'build';
                             MM.setMode('build'); }""")
    page.wait_for_function("() => MM.mode === 'build'", timeout=10000)
    page.evaluate("() => Build.select('sharknado')")
    page.wait_for_function("() => Build.deckSlug === 'sharknado' && "
                           "document.querySelector('#deckPanel .lens-title')", timeout=30000)


def test_a_set_with_every_card_judged_collapses_to_a_chip(page):
    posted = _mock_api(page)
    doc = _all_reviewed_watchlist()
    _open_sharknado(page, doc)
    page.wait_for_selector("#cgChip", timeout=10000)
    s = doc["sets"][0]
    n = len(s["cards"])
    r = page.evaluate("""() => {
        const c = document.getElementById('cgChip');
        return {grid: !!document.getElementById('candGrid'), text: c.innerText,
                h: c.getBoundingClientRect().height,
                plot: document.getElementById('plot').getBoundingClientRect().height,
                card: Build.gridCard};
    }""")
    assert not r["grid"], "the grid opened on a set with nothing left to review"
    assert s["title"] in r["text"] and f"{n} / {n} ✓" in r["text"], r
    assert r["h"] < 48, f"the chip is not one line: {r}"
    assert r["card"] is None, "Build.gridCard names a card while the grid is collapsed"
    # A collapsed grid is not a live one: its keys must not reach a write.
    page.keyboard.press("w")
    page.wait_for_timeout(300)
    assert posted == [], posted
    # The chip opens the grid, and the grid's own rules are untouched by it.
    page.click("#cgChip")
    page.wait_for_selector("#candGrid .cg-header", timeout=5000)
    assert page.evaluate("() => !document.getElementById('cgChip')")
    assert "every card has a verdict" in page.inner_text("#candGrid")
    # Closing it leaves the chip, not nothing.
    page.click("#candGrid [data-close]")
    page.wait_for_selector("#cgChip", timeout=5000)
    assert page.evaluate("() => !document.getElementById('candGrid')")
    assert page.js_errors == []


def test_a_set_with_cards_left_still_opens_the_grid(page):
    """The control: the frozen fixture has unreviewed cards, and the grid opens."""
    _mock_api(page)
    _open_sharknado(page)
    page.wait_for_function("() => document.querySelectorAll('#candGrid .cg-tile').length > 0",
                           timeout=30000)
    assert page.evaluate("() => !document.getElementById('cgChip')")
    assert page.js_errors == []


def test_the_deck_header_carries_its_badges_from_the_files(page):
    """Name, version, rung, bracket, ★ grade, W–L and the DTI with its date — each the
    file's own value, read here from the same files the page reads."""
    _mock_api(page)
    _open_sharknado(page)
    entry = next(d for d in _manifest()["decks"] if d["slug"] == "sharknado")
    info = json.loads((DATA / "decks" / "sharknado" / "info.json").read_text(encoding="utf-8"))
    page.wait_for_selector("#deckPanel .deck-badges", timeout=10000)
    r = page.evaluate("""() => {
        const p = document.getElementById('deckPanel');
        const t = s => { const e = p.querySelector(s); return e ? e.textContent : null; };
        return {title: t('.lens-title'), sub: t('.lens-sub'), version: t('.deck-badge-version'),
                rung: t('.deck-badge-rung'), bracket: t('.deck-badge-bracket'),
                grade: t('.deck-badge-grade'), record: t('.deck-badge-record'),
                dti: t('.deck-badge-dti')};
    }""")
    if entry["deck_name"] == entry["commander"]:
        assert r["title"] == "Sharknado", r          # named by its slug, not the commander twice
    assert "Brallin, Skyshark Rider" in r["sub"], "a partner commander is missing from the header"
    want = (entry.get("paper") or {}).get("release") or (entry.get("version") or {}).get("release")
    assert want and r["version"] == want, (want, r)
    assert r["rung"] == "Sleeved", r
    b = info["bracket"]
    assert r["bracket"] == f"B{b.get('target') or b['floor']}", r
    if entry["has"].get("rating"):
        rating = json.loads((DATA / "decks" / "sharknado" / "rating.json").read_text(encoding="utf-8"))
        assert r["grade"] == "★ " + (rating.get("grade_display") or rating["grade"]), r
    rec = info.get("record") or {}
    if rec.get("games"):
        assert r["record"].startswith(f"{rec.get('win', 0)}–{rec.get('loss', 0)}"), r
    dc = info.get("deckcheck")
    if dc:
        assert f"DTI {dc['score']}/{dc['max_score']}" in r["dti"] and dc["as_of"] in r["dti"], r
    assert page.js_errors == []


def test_absent_figures_are_absent_never_zero(page):
    """No DeckCheck reading, no logged game, no rating: no record badge, no grade badge,
    a muted "no DTI" — and no 0 or dash standing in for a value anywhere."""
    _mock_api(page)
    info = json.loads((DATA / "decks" / "sharknado" / "info.json").read_text(encoding="utf-8"))
    info.pop("deckcheck", None)
    info["record"] = {"games": 0, "win": 0, "loss": 0, "draw": 0}
    _open_sharknado(page, routes=[("info.json", info), ("rating.json", None)])
    page.wait_for_selector("#deckPanel .deck-badges", timeout=10000)
    r = page.evaluate("""() => {
        const p = document.querySelector('#deckPanel .deck-badges');
        return {text: p.innerText, record: !!p.querySelector('.deck-badge-record'),
                grade: !!p.querySelector('.deck-badge-grade'),
                dti: (p.querySelector('.deck-badge-dti') || {}).textContent,
                absent: !!p.querySelector('.deck-badge-dti.is-absent')};
    }""")
    assert not r["record"], f"a record badge drew with no logged game: {r}"
    assert not r["grade"], f"a grade badge drew with no rating on file: {r}"
    assert r["dti"] == "no DTI" and r["absent"], r
    assert "0–0" not in r["text"] and "—" not in r["text"], r
    assert page.js_errors == []


# ── 5. The status line after the graph draws ──


def test_build_does_not_leave_loading_embeddings_in_the_status(page):
    """Build set its status, then seeded the graph, which wrote "Loading embeddings…"
    over it and never took it down once the matrix landed."""
    _mock_api(page)
    _open_sharknado(page)
    page.wait_for_function("() => window.Force && Force.nodeCount > 50", timeout=60000)
    page.wait_for_function(
        "() => document.getElementById('status').textContent !== 'Loading embeddings…'",
        timeout=30000)
    status = page.text_content("#status")
    assert "cards lit" in status, status
    assert page.js_errors == []
