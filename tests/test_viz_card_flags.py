"""Bans, Game Changers and combo lines in the card panel (Area A4, 2026-10-09).

For a year the panel could not say BANNED: the projection's `f` lists LEGAL formats
only, so a banned card and a card from the wrong era both read "not legal" behind a
scary title. `data/card_flags.json` lands at boot now and the facts line has three
states; the Game Changer pill rides `quickStatsHtml`; and `comboLinesHtml` draws what
the card goes infinite with — the open deck's `combos.json` first, then the lazily
fetched corpus index.

The banned card and the Game Changer are picked AT TEST TIME out of `card_flags.json`
and the projection, never hardcoded — a ban list changes every few months. Nothing here
depends on a real per-deck `combos.json` (another agent is producing those): the fixture
is routed in at the URL Build fetches, the way the review grid mocks `serve` and the
panel tests route Scryfall. Every interaction is a real click or a real selection.
"""

from __future__ import annotations

import functools
import json

import pytest

from conftest_viz import page  # noqa: F401  (fixture)
from manamap.config import DATA_DIR
from test_viz_card_panel import _open, _route_scryfall
from test_viz_review_grid import _mock_api, _open_grid

pytestmark = pytest.mark.browser



# Read lazily, inside the tests: the unit tier's isolation run collects this file
# against an EMPTY data dir, and a module-level read there is a collection error
# that fails the whole tier before a single browser test is deselected.
@functools.lru_cache(maxsize=None)
def _flags():
    return json.loads((DATA_DIR / "card_flags.json").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def _index():
    return json.loads((DATA_DIR / "combo_index.json").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def _shark():
    doc = json.loads((DATA_DIR / "decks" / "sharknado" / "cards.json").read_text(encoding="utf-8"))
    return [c["name"] for c in doc["cards"]]


def _first_on_map(page, names):
    """The first of `names` the projection knows — chosen in the page, from its data."""
    name = page.evaluate("""names => { const have = new Set(MM.allData.map(d => d.n));
                                      return names.find(n => have.has(n)) || null; }""", names)
    assert name, "none of the candidates is in the projection"
    return name


def _wait_flags(page):
    page.wait_for_function("() => MM.cardFlags().as_of !== null", timeout=10000)


def _route_deck_file(page, slug, file, doc):
    page.route(f"**/data/decks/{slug}/{file}*",
               lambda route: route.fulfill(status=200, content_type="application/json",
                                           body=json.dumps(doc)))


def test_a_banned_card_reads_banned_and_a_game_changer_gets_its_pill(page):
    """BANNED from the flags file, not "not legal" from the absence of a legal entry;
    the GC pill on the header's stats line; and a card that is neither reads legal
    with no pill. Fails on the pre-flags `mana-map.js`, which has no `.legal-banned`."""
    _route_scryfall(page)
    _wait_flags(page)
    banned = _first_on_map(page, _flags()["banned"]["commander"])
    gc = _first_on_map(page, [n for n in _flags()["game_changers"]
                              if n not in _flags()["banned"]["commander"]])

    _open(page, banned)
    r = page.evaluate("""() => {
        const inner = document.getElementById('detailInner');
        const b = inner.querySelector('.detail-facts .legal-banned');
        return {banned: b ? b.textContent : null, title: b ? b.title : null,
                legalYes: !!inner.querySelector('.detail-facts .legal-yes'),
                legalNo: !!inner.querySelector('.detail-facts .legal-no')}; }""")
    assert r["banned"] == "Commander: BANNED", r
    assert _flags()["as_of"] in r["title"], r
    assert not r["legalYes"] and not r["legalNo"], r

    _open(page, gc)
    r = page.evaluate("""() => {
        const inner = document.getElementById('detailInner');
        const pill = inner.querySelector('.viewer-header .viewer-quickstats .gc-pill');
        const facts = inner.querySelector('.detail-facts');
        return {pill: pill ? pill.textContent : null, title: pill ? pill.title : null,
                facts: facts ? facts.textContent : '',
                notTitle: (inner.querySelector('.legal-not') || {}).title || ''}; }""")
    assert r["pill"] == "GC" and "Game Changer" in r["title"], r
    assert "BANNED" not in r["facts"], r
    assert r["notTitle"] == "", "not-legal must carry no scary title"

    # The other formats fold into the same three states, with a banned one somewhere.
    _open(page, banned)
    states = page.eval_on_selector_all("#detailInner .format-badge",
                                       "els => els.map(e => e.className.replace('format-badge', '').trim())")
    assert len(states) == 7 and set(states) <= {"", "legal", "banned"}, states
    assert page.js_errors == []


def test_a_card_with_no_flags_and_no_combos_shows_no_combo_block(page):
    """A plain legal card with nothing in the index: legal, no pill, no Combos header —
    the block renders NOTHING rather than an empty heading, and the lazy slot is gone
    once the index has answered."""
    _route_scryfall(page)
    _wait_flags(page)
    name = "The Elder Dragon War"
    assert name not in _index()["by_card"]
    _open(page, name)
    page.wait_for_function("() => !document.querySelector('#detailInner .combo-index-slot')",
                           timeout=30000)
    assert page.query_selector("#detailInner .combo-block") is None
    assert page.query_selector("#detailInner .gc-pill") is None
    assert "Commander: legal" in page.text_content("#detailInner .detail-facts")
    assert page.js_errors == []


def test_outside_build_the_index_lines_appear_and_nobody_is_in_a_deck(page):
    """Explore: the corpus index's lines land after the lazy fetch, every partner chip
    is unlit, the count line quotes `by_card`, and the title says to open a deck."""
    _route_scryfall(page)
    _wait_flags(page)
    name = _first_on_map(page, [n for n in _shark() if n in _index()["by_card"]])
    entry = _index()["by_card"][name]
    _open(page, name)
    page.wait_for_selector("#detailInner .combo-block .combo-line.is-index", timeout=30000)
    r = page.evaluate("""() => {
        const b = document.querySelector('#detailInner .combo-block');
        return {lines: b.querySelectorAll('.combo-line.is-index').length,
                lit: b.querySelectorAll('.combo-partner.is-in-deck').length,
                partners: b.querySelectorAll('.combo-partner').length,
                links: [...b.querySelectorAll('.combo-link')].map(a => a.href),
                count: b.querySelector('.combo-count').textContent,
                hint: b.querySelector('.combo-title').textContent}; }""")
    assert r["lines"] == len(entry["top"]), r
    assert r["lit"] == 0 and r["partners"] > 0, r
    assert all(h.startswith("https://commanderspellbook.com/combo/") for h in r["links"]), r
    assert f"in {entry['n']:,} known combo" in r["count"] and f"({entry['inf']:,} infinite)" in r["count"], r
    assert "open a deck in Build" in r["hint"], r
    assert page.js_errors == []


def test_build_combo_lines_light_the_partners_you_run(page):
    """With sharknado open and a fixture `combos.json` routed in: the deck's included
    line comes first, synchronously, its partner chip lit because the partner is in
    the 99; the near line names what is missing; the index lines follow the lazy load
    and dedupe against the deck's ids."""
    _route_scryfall(page)
    _mock_api(page)
    _wait_flags(page)
    name = _first_on_map(page, [n for n in _shark() if n in _index()["by_card"]])
    partner = _first_on_map(page, [n for n in _shark() if n != name])
    top_id = _index()["combos"][_index()["by_card"][name]["top"][0]][0]
    combos = {
        "summary": {"included": 1, "near": 1},
        "included": [{"id": top_id, "cards": [name, partner], "produces": ["Infinite mana"],
                      "infinite": True, "bracket": 3, "banned": False,
                      "mana_value_needed": 4, "popularity": 10,
                      "assumes_other_commander": True}],
        "near": [{"id": "fixture-near-1", "cards": [name, "Thassa's Oracle"],
                  "missing": ["Thassa's Oracle"], "infinite": False, "bracket": None,
                  "banned": False, "mana_value_needed": 2, "popularity": 1,
                  "assumes_other_commander": False}],
    }
    _route_deck_file(page, "sharknado", "combos.json", combos)
    _open_grid(page)
    assert page.evaluate("() => Build.deckCombos() !== null"), "the fixture was not loaded"
    _open(page, name)
    deck = page.evaluate("""() => {
        const b = document.querySelector('#detailInner .combo-block');
        const line = b.querySelector('.combo-line.is-deck');
        const near = b.querySelector('.combo-line.is-near');
        return {firstIsDeck: b.querySelector('.combo-line') === line,
                tag: line.querySelector('.combo-tag').textContent,
                partners: [...line.querySelectorAll('.combo-partner')]
                    .map(e => ({n: e.textContent, lit: e.classList.contains('is-in-deck')})),
                inf: !!line.querySelector('.inf-badge'),
                bracket: line.querySelector('.bracket-pill').textContent,
                note: (line.querySelector('.combo-note') || {}).textContent,
                nearTag: near.querySelector('.combo-tag').textContent,
                nearBanned: near.querySelector('.bracket-pill').classList.contains('banned'),
                hint: b.querySelector('.combo-title').textContent}; }""")
    assert deck["firstIsDeck"], deck
    assert deck["tag"] == "in this deck", deck
    assert deck["partners"] == [{"n": partner, "lit": True}], deck
    assert deck["inf"] and deck["bracket"] == "B3", deck
    assert deck["note"] == "assumes its own commander", deck
    assert deck["nearTag"] == "one card short: Thassa's Oracle", deck
    assert deck["nearBanned"], "a null bracket reads as banned"
    assert "open a deck in Build" not in deck["hint"], deck

    page.wait_for_selector("#detailInner .combo-block .combo-line.is-index", timeout=30000)
    ids = page.eval_on_selector_all("#detailInner .combo-line",
                                    "els => els.map(e => e.getAttribute('data-id'))")
    assert ids.count(top_id) == 1, f"the deck's id was drawn again by the index: {ids}"
    assert len(ids) == 2 + len(_index()["by_card"][name]["top"]) - 1, ids
    assert page.js_errors == []


def test_build_deck_block_says_where_a_game_changer_stands(page):
    """A GC in sharknado's 99 against the deck's own report (floor 4: no limit), then a
    fixture report at floor 3 with three GCs: a GC NOT in the deck would be the 4th
    and the floor moves to 4. A non-GC card gets no row at all."""
    _route_scryfall(page)
    _mock_api(page)
    _wait_flags(page)
    report = json.loads((DATA_DIR / "decks" / "sharknado" / "bracket_report.json").read_text())
    in_deck = _first_on_map(page, [n for n in _shark() if n in _flags()["game_changers"]])
    outside = _first_on_map(page, [n for n in _flags()["game_changers"] if n not in _shark()
                                   and n not in _flags()["banned"]["commander"]])
    _open_grid(page)
    _open(page, in_deck)
    row = "#detailInner .deck-ctx .deck-ctx-gc"
    page.wait_for_selector(row, timeout=5000)
    text = page.text_content(row)
    assert "Game Changer" in text and f"this deck runs {len(report['game_changers'])}" in text, text
    assert f"bracket {report['floor']}" in text, text
    ctx = page.evaluate("n => Build.cardContext(MM.allData.findIndex(d => d.n === n))", in_deck)
    assert ctx["gameChanger"] and ctx["bracket"]["floor"] == report["floor"], ctx
    assert ctx["bracket"]["gcLimit"] == {1: 0, 2: 0, 3: 3}.get(report["floor"]), ctx

    _open(page, "Arcane Signet")
    page.wait_for_selector("#detailInner .deck-ctx", timeout=5000)
    assert page.query_selector(row) is None, "a non-GC card has no Game Changer row"

    # Floor 3 with the allowance used up: the next GC moves the floor.
    _route_deck_file(page, "sharknado", "bracket_report.json",
                     dict(report, floor=3, floor_name="Upgraded",
                          game_changers=report["game_changers"][:3]))
    page.evaluate("() => Build.select('sharknado')")
    page.wait_for_function("() => Build.cardContext(0) && Build.cardContext(0).bracket"
                           " && Build.cardContext(0).bracket.floor === 3", timeout=30000)
    _open(page, outside)
    page.wait_for_selector(row, timeout=5000)
    text = page.text_content(row)
    assert "would be the 4th: floor moves to 4" in text, text
    assert "Not in the 99" in page.text_content("#detailInner .deck-ctx")
    assert page.js_errors == []
