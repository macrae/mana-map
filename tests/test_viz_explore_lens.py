"""Explore as a LENS, not a museum — the pilot's UI review, 2026-10-28 (Atlas P1, Explore half).

Explore opened coloured by supertype under nine type pills, a set picker, two dates, Topo /
Labels / Motion and "34,964 cards shown". His questions were "where do my decks live",
"where's the new set" and "what is this region FOR". Each test drives the real page and was
proven by putting the old behaviour back (docs/viz.md, "Explore opens as a lens").

Deck membership comes from a ROUTED manifest, never the tracked one: which decks are
sleeved this week is the pilot's business, and a test that read `data/decks/index.json`
would fail every time he sleeves a deck. The set lens reads the tracked `sets.json` and
`viz_index.json` because deriving the newest set FROM THE DATA is the behaviour under test
— and a second test moves a release date to prove the pill follows the data, not a literal.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from conftest_viz import _record, _wait_for_boot

pytestmark = pytest.mark.browser

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Two sleeved decks sharing Sol Ring, one bench deck. Real card names, so they resolve
# against the corpus; the deck names are made up so nothing collides with the fleet.
# Beta's name is SHOUTED, as five of the fleet's were authored (EDGAR MARKOV, THE
# UR-DRAGON): the legend draws it in title case (`DeckFacts.displayName`, 2026-10-29).
MANIFEST = {"drafts": [], "decks": [
    {"slug": "fixture-alpha", "deck_name": "Fixture Alpha", "commander": "Nobody",
     "format": "commander", "stage": "sleeved", "has": {}, "branches": [],
     "cards": ["Sol Ring", "Lightning Bolt"]},
    {"slug": "fixture-beta", "deck_name": "FIXTURE BETA", "commander": "Nobody",
     "format": "commander", "stage": "sleeved", "has": {}, "branches": [],
     "cards": ["Sol Ring", "Counterspell"]},
    {"slug": "fixture-gamma", "deck_name": "Fixture Gamma", "commander": "Nobody",
     "format": "commander", "stage": "bench", "has": {}, "branches": [],
     "cards": ["Craterhoof Behemoth"]},
]}

#: Mirrors `LENS_SET_TYPES` in viz/js/mana-map.js — what counts as a release for the
#: lens, by Scryfall's `set_type` in sets.json (a 100-card floor was its proxy until
#: 2026-10-30).
LENS_SET_TYPES = {"expansion", "core", "masters", "draft_innovation", "commander"}


def _open(browser, viz_server, query="?mode=explore", sets_body=None):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list[str] = []
    add = _record(errors)
    page.on("pageerror", lambda e: add(e))
    page.on("console", lambda m: add(m.text) if m.type == "error" else None)
    page.route("**/data/decks/index.json*", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(MANIFEST)))
    if sets_body is not None:
        page.route("**/data/sets.json*", lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(sets_body)))
    page.goto(f"{viz_server}/viz/index.html{query}")
    page.add_style_tag(content="*, *::before, *::after {"
                               " transition: none !important; animation: none !important; }")
    page.js_errors = errors
    _wait_for_boot(page, "() => window.MM && MM.allData && MM.allData.length > 0",
                   "the projection to load")
    _wait_for_boot(page, "() => MM.mapRenderer && MM.mapRenderer.getCamera() !== null"
                         " && document.querySelector('.map-legend-row')",
                   "the atlas to draw its legend")
    return page


@pytest.fixture
def lens(browser, viz_server):
    page = _open(browser, viz_server)
    # The manifest arrives through Discovery, and `MM.setDecks` repaints the legend
    # synchronously when it lands — so waiting on the manifest, not on the legend, keeps
    # a wrong default a FAILURE in the test rather than a timeout here.
    page.wait_for_function("() => window.Discovery && Discovery.decks"
                           ".some(d => d.slug === 'fixture-alpha')", timeout=30000)
    try:
        yield page
    finally:
        page.close()


def _layers(page):
    return page.evaluate("""() => MM.mapRenderer.layers
        .filter(l => l.mode === 'markers' && l.marker && l.marker.glow !== undefined)
        .map(l => ({name: l.name, n: l.customdata.length, rows: l.customdata,
                    opacity: Array.isArray(l.marker.opacity) ? null : l.marker.opacity,
                    color: l.marker.color, glow: l.marker.glow}))""")


def _row(page, name):
    return page.evaluate("n => MM.allData.findIndex(d => d.n === n)", name)


# ── (a) Explore opens coloured by MY DECKS, with a legend naming each one ──────────────


def test_explore_opens_coloured_by_my_decks(lens):
    page = lens
    assert page.evaluate("MM.grouping") == "decks", "Explore did not open on My decks"
    assert page.eval_on_selector("#colorBy", "s => s.value") == "decks"

    legend = page.evaluate("""() => ({
        head: (document.querySelector('.map-legend-head') || {}).textContent,
        rows: [...document.querySelectorAll('.map-legend-row')].map(r => r.textContent)})""")
    assert legend["head"] == "Your sleeved decks"
    assert legend["rows"][:4] == ["Fixture Alpha", "Fixture Beta", "In 2+ decks", "Not in a deck"], (
        "the legend should name each sleeved deck, then the shared mark, then the texture: "
        f"{legend['rows']}")
    assert "Fixture Gamma" not in legend["rows"], "a bench deck lit by default"
    assert legend["rows"][-1].startswith("Bench & brewing (1)")

    layers = {l["name"]: l for l in _layers(page)}
    assert _row(page, "Lightning Bolt") in layers["Fixture Alpha"]["rows"]
    assert _row(page, "Counterspell") in layers["Fixture Beta"]["rows"]
    # In two decks is SHARED, never the first deck's colour.
    assert _row(page, "Sol Ring") in layers["In 2+ decks"]["rows"]
    assert _row(page, "Sol Ring") not in layers["Fixture Alpha"]["rows"]
    # Everything else is texture: faint, no halo, and drawn FIRST so decks paint over it.
    rest = layers["Not in a deck"]
    assert rest["n"] > 30000 and rest["opacity"] < 0.3 and rest["glow"] is False
    assert layers["Fixture Alpha"]["opacity"] >= 0.8
    order = [l["name"] for l in _layers(page)]
    assert order[0] == "Not in a deck", f"the texture must draw under the decks: {order}"

    # No corpus count with nothing narrowed — what is lit, said once.
    status = page.text_content("#status")
    assert "cards shown" not in status and "34," not in status, status
    assert "sleeved decks" in status

    # The bench toggle is a REAL control (hit-tested click, not a synthesised one).
    page.click(".map-legend-row[data-action=bench]")
    page.wait_for_function("() => [...document.querySelectorAll('.map-legend-row')]"
                           ".some(r => r.textContent === 'Fixture Gamma')")
    layers = {l["name"]: l for l in _layers(page)}
    assert _row(page, "Craterhoof Behemoth") in layers["Fixture Gamma"]["rows"]
    assert page.evaluate("""() => document.querySelector('.map-legend-row[data-action=bench]')
        .getAttribute('aria-checked')""") == "true"
    assert not page.js_errors, page.js_errors


def test_an_explicit_colour_in_the_url_wins(browser, viz_server):
    page = _open(browser, viz_server, "?mode=explore&color=rarity")
    try:
        assert page.evaluate("MM.grouping") == "rarity"
        assert page.eval_on_selector("#colorBy", "s => s.value") == "rarity"
        # My decks is Explore's language only: Build does not offer it.
        page.evaluate("MM.setMode('build')")
        offered = page.evaluate("""() => [...document.getElementById('colorBy').options]
            .filter(o => !o.hidden).map(o => o.value)""")
        assert "decks" not in offered
        assert not page.js_errors, page.js_errors
    finally:
        page.close()


# ── (b) The set lens: one click to the newest set, derived from the data ────────────────


def _expected_newest(sets):
    """The rule as documented: released on or before today, a release-class set type with
    at least one corpus card, latest date, ties to the bigger set."""
    index = json.loads((DATA / "viz_index.json").read_text(encoding="utf-8"))
    tally: dict[str, int] = {}
    for rec in index:
        if rec and rec.get("e"):
            tally[rec["e"]] = tally.get(rec["e"], 0) + 1
    today = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
    best = None
    for code, meta in sets.items():
        rel, n = meta.get("released_at") or "", tally.get(code, 0)
        if not rel or rel > today or not n or meta.get("set_type") not in LENS_SET_TYPES:
            continue
        if best is None or rel > best[1] or (rel == best[1] and n > best[2]):
            best = (code, rel, n)
    assert best, "no set qualifies for the lens; the data or the rule moved"
    return best


def test_the_set_lens_lights_the_newest_set(lens):
    page = lens
    sets = json.loads((DATA / "sets.json").read_text(encoding="utf-8"))
    code, _rel, n = _expected_newest(sets)
    page.wait_for_selector("#setLens:not([hidden])")
    assert page.get_attribute("#setLens", "data-set") == code
    assert sets[code]["name"] in page.text_content("#setLens")

    page.click("#setLens")
    pf = page.evaluate("MM.printFocus")
    assert pf["set"] == code and pf["count"] == n
    assert page.evaluate("document.getElementById('setLens').classList.contains('active')")
    # It IS the set filter: the Filters count says so, and the colouring is untouched.
    assert page.text_content("#filtersCount") == "1"
    assert page.evaluate("MM.grouping") == "decks"
    # The lens's non-deck cards LIGHT (they are most of the set); the rest sink.
    rest = next(l for l in page.evaluate("""() => MM.mapRenderer.layers
        .filter(l => l.name === 'Not in a deck').map(l => ({rows: l.customdata,
                op: l.marker.opacity}))"""))
    lit = sum(1 for o in rest["op"] if o >= 0.8)
    assert 0 < lit <= n, f"{lit} texture points lit for a {n}-card set"

    page.click("#setLens")
    assert page.evaluate("MM.printFocus") is None
    assert page.is_hidden("#filtersCount")
    assert not page.js_errors, page.js_errors


def test_the_set_lens_follows_the_data_not_a_literal(browser, viz_server):
    """Move the newest set's release into the past: the pill must pick the next one."""
    sets = json.loads((DATA / "sets.json").read_text(encoding="utf-8"))
    first, _rel, _n = _expected_newest(sets)
    moved = dict(sets)
    moved[first] = dict(sets[first], released_at="2001-01-01")
    second, _rel2, _n2 = _expected_newest(moved)
    assert second != first
    page = _open(browser, viz_server, sets_body=moved)
    try:
        page.wait_for_selector("#setLens:not([hidden])", timeout=30000)
        assert page.get_attribute("#setLens", "data-set") == second
        assert not page.js_errors, page.js_errors
    finally:
        page.close()


# ── (c) Filters behind one button, counting what is active ──────────────────────────────


def test_filters_live_behind_one_button_that_counts_them(lens):
    page = lens
    # Out of sight on arrival: the set picker, the dates and the nine type pills.
    assert page.is_hidden("#filtersPop")
    for sel in ("#setSelect", "#firstAfter", "#firstBefore", "#toggles button"):
        assert not page.is_visible(sel), f"{sel} is still in the toolbar"
    assert page.is_hidden("#filtersCount")
    assert page.evaluate("document.getElementById('toggles').closest('#filtersPop') !== null")

    page.click("#filtersBtn")
    assert page.is_visible("#setSelect") and page.is_visible("#toggles button")
    page.click("#toggles button[data-supertype=Land]")
    assert page.text_content("#filtersCount") == "1"
    status = page.text_content("#status")
    assert "cards of the chosen types" in status, status   # a count, because one is narrowed

    code = page.evaluate("[...document.getElementById('setSelect').options]"
                         ".map(o => o.value).find(Boolean)")
    page.select_option("#setSelect", code)
    assert page.text_content("#filtersCount") == "2"
    page.fill("#firstAfter", "2020-01-01")
    page.dispatch_event("#firstAfter", "change")
    assert page.text_content("#filtersCount") == "3"
    assert page.get_attribute("#filtersBtn", "title").startswith("Active: ")

    # Escape closes the popover FIRST and peels nothing.
    page.keyboard.press("Escape")
    assert page.is_hidden("#filtersPop")
    assert page.evaluate("MM.printFocus")["set"] == code
    # A click outside closes it too.
    page.click("#filtersBtn")
    page.mouse.click(300, 600)
    assert page.is_hidden("#filtersPop")

    # The filter API is unchanged and drives the badge with the popover shut.
    page.evaluate("MM.clearPrintFocus()")
    assert page.text_content("#filtersCount") == "1"          # the Land pill is still off
    page.evaluate(f"MM.setPrintFocus({json.dumps(code)})")
    page.wait_for_function("document.getElementById('filtersCount').textContent === '2'")
    assert not page.js_errors, page.js_errors


# ── (d) Topo, Labels and Motion in the settings gear ───────────────────────────────────


def test_topo_and_motion_live_in_the_settings_menu(lens):
    page = lens
    for bid in ("toggleContours", "toggleLabels", "toggleMotion"):
        where = page.evaluate(f"""() => {{
            const b = document.getElementById('{bid}');
            return {{inToolbar: !!b.closest('.toolbar'), inMenu: !!b.closest('#exploreSettings')}};
        }}""")
        assert where == {"inToolbar": False, "inMenu": True}, (bid, where)
        assert not page.is_visible(f"#{bid}")

    assert page.is_visible("#exploreSettingsBtn")
    page.click("#exploreSettingsBtn")
    assert page.is_visible("#toggleContours") and page.is_visible("#toggleMotion")
    assert page.get_attribute("#exploreSettingsBtn", "aria-expanded") == "true"

    # A real, hit-tested click: the menu must be reachable over the canvas, and the
    # button must still be wired to the handler that turns the contours on.
    assert "active" not in page.get_attribute("#toggleContours", "class")
    page.click("#toggleContours")
    assert "active" in page.get_attribute("#toggleContours", "class"), (
        "Topo in the menu did not switch the contours on")
    motion = page.evaluate("MM.mapRenderer.motion")
    page.click("#toggleMotion")
    assert page.evaluate("MM.mapRenderer.motion") is (not motion)

    page.keyboard.press("Escape")
    assert page.is_hidden("#exploreSettings")
    # The gear belongs to Explore: another mode takes it away.
    page.evaluate("MM.setMode('discover')")
    assert page.is_hidden("#exploreBar")
    assert not page.js_errors, page.js_errors


# ── (e) A region label carries a plain-words subtitle ───────────────────────────────────


def test_a_region_label_carries_a_functional_subtitle(lens):
    page = lens
    page.wait_for_function("() => document.querySelector('.map-label .map-label-sub')",
                           timeout=30000)
    # Every clickable label carries one, and its tooltip says how many cards and from what.
    info = page.evaluate("""() => {
        const l = [...document.querySelectorAll('.map-label-l0')]
            .find(el => el.style.display !== 'none' && el.querySelector('.map-label-sub'));
        if (!l) return null;
        const sub = l.querySelector('.map-label-sub');
        const r = l.getBoundingClientRect();
        return {name: l.querySelector('.map-label-name').textContent, sub: sub.textContent,
                title: l.title, shown: getComputedStyle(sub).display,
                x: r.x + r.width / 2, y: r.y + r.height / 2};
    }""")
    assert info, "no L0 label with a subtitle on the fitted atlas"
    assert info["sub"] and info["sub"] != info["name"]
    assert info["shown"] == "none", "at the whole-map fit the subtitle waits for hover"
    assert " cards (from " in info["title"]
    page.mouse.move(info["x"], info["y"])
    page.wait_for_function("""name => [...document.querySelectorAll('.map-label-l0')]
        .some(l => l.querySelector('.map-label-name').textContent === name
              && getComputedStyle(l.querySelector('.map-label-sub')).display === 'block')""",
                           arg=info["name"])

    # At zoom it shows outright: frame a small window and read the states' subtitles.
    page.mouse.move(5, 880)
    page.evaluate("MM.mapRenderer.setCamera({x: [-29, -17], y: [-3, 4]})")
    page.wait_for_function("""() => [...document.querySelectorAll('.map-label.show-sub')]
        .some(l => l.style.display !== 'none'
              && getComputedStyle(l.querySelector('.map-label-sub')).display === 'block'
              && l.querySelector('.map-label-sub').textContent.length > 3)""", timeout=15000)
    assert not page.js_errors, page.js_errors
