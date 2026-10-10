"""Moxfield on the pages (Area D, 2026-10-09): out by paste, a link by hand, in by paste.

Moxfield has no API and 403s every server-side request (docs/integrations.md), so
the pages never talk to it except ONE opportunistic browser fetch in Discover's
import box — and every test here routes that host, so the suite never does either.

Three surfaces:

- **The deck page's "Copy for Moxfield"** copies `deck-export <slug> --format
  moxfield`'s text. With a local API it runs that command through `cli`; on the
  static site it renders the deck's `decklist.txt` through `Decklist.parse(…,
  {printings: true})` + `Decklist.render`, the browser mirror of
  `check_in.render_decklist`. The mirror is held to the CLI's BYTES on every
  tracked deck — the production function on both sides, nothing re-derived here.
- **A recorded link** (`links.moxfield` on a manifest entry) shows as "Moxfield ↗"
  on the cover and on the workbench card. No deck has a `links.json` yet, so the
  manifest is routed.
- **Discover's import box** tries a pasted Moxfield deck URL once, browser-side,
  and on any failure says how to paste the list instead.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from conftest_viz import BOOT_TIMEOUT_MS, _record, discover_page  # noqa: F401  (fixtures)
from manamap.pilot import deck_export

ROOT = pathlib.Path(__file__).resolve().parent.parent
DECKS = ROOT / "data" / "decks"
TRACKED = sorted(p.parent.name for p in DECKS.glob("*/decklist.txt"))
MOX_URL = "https://moxfield.com/decks/AbC123xyz"
NEW_DECK = "https://moxfield.com/decks/personal"
BLOCKED = ("Moxfield blocks direct reads. Open the deck on Moxfield, "
           "Export → Copy for Moxfield, and paste the text here.")

_STUBS = """
window.__opened = null; window.__copied = null;
window.open = function (u, t, f) { window.__opened = [u, t, f]; return null; };
Object.defineProperty(navigator, 'clipboard', {
  configurable: true,
  value: { writeText: function (t) { window.__copied = t; return Promise.resolve(); } }
});
"""


def _cli_text(slug):
    return deck_export.export(slug, fmt="moxfield")[0]


def _errors(page):
    errors: list[str] = []
    add = _record(errors)
    page.on("pageerror", lambda e: add(e))
    page.on("console", lambda m: add(m.text) if m.type == "error" else None)
    page.js_errors = errors
    return errors


def _manifest_with_link(slug, url=MOX_URL):
    doc = json.loads((DECKS / "index.json").read_text(encoding="utf-8"))
    hit = 0
    for e in doc["decks"]:
        if e["slug"] == slug:
            e["links"] = {"moxfield": {"url": url, "id": url.rsplit("/", 1)[-1],
                                       "as_of": "2026-10-09"}}
            hit += 1
    assert hit == 1, f"{slug} is not in the manifest"
    return doc


def _route_manifest(page, doc):
    page.route("**/data/decks/index.json*",
               lambda r: r.fulfill(status=200, content_type="application/json",
                                   body=json.dumps(doc)))


def _deck_page(browser, viz_server, slug, manifest=None, api_calls=None, cli_stdout=None):
    page = browser.new_page(viewport={"width": 1440, "height": 1200})
    _errors(page)
    page.add_init_script(_STUBS)
    if manifest is not None:
        _route_manifest(page, manifest)
    if api_calls is not None:
        def _json(route, body):
            route.fulfill(status=200, content_type="application/json", body=json.dumps(body))
        page.route("**/api/page/state", lambda r: _json(r, {"ok": True, "result": {}}))
        page.route("**/api/health",
                   lambda r: _json(r, {"ok": True, "command": "health",
                                       "result": {"ok": True, "api": 1}}))
        page.route("**/api/", lambda r: _json(r, {"commands": ["cli", "health"]}))
        page.route("**/api/deck/measures",
                   lambda r: _json(r, {"ok": True, "result": {"stages": {}, "drafts": []}}))

        def cli(route):
            api_calls.append(route.request.post_data_json)
            _json(route, {"ok": True, "command": "cli",
                          "result": {"stdout": cli_stdout, "exit": 0}})
        page.route("**/api/cli", cli)
    page.goto(f"{viz_server}/viz/deck.html?deck={slug}")
    page.wait_for_selector('[data-act="moxfield-copy"]', timeout=BOOT_TIMEOUT_MS)
    return page


# ── the static export IS the CLI's export ────────────────────────────────────

@pytest.mark.browser
def test_the_browser_export_is_deck_export_for_every_tracked_deck(browser, viz_server):
    """`Decklist.render(Decklist.parse(decklist.txt, {printings: true}))` against
    `deck_export.export(slug, 'moxfield')`, byte for byte, on every tracked deck —
    sharknado (no printings, two commanders) and elves (a 60 with printings, no
    commander) by name among them."""
    assert {"sharknado", "elves"} <= set(TRACKED)
    page = browser.new_page()
    errors = _errors(page)
    page.goto(f"{viz_server}/tests/fixtures/parser_page.html")
    page.wait_for_function("() => !!window.Decklist", timeout=BOOT_TIMEOUT_MS)
    checked = 0
    try:
        for slug in TRACKED:
            text = (DECKS / slug / "decklist.txt").read_text(encoding="utf-8")
            got = page.evaluate(
                "t => Decklist.render(Decklist.parse(t, {printings: true}))", text)
            assert got == _cli_text(slug), slug
            checked += 1
        assert checked >= 10, f"only {checked} decks checked"
        assert errors == []
    finally:
        page.close()


@pytest.mark.browser
def test_the_mirror_writes_what_render_line_writes(browser, viz_server):
    """The line form's corners no tracked deck exercises today: an etched marker
    reads as foil, a commander in the sideboard keeps `*CMDR*`, and a sideboard
    section appears only when there is one — each against `check_in` itself."""
    from manamap.pilot import check_in
    from manamap.pilot.fetch_deck import parse_decklist

    text = ("Commander:\n1 Zur the Enchanter (CSP) 140 *F*\n\nDeck:\n"
            "1 Sol Ring (C21) 263 *E*\n2 Æther Vial\n1 Arcane Signet\n\n"
            "Sideboard:\n1 Yawgmoth, Thran Physician *CMDR*\n3 Duress (M19) 94\n")
    page = browser.new_page()
    errors = _errors(page)
    page.goto(f"{viz_server}/tests/fixtures/parser_page.html")
    page.wait_for_function("() => !!window.Decklist", timeout=BOOT_TIMEOUT_MS)
    try:
        got = page.evaluate("t => Decklist.render(Decklist.parse(t, {printings: true}))", text)
        assert got == check_in.render_decklist(parse_decklist(text))
        assert "1 Yawgmoth, Thran Physician *CMDR*" in got
        assert "1 Sol Ring (C21) 263 *F*" in got
        # Without the option the parse is the contract projection it always was.
        plain = page.evaluate("t => Decklist.parse(t)", text)
        assert all(set(e) == {"name", "quantity", "is_commander", "board"} for e in plain)
        assert errors == []
    finally:
        page.close()


# ── the deck page ────────────────────────────────────────────────────────────

@pytest.mark.browser
@pytest.mark.parametrize("slug", ["sharknado", "elves"])
def test_copy_for_moxfield_on_the_static_site(browser, viz_server, slug):
    """No server: the button copies the CLI's text, rendered in the browser, and
    opens Moxfield's new-deck page — the deck has no recorded link — with the
    paste hint beside it."""
    page = _deck_page(browser, viz_server, slug)
    try:
        assert "paste into Moxfield’s import box" in page.inner_text(".cov-export")
        assert page.query_selector("a.cov-moxfield") is None, "no link recorded"
        page.click('[data-act="moxfield-copy"]')
        page.wait_for_function("() => window.__opened !== null", timeout=BOOT_TIMEOUT_MS)
        assert page.evaluate("() => window.__copied") == _cli_text(slug)
        assert page.evaluate("() => window.__opened") == [NEW_DECK, "_blank", "noopener"]
        assert page.js_errors == [], page.js_errors
    finally:
        page.close()


@pytest.mark.browser
def test_copy_for_moxfield_asks_the_local_server(browser, viz_server):
    """With a local API the text is `deck-export`'s own stdout, asked for through
    the read-only `cli` endpoint with exactly the terminal's argv."""
    calls: list = []
    stdout = "Deck:\n1 Sol Ring\n"
    page = _deck_page(browser, viz_server, "sharknado", api_calls=calls, cli_stdout=stdout)
    try:
        page.wait_for_function("() => Api.ready && Api.has('cli')", timeout=BOOT_TIMEOUT_MS)
        page.click('[data-act="moxfield-copy"]')
        page.wait_for_function("() => window.__opened !== null", timeout=BOOT_TIMEOUT_MS)
        assert calls == [{"argv": ["deck-export", "sharknado", "--format", "moxfield"]}]
        assert page.evaluate("() => window.__copied") == stdout
        assert page.evaluate("() => window.__opened")[0] == NEW_DECK
        assert page.js_errors == [], page.js_errors
    finally:
        page.close()


@pytest.mark.browser
def test_a_recorded_link_shows_on_the_cover_and_is_where_copy_opens(browser, viz_server):
    page = _deck_page(browser, viz_server, "sharknado",
                      manifest=_manifest_with_link("sharknado"))
    try:
        a = page.wait_for_selector("#panels a.cov-moxfield", timeout=BOOT_TIMEOUT_MS)
        assert a.inner_text() == "Moxfield ↗"
        assert a.get_attribute("href") == MOX_URL
        assert a.get_attribute("target") == "_blank"
        assert "noopener" in a.get_attribute("rel")
        page.click('[data-act="moxfield-copy"]')
        page.wait_for_function("() => window.__opened !== null", timeout=BOOT_TIMEOUT_MS)
        assert page.evaluate("() => window.__opened")[0] == MOX_URL
        assert page.js_errors == [], page.js_errors
    finally:
        page.close()


@pytest.mark.browser
def test_a_link_off_moxfield_is_not_rendered(browser, viz_server):
    """The href is the one place a bad string becomes a live link, so the page
    checks the host itself rather than trusting the file — and Copy falls back
    to the new-deck page."""
    page = _deck_page(browser, viz_server, "sharknado",
                      manifest=_manifest_with_link("sharknado", "javascript:alert(1)"))
    try:
        assert page.query_selector("a.cov-moxfield") is None
        page.click('[data-act="moxfield-copy"]')
        page.wait_for_function("() => window.__opened !== null", timeout=BOOT_TIMEOUT_MS)
        assert page.evaluate("() => window.__opened")[0] == NEW_DECK
        assert page.js_errors == [], page.js_errors
    finally:
        page.close()


# ── the workbench ────────────────────────────────────────────────────────────

@pytest.mark.browser
def test_the_workbench_card_links_to_moxfield(browser, viz_server):
    page = browser.new_page(viewport={"width": 1440, "height": 1200})
    errors = _errors(page)
    _route_manifest(page, _manifest_with_link("sharknado"))
    page.goto(f"{viz_server}/viz/workbench.html")
    try:
        page.wait_for_selector(".wb-card a.wb-title[href='deck.html?deck=sharknado']",
                               state="attached", timeout=BOOT_TIMEOUT_MS)
        got = page.evaluate("""() => {
            const card = document.querySelector(
              ".wb-card a.wb-title[href='deck.html?deck=sharknado']").closest('.wb-card');
            const a = card.querySelector('a.wb-moxfield');
            return { text: a && a.textContent, href: a && a.getAttribute('href'),
                     others: document.querySelectorAll('a.wb-moxfield').length };
        }""")
        assert got == {"text": "Moxfield ↗", "href": MOX_URL, "others": 1}, got
        assert errors == []
    finally:
        page.close()


# ── Discover's import box ────────────────────────────────────────────────────

_MOX_DOC = {
    "name": "fixture",
    "commanders": {"Zur the Enchanter": {"quantity": 1,
                                         "card": {"name": "Zur the Enchanter",
                                                  "set": "csp", "cn": "140"}}},
    "mainboard": {"Sol Ring": {"quantity": 1, "card": {"name": "Sol Ring"}},
                  "Rhystic Study": {"quantity": 1, "card": {"set": "pcy", "cn": "45"}}},
    "sideboard": {"Duress": {"quantity": 1, "card": {"name": "Duress"}}},
}
_CORS = {"Access-Control-Allow-Origin": "*"}


def _paste(page, text):
    page.wait_for_selector("#dcImport", state="attached", timeout=BOOT_TIMEOUT_MS)
    return page.evaluate("""async t => {
        const before = Discovery.library.names().length;
        document.getElementById('dcImport').value = t;
        const res = await Discovery.onImport();
        return { before, after: Discovery.library.names().length,
                 names: Discovery.library.names(), res: res || null,
                 note: document.getElementById('dcImportNote').textContent };
    }""", text)


@pytest.mark.browser
@pytest.mark.parametrize("how", ["403", "abort", "not-json"])
def test_a_moxfield_url_moxfield_refuses_says_how_to_paste(discover_page, how):
    page = discover_page
    seen: list[str] = []

    def refuse(route):
        seen.append(route.request.url)
        if how == "403":
            route.fulfill(status=403, headers=_CORS, content_type="text/html", body="no")
        elif how == "abort":
            route.abort()
        else:
            route.fulfill(status=200, headers=_CORS, content_type="text/html",
                          body="<html>challenge</html>")
    page.route("https://api2.moxfield.com/**", refuse)
    got = _paste(page, "https://www.moxfield.com/decks/AbC123xyz")
    assert seen == ["https://api2.moxfield.com/v2/decks/all/AbC123xyz"], "one try, no retry"
    assert got["note"] == BLOCKED
    assert got["after"] == got["before"], "nothing imported"
    assert page.js_errors == [], page.js_errors


@pytest.mark.browser
def test_a_moxfield_url_moxfield_answers_imports_the_deck(discover_page):
    page = discover_page
    page.route("https://api2.moxfield.com/**",
               lambda r: r.fulfill(status=200, headers=_CORS,
                                   content_type="application/json",
                                   body=json.dumps(_MOX_DOC)))
    got = _paste(page, "https://moxfield.com/decks/AbC123xyz")
    assert got["note"] == ""
    assert got["res"]["missing"] == [], got["res"]
    assert got["res"]["resolved"] == 3, "commander + the two mainboard cards"
    assert {"Zur the Enchanter", "Sol Ring", "Rhystic Study"} <= set(got["names"])
    assert "Duress" not in got["names"], "the sideboard is not the deck"
    assert page.evaluate("() => Discovery.index[Session.commander].n") == "Zur the Enchanter"
    assert page.js_errors == [], page.js_errors


@pytest.mark.browser
def test_a_plain_list_never_touches_moxfield(discover_page):
    page = discover_page
    seen: list[str] = []
    page.route("https://api2.moxfield.com/**",
               lambda r: (seen.append(r.request.url), r.abort()))
    got = _paste(page, "1 Sol Ring\n1 Rhystic Study")
    assert seen == []
    assert got["res"]["resolved"] == 2
    assert page.js_errors == [], page.js_errors
