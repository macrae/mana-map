"""The Atlas's CARD DETAIL PANEL: text first, one place per fact, and the deck in Build.

An audit on 2026-10-08 (1440x900) measured the panel: header 81px, then a 443px card
image, so on a long card the oracle began at the fold — while the image only repeated
the name, cost, type and text. Keywords repeated the oracle, CMC repeated the header,
every format had its own badge, obsolescence appeared twice ("Compare with" and
"Outclassed by" read the same index), a DFC's faces were joined unlabelled, and Build
with a deck loaded said nothing about the deck. `buildCardDetailHtml` now emits:
deck context (Build only) · type + oracle · relations + the comparison + Keep · the
image · one facts line.

Every interaction here is a REAL click. **Nothing may reach a real `serve`**: the
Build tests install the review grid's own `_mock_api` (`watch/mark` mocked, every other
`/api/` call refused), so the tracked `data/decks/sharknado/watchlist.json` is never
written. Scryfall is routed too, so no test depends on the network or its rate limit.
"""

from __future__ import annotations

import base64
import json

import pytest

from conftest_viz import page  # noqa: F401  (fixture)
from manamap.config import DATA_DIR
from test_viz_review_grid import _mock_api, _open_grid, _release

pytestmark = pytest.mark.browser

# A 1x1 PNG: Scryfall stands in for itself without the network.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")


def _route_scryfall(page, refuse_back=False):
    """Every card image answers with a pixel; `face=back` is refused (422, as Scryfall
    does for a card with no printed back) when `refuse_back`."""
    seen: list[str] = []

    def answer(route):
        url = route.request.url
        seen.append(url)
        if refuse_back and "face=back" in url:
            route.fulfill(status=422, content_type="application/json", body="{}")
        else:
            route.fulfill(status=200, content_type="image/png", body=_PNG)

    page.route("**/api.scryfall.com/**", answer)
    return seen


def _open(page, name):
    """Select a card on the map and wait for its panel body."""
    page.evaluate("n => { MM.closeDetail(); MM.selectByName(n); }", name)
    page.wait_for_function(
        """n => { const h = document.querySelector('#detailInner h2');
                  return h && h.textContent === n
                      && document.querySelector('#detailInner .detail-card-image'); }""",
        arg=name, timeout=10000)


def _geometry(page):
    return page.evaluate("""() => {
        const inner = document.getElementById('detailInner');
        const top = inner.getBoundingClientRect().top;
        const y = s => { const e = inner.querySelector(s);
                         return e ? Math.round(e.getBoundingClientRect().top - top) : null; };
        return {oracle: y('.detail-oracle'), image: y('.detail-card-image'),
                relations: y('.discover-relations'), facts: y('.detail-facts')};
    }""")


def test_the_oracle_comes_before_the_image(page):
    """On a long card the oracle used to start at the fold, under a 443px image.
    It must lead now: above the image and within the panel's first ~250px.
    Fails on the pre-rework `mana-map.js`, where the image is the first thing drawn."""
    _route_scryfall(page)
    _open(page, "The Elder Dragon War")
    g = _geometry(page)
    assert g["oracle"] is not None and g["image"] is not None, g
    assert g["oracle"] < g["image"], f"the image leads the text again: {g}"
    assert g["oracle"] <= 250, f"the oracle starts {g['oracle']}px down the panel: {g}"
    assert g["oracle"] < g["relations"] < g["image"] < g["facts"], g
    text = page.inner_text("#detailInner")
    assert "Read ahead" in text, "the oracle is not the card's"
    assert page.js_errors == []


def test_legality_is_one_commander_line(page):
    """Eight format badges for a Commander workbench became one line; the rest fold.
    The CMC row (the cost is in the header) and the Keywords section (they repeat the
    oracle) are gone, but keywords stay searchable."""
    _route_scryfall(page)
    _open(page, "The Elder Dragon War")
    r = page.evaluate("""() => {
        const inner = document.getElementById('detailInner');
        const facts = inner.querySelector('.detail-facts');
        const more = inner.querySelector('details.detail-formats-more');
        const badges = [...inner.querySelectorAll('.format-badge')];
        return {facts: facts ? facts.textContent : null,
                moreOpen: more ? more.open : null,
                badges: badges.length,
                badgesFolded: badges.every(b => b.closest('details.detail-formats-more')),
                visible: inner.innerText};
    }""")
    assert r["facts"] is not None
    assert r["facts"].count("Commander:") == 1 and "Commander: legal" in r["facts"], r["facts"]
    assert "EDHREC #9,493" in r["facts"], r["facts"]
    assert "Identity R" in r["facts"], r["facts"]
    assert r["moreOpen"] is False, "the other formats are not collapsed"
    assert r["badges"] == 7 and r["badgesFolded"], r
    assert "FORMAT LEGALITY" not in r["visible"].upper()
    assert "CMC:" not in r["visible"]
    assert "KEYWORDS" not in r["visible"].upper()
    # The keyword data is still searchable: 'read ahead' is a keyword on this card.
    assert page.evaluate("() => MM.allData.find(d => d.n === 'The Elder Dragon War').k") == "Read Ahead"
    assert page.js_errors == []


def test_obsolescence_appears_once_under_its_button(page):
    """"Compare with" and "Outclassed by" read the same index 115px apart. Now the
    comparison is the button's own detail: one section, collapsed, directly under the
    relation row, with the strength / gains / costs only it carries."""
    _route_scryfall(page)
    _open(page, "The Elder Dragon War")
    page.wait_for_function("() => !document.querySelector('#detailInner .obsolescence-placeholder')",
                           timeout=10000)
    r = page.evaluate("""() => {
        const inner = document.getElementById('detailInner');
        const secs = [...inner.querySelectorAll('.obsolescence-section')];
        const rel = inner.querySelector('.discover-relations');
        const btn = [...inner.querySelectorAll('.discover-rel')]
            .find(b => b.textContent.startsWith('Outclassed by'));
        const s = secs[0];
        return {n: secs.length, tag: s && s.tagName, open: s && s.open,
                summary: s && s.querySelector('summary').textContent,
                items: s ? s.querySelectorAll('.obsolescence-item').length : 0,
                strength: !!(s && s.querySelector('.obsolescence-strength')),
                button: btn ? btn.querySelector('.discover-count').textContent : null,
                follows: s && rel ? s.previousElementSibling === rel : false,
                html: inner.innerHTML};
    }""")
    assert r["n"] == 1 and r["tag"] == "DETAILS" and r["open"] is False, r
    assert r["follows"], "the comparison is not directly under the relation buttons"
    assert r["button"] == "2" and r["items"] == 2, r
    assert "Compare the 2" in r["summary"]
    assert r["strength"], "the strength the comparison uniquely carries is gone"
    assert "Compare with" not in r["html"], "the separate Compare-with box is back"
    # It opens with a real click on its summary.
    page.click("#detailInner .obsolescence-section > summary")
    assert page.evaluate("() => document.querySelector('#detailInner .obsolescence-section').open")
    assert "The Misty Mountains Cold" in page.inner_text("#detailInner .obsolescence-section")
    assert page.js_errors == []


def test_a_dfc_labels_both_faces_and_flips(page):
    """Faces were joined with an unlabelled <br><br> and only the front was drawn."""
    seen = _route_scryfall(page)
    _open(page, "Delver of Secrets // Insectile Aberration")
    faces = page.eval_on_selector_all(
        "#detailInner .detail-face",
        "els => els.map(e => ({label: e.querySelector('.detail-face-label').textContent.trim(),"
        " type: (e.querySelector('.detail-type') || {}).textContent,"
        " oracle: (e.querySelector('.detail-oracle') || {}).textContent}))")
    assert [f["label"] for f in faces] == ["Front — Delver of Secrets",
                                           "Back — Insectile Aberration"], faces
    assert faces[0]["type"] == "Creature — Human Wizard"
    assert faces[1]["type"] == "Creature — Human Insect"
    assert "transform this creature" in faces[0]["oracle"]
    assert faces[1]["oracle"] == "Flying"
    page.click("#detailInner .detail-flip")
    page.wait_for_function(
        "() => document.querySelector('#detailInner .detail-card-image img').src.includes('face=back')",
        timeout=5000)
    assert any("face=back" in u and "Delver%20of%20Secrets" in u for u in seen), seen
    assert "Front — Delver of Secrets" in page.inner_text("#detailInner .detail-flip")
    # Click the image: it toggles full width, and back.
    w0 = page.evaluate("() => document.querySelector('#detailInner .detail-card-image img').getBoundingClientRect().width")
    page.click("#detailInner .detail-card-image img")
    w1 = page.evaluate("() => document.querySelector('#detailInner .detail-card-image img').getBoundingClientRect().width")
    assert 235 <= w0 <= 245 and w1 > w0 + 40, (w0, w1)
    assert page.js_errors == []


def test_a_card_with_no_back_image_says_so(page):
    """A split card prints both halves on one face: Scryfall refuses `face=back`, and
    the panel puts the front back and retires the flip instead of showing a hole."""
    _route_scryfall(page, refuse_back=True)
    _open(page, "Fire // Ice")
    labels = page.eval_on_selector_all("#detailInner .detail-face-label",
                                       "els => els.map(e => e.textContent.trim())")
    assert labels[0].startswith("Front — Fire") and labels[1].startswith("Back — Ice"), labels
    page.click("#detailInner .detail-flip")
    page.wait_for_function("() => document.querySelector('#detailInner .detail-flip').disabled",
                           timeout=5000)
    src = page.evaluate("() => document.querySelector('#detailInner .detail-card-image img').src")
    assert "face=back" not in src
    assert "both faces" in page.inner_text("#detailInner .detail-flip")
    # A refused image is not a page error; anything else would be.
    assert [e for e in page.js_errors if "422" not in e] == []


def _watch_row(name):
    doc = json.loads((DATA_DIR / "decks" / "sharknado" / "watchlist.json").read_text())
    for s in doc["sets"]:
        for c in s["cards"]:
            if c["name"] == name:
                return s, c
    raise AssertionError(f"{name} is not in sharknado's watch list")


def test_build_panel_watch_posts_once_through_the_grid_queue(page):
    """A card from the loaded watch set shows its why, axis, pays and verdict, and its
    Watch button goes through the grid's one guarded write path: a second click while
    the first is on the wire is dropped, and the grid's tile follows the panel."""
    name = "The Elder Dragon War"
    wset, row = _watch_row(name)
    _route_scryfall(page)
    posted, held = _mock_api(page, hold=True)
    _open_grid(page)
    page.click("#candGrid .cg-filter[data-filter='all']")
    _open(page, name)
    ctx = "#detailInner .deck-ctx"
    page.wait_for_selector(ctx, timeout=5000)
    text = page.inner_text(ctx)
    assert row["why"] in text, text
    assert row["axis"].upper() in text.upper() and "pays both" in text, text
    assert "unreviewed" in text, text
    assert "Not in the 99" in text

    page.click(f"{ctx} .deck-ctx-act[data-verdict='watching']")
    page.wait_for_function(f"() => document.querySelector('{ctx} .deck-ctx-verdict')"
                           ".textContent === 'saving…'", timeout=5000)
    page.click(f"{ctx} .deck-ctx-act[data-verdict='watching']")
    page.wait_for_timeout(400)
    assert len(posted) == 1, f"one card, one pending write — got {posted}"
    assert posted[0] == {"slug": "sharknado", "set": wset["id"], "card": name,
                         "verdict": "watching"}, posted
    _release(held)
    tile = f"#candGrid .cg-tile[data-card='{name}']"
    page.wait_for_function(f"() => document.querySelector(\"{tile}\")"
                           ".classList.contains('v-watching')", timeout=5000)
    page.wait_for_function(f"() => document.querySelector('{ctx} .deck-ctx-verdict')"
                           ".textContent === '★ watching'", timeout=5000)
    assert len(posted) == 1
    assert "★ watching" in page.text_content(tile)
    assert page.js_errors == []


def test_build_panel_a_card_in_the_99_shows_its_roles(page):
    """In the 99, its roles as the role grouping assigns them, and colour identity
    against BOTH partner commanders (Shabraz WU + Brallin R)."""
    name = "Arcane Signet"
    roles = json.loads((DATA_DIR / "card_roles.json").read_text())["roles"][name]
    _route_scryfall(page)
    _mock_api(page)
    _open_grid(page)
    _open(page, name)
    ctx = "#detailInner .deck-ctx"
    page.wait_for_selector(ctx, timeout=5000)
    text = page.inner_text(ctx)
    assert "In the 99" in text and "Not in the 99" not in text, text
    assert "SHARKNADO" in text.upper()
    chips = page.eval_on_selector_all(f"{ctx} .deck-ctx-role", "els => els.map(e => e.textContent)")
    assert sorted(c.replace(": ", ":") for c in chips) == sorted(roles), (chips, roles)
    assert "fits WUR" in text, text
    assert page.query_selector(f"{ctx} .deck-ctx-watch") is None, "not on the watch list"
    # A red card is IN identity: the partner fix. Before it, only Shabraz's WU counted.
    assert page.evaluate("() => Build.cardContext(MM.allData.findIndex(d => d.n === "
                         "'Lightning Bolt')).colour.off") == []
    assert page.js_errors == []


def test_no_deck_block_outside_build(page):
    """Explore has no deck: no block, whatever Build last had open."""
    _route_scryfall(page)
    _open(page, "Arcane Signet")
    assert page.query_selector("#detailInner .deck-ctx") is None
    # Build with no deck loaded: still no block.
    page.evaluate("""() => { document.getElementById('modeSelect').value = 'build';
                             MM.setMode('build'); }""")
    page.wait_for_function("() => MM.mode === 'build'", timeout=10000)
    assert page.evaluate("() => Build.deckSlug") is None
    _open(page, "Arcane Signet")
    assert page.query_selector("#detailInner .deck-ctx") is None
    assert page.js_errors == []
