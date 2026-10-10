"""The deck pickers offer only the decks you can WORK ON (2026-10-09).

The pilot's words: "in the Build section, when choosing a deck, exclude any
archived decks — just sleeved, on the bench and brewing!" Both pickers — Build's
`#deckLensSelect` and Discover's "Load one of my decks…" `#dcDeck` — read one
predicate, `isWorkableDeck` (api.js), over the manifest's `stage`
(`promote.stage`): "dev" | "bench" | "sleeved", or null for a deck in a pile.

The manifest is ROUTED, one deck per state, so the test does not depend on which
of the real fleet happens to be broken down this week. Every other file (cards,
info, stacks) is the real deck's own, so the pages load exactly as they would.
A superseded deck is built from a real slug with its status rewritten: no deck
in the fleet is superseded today, and that state must be filtered all the same.

Proven by re-introduction: with the filter removed from `deckPickerOptions`
(build.js) the Build assertions fail on the two archived slugs.
"""

from __future__ import annotations

import json

import pytest

from conftest_viz import BOOT_TIMEOUT_MS, _record, serve_card_images_locally
from manamap.config import DECKS_DIR

pytestmark = pytest.mark.browser

# One real slug per state; the state is what the routed manifest says.
STATES = {
    "emiel-blink": ("dev", None),
    "meren-recursion": ("bench", None),
    "heliod": ("sleeved", None),
    "hapatra": (None, ["broken-down", "BROKEN DOWN FOR PARTS", "Pulled for parts."]),
    "sisay": (None, ["superseded", "SUPERSEDED", "Replaced by a newer list."]),
}
WORKABLE = {"Brewing": ["emiel-blink"], "On the bench": ["meren-recursion"],
            "Sleeved": ["heliod"]}
ARCHIVED = "hapatra"


def _manifest():
    doc = json.loads((DECKS_DIR / "index.json").read_text(encoding="utf-8"))
    by = {d["slug"]: d for d in doc["decks"]}
    decks = []
    for slug, (stage, status) in STATES.items():
        e = dict(by[slug])
        e["stage"], e["status"] = stage, status
        e["locked"] = stage == "sleeved"
        decks.append(e)
    return json.dumps({"drafts": [], "decks": decks})


def _open(browser, viz_server, query):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list[str] = []
    add = _record(errors)
    page.on("pageerror", lambda e: add(e))
    page.on("console", lambda m: add(m.text) if m.type == "error" else None)
    body = _manifest()
    page.route("**/data/decks/index.json*",
               lambda route: route.fulfill(status=200, content_type="application/json",
                                           body=body))
    serve_card_images_locally(page)
    page.goto(f"{viz_server}/viz/index.html{query}")
    page.js_errors = errors
    return page


def _groups(page, select):
    """{optgroup label: [option values]} plus the loose options, in order."""
    return page.evaluate("""sel => {
        const el = document.querySelector(sel);
        const groups = {};
        el.querySelectorAll('optgroup').forEach(g => {
            groups[g.label] = [...g.querySelectorAll('option')].map(o => o.value);
        });
        const loose = [...el.children].filter(c => c.tagName === 'OPTION')
            .map(o => ({value: o.value, text: o.textContent,
                        disabled: o.disabled, selected: o.selected}));
        return {groups, order: [...el.querySelectorAll('optgroup')].map(g => g.label), loose};
    }""", select)


def _deck_groups(got):
    """Only the deck rungs — Build also carries an "In progress" drafts group."""
    return {k: v for k, v in got["groups"].items() if k in WORKABLE}


def test_build_lists_only_workable_decks_grouped_by_rung(browser, viz_server):
    # Entered the way the mode select does — `?mode=build` alone sets the mode
    # without entering it (only `?deck=`/`?draft=` call `setMode('build')`).
    page = _open(browser, viz_server, "?mode=explore")
    try:
        page.wait_for_function("() => window.MM && MM.allData && MM.allData.length > 0",
                               timeout=BOOT_TIMEOUT_MS)
        page.evaluate("""() => { document.getElementById('modeSelect').value = 'build';
                                 MM.setMode('build'); }""")
        page.wait_for_function(
            "() => document.querySelector('#deckLensSelect optgroup')",
            timeout=BOOT_TIMEOUT_MS)
        got = _groups(page, "#deckLensSelect")
        assert _deck_groups(got) == WORKABLE, got
        assert [g for g in got["order"] if g in WORKABLE] == list(WORKABLE), got["order"]
        values = [o["value"] for o in got["loose"]] + \
            [v for vs in got["groups"].values() for v in vs]
        assert "hapatra" not in values and "sisay" not in values, values
        assert page.js_errors == []
    finally:
        page.close()


def test_discover_lists_the_same_three_decks(browser, viz_server):
    page = _open(browser, viz_server, "?card=Craterhoof%20Behemoth")
    try:
        page.wait_for_function(
            "() => window.Discovery && Discovery.isReady() && "
            "document.querySelector('#dcDeck optgroup')", timeout=BOOT_TIMEOUT_MS)
        got = _groups(page, "#dcDeck")
        assert got["groups"] == WORKABLE, got
        assert got["order"] == list(WORKABLE), got["order"]
        assert [o["value"] for o in got["loose"]] == [""], got["loose"]
        assert page.js_errors == []
    finally:
        page.close()


def test_an_archived_deck_still_opens_by_link_read_only(browser, viz_server):
    """`?deck=<slug>` is an inbound contract from every published page."""
    page = _open(browser, viz_server, f"?mode=build&deck={ARCHIVED}")
    try:
        page.wait_for_function(
            "s => window.Build && Build.deckSlug === s && "
            "document.querySelector('#deckLensSelect option[disabled]')",
            arg=ARCHIVED, timeout=BOOT_TIMEOUT_MS)
        got = _groups(page, "#deckLensSelect")
        held = [o for o in got["loose"] if o["disabled"]]
        assert len(held) == 1, got["loose"]
        assert held[0]["value"] == ARCHIVED and held[0]["selected"], held
        assert held[0]["text"].endswith("— archived, read only"), held
        # It is shown, not offered: the rungs are still exactly the three.
        assert _deck_groups(got) == WORKABLE, got
        assert page.js_errors == []
    finally:
        page.close()
