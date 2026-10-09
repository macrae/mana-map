"""The deck page's Known combos panel and the Game Changers by name (Area A5, 2026-10-09).

`deck-view.js` fetches `combos.json` beside the other per-deck files and renders
`combosPanel` after the bracket floor: the lines inside the list, then One card
short with the missing card linked to the Atlas and the uncapped total stated. The
bracket panel names its Game Changers rather than only counting them.

The combos fixture is ROUTED IN at the URL the page fetches, the way the card-flags
tests do, so the assertions do not move when `deck-combos --write` re-runs on a
refreshed Spellbook dump; the absent case routes a 404 and an info.json whose todo
names the stage, which is what `absent()` reads. The Game Changer names come from
the deck's own tracked `bracket_report.json` at test time, never hardcoded.
"""
import json

import pytest

from manamap.config import DECKS_DIR
from test_viz_deck_context import _manifest, _open

pytestmark = pytest.mark.browser

COMBOS = {
    "slug": "sharknado", "branch": None, "decklist_sha256": "0" * 64,
    "combo_data": {"source_timestamp": "2026-10-09T07:10:08+00:00", "combo_count": 114383},
    "summary": {"included": 2, "infinite": 1, "two_card_infinite": 1,
                "excluded_commander_assumption": 1, "near": 2, "near_total": 312,
                "highest_bracket": 3},
    "included": [
        {"id": "690-3966", "cards": ["Sanguine Bond", "Exquisite Blood"],
         "produces": ["Infinite lifeloss"], "infinite": True, "bracket": 3, "banned": False,
         "mana_value_needed": 0, "popularity": 148970, "assumes_other_commander": False},
        {"id": "1-2", "cards": ["Windfall", "Brallin, Skyshark Rider"],
         "produces": ["Damage"], "infinite": False, "bracket": None, "banned": True,
         "mana_value_needed": 0, "popularity": None, "assumes_other_commander": True},
    ],
    "near": [
        {"id": "3966-5755", "cards": ["Exquisite Blood", "Enduring Tenacity"],
         "missing": "Enduring Tenacity", "infinite": True, "bracket": 3,
         "mana_value_needed": 0, "popularity": 74756},
        {"id": "7-8", "cards": ["Windfall", "Thought Vessel", "Zur's Weirding"],
         "missing": "Zur's Weirding", "infinite": False, "bracket": 2,
         "mana_value_needed": 4, "popularity": 10},
    ],
}


def _route(page, slug, file, body, status=200):
    page.route(f"**/data/decks/{slug}/{file}*",
               lambda route: route.fulfill(status=status, content_type="application/json",
                                           body=body))


def _new_page(browser, viz_server, slug, routes):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    for file, body, status in routes:
        _route(page, slug, file, body, status)
    page.goto(f"{viz_server}/viz/deck.html?deck={slug}")
    page.wait_for_selector("#panels section", timeout=15000)
    return page, errors


def test_the_known_combos_panel_lists_the_lines_and_links_the_missing_card(browser, viz_server):
    page, errors = _new_page(browser, viz_server, "sharknado",
                             [("combos.json", json.dumps(COMBOS), 200)])
    try:
        panel = page.locator("#panel-combos")
        assert panel.count() == 1
        assert errors == []
        # Order: right after the bracket floor.
        ids = page.locator("#panels section").evaluate_all("els => els.map(e => e.id)")
        assert ids.index("panel-combos") == ids.index("panel-bracket") + 1

        deck = panel.locator(".combo-line.is-deck")
        assert deck.count() == 2
        first = deck.first
        assert first.get_attribute("data-id") == "690-3966"
        assert first.locator(".combo-partner").all_inner_texts() == ["Sanguine Bond", "Exquisite Blood"]
        assert first.locator(".inf-badge").inner_text() == "∞"
        assert first.locator(".bracket-pill").text_content() == "B3"
        assert first.locator("a.combo-link").get_attribute("href") == \
            "https://commanderspellbook.com/combo/690-3966/"
        second = deck.nth(1)
        assert second.locator(".bracket-pill.banned").text_content() == "banned"
        assert "assumes its own commander" in second.locator(".combo-note").inner_text()
        assert second.locator(".inf-badge").count() == 0
        assert "2 known lines — 1 infinite, 1 two-card" in panel.locator(".combo-head").first.text_content()

        near = panel.locator(".combo-line.is-near")
        assert near.count() == 2
        link = near.first.locator(".combo-partner.is-missing a.cardref")
        assert link.inner_text() == "Enduring Tenacity"
        assert link.get_attribute("href") == "index.html?cards=Enduring%20Tenacity"
        assert near.first.locator(".combo-partner.is-missing").count() == 1
        assert "showing 2 of 312" in panel.locator(".combo-head").nth(1).text_content()
        assert "2026-10-09" in panel.text_content()
        assert "1 line(s) assume a commander this deck does not run" in panel.text_content()
        assert errors == []
    finally:
        page.close()


def test_without_combos_json_the_panel_is_the_call_to_action(browser, viz_server):
    """Absent, never silent — the page hands over the command, as it does for every
    stage `deck_status` knows. The todo is routed in with the 404 so the test does
    not depend on which tracked deck happens to lack the artifact today."""
    slug = "sharknado"
    info = json.loads((DECKS_DIR / slug / "info.json").read_text(encoding="utf-8"))
    status = info.setdefault("status", {})
    todo = [t for t in (status.get("todo") or []) if t.get("stage") != "combos"]
    todo.insert(0, {"stage": "combos", "what": "known lines and near misses — `deck-combos --write`",
                    "how": f"manamap pilot deck-combos {slug} --write"})
    status["todo"] = todo
    page, errors = _new_page(browser, viz_server, slug,
                             [("combos.json", "not found", 404),
                              ("info.json", json.dumps(info), 200)])
    try:
        panel = page.locator("#panel-combos")
        assert panel.count() == 1
        assert panel.locator(".combo-line").count() == 0
        assert panel.locator(".todo-cmd code").inner_text() == \
            f"manamap pilot deck-combos {slug} --write"
        assert errors == []
    finally:
        page.close()


def test_the_bracket_panel_names_its_game_changers(browser, viz_server):
    slug = next(d["slug"] for d in _manifest()
                if (d.get("has") or {}).get("bracket_report")
                and json.loads((DECKS_DIR / d["slug"] / "bracket_report.json")
                               .read_text(encoding="utf-8")).get("game_changers"))
    report = json.loads((DECKS_DIR / slug / "bracket_report.json").read_text(encoding="utf-8"))
    page, errors = _open(browser, viz_server, slug)
    try:
        pills = page.locator("#panel-bracket .gc-pill")
        assert pills.all_inner_texts() == report["game_changers"]
        assert len(report["game_changers"]) >= 1
        assert errors == []
    finally:
        page.close()
