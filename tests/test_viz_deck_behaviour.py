"""Browser behaviour: the deck page (viz/deck.html).

THE BEHAVIOUR CORE (2026-11-22). Split from `test_viz_behaviour.py` (272 tests) and cut
to what a page must DO: boot without a JS error, read and write the right files,
land deep links, and show exactly what the artifacts hold. Wording, layout, pixels,
timing and JS source text are not tested here — the pages change fast and those
tests broke on every change. A new test asserts a behaviour, never a phrase.
"""
from __future__ import annotations
import base64
import json
import pathlib
import re
import pytest
from conftest import A_BRANCH, requires_branch, requires_deck
from manamap.config import SYNERGY_RULES
from conftest_viz import await_projection  # noqa: F401
from conftest_viz import (  # noqa: F401
    BOOT_TIMEOUT_MS, _record, canvas_page, corpus_count, discover_page, page,
    idle, serve_watchlist_fixture, still_page,
)
pytestmark = pytest.mark.browser


def _dossier(browser, viz_server, slug):
    page = browser.new_page()
    page.goto(f"{viz_server}/viz/deck.html?deck={slug}")
    idle(page)
    return page


def test_the_dossier_offers_no_button_it_cannot_press(browser, viz_server):
    """THE DEPLOYED SHAPE, asserted from the outside.

    `viz_server` is a static file server with no `/api` — which is exactly
    GitHub Pages. `Api.probe()` gets a 404, and a 404 is the deployed SHAPE
    rather than a malfunction, so the page must render its static half: the
    absent sections still say what they are and still print the command, and
    there is no Measure button, because pressing one would do nothing.

    The PRD's words for this: agent-dependent affordances are ABSENT, not
    broken. This is where that is checked for the dossier.
    """
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        page.goto(f"{viz_server}/viz/deck.html?deck=zur-enchantress")
        page.wait_for_function("() => document.querySelectorAll('.panel').length > 2",
                               timeout=25000)
        state = page.evaluate("""() => ({
            apiReady: !!(window.Api && Api.ready),
            runButtons: document.querySelectorAll('.todo-run').length,
            absent: document.querySelectorAll('.cg-row').length,
            commands: document.querySelectorAll('.todo-cmd code').length,
        })""")
        assert state["apiReady"] is False, "a static host answered /api/health"
        assert state["absent"] > 0, "nothing to check — the deck has every stage"
        assert state["runButtons"] == 0, (
            "the deployed page offered a button that cannot do anything")
        assert state["commands"] > 0, (
            "with no server the command is the whole affordance, and it is missing")
        assert errors == [], errors
    finally:
        page.close()


def test_the_brief_is_readable_on_the_dossier(browser, viz_server):
    """THE BRIEF WAS A STAGED ARTIFACT WITH NO SURFACE, and then it was the
    WRONG one — which is the harder failure.

    `deck_status.STAGES` reports it and `build_deck` consumes it, so it earned a
    surface. But ur-dragon's `brief.json` was the design brief of the treasure
    refactor: a branch that was measured, found worse and deleted, while its
    brief stayed on disk and `deck-info` kept composing it into `info.json`. The
    dossier opened by calling a 24-creature deck a "non-creature treasure
    engine" with an "RGW-only" mana base, naming five cards it does not run.

    So this asserts BOTH halves: a deck with a brief renders it, and a deck
    without one says so rather than showing someone else's.
    """
    from manamap.config import DECKS_DIR
    import json as _json

    with_brief = None
    for d in sorted(DECKS_DIR.iterdir()):
        info = d / "info.json"
        if not info.exists():
            continue
        try:
            if (_json.loads(info.read_text()) or {}).get("brief"):
                with_brief = d.name
                break
        except Exception:
            continue

    # The absent case, on a deck whose brief was archived for describing a
    # different list. "No brief authored" is the honest render.
    # `#panel-cover`, not `#panel-casefile`. The case file became the DOSSIER's
    # cover sheet — the brief's absence is a statement about what the deck is
    # trying to BE, so it moved onto the booking record with the MO. This is the
    # one assertion in the deck-page set that named a layout id rather than a
    # contract, and it is updated rather than worked around.
    page = _dossier(browser, viz_server, "ur-dragon")
    page.wait_for_selector("#panel-cover", timeout=25000)
    body = page.inner_text("body").lower()
    assert "no brief authored" in body
    assert "treasure is the engine" not in body, (
        "the deleted branch's brief is being rendered as this deck's intent")
    # `_dossier` does not record page errors; the dedicated error tests cover
    # that, and asserting an attribute this helper never sets would pass by
    # raising instead of by checking.
    page.close()

    if with_brief is None:
        pytest.skip("no deck on the bench has a brief to render")
    page = _dossier(browser, viz_server, with_brief)
    page.wait_for_selector("#panel-brief", timeout=25000)
    assert page.locator("#panel-brief").count() == 1
    page.close()


@requires_branch
@pytest.mark.browser
def _a_live_branch():
    """A deck with an OPEN branch and that branch's info.json row, from the
    tracked manifest — Edgar's mean-v1 was pinned here on 2026-10-19 and was
    rejected (rebased) a day later. The branch is a pilot artifact, not a
    fixture: skip, never fail, when none is open."""
    import json as _j
    from manamap.config import DATA_DIR
    idx = DATA_DIR / "decks" / "index.json"
    if not idx.exists():
        return None
    for deck in _j.loads(idx.read_text(encoding="utf-8")).get("decks", []):
        if deck.get("status") or not deck.get("branches_open"):
            continue
        info_path = DATA_DIR / "decks" / deck["slug"] / "info.json"
        if not info_path.exists():
            continue
        rows = [b for b in _j.loads(info_path.read_text(encoding="utf-8")).get("branches", [])
                if str(b.get("state", "")).upper() == "OPEN"]
        if rows:
            return deck["slug"], rows[0], _j.loads(info_path.read_text(encoding="utf-8"))
    return None


@pytest.mark.browser
def test_the_deck_page_puts_branches_right_after_next_with_every_state_badged(browser, viz_server):
    """"Further explorations" was the last of ~26 panels and showed an open
    branch as a shopping list. The rack is now "Branches", sits after NEXT, and
    every row carries its state. Chosen from the tracked manifest: a deck with an
    open branch, and the decided fold where it has decided ones."""
    live = _a_live_branch()
    if live is None:
        pytest.skip("no tracked deck has an OPEN branch")
    slug, row, info = live
    decided = [b for b in info.get("branches") or [] if re.match(r"^(MERGED|REJECT|WITHDRAW)", str(b.get("state", "")))]
    page = _dossier(browser, viz_server, slug)
    try:
        page.wait_for_selector("#panel-branches", timeout=25000)
        # In the record chapter, which the sticky nav reaches, and OPEN while a
        # branch is in flight (2026-11-22: the page folds; a decided-only rack folds too).
        assert page.locator("#ch-record #panel-branches").count() == 1
        assert page.evaluate("document.querySelector('#panel-branches > details').open") is True
        assert "Branches" in page.text_content("#panel-branches .panel-title")
        states = page.locator("#panel-branches .br-state[data-state='open']")
        assert states.count() >= 1
        assert page.locator("#panel-branches details.branch-decided-fold").count() == (1 if decided else 0)
        assert row["name"] in page.text_content("#panel-branches .branch:not(.branch-decided) h3")
    finally:
        page.close()


@requires_deck
@pytest.mark.browser
def test_the_deck_page_lands_on_its_branches_rack_from_a_deep_link(browser, viz_server):
    """`deck.html?deck=…#panel-branches` is where the workbench sends a deck with
    several live branches. The panel id is prefixed (`panel()`), the old link
    said `#branches` and nothing rendered that, and even the right id needed a
    post-render scroll."""
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    try:
        page.goto(f"{viz_server}/viz/deck.html?deck=edgar-vampires#panel-branches")
        idle(page)
        assert page.locator("#panel-branches").count() == 1
        assert page.evaluate("window.scrollY") > 0, "the page did not scroll to #panel-branches after its render"
    finally:
        page.close()


def _marked_info(slug):
    import json as _json
    from manamap.config import DECKS_DIR
    info = _json.loads((DECKS_DIR / slug / "info.json").read_text())
    rec = info.setdefault("record", {})
    rec["marks_n"] = 3
    rec["marks"] = {"first_wheel_turn": {"n": 3, "min": 0, "max": 6, "never": 1, "mean": 5.0},
                    "both_turn": {"n": 3, "min": 0, "max": 7, "never": 1, "mean": 6.0},
                    "wheel_with_commander_rate": 0.5, "kept_hand_passes_rule_rate": 0.5,
                    "commander_removed": {"n": 2, "min": 0, "max": 2, "mean": 1.0}}
    return info


@pytest.mark.browser
@requires_deck
def test_the_pages_show_the_pilots_marks_as_the_table_instrument(browser, viz_server):
    """The deck page's status panel and the branch page's 'At the table' block
    read `record.marks` off info.json and print them in words; a deck with no
    marked game shows neither. The page computes nothing."""
    import json as _json
    slug, branch = "sharknado", "finisher-v1"
    from manamap.config import DECKS_DIR
    if not (DECKS_DIR / slug / "branches" / branch / "branch.json").exists():
        pytest.skip(f"{slug}@{branch} is not on disk")
    info = _marked_info(slug)
    page = browser.new_page()
    page.route(f"**/data/decks/{slug}/info.json", lambda route: route.fulfill(
        status=200, content_type="application/json", body=_json.dumps(info)))
    try:
        page.goto(f"{viz_server}/viz/deck.html?deck={slug}")
        idle(page)
        status = page.locator("#panel-status").text_content()
        assert "At the table" in status and "first wheel T5" in status and "never in 1" in status
        assert "landed on a commander in 50%" in status and "3 marked games" in status
        page.goto(f"{viz_server}/viz/branch.html?deck={slug}&branch={branch}")
        idle(page)
        block = page.locator("#table-marks")
        assert block.count() == 1
        text = block.text_content()
        assert "3 marked games" in text and "not this branch" in text
        assert "first wheel resolved: turn 5 on average, never in 1" in text
        assert "kept seven passed the keep rule: 50%" in text
    finally:
        page.close()
    # The control: a deck with no marked game renders no block. Routed, not read off
    # disk — the tracked info.json gained a marked game on 2026-10-27 (sharknado 004,
    # kept_wheels=1), and a control that leans on live data fails the day the pilot logs.
    bare = _json.loads((DECKS_DIR / slug / "info.json").read_text())
    rec = bare.setdefault("record", {})
    rec.pop("marks", None)
    rec["marks_n"] = 0
    page = browser.new_page()
    page.route(f"**/data/decks/{slug}/info.json", lambda route: route.fulfill(
        status=200, content_type="application/json", body=_json.dumps(bare)))
    try:
        page.goto(f"{viz_server}/viz/branch.html?deck={slug}&branch={branch}")
        idle(page)
        assert page.locator("#table-marks").count() == 0
    finally:
        page.close()
