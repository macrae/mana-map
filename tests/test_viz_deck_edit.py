"""Build's edit tray (deck-edit.js) and the branch page's two verbs, in a browser.

The pilot's ruling (2026-10-10): a bench or brewing deck is edited STRAIGHT ON
THE DECK from Build, with live before/after numbers, undo and redo; a SLEEVED
deck offers "Start a branch" instead; an archived deck is read-only; without
`manamap serve` the page says so in one sentence.

Nothing here reaches a server: the manifest is ROUTED (one real deck per rung,
as `test_viz_deck_picker` does, so the test does not depend on which deck is
on which rung this week) and `/api/**` is a fake that records every body, as
`test_viz_printings` does. Every other file is the real deck's own.

Proven by re-introduction: with `wireOps` sending the tray's `name` key
instead of `card` (the wire's word, `serve._OP_KEYS`), the bench test fails
on the posted ops — the exact shape `_oplist` would refuse with a 400.
"""

from __future__ import annotations

import json

import pytest

from conftest_viz import BOOT_TIMEOUT_MS, _record, serve_card_images_locally
from manamap.config import DECKS_DIR

pytestmark = pytest.mark.browser

BENCH, SLEEVED, ARCHIVED = "meren-recursion", "heliod", "hapatra"
CUT, ADD = "Animate Dead", "Llanowar Elves"
BRANCH = "drain-density-v1"

COMMANDS = ["health", "job", "deck/edit", "deck/edit/preview", "deck/edit/undo",
            "deck/edit/redo", "deck/edit/history", "deck/save-version", "branch/new",
            "branch/stage", "branch/axes", "branch/net-change", "branch/upgrades"]

GOLDFISH = {
    "table": [
        {"measure": "kill_by_8", "champion": 0.269, "branch": 0.301, "delta": 0.032,
         "ci95_diff": [0.011, 0.053], "verdict": "better"},
        {"measure": "stall", "champion": 0.089, "branch": 0.087, "delta": -0.002,
         "ci95_diff": [-0.008, 0.004], "verdict": "noise"},
    ],
    "call": "better", "better": ["kill_by_8"], "worse": [],
    "trust": "every card in and out is seen by the goldfish",
    "caveat": "the goldfish has no blockers and no removal",
}


def _sha(slug):
    return json.loads((DECKS_DIR / slug / "cards.json").read_text())["decklist_sha256"]


def _manifest(stages):
    doc = json.loads((DECKS_DIR / "index.json").read_text(encoding="utf-8"))
    by = {d["slug"]: d for d in doc["decks"]}
    decks = []
    for slug, (stage, status) in stages.items():
        e = dict(by[slug])
        e["stage"], e["status"] = stage, status
        e["locked"] = stage == "sleeved"
        decks.append(e)
    return json.dumps({"drafts": [], "decks": decks})


STAGES = {BENCH: ("bench", None), SLEEVED: ("sleeved", None),
          ARCHIVED: (None, ["broken-down", "BROKEN DOWN FOR PARTS", "Pulled for parts."])}


def _fake_api(page, slug):
    """A fake `manamap serve`. Returns the dict every POST body lands in."""
    st = {"posted": {}, "undo": 0, "redo": 0, "sha": _sha(slug), "jobs": {}}

    def ok(route, result):
        route.fulfill(status=200, content_type="application/json",
                      body=json.dumps({"ok": True, "result": result}))

    def body(route):
        return json.loads(route.request.post_data or "{}")

    def record(name, route):
        b = body(route)
        st["posted"].setdefault(name, []).append(b)
        return b

    def history(route):
        ok(route, {"slug": slug, "entries": [], "undo": st["undo"], "redo": st["redo"],
                   "in_sync": True, "decklist_sha256": st["sha"],
                   "since_save": {"out": {}, "in": {}}, "saved_version": 3,
                   "rebuilding": False})

    def preview(route):
        b = record("deck/edit/preview", route)
        st["jobs"]["pv1"] = {"id": "pv1", "state": "done",
                             "result": {"superseded": False, "goldfish": GOLDFISH}}
        ok(route, {"slug": slug, "base_sha": st["sha"], "ops": b["ops"],
                   "diff": {"out": {CUT: 1}, "in": {ADD: 1}},
                   "size": {"before": 100, "after": 100}, "blocking": [], "warnings": [],
                   "keep_list_hits": [],
                   "curve": {"before": {"1": 5, "2": 12}, "after": {"1": 6, "2": 11}},
                   "roles": {"ramp:dork": 1, "recursion": -1},
                   "combos": {"gained": [], "lost": [{"cards": [CUT, "Worldgorger Dragon"]}],
                              "before": 3, "after": 2},
                   "price": {"as_of": "2026-10-01", "source": "manapool",
                             "dates": ["2026-10-01"], "delta_cents": -150, "unpriced": []},
                   "colour_sources": {"B": {"before": 30, "after": 30, "target": 25,
                                            "short_after": 0}},
                   "goldfish": {"pending": "pv1"}, "job": {"id": "pv1", "state": "running"},
                   "token": "t1"})

    def edit(route):
        record("deck/edit", route)
        st["sha"], st["undo"], st["redo"] = "sha-after-edit", st["undo"] + 1, 0
        st["jobs"]["rb1"] = {"id": "rb1", "state": "done",
                             "result": {"decklist_sha256": st["sha"], "ran": ["fetch-deck"]}}
        ok(route, {"slug": slug, "written": True, "after_sha": st["sha"],
                   "diff": {"out": {CUT: 1}, "in": {ADD: 1}}, "warnings": [],
                   "entry": {"kind": "edit", "source": "ui"},
                   "job": {"id": "rb1", "state": "running"}})

    def undo(route):
        record("deck/edit/undo", route)
        st["sha"], st["undo"], st["redo"] = _sha(slug), st["undo"] - 1, st["redo"] + 1
        st["jobs"]["rb2"] = {"id": "rb2", "state": "done",
                             "result": {"decklist_sha256": st["sha"], "ran": []}}
        ok(route, {"slug": slug, "after_sha": st["sha"],
                   "diff": {"out": {ADD: 1}, "in": {CUT: 1}},
                   "entry": {"kind": "undo"}, "job": {"id": "rb2", "state": "running"}})

    def job(route):
        b = body(route)
        ok(route, st["jobs"].get(b.get("id"), {"id": b.get("id"), "state": "failed",
                                                "error": "unknown job"}))

    def generic(name, result):
        def h(route):
            record(name, route)
            ok(route, result)
        return h

    page.route("**/api/**", lambda route: route.fulfill(
        status=500, content_type="application/json", body='{"error": "unmocked in test"}'))
    page.route("**/api/health", lambda route: ok(route, {"ok": True, "api": 1}))
    page.route("**/api/", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps({"commands": COMMANDS})))
    page.route("**/api/deck/edit/history*", history)
    page.route("**/api/deck/edit/preview", preview)
    page.route("**/api/deck/edit", edit)
    page.route("**/api/deck/edit/undo", undo)
    page.route("**/api/job", job)
    page.route("**/api/branch/axes*", lambda route: ok(route, {"slug": slug, "read": True, "axes": [
        {"axis": "kill_by_8", "current": 0.269, "lower_is_better": False, "needs": None},
        {"axis": "stall", "current": 0.089, "lower_is_better": True, "needs": None}]}))
    page.route("**/api/branch/new", generic("branch/new", {"url": "branch.html"}))
    page.route("**/api/branch/stage", generic("branch/stage", {"ok": True}))
    page.route("**/api/branch/upgrades", lambda route: ok(route, {"swaps": [], "notes": []}))

    def net_change(route):
        record("branch/net-change", route)
        st["jobs"]["nc1"] = {"id": "nc1", "state": "done", "result": {"recommendation": {}}}
        ok(route, {"id": "nc1", "state": "running"})
    page.route("**/api/branch/net-change", net_change)
    return st


def _open(browser, viz_server, query, api_slug=None, path="index.html", stages=None):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list[str] = []
    add = _record(errors)
    page.on("pageerror", lambda e: add(e))
    page.on("console", lambda m: add(m.text) if m.type == "error" else None)
    body = _manifest(stages or STAGES)
    page.route("**/data/decks/index.json*",
               lambda route: route.fulfill(status=200, content_type="application/json",
                                           body=body))
    serve_card_images_locally(page)
    st = _fake_api(page, api_slug) if api_slug else None
    page.goto(f"{viz_server}/viz/{path}{query}")
    page.add_style_tag(content="*, *::before, *::after {"
                               " transition: none !important; animation: none !important; }")
    page.js_errors = errors
    return page, st


def _wait_mode(page, slug, mode):
    page.wait_for_function(
        "([s, m]) => window.Build && Build.deckSlug === s && window.DeckEdit && DeckEdit.mode === m",
        arg=[slug, mode], timeout=BOOT_TIMEOUT_MS)


def _until(page, cond, timeout_ms=10000):
    """Let the page run (route handlers fire inside Playwright waits) until `cond()`."""
    waited = 0
    while not cond():
        assert waited < timeout_ms, "timed out waiting on the fake API"
        page.wait_for_timeout(50)
        waited += 50


def _card(page, name):
    page.evaluate("n => { MM.closeDetail(); MM.selectByName(n); }", name)
    page.wait_for_selector("#detailInner .de-card button", timeout=10000)


def test_a_bench_edit_previews_applies_and_undoes(browser, viz_server):
    # The tray is Build's Change tab (2026-10-28); a bench deck opens on Deck.
    page, st = _open(browser, viz_server, f"?mode=build&deck={BENCH}&tab=change", api_slug=BENCH)
    try:
        _wait_mode(page, BENCH, "edit")
        page.wait_for_function("() => DeckEdit.baseSha", timeout=10000)
        assert page.query_selector("#deckEditTray") is not None

        _card(page, CUT)
        page.click("#detailInner .de-card [data-de-act='cut']")
        _card(page, ADD)
        page.click("#detailInner .de-card [data-de-act='add']")
        assert page.evaluate("() => DeckEdit.tray") == [
            {"op": "cut", "card": CUT, "qty": 1}, {"op": "add", "card": ADD, "qty": 1}]
        rows = page.inner_text("#deRows")
        assert f"− {CUT}" in rows and f"+ {ADD}" in rows, rows
        assert "1 out, 1 in — 100 cards" in rows, rows

        # The instant tier, then the goldfish tier with the interval ON THE DIFFERENCE.
        page.wait_for_function(
            "() => (document.querySelector('#dePreview .de-preview:not(.is-stale) .de-gf') || {}).textContent",
            timeout=10000)
        pv = page.evaluate("() => document.getElementById('dePreview').textContent")
        assert "[+0.011, +0.053]" in pv, pv
        # Grouped by axis, no single call line (test_viz_build_tabs holds the groups).
        assert "kill_by_8" in pv and "No call" in pv and "Better" in pv and "⇒" not in pv, pv
        assert "−$1.50" in pv and "as of 2026-10-01" in pv, pv
        assert "3 → 2" in pv and "Worldgorger Dragon" in pv, pv
        assert "no blockers" in pv and "every card in and out is seen" in pv, pv
        assert st["posted"]["deck/edit/preview"][-1] == {
            "slug": BENCH, "ops": [{"op": "cut", "card": CUT, "qty": 1},
                                   {"op": "add", "card": ADD, "qty": 1}]}

        page.click("#deActions .de-apply")
        page.wait_for_function("() => DeckEdit.tray.length === 0 && DeckEdit.baseSha === 'sha-after-edit'",
                               timeout=10000)
        assert st["posted"]["deck/edit"] == [{
            "slug": BENCH, "expect_sha": _sha(BENCH),
            "ops": [{"op": "cut", "card": CUT, "qty": 1}, {"op": "add", "card": ADD, "qty": 1}]}]

        # The rebuild lands, Build reloads, and Undo is offered from the history.
        page.wait_for_function(
            "() => !document.querySelector('.de-measuring') && "
            "document.querySelector('#deActions .de-undo') && "
            "!document.querySelector('#deActions .de-undo').disabled", timeout=15000)
        _card(page, "Arcane Signet")
        page.click("#detailInner .de-card [data-de-act='cut']")
        assert len(page.evaluate("() => DeckEdit.tray")) == 1
        page.click("#deActions .de-undo")
        page.wait_for_function("() => DeckEdit.tray.length === 0", timeout=10000)
        assert st["posted"]["deck/edit/undo"] == [{"slug": BENCH, "expect_sha": "sha-after-edit"}]
        assert page.js_errors == []
    finally:
        page.close()


def _count(page):
    el = page.locator("#deckInner #buildDeckCount")
    return el.inner_text(), "is-off" in (el.get_attribute("class") or "")


def test_the_header_count_follows_the_tray(browser, viz_server):
    """Build's header count moves with the tray's pending edits, painted in place:
    a cut reads short and off-size, the add that rebalances it reads whole again."""
    page, _ = _open(browser, viz_server, f"?mode=build&deck={BENCH}", api_slug=BENCH)
    try:
        _wait_mode(page, BENCH, "edit")
        page.wait_for_function("() => DeckEdit.baseSha", timeout=10000)
        assert _count(page) == ("100 / 100 cards", False)
        _card(page, CUT)
        page.click("#detailInner .de-card [data-de-act='cut']")
        assert _count(page) == ("99 / 100 cards · 1 pending", True)
        _card(page, ADD)
        page.click("#detailInner .de-card [data-de-act='add']")
        text, off = _count(page)
        assert text.startswith("100 / 100 cards") and off is False, (text, off)
        assert page.js_errors == []
    finally:
        page.close()


def test_a_sleeved_deck_offers_a_branch_and_no_apply(browser, viz_server):
    page, st = _open(browser, viz_server, f"?mode=build&deck={SLEEVED}", api_slug=SLEEVED)
    try:
        _wait_mode(page, SLEEVED, "branch")
        page.wait_for_selector("#deckEditTray[data-mode='branch']", timeout=10000)
        assert page.query_selector("#deckEditTray .de-apply") is None
        assert page.query_selector("#deckEditTray .de-undo") is None
        # Nothing is offered until the tray has a row (2026-10-28).
        assert page.query_selector("#deckEditTray .de-start-branch") is None

        # An unbalanced tray is refused with the reason, and nothing is posted.
        _card(page, "Arcane Denial")
        page.click("#detailInner .de-card [data-de-act='cut']")
        page.wait_for_selector("#deActions .de-start-branch", timeout=5000)
        assert "Start a branch" in page.inner_text("#deActions")
        # The axes select carries the deck's current reading; the guard is optional.
        page.wait_for_function("() => [...document.querySelectorAll('[data-de-field=axis] option')]"
                               ".some(o => o.value === 'kill_by_8')", timeout=10000)
        page.select_option("[data-de-field=axis]", "kill_by_8")
        assert "kill_by_8 (now 0.269)" in page.inner_text("#deForm")
        assert page.input_value("[data-de-field=value]") == "0.269"
        page.fill("[data-de-field=branch]", "trim-v1")
        page.click("#deActions .de-start-branch")
        page.wait_for_selector("#deStatus .de-error", timeout=5000)
        assert "1 out, 0 in" in page.inner_text("#deStatus")
        assert "branch/new" not in st["posted"]
        assert page.js_errors == []
    finally:
        page.close()


def test_without_the_local_server_the_page_says_so_and_offers_no_edit(browser, viz_server):
    page, _ = _open(browser, viz_server, f"?mode=build&deck={BENCH}&tab=change")
    try:
        _wait_mode(page, BENCH, "noapi")
        page.wait_for_selector("#deckEditNoApi", timeout=10000)
        text = page.inner_text("#deckEditNoApi")
        assert "Editing needs the local bench" in text, text
        assert "run manamap serve and open this page from it" in text, text
        assert page.query_selector("#deckEditTray") is None
        page.evaluate("n => { MM.closeDetail(); MM.selectByName(n); }", CUT)
        page.wait_for_selector("#detailInner .viewer-header", timeout=10000)
        assert page.query_selector("#detailInner .de-card button") is None
        assert page.js_errors == []
    finally:
        page.close()


def test_an_archived_deep_link_has_no_tray(browser, viz_server):
    page, st = _open(browser, viz_server, f"?mode=build&deck={ARCHIVED}&tab=change", api_slug=ARCHIVED)
    try:
        _wait_mode(page, ARCHIVED, "readonly")
        page.wait_for_selector("#deckEditReadonly", timeout=10000)
        text = page.inner_text("#deckEditReadonly")
        assert "archived (broken-down)" in text and "read only" in text, text
        assert f"deck-state {ARCHIVED} revive" in text, text
        assert page.query_selector("#deckEditTray") is None
        assert "deck/edit/history" not in st["posted"]
        assert page.js_errors == []
    finally:
        page.close()


def test_the_branch_page_measures_and_unstages(browser, viz_server):
    meta = json.loads((DECKS_DIR / BENCH / "branches" / BRANCH / "branch.json").read_text())
    first = meta["staged"][0]
    page, st = _open(browser, viz_server, f"?deck={BENCH}&branch={BRANCH}",
                     api_slug=BENCH, path="branch.html")
    try:
        page.wait_for_selector("#benchActions [data-act='unstage']", timeout=30000)
        assert len(page.query_selector_all("#benchActions [data-act='unstage']")) == len(meta["staged"])
        page.click("#benchActions [data-act='unstage']")
        _until(page, lambda: st["posted"].get("branch/stage"))
        page.wait_for_selector("#benchActions [data-act='measure']", timeout=10000)
        assert st["posted"]["branch/stage"] == [{"slug": BENCH, "branch": BRANCH, "undo": True,
                                                "out": first["out"], "card": first["in"]}]
        page.click("#benchActions [data-act='measure']")
        _until(page, lambda: st["posted"].get("branch/net-change"))
        # The job is polled to `done` and the page re-renders from the artifacts:
        # a fresh, enabled Measure button with no "measuring" left on it.
        page.wait_for_function(
            "() => { const b = document.querySelector('#benchActions [data-act=measure]');"
            " const s = document.getElementById('measureState');"
            " return b && !b.disabled && s && s.textContent === ''; }", timeout=10000)
        assert st["posted"]["branch/net-change"] == [{"slug": BENCH, "branch": BRANCH}]
        assert page.js_errors == []
    finally:
        page.close()


# ── the tray across a reload, a 60-card list, a superseded deck ──────────

SIXTY, SIXTY_CUT = "elves", "Llanowar Elves"
SUPERSEDED = "gishath"


def _stored(page):
    return page.evaluate("""() => Object.keys(sessionStorage)
        .filter(k => k.indexOf('mm.deckEdit.') === 0).sort()""")


def test_the_tray_survives_a_reload_until_the_list_moves(browser, viz_server):
    """sessionStorage under slug + the list's sha: the same list gets its tray
    back after a reload; a list that moved (an edit in another tab, the CLI)
    drops it, because a tray built against one list means nothing on another."""
    page, st = _open(browser, viz_server, f"?mode=build&deck={BENCH}&tab=change", api_slug=BENCH)
    try:
        _wait_mode(page, BENCH, "edit")
        page.wait_for_function("() => DeckEdit.baseSha", timeout=10000)
        _card(page, CUT)
        page.click("#detailInner .de-card [data-de-act='cut']")
        tray = [{"op": "cut", "card": CUT, "qty": 1}]
        assert page.evaluate("() => DeckEdit.tray") == tray
        assert _stored(page) == [f"mm.deckEdit.{BENCH}.{st['sha']}"]

        page.reload()
        _wait_mode(page, BENCH, "edit")
        page.wait_for_function("() => DeckEdit.baseSha && DeckEdit.tray.length === 1",
                               timeout=10000)
        assert page.evaluate("() => DeckEdit.tray") == tray
        assert f"− {CUT}" in page.inner_text("#deRows")

        old = st["sha"]
        st["sha"] = "sha-moved-elsewhere"            # the next history read says so
        page.reload()
        _wait_mode(page, BENCH, "edit")
        page.wait_for_function("s => DeckEdit.baseSha === s", arg=st["sha"], timeout=10000)
        assert page.evaluate("() => DeckEdit.tray") == []
        assert f"mm.deckEdit.{BENCH}.{old}" not in _stored(page)
        assert page.query_selector("#deRows .de-empty") is not None
        assert page.js_errors == []
    finally:
        page.close()


def test_a_sixty_card_deck_edits_copy_counts(browser, viz_server):
    """Modern is `exact: false` with four copies: every cut/add row carries a
    copy-count input, and the typed count is the `qty` the preview and Apply post
    (on the cut/add op — the tray does not send `set`)."""
    stages = dict(STAGES, **{SIXTY: ("bench", None)})
    page, st = _open(browser, viz_server, f"?mode=build&deck={SIXTY}&tab=change", api_slug=SIXTY,
                     stages=stages)
    try:
        _wait_mode(page, SIXTY, "edit")
        page.wait_for_function("() => DeckEdit.baseSha", timeout=10000)
        _card(page, SIXTY_CUT)
        page.click("#detailInner .de-card [data-de-act='cut']")
        # The header count follows the tray: one copy out of a 60-card deck.
        assert _count(page) == ("59 / 60 cards · 1 pending", True)
        qty = page.wait_for_selector("#deRows input.de-qty[data-de-qty='0']", timeout=5000)
        assert qty.input_value() == "1"
        qty.fill("3")
        qty.dispatch_event("change")
        want = [{"op": "cut", "card": SIXTY_CUT, "qty": 3}]
        page.wait_for_function("() => DeckEdit.tray[0].qty === 3", timeout=5000)
        assert "3 out, 0 in" in page.inner_text("#deRows")
        _until(page, lambda: (st["posted"].get("deck/edit/preview") or [{}])[-1].get("ops") == want)
        page.click("#deActions .de-apply")
        _until(page, lambda: st["posted"].get("deck/edit"))
        assert st["posted"]["deck/edit"] == [{"slug": SIXTY, "expect_sha": _sha(SIXTY),
                                             "ops": want}]
        assert page.js_errors == []
    finally:
        page.close()


def test_a_superseded_deep_link_is_read_only(browser, viz_server):
    """Superseded is not "apart" (`common.deck_is_apart`), but it is off every
    rung, so it is not a deck to edit: no tray, the read-only line, and not one
    call to the edit verbs."""
    stages = dict(STAGES, **{SUPERSEDED: (None, ["superseded", "SUPERSEDED",
                                                 "Replaced by a newer build."])})
    page, st = _open(browser, viz_server, f"?mode=build&deck={SUPERSEDED}&tab=change",
                     api_slug=SUPERSEDED, stages=stages)
    try:
        _wait_mode(page, SUPERSEDED, "readonly")
        page.wait_for_selector("#deckEditReadonly", timeout=10000)
        text = page.inner_text("#deckEditReadonly")
        assert "archived (superseded)" in text and "read only" in text, text
        assert f"deck-state {SUPERSEDED} revive" in text, text
        assert page.query_selector("#deckEditTray") is None
        assert not any(k.startswith("deck/edit") for k in st["posted"]), st["posted"]
        assert page.js_errors == []
    finally:
        page.close()


def test_the_branch_page_says_when_the_base_moved(browser, viz_server):
    """`deck_branch.base_moved`'s sentence travels on info.json's branch row; the
    page shows it as one line and computes nothing."""
    info = json.loads((DECKS_DIR / BENCH / "info.json").read_text())
    line = ("base moved since 0123456789ab — the deck's list changed after this "
            "branch was cut, so its comparison is against an older deck")
    for b in info.get("branches") or []:
        if b.get("name") == BRANCH:
            b["base_moved"] = line
    assert any(b.get("base_moved") for b in info.get("branches") or []), "no branch row"
    body = json.dumps(info)
    page, _ = _open(browser, viz_server, f"?deck={BENCH}&branch={BRANCH}", path="branch.html")
    try:
        page.route(f"**/data/decks/{BENCH}/info.json*", lambda route: route.fulfill(
            status=200, content_type="application/json", body=body))
        page.reload()
        page.wait_for_selector("#baseMoved", timeout=30000)
        assert page.inner_text("#baseMoved") == line
        assert len(page.query_selector_all("#baseMoved")) == 1
        assert page.js_errors == []
    finally:
        page.close()
