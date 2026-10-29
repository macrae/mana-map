"""Send to Jarvis, and a deck that opens without waiting on its matrix — P2 of the
pilot's Atlas review, 2026-10-28.

The pilot talks to Jarvis in the Claude Code terminal and looks at these pages. A
small "Ask Jarvis" button sits on the card panel (every mode, the Build overlay
included), on the swap being tried in Build's Change tab, and on a branch row (Build's
Change tab and `branch.html`). It opens a one-line prompt prefilled with a question
for that thing; Send copies `/jarvis <text>` to the clipboard and, under a local
serve, leaves the question on the page-state record as the pending ASK, which
`manamap pilot page-state` prints and `--clear-ask` marks taken. The page never
calls an LLM.

Three servers, three contracts: serve's own `Handler` with `page_state.PATH` in tmp
(the question lands on the record — the real `.progress/` is never touched), the
fake `/api` from `test_viz_deck_edit` with `page/state` added (Build's tray, nothing
written), and the suite's static `viz_server` (no serve: the clipboard is the whole
of it, and nothing is POSTed).

Each test was proven by putting the old behaviour back: the post without its `ask`
(serve answers "old", not "held"), no button on the tray, the branch rows without
`ask`, the prompt's keys reaching the page (Escape closed Build's card), the copy
without its fallback, and the deck open without the boot prefetch (the matrix was
requested after every one of the deck's files).
"""
from __future__ import annotations

import json

import pytest

from conftest_viz import BOOT_TIMEOUT_MS, _record, await_projection, serve_card_images_locally
from test_viz_deck_edit import ADD, BENCH, COMMANDS, CUT, _card, _fake_api, _manifest, STAGES, _wait_mode
from test_viz_page_state import serve_server  # noqa: F401  (the fixture)

pytestmark = pytest.mark.browser

BRANCH = "drain-density-v1"     # meren-recursion's live branch (its info.json)
EMB = "embeddings_cardbert.bin"


def _ctx(browser, clipboard=True):
    perms = ["clipboard-read", "clipboard-write"] if clipboard else []
    return browser.new_context(viewport={"width": 1440, "height": 900}, permissions=perms)


def _watch(page):
    errors: list[str] = []
    add = _record(errors)
    page.on("pageerror", lambda e: add(e))
    page.on("console", lambda m: add(m.text) if m.type == "error" else None)
    page.js_errors = errors
    posts: list[dict] = []
    page.on("request", lambda r: posts.append(json.loads(r.post_data or "{}"))
            if r.method == "POST" and "/api/page/state" in r.url else None)
    page.state_posts = posts
    return page


def _send(page):
    page.click("#mmAsk .mm-ask-send")
    page.wait_for_function("() => PageState.lastAsk && PageState.lastAsk.posted", timeout=10_000)
    return page.evaluate("() => PageState.lastAsk")


def _note(page):
    return page.inner_text("#mmAsk .mm-ask-note")


# ── the card panel, under serve: the question lands on the record ──────────


def test_a_card_question_is_copied_and_left_on_the_record_for_jarvis(browser, serve_server, capsys):
    base, state = serve_server
    ctx = _ctx(browser)
    page = _watch(ctx.new_page())
    serve_card_images_locally(page)
    try:
        page.goto(f"{base}/viz/index.html?card=Windfall")
        page.wait_for_selector(".detail-jarvis", timeout=BOOT_TIMEOUT_MS)
        page.click(".detail-jarvis")
        assert page.input_value("#mmAsk .mm-ask-text") == \
            "What does Windfall do, and which of my decks wants it?"
        got = _send(page)
        assert got["posted"] == "held" and got["copied"] == "copied", got
        assert page.evaluate("() => navigator.clipboard.readText()") == \
            "/jarvis What does Windfall do, and which of my decks wants it?"
        assert _note(page).startswith("Copied — paste it into Claude Code."), _note(page)

        tabs = json.loads(state.read_text())["tabs"]
        asks = [t["ask"] for t in tabs.values() if t.get("ask")]
        assert len(asks) == 1, tabs
        assert asks[0]["text"] == "What does Windfall do, and which of my decks wants it?"
        assert asks[0]["cards"] == ["Windfall"] and asks[0]["page"] == "atlas"

        # What Jarvis reads: the ask first, on its own line; then taken.
        from manamap.pilot import page_state
        page_state.main(type("A", (), {"as_json": False, "clear_ask": False}))
        out = capsys.readouterr().out.splitlines()
        assert out[0] == 'ASK      "What does Windfall do, and which of my decks wants it?"', out
        assert page_state.clear_ask()["text"].startswith("What does Windfall")
        assert page_state.pending_ask(page_state.read()) is None
        assert page.js_errors == []
    finally:
        ctx.close()


# ── Build's Change tab: the swap being tried, against the fake /api ────────


def _open_build(browser, viz_server, query, clipboard=True):
    """`test_viz_deck_edit`'s fake serve, with `page/state` added: it records the body
    and answers as a serve that keeps an ask does (the ask in the returned row)."""
    ctx = _ctx(browser, clipboard)
    page = _watch(ctx.new_page())
    body = _manifest(STAGES)
    page.route("**/data/decks/index.json*", lambda r: r.fulfill(
        status=200, content_type="application/json", body=body))
    serve_card_images_locally(page)
    _fake_api(page, BENCH)
    page.route("**/api/", lambda r: r.fulfill(status=200, content_type="application/json",
                                              body=json.dumps({"commands": COMMANDS + ["page/state"]})))

    def state(route):
        b = json.loads(route.request.post_data or "{}")
        row = dict(b.get("state") or {})
        route.fulfill(status=200, content_type="application/json",
                      body=json.dumps({"ok": True, "result": row}))
    page.route("**/api/page/state", state)
    page.goto(f"{viz_server}/viz/index.html{query}")
    page.add_style_tag(content="*, *::before, *::after { transition: none !important;"
                               " animation: none !important; }")
    return ctx, page


def test_the_swap_being_tried_asks_should_i_and_the_clipboard_fallback_selects_it(browser, viz_server):
    ctx, page = _open_build(browser, viz_server, f"?mode=build&deck={BENCH}&tab=change")
    try:
        _wait_mode(page, BENCH, "edit")
        page.wait_for_function("() => DeckEdit.baseSha", timeout=10_000)
        assert page.query_selector("#deRows .de-jarvis") is None, "no tray, no question"
        _card(page, CUT)
        page.click("#detailInner .de-card [data-de-act='cut']")
        _card(page, ADD)
        page.click("#detailInner .de-card [data-de-act='add']")
        page.wait_for_selector("#deRows .de-jarvis", timeout=10_000)
        page.click("#deRows .de-jarvis")
        name = page.evaluate("() => Build.deckName")
        assert name and page.input_value("#mmAsk .mm-ask-text") == \
            f"Should I cut {CUT} and add {ADD} in {name}?"

        # Editable; and a clipboard that refuses leaves the exact text SELECTED for ⌘C.
        page.fill("#mmAsk .mm-ask-text", f"Should I swap {CUT} for {ADD} here?")
        page.evaluate("""() => {
            navigator.clipboard.writeText = () => Promise.reject(new Error('denied'));
            document.execCommand = () => false;
        }""")
        got = _send(page)
        assert got["posted"] == "held" and got["copied"] == "manual", got
        sel = page.evaluate("""() => { const i = document.querySelector('#mmAsk .mm-ask-text');
            return {v: i.value, s: i.selectionStart, e: i.selectionEnd, ro: i.readOnly}; }""")
        clip = f"/jarvis Should I swap {CUT} for {ADD} here?"
        assert sel == {"v": clip, "s": 0, "e": len(clip), "ro": True}, sel
        assert "⌘C" in _note(page) and "paste it into Claude Code" in _note(page)

        ask = [p["state"]["ask"] for p in page.state_posts if (p.get("state") or {}).get("ask")]
        assert ask == [{"text": f"Should I swap {CUT} for {ADD} here?", "page": "atlas",
                        "deck": BENCH, "cards": [CUT, ADD]}], ask
        assert page.js_errors == []
    finally:
        ctx.close()


# ── a branch row, with no serve: the clipboard is the whole of it ──────────


def test_a_branch_row_asks_for_a_review_and_without_serve_says_so(browser, viz_server):
    ctx = _ctx(browser)
    page = _watch(ctx.new_page())
    serve_card_images_locally(page)
    try:
        page.goto(f"{viz_server}/viz/index.html?deck={BENCH}&tab=change")
        # No serve: the tray says so in one line and the live branches still list.
        page.wait_for_selector("#deckInner .de-branches .br-jarvis", timeout=BOOT_TIMEOUT_MS)
        page.click("#deckInner .de-branches .br-jarvis")
        name = page.evaluate("() => Build.deckName")
        assert page.input_value("#mmAsk .mm-ask-text") == f"Review branch {BRANCH} on {name}"
        got = _send(page)
        assert got["posted"] == "static" and got["copied"] == "copied", got
        assert got["ask"]["branch"] == BRANCH and got["ask"]["deck"] == BENCH
        assert "No local serve" in _note(page), _note(page)
        assert page.evaluate("() => navigator.clipboard.readText()") == \
            f"/jarvis Review branch {BRANCH} on {name}"
        assert page.state_posts == []

        page.goto(f"{viz_server}/viz/branch.html?deck={BENCH}&branch={BRANCH}")
        page.wait_for_selector("#branchAsk .br-jarvis", timeout=BOOT_TIMEOUT_MS)
        page.click("#branchAsk .br-jarvis")
        assert page.input_value("#mmAsk .mm-ask-text") == f"Review branch {BRANCH} on Meren Recursion"
        assert page.js_errors == []
    finally:
        ctx.close()


def test_the_prompts_keys_are_its_own_escape_closes_it_not_builds_card(browser, viz_server):
    """Escape in the prompt closes the prompt; Build's peel chain (card overlay first)
    must not see it, or one key closes both."""
    ctx = _ctx(browser)
    page = _watch(ctx.new_page())
    serve_card_images_locally(page)
    try:
        page.goto(f"{viz_server}/viz/index.html?deck={BENCH}")
        # The graph seeded (it pins the commander, which moves the focus) before the card
        # is opened, so the overlay is the card this test opened.
        page.wait_for_function("s => window.Build && Build.deckSlug === s && Force.nodeCount >= 90",
                               arg=BENCH, timeout=BOOT_TIMEOUT_MS)
        page.evaluate("() => Build.openCard(MM.allData.findIndex(d => d.n === 'Animate Dead'))")
        page.wait_for_function("""() => { const b = document.querySelector('#deckInner .detail-jarvis');
            return b && b.getAttribute('data-jarvis-ask').includes('Animate Dead'); }""", timeout=10_000)
        page.click("#deckInner .detail-jarvis")
        name = page.evaluate("() => Build.deckName")
        assert page.input_value("#mmAsk .mm-ask-text") == f"What does Animate Dead do for {name}?"
        page.keyboard.press("Escape")
        assert page.query_selector("#mmAsk") is None
        assert page.evaluate("() => Build.cardOpen") is True
        assert page.js_errors == []
    finally:
        ctx.close()


# ── load: the matrix starts when a deck will open, and only then ───────────


def _requests(page):
    serve_card_images_locally(page)    # an unanswered Scryfall image holds `load`
    seen: list[str] = []
    page.on("request", lambda r: seen.append(r.url))
    return seen


def test_a_deck_link_requests_the_embeddings_before_the_graph_can_seed(browser, viz_server):
    """`?deck=` starts the 17.9 MB matrix at boot. It used to be the LAST request of the
    open — after cards.json, every stack, info.json — because only `Force.enter` asked
    for it, and that runs once the deck's files are in."""
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    seen = _requests(page)
    try:
        page.goto(f"{viz_server}/viz/index.html?deck=sharknado")
        page.wait_for_function("() => window.Force && Force.nodeCount >= 90", timeout=BOOT_TIMEOUT_MS)
        emb = [i for i, u in enumerate(seen) if EMB in u]
        cards = [i for i, u in enumerate(seen) if "/decks/sharknado/cards.json" in u]
        assert len(emb) == 1 and cards, seen
        assert emb[0] < cards[0], [u.rsplit("/", 1)[-1] for u in seen]
    finally:
        page.close()


def test_the_deck_picker_starts_the_matrix_and_the_landing_never_does(browser, viz_server):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    seen = _requests(page)
    try:
        page.goto(f"{viz_server}/viz/index.html?card=Windfall")
        page.wait_for_function("() => window.Discovery && Discovery.isReady() && Force.nodeCount > 0",
                               timeout=BOOT_TIMEOUT_MS)
        await_projection(page)      # the landing's whole boot, then a beat
        page.wait_for_timeout(800)
        assert not [u for u in seen if EMB in u], "the Discover landing fetched the matrix"

        page.evaluate("() => MM.setMode('build')")
        page.wait_for_selector("#deckLensSelect", timeout=BOOT_TIMEOUT_MS)
        page.wait_for_timeout(300)
        assert not [u for u in seen if EMB in u], "Build without a deck fetched the matrix"
        page.focus("#deckLensSelect")
        for _ in range(40):
            if [u for u in seen if EMB in u]:
                break
            page.wait_for_timeout(100)
        assert len([u for u in seen if EMB in u]) == 1, "opening the picker did not start it"
    finally:
        page.close()
