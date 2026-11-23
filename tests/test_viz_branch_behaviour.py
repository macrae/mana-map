"""Browser behaviour: the branch page (viz/branch.html).

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


def _branch_page(browser, viz_server, slug, branch):
    page = browser.new_page(viewport={"width": 1440, "height": 1200})
    errors: list[str] = []
    add = _record(errors)
    page.on("pageerror", lambda e: add(e))
    # A 404 on an OPTIONAL artifact is the page probing, not a fault — a branch
    # with no net_change.json is a normal state the page renders a panel for.
    page.on("console",
            lambda m: add(m.text)
            if m.type == "error" and "Failed to load resource" not in m.text
            else None)
    page.goto(f"{viz_server}/viz/branch.html?deck={slug}&branch={branch}")
    idle(page)
    # Past the call the page folds (2026-11-22); these tests read the panels'
    # content, so they open every fold first. The goldfish's own <details> is left
    # as the page draws it: the profile test opens it itself to measure the order.
    page.evaluate("document.querySelectorAll('section.panel.fold > details').forEach(d => { d.open = true; })")
    page.js_errors = errors
    return page


@requires_branch  # a branch is a pilot artifact, not a fixture: skip, never fail
@pytest.mark.browser
def test_the_branch_page_renders_the_decision(browser, viz_server):
    """A REAL RENDER, not a source assertion.

    This repo's own record: a `ReferenceError` that killed drill mode outright
    passed all 13 source-grepping drill tests. The branch page is the surface a
    spending decision is read on, so it gets a real browser or it gets nothing.
    """
    page = _branch_page(browser, viz_server, "ur-dragon", A_BRANCH)
    try:
        assert not page.js_errors, page.js_errors
        body = page.inner_text("body").lower()
        # AN UNMEASURED BRANCH IS A NORMAL STATE, and the page renders it with
        # the command that fixes it. This used to assert the treasure branch's
        # own table — that branch was measured, found worse and deleted, which
        # is exactly why a test may not pin an experimental deck (PLAN.md, the
        # 2026-08-27 issue). Both states are asserted; neither is assumed.
        if "no goldfish report yet" in body:
            assert "net-change" in body, "an absent report must name the command"
            assert "guard" in body, "a branch's guard shows before it is measured"
            return
        # The study, the instrument's table (folded under "The goldfish"), the ledger,
        # the bill. Lower-cased because `.panel h2` uppercases in CSS, and asserting
        # the source casing would be testing my own string rather than the page.
        assert "the study" in body
        assert "the goldfish" in body
        assert "what this cannot see" in body
        assert "the mana" in body
        # No bill, no ownership (2026-11-22): the proposal says what moves, never
        # what is owned or what to buy.
        assert "the proposal" in body
        assert "already own" not in body and "to buy" not in body
        # THE ENGINE LIFT WAS DELETED 2026-08-28: it split games by the
        # `required` flags in an AUTHORED declaration, so three defensible
        # wordings of one Ur-Dragon list gave lifts of +0.007, -0.036 and
        # +0.014 over the same 10,000 games. The page must not show it again.
        assert "does the engine make it win?" not in body
        # Every row states what it measures and how to read it, so a mean is
        # never taken for a rate nor a clock for a win rate: one panel at the
        # foot, each row's name linking to its entry.
        assert "what the figures mean" in body
        first = page.locator("#measured th.nc-name a").first.get_attribute("href")
        assert page.locator(first).count() == 1, first
        # And the figures are TEXT, not markup arriving as data — the `facts()`
        # escaping defect one page over.
        assert "<b>" not in body and "&amp;" not in body
    finally:
        page.close()


@requires_branch  # a branch is a pilot artifact, not a fixture: skip, never fail
@pytest.mark.browser
def test_the_branch_page_leads_with_the_executive_summary(browser, viz_server):
    """THE PILOT, 2026-11-14: "the exact cards are kind of hidden ... what I really
    want is an executive summary." The summary is the first panel, names the deck
    and why, and pairs every swap with a card name that pops its image.

    And the pairs must be TRUE: Heliod b3-v1's diff paired Top with Mana Drain, a
    card the sleeved list never had, so the table showed a cut that never happened.
    Every OUT name in the summary must be one the Cards panel also lists as cut."""
    page = _branch_page(browser, viz_server, "ur-dragon", A_BRANCH)
    try:
        assert not page.js_errors, page.js_errors
        # THE STUDY LEADS (2026-11-21), the summary follows it: the decision in the
        # pilot's words before the proposal's own account of itself.
        # In the call chapter, the first of four (2026-11-22).
        assert page.locator("#ch-call .panels > section.panel").first.get_attribute("id") == "study", "the study does not lead the page"
        first = page.locator("#ch-call .panels > section.panel").nth(1)
        assert first.get_attribute("id") == "summary", "the summary does not follow the study"
        text = first.inner_text().lower()
        assert "the deck" in text and "why" in text
        rows = page.locator("#summary table.exec-swaps tbody tr")
        if rows.count() == 0:
            return            # an unmeasured branch with nothing staged: nothing to pair
        assert page.locator("#summary table.exec-swaps td.add a.cardname").count() >= 1, (
            "an incoming card is not a hover-pop link")
        if page.locator("#net-change").count():
            cut = set(page.locator("#net-change li.del a.cardname").all_inner_texts())
            shown = set(page.locator("#summary table.exec-swaps td.del a.cardname").all_inner_texts())
            assert shown <= cut, f"the summary pairs a cut the diff never made: {sorted(shown - cut)}"
    finally:
        page.close()


@requires_branch  # a branch is a pilot artifact, not a fixture: skip, never fail
@pytest.mark.browser
def test_the_branch_page_names_what_left(
        browser, viz_server):
    """THE DELETIONS WERE MISSING FROM THE PAGE ENTIRELY.

    The panel this replaced read `branch.json`'s `staged` array, and a branch
    opened with `deck-branch new --from <list>` stages nothing — it sets the
    whole 99 at once. So a 17-for-17 refactor rendered as "The change (0)" while
    the table beneath it measured all 34 cards. The incoming cards appeared on
    the bill; the outgoing cards appeared NOWHERE. A diff that omits its
    deletions is not a diff.
    """
    page = _branch_page(browser, viz_server, "ur-dragon", A_BRANCH)
    try:
        assert not page.js_errors, page.js_errors
        body = page.inner_text("body").lower()
        if "has not been measured" in body:
            pytest.skip("this branch has no report to diff against")
        assert "the cards" in body
        assert page.locator("li.del").count() >= 1, "removals must be visible"
        assert page.locator("li.add").count() >= 1

        # THE TWO SIDES BALANCE, AND THAT IS THE ASSERTION — not the sentence
        # that explains it. An earlier version of this test pinned the panel's
        # prose, which is a test of my own string; the contract is that the
        # headline counts COPIES, so a legal branch reads the same number twice.
        head = page.inner_text(".panel.netchange .headline")
        nums = [int(n) for n in re.findall(r"\d+", head)]
        assert len(nums) >= 4, head
        out_n, in_n, base_size, size = nums[0], nums[1], nums[2], nums[3]
        assert base_size == size, "a legal branch is the same size as its deck"
        assert out_n == in_n, (
            "copies out must equal copies in on a same-size branch — this read "
            "18 against 21 while counting NAMES, hiding three removals: a basic "
            "cut from four copies to two leaves the name in the list")

        # And those three are RENDERED, not merely counted, or the columns do
        # not add up to the headline.
        if page.locator("li.countrow").count():
            row = page.inner_text("li.countrow")
            assert "\u2192" in row and "cop" in row.lower(), row
    finally:
        page.close()


@pytest.mark.browser
def test_an_unknown_branch_names_the_command_instead_of_breaking(browser, viz_server):
    page = _branch_page(browser, viz_server, "ur-dragon", "no-such-branch")
    try:
        assert not page.js_errors, page.js_errors
        body = page.inner_text("body").lower()
        assert "no such branch" in body
        assert "deck-branch ur-dragon list" in body
    finally:
        page.close()


@requires_branch  # a branch is a pilot artifact, not a fixture: skip, never fail
@pytest.mark.browser
def test_a_pile_can_be_handed_over_without_throwing(browser, viz_server):
    """THE DEFECT THIS EXISTS AGAINST, and it shipped for months.

    `Shell.consider()` read `store.zoneNames()` — a GETTER, not a method — so
    every press raised `TypeError: store.zoneNames is not a function` before it
    reached the server check. `node --check` passes that happily and no test
    touched it. Here the pile is real and `Api.ready` is false (this suite has
    no `/api`), so the correct behaviour is the "needs a local server" notice
    and NO exception.
    """
    page = _branch_page(browser, viz_server, "ur-dragon", A_BRANCH)
    got = page.evaluate("""() => {
      Session.library.add('Sol Ring');
      const names = Session.library.zoneNames;
      let alerted = null;
      window.alert = (m) => { alerted = m; };
      let threw = null;
      try { Shell.consider(); } catch (e) { threw = String(e); }
      return { n: names.length, threw, alerted };
    }""")
    assert got["n"] >= 1, "the pile must be non-empty for the path to run"
    assert got["threw"] is None, got["threw"]
    assert "local server" in (got["alerted"] or "")
    assert page.js_errors == []
    page.close()


@requires_branch  # a branch is a pilot artifact, not a fixture: skip, never fail
@pytest.mark.browser
def test_a_direction_that_does_not_resolve_falls_through_to_the_pilot(browser,
                                                                     viz_server):
    """A REFUSAL IS AN ANSWER, and it must not be a dead end.

    "Make it better" names no axis. The charter returns `axis: null` rather than
    guessing, and the page then shows the doctor's own reasoning and lets the
    pilot type one — which is more than they had before they asked. Driven
    against a stubbed Api, because this suite serves static files with no /api.
    """
    page = _branch_page(browser, viz_server, "ur-dragon", A_BRANCH)
    got = page.evaluate("""async () => {
      const calls = [];
      window.Api = {
        ready: true,
        has: () => true,
        call: (name, p) => {
          calls.push(name);
          if (name === 'branch/objective') return Promise.resolve({id: 'j1'});
          if (name === 'job') return Promise.resolve({
            state: 'done',
            output: 'wrote the file. ' + JSON.stringify({
              objective: {axis: null}, unresolved: 'faster than what?'})});
          return Promise.resolve({});
        }
      };
      window.confirm = () => true;
      let asked = null;
      window.prompt = (m) => { asked = m; return 'kill_by_8 >= 0.30'; };
      const out = await Shell.__objectiveFor('ur-dragon', 'make it better');
      return {out, asked, calls};
    }""")
    assert got["calls"] == ["branch/objective", "job"]
    assert "could not resolve" in got["asked"]
    assert "faster than what?" in got["asked"], "the doctor's reasoning is shown"
    assert got["out"] == "kill_by_8 >= 0.30", "the pilot's own answer is used"
    assert page.js_errors == []
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


def test_the_branch_page_says_its_state_and_what_the_instruments_could_not_see(browser, viz_server):
    """The branch the pilot could not find and could not judge from the page: no
    state for an open branch, no blind spots, no cast proofs, no stacks, no way
    to the siblings. Each renders from the committed artifacts that exist for
    the branch — the page computes nothing, and a panel whose artifact is absent
    is absent. Chosen from the tracked manifest, never pinned to one branch."""
    live = _a_live_branch()
    if live is None:
        pytest.skip("no tracked deck has an OPEN branch")
    slug, row, info = live
    page = _branch_page(browser, viz_server, slug, row["name"])
    try:
        assert not page.js_errors, page.js_errors
        page.wait_for_selector("#branchState", timeout=15000)
        assert "OPEN" in page.text_content("#branchState")
        assert "propose" in page.text_content("#branchState")
        if row.get("blind_spots"):
            assert page.locator("#blindSpots").count() == 1
        if row.get("cast_proofs"):
            page.wait_for_selector("#castProofs", timeout=15000)
            assert page.locator("#castProofs .br-cast").count() >= 1
        if row.get("stacks"):
            page.wait_for_selector("#branchStacks", timeout=15000)
            assert row["stacks"][0]["id"] in page.text_content("#branchStacks")
        if len(info.get("branches") or []) > 1:
            page.wait_for_selector("#branchSiblings", timeout=15000)
            assert row["name"] not in page.text_content("#branchSiblings")
    finally:
        page.close()


@requires_branch  # a branch is a pilot artifact, not a fixture: skip, never fail
@pytest.mark.browser
def test_the_branch_page_has_the_anchors_the_workbench_links_and_lands_on_them(browser, viz_server):
    """The workbench's branch block links `#measured` (the delta table),
    `#net-change` (the diff) and `#swaps` (the proposals). Both pages build their
    panels after a fetch, so the browser's own jump fired before the element
    existed and every deep link landed at the top; `scrollToHash` runs after the
    render. Re-introduce the bug (drop the call) and the second assertion fails."""
    page = _branch_page(browser, viz_server, "ur-dragon", A_BRANCH)
    try:
        for anchor in ("measured", "net-change", "swaps"):
            assert page.locator("#" + anchor).count() == 1, anchor
        assert page.js_errors == []
    finally:
        page.close()
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    try:
        page.goto(f"{viz_server}/viz/branch.html?deck=ur-dragon&branch={A_BRANCH}#measured")
        idle(page)
        assert page.evaluate("window.scrollY") > 0, "the page did not scroll to #measured after its render"
    finally:
        page.close()


_NC_REPORT = {
    "slug": "ur-dragon", "branch": "x", "decklist_sha256": "0" * 64, "staged": [],
    "harness": {"iterations": 10000, "seed": 1, "model_version": "m"},
    "design": {"exploratory_rows": 2, "primary": None},
    "table": [
        {"measure": "killed by T6", "unit": "rate", "better_is": "higher", "champion": 0.083,
         "branch": 0.161, "delta": 0.078, "ci95_diff": [0.07, 0.087], "mde": 0.01, "verdict": "better",
         "reads_as": "8 games in 100 -> 16 in 100", "what": "Share of games.", "why_we_care": "A clock."},
        {"measure": "a metric from next year", "unit": "mean", "better_is": "lower", "champion": 12.0,
         "branch": 12.5, "delta": 0.5, "ci95_diff": [-0.1, 1.1], "mde": 0.9, "verdict": "noise",
         "reads_as": "no call", "what": "New.", "why_we_care": "New."},
    ],
    "recommendation": {"state": "do not merge", "because": "Guard missed. WARNING — a Forge run lost.",
                       "forge_warning": "WARNING — a Forge run lost.", "notes": [],
                       "reward": [], "risk": [], "rose": [], "fell": [], "no_call": []},
    "objective": {"axis": "stall", "op": "<=", "value": 0.11, "why": "because"},
    "objective_grade": {"state": "not met", "reading": 0.145, "why": "it did not"},
    "blind_spots": [{"class": "removal", "cards": ["Swords to Plowshares"], "headline": "1 card",
                     "why": "a removal spell has nothing to remove"}],
    "limits": ["The goldfish has no opponents."],
    "mana": {"available": True, "note": "Deterministic.", "lands": [35, 36], "enters_tapped_always": [4, 2],
             "life": [{"recurring_per_tap_cycle": 1, "one_time_on_entry": 9},
                      {"recurring_per_tap_cycle": 4, "one_time_on_entry": 9}],
             "on_curve": {"G": [0.82, 0.88], "U": [None, None]},
             "colours": [{"colour": "U", "target": [0, 0], "have": [0, 0], "gap": [0, 0], "delta": 0},
                         {"colour": "G", "target": [36, 36], "have": [30, 34], "gap": [-6, -2], "delta": 4}]},
    "forge": {"available": False, "why": "no Forge run"},
    "changes": {},
    "definitions": {"derived": [{"name": "PRIMARY", "what": "The sixteen rows are a family.", "why": "y"}],
                    "forge_endpoints": {}, "rows": {}},
}


def _report_page(browser, viz_server, body, status=200, width=1440):
    """The branch page over a stated report, so the shapes no tracked branch has
    today (a Forge warning, a metric nobody grouped, a broken file) are reachable."""
    page = browser.new_page(viewport={"width": width, "height": 900})
    page.js_errors = []
    page.on("pageerror", lambda e: page.js_errors.append(str(e)))
    page.route("**/net_change.json*", lambda r: r.fulfill(
        status=status, content_type="application/json", body=body))
    page.goto(f"{viz_server}/viz/branch.html?deck=ur-dragon&branch={A_BRANCH}")
    idle(page)
    # The page folds past the call and closes the goldfish (2026-11-22); these tests
    # read the report's text, so every <details> is opened first, the goldfish's own included.
    page.evaluate("document.querySelectorAll('details').forEach(d => { d.open = true; })")
    return page


@requires_branch
@pytest.mark.browser
def test_the_report_draws_what_it_used_to_drop(browser, viz_server):
    """Four things the artifact carried and the page lost, each proven by the old
    code: blind spots read `what`/`detail` where the keys are `headline`/`why`;
    `nc.mana` was never read; a guard row printed as the headline; a colour the
    deck never touches printed as a row."""
    page = _report_page(browser, viz_server, json.dumps(_NC_REPORT))
    try:
        assert page.js_errors == []
        assert "a removal spell has nothing to remove" in page.inner_text("#blindSpots")
        mana = page.inner_text("#mana")
        assert "green" in mana.lower() and "88%" in mana and "blue" not in mana.lower()
        assert "Guard not met" in page.inner_text("#guard") and "0.145" in page.inner_text("#guard")
        assert "guard not met" in page.inner_text("#verdict") and "do not merge" not in page.inner_text("#verdict")
        assert "The 2 rows are a family" in page.inner_text("#definitions")
    finally:
        page.close()


@requires_branch
@pytest.mark.browser
def test_a_row_nobody_grouped_is_still_on_the_page(browser, viz_server):
    page = _report_page(browser, viz_server, json.dumps(_NC_REPORT))
    try:
        table = page.inner_text("#measured table")
        assert "OTHER" in table.upper() and "a metric from next year" in table
        assert "8.3%" in table and "16.1%" in table and "+7.8 pts" in table, table
        assert "No call" in table
        assert page.locator("#measured .c-better b").inner_text() == "1"
    finally:
        page.close()


@requires_branch
@pytest.mark.browser
def test_a_broken_report_is_not_called_unmeasured(browser, viz_server):
    """It told the pilot to measure a branch that had been measured."""
    page = _report_page(browser, viz_server, "{not json")
    try:
        body = page.inner_text("#panels").lower()
        assert "could not be read" in body and "will not parse" in body
        assert "has not been measured" not in body
    finally:
        page.close()


def test_the_page_groups_rows_exactly_as_the_terminal_does():
    """Two copies of one list, and only this holds them equal."""
    from manamap.pilot import net_change
    js = pathlib.Path("viz/js/branch-view.js").read_text()
    block = js[js.index("var ROW_GROUPS = ["):js.index("var READ_WORD")]
    for title, names in net_change.ROW_GROUPS:
        at = block.index(f"['{title}'")
        for name in names:
            quoted = f"'{name}'" if "'" not in name else f'"{name}"'
            assert quoted in block[at:], f"{name} is not under {title} on the page"
    grouped = sum(len(n) for _, n in net_change.ROW_GROUPS)
    # every row label's wording, so a label on the page and in no group is counted
    # ("big bodies": Meren's turns-saved row, 2026-11-07)
    assert len(re.findall(r"""['"][^'"]*(?:@T|by T|mulliganed|in a row|keep rule|big bodies)[^'"]*['"]""",
                          block)) == grouped
