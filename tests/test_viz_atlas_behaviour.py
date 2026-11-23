"""Browser behaviour: the card atlas and Build mode (viz/index.html).

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


def test_map_boots_clean(page, corpus_count):
    assert page.evaluate("MM.allData.length") == corpus_count
    traces = page.evaluate("MM.mapRenderer.layers.length")
    assert traces >= 6, "base scatter did not render"
    assert page.js_errors == [], f"console/page errors at boot: {page.js_errors}"


def test_drill_renders_its_layout(page):
    """The regression gate.

    Against the broken revision: `drillTraces` is 0, the breadcrumb is hidden, and a
    ReferenceError lands in `page.js_errors`. All three assertions fail.
    """
    result = page.evaluate("""async () => {
        const rd = await MM.getRegionData();
        const reg = rd.regions.find(r => r.level === 1 && r.count > 100);
        await Drill.enterRegion(reg.id);
        await MM.whenIdle();
        const gd = document.getElementById('plot');
        const dt = MM.mapRenderer.layers.find(t => t._isDrill);
        let moved = 0;
        if (dt) {
            for (let i = 0; i < dt.customdata.length; i++) {
                const w = MM.allData[dt.customdata[i]];
                if (Math.hypot(dt.x[i] - w.x, dt.y[i] - w.y) > 0.05) moved++;
            }
        }
        return {
            region: reg.label,
            expected: reg.count,
            drillTraces: MM.mapRenderer.layers.filter(t => t._isDrill).length,
            points: dt ? dt.x.length : 0,
            movedFraction: dt ? moved / dt.customdata.length : 0,
            barVisible: document.getElementById('drillBar').style.display !== 'none',
            active: Drill.isActive(),
        };
    }""")

    assert page.js_errors == [], f"drill threw: {page.js_errors}"
    assert result["drillTraces"] == 1, "the drill layout did not render"
    assert result["points"] == result["expected"], (
        f"drilled {result['points']} of {result['expected']} cards in {result['region']}")
    assert result["movedFraction"] > 0.9, (
        "points did not move — the layout was never recomputed from the embeddings")
    assert result["barVisible"], "no breadcrumb, so nothing says the coordinates are local"


def test_walk_survives_a_round_trip_through_explore(page):
    """Leaving for the map and coming back must not lose the graph.

    Two separate faults met here. `exit()` set an inline `display: none` on the canvas
    that nothing ever cleared, so re-entry rebuilt the graph correctly into a 0x0 hidden
    element — 79 nodes, 156 links, the right status line, and a blank screen. And the
    graph was discarded on exit, so a walk you spent minutes growing had to be rebuilt
    just because you glanced at the map.
    """
    r = page.evaluate("""async () => {
        const setMode = m => { document.getElementById('modeSelect').value = m; MM.setMode(m); };
        const cv = () => document.getElementById('forceCanvas');
        const ink = () => {
            const c = cv();
            if (!c || !c.width) return 0;
            const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
            let l = 0, n = 0;
            for (let i = 3; i < d.length; i += 4 * 30) { n++; if (d[i] > 20) l++; }
            return 100 * l / n;
        };

        setMode('discover');
        await MM.whenIdle();
        await Discovery.loadDeck('goblin-storm');
        await MM.whenIdle();
        const i = MM.allData.findIndex(d => d.n === 'Past in Flames');
        if (i >= 0) Force.focusCard(i);
        await MM.whenIdle();
        const before = {nodes: Force.nodeCount, trail: Force.trailLength,
                        w: cv().clientWidth, ink: ink()};

        setMode('explore');
        await MM.whenIdle();
        const explore = {traces: MM.mapRenderer.layers.length,
                         inlineDisplay: cv().style.display || ''};

        setMode('discover');
        await MM.whenIdle();
        const after = {nodes: Force.nodeCount, trail: Force.trailLength,
                       w: cv().clientWidth, ink: ink()};
        return {before, explore, after};
    }""")
    assert page.js_errors == []
    assert r["explore"]["traces"] >= 6, "the map did not come back"
    assert r["explore"]["inlineDisplay"] == "", (
        "exit() must not set an inline display — the CSS class owns visibility, and an "
        "inline hide survives re-entry")
    assert r["after"]["nodes"] == r["before"]["nodes"], "the graph was discarded on exit"
    assert r["after"]["trail"] == r["before"]["trail"], "the trail was discarded on exit"
    assert r["after"]["w"] > 400, f"canvas came back at {r['after']['w']}px wide"
    assert r["after"]["ink"] > 0.5, "the graph is not being drawn after re-entry"


def test_canvas_click_selects_the_card_under_the_pointer(canvas_page):
    r = canvas_page.evaluate("""async () => {
        const c = document.querySelector('.map-canvas');
        const rect = c.getBoundingClientRect();
        // Aim at a real card via the renderer's own projection. Guessing a pixel depends
        // on the viewport size and lands on empty space as often as not.
        const i = MM.allData.findIndex(d => d.n === 'Sol Ring');
        const p = MM.mapRenderer.dataToPixel(MM.allData[i].x, MM.allData[i].y);
        c.dispatchEvent(new MouseEvent('click',
            {bubbles: true, clientX: rect.left + p[0], clientY: rect.top + p[1]}));
        await MM.whenIdle();
        const rows = MM.selectedRows();
        return {n: rows.length, name: rows.length ? MM.allData[rows[0]].n : null,
                aimedAt: MM.allData[i].n};
    }""")
    assert canvas_page.js_errors == []
    assert r["n"] == 1, "a click on the canvas selected nothing"
    # Not necessarily the card aimed at: the quadtree returns the nearest within the pick
    # radius, and the map is dense. What must hold is that a click resolves to a real card.
    assert r["name"], "selected a row with no card behind it"


def test_landing_paints_a_card_without_the_projection(discover_page):
    """The reframe's whole claim.

    Boot used to block on 12.9 MB of projection before a single pixel appeared, and the
    first click then needed 16.8 MB of incompressible float32 on top. The landing now
    renders from viz_index alone — this asserts the card is *there* while allData may
    still be empty, which is the only way to prove the dependency is gone.
    """
    r = discover_page.evaluate("""() => ({
        mode: document.getElementById('modeSelect').value,
        card: Discovery.index[Discovery.current].n,
        panel: !!document.querySelector('.discover-filters'),
        cardInPanel: document.getElementById('deckInner').textContent
                       .includes('Craterhoof Behemoth'),
        stuckPopup: (() => {
            const p = document.querySelector('.card-popup');
            return !!p && p.style.display === 'block';
        })(),
        nodes: Force.nodeCount,
    })""")
    assert discover_page.js_errors == []
    assert r["mode"] == "discover"
    assert r["card"] == "Craterhoof Behemoth", "?card= deep link was not honoured"
    assert r["panel"], "the discovery controls did not render"
    assert r["nodes"] == 1, "the landing card should be the graph's single seed"
    assert r["cardInPanel"], "the landing card must be readable in the panel"
    assert not r["stuckPopup"], (
        "a card is floating over the graph with nothing hovered — the popup is for hover "
        "only, and a persistent one cannot be dismissed and covers the points behind it"
    )


def test_a_real_moxfield_export_imports_completely(discover_page):
    """The end-to-end claim: paste a decklist, get your deck as a graph.

    Checked against a deck the CLI already ingested, so `cards.json` is an independent
    answer to what that list contains. Resolution goes through `viz_index`, NOT
    `data/decks/index.json` — Deck Lens refuses any slug it does not already know, and an
    imported deck has no slug and never will.
    """
    r = discover_page.evaluate("""async () => {
        const text = await (await fetch('../data/decks/edgar-vampires/decklist.txt')).text();
        const doc = await (await fetch('../data/decks/edgar-vampires/cards.json')).json();
        const res = Discovery.importText(text);
        await MM.whenIdle();
        const expected = new Set(doc.cards.map(c => c.name));
        const got = new Set(Discovery.library.names());
        const missingFromGraph = [...expected].filter(n => !got.has(n));
        return {
            resolved: res.resolved,
            unresolved: res.missing,
            nodes: Force.nodeCount,
            commander: res.commander >= 0 ? Discovery.index[res.commander].n : null,
            missingFromGraph: missingFromGraph.slice(0, 10),
            missingCount: missingFromGraph.length,
        };
    }""")
    assert discover_page.js_errors == []
    assert r["unresolved"] == [], f"names the index could not resolve: {r['unresolved']}"
    assert r["commander"] == "Edgar Markov", "the commander was not identified"
    assert r["missingCount"] == 0, f"cards in the deck but not the graph: {r['missingFromGraph']}"
    assert r["nodes"] == r["resolved"], (
        f"{r['nodes']} nodes from {r['resolved']} cards — the import grew the deck, which "
        f"is what focusCard did before pinCard existed"
    )


def test_a_checked_in_deck_loads_with_its_commander(discover_page):
    """Load one of the published decks by slug and walk out from it.

    Distinct from a pasted import in the way that matters: the manifest carries a KNOWN
    commander, so it is ringed and centred rather than inferred from a `*CMDR*` marker.
    """
    r = discover_page.evaluate("""async () => {
        const res = await Discovery.loadDeck('goblin-storm');
        await MM.whenIdle();
        return {
            res: res,
            commanderName: res && res.commander >= 0
                ? Discovery.index[res.commander].n : null,
            panel: (document.querySelector('#deckInner .lens-title') || {}).textContent,
            membership: Force.membership(),
            tray: Discovery.library.list.length,
            decks: Discovery.decks.map(d => d.slug),
        };
    }""")
    assert discover_page.js_errors == []
    assert "goblin-storm" in r["decks"] and len(r["decks"]) >= 7
    assert r["res"]["missing"] == [], f"unresolved deck cards: {r['res']['missing']}"
    assert r["commanderName"] == "Zada, Hedron Grinder"
    assert r["panel"] == "Zada, Hedron Grinder", "the panel did not open on the commander"
    assert r["membership"]["commander"] == 1, "exactly one card should be ringed as commander"
    assert r["membership"]["deck"] > 50
    assert r["membership"]["explored"] == 0, "a freshly loaded deck has nothing explored yet"
    assert r["tray"] == r["membership"]["deck"], "the loaded deck should fill the tray"


def test_loading_a_deck_replaces_the_previous_one(discover_page):
    """Two decks at once would be an unreadable pile, and the membership flags would be
    ambiguous — a card in both decks could only be one colour."""
    r = discover_page.evaluate("""async () => {
        await Discovery.loadDeck('goblin-storm');
        await MM.whenIdle();
        const first = {m: Force.membership(),
                       cmdr: Discovery.index[Discovery.current].n};
        await Discovery.loadDeck('heliod');
        await MM.whenIdle();
        return {first: first, second: {m: Force.membership(),
                                       cmdr: Discovery.index[Discovery.current].n}};
    }""")
    assert discover_page.js_errors == []
    assert r["first"]["cmdr"] == "Zada, Hedron Grinder"
    assert r["second"]["cmdr"] != r["first"]["cmdr"], "the second deck did not take over"
    assert r["second"]["m"]["commander"] == 1, "two commanders are ringed at once"
    assert r["second"]["m"]["explored"] == 0, "the previous deck leaked in as explored cards"


def test_a_relation_always_grows_the_graph(discover_page):
    """One button, one behaviour, everywhere: it grows what you are holding.

    This control has been rewritten twice, and the history is the point. It first FORKED —
    graph modes grew the graph, Explore opened a linear browse set — on the reasoning that
    a scatter plot cannot grow. One control meaning two things is why the atlas felt dead.
    Then it CARRIED YOU OUT: clicking a relation in Explore switched to Discover. Better,
    but it still meant the map could only ever hand you off, never respond.

    Now it grows the graph wherever you are. From Explore that means the constellation and
    its arcs appear on the map and you stay put; from the graph it branches as before. What
    is invariant, and all this test asserts, is that the graph is strictly larger and
    nothing was thrown away.
    """
    r = discover_page.evaluate("""async () => {
        const row = Discovery.rowByName("Ashnod's Altar");

        Discovery.show(row);
        await MM.whenIdle();
        const before = Session.size();
        MM.relate(row, 'synergy');
        await MM.whenIdle();
        const fromDiscover = {grew: Session.size() > before, held: Session.rows().slice()};

        // From the atlas: same control, same direction of travel — bigger, never smaller.
        document.getElementById('modeSelect').value = 'explore';
        MM.setMode('explore');
        await MM.whenIdle();
        const beforeExplore = Session.size();
        MM.relate(row, 'similar');
        await MM.whenIdle();
        const after = Session.rows();
        return {
            fromDiscover: {grew: fromDiscover.grew},
            mode: document.getElementById('modeSelect').value,
            grewAgain: Session.size() > beforeExplore,
            keptEverything: fromDiscover.held.every(x => after.indexOf(x) !== -1),
            stillHasIt: Session.has(row),
            noBrowseSet: !MM.browseSet,
        };
    }""")
    assert discover_page.js_errors == []
    assert r["fromDiscover"]["grew"], "a relation in Discover did not grow the graph"
    assert r["mode"] == "explore", "growing from the atlas left the atlas"
    assert r["grewAgain"], "a relation in Explore did not grow the graph"
    assert r["keptEverything"], "growing from the atlas dropped cards already held"
    assert r["stillHasIt"] and r["noBrowseSet"]


def test_growing_from_the_atlas_never_destroys_the_graph(discover_page):
    """Growing must not be able to delete.

    `MM.relate` used to call `Discovery.show(row)` for any card not already on the walk,
    and `show` calls `Force.newWalk(true)`, which empties `nodes`, `links` and `trail`.
    So building a graph, switching to Explore, and clicking a relation on some card you
    had not walked to threw the whole graph away without a word. `discovery.js` carries a
    comment warning about that exact hazard on `focus()`; the Explore path took the
    destructive branch anyway.

    The seed-vs-adopt rule is the fix: reseed only when there is nothing to lose.
    """
    r = discover_page.evaluate("""async () => {
        // Build a real graph by branching a few times.
        Discovery.show(Discovery.rowByName("Ashnod's Altar"));
        await MM.whenIdle();
        for (let i = 0; i < 3; i++) {
            const rows = Force.rows();
            MM.relate(rows[rows.length - 1], 'similar');
            await MM.whenIdle();
        }
        const before = Force.rows().slice();
        if (before.length < 8) return {tooSmall: before.length};

        // Find a card that is NOT on the graph — the destructive case.
        let outsider = -1;
        for (let i = 0; i < Discovery.index.length && outsider < 0; i += 97) {
            if (!Force.hasRow(i) && Discovery.counts(i).similar > 0) outsider = i;
        }

        // Go to the atlas and grow from it, exactly as a user would.
        document.getElementById('modeSelect').value = 'explore';
        MM.setMode('explore');
        await MM.whenIdle();
        MM.relate(outsider, 'similar');
        await MM.whenIdle();

        const after = Force.rows();
        const kept = before.filter(x => after.indexOf(x) !== -1);
        return {
            before: before.length, after: after.length,
            kept: kept.length, lost: before.length - kept.length,
            outsiderJoined: Force.hasRow(outsider),
            grew: after.length > before.length,
        };
    }""")
    assert discover_page.js_errors == []
    assert not r.get("tooSmall"), f"could not build a graph to test with: {r}"
    assert r["lost"] == 0, (
        f"growing from the atlas destroyed {r['lost']} of {r['before']} nodes"
    )
    assert r["outsiderJoined"], "the card grown from did not join the graph"
    assert r["grew"], "the graph did not grow"


def test_the_brief_is_the_schema_build_deck_reads(discover_page):
    """The export must BE `brief.json`, not a description of one.

    It used to emit `{generated_by, card_count, cards, commander_candidates, next_step}`,
    which is none of the shape `pilot/build_deck.py:load_brief` requires — so every export
    had to be hand-translated before the loop could run, and the browser's own answers
    (which card IS the commander, what you kept versus what you found) were thrown away
    and re-derived by the agent.

    Two rules come from the Python side and are honoured rather than guessed: colour
    identity is DERIVED from the commander and never authored, so it rides in the
    provenance block the builder ignores; and budget is unsupported because prices are
    stripped from the card data.

    Round-tripped for real: `load_brief` accepts this document unchanged, tolerates the
    `_manamap` and `next_step` extras, and its derived identity for Edgar Markov ({B,R,W})
    matches what the browser emits.
    """
    # Reads MM.allData, which this fixture deliberately does not wait for.
    await_projection(discover_page)
    r = discover_page.evaluate("""async () => {
        const edgar = MM.allData.findIndex(d => d.n === 'Edgar Markov');

        // With no commander the brief must refuse rather than emit an unusable document.
        const blocked = Discovery.brief();

        Discovery.show(edgar);
        await MM.whenIdle();
        MM.setCommander(edgar);
        await MM.whenIdle();
        for (let i = 0; i < 3; i++) {
            MM.relate(Force.rows()[Force.nodeCount - 1], 'similar');
            await MM.whenIdle();
        }
        Force.rows().filter(r => r !== edgar).slice(0, 6).forEach(function (r) {
            if (!Session.library.has(r)) Session.library.toggle(r);
        });
        const b = Discovery.brief();
        return {
            blocked: {commander: blocked.commander, hasBlocked: !!blocked._manamap.blocked},
            keys: Object.keys(b),
            slug: b.slug, commander: b.commander, bracket: b.bracket,
            mustInclude: b.must_include, mustExclude: b.must_exclude,
            ci: b._manamap.colour_identity,
            budget: b._manamap.budget,
            sources: Array.from(new Set(b._manamap.pool.map(p => p.source))).sort(),
            commanderInMustInclude: b.must_include.indexOf('Edgar Markov') !== -1,
        };
    }""")
    assert discover_page.js_errors == []

    # Blocked without a commander — `build-deck` cannot start without one.
    assert r["blocked"]["commander"] is None and r["blocked"]["hasBlocked"]

    # The three keys `load_brief` requires, in the shape it requires them.
    for key in ("slug", "commander", "bracket", "must_include", "must_exclude"):
        assert key in r["keys"], f"brief is missing {key} — load_brief will reject it"
    assert r["slug"] == "edgar-markov", f"slug not derived from the commander: {r['slug']}"
    assert r["commander"] == "Edgar Markov"
    assert r["bracket"] in range(1, 6), "bracket must be 1-5"
    assert isinstance(r["mustInclude"], list) and isinstance(r["mustExclude"], list)
    assert len(r["mustInclude"]) == 6, "the library is what must_include means"
    assert not r["commanderInMustInclude"], (
        "the commander occupies its own slot and must not also be in must_include"
    )

    # Derived, informational, and correct — Edgar Markov is Mardu.
    assert sorted(r["ci"]) == ["B", "R", "W"]
    assert "unsupported" in r["budget"], "budget must say it is unsupported, not guess"

    # What you brought vs what you found: the graph knows, so the agent should not guess.
    assert r["sources"] == ["found", "kept"], f"pool provenance lost: {r['sources']}"


def _build_page(browser, viz_server, slug, tab="deck", route=None):
    """Build with `slug` loaded, on `tab`. The panel is tabs since 2026-10-28 and a
    sleeved deck opens on Change; the verified lines and the role budget are Deck's,
    the curve is Numbers'."""
    from conftest_viz import _boot
    page = _boot(browser, viz_server, f"?deck={slug}&tab={tab}", route=route)
    page.wait_for_function("() => window.Force && Force.nodeCount > 0", timeout=BOOT_TIMEOUT_MS)
    idle(page)
    return page


def _a_deck_with_a_drawable_line():
    import json
    from manamap.config import DECKS_DIR
    manifest = json.loads((DECKS_DIR / "index.json").read_text())
    for d in manifest["decks"]:
        if any(len(v) >= 2 for v in (d.get("stack_cards") or {}).values()):
            return d["slug"]
    return None


def test_verified_lines_become_edges_in_the_graph(browser, viz_server):
    """The graph's links come only from embedding similarity, so two combo pieces that
    are not near-neighbours had no edge at all. A verified line is a claim about which
    cards talk to each other; the graph must be able to say it."""
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck with a drawable verified line")
    page = _build_page(browser, viz_server, slug)
    try:
        assert page.js_errors == []
        assert page.evaluate("Force.verifiedLinkCount") > 0
    finally:
        page.close()


def test_clicking_a_verified_line_spotlights_it(browser, viz_server):
    """Click the line: its rows go under a spotlight and the row marks itself."""
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck with a drawable verified line")
    page = _build_page(browser, viz_server, slug)
    try:
        assert page.evaluate("Build.activeLine") == -1
        page.click(".lens-line")
        idle(page)

        assert page.evaluate("Build.activeLine") == 0
        assert page.evaluate("Force.activeLine") is not None
        assert page.evaluate("Force.lineRowCount") >= 2
        assert page.evaluate(
            "document.querySelector('.lens-line').classList.contains('is-on')")
        assert page.js_errors == []
    finally:
        page.close()


def test_engine_lines_give_verified_edges_a_direction(browser, viz_server):
    """A clique has no arrows; `engine.json` is the one artifact that knows.

    A verified line becomes a clique over the cards its stack names, so
    `{source, target}` is whichever order the pair was built in — drawing an
    arrow on that would be array order wearing a claim. The engine model
    declares each line as `from -> to` across two of eight stages with a
    `carries` noun, written by an engineer and attacked by a critic.

    So only the pairs that actually SPAN the two stages get an arrowhead. A
    pair sitting wholly inside one stage stays undirected, which is honest:
    for that pair the direction genuinely is not known.
    """
    page = _build_page(browser, viz_server, "ur-dragon")
    try:
        assert page.js_errors == [], page.js_errors
        links = page.evaluate("Force.links()")
        directed = [l for l in links if l.get("dir")]
        assert directed, "no verified edge picked up a direction from engine.json"
        # Every directed edge names what it moves — that is the whole point of
        # the arrow, and an unlabelled one would just be a prettier line.
        assert all(l.get("carries") for l in directed), directed
        # And most edges stay undirected: the arrow is the exception, earned.
        assert len(directed) < len(links)
    finally:
        page.close()


def test_a_stack_carrying_two_engine_lines_keeps_both_nouns(browser, viz_server):
    """ur-dragon's stack 002 is cited by two lines — `bodies` and `triggers`.

    Taking the first match would silently drop half of what that board proves.
    They agree on direction, so the nouns are joined; if two lines citing one
    stack DISAGREED on direction, no arrowhead is drawn at all, because a pair
    pointing two ways is a pair whose direction is not a fact.
    """
    # The SHAPE is fed directly (2026-11-22): engine.json v2 stamps no stack with two
    # directed lines today, and the behaviour must not depend on that. A copy of the
    # tracked record with the first directed stamped line duplicated under a second noun.
    import json
    from manamap.config import DECKS_DIR
    doc = json.loads((DECKS_DIR / "ur-dragon" / "engine.json").read_text())
    line = next((l for l in doc["lines"] if l.get("verified_by") and l.get("from") and l.get("to")
                 and l["from"] != l["to"]), None)
    if line is None:
        pytest.skip("ur-dragon has no directed stack-stamped line")
    doc["lines"].append(dict(line, id="LX", carries="a second noun"))
    body = json.dumps(doc)
    page = _build_page(browser, viz_server, "ur-dragon",
                       route=("**/ur-dragon/engine.json*", body))
    try:
        carries = page.evaluate(
            "[...new Set(Force.links().filter(l => l.dir).map(l => l.carries))]")
        assert any(" · " in (c or "") for c in carries), carries
    finally:
        page.close()


def test_the_prose_is_the_artifact_verbatim(browser, viz_server):
    """A checker read these words.

    The panel may choose which of the three sources to print and may leave one
    out — it must never re-word one. Summarising a resolution in the browser
    puts a ✓ over prose no checker saw, which is the same mistake as editing a
    resolution's step text to fix a stale cross-reference.
    """
    slug = _a_deck_with_a_drawable_line()
    if not slug:
        pytest.skip("no deck with a drawable verified line")
    page = _build_page(browser, viz_server, slug)
    try:
        page.click(".lens-line")
        idle(page)
        shown = page.locator(".lens-line-prose").inner_text()

        # Pull the same line's artifact prose out of Build's own state, then
        # assert the rendered block contains it whole.
        src = page.evaluate("Build.__lineProse ? Build.__lineProse(0) : null")
        assert src, "Build did not expose the open line's prose"
        for key in ("note", "answer", "summary"):
            val = src.get(key)
            if not val:
                continue
            # innerText collapses the source's newlines; compare a run that has
            # none rather than weakening the assertion to a prefix.
            probe = max(val.split("\n"), key=len).strip()
            assert probe and probe in shown, f"{key} was not printed verbatim"
        assert page.js_errors == []
    finally:
        page.close()


def test_a_walk_can_be_linked_to(browser, viz_server):
    """`?cards=` is the shareable form, and it seeds ONCE.

    Seeding twice — show() then a reseed — draws the first card, throws it away
    and draws the set, which reads as a flicker on the first frame the pilot
    ever sees.
    """
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        page.goto(f"{viz_server}/viz/index.html"
                  "?cards=1)%20Miirym,%20Sentinel%20Wyrm,%202)%20Sol%20Ring")
        page.wait_for_function(
            "() => window.Force && Force.nodeCount >= 2", timeout=30000)
        assert page.evaluate("MM.mode") == "discover"
        assert page.evaluate("Force.nodeCount") == 2
        assert errors == []
    finally:
        page.close()


def test_the_library_survives_a_reload(discover_page, viz_server):
    """The whole point of persisting it."""
    page = discover_page
    kept = page.evaluate("""async () => {
        const rows = ['Sol Ring', 'Command Tower', 'Rhystic Study']
            .map(Discovery.rowByName).filter(r => r >= 0);
        rows.forEach(r => { if (!Session.library.has(r)) Session.library.toggle(r); });
        return Discovery.library.names();
    }""")
    assert len(kept) == 3

    page.reload()
    page.wait_for_function(
        "() => window.Discovery && Discovery.isReady() && Discovery.current >= 0",
        timeout=30000)
    back = page.evaluate("Discovery.library.names()")
    assert sorted(back) == sorted(kept), f"library did not survive: {back}"
    assert page.js_errors == []


def test_the_library_stores_names_not_row_indices(discover_page):
    """THE bug that deleted the previous attempt at this.

    `localStorage['manamap-deck']` stored raw positional row indices with no
    schema version. A Scryfall refresh reorders `cards.csv`, every index shifts,
    and a saved deck silently reinterprets as DIFFERENT CARDS — no error, no
    warning, a plausible wrong answer.

    Asserted against the stored bytes rather than the behaviour, because the
    behaviour only diverges after a corpus refresh and by then it is too late.
    """
    page = discover_page
    raw = page.evaluate("""() => {
        Session.library.clear();
        const r = Discovery.rowByName('Sol Ring');
        Session.library.toggle(r);
        return {stored: localStorage.getItem('manamap-library'), row: r};
    }""")
    doc = json.loads(raw["stored"])
    # The VERSION matters, not the number: its absence is the exact omission
    # that sank the previous attempt, because a document with no schema cannot
    # be upgraded — only guessed at. v2 added piles and upgraded v1 in place.
    assert doc["v"], "no schema version — the exact omission that sank the last one"
    assert doc["v"] >= 2
    on_disk = _names_on_disk(doc)
    assert on_disk == ["Sol Ring"], f"stored {on_disk!r}, expected names"
    assert raw["row"] not in on_disk, "a row index reached the store"


def test_a_reference_deck_is_not_loaded_as_your_deck(browser, viz_server, tmp_path):
    """THE distinction step 10 rests on.

    A reference deck is somebody else's list, fetched by the CLI because the
    page cannot reach EDHREC. Loading it with `opts.deck` would ring a
    commander, ink the cards as yours and put all eighty in your library — the
    opposite of harvesting a few out of another brew. It seeds the graph and
    nothing else; the library only grows when you Keep something.
    """
    import json as _json
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    ref_dir = root / "data" / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    slug = "zz-test-reference"
    (ref_dir / f"{slug}.json").write_text(_json.dumps({
        "slug": slug, "commander": "Test Commander",
        "cards": ["Sol Ring", "Arcane Signet", "Command Tower",
                  "Rhystic Study", "Cyclonic Rift"],
        "unresolved": [], "source": "test",
    }))
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        page.goto(f"{viz_server}/viz/index.html?ref={slug}")
        page.wait_for_function("() => window.Force && Force.nodeCount >= 4",
                               timeout=BOOT_TIMEOUT_MS)
        state = page.evaluate("""() => {
            Session.library.clear();
            return {nodes: Force.nodeCount, library: Session.library.size,
                    commander: Session.commander};
        }""")
        assert state["nodes"] >= 4, "the reference deck did not seed the graph"
        assert state["library"] == 0, (
            "loading a reference deck put its cards in your library — that is "
            "loading it AS your deck, not harvesting from it")
        assert state["commander"] == -1, "a reference deck claimed the commander ring"

        # …and harvesting one card is exactly one card.
        grew = page.evaluate("""() => {
            const before = Session.library.size;
            Session.library.toggle(Force.rows()[0]);
            return {before: before, after: Session.library.size};
        }""")
        assert grew["after"] == grew["before"] + 1
        assert errors == []
    finally:
        page.close()
        (ref_dir / f"{slug}.json").unlink(missing_ok=True)


def test_the_mode_select_stays_the_one_source_of_truth(page):
    """The tabs are a VIEW of `modeSelect`, never a second answer.

    `?mode=` applies through it and the browser suite reads it; two places
    holding "which mode is current" is the bug this repo keeps undoing.
    """
    page.evaluate("document.querySelector('.mode-tab[data-mode=\\'build\\']').click()")
    idle(page)
    assert page.evaluate("document.getElementById('modeSelect').value") == "build"
    assert page.evaluate("MM.mode") == "build"
    marked = page.eval_on_selector_all(
        ".mode-tab.is-on", "els => els.map(e => e.getAttribute('data-mode'))")
    assert marked == ["build"], marked
    assert page.js_errors == []


def test_agent_affordances_are_absent_not_broken_without_a_server(browser, viz_server):
    """The PRD's rule for the static build, and the answer to `CLAUDE.md`'s
    objection that a local bridge makes two products.

    It does make two, and the page is the thing that says which one you are
    holding. The browser suite's `viz_server` is a plain static server with no
    `/api`, which is exactly the deployed shape — so this test runs in the
    condition it is about rather than simulating it.
    """
    # `_build_page` is the helper every other Build test uses — it waits for the
    # graph to have nodes, which is the signal that a deck actually loaded.
    # Hand-rolling the wait here read "Loading artifacts…" and reported a
    # missing message that had simply not been written yet.
    page = _build_page(browser, viz_server, "heliod", tab="change")
    errors = page.js_errors
    try:
        page.wait_for_function("() => window.Api && Api.probed", timeout=BOOT_TIMEOUT_MS)
        state = page.evaluate("() => ({ready: Api.ready, reason: Api.reason})")
        assert state["ready"] is False
        assert "manamap serve" in (state["reason"] or ""), state
        body = page.inner_text("#deckInner")
        assert "manamap serve" in body, (
            "the static build did not say what is missing — absent is fine, "
            "silent is not")
        assert page.query_selector("#bdAsk") is None, (
            "an Ask box was offered with no server behind it")
        assert errors == [], errors
    finally:
        page.close()


def test_typing_a_slug_does_not_create_a_deck_per_keystroke(browser, viz_server):
    """Typing "zur-enchantress" left `zur`, `zur-en`, `zur-enchan` and
    `zur-enchantress` on the bench — three of them junk.

    The slug names a DIRECTORY. Every other field can autosave noisily because
    it edits a draft that already exists; this one decides WHICH draft that is,
    so it commits on `change` (blur or enter) rather than on `input`.
    """
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        page.goto(f"{viz_server}/viz/index.html?mode=build")
        page.wait_for_function("() => window.Build && window.Api && Api.probed",
                               timeout=BOOT_TIMEOUT_MS)
        saved = page.evaluate("""async () => {
            const calls = [];
            Object.defineProperty(Api, 'ready', {get: () => true, configurable: true});
            Api.call = function (name, payload) {
                if (name === 'build/save') calls.push(payload.slug);
                if (name === 'formats') return Promise.resolve({formats: [
                    {key: 'commander', name: 'Commander', deck_size: 100,
                     exact_size: true, singleton: true, commanders: 1,
                     buildable: true}]});
                return Promise.resolve({slug: payload && payload.slug});
            };
            Build.newDeck();
            await new Promise(r => setTimeout(r, 300));
            const box = document.getElementById('ndSlug');
            // Type it, character by character, exactly as a person does.
            for (const ch of 'zur-ench') {
                box.value += ch;
                box.dispatchEvent(new Event('input', {bubbles: true}));
                await new Promise(r => setTimeout(r, 90));
            }
            await new Promise(r => setTimeout(r, 900));   // past the debounce
            const duringTyping = calls.length;
            box.dispatchEvent(new Event('change', {bubbles: true}));   // blur
            await new Promise(r => setTimeout(r, 900));
            return {duringTyping, after: calls.slice()};
        }""")
        assert saved["duringTyping"] == 0, (
            f"typing saved {saved['duringTyping']} draft(s) — one directory per "
            f"prefix is exactly the reported bug")
        assert saved["after"] == ["zur-ench"], saved["after"]
        assert errors == [], errors
    finally:
        page.close()


def _names_on_disk(doc):
    """The library's card names out of a stored document, either schema.

    v1 kept a flat `cards`; v2 keeps `zones[].cards`. Tests that assert on the
    stored BYTES go through here so a schema move does not quietly turn them
    into assertions about nothing.
    """
    if doc is None:
        return None
    if isinstance(doc.get("cards"), list):
        return doc["cards"]
    return [c for z in (doc.get("zones") or []) for c in (z.get("cards") or [])]


def _open_drawer(page):
    page.evaluate("() => { if (!Shell.isOpen) Shell.toggle(); }")
    page.wait_for_function("() => document.querySelector('.lib-head')", timeout=10000)


def test_the_strip_the_tray_and_the_tiles_cannot_disagree(discover_page):
    """THE REPORTED BUG: "it says two cards in the library, only shows one".

    `MM.keep` repainted only in explore and build. In discover it wrote to
    Session and told nobody, so the strip — which Session notifies — moved while
    the panel's own count sat at whatever it last rendered. Measured against the
    real page before the fix, with no reload involved:

        cleared   strip "library empty"    tray "2 in library"
        kept 1    strip "1 in your library"  tray "2 in library"
        kept 2    strip "2 in your library"  tray "2 in library"

    Adding `discover` to that list of modes would have fixed the symptom and
    left the fourth panel to rediscover it, so the repaint is a subscription now
    and this asserts all three readings of one library at every step.
    """
    page = discover_page
    steps = page.evaluate("""() => {
        const strip = () => (document.querySelector('.shell-lib-btn') || {}).textContent || '';
        const tray = () => (document.querySelector('.discover-traycount') || {}).textContent || '';
        const out = [];
        const read = (label) => out.push({
            label,
            strip: (strip().match(/\\d+/) || [strip().indexOf('empty') >= 0 ? '0' : '?'])[0],
            tray: (tray().match(/\\d+/) || ['?'])[0],
            size: String(Session.library.size),
        });
        Session.library.clear();
        read('cleared');
        MM.keep(Discovery.rowByName('Sol Ring'));
        read('kept one');
        MM.keep(Discovery.rowByName('Rhystic Study'));
        read('kept two');
        return out;
    }""")
    for step in steps:
        assert step["strip"] == step["size"] == step["tray"], (
            f"at {step['label']!r} the strip says {step['strip']}, the tray "
            f"{step['tray']} and the library holds {step['size']}")

    _open_drawer(page)
    tiles = page.locator(".lib-tile").count()
    assert tiles == 2, f"{tiles} tiles for a library of 2"
    assert page.js_errors == []


def test_an_unreadable_draft_does_not_empty_your_library(browser, viz_server):
    """RESUMING A DRAFT REPLACES YOUR LIBRARY, so it must read before it clears.

    `resumeDraft` ran `Session.library.clear()` and only then looped over
    `(brief && brief.must_include) || []`. A 404 is safe by luck — `build.js`'s
    `getJSON` throws on `!res.ok`, so the `.catch` fires before any damage — but
    a brief that LOADS and has no `must_include` is not. That is a documented,
    reachable file: `load_brief`'s stated minimum is
    `{"slug": …, "commander": …, "bracket": 3}`, with no card list at all, and
    `brew --commander` scaffolds from that shape. Resuming one emptied the
    library and reported "0 card(s) kept" as though that were the draft's
    content.

    Both halves are asserted: the unreadable brief (404) and the readable one
    with nothing to restore.
    """
    import json as _json

    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        page.route("**/data/decks/index.json*", lambda route: route.fulfill(
            status=200, content_type="application/json",
            body=_json.dumps({
                "decks": [],
                "drafts": [{"slug": "a-draft-that-was-deleted", "deck_name": "gone",
                            "commander": "Zur the Enchanter", "theme": None,
                            "bracket": None, "kept": 2, "started": "2026-08-25"}],
            })))
        # The directory is gone, so its brief is a 404 — which `getJSON` turns
        # into a resolved `null` rather than a rejection.
        # A brief that LOADS and carries no card list — the documented minimum
        # shape, and the case a 404 does not cover.
        # The trailing `*` is load-bearing: `build.js` cache-busts every fetch,
        # so the URL carries `?v=` and a glob without it silently does not match
        # — the request falls through to a real 404 and the test then measures
        # the 404 path while believing it measures the other one. This repo has
        # paid for that glob once already.
        page.route("**/a-draft-that-was-deleted/brief.json*",
                   lambda route: route.fulfill(
                       status=200, content_type="application/json",
                       body=_json.dumps({"slug": "a-draft-that-was-deleted",
                                         "commander": "Zur the Enchanter",
                                         "bracket": 3})))
        page.goto(f"{viz_server}/viz/index.html")
        page.wait_for_function(
            "() => window.Discovery && Discovery.isReady() && window.Build",
            timeout=30000)

        out = page.evaluate("""async () => {
            Session.library.clear();
            ['Sol Ring', 'Rhystic Study'].forEach(n => Session.library.add(n));
            const before = Session.library.names;
            MM.setMode('build');
            // Build reads the manifest lazily; resuming before it lands returns
            // 'no draft "…"' and the destructive path never runs — which is a
            // test that passes by not testing anything.
            await new Promise(r => { const t = setInterval(() => {
                if (Build.draftSlugs && Build.draftSlugs().length) { clearInterval(t); r(); }
            }, 100); setTimeout(() => { clearInterval(t); r(); }, 15000); });
            const sawDraft = Build.draftSlugs().indexOf('a-draft-that-was-deleted') !== -1;
            Build.resumeDraft('a-draft-that-was-deleted');
            await MM.whenIdle();
            const status = (document.getElementById('status') || {}).textContent || '';
            return { before, sawDraft, status, after: Session.library.names,
                     stored: (d => !d ? null : (Array.isArray(d.cards) ? d.cards : (d.zones || []).reduce((a, z) => a.concat(z.cards || []), [])))(JSON.parse(localStorage.getItem('manamap-library') || 'null')) };
        }""")
        # Without these the test can pass by never running the code it is about.
        assert out["sawDraft"], "Build never saw the draft — the path did not run"
        assert "could not be read" in out["status"], (
            f"resumeDraft did not reach the guard: {out['status']!r}")
        assert out["after"] == out["before"], (
            f"an unreadable draft emptied the library: "
            f"{out['before']} -> {out['after']}")
        assert out["stored"] == out["before"], "and it wrote the loss to disk"
        assert errors == [], errors
    finally:
        page.close()


def test_a_v1_library_migrates_without_losing_a_card(discover_page):
    """THE ONE THAT MATTERS: this runs against a real person's saved cards.

    The "unknown schema is left on disk and ignored" rule protects a NEWER save
    from an OLDER build. This is the other direction — the data is ours, it is
    readable, and refusing it would silently empty a library somebody spent
    months filling. Every card lands in one pile and nothing is dropped.
    """
    page = discover_page
    out = page.evaluate("""() => {
        localStorage.setItem('manamap-library', JSON.stringify({
            v: 1, corpus: '34890',
            cards: ['Sol Ring', 'Rhystic Study', 'Command Tower'],
        }));
        location.reload();
    }""")
    page.wait_for_function(
        "() => window.Discovery && Discovery.isReady() && Session.library.size === 3",
        timeout=30000)
    state = page.evaluate("""() => ({
        names: Session.library.names,
        zones: Session.library.zones,
        active: Session.library.active,
        stored: JSON.parse(localStorage.getItem('manamap-library')),
        migrated: (Session.restoreReport || {}).migrated,
    })""")
    assert sorted(state["names"]) == ["Command Tower", "Rhystic Study", "Sol Ring"]
    assert len(state["zones"]) == 1 and state["zones"][0]["count"] == 3
    assert state["migrated"] == 1, "the upgrade was not reported"
    # And it is WRITTEN, so the next boot is not a migration again.
    assert state["stored"]["v"] == 2
    assert state["stored"]["zones"][0]["cards"] == state["names"]
    assert page.js_errors == []


def _search(page, term):
    page.evaluate("""(t) => {
        const i = document.getElementById('search');
        i.value = t; i.dispatchEvent(new Event('input'));
    }""", term)
    idle(page)


def test_discover_search_lands_you_on_the_card(discover_page):
    """IT WAS DOING NOTHING AT ALL, which is what this fixes.

    Typing set `searchTerm` and called the MAP's `render()` — but Discover hides
    the atlas canvas, so the white diamonds it drew landed on a surface nobody
    could see and the only observable effect was a status line. There was
    nothing to click.
    """
    page = discover_page
    _search(page, "craterhoof")
    hits = page.locator(".search-pick")
    assert hits.count() >= 1, "no results under the search box"
    assert "Craterhoof" in hits.first.inner_text()

    hits.first.click()
    page.wait_for_function("() => Force.nodeCount === 1", timeout=15000)
    state = page.evaluate("""() => ({
        open: (MM.cardRecord(Discovery.current) || {}).n,
        nodes: Force.nodeCount,
        boxHidden: document.getElementById('searchResults').hidden,
        input: document.getElementById('search').value,
    })""")
    assert state["open"] == "Craterhoof Behemoth"
    assert state["nodes"] == 1, "the searched card should be the whole walk"
    assert state["boxHidden"] and state["input"] == "", "the picker stayed open"
    assert page.js_errors == []


def test_the_neighbours_are_cardberts(page):
    """Asserted on real neighbour ids against the matrix itself, because a flag
    would pass in exactly the broken case: Discover's precomputed top-3 for a card
    must be the top-3 of a cosine scan over `embeddings_cardbert.bin`."""
    page.wait_for_function("window.Discovery && Discovery.isReady()")
    r = page.evaluate("""async () => {
        const row = MM.allData.findIndex(d => d.n === 'Sol Ring');
        const table = Discovery.neighbours(row, 'similar').slice(0, 3).map(n => n.row);
        const scan = (await MM.nearestTo(row, 3, {respectFilters: false})).map(h => h.i);
        return {table, scan};
    }""")
    assert r["table"] and r["table"] == r["scan"], r
    assert page.js_errors == []


def _open_watch(page):
    serve_watchlist_fixture(page)
    return page.evaluate("""async () => {
        document.getElementById('modeSelect').value = 'build';
        MM.setMode('build');
        await MM.whenIdle();
        await Build.select('sharknado');
        await MM.whenIdle();
        Build.setTab('candidates');   // the set picker is the Candidates tab's
        const g = document.getElementById('candGrid');
        return {
            set: Build.watchSetId,
            grid: !!g,
            tiles: g ? g.querySelectorAll('.cg-tile').length : 0,
            buttons: g ? g.querySelectorAll('.cg-act').length : 0,
            keys: g ? g.querySelector('.cg-keys').textContent : '',
            picker: !!document.getElementById('candSetSelect'),
        };
    }""")


def test_watch_moves_a_card_out_of_unreviewed_through_the_endpoint(page):
    """With the API mocked, Watch posts `watch/mark` and the tile leaves Unreviewed.
    The mock never writes, so the tracked watchlist.json is untouched."""
    import json as _json
    posted = []

    def mark(route):
        body = _json.loads(route.request.post_data or "{}")
        posted.append(body)
        route.fulfill(status=200, content_type="application/json", body=_json.dumps({"result": {
            "slug": body["slug"], "set": body["set"],
            "card": {"name": body["card"], "verdict": body.get("verdict", "unreviewed"),
                     "note": body.get("note"), "at": "2026-10-07T12:00:00-07:00"}}}))

    page.route("**/api/health", lambda route: route.fulfill(
        status=200, content_type="application/json", body='{"result": {"ok": true}}'))
    page.route("**/api/", lambda route: route.fulfill(
        status=200, content_type="application/json", body='{"commands": ["watch/mark"]}'))
    page.route("**/api/watch/mark", mark)
    page.evaluate("() => Api.refresh()")
    before = _open_watch(page)
    assert before["buttons"] > 0, "the write buttons did not appear with the API up"
    name = page.evaluate("() => document.querySelector('#candGrid .cg-tile').getAttribute('data-card')")
    # A REAL click, through hit-testing. `el.click()` in script reached the button while
    # the graph canvas (z-index 10) sat over the grid and ate every click Sean made —
    # this test passed against a grid nobody could use (2026-10-07).
    page.locator('#candGrid .cg-tile').first.locator('[data-verdict="watching"]').click(timeout=3000)
    idle(page)
    after = page.evaluate("""() => ({left: Build.__gridCards(),
        tiles: document.querySelectorAll('#candGrid .cg-tile').length})""")
    after["name"] = name
    assert page.js_errors == []
    assert posted and posted[0]["verdict"] == "watching" and posted[0]["set"] == "wheel-interaction"
    assert after["name"] not in after["left"], "a watched card stayed in Unreviewed"
    assert after["tiles"] == before["tiles"] - 1
