"""Page fixtures for the behavioural viz tests.

The session-scoped `browser` and `viz_server` fixtures live in `conftest.py`, NOT here.
They used to live here and be imported by each test module — which registers a SEPARATE
session fixture per importing module, so two files importing `browser` opened two
concurrent `sync_playwright()` contexts and every browser test errored at setup. It only
showed up in a full run: each file passed alone.

Playwright is still not imported unless a browser test actually runs; `importorskip` sits
inside the fixture body, so the 1,000+ non-browser tests pay nothing.

The suite serves the repo root on an ephemeral port, because every fetch in `viz/` is
`../data/<file>` relative to `viz/index.html` — `viz/` and `data/` must be siblings under
the server root, which is the same constraint GitHub Pages imposes.
"""

from __future__ import annotations

import pytest

# Card art comes from Scryfall over the public internet. A failed image fetch surfaces as
# a console error, so a rate-limit or a dropped connection failed whichever browser test
# happened to be running — `ERR_CONNECTION_RESET` took out `test_escape_returns_the_whole_atlas`,
# which does not load an image on purpose and has nothing to do with Scryfall.
#
# Filtered NARROWLY, and only for resource loads the app does not control. A ReferenceError,
# a TypeError, a failed fetch of our OWN data — all still fail the test, which is the whole
# point of capturing these. See docs/testing.md on waiting for conditions rather than timers:
# this is the same lesson one layer down, about not asserting on the network.
_EXTERNAL_NOISE = ("Failed to load resource",)


def _record(errors):
    """Console/page error sink that ignores third-party resource failures."""
    def add(text):
        text = str(text)
        if any(marker in text for marker in _EXTERNAL_NOISE):
            return
        errors.append(text)
    return add

# The map eagerly fetches a 12.9 MB projection before it renders anything, then loads
# region labels in the background. Every wait below is generous on purpose: a flaky
# browser test is worse than a slow one, because it teaches you to ignore red.
#
# Raised from 60s after one unreproducible fixture ERROR during a full 1,007-test run.
# It recurred once at 1,092 tests and then did NOT reproduce across two consecutive full
# runs — so it is rare and still undiagnosed. Per this comment's own earlier instruction
# the timeout has deliberately NOT been raised again; `_wait_for_boot` below now reports
# page state on failure instead, so the next occurrence says something instead of just
# saying "ERROR at setup".
BOOT_TIMEOUT_MS = 120_000


# Discovery is the landing now, so every existing fixture asks for the map explicitly.
# A test about rendering 34,322 points should say so rather than rely on what boot
# happens to produce — and it documents the change for whoever reads these next.
EXPLORE = "?mode=explore"


def _wait_for_boot(page, condition, what):
    """Wait, and if it times out say what the page was actually doing.

    A bare `wait_for_function` timeout surfaces as "ERROR at setup" with no page state at
    all, which is why the last occurrence of this went undiagnosed and got a bigger
    timeout instead of an explanation. Everything gathered here is cheap and is exactly
    what the first debugging question would ask for.
    """
    try:
        page.wait_for_function(condition, timeout=BOOT_TIMEOUT_MS)
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        try:
            state = page.evaluate("""() => ({
                url: location.href,
                readyState: document.readyState,
                hasMM: typeof window.MM !== 'undefined',
                rows: (window.MM && MM.allData) ? MM.allData.length : -1,
                hasDiscovery: typeof window.Discovery !== 'undefined',
                discoveryReady: !!(window.Discovery && Discovery.isReady()),
                scripts: [...document.scripts].map(s => s.src.split('/').pop()),
            })""")
        except Exception as inner:  # noqa: BLE001
            state = f"page unreachable: {inner}"
        raise AssertionError(
            f"boot never satisfied {what!r} within {BOOT_TIMEOUT_MS} ms.\n"
            f"page state: {state}\n"
            f"console errors: {getattr(page, 'js_errors', None)}"
        ) from exc


def _boot(browser, viz_server, query=""):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list[str] = []
    _add = _record(errors)
    page.on("pageerror", lambda e: _add(e))
    page.on("console", lambda m: _add(m.text) if m.type == "error" else None)
    page.goto(f"{viz_server}/viz/index.html{query}")
    page.add_style_tag(content="*, *::before, *::after {"
                               " transition: none !important; animation: none !important; }")
    page.js_errors = errors
    _wait_for_boot(page, "() => window.MM && MM.allData && MM.allData.length > 0",
                   "the projection to load")
    return page




@pytest.fixture
def discover_page(browser, viz_server):
    """The landing, on a card chosen by deep link so the test is not random.

    Waits on `Discovery.isReady()` rather than `MM.allData` — the whole point of the
    front door is that it paints from 0.56 MB of viz_index without the 12.9 MB
    projection, and a fixture that waited for allData could not observe that.
    """
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list[str] = []
    _add = _record(errors)
    page.on("pageerror", lambda e: _add(e))
    page.on("console", lambda m: _add(m.text) if m.type == "error" else None)
    page.goto(f"{viz_server}/viz/index.html?card=Craterhoof%20Behemoth")
    page.add_style_tag(content="*, *::before, *::after {"
                               " transition: none !important; animation: none !important; }")
    page.js_errors = errors
    _wait_for_boot(page, "() => window.Discovery && Discovery.isReady() && Discovery.current >= 0",
                   "discovery to land on a card")
    try:
        yield page
    finally:
        page.close()


@pytest.fixture
def canvas_page(browser, viz_server):
    """The map under the canvas renderer (Phase 2 of the Plotly migration).

    Same page, same code, `?renderer=canvas` — which is the point of the strangler: both
    renderers are live at once so they can be compared on identical data.
    """
    page = _boot(browser, viz_server, "?renderer=canvas&mode=explore")
    _wait_for_boot(page, "() => !!document.querySelector('.map-canvas')",
                   "the canvas renderer to attach")
    # PIN THE MAP. Every test on this fixture was written against the colour+type map —
    # its contour density, its region label layout, and the fact that it is the one map
    # that draws similarity arcs (`MAP_ARC_RELATIONS`; the ability map deliberately draws
    # none). Inheriting the map from the boot default made all of that implicit, so
    # flipping the app's default to Abilities silently repointed ten tests at a map whose
    # answers are different by design — they failed as if the renderer had broken.
    # A test that cares which map it is on has to say so.
    page.select_option("#mapSelect", "default")
    _wait_for_boot(page, "() => window.MM && MM.currentMap === 'default' && MM.allData.length",
                   "the colour+type map to load")
    page.wait_for_timeout(600)
    try:
        yield page
    finally:
        page.close()


@pytest.fixture
def page(browser, viz_server):
    """A booted map page.

    Waits for `MM.allData` to be populated rather than for a timer — the projection is
    12.9 MB and its parse time depends on the machine.
    """
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list[str] = []
    _add = _record(errors)
    page.on("pageerror", lambda e: _add(e))
    page.on("console", lambda m: _add(m.text) if m.type == "error" else None)
    page.goto(f"{viz_server}/viz/index.html{EXPLORE}")
    # Kill CSS transitions. Playwright pages run backgrounded, and Chrome throttles
    # transitions there — the side panels' `transition: width 0.25s` never advances, so
    # `.deck-panel.open` sits at 1px forever and every width assertion is meaningless.
    # Verified: with transitions off the same panel measures 420px. This blinded the
    # suite to a real bug (the panel being unreachable), so it is not a nicety.
    page.add_style_tag(content="*, *::before, *::after {"
                               " transition: none !important; animation: none !important; }")
    page.js_errors = errors        # the whole point: a ReferenceError must fail a test
    _wait_for_boot(page, "() => window.MM && MM.allData && MM.allData.length > 0",
                   "the projection to load")
    try:
        yield page
    finally:
        page.close()


@pytest.fixture
def still_page(browser, viz_server):
    """The map for a viewer who has asked their OS not to animate things.

    A separate fixture rather than a flag on `page`, because `prefers-reduced-motion` is
    read once at renderer construction — it decides a default, not a per-frame condition,
    so it cannot be toggled after boot and a test that tried would be testing nothing.
    """
    page = browser.new_page(viewport={"width": 1440, "height": 900},
                            reduced_motion="reduce")
    errors: list[str] = []
    _add = _record(errors)
    page.on("pageerror", lambda e: _add(e))
    page.on("console", lambda m: _add(m.text) if m.type == "error" else None)
    page.goto(f"{viz_server}/viz/index.html{EXPLORE}")
    page.js_errors = errors
    _wait_for_boot(page, "() => window.MM && MM.allData && MM.allData.length > 0",
                   "the projection to load")
    try:
        yield page
    finally:
        page.close()


@pytest.fixture(scope="session")
def corpus_count():
    """The corpus size the viz should display, DERIVED rather than hardcoded.

    `projection_2d.json` is literally what boots into `MM.allData`, and by the
    index-alignment invariant its length IS the corpus size — so asserting
    against it keeps the browser suite honest across Scryfall refreshes instead
    of failing on a stale literal every time the card count moves.
    """
    import json

    from manamap.config import PROJECTION_PATH

    with open(PROJECTION_PATH) as f:
        return len(json.load(f))
