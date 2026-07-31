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

# The map eagerly fetches a 12.9 MB projection before it renders anything, then loads
# region labels in the background. Every wait below is generous on purpose: a flaky
# browser test is worse than a slow one, because it teaches you to ignore red.
#
# Raised from 60s after one unreproducible fixture ERROR during a full 1,007-test run —
# the browser tests come last, after ~4 minutes of CPU-bound pipeline tests, and each
# opens a fresh page that re-parses the projection. The cause was not diagnosed; this is
# insurance, not a fix. If it recurs, instrument the fixture rather than raising it again.
BOOT_TIMEOUT_MS = 120_000


# Discovery is the landing now, so every existing fixture asks for the map explicitly.
# A test about rendering 34,322 points should say so rather than rely on what boot
# happens to produce — and it documents the change for whoever reads these next.
EXPLORE = "?mode=explore"


def _boot(browser, viz_server, query=""):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(f"{viz_server}/viz/index.html{query}")
    page.add_style_tag(content="*, *::before, *::after {"
                               " transition: none !important; animation: none !important; }")
    page.wait_for_function("() => window.MM && MM.allData && MM.allData.length > 0",
                           timeout=BOOT_TIMEOUT_MS)
    page.js_errors = errors
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
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(f"{viz_server}/viz/index.html?card=Craterhoof%20Behemoth")
    page.add_style_tag(content="*, *::before, *::after {"
                               " transition: none !important; animation: none !important; }")
    page.wait_for_function(
        "() => window.Discovery && Discovery.isReady() && Discovery.current >= 0",
        timeout=BOOT_TIMEOUT_MS)
    page.js_errors = errors
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
    page.wait_for_function("() => !!document.querySelector('.map-canvas')",
                           timeout=BOOT_TIMEOUT_MS)
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
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(f"{viz_server}/viz/index.html{EXPLORE}")
    # Kill CSS transitions. Playwright pages run backgrounded, and Chrome throttles
    # transitions there — the side panels' `transition: width 0.25s` never advances, so
    # `.deck-panel.open` sits at 1px forever and every width assertion is meaningless.
    # Verified: with transitions off the same panel measures 420px. This blinded the
    # suite to a real bug (the panel being unreachable), so it is not a nicety.
    page.add_style_tag(content="*, *::before, *::after {"
                               " transition: none !important; animation: none !important; }")
    page.wait_for_function("() => window.MM && MM.allData && MM.allData.length > 0",
                           timeout=BOOT_TIMEOUT_MS)
    page.js_errors = errors        # the whole point: a ReferenceError must fail a test
    try:
        yield page
    finally:
        page.close()
