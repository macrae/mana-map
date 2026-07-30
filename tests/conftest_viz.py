"""Browser fixtures for the behavioural viz tests.

Kept out of `conftest.py` so the 900+ non-browser tests never import playwright and never
pay for it. `test_viz_behaviour.py` imports these explicitly.

The suite serves the repo root on an ephemeral port, because every fetch in `viz/` is
`../data/<file>` relative to `viz/index.html` — `viz/` and `data/` must be siblings under
the server root, which is the same constraint GitHub Pages imposes.
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import socket
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The map eagerly fetches a 12.9 MB projection before it renders anything, then loads
# region labels in the background. Every wait below is generous on purpose: a flaky
# browser test is worse than a slow one, because it teaches you to ignore red.
BOOT_TIMEOUT_MS = 60_000


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D102 - silence per-request logging
        pass


@pytest.fixture(scope="session")
def viz_server() -> str:
    """Serve the repo root; yield the base URL."""
    port = _free_port()
    handler = functools.partial(_QuietHandler, directory=str(ROOT))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture(scope="session")
def browser():
    playwright = pytest.importorskip(
        "playwright.sync_api",
        reason="browser tests need playwright: pip install playwright && playwright install chromium",
    )
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            yield browser
        finally:
            browser.close()


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
    page.goto(f"{viz_server}/viz/index.html")
    page.wait_for_function("() => window.MM && MM.allData && MM.allData.length > 0",
                           timeout=BOOT_TIMEOUT_MS)
    page.js_errors = errors        # the whole point: a ReferenceError must fail a test
    try:
        yield page
    finally:
        page.close()
