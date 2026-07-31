"""Shared pytest fixtures and markers for the mana-map test suite."""

import contextlib
import functools
import http.server
import socket
import threading
from pathlib import Path

import pytest

from manamap import config

ROOT = Path(__file__).resolve().parents[1]

# Marker for tests that need generated data/ artifacts (run the pipeline first).
# Gate on embeddings.npy — the last artifact of the train/embed stage — so a
# partially-populated data/ dir still skips cleanly.
requires_data = pytest.mark.skipif(
    not config.EMBEDDINGS_PATH.exists(),
    reason="requires generated data/ artifacts (run `manamap run` first)",
)

# Pilot subsystem artifact gates (same pattern).
requires_rules = pytest.mark.skipif(
    not config.RULES_INDEX_PATH.exists(),
    reason="requires the rules DB (run `manamap pilot download-rules && manamap pilot build-rules-db`)",
)

requires_deck = pytest.mark.skipif(
    not (config.DECKS_DIR / "goblin-storm" / "cards.json").exists(),
    reason="requires a fetched deck (run `manamap pilot fetch-deck goblin-storm`)",
)

requires_strategy = pytest.mark.skipif(
    not config.STRATEGY_INDEX_PATH.exists(),
    reason="requires the strategy DB (run `manamap pilot build-strategy-db`)",
)

requires_roles = pytest.mark.skipif(
    not config.CARD_ROLES_PATH.exists(),
    reason="requires the role index (run `manamap card-roles`)",
)


@pytest.fixture(scope="session")
def data_dir():
    """The resolved data directory (honors MANAMAP_DATA_DIR)."""
    return config.DATA_DIR


# ── Browser fixtures ────────────────────────────────────────────────────
#
# Session-scoped and defined HERE rather than in conftest_viz.py, because a fixture
# imported into two test modules is registered twice — which meant two concurrent
# `sync_playwright()` contexts and a full-run-only failure that neither file showed alone.
# Playwright itself is imported lazily inside `browser`, so non-browser runs pay nothing.

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


# Discovery is the landing now, so every existing fixture asks for the map explicitly.
# A test about rendering 34,322 points should say so rather than rely on what boot
# happens to produce — and it documents the change for whoever reads these next.
EXPLORE = "?mode=explore"


