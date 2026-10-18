"""The deck page's DeckCheck Threat Index panel, beside the bracket floor.

`deckcheck.json` is dated external evidence: the panel draws the newest reading,
our floor against their bracket, the twelve grades in their five groups and the
omni over readings of this list — and draws NOTHING when there is no file.
"""

import json

import pytest

from conftest_viz import page  # noqa: F401  (fixture)
from test_viz_deck_combos import _new_page

pytestmark = pytest.mark.browser

BM = {"R1": "A", "R2": "B", "A1": "S", "A2": "A", "P1": "B", "P2": "C", "P3": "B",
      "I1": "A", "I2": "B", "S1": "C", "S2": "B", "S3": "F"}
READING = {"as_of": "2026-10-17", "source_updated_at": "2026-10-17T10:05:00Z",
           "decklist_sha256": "0" * 64, "deckcheck_id": "Q9",
           "url": "https://deckcheck.co/app/decklist/Q9", "score": 61, "max_score": 96,
           "bracket": 4, "benchmarks": BM}
DOC = {"schema_version": 1, "slug": "sharknado", "branch": None, "source": "deckcheck.co",
       "last_pulled": "2026-10-17", "readings": [READING],
       "omni": {"decklist_sha256": "0" * 64, "n": 1, "median_score": 61, "median_bracket": 4,
                "benchmarks": BM, "rule": "ours"},
       "floor_vs_dti": {"floor": 4, "floor_name": "Optimized", "floor_decklist_sha256": "0" * 64,
                        "their_bracket": 4, "relation": "equal", "reading_as_of": "2026-10-17"}}


def test_the_panel_draws_the_reading_the_floor_and_the_twelve_grades(browser, viz_server):
    page, errors = _new_page(browser, viz_server, "sharknado",
                             [("deckcheck.json", json.dumps(DOC), 200)])
    try:
        assert errors == []
        panel = page.locator("#panel-dti")
        assert panel.count() == 1
        text = panel.inner_text()
        assert "61/96" in text and "2026-10-17" in text and "agrees with our floor" in text
        assert "the floor gates, this never does" in text
        chips = panel.locator(".dti-chip")
        assert chips.count() == 12
        assert chips.first.get_attribute("data-grade") in set(BM.values())
        assert "ours, not DeckCheck" in text
        assert panel.locator("a[href='https://deckcheck.co/app/decklist/Q9']").count() == 1
    finally:
        page.close()


def test_no_file_means_no_panel(browser, viz_server):
    page, errors = _new_page(browser, viz_server, "sharknado",
                             [("deckcheck.json", "", 404)])
    try:
        assert errors == []
        assert page.locator("#panel-dti").count() == 0
    finally:
        page.close()
