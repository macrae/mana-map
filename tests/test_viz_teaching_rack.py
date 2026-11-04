"""The workbench's Teaching rack, in a real browser, on the real manifest."""
from __future__ import annotations

import json

import pytest

from conftest_viz import page  # noqa: F401
from manamap.config import DECKS_DIR

pytestmark = pytest.mark.browser

TEACH = {f"teach-{c}" for c in ("white", "blue", "black", "red", "green")}


def _racks(page):
    """Rack title -> the slugs of the deck cards in it."""
    return page.evaluate(r"""() => {
      const out = {};
      for (const r of document.querySelectorAll('#racks .wb-rack')) {
        const h = r.querySelector('h2');
        if (!h) continue;
        const slugs = new Set();
        for (const a of r.querySelectorAll('a[href*="deck="]')) {
          const m = /deck=([a-z0-9-]+)/.exec(a.getAttribute('href'));
          if (m) slugs.add(m[1]);
        }
        out[h.textContent.replace(/\s*\d+\s*$/, '').trim()] = [...slugs];
      }
      return out;
    }""")


def _open(page, manifest=None, view=""):
    if manifest is not None:
        page.route("**/data/decks/index.json*", lambda r: r.fulfill(
            status=200, content_type="application/json", body=json.dumps(manifest)))
    page.goto(page.url.split("/viz/")[0] + "/viz/workbench.html" + view)
    page.wait_for_selector("#racks .wb-rack, #racks table", timeout=30000)


def test_the_teaching_rack_holds_exactly_the_five_and_no_other_rack_does(page):
    _open(page)
    racks = _racks(page)
    assert set(racks["Teaching"]) == TEACH, racks
    for title, slugs in racks.items():
        if title not in ("Teaching", "Branches"):
            assert not TEACH & set(slugs), (title, slugs)


def test_a_teaching_deck_is_not_told_to_climb_the_ladder(page):
    _open(page)
    rack = page.locator('.wb-rack:has(h2:has-text("Teaching"))')
    page.wait_for_function("() => document.querySelectorAll('.wb-rack .wb-chip').length > 0", timeout=15000)
    assert "gates to" not in rack.inner_text().lower()


def test_the_rack_sits_directly_under_sleeved(page):
    _open(page)
    titles = list(_racks(page))
    assert titles.index("Teaching") == titles.index("Sleeved") + 1, titles


def test_a_paper_lock_does_not_move_a_teaching_deck_to_sleeved(page):
    manifest = json.loads((DECKS_DIR / "index.json").read_text(encoding="utf-8"))
    row = next(d for d in manifest["decks"] if d["slug"] == "teach-red")
    row.update(locked=True, stage="sleeved")
    _open(page, manifest)
    racks = _racks(page)
    assert "teach-red" in racks["Teaching"] and "teach-red" not in racks["Sleeved"]


def test_the_fleet_table_leaves_them_out(page):
    _open(page, view="?view=table")
    page.wait_for_selector("#racks table", timeout=30000)
    text = page.inner_text("#racks table")
    assert "teach-red" not in text.lower(), text[:400]
