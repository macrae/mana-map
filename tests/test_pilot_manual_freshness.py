"""Every tracked manual must match a fresh render of its own artifacts.

`build-manual` is free, deterministic and deliberately uncached, which is
exactly why nothing was watching it. Four issues — gishath, goblin-storm, sisay
and ur-dragon — spent between two and five days serving content their artifacts
no longer supported: they predated the 2026-07-31 synergy re-ranking (so their
synergy chips were in the old order) and the 2026-08-02 goldfish change (so
their assumptions block never mentioned that tutors are modelled as wildcards).
Nobody noticed, because a stale manual renders perfectly.

The check is the one the plan for the deck-builder work already wanted:
"build-manual byte-identical for all 8 decks". It costs well under a second per
deck and it is the only thing standing between an artifact change and a
published page that disagrees with it.
"""

import shutil

import pytest

from manamap.config import DECKS_DIR, MANUALS_DIR, SYNERGY_GRAPH_PATH
from manamap.pilot import build_manual

from conftest import SRC, requires_deck

# The renderer's whole world: the pilot source tree (a verified closure — nothing
# under it imports outside `manamap.pilot` and `manamap.config`), the deck's
# artifacts, the published page, and the one global graph it reads.
CODE = (SRC / "pilot", SRC / "config.py")


def _deck_slugs():
    if not DECKS_DIR.is_dir():
        return []
    return sorted(d.name for d in DECKS_DIR.iterdir()
                  if d.is_dir() and (MANUALS_DIR / f"{d.name}.html").exists())


@requires_deck
@pytest.mark.parametrize("slug", _deck_slugs())
def test_tracked_manual_matches_a_fresh_render(slug, tmp_path, unchanged):
    """A committed issue must be what its artifacts render to, today."""
    published = MANUALS_DIR / f"{slug}.html"
    unchanged(*CODE, DECKS_DIR / slug, published, SYNERGY_GRAPH_PATH)
    backup = tmp_path / f"{slug}.html"
    shutil.copy2(published, backup)
    try:
        build_manual.main(type("Args", (), {"slug": slug})())
        fresh = published.read_text()
    finally:
        shutil.copy2(backup, published)
    assert fresh == backup.read_text(), (
        f"manuals/{slug}.html is stale — its artifacts render something else. "
        f"Run `manamap pilot build-manual {slug}` (it is free) and commit the result."
    )
