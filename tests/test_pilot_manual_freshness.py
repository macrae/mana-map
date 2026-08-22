"""Every tracked manual must match a fresh render of its own artifacts.

LEGACY (2026-08-19): the magazine renderer. It still renders the nine frozen issues from
artifacts nothing regenerates any more (issue_plan.json, the panel keys,
card_roles/mana_base/upgrades, considering.json), and it is replaced by the compact deck
page in docs/manual-v5-spec.md. Do not extend it; internals below are accurate for what it
does.

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
from manamap.pilot import build_index, build_manual

from conftest import SRC, requires_deck

# The renderer's whole world: the pilot source tree (a verified closure — nothing
# under it imports outside `manamap.pilot` and `manamap.config`), the deck's
# artifacts, the published page, and the one global graph it reads.
# The whole package: `pilot/` alone is NOT a closure — nine modules import
# from `manamap.sim`, `manamap.analysis` and `manamap.ingest`. See
# tests/test_pilot_artifact_freshness.py for the full accounting.
CODE = (SRC,)


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


@requires_deck
def test_the_newsstand_matches_a_fresh_render(tmp_path, unchanged):
    """`manuals/index.html` and the deck manifest, both from `build-index`.

    Added after this file's own gap shipped a defect. The nine issue pages were
    covered and the newsstand that links them was not, so when a `design.py`
    edit moved the content-addressed stylesheet hash and only `build-manual` was
    re-run, `index.html` went on referencing `magazine.css?v=7dd470ac` — two
    generations stale — while all nine issues had moved to `?v=71fd65ae`. It
    still rendered, which is the whole problem: a cache-buster that busts nothing
    fails silently and only for the returning reader.

    `build_index.main` writes both files, so both are checked here rather than
    trusting that whoever regenerated one regenerated the other.
    """
    outputs = [MANUALS_DIR / "index.html", DECKS_DIR / "index.json"]
    unchanged(*CODE, DECKS_DIR, MANUALS_DIR / "magazine.css", *outputs)

    backups = []
    for path in outputs:
        backup = tmp_path / path.name
        shutil.copy2(path, backup)
        backups.append((path, backup))
    try:
        build_index.main(type("Args", (), {})())
        fresh = {path: path.read_text() for path, _ in backups}
    finally:
        for path, backup in backups:
            shutil.copy2(backup, path)

    stale = [path.name for path, backup in backups
             if fresh[path] != backup.read_text()]
    assert not stale, (
        f"{stale} out of date — run `manamap pilot build-index` (it is free) "
        f"and commit. The commonest cause is a `design.py` edit followed by "
        f"`build-manual` alone: the stylesheet is content-addressed, so its hash "
        f"moves in the nine issue pages and not in the newsstand.")
