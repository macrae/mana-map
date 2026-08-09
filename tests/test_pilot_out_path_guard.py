"""`--out` must be slug-scoped: the fix for the silent cross-deck clobber.

Concurrent deck agents share one scratchpad directory and were writing per-deck
views to generic names (`audit.json`, `aud.json`, `audit2.json`). The second write
silently replaced the first, so an agent working one deck read another deck's
numbers under its own invocation. Seven agents hit it across two sessions and every
catch was an agent noticing an implausible figure — luck, not a control.
"""

import pytest

from manamap.pilot.common import deck_dir, resolve_out_path


def test_a_directory_auto_names_per_slug(tmp_path):
    """The safe path is the easy one: hand it a directory, it cannot collide."""
    a = resolve_out_path(tmp_path, "heliod", "deck-audit")
    b = resolve_out_path(tmp_path, "goblin-storm", "deck-audit")
    assert a != b
    assert a.name == "deck-audit-heliod.json"
    assert b.name == "deck-audit-goblin-storm.json"


def test_a_trailing_slash_counts_as_a_directory(tmp_path):
    out = resolve_out_path(str(tmp_path / "views") + "/", "sisay", "deck-facts")
    assert out.name == "deck-facts-sisay.json"


def test_a_generic_path_elsewhere_is_refused(tmp_path):
    """The exact case that used to corrupt silently."""
    with pytest.raises(SystemExit) as e:
        resolve_out_path(tmp_path / "audit.json", "heliod", "deck-audit")
    msg = str(e.value)
    assert "does not contain the slug" in msg
    assert "heliod" in msg


def test_a_path_carrying_the_slug_is_allowed(tmp_path):
    out = resolve_out_path(tmp_path / "audit-heliod.json", "heliod", "deck-audit")
    assert out.name == "audit-heliod.json"


def test_a_bare_filename_still_means_the_decks_own_directory():
    """Unchanged behaviour: no separator means the deck's own folder, which is
    already unambiguous — two decks cannot share it."""
    out = resolve_out_path("view.json", "hapatra", "deck-facts")
    assert out == deck_dir("hapatra") / "view.json"


def test_two_decks_cannot_resolve_to_one_path(tmp_path):
    """The property the whole guard exists to hold."""
    slugs = ("edgar-vampires", "gishath", "goblin-storm", "hapatra",
             "heliod", "sisay", "ur-dragon", "yawgmoth-swarm")
    paths = {resolve_out_path(tmp_path, s, "deck-audit") for s in slugs}
    assert len(paths) == len(slugs)
