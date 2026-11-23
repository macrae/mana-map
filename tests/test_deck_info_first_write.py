"""A first `deck-info --write` must already be the fixed point (2026-11-12).

`info.json` is one of its own stages (`dossier`), so the first write on a new deck
recorded itself as missing and the freshness test read the second write as a different
file: valgavoth's first gate run failed on exactly that. Proved by re-introducing the
bug: drop the second compose in `deck_info.main` and this test fails on `status`.
"""
import json

import pytest

from conftest import requires_deck
from manamap.config import DECKS_DIR
from manamap.pilot import deck_info

SLUG = "valgavoth"


@pytest.mark.regression
@requires_deck
def test_the_first_write_equals_the_second(tmp_path, monkeypatch):
    """ON A COPY (2026-11-22). It deleted the tracked `valgavoth/info.json` in place, and
    under parallel workers `test_info_json_matches_a_fresh_run[valgavoth]` read the hole
    — a prepush failed on a file that was fine. No test mutates a tracked deck file that
    another worker can read."""
    import shutil
    src = DECKS_DIR / SLUG
    if not (src / "info.json").is_file():
        pytest.skip(f"{SLUG} has no tracked info.json")
    decks = tmp_path / "decks"
    shutil.copytree(src, decks / SLUG)
    monkeypatch.setattr("manamap.config.DECKS_DIR", decks)
    path = decks / SLUG / "info.json"
    args = type("Args", (), {"slug": SLUG, "branch": None, "write": True})()
    path.unlink()
    deck_info.main(args)
    first = json.loads(path.read_text())
    deck_info.main(args)
    second = json.loads(path.read_text())
    assert first == second
    assert (src / "info.json").is_file(), "the tracked file was never touched"
