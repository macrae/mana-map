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
def test_the_first_write_equals_the_second():
    path = DECKS_DIR / SLUG / "info.json"
    if not path.is_file():
        pytest.skip(f"{SLUG} has no tracked info.json")
    kept = path.read_bytes()
    args = type("Args", (), {"slug": SLUG, "branch": None, "write": True})()
    try:
        path.unlink()
        deck_info.main(args)
        first = json.loads(path.read_text())
        deck_info.main(args)
        second = json.loads(path.read_text())
    finally:
        path.write_bytes(kept)
    assert first == second
