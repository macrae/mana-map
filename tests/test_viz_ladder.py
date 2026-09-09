"""The rack and the CLI must agree about the ladder.

`workbench.js` used to DEFINE "on the bench" as `living.filter(e => !e.locked)`
— a third environment invented in JavaScript, which no command could gate on and
no test could see. `info.stage` replaced it with `promote.stage`, and this file
is the thing that keeps the two from drifting apart again.

THE TEST COMPARES; IT DOES NOT RE-DERIVE. Asserting that `info.json` says what
`promote` says is only evidence if the two are computed independently — a test
that called `promote.stage` and compared it to `promote.stage` would pass
forever and mean nothing. So one side is read off the COMMITTED ARTIFACT the
browser actually fetches, and the other is computed live.
"""

import json

import pytest

from manamap.config import DECKS_DIR
from manamap.pilot import promote

from conftest import requires_deck


def _decks_with_info():
    return sorted(p.parent.name for p in DECKS_DIR.glob("*/info.json"))


@requires_deck
def test_every_dossier_agrees_with_the_ladder_about_its_rung():
    """The artifact the page fetches, against the predicate the CLI prints."""
    checked = 0
    for slug in _decks_with_info():
        info = json.loads((DECKS_DIR / slug / "info.json").read_text())
        assert info.get("stage") == promote.stage(slug), (
            f"{slug}: info.json says {info.get('stage')!r}, "
            f"promote says {promote.stage(slug)!r}")
        checked += 1
    assert checked >= 5, f"only {checked} dossiers checked"


@requires_deck
def test_the_gate_count_matches_what_promote_would_print():
    """A deck two requirements from the table must read the same on both
    surfaces, or the rack is quietly optimistic about work still to do."""
    checked = 0
    for slug in _decks_with_info():
        info = json.loads((DECKS_DIR / slug / "info.json").read_text())
        gates = info.get("gates")
        stage = promote.stage(slug)
        order = list(promote.LADDER)
        if stage is None or stage == order[-1]:
            assert gates is None, f"{slug}: at the top rung but carries a gate count"
            continue
        assert gates, f"{slug}: no gate count in info.json"
        rows = promote.gate(slug, order[order.index(stage) + 1])
        assert gates["of"] == len(rows)
        assert gates["met"] == len(rows) - len(promote.blockers(rows))
        assert gates["blocking"] == [r["label"] for r in promote.blockers(rows)]
        checked += 1
    assert checked >= 3


def test_the_rack_reads_the_stage_rather_than_inventing_one():
    """The fallback may stay — a deck mid-refresh must not empty a rack — but
    the stored stage has to win when it is there."""
    from manamap import config

    src = (config.VIZ_DIR / "js" / "workbench.js").read_text()
    assert "info.stage" in src, "the rack stopped reading the ladder"
    assert "info.gates" in src, "the rack stopped reading the gate count"
    # The old expression survives ONLY as a fallback, never as the definition.
    assert "(info && info.stage) ||" in src, (
        "the stored stage must be preferred over the derived one")
