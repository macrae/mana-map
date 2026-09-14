"""No live run may be piloted by a commander its deck is not built around.

SIX OF ZUR'S EIGHT RUNS WERE. The deck is built on `Zur, Eternal Schemer`;
Forge played `Zur the Enchanter` — a different card with different abilities —
in 240 of its 300 recorded games. Nothing detected it for weeks, and nothing
could have: the played commander was in the run record, the declared one was in
`cards.json`, and no code had ever put them in the same view.

The figures were not obviously wrong. They were a real interval over real games
of a deck nobody owns, sitting beside the valid runs, indistinguishable at a
glance and equally quotable.

This is the gate that makes it detectable FLEET-WIDE and permanently. The six
bad runs live under `sim/quarantine/`, which works because `forge.list_runs`
globs `sim/*.json` NON-RECURSIVELY — a subdirectory is a real quarantine rather
than a naming convention.
"""

import json

import pytest

from manamap.config import DECKS_DIR

from conftest import requires_deck


def _decks_with_runs():
    # `p` is `<decks>/<slug>/sim`, so the slug is `p.parent.name`. The first
    # version of this walked one level too far and found nothing — caught only
    # by the `assert checked >= 10` below, which is why this repo requires one
    # on any loop over a collection that could legitimately be empty.
    return sorted(p.parent.name for p in DECKS_DIR.glob("*/sim")
                  if any(p.glob("*.json")))


def _declared_commanders(slug):
    """EVERY commander, because a PARTNER DECK HAS TWO.

    This returned the FIRST `is_commander` card and the caller compared it to
    the FIRST name in the run's seat. On a single-commander deck those are the
    same card. On sharknado — Shabraz / Brallin, the fleet's first partner deck
    to get a Forge run — `cards.json` lists Shabraz first and the run reports
    Brallin first, so a run in which BOTH partners were cast (Shabraz 52 times,
    Brallin 41, across 120 games) read as "piloted by the wrong commander" and
    the fix on offer was to quarantine it.

    Comparing sets keeps what this test is for — zur's six runs really were
    piloted by a card the deck is not built on — and stops it firing on a deck
    that is built on two.
    """
    path = DECKS_DIR / slug / "cards.json"
    if not path.exists():
        return frozenset()
    return frozenset(
        _front(card.get("name"))
        for card in json.loads(path.read_text()).get("cards", [])
        if card.get("is_commander") and card.get("name"))


def _front(name):
    return (name or "").split(" // ")[0].strip().lower()


@requires_deck
def test_no_live_run_used_the_wrong_commander():
    """The whole fleet, every run `list_runs` can still see."""
    from manamap.sim import forge

    offenders, checked = [], 0
    for slug in _decks_with_runs():
        declared = _declared_commanders(slug)
        if not declared:
            continue
        for doc in forge.list_runs(slug) or []:
            played = (doc.get("seats") or [{}])[0].get("commander") or []
            checked += 1
            # EVERY name the seat played must be one the deck is built on. A
            # partner pair supplies two and either order is correct; a foreign
            # name is the defect, whichever slot it sits in.
            stray = [n for n in played if n and _front(n) not in declared]
            if stray:
                offenders.append(
                    f"{slug}: {doc.get('run_id', '?')[:44]} played {stray!r}, "
                    f"deck is built on {sorted(declared)!r}")
    assert checked >= 10, f"only {checked} runs checked — the walk is broken"
    assert not offenders, (
        "runs piloted by the wrong commander are live and quotable:\n  "
        + "\n  ".join(offenders)
        + "\n\nMove them to sim/quarantine/ — `list_runs` globs sim/*.json "
          "non-recursively, so a subdirectory hides them from every consumer.")


@requires_deck
def test_the_quarantine_is_a_real_quarantine():
    """A directory only quarantines if the discovery path cannot see into it."""
    from manamap.sim import forge

    qdir = DECKS_DIR / "zur-enchantress" / "sim" / "quarantine"
    if not qdir.is_dir():
        pytest.skip("nothing quarantined on this checkout")
    quarantined = {p.stem for p in qdir.glob("*.json") if p.name != "MANIFEST.json"}
    assert quarantined, "an empty quarantine directory"
    live = {(doc.get("run_id") or "") for doc in forge.list_runs("zur-enchantress")}
    assert not (quarantined & live), "a quarantined run is still being listed"


@requires_deck
def test_the_quarantine_says_why_it_exists():
    """A directory of moved files with no explanation gets restored by the next
    person who finds it."""
    manifest = (DECKS_DIR / "zur-enchantress" / "sim" / "quarantine"
                / "MANIFEST.json")
    if not manifest.exists():
        pytest.skip("nothing quarantined on this checkout")
    doc = json.loads(manifest.read_text())
    assert doc["declared_commander"] == "Zur, Eternal Schemer"
    assert doc["runs"], "a manifest listing nothing"
    assert "do_not" in doc and "comparison" in doc["do_not"]
    for run in doc["runs"]:
        assert _front(run["played"]) != _front(doc["declared_commander"])
