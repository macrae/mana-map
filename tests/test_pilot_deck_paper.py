"""The paper lock: is this deck BUILT, and does the cardboard still match the repo?

WHY IT IS A NEW FIELD AND NOT A STATUS. `common.DECK_STATUSES` holds three
values and all three are obituaries — `broken-down`, `superseded`, `retired` —
and `deck_status_of` returns None for a healthy deck. So "live" has always meant
"not explicitly killed": an absence. A workbench whose front door shows the decks
you can play tonight needs an assertion, and no artifact can derive it, because
it is a fact about cardboard rather than about a list.

WHY IT HANGS OFF A VERSION. What is sleeved is one exact 99. That is also what
makes drift computable, and drift is the thing worth showing: the repo moves on
every swap and the sleeves do not.
"""

import json
import shutil

import pytest

from manamap.pilot import deck_versions as dv
from manamap.pilot.common import deck_dir

from conftest import requires_deck

SLUG = "edgar-vampires"


@pytest.fixture
def unlocked(tmp_path, monkeypatch):
    """A writable stand-in for the deck directory.

    These tests write `deck_versions.json`, and the real path is TRACKED and
    scanned by `build_index.gather_entries` — so writing it in place races every
    manifest test running in another xdist worker, which is exactly how
    `test_the_manifest_is_byte_deterministic` started failing under `-n auto`
    while passing alone.

    Only the FILES move. `versions()` shells out to git against a path literal,
    so version history still comes from the real repo; `diff_vs_working` reads
    `decklist.txt` from the deck dir, so a copy goes with it.
    """
    shutil.copy(deck_dir(SLUG) / "decklist.txt", tmp_path / "decklist.txt")
    monkeypatch.setattr(dv, "deck_dir", lambda slug: tmp_path)
    return tmp_path / dv.TAGS_FILE


@requires_deck
def test_an_unlocked_deck_reports_none_not_a_falsey_dict(unlocked):
    """`None` and `{"locked": False}` read the same in Python and differently in
    a template. The front door filters on presence."""
    assert dv.paper(SLUG) is None
    assert dv.paper_state(SLUG) is None


@requires_deck
def test_locking_a_version_records_it_and_survives_a_reread(unlocked):
    dv.set_paper(SLUG, ref="V5", note="sleeved for the Orinda weekly")
    p = dv.paper(SLUG)
    assert p["version"] == 5
    assert p["note"] == "sleeved for the Orinda weekly"
    assert p["built_at"]                      # defaults to today
    doc = json.loads(unlocked.read_text())
    assert list(doc) == ["slug", "paper", "tags"], "key order is authored, not incidental"


@requires_deck
def test_drift_names_the_cards_and_takes_the_hands_side(unlocked):
    """V5 -> V6 is edgar's THE LOCK, twelve swaps. The two sides are the physical
    instruction: `pull` leaves the sleeves, `add` goes in."""
    dv.set_paper(SLUG, ref="V5")
    s = dv.paper_state(SLUG)
    assert s["locked"] is True
    assert s["in_sync"] is False
    assert s["versions_behind"] == 1
    assert len(s["drift"]["pull"]) == 12
    assert len(s["drift"]["add"]) == 12
    assert "Cathars' Crusade" in s["drift"]["pull"]
    assert "Blood Artist" in s["drift"]["add"]


@requires_deck
def test_a_lock_on_the_current_version_is_in_sync_and_has_no_drift(unlocked):
    dv.set_paper(SLUG, ref="V6")
    s = dv.paper_state(SLUG)
    assert s["in_sync"] is True
    assert s["versions_behind"] == 0
    assert s["drift"] is None


@requires_deck
def test_report_carries_the_lock_and_says_what_to_do_about_it(unlocked):
    dv.set_paper(SLUG, ref="V5")
    doc = dv.report(SLUG)
    assert doc["paper"]["version"] == 5
    assert any("pull 12, add 12" in n for n in doc["notes"]), doc["notes"]


@requires_deck
def test_an_unresolvable_lock_is_reported_not_crashed(unlocked):
    """A lock naming a version git no longer carries — a rewritten history, or a
    hand-edited file. Silently unlocking would be worse than saying so."""
    unlocked.write_text(json.dumps(
        {"slug": SLUG, "paper": {"version": 999}, "tags": {}}))
    s = dv.paper_state(SLUG)
    assert s["unresolved"] is True
    assert s["in_sync"] is None


@requires_deck
def test_clearing_withdraws_the_lock_and_leaves_tags_alone(unlocked):
    dv.tag(SLUG, "the-lock", ref="V6", note="frozen for piloting practice")
    dv.set_paper(SLUG, ref="V6")
    dv.set_paper(SLUG, clear=True)
    assert dv.paper(SLUG) is None
    assert "the-lock" in dv.tags(SLUG), "clearing the lock must not touch tags"


@requires_deck
def test_locking_a_version_that_does_not_exist_refuses(unlocked):
    with pytest.raises(SystemExit):
        dv.set_paper(SLUG, ref="V99")


@requires_deck
def test_clearing_an_unlocked_deck_refuses_rather_than_no_opping(unlocked):
    with pytest.raises(SystemExit):
        dv.set_paper(SLUG, clear=True)
