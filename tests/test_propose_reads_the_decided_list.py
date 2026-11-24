"""`propose --study` reads the list the call was DECIDED on (2026-11-23). A study whose list
moved while it ran is installed with `decided_on_decklist_sha256`; propose compared only the
list it OPENED on and refused heliod's land pass, decided on the exact list on disk."""
import pytest

from manamap.pilot import deck_branch
from manamap.pilot import study as st


def _doc(opened, decided=None):
    d = {"branch": "b", "call": {"decision": "approve"}, "decided_at": "2026-11-23",
         "decklist_sha256": opened}
    if decided:
        d["decided_on_decklist_sha256"] = decided
    return d


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setattr(deck_branch, "_sha_of_list", lambda slug, branch: "LIVE")
    return lambda doc: monkeypatch.setattr(st, "read", lambda slug, sid: doc)


def test_a_study_decided_on_the_live_list_proposes_though_it_opened_on_another(live):
    live(_doc("OLD", decided="LIVE"))
    assert deck_branch._study_for("d", "b", "004")["decided_on_decklist_sha256"] == "LIVE"


def test_a_study_whose_decided_list_moved_on_is_still_refused(live):
    live(_doc("OLD", decided="OLDER"))
    with pytest.raises(SystemExit):
        deck_branch._study_for("d", "b", "004")


def test_a_study_with_no_decided_list_reads_the_one_it_opened_on(live):
    live(_doc("LIVE"))
    assert deck_branch._study_for("d", "b", "004")
    live(_doc("OLD"))
    with pytest.raises(SystemExit):
        deck_branch._study_for("d", "b", "004")
