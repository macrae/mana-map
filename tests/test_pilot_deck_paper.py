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

WHY A SYNTHETIC REPO. The first version of this file used `edgar-vampires` and
asserted on its real history — V5, V6, "twelve swaps", `versions_behind == 1`.
Every one of those is a fact about a deck the pilot edits, so the tests were
one card swap away from failing for a reason that had nothing to do with the
lock. They broke the moment that deck was re-baselined. A three-commit repo
built here owns its own history and cannot be surprised.
"""

import hashlib
import json
import subprocess

import pytest

from manamap.pilot import deck_history as dh
from manamap.pilot import deck_versions as dv

SLUG = "pdeck"
V1 = "1 Edgar Markov *CMDR*\n1 Sol Ring\n1 Blood Artist\n30 Swamp\n"
V2 = V1.replace("1 Blood Artist\n", "1 Cruel Celebrant\n")
V3 = V2.replace("1 Sol Ring\n", "1 Arcane Signet\n")


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                        "HOME": str(root), "PATH": "/usr/bin:/bin:/usr/local/bin"})


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def unlocked(tmp_path, monkeypatch):
    """Three committed versions of our own, and a writable deck directory."""
    root = tmp_path
    deck = root / "data" / "decks" / SLUG
    deck.mkdir(parents=True)
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", root / "data" / "decks")
    monkeypatch.setattr(dh, "_REPO_ROOT", root)
    monkeypatch.setattr(dv, "deck_dir", lambda slug: deck)
    _git(root, "init", "-q")
    for text, msg in ((V1, "the baseline"), (V2, "Celebrant for the Artist"),
                      (V3, "Signet for the Ring")):
        (deck / "decklist.txt").write_text(text)
        _git(root, "add", "."); _git(root, "commit", "-q", "-m", msg)
    return deck / dv.TAGS_FILE


def test_an_unlocked_deck_reports_none_not_a_falsey_dict(unlocked):
    """`None` and `{"locked": False}` read the same in Python and differently in
    a template. The front door filters on presence."""
    assert dv.paper(SLUG) is None
    assert dv.paper_state(SLUG) is None


def test_locking_a_version_records_it_and_survives_a_reread(unlocked):
    dv.set_paper(SLUG, ref="V2", note="sleeved for the Orinda weekly")
    p = dv.paper(SLUG)
    assert p["version"] == 2
    assert p["note"] == "sleeved for the Orinda weekly"
    assert p["built_at"]                      # defaults to today
    doc = json.loads(unlocked.read_text())
    assert list(doc) == ["slug", "paper", "tags"], "key order is authored, not incidental"


def test_drift_names_the_cards_and_takes_the_hands_side(unlocked):
    """V2 -> V3 swaps one card. The two sides are the physical instruction:
    `pull` leaves the sleeves, `add` goes in."""
    dv.set_paper(SLUG, ref="V2")
    s = dv.paper_state(SLUG)
    assert s["locked"] is True
    assert s["in_sync"] is False
    assert s["versions_behind"] == 1
    assert s["drift"]["pull"] == ["Sol Ring"]
    assert s["drift"]["add"] == ["Arcane Signet"]


def test_a_lock_on_the_current_version_is_in_sync_and_has_no_drift(unlocked):
    dv.set_paper(SLUG, ref="V3")
    s = dv.paper_state(SLUG)
    assert s["in_sync"] is True
    assert s["versions_behind"] == 0
    assert s["drift"] is None


def test_report_carries_the_lock_and_says_what_to_do_about_it(unlocked):
    dv.set_paper(SLUG, ref="V2")
    doc = dv.report(SLUG)
    assert doc["paper"]["version"] == 2
    assert any("pull 1, add 1" in n for n in doc["notes"]), doc["notes"]


def test_an_unresolvable_lock_is_reported_not_crashed(unlocked):
    """A lock naming a version git no longer carries — a rewritten history, or a
    hand-edited file. Silently unlocking would be worse than saying so."""
    unlocked.write_text(json.dumps(
        {"slug": SLUG, "paper": {"version": 999}, "tags": {}}))
    s = dv.paper_state(SLUG)
    assert s["unresolved"] is True
    assert s["in_sync"] is None


def test_clearing_withdraws_the_lock_and_leaves_tags_alone(unlocked):
    dv.tag(SLUG, "the-lock", ref="V3", note="frozen for piloting practice")
    dv.set_paper(SLUG, ref="V3")
    dv.set_paper(SLUG, clear=True)
    assert dv.paper(SLUG) is None
    assert "the-lock" in dv.tags(SLUG), "clearing the lock must not touch tags"


def test_locking_a_version_that_does_not_exist_refuses(unlocked):
    with pytest.raises(SystemExit):
        dv.set_paper(SLUG, ref="V99")


def test_clearing_an_unlocked_deck_refuses_rather_than_no_opping(unlocked):
    with pytest.raises(SystemExit):
        dv.set_paper(SLUG, clear=True)
