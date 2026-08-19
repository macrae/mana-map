"""Deck versions: derived from git, tagged by the pilot, joined to the log.

Pinned here: a version is a CONTENT change (a comment-only edit adds a byte-sha to
the version it belongs to, never a new version); a log entry finds its list by the
byte-sha the captain's log stamped; tags are authored and resolve; restore is a dry
run unless asked to write; and the current working list is reported as uncommitted
when it matches no commit — which is the common case right after a swap.
"""

import hashlib
import json
import subprocess

import pytest

from manamap.pilot import deck_history as dh
from manamap.pilot import deck_versions as dv
from manamap.pilot.deck_notes import append_entry

SLUG = "vdeck"
V1 = "1 Radagast of Rhosgobel *CMDR*\n1 Craterhoof Behemoth\n1 Llanowar Elves\n30 Forest\n"
V1_COMMENT = "# the same list, commented\n" + V1
V2 = V1.replace("1 Llanowar Elves\n", "1 Heroic Intervention\n")


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(root), "PATH": "/usr/bin:/bin:/usr/local/bin"})


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A real git repo with data/decks/<slug>/decklist.txt committed three times:
    V1, a comment-only edit of V1, then V2."""
    root = tmp_path
    deck = root / "data" / "decks" / SLUG
    deck.mkdir(parents=True)
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", root / "data" / "decks")
    monkeypatch.setattr(dh, "_REPO_ROOT", root)
    _git(root, "init", "-q")
    (deck / "decklist.txt").write_text(V1)
    _git(root, "add", "."); _git(root, "commit", "-q", "-m", "the baseline")
    (deck / "decklist.txt").write_text(V1_COMMENT)
    _git(root, "add", "."); _git(root, "commit", "-q", "-m", "a comment, not a swap")
    (deck / "decklist.txt").write_text(V2)
    _git(root, "add", "."); _git(root, "commit", "-q", "-m", "Intervention for the Elves")
    (deck / "cards.json").write_text(json.dumps({"deck": SLUG, "cards": []}))
    return deck


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_a_version_is_a_content_change_not_a_commit(repo):
    vers = dv.versions(SLUG)
    assert [v["version"] for v in vers] == [1, 2]
    assert vers[0]["decklist_sha256s"] == [_sha(V1), _sha(V1_COMMENT)], (
        "the comment-only edit adds a byte-sha to V1, not a V2")
    assert vers[1]["in"] == ["Heroic Intervention"] and vers[1]["out"] == ["Llanowar Elves"]
    assert vers[1]["subject"] == "Intervention for the Elves"


def test_log_entries_join_their_version_by_the_stamped_sha(repo):
    # a game played on the commented V1 (stamp = that file's bytes), one on V2
    e1 = append_entry(SLUG, "played on the old list", result="win", at="2026-08-01T10:00:00")
    (repo / "decklist.txt").write_text(V1_COMMENT)
    e2 = append_entry(SLUG, "old list again", result="loss", at="2026-08-02T10:00:00")
    (repo / "decklist.txt").write_text(V2)
    doc = dv.report(SLUG)
    by_v = {v["version"]: v for v in doc["versions"]}
    assert by_v[2]["log_ids"] == [e1["id"]] and by_v[2]["record"]["win"] == 1
    assert by_v[1]["log_ids"] == [e2["id"]] and by_v[1]["record"]["loss"] == 1
    assert doc["current_version"] == 2 and not doc["unmatched_log_entries"]


def test_an_uncommitted_working_list_is_reported_not_guessed(repo):
    (repo / "decklist.txt").write_text(V2 + "1 Sol Ring\n")
    e = append_entry(SLUG, "played the uncommitted list")
    doc = dv.report(SLUG)
    assert doc["current_version"] is None
    assert doc["unmatched_log_entries"] == [e["id"]]
    assert any("uncommitted" in n for n in doc["notes"])


def test_tags_are_authored_and_resolve(repo):
    v = dv.tag(SLUG, "the-lock", ref="V1", note="took it to Orinda")
    assert v["version"] == 1
    doc = json.loads((repo / "deck_versions.json").read_text())
    assert doc["tags"]["the-lock"]["version"] == 1 and doc["tags"]["the-lock"]["note"] == "took it to Orinda"
    assert dv.resolve(SLUG, "the-lock")["version"] == 1
    assert dv.resolve(SLUG, "v2")["version"] == 2 and dv.resolve(SLUG, "2")["version"] == 2
    assert dv.resolve(SLUG, dv.versions(SLUG)[0]["first_sha"][:7])["version"] == 1
    assert dv.resolve(SLUG, "nope") is None
    assert dv.report(SLUG)["versions"][0]["tags"] == ["the-lock"]
    # default target is the committed working list; refuse an uncommitted one
    assert dv.tag(SLUG, "now")["version"] == 2
    (repo / "decklist.txt").write_text(V2 + "1 Sol Ring\n")
    with pytest.raises(SystemExit):
        dv.tag(SLUG, "dirty")
    with pytest.raises(SystemExit):
        dv.tag(SLUG, "two words", ref="V1")


def test_restore_is_a_dry_run_unless_asked_to_write(repo):
    v1 = dv.resolve(SLUG, "V1")
    d = dv.restore(SLUG, v1, write=False)
    assert d["in_then_not_now"] == ["Llanowar Elves"] and d["in_now_not_then"] == ["Heroic Intervention"]
    assert (repo / "decklist.txt").read_text() == V2, "a dry run writes nothing"
    dv.restore(SLUG, v1, write=True)
    assert (repo / "decklist.txt").read_text() == V1
    assert dv.report(SLUG)["current_version"] == 1, "the restored list IS V1 again"


def test_no_git_history_is_an_empty_list_not_an_error(tmp_path, monkeypatch):
    deck = tmp_path / "data" / "decks" / SLUG
    deck.mkdir(parents=True)
    (deck / "decklist.txt").write_text(V1)
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", tmp_path / "data" / "decks")
    monkeypatch.setattr(dh, "_REPO_ROOT", tmp_path)
    doc = dv.report(SLUG)
    assert doc["versions"] == [] and doc["current_version"] is None
