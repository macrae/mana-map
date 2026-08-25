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


def test_an_all_digit_sha_prefix_still_resolves(repo, monkeypatch):
    """A git sha is hex, so a short prefix is all digits about one time in 27 —
    (10/16)**7. `resolve` matched the version-number branch on those and returned
    early, so a real sha resolved to nothing 3.7% of the time. It surfaced as a
    flake in the test above (passing alone, failing under the full suite) and
    would have hit any pilot whose commit happened to start with digits."""
    vers = dv.versions(SLUG)
    monkeypatch.setattr(dv, "versions", lambda slug: [
        {**v, "first_sha": "1234567890ab", "sha": "1234567890ab"} if v["version"] == 1 else v
        for v in vers])
    hit = dv.resolve(SLUG, "1234567")
    assert hit is not None, "an all-digit sha prefix must not be read as a version number"
    assert hit["version"] == 1
    # and a real version number still wins over a sha that starts with it
    assert dv.resolve(SLUG, "2")["version"] == 2


# ── The baseline: history restarts, nothing is destroyed ────────────────────

def test_a_baseline_restarts_numbering_at_the_list_it_names(repo):
    """A deck gets rebuilt in paper and the pilot wants its history to start
    there rather than carry the development scaffolding before it."""
    assert [v["version"] for v in dv.versions(SLUG)] == [1, 2]
    dv.set_baseline(SLUG, sha256=_sha(V2), note="the paper deck")
    vers = dv.versions(SLUG)
    assert [v["version"] for v in vers] == [1], "numbering must restart at the baseline"
    assert vers[0]["decklist_sha256"] == _sha(V2)


def test_the_baseline_names_a_LIST_not_a_commit(repo):
    """The load-bearing choice. A commit's sha is not knowable inside the commit
    that creates it — the same reason `versions.json` cannot be tracked — so a
    baseline anchored to a commit could never be written alongside the list it
    names. A content hash is known before anything is committed."""
    b = dv.set_baseline(SLUG, sha256=_sha(V2))
    assert "decklist_sha256" in b and "sha" not in b
    assert b["decklist_sha256"] == _sha(V2)


def test_baselining_the_working_list_needs_no_argument(repo):
    """The ordinary path: check a list in, then baseline it."""
    b = dv.set_baseline(SLUG)
    assert b["decklist_sha256"] == dv.working_sha(SLUG)
    assert [v["version"] for v in dv.versions(SLUG)] == [1]


def test_nothing_is_destroyed_by_a_baseline(repo):
    """Pre-baseline commits stay in git and stay reachable by sha. They simply
    stop being numbered, which is what 'pre-history' means."""
    before = dv.versions(SLUG)
    first_sha = before[0]["first_sha"]
    dv.set_baseline(SLUG, sha256=_sha(V2))
    assert dv.resolve(SLUG, first_sha) is None, "it is no longer a numbered version"
    assert dh._git("show", f"{first_sha}:data/decks/{SLUG}/decklist.txt") is not None, (
        "but the list itself is still in git")


def test_clearing_a_baseline_restores_the_full_history(repo):
    dv.set_baseline(SLUG, sha256=_sha(V2))
    assert len(dv.versions(SLUG)) == 1
    dv.set_baseline(SLUG, clear=True)
    assert [v["version"] for v in dv.versions(SLUG)] == [1, 2]


def test_a_baseline_naming_an_unknown_list_is_ignored_not_fatal(repo):
    """A hand-edited file, or a list that never landed. Numbering the whole
    history is a better failure than raising on every read."""
    dv.set_baseline(SLUG, sha256="0" * 64)
    assert [v["version"] for v in dv.versions(SLUG)] == [1, 2]


def test_report_says_the_deck_was_rebaselined(repo):
    """A deck showing one version could be new or could be re-baselined, and the
    page must be able to tell a reader which."""
    dv.set_baseline(SLUG, sha256=_sha(V2), note="the paper deck")
    doc = dv.report(SLUG)
    assert doc["baseline"]["note"] == "the paper deck"
    assert any("pre-baseline" in n or "restarts" in n for n in doc["notes"]), doc["notes"]


def test_a_baseline_and_a_paper_lock_coexist(repo):
    """Both are authored, both live in the same file, and the writer must keep
    them apart."""
    dv.set_baseline(SLUG, sha256=_sha(V2))
    dv.set_paper(SLUG, ref="V1")          # V1 is now the baselined list
    dv.tag(SLUG, "v1.0.0", ref="V1")
    doc = json.loads((repo / dv.TAGS_FILE).read_text())
    assert list(doc) == ["slug", "baseline", "paper", "tags"], "key order is authored"
    assert dv.report(SLUG)["paper"]["in_sync"] is True


def test_semver_tags_a_baselined_version(repo):
    """`v1.0.0` must not be shadowed by version-number resolution: it starts with
    a V, but '1.0.0'.isdigit() is False, so it falls through to the tag lookup."""
    dv.set_baseline(SLUG, sha256=_sha(V2))
    dv.tag(SLUG, "v1.0.0", ref="V1")
    assert dv.resolve(SLUG, "v1.0.0")["version"] == 1


def test_a_V_prefixed_miss_does_not_fall_through_to_sha_matching(repo, monkeypatch):
    """`V9` is an unambiguous request for version 9. Letting it reach the sha
    matcher meant a hex sha beginning with 9 answered it — about one time in
    seven, which is a wrong answer that looks like a flake."""
    vers = dv.versions(SLUG)
    monkeypatch.setattr(dv, "versions", lambda slug: [
        {**v, "first_sha": "9abcdef01234", "sha": "9abcdef01234"} for v in vers])
    assert dv.resolve(SLUG, "V9") is None, "a V-prefixed miss must be a miss"
    # …while a BARE number still falls through, because it may be a sha prefix
    assert dv.resolve(SLUG, "9") is not None


# ── Release tags ───────────────────────────────────────────────────────────
#
# A tag is where the pilot says how big a change was (docs/pilot.md, "What a
# version bump means"). The numbering from git is mechanical and says nothing
# about size; the tag is the judgement, so it has to sort and mean what it says.


def test_releases_sort_by_number_not_by_string():
    """`v1.10.0` sorts BEFORE `v1.9.0` under plain lexical order — the tenth
    minor bump files itself between the first and the second and stays there.

    A deck reaches v1.10.0 by shipping ten changes that alter what it can do,
    which is an ordinary year, so this is a bug with a date on it.
    """
    names = ["v1.9.0", "v1.10.0", "v1.2.0", "the-lock", "v2.0.0", "v1.0.0"]
    assert sorted(names) != sorted(names, key=dv._tag_key), (
        "lexical and semantic order agree here — pick a sharper case")
    assert sorted(names, key=dv._tag_key) == [
        "v1.0.0", "v1.2.0", "v1.9.0", "v1.10.0", "v2.0.0", "the-lock"], (
        "releases must sort numerically, and nicknames after them")


@pytest.mark.parametrize("name", ["v1.2", "v1", "1.2.3.4", "v2"])
def test_a_near_miss_release_tag_is_refused(repo, name):
    """Something that is nothing but digits and dots plainly MEANT to be a
    release. Filed as a nickname it sorts alphabetically among the real ones and
    looks correct until there are enough versions for the order to matter.
    """
    with pytest.raises(SystemExit) as e:
        dv.tag(SLUG, name, ref="V1")
    assert "release tag" in str(e.value)


@pytest.mark.parametrize("name", ["the-lock", "3rd-rebuild", "2026-rebuild"])
def test_a_nickname_that_starts_with_a_digit_is_still_a_nickname(repo, name):
    """The near-miss rule reads the WHOLE name, not its first character. A first
    cut matched `^v?\\d` and refused both of the last two, which are fine."""
    dv.tag(SLUG, name, ref="V1")
    doc = json.loads((repo / dv.TAGS_FILE).read_text())
    assert name in doc["tags"]


def test_a_tag_is_a_claim_about_one_list(repo):
    """Re-tagging is a silent overwrite: the old version keeps its games and
    loses its label, and every artifact that quoted the name now points at a
    different 99."""
    dv.tag(SLUG, "v1.0.0", ref="V1")
    with pytest.raises(SystemExit) as e:
        dv.tag(SLUG, "v1.0.0", ref="V2")
    assert "one exact list" in str(e.value)

    # Re-tagging the SAME version is idempotent, not an error — re-running a
    # command must not fail because it already succeeded.
    dv.tag(SLUG, "v1.0.0", ref="V1")

    # And --force is the way to actually move it.
    dv.tag(SLUG, "v1.0.0", ref="V2", force=True)
    doc = json.loads((repo / dv.TAGS_FILE).read_text())
    assert doc["tags"]["v1.0.0"]["version"] == 2


# ── Sleeving proposes the release ──────────────────────────────────────────


def test_sleeving_proposes_v1_0_0_when_there_is_no_release(repo):
    """v0.x is a list; v1.0.0 is a deck you can hold.

    A deck lives on the bench at v0.1.0, v0.4.2, whatever — digital, unproven,
    freely rewritten. Sleeving it is the act that makes it real, so that is
    where 1.0.0 belongs and the major version then means something physical.
    """
    v = dv.versions(SLUG)[0]
    lines = dv.release_suggestion(SLUG, v)
    assert lines, "a freshly sleeved deck with no release got no suggestion"
    assert "v1.0.0" in " ".join(lines)
    assert f"--at V{v['version']}" in " ".join(lines), "the proposal must name the version"


def test_the_proposal_does_not_write_a_tag(repo):
    """THE TOOL PROPOSES; THE PILOT CONFIRMS.

    Auto-tagging would also make `paper` non-idempotent — a re-run would either
    fail on the duplicate or silently move the name, and re-running a command
    must not fail for having already succeeded.
    """
    dv.set_paper(SLUG, ref="V1", note="sleeved")
    dv.release_suggestion(SLUG, dv.versions(SLUG)[0])
    doc = json.loads((repo / dv.TAGS_FILE).read_text())
    assert doc.get("tags") == {}, "the suggestion wrote a tag behind the pilot's back"


def test_it_goes_quiet_once_the_deck_has_released(repo):
    """Silent when this version carries a release, and silent when the DECK has
    released before — proposing v1.0.0 to a deck already at v2.1.0 is worse than
    proposing nothing."""
    vs = dv.versions(SLUG)
    dv.tag(SLUG, "v1.0.0", ref="V1")
    assert dv.release_suggestion(SLUG, vs[0]) == [], "still proposing on a tagged version"
    assert dv.release_suggestion(SLUG, vs[-1]) == [], (
        "proposed v1.0.0 for a later version of a deck that has already released")


def test_a_nickname_does_not_count_as_a_release(repo):
    """`the-lock` is a name, not a version. A deck tagged only with nicknames
    has still never released, and must still be offered 1.0.0."""
    dv.tag(SLUG, "the-lock", ref="V1")
    assert dv.release_suggestion(SLUG, dv.versions(SLUG)[0]), (
        "a nickname suppressed the release proposal")
