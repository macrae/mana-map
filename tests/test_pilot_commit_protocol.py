"""The one rule that makes a git-derived artifact trackable.

`versions.json` is derived from `git log --follow -- decklist.txt`, and a
version's row carries the sha and date of the commit that created it. Those are
not knowable *inside* that commit — so a `versions.json` written in the same
commit as the decklist change it describes is one version behind the moment it
lands, and the freshness gate fails with a byte diff that explains nothing.

The rule that fixes it is one sentence: **a commit may not contain both a
decklist and its own versions.json.** Change the list in one commit; regenerate
and commit the derived artifacts in the next.

WHY THIS IS NOT A NEW BURDEN. The pilot already does it. `deck-version <slug>
paper` and `deck-version <slug> tag` both write a COMMIT SHA of the list they
name into the tracked `deck_versions.json` — which is only possible in a later
commit than the one being named. This test makes the existing habit explicit,
and turns an inscrutable freshness failure into an instruction.

WHY A TEST AND NOT A COMMENT. `docs/` has carried the reasoning for this class
of artifact since `versions.json` was first gitignored, and the deploy-time step
that reasoning promised was never built — so the deck page's rap sheet rendered
its empty state to every reader on the internet for months. A rule nothing
checks is a rule that decays into prose.
"""

import subprocess

import pytest

from manamap.config import DECKS_DIR

REPO = DECKS_DIR.parent.parent


def _files_in(rev):
    """Paths touched by one commit, or None when git cannot answer."""
    out = subprocess.run(
        ["git", "show", "--name-only", "--pretty=format:", rev],
        capture_output=True, text=True, cwd=REPO, check=False)
    if out.returncode != 0:
        return None
    return [p for p in out.stdout.split("\n") if p.strip()]


def _slug_of(path, leaf):
    parts = path.split("/")
    if len(parts) == 4 and parts[0] == "data" and parts[1] == "decks" and parts[3] == leaf:
        return parts[2]
    return None


def _violations(paths):
    """Slugs whose decklist AND versions.json are both in one changeset."""
    lists = {s for s in (_slug_of(p, "decklist.txt") for p in paths) if s}
    versions = {s for s in (_slug_of(p, "versions.json") for p in paths) if s}
    return sorted(lists & versions)


def test_the_working_tree_does_not_stage_a_decklist_with_its_version_list():
    """The check that fires while you can still act on it.

    Staged, not committed — so the answer arrives before the commit exists
    rather than after, which is the only moment the fix is cheap.
    """
    out = subprocess.run(["git", "diff", "--cached", "--name-only"],
                         capture_output=True, text=True, cwd=REPO, check=False)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    bad = _violations(out.stdout.split("\n"))
    assert not bad, (
        f"staged together for {bad}: a decklist and its versions.json. A "
        f"version's sha is not knowable inside the commit that creates it, so "
        f"the version list has to be regenerated AFTERWARDS:\n"
        f"    git restore --staged data/decks/{bad[0]}/versions.json\n"
        f"    git commit            # the decklist change\n"
        f"    make manuals && git add data/decks && git commit   # the record")


def test_no_commit_in_recent_history_broke_the_protocol():
    """The same rule held at rest over what is already committed.

    Bounded to recent history on purpose: the rule starts when `versions.json`
    became tracked (2026-09-02), and asserting it over the whole history would
    fail on commits made when the file was gitignored and could not have been
    in them. A rule that fires on correct past data is the validator failure
    this repo names in CLAUDE.md.
    """
    out = subprocess.run(
        ["git", "log", "--format=%H", "-40"],
        capture_output=True, text=True, cwd=REPO, check=False)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    revs = [r for r in out.stdout.split("\n") if r.strip()]
    if not revs:
        pytest.skip("no history (a shallow clone?)")
    checked = 0
    for rev in revs:
        paths = _files_in(rev)
        if paths is None:
            continue
        # Only commits that actually carry a tracked version list can break it.
        if not any(_slug_of(p, "versions.json") for p in paths):
            continue
        checked += 1
        bad = _violations(paths)
        assert not bad, (
            f"commit {rev[:12]} changed both the decklist and versions.json "
            f"for {bad} — regenerate the version list in a separate commit")
    # No floor assertion: until the first version list is committed there is
    # genuinely nothing to check, and inventing a minimum would make a correct
    # repo fail on the day the rule shipped.
    assert checked >= 0
