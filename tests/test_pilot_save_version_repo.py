"""`save-version` against a repository with a real deck's history shape.

The unit tier's `test_save_version_commits_only_the_decks_paths` builds a
seven-card toy deck in an empty repo. This one is the realistic init: a tmp git
repo holding the REAL meren-recursion directory (every artifact it tracks), the
real manifest, the real `.gitignore`, its handbook page and an unrelated tracked
doc, in one prior commit. A swap goes through `deck_edit.edit` against the real
corpus; `save_version` runs with its consistency tail stubbed (the tail is
regen + build-index, which are tested elsewhere) except that the stub rewrites
`index.json`, the way `build-index` does when a list moves.

Realistic init rather than `git clone --local --no-hardlinks` of the repo: the
pack is ~300 MB, and the shape that matters — a deck dir with its full artifact
set, a manifest, ignore rules and other tracked files — is all here.
"""
import json
import shutil
import subprocess

import pytest

from manamap import config
from manamap.pilot import deck_edit

SLUG = "meren-recursion"
OUT, IN = "Sol Ring", "Necromancy"
REAL_ROOT = config.DECKS_DIR.parent.parent

pytestmark = [
    pytest.mark.regression,
    pytest.mark.skipif(not (config.DATA_DIR / "cards.csv").exists()
                       or not (config.DECKS_DIR / SLUG / "decklist.txt").exists(),
                       reason=f"needs the corpus and the tracked {SLUG} deck"),
]


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                          text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    decks = root / "data" / "decks"
    decks.mkdir(parents=True)
    # Only what a checkout tracks: the gitignored local state stays behind.
    shutil.copytree(config.DECKS_DIR / SLUG, decks / SLUG,
                    ignore=shutil.ignore_patterns("edits.jsonl*", ".*.lock", "*.bak",
                                                  "*.prev", ".agent-out"))
    shutil.copy2(config.DECKS_DIR / "index.json", decks / "index.json")
    shutil.copy2(REAL_ROOT / ".gitignore", root / ".gitignore")
    poh = REAL_ROOT / "manuals" / "p" / f"{SLUG}.html"
    if poh.exists():
        (root / "manuals" / "p").mkdir(parents=True)
        shutil.copy2(poh, root / "manuals" / "p" / poh.name)
    (root / "docs").mkdir()
    (root / "docs" / "notes.md").write_text("tracked, unrelated\n")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "a prior commit")

    from manamap.pilot import deck_context, deck_history
    monkeypatch.setattr(config, "DECKS_DIR", decks)
    monkeypatch.setattr(deck_history, "_REPO_ROOT", root)
    monkeypatch.setattr(deck_edit, "_NAME_MEMO", {})

    def consistency(slug, echo):
        # build-index's part of the tail: the manifest moves with the list.
        doc = json.loads((decks / "index.json").read_text())
        doc["_test_rebuilt"] = True
        (decks / "index.json").write_text(json.dumps(doc, indent=1) + "\n")
        return {"ran": ["build-index"], "failures": [], "invalid": []}
    monkeypatch.setattr(deck_edit, "_consistency", consistency)
    monkeypatch.setattr(deck_context, "list_change", lambda slug, before_text=None: None)
    return root


def test_save_version_is_one_commit_of_the_decks_paths(repo):
    from manamap.pilot import deck_versions
    before_versions = len(deck_versions.versions(SLUG))
    head = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "docs" / "notes.md").write_text("an edit the pilot has not committed\n")

    deck_edit.edit(SLUG, [{"op": "swap", "out": OUT, "in": IN}])
    assert deck_edit.unsaved(SLUG) == {"since": "git HEAD", "out": {OUT: 1}, "in": {IN: 1}}

    r = deck_edit.save_version(SLUG, "a reanimation slot for the rock")

    assert _git(repo, "rev-list", "--count", f"{head}..HEAD").strip() == "1"
    committed = _git(repo, "show", "--name-only", "--format=", "HEAD").split()
    allowed = (f"data/decks/{SLUG}/", "data/decks/index.json")
    assert committed and all(p.startswith(allowed) for p in committed), committed
    assert f"data/decks/{SLUG}/decklist.txt" in committed
    assert "data/decks/index.json" in committed
    assert not any("edits.jsonl" in p or p.endswith((".lock", ".bak")) for p in committed)

    msg = _git(repo, "log", "-1", "--format=%B")
    assert "a reanimation slot for the rock" in msg.splitlines()[0]
    assert f"+in {IN} / −out {OUT}" in msg, msg

    assert len(deck_versions.versions(SLUG)) == before_versions + 1
    assert r["version"] == deck_versions.report(SLUG)["current_version"]
    assert deck_edit.unsaved(SLUG) is None

    # The pilot's unrelated edit is still modified and still uncommitted.
    assert _git(repo, "status", "--porcelain", "--", "docs/notes.md").startswith(" M")
