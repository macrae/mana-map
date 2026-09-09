"""No credential may enter this repository, and the check runs before commit.

Written the day a key was pasted into a chat rather than into a file — which is
the more common way one leaks, and the one a repo cannot defend against. What a
repo CAN defend against is the second mistake: the key going into `.env`, or a
config, or a doc showing "how to set it up", and then into a commit.

`.env` was not in `.gitignore` until this file was written. Nothing had gone
wrong yet only because nobody had created one, and `git add -A` is the habit
here — every commit in this session used it.

SCOPED TO NOT FIRE ON CORRECT DATA, which is this repo's oldest rule about
validators: the pattern matches Anthropic's actual key prefix followed by real
key material, so a doc saying `export ANTHROPIC_API_KEY=sk-...` is fine and
intended. Six proposed checks have been rejected here for firing on correct
data; this one has to earn its place the same way.
"""

import re
import subprocess

from manamap import config

#: Assembled rather than written out, so this file does not itself contain the
#: literal it is looking for — a test that trips its own check is a test nobody
#: can grep for.
_ANTHROPIC = re.compile("sk-" + "ant-api" + r"\d{2}-[A-Za-z0-9_\-]{40,}")
_GENERIC = re.compile(r"(?i)\b(api[_-]?key|secret|token)\s*[=:]\s*['\"][A-Za-z0-9_\-]{32,}")

#: Where a key would plausibly land. Not the whole tree: `data/` holds card
#: oracle text full of arbitrary strings, and scanning it would be a validator
#: firing on correct data.
_SCAN = ("src", "tests", "docs", ".claude", "viz")
_SCAN_FILES = ("CLAUDE.md", "README.md", "PLAN.md", "pyproject.toml", "Makefile")


def _tracked_text_files():
    out = subprocess.run(["git", "ls-files"], cwd=config._REPO_ROOT,
                         capture_output=True, text=True, check=True)
    for rel in out.stdout.splitlines():
        if rel.startswith(_SCAN) or rel in _SCAN_FILES:
            path = config._REPO_ROOT / rel
            if path.is_file() and path.suffix not in (".png", ".bin", ".npy"):
                yield rel, path


def test_no_tracked_file_carries_a_credential():
    checked, offenders = 0, []
    for rel, path in _tracked_text_files():
        text = path.read_text(errors="replace")
        if _ANTHROPIC.search(text) or _GENERIC.search(text):
            offenders.append(rel)
        checked += 1
    assert checked > 100, f"only scanned {checked} files — the walk is broken"
    assert not offenders, f"a credential is committed in: {offenders}"


def test_the_pattern_would_actually_catch_one():
    """A secret scanner nobody has fired once is a scanner that matches nothing.

    Proven against a synthetic key of the right SHAPE, and against the
    placeholder form that must NOT match — because a doc explaining the variable
    is correct data, and a check that fires on it would be turned off within a
    week.
    """
    synthetic = "sk-" + "ant-api03-" + "A" * 60
    assert _ANTHROPIC.search(synthetic), "the pattern misses a real-shaped key"
    assert not _ANTHROPIC.search("export ANTHROPIC_API_KEY=sk-..."), (
        "the pattern fires on the documented placeholder")
    assert not _ANTHROPIC.search("ANTHROPIC_API_KEY in the environment")


def test_env_files_are_ignored_before_one_exists():
    """The ordering is the whole point. A `.gitignore` entry added after the
    file is created has already lost."""
    ignored = (config._REPO_ROOT / ".gitignore").read_text()
    for pattern in (".env", ".env.*"):
        assert pattern in ignored, f"{pattern} is not ignored"
    check = subprocess.run(["git", "check-ignore", "-q", ".env"],
                           cwd=config._REPO_ROOT)
    assert check.returncode == 0, "git does not actually ignore .env"
