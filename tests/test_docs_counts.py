"""Every count stated in prose must match what the repo actually contains.

Documentation drifts one number at a time and nothing notices, because a wrong
count reads exactly like a right one. A survey of this repo once found the test
total stated in five files with four different values, the pilot subcommand
count off by twelve, and a whole published issue missing from three inventories.

So the counts are DERIVED here and the prose is checked against them. Adding a
subcommand, an agent, a deck or a test now fails this test until the surfaces
that quote it are updated — which is the point: the failure message names the
file and the phrase.

Only facts the repo can answer **exactly and cheaply** are guarded. The test
total is deliberately not among them: 144 of the cases are parametrized over
lists computed at collection time, so the only way to count them is to run
pytest, and doing that from inside pytest recurses. It also changes on almost
every commit. The durable fix there is editorial rather than mechanical — the
number lives in `docs/testing.md` alone, next to the command that prints it,
instead of in six files that drift apart.

Two deliberate exclusions:

* `docs/history/` and the design records (`deck-builder-v2.md`,
  `frontend-v2.md`) quote the numbers of their own era on purpose. A design doc
  is supposed to date; rewriting one destroys the record it exists to be.
* Ranges and approximations ("~40 groups", "2-6 spawns") are not matched.
"""

import pathlib
import re

import pytest

# Shared with `test_docs_section_count.py`: one pruned walk instead of a
# full-tree `rglob` per name. See `tests/repo_tree.py`.
from repo_tree import exists_anywhere

ROOT = pathlib.Path(__file__).resolve().parent.parent
DESIGN_RECORDS = {"deck-builder-v2.md", "frontend-v2.md"}

SURFACES = [
    ROOT / "CLAUDE.md",
    ROOT / "PLAN.md",
    ROOT / "README.md",
    ROOT / "STYLEv3.md",
    *sorted((ROOT / ".claude").rglob("*.md")),
    *[p for p in sorted((ROOT / "docs").glob("*.md"))
      if p.name not in DESIGN_RECORDS],
]

_WORDS = {
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_NUM = r"(?<![\d.,])([\d,]{1,6}|" + "|".join(_WORDS) + r")"

# A count STATEMENT ("12 agents, 15 skills") versus prose that happens to count
# something else ("seven agents read the wrong deck's numbers"). Without this,
# the guard fires on six correct charters and one correct gotcha — and a check
# that fails on accurate text trains people to ignore it.
_NOT_A_COUNT = r"(?!\s+(?:read|hit|share|shared|wrote|ran|work(?:ing|ed)?|"
_NOT_A_COUNT += r"across|noticed|reported|failed|produced|spawned|returned))"


def _truths():
    """Everything this test knows, derived — never typed.

    Cheap by construction: a registry length, a glob, a directory listing. No
    subprocess, so this cannot recurse into its own collection.
    """
    from manamap.config import AGENT_ROUTINES
    from manamap.pilot.registry import PILOT_STEPS

    from manamap.pipeline import STEPS

    decks = [p for p in (ROOT / "data" / "decks").iterdir() if p.is_dir()]
    return {
        # Two different subcommand counts exist and prose must say which it
        # means: `manamap <step>` (the pipeline, plus `run` and `pilot`) and
        # `manamap pilot <cmd>`. An unqualified "N subcommands" is ambiguous by
        # construction, so both patterns require their qualifier and an
        # unqualified phrase is a documentation bug in its own right.
        "pilot-subcommands": (
            len(PILOT_STEPS),
            r"(?:`manamap pilot`|pilot)[ ]+subcommands?\b"),
        "top-level-subcommands": (
            len(STEPS) + 2, r"top-level[ ]+subcommands?\b"),
        "agents": (
            len(list((ROOT / ".claude" / "agents").glob("*.md"))),
            r"agents?\b"),
        "skills": (
            len(list((ROOT / ".claude" / "skills").glob("*/"))),
            r"skills?\b"),
        "published-issues": (
            len(decks), r"issues?[ ]+(?:live|published)\b"),
        "cached-routines": (
            len(AGENT_ROUTINES), r"(?:static[ ]+)?routines?\b"),
    }


TRUTHS = _truths()


@pytest.mark.parametrize("label", sorted(TRUTHS))
def test_no_surface_states_a_wrong_count(label):
    truth, noun = TRUTHS[label]
    pattern = re.compile(_NUM + r"[ -]" + noun + _NOT_A_COUNT, re.IGNORECASE)
    wrong = []
    for path in SURFACES:
        if "history" in path.parts:
            continue
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            token = match.group(1).lower().replace(",", "")
            stated = _WORDS.get(token, int(token) if token.isdigit() else None)
            if stated is not None and stated != truth:
                wrong.append(f"{path.relative_to(ROOT)}: {match.group().strip()!r} "
                             f"(actual: {truth})")
    assert not wrong, (
        f"stale {label} count — the repo has {truth}:\n  " + "\n  ".join(wrong))


def test_the_pipeline_step_count_is_never_a_literal():
    """`manamap --help` derives it; prose must agree with the registry.

    The registry holds 16 entries but numbers them 1-15, because `train` and
    `train-ability` are steps 4a and 4b. `len(STEPS)` is therefore the WRONG
    derivation and the highest declared step number is the right one.
    """
    from manamap.cli import pipeline_step_count

    truth = pipeline_step_count()
    pattern = re.compile(_NUM + r"[ -]step\b" + _NOT_A_COUNT, re.IGNORECASE)
    wrong = []
    for path in SURFACES:
        if "history" in path.parts:
            continue
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            token = match.group(1).lower().replace(",", "")
            stated = _WORDS.get(token, int(token) if token.isdigit() else None)
            if stated is not None and stated != truth:
                wrong.append(f"{path.relative_to(ROOT)}: {match.group().strip()!r}")
    assert not wrong, (
        f"the pipeline has {truth} numbered steps:\n  " + "\n  ".join(wrong))


def test_no_surface_names_a_deleted_module():
    """A doc naming a file that no longer exists is worse than no doc.

    Each of these was deleted and left behind references that read as current.
    The list is FIXED — the docstring used to claim otherwise — but each entry is
    re-verified as still-gone before it is enforced, so a resurrected module drops
    out instead of failing forever.

    It coexists with `test_docs_section_count.py`'s general version, which derives
    from the filesystem instead of a list. The two have genuinely different
    coverage and neither subsumes the other: this one matches **unquoted** prose
    mentions of these seven specific names, while the general check only inspects
    backtick-quoted references (anything looser drowns in false positives) but
    covers *every* file rather than a curated seven. The general one also exempts a
    mention sitting in a sentence about the file's removal, which is correct for it
    and would defeat this one's purpose.
    """
    deleted = [
        "deck-builder.js", "deck-map.js", "mana-map.js.map",
        "sideboard_facts.py", "validate_sideboard.py",
        "upgrade_facts.py", "validate_upgrade_watch.py",
    ]
    still_gone = [name for name in deleted if not exists_anywhere(name)]
    offenders = []
    for path in SURFACES:
        if "history" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for name in still_gone:
            if name in text:
                offenders.append(f"{path.relative_to(ROOT)} names {name}")
    assert not offenders, (
        "docs reference modules that do not exist:\n  " + "\n  ".join(offenders))
