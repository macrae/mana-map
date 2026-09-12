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
# `prd-2026-08.md` is here for the same reason as the design records: it is a
# SUPERSEDED document kept verbatim so that ~27 `PRD-v1 §N` citations resolve.
# Its counts are the counts of August, on purpose. `prd.md` is deliberately NOT
# excluded — it is live, and its intake notes are ours to keep accurate.
DESIGN_RECORDS = {"deck-builder-v2.md", "frontend-v2.md", "prd-2026-08.md"}

SURFACES = [
    ROOT / "CLAUDE.md",
    ROOT / "PLAN.md",
    ROOT / "README.md",
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
_NOT_A_COUNT += r"across|noticed|reported|failed|produced|spawned|returned|"
# `tokens` joined the list when the pattern started matching across a line
# break: `docs/agent-cost.md` says an enrichment "cost **0 agent tokens**",
# which is a count of TOKENS, not of agents. It was always a false positive —
# the newline was the only thing hiding it.
_NOT_A_COUNT += r"tokens?|spend|cost))"


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
            # `\s+`, NOT `[ ]+`: the wrap that hid a stale count for three days
            # fell BETWEEN "`manamap pilot`" and "subcommands", inside this
            # noun — not before the number.
            r"(?:`manamap pilot`|pilot)\s+subcommands?\b"),
        # Read from the PARSER, not from `len(STEPS) + 2`. That expression was
        # correct while every top-level subcommand was either a pipeline step,
        # `run` or `pilot` — and it silently undercounted the moment
        # `eval-commander-search` was added deliberately OUTSIDE `STEPS` (it
        # needs the network and a frozen snapshot, so `manamap run` must not
        # invoke it). A count derived from a proxy is a count that is right
        # until it isn't, which is the failure this whole module exists to catch.
        "top-level-subcommands": (
            _top_level_count(), r"top-level[ ]+subcommands?\b"),
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


def _top_level_count():
    """Every `manamap <x>` subcommand, from the real argument parser."""
    from manamap.cli import build_parser
    import argparse

    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return len(action.choices)
    raise AssertionError("no subparsers on the top-level parser")


TRUTHS = _truths()


@pytest.mark.parametrize("label", sorted(TRUTHS))
def test_no_surface_states_a_wrong_count(label):
    truth, noun = TRUTHS[label]
    # `\s` NOT `[ -]`: A COUNT THAT WRAPPED ACROSS A LINE WAS INVISIBLE TO THIS
    # GATE. PLAN.md carried "83 `manamap pilot`\nsubcommands" for three days
    # after the real figure became 85, while CLAUDE.md and README.md — which
    # happen to keep the phrase on one line — were caught the same afternoon.
    # Markdown reflows prose; a gate that only sees one line only sees the
    # surfaces that happen not to have wrapped there.
    pattern = re.compile(_NUM + r"[\s-]" + noun + _NOT_A_COUNT,
                         re.IGNORECASE)
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


# ── Three gates added 2026-09-12, each MEASURED before it was committed ──────
#
# The audit found twelve wrong counts in prose that this file already guards the
# shape of, so the first drafts of these were blunter and were narrowed on the
# evidence — which is the rule they are written under: a check that fires on
# correct data is worse than no check.
#
# Measured, first draft against final, on the repo as it stood:
#
#   "a bare `N subcommands` must be qualified"   5 hits, 1 true  → REWRITTEN as
#       an ANCHORED check on the `manamap --help` sentence: 1 hit, 1 true. The
#       four false ones were the audit and the plan QUOTING the defect, and the
#       magazine's own "eight subcommands", which is a correct count of a subset.
#       An unqualified count is ambiguous, but it is not always wrong, and a gate
#       cannot tell the difference from the phrase alone.
#
#   "the page count must equal len(viz/*.html)"  16 hits, 5 true → REPLACED by
#       `test_every_page_is_named_where_the_pages_are_listed`, which checks the
#       LIST rather than the number. Eleven of the sixteen were prose about a
#       SUBSET ("the two pages that share tokens.css") or the audit quoting the
#       defect. Naming is mechanical; counting prose is not.


def test_the_subcommand_count_beside_manamap_help_is_the_real_one():
    """CLAUDE.md said "see `manamap --help` for all 18 subcommands" and the
    parser had 28.

    `TRUTHS["top-level-subcommands"]` would have caught it, and did not, because
    that pattern requires the words "top-level" and this sentence does not have
    them. The docstring above already calls an unqualified count "a
    documentation bug in its own right" — nothing failed on one.

    So this is ANCHORED on the sentence rather than on the noun: a number
    introduced by `manamap --help` is a claim about the top-level parser
    whatever adjective it uses. Measured: one hit, the true one.
    """
    truth = _top_level_count()
    pattern = re.compile(r"manamap --help`?[^\n]{0,60}?(\d{1,4})\s+"
                         r"(?:top-level\s+)?subcommands?\b", re.IGNORECASE)
    wrong = []
    for path in SURFACES:
        if "history" in path.parts:
            continue
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            if int(match.group(1)) != truth:
                wrong.append(f"{path.relative_to(ROOT)}: {match.group().strip()!r}")
    assert not wrong, (
        f"`manamap --help` lists {truth} subcommands:\n  " + "\n  ".join(wrong))


#: Where the pages are ENUMERATED, and must stay enumerated. Not every doc that
#: mentions a page — only the two that hold the list a reader navigates by.
_PAGE_INDEXES = ("CLAUDE.md", "docs/viz.md")


def test_every_page_is_named_where_the_pages_are_listed():
    """`viz/spaces.html` shipped 2026-09-01, was documented in `docs/viz.md`,
    tested by `tests/test_viz_spaces_page.py`, linked from `shell.js`'s SURFACES
    nav — and absent from CLAUDE.md, which went on saying FOUR pages for eleven
    days.

    Checking the LIST rather than the count is what makes this measurable. A
    stated number is ambiguous — most prose that says "two pages" is talking
    about a subset and is correct — but a page that exists and is named nowhere
    is a fact.
    """
    pages = sorted(p.name for p in (ROOT / "viz").glob("*.html"))
    assert len(pages) >= 4, f"only {len(pages)} pages found — has viz/ moved?"
    missing = []
    for doc in _PAGE_INDEXES:
        text = (ROOT / doc).read_text(encoding="utf-8")
        missing += [f"{doc} does not name {name}" for name in pages
                    if name not in text]
    assert not missing, (
        "a page exists that the page list does not mention:\n  "
        + "\n  ".join(missing))


def test_the_docs_index_is_complete_and_its_sizes_are_real():
    """`docs/README.md`'s size column was stale on seventeen of its rows and
    mixed two units — the gotchas rows quoted BULLET counts (99 for a file of
    1,731 lines) while every other row quoted lines. Four files were in `docs/`
    and indexed nowhere.

    Both halves are mechanical, so neither should ever have drifted. The index
    is how a reader finds a doc; a row that is absent hides one, and a size that
    is wrong by 17x misrepresents what it costs to read.
    """
    index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    live = sorted(p.name for p in (ROOT / "docs").glob("*.md")
                  if p.name != "README.md")
    unindexed = [n for n in live if f"({n})" not in index]
    assert not unindexed, (
        "docs/ files missing from the index — add a row, or move the file to "
        f"docs/history/:\n  {unindexed}")

    wrong, checked = [], 0
    for match in re.finditer(r"\[([a-z0-9\-.]+\.md)\]\(\1\)\*{0,2} \| ([\d,]+) \|",
                             index):
        name, stated = match.group(1), int(match.group(2).replace(",", ""))
        target = ROOT / "docs" / name
        if not target.exists():                  # history rows carry a prefix
            continue
        checked += 1
        real = len(target.read_text(encoding="utf-8").splitlines())
        if real != stated:
            wrong.append(f"{name}: index says {stated}, file is {real} lines")
    assert checked >= 15, f"only {checked} index rows had a size to check"
    assert not wrong, ("stale sizes in docs/README.md — the column is LINES, "
                       "one unit:\n  " + "\n  ".join(wrong))
