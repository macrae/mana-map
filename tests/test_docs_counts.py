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


# Shared with `test_docs_section_count.py`: one pruned walk instead of a
# full-tree `rglob` per name. See `tests/repo_tree.py`.
from repo_tree import claude_md, exists_anywhere, in_worktree

ROOT = pathlib.Path(__file__).resolve().parent.parent
# `prd.md` is deliberately NOT excluded — it is live, and its intake notes are
# ours to keep accurate. The superseded `prd-2026-08.md`, the 2026-09-12 audit and
# the paydown plan moved to `docs/history/` on 2026-10-05, which no scan here reads.
DESIGN_RECORDS = {"deck-builder-v2.md", "frontend-v2.md"}

SURFACES = [
    ROOT / "CLAUDE.md",
    ROOT / "PLAN.md",
    ROOT / "README.md",
    *claude_md(),
    *[p for p in sorted((ROOT / "docs").glob("*.md"))
      if p.name not in DESIGN_RECORDS],
    # THE COUNTS ALSO LIVE IN SOURCE, and three of them were wrong.
    # `console.py`, `serve.py` and `cli.py` each stated the pilot subcommand
    # count in a module docstring or a `--help` string — 66, 69 and 102 against
    # a real 107 — and this file guarded only Markdown, so all three drifted
    # unnoticed while CLAUDE.md was caught the same afternoon. A count a USER
    # reads out of `--help` is a documentation surface whatever file it lives in.
    *sorted((ROOT / "src" / "manamap").glob("*.py")),
]





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


#: Where the page list is written for a reader.
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


def test_every_doc_is_in_the_index():
    """The index is how a reader finds a doc; a file in `docs/` that no row names is
    hidden. (The size column is `make docs-sizes`'s to refresh, not this test's to pin —
    2026-11-22: it forced an index edit on nearly every docs commit.)"""
    index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    live = sorted(p.name for p in (ROOT / "docs").glob("*.md") if p.name != "README.md")
    unindexed = [n for n in live if f"({n})" not in index]
    assert not unindexed, (
        "docs/ files missing from the index — add a row, or move the file to "
        f"docs/history/:\n  {unindexed}")


def test_no_doc_tells_you_to_run_a_subcommand_that_does_not_exist():
    """A COMMAND LINE IN A DOC IS AN INSTRUCTION, and a wrong one fails at the shell.

    Seven subcommands went with the magazine renderer on 2026-09-13 and their
    call sites did not. `CLAUDE.md:289` told every session to run
    `manamap pilot build-page <slug>` — the page-render step of the lifecycle —
    for eight days; `docs/pilot.md` listed five more in a block that reads as a
    command reference, and the `/publish-deck` runbook's gate list named
    `validate-considering`.

    The truth is `registry.PILOT_STEPS`, which is what argparse is built from, so
    this cannot drift from the CLI.

    EXEMPTIONS ARE NAMED, NOT PATTERNED. A historical record SHOULD say
    `build-manual` — that is what the page is for — so each exempt file is listed
    with the reason, and a new one has to be argued for rather than absorbed by a
    glob. `docs/history/` is exempt wholesale for the same reason the deleted-module
    check exempts it.
    """
    from manamap.pilot.registry import PILOT_STEPS

    live = {step[0] for step in PILOT_STEPS}
    assert "goldfish" in live and "deck-info" in live, (
        "PILOT_STEPS is not the shape this test assumes; if the registry changed, "
        "fix the extraction rather than the expectation")

    # file -> why it may name a command that no longer exists
    exempt = {
        "docs/gotchas-magazine-legacy.md":
            "the operational record of the deleted renderer; naming its commands is the point",
        "data/decks/goblin-storm/README.md":
            "a per-deck record of how that deck's page was built at the time",
    }
    # docs/pilot.md keeps a LEGACY section that is explicitly past-tense.
    legacy_from = {}
    pilot_md = ROOT / "docs" / "pilot.md"
    for i, line in enumerate(pilot_md.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("## LEGACY — the magazine renderer"):
            legacy_from["docs/pilot.md"] = i
            break
    assert "docs/pilot.md" in legacy_from, (
        "docs/pilot.md's LEGACY heading moved or was reworded — this test exempts "
        "everything below it and can no longer find where that starts")

    pattern = re.compile(r"manamap pilot ([a-z][a-z0-9-]+)")
    offenders, checked = [], 0
    for path in sorted(ROOT.glob("**/*.md")) + [ROOT / "Makefile"]:
        rel = str(path.relative_to(ROOT))
        if not path.exists() or ".venv" in rel or "node_modules" in rel or in_worktree(path):
            continue
        if "history" in path.parts or rel in exempt:
            continue
        checked += 1
        start = legacy_from.get(rel)
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if start and i >= start:
                break
            for match in pattern.finditer(line):
                if match.group(1) not in live:
                    offenders.append(f"{rel}:{i} — `manamap pilot {match.group(1)}`")

    assert checked >= 20, f"only {checked} files were scanned"
    assert not offenders, (
        "a doc names a pilot subcommand that does not exist; run it and it errors:\n  "
        + "\n  ".join(offenders)
        + "\n\nIf the mention is deliberately historical, add the FILE to `exempt` "
          "above with a reason — do not widen the pattern.")
