"""No prose may restate the section count — the spec is the only authority.

`issue_spec.DEPARTMENTS` grew from 15 to 17 and four places kept saying 15: the
magazine-editor's own prompt (which also enumerated the old ids, in the old
order, three lines after telling itself to read the spec), STYLEv3's review
checklist, the design-issue skill, and a CLI help string. A transcribed list
goes stale the moment a section is added; this test makes that a failure.
"""

import re
from pathlib import Path

from manamap.pilot.issue_spec import DEPARTMENT_IDS
# One pruned walk shared with `test_docs_counts.py`, instead of a full-tree glob
# per reference. `tests/repo_tree.py` records what that cost.
from repo_tree import exists_anywhere

ROOT = Path(__file__).resolve().parent.parent
# Every surface that tells an agent or a human how many sections there are.
# `docs/` and CLAUDE.md are included deliberately: an audit found the earlier,
# narrower version let a stale count sit in the docs and pass green.
# Design records are excluded: they deliberately quote the numbers of their own
# era ("of 32 sections, roughly 10 contain a deckbuilding clause") and carry
# inline departure marks where reality moved. Rewriting them would destroy the
# record; the whole point of a design doc is that it dates.
DESIGN_RECORDS = {"deck-builder-v2.md", "frontend-v2.md"}
SURFACES = [
    ROOT / "STYLEv3.md",
    ROOT / "CLAUDE.md",
    ROOT / "PLAN.md",
    ROOT / "README.md",
    *sorted((ROOT / ".claude").rglob("*.md")),
    *[p for p in sorted((ROOT / "docs").glob("*.md"))
      if p.name not in DESIGN_RECORDS],
    *sorted((ROOT / "src" / "manamap" / "pilot").glob("*.py")),
]

_WORDS = {"fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18}
# `[^|\n]` guards against markdown table rows ("| # | Section |"), where the
# number and the noun are different cells rather than one phrase.
# The lookbehind rejects heading numbers ("§5.1 Section specifications") and the
# `[ -]` (not `\s`) rejects markdown table rows, where the number and the noun are
# different cells rather than one phrase.
_COUNT_RE = re.compile(
    r"(?<![\d.])\b(\d{1,2}|fifteen|sixteen|seventeen|eighteen)"
    r"[ -](?:fixed[ ]+)?(?:departments?|sections?)\b", re.IGNORECASE)


def test_no_surface_states_a_wrong_section_count():
    truth = len(DEPARTMENT_IDS)
    wrong = []
    for path in SURFACES:
        if "history" in path.parts:
            continue
        for match in _COUNT_RE.finditer(path.read_text(encoding="utf-8")):
            token = match.group(1).lower()
            stated = _WORDS.get(token, int(token) if token.isdigit() else None)
            if stated is not None and stated != truth:
                wrong.append(f"{path.relative_to(ROOT)}: {match.group()!r} "
                             f"(spec says {truth})")
    assert not wrong, "stale section counts:\n  " + "\n  ".join(wrong)


def test_no_surface_hardcodes_the_department_id_list():
    """An enumerated id list in a prompt is the same bug with more surface.

    The probe is DERIVED, not typed. It used to be the literal string
    "cover, contents, first-turns" — the opening of the order at the time — and
    the v3.4 resequence would have turned it into a sentinel that matches
    nothing and silently protects nothing, which is worse than no test.
    """
    probe = ", ".join(DEPARTMENT_IDS[:3])
    offenders = [p.relative_to(ROOT) for p in SURFACES
                 if "history" not in p.parts and probe in p.read_text(encoding="utf-8")]
    assert not offenders, (
        f"{offenders} enumerate section ids; read issue_spec.DEPARTMENT_IDS instead")


def test_every_step_module_agrees_with_the_registry_about_its_number():
    """A step's number lives in `pipeline.STEPS`; the module must not disagree.

    Two had drifted and neither could be caught by reading either file alone:
    `train.py` said "Step 4" where the registry says "Step 4a" (the layout/function
    split), and `eval_embeddings.py` said "Step 14" — a number `viz_index.py`
    already owned, so the pipeline had two step 14s and no step 15 while every
    doc referred to "step 15" for exactly this module.

    The registry is the authority because it is what `manamap run --from STEP`
    executes. Docstring numbers are documentation, and documentation that
    contradicts the runner is worse than none: it is the thing a reader checks.

    **Read by AST, not by import.** `importlib.import_module` over every step
    pulls in torch, sentence-transformers and hdbscan — 3.3 s, and the source of
    the three `SwigPy*` DeprecationWarnings printed on every run of the whole
    suite — to read a string literal the parser can see for free. That is the
    same argument `test_pilot_imports.py` already makes, and the same reason the
    pipeline's own registry uses lazy imports.
    """
    import ast
    import re

    from manamap.pipeline import STEPS

    def module_docstring(module_path):
        path = ROOT / "src" / (module_path.replace(".", "/") + ".py")
        if not path.exists():
            return None
        return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))

    wrong = []
    for name, module_path, description in STEPS:
        expected = re.match(r"Step (\S+?):", description).group(1)
        doc = module_docstring(module_path)
        if doc is None:
            wrong.append(f"{module_path}: registry names a module with no source file")
            continue
        first = doc.strip().splitlines()[0] if doc.strip() else ""
        found = re.match(r"Step (\S+?):", first)
        if not found:
            wrong.append(f"{module_path}: docstring does not open 'Step N: …'")
        elif found.group(1) != expected:
            wrong.append(f"{module_path}: says Step {found.group(1)}, "
                         f"registry says Step {expected}")
    assert not wrong, "step numbers disagree with pipeline.STEPS:\n  " + "\n  ".join(wrong)


def test_no_two_pipeline_steps_share_a_number():
    """Uniqueness only — the prose count is `test_docs_counts.py`'s job.

    That file already derives the pipeline's step count from the registry via
    `cli.pipeline_step_count()` and checks every surface against it, including the
    subtlety that `len(STEPS)` is 16 while the highest declared number is 15
    (`train` and `train-ability` are 4a and 4b). Duplicating that here would be two
    validators answering one question, which this repo has paid for before.

    What nothing else checks is that the numbers are DISTINCT. They were not:
    `eval_embeddings.py` and `viz_index.py` both claimed step 14, so the registry
    described a pipeline with two step 14s and no step 15 — and a count check
    passes happily on that, because the highest number was still 14 when the docs
    said 15... which read as a stale doc rather than a duplicated step.
    """
    import re

    from manamap.pipeline import STEPS

    numbers = [re.match(r"Step (\S+?):", d).group(1) for _, _, d in STEPS]
    duplicates = {n for n in numbers if numbers.count(n) > 1}
    assert not duplicates, f"step numbers used more than once: {sorted(duplicates)}"


def test_docs_pilot_lists_every_pilot_subcommand():
    """`docs/pilot.md`'s command block is the discovery surface — it must be complete.

    Nine subcommands existed only in the CLI: `mana-analysis`, `scenario-facts`,
    `diagnosis-report`, `merge-prose`, `cache-snapshot`, `cache-rerecord` and three
    validators. A reader looking for "how do I check the goldfish declaration"
    found nothing and concluded there was no way, which is how a validator that
    already exists gets written twice.

    Checked against the registry, not against a hand-kept list here — the same
    argument `build-index` makes for the deck manifest.
    """
    import re

    from manamap.pilot.registry import PILOT_STEPS

    documented = set(re.findall(r"^manamap pilot ([a-z][a-z-]+)",
                                (ROOT / "docs" / "pilot.md").read_text(encoding="utf-8"),
                                re.MULTILINE))
    real = {name for name, *_ in PILOT_STEPS} if not isinstance(PILOT_STEPS, dict) \
        else set(PILOT_STEPS)
    missing = sorted(real - documented)
    assert not missing, (
        "pilot subcommands missing from docs/pilot.md's command block:\n  "
        + "\n  ".join(missing))


def test_every_tracked_per_deck_artifact_is_documented():
    """A tracked file nobody documented is a file nobody can safely delete.

    Found this way: `data/decks/radagast/None` — 25 KB of superseded deck map,
    written by an early `deck-map` run that handed `resolve_out_path` a literal
    `None` (the call sites all guard with `if out:` now, and `deck_map.main` says
    so in a comment). Nothing read it, its city labels were all null, and it sat
    tracked for weeks because no inventory ever compared the directory against
    the docs.

    Filenames only — semantics live in `docs/pilot.md`. Per-stack, per-decision,
    per-prescription, per-simulation-run and per-experiment files are excluded:
    they are keyed instances of five documented shapes, not distinct artifacts.

    `/experiments/` joined that list late and the omission turned `main` red the
    moment the first one was tracked: `experiments/<id>.json` is documented in
    `docs/data-artifacts.md` exactly like the other four, but the run id in the
    filename made it read as an undocumented one-off. When a command starts
    writing a keyed instance under a new directory, it belongs here in the same
    commit.
    """
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "data/decks/"], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout.split()
    instanced = ("/stacks/", "/decisions/", "/prescriptions/", "/sim/", "/experiments/")
    names = sorted({Path(t).name for t in tracked
                    if not any(d in t for d in instanced)})
    prose = ((ROOT / "docs" / "data-artifacts.md").read_text(encoding="utf-8")
             + (ROOT / "docs" / "pilot.md").read_text(encoding="utf-8"))
    undocumented = [n for n in names if n not in prose]
    assert not undocumented, (
        "tracked per-deck files documented nowhere:\n  " + "\n  ".join(undocumented)
        + "\n(document them, or delete them if they are strays)")


def test_live_docs_do_not_name_source_files_that_do_not_exist():
    """A backtick-quoted filename reads as "this exists"; if it doesn't, the reader hunts.

    Caught `test_viz_camera.py` in CLAUDE.md's list of source-assertion test files —
    retired during the camera-logic consolidation, and `docs/testing.md` says so
    correctly two files away.

    **The first version of this check fired on that correct mention too**, which by
    this repo's own rule makes it worse than no check: 270 references scanned, one
    true positive and one false positive. It is narrowed by looking at the prose
    AROUND the reference — a filename inside a sentence about retirement, deletion
    or supersession is a record, not a claim. Re-measured after narrowing: zero
    false positives across every live doc, and reintroducing the CLAUDE.md defect
    still trips it. Both halves were checked; a threshold verified only on the
    passing side proves nothing.

    DESIGN RECORDS ARE EXEMPT wholesale, for the same reason they are exempt from
    the section-count check: `frontend-v2.md` and `deck-builder-v2.md` deliberately
    name files that were planned and never built. PLAN.md is checked, but a path
    into a directory that does not exist yet is a PROPOSAL and passes — a plan for
    `viz/js/engine/constants.js` is fine while `viz/js/engine/` does not exist, and
    stops being fine the moment it does.
    """
    import re

    # A mention inside a sentence about the file's removal is a record, not a claim.
    gone = re.compile(r"retired|deleted|removed|no longer|is gone|was gone|never built|"
                      r"used to|formerly|superseded|historical", re.IGNORECASE)
    live = [ROOT / "CLAUDE.md", ROOT / "README.md", ROOT / "PLAN.md",
            *[p for p in sorted((ROOT / "docs").glob("*.md"))
              if p.name not in DESIGN_RECORDS]]
    missing = []
    for doc in live:
        text = doc.read_text(encoding="utf-8")
        for match in re.finditer(r"`([a-zA-Z0-9_/.-]+\.(?:py|js|html|css))`", text):
            ref = match.group(1)
            if "/" in ref:
                if (ROOT / ref).exists():
                    continue
                if not (ROOT / ref).parent.is_dir():
                    continue          # a proposal, not a claim
            elif exists_anywhere(ref):
                continue
            window = text[max(0, match.start() - 220):match.end() + 220]
            if not gone.search(window):
                missing.append(f"{doc.name}: {ref}")
    assert not missing, ("live docs name files that do not exist:\n  "
                         + "\n  ".join(sorted(missing)))
