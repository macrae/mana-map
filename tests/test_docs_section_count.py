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
    """An enumerated id list in a prompt is the same bug with more surface."""
    probe = "cover, contents, first-turns"      # the old order's opening
    offenders = [p.relative_to(ROOT) for p in SURFACES
                 if "history" not in p.parts and probe in p.read_text(encoding="utf-8")]
    assert not offenders, (
        f"{offenders} enumerate section ids; read issue_spec.DEPARTMENT_IDS instead")
