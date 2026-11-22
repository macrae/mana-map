"""No surface calls the goldfish the decider (2026-11-21).

The pilot said "the goldfish is not the final arbiter" at least fourteen times in
forty-six days, and the words that kept it one lived in `CLAUDE.md` ("the decision loop
is the goldfish"), the vision ("the decision instrument"), the doctor's charter ("`try` is
the primary evidence") and the release skill ("the primary row"). A memory note did not
hold; a test does. The phrases are banned on every surface an agent or a session reads.
`docs/history/` is the record and may say anything.
"""

import re

from conftest import ROOT

#: Each phrase, and the rule it would reintroduce.
BANNED = {
    "decision instrument": "the goldfish is one instrument among the study's sources",
    "the decision loop is the goldfish": "the decision is a study; the goldfish is its fast instrument",
    "the decision loop is `try`": "the decision is a study; `try` is the goldfish's read",
    "the goldfish decides": "the study decides; the goldfish reads",
    "DO NOT MERGE": "no instrument's word is an imperative; the five states describe the guard's read",
    "primary evidence": "the log, the recon and the audit lead; the goldfish is one input",
    "the primary row": "a report is read as a whole profile by axis, never one row",
}

SURFACES = ("CLAUDE.md", "README.md", "PLAN.md", "docs/*.md", ".claude/skills/*/SKILL.md",
            ".claude/agents/*.md", ".claude/agents-common.md")


def _files():
    out = []
    for pat in SURFACES:
        out += [p for p in ROOT.glob(pat) if p.is_file() and "history" not in p.parts]
    return sorted(set(out))


def test_no_surface_calls_the_goldfish_the_decider():
    offenders, checked = [], 0
    for path in _files():
        checked += 1
        text = path.read_text(encoding="utf-8")
        for phrase in BANNED:
            for m in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE):
                line = text.count("\n", 0, m.start()) + 1
                offenders.append(f"{path.relative_to(ROOT)}:{line}: {phrase!r} — {BANNED[phrase]}")
    assert checked >= 40, f"only {checked} surfaces scanned — the glob moved"
    assert not offenders, "the goldfish is being called the decider again:\n  " + "\n  ".join(offenders)
