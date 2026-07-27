"""Pilot: mechanically enforce form on the strategy knowledge base.

The strategy doc (data/strategy/strategy.md) is tier-★ grounding: curated,
sourced, human-reviewed coaching theory. Substance is judged by founder review
of tracked diffs; this module enforces *form*, mirroring validate_stack.py:

- Every chunkable heading is `## strategy:<id> — <Title>` (pillar) or
  `### strategy:<parent>.<child> — <Title>` (subsection), IDs unique and
  matching STRATEGY_ID_RE; a child's dotted parent must exist as a pillar.
- Every section ends with a `Sources:` block of `- Author, "Title" — URL`
  bullets (URL required; `(print)` allowed for books).
- CHANGELOG.md entries are dated, and every bullet is
  `added|amended|renamed|deprecated strategy:<id> ...`, where added/amended
  IDs must exist in the doc (renamed/deprecated ones may be gone).

The parser lives here (not in build_strategy_db) so validation never pays the
sentence-transformers import.
"""

import re

from manamap.config import (
    STRATEGY_CHANGELOG_PATH,
    STRATEGY_DOC_PATH,
    STRATEGY_SECTION_WARN_CHARS,
)
from manamap.pilot.common import STRATEGY_ID_RE, report_errors

HEADING_RE = re.compile(r"^(#{2,3})\s+(strategy:\S+)\s+—\s+(\S.*)$")
SOURCE_BULLET_RE = re.compile(r"^- \S.*$")
CHANGELOG_ENTRY_RE = re.compile(r"^## \d{4}-\d{2}-\d{2} — \S.*$")
CHANGELOG_BULLET_RE = re.compile(
    r"^- (added|amended|renamed|deprecated) (strategy:\S+)(\s.*)?$"
)


def _source_ok(bullet):
    """A source bullet must carry a URL or be explicitly marked print."""
    return "http" in bullet or bullet.rstrip().endswith("(print)")


def parse_strategy(text):
    """Parse strategy.md into chunks. Returns (chunks, errors).

    chunks: [{"id", "title", "text", "section", "parent", "sources"}]
    text is verbatim prose (the quote-substring check runs against it);
    the Sources block is extracted to metadata, never part of text.
    """
    chunks = []
    errors = []
    current = None          # chunk being accumulated
    in_sources = False
    pillar_title = ""       # owning ## title, becomes `section` for children

    def finalize():
        nonlocal current, in_sources
        if current is None:
            return
        current["text"] = "\n".join(current.pop("_lines")).strip()
        if not current["text"]:
            errors.append(f"{current['id']}: section has no prose")
        if not current["sources"]:
            errors.append(f"{current['id']}: missing Sources block (>=1 bullet required)")
        chunks.append(current)
        current, in_sources = None, False

    for lineno, line in enumerate(text.split("\n"), 1):
        stripped = line.rstrip()
        m = HEADING_RE.match(stripped)
        if m:
            finalize()
            hashes, section_id, title = m.groups()
            if not STRATEGY_ID_RE.match(section_id):
                errors.append(f"line {lineno}: malformed section id {section_id!r}")
            level = len(hashes)
            if level == 2:
                if "." in section_id.split(":", 1)[1]:
                    errors.append(f"{section_id}: pillar (##) ids must not be dotted")
                pillar_title = title
                parent = None
            else:
                if "." not in section_id.split(":", 1)[1]:
                    errors.append(f"{section_id}: subsection (###) ids must be dotted")
                parent = section_id.rsplit(".", 1)[0]
            current = {
                "id": section_id, "title": title, "section": pillar_title,
                "parent": parent, "sources": [], "_lines": [],
            }
            in_sources = False
            continue
        if stripped.startswith("##") and current is not None and "strategy:" not in stripped:
            errors.append(f"line {lineno}: heading without a strategy: id — {stripped!r}")
            finalize()
            continue
        if current is None:
            continue  # preamble / between sections
        if stripped == "Sources:":
            in_sources = True
            continue
        if in_sources:
            if not stripped:
                continue
            if SOURCE_BULLET_RE.match(stripped):
                if not _source_ok(stripped):
                    errors.append(
                        f"{current['id']}: source bullet needs a URL or '(print)': {stripped!r}"
                    )
                current["sources"].append(stripped[2:])
            else:
                errors.append(
                    f"{current['id']}: unexpected content after Sources block: {stripped!r}"
                )
            continue
        current["_lines"].append(line)
    finalize()

    # Cross-section checks: uniqueness and parentage.
    seen = set()
    ids = set()
    for c in chunks:
        if c["id"] in seen:
            errors.append(f"duplicate section id {c['id']}")
        seen.add(c["id"])
        ids.add(c["id"])
    for c in chunks:
        if c["parent"] and c["parent"] not in ids:
            errors.append(f"{c['id']}: parent {c['parent']} does not exist as a section")
    return chunks, errors


def oversize_warnings(chunks):
    """Sections past the embed-truncation comfort zone (MiniLM ~256 tokens)."""
    return [
        f"{c['id']}: {len(c['text'])} chars > {STRATEGY_SECTION_WARN_CHARS} — "
        f"consider splitting into ### subsections (embedding truncates long text)"
        for c in chunks
        if len(c["text"]) > STRATEGY_SECTION_WARN_CHARS
    ]


def validate_changelog(text, known_ids):
    """Form-check CHANGELOG.md against the doc's section ids. Returns errors."""
    errors = []
    has_entry = False
    for lineno, line in enumerate(text.split("\n"), 1):
        stripped = line.rstrip()
        if stripped.startswith("## "):
            if CHANGELOG_ENTRY_RE.match(stripped):
                has_entry = True
            else:
                errors.append(
                    f"changelog line {lineno}: entry heading must be "
                    f"'## YYYY-MM-DD — summary', got {stripped!r}"
                )
            continue
        if stripped.startswith("- "):
            m = CHANGELOG_BULLET_RE.match(stripped)
            if not m:
                errors.append(
                    f"changelog line {lineno}: bullet must start "
                    f"'added|amended|renamed|deprecated strategy:<id>', got {stripped!r}"
                )
                continue
            verb, section_id = m.group(1), m.group(2)
            if not STRATEGY_ID_RE.match(section_id):
                errors.append(f"changelog line {lineno}: malformed id {section_id!r}")
            elif verb in {"added", "amended"} and section_id not in known_ids:
                errors.append(
                    f"changelog line {lineno}: {verb} {section_id} but no such "
                    f"section exists in strategy.md"
                )
    if not has_entry:
        errors.append("changelog has no dated entries")
    return errors


def main(args=None):
    if not STRATEGY_DOC_PATH.exists():
        raise SystemExit(f"{STRATEGY_DOC_PATH} not found — nothing to validate.")
    chunks, errors = parse_strategy(STRATEGY_DOC_PATH.read_text(encoding="utf-8"))
    known_ids = {c["id"] for c in chunks}

    if STRATEGY_CHANGELOG_PATH.exists():
        errors += validate_changelog(
            STRATEGY_CHANGELOG_PATH.read_text(encoding="utf-8"), known_ids
        )
    else:
        errors.append(f"{STRATEGY_CHANGELOG_PATH} not found — every doc needs its changelog")

    for warning in oversize_warnings(chunks):
        print(f"WARN {warning}")
    report_errors(
        "strategy doc", errors,
        f"OK   strategy.md — {len(chunks)} sections, form holds; CHANGELOG.md consistent")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-strategy`.")
