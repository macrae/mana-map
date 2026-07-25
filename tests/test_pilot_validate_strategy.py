"""Strategy form validator: bad-doc fixtures, changelog checks, citation dispatch."""

from manamap.pilot.validate_strategy import parse_strategy, validate_changelog

GOOD_SECTION = """\
## strategy:tempo — Tempo

Prose about tempo.

Sources:
- Reid Duke, "Level One: Tempo" — https://example.com/tempo
"""


def errors_of(doc):
    _, errors = parse_strategy(doc)
    return errors


def test_good_doc_no_errors():
    assert errors_of(GOOD_SECTION) == []


def test_duplicate_id():
    errors = errors_of(GOOD_SECTION + "\n" + GOOD_SECTION)
    assert any("duplicate section id strategy:tempo" in e for e in errors)


def test_malformed_id():
    doc = GOOD_SECTION.replace("strategy:tempo", "strategy:Tempo!")
    assert any("malformed section id" in e for e in errors_of(doc))


def test_missing_sources_block():
    doc = "## strategy:tempo — Tempo\n\nProse only, no sources.\n"
    assert any("missing Sources block" in e for e in errors_of(doc))


def test_source_bullet_needs_url_or_print():
    doc = GOOD_SECTION.replace(
        '- Reid Duke, "Level One: Tempo" — https://example.com/tempo',
        "- Reid Duke, some article I half remember",
    )
    assert any("needs a URL or '(print)'" in e for e in errors_of(doc))


def test_orphan_child():
    doc = (
        "### strategy:tempo.mana — Mana\n\nProse.\n\nSources:\n"
        "- A, \"B\" — https://example.com\n"
    )
    assert any("parent strategy:tempo does not exist" in e for e in errors_of(doc))


def test_pillar_must_not_be_dotted():
    doc = GOOD_SECTION.replace("## strategy:tempo —", "## strategy:tempo.fast —")
    assert any("pillar (##) ids must not be dotted" in e for e in errors_of(doc))


def test_section_without_prose():
    doc = "## strategy:tempo — Tempo\n\nSources:\n- A, \"B\" — https://example.com\n"
    assert any("section has no prose" in e for e in errors_of(doc))


def test_heading_without_strategy_id():
    doc = GOOD_SECTION + "\n## Loose Heading\n\nStray prose.\n"
    assert any("heading without a strategy: id" in e for e in errors_of(doc))


# ── Changelog ────────────────────────────────────────────────────────────

KNOWN = {"strategy:tempo", "strategy:card-advantage"}


def test_changelog_good():
    log = (
        "# Strategy Changelog\n\n"
        "## 2026-07-24 — initial seed\n"
        "- added strategy:tempo — seeded from founder baseline\n"
        "- amended strategy:card-advantage with virtual card advantage\n"
    )
    assert validate_changelog(log, KNOWN) == []


def test_changelog_bad_heading():
    log = "## July 24 — initial seed\n- added strategy:tempo\n"
    assert any("entry heading must be" in e for e in validate_changelog(log, KNOWN))


def test_changelog_bad_verb():
    log = "## 2026-07-24 — x\n- tweaked strategy:tempo\n"
    assert any("bullet must start" in e for e in validate_changelog(log, KNOWN))


def test_changelog_unknown_id():
    log = "## 2026-07-24 — x\n- added strategy:nonexistent\n"
    errors = validate_changelog(log, KNOWN)
    assert any("no such section exists" in e for e in errors)


def test_changelog_deprecated_may_be_absent():
    log = "## 2026-07-24 — x\n- deprecated strategy:old-thinking replaced by tempo\n"
    assert validate_changelog(log, KNOWN) == []


def test_changelog_requires_an_entry():
    assert any("no dated entries" in e for e in validate_changelog("# Title\n", KNOWN))


# ── strategy: citations through the stack/decision citation contract ─────


def test_strategy_citation_verbatim_quote_passes():
    from manamap.pilot.validate_stack import _validate_citations

    strategy_sections = {"strategy:tempo": {"text": "Spend all your mana every turn."}}
    errors = []
    _validate_citations(
        [{"rule": "strategy:tempo", "quote": "all your mana every turn"}],
        {}, "branch 0", errors, strategy_sections=strategy_sections,
    )
    assert errors == []


def test_strategy_citation_bad_quote_fails():
    from manamap.pilot.validate_stack import _validate_citations

    strategy_sections = {"strategy:tempo": {"text": "Spend all your mana every turn."}}
    errors = []
    _validate_citations(
        [{"rule": "strategy:tempo", "quote": "hoard your mana forever"}],
        {}, "branch 0", errors, strategy_sections=strategy_sections,
    )
    assert any("not verbatim" in e for e in errors)


def test_strategy_citation_nonexistent_section_fails():
    from manamap.pilot.validate_stack import _validate_citations

    errors = []
    _validate_citations(
        [{"rule": "strategy:nope", "quote": "anything"}],
        {}, "branch 0", errors, strategy_sections={},
    )
    assert any("nonexistent" in e for e in errors)
