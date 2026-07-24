"""Tests for the comprehensive-rules chunker and rules DB (pilot subsystem)."""

import pytest

from manamap.pilot.build_rules_db import build_embed_text, kebab_term, parse_rules
from manamap.pilot.common import RULE_ID_RE, load_rules_db

from conftest import requires_rules


# A miniature CR file exercising the edge cases: TOC repeats section headers,
# lettered sub-rules, an Example line, a curly quote, and a glossary block.
CR_FIXTURE = """Magic: The Gathering Comprehensive Rules

These rules are effective as of June 19, 2026.

Contents

1. Game Concepts
601. Casting Spells and Activating Activated Abilities
Glossary
Credits

1. Game Concepts

601. Casting Spells and Activating Activated Abilities

601.1. Previously, the action of casting a spell was referred to on cards as “playing” that spell.

601.2. To cast a spell is to take it from where it is, put it on the stack, and pay its costs.
Example: A player casts a creature spell during their main phase.

601.2a To propose the casting of a spell, a player first moves that card to the stack.

601.2b If the spell is modal, the player announces the mode choice.

601.2e The game checks whether the proposed spell can legally be cast.

Glossary

Casting a Spell
To take a card from where it is, put it on the stack, and pay its costs.

Storm
A keyword ability that copies a spell. See rule 702.40, “Storm.”

Credits

Wizards of the Coast
"""


def test_chunk_ids_all_valid():
    chunks = parse_rules(CR_FIXTURE)
    assert chunks, "no chunks parsed"
    for c in chunks:
        assert RULE_ID_RE.match(c["id"]), f"invalid chunk id: {c['id']!r}"


def test_numbered_and_lettered_rules_parsed():
    ids = [c["id"] for c in parse_rules(CR_FIXTURE)]
    for expected in ["601.1", "601.2", "601.2a", "601.2b", "601.2e"]:
        assert expected in ids


def test_toc_does_not_create_chunks():
    # The TOC lists "601. Casting Spells..." — a section header, never a chunk.
    ids = [c["id"] for c in parse_rules(CR_FIXTURE)]
    assert "601" not in ids
    assert len(ids) == len(set(ids)), "duplicate chunk ids (TOC leaked into chunks?)"


def test_example_line_attached_to_owning_rule():
    chunks = {c["id"]: c for c in parse_rules(CR_FIXTURE)}
    assert "Example: A player casts a creature spell" in chunks["601.2"]["text"]


def test_curly_quotes_preserved():
    chunks = {c["id"]: c for c in parse_rules(CR_FIXTURE)}
    assert "“playing”" in chunks["601.1"]["text"]


def test_section_and_parent():
    chunks = {c["id"]: c for c in parse_rules(CR_FIXTURE)}
    assert chunks["601.2a"]["section"].startswith("601. Casting Spells")
    assert chunks["601.2a"]["parent"] == "601.2"
    assert chunks["601.2"]["parent"] == "601"


def test_glossary_chunks():
    chunks = {c["id"]: c for c in parse_rules(CR_FIXTURE)}
    assert "glossary:casting-a-spell" in chunks
    assert "glossary:storm" in chunks
    assert chunks["glossary:storm"]["section"] == "Glossary"
    assert "copies a spell" in chunks["glossary:storm"]["text"]


def test_credits_dropped():
    for c in parse_rules(CR_FIXTURE):
        assert "Wizards of the Coast" not in c["text"]


def test_kebab_term():
    assert kebab_term("Casting a Spell") == "casting-a-spell"
    assert kebab_term("Active Player, Nonactive Player Order") == "active-player-nonactive-player-order"
    assert kebab_term("Owner's Control") == "owner's-control"


def test_embed_text_has_context_prefix():
    chunks = {c["id"]: c for c in parse_rules(CR_FIXTURE)}
    embed = build_embed_text(chunks["601.2a"])
    assert embed.startswith("601.2a 601. Casting Spells")


# ── Data-gated: real rules DB ──


@requires_rules
def test_real_db_known_rules_present():
    rules, order, embeddings = load_rules_db()
    assert "601.2a" in rules
    assert "702.40a" in rules
    assert "copy" in rules["702.40a"]["text"].lower()
    assert len(order) == embeddings.shape[0]


@requires_rules
def test_real_db_all_ids_valid():
    rules, _, _ = load_rules_db()
    bad = [r for r in rules if not RULE_ID_RE.match(r)]
    assert not bad, f"invalid ids in real DB: {bad[:10]}"
