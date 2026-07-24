"""Tests for rules-DB querying: semantic top-k and exact lookup (pilot)."""

import pytest

from manamap.pilot.query_rules import lookup, query

from conftest import requires_rules


@requires_rules
def test_storm_query_returns_702_40():
    results = query("storm", k=8)
    ids = [rule_id for rule_id, _, _ in results]
    assert any(i.startswith("702.40") or i == "glossary:storm" for i in ids), ids


@requires_rules
def test_query_scores_descending():
    results = query("counter target spell", k=5)
    scores = [s for _, _, s in results]
    assert scores == sorted(scores, reverse=True)


@requires_rules
def test_exact_lookup():
    rule = lookup("702.40a")
    assert rule["id"] == "702.40a"
    assert rule["section"].startswith("702.")
    assert rule["text"]


@requires_rules
def test_lookup_miss_raises_with_suggestion():
    # "601" is a section header, not a rule chunk — the miss must suggest 601.x rules.
    with pytest.raises(KeyError) as exc:
        lookup("601")
    message = str(exc.value)
    assert "601" in message and "Did you mean" in message


@requires_rules
def test_lookup_total_miss():
    with pytest.raises(KeyError):
        lookup("999.99")
