"""Strategy DB: chunker/parser unit tests + data-gated checks on the real DB."""

import numpy as np
import pytest

from conftest import requires_strategy
from manamap.pilot.common import STRATEGY_ID_RE
from manamap.pilot.validate_strategy import oversize_warnings, parse_strategy

STRATEGY_FIXTURE = """\
# Test Strategy Companion

Preamble prose that must never become a chunk.

## strategy:tempo — Tempo

Maximizing mana efficiency each turn to dictate the pace of the game,
even at the cost of long-term card value.

Sources:
- Reid Duke, "Level One: Tempo" — https://example.com/level-one-tempo
- Some Author, "Tempo Book" (print)

### strategy:tempo.mana-efficiency — Mana Efficiency

Spending all your mana every turn is the simplest tempo heuristic.

Sources:
- Reid Duke, "Level One: Mana" — https://example.com/mana

## strategy:card-advantage — Card Advantage

Generating more total resources than your opponent through draw,
two-for-ones, or board persistence.

Sources:
- Mike Flores, "Who's the Beatdown?" — https://example.com/beatdown
"""


@pytest.fixture(scope="module")
def parsed():
    chunks, errors = parse_strategy(STRATEGY_FIXTURE)
    assert errors == []
    return chunks


def test_preamble_not_chunked(parsed):
    assert all("Preamble" not in c["text"] for c in parsed)
    assert len(parsed) == 3


def test_ids_valid_and_unique(parsed):
    ids = [c["id"] for c in parsed]
    assert len(ids) == len(set(ids))
    assert all(STRATEGY_ID_RE.match(i) for i in ids)


def test_parent_and_section_fields(parsed):
    by_id = {c["id"]: c for c in parsed}
    assert by_id["strategy:tempo"]["parent"] is None
    assert by_id["strategy:tempo.mana-efficiency"]["parent"] == "strategy:tempo"
    assert by_id["strategy:tempo.mana-efficiency"]["section"] == "Tempo"
    assert by_id["strategy:card-advantage"]["section"] == "Card Advantage"


def test_sources_extracted_and_excluded_from_text(parsed):
    by_id = {c["id"]: c for c in parsed}
    tempo = by_id["strategy:tempo"]
    assert len(tempo["sources"]) == 2
    assert "Level One: Tempo" in tempo["sources"][0]
    assert "Sources" not in tempo["text"]
    assert "https://" not in tempo["text"]
    assert tempo["text"].startswith("Maximizing mana efficiency")


def test_embed_text_has_context_prefix(parsed):
    from manamap.pilot.build_strategy_db import build_embed_text

    embed = build_embed_text(parsed[0])
    assert embed.startswith("strategy:tempo Tempo:")


def test_oversize_warning():
    big = STRATEGY_FIXTURE.replace(
        "Spending all your mana every turn is the simplest tempo heuristic.",
        "word " * 400,
    )
    chunks, errors = parse_strategy(big)
    assert errors == []
    warnings = oversize_warnings(chunks)
    assert len(warnings) == 1
    assert "strategy:tempo.mana-efficiency" in warnings[0]


# ── Data-gated: real strategy DB ─────────────────────────────────────────


@requires_strategy
def test_real_db_known_sections_exist():
    from manamap.pilot.common import load_strategy_db

    sections, order, embeddings = load_strategy_db()
    for expected in ("strategy:tempo", "strategy:card-advantage", "strategy:whos-the-beatdown"):
        assert expected in sections


@requires_strategy
def test_real_db_aligned_and_normalized():
    from manamap.pilot.common import load_strategy_db

    sections, order, embeddings = load_strategy_db()
    assert len(order) == embeddings.shape[0] == len(sections)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


@requires_strategy
def test_real_db_all_ids_valid_and_sourced():
    from manamap.pilot.common import load_strategy_db

    sections, _, _ = load_strategy_db()
    for sid, entry in sections.items():
        assert STRATEGY_ID_RE.match(sid), sid
        assert entry["sources"], f"{sid} has no sources"
