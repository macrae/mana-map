"""One retrieval core over four corpora, and the evidence the collapse was safe.

`query_rules` and `query_strategy` were two copies of the same twenty lines of
ranking. `docs` and `code` would have made four. They now share
`retrieve.search`, and the old modules delegate rather than being deleted —
they are the agent interface, and their tuple shapes are what callers read.

REFACTORING A RANKING FUNCTION IS THE KIND OF CHANGE THAT LOOKS FINE. A
mis-normalized query still returns plausible rules in a plausible order, and
nothing errors; the only way to know is to compare against what it returned
before. `test_the_delegation_is_byte_identical` is that comparison, run live
against both implementations rather than against a stored snapshot, so it cannot
rot into asserting yesterday's index.
"""

import numpy as np
import pytest

from manamap import config
from manamap.pilot import build_corpus, retrieve

from conftest import requires_data, requires_rules, requires_strategy

DOCS_BUILT = config.DOCS_INDEX_PATH.exists()
CODE_BUILT = config.CODE_INDEX_PATH.exists()

requires_docs = pytest.mark.skipif(
    not DOCS_BUILT, reason="run `manamap pilot build-docs-db`")
requires_code = pytest.mark.skipif(
    not CODE_BUILT, reason="run `manamap pilot build-code-db`")


def _old_rules_query(text, k):
    """The implementation that existed before the collapse, inline.

    Kept HERE rather than in the module it replaced, so the module can be clean
    and the comparison still runs against the real prior behaviour.
    """
    from manamap.ingest.preprocess import compute_text_embeddings
    from manamap.pilot.common import load_rules_db

    rules, order, embeddings = load_rules_db()
    q = compute_text_embeddings([text])[0]
    q = q / max(np.linalg.norm(q), 1e-8)
    scores = embeddings @ q
    top = np.argsort(-scores)[:k]
    return [(order[i], rules[order[i]]["text"], float(scores[i])) for i in top]


# ── the collapse ──────────────────────────────────────────────────────────

@requires_data
@requires_rules
def test_the_delegation_is_byte_identical():
    """The refactor's only real evidence. Both implementations, same queries."""
    from manamap.pilot import query_rules

    checked = 0
    for question in ["can I respond to a triggered ability",
                     "state based actions", "commander tax", "priority"]:
        assert query_rules.query(question, k=5) == _old_rules_query(question, 5), (
            f"the delegation changed the ranking for {question!r}")
        checked += 1
    assert checked == 4


@requires_data
@requires_rules
def test_the_old_modules_keep_their_tuple_shapes():
    """`query_rules` returns 3-tuples and `query_strategy` 4-tuples. Callers —
    including two agent charters — read those positionally."""
    from manamap.pilot import query_rules

    hits = query_rules.query("triggered ability", k=2)
    assert all(len(h) == 3 for h in hits)
    rule_id, text, score = hits[0]
    assert isinstance(rule_id, str) and isinstance(text, str)
    assert 0.0 <= score <= 1.0


@requires_data
@requires_rules
def test_a_lookup_miss_keeps_its_wording():
    """The rules-checker charter quotes this sentence. An agent grepping for it
    must keep finding it."""
    from manamap.pilot import query_rules

    with pytest.raises(KeyError, match="not found in the rules index"):
        query_rules.lookup("999.999")


def test_an_unknown_corpus_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="docs"):
        retrieve.search("cards", "anything")


# ── chunking ──────────────────────────────────────────────────────────────

def test_a_section_with_no_blank_lines_is_still_split():
    """THE BUG THIS CAUGHT ON ITS FIRST BUILD. `docs/gotchas-bench.md` produced
    a single 107,197-character chunk, because its longest section is one
    continuous markdown table and the splitter only cut on blank lines.

    That is not cosmetic overflow: a 107 KB chunk embeds to a vector that means
    nothing in particular, never ranks, and takes the passage a question wanted
    down with it. The most expensively-earned content in the repo was one dead
    chunk. Re-introduce it by splitting only on "\\n\\n" and this fails.
    """
    table = "\n".join(f"| row {i} | some measured value {i} |" for i in range(400))
    # Driven through `_pack`, the PRODUCTION path — an earlier version of this
    # test called `_paragraphs` directly and passed while `_pack` was ignoring
    # it entirely, which is the exact bug it was written for.
    chunks = [text for _anchor, _title, text
              in build_corpus._pack([("t", "A table", table)], "f.md")]
    assert len(chunks) > 1, "a table with no blank lines was never split"
    assert max(len(c) for c in chunks) <= config.CORPUS_MAX_CHARS * 1.1
    # And nothing is lost — every row survives somewhere.
    assert sum(c.count("| row ") for c in chunks) == 400


def test_a_tiny_section_is_folded_rather_than_indexed_alone():
    """A lone heading embeds to noise and then crowds out the real passage."""
    sections = [("a", "A", "short"), ("b", "B", "x" * 500)]
    packed = list(build_corpus._pack(sections, "f.md"))
    assert len(packed) == 1, "the stub should have merged forward"
    assert "short" in packed[0][2] and "x" * 500 in packed[0][2]


def test_history_is_not_indexed():
    """`docs/history/` holds what is no longer true, by design. Surfacing it
    beside current docs is how a reader cites an abandoned design."""
    sources = {c["source"] for c in build_corpus.collect("docs")}
    assert not any(s.startswith("docs/history/") for s in sources)
    assert len(sources) >= 10


def test_the_embedded_string_carries_the_title():
    """"gotchas" and "the rules that bite" are the words a question uses, and a
    body often never restates them."""
    chunk = {"title": "The rules that bite", "text": "A body that never says so."}
    assert chunk["title"] in build_corpus.embed_text(chunk)


# ── the built indexes ─────────────────────────────────────────────────────

@requires_data
@requires_docs
def test_the_docs_index_answers_a_question_about_this_repo():
    hits = retrieve.search("docs", "why does the goldfish disagree with Forge?", k=5)
    assert hits and hits[0][2] > 0.4
    sources = {rec.get("source") for _id, rec, _s in hits}
    assert any("gotchas" in s or "prd" in s or "known-issues" in s for s in sources), (
        f"expected a gotchas/prd/known-issues hit, got {sources}")


@requires_data
@requires_docs
def test_every_stored_chunk_can_be_fetched_by_its_own_id():
    """The ids `search` returns must be the ids `fetch` accepts — otherwise a
    citation Sven prints cannot be looked up."""
    checked = 0
    for chunk_id, _rec, _score in retrieve.search("docs", "the evidence contract", k=5):
        assert retrieve.fetch("docs", chunk_id)["id"] == chunk_id
        checked += 1
    assert checked >= 3


@requires_data
@requires_docs
def test_the_index_and_its_embeddings_agree():
    """A half-rebuilt DB is silently answerable and silently wrong — it returns
    one chunk's text under another chunk's id."""
    records, order, embeddings = retrieve.CORPORA["docs"][0]()
    assert len(order) == embeddings.shape[0]
    assert set(order) <= set(records)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), (
        "rows must be L2-normalized at build time — `search` relies on it")
