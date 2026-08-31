"""The span encoder: extraction, slotting, masking, and the cache's guards.

Structured the same way as `test_card_fields.py` — by example, by PROPERTY over
adversarial input, and by corpus sweep — because this is the other half of the
model's input layer and the same class of bug lives in it.
"""

import gzip
import json

import numpy as np
import pytest

from manamap.config import OUTPUT_CSV_PATH, RAW_JSON_PATH
from manamap.training import span_encoder as SE


def _card(**over):
    card = {
        "oracle_id": "abc", "name": "Test Card", "type_line": "Creature — Human",
        "oracle_text": "Flying\nWhen this creature enters, draw a card.",
        "flavor_text": "A flavourful line.",
    }
    card.update(over)
    return card


# ── extraction ──────────────────────────────────────────────────────────


def test_the_newline_is_the_ability_boundary():
    """The whole reason this module does not read `cards.csv`.

    `extract.py:157` flattens newlines so the CSV stays one row per card. Read
    from there, EVERY card has exactly one ability line and the ability structure
    does not exist; the raw dump has 66,431 lines against the CSV's 34,890.
    """
    spans = SE.card_spans(_card())
    assert spans["keyword"] == ["Flying"]
    assert spans["triggered"] == ["When this creature enters, draw a card."]

    flattened = _card(oracle_text="Flying When this creature enters, draw a card.")
    assert list(SE.card_spans(flattened)) == ["name", "flavor", "static"]


def test_a_card_with_no_text_has_no_ability_slots_and_no_nan():
    """`str(x or "")` does not handle a pandas NaN — a float nan is truthy, so it
    survives the `or` and stringifies to the literal `"nan"`.

    This shipped: 409 cards had the word "nan" as their sole ability line, handed
    to the encoder as if it read like a card. Third instance in the repo, after
    the serialiser and `keywords_of`.
    """
    for empty in (float("nan"), None, "", "   "):
        spans = SE.card_spans(_card(oracle_text=empty, flavor_text=empty))
        assert set(spans) == {"name"}, f"{empty!r} produced {spans}"
        assert "nan" not in json.dumps(spans)


def test_name_and_flavor_are_their_own_slots():
    spans = SE.card_spans(_card())
    assert spans["name"] == ["Test Card"]
    assert spans["flavor"] == ["A flavourful line."]
    assert SE.card_spans(_card(flavor_text=float("nan"))).get("flavor") is None


def test_lines_of_one_kind_pool_into_one_slot():
    """A card can carry 17 static abilities, so a positional slot per line is not
    an option. They pool by CLASS, which also stops the model learning from an
    ordering that carries no information."""
    card = _card(oracle_text="When this enters, draw a card.\n"
                             "Whenever this attacks, gain 1 life.\n"
                             "At the beginning of your upkeep, scry 1.")
    assert len(SE.card_spans(card)["triggered"]) == 3


# ── properties, over deliberately hostile input ─────────────────────────


ADVERSARIAL = [
    float("nan"), None, "", "   ", "\n\n\n", "{}", "[]", "🜏🜏🜏", "nan", "NaN",
    "a" * 5000, "Flying\n" * 50, "—", "//", " // ", "\t", "1d4+1", "0",
    "Creature — ", "When , .", '"quoted"', "{T}: Add {G}.",
]


@pytest.mark.parametrize("value", ADVERSARIAL)
def test_extraction_never_raises_and_never_emits_nan(value):
    for field in ("oracle_text", "name", "flavor_text", "type_line"):
        spans = SE.card_spans(_card(**{field: value}))
        assert isinstance(spans, dict)
        for slot, lines in spans.items():
            assert slot in SE.SPAN_SLOTS, slot
            assert all(isinstance(x, str) and x.strip() for x in lines)
            assert "nan" not in [x.lower() for x in lines] or value in ("nan", "NaN")


def test_unique_spans_is_sorted_and_deduplicated():
    """The row a span lands on is part of the cache's contract. An unordered set
    would reshuffle it on every rebuild, making two caches of the same corpus
    silently incompatible — the failure that looks like a bad hyperparameter."""
    cards = [_card(), _card(name="Other"), _card()]
    spans = SE.unique_spans(cards)
    assert spans == sorted(spans)
    assert len(spans) == len(set(spans))
    assert "Flying" in spans and "Test Card" in spans and "Other" in spans


# ── slotting and masking, against a synthetic cache ──────────────────────


@pytest.fixture
def cache():
    """A cache over the fixture cards, with deterministic stand-in vectors."""
    cards = [_card()]
    spans = SE.unique_spans(cards)
    rng = np.random.default_rng(0)
    return SE.SpanCache(rng.normal(size=(len(spans), 8)).astype(np.float32), spans)


def test_every_slot_is_represented_present_or_not(cache):
    slots = cache.slot_vectors(_card())
    assert list(slots) == list(SE.SPAN_SLOTS)
    assert slots["keyword"][0] == SE.PRESENT
    assert slots["activated"][0] == SE.ABSENT
    assert not slots["activated"][1].any(), "an absent slot must be zeroed"


def test_masking_a_slot_zeroes_it_and_touches_nothing_else(cache):
    plain, offsets = cache.encode(_card())
    hidden, _ = cache.encode(_card(), masked="keyword")

    moved = [s for s, (lo, hi) in offsets.items()
             if not np.array_equal(plain[lo:hi], hidden[lo:hi])]
    assert moved == ["keyword"], moved

    lo, hi = offsets["keyword"]
    assert plain[lo:hi][:-2].any(), "the unmasked slot should carry a vector"
    assert not hidden[lo:hi][:-2].any(), "a MASKED slot must be zeroed, not flagged"
    assert list(plain[lo:hi][-2:]) == [1.0, 0.0]      # present, unmasked
    assert list(hidden[lo:hi][-2:]) == [1.0, 1.0]     # present, MASKED


def test_absent_and_masked_are_different_states(cache):
    """A slot the card does not have is not a slot that was hidden from the model.
    Both are zero vectors; only the flags tell them apart."""
    _, offsets = cache.encode(_card())
    absent, _ = cache.encode(_card())
    masked, _ = cache.encode(_card(), masked="keyword")
    lo, hi = offsets["activated"]
    assert list(absent[lo:hi][-2:]) == [0.0, 0.0]     # ABSENT
    lo, hi = offsets["keyword"]
    assert list(masked[lo:hi][-2:]) == [1.0, 1.0]     # MASKED


def test_masking_an_unknown_slot_is_an_error(cache):
    with pytest.raises(ValueError, match="not span slots"):
        cache.encode(_card(), masked="oracle_text")


def test_pooling_is_magnitude_independent(cache):
    """Lines are unit-normalised before the mean, so an 80-word rules line cannot
    outweigh `Flying` by sheer magnitude."""
    spans = ["short", "a much longer line of rules text that goes on and on"]
    matrix = np.array([[3.0, 0.0], [0.0, 0.1]], dtype=np.float32)
    small = SE.SpanCache(matrix, spans)
    # No name or flavor: this cache holds only the two static lines, and
    # `vector` raising on anything else is the behaviour we want — a span missing
    # from the cache means the cache is stale, not that the slot is empty.
    card = _card(oracle_text="short\na much longer line of rules text that goes on and on",
                 type_line="Creature", name=float("nan"), flavor_text=float("nan"))
    pooled = small.slot_vectors(card)["static"][1]
    assert pooled == pytest.approx([0.5, 0.5], abs=1e-6)


# ── the corpus, and the cache's guards ──────────────────────────────────


@pytest.mark.skipif(not RAW_JSON_PATH.exists(), reason="raw dump not downloaded")
def test_every_corpus_card_is_in_the_raw_dump():
    """The join that everything else rests on. A card missing from the dump would
    silently lose all of its ability slots rather than fail."""
    import pandas as pd

    if not OUTPUT_CSV_PATH.exists():
        pytest.skip("corpus not built")
    frame = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
    texts = SE.oracle_text_by_id()
    missing = [o for o in frame["oracle_id"] if o not in texts]
    assert missing == [], f"{len(missing)} cards absent from the dump"


@pytest.mark.skipif(not RAW_JSON_PATH.exists(), reason="raw dump not downloaded")
def test_the_dump_restores_ability_structure_the_csv_lost():
    import pandas as pd

    if not OUTPUT_CSV_PATH.exists():
        pytest.skip("corpus not built")
    frame = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
    cards = frame.to_dict("records")[:4000]
    texts = SE.oracle_text_by_id()

    from_dump = sum(len(v) for c in cards
                    for k, v in SE.card_spans(c, texts[c["oracle_id"]]).items()
                    if k not in ("name", "flavor"))
    from_csv = sum(len(v) for c in cards for k, v in SE.card_spans(c).items()
                   if k not in ("name", "flavor"))
    assert from_dump > from_csv * 1.5, (
        f"the dump gave {from_dump} ability lines, the flattened CSV {from_csv} — "
        "the newline boundary is not being restored")


def test_a_cache_from_a_different_encoder_is_refused(tmp_path, monkeypatch):
    """REFUSES, never warns. A head trained against vectors from a different
    encoder produces a plausible loss curve and a meaningless space."""
    vectors, index = tmp_path / "v.npy", tmp_path / "i.json.gz"
    np.save(vectors, np.zeros((2, 8), dtype=np.float32))
    with gzip.open(index, "wt", encoding="utf-8") as fh:
        json.dump({"encoder": "sentence-transformers/some-other-model",
                   "slots": list(SE.SPAN_SLOTS), "dim": 8,
                   "spans": ["a", "b"], "corpus_rows": 2}, fh)
    monkeypatch.setattr(SE, "VECTORS_PATH", vectors)
    monkeypatch.setattr(SE, "INDEX_PATH", index)
    cache, why = SE.load()
    assert cache is None and "some-other-model" in why


def test_a_cache_with_the_wrong_slots_is_refused(tmp_path, monkeypatch):
    vectors, index = tmp_path / "v.npy", tmp_path / "i.json.gz"
    np.save(vectors, np.zeros((2, 8), dtype=np.float32))
    with gzip.open(index, "wt", encoding="utf-8") as fh:
        json.dump({"encoder": f"sentence-transformers/{SE.TEXT_MODEL_NAME}",
                   "slots": ["name"], "dim": 8, "spans": ["a", "b"],
                   "corpus_rows": 2}, fh)
    monkeypatch.setattr(SE, "VECTORS_PATH", vectors)
    monkeypatch.setattr(SE, "INDEX_PATH", index)
    cache, why = SE.load()
    assert cache is None and "slots" in why


def test_a_truncated_cache_is_refused(tmp_path, monkeypatch):
    vectors, index = tmp_path / "v.npy", tmp_path / "i.json.gz"
    np.save(vectors, np.zeros((2, 8), dtype=np.float32))
    with gzip.open(index, "wt", encoding="utf-8") as fh:
        json.dump({"encoder": f"sentence-transformers/{SE.TEXT_MODEL_NAME}",
                   "slots": list(SE.SPAN_SLOTS), "dim": 8,
                   "spans": ["a", "b", "c"], "corpus_rows": 2}, fh)
    monkeypatch.setattr(SE, "VECTORS_PATH", vectors)
    monkeypatch.setattr(SE, "INDEX_PATH", index)
    cache, why = SE.load()
    assert cache is None and "disagrees" in why
