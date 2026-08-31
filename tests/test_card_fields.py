"""The field schema: three states, no leakage, and no crash on a malformed card.

This is the piece the whole tabular model rests on, so it is tested three ways:
by example, by property (random and adversarial cards), and by sweeping the real
corpus. The architecture it replaces failed because a number was never a number;
the way THIS fails is subtler — a masked field that still leaks its value makes
the imputation task trivial and shows up only as suspiciously good scores.
"""

import random

import numpy as np
import pytest

from manamap.training import card_fields as CF

VOCABS = {
    "supertype": ["Creature", "Instant", "Land"],
    "rarity": ["common", "rare"],
    "layout": ["normal", "transform"],
    "card_types": ["Creature", "Legendary", "Land", "Instant"],
    "subtypes": ["Elf", "Druid", "Human"],
    "keywords": ["Flying", "Trample"],
    "mana_symbols": ["G", "U", "1", "2"],
}
SCHEMA = CF.build_schema(VOCABS)


def _card(**kw):
    base = {"cmc": 1, "power": "1", "toughness": "1", "type_line": "Creature — Elf Druid",
            "mana_cost": "{G}", "supertype": "Creature", "rarity": "common",
            "layout": "normal", "color_identity": "G", "keywords": "Flying",
            "edhrec_rank": 500}
    base.update(kw)
    return base


# ── the three states ──


def test_absent_and_masked_never_encode_the_same():
    """THE DISTINCTION THE SCHEMA TURNS ON. A land has no power; a creature whose
    power was hidden does. A model that cannot tell them apart learns to predict
    absence, which is the cheapest way to be right."""
    land = _card(power=None, toughness=None, type_line="Land")
    creature = _card(power="3", toughness="3")
    field = next(f for f in SCHEMA if f.name == "power")
    absent = field.encode(land)
    masked = field.encode(creature, masked=True)
    assert not np.allclose(absent, masked), "absent and masked are indistinguishable"
    assert absent[-2] == 0.0 and absent[-1] == 0.0, "absent: not present, not masked"
    assert masked[-2] == 1.0 and masked[-1] == 1.0, "masked: present AND masked"


def test_masking_zeroes_the_value_so_nothing_leaks():
    """Leaving the true value behind a flag makes imputation a lookup, and it
    would surface only as suspiciously good numbers."""
    card = _card(power="7", subtypes=None)
    for name in ("power", "cmc", "subtypes", "color_identity", "supertype"):
        field = next(f for f in SCHEMA if f.name == name)
        plain, hidden = field.encode(card), field.encode(card, masked=True)
        assert np.allclose(hidden[:field.width], 0.0), f"{name} leaked its value"
        if field.read(card)[0] == CF.PRESENT:
            assert not np.allclose(plain[:field.width], 0.0), f"{name} encodes nothing"


def test_an_empty_set_is_absent_not_a_zero_vector():
    """'no subtypes' and 'subtypes hidden' must not encode identically."""
    field = next(f for f in SCHEMA if f.name == "subtypes")
    assert field.read(_card(type_line="Instant"))[0] == CF.ABSENT
    assert field.read(_card(type_line="Creature — Elf"))[0] == CF.PRESENT


# ── the edge cases the corpus survey turned up ──


def test_the_million_mana_card_cannot_set_the_scale():
    """cmc runs 0-8 at p99 and ONE card is 1,000,000 (Gleemax). Bounds are fixed
    constants, never computed from the corpus, or that one card defines the
    scale for all 34,890."""
    field = next(f for f in SCHEMA if f.name == "cmc")
    huge = field.encode(_card(cmc=1_000_000))
    normal = field.encode(_card(cmc=8))
    assert 0.0 <= huge[0] <= 1.0, "clipping failed"
    assert huge[0] == 1.0 and normal[0] < 1.0


def test_star_power_is_variable_not_zero():
    """253 power and 191 toughness values are non-numeric — `*`, `*²`, `1+*`.
    Casting throws the fact away; the flag keeps 'defined by something else'."""
    field = next(f for f in SCHEMA if f.name == "power")
    for text in ("*", "*²", "1+*", "2+*"):
        state, value = field.read(_card(power=text))
        assert state == CF.PRESENT and isinstance(value, str), text
        assert field.encode(_card(power=text))[1] == 1.0, f"{text} not flagged variable"
    assert field.encode(_card(power="3"))[1] == 0.0


def test_negative_toughness_survives():
    """Toughness genuinely reaches -1 in the corpus."""
    field = next(f for f in SCHEMA if f.name == "toughness")
    assert field.read(_card(toughness="-1")) == (CF.PRESENT, -1.0)
    assert 0.0 <= field.encode(_card(toughness="-1"))[0] <= 1.0


def test_nan_is_absent_not_the_string_nan():
    """`str(float('nan'))` is `"nan"`, and pandas hands every missing cell
    through as a float NaN. This shipped once in the serialiser.

    THE FIRST VERSION OF THIS TEST WAS PARTLY VACUOUS and a bug probe found it.
    It asserted `state in (PRESENT, ABSENT)` — always true — and then checked
    `power`, which `Numeric.read` guards with `_missing()` BEFORE `_clean` is
    ever reached. The `"nan"` string can only leak through CATEGORICAL and SET
    fields, which call `_clean` directly, so the test exercised every path
    except the one it was written for.
    """
    # the numeric path, guarded by `_missing`
    assert next(f for f in SCHEMA if f.name == "power").read(
        _card(power=float("nan")))[0] == CF.ABSENT

    # THE PATH THAT ACTUALLY LEAKS: categorical and set fields
    for name in ("supertype", "rarity", "layout"):
        field = next(f for f in SCHEMA if f.name == name)
        state, value = field.read(_card(**{name: float("nan")}))
        assert state == CF.ABSENT, f"{name} read NaN as {state}/{value!r}"
        assert value != "nan"
    for name, column in (("keywords", "keywords"), ("color_identity", "color_identity"),
                         ("subtypes", "type_line")):
        field = next(f for f in SCHEMA if f.name == name)
        state, value = field.read(_card(**{column: float("nan")}))
        assert state == CF.ABSENT, f"{name} read NaN as {state}/{value!r}"

    # and the string "nan" arriving as text, which pandas also produces
    field = next(f for f in SCHEMA if f.name == "supertype")
    assert field.read(_card(supertype="nan"))[0] == CF.ABSENT


def test_out_of_vocabulary_lands_in_OTHER_rather_than_vanishing():
    """539 subtypes exist and 256 are modelled. 'has a subtype I do not model'
    must stay distinguishable from 'has no subtypes'."""
    field = next(f for f in SCHEMA if f.name == "subtypes")
    known = field.encode(_card(type_line="Creature — Elf"))
    exotic = field.encode(_card(type_line="Creature — Vedalken"))
    assert exotic[field.width - 1] == 1.0, "OTHER column not set"
    assert not np.allclose(known, exotic)
    assert field.read(_card(type_line="Creature — Vedalken"))[0] == CF.PRESENT


# ── properties, over random and adversarial cards ──


ADVERSARIAL = [None, float("nan"), "", "  ", "nan", "NaN", 0, -1, 1e9, "*", "1+*",
               "X", "1d4+1", "{G}{U}", "—", "//", "Creature — ", " — Elf",
               "Creature — Elf // Land — Forest", "\n", "\t", "é", "🙂", '"quoted"',
               [], {}, True, False, "999999999999999999999"]


def _fuzz(rng):
    keys = ["cmc", "power", "toughness", "loyalty", "edhrec_rank", "type_line",
            "mana_cost", "supertype", "rarity", "layout", "color_identity", "keywords"]
    return {k: rng.choice(ADVERSARIAL) for k in keys}


def test_encoding_never_raises_on_a_malformed_card():
    """PROPERTY. Every field reads from a CSV row that may hold anything, and a
    crash deep in a DataLoader worker is the worst place to discover it."""
    rng = random.Random(0)
    for _ in range(2000):
        vector, offsets = CF.encode(_fuzz(rng), SCHEMA)
        assert vector.dtype == np.float32
        assert np.isfinite(vector).all(), "a non-finite value reached the encoding"
        assert len(offsets) == len(SCHEMA)


def test_the_width_is_constant_whatever_the_card():
    """PROPERTY. A ragged row would corrupt every batch after it."""
    rng = random.Random(1)
    expected = sum(f.total_width for f in SCHEMA)
    widths = {len(CF.encode(_fuzz(rng), SCHEMA)[0]) for _ in range(500)}
    widths.add(len(CF.encode(_card(), SCHEMA)[0]))
    widths.add(len(CF.encode({}, SCHEMA)[0]))
    assert widths == {expected}, f"ragged widths: {sorted(widths)}"


def test_encoding_is_deterministic():
    """PROPERTY. Every artifact in this repo rests on a seeded rerun producing
    the same bytes."""
    rng = random.Random(2)
    for _ in range(200):
        card = _fuzz(rng)
        assert np.array_equal(CF.encode(card, SCHEMA)[0], CF.encode(card, SCHEMA)[0])


def test_values_stay_in_range_however_absurd_the_input():
    """PROPERTY. An unbounded feature silently dominates every downstream norm."""
    rng = random.Random(3)
    for _ in range(1000):
        vector, _o = CF.encode(_fuzz(rng), SCHEMA)
        assert vector.min() >= -1e-6 and vector.max() <= 1.0 + 1e-6, \
            f"out of range: [{vector.min()}, {vector.max()}]"


def test_masking_any_subset_of_fields_is_consistent():
    """PROPERTY. Masking N fields must zero exactly those N and nothing else."""
    rng = random.Random(4)
    names = [f.name for f in SCHEMA]
    for _ in range(300):
        card = _card()
        chosen = rng.sample(names, rng.randint(1, 4))
        vector, offsets = CF.encode(card, SCHEMA, chosen)
        plain, _o = CF.encode(card, SCHEMA)
        for field in SCHEMA:
            start, end = offsets[field.name]
            if field.name in chosen:
                assert np.allclose(vector[start:end - 2], 0.0), field.name
                assert vector[end - 1] == 1.0, f"{field.name} not flagged masked"
            else:
                assert np.array_equal(vector[start:end], plain[start:end]), field.name


def test_masking_an_unknown_field_is_an_error_not_a_no_op():
    with pytest.raises(ValueError, match="not fields"):
        CF.encode(_card(), SCHEMA, ("nonexistent",))
    with pytest.raises(ValueError):
        CF.encode(_card(), SCHEMA, ("name",))          # deliberately not a field


def test_the_name_is_not_a_field():
    """MEASURED: recall@10 went 0.187 -> 0.248 when the name left the embedding
    text, and 34,814 distinct names over 34,890 cards make it a near-unique key —
    with it visible, imputing anything else is memorisation."""
    assert "name" not in {f.name for f in SCHEMA}
    assert "flavor_text" not in {f.name for f in SCHEMA}


# ── the real corpus ──


def test_every_field_is_populated_somewhere_and_absent_somewhere():
    """A field that is always present carries no `absent` signal to learn from;
    one that is never present is a column of zeros wasting capacity."""
    import pandas as pd

    from manamap.config import OUTPUT_CSV_PATH

    if not OUTPUT_CSV_PATH.exists():
        pytest.skip("corpus not built")
    cards = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False).to_dict("records")
    schema = CF.build_schema(CF.vocabularies(cards))
    seen = {f.name: {CF.PRESENT: 0, CF.ABSENT: 0} for f in schema}
    for card in cards[:6000]:
        for field in schema:
            seen[field.name][field.read(card)[0]] += 1
    for name, counts in seen.items():
        assert counts[CF.PRESENT] > 0, f"{name} is never present"
        if name not in ("cmc", "supertype", "rarity", "layout", "card_types"):
            assert counts[CF.ABSENT] > 0, f"{name} is never absent — no signal in it"


def test_the_real_corpus_encodes_without_a_single_non_finite_value():
    import pandas as pd

    from manamap.config import OUTPUT_CSV_PATH

    if not OUTPUT_CSV_PATH.exists():
        pytest.skip("corpus not built")
    cards = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False).to_dict("records")
    schema = CF.build_schema(CF.vocabularies(cards))
    checked = 0
    for card in cards[::7]:
        vector, _o = CF.encode(card, schema)
        assert np.isfinite(vector).all(), card.get("name")
        assert vector.min() >= -1e-6 and vector.max() <= 1.0 + 1e-6, card.get("name")
        checked += 1
    assert checked > 4000, f"only {checked} cards swept"
