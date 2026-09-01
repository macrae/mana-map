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
    for name, column in (("keywords_other", "keywords"),
                         ("color_identity", "color_identity"),
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
    """Presence and absence, asserted against a DECLARED set, in both directions.

    A field that is never present is a column of zeros wearing a name; a field
    that is never absent carries no absence signal to learn from. Both are worth
    knowing, but some fields are legitimately always present — every `kw_*`
    binary, because a card definitively has flying or does not.

    The first version carried a hand-written exemption list inside the test.
    That list only ever grows, and it grows silently: when the mana fields landed
    reading `generic_pips = 0.0` on a land, the honest fix was a guard, and an
    exemption list makes "add the name to the list" the cheaper move. So the set
    lives in `card_fields.ALWAYS_PRESENT` and is checked BOTH WAYS — a field that
    starts being always-present fails, and one that stops being always-present
    fails too.
    """
    import pandas as pd

    from manamap.config import OUTPUT_CSV_PATH

    if not OUTPUT_CSV_PATH.exists():
        pytest.skip("corpus not built")
    from manamap.training import card_source

    cards = card_source.enriched(
        pd.read_csv(OUTPUT_CSV_PATH, low_memory=False).to_dict("records"))
    schema = CF.build_schema(CF.vocabularies(cards))
    seen = {f.name: {CF.PRESENT: 0, CF.ABSENT: 0} for f in schema}
    for card in cards:
        for field in schema:
            seen[field.name][field.read(card)[0]] += 1

    assert len(seen) > 40, "schema shrank unexpectedly"
    for name, counts in seen.items():
        assert counts[CF.PRESENT] > 0, f"{name} is never present"
    measured = {n for n, c in seen.items() if c[CF.ABSENT] == 0}
    assert measured == set(CF.ALWAYS_PRESENT), (
        f"never absent but undeclared: {sorted(measured - set(CF.ALWAYS_PRESENT))}; "
        f"declared but absent somewhere: {sorted(set(CF.ALWAYS_PRESENT) - measured)}")


def test_a_card_with_no_mana_cost_is_absent_not_zero():
    """Command Tower does not cost zero generic mana — it has no cost at all.

    THE BUG THIS WAS WRITTEN FOR SHIPPED AND WAS CAUGHT BY THE TEST ABOVE. Every
    mana field read `PRESENT, 0.0` on all 1,760 costless cards, which tells the
    model a land's cost was MEASURED at zero. Ornithopter's `{0}` is a real
    measurement of zero, and the two must not encode identically.
    """
    schema = CF.build_schema(CF.vocabularies([_card()]))
    mana = [f for f in schema
            if f.name.startswith(("pips_", "generic_", "is_x", "has_hy", "has_ph"))]
    assert len(mana) == 9

    costless = _card(mana_cost=float("nan"), type_line="Land")
    for field in mana:
        assert field.read(costless)[0] == CF.ABSENT, f"{field.name} on a costless card"

    free = _card(mana_cost="{0}")
    generic = next(f for f in mana if f.name == "generic_pips")
    assert generic.read(free) == (CF.PRESENT, 0.0)

    # And the two must not produce the same columns.
    costless_row, offsets = CF.encode(costless, schema)
    free_row, _ = CF.encode(free, schema)
    lo, hi = offsets["generic_pips"]
    assert not np.array_equal(costless_row[lo:hi], free_row[lo:hi])
    assert list(costless_row[lo:hi]) == [0.0, 0.0, 0.0, 0.0]   # value, is_var, ABSENT, unmasked
    assert list(free_row[lo:hi]) == [0.0, 0.0, 1.0, 0.0]       # a measured zero, PRESENT

def test_the_real_corpus_encodes_without_a_single_non_finite_value():
    import pandas as pd

    from manamap.config import OUTPUT_CSV_PATH

    if not OUTPUT_CSV_PATH.exists():
        pytest.skip("corpus not built")
    from manamap.training import card_source

    cards = card_source.enriched(
        pd.read_csv(OUTPUT_CSV_PATH, low_memory=False).to_dict("records"))
    schema = CF.build_schema(CF.vocabularies(cards))
    checked = 0
    for card in cards[::7]:
        vector, _o = CF.encode(card, schema)
        assert np.isfinite(vector).all(), card.get("name")
        assert vector.min() >= -1e-6 and vector.max() <= 1.0 + 1e-6, card.get("name")
        checked += 1
    assert checked > 4000, f"only {checked} cards swept"


def test_pips_are_counted_per_colour_including_hybrid():
    """Each colour's pips are their own field, so each can be masked alone.

    Hybrid counts toward BOTH halves here, which is deliberately NOT what
    `manabase.count_pips` does — that one splits a hybrid half-and-half because
    it is sizing a mana base, where half a source is the honest answer. The
    question here is "can this card want blue", and a `{U/R}` card genuinely can.
    """
    schema = CF.build_schema(CF.vocabularies([_card()]))
    pip = {c: next(f for f in schema if f.name == f"pips_{c}") for c in "WUBRG"}
    generic = next(f for f in schema if f.name == "generic_pips")

    cases = [
        ("{5}{R}{G}{W}", {"W": 1, "R": 1, "G": 1}, 5),      # Gishath
        ("{U}{U}{U}", {"U": 3}, 0),                          # triple pip
        ("{2}{U/R}", {"U": 1, "R": 1}, 2),                   # hybrid: both halves
        ("{U/P}", {"U": 1}, 0),                              # Phyrexian
        ("{X}{R}", {"R": 1}, 0),                             # X is not generic
        ("{0}", {}, 0),                                      # a real zero
    ]
    for cost, expected, want_generic in cases:
        card = _card(mana_cost=cost)
        for colour in "WUBRG":
            state, value = pip[colour].read(card)
            assert state == CF.PRESENT
            assert value == expected.get(colour, 0), f"{cost} pips_{colour}={value}"
        assert generic.read(card)[1] == want_generic, cost


def test_x_hybrid_and_phyrexian_are_separate_questions():
    schema = CF.build_schema(CF.vocabularies([_card()]))
    flag = {n: next(f for f in schema if f.name == n)
            for n in ("is_x_spell", "has_hybrid", "has_phyrexian")}
    for cost, x, hybrid, phyrexian in [
        ("{X}{R}", True, False, False),
        ("{2}{U/R}", False, True, False),
        ("{U/P}", False, True, True),        # Phyrexian is a slash, so also hybrid
        ("{1}{G}", False, False, False),
        ("{X}{X}{U/B}", True, True, False),
    ]:
        card = _card(mana_cost=cost)
        assert flag["is_x_spell"].read(card)[1] is x, cost
        assert flag["has_hybrid"].read(card)[1] is hybrid, cost
        assert flag["has_phyrexian"].read(card)[1] is phyrexian, cost


def test_each_keyword_is_its_own_maskable_field():
    """THE POINT OF THE SPLIT: masking flying must not hide trample.

    While keywords were one 131-wide set, the only question the model could be
    asked was "what abilities does this card have". Masking hid flying AND
    trample AND lifelink together, so "does this creature fly" — the question a
    player actually asks — was not expressible.
    """
    schema = CF.build_schema(CF.vocabularies([_card()]))
    card = _card(keywords=["Flying", "Trample", "Lifelink"], type_line="Creature")

    flying = next(f for f in schema if f.name == "kw_flying")
    trample = next(f for f in schema if f.name == "kw_trample")
    deathtouch = next(f for f in schema if f.name == "kw_deathtouch")
    assert flying.read(card)[1] is True
    assert trample.read(card)[1] is True
    assert deathtouch.read(card)[1] is False

    plain, offsets = CF.encode(card, schema)
    hidden, _ = CF.encode(card, schema, masked=("kw_flying",))

    # Exactly one field's columns moved, and it is the one that was masked.
    # Walk `offsets` rather than re-deriving the layout: a test that recomputes
    # the thing it is checking is testing itself.
    moved = [name for name, (lo, hi) in offsets.items()
             if not np.array_equal(plain[lo:hi], hidden[lo:hi])]
    assert moved == ["kw_flying"], moved

    # And the mask actually HID it — the value slot is zero, the flag is set.
    lo, hi = offsets["kw_flying"]
    assert list(plain[lo:hi]) == [1.0, 1.0, 0.0]      # flying, present, unmasked
    assert list(hidden[lo:hi]) == [0.0, 1.0, 1.0]     # zeroed, present, MASKED


def test_masking_a_keyword_does_not_leak_through_another_field():
    """Hiding `kw_flying` must not leave the answer sitting in `keywords_other`.

    The tail set is a real leak risk: if the split were done by filtering the
    vocabulary rather than the values, "Flying" would still be in the set field
    and masking the flag would hide nothing at all.
    """
    schema = CF.build_schema(CF.vocabularies([
        _card(keywords=["Flying", "Amplify"]), _card(keywords=["Trample"])]))
    tail = next(f for f in schema if f.name == "keywords_other")
    values = tail.read(_card(keywords=["Flying", "Amplify"]))[1]
    assert "Flying" not in values
    assert "Amplify" in values


def test_what_a_card_taps_for_is_its_own_set_of_fields():
    """`cards.csv` cannot answer this at all, and it is the difference between
    Command Tower encoding as a blank and encoding as a five-colour source.

    Its row is `Land` with no mana cost, no power, no subtypes — before these
    fields the second-most-played card in Commander carried almost nothing while
    a vanilla French creature carried plenty.
    """
    schema = CF.build_schema(CF.vocabularies([_card()]))
    fields = {f.name: f for f in schema if f.name.startswith("produces_")}
    assert set(fields) == {f"produces_{s}" for s in "WUBRGC"}

    tower = _card(produced_mana=["B", "G", "R", "U", "W"])
    for sym in "WUBRG":
        assert fields[f"produces_{sym}"].read(tower) == (CF.PRESENT, True), sym
    assert fields["produces_C"].read(tower) == (CF.PRESENT, False)

    bolt = _card(produced_mana=[])
    assert all(f.read(bolt) == (CF.PRESENT, False) for f in fields.values())

    ring = _card(produced_mana=["C"])
    assert fields["produces_C"].read(ring)[1] is True
    assert fields["produces_G"].read(ring)[1] is False


def test_an_unenriched_record_reports_absent_not_false():
    """The three-state contract doing real work.

    A record that never saw `card_source.enrich` has no `produced_mana` key. If
    that read as False, every card in the corpus would report making no mana —
    silently, plausibly, and wrongly. It reads ABSENT instead: nobody measured it.
    """
    schema = CF.build_schema(CF.vocabularies([_card()]))
    produces = [f for f in schema if f.name.startswith("produces_")]

    bare = _card()
    assert "produced_mana" not in bare
    assert all(f.read(bare) == (CF.ABSENT, None) for f in produces)

    # And the two encode differently — absent is not a quiet False.
    from manamap.training import card_source
    enriched = card_source.enrich(bare, {})
    enriched["produced_mana"] = []
    bare_row, offsets = CF.encode(bare, schema)
    rich_row, _ = CF.encode(enriched, schema)
    lo, hi = offsets["produces_G"]
    assert list(bare_row[lo:hi]) == [0.0, 0.0, 0.0]     # value, ABSENT, unmasked
    assert list(rich_row[lo:hi]) == [0.0, 1.0, 0.0]     # a measured False, PRESENT


def test_each_produced_colour_masks_independently():
    schema = CF.build_schema(CF.vocabularies([_card()]))
    tower = _card(produced_mana=["B", "G", "R", "U", "W"])
    plain, offsets = CF.encode(tower, schema)
    hidden, _ = CF.encode(tower, schema, masked=("produces_U",))
    moved = [n for n, (lo, hi) in offsets.items()
             if not np.array_equal(plain[lo:hi], hidden[lo:hi])]
    assert moved == ["produces_U"], moved
