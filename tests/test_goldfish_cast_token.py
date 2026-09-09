"""Cast-triggered bodies — the enchantment archetype's payoff.

`cast_token_profile` existed and PARSED SIGIL OF THE EMPTY THRONE CORRECTLY the
whole time. Its output was thrown away twice over: the profile was computed only
for the commander, and its gate was matched against a cast card's creature
`subtypes` — Vampire, for Edgar Markov's eminence — so a gate naming a CARD TYPE
could never fire. On a list of 44 enchantments, "whenever you cast an enchantment
spell, create a 4/4 white Angel with flying" was worth nothing.

`_CONSTELLATION_TOKEN_RE` is the neighbouring channel and reads a different
trigger: "whenever an enchantment you control ENTERS". Both exist because they
are different events.
"""

import pytest

from manamap.pilot import card_pool, goldfish


def _prof(name):
    text = card_pool.corpus_oracle().get(name)
    assert text, f"{name} is not in the corpus"
    return goldfish.cast_token_profile({"name": name, "oracle_text": text})


def test_a_card_type_gate_is_read():
    """The bug. Re-introduce it by dropping `_CAST_TOKEN_TYPE_GATES` and this
    returns gate_kind 'subtype', which matches no type line."""
    got = _prof("Sigil of the Empty Throne")
    assert got["gate_kind"] == "type"
    assert got["subtype"] == "Enchantment"
    assert got["power"] == 4 and got["bodies"] == 1


def test_the_commanders_subtype_gate_still_works():
    """Edgar Markov's eminence is the original caller and must not regress."""
    got = _prof("Edgar Markov")
    assert got["gate_kind"] == "subtype"
    assert got["subtype"] == "Vampire"


def test_a_token_this_model_cannot_size_is_floored_and_flagged():
    """Hallowed Haunting's token is 'equal to the number of SPIRITS you control'
    — self-referential, since the only Spirits are the ones it already made.
    Nothing tracks a creature-type count, so it is priced at its floor of 1 and
    SAID SO rather than guessed."""
    got = _prof("Hallowed Haunting")
    assert got["gate_kind"] == "type" and got["subtype"] == "Enchantment"
    assert got["power"] == 1, "a token this model cannot size must take the floor"
    assert got["scales"] is True, "and the understatement must be flagged"


def test_a_card_with_no_such_trigger_returns_none():
    assert _prof("Archon of Sun's Grace") is None    # constellation, not cast


def test_the_type_gates_are_the_same_four_as_the_draw_channel():
    """One rule, two channels: a cast trigger fires where a non-land PERMANENT
    joins the battlefield, so only permanent-type gates are counted completely."""
    assert set(goldfish._CAST_TOKEN_TYPE_GATES) == set(goldfish._CAST_DRAW_GATES)


# ── it must actually be cast, and actually fire ───────────────────────────

def test_a_cast_token_engine_is_castable():
    """A CARD READ CORRECTLY AND NEVER PLAYED is the seventh instance of this
    class. Sigil has no body, makes no mana and draws nothing."""
    import inspect

    assert 'c["cast_token"]' in inspect.getsource(goldfish), (
        "cast_token is not in the casting predicate")


def test_it_fires_at_both_doors_and_registers_at_both():
    import inspect

    src = inspect.getsource(goldfish)
    assert src.count("for _eng in cast_token_engines:") == 2, "one door only"
    assert src.count('cast_token_engines.append(card["cast_token"])') == 2


def test_a_type_gate_is_matched_against_the_type_line_not_the_subtypes():
    """The exact confusion that hid the channel."""
    import inspect

    src = inspect.getsource(goldfish)
    assert '_eng["subtype"] in _tl2 if _eng["gate_kind"] == "type"' in src


def test_the_sweep_still_finds_the_two_enchantment_gated_cards():
    """If a set prints a third, this fails and the family gets re-read."""
    oracle = card_pool.corpus_oracle()
    assert len(oracle) > 30000, "corpus did not load"
    found = set()
    checked = 0
    for name, text in oracle.items():
        checked += 1
        p = goldfish.cast_token_profile({"name": name, "oracle_text": text})
        if p and p["gate_kind"] == "type" and p["subtype"] == "Enchantment":
            found.add(name)
    assert checked > 30000
    assert found == {"Sigil of the Empty Throne", "Hallowed Haunting"}, sorted(found)
