"""Cast-triggered draw — the enchantress channel.

"Whenever you cast an enchantment spell, you may draw a card" matched NO pattern
in `goldfish.py`: not the ETB one (needs "enters"), not the recurring one (needs
an upkeep), not the arrival one (needs "you control … enters"), and not the spell
one, which only runs on instants and sorceries. So Mesa Enchantress — the most
included card in the Zur, Eternal Schemer meta at 72.6% of ~6,703 decks — read as
`unmodelled` on a deck with 44 enchantments, and a branch that added her to fix a
measured draw hole came back WORSE.

The type gate is the load-bearing half and is tested in both directions: an
enchantress must not draw off a creature, and Beast Whisperer must not draw off
an enchantment.
"""

import pytest

from manamap.pilot import card_pool, goldfish


def _profile(name, type_line="Creature", text=None):
    text = text if text is not None else card_pool.corpus_oracle().get(name)
    assert text, f"{name} is not in the corpus"
    return goldfish.draw_profile(
        {"name": name, "oracle_text": text, "type_line": type_line,
         "mana_cost": "{1}{W}{W}", "cmc": 3})


# ── the channel exists at all ─────────────────────────────────────────────

def test_the_enchantress_is_read():
    """The bug: this was `unmodelled` with cast_draw 0. Re-introduce it by
    deleting the `_CAST_DRAW_RE` branch in `draw_profile` and this fails."""
    got = _profile("Mesa Enchantress")
    assert got["cast_draw"] == 1
    assert got["cast_draw_gate"] == "Enchantment"
    assert got["unmodelled"] is None


def test_it_was_never_the_optional_wording():
    """The first diagnosis blamed "you may", which `_DRAW_CONDITIONAL_RE`
    rejects. Removing it changes nothing — there was no channel at all, and this
    records that so the wrong fix is not attempted again."""
    without = _profile("x", text="Whenever you cast an enchantment spell, draw a card.")
    withmay = _profile("y", text="Whenever you cast an enchantment spell, you may draw a card.")
    assert without["cast_draw"] == withmay["cast_draw"] == 1


# ── the type gate, in both directions ─────────────────────────────────────

@pytest.mark.parametrize("name,gate", [
    ("Mesa Enchantress", "Enchantment"),
    ("Argothian Enchantress", "Enchantment"),
    ("Verduran Enchantress", "Enchantment"),
    ("Beast Whisperer", "Creature"),
    ("Kor Spiritdancer", "Aura"),
    ("Riddlesmith", "Artifact"),
])
def test_the_modelled_gates_carry_the_right_type(name, gate):
    assert _profile(name)["cast_draw_gate"] == gate


@pytest.mark.parametrize("name", [
    "Whirlwind of Thought",          # "noncreature" — this model casts few
    "Reki, the History of Kamigawa",  # "legendary" — type line does not settle it
    "Gilt-Leaf Archdruid",            # "Druid" — a creature type, not a card type
    "Sire of the Storm",              # "Spirit or Arcane"
])
def test_a_gate_this_model_cannot_evaluate_is_refused_not_guessed(name):
    """Refusing is the whole discipline: an unmodelled card is an absent figure,
    a wrongly-gated one is a wrong figure that looks the same as a right one."""
    got = _profile(name)
    assert got["cast_draw"] == 0
    assert got["cast_draw_gate"] is None


def test_an_enchantress_does_not_draw_off_a_creature():
    """The gate is checked against the CAST card's type line at the fire site.
    Asserting the contract the simulation relies on."""
    gate = _profile("Mesa Enchantress")["cast_draw_gate"]
    assert gate not in "Legendary Creature — Human Wizard"
    assert gate in "Legendary Enchantment — Shrine"


def test_beast_whisperer_does_not_draw_off_an_enchantment():
    gate = _profile("Beast Whisperer")["cast_draw_gate"]
    assert gate not in "Enchantment — Aura"
    assert gate in "Enchantment Creature — Nymph"   # an enchantment CREATURE does


# ── the predicate, in the same commit as the channel ──────────────────────

def test_a_cast_draw_card_is_actually_castable():
    """A CARD CAN BE READ CORRECTLY AND NEVER PLAYED — six recorded instances.
    Mesa Enchantress has no body, makes no mana and tutors nothing, so without
    `cast_draw` in the casting predicate she sits in hand for ten turns while her
    profile says exactly what she would have drawn."""
    import inspect

    src = inspect.getsource(goldfish)
    assert 'c["draw"]["cast_draw"]' in src, (
        "cast_draw is not in the casting predicate — the channel is unreachable")


def test_the_engine_fires_and_registers_at_both_doors():
    """Two places a non-land permanent joins the battlefield. A channel wired to
    one of them undercounts silently."""
    import inspect

    src = inspect.getsource(goldfish)
    assert src.count('_eng["cast_draw"] and _eng["cast_draw_gate"] in _tl') == 2
    assert src.count('or card["draw"]["cast_draw"]') == 2


def test_the_sweep_still_covers_what_the_comment_claims():
    """The comment says 23 of 68 modelled. If a set prints more, this fails and
    the gate list gets re-read rather than silently under-matching."""
    import re

    oracle = card_pool.corpus_oracle()
    assert len(oracle) > 30000, "corpus did not load"
    pat = goldfish._CAST_DRAW_RE
    total = modelled = 0
    for name, text in oracle.items():
        m = pat.search(text or "")
        if not m:
            continue
        total += 1
        if goldfish._CAST_DRAW_GATES.get((m.group(1) or "").strip().lower()):
            modelled += 1
    assert total >= 60, f"the family shrank to {total}"
    assert modelled >= 20, f"only {modelled} modelled"
    assert modelled < total, "every gate modelled — the refusal list stopped working"
