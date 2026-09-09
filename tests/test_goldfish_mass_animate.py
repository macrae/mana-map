"""Mass animation, both shapes.

`_MASS_ANIMATE_RE` shipped reading only the CONDITIONAL form — "as long as you
control five or more enchantments" — and its comment asserted that Starfield of
Nyx was the only card in the corpus that did this. Opalescence does the same
thing with no condition at all, and returned `mass_animate_threshold=0`, which
is the sentinel for "this card does not mass-animate". So on a list running 44
enchantments the strictly stronger of the two cards was worth nothing, and the
casting predicate that gates on the same field would never have cast it either.

These drive `goldfish.combat_profile` rather than re-deriving the rule.
"""

import re

import pytest

from manamap.pilot import card_pool, goldfish


def _profile(name):
    text = card_pool.corpus_oracle().get(name)
    assert text, f"{name} is not in the corpus"
    return goldfish.combat_profile(
        {"name": name, "oracle_text": text, "type_line": "Enchantment",
         "mana_cost": "{2}{W}{W}", "cmc": 4})


def test_the_conditional_form_carries_its_threshold():
    assert _profile("Starfield of Nyx")["mass_animate_threshold"] == 5


def test_the_unconditional_form_is_read_at_all():
    """The bug: this returned 0, the same value as a card with no such ability.

    Re-introduce it by deleting the `_MASS_ANIMATE_ALWAYS_RE` branch in
    `combat_profile` and this fails.
    """
    assert _profile("Opalescence")["mass_animate_threshold"] == 1


def test_unconditional_reads_as_always_on_and_never_as_absent():
    """1 rather than 0, because the consumer gates on truthiness first and then
    on `ench >= threshold`. A 0 would be silently skipped by both."""
    got = _profile("Opalescence")["mass_animate_threshold"]
    assert got, "0 is the absent sentinel — an unconditional effect must be truthy"
    assert got <= 1, "an unconditional effect must not impose a real threshold"


def test_a_flat_power_animator_is_deliberately_not_matched():
    """Bello makes a 4/4, not a body whose power is its mana value. Matching it
    here would price a different effect as this one."""
    assert _profile("Bello, Bard of the Brambles")["mass_animate_threshold"] == 0


def test_the_sweep_still_returns_exactly_these_two():
    """The comment above the pattern claims two cards in the corpus. If a set
    prints a third, this fails and the pattern gets re-read rather than
    silently under-matching."""
    broad = re.compile(
        r"enchantment[^.]{0,80}?\bis a creature|enchantments?[^.]{0,80}?are creatures",
        re.I)
    oracle = card_pool.corpus_oracle()
    hits = {n for n, t in oracle.items() if broad.search(t or "")}
    assert len(oracle) > 30000, "corpus did not load"
    assert hits == {"Opalescence", "Starfield of Nyx"}, sorted(hits)


@pytest.mark.parametrize("name,expected", [
    ("Starfield of Nyx", 5),
    ("Opalescence", 1),
    ("Mesa Enchantress", 0),
    ("Sol Ring", 0),
])
def test_the_field_is_set_only_by_the_two_cards_that_earn_it(name, expected):
    assert _profile(name)["mass_animate_threshold"] == expected


def test_two_animators_take_the_easiest_threshold_not_the_last_one():
    """Holding Opalescence AND Starfield of Nyx must not be worse than holding
    Opalescence alone.

    The consumer kept a single `mass_animate_threshold` and PLAIN-ASSIGNED it
    for every animator on the battlefield, so whichever resolved last won. With
    one such card in the corpus that was invisible. With two it means Starfield
    (threshold 5) silently cancels Opalescence's unconditional effect whenever
    it lands second, and the board stays down below five enchantments.

    Re-introduce it by restoring the plain assignment and this fails.
    """
    import inspect

    from manamap.pilot import goldfish

    src = inspect.getsource(goldfish)
    assert src.count("min(mass_animate_threshold, _mat)") == 2, (
        "both accumulation sites must take the minimum")
    assert "mass_animate_threshold = card[\"combat\"][\"mass_animate_threshold\"]" \
        not in src, "a plain assignment survives — last-seen would win again"
