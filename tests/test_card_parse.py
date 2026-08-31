"""A card as typed fields — the parser the tabular model is built on.

The architecture this replaces serialised every card into one sentence and
learned less than a random projection of MiniLM (function 0.618 trained against
0.602 random and 0.648 from PCA). The cause was structural: CMC never existed as
a number and colour identity never existed as a set, so the only thing to learn
from was a sentence encoder's opinion of a sentence.

MEASURED over 34,704 cards / 64,895 ability lines:

    triggered  16,259  25.1%      keyword  13,437  20.7%
    static     12,988  20.0%      spell    11,378  17.5%
    activated  10,833  16.7%      (99.6% of these yield a cost)
"""

import pytest

from manamap.training import card_parse as CP


def test_an_ability_word_does_not_hide_the_trigger():
    """THE SWEEP'S FIRST CORRECTION. `Landfall — Whenever a land you control
    enters…` read as STATIC because the classifier saw `Landfall` first.
    Stripping the `Word —` prefix moved 1,084 lines from static to triggered."""
    line = "Landfall — Whenever a land you control enters, you get {E}{E}."
    assert CP.classify_line(line, "Creature — Elemental") == "triggered"
    assert CP.classify_line("Metalcraft — Whenever this attacks, draw a card.",
                            "Creature — Golem") == "triggered"


def test_a_spell_effect_is_not_a_static_ability():
    """THE SWEEP'S SECOND CORRECTION, and the one that matters most. An instant
    or sorcery has no persistent abilities — its text IS the spell. Filing
    `Destroy target artifact` beside `Equipped creature gets +1/+1` tells the
    model those are the same kind of object.

    Decided by the TYPE LINE, never by the sentence: the identical words are a
    spell on a sorcery and a static ability on an enchantment.
    """
    words = "Destroy target artifact or enchantment."
    assert CP.classify_line(words, "Sorcery") == "spell"
    assert CP.classify_line(words, "Instant") == "spell"
    assert CP.classify_line("Equipped creature gets +1/+1.", "Artifact — Equipment") == "static"


def test_keyword_lines_are_their_own_class_even_with_a_cost():
    """`Ward {2}` and `Equip {3}` carry a cost and no colon to find it by. They
    are a vocabulary, not a sentence."""
    for line in ("Flying", "Flying, haste", "Ward {2}", "Equip {3}", "Morph {5}{G}"):
        assert CP.classify_line(line, "Creature — Human") == "keyword", line


def test_a_granted_ability_belongs_to_the_card_that_grants_it():
    """`Elves you control have "{T}: Add {G}{G}."` — the colon is inside quotes.
    The card's own ability is STATIC; the activated one belongs to the elves."""
    line = 'Elves you control have "{T}: Add {G}{G}."'
    assert CP.classify_line(line, "Enchantment") == "static"


def test_an_activated_ability_splits_into_cost_and_effect():
    """The cost is the half that says what the ability is WORTH — {T} against
    {3}{B}, Sacrifice a creature — and it is structured text deserving its own
    field, not the first few words of a sentence."""
    cost, effect = CP.split_activated("{1}, Sacrifice an artifact: Draw a card.")
    assert cost == "{1}, Sacrifice an artifact"
    assert effect == "Draw a card."
    assert CP.split_activated("Flying")[0] is None


def test_lines_split_on_the_ability_boundary():
    """The newline IS the ability boundary, and `extract.py:157` flattens it for
    `embedding_text` — right for a pooled vector, wrong here."""
    text = "Flying\n{T}: Add {G}.\nWhen this dies, draw a card."
    assert len(CP.ability_lines(text)) == 3
    assert CP.ability_lines("") == [] and CP.ability_lines(None) == []


def test_parse_returns_typed_abilities_with_counts():
    card = {"type_line": "Creature — Elf Druid",
            "oracle_text": "Flying\n{T}: Add {G}.\nWhen this dies, draw a card."}
    out = CP.parse(card)
    assert [a["kind"] for a in out["abilities"]] == ["keyword", "activated", "triggered"]
    assert out["abilities"][1]["cost"] == "{T}"
    assert out["counts"]["keyword"] == 1 and out["counts"]["spell"] == 0


def test_every_kind_is_reachable_on_the_real_corpus():
    """A class nothing lands in is a class that does not exist, and a sweep is
    the only way to know."""
    import collections
    import gzip
    import json

    from manamap.config import RAW_JSON_PATH

    if not RAW_JSON_PATH.exists():
        pytest.skip("raw Scryfall dump not downloaded")
    counts, lines = collections.Counter(), 0
    with gzip.open(RAW_JSON_PATH, "rt") as handle:
        for raw in handle:
            raw = raw.strip().rstrip(",")
            if not raw or raw in "[]":
                continue
            try:
                card = json.loads(raw)
            except ValueError:
                continue
            for line in CP.ability_lines(card.get("oracle_text")):
                counts[CP.classify_line(line, card.get("type_line", ""))] += 1
                lines += 1
    assert lines > 50_000, f"only {lines} ability lines swept"
    for kind in CP.ABILITY_KINDS:
        share = counts[kind] / lines
        assert 0.05 < share < 0.45, f"{kind} is {share:.1%} of lines — check the classifier"
