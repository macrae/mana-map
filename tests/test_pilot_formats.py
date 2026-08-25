"""What a legal deck IS, as a parameter.

PRD §13 asks for the format constraints to be parameters rather than
assumptions, "cheap if designed in now, expensive later". These tests pin the
two things that make that true: that there is exactly ONE place saying how big a
Commander deck is, and that a second format could be added without editing the
callers.
"""

import ast
import pathlib

import pytest

from conftest import requires_data
from manamap.pilot import check_in, formats, manabase, validate_deck

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "manamap"


def test_commander_is_what_commander_is():
    c = formats.COMMANDER
    assert (c.deck_size, c.singleton, c.commanders) == (100, True, 1)
    assert c.colour_identity and c.basics_exempt
    assert c.legality_key == "commander"


def test_the_library_size_is_derived_not_stored():
    """`99` was a second literal for `100 - 1`.

    A format whose commander starts in the library rather than the command zone
    would have needed both numbers changed and would have got one.
    """
    assert formats.COMMANDER.library_size == 99
    assert "library_size" in dir(formats.FormatSpec)
    # Prove it is arithmetic rather than a constant that happens to agree.
    other = formats.FormatSpec(name="x", deck_size=60, exact_size=False,
                               singleton=False, commanders=0,
                               colour_identity=False, basics_exempt=True,
                               legality_key="modern")
    assert other.library_size == 60 and other.max_copies == 4


def test_every_consumer_reads_the_same_number():
    """The whole point. Four places used to say 100 (or 99) independently, and
    one of them — `check_in` — shadowed `config` with its own constant."""
    assert check_in.DECK_SIZE == formats.DEFAULT.deck_size
    assert manabase.DECK_SIZE_AFTER_COMMANDER == formats.DEFAULT.library_size


def test_config_no_longer_owns_deck_size():
    """Legality lives in `formats`; SHAPE (role budget, curve targets) stays in
    `config`. The layering settles it — everything imports `config`, including
    `pilot`, so `config` cannot import `formats`."""
    import manamap.config as config

    assert not hasattr(config, "DECK_SIZE"), (
        "config.DECK_SIZE is back — legality and tuning have re-merged")


def test_no_bare_deck_size_literal_survives_in_the_rule_checks():
    """The bare `if total != 100` in `validate_deck` was the copy no shared
    constant could reach. Scanned rather than asserted by behaviour, because a
    literal that agrees with the spec passes every behavioural test."""
    # Scanned through the AST, not with a regex over the text. A first cut
    # stripped `#` lines and then flagged the word "100" inside a DOCSTRING —
    # prose about the format, which is exactly what those modules should say.
    # The AST sees only literals that are actually evaluated, which is the
    # thing being banned.
    for name in ("validate_deck.py", "check_in.py", "validate_build.py"):
        tree = ast.parse((SRC / "pilot" / name).read_text(encoding="utf-8"))
        hits = [n for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and n.value in (99, 100)]
        assert not hits, (
            f"{name} evaluates a bare deck-size literal at line(s) "
            f"{[n.lineno for n in hits]}")


def test_an_unknown_format_is_an_error_not_a_fallback():
    """Silently building Commander because a name was misspelled is the class of
    bug this module exists to prevent."""
    assert formats.get(None) is formats.DEFAULT
    assert formats.get("Commander") is formats.COMMANDER
    with pytest.raises(SystemExit):
        formats.get("pendragon")


def test_pendragon_is_deliberately_absent():
    """The PRD lists it and flags its own description as unverified. A spec that
    encodes a guess is worse than one that omits it, because the guess becomes
    invisible the moment it is in a table."""
    assert "pendragon" not in formats.FORMATS


def test_validate_deck_enforces_the_spec_it_is_given():
    """A second format must work by passing a parameter, not by editing rules."""
    doc = {"cards": [
        {"name": "Sol Ring", "quantity": 4, "type_line": "Artifact", "color_identity": []},
    ]}
    loose = formats.FormatSpec(name="Loose", deck_size=1, exact_size=True,
                               singleton=False, commanders=0,
                               colour_identity=False, basics_exempt=True,
                               legality_key="modern")
    # 4 copies is fine where singleton does not apply, and the size check reads
    # the spec's number rather than 100.
    errors = validate_deck.validate(doc, loose)
    assert not any("Singleton" in e for e in errors)
    assert not any("expected exactly 100" in e for e in errors)

    # …and the same document fails Commander on both counts.
    errors = validate_deck.validate(doc, formats.COMMANDER)
    assert any("Singleton" in e for e in errors)
    assert any("expected exactly 100" in e for e in errors)


def test_basics_are_exempt_from_singleton_only_when_the_spec_says_so():
    doc = {"cards": [{"name": "Forest", "quantity": 30,
                      "type_line": "Basic Land — Forest", "color_identity": ["G"]}]}
    assert not any("Singleton" in e
                   for e in validate_deck.validate(doc, formats.COMMANDER))
    strict = formats.FormatSpec(name="Strict", deck_size=30, exact_size=True,
                                singleton=True, commanders=0,
                                colour_identity=False, basics_exempt=False,
                                legality_key="commander")
    assert any("Singleton" in e for e in validate_deck.validate(doc, strict))


# ── The 60-card formats ────────────────────────────────────────────────────


@pytest.mark.parametrize("key", ["standard", "modern", "pioneer", "pauper"])
def test_constructed_is_sixty_or_more_not_exactly_sixty(key):
    """"Your deck must contain at least sixty cards." A 63-card Modern deck is
    legal, and enforcing an exact 60 would reject legal decks while looking
    rigorous. Commander is the opposite: exactly 100."""
    spec = formats.FORMATS[key]
    assert spec.exact_size is False
    assert spec.size_error(60) is None
    assert spec.size_error(63) is None
    assert spec.size_error(59) and "at least 60" in spec.size_error(59)
    assert formats.COMMANDER.size_error(101), "Commander must stay exact"


@pytest.mark.parametrize("key", ["standard", "modern", "pioneer", "pauper"])
def test_constructed_has_no_commander_and_no_identity(key):
    spec = formats.FORMATS[key]
    assert spec.commanders == 0
    assert spec.colour_identity is False
    assert spec.max_copies == 4 and spec.singleton is False
    assert spec.library_size == 60, "no commander leaves the whole deck in the library"


@requires_data
def test_pauper_is_not_commons_only(caplog):
    """MEASURED, and it contradicts the PRD.

    §13 describes Pauper as "commons only". Scryfall's `legal_pauper` disagrees
    for 373 cards, because a card printed at common ANYWHERE is pauper-legal
    even where this printing is not. Consulting the column is both simpler and
    correct; a rarity filter would look stricter and be wrong 373 times.
    """
    import pandas as pd

    from manamap.config import OUTPUT_CSV_PATH
    from manamap.pilot import card_pool

    frame = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False,
                        usecols=["name", "rarity", "legal_pauper"])
    odd = frame[(frame.legal_pauper == "legal") & (frame.rarity != "common")]
    assert len(odd) > 100, (
        "no pauper-legal non-commons found — if this is really zero, the "
        "'commons only' shortcut would be safe and this note should change")
    status = card_pool.legality("legal_pauper")
    assert status.get(odd.name.iloc[0]) == "legal"


@requires_data
def test_legality_is_checked_at_all_and_names_the_reason(bare_deck_cards=None):
    """NOTHING checked this before.

    Every deck here is Commander and every card in them is Commander-legal, so
    the gap was invisible — and it stops being invisible the moment a 60-card
    format arrives, where a Modern deck holding a rotated card is the commonest
    mistake there is. `banned` and `not_legal` are reported distinctly, because
    they are different problems with different fixes.
    """
    from manamap.pilot import validate_deck

    cards = [
        {"name": "Dig Through Time", "quantity": 4, "type_line": "Instant",
         "color_identity": ["U"]},
        {"name": "Lightning Bolt", "quantity": 4, "type_line": "Instant",
         "color_identity": ["R"]},
    ]
    errors = validate_deck.illegal_cards(cards, formats.MODERN)
    joined = " ".join(errors)
    assert "Dig Through Time" in joined and "banned" in joined
    assert "Lightning Bolt" not in joined, "a Modern staple was reported illegal"


@requires_data
def test_a_promo_printing_cannot_make_a_staple_illegal():
    """FIRST PRINTING DOES NOT WIN for legality, and this is why.

    `cards.csv` carries two rows for Savage Lands — a store-championship promo
    (`fmsc`) marked `not_legal` and the real one (`msc`) marked `legal` — and
    the promo sorts first. First-wins is right for IDENTITY (any printing of Sol
    Ring is Sol Ring) and wrong here, and it failed ur-dragon and radagast on
    their own tracked decklists the moment the check was added.
    """
    from manamap.pilot import card_pool

    status = card_pool.legality("legal_commander")
    assert status.get("Savage Lands") == "legal"


@requires_data
def test_the_combining_rule_is_measured_not_guessed():
    """Across the corpus, 16 names disagree between printings and EVERY
    disagreement is exactly {legal, not_legal} — `banned` never co-occurs. So
    "any legal printing makes it legal" is unambiguous.

    If a card ever appears both banned and legal, this test fails and the
    combining rule needs re-arguing rather than re-fitting — `banned` should
    win, which is what the code already does on an unreachable branch.
    """
    import collections

    import pandas as pd

    from manamap.config import OUTPUT_CSV_PATH

    frame = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False,
                        usecols=["name", "legal_commander"])
    by = collections.defaultdict(set)
    for name, value in zip(frame["name"], frame["legal_commander"]):
        by[name].add(value)
    disagreeing = [v for v in by.values() if len(v) > 1]
    assert disagreeing, "no name disagrees — the promo hazard may have gone away"
    assert all(v == {"legal", "not_legal"} for v in disagreeing), (
        "a name is both banned and legal somewhere; the combining rule now has "
        "a real decision to make and 'any legal printing wins' is not it")


def test_only_commander_is_buildable_and_the_spec_says_so():
    """"I tried to build a Standard deck and nothing happened."

    The UI offered a five-format picker in front of a builder that builds one.
    The gap is real rather than a missing flag: `build_deck` is anchored on a
    commander at every step — colour identity gates the candidate pool, the
    similarity score is seeded from its name, its tags drive synergy, the
    bracket engine reads it, and `manabase` sizes against a 99-card library. A
    constructed deck has no such anchor.

    So a format the tool cannot build is something the tool SAYS it cannot
    build. Flip this the day the builder learns a second strategy.
    """
    assert formats.COMMANDER.buildable is True
    for key in ("standard", "modern", "pioneer", "pauper"):
        assert formats.FORMATS[key].buildable is False, key


def test_buildable_is_not_the_same_question_as_legal():
    """A format the builder cannot build is still fully validated and searched —
    those are different capabilities and the vocabulary must not merge them."""
    modern = formats.FORMATS["modern"]
    assert modern.buildable is False
    assert modern.legality_column == "legal_modern"
    assert modern.deck_size == 60 and modern.max_copies == 4
