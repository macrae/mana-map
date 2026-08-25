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
    other = formats.FormatSpec(name="x", deck_size=60, singleton=False, commanders=0,
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
    loose = formats.FormatSpec(name="Loose", deck_size=1, singleton=False, commanders=0,
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
    strict = formats.FormatSpec(name="Strict", deck_size=30, singleton=True, commanders=0,
                                colour_identity=False, basics_exempt=False,
                                legality_key="commander")
    assert any("Singleton" in e for e in validate_deck.validate(doc, strict))
