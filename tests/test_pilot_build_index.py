"""`build_index.py` — the manifest the browser reads instead of listing a directory.

A browser can list neither `data/decks/` nor `stacks/`, so the manifest carries the deck
list, each deck's passing stack filenames, and — since a scenario's cards used to be
guessed from prose — the cards each verified line is actually made of.
"""

from conftest import requires_deck

# ── The cards a verified line is made of ────────────────────────────────


def test_line_cards_reads_the_stack_not_the_board():
    """A verified line is what is on the STACK and in HAND, not what is on the table.

    The browser used to derive this by substring-matching every deck card name against the
    whole scenario blob, `board` included. Measured on the real artifact: heliod's
    Approach-of-the-Second-Sun scenario drew "verified" edges to Ancient Tomb and Howling
    Mine — lands that happened to be on the battlefield — while Swan Song, the actual
    interaction, was cut by a 4-card cap that truncated in NAME-LENGTH order.

    `board` is where a line is cast; it is not what the line is made of.
    """
    from manamap.pilot.build_index import line_cards

    scenario = {
        "board": {"you": ["Howling Mine", "Island x5", "Plains x3", "Ancient Tomb"],
                  "opponents": [{"life": 25, "board": ["untapped lands x2"]}]},
        "hand": ["Approach of the Second Sun", "Swan Song"],
        "graveyard": ["Mystical Tutor"],
        "stack": [{"pos": 0, "object": "Approach of the Second Sun", "controller": "you"}],
    }
    cards = line_cards(scenario)
    assert cards == ["Approach of the Second Sun", "Swan Song", "Mystical Tutor"], cards
    for bystander in ("Ancient Tomb", "Howling Mine", "Island", "Plains"):
        assert bystander not in cards, f"{bystander} is board furniture, not the line"


def test_line_cards_strips_annotations_and_quantities():
    """Scenario entries carry annotations a card name never has."""
    from manamap.pilot.build_index import line_cards

    cards = line_cards({
        "stack": [{"object": "Polyraptor (creature spell)"}],
        "hand": ["Island x5", "untapped lands x2", "Vampire Nighthawk (lifelink, 2/3)"],
    })
    assert "Polyraptor" in cards, cards
    assert "Vampire Nighthawk" in cards, cards
    # A quantity phrase with no name left is furniture, not a card.
    assert not any("x5" in c or c.lower() == "lands" for c in cards), cards


@requires_deck
def test_the_manifest_carries_the_line_cards():
    """The browser cannot list `stacks/`, and it must not guess at their contents either."""
    import json

    from manamap.config import DECKS_DIR

    manifest = json.loads((DECKS_DIR / "index.json").read_text(encoding="utf-8"))
    decks = manifest["decks"]
    assert decks, "no decks in the manifest"
    with_cards = [d for d in decks if d.get("stack_cards")]
    assert with_cards, "no deck carries stack_cards — the browser is back to guessing"
    for deck in with_cards:
        for name, cards in deck["stack_cards"].items():
            assert name in deck["stack_files"], (
                f"{deck['slug']}: stack_cards names {name}, which is not a passing stack"
            )
            assert cards and all(isinstance(c, str) and c.strip() for c in cards)
