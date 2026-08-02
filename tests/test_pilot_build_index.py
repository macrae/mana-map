"""`build_index.py` — the manifest the browser reads instead of listing a directory.

A browser can list neither `data/decks/` nor `stacks/`, so the manifest carries the deck
list, each deck's passing stack filenames, and — since a scenario's cards used to be
guessed from prose — the cards each verified line is actually made of.
"""

import json

from conftest import requires_deck
from manamap.pilot import build_index

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
def test_line_cards_drops_basic_lands_and_tokens():
    """Furniture that looks like a card name and passes every other filter.

    A scenario states its mana, so basics appear on nearly every board. Unlike a
    token, a basic IS in the deck, so it survives the browser's name check and
    draws a real edge to a real card that had nothing to do with the line — the
    Ancient-Tomb-and-Howling-Mine failure recurring one level down. Measured on
    yawgmoth-swarm, whose line derived "Insect token X" and "Swamp" and escaped
    only because a four-card cap truncated before reaching the Swamp.
    """
    from manamap.pilot.build_index import line_cards

    scenario = {
        "stack": [],
        "hand": [],
        "graveyard": ["Fume Spitter"],
        "board": {"you": [
            "Blowfly Infestation (enchantment)",
            "Nest of Scarabs (enchantment)",
            "Zulaport Cutthroat (1/1, no counters)",
            "Insect token X (1/1 black Insect, no counters)",
            "Snake token B",
            "Swamp (untapped)",
            "Snow-Covered Forest",
        ]},
    }
    assert line_cards(scenario) == [
        "Fume Spitter", "Blowfly Infestation", "Nest of Scarabs", "Zulaport Cutthroat",
    ]


def test_line_cards_keeps_nonbasic_lands():
    """Only BASICS are furniture. A utility land can genuinely be part of a line."""
    from manamap.pilot.build_index import line_cards

    scenario = {"stack": [], "hand": [], "graveyard": [],
                "board": {"you": ["Phyrexian Tower", "Ancient Tomb", "Swamp"]}}
    assert line_cards(scenario) == ["Phyrexian Tower", "Ancient Tomb"]


def test_the_count_phrase_guard_actually_matches():
    """It was written `r"^(\\d+|...)\\s"` — a literal backslash — and matched nothing.

    The guard was dead from the day it was written, so "2 Vampire tokens" passed
    straight through it. Assert on the behaviour, not the pattern.
    """
    from manamap.pilot.build_index import _COUNT_PHRASE

    for furniture in ("2 Vampire tokens", "five lands, all untapped", "a land",
                      "one Swamp", "10 Forests"):
        assert _COUNT_PHRASE.match(furniture), furniture
    for real in ("Blowfly Infestation", "Ancient Tomb", "Sol Ring"):
        assert not _COUNT_PHRASE.match(real), real


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


# ── The rack and the manifest answer different questions ────────────────


def _deck(tmp_path, slug, published, manuals):
    d = tmp_path / slug
    (d / "stacks").mkdir(parents=True)
    (d / "decisions").mkdir()
    (d / "cards.json").write_text(json.dumps(
        {"cards": [{"name": "Yawgmoth, Thran Physician", "is_commander": True}]}))
    if published:
        (manuals / f"{slug}.html").write_text("<html></html>")
    return d


def test_a_deck_without_a_manual_is_still_in_the_manifest(tmp_path, monkeypatch):
    """A built, validated, rules-verified deck must be loadable in the viz.

    Gating the manifest on `manuals/<slug>.html` meant a deck stayed invisible in
    the frontend until someone spent an agent budget on magazine prose for it.
    """
    manuals = tmp_path / "manuals"; manuals.mkdir()
    _deck(tmp_path, "published-deck", True, manuals)
    _deck(tmp_path, "unpublished-deck", False, manuals)
    monkeypatch.setattr(build_index, "DECKS_DIR", tmp_path)
    monkeypatch.setattr(build_index, "MANUALS_DIR", manuals)

    entries = build_index.gather_entries()
    slugs = {e["slug"]: e["published"] for e in entries}
    assert slugs == {"published-deck": True, "unpublished-deck": False}


def test_the_rack_shows_only_published_issues(tmp_path, monkeypatch):
    manuals = tmp_path / "manuals"; manuals.mkdir()
    _deck(tmp_path, "published-deck", True, manuals)
    _deck(tmp_path, "unpublished-deck", False, manuals)
    monkeypatch.setattr(build_index, "DECKS_DIR", tmp_path)
    monkeypatch.setattr(build_index, "MANUALS_DIR", manuals)

    html = build_index.render_index(build_index.gather_entries())
    assert "published-deck" in html
    assert "unpublished-deck" not in html


def test_the_manifest_carries_the_published_flag(tmp_path, monkeypatch):
    """The browser needs to tell a loadable deck from one that went to press."""
    manuals = tmp_path / "manuals"; manuals.mkdir()
    _deck(tmp_path, "unpublished-deck", False, manuals)
    monkeypatch.setattr(build_index, "DECKS_DIR", tmp_path)
    monkeypatch.setattr(build_index, "MANUALS_DIR", manuals)

    entries = build_index.gather_entries()
    manifest = {k: v for k, v in ((e["slug"], e) for e in entries)}
    assert "published" in manifest["unpublished-deck"]


def test_a_stray_file_in_decks_dir_is_ignored(tmp_path, monkeypatch):
    """`data/decks/index.json` lives beside the deck directories."""
    manuals = tmp_path / "manuals"; manuals.mkdir()
    _deck(tmp_path, "a-deck", False, manuals)
    (tmp_path / "index.json").write_text("{}")
    monkeypatch.setattr(build_index, "DECKS_DIR", tmp_path)
    monkeypatch.setattr(build_index, "MANUALS_DIR", manuals)
    assert [e["slug"] for e in build_index.gather_entries()] == ["a-deck"]
