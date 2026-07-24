"""Tests for deck ingestion: decklist parsing, Scryfall shaping, failure modes."""

import json

import pytest

from manamap.pilot import fetch_deck
from manamap.pilot.fetch_deck import parse_decklist, resolve_entries, shape_card
from manamap.pilot.validate_deck import validate

from conftest import requires_deck

REQUIRED_FIELDS = {
    "name", "quantity", "is_commander", "is_sideboard", "mana_cost", "cmc",
    "type_line", "oracle_text", "colors", "color_identity", "keywords", "power",
    "toughness", "loyalty", "layout", "image", "scryfall_uri", "card_faces",
}

VOLATILE_FIELDS = {"prices", "purchase_uris", "related_uris", "released_at", "edhrec_rank"}


def scryfall_card(name, **overrides):
    """Minimal Scryfall card object with volatile fields present (must be stripped)."""
    card = {
        "name": name,
        "mana_cost": "{R}",
        "cmc": 1.0,
        "type_line": "Creature — Goblin",
        "oracle_text": "Haste",
        "colors": ["R"],
        "color_identity": ["R"],
        "keywords": ["Haste"],
        "power": "1",
        "toughness": "1",
        "layout": "normal",
        "image_uris": {"normal": f"https://cards.scryfall.io/normal/{name}.jpg", "small": "x"},
        "scryfall_uri": f"https://scryfall.com/card/{name}",
        "prices": {"usd": "1.00"},
        "purchase_uris": {"tcgplayer": "x"},
        "related_uris": {"edhrec": "x"},
        "edhrec_rank": 1234,
    }
    card.update(overrides)
    return card


ADVENTURE_CARD = scryfall_card(
    "Bonecrusher Giant // Stomp",
    layout="adventure",
    oracle_text=None,
    image_uris=None,
    card_faces=[
        {
            "name": "Bonecrusher Giant",
            "mana_cost": "{2}{R}",
            "type_line": "Creature — Giant",
            "oracle_text": "Whenever this creature becomes the target of a spell...",
            "power": "4",
            "toughness": "3",
            "image_uris": {"normal": "https://cards.scryfall.io/normal/bonecrusher.jpg"},
        },
        {
            "name": "Stomp",
            "mana_cost": "{1}{R}",
            "type_line": "Instant — Adventure",
            "oracle_text": "Damage can't be prevented this turn. Stomp deals 2 damage...",
            "power": None,
            "toughness": None,
            "image_uris": None,
        },
    ],
)


# ── parse_decklist ──


def test_parse_decklist_formats():
    entries = parse_decklist(
        "# my deck\n"
        "Commander:\n"
        "1 Wort, Boggart Auntie\n"
        "\n"
        "Deck:\n"
        "1x Skirk Prospector\n"
        "10 Mountain\n"
        "Empty the Warrens\n"
        "// comment\n"
    )
    assert entries[0] == {"name": "Wort, Boggart Auntie", "quantity": 1,
                          "is_commander": True, "is_sideboard": False}
    assert entries[1]["name"] == "Skirk Prospector"
    assert entries[2] == {"name": "Mountain", "quantity": 10,
                          "is_commander": False, "is_sideboard": False}
    assert entries[3]["name"] == "Empty the Warrens"


def test_parse_decklist_cmdr_marker():
    entries = parse_decklist("1 Krenko, Mob Boss *CMDR*\n1 Mountain\n")
    assert entries[0]["is_commander"] is True
    assert entries[0]["name"] == "Krenko, Mob Boss"
    assert entries[1]["is_commander"] is False


def test_parse_decklist_moxfield_annotations():
    entries = parse_decklist(
        "1 Zada, Hedron Grinder (SLD) 2406 *F*\n"
        "1 Arena of Glory (PLST) MH3-215\n"
        "7 Mountain (SLD) 2418 *F*\n"
    )
    assert entries[0] == {"name": "Zada, Hedron Grinder", "quantity": 1,
                          "is_commander": False, "is_sideboard": False,
                          "set": "sld", "collector_number": "2406"}
    assert entries[1]["name"] == "Arena of Glory"
    assert entries[1]["set"] == "plst"
    assert entries[1]["collector_number"] == "MH3-215"
    assert entries[2]["quantity"] == 7


def test_parse_decklist_sideboard_section():
    entries = parse_decklist(
        "1 Skirk Prospector\n"
        "SIDEBOARD:\n"
        "1 Sazacap's Brew (PLST) BLB-151\n"
        "1 Storm Counter (SLD) 2422 *F*\n"
    )
    assert entries[0]["is_sideboard"] is False
    assert entries[1] == {"name": "Sazacap's Brew", "quantity": 1,
                          "is_commander": False, "is_sideboard": True,
                          "set": "plst", "collector_number": "BLB-151"}
    assert entries[2]["name"] == "Storm Counter"
    assert entries[2]["is_sideboard"] is True


def test_duplicate_basic_printings_merge():
    by_name = {"mountain": scryfall_card("Mountain", type_line="Basic Land — Mountain")}
    entries = [
        {"name": "Mountain", "quantity": 7, "is_commander": False, "is_sideboard": False},
        {"name": "Mountain", "quantity": 8, "is_commander": False, "is_sideboard": False},
        {"name": "Mountain", "quantity": 7, "is_commander": False, "is_sideboard": False},
    ]
    shaped, unmatched = resolve_entries(entries, by_name)
    assert unmatched == []
    assert len(shaped) == 1
    assert shaped[0]["quantity"] == 22


# ── check 1: mocked collection response → full schema ──


def mock_post(cards, not_found=None):
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": cards, "not_found": not_found or []}

    def _post(url, json=None, timeout=None):
        return FakeResponse()

    return _post


def test_mocked_three_card_fetch_has_all_fields(monkeypatch, tmp_path):
    cards = [scryfall_card("Skirk Prospector"), scryfall_card("Mountain",
             type_line="Basic Land — Mountain", mana_cost="", colors=[], keywords=[],
             power=None, toughness=None, oracle_text="{T}: Add {R}."), ADVENTURE_CARD]
    monkeypatch.setattr(fetch_deck.SESSION, "post", mock_post(cards))

    by_name, not_found = fetch_deck.fetch_collection(
        ["Skirk Prospector", "Mountain", "Bonecrusher Giant // Stomp"])
    assert not_found == []
    entries = [
        {"name": "Skirk Prospector", "quantity": 1, "is_commander": True, "is_sideboard": False},
        {"name": "Mountain", "quantity": 10, "is_commander": False, "is_sideboard": False},
        {"name": "Bonecrusher Giant // Stomp", "quantity": 1, "is_commander": False, "is_sideboard": False},
    ]
    shaped, unmatched = resolve_entries(entries, by_name)
    assert unmatched == []
    for card in shaped:
        assert set(card.keys()) == REQUIRED_FIELDS
        assert not VOLATILE_FIELDS & set(card.keys())

    adventure = shaped[2]
    assert len(adventure["card_faces"]) == 2
    assert adventure["card_faces"][0]["name"] == "Bonecrusher Giant"
    assert adventure["image"] == "https://cards.scryfall.io/normal/bonecrusher.jpg"
    assert "Stomp deals 2 damage" in adventure["oracle_text"]


def test_resolve_by_single_face_name():
    by_name = {"bonecrusher giant // stomp": ADVENTURE_CARD}
    shaped, unmatched = resolve_entries(
        [{"name": "Bonecrusher Giant", "quantity": 1, "is_commander": False,
          "is_sideboard": False}], by_name)
    assert unmatched == []
    assert shaped[0]["name"] == "Bonecrusher Giant // Stomp"


def test_shape_is_json_stable():
    a = json.dumps(shape_card(scryfall_card("X"), 1, False), sort_keys=True)
    b = json.dumps(shape_card(scryfall_card("X"), 1, False), sort_keys=True)
    assert a == b


# ── check 3: misspelled card fails loudly, naming the card ──


def test_misspelled_card_fails_loudly(monkeypatch, tmp_path):
    deck = tmp_path / "decks" / "test-deck"
    deck.mkdir(parents=True)
    (deck / "decklist.txt").write_text("1 Gobiln Matron\n")
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", tmp_path / "decks")
    monkeypatch.setattr(fetch_deck.SESSION, "post",
                        mock_post([], not_found=[{"name": "Gobiln Matron"}]))

    class Args:
        slug = "test-deck"

    with pytest.raises(SystemExit) as exc:
        fetch_deck.main(Args())
    assert "Gobiln Matron" in str(exc.value)


# ── validate_deck unit checks ──


def test_validate_deck_catches_violations():
    doc = {"cards": [
        {"name": "A", "quantity": 2, "is_commander": True, "type_line": "Creature",
         "color_identity": ["R"]},
        {"name": "B", "quantity": 97, "is_commander": False,
         "type_line": "Basic Land — Mountain", "color_identity": []},
        {"name": "C", "quantity": 1, "is_commander": False, "type_line": "Instant",
         "color_identity": ["U"]},
    ]}
    errors = validate(doc)
    assert any("Singleton violation: A" in e for e in errors)
    assert any("Color identity violation: C" in e for e in errors)
    assert not any("100" in e for e in errors)  # total is exactly 100


# ── check 2: real deck (data-gated until decklist pasted) ──


@requires_deck
def test_real_deck_is_100_with_commander():
    from manamap.pilot.common import load_deck_cards

    doc = load_deck_cards("goblin-storm")
    assert sum(c["quantity"] for c in doc["cards"]) == 100
    assert any(c["is_commander"] for c in doc["cards"])
    assert validate(doc) == []
