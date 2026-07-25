"""Tests for deck ingestion: decklist parsing, Scryfall shaping, failure modes."""

import argparse
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
    # Printing identity — which physical card the pilot owns.
    "art_crop", "set", "set_name", "collector_number", "artist",
    "border_color", "frame_effects", "finishes", "foil",
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
                          "is_commander": True, "is_sideboard": False, "foil": False}
    assert entries[1]["name"] == "Skirk Prospector"
    assert entries[2] == {"name": "Mountain", "quantity": 10,
                          "is_commander": False, "is_sideboard": False, "foil": False}
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
                          "set": "sld", "collector_number": "2406", "foil": True}
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
                          "set": "plst", "collector_number": "BLB-151", "foil": False}
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
    main = [c for c in doc["cards"] if not c["is_sideboard"]]
    assert sum(c["quantity"] for c in main) == 100
    assert any(c["is_commander"] for c in main)
    assert validate(doc) == []


# ── Decklist-unchanged short-circuit ─────────────────────────────────────


def test_unchanged_decklist_skips_scryfall(monkeypatch, tmp_path, capsys):
    """A no-op re-fetch must make no network call and leave cards.json alone.

    Beyond the saved round trips: rewriting cards.json would invalidate every
    downstream agent routine that reads it.
    """
    import hashlib

    decks = tmp_path / "decks"
    base = decks / "d"
    base.mkdir(parents=True)
    decklist = "1 Sol Ring\n"
    (base / "decklist.txt").write_text(decklist)
    sha = hashlib.sha256(decklist.encode("utf-8")).hexdigest()
    cards_path = base / "cards.json"
    cards_path.write_text(json.dumps({"deck": "d", "decklist_sha256": sha, "cards": []}))
    before = cards_path.read_bytes()

    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", decks)

    def explode(*a, **k):
        raise AssertionError("Scryfall must not be called for an unchanged decklist")

    monkeypatch.setattr(fetch_deck, "fetch_collection", explode)
    fetch_deck.main(argparse.Namespace(slug="d", force=False))

    assert cards_path.read_bytes() == before
    assert "Already up to date" in capsys.readouterr().out


def test_force_refetches_even_when_unchanged(monkeypatch, tmp_path):
    import hashlib

    decks = tmp_path / "decks"
    base = decks / "d"
    base.mkdir(parents=True)
    decklist = "1 Sol Ring\n"
    (base / "decklist.txt").write_text(decklist)
    sha = hashlib.sha256(decklist.encode("utf-8")).hexdigest()
    (base / "cards.json").write_text(
        json.dumps({"deck": "d", "decklist_sha256": sha, "cards": []}))

    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", decks)
    called = {"n": 0}

    def fake_fetch(names):
        called["n"] += 1
        return {"sol ring": {"name": "Sol Ring", "cmc": 1.0, "type_line": "Artifact",
                             "oracle_text": "", "image_uris": {"normal": "u"}}}, []

    monkeypatch.setattr(fetch_deck, "fetch_collection", fake_fetch)
    fetch_deck.main(argparse.Namespace(slug="d", force=True))
    assert called["n"] == 1


def test_changed_decklist_refetches(monkeypatch, tmp_path):
    decks = tmp_path / "decks"
    base = decks / "d"
    base.mkdir(parents=True)
    (base / "decklist.txt").write_text("1 Sol Ring\n")
    (base / "cards.json").write_text(
        json.dumps({"deck": "d", "decklist_sha256": "stale", "cards": []}))

    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", decks)
    called = {"n": 0}

    def fake_fetch(names):
        called["n"] += 1
        return {"sol ring": {"name": "Sol Ring", "cmc": 1.0, "type_line": "Artifact",
                             "oracle_text": "", "image_uris": {"normal": "u"}}}, []

    monkeypatch.setattr(fetch_deck, "fetch_collection", fake_fetch)
    fetch_deck.main(argparse.Namespace(slug="d", force=False))
    assert called["n"] == 1


# ── Exact printings ──────────────────────────────────────────────────────

SECRET_LAIR = {
    "name": "Zada, Hedron Grinder", "mana_cost": "{3}{R}", "cmc": 4.0,
    "type_line": "Legendary Creature — Goblin Ally", "oracle_text": "Copy that spell.",
    "colors": ["R"], "color_identity": ["R"], "keywords": [], "power": "3",
    "toughness": "3", "loyalty": None, "layout": "normal",
    "set": "sld", "set_name": "Secret Lair Drop", "collector_number": "2406",
    "artist": "Wizard of Barge", "border_color": "borderless",
    "frame_effects": ["legendary", "inverted"], "finishes": ["foil"],
    "image_uris": {"normal": "https://cards.scryfall.io/normal/sld.jpg?1783903430",
                   "art_crop": "https://cards.scryfall.io/art_crop/sld.jpg?1783903430"},
    "scryfall_uri": "https://scryfall.com/card/sld/2406",
    "prices": {"usd": "9.99"},
}

DEFAULT_PRINTING = dict(
    SECRET_LAIR, set="cmm", set_name="Commander Masters", collector_number="268",
    artist="Someone Else", border_color="black", frame_effects=[],
    finishes=["nonfoil", "foil"],
    image_uris={"normal": "https://cards.scryfall.io/normal/cmm.jpg?1"},
)


def test_printing_annotation_wins_over_name_lookup():
    """A Moxfield export names the physical card; a default reprint must lose."""
    entry = {"name": "Zada, Hedron Grinder", "quantity": 1, "is_commander": True,
             "is_sideboard": False, "foil": True, "set": "sld",
             "collector_number": "2406"}
    shaped, unmatched = resolve_entries(
        [entry],
        by_name={"zada, hedron grinder": DEFAULT_PRINTING},
        by_printing={("sld", "2406"): SECRET_LAIR},
    )
    assert unmatched == []
    card = shaped[0]
    assert card["set"] == "sld"
    assert card["collector_number"] == "2406"
    assert card["artist"] == "Wizard of Barge"
    assert card["border_color"] == "borderless"
    assert card["foil"] is True


def test_name_lookup_is_the_fallback_when_printing_unresolvable():
    entry = {"name": "Zada, Hedron Grinder", "quantity": 1, "is_commander": True,
             "is_sideboard": False, "foil": False, "set": "xxx",
             "collector_number": "999"}
    shaped, unmatched = resolve_entries(
        [entry], by_name={"zada, hedron grinder": DEFAULT_PRINTING}, by_printing={})
    assert unmatched == [] and shaped[0]["set"] == "cmm"


def test_image_urls_drop_cache_busting_query():
    """Scryfall's ?timestamp churns cards.json on every re-fetch for no visual change."""
    card = shape_card(SECRET_LAIR, 1, True, False, foil=True)
    assert card["image"] == "https://cards.scryfall.io/normal/sld.jpg"
    assert card["art_crop"] == "https://cards.scryfall.io/art_crop/sld.jpg"
    assert "?" not in card["image"] and "?" not in card["art_crop"]


def test_foil_marker_flows_from_decklist_to_card():
    entries = parse_decklist("1 Zada, Hedron Grinder (SLD) 2406 *F*\n")
    assert entries[0]["foil"] is True
    shaped, _ = resolve_entries(entries, by_name={},
                                by_printing={("sld", "2406"): SECRET_LAIR})
    assert shaped[0]["foil"] is True


def test_printing_metadata_is_not_agent_semantic():
    """Enriching printings must not invalidate agent routines (docs/agent-cost.md)."""
    from manamap.pilot.agent_cache import CARD_SEMANTIC_FIELDS

    plain = shape_card(DEFAULT_PRINTING, 1, True, False, foil=False)
    fancy = shape_card(SECRET_LAIR, 1, True, False, foil=True)
    assert {k: plain[k] for k in CARD_SEMANTIC_FIELDS} == \
           {k: fancy[k] for k in CARD_SEMANTIC_FIELDS}
    assert plain["artist"] != fancy["artist"]     # but presentation differs
