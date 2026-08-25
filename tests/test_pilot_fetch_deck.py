"""Tests for deck ingestion: decklist parsing, Scryfall shaping, failure modes."""

import argparse
import json

import pytest

from manamap.pilot import fetch_deck
from manamap.pilot.fetch_deck import parse_decklist, resolve_entries, shape_card
from manamap.pilot.validate_deck import validate

from conftest import requires_deck

REQUIRED_FIELDS = {
    "name", "quantity", "is_commander", "mana_cost", "cmc",
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
                          "is_commander": True, "foil": False}
    assert entries[1]["name"] == "Skirk Prospector"
    assert entries[2] == {"name": "Mountain", "quantity": 10,
                          "is_commander": False, "foil": False}
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
                          "is_commander": False,
                          "set": "sld", "collector_number": "2406", "foil": True}
    assert entries[1]["name"] == "Arena of Glory"
    assert entries[1]["set"] == "plst"
    assert entries[1]["collector_number"] == "MH3-215"
    assert entries[2]["quantity"] == 7


def test_duplicate_basic_printings_merge():
    by_name = {"mountain": scryfall_card("Mountain", type_line="Basic Land — Mountain")}
    entries = [
        {"name": "Mountain", "quantity": 7, "is_commander": False},
        {"name": "Mountain", "quantity": 8, "is_commander": False},
        {"name": "Mountain", "quantity": 7, "is_commander": False},
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
        {"name": "Skirk Prospector", "quantity": 1, "is_commander": True},
        {"name": "Mountain", "quantity": 10, "is_commander": False},
        {"name": "Bonecrusher Giant // Stomp", "quantity": 1, "is_commander": False},
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
        [{"name": "Bonecrusher Giant", "quantity": 1, "is_commander": False}], by_name)
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
    cards = doc["cards"]
    assert sum(c["quantity"] for c in cards) == 100
    assert any(c["is_commander"] for c in cards)
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
    entry = {"name": "Zada, Hedron Grinder", "quantity": 1, "is_commander": True, "foil": True, "set": "sld",
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
    entry = {"name": "Zada, Hedron Grinder", "quantity": 1, "is_commander": True, "foil": False, "set": "xxx",
             "collector_number": "999"}
    shaped, unmatched = resolve_entries(
        [entry], by_name={"zada, hedron grinder": DEFAULT_PRINTING}, by_printing={})
    assert unmatched == [] and shaped[0]["set"] == "cmm"


def test_image_urls_drop_cache_busting_query():
    """Scryfall's ?timestamp churns cards.json on every re-fetch for no visual change."""
    card = shape_card(SECRET_LAIR, 1, True, foil=True)
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

    plain = shape_card(DEFAULT_PRINTING, 1, True, foil=False)
    fancy = shape_card(SECRET_LAIR, 1, True, foil=True)
    assert {k: plain[k] for k in CARD_SEMANTIC_FIELDS} == \
           {k: fancy[k] for k in CARD_SEMANTIC_FIELDS}
    assert plain["artist"] != fancy["artist"]     # but presentation differs


# ── Multi-face colours ───────────────────────────────────────────────────
#
# Scryfall omits the top-level `colors` for transform and modal_dfc layouts —
# it lives on card_faces. fetch_deck used to copy the top-level field verbatim
# AND drop colours when shaping faces, so cards.json recorded `[]` for every
# DFC with no way to recover, while cards.csv (which uses get_colors) was right.
# `colors` is in CARD_SEMANTIC_FIELDS, so the wrong value was agent-facing.

TRANSFORM_DFC = {
    "name": "Rona, Herald of Invasion // Rona, Tolarian Obliterator",
    "layout": "transform",
    "cmc": 2.0,
    "type_line": "Legendary Creature — Human Artificer // Legendary Creature — Phyrexian Praetor",
    "color_identity": ["B", "U"],
    # NOTE: no top-level "colors" key at all — this is what Scryfall actually sends.
    "card_faces": [
        {"name": "Rona, Herald of Invasion", "mana_cost": "{1}{U}", "colors": ["U"],
         "type_line": "Legendary Creature — Human Artificer", "oracle_text": "front"},
        {"name": "Rona, Tolarian Obliterator", "mana_cost": "", "colors": ["B"],
         "type_line": "Legendary Creature — Phyrexian Praetor", "oracle_text": "back"},
    ],
}

DFC_LAND = {
    "name": "Darkbore Pathway // Slitherbore Pathway",
    "layout": "modal_dfc",
    "cmc": 0.0,
    "type_line": "Land // Land",
    "color_identity": ["B", "G"],
    "card_faces": [
        {"name": "Darkbore Pathway", "mana_cost": "", "colors": [],
         "type_line": "Land", "oracle_text": "{T}: Add {B}."},
        {"name": "Slitherbore Pathway", "mana_cost": "", "colors": [],
         "type_line": "Land", "oracle_text": "{T}: Add {G}."},
    ],
}


def test_transform_dfc_colors_union_the_faces():
    card = shape_card(TRANSFORM_DFC, 1, False)
    assert card["colors"] == ["U", "B"], "face colours must be unioned in WUBRG order"


def test_dfc_faces_carry_their_own_colors():
    """Without this the union has nothing to read and cannot be recomputed later."""
    card = shape_card(TRANSFORM_DFC, 1, False)
    assert [f["colors"] for f in card["card_faces"]] == [["U"], ["B"]]


def test_dfc_land_is_legitimately_colorless():
    """A Pathway is two Lands: colourless objects. The B/G lives in color_identity."""
    card = shape_card(DFC_LAND, 1, False)
    assert card["colors"] == []
    assert card["color_identity"] == ["B", "G"]


def test_single_faced_card_colors_still_come_from_the_top_level():
    card = shape_card(DEFAULT_PRINTING, 1, True)
    assert card["colors"] == DEFAULT_PRINTING["colors"]


@requires_deck
def test_real_deck_dfc_colors_agree_with_cards_csv():
    """The two sources must not disagree — they share get_colors by construction."""
    import pandas as pd

    from manamap.config import DECKS_DIR, OUTPUT_CSV_PATH

    if not OUTPUT_CSV_PATH.exists():
        pytest.skip("cards.csv not built")
    csv = pd.read_csv(OUTPUT_CSV_PATH, usecols=["name", "colors"])
    expected = {
        r["name"]: (set(str(r["colors"]).split(", ")) if pd.notna(r["colors"]) else set())
        for _, r in csv.iterrows()
    }
    checked = 0
    for cards_json in sorted(DECKS_DIR.glob("*/cards.json")):
        with open(cards_json) as f:
            doc = json.load(f)
        for card in doc["cards"]:
            if card.get("layout") not in ("transform", "modal_dfc", "reversible_card"):
                continue
            want = expected.get(card["name"])
            if want is None:
                continue
            assert set(card["colors"]) == want, (
                f"{cards_json.parent.name}/{card['name']}: "
                f"cards.json {sorted(card['colors'])} != cards.csv {sorted(want)}"
            )
            checked += 1
    assert checked, "no multi-face cards found in any committed deck"


# ---------------------------------------------------------------------------
# A dropped connection is not a status code
#
# `_post_collection` retries 429 and 5xx, and did so by inspecting
# `resp.status_code` — which means it could only see a failure the server was
# well enough to describe. A closed keep-alive socket raises inside
# `SESSION.post`, so there is no response to inspect, and the exception went
# straight past four retries written to survive exactly this. It surfaced in the
# browser, mid-build, as `ConnectionError: ('Connection aborted.',
# RemoteDisconnected(...))`.
# ---------------------------------------------------------------------------

class _Resp:
    status_code = 200

    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": self._data, "not_found": []}


def test_a_dropped_connection_is_retried_rather_than_raised(monkeypatch):
    """The transport failure that used to escape the retry loop entirely."""
    import requests

    from manamap.pilot import fetch_deck

    calls = []

    def flaky(url, json=None, timeout=None):
        calls.append(json)
        if len(calls) == 1:
            raise requests.exceptions.ConnectionError(
                "('Connection aborted.', RemoteDisconnected('Remote end closed "
                "connection without response'))")
        return _Resp([{"name": "Sol Ring"}])

    monkeypatch.setattr(fetch_deck.SESSION, "post", flaky)
    monkeypatch.setattr(fetch_deck.time, "sleep", lambda s: None)

    cards, not_found = fetch_deck._post_collection([{"name": "Sol Ring"}])

    assert len(calls) == 2, "the first attempt dropped; it must be retried"
    assert [c["name"] for c in cards] == ["Sol Ring"]
    assert not_found == []


def test_the_retry_reuses_no_dead_socket(monkeypatch):
    """A dropped keep-alive socket stays in the pool unless the session closes.

    Retrying over the same pooled connection fails identically, which would
    make the retry loop look like it ran without doing anything.
    """
    import requests

    from manamap.pilot import fetch_deck

    closed = []
    monkeypatch.setattr(fetch_deck.SESSION, "close", lambda: closed.append(1))
    monkeypatch.setattr(fetch_deck.time, "sleep", lambda s: None)

    state = {"n": 0}

    def flaky(url, json=None, timeout=None):
        state["n"] += 1
        if state["n"] == 1:
            raise requests.exceptions.ConnectionError("aborted")
        return _Resp([])

    monkeypatch.setattr(fetch_deck.SESSION, "post", flaky)
    fetch_deck._post_collection([{"name": "Sol Ring"}])

    assert closed, "the dead connection must be dropped before retrying"


def test_giving_up_says_what_to_do_rather_than_naming_a_socket(monkeypatch):
    """Exhausting the retries is an ordinary operating condition, not a bug.

    What reaches the pilot must be a sentence, not the repr of a urllib3
    exception. The deck is untouched at this point, so "run it again" is both
    true and the entire remedy.
    """
    import requests

    from manamap.pilot import fetch_deck

    monkeypatch.setattr(fetch_deck.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        fetch_deck.SESSION, "post",
        lambda *a, **k: (_ for _ in ()).throw(
            requests.exceptions.ConnectionError("aborted")))

    with pytest.raises(RuntimeError) as exc:
        fetch_deck._post_collection([{"name": "Sol Ring"}])

    message = str(exc.value)
    assert "Scryfall" in message
    assert "again" in message, "it must say what to do"
    assert "RemoteDisconnected" not in message
    assert "urllib3" not in message
