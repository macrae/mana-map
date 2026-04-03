"""Tests for extract.py — verifies derived columns and multi-face handling."""

import pytest

from extract import (
    build_embedding_text,
    combine_faces,
    derive_primary_color,
    derive_supertype,
    first_face_value,
    get_colors,
    process_card,
)


# ---------------------------------------------------------------------------
# Fixtures: realistic Scryfall card objects
# ---------------------------------------------------------------------------

def _lightning_bolt():
    """Normal single-face card."""
    return {
        "oracle_id": "abc-123",
        "name": "Lightning Bolt",
        "layout": "normal",
        "type_line": "Instant",
        "mana_cost": "{R}",
        "cmc": 1.0,
        "colors": ["R"],
        "color_identity": ["R"],
        "oracle_text": "Lightning Bolt deals 3 damage to any target.",
        "keywords": [],
        "power": None,
        "toughness": None,
        "loyalty": None,
        "defense": None,
        "rarity": "uncommon",
        "set": "leb",
        "set_name": "Limited Edition Beta",
        "released_at": "1993-10-04",
        "artist": "Christopher Rush",
        "flavor_text": None,
        "edhrec_rank": 5,
        "reserved": False,
        "legalities": {
            "standard": "not_legal",
            "modern": "legal",
            "legacy": "legal",
            "vintage": "legal",
            "commander": "legal",
            "pioneer": "not_legal",
            "pauper": "legal",
            "historic": "legal",
        },
    }


def _fire_ice():
    """Split card — colors on top-level, NOT on faces."""
    return {
        "oracle_id": "split-001",
        "name": "Fire // Ice",
        "layout": "split",
        "type_line": "Instant // Instant",
        "cmc": 4.0,
        "colors": ["R", "U"],
        "color_identity": ["R", "U"],
        "keywords": [],
        "rarity": "uncommon",
        "set": "dmr",
        "set_name": "Dominaria Remastered",
        "released_at": "2023-01-13",
        "artist": "David Martin",
        "flavor_text": None,
        "edhrec_rank": 3000,
        "reserved": False,
        "legalities": {
            "standard": "not_legal",
            "modern": "legal",
            "legacy": "legal",
            "vintage": "legal",
            "commander": "legal",
            "pioneer": "not_legal",
            "pauper": "not_legal",
            "historic": "not_legal",
        },
        "card_faces": [
            {
                "name": "Fire",
                "mana_cost": "{1}{R}",
                "type_line": "Instant",
                "oracle_text": "Fire deals 2 damage divided as you choose among one or two targets.",
                "colors": None,
            },
            {
                "name": "Ice",
                "mana_cost": "{1}{U}",
                "type_line": "Instant",
                "oracle_text": "Tap target permanent.\nDraw a card.",
                "colors": None,
            },
        ],
    }


def _bonecrusher_giant():
    """Adventure card — colors on top-level, NOT on faces."""
    return {
        "oracle_id": "adventure-001",
        "name": "Bonecrusher Giant // Stomp",
        "layout": "adventure",
        "type_line": "Creature — Giant // Instant — Adventure",
        "cmc": 3.0,
        "colors": ["R"],
        "color_identity": ["R"],
        "keywords": ["Adventure"],
        "power": None,
        "toughness": None,
        "loyalty": None,
        "defense": None,
        "rarity": "rare",
        "set": "eld",
        "set_name": "Throne of Eldraine",
        "released_at": "2019-10-04",
        "artist": "Victor Adame Minguez",
        "flavor_text": None,
        "edhrec_rank": 500,
        "reserved": False,
        "legalities": {
            "standard": "not_legal",
            "modern": "legal",
            "legacy": "legal",
            "vintage": "legal",
            "commander": "legal",
            "pioneer": "legal",
            "pauper": "not_legal",
            "historic": "legal",
        },
        "card_faces": [
            {
                "name": "Bonecrusher Giant",
                "mana_cost": "{2}{R}",
                "type_line": "Creature — Giant",
                "oracle_text": "Whenever this creature becomes the target of a spell, this creature deals 2 damage to that spell's controller.",
                "colors": None,
                "power": "4",
                "toughness": "3",
            },
            {
                "name": "Stomp",
                "mana_cost": "{1}{R}",
                "type_line": "Instant — Adventure",
                "oracle_text": "Damage can't be prevented this turn. Stomp deals 2 damage to any target.",
                "colors": None,
            },
        ],
    }


def _fable():
    """Transform card — colors on faces, NOT on top-level."""
    return {
        "oracle_id": "transform-001",
        "name": "Fable of the Mirror-Breaker // Reflection of Kiki-Jiki",
        "layout": "transform",
        "type_line": "Enchantment — Saga // Enchantment Creature — Goblin Shaman",
        "cmc": 3.0,
        "colors": None,
        "color_identity": ["R"],
        "keywords": [],
        "rarity": "rare",
        "set": "neo",
        "set_name": "Kamigawa: Neon Dynasty",
        "released_at": "2022-02-18",
        "artist": "Anna Steinbauer",
        "flavor_text": None,
        "edhrec_rank": 100,
        "reserved": False,
        "legalities": {
            "standard": "not_legal",
            "modern": "legal",
            "legacy": "legal",
            "vintage": "legal",
            "commander": "legal",
            "pioneer": "legal",
            "pauper": "not_legal",
            "historic": "legal",
        },
        "card_faces": [
            {
                "name": "Fable of the Mirror-Breaker",
                "mana_cost": "{2}{R}",
                "type_line": "Enchantment — Saga",
                "oracle_text": "(As this Saga enters and after your draw step, add a lore counter.) I — Create a 2/2 red Goblin Shaman creature token with \"Whenever this creature attacks, create a Treasure token.\"",
                "colors": ["R"],
            },
            {
                "name": "Reflection of Kiki-Jiki",
                "mana_cost": "",
                "type_line": "Enchantment Creature — Goblin Shaman",
                "oracle_text": "{1}, {T}: Create a token that's a copy of another target nonlegendary creature you control, except it has haste. Sacrifice it at the beginning of the next end step.",
                "colors": ["R"],
                "power": "2",
                "toughness": "2",
            },
        ],
    }


def _jace():
    """Planeswalker — should have loyalty."""
    return {
        "oracle_id": "pw-001",
        "name": "Jace, the Mind Sculptor",
        "layout": "normal",
        "type_line": "Legendary Planeswalker — Jace",
        "mana_cost": "{2}{U}{U}",
        "cmc": 4.0,
        "colors": ["U"],
        "color_identity": ["U"],
        "oracle_text": "+2: Look at the top card of target player's library.",
        "keywords": [],
        "power": None,
        "toughness": None,
        "loyalty": "3",
        "defense": None,
        "rarity": "mythic",
        "set": "wwk",
        "set_name": "Worldwake",
        "released_at": "2010-02-05",
        "artist": "Jason Chan",
        "flavor_text": None,
        "edhrec_rank": 1000,
        "reserved": False,
        "legalities": {
            "standard": "not_legal",
            "modern": "legal",
            "legacy": "legal",
            "vintage": "restricted",
            "commander": "legal",
            "pioneer": "not_legal",
            "pauper": "not_legal",
            "historic": "not_legal",
        },
    }


def _breeding_pool():
    """Dual land — empty colors, but color_identity has G, U."""
    return {
        "oracle_id": "land-001",
        "name": "Breeding Pool",
        "layout": "normal",
        "type_line": "Land — Forest Island",
        "mana_cost": "",
        "cmc": 0.0,
        "colors": [],
        "color_identity": ["G", "U"],
        "oracle_text": "({T}: Add {G} or {U}.)\nAs Breeding Pool enters, you may pay 2 life. If you don't, it enters tapped.",
        "keywords": [],
        "power": None,
        "toughness": None,
        "loyalty": None,
        "defense": None,
        "rarity": "rare",
        "set": "rna",
        "set_name": "Ravnica Allegiance",
        "released_at": "2019-01-25",
        "artist": "Jenn Ravenna",
        "flavor_text": None,
        "edhrec_rank": 50,
        "reserved": False,
        "legalities": {
            "standard": "not_legal",
            "modern": "legal",
            "legacy": "legal",
            "vintage": "legal",
            "commander": "legal",
            "pioneer": "legal",
            "pauper": "not_legal",
            "historic": "legal",
        },
    }


def _mountain():
    """Basic land — empty colors, color_identity R."""
    return {
        "oracle_id": "land-002",
        "name": "Mountain",
        "layout": "normal",
        "type_line": "Basic Land — Mountain",
        "mana_cost": "",
        "cmc": 0.0,
        "colors": [],
        "color_identity": ["R"],
        "oracle_text": "({T}: Add {R}.)",
        "keywords": [],
        "power": None,
        "toughness": None,
        "loyalty": None,
        "defense": None,
        "rarity": "common",
        "set": "leb",
        "set_name": "Limited Edition Beta",
        "released_at": "1993-10-04",
        "artist": "Douglas Shuler",
        "flavor_text": None,
        "edhrec_rank": None,
        "reserved": False,
        "legalities": {
            "standard": "legal",
            "modern": "legal",
            "legacy": "legal",
            "vintage": "legal",
            "commander": "legal",
            "pioneer": "legal",
            "pauper": "legal",
            "historic": "legal",
        },
    }


def _modal_dfc():
    """Modal DFC — colors on faces (like transform)."""
    return {
        "oracle_id": "mdfc-001",
        "name": "Emeria's Call // Emeria, Shattered Skyclave",
        "layout": "modal_dfc",
        "type_line": "Sorcery // Land",
        "cmc": 7.0,
        "colors": None,
        "color_identity": ["W"],
        "keywords": [],
        "rarity": "mythic",
        "set": "znr",
        "set_name": "Zendikar Rising",
        "released_at": "2020-09-25",
        "artist": "Matt Stewart",
        "flavor_text": None,
        "edhrec_rank": 500,
        "reserved": False,
        "legalities": {
            "standard": "not_legal",
            "modern": "legal",
            "legacy": "legal",
            "vintage": "legal",
            "commander": "legal",
            "pioneer": "legal",
            "pauper": "not_legal",
            "historic": "legal",
        },
        "card_faces": [
            {
                "name": "Emeria's Call",
                "mana_cost": "{4}{W}{W}{W}",
                "type_line": "Sorcery",
                "oracle_text": "Create two 4/4 white Angel Warrior creature tokens with flying. Non-Angel creatures you control gain indestructible until your next turn.",
                "colors": ["W"],
            },
            {
                "name": "Emeria, Shattered Skyclave",
                "mana_cost": "",
                "type_line": "Land",
                "oracle_text": "As this land enters, you may pay 3 life. If you don't, it enters tapped.\n{T}: Add {W}.",
                "colors": [],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Tests: get_colors
# ---------------------------------------------------------------------------

class TestGetColors:
    def test_split_uses_top_level_colors(self):
        """Split cards (Fire // Ice) have colors on top-level, not faces."""
        card = _fire_ice()
        assert get_colors(card) == ["U", "R"]

    def test_adventure_uses_top_level_colors(self):
        """Adventure cards have colors on top-level, not faces."""
        card = _bonecrusher_giant()
        assert get_colors(card) == ["R"]

    def test_transform_uses_face_colors(self):
        """Transform cards have colors on faces, not top-level."""
        card = _fable()
        assert get_colors(card) == ["R"]

    def test_modal_dfc_uses_face_colors(self):
        """Modal DFC cards have colors on faces."""
        card = _modal_dfc()
        assert get_colors(card) == ["W"]

    def test_preserves_wubrg_order(self):
        """Colors should always be in WUBRG order regardless of input."""
        card = _fire_ice()
        # Even though card has ["R", "U"], output should be U then R? No — WUBRG order is W, U, B, R, G
        assert get_colors(card) == ["U", "R"]

    def test_no_faces_no_toplevel(self):
        """Card with no colors anywhere returns empty."""
        card = {"card_faces": [{"colors": None}, {"colors": None}], "colors": None}
        assert get_colors(card) == []


# ---------------------------------------------------------------------------
# Tests: combine_faces
# ---------------------------------------------------------------------------

class TestCombineFaces:
    def test_split_oracle_text(self):
        card = _fire_ice()
        result = combine_faces(card, "oracle_text")
        assert "Fire deals 2 damage" in result
        assert " // " in result
        assert "Tap target permanent" in result

    def test_split_mana_cost(self):
        card = _fire_ice()
        assert combine_faces(card, "mana_cost") == "{1}{R} // {1}{U}"

    def test_transform_mana_cost_skips_empty(self):
        """Transform back faces often have no mana cost — should not produce trailing ' // '."""
        card = _fable()
        assert combine_faces(card, "mana_cost") == "{2}{R}"

    def test_missing_field(self):
        card = _fire_ice()
        assert combine_faces(card, "nonexistent") == ""


# ---------------------------------------------------------------------------
# Tests: first_face_value
# ---------------------------------------------------------------------------

class TestFirstFaceValue:
    def test_adventure_power_from_creature_face(self):
        card = _bonecrusher_giant()
        assert first_face_value(card, "power") == "4"
        assert first_face_value(card, "toughness") == "3"

    def test_transform_power_from_back_face(self):
        """Fable front face has no p/t, back face does."""
        card = _fable()
        assert first_face_value(card, "power") == "2"
        assert first_face_value(card, "toughness") == "2"

    def test_missing_field_returns_none(self):
        card = _fire_ice()
        assert first_face_value(card, "loyalty") is None


# ---------------------------------------------------------------------------
# Tests: derive_primary_color
# ---------------------------------------------------------------------------

class TestDerivePrimaryColor:
    def test_single_color(self):
        assert derive_primary_color(["R"], ["R"]) == "R"

    def test_multicolor(self):
        assert derive_primary_color(["U", "R"], ["U", "R"]) == "Multicolor"

    def test_colorless(self):
        assert derive_primary_color([], []) == "Colorless"

    def test_land_falls_back_to_identity(self):
        """Lands have empty colors — should use color_identity."""
        assert derive_primary_color([], ["R"]) == "R"

    def test_dual_land_multicolor(self):
        assert derive_primary_color([], ["G", "U"]) == "Multicolor"


# ---------------------------------------------------------------------------
# Tests: derive_supertype
# ---------------------------------------------------------------------------

class TestDeriveSupertype:
    def test_creature(self):
        assert derive_supertype("Creature — Human Wizard") == "Creature"

    def test_artifact_creature_is_creature(self):
        """Creature takes priority over Artifact."""
        assert derive_supertype("Artifact Creature — Golem") == "Creature"

    def test_enchantment_creature_is_creature(self):
        assert derive_supertype("Enchantment Creature — God") == "Creature"

    def test_instant(self):
        assert derive_supertype("Instant") == "Instant"

    def test_sorcery(self):
        assert derive_supertype("Sorcery") == "Sorcery"

    def test_land(self):
        assert derive_supertype("Basic Land — Mountain") == "Land"

    def test_planeswalker(self):
        assert derive_supertype("Legendary Planeswalker — Jace") == "Planeswalker"

    def test_multi_face_uses_front_only(self):
        """Sorcery // Land should be Sorcery (front face)."""
        assert derive_supertype("Sorcery // Land") == "Sorcery"

    def test_enchantment_saga_front_creature_back(self):
        """Enchantment — Saga // Enchantment Creature — Goblin Shaman → Enchantment."""
        assert derive_supertype("Enchantment — Saga // Enchantment Creature — Goblin Shaman") == "Enchantment"

    def test_summon_is_creature(self):
        assert derive_supertype("Summon Dragon") == "Creature"

    def test_summon_licid_is_creature(self):
        assert derive_supertype("Summon Licid") == "Creature"

    def test_summon_dash_is_creature(self):
        assert derive_supertype("Summon — Specter") == "Creature"

    def test_lowercase_instant(self):
        """Un-card 'capital offense' has type line 'instant'."""
        assert derive_supertype("instant") == "Instant"

    def test_empty_type_line(self):
        assert derive_supertype("") == "Unknown"

    def test_none_type_line(self):
        assert derive_supertype(None) == "Unknown"

    def test_card_type_is_unknown(self):
        """Jumpstart theme cards have type 'Card' — genuinely Unknown."""
        assert derive_supertype("Card") == "Unknown"


# ---------------------------------------------------------------------------
# Tests: build_embedding_text
# ---------------------------------------------------------------------------

class TestBuildEmbeddingText:
    def test_full_card(self):
        result = build_embedding_text(
            "Lightning Bolt", "Instant",
            "Lightning Bolt deals 3 damage to any target.", ""
        )
        assert result == "Lightning Bolt. Instant. Lightning Bolt deals 3 damage to any target."

    def test_with_keywords(self):
        result = build_embedding_text(
            "Questing Beast", "Legendary Creature — Beast",
            "Vigilance, deathtouch, haste", "Vigilance, Deathtouch, Haste"
        )
        assert "Keywords: Vigilance, Deathtouch, Haste" in result

    def test_no_oracle_text(self):
        result = build_embedding_text("Mountain", "Basic Land — Mountain", "", "")
        assert result == "Mountain. Basic Land — Mountain"

    def test_name_only(self):
        result = build_embedding_text("Mystery", "", "", "")
        assert result == "Mystery"


# ---------------------------------------------------------------------------
# Tests: process_card (full integration)
# ---------------------------------------------------------------------------

class TestProcessCard:
    def test_lightning_bolt(self):
        row = process_card(_lightning_bolt())
        assert row["name"] == "Lightning Bolt"
        assert row["supertype"] == "Instant"
        assert row["primary_color"] == "R"
        assert row["colors"] == "R"
        assert row["mana_cost"] == "{R}"
        assert row["cmc"] == 1.0
        assert row["power"] is None
        assert row["loyalty"] is None
        assert row["legal_modern"] == "legal"
        assert row["legal_standard"] == "not_legal"
        assert "Lightning Bolt deals 3 damage" in row["embedding_text"]

    def test_fire_ice_split(self):
        row = process_card(_fire_ice())
        assert row["name"] == "Fire // Ice"
        assert row["supertype"] == "Instant"
        assert row["primary_color"] == "Multicolor"
        assert row["mana_cost"] == "{1}{R} // {1}{U}"
        assert " // " in row["oracle_text"]
        # Colors should be populated (bug fix)
        assert "U" in row["colors"]
        assert "R" in row["colors"]

    def test_bonecrusher_adventure(self):
        row = process_card(_bonecrusher_giant())
        assert row["name"] == "Bonecrusher Giant // Stomp"
        assert row["supertype"] == "Creature"
        assert row["primary_color"] == "R"
        assert row["power"] == "4"
        assert row["toughness"] == "3"
        # Colors should be populated (bug fix)
        assert row["colors"] == "R"

    def test_fable_transform(self):
        row = process_card(_fable())
        assert row["name"] == "Fable of the Mirror-Breaker // Reflection of Kiki-Jiki"
        assert row["supertype"] == "Enchantment"
        assert row["primary_color"] == "R"
        assert row["colors"] == "R"
        assert row["mana_cost"] == "{2}{R}"
        # Power/toughness from back face
        assert row["power"] == "2"
        assert row["toughness"] == "2"

    def test_jace_planeswalker(self):
        row = process_card(_jace())
        assert row["supertype"] == "Planeswalker"
        assert row["primary_color"] == "U"
        assert row["loyalty"] == "3"
        assert row["power"] is None
        assert row["legal_vintage"] == "restricted"

    def test_breeding_pool_dual_land(self):
        row = process_card(_breeding_pool())
        assert row["supertype"] == "Land"
        assert row["primary_color"] == "Multicolor"
        assert row["colors"] == ""  # Lands have no colors
        assert row["color_identity"] == "G, U"

    def test_mountain_basic_land(self):
        row = process_card(_mountain())
        assert row["supertype"] == "Land"
        assert row["primary_color"] == "R"
        assert row["colors"] == ""
        assert row["color_identity"] == "R"

    def test_modal_dfc(self):
        row = process_card(_modal_dfc())
        assert row["supertype"] == "Sorcery"
        assert row["primary_color"] == "W"
        assert row["colors"] == "W"
        assert " // " in row["oracle_text"]

    def test_newlines_stripped_from_oracle_text(self):
        """Oracle text newlines should be replaced with spaces for clean CSV rows."""
        card = _modal_dfc()
        # The back face has a \n in oracle_text
        row = process_card(card)
        assert "\n" not in row["oracle_text"]
        assert "{T}: Add {W}." in row["oracle_text"]

    def test_newlines_stripped_from_flavor_text(self):
        card = _lightning_bolt()
        card["flavor_text"] = "Line one.\nLine two."
        row = process_card(card)
        assert "\n" not in row["flavor_text"]
        assert row["flavor_text"] == "Line one. Line two."

    def test_newlines_stripped_from_embedding_text(self):
        """Embedding text is built from oracle_text, so newlines should be gone there too."""
        card = _modal_dfc()
        row = process_card(card)
        assert "\n" not in row["embedding_text"]

    def test_all_legality_columns_present(self):
        row = process_card(_lightning_bolt())
        for fmt in ["standard", "modern", "legacy", "vintage", "commander", "pioneer", "pauper", "historic"]:
            assert f"legal_{fmt}" in row

    def test_csv_column_count(self):
        """Every row should have exactly 33 columns (25 base + 8 legality)."""
        row = process_card(_lightning_bolt())
        assert len(row) == 33
