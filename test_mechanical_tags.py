"""Tests for mechanical tag extraction (mechanical_tags.py)."""

import numpy as np
import pandas as pd
import pytest

from mechanical_tags import encode_tags_multihot, tag_oracle_text


# ── tag_oracle_text: Trigger tags ──


def test_etb_enters_the_battlefield():
    text = "When Mulldrifter enters the battlefield, draw two cards."
    tags = tag_oracle_text(text)
    assert "etb" in tags
    assert "draw" in tags


def test_etb_enters_under_control():
    text = "Whenever a creature enters under your control, you gain 1 life."
    tags = tag_oracle_text(text)
    assert "etb" in tags
    assert "lifegain" in tags


def test_death_trigger():
    text = "Whenever another creature you control dies, each opponent loses 1 life."
    tags = tag_oracle_text(text)
    assert "death_trigger" in tags


def test_attack_trigger():
    text = "Whenever Etali, Primal Storm attacks, exile the top card of each player's library."
    tags = tag_oracle_text(text)
    assert "attack_trigger" in tags


def test_damage_trigger():
    text = "Whenever Niv-Mizzet, Parun deals damage to an opponent, draw a card."
    tags = tag_oracle_text(text)
    assert "damage_trigger" in tags
    assert "draw" in tags


def test_upkeep_trigger():
    text = "At the beginning of your upkeep, create a 1/1 white Spirit creature token with flying."
    tags = tag_oracle_text(text)
    assert "upkeep_trigger" in tags
    assert "tokens" in tags


# ── tag_oracle_text: Effect tags ──


def test_sacrifice():
    text = "Sacrifice a creature: Scry 1."
    tags = tag_oracle_text(text)
    assert "sacrifice" in tags


def test_draw():
    text = "Draw three cards."
    tags = tag_oracle_text(text)
    assert "draw" in tags


def test_removal_destroy():
    text = "Destroy target creature."
    tags = tag_oracle_text(text)
    assert "removal" in tags


def test_removal_exile():
    text = "Exile target artifact or enchantment."
    tags = tag_oracle_text(text)
    assert "removal" in tags


def test_removal_buff_not_tagged():
    """A +1/+1 buff should NOT be tagged as removal."""
    text = "Target creature gets +1/+1 until end of turn."
    tags = tag_oracle_text(text)
    assert "removal" not in tags


def test_removal_debuff_tagged():
    """A -3/-3 debuff should be tagged as removal."""
    text = "Target creature gets -3/-3 until end of turn."
    tags = tag_oracle_text(text)
    assert "removal" in tags


def test_removal_mixed_negative_toughness():
    """A +3/-3 effect should be tagged as removal (kills via negative toughness)."""
    text = "Target creature gets +3/-3 until end of turn."
    tags = tag_oracle_text(text)
    assert "removal" in tags


def test_bounce():
    text = "Return target creature to its owner's hand."
    tags = tag_oracle_text(text)
    assert "bounce" in tags


def test_counterspell():
    text = "Counter target spell."
    tags = tag_oracle_text(text)
    assert "counterspell" in tags


def test_blink():
    text = "Exile target creature you control, then return it to the battlefield under your control."
    tags = tag_oracle_text(text)
    assert "blink" in tags


def test_reanimate():
    text = "Return target creature card from your graveyard to the battlefield."
    tags = tag_oracle_text(text)
    assert "reanimate" in tags


def test_tutor():
    text = "Search your library for a card, put it into your hand, then shuffle."
    tags = tag_oracle_text(text)
    assert "tutor" in tags


# ── tag_oracle_text: Generator tags ──


def test_tokens():
    text = "Create two 1/1 white Soldier creature tokens."
    tags = tag_oracle_text(text)
    assert "tokens" in tags


def test_counters_plus():
    text = "Put a +1/+1 counter on target creature."
    tags = tag_oracle_text(text)
    assert "counters_plus" in tags


def test_counters_minus():
    text = "Put a -1/-1 counter on target creature."
    tags = tag_oracle_text(text)
    assert "counters_minus" in tags


def test_ramp():
    text = "Search your library for a basic land card and put it onto the battlefield tapped."
    tags = tag_oracle_text(text)
    assert "ramp" in tags


def test_lifegain():
    text = "You gain 3 life."
    tags = tag_oracle_text(text)
    assert "lifegain" in tags


def test_mill():
    text = "Target player mills four cards."
    tags = tag_oracle_text(text)
    assert "mill" in tags


# ── tag_oracle_text: Modifier tags ──


def test_anthem():
    text = "Other creatures you control get +1/+1."
    tags = tag_oracle_text(text)
    assert "anthem" in tags


def test_cost_reduction():
    text = "Creature spells you cast cost {1} less to cast."
    tags = tag_oracle_text(text)
    assert "cost_reduction" in tags


def test_copy():
    text = "Copy target instant or sorcery spell. You may choose new targets for the copy."
    tags = tag_oracle_text(text)
    assert "copy" in tags


def test_protection_hexproof():
    text = "Hexproof"
    tags = tag_oracle_text(text)
    assert "protection" in tags


def test_evasion_flying():
    text = "Flying"
    tags = tag_oracle_text(text)
    assert "evasion_flying" in tags


def test_evasion_trample():
    text = "Trample"
    tags = tag_oracle_text(text)
    assert "evasion_trample" in tags


def test_evasion_menace():
    text = "Menace"
    tags = tag_oracle_text(text)
    assert "evasion_menace" in tags


def test_evasion_unblockable():
    text = "This creature can't be blocked."
    tags = tag_oracle_text(text)
    assert "evasion_unblockable" in tags


def test_old_evasion_tag_gone():
    """The old combined 'evasion' tag should no longer exist."""
    text = "Flying, trample, menace"
    tags = tag_oracle_text(text)
    assert "evasion" not in tags


# ── tag_oracle_text: Permanent tags ──


def test_equipment():
    text = "Equipped creature gets +2/+2. Equip {2}"
    tags = tag_oracle_text(text)
    assert "equipment" in tags


def test_aura():
    text = "Enchant creature. Enchanted creature gets +1/+1."
    tags = tag_oracle_text(text)
    assert "aura" in tags


def test_tap_ability():
    text = "{T}: Add {G}."
    tags = tag_oracle_text(text)
    assert "tap_ability" in tags
    assert "ramp" in tags


# ── tag_oracle_text: Other tags ──


def test_graveyard_matters():
    text = "Flashback {2}{B}"
    tags = tag_oracle_text(text)
    assert "graveyard_matters" in tags


def test_storm():
    text = "Storm (When you cast this spell, copy it for each spell cast before it this turn.)"
    tags = tag_oracle_text(text)
    assert "storm" in tags


# ── Edge cases ──


def test_empty_text():
    assert tag_oracle_text("") == []
    assert tag_oracle_text(None) == []


def test_vanilla_creature():
    """A vanilla creature with no oracle text should get no tags."""
    tags = tag_oracle_text("")
    assert tags == []


def test_tags_are_sorted():
    text = "When this enters the battlefield, draw a card. Flying."
    tags = tag_oracle_text(text)
    assert tags == sorted(tags)


def test_multiple_tags():
    """Mulldrifter-like card should get ETB + draw."""
    text = "When Mulldrifter enters the battlefield, draw two cards. Evoke {2}{U}"
    tags = tag_oracle_text(text)
    assert "etb" in tags
    assert "draw" in tags


def test_viscera_seer_like():
    """Sac outlet + scry."""
    text = "Sacrifice a creature: Scry 1."
    tags = tag_oracle_text(text)
    assert "sacrifice" in tags


# ── encode_tags_multihot ──


def test_encode_tags_multihot_basic():
    df = pd.DataFrame({
        "mechanical_tags": ["etb, draw", "sacrifice", "", "tokens, anthem"]
    })
    tag_names = ["anthem", "draw", "etb", "sacrifice", "tokens"]
    result = encode_tags_multihot(df, tag_names)

    assert result.shape == (4, 5)
    assert result.dtype == np.float32

    # Row 0: etb + draw
    assert result[0, 2] == 1.0  # etb
    assert result[0, 1] == 1.0  # draw
    assert result[0, 0] == 0.0  # anthem

    # Row 1: sacrifice
    assert result[1, 3] == 1.0  # sacrifice

    # Row 2: empty
    assert result[2].sum() == 0.0

    # Row 3: tokens + anthem
    assert result[3, 0] == 1.0  # anthem
    assert result[3, 4] == 1.0  # tokens


def test_encode_tags_multihot_handles_nan():
    df = pd.DataFrame({"mechanical_tags": [None, "etb"]})
    tag_names = ["etb"]
    result = encode_tags_multihot(df, tag_names)
    assert result[0, 0] == 0.0
    assert result[1, 0] == 1.0
