"""Tests for the deterministic deck builder (pilot/build_deck.py)."""

import json

import numpy as np
import pandas as pd
import pytest

from manamap.config import DECK_ROLE_BUDGET, DECK_SIZE
from manamap.pilot.build_deck import (
    BriefError,
    candidate_pool,
    castability,
    commander_identity,
    curve_fit,
    decklist_name,
    decklist_text,
    edhrec_component,
    enforce_bracket,
    fill_slots,
    role_group,
    synergy_affinity,
)


def _row(name, type_line="Legendary Creature — Snake", ci="B, G", legal="legal", text=""):
    return {"name": name, "type_line": type_line, "color_identity": ci,
            "legal_commander": legal, "oracle_text": text}


# ── commander legality ──


def test_legendary_creature_is_a_legal_commander():
    assert commander_identity(_row("Hapatra")) == {"B", "G"}


def test_nonlegendary_creature_is_rejected():
    with pytest.raises(BriefError, match="not a legal commander"):
        commander_identity(_row("Grizzly Bears", "Creature — Bear"))


def test_planeswalker_with_the_magic_words_is_accepted():
    row = _row("Daretti", "Legendary Planeswalker — Daretti", "R",
               text="Daretti can be your commander.")
    assert commander_identity(row) == {"R"}


def test_planeswalker_without_the_magic_words_is_rejected():
    with pytest.raises(BriefError):
        commander_identity(_row("Jace", "Legendary Planeswalker — Jace", "U"))


def test_banned_commander_is_rejected():
    with pytest.raises(BriefError, match="not legal"):
        commander_identity(_row("Golos", legal="banned"))


def test_colorless_commander_has_empty_identity():
    assert commander_identity(_row("Kozilek", "Legendary Creature — Eldrazi", "")) == set()


# ── the pool filter is where the tier guarantee starts ──


def _pool_df():
    return pd.DataFrame([
        {"name": "In Identity", "color_identity": "B", "legal_commander": "legal",
         "game_changer": False, "supertype": "Creature", "type_line": "Creature"},
        {"name": "Off Identity", "color_identity": "U", "legal_commander": "legal",
         "game_changer": False, "supertype": "Creature", "type_line": "Creature"},
        {"name": "Colorless", "color_identity": "", "legal_commander": "legal",
         "game_changer": False, "supertype": "Artifact", "type_line": "Artifact"},
        {"name": "Demonic Tutor", "color_identity": "B", "legal_commander": "legal",
         "game_changer": True, "supertype": "Sorcery", "type_line": "Sorcery"},
        {"name": "Banned Thing", "color_identity": "B", "legal_commander": "banned",
         "game_changer": False, "supertype": "Sorcery", "type_line": "Sorcery"},
        {"name": "Armageddon", "color_identity": "W", "legal_commander": "legal",
         "game_changer": False, "supertype": "Sorcery", "type_line": "Sorcery"},
    ])


def _brief(**kwargs):
    base = {"commander": "Hapatra", "bracket": 3, "must_include": [], "must_exclude": []}
    base.update(kwargs)
    return base


def test_pool_excludes_game_changers_at_low_brackets():
    """A Bracket 2 build never *sees* a Game Changer, so it can't be argued into one."""
    pool = candidate_pool(_pool_df(), {"B", "G"}, 2, _brief())
    assert "Demonic Tutor" not in set(pool["name"])


def test_pool_allows_game_changers_at_bracket_three():
    pool = candidate_pool(_pool_df(), {"B", "G"}, 3, _brief())
    assert "Demonic Tutor" in set(pool["name"])


def test_pool_enforces_color_identity():
    pool = candidate_pool(_pool_df(), {"B", "G"}, 3, _brief())
    names = set(pool["name"])
    assert "In Identity" in names
    assert "Colorless" in names
    assert "Off Identity" not in names


def test_pool_excludes_banned_cards():
    assert "Banned Thing" not in set(candidate_pool(_pool_df(), {"B", "G"}, 3, _brief())["name"])


def test_pool_excludes_mass_land_denial_below_bracket_four():
    pool = candidate_pool(_pool_df(), set("WUBRG"), 3, _brief())
    assert "Armageddon" not in set(pool["name"])


def test_pool_allows_mass_land_denial_at_bracket_four():
    pool = candidate_pool(_pool_df(), set("WUBRG"), 4, _brief())
    assert "Armageddon" in set(pool["name"])


def test_pool_honours_must_exclude():
    pool = candidate_pool(_pool_df(), {"B", "G"}, 3, _brief(must_exclude=["In Identity"]))
    assert "In Identity" not in set(pool["name"])


def test_pool_never_contains_the_commander():
    df = _pool_df()
    df.loc[len(df)] = {"name": "Hapatra", "color_identity": "B, G",
                       "legal_commander": "legal", "game_changer": False,
                       "supertype": "Creature", "type_line": "Legendary Creature"}
    assert "Hapatra" not in set(candidate_pool(df, {"B", "G"}, 3, _brief())["name"])


# ── scoring components ──


def test_curve_fit_prefers_the_sweet_spot():
    assert curve_fit(2) > curve_fit(7)


def test_curve_fit_floors_at_zero_for_absurd_costs():
    assert curve_fit(20) == 0.0


def test_curve_fit_handles_missing_cmc():
    assert curve_fit(None) == 0.5


def test_castability_penalises_pip_intensity():
    """Colour legality is a pool filter; what varies is how heavy the cost is."""
    assert castability("{2}{B}") == 1.0
    assert castability("{B}{B}") < castability("{2}{B}")
    assert castability("{B}{B}{B}") < castability("{B}{B}")


def test_castability_of_colourless():
    assert castability("{4}") == 1.0


def test_edhrec_is_log_scaled_not_linear():
    """A linear rank transform says rank 1 and rank 100 are nearly identical."""
    top = edhrec_component(1, 30000)
    near = edhrec_component(100, 30000)
    mid = edhrec_component(15000, 30000)
    assert top > near > mid
    assert (top - near) > (near - mid) * 0.5


def test_edhrec_missing_rank_is_neutral():
    assert edhrec_component(None, 30000) == 0.3


def test_synergy_affinity_scores_against_the_whole_deck():
    """The graph is a top-10 shortlist; the rules apply to the deck's tag union."""
    assert synergy_affinity({"blink"}, {"etb"}) > 0
    assert synergy_affinity({"blink"}, {"storm"}) == 0


def test_synergy_affinity_is_bidirectional():
    assert synergy_affinity({"etb"}, {"blink"}) == synergy_affinity({"blink"}, {"etb"})


# ── role grouping ──


def test_role_group_maps_specific_roles():
    assert role_group(["ramp:rock"]) == "ramp"
    assert role_group(["removal:sweeper"]) == "sweeper"
    assert role_group(["counterspell"]) == "removal"


def test_role_group_falls_back_to_flex():
    assert role_group(["threat:body"]) == "flex"
    assert role_group([]) == "flex"


# ── slot filling ──


def _scored(names):
    return [{"name": n, "score": 1.0 - i * 0.01, "components": {}}
            for i, n in enumerate(names)]


def test_fill_slots_respects_the_budget():
    scored = _scored([f"C{i}" for i in range(200)])
    roles = {f"C{i}": ["ramp:rock"] for i in range(200)}
    budget = {"lands": 36, "ramp": 5, "flex": 3}
    slots, _ = fill_slots(scored, roles, budget, [])
    assert sum(1 for s in slots if s["role"] == "ramp") == 5
    assert len(slots) == 8


def test_fill_slots_never_repeats_a_card():
    scored = _scored([f"C{i}" for i in range(50)])
    roles = {f"C{i}": ["ramp:rock"] for i in range(50)}
    slots, _ = fill_slots(scored, roles, {"lands": 36, "ramp": 5, "flex": 5}, [])
    names = [s["name"] for s in slots]
    assert len(names) == len(set(names))


def test_fill_slots_honours_must_include_regardless_of_score():
    scored = _scored([f"C{i}" for i in range(50)])
    roles = {f"C{i}": ["ramp:rock"] for i in range(50)}
    slots, _ = fill_slots(scored, roles, {"lands": 36, "ramp": 2}, ["C49"])
    assert any(s["name"] == "C49" and s["reason"] == "must_include" for s in slots)


def test_fill_slots_keeps_alternates():
    scored = _scored([f"C{i}" for i in range(20)])
    roles = {f"C{i}": ["ramp:rock"] for i in range(20)}
    slots, _ = fill_slots(scored, roles, {"lands": 36, "ramp": 3}, [])
    ramp = [s for s in slots if s["role"] == "ramp"]
    assert all(s["alternates"] for s in ramp)
    assert all("delta" in alt for s in ramp for alt in s["alternates"])


def test_fill_slots_alternate_deltas_are_non_negative():
    scored = _scored([f"C{i}" for i in range(20)])
    roles = {f"C{i}": ["ramp:rock"] for i in range(20)}
    slots, _ = fill_slots(scored, roles, {"lands": 36, "ramp": 2}, [])
    assert all(alt["delta"] >= 0 for s in slots for alt in s["alternates"])


def test_fill_slots_is_deterministic():
    scored = _scored([f"C{i}" for i in range(30)])
    roles = {f"C{i}": ["ramp:rock"] for i in range(30)}
    budget = {"lands": 36, "ramp": 4, "flex": 4}
    first, _ = fill_slots(scored, roles, budget, [])
    second, _ = fill_slots(scored, roles, budget, [])
    assert [s["name"] for s in first] == [s["name"] for s in second]


# ── the emergent-combo pass ──


def _details(combos):
    by_card = {}
    for i, combo in enumerate(combos):
        for name in combo["cards"]:
            by_card.setdefault(name, []).append(i)
    return {"combos": combos, "by_card": by_card}


def _combo(cards, bracket=1, produces=("Infinite mana",), mv=2):
    return {"cards": list(cards), "produces": list(produces), "ci": "",
            "bracket": bracket, "mana_value_needed": mv, "popularity": 0}


def _flags(names):
    return {n: {"game_changer": False, "legal_commander": "legal"} for n in names}


def test_enforce_bracket_leaves_a_compliant_deck_alone():
    slots = [{"name": "A", "role": "flex", "score": 1.0, "alternates": []}]
    slots, cut, report = enforce_bracket(
        slots, _scored(["A"]), {}, _flags(["A"]), _details([]), {"CMD"}, 2)
    assert cut == []
    assert report["floor"] == 1


def test_enforce_bracket_cuts_an_emergent_two_card_infinite():
    """Two cards each individually fine can combine into a line that isn't."""
    slots = [
        {"name": "A", "role": "flex", "score": 0.9, "alternates": []},
        {"name": "B", "role": "flex", "score": 0.5, "alternates": []},
    ]
    scored = _scored(["A", "B", "REPLACEMENT"])
    details = _details([_combo(["A", "B"], bracket=4)])
    slots, cut, report = enforce_bracket(
        slots, scored, {}, _flags(["A", "B", "REPLACEMENT"]), details, {"CMD"}, 2)
    assert cut, "expected a cut"
    assert report["floor"] <= 2
    assert cut[0]["reasons"]


def test_enforce_bracket_records_what_it_replaced_with():
    slots = [
        {"name": "A", "role": "flex", "score": 0.9, "alternates": []},
        {"name": "B", "role": "flex", "score": 0.1, "alternates": []},
    ]
    scored = _scored(["A", "B", "SAFE"])
    details = _details([_combo(["A", "B"], bracket=4)])
    _, cut, _ = enforce_bracket(
        slots, scored, {}, _flags(["A", "B", "SAFE"]), details, {"CMD"}, 2)
    assert cut[0]["replaced_by"] == "SAFE"


def test_enforce_bracket_cuts_even_with_no_replacement_available():
    """Better an empty slot than an off-bracket deck."""
    slots = [{"name": "Armageddon", "role": "flex", "score": 0.5, "alternates": []}]
    slots, cut, report = enforce_bracket(
        slots, _scored(["Armageddon"]), {}, _flags(["Armageddon"]), _details([]), {"CMD"}, 2)
    assert cut[0]["name"] == "Armageddon"
    assert cut[0]["replaced_by"] is None
    assert report["floor"] <= 2


def test_enforce_bracket_fails_loudly_rather_than_shipping_off_bracket():
    """Every replacement reintroduces the violation — bail, don't ship it."""
    denial = ["Armageddon", "Jokulhaups", "Obliterate", "Wildfire", "Ruination",
              "Catastrophe", "Sunder", "Cataclysm", "Global Ruin", "Devastation",
              "Epicenter", "Worldfire"]
    slots = [{"name": denial[0], "role": "flex", "score": 0.5, "alternates": []}]
    with pytest.raises(BriefError, match="could not reach bracket"):
        enforce_bracket(
            slots, _scored(denial), {}, _flags(denial), _details([]), {"CMD"}, 2)


# ── decklist rendering ──


def test_decklist_name_uses_the_front_face_for_transform_cards():
    """Scryfall's collection endpoint rejects the joined form for DFCs."""
    assert decklist_name("Mosswood Dreadknight // Dread Whispers", "adventure") == \
        "Mosswood Dreadknight"


def test_decklist_name_keeps_the_full_name_for_split_cards():
    assert decklist_name("Fire // Ice", "split") == "Fire // Ice"


def test_decklist_name_passes_single_faced_cards_through():
    assert decklist_name("Sol Ring", "normal") == "Sol Ring"


def test_decklist_text_marks_the_commander():
    plan = {"commander": "Hapatra", "slots": [{"name": "Sol Ring"}],
            "land_counts": {"Swamp": 2}}
    text = decklist_text(plan)
    assert text.splitlines()[0] == "1 Hapatra *CMDR*"
    assert "2 Swamp" in text


def test_decklist_text_is_deterministic():
    plan = {"commander": "Hapatra", "slots": [{"name": "B"}, {"name": "A"}],
            "land_counts": {"Swamp": 1}}
    assert decklist_text(plan) == decklist_text(plan)


# ── budget sanity ──


def test_role_budget_sums_to_the_deck_size():
    """36 lands + 63 spells + 1 commander = 100."""
    assert sum(DECK_ROLE_BUDGET.values()) == DECK_SIZE - 1
