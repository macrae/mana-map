"""Tests for the deterministic deck builder (pilot/build_deck.py)."""


import collections
import json

import pandas as pd
import pytest

from conftest import requires_data, requires_deck

from manamap.config import DECK_ROLE_BUDGET, DECK_SIZE, DECKS_DIR
from manamap.pilot import build_deck
from manamap.pilot.card_pool import load_pool
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
    slots, _, _budget = fill_slots(scored, roles, budget, [])
    assert sum(1 for s in slots if s["role"] == "ramp") == 5
    assert len(slots) == 8


def test_fill_slots_never_repeats_a_card():
    scored = _scored([f"C{i}" for i in range(50)])
    roles = {f"C{i}": ["ramp:rock"] for i in range(50)}
    slots, _, _budget = fill_slots(scored, roles, {"lands": 36, "ramp": 5, "flex": 5}, [])
    names = [s["name"] for s in slots]
    assert len(names) == len(set(names))


def test_fill_slots_honours_must_include_regardless_of_score():
    scored = _scored([f"C{i}" for i in range(50)])
    roles = {f"C{i}": ["ramp:rock"] for i in range(50)}
    slots, _, _budget = fill_slots(scored, roles, {"lands": 36, "ramp": 2}, ["C49"])
    assert any(s["name"] == "C49" and s["reason"] == "must_include" for s in slots)


def test_fill_slots_keeps_alternates():
    scored = _scored([f"C{i}" for i in range(20)])
    roles = {f"C{i}": ["ramp:rock"] for i in range(20)}
    slots, _, _budget = fill_slots(scored, roles, {"lands": 36, "ramp": 3}, [])
    ramp = [s for s in slots if s["role"] == "ramp"]
    assert all(s["alternates"] for s in ramp)
    assert all("delta" in alt for s in ramp for alt in s["alternates"])


def test_fill_slots_alternate_deltas_are_non_negative():
    scored = _scored([f"C{i}" for i in range(20)])
    roles = {f"C{i}": ["ramp:rock"] for i in range(20)}
    slots, _, _budget = fill_slots(scored, roles, {"lands": 36, "ramp": 2}, [])
    assert all(alt["delta"] >= 0 for s in slots for alt in s["alternates"])


def test_fill_slots_is_deterministic():
    scored = _scored([f"C{i}" for i in range(30)])
    roles = {f"C{i}": ["ramp:rock"] for i in range(30)}
    budget = {"lands": 36, "ramp": 4, "flex": 4}
    first, _, _b1 = fill_slots(scored, roles, budget, [])
    second, _, _b2 = fill_slots(scored, roles, budget, [])
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


# ── Sideboard preservation ───────────────────────────────────────────────
#
# decklist_text() renders commander + slots + lands from the plan, so rewriting
# decklist.txt used to erase any hand-authored sideboard. Harmless while no built
# deck had one; a live hazard the moment one does.

SIDEBOARD_BLOCK = (
    "SIDEBOARD:\n"
    "1 Sazacap's Brew (PLST) BLB-151\n"
    "1 Red Mana (SLD) 7082 *F*"
)

TOY_PLAN = {
    "commander": "Zada, Hedron Grinder",
    "slots": [{"name": "Sol Ring"}],
    "land_counts": {"Mountain": 2},
}


# ── Building from a physical collection, not the format ─────────────────


def _pool_brief(**kw):
    brief = {"commander": "Hapatra, Vizier of Poisons", "must_include": [], "must_exclude": []}
    brief.update(kw)
    return brief


def test_no_pool_declared_means_the_whole_format():
    assert build_deck.resolve_pool(_pool_brief()) is None


def test_an_explicit_name_list_becomes_the_pool():
    pool = build_deck.resolve_pool(
        _pool_brief(pool=["Hapatra, Vizier of Poisons", "Blood Artist"])
    )
    assert pool == {"Hapatra, Vizier of Poisons", "Blood Artist"}


def test_a_pool_without_the_commander_is_rejected():
    """Building a deck you cannot lead is worse than not building it."""
    with pytest.raises(BriefError, match="does not contain the commander"):
        build_deck.resolve_pool(_pool_brief(pool=["Blood Artist"]))


def test_must_include_outside_the_pool_is_rejected():
    with pytest.raises(BriefError, match="must_include names outside the pool"):
        build_deck.resolve_pool(_pool_brief(
            pool=["Hapatra, Vizier of Poisons"], must_include=["Black Lotus"],
        ))


# Reads the POOL through `card_pool`, which is the only reader of the
# gitignored `cards.csv` — so this needs the corpus, not just a deck. It was
# ungated and failed on a fresh clone.
@requires_data
def test_pool_files_are_parsed_by_the_same_reader_pool_facts_uses(tmp_path):
    """A box analysed and a box built from must never disagree about its contents."""
    lst = tmp_path / "box.txt"
    lst.write_text("1 Hapatra, Vizier of Poisons (ECC) 123\n1 Blood Artist *F*\n")
    pool = build_deck.resolve_pool(_pool_brief(pool_files=[str(lst)]))
    assert pool == {"Hapatra, Vizier of Poisons", "Blood Artist"}


@requires_data
def test_an_unresolvable_pool_card_is_rejected_not_ignored(tmp_path):
    lst = tmp_path / "box.txt"
    lst.write_text("1 Hapatra, Vizier of Poisons\n1 Not A Real Card\n")
    with pytest.raises(BriefError, match="did not resolve"):
        build_deck.resolve_pool(_pool_brief(pool_files=[str(lst)]))


def test_candidate_pool_is_restricted_to_the_declared_collection():
    df = pd.DataFrame([
        {"name": "Blood Artist", "type_line": "Creature — Vampire", "color_identity": "B",
         "legal_commander": "legal", "game_changer": False, "supertype": "Creature"},
        {"name": "Zulaport Cutthroat", "type_line": "Creature — Human", "color_identity": "B",
         "legal_commander": "legal", "game_changer": False, "supertype": "Creature"},
    ])
    brief = _pool_brief()
    brief["_pool"] = {"Hapatra, Vizier of Poisons", "Blood Artist"}
    pool = candidate_pool(df, {"B", "G"}, 4, brief)
    assert sorted(pool["name"]) == ["Blood Artist"]


def test_candidate_pool_without_a_pool_key_keeps_every_legal_card():
    df = pd.DataFrame([
        {"name": "Blood Artist", "type_line": "Creature — Vampire", "color_identity": "B",
         "legal_commander": "legal", "game_changer": False, "supertype": "Creature"},
        {"name": "Zulaport Cutthroat", "type_line": "Creature — Human", "color_identity": "B",
         "legal_commander": "legal", "game_changer": False, "supertype": "Creature"},
    ])
    pool = candidate_pool(df, {"B", "G"}, 4, _pool_brief())
    assert len(pool) == 2


# ── must_include overflow: the 101-card plan ────────────────────────────


def test_must_include_overflow_is_charged_to_flex():
    """Pinning past a role's allowance must not grow the deck.

    Must-includes are pinned before any budget line is consulted, and the
    per-group `max(0, count - filled)` only stops the builder ADDING more — it
    never gives the slot back. A brief pinning 23 cards put `wincon` at 4
    against a budget of 3 and produced a 101-card plan.
    """
    roles = {"W1": ["wincon:drain"], "W2": ["wincon:drain"], "F1": [], "F2": [], "F3": []}
    scored = [{"name": n, "score": 1.0, "components": {}} for n in ["F1", "F2", "F3"]]
    budget = {"lands": 36, "wincon": 1, "flex": 3}
    slots, _, effective = fill_slots(scored, roles, budget, ["W1", "W2"])

    assert len(slots) == 4                    # not 5: one wincon over, one flex fewer
    assert effective["wincon"] == 2           # what was actually filled
    assert effective["flex"] == 2             # the overflow is charged here
    assert sum(v for k, v in effective.items() if k != "lands") == len(slots)


def test_effective_budget_matches_the_slots_it_describes():
    """`validate_build._validate_budget` compares them; they must agree."""
    roles = {"A": ["ramp:rock"], "B": []}
    scored = [{"name": n, "score": 1.0, "components": {}} for n in ["A", "B"]]
    budget = {"lands": 36, "ramp": 1, "flex": 1}
    slots, _, effective = fill_slots(scored, roles, budget, [])
    counts = {}
    for s in slots:
        counts[s["role"]] = counts.get(s["role"], 0) + 1
    for group, want in effective.items():
        if group == "lands":
            continue
        assert counts.get(group, 0) == want, group


def test_no_overflow_leaves_the_budget_untouched():
    roles = {"A": ["ramp:rock"], "B": []}
    scored = [{"name": n, "score": 1.0, "components": {}} for n in ["A", "B"]]
    budget = {"lands": 36, "ramp": 1, "flex": 1}
    _, _, effective = fill_slots(scored, roles, budget, [])
    assert effective == budget


def test_a_group_that_runs_dry_gives_its_slots_to_flex():
    """The other half of the wrong-size-plan bug, and it shipped a 99.

    Mono-black held only 2 cards the taxonomy calls `sweeper` against a budget of
    3. The group under-filled and nothing gave the slot back, so the plan came
    out one card short. Flex is filled last precisely so it can absorb this.
    """
    roles = {"S1": ["removal:sweeper"], "F1": [], "F2": [], "F3": [], "F4": []}
    scored = [{"name": n, "score": 1.0, "components": {}}
              for n in ["S1", "F1", "F2", "F3", "F4"]]
    budget = {"lands": 36, "sweeper": 3, "flex": 2}
    slots, _, effective = fill_slots(scored, roles, budget, [])

    assert effective["sweeper"] == 1          # only one candidate existed
    assert effective["flex"] == 4             # 2 budgeted + 2 the sweeper line could not use
    assert len(slots) == 5                    # the non-land total is still 3 + 2
    assert sum(v for k, v in effective.items() if k != "lands") == len(slots)


def test_overflow_and_shortfall_together_still_hit_the_total():
    """Both failure modes at once must still land on the non-land total."""
    roles = {"W1": ["wincon:drain"], "W2": ["wincon:drain"], "S1": ["removal:sweeper"],
             "F1": [], "F2": [], "F3": []}
    scored = [{"name": n, "score": 1.0, "components": {}} for n in ["S1", "F1", "F2", "F3"]]
    budget = {"lands": 36, "wincon": 1, "sweeper": 3, "flex": 2}
    slots, _, effective = fill_slots(scored, roles, budget, ["W1", "W2"])

    assert effective["wincon"] == 2           # pinned over budget
    assert effective["sweeper"] == 1          # ran dry
    assert sum(v for k, v in effective.items() if k != "lands") == len(slots)
    assert len(slots) == 6                    # 1 + 3 + 2 non-land slots


# ── A deck built from a collection names the PHYSICAL card ───────────────


def test_decklist_renders_the_owned_printing():
    """Bare names let Scryfall pick a default for every card.

    A Sol Ring built from a pool that holds ECC #58 came back as Marvel Super
    Heroes Commander #211 — and Featured Artist credits artists per printing, so
    the manual credited someone who painted none of the pilot's cards.
    """
    plan = {
        "commander": "Yawgmoth, Thran Physician",
        "slots": [{"name": "Sol Ring"}, {"name": "Blowfly Infestation"}],
        "land_counts": {"Swamp": 3},
        "printings": {
            "Yawgmoth, Thran Physician": {"set": "dmr", "collector_number": "110", "foil": False},
            "Sol Ring": {"set": "ecc", "collector_number": "58", "foil": False},
            "Swamp": {"set": "ecl", "collector_number": "271", "foil": False},
        },
    }
    text = decklist_text(plan)
    assert "1 Yawgmoth, Thran Physician (DMR) 110 *CMDR*" in text
    assert "1 Sol Ring (ECC) 58" in text
    assert "3 Swamp (ECL) 271" in text
    # No printing known for this one — it must still render, just bare.
    assert "1 Blowfly Infestation\n" in text


def test_foil_marker_survives_the_round_trip():
    """`*F*` must come off before the $-anchored printing regex — the parser's
    own hazard, and the writer has to emit it in an order the parser accepts."""
    from manamap.pilot.fetch_deck import parse_decklist

    plan = {
        "commander": "Zada, Hedron Grinder",
        "slots": [], "land_counts": {},
        "printings": {"Zada, Hedron Grinder": {"set": "sld", "collector_number": "2406", "foil": True}},
    }
    text = decklist_text(plan)
    assert "(SLD) 2406 *F*" in text
    entry = parse_decklist(text)[0]
    assert entry["set"] == "sld"
    assert entry["collector_number"] == "2406"
    assert entry["foil"] is True
    assert entry["is_commander"] is True


def test_no_pool_means_no_printings_and_bare_names():
    plan = {"commander": "Sol Ring", "slots": [], "land_counts": {}}
    assert decklist_text(plan).strip() == "1 Sol Ring *CMDR*"


# ── The output, not the parts ────────────────────────────────────────────────
#
# Everything above tests a pure function in isolation or a synthetic `_scored`
# list. `build()` was never called in this file at all, which is how a deck with
# NOTHING above mana value 3 and zero combo completions passed every test and
# `validate_build` too. These assert on the shape of a real deck.

def test_curve_targets_follow_the_cited_axis_and_total_exactly():
    """The target is `DECK_AXIS_TARGETS["curve"]`'s quote — modal two, ~15 two-drops
    and ~15 three-drops, ~1.5 at mana value 8+ — so the builder adds no new uncited
    constant. `deck_audit` has always measured against it; the builder never read it.
    """
    t = build_deck.curve_targets(64)
    assert sum(t.values()) == 64, "rounding remainder must be handed back, not lost"
    assert t[2] == max(t.values()), "the cited mode is two"
    assert t[2] >= 14 and t[3] >= 14, "~15.7 two-drops and ~15.4 three-drops"
    assert t[8] <= 2, "'only ~1.5 cards at mana value 8+'"
    assert sum(build_deck.curve_targets(40).values()) == 40, "shares scale"


def test_curve_bucket_folds_the_tail_at_eight():
    assert build_deck.curve_bucket(0) == 0 and build_deck.curve_bucket(3) == 3
    assert build_deck.curve_bucket(12) == 8, "8 means 8-or-more, matching the quote"
    assert build_deck.curve_bucket(None) == 0


@requires_data
@requires_deck
def test_a_built_deck_has_a_TOP_END(tmp_path):
    """The defect this phase exists for.

    The first kinnan baseline was 64 nonland cards with curve {0:1, 1:11, 2:28, 3:24}
    — nothing above mana value 3 — and 29 of them mana producers: a legal deck that
    ramps into nothing. `validate_build` passed it (100 cards, singleton, legal, floor
    within target, role counts exact), because it checks form and this is substance.
    Cause: cards are scored INDEPENDENTLY and the top N taken, while `curve_fit` is a
    flat plateau to 3 then a monotone decline — so the top N are always cheap. N
    independent maxima are not a distribution.
    """
    plan = json.loads((DECKS_DIR / "kinnan" / "build_plan.json").read_text())
    pool = load_pool()
    curve = collections.Counter()
    for slot in plan["slots"]:
        row = pool.get(slot["name"])
        if row is not None:
            curve[build_deck.curve_bucket(row["cmc"])] += 1
    assert sum(n for mv, n in curve.items() if mv >= 4) >= 15, \
        f"a deck with no top end ramps into nothing: {dict(sorted(curve.items()))}"
    # The quota is crossed with the ROLE quota, and a role can run out of cards in
    # a bucket — so the realised mode lands at 2 or 3 rather than exactly on the
    # cited 2. Measured here: 2:15 against 3:16, one card apart. Asserting exactness
    # would be asserting that no role ever runs dry, which is not true and not the
    # property that matters.
    mode = max(curve, key=lambda mv: (curve[mv], -mv))
    assert mode in (2, 3), f"the cheap end should still be the bulk: {dict(curve)}"
    assert curve[2] + curve[3] >= 28, "the cited pair is ~15.7 + ~15.4"
    assert max(curve) >= 6, "something expensive to ramp INTO"


@requires_data
@requires_deck
def test_a_built_deck_completes_a_combo_line_it_half_holds():
    """kinnan's defining line is a two-card infinite and the baseline contained
    **23 Kinnan combo partners and not one completion** — the scorer sees cards, not
    pairs, so a card that FINISHES a line earns nothing for finishing it."""
    plan = json.loads((DECKS_DIR / "kinnan" / "build_plan.json").read_text())
    completions = plan.get("combo_completions") or []
    assert completions, "the builder must assemble at least one line it half-held"
    for c in completions:
        assert c["added"] and c["replaced"] and c["line"], c
        assert c["added"] in c["line"], "a completion must belong to the line it names"
        assert {s["name"] for s in plan["slots"]} >= {c["added"]}, "and be in the deck"
    names = {s["name"] for s in plan["slots"]} | {plan["commander"]}
    assert any(set(c["line"]) <= names for c in completions), \
        "at least one named line must be FULLY contained once the swap is made"
