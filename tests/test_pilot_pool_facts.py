"""Tests for pool-facts: analysing a box of cards rather than a finished deck.

The pure functions are tested against hand-built card tables so they hold
without any generated artifact. The end-to-end test needs cards.csv and the
graphs and is gated behind `requires_data`.
"""

import json

import pytest

from conftest import requires_data, requires_roles
from manamap.pilot import pool_facts
from manamap.pilot.common import commander_rejection

# A miniature cards.csv: enough shape for identity, land and commander logic.
CARDS = {
    "Hapatra, Vizier of Poisons": {
        "name": "Hapatra, Vizier of Poisons",
        "type_line": "Legendary Creature — Zombie Cleric",
        "oracle_text": "Deathtouch",
        "color_identity": "B, G",
        "legal_commander": "legal",
        "edhrec_rank": 4033.0,
        "mechanical_tags": "counters_minus, tokens",
    },
    "Grist, Voracious Larva // Grist, the Plague Swarm": {
        "name": "Grist, Voracious Larva // Grist, the Plague Swarm",
        "type_line": "Legendary Creature — Insect // Legendary Planeswalker — Grist",
        "oracle_text": "",
        "color_identity": "B, G",
        "legal_commander": "legal",
        "edhrec_rank": 4455.0,
        "mechanical_tags": "tokens",
    },
    "Overgrown Tomb": {
        "name": "Overgrown Tomb",
        "type_line": "Land — Swamp Forest",
        "oracle_text": "({T}: Add {B} or {G}.)",
        "color_identity": "B, G",
        "legal_commander": "legal",
        "edhrec_rank": 100.0,
        "mechanical_tags": "",
    },
    "Command Tower": {
        "name": "Command Tower",
        "type_line": "Land",
        "oracle_text": "{T}: Add one mana of any color that a commander in the command zone could produce.",
        "color_identity": "",
        "legal_commander": "legal",
        "edhrec_rank": 1.0,
        "mechanical_tags": "",
    },
    "Island": {
        "name": "Island",
        "type_line": "Basic Land — Island",
        "oracle_text": "",
        "color_identity": "U",
        "legal_commander": "legal",
        "edhrec_rank": None,
        "mechanical_tags": "",
    },
    "Sol Ring": {
        "name": "Sol Ring",
        "type_line": "Artifact",
        "oracle_text": "{T}: Add {C}{C}.",
        "color_identity": "",
        "legal_commander": "legal",
        "edhrec_rank": 1.0,
        "mechanical_tags": "ramp",
    },
    "Black Lotus": {  # legal_commander != legal — must never be a commander
        "name": "Black Lotus",
        "type_line": "Artifact",
        "oracle_text": "",
        "color_identity": "",
        "legal_commander": "banned",
        "edhrec_rank": None,
        "mechanical_tags": "",
    },
}


# ── The shared commander predicate ──────────────────────────────────────


def test_legendary_creature_is_a_commander():
    assert commander_rejection(CARDS["Hapatra, Vizier of Poisons"]) is None


def test_front_face_decides_for_a_dfc():
    """Grist's back face is a planeswalker; the front face is what matters."""
    assert commander_rejection(CARDS["Grist, Voracious Larva // Grist, the Plague Swarm"]) is None


def test_non_creature_without_the_text_is_rejected():
    reason = commander_rejection(CARDS["Sol Ring"])
    assert reason and "not a legal commander" in reason


def test_can_be_your_commander_text_qualifies():
    row = dict(CARDS["Sol Ring"], oracle_text="Sol Ring can be your commander.")
    assert commander_rejection(row) is None


def test_banned_card_is_rejected_even_if_legendary():
    row = dict(CARDS["Hapatra, Vizier of Poisons"], legal_commander="banned")
    assert commander_rejection(row) == "not legal in Commander"


def test_build_deck_still_raises_with_the_reason():
    """The refactor kept build_deck's error messages, which name the card."""
    from manamap.pilot.build_deck import BriefError, commander_identity

    with pytest.raises(BriefError, match="not a legal commander"):
        commander_identity(CARDS["Sol Ring"])
    assert commander_identity(CARDS["Hapatra, Vizier of Poisons"]) == {"B", "G"}


# ── Name resolution: the front-face translation ─────────────────────────


def test_front_face_map_keys_on_the_front_only():
    mapping = pool_facts.front_face_map(CARDS)
    assert mapping == {"Grist, Voracious Larva": "Grist, Voracious Larva // Grist, the Plague Swarm"}


def test_decklist_front_face_resolves_to_the_joined_name(tmp_path):
    """A decklist names the front face; every global artifact keys on the join.

    Without this translation the card is silently absent from every count
    rather than reported unresolved — the expensive failure mode.
    """
    lst = tmp_path / "deck.txt"
    lst.write_text("1 Grist, Voracious Larva (MH3) 251\n2 Sol Ring\n")
    per_file, unresolved = pool_facts.read_sources([lst], CARDS)
    assert unresolved == {}
    assert per_file["deck.txt"] == {
        "Grist, Voracious Larva // Grist, the Plague Swarm": 1,
        "Sol Ring": 2,
    }


def test_unresolved_names_are_reported_not_dropped(tmp_path):
    lst = tmp_path / "deck.txt"
    lst.write_text("1 Sol Ring\n1 Not A Real Card\n")
    per_file, unresolved = pool_facts.read_sources([lst], CARDS)
    assert per_file["deck.txt"] == {"Sol Ring": 1}
    assert unresolved == {"Not A Real Card": ["deck.txt"]}


def test_quantities_accumulate_across_files(tmp_path):
    (tmp_path / "a.txt").write_text("2 Sol Ring\n")
    (tmp_path / "b.txt").write_text("1 Sol Ring\n")
    per_file, _ = pool_facts.read_sources(
        [tmp_path / "a.txt", tmp_path / "b.txt"], CARDS
    )
    assert per_file["a.txt"]["Sol Ring"] == 2
    assert per_file["b.txt"]["Sol Ring"] == 1


def test_collect_paths_expands_a_directory_and_honours_exclude(tmp_path):
    (tmp_path / "a.txt").write_text("1 Sol Ring\n")
    (tmp_path / "b.txt").write_text("1 Sol Ring\n")
    (tmp_path / "notes.md").write_text("ignored")
    assert len(pool_facts.collect_paths([tmp_path], [])) == 2
    kept = pool_facts.collect_paths([tmp_path], [tmp_path / "b.txt"])
    assert [p.name for p in kept] == ["a.txt"]


def test_collect_paths_rejects_a_missing_target(tmp_path):
    with pytest.raises(SystemExit):
        pool_facts.collect_paths([tmp_path / "nope"], [])


# ── Castability: the number depth hides ─────────────────────────────────


def test_sources_count_copies_not_entries():
    pool = {"Overgrown Tomb": 1, "Island": 9}
    assert pool_facts.sources_for(pool, CARDS, {"U"}) == {"U": 9}


def test_any_colour_land_is_a_source_for_every_colour_in_identity():
    """Command Tower is why a naive '{B} appears in the text' count understates."""
    pool = {"Command Tower": 1}
    assert pool_facts.sources_for(pool, CARDS, {"B", "G"}) == {"B": 1, "G": 1}


def test_out_of_identity_lands_are_not_sources():
    """An Island in the box does nothing for a Golgari commander."""
    pool = {"Island": 5, "Overgrown Tomb": 1}
    assert pool_facts.sources_for(pool, CARDS, {"B", "G"}) == {"B": 1, "G": 1}


def test_nonland_mana_is_not_counted_as_a_source():
    """Sol Ring ramps; it is not a coloured source and never fixes."""
    assert pool_facts.sources_for({"Sol Ring": 1}, CARDS, {"B"}) == {"B": 0}


def test_in_identity_admits_colourless_and_rejects_off_colour():
    pool = {"Sol Ring": 1, "Island": 1, "Hapatra, Vizier of Poisons": 1}
    assert sorted(pool_facts.in_identity(pool, CARDS, {"B", "G"})) == [
        "Hapatra, Vizier of Poisons", "Sol Ring",
    ]


def test_identity_key_is_wubrg_ordered():
    assert pool_facts.identity_key({"G", "B"}) == "BG"
    assert pool_facts.identity_key({"G", "W", "U", "B", "R"}) == "WUBRG"
    assert pool_facts.identity_key(set()) == "C"


# ── Combo containment and dedup ─────────────────────────────────────────


def _details(records):
    by_card = {}
    for i, combo in enumerate(records):
        for card in combo["cards"]:
            by_card.setdefault(card, []).append(i)
    return {"combos": records, "by_card": by_card}


def test_only_fully_contained_lines_are_reported():
    details = _details([
        {"cards": ["Sol Ring", "Command Tower"], "produces": ["Infinite mana"],
         "bracket": 1, "popularity": 10},
        {"cards": ["Sol Ring", "Black Lotus"], "produces": ["Win the game"],
         "bracket": 4, "popularity": 99},
    ])
    lines = pool_facts.contained_combos({"Sol Ring": 1, "Command Tower": 1}, CARDS, details)
    assert [c["cards"] for c in lines] == [["Command Tower", "Sol Ring"]]


def test_duplicate_records_for_one_card_set_collapse():
    """The raw artifact holds several variants of one interaction.

    Springheart Nantuko + Tireless Provisioner appears twice in the real file;
    reporting it twice would overstate how many distinct lines a box contains.
    """
    details = _details([
        {"cards": ["Sol Ring", "Command Tower"], "produces": ["a"], "bracket": 1, "popularity": 10},
        {"cards": ["Command Tower", "Sol Ring"], "produces": ["b"], "bracket": 1, "popularity": 40},
    ])
    lines = pool_facts.contained_combos({"Sol Ring": 1, "Command Tower": 1}, CARDS, details)
    assert len(lines) == 1
    assert lines[0]["popularity"] == 40  # the most popular record wins


def test_every_line_is_marked_unverified():
    details = _details([
        {"cards": ["Sol Ring", "Command Tower"], "produces": ["Infinite mana"],
         "bracket": 1, "popularity": 10},
    ])
    lines = pool_facts.contained_combos({"Sol Ring": 1, "Command Tower": 1}, CARDS, details)
    assert lines[0]["verified"] is False
    assert lines[0]["infinite"] is True


def test_line_identity_is_the_union_of_its_pieces():
    details = _details([
        {"cards": ["Hapatra, Vizier of Poisons", "Island"], "produces": ["x"],
         "bracket": 2, "popularity": 1},
    ])
    lines = pool_facts.contained_combos(
        {"Hapatra, Vizier of Poisons": 1, "Island": 1}, CARDS, details
    )
    assert lines[0]["identity"] == "UBG"


# ── In-box upgrades ─────────────────────────────────────────────────────


def test_obsolescence_only_reports_replacements_you_own():
    index = {
        "Sol Ring": {"compare_with": [
            {"name": "Command Tower", "gains": ["Lower CMC"], "similarity": 0.9},
            {"name": "Mana Crypt", "gains": ["Free"], "similarity": 0.95},
        ]},
    }
    out = pool_facts.obsolete_in_pool({"Sol Ring": 1, "Command Tower": 1}, index)
    # THE SHAPE CARRIES BOTH SIDES AND THE MEASURE NOW. A one-sided
    # `advantages` list is how a card that CHARGED you something read as pure
    # upside.
    assert len(out) == 1 and out[0]["card"] == "Sol Ring"
    row = out[0]["compare_with"][0]
    assert row["name"] == "Command Tower"
    assert set(row) >= {"name", "strength", "gains", "costs", "narrows",
                        "played_more"}


def test_obsolescence_is_empty_when_nothing_is_owned():
    index = {"Sol Ring": {"compare_with": [{"name": "Mana Crypt", "gains": [], "similarity": 1}]}}
    assert pool_facts.obsolete_in_pool({"Sol Ring": 1}, index) == []


# ── Notes: the traps said out loud ──────────────────────────────────────


def test_a_deep_but_uncastable_identity_is_called_out():
    """Depth without castability is the trap the module exists to close."""
    facts = {
        "resolution": {"unresolved": []},
        "identities": [{"key": "WUBG", "depth": 663, "sources": {"W": 10, "U": 0, "B": 44, "G": 35}}],
        "combos": {"total": 0, "reported": 0},
        "bracket": {},
    }
    notes = pool_facts.build_notes(facts)
    assert any("not castable" in n and "U=0" in n for n in notes)


def test_a_castable_identity_gets_no_warning():
    facts = {
        "resolution": {"unresolved": []},
        "identities": [{"key": "BG", "depth": 524, "sources": {"B": 42, "G": 34}}],
        "combos": {"total": 0, "reported": 0},
        "bracket": {},
    }
    assert not any("not castable" in n for n in pool_facts.build_notes(facts))


def test_role_holes_carry_the_taxonomy_caveat():
    """A hole is only as good as the taxonomy's coverage, and it must say so.

    Necropotence classifies as `stax` and Sylvan Library not at all, so a deck
    with six real draw engines reports `draw -8`. Reporting that as a deck
    weakness is a confidently wrong answer.
    """
    facts = {
        "resolution": {"unresolved": []},
        "identities": [{"key": "BG", "depth": 86, "sources": {"B": 42, "G": 34},
                        "role_holes": {"draw": 8}, "role_unclassified": 21}],
        "combos": {"total": 0, "reported": 0},
        "bracket": {},
    }
    notes = pool_facts.build_notes(facts)
    assert any("no card_roles.json entry" in n and "21" in n for n in notes)


def test_no_taxonomy_caveat_when_coverage_is_complete():
    facts = {
        "resolution": {"unresolved": []},
        "identities": [{"key": "BG", "depth": 86, "sources": {"B": 42, "G": 34},
                        "role_holes": {"draw": 8}, "role_unclassified": 0}],
        "combos": {"total": 0, "reported": 0},
        "bracket": {},
    }
    assert not any("card_roles.json entry" in n for n in pool_facts.build_notes(facts))


def test_role_coverage_counts_the_unclassified():
    """Cards with no roles land in flex; the count is the hole's denominator."""
    roles = {"Hapatra, Vizier of Poisons": ["threat:body"]}
    names = ["Hapatra, Vizier of Poisons", "Sol Ring", "Overgrown Tomb"]
    histogram, _, unclassified = pool_facts.role_coverage(names, CARDS, roles)
    assert unclassified == 1                      # Sol Ring; the land is skipped
    assert sum(histogram.values()) == 2           # lands are excluded entirely
    # Both land in flex, but for different reasons — Sol Ring has no entry,
    # Hapatra has one whose role belongs to no budget group. Only the first is
    # a taxonomy gap, which is why `unclassified` is counted separately.
    assert histogram.get("flex") == 2


def test_in_box_upgrades_are_flagged_as_candidates():
    """The index compares structure; it gets read as function, and it is wrong.

    Boggart Mischief drains only when a *Goblin* dies and was offered as a
    replacement for Bastion of Remembrance, which drains on any creature.
    """
    facts = {
        "resolution": {"unresolved": []}, "identities": [],
        "combos": {"total": 0, "reported": 0}, "bracket": {},
        "obsolescence": [{"card": "Bastion of Remembrance", "compare_with": []}],
    }
    notes = pool_facts.build_notes(facts)
    # The caveat says what the data now supports: a STRENGTH, not a verdict,
    # and the classes it reads. It must still refuse to be read as fact.
    assert any("CANDIDATE" in n.upper() and "STRENGTH" in n.upper()
               for n in notes), notes
    assert any("cannot read a card" in n for n in notes), notes


def test_no_upgrade_caveat_when_there_are_no_upgrades():
    facts = {
        "resolution": {"unresolved": []}, "identities": [],
        "combos": {"total": 0, "reported": 0}, "bracket": {}, "obsolescence": [],
    }
    assert not any("CANDIDATES" in n for n in pool_facts.build_notes(facts))


def test_truncation_is_announced():
    facts = {
        "resolution": {"unresolved": []},
        "identities": [],
        "combos": {"total": 31, "reported": 25},
        "bracket": {},
    }
    notes = pool_facts.build_notes(facts)
    assert any("31 contained lines" in n and "25 most popular" in n for n in notes)


def test_unresolved_names_are_surfaced_as_a_note():
    facts = {
        "resolution": {"unresolved": [{"name": "Nope", "files": ["a.txt"]}]},
        "identities": [], "combos": {"total": 0, "reported": 0}, "bracket": {},
    }
    assert any("did not resolve" in n for n in pool_facts.build_notes(facts))


# ── End to end, against the real artifacts ──────────────────────────────


@requires_data
@requires_roles
def test_analyze_runs_end_to_end_on_a_small_pool(tmp_path):
    """A three-card list through the real cards.csv and graphs."""
    lst = tmp_path / "box.txt"
    lst.write_text("1 Sol Ring\n1 Command Tower\n1 Hapatra, Vizier of Poisons\n")
    facts = pool_facts.analyze([lst])

    assert facts["meta"]["distinct"] == 3
    assert facts["resolution"]["unresolved"] == []
    assert [c["name"] for c in facts["commanders"]] == ["Hapatra, Vizier of Poisons"]
    assert facts["commanders"][0]["color_identity"] == "BG"
    # Sol Ring and Command Tower are colourless, so all three are playable.
    assert facts["identities"][0]["depth"] == 3
    json.dumps(facts)  # the artifact must be serialisable


@requires_data
@requires_roles
def test_analyze_excludes_a_named_file(tmp_path):
    (tmp_path / "a.txt").write_text("1 Sol Ring\n")
    (tmp_path / "b.txt").write_text("1 Command Tower\n")
    facts = pool_facts.analyze([tmp_path], exclude=[tmp_path / "b.txt"])
    assert facts["meta"]["distinct"] == 1
    assert [s["file"] for s in facts["sources"]] == ["a.txt"]


@requires_data
@requires_roles
def test_a_pool_of_only_unresolvable_names_fails_loudly(tmp_path):
    lst = tmp_path / "box.txt"
    lst.write_text("1 Definitely Not A Magic Card\n")
    with pytest.raises(SystemExit):
        pool_facts.analyze([lst])


# ── One parse per process ────────────────────────────────────────────────


@requires_data
def test_load_cards_is_memoized(monkeypatch):
    """A single `pool-facts` run scanned the 24 MB CSV twice — once here and once
    inside `bracket.load_reference`. Same key discipline as `bracket._card_flags`."""
    import pandas as pd
    from manamap.pilot import pool_facts as pf
    from manamap.pilot.common import clear_memo

    clear_memo()
    calls = []
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    first = pf.load_cards()
    second = pf.load_cards()
    assert len(calls) == 1, "second call re-read the CSV"
    assert first is second, "callers must share one parse"


@requires_data
def test_load_cards_reparses_when_the_file_changes():
    """Keyed on (mtime_ns, size), so a regenerated cards.csv is picked up.

    The storage moved to `common._MTIME_MEMO` when five hand-rolled copies of
    this cache were consolidated; the property under test is unchanged.
    """
    from manamap.pilot import pool_facts as pf
    from manamap.pilot.common import _MTIME_MEMO, clear_memo

    clear_memo()
    pf.load_cards()
    sig, _ = _MTIME_MEMO["pool_facts:cards"]
    _MTIME_MEMO["pool_facts:cards"] = ((sig[0] - 1, sig[1]), {"stale": True})
    assert "stale" not in pf.load_cards()


def test_the_commander_rank_says_what_it_is_and_is_not():
    """`cards.csv`'s `edhrec_rank` is a CARD's popularity across the whole format in
    every role — not a rating of the card as a commander.

    In `pool_facts` it is simultaneously a displayed column and the shortlist's
    tiebreak sort key, which invited exactly one reading and it was the wrong one: on
    a real 931-card collection it put Selvala, Heart of the Wilds (card rank 430,
    commander rank #448) above Atraxa, Praetors' Voice (commander rank #4). The key is
    named `edhrec_card_rank` and the caveat rides in `notes` so it travels with the
    JSON, not just the human report.
    """
    facts = {"commanders": [{"name": "X", "edhrec_card_rank": 430, "depth": 1}],
             "resolution": {"unresolved": []}, "identities": [],
             "combos": {"total": 0, "reported": 0}, "bracket": {}}
    notes = pool_facts.build_notes(facts)
    note = next((n for n in notes if "edhrec_card_rank" in n), None)
    assert note, "the rank caveat must be a note, not only a column heading"
    assert "NOT a rating of the card as a commander" in note
    assert "tiebreak" in note, "the note must say why the ordering is what it is"


def test_no_rank_caveat_when_there_are_no_commanders():
    """A note about a section the report does not contain is noise."""
    facts = {"commanders": [], "resolution": {"unresolved": []}, "identities": [],
             "combos": {"total": 0, "reported": 0}, "bracket": {}}
    assert not [n for n in pool_facts.build_notes(facts) if "edhrec_card_rank" in n]
