"""Artist credits: standout detection, honest counting, theme detection."""

from conftest import requires_deck
from manamap.pilot.artist_credits import (
    analyze,
    drop_runs,
    find_standout,
    rank_artists,
    roster_overlap,
    treatments,
)


def card(name, artist, **overrides):
    base = {
        "name": name, "artist": artist, "quantity": 1, "type_line": "Creature — Goblin",
        "set": "sld", "set_name": "Secret Lair Drop", "collector_number": "100",
        "border_color": "black", "frame_effects": [], "finishes": ["nonfoil"],
        "foil": False, "is_commander": False, "is_sideboard": False,
    }
    base.update(overrides)
    return base


def deck(pairs, **common):
    return [card(name, artist, **common) for name, artist in pairs]


# ── Ranking and counting basis ───────────────────────────────────────────


def test_ranking_is_descending_and_deterministic():
    cards = deck([("A", "Zoe"), ("B", "Zoe"), ("C", "Abe")])
    ranking = rank_artists(cards)
    assert [r["artist"] for r in ranking] == ["Zoe", "Abe"]
    assert rank_artists(cards) == rank_artists(list(reversed(cards)))


def test_ties_break_by_artist_name():
    ranking = rank_artists(deck([("A", "Zoe"), ("B", "Abe")]))
    assert [r["artist"] for r in ranking] == ["Abe", "Zoe"]


def test_basics_count_once_per_entry_but_copies_reported():
    """Per-copy would inflate a basic-land artist; per-entry is the honest basis."""
    cards = [card("Mountain", "Barge", quantity=22), card("Sol Ring", "Barge")]
    ranking = rank_artists(cards)
    assert ranking[0]["entries"] == 2
    assert ranking[0]["copies"] == 23


def test_accessories_are_excluded_from_counts():
    cards = deck([("Real Card", "Barge")]) + [
        card("Storm Counter", "Barge", type_line="Card")]
    ranking = rank_artists(cards)
    assert ranking[0]["entries"] == 1
    assert analyze(cards)["totals"]["accessories"] == 1


def test_cards_without_an_artist_are_skipped():
    cards = deck([("A", "Barge")]) + [card("Mystery", None)]
    assert rank_artists(cards)[0]["entries"] == 1
    assert len(rank_artists(cards)) == 1


# ── Standout detection ───────────────────────────────────────────────────


def test_standout_qualifies_on_ratio():
    ranking = rank_artists(deck([("A", "Barge"), ("B", "Barge"), ("C", "Barge"),
                                 ("D", "Other")]))
    standout = find_standout(ranking, 4)
    assert standout["artist"] == "Barge" and standout["entries"] == 3


def test_standout_rejected_when_lead_is_narrow():
    """5 vs 4 is not a story — it fails the 2x rule and the share rule."""
    cards = deck([(f"L{i}", "Lead") for i in range(5)]
                 + [(f"R{i}", "Runner") for i in range(4)]
                 + [(f"X{i}", f"Solo{i}") for i in range(25)])
    ranking = rank_artists(cards)
    assert find_standout(ranking, len(cards)) is None


def test_standout_qualifies_on_share_even_without_ratio():
    cards = deck([(f"L{i}", "Lead") for i in range(4)]
                 + [(f"R{i}", "Runner") for i in range(3)]
                 + [(f"X{i}", f"Solo{i}") for i in range(3)])
    standout = find_standout(rank_artists(cards), len(cards))
    assert standout["artist"] == "Lead"      # 4/10 = 40% share


def test_no_standout_when_every_artist_is_unique():
    cards = deck([(f"C{i}", f"Artist{i}") for i in range(40)])
    result = analyze(cards)
    assert result["standout"] is None
    assert any("No standout" in n for n in result["notes"])


def test_standout_requires_minimum_entries():
    """Two cards out of three is a high share but too small to be a story."""
    ranking = rank_artists(deck([("A", "Barge"), ("B", "Barge"), ("C", "Other")]))
    assert find_standout(ranking, 3) is None


def test_clusters_exclude_the_standout():
    cards = deck([("A", "Lead"), ("B", "Lead"), ("C", "Lead"), ("D", "Pair"),
                  ("E", "Pair"), ("F", "Solo")])
    result = analyze(cards)
    assert result["standout"]["artist"] == "Lead"
    assert [c["artist"] for c in result["clusters"]] == ["Pair"]


# ── Roster overlap ───────────────────────────────────────────────────────


def test_roster_overlap_finds_concentration_and_dispersion():
    cards = deck([("E1", "Barge"), ("E2", "Barge"), ("E3", "Barge"), ("E4", "Other"),
                  ("S1", "A1"), ("S2", "A2"), ("S3", "A3")])
    roster = [{"role": "The engine", "cards": ["E1", "E2", "E3", "E4"]},
              {"role": "Spells", "cards": ["S1", "S2", "S3"]}]
    rows = {r["group"]: r for r in roster_overlap(cards, roster)}
    assert rows["The engine"]["artist"] == "Barge"
    assert rows["The engine"]["painted"] == 3 and rows["The engine"]["of"] == 4
    assert rows["Spells"]["distinct_artists"] == 3   # total dispersion
    assert rows["Spells"]["painted"] == 1


def test_roster_overlap_ignores_unknown_card_names():
    cards = deck([("A", "Barge")])
    rows = roster_overlap(cards, [{"role": "R", "cards": ["A", "Not In Deck"]}])
    assert rows[0]["of"] == 1


def test_roster_overlap_empty_without_roster():
    assert roster_overlap(deck([("A", "Barge")]), None) == []


# ── Treatments and drop runs ─────────────────────────────────────────────


def test_treatments_counts_borderless_foil_and_frames():
    cards = [card("A", "Barge", border_color="borderless", foil=True,
                  frame_effects=["inverted", "legendary"]),
             card("B", "Other")]
    result = treatments(cards)
    assert result["borderless"] == 1 and result["foil"] == 1
    assert result["frame_effects"] == {"inverted": 1, "legendary": 1}


def test_drop_run_detects_a_whole_drop():
    cards = [card(f"C{n}", "Barge", collector_number=str(n)) for n in (10, 11, 12, 13)]
    runs = drop_runs(cards)
    assert len(runs) == 1
    assert runs[0]["from"] == "10" and runs[0]["to"] == "13" and runs[0]["entries"] == 4


def test_drop_run_ignores_short_and_broken_sequences():
    cards = [card(f"C{n}", "Barge", collector_number=str(n)) for n in (10, 11, 50)]
    assert drop_runs(cards) == []


def test_drop_run_skips_non_numeric_collector_numbers():
    """The List uses forms like LCI-132."""
    cards = [card(f"C{i}", "Barge", set="plst", collector_number=f"LCI-{i}")
             for i in range(5)]
    assert drop_runs(cards) == []


def test_drop_run_requires_same_artist():
    cards = [card("A", "Barge", collector_number="10"),
             card("B", "Other", collector_number="11"),
             card("C", "Barge", collector_number="12")]
    assert drop_runs(cards) == []


# ── Notes and determinism ────────────────────────────────────────────────


def test_notes_flag_multi_copy_entries():
    result = analyze([card("Mountain", "Barge", quantity=22),
                      card("A", "Barge"), card("B", "Barge")])
    assert any("22 copies" in n and "counted once" in n for n in result["notes"])


def test_notes_warn_when_concentration_may_be_structural():
    """A dispersed remainder means the lead came from a product, not taste."""
    cards = ([card(f"D{i}", "Barge", set="sld") for i in range(6)]
             + [card(f"L{i}", f"Artist{i}", set="plst") for i in range(20)])
    result = analyze(cards)
    assert result["standout"]["artist"] == "Barge"
    assert any("structural rather than curated" in n for n in result["notes"])


def test_analyze_is_deterministic():
    cards = deck([("A", "Zoe"), ("B", "Abe"), ("C", "Zoe")])
    assert analyze(cards) == analyze(list(reversed(cards)))


def test_analyze_handles_an_empty_deck():
    result = analyze([])
    assert result["standout"] is None
    assert result["totals"]["entries"] == 0


# ── Data-gated: the real deck ────────────────────────────────────────────


@requires_deck
def test_real_deck_finds_the_secret_lair_story():
    from manamap.pilot.artist_credits import load_roster
    from manamap.pilot.common import load_deck_cards

    cards = load_deck_cards("goblin-storm")["cards"]
    result = analyze(cards, load_roster("goblin-storm"))

    standout = result["standout"]
    assert standout and standout["artist"] == "Wizard of Barge"
    # Accessories excluded, so 14 real cards — not the 16 printings.
    assert standout["entries"] == 14
    assert "Zada, Hedron Grinder" in standout["cards"]

    # Roster group labels are editorial copy and change between plans — assert
    # against the analysis's substance, not any specific headline. The Wizard
    # of Barge concentration must surface in whichever group holds Zada.
    zada_groups = [r for r in result["roster_overlap"]
                   if r["artist"] == "Wizard of Barge" and r["painted"] >= 3]
    assert zada_groups, result["roster_overlap"]

    assert any(r["set"] == "sld" and r["from"] == "2406" for r in result["drop_runs"])
    assert result["treatments"]["borderless"] == result["treatments"]["foil"] == 14
