"""The pilot's Meren goal as paired rows (2026-11-23): "experience 5+ by T6" and "a 5+
drop via Meren by T6" — read from stated games, never from a simulation, and a cached
`try` reading that predates them is not served."""
from manamap.pilot import diagnostic, net_change, try_swap


def _game(experience, arrivals):
    return {"experience_by_turn": experience, "big_arrivals": arrivals,
            "mana_by_turn": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}


GAMES = [
    # experience 5 at T6 and Meren returned a 5-drop on T6: both rows hit
    _game([0, 0, 0, 1, 3, 5, 7, 9, 11, 13], [(5, 6, "meren")]),
    # experience 4 at T6; the 6-drop came from Reanimate, not Meren: neither row
    _game([0, 0, 0, 1, 2, 4, 6, 8, 10, 12], [(6, 3, "reanimate")]),
    # experience 6 at T6, but Meren's first fatty is on T7: only the experience row
    _game([0, 0, 1, 2, 4, 6, 8, 9, 10, 11], [(6, 7, "meren"), (5, 5, "cast")]),
    # a 4-drop returned by Meren on T5 is not a fatty
    _game([0, 0, 0, 0, 0, 0, 1, 2, 3, 4], [(4, 5, "meren")]),
]


def _per_game(key):
    return diagnostic.per_game(GAMES)[f"recursion|{key}|6"]


def test_experience_goal_counts_games_at_five_or_more_by_turn_six():
    assert _per_game("experience_goal_by_turn") == [1, 0, 1, 0]


def test_the_meren_route_counts_only_her_end_step_returns_of_a_five_drop_by_turn_six():
    assert _per_game("fatty_via_meren_by_turn") == [1, 0, 0, 0]


def test_the_reading_block_carries_both_series_with_the_same_counts():
    block = diagnostic.recursion(GAMES)
    assert block["experience_goal_by_turn"]["6"]["rate"] == 0.5
    assert block["fatty_via_meren_by_turn"]["6"]["rate"] == 0.25
    assert block["fatty_via_meren_by_turn"]["7"]["rate"] == 0.5


def test_a_deck_with_no_graveyard_tracked_reads_absent_not_zero():
    bare = [{"mana_by_turn": [1] * 10}]
    got = diagnostic.per_game(bare)
    assert got["recursion|experience_goal_by_turn|6"] == [None]
    assert got["recursion|fatty_via_meren_by_turn|6"] == [None]
    assert diagnostic.recursion(bare) is None


def test_both_rows_are_registered_defined_and_grouped():
    labels = {label: (block, key, turn) for label, block, key, turn, _ in net_change.ROWS}
    for label in ("experience 5+ by T6", "a 5+ drop via Meren by T6"):
        block, key, turn = labels[label]
        assert (block, key, turn) in diagnostic.PER_GAME
        assert label in net_change.METRICS
        assert any(label in names for _, names in net_change.ROW_GROUPS)


def test_the_try_cache_key_moves_when_the_reading_code_changes(tmp_path, monkeypatch):
    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "cards.json").write_text("[]")
    monkeypatch.setattr(try_swap, "deck_dir", lambda slug, branch=None: deck)
    monkeypatch.setattr(try_swap, "deck_file", lambda slug, name, branch=None: deck / name)
    before = try_swap._champion_key("no-such-deck", None, 100, 7)
    copy = tmp_path / "diagnostic.py"
    copy.write_bytes(open(diagnostic.__file__, "rb").read() + b"\n# a new row\n")
    monkeypatch.setattr(diagnostic, "__file__", str(copy))
    assert try_swap._champion_key("no-such-deck", None, 100, 7) != before
