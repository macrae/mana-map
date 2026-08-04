

# ── color_identity parsing (2026-08-03) ──────────────────────────────────

def test_load_pool_parses_a_multicolour_identity():
    """cards.csv stores "B, R, W" — comma AND space separated.

    `set(raw)` yields {' ', ',', 'B', 'R', 'W'}, which is never a subset of a
    commander's identity, so `_eligible` silently rejected EVERY multicoloured
    card in the format. Mono-coloured and colourless rows have no separator and
    worked, which is why it survived: the mono-black deck this brief was built
    against could not see the difference. It hid 5,462 cards from a five-colour
    commander and 1,421 from Mardu.
    """
    from manamap.pilot.upgrade_facts import load_pool
    pool = load_pool()
    if pool is None:
        import pytest
        pytest.skip("requires data/cards.csv (a pipeline run)")
    assert pool["Edgar Markov"]["color_identity"] == {"B", "R", "W"}
    assert pool["Sol Ring"]["color_identity"] == set()
    assert pool["Swamp"]["color_identity"] == {"B"}
    for info in pool.values():
        assert info["color_identity"] <= set("WUBRG"), info["color_identity"]


def test_a_multicolour_card_is_eligible_in_an_identity_that_contains_it():
    from manamap.pilot.upgrade_facts import _eligible
    pool = {"Anguished Unmaking": {"color_identity": {"B", "W"}, "legal": True}}
    assert _eligible("Anguished Unmaking", pool, {"B", "R", "W"}, set())
    assert not _eligible("Anguished Unmaking", pool, {"B", "G"}, set())
