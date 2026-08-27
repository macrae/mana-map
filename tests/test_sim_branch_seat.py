"""A branch is a seat: `<slug>@<branch>` sits down at the table like any list.

The load-bearing test is the win tally. A branch seat is written to Forge as
`ur-dragon-treasure-v2` because `@` has no business in a deck registry — so an
outcome names the flattened form while the seat list holds the slug, and a bare
`==` matched every OTHER seat while scoring ours zero. The run printed
"wins 0" for a list that had won ELEVEN of a hundred, with the other three
seats' counts correct beside it, which is exactly the shape that gets believed.
"""

import pytest

from manamap.sim import forge


def test_a_branch_seat_resolves_to_its_own_directory():
    base, branch = forge.split_seat("ur-dragon@treasure-v2")
    assert (base, branch) == ("ur-dragon", "treasure-v2")
    assert forge.split_seat("vito") == ("vito", None)


def test_a_branch_run_is_filed_beside_the_list_it_measured():
    """A branch's win rate under the champion's name is the silent-overwrite
    class this repo keeps finding."""
    from manamap.pilot.common import DECKS_DIR
    if not (DECKS_DIR / "ur-dragon" / "branches" / "treasure-v2").is_dir():
        pytest.skip("no branch fixture")
    out = forge._out_dir("ur-dragon@treasure-v2")
    assert "branches" in out.parts and "treasure-v2" in out.parts
    assert forge._out_dir("ur-dragon") == DECKS_DIR / "ur-dragon" / forge.SIM_DIR


def test_the_forge_name_flattens_but_the_tally_still_finds_the_seat():
    """The bug, as a property: whatever name Forge is given, the seat's wins
    must be found. Counting on the raw slug scored a branch zero."""
    assert forge.deck_meta_name("ur-dragon@treasure-v2") == "ur-dragon-treasure-v2"
    assert forge.deck_meta_name("vito") == "vito"
    seats = ["ur-dragon@treasure-v2", "vito"]
    outcomes = [{"winner": "ur-dragon-treasure-v2"}, {"winner": "vito"},
                {"winner": "ur-dragon-treasure-v2"}]
    wins = {s: sum(1 for o in outcomes if o["winner"] == forge.deck_meta_name(s))
            for s in seats}
    assert wins["ur-dragon@treasure-v2"] == 2, (
        "the branch seat's wins were not found — the tally is matching the raw "
        "slug against a Forge name that flattens it")


def test_a_tracked_branch_run_agrees_with_its_own_analysis():
    """`summary.wins` and `analysis.seats` must not disagree: the first is the
    headline and the second is the detail, and the run shipped with them
    contradicting each other by eleven games."""
    import glob
    import json
    hits = glob.glob("data/decks/*/branches/*/sim/*.json")
    if not hits:
        pytest.skip("no branch run on this machine")
    for path in hits:
        rec = json.load(open(path))
        summary = (rec.get("summary") or {}).get("wins") or {}
        seats = (rec.get("analysis") or {}).get("seats") or {}
        for slug, n in summary.items():
            key = forge.deck_meta_name(slug)
            if key in seats:
                assert seats[key].get("wins") == n, (
                    f"{path}: summary says {slug} won {n}, analysis says "
                    f"{seats[key].get('wins')}")
