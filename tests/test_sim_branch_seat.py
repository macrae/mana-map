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
from conftest import A_BRANCH, ROOT, requires_branch


def test_a_branch_seat_resolves_to_its_own_directory():
    # A LITERAL ON BOTH SIDES: this parses a seat string and touches no disk, so
    # it must not depend on which branches happen to exist.
    base, branch = forge.split_seat("ur-dragon@some-branch")
    assert (base, branch) == ("ur-dragon", "some-branch")
    assert forge.split_seat("vito") == ("vito", None)


def test_a_branch_run_is_filed_beside_the_list_it_measured():
    """A branch's win rate under the champion's name is the silent-overwrite
    class this repo keeps finding."""
    from manamap.pilot.common import DECKS_DIR
    if A_BRANCH is None:
        pytest.skip("no branch on ur-dragon")
    out = forge._out_dir(f"ur-dragon@{A_BRANCH}")
    assert "branches" in out.parts and A_BRANCH in out.parts
    assert forge._out_dir("ur-dragon") == DECKS_DIR / "ur-dragon" / forge.SIM_DIR


def test_the_forge_name_flattens_but_the_tally_still_finds_the_seat():
    """The bug, as a property: whatever name Forge is given, the seat's wins
    must be found. Counting on the raw slug scored a branch zero."""
    assert forge.deck_meta_name("ur-dragon@treasure-v2") == "ur-dragon-treasure-v2"
    assert forge.deck_meta_name("vito") == "vito"
    seats = ["ur-dragon@treasure-v2", "vito"]
    outcomes = [{"winner": "ur-dragon-treasure-v2"}, {"winner": "vito"},
                {"winner": "ur-dragon-treasure-v2"}]
    # THROUGH THE PRODUCTION TALLY. This test used to build its own correct copy
    # of `forge.py`'s expression — character for character, the very expression
    # the bug lived in — so a regression to `o["winner"] == s` left it green.
    wins = forge.tally_wins(outcomes, seats)
    assert wins["ur-dragon@treasure-v2"] == 2, (
        "the branch seat's wins were not found — the tally is matching the raw "
        "slug against a Forge name that flattens it")
    assert wins["vito"] == 1, "a plain seat stopped being counted"


def test_a_tracked_branch_run_agrees_with_its_own_analysis():
    """`summary.wins` and `analysis.seats` must not disagree: the first is the
    headline and the second is the detail, and the run shipped with them
    contradicting each other by eleven games."""
    import glob
    import json
    hits = glob.glob(str(ROOT / "data/decks/*/branches/*/sim/*.json"))
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
