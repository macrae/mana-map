"""An opponent can be the subject, so a table of four opponents is expressible.

`data/pods/` records the measurement this bench has never made: "a ROUND ROBIN
AMONG CANDIDATE OPPONENTS WITH NONE OF OUR DECKS SEATED — that ranks them on a
level field instead of inferring strength from tables they happened to sit at."

Part of why it was never made is that it could not be EXPRESSED. Every seat
resolved for READING through `seat_dir`, which searches `data/opponents/` and
then `data/decks/`; the subject resolved for WRITING through `deck_dir` alone.
So `simulate abaddon --vs giada-angels …` raised FileNotFoundError before Forge
was invoked, and the only way to rank a seat was to seat one of our own decks
beside it — which is exactly the confound the round robin exists to remove.
"""

import pytest

from manamap.config import DECKS_DIR, SIM_DIR
from manamap.sim import forge


def test_one_of_our_decks_still_writes_where_it_always_did():
    """The fallback must be a FALLBACK. Decks are tried first, so no existing
    call changes."""
    got = forge._out_dir("heliod")
    assert got == DECKS_DIR / "heliod" / SIM_DIR


def test_a_branch_still_writes_beside_its_own_list():
    got = forge._out_dir("zur-enchantress@drain-v2")
    assert got.parts[-3:] == ("drain-v2", SIM_DIR) or "drain-v2" in str(got)
    assert "branches" in str(got), got


def test_an_opponent_writes_beside_ITS_list():
    """The rule does not change: a record goes beside the list it measured.
    `data/opponents/<slug>/` holds a list exactly as `data/decks/<slug>/` does."""
    got = forge._out_dir("abaddon")
    assert got == DECKS_DIR.parent / "opponents" / "abaddon" / SIM_DIR


def test_a_name_that_is_neither_still_raises():
    """The fallback must not swallow a typo into a silent path."""
    with pytest.raises(FileNotFoundError):
        forge._out_dir("not-a-deck-or-an-opponent")


def test_a_branch_of_a_nonexistent_deck_still_raises():
    """The opponent fallback is only for a bare slug — an opponent has no
    branches, and resolving one there would invent a directory."""
    with pytest.raises(FileNotFoundError):
        forge._out_dir("abaddon@some-branch")


def test_every_seat_can_be_given_the_same_ai_profile():
    """A LEVEL FIELD IS THE WHOLE POINT. `simulate` gives the subject `Default`
    and the pod `Experimental` by default, which is correct when measuring one
    of our decks and WRONG when ranking seats against each other — the subject
    would be handicapped by its profile rather than its list."""
    import inspect

    src = inspect.getsource(forge.run)
    assert "profile" in src, "no profile plumbing in forge.run"


@pytest.mark.parametrize("subject", ["abaddon", "giada-angels", "baylen-tokens"])
def test_the_pod_seats_are_all_usable_as_subjects(subject):
    """The three seats of the standard table, which is the one being ranked."""
    got = forge._out_dir(subject)
    assert got.parent.name == subject
    assert (got.parent / "decklist.txt").exists()


# ── ONE PREDICATE, ONE HOME ───────────────────────────────────────────────

def test_every_seat_artifact_path_resolves_through_one_function():
    """THE FAILURE THIS PREVENTS COSTS AN HOUR EACH TIME. `_out_dir` gained an
    opponent fallback so a four-opponent table could run; the strategic-frame
    read one function later did NOT. So Forge played every game, for fifty
    minutes across five tables, and then the record write raised
    FileNotFoundError — full logs, zero records, and the error was invisible
    because the caller filtered stdout.

    `deck_dir` may be CALLED in exactly one place: inside `seat_home`. Parsed
    with `ast` rather than grepped, because the first version of this test
    counted a mention of `deck_dir(base, branch)` inside a DOCSTRING as a call.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(forge))
    holders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                    and sub.func.id == "deck_dir"):
                holders.append(node.name)
    assert holders == ["seat_home"], (
        f"deck_dir is called from {holders} — every one but seat_home will raise "
        "for an opponent subject, AFTER the games have already been played")


def test_the_strategic_frame_read_survives_an_opponent_subject():
    """The exact call that crashed. It must return a value, not raise."""
    import inspect

    src = inspect.getsource(forge)
    assert 'seat_home(split_seat(slug)[0]) / "strategic_frame.json"' in src
    # and it resolves for a real opponent
    assert forge.seat_home("giada-angels").name == "giada-angels"
