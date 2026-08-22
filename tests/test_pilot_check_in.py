"""check-in: a paper list arrives, and the repo refuses to guess about it.

The command exists because the recipe was being run from memory, and every way
of getting it slightly wrong is SILENT. A card written twice, a name
misremembered, ninety-nine cards where there should be a hundred — `fetch-deck`
would resolve what it could and carry on, and everything downstream would then
measure a deck that does not exist in cardboard.

So the tests here are mostly about refusal. The diff is the easy half.
"""

import shutil

import pytest

from manamap.pilot import check_in
from manamap.pilot.common import deck_dir
from manamap.pilot.fetch_deck import parse_decklist

from conftest import requires_deck, requires_data

SLUG = "edgar-vampires"


@pytest.fixture
def paper():
    return (deck_dir(SLUG) / "decklist.txt").read_text(encoding="utf-8")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Writes go to a copy. `decklist.txt` is tracked and other tests read it."""
    shutil.copy(deck_dir(SLUG) / "decklist.txt", tmp_path / "decklist.txt")
    monkeypatch.setattr(check_in, "deck_dir", lambda slug: tmp_path)
    return tmp_path


@requires_deck
def test_a_deck_against_itself_is_a_no_op(paper):
    d = check_in.analyze(SLUG, paper)
    assert d["pull"] == {} and d["add"] == {}
    assert not d["blocking"]
    assert d["cards"] == 100


@requires_deck
def test_the_diff_is_in_copies_and_names_the_cards(paper):
    """Counting entries instead of copies is the mistake this repo has published
    before — "18 lands" for a 33-land deck."""
    lines = paper.replace("1 Anguished Unmaking\n", "1 Sol Ring\n")
    d = check_in.analyze(SLUG, lines)
    assert d["pull"] == {"Anguished Unmaking": 1}
    assert d["add"] == {"Sol Ring": 1}
    assert d["unchanged"] == 99


@requires_deck
@requires_data
def test_a_card_written_twice_is_refused(paper):
    """The characteristic paper-list error: you read the sleeve, write it down,
    and meet it again forty cards later. Singleton makes it illegal."""
    text = paper.replace("1 Anguished Unmaking\n", "") + "1 Akroma's Will\n"
    d = check_in.analyze(SLUG, text)
    assert any("more than once" in b for b in d["blocking"])
    assert any("Akroma's Will" in b for b in d["blocking"])


@requires_deck
def test_basics_may_repeat(paper):
    """A deck legitimately holds many Swamps; singleton does not bind them."""
    text = paper.replace("1 Anguished Unmaking\n", "1 Swamp\n1 Swamp\n").replace(
        "1 Akroma's Will (M3C) 165\n", "")
    d = check_in.analyze(SLUG, text)
    assert not any("more than once" in b for b in d["blocking"]), d["blocking"]


@requires_deck
def test_the_wrong_card_count_is_refused(paper):
    d = check_in.analyze(SLUG, paper.replace("1 Anguished Unmaking\n", ""))
    assert any("not 100" in b for b in d["blocking"])


@requires_deck
@requires_data
def test_a_misremembered_name_is_refused(paper):
    """A typo here becomes a card the deck does not have, and `fetch-deck` would
    simply not resolve it and move on."""
    d = check_in.analyze(SLUG, paper.replace("Anguished Unmaking", "Anguished Unmakeing"))
    assert any("no card in the corpus" in b for b in d["blocking"])


@requires_deck
def test_a_list_with_no_commander_is_refused(paper):
    text = "\n".join(l for l in paper.split("\n")
                     if "Edgar Markov" not in l and l != "Commander:")
    d = check_in.analyze(SLUG, text)
    assert any("no commander" in b for b in d["blocking"])


@requires_deck
@requires_data
def test_a_changed_commander_warns_rather_than_refuses(paper):
    """It is a different deck, and that is the pilot's call — but a new slug is
    almost always what they meant."""
    text = paper.replace("1 Edgar Markov (INR) 234", "1 Sol Ring").replace(
        "1 Sol Ring (M3C) 264\n", "1 Edgar Markov *CMDR*\n")
    d = check_in.analyze(SLUG, text)
    assert not any("commander" in b for b in d["blocking"]), d["blocking"]


@requires_deck
def test_nothing_is_written_on_a_dry_run(sandbox, paper):
    before = (sandbox / "decklist.txt").read_text()
    check_in.analyze(SLUG, paper.replace("1 Anguished Unmaking\n", "1 Sol Ring\n"))
    assert (sandbox / "decklist.txt").read_text() == before


@requires_deck
def test_apply_writes_a_canonical_list_that_round_trips(sandbox, paper):
    entries = parse_decklist(paper.replace("1 Anguished Unmaking\n", "1 Sol Ring\n"))
    check_in.apply(SLUG, entries, run_chain=False)
    again = check_in.analyze(SLUG, (sandbox / "decklist.txt").read_text())
    assert again["pull"] == {} and again["add"] == {}, "the written list must be a fixed point"
    assert again["cards"] == 100


@requires_deck
def test_apply_preserves_printings_and_foils(sandbox, paper):
    """`fetch-deck` resolves exact printings from these; dropping them silently
    re-resolves a Secret Lair to its cheapest reprint."""
    check_in.apply(SLUG, parse_decklist(paper), run_chain=False)
    written = (sandbox / "decklist.txt").read_text()
    assert "(INR) 234" in written
    assert "(M3C) 165" in written


@requires_deck
def test_apply_keeps_a_backup_of_what_it_replaced(sandbox, paper):
    check_in.apply(SLUG, parse_decklist(paper), run_chain=False)
    assert (sandbox / "decklist.txt.bak").exists()


@requires_deck
def test_reformatting_alone_cannot_manufacture_a_version(sandbox, paper):
    """`deck-history` and `deck-version` compare PARSED entries, so the canonical
    rewrite must be entry-identical to what came in — otherwise every check-in
    would look like a swap."""
    entries = parse_decklist(paper)
    check_in.apply(SLUG, entries, run_chain=False)
    after = parse_decklist((sandbox / "decklist.txt").read_text())
    def key(es):
        return sorted((e["name"], e.get("quantity") or 1, bool(e.get("is_commander")))
                      for e in es)
    assert key(after) == key(entries)
