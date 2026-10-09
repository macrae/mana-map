"""check-in: a paper list arrives, and the repo refuses to guess about it.

The command exists because the recipe was being run from memory, and every way
of getting it slightly wrong is SILENT. A card written twice, a name
misremembered, ninety-nine cards where there should be a hundred — `fetch-deck`
would resolve what it could and carry on, and everything downstream would then
measure a deck that does not exist in cardboard.

So the tests here are mostly about refusal. The diff is the easy half.

WHY A SYNTHETIC LIST. The first version read `edgar-vampires/decklist.txt` and
did literal `str.replace` on card lines. That binds every case to one deck's
exact bytes: the moment the pilot checked in their real paper list — which
carries printing annotations — `"1 Anguished Unmaking\n"` stopped matching and
two tests failed for a reason that had nothing to do with check-in. The list
below is ours, uses real card names so the corpus check is exercised honestly,
and cannot be edited out from under us.
"""

import argparse
import shutil

import pytest

from manamap.pilot import check_in
from manamap.pilot.common import deck_dir
from manamap.pilot.fetch_deck import parse_decklist

from conftest import requires_deck, requires_data

SLUG = "cdeck"

# Real names, so `--owned`-style corpus checks are exercised for real. Printings
# on two lines, because that is what an export looks like and the canonical
# writer has to carry them through.
PAPER = """Commander:
1 Edgar Markov (INR) 234

Deck:
1 Akroma's Will (M3C) 165
1 Anguished Unmaking
1 Blood Artist
1 Sol Ring
""" + "".join(f"1 {n}\n" for n in (
    "Command Tower", "Blood Crypt", "Godless Shrine", "Sacred Foundry",
    "Bloodstained Mire", "Marsh Flats", "Vampiric Tutor", "Path to Exile",
)) + "87 Swamp\n"


@pytest.fixture
def paper():
    return PAPER


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Writes go to a directory of our own; nothing tracked is touched."""
    (tmp_path / "decklist.txt").write_text(PAPER)
    monkeypatch.setattr(check_in, "deck_dir", lambda slug: tmp_path)
    return tmp_path


@requires_deck
def test_a_deck_against_itself_is_a_no_op(sandbox, paper):
    d = check_in.analyze(SLUG, paper)
    assert d["pull"] == {} and d["add"] == {}
    assert not d["blocking"]
    assert d["cards"] == 100


@requires_deck
def test_the_diff_is_in_copies_and_names_the_cards(sandbox, paper):
    """Counting entries instead of copies is the mistake this repo has published
    before — "18 lands" for a 33-land deck."""
    lines = paper.replace("1 Anguished Unmaking\n", "1 Sol Ring\n")
    d = check_in.analyze(SLUG, lines)
    assert d["pull"] == {"Anguished Unmaking": 1}
    assert d["add"] == {"Sol Ring": 1}
    assert d["unchanged"] == 99


@requires_deck
@requires_data
def test_a_card_written_twice_is_refused(sandbox, paper):
    """The characteristic paper-list error: you read the sleeve, write it down,
    and meet it again forty cards later. Singleton makes it illegal."""
    text = paper.replace("1 Anguished Unmaking\n", "") + "1 Akroma's Will\n"
    d = check_in.analyze(SLUG, text)
    assert any("more than once" in b for b in d["blocking"])
    assert any("Akroma's Will" in b for b in d["blocking"])


@requires_deck
def test_basics_may_repeat(sandbox, paper):
    """A deck legitimately holds many Swamps; singleton does not bind them."""
    text = paper.replace("1 Anguished Unmaking\n", "1 Swamp\n1 Swamp\n").replace(
        "1 Akroma's Will (M3C) 165\n", "")
    d = check_in.analyze(SLUG, text)
    assert not any("more than once" in b for b in d["blocking"]), d["blocking"]


@requires_deck
def test_the_wrong_card_count_is_refused(sandbox, paper):
    d = check_in.analyze(SLUG, paper.replace("1 Anguished Unmaking\n", ""))
    assert any("expected exactly 100" in b for b in d["blocking"])


@requires_deck
def test_a_maybeboard_on_a_commander_list_is_dropped_with_a_warning(sandbox, paper):
    """A Moxfield export of a Commander deck routinely carries a Maybeboard,
    which the parser files as `side`. Commander has no sideboard, so those
    cards are DROPPED and said so — not refused, which would block a pilot
    pasting their real list over cards they were only thinking about."""
    d = check_in.analyze(SLUG, paper + "Maybeboard:\n1 Vito, Thorn of the Dusk Rose\n"
                                       "2 Blood Crypt\n")
    assert not d["blocking"], d["blocking"]
    assert d["cards"] == 100 and d["sideboard"] == 0
    assert all(e.get("board", "main") == "main" for e in d["entries"])
    [w] = [w for w in d["warnings"] if "dropped" in w]
    assert w.startswith("3 card(s) after Sideboard:/Maybeboard dropped")
    assert "Commander has no sideboard" in w
    # …and the dropped cards are not in the diff either.
    assert d["add"] == {} and d["pull"] == {}


@requires_deck
@requires_data
def test_a_misremembered_name_is_refused(sandbox, paper):
    """A typo here becomes a card the deck does not have, and `fetch-deck` would
    simply not resolve it and move on."""
    d = check_in.analyze(SLUG, paper.replace("Anguished Unmaking", "Anguished Unmakeing"))
    assert any("no card in the corpus" in b for b in d["blocking"])


@requires_deck
def test_a_list_with_no_commander_is_refused(sandbox, paper):
    text = "\n".join(l for l in paper.split("\n")
                     if "Edgar Markov" not in l and l != "Commander:")
    d = check_in.analyze(SLUG, text)
    assert any("no commander" in b for b in d["blocking"])


@requires_deck
@requires_data
def test_a_changed_commander_warns_rather_than_refuses(sandbox, paper):
    """It is a different deck, and that is the pilot's call — but a new slug is
    almost always what they meant."""
    text = paper.replace("1 Edgar Markov (INR) 234", "1 Vito, Thorn of the Dusk Rose")
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


# ── set_printing: one line, the exact card the pilot sleeves ───────────────
#
# These touch no corpus and no git: a scratch deck directory and the chain
# patched out, so they are the unit tier. The one writer rewrites ONE line.

LIST = ("Commander:\n"
        "1 Edgar Markov (INR) 234\n"
        "\n"
        "Deck:\n"
        "1 Akroma's Will (M3C) 165\n"
        "1 Anguished Unmaking\n"
        "1 Blood Artist (C17) 100 *F*\n"
        "1 Fire // Ice\n"
        "1 Sol Ring\n"
        "87 Swamp\n")


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """A deck directory of our own, the chain recorded rather than run."""
    (tmp_path / "decklist.txt").write_text(LIST, encoding="utf-8")
    monkeypatch.setattr(check_in, "deck_dir", lambda slug, branch=None: tmp_path)
    ran = []
    monkeypatch.setattr(check_in, "_run_chain",
                        lambda slug, branch=None: ran.append((slug, branch)) or ["fetch-deck", "goldfish", "mana-analysis"])
    return tmp_path, ran


def test_set_printing_rewrites_one_line_and_no_other_byte(scratch):
    deck, ran = scratch
    r = check_in.set_printing("cdeck", "Sol Ring", "SLD", "1234")
    assert r == {"changed": True, "line": "1 Sol Ring (SLD) 1234",
                 "ran": ["fetch-deck", "goldfish", "mana-analysis"]}
    after = (deck / "decklist.txt").read_text(encoding="utf-8")
    assert after == LIST.replace("1 Sol Ring\n", "1 Sol Ring (SLD) 1234\n"), (
        "every other line — the headers, the blank, the other printings — is byte-identical")
    assert "1 Blood Artist (C17) 100 *F*" in after, "another line's *F* survives"
    assert ran == [("cdeck", None)], "the chain ran once, on the deck"
    assert (deck / "decklist.txt.bak").read_text(encoding="utf-8") == LIST


def test_set_printing_is_idempotent(scratch):
    deck, ran = scratch
    check_in.set_printing("cdeck", "Sol Ring", "sld", "1234")
    once = (deck / "decklist.txt").read_text(encoding="utf-8")
    again = check_in.set_printing("cdeck", "Sol Ring", "SLD", "1234")
    assert again == {"changed": False, "line": "1 Sol Ring (SLD) 1234", "ran": []}
    assert (deck / "decklist.txt").read_text(encoding="utf-8") == once
    assert len(ran) == 1, "the same printing twice runs nothing the second time"
    # A foil flip on the same printing IS a change: the line gains *F*.
    r = check_in.set_printing("cdeck", "Sol Ring", "sld", "1234", foil=True)
    assert r["changed"] and r["line"] == "1 Sol Ring (SLD) 1234 *F*"
    parsed = {e["name"]: e for e in parse_decklist((deck / "decklist.txt").read_text())}
    assert parsed["Sol Ring"]["foil"] and parsed["Sol Ring"]["set"] == "sld"
    assert parsed["Blood Artist"]["foil"] and parsed["Blood Artist"]["collector_number"] == "100"


def test_set_printing_refuses_a_card_the_list_does_not_hold(scratch):
    deck, ran = scratch
    with pytest.raises(SystemExit, match="not in decklist.txt"):
        check_in.set_printing("cdeck", "Lightning Bolt", "sld", "1")
    with pytest.raises(SystemExit, match="set code and a collector number"):
        check_in.set_printing("cdeck", "Sol Ring", "", "1")
    assert (deck / "decklist.txt").read_text(encoding="utf-8") == LIST and ran == []


def test_set_printing_refuses_a_name_on_two_lines(scratch):
    deck, ran = scratch
    (deck / "decklist.txt").write_text(LIST.replace("87 Swamp\n", "80 Swamp\n7 Swamp (SLD) 9\n"))
    with pytest.raises(SystemExit, match="2 lines"):
        check_in.set_printing("cdeck", "Swamp", "unf", "1")
    assert ran == []


def test_set_printing_matches_a_dfc_by_its_front_face_and_keeps_cmdr(scratch):
    deck, ran = scratch
    r = check_in.set_printing("cdeck", "Fire", "apc", "128")
    assert r["line"] == "1 Fire // Ice (APC) 128"
    assert "1 Fire // Ice (APC) 128\n" in (deck / "decklist.txt").read_text()
    # A `*CMDR*`-marked line keeps its marker after the printing and the foil.
    (deck / "decklist.txt").write_text("1 Zur the Enchanter *CMDR*\n99 Plains\n")
    r = check_in.set_printing("cdeck", "Zur the Enchanter", "sld", "77", foil=True)
    assert r["line"] == "1 Zur the Enchanter (SLD) 77 *F* *CMDR*"
    [cmdr] = [e for e in parse_decklist((deck / "decklist.txt").read_text()) if e["is_commander"]]
    assert cmdr["name"] == "Zur the Enchanter" and cmdr["set"] == "sld" and cmdr["foil"]


def test_set_printing_no_chain_writes_the_line_and_runs_nothing(scratch):
    deck, ran = scratch
    r = check_in.set_printing("cdeck", "Sol Ring", "sld", "1234", run_chain=False)
    assert r["changed"] and r["ran"] == [] and ran == []
    assert "1 Sol Ring (SLD) 1234\n" in (deck / "decklist.txt").read_text()


def test_set_printing_on_a_branch_writes_the_branch_and_runs_its_chain(tmp_path, monkeypatch):
    deck = tmp_path / "d"
    branch = deck / "branches" / "b"
    branch.mkdir(parents=True)
    (deck / "decklist.txt").write_text(LIST)
    (branch / "decklist.txt").write_text(LIST)
    monkeypatch.setattr(check_in, "deck_dir", lambda slug, branch=None: branch and (deck / "branches" / branch) or deck)
    ran = []
    monkeypatch.setattr(check_in, "_run_chain", lambda slug, branch=None: ran.append(branch) or [])
    check_in.set_printing("d", "Sol Ring", "sld", "1", branch="b")
    assert "(SLD) 1" in (branch / "decklist.txt").read_text()
    assert "(SLD) 1" not in (deck / "decklist.txt").read_text(), "the deck's own list is untouched"
    assert ran == ["b"]


def test_the_cli_twin_takes_name_and_printing_and_needs_no_from(scratch, capsys):
    from types import SimpleNamespace
    deck, ran = scratch
    assert check_in.parse_printing_arg("(SLD) 1234") == ("sld", "1234")
    assert check_in.parse_printing_arg("sld 1234") == ("sld", "1234")
    with pytest.raises(SystemExit, match="not a printing"):
        check_in.parse_printing_arg("SLD1234")
    check_in.main(SimpleNamespace(slug="cdeck", source=None, set_printing=["Sol Ring", "(SLD) 1234"],
                                  foil=True, no_chain=True, branch=None))
    out = capsys.readouterr().out
    assert "1 Sol Ring (SLD) 1234 *F*" in out and "chain skipped" in out and "not a new version" in out
    assert ran == []
    check_in.main(SimpleNamespace(slug="cdeck", source=None, set_printing=["Sol Ring", "(SLD) 1234"],
                                  foil=True, no_chain=False, branch=None))
    assert "nothing written, nothing run" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--from"):
        check_in.main(SimpleNamespace(slug="cdeck", source=None, set_printing=None))


def test_a_name_the_deck_already_holds_is_known_even_outside_the_corpus(tmp_path, monkeypatch):
    """ingris-infect's commander is not in the corpus until Reality Fracture
    releases; a branch that keeps her must not be refused on her name."""
    from manamap.pilot import check_in
    monkeypatch.setattr(check_in, "corpus_names", lambda: {"Plains", "Sol Ring"})
    deck = tmp_path / "ghost"
    deck.mkdir()
    (deck / "decklist.txt").write_text("1 Ghost Commander *CMDR*\n98 Plains\n1 Sol Ring\n")
    monkeypatch.setattr(check_in, "deck_dir", lambda slug, branch=None: deck)
    same = check_in.analyze("ghost", "1 Ghost Commander *CMDR*\n98 Plains\n1 Sol Ring\n")
    assert not any("corpus" in b for b in same["blocking"]), same["blocking"]
    typo = check_in.analyze("ghost", "1 Ghost Commander *CMDR*\n97 Plains\n1 Sol Ring\n1 Sol Rong\n")
    assert any("Sol Rong" in b for b in typo["blocking"]), "a real typo is still refused"


# ── The sideboard round-trips (Area C2, 2026-10-09) ──────────────────────────

SIXTY = ("4 Lightning Bolt\n4 Monastery Swiftspear (2X2) 117 *F*\n52 Mountain\n"
         "Sideboard:\n3 Rending Flame\n2 Lightning Bolt\n")


def test_render_decklist_round_trips_every_board():
    """parse(render(entries)) == entries, boards included — and a list with no
    commander renders no `Commander:` header."""
    entries = parse_decklist(SIXTY)
    text = check_in.render_decklist(entries)
    assert not text.startswith("Commander:")
    assert "\nSideboard:\n" in text
    assert parse_decklist(text) == sorted(
        entries, key=lambda e: (e["board"], e["name"]))
    # Idempotent: rendering the rendered form changes nothing.
    assert check_in.render_decklist(parse_decklist(text)) == text
    # A Commander list with no sideboard renders exactly as it always did.
    plain = check_in.render_decklist(parse_decklist(
        "Commander:\n1 Edgar Markov\n\nDeck:\n1 Sol Ring\n"))
    assert plain == "Commander:\n1 Edgar Markov\n\nDeck:\n1 Sol Ring\n"


def test_a_cmdr_marker_in_the_sideboard_survives_the_round_trip():
    """The file keeps saying what was pasted; `validate_deck` is where it is
    called wrong."""
    entries = parse_decklist("1 Sol Ring\nSideboard:\n1 Edgar Markov *CMDR*\n")
    again = parse_decklist(check_in.render_decklist(entries))
    assert again == entries


@pytest.fixture
def brief_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(check_in, "deck_dir", lambda slug, branch=None: tmp_path)
    return tmp_path


def test_set_brief_format_creates_a_minimal_brief(brief_dir):
    import json
    check_in.set_brief_format("cdeck", "standard")
    assert json.loads((brief_dir / "brief.json").read_text()) == {
        "slug": "cdeck", "format": "standard"}


def test_set_brief_format_leaves_an_existing_default_brief_alone(brief_dir):
    """zur-enchantress's brief says `"commander"` and the fourteen tracked
    briefs must not change for a check-in that declares the default."""
    import json
    (brief_dir / "brief.json").write_text(json.dumps({"slug": "cdeck", "commander": "X"}))
    check_in.set_brief_format("cdeck", "commander")
    assert json.loads((brief_dir / "brief.json").read_text()) == {
        "slug": "cdeck", "commander": "X"}
    check_in.set_brief_format("cdeck", "standard")
    assert json.loads((brief_dir / "brief.json").read_text())["format"] == "standard"
    # …and back: a declared Standard moved to Commander is written explicitly.
    check_in.set_brief_format("cdeck", "commander")
    assert json.loads((brief_dir / "brief.json").read_text())["format"] == "commander"


def test_set_brief_format_refuses_an_unknown_name(brief_dir):
    with pytest.raises(SystemExit):
        check_in.set_brief_format("cdeck", "pendragon")
    assert not (brief_dir / "brief.json").exists()


def test_a_dry_run_with_a_format_names_it_and_writes_no_brief(brief_dir, capsys):
    (brief_dir / "decklist.txt").write_text("1 Sol Ring\n")
    (brief_dir / "paper.txt").write_text(SIXTY)
    args = argparse.Namespace(slug="cdeck", source=str(brief_dir / "paper.txt"),
                              set_printing=None, as_json=False, write=False,
                              force=False, no_chain=True, format="standard")
    check_in.main(args)
    out = capsys.readouterr().out
    assert "format: Standard" in out and "on --write" in out
    assert not (brief_dir / "brief.json").exists()
    assert (brief_dir / "decklist.txt").read_text() == "1 Sol Ring\n"


def test_fetch_deck_and_check_in_take_the_same_format_choices():
    """The registry row, not a test re-deriving it: both commands accept every
    name in `formats.FORMATS` and nothing else, default None (the resolver)."""
    from manamap.pilot import formats
    from manamap.pilot.registry import add_pilot_parser

    parser = argparse.ArgumentParser()
    add_pilot_parser(parser.add_subparsers(dest="command"))
    for cmd in ("fetch-deck", "check-in"):
        ns = parser.parse_args(["pilot", cmd, "x", "--format", "standard"])
        assert ns.format == "standard"
        assert parser.parse_args(["pilot", cmd, "x"]).format is None
        with pytest.raises(SystemExit):
            parser.parse_args(["pilot", cmd, "x", "--format", "pendragon"])
    # validate-deck's pre-existing flag still lists the same names.
    for name in formats.FORMATS:
        for cmd in ("fetch-deck", "check-in", "validate-deck"):
            assert parser.parse_args(["pilot", cmd, "x", "--format", name]).format == name


# ── A 60-card deck is checked in per ITS format (Area C3, 2026-10-09) ────────
#
# The sandbox is a Modern deck: a brief declaring the format, a canonical
# 60 + 15 on disk, `config.DECKS_DIR` pointed at it so `formats.for_deck`
# resolves the spec the way the command does. Real names, so the corpus check
# is exercised honestly where a corpus exists and warns where it does not.

MODERN_MAIN = ("4 Lightning Bolt\n4 Monastery Swiftspear\n52 Mountain\n")
MODERN_SIDE = ("Sideboard:\n4 Smash to Smithereens\n4 Rending Volley\n"
               "4 Blood Moon\n3 Relic of Progenitus\n")
MODERN = MODERN_MAIN + MODERN_SIDE


@pytest.fixture
def modern(tmp_path, monkeypatch):
    import json
    decks = tmp_path / "decks"
    deck = decks / "md"
    deck.mkdir(parents=True)
    (deck / "brief.json").write_text(json.dumps({"slug": "md", "format": "modern"}))
    (deck / "decklist.txt").write_text(check_in.render_decklist(parse_decklist(MODERN)))
    monkeypatch.setattr("manamap.config.DECKS_DIR", decks)
    return deck


def _corpus_free(d):
    """The refusals that are not about the corpus (absent on a fresh clone)."""
    return [b for b in d["blocking"] if "corpus" not in b]


def test_a_modern_deck_is_sized_and_counted_by_its_own_spec(modern):
    d = check_in.analyze("md", MODERN)
    assert d["format"] == "modern"
    assert d["cards"] == 60 and d["sideboard"] == 15
    assert _corpus_free(d) == []
    assert not any("commander" in b for b in d["blocking"]), "no commander is not a defect here"
    assert d["pull"] == {} and d["add"] == {} and not d["sideboard_changed"]


def test_four_copies_pass_and_five_block_counted_across_both_boards(modern):
    """CR 100.4a: the limit is main and sideboard TOGETHER. Four Bolts in the
    sixty and one more in the fifteen is five."""
    four = check_in.analyze("md", MODERN)
    assert not any("copies" in b for b in four["blocking"])
    five = check_in.analyze("md", MODERN.replace("3 Relic of Progenitus\n",
                                                  "2 Relic of Progenitus\n1 Lightning Bolt\n"))
    [b] = [b for b in five["blocking"] if "more than 4 copies" in b]
    assert "Lightning Bolt x5" in b and "CR 100.4a" in b
    assert "singleton" not in b


def test_fifty_nine_cards_block_and_sixty_three_do_not(modern):
    """"At least sixty": a 63-card Modern deck is legal."""
    short = check_in.analyze("md", MODERN.replace("52 Mountain", "51 Mountain"))
    [b] = [b for b in short["blocking"] if "at least 60" in b]
    assert b.startswith("Deck has 59 cards")
    assert "(Modern)" in b
    long = check_in.analyze("md", MODERN.replace("52 Mountain", "55 Mountain"))
    assert not any("at least" in b or "exactly" in b for b in long["blocking"])


def test_sixteen_sideboard_cards_block(modern):
    d = check_in.analyze("md", MODERN.replace("3 Relic of Progenitus", "4 Relic of Progenitus"))
    [b] = [b for b in d["blocking"] if "sideboard" in b]
    assert b.startswith("16 sideboard cards") and "at most 15" in b
    assert d["sideboard"] == 16


def test_apply_round_trips_the_sideboard_block(modern):
    """A check-in that lost the fifteen would be a silent edit of the deck."""
    text = MODERN.replace("4 Monastery Swiftspear", "4 Goblin Guide")
    d = check_in.analyze("md", text)
    assert d["pull"] == {"Monastery Swiftspear": 4} and d["add"] == {"Goblin Guide": 4}
    r = check_in.apply("md", d["entries"], run_chain=False)
    written = (modern / "decklist.txt").read_text()
    block = written[written.index("Sideboard:"):]
    assert block == check_in.render_decklist(parse_decklist(MODERN_SIDE)).split(
        "Deck:\n\n", 1)[1]
    again = check_in.analyze("md", written)
    assert again["pull"] == {} and again["add"] == {} and not again["sideboard_changed"]
    assert again["cards"] == 60 and again["sideboard"] == 15
    # The chain's plan for a format with no goldfish, said rather than implied.
    assert r["ran"] == []
    assert set(r["skipped"]) == {"goldfish"}
    assert "Commander-only" in r["skipped"]["goldfish"] and "Modern" in r["skipped"]["goldfish"]


def test_a_sideboard_only_edit_is_reported_and_is_not_a_version(modern):
    d = check_in.analyze("md", MODERN.replace("4 Blood Moon", "4 Alpine Moon"))
    assert d["pull"] == {} and d["add"] == {}
    assert d["sideboard_changed"] is True


def test_the_chain_plan_reads_the_spec():
    from manamap.pilot import formats
    assert check_in.chain_plan(formats.COMMANDER) == (
        ["fetch-deck", "goldfish", "mana-analysis"], {})
    stages, skipped = check_in.chain_plan(formats.STANDARD)
    assert stages == ["fetch-deck", "mana-analysis"]
    assert skipped == {"goldfish": "not modelled for Standard — the goldfish is "
                                   "Commander-only (docs/simulation.md)"}


def test_the_chain_runs_the_format_plan(modern, monkeypatch):
    """`_run_chain` runs what `chain_plan` says, resolved from the deck."""
    import importlib
    ran = []

    class _Mod:
        def __init__(self, name):
            self.name = name

        def main(self, args):
            ran.append((self.name, args.slug, args.branch))

    monkeypatch.setattr(importlib, "import_module", lambda dotted: _Mod(dotted.rsplit(".", 1)[1]))
    assert check_in._run_chain("md") == ["fetch-deck", "mana-analysis"]
    assert ran == [("fetch_deck", "md", None), ("mana_analysis", "md", None)]


def test_the_header_names_the_format_and_the_sideboard(modern, capsys):
    check_in._print(check_in.analyze("md", MODERN), write=False)
    out = capsys.readouterr().out
    assert "CHECK-IN — md  (60 cards + 15 side, Modern)" in out
    assert "commander" not in out.split("\n")[0]


def test_a_new_deck_declared_on_the_command_line_is_analysed_as_that_format(
        tmp_path, monkeypatch, capsys):
    """`--format modern` on a deck with no brief: the spec is threaded from
    `main` so the first check-in is held to Modern, not to a hundred."""
    decks = tmp_path / "decks"
    (decks / "fresh").mkdir(parents=True)
    monkeypatch.setattr("manamap.config.DECKS_DIR", decks)
    (tmp_path / "paper.txt").write_text(MODERN)
    args = argparse.Namespace(slug="fresh", source=str(tmp_path / "paper.txt"),
                              set_printing=None, as_json=False, write=False,
                              force=False, no_chain=True, format="modern")
    check_in.main(args)
    out = capsys.readouterr().out
    assert "(60 cards + 15 side, Modern)" in out
    assert "expected exactly 100" not in out and "no commander" not in out
    assert "format: Modern" in out
