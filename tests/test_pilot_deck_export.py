"""deck-export: a list as the text another tool's import box takes.

Moxfield has no API and 403s server-side traffic, so publishing is a paste, and the
paste has to be right without edits. Pinned here: the moxfield form IS
`render_decklist` and parses back to the deck's own entries (boards, commander,
printings, foils); the arena form has no headers and one blank line before the
sideboard; plain keeps sections and drops printings; `--version` reads a past list
out of git; `--branch` reads a branch; `--out` is slug-scoped.
"""

import argparse
import subprocess

import pytest

from manamap.pilot import check_in, deck_export
from manamap.pilot import deck_history as dh
from manamap.pilot.fetch_deck import parse_decklist

CMDR = """Commander:
1 Edgar Markov (INR) 234 *F*

Deck:
1 Blood Artist
1 Sol Ring (C21) 263
30 Swamp
"""

SIXTY = """Deck:
4 Llanowar Elves (DOM) 168
4 Elvish Mystic
20 Forest

Sideboard:
2 Thoughtseize (THS) 107 *F*
1 Choke
"""


def _deck(root, slug, text):
    d = root / slug
    d.mkdir(parents=True)
    (d / "decklist.txt").write_text(text)
    return d


@pytest.fixture
def decks(tmp_path, monkeypatch):
    root = tmp_path / "data" / "decks"
    monkeypatch.setattr("manamap.config.DECKS_DIR", root)
    _deck(root, "cdeck", CMDR)
    _deck(root, "sdeck", SIXTY)
    return root


def _key(entries):
    return sorted((e["name"], e["quantity"], e.get("board"), bool(e.get("is_commander")),
                   e.get("set"), e.get("collector_number"), bool(e.get("foil")))
                  for e in entries)


@pytest.mark.parametrize("slug,text", [("cdeck", CMDR), ("sdeck", SIXTY)])
def test_the_moxfield_form_round_trips_to_the_decks_own_entries(decks, slug, text):
    out, label = deck_export.export(slug, fmt="moxfield")
    assert label == slug
    assert out == check_in.render_decklist(parse_decklist(text))
    assert _key(parse_decklist(out)) == _key(parse_decklist(text))


def test_moxfield_carries_headers_printings_and_foils(decks):
    out, _ = deck_export.export("cdeck", fmt="moxfield")
    assert out.startswith("Commander:\n1 Edgar Markov (INR) 234 *F*\n\nDeck:\n")
    assert "1 Sol Ring (C21) 263\n" in out
    side, _ = deck_export.export("sdeck", fmt="moxfield")
    assert "\nSideboard:\n1 Choke\n2 Thoughtseize (THS) 107 *F*\n" in side


def test_arena_has_no_headers_and_a_blank_line_before_the_sideboard(decks):
    out, _ = deck_export.export("sdeck", fmt="arena")
    assert out == ("4 Elvish Mystic\n20 Forest\n4 Llanowar Elves (DOM) 168\n"
                   "\n1 Choke\n2 Thoughtseize (THS) 107\n")
    cmd, _ = deck_export.export("cdeck", fmt="arena")
    assert cmd.splitlines()[0] == "1 Edgar Markov (INR) 234", "the commander first, no foil"
    assert "Commander" not in cmd and "Deck:" not in cmd and "\n\n" not in cmd
    assert "*F*" not in cmd


def test_plain_keeps_sections_and_drops_printings(decks):
    out, _ = deck_export.export("sdeck", fmt="plain")
    assert out == ("Deck:\n4 Elvish Mystic\n20 Forest\n4 Llanowar Elves\n"
                   "\nSideboard:\n1 Choke\n2 Thoughtseize\n")
    cmd, _ = deck_export.export("cdeck", fmt="plain")
    assert cmd.startswith("Commander:\n1 Edgar Markov\n\nDeck:\n")
    assert "(" not in cmd and "*F*" not in cmd
    # Plain still parses back to the same cards and boards.
    strip = lambda es: sorted((e["name"], e["quantity"], e["board"], bool(e["is_commander"]))  # noqa: E731
                              for e in es)
    assert strip(parse_decklist(out)) == strip(parse_decklist(SIXTY))


def test_a_branch_is_read_from_its_own_list(decks):
    b = decks / "cdeck" / "branches" / "try-1"
    b.mkdir(parents=True)
    (b / "decklist.txt").write_text(CMDR.replace("1 Blood Artist", "1 Cruel Celebrant"))
    out, label = deck_export.export("cdeck", fmt="plain", branch="try-1")
    assert label == "cdeck@try-1" and "Cruel Celebrant" in out and "Blood Artist" not in out


def test_version_and_branch_together_is_refused(decks):
    with pytest.raises(SystemExit, match="one, not both"):
        deck_export.export("cdeck", version="V1", branch="try-1")


def test_an_unknown_format_is_refused(decks):
    with pytest.raises(SystemExit, match="unknown format"):
        deck_export.export("cdeck", fmt="mtgo")


def test_out_is_slug_scoped(decks, tmp_path, capsys):
    args = argparse.Namespace(slug="cdeck", format="arena", version=None, branch=None,
                              out=str(tmp_path / "list.txt"))
    with pytest.raises(SystemExit, match="does not contain the slug"):
        deck_export.main(args)
    outdir = tmp_path / "views"
    outdir.mkdir()
    args.out = str(outdir)
    deck_export.main(args)
    written = outdir / "deck-export-arena-cdeck.txt"
    assert written.read_text().startswith("1 Edgar Markov (INR) 234\n")


def test_main_prints_the_text(decks, capsys):
    deck_export.main(argparse.Namespace(slug="sdeck", format=None, version=None,
                                        branch=None, out=None))
    assert capsys.readouterr().out == check_in.render_decklist(parse_decklist(SIXTY))


# ── --version: a past list out of git ──────────────────────────────────────

def _run(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                        "HOME": str(root), "PATH": "/usr/bin:/bin:/usr/local/bin"})


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path
    deck = root / "data" / "decks" / "vdeck"
    deck.mkdir(parents=True)
    monkeypatch.setattr("manamap.config.DECKS_DIR", root / "data" / "decks")
    monkeypatch.setattr(dh, "_REPO_ROOT", root)
    _run(root, "init", "-q")
    (deck / "decklist.txt").write_text(CMDR)
    _run(root, "add", "."); _run(root, "commit", "-q", "-m", "V1")
    (deck / "decklist.txt").write_text(CMDR.replace("1 Blood Artist", "1 Cruel Celebrant"))
    _run(root, "add", "."); _run(root, "commit", "-q", "-m", "V2")
    return deck


def test_a_version_is_read_out_of_git_not_off_disk(repo):
    old, label = deck_export.export("vdeck", fmt="moxfield", version="V1")
    assert label == "vdeck V1"
    assert "Blood Artist" in old and "Cruel Celebrant" not in old
    assert old == check_in.render_decklist(parse_decklist(CMDR))
    now, _ = deck_export.export("vdeck", fmt="moxfield")
    assert "Cruel Celebrant" in now


def test_an_unknown_version_is_refused(repo):
    with pytest.raises(SystemExit, match="no version"):
        deck_export.export("vdeck", version="V9")
