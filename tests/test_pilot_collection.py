"""The pilot's collection: one reader, memoized, and one definition of "owned".

`COLLECTION_DIR` had exactly one dereference in the whole repo while two different
parsers answered "does he own this" differently over the same nine files. These pin
the semantics that settles it and the memo that stops it being re-parsed per call.
"""

import json

from manamap.pilot import collection as coll
from manamap.pilot.common import clear_memo


def _box(tmp_path, monkeypatch, files, decks=None):
    """A fake COLLECTION_DIR (and optionally DECKS_DIR) with the memo cleared."""
    cdir = tmp_path / "collection"
    cdir.mkdir()
    for name, text in files.items():
        (cdir / name).write_text(text)
    monkeypatch.setattr(coll, "COLLECTION_DIR", cdir)
    ddir = tmp_path / "decks"
    ddir.mkdir()
    for slug, names in (decks or {}).items():
        (ddir / slug).mkdir()
        (ddir / slug / "cards.json").write_text(json.dumps(
            {"deck": slug, "cards": [{"name": n} for n in names]}))
    monkeypatch.setattr(coll, "DECKS_DIR", ddir)
    clear_memo()
    return cdir, ddir


def test_a_card_in_a_deck_is_owned_but_has_no_box(tmp_path, monkeypatch):
    """Owning a card is not the same as it being on a shelf.

    kinnan's recon marked Drover of the Mighty owned while the box did not hold it,
    because it is sleeved into a tracked deck — and it was right. You would have to
    unsleeve it, which is a decision, not a purchase. So the default sense is the
    union, the narrow sense is opt-out, and only the box sense can name a source.
    """
    _box(tmp_path, monkeypatch, {"Green.txt": "1 Llanowar Elves\n"},
         decks={"kinnan": ["Drover of the Mighty"]})
    assert coll.owns("Llanowar Elves") and coll.owns("Drover of the Mighty")
    assert coll.owns("Llanowar Elves", include_decks=False)
    assert not coll.owns("Drover of the Mighty", include_decks=False)
    assert coll.sources_for("Llanowar Elves") == {"Green"}
    assert coll.sources_for("Drover of the Mighty") == set(), \
        "a card that is in no box has no box to name"


def test_both_faces_answer_yes(tmp_path, monkeypatch):
    """A decklist may name either face; an index keyed only on the joined form says
    no to every DFC."""
    _box(tmp_path, monkeypatch,
         {"Green.txt": "1 Bala Ged Recovery // Bala Ged Sanctuary\n"})
    for name in ("Bala Ged Recovery // Bala Ged Sanctuary",
                 "Bala Ged Recovery", "Bala Ged Sanctuary"):
        assert coll.owns(name), name


def test_an_absent_collection_is_not_an_error(tmp_path, monkeypatch):
    """No collection means no ownership claim — the contract COLLECTION_DIR is
    declared under. A fresh clone must not raise."""
    monkeypatch.setattr(coll, "COLLECTION_DIR", tmp_path / "nope")
    monkeypatch.setattr(coll, "DECKS_DIR", tmp_path / "also-nope")
    clear_memo()
    assert coll.owned_index() == {} and coll.owned_names() == set()
    assert coll.owns("Sol Ring") is False


def test_the_memo_notices_a_NEW_box_not_just_an_edited_one(tmp_path, monkeypatch):
    """The reason this does not use `mtime_memo`: that keys on a single path, and a
    new `.txt` appearing changes the answer while every existing file is untouched.
    The signature covers the directory listing."""
    cdir, _ = _box(tmp_path, monkeypatch, {"Green.txt": "1 Llanowar Elves\n"})
    assert not coll.owns("Sol Ring")
    (cdir / "Artifacts.txt").write_text("1 Sol Ring\n")
    assert coll.owns("Sol Ring"), "a box added after the first read must be seen"


def test_the_memo_notices_an_edited_box(tmp_path, monkeypatch):
    cdir, _ = _box(tmp_path, monkeypatch, {"Green.txt": "1 Llanowar Elves\n"})
    assert not coll.owns("Birds of Paradise")
    (cdir / "Green.txt").write_text("1 Llanowar Elves\n1 Birds of Paradise\n")
    assert coll.owns("Birds of Paradise")


def test_a_malformed_box_file_is_skipped_not_fatal(tmp_path, monkeypatch):
    """One unreadable box must not take the whole ownership answer down."""
    _box(tmp_path, monkeypatch,
         {"Green.txt": "1 Llanowar Elves\n", "Junk.txt": "\x00\x00 not a decklist\n"})
    assert coll.owns("Llanowar Elves")


def test_deck_history_uses_the_shared_reader_and_keeps_the_box_sense(tmp_path, monkeypatch):
    """`pending()` reports which box to pull a card from, so it must keep the BOX
    sense — a card sleeved in another deck has no source file to name, and an
    ownership claim nobody can source is not evidence."""
    from manamap.pilot import deck_history
    _box(tmp_path, monkeypatch, {"Green.txt": "1 Llanowar Elves\n"},
         decks={"other": ["Sol Ring"]})
    index = deck_history._owned_index()
    assert index.get("Llanowar Elves") == {"Green"}
    assert "Sol Ring" not in index, "deck membership is not a box"
