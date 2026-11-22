"""Reading Forge's card scripts in bulk (2026-11-22, the engine graph's first commit).

`script()` reopened the archive per call and consulted a `Name:` index that
decompresses all 33,619 members on EVERY call, so a 1,235-card survey took 222 s.
`scripts(names)` opens the zip once, tries the filename stems first, and asks the
index only for the names the stems missed; the index is kept on disk under the
archive's content digest. These tests build a tiny archive, so the unit tier needs
no Forge.
"""
import json
import zipfile

import pytest

from manamap.sim import forge_cards

SOL = "Name:Sol Ring\nManaCost:1\nTypes:Artifact\nA:AB$ Mana | Cost$ T | Produced$ C | Amount$ 2\n"
ARTIST = ("Name:Blood Artist\nManaCost:1 B\nTypes:Creature Vampire\nPT:0/1\n"
          "T:Mode$ ChangesZone | Origin$ Battlefield | Destination$ Graveyard | "
          "ValidCard$ Card.Self,Creature.Other | Execute$ TrigLoseLife\n")
FALLAJI = "Name:Fallaji Archaeologist\nManaCost:1 U\nTypes:Creature Human Artificer\nPT:1/3\n"


def _forge(tmp_path, monkeypatch, members):
    home = tmp_path / "forge"
    folder = home / "res" / "cardsfolder"
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / "cardsfolder.zip", "w") as z:
        for member, text in members.items():
            z.writestr(member, text)
    (home / "res" / "tokenscripts").mkdir()
    (home / "res" / "tokenscripts" / "b_1_1_vampire.txt").write_text(
        "Name:Vampire Token\nColors:black\nTypes:Creature Vampire\nPT:1/1\n", encoding="utf-8")
    monkeypatch.setattr(forge_cards, "FORGE_HOME", home)
    forge_cards._name_index.cache_clear()
    forge_cards._digest_of.cache_clear()
    forge_cards._flags.cache_clear()
    return home


class _Counting:
    def __init__(self):
        self.opens = 0
        self._real = zipfile.ZipFile

    def __call__(self, *a, **k):
        self.opens += 1
        return self._real(*a, **k)


def test_scripts_opens_the_zip_once(tmp_path, monkeypatch):
    _forge(tmp_path, monkeypatch, {"s/sol_ring.txt": SOL, "b/blood_artist.txt": ARTIST})
    counter = _Counting()
    monkeypatch.setattr(forge_cards.zipfile, "ZipFile", counter)
    got = forge_cards.scripts(["Sol Ring", "Blood Artist", "Sol Ring"])
    assert set(got) == {"Sol Ring", "Blood Artist"}
    assert got["Sol Ring"].startswith("Name:Sol Ring") and "ChangesZone" in got["Blood Artist"]
    # One open for two names found by their stems, and no index built for a hit.
    assert counter.opens == 1, counter.opens
    assert not list((tmp_path / "forge").glob(".name_index-*.json"))


def test_a_misspelled_stem_is_found_through_the_cached_index(tmp_path, monkeypatch):
    """Forge files Fallaji Archaeologist as `fallaji_archeologist.txt`. The stem
    misses; the `Name:` index finds it, is written to disk under the archive's
    digest, and is READ from disk next time rather than rebuilt."""
    home = _forge(tmp_path, monkeypatch, {"f/fallaji_archeologist.txt": FALLAJI, "s/sol_ring.txt": SOL})
    counter = _Counting()
    monkeypatch.setattr(forge_cards.zipfile, "ZipFile", counter)
    got = forge_cards.scripts(["Fallaji Archaeologist", "Not A Real Card At All"])
    assert got["Fallaji Archaeologist"].startswith("Name:Fallaji")
    assert got["Not A Real Card At All"] is None
    caches = list(home.glob(".name_index-*.json"))
    assert len(caches) == 1 and json.loads(caches[0].read_text())["Fallaji Archaeologist"] == "f/fallaji_archeologist.txt"
    first = counter.opens
    # The lookup, the digest (the cache's name) and the index build: three opens,
    # and the last two happen only on a miss.
    assert first == 3, first
    # A fresh process: the lru cache is gone, the disk cache is not.
    forge_cards._name_index.cache_clear()
    forge_cards.scripts(["Fallaji Archaeologist"])
    assert counter.opens == first + 1, "the second lookup rebuilt the index instead of reading it"
    # Drop the disk cache's fallback (reintroduce the bug): a miss with no index stays a miss.
    monkeypatch.setattr(forge_cards, "_name_index", lambda: {})
    assert forge_cards.scripts(["Fallaji Archaeologist"])["Fallaji Archaeologist"] is None


def test_the_digest_moves_when_a_member_changes_and_not_when_the_zip_is_touched(tmp_path, monkeypatch):
    home = _forge(tmp_path, monkeypatch, {"s/sol_ring.txt": SOL, "b/blood_artist.txt": ARTIST})
    d1 = forge_cards.cardsfolder_digest()
    assert d1 and len(d1) == 16
    # Re-written with the same contents: new mtime and new zip bytes, same digest.
    forge_cards._digest_of.cache_clear()
    with zipfile.ZipFile(home / "res" / "cardsfolder" / "cardsfolder.zip", "w") as z:
        z.writestr("b/blood_artist.txt", ARTIST)
        z.writestr("s/sol_ring.txt", SOL)
    assert forge_cards.cardsfolder_digest() == d1
    # One member's text changes: the digest moves.
    forge_cards._digest_of.cache_clear()
    with zipfile.ZipFile(home / "res" / "cardsfolder" / "cardsfolder.zip", "w") as z:
        z.writestr("b/blood_artist.txt", ARTIST.replace("Graveyard", "Exile"))
        z.writestr("s/sol_ring.txt", SOL)
    assert forge_cards.cardsfolder_digest() != d1


def test_a_token_script_is_read_from_the_token_folder_and_is_absent_otherwise(tmp_path, monkeypatch):
    _forge(tmp_path, monkeypatch, {"s/sol_ring.txt": SOL})
    text = forge_cards.token_script("b_1_1_vampire")
    assert "Types:Creature Vampire" in text and "Colors:black" in text
    assert forge_cards.token_script("no_such_token") is None


def test_script_is_the_single_name_case_of_scripts(tmp_path, monkeypatch):
    _forge(tmp_path, monkeypatch, {"s/sol_ring.txt": SOL})
    assert forge_cards.script("Sol Ring") == forge_cards.scripts(["Sol Ring"])["Sol Ring"]
    assert forge_cards.script("Nothing Here") is None


def test_no_forge_is_none_for_every_name(tmp_path, monkeypatch):
    monkeypatch.setattr(forge_cards, "FORGE_HOME", tmp_path / "absent")
    forge_cards._name_index.cache_clear()
    forge_cards._digest_of.cache_clear()
    assert forge_cards.scripts(["Sol Ring"]) == {"Sol Ring": None}
    assert forge_cards.cardsfolder_digest() is None
