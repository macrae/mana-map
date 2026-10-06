"""Two ways `try` screened something other than what it named (2026-10-05).

Both were found by the edgar doctor on its third pass, and both made a figure
the pilot was shown wrong:

1. THE STAMP READ THE DISK, NOT THE CODE. `goldfish.model_version()` re-hashed
   the simulator's files on every call, so `manamap serve`'s warm worker —
   running code loaded before an edit — stamped its champion with the AFTER
   version. `try` cached it under that key, and every later screen subtracted a
   pre-fix baseline: Edgar's package read draw +0.031 against a true +0.151.

2. THE FIRST NAME IN THE DUMP WON. Art-series cards carry a real card's name
   and no oracle text, so Phyrexian Arena, Yawgmoth and Sorin screened as blank.
"""
import json
import pathlib

from manamap import config
from manamap.pilot import goldfish, try_swap


def test_the_model_version_is_the_code_this_process_loaded(monkeypatch):
    """After import, the stamp never touches the disk — so a process cannot
    report a version it is not running. Re-introduce the per-call hash and
    this raises."""
    before = goldfish.model_version()

    def no_disk(self):
        raise AssertionError(f"model_version read {self} after import")

    monkeypatch.setattr(pathlib.Path, "read_bytes", no_disk)
    assert goldfish.model_version() == before


def test_an_art_series_card_is_never_screened_as_the_card(tmp_path, monkeypatch):
    dump = tmp_path / "dump.jsonl"
    dump.write_text("\n".join(json.dumps(o) for o in (
        {"name": "Phyrexian Arena // Phyrexian Arena", "layout": "art_series"},
        {"name": "Phyrexian Arena", "layout": "token", "oracle_text": ""},
        {"name": "Phyrexian Arena", "layout": "normal", "type_line": "Enchantment",
         "oracle_text": "At the beginning of your upkeep, you draw a card and you lose 1 life."},
    )) + "\n")
    monkeypatch.setattr(config, "RAW_JSON_PATH", dump)
    monkeypatch.setattr(try_swap, "_SCRYFALL", {})
    got = try_swap._scryfall_objects(["Phyrexian Arena"])["phyrexian arena"]
    assert got["layout"] == "normal" and "draw a card" in got["oracle_text"]
