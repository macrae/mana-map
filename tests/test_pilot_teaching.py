"""The teach decks' shelf: `brief.json`'s `purpose`, the manifest key, the nudge it stops.

A teaching deck is cardboard the pilot still tunes and is not in the fleet.
"""
import json

import pytest

from manamap import config
from manamap.pilot import common, deck_edit, fleet_analytics, regen
from manamap.pilot.validate_brief import PURPOSES, validate

TEACH = [f"teach-{c}" for c in ("white", "blue", "black", "red", "green")]


@pytest.fixture
def decks(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DECKS_DIR", tmp_path)
    for slug, brief in (("t", {"slug": "t", "format": "modern", "purpose": "teaching"}),
                        ("p", {"slug": "p", "format": "modern"})):
        (tmp_path / slug).mkdir()
        (tmp_path / slug / "brief.json").write_text(json.dumps(brief))
    (tmp_path / "none").mkdir()
    return tmp_path


def test_deck_purpose_reads_the_brief_and_absent_means_absent(decks):
    assert common.deck_purpose("t") == "teaching"
    assert common.deck_purpose("p") is None
    assert common.deck_purpose("none") is None


def test_the_validator_refuses_a_misspelt_purpose():
    assert PURPOSES == {"teaching"}
    bad, _ = validate({"slug": "t", "format": "modern", "purpose": "teachng"}, "t", rows=[], names=set())
    assert any("purpose" in e for e in bad)
    good, _ = validate({"slug": "t", "format": "modern", "purpose": "teaching"}, "t", rows=[], names=set())
    assert not any("purpose" in e for e in good)


@pytest.mark.skipif(not (config.DECKS_DIR / "index.json").exists(), reason="no tracked manifest")
def test_only_the_five_teach_decks_carry_the_purpose():
    manifest = json.loads((config.DECKS_DIR / "index.json").read_text(encoding="utf-8"))
    marked = sorted(d["slug"] for d in manifest["decks"] if "purpose" in d)
    assert marked == sorted(TEACH)
    assert all(d["purpose"] == "teaching" for d in manifest["decks"] if "purpose" in d)


@pytest.mark.skipif(not (config.DECKS_DIR / "index.json").exists(), reason="no tracked manifest")
@pytest.mark.parametrize("slug", TEACH)
def test_a_teach_deck_stays_editable_out_of_the_fleet_and_unnagged(slug):
    assert deck_edit.guard(slug)                       # in-place editing is the point
    assert not regen.is_pinned(slug)                   # so nothing bootstraps a handbook
    assert not fleet_analytics.is_member(fleet_analytics.row(slug))
    info = json.loads((config.DECKS_DIR / slug / "info.json").read_text(encoding="utf-8"))
    assert not any("not marked as built in paper" in n for n in info["next"])


@pytest.mark.skipif(not (config.DECKS_DIR / "index.json").exists(), reason="no tracked manifest")
def test_a_deck_with_no_purpose_is_still_asked_about_paper():
    manifest = json.loads((config.DECKS_DIR / "index.json").read_text(encoding="utf-8"))
    others = [d["slug"] for d in manifest["decks"]
              if "purpose" not in d and not d.get("locked") and not d.get("status")]
    asked = 0
    for slug in others:
        info = json.loads((config.DECKS_DIR / slug / "info.json").read_text(encoding="utf-8"))
        asked += any("not marked as built in paper" in n for n in info.get("next", []))
    assert asked >= 1
