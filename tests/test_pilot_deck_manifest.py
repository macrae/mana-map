"""The deck manifest — the contract between the pilot artifacts and viz/deck.html.

LEGACY (2026-08-19): the magazine renderer. It still renders the nine frozen issues from
artifacts nothing regenerates any more (issue_plan.json, the panel keys,
card_roles/mana_base/upgrades, considering.json), and it is replaced by the compact deck
page in docs/manual-v5-spec.md. Do not extend it; internals below are accurate for what it
does.

A browser can list neither the deck directory nor `stacks/`, so `build-index`
writes `data/decks/index.json` from the same scan that builds the newsstand. If
that manifest drifts from the artifacts, the dossier renders a stale or empty
page — silently, because a 404 on a stack file is indistinguishable from a deck
that has none. These tests are that guard.
"""

import json

import pytest
from conftest import requires_deck

from manamap.config import DECKS_DIR
from manamap.pilot import build_index
from manamap.pilot.common import checker_passed, load_json


def test_manifest_carries_the_fields_the_dossier_reads():
    entries = [{
        "slug": "x", "volume": 1, "deck_name": "X", "commander": "C",
        "coverline": "L", "verified": 2, "decisions": 1,
        "stack_files": ["001-a.json"], "image": None, "issue_date": "", "mean_cast": None,
    }]
    manifest = {"decks": [
        {k: e[k] for k in ("slug", "volume", "deck_name", "commander",
                           "coverline", "verified", "decisions", "stack_files")}
        for e in entries
    ]}
    # The dossier destructures exactly these keys; a rename breaks the page.
    assert set(manifest["decks"][0]) == {
        "slug", "volume", "deck_name", "commander", "coverline",
        "verified", "decisions", "stack_files"}


@requires_deck
def test_tracked_manifest_matches_the_artifacts_on_disk():
    path = DECKS_DIR / "index.json"
    if not path.exists():
        pytest.skip("no manifest yet — run `manamap pilot build-index`")
    manifest = json.loads(path.read_text())

    fresh = {e["slug"]: e for e in build_index.gather_entries()}
    listed = {d["slug"]: d for d in manifest["decks"]}
    assert set(listed) == set(fresh), (
        "manifest deck set is stale — re-run `manamap pilot build-index`")

    for slug, deck in listed.items():
        assert deck["verified"] == fresh[slug]["verified"], slug
        assert deck["stack_files"] == fresh[slug]["stack_files"], slug
        for name in deck["stack_files"]:
            stack_path = DECKS_DIR / slug / "stacks" / name
            assert stack_path.exists(), f"{slug}: manifest names a missing {name}"
            # Only passing stacks publish — the dossier must not be handed a
            # failed artifact to render as fact.
            assert checker_passed(load_json(stack_path, {})), (
                f"{slug}: {name} is in the manifest but is not checker-passed")
        assert len(deck["stack_files"]) == deck["verified"], slug
