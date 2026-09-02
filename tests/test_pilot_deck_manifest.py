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


# The dossier destructures exactly these; a rename breaks the page silently,
# because `getJSON` swallows every failure to null and an absent key reads the
# same as an absent artifact.
MANIFEST_KEYS = {
    "slug", "volume", "deck_name", "commander", "coverline", "verified",
    "decisions", "stack_files", "stack_cards", "published", "status",
    "sim_runs", "experiments", "prescriptions", "decision_files", "has",
    # The workbench landing (viz/workbench.html): `image` is the rack's art and
    # `locked`/`paper` are the predicate it filters on — which deck is built in
    # paper and playable at a table tonight. `paper` is null on an unlocked deck
    # rather than absent, so the browser never has to distinguish "no lock" from
    # "field not in this manifest".
    "image", "paper", "locked",
    # `version` is the latest RELEASE TAG, carried for every deck rather than
    # only the sleeved ones, because the card stamps it over the commander art.
    # `paper` cannot serve that: it is null on an unlocked deck, and `info.json`
    # deliberately strips its own `version` block (a git walk, one commit behind
    # forever). The PIN on the stamp still means sleeved and nothing else.
    "version",
    # WHETHER THIS DECK MAY BE DELETED, decided once by
    # `deck_delete.blockers`. If the page re-derived "never sleeved, never
    # played, never published" from `locked` and `record.games` it would be a
    # second implementation of the refusal, free to disagree with the command's
    # — and the disagreement would only ever surface as a button that offers to
    # delete something and is then refused.
    "deletable", "undeletable_because",
}


@requires_deck
def test_manifest_carries_the_fields_the_dossier_reads():
    path = DECKS_DIR / "index.json"
    if not path.exists():
        pytest.skip("no manifest yet — run `manamap pilot build-index`")
    for deck in json.loads(path.read_text())["decks"]:
        assert set(deck) == MANIFEST_KEYS, deck["slug"]


@requires_deck
def test_the_manifest_names_every_directory_a_browser_cannot_list():
    """`stacks/` was named here from the start for exactly this reason, and four
    other directories of keyed instances were not — so the dossier could show eight
    panels and no simulation, no experiment, no prescription and no decision.

    A browser cannot list a directory. If the manifest does not name the file, the
    page cannot fetch it, and no amount of frontend work fixes that.
    """
    manifest = {d["slug"]: d for d in
                json.loads((DECKS_DIR / "index.json").read_text())["decks"]}
    for slug, deck in manifest.items():
        for key, subdir in (("sim_runs", "sim"), ("experiments", "experiments"),
                            ("prescriptions", "prescriptions")):
            on_disk = sorted(p.name for p in (DECKS_DIR / slug / subdir).glob("*.json"))
            assert deck[key] == on_disk, f"{slug}.{key} disagrees with {subdir}/"
            for name in deck[key]:
                assert (DECKS_DIR / slug / subdir / name).exists(), f"{slug}: {name}"


@requires_deck
def test_presence_flags_match_the_files_on_disk():
    """`has` exists so the page knows whether to fetch at all. `getJSON` swallows a
    404 to null, so "absent artifact" and "failed request" are indistinguishable —
    the flag is what makes them different."""
    from manamap.config import MANUALS_DIR
    for deck in json.loads((DECKS_DIR / "index.json").read_text())["decks"]:
        slug = deck["slug"]
        base = DECKS_DIR / slug
        for name, flag in deck["has"].items():
            # Almost every flag names a file in the deck's own directory. `page`
            # is the exception on purpose: the compact Pilot's Manual is a
            # RENDERED page, so it lives under `manuals/p/` beside the magazine
            # it replaces, not in the artifact directory it was rendered from.
            if name == "page":
                path = MANUALS_DIR / "p" / f"{slug}.html"
            elif name == "log":
                path = base / "log.jsonl"
            else:
                path = base / f"{name}.json"
            assert flag == path.exists(), f"{slug}.has[{name}] vs {path}"


def test_the_manifest_is_byte_deterministic():
    """CI's last step is `make manuals && git diff --exit-code -- … index.json`, so a
    manifest that reorders between runs turns main red for no reason."""
    a = build_index.gather_entries()
    b = build_index.gather_entries()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@requires_deck
def test_tracked_manifest_matches_the_artifacts_on_disk():
    path = DECKS_DIR / "index.json"
    if not path.exists():
        pytest.skip("no manifest yet — run `manamap pilot build-index`")
    manifest = json.loads(path.read_text())

    # DRAFTS ARE NOT DECKS, and `gather_entries` returns both. A draft is a
    # brief with no 99, filed under `manifest.drafts` because every consumer of
    # `decks` assumes a `cards.json` that a draft does not have. Comparing the
    # whole scan against `decks` reported a stale manifest the moment a draft
    # existed — this test was the consumer the split forgot.
    entries = build_index.gather_entries()
    fresh = {e["slug"]: e for e in entries if not e.get("draft")}
    listed = {d["slug"]: d for d in manifest["decks"]}
    assert set(listed) == set(fresh), (
        "manifest deck set is stale — re-run `manamap pilot build-index`")

    fresh_drafts = {e["slug"] for e in entries if e.get("draft")}
    assert {d["slug"] for d in manifest.get("drafts", [])} == fresh_drafts, (
        "manifest draft set is stale — re-run `manamap pilot build-index`")

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
