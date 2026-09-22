"""validate-recon: the gate `deck_recon.json` did not have.

Recon's whole value is that every card it names is real, legal and castable in this
deck. That was checked by hand once, on 110 cards, and the repo's rule is that such
work becomes a mechanical check or it is re-spent every run.

These also pin the two things the gate deliberately does NOT fail on, because both
fire on correct data: a card that has since joined the 99, and age.
"""

import json

import pytest

from manamap.pilot import validate_recon

from conftest import requires_data, requires_deck


# A one-card corpus. An EMPTY dict would mean "an empty cards.csv", in which every
# card is correctly missing — not "skip the card checks".
POOL = {"Craterhoof Behemoth": {"color_identity": {"G"}, "legal": True,
                                "edhrec_rank": 1, "game_changer": False,
                                "type_line": "Creature", "cmc": 8.0,
                                "mana_cost": "{5}{G}{G}{G}"}}


def _doc(**over):
    doc = {
        "slug": "radagast", "commander": "Radagast of Rhosgobel", "as_of": "2026-08-12",
        "consensus": "…", "sources": [{"title": "T", "url": "https://example.com/a"}],
        "findings": [{"claim": "c", "cards": ["Craterhoof Behemoth"],
                      "confidence": "widely agreed",
                      "sources": ["https://example.com/a"]}],
    }
    doc.update(over)
    return doc


def test_a_clean_recon_passes():
    assert validate_recon.validate(_doc(), "radagast", pool=POOL) == []


def test_structure_errors_return_early_but_field_errors_do_not():
    """An early return must be about CASCADE, not about severity.

    The first version returned on any shape error, so a `confidence` typo suppressed
    a nonexistent-card error in the same file — proven when the gate was re-broken
    four ways and reported one. Missing keys still return early, because every later
    check would crash on them.
    """
    errs = validate_recon.validate({"slug": "radagast"}, "radagast", pool=POOL)
    assert any("missing required key" in e for e in errs)
    assert not any("not in cards.csv" in e or "not in the top-level" in e
                   for e in errs), \
        "structural errors must return before the checks that would crash on them"

    bad = _doc()
    bad["findings"][0]["confidence"] = "pretty sure"
    bad["findings"][0]["sources"] = ["https://example.invalid/never-fetched"]
    errs = validate_recon.validate(bad, "radagast", pool=POOL)
    assert any("confidence" in e for e in errs)
    assert any("not in the top-level sources" in e for e in errs), \
        "a field error must not hide the checks after it"


def test_a_finding_may_not_cite_a_source_nobody_recorded_fetching():
    bad = _doc()
    bad["findings"][0]["sources"] = ["https://example.com/never-listed"]
    errs = validate_recon.validate(bad, "radagast", pool=POOL)
    assert any("not in the top-level sources" in e for e in errs)


def test_the_slug_must_match_the_directory_it_lives_in():
    errs = validate_recon.validate(_doc(slug="heliod"), "radagast", pool=POOL)
    assert any("lives in radagast/" in e for e in errs)


def test_as_of_must_be_a_real_date_because_recon_is_perishable():
    errs = validate_recon.validate(_doc(as_of="last tuesday"), "radagast", pool=POOL)
    assert any("not an ISO date" in e for e in errs)


@requires_data
@requires_deck
def test_a_card_outside_the_identity_or_the_corpus_fails():
    """The charter's rule 5, finally enforced: in identity, legal, and real."""
    bad = _doc()
    bad["findings"][0]["cards"] = ["Lightning Bolt", "Not A Real Card At All"]
    errs = validate_recon.validate(bad, "radagast")
    assert any("outside radagast's G identity" in e for e in errs)
    assert any("not in cards.csv" in e for e in errs)


@requires_data
def test_a_double_faced_card_named_by_its_front_face_resolves():
    """`load_pool` keys only on the joined `"A // B"` form, so a recon naming
    `Legion's Landing` — how everyone writes it — looked like a card that does not
    exist. It failed three real cards across two tracked recons the first time this
    gate ran, and every one was correct data."""
    from manamap.pilot.card_pool import load_pool
    pool = load_pool()
    for name in ("Legion's Landing", "Edgar, Charmed Groom", "Disciple of Freyalise"):
        assert name not in pool, f"{name} is keyed on its joined form, as expected"
        assert validate_recon._resolve(pool, name) is not None, name


@requires_data
@requires_deck
def test_a_card_since_added_to_the_99_is_a_WARN_and_never_an_error():
    """The charter says a recon may not name a card already in the 99 — and enforcing
    that as an error would fail 4 of 5 tracked recons on 37 card-instances. Recon is
    DATED and the decklist moves under it; a card correctly outside the 99 in August
    is inside it in September without the artifact changing a byte.

    THE WARN IS PROVEN ON A DOC THIS TEST BUILDS. It used to require radagast's
    tracked recon to stay out of step with its 99 — so bringing the two into
    line, which is the whole point of reading a recon, would have reddened the
    suite. The tracked sweep below is still run for the ERROR half (no recon may
    fail validation); only the WARN path is constructed.
    """
    doc = json.loads((validate_recon.deck_dir("radagast") / "deck_recon.json").read_text())
    assert validate_recon.validate(doc, "radagast") == [], (
        "a tracked recon must never fail validation, in sync or not")

    # A card that IS in the 99, named by a FINDING on a recon this test builds.
    # `_named_cards` reads `findings[].cards` and nothing else, so that is where
    # the card has to go for either half of the contract to see it.
    in_deck = sorted(validate_recon.deck_names("radagast"))[0]
    constructed = json.loads(json.dumps(doc))
    constructed["findings"] = [dict(constructed["findings"][0], cards=[in_deck])]
    assert validate_recon.validate(constructed, "radagast") == [], (
        "naming a card already in the 99 must not be an ERROR — enforcing that "
        "failed 4 of 5 tracked recons on 37 card-instances")
    assert in_deck in validate_recon.in_the_99(constructed, "radagast"), (
        "...and it must still be REPORTED, or the staleness is invisible")


def test_ownership_is_falsified_against_the_boxes(tmp_path, monkeypatch):
    """`ownership` is a claim about the pilot's cards, so it is checkable — and this
    is the check that caught the tracked collection being two boxes short."""
    from manamap.pilot import collection as coll
    from manamap.pilot.common import clear_memo
    cdir = tmp_path / "collection"
    cdir.mkdir()
    (cdir / "Green.txt").write_text("1 Craterhoof Behemoth\n")
    monkeypatch.setattr(coll, "COLLECTION_DIR", cdir)
    clear_memo()

    ok = _doc(ownership={"Craterhoof Behemoth": True})
    assert not [e for e in validate_recon._validate_ownership(ok) if "says" in e]

    lying = _doc(ownership={"Craterhoof Behemoth": False})
    assert any("says False but no box in COLLECTION_DIR holds it"
               in e for e in validate_recon._validate_ownership(lying))


def test_a_named_card_missing_from_ownership_is_reported(tmp_path, monkeypatch):
    """A card the recon recommends without stating its cost leaves the pilot to
    find out at the till."""
    from manamap.pilot import collection as coll
    from manamap.pilot.common import clear_memo
    monkeypatch.setattr(coll, "COLLECTION_DIR", tmp_path / "none")
    clear_memo()
    doc = _doc(ownership={"Something Else": False})
    assert any("absent from `ownership`" in e
               for e in validate_recon._validate_ownership(doc))


def test_optional_keys_stay_optional():
    """`ownership`, `_checked_against` and `commander_text_verified` each appear on
    exactly ONE tracked deck — things a particular run had reason to record, not a
    schema. Requiring any of them would fail the other four."""
    assert validate_recon.validate(_doc(), "radagast", pool=POOL) == []
    assert "ownership" not in validate_recon.REQUIRED_KEYS
