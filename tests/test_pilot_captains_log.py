"""The captain's log — the deterministic half, which is what the gate rests on.

The design of this feature is one trade: push everything that CAN be computed out
of the prose and into Python, so a validator can hold the artifact to account
without judging a word of writing. These tests pin the computed half. If the
stardate or the grouping drifts, the header of every rendered log quotes a number
the record no longer carries, and `validate-captains-log` fails fleet-wide.
"""

import json

import pytest

from manamap.config import DECKS_DIR
from manamap.pilot import captains_log as cl
from manamap.pilot.deck_notes import read_log

from conftest import requires_deck


# THE ELEVEN REAL ENTRIES, as literals. Cheap, and it catches the refactor that
# silently moves the epoch or changes the flooring — neither of which any other
# test in the suite would notice until a fleet regeneration.
STARDATES = [
    ("edgar-vampires", "001", "2026-08-22T21:00:00",       "80234.8", "2026-08-22"),
    ("edgar-vampires", "002", "2026-08-25T22:00",          "80237.9", "2026-08-25"),
    ("edgar-vampires", "003", "2026-08-28T20:00",          "80240.8", "2026-08-28"),
    ("edgar-vampires", "004", "2026-09-01T21:30:00-07:00", "80244.8", "2026-09-01"),
    ("gishath",        "001", "2026-08-28T22:00",          "80240.9", "2026-08-28"),
    ("goblin-storm",   "001", "2026-08-28T19:00",          "80240.7", "2026-08-28"),
    ("goblin-storm",   "002", "2026-09-01T19:00:00-07:00", "80244.7", "2026-09-01"),
    ("heliod",         "001", "2026-08-28T21:00",          "80240.8", "2026-08-28"),
    ("ur-dragon",      "001", "2026-08-22T20:00:00",       "80234.8", "2026-08-22"),
    ("ur-dragon",      "002", "2026-08-25T20:00",          "80237.8", "2026-08-25"),
    ("ur-dragon",      "003", "2026-09-01T20:15:00-07:00", "80244.8", "2026-09-01"),
]


@pytest.mark.parametrize("slug,eid,at,want_sd,want_night", STARDATES)
def test_the_stardate_and_the_night_are_what_the_table_says(slug, eid, at,
                                                            want_sd, want_night):
    assert cl.stardate(at) == want_sd
    assert cl.night_key(at) == want_night


def test_the_epoch_reproduces_the_pilots_own_example():
    """1 September 2026 is day 244, and the pilot's example stardate is 80244.6.

    The integer part is the whole of what the examples pin down — the decimal in
    them could not be derived from any timestamp under any rule tried, so it is
    treated as illustrative and the decimal is the time of day.
    """
    assert cl.stardate("2026-09-01T00:00:00").startswith("80244.")
    assert cl.stardate("2026-01-01T00:00:00") == "80001.0"


def test_a_timezone_aware_entry_is_read_as_local_wall_clock():
    """RE-INTRODUCE THE BUG AND THIS FAILS — which is the only way to know it works.

    Five of the eleven entries on disk are naive and four carry `-07:00`.
    Normalising to UTC moves a 21:30-07:00 game to the following day while
    leaving the naive half untouched, so the fleet would split down the middle on
    a property of how the note was TYPED rather than when the game was PLAYED.

    edgar 004 (21:30-07:00) and ur-dragon 003 (20:15-07:00) were played on the
    same evening as goblin-storm 002 (19:00-07:00). All three must land on the
    same night and the same stardate day.
    """
    trio = ["2026-09-01T19:00:00-07:00", "2026-09-01T20:15:00-07:00",
            "2026-09-01T21:30:00-07:00"]
    assert {cl.night_key(a) for a in trio} == {"2026-09-01"}
    assert {cl.stardate(a).split(".")[0] for a in trio} == {"80244"}

    # The exact bug: `.astimezone(timezone.utc)` would push 21:30-07:00 to
    # 04:30 the next day. Asserted here as arithmetic so the reason survives.
    from datetime import timezone
    from datetime import datetime as _dt
    moved = _dt.fromisoformat(trio[2]).astimezone(timezone.utc)
    assert moved.date().isoformat() == "2026-09-02", (
        "if this ever stops being true the regression is no longer possible and "
        "this test has lost its subject")


def test_a_late_game_belongs_to_the_night_it_started():
    """Commander runs late. A game logged at 01:30 is the same night's."""
    assert cl.night_key("2026-09-02T01:30:00") == "2026-09-01"
    assert cl.night_key("2026-09-02T03:59:00") == "2026-09-01"
    assert cl.night_key("2026-09-02T04:00:00") == "2026-09-02"


def test_stardates_sort_the_way_time_does():
    """The property that actually breaks if somebody 'improves' the epoch."""
    ats = [a for _s, _i, a, _d, _n in STARDATES]
    from datetime import datetime
    by_time = sorted(ats, key=lambda a: datetime.fromisoformat(a).replace(tzinfo=None))
    by_stardate = sorted(ats, key=cl.stardate)
    assert [cl.stardate(a) for a in by_time] == [cl.stardate(a) for a in by_stardate]


def test_the_year_rolls_by_a_thousand():
    assert cl.stardate("2026-12-31T23:00:00") == "80365.9"
    assert cl.stardate("2027-01-01T00:30:00") == "81001.0"


# ---------------------------------------------------------------- the skeleton

@requires_deck
def test_every_logged_game_reaches_exactly_one_night():
    """THE RAW NOTES STAY REACHABLE, checked as arithmetic.

    A game that no night claims is invisible on the page; a game two nights claim
    appears under two stardates and the reader can trust neither. This is the
    property the whole design exists to keep, so it is asserted over the real
    fleet and not over a fixture.
    """
    checked = 0
    for deck in sorted(DECKS_DIR.glob("*/log.jsonl")):
        slug = deck.parent.name
        entries = read_log(slug)
        if not entries:
            continue
        seen = []
        for night in cl.nights(slug).values():
            seen.extend(night["source_ids"])
        assert sorted(seen) == sorted(e["id"] for e in entries), slug
        assert len(seen) == len(set(seen)), f"{slug}: a game filed twice"
        checked += len(entries)
    assert checked >= 10, "the fleet's eleven logged games should all be checked"


@requires_deck
def test_the_night_places_itself_in_an_evening_that_spans_decks():
    """The pilot flies a DIFFERENT DECK each game, so a night is a fleet-wide
    event each deck sees one slice of. The pilot already writes this by hand —
    heliod 001 says "Game three of four on the night" — which is the evidence
    that it belongs in the record rather than being invented here.
    """
    fleet = cl.evening("2026-09-01")
    assert [g["slug"] for g in fleet] == ["goblin-storm", "ur-dragon", "edgar-vampires"]

    pos = cl.nights("ur-dragon")["2026-09-01"]["position_in_evening"]
    assert pos == {"n": 2, "of": 3, "after": "goblin-storm"}
    # The first ship of the evening follows nobody — absent, not a placeholder.
    assert cl.nights("goblin-storm")["2026-09-01"]["position_in_evening"]["after"] is None


@requires_deck
def test_the_skeleton_carries_the_version_and_the_cause_it_was_filed_under():
    n = cl.nights("ur-dragon")["2026-09-01"]
    assert n["version"] == "v1.0.2", "the version is read from the tracked tags"
    assert n["games"][0]["cause"] == "politics"
    assert n["games"][0]["result"] == "loss"


@requires_deck
def test_a_deck_that_has_never_been_played_has_no_nights():
    """Absent means absent — not an empty log with a stardate on it."""
    assert cl.nights("radagast") == {}
    assert cl.skeleton("radagast")["nights"] == {}


def test_the_vocabularies_are_closed_and_small():
    assert set(cl.STATIONS) == {"engineering", "tactical", "ops", "helm"}
    assert cl.ATTRIBUTION_ORDER == ("self", "ship", "circumstance")
    assert cl.LOG_KINDS == ("ship", "personal")
    assert set(cl.STATION_ROLES) == set(cl.STATIONS), (
        "every station needs a role mapping or its cards are unreachable")
    assert len(cl.SECTION_KEYS) == 6


@requires_deck
def test_every_station_has_cards_on_a_real_deck():
    """A station nothing answers to is a word the agent cannot use.

    `helm` is the reason this test exists: it was added because Engineering /
    Tactical / Ops left the deck's win route with nowhere to file, and a
    vocabulary that is incomplete makes the agent misfile under the nearest
    station — which reads as correct and counts as nothing.
    """
    roster = cl.stations_for_deck("ur-dragon")
    for station in cl.STATIONS:
        assert roster[station], f"no ur-dragon card answers to {station}"


# ------------------------------------------------------------------- the merge

def _handoff(tmp_deck, payload):
    out = tmp_deck / ".agent-out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "captains-log.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def fake_deck(tmp_path, monkeypatch):
    """A two-game night, which the real fleet does not have and cannot exercise.

    NO DECK HAS EVER BEEN PLAYED TWICE IN ONE NIGHT — the pilot flies a different
    deck each game — so `supplementals` has zero instances fleet-wide and would
    otherwise ship as a path nothing has run.
    """
    deck = tmp_path / "decks" / "twice"
    deck.mkdir(parents=True)
    (deck / "log.jsonl").write_text("".join(
        json.dumps({"id": i, "at": at, "decklist_sha256": None, "result": r,
                    "opponents": 3, "tags": [], "text": t}) + "\n"
        for i, at, r, t in [
            ("001", "2026-09-01T19:00:00", "loss", "first game"),
            ("002", "2026-09-01T21:00:00", "win", "second game"),
        ]), encoding="utf-8")
    monkeypatch.setattr(cl, "DECKS_DIR", tmp_path / "decks")
    import manamap.pilot.deck_notes as dn
    monkeypatch.setattr(dn, "deck_dir", lambda s: tmp_path / "decks" / s)
    import manamap.pilot.merge_captains_log as mcl
    monkeypatch.setattr(mcl, "DECKS_DIR", tmp_path / "decks")
    return deck


def test_two_games_on_one_night_become_one_log(fake_deck):
    nights = cl.nights("twice")
    assert list(nights) == ["2026-09-01"], "one night, not two logs"
    n = nights["2026-09-01"]
    assert n["source_ids"] == ["001", "002"]
    assert [g["supplemental_index"] for g in n["games"]] == [0, 1]
    assert n["stardate"] == cl.stardate("2026-09-01T19:00:00"), (
        "the night is stamped with when it STARTED")


def test_the_merge_takes_prose_and_recomputes_every_fact(fake_deck):
    """THE WHITELIST IS THE POINT.

    An agent that invents a stardate — or groups two nights into one because the
    story flowed better — must not have that land. It is discarded at the merge,
    and the validator then fails because the header quotes a number the record
    does not carry.
    """
    from manamap.pilot import merge_captains_log as mcl

    _handoff(fake_deck, {"nights": {"2026-09-01": {
        "header": "Captain's log, stardate 99999.9.",
        "situation": "s", "narrative": "n",
        "assessment": [{"attribution": "self", "text": "a"}],
        "orders": [{"station": "helm", "text": "I have ordered Helm to hold."}],
        "coda": "c",
        # Smuggled facts, every one of which must be ignored.
        "stardate": "99999.9", "source_ids": ["999"], "version": "v9.9.9",
        "games": [], "night": "1999-01-01",
    }}})
    merged, rejected, path = mcl.merge("twice")
    assert merged == ["2026-09-01"] and not rejected
    doc = json.loads(path.read_text())
    night = doc["nights"]["2026-09-01"]
    assert night["stardate"] == "80244.7", "the merge recomputed the stardate"
    assert night["source_ids"] == ["001", "002"]
    assert night["version"] is None
    assert set(night["logs"]["ship"]) == set(cl.SECTION_KEYS), (
        "only the six prose sections survive the whitelist")


def test_the_merge_refuses_a_night_the_log_does_not_have(fake_deck):
    from manamap.pilot import merge_captains_log as mcl

    _handoff(fake_deck, {"nights": {"1999-01-01": {"header": "h", "situation": "s",
                                                   "narrative": "n", "coda": "c"}}})
    with pytest.raises(SystemExit) as e:
        mcl.merge("twice")
    assert "the log is the authority" in str(e.value)


def test_the_merge_refuses_an_empty_handoff(fake_deck):
    """Merging nothing and reporting success is how a log reads as rendered with
    every check still green."""
    from manamap.pilot import merge_captains_log as mcl

    _handoff(fake_deck, {"nights": {}})
    with pytest.raises(SystemExit) as e:
        mcl.merge("twice")
    assert "nothing to merge" in str(e.value)


def test_a_second_merge_carries_the_first_nights_prose_forward(fake_deck):
    """A scoped re-spawn must not drop the nights it was not asked about."""
    from manamap.pilot import merge_captains_log as mcl

    good = {"header": "Captain's log, stardate 80244.7.", "situation": "s",
            "narrative": "n", "assessment": [{"attribution": "self", "text": "a"}],
            "orders": [{"station": "ops", "text": "I have ordered Ops to look."}],
            "coda": "c"}
    _handoff(fake_deck, {"nights": {"2026-09-01": good}})
    mcl.merge("twice")

    # A later run that renders nothing new still must not erase what is there.
    _handoff(fake_deck, {"nights": {"2026-09-01": dict(good, coda="revised")}})
    _m, _r, path = mcl.merge("twice")
    doc = json.loads(path.read_text())
    assert doc["nights"]["2026-09-01"]["logs"]["ship"]["coda"] == "revised"


# --------------------------------------------------------------- the validator

def _night_doc(**over):
    block = {"header": "Captain's log, stardate 80244.7.",
             "situation": "s", "narrative": "n",
             "assessment": [{"attribution": "self", "text": "mine"},
                            {"attribution": "ship", "text": "hers"}],
             "orders": [{"station": "tactical", "text": "I have ordered Tactical to hold."}],
             "coda": "c"}
    block.update(over)
    return {"slug": "twice", "ship": None,
            "nights": {"2026-09-01": dict(cl.nights("twice")["2026-09-01"],
                                          logs={"ship": block})}}


def test_a_sound_log_passes(fake_deck):
    from manamap.pilot import validate_captains_log as v
    errors, _notes = v.validate(_night_doc(), "twice")
    assert errors == []


def test_the_header_must_quote_the_stardate(fake_deck):
    from manamap.pilot import validate_captains_log as v
    errors, _ = v.validate(_night_doc(header="Captain's log, stardate 12345.6."),
                           "twice")
    assert any("does not quote the stardate" in e for e in errors)


def test_responsibility_goes_to_the_captain_first(fake_deck):
    """The pilot's hardest style rule, and it is only checkable because the
    ordering was pushed out of prose and into structure."""
    from manamap.pilot import validate_captains_log as v

    errors, _ = v.validate(_night_doc(assessment=[
        {"attribution": "circumstance", "text": "the table"},
        {"attribution": "self", "text": "mine"}]), "twice")
    assert any("does not begin with `self`" in e for e in errors)

    errors, _ = v.validate(_night_doc(assessment=[
        {"attribution": "self", "text": "mine"},
        {"attribution": "circumstance", "text": "the table"},
        {"attribution": "ship", "text": "hers"}]), "twice")
    assert any("out of order" in e for e in errors)


def test_an_exclamation_mark_fails(fake_deck):
    from manamap.pilot import validate_captains_log as v
    errors, _ = v.validate(_night_doc(coda="We came within a turn!"), "twice")
    assert any("exclamation mark" in e for e in errors)


def test_a_station_outside_the_vocabulary_fails(fake_deck):
    from manamap.pilot import validate_captains_log as v
    errors, _ = v.validate(_night_doc(
        orders=[{"station": "weapons", "text": "I have ordered it."}]), "twice")
    assert any("is not a station" in e for e in errors)


def test_five_of_six_sections_fails(fake_deck):
    """A short log frozen as a cache HIT renders short forever, green throughout."""
    from manamap.pilot import validate_captains_log as v
    errors, _ = v.validate(_night_doc(coda=""), "twice")
    assert any("coda is missing or empty" in e for e in errors)


def test_a_drifted_stardate_is_caught_by_recomputation(fake_deck):
    from manamap.pilot import validate_captains_log as v
    doc = _night_doc()
    doc["nights"]["2026-09-01"]["stardate"] = "80244.9"
    errors, _ = v.validate(doc, "twice")
    assert any("stardate" in e and "recomputed as" in e for e in errors)


def test_the_style_checks_that_cannot_be_proved_harmless_only_report(fake_deck):
    """SHIPPED REPORTING-ONLY, ON PURPOSE.

    Shouty capitals, jargon, superlatives and order phrasing are the abstraction
    layer's real failure modes, and none of them could be proved harmless before
    there was any prose to measure. A validator that fires on correct data is
    worse than no validator, so these print and do not fail until a fleet run has
    graded them.
    """
    from manamap.pilot import validate_captains_log as v
    errors, notes = v.validate(_night_doc(
        narrative="THE MULLIGAN was my mistake; the wipe was brutal.",
        orders=[{"station": "ops", "text": "Ops will review the draw."}]), "twice")
    assert errors == [], "none of these may fail the gate yet"
    joined = " ".join(notes)
    assert "shouty caps" in joined and "jargon" in joined
    assert "superlative" in joined
    assert "already issued" in joined
