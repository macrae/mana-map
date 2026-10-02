"""The two columns that answer "is this the eighth copy, or a new multiplier?".

Four consecutive edgar-vampires branches each added a drain body to a deck that already
ran seven of them, and nothing in the scanner asked. `function_overlap` says how close a
candidate is IN FUNCTION to the nearest card already in the 99; `trigger_events` says
which event it keys on. Neither is useful alone, which is why both are tested together
and why the report prints them on one line.

Every test here drives the production functions, and each one names the bug it was
written for so the bug can be re-introduced to prove the test fails.
"""
import re

from conftest import requires_data, requires_deck

from manamap.pilot import candidate_scan as cs
from manamap.pilot.card_pool import corpus_oracle

# `requires_data` and `requires_deck` are skipifs from conftest, not named marks: the
# whole module needs the ability embeddings (the function space) and, for the one test
# that drives `scan`, a real deck on disk.
pytestmark = [requires_data]


# ── trigger_events ────────────────────────────────────────────────────────────

def test_the_compound_subject_fires_both_self_and_other():
    """BUG: "whenever THIS CREATURE OR ANOTHER CREATURE dies" read as no_trigger.

    The first pattern anchored on `whenever (another|a|one or more|each)`, which is not
    how the cards are printed. Blood Artist, Zulaport Cutthroat and Cordial Vampire all
    begin "Whenever this creature or another creature…", so every aristocrat in the
    format classified as having no trigger at all — the single worst possible miss for
    a deck built on deaths.

    Re-introduce by replacing the `enters.other`/`dies.other` subject wildcard with a
    hard anchor on "whenever (another|a|one or more|each)".
    """
    oracle = corpus_oracle()
    checked = 0
    for name in ("Blood Artist", "Zulaport Cutthroat", "Cordial Vampire"):
        text = oracle.get(name)
        if not text:
            continue
        assert "this creature or another creature" in text.lower(), (
            f"{name}'s wording changed; this test is about the compound subject")
        events = cs.trigger_events(text)
        assert "dies.other" in events, (name, events)
        assert "dies.self" in events, (
            f"{name} covers its OWN death too and the deck cares: {events}")
        checked += 1
    assert checked >= 3, f"only {checked} of the three cards were present"


def test_a_one_shot_entry_is_not_a_recurring_one():
    """BUG: a one-shot ETB and a recurring trigger read the same, and cost a slot.

    Vampire Socialite is "WHEN THIS CREATURE ENTERS, if an opponent lost life this turn,
    put a +1/+1 counter on each other Vampire" — it fires once. Cordial Vampire's
    "whenever this creature or another creature dies" never stops. Judging them alike is
    what put Cordial Vampire on a cut list as "the flat-growth shape the pilot ruled out".

    Re-introduce by merging `enters.self` into `enters.other`.
    """
    oracle = corpus_oracle()
    socialite = cs.trigger_events(oracle.get("Vampire Socialite"))
    assert "enters.self" in socialite, socialite
    assert "enters.other" not in socialite, (
        f"Vampire Socialite fires on ITSELF entering, once: {socialite}")

    knight = cs.trigger_events(oracle.get("Corpse Knight"))
    assert "enters.other" in knight, knight
    assert "enters.self" not in knight, (
        f"Corpse Knight fires on OTHERS entering, all game: {knight}")


def test_the_token_event_exists_because_the_sweep_found_it():
    """BUG: Mirkwood Bats read no_trigger.

    "Whenever you create or sacrifice a token, each opponent loses 1 life" keys on the
    TOKEN, not on a creature entering, so no creature-entry pattern covers it. On a
    commander minting 8.04 tokens a game that is the most valuable event there is, and
    the first pass hid it. Only five cards in the corpus carry it.

    Re-introduce by deleting the `token_created` pattern.
    """
    events = cs.trigger_events(corpus_oracle().get("Mirkwood Bats"))
    assert "token_created" in events, events
    assert "no_trigger" not in events, events


def test_no_trigger_is_the_absence_of_an_event_and_never_an_event():
    """BUG: two instants both reading no_trigger were called the SAME EVENT.

    Feed the Swarm sits at 0.96 to Anguished Unmaking and both are one-shot spells, so
    intersecting their event sets matched on `no_trigger` and the report announced a
    duplicate. The absence of a trigger is not something two cards can share.

    This test covers `trigger_events`; `test_the_report_withholds_a_verdict_on_two_
    one_shots` below covers the `format_report` half, because the subtraction that
    fixes it lives there and a docstring is not a test.
    """
    assert cs.trigger_events("Flying") == ["no_trigger"]
    assert cs.trigger_events("") == ["no_trigger"]
    assert cs.trigger_events(None) == ["no_trigger"]
    # and it is never reported alongside a real event
    for name in ("Blood Artist", "Corpse Knight", "Mirkwood Bats"):
        events = cs.trigger_events(corpus_oracle().get(name))
        assert "no_trigger" not in events, (name, events)


def test_the_event_patterns_travel_with_the_artifact():
    """A reader must be able to ask "why is this row enters.other" from the FILE.

    The admission predicates are recorded in `sources.predicates` for exactly this
    reason; the event patterns are evidence of the same kind.
    """
    assert set(cs._E) == set(cs.EVENTS), "EVENTS and _E disagree"
    for name, pattern in cs._E.items():
        re.compile(pattern)  # every pattern must compile as written into the artifact


# ── function_overlap ──────────────────────────────────────────────────────────

def test_function_overlap_names_the_card_you_already_own():
    """The column's whole purpose, on the slot it would have caught.

    treasury-v1 added Zulaport Cutthroat to a deck already running Blood Artist. They sit
    at ~0.98 in the ability space AND key on the same event, which is the only combination
    that means redundancy.
    """
    got = cs.function_overlap("Zulaport Cutthroat", {"Blood Artist", "Sol Ring", "Command Tower"})
    assert got is not None, "the ability space is required for this test"
    assert got["nearest_in_99"] == "Blood Artist", got
    assert got["cosine"] >= 0.90, got
    assert got["space"] == "embeddings_ability.npy", (
        "similarity must come from the FUNCTION space; the layout space knows only "
        f"colour and type and would answer a different question: {got}")


def test_function_overlap_discriminates_rather_than_saying_everything_is_similar():
    """BUG guard: a column that reads high for everything carries no information.

    Sol Ring against a drain body must read far lower than one drain body against
    another, or the threshold in the report is meaningless.
    """
    near = cs.function_overlap("Zulaport Cutthroat", {"Blood Artist"})
    far = cs.function_overlap("Sol Ring", {"Blood Artist"})
    assert near and far
    assert near["cosine"] - far["cosine"] > 0.25, (near, far)


def test_function_overlap_is_ABSENT_not_zero_when_it_cannot_be_measured():
    """Absent means absent. A 0.0 cosine reads as a measured dissimilarity."""
    assert cs.function_overlap("Zulaport Cutthroat", set()) is None
    assert cs.function_overlap("Not A Real Card At All", {"Blood Artist"}) is None


@requires_deck
def test_a_scanned_row_carries_both_columns():
    """The row contract, driven through `scan` rather than asserted about it."""
    doc = cs.scan("edgar-vampires", dimensions=("drain",), limit=6)
    rows = doc["dimensions"]["drain"]["candidates"]
    assert rows, "no drain candidates — the scan cannot be exercised"
    for r in rows:
        assert r["trigger_events"], r["name"]
        assert "function_overlap" in r, (
            f"{r['name']} has no function_overlap key at all — absent must be an "
            "explicit None, so a reader can tell 'not measured' from 'not similar'")
    assert "events" in doc["sources"], "the event patterns must reach the artifact"
    assert any("function_overlap" in lim for lim in doc["limits"]), (
        "a new column ships with the limit that stops it being over-read")


def test_the_report_withholds_a_verdict_on_two_one_shots():
    """BUG: the report called two instants a duplicate because both read no_trigger.

    Feed the Swarm sits at 0.96 to Anguished Unmaking and both are one-shot spells, so
    intersecting their event sets matched on `no_trigger` and the line announced
    "SAME EVENT — likely a duplicate". The absence of a trigger is not something two
    cards can share, and two one-shots that read alike may both be worth running.

    Driven through `format_report` on a minimal doc, because the subtraction being
    guarded lives in the formatter and not in `trigger_events`.

    Re-introduce by dropping the `- {"no_trigger"}` subtraction in `format_report`.
    """
    def doc_for(candidate_text, owned_text, cosine):
        return {
            "slug": "x", "as_of": "2026-10-02", "identity": ["B"],
            "sources": {"edhrec_cards": None},
            "excluded": {"in_99": 0, "identity": 0, "illegal": 0, "game_changer": []},
            "_oracle": {"Owned Card": owned_text},
            "dimensions": {"drain": {
                "matched": 1, "truncated": 0, "flagged_infinite": 0,
                "candidates": [{
                    "name": "Candidate", "mana_cost": "{1}{B}", "power": None,
                    "toughness": None, "keywords": [], "edhrec_rank": 1,
                    "matched": {"oracle": ["drain.equal_to"]}, "flags": {},
                    "synergy_into_99": [],
                    "combos": {"infinite_with": [], "two_card_lines": 0, "bracket_max": None},
                    "trigger_events": cs.trigger_events(candidate_text),
                    "function_overlap": {"nearest_in_99": "Owned Card", "cosine": cosine,
                                         "space": "embeddings_ability.npy"},
                }],
            }},
        }

    # two one-shots, high cosine: NO duplicate verdict
    out = cs.format_report(doc_for("Destroy target creature.", "Exile target creature.", 0.96))
    assert "SAME EVENT" not in out, out
    assert "neither keys on an event" in out, out

    # same real event, high cosine: the verdict fires
    dies = "Whenever this creature or another creature dies, each opponent loses 1 life."
    out = cs.format_report(doc_for(dies, dies, 0.98))
    assert "SAME EVENT — likely a duplicate" in out, out

    # different real events, high cosine: it stacks
    enters = "Whenever another creature you control enters, each opponent loses 1 life."
    out = cs.format_report(doc_for(enters, dies, 0.97))
    assert "different event — stacks rather than duplicates" in out, out

    # below the threshold: no verdict at all, only the cosine
    out = cs.format_report(doc_for(dies, dies, 0.80))
    assert "SAME EVENT" not in out and "different event" not in out, out
    assert "nearest in 99: Owned Card 0.80" in out, out
