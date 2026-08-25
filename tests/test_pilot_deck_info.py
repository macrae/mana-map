"""deck-info: the workbench view composes what other commands own and computes
nothing new — so it must (a) never crash on a deck that has almost nothing, and
(b) derive every `next` suggestion from a condition that is true right now."""

import json

import pytest

from manamap.pilot import deck_info
from manamap.pilot.deck_notes import append_entry

from conftest import requires_deck

SLUG = "infodeck"


@pytest.fixture
def bare_deck(tmp_path, monkeypatch):
    """decklist + cards.json and nothing else — a deck someone just made."""
    decks = tmp_path / "decks"
    base = decks / SLUG
    base.mkdir(parents=True)
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", decks)
    (base / "decklist.txt").write_text("1 Radagast of Rhosgobel *CMDR*\n1 Forest\n")
    (base / "cards.json").write_text(json.dumps({
        "deck": SLUG, "decklist_sha256": "x",
        "cards": [{"name": "Radagast of Rhosgobel", "is_commander": True, "type_line": "Legendary Creature",
                   "mana_cost": "{3}{G}", "cmc": 4, "colors": ["G"], "color_identity": ["G"],
                   "oracle_text": "", "layout": "normal"},
                  {"name": "Forest", "quantity": 1, "type_line": "Basic Land — Forest",
                   "mana_cost": "", "cmc": 0, "colors": [], "color_identity": [],
                   "oracle_text": "({T}: Add {G}.)", "layout": "normal"}]}))
    return base


def test_a_bare_deck_composes_and_is_told_to_play(bare_deck):
    info = deck_info.compose(SLUG)
    assert info["commander"] == ["Radagast of Rhosgobel"]
    assert info["record"]["games"] == 0 and info["engine"] is None and info["diagnosis"] is None
    assert info["version"]["of"] == 0, "no git history is an empty list, not an error"
    assert any("captain's log" in n for n in info["next"])
    assert not any("un-debriefed" in n for n in info["next"])


def test_next_derives_from_what_is_true(bare_deck):
    append_entry(SLUG, "game one", result="loss")
    info = deck_info.compose(SLUG)
    assert info["record"] == {"games": 1, "win": 0, "loss": 1, "draw": 0,
                              "last_played": info["record"]["last_played"], "undebriefed": ["001"]}
    assert any("not yet debriefed" in n and "/debrief" in n for n in info["next"])
    (bare_deck / "log_annotations.json").write_text(json.dumps(
        {"slug": SLUG, "entries": {"001": {"summary": "s", "takeaways": [],
                                           "open_questions": [{"question": "?", "settled_by": "goldfish",
                                                               "why_it_matters": "?"}]}}}))
    info = deck_info.compose(SLUG)
    assert not any("debriefed" in n for n in info["next"])
    assert info["open_questions"][0]["from"] == "log:001"
    assert any("goldfish ×1" in n for n in info["next"])


def test_json_and_print_agree_on_the_same_dict(bare_deck, capsys):
    deck_info.main(type("A", (), {"slug": SLUG, "as_json": True})())
    out = json.loads(capsys.readouterr().out)
    assert set(out) >= {"slug", "version", "status", "record", "next"}
    deck_info.main(type("A", (), {"slug": SLUG, "as_json": False})())
    text = capsys.readouterr().out
    # The deck names itself; the header used to say "WORKBENCH — <slug>", which
    # named the landing page at the top of a per-deck command.
    assert "INFODECK — " in text and "NEXT" in text


@requires_deck
def test_a_real_deck_composes_every_panel():
    info = deck_info.compose("radagast")
    assert info["engine"]["critic"] == "pass" and info["engine"]["verified_lines"] >= 1
    assert info["goldfish"]["commander_mean_cast_turn"] is not None
    assert info["bracket"]["floor"] is not None and info["audit"]["archetype"]
    assert info["version"]["of"] >= 1 and not info["status"]["invalid"]


def test_a_broken_down_deck_is_not_told_to_go_and_play_it(bare_deck):
    """The workbench told the pilot to play a deck that no longer existed.

    `hapatra` was marked `broken-down` — its cards pulled and sleeved into
    yawgmoth-swarm — before the pivot, and `deck-info`, the START HERE command,
    kept answering "nothing in the captain's log — play it". The status was
    authored on `issue.json` and read only by the magazine renderer, so the
    bench could not see it.

    Three things are asserted because each failed differently: the status is
    SAID (a shorter list is not a statement), the impossible instructions are
    withheld, and the still-possible work survives — a published record can
    still have a failing gate or an open rules question.
    """
    info = deck_info.compose(SLUG)
    assert info["lifecycle"] is None, "no issue.json means live, not unknown"
    assert any("play it" in n for n in info["next"])

    (bare_deck / "issue.json").write_text(json.dumps(
        {"commander": "Radagast of Rhosgobel", "status": "broken-down"}))
    (bare_deck / "log_annotations.json").write_text(json.dumps(
        {"slug": SLUG, "entries": {"001": {"summary": "s", "takeaways": [],
                                           "open_questions": [{"question": "?",
                                                               "settled_by": "resolve-stack",
                                                               "why_it_matters": "?"}]}}}))
    info = deck_info.compose(SLUG)
    assert info["lifecycle"]["status"] == "broken-down"
    assert any("BROKEN DOWN FOR PARTS".lower() in n.lower() for n in info["next"]), \
        "the status must be stated, not merely acted on"
    # Match the COMMAND forms, not the words: the withholding line itself has to
    # name what it withheld, so a bare word scan flags the statement it wants.
    assert not any("play it" in n or "`simulate " in n or "`experiment " in n
                   for n in info["next"]), "an instruction the pilot cannot follow"
    assert any("resolve-stack ×1" in n for n in info["next"]), \
        "settling a rules question needs no cardboard"


def test_a_superseded_deck_is_still_playable(bare_deck):
    """`superseded` is deliberately outside `UNPLAYABLE_STATUSES`: a superseded
    list is still sleeved and can still be played, it is just no longer the best
    version of itself. Collapsing the three statuses into "not live" would have
    silenced a deck the pilot can pick up tonight."""
    (bare_deck / "issue.json").write_text(json.dumps({"status": "superseded"}))
    info = deck_info.compose(SLUG)
    assert info["lifecycle"]["status"] == "superseded"
    assert any("play it" in n for n in info["next"])


def test_an_unlocked_deck_is_not_the_same_as_a_dead_one(bare_deck):
    """Three states, and only one of them earns "go and play it".

    LOCKED is the pilot's assertion that this exact list is sleeved. A dead
    `status` says it demonstrably is not. ABSENT says nobody has claimed either
    — which is where every build plan sits, and where three decks sat while the
    front door cheerfully told the pilot to go and play them.

    This is the quiet half of the defect the broken-down test above covers. That
    one was loud: the cards were provably in another deck's sleeves. This one
    just does not know, and said nothing, which reads identically to knowing.
    """
    # The lock is written as the artifact rather than through `set_paper`,
    # which needs a git history to hang the version off — correctly, since a
    # lock is a claim about one exact list. `bare_deck` has no repo.
    def lock(**kw):
        (bare_deck / "deck_versions.json").write_text(json.dumps(
            {"slug": SLUG, "paper": {"version": 1, "built_at": "2026-08-25", **kw},
             "tags": {}}))

    # No lock: say so, and say what would fix it.
    info = deck_info.compose(SLUG)
    assert info["paper"] is None
    unlocked = [n for n in info["next"] if "not marked as built in paper" in n]
    assert len(unlocked) == 1, "an unlocked deck must say it is unlocked"
    assert f"deck-version {SLUG} paper" in unlocked[0], "name the command that fixes it"
    # It informs, it does not withhold — an unbuilt deck is not a closed one.
    assert any("play it" in n for n in info["next"])

    # Locked: the question is answered, so the line goes.
    lock(note="sleeved")
    info = deck_info.compose(SLUG)
    assert info["paper"]["note"] == "sleeved"
    assert not any("not marked as built in paper" in n for n in info["next"])

    # Dead: the lock is irrelevant, and the closed-loop line must not be joined
    # by a second one telling the pilot to lock a deck that no longer exists.
    (bare_deck / "deck_versions.json").write_text(json.dumps({"slug": SLUG, "tags": {}}))
    (bare_deck / "issue.json").write_text(json.dumps(
        {"commander": "Radagast of Rhosgobel", "status": "broken-down"}))
    info = deck_info.compose(SLUG)
    assert not any("not marked as built in paper" in n for n in info["next"])


def test_the_lock_in_info_json_is_the_authored_half_only(bare_deck):
    """`info.json` is committed and omits everything git-derived, because the
    commit that changes `decklist.txt` receives its sha after anything written
    in the same commit — so a stored version number is one behind forever.

    `paper()` is a plain read of `deck_versions.json` and carries no drift, so
    it is on the safe side of that split. If `in_sync` or `drift` ever appear
    here, the artifact has started making a claim it cannot keep current.
    """
    (bare_deck / "deck_versions.json").write_text(json.dumps(
        {"slug": SLUG, "tags": {},
         "paper": {"version": 1, "sha": "abc123def456", "decklist_sha256": "0" * 64,
                   "built_at": "2026-08-25", "note": "sleeved"}}))
    info = deck_info.compose(SLUG)
    assert set(info["paper"]) <= {"version", "sha", "decklist_sha256", "built_at", "note"}
    assert "in_sync" not in info["paper"] and "drift" not in info["paper"]
    # `bare_deck` is not a git repository, so this whole composition ran with no
    # git available — which is the property that matters: CI can build
    # `info.json`, lock included, and the drift stays where it can be kept
    # current. The version block comes back empty here for exactly that reason.
    assert info["version"]["current"] is None and info["version"]["of"] == 0


# ── Three states, three words ──────────────────────────────────────────────


def test_the_three_states_read_as_three_words(bare_deck):
    """A deck is sleeved, on the bench, or retired — and the front door says
    which in the second line, before any figure.

    The PRD calls the first "Pinned" (§3.1) and that word is deliberately NOT
    used: the same document uses "pin" for the immutable decklist hash (§3.2),
    and one word for a deck's physical existence and for a content sha is the
    collision the ManaMap/Atlas rename exists to avoid.
    """
    info = deck_info.compose(SLUG)
    assert deck_info.deck_state(info)[0] == deck_info.STATE_ON_BENCH

    (bare_deck / "deck_versions.json").write_text(json.dumps(
        {"slug": SLUG, "tags": {},
         "paper": {"version": 1, "built_at": "2026-08-25", "note": "sleeved"}}))
    info = deck_info.compose(SLUG)
    state, why = deck_info.deck_state(info)
    assert state == deck_info.STATE_SLEEVED
    assert "V1" in why and "2026-08-25" in why

    # Dead outranks sleeved: a broken-down deck may still carry an old lock, and
    # "SLEEVED" over cards that are in another deck's sleeves is the exact lie
    # this vocabulary exists to prevent.
    (bare_deck / "issue.json").write_text(json.dumps(
        {"commander": "Radagast of Rhosgobel", "status": "broken-down"}))
    assert deck_info.deck_state(deck_info.compose(SLUG))[0] == deck_info.STATE_RETIRED


def test_the_header_names_the_deck_not_the_landing_page(bare_deck, capsys):
    """It read "WORKBENCH — <slug>" at the top of a per-deck command.

    Harmless while "workbench" meant the whole bench; wrong once the Workbench
    became one surface — the landing page — with the Atlas as a tool on it.
    """
    deck_info.main(type("A", (), {"slug": SLUG, "as_json": False, "write": False})())
    out = capsys.readouterr().out
    first = out.strip().splitlines()[0]
    assert first.startswith(SLUG.upper()), first
    assert "WORKBENCH" not in first
