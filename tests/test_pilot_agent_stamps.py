"""AN AGENT ARTIFACT MUST SAY WHAT LIST IT WAS WRITTEN AGAINST.

`regen` keeps the MEASURED artifacts current on a sleeved deck automatically —
goldfish, mana-analysis, diagnose, benchmark. It deliberately does not touch the
AGENT artifacts, because those cost tokens and a sweep that re-spawns six agents
per deck per decklist edit is not a sweep anybody would leave switched on.

Which makes staleness DETECTION the only thing standing between a sleeved deck
and a green board over rotted prose. And it was not working:

    engine.json          stamped 0, unstamped 9
    tutor_guide.json     stamped 0, unstamped 9
    manual_prose.json    stamped 1, unstamped 8
    strategic_frame.json stamped 3, unstamped 6

`deck_status.STAGES` has declared `engine.json`'s staleness key as
`decklist_sha256` for months. No engine model in the fleet carried any sha, so
every row read "?" — staleness unknown — permanently, and a stale artifact was
indistinguishable from a fresh one.

WHAT THAT COST, twice. `deck_status.py` records the first: edgar-vampires'
engine named twelve cards that were not in the 99. The second was found by
accident on 2026-09-02, when a tutor-guide agent mentioned in passing that
ur-dragon's engine named SEVENTEEN — Counterspell, Smothering Tithe, Swan Song,
Moltensteel Dragon among them.

The fix is `install_agent`, which is the one place a whole-file handoff becomes
a tracked artifact, and which stamps. These tests are what stop the next
`cp` from going around it.
"""

import json

import pytest

from manamap import config
from manamap.config import AGENT_ROUTINES, DECKS_DIR
from manamap.pilot import install_agent
from manamap.pilot.deck_status import STAGES
from conftest import requires_deck

#: Artifacts whose staleness `deck_status` claims to check.
STAMP_CHECKED = {name: sha for _k, name, sha, _r, _w, _h in STAGES if sha}


def test_every_stamp_checked_artifact_has_a_way_to_be_stamped():
    """A DECLARED CHECK WITH NO PRODUCER IS A CHECK THAT CANNOT FIRE.

    `STAGES` naming a sha key for an artifact is a promise that something writes
    it. For `engine.json` and `tutor_guide.json` nothing did, and the promise
    went unkept for months while the row read "?" and everybody read past it.
    """
    from manamap.pilot import merge_prose, merge_debrief, merge_captains_log  # noqa: F401

    producers = set()
    for routine, spec in AGENT_ROUTINES.items():
        if routine in install_agent.MERGED_ELSEWHERE:
            producers.add(spec["artifact"])   # its own merge module stamps
        else:
            producers.add(spec["artifact"])   # install-agent stamps
    # Artifacts produced deterministically by Python stamp themselves.
    deterministic = {"deck_map.json", "goldfish_metrics.json", "mana_analysis.json",
                     "bracket_report.json", "benchmark.json", "diagnostic.json",
                     "cards.json", "info.json", "versions.json", "log.jsonl",
                     "issue.json", "deck_versions.json", "pending.json"}
    orphans = sorted(a for a in STAMP_CHECKED
                     if a not in producers and a not in deterministic
                     and not a.endswith("/"))
    assert not orphans, (
        f"{orphans} have a declared staleness key in deck_status.STAGES and "
        f"nothing that writes one — the check can never fire")


def test_install_refuses_to_stamp_a_deck_it_cannot_date(tmp_path, monkeypatch):
    """A WRONG STAMP IS WORSE THAN NO STAMP.

    An absent stamp reads as UNKNOWN, which is honest. A stamp taken from
    nowhere reads as CURRENT, which is the lie the missing stamp merely allowed.
    So a deck with no `cards.json` sha is refused rather than stamped with
    whatever was to hand.
    """
    deck = tmp_path / "decks" / "nodate"
    (deck / ".agent-out").mkdir(parents=True)
    (deck / ".agent-out" / "deck-engineer.json").write_text('{"slug": "nodate"}')
    (deck / "cards.json").write_text('{"cards": []}')      # no decklist_sha256
    monkeypatch.setattr(config, "DECKS_DIR", tmp_path / "decks")

    with pytest.raises(SystemExit) as e:
        install_agent.install("nodate", "deck-engine")
    assert "no decklist_sha256" in str(e.value)

    # --force installs it UNSTAMPED, and says so — never with a borrowed sha.
    dst, sha = install_agent.install("nodate", "deck-engine", force=True)
    assert sha is None
    assert install_agent.STAMP_KEY not in json.loads(dst.read_text())


def test_a_merged_artifact_is_refused_by_name(tmp_path, monkeypatch):
    """`log_annotations.json` ACCUMULATES — a whole-file copy would drop every
    annotation already in it. The refusal names the right command rather than
    failing vaguely, because the two operations look identical from outside."""
    for routine in ("debrief", "captains-log", "pilot-notes"):
        with pytest.raises(SystemExit) as e:
            install_agent.install("anything", routine)
        assert "MERGED, not replaced" in str(e.value)
        assert "manamap pilot" in str(e.value)


def test_install_stamps_and_keeps_the_previous_copy(tmp_path, monkeypatch):
    deck = tmp_path / "decks" / "d"
    (deck / ".agent-out").mkdir(parents=True)
    (deck / ".agent-out" / "deck-engineer.json").write_text('{"thesis": "new"}')
    (deck / "cards.json").write_text('{"decklist_sha256": "' + "a" * 64 + '"}')
    (deck / "engine.json").write_text('{"thesis": "old"}')
    monkeypatch.setattr(config, "DECKS_DIR", tmp_path / "decks")

    dst, sha = install_agent.install("d", "deck-engine")
    doc = json.loads(dst.read_text())
    assert doc["thesis"] == "new"
    assert doc[install_agent.STAMP_KEY] == "a" * 12
    # The replaced copy survives beside the handoff — a whole-file install is
    # destructive and the previous model is what a revision is judged against.
    prev = json.loads((deck / ".agent-out" / "engine.json.prev").read_text())
    assert prev["thesis"] == "old"


@requires_deck
def test_no_agent_artifact_names_a_card_the_deck_does_not_run():
    """THE FAILURE ITSELF, held at rest on the fleet.

    Not a stamp check — a check that the thing the stamp exists to catch is not
    presently true. `engine.json` places cards into stages by name, so a card
    that has left the 99 is mechanically detectable, and seventeen of them sat
    in a tracked model until an unrelated agent noticed.

    Scoped to SLEEVED decks. A deck on the bench changes daily and its agent
    artifacts are allowed to lag; a deck that exists in cardboard is one the
    pilot is reading before a game.
    """
    from manamap.pilot.common import expand_copies, load_json
    from manamap.pilot import regen

    checked, bad = 0, []
    for deck in sorted(DECKS_DIR.iterdir()):
        if not deck.is_dir() or not regen.is_pinned(deck.name):
            continue
        eng = load_json(deck / "engine.json")
        cards = load_json(deck / "cards.json")
        if not eng or not cards:
            continue
        have = {c["name"] for c in expand_copies(cards.get("cards", []))}
        gone = sorted({n for s in eng.get("stages", [])
                       for n in (s.get("cards") or []) if n not in have})
        checked += 1
        if gone:
            bad.append(f"{deck.name}: {len(gone)} card(s) not in the 99 — "
                       f"{', '.join(gone[:5])}{'…' if len(gone) > 5 else ''}")
    assert checked >= 1, "no sleeved deck has an engine model to check"
    assert not bad, "\n".join(
        ["a sleeved deck's engine model names cards it no longer runs "
         "(re-run /analyze-engine, then `install-agent --routine deck-engine`):"]
        + bad)


# ── the ✓ tier, which rots without any citation becoming wrong ──────────

def test_a_finished_stack_is_checked_against_the_deck_it_claims_to_be_from(tmp_path):
    """A CHECK THAT EXISTS AND IS UNREACHABLE IS NOT A CHECK.

    `validate_stack.unknown_cards` was written carefully, works, and was called
    from exactly one place: the `--scenario-only` preflight that runs BEFORE a
    resolver is spawned. So every scenario was checked on the way in and no
    scenario was ever checked again — while the thing it guards against, a
    decklist moving underneath its stacks, happens on every swap.

    Measured the day it was wired into the normal path: 6 of edgar-vampires' 11
    checker-passed stacks named a card the deck no longer runs, and 3 of
    ur-dragon's. All eleven printed OK. The CITATIONS were still correct — CR
    rules do not rot — and what had rotted was the board they were cited
    against, which nothing looked at.

    That is the worst shape a staleness bug can take here: these are the ✓ tier,
    the only fact tier in the repo, and they render under a heading that says
    this is how the deck wins.
    """
    import json
    import os
    import subprocess
    import sys

    from manamap.pilot import validate_stack

    # THE UNIT, on a scenario this test builds. It used to assert that edgar's
    # committed stack 001 still names Exquisite Blood — a card the SLEEVED deck
    # dropped — so re-adding the card, or re-resolving the stack, reddened a
    # green suite. A test must not need a live deck to stay wrong.
    scenario = {"id": "001", "slug": "edgar-vampires", "deck": "Edgar",
                "title": "t", "scenario": {
                    "board": {"you": ["Exquisite Blood"],
                              "opponents": [{"life": 40, "board": []}]},
                    "hand": [], "mana_available": "{0}", "stack": [],
                    "question": "?"}}
    _errs, warns = validate_stack.unknown_cards(scenario, "edgar-vampires")
    assert any("Exquisite Blood" in w for w in warns), (
        "unknown_cards no longer detects a card the deck does not run")

    # THE REGRESSION THAT ACTUALLY BIT: the command must SURFACE it. Re-introduce
    # the bug by deleting the `unknown_cards` call from main's normal path and
    # this fails while the assertion above still passes — which is exactly how
    # the bug survived.
    #
    # BUILT, NOT FOUND. This used to scan the fleet for a deck with an UNMARKED
    # stale stack and `pytest.skip` when there was none — so the day the fleet
    # is clean, which is the goal, the check loses its coverage with no signal.
    # A stale deck in a tmp data dir proves the same thing forever.
    from manamap import config as _config

    # Everything but `decks/` is symlinked from the real data dir: the command
    # loads the rules index and the corpus, and neither has anything to do with
    # what is being proved here.
    data = tmp_path / "data"
    data.mkdir()
    for child in _config.DATA_DIR.iterdir():
        if child.name != "decks":
            (data / child.name).symlink_to(child)
    deck = data / "decks" / "staleish"
    (deck / "stacks").mkdir(parents=True)
    (deck / "cards.json").write_text(json.dumps({"cards": [
        {"name": "Sol Ring", "quantity": 1, "type_line": "Artifact",
         "oracle_text": "", "cmc": 1.0}]}), encoding="utf-8")
    # A REAL checker-passed artifact, re-slugged onto a deck that runs none of
    # its cards. Hand-rolling one would have to satisfy the citation contract,
    # which is a different test's job.
    template = json.loads(
        (_config.DECKS_DIR / "edgar-vampires" / "stacks"
         / "001-exquisite-vito-drain-loop.json").read_text(encoding="utf-8"))
    template["slug"] = "staleish"
    template["deck"] = "Staleish"
    template.pop("presentable", None)
    template.pop("presentable_note", None)
    assert (template.get("checker") or {}).get("verdict") == "pass", (
        "the template stack must be checker-passed, or the warning under test "
        "is not the one that fires on a PASSED line")
    (deck / "stacks" / "001-stale.json").write_text(
        json.dumps(template), encoding="utf-8")

    env = dict(os.environ, MANAMAP_DATA_DIR=str(data))
    out = subprocess.run(
        [sys.executable, "-m", "manamap.cli", "pilot", "validate-stack", "staleish"],
        capture_output=True, text=True, check=False, env=env)
    assert "does not run" in out.stdout, (
        "validate-stack ran clean on a deck whose checker-PASSED, unwithheld "
        "stack names a card the 99 does not run — the check is reachable only "
        f"from --scenario-only again.\nstdout: {out.stdout}\nstderr: {out.stderr}")


def test_the_stale_stack_warning_never_fails_the_gate():
    """A FINISHED ARTIFACT IS FINISHED WORK.

    Failing these would cost a resolver respawn per stack to fix something
    nobody misread, and some of them SHOULD be retired rather than re-resolved —
    edgar's 008 and 009 prove a Purphoros token-conversion line the pilot's own
    captain's log records as superseded. Re-proving an abandoned plan is worse
    than leaving it marked.

    So it warns, and the exit code stays clean. What it must never do is go
    quiet.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "manamap.cli", "pilot", "validate-stack",
         "edgar-vampires"],
        capture_output=True, text=True, check=False)
    assert out.returncode == 0, (
        "a stale board must warn, not fail — see the docstring")
    assert "FAIL" not in out.stdout


def test_withholding_a_stack_is_the_answer_to_the_warning_not_a_second_one():
    """A DECISION ALREADY TAKEN MUST NOT KEEP WARNING.

    `presentable: false` with a `presentable_note` IS the response to "this
    board holds a card the deck no longer runs" — the pilot read it, decided the
    line describes a board v1.0.1 cannot make, and withheld it with the card
    named. Warning again on every run afterwards is how a reader learns the mark
    means nothing, which is the same failure as a validator that fires on
    correct data.

    The resolution text of a withheld stack is NOT edited. It passed the
    citation contract, so its steps are evidence, and rewriting them post-hoc
    would put a checker's tick over words no checker read. Only the curation
    flag moves.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    from manamap.config import DECKS_DIR

    withheld = []
    for f in sorted((DECKS_DIR / "edgar-vampires" / "stacks").glob("*.json")):
        doc = json.loads(f.read_text())
        if doc.get("presentable") is False:
            withheld.append(f.name)
            assert doc.get("presentable_note"), (
                f"{f.name} is withheld with no note — the note is the reader's "
                f"only account of why a checker-passed line is not shown")
            assert (doc.get("checker") or {}).get("verdict") == "pass", (
                f"{f.name} is withheld but did not pass — withholding is for "
                f"lines whose RULES finding stands and whose board has moved")
    # NO FLOOR. This asserted `>= 4`, which required at least four of edgar's
    # checker-passed stacks to stay withheld BECAUSE their boards name cards the
    # deck dropped — so re-resolving them or retiring them, the two honest
    # things to do, reddened the suite. The per-stack invariants above are the
    # real content and they hold for however many there are; if there are none,
    # there is nothing to prove here and that is a good state, not a failure.
    if not withheld:
        pytest.skip("no stack is currently withheld — nothing to prove the "
                    "note-and-verdict invariant on")

    out = subprocess.run(
        [sys.executable, "-m", "manamap.cli", "pilot", "validate-stack",
         "edgar-vampires"],
        capture_output=True, text=True, check=False)
    assert "does not run" not in out.stdout, (
        "a withheld stack warned anyway — the warning must stop once the "
        "decision is on record, or it becomes noise")
