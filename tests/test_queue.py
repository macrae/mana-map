"""The hypothesis queue (PRD v2 Phase 2): data/queue.jsonl, state derived from lines.

Unit tests on a temporary file; the decks are faked. The tracked file's gate is the
last test (regression tier).
"""
import datetime
import json

import pytest

from manamap.pilot import queue as q
from manamap.pilot import validate_queue


@pytest.fixture
def qfile(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "PATH", tmp_path / "queue.jsonl")
    monkeypatch.setattr(q, "deck_dir", lambda slug: tmp_path if slug in ("sharknado", "edgar-vampires")
                        else tmp_path / "nope")
    monkeypatch.setattr(q, "last_played", lambda: {"sharknado": datetime.date(2026, 10, 6),
                                                   "edgar-vampires": datetime.date(2026, 9, 29)})
    return tmp_path / "queue.jsonl"


def hyp(claim="Adding a second wipe-proof draw engine stops the T6-8 stall", method="try"):
    return {"claim": claim, "expected_effect": "more cards drawn by T8",
            "test": {"method": method, "how": "try --out X --in Y, cards_drawn_8"}, "why": "logs 004-006"}


def incubate(deck="sharknado", n=1, **kw):
    return q.apply({"kind": "incubation", "deck": deck, "source": "pilot",
                    "hypotheses": [hyp(**kw) for _ in range(n)]})


def challenge(qid, verdict, reason="cheaper explanation: variance over six games"):
    return q.apply({"kind": "challenge", "items": [{"of": qid, "verdict": verdict, "reason": reason}]})


def state(qid, today=None):
    return q.state_of(qid, q.read(), today)


def test_the_whole_loop_derives_each_state_from_the_lines(qfile):
    incubate()
    assert state("Q001") == "INCUBATING"
    challenge("Q001", "promote")                       # apply settles the round: promote
    assert state("Q001") == "PROMOTED"
    q.apply({"kind": "result", "of": "Q001", "method": "try", "verdict": "supported",
             "answer": "+0.4 cards by T8, interval excludes zero", "evidence": ["manamap pilot try …"]})
    assert state("Q001") == "TESTED"
    q.append([{"kind": "decide", "of": "Q001", "by": "sean", "decision": "stage"}])
    assert state("Q001") == "DECIDED"
    assert not any("state" in e for e in q.read())   # nothing stores a state


def test_a_revise_gets_one_rebuttal_and_then_it_is_settled(qfile):
    incubate(n=2)
    challenge("Q001", "revise", "untestable as written: say which cards")
    challenge("Q002", "revise", "duplicate of the context's open question")
    assert state("Q001") == "CHALLENGED"
    q.apply({"kind": "rebuttal", "items": [
        {"of": "Q001", "action": "revise", "claim": "Rhystic Study in for Windfall draws more by T8",
         "why": "named the cards"},
        {"of": "Q002", "action": "withdraw", "why": "it is a duplicate"}]})
    assert state("Q001") == "PROMOTED" and state("Q002") == "DROPPED"
    assert q.current("Q001", q.read())["claim"].startswith("Rhystic Study")
    drop = next(e for e in q.read() if e["kind"] == "drop")
    assert "withdrawn" in drop["reason"]


def test_one_round_only(qfile):
    incubate()
    challenge("Q001", "revise", "say which cards")
    with pytest.raises(SystemExit, match="one round only"):
        challenge("Q001", "promote")
    q.apply({"kind": "rebuttal", "items": [{"of": "Q001", "action": "revise", "why": "named them"}]})
    with pytest.raises(SystemExit):
        q.apply({"kind": "rebuttal", "items": [{"of": "Q001", "action": "revise", "why": "again"}]})


def test_a_result_before_a_promote_is_refused_and_nothing_is_written(qfile):
    incubate()
    before = qfile.read_text()
    with pytest.raises(SystemExit, match="follows a promote"):
        q.apply({"kind": "result", "of": "Q001", "method": "try", "verdict": "supported", "answer": "x"})
    assert qfile.read_text() == before


def test_apply_refuses_unknown_ids_unknown_decks_and_phase_3_methods(qfile):
    with pytest.raises(SystemExit, match="names no item"):
        challenge("Q009", "promote")
    with pytest.raises(SystemExit, match="no deck"):
        incubate(deck="not-a-deck")
    with pytest.raises(SystemExit, match="Phase 3"):
        incubate(method="scenario")
    assert not qfile.exists() or qfile.read_text() == ""


def test_expiry_is_derived_from_dates_never_written(qfile):
    incubate()
    challenge("Q001", "promote")
    at = q._day(q.read()[-1]["at"])
    assert state("Q001", at + datetime.timedelta(days=13)) == "PROMOTED"
    assert state("Q001", at + datetime.timedelta(days=14)) == "EXPIRED"
    assert state("Q001") == "PROMOTED"                 # no `today`: the Deck Context's view


def test_order_is_seans_rank_then_recency_of_play_then_age(qfile):
    incubate(deck="edgar-vampires")                     # Q001: edgar, played 09-29
    incubate(deck="sharknado")                          # Q002: sharknado, played 10-06
    incubate(deck="edgar-vampires")                     # Q003: edgar, newer than Q001
    assert q.ordered(q.read()) == ["Q002", "Q001", "Q003"]
    q.append([{"kind": "rank", "by": "sean", "order": ["Q003"]}])
    assert q.ordered(q.read()) == ["Q003", "Q002", "Q001"]


def test_closed_items_take_only_a_kill_and_a_kill_needs_a_reason(qfile):
    incubate()
    challenge("Q001", "drop", "already answered in the context")
    assert state("Q001") == "DROPPED"
    with pytest.raises(SystemExit, match="closed"):
        q.append([{"kind": "promote", "of": "Q001", "by": "jarvis"}])
    with pytest.raises(SystemExit, match="needs a reason"):
        q.append([{"kind": "kill", "of": "Q001", "by": "sean"}])


def test_more_sends_a_tested_item_back_to_be_tested_again(qfile):
    incubate()
    challenge("Q001", "promote")
    res = {"kind": "result", "of": "Q001", "method": "data", "verdict": "inconclusive", "answer": "thin"}
    q.apply(res)
    q.append([{"kind": "decide", "of": "Q001", "by": "sean", "decision": "more"}])
    assert state("Q001") == "PROMOTED"
    q.apply(res)
    assert state("Q001") == "TESTED"


def test_the_validator_replays_the_file_and_catches_a_hand_edit(qfile):
    incubate()
    challenge("Q001", "promote")
    assert validate_queue.validate(q.read()) == []
    lines = q.read()
    lines.append({"kind": "challenge", "of": "Q001", "verdict": "drop", "reason": "late", "at": "2026-10-07"})
    lines.append({"id": "Q005", "kind": "hypothesis", "deck": "sharknado", "at": "2026-10-07", **hyp()})
    errors = validate_queue.validate(lines)
    assert any("closed" in e or "one round" in e for e in errors)
    assert any("not the next id" in e for e in errors)


def test_the_deck_block_lists_live_items_and_counts_the_closed(qfile):
    incubate(n=2)
    challenge("Q001", "promote")
    challenge("Q002", "drop", "variance")
    block = q.deck_block("sharknado")
    assert "**Q001** promoted" in block and "Q002" not in block and "1 dropped or killed" in block
    assert "Nothing queued" in q.deck_block("edgar-vampires")


def test_list_json_is_the_live_queue(qfile, capsys):
    incubate()
    q.main(type("A", (), {"verb": "list", "rest": [], "json": True, "all": False, "deck": None})())
    rows = json.loads(capsys.readouterr().out)
    assert [r["id"] for r in rows] == ["Q001"] and rows[0]["state"] == "INCUBATING"


def test_seans_own_claim_enters_as_a_hypothesis_and_still_faces_the_challenger(qfile):
    """`add` skips the pod, never the round: a claim of Sean's is INCUBATING until
    the Challenger has argued once, exactly like one the pod proposed."""
    e = q.add("sharknado", "Brallin dies before T6 in most games", "protection beats ramp",
              "argument", "break-even on the recast tax", why="game 001")
    assert e["id"] == "Q001" and e["by"] == "sean" and state("Q001") == "INCUBATING"
    with pytest.raises(SystemExit):        # no result before the round
        q.apply({"kind": "result", "of": "Q001", "method": "argument", "verdict": "supported",
                 "answer": "yes"})
    challenge("Q001", "promote")
    assert state("Q001") == "PROMOTED"


def test_add_is_held_to_the_hypothesis_rule_and_writes_nothing_when_refused(qfile):
    for bad in (dict(expected_effect=""), dict(method="scenario"), dict(method="vibes"),
                dict(how=" "), dict(deck="no-such-deck")):
        kw = dict(deck="sharknado", claim="c", expected_effect="e", method="try", how="h") | bad
        with pytest.raises(SystemExit):
            q.add(**kw)
    assert q.read() == []


@pytest.mark.regression
def test_the_tracked_queue_passes_its_gate():
    if not q.PATH.exists():
        pytest.skip("no queue yet — absent means absent")
    assert validate_queue.validate(q.read()) == []
