"""The turn loop, driven with a scripted transport — no key, no network.

`ScriptedTurn` is not a mock of the model; it is a real implementation of the
`Turn` protocol that happens to replay canned events. That distinction matters:
the loop, the tools, the caches and the wire are all exercised for real here,
and the only thing standing in for the model is the model's own output.

What this file protects:

  - a refusal reaches the MODEL as content, so it can correct itself, rather
    than raising and ending the turn
  - a statistical answer is the library's own return value, never re-typed
  - escalation carries the work already done instead of restarting
  - the answer cache round-trips and then invalidates
  - a tool result too large to be useful is truncated with a way out
"""

import json

import pytest

from manamap.sven import cache, core, llm, loop, tools

from conftest import requires_deck


def _turn(*rounds):
    return llm.ScriptedTurn(list(rounds))


def _end(reason="end_turn"):
    return {"type": "end", "stop_reason": reason}


def _use(tool_id, name, **inp):
    return {"type": "tool_use", "id": tool_id, "name": name, "input": inp}


def _collect(frames):
    out = {"text": "", "tool": [], "note": [], "error": [], "done": None}
    for kind, data in frames:
        if kind == "text":
            out["text"] += data
        elif kind == "done":
            out["done"] = data
        else:
            out[kind].append(data)
    return out


# ── the loop runs real tools ──────────────────────────────────────────────

@requires_deck
def test_a_turn_runs_the_command_and_narrates_it():
    turn = _turn(
        [_use("t1", "run_command", command="deck-status", args=["heliod"]),
         _end("tool_use")],
        [{"type": "text", "text": "heliod is sleeved."}, _end()],
    )
    got = _collect(loop.run("where is heliod?", turn=turn, use_cache=False))
    assert got["text"] == "heliod is sleeved."
    assert got["tool"] == ["reading deck-status heliod"]
    assert got["done"]["tool_calls"] == 1

    # The model must actually have RECEIVED the command's output.
    second = turn.seen[1]["messages"]
    result = second[-1]["content"][0]
    assert result["type"] == "tool_result" and not result.get("is_error")
    assert "heliod" in result["content"]


@requires_deck
def test_a_refusal_goes_back_to_the_model_rather_than_ending_the_turn():
    """The model should see argparse's own message and correct itself — one
    round trip, self-teaching. Raising instead would end the turn with nothing,
    and the pilot would see a crash where a retry belonged."""
    turn = _turn(
        [_use("t1", "run_command", command="deck-info", args=["heliod", "--write"]),
         _end("tool_use")],
        [{"type": "text", "text": "I cannot write; here is the read-only view."}, _end()],
    )
    got = _collect(loop.run("update heliod's dossier", turn=turn, use_cache=False))
    assert got["error"], "the refusal should have been narrated"
    result = turn.seen[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "--write" in result["content"]


def test_an_unknown_command_is_refused_before_it_runs():
    turn = _turn(
        [_use("t1", "run_command", command="deck-delete", args=["heliod"]),
         _end("tool_use")],
        [{"type": "text", "text": "That is not something I can do."}, _end()],
    )
    got = _collect(loop.run("delete heliod", turn=turn, use_cache=False))
    assert got["error"]
    assert got["done"]["cacheable"] is False


# ── statistics are the library's, not the model's ─────────────────────────

def test_a_statistic_is_the_librarys_own_return_value():
    """Sven may not re-type an interval. The tool returns `stats.py`'s value and
    the model is handed that JSON verbatim."""
    from manamap.sim import stats

    turn = _turn(
        [_use("t1", "stats", fn="wilson", args={"k": 27, "n": 120}), _end("tool_use")],
        [{"type": "text", "text": "0.252 [0.18, 0.34]"}, _end()],
    )
    _collect(loop.run("what is heliod's win rate interval?", turn=turn, use_cache=False))
    payload = json.loads(turn.seen[1]["messages"][-1]["content"][0]["content"])
    assert payload["result"] == list(stats.wilson_bounds(27, 120)) or \
           tuple(payload["result"]) == stats.wilson_bounds(27, 120)


def test_a_bad_statistic_call_returns_the_error_not_a_crash():
    turn = _turn(
        [_use("t1", "stats", fn="wilson", args={"k": 1}), _end("tool_use")],
        [{"type": "text", "text": "I need n as well."}, _end()],
    )
    got = _collect(loop.run("wilson for one win", turn=turn, use_cache=False))
    assert got["error"]
    assert got["done"]["cacheable"] is False


# ── escalation ────────────────────────────────────────────────────────────

def test_escalation_switches_model_and_keeps_the_work():
    """One more step, not a restart. The escalated round must still see whatever
    was already read, or escalating would cost the turn twice."""
    turn = _turn(
        [_use("t1", "stats", fn="wilson", args={"k": 27, "n": 120}), _end("tool_use")],
        [_use("t2", "escalate", reason="this is a comparison, not a lookup"),
         _end("tool_use")],
        [{"type": "text", "text": "The intervals overlap; that implies nothing."}, _end()],
    )
    got = _collect(loop.run("is heliod better than zur?", turn=turn, use_cache=False))
    assert got["done"]["escalated"] is True
    assert got["done"]["model"] == llm.DEEP_MODEL
    assert any("escalating" in n for n in got["note"])

    models = [s["model"] for s in turn.seen]
    assert models[0] == llm.FAST_MODEL and models[-1] == llm.DEEP_MODEL
    # the stats result is still in the transcript the strong model reads
    assert any("tool_result" in json.dumps(m) for m in turn.seen[-1]["messages"])


def test_deep_starts_on_the_strong_model_without_escalating():
    turn = _turn([{"type": "text", "text": "answer"}, _end()])
    got = _collect(loop.run("a hard one", turn=turn, model=llm.DEEP_MODEL,
                            use_cache=False))
    assert got["done"]["model"] == llm.DEEP_MODEL
    assert got["done"]["escalated"] is False


# ── the answer cache, through the loop ────────────────────────────────────

@requires_deck
def test_an_answer_round_trips_and_then_invalidates(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "SVEN_CACHE_DIR", tmp_path / "answers")
    q = "where is heliod?"
    turn = _turn(
        [_use("t1", "run_command", command="deck-status", args=["heliod"]),
         _end("tool_use")],
        [{"type": "text", "text": "heliod is sleeved."}, _end()],
    )
    _collect(loop.run(q, turn=turn))

    hit = loop.cached_answer(q)
    assert hit and hit["answer"] == "heliod is sleeved."

    # Move something the turn read. The stored answer must stop being served.
    from manamap.pilot.common import deck_dir
    probe = deck_dir("heliod") / ".sven-probe"
    probe.write_text("x")
    try:
        assert loop.cached_answer(q) is None, (
            "an answer must not survive a change to the deck it read")
    finally:
        probe.unlink()
    assert loop.cached_answer(q) is not None, "and must return once undone"


@requires_deck
def test_a_volatile_turn_is_never_stored(tmp_path, monkeypatch):
    """`sim-progress` is true only when asked. Storing it would report a
    finished run as still going."""
    monkeypatch.setattr(cache, "SVEN_CACHE_DIR", tmp_path / "answers")
    turn = _turn(
        [_use("t1", "run_command", command="sim-progress", args=["heliod"]),
         _end("tool_use")],
        [{"type": "text", "text": "Nothing running."}, _end()],
    )
    got = _collect(loop.run("how's that run going?", turn=turn))
    assert got["done"]["cacheable"] is False
    assert loop.cached_answer("how's that run going?") is None
    # For the RIGHT reason. This test first passed while `sim-progress` was not
    # in the allow-list at all, so the turn was uncacheable because the command
    # was REFUSED — a green test proving nothing about volatility.
    assert got["done"]["uncacheable"] == [
        "sim-progress is true only at the moment it is asked"]
    assert not got["error"], "the command should have run, not been refused"


# ── bounds ────────────────────────────────────────────────────────────────

def test_a_huge_tool_result_is_truncated_with_a_way_out():
    session = core.Session()
    huge = "x" * (tools.TOOL_RESULT_CAP + 5000)
    capped = loop._cap(huge)
    assert len(capped) < len(huge)
    assert "truncated" in capped and "--limit" in capped


def test_the_loop_stops_rather_than_calling_tools_forever():
    """A question that cannot be answered should fail in seconds, not dollars."""
    forever = llm.ScriptedTurn(
        lambda messages, tool_list: [
            _use(f"t{len(messages)}", "stats", fn="wilson", args={"k": 1, "n": 2}),
            _end("tool_use")])
    got = _collect(loop.run("loop forever", turn=forever, use_cache=False))
    assert any("stopped after" in n for n in got["note"])
    assert got["done"]["tool_calls"] == loop.MAX_ROUNDS


# ── the charter ───────────────────────────────────────────────────────────

def test_the_charter_reaches_the_model_and_carries_the_evidence_rules():
    turn = _turn([{"type": "text", "text": "ok"}, _end()])
    _collect(loop.run("hello", turn=turn, use_cache=False))
    system = turn.seen[0]["system"]
    for rule in ["interval", "Absent means absent", "read-only", "the bench"]:
        assert rule in system, f"the charter lost: {rule}"


def test_editing_the_charter_reaches_the_next_turn(tmp_path, monkeypatch):
    """Read per turn through `mtime_memo`, so the file a pilot edits daily does
    not need a server restart to take effect."""
    path = tmp_path / "prompt.md"
    path.write_text("first")
    monkeypatch.setattr(loop, "PROMPT_PATH", path)
    assert loop.charter() == "first"
    path.write_text("second")
    assert loop.charter() == "second"
