"""Sven's deterministic half — no API key, no network, no model.

The whole point of the `sven/` split is that this file can exist. If proving
Sven's facts required a key, CI could not prove them at all, and the first thing
to rot would be the part that decides whether an answer is stale.

What is asserted here:

  - the tool surface IS `serve.CLI_READONLY`, derived rather than restated, so a
    command added there becomes a capability and one removed revokes it
  - every write path is refused, and refused by the DISPATCHER rather than by
    instructions in a prompt
  - the fact cache returns byte-identical results and re-runs when inputs move
  - the answer cache self-invalidates, which is the only property that makes it
    safe to have at all
  - a turn that read a volatile command is never cacheable
"""

import json

import pytest

from manamap.sven import cache, core, stream, tools

from conftest import requires_deck


# ── the tool surface is derived, not declared ─────────────────────────────

def test_the_tool_surface_is_exactly_the_servers_allow_list():
    """Two lists of "what is safe" would disagree within a month. There is one."""
    from manamap.serve import CLI_READONLY

    assert {name for name, _desc in tools.readonly_commands()} == set(CLI_READONLY)


def test_every_command_carries_its_own_description():
    """Descriptions come from `PILOT_STEPS`, so a reworded help string reaches
    Sven without anyone editing `sven/tools.py`."""
    rows = tools.readonly_commands()
    assert len(rows) >= 15
    assert all(desc and desc.strip() for _name, desc in rows)


# ── the refusals are structural ───────────────────────────────────────────

@pytest.mark.parametrize("argv", [
    ["deck-info", "heliod", "--write"],       # a read-only command with a write flag
    ["goldfish", "heliod"],                   # writes a tracked artifact
    ["deck-branch", "heliod", "merge"],       # spends cardboard
    ["deck-delete", "heliod"],                # destructive
    ["promote", "heliod", "--to", "sleeved"], # moves the ladder
])
def test_sven_cannot_reach_a_write_path(argv):
    """Refused by `serve._cli`'s own two gates, not by anything Sven is told.

    A charter instruction is a request; this is a wall. The distinction matters
    because the charter is prose a model reads and the gate is code it runs
    inside.
    """
    session = core.Session()
    got = session.call("run_readonly", argv=argv)
    assert "error" in got, f"{argv} was not refused"
    assert not session.cacheable(), "a refused turn must not be cacheable"


def test_an_unknown_tool_names_what_does_exist():
    session = core.Session()
    got = session.call("delete_everything")
    assert "error" in got
    assert "run_readonly" in got["error"]


# ── the fact cache ────────────────────────────────────────────────────────

@requires_deck
def test_the_fact_cache_returns_the_same_bytes_and_says_it_cached():
    facts = cache.FactCache()
    first = tools.run(["deck-status", "heliod"], facts=facts)
    second = tools.run(["deck-status", "heliod"], facts=facts)
    assert first["stdout"] == second["stdout"]
    assert first["cached"] is False and second["cached"] is True
    assert facts.stats()["hits"] == 1


@requires_deck
def test_the_fact_cache_re_runs_when_the_deck_changes(tmp_path, monkeypatch):
    """The property that makes it safe. A cache that cannot notice a changed
    input is a cache that reports yesterday's deck."""
    deck = tmp_path / "decks" / "probe"
    deck.mkdir(parents=True)
    (deck / "decklist.txt").write_text("1 Island\n")

    facts = cache.FactCache()
    calls = []
    build = lambda: (calls.append(1), "value")[1]

    facts.get_or_call("k", [deck], build)
    facts.get_or_call("k", [deck], build)
    assert len(calls) == 1, "second call should have hit the cache"

    (deck / "decklist.txt").write_text("1 Island\n1 Plains\n")
    facts.get_or_call("k", [deck], build)
    assert len(calls) == 2, "a changed input must re-run the build"


def test_an_artifact_appearing_invalidates_the_answer_that_said_it_was_missing(tmp_path):
    """"There is no engine model for this deck yet" is an answer produced entirely
    by reads that FAILED, and it must stop being true the moment one appears.

    This works because `depends_on` names the DECK DIRECTORY, not the files a
    command happened to open — so the walk picks up an artifact nobody predicted
    would be looked for. A recorder that logged only successful opens would hold
    that answer forever, which is the subtle half of this problem and the reason
    directory granularity is the right coarseness rather than a compromise.
    """
    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "decklist.txt").write_text("1 Island\n")
    before = cache.signature([deck])

    (deck / "engine.json").write_text('{"stages": []}')
    after = cache.signature([deck])
    assert after != before, "an artifact appearing must invalidate the answer"

    (deck / "engine.json").unlink()
    assert cache.signature([deck]) == before, (
        "and removing it must return the signature to what it was")


def test_churn_that_must_not_invalidate_is_skipped(tmp_path):
    """`.agent-cache.json` moves when an unrelated routine is recorded, and a
    running Forge batch rewrites `logs/` continuously. Neither is a reason to
    drop an answer about a decklist."""
    deck = tmp_path / "deck"
    (deck / "logs").mkdir(parents=True)
    (deck / "decklist.txt").write_text("1 Island\n")
    before = cache.signature([deck])

    (deck / ".agent-cache.json").write_text('{"noise": 1}')
    (deck / "logs" / "part-1.log").write_text("Game Result: Game 1 ended in 5 ms\n")
    assert cache.signature([deck]) == before

    (deck / "decklist.txt").write_text("1 Island\n1 Plains\n")
    assert cache.signature([deck]) != before, "a real edit must still register"


# ── the answer cache ──────────────────────────────────────────────────────

def test_an_answer_is_served_back_and_then_invalidated(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "SVEN_CACHE_DIR", tmp_path / "answers")
    art = tmp_path / "deck.json"
    art.write_text("{}")

    sig = cache.signature([art])
    cache.answer_put("is zur ready?", sig, "haiku", "Four of six gates.")
    hit = cache.answer_get("is zur ready?", sig, "haiku")
    assert hit and hit["answer"] == "Four of six gates."

    art.write_text('{"changed": true}')
    assert cache.answer_get("is zur ready?", cache.signature([art]), "haiku") is None, (
        "the answer must not survive a change to what it read")


def test_the_model_is_in_the_answer_key(tmp_path, monkeypatch):
    """An escalated turn's answer must not be served later from the cheap
    model's entry — they are different artifacts."""
    monkeypatch.setattr(cache, "SVEN_CACHE_DIR", tmp_path / "answers")
    cache.answer_put("q", "sig", "haiku", "the quick read")
    assert cache.answer_get("q", "sig", "sonnet") is None
    assert cache.answer_get("q", "sig", "haiku")["answer"] == "the quick read"


def test_wording_of_the_question_is_normalised(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "SVEN_CACHE_DIR", tmp_path / "answers")
    cache.answer_put("Is  Zur READY?", "sig", "haiku", "yes")
    assert cache.answer_get("is zur ready?", "sig", "haiku")["answer"] == "yes"


@requires_deck
def test_a_volatile_command_makes_the_turn_uncacheable():
    """`sim-progress` reads logs the signature deliberately skips, so its answer
    is true only when asked. Caching it would report a finished run as 40% done."""
    session = core.Session()
    session.call("run_readonly", argv=["sim-progress", "heliod"])
    assert not session.cacheable()
    assert any("moment it is asked" in r for r in session.uncacheable)


def test_a_turn_that_read_nothing_says_so_rather_than_looking_empty():
    session = core.Session()
    session.call("stat_test", kind="wilson", k=27, n=120)
    assert session.touched_signature() == "no-files-read"
    assert session.cacheable()


# ── statistics are never paraphrased ──────────────────────────────────────

def test_stat_test_returns_the_librarys_own_value():
    """Sven prints what `stats.py` returns. Misreading an interval is a failure
    this bench has paid for, and the fix is to remove the opportunity."""
    from manamap.sim import stats

    got = tools.stat_test("wilson", k=27, n=120)
    assert got["result"] == stats.wilson_bounds(27, 120)
    assert got["test"] == "wilson"


def test_an_unknown_test_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="diff_proportions"):
        tools.stat_test("t_test", k=1, n=2)


# ── the wire keeps the answer out of the theatre ──────────────────────────

def test_only_answer_text_reaches_stdout():
    """`console.py`'s first rule: stdout is the ANSWER, stderr is the theatre.
    `mm ask --json | jq` has to stay byte-clean while Sven narrates."""
    import io

    out, err = io.StringIO(), io.StringIO()
    for kind, data in [("tool", "reading zur's gates"), ("text", "Four of six."),
                       ("note", "60 games is about 18 min"), ("done", {"summary": "ok"})]:
        stream.route(kind, data, out, err)
    assert out.getvalue() == "Four of six."
    assert "reading zur's gates" in err.getvalue()
    assert "Four of six." not in err.getvalue()


def test_frames_round_trip():
    frames = [("text", "hello"), ("tool", "reading"), ("done", {"n": 1})]
    blob = "".join(stream.encode(k, d) for k, d in frames)
    assert list(stream.iter_frames(blob.splitlines(keepends=True))) == frames


def test_a_payload_containing_a_newline_does_not_split_a_frame():
    blob = stream.encode("text", "line one\nline two")
    assert list(stream.iter_frames(blob.splitlines(keepends=True))) == [
        ("text", "line one\nline two")]


def test_an_unknown_frame_kind_is_refused_at_the_encoder():
    with pytest.raises(ValueError, match="not one of"):
        stream.encode("shell_command", "rm -rf /")
