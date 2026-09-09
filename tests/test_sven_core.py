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
import re

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
    got = session.call("run_command", command=argv[0], args=argv[1:])
    assert "error" in got, f"{argv} was not refused"
    assert not session.cacheable(), "a refused turn must not be cacheable"


def test_an_unknown_tool_names_what_does_exist():
    session = core.Session()
    got = session.call("delete_everything")
    assert "error" in got
    assert "run_command" in got["error"]


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
    session.call("run_command", command="sim-progress", args=["heliod"])
    assert not session.cacheable()
    assert any("moment it is asked" in r for r in session.uncacheable)


def test_a_turn_that_read_nothing_says_so_rather_than_looking_empty():
    session = core.Session()
    session.call("stats", fn="wilson", args={"k": 27, "n": 120})
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


# ── what the first four real turns taught ─────────────────────────────────

def test_every_advertised_tool_can_actually_be_called():
    """SVEN'S FIRST WRONG ANSWER WAS THIS BUG. Three lists disagreed: the tool
    block advertised four capabilities, the session's dispatcher knew six, and
    the loop's result handler knew three. Asked "is zur ready" he could not
    reach `deck_state`, fell back to `deck-status`, and reported LIFECYCLE
    STAGES as PROMOTION GATES — with the real blocker never mentioned.

    A capability that exists and is not reachable is worse than one that does
    not exist: the model routes around it and sounds just as certain.

    Asserted by CALLING each one, not by grepping for its name. The earlier
    version of this test read the source of the handler chain, and went obsolete
    the moment that chain became a pass-through — a test that checks HOW rather
    than WHETHER stops meaning anything the moment the how changes.
    """
    from manamap.sven import api

    probes = {
        "deck_state": {"slug": "heliod"},
        "fleet": {},
        "search_docs": {"query": "evidence contract", "k": 1},
        "search_code": {"query": "goldfish", "k": 1},
        "stats": {"fn": "wilson", "args": {"k": 1, "n": 10}},
        "run_command": {"command": "deck-status", "args": ["heliod"]},
        "command_help": {"command": "card-search"},
    }
    session = core.Session()
    checked = 0
    for entry in api._ensure():
        if entry["handler"] is None:          # `escalate` is handled by the loop
            continue
        name = entry["name"]
        assert name in probes, f"{name} is advertised with no probe in this test"
        got = session.call(name, **probes[name])
        assert not (isinstance(got, dict) and set(got) == {"error"}), (
            f"{name} is advertised and cannot be called: {got}")
        checked += 1
    assert checked >= 6


@requires_deck
def test_the_pilots_shorthand_resolves_to_a_deck():
    """"zur" is what a person says. The first time this tool was reachable, Sven
    passed the shorthand straight through, got a FileNotFoundError, and offered
    to create a new deck."""
    assert tools.resolve_slug("zur") == "zur-enchantress"
    assert tools.resolve_slug("ur") == "ur-dragon"
    assert tools.resolve_slug("heliod") == "heliod"


@requires_deck
def test_an_ambiguous_shorthand_is_refused_rather_than_guessed():
    """The wrong deck's figures are indistinguishable from the right deck's
    until someone notices they describe another list."""
    with pytest.raises(ValueError, match="Say which one"):
        tools.resolve_slug("")            # matches everything


@requires_deck
def test_an_unknown_deck_lists_the_ones_that_exist():
    with pytest.raises(ValueError, match="On the bench"):
        tools.resolve_slug("no-such-deck-anywhere")


@requires_deck
def test_a_dependency_walk_never_decides_a_deck_does_not_exist():
    """`deck_dir` RAISES for an unknown slug — correct for a command, wrong for
    a function whose only job is working out what to hash. It raised before
    `deck_state` could resolve the shorthand, so the dependency calculation
    answered a question about existence that was not its to answer."""
    paths = tools.depends_on(["deck-info", "not-a-deck"])
    assert paths, "an unknown slug should fall back to the decks tree, not raise"


@requires_deck
def test_the_shorthand_is_resolved_before_the_dependencies_are_recorded():
    """Otherwise the answer is keyed on the pilot's wording rather than on the
    deck, and two spellings of one deck cache separately."""
    session = core.Session()
    got = session.call("deck_state", slug="zur")
    assert got["slug"] == "zur-enchantress"
    assert any("zur-enchantress" in p for p in session.touched), (
        f"touched the wrong paths: {session.touched}")


# ── the comparison the model must not do itself ───────────────────────────

@requires_deck
def test_a_win_rate_arrives_with_its_comparisons_already_computed():
    """ASKED WHETHER 25.2% WAS GOOD, THE MODEL SAID "well below functional in a
    four-player pod" — where par IS 25% and this table's measured null is 14.4%.
    It reasoned about rates in prose because the tool handed it a bare number
    and a charter instruction not to.

    Telling a model to be careful with arithmetic is not a control. Handing it
    the arithmetic is.
    """
    sim = tools.deck_state("heliod")["simulation"]
    comparisons = sim.get("comparisons")
    assert comparisons, "a measured win rate arrived with nothing to compare it to"
    for key in ("vs_par", "vs_null"):
        block = comparisons[key]
        assert "diff" in block and "ci95" in block
        assert block["method"].startswith("Newcombe"), (
            "the interval must be ON THE DIFFERENCE, not two marginal intervals")


@requires_deck
def test_each_comparison_carries_a_sentence_rather_than_a_boolean():
    """Handed `excludes_zero: false`, the model wrote "the interval is
    [-10.9%, +10.9%] — it excludes zero". Right conclusion, inverted reason, and
    a reader skimming for "excludes zero" takes away the opposite of the truth.

    `power.preflight` returns lines and `simulation.piloting` carries a
    `reading` for the same reason. This is that pattern applied to the figure
    this bench misreads most.
    """
    comparisons = tools.deck_state("heliod")["simulation"]["comparisons"]
    for key, block in comparisons.items():
        reading = block["reading"]
        spans = not block["excludes_zero"]
        assert ("SPANS ZERO" in reading) is spans, (
            f"{key}: the sentence disagrees with the arithmetic — {reading}")
        assert ("EXCLUDES ZERO" in reading) is (not spans)


def test_an_indistinguishable_result_is_not_called_a_tie():
    """"Not resolved by this sample" is a third state. Reporting it as equality
    is the same error as reporting a null as a finding, which this repo refuses
    in three other places."""
    reading = tools._reading("vs_par", {"diff": 0.0, "ci95": [-0.109, 0.109],
                                        "excludes_zero": False})
    assert "INDISTINGUISHABLE" in reading
    assert "not the same as saying they are equal" in reading


def test_a_deck_with_no_simulation_gets_no_comparison():
    """Absent means absent. A comparison against nothing is not a zero."""
    assert tools._sim_with_comparison(None) is None
    assert tools._sim_with_comparison({"runs": 0}) == {"runs": 0}


# ── what a tool must never leave for the model to fill in ─────────────────

@requires_deck
def test_a_deck_state_names_its_own_commander():
    """Sven could not catch six wrong-commander runs because he was never told
    what the deck's commander IS. A tool that describes a deck without naming it
    leaves nothing to check against."""
    got = tools.deck_state("zur")
    assert got["commander"], "a deck's own commander is not optional context"


def test_a_run_piloted_by_the_wrong_commander_is_flagged_not_left_to_notice(monkeypatch):
    """SIX OF ZUR'S EIGHT RUNS were piloted by `Zur the Enchanter` while the
    deck is built on `Zur, Eternal Schemer`. The played commander was in the run
    record, the declared one was in cards.json, and nothing had ever put them in
    the same view.

    Asserted on a SYNTHETIC record, not on the repo's own state. The first
    version of this test asserted that zur HAD such runs — true when written,
    and it died the moment they were quarantined. A test that depends on a
    defect still existing stops protecting anything the day the defect is fixed,
    and reads as a regression when it fails.
    """
    good = {"seats": [{"commander": ["Zur, Eternal Schemer"]}],
            "run_id": "good-run", "games_completed": 60}
    bad = {"seats": [{"commander": ["Zur the Enchanter"]}],
           "run_id": "bad-run", "games_completed": 60}

    from manamap.sim import forge

    monkeypatch.setattr(forge, "list_runs", lambda _slug: [bad, good])
    got = tools._commander_check("zur-enchantress", {"latest": "good-run"})

    assert got["commander_declared"] == "Zur, Eternal Schemer"
    assert got["commander_forge_played"] == "Zur, Eternal Schemer"
    bad_runs = got["runs_with_the_wrong_commander"]
    assert [r["played"] for r in bad_runs] == ["Zur the Enchanter"]
    assert "never be mixed into a comparison" in got["runs_warning"]


@requires_deck
def test_the_fleet_currently_has_no_wrong_commander_runs():
    """The state this repo should stay in. `test_sim_commander_integrity.py`
    gates it fleet-wide; this is the one-line version for the tool's own view."""
    assert not tools.deck_state("zur")["simulation"].get(
        "runs_with_the_wrong_commander")


@requires_deck
def test_a_clean_deck_is_not_warned_about():
    """A validator that fires on correct data is worse than none."""
    sim = tools.deck_state("heliod")["simulation"]
    assert not sim.get("runs_with_the_wrong_commander")
    assert "WARNING" not in sim


@requires_deck
def test_a_banded_deck_says_it_has_no_single_number():
    """Quoting one end of a band as the figure is the mistake twenty-four zur
    branches were graded on."""
    band = tools.deck_state("zur")["band"]
    assert band and band["rows"], "zur declares an ability and has no band"
    assert "NO SINGLE KILL NUMBER" in band["reading"]
    assert tools.deck_state("heliod")["band"] is None, (
        "a deck that declares nothing must carry no band — absent means absent")


@requires_deck
def test_every_payload_says_what_it_does_not_cover():
    """THE ANTI-INVENTION CONTRACT. Asked what was next for zur, the model wrote
    "you died by turn 5-6" against a recorded median of 34 — because elimination
    timing was in no field it could see, and a gap is where a model narrates
    from nothing.

    Naming the gap, and the command that closes it, turns invention into
    routing.
    """
    absent = tools.deck_state("zur")["not_included"]
    assert absent, "a partial payload that does not say what it omits"
    joined = " ".join(absent.values()).lower()
    assert "elimination" in " ".join(absent).lower()
    assert "do not estimate" in joined
    for pointer in absent.values():
        assert "run_command" in pointer or "read " in pointer, (
            f"a gap named with no way to close it: {pointer}")


# ── a count is never rebuilt from a rate ──────────────────────────────────

def test_a_win_count_is_read_not_reconstructed_from_the_rate():
    """THE BUG THE ENGINE CRITIC CAUGHT, and it was live in shipped code.

    `win_rate` is over DECIDED games; `games` is the TOTAL. heliod's run is 20
    wins in 100 decided out of 120 played — 21 clocked out, and a clock-out has
    no winner, which the same payload states two fields earlier. The old code
    computed `round(win_rate * games)` = 24: a count belonging to neither
    denominator, used in every comparison the tool printed.

    The synthetic below is chosen so the two disagree loudly. If anyone
    reintroduces the reconstruction, `diff` moves and this fails.
    """
    sim = {"games": 120, "decided": 100, "wins": 20, "win_rate": 0.202,
           "vs": ["a", "b", "c"], "latest": "run-1"}
    got = tools._sim_with_comparison(sim)
    par = got["comparisons"]["vs_par"]
    # 20/100 against par 25/100 is -0.05. The reconstruction would have used
    # 24/120 = 0.20 against 30/120, giving the same diff but a NARROWER interval
    # off a bigger n — the tell is the interval, not the point estimate.
    assert par["diff"] == pytest.approx(-0.05, abs=1e-9)
    assert par["ci95"][0] == pytest.approx(-0.1644, abs=1e-3), (
        "the interval was computed off the wrong sample size")


def test_a_run_without_wins_and_decided_offers_no_comparison():
    """Absent means absent. An older `info.json` predates these fields, and a
    comparison computed from a rate alone is a guess wearing the rate's
    authority."""
    sim = {"games": 120, "win_rate": 0.202, "vs": ["a", "b", "c"]}
    got = tools._sim_with_comparison(sim)
    assert "comparisons" not in got
    assert "regenerate" in got["comparisons_unavailable"]


def test_nothing_in_sven_rebuilds_a_count_from_a_rate():
    """The shape, not the instance. `rate * n` reconstructing a numerator is
    what produced the bug, and it is easy to write again in a different
    function."""
    import re
    from pathlib import Path

    from manamap import config

    offenders = []
    for path in sorted((config._REPO_ROOT / "src" / "manamap" / "sven").glob("*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"(win_rate|rate)\s*\*\s*\w*(games|n)\b", line) and \
                    "never" not in line.lower() and not line.strip().startswith("#"):
                offenders.append(f"{path.name}:{n}  {line.strip()}")
    assert not offenders, "a count is being rebuilt from a rate:\n  " + "\n  ".join(offenders)
