"""Agent-invocation cache: fingerprint stability, staleness detection, contract guards."""

import json

import pytest

from conftest import requires_deck
from manamap import config
from manamap.pilot import agent_cache as ac

SLUG = "test-deck"


# ── Fixtures ─────────────────────────────────────────────────────────────


def write_json(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def stack_doc(sid="001", verdict="pass"):
    return {
        "id": sid, "title": f"Line {sid}",
        "scenario": {"question": "What happens?", "stack": [{"pos": 0, "object": "X"}]},
        "resolution": {"steps": [{"n": 1, "action": "A", "effect": "B",
                                  "citations": [{"rule": "702.40a", "quote": "q"}]}],
                       "final_state": {"summary": "done"}},
        "checker": {"verdict": verdict, "iterations": 1, "findings": []},
    }


PROSE = {
    "cover": {"tagline": "A tagline", "identity": "An identity"},
    "how_it_wins": "Bodies first.",
    "combo_lines": {"001": "Intro."},
    "card_roles": {"Sac Outlet": "A role."},
    "mulligan": "Keep bodies.",
    "upgrades": "Swap these.",
    "threat_assessment": "They turn here.",
    "matchups": "Against sweepers.",
}


@pytest.fixture
def deck(tmp_path, monkeypatch):
    """An isolated deck dir wired into the module's path resolution."""
    decks = tmp_path / "decks"
    base = decks / SLUG
    base.mkdir(parents=True)
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", decks)

    write_json(base / "cards.json", {"deck": SLUG, "decklist_sha256": "abc",
                                     "cards": [{"name": "Sac Outlet"}]})
    write_json(base / "goldfish_metrics.json", {"meta": {"seed": 42}, "metrics": {}})
    write_json(base / "strategic_frame.json", {"slug": SLUG, "angle": "An angle."})
    write_json(base / "manual_prose.json", PROSE)
    write_json(base / "issue.json", {"volume": 1, "deck_name": "TEST"})
    write_json(base / "issue_plan.json", {"slug": SLUG, "angle": "x", "departments": []})
    write_json(base / "stacks" / "001-first.json", stack_doc("001"))
    write_json(base / "stacks" / "002-failed.json", stack_doc("002", verdict="fail"))
    write_json(base / "decisions" / "001-a-call.json",
               {"id": "001", "kind": "decision", "title": "A call",
                "scenario": {"question": "?", "board": {}}, "branches": [],
                "recommendation": {}})
    ac._SHA_MEMO.clear()
    return base


def fp(slug, routine):
    spec = ac.routine_spec(slug, routine)
    entries, extra = ac.resolve_inputs(slug, spec)
    return ac.fingerprint(routine, spec, entries, extra)


# ── Fingerprint ──────────────────────────────────────────────────────────


def test_fingerprint_stable_across_calls(deck):
    assert fp(SLUG, "coach-prose") == fp(SLUG, "coach-prose")


def test_fingerprint_order_independent(deck):
    spec = ac.routine_spec(SLUG, "coach-prose")
    entries, extra = ac.resolve_inputs(SLUG, spec)
    a = ac.fingerprint("coach-prose", spec, entries, extra)
    b = ac.fingerprint("coach-prose", spec, list(reversed(entries)), extra)
    assert a == b


def test_fingerprint_changes_on_content_change(deck):
    before = fp(SLUG, "coach-prose")
    write_json(deck / "cards.json", {"deck": SLUG, "cards": [{"name": "Other"}]})
    ac._SHA_MEMO.clear()
    assert fp(SLUG, "coach-prose") != before


def test_routine_id_prevents_collision(deck):
    """Two routines with overlapping inputs must not share a fingerprint."""
    assert fp(SLUG, "coach-prose") != fp(SLUG, "strategic-frame")


def test_fingerprint_changes_on_cache_version_bump(deck, monkeypatch):
    before = fp(SLUG, "coach-prose")
    monkeypatch.setattr(ac, "AGENT_CACHE_VERSION", 999)
    assert fp(SLUG, "coach-prose") != before


def test_fingerprint_changes_on_agent_prompt_edit(deck, monkeypatch, tmp_path):
    prompts = tmp_path / "agents"
    prompts.mkdir()
    (prompts / "pilot-coach.md").write_text("v1")
    monkeypatch.setattr(ac, "AGENT_PROMPTS_DIR", prompts)
    before = fp(SLUG, "coach-prose")
    (prompts / "pilot-coach.md").write_text("v2")
    assert fp(SLUG, "coach-prose") != before


def test_agent_prompt_digest_covers_every_part_of_a_loop(tmp_path, monkeypatch):
    prompts = tmp_path / "agents"
    prompts.mkdir()
    (prompts / "stack-resolver.md").write_text("a")
    (prompts / "rules-checker.md").write_text("b")
    monkeypatch.setattr(ac, "AGENT_PROMPTS_DIR", prompts)
    before = ac.agent_prompt_sha256("stack-resolver+rules-checker")
    (prompts / "rules-checker.md").write_text("b2")
    assert ac.agent_prompt_sha256("stack-resolver+rules-checker") != before


# ── Input resolution ─────────────────────────────────────────────────────


def test_missing_optional_input_recorded_as_null(deck):
    (deck / "strategic_frame.json").unlink()
    spec = ac.routine_spec(SLUG, "coach-prose")
    entries, _ = ac.resolve_inputs(SLUG, spec)
    frame = [e for e in entries if e["path"].endswith("strategic_frame.json")]
    assert len(frame) == 1 and frame[0]["sha256"] is None


def test_optional_input_appearing_changes_fingerprint(deck):
    (deck / "strategic_frame.json").unlink()
    ac._SHA_MEMO.clear()
    without = fp(SLUG, "coach-prose")
    write_json(deck / "strategic_frame.json", {"slug": SLUG, "angle": "back"})
    ac._SHA_MEMO.clear()
    assert fp(SLUG, "coach-prose") != without


def test_missing_required_input_raises(deck):
    (deck / "cards.json").unlink()
    with pytest.raises(ac.MissingInput):
        fp(SLUG, "coach-prose")


def test_only_passing_stacks_are_inputs(deck):
    spec = ac.routine_spec(SLUG, "coach-prose")
    entries, _ = ac.resolve_inputs(SLUG, spec)
    paths = [e["path"] for e in entries]
    assert any("001-first.json" in p for p in paths)
    assert not any("002-failed.json" in p for p in paths)


def test_failing_stack_edit_does_not_invalidate_downstream(deck):
    before = fp(SLUG, "coach-prose")
    doc = stack_doc("002", verdict="fail")
    doc["resolution"]["final_state"]["summary"] = "rewritten"
    write_json(deck / "stacks" / "002-failed.json", doc)
    ac._SHA_MEMO.clear()
    assert fp(SLUG, "coach-prose") == before


def test_stack_flipping_to_pass_invalidates_downstream(deck):
    before = fp(SLUG, "coach-prose")
    write_json(deck / "stacks" / "002-failed.json", stack_doc("002", verdict="pass"))
    ac._SHA_MEMO.clear()
    assert fp(SLUG, "coach-prose") != before


def test_scenario_digest_ignores_resolution_and_checker(deck):
    path = deck / "stacks" / "001-first.json"
    before = ac.scenario_block_digest(path)
    doc = stack_doc("001")
    doc["resolution"]["final_state"]["summary"] = "totally different"
    doc["checker"]["iterations"] = 3
    write_json(path, doc)
    assert ac.scenario_block_digest(path) == before


def test_stack_routine_does_not_self_invalidate(deck):
    """The loop writes resolution+checker into its own artifact."""
    before = fp(SLUG, "stack:001")
    doc = stack_doc("001")
    doc["resolution"]["steps"].append({"n": 2, "action": "C", "effect": "D",
                                       "citations": [{"rule": "117.5", "quote": "q"}]})
    write_json(deck / "stacks" / "001-first.json", doc)
    ac._SHA_MEMO.clear()
    assert fp(SLUG, "stack:001") == before


def test_unknown_routine_raises(deck):
    with pytest.raises(ac.UnknownRoutine):
        ac.routine_spec(SLUG, "not-a-routine")


def test_discover_routines_finds_dynamic_ones(deck):
    routines = ac.discover_routines(SLUG)
    assert "issue-plan" in routines
    assert "stack:001" in routines and "stack:002" in routines
    assert "decision:001" in routines


# ── Prose shape ──────────────────────────────────────────────────────────


def test_prose_shape_ignores_wording(deck):
    before = ac.prose_shape(deck / "manual_prose.json")
    reworded = dict(PROSE, how_it_wins="COMPLETELY different wording here.")
    write_json(deck / "manual_prose.json", reworded)
    assert ac.prose_shape(deck / "manual_prose.json") == before


def test_prose_reword_does_not_invalidate_issue_plan(deck):
    """The headline win: a typo fix must not cost a re-plan."""
    before = fp(SLUG, "issue-plan")
    write_json(deck / "manual_prose.json", dict(PROSE, mulligan="Reworded entirely."))
    ac._SHA_MEMO.clear()
    assert fp(SLUG, "issue-plan") == before


def test_new_section_does_invalidate_issue_plan(deck):
    before = fp(SLUG, "issue-plan")
    grown = dict(PROSE, combo_lines={"001": "Intro.", "002": "New line."})
    write_json(deck / "manual_prose.json", grown)
    ac._SHA_MEMO.clear()
    assert fp(SLUG, "issue-plan") != before


# ── Artifact key isolation ───────────────────────────────────────────────


def test_coach_and_writer_keys_are_independent(deck):
    ac.record(SLUG, "coach-prose")
    ac.record(SLUG, "writer-prose")
    rewritten = dict(PROSE, how_it_wins="Writer rewrote this.")
    write_json(deck / "manual_prose.json", rewritten)
    ac._SHA_MEMO.clear()
    assert ac.status(SLUG, "coach-prose")["status"] == "HIT"
    assert ac.status(SLUG, "writer-prose")["status"] == "EDITED"


# ── Status ───────────────────────────────────────────────────────────────


def test_status_miss_when_no_record(deck):
    assert ac.status(SLUG, "coach-prose")["status"] == "MISS"


def test_status_hit_after_record(deck):
    ac.record(SLUG, "coach-prose")
    assert ac.status(SLUG, "coach-prose")["status"] == "HIT"


def test_status_miss_names_the_changed_input(deck):
    ac.record(SLUG, "coach-prose")
    write_json(deck / "strategic_frame.json", {"slug": SLUG, "angle": "different"})
    ac._SHA_MEMO.clear()
    result = ac.status(SLUG, "coach-prose")
    assert result["status"] == "MISS"
    changed = [c for c in result["changed"] if c["path"].endswith("strategic_frame.json")]
    assert changed and changed[0]["change"] == "modified"


def test_status_reports_added_stack_as_now_passing(deck):
    ac.record(SLUG, "coach-prose")
    write_json(deck / "stacks" / "003-new.json", stack_doc("003"))
    ac._SHA_MEMO.clear()
    result = ac.status(SLUG, "coach-prose")
    added = [c for c in result["changed"] if c["change"] == "added"]
    assert added and added[0]["note"] == "now passing"


def test_status_edited_when_artifact_hand_edited(deck):
    ac.record(SLUG, "issue-plan")
    write_json(deck / "issue_plan.json",
               {"slug": SLUG, "angle": "hand-tuned headline", "departments": []})
    ac._SHA_MEMO.clear()
    assert ac.status(SLUG, "issue-plan")["status"] == "EDITED"


def test_status_miss_when_artifact_deleted(deck):
    ac.record(SLUG, "issue-plan")
    (deck / "issue_plan.json").unlink()
    assert ac.status(SLUG, "issue-plan")["status"] == "MISS"


def test_force_always_misses(deck):
    ac.record(SLUG, "coach-prose")
    result = ac.status(SLUG, "coach-prose", force=True)
    assert result["status"] == "MISS" and result["reason"] == "forced"


# ── Record guards ────────────────────────────────────────────────────────


def test_record_refuses_missing_artifact(deck):
    (deck / "issue_plan.json").unlink()
    with pytest.raises(ac.MissingInput):
        ac.record(SLUG, "issue-plan")


def test_record_refuses_stack_without_checker(deck):
    doc = stack_doc("001")
    del doc["checker"]
    write_json(deck / "stacks" / "001-first.json", doc)
    with pytest.raises(ac.MissingInput):
        ac.record(SLUG, "stack:001")


def test_record_stores_verdict_and_iterations(deck):
    entry, _ = ac.record(SLUG, "stack:001")
    assert entry["verdict"] == "pass" and entry["iterations"] == 1


def test_record_refuses_when_owned_keys_absent(deck):
    write_json(deck / "manual_prose.json", {"how_it_wins": "only writer keys"})
    with pytest.raises(ac.MissingInput):
        ac.record(SLUG, "coach-prose")


def test_record_refuses_a_partial_artifact(deck):
    """ALL the routine's keys, not any of them.

    The guard used to be `any`, so an agent that produced one of six declared
    keys recorded cleanly and became a permanent HIT on a manual that was five
    sections short. artifact_digest hashes absent keys as None, so nothing
    downstream would have noticed.
    """
    keys = config.AGENT_ROUTINES["writer-prose"]["artifact_keys"]
    partial = {keys[0]: "the writer stopped after one section"}
    write_json(deck / "manual_prose.json", partial)
    with pytest.raises(ac.MissingInput) as excinfo:
        ac.record(SLUG, "writer-prose")
    message = str(excinfo.value)
    # The message must name what is missing, so the fix is obvious without a re-read.
    for absent in keys[1:]:
        assert absent in message


def test_record_accepts_a_complete_artifact(deck):
    keys = config.AGENT_ROUTINES["writer-prose"]["artifact_keys"]
    write_json(deck / "manual_prose.json", {k: f"{k} prose" for k in keys})
    entry, _ = ac.record(SLUG, "writer-prose")
    assert entry["artifact"].endswith("manual_prose.json")


def test_writer_prose_does_not_own_a_cover_key(deck):
    """issue_plan.json owns the cover; build_manual never reads prose['cover'].

    The key was declared, produced on every deck, cached — and rendered nowhere.
    """
    assert "cover" not in config.AGENT_ROUTINES["writer-prose"]["artifact_keys"]


# ── Sidecar ──────────────────────────────────────────────────────────────


def test_save_cache_skips_identical_write(deck):
    ac.record(SLUG, "coach-prose")
    _, wrote_again = ac.record(SLUG, "coach-prose")
    assert wrote_again is False


def test_sidecar_has_no_timestamps(deck):
    """No generated dates anywhere — every diff line must mean something.

    Checks keys rather than raw text: paths legitimately contain arbitrary
    words (pytest's own tmp dir is named after this test).
    """
    ac.record(SLUG, "coach-prose")
    cache = json.loads(ac.cache_path(SLUG).read_text())

    def keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys(v)
        elif isinstance(obj, list):
            for item in obj:
                yield from keys(item)

    for key in keys(cache):
        lowered = key.lower()
        assert not lowered.endswith("_at"), key
        assert "timestamp" not in lowered and "date" not in lowered, key


def test_clear_drops_records(deck):
    ac.record(SLUG, "coach-prose")
    ac.record(SLUG, "writer-prose")
    dropped = ac.clear(SLUG, "coach-prose")
    assert dropped == ["coach-prose"]
    assert ac.status(SLUG, "coach-prose")["status"] == "MISS"
    assert ac.status(SLUG, "writer-prose")["status"] == "HIT"


# ── The strategy-DB hazard ───────────────────────────────────────────────


def test_strategy_db_rebuild_does_not_invalidate(deck, monkeypatch):
    """Rebuilding the derived index must not cost 330k tokens."""
    calls = {"n": 0}

    def fake_digest():
        calls["n"] += 1
        return "constant-doc-hash"

    monkeypatch.setattr(ac, "strategy_doc_digest", fake_digest)
    before = fp(SLUG, "coach-prose")
    # A rebuild rewrites strategy_index.json / .strategy-db-meta.json, never the doc.
    assert fp(SLUG, "coach-prose") == before
    assert calls["n"] == 2


# ── Data-gated smoke ─────────────────────────────────────────────────────


@requires_deck
def test_real_deck_routines_resolve():
    """Every routine whose inputs a hand-built deck has must resolve cleanly.

    The construction routines (candidate-pool, deck-build) require an authored
    brief.json, which a hand-built deck like goblin-storm has no reason to
    carry. MissingInput there is the designed behaviour — it becomes exit 2,
    "stop, don't spawn" — not a failure to resolve.
    """
    resolved = 0
    for routine in ac.discover_routines("goblin-storm"):
        try:
            result = ac.status("goblin-storm", routine)
        except ac.MissingInput:
            continue
        assert result["status"] in {"HIT", "MISS", "EDITED"}
        resolved += 1
    assert resolved, "no routine resolved for goblin-storm at all"


def test_build_routines_require_a_brief():
    """A deck with no brief cannot be built — that must be exit 2, not a MISS."""
    with pytest.raises(ac.MissingInput, match="brief.json"):
        ac.status("goblin-storm", "candidate-pool")


# ── the all-routines scan must survive routines that don't apply ──


class _Args:
    def __init__(self, slug, routine=None, as_json=False, force=False):
        self.slug = slug
        self.pilot_command = "cache-status"
        self.routine = routine
        self.as_json = as_json
        self.force = force


def _run(args):
    """Run main(args), returning (exit_code, stdout)."""
    import contextlib
    import io
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ac.main(args)
    except SystemExit as e:
        return (e.code if isinstance(e.code, int) else 1), buf.getvalue()
    return 0, buf.getvalue()


@requires_deck
def test_scan_reports_every_applicable_routine_despite_an_inapplicable_one():
    """A hand-built deck has no brief.json and never will.

    Before this, one inapplicable routine aborted the whole scan with exit 2,
    which broke a command that had worked for months the moment the build
    routines were registered.
    """
    code, out = _run(_Args("goblin-storm"))
    assert code in (0, 1), f"scan should not exit 2, got {code}"
    assert "N/A" in out and "candidate-pool" in out
    # ...and the routines that DO apply are still reported
    assert "writer-prose" in out
    assert "stack:001" in out


@requires_deck
def test_scan_exit_code_ignores_inapplicable_routines():
    """N/A is neither a hit nor a miss — it must not force a spawn signal.

    Asserted as an invariant rather than against a fixed deck: the exit code
    must track *applicable* misses only, whatever state the decks are in.
    """
    code, out = _run(_Args("goblin-storm", as_json=True))
    doc = json.loads(out)
    assert doc["not_applicable"], "expected at least one N/A routine on a hand-built deck"
    expected = 1 if any(r["status"] == "MISS" for r in doc["routines"]) else 0
    assert code == expected
    assert code != 2


@requires_deck
def test_explicit_routine_with_a_missing_input_still_exits_2():
    """Asked directly about a routine, a missing input is still 'stop, fix it'."""
    code, _ = _run(_Args("goblin-storm", routine="candidate-pool"))
    assert code == 2


@requires_deck
def test_scan_json_separates_applicable_from_not():
    code, out = _run(_Args("goblin-storm", as_json=True))
    doc = json.loads(out)
    assert {"slug", "any_miss", "routines", "not_applicable"} <= set(doc)
    assert any(r["routine"] == "candidate-pool" for r in doc["not_applicable"])
    assert all("brief.json" in r["reason"] for r in doc["not_applicable"])


# ── The iteration bound, enforced rather than quoted ─────────────────────


def test_record_refuses_iterations_over_the_bound_without_a_reason(deck):
    """RESOLVE_MAX_ITERATIONS used to be enforced by a model reading markdown.

    No Python imported it, so hapatra's stack 001 ran to 4 and recorded cleanly.
    Overriding is still allowed — it just has to be declared.
    """
    doc = stack_doc("001")
    doc["checker"]["iterations"] = config.RESOLVE_MAX_ITERATIONS + 1
    write_json(deck / "stacks" / "001-first.json", doc)
    with pytest.raises(ac.MissingInput) as excinfo:
        ac.record(SLUG, "stack:001")
    assert "iteration_bound_override" in str(excinfo.value)


def test_record_accepts_an_over_bound_run_with_a_stated_reason(deck):
    doc = stack_doc("001")
    doc["checker"]["iterations"] = config.RESOLVE_MAX_ITERATIONS + 1
    doc["checker"]["iteration_bound_override"] = {"reason": "checker confirmed convergence"}
    write_json(deck / "stacks" / "001-first.json", doc)
    entry, _ = ac.record(SLUG, "stack:001")
    assert entry["iteration_bound_override"]["reason"]


def test_override_may_be_a_bare_string(deck):
    """The key was invented ad hoc as free text; accept the shape that exists."""
    doc = stack_doc("001")
    doc["checker"]["iterations"] = config.RESOLVE_MAX_ITERATIONS + 1
    doc["checker"]["iteration_bound_override"] = "deliberate fourth pass, operator approved"
    write_json(deck / "stacks" / "001-first.json", doc)
    entry, _ = ac.record(SLUG, "stack:001")
    assert "fourth pass" in entry["iteration_bound_override"]["reason"]


def test_an_empty_override_is_not_a_justification(deck):
    doc = stack_doc("001")
    doc["checker"]["iterations"] = config.RESOLVE_MAX_ITERATIONS + 1
    doc["checker"]["iteration_bound_override"] = {"reason": "   "}
    write_json(deck / "stacks" / "001-first.json", doc)
    with pytest.raises(ac.MissingInput):
        ac.record(SLUG, "stack:001")


def test_runs_within_the_bound_need_no_override(deck):
    entry, _ = ac.record(SLUG, "stack:001")
    assert "iteration_bound_override" not in entry


# ── Sideboard N/A gating ─────────────────────────────────────────────────


def sideboard_analysis_doc():
    return {"slug": SLUG, "assessment": "x", "swaps": [], "opens_lines": [],
            "long_term_defaults": [], "gaps": []}


def test_sideboard_routine_is_na_without_a_real_sideboard(deck):
    """A deck with no sideboard can never satisfy the routine — N/A, not MISS."""
    with pytest.raises(ac.MissingInput, match="no analysable sideboard"):
        ac.status(SLUG, "sideboard-analysis")


def test_accessories_alone_do_not_make_the_routine_applicable(deck):
    doc = json.loads((deck / "cards.json").read_text())
    doc["cards"].append({"name": "Storm Counter", "type_line": "Card",
                         "is_sideboard": True})
    write_json(deck / "cards.json", doc)
    with pytest.raises(ac.MissingInput, match="no analysable sideboard"):
        ac.status(SLUG, "sideboard-analysis")


def test_a_real_sideboard_card_makes_the_routine_applicable(deck):
    doc = json.loads((deck / "cards.json").read_text())
    doc["cards"].append({"name": "Sazacap's Brew", "type_line": "Instant",
                         "is_sideboard": True})
    write_json(deck / "cards.json", doc)
    result = ac.status(SLUG, "sideboard-analysis")
    assert result["status"] == "MISS"  # applicable, just never recorded


def test_record_refuses_the_inapplicable_sideboard_routine(deck):
    write_json(deck / "sideboard_analysis.json", sideboard_analysis_doc())
    with pytest.raises(ac.MissingInput, match="no analysable sideboard"):
        ac.record(SLUG, "sideboard-analysis")


# ── Memoized artifact loading (common.load_json_memo) ────────────────────


def test_load_json_memo_returns_one_parse(tmp_path):
    from manamap.pilot import common
    path = tmp_path / "artifact.json"
    path.write_text('{"a": 1}')
    first = common.load_json_memo(path)
    assert common.load_json_memo(path) is first  # same object, not a re-parse


def test_load_json_memo_sees_a_rewrite(tmp_path):
    from manamap.pilot import common
    path = tmp_path / "artifact.json"
    path.write_text('{"a": 1}')
    assert common.load_json_memo(path) == {"a": 1}
    path.write_text('{"a": 2, "b": 3}')
    assert common.load_json_memo(path) == {"a": 2, "b": 3}


def test_clear_memo_drops_cached_parses(tmp_path):
    from manamap.pilot import common
    path = tmp_path / "artifact.json"
    path.write_text('{"a": 1}')
    first = common.load_json_memo(path)
    common.clear_memo()
    assert common.load_json_memo(path) is not first
