"""Agent-invocation cache: fingerprint stability, staleness detection, contract guards."""

import json

import pytest

from conftest import requires_deck
from manamap import config
from manamap.pilot import agent_cache as ac
from manamap.pilot import common

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
    "mana_base": "Twenty-four lands, honestly counted.",
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
    na = {r["routine"]: r["reason"] for r in doc["not_applicable"]}
    assert "brief.json" in na["candidate-pool"]
    # the-ten has no applicability gate — every deck gets a Short List.
    assert "the-ten" not in na


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


# ── Applicability gating (tutor-guide; the-ten has no gate) ──────────────


def test_tutor_guide_is_na_without_tutors(deck):
    """A deck with zero library-search tutors — N/A, not a permanent MISS."""
    with pytest.raises(ac.MissingInput, match="zero library-search tutors"):
        ac.status(SLUG, "tutor-guide")


def test_a_tutor_makes_the_routine_applicable(deck):
    doc = json.loads((deck / "cards.json").read_text())
    doc["cards"].append({"name": "Diabolic Tutor", "type_line": "Sorcery",
                         "oracle_text": "Search your library for a card..."})
    write_json(deck / "cards.json", doc)
    result = ac.status(SLUG, "tutor-guide")
    assert result["status"] == "MISS"  # applicable, just never recorded


def test_fetch_lands_do_not_make_tutor_guide_applicable(deck):
    doc = json.loads((deck / "cards.json").read_text())
    doc["cards"].append({"name": "Evolving Wilds", "type_line": "Land",
                         "oracle_text": "Search your library for a basic land "
                                        "card..."})
    write_json(deck / "cards.json", doc)
    with pytest.raises(ac.MissingInput, match="zero library-search tutors"):
        ac.status(SLUG, "tutor-guide")


def test_the_ten_applies_to_every_deck(deck):
    """Bench or no bench, the Short List routine is live (MISS, never N/A)."""
    assert ac.status(SLUG, "the-ten")["status"] == "MISS"
    doc = json.loads((deck / "cards.json").read_text())
    doc["cards"].append({"name": "Sazacap's Brew", "type_line": "Instant",
                         "is_sideboard": True})
    write_json(deck / "cards.json", doc)
    assert ac.status(SLUG, "the-ten")["status"] == "MISS"


def test_record_refuses_the_inapplicable_tutor_guide(deck):
    write_json(deck / "tutor_guide.json",
               {"slug": SLUG, "assessment": "x", "tutors": [], "gaps": []})
    with pytest.raises(ac.MissingInput, match="zero library-search tutors"):
        ac.record(SLUG, "tutor-guide")


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


# ── Card-scoped invalidation: STALE_OK, refs, rebless ────────────────────


def _two_card_deck(base):
    """A deck where 'Sac Outlet' is referenced by the stack and 'Filler Land'
    is not — the shape every STALE_OK test needs."""
    write_json(base / "cards.json", {"deck": SLUG, "decklist_sha256": "abc", "cards": [
        {"name": "Sac Outlet", "oracle_text": "Sacrifice a creature: draw."},
        {"name": "Filler Land", "oracle_text": "T: Add C.", "is_sideboard": False},
    ]})
    stack = stack_doc("001")
    stack["scenario"]["question"] = "What does Sac Outlet do here?"
    write_json(base / "stacks" / "001-first.json", stack)
    ac._SHA_MEMO.clear()
    common.clear_memo()


def test_diff_card_maps_names_changed_cards():
    old = {"A\x000": "1", "B\x000": "2", "C\x001": "3"}
    new = {"A\x000": "1", "B\x000": "9", "C\x000": "3"}  # B changed, C zone-moved
    assert ac.diff_card_maps(old, new) == ["B", "C"]


def test_record_stores_refs_and_card_map(deck):
    _two_card_deck(deck)
    entry, _ = ac.record(SLUG, "stack:001")
    assert "Sac Outlet" in entry["card_refs"]
    assert "Filler Land" not in entry["card_refs"]
    cache = ac.load_cache(SLUG)
    assert cache["cards_map"]["digest"] == entry["extra"]["cards_semantic"]
    assert len(cache["cards_map"]["cards"]) == 2


def test_unreferenced_card_change_is_stale_ok(deck):
    _two_card_deck(deck)
    ac.record(SLUG, "stack:001")
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"][1]["oracle_text"] = "T: Add one mana of any color."
    write_json(deck / "cards.json", cards)
    ac._SHA_MEMO.clear(); common.clear_memo()
    result = ac.status(SLUG, "stack:001")
    assert result["status"] == "STALE_OK"
    assert any("Filler Land" in c.get("note", "") for c in result["changed"])


def test_referenced_card_change_is_a_real_miss(deck):
    _two_card_deck(deck)
    ac.record(SLUG, "stack:001")
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"][0]["oracle_text"] = "Sacrifice two creatures: draw two."
    write_json(deck / "cards.json", cards)
    ac._SHA_MEMO.clear(); common.clear_memo()
    result = ac.status(SLUG, "stack:001")
    assert result["status"] == "MISS"
    assert any("Sac Outlet" in c.get("note", "") for c in result["changed"])


def test_record_without_refs_keeps_classic_miss(deck):
    """Migration: a pre-refs record must never be silently STALE_OK'd."""
    _two_card_deck(deck)
    ac.record(SLUG, "stack:001")
    cache = ac.load_cache(SLUG)
    cache["routines"]["stack:001"].pop("card_refs", None)
    ac.save_cache(SLUG, cache)
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"][1]["oracle_text"] = "changed"
    write_json(deck / "cards.json", cards)
    ac._SHA_MEMO.clear(); common.clear_memo()
    assert ac.status(SLUG, "stack:001")["status"] == "MISS"


def test_stale_ok_requires_matching_card_map(deck):
    """An older record whose deck state the stored map doesn't describe
    cannot compute its changed set — classic MISS."""
    _two_card_deck(deck)
    ac.record(SLUG, "stack:001")
    cache = ac.load_cache(SLUG)
    cache["cards_map"]["digest"] = "stale-digest"
    ac.save_cache(SLUG, cache)
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"][1]["oracle_text"] = "changed"
    write_json(deck / "cards.json", cards)
    ac._SHA_MEMO.clear(); common.clear_memo()
    assert ac.status(SLUG, "stack:001")["status"] == "MISS"


def test_prompt_change_disqualifies_stale_ok(deck, monkeypatch):
    _two_card_deck(deck)
    ac.record(SLUG, "stack:001")
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"][1]["oracle_text"] = "changed"
    write_json(deck / "cards.json", cards)
    ac._SHA_MEMO.clear(); common.clear_memo()
    monkeypatch.setattr(ac, "agent_prompt_sha256", lambda agent: "edited-prompt")
    assert ac.status(SLUG, "stack:001")["status"] == "MISS"


def test_rebless_clears_stale_ok_and_seeds_refs(deck):
    _two_card_deck(deck)
    ac.record(SLUG, "stack:001")
    # strip refs from one record to exercise the migration path too
    cache = ac.load_cache(SLUG)
    cache["routines"]["stack:001"].pop("card_refs", None)
    ac.save_cache(SLUG, cache)
    reblessed, _ = ac.rebless(SLUG)
    assert "stack:001" in reblessed  # HIT-without-refs is reblessed
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"][1]["oracle_text"] = "changed again"
    write_json(deck / "cards.json", cards)
    ac._SHA_MEMO.clear(); common.clear_memo()
    assert ac.status(SLUG, "stack:001")["status"] == "STALE_OK"
    reblessed, _ = ac.rebless(SLUG)
    assert "stack:001" in reblessed
    assert ac.status(SLUG, "stack:001")["status"] == "HIT"


def test_rebless_clears_EVERY_stale_ok_routine_not_just_the_first(deck):
    """The sweep used to bless exactly one routine per invocation.

    `record()` rewrites the deck-wide `cards_map` baseline, and STALE_OK requires
    that baseline to still match the record's own `extra.cards_semantic`. The old
    loop classified inside the record pass, so the first record moved the baseline
    and every routine after it saw `changed_cards is None` and fell to MISS —
    permanently, because nothing can restore the old baseline. One routine is
    exactly the case the original test covered, which is why it never showed.
    """
    _two_card_deck(deck)
    for sid in ("002", "003", "004"):
        write_json(deck / "stacks" / f"{sid}-more.json", stack_doc(sid))
    ac._SHA_MEMO.clear(); common.clear_memo()

    stacks = ["stack:001", "stack:002", "stack:003", "stack:004"]
    for r in stacks:
        ac.record(SLUG, r)

    # Touch only the card NO stack references.
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"][1]["oracle_text"] = "T: Add one mana of any color."
    write_json(deck / "cards.json", cards)
    ac._SHA_MEMO.clear(); common.clear_memo()

    assert [ac.status(SLUG, r)["status"] for r in stacks] == ["STALE_OK"] * 4

    reblessed, skipped = ac.rebless(SLUG)
    missed = [r for r in stacks if r not in reblessed]
    assert not missed, (
        f"swept {len(reblessed)}, missed {missed} — the baseline advanced mid-sweep. "
        f"skipped={skipped}")
    assert [ac.status(SLUG, r)["status"] for r in stacks] == ["HIT"] * 4

    # `discover_routines` yields the non-stack routines first, and the ones that do
    # not declare `cards:semantic` (candidate-pool, deck-build) are HIT-without-refs,
    # so they are re-recorded for migration BEFORE any stack is reached. Under the old
    # loop each of those advanced the baseline too, which is how a sweep could consume
    # itself on migration work and never reach the artifacts it existed to clear.
    order = ac.discover_routines(SLUG)
    assert order.index("stack:001") > 0, "stacks should not sort first"


def test_keyed_routine_records_refs_by_key(deck):
    _two_card_deck(deck)
    prose = json.loads((deck / "manual_prose.json").read_text())
    prose["mulligan"] = "Keep Sac Outlet hands."
    prose["how_it_wins"] = "No cards named here."
    write_json(deck / "manual_prose.json", prose)
    ac._SHA_MEMO.clear(); common.clear_memo()
    entry, _ = ac.record(SLUG, "writer-prose")
    assert "Sac Outlet" in entry["card_refs_by_key"]["mulligan"]
    assert entry["card_refs_by_key"]["how_it_wins"] == []
    # card_roles keys count as refs
    assert "Sac Outlet" in entry["card_refs_by_key"]["card_roles"]


def test_scan_exit_zero_with_stale_ok(deck):
    _two_card_deck(deck)
    ac.record(SLUG, "stack:001")
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"][1]["oracle_text"] = "changed"
    write_json(deck / "cards.json", cards)
    ac._SHA_MEMO.clear(); common.clear_memo()
    code, out = _run(_Args(SLUG, routine="stack:001"))
    assert code == 0
    assert "STALE_OK" in out


# ── Per-key staleness (scoped regeneration) ──────────────────────────────


def test_record_stores_key_fingerprints(deck):
    _two_card_deck(deck)
    entry, _ = ac.record(SLUG, "writer-prose")
    assert set(entry["key_fingerprints"]) == {
        "how_it_wins", "combo_lines", "card_roles", "mulligan", "upgrades",
        "mana_base"}


def test_goldfish_change_stales_only_goldfish_keys(deck):
    _two_card_deck(deck)
    ac.record(SLUG, "writer-prose")
    write_json(deck / "goldfish_metrics.json", {"meta": {"seed": 42},
                                                "metrics": {"new": True}})
    ac._SHA_MEMO.clear(); common.clear_memo()
    result = ac.status(SLUG, "writer-prose")
    assert result["status"] == "MISS"
    assert set(result["stale_keys"]) == {"how_it_wins", "mulligan",
                                         "mana_base"}


def test_new_passing_stack_stales_stack_keys(deck):
    _two_card_deck(deck)
    ac.record(SLUG, "writer-prose")
    write_json(deck / "stacks" / "003-new.json", stack_doc("003"))
    ac._SHA_MEMO.clear(); common.clear_memo()
    result = ac.status(SLUG, "writer-prose")
    assert result["status"] == "MISS"
    assert set(result["stale_keys"]) == {"combo_lines", "how_it_wins", "upgrades"}


def test_unreferenced_card_change_refines_stale_keys(deck):
    """A card change only stales the keys whose refs include a changed card;
    with no key referencing it, the whole routine is STALE_OK instead."""
    _two_card_deck(deck)
    prose = json.loads((deck / "manual_prose.json").read_text())
    prose["mulligan"] = "Keep Sac Outlet hands."
    prose["card_roles"] = {"Sac Outlet": "the engine."}
    write_json(deck / "manual_prose.json", prose)
    ac._SHA_MEMO.clear(); common.clear_memo()
    ac.record(SLUG, "writer-prose")
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"][1]["oracle_text"] = "changed filler"
    write_json(deck / "cards.json", cards)
    ac._SHA_MEMO.clear(); common.clear_memo()
    result = ac.status(SLUG, "writer-prose")
    assert result["status"] == "STALE_OK"


def test_referenced_card_change_names_the_stale_keys(deck):
    _two_card_deck(deck)
    prose = json.loads((deck / "manual_prose.json").read_text())
    prose["mulligan"] = "Keep Sac Outlet hands."
    prose["how_it_wins"] = "No names here."
    prose["card_roles"] = {"Sac Outlet": "the engine."}
    write_json(deck / "manual_prose.json", prose)
    ac._SHA_MEMO.clear(); common.clear_memo()
    ac.record(SLUG, "writer-prose")
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"][0]["oracle_text"] = "Sacrifice everything."
    write_json(deck / "cards.json", cards)
    ac._SHA_MEMO.clear(); common.clear_memo()
    result = ac.status(SLUG, "writer-prose")
    assert result["status"] == "MISS"
    assert "mulligan" in result["stale_keys"]
    assert "card_roles" in result["stale_keys"]
    assert "how_it_wins" not in result["stale_keys"]


def test_card_refs_version_is_stamped_and_reseeds_without_invalidating(deck, monkeypatch):
    """Bumping CARD_REFS_VERSION must re-seed refs and invalidate nothing.

    Refs ride outside the fingerprint, so a change to the EXTRACTION rules has to
    be able to reach already-recorded routines without costing a spawn. Before
    this marker, `rebless` skipped any HIT that merely had refs — however stale
    their derivation — so an extractor fix was inert on all 106 live records.
    """
    _two_card_deck(deck)
    entry, _ = ac.record(SLUG, "stack:001")
    assert entry["card_refs_version"] == ac.CARD_REFS_VERSION

    before = ac.status(SLUG, "stack:001")
    monkeypatch.setattr(ac, "CARD_REFS_VERSION", ac.CARD_REFS_VERSION + 1)
    after = ac.status(SLUG, "stack:001")

    # The bump changes NOTHING about the fingerprint or the verdict.
    assert after["status"] == before["status"] == "HIT"
    assert after["fingerprint"] == before["fingerprint"]

    # But the sweep now re-seeds it.
    reblessed, _ = ac.rebless(SLUG)
    assert "stack:001" in reblessed
    assert (ac.load_cache(SLUG)["routines"]["stack:001"]["card_refs_version"]
            == ac.CARD_REFS_VERSION)
    # ...and it settles: a second sweep has nothing to do.
    assert "stack:001" not in ac.rebless(SLUG)[0]
