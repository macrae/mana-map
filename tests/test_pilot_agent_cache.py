"""Agent-invocation cache: fingerprint stability, staleness detection, contract guards."""

import json

import pytest

from conftest import requires_deck
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
