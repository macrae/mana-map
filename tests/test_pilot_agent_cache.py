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
    write_json(base / "deck_recon.json", {"slug": SLUG, "as_of": "2026-01-01", "findings": []})
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
    assert fp(SLUG, "pilot-notes") == fp(SLUG, "pilot-notes")


def test_fingerprint_order_independent(deck):
    spec = ac.routine_spec(SLUG, "pilot-notes")
    entries, extra = ac.resolve_inputs(SLUG, spec)
    a = ac.fingerprint("pilot-notes", spec, entries, extra)
    b = ac.fingerprint("pilot-notes", spec, list(reversed(entries)), extra)
    assert a == b


def test_fingerprint_changes_on_content_change(deck):
    before = fp(SLUG, "pilot-notes")
    write_json(deck / "cards.json", {"deck": SLUG, "cards": [{"name": "Other"}]})
    ac._SHA_MEMO.clear()
    assert fp(SLUG, "pilot-notes") != before


def test_routine_id_prevents_collision(deck):
    """Two routines with overlapping inputs must not share a fingerprint."""
    assert fp(SLUG, "pilot-notes") != fp(SLUG, "strategic-frame")


def test_fingerprint_changes_on_cache_version_bump(deck, monkeypatch):
    before = fp(SLUG, "pilot-notes")
    monkeypatch.setattr(ac, "AGENT_CACHE_VERSION", 999)
    assert fp(SLUG, "pilot-notes") != before


def test_fingerprint_changes_on_agent_prompt_edit(deck, monkeypatch, tmp_path):
    prompts = tmp_path / "agents"
    prompts.mkdir()
    (prompts / "pilot-notes.md").write_text("v1")
    monkeypatch.setattr(ac, "AGENT_PROMPTS_DIR", prompts)
    before = fp(SLUG, "pilot-notes")
    (prompts / "pilot-notes.md").write_text("v2")
    assert fp(SLUG, "pilot-notes") != before


def test_agent_prompt_digest_covers_every_part_of_a_loop(tmp_path, monkeypatch):
    prompts = tmp_path / "agents"
    prompts.mkdir()
    (prompts / "stack-resolver.md").write_text("a")
    (prompts / "rules-checker.md").write_text("b")
    monkeypatch.setattr(ac, "AGENT_PROMPTS_DIR", prompts)
    before = ac.agent_prompt_sha256("stack-resolver+rules-checker")
    (prompts / "rules-checker.md").write_text("b2")
    assert ac.agent_prompt_sha256("stack-resolver+rules-checker") != before


def test_agent_prompt_digest_covers_the_shared_contract(tmp_path, monkeypatch):
    """`.claude/agents-common.md` is read by every charter, so editing it must
    invalidate every routine. Hashed inside agent_prompt_sha256 rather than
    listed per routine so a new routine cannot forget it — a missed edge serves
    a stale pass rather than failing."""
    prompts = tmp_path / "agents"
    prompts.mkdir()
    (prompts / "pilot-notes.md").write_text("same")
    common = tmp_path / "agents-common.md"
    common.write_text("v1")
    monkeypatch.setattr(ac, "AGENT_PROMPTS_DIR", prompts)
    monkeypatch.setattr(ac, "AGENT_COMMON_PROMPT", common)
    before = ac.agent_prompt_sha256("pilot-notes")
    common.write_text("v2")
    assert ac.agent_prompt_sha256("pilot-notes") != before


# ── Input resolution ─────────────────────────────────────────────────────


def test_missing_optional_input_recorded_as_null(deck):
    (deck / "strategic_frame.json").unlink()
    spec = ac.routine_spec(SLUG, "pilot-notes")
    entries, _ = ac.resolve_inputs(SLUG, spec)
    frame = [e for e in entries if e["path"].endswith("strategic_frame.json")]
    assert len(frame) == 1 and frame[0]["sha256"] is None


def test_optional_input_appearing_changes_fingerprint(deck):
    (deck / "strategic_frame.json").unlink()
    ac._SHA_MEMO.clear()
    without = fp(SLUG, "pilot-notes")
    write_json(deck / "strategic_frame.json", {"slug": SLUG, "angle": "back"})
    ac._SHA_MEMO.clear()
    assert fp(SLUG, "pilot-notes") != without


def test_missing_required_input_raises(deck):
    (deck / "cards.json").unlink()
    with pytest.raises(ac.MissingInput):
        fp(SLUG, "pilot-notes")


def test_only_passing_stacks_are_inputs(deck):
    spec = ac.routine_spec(SLUG, "pilot-notes")
    entries, _ = ac.resolve_inputs(SLUG, spec)
    paths = [e["path"] for e in entries]
    assert any("001-first.json" in p for p in paths)
    assert not any("002-failed.json" in p for p in paths)


def test_failing_stack_edit_does_not_invalidate_downstream(deck):
    before = fp(SLUG, "pilot-notes")
    doc = stack_doc("002", verdict="fail")
    doc["resolution"]["final_state"]["summary"] = "rewritten"
    write_json(deck / "stacks" / "002-failed.json", doc)
    ac._SHA_MEMO.clear()
    assert fp(SLUG, "pilot-notes") == before


def test_stack_flipping_to_pass_invalidates_downstream(deck):
    before = fp(SLUG, "pilot-notes")
    write_json(deck / "stacks" / "002-failed.json", stack_doc("002", verdict="pass"))
    ac._SHA_MEMO.clear()
    assert fp(SLUG, "pilot-notes") != before


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
    assert "deck-recon" in routines
    assert "stack:001" in routines and "stack:002" in routines
    assert "decision:001" in routines


# ── Artifact key isolation ───────────────────────────────────────────────


def test_a_legacy_key_edit_is_not_an_edit_to_the_notes(deck):
    """The published decks carry card_roles/mana_base/upgrades as frozen copy no
    routine owns. Touching one must not read as a hand edit to pilot-notes —
    the routine digests only its five keys, so the legacy keys are invisible."""
    ac.record(SLUG, "pilot-notes")
    write_json(deck / "manual_prose.json", dict(PROSE, mana_base="Legacy, retouched."))
    ac._SHA_MEMO.clear()
    assert ac.status(SLUG, "pilot-notes")["status"] == "HIT"
    write_json(deck / "manual_prose.json", dict(PROSE, how_it_wins="Hand-tuned."))
    ac._SHA_MEMO.clear()
    assert ac.status(SLUG, "pilot-notes")["status"] == "EDITED"


# ── Status ───────────────────────────────────────────────────────────────


def test_status_miss_when_no_record(deck):
    assert ac.status(SLUG, "pilot-notes")["status"] == "MISS"


def test_status_hit_after_record(deck):
    ac.record(SLUG, "pilot-notes")
    assert ac.status(SLUG, "pilot-notes")["status"] == "HIT"


def test_status_miss_names_the_changed_input(deck):
    ac.record(SLUG, "pilot-notes")
    write_json(deck / "strategic_frame.json", {"slug": SLUG, "angle": "different"})
    ac._SHA_MEMO.clear()
    result = ac.status(SLUG, "pilot-notes")
    assert result["status"] == "MISS"
    changed = [c for c in result["changed"] if c["path"].endswith("strategic_frame.json")]
    assert changed and changed[0]["change"] == "modified"


def test_status_reports_added_stack_as_now_passing(deck):
    ac.record(SLUG, "pilot-notes")
    write_json(deck / "stacks" / "003-new.json", stack_doc("003"))
    ac._SHA_MEMO.clear()
    result = ac.status(SLUG, "pilot-notes")
    added = [c for c in result["changed"] if c["change"] == "added"]
    assert added and added[0]["note"] == "now passing"


def test_status_edited_when_artifact_hand_edited(deck):
    ac.record(SLUG, "deck-recon")
    write_json(deck / "deck_recon.json",
               {"slug": SLUG, "as_of": "2026-01-01", "findings": ["hand-added"]})
    ac._SHA_MEMO.clear()
    assert ac.status(SLUG, "deck-recon")["status"] == "EDITED"


def test_status_miss_when_artifact_deleted(deck):
    ac.record(SLUG, "deck-recon")
    (deck / "deck_recon.json").unlink()
    assert ac.status(SLUG, "deck-recon")["status"] == "MISS"


def test_force_always_misses(deck):
    ac.record(SLUG, "pilot-notes")
    result = ac.status(SLUG, "pilot-notes", force=True)
    assert result["status"] == "MISS" and result["reason"] == "forced"


# ── Record guards ────────────────────────────────────────────────────────


def test_record_refuses_missing_artifact(deck):
    (deck / "deck_recon.json").unlink()
    with pytest.raises(ac.MissingInput):
        ac.record(SLUG, "deck-recon")


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
        ac.record(SLUG, "pilot-notes")


def test_record_refuses_a_partial_artifact(deck):
    """ALL the routine's keys, not any of them.

    The guard used to be `any`, so an agent that produced one of six declared
    keys recorded cleanly and became a permanent HIT on a manual that was five
    sections short. artifact_digest hashes absent keys as None, so nothing
    downstream would have noticed.
    """
    keys = config.AGENT_ROUTINES["pilot-notes"]["artifact_keys"]
    partial = {keys[0]: "the writer stopped after one section"}
    write_json(deck / "manual_prose.json", partial)
    with pytest.raises(ac.MissingInput) as excinfo:
        ac.record(SLUG, "pilot-notes")
    message = str(excinfo.value)
    # The message must name what is missing, so the fix is obvious without a re-read.
    for absent in keys[1:]:
        assert absent in message


def test_record_accepts_a_complete_artifact(deck):
    keys = config.AGENT_ROUTINES["pilot-notes"]["artifact_keys"]
    write_json(deck / "manual_prose.json", {k: f"{k} prose" for k in keys})
    entry, _ = ac.record(SLUG, "pilot-notes")
    assert entry["artifact"].endswith("manual_prose.json")


def test_pilot_notes_does_not_own_a_cover_key(deck):
    """The cover is renderer furniture; build_manual never reads prose['cover'].

    The key was declared, produced on every deck, cached — and rendered nowhere.
    """
    assert "cover" not in config.AGENT_ROUTINES["pilot-notes"]["artifact_keys"]


# ── Sidecar ──────────────────────────────────────────────────────────────


def test_save_cache_skips_identical_write(deck):
    ac.record(SLUG, "pilot-notes")
    _, wrote_again = ac.record(SLUG, "pilot-notes")
    assert wrote_again is False


def test_sidecar_has_no_timestamps(deck):
    """No generated dates anywhere — every diff line must mean something.

    Checks keys rather than raw text: paths legitimately contain arbitrary
    words (pytest's own tmp dir is named after this test).
    """
    ac.record(SLUG, "pilot-notes")
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
    ac.record(SLUG, "pilot-notes")
    ac.record(SLUG, "deck-recon")
    dropped = ac.clear(SLUG, "pilot-notes")
    assert dropped == ["pilot-notes"]
    assert ac.status(SLUG, "pilot-notes")["status"] == "MISS"
    assert ac.status(SLUG, "deck-recon")["status"] == "HIT"


# ── The strategy-DB hazard ───────────────────────────────────────────────


def test_strategy_db_rebuild_does_not_invalidate(deck, monkeypatch):
    """Rebuilding the derived index must not cost 330k tokens."""
    calls = {"n": 0}

    def fake_digest():
        calls["n"] += 1
        return "constant-doc-hash"

    monkeypatch.setattr(ac, "strategy_doc_digest", fake_digest)
    before = fp(SLUG, "pilot-notes")
    # A rebuild rewrites strategy_index.json / .strategy-db-meta.json, never the doc.
    assert fp(SLUG, "pilot-notes") == before
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
    assert "pilot-notes" in out
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
    # strategic-frame has no applicability gate — every deck gets one.
    assert "strategic-frame" not in na


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


# ── Applicability gating (tutor-guide, debrief) ─────────────────────────


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
        {"name": "Filler Land", "oracle_text": "T: Add C."},
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
    entry, _ = ac.record(SLUG, "pilot-notes")
    assert "Sac Outlet" in entry["card_refs_by_key"]["mulligan"]
    assert entry["card_refs_by_key"]["how_it_wins"] == []
    # a legacy key (card_roles) is not owned, so it is not a ref source either
    assert "card_roles" not in entry["card_refs_by_key"]


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
    entry, _ = ac.record(SLUG, "pilot-notes")
    assert set(entry["key_fingerprints"]) == {
        "how_it_wins", "combo_lines", "mulligan", "threat_assessment", "matchups"}


def test_goldfish_change_stales_only_goldfish_keys(deck):
    _two_card_deck(deck)
    ac.record(SLUG, "pilot-notes")
    write_json(deck / "goldfish_metrics.json", {"meta": {"seed": 42},
                                                "metrics": {"new": True}})
    ac._SHA_MEMO.clear(); common.clear_memo()
    result = ac.status(SLUG, "pilot-notes")
    assert result["status"] == "MISS"
    assert set(result["stale_keys"]) == {"how_it_wins", "mulligan",
                                         "threat_assessment", "matchups"}
    assert "combo_lines" not in result["stale_keys"], "the one key that ignores goldfish"


def test_new_passing_stack_stales_stack_keys(deck):
    _two_card_deck(deck)
    ac.record(SLUG, "pilot-notes")
    write_json(deck / "stacks" / "003-new.json", stack_doc("003"))
    ac._SHA_MEMO.clear(); common.clear_memo()
    result = ac.status(SLUG, "pilot-notes")
    assert result["status"] == "MISS"
    assert set(result["stale_keys"]) == {"combo_lines", "how_it_wins",
                                         "threat_assessment", "matchups"}
    assert "mulligan" not in result["stale_keys"], "the one key that ignores stacks"


def test_unreferenced_card_change_refines_stale_keys(deck):
    """A card change only stales the keys whose refs include a changed card;
    with no key referencing it, the whole routine is STALE_OK instead."""
    _two_card_deck(deck)
    prose = json.loads((deck / "manual_prose.json").read_text())
    prose["mulligan"] = "Keep Sac Outlet hands."
    write_json(deck / "manual_prose.json", prose)
    ac._SHA_MEMO.clear(); common.clear_memo()
    ac.record(SLUG, "pilot-notes")
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"][1]["oracle_text"] = "changed filler"
    write_json(deck / "cards.json", cards)
    ac._SHA_MEMO.clear(); common.clear_memo()
    result = ac.status(SLUG, "pilot-notes")
    assert result["status"] == "STALE_OK"


def test_referenced_card_change_names_the_stale_keys(deck):
    _two_card_deck(deck)
    prose = json.loads((deck / "manual_prose.json").read_text())
    prose["mulligan"] = "Keep Sac Outlet hands."
    prose["how_it_wins"] = "No names here."
    write_json(deck / "manual_prose.json", prose)
    ac._SHA_MEMO.clear(); common.clear_memo()
    ac.record(SLUG, "pilot-notes")
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"][0]["oracle_text"] = "Sacrifice everything."
    write_json(deck / "cards.json", cards)
    ac._SHA_MEMO.clear(); common.clear_memo()
    result = ac.status(SLUG, "pilot-notes")
    assert result["status"] == "MISS"
    assert "mulligan" in result["stale_keys"]
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


# ── Bulk re-record: the gates are the feature ────────────────────────────


def _snap_of(slug):
    return ac.snapshot(slug)


def test_snapshot_captures_status_and_artifact_digest(deck):
    _two_card_deck(deck)
    ac.record(SLUG, "stack:001")
    snap = _snap_of(SLUG)
    assert snap["stack:001"]["status"] == "HIT"
    assert snap["stack:001"]["artifact_sha256"]


def test_rerecord_refixes_what_a_format_change_invalidated(deck, monkeypatch):
    """The intended path: HIT before, MISS purely because inputs were rescoped."""
    _two_card_deck(deck)
    ac.record(SLUG, "stack:001")
    snap = _snap_of(SLUG)

    # Simulate a cache-format change: same artifact, different fingerprint.
    real = ac.fingerprint
    monkeypatch.setattr(ac, "fingerprint", lambda *a, **k: "deadbeef" + real(*a, **k)[:8])
    assert ac.status(SLUG, "stack:001")["status"] == "MISS"

    planned, refused, done = ac.rerecord(SLUG, snap)
    assert planned == ["stack:001"] and done == ["stack:001"]
    # Every OTHER routine in the fixture was never recorded, so it was MISS in the
    # snapshot too — and is correctly refused rather than swept along.
    assert "stack:001" not in dict(refused)
    assert all("real work" in why for _, why in refused)
    assert ac.status(SLUG, "stack:001")["status"] == "HIT"


def test_rerecord_REFUSES_a_routine_that_was_already_miss(deck, monkeypatch):
    """The gate that matters: never freeze genuinely stale work as a HIT."""
    _two_card_deck(deck)
    # Never recorded -> MISS in the snapshot.
    snap = _snap_of(SLUG)
    assert snap["stack:001"]["status"] == "MISS"

    planned, refused, done = ac.rerecord(SLUG, snap)
    assert planned == [] and done == []
    assert ("stack:001", "was MISS before the change — real work") in refused


def test_rerecord_REFUSES_when_the_artifact_changed_since_the_snapshot(deck, monkeypatch):
    _two_card_deck(deck)
    ac.record(SLUG, "stack:001")
    snap = _snap_of(SLUG)

    # Someone edited the artifact after the snapshot — that is real content movement.
    doc = stack_doc("001")
    doc["resolution"]["final_state"]["summary"] = "rewritten by hand"
    write_json(deck / "stacks" / "001-first.json", doc)
    ac._SHA_MEMO.clear(); common.clear_memo()
    ac.record(SLUG, "stack:001")          # re-record so artifact_sha256 moves
    real = ac.fingerprint
    monkeypatch.setattr(ac, "fingerprint", lambda *a, **k: "deadbeef" + real(*a, **k)[:8])

    planned, refused, done = ac.rerecord(SLUG, snap)
    assert planned == [] and done == []
    assert any("artifact changed" in why for _, why in refused)


def test_rerecord_dry_run_changes_nothing(deck, monkeypatch):
    _two_card_deck(deck)
    ac.record(SLUG, "stack:001")
    snap = _snap_of(SLUG)
    before = json.dumps(ac.load_cache(SLUG), sort_keys=True)
    real = ac.fingerprint
    monkeypatch.setattr(ac, "fingerprint", lambda *a, **k: "deadbeef" + real(*a, **k)[:8])

    planned, refused, done = ac.rerecord(SLUG, snap, dry_run=True)
    assert planned == ["stack:001"] and done == []
    assert json.dumps(ac.load_cache(SLUG), sort_keys=True) == before


def test_rerecord_without_a_snapshot_refuses_outright(deck):
    _two_card_deck(deck)
    ac.record(SLUG, "stack:001")
    with pytest.raises(ac.MissingInput, match="no snapshot entry"):
        ac.rerecord(SLUG, None)


def test_refs_ignore_the_checker_block(deck):
    """A checker's notes ABOUT a card are not the artifact USING that card.

    Observed on yawgmoth-swarm: an `iteration_bound_override.reason` written by the
    orchestrator named the cards a swap had just added, so those cards entered the
    stack's refs and it could never be STALE_OK for that swap — the artifact was
    pinned by prose describing the very change being evaluated. `checker` is already
    excluded from the fingerprint by `scenario_block_digest`; refs now agree.
    """
    _two_card_deck(deck)
    doc = stack_doc("001")
    doc["scenario"]["question"] = "What does Sac Outlet do here?"
    doc["checker"]["iteration_bound_override"] = {
        "reason": "the deck changed (Filler Land in) and re-MISSed this routine"
    }
    write_json(deck / "stacks" / "001-first.json", doc)
    ac._SHA_MEMO.clear(); common.clear_memo()

    entry, _ = ac.record(SLUG, "stack:001")
    assert "Sac Outlet" in entry["card_refs"]          # named by the scenario
    assert "Filler Land" not in entry["card_refs"]     # named ONLY by checker prose

    # ...so a change to that card is now correctly STALE_OK rather than a MISS.
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"][1]["oracle_text"] = "T: Add one mana of any color."
    write_json(deck / "cards.json", cards)
    ac._SHA_MEMO.clear(); common.clear_memo()
    assert ac.status(SLUG, "stack:001")["status"] == "STALE_OK"


def test_goldfish_provenance_stamp_is_excluded_from_the_fingerprint(deck):
    """A decklist edit that moves no metric must not MISS the prose routines.

    goldfish_metrics.json embeds meta.decklist_sha256, so ANY decklist change
    rewrote the file and invalidated strategic-frame, pilot-notes,
    tutor-guide and every decision — regardless of whether a single
    figure moved. Observed directly: restoring comment lines to a decklist
    re-MISSed five prose routines whose numbers were byte-identical.
    """
    gf = deck / "goldfish_metrics.json"
    write_json(gf, {"meta": {"seed": 42, "decklist_sha256": "aaa"},
                    "metrics": {"commander": {"mean_cast_turn": 4.2}}})
    ac._SHA_MEMO.clear(); common.clear_memo()
    before = fp(SLUG, "pilot-notes")

    # Same metrics, new provenance stamp — the decklist changed, the maths did not.
    write_json(gf, {"meta": {"seed": 42, "decklist_sha256": "bbb"},
                    "metrics": {"commander": {"mean_cast_turn": 4.2}}})
    ac._SHA_MEMO.clear(); common.clear_memo()
    assert fp(SLUG, "pilot-notes") == before, "provenance stamp still invalidates"

    # But a real metric moving must still invalidate.
    write_json(gf, {"meta": {"seed": 42, "decklist_sha256": "bbb"},
                    "metrics": {"commander": {"mean_cast_turn": 5.9}}})
    ac._SHA_MEMO.clear(); common.clear_memo()
    assert fp(SLUG, "pilot-notes") != before, "a changed metric must invalidate"


def test_exclusion_is_by_omission_so_new_fields_stay_covered(deck):
    """Exclusion, not inclusion: a field added later is covered by default."""
    gf = deck / "goldfish_metrics.json"
    write_json(gf, {"meta": {"decklist_sha256": "aaa"}, "metrics": {"a": 1}})
    ac._SHA_MEMO.clear(); common.clear_memo()
    before = fp(SLUG, "pilot-notes")
    # A metric nobody anticipated appears. An inclusion list would ignore it.
    write_json(gf, {"meta": {"decklist_sha256": "aaa"},
                    "metrics": {"a": 1}, "brand_new_section": {"x": 1}})
    ac._SHA_MEMO.clear(); common.clear_memo()
    assert fp(SLUG, "pilot-notes") != before
