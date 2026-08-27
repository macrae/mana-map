"""Issue-plan form validator: identity block, department system, contract integrity.

LEGACY (2026-08-19): the magazine renderer. It still renders the nine frozen issues from
artifacts nothing regenerates any more (issue_plan.json, the panel keys,
card_roles/mana_base/upgrades, considering.json), and it is replaced by the compact deck
page in docs/manual-v5-spec.md. Do not extend it; internals below are accurate for what it
does.
"""

from manamap.pilot.issue_spec import COPY_DEPARTMENTS, DEPARTMENT_IDS
from manamap.pilot.validate_issue import validate_identity, validate_plan
from conftest import ROOT

GOOD_IDENTITY = {
    "volume": 1,
    "issue_date": "August 2026",
    "cover_price": "$4.95",
    "deck_name": "GOBLIN STORM",
    "commander": "Zada, Hedron Grinder",
    "cover_tagline": "Goblins all the way down",
    "next_issue": "HAPATRA, VIZIER OF POISON",
    "decklist_sha256": "c" * 64,
}

CARD_NAMES = {"Zada, Hedron Grinder", "Skirk Prospector", "Haze of Rage"}


def good_plan():
    """A minimal plan that passes: all departments, copy where required."""
    departments = []
    for dept_id in DEPARTMENT_IDS:
        dept = {"id": dept_id}
        if dept_id in COPY_DEPARTMENTS:
            dept.update(kicker="KICK", headline="HEAD", dek="A dek sentence.")
        departments.append(dept)
    return {
        "slug": "goblin-storm",
        "angle": "A deck that looks like chump blockers until it draws five cards.",
        "cover": {
            "dominant_coverline": "THE HAZE LOOP",
            "teases": ["Krenko's infinite, busted"],
            "violators": [{"text": "5 VERIFIED LINES!"}],
        },
        "departments": departments,
    }


# ── Identity ─────────────────────────────────────────────────────────────


def test_good_identity():
    assert validate_identity(GOOD_IDENTITY) == []


def test_identity_missing_keys():
    bad = {k: v for k, v in GOOD_IDENTITY.items() if k != "next_issue"}
    assert any("missing keys" in e for e in validate_identity(bad))


def test_identity_volume_must_be_positive_int():
    bad = dict(GOOD_IDENTITY, volume="one")
    assert any("positive integer" in e for e in validate_identity(bad))


# ── Plan structure ───────────────────────────────────────────────────────


def test_good_plan_passes():
    assert validate_plan(good_plan(), CARD_NAMES) == []


def test_missing_top_level_key():
    plan = good_plan()
    del plan["angle"]
    assert any("Missing top-level keys" in e for e in validate_plan(plan))


def test_angle_required():
    plan = good_plan()
    plan["angle"] = ""
    assert any("angle is required" in e for e in validate_plan(plan, CARD_NAMES))


def test_cover_needs_coverline_and_tease():
    plan = good_plan()
    plan["cover"] = {"dominant_coverline": "", "teases": []}
    errors = validate_plan(plan, CARD_NAMES)
    assert any("dominant_coverline" in e for e in errors)
    assert any("teases" in e for e in errors)


def test_too_many_violators():
    plan = good_plan()
    plan["cover"]["violators"] = [{"text": "A"}, {"text": "B"}, {"text": "C"}]
    assert any("violators" in e for e in validate_plan(plan, CARD_NAMES))


def test_missing_department_flagged():
    plan = good_plan()
    plan["departments"] = [d for d in plan["departments"] if d["id"] != "command-zone"]
    assert any("command-zone" in e for e in validate_plan(plan, CARD_NAMES))


def test_unknown_department_flagged():
    plan = good_plan()
    plan["departments"].append({"id": "letters-page"})
    assert any("unknown department" in e for e in validate_plan(plan, CARD_NAMES))


def test_out_of_order_departments():
    plan = good_plan()
    plan["departments"][2], plan["departments"][5] = (
        plan["departments"][5], plan["departments"][2],
    )
    assert any("canonical order" in e for e in validate_plan(plan, CARD_NAMES))


def test_copy_department_missing_headline():
    plan = good_plan()
    for dept in plan["departments"]:
        if dept["id"] == "the-kill":
            dept["headline"] = ""
    assert any("the-kill" in e and "headline" in e for e in validate_plan(plan, CARD_NAMES))


# ── Contract integrity ───────────────────────────────────────────────────


def test_department_may_not_restyle_its_tier():
    plan = good_plan()
    for dept in plan["departments"]:
        if dept["id"] == "at-the-table":         # coaching department...
            dept["tiers"] = ["verified"]          # ...claiming verified costume
    errors = validate_plan(plan, CARD_NAMES)
    assert any("may not restyle its evidence tier" in e for e in errors)


def test_correct_tier_claim_is_allowed():
    plan = good_plan()
    for dept in plan["departments"]:
        if dept["id"] == "by-the-numbers":
            dept["tiers"] = ["data"]
    assert validate_plan(plan, CARD_NAMES) == []


def test_unknown_component_rejected():
    plan = good_plan()
    plan["departments"][2]["components"] = ["pilot-tip", "lava-lamp"]
    assert any("unknown component" in e for e in validate_plan(plan, CARD_NAMES))


def test_pilot_tip_must_name_a_real_card():
    plan = good_plan()
    plan["departments"][2]["pilot_tips"] = [{"card": "Black Lotus", "text": "Nope."}]
    assert any("not in the deck" in e for e in validate_plan(plan, CARD_NAMES))


def test_pilot_tip_needs_text():
    plan = good_plan()
    plan["departments"][2]["pilot_tips"] = [{"card": "Skirk Prospector", "text": ""}]
    assert any("has no text" in e for e in validate_plan(plan, CARD_NAMES))


def test_caption_must_name_a_real_card():
    plan = good_plan()
    plan["departments"][2]["captions"] = {"Mox Emerald": "**NOPE:** not here."}
    assert any("not in the deck" in e for e in validate_plan(plan, CARD_NAMES))


def test_card_checks_skipped_when_names_unavailable():
    plan = good_plan()
    plan["departments"][2]["pilot_tips"] = [{"card": "Anything", "text": "fine"}]
    assert validate_plan(plan, None) == []


def test_departments_are_stable_across_issues():
    """The department system is the reading experience — it must not drift."""
    assert DEPARTMENT_IDS[0] == "cover"
    assert DEPARTMENT_IDS[-1] == "back-page"
    assert "command-zone" in DEPARTMENT_IDS      # the Commander Mandate
    assert "judges-desk" in DEPARTMENT_IDS       # the proof
    assert "featured-artist" in DEPARTMENT_IDS   # who painted your deck
    assert "at-the-table" in DEPARTMENT_IDS      # the whole of Act III
    assert "sources-say" in DEPARTMENT_IDS       # the mana audit
    # NOT `len(...) == 17`. A hardcoded count here is the same mistake this repo
    # bans in prose — "never transcribe the section list or its count" — and it
    # turns every deliberate addition into a test edit that says nothing. What
    # must not drift is the SHAPE: the five acts partition the list exactly, every
    # department is spoken for, and an optional one is a real department rather
    # than a stray id. `test_docs_section_count` guards the number against the
    # spec, which is the only place it belongs.
    from manamap.pilot.issue_spec import ACTS, OPTIONAL_DEPARTMENTS
    flattened = [d for _title, ids in ACTS for d in ids]
    assert flattened == [d for d in DEPARTMENT_IDS if d not in ("cover", "contents")]
    assert OPTIONAL_DEPARTMENTS <= set(DEPARTMENT_IDS)


def test_canonical_plan_has_no_adjacent_dense_departments():
    """The shipped department order must itself satisfy the rhythm rule."""
    assert validate_plan(good_plan(), CARD_NAMES) == []


# ── Featured Artist ──────────────────────────────────────────────────────

ARTISTS = {"Wizard of Barge", "Jesper Ejsing"}


def test_featured_artist_must_have_painted_a_card():
    plan = good_plan()
    for dept in plan["departments"]:
        if dept["id"] == "featured-artist":
            dept["featured"] = {"artist": "Rembrandt", "note": "Nope."}
    errors = validate_plan(plan, CARD_NAMES, ARTISTS)
    assert any("painted no card" in e for e in errors)


def test_also_worth_noting_artists_are_checked_too():
    plan = good_plan()
    for dept in plan["departments"]:
        if dept["id"] == "featured-artist":
            dept["also_worth_noting"] = [{"artist": "Nobody At All", "note": "x"}]
    assert any("painted no card" in e for e in validate_plan(plan, CARD_NAMES, ARTISTS))


def test_real_artists_pass():
    plan = good_plan()
    for dept in plan["departments"]:
        if dept["id"] == "featured-artist":
            dept["featured"] = {"artist": "Wizard of Barge", "note": "The drop."}
            dept["also_worth_noting"] = [{"artist": "Jesper Ejsing", "note": "Four."}]
    assert validate_plan(plan, CARD_NAMES, ARTISTS) == []


def test_artist_checks_skipped_when_unavailable():
    plan = good_plan()
    for dept in plan["departments"]:
        if dept["id"] == "featured-artist":
            dept["featured"] = {"artist": "Anyone", "note": "x"}
    assert validate_plan(plan, CARD_NAMES, None) == []


# ── decklist_sha256 stamping (deck versioning, zeroth step) ──────────────


def _identity(**overrides):
    doc = {"volume": 1, "issue_date": "August 2026", "cover_price": "$4.95",
           "deck_name": "TEST", "commander": "Zada, Hedron Grinder",
           "cover_tagline": "t", "next_issue": "NEXT",
           "decklist_sha256": "a" * 64}
    doc.update(overrides)
    return doc


def test_issue_without_a_decklist_hash_fails():
    from manamap.pilot.validate_issue import validate_identity
    doc = _identity()
    del doc["decklist_sha256"]
    errors = validate_identity(doc)
    assert any("decklist_sha256" in e for e in errors)


def test_matching_decklist_hash_passes():
    from manamap.pilot.validate_issue import validate_identity
    assert validate_identity(_identity(), deck_sha256="a" * 64) == []


def test_a_changed_decklist_is_caught_at_the_identity_gate():
    from manamap.pilot.validate_issue import validate_identity
    errors = validate_identity(_identity(), deck_sha256="b" * 64)
    assert any("decklist changed after this issue was published" in e for e in errors)


def test_no_cards_json_means_no_hash_comparison():
    from manamap.pilot.validate_issue import validate_identity
    assert validate_identity(_identity(), deck_sha256=None) == []


# ── Self-containment (STYLEv3 L10) ───────────────────────────────────────


from manamap.pilot.validate_issue import _lint_strings, _CONTINUITY_RE


def test_lint_catches_version_references():
    errs = _lint_strings({"body": "V2's answer is: accruing."}, "plan")
    assert errs and "changelog voice" in errs[0]


def test_lint_catches_history_and_supersession():
    assert _lint_strings({"x": "see HISTORY.md for the record"}, "f")
    assert _lint_strings({"x": "the previous build ran this"}, "f")
    assert _lint_strings({"x": "superseded by the new list"}, "f")


def test_lint_passes_legitimate_vocabulary():
    clean = {"a": "deploy in waves against the sweeper",
             "b": "the sideboard swap costs one slot",
             "c": "a d20 roll of 15 draws fifteen cards",
             "d": "Vol. 007 of Pilot's Manual"}
    assert _lint_strings(clean, "f") == []


def test_lint_is_case_insensitive():
    assert _CONTINUITY_RE.search("v3 added a second entry")
    assert _CONTINUITY_RE.search("V3 added a second entry")


def test_lint_names_the_path():
    errs = _lint_strings({"departments": [{"body": "as v2 showed"}]}, "issue_plan.json")
    assert "departments[0].body" in errs[0]


# ── Land counts: entries are never a land count ──────────────────────────


def _deck_with_mana(tmp_path, monkeypatch, prose, entries=18, total=33):
    import json as _json
    decks = tmp_path / "decks"
    base = decks / "test-deck"
    base.mkdir(parents=True)
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", decks)
    (base / "mana_analysis.json").write_text(_json.dumps(
        {"lands": {"total": total, "entries": entries, "enters_tapped": 5}}))
    (base / "manual_prose.json").write_text(_json.dumps(prose))
    return base


def test_entry_count_quoted_as_a_land_count_is_rejected(tmp_path, monkeypatch):
    """The exact bug that shipped: prose read lands.entries as the land count."""
    from manamap.pilot.validate_issue import validate_land_counts

    base = _deck_with_mana(tmp_path, monkeypatch,
                           {"mana_base": "The 18-land bet shows in the goldfish."})
    errors = validate_land_counts(base, good_plan())
    assert any("distinct land CARDS" in e and "33 lands" in e for e in errors)


def test_true_land_count_passes(tmp_path, monkeypatch):
    from manamap.pilot.validate_issue import validate_land_counts

    base = _deck_with_mana(tmp_path, monkeypatch,
                           {"mana_base": "Thirty-three lands, 18 entries, 5 tapped."})
    assert validate_land_counts(base, good_plan()) == []


def test_lint_is_silent_when_entries_equal_copies(tmp_path, monkeypatch):
    """A deck with no stacked basics has nothing to confuse."""
    from manamap.pilot.validate_issue import validate_land_counts

    base = _deck_with_mana(tmp_path, monkeypatch,
                           {"mana_base": "A 33-land base."}, entries=33, total=33)
    assert validate_land_counts(base, good_plan()) == []


# ── The per-byline voice lint (STYLEv3 §7.7) ────────────────────────────


def test_the_voice_lint_fires_on_the_sentence_that_caused_it():
    """The exemplar, verbatim from the editor who caught it.

    Coach Sunny Brightside — whose bio is "has never once believed you're going to
    lose" — shipped "the deflection posture the strategic frame prescribes" in
    Vol. 009. The verdict was "that's not a coach, that's a McKinsey deck".
    """
    from manamap.pilot.validate_issue import _voice_violations
    hits = set(_voice_violations(
        "Coach Sunny Brightside",
        "Adopt the deflection posture the strategic frame prescribes."))
    assert {"posture", "prescribes", "strategic frame"} <= hits


def test_the_voice_lint_matches_words_not_substrings():
    """`"very "` as a substring matches "e-very ".

    The first version of this lint reported 13 violations across the fleet, and
    every `very` hit was the word "every" inside a correct sentence. It was caught
    by running it on real decks before shipping — which is the rule this repo has
    written down repeatedly and broke here anyway.
    """
    from manamap.pilot.validate_issue import _voice_violations
    clean = 'A discount is one generic off every black spell, every turn.'
    assert not list(_voice_violations('"Ledger" Lin Marginal', clean))
    assert "huge" in set(_voice_violations('"Ledger" Lin Marginal', "A huge 40.2%."))


def test_a_shared_department_only_bans_what_both_voices_are_barred_from():
    """Keep or Ship is signed by Sunny AND Ledger, so either may have written a
    given sentence. Flagging a Sunny-only ban there would fire on correct copy."""
    from manamap.pilot.issue_spec import voices_for
    assert len(voices_for("mulligan")) == 2
    assert len(voices_for("mana_base")) == 1


def test_every_prose_key_the_renderer_reads_is_voice_mapped():
    """A key that drifts out of `PROSE_KEY_DEPARTMENT` stops being voice-checked
    SILENTLY — the lint finds nothing and passes, which looks identical to clean
    prose. This is the drift guard for that correspondence."""
    import re
    from pathlib import Path

    from manamap.pilot.issue_spec import PROSE_KEY_DEPARTMENT
    src = (ROOT / "src/manamap/pilot/build_manual.py").read_text()
    rendered = set(re.findall(r'prose\(prose_doc, "(\w+)"', src))
    # `pilots_log` carries a voice per turn and is checked by a different path;
    # `card_roles` is a dict the renderer reads directly rather than via prose().
    unmapped = rendered - set(PROSE_KEY_DEPARTMENT) - {"pilots_log"}
    assert not unmapped, (
        f"{sorted(unmapped)} render into the magazine and are not in "
        f"PROSE_KEY_DEPARTMENT, so no voice check applies to them")


# ── The length budget (STYLEv3 7.1) ──────────────────────────────────────


def test_the_budget_is_reported_by_default_and_fails_only_under_strict(tmp_path):
    """Eight issues were written before any budget existed. Failing them all the
    day it lands turns eight tracked artifacts red for copy that was correct when
    it shipped, and a team that sees permanent red learns to ignore red. `--strict`
    is the gate for new work; the plain run says where you stand."""
    import json
    from manamap.pilot.validate_issue import validate_budget

    plan = {"departments": [
        {"id": "the-kill",
         "dek": "One. Two. Three.",
         "callouts": [{"text": "A. B. C. D."}],
         "pilot_tips": [{"card": "X", "text": "Do it. Then do it again."}]},
    ]}
    (tmp_path / "manual_prose.json").write_text(json.dumps({
        "matchups": "x" * 4000, "mulligan": "ok"}))
    notes = validate_budget(tmp_path, plan)
    joined = "\n".join(notes)
    assert "the-kill.dek: 3 sentences" in joined
    assert "the-kill.callouts[0]: 4 sentences" in joined
    assert "the-kill.pilot_tips[0]: 2 sentences" in joined
    assert "matchups: 4,000 chars" in joined
    # A field under budget is silent — a validator that lists compliant fields is
    # a validator nobody reads to the end of.
    assert "mulligan" not in joined


def test_every_budget_is_a_length_some_deck_already_achieves():
    """A cap nothing can reach is a target, not a limit. Each one here was
    calibrated against the fleet — EXCEPT the two the budget exists to bind,
    whose number comes from the engine model's own measured cap instead."""
    from manamap.pilot.issue_spec import PROSE_BUDGET
    from manamap.pilot.validate_engine import MAX_WHAT_IT_DOES

    deliberately_unreachable = {"threat_assessment", "matchups"}
    assert deliberately_unreachable <= set(PROSE_BUDGET)
    for key in deliberately_unreachable:
        assert PROSE_BUDGET[key] > MAX_WHAT_IT_DOES, (
            f"{key} is capped below the measured revisability limit — that is a "
            f"stricter claim than the evidence supports")
    # Nothing is capped so tightly that a normal paragraph breaches it.
    assert all(cap >= 800 for cap in PROSE_BUDGET.values())


def test_the_budget_reads_the_spec_not_a_transcribed_list():
    """The same rule the department ids live under: one source, no copies."""
    import inspect

    from manamap.pilot import validate_issue
    source = inspect.getsource(validate_issue.validate_budget)
    for constant in ("PROSE_BUDGET", "ENTRY_BUDGET", "BRANCH_BUDGET"):
        assert constant in source
    # No cap typed into the checker itself.
    assert "2500" not in source and "1900" not in source


# ── The Kill's feature set ───────────────────────────────────────────────

import json as _json

from manamap.pilot.validate_issue import validate_features


def _deck_with_stacks(tmp_path, ids=("001", "002", "003"), refused=()):
    """A deck dir holding presentable stacks, plus any marked non-presentable."""
    base = tmp_path / "deck"
    (base / "stacks").mkdir(parents=True)
    for sid in list(ids) + list(refused):
        doc = {
            "id": sid, "title": f"Line {sid}: the question",
            "resolution": {"steps": [], "final_state": {"summary": "s"}},
            "checker": {"verdict": "pass", "iterations": 1},
        }
        if sid in refused:
            doc["presentable"] = False
        (base / "stacks" / f"{sid}-line.json").write_text(_json.dumps(doc))
    return base


def _plan_with_features(features):
    dept = {"id": "the-kill", "kicker": "K", "headline": "H", "dek": "D"}
    if features is not None:
        dept["features"] = features
    return {"departments": [dept]}


def test_features_absent_is_the_default_and_checks_nothing(tmp_path):
    base = _deck_with_stacks(tmp_path)
    assert validate_features(base, _plan_with_features(None)) == []


def test_a_valid_subset_passes(tmp_path):
    base = _deck_with_stacks(tmp_path)
    assert validate_features(base, _plan_with_features(["002"])) == []


def test_featuring_a_line_that_does_not_exist_is_an_error(tmp_path):
    """The sharp failure mode: a typo silently demotes the issue's best line to
    an index row and nothing else changes, because the renderer skips unknown
    ids rather than crashing."""
    base = _deck_with_stacks(tmp_path)
    errors = validate_features(base, _plan_with_features(["009"]))
    assert any("'009'" in e and "not a presentable stack" in e for e in errors)


def test_featuring_a_refused_line_is_an_error(tmp_path):
    """A non-presentable stack failed the publication gate. Featuring one is the
    single mistake this department must never make."""
    base = _deck_with_stacks(tmp_path, ids=("001",), refused=("004",))
    errors = validate_features(base, _plan_with_features(["004"]))
    assert any("'004'" in e for e in errors)


def test_a_repeated_feature_is_an_error(tmp_path):
    base = _deck_with_stacks(tmp_path)
    errors = validate_features(base, _plan_with_features(["002", "002"]))
    assert any("repeats" in e for e in errors)


def test_an_empty_features_list_is_an_error(tmp_path):
    """`[]` would read as "feature nothing" in the issue's peak department.
    Omitting the key is how you say "feature everything"."""
    base = _deck_with_stacks(tmp_path)
    errors = validate_features(base, _plan_with_features([]))
    assert any("non-empty" in e for e in errors)


def test_features_naming_every_stack_is_an_error(tmp_path):
    """It renders identically to omitting the key and leaves an empty index —
    so it is a plan restating the default, which will rot the moment a stack
    is added and nobody remembers to extend the list."""
    base = _deck_with_stacks(tmp_path)
    errors = validate_features(base, _plan_with_features(["001", "002", "003"]))
    assert any("every presentable stack" in e for e in errors)
