"""Issue-plan form validator: identity block, department system, contract integrity."""


from manamap.pilot.issue_spec import COPY_DEPARTMENTS, DEPARTMENT_IDS
from manamap.pilot.validate_issue import validate_identity, validate_plan

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
        if dept["id"] == "politics-table":       # coaching department...
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
    assert "fetch-quests" in DEPARTMENT_IDS      # what to tutor
    assert "sources-say" in DEPARTMENT_IDS       # the mana audit
    assert len(DEPARTMENT_IDS) == 17


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
