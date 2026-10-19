"""The fleet artifact: profiles across decks, bands per bracket, never a score.

Driven through `fleet_analytics.build()` on a sandbox fleet of hand-made decks,
so every rule is proven on data whose answer is known: the group key, the
bands at n=3 and their absence at n=2, absent-never-zero, determinism, the
Spearman refusal, and the validator's refusal of a composite number.
"""

import copy
import json

import pytest

from manamap import config
from manamap.pilot import calibrate, diagnostic, fleet_analytics as fa, validate_fleet_analytics as vfa

SHA = {s: (chr(97 + i) * 64) for i, s in enumerate(("alpha", "bravo", "charlie", "delta", "echo", "foxtrot"))}


def _deck(base, slug, *, target=None, floor=3, kill8=0.5, missed=0.2, bench=True, modern=False,
          dti=None, feel=None, apart=False, stage="sleeved"):
    d = base / slug
    d.mkdir(parents=True)
    cards = {"deck": slug, "decklist_sha256": SHA[slug],
             "cards": [{"name": "Zada, Hedron Grinder", "is_commander": not modern,
                        "type_line": "Legendary Creature"}]}
    if modern:
        cards["format"] = "modern"
    (d / "cards.json").write_text(json.dumps(cards))
    (d / "decklist.txt").write_text("1 Zada, Hedron Grinder\n")
    (d / "info.json").write_text(json.dumps({"stage": stage, "simulation": None}))
    if not modern:
        (d / "bracket_report.json").write_text(json.dumps(
            {"floor": floor, "floor_name": "Upgraded", "target": target, "decklist_sha256": SHA[slug]}))
    if bench and not modern:
        (d / "benchmark.json").write_text(json.dumps({
            "slug": slug, "harness": {"version": 1, "iterations": 1000}, "decklist_sha256": SHA[slug],
            "metrics": {"mana_screw": {"missed_land_drop_rate": missed, "mulligan_rate": 0.2,
                                       "mana_at_turn_five": 4.5},
                        "speed": {"kill_by_turn_6": kill8 / 2, "kill_by_turn_8": kill8,
                                  "kill_by_turn_10": min(1.0, kill8 + 0.3), "never_kills_rate": 0.1,
                                  "median_kill_turn": 8},
                        "consistency": {"mana_stdev_turn_five": 1.2},
                        "response": {"answer_cards": 12, "classes_covered": 4, "classes_possible": 5}},
            "series": {"kill_by_turn_rate": {str(t): kill8 * t / 10 for t in range(1, 11)},
                       "mean_available_mana_by_turn": {str(t): float(t) for t in range(1, 11)},
                       "land_drop_hit_rate_by_turn": {str(t): 0.9 for t in range(1, 11)}},
            "score": None, "limits": []}))
    if dti is not None:
        (d / "deckcheck.json").write_text(json.dumps({
            "schema_version": 1, "slug": slug, "branch": None, "source": "deckcheck.co",
            "last_pulled": "2026-10-18",
            "readings": [{"as_of": "2026-10-18", "source_updated_at": "x", "decklist_sha256": SHA[slug],
                          "deckcheck_id": "id", "url": "u", "score": dti, "max_score": 96, "bracket": 3,
                          "benchmarks": {k: "B" for k in sorted(
                              sum((list(v) for v in {"R": ("R1", "R2"), "A": ("A1", "A2"),
                                                     "P": ("P1", "P2", "P3"), "I": ("I1", "I2"),
                                                     "S": ("S1", "S2", "S3")}.values()), []))}}],
            "omni": {"n": 1, "median_score": dti, "median_bracket": 3, "benchmarks": {}},
            "floor_vs_dti": {}}))
    if feel:
        lines = [{"id": f"{i + 1:03d}", "at": "2026-10-01T20:00", "decklist_sha256": SHA[slug],
                  "result": "win", "opponents": 3, "tags": [], "text": "g", "feel": f}
                 for i, f in enumerate(feel)]
        (d / "log.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    if apart:
        (d / "deck_versions.json").write_text(json.dumps(
            {"slug": slug, "paper": None, "tags": {},
             "lifecycle": {"status": "broken-down", "at": "2026-01-01", "reason": "x"}}))
    return d


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    decks = tmp_path / "decks"
    decks.mkdir()
    monkeypatch.setattr("manamap.config.DECKS_DIR", decks)
    monkeypatch.setattr("manamap.config.FLEET_DIR", tmp_path / "fleet")
    monkeypatch.setattr("manamap.config.FLEET_ANALYTICS_PATH", tmp_path / "fleet" / "analytics.json")
    monkeypatch.setattr(fa, "ARTIFACT", tmp_path / "fleet" / "analytics.json")
    _deck(decks, "alpha", target=3, kill8=0.6, missed=0.1, dti=40, feel=[{"power": 4}, {"power": 2, "fun": 5}])
    _deck(decks, "bravo", target=None, floor=3, kill8=0.4, missed=0.3)
    _deck(decks, "charlie", target=3, floor=1, kill8=0.5, missed=0.2, dti=30)
    _deck(decks, "delta", target=4, floor=4, kill8=0.9, missed=0.15)
    _deck(decks, "echo", modern=True)
    _deck(decks, "foxtrot", target=3, kill8=0.95, missed=0.05, stage="bench")   # a bench deck: a row, never a member
    return decks


# ── Rows ─────────────────────────────────────────────────────────────────


def test_the_group_key_is_the_target_then_the_floor_and_a_modern_deck_is_not_modelled(fleet):
    doc = fa.build()
    by = {r["slug"]: r for r in doc["decks"]}
    assert by["alpha"]["bracket"]["group"] == "B3" and by["alpha"]["bracket"]["group_basis"] == "target"
    assert by["bravo"]["bracket"]["group"] == "B3" and by["bravo"]["bracket"]["group_basis"] == "floor"
    assert by["charlie"]["bracket"]["group"] == "B3", "a floor-1 deck declared at 3 competes at 3"
    assert by["delta"]["bracket"]["group"] == "B4"
    assert by["echo"]["bracket"]["group"] == "unbracketed"
    assert by["echo"]["format"] == "Modern" and fa.is_absent(by["echo"]["benchmark"])
    assert "not modelled for Modern" in by["echo"]["benchmark"]["absent_because"]


def test_the_fleet_is_the_sleeved_decks(fleet):
    """A bench deck keeps its row and its figures, but joins no group, takes no
    placement, and never moves a sleeved deck's band — even when it is the
    fastest list on disk."""
    doc = fa.build()
    by = {r["slug"]: r for r in doc["decks"]}
    assert by["foxtrot"]["member"] is False and by["foxtrot"]["placements"] == {}
    assert by["alpha"]["member"] is True
    assert "foxtrot" not in doc["groups"]["B3"]["members"]
    assert doc["groups"]["B3"]["bands"]["kill_by_turn_8"]["max"] == 0.6, "foxtrot's 0.95 is not in the band"
    assert "sleeved" in " ".join(doc["limits"]).lower()
    errors, _ = vfa.validate(doc)
    assert errors == []
    by["alpha"]["member"] = False
    assert any("member: true" in e for e in vfa.validate(doc)[0])


def test_absent_is_absent_never_zero(fleet):
    doc = fa.build()
    by = {r["slug"]: r for r in doc["decks"]}
    assert fa.is_absent(by["bravo"]["dti"]) and fa.is_absent(by["bravo"]["survey"])
    assert fa.is_absent(by["bravo"]["price"]) and fa.is_absent(by["bravo"]["table"])
    assert fa.is_absent(by["bravo"]["diagnostic"])
    assert by["bravo"]["rating"] is None
    assert not any(isinstance(v, (int, float)) and v == 0
                   for k, v in by["bravo"]["dti"].items() if k != "source")


def test_every_benchmark_rate_carries_a_wilson_interval_with_its_n(fleet):
    doc = fa.build()
    k8 = next(r for r in doc["decks"] if r["slug"] == "alpha")["benchmark"]["kill_by_turn_8"]
    assert k8["value"] == 0.6 and k8["n"] == 1000
    assert k8["ci95"][0] < 0.6 < k8["ci95"][1] and "Wilson" in k8["basis"]


def test_the_survey_is_authored_with_n_min_max_beside_the_mean(fleet):
    doc = fa.build()
    sv = next(r for r in doc["decks"] if r["slug"] == "alpha")["survey"]
    assert sv["tier"] == "authored" and sv["games_surveyed"] == 2
    assert sv["power"] == {"mean": 3.0, "n": 2, "min": 2, "max": 4}
    assert sv["fun"] == {"mean": 5.0, "n": 1, "min": 5, "max": 5}
    assert "resilience" not in sv


# ── Bands and placements ─────────────────────────────────────────────────


def test_bands_are_min_median_max_at_three_and_absent_below(fleet):
    doc = fa.build()
    b3 = doc["groups"]["B3"]
    assert b3["members"] == ["alpha", "bravo", "charlie"] and b3["n"] == 3
    assert b3["bands"]["kill_by_turn_8"] == {"min": 0.4, "median": 0.5, "max": 0.6, "n": 3}
    assert fa.is_absent(b3["bands"]["dti_score"]), "only two B3 decks carry a DTI"
    b4 = doc["groups"]["B4"]
    assert b4["n"] == 1 and all(fa.is_absent(v) for v in b4["bands"].values())
    assert "B4 has 1 member(s)" in " ".join(doc["limits"])


def test_placements_use_one_vocabulary_and_say_no_band(fleet):
    doc = fa.build()
    by = {r["slug"]: r for r in doc["decks"]}
    assert by["alpha"]["placements"]["group"]["kill_by_turn_8"] == "above the bracket median"
    assert by["bravo"]["placements"]["group"]["kill_by_turn_8"] == "below the bracket median"
    assert by["alpha"]["placements"]["group"]["dti_score"] == "no band (n=2)"
    assert by["delta"]["placements"]["group"]["kill_by_turn_8"] == "no band (n=1)"
    # delta is a fleet member, so its own 0.9 is the fleet band's max: inside, not beyond.
    assert by["delta"]["placements"]["fleet"]["kill_by_turn_8"] == "above the fleet median"
    assert by["bravo"]["placements"]["fleet"]["missed_land_drop_rate"] == "worse than any tracked deck" \
        or by["bravo"]["placements"]["fleet"]["missed_land_drop_rate"] == "above the fleet median"
    assert by["echo"]["placements"] == {}


def test_place_in_keeps_the_dossiers_words_and_flips_for_higher_is_better():
    band = {"min": 0.1, "median": 0.2, "max": 0.3}
    assert diagnostic.place_in(0.05, band) == "better than any tracked deck"
    assert diagnostic.place_in(0.35, band) == "worse than any tracked deck"
    assert diagnostic.place_in(0.25, band) == "above the fleet median"
    assert diagnostic.place_in(0.05, band, higher_is_better=True) == "worse than any tracked deck"
    assert diagnostic.place_in(0.25, band, higher_is_better=True, scope="bracket") == "above the bracket median"
    assert diagnostic._place(0.001, "stall_two_in_a_row") == "better than any tracked deck"
    assert diagnostic.place_in(None, band) is None and diagnostic.place_in(0.2, None) is None


# ── Comparisons ──────────────────────────────────────────────────────────


def test_spearman_is_refused_below_min_decks_and_measured_above(fleet):
    doc = fa.build()
    sp = doc["comparisons"]["spearman"]["dti_vs_kill_by_8"]
    assert sp["verdict"] == "NOT ANSWERABLE" and sp["n"] == 2
    assert str(calibrate.MIN_DECKS) in sp["why"]
    pairs = [(i, i * 2.0) for i in range(calibrate.MIN_DECKS)]
    cell = fa._spearman_or_refusal(pairs, "x", "more")
    assert cell["verdict"] == "measured" and cell["rho"] == 1.0 and cell["significant"] is True


def test_curves_come_from_the_benchmark_and_the_grade_grid_counts(fleet):
    doc = fa.build()
    c = doc["comparisons"]
    assert set(c["kill_curves"]) == {"alpha", "bravo", "charlie", "delta", "foxtrot"}
    assert c["kill_curves"]["alpha"]["10"] == 0.6
    assert c["benchmark_grades"]["distribution"]["R1"]["B"] == 2
    assert [x["slug"] for x in c["floor_vs_dti"]] == ["alpha", "charlie"]
    assert c["dti_history"]["alpha"][0]["current_list"] is True


# ── The document ─────────────────────────────────────────────────────────


def test_build_is_byte_identical_and_order_independent(fleet, monkeypatch):
    a = fa.dumps(fa.build())
    monkeypatch.setattr(fa, "live_slugs", lambda: sorted(["echo", "alpha", "delta", "bravo", "charlie", "foxtrot"]))
    assert fa.dumps(fa.build()) == a
    p = fa.write()
    assert p.read_text() == a


def test_write_then_validate_passes_and_a_moved_list_only_warns(fleet):
    fa.write()
    doc = fa.load()
    slugs = set(fa.live_slugs())
    errors, warnings = vfa.validate(doc, current_shas={s: SHA[s] for s in slugs}, deck_dirs=slugs)
    assert (errors, warnings) == ([], [])
    moved = dict(SHA)
    moved["alpha"] = "f" * 64
    errors, warnings = vfa.validate(doc, current_shas=moved, deck_dirs=slugs)
    assert errors == [] and warnings == ["stale: alpha list moved since this was built — "
                                         "`manamap pilot fleet-analytics --write`"]


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d["decks"][0].__setitem__("score", 71.5), "bare composite number"),
    (lambda d: d["decks"][0]["bracket"].__setitem__("rank", 2), "bare composite number"),
    (lambda d: d["decks"][0].__setitem__("rating", {"score": 1}), "rating must be null"),
    (lambda d: d["decks"][0]["bracket"].__setitem__("group", "B4"), "does not follow"),
    (lambda d: d["decks"][0]["benchmark"]["kill_by_turn_8"].__setitem__("value", 0.99), "outside its ci95"),
    (lambda d: d["decks"][0]["benchmark"]["kill_by_turn_8"].pop("n"), "interval without its n"),
    (lambda d: d["decks"][0]["benchmark"].__setitem__("kill_by_turn_8", {"value": 2.0, "source": "x"}), "rate outside"),
    (lambda d: d["decks"][0]["survey"].__setitem__("tier", "measured"), "tier 'authored'"),
    (lambda d: d["groups"]["B3"]["bands"]["kill_by_turn_8"].__setitem__("max", 0.9), "do not re-add"),
    (lambda d: d["groups"]["B3"]["members"].pop(), "members"),
    (lambda d: d["groups"].pop("B4"), "do not match"),
    (lambda d: d["meta"].__setitem__("no_composite", "a score would be nice"), "refusal"),
    (lambda d: d["comparisons"]["spearman"]["dti_vs_kill_by_8"].update(
        {"verdict": "measured", "n": 2, "rho": 1.0}), "below the refusal line"),
    (lambda d: d["decks"].reverse(), "not sorted"),
    (lambda d: d.__setitem__("schema_version", 2), "schema_version"),
], ids=lambda x: x if isinstance(x, str) else "")
def test_validator_negatives(fleet, mutate, needle):
    doc = fa.build()
    mutate(doc)
    errors, _ = vfa.validate(doc)
    assert errors and any(needle in e for e in errors), errors


def test_main_absent_is_absent(fleet, capsys):
    vfa.main(None)
    assert "absent means absent" in capsys.readouterr().out
    fa.write()
    vfa.main(None)
    assert capsys.readouterr().out.startswith("OK")


def test_the_fleet_gate_and_the_regen_stage_are_registered():
    from manamap.pilot import deck_status, regen
    assert deck_status.FLEET_VALIDATED["fleet/analytics.json"] == "manamap.pilot.validate_fleet_analytics"
    assert "fleet-analytics" in regen.STAGE_NAMES
    assert not any(s[0] == "fleet-analytics" for s in regen.STAGES), "a fleet stage is not a per-deck row"
    assert regen.FLEET_STAGES[0][1] == "fleet/analytics.json"
