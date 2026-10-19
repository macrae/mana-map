"""The rating scorecard: ★ coaching over ◆ inputs, within a bracket.

Driven on a sandbox fleet (the fleet-analytics fixture shape) with the audit
monkeypatched, so every rule is proven on known numbers: the grade bands,
withheld-never-zero with renormalised weights, placement withheld below three,
the monotone band score, the judgment carried forward only against the same
list and spec, byte-identical writes, and the spec sha moving with a weight.
"""

import json

import pytest

from manamap import config
from manamap.pilot import fleet_analytics as fa, rating, validate_rating as vr

from test_pilot_fleet_analytics import SHA, _deck  # noqa: F401 — the fixture helpers


def _audit_for(values):
    """A fake `deck_audit.analyze`: axes with the real bands and the given values."""
    bands = {"mana-base": (36, 38), "ramp": (10, 12), "card-advantage": (10, None), "interaction": (8, None),
             "sweepers": (1, 3), "protection": (5, 7), "threat-density": (3, None),
             "consistency": (0.74, None), "interaction-breadth": (5, None)}
    axes = []
    for name, (lo, hi) in bands.items():
        if name in values:
            axes.append({"axis": name, "measured": {"value": values[name], "unit": "cards", "how": "x"},
                         "target": {"low": lo, "high": hi, "source": "strategy:deckbuilding.ratios", "quote": "q"},
                         "verdict": "at"})
    return {"axes": axes, "archetype": {}}


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    decks = tmp_path / "decks"
    decks.mkdir()
    monkeypatch.setattr("manamap.config.DECKS_DIR", decks)
    monkeypatch.setattr("manamap.config.FLEET_DIR", tmp_path / "fleet")
    monkeypatch.setattr("manamap.config.FLEET_ANALYTICS_PATH", tmp_path / "fleet" / "analytics.json")
    monkeypatch.setattr(fa, "ARTIFACT", tmp_path / "fleet" / "analytics.json")
    _deck(decks, "alpha", target=3, kill8=0.6, missed=0.1)
    _deck(decks, "bravo", target=3, kill8=0.4, missed=0.3)
    _deck(decks, "charlie", target=3, kill8=0.5, missed=0.2)
    _deck(decks, "delta", target=4, kill8=0.9, missed=0.15)
    _deck(decks, "foxtrot", target=3, kill8=0.95, missed=0.05, stage="bench")
    audits = {"alpha": _audit_for({"mana-base": 37, "ramp": 11, "consistency": 0.9, "interaction-breadth": 4}),
              "bravo": _audit_for({"mana-base": 33, "ramp": 8, "consistency": 0.6, "interaction-breadth": 2}),
              "charlie": _audit_for({"mana-base": 37, "ramp": 11, "consistency": 0.8, "interaction-breadth": 3}),
              "delta": _audit_for({"mana-base": 37}), "foxtrot": _audit_for({"mana-base": 37})}
    monkeypatch.setattr(rating.deck_audit, "analyze", lambda slug: audits[slug])
    fa.write()
    return decks


# ── Primitives ───────────────────────────────────────────────────────────


def test_grade_bands_at_every_boundary():
    assert [rating.grade_of(x) for x in (100, 80, 79.9, 65, 64.9, 50, 49.9, 35, 34.9, 0)] == \
        ["A", "A", "B", "B", "C", "C", "D", "D", "F", "F"]


def test_band_score_is_100_inside_and_falls_monotonically_outside():
    assert rating.band_score(37, 36, 38) == 100.0
    assert rating.band_score(36, 36, 38) == 100.0 and rating.band_score(38, 36, 38) == 100.0
    assert rating.band_score(35, 36, 38) == 50.0           # one below a span of 2
    assert rating.band_score(34, 36, 38) == 0.0
    assert rating.band_score(12, 10, None) == 100.0        # floor only: over is fine
    assert rating.band_score(5, 10, None) == 50.0
    assert rating.band_score(0.9, 0.74, None) == 100.0
    xs = [rating.band_score(v, 10, 12) for v in (6, 8, 10, 11, 12, 14, 16)]
    assert xs == sorted(xs[:3]) + [100.0, 100.0] + sorted(xs[5:], reverse=True)
    assert rating.band_score(None, 1, 2) is None and rating.band_score(1, None, None) is None


def test_placement_is_a_percentile_with_ties_averaged_and_withheld_below_three():
    vals = {"a": 0.6, "b": 0.4, "c": 0.5}
    assert rating.placement(vals, "a", True)[0] == 100.0
    assert rating.placement(vals, "b", True)[0] == 0.0
    assert rating.placement(vals, "c", True)[0] == 50.0
    assert rating.placement(vals, "b", False)[0] == 100.0, "lower is better flips the ends"
    assert rating.placement({"a": 0.5, "b": 0.5, "c": 0.4}, "a", True)[0] == 75.0, "a tie averages"
    assert rating.placement({"a": 0.6, "b": 0.4}, "a", True) == (None, None)
    assert rating.placement({"a": 0.6, "b": None, "c": None}, "a", True) == (None, None)


def test_the_spec_sha_moves_when_a_weight_or_a_prior_moves(monkeypatch):
    before = rating.spec_sha()
    monkeypatch.setitem(rating.WEIGHTS, "speed", 0.16)
    assert rating.spec_sha() != before
    monkeypatch.setitem(rating.WEIGHTS, "speed", 0.12)
    assert rating.spec_sha() == before
    monkeypatch.setattr(rating, "WEIGHT_CONCENTRATION", 21.0)
    assert rating.spec_sha() != before, "κ is a prior that moves every number"


def test_placement_weight_shrinks_toward_the_anchor_by_group_size():
    assert abs(rating.placement_weight(3) - 1 / 3) < 1e-12
    assert abs(rating.placement_weight(5) - 1 / 2) < 1e-12
    assert abs(rating.placement_weight(9) - 2 / 3) < 1e-12
    assert rating.placement_weight(1) == 0.0 and rating.placement_weight(None) == 0.0


# ── The skeleton ─────────────────────────────────────────────────────────


def test_a_skeleton_is_labelled_coaching_withholds_what_it_lacks_and_renormalises(fleet):
    doc = rating.build("alpha")
    assert doc["tier"] == "★" and doc["label"] == rating.RATING_LABEL.format(n=3)
    assert doc["bracket"] == {"key": 3, "source": "target", "floor": 3, "floor_name": "Upgraded", "within_target": True}
    names = [c["name"] for c in doc["components"]]
    assert names == ["speed", "mana", "resilience", "construction"]
    withheld = {w["component"]: w["reason"] for w in doc["withheld"]}
    assert set(withheld) == {"dti", "table_record", "pilot_feel", "forge", "judgment"}
    assert "5 needed" in withheld["table_record"] and "3 needed" in withheld["pilot_feel"]
    assert "deckcheck.json" in withheld["dti"]
    assert abs(sum(c["weight_effective"] for c in doc["components"]) - 1.0) < 1e-3
    assert doc["judgment"] is None and doc["score_point"] == doc["code_score"]
    assert doc["grade"] == rating.grade_of(doc["score"]), "the grade is the band of the MC median"
    assert doc["score_ci"][0] < doc["score"] < doc["score_ci"][1], "the interval has width now"
    assert abs(sum(doc["grade_probabilities"].values()) - 1.0) < 1e-6
    assert abs(sum(doc["rank"]["acceptability"].values()) - 1.0) < 1e-6
    assert doc["uncertainty"]["k"] == rating.RATING_MC_K and doc["uncertainty"]["seed"] == rating.RATING_MC_SEED
    assert doc["concordance"]["level"] in ("concordant", "mixed", "divergent")
    assert doc["rank"]["of"] == 3 and doc["rank"]["group"][0] == "alpha"
    assert "no significance claim" in " ".join(doc["limits"])
    # Never a zero for an absent component: the withheld ones are not in components.
    assert all(c["sub_score"] > 0 for c in doc["components"])


def test_speed_reads_placement_and_the_brackets_anchor_with_its_interval(fleet):
    doc = rating.build("bravo")
    speed = next(c for c in doc["components"] if c["name"] == "speed")
    assert speed["placement"] == 0.0, "the slowest of three"
    assert speed["anchor"] == 80.0, "kill by 8 at 0.4 / 0.5"
    assert speed["sub_score"] == 53.33, "λ(3) = 1/3 of the placement, 2/3 of the anchor"
    assert speed["weight_validity"] == 1.0 and "speed validity unchecked" in " ".join(doc["limits"])
    assert speed["sub_ci"][0] < 53.33 < speed["sub_ci"][1], "the placement bin and the Beta draw both widen it"
    assert speed["inputs"][0]["figure"] == "kill_by_turn_8" and speed["inputs"][0]["n"] == 1000
    assert speed["citations"][0]["ref"] == "strategy:deckbuilding.power-level"


def test_a_bench_deck_is_refused_and_never_in_a_group(fleet):
    """The fleet is the sleeved decks: a bench deck has no rank to hold."""
    with pytest.raises(SystemExit, match="not a fleet member"):
        rating.build("foxtrot")
    assert rating.build("alpha")["rank"]["of"] == 3
    from manamap.pilot import agent_cache as ac
    assert not ac._rating_applicable("foxtrot"), "no paper lock, so no routine"


def test_a_group_below_three_scores_on_anchors_alone(fleet):
    doc = rating.build("delta")
    assert doc["rank"]["of"] == 1
    for c in doc["components"]:
        assert c["placement"] is None and "placement withheld" in c["method"] or "anchor only" in c["method"]


def test_write_is_byte_identical_and_a_moved_list_drops_the_judgment_with_its_reason(fleet):
    doc = rating.build("alpha")
    p = rating.write("alpha", doc)
    first = p.read_text()
    assert rating.write("alpha", rating.build("alpha")).read_text() == first
    # A judgment against this list and spec is carried; against another list it is withheld.
    judged = json.loads(first)
    judged["judgment"] = {"sub_score": 70, "argument": "fine", "citations": [],
                          "against": {"decklist_sha256": doc["decklist_sha256"], "spec_sha": doc["model_version"]}}
    judged["iterations"] = 1
    carried = rating.build("alpha", previous=judged)
    assert carried["judgment"]["sub_score"] == 70 and carried["iterations"] == 1
    assert not any(w["component"] == "judgment" for w in carried["withheld"])
    assert carried["score"] != carried["code_score"]
    judged["judgment"]["against"]["decklist_sha256"] = "f" * 64
    dropped = rating.build("alpha", previous=judged)
    assert dropped["judgment"] is None
    assert any(w["component"] == "judgment" and "written against ffffffffffff" in w["reason"]
               for w in dropped["withheld"])


def test_the_composite_is_the_weighted_mean_and_the_judgment_is_bounded_by_its_weight(fleet):
    doc = rating.build("alpha")
    code = doc["code_score"]
    j = {"sub_score": 100, "argument": "x", "citations": [],
         "against": {"decklist_sha256": doc["decklist_sha256"], "spec_sha": doc["model_version"]}}
    hi = rating._finish(json.loads(json.dumps(doc)), dict(j))["score_point"]
    j["sub_score"] = 0
    lo = rating._finish(json.loads(json.dumps(doc)), dict(j))["score_point"]
    code_total = sum(rating.WEIGHTS[c["name"]] * c["weight_validity"] for c in doc["components"])
    assert hi - lo <= 100 * rating.WEIGHTS["judgment"] / (rating.WEIGHTS["judgment"] + code_total) + 0.1
    assert lo < code < hi


def test_the_leaderboard_groups_by_bracket_and_calibrate_refuses(fleet):
    board = rating.leaderboard()
    assert set(board) == {"B3", "B4"} and [d["slug"] for d in board["B3"]][0] == "alpha"
    cal = rating.calibrate_rating()
    assert all(c["verdict"] == "NOT ANSWERABLE" for c in cal["cells"].values())
    assert cal["min_decks"] == 10


# ── The gate ─────────────────────────────────────────────────────────────


def _judged(fleet):
    doc = rating.build("alpha")
    doc["withheld"] = [w for w in doc["withheld"] if w["component"] != "judgment"]
    j = {"sub_score": 70, "weight": rating.WEIGHTS["judgment"],
         "argument": "Alpha kills by turn 8 in 60.0% of games, the fastest of its three, and its mana base of 37 lands sits in the band; the record is thin.",
         "citations": [{"kind": "skeleton", "ref": "skeleton:speed", "quote": "kill_by_turn_8"},
                       {"kind": "log", "ref": "log.jsonl", "quote": "no games"}],
         "against": {"decklist_sha256": doc["decklist_sha256"], "spec_sha": doc["model_version"]}}
    rating._finish(doc, j)
    doc = rating.ranked(doc)
    doc["iterations"] = 1
    return doc


def test_a_sound_judged_document_passes_and_each_bug_is_caught(fleet):
    doc = _judged(fleet)
    assert vr.validate("alpha", doc, fresh=rating.build("alpha", previous=doc))[0] == []

    def errs(mutate):
        d = json.loads(json.dumps(doc))
        mutate(d)
        return vr.validate("alpha", d, fresh=rating.build("alpha", previous=d))[0]

    assert any("sub_score" in e for e in errs(lambda d: d["judgment"].__setitem__("sub_score", 101)))
    assert any("sub_score" in e for e in errs(lambda d: d["judgment"].__setitem__("sub_score", 70.5)))
    assert any("weight" in e for e in errs(lambda d: d["components"][0].__setitem__("weight", 0.5)))
    assert any("differ from a fresh computation" in e for e in errs(lambda d: d["components"][0].__setitem__("sub_score", 99.0)))
    assert any("grade" in e for e in errs(lambda d: d.__setitem__("grade", "F")))
    assert any("tier" in e for e in errs(lambda d: d.__setitem__("tier", "◆")))
    assert any("label" in e for e in errs(lambda d: d.__setitem__("label", "A rating.")))
    assert any("appears in no cited material" in e for e in errs(
        lambda d: d["judgment"].__setitem__("argument", d["judgment"]["argument"] + " It wins 97.3% of games.")))
    assert any("two citations" in e for e in errs(lambda d: d["judgment"].__setitem__("citations", d["judgment"]["citations"][:1])))
    assert any("not in the log" in e for e in errs(
        lambda d: d["judgment"]["citations"].append({"kind": "log", "ref": "log:009", "quote": "x"})))
    assert any("withheld or unknown" in e for e in errs(
        lambda d: d["judgment"]["citations"].append({"kind": "skeleton", "ref": "skeleton:forge", "quote": "x"})))
    assert any("does not name this list" in e for e in errs(
        lambda d: d["judgment"]["against"].__setitem__("decklist_sha256", "f" * 64)))
    assert any("pass beside a finding" in e for e in errs(
        lambda d: d.__setitem__("verifier", {"verdict": "pass", "findings": [{"where": "judgment", "status": "over-claimed", "note": "n"}]})))
    assert any("closed set" in e for e in errs(
        lambda d: d.__setitem__("verifier", {"verdict": "fail", "findings": [{"where": "judgment", "status": "meh", "note": "n"}]})))
    assert any("settled_by" in e for e in errs(lambda d: d.__setitem__("open_questions", [{"question": "q", "settled_by": "vibes"}])))


def test_a_stale_document_is_form_checked_and_said_so(fleet, capsys):
    doc = _judged(fleet)
    rating.write("alpha", doc)
    (config.DECKS_DIR / "alpha" / "cards.json").write_text(json.dumps(
        {"deck": "alpha", "decklist_sha256": "e" * 64, "cards": [{"name": "Zada, Hedron Grinder", "is_commander": True}]}))
    assert "the list moved" in vr.staleness("alpha", doc)
    vr.main(type("A", (), {"slug": "alpha"})())
    out = capsys.readouterr().out
    assert out.startswith("STALE alpha") and "(STALE)" in out


def test_the_gate_is_registered_and_the_install_refuses_the_routine():
    from manamap.pilot import deck_status, install_agent
    assert deck_status.VALIDATED["rating.json"] == "manamap.pilot.validate_rating"
    assert "deck-rating" in install_agent.MERGED_ELSEWHERE


# ── The cache routine ────────────────────────────────────────────────────


def test_the_rating_routine_fingerprints_the_fleet_and_the_scorecard_but_not_the_judgment(fleet, monkeypatch):
    """A rebuilt fleet artifact or an edited weight must MISS the routine on
    every deck; a merged judgment must NOT invalidate its own routine (the
    agent-owned keys are excluded), or the loop could never record a HIT."""
    from manamap.pilot import agent_cache as ac
    rating.write("alpha", rating.build("alpha"))
    spec = config.AGENT_ROUTINES["deck-rating"]
    base = ac.resolve_inputs("alpha", spec)
    doc = rating.build("alpha")
    doc["judgment"] = {"sub_score": 70, "argument": "x", "citations": [],
                       "against": {"decklist_sha256": doc["decklist_sha256"], "spec_sha": doc["model_version"]}}
    doc["iterations"] = 1
    rating.write("alpha", doc)
    assert ac.resolve_inputs("alpha", spec) == base, "the judgment is excluded from its own fingerprint"
    fa.ARTIFACT.write_text(fa.ARTIFACT.read_text() + "\n")
    assert ac.resolve_inputs("alpha", spec) != base, "the fleet artifact is an input"


def test_the_routine_is_applicable_only_to_a_sleeved_bracketed_commander_deck(fleet):
    from manamap.pilot import agent_cache as ac
    assert not ac._rating_applicable("alpha"), "no paper lock yet: the fleet is the sleeved decks"
    (config.DECKS_DIR / "alpha" / "deck_versions.json").write_text(json.dumps(
        {"slug": "alpha", "paper": {"version": 1, "sha": "abc", "decklist_sha256": SHA["alpha"],
                                    "built_at": "2026-10-01", "note": ""}, "tags": {}}))
    assert ac._rating_applicable("alpha")
    (config.DECKS_DIR / "alpha" / "bracket_report.json").unlink()
    assert not ac._rating_applicable("alpha")
    assert not ac._rating_applicable("nope")


# ── Version 2: the prior, the validity rule, the DTI, the concordance ─────


def test_a_win_the_goldfish_cannot_see_halves_speeds_weight_and_says_which_card(fleet):
    d = config.DECKS_DIR / "alpha"
    cards = json.loads((d / "cards.json").read_text())
    cards["cards"].append({"name": "Approach of the Second Sun", "type_line": "Sorcery",
                           "oracle_text": "If this spell was cast from your hand and you've cast another spell "
                                          "named Approach of the Second Sun this game, you win the game."})
    (d / "cards.json").write_text(json.dumps(cards))
    (d / "engine.json").write_text(json.dumps({"slug": "alpha", "thesis": "t", "stages": [
        {"stage": "wincon", "label": "w", "cards": ["Approach of the Second Sun"], "evidence": [], "what_it_does": ""}]}))
    assert rating.alt_win_cards("alpha") == (["Approach of the Second Sun"], None)
    doc = rating.build("alpha")
    speed = next(c for c in doc["components"] if c["name"] == "speed")
    assert speed["weight_validity"] == rating.SPEED_VALIDITY_ALT_WIN
    assert speed["weight"] == rating.WEIGHTS["speed"], "the prior weight is unchanged; the multiplier is separate"
    assert any("cannot see Approach of the Second Sun" in l for l in doc["limits"])
    eff = {c["name"]: c["weight_effective"] for c in doc["components"]}
    assert eff["speed"] < eff["mana"] * (rating.WEIGHTS["speed"] / rating.WEIGHTS["mana"]), "speed earns less than its prior share"
    # Without the alt-win card in the engine's wincon stage the rule does not fire.
    (d / "engine.json").write_text(json.dumps({"slug": "alpha", "thesis": "t", "stages": [
        {"stage": "wincon", "label": "w", "cards": ["Zada, Hedron Grinder"], "evidence": [], "what_it_does": ""}]}))
    assert rating.alt_win_cards("alpha") == ([], None)
    assert next(c for c in rating.build("alpha")["components"] if c["name"] == "speed")["weight_validity"] == 1.0


def test_the_dti_places_on_the_score_anchors_on_the_rubric_and_is_withheld_when_stale(fleet):
    for slug, score in (("alpha", 40), ("bravo", 30), ("charlie", 35)):
        d = config.DECKS_DIR / slug
        import shutil
        shutil.rmtree(d)
        from test_pilot_fleet_analytics import _deck as mk
        mk(config.DECKS_DIR, slug, target=3, kill8={"alpha": 0.6, "bravo": 0.4, "charlie": 0.5}[slug],
           missed={"alpha": 0.1, "bravo": 0.3, "charlie": 0.2}[slug], dti=score)
    fa.write()
    doc = rating.build("alpha")
    dti = next(c for c in doc["components"] if c["name"] == "dti")
    assert dti["placement"] == 100.0, "40 is the highest of three"
    assert dti["anchor"] == 50.0, "twelve B grades anchor at 50, not 100·40/96"
    assert dti["sub_score"] == round(100.0 / 3 + 50.0 * 2 / 3, 2)
    assert "declared prior" in dti["method"] and dti["inputs"][1]["figure"] == "dti.rubric_mean"
    assert not any(w["component"] == "dti" for w in doc["withheld"])
    # A second reading on the same list: the sd comes from the readings.
    p = config.DECKS_DIR / "alpha" / "deckcheck.json"
    dc = json.loads(p.read_text())
    second = json.loads(json.dumps(dc["readings"][0]))
    second["benchmarks"]["S2"] = "A"
    dc["readings"].append(second)
    p.write_text(json.dumps(dc))
    dti2 = next(c for c in rating.build("alpha")["components"] if c["name"] == "dti")
    assert "sd of the mean over 2 readings" in dti2["method"] and dti2["inputs"][1]["n"] == 2
    # A reading on another list is withheld, never scored.
    for r in dc["readings"]:
        r["decklist_sha256"] = "e" * 64
    p.write_text(json.dumps(dc))
    fa.write()
    doc3 = rating.build("alpha")
    assert not any(c["name"] == "dti" for c in doc3["components"])
    assert any(w["component"] == "dti" and "another list" in w["reason"] for w in doc3["withheld"])


def test_the_flip_share_is_exact_and_leave_one_out_renormalises():
    from manamap.pilot import rating_mc as mc
    entries = [("speed", 0.12, 20.0), ("mana", 0.08, 70.0), ("construction", 0.13, 90.0)]
    sens = mc.sensitivity(entries, rating.GRADES, rating.grade_of)
    total = sum(w for _n, w, _s in entries)
    point = sum(w * s for _n, w, s in entries) / total
    loo = sens["leave_one_out"]["speed"]
    assert loo["score"] == round((0.08 * 70 + 0.13 * 90) / 0.21, 1) and loo["grade"] == rating.grade_of(loo["score"])
    for f in sens["flips"]:
        name, w, s = next(e for e in entries if e[0] == f["component"])
        rest = [e for e in entries if e[0] != name]
        s_rest = sum(w2 * s2 for _n, w2, s2 in rest) / sum(w2 for _n, w2, _s in rest)
        u = f["share_at_flip"]
        assert abs((u * s + (1 - u) * s_rest) - f["boundary"]) < 0.06, "at the flip share the score meets the boundary"
        assert 0 < u < 1 and f["direction"] in ("raise", "lower")
        assert abs(f["share_now"] - w / total) < 1e-3
    assert rating.grade_of(point) == "C" and round(point, 1) == 59.7


def test_concordance_reads_the_spread_of_a_decks_ranks_and_names_the_outlier_side():
    from manamap.pilot import rating_mc as mc
    kw = dict(min_group=3, mixed_at=0.34, divergent_at=0.67, seed=1, exact_max=50_000, iterations=200)
    subs = {"a": {"speed": 90, "mana": 80, "dti": 10}, "b": {"speed": 50, "mana": 50, "dti": 50},
            "c": {"speed": 10, "mana": 20, "dti": 90}}
    ca = mc.concordance(subs, "a", **kw)
    assert ca["level"] == "divergent" and ca["spread"] == 1.0
    assert ca["instruments"] == {"dti": 3.0, "mana": 1.0, "speed": 1.0}
    assert ca["outliers"] == [{"instrument": "dti", "rank": 3.0, "side": "worse"}], "the smaller side is the outlier"
    assert ca["group"]["m"] == 3 and ca["group"]["n"] == 3 and ca["group"]["p_method"].startswith("exact")
    cb = mc.concordance(subs, "b", **kw)
    assert cb["level"] == "concordant" and cb["outliers"] == []
    mixed = {"a": {"speed": 90, "mana": 60}, "b": {"speed": 50, "mana": 80}, "c": {"speed": 10, "mana": 20}}
    assert mc.concordance(mixed, "a", **kw)["level"] == "mixed"
    small = mc.concordance({"a": {"speed": 1}, "b": {"speed": 2}}, "a", **kw)
    assert small["level"] == "withheld"
    # The judgment is ranked and counts toward W, but is never its own outlier: with
    # it in the sides, a three-against-three tie at n=3 listed every instrument.
    with_j = {m: dict(v, judgment={"a": 80, "b": 50, "c": 20}[m]) for m, v in subs.items()}
    cj = mc.concordance(with_j, "a", **kw)
    assert "judgment" in cj["instruments"] and cj["group"]["m"] == 4
    assert [o["instrument"] for o in cj["outliers"]] == ["dti"]


def test_a_divergent_deck_prints_a_grade_range_and_its_judgment_must_name_the_outlier(fleet):
    doc = _judged(fleet)
    doc["concordance"] = {"level": "divergent", "instruments": {"speed": 1.0, "dti": 3.0}, "of": 3, "spread": 1.0,
                          "outliers": [{"instrument": "dti", "rank": 3.0, "side": "worse"}], "group": None}
    doc["grade_range"] = ["C", "B"]
    errors = vr.validate("alpha", doc, fresh=None)[0]
    assert any("never addresses the outlier instrument 'dti'" in e for e in errors), errors
    assert any("grade_display" in e for e in errors), "a divergent deck prints a range"
    doc["grade_display"] = "C–B"
    doc["judgment"]["argument"] += " DeckCheck reads it lower than the goldfish does."
    assert not [e for e in vr.validate("alpha", doc, fresh=None)[0] if "outlier" in e or "grade_display" in e]
    # A skeleton citation of the instrument also counts as addressing it.
    doc["judgment"]["argument"] = _judged(fleet)["judgment"]["argument"]
    doc["judgment"]["citations"].append({"kind": "skeleton", "ref": "skeleton:dti", "quote": "DeckCheck Threat Index"})
    assert not [e for e in vr.validate("alpha", doc, fresh=None)[0] if "outlier" in e]


def test_the_group_simulation_is_the_same_from_every_members_seat(fleet):
    """alpha's score as computed while building bravo must equal alpha's own
    build: one seed per group, a fixed draw order, nothing else."""
    a = rating.build("alpha")
    b = rating.build("bravo")
    assert b["rank"]["scores"]["alpha"] == a["score"]
    assert b["rank"]["group"] == a["rank"]["group"]
    assert b["concordance"]["group"] == a["concordance"]["group"]
