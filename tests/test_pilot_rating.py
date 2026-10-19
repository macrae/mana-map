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
    audits = {"alpha": _audit_for({"mana-base": 37, "ramp": 11, "consistency": 0.9, "interaction-breadth": 4}),
              "bravo": _audit_for({"mana-base": 33, "ramp": 8, "consistency": 0.6, "interaction-breadth": 2}),
              "charlie": _audit_for({"mana-base": 37, "ramp": 11, "consistency": 0.8, "interaction-breadth": 3}),
              "delta": _audit_for({"mana-base": 37})}
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


def test_the_spec_sha_moves_when_a_weight_moves(monkeypatch):
    before = rating.spec_sha()
    monkeypatch.setitem(rating.WEIGHTS, "speed", 0.16)
    assert rating.spec_sha() != before


# ── The skeleton ─────────────────────────────────────────────────────────


def test_a_skeleton_is_labelled_coaching_withholds_what_it_lacks_and_renormalises(fleet):
    doc = rating.build("alpha")
    assert doc["tier"] == "★" and doc["label"] == rating.RATING_LABEL.format(n=3)
    assert doc["bracket"] == {"key": 3, "source": "target", "floor": 3, "floor_name": "Upgraded", "within_target": True}
    names = [c["name"] for c in doc["components"]]
    assert names == ["speed", "mana", "resilience", "construction"]
    withheld = {w["component"]: w["reason"] for w in doc["withheld"]}
    assert set(withheld) == {"table_record", "pilot_feel", "forge", "judgment"}
    assert "5 needed" in withheld["table_record"] and "3 needed" in withheld["pilot_feel"]
    assert abs(sum(c["weight_effective"] for c in doc["components"]) - 1.0) < 1e-3
    assert doc["judgment"] is None and doc["score"] == doc["code_score"]
    assert doc["grade"] == rating.grade_of(doc["score"])
    assert doc["rank"]["of"] == 3 and doc["rank"]["group"][0] == "alpha"
    assert "no significance claim" in " ".join(doc["limits"])
    # Never a zero for an absent component: the withheld ones are not in components.
    assert all(c["sub_score"] > 0 for c in doc["components"])


def test_speed_reads_placement_and_the_brackets_anchor_with_its_interval(fleet):
    doc = rating.build("bravo")
    speed = next(c for c in doc["components"] if c["name"] == "speed")
    assert speed["placement"] == 0.0, "the slowest of three"
    assert speed["anchor"] == 80.0, "kill by 8 at 0.4 / 0.5"
    assert speed["sub_score"] == 40.0
    assert speed["sub_ci"][0] < 40.0 < speed["sub_ci"][1], "the anchor carries the Wilson interval"
    assert speed["inputs"][0]["figure"] == "kill_by_turn_8" and speed["inputs"][0]["n"] == 1000
    assert speed["citations"][0]["ref"] == "strategy:deckbuilding.power-level"


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
    hi = rating._finish(json.loads(json.dumps(doc)), dict(j))["score"]
    j["sub_score"] = 0
    lo = rating._finish(json.loads(json.dumps(doc)), dict(j))["score"]
    assert hi - lo <= 100 * rating.WEIGHTS["judgment"] / (rating.WEIGHTS["judgment"] + 0.55) + 0.1
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
