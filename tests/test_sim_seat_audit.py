"""The seat audit (2026-10-21): can Forge's AI pilot every seat at a pod. Unit on a
synthetic pod with scripted cards; regression on sharknado's committed run at the
standard table; the `simulate --pod` gate."""

import json

import pytest

from manamap.sim import seat_audit

from conftest import requires_deck


def _scripts(monkeypatch, scripts):
    """Patch the installed-script reader: {name: text or None}."""
    from manamap.sim import forge_cards

    def script(name, installed_copy=False):
        return scripts.get(name)
    monkeypatch.setattr(forge_cards, "script", script)


def _seat(monkeypatch, cards):
    """Patch the seat's doc: a list of (name, type_line)."""
    monkeypatch.setattr(seat_audit, "_seat_doc",
                        lambda slug: {"cards": [{"name": n, "type_line": t} for n, t in cards]})
    monkeypatch.setattr(seat_audit, "_floors", lambda slug: [])


def test_static_names_the_flagged_and_the_unfound_and_lands_are_skipped(monkeypatch):
    _seat(monkeypatch, [("Flagged Commander", "Legendary Creature"), ("Fine Spell", "Sorcery"),
                        ("No Script", "Instant"), ("Darkwater Catacombs", "Land"),
                        ("Gated", "Sorcery")])
    _scripts(monkeypatch, {
        "Flagged Commander": "Name:Flagged Commander\nAI:RemoveDeck:All\nOracle:x",
        "Fine Spell": "Name:Fine Spell\nA:SP$ Draw | NumCards$ 1\nOracle:x",
        "Darkwater Catacombs": "Name:Darkwater Catacombs\nAI:RemoveDeck:All\nOracle:x",
        "Gated": "Name:Gated\nSVar:NeedsToPlay:Creature.IsCommander+YouCtrl\nOracle:x",
    })
    st = seat_audit.static_seat("x")
    assert st["flagged"] == ["Flagged Commander"], "a land's flag is harmless and not counted"
    assert st["unfound"] == ["No Script"]
    assert st["gated"] == {"Gated": ["NeedsToPlay:Creature.IsCommander+YouCtrl"]}
    assert st["nonland"] == 4 and st["clean"] is False
    verdict, why = seat_audit._verdict("x", st, None, ours=False)
    assert verdict == "NOT" and any("will not cast" in w for w in why) and any("without a script" in w for w in why)


def test_a_clean_static_seat_is_pilotable_and_the_dynamic_rules_apply_when_a_run_exists(monkeypatch):
    _seat(monkeypatch, [("Fine Spell", "Sorcery")])
    _scripts(monkeypatch, {"Fine Spell": "Name:Fine Spell\nOracle:x"})
    st = seat_audit.static_seat("x")
    assert st["clean"]
    assert seat_audit._verdict("x", st, None, ours=False) == ("PILOTABLE", [])
    dyn = {"commander_resolved_rate": 0.5, "engine": {"Fine Spell": {"cast": 0, "held": True, "never_cast": False}},
           "wheel_timing": {"wheels_per_game": 1.1, "with_commander": 0.47}}
    verdict, why = seat_audit._verdict("x", st, dyn, ours=True)
    assert verdict == "NOT"
    assert any("commander resolved in 50%" in w for w in why)
    assert any("engine card" in w and "Fine Spell" in w for w in why)
    assert any("47%" in w for w in why)
    # a floor is reported, never a failure
    st["floors"] = ["Fine Spell"]
    dyn2 = {"commander_resolved_rate": 0.9, "engine": dyn["engine"]}
    assert seat_audit._verdict("x", st, dyn2, ours=False) == ("PILOTABLE", [])


def test_validate_holds_the_table_to_its_seats():
    doc = {"pod": "p", "at": "2026-10-21", "since": "2026-10-21", "harness": {"overrides": "x"},
           "definition": {}, "limits": [],
           "seats": [{"slug": "a", "verdict": "PILOTABLE", "why": [],
                      "static": {"flagged": [], "unfound": [], "clean": True}}],
           "verdict": "PILOTABLE"}
    assert seat_audit.validate(doc) == []
    bad = json.loads(json.dumps(doc)); bad["verdict"] = "NOT"
    assert any("disagrees" in e for e in seat_audit.validate(bad))
    bad = json.loads(json.dumps(doc)); bad["seats"][0]["verdict"] = "NOT"
    assert any("no reason" in e for e in seat_audit.validate(bad))
    bad = json.loads(json.dumps(doc)); bad["seats"][0]["static"]["clean"] = False
    assert any("clean" in e for e in seat_audit.validate(bad))


def test_the_gate_refuses_without_an_audit_under_this_harness_and_anyway_stamps(monkeypatch, tmp_path):
    from manamap.sim import forge, pods
    monkeypatch.setattr(pods, "PODS_DIR", tmp_path)
    monkeypatch.setattr(forge, "card_overrides", lambda: {"sha": "ov-now"})
    with pytest.raises(SystemExit, match="no seat audit"):
        seat_audit.gate("p")
    got = seat_audit.gate("p", anyway=True)
    assert got["anyway"] and got["verdict"] == "none"
    (tmp_path / "p.audit.json").write_text(json.dumps(
        {"pod": "p", "at": "2026-10-21", "harness": {"overrides": "ov-old"}, "verdict": "PILOTABLE", "seats": []}))
    with pytest.raises(SystemExit, match="re-run"):
        seat_audit.gate("p")
    (tmp_path / "p.audit.json").write_text(json.dumps(
        {"pod": "p", "at": "2026-10-21", "harness": {"overrides": "ov-now"}, "verdict": "NOT",
         "seats": [{"slug": "araumi-edict", "verdict": "NOT", "why": ["Araumi flagged"]}]}))
    with pytest.raises(SystemExit, match="cannot pilot every seat"):
        seat_audit.gate("p")
    (tmp_path / "p.audit.json").write_text(json.dumps(
        {"pod": "p", "at": "2026-10-21", "harness": {"overrides": "ov-now"}, "verdict": "PILOTABLE", "seats": []}))
    assert seat_audit.gate("p") == {"verdict": "PILOTABLE", "at": "2026-10-21",
                                    "harness": {"overrides": "ov-now"}, "anyway": False}


@requires_deck
def test_the_standard_table_reads_every_seat_from_sharknados_run():
    """The committed V6 run (logs on disk): four seats, each with its commander
    rate and held list; our seat carries the wheel timing. Also the finding of
    2026-10-21 — the calibrated table's opponents carry flagged cards too."""
    from manamap.config import DECKS_DIR
    run = "sythis-enchantress-vs-jarad-graveyard-vs-abaddon-n200-a0874c33-s693221427-podExperimental-c600-ov8c347642-tl9473cf35"
    if not (DECKS_DIR / "sharknado" / "sim" / "logs" / run).is_dir():
        pytest.skip("the V6 logs are not on disk")
    doc = seat_audit.audit("standard-v3", run_id=run)
    assert [r["slug"] for r in doc["seats"]] == ["sharknado", "sythis-enchantress", "jarad-graveyard", "abaddon"]
    ours = doc["seats"][0]
    assert ours["ours"] and ours["dynamic"]["games"] == 200
    assert ours["dynamic"]["commander_resolved_rate"] > 0.8
    assert ours["dynamic"]["wheel_timing"]["with_commander"] is not None
    for r in doc["seats"][1:]:
        assert r["dynamic"]["commander_resolved_rate"] is not None
        assert isinstance(r["dynamic"]["held_while_castable"], list)
    assert seat_audit.validate(doc) == []
    assert doc["verdict"] == "NOT", "at least one opponent seat at standard-v3 is flagged today"


def test_an_experiment_at_a_pod_is_gated_like_simulate(monkeypatch):
    """`experiment --pod` bypassed the seat audit until 2026-10-21; it now calls the
    same gate and the record carries the verdict. Proven by deleting the gate call:
    `run` then receives no seat_audit and the refusal never happens."""
    import argparse
    from manamap.sim import experiment, seat_audit
    seen = {}
    monkeypatch.setattr(experiment, "forge_resolve_table", lambda args: (["o"], None))
    def fake_gate(pod, anyway=False):
        if not anyway:
            raise SystemExit("refused")
        return {"verdict": "NOT", "anyway": True}
    monkeypatch.setattr(seat_audit, "gate", fake_gate)
    def fake_run(*a, **k):
        seen.update(k); raise SystemExit("ran")
    monkeypatch.setattr(experiment, "run", fake_run)
    args = argparse.Namespace(slug="d", a="working", b="@x", pod="p", games=1, jobs=1, clock=None,
                              list=False, analyze=None, dry_run=False, anyway=False)
    with pytest.raises(SystemExit, match="refused"):
        experiment.main(args)
    args.anyway = True
    with pytest.raises(SystemExit, match="ran"):
        experiment.main(args)
    assert seen["seat_audit"] == {"verdict": "NOT", "anyway": True}
