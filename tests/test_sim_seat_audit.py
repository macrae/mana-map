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


def test_a_wheel_with_no_commander_gate_is_named_and_never_fails_the_seat(monkeypatch):
    """Incendiary Command and Wheel of Fortune sat on their own rules until a hint said
    otherwise, and only a dynamic 86% fail showed it (2026-10-25). Proven by gating the
    second wheel: it leaves the list."""
    wheel = "Each player discards their hand, then draws seven cards."
    cards = [{"name": n, "type_line": "Sorcery", "cmc": 3, "mana_cost": "{2}{R}", "oracle_text": wheel}
             for n in ("Wheel A", "Wheel B")]
    monkeypatch.setattr(seat_audit, "_seat_doc", lambda slug: {"cards": cards})
    monkeypatch.setattr(seat_audit, "_floors", lambda slug: [])
    gate = "SVar:NeedsToPlay:Creature.IsCommander+YouCtrl\n"
    _scripts(monkeypatch, {"Wheel A": "Name:Wheel A\n" + gate, "Wheel B": "Name:Wheel B\n"})
    st = seat_audit.static_seat("x")
    assert st["ungated_wheels"] == ["Wheel B"]
    assert seat_audit._verdict("x", st, None, ours=True) == ("PILOTABLE", [])
    _scripts(monkeypatch, {"Wheel A": "Name:Wheel A\n" + gate, "Wheel B": "Name:Wheel B\n" + gate})
    assert seat_audit.static_seat("x")["ungated_wheels"] == []


def test_a_held_card_is_marked_gated_or_reactive_and_a_plain_hold_is_not(monkeypatch):
    _scripts(monkeypatch, {
        "Gated Wheel": "Name:Gated Wheel\nTypes:Sorcery\nA:SP$ Discard | Mode$ Hand\n"
                       "SVar:NeedsToPlay:Creature.IsCommander+YouCtrl\n",
        "Negate": "Name:Negate\nTypes:Instant\nA:SP$ Counter | TargetType$ Spell\n",
        "Plain": "Name:Plain\nTypes:Sorcery\nA:SP$ Draw | NumCards$ 2\n"})
    assert seat_audit._held_marks(["Gated Wheel", "Negate", "Plain"]) == {
        "Gated Wheel": ["gated"], "Negate": ["reactive"]}


def test_a_wheel_rate_under_one_a_game_fails_the_stated_minimum():
    """The definition says at least one a game; `if not rate` passed 0.35 (olivers-table-3p,
    run caeeec41). Proven by restoring the truthiness: 0.35 and 0.99 read PILOTABLE."""
    st = {"flagged": [], "unfound": [], "floors": []}
    def wheels(rate):
        return {"commander_resolved_rate": 0.9, "engine": {},
                "wheel_timing": {"wheels_per_game": rate, "with_commander": 0.95}}
    for rate in (0, None, 0.35, 0.99):
        verdict, why = seat_audit._verdict("x", st, wheels(rate), ours=True)
        assert verdict == "NOT" and any("wheels a game" in w for w in why), rate
    assert seat_audit._verdict("x", st, wheels(1.0), ours=True) == ("PILOTABLE", [])
    assert seat_audit._verdict("x", st, wheels(0.35), ours=False) == ("PILOTABLE", []), "our seat's rule"


def test_our_seat_is_judged_on_the_decks_own_engine(monkeypatch):
    """Our seat is not in the pod file, so `engine` never named it and the definition's
    engine line never read it (2026-11-04). Proven by deleting the `_our_engine` call:
    `seen` has no row for ours. A branch seat reads the branch's list."""
    from manamap.sim import engine_casts, forge, pods
    asked = {}
    def engine_set(slug, branch=None):
        asked[slug] = branch
        return {"cards": ["Windfall"], "scaffolded": slug == "scaffold", "sources": ["engine.json"]}
    monkeypatch.setattr(engine_casts, "engine_set", engine_set)
    assert seat_audit._our_engine("d@b") == ["Windfall"] and asked["d"] == "b"
    assert seat_audit._our_engine("scaffold") == [], "a scaffold is role axes, not an engine"

    seen = {}
    def dynamic(rec, texts, engine_by_slug=None):
        seen.update(engine_by_slug)
        return {"d@b": {"engine": {"Windfall": {"cast": 0, "held": False, "never_cast": True}}}}
    monkeypatch.setattr(seat_audit, "dynamic_seats", dynamic)
    monkeypatch.setattr(seat_audit, "static_seat", lambda slug: {"flagged": [], "unfound": [], "floors": []})
    monkeypatch.setattr(seat_audit, "_find_run", lambda run_id: _Rec())
    monkeypatch.setattr(seat_audit, "_logs_for", lambda path, rec: ["log"])
    monkeypatch.setattr(pods, "load", lambda name: {"seats": [{"slug": "opp", "engine": ["Opp Card"]}]})
    monkeypatch.setattr(forge, "card_overrides", lambda: {"sha": "x"})
    monkeypatch.setattr(forge, "telemetry_tag", lambda: "")
    doc = seat_audit.audit("p", run_id="r")
    assert seen == {"opp": ["Opp Card"], "d@b": ["Windfall"]}
    ours = next(r for r in doc["seats"] if r["ours"])
    assert ours["verdict"] == "NOT" and any("Windfall" in w for w in ours["why"])


class _Rec:
    """`_find_run`'s path, read back as a one-seat record."""
    def read_text(self, encoding=None):
        return json.dumps({"seats": [{"slug": "d@b"}], "summary": {"decided": 1}})


def test_a_branch_seat_reads_its_decks_floors(monkeypatch, tmp_path):
    """Hints live on the deck; `seat_home("d@b")` is a folder with no such file, so a branch
    seat read no floors and failed on a card the pilot had named one. Proven by passing
    the raw slug to `seat_home`: `_floors` reads []."""
    from manamap.sim import forge
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "forge_hints.json").write_text(json.dumps({"hints": [], "floors": ["Ghostly Pilferer"]}))
    monkeypatch.setattr(forge, "seat_home", lambda slug: tmp_path / slug.replace("@", "/branches/"))
    assert seat_audit._floors("d@b") == ["Ghostly Pilferer"]


def test_an_engine_list_is_a_form_and_says_why(tmp_path, monkeypatch):
    from manamap.sim import pods
    monkeypatch.setattr(pods, "PODS_DIR", tmp_path)
    def pod(**seat):
        (tmp_path / "p.json").write_text(json.dumps({"seats": [{"slug": "s", **seat}]}))
    pod(engine=["A"], engine_why="its plan")
    assert pods.load("p")["seats"][0]["engine"] == ["A"]
    pod(engine=["A"])
    with pytest.raises(pods.PodError, match="engine_why"):
        pods.load("p")
    pod(engine="A", engine_why="x")
    with pytest.raises(pods.PodError, match="non-empty list"):
        pods.load("p")


@requires_deck
def test_every_engine_card_a_pod_names_is_in_that_seats_list_and_pods_agree():
    """An engine card the list does not hold reads as `cast 0, not held, not never`: a
    silent pass. And one seat at two tables is one plan. Proven by renaming a card in
    a pod: the first assertion names it."""
    from manamap.sim import forge, pods
    checked, by_slug = 0, {}
    for name in pods.available():
        for seat in pods.load(name)["seats"]:
            for card in seat.get("engine") or []:
                assert card in forge._seat_names(seat["slug"]), f"{name}/{seat['slug']}: {card}"
                checked += 1
            if seat.get("engine"):
                assert by_slug.setdefault(seat["slug"], seat["engine"]) == seat["engine"], seat["slug"]
    assert checked >= 30 and len(by_slug) >= 6


def test_the_audit_judges_a_hold_net_of_the_gate_and_names_what_that_cleared(monkeypatch):
    """Driven through `dynamic_seats` on the real telemetry fixture. Goblin Bombardment read
    six raw turns; gated with no commander on the board, none of them was a hold. Proven by
    judging on `raw_held` again: it stays on the held list."""
    from pathlib import Path
    from manamap.sim import forge
    text = (Path(__file__).parent / "fixtures" / "forge" / "two-seat-one-game-telemetry.log").read_text()
    cmc = {"Goblin Bombardment": 2, "Great Train Heist": 1}
    monkeypatch.setattr(forge, "cmc_map", lambda slug, doc=None: dict(cmc))
    monkeypatch.setattr(forge, "keyword_map", lambda slug, doc=None: {})
    monkeypatch.setattr(forge, "played_doc", lambda slug, rec: None)
    monkeypatch.setattr(forge, "record_commanders", lambda rec: {})
    monkeypatch.setattr(seat_audit, "commander_gated_names", lambda names: {"Goblin Bombardment"})
    monkeypatch.setattr(seat_audit, "_held_marks", lambda held: {})
    rec = {"seats": [{"slug": "goblin-storm", "forge_name": "mm-goblin-storm"},
                     {"slug": "giada-angels", "forge_name": "mm-giada-angels"}]}
    ours = seat_audit.dynamic_seats(rec, [text])["goblin-storm"]
    assert "Great Train Heist" in ours["held_while_castable"]
    assert "Goblin Bombardment" not in ours["held_while_castable"]
    assert ours["raw_only"] == ["Goblin Bombardment"]
    row = ours["held_turns"]["Great Train Heist"]
    assert row["payable"] <= row["raw"] and row["raw"] > 0
