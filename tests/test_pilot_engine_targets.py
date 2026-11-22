"""ONE engine model (2026-11-22): a deck that opts in with `targets_from:
engine.json` has its goldfish targets GENERATED from the engine graph's components
by `engine-graph --write`, the pilot's marks (`required`, `routes`, `overrides`,
`authored_targets`) kept beside them; a deck that does not opt in is byte-identical.
On the synthetic deck `test_pilot_engine_graph` builds.
"""
import json

import pytest

from manamap.pilot import engine_graph as eg
from manamap.pilot import scaffold_targets as st
from manamap.pilot import validate_goldfish_targets as vgt
from test_pilot_engine_graph import SLUG, synthetic_deck  # noqa: F401 - the fixture

DECLARATION = {
    "model_sacrifice": True, "model_drain": True,
    "targets_from": "engine.json",
    "components": {
        "required": ["ev:DIES"],
        "routes": {"ev:LIFE_GAIN": "drain"},
        "overrides": [{"component": "ev:DIES", "drop": ["Bloodletter of Aclazotz"], "add": [],
                       "why": "a doubler, not a death"}],
        "authored_targets": [{"label": "THE PILOT'S OWN: Artist in hand", "need": [{"any_of": ["Blood Artist"]}]}],
    },
    "targets": [{"label": "stale hand-written", "need": [{"any_of": ["Sol Ring"]}]}],
}


def _declare(base, doc=DECLARATION):
    (base / "goldfish_targets.json").write_text(json.dumps(doc, indent=1) + "\n")
    return base / "goldfish_targets.json"


def test_generated_targets_follow_the_graph_and_keep_authored_marks(synthetic_deck):  # noqa: F811
    path = _declare(synthetic_deck)
    eg.write(SLUG)
    doc = json.loads(path.read_text())
    gen = [t for t in doc["targets"] if t.get("generated")]
    assert gen, doc["targets"]
    by = {t["component"]: t for t in gen}
    # The death component: outlets make it, the drains read it; required by the pilot.
    dies = by["ev:DIES"]
    assert dies["required"] is True
    assert "Viscera Seer" in dies["need"][0]["any_of"]
    assert {"Blood Artist", "Teysa, Orzhov Scion"} <= set(dies["need"][1]["any_of"])
    # The override dropped a name from the producer leg and said why.
    assert "Bloodletter of Aclazotz" not in dies["need"][0]["any_of"] and dies["_override"]
    # CAST is read by the commander alone: the leg becomes the goldfish's own gate.
    cast = by["ev:CAST"]
    assert cast.get("commander") is True and len(cast["need"]) == 1
    assert "Vampire Nighthawk" in cast["need"][0]["any_of"] and "Edgar Markov" not in cast["need"][0]["any_of"]
    # The route mark travels; the substrate (MANA, TURN_PHASE) makes no target.
    assert all(t["component"] not in ("ev:MANA", "ev:TURN_PHASE") for t in gen)
    # The pilot's own target follows, verbatim, and the stale hand-written one is gone.
    assert doc["targets"][-1] == DECLARATION["components"]["authored_targets"][0]
    assert not any(t["label"] == "stale hand-written" for t in doc["targets"])
    # Every other key is kept.
    assert doc["model_sacrifice"] is True and doc["components"] == DECLARATION["components"]
    # And the declaration validates.
    errs = vgt.validate(doc, SLUG, synthetic_deck)
    assert errs == [], errs


def test_a_hand_edit_to_a_generated_target_is_refused(synthetic_deck):  # noqa: F811
    path = _declare(synthetic_deck)
    eg.write(SLUG)
    doc = json.loads(path.read_text())
    gen = next(t for t in doc["targets"] if t.get("component") == "ev:DIES")
    gen["need"][0]["any_of"].append("Sol Ring")
    errs = vgt.validate(doc, SLUG, synthetic_deck)
    assert any("differs from a fresh derivation" in e and "overrides" in e for e in errs), errs
    # A generated target removed by hand is refused too.
    doc = json.loads(path.read_text())
    doc["targets"] = [t for t in doc["targets"] if t.get("component") != "ev:DIES"]
    errs = vgt.validate(doc, SLUG, synthetic_deck)
    assert any("ev:DIES has no generated target" in e for e in errs), errs
    # And an authored tail that is not the kept list.
    doc = json.loads(path.read_text())
    doc["targets"].append({"label": "smuggled", "need": [{"any_of": ["Sol Ring"]}]})
    errs = vgt.validate(doc, SLUG, synthetic_deck)
    assert any("authored_targets verbatim" in e for e in errs), errs


def test_a_deck_without_targets_from_is_byte_identical(synthetic_deck):  # noqa: F811
    plain = {"model_sacrifice": True, "targets": [{"label": "hand-written", "need": [{"any_of": ["Blood Artist"]}]}]}
    path = _declare(synthetic_deck, plain)
    before = path.read_bytes()
    eg.write(SLUG)
    assert path.read_bytes() == before, "a declaration that did not opt in is never touched"
    assert st.regenerate_targets(SLUG) is None
    assert vgt.validate(json.loads(before), SLUG, synthetic_deck) == []


def test_engine_activation_prices_components_when_the_graph_exists(synthetic_deck):  # noqa: F811
    from manamap.pilot import deck_audit
    from manamap.pilot.common import load_deck_cards
    _declare(synthetic_deck)
    eg.write(SLUG)
    cards = load_deck_cards(SLUG)["cards"]
    out = deck_audit.engine_activation(SLUG, cards, {}, {"W", "B", "R"}, lambda: None)
    assert out["available"]
    rows = {r.get("component"): r for r in out["targets"] if r.get("component")}
    assert "ev:DIES" in rows
    dies = rows["ev:DIES"]
    assert dies["components"][0]["odds"] == deck_audit._component(
        {"any_of": dies["components"][0]["cards"]}, {c["name"] for c in cards}, {}, 99)["odds"]
    assert rows["ev:CAST"]["commander_gated"] is True


def test_the_engine_set_is_producers_and_consumers_outside_the_substrate(synthetic_deck):  # noqa: F811
    from manamap.sim import engine_casts
    _declare(synthetic_deck)
    eg.write(SLUG)
    got = engine_casts.engine_set(SLUG) if hasattr(engine_casts, "engine_set") else engine_casts._engine_set(SLUG)
    cards = set(got["cards"])
    assert {"Viscera Seer", "Blood Artist", "Vampire Nighthawk"} <= cards
    assert "Sol Ring" not in cards and "Swamp" not in cards
    assert "engine.json" in got["sources"]


def test_derive_refuses_without_a_v2_engine(synthetic_deck):  # noqa: F811
    _declare(synthetic_deck)
    with pytest.raises(SystemExit, match="engine-graph"):
        st.derive_from_engine(SLUG)
