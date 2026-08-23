"""deck-info's combat block: the consumer must ask for keys the producer emits.

WHY THIS FILE EXISTS. `deck_info._goldfish` read three keys —
`kill_by_turn_8_pct`, `kill_turn_distribution`, `never_by_turn_10_pct` — and
`goldfish.aggregate` emits `kill_by_turn_rate`, `kill_turn_histogram` and
`no_kill_by_max_turn_rate`. Not one of the three matched, so the branch produced
`{}` for every input it could ever be given.

Nothing caught it, and nothing COULD have: the block only runs when a deck sets
`"model_combat": true` in `goldfish_targets.json`, and no deck ever has. Dead
code guarded by an opt-in nobody had opted into is invisible to a suite that
only ever reads committed artifacts — so this test builds the producer's output
directly instead of waiting for a deck to enable the flag.

The assertion is a CONTRACT, not a golden value: every key the consumer reaches
for must resolve against a real `aggregate(..., model_combat=True)` document.
Rename a producer key and this fails, which is the whole point.
"""

import json
import random

import pytest

from manamap.pilot import deck_info, goldfish

from conftest import requires_deck

SLUG = "edgar-vampires"


@pytest.fixture(scope="module")
def combat_metrics():
    """A real goldfish document with the combat model on, cheaply.

    100 iterations, not 10,000: this asserts the SHAPE of the join, and a shape
    does not get truer with more samples. The seed is fixed so a failure is
    reproducible.
    """
    doc = goldfish.load_deck_cards(SLUG)
    library, commanders = goldfish.build_library(doc)
    commander_cmc = int(commanders[0].get("cmc") or 0)
    rng = random.Random(goldfish.GOLDFISH_SEED)
    max_turn = goldfish.GOLDFISH_MAX_TURN
    results = [
        goldfish.simulate_once(rng, library, commander_cmc, [], max_turn,
                               model_treasures=False, model_combat=True)
        for _ in range(100)
    ]
    metrics = goldfish.aggregate(results, [], max_turn, False, True)
    return {"meta": {"seed": goldfish.GOLDFISH_SEED, "max_turn": max_turn},
            "metrics": metrics}


def _as_deck_dir(tmp_path, doc):
    """`_goldfish` takes a deck DIRECTORY and loads the artifact itself, so a
    document has to be written where the consumer looks for it."""
    (tmp_path / "goldfish_metrics.json").write_text(json.dumps(doc))
    return tmp_path


@requires_deck
def test_the_producer_emits_a_combat_block_at_all(combat_metrics):
    assert combat_metrics["metrics"].get("combat"), (
        "model_combat produced no combat block — the rest of this file is vacuous")


@requires_deck
def test_every_key_deck_info_reaches_for_resolves(combat_metrics, tmp_path):
    """The bug, stated as a test: the consumer asked for three keys that the
    producer has never emitted under any input."""
    out = deck_info._goldfish(_as_deck_dir(tmp_path, combat_metrics))
    combat = out.get("combat")
    assert combat, "deck-info dropped the combat block entirely"

    # A key present but None means the producer renamed it and the consumer did
    # not follow — exactly the failure mode being guarded, and it is NOT the same
    # as a legitimately absent figure, which is why each is checked by name.
    for key in ("mean_kill_turn", "median_kill_turn", "kill_by_turn_6_pct",
                "kill_by_turn_8_pct", "max_turn", "no_kill_by_max_turn_pct",
                "kill_turn_histogram"):
        assert key in combat, f"deck-info stopped emitting {key}"

    assert combat["max_turn"] == goldfish.GOLDFISH_MAX_TURN
    assert combat["no_kill_by_max_turn_pct"] is not None
    assert isinstance(combat["kill_turn_histogram"], dict)


@requires_deck
def test_the_never_key_is_named_from_the_artifact_not_hardcoded(combat_metrics, tmp_path):
    """The old key was `never_by_turn_10_pct`. GOLDFISH_MAX_TURN is a config
    value; a key that hardcodes its current value lies the moment it moves."""
    doc = dict(combat_metrics)
    doc["meta"] = {**combat_metrics["meta"], "max_turn": 7}
    combat = deck_info._goldfish(_as_deck_dir(tmp_path, doc))["combat"]
    assert combat["max_turn"] == 7
    assert not any("10" in k for k in combat), (
        "a combat key still hardcodes turn 10")


@requires_deck
def test_no_combat_block_when_the_flag_is_off(tmp_path):
    """Absent, not zeroed — the opt-in contract's whole point. Every tracked
    deck is in this state today, which is why this is the path that has always
    run and the one above is the path that never had."""
    doc = goldfish.load_deck_cards(SLUG)
    library, commanders = goldfish.build_library(doc)
    rng = random.Random(goldfish.GOLDFISH_SEED)
    max_turn = goldfish.GOLDFISH_MAX_TURN
    results = [
        goldfish.simulate_once(rng, library, int(commanders[0].get("cmc") or 0), [],
                               max_turn, model_treasures=False, model_combat=False)
        for _ in range(20)
    ]
    metrics = goldfish.aggregate(results, [], max_turn, False, False)
    assert "combat" not in metrics
    out = deck_info._goldfish(_as_deck_dir(tmp_path, {"meta": {}, "metrics": metrics}))
    assert "combat" not in out


# ── Stale measurements must be marked ────────────────────────────────────────

@requires_deck
def test_a_simulation_run_on_an_older_list_is_marked_stale():
    """A run record stamps every seat's decklist sha, so a measurement made
    against a list the deck no longer holds is mechanically detectable — and
    nothing was detecting it. Edgar showed a 0.25 win rate on the workbench for a
    deck that had been checked in and re-baselined under it. A stale figure
    presented as current is worse than an absent one: it is exactly as
    precise-looking as a true one, and the reader has no way to tell."""
    from manamap.pilot import deck_info
    sim = deck_info._simulation("edgar-vampires")
    if not sim:
        pytest.skip("no sim runs on this deck")
    assert "stale" in sim, "a run must say whether it measured THIS list"
    assert sim["ran_on_decklist_sha256"], "and which list it did measure"
    assert sim["stale"] is (sim["ran_on_decklist_sha256"]
                            != deck_info._current_sha("edgar-vampires"))


@requires_deck
def test_an_experiment_is_stale_only_when_NEITHER_arm_is_current():
    """An A/B compares two lists. It is still about this deck while one of them
    is the list the deck holds; it becomes history when the deck moves past
    both."""
    from manamap.pilot import deck_info
    xp = deck_info._experiments("edgar-vampires")
    if not xp:
        pytest.skip("no experiments on this deck")
    assert "stale" in xp["latest"]
