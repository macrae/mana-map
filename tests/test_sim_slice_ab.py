"""A/B on one board (PRD v2 Step 5, the wrapper). Unit tests with a fake runner: the
pairing, the measures, the answer line and the misplay flag."""
import pytest

from manamap.sim import slice_ab as ab
from manamap.sim import slice_state as ss

YOU = "Ai(1)-mm-shark"


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(ss, "_deck_copies", lambda slug: (
        ["Brallin, Skyshark Rider", "Shabraz, the Skyshark", "Lightning Greaves", "Arcane Signet",
         "Windfall"] + ["Island"] * 10, ["Brallin, Skyshark Rider", "Shabraz, the Skyshark"]))
    monkeypatch.setattr(ss, "to_forge_state", lambda sc, slugs, seed, known_names=None: (
        f"hand={sc['seats'][0]['hand']} seed={seed}", []))
    from manamap.sim import forge
    monkeypatch.setattr(forge, "SIM_DECK_PREFIX", "mm-")
    monkeypatch.setattr(forge, "deck_meta_name", lambda s: s)


def scenario(card):
    return {"version": 2, "turn": 6, "phase": "precombat main", "active_seat": "you",
            "stack": [], "actions": [],
            "seats": [{"seat": "you", "life": 40, "board": ["Brallin, Skyshark Rider"],
                       "hand": [card, "Windfall"]},
                      {"seat": "opp1", "life": 40, "board": []}]}


def record(label, seed, opp_end, brallin_alive, cast=True, card="Lightning Greaves"):
    bf = ["Island"] + (["Brallin, Skyshark Rider"] if brallin_alive else [])
    log = [f"Add To Stack: {YOU} cast {card}"] if cast else []
    log += [f"Zone Change: Island ({i}) was put into Hand from Library. owner {YOU}" for i in range(2)]
    return {"label": label, "seed": seed, "start": [
                {"life": 40, "battlefield": ["Brallin, Skyshark Rider"], "hand": [card, "Windfall"], "lost": False},
                {"life": 40, "battlefield": [], "hand": [], "lost": False}],
            "end": [{"life": 37, "battlefield": bf, "hand": [], "lost": False},
                    {"life": opp_end, "battlefield": [], "hand": [], "lost": False}],
            "log": log}


def fake_run(table):
    def run(order, cases, rounds=1):
        assert order == ["shark", "angels"]
        return [table[(label, seed)] for label, seed, _ in cases]
    return run


SLUGS = {"you": "shark", "opp1": "angels"}
ARMS = {"greaves": scenario("Lightning Greaves"), "signet": scenario("Arcane Signet")}


def test_the_difference_is_read_seed_by_seed_on_the_named_primary():
    seeds = list(range(1, 9))
    table = {}
    for s in seeds:
        table[("greaves", s)] = record("greaves", s, 30, brallin_alive=True)
        table[("signet", s)] = record("signet", s, 34, brallin_alive=s % 2 == 0, card="Arcane Signet")
    rep = ab.compare(ARMS["greaves"], SLUGS, ARMS, seeds, "opp_life_lost", run=fake_run(table))
    row = next(r for r in rep["rows"] if r["primary"])
    assert row["measure"] == "opp_life_lost" and row["a_mean"] == 10 and row["b_mean"] == 6
    assert row["paired"]["diff"] == -4 and row["paired"]["excludes_zero"]
    out = next(r for r in rep["rows"] if r["measure"] == "commander_out")
    assert out["a_mean"] == 1.0 and out["b_mean"] == 0.5 and "holm" in out
    drawn = next(r for r in rep["rows"] if r["measure"] == "cards_drawn")
    assert drawn["a_mean"] == 2.0
    assert rep["setup"]["arms"] == {"greaves": ["Lightning Greaves"], "signet": ["Arcane Signet"]}
    assert rep["answer"].startswith("signet gives less opp_life_lost than greaves: -4")


def test_an_interval_that_spans_zero_says_no_difference():
    seeds = [1, 2, 3, 4]
    table = {(lab, s): record(lab, s, 30 + (s % 2) * (2 if lab == "signet" else -2), True,
                              card="Arcane Signet" if lab == "signet" else "Lightning Greaves")
             for lab in ARMS for s in seeds}
    rep = ab.compare(ARMS["greaves"], SLUGS, ARMS, seeds, "opp_life_lost", run=fake_run(table))
    assert rep["answer"].startswith("No difference on opp_life_lost")


def test_a_card_the_ai_never_cast_flags_its_arm_for_discount():
    seeds = [1, 2, 3, 4]
    table = {}
    for s in seeds:
        table[("greaves", s)] = record("greaves", s, 30, True)
        table[("signet", s)] = record("signet", s, 30, True, cast=(s == 1), card="Arcane Signet")
    rep = ab.compare(ARMS["greaves"], SLUGS, ARMS, seeds, "opp_life_lost", run=fake_run(table))
    assert rep["misplays"] == [{"arm": "signet", "card": "Arcane Signet", "held": 3, "of": 4}]
    assert "DISCOUNT signet: the AI held Arcane Signet uncast in 3 of 4" in rep["answer"]


def test_a_failed_replicate_drops_its_seed_from_both_arms():
    seeds = [1, 2, 3]
    table = {(lab, s): record(lab, s, 30, True, card="Arcane Signet" if lab == "signet" else "Lightning Greaves")
             for lab in ARMS for s in seeds}
    table[("signet", 2)] = {"label": "signet", "seed": 2, "error": "timeout"}
    rep = ab.compare(ARMS["greaves"], SLUGS, ARMS, seeds, "opp_life_lost", run=fake_run(table))
    assert rep["setup"]["seeds"] == 2 and rep["errors"] == ["signet seed 2: timeout"]


def test_the_primary_must_be_a_measure_and_there_are_two_arms():
    with pytest.raises(ValueError, match="not a measure"):
        ab.compare(ARMS["greaves"], SLUGS, ARMS, [1], "vibes", run=fake_run({}))
    with pytest.raises(ValueError, match="exactly two"):
        ab.compare(ARMS["greaves"], SLUGS, {"a": ARMS["greaves"]}, [1], "opp_life_lost", run=fake_run({}))


def test_the_keyed_deal_keeps_two_arms_paired_apart_from_the_card_tested():
    """Removing one card from the pile must not reshuffle the rest — that is what
    makes seed k of arm A and seed k of arm B the same draws."""
    pile = ["Island"] * 5 + ["Windfall", "Sol Ring", "Arcane Signet", "Lightning Greaves"]
    a = ss.keyed_order([c for c in pile if c != "Arcane Signet"], seed=7)
    b = ss.keyed_order([c for c in pile if c != "Lightning Greaves"], seed=7)
    assert [c for c in a if c != "Lightning Greaves"] == [c for c in b if c != "Arcane Signet"]
    assert ss.keyed_order(pile, 7) != ss.keyed_order(pile, 8)
