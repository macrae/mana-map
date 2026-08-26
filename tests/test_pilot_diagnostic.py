"""The diagnostic layer: engine health, stall, and candidate sweeps.

Three of these guard measurements that would be WRONG IN A WAY NOBODY COULD SEE:
a joint probability composed from marginals, an engine figure computed from a
declaration about a different deck, and a stall metric that measures the model
rather than the deck. All three produce ordinary-looking numbers.
"""

import pytest

from manamap.pilot import candidates, diagnostic
from manamap.pilot.common import DECKS_DIR

SLUG = "ur-dragon"
FAST = 400


def _has(slug):
    return (DECKS_DIR / slug / "cards.json").exists()


needs_deck = pytest.mark.skipif(not _has(SLUG), reason=f"no {SLUG} fixture")


@needs_deck
def test_engine_online_is_measured_not_multiplied():
    """P(A and B) is not P(A)P(B) when A and B share cards, and the components of
    an engine always do.

    Measured on ur-dragon at 3000 games: the joint is 0.1010 by turn three
    against a product of marginals of 0.0582 — **1.74x**. A product would
    understate the engine by 42% and would look entirely plausible doing it. If
    this ever stops differing, the joint has quietly become a product.
    """
    from manamap.pilot import goldfish
    from manamap.pilot.common import deck_file, load_json
    targets = (load_json(deck_file(SLUG, "goldfish_targets.json")) or {}).get("targets") or []
    req = [i for i, t in enumerate(targets) if t.get("required")]
    if not req:
        pytest.skip("no required targets declared on this deck")
    rows = goldfish.run(SLUG, with_results=True, iterations=1500,
                        seed=diagnostic.HARNESS["seed"], max_turn=10,
                        quiet=True)["_results"]
    n = len(rows)

    def met(r, i, turn):
        v = r["target_turns"][i]
        return v is not None and v <= turn

    joint = sum(1 for r in rows if all(met(r, i, 3) for i in req)) / n
    product = 1.0
    for i in req:
        product *= sum(1 for r in rows if met(r, i, 3)) / n
    assert joint > product, (
        f"the joint ({joint:.4f}) is not above the product of marginals "
        f"({product:.4f}) — components that share cards should be positively "
        f"correlated, so this suggests the joint is being composed rather than "
        f"counted")


@needs_deck
def test_an_absent_declaration_gives_an_absent_figure_never_a_zero():
    """"0.0" is a measurement nobody made. Same contract as `model_treasures`."""
    got = diagnostic.engine([], [{"label": "x", "need": []}])
    assert got["available"] is False
    assert "required" in got["why"]
    assert "rate" not in str(got.get("online_by_turn", ""))


@needs_deck
def test_a_declaration_about_another_deck_is_refused():
    """A branch inherits the deck's declaration, which is right for a swap and
    wrong for a rebuild. A number computed from the wrong one is a real
    measurement OF A DIFFERENT DECK and looks completely ordinary."""
    targets = [{"label": "needs a card this list lacks", "required": True,
                "need": [{"any_of": ["Black Lotus", "Time Walk"]}]}]
    missing = diagnostic.declaration_fits(targets, {"Sol Ring", "Mountain"})
    assert missing, "a target naming cards the list lacks was not detected"
    got = diagnostic.engine([], targets, missing)
    assert got["available"] is False
    assert "does not describe this list" in got["why"]


@needs_deck
def test_a_stall_is_a_turn_with_nothing_castable():
    """NOT a turn with nothing cast. The goldfish is a resource model — it never
    casts a wipe or a counterspell — so 'nothing was cast' measures what the
    model declines to represent. Scored that way ur-dragon reads 6.4 dead turns
    in ten while its hand grows to eleven cards."""
    from manamap.pilot import goldfish
    rows = goldfish.run(SLUG, with_results=True, iterations=300, seed=7,
                        quiet=True)["_results"]
    assert "stall_by_turn" in rows[0], "the goldfish stopped recording stalls"
    got = diagnostic.stall(rows)
    assert "castable" in got["basis"]
    # Turn one is nearly always a stall (one mana, almost no one-drops), and
    # turn three rarely is. If that ordering ever inverts the measure is broken.
    t1 = got["by_turn"]["1"]["rate"]
    t3 = got["by_turn"]["3"]["rate"]
    assert t1 > t3, f"P(stall) rose from turn 1 ({t1}) to turn 3 ({t3})"
    assert got["from_turn"] > 1, "the headline includes turn one, which is structural"


@needs_deck
def test_the_mde_switches_method_rather_than_overflowing():
    """`stats.mde_proportion` walks an EXACT binomial grid — right at twenty
    games an arm, and `math.comb(4000, k)` overflows a float. The normal
    approximation is not a compromise at that size; it is the regime it is valid
    in."""
    small = diagnostic._mde(0.2, 100, 100)
    big = diagnostic._mde(0.2, 10000, 10000)
    assert small is not None and big is not None
    assert big < small, "more games should detect a SMALLER difference"
    assert big < 0.05, f"10k games should see well under 5 points, got {big}"


@needs_deck
def test_a_candidate_sweep_marks_what_it_cannot_detect():
    """A ranking of deltas smaller than the MDE is a ranking of noise."""
    got = candidates.sweep(SLUG, ["Jeweled Lotus", "Mana Crypt"],
                           axis="stall", iterations=FAST)
    assert got["mde"] is not None
    assert got["baseline"]["rate"] is not None
    assert got["candidates"], "the sweep produced no rows"
    for r in got["candidates"]:
        if "rate" in r:
            assert "in_declaration" in r


@needs_deck
def test_the_sweep_never_cuts_a_card_the_declaration_names():
    """The first version took the most expensive card outright — Utvara Hellkite
    on ur-dragon, which a declared target names — so every candidate came back
    'no reading': the engine correctly refused to measure a declaration that no
    longer described the list. The sweep was testing its own cut."""
    declared = candidates._declared_cards(SLUG, None)
    if not declared:
        pytest.skip("no declaration on this deck")
    got = candidates.sweep(SLUG, ["Jeweled Lotus"], axis="stall", iterations=FAST)
    cuts = {r.get("cut") for r in got["candidates"] if r.get("cut")}
    assert not (cuts & declared), f"the sweep cut a declared card: {cuts & declared}"


@needs_deck
def test_a_truncated_pool_says_so():
    """A silently truncated list reads as 'these are all of them'."""
    got = candidates.sweep(SLUG, ["Jeweled Lotus", "Mana Crypt", "Sensei's Divining Top"],
                           axis="stall", iterations=FAST, limit=1)
    assert len(got["not_considered"]) == 2, got["not_considered"]
    assert got["considered"] == 1
