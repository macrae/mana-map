"""The diagnostic layer: engine health, stall, and candidate sweeps.

Three of these guard measurements that would be WRONG IN A WAY NOBODY COULD SEE:
a joint probability composed from marginals, an engine figure computed from a
declaration about a different deck, and a stall metric that measures the model
rather than the deck. All three produce ordinary-looking numbers.
"""

import pytest

from manamap.pilot import candidates, diagnostic
from manamap.pilot.common import DECKS_DIR
from conftest import ROOT

SLUG = "ur-dragon"
FAST = 400


def _has(slug):
    return (DECKS_DIR / slug / "cards.json").exists()


needs_deck = pytest.mark.skipif(not _has(SLUG), reason=f"no {SLUG} fixture")


def _rows(pairs, turns=10):
    """Per-iteration rows shaped as `goldfish.run(with_results=True)` returns.

    `pairs` is one (turn_a, turn_b) per iteration — the turn each required
    target was assembled, or None for never.
    """
    return [{"target_turns": list(p), "stall_by_turn": [False] * turns}
            for p in pairs]


REQUIRED_TWO = [{"label": "A", "required": True, "need": []},
                {"label": "B", "required": True, "need": []}]


def test_engine_online_is_counted_not_multiplied():
    """P(A and B) is not P(A)P(B) when A and B share cards, and an engine's
    components always do. Measured on ur-dragon at 3000 games the joint is
    0.1010 by turn three against a product of marginals of 0.0582 — 1.74x, so a
    product would understate the engine by 42% and look plausible doing it.

    THIS TEST USED TO PROVE ARITHMETIC. It re-implemented the `required` filter
    and the joint count out of `diagnostic.py` and never called
    `diagnostic.engine()` — so changing `_engine` to multiply its marginals left
    it green. It also skipped when no deck declared `required`, which is 1 of 13,
    putting the flagship metric's only control one artifact edit from never
    running at all.

    Driven through the production function on rows built so the two answers
    cannot coincide: A and B are perfectly correlated, so the joint is 0.5 and
    the product of marginals is 0.25.
    """
    rows = _rows([(1, 1)] * 50 + [(None, None)] * 50)
    got = diagnostic.engine(rows, REQUIRED_TWO)
    assert got["available"] is True
    assert got["online_by_turn"]["3"]["rate"] == pytest.approx(0.5), (
        f"expected the COUNTED joint (0.5); a product of marginals would give "
        f"0.25 and got {got['online_by_turn']['3']['rate']}")


def test_the_joint_and_the_product_disagree_on_the_real_deck_too():
    """The synthetic case proves the code counts; this proves it MATTERS on a
    real declaration. No skip: a deck with no `required` marking is a fixture
    problem and must fail loudly rather than pass quietly."""
    from manamap.pilot import goldfish
    from manamap.pilot.common import deck_file, load_json
    targets = (load_json(deck_file(SLUG, "goldfish_targets.json")) or {}).get("targets") or []
    req = [i for i, t in enumerate(targets) if t.get("required")]
    assert req, (
        f"{SLUG} declares no `required` target, so the engine figure cannot be "
        f"computed for it — pick a fixture deck that can, rather than skipping")
    rows = goldfish.run(SLUG, with_results=True, iterations=1500,
                        seed=diagnostic.HARNESS["seed"], max_turn=10,
                        quiet=True)["_results"]
    joint = diagnostic.engine(rows, targets)["online_by_turn"]["3"]["rate"]
    product = 1.0
    for i in req:
        one = [dict(t, required=(j == i)) for j, t in enumerate(targets)]
        product *= diagnostic.engine(rows, one)["online_by_turn"]["3"]["rate"]
    assert joint > product * 1.1, (
        f"the joint ({joint:.4f}) is not meaningfully above the product of "
        f"marginals ({product:.4f}) — components that share cards are "
        f"positively correlated, so this suggests it is being composed")


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


# ── Calibration ──────────────────────────────────────────────────────────
#
# The constants in `diagnostic.FLEET` came from running every tracked deck. These
# re-derive them, so a band cannot outlive the evidence it was measured from —
# the discipline `scaffold_targets.BROAD_GROUP` already keeps.

FLEET_ITERATIONS = 1200


def _fleet_readings():
    import glob
    out = []
    for path in sorted(glob.glob(str(ROOT / "data/decks/*/cards.json"))):
        slug = path.split("/")[2]
        try:
            out.append(diagnostic.run(slug, iterations=FLEET_ITERATIONS, quiet=True))
        except Exception:
            continue
    return out


@needs_deck
@pytest.mark.fleet
def test_the_fleet_bands_still_describe_the_fleet():
    """`FLEET` is context for placing a reading, and it must keep describing the
    decks it was measured from."""
    rows = _fleet_readings()
    if len(rows) < 5:
        pytest.skip("too few decks on this machine to calibrate against")
    got = [((r.get("stall") or {}).get("two_in_a_row") or {}).get("rate") for r in rows]
    got = [x for x in got if x is not None]
    band = diagnostic.FLEET["stall_two_in_a_row"]
    # Sampling moves these, so the assertion is that the band still CONTAINS the
    # fleet's centre rather than that the endpoints are unchanged.
    import statistics
    med = statistics.median(got)
    assert band["min"] <= med <= band["max"], (
        f"the recorded band {band} no longer contains the fleet median {med:.3f} "
        f"— re-measure it rather than widening it")


@needs_deck
@pytest.mark.fleet
def test_the_prd_threshold_would_fire_on_nothing():
    """A red line that can never go red is as useless as one that always does.

    The PRD asks for `P(stall by turn 4) > 0.15 -> red`. The whole fleet's
    highest reading is 0.079. This asserts the finding rather than the number:
    if a deck ever exceeds it, the threshold becomes meaningful and this test
    says so by failing.
    """
    rows = _fleet_readings()
    if len(rows) < 5:
        pytest.skip("too few decks")
    got = [((r.get("stall") or {}).get("two_in_a_row") or {}).get("rate") or 0
           for r in rows]
    fired = [x for x in got if x > diagnostic.PRD_STALL_THRESHOLD_REJECTED]
    assert not fired, (
        f"a deck now exceeds the PRD's 0.15 stall threshold ({max(got):.3f}) — "
        f"it is no longer inert and is worth reconsidering as a real red line")


@needs_deck
@pytest.mark.fleet
def test_the_mana_readings_are_still_one_measurement():
    """Three readings that move together are one finding, not three.

    `benchmark.py` recorded the two-way version (r = 0.97) and refused to sum
    them. If they ever decouple, the note that says "read them as one signal"
    becomes wrong and should be removed.
    """
    import statistics
    rows = _fleet_readings()
    if len(rows) < 5:
        pytest.skip("too few decks")
    a = [((r.get("mana") or {}).get("missed_land_drop_by_five") or {}).get("rate")
         for r in rows]
    b = [((r.get("mana") or {}).get("mulliganed") or {}).get("rate") for r in rows]
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    r = num / den if den else 0
    assert r > 0.85, (
        f"missed-drop and mulligan rate have decoupled (r = {r:.3f}); the note "
        f"claiming they are one measurement is no longer true")


@needs_deck
def test_an_override_reaches_the_simulation_not_just_the_report():
    """`--as` asks a hypothetical: if this card counted toward that component,
    how far would the engine move?

    The first cut passed the modified declaration to the REPORTING layer while
    the goldfish still read the file, so `target_turns` stayed indexed by the
    file's targets and the override changed nothing. The tell was eight
    different candidates returning the identical 0.501 — which is also what a
    CORRECT run looks like, so the two are told apart by whether widening moves
    the number at all.
    """
    from manamap.pilot import goldfish
    from manamap.pilot.common import deck_file, load_json
    doc = load_json(deck_file(SLUG, "goldfish_targets.json")) or {}
    targets = doc.get("targets") or []
    req = [t for t in targets if t.get("required")]
    if not req:
        pytest.skip("no required targets")
    import copy
    from manamap.pilot.common import load_deck_cards
    widened = copy.deepcopy(targets)
    held = [c["name"] for c in load_deck_cards(SLUG)["cards"]
            if not c.get("is_commander")]
    target = next(t for t in widened if t.get("required"))
    group = target["need"][0]["any_of"]
    # A card that is IN the deck and NOT already in the group — adding one the
    # group already names is a no-op, which is how the first version of this
    # test failed against working code.
    extra = next(n for n in held if n not in group)
    group.append(extra)
    base = goldfish.run(SLUG, with_results=True, iterations=800, seed=3, quiet=True)
    alt = goldfish.run(SLUG, with_results=True, iterations=800, seed=3, quiet=True,
                       targets_override=widened)
    i = widened.index(target)

    def hits(res):
        return sum(1 for r in res["_results"]
                   if r["target_turns"][i] is not None and r["target_turns"][i] <= 3)

    assert hits(alt) > hits(base), (
        "widening a required group did not raise its assembly rate — the "
        "override is not reaching the simulation")


def test_a_frequency_never_erases_a_difference_the_interval_shows():
    """A frequency is easier to hold than a rate and it is a PRESENTATION AID.

    At an absolute tolerance both 0.039 and 0.053 rounded to "1 game in 20" —
    and the interval on that difference excludes zero, so the phrasing was
    hiding a real cost behind identical words. The moment two numbers a
    confidence interval separates would print the same, the numbers win.
    """
    assert diagnostic._pair(0.039, 0.053) != "1 game in 20 -> 1 game in 20"
    a, b = diagnostic.as_frequency(0.039), diagnostic.as_frequency(0.053)
    assert a != b or "%" in diagnostic._pair(0.039, 0.053)
    # Fractions reduce: "2 games in 50" is "1 game in 25".
    for rate in (0.04, 0.05, 0.1, 0.2, 0.25, 0.5, 0.6, 0.75):
        got = diagnostic.as_frequency(rate)
        if " in " in got:
            num, denom = got.split(" in ")
            import math
            assert math.gcd(int(num.split()[0]), int(denom)) == 1, got


def test_the_reading_tells_no_change_from_cannot_see():
    """They look identical on the page and they are opposite findings."""
    a = {"engine": {}, "stall": {}, "mana": {}}
    b = dict(a)
    # A difference smaller than the MDE: evidence of nothing.
    unseen = {"x": {"label": "stall (2 in a row)", "a": 0.10, "b": 0.104,
                    "delta": 0.004, "ci95_diff": [-0.02, 0.03],
                    "excludes_zero": False, "mde": 0.02}}
    got = diagnostic.interpret(a, b, unseen)
    row = next(r for r in got if r["measure"] == "stall (2 in a row)")
    assert row["kind"] == "unseen"
    assert "evidence of NOTHING" in row["detail"]
    # A difference the run COULD have resolved and did not: evidence of no change.
    flat = {"x": {"label": "stall (2 in a row)", "a": 0.10, "b": 0.101,
                  "delta": 0.001, "ci95_diff": [-0.004, 0.006],
                  "excludes_zero": False, "mde": 0.0005}}
    row = next(r for r in diagnostic.interpret(a, b, flat)
               if r["measure"] == "stall (2 in a row)")
    assert row["kind"] == "flat"
    assert "COULD have resolved" in row["detail"]


def test_the_reading_knows_which_direction_is_better():
    """Without this, a lower stall reads as a loss."""
    a = {"engine": {}}
    worse_stall = {"x": {"label": "stall (2 in a row)", "a": 0.03, "b": 0.05,
                         "delta": 0.02, "ci95_diff": [0.01, 0.03],
                         "excludes_zero": True, "mde": 0.005}}
    row = diagnostic.interpret(a, a, worse_stall)[0]
    assert row["kind"] == "cost", "a higher stall was read as a gain"
    better_engine = {"y": {"label": "engine online by turn 3", "a": 0.1, "b": 0.6,
                           "delta": 0.5, "ci95_diff": [0.48, 0.52],
                           "excludes_zero": True, "mde": 0.008}}
    row = diagnostic.interpret(a, a, better_engine)[0]
    assert row["kind"] == "gain"


@needs_deck
def test_a_source_is_blind_only_if_every_channel_is_blind():
    """`treasure_sources_not_modelled` exists so a low hoard figure is LEGIBLE,
    which makes over-reporting it the same failure as omitting it.

    It was built from `treasure_profile` alone while the model has three ways to
    see a Treasure: the trigger table, `treasure_bonus` (a multiplier) and
    `combat.attack_treasure` once `model_combat` is on. On ur-dragon's treasure
    branch it named nineteen sources invisible when eight were being simulated.
    """
    from manamap.pilot import goldfish
    from manamap.pilot.common import load_deck_cards
    doc = load_deck_cards(SLUG, "treasure-v2")
    names = {c["name"] for c in doc["cards"]}
    if "Xorn" not in names:
        pytest.skip("no multiplier in this branch to test with")
    got = goldfish.run(SLUG, branch="treasure-v2", iterations=20, quiet=True,
                       model_treasures=True, model_combat=True)
    blind = set((got.get("meta") or {}).get("treasure_sources_not_modelled") or [])
    # A multiplier IS modelled — `treasure_bonus` feeds every creation event.
    assert "Xorn" not in blind, "a multiplier was reported as invisible"
    # An attack trigger IS modelled once combat is on.
    for n in ("Goldspan Dragon", "Ragavan, Nimble Pilferer"):
        if n in names:
            assert n not in blind, f"{n} is simulated via attack_treasure"
    # With combat OFF the same card genuinely is blind, and must be named.
    off = goldfish.run(SLUG, branch="treasure-v2", iterations=20, quiet=True,
                       model_treasures=True, model_combat=False)
    blind_off = set((off.get("meta") or {}).get("treasure_sources_not_modelled") or [])
    if "Goldspan Dragon" in names:
        assert "Goldspan Dragon" in blind_off, (
            "with no combat model an attack trigger produces nothing and must "
            "be named, or a low hoard figure is illegible")
    assert len(blind_off) > len(blind)
