"""Discover N, the FLOOR reading (CR 701.57), 2026-10-11.

"Exile cards from the top of your library until you exile a nonland card with
mana value N or less. Cast it without paying its mana cost or put it into your
hand. Put the rest on the bottom in a random order."

The goldfish takes the HAND half: the hit goes to hand and is cast later at full
cost, so every figure this moves errs low. Before this a discover spell fed no
channel and no casting loop selected it — Daring Discovery read as a dead card on
the deck it was printed for.
"""

import json
import random

import pytest

from manamap.pilot import goldfish
from manamap.pilot.goldfish_profiles import discover_n

from conftest import requires_data, requires_deck


DARING_TEXT = ("Up to three target creatures can't block this turn. Discover 4. "
               "(Exile cards from the top of your library until you exile a "
               "nonland card with mana value 4 or less. Cast it without paying "
               "its mana cost or put it into your hand. Put the rest on the "
               "bottom in a random order.)")
ZADA_TEXT = ("Whenever you cast an instant or sorcery spell that targets only "
             "Zada, copy that spell for each other creature you control that "
             "the spell could target. Each copy targets a different one of "
             "those creatures.")


def _c(name, text="", cmc=1, type_line="Sorcery", quantity=1, **kw):
    return dict({"name": name, "type_line": type_line, "cmc": cmc,
                 "oracle_text": text, "quantity": quantity,
                 "is_commander": False, "power": None, "toughness": None}, **kw)


# ── the profile ──


def test_discover_n_reads_a_fixed_n_on_a_spell_only():
    assert discover_n(_c("Daring Discovery", DARING_TEXT, 5)) == 4
    # The reminder text is stripped, so "(To discover 10, …" cannot count —
    # the "Discover 10." outside it does.
    assert discover_n(_c(
        "Hit the Mother Lode",
        "Discover 10. If the discovered card's mana value is less than 10, create "
        "a number of tapped Treasure tokens equal to the difference. (To discover "
        "10, exile cards from the top of your library until you exile a nonland "
        "card with mana value 10 or less.)", 7)) == 10
    # OUT OF SCOPE, named in docs/simulation.md: a permanent's discover (entry,
    # death, a land's sacrifice) and discover X.
    assert discover_n(_c("Appraiser", "When this creature enters, if you cast "
                         "it, discover 3.", 4, "Creature — Human Artificer")) == 0
    assert discover_n(_c("Hidden Volcano", "{4}{R}, {T}, Sacrifice this land: "
                         "Discover 4. Activate only as a sorcery.", 0,
                         "Land — Cave")) == 0
    assert discover_n(_c("Bond", "Discover X, where X is the amount of mana "
                         "spent to cast this spell.", 4, "Legendary Sorcery")) == 0
    assert discover_n(_c("Shock", "Shock deals 2 damage to any target.", 1,
                         "Instant")) == 0


@requires_data
def test_the_discover_sweep_is_pinned():
    """CORPUS SWEEP 2026-10-11 (34,888 names): 21 cards say "discover <N>", and
    exactly three are instants or sorceries. If a set adds one, read it."""
    from manamap.pilot import card_pool
    frame = card_pool.load_frame()
    read = sorted({n for n, tl, t in zip(frame["name"], frame["type_line"],
                                         frame["oracle_text"])
                   if discover_n({"type_line": tl,
                                  "oracle_text": t if isinstance(t, str) else ""})})
    assert read == ["Daring Discovery", "Hit the Mother Lode",
                    "Walk with the Ancestors"], read


# ── the walk and the cast ──


def _deck(discover=True, zada=False, prizes=1, beater_cmc=5):
    """Lands, a discover spell, and ONE prize it can find: every other nonland
    card costs 5 or more, so the walk skips it and the prize is the only hit.
    The control swaps the discover text for nothing, so the same card is there
    and is simply never cast."""
    spell_text = ("Target creature can't block this turn. Discover 4."
                  if discover else "Target creature can't block this turn.")
    cards = [
        {"name": "Mountain", "type_line": "Basic Land — Mountain", "cmc": 0,
         "oracle_text": "", "quantity": 36, "is_commander": False,
         "power": None, "toughness": None},
        _c("Dig", spell_text, 5, quantity=10),
        _c("Brick", "", 9, "Artifact", quantity=39 - prizes),
        _c("Prize", "", 4, "Artifact", quantity=prizes),
        _c("Beater", "", beater_cmc, "Creature — Goblin", quantity=14, power="1",
           toughness="1"),
    ]
    if zada:
        cards.append(dict(_c("Zada", ZADA_TEXT, 3,
                             "Legendary Creature — Goblin Ally", power="3",
                             toughness="3"), is_commander=True))
    return {"cards": cards}


def _run(doc, commander_copy=False, iterations=300, model_combat=False):
    library, _ = goldfish.build_library(doc)
    targets = [{"label": "prize", "need": [{"any_of": ["Prize"]}]}]
    rng = random.Random(7)
    return [goldfish.simulate_once(rng, library, 3, targets, 10,
                                   model_combat=model_combat,
                                   commander_copy=commander_copy)
            for _ in range(iterations)]


def _rate(results, by=8):
    return sum(1 for r in results
               if r["target_turns"][0] is not None
               and r["target_turns"][0] <= by) / len(results)


def test_a_discover_spell_is_cast_and_finds_the_card():
    """THE CHANNEL AND ITS CASTING PREDICATE, in one assertion. The prize is
    the only nonland card with mana value 4 or less behind the bricks, so a
    discover that is cast reaches it far sooner than the draw step does — and a
    discover that is read and never cast reads exactly like the control.
    Re-introduce the bug by deleting the discover loop in goldfish_turn."""
    with_it = _rate(_run(_deck(discover=True)))
    without = _rate(_run(_deck(discover=False)))
    assert with_it > without + 0.15, (with_it, without)


def test_a_discovered_card_is_not_a_draw():
    """It goes to HAND and into `seen`, not through `draw_n`: no draw doubler
    touches it and `drawn_extra` does not count it."""
    for r in _run(_deck(discover=True), iterations=100):
        assert not any(r["drawn_extra_by_turn"]), r["drawn_extra_by_turn"]


def test_a_zada_copy_discovers_again():
    """Copied by a Zada-style commander, each copy is one more resolution — the
    same predicate (`copy_fodder` + `commander_copy`) every other copy site
    uses. More resolutions put more cards in hand — given cards to find: with
    one prize the first resolution takes it and every copy walks the deck for
    nothing, which is the rule (a miss exiles everything and finds nothing)."""
    doc = _deck(discover=True, zada=True, prizes=15, beater_cmc=1)
    lib, _ = goldfish.build_library(doc)
    assert next(c for c in lib if c["name"] == "Dig")["copy_fodder"]
    on = _run(doc, commander_copy=True, model_combat=True)
    off = _run(doc, commander_copy=False, model_combat=True)

    def hand(rs, t=10):
        return sum(r["hand_size_by_turn"][t - 1] for r in rs) / len(rs)
    # Measured 8.45 against 7.71 when this shipped.
    assert hand(on) > hand(off) + 0.4, (hand(on), hand(off))


# ── a deck without Discover does not move ──


@requires_data
@requires_deck
@pytest.mark.regression
def test_a_deck_without_discover_is_bit_identical():
    """THE CONTROL, and what made this safe to land as an ALWAYS-ON channel.

    The rng is touched only inside `_discover`, which only a discover spell
    reaches, so a deck without one replays its seeds exactly. Measured on the
    whole fleet at 2,000 games, main vs this change: twelve decks identical,
    goblin-storm (Daring Discovery) the only one that moved. Pinned here
    against heliod's COMMITTED figures, produced by the simulator before it —
    when the model next changes and moves heliod, regenerate it, as the
    staleness test already demands."""
    from manamap.pilot.common import deck_dir, load_deck_cards
    path = deck_dir("heliod") / "goldfish_metrics.json"
    if not path.exists():
        pytest.skip("heliod has no goldfish_metrics.json")
    committed = json.loads(path.read_text(encoding="utf-8"))
    lib, _ = goldfish.build_library(load_deck_cards("heliod"))
    assert not any(c["discover_n"] for c in lib), "heliod now runs a discover card"
    if committed["meta"].get("decklist_sha256") != goldfish.run(
            "heliod", iterations=1, quiet=True)["meta"].get("decklist_sha256"):
        pytest.skip("heliod's list moved since its figures were committed")
    now = goldfish.run("heliod", iterations=committed["meta"]["iterations"],
                       quiet=True)
    assert now["metrics"] == committed["metrics"]
