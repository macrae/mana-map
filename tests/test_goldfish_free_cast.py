"""The free cast (Discover's and Cascade's hits), Cascade, and magecraft on
copies — C2, 2026-10-11.

Discover N (CR 701.57a) and Cascade (702.85a) both say "cast it without paying
its mana cost". C1 took Discover's other option (into hand) as a floor; the
goldfish now casts the hit, routed by its profile through the resolution a paid
cast of it gets (`_free_cast_route` / `_free_cast` in goldfish_turn). Anything
without a free door goes to hand (discover) or the bottom (cascade).

Magecraft reads "cast OR copy"; until this commit only the storm and discover
sites fired it for a Zada copy.
"""

import json
import random

import pytest

from manamap.pilot import goldfish
from manamap.pilot.goldfish_profiles import FREE_CAST_DEPTH_LIMIT, cascade_n

from conftest import requires_data, requires_deck


ZADA_TEXT = ("Whenever you cast an instant or sorcery spell that targets only "
             "Zada, copy that spell for each other creature you control that "
             "the spell could target. Each copy targets a different one of "
             "those creatures.")
CASCADE_REMINDER = ("(When you cast this spell, exile cards from the top of your "
                    "library until you exile a nonland card that costs less. You "
                    "may cast it without paying its mana cost. Put the exiled "
                    "cards on the bottom in a random order.)")
SKA_TEXT = ("This creature gets +1/+0 for each artifact you control.\n"
            "Magecraft — Whenever you cast or copy an instant or sorcery spell, "
            "create a Treasure token.")


def _c(name, text="", cmc=1, type_line="Sorcery", quantity=1, **kw):
    return dict({"name": name, "type_line": type_line, "cmc": cmc,
                 "oracle_text": text, "quantity": quantity,
                 "is_commander": False, "power": None, "toughness": None}, **kw)


def _land(n):
    return {"name": "Mountain", "type_line": "Basic Land — Mountain", "cmc": 0,
            "oracle_text": "", "quantity": n, "is_commander": False,
            "power": None, "toughness": None}


def _zada():
    return dict(_c("Zada", ZADA_TEXT, 3, "Legendary Creature — Goblin Ally",
                   power="3", toughness="3"), is_commander=True)


def _run(doc, iterations=300, seed=7, **kw):
    library, _ = goldfish.build_library(doc)
    rng = random.Random(seed)
    return [goldfish.simulate_once(rng, library, 3, [], 10, **kw)
            for _ in range(iterations)]


def _mean(rs, key):
    return sum(r[key] for r in rs) / len(rs)


# ── the cascade profile ──


def test_cascade_n_reads_the_cards_own_keyword_and_counts_it():
    # Scryfall's line breaks AND the corpus frame's flattened spaces.
    assert cascade_n({"oracle_text": "Haste\nCascade " + CASCADE_REMINDER}) == 1
    assert cascade_n({"oracle_text": "Haste " + CASCADE_REMINDER.join(["", ""])
                      .replace("(", "Cascade (", 1)}) == 1
    assert cascade_n({"oracle_text": "Creatures you control have haste.\n"
                      "Cascade, cascade " + CASCADE_REMINDER}) == 2
    assert cascade_n({"oracle_text": "Cascade, cascade, cascade, cascade"}) == 4
    assert cascade_n({"oracle_text": "Cascade " + CASCADE_REMINDER +
                      " Creatures you control get +1/+0 until end of turn."}) == 1
    assert cascade_n({"oracle_text": "Equipped creature gets +4/+0 and has "
                      "trample.\nEquip {4}\nCascade"}) == 1
    # A GRANT, a condition, a rider and a NAME are not the keyword.
    assert cascade_n({"oracle_text": "The first spell you cast each turn has cascade."}) == 0
    assert cascade_n({"oracle_text": "Commander spells you cast have cascade."}) == 0
    assert cascade_n({"oracle_text": "Delirium — This spell has cascade as long "
                      "as there are four or more card types among cards in your "
                      "graveyard."}) == 0
    assert cascade_n({"oracle_text": "As you cascade, you may put a land card from "
                      "among the exiled cards onto the battlefield tapped."}) == 0
    assert cascade_n({"oracle_text": "Plasma Cascade deals 3 damage to any target."}) == 0
    # Aurora Phoenix: its own keyword counts, the rider about others does not.
    assert cascade_n({"oracle_text": "Flying\nCascade\nWhenever you cast a spell "
                      "with cascade, return this card from your graveyard to "
                      "your hand."}) == 1


#: THE SWEEP, 2026-10-11 (34,888 corpus names): 56 cards say "cascade", these
#: 38 carry the keyword. Grants, conditions, riders and names are the other 18,
#: named in goldfish_profiles beside `cascade_n`.
CASCADE_CORPUS = sorted([
    "Annoyed Altisaur", "Apex Devastator", "Ardent Plea", "Aurora Phoenix",
    "Bituminous Blast", "Bloodbraid Challenger", "Bloodbraid Elf", "Boarding Party",
    "Call Forth the Tempest", "Captured Sunlight", "Demonic Dread", "Deny Reality",
    "Enigma Sphinx", "Enlisted Wurm", "Etherium-Horn Sorcerer", "Ethersworn Sphinx",
    "Forceful Denial", "Garbage Elemental", "Heralds of Tzeentch",
    "Imoti, Celebrant of Bounty", "Ingenuity Engine", "Into the Time Vortex",
    "Kathari Remnant", "Let the Galaxy Burn", "Maelstrom Colossus",
    "Maelstrom Wanderer", "Meteoric Mace", "Natural Reclamation", "Noise Marine",
    "Quandrix, the Proof", "Sakashima's Protege", "Shardless Agent",
    "Stormcaller's Boon", "Sweet-Gum Recluse", "The First Sliver", "Throes of Chaos",
    "Violent Outburst", "Volcanic Torrent"])


@requires_data
def test_the_cascade_sweep_is_pinned():
    """If a set adds a cascade card, or the matcher widens, read the diff."""
    from manamap.pilot import card_pool
    frame = card_pool.load_frame()
    read = sorted({n for n, t in zip(frame["name"], frame["oracle_text"])
                   if cascade_n({"oracle_text": t if isinstance(t, str) else ""})})
    assert read == CASCADE_CORPUS, (sorted(set(read) - set(CASCADE_CORPUS)),
                                    sorted(set(CASCADE_CORPUS) - set(read)))


# ── the dispatcher, one route per kind ──
#
# A discover spell, and behind it bricks (mana value 9, never a hit) and ONE
# kind of hit. Each route either casts the hit (`free_casts`, nothing to hand)
# or does not (`discover_to_hand`), and the routed ones are checked for the
# effect the route is for.

HITS = {
    "creature": _c("Hit", "", 2, "Creature — Goblin", power="2", toughness="2"),
    "draw": _c("Hit", "Target creature gets +1/+0 until end of turn. Draw a card.",
               1, "Instant"),
    "storm": _c("Hit", "Hit deals 1 damage to any target.\nStorm (When you cast "
                "this spell, copy it for each spell cast before it this turn.)",
                1, "Instant"),
    "pump": _c("Hit", "Target creature gets +3/+0 until end of turn.", 1, "Instant"),
    "ritual": _c("Hit", "Add {R}{R}{R}.", 1, "Instant"),
    "other": _c("Hit", "", 2, "Artifact"),
}


def _discover_deck(hit, hits=12, zada=False):
    cards = [_land(36),
             _c("Dig", "Target creature can't block this turn. Discover 4.", 5,
                quantity=12),
             _c("Brick", "", 9, "Artifact", quantity=41 - hits - (1 if zada else 0)),
             # A BOARD for the pump route, at mana value 5 so never a hit.
             _c("Ogre", "", 5, "Creature — Ogre", quantity=10, power="3",
                toughness="3"),
             dict(hit, quantity=hits)]
    if zada:
        cards.append(_zada())
    return {"cards": cards}


@pytest.mark.parametrize("kind,cast", [("creature", True), ("draw", True),
                                       ("storm", True), ("pump", "board"),
                                       ("ritual", True), ("other", False)])
def test_the_dispatcher_routes_each_kind(kind, cast):
    rs = _run(_discover_deck(HITS[kind]), model_combat=True, model_draw=True)
    if cast is True:
        assert _mean(rs, "free_casts") > 0.5, kind
        assert _mean(rs, "discover_to_hand") == 0, kind
    elif cast == "board":
        # A pump is cast free only onto a board, as the pump loop casts it;
        # discovered onto an empty one it goes to hand. Both happen here.
        assert _mean(rs, "free_casts") > 0.1, kind
        assert _mean(rs, "discover_to_hand") > 0.1, kind
    else:
        assert _mean(rs, "free_casts") == 0, kind
        assert _mean(rs, "discover_to_hand") > 0.5, kind


def test_a_free_cast_does_what_its_route_says():
    """Each route's EFFECT, against the same deck whose hits go to hand."""
    base = _run(_discover_deck(HITS["other"]), model_combat=True, model_draw=True)
    # draw: a free-cast cantrip draws (C1's hand half never touched draw_n)
    draw = _run(_discover_deck(HITS["draw"]), model_combat=True, model_draw=True)
    assert _mean(draw, "free_casts") > 0.5
    assert (sum(r["drawn_extra_by_turn"][-1] for r in draw)
            > sum(r["drawn_extra_by_turn"][-1] for r in base))
    # creature: a body joins the board with no mana spent on it
    body = _run(_discover_deck(HITS["creature"]), model_combat=True, model_draw=True)
    assert (sum(r["bodies_by_turn"][5] for r in body)
            > sum(r["bodies_by_turn"][5] for r in base))
    # storm: the copies deal damage
    storm = _run(_discover_deck(HITS["storm"]), model_combat=True, model_draw=True)
    assert (sum(r["damage_by_turn"][-1] for r in storm)
            > sum(r["damage_by_turn"][-1] for r in base))


def test_without_a_board_a_pump_hit_goes_to_hand():
    """The pump route needs model_combat and a board, as the pump loop does —
    off the flag it is not cast free and waits in hand."""
    rs = _run(_discover_deck(HITS["pump"]), model_combat=False, model_draw=True)
    assert _mean(rs, "free_casts") == 0
    assert _mean(rs, "discover_to_hand") > 0.5


# ── the depth guard ──


def _chain_deck():
    """Every nonland card discovers into another: a cmc-3 'Discover 4' sorcery
    targeting only a creature (Zada fodder), so each copy discovers again."""
    return {"cards": [_land(36),
                      _c("Loop", "Target creature can't block this turn. Discover 4.",
                         3, quantity=47),
                      _c("Bear", "", 2, "Creature — Goblin", quantity=16,
                         power="2", toughness="2"),
                      _zada()]}


def test_the_depth_guard_stops_a_chain_and_is_deterministic():
    kw = dict(model_combat=True, model_draw=True, commander_copy=True)
    a = _run(_chain_deck(), iterations=60, seed=3, **kw)
    b = _run(_chain_deck(), iterations=60, seed=3, **kw)
    assert a == b, "a free-cast chain must replay its seed exactly"
    assert max(r["free_cast_depth_max"] for r in a) == FREE_CAST_DEPTH_LIMIT
    assert sum(r["free_cast_depth_hits"] for r in a) > 0
    assert all(r["free_cast_depth_max"] <= FREE_CAST_DEPTH_LIMIT for r in a)


# ── cascade ──


def _cascade_deck(lesser, zada=False, cascader=None):
    """A cascade spell of mana value 4. Every other nonland card costs exactly
    4 — EQUAL, so never a hit — unless `lesser`, which adds cmc-3 bodies."""
    cascader = cascader or _c("Cascader", "Cascade " + CASCADE_REMINDER, 4)
    cards = [_land(36), dict(cascader, quantity=12),
             _c("Equal", "", 4, "Creature — Goblin", quantity=40 if lesser else 51,
                power="2", toughness="2")]
    if lesser:
        cards.append(_c("Lesser", "", 3, "Creature — Goblin", quantity=11,
                        power="1", toughness="1"))
    if zada:
        cards[2]["quantity"] -= 1
        cards.append(_zada())
    return {"cards": cards}


def test_cascade_hits_strictly_less():
    """Mana value LESS than the spell's, not equal: with only cmc-4 cards
    behind a cmc-4 cascade spell, every walk misses. Re-introduce `<=` in
    `_cascade` and the cmc-4 bodies are cast free."""
    equal = _run(_cascade_deck(lesser=False))
    assert _mean(equal, "cascades") > 0.25, "the cascade spell is never cast"
    assert _mean(equal, "free_casts") == 0
    lesser = _run(_cascade_deck(lesser=True))
    assert _mean(lesser, "free_casts") > 0.5


def test_copies_do_not_cascade():
    """A Zada-copied pump with cascade: copies are resolved, never cast, so the
    cascade count is the SAME game by game with the copy ability on and off.
    The pump copies move only this turn's damage, so the two runs replay the
    same draws, casts and walks — and an extra cascade per copy would break the
    equality."""
    pump_cascade = _c("Cascader", "Cascade " + CASCADE_REMINDER +
                      " Target creature gets +1/+0 until end of turn.", 4, "Instant")
    doc = _cascade_deck(lesser=True, zada=True, cascader=pump_cascade)
    lib, _ = goldfish.build_library(doc)
    card = next(c for c in lib if c["name"] == "Cascader")
    assert card["copy_fodder"] and card["cascade"] == 1
    on = _run(doc, model_combat=True, commander_copy=True)
    off = _run(doc, model_combat=True, commander_copy=False)
    assert [r["cascades"] for r in on] == [r["cascades"] for r in off]
    assert sum(r["cascades"] for r in on) > 0
    # ...and the copies did happen: the damage moved.
    assert (sum(r["damage_by_turn"][-1] for r in on)
            > sum(r["damage_by_turn"][-1] for r in off))


def test_a_cascade_card_with_no_other_effect_is_cast():
    """THE CASTING PREDICATE SHIPS WITH THE CHANNEL: a sorcery whose only
    readable text is cascade matched no loop before the cascade loop."""
    rs = _run(_cascade_deck(lesser=True))
    assert _mean(rs, "cascades") > 0.5


# ── magecraft on copies ──


def _ska_deck(spell):
    return {"cards": [_land(36),
                      _c("Storm-Kiln Artist", SKA_TEXT, 4,
                         "Creature — Dwarf Shaman", quantity=4, power="2",
                         toughness="2"),
                      _c("Goblin", "", 1, "Creature — Goblin", quantity=30,
                         power="1", toughness="1"),
                      dict(spell, quantity=28),
                      _zada()]}


def _hoard(rs):
    return sum(r["treasures_by_turn"][-1] for r in rs) / len(rs)


def test_magecraft_fires_for_each_copy_of_a_pump():
    """Storm-Kiln Artist makes a Treasure for every Zada COPY, not only the
    cast. A pump moves no card and no mana, so with the copy ability off the
    game replays the same casts: before the fix the two hoards were EQUAL.
    Re-introduce the bug by deleting `_magecraft` at the top of
    `_resolve_pump`."""
    pump = _c("Pump", "Target creature gets +2/+0 until end of turn.", 1, "Instant")
    kw = dict(model_combat=True, model_treasures=True, iterations=300)
    on = _run(_ska_deck(pump), commander_copy=True, **kw)
    off = _run(_ska_deck(pump), commander_copy=False, **kw)
    assert _hoard(on) > _hoard(off) + 1.0, (_hoard(on), _hoard(off))


def test_magecraft_fires_for_each_copy_of_a_cantrip():
    """The draw site. Copies DRAW, so the copy-on run sees more cards and casts
    more spells, and that alone raises its hoard: the deck is built so it
    cannot raise it far — eight cantrips, bricks for the rest of the draws, and
    eight cheap Artists so the payoff is out early. Measured when this
    shipped (300 games): 5.91 against 1.59 with the fix, 3.46 against 1.59
    without it, so the margin of 3 sits between the two. Re-introduce by
    deleting `_magecraft(others)` in `_resolve_draw`."""
    cantrip = _c("Cantrip", "Target creature gets +1/+0 until end of turn. "
                 "Draw a card.", 1, "Instant")
    doc = {"cards": [_land(36),
                     _c("Storm-Kiln Artist", SKA_TEXT, 2, "Creature — Dwarf Shaman",
                        quantity=8, power="2", toughness="2"),
                     _c("Goblin", "", 1, "Creature — Goblin", quantity=40,
                        power="1", toughness="1"),
                     _c("Brick", "", 20, "Artifact", quantity=7),
                     dict(cantrip, quantity=8), _zada()]}
    kw = dict(model_combat=True, model_treasures=True, model_draw=True,
              iterations=300)
    on = _run(doc, commander_copy=True, **kw)
    off = _run(doc, commander_copy=False, **kw)
    assert _hoard(on) > _hoard(off) + 3.0, (_hoard(on), _hoard(off))


# ── decks without any of it do not move ──


@requires_data
@requires_deck
@pytest.mark.regression
@pytest.mark.parametrize("slug", ["heliod", "meren-recursion", "gishath"])
def test_a_deck_without_a_free_cast_or_a_copied_magecraft_is_bit_identical(slug):
    """THE CONTROL. heliod has none of it; meren-recursion runs Sedgemoor Witch
    (magecraft, no copy channel); gishath runs Trumpeting Carnosaur and
    Pantlaza (a permanent's discover, not read). Pinned against their
    COMMITTED figures, produced by the simulator before this change. Measured
    on the whole fleet at 2,000 games, main vs this change: thirteen decks
    identical, goblin-storm the only one moved. When the model next moves one
    of these, regenerate it, as the staleness test already demands."""
    from manamap.pilot.common import deck_dir, load_deck_cards
    path = deck_dir(slug) / "goldfish_metrics.json"
    if not path.exists():
        pytest.skip(f"{slug} has no goldfish_metrics.json")
    committed = json.loads(path.read_text(encoding="utf-8"))
    lib, _ = goldfish.build_library(load_deck_cards(slug))
    assert not any(c["discover_n"] or c["cascade"] for c in lib), slug
    if committed["meta"].get("decklist_sha256") != goldfish.run(
            slug, iterations=1, quiet=True)["meta"].get("decklist_sha256"):
        pytest.skip(f"{slug}'s list moved since its figures were committed")
    now = goldfish.run(slug, iterations=committed["meta"]["iterations"], quiet=True)
    assert now["metrics"] == committed["metrics"]
    assert "free_cast" not in now["metrics"]
