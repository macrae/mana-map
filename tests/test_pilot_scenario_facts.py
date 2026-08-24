"""scenario-facts: the deterministic brief that makes recalled figures unnecessary.

Every assertion here corresponds to an error that reached an agent brief during
the Vol. 008 session and was refused by the agent rather than written down.
"""



from conftest import requires_deck
from manamap.pilot import scenario_facts as sf


# ── Board parsing ────────────────────────────────────────────────────────


def test_tokens_are_bodies_not_furniture():
    """In a sacrifice deck the tokens ARE the bodies.

    The first cut of this module filtered tokens out as "not cards", which made
    yawgmoth's 002 and 003 look identical and erased the extra Human Soldier —
    the whole reason their matching totals are not comparable.
    """
    split = sf.board_bodies([
        "Yawgmoth, Thran Physician (2/4, untapped, no counters)",
        "Nest of Scarabs (enchantment)",
        "Insect token B (1/1 black Insect, no counters)",
        "Human Soldier token (1/1 white, no counters)",
        "Swamp (untapped)",
    ])
    assert split["creature_bodies"] == [
        "Yawgmoth, Thran Physician", "Insect token B", "Human Soldier token"]
    assert split["other_permanents"] == ["Nest of Scarabs"]
    assert split["lands"] == ["Swamp"]


def test_an_already_paid_cost_is_not_on_the_battlefield():
    """The annotated body is LISTED and NOT available — it changes every bound."""
    split = sf.board_bodies([
        "Insect token A (1/1 black Insect) — already sacrificed to pay the cost of "
        "the ability now on the stack",
        "Insect token B (1/1 black Insect, no counters)",
    ])
    assert split["spent_paying_a_cost"] == ["Insect token A"]
    assert split["creature_bodies"] == ["Insect token B"]


# ── The arithmetic that reached a brief wrong ────────────────────────────


def test_per_opponent_and_pod_total_are_stated_separately():
    """"28 from each opponent" was 7 per opponent and 28 across four seats."""
    opps = [{"seat": f"opponent_{c}", "life": 40} for c in "abcd"]
    facts = sf.drain_arithmetic(opps)
    assert facts["opponents"] == 4
    assert facts["opposing_life_total"] == 160
    assert "per seat" in facts["note"] and "across the pod" in facts["note"]


def test_both_board_shapes_are_read():
    """Seven decks use `opponents: [...]`; yawgmoth uses `opponent_a..d`."""
    listed = sf.opponents_of(
        {"board": {"opponents": [{"name": "P2", "life": 33, "board": ["a 4/4"]}]}})
    assert listed == [{"seat": "P2", "life": 33, "board": ["a 4/4"]}]

    keyed = sf.opponents_of({
        "board": {"you": [], "opponent_a": ["no permanents"], "opponent_b": ["x"]},
        "extras": {"life_totals": {"you": 39, "opponent_a": 40, "opponent_b": 33}},
    })
    assert [o["seat"] for o in keyed] == ["opponent_a", "opponent_b"]
    assert [o["life"] for o in keyed] == [40, 33]
    # In the keyed shape the value under the key IS that seat's board.
    assert [o["board"] for o in keyed] == [["no permanents"], ["x"]]


def test_decision_scenarios_name_their_seats():
    """Decision boards carry `seat`, not `name` — reading only `name` numbered
    every seat generically and threw away the archetype the coach wrote."""
    opps = sf.opponents_of({"board": {"opponents": [
        {"seat": "A — Azorius flash control", "life": 35, "board": "four untapped lands"}]}})
    assert opps[0]["seat"] == "A — Azorius flash control"
    assert opps[0]["board"] == "four untapped lands"


def test_a_string_extras_block_does_not_throw():
    """`extras` is free-form and decisions write it as a string, not a dict."""
    assert sf.opponents_of({"board": {"opponents": []}, "extras": "prose here"}) == []


# ── Sibling comparability ────────────────────────────────────────────────


def test_same_body_count_still_reports_what_differs_both_ways():
    """Two boards can match on count and still answer different questions.

    A one-directional diff hid exactly this: 002 reaches three bodies with a Human
    Soldier where 003 uses Zulaport, because Bastion is an enchantment and cannot
    be a body itself. Reconciling that by hand cost two rounds of stack 008.
    """
    scenarios = {
        "002": {"board": {"you": ["Yawgmoth (2/4)", "Insect token B (1/1)",
                                  "Human Soldier token (1/1 white)",
                                  "Bastion of Remembrance (enchantment)"]}},
        "003": {"board": {"you": ["Yawgmoth (2/4)", "Insect token B (1/1)",
                                  "Zulaport Cutthroat (1/1)"]}},
    }
    [sib] = sf.comparable_siblings("002", scenarios)
    assert sib["stack"] == "003"
    assert sib["same_body_count"] is True
    assert sib["only_on_that_board"] == ["Zulaport Cutthroat"]
    assert sib["only_on_this_board"] == ["Human Soldier token"]


# ── Membership ───────────────────────────────────────────────────────────


def test_membership_names_what_left_the_deck():
    got = sf.membership(["Nest of Scarabs", "Ad Nauseam"], {"Nest of Scarabs"})
    assert got["in_the_deck"] == ["Nest of Scarabs"]
    assert got["NOT_IN_THE_DECK"] == ["Ad Nauseam"]


def test_tokens_are_not_reported_as_missing_cards():
    """A warning that fires on every scenario is one an agent learns to skip.

    Tokens are never in a decklist, so membership-checking them made the
    "not in the deck" note fire with nothing wrong — burying the one case that
    matters, a real card of yours that has left the 99.
    """
    got = sf.membership(["Insect token A", "Human Soldier token", "Ad Nauseam"],
                        {"Nest of Scarabs"})
    assert got["NOT_IN_THE_DECK"] == ["Ad Nauseam"]
    assert got["tokens_not_checked"] == ["Human Soldier token", "Insect token A"]


# ── Against the real committed decks ─────────────────────────────────────


@requires_deck
def test_runs_on_every_committed_deck_with_stacks():
    from manamap.config import DECKS_DIR
    ran = 0
    for deck in sorted(DECKS_DIR.iterdir()):
        if not (deck / "stacks").is_dir() or not (deck / "cards.json").exists():
            continue
        facts = sf.analyze(deck.name)
        assert facts["slug"] == deck.name
        for sid, s in facts["stacks"].items():
            assert "drain_arithmetic" in s and "card_membership" in s
        ran += 1
    assert ran >= 1


# ── A bare board entry is still a creature ───────────────────────────────────

def test_a_creature_named_without_a_pt_is_still_a_body():
    """A v1 board entry is PROSE. "Mondrak, Glory Dominus" carries no `4/4`, so the
    classifier could only ever go on what was written and filed 46 of 127 board
    entries across seven decks as non-creatures while the deck's own `cards.json`
    typed them Creature.

    The gate never caught it — `validate_stack` unions all three buckets, so
    membership still got checked. What it corrupted is the BRIEF every resolver is
    told to read first, and the sibling-claim body counts computed from it."""
    from manamap.pilot.scenario_facts import board_bodies
    bare = ["Mondrak, Glory Dominus", "Anointed Procession", "five lands, all untapped"]
    blind = board_bodies(bare)
    assert blind["creature_bodies"] == [], "without the deck it can only read prose"
    seeing = board_bodies(bare, {"Mondrak, Glory Dominus"})
    assert seeing["creature_bodies"] == ["Mondrak, Glory Dominus"]
    assert seeing["other_permanents"] == ["Anointed Procession"]
    assert seeing["lands"] == ["five lands, all untapped"]


def test_prose_always_beats_the_card_database():
    """The board describes a game state; the decklist is not entitled to overrule
    it. An explicit annotation must win, or a card the board says was sacrificed
    comes back as a body."""
    from manamap.pilot.scenario_facts import board_bodies
    cr = {"Mondrak, Glory Dominus"}
    spent = board_bodies(["Mondrak, Glory Dominus (already sacrificed)"], cr)
    assert spent["spent_paying_a_cost"] == ["Mondrak, Glory Dominus"]
    assert spent["creature_bodies"] == []


@requires_deck
def test_a_god_is_not_promoted_on_its_type_line_alone():
    """Purphoros is a Legendary Enchantment Creature whose own text says he "isn't a
    creature" below five devotion to red. Promoting him on the type line would
    assert a devotion count nobody wrote down — and he is on two of this repo's
    boards, so it is not hypothetical."""
    from manamap.pilot.scenario_facts import unconditional_creatures
    cr = unconditional_creatures("edgar-vampires")
    assert "Mondrak, Glory Dominus" in cr
    assert "Purphoros, God of the Forge" not in cr


@requires_deck
def test_the_fleet_no_longer_miscounts_its_boards():
    """The measurable form of the bug: 46 creatures filed as non-creatures before,
    and only the conditional ones after."""
    import json
    from pathlib import Path
    from manamap.config import DECKS_DIR
    from manamap.pilot.scenario_facts import board_bodies, your_board, unconditional_creatures

    missed = []
    for deck in sorted(DECKS_DIR.iterdir()):
        if not (deck / "cards.json").exists():
            continue
        cr = unconditional_creatures(deck.name)
        types = {c["name"]: (c.get("type_line") or "")
                 for c in json.loads((deck / "cards.json").read_text())["cards"]}
        for sp in sorted((deck / "stacks").glob("*.json")):
            sc = json.loads(sp.read_text()).get("scenario") or {}
            for n in board_bodies(your_board(sc), cr)["other_permanents"]:
                if "creature" in types.get(n, "").lower():
                    missed.append((deck.name, n))
    # Only conditional creatures may remain, and they must be conditional for a
    # reason the card states rather than because we gave up on them.
    for slug, name in missed:
        cards = json.loads((DECKS_DIR / slug / "cards.json").read_text())["cards"]
        oracle = next(c.get("oracle_text", "") for c in cards if c["name"] == name)
        assert "isn't a creature" in oracle.lower() or "is not a creature" in oracle.lower(), (
            f"{slug}: {name} is a creature by type line and is not filed as a body")
