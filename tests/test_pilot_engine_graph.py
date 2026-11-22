"""The engine as a computed graph of typed events (2026-11-22).

A synthetic Edgar-shaped deck on disk, its Forge scripts in a tiny archive, two
stack scenarios — so the unit tier needs no Forge and no tracked data. Each test
drives `engine_graph.build()` and is red when its rule is dropped: swap two stage
rules, accept a filter without checking a predicate, count a doubler as an edge,
stamp a line from a failing stack, and the matching test fails.
"""
import json
import zipfile

import pytest

from manamap import config
from manamap.pilot import engine_graph as eg
from manamap.sim import forge_cards

SLUG = "test-edgar"

SCRIPTS = {
    "e/edgar_markov.txt": """Name:Edgar Markov
ManaCost:3 R W B
Types:Legendary Creature Vampire Knight
PT:4/4
K:Eminence
T:Mode$ SpellCast | ValidCard$ Card.Vampire+Other | ValidActivatingPlayer$ You | TriggerZones$ Command | Execute$ TrigToken | TriggerDescription$ Eminence
SVar:TrigToken:DB$ Token | TokenAmount$ 1 | TokenScript$ b_1_1_vampire | TokenOwner$ You
""",
    "b/blood_artist.txt": """Name:Blood Artist
ManaCost:1 B
Types:Creature Vampire
PT:0/1
T:Mode$ ChangesZone | Origin$ Battlefield | Destination$ Graveyard | ValidCard$ Card.Self,Creature.Other | TriggerZones$ Battlefield | Execute$ TrigLoseLife | TriggerDescription$ drain
SVar:TrigLoseLife:DB$ LoseLife | ValidTgts$ Player | LifeAmount$ 1 | SubAbility$ DBGainLife
SVar:DBGainLife:DB$ GainLife | Defined$ You | LifeAmount$ 1
""",
    "v/viscera_seer.txt": """Name:Viscera Seer
ManaCost:B
Types:Creature Vampire Wizard
PT:1/1
A:AB$ Scry | Cost$ Sac<1/Creature> | ScryNum$ 1 | SpellDescription$ Scry 1.
AI:RemoveDeck:All
""",
    "v/vampire_nighthawk.txt": """Name:Vampire Nighthawk
ManaCost:1 B B
Types:Creature Vampire Shaman
PT:2/3
K:Flying
K:Deathtouch
""",
    "t/teysa_orzhov_scion.txt": """Name:Teysa, Orzhov Scion
ManaCost:1 W B
Types:Legendary Creature Human Advisor
PT:2/3
T:Mode$ ChangesZone | Origin$ Battlefield | Destination$ Graveyard | ValidCard$ Creature.Black+Other+YouCtrl | TriggerZones$ Battlefield | Execute$ TrigToken | TriggerDescription$ spirit
SVar:TrigToken:DB$ Token | TokenScript$ w_1_1_spirit_flying | TokenOwner$ You
""",
    "r/rhystic_study.txt": """Name:Rhystic Study
ManaCost:2 U
Types:Enchantment
T:Mode$ SpellCast | ValidCard$ Card | ValidActivatingPlayer$ Opponent | TriggerZones$ Battlefield | Execute$ TrigDraw | TriggerDescription$ draw
SVar:TrigDraw:DB$ Draw | Defined$ You | NumCards$ 1
""",
    "a/anointed_procession.txt": """Name:Anointed Procession
ManaCost:3 W
Types:Enchantment
R:Event$ CreateToken | ActiveZones$ Battlefield | ValidToken$ Card.YouCtrl | ReplaceWith$ DoubleToken | Description$ twice the tokens
SVar:DoubleToken:DB$ ReplaceToken | Type$ Amount
""",
    "b/bloodletter_of_aclazotz.txt": """Name:Bloodletter of Aclazotz
ManaCost:1 B B B
Types:Creature Vampire Demon
PT:2/4
K:Flying
R:Event$ LifeReduced | ValidPlayer$ Opponent | PlayerTurn$ True | ReplaceWith$ LoseTwice | ActiveZones$ Battlefield | Description$ twice
SVar:LoseTwice:DB$ ReplaceEffect | VarName$ Amount | VarValue$ ReplaceCount$Amount/Twice
""",
    "s/sol_ring.txt": "Name:Sol Ring\nManaCost:1\nTypes:Artifact\nA:AB$ Mana | Cost$ T | Produced$ C | Amount$ 2\n",
    "s/swamp.txt": "Name:Swamp\nManaCost:no cost\nTypes:Basic Land Swamp\nA:AB$ Mana | Cost$ T | Produced$ B\n",
}
TOKENS = {
    "b_1_1_vampire": "Name:Vampire Token\nManaCost:no cost\nColors:black\nTypes:Creature Vampire\nPT:1/1\n",
    "w_1_1_spirit_flying": "Name:Spirit Token\nManaCost:no cost\nColors:white\nTypes:Creature Spirit\nPT:1/1\nK:Flying\n",
}


def _card(name, type_line, cost, is_commander=False, quantity=1, **kw):
    row = {"name": name, "type_line": type_line, "mana_cost": cost, "cmc": 2.0, "oracle_text": kw.pop("oracle", ""),
           "is_commander": is_commander, "quantity": quantity, "colors": [], "color_identity": ["W", "B", "R"]}
    row.update(kw)
    return row


def _stack(sid, title, you, verdict):
    return {"id": sid, "slug": SLUG, "title": title,
            "scenario": {"board": {"you": you, "opponents": [{"life": 40, "board": []}]}, "hand": [],
                         "mana_available": "{0}", "stack": [], "question": "?"},
            "resolution": {"steps": []}, "checker": {"verdict": verdict, "iterations": 1}}


@pytest.fixture
def synthetic_deck(tmp_path, monkeypatch):
    decks = tmp_path / "decks"
    base = decks / SLUG
    (base / "stacks").mkdir(parents=True)
    monkeypatch.setattr("manamap.config.DECKS_DIR", decks)
    cards = [
        _card("Edgar Markov", "Legendary Creature — Vampire Knight", "{3}{R}{W}{B}", is_commander=True),
        _card("Blood Artist", "Creature — Vampire", "{1}{B}"),
        _card("Viscera Seer", "Creature — Vampire Wizard", "{B}"),
        _card("Vampire Nighthawk", "Creature — Vampire Shaman", "{1}{B}{B}"),
        _card("Teysa, Orzhov Scion", "Legendary Creature — Human Advisor", "{1}{W}{B}"),
        _card("Rhystic Study", "Enchantment", "{2}{U}"),
        _card("Anointed Procession", "Enchantment", "{3}{W}"),
        _card("Bloodletter of Aclazotz", "Creature — Vampire Demon", "{1}{B}{B}{B}"),
        _card("Sol Ring", "Artifact", "{1}", oracle="{T}: Add {C}{C}."),
        _card("Swamp", "Basic Land — Swamp", "", quantity=30, oracle="({T}: Add {B}.)"),
    ]
    (base / "cards.json").write_text(json.dumps({"deck": SLUG, "decklist_sha256": "a" * 64, "cards": cards}))
    (base / "stacks" / "001-seer-artist.json").write_text(json.dumps(_stack(
        "001", "Seer eats a token, Artist drains",
        ["Edgar Markov (4/4)", "Viscera Seer (1/1)", "Blood Artist (0/1)", "Vampire Nighthawk (2/3)"], "pass")))
    (base / "stacks" / "002-teysa.json").write_text(json.dumps(_stack(
        "002", "Teysa makes a spirit", ["Edgar Markov (4/4)", "Viscera Seer (1/1)", "Teysa, Orzhov Scion (2/3)"], "fail")))
    # Forge: a tiny archive and the two token scripts.
    home = tmp_path / "forge"
    (home / "res" / "cardsfolder").mkdir(parents=True)
    with zipfile.ZipFile(home / "res" / "cardsfolder" / "cardsfolder.zip", "w") as z:
        for member, text in SCRIPTS.items():
            z.writestr(member, text)
    (home / "res" / "tokenscripts").mkdir()
    for stem, text in TOKENS.items():
        (home / "res" / "tokenscripts" / f"{stem}.txt").write_text(text)
    monkeypatch.setattr(forge_cards, "FORGE_HOME", home)
    forge_cards._name_index.cache_clear()
    forge_cards._digest_of.cache_clear()
    forge_cards._flags.cache_clear()
    return base


def _stage(doc, name):
    return next((st["cards"] for st in doc["stages"] if st["stage"] == name), [])


def _edges(doc, **match):
    return [e for e in doc["graph"]["edges"] if all(e.get(k) == v for k, v in match.items())]


def test_an_edge_needs_the_event_and_every_predicate(synthetic_deck):
    doc = eg.build(SLUG)
    # Edgar's black Vampire token dies -> Teysa (Creature.Black+Other+YouCtrl) is fed,
    # through Viscera Seer's sacrifice, with the token as the via.
    seer_to_teysa = _edges(doc, kind="feeds", **{"from": "Viscera Seer", "to": "Teysa, Orzhov Scion"})
    assert seer_to_teysa and seer_to_teysa[0]["event"] == "DIES" and seer_to_teysa[0]["proven"] is True
    assert "token:b_1_1_vampire" in seer_to_teysa[0]["via"]
    # Teysa's own white Spirit token never feeds Teysa: the colour predicate fails.
    assert "token:w_1_1_spirit_flying" not in seer_to_teysa[0]["via"]
    # Rhystic Study is fed by the OPPONENT's casts, never by the deck's own.
    into_study = _edges(doc, kind="feeds", to="Rhystic Study")
    assert [e["from"] for e in into_study] == [eg.OPPONENT]
    # Edgar reads a Vampire cast (Nighthawk) and not a Human's (Teysa).
    into_edgar = {e["from"] for e in _edges(doc, kind="feeds", to="Edgar Markov")}
    assert "Vampire Nighthawk" in into_edgar and "Teysa, Orzhov Scion" not in into_edgar
    # Reintroduce the bug: accept every filter -> the Spirit would feed Teysa.
    from manamap.pilot import ability_graph as ag
    spirit = ag.subject_of({"name": "x", "types": ["Creature"], "subtypes": ["Spirit"], "colors": ["W"], "is_token": True})
    teysa_filter = ag.parse_filter("Creature.Black+Other+YouCtrl")
    assert ag.filter_accepts(teysa_filter, spirit, source="Teysa, Orzhov Scion") is False


def test_a_doubler_is_a_modifier_not_an_edge(synthetic_deck):
    doc = eg.build(SLUG)
    scales = {(m["card"], m["event"]) for m in doc["graph"]["scales"]}
    assert ("Anointed Procession", "TOKEN") in scales
    assert ("Bloodletter of Aclazotz", "LIFE_LOSS") in scales
    assert not _edges(doc, kind="feeds", **{"from": "Anointed Procession"}), "a replacement makes no feeds edge"
    assert not [e for e in _edges(doc, kind="feeds", to="Anointed Procession") if e["from"] != eg.OPPONENT]
    # And the component it scales lists it.
    comps = {c["event"]: c for c in doc["components"]}
    assert "Bloodletter of Aclazotz" not in comps.get("DIES", {}).get("producers", [])


def test_the_commander_is_ignition_from_the_command_zone(synthetic_deck):
    doc = eg.build(SLUG)
    edgar = next(n for n in doc["graph"]["nodes"] if n["id"] == "Edgar Markov")
    assert edgar["zone"] == "command"
    assert "Edgar Markov" in _stage(doc, "ignition")
    # A Vampire whose only out-edge is its own implicit cast is NOT ignition: it is
    # a body for the outlet (fodder), and the CAST component counts it as the class.
    assert doc["placed"]["Vampire Nighthawk"]["stage"] == "fodder"
    cast = next(c for c in doc["components"] if c["event"] == "CAST")
    assert "Vampire Nighthawk" in cast["producers"] and "Edgar Markov" in cast["consumers"]
    # Rhystic Study is the other CAST consumer, fed by the opponents alone.
    assert cast["from_opponent"] is True
    # The commander is the single consumer of CAST: a single point of failure.
    kinds = {(s["card"], s["kind"]) for s in doc["spof"]}
    assert ("Edgar Markov", "single_consumer") in kinds


def test_stage_rules_apply_in_registry_order(synthetic_deck):
    doc = eg.build(SLUG)
    p = doc["placed"]
    assert p["Sol Ring"]["stage"] == "mana" and p["Swamp"]["stage"] == "mana"
    assert p["Bloodletter of Aclazotz"]["stage"] == "output", "a multiplier of a terminal is output, never fodder"
    assert p["Rhystic Study"]["stage"] == "fuel"
    assert p["Viscera Seer"]["stage"] == "conversion"
    assert p["Blood Artist"]["stage"] in ("output", "wincon")
    for card, row in p.items():
        if row["stage"] != "unplaced":
            order = [row["stage"]] + row["also"]
            assert order == sorted(order, key=config.ENGINE_STAGE_RULES.index), card
    # Every card in the 99 is placed or explicitly unplaced with a why.
    names = {n["id"] for n in doc["graph"]["nodes"] if n["zone"] != "token"}
    assert set(p) == names
    for u in doc["unassigned"]:
        assert u["why"]
    # The list shape every renderer reads: one block per populated stage, in rule order.
    assert [st["stage"] for st in doc["stages"]] == [s for s in config.ENGINE_STAGE_RULES if _stage(doc, s)]
    assert all(st["label"] and st["what_it_does"].startswith(f"{len(st['cards'])} card") for st in doc["stages"])


def test_a_line_runs_from_ignition_to_a_terminal_ranked_verified_first(synthetic_deck):
    doc = eg.build(SLUG)
    shapes = {tuple(l["events"]): l for l in doc["lines"]}
    loop = shapes.get(("CAST", "BODIES", "DIES", "LIFE_LOSS"))
    assert loop, list(shapes)
    hops = loop["hops"]
    assert hops[0]["cards"] and all("Vampire" in c or c in ("Blood Artist", "Viscera Seer", "Bloodletter of Aclazotz")
                                    for c in hops[0]["cards"])
    assert hops[1]["cards"] == ["Edgar Markov"]
    assert hops[2]["stocks"] == ["BODIES"]
    assert hops[3]["cards"] == ["Viscera Seer"] and hops[4]["cards"] == ["Blood Artist"]
    assert loop["terminal"] == "LIFE_LOSS"
    # The arrow the build page draws: from the first hop's stage to the last's, the
    # events as the noun it carries, the thin hops as `via`.
    assert loop["from"] in {doc["placed"][c]["stage"] for c in hops[0]["cards"]}
    assert loop["to"] in ("output", "wincon")
    assert loop["carries"] == "CAST › BODIES › DIES › LIFE_LOSS"
    assert {"Edgar Markov", "Viscera Seer", "Blood Artist"} <= set(loop["via"])
    # Verified lines come first; a line has at least two hops; ids are dense.
    assert doc["lines"][0]["verified_by"], "a stack-stamped line is ranked first"
    assert all(len(l["events"]) >= 2 for l in doc["lines"])
    assert [l["id"] for l in doc["lines"]] == [f"L{i:02d}" for i in range(1, len(doc["lines"]) + 1)]
    assert len(doc["lines"]) <= config.ENGINE_MAX_LINES


def test_a_stack_stamp_needs_a_passing_scenario_naming_every_card(synthetic_deck):
    doc = eg.build(SLUG)
    loop = next(l for l in doc["lines"] if l["events"] == ["CAST", "BODIES", "DIES", "LIFE_LOSS"])
    assert loop["verified_by"] == ["001"] and loop["stamp"] == "✓"
    # Stack 002 FAILED its checker and names Teysa: the Teysa line stays ◆.
    teysa_lines = [l for l in doc["lines"] if any("Teysa, Orzhov Scion" in h["cards"] for h in l["hops"])]
    assert teysa_lines and all("002" not in l["verified_by"] for l in teysa_lines)
    assert "002" not in doc["stamps"]["stacks"]


def test_a_cut_vertex_is_a_spof(synthetic_deck):
    doc = eg.build(SLUG)
    seer = [s for s in doc["spof"] if s["card"] == "Viscera Seer"]
    assert seer, doc["spof"]
    assert seer[0]["kind"] in ("cut_vertex", "single_producer", "single_consumer")
    assert any(l["id"] in seer[0]["severs"] for l in doc["lines"] if l["events"] == ["CAST", "BODIES", "DIES", "LIFE_LOSS"])
    assert not any(s["card"] in eg.STOCKS for s in doc["spof"]), "a stock is never reported as a point of failure"


def test_component_odds_are_deck_audits_component(synthetic_deck):
    from manamap.pilot.deck_audit import _component
    doc = eg.build(SLUG)
    dies = next(c for c in doc["components"] if c["event"] == "DIES")
    names = {n["id"] for n in doc["graph"]["nodes"] if n["zone"] in ("command", "library")}
    expect = _component({"any_of": [p for p in dies["producers"] if p != "Edgar Markov"]}, names, {}, 99)
    assert dies["activation"]["producers"]["odds"] == expect["odds"]
    assert dies["activation"]["producers"]["size"] == expect["size"]
    assert dies["thin"] == (dies["redundancy"]["producers"] <= config.ENGINE_THIN_GROUP)


def test_a_deck_with_no_forge_falls_back_to_the_goldfish_reader_and_says_so(synthetic_deck, tmp_path, monkeypatch):
    monkeypatch.setattr(forge_cards, "FORGE_HOME", tmp_path / "absent")
    forge_cards._name_index.cache_clear()
    forge_cards._digest_of.cache_clear()
    doc = eg.build(SLUG)
    assert doc["_meta"]["forge_digest"] is None
    assert doc["_meta"]["readers"].get("forge", 0) == 0
    assert doc["coverage"]["readers"].get("goldfish", 0) >= 1, "Sol Ring and the Swamps are read by the goldfish"
    assert doc["placed"]["Sol Ring"]["stage"] == "mana"
    sol = next(n for n in doc["graph"]["nodes"] if n["id"] == "Sol Ring")
    assert sol["coverage"]["reader"] == "goldfish"


def test_the_record_is_byte_identical_across_runs(synthetic_deck):
    a = json.dumps(eg.build(SLUG), sort_keys=True)
    b = json.dumps(eg.build(SLUG), sort_keys=True)
    assert a == b


def test_the_reading_is_migrated_from_v1_and_marked_stale_for_another_list(synthetic_deck):
    fresh = eg.build(SLUG)
    v1 = {"slug": SLUG, "decklist_sha256": "b" * 64, "thesis": "Edgar makes bodies, outlets kill them, Artist bills it.",
          "stages": [{"stage": "ignition", "label": "Ignition", "what_it_does": "w", "cards": ["Edgar Markov"]}],
          "lines": [{"from": "ignition", "to": "wincon", "via": ["Edgar Markov", "Viscera Seer", "Blood Artist"],
                     "note": "the loop", "verified_by": ["001"]},
                    {"from": "fuel", "to": "mana", "via": ["Nothing Here"], "note": "orphan"}],
          "open_questions": [{"q": "does Teysa matter"}], "critic": {"verdict": "pass", "iterations": 2}}
    r = eg.migrate_reading(v1, fresh)
    assert r["thesis"] == v1["thesis"] and r["critic"]["verdict"] == "pass"
    assert r["written_for"] == "b" * 64 and r["stale"] is True, "written for another list"
    assert r["lines"] and r["lines"][0]["note"] == "the loop"
    assert r["lines"][0]["line"] in {l["id"] for l in fresh["lines"]}
    assert r["unmatched_v1_lines"][0]["note"] == "orphan"
    assert "Nothing Here" not in r["names_departed"], "an unmatched line's cards are kept with the line, not flagged"
    # The same list: not stale. A v2 reading is copied forward untouched.
    v1["decklist_sha256"] = "a" * 64
    assert eg.migrate_reading(v1, fresh)["stale"] is False
    v2 = {"_meta": {"v": 2}, "reading": {"written_for": "a" * 64, "thesis": "mine", "lines": [], "stage_overrides": []}}
    again = eg.migrate_reading(v2, fresh)
    assert again["thesis"] == "mine" and again["stale"] is False
    assert r["v1_stages"][0]["cards"] == ["Edgar Markov"], "the v1 placement is kept on the reading"


def test_write_puts_the_reading_beside_a_fresh_record(synthetic_deck):
    doc, path = eg.write(SLUG)
    on_disk = json.loads(path.read_text())
    assert on_disk["_meta"]["v"] == 2 and on_disk["reading"]["written_for"] == "a" * 64
    assert eg.computed_part(on_disk) == eg.computed_part(eg.build(SLUG) | {"reading": None})
    assert on_disk["decklist_sha256"] == "a" * 64, "deck-status reads the stamp at the top level"
