"""The engine graph's parser: a Forge card script -> typed events (2026-11-22).

Inline scripts, copied from the shipped archive, so the unit tier needs no Forge.
The memo `docs/research/engine-model.md` §3.2 is the argument for the vocabulary;
these tests hold the parser to the three rules its module docstring states: the
vocabulary is closed, a filter it cannot read makes an UNPROVEN edge rather than a
silent yes or no, and nothing is guessed.
"""
import pytest

from manamap import config
from manamap.pilot import ability_graph as ag

BLOOD_ARTIST = """Name:Blood Artist
ManaCost:1 B
Types:Creature Vampire
PT:0/1
T:Mode$ ChangesZone | Origin$ Battlefield | Destination$ Graveyard | ValidCard$ Card.Self,Creature.Other | TriggerZones$ Battlefield | Execute$ TrigLoseLife | TriggerDescription$ Whenever CARDNAME or another creature dies, target player loses 1 life and you gain 1 life.
SVar:TrigLoseLife:DB$ LoseLife | ValidTgts$ Player | LifeAmount$ 1 | SubAbility$ DBGainLife
SVar:DBGainLife:DB$ GainLife | Defined$ You | LifeAmount$ 1
"""
EDGAR = """Name:Edgar Markov
ManaCost:3 R W B
Types:Legendary Creature Vampire Knight
PT:4/4
K:Eminence
T:Mode$ SpellCast | ValidCard$ Card.Vampire+Other | ValidActivatingPlayer$ You | TriggerZones$ Command | Execute$ TrigToken | TriggerDescription$ Eminence — Whenever you cast another Vampire spell, if CARDNAME is in the command zone or on the battlefield, create a 1/1 black Vampire creature token.
SVar:TrigToken:DB$ Token | TokenAmount$ 1 | TokenScript$ b_1_1_vampire | TokenOwner$ You
"""
TEYSA = """Name:Teysa, Orzhov Scion
ManaCost:1 W B
Types:Legendary Creature Human Advisor
PT:2/3
T:Mode$ ChangesZone | Origin$ Battlefield | Destination$ Graveyard | ValidCard$ Creature.Black+Other+YouCtrl | TriggerZones$ Battlefield | Execute$ TrigToken | TriggerDescription$ Whenever another black creature you control dies, create a 1/1 white Spirit creature token with flying.
SVar:TrigToken:DB$ Token | TokenScript$ w_1_1_spirit_flying | TokenOwner$ You
"""
RHYSTIC = """Name:Rhystic Study
ManaCost:2 U
Types:Enchantment
T:Mode$ SpellCast | ValidCard$ Card | ValidActivatingPlayer$ Opponent | TriggerZones$ Battlefield | Execute$ TrigDraw | TriggerDescription$ Whenever an opponent casts a spell, you may draw a card unless that player pays {1}.
SVar:TrigDraw:DB$ Draw | Defined$ You | NumCards$ 1 | UnlessCost$ 1 | UnlessPayer$ TriggeredCardController
"""
TREASURE = """Name:Treasure Token
ManaCost:no cost
Types:Artifact Treasure
A:AB$ Mana | Cost$ T Sac<1/CARDNAME/this token> | Produced$ Any | Amount$ 1 | SpellDescription$ Add one mana of any color.
"""
AKROMAS_WILL = """Name:Akroma's Will
ManaCost:3 W
Types:Instant
A:SP$ Charm | MinCharmNum$ 1 | CharmNum$ Count$Compare Y GE1.2.1 | Choices$ DBFly,DBLife
SVar:DBFly:DB$ PumpAll | ValidCards$ Creature.YouCtrl | KW$ Flying & Vigilance & Double Strike
SVar:DBLife:DB$ PumpAll | ValidCards$ Creature.YouCtrl | KW$ Lifelink & Indestructible & Protection from each color
SVar:Y:Count$Valid Card.IsCommander+YouCtrl
"""
UNMAPPED = """Name:Made Up Card
ManaCost:2
Types:Artifact
T:Mode$ ChaosEnsues | ValidCard$ Card | Execute$ TrigX | TriggerDescription$ Whenever chaos ensues, do a thing.
SVar:TrigX:DB$ FrobnicateAll | Defined$ You
"""


def _one(script, kind="trigger"):
    rows = [a for a in ag.parse_script(script)["abilities"] if a["kind"] == kind]
    assert rows, kind
    return rows


def test_blood_artist_consumes_dies_of_self_or_another_creature():
    """A Battlefield→Graveyard trigger is DIES and ONLY DIES (the less specific
    LTB match is not a second ability); the filter keeps both alternatives; the
    chain through SubAbility$ yields a life loss aimed at a player and a gain."""
    rows = _one(BLOOD_ARTIST)
    assert [r["consumes"]["event"] for r in rows] == ["DIES"]
    alts = rows[0]["consumes"]["filters"]
    assert {"any": "Card", "self": True} in alts
    assert {"types": ["Creature"], "other": True} in alts
    produced = [(p["event"], p["actor"], p["amount"]) for p in rows[0]["produces"]]
    assert produced == [("LIFE_LOSS", "player", 1), ("LIFE_GAIN", "you", 1)]
    # The subject lattice: a Vampire token someone else controls dies -> accepted
    # by the second alternative; Blood Artist itself -> accepted by the first.
    token = ag.subject_of({"name": "Vampire Token", "types": ["Creature"], "subtypes": ["Vampire"],
                           "colors": ["B"], "is_token": True})
    assert ag.filter_accepts(alts, token, source="Blood Artist") is True
    me = ag.subject_of({"name": "Blood Artist", "types": ["Creature"], "subtypes": ["Vampire"], "colors": ["B"]})
    assert ag.filter_accepts(alts, me, source="Blood Artist") is True
    land = ag.subject_of({"name": "Swamp", "types": ["Land"], "subtypes": ["Swamp"]})
    assert ag.filter_accepts(alts, land, source="Blood Artist") is False


def test_edgar_eminence_consumes_other_vampire_casts_from_the_command_zone():
    (row,) = _one(EDGAR)
    c = row["consumes"]
    assert c["event"] == "CAST" and c["actor"] == "you"
    assert c["filters"] == [{"any": "Card", "subtypes": ["Vampire"], "other": True}]
    assert row["zone"] == "command"
    assert [(p["event"], p.get("token")) for p in row["produces"]] == [
        ("ETB", "b_1_1_vampire"), ("TOKEN", "b_1_1_vampire")]
    # A Vampire spell is accepted; Edgar's own cast is `Other`; a Human is not a Vampire.
    vamp = ag.subject_of({"name": "Vito", "types": ["Creature"], "subtypes": ["Vampire", "Cleric"], "colors": ["B"]})
    assert ag.filter_accepts(c["filters"], vamp, source="Edgar Markov") is True
    assert ag.filter_accepts(c["filters"], ag.subject_of({"name": "Edgar Markov", "types": ["Creature"],
                                                         "subtypes": ["Vampire"]}), source="Edgar Markov") is False
    human = ag.subject_of({"name": "Teysa", "types": ["Creature"], "subtypes": ["Human"], "colors": ["W", "B"]})
    assert ag.filter_accepts(c["filters"], human, source="Edgar Markov") is False


def test_teysa_refuses_a_white_token_and_takes_a_black_one():
    """`Creature.Black+Other+YouCtrl`: colour is checked against the subject's
    colours, which for a token come from ITS script's `Colors:` line."""
    (row,) = _one(TEYSA)
    alts = row["consumes"]["filters"]
    assert alts == [{"types": ["Creature"], "colors": ["B"], "other": True, "controller": "you"}]
    black = ag.subject_of({"name": "Vampire Token", "types": ["Creature"], "subtypes": ["Vampire"],
                           "colors": ["B"], "is_token": True})
    white = ag.subject_of({"name": "Spirit Token", "types": ["Creature"], "subtypes": ["Spirit"],
                           "colors": ["W"], "is_token": True})
    assert ag.filter_accepts(alts, black, source="Teysa, Orzhov Scion") is True
    assert ag.filter_accepts(alts, white, source="Teysa, Orzhov Scion") is False
    # The colour of a card with no Colors: line is read off its mana cost.
    assert ag.subject_of({"name": "x", "mana_cost": "1 W B", "types": ["Creature"]})["colors"] == ["W", "B"]


def test_rhystic_study_consumes_an_opponents_cast_not_yours():
    (row,) = _one(RHYSTIC)
    assert row["consumes"]["event"] == "CAST" and row["consumes"]["actor"] == "opponent"
    assert row["produces"] == [{"event": "DRAW", "actor": "you", "amount": 1}]
    (edgar,) = _one(EDGAR)
    assert edgar["consumes"]["actor"] == "you", "the same Mode, the other actor"


def test_an_unmapped_mode_is_reported_never_guessed():
    doc = ag.parse_script(UNMAPPED)
    assert doc["abilities"] == [], "no event was invented for a Mode the registry lacks"
    assert doc["unmapped"] == ["effect:FrobnicateAll", "trigger:ChaosEnsues"]
    row = ag.card_abilities("Made Up Card", UNMAPPED)
    assert row["reader"] == "forge" and row["partial"] is True
    # And a filter property outside the whitelist leaves the edge UNPROVEN (None),
    # neither accepted nor refused.
    alts = ag.parse_filter("Creature.IsRemembered+YouCtrl")
    assert alts[0]["unparsed"] == ["IsRemembered"]
    body = ag.subject_of({"name": "x", "types": ["Creature"]})
    assert ag.filter_accepts(alts, body) is None


def test_a_treasure_token_produces_mana_by_its_own_script():
    """A Treasure is not MANA by rule; it is MANA because `res/tokenscripts/
    c_a_treasure_sac.txt` says `AB$ Mana` with a `Sac<1/CARDNAME>` cost — which
    also makes it a DIES producer (it is sacrificed) consuming the BODIES stock."""
    (row,) = _one(TREASURE, kind="activated")
    assert [(p["event"], p["amount"]) for p in row["produces"]] == [("MANA", 1)]
    costs = {k["event"]: k for k in row["costs"]}
    assert set(costs) == {"DIES", "TAP"}
    assert costs["DIES"]["filters"] == [{"self": True}] and costs["DIES"]["stock"] == "BODIES"


def test_a_charm_yields_one_alternative_per_mode_and_is_partial():
    (row,) = _one(AKROMAS_WILL, kind="spell")
    assert row.get("charm") is True
    kws = [p["keywords"] for p in row["produces"]]
    assert kws == [["Flying", "Vigilance", "Double Strike"],
                   ["Lifelink", "Indestructible", "Protection from each color"]]
    card = ag.card_abilities("Akroma's Will", AKROMAS_WILL)
    assert card["loose"] == ["Charm"] and card["partial"] is True and card["unmapped"] == []


def test_every_goldfish_channel_has_an_event_in_the_registry():
    """The goldfish's channels and the graph's events are ONE vocabulary: every
    `model_coverage.CHANNELS` key that reads a card effect names at least one
    registry row, so `try`'s coverage line and the engine agree on what a card is."""
    from manamap.pilot import model_coverage
    channels = set()
    for v in config.ENGINE_EVENTS.values():
        ch = v.get("channel")
        channels.update(ch if isinstance(ch, list) else [ch])
    channels.discard(None)
    missing = [k for k in model_coverage.CHANNELS if k not in channels and k != "target"]
    assert not missing, missing
    assert config.ENGINE_TERMINALS and all(k in config.ENGINE_EVENTS for k in config.ENGINE_TERMINALS)
    for row in config.ENGINE_EVENTS.values():
        assert row.get("triggers") or row.get("effects") or row.get("replacements") \
            or row.get("statics") or row.get("keywords"), "a kind with no Forge name is unreachable"


def test_a_card_with_no_script_falls_back_to_the_goldfish_reader_and_says_so():
    card = {"name": "Imaginary Signet", "type_line": "Artifact", "oracle_text": "{1}, {T}: Add {W}{B}.",
            "mana_cost": "{2}", "cmc": 2.0, "colors": [], "color_identity": ["W", "B"]}
    row = ag.card_abilities("Imaginary Signet", None, card)
    assert row["reader"] == "goldfish"
    assert [p["event"] for a in row["abilities"] for p in a["produces"]] == ["MANA"]
    assert row["abilities"][0]["kind"] == "goldfish"
    nothing = ag.card_abilities("Nothing", None, None)
    assert nothing["reader"] == "none" and nothing["abilities"] == []
    cov = ag.coverage({"a": row, "b": nothing})
    assert cov["readers"] == {"forge": 0, "goldfish": 1, "none": 1}


@pytest.mark.skipif(not __import__("manamap.sim.forge_cards", fromlist=["x"]).installed(),
                    reason="Forge is not installed at FORGE_HOME")
def test_every_registry_forge_name_is_a_name_the_shipped_archive_uses():
    """A registry row naming a Mode or API no script uses is a typo that would
    never fire. Read once over the archive (about a second with the digest cache)."""
    import re
    import zipfile
    from manamap.sim import forge_cards
    seen = set()
    with zipfile.ZipFile(forge_cards._zip(False)) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".txt"):
                continue
            text = zf.read(info).decode("utf-8", errors="replace")
            seen.update(re.findall(r"Mode\$ (\w+)", text))
            seen.update(re.findall(r"(?:DB|SP|AB)\$ (\w+)", text))
            seen.update(re.findall(r"Event\$ (\w+)", text))
            seen.update(m.split(":")[0] for m in re.findall(r"^K:(\w[\w ]*)", text, re.M))
    unknown = []
    for event, row in config.ENGINE_EVENTS.items():
        for field in ("triggers", "effects", "replacements", "statics", "keywords"):
            for m in row.get(field) or ():
                name = m if isinstance(m, str) else m[0]
                if name not in seen:
                    unknown.append(f"{event}.{field}:{name}")
    assert not unknown, unknown
