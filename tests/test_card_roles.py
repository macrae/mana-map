"""Tests for the deckbuilding role classifier (analysis/card_roles.py)."""

import json

import pandas as pd

from manamap.config import (
    CARD_ROLES_PATH,
    ROLE_BODY_FALLBACK,
    ROLE_COVERAGE_TARGET,
    ROLE_SPECIFIC_COVERAGE_TARGET,
)
from manamap.analysis.card_roles import (
    build_roles,
    classify_land,
    classify_row,
)
from tests.conftest import requires_roles


def _card(name, supertype, type_line, oracle_text, keywords="", legal="legal"):
    return {
        "name": name,
        "supertype": supertype,
        "type_line": type_line,
        "oracle_text": oracle_text,
        "keywords": keywords,
        "legal_commander": legal,
    }


# ── mana production is disambiguated by type line ──


def test_artifact_that_taps_for_mana_is_a_rock():
    row = _card("Sol Ring", "Artifact", "Artifact", "{T}: Add {C}{C}.")
    assert "ramp:rock" in classify_row(row)


def test_creature_that_taps_for_mana_is_a_dork():
    row = _card("Llanowar Elves", "Creature", "Creature — Elf Druid", "{T}: Add {G}.")
    assert "ramp:dork" in classify_row(row)


def test_sorcery_that_adds_mana_is_a_ritual():
    row = _card("Dark Ritual", "Instant", "Instant", "Add {B}{B}{B}.")
    assert "ramp:ritual" in classify_row(row)


def test_rock_dork_and_ritual_are_distinguishable():
    """The whole point of the taxonomy — v1's single `ramp` tag scored these alike."""
    rock = classify_row(_card("Signet", "Artifact", "Artifact", "{1}, {T}: Add {W}{U}."))
    dork = classify_row(_card("Elves", "Creature", "Creature — Elf", "{T}: Add {G}."))
    ritual = classify_row(_card("Ritual", "Sorcery", "Sorcery", "Add {B}{B}{B}."))
    assert {"ramp:rock"} <= set(rock)
    assert {"ramp:dork"} <= set(dork)
    assert {"ramp:ritual"} <= set(ritual)
    assert not ({"ramp:dork", "ramp:ritual"} & set(rock))


def test_land_search_is_ramp_land():
    row = _card("Rampant Growth", "Sorcery", "Sorcery",
                "Search your library for a basic land card, put it onto the battlefield tapped.")
    assert "ramp:land" in classify_row(row)


# ── interaction ──


def test_sweeper_does_not_also_count_as_spot_removal():
    """Otherwise a builder counts Wrath of God toward its single-target budget."""
    row = _card("Wrath of God", "Sorcery", "Sorcery", "Destroy all creatures.")
    roles = classify_row(row)
    assert "removal:sweeper" in roles
    assert "removal:spot" not in roles


def test_damage_based_removal_is_removal():
    row = _card("Lightning Bolt", "Instant", "Instant",
                "Lightning Bolt deals 3 damage to any target.")
    assert "removal:damage" in classify_row(row)


def test_counterspell():
    row = _card("Counterspell", "Instant", "Instant", "Counter target spell.")
    assert "counterspell" in classify_row(row)


def test_edict_is_opponent_facing_only():
    """The bare pattern fired on your own activated sacrifice costs, which made
    two thirds of removal:edict sacrifice outlets rather than interaction."""
    edict = _card("Diabolic Edict", "Instant", "Instant",
                  "Target opponent sacrifices a creature.")
    outlet = _card("Viscera Seer", "Creature", "Creature — Vampire Wizard",
                   "Sacrifice a creature: Scry 1.")
    assert "removal:edict" in classify_row(edict)
    assert "removal:edict" not in classify_row(outlet)


def test_sacrifice_cost_is_its_own_role():
    outlet = _card("Ashnod's Altar", "Artifact", "Artifact",
                   "Sacrifice a creature: Add {C}{C}.")
    assert "sac-cost" in classify_row(outlet)


# ── typal templating ──


def test_typal_pump_without_the_word_creatures():
    """Modern templating says "Other Vampires you control get +1/+1".

    Requiring the literal word "creatures" made Legion Lieutenant — a two-mana
    tribal lord, the best cheap card in a Vampire deck — carry no role, no
    mechanical tag, no synergy-graph entry, and a negative embedding cosine to
    its own commander. Every automated signal called it irrelevant.
    """
    lord = _card("Legion Lieutenant", "Creature", "Creature — Vampire Knight",
                 "Other Vampires you control get +1/+1.")
    assert "buff:pump" in classify_row(lord)


def test_generic_anthem_still_matches():
    row = _card("Glorious Anthem", "Enchantment", "Enchantment",
                "Creatures you control get +1/+1.")
    assert "buff:pump" in classify_row(row)


def test_typal_payoffs_that_never_name_the_tribe():
    """"As this enters, choose a creature type" — no text search for a tribe and
    no role query could find these before, for any typal commander."""
    horn = _card("Herald's Horn", "Artifact", "Artifact",
                 "As this artifact enters, choose a creature type. "
                 "Creature spells you cast of the chosen type cost {1} less to cast.")
    assert "payoff:typal" in classify_row(horn)


def test_unrestricted_tutor_does_not_also_count_as_narrow():
    row = _card("Demonic Tutor", "Sorcery", "Sorcery",
                "Search your library for a card, put it into your hand, then shuffle.")
    roles = classify_row(row)
    assert "tutor:unrestricted" in roles
    assert "tutor:narrow" not in roles


# ── the body fallback ──


def test_vanilla_creature_gets_the_body_fallback():
    row = _card("Grizzly Bears", "Creature", "Creature — Bear", "")
    assert classify_row(row) == [ROLE_BODY_FALLBACK]


def test_vanilla_noncreature_gets_nothing():
    """A null is honest — better than a bucket the slot filler would trust."""
    row = _card("Weird Thing", "Sorcery", "Sorcery", "Do something inscrutable.")
    assert classify_row(row) == []


def test_creature_keeps_specific_roles_alongside_the_fallback():
    row = _card("Eternal Witness", "Creature", "Creature — Human Shaman",
                "When this creature enters, return target card from your graveyard to your hand.")
    roles = classify_row(row)
    assert ROLE_BODY_FALLBACK in roles
    assert "recursion" in roles


# ── lands never carry spell roles ──


def test_land_gets_land_roles_only():
    row = _card("Temple Garden", "Land", "Land — Forest Plains",
                "As this land enters, you may pay 2 life. If you don't, it enters tapped. "
                "{T}: Add {G} or {W}.")
    roles = classify_row(row)
    assert all(r.startswith("land:") for r in roles), roles


def test_basic_land_flagged():
    assert "land:basic" in classify_land("Basic Land — Mountain", "{T}: Add {R}.")


def test_tapped_land_is_not_an_untapped_dual():
    roles = classify_land("Land", "This land enters tapped. {T}: Add {W} or {U}.")
    assert "land:tapped" in roles
    assert "land:untapped-dual" not in roles


def test_land_that_ramps_is_not_scored_as_a_spell():
    """A fetchland searches a library but is not `ramp:land` — it's a land slot."""
    roles = classify_row(_card(
        "Windswept Heath", "Land", "Land",
        "{T}, Pay 1 life, Sacrifice this land: Search your library for a Forest or Plains card."))
    assert "ramp:land" not in roles
    assert "land:fetch" in roles


# ── build_roles / meta ──


def test_build_roles_reports_coverage_both_ways():
    df = pd.DataFrame([
        _card("Bear", "Creature", "Creature — Bear", ""),
        _card("Bolt", "Instant", "Instant", "Bolt deals 3 damage to any target."),
    ])
    roles, meta = build_roles(df)
    assert meta["coverage"] == 1.0
    # Only Bolt carries a role beyond the body fallback
    assert meta["specific_coverage"] == 0.5


def test_build_roles_excludes_illegal_cards_from_coverage():
    df = pd.DataFrame([
        _card("Bolt", "Instant", "Instant", "Bolt deals 3 damage to any target."),
        _card("Shahrazad", "Sorcery", "Sorcery", "Inscrutable.", legal="banned"),
    ])
    _, meta = build_roles(df)
    assert meta["commander_legal_count"] == 1
    assert meta["coverage"] == 1.0


def test_build_roles_is_deterministic():
    df = pd.DataFrame([_card("Bolt", "Instant", "Instant", "Bolt deals 3 damage to any target.")])
    assert build_roles(df)[0] == build_roles(df)[0]


# ── the generated artifact ──


@requires_roles
class TestGeneratedRoles:

    def _doc(self):
        with open(CARD_ROLES_PATH) as f:
            return json.load(f)

    def test_meets_coverage_floor(self):
        meta = self._doc()["meta"]
        assert meta["coverage"] >= ROLE_COVERAGE_TARGET, (
            f"role coverage regressed to {meta['coverage']:.1%}"
        )

    def test_meets_specific_coverage_floor(self):
        """Guards against 'classify every creature as a body and call it done'."""
        meta = self._doc()["meta"]
        assert meta["specific_coverage"] >= ROLE_SPECIFIC_COVERAGE_TARGET

    def test_known_cards_land_in_the_right_buckets(self):
        roles = self._doc()["roles"]
        assert "ramp:rock" in roles["Sol Ring"]
        assert "tutor:unrestricted" in roles["Demonic Tutor"]
        assert "removal:sweeper" in roles["Wrath of God"]
        assert "counterspell" in roles["Counterspell"]
