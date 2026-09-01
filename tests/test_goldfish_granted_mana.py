"""A granted mana ability belongs to whoever received it.

Until 2026-08-31 `produced_mana` counted every quoted ability as the card's own:
145 corpus cards, 8 of them sleeved across five decks and five of those in
kinnan. Leyline Immersion — an Aura — read as a five-mana rock.

THE OBVIOUS FIX IS WRONG, which is why these cases are pinned by name. Stripping
quoted text zeroes fifteen cards that are correct: a creature granting
`{T}: Add {G}` to "creatures you control" IS one of those creatures.
"""

import pytest

from manamap.pilot.goldfish import produced_mana

# (name, type line, oracle text, expected) — every one verified by reading the
# card. The two halves are what make this a test rather than a snapshot.
GRANTED_TO_SELF = [
    ("Citanul Hierophants", "Creature — Human Druid",
     'Creatures you control have "{T}: Add {G}."', 1),
    ("Gemhide Sliver", "Creature — Sliver",
     'All Slivers have "{T}: Add one mana of any color."', 1),
    ("Inga and Esika", "Legendary Creature — Human God",
     'Creatures you control have vigilance and "{T}: Add one mana of any color."', 1),
    ("Sachi, Daughter of Seshiro", "Legendary Creature — Snake Shaman",
     'Other Snake creatures you control get +0/+1.\n'
     'Shamans you control have "{T}: Add {G}{G}."', 2),
    ("Dryad Arbor", "Land Creature — Forest Dryad",
     '(This land isn\'t a spell, it\'s affected by summoning sickness, '
     'and it has "{T}: Add {G}.")', 1),
    ("Katilda, Dawnhart Prime", "Legendary Creature — Human Warlock",
     'Each Human you control has "{T}: Add one mana of any color."', 1),
]

GRANTED_AWAY = [
    ("Cryptolith Rite", "Enchantment",
     'Creatures you control have "{T}: Add one mana of any color."', 0),
    ("Thranduil the Strategist", "Legendary Creature — Elf Noble",
     'Other Elves you control have "{T}: Add {G} or {U}."', 0),
    ("Leyline Immersion", "Enchantment — Aura",
     'Enchant legendary creature\nEnchanted creature has ward {2} and '
     '"{T}: Add five mana in any combination of colors."', 0),
    ("Liliana of the Dark Realms", "Legendary Planeswalker — Liliana",
     '−6: You get an emblem with "Swamps you control have '
     '\'{T}: Add {B}{B}{B}{B}.\'"', 0),
    ("Nexos", "Creature — Human Tyranid Advisor",
     'Basic lands you control have "{T}: Add {C}{C}."', 0),
    ("Llanowar Mentor", "Creature — Elf Spellshaper",
     '{G}, {T}, Discard a card: Create a 1/1 green Elf Druid creature token '
     'named Llanowar Elves. It has "{T}: Add {G}."', 0),
    ("Nature's Embrace", "Enchantment — Aura",
     'As long as enchanted permanent is a land, it has '
     '"{T}: Add two mana of any one color."', 0),
    ("Jiang Yanggu, Wildcrafter", "Legendary Planeswalker — Yanggu",
     'Each creature you control with a +1/+1 counter on it has '
     '"{T}: Add one mana of any color."', 0),
    ("Resonating Lute", "Artifact",
     'Lands you control have "{T}: Add two mana of any one color."', 0),
]


@pytest.mark.parametrize("name,type_line,text,want", GRANTED_TO_SELF)
def test_a_card_in_the_class_it_grants_to_keeps_the_ability(name, type_line, text, want):
    assert produced_mana(text, type_line) == want, name


@pytest.mark.parametrize("name,type_line,text,want", GRANTED_AWAY)
def test_a_card_outside_the_class_it_grants_to_does_not(name, type_line, text, want):
    assert produced_mana(text, type_line) == want, name


def test_the_window_does_not_cross_a_clause_boundary():
    """Sachi opens with "OTHER Snake creatures you control get +0/+1" and then
    grants to "Shamans you control", which she is. A flat backward window read
    the stray "Other" and marked her own ability foreign."""
    text = ('Other Snake creatures you control get +0/+1.\n'
            'Shamans you control have "{T}: Add {G}{G}."')
    assert produced_mana(text, "Legendary Creature — Snake Shaman") == 2
    # …while an "Other" that really does introduce the grant still excludes it.
    assert produced_mana('Other Shamans you control have "{T}: Add {G}{G}."',
                         "Legendary Creature — Snake Shaman") == 0


def test_a_token_created_a_sentence_earlier_is_not_this_card():
    """The near window cannot see the token, which is why the token and aura
    markers are checked against a wider one that does cross the boundary."""
    assert produced_mana(
        'Create a 1/1 green Elf Druid creature token named Llanowar Elves. '
        'It has "{T}: Add {G}."', "Creature — Elf Spellshaper") == 0


def test_an_unquoted_ability_is_untouched_by_any_of_this():
    """34,742 cards grant nothing and must read exactly as before."""
    assert produced_mana("{T}: Add {C}{C}.", "Artifact") == 2
    assert produced_mana("{1}, {T}: Add {W}{U}.", "Artifact") == 2
    assert produced_mana("{T}: Add {R}, {G}, or {W}.", "Land") == 1
    assert produced_mana("{T}, Sacrifice this artifact: Add three mana.", "Artifact") == 0


def test_type_line_defaults_to_the_conservative_reading():
    """Omitting it reads every granted ability as foreign — undercounting only
    makes a deck look slower, while overcounting tells the model it can cast
    things it cannot, which is the failure that produced this bug."""
    text = 'Creatures you control have "{T}: Add {G}."'
    assert produced_mana(text) == 0
    assert produced_mana(text, "Creature — Human Druid") == 1
