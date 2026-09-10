"""Poison is a second clock on the same seat.

Infect makes every point of damage a source deals to a player a poison counter
(CR 702.90b) — combat or not — and ten counters lose the game (704.5c). Toxic N
adds N counters when the creature deals combat damage. Until 2026-09-10 the
goldfish counted life only, so an infect deck read as a 40-life clock on bodies
that need ten, and a commander whose attack trigger says "THAT CREATURE deals 1
damage to each opponent" (Ingris Stingerquill) was worth nothing on an infect
board and one life per combat on any other.

Two parser faults were found on the way and are locked here: the generic
attack-trigger reader counted her per-attacker ping a second time as a flat
"deals 1 damage" per combat, and the 220-character trigger window ran into her
"{4}: Create a 2/2 … Cadet" activation and credited a free token per attack.
"""

import csv

import pytest

from manamap.config import OUTPUT_CSV_PATH
from manamap.pilot import goldfish
from conftest import requires_data, requires_deck

INGRIS = {
    "name": "Ingris Stingerquill",
    "type_line": "Legendary Creature — Elder Sphinx",
    "power": "1", "toughness": "4",
    "oracle_text": ("Flying\nWhenever a creature you control attacks, that creature "
                    "deals 1 damage to each opponent.\n{4}: Create a 2/2 colorless "
                    "Wizard Soldier creature token named Cadet. Then creatures you "
                    "control gain haste until end of turn."),
}


def _profile(text, type_line="Creature — Phyrexian", power="2"):
    return goldfish.combat_profile({"oracle_text": text, "type_line": type_line, "power": power})


def test_the_keyword_is_read_and_a_grant_is_not():
    assert _profile("Flying\nInfect (This creature deals damage…)")["infect"] is True
    # cards.csv flattens the newline to a space; the deck's cards.json keeps it.
    assert _profile("First strike, protection from red and from white Infect (…)")["infect"] is True
    assert _profile("Trample, infect, indestructible")["infect"] is True
    assert _profile("Deathtouch\nToxic 1 (Players dealt combat damage…)")["toxic"] == 1
    assert _profile("Flying, toxic 3")["toxic"] == 3
    # Grants are priced at nothing and say so in the commit, not here.
    assert _profile("Equipped creature gets +2/+2 and has infect.", "Artifact — Equipment", None)["infect"] is False
    assert _profile("{B}: This creature gains infect until end of turn.")["infect"] is False
    assert _profile("Enchanted creature has infect.", "Enchantment — Aura", None)["infect"] is False


def test_the_per_attacker_ping_is_read_once_and_the_activation_is_not_a_trigger():
    p = goldfish.combat_profile(INGRIS)
    assert p["attack_ping_per_attacker"] == 1
    assert p["attack_damage"] == 0, "the ping was also read as a flat trigger — counted twice"
    assert p["attack_token_bodies"] == 0, "the {4} Cadet activation read as a token per attack"
    assert p["unreadable"] is None


@requires_data
def test_the_corpus_sweep_is_locked():
    """46 infect creatures, 46 toxic cards, ZERO per-attacker pings in the
    2026-08 corpus. A widened pattern, or the next set, moves these on purpose."""
    inf = tox = ping = 0
    with open(OUTPUT_CSV_PATH, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            p = goldfish.combat_profile({"oracle_text": row["oracle_text"],
                                         "type_line": row["type_line"], "power": row.get("power")})
            inf += p["infect"]; tox += bool(p["toxic"]); ping += bool(p["attack_ping_per_attacker"])
    assert (inf, tox, ping) == (46, 46, 0), (inf, tox, ping)


@requires_data
@requires_deck
def test_an_infect_deck_wins_on_poison_and_a_plain_deck_never_does(monkeypatch):
    doc = goldfish.run("ingris-infect", iterations=1500, quiet=True)
    c = doc["metrics"]["combat"]
    assert c["kill_by_poison_rate"] > 0.5, c["kill_by_poison_rate"]
    assert c["mean_poison_by_turn"]["8"] > c["mean_poison_by_turn"]["4"] > 0
    # PROVEN BY REMOVING THE CLOCK: with the threshold out of reach every kill
    # is a life kill and it comes later, so the figure above is the poison.
    monkeypatch.setattr(goldfish, "GOLDFISH_POISON_TO_LOSE", 10 ** 9)
    off = goldfish.run("ingris-infect", iterations=1500, quiet=True)["metrics"]["combat"]
    assert off["kill_by_poison_rate"] == 0.0
    assert off["mean_kill_turn"] > c["mean_kill_turn"]


@requires_data
@requires_deck
def test_a_deck_with_no_poison_source_reads_zero_not_absent():
    """Zero is a measurement here: every source was read and none makes counters."""
    c = goldfish.run("zur-enchantress", iterations=600, quiet=True)["metrics"]["combat"]
    assert c["kill_by_poison_rate"] == 0.0
    assert all(v == 0.0 for v in c["mean_poison_by_turn"].values())
