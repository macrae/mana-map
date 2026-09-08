"""Rank a pile against a MEASURED failure, not against a taste.

Hexproof and shroud stop targeted removal and nothing else. Phasing stops both.
Indestructible stops neither exile nor bounce. A pile of protection cards sorted
by feel puts all three side by side; sorted against heliod's 119 recorded
commander departures — 48% of them WIPES — it does not.

This is the cheapest rigorous instrument in the repo: it costs no games, and it
answers the only empirical part of "is this card worth a slot".
"""

import pytest

from manamap.sim import failure


def _decomp(mass, targeted, other, graveyard, exile, departures=None, games=340):
    tot = mass + targeted + other
    return {"slug": "x", "commander": "C", "games": games,
            "resolutions": tot * 2, "departures": departures or tot,
            "zones": {"Graveyard": graveyard, "Exile": exile},
            "causes": {k: {"n": v, "share": v / tot, "ci95": [0, 1]}
                       for k, v in (("mass", mass), ("targeted", targeted),
                                    ("other", other), ("combat", 0))}}


D = _decomp(mass=57, targeted=41, other=21, graveyard=86, exile=33)


def test_phasing_answers_both_classes_and_hexproof_only_one():
    """THE DISTINCTION THAT PICKED THE CARDS. Phase out in response and a
    targeted spell fizzles for want of a legal target AND a wipe finds nothing;
    hexproof does nothing about a wipe, because a wipe does not target."""
    guardian = failure.coverage("Flash Vigilance When this creature enters, any "
                                "number of other target creatures you control "
                                "phase out.", D)
    greaves = failure.coverage("Equipped creature has haste and shroud.", D)
    assert guardian["stops"] == ["targeted", "mass"]
    assert greaves["stops"] == ["targeted"]
    assert guardian["share_of_losses"] > 2 * greaves["share_of_losses"]


def test_indestructible_is_discounted_by_the_exiles_it_cannot_stop():
    """Getting this wrong would price Mithril Coat identically to Guardian of
    Faith. Indestructible answers DESTROY; 33 of 119 departures were EXILE."""
    coat = failure.coverage("Flash Indestructible When this enters, attach it to "
                            "target legendary creature you control.", D)
    guardian = failure.coverage("all creatures you control phase out.", D)
    assert coat["stops"] == ["targeted", "mass"]
    assert coat["share_of_losses"] < guardian["share_of_losses"]
    assert "exile" in coat["caveat"]
    # exactly the graveyard fraction of the full share
    assert coat["share_of_losses"] == pytest.approx(
        guardian["share_of_losses"] * 86 / 119, abs=0.002)


def test_a_card_that_answers_nothing_scores_nothing_rather_than_zero():
    """ABSENT, not 0.0. A draw spell is not a protection card that happens to be
    bad at it, and a 0.0 in the column would read as one."""
    assert failure.coverage("Whenever an opponent casts a spell, you may draw a "
                            "card unless they pay {1}.", D) is None


def test_mass_is_detected_by_coincidence_not_by_a_list_of_sweeper_names():
    """A first cut matched sweeper NAMES and left 35% unattributed; reading those
    showed several permanents leaving in the same window, which is a wipe
    whatever named it. Counting how many OTHER permanents left on the same turn
    catches the wipe the name list has never heard of."""
    assert failure.MASS_THRESHOLD == 3
    log = (
        "Turn: Turn 5 (Ai(1)-mm-x)\n"
        "Resolve Stack: C - Creature 4 / 4\n"
        "Turn: Turn 9 (Ai(2)-mm-y)\n"
        "Add To Stack: Ai(2)-mm-y cast Nobody Has Heard Of This Wipe\n"
        "Zone Change: Some Token (1) was put into Graveyard from Battlefield.\n"
        "Zone Change: Another Thing (2) was put into Graveyard from Battlefield.\n"
        "Zone Change: A Third (3) was put into Graveyard from Battlefield.\n"
        "Zone Change: C (100) was put into Graveyard from Battlefield.\n")
    import pathlib
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp) / "decks" / "x" / "sim" / "logs" / "run"
        d.mkdir(parents=True)
        (d / "part-00.log").write_text(log)
        import manamap.sim.failure as f
        old = f.DECKS_DIR
        f.DECKS_DIR = pathlib.Path(tmp) / "decks"
        try:
            got = f.commander_departures("x", "C")
        finally:
            f.DECKS_DIR = old
    assert got["departures"] == 1
    assert got["causes"]["mass"]["n"] == 1, (
        "four permanents left on one turn — that is a wipe whether or not the "
        "spell's name is in any list")


def test_an_experiments_arm_B_is_excluded():
    """`b-part-*.log` is a DIFFERENT LIST. Folding it in would decompose two
    decks as one, and the whole point is a decomposition of THIS deck."""
    import inspect
    src = inspect.getsource(failure._logs)
    assert "b-part" in src and "not in" in src


# ── The keyword must land on something YOU control ────────────────────────
#
# The first cut matched a protection word anywhere in the text. On a real
# 46-card pile that scored six of the thirteen top-ranked cards for protecting
# something other than the commander — in a ranking about to decide a purchase.

@pytest.mark.parametrize("name,text", [
    ("Thassa, God of the Sea", "Indestructible\nAs long as your devotion to blue "
     "is less than five, Thassa isn't a creature."),
    ("Aegis of the Gods", "You have hexproof."),
    ("Dragonlord Ojutai", "Flying\nDragonlord Ojutai has hexproof as long as "
     "it's untapped."),
    ("Darksteel Sentinel", "Flash\nVigilance\nIndestructible"),
    ("Defender of Law", "Flash\nProtection from red"),
    ("Seht's Tiger", "Flash\nWhen this creature enters, you gain protection from "
     "the color of your choice until end of turn."),
])
def test_protection_a_card_gives_ITSELF_or_YOU_scores_nothing(name, text):
    """Six real cards from one pile. A God that is itself indestructible does
    not keep the commander alive, and 'You have hexproof' is the player."""
    assert failure.coverage(text, D) is None, name


@pytest.mark.parametrize("name,text", [
    ("Teferi's Protection", "Your life total can't change. You gain protection "
     "from everything. All permanents you control phase out."),
    ("Guardian of Faith", "Flash\nVigilance\nWhen this creature enters, any number "
     "of other target creatures you control phase out."),
    ("Spectacular Spider-Man", "Flash\n{1}: Spectacular Spider-Man gains flying "
     "until end of turn.\n{1}, Sacrifice Spectacular Spider-Man: Creatures you "
     "control gain hexproof and indestructible until end of turn."),
    ("Lightning Greaves", "Equipped creature has haste and shroud.\nEquip {0}"),
])
def test_protection_GRANTED_to_what_you_control_does_score(name, text):
    assert failure.coverage(text, D) is not None, name


def test_permanents_counts_as_well_as_creatures():
    """THE FALSE NEGATIVE THE FIX INTRODUCED, caught by checking the new rule
    against a card whose answer was already known. Teferi's Protection — the
    card already in this deck and the reference case for the whole column —
    phases out PERMANENTS, and a creature-only grant list scored it at nothing.

    That is the only way a false negative in a filter like this ever surfaces:
    a positive control that must keep passing."""
    tp = failure.coverage("All permanents you control phase out.", D)
    assert tp is not None and tp["stops"] == ["targeted", "mass"]


def test_a_keyword_on_one_line_does_not_bless_a_grant_on_another():
    """Sentence-scoped. Spectacular Spider-Man carries Flash on line one and
    grants hexproof on line three; a card carrying Flash and granting NOTHING
    must not inherit a score from the keyword sitting elsewhere."""
    assert failure.coverage("Flash\nVigilance\nWhen this creature enters, draw a "
                            "card.", D) is None
