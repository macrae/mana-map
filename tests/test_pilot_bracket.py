"""Tests for the Commander bracket engine (pilot/bracket.py)."""

import pytest

from manamap.config import COMBO_DETAILS_PATH
from manamap.pilot.bracket import (
    assess,
    assumes_other_commander,
    combos_in_deck,
    format_report,
    is_infinite,
    load_reference,
    offending_cards,
)
from manamap.pilot.common import load_deck_cards
from tests.conftest import requires_data, requires_deck, requires_roles


def _details(combos):
    by_card = {}
    for i, combo in enumerate(combos):
        for name in combo["cards"]:
            by_card.setdefault(name, []).append(i)
    return {"combos": combos, "by_card": by_card}


def _combo(cards, bracket=1, produces=None, mv=0):
    return {
        "cards": cards,
        "produces": produces if produces is not None else ["Value"],
        "ci": "",
        "bracket": bracket,
        "mana_value_needed": mv,
        "popularity": 0,
    }


def _flags(names, game_changers=(), banned=()):
    return {
        n: {"game_changer": n in game_changers,
            "legal_commander": "banned" if n in banned else "legal"}
        for n in names
    }


def _assess(names, combos=(), game_changers=(), banned=(), roles=None, commanders=()):
    details = _details(list(combos))
    return assess(names, _flags(names, game_changers, banned), roles or {}, details, commanders)


# ── combos_in_deck ──


def test_combos_in_deck_requires_every_card_present():
    details = _details([_combo(["A", "B"]), _combo(["A", "Z"])])
    assert combos_in_deck(["A", "B"], details) == [0]


def test_combos_in_deck_empty_when_nothing_matches():
    details = _details([_combo(["A", "B"])])
    assert combos_in_deck(["C", "D"], details) == []


def test_combos_in_deck_finds_multiple():
    details = _details([_combo(["A", "B"]), _combo(["B", "C"]), _combo(["A", "Z"])])
    assert combos_in_deck(["A", "B", "C"], details) == [0, 1]


# ── is_infinite ──


def test_is_infinite_matches_infinite_prefix():
    assert is_infinite(_combo(["A", "B"], produces=["Infinite colorless mana"]))


def test_is_infinite_false_for_ordinary_value():
    assert not is_infinite(_combo(["A", "B"], produces=["Lock", "Card draw"]))


# ── the commander assumption (Judge's Desk A-004) ──


def test_combo_assuming_its_own_commander_is_excluded():
    combo = _combo(["Krenko", "Prospector"], produces=["Infinite commander casts"])
    assert assumes_other_commander(combo, commanders={"Zada"})


def test_combo_is_kept_when_a_piece_really_is_the_commander():
    combo = _combo(["Krenko", "Prospector"], produces=["Infinite commander casts"])
    assert not assumes_other_commander(combo, commanders={"Krenko"})


def test_combo_without_the_commander_tell_is_never_excluded():
    combo = _combo(["A", "B"], produces=["Infinite colorless mana"])
    assert not assumes_other_commander(combo, commanders={"Zada"})


def test_excluded_combo_does_not_raise_the_floor():
    """The graph promised an infinite the rules refuse — it must not count."""
    combos = [_combo(["Krenko", "Prospector"], bracket=4,
                     produces=["Infinite commander casts"])]
    report = _assess(["Krenko", "Prospector", "Zada"], combos, commanders={"Zada"})
    assert report["floor"] == 1
    assert report["combo_count"] == 0
    assert len(report["excluded_commander_assumption"]) == 1
    assert any("903.9a" in n for n in report["notes"])


# ── game changers ──


def test_no_game_changers_leaves_floor_at_one():
    assert _assess(["A", "B"])["floor"] == 1


def test_up_to_three_game_changers_forces_bracket_three():
    report = _assess(["A", "B", "C"], game_changers={"A", "B", "C"})
    assert report["floor"] == 3
    assert report["game_changers"] == ["A", "B", "C"]


def test_more_than_three_game_changers_forces_bracket_four():
    report = _assess(list("ABCD"), game_changers=set("ABCD"))
    assert report["floor"] == 4


def test_game_changer_driver_names_the_cards():
    report = _assess(["A"], game_changers={"A"})
    driver = next(d for d in report["drivers"] if d["signal"] == "game_changers")
    assert "A" in driver["detail"]


# ── combo content ──


def test_combo_bracket_raises_the_floor():
    report = _assess(["A", "B", "C"], [_combo(["A", "B"], bracket=3, produces=["Lock"])])
    assert report["floor"] == 3
    assert report["combo_bracket_floor"] == 3


def test_highest_bracket_combo_wins():
    combos = [_combo(["A", "B"], bracket=1), _combo(["B", "C"], bracket=4)]
    assert _assess(["A", "B", "C"], combos)["floor"] == 4


def test_combos_not_fully_contained_are_ignored():
    combos = [_combo(["A", "B", "MISSING"], bracket=4)]
    assert _assess(["A", "B"], combos)["floor"] == 1


# ── two-card infinites ──


def test_late_two_card_infinite_forces_three():
    combos = [_combo(["A", "B"], bracket=1, produces=["Infinite mana"], mv=9)]
    report = _assess(["A", "B"], combos)
    assert report["floor"] == 3


def test_early_two_card_infinite_forces_four():
    combos = [_combo(["A", "B"], bracket=1, produces=["Infinite mana"], mv=4)]
    report = _assess(["A", "B"], combos)
    assert report["floor"] == 4
    assert len(report["two_card_infinites"]) == 1


def test_three_card_infinite_is_not_a_two_card_infinite():
    combos = [_combo(["A", "B", "C"], bracket=1, produces=["Infinite mana"], mv=2)]
    report = _assess(["A", "B", "C"], combos)
    assert report["two_card_infinites"] == []


# ── mass land denial and banned cards ──


def test_mass_land_denial_forces_four():
    report = _assess(["Armageddon", "A"])
    assert report["floor"] == 4
    assert report["mass_land_denial"] == ["Armageddon"]


def test_banned_cards_are_reported_but_do_not_set_a_bracket():
    report = _assess(["A", "B"], banned={"A"})
    assert report["banned"] == ["A"]
    assert any("banned" in n.lower() for n in report["notes"])


# ── tutors are advisory only ──


def test_tutors_are_counted_but_never_raise_the_floor():
    """WotC says 'few tutors' and never gives a number — we don't invent one."""
    roles = {n: ["tutor:unrestricted"] for n in "ABCDEFGH"}
    report = _assess(list("ABCDEFGH"), roles=roles)
    assert len(report["tutors"]) == 8
    assert report["floor"] == 1
    assert not any(d["signal"] == "tutors" for d in report["drivers"])


# ── offending_cards ──


def test_offending_cards_empty_when_within_target():
    report = _assess(["A", "B"])
    assert offending_cards(report, 3) == []


def test_offending_cards_names_both_halves_of_an_infinite():
    combos = [_combo(["A", "B"], produces=["Infinite mana"], mv=2)]
    report = _assess(["A", "B"], combos)
    names = [c["name"] for c in offending_cards(report, 2)]
    assert set(names) == {"A", "B"}


def test_offending_cards_flags_mass_land_denial():
    report = _assess(["Armageddon", "A"])
    cards = offending_cards(report, 2)
    assert cards[0]["name"] == "Armageddon"
    assert "mass land denial" in cards[0]["reasons"]


def test_offending_cards_flags_excess_game_changers():
    report = _assess(["A", "B"], game_changers={"A", "B"})
    assert {c["name"] for c in offending_cards(report, 2)} == {"A", "B"}


def test_offending_cards_ranks_higher_forcing_first():
    combos = [_combo(["A", "B"], produces=["Infinite mana"], mv=2)]
    report = _assess(["A", "B", "Armageddon"], combos, game_changers={"A"})
    ranked = offending_cards(report, 1)
    assert ranked[0]["forces"] >= ranked[-1]["forces"]


# ── report shape ──


def test_report_always_carries_the_not_a_calculator_note():
    assert any("not a calculator" in n for n in _assess(["A"])["notes"])


def test_floor_name_matches_the_ladder():
    report = _assess(["Armageddon"])
    assert report["floor_name"] == "Optimized"


def test_assess_is_deterministic():
    combos = [_combo(["A", "B"], bracket=3)]
    assert _assess(["A", "B"], combos) == _assess(["B", "A"], combos)


def test_format_report_marks_an_over_target_deck():
    report = _assess(["Armageddon"])
    text = format_report("test-deck", report, target=2)
    assert "OVER" in text
    assert "Armageddon" in text


def test_format_report_marks_a_compliant_deck():
    assert "OK" in format_report("test-deck", _assess(["A"]), target=3)


# ── known ground: goblin-storm, whose lines were verified by the rules checker ──


@requires_data
@requires_deck
@requires_roles
@pytest.mark.skipif(not COMBO_DETAILS_PATH.exists(), reason="requires combo details")
class TestGoblinStorm:
    """The engine's answers must agree with the adversarially-verified stacks.

    Stack 005 proved Storm-Kiln Artist + Haze of Rage is a true infinite.
    Stack 004 *refuted* the Krenko/Prospector lines for this deck, because CR
    903.9a scopes the graveyard-to-command-zone action to a commander and Zada
    holds that seat. A bracket engine that disagreed with either would be
    reporting something the rules already settled.
    """

    def _report(self):
        doc = load_deck_cards("goblin-storm")
        cards = [c for c in doc["cards"] if not c.get("is_sideboard")]
        flags, roles, details = load_reference()
        return assess(
            [c["name"] for c in cards],
            flags, roles, details,
            [c["name"] for c in cards if c.get("is_commander")],
        )

    def test_floor_is_driven_by_the_verified_infinite(self):
        report = self._report()
        assert report["floor"] == 4
        pairs = {frozenset(c["cards"]) for c in report["two_card_infinites"]}
        assert frozenset({"Storm-Kiln Artist", "Haze of Rage"}) in pairs

    def test_krenko_lines_are_excluded_not_counted(self):
        report = self._report()
        excluded = {frozenset(c["cards"]) for c in report["excluded_commander_assumption"]}
        assert any("Krenko, Mob Boss" in cards for cards in excluded)
        contained = {frozenset(c["cards"]) for c in report["two_card_infinites"]}
        assert not any("Krenko, Mob Boss" in cards for cards in contained)

    def test_deck_has_no_game_changers_or_land_denial(self):
        report = self._report()
        assert report["game_changers"] == []
        assert report["mass_land_denial"] == []


# ── The report is a deck property, not an invocation's output ─────────────

_STUB_REPORT = {
    "floor": 4, "floor_name": "Optimized", "drivers": [], "notes": [],
    "combo_count": 0, "two_card_infinites": [], "tutors": [],
    "excluded_commander_assumption": [], "banned": [],
}

def test_a_bare_rerun_inherits_the_recorded_target(tmp_path, monkeypatch):
    """`bracket-check <slug>` with no --target must not strip the target.

    It silently rewrote the tracked report without `target`, `within_target` or
    `cut_candidates` — the file stopped answering "is this deck inside its
    bracket" and nothing said so. It happened twice to hapatra in one session,
    each time as a side effect of an agent re-deriving an unrelated figure.
    """
    import json as _json
    from types import SimpleNamespace

    from manamap.pilot import bracket as bracket_mod

    deck = tmp_path / "somedeck"
    deck.mkdir()
    (deck / "bracket_report.json").write_text(_json.dumps(
        {"slug": "somedeck", "floor": 4, "target": 4, "within_target": True,
         "cut_candidates": []}))

    monkeypatch.setattr(bracket_mod, "deck_dir", lambda slug: deck)
    monkeypatch.setattr(bracket_mod, "load_deck_cards",
                        lambda slug: {"cards": [{"name": "Sol Ring",
                                                 "quantity": 1}]})
    monkeypatch.setattr(bracket_mod, "load_reference", lambda: ({}, {}, {}))
    monkeypatch.setattr(bracket_mod, "assess",
                        lambda *a, **k: dict(_STUB_REPORT))

    bracket_mod.main(SimpleNamespace(slug="somedeck", target=None, as_json=False))

    written = _json.loads((deck / "bracket_report.json").read_text())
    assert written["target"] == 4, "the recorded target must survive a bare re-run"
    assert written["within_target"] is True
    assert "cut_candidates" in written


def test_an_explicit_target_still_overrides_the_recorded_one(tmp_path, monkeypatch):
    import json as _json
    from types import SimpleNamespace

    from manamap.pilot import bracket as bracket_mod

    deck = tmp_path / "somedeck"
    deck.mkdir()
    (deck / "bracket_report.json").write_text(_json.dumps(
        {"slug": "somedeck", "floor": 4, "target": 4, "within_target": True}))

    monkeypatch.setattr(bracket_mod, "deck_dir", lambda slug: deck)
    monkeypatch.setattr(bracket_mod, "load_deck_cards",
                        lambda slug: {"cards": [{"name": "Sol Ring",
                                                 "quantity": 1}]})
    monkeypatch.setattr(bracket_mod, "load_reference", lambda: ({}, {}, {}))
    monkeypatch.setattr(bracket_mod, "assess",
                        lambda *a, **k: dict(_STUB_REPORT))
    monkeypatch.setattr(bracket_mod, "offending_cards", lambda *a, **k: [])

    with pytest.raises(SystemExit):
        bracket_mod.main(SimpleNamespace(slug="somedeck", target=3, as_json=False))

    written = _json.loads((deck / "bracket_report.json").read_text())
    assert written["target"] == 3
    assert written["within_target"] is False
