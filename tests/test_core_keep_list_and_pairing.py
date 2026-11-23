"""Two sacred rules, on a deck built in `tmp_path` so they run in the core on every change.

THE KEEP LIST. `protected.json` is the pilot's word that a card stays: edgar-vampires'
draw-v1 cut Vish Kal, his favourite card, and nothing read the OUT side before eight
hours of Forge. Every path that removes a card refuses a protected one — the swap
arithmetic, `try`, opening a branch, proposing and merging — and each refusal is
proven by switching the keep list off and watching the identical call go through.

THE PAIRING. Two lists are read on the same games: same seed, same list, every game
identical and every paired difference exactly zero; same list, two seeds, no row may
earn a verdict. Before 2026-10-04 one shared random stream made the arms of every
comparison independent after their first differing card.

(The fleet versions of these tests read edgar-vampires and goblin-storm and run in the
regression tier; these are the same rules on inline data.)
"""
import json

import pytest

from manamap.pilot import deck_branch, protected, try_swap, validate_protected

SLUG = "keep-deck"
KEEP = "Vish Kal, Blood Arbiter"
COMMANDER = "Edgar Markov"


def _card(name, type_line="Creature — Vampire", cmc=3, text="", commander=False, qty=1):
    return {"name": name, "type_line": type_line, "cmc": cmc, "mana_cost": "{2}{B}",
            "oracle_text": text, "color_identity": ["B"], "colors": ["B"], "quantity": qty,
            "is_commander": commander, "power": "2", "toughness": "2", "legalities": {"commander": "legal"}}


@pytest.fixture
def keep_deck(tmp_path, monkeypatch):
    decks = tmp_path / "decks"
    base = decks / SLUG
    base.mkdir(parents=True)
    spells = [f"Vampire {i}" for i in range(1, 62)]       # 1 + 1 + 61 + 37 = 100
    cards = ([_card(COMMANDER, "Legendary Creature — Vampire Knight", 6, commander=True),
              _card(KEEP, "Legendary Creature — Vampire Bat", 5,
                    "Whenever another creature you control dies, each opponent loses 1 life.")]
             + [_card(n, cmc=2 + i % 4, text="When this creature enters, draw a card." if i % 5 == 0 else "")
                for i, n in enumerate(spells)]
             + [_card("Swamp", "Basic Land — Swamp", 0, qty=37)])
    lines = ([f"1 {COMMANDER} *CMDR*", f"1 {KEEP}"] + [f"1 {n}" for n in spells] + ["37 Swamp"])
    (base / "decklist.txt").write_text("\n".join(lines) + "\n")
    (base / "cards.json").write_text(json.dumps({"deck": SLUG, "decklist_sha256": "a" * 64, "cards": cards}))
    (base / "protected.json").write_text(json.dumps({"cards": [
        {"name": KEEP, "why": "the pilot's favourite card", "at": "2026-10-19"}]}))
    monkeypatch.setattr("manamap.config.DECKS_DIR", decks)
    return base


def test_the_keep_list_refuses_the_swap_arithmetic_and_lifting_it_lets_the_same_call_through(keep_deck, monkeypatch):
    entries = deck_branch._parsed(SLUG)
    with pytest.raises(SystemExit, match="PROTECTED"):
        deck_branch.swap_entries(SLUG, None, entries, KEEP, "Nighthawk Scavenger")
    monkeypatch.setattr(protected, "load", lambda slug: [])
    staged, out_e, _ = deck_branch.swap_entries(SLUG, None, entries, KEEP, "Nighthawk Scavenger")
    assert out_e["name"] == KEEP and KEEP not in {e["name"] for e in staged}


def test_try_refuses_a_protected_cut_before_spending_a_game(keep_deck, monkeypatch):
    from manamap.pilot import diagnostic
    calls = []
    monkeypatch.setattr(diagnostic, "run_on", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(diagnostic, "run", lambda *a, **k: calls.append(1))
    with pytest.raises(SystemExit, match="PROTECTED"):
        try_swap.run(SLUG, [(KEEP, "Nighthawk Scavenger")], iterations=50)
    assert calls == [], "a refused swap must cost nothing"


def test_a_branch_without_the_card_is_never_opened(keep_deck, tmp_path, monkeypatch):
    text = (keep_deck / "decklist.txt").read_text()
    root = tmp_path / "branches"
    monkeypatch.setattr(deck_branch, "branch_root", lambda slug: root)
    with pytest.raises(SystemExit, match="PROTECTED"):
        deck_branch.new(SLUG, "keep-test", text.replace(f"1 {KEEP}\n", "1 Swamp\n", 1))
    assert not (root / "keep-test").exists(), "a refused branch must not be created"


def test_propose_and_merge_read_the_same_refusal(keep_deck, monkeypatch):
    """Both gates fold `protected.refusals` into their blocking list; merge has no
    override at all — the pilot releases a card by editing protected.json."""
    lines = protected.refusals(SLUG, [KEEP, "Vampire 3"])
    assert len(lines) == 1 and KEEP in lines[0]
    assert protected.refusals(SLUG, ["Vampire 3"]) == []
    import inspect
    assert "force" not in inspect.signature(deck_branch.merge).parameters


def test_the_keep_list_validator_fails_every_broken_shape(keep_deck):
    bad = {"cards": [{"name": "Not A Real Card", "why": "x"},
                     {"name": COMMANDER, "why": "x"},
                     {"name": KEEP},
                     {"name": KEEP, "why": "dup", "at": "yesterday"}]}
    errs = " | ".join(validate_protected.validate(SLUG, bad))
    for needle in ("not in", "is the commander", "no why", "listed twice", "not an ISO date"):
        assert needle in errs, needle
    assert validate_protected.validate(SLUG, {"cards": []}), "an empty keep list is a mistake, not a list"


def test_a_list_against_itself_reads_exactly_zero_and_two_seeds_make_no_call(keep_deck):
    from manamap.pilot import diagnostic, net_change
    a = diagnostic.run(SLUG, iterations=400, seed=7, quiet=True, keep_games=True)
    same = diagnostic.run(SLUG, iterations=400, seed=7, quiet=True, keep_games=True)
    assert a["_games"] == same["_games"], "the same seed must play the same games"
    paired = [r for r in net_change.compare_readings(a, same) if "paired" in (r.get("method") or "")]
    assert paired, "no paired row to check"
    for row in paired:
        assert row["ci95_diff"] == [0.0, 0.0], row["measure"]
    other = diagnostic.run(SLUG, iterations=400, seed=8, quiet=True, keep_games=True)
    assert all(r["verdict"] == "noise" for r in net_change.compare_readings(a, other))
