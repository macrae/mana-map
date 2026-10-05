"""`try` and the pilot's keep list (2026-10-04).

The keep list exists because edgar-vampires/draw-v1 cut Vish Kal, Blood Arbiter —
the pilot's favourite card — and nothing read the OUT side before eight hours of
Forge. `try` exists because a swap idea took a day to reach an answer. Both are
driven here through the production functions, on real decks, and each refusal is
proven by switching the keep list off and watching the same call go through.
"""
import json

import pytest

from conftest import requires_data, requires_deck

from manamap import config
from manamap.pilot import deck_branch, protected, try_swap, validate_protected

EDGAR = "edgar-vampires"
VISH = "Vish Kal, Blood Arbiter"

requires_edgar_keep = pytest.mark.skipif(
    not (config.DECKS_DIR / EDGAR / "protected.json").exists()
    or not (config.DECKS_DIR / EDGAR / "cards.json").exists(),
    reason="requires edgar-vampires with its protected.json")


def _some_other_card(slug):
    entries = deck_branch._parsed(slug)
    return next(e["name"] for e in entries
                if not e.get("is_commander") and e["name"] != VISH
                and "Land" not in e["name"])


@requires_edgar_keep
def test_the_keep_list_validates_on_the_real_deck():
    doc = json.loads((config.DECKS_DIR / EDGAR / "protected.json").read_text())
    assert validate_protected.validate(EDGAR, doc) == []
    assert VISH in protected.names(EDGAR)


@requires_edgar_keep
def test_the_validator_fails_the_broken_shapes():
    bad = {"cards": [{"name": "Not A Real Card", "why": "x"},
                     {"name": "Edgar Markov", "why": "x"},
                     {"name": VISH},
                     {"name": VISH, "why": "dup", "at": "yesterday"}]}
    errs = " | ".join(validate_protected.validate(EDGAR, bad))
    assert "not in edgar-vampires's 99" in errs
    assert "is the commander" in errs
    assert "no why" in errs
    assert "listed twice" in errs
    assert "not an ISO date" in errs
    assert validate_protected.validate(EDGAR, {"cards": []})


@requires_edgar_keep
def test_a_protected_cut_is_refused_by_the_swap_arithmetic(monkeypatch):
    entries = deck_branch._parsed(EDGAR)
    with pytest.raises(SystemExit, match="PROTECTED"):
        deck_branch.swap_entries(EDGAR, None, entries, VISH, "Morbid Opportunist")
    # PROVEN BY SWITCHING IT OFF: the identical call goes through.
    monkeypatch.setattr(protected, "load", lambda slug: [])
    staged, out_e, _ = deck_branch.swap_entries(EDGAR, None, entries, VISH, "Morbid Opportunist")
    assert out_e["name"] == VISH
    assert VISH not in {e["name"] for e in staged}


@requires_edgar_keep
@requires_data
def test_try_refuses_a_protected_cut_before_measuring_anything(monkeypatch):
    calls = []
    from manamap.pilot import diagnostic
    monkeypatch.setattr(diagnostic, "run_on", lambda *a, **k: calls.append(1))
    with pytest.raises(SystemExit, match="PROTECTED"):
        try_swap.run(EDGAR, [(VISH, "Morbid Opportunist")], iterations=200)
    assert calls == [], "a refused swap must cost nothing"


@requires_edgar_keep
def test_opening_a_branch_without_the_card_is_refused(tmp_path, monkeypatch):
    text = (config.DECKS_DIR / EDGAR / "decklist.txt").read_text(encoding="utf-8")
    assert VISH in text
    # Swap Vish Kal for a basic in the text, the way a pasted list would.
    cut = text.replace(f"1 {VISH}", "1 Swamp", 1)
    monkeypatch.setattr(deck_branch, "branch_root", lambda slug: tmp_path)
    with pytest.raises(SystemExit, match="PROTECTED"):
        deck_branch.new(EDGAR, "keep-test", cut)
    assert not (tmp_path / "keep-test").exists(), "a refused branch must not be created"


@requires_edgar_keep
def test_propose_and_merge_refuse_a_branch_that_cuts_it(monkeypatch):
    monkeypatch.setattr(deck_branch, "diff", lambda slug, branch: {"out": [VISH], "add": ["X"]})
    lines = protected.refusals(EDGAR, deck_branch.diff(EDGAR, "any")["out"])
    assert len(lines) == 1 and VISH in lines[0]
    # The merge gate folds the keep list into its blocking list, so `--force`
    # (a sourcing override) cannot reach it.
    import inspect
    src = inspect.getsource(deck_branch.merge)
    assert "protected.refusals" in src and "force" in src
    src = inspect.getsource(deck_branch.propose)
    assert "protected.refuse" in src


# ── try ──────────────────────────────────────────────────────────────────

def _a_fresh_branch_with_swaps():
    """(slug, branch, [(out, in)]) for some live branch whose cards.json matches its
    list and whose net diff pairs one-for-one, or None. A branch is a pilot artifact,
    never a fixture, so the test looks for one rather than naming it."""
    from manamap.pilot.common import decklist_sha256, load_json
    for slug_dir in sorted(config.DECKS_DIR.glob("*/branches")):
        slug = slug_dir.parent.name
        for b in sorted(slug_dir.iterdir()):
            meta = load_json(b / "branch.json") or {}
            if meta.get("merged") or not (b / "cards.json").exists():
                continue
            cards = load_json(b / "cards.json") or {}
            if cards.get("decklist_sha256") not in (None, decklist_sha256(slug, b.name)):
                continue
            try:
                d = deck_branch.diff(slug, b.name)
            except (Exception, SystemExit):      # noqa: BLE001 - a broken branch is not this test's
                continue
            outs, ins = d.get("out") or [], d.get("add") or []
            if outs and len(outs) == len(ins) and len(outs) <= 6 and not d.get("quantity"):
                if protected.refusals(slug, outs):
                    continue
                return slug, b.name, list(zip(outs, ins))
    return None


@requires_data
@requires_deck
def test_try_measures_exactly_what_a_staged_and_fetched_branch_measures():
    """THE SAME SWAPS, TWO ROUTES, ONE ANSWER. `try` builds its list in memory from
    the Scryfall dump; a branch is staged, fetched and measured from disk. On the
    identical seed every reading must match — measured 2026-10-04 on sharknado's six
    momentum-v1 swaps, after two defects that made them differ (a corpus record that
    flattened oracle text, and new cards appended rather than in decklist order)."""
    from manamap.pilot import diagnostic
    found = _a_fresh_branch_with_swaps()
    if found is None:
        pytest.skip("no live branch with a fresh cards.json and a one-for-one diff")
    slug, branch, swaps = found
    from manamap.pilot.common import load_deck_cards
    doc, _, _, _ = try_swap.apply_swaps(slug, None, swaps)
    # net-change measures the branch in the champion's slots (`diagnostic.align`).
    cand = dict(load_deck_cards(slug, branch))
    cand["cards"] = diagnostic.align(load_deck_cards(slug)["cards"], cand["cards"])
    staged = diagnostic.run_on(cand, slug, branch=branch, iterations=1500, quiet=True)
    held = diagnostic.run_on(doc, slug, iterations=1500, quiet=True)
    assert [c["name"] for c in doc["cards"]] == [c["name"] for c in cand["cards"]]
    assert held["output"] == staged["output"]
    assert held["steam"] == staged["steam"]
    assert held["mana"] == staged["mana"]


@requires_data
def test_a_new_card_is_shaped_exactly_like_a_fetched_one():
    """The dump-built record must carry the line breaks and P/T the goldfish reads."""
    rec = try_swap.corpus_card("Scrawling Crawler")
    assert rec is not None
    assert "\n" in rec["oracle_text"], "line breaks are how an ability window ends"
    assert rec["power"] == "3" and rec["toughness"] == "2"
    assert try_swap.corpus_card("Definitely Not A Card Name") is None


@requires_data
@requires_deck
def test_a_list_against_itself_pairs_to_exactly_zero_and_two_seeds_make_no_call():
    """THE PAIRING, PROVEN. Same list, same seed: every game identical, so every
    paired difference is exactly zero. Same list, two seeds: no row may earn a
    verdict (the A/A). Before 2026-10-04 one shared random stream made the arms of
    any comparison effectively independent after their first differing card."""
    from manamap.pilot import diagnostic, net_change
    a = diagnostic.run("goblin-storm", iterations=1500, seed=7, quiet=True, keep_games=True)
    same = diagnostic.run("goblin-storm", iterations=1500, seed=7, quiet=True, keep_games=True)
    assert a["_games"] == same["_games"]
    for row in net_change.compare_readings(a, same):
        if "paired" in (row.get("method") or ""):
            assert row["ci95_diff"] == [0.0, 0.0], row["measure"]
    other = diagnostic.run("goblin-storm", iterations=1500, seed=8, quiet=True, keep_games=True)
    assert all(r["verdict"] == "noise" for r in net_change.compare_readings(a, other))


def test_align_keeps_shared_cards_in_their_slots():
    from manamap.pilot import diagnostic
    base = [{"name": n} for n in ("A", "B", "C", "D")]
    cand = [{"name": n} for n in ("A", "C", "D", "Z")]       # B out, Z in
    assert [c["name"] for c in diagnostic.align(base, cand)] == ["A", "Z", "C", "D"]
