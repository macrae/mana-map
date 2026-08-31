"""mana-analysis's deterministic core: classes, sources, producer kinds."""

from conftest import requires_deck

from manamap.pilot.mana_analysis import land_classes, nonland_producer_kind
from manamap.pilot.manabase import land_colors
from conftest import ROOT


def test_tapped_snow_dual_carries_all_its_classes():
    card = {"name": "Alpine Meadow", "type_line": "Snow Land — Mountain Plains",
            "oracle_text": "({T}: Add {R} or {W}.)\nThis land enters tapped."}
    assert {"snow", "tapped"} <= land_classes(card)
    assert land_colors(card) == {"R", "W"}


def test_add_or_clause_counts_both_colours():
    card = {"name": "Clifftop Retreat", "type_line": "Land",
            "oracle_text": "Clifftop Retreat enters tapped unless you control "
                           "a Mountain or a Plains.\n{T}: Add {R} or {W}.",
            "color_identity": []}
    assert land_colors(card) == {"R", "W"}


def test_fetch_land_classified():
    card = {"name": "Windswept Heath", "type_line": "Land",
            "oracle_text": "{T}, Pay 1 life, Sacrifice this land: Search your "
                           "library for a Forest or Plains card..."}
    assert "fetch" in land_classes(card)


def test_producer_kinds_follow_the_type_line():
    rock = {"name": "Arcane Signet", "type_line": "Artifact",
            "oracle_text": "{T}: Add one mana of any color in your "
                           "commander's color identity."}
    dork = {"name": "Llanowar Elves", "type_line": "Creature — Elf Druid",
            "oracle_text": "{T}: Add {G}."}
    ritual = {"name": "Seething Song", "type_line": "Instant",
              "oracle_text": "Add {R}{R}{R}{R}{R}."}
    plain = {"name": "Bear", "type_line": "Creature — Bear", "oracle_text": ""}
    assert nonland_producer_kind(rock) == "ramp:rock"
    assert nonland_producer_kind(dork) == "ramp:dork"
    assert nonland_producer_kind(ritual) == "ramp:ritual"
    assert nonland_producer_kind(plain) is None


def test_restricted_mana_produces_nothing():
    card = {"name": "Haven", "type_line": "Land", "color_identity": [],
            "oracle_text": "{T}: Add one mana of any color. Spend this mana "
                           "only to cast a Dragon creature spell."}
    assert land_colors(card) == set()


# ── Copies, not entries: the bug that published "18 lands" for a 33-land deck ──


def _deck(tmp_path, monkeypatch, cards):
    import json
    decks = tmp_path / "decks"
    base = decks / "test-deck"
    base.mkdir(parents=True)
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", decks)
    (base / "cards.json").write_text(json.dumps(
        {"deck": "test-deck", "decklist_sha256": "abc", "cards": cards}))
    return base


def _land(name, oracle, quantity=1, type_line="Basic Land — Island"):
    return {"name": name, "type_line": type_line, "oracle_text": oracle,
            "quantity": quantity, "is_commander": False,
            "color_identity": []}


def test_basic_land_copies_all_count_as_sources(tmp_path, monkeypatch):
    """Eleven Islands are eleven blue sources, not one entry."""
    from manamap.pilot import common
    from manamap.pilot.mana_analysis import analyze

    _deck(tmp_path, monkeypatch, [
        {"name": "Cmd", "type_line": "Legendary Creature", "oracle_text": "",
         "quantity": 1, "is_commander": True,
         "color_identity": ["U"], "mana_cost": "{1}{U}", "cmc": 2.0},
        _land("Island", "({T}: Add {U}.)", quantity=11),
        _land("Reliquary Tower", "{T}: Add {C}.", type_line="Land"),
    ])
    common.clear_memo()
    result = analyze("test-deck")

    assert result["lands"]["total"] == 12      # copies
    assert result["lands"]["entries"] == 2     # distinct cards
    assert result["sources"]["lands"]["U"] == 11
    # The table stays one row per distinct land, carrying its copy count.
    island = [r for r in result["lands"]["list"] if r["name"] == "Island"][0]
    assert island["copies"] == 11


def test_tapped_ratio_uses_copies_in_the_denominator(tmp_path, monkeypatch):
    """A lone tapped land beside 11 basics is 1-in-12, not 1-in-2."""
    from manamap.pilot import common
    from manamap.pilot.mana_analysis import analyze

    _deck(tmp_path, monkeypatch, [
        {"name": "Cmd", "type_line": "Legendary Creature", "oracle_text": "",
         "quantity": 1, "is_commander": True,
         "color_identity": ["U"], "mana_cost": "{1}{U}", "cmc": 2.0},
        _land("Island", "({T}: Add {U}.)", quantity=11),
        _land("Sunken Hollow", "This land enters tapped.\n{T}: Add {U}.",
              type_line="Land"),
    ])
    common.clear_memo()
    result = analyze("test-deck")

    assert result["lands"]["enters_tapped"] == 1
    assert result["lands"]["total"] == 12
    # 1/12 is under the one-in-three budget, so no note fires.
    assert not [n for n in result["notes"] if "enter tapped" in n]


@requires_deck
def test_tracked_mana_analysis_artifacts_are_current():
    """Every committed mana_analysis.json must match a fresh computation.

    The published-18-lands bug shipped because nothing compared the artifact to
    the code that makes it. This is that comparison.
    """
    import json
    from pathlib import Path
    from manamap.pilot.mana_analysis import analyze

    checked = 0
    for path in sorted((ROOT / "data/decks").glob("*/mana_analysis.json")):
        info = path.parent / "info.json"
        if info.exists():
            try:
                if (json.loads(info.read_text()) or {}).get("lifecycle"):
                    # RETIRED: history, not a claim. A model correction leaves
                    # these stale forever and regenerating them means measuring
                    # a deck nobody plays (the pilot's rule, 2026-08-27).
                    continue
            except Exception:                    # pragma: no cover - defensive
                pass
        tracked = json.loads(path.read_text())
        assert tracked == analyze(path.parent.name), (
            f"{path} is stale — re-run `manamap pilot mana-analysis "
            f"{path.parent.name}`")
        checked += 1
    assert checked >= 5, "no live deck was checked; the glob or the skip is wrong"


# ── what the base charges in life ──


def _l(name, text, type_line="Land"):
    return {"name": name, "oracle_text": text, "type_line": type_line}


def test_recurring_life_is_a_cost_paid_on_every_activation():
    from manamap.pilot.mana_analysis import life_cost
    assert life_cost(_l("Tarnished Citadel",
                        "{T}: Add {C}. {T}: Add one mana of any color. "
                        "This land deals 3 damage to you."))["recurring"] == 3
    assert life_cost(_l("Mana Confluence",
                        "{T}, Pay 1 life: Add one mana of any color."))["recurring"] == 1
    assert life_cost(_l("City of Brass",
                        "Whenever City of Brass becomes tapped, it deals 1 damage "
                        "to you. {T}: Add one mana of any color."))["recurring"] == 1


def test_a_life_cost_that_buys_no_mana_is_not_a_mana_cost():
    """RE-INTRODUCING THE FIRST BUG. Sorrow's Path has NO mana ability — it just
    hurts you when it taps — and without the `add` gate it read as a 2-life
    source."""
    from manamap.pilot.mana_analysis import life_cost
    path = _l("Sorrow's Path",
              "{T}: Choose two target blocking creatures. Whenever Sorrow's Path "
              "becomes tapped, it deals 2 damage to you.")
    assert life_cost(path)["recurring"] == 0


def test_one_time_life_does_not_require_a_mana_ability():
    """RE-INTRODUCING THE SECOND BUG, and it is the one that would have shipped.

    A FETCHLAND MAKES NO MANA ITSELF. Applying the `add` gate to the one-time
    figure as well zeroed all four of ur-dragon's fetches, and the shocklands
    name THEMSELVES where the modern MDFCs say "this land". The ledger read zero
    for a list holding six shocklands and four fetches, and looked plausible.
    """
    from manamap.pilot.mana_analysis import life_cost
    fetch = _l("Wooded Foothills",
               "{T}, Pay 1 life, Sacrifice this land: Search your library for a "
               "Mountain or Forest card, put it onto the battlefield, then shuffle.")
    shock = _l("Blood Crypt",
               "As Blood Crypt enters, you may pay 2 life. If you don't, it "
               "enters tapped. {T}: Add {B} or {R}.", "Land — Swamp Mountain")
    mdfc = _l("Fell Mire",
              "As this land enters, you may pay 3 life. If you don't, it enters "
              "tapped. {T}: Add {B}.")
    assert life_cost(fetch)["one_time"] == 1, "a fetch has no `add` clause"
    assert life_cost(shock)["one_time"] == 2, "a shockland names itself"
    assert life_cost(mdfc)["one_time"] == 3
    # …and none of the three charges anything RECURRING.
    for card in (fetch, shock, mdfc):
        assert life_cost(card)["recurring"] == 0, card["name"]


def test_the_two_life_figures_are_never_the_same_number():
    """They are different KINDS of cost and summing them hides the distinction:
    a painland charges every tap, a fetch charges once."""
    from manamap.pilot.mana_analysis import life_cost
    forge = _l("Battlefield Forge",
               "{T}: Add {C}. {T}: Add {R} or {W}. This land deals 1 damage to you.")
    assert life_cost(forge) == {"recurring": 1, "one_time": 0}


def test_no_corpus_land_is_both_recurring_and_one_time():
    """The sweep's structural claim, over the real corpus: 52 recurring, 38
    one-time, zero overlap."""
    import pytest
    from manamap.pilot.mana_analysis import life_cost
    try:
        from manamap.pilot import card_pool
        pool, oracle = card_pool.load_pool(), card_pool.corpus_oracle()
    except Exception:  # pragma: no cover
        pytest.skip("corpus not built")
    rec = one = checked = 0
    for name, info in pool.items():
        if "Land" not in (info.get("type_line") or ""):
            continue
        checked += 1
        cost = life_cost(dict(info, name=name, oracle_text=oracle.get(name, "")))
        rec += bool(cost["recurring"]); one += bool(cost["one_time"])
        assert not (cost["recurring"] and cost["one_time"]), name
    assert checked >= 1000, f"only {checked} lands swept"
    assert rec >= 40 and one >= 30, f"rec={rec} one={one}"
