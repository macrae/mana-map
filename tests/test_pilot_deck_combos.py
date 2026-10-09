"""Tests for the per-deck combo report (pilot/deck_combos.py) and its gate.

All synthetic: a hand-built `details` and `pool`, so every rule is driven
through the production function and never re-derived here.
"""

from manamap.pilot import deck_combos as dc
from manamap.pilot import validate_deck_combos as vdc


def _details(combos):
    by_card = {}
    for i, combo in enumerate(combos):
        for name in combo["cards"]:
            by_card.setdefault(name, []).append(i)
    return {"combos": combos, "by_card": by_card,
            "meta": {"combo_count": len(combos), "source": {"timestamp": "2026-10-09T00:00:00+00:00"}}}


def _combo(id_, cards, bracket=1, produces=("Value",), mv=0, popularity=None, banned=False):
    c = {"id": id_, "cards": list(cards), "produces": list(produces), "ci": "",
         "bracket": bracket, "mana_value_needed": mv, "popularity": popularity}
    if banned:
        c["banned"] = True
        c["bracket"] = None
    return c


def _pool(**cards):
    """name -> {legal, color_identity}; `cards` values are (legal, identity)."""
    return {name.replace("_", " "): {"legal": legal, "color_identity": set(ident)}
            for name, (legal, ident) in cards.items()}


DECK = ["A", "B", "C"]
CMDR = ["K"]


def test_included_needs_every_card_and_counts_the_commander():
    details = _details([
        _combo("x", ["A", "B"]),            # in
        _combo("y", ["A", "K"]),            # in, via the command zone
        _combo("z", ["A", "Q"]),            # one short — not included
    ])
    rows = dc.included_combos(DECK, CMDR, details)
    assert [r["id"] for r in rows] == ["x", "y"]
    assert all(k in rows[0] for k in ("id", "cards", "produces", "infinite", "bracket",
                                      "banned", "mana_value_needed", "popularity",
                                      "assumes_other_commander"))


def test_included_is_ranked_bracket_then_popularity_then_id_and_file_order_on_request():
    details = _details([
        _combo("c", ["A", "B"], bracket=1, popularity=5),
        _combo("a", ["A", "C"], bracket=3, popularity=1),
        _combo("b", ["B", "C"], bracket=3, popularity=9),
        _combo("d", ["A", "B", "C"], bracket=3, popularity=9),
        _combo("e", ["K", "A"], bracket=None, popularity=None, banned=True),
    ])
    ranked = [r["id"] for r in dc.included_combos(DECK, CMDR, details)]
    assert ranked == ["b", "d", "a", "c", "e"]
    assert ranked == [r["id"] for r in dc.included_combos(list(reversed(DECK)), CMDR, details)]
    # `assess` reads file order so the tracked bracket reports keep their bytes.
    assert [r["id"] for r in dc.included_combos(DECK, CMDR, details, ranked=False)] == \
        ["c", "a", "b", "d", "e"]


def test_the_commander_assumption_is_flagged_not_dropped():
    details = _details([
        _combo("own", ["K", "A"], produces=["Infinite mana", "commander in the zone"]),
        _combo("other", ["A", "B"], produces=["Infinite mana", "requires commander"]),
    ])
    rows = {r["id"]: r for r in dc.included_combos(DECK, CMDR, details)}
    assert rows["own"]["assumes_other_commander"] is False
    assert rows["other"]["assumes_other_commander"] is True
    s = dc.summarize(list(rows.values()), [], 0)
    assert s["included"] == 2 and s["excluded_commander_assumption"] == 1
    assert s["infinite"] == 1, "a line assuming another commander is not counted as infinite"


def test_near_misses_are_exactly_one_short_legal_in_identity_and_not_banned():
    details = _details([
        _combo("one", ["A", "N1"], popularity=10),                 # near
        _combo("two", ["A", "N1", "N2"]),                          # two short
        _combo("illegal", ["B", "Bad"], popularity=99),            # banned card
        _combo("off", ["B", "Red"], popularity=99),                # outside identity
        _combo("banned", ["C", "N3"], banned=True, popularity=99), # banned line
        _combo("unknown", ["C", "Ghost"], popularity=99),          # not in the corpus
        _combo("in", ["A", "B"], popularity=99),                   # fully included
        _combo("cmdr", ["A", "N3"], produces=["Infinite commander copies"], popularity=99),
        _combo("inf", ["K", "N2"], produces=["Infinite mana"], popularity=3),
    ])
    pool = _pool(N1=(True, "W"), N2=(True, "U"), N3=(True, ""), Bad=(False, "W"), Red=(True, "R"))
    rows, total = dc.near_misses(DECK, CMDR, {"W", "U"}, details, pool, limit=50)
    assert [r["id"] for r in rows] == ["one", "inf"] and total == 2
    assert rows[0]["missing"] == "N1" and rows[1]["infinite"] is True
    assert all(k in rows[0] for k in ("id", "cards", "missing", "infinite", "bracket",
                                      "mana_value_needed", "popularity"))


def test_near_misses_cap_and_report_the_uncapped_total():
    combos = [_combo(f"n{i:02d}", ["A", f"M{i}"], popularity=i) for i in range(7)]
    pool = _pool(**{f"M{i}": (True, "") for i in range(7)})
    rows, total = dc.near_misses(DECK, CMDR, set(), _details(combos), pool, limit=3)
    assert total == 7 and [r["id"] for r in rows] == ["n06", "n05", "n04"]
    assert dc.summarize([], rows, total)["near"] == 3
    assert dc.summarize([], rows, total)["near_total"] == 7


def test_summary_highest_bracket_is_absent_when_nothing_carries_one():
    assert dc.summarize([], [], 0)["highest_bracket"] is None
    rows = dc.included_combos(DECK, CMDR, _details([_combo("x", ["A", "B"], bracket=4),
                                                     _combo("y", ["A", "C"], bracket=2)]))
    assert dc.summarize(rows, [], 0)["highest_bracket"] == 4


# ── the gate ────────────────────────────────────────────────────────────────

def _good():
    details = _details([
        _combo("x", ["A", "B"], bracket=3, popularity=4),
        _combo("y", ["A", "N1"], produces=["Infinite mana"], popularity=8),
    ])
    pool = _pool(N1=(True, "W"))
    included = dc.included_combos(DECK, CMDR, details)
    near, total = dc.near_misses(DECK, CMDR, {"W"}, details, pool)
    doc = {"slug": "d", "branch": None, "decklist_sha256": "abc123def456",
           "combo_data": {"source_timestamp": "t", "combo_count": 2},
           "summary": dc.summarize(included, near, total), "included": included, "near": near}
    kw = dict(branch=None, present=set(DECK) | set(CMDR), identity={"W"}, pool=pool,
              live_sha="abc123def456", combo_count=2)
    return doc, kw


def test_a_fresh_report_passes():
    doc, kw = _good()
    assert vdc.validate("d", doc, **kw) == []


def _fails_with(doc, kw, phrase):
    errors = vdc.validate("d", doc, **kw)
    assert any(phrase in e for e in errors), f"{phrase!r} not in {errors}"


def test_each_broken_form_is_named():
    doc, kw = _good()
    doc["slug"] = "other"
    _fails_with(doc, kw, "slug is 'other'")

    doc, kw = _good()
    doc["decklist_sha256"] = "ffffffffffff"
    _fails_with(doc, kw, "stale")

    doc, kw = _good()
    doc["included"][0]["cards"] = ["A", "Z"]
    _fails_with(doc, kw, "not in the list")

    doc, kw = _good()
    doc["near"][0]["missing"] = "A"
    _fails_with(doc, kw, "is in the list")

    doc, kw = _good()
    kw["pool"] = _pool(N1=(False, "W"))
    _fails_with(doc, kw, "not Commander-legal")

    doc, kw = _good()
    kw["identity"] = {"B"}
    _fails_with(doc, kw, "outside the identity")

    doc, kw = _good()
    kw["pool"] = {}
    _fails_with(doc, kw, "not in the corpus")

    doc, kw = _good()
    doc["near"][0]["id"] = doc["included"][0]["id"]
    _fails_with(doc, kw, "duplicate combo id")

    doc, kw = _good()
    doc["summary"]["infinite"] = 7
    _fails_with(doc, kw, "summary.infinite")

    doc, kw = _good()
    doc["combo_data"]["combo_count"] = 1
    _fails_with(doc, kw, "combo_count")

    doc, kw = _good()
    doc["branch"] = "b1"
    _fails_with(doc, kw, "branch is 'b1'")

    doc, kw = _good()
    del doc["near"]
    _fails_with(doc, kw, "missing required key 'near'")


def test_near_misses_read_legality_from_the_decks_format_column_when_given_one():
    """`legal` is the deck's own column (`card_pool.legality(spec.legality_column)`).
    Given, it decides and the pool's Commander flag does not; absent — the
    fleet's path — the pool's flag decides as it always has, so the fourteen
    tracked reports did not move."""
    details = _details([
        _combo("modern-ok", ["A", "N1"], popularity=10),     # Commander-illegal, Modern-legal
        _combo("cmdr-only", ["A", "N2"], popularity=20),     # Commander-legal, not Modern
        _combo("both", ["B", "N3"], popularity=5),
    ])
    pool = _pool(N1=(False, ""), N2=(True, ""), N3=(True, ""))
    by_commander, _ = dc.near_misses(DECK, CMDR, set(), details, pool)
    assert [r["id"] for r in by_commander] == ["cmdr-only", "both"]
    modern = {"N1": "legal", "N2": "not_legal", "N3": "legal"}
    by_modern, total = dc.near_misses(DECK, CMDR, set(), details, pool, legal=modern)
    assert [r["id"] for r in by_modern] == ["modern-ok", "both"] and total == 2
    # A name the column has never heard of is not legal; a banned one is not either.
    assert dc.near_misses(DECK, CMDR, set(), details, pool, legal={"N3": "banned"})[1] == 0
