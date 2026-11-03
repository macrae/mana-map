"""Pinning art: `manamap pilot art` and `order`, and the false swaps they end.

Tmp deck tree and stubbed Scryfall, like `test_pilot_deck_edit`: the unit tier needs
no data and no network. Every case drives the production function.
"""
import json

import pytest

from manamap import config
from manamap.pilot import (card_pool, check_in, common, deck_edit, deck_status, deck_versions,
                           fetch_deck, order, printings, prices, protected, try_swap)
from manamap.pilot.fetch_deck import parse_decklist, resolve_entries, shape_card

LIST = ("Commander:\n1 Zada, Hedron Grinder (SLD) 2406 *F*\n\nDeck:\n"
        "# keep this comment\n1 Sol Ring\n1 Goblin Lackey\n"
        "48 Mountain (SLD) 2418 *F*\n49 Mountain (SLD) 2419 *F*\n")

PRINTS = {
    ("evg", "47"): {"name": "Raging Goblin", "games": ["paper"], "finishes": ["nonfoil"]},
    ("oana", "71"): {"name": "Raging Goblin", "games": ["arena"], "finishes": ["nonfoil"]},
    ("sld", "2417"): {"name": "Sol Ring", "games": ["paper"], "finishes": ["foil"]},
    ("c21", "263"): {"name": "Sol Ring", "games": ["paper"], "finishes": ["nonfoil"]},
    ("sld", "2418"): {"name": "Mountain", "games": ["paper"], "finishes": ["foil"]},
    ("sld", "2419"): {"name": "Mountain", "games": ["paper"], "finishes": ["foil"]},
    ("sld", "2406"): {"name": "Zada, Hedron Grinder", "games": ["paper"], "finishes": ["foil"]},
    ("sld", "1311"): {"name": "Goblin Lackey", "games": ["paper"], "finishes": ["nonfoil"]},
}


def _card(name, qty, cmdr=False, set_="def", cn="1", **kw):
    return dict({"name": name, "quantity": qty, "is_commander": cmdr, "type_line":
                 "Basic Land" if name == "Mountain" else "Artifact", "color_identity": [],
                 "cmc": 1.0, "oracle_text": "", "mana_cost": "", "set": set_,
                 "collector_number": cn, "scryfall_id": f"id-{name}"}, **kw)


@pytest.fixture
def deck(tmp_path, monkeypatch):
    root = tmp_path / "repo" / "data" / "decks"
    root.mkdir(parents=True)
    monkeypatch.setattr(config, "DECKS_DIR", root)
    names = {"Zada, Hedron Grinder", "Sol Ring", "Goblin Lackey", "Mountain", "Raging Goblin"}
    monkeypatch.setattr(check_in, "corpus_names", lambda: names)
    pool = {n: {"color_identity": [], "type_line": "Artifact", "cmc": 1.0, "legal": True,
                "mana_cost": ""} for n in names}
    monkeypatch.setattr(card_pool, "load_pool", lambda: pool)
    monkeypatch.setattr(card_pool, "legality", lambda column: {n: "legal" for n in names})
    monkeypatch.setattr(deck_edit, "_NAME_MEMO", {})
    monkeypatch.setattr(printings, "fetch_printings",
                        lambda wanted, **kw: {k: dict(PRINTS[k], set=k[0], collector_number=k[1])
                                              for k in wanted if k in PRINTS})
    d = root / "gs"
    d.mkdir()
    (d / "decklist.txt").write_text(LIST, encoding="utf-8")
    (d / "cards.json").write_text(json.dumps({"deck": "gs", "decklist_sha256": "x", "cards": [
        _card("Zada, Hedron Grinder", 1, True, "sld", "2406", foil=True),
        _card("Sol Ring", 1, set_="c21", cn="263"), _card("Goblin Lackey", 1, set_="sld", cn="2425"),
        _card("Mountain", 97, set_="sld", cn="2418", foil=True)]}))
    return d


def _text(d):
    return (d / "decklist.txt").read_text(encoding="utf-8")


def _pin(name, spec):
    return {name: [printings.parse_printing(spec)]}


# ── the normaliser ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text,want", [
    ("Raging Goblin (EVG) 47", ("Raging Goblin", "evg", "47", False)),
    ("Zada, Hedron Grinder (SLD) 2406 *F*", ("Zada, Hedron Grinder", "sld", "2406", True)),
    ("Raging Goblin (10E) 224★", ("Raging Goblin", "10e", "224★", False)),
    ("Goblin Matron (PLST) DMR-123", ("Goblin Matron", "plst", "DMR-123", False)),
    ("Sol Ring", ("Sol Ring", None, None, False)),
])
def test_split_printing_reads_what_the_parser_reads(text, want):
    p = common.split_printing(text)
    assert (p["name"], p["set"], p["collector_number"], p["foil"]) == want
    parsed = parse_decklist(f"1 {text}")[0]
    assert parsed["name"] == want[0] and parsed.get("set") == want[1]


# ── a change of art is not a swap ────────────────────────────────────────

def test_printing_only_swap_is_not_a_swap(deck):
    p = deck_edit.plan("gs", [{"op": "swap", "out": "Sol Ring", "in": "Sol Ring (C21) 263"}])
    assert p["blocking"] == [] and p["diff"] == {"out": {}, "in": {}}
    assert p["warnings"] == []
    assert p["ops"][0]["op"] == "printing"


def test_keep_list_does_not_refuse_a_change_of_art(deck, monkeypatch):
    monkeypatch.setattr(protected, "names", lambda slug: {"Sol Ring"})
    (deck / "protected.json").write_text(json.dumps(
        {"slug": "gs", "cards": [{"name": "Sol Ring", "why": "keep"}]}))
    p = deck_edit.plan("gs", [{"op": "swap", "out": "Sol Ring", "in": "Sol Ring (C21) 263"}])
    assert p["blocking"] == []


def test_a_commanders_art_can_change(deck):
    p = deck_edit.plan("gs", [{"op": "swap", "out": "Zada, Hedron Grinder",
                               "in": "Zada, Hedron Grinder (SLD) 2406"}])
    assert p["blocking"] == [] and p["diff"] == {"out": {}, "in": {}}


def test_a_real_swap_to_a_named_printing_still_is_one(deck):
    p = deck_edit.plan("gs", [{"op": "swap", "out": "Goblin Lackey",
                               "in": "Raging Goblin (EVG) 47"}])
    assert p["diff"] == {"out": {"Goblin Lackey": 1}, "in": {"Raging Goblin": 1}}
    assert "1 Raging Goblin (EVG) 47" in p["text_after"]


def test_art_only_edit_leaves_every_other_byte(deck):
    got = deck_edit.edit("gs", [{"op": "swap", "out": "Sol Ring", "in": "Sol Ring (C21) 263"}])
    assert got["written"]
    assert _text(deck) == LIST.replace("1 Sol Ring\n", "1 Sol Ring (C21) 263\n")


def test_try_says_art_has_nothing_to_measure(deck):
    with pytest.raises(SystemExit, match="only changes art"):
        try_swap.apply_ops("gs", None, [{"op": "swap", "out": "Sol Ring",
                                          "in": "Sol Ring (C21) 263"}])


# ── the guard: sleeved yes, archived no ──────────────────────────────────

def test_sleeved_deck_takes_art_and_refuses_a_swap(deck, monkeypatch):
    from manamap.pilot import promote
    monkeypatch.setattr(promote, "stage", lambda slug: promote.SLEEVED)
    with pytest.raises(SystemExit, match="SLEEVED"):
        deck_edit.edit("gs", [{"op": "swap", "out": "Goblin Lackey", "in": "Raging Goblin"}])
    r = printings.apply("gs", _pin("Sol Ring", "C21 263"), run=False)
    assert r["changed"]
    assert "1 Sol Ring (C21) 263" in _text(deck)


def test_edit_lets_art_through_on_a_sleeved_deck(deck, monkeypatch):
    from manamap.pilot import promote
    monkeypatch.setattr(promote, "stage", lambda slug: promote.SLEEVED)
    got = deck_edit.edit("gs", [{"op": "swap", "out": "Sol Ring", "in": "Sol Ring (C21) 263"}])
    assert got["written"] and "1 Sol Ring (C21) 263" in _text(deck)


def test_archived_deck_refuses_art(deck, monkeypatch):
    monkeypatch.setattr(common, "deck_is_apart", lambda slug: True)
    monkeypatch.setattr(common, "deck_lifecycle", lambda slug: ("retired",))
    with pytest.raises(SystemExit, match="archived"):
        printings.apply("gs", _pin("Sol Ring", "C21 263"), run=False)


def test_art_undo_restores_the_exact_bytes_and_refuses_a_card_move(deck, monkeypatch):
    from manamap.pilot import promote
    monkeypatch.setattr(promote, "stage", lambda slug: promote.SLEEVED)
    printings.apply("gs", _pin("Sol Ring", "C21 263"), run=False)
    deck_edit.undo("gs", art_only=True)
    assert _text(deck) == LIST
    monkeypatch.setattr(promote, "stage", lambda slug: promote.BENCH)
    deck_edit.edit("gs", [{"op": "swap", "out": "Goblin Lackey", "in": "Raging Goblin"}])
    monkeypatch.setattr(promote, "stage", lambda slug: promote.SLEEVED)
    with pytest.raises(SystemExit, match="moved cards"):
        deck_edit.undo("gs", art_only=True)


# ── what may be pinned ───────────────────────────────────────────────────

@pytest.mark.parametrize("name,spec,why", [
    ("Sol Ring", "XXX 999", "no such printing"),
    ("Sol Ring", "EVG 47", "that printing is"),
    ("Raging Goblin", "OANA 71", "never on paper"),
    ("Sol Ring", "C21 263 *F*", "never printed in foil"),
])
def test_refuses_art_that_cannot_be_bought(deck, name, spec, why):
    with pytest.raises(SystemExit, match=why):
        printings.check_pins("gs", _pin(name, spec))


def test_foil_only_printing_is_corrected_and_said(deck):
    pins = _pin("Sol Ring", "SLD 2417")
    notes = printings.check_pins("gs", pins)
    assert pins["Sol Ring"][0]["foil"] is True and "foil only" in notes[0]


# ── split art ────────────────────────────────────────────────────────────

def test_split_must_sum_to_the_cards_count(deck):
    _, pins = printings.parse_split("Mountain=SLD 2418 *F*:90,SLD 2419 *F*:1")
    with pytest.raises(SystemExit, match="never changes a count"):
        printings.rewrite(LIST, {"Mountain": pins})
    _, pins = printings.parse_split("Mountain=SLD 2418 *F*:50,SLD 2419 *F*:47")
    out = printings.rewrite(LIST, {"Mountain": pins})
    assert "50 Mountain (SLD) 2418 *F*\n47 Mountain (SLD) 2419 *F*" in out


def test_one_pin_gathers_a_split_card_onto_one_line(deck):
    out = printings.rewrite(LIST, _pin("Mountain", "SLD 2418 *F*"))
    assert "97 Mountain (SLD) 2418 *F*" in out and "2419" not in out


def test_rewrite_refuses_a_card_the_list_does_not_hold(deck):
    with pytest.raises(SystemExit, match="is not in this list"):
        printings.rewrite(LIST, _pin("Raging Goblin", "EVG 47"))


def test_cards_json_keeps_every_art_of_a_split_card():
    sc = lambda cn: {"name": "Mountain", "set": "sld", "collector_number": cn,
                     "finishes": ["foil"], "id": cn, "artist": "Wizard of Barge"}
    entries = parse_decklist("8 Mountain (SLD) 2418 *F*\n7 Mountain (SLD) 2419 *F*\n")
    cards, _ = resolve_entries(entries, {}, {("sld", "2418"): sc("2418"),
                                             ("sld", "2419"): sc("2419")})
    assert len(cards) == 1 and cards[0]["quantity"] == 15
    assert [(p["collector_number"], p["quantity"]) for p in cards[0]["printings"]] == [
        ("2418", 8), ("2419", 7)]


def test_a_card_on_one_printing_gets_no_printings_key():
    entries = parse_decklist("4 Shock (MSC) 809\n")
    sc = {"name": "Shock", "set": "msc", "collector_number": "809"}
    cards, _ = resolve_entries(entries, {}, {("msc", "809"): sc})
    assert "printings" not in cards[0]
    assert cards[0] == shape_card(sc, 4, False, False)


# ── pin everything shown ─────────────────────────────────────────────────

def test_pin_shown_pins_what_shows_and_leaves_pinned_alone(deck):
    pins, skipped = printings.unpinned_pins("gs")
    assert sorted(pins) == ["Sol Ring"]            # Lackey shows a print Scryfall lacks here
    assert [n for n, _ in skipped] == ["Goblin Lackey"]
    assert "Mountain" not in pins and "Zada, Hedron Grinder" not in pins


def test_pin_shown_skips_a_digital_only_row(deck, monkeypatch):
    monkeypatch.setitem(PRINTS, ("c21", "263"), {"name": "Sol Ring", "games": ["arena"],
                                                 "finishes": ["nonfoil"]})
    pins, skipped = printings.unpinned_pins("gs")
    assert "Sol Ring" not in pins
    assert any(n == "Sol Ring" and "arena" in w for n, w in skipped)


# ── fetch never quietly swaps art ────────────────────────────────────────

def test_name_lookup_leaves_a_digital_only_default(monkeypatch):
    seen = {}
    def fake(url, params=None, **kw):
        seen.update(params)
        return {"data": [{"name": "Raging Goblin", "set": "evg", "collector_number": "47",
                          "games": ["paper"]}]}
    monkeypatch.setattr(fetch_deck.net, "get_json", fake)
    got = fetch_deck._paper_print({"name": "Raging Goblin", "set": "oana", "games": ["arena"]})
    assert got["set"] == "evg" and "game:paper" in seen["q"]


def test_the_paper_lookup_never_takes_a_card_that_merely_has_that_face(monkeypatch):
    def fake(url, params=None, **kw):
        return {"data": [
            {"name": "Naktamun Lorespinner // Wheel of Fortune", "set": "soc", "games": ["paper"]},
            {"name": "Wheel of Fortune", "set": "3ed", "games": ["paper"]}]}
    monkeypatch.setattr(fetch_deck.net, "get_json", fake)
    got = fetch_deck._paper_print({"name": "Wheel of Fortune", "set": "vma", "games": ["mtgo"]})
    assert got["name"] == "Wheel of Fortune" and got["set"] == "3ed"


def test_the_paper_lookup_keeps_the_card_when_no_row_is_that_card(monkeypatch):
    monkeypatch.setattr(fetch_deck.net, "get_json", lambda *a, **k: {"data": [
        {"name": "Some // Other", "set": "soc", "games": ["paper"]}]})
    card = {"name": "Vintage Only", "set": "vma", "games": ["mtgo"]}
    assert fetch_deck._paper_print(card) is card


def test_a_paper_answer_is_not_looked_up_again(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no lookup for a paper print")
    monkeypatch.setattr(fetch_deck.net, "get_json", boom)
    card = {"name": "Sol Ring", "games": ["paper", "mtgo"]}
    assert fetch_deck._paper_print(card) is card
    bare = {"name": "Sol Ring"}
    assert fetch_deck._paper_print(bare) is bare


# ── art never stales an artifact, cards do ───────────────────────────────

def test_same_list_is_art_blind_and_card_sensitive(deck, monkeypatch):
    old = LIST
    sha_old = common.list_sha256(old)
    new = old.replace("1 Sol Ring\n", "1 Sol Ring (C21) 263\n")
    (deck / "decklist.txt").write_text(new, encoding="utf-8")
    monkeypatch.setattr(deck_versions, "_SAME_MEMO", {})
    monkeypatch.setattr(deck_versions.dh, "revisions", lambda slug: [{"sha": "aaa"}])
    monkeypatch.setattr(deck_versions.dh, "_blobs", lambda revs, path: {"aaa": old})
    truth = common.list_sha256(new)
    assert deck_versions.same_list("gs", sha_old, truth)
    assert not deck_status._stamp_is_stale(sha_old, truth, "gs")
    assert deck_status._stamp_is_stale(sha_old, truth)      # no slug: bytes, as before

    moved = old.replace("1 Sol Ring\n", "1 Raging Goblin\n")
    (deck / "decklist.txt").write_text(moved, encoding="utf-8")
    monkeypatch.setattr(deck_versions, "_SAME_MEMO", {})
    assert not deck_versions.same_list("gs", sha_old, common.list_sha256(moved))


# ── prices describe the pinned art ───────────────────────────────────────

FEED = {"id-Sol Ring": {"name": "Sol Ring", "set_code": "C21", "number": "263",
                        "scryfall_id": "id-Sol Ring", "price_cents_nm": 100,
                        "available_quantity": 5, "url": "https://manapool.com/a"},
        "other": {"name": "Goblin Lackey", "set_code": "OTH", "number": "9",
                  "scryfall_id": "other", "price_cents_nm": 50, "available_quantity": 5,
                  "url": "https://manapool.com/b"}}


def test_a_pinned_card_out_of_stock_keeps_its_printing_and_leaves_the_total():
    cheapest = {"goblin lackey": FEED["other"]}
    card = _card("Goblin Lackey", 2, set_="sld", cn="1311")
    row = prices._manapool_row(card, "missing", FEED, cheapest, pinned=True)
    assert row["printing"] == "(SLD) 1311" and row["pinned_unavailable"]
    assert row["nm_cents"] is None and row["cheapest_other"]["nm_cents"] == 50
    assert prices._totals({"Goblin Lackey": row})["total_cents"] == 0
    unpinned = prices._manapool_row(card, "missing", FEED, cheapest, pinned=False)
    assert unpinned["printing"] == "(OTH) 9" and "cheapest" in unpinned["note"]


def test_a_foil_pin_costs_the_foil_price():
    row = {"nm_cents": None, "foil_cents": 782, "quantity": 1, "foil": True}
    assert prices.unit_cents(row) == 782
    assert prices._totals({"Zada": row})["total_cents"] == 782
    assert prices.unit_cents({"nm_cents": 10, "foil_cents": 99, "quantity": 1}) == 10


def test_the_validator_adds_foil_and_split_art_the_way_prices_does():
    from manamap.pilot import validate_prices
    rows = {
        "Zada": {"scryfall_id": "a", "printing": "(SLD) 2406", "nm_cents": None, "lp_cents": None,
                 "foil_cents": 782, "url": None, "quantity": 1, "note": None, "foil": True},
        "Mountain": {"scryfall_id": "b", "printing": "(SLD) 2418", "nm_cents": None, "lp_cents": None,
                     "foil_cents": 268, "url": None, "quantity": 3, "note": None, "foil": True,
                     "printings": [{"foil": True, "foil_cents": 268, "nm_cents": None, "quantity": 2},
                                   {"foil": True, "foil_cents": 300, "nm_cents": None, "quantity": 1}]},
    }
    doc = {"slug": "gs", "branch": None, "as_of": "2026-11-03", "source": "manapool", "currency": "USD",
           "decklist_sha256": "x", "cards": rows, "missing": [], **prices._totals(rows)}
    assert doc["total_cents"] == 782 + 268 * 2 + 300
    errors = validate_prices.validate("gs", None, doc)
    assert not [e for e in errors if "total" in e], errors
    doc["total_cents"] += 1
    assert [e for e in validate_prices.validate("gs", None, doc) if "total_cents" in e]


def test_split_art_totals_each_printing():
    row = {"nm_cents": None, "foil_cents": 268, "quantity": 15, "foil": True, "printings": [
        {"foil": True, "foil_cents": 268, "nm_cents": None, "quantity": 8},
        {"foil": True, "foil_cents": 300, "nm_cents": None, "quantity": 7}]}
    assert prices._totals({"Mountain": row})["total_cents"] == 268 * 8 + 300 * 7


# ── the order ────────────────────────────────────────────────────────────

@pytest.fixture
def two(deck):
    root = deck.parent
    for slug, text, cards in (
        ("a", "4 Shock (MSC) 809\n2 Sol Ring\n10 Mountain (10E) 378\n",
         [_card("Shock", 4, set_="msc", cn="809"), _card("Sol Ring", 2),
          _card("Mountain", 10, set_="10e", cn="378")]),
        ("b", "4 Shock (MSC) 809\n2 Shock (M21) 159\n1 Sol Ring\n",
         [_card("Shock", 6, set_="msc", cn="809"), _card("Sol Ring", 1)]),
    ):
        (root / slug).mkdir()
        (root / slug / "decklist.txt").write_text(text, encoding="utf-8")
        (root / slug / "cards.json").write_text(json.dumps({"deck": slug, "cards": cards}))
    return root


def test_order_merges_same_printing_and_keeps_different_art_apart(two):
    doc = order.build(["a", "b"])
    assert order.paste(doc).splitlines() == [
        "2 Shock (M21) 159", "8 Shock (MSC) 809", "3 Sol Ring",
        "10 Mountain (10E) 378"]
    assert doc["count"] == 23 and doc["unpinned"] == ["Sol Ring"]


def test_plain_order_drops_every_printing(two):
    assert order.paste(order.build(["a", "b"]), plain=True).splitlines() == [
        "10 Shock", "3 Sol Ring", "10 Mountain"]


def test_order_never_says_which_deck_holds_a_card(two):
    doc = order.build(["a", "b"])
    assert all("slug" not in r and "deck" not in r for r in doc["lines"])
    assert "a" not in [r.get("from") for r in doc["lines"]]


def test_check_names_what_cannot_be_filled(two, monkeypatch):
    feed = {"id-Shock": {"name": "Shock", "price_cents_nm": 15, "available_quantity": 3,
                         "url": "u"},
            "id-Mountain": {"name": "Mountain", "price_cents_nm": 15, "available_quantity": 99,
                            "url": "u"},
            "x": {"name": "Sol Ring", "price_cents_nm": 200, "available_quantity": 9, "url": "u"}}
    def fake(session=None, meta=None):
        meta["as_of"] = "2026-11-03"
        return feed
    monkeypatch.setattr(prices, "manapool_feed", fake)
    got = order.check(order.build(["a"]))
    assert got["as_of"] == "2026-11-03"
    assert [r["name"] for r in got["unfillable"]] == ["Shock"]
    assert "only 3" in got["unfillable"][0]["status"]
    assert got["subtotal_cents"] == 200 * 2 + 15 * 10     # the Shocks are not in it


def test_check_treats_a_pinned_line_missing_from_the_feed_as_unfillable(two, monkeypatch):
    def fake(session=None, meta=None):
        return {"x": {"name": "Shock", "price_cents_nm": 15, "available_quantity": 99}}
    monkeypatch.setattr(prices, "manapool_feed", fake)
    got = order.check(order.build(["a"]))
    assert "Shock" in [r["name"] for r in got["unfillable"]]   # another Shock is not this art
