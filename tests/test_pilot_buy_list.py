"""`buy-list`: the bill's BUY rows as the paste Mana Pool's mass entry takes.

A tmp deck with one branch, the collection patched to a single box the way
`test_pilot_collection.py` does it, so `deck_branch.source` — the ONLY sourcing
answer — decides the states and this module only reads them. Nothing here reads
the boxes itself, and nothing here says which deck holds a card.
"""

import argparse
import json
import types

import pytest

from manamap import config
from manamap.pilot import buy_list, check_in
from manamap.pilot import collection as coll
from manamap.pilot.common import clear_memo

SLUG, BRANCH = "td", "b1"


def _card(name, set_=None, cn=None, foil=False, qty=1, commander=False):
    return {"name": name, "quantity": qty, "set": set_, "collector_number": cn,
            "foil": foil, "is_commander": commander}


def _bench(tmp_path, monkeypatch, cards_json=True, prices=None):
    """Deck: Cmdr + Kept Card + Cut Card. Branch: keeps Kept Card, adds one card
    that is in a box, one foil, three of a plain one, and a DFC named by its
    front face in the decklist and by its joined name in cards.json."""
    cdir = tmp_path / "collection"
    cdir.mkdir()
    (cdir / "box.txt").write_text("1 Boxed Card\n")
    monkeypatch.setattr(coll, "COLLECTION_DIR", cdir)
    ddir = tmp_path / "decks"
    ddir.mkdir()
    monkeypatch.setattr(config, "DECKS_DIR", ddir)
    clear_memo()
    deck = ddir / SLUG
    deck.mkdir()
    (deck / "decklist.txt").write_text("Commander:\n1 Cmdr\n\nDeck:\n1 Kept Card\n1 Cut Card\n")
    (deck / "cards.json").write_text(json.dumps({"deck": SLUG, "cards": [
        _card("Cmdr", commander=True), _card("Kept Card"), _card("Cut Card")]}))
    br = deck / "branches" / BRANCH
    br.mkdir(parents=True)
    (br / "decklist.txt").write_text(
        "Commander:\n1 Cmdr\n\nDeck:\n1 Kept Card\n1 Boxed Card\n1 Shiny Card\n"
        "3 Plain Card\n1 Front\n")
    if cards_json:
        (br / "cards.json").write_text(json.dumps({"deck": SLUG, "cards": [
            _card("Cmdr", commander=True), _card("Kept Card"),
            _card("Boxed Card", "abc", "1"),
            _card("Shiny Card", "slx", "12", foil=True),
            _card("Plain Card", "m21", "7", qty=3),
            _card("Front // Back", "mid", "99")]}))
    if prices is not None:
        (br / "prices.json").write_text(json.dumps(prices))
    return br


def _args(**kw):
    base = dict(slug=SLUG, branch=BRANCH, exact=False, as_json=False, out=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


PRICES = {"cards": {"Front // Back": {"nm_cents": 100}, "Plain Card": {"nm_cents": 50},
                    "Shiny Card": {"nm_cents": 1000}},
          "as_of": "2026-10-09", "source": "Manapool"}


def test_rows_are_exactly_the_buy_cards(tmp_path, monkeypatch):
    """In the deck already, in a box: neither is a purchase. The DFC resolves by
    its front face to the joined name cards.json keys."""
    _bench(tmp_path, monkeypatch)
    rows = buy_list.rows(SLUG, BRANCH)
    assert [r["name"] for r in rows] == ["Front // Back", "Plain Card", "Shiny Card"]
    assert all(set(r) == set(buy_list.ROW_KEYS) for r in rows)
    by = {r["name"]: r for r in rows}
    assert by["Plain Card"]["quantity"] == 3
    assert by["Shiny Card"]["foil"] is True and by["Shiny Card"]["set"] == "slx"
    assert by["Front // Back"]["collector_number"] == "99"


def test_plain_and_exact_forms(tmp_path, monkeypatch):
    _bench(tmp_path, monkeypatch)
    rows = buy_list.rows(SLUG, BRANCH)
    assert buy_list.render(rows) == "1 Front // Back\n3 Plain Card\n1 Shiny Card *F*"
    assert buy_list.render(rows, exact=True) == (
        "1 Front // Back (MID) 99\n3 Plain Card (M21) 7\n1 Shiny Card (SLX) 12 *F*")


def test_the_exact_form_is_check_ins_own_line(tmp_path, monkeypatch):
    """ONE FORMATTER. The exact form is byte-for-byte what `check-in` writes to
    decklist.txt — re-deriving it is how a set code ends up cased differently on
    two surfaces."""
    _bench(tmp_path, monkeypatch)
    for r in buy_list.rows(SLUG, BRANCH):
        line = buy_list.render([r], exact=True)
        assert line == check_in.decklist_line(r)
        assert line in check_in.render_decklist([r]).splitlines()


def test_no_price_file_means_no_figure(tmp_path, monkeypatch, capsys):
    """ABSENT MEANS ABSENT. The footer carries the count and no dollar sign."""
    _bench(tmp_path, monkeypatch)
    doc = buy_list.payload(SLUG, BRANCH)
    assert doc["buy_cents"] is None and doc["as_of"] is None
    buy_list.main(_args())
    out = capsys.readouterr().out
    assert out.rstrip().endswith("3 cards to buy")
    assert "$" not in out


def test_a_price_file_beside_the_branch_totals_quantity_times_nm(tmp_path, monkeypatch, capsys):
    _bench(tmp_path, monkeypatch, prices=PRICES)
    doc = buy_list.payload(SLUG, BRANCH)
    assert doc["buy_cents"] == 100 + 3 * 50 + 1000
    assert doc["as_of"] == "2026-10-09"
    buy_list.main(_args())
    assert "3 cards to buy  ≈ $12.50 (Manapool, 2026-10-09)" in capsys.readouterr().out


def test_a_partial_price_file_is_not_a_total(tmp_path, monkeypatch):
    """A sum over two of three rows presented as the bill is a figure nobody
    measured."""
    partial = {"cards": {k: v for k, v in PRICES["cards"].items() if k != "Plain Card"},
               "as_of": "2026-10-09"}
    _bench(tmp_path, monkeypatch, prices=partial)
    assert buy_list.payload(SLUG, BRANCH)["buy_cents"] is None
    assert buy_list.total_cents([], {"cards": {}}) == 0, "nothing to buy is a real zero"
    assert buy_list.total_cents([{"name": "X", "quantity": 1}], None) is None


def test_json_shape(tmp_path, monkeypatch, capsys):
    _bench(tmp_path, monkeypatch)
    buy_list.main(_args(as_json=True, exact=True))
    doc = json.loads(capsys.readouterr().out)
    assert set(doc) == {"text", "count", "buy_cents", "as_of"}
    assert doc["count"] == 3 and "(SLX) 12" in doc["text"]


def test_out_is_slug_scoped(tmp_path, monkeypatch, capsys):
    """A generic name in a shared scratch directory is refused; a bare name
    lands in the deck's own directory."""
    br = _bench(tmp_path, monkeypatch)
    (tmp_path / "scratch").mkdir()
    with pytest.raises(SystemExit, match="slug"):
        buy_list.main(_args(out=str(tmp_path / "scratch" / "buy.txt")))
    buy_list.main(_args(out="buy.txt"))
    written = br.parent.parent / "buy.txt"
    assert written.exists()
    assert written.read_text(encoding="utf-8").startswith("1 Front // Back\n")
    assert "wrote" in capsys.readouterr().out


def test_a_branch_without_cards_json_is_refused_naming_fetch_deck(tmp_path, monkeypatch):
    _bench(tmp_path, monkeypatch, cards_json=False)
    with pytest.raises(SystemExit, match="fetch-deck"):
        buy_list.rows(SLUG, BRANCH)


def test_the_command_is_registered_and_needs_a_branch():
    """A deck's own list has no adds, so `--branch` is required by the parser."""
    from manamap.pilot.registry import PILOT_STEPS, _DECK_COMMANDS, add_pilot_parser
    assert "buy-list" in {name for name, *_ in PILOT_STEPS}
    assert "buy-list" in _DECK_COMMANDS
    parser = argparse.ArgumentParser()
    add_pilot_parser(parser.add_subparsers(dest="command"))
    with pytest.raises(SystemExit):
        parser.parse_args(["pilot", "buy-list", SLUG])
    args = parser.parse_args(["pilot", "buy-list", SLUG, "--branch", BRANCH, "--exact", "--json"])
    assert args.branch == BRANCH and args.exact and args.as_json and args.out is None
