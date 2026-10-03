"""`scan-candidates`: one pass over the corpus along a deck's dimensions, every row
labelled with the predicate that admitted it, converters and two-card infinites FLAGGED
and sorted last rather than ranked away or dropped.

Driven on the real edgar-vampires deck against the real corpus, roles and combo index,
because every claim below is about those artifacts; the cards named are the ones the
drain-v1 plan and the 2026-09-30 recon argue about.
"""
import pytest

from manamap import config
from manamap.pilot import candidate_scan as cs
from manamap.pilot import validate_candidate_scan as vcs

from conftest import requires_corpus, requires_deck, requires_roles

pytestmark = [requires_corpus, requires_deck, requires_roles,
              pytest.mark.skipif(not (config.DECKS_DIR / "edgar-vampires" / "cards.json").exists(),
                                 reason="requires the edgar-vampires deck"),
              pytest.mark.skipif(not config.COMBO_DETAILS_PATH.exists(), reason="requires combo_details.json")]


@pytest.fixture(scope="module")
def doc():
    return cs.scan("edgar-vampires", limit=5000)


def _rows(doc, dim):
    return {r["name"]: r for r in doc["dimensions"][dim]["candidates"]}


def test_a_card_in_the_99_and_a_game_changer_never_appear(doc):
    """The two exclusions that waste a reader's time or break the bracket."""
    from manamap.pilot.card_search import deck_names
    present = deck_names("edgar-vampires")
    for dim, block in doc["dimensions"].items():
        names = [r["name"] for r in block["candidates"]]
        assert not (set(names) & present), f"{dim}: a card already in the 99 surfaced"
        assert all(not r.get("game_changer") for r in block["candidates"])
    assert "Blood Artist" not in _rows(doc, "drain") and "Vito, Thorn of the Dusk Rose" not in _rows(doc, "drain")
    assert "Necropotence" not in _rows(doc, "draw") and "Necropotence" in doc["excluded"]["game_changer"]
    assert doc["excluded"]["in_99"] >= 90


def test_converters_are_admitted_flagged_and_sorted_last_not_ranked_or_dropped(doc):
    """The bug this guards: a converter ranked by EDHREC like any drain — Exquisite Blood
    sits near the top of every drain list and is the one card bracket 3 forbids here."""
    drain = doc["dimensions"]["drain"]["candidates"]
    by = {r["name"]: r for r in drain}
    eb = by["Exquisite Blood"]
    assert eb["flags"]["converter"] == "converter_loss_to_gain"
    assert set(eb["combos"]["infinite_with"]) >= {"Sanguine Bond", "Vito, Thorn of the Dusk Rose"}
    cliff = by["Cliffhaven Vampire"]
    assert cliff["flags"]["converter"] == "converter_gain_to_loss" and cliff["combos"]["infinite_with"] == ["Bloodthirsty Conqueror"]
    first_flagged = next(i for i, r in enumerate(drain) if r["combos"]["infinite_with"])
    assert all(not r["combos"]["infinite_with"] for r in drain[:first_flagged])
    assert all(r["combos"]["infinite_with"] for r in drain[first_flagged:])
    assert doc["dimensions"]["drain"]["flagged_infinite"] == len(drain) - first_flagged >= 3
    # a plain death drain is unflagged and says why it is here
    z = by["Zulaport Cutthroat"]
    assert z["flags"] == {"death_drain": True, "converter": None} and "wincon:drain" in z["matched"]["roles"]


def test_death_draw_splits_on_the_word_nontoken(doc):
    """No source makes the distinction; the row does. Midnight Reaper and Grim Haruspex
    miss every eminence token; Species Specialist and Liliana's Standard Bearer do not."""
    draw = _rows(doc, "draw")
    assert draw["Midnight Reaper"]["flags"] == {"trigger": "death", "nontoken": True, "costs_life": False}
    assert draw["Grim Haruspex"]["flags"]["nontoken"] is True
    assert draw["Species Specialist"]["flags"] == {"trigger": "death", "nontoken": False, "costs_life": False}
    assert draw["Liliana's Standard Bearer"]["flags"]["trigger"] == "death"
    assert draw["Dawn of Hope"]["flags"]["trigger"] == "gain" and draw["Well of Lost Dreams"]["flags"]["trigger"] == "gain"
    assert draw["Phyrexian Arena"]["flags"] == {"trigger": "plain", "nontoken": False, "costs_life": True}
    assert "draw.put_into_hand" in draw["Twilight Prophet"]["matched"]["oracle"]
    # admission by TAG alone (Midnight Reaper has no draw role) and by ORACLE alone (Standard Bearer has neither)
    assert draw["Midnight Reaper"]["matched"]["roles"] == [] and draw["Midnight Reaper"]["matched"]["tags"] == ["draw"]
    assert draw["Liliana's Standard Bearer"]["matched"] == {"oracle": ["draw.cards"], "roles": [], "tags": [], "keywords": []}


def test_the_threat_dimension_reads_printed_power_and_keywords(doc):
    threat = _rows(doc, "threat")
    vr = threat["Vein Ripper"]
    assert vr["power"] == 6 and vr["matched"]["keywords"] == ["Flying"] and vr["flags"]["typal"] and vr["flags"]["death_drain"]
    assert threat["Malakir Bloodwitch"]["flags"]["typal"] is True
    assert "Twilight Prophet" not in threat, "a 2/4 is not big; it is a drain and draw card"
    assert all(r["power"] is not None and r["power"] >= cs.THREAT_MIN_POWER for r in threat.values())


def test_every_row_names_what_admitted_it_and_the_sort_is_edhrec_rank(doc):
    from manamap.pilot.card_search import UNRANKED
    checked = 0
    for dim, block in doc["dimensions"].items():
        rows = block["candidates"]
        assert block["matched"] >= len(rows) and block["truncated"] == 0
        for r in rows:
            assert any(r["matched"].values()), f"{dim}: {r['name']} admitted by nothing"
            checked += 1
        unflagged = [r for r in rows if not r["combos"]["infinite_with"]]
        ranks = [r["edhrec_rank"] if r["edhrec_rank"] is not None else UNRANKED for r in unflagged]
        assert ranks == sorted(ranks), f"{dim}: not sorted by EDHREC rank"
    assert checked > 500


def test_the_validator_passes_the_real_scan_and_fails_the_broken_shapes(doc):
    errors, warns = vcs.validate("edgar-vampires", doc)
    assert errors == [], errors
    import json
    bad = json.loads(json.dumps(doc))
    drain = bad["dimensions"]["drain"]["candidates"]
    drain[0]["name"] = "Not A Card"
    drain[1]["combos"]["infinite_with"] = ["Sol Ring"]              # no such two-card line
    drain.insert(0, drain[-1])                                       # a flagged row before unflagged ones
    bad["dimensions"]["threat"]["candidates"].append({"name": "Necropotence", "combos": {"infinite_with": []}})
    errs, _ = vcs.validate("edgar-vampires", bad)
    assert any("not in the corpus" in e for e in errs) and any("not a two-card infinite" in e for e in errs) \
        and any("sorts after a flagged row" in e for e in errs) and any("Game Changer" in e for e in errs)
    bad2 = json.loads(json.dumps(doc)); bad2["as_of"] = "yesterday"; bad2["dimensions"]["x"] = {"candidates": []}
    errs2, _ = vcs.validate("edgar-vampires", bad2)
    assert any("ISO date" in e for e in errs2) and any("not a known dimension" in e for e in errs2)


def test_the_shortlist_joins_every_source_and_predicts_a_direction_not_a_number():
    """The Phase 3 join: scan dimensions + flags, the prescription's rank, the recon
    findings naming the card, the EDHREC page, assess's read, and a predicted direction
    per Forge axis read off the dimension — a rule stated in AXIS_PREDICTION, never a
    score. Twilight Prophet is the row that found the bug: in the tracked scan's top
    forty of drain and past the cut of draw, so a join over the tracked file read it as
    'not a draw card'; the join scans live."""
    doc = cs.shortlist("edgar-vampires", ["Twilight Prophet", "Exquisite Blood", "Not A Card"])
    rows = {r["name"]: r for r in doc["rows"]}
    tp = rows["Twilight Prophet"]
    assert set(tp["dimensions"]) >= {"drain", "draw"} and "draw.put_into_hand" in tp["matched"]["draw"]["oracle"]
    assert tp["predicted"]["forge.extra_draw_per_turn"] == "up" and tp["predicted"]["forge.drain_dealt"] == "up"
    assert tp["prescription"] and tp["prescription"]["as"] == "add"
    assert any(f["dimension"] == "draw" for f in tp["recon"])
    assert tp["edhrec"] and tp["edhrec"]["num_decks"] > 1000
    assert tp["assess"] and tp["assess"]["job"]
    eb = rows["Exquisite Blood"]
    assert "Sanguine Bond" in eb["infinite_with"] and eb["dimensions"]["drain"]["converter"] == "converter_loss_to_gain"
    assert rows["Not A Card"]["in_scan"] is False and rows["Not A Card"]["predicted"] == {}
    assert doc["sources"]["assess"] == "ok" and doc["sources"]["recon"] and doc["sources"]["prescription"]
    assert all(v in ("up", "down") or v.startswith(("up", "down")) for r in doc["rows"] for v in r["predicted"].values())
