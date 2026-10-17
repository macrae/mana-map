"""`pods <name> --power`: three readings of power per seat, never merged, and a
Spearman cell that refuses below `calibrate.MIN_DECKS`."""

import json
from datetime import date

import pytest

from manamap import config
from manamap.pilot import calibrate, dti
from manamap.sim import pods, table_power

BM = {k: "B" for k in sorted(dti.BENCHMARKS)}


def _doc(score, bracket, name="x"):
    return {"id": name, "publicUrl": f"https://deckcheck.co/deck/{name}", "name": name,
            "updatedAtUtc": "2026-10-17T00:00:00Z",
            "dti": {"score": score, "maxScore": 96, "bracket": bracket, "benchmarks": dict(BM)},
            "boards": {"commanders": {"cards": [{"quantity": 1, "card": {"name": "Krenko, Mob Boss"}}]},
                       "mainboard": {"cards": [{"quantity": 1, "card": {"name": "Mountain"}}]}}}


@pytest.fixture
def world(tmp_path, monkeypatch):
    data = tmp_path / "data"
    decks, opps, podsdir = data / "decks", data / "opponents", data / "pods"
    for d in (decks, opps, podsdir):
        d.mkdir(parents=True)
    monkeypatch.setattr(config, "DECKS_DIR", decks)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(pods, "PODS_DIR", podsdir)
    # ours: a deck with a stamped floor and a DTI on file
    mine = decks / "mine"
    mine.mkdir()
    (mine / "decklist.txt").write_text("1 Sol Ring\n")
    (mine / "cards.json").write_text(json.dumps({"decklist_sha256": "a" * 64, "cards": []}))
    (mine / "bracket_report.json").write_text(json.dumps(
        {"floor": 3, "floor_name": "Upgraded", "drivers": [], "decklist_sha256": "a" * 64}))
    # two opponents as stand-ins
    for slug in ("opp-a", "opp-b"):
        (opps / slug).mkdir()
        (opps / slug / "decklist.txt").write_text("1 Krenko, Mob Boss *CMDR*\n1 Mountain\n")
    return {"decks": decks, "opps": opps, "pods": podsdir}


def _pod(world, seats, name="t"):
    (world["pods"] / f"{name}.json").write_text(json.dumps({"name": name, "seats": seats}))
    return name


FLOOR = lambda n, c, ref: {"floor": 1, "floor_name": "Exhibition", "drivers_n": 0}   # noqa: E731
NO_CAL = {"measured": False, "note": "never simulated"}


def test_zero_partial_and_full_dti_coverage(world, monkeypatch):
    monkeypatch.setattr(table_power, "_floor_of", FLOOR)
    name = _pod(world, [{"slug": "mine"}, {"slug": "opp-a", "bracket": 3}, {"slug": "opp-b"}])
    doc = table_power.table(name, calibration=NO_CAL, reference=())
    assert doc["matched_within"] == {"within": None, "seats_with_dti": 0, "seats": 3}
    assert doc["spearman"]["verdict"] == "NOT ANSWERABLE" and doc["spearman"]["n"] == 0
    assert "absent_because" in doc["balance"]
    assert [r["floor"]["floor"] for r in doc["seats"]] == [3, 1, 1]
    assert doc["seats"][0]["floor"]["source"] == "bracket_report.json"
    assert "stand-in" in doc["seats"][1]["floor"]["source"]
    assert "typed B3 ≠ floor B1" in doc["seats"][1]["flags"]
    out = table_power.format_table(doc)
    assert "0 of 3 seats have a DTI" in out and "NOT ANSWERABLE at n=0" in out

    # partial: ours gets a DTI on file; one friend's real deck by URL, read live
    dti.append("mine", dti.reading_from_payload(_doc(40, 3), deck_id="m", url="https://deckcheck.co/deck/m",
                                                as_of="2026-10-17", decklist_sha256="a" * 64),
               today=date(2026, 10, 17))
    name = _pod(world, [{"slug": "mine"}, {"slug": "opp-a", "deckcheck": "https://deckcheck.co/deck/A1"},
                        {"slug": "opp-b"}])
    doc = table_power.table(name, calibration=NO_CAL, reference=(),
                            fetch=lambda **kw: _doc(55, 4, "A1"))
    m = doc["matched_within"]
    assert m == {"within": 15, "seats_with_dti": 2, "seats": 3}
    live = doc["seats"][1]
    assert live["dti"]["score"] == 55 and live["dti"]["source"] == "deckcheck.co (live)"
    assert live["floor"]["source"] == "computed from the DeckCheck copy"
    assert "matched within 15 DTI points over 2 of 3 seats; 1 without a DTI" in table_power.format_table(doc)


def test_the_flags_name_every_disagreement_and_only_with_a_clock_the_age(world, monkeypatch):
    monkeypatch.setattr(table_power, "_floor_of", FLOOR)
    dti.append("mine", dti.reading_from_payload(_doc(40, 2), deck_id="m", url="https://deckcheck.co/deck/m",
                                                as_of="2026-09-01", decklist_sha256="b" * 64),
               today=date(2026, 9, 1))
    name = _pod(world, [{"slug": "mine", "bracket": 4}])
    row = table_power.table(name, calibration=NO_CAL, reference=())["seats"][0]
    assert "typed B4 ≠ floor B3" in row["flags"] and "typed B4 ≠ DTI B2" in row["flags"]
    assert any("below floor B3" in f for f in row["flags"])
    assert "DTI describes an older list" in row["flags"]
    assert not any("days old" in f for f in row["flags"]), "no clock, no age"
    row = table_power.table(name, calibration=NO_CAL, reference=(), today=date(2026, 12, 1))["seats"][0]
    assert any("days old" in f for f in row["flags"])


def test_a_deckcheck_seat_offline_is_an_absence_and_the_table_still_renders(world, monkeypatch):
    from manamap import net
    monkeypatch.setattr(table_power, "_floor_of", FLOOR)

    def down(**kw):
        raise net.Offline("no network")
    name = _pod(world, [{"slug": "opp-a", "deckcheck": "https://deckcheck.co/deck/A1"}])
    doc = table_power.table(name, calibration=NO_CAL, reference=(), fetch=down)
    row = doc["seats"][0]
    assert "could not reach DeckCheck" in row["floor"]["absent_because"] and row["dti"] is None
    assert "could not reach DeckCheck" in table_power.format_table(doc)


def test_spearman_is_measured_only_from_ten_seats_with_both_readings(world, monkeypatch):
    monkeypatch.setattr(table_power, "_floor_of", FLOOR)
    seats, cal_rows = [], []
    for i in range(10):
        slug = f"s{i}"
        (world["opps"] / slug).mkdir()
        (world["opps"] / slug / "decklist.txt").write_text("1 Mountain\n")
        seats.append({"slug": slug, "deckcheck": f"https://deckcheck.co/deck/{slug}"})
        cal_rows.append({"seat": slug, "wins": i, "games": 20, "rate": i / 20,
                         "ci95": [0.0, 1.0], "share_of_fair": round((i / 20) / 0.1, 2)})
    name = _pod(world, seats)
    scores = {f"s{i}": 30 + 5 * i for i in range(10)}
    cal = {"measured": True, "seats": cal_rows, "balance": {"dominant": [], "floor": [], "spread": 0.45}}
    doc = table_power.table(name, calibration=cal, reference=(),
                            fetch=lambda deck_url=None, **kw: _doc(scores[deck_url.rsplit("/", 1)[-1]], 3))
    s = doc["spearman"]
    assert s["verdict"] == "measured" and s["n"] == 10 and s["rho"] == 1.0
    assert s["critical_rho"] == calibrate.critical_rho(10) and s["significant"] is True
    assert doc["seats"][0]["share_of_fair"] == 0.0 and doc["seats"][9]["games"] == 20


def test_the_public_spearman_is_the_one_calibrate_uses():
    assert calibrate.spearman is calibrate._spearman


def test_a_seat_may_carry_a_deckcheck_url_and_nothing_else_reads_it(world):
    name = _pod(world, [{"slug": "opp-a", "deckcheck": "https://deckcheck.co/deck/A1"}])
    doc = pods.load(name)
    assert doc["seats"][0]["deckcheck"].startswith("https://deckcheck.co/")
    assert pods.compose(name)["seats"][0]["slug"] == "opp-a"
