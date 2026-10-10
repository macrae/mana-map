"""`prices` — a list's card prices as DATED evidence, and the form gate on the file.

Every network call here goes through a stand-in `session=` handed to `manamap.net`,
so the unit tier's offline switch is lifted per test (the cache sits in `tmp_path`
too). The one test that reaches Mana Pool for real is marked `network` and skips
without the two environment variables.
"""

import json
from datetime import date

import pytest
import requests

from manamap import config, net
from manamap.pilot import net_change, prices, validate_prices
from manamap.pilot.agent_cache import CARD_SEMANTIC_FIELDS, cards_semantic_digest
from manamap.pilot.fetch_deck import shape_card

SID_A, SID_B, SID_C = "a" * 32, "b" * 32, "c" * 32

CARDS = {
    "deck": "fixture", "decklist_sha256": "0" * 64,
    "cards": [
        {"name": "Sol Ring", "quantity": 1, "set": "c20", "collector_number": "232",
         "scryfall_id": SID_A},
        {"name": "Island", "quantity": 11, "set": "c20", "collector_number": "300",
         "scryfall_id": SID_B},
        # No scryfall_id — a cards.json from before fetch-deck carried one.
        {"name": "Windfall", "quantity": 1, "set": "c20", "collector_number": "111"},
        {"name": "Obscure Promo", "quantity": 1, "set": "pxx", "collector_number": "1",
         "scryfall_id": SID_C},
    ],
}

FEED = [
    {"url": "https://manapool.com/card/c20/232/sol-ring", "name": "Sol Ring", "set_code": "c20",
     "number": "232", "scryfall_id": SID_A, "price_cents": 150, "price_cents_lp_plus": 150,
     "price_cents_nm": 199, "price_cents_nm_foil": 1200, "available_quantity": 9},
    {"url": "https://manapool.com/card/c20/300/island", "name": "Island", "set_code": "c20",
     "number": "300", "scryfall_id": SID_B, "price_cents": 10, "price_cents_lp_plus": None,
     "price_cents_nm": 25, "price_cents_nm_foil": None, "available_quantity": 0},
    # Windfall listed under a DIFFERENT printing than the deck's; the name is the join.
    {"url": "https://manapool.com/card/c21/50/windfall", "name": "Windfall", "set_code": "c21",
     "number": "50", "scryfall_id": "d" * 32, "price_cents": 300, "price_cents_lp_plus": 300,
     "price_cents_nm": 350, "price_cents_nm_foil": None, "available_quantity": 2},
    {"url": "https://manapool.com/card/c19/50/windfall", "name": "Windfall", "set_code": "c19",
     "number": "50", "scryfall_id": "e" * 32, "price_cents": 500, "price_cents_lp_plus": None,
     "price_cents_nm": 500, "price_cents_nm_foil": None, "available_quantity": 1},
]

SCRYFALL = {
    "data": [
        {"id": SID_A, "name": "Sol Ring", "set": "c20", "collector_number": "232",
         "scryfall_uri": "https://scryfall.com/card/c20/232/sol-ring",
         "prices": {"usd": "1.99", "usd_foil": "12.00"}},
        {"id": SID_B, "name": "Island", "set": "c20", "collector_number": "300",
         "scryfall_uri": "https://scryfall.com/card/c20/300/island",
         "prices": {"usd": "0.25", "usd_foil": None}},
        {"id": "f" * 32, "name": "Windfall", "set": "c20", "collector_number": "111",
         "scryfall_uri": "https://scryfall.com/card/c20/111/windfall?utm_source=api",
         "prices": {"usd": "3.50", "usd_foil": None}},
        {"id": SID_C, "name": "Obscure Promo", "set": "pxx", "collector_number": "1",
         "scryfall_uri": "https://scryfall.com/card/pxx/1/obscure-promo",
         "prices": {"usd": None, "usd_foil": None}},
    ],
    "not_found": [],
}


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            exc = requests.HTTPError(f"{self.status_code}")
            exc.response = self
            raise exc


class FakeSession:
    """`get` answers the Mana Pool feed, `post` the Scryfall collection; every
    call is recorded so a test can say which service was reached."""

    def __init__(self, feed_status=200, feed=FEED, scryfall=SCRYFALL):
        self.calls, self.feed_status, self.feed, self.scryfall = [], feed_status, feed, scryfall

    def get(self, url, **kw):
        self.calls.append(("get", url, kw))
        return _Resp(self.feed_status, self.feed)

    def post(self, url, **kw):
        self.calls.append(("post", url, kw))
        return _Resp(200, self.scryfall)

    def close(self):
        pass


@pytest.fixture
def fake_deck(tmp_path, monkeypatch):
    """A deck and a branch under a temporary tree, the cache beside them, the
    offline switch OFF (the session is a stand-in), no Mana Pool token."""
    root = tmp_path / "decks"
    (root / "fixture" / "branches" / "b1").mkdir(parents=True)
    (root / "fixture" / "cards.json").write_text(json.dumps(CARDS))
    branch_cards = dict(CARDS, decklist_sha256="1" * 64,
                        cards=CARDS["cards"][:2])
    (root / "fixture" / "branches" / "b1" / "cards.json").write_text(json.dumps(branch_cards))
    monkeypatch.setattr(config, "DECKS_DIR", root)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setenv(net.OFFLINE_ENV, "0")
    monkeypatch.delenv(prices.TOKEN_ENV, raising=False)
    monkeypatch.delenv(prices.EMAIL_ENV, raising=False)
    return root


def _with_token(monkeypatch):
    monkeypatch.setenv(prices.TOKEN_ENV, "sekrit-token")
    monkeypatch.setenv(prices.EMAIL_ENV, "pilot@example.com")


# ── the document ─────────────────────────────────────────────────────────

def test_the_manapool_document_has_the_shape_the_readers_quote(fake_deck, monkeypatch):
    _with_token(monkeypatch)
    s = FakeSession()
    doc = prices.build("fixture", source="auto", session=s)
    assert doc["source"] == "manapool" and doc["currency"] == "USD"
    assert doc["slug"] == "fixture" and doc["branch"] is None
    assert doc["as_of"] == date.today().isoformat()
    assert doc["decklist_sha256"] == "0" * 64
    sol = doc["cards"]["Sol Ring"]
    assert sol == {"scryfall_id": SID_A, "printing": "(C20) 232", "nm_cents": 199,
                   "lp_cents": 150, "foil_cents": 1200, "quantity": 1, "note": None,
                   "url": "https://manapool.com/card/c20/232/sol-ring"}
    island = doc["cards"]["Island"]
    assert island["quantity"] == 11 and island["nm_cents"] == 25
    assert island["lp_cents"] is None and island["note"] == "out of stock"
    # The deck's Windfall printing is not listed: the CHEAPEST listing of the name,
    # labelled as such and carrying that printing, not the deck's.
    wf = doc["cards"]["Windfall"]
    assert wf["nm_cents"] == 350 and wf["printing"] == "(C21) 50"
    assert wf["note"] == "cheapest listed printing, not the deck's"
    assert doc["missing"] == ["Obscure Promo"]
    assert doc["total_cents"] == 199 + 25 * 11 + 350
    assert doc["total_nm_cents"] == 199 + 25 + 350
    # The auth rode in the headers of the feed GET, and the feed path is config's.
    get = [c for c in s.calls if c[0] == "get"]
    assert len(get) == 1
    assert get[0][1] == config.MANAPOOL_API_BASE + config.MANAPOOL_PRICES_PATH
    assert get[0][2]["headers"] == {config.MANAPOOL_TOKEN_HEADER: "sekrit-token",
                                    config.MANAPOOL_EMAIL_HEADER: "pilot@example.com"}
    # Windfall had no scryfall_id, so ONE collection POST resolved it.
    assert len([c for c in s.calls if c[0] == "post"]) == 1
    assert not validate_prices.validate("fixture", None, doc,
                                        {c["name"] for c in CARDS["cards"]})


def test_the_feed_is_public_so_no_token_still_reads_mana_pool(fake_deck):
    """Verified against the OpenAPI spec 2026-10-09: `GET prices/singles` carries
    no security. No token means no auth headers, not no Mana Pool."""
    s = FakeSession()
    doc = prices.build("fixture", session=s)
    assert doc["source"] == "manapool"
    get = [c for c in s.calls if c[0] == "get"]
    assert len(get) == 1 and get[0][2]["headers"] == {}
    assert doc["cards"]["Sol Ring"]["nm_cents"] == 199


def test_the_documented_envelope_is_read_and_its_stamp_kept(fake_deck):
    """The spec's shape: `{meta: {as_of, base_url}, data: [...]}`. The feed's own
    timestamp is kept beside the day we read it; the validator form-checks it."""
    stamp = "2026-10-10T01:21:13.523Z"
    s = FakeSession(feed={"meta": {"as_of": stamp, "base_url": "https://manapool.com"},
                          "data": FEED})
    doc = prices.build("fixture", session=s)
    assert doc["source"] == "manapool" and doc["feed_as_of"] == stamp
    names = {c["name"] for c in CARDS["cards"]}
    assert not validate_prices.validate("fixture", None, doc, names)
    assert validate_prices.validate("fixture", None, dict(doc, feed_as_of="yesterday"), names)
    assert validate_prices.validate("fixture", None, dict(doc, source="scryfall"), names)


def test_scryfall_is_the_fallback_and_says_so_in_the_document(fake_deck):
    s = FakeSession()
    doc = prices.build("fixture", source="scryfall", session=s)
    assert doc["source"] == "scryfall" and "feed_as_of" not in doc
    assert [c[0] for c in s.calls] == ["post"]
    assert doc["cards"]["Sol Ring"]["nm_cents"] == 199
    assert doc["cards"]["Sol Ring"]["foil_cents"] == 1200
    assert doc["cards"]["Sol Ring"]["lp_cents"] is None
    assert doc["cards"]["Sol Ring"]["url"] == "https://scryfall.com/card/c20/232/sol-ring"
    # The id came back from Scryfall for the card that had none.
    assert doc["cards"]["Windfall"]["scryfall_id"] == "f" * 32
    # A URL with a query string is dropped rather than written.
    assert doc["cards"]["Windfall"]["url"] is None
    assert doc["missing"] == ["Obscure Promo"]
    assert doc["total_cents"] == 199 + 25 * 11 + 350
    assert not validate_prices.validate("fixture", None, doc,
                                        {c["name"] for c in CARDS["cards"]})


def test_a_404_on_the_feed_path_falls_back_to_scryfall_with_one_line(fake_deck, monkeypatch, capsys):
    """The path is reconstructed, not read from the docs; a wrong one must not be
    a traceback."""
    _with_token(monkeypatch)
    s = FakeSession(feed_status=404)
    doc = prices.build("fixture", session=s)
    assert doc["source"] == "scryfall"
    out = capsys.readouterr().out
    assert "404" in out and "MANAPOOL_PRICES_PATH" in out


def test_a_server_error_on_the_feed_is_not_a_wrong_path(fake_deck, monkeypatch):
    """A 503 after every retry is an outage, and propagates — only 400/404 mean
    "the path is wrong, use Scryfall"."""
    _with_token(monkeypatch)
    monkeypatch.setattr(config, "NET_BACKOFF_S", 0.0)
    with pytest.raises(requests.HTTPError):
        prices.build("fixture", session=FakeSession(feed_status=503))


def test_source_manapool_refuses_when_the_feed_path_is_wrong(fake_deck):
    with pytest.raises(SystemExit, match="400/404"):
        prices.build("fixture", source="manapool", session=FakeSession(feed_status=404))


def test_source_scryfall_never_touches_the_feed(fake_deck, monkeypatch):
    _with_token(monkeypatch)
    s = FakeSession()
    doc = prices.build("fixture", source="scryfall", session=s)
    assert doc["source"] == "scryfall"
    assert [c[0] for c in s.calls] == ["post"]


def test_a_branch_is_priced_from_its_own_list_and_written_beside_it(fake_deck, monkeypatch):
    doc = prices.build("fixture", "b1", session=FakeSession())
    assert doc["branch"] == "b1" and doc["decklist_sha256"] == "1" * 64
    assert set(doc["cards"]) == {"Sol Ring", "Island"} and doc["missing"] == []
    monkeypatch.setattr(prices, "build", lambda *a, **k: doc)
    prices.main(type("A", (), {"slug": "fixture", "branch": "b1", "source": "auto",
                               "write": True, "as_json": False})())
    path = fake_deck / "fixture" / "branches" / "b1" / "prices.json"
    assert json.loads(path.read_text()) == doc
    assert not (fake_deck / "fixture" / "prices.json").exists()


def test_write_refuses_offline_and_writes_nothing(fake_deck, monkeypatch, capsys):
    monkeypatch.setenv(net.OFFLINE_ENV, "1")
    with pytest.raises(SystemExit) as exit_:
        prices.main(type("A", (), {"slug": "fixture", "branch": None, "source": "scryfall",
                                   "write": True, "as_json": False})())
    assert exit_.value.code == 1
    assert "network could not be had" in capsys.readouterr().out
    assert not (fake_deck / "fixture" / "prices.json").exists()


def test_the_report_names_the_source_the_date_and_the_missing(fake_deck, capsys):
    doc = prices.build("fixture", session=FakeSession())
    prices.print_report(doc)
    out = capsys.readouterr().out
    assert f"from manapool as of {doc['as_of']}" in out
    assert "1 missing" in out and "Obscure Promo" in out
    assert "Sol Ring" in out and "$1.99" in out


# ── the validator: one negative per rule ─────────────────────────────────

def _good():
    return {
        "slug": "fixture", "branch": None, "as_of": "2026-10-09", "source": "scryfall",
        "currency": "USD", "decklist_sha256": "0" * 64,
        "cards": {"Sol Ring": {"scryfall_id": SID_A, "printing": "(C20) 232", "nm_cents": 199,
                               "lp_cents": None, "foil_cents": 1200, "quantity": 1,
                               "note": None, "url": "https://scryfall.com/card/c20/232/x"},
                  "Island": {"scryfall_id": SID_B, "printing": "(C20) 300", "nm_cents": 25,
                             "lp_cents": None, "foil_cents": None, "quantity": 11,
                             "note": None, "url": None}},
        "total_cents": 199 + 25 * 11, "total_nm_cents": 199 + 25,
        "missing": ["Windfall"],
    }


NAMES = {"Sol Ring", "Island", "Windfall"}


def test_a_well_formed_document_passes():
    assert validate_prices.validate("fixture", None, _good(), NAMES) == []


@pytest.mark.parametrize("mutate, phrase", [
    (lambda d: d.pop("as_of"), "missing required key 'as_of'"),
    (lambda d: d.update(as_of="yesterday"), "not an ISO date"),
    (lambda d: d.update(source="tcgplayer"), "source 'tcgplayer'"),
    (lambda d: d.update(currency="EUR"), "not USD"),
    (lambda d: d.update(slug="other"), "artifact lives in fixture/"),
    (lambda d: d.update(branch="b1"), "branch is 'b1'"),
    (lambda d: d["cards"].update({"Not In Deck": d["cards"]["Island"]}), "not in the list's cards.json"),
    (lambda d: d["missing"].append("Also Not In Deck"), "'Also Not In Deck' is not in the list"),
    (lambda d: d["missing"].append("Sol Ring"), "missing and cards overlap: Sol Ring"),
    (lambda d: d["cards"]["Sol Ring"].update(nm_cents=-1), "nm_cents -1 is not a non-negative"),
    (lambda d: d["cards"]["Sol Ring"].update(foil_cents=12.5), "foil_cents 12.5 is not"),
    (lambda d: d["cards"]["Sol Ring"].update(lp_cents="150"), "lp_cents '150' is not"),
    (lambda d: d["cards"]["Island"].update(quantity=0), "quantity 0 is not a positive"),
    (lambda d: d["cards"]["Sol Ring"].pop("printing"), "lacks 'printing'"),
    (lambda d: d.update(total_cents=1), "total_cents 1 != sum of nm_cents x quantity (474)"),
    (lambda d: d.update(total_nm_cents=1), "total_nm_cents 1 != sum of nm_cents (224)"),
    (lambda d: d["cards"]["Sol Ring"].update(url="https://evil.example/x"), "not on an allowed host"),
    (lambda d: d["cards"]["Sol Ring"].update(url="http://scryfall.com/x"), "not on an allowed host"),
    (lambda d: d["cards"]["Sol Ring"].update(url="https://manapool.com/x?token=abc"),
     "carries a query string"),
])
def test_each_rule_fires_on_its_own_defect(mutate, phrase):
    doc = _good()
    mutate(doc)
    errors = validate_prices.validate("fixture", None, doc, NAMES)
    assert any(phrase in e for e in errors), (phrase, errors)


def test_the_command_exits_one_on_a_bad_file_and_is_quiet_when_absent(fake_deck, capsys):
    args = type("A", (), {"slug": "fixture", "branch": None})()
    validate_prices.main(args)
    assert "absent means absent" in capsys.readouterr().out
    bad = _good()
    bad["cards"]["Sol Ring"]["url"] = "https://manapool.com/x?token=abc"
    (fake_deck / "fixture" / "prices.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit) as exit_:
        validate_prices.main(args)
    assert exit_.value.code == 1
    assert "query string" in capsys.readouterr().out
    good = _good()
    good["missing"] = ["Windfall", "Obscure Promo"]
    (fake_deck / "fixture" / "prices.json").write_text(json.dumps(good))
    validate_prices.main(args)
    assert capsys.readouterr().out.startswith("OK")


def test_the_registry_gates_the_artifact_as_form_only_and_never_as_a_stage():
    from manamap.pilot import deck_status, regen
    assert deck_status.VALIDATED["prices.json"] == "manamap.pilot.validate_prices"
    assert "prices.json" not in {row[1] for row in deck_status.STAGES}
    assert not any("prices" in str(row) for row in regen.STAGES)


# ── fetch-deck's new field, and the digest it must not touch ─────────────

def test_scryfall_id_is_shaped_in_and_is_not_agent_semantic(tmp_path):
    sc = {"name": "Sol Ring", "id": SID_A, "mana_cost": "{1}", "cmc": 1.0,
          "type_line": "Artifact", "oracle_text": "{T}: Add {C}{C}.", "colors": [],
          "color_identity": [], "keywords": [], "layout": "normal", "set": "c20",
          "collector_number": "232", "image_uris": {"normal": "https://x/y.jpg"}}
    shaped = shape_card(sc, 1, False)
    assert shaped["scryfall_id"] == SID_A
    assert "scryfall_id" not in CARD_SEMANTIC_FIELDS
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    plain = dict(shaped)
    plain.pop("scryfall_id")
    before.write_text(json.dumps({"cards": [plain]}))
    after.write_text(json.dumps({"cards": [shaped]}))
    assert cards_semantic_digest(before) == cards_semantic_digest(after), (
        "the new key invalidates every agent routine on the next fetch")


# ── net-change's bill ────────────────────────────────────────────────────

def test_the_bill_in_dollars_is_said_at_print_time_never_written_into_the_report(fake_deck):
    bill = {"counts": {"buy": 2, "in_deck": 1},
            "cards": [{"name": "Sol Ring", "state": "buy", "where": [], "free": False},
                      {"name": "Island", "state": "buy", "where": [], "free": False},
                      {"name": "Windfall", "state": "in_deck", "where": [], "free": False}]}
    doc = {"slug": "fixture", "branch": "b1", "bill": bill}
    # No prices file: no line. Absent means absent.
    assert net_change.priced_bill_line(doc) is None
    priced = _good()
    priced["branch"] = "b1"
    (fake_deck / "fixture" / "branches" / "b1" / "prices.json").write_text(json.dumps(priced))
    assert net_change.priced_bill_line(doc) == "≈ $4.74 to buy (as of 2026-10-09, scryfall)"
    # THE REPORT NEVER CARRIES IT: `cost()` is a measurement of the list, and a
    # price refresh must not make a tracked net_change.json stale (the regen gate
    # caught exactly that on 2026-10-09).
    got = net_change.cost(doc)
    for key in ("buy_cents", "buy_unpriced", "prices_as_of", "prices_source"):
        assert key not in got, key
    assert "$" not in got["reads_as"]
    # A buy card the file does not price is NAMED, never counted as zero.
    bill["cards"].append({"name": "Windfall", "state": "buy", "where": [], "free": False})
    line = net_change.priced_bill_line(doc)
    assert line.startswith("≈ $4.74 to buy") and "1 unpriced: Windfall" in line
    # The deck's own prices file is not the branch's.
    (fake_deck / "fixture" / "branches" / "b1" / "prices.json").unlink()
    (fake_deck / "fixture" / "prices.json").write_text(json.dumps(_good()))
    assert net_change.priced_bill_line(doc) is None


def test_a_test_skeleton_with_no_slug_still_costs_out():
    doc = {"bill": {"counts": {}, "cards": []}}
    assert "buy_cents" not in net_change.cost(doc)
    assert net_change.priced_bill_line(doc) is None


# ── the one real call ────────────────────────────────────────────────────

@pytest.mark.network
def test_mana_pool_prices_one_card_for_real(monkeypatch, tmp_path):
    """Integration: the reconstructed feed path against the live API. A 400/404 here
    is the signal to edit `config.MANAPOOL_PRICES_PATH` — the test then reads the
    Scryfall fallback and says so in its failure."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.delenv(net.OFFLINE_ENV, raising=False)
    feed = prices.manapool_feed()
    assert feed is not None, "the feed path answered 400/404 — edit config.MANAPOOL_PRICES_PATH"
    assert feed, "the feed answered with no rows"
    row = next(iter(feed.values()))
    assert row.get("name") and prices._row_nm(row) is not None
