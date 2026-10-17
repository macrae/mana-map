"""`manamap pilot deckcheck <slug>`: the deck's DeckCheck copy read back.

DeckCheck's API is read-only, so publishing stays a paste; this is the other
direction. Every test here goes through a `session=` stand-in with the offline
switch set to "0" for the test alone; one `@pytest.mark.network` test reads a
real deck and skips without `DECKCHECK_TEST_DECK_ID`.
"""

import copy
import json
import os
from pathlib import Path

import pytest
import requests

from manamap import config, net
from manamap.pilot import check_in, deck_link, deckcheck
from manamap.pilot.fetch_deck import parse_decklist

FIXTURE = Path(__file__).parent / "fixtures" / "deckcheck" / "sample.json"
DOC = json.loads(FIXTURE.read_text())
LIST = "1 Radagast of Rhosgobel *CMDR*\n1 Forest\n"
DC_URL = "https://deckcheck.co/deck/AbC_12-xyz"


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)


class FakeSession:
    def __init__(self, status=200, body=None):
        self.status, self.body, self.calls = status, body if body is not None else DOC, []

    def get(self, url, **kw):
        self.calls.append((url, kw.get("params"), kw.get("headers")))
        return _Resp(self.status, self.body)


@pytest.fixture
def fake_deck(tmp_path, monkeypatch):
    root = tmp_path / "data" / "decks"
    d = root / "ldeck"
    d.mkdir(parents=True)
    (d / "decklist.txt").write_text(LIST)
    (d / "cards.json").write_text(json.dumps({"cards": [
        {"name": "Radagast of Rhosgobel", "is_commander": True}, {"name": "Forest"}]}))
    monkeypatch.setattr(config, "DECKS_DIR", root)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setenv(net.OFFLINE_ENV, "0")
    monkeypatch.delenv(config.DECKCHECK_TOKEN_ENV, raising=False)
    (d / "links.json").write_text(json.dumps(
        {"deckcheck": {"url": DC_URL, "id": "AbC_12-xyz", "as_of": "2026-10-17"}}))
    return d


def _tree(path):
    return sorted((str(p.relative_to(path)), p.read_bytes()) for p in path.rglob("*") if p.is_file())


# ── projection ─────────────────────────────────────────────────────────────

def test_the_projection_renders_and_parses_back_to_the_same_entries():
    entries = deckcheck.project(DOC)
    text = check_in.render_decklist(entries)
    assert text.startswith("Commander:\n1 Radagast of Rhosgobel (LTC) 63\n")
    assert "1 Forest (LTC) 300 *F*" in text
    back = parse_decklist(text)
    assert [(e["name"], e["quantity"], e["board"], e["is_commander"], e.get("set"),
             e.get("collector_number"), e["foil"]) for e in back] == \
           [(e["name"], e["quantity"], e["board"], e["is_commander"], e.get("set"),
             e.get("collector_number"), e["foil"]) for e in entries]


def test_an_unknown_board_is_a_sideboard_and_a_list_of_cards_is_tolerated():
    doc = copy.deepcopy(DOC)
    doc["boards"]["maybeboard"] = {"count": 1, "cards": [
        {"quantity": 2, "card": {"name": "Llanowar Elves"}}]}
    entries = deckcheck.project(doc)
    side = [e for e in entries if e["board"] == "side"]
    assert side == [{"name": "Llanowar Elves", "quantity": 2, "board": "side",
                     "is_commander": False, "foil": False}]


# ── the report ─────────────────────────────────────────────────────────────

def test_an_in_sync_copy_reads_as_matching_the_working_list(fake_deck):
    s = FakeSession()
    r = deckcheck.report("ldeck", session=s)
    assert r["pull"] == {} and r["add"] == {} and r["matches"]["working"] is True
    assert r["matches"]["sleeved"] is None, "no paper lock, so no claim about one"
    assert r["in_sync"] is True and r["source_updated_at"] == "2026-10-17T10:05:00Z"
    assert r["url"] == DC_URL and r["cards"] == 2
    (url, params, headers) = s.calls[0]
    assert url == config.DECKCHECK_API_BASE + config.DECKCHECK_DECK_PATH
    assert params == {"deck_id": "AbC_12-xyz"} and headers == {}
    assert r["read_at"], "dated by the cache record it was written to"
    assert "IN SYNC" not in deckcheck.format_report(r)
    assert "matches the working list" in deckcheck.format_report(r)


def test_a_swapped_card_shows_as_pull_and_add(fake_deck):
    (fake_deck / "decklist.txt").write_text("1 Radagast of Rhosgobel *CMDR*\n1 Island\n")
    r = deckcheck.report("ldeck", session=FakeSession())
    assert r["pull"] == {"Island": 1} and r["add"] == {"Forest": 1}
    assert r["in_sync"] is False
    out = deckcheck.format_report(r)
    assert "- 1 Island" in out and "+ 1 Forest" in out and "--text | manamap pilot check-in" in out


def test_a_re_render_is_not_drift(fake_deck):
    """Same cards, a printing dropped: the copies match, nothing to pull or add."""
    (fake_deck / "decklist.txt").write_text("1 Radagast of Rhosgobel *CMDR*\n1 Forest (M21) 1\n")
    r = deckcheck.report("ldeck", session=FakeSession())
    assert r["pull"] == {} and r["add"] == {} and r["matches"]["working"] is True


def test_a_key_in_the_environment_rides_as_the_header(fake_deck, monkeypatch):
    monkeypatch.setenv(config.DECKCHECK_TOKEN_ENV, "sekrit")
    s = FakeSession()
    deckcheck.report("ldeck", session=s)
    assert s.calls[0][2] == {config.DECKCHECK_TOKEN_HEADER: "sekrit"}


def test_url_overrides_the_recorded_link_and_an_unlinked_deck_needs_one(fake_deck):
    (fake_deck / "links.json").unlink()
    with pytest.raises(SystemExit, match="no DeckCheck link"):
        deckcheck.report("ldeck", session=FakeSession())
    s = FakeSession()
    r = deckcheck.report("ldeck", url="https://deckcheck.co/deck/Other1", session=s)
    assert s.calls[0][1] == {"deck_id": "Other1"} and r["id"] == "Other1"


# ── failing open ───────────────────────────────────────────────────────────

def test_a_private_deck_and_a_missing_one_write_nothing(fake_deck, capsys):
    before = _tree(fake_deck.parent.parent)
    with pytest.raises(deckcheck.Private, match="private"):
        deckcheck.fetch(deck_id="AbC_12-xyz", session=FakeSession(403, {"error": "private"}))
    with pytest.raises(SystemExit, match="404"):
        deckcheck.fetch(deck_id="nope", session=FakeSession(404, {"error": "not found"}))
    assert _tree(fake_deck.parent.parent) == before
    # and through `main`, a private deck is one line and exit 1
    s = FakeSession(403, {"error": "private"})
    import manamap.pilot.deckcheck as m
    orig = m.fetch
    try:
        m.fetch = lambda deck_id=None, deck_url=None, session=None: orig(
            deck_id=deck_id, deck_url=deck_url, session=s)
        with pytest.raises(SystemExit) as exc:
            m.main(type("A", (), {"slug": "ldeck", "url": None, "text": False, "as_json": False})())
    finally:
        m.fetch = orig
    assert exc.value.code == 1 and "private" in capsys.readouterr().out
    assert _tree(fake_deck.parent.parent) == before


def test_offline_prints_one_line_exits_1_and_writes_nothing(fake_deck, monkeypatch, capsys):
    monkeypatch.setenv(net.OFFLINE_ENV, "1")
    before = _tree(fake_deck.parent.parent)
    with pytest.raises(SystemExit) as exc:
        deckcheck.main(type("A", (), {"slug": "ldeck", "url": None, "text": False,
                                      "as_json": False})())
    assert exc.value.code == 1
    assert "network could not be had" in capsys.readouterr().out
    assert _tree(fake_deck.parent.parent) == before


# ── the pull path ──────────────────────────────────────────────────────────

def test_text_is_exactly_the_rendered_list_and_check_in_accepts_it(fake_deck, monkeypatch, capsys):
    monkeypatch.setattr(deckcheck, "fetch", lambda **kw: DOC)
    deckcheck.main(type("A", (), {"slug": "ldeck", "url": None, "text": True, "as_json": False})())
    out = capsys.readouterr().out
    assert out == check_in.render_decklist(deckcheck.project(DOC))
    assert parse_decklist(out)[0]["name"] == "Radagast of Rhosgobel"


@pytest.mark.parametrize("src", [DC_URL, "https://www.deckcheck.co/decks/Q9/"])
def test_check_in_refuses_a_deckcheck_url_naming_the_pipe(src):
    with pytest.raises(SystemExit) as exc:
        check_in.read_list(src)
    assert str(exc.value) == check_in.DECKCHECK_REFUSAL
    assert "deckcheck <slug> --text | manamap pilot check-in" in str(exc.value)


def test_other_urls_still_get_the_moxfield_sentence():
    with pytest.raises(SystemExit) as exc:
        check_in.read_list("https://archidekt.com/decks/123")
    assert str(exc.value) == check_in.URL_REFUSAL


# ── the server: read-only, --write refused ─────────────────────────────────

def test_the_server_runs_it_read_only(fake_deck, monkeypatch):
    from manamap import serve

    monkeypatch.setattr(deckcheck, "fetch", lambda **kw: DOC)
    got = serve._cli(["deckcheck", "ldeck", "--json"])
    assert got["exit"] == 0 and json.loads(got["stdout"])["in_sync"] is True
    # There is no --write today; when one arrives it is in `_CLI_WRITE_ATTRS`
    # and refused there. Either way the server never writes through this.
    with pytest.raises(ValueError, match="not available over the API"):
        serve._cli(["deckcheck", "ldeck", "--write"])


# ── the live endpoint, opt-in ──────────────────────────────────────────────

@pytest.mark.network
def test_deckcheck_reads_one_deck_for_real():
    deck_id = os.environ.get("DECKCHECK_TEST_DECK_ID")
    if not deck_id:
        pytest.skip("set DECKCHECK_TEST_DECK_ID to one of the pilot's public decks")
    doc = deckcheck.fetch(deck_id=deck_id)
    assert deckcheck.project(doc), "no cards came back"
    assert deck_link.SERVICES["deckcheck"]["url_re"].match(doc["publicUrl"]), doc["publicUrl"]
