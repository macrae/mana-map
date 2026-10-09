"""`printings` and `printing/set`: the physical card, chosen from the page.

`printings` is a READ — Scryfall's search for every printing of a name, cached a
week under `net`'s `scryfall` service and projected to what a thumbnail strip
needs. `printing/set` is a WRITE to `decklist.txt` and its chain, so it is
POST-only and routes to the one writer, `check_in.set_printing`. Nothing here
reaches Scryfall: `net.get_json` is the seam, and the unit tier sets
`MANAMAP_NET_OFFLINE=1` so a forgotten patch fails with a sentence.
"""

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from manamap import net, serve

FIXTURES = Path(__file__).parent / "fixtures" / "scryfall"


def _fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def scryfall(monkeypatch):
    """`net.get_json` answers from the fixtures, recording what was asked."""
    calls = []

    def fake(url, *, params=None, headers=None, service=None, ttl_s=None, **kw):
        calls.append({"url": url, "params": params, "service": service, "ttl_s": ttl_s})
        q = (params or {}).get("q", "")
        if "Delver" in q:
            return _fixture("prints_dfc.json")
        if "Arcane Signet" in q:
            return _fixture("prints_arcane_signet.json")
        return {"object": "list", "has_more": False, "data": []}

    monkeypatch.setattr(net, "get_json", fake)
    return calls


def test_printings_projects_scryfall_to_the_strip_and_drops_digital(scryfall):
    out = serve.call("printings", {"name": "Arcane Signet"})
    assert out["name"] == "Arcane Signet" and out["truncated"] is False
    sets = [(p["set"], p["collector_number"]) for p in out["printings"]]
    assert sets == [("eld", "331"), ("sld", "1234")], "the Arena-only printing must go"
    sld = out["printings"][1]
    assert sld == {
        "set": "sld", "set_name": "Secret Lair Drop", "collector_number": "1234",
        "finishes": ["nonfoil", "foil"],
        "image": "https://cards.scryfall.io/normal/front/5/1/51d00000-0000-4000-8000-000000001234.jpg",
        "art_crop": "https://cards.scryfall.io/art_crop/front/5/1/51d00000-0000-4000-8000-000000001234.jpg",
        "faces": [], "artist": "Seb McKinnon", "border_color": "borderless",
        "frame_effects": ["inverted"], "released_at": "2024-02-05", "promo": False,
        "digital": False, "prices_usd": "12.00",
    }, "the cache-buster is stripped, the fields are exactly the strip's"
    # The ask is exact-name, every printing, oldest first, cached a week.
    assert scryfall[0]["params"] == {"q": '!"Arcane Signet"', "unique": "prints",
                                     "order": "released"}
    assert scryfall[0]["service"] == "scryfall" and scryfall[0]["ttl_s"] == 7 * 86400
    # Asked for, the digital printing is there.
    both = serve.call("printings", {"name": "Arcane Signet", "digital": True})
    assert [p["set"] for p in both["printings"]] == ["eld", "ha2", "sld"]


def test_printings_gives_a_dfc_both_faces(scryfall):
    out = serve.call("printings", {"name": "Delver of Secrets // Insectile Aberration"})
    [p] = out["printings"]
    front = "https://cards.scryfall.io/normal/front/1/1/11bf83bb-c95b-4b4f-9a56-ce7a1816307a.jpg"
    back = "https://cards.scryfall.io/normal/back/1/1/11bf83bb-c95b-4b4f-9a56-ce7a1816307a.jpg"
    assert p["image"] == front, "a transform card has no top-level image; the front stands in"
    assert [f["name"] for f in p["faces"]] == ["Delver of Secrets", "Insectile Aberration"]
    assert [f["image"] for f in p["faces"]] == [front, back]


def test_printings_offline_is_the_endpoints_error_with_nets_sentence(monkeypatch):
    def refuse(url, **kw):
        raise net.Offline("MANAMAP_NET_OFFLINE=1: wanted GET " + url + "; nothing was written.")
    monkeypatch.setattr(net, "get_json", refuse)
    with pytest.raises(RuntimeError, match="nothing was written"):
        serve.call("printings", {"name": "Sol Ring"})
    with pytest.raises(ValueError):
        serve.call("printings", {})


def test_printings_follows_pages_up_to_the_cap(monkeypatch):
    pages = []

    def fake(url, *, params=None, **kw):
        pages.append((url, params))
        n = len(pages)
        card = {"name": "Forest", "set": f"s{n}", "collector_number": str(n), "digital": False,
                "image_uris": {"normal": f"https://cards.scryfall.io/normal/{n}.jpg?x"}}
        return {"has_more": True, "next_page": f"https://api.scryfall.com/cards/search?page={n + 1}",
                "data": [card]}
    monkeypatch.setattr(net, "get_json", fake)
    out = serve.call("printings", {"name": "Forest"})
    assert len(pages) == serve.PRINTINGS_MAX_PAGES and out["truncated"] is True
    assert pages[0][1] is not None and pages[1][1] is None, "next_page carries its own query"
    assert [p["set"] for p in out["printings"]] == ["s1", "s2", "s3", "s4"]


def test_printing_set_is_post_only_and_routes_to_the_one_writer(monkeypatch):
    from manamap.pilot import check_in
    assert "printings" in serve.GETTABLE
    assert "printing/set" in serve.ENDPOINTS and "printing/set" not in serve.GETTABLE
    seen = {}

    def fake(slug, name, set_code, collector_number, foil=False, run_chain=True, branch=None):
        seen.update(slug=slug, name=name, set=set_code, cn=collector_number, foil=foil,
                    run_chain=run_chain)
        return {"changed": True, "line": "1 Sol Ring (SLD) 1234 *F*", "ran": ["fetch-deck"]}
    monkeypatch.setattr(check_in, "set_printing", fake)
    out = serve.call("printing/set", {"slug": "d", "card": "Sol Ring", "set": "SLD",
                                      "collector_number": 1234, "foil": "true"})
    assert seen == {"slug": "d", "name": "Sol Ring", "set": "SLD", "cn": "1234", "foil": True,
                    "run_chain": True}, "coerced to strings and a bool, chain on"
    assert out["changed"] is True and out["line"] == "1 Sol Ring (SLD) 1234 *F*"
    assert out["slug"] == "d" and out["card"] == "Sol Ring"
    # A refusal from the writer is the caller's mistake, not a crashed server.
    monkeypatch.setattr(check_in, "set_printing",
                        lambda *a, **k: (_ for _ in ()).throw(SystemExit("not in decklist.txt")))
    with pytest.raises(ValueError, match="not in decklist"):
        serve.call("printing/set", {"slug": "d", "card": "X", "set": "a", "collector_number": "1"})
    with pytest.raises(ValueError):
        serve.call("printing/set", {"slug": "d", "card": "X"})


def test_a_get_carries_its_query_string_and_a_write_is_refused_on_get(scryfall):
    """`GETTABLE` promised `printings` and `do_GET` dispatched an EMPTY payload, so
    the promise could not be kept — `?name=` reaches the endpoint now, coerced
    like a POST body. `printing/set` on GET is a 405 naming the method."""
    httpd = serve.serve(port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        host, port = httpd.server_address[0], httpd.server_address[1]
        with urllib.request.urlopen(f"http://{host}:{port}/api/printings?name=Arcane%20Signet") as r:
            doc = json.loads(r.read())
        assert doc["ok"] and [p["set"] for p in doc["result"]["printings"]] == ["eld", "sld"]
        try:
            urllib.request.urlopen(f"http://{host}:{port}/api/printing/set?slug=d")
            raise AssertionError("a write answered a GET")
        except urllib.error.HTTPError as exc:
            assert exc.code == 405
            assert "POST" in json.loads(exc.read())["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()
