"""deck-link, validate-links, the manifest's `links`, and check-in's URL refusal.

A link is recorded by hand because nothing here can reach Moxfield (no API; its
Cloudflare front 403s server-side requests). So the writer and its gate check FORM:
the host, the id, the date — and a URL handed to `check-in --from` is refused with
the one sentence that says how to get the list in instead.
"""

import argparse
import json
from datetime import date

import pytest

from manamap.pilot import check_in, deck_link, deck_manifest, validate_links

URL = "https://moxfield.com/decks/AbC_12-xyz"


@pytest.fixture
def decks(tmp_path, monkeypatch):
    root = tmp_path / "decks"
    monkeypatch.setattr("manamap.config.DECKS_DIR", root)
    monkeypatch.setattr("manamap.config.MANUALS_DIR", tmp_path / "manuals")
    d = root / "ldeck"
    d.mkdir(parents=True)
    (d / "decklist.txt").write_text("1 Radagast of Rhosgobel *CMDR*\n1 Forest\n")
    (d / "cards.json").write_text(json.dumps({"cards": [
        {"name": "Radagast of Rhosgobel", "is_commander": True,
         "type_line": "Legendary Creature", "art_crop": "r.jpg"},
        {"name": "Forest", "type_line": "Basic Land — Forest"}]}))
    return root


def _args(**kw):
    base = {"slug": "ldeck", "action": "moxfield", "url": None, "note": None, "remove": False}
    base.update(kw)
    return argparse.Namespace(**base)


def test_write_list_and_remove(decks, capsys):
    e = deck_link.set_link("ldeck", "moxfield", URL + "/", note="the paper list",
                           today=date(2026, 10, 9))
    assert e == {"url": URL, "id": "AbC_12-xyz", "as_of": "2026-10-09",
                 "note": "the paper list"}
    doc = json.loads((decks / "ldeck" / "links.json").read_text())
    assert doc == {"moxfield": e}
    assert validate_links.validate(doc) == []

    deck_link.main(_args(action="list"))
    assert URL in capsys.readouterr().out

    assert deck_link.remove_link("ldeck", "moxfield") is True
    assert not (decks / "ldeck" / "links.json").exists(), "the file goes with the last link"
    assert deck_link.remove_link("ldeck", "moxfield") is False
    deck_link.main(_args(action="list"))
    assert "no links" in capsys.readouterr().out


def test_main_writes_through_the_cli_shape(decks, capsys):
    deck_link.main(_args(url="https://www.moxfield.com/decks/Q9"))
    assert deck_link.load("ldeck")["moxfield"]["id"] == "Q9"
    assert "note" not in deck_link.load("ldeck")["moxfield"], "no note -> no key"
    with pytest.raises(SystemExit, match="needs a URL"):
        deck_link.main(_args())
    deck_link.main(_args(remove=True))
    assert deck_link.load("ldeck") == {}


@pytest.mark.parametrize("bad", [
    "http://moxfield.com/decks/abc",              # not https
    "https://archidekt.com/decks/123",            # wrong host
    "https://moxfield.com.evil.io/decks/abc",     # look-alike host
    "https://moxfield.com/users/abc",             # not a deck path
    "https://moxfield.com/decks/",                # no id
    "https://moxfield.com/decks/ab$c",            # id outside [A-Za-z0-9_-]
    "https://moxfield.com/decks/abc?x=1",         # a query string
    "moxfield.com/decks/abc",                     # no scheme
])
def test_a_bad_url_is_refused_and_nothing_is_written(decks, bad):
    with pytest.raises(SystemExit, match="not a moxfield deck URL"):
        deck_link.set_link("ldeck", "moxfield", bad)
    assert not (decks / "ldeck" / "links.json").exists()


def test_an_unknown_service_is_refused(decks):
    with pytest.raises(SystemExit, match="unknown service"):
        deck_link.set_link("ldeck", "archidekt", "https://archidekt.com/decks/1")


# ── the validator: one negative per rule ───────────────────────────────────

GOOD = {"moxfield": {"url": URL, "id": "AbC_12-xyz", "as_of": "2026-10-09"}}


def _with(**kw):
    entry = dict(GOOD["moxfield"])
    for k, v in kw.items():
        if v is None:
            entry.pop(k, None)
        else:
            entry[k] = v
    return {"moxfield": entry}


@pytest.mark.parametrize("doc,needle", [
    ({}, "non-empty object"),
    ({"archidekt": GOOD["moxfield"]}, "unknown service"),
    ({"moxfield": "a string"}, "is not an object"),
    (_with(url=None), "lacks 'url'"),
    (_with(id=None), "lacks 'id'"),
    (_with(as_of=None), "lacks 'as_of'"),
    (_with(extra="x"), "unknown key 'extra'"),
    (_with(note=3), "note is not a string"),
    (_with(url="http://moxfield.com/decks/AbC_12-xyz"), "is not https"),
    (_with(url="https://evil.com/decks/AbC_12-xyz"), "is not https"),
    (_with(url="https://moxfield.com/users/AbC_12-xyz"), "is not a moxfield deck URL"),
    (_with(id="other"), "is not the id the url carries"),
    (_with(id="ab c"), r"does not match \[A-Za-z0-9_-\]\+"),
    (_with(as_of="9 Oct 2026"), "is not an ISO date"),
])
def test_validator_negatives(doc, needle):
    errors = validate_links.validate(doc)
    assert any(__import__("re").search(needle, e) for e in errors), errors


def test_validator_passes_good_and_absent(decks, capsys):
    assert validate_links.validate(GOOD) == []
    assert validate_links.validate(_with(note="x")) == []
    validate_links.main(argparse.Namespace(slug="ldeck", branch=None))
    assert "absent means absent" in capsys.readouterr().out
    (decks / "ldeck" / "links.json").write_text(json.dumps(_with(id="other")))
    with pytest.raises(SystemExit):
        validate_links.main(argparse.Namespace(slug="ldeck", branch=None))


# ── the manifest carries `links`, or no key at all ─────────────────────────

def test_manifest_links_present_and_absent(decks):
    (e,) = deck_manifest.gather_entries()
    assert "links" not in e
    doc = json.loads(deck_manifest.write_manifest([e]).read_text())
    assert "links" not in doc["decks"][0], "absent means absent: no key, not null"

    deck_link.set_link("ldeck", "moxfield", URL, today=date(2026, 10, 9))
    (e,) = deck_manifest.gather_entries()
    assert e["links"] == {"moxfield": {"url": URL, "id": "AbC_12-xyz", "as_of": "2026-10-09"}}
    doc = json.loads(deck_manifest.write_manifest([e]).read_text())
    assert doc["decks"][0]["links"]["moxfield"]["url"] == URL


# ── check-in refuses a URL ─────────────────────────────────────────────────

SENTENCE = ("Moxfield blocks server-side access; open the deck on Moxfield, Export → "
            "copy, then `manamap pilot check-in <slug> --from -` and paste")


@pytest.mark.parametrize("src", [URL, "http://moxfield.com/decks/abc",
                                 "https://archidekt.com/decks/123", "HTTPS://example.com/x"])
def test_check_in_refuses_a_url_with_the_one_sentence(src):
    with pytest.raises(SystemExit) as exc:
        check_in.read_list(src)
    assert str(exc.value) == SENTENCE


def test_check_in_still_reads_a_file(tmp_path):
    p = tmp_path / "list.txt"
    p.write_text("1 Sol Ring\n")
    assert check_in.read_list(str(p)) == "1 Sol Ring\n"
