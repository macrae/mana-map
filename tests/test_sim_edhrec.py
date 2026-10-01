"""EDHREC's commander page as a dated artifact: the parser flattens EVERY cardlist tag,
refuses a page with none, merges themes side by side, and the validator holds the file
to reality (names in the corpus, synergy in range, URLs on EDHREC).

Pinned on a trimmed fixture of the live aristocrats page (three lists, three cards each,
2026-09-30) so no test touches the network.
"""
import json
from pathlib import Path

import pytest

from manamap.sim import edhrec

FIX = Path(__file__).parent / "fixtures" / "edhrec" / "edgar-markov-aristocrats.json"
URL = "https://json.edhrec.com/pages/commanders/edgar-markov/aristocrats.json"


@pytest.fixture(scope="module")
def page():
    return edhrec.flatten(json.loads(FIX.read_text()), URL)


def test_the_parser_flattens_every_cardlist_tag_not_just_the_first(page):
    """The bug this guards: reading only the first panel's cardviews, which is the
    'New Cards' list on the live page and names nothing a deck wants."""
    tags = {r["tag"] for r in page["rows"]}
    assert tags == {"highsynergycards", "topcards", "creatures"}
    assert len(page["rows"]) == 9
    assert page["commander"] == "Edgar Markov" and page["num_decks"] == 1500
    bw = next(r for r in page["rows"] if r["name"] == "Malakir Bloodwitch")
    assert bw["tag"] == "creatures" and isinstance(bw["num_decks"], int) and -1 <= bw["synergy"] <= 1


def test_a_page_without_cardlists_is_refused_not_read_as_empty():
    with pytest.raises(ValueError):
        edhrec.flatten({"container": {"json_dict": {"card": {"name": "X"}}}}, URL)
    with pytest.raises(ValueError):
        edhrec.flatten({}, URL)


def test_merge_keeps_every_theme_side_by_side_and_the_base_page_leads(page):
    base = dict(page, url=edhrec.url_for("edgar-markov"),
                rows=[dict(r, synergy=0.5, tag="topcards") for r in page["rows"][:2]])
    doc = edhrec.merge("Edgar Markov", "edgar-vampires", {edhrec.BASE: base, "aristocrats": page})
    assert set(doc["themes"]) == {"base", "aristocrats"} and doc["as_of"]
    art = doc["cards"]["Blood Artist"]
    assert set(art["by_theme"]) == {"base", "aristocrats"} and art["from"] == "base" and art["synergy"] == 0.5
    only_theme = doc["cards"]["Malakir Bloodwitch"]
    assert set(only_theme["by_theme"]) == {"aristocrats"} and only_theme["from"] == "aristocrats"
    assert "creatures@aristocrats" in only_theme["lists"]


def test_the_slug_and_url_forms():
    assert edhrec.edhrec_slug("Edgar Markov") == "edgar-markov"
    assert edhrec.edhrec_slug("Vish Kal, Blood Arbiter") == "vish-kal-blood-arbiter"
    assert edhrec.url_for("edgar-markov", "aristocrats") == URL
    assert edhrec.url_for("edgar-markov").endswith("/commanders/edgar-markov.json")


def test_the_validator_fails_on_an_unknown_name_a_bad_url_and_a_synergy_out_of_range(page, tmp_path, monkeypatch):
    from manamap import config
    from manamap.pilot import validate_edhrec_cards as v
    if not config.OUTPUT_CSV_PATH.exists():
        pytest.skip("requires the card corpus")
    doc = edhrec.merge("Edgar Markov", "d", {"aristocrats": page})
    assert v.validate("d", doc) == []
    bad = json.loads(json.dumps(doc))
    bad["cards"]["Not A Card"] = {"synergy": 0.1, "num_decks": 1, "by_theme": {"aristocrats": {}}}
    bad["cards"]["Blood Artist"]["synergy"] = 1.5
    bad["themes"]["aristocrats"]["url"] = "https://example.com/x.json"
    errs = v.validate("d", bad)
    assert any("not in the corpus" in e for e in errs) and any("outside [-1, 1]" in e for e in errs) \
        and any("not on https://json.edhrec.com/" in e for e in errs)


def test_cards_newer_than_the_corpus_are_listed_apart_not_failed(page):
    """EDHREC lists a card the day it is previewed; the corpus is a dated dump. Three of
    Edgar's rows were newer than it on 2026-09-30, and the validator must not fire on
    correct data — they are partitioned into `not_in_corpus` and the rest validate."""
    doc = edhrec.merge("Edgar Markov", "d", {"aristocrats": page})
    doc["cards"]["Brand New Preview"] = {"synergy": 0.2, "num_decks": 3, "by_theme": {"aristocrats": {}}}
    known = {r["name"] for r in page["rows"]}
    out = edhrec.partition(doc, known)
    assert out["not_in_corpus"] == ["Brand New Preview"] and "Brand New Preview" not in out["cards"]
    assert set(out["cards"]) == known
    assert edhrec.partition({"cards": {"X": {}}}, set()) == {"cards": {"X": {}}}, "no corpus: nothing partitioned"
