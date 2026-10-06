"""`manamap pilot proxies`: a print-ready sheet for the cards still in the mail.

No network and no tracked data: branches, cards and images are stubbed, so the
whole command — which cards, which faces, the true-size layout, the PDF — is
held here in the unit tier.
"""
import io
import types

import pytest

from manamap.pilot import proxies


def _png(w=745, h=1040, colour=(200, 30, 30, 255)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGBA", (w, h), colour).save(buf, format="PNG")
    return buf.getvalue()


def _card(name, faces=None):
    url = f"https://cards.scryfall.io/normal/front/a/b/{abs(hash(name)) % 10**8}.jpg?1"
    entry = {"name": name, "image": url, "card_faces": []}
    if faces:
        entry["card_faces"] = [{"name": f, "image": f"https://cards.scryfall.io/normal/{side}/a/b/x.jpg"}
                               for f, side in zip(faces, ("front", "back"))]
    return entry


@pytest.fixture
def branch(monkeypatch):
    """A fake branch: four adds, one of them a transform card."""
    from manamap.pilot import common, deck_branch
    cards = [_card("Phyrexian Arena"), _card("Exquisite Blood"), _card("Swamp"),
             _card("Bloodline Keeper // Lord of Lineage", ["Bloodline Keeper", "Lord of Lineage"]),
             _card("Tithe Drinker")]
    monkeypatch.setattr(deck_branch, "diff", lambda slug, branch: {
        "add": ["Bloodline Keeper // Lord of Lineage", "Exquisite Blood", "Phyrexian Arena",
                "Tithe Drinker"],
        "quantity": [{"name": "Swamp", "from": 3, "to": 4}]})
    monkeypatch.setattr(common, "load_deck_cards", lambda slug, branch=None: {"cards": cards})
    return cards


def test_print_url_takes_the_png_render_of_the_same_card():
    assert proxies.print_url("https://cards.scryfall.io/normal/front/6/b/6bf.jpg?1700") == \
        "https://cards.scryfall.io/png/front/6/b/6bf.png"
    assert proxies.print_url("https://cards.scryfall.io/normal/back/1/2/x.jpg") == \
        "https://cards.scryfall.io/png/back/1/2/x.png"
    already = "https://cards.scryfall.io/png/front/6/b/6bf.png"
    assert proxies.print_url(already) == already


def test_cards_are_true_size_and_nine_to_a_page():
    assert proxies.CARD_PX == (744, 1039)          # 63 x 88 mm at 300 dpi
    for paper in ("letter", "a4"):
        (pw, ph), pages, slots = proxies.layout(19, paper)
        assert pages == 3
        assert slots[8][0] == 0 and slots[9][0] == 1 and slots[18][0] == 2
        assert all(x >= 0 and y >= 0 and x + 744 <= pw and y + 1039 <= ph for _p, x, y in slots)


def test_a_branch_prints_its_adds_minus_what_you_have(branch):
    got, report = proxies.resolve(["edgar-vampires@mardu-combo-v1"],
                                  have=["exquisite blood", "Demonic Tutr"])
    names = [p["name"] for p in got]
    assert "Exquisite Blood" not in names, "a card you hold is not printed"
    assert "Swamp" not in names, "a basic whose count moved is never printed"
    assert report["kept_back"] == ["exquisite blood"]
    assert report["unmatched_have"] == ["Demonic Tutr"], "a typo is reported, not ignored"


def test_a_double_faced_card_prints_both_faces(branch):
    got, _ = proxies.resolve(["edgar-vampires@mardu-combo-v1"])
    keeper = [p for p in got if p["card"].startswith("Bloodline Keeper")]
    assert [(p["name"], p["face"]) for p in keeper] == [
        ("Bloodline Keeper", "front"), ("Lord of Lineage", "back")]
    assert len(got) == 5       # 3 single-faced adds + 2 faces


def test_a_target_must_name_its_branch():
    with pytest.raises(SystemExit, match="slug>@<branch"):
        proxies.resolve(["edgar-vampires"])


def test_one_off_cards_come_from_the_corpus(monkeypatch):
    from manamap.pilot import try_swap
    monkeypatch.setattr(try_swap, "corpus_card", lambda name: _card(name))
    got, _ = proxies.resolve(names=["Throes of Chaos"])
    assert [(p["name"], p["source"]) for p in got] == [("Throes of Chaos", "--card")]


def test_an_image_is_fetched_once_and_then_served_from_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(proxies.config, "SCRYFALL_REQUEST_DELAY_S", 0)
    calls = []
    def get(url):
        calls.append(url)
        return _png(10, 10)
    a = proxies.fetch("https://x/png/a.png", cache_dir=tmp_path, get=get)
    b = proxies.fetch("https://x/png/a.png", cache_dir=tmp_path, get=get)
    assert a == b and len(calls) == 1


def test_the_command_writes_one_pdf_page_per_nine_cards(tmp_path, monkeypatch):
    """Ten cards -> exactly two pages, at the paper's exact size."""
    from manamap.pilot import try_swap
    monkeypatch.setattr(try_swap, "corpus_card", lambda name: _card(name))
    monkeypatch.setattr(proxies, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(proxies.config, "SCRYFALL_REQUEST_DELAY_S", 0)
    dest = tmp_path / "sheet.pdf"
    args = types.SimpleNamespace(targets=[], have=[], have_file=None,
                                 card=[f"Card {i}" for i in range(10)], paper="letter",
                                 dest=str(dest), no_open=True, dry_run=False)
    proxies.main(args, get=lambda url: _png(), opener=None)
    pdf = dest.read_bytes()
    assert pdf.startswith(b"%PDF")
    pages = pdf.count(b"/Type /Page") - pdf.count(b"/Type /Pages")
    assert pages == 2
    assert pdf.count(b"/MediaBox [ 0 0 612 792 ]") == 2, "letter is 612 x 792 pt, every page"


def test_a_dry_run_downloads_and_writes_nothing(branch, tmp_path, capsys):
    args = types.SimpleNamespace(targets=["edgar-vampires@mardu-combo-v1"], have=[],
                                 have_file=None, card=[], paper="letter",
                                 dest=str(tmp_path / "x.pdf"), no_open=True, dry_run=True)
    proxies.main(args, get=lambda url: pytest.fail("a dry run fetched"), opener=None)
    assert not (tmp_path / "x.pdf").exists()
    assert "5 to print" in capsys.readouterr().out


def test_a_have_file_takes_plain_or_counted_lines(tmp_path):
    f = tmp_path / "have.txt"
    f.write_text("1 Mana Geyser\n2x Arid Mesa\n# comment\nDemonic Tutor\n\n")
    have = proxies._read_have(types.SimpleNamespace(have=["Kirol"], have_file=str(f)))
    assert have == ["Kirol", "Mana Geyser", "Arid Mesa", "Demonic Tutor"]
