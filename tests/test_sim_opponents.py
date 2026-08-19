"""`fetch-opponent`: a pod seat under data/opponents/ from EDHREC's average deck.

Offline: the slug EDHREC keys on, the decklist text in the repo's own format (commander
marked, basics as one quantity line), and the on-disk shape with its provenance file.
The network call is a one-liner over urllib and is not mocked here."""

import json

from manamap.sim import opponents
from manamap.sim.forge import seat_dir, to_dck


def test_edhrec_slug_matches_the_site():
    assert opponents.edhrec_slug("Giada, Font of Hope") == "giada-font-of-hope"
    assert opponents.edhrec_slug("Vito, Thorn of the Dusk Rose") == "vito-thorn-of-the-dusk-rose"
    assert opponents.edhrec_slug("Baylen, the Haymaker") == "baylen-the-haymaker"


def test_decklist_text_is_the_repo_format(tmp_path, monkeypatch):
    avg = {"url": "u", "slug": "x", "commanders": ["Giada, Font of Hope"],
           "cards": [("Sol Ring", 1), ("Plains", 30), ("Lyra Dawnbringer", 1)]}
    text = opponents.decklist_text(avg)
    assert text.splitlines()[0] == "1 Giada, Font of Hope *CMDR*"
    assert "30 Plains" in text and text.strip().endswith("30 Plains"), "basics last, one line"
    monkeypatch.setattr(opponents, "OPPONENTS_DIR", tmp_path / "opponents")
    monkeypatch.setattr("manamap.sim.forge.DECKS_DIR", tmp_path / "decks")
    base, total = opponents.write_opponent("giada-angels", avg, note="the Orinda pod")
    assert total == 33 and (base / "decklist.txt").exists()
    src = json.loads((base / "source.json").read_text())
    assert src["source"] == "edhrec average deck" and src["note"] == "the Orinda pod"
    # the harness resolves it as a seat and converts it
    assert seat_dir("giada-angels") == base
    dck = to_dck("giada-angels")
    assert "[Commander]\n1 Giada, Font of Hope\n[Main]" in dck and "30 Plains" in dck
