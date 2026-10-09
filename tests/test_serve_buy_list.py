"""`branch/buy-list`: the bill's BUY rows as a paste, over the local bridge."""

import pytest

from manamap import serve


def test_the_buy_list_endpoint_is_the_cli_function_read_only_and_gettable(tmp_path, monkeypatch):
    """`branch/buy-list` returns what `manamap pilot buy-list --json` prints,
    from the SAME function — a second rendering here is how the page's paste and
    the terminal's would come to differ by a set code. `exact` is coerced like
    every other flag, the branch must exist, and the endpoint is GET-safe."""
    from manamap.pilot import buy_list
    seen = []

    def fake(slug, branch, exact=False):
        seen.append({"slug": slug, "branch": branch, "exact": exact})
        return {"text": "1 Sol Ring", "count": 1, "buy_cents": None, "as_of": None}

    monkeypatch.setattr(buy_list, "payload", fake)
    (tmp_path / "decklist.txt").write_text("Deck:\n1 Sol Ring\n")
    monkeypatch.setattr("manamap.pilot.common.deck_dir", lambda slug, branch=None: tmp_path)

    assert serve.call("branch/buy-list", {"slug": "td", "branch": "b1", "exact": "true"}) == {
        "text": "1 Sol Ring", "count": 1, "buy_cents": None, "as_of": None}
    serve.call("branch/buy-list", {"slug": "td", "branch": "b1", "exact": "false"})
    serve.call("branch/buy-list", {"slug": "td", "branch": "b1"})
    assert [s["exact"] for s in seen] == [True, False, False]
    assert seen[0]["slug"] == "td" and seen[0]["branch"] == "b1"
    with pytest.raises(ValueError, match="branch"):
        serve.call("branch/buy-list", {"slug": "td"})
    assert "branch/buy-list" in serve.GETTABLE
    assert "buy-list" in serve.CLI_READONLY


def test_a_missing_branch_is_refused_before_anything_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr("manamap.pilot.common.deck_dir", lambda slug, branch=None: tmp_path)
    with pytest.raises(ValueError, match="no branch"):
        serve.call("branch/buy-list", {"slug": "td", "branch": "nope"})
