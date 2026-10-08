"""page-state: what Sean has open, latest per tab, for Jarvis (PRD v2 Step 7)."""
import json

import pytest

from manamap import serve
from manamap.pilot import page_state as ps


@pytest.fixture
def path(tmp_path, monkeypatch):
    p = tmp_path / "page" / "state.json"
    monkeypatch.setattr(ps, "PATH", p)
    return p


def test_the_latest_snapshot_per_tab_is_kept_and_unknown_fields_are_dropped(path):
    ps.record("t1", {"page": "atlas", "mode": "build", "focus": "Windfall", "secret": "x"}, now=100)
    ps.record("t1", {"page": "atlas", "mode": "explore", "focus": "Timetwister"}, now=110)
    tabs = json.loads(path.read_text())["tabs"]
    assert list(tabs) == ["t1"]
    assert tabs["t1"]["mode"] == "explore" and tabs["t1"]["focus"] == "Timetwister"
    assert "secret" not in tabs["t1"] and tabs["t1"]["at"] == 110


def test_lists_and_text_are_capped_so_a_page_cannot_grow_a_second_store(path):
    row = ps.record("t1", {"selected": [f"Card {i}" for i in range(500)], "title": "x" * 5000},
                    now=1)
    assert len(row["selected"]) == ps.MAX_LIST and len(row["title"]) == ps.MAX_TEXT


def test_the_most_recently_focused_tab_reads_first_even_after_it_blurs(path):
    ps.record("deck", {"page": "deck", "deck": "sharknado", "focused": True}, now=100)
    ps.record("atlas", {"page": "atlas", "focused": True}, now=200)
    ps.record("atlas", {"page": "atlas", "focused": False}, now=210)   # blurred: still the last focused
    ps.record("deck", {"page": "deck", "deck": "sharknado", "focused": False}, now=220)
    tabs = ps.read(now=230)
    assert [t["tab"] for t in tabs] == ["atlas", "deck"]
    assert tabs[0]["age_s"] == 20


def test_a_tab_gone_quiet_is_dropped(path):
    ps.record("old", {"page": "deck"}, now=0)
    ps.record("new", {"page": "atlas"}, now=ps.KEEP_S + 10)
    assert [t["tab"] for t in ps.read(now=ps.KEEP_S + 20)] == ["new"]
    assert list(json.loads(path.read_text())["tabs"]) == ["new"]


def test_the_line_names_what_resolves_this_and_flags_a_stale_tab():
    t = {"page": "atlas", "mode": "build", "deck": "sharknado", "focus": "Windfall",
         "selected": ["Windfall", "Wheel of Fortune"], "filters": {"role": "draw"}, "age_s": 5}
    s = ps.line(t)
    assert "mode build · deck sharknado · focus Windfall" in s and "role=draw" in s
    assert "STALE" not in s and "STALE" in ps.line(dict(t, age_s=ps.FRESH_S + 1))


def test_cli_says_nothing_is_open_when_no_page_reported(path, capsys):
    ps.main(type("A", (), {"as_json": False}))
    assert "nothing open" in capsys.readouterr().out


def test_cli_prints_the_focused_tab_first(path, capsys):
    ps.record("a", {"page": "deck", "deck": "edgar-vampires", "focused": True})
    ps.record("b", {"page": "atlas", "mode": "discover"})
    ps.main(type("A", (), {"as_json": False}))
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("FOCUSED  deck · deck edgar-vampires")
    assert any(l.startswith("  also   atlas · mode discover") for l in out)


def test_serve_endpoint_records_and_refuses_a_bad_body(path):
    row = serve.call("page/state", {"tab": "t9", "state": {"page": "branch", "deck": "zur",
                                                          "branch": "mana-v1"}})
    assert row["branch"] == "mana-v1" and ps.read()[0]["tab"] == "t9"
    with pytest.raises(ValueError):
        serve.call("page/state", {"tab": "t9", "state": "not an object"})
    with pytest.raises(ValueError):
        serve.call("page/state", {"state": {"page": "x"}})


def test_page_state_is_post_only_and_the_cli_is_read_only():
    assert "page/state" in serve.ENDPOINTS and "page/state" not in serve.GETTABLE
    assert "page-state" in serve.CLI_READONLY


def test_the_file_lives_where_the_job_band_never_reads_it():
    """The band reads `.progress/*.json` as jobs; a subdirectory is invisible to it."""
    assert ps.PATH is None and ps._path().parent.parent == ps.progress.DIR
