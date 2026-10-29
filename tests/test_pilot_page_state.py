"""page-state: what Sean has open, latest per tab, for Jarvis (PRD v2 Step 7)."""
import json
import time

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


def test_builds_panel_tab_is_kept_and_named_in_the_line(path):
    """Build's side panel is tabs (2026-10-28); the page reports which as `panel`.
    Without it in FIELDS it would be dropped like any unknown field."""
    row = ps.record("t1", {"page": "atlas", "mode": "build", "deck": "heliod", "view": "graph",
                           "panel": "change"}, now=1)
    assert row["panel"] == "change"
    assert "view graph · tab change" in ps.line(dict(row, age_s=1))


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


# ── Send to Jarvis: one pending ask on the record (2026-10-28) ──────────────


def test_an_ask_is_kept_stamped_and_carried_over_by_the_next_snapshot(path):
    """The page posts its snapshot WITH an ask once; every later snapshot comes without
    one, and must not erase a question nobody has answered."""
    row = ps.record("t1", {"page": "atlas", "deck": "sharknado",
                           "ask": {"text": "  Should I swap  Opt for Windfall in Sharknado? ",
                                   "page": "atlas", "deck": "sharknado",
                                   "cards": ["Opt", "Windfall"], "at": 1, "secret": "x"}},
                    now=100)
    assert row["ask"] == {"text": "Should I swap Opt for Windfall in Sharknado?", "page": "atlas",
                          "deck": "sharknado", "cards": ["Opt", "Windfall"], "at": 100}
    ps.record("t1", {"page": "atlas", "deck": "sharknado", "focus": "Opt"}, now=110)
    tabs = json.loads(path.read_text())["tabs"]
    assert tabs["t1"]["focus"] == "Opt" and tabs["t1"]["ask"]["at"] == 100


def test_there_is_one_pending_ask_and_a_newer_one_from_any_tab_replaces_it(path):
    ps.record("a", {"page": "atlas", "ask": {"text": "first"}}, now=100)
    ps.record("b", {"page": "branch", "ask": {"text": "second", "branch": "sharks-v1"}}, now=110)
    ps.record("a", {"page": "atlas"}, now=120)       # a's old ask must not come back
    tabs = json.loads(path.read_text())["tabs"]
    assert "ask" not in tabs["a"] and tabs["b"]["ask"]["text"] == "second"
    a = ps.pending_ask(ps.read(now=120))
    assert a["text"] == "second" and a["tab"] == "b" and a["branch"] == "sharks-v1"


def test_an_ask_without_a_question_is_no_ask_and_its_fields_are_capped(path):
    row = ps.record("t1", {"page": "atlas", "ask": {"text": "   ", "cards": ["Opt"]}}, now=1)
    assert "ask" not in row
    row = ps.record("t1", {"ask": {"text": "x" * 5000, "cards": [f"C{i}" for i in range(99)]}}, now=2)
    assert len(row["ask"]["text"]) == ps.MAX_ASK and len(row["ask"]["cards"]) == ps.MAX_LIST
    assert "ask" in ps.FIELDS


def test_the_cli_prints_the_pending_ask_first_on_its_own_line(path, capsys):
    ps.record("a", {"page": "atlas", "mode": "build", "deck": "sharknado", "focused": True,
                    "ask": {"text": "What does Windfall do for Sharknado?", "page": "atlas",
                            "deck": "sharknado", "cards": ["Windfall"]}})
    ps.main(type("A", (), {"as_json": False, "clear_ask": False}))
    out = capsys.readouterr().out.splitlines()
    assert out[0] == 'ASK      "What does Windfall do for Sharknado?"', out
    assert "deck sharknado" in out[1] and "cards Windfall" in out[1] and "--clear-ask" in out[1]
    assert out[2].startswith("FOCUSED  atlas · mode build · deck sharknado"), out


def test_clear_ask_marks_it_taken_and_leaves_the_tab(path, capsys):
    ps.record("a", {"page": "atlas", "deck": "sharknado", "ask": {"text": "Review branch x"}})
    ps.main(type("A", (), {"as_json": False, "clear_ask": True}))
    assert capsys.readouterr().out.strip() == 'cleared the pending ask: "Review branch x"'
    tabs = json.loads(path.read_text())["tabs"]
    assert "ask" not in tabs["a"] and tabs["a"]["deck"] == "sharknado"
    assert ps.pending_ask(ps.read()) is None
    ps.main(type("A", (), {"as_json": False, "clear_ask": True}))
    assert capsys.readouterr().out.strip() == "no pending ask"
    # Cleared means cleared: the page's next ordinary snapshot does not bring it back.
    ps.record("a", {"page": "atlas", "deck": "sharknado"})
    assert ps.pending_ask(ps.read()) is None


def test_the_json_carries_the_ask_and_a_stale_one_says_so(path, capsys):
    ps.record("a", {"page": "atlas", "ask": {"text": "old question"}}, now=time.time() - ps.FRESH_S - 60)
    ps.main(type("A", (), {"as_json": True, "clear_ask": False}))
    doc = json.loads(capsys.readouterr().out)
    assert doc["ask"]["text"] == "old question"
    assert "STALE" in ps.ask_line(doc["ask"])


def test_the_clear_flag_parses_and_is_refused_over_the_api(path):
    """`--clear-ask` writes, so the warm daemon must refuse it (the CLI then runs it
    locally) and MCP's run_command can never reach it."""
    import argparse

    from manamap.pilot.registry import add_pilot_parser

    p = argparse.ArgumentParser(prog="manamap")
    add_pilot_parser(p.add_subparsers(dest="command"))
    assert p.parse_args(["pilot", "page-state", "--clear-ask"]).clear_ask is True
    ps.record("a", {"page": "atlas", "ask": {"text": "keep me"}})
    with pytest.raises(ValueError, match="clear_ask"):
        serve._cli(["page-state", "--clear-ask"])
    assert ps.pending_ask(ps.read())["text"] == "keep me"
    assert serve._cli(["page-state"])["exit"] == 0
