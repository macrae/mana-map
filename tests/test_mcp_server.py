"""The MCP server — protocol, and the refusals that matter.

Claude Code reaches this repo today by shelling out to `manamap pilot <cmd>` and
parsing the prose it prints. That means an agent recovers numbers from a rendered
table that the Python had in a dict a moment earlier, and pays a cold start every
call — `deck-status heliod` is 2.9 s cold against 0.003 s warm, byte-identical.
This server hands over the dict instead.

Two things are asserted hard:

  - **stdout is the protocol.** Anything printed to stdout that is not a
    JSON-RPC frame breaks the client's parser outright. The same discipline
    `console.py` enforces for `--json`, with a worse failure mode.
  - **a write cannot be reached.** Not by a flag, not by a command outside the
    allow-list, and not by rewording — the refusal lives in `serve._cli`, which
    this imports rather than reimplements.
"""

import io
import json
import subprocess
import sys

import pytest

from manamap import mcp_server as mcp

from conftest import requires_deck


def _drive(*frames):
    """Run frames through `serve` and return the parsed responses."""
    out = io.StringIO()
    mcp.serve(stdin=iter(json.dumps(f) + "\n" for f in frames), stdout=out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def _call(name, **arguments):
    responses = _drive({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                        "params": {"name": name, "arguments": arguments}})
    assert len(responses) == 1
    return responses[0]["result"]


def _payload(result):
    return json.loads(result["content"][0]["text"])


# ── protocol ──────────────────────────────────────────────────────────────

def test_the_handshake_answers_with_a_protocol_version():
    got = _drive({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert got[0]["result"]["protocolVersion"] == mcp.PROTOCOL_VERSION
    assert got[0]["result"]["serverInfo"]["name"] == "manamap"
    assert "tools" in got[0]["result"]["capabilities"]


def test_a_notification_gets_no_reply_at_all():
    """A response to a notification is a protocol violation, and the client will
    be waiting for a frame that answers a request it never made."""
    assert _drive({"jsonrpc": "2.0", "method": "notifications/initialized"}) == []


def test_a_malformed_frame_does_not_end_the_session():
    """One bad line must not take down a session that is otherwise fine."""
    out = io.StringIO()
    mcp.serve(stdin=iter(["not json\n", "\n",
                          json.dumps({"jsonrpc": "2.0", "id": 1,
                                      "method": "ping"}) + "\n"]),
              stdout=out)
    responses = [json.loads(line) for line in out.getvalue().splitlines()]
    assert len(responses) == 1 and responses[0]["id"] == 1


def test_an_unknown_method_is_an_error_not_a_crash():
    got = _drive({"jsonrpc": "2.0", "id": 1, "method": "resources/list"})
    assert got[0]["error"]["code"] == -32601


def test_every_advertised_tool_has_a_schema_and_a_description():
    for tool in mcp._tool_defs():
        assert tool["description"].strip(), f"{tool['name']} has no description"
        assert tool["inputSchema"]["type"] == "object"
        for required in tool["inputSchema"].get("required", []):
            assert required in tool["inputSchema"]["properties"], (
                f"{tool['name']} requires {required!r} but does not declare it")


# ── stdout is the protocol ────────────────────────────────────────────────

def test_nothing_but_json_frames_reach_stdout():
    """THE FAILURE THIS PREVENTS IS TOTAL, not cosmetic: one stray line and the
    client cannot parse the stream at all.

    Driven through a real subprocess, because that is the only way to see what
    the pipe receives — and because several commands underneath print to stdout
    themselves, which is exactly the leak being guarded.
    """
    frames = "\n".join(json.dumps(f) for f in [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "run_command",
                    "arguments": {"command": "deck-status", "args": ["heliod"]}}},
    ])
    proc = subprocess.run([sys.executable, "-m", "manamap.mcp_server"],
                          input=frames + "\n", capture_output=True, text=True)
    assert proc.stdout.strip(), "no frames at all"
    for line in proc.stdout.splitlines():
        json.loads(line)          # raises if anything non-JSON leaked through


def test_diagnostics_go_to_stderr():
    proc = subprocess.run([sys.executable, "-m", "manamap.mcp_server"],
                          input="", capture_output=True, text=True)
    assert "manamap-mcp" in proc.stderr
    assert "manamap-mcp" not in proc.stdout


# ── the refusals ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("command,args", [
    ("deck-info", ["heliod", "--write"]),
    ("goldfish", ["heliod"]),
    ("deck-delete", ["heliod"]),
    ("promote", ["heliod", "--to", "sleeved"]),
])
def test_no_write_is_reachable(command, args):
    """The gate is `serve._cli`, imported rather than reimplemented. A second
    allow-list would disagree with the first within a month."""
    result = _call("run_command", command=command, args=args)
    assert result["isError"] is True, f"{command} {args} was not refused"


def test_a_refusal_is_a_result_not_a_transport_error():
    """`isError` lets the calling agent read the message and correct itself. A
    JSON-RPC error would look like the server broke, and the agent would retry
    the transport rather than fix the call."""
    result = _call("run_command", command="deck-info", args=["heliod", "--write"])
    assert result["isError"] is True
    assert "--write" in result["content"][0]["text"]


def test_an_unknown_tool_is_reported_rather_than_raised():
    result = _call("delete_everything")
    assert result["isError"] is True
    assert "unknown tool" in result["content"][0]["text"]


def test_the_tool_list_offers_no_way_to_write():
    """Read the advertised surface the way an agent would, and check there is
    nothing on it that mutates."""
    names = {t["name"] for t in mcp._tool_defs()}
    assert names == {"deck_state", "fleet", "search_docs", "search_code",
                     "stats", "run_command", "command_help"}
    enum = next(t for t in mcp._tool_defs()
                if t["name"] == "run_command")["inputSchema"]["properties"]["command"]["enum"]
    from manamap.serve import CLI_READONLY
    assert set(enum) == set(CLI_READONLY), "the enum drifted from the allow-list"


# ── the payloads are data, not prose ──────────────────────────────────────

@requires_deck
def test_deck_state_returns_structure_rather_than_a_rendered_table():
    """The point of the whole server: an agent should not parse a column to
    recover a number the Python had in a dict."""
    payload = _payload(_call("deck_state", slug="zur-enchantress"))
    assert payload["slug"] == "zur-enchantress"
    assert payload["stage"] in ("dev", "bench", "sleeved")
    assert isinstance(payload["gates_met"], int)
    assert isinstance(payload["blockers"], list)


def test_a_statistic_is_the_librarys_own_value():
    from manamap.sim import stats

    payload = _payload(_call("stats", fn="wilson", args={"k": 27, "n": 120}))
    assert tuple(payload["result"]) == stats.wilson_bounds(27, 120)


@requires_deck
def test_the_second_call_about_a_deck_is_served_from_the_first_ones_reads():
    """One session outlives a single question, which is the reason the server is
    long-lived at all."""
    _call("run_command", command="deck-status", args=["heliod"])
    result = _call("run_command", command="deck-status", args=["heliod"])
    assert _payload(result)["cached"] is True


# ── one registry, two transports ──────────────────────────────────────────

def test_both_transports_render_the_same_capabilities():
    """THEY DID NOT, and nothing noticed. The same tool was `run_command` for
    Sven and `run_readonly` for MCP, `stats` and `stat_test`, and two tools
    existed on one side only — so a question Claude Code could answer, Sven
    could not.

    `escalate` is the one deliberate asymmetry: there is nothing above Claude
    Code to escalate to.
    """
    from manamap.sven import api, tools

    sven = {t["name"] for t in tools.tool_block()}
    mcp_names = {t["name"] for t in mcp._tool_defs()}
    assert sven - mcp_names == {"escalate"}
    assert mcp_names - sven == set()


def test_the_two_renderings_differ_only_in_the_schema_key():
    """One is `input_schema`, the other `inputSchema`. Everything else about a
    capability must be identical, or it behaves differently depending on who
    asked."""
    from manamap.sven import api

    a = {t["name"]: t for t in api.anthropic_block()}
    m = {t["name"]: t for t in api.mcp_block()}
    for name in set(a) & set(m):
        assert a[name]["description"] == m[name]["description"]
        assert a[name]["input_schema"] == m[name]["inputSchema"]


def test_one_dispatcher_serves_both():
    """A second implementation is a second place for a capability to drift."""
    import inspect

    from manamap.sven import api

    assert "api.call" in inspect.getsource(mcp.call_tool)
