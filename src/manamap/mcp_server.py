"""Sven's read-only surface, as an MCP server for Claude Code.

WHY THIS EXISTS. Claude Code already reaches this repo by shelling out to
`manamap pilot <cmd>` and reading the prose it prints. That works and it is what
most of this repo's development has run on — but it means an agent parses a
rendered table to recover numbers the Python had in a dict a moment earlier, and
it pays a cold start every call. Measured today: `deck-status heliod` is 2.9 s
cold and 0.003 s through the warm daemon, byte-identical.

So this exposes the SAME tools Sven has, to the agent sitting in the terminal:
structured data instead of parsed prose, and the warm process instead of a fresh
interpreter.

NO SDK. MCP is JSON-RPC 2.0 over stdio, and the subset a tool server needs —
`initialize`, `tools/list`, `tools/call` — is the ~150 lines below. Taking a
dependency for that would sit badly beside `sim/stats.py`, which computes
Newcombe intervals and exact binomial power with no scipy on the same reasoning:
this is arithmetic and protocol, both auditable, and a dependency is a thing that
breaks on a Tuesday for reasons unrelated to Magic.

STDOUT IS THE PROTOCOL. Every diagnostic goes to stderr — the same split
`console.py` enforces for `--json`, arrived at for the same reason, and here the
consequence of getting it wrong is not a messy terminal but a client that cannot
parse the stream at all.

THE ALLOW-LIST IS NOT RESTATED. `sven.tools` reads `serve.CLI_READONLY` and
`serve._cli` enforces it; this imports both. A tool here cannot write, and it
cannot be made to by rewording a request, because the refusal is in the
dispatcher rather than in any prompt.
"""

import json
import sys
import traceback

PROTOCOL_VERSION = "2024-11-05"
SERVER = {"name": "manamap", "version": "0.1.0"}


def _log(message):
    """Diagnostics to stderr. Never stdout — see the module docstring."""
    print(f"[manamap-mcp] {message}", file=sys.stderr, flush=True)


# ── the tools ─────────────────────────────────────────────────────────────
#
# Shaped for an agent, not for a terminal: each returns JSON, so nothing has to
# recover a number from a column. `run_readonly` is the escape hatch onto the
# other eighteen commands, and its description carries their names.

def _tool_defs():
    """The MCP rendering of `sven.api` — the SAME registry Sven reads.

    Hand-maintained here until the two lists were compared and found already
    drifted. One registry, two spellings of one schema key.
    """
    from manamap.sven import api

    return api.mcp_block()


class _Session:
    """One long-lived fact cache, shared across every call in this process.

    An MCP server outlives a single question, so the second call about a deck is
    served from the first call's reads — the same reason `serve.py` keeps one
    Sven session rather than building one per request.
    """

    def __init__(self):
        self._core = None

    @property
    def core(self):
        if self._core is None:
            from manamap.sven import core
            self._core = core.Session()
        return self._core


_SESSION = _Session()


def call_tool(name, args):
    """Run one tool through `sven.api.call` — the same dispatcher Sven uses.

    A second implementation here would be a second place for a capability to
    behave differently depending on who asked, which is the class of bug this
    whole module was collapsed to remove.
    """
    # THROUGH THE SESSION, not straight to `api.call`. The session is what
    # carries the fact cache across calls — an MCP server outlives a single
    # question, so the second call about a deck should be answered out of the
    # first call's reads, exactly as it is for Sven.
    got = _SESSION.core.call(name, **(args or {}))
    if isinstance(got, dict) and set(got) == {"error"}:
        raise ValueError(got["error"])
    return got


# ── the protocol ──────────────────────────────────────────────────────────

def handle(request):
    """One JSON-RPC request -> a response dict, or None for a notification."""
    method = request.get("method")
    request_id = request.get("id")

    if method == "initialize":
        return _ok(request_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER,
        })
    if method in ("notifications/initialized", "initialized"):
        return None                       # a notification has no reply, ever
    if method == "ping":
        return _ok(request_id, {})
    if method == "tools/list":
        return _ok(request_id, {"tools": _tool_defs()})
    if method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name")
        try:
            result = call_tool(name, params.get("arguments"))
        except Exception as exc:          # noqa: BLE001
            # A REFUSAL IS A RESULT, NOT A TRANSPORT ERROR. `isError` lets the
            # calling agent read the message and correct itself — the same
            # reason Sven's loop hands argparse's text back to the model rather
            # than raising. A JSON-RPC error would look like the server broke.
            _log(f"tool {name} failed: {exc.__class__.__name__}: {exc}")
            return _ok(request_id, {
                "content": [{"type": "text",
                             "text": f"{exc.__class__.__name__}: {exc}"}],
                "isError": True})
        return _ok(request_id, {
            "content": [{"type": "text",
                         "text": json.dumps(result, indent=2, default=str)}],
            "isError": False})

    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": -32601, "message": f"unknown method {method!r}"}}


def _ok(request_id, result):
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def serve(stdin=None, stdout=None):
    """Read requests line by line; write one response per line.

    Line-delimited JSON, which is what the stdio transport specifies. A blank
    line is skipped rather than treated as EOF, and a malformed one is logged
    and skipped — one bad frame must not end a session that is otherwise fine.
    """
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            _log(f"skipped a malformed frame ({len(line)} bytes)")
            continue
        try:
            response = handle(request)
        except Exception:                 # noqa: BLE001 — a server must not die
            _log(traceback.format_exc())
            response = {"jsonrpc": "2.0", "id": request.get("id"),
                        "error": {"code": -32603, "message": "internal error"}}
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


def main(args=None):
    _log(f"ready — {len(_tool_defs())} tools, read-only")
    serve()


if __name__ == "__main__":
    main()
