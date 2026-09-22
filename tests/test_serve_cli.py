"""The warm-process CLI: `/api/cli`, and the terminal client that uses it.

Every `manamap pilot …` invocation is a cold process, and the memos that make
this repo quick are all module-level dicts — the corpus parse, the 28MB synergy
graph, the rules index, and the frozen MiniLM behind `query-rules`, which costs
about eight seconds to import and construct and is then thrown away. `manamap
serve` already holds every one of them warm.

MEASURED on this machine, cold against warm, output byte-identical in each case:

    query-rules      6.93s -> 0.16s   (43x)
    query-strategy   6.87s -> 0.16s   (43x)
    deck-facts       1.44s -> 0.14s
    card-search      0.93s -> 0.20s
    deck-audit       2.26s -> 0.59s
    deck-status      3.02s -> 0.88s
    deck-info        3.15s -> 1.25s

These tests are about the SEAM, not the speed: what the server will run, what it
refuses, and that the client falls open to the local path whenever anything at
all goes wrong.

They live apart from `test_serve.py` deliberately. Several tests there point
DECKS_DIR at a tmp directory, and a resolved-path memo survives the monkeypatch
teardown, so anything reading a real deck afterwards sees the tmp one. That is a
latent defect in the memo rather than in this endpoint; keeping these cases in
their own module makes them deterministic instead of order-dependent.
"""

import pytest

from manamap import serve

from conftest import requires_data

@requires_data
def test_cli_runs_a_read_only_command_and_returns_its_stdout():
    """The point of the endpoint: a warm process answers what a cold one
    re-derives. Measured on this machine, `query-rules` 6.93s cold against
    0.16s warm — ~43x, and almost all of it is the MiniLM that the previous
    invocation had already built and thrown away."""
    out = serve._cli(["card-search", "--deck", "ur-dragon", "--oracle", "flying"])
    assert out["exit"] == 0
    assert "CARD SEARCH" in out["stdout"]


def test_cli_refuses_a_command_that_is_not_on_the_allow_list():
    """A server that can run any subcommand is a server that can be talked into
    writing something. Same argument as `ENDPOINTS` itself."""
    for argv in (["goldfish", "ur-dragon"],          # writes goldfish_metrics
                 ["fetch-deck", "ur-dragon"],        # writes cards.json
                 ["mana-analysis", "ur-dragon"],     # writes mana_analysis
                 ["deck-branch", "ur-dragon", "merge", "x"]):
        with pytest.raises(ValueError, match="read-only"):
            serve._cli(argv)


def test_cli_refuses_a_write_flag_on_a_command_it_would_otherwise_run():
    """THE SECOND GATE. `deck-info` IS on the allow-list and `deck-info --write`
    must still be refused — a read-only command can grow a writing flag later
    and nobody will remember this file. Re-introducing the bug: drop
    `_CLI_WRITE_ATTRS` and this is the test that goes red."""
    assert "deck-info" in serve.CLI_READONLY
    assert serve._cli(["deck-info", "ur-dragon"])["exit"] == 0
    with pytest.raises(ValueError, match="write"):
        serve._cli(["deck-info", "ur-dragon", "--write"])


def test_every_allow_listed_name_is_a_real_pilot_command():
    """A typo here fails open — the command would simply never be served, and
    nothing would say so."""
    from manamap.pilot.registry import PILOT_STEPS

    known = {name for name, _module, _desc in PILOT_STEPS}
    unknown = sorted(serve.CLI_READONLY - known)
    assert not unknown, f"not pilot commands: {unknown}"
    assert len(serve.CLI_READONLY) >= 10


def test_cli_accepts_the_pilot_prefix_because_argv_carries_it():
    """The client forwards `sys.argv[2:]`, but a caller pasting a full command
    line is the obvious mistake and costs nothing to absorb."""
    a = serve._cli(["deck-facts", "ur-dragon"])
    b = serve._cli(["pilot", "deck-facts", "ur-dragon"])
    assert a["stdout"] == b["stdout"]


def test_the_client_fails_open_when_no_server_is_listening(monkeypatch):
    """FAILING OPEN IS THE WHOLE DESIGN. A dead port, a wrong port, a server
    that refuses the command — every one of them returns None so the command
    runs locally exactly as it did before.

    THE PORT IS SET, NOT ASSUMED. This said `# nothing on :1` for a fortnight
    and never set `MANAMAP_DAEMON`, so it dialled the default 127.0.0.1:8000
    and got a real answer from the pilot's own `manamap serve` — which
    `CLAUDE.md` tells you to keep running, because it is what makes every
    read-only command warm. So the test failed for anyone following the
    project's own workflow and passed in CI, where nothing is listening. A
    fail-open test that depends on nothing listening is not testing fail-open.
    """
    from manamap import cli

    monkeypatch.setenv("MANAMAP_DAEMON", "127.0.0.1:1")
    assert cli._daemon_run(["deck-facts", "ur-dragon"]) is None


def test_the_client_fails_open_on_a_stranger_answering_the_right_port(monkeypatch):
    """THE CONTROL THE OTHER TEST CANNOT BE. A closed socket proves only that
    a connection refusal is handled; the dangerous case is something that
    ANSWERS. `python -m http.server 8000` from the repo root is in CLAUDE.md
    one line below `manamap serve`, and a pilot who starts the static server
    instead has a live listener on the daemon's default port that knows
    nothing about `/api/cli`.

    It replies 501 to a POST, so `cli._daemon_run`'s status check covers it —
    but nothing pinned that, and the whole fail-open contract rests on it.
    """
    import http.server
    import threading

    from manamap import cli

    class Static(http.server.BaseHTTPRequestHandler):
        def do_POST(self):                       # noqa: N802 — stdlib's name
            self.send_error(501, "Unsupported method ('POST')")

        def log_message(self, *args):
            pass                                 # keep the suite's output clean

    server = http.server.HTTPServer(("127.0.0.1", 0), Static)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        host, port = server.server_address[0], server.server_address[1]
        monkeypatch.setenv("MANAMAP_DAEMON", f"{host}:{port}")
        assert cli._daemon_run(["deck-facts", "ur-dragon"]) is None
    finally:
        server.shutdown()
        server.server_close()


def test_the_client_can_be_switched_off(monkeypatch):
    from manamap import cli

    monkeypatch.setenv("MANAMAP_NO_DAEMON", "1")
    assert cli._daemon_run(["deck-facts", "ur-dragon"]) is None


def _positional_choices(command):
    """Every positional-with-choices on one pilot subcommand, from the parser."""
    import argparse

    from manamap.pilot.registry import add_pilot_parser

    parser = argparse.ArgumentParser(prog="manamap")
    add_pilot_parser(parser.add_subparsers(dest="command"))
    top = [a for a in parser._actions
           if isinstance(a, argparse._SubParsersAction)][0]
    pilot = [a for a in top.choices["pilot"]._actions
             if isinstance(a, argparse._SubParsersAction)][0]
    cmd = pilot.choices[command]
    return {a.dest: list(a.choices) for a in cmd._actions
            if not a.option_strings and a.choices}


def test_a_positional_write_cannot_reach_the_api():
    """THE GATE READ FLAGS AND THE WRITE WAS A WORD.

    `_CLI_WRITE_ATTRS` refuses `--write`, `--force`, `--apply`, `--record` and
    `--anyway`. It has nothing to say about a command whose verb is the
    positional after the slug — and on 2026-09-12 `deck-version ur-dragon paper`
    was POSTed to `/api/cli`, ran `set_paper`, and rewrote the tracked
    `deck_versions.json`, wiping the note on the lock as it went. `CLAUDE.md`
    said of this server, at the time: "It CANNOT write."

    Four of `deck-version`'s six actions write. This asserts the three that do
    are refused and the two that do not are served.
    """
    for action in ("paper", "tag", "restore", "baseline"):
        with pytest.raises(ValueError) as caught:
            serve._cli(["deck-version", "ur-dragon", action])
        assert "writes" in str(caught.value), action

    assert serve._cli(["deck-version", "ur-dragon", "list"])["exit"] == 0
    # `show` is PERMITTED but needs a `ref`, and the gate is about permission,
    # not about whether the command then succeeds. Asserting only that the
    # refusal is not the gate's: `show` without a ref raises its own.
    try:
        serve._cli(["deck-version", "ur-dragon", "show"])
    except ValueError as exc:
        assert "writes" not in str(exc)


def test_every_gated_command_with_a_positional_verb_is_covered():
    """A NEW ACTION MUST FAIL HERE RATHER THAN OPEN A HOLE QUIETLY.

    `CLI_READONLY_ACTIONS` is an allowlist, so a new *write* action on
    `deck-version` is refused by construction. The danger is the other
    direction: a command joining `CLI_READONLY` that has a positional verb
    nobody classified. This pins the full action list of every gated command —
    so adding one, or adding a command that has one, is a decision somebody
    takes rather than an oversight that ships.
    """
    checked = 0
    for command in sorted(serve.CLI_READONLY):
        for dest, choices in _positional_choices(command).items():
            checked += 1
            assert command in serve.CLI_READONLY_ACTIONS, (
                f"`{command}` is read-only over the API and takes a positional "
                f"{dest!r} out of {choices} — classify its actions in "
                f"serve.CLI_READONLY_ACTIONS, or drop it from CLI_READONLY")
            unknown = set(choices) - set(serve.CLI_READONLY_ACTIONS[command])
            # Everything not on the allowlist is refused, which is the safe
            # direction — this only asserts the allowlist names real actions.
            assert set(serve.CLI_READONLY_ACTIONS[command]) - {None} <= set(choices), (
                f"{command}: CLI_READONLY_ACTIONS names actions the parser does "
                f"not have: {set(serve.CLI_READONLY_ACTIONS[command]) - {None} - set(choices)}")
            assert unknown, (
                f"{command}: every action is allowed, so the gate does nothing — "
                f"either it has no writers (drop the entry) or one is missing")
    assert checked >= 1, "no gated command has a positional verb — has the gate moved?"
