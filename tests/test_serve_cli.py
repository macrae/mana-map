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


def test_the_client_fails_open_when_no_server_is_listening():
    """FAILING OPEN IS THE WHOLE DESIGN. A dead port, a wrong port, a server
    that refuses the command — every one of them returns None so the command
    runs locally exactly as it did before."""
    from manamap import cli

    assert cli._daemon_run(["deck-facts", "ur-dragon"]) is None  # nothing on :1


def test_the_client_can_be_switched_off(monkeypatch):
    from manamap import cli

    monkeypatch.setenv("MANAMAP_NO_DAEMON", "1")
    assert cli._daemon_run(["deck-facts", "ur-dragon"]) is None
