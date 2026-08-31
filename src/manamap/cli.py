"""Command-line interface: `manamap <step>` or `manamap run [--from STEP]`.

Step modules are imported lazily at dispatch time so `manamap --help`
stays fast (no torch import).
"""

import argparse
import re

from manamap import console
from manamap.pilot.registry import add_pilot_parser, run_pilot_step
from manamap.pipeline import STEP_NAMES, STEPS, run, run_step


#: ROUTE A READ-ONLY QUESTION TO A WARM PROCESS IF ONE IS LISTENING.
#:
#: Every CLI invocation is a cold start, and the memos that make this repo quick
#: are all per-process: the corpus parse, the 28MB synergy graph, the rules
#: index, and — the expensive one — the frozen MiniLM behind `query-rules`,
#: which costs ~8s to import and construct and is then thrown away. `manamap
#: serve` already holds all of it warm; this points the terminal at it.
#:
#: FAILING OPEN IS THE WHOLE DESIGN. Any error — no server, wrong version, a
#: command the server will not run — returns None and the command runs locally
#: exactly as before. The probe is a TCP connect to loopback with a 150ms cap,
#: which costs well under a millisecond when nothing is there. Set
#: MANAMAP_NO_DAEMON=1 to skip it, MANAMAP_DAEMON=host:port to point elsewhere.
def _daemon_run(argv):
    """`exit code` if a warm server answered, else None. Never raises."""
    import os

    if os.environ.get("MANAMAP_NO_DAEMON"):
        return None
    target = os.environ.get("MANAMAP_DAEMON") or "127.0.0.1:8000"
    try:
        import http.client
        import json as _json
        import sys

        host, _, port = target.partition(":")
        conn = http.client.HTTPConnection(host or "127.0.0.1",
                                          int(port or 8000), timeout=0.15)
        body = _json.dumps({"argv": list(argv)})
        conn.request("POST", "/api/cli", body,
                     {"Content-Type": "application/json"})
        # The command itself may legitimately take a while once the server has
        # accepted it; only the CONNECT needs to be impatient.
        conn.sock.settimeout(600)
        response = conn.getresponse()
        payload = _json.loads(response.read() or b"{}")
        if response.status != 200 or not payload.get("ok"):
            return None
        # `_run` wraps every handler's return value: {ok, command, result}.
        result = payload.get("result")
        if not isinstance(result, dict) or "stdout" not in result:
            return None
        sys.stdout.write(result["stdout"])
        sys.stdout.flush()
        return int(result.get("exit") or 0)
    except Exception:                                  # noqa: BLE001 - fail open
        return None
    finally:
        try:
            conn.close()
        except Exception:                              # noqa: BLE001
            pass


def pipeline_step_count():
    """The highest step NUMBER the registry declares — not `len(STEPS)`.

    The two are different and the difference is the whole reason this is
    derived: `train`/`train-ability` are steps 4a and 4b, so 16 registry
    entries are 15 numbered steps. A hand-written literal here said "13" for
    long enough that `docs/pipeline.md` agreed with it while CLAUDE.md and
    README said 15 — three documents, two answers, and the CLI itself was the
    one telling users the wrong one.
    """
    return max(int(m.group(1))
               for _, _, description in STEPS
               if (m := re.match(r"Step (\d+)", description)))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="manamap",
        description="MTG card embedding pipeline — run all steps or one at a time.",
    )
    # Global, and it must be on the ROOT parser rather than per-command: it is a
    # property of the terminal you are running in, not of the job you asked for.
    # `console.is_plain()` already infers this from `stderr.isatty()`, so the
    # flag exists for the case inference cannot reach — a terminal where you
    # want the output quiet anyway, and recording a session for a transcript.
    parser.add_argument(
        "--plain", action="store_true",
        help="no progress bars, spinners or colour — plain lines only "
             "(also: MANAMAP_PLAIN=1, NO_COLOR=1, or any non-terminal stderr)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help=f"Run the full {pipeline_step_count()}-step pipeline in order")
    run_parser.add_argument(
        "--from",
        dest="start",
        metavar="STEP",
        choices=STEP_NAMES,
        help="Start from this step instead of the beginning",
    )

    for name, _, description in STEPS:
        subparsers.add_parser(name, help=description)

    # NOT a pipeline step, deliberately. `eval-embeddings` is step 15 because it
    # scores artifacts the pipeline just built; this one needs the network and a
    # frozen EDHREC snapshot, so putting it in STEPS would make `manamap run`
    # fetch eighty decklists on its way to a projection.
    srv = subparsers.add_parser(
        "serve",
        help="Serve viz/ AND a local /api the deployed site does not have")
    srv.add_argument("--port", type=int, default=8000)

    # Also not a pipeline step, and for the same shape of reason as the two
    # below: it SCORES an artifact the pipeline built rather than building one.
    subparsers.add_parser(
        "eval-obsolescence",
        help="Score the obsolescence index against its known failure classes")

    ecs = subparsers.add_parser(
        "eval-commander-search",
        help="Spike S1: can the embedding rank commanders from a 20-card seed?")
    ecs.add_argument("--refresh", action="store_true",
                     help="re-fetch the frozen candidate pool from EDHREC "
                          "(a deliberate act — commit the result)")
    ecs.add_argument("--per-identity", type=int, default=8, dest="per_identity",
                     help="commanders per colour identity when refreshing (default 8)")
    ecs.add_argument("--limit", type=int, default=None,
                     help="cap the number of decks fetched when refreshing")

    add_pilot_parser(subparsers)

    return parser


def main():
    args = build_parser().parse_args()
    # Before dispatch, so a step that draws on import still sees the decision.
    if getattr(args, "plain", False):
        console.set_plain(True)
    if args.command == "run":
        run(start=args.start)
    elif args.command == "serve":
        from manamap import serve
        serve.main(args)
    elif args.command == "eval-obsolescence":
        from manamap.analysis import eval_obsolescence
        eval_obsolescence.main(args)
    elif args.command == "eval-commander-search":
        from manamap.analysis import eval_commander_search
        eval_commander_search.main(args)
    elif args.command == "pilot":
        import sys

        code = _daemon_run(sys.argv[2:])
        if code is not None:
            raise SystemExit(code)
        run_pilot_step(args)
    else:
        run_step(args.command)


if __name__ == "__main__":
    main()
