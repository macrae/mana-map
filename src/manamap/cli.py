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

    from manamap import spaces as _spaces
    from manamap.pipeline import SPACE_AWARE

    for name, _, description in STEPS:
        step = subparsers.add_parser(name, help=description)
        if name in SPACE_AWARE:
            step.add_argument(
                "--space", default=None, choices=_spaces.choices(),
                help="build this embedding space's artifact instead of the "
                     f"pipeline's usual set (default: {_spaces.DEFAULT})")

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

    # NOT a pipeline step either, and the reason is money rather than network:
    # this trains a model for 18-40 minutes and writes a SHADOW artifact that
    # nothing downstream reads yet. `manamap run` must not pay that on its way
    # to a projection, and it must not silently replace the live space.
    vae = subparsers.add_parser(
        "train-vae",
        help="Train the masked-imputation VAE (shadow artifact; not part of `run`)")
    vae.add_argument("--unfreeze", type=int, default=0, metavar="N",
                     help="thaw the top N encoder layers. DEFAULT 0: the corpus "
                          "is 2.2M tokens against 0.94M trainable parameters "
                          "frozen and 11.58M at N=6, with 2,705 duplicate "
                          "oracle texts waiting to be memorised")
    vae.add_argument("--epochs", type=int, default=None,
                     help="override the epoch ceiling (default 20, early stop 4)")

    vc = subparsers.add_parser(
        "vae-cache",
        help="Cache the FROZEN encoder's output so a VAE sweep costs minutes")
    vc.add_argument("--force", action="store_true", help="rebuild an existing cache")

    sc = subparsers.add_parser(
        "span-cache",
        help="Cache a frozen sentence vector per DISTINCT card text span")
    sc.add_argument("--force", action="store_true", help="rebuild an existing cache")

    ps = subparsers.add_parser(
        "project-spaces",
        help="Project every embedding space to 2D/3D side by side, to LOOK at them")
    ps.add_argument("--components", type=int, default=2, help="2 or 3")
    ps.add_argument("--sample", type=int, default=None)

    rc = subparsers.add_parser(
        "recoverability",
        help="Which fields does a linear probe already solve? Gates the loss weights")

    cb = subparsers.add_parser(
        "train-cardbert",
        help="Masked field imputation over the card schema (BERT over FIELDS)")
    cb.add_argument("--epochs", type=int, default=None)
    cb.add_argument("--d-model", dest="d_model", type=int, default=None)
    cb.add_argument("--layers", type=int, default=None)
    cb.add_argument("--view-weight", dest="view_weight", type=float, default=None,
                    help="weight on the card-level contrastive term (default 1.0)")
    cb.add_argument("--objective", choices=("infonce", "vicreg"), default=None,
                    help="the card-level view term: contrastive, or VICReg (no negatives)")
    cb.add_argument("--tag", default=None,
                    help="suffix the artifacts, so a sweep keeps its runs apart")
    cb.add_argument("--embed-only", dest="embed_only", action="store_true",
                    help="regenerate embeddings from the saved checkpoint, no training")

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

    ask_parser = subparsers.add_parser(
        "ask", help="Ask Sven — the one front door over the 102 commands")
    ask_parser.add_argument("question", nargs="*", help="what you want to know")
    ask_parser.add_argument(
        "--no-cache", action="store_true",
        help="re-ask even if the same question was answered against this exact "
             "state of the repo")
    ask_parser.add_argument(
        "--deep", action="store_true",
        help="start on the stronger model instead of escalating into it")
    ask_parser.add_argument(
        "--json", dest="as_json", action="store_true",
        help="emit the raw event stream on stdout, nothing on stderr")
    ask_parser.add_argument(
        "--spend", action="store_true",
        help="what Sven has cost so far (a local ESTIMATE — the console is the "
             "authority) and the ceiling in force")

    add_pilot_parser(subparsers)

    return parser


def _sven_frames(question, deep=False, no_cache=False):
    """Frames from a WARM Sven, or None if there is no server. Never raises.

    `_daemon_run`'s twin, and the same doctrine: FAILING OPEN IS THE WHOLE
    DESIGN. No server, a refusal, a malformed frame — any of them returns None
    and the caller runs the identical loop in-process. The only difference is
    that the local path pays ~5.5s to import sentence-transformers and build the
    MiniLM, which the daemon paid once at boot.

    Deliberately does NOT start a server. `serve.py` is unauthenticated on
    localhost and its lifecycle is a decision somebody should make on purpose;
    the caller says how to make it instead.
    """
    import os

    if os.environ.get("MANAMAP_NO_DAEMON"):
        return None
    target = os.environ.get("MANAMAP_DAEMON") or "127.0.0.1:8000"
    try:
        import http.client
        import json as _json

        from manamap.sven import stream

        host, _, port = target.partition(":")
        conn = http.client.HTTPConnection(host or "127.0.0.1",
                                          int(port or 8000), timeout=0.15)
        body = _json.dumps({"question": question, "deep": deep,
                            "no_cache": no_cache})
        conn.request("POST", "/api/ask/stream", body,
                     {"Content-Type": "application/json"})
        conn.sock.settimeout(600)      # only the CONNECT needs to be impatient
        response = conn.getresponse()
        if response.status != 200:
            return None
        return stream.iter_frames(iter(lambda: response.readline(), b""))
    except Exception:                                  # noqa: BLE001 - fail open
        return None


def _ask(args):
    """`mm ask` — one question, streamed.

    THE STDOUT/STDERR SPLIT IS THE WHOLE CONTRACT, and it is `console.py`'s first
    rule: stdout is the ANSWER, stderr is the theatre. So `mm ask "..." > out.txt`
    captures the answer and nothing else, while the narration still reaches a
    terminal.

    On a TTY the two streams share a cursor, so a narration line arriving
    mid-sentence would land beside the answer text. The fix is to write the
    newline TO STDERR — `_col` tracks how far into a line stdout is, and the
    break goes on the theatre stream. Injecting it into stdout would corrupt a
    redirect, which is exactly what the rule above forbids.
    """
    import sys

    from manamap import console

    if getattr(args, "spend", False):
        from manamap.sven import spend

        print(spend.report())
        return 0

    question = " ".join(args.question)
    out, err = sys.stdout, sys.stderr
    as_json = getattr(args, "as_json", False)

    try:
        from manamap.sven import llm, loop
    except ImportError as exc:                      # pragma: no cover - defensive
        err.write(f"sven is not importable: {exc}\n")
        return 1

    deep = getattr(args, "deep", False)
    no_cache = getattr(args, "no_cache", False)
    col = 0
    started = __import__("time").time()
    try:
        frames = _sven_frames(question, deep=deep, no_cache=no_cache)
        warm = frames is not None
        if not warm:
            frames = loop.run(question,
                              model=llm.DEEP_MODEL if deep else None,
                              use_cache=not no_cache)
        for kind, data in frames:
            if as_json:
                import json as _json
                out.write(_json.dumps({"t": kind, "d": data}) + "\n")
                out.flush()
                continue
            if kind == "text":
                out.write(data)
                out.flush()
                col = 0 if data.endswith("\n") else col + len(data)
                continue
            if col:                                  # break the line on STDERR
                err.write("\n")
                col = 0
            if kind == "tool":
                err.write(f"  · {data}\n" if not console.is_plain()
                          else f"  tool {data}\n")
            elif kind == "note":
                err.write(f"  {data}\n")
            elif kind == "error":
                err.write(f"  ! {data}\n")
            elif kind == "done" and isinstance(data, dict):
                err.write(f"  {data.get('summary', '')}\n")
            err.flush()
    except llm.SvenUnavailable as exc:
        err.write(f"\n{exc}\n")
        return 1
    except KeyboardInterrupt:
        err.write("\n  interrupted\n")
        return 130
    if not as_json:
        out.write("\n")
        # SAY WHAT THE COLD PATH COST, and how to stop paying it. A slow answer
        # with no explanation reads as "this tool is slow"; the same answer with
        # this line reads as "there is a server I did not start". The threshold
        # is 2s so a fast local answer stays quiet.
        elapsed = __import__("time").time() - started
        if not warm and elapsed > 2:
            err.write(f"  · cold start ({elapsed:.1f}s) — `manamap serve` in "
                      f"another window makes this about 0.4s\n")
    return 0


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
    elif args.command == "vae-cache":
        from manamap.training import vae_cache

        vae_cache.main(args)
    elif args.command == "project-spaces":
        from manamap.analysis import project_spaces

        project_spaces.main(args)
    elif args.command == "recoverability":
        from manamap.training import recoverability

        recoverability.main(args)
    elif args.command == "train-cardbert":
        from manamap.training import train_cardbert

        train_cardbert.main(args)
    elif args.command == "span-cache":
        from manamap.training import span_encoder

        span_encoder.main(args)
    elif args.command == "train-vae":
        from manamap.training import train_vae

        train_vae.main(args)
    elif args.command == "eval-commander-search":
        from manamap.analysis import eval_commander_search
        eval_commander_search.main(args)
    elif args.command == "ask":
        raise SystemExit(_ask(args))
    elif args.command == "pilot":
        import sys

        code = _daemon_run(sys.argv[2:])
        if code is not None:
            raise SystemExit(code)
        run_pilot_step(args)
    else:
        run_step(args.command, space=getattr(args, "space", None))


if __name__ == "__main__":
    main()
