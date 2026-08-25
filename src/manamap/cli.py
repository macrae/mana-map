"""Command-line interface: `manamap <step>` or `manamap run [--from STEP]`.

Step modules are imported lazily at dispatch time so `manamap --help`
stays fast (no torch import).
"""

import argparse
import re

from manamap import console
from manamap.pilot.registry import add_pilot_parser, run_pilot_step
from manamap.pipeline import STEP_NAMES, STEPS, run, run_step


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
    elif args.command == "eval-commander-search":
        from manamap.analysis import eval_commander_search
        eval_commander_search.main(args)
    elif args.command == "pilot":
        run_pilot_step(args)
    else:
        run_step(args.command)


if __name__ == "__main__":
    main()
