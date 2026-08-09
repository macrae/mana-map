"""Command-line interface: `manamap <step>` or `manamap run [--from STEP]`.

Step modules are imported lazily at dispatch time so `manamap --help`
stays fast (no torch import).
"""

import argparse
import re

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

    add_pilot_parser(subparsers)

    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "run":
        run(start=args.start)
    elif args.command == "pilot":
        run_pilot_step(args)
    else:
        run_step(args.command)


if __name__ == "__main__":
    main()
