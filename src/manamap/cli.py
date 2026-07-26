"""Command-line interface: `manamap <step>` or `manamap run [--from STEP]`.

Step modules are imported lazily at dispatch time so `manamap --help`
stays fast (no torch import).
"""

import argparse

from manamap.pilot.registry import add_pilot_parser, run_pilot_step
from manamap.pipeline import STEP_NAMES, STEPS, run, run_step


def build_parser():
    parser = argparse.ArgumentParser(
        prog="manamap",
        description="MTG card embedding pipeline — run all steps or one at a time.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full 13-step pipeline in order")
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
