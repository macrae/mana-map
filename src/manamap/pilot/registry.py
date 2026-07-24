"""Pilot subcommand registry and argparse wiring.

Unlike the pipeline STEPS, pilot commands are standalone and per-deck
parameterized. Modules import lazily at dispatch so `manamap --help` stays fast.
"""

import importlib

# (name, dotted module, description)
PILOT_STEPS = [
    ("fetch-deck", "manamap.pilot.fetch_deck", "decklist.txt -> cards.json via Scryfall"),
    ("validate-deck", "manamap.pilot.validate_deck", "Check 100-card/commander/singleton invariants"),
    ("download-rules", "manamap.pilot.download_rules", "Download the Comprehensive Rules TXT"),
    ("build-rules-db", "manamap.pilot.build_rules_db", "Chunk + embed the CR into the rules DB"),
    ("query-rules", "manamap.pilot.query_rules", "Semantic top-k rules search"),
    ("lookup-rule", "manamap.pilot.query_rules", "Exact rule fetch by number"),
    ("validate-stack", "manamap.pilot.validate_stack", "Enforce the citation contract on scenarios"),
    ("goldfish", "manamap.pilot.goldfish", "Seeded Monte Carlo resource-development metrics"),
    ("build-manual", "manamap.pilot.build_manual", "Render the zine HTML from verified artifacts"),
    ("build-index", "manamap.pilot.build_index", "Render the manuals/index.html gallery"),
]

_DECK_COMMANDS = {"fetch-deck", "validate-deck", "validate-stack", "goldfish", "build-manual"}


def add_pilot_parser(subparsers):
    """Attach the `pilot` subcommand group to the top-level subparsers."""
    pilot = subparsers.add_parser("pilot", help="Pilot's-manual subsystem commands")
    pilot_sub = pilot.add_subparsers(dest="pilot_command", required=True)

    for name, _, description in PILOT_STEPS:
        cmd = pilot_sub.add_parser(name, help=description)
        if name in _DECK_COMMANDS:
            cmd.add_argument("slug", help="Deck slug (kebab-case, e.g. goblin-storm)")
        if name == "query-rules":
            cmd.add_argument("query", help="Natural-language rules question")
            cmd.add_argument("--k", type=int, default=None, help="Number of results")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "lookup-rule":
            cmd.add_argument("rule_id", help="Exact rule number, e.g. 702.40a")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "validate-stack":
            cmd.add_argument("--stack", default=None, help="Only this scenario id (e.g. 001)")


def run_pilot_step(args):
    """Dispatch a parsed pilot command to its module's main(args)."""
    for name, module_path, _ in PILOT_STEPS:
        if name == args.pilot_command:
            importlib.import_module(module_path).main(args)
            return
    raise ValueError(f"Unknown pilot command: {args.pilot_command!r}")
