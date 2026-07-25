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
    ("bracket-check", "manamap.pilot.bracket", "Computed bracket floor and its evidence"),
    ("artist-credits", "manamap.pilot.artist_credits", "Standout artists and art themes in a deck"),
    ("build-manual", "manamap.pilot.build_manual", "Render a deck's 15-department magazine issue"),
    ("build-index", "manamap.pilot.build_index", "Render manuals/index.html — the newsstand"),
    ("validate-issue", "manamap.pilot.validate_issue", "Form-check issue.json + issue_plan.json"),
    ("cache-status", "manamap.pilot.agent_cache", "Have an agent routine's inputs changed?"),
    ("cache-record", "manamap.pilot.agent_cache", "Record the fingerprint that produced an artifact"),
    ("cache-clear", "manamap.pilot.agent_cache", "Drop cache records for a deck or routine"),
    ("validate-strategy", "manamap.pilot.validate_strategy", "Form-check strategy.md + CHANGELOG"),
    ("build-strategy-db", "manamap.pilot.build_strategy_db", "Chunk + embed strategy.md into the strategy DB"),
    ("query-strategy", "manamap.pilot.query_strategy", "Semantic top-k strategy search"),
    ("lookup-strategy", "manamap.pilot.query_strategy", "Exact strategy section fetch by id"),
]

_DECK_COMMANDS = {
    "fetch-deck", "validate-deck", "validate-stack", "goldfish", "build-manual",
    "validate-issue", "cache-status", "cache-record", "cache-clear", "artist-credits",
    "bracket-check",
}


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
        if name == "query-strategy":
            cmd.add_argument("query", help="Natural-language strategy question")
            cmd.add_argument("--k", type=int, default=None, help="Number of results")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "lookup-strategy":
            cmd.add_argument("section_id", help="Exact section id, e.g. strategy:tempo")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "artist-credits":
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "bracket-check":
            cmd.add_argument("--target", type=int, default=None,
                             help="Target bracket 1-5; exits 1 if the floor exceeds it")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "cache-status":
            cmd.add_argument("--routine", default=None,
                             help="Routine id (e.g. writer-prose, stack:001); omit for all")
            cmd.add_argument("--json", action="store_true", dest="as_json")
            cmd.add_argument("--force", action="store_true",
                             help="Always report MISS (deliberate rebuild)")
        if name == "cache-record":
            cmd.add_argument("--routine", required=True,
                             help="Routine id (e.g. issue-plan, stack:004)")
        if name == "cache-clear":
            cmd.add_argument("--routine", default=None,
                             help="Routine id; omit to clear the whole deck")
        if name == "fetch-deck":
            cmd.add_argument("--force", action="store_true",
                             help="Re-fetch from Scryfall even if the decklist is unchanged")


def run_pilot_step(args):
    """Dispatch a parsed pilot command to its module's main(args)."""
    for name, module_path, _ in PILOT_STEPS:
        if name == args.pilot_command:
            importlib.import_module(module_path).main(args)
            return
    raise ValueError(f"Unknown pilot command: {args.pilot_command!r}")
