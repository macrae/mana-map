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
    ("deck-facts", "manamap.pilot.deck_facts", "Deterministic deck facts agents would else re-derive"),
    ("deck-history", "manamap.pilot.deck_history", "Applied swaps (from git) + the swaps still pending"),
    ("validate-deck-map", "manamap.pilot.validate_deck_map",
     "Form-check a named deck map (distinct names, membership untouched)"),
    ("merge-deck-map", "manamap.pilot.merge_deck_map",
     "Merge the cartographer's names into deck_map.json (names only)"),
    ("deck-map", "manamap.pilot.deck_map",
     "A deck's own constellation: local layout + 2-level clusters"),
    ("deck-audit", "manamap.pilot.deck_audit", "Cited axis targets + engine activation: is this deck any good?"),
    ("mana-analysis", "manamap.pilot.mana_analysis", "Deterministic mana/land analysis behind Sources Say"),
    ("pool-facts", "manamap.pilot.pool_facts", "What deck can I build from a box of cards?"),
    ("build-deck", "manamap.pilot.build_deck", "brief.json -> build_plan.json, deterministic"),
    ("validate-build", "manamap.pilot.validate_build", "Form-check a build plan against the contract"),
    ("validate-considering", "manamap.pilot.validate_considering", "Form-check The Short List (the ten)"),
    ("validate-diagnosis", "manamap.pilot.validate_diagnosis", "Form-check a deck diagnosis (axes re-derived, cuts checked against verified stacks)"),
    ("validate-goldfish-targets", "manamap.pilot.validate_goldfish_targets", "Form-check the engine declaration goldfish and deck-audit price"),
    ("diagnosis-report", "manamap.pilot.diagnosis_report", "Render a deck diagnosis as readable markdown"),
    ("validate-tutor-guide", "manamap.pilot.validate_tutor_guide", "Form-check the Fetch Quests tutor guide"),
    ("validate-strategic-frame", "manamap.pilot.validate_strategic_frame", "Form-check a strategic frame"),
    ("artist-credits", "manamap.pilot.artist_credits", "Standout artists and art themes in a deck"),
    ("build-manual", "manamap.pilot.build_manual", "Render a deck's magazine issue (sections per issue_spec)"),
    ("build-index", "manamap.pilot.build_index", "Render manuals/index.html — the newsstand"),
    ("validate-issue", "manamap.pilot.validate_issue", "Form-check issue.json + issue_plan.json"),
    ("scenario-facts", "manamap.pilot.scenario_facts", "Deterministic brief for a stack scenario (board, bodies, drain arithmetic)"),
    ("merge-prose", "manamap.pilot.merge_prose", "Merge an agent's .agent-out prose into manual_prose.json, keys it owns only"),
    ("cache-status", "manamap.pilot.agent_cache", "Have an agent routine's inputs changed?"),
    ("cache-record", "manamap.pilot.agent_cache", "Record the fingerprint that produced an artifact"),
    ("cache-clear", "manamap.pilot.agent_cache", "Drop cache records for a deck or routine"),
    ("cache-rebless", "manamap.pilot.agent_cache", "Re-record every STALE_OK routine without spawning"),
    ("cache-snapshot", "manamap.pilot.agent_cache", "Record every routine's status BEFORE a cache-format change"),
    ("cache-rerecord", "manamap.pilot.agent_cache", "Re-fingerprint what a format change invalidated (gated on a snapshot)"),
    ("impact", "manamap.pilot.impact", "What does the latest deck change touch? Deterministic, report-only"),
    ("validate-strategy", "manamap.pilot.validate_strategy", "Form-check strategy.md + CHANGELOG"),
    ("build-strategy-db", "manamap.pilot.build_strategy_db", "Chunk + embed strategy.md into the strategy DB"),
    ("query-strategy", "manamap.pilot.query_strategy", "Semantic top-k strategy search"),
    ("lookup-strategy", "manamap.pilot.query_strategy", "Exact strategy section fetch by id"),
]

_DECK_COMMANDS = {
    "fetch-deck", "validate-deck", "validate-stack", "goldfish", "build-manual",
    "validate-issue", "cache-status", "cache-record", "cache-clear", "cache-rebless",
    "cache-snapshot", "cache-rerecord",
    "artist-credits",
    "bracket-check", "build-deck", "validate-build", "deck-facts", "deck-audit", "deck-map", "merge-deck-map", "validate-deck-map", "deck-history",
    "mana-analysis", "validate-strategic-frame",
    "validate-considering", "validate-diagnosis", "validate-goldfish-targets",
    "diagnosis-report",
    "validate-tutor-guide", "impact", "scenario-facts", "merge-prose",
}


def add_pilot_parser(subparsers):
    """Attach the `pilot` subcommand group to the top-level subparsers."""
    pilot = subparsers.add_parser("pilot", help="Pilot's-manual subsystem commands")
    pilot_sub = pilot.add_subparsers(dest="pilot_command", required=True)

    for name, _, description in PILOT_STEPS:
        cmd = pilot_sub.add_parser(name, help=description)
        if name in _DECK_COMMANDS:
            cmd.add_argument("slug", help="Deck slug (kebab-case, e.g. goblin-storm)")
        if name == "merge-prose":
            cmd.add_argument("routine", choices=["coach-prose", "writer-prose"],
                             help="Which routine's keys to merge; it may write "
                                  "ONLY the keys that routine owns")
        if name == "query-rules":
            cmd.add_argument("query", help="Natural-language rules question")
            cmd.add_argument("--k", type=int, default=None, help="Number of results")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "lookup-rule":
            cmd.add_argument("rule_id", help="Exact rule number, e.g. 702.40a")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "validate-stack":
            cmd.add_argument("--stack", default=None, help="Only this scenario id (e.g. 001)")
            cmd.add_argument("--scenario-only", action="store_true", dest="scenario_only",
                             help="Preflight the scenario before resolving it (free; "
                                  "run this BEFORE spawning a resolver)")
        if name == "query-strategy":
            cmd.add_argument("query", help="Natural-language strategy question")
            cmd.add_argument("--k", type=int, default=None, help="Number of results")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "lookup-strategy":
            cmd.add_argument("section_id", help="Exact section id, e.g. strategy:tempo")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "artist-credits":
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "build-deck":
            cmd.add_argument("--write-decklist", action="store_true", dest="write_decklist",
                             help="Also write decklist.txt for fetch-deck")
        if name == "bracket-check":
            cmd.add_argument("--target", type=int, default=None,
                             help="Target bracket 1-5; exits 1 if the floor exceeds it")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "cache-status":
            cmd.add_argument("--routine", default=None,
                             help="Routine id (e.g. writer-prose, stack:001); omit for all")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "scenario-facts":
            cmd.add_argument("--stack", default=None, help="Only this scenario id (e.g. 001)")
            cmd.add_argument("--out", default=None, help="Also write JSON here (a view, never tracked)")
        if name == "cache-snapshot":
            # NOT slug-guarded, deliberately: a snapshot is explicitly merged
            # across decks, so one file covering the fleet is the intended use.
            # `resolve_out_path` would forbid the correct filename.
            cmd.add_argument("--out", required=True,
                             help="Snapshot file; merged across decks so one file covers the fleet")
        if name == "cache-rerecord":
            cmd.add_argument("--snapshot", required=True,
                             help="Snapshot taken BEFORE the change")
            cmd.add_argument("--dry-run", action="store_true", dest="dry_run",
                             help="Report what would be re-recorded and change nothing")
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
        if name == "deck-facts":
            cmd.add_argument("--out", default=None,
                             help="Write JSON here instead of stdout (a view, never tracked)")
        if name == "pool-facts":
            # Takes paths, not a slug: a collection is not a deck, and forcing it
            # into data/decks/<slug>/ would put it in reach of validate-deck.
            cmd.add_argument("targets", nargs="+",
                             help="Decklist files or directories of them (e.g. share/)")
            cmd.add_argument("--exclude", action="append", default=[],
                             help="A file to leave out (repeatable) — e.g. a deck "
                                  "you are keeping assembled")
            cmd.add_argument("--json", action="store_true", dest="as_json")
            # NOT slug-guarded: pool-facts takes paths, not a slug — there is no
            # slug to scope the filename to. A collection is not a deck.
            cmd.add_argument("--out", default=None,
                             help="Write JSON here as well (a view, never tracked)")
        if name == "diagnosis-report":
            cmd.add_argument("--out", default=None,
                             help="Write markdown here instead of stdout (a view, never tracked)")
        if name == "deck-audit":
            cmd.add_argument("--archetype", default=None,
                             help="aggro|control|combo|voltron — overrides what "
                                  "strategic_frame.json says; omit for the base targets")
        if name in ("impact", "deck-audit",
                    "deck-history"):
            cmd.add_argument("--json", action="store_true", dest="as_json")
            cmd.add_argument("--out", default=None,
                             help="Write JSON here instead of stdout (a view, never tracked)")


def run_pilot_step(args):
    """Dispatch a parsed pilot command to its module's main(args)."""
    for name, module_path, _ in PILOT_STEPS:
        if name == args.pilot_command:
            importlib.import_module(module_path).main(args)
            return
    raise ValueError(f"Unknown pilot command: {args.pilot_command!r}")
