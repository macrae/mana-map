"""Pilot subcommand registry and argparse wiring.

Unlike the pipeline STEPS, pilot commands are standalone and per-deck
parameterized. Modules import lazily at dispatch so `manamap --help` stays fast.
"""

import importlib

# (name, dotted module, description)
PILOT_STEPS = [
    ("check-in", "manamap.pilot.check_in", "a paper decklist -> decklist.txt (diff, refuse, apply, re-derive)"),
    ("build-page", "manamap.pilot.build_page", "the compact deck page (the Pilot's Manual)"),
    ("targeting", "manamap.sim.threat", "who the pod attacks, measured from sim logs (opponent modelling)"),
    ("fetch-deck", "manamap.pilot.fetch_deck", "decklist.txt -> cards.json via Scryfall"),
    ("validate-deck", "manamap.pilot.validate_deck", "Check 100-card/commander/singleton invariants"),
    ("download-rules", "manamap.pilot.download_rules", "Download the Comprehensive Rules TXT"),
    ("build-rules-db", "manamap.pilot.build_rules_db", "Chunk + embed the CR into the rules DB"),
    ("query-rules", "manamap.pilot.query_rules", "Semantic top-k rules search"),
    ("lookup-rule", "manamap.pilot.query_rules", "Exact rule fetch by number"),
    ("validate-stack", "manamap.pilot.validate_stack", "Enforce the citation contract on scenarios"),
    ("goldfish", "manamap.pilot.goldfish", "Seeded Monte Carlo resource-development metrics"),
    ("benchmark", "manamap.pilot.benchmark",
     "The standard benchmark: four measures under one frozen configuration"),
    ("scaffold-targets", "manamap.pilot.scaffold_targets",
     "A STARTING goldfish_targets.json, derived and marked as a draft to edit"),
    ("bracket-check", "manamap.pilot.bracket", "Computed bracket floor and its evidence"),
    ("deck-facts", "manamap.pilot.deck_facts", "Deterministic deck facts agents would else re-derive"),
    ("validate-recon", "manamap.pilot.validate_recon",
     "Form-check deck_recon.json: cards real, legal, in identity; ownership falsified"),
    ("card-search", "manamap.pilot.card_search",
     "Mine the corpus for candidates: colour identity, oracle regex, role, cmc"),
    ("commander-search", "manamap.pilot.commander_search_cmd",
     "Cards in, commanders out: rank real commanders by proximity to a seed"),
    ("brew", "manamap.pilot.brew",
     "Start a deck: a commander, the cards you kept, and a style -> brief.json"),
    ("archetypes", "manamap.pilot.archetypes",
     "How a commander is actually built, and the role template each style wants"),
    ("deck-history", "manamap.pilot.deck_history", "Applied swaps (from git) + the swaps still pending"),
    ("deck-notes", "manamap.pilot.deck_notes",
     "The captain's log: add a note about a game, list them, show one (append-only, authored)"),
    ("validate-debrief", "manamap.pilot.validate_debrief",
     "Form-check the debrief annotations against the log they annotate"),
    ("merge-debrief", "manamap.pilot.merge_debrief",
     "Merge the debrief agent's annotations into log_annotations.json, by entry id"),
    ("install-agent", "manamap.pilot.install_agent",
     "An agent's whole-file handoff becomes the tracked artifact, STAMPED"),
    ("captains-log", "manamap.pilot.captains_log",
     "The captain's log: which nights this deck flew, and which are rendered"),
    ("merge-captains-log", "manamap.pilot.merge_captains_log",
     "Merge the captains-log agent's prose into captains_log.json"),
    ("validate-captains-log", "manamap.pilot.validate_captains_log",
     "Form-check captains_log.json against the log it renders"),
    ("simulate", "manamap.sim.forge",
     "Run N Commander games of this deck against opponents in Forge, headless; record the run (◆ sampled)"),
    ("experiment", "manamap.sim.experiment",
     "A/B two versions of one deck against the same table: one artifact, each figure for both arms, the delta and whether it is noise"),
    ("fetch-opponent", "manamap.sim.opponents",
     "An opponent seat under data/opponents/<slug>/ from EDHREC's average deck for a commander"),
    ("sim-scenario", "manamap.sim.bridge",
     "Lift one game at one moment out of a Forge run into a game_state v2 scenario for resolve-stack"),
    ("validate-sim", "manamap.sim.validate_sim",
     "Form-check simulation run records; re-derive the analysis from logs where they exist"),
    ("model-coverage", "manamap.pilot.model_coverage",
     "What the goldfish CANNOT see in this deck, before you run it"),
    ("regen", "manamap.pilot.regen",
     "Regenerate the fleet's derived artifacts in dependency order, in parallel"),
    ("deck-info", "manamap.pilot.deck_info",
     "The workbench view: one deck, one screen — version, record, status, figures, and what to do next"),
    ("diagnose", "manamap.pilot.diagnostic",
     "One diagnostic run: stall risk, engine online against what the deck DECLARES, "
     "and the mana under both — every rate with its interval"),
    ("assess", "manamap.pilot.assess",
     "Triage a pile of cards against one deck before spending: legality, cost, what it does, "
     "what it is gated on, whether any model here can see it, and what it would replace"),
    ("validate-diagnostic", "manamap.pilot.validate_diagnostic",
     "Form-check diagnostic.json: every rate carries its interval"),
    ("validate-net-change", "manamap.pilot.validate_net_change",
     "Form-check net_change.json: nothing under the MDE is ranked"),
    ("validate-branch", "manamap.pilot.validate_branch",
     "Form-check branch.json: the objective is falsifiable and a proposal "
     "freezes what it was accepted on"),
    ("net-change", "manamap.pilot.net_change",
     "What a branch costs, what it buys, and whether it met its objective"),
    ("calibrate", "manamap.pilot.calibrate",
     "Does the model track real outcomes? Refuses below a usable sample"),
    ("close", "manamap.pilot.close",
     "Turn the diagnostic's bottleneck into a candidate pool"),
    ("mana-fit", "manamap.pilot.mana_fit",
     "Right-size lands, rocks and dorks against the pip distribution the list "
     "actually has. Run it whenever the nonland half moves"),
    ("upgrades", "manamap.pilot.upgrades",
     "What in this list has a cheaper card doing its job — the obsolescence index, "
     "read deck-aware. Proposes comparisons with BOTH sides; `candidates` measures"),
    ("candidates", "manamap.pilot.candidates",
     "Rank a pool of cards by what each MEASURABLY does: substitute it in, re-run the "
     "diagnostic, report the difference with an interval"),
    ("deck-branch", "manamap.pilot.deck_branch",
     "A candidate 99 you cannot yet sleeve: diff it, price it against your collection, "
     "measure it, and merge it only when the cards exist"),
    ("deck-version", "manamap.pilot.deck_versions",
     "Every list this deck has been: numbered from git, tagged by you, joined to the games played on it"),
    ("validate-log-causes", "manamap.pilot.validate_log_causes",
     "Form-check log_causes.json: the vocabulary, the ids it names, and a cause that contradicts its result"),
    ("validate-deck-versions", "manamap.pilot.validate_deck_versions",
     "Form-check deck_versions.json: the lifecycle, the paper lock, and that they do not contradict"),
    ("deck-state", "manamap.pilot.deck_state",
     "Is this still a deck or a pile of cards: archive / supersede / retire / revive"),
    ("deck-delete", "manamap.pilot.deck_delete",
     "Remove a deck that was never sleeved, never played and never published"),
    ("prescribe", "manamap.pilot.prescribe",
     "Ask the deck doctor one question: open a prescription, list them, or merge the answer"),
    ("validate-prescription", "manamap.pilot.validate_prescription",
     "Form-check prescriptions (the diagnosis contract, scoped to a question)"),
    ("validate-deck-map", "manamap.pilot.validate_deck_map",
     "Form-check a named deck map (distinct names, membership untouched)"),
    ("merge-deck-map", "manamap.pilot.merge_deck_map",
     "Merge the cartographer's names into deck_map.json (names only)"),
    ("validate-engine", "manamap.pilot.validate_engine",
     "Form-check an engine model (stages, completeness, verified_by re-checked)"),
    ("deck-status", "manamap.pilot.deck_status",
     "Lifecycle completeness + staleness: is this deck finished, and is any of it stale?"),
    ("engine-facts", "manamap.pilot.engine_facts",
     "Deterministic engine brief: declaration + verified pairings + map scatter"),
    ("deck-map", "manamap.pilot.deck_map",
     "A deck's own constellation: local layout + 2-level clusters"),
    ("deck-audit", "manamap.pilot.deck_audit", "Cited axis targets + engine activation: is this deck any good?"),
    ("card-value", "manamap.pilot.card_value",
     "What is each card WORTH: swap it for a blank and measure what the deck loses"),
    ("mana-analysis", "manamap.pilot.mana_analysis", "Deterministic mana/land analysis: pips vs sources, castability, tapped lands"),
    ("pool-facts", "manamap.pilot.pool_facts", "What deck can I build from a box of cards?"),
    ("build-deck", "manamap.pilot.build_deck", "brief.json -> build_plan.json, deterministic"),
    ("validate-build", "manamap.pilot.validate_build", "Form-check a build plan against the contract"),
    ("validate-considering", "manamap.pilot.validate_considering", "Form-check a legacy considering.json (the retired Short List; frozen on published decks)"),
    ("validate-pending", "manamap.pilot.validate_pending",
     "The queue of changes decided but not applied; closure is DERIVED from the deck"),
    ("validate-diagnosis", "manamap.pilot.validate_diagnosis", "Form-check a deck diagnosis (axes re-derived, cuts checked against verified stacks)"),
    ("validate-goldfish-targets", "manamap.pilot.validate_goldfish_targets", "Form-check the engine declaration goldfish and deck-audit price"),
    ("diagnosis-report", "manamap.pilot.diagnosis_report", "Render a deck diagnosis as readable markdown"),
    ("validate-tutor-guide", "manamap.pilot.validate_tutor_guide", "Form-check the tutor guide (one wish per tutor)"),
    ("validate-strategic-frame", "manamap.pilot.validate_strategic_frame", "Form-check a strategic frame"),
    ("artist-credits", "manamap.pilot.artist_credits", "Standout artists and art themes in a deck"),
    ("short-list-art", "manamap.pilot.short_list_art", "LEGACY: resolve a frozen considering.json's ten to card art (Scryfall; tracked sidecar)"),
    ("build-manual", "manamap.pilot.build_manual", "LEGACY: render a deck's page with the frozen magazine renderer (replaced by manual-v5)"),
    ("build-index", "manamap.pilot.build_index", "Render manuals/index.html and data/decks/index.json (the deck list + manifest)"),
    ("issue-length", "manamap.pilot.issue_length", "LEGACY: how long is this page, and where did the length go?"),
    ("validate-issue", "manamap.pilot.validate_issue", "Form-check issue.json + the LEGACY issue_plan.json on published decks"),
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
    "check-in", "targeting", "build-page", "fetch-deck", "validate-deck", "validate-stack", "goldfish", "build-manual",
    "validate-issue", "cache-status", "cache-record", "cache-clear", "cache-rebless",
    "cache-snapshot", "cache-rerecord",
    "artist-credits",
    "model-coverage",
    "bracket-check", "build-deck", "validate-build", "deck-facts", "deck-audit", "deck-map", "deck-status", "engine-facts", "validate-engine", "merge-deck-map", "validate-deck-map", "deck-history",
    "mana-analysis", "validate-strategic-frame", "scaffold-targets",
    "validate-considering", "validate-diagnosis", "validate-goldfish-targets",
    "validate-recon",
    "diagnosis-report",
    "validate-tutor-guide", "impact", "scenario-facts", "merge-prose",
    "short-list-art", "issue-length", "card-value", "validate-pending",
    "deck-notes", "validate-debrief", "merge-debrief", "prescribe", "validate-prescription",
    "captains-log", "merge-captains-log", "validate-captains-log",
    "install-agent",
    "deck-version", "deck-state", "deck-delete", "validate-deck-versions",
    "validate-log-causes",
    "deck-branch", "diagnose", "assess", "candidates", "close",
    "upgrades", "mana-fit",
    "validate-diagnostic", "net-change", "validate-net-change", "validate-branch", "deck-info", "simulate", "validate-sim", "sim-scenario", "experiment",
}


def _causes():
    """`deck_notes.CAUSES` keys, imported lazily so `--help` stays fast."""
    from manamap.pilot.deck_notes import CAUSES

    return CAUSES


def add_pilot_parser(subparsers):
    """Attach the `pilot` subcommand group to the top-level subparsers."""
    pilot = subparsers.add_parser("pilot", help="The pilot bench: decks, log, versions, sim, evidence")
    pilot_sub = pilot.add_subparsers(dest="pilot_command", required=True)

    for name, _, description in PILOT_STEPS:
        cmd = pilot_sub.add_parser(name, help=description)
        if name in _DECK_COMMANDS:
            # `deck-status --all` is the fleet view, so its slug is optional.
            # Every other per-deck command still requires one — an optional slug
            # elsewhere is how a command silently operates on the wrong deck.
            nargs = "?" if name == "deck-status" else None
            cmd.add_argument("slug", nargs=nargs,
                             help="Deck slug (kebab-case, e.g. goblin-storm)")
        if name == "deck-map":
            # `--out` EXISTS BECAUSE `--space` DOES. `deck_map.main` has always
            # read `args.out`, but the parser never offered it, so the only way
            # to run a non-default space would be to OVERWRITE the tracked
            # `deck_map.json` — whose agent-authored city names were written
            # against the function space's clusters and do not survive a
            # re-clustering. `resolve_out_path` keeps it slug-scoped.
            cmd.add_argument("--out", default=None,
                             help="Also write JSON here (a view, never tracked). "
                                  "Use this to try a non-default --space without "
                                  "replacing the committed map")
        if name in ("deck-map", "build-deck", "close"):
            from manamap import spaces as _spaces
            cmd.add_argument(
                "--space", default=None, choices=_spaces.choices(),
                help=f"which embedding to read (default {_spaces.DEFAULT}). "
                     "cardbert clusters by TRIBE rather than by function — a "
                     "different question, and it loses functional similarity at "
                     "every pool size measured")
        if name == "merge-prose":
            # Choices come from `merge_prose.AGENT_FILE`, never a literal list.
            # A hardcoded pair here is the same mistake this repo bans in prompts —
            # a third prose routine was added and the CLI silently refused it while
            # every other layer accepted it.
            from manamap.pilot.merge_prose import AGENT_FILE
            cmd.add_argument("routine", choices=sorted(AGENT_FILE),
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
        if name == "install-agent":
            cmd.add_argument("--routine", required=True,
                             help="which routine's handoff to install (deck-engine, "
                                  "tutor-guide, strategic-frame, deck-map-names, …)")
            cmd.add_argument("--force", action="store_true",
                             help="install without a stamp when the deck has no "
                                  "cards.json sha (reads as staleness-unknown)")
        if name == "captains-log":
            # The skeleton the agent quotes: every deterministic fact about the
            # deck's nights and no prose. A VIEW, never tracked — same rule as
            # `deck-facts`; the tracked artifact is written by the merge.
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "merge-captains-log":
            cmd.add_argument("--kind", default="ship",
                             help="which log to merge into (ship; personal is reserved)")
        if name == "validate-issue":
            cmd.add_argument("--strict", action="store_true",
                             help="Fail on the length budget, not just report it")
        if name == "issue-length":
            cmd.add_argument("--json", action="store_true", dest="as_json")
            cmd.add_argument("--rendered", action="store_true",
                             help="True scroll height via playwright (slower)")
        if name == "benchmark":
            cmd.add_argument("slug", nargs="?", default=None,
                             help="One deck; omit (or --all) for the whole bench")
            cmd.add_argument("--all", action="store_true", dest="all_decks",
                             help="Every deck with a cards.json")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "scaffold-targets":
            cmd.add_argument("--force", action="store_true",
                             help="Overwrite an existing file (refused for an AUTHORED one)")
        if name == "build-deck":
            cmd.add_argument("--write-decklist", action="store_true", dest="write_decklist",
                             help="Also write decklist.txt for fetch-deck")
        if name == "bracket-check":
            cmd.add_argument("--target", type=int, default=None,
                             help="Target bracket 1-5; exits 1 if the floor exceeds it")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "cache-status":
            cmd.add_argument("--routine", default=None,
                             help="Routine id (e.g. pilot-notes, stack:001); omit for all")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "deck-status":
            cmd.add_argument("--json", action="store_true", dest="as_json")
            # The fleet view. Nine decks and no way to ask "what is outstanding
            # everywhere" was the other half of the problem `pending.json` fixes.
            cmd.add_argument("--all", action="store_true", dest="all_decks",
                             help="Every deck, one row each, plus fleet totals")
        if name == "validate-pending":
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "card-value":
            cmd.add_argument("--metric", default="kill-by-8",
                             choices=["kill-by-8", "kill-by-10", "board-power", "hoard"],
                             help="What to rank by (they rank differently — say which)")
            cmd.add_argument("--iterations", type=int, default=None,
                             help="Games per card (default 3000; the full 10000 is ~3x slower)")
            cmd.add_argument("--json", action="store_true", dest="as_json")
            cmd.add_argument("--out", default=None,
                             help="Also write JSON here (a view, never tracked)")
        if name == "engine-facts":
            # Both flags exist in `engine_facts.main` and neither was registered,
            # so the agent this brief was built for could not reach the arrays it
            # is meant to read and called `build()` directly. A CLI that ignores
            # its own documented flags is worse than one that lacks them.
            cmd.add_argument("--json", action="store_true", dest="as_json")
            cmd.add_argument("--out", default=None,
                             help="Also write JSON here (a view, never tracked)")
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
                             help="Routine id (e.g. pilot-notes, stack:004, prescription:<id>)")
        if name == "cache-clear":
            cmd.add_argument("--routine", default=None,
                             help="Routine id; omit to clear the whole deck")
        if name == "build-page":
            cmd.add_argument("--out", default=None,
                             help="write elsewhere (a directory, or a path naming the slug)")
        if name == "targeting":
            cmd.add_argument("--run", action="append",
                             help="limit to one run id (repeatable); default pools every run")
            cmd.add_argument("--seed", type=int, default=0,
                             help="permutation seed (default 0; the test replays exactly)")
            cmd.add_argument("--iterations", type=int, default=None,
                             help="permutation iterations (default 10000)")
            cmd.add_argument("--json", action="store_true", dest="as_json",
                             help="print instead of writing threat/targeting.json")
        if name == "check-in":
            cmd.add_argument("--from", dest="source", required=True,
                             help="the paper decklist: a file, or - for stdin")
            cmd.add_argument("--write", action="store_true",
                             help="apply it, then run fetch-deck -> goldfish -> mana-analysis "
                                  "(default is a dry-run diff)")
            cmd.add_argument("--no-chain", action="store_true", dest="no_chain",
                             help="write decklist.txt only; skip the re-derivation")
            cmd.add_argument("--force", action="store_true",
                             help="apply despite a refusal. You want this approximately never")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "fetch-deck":
            cmd.add_argument("--force", action="store_true",
                             help="Re-fetch from Scryfall even if the decklist is unchanged")
        if name == "deck-notes":
            cmd.add_argument("action", choices=["add", "list", "show", "cause"],
                             help="add a note / list the log / show one entry / "
                                  "record how a game ended")
            cmd.add_argument("text", nargs="?", default=None,
                             help="the note (add), or the entry id (show, cause)")
            cmd.add_argument("--file", default=None,
                             help="read the note from this file (`-` for stdin) instead")
            cmd.add_argument("--result", default=None, choices=["win", "loss", "draw"])
            cmd.add_argument("--opponents", type=int, default=None,
                             help="how many OTHER players sat down")
            cmd.add_argument("--tag", action="append", default=[],
                             help="free tag, repeatable (e.g. --tag orinda --tag weekly)")
            cmd.add_argument("--at", default=None,
                             help="ISO timestamp override, for backfilling a game played earlier")
            # HOW IT ENDED, from a closed vocabulary — `deck_notes.CAUSES`. The
            # choices are derived, never re-typed here: a hand-copied literal is
            # how `pilot/registry.py` came to accept an embedding-space slug the
            # registry no longer knew.
            cmd.add_argument("--cause", default=None,
                             choices=sorted(_causes()),
                             help="how the game ended (add, or cause) — "
                                  "mana-drought | removal | wipe | combo | "
                                  "politics | raced | stalled | won")
            cmd.add_argument("--note", default=None,
                             help="cause: one line on why you filed it that way")
            cmd.add_argument("--since", default=None,
                             help="list: only entries at or after this ISO date")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "simulate":
            cmd.add_argument("--vs", action="append", default=[], metavar="SLUG",
                             help="an opponent seat (data/opponents/<slug> or data/decks/<slug>); repeatable")
            cmd.add_argument("--games", type=int, default=None,
                             help="number of games (default SIM_DEFAULT_GAMES)")
            cmd.add_argument("--jobs", type=int, default=None,
                             help="JVMs to run in parallel (default: CPUs - 1)")
            cmd.add_argument("--clock", type=int, default=None,
                             help="seconds before Forge calls a game a draw (default SIM_GAME_CLOCK_SECONDS)")
            cmd.add_argument("--list", action="store_true", help="list this deck's runs")
            cmd.add_argument("--dry-run", action="store_true", dest="dry_run",
                             help="print the JVM commands and the run id; run nothing")
            cmd.add_argument("--seed", type=int, default=None,
                             help="RNG seed (default derives from the configuration, so the default REPLAYS; pass one for a new sample)")
            cmd.add_argument("--vs-profile", default=None, dest="vs_profile",
                             choices=["Default", "Cautious", "Reckless", "Experimental"],
                             help="AI profile for every OPPONENT seat (default: Experimental, "
                                  "the standard pod since 2026-08-30). "
                                  "The pod is part of the instrument: a win rate is "
                                  "relative to how well the table plays.")
            cmd.add_argument("--profile", default=None, choices=["Default", "Cautious", "Reckless", "Experimental"],
                             help="AI profile for YOUR seat (opponents stay Default; measured: aggro profiles make a hold-up deck worse)")
            cmd.add_argument("--force", action="store_true",
                             help="replay an existing run id (it writes the same bytes)")
            cmd.add_argument("--analyze", default=None, metavar="RUN_ID",
                             help="re-derive a run's analysis from its kept logs (where the run was made)")
        if name == "model-coverage":
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "regen":
            # `--slug` is a FLAG, not a positional: regen's normal subject is the
            # whole fleet, and a required slug would make the common case the
            # awkward one.
            cmd.add_argument("--slug", default=None,
                             help="just this deck (and its branches)")
            cmd.add_argument("--only", action="append", default=None,
                             metavar="STAGE",
                             help="just this stage (repeatable): goldfish, "
                                  "mana-analysis, net-change, diagnose, "
                                  "benchmark, deck-info")
            cmd.add_argument("--jobs", type=int, default=None,
                             help="parallel workers (default: one per core). "
                                  "Targets run in parallel; the GAMES inside one "
                                  "run never do, so the output is bit-identical")
            cmd.add_argument("--dry-run", action="store_true",
                             help="print what would be regenerated, write nothing")
        if name == "deck-info":
            cmd.add_argument("--json", action="store_true", dest="as_json")
            # Running the fourteen gates costs ~2.3s of the ~5s this command
            # used to take, and this is the command the pilot runs to REMEMBER
            # WHERE A DECK STANDS. `--write` implies it: the committed
            # info.json must carry real verdicts, never "not checked".
            cmd.add_argument("--verify", action="store_true",
                             help="run every artifact's validator (slower); "
                                  "without it the gates are reported as not run, "
                                  "never as clean. Implied by --write")
            cmd.add_argument("--write", action="store_true",
                             help="write data/decks/<slug>/info.json for the deck page "
                                  "(committed, staleness-gated, no version block)")
        # --branch: measure a CANDIDATE list without touching the deck's own
        # artifacts. Scoping is structural rather than per-command — `deck_dir`
        # resolves the branch directory, so a branch run writes beside the
        # branch's own decklist and cannot overwrite the tracked one.
        if name in ('fetch-deck', 'bracket-check', 'mana-analysis', 'goldfish', 'deck-facts', 'deck-audit', 'deck-map', 'diagnose', 'candidates', 'assess', 'close', 'upgrades', 'mana-fit',
                    # The validators too: a branch's artifacts were gated by
                    # NOTHING, because no validator could be pointed at one.
                    'validate-deck', 'validate-deck-map',
                    'validate-goldfish-targets', 'validate-diagnostic',
                    'net-change', 'validate-net-change', 'validate-branch'):
            cmd.add_argument("--branch", default=None, metavar="NAME",
                             help="run against a branch (see `deck-branch <slug> list`) "
                                  "instead of the deck's own list")
        if name == "net-change":
            cmd.add_argument("--iterations", type=int, default=None)
            cmd.add_argument("--seed", type=int, default=None)
            cmd.add_argument("--write", action="store_true",
                             help="write net_change.json beside the branch")
            cmd.add_argument("--json", action="store_true")
        if name == "close":
            cmd.add_argument("--component", default=None, metavar="SUBSTRING",
                             help="which declared component to close "
                                  "(default: the diagnostic's own bottleneck)")
            cmd.add_argument("--limit", type=int, default=None)
            cmd.add_argument("--owned", action="store_true",
                             help="only cards in a box (collection.py; never "
                                  "deck membership)")
            cmd.add_argument("--write", action="store_true",
                             help="write data/decks/<slug>/pool.txt for "
                                  "`assess`/`candidates --pool library`")
            cmd.add_argument("--json", action="store_true")
        if name == "assess":
            cmd.add_argument("--pool", default=None,
                             help="a file of card names, a decklist, , or ")
            cmd.add_argument("--json", action="store_true")
        if name == "mana-fit":
            cmd.add_argument("--owned", action="store_true",
                             help="only propose cards in a box (collection.py)")
            cmd.add_argument("--limit", type=int, default=None)
            cmd.add_argument("--json", action="store_true")
        if name == "upgrades":
            cmd.add_argument("--pool", default=None,
                             help="a pile to cross against the list: a file of "
                                  "card names, a decklist, `library` for the "
                                  "Atlas's pool.txt, or `-` for stdin. A pile "
                                  "card with no comparison is REPORTED, never "
                                  "silently dropped")
            cmd.add_argument("--min-strength", dest="min_strength", type=float,
                             default=None, metavar="F",
                             help="floor on the index's 0-1 strength "
                                  "(default 0.4; nothing under it is a claim)")
            cmd.add_argument("--limit", type=int, default=None)
            cmd.add_argument("--owned", action="store_true",
                             help="only replacements in a box (collection.py; "
                                  "never deck membership)")
            cmd.add_argument("--out", default=None, metavar="PATH")
            cmd.add_argument("--json", action="store_true")
        if name == "candidates":
            cmd.add_argument("--pool", default=None,
                             help="a file of card names or a decklist; `-` for stdin")
            cmd.add_argument("--axis", default="engine_online_3",
                             help="engine_online_3|5|8, any_route_8, stall, land_drop")
            cmd.add_argument("--cut", default=None,
                             help="the card each candidate replaces (default: the "
                                  "most expensive spell the declaration does not name)")
            cmd.add_argument("--as", dest="join", default=None, metavar="TARGET",
                             help="count each candidate toward this declared "
                                  "target for the measurement — the hypothetical "
                                  "'if this were a multiplier, how far would the "
                                  "engine move'. Never written to the declaration")
            cmd.add_argument("--limit", type=int, default=None)
            cmd.add_argument("--iterations", type=int, default=None)
            cmd.add_argument("--json", action="store_true")
        if name == "diagnose":
            cmd.add_argument("--vs", default=None, metavar="REF",
                             help="compare against another list — `main` for the "
                                  "committed decklist, or a branch name")
            cmd.add_argument("--iterations", type=int, default=None)
            cmd.add_argument("--seed", type=int, default=None)
            cmd.add_argument("--no-read", action="store_true", dest="no_read",
                             help="skip the plain-language reading of a --vs comparison")
            cmd.add_argument("--json", action="store_true")
            cmd.add_argument("--write", action="store_true",
                             help="write diagnostic.json beside the list it measured")
        if name == "deck-branch":
            cmd.add_argument("action", nargs="?", default="list",
                             choices=["list", "new", "show", "diff", "source",
                                      "stage", "unstage", "commit", "log",
                                      "propose", "withdraw", "merge", "delete"],
                             help="list branches / new: open one from a list / show it / "
                                  "diff it against the deck / stage: one card out, one in / "
                                  "source: where every added card comes from / "
                                  "propose: accept it as the deck's next version "
                                  "and wait for the cardboard / withdraw it / "
                                  "merge it into decklist.txt")
            cmd.add_argument("name", nargs="?", default=None, help="the branch name")
            cmd.add_argument("--from", dest="source", default=None,
                             help="new: the candidate list (a file, or `-` for stdin)")
            cmd.add_argument("--why", default=None,
                             help="new: why this branch exists; stage: why this swap")
            # THE SWAP IS THE UNIT. `--out` and `--in` are one edit that already
            # says what it displaced, so `net-change` can name which swaps bought
            # the delta. `in` is a keyword, hence the dest.
            cmd.add_argument("--out", dest="swap_out", default=None, metavar="CARD",
                             help="stage/unstage: the card leaving the list")
            cmd.add_argument("--in", dest="swap_in", default=None, metavar="CARD",
                             help="stage/unstage: the card taking its place")
            cmd.add_argument("--strength", type=float, default=None,
                             help="stage: the `upgrades` strength behind this swap, "
                                  "recorded as provenance and never as a claim")
            cmd.add_argument("--objective", default=None, metavar="EXPR",
                             help='new: what would make this branch worth merging, '
                                  'as `<measure> <op> <number>` — e.g. '
                                  '"kill_by_8 >= 0.30". Required: a branch that '
                                  'cannot be falsified gets graded on whether it '
                                  'did what it does.')
            cmd.add_argument("-m", "--message", default=None,
                             help="commit: why this list and not the last one")
            cmd.add_argument("--write", action="store_true",
                             help="merge: actually write decklist.txt (dry run without it)")
            cmd.add_argument("--force", action="store_true",
                             help="merge: apply despite unsourced cards (needs --reason)")
            cmd.add_argument("--reason", default=None,
                             help="merge: why the sourcing gate is being skipped; "
                                  "propose: why the report is being overridden")
            cmd.add_argument("--as", dest="as_version", default=None,
                             metavar="VERSION",
                             help="propose: the release tag this list is meant to "
                                  "become, e.g. v1.0.2. Required — a proposal that "
                                  "does not say what it intends to become cannot "
                                  "be found to have been outrun by another merge.")
            cmd.add_argument("--ordered", default=None, metavar="NOTE",
                             help="propose: a free-text note about procurement "
                                  '("6 ordered 2026-08-28 via TCGplayer"). A NOTE '
                                  "only: ownership means a box, so nothing here "
                                  "moves the blocker")
            cmd.add_argument("--anyway", action="store_true",
                             help="propose: accept a branch the net change says "
                                  "DO NOT MERGE (needs --reason)")
            cmd.add_argument("--proxy", action="store_true",
                             help="count cards sleeved in your OTHER decks as "
                                  "sourced — you own them, and proxying across "
                                  "your own decks is logistics. Never applies to "
                                  "cards nobody owns")
            cmd.add_argument("--json", action="store_true")
        if name == "deck-state":
            # The pilot's word first, the vocabulary second. `archive` sets
            # `broken-down`, which is what the rest of the repo reads — see
            # `deck_state.ACTIONS`. No action at all means SHOW, so the safe
            # invocation is the short one.
            cmd.add_argument("action", nargs="?", default=None,
                             choices=["archive", "supersede", "retire", "revive"],
                             help="omit to show; archive = broken down for parts")
            cmd.add_argument("--reason", default=None,
                             help="why you did this — a note about a DECISION, "
                                  "never a claim about cardboard")
        if name == "deck-delete":
            cmd.add_argument("--force", action="store_true",
                             help="delete anyway. A deck that was sleeved, "
                                  "played or published is a record — archive it")
        if name == "deck-version":
            cmd.add_argument("action", nargs="?", default="list",
                             choices=["list", "show", "tag", "restore", "paper", "baseline"],
                             help="list versions / show one vs the working list / name one / "
                                  "write one back to decklist.txt (dry run without --write) / "
                                  "paper: mark the version you have SLEEVED (locked) / "
                                  "baseline: restart version numbering at the working list")
            cmd.add_argument("ref", nargs="?", default=None,
                             help="V4, a tag name, or a sha prefix. For `tag`: the new "
                                  "name — vMAJOR.MINOR.PATCH for a release (patch = mana "
                                  "only, minor = the deck can do something new, major = "
                                  "the strategy or the commander changed; every slug "
                                  "starts at v1.0.0), or any word not starting with a "
                                  "digit for a nickname (the-lock)")
            cmd.add_argument("--at", default=None, dest="at",
                             help="tag: the version to name (default: the committed working list)")
            cmd.add_argument("--note", default=None,
                             help="tag/paper: why this list earned a name, or how it was built")
            cmd.add_argument("--built-at", default=None, dest="built_at",
                             help="paper: the date you sleeved it (default: today)")
            cmd.add_argument("--clear", action="store_true",
                             help="paper/baseline: withdraw it")
            cmd.add_argument("--force", action="store_true",
                             help="tag: move a name that already points at another version")
            cmd.add_argument("--full", action="store_true", help="show: print the whole decklist")
            cmd.add_argument("--write", action="store_true",
                             help="restore: actually write decklist.txt (default is a dry run)")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "prescribe":
            cmd.add_argument("prompt", nargs="?", default=None,
                             help="the question; omit (or --list) to list prescriptions")
            cmd.add_argument("--list", action="store_true",
                             help="list this deck's prescriptions")
            cmd.add_argument("--merge", default=None, metavar="ID",
                             help="merge the doctor's (and skeptic's) handoff into prescription ID")
        if name == "experiment":
            cmd.add_argument("--a", dest="a", default=None, metavar="REF",
                             help="arm A: a version (V4 / tag / sha) or `working`")
            cmd.add_argument("--b", dest="b", default=None, metavar="REF",
                             help="arm B: a version ref or `working`")
            cmd.add_argument("--vs", action="append", default=[], metavar="SLUG",
                             help="an opponent seat; repeatable — the same table for both arms")
            cmd.add_argument("--games", type=int, default=None, help="games PER ARM (default SIM_DEFAULT_GAMES)")
            cmd.add_argument("--jobs", type=int, default=None)
            cmd.add_argument("--clock", type=int, default=None)
            cmd.add_argument("--seed", type=int, default=None,
                             help="seed base (default derives from both arms' lists; same seed replays)")
            cmd.add_argument("--profile", default=None, choices=["Default", "Cautious", "Reckless", "Experimental"],
                             help="AI profile for YOUR seat in both arms (opponents stay Default)")
            cmd.add_argument("--list", action="store_true")
            cmd.add_argument("--dry-run", action="store_true", dest="dry_run")
            cmd.add_argument("--analyze", default=None, metavar="EXPERIMENT_ID",
                             help="re-derive BOTH arms' analysis from their kept logs "
                                  "(where the experiment was run)")
        if name == "fetch-opponent":
            cmd.add_argument("commander", nargs="?", default=None,
                             help="commander name or EDHREC slug; omit (or --list) to list the pod")
            cmd.add_argument("--as", dest="as_slug", default=None, help="opponent slug (default: from the commander)")
            cmd.add_argument("--note", default=None, help="why this seat is at your table")
            cmd.add_argument("--list", action="store_true")
        if name == "sim-scenario":
            cmd.add_argument("run", help="the run id (see `simulate <slug> --list`)")
            cmd.add_argument("--game", type=int, required=True, help="1-based game index in the run")
            cmd.add_argument("--turn", type=int, required=True, help="GLOBAL turn number to cut at")
            cmd.add_argument("--step", default=None,
                             help="CR step/phase name to cut at the START of (default: precombat main)")
            cmd.add_argument("--stack", action="store_true",
                             help="write into stacks/NNN-sim-….json (next number) instead of sim/scenarios/")
        if name == "validate-prescription":
            cmd.add_argument("--id", default=None, help="only this prescription; omit for all")
        if name == "deck-facts":
            cmd.add_argument("--out", default=None,
                             help="Write JSON here instead of stdout (a view, never tracked)")
        if name == "validate-deck":
            # One format ships, and the flag exists anyway: it is the seam PRD
            # §13 asks for, and a seam nobody can reach is a seam nobody tests.
            from manamap.pilot.formats import FORMATS
            cmd.add_argument("--format", default=None, choices=sorted(FORMATS),
                             help="rules to validate against (default: commander)")
        if name == "brew":
            cmd.add_argument("slug", help="the new deck's slug (kebab-case)")
            cmd.add_argument("--commander", required=True, help="the commander's name")
            cmd.add_argument("--theme", default=None,
                             help="an EDHREC archetype slug — its role histogram "
                                  "shapes the budget instead of the flat provisional one "
                                  "(`manamap pilot archetypes \"<commander>\"` lists them)")
            cmd.add_argument("--library", nargs="*", default=[],
                             help="cards you are keeping — they become must_include")
            cmd.add_argument("--from", dest="from_file", default=None, metavar="FILE",
                             help="read the library from a file, or a brief.json "
                                  "exported from the Atlas ('-' for stdin)")
            cmd.add_argument("--bracket", type=int, default=None,
                             help="power bracket target (default: the repo default)")
            cmd.add_argument("--build", action="store_true",
                             help="run the builder immediately and write decklist.txt")
        if name == "archetypes":
            cmd.add_argument("commander", help="commander name, e.g. \"Zur the Enchanter\"")
            cmd.add_argument("--theme", default=None,
                             help="derive the role template for this style (its slug)")
            cmd.add_argument("--limit", type=int, default=12,
                             help="how many styles to list (default 12)")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "commander-search":
            # Slugless for the same reason as `card-search`: a search is not
            # per-deck. The seed comes from names, a file, or a deck's own 99.
            cmd.add_argument("cards", nargs="*", default=[],
                             help="seed card names; or use --from / --deck")
            cmd.add_argument("--from", dest="from_file", default=None, metavar="FILE",
                             help="read seed card names from a file, one per line "
                                  "('-' for stdin); a decklist works as-is")
            cmd.add_argument("--deck", default=None,
                             help="seed from a tracked deck's own 99")
            # CHOICES ARE DERIVED, NOT DUPLICATED. This was a hand-copied literal
            # of `commander_search.SPACES.keys()`, so adding or renaming a space
            # left the flag accepting a slug nothing could resolve — or refusing
            # one every other layer accepted. Same mistake `merge-prose` above
            # already documents.
            from manamap import spaces as _spaces
            cmd.add_argument("--space", default="text",
                             choices=_spaces.choices(),
                             help="which embedding to rank in. Default TEXT, because it "
                                  "measures better than the trained space: top-1 0.584 "
                                  "vs 0.410 over 10 held-out draws "
                                  "(`manamap eval-commander-search`)")
            cmd.add_argument("--no-type-control", action="store_true",
                             dest="no_type_control",
                             help="do not match the reference to the seed's type mix "
                                  "(§6.1 step 6); measured as 'does not hurt', not as a win")
            cmd.add_argument("--limit", type=int, default=10,
                             help="how many commanders to print (default 10)")
            cmd.add_argument("--candidates", type=int, default=25, dest="per_identity",
                             help="how many of the identity's top commanders to score "
                                  "(default 25, per §6.1 step 4)")
            cmd.add_argument("--open", type=int, default=None, dest="open_rank",
                             metavar="N",
                             help="write result N's reference deck under data/reference/ "
                                  "and print the Atlas URL that opens it (PRD 6.1 step 9)")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "card-search":
            # NO slug positional: a search is not per-deck. `--deck` scopes it
            # (identity DERIVED from the commander, the deck's own cards excluded)
            # without making the corpus a per-deck artifact.
            cmd.add_argument("--deck", default=None,
                             help="scope to a deck: identity derived from its commander, "
                                  "and its own 99 excluded from the results")
            cmd.add_argument("--identity", default=None,
                             help="colour identity to stay within (e.g. GU); mutually "
                                  "exclusive with --deck, whose identity is derived")
            cmd.add_argument("--include-owned", action="store_true", dest="include_owned",
                             help="--deck: do NOT exclude cards already in the deck")
            cmd.add_argument("--oracle", action="append", default=[], metavar="REGEX",
                             help="oracle-text regex, repeatable (ANY match unless --all)")
            cmd.add_argument("--all", action="store_true", dest="require_all",
                             help="require EVERY --oracle pattern rather than any")
            cmd.add_argument("--name", action="append", default=[], metavar="REGEX",
                             help="card-NAME regex, repeatable (any) — --oracle searches "
                                  "rules text and would not find a card by its name")
            cmd.add_argument("--type", action="append", default=[], metavar="REGEX",
                             help="type-line regex, repeatable (any)")
            cmd.add_argument("--role", action="append", default=[], metavar="ROLE",
                             help="card_roles.json role, repeatable (any)")
            cmd.add_argument("--cmc-max", type=float, default=None, dest="cmc_max")
            cmd.add_argument("--cmc-min", type=float, default=None, dest="cmc_min")
            cmd.add_argument("--no-game-changers", action="store_true", dest="no_game_changers",
                             help="drop Game Changers (they force bracket 4)")
            group = cmd.add_mutually_exclusive_group()
            group.add_argument("--owned", action="store_true",
                               help="only cards the pilot has (a box OR sleeved in a deck)")
            group.add_argument("--unowned", action="store_true",
                               help="only cards the pilot does NOT have — the buy list")
            cmd.add_argument("--limit", type=int, default=None,
                             help=f"max results (default {50})")
            cmd.add_argument("--json", action="store_true", dest="as_json")
            # NOT slug-guarded: the results are corpus rows, not this deck's numbers,
            # even when --deck scoped the query.
            cmd.add_argument("--out", default=None,
                             help="Write JSON here as well (a view, never tracked)")
        if name == "pool-facts":
            # Takes paths, not a slug: a collection is not a deck, and forcing it
            # into data/decks/<slug>/ would put it in reach of validate-deck.
            # nargs="*": with no target it analyses COLLECTION_DIR, which is the
            # question asked nearly every time. Typing the path on every invocation
            # is how the same nine files got parsed by hand ten times in one session.
            cmd.add_argument("targets", nargs="*", default=None,
                             help="Decklist files or directories of them; "
                                  "omit for the pilot's collection (COLLECTION_DIR)")
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
