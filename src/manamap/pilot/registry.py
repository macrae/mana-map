"""Pilot subcommand registry and argparse wiring — every `manamap pilot <cmd>` is here.

Unlike the pipeline STEPS, pilot commands are standalone and per-deck
parameterized. Modules import lazily at dispatch so `manamap --help` stays fast.

MAP OF THIS FILE (in order):
  PILOT_STEPS       (name, module, help) for every subcommand — the ONE table.
                    `docs/pilot.md` must list every name (a test checks), and the
                    count in CLAUDE.md is asserted against its length.
  _DECK_COMMANDS    the commands that take a positional `slug` (everything that
                    works on one deck); a command not named here takes none.
  add_pilot_parser  one subparser per row, then each command's own flags in a long
                    `if name == ...` run — find a command's flags by searching
                    for its name in quotes.
  run_pilot_step    dispatch to `module.main(args)`; a malformed goldfish
                    declaration becomes a refusal (SystemExit), not a traceback.

Adding a command: a row in PILOT_STEPS, the name in _DECK_COMMANDS if it takes a
deck, its flags in add_pilot_parser, a line in docs/pilot.md, and — if it is
read-only and should run warm — `serve.CLI_READONLY`.
"""

import importlib

# (name, dotted module, description)
PILOT_STEPS = [
    ("check-in", "manamap.pilot.check_in", "a paper decklist -> decklist.txt (diff, refuse, apply, re-derive)"),
    ("targeting", "manamap.sim.threat", "who the pod attacks, measured from sim logs (opponent modelling)"),
    ("fetch-deck", "manamap.pilot.fetch_deck", "decklist.txt -> cards.json via Scryfall"),
    ("validate-deck", "manamap.pilot.validate_deck",
     "Check cards.json against the deck's format: size, copies, commander, identity, legality, sideboard"),
    ("download-rules", "manamap.pilot.download_rules", "Download the Comprehensive Rules TXT"),
    ("build-rules-db", "manamap.pilot.build_rules_db", "Chunk + embed the CR into the rules DB"),
    ("query-rules", "manamap.pilot.query_rules", "Semantic top-k rules search"),
    ("lookup-rule", "manamap.pilot.query_rules", "Exact rule fetch by number"),
    ("download-rulings", "manamap.pilot.download_rulings",
     "Download Scryfall's card-rulings dump (INPUT to the resolve loop; never a citation)"),
    ("card-rulings", "manamap.pilot.rulings",
     "Official WotC rulings for named cards, from the local dump (INPUT, never a citation)"),
    ("validate-stack", "manamap.pilot.validate_stack", "Enforce the citation contract on scenarios"),
    ("goldfish", "manamap.pilot.goldfish", "Seeded Monte Carlo resource-development metrics"),
    ("benchmark", "manamap.pilot.benchmark",
     "The standard benchmark: four measures under one frozen configuration"),
    ("scaffold-targets", "manamap.pilot.scaffold_targets",
     "A STARTING goldfish_targets.json, derived and marked as a draft to edit"),
    ("bracket-check", "manamap.pilot.bracket", "Computed bracket floor and its evidence"),
    ("deck-combos", "manamap.pilot.deck_combos",
     "Known Spellbook lines in the list and the one-card near misses (legal, in identity, not banned) — `--write` -> combos.json"),
    ("validate-deck-combos", "manamap.pilot.validate_deck_combos",
     "Form-check combos.json: sha current, every line inside the list, every near miss real/legal/in identity, summary recomputed"),
    ("deck-facts", "manamap.pilot.deck_facts", "Deterministic deck facts agents would else re-derive"),
    ("validate-recon", "manamap.pilot.validate_recon",
     "Form-check deck_recon.json: cards real, legal, in identity; ownership falsified"),
    ("proxies", "manamap.pilot.proxies",
     "Print-ready proxy sheet (63x88 mm, 3x3, crop marks) for cards waiting on cardboard: a branch's adds minus what you have"),
    ("buy-list", "manamap.pilot.buy_list",
     "A branch's purchases (the bill's BUY rows) as one list Mana Pool's mass entry takes — "
     "paste at manapool.com/add-deck; --exact pins printings `N Name (SET) CN`"),
    ("deck-export", "manamap.pilot.deck_export",
     "A deck's list as import text: --format moxfield (Commander:/Deck:/Sideboard:, `(SET) CN`, "
     "`*F*` — paste into Moxfield's import box), arena (no headers, blank line, sideboard) or plain"),
    ("deck-link", "manamap.pilot.deck_link",
     "Record where a deck lives on another site, by hand: `moxfield <url> [--note]`, "
     "`moxfield --remove`, `list` -> links.json"),
    ("validate-links", "manamap.pilot.validate_links",
     "Form-check links.json: known service, https on its host allow-list, id parsed from the url, as_of a date"),
    ("card-search", "manamap.pilot.card_search",
     "Mine the corpus for candidates: colour identity, oracle regex, role, cmc"),
    ("scenario-ab", "manamap.pilot.scenario_ab",
     "A Forge scenario slice A/B from one spec: two arms of one board, paired seed by seed "
     "on one primary measure; --check shows the board for an OK without playing"),
    ("similar-cards", "manamap.pilot.similar_cards",
     "Cards that DO what a named card does: nearest neighbours in the ability space, "
     "identity-derived from --deck, the 99 and illegal cards excluded"),
    ("page-state", "manamap.pilot.page_state",
     "What Sean has open in the browser (page, deck, mode, focused card, selection), "
     "latest per tab, as reported to `manamap serve`; for resolving \"this card\""),
    ("scan-candidates", "manamap.pilot.candidate_scan",
     "One pass over the corpus along a deck's DIMENSIONS (drain, gain, threat, outlet, sweeper, draw): "
     "every row says which predicate found it; converters and two-card infinites FLAGGED, never ranked"),
    ("validate-candidate-scan", "manamap.pilot.validate_candidate_scan",
     "Form-check candidate_scan.json: real, legal, in-identity cards, no Game Changer, every infinite_with a real two-card line"),
    ("fetch-edhrec", "manamap.sim.edhrec",
     "EDHREC's commander page (and --theme pages) as dated per-card synergy/inclusion: edhrec_cards.json"),
    ("validate-edhrec-cards", "manamap.pilot.validate_edhrec_cards",
     "Form-check edhrec_cards.json: names resolve, as_of a date, URLs on EDHREC, synergy in [-1, 1]"),
    ("prices", "manamap.pilot.prices",
     "A list's card prices as DATED evidence (Mana Pool with a token, else Scryfall): prices.json with --write"),
    ("validate-prices", "manamap.pilot.validate_prices",
     "Form-check prices.json: names in the list, cents non-negative ints or null, totals re-add, URLs on manapool/scryfall"),
    ("commander-search", "manamap.pilot.commander_search_cmd",
     "Cards in, commanders out: rank real commanders by proximity to a seed"),
    ("promote", "manamap.pilot.promote",
     "Which ENVIRONMENT a deck is in (dev / bench / sleeved), and every requirement for the next rung"),
    ("demote", "manamap.pilot.promote",
     "Step a deck back down the ladder"),
    ("pods", "manamap.sim.pods",
     "The named tables: which decks, which archetypes, which brackets — and the --vs flags each expands to"),
    ("validate-poh-procedures", "manamap.pilot.validate_poh_procedures",
     "Form-check poh_procedures.json: closed condition vocabulary, the five phases, ordered steps, and grounded_in against the log"),
    ("validate-pilot-policy", "manamap.pilot.validate_pilot_policy",
     "Form-check pilot_policy.json: every rule named, reasoned, and keyed on a channel the simulator computes"),
    ("forge-install", "manamap.sim.forge_pilot",
     "Install the tracked card-script overrides into Forge and VERIFY the engine carries them — the provenance a run record stamps"),
    ("validate-forge-hints", "manamap.pilot.validate_forge_hints",
     "Form-check forge_hints.json: every hinted card is in the 99, says why, and carries a logic or a preference"),
    ("context", "manamap.pilot.deck_context",
     "The Deck Context (CONTEXT.md): print it, --slice SECTION…, --check, --scaffold, "
     "--refresh [--all] the generated blocks, --install the Context Keeper's draft"),
    ("validate-context", "manamap.pilot.validate_context",
     "Gate CONTEXT.md: generated blocks current, every card it describes in the 99, stamp, sections"),
    ("queue", "manamap.pilot.queue",
     "The fleet hypothesis queue (data/queue.jsonl): list [--all] [--deck S], show Q###, "
     "apply DRAFT, rank Q… , kill Q### --reason, decide Q### stage|drop|watch|more"),
    ("validate-queue", "manamap.pilot.validate_queue",
     "Gate data/queue.jsonl: every line replayed through the queue's transition check"),
    ("watch", "manamap.pilot.watchlist",
     "Candidate watch lists (watchlist.json): list, mark SET \"Card\" watching|pass|unreviewed [--note], "
     "note SET \"Card\" \"…\" — reviewed in the Atlas's Build mode"),
    ("validate-watchlist", "manamap.pilot.validate_watchlist",
     "Gate watchlist.json: real, legal, in-identity cards; closed vocabularies; sources that exist"),
    ("sla-report", "manamap.pilot.sla_report",
     "Sub-agents against their response-time targets: runs, median, slowest, missed (from the job band's log)"),
    ("validate-protected", "manamap.pilot.validate_protected",
     "Form-check protected.json: the pilot's keep list — every card in the 99, not the commander, with a why"),
    ("forge-cast-check", "manamap.sim.cast_check",
     "PROVE the Forge AI will cast/activate a card before a branch depends on it: a short two-seat shell, "
     "drawn / cast / activated / held-while-castable counted. Unflagged is not castable (Toxic Deluge: 28 drawn, 0 cast)"),
    ("validate-cast-proofs", "manamap.pilot.validate_cast_proofs",
     "Form-check a branch's cast_proofs.json: the harness stamp, every row's counts and a verdict in the vocabulary"),
    ("forge-telemetry", "manamap.sim.telemetry",
     "The patched log formatter (every zone change, by name and owner): what the repo declares, what the jar carries, --build to compile it"),
    ("metrics", "manamap.metrics",
     "The metrics catalog: one definition per figure, which engine answers it, and what is unavailable and why"),
    ("build", "manamap.pilot.autobuild",
     "A brief -> a legal, measured 99 on the bench: intent, anchor, build, "
     "resolve, measure, land"),
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
    ("validate-poh", "manamap.pilot.validate_poh",
     "Form-check a rendered handbook: dangling xrefs, callout cap, no script, no build date"),
    ("build-poh", "manamap.pilot.poh",
     "the Pilot's Operating Handbook — numbered, printable, emergencies first"),
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
    ("sim-findings", "manamap.pilot.pilot_findings",
     "THE SIM DEBRIEF'S SKELETON: what each run says about how the deck was flown, as findings "
     "with ids, intervals and sources — rate vs the null, piloting, how we lost, what was held, "
     "first attack, wipes, targeting; `--boards` adds recurring board shapes where the logs are"),
    ("validate-sim-findings", "manamap.pilot.validate_sim_findings",
     "Every citation in the prose is a finding of that run and every number in it is the record's"),
    ("merge-sim-findings", "manamap.pilot.merge_sim_findings",
     "Recompute the skeleton and take the sim-debrief agent's PROSE, per run, whitelisted"),
    ("sim-boards", "manamap.sim.boards",
     "Which moments in a run are worth lifting, and how often they RECUR: a criterion "
     "(widest / modal / death / held / pre-wipe / lethal-missed / first-attack) names a cut "
     "and a shape; the shortlist is ranked by games reaching that shape, with a Wilson "
     "interval; --lift hands the exemplar to sim-scenario with the finder's provenance"),
    ("validate-lift", "manamap.pilot.validate_lift",
     "A committed lifted scenario against a fresh lift of the same cut: FAIL while "
     "unresolved, a NOTE once checker-passed; the bridge version where the logs are absent"),
    ("sim-scenario", "manamap.sim.bridge",
     "Lift one game at one moment out of a Forge run into a game_state v2 scenario for resolve-stack"),
    ("decisions", "manamap.pilot.decisions",
     "THE DECISION LEDGER (decisions.jsonl, append-only): every propose / withdraw / reject / "
     "merge with the prediction frozen at that moment; `outcome` joins a merged list's own "
     "runs to it; `backfill` seeds it from branch.json"),
    ("validate-decisions", "manamap.pilot.validate_decisions",
     "Form-check decisions.jsonl: sequential ids, the closed kind vocabulary, reasons, "
     "outcomes that name a merge"),
    ("campaign", "manamap.sim.campaign",
     "A PRE-REGISTERED queue of Forge A/Bs (data/campaigns/<name>.json): plan pins the arms "
     "and prepends an A/A per harness, run measures overnight and never merges, status derives "
     "each entry's state from its record"),
    ("sim-progress", "manamap.sim.progress",
     "What a RUNNING simulation has done so far: a bar, the rate, an ETA, "
     "the estimate with its interval, and how far the interval still has to shrink"),
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
    ("try", "manamap.pilot.try_swap",
     "A swap idea to an answer in under two minutes: the cards, the keep list, the mana, "
     "the goldfish rows and one line — nothing written unless --stage"),
    ("edit", "manamap.pilot.deck_edit",
     "Edit a bench or brewing deck IN PLACE: --add/--cut/--swap OUT=IN/--set N=Q, validated as one "
     "change, journalled for undo|redo|history, then rebuilt (a sleeved deck is refused: branch it)"),
    ("save-version", "manamap.pilot.deck_edit",
     "Commit a bench deck's accumulated edits as ONE git version with a note — only the deck's "
     "own paths (git commit --only), after merge's consistency pass"),
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
    ("validate-brief", "manamap.pilot.validate_brief",
     "Form-check brief.json: the commander real and legal, every named card in the corpus and in identity, the pools on disk"),
    ("validate-build", "manamap.pilot.validate_build", "Form-check a build plan against the contract"),
    ("validate-pending", "manamap.pilot.validate_pending",
     "The queue of changes decided but not applied; closure is DERIVED from the deck"),
    ("validate-diagnosis", "manamap.pilot.validate_diagnosis", "Form-check a deck diagnosis (axes re-derived, cuts checked against verified stacks)"),
    ("validate-goldfish-targets", "manamap.pilot.validate_goldfish_targets", "Form-check the engine declaration goldfish and deck-audit price"),
    ("diagnosis-report", "manamap.pilot.diagnosis_report", "Render a deck diagnosis as readable markdown"),
    ("validate-tutor-guide", "manamap.pilot.validate_tutor_guide", "Form-check the tutor guide (one wish per tutor)"),
    ("validate-strategic-frame", "manamap.pilot.validate_strategic_frame", "Form-check a strategic frame"),
    ("build-index", "manamap.pilot.deck_manifest", "Write data/decks/index.json — the deck list + manifest the frontend fetches"),
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
    # The two corpora built from this repo's own working tree. `docs` is the
    # "why did we do it this way" index — 7,500 lines of measurements that were
    # previously findable only by remembering they existed.
    ("query-docs", "manamap.pilot.retrieve", "Semantic search over this repo's docs"),
    ("lookup-doc", "manamap.pilot.retrieve", "Exact doc chunk fetch by id"),
    ("query-code", "manamap.pilot.retrieve", "Semantic search over this repo's source"),
    ("build-docs-db", "manamap.pilot.build_corpus", "Index docs/ for semantic search"),
    ("build-code-db", "manamap.pilot.build_corpus", "Index src/ for semantic search"),
]

_DECK_COMMANDS = {
    "context", "validate-context", "watch", "validate-watchlist",
    "validate-poh-procedures", "validate-pilot-policy", "validate-forge-hints", "validate-protected",
    "scan-candidates", "validate-candidate-scan", "fetch-edhrec", "validate-edhrec-cards",
    "prices", "validate-prices",
    "deck-export", "deck-link", "validate-links",
    "forge-cast-check", "validate-cast-proofs",
    "decisions", "validate-decisions",
    "check-in", "targeting", "fetch-deck", "validate-deck", "validate-stack", "goldfish",
    "cache-status", "cache-record", "cache-clear", "cache-rebless",
    "cache-snapshot", "cache-rerecord",
    "artist-credits",
    "model-coverage",
    "bracket-check", "deck-combos", "validate-deck-combos", "build-deck", "validate-build", "deck-facts", "deck-audit", "deck-map", "deck-status", "engine-facts", "validate-engine", "merge-deck-map", "validate-deck-map", "deck-history",
    "mana-analysis", "validate-strategic-frame", "scaffold-targets",
    "validate-diagnosis", "validate-goldfish-targets",
    "validate-recon",
    "diagnosis-report",
    "validate-tutor-guide", "impact", "scenario-facts", "merge-prose",
    "card-value", "validate-pending",
    "deck-notes", "validate-debrief", "merge-debrief", "prescribe", "validate-prescription",
    "captains-log", "merge-captains-log", "validate-captains-log",
    "install-agent", "build-poh", "validate-poh",
    "deck-version", "deck-state", "deck-delete", "validate-deck-versions",
    "validate-log-causes",
    "build", "validate-brief", "promote", "demote",
    "deck-branch", "buy-list", "diagnose", "assess", "candidates", "close",
    "upgrades", "mana-fit",
    "validate-diagnostic", "try", "edit", "save-version", "net-change", "validate-net-change", "validate-branch", "deck-info", "simulate", "validate-sim", "sim-progress", "sim-scenario", "experiment",
    "sim-boards", "validate-lift", "sim-findings", "validate-sim-findings", "merge-sim-findings",
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
            nargs = "?" if name in ("deck-status", "context") else None
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
        if name == "download-rulings":
            cmd.add_argument("--force", action="store_true",
                             help="Re-download and rewrite even when the catalog stamp "
                                  "and the content sha both match")
        if name == "card-rulings":
            # Slugless like `card-search`: a ruling is a fact about a card, not a deck.
            cmd.add_argument("names", nargs="+", metavar="NAME",
                             help="Card name(s), either face of a DFC accepted")
            cmd.add_argument("--json", action="store_true", dest="as_json")
            cmd.add_argument("--all-sources", action="store_true", dest="all_sources",
                             help="Include Scryfall's own editorial notes (default: WotC only)")
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
        if name in ("query-docs", "query-code"):
            cmd.add_argument("query", help="Natural-language question")
            cmd.add_argument("--k", type=int, default=None, help="Number of results")
            cmd.add_argument("--full", action="store_true",
                             help="print each hit's whole passage, not one line")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "lookup-doc":
            cmd.add_argument("chunk_id", help="Exact chunk id, e.g. docs/vision.md#the-bench")
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
            cmd.add_argument("--kind", default="pilot",
                             help="which log to merge into (ship; personal is reserved)")
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
        if name in ("build-deck", "build"):
            cmd.add_argument("--overwrite", action="store_true",
                             help="rebuild the list of a deck that already has "
                                  "cards.json, keeping the old one as "
                                  "decklist.txt.bak. Refused for a SLEEVED or "
                                  "archived deck regardless")
        if name == "deck-combos":
            cmd.add_argument("--json", action="store_true", dest="as_json")
            cmd.add_argument("--write", action="store_true",
                             help="write combos.json beside the list (the deck's, or the branch's)")
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
            # `--from` is optional only because `--set-printing` is the other
            # way in; `check_in.main` refuses a call that gives neither.
            cmd.add_argument("--from", dest="source", default=None,
                             help="the paper decklist: a file, or - for stdin")
            cmd.add_argument("--set-printing", nargs=2, metavar=("NAME", "PRINTING"),
                             dest="set_printing", default=None,
                             help="point ONE line at an exact printing: "
                                  "--set-printing \"Sol Ring\" \"(SLD) 1234\"; writes the "
                                  "line and runs the chain, no new version, no agents")
            cmd.add_argument("--foil", action="store_true",
                             help="with --set-printing: the pilot's copy is foil (*F*)")
            cmd.add_argument("--branch", default=None, metavar="NAME",
                             help="with --set-printing: write the branch's list, not the deck's")
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
        if name in ("fetch-deck", "check-in"):
            # The format's home is `brief.json` (check-in writes it there on
            # --write; fetch-deck reads it from there), so the flag is a
            # declaration made once, not one repeated on every fetch. Absent
            # means Commander, and a Commander deck's cards.json carries no key.
            from manamap.pilot.formats import FORMATS
            cmd.add_argument("--format", default=None, choices=sorted(FORMATS),
                             help="the deck's format (default: what brief.json / "
                                  "cards.json already say, else commander)")
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
        if name == "sim-progress":
            cmd.add_argument("run", nargs="?", default=None,
                             help="a run id or any substring of one (default: the "
                                  "most recently touched)")
            cmd.add_argument("--all", action="store_true",
                             help="every run with logs, not just the newest")
        if name == "simulate":
            cmd.add_argument("--vs", action="append", default=[], metavar="SLUG",
                             help="an opponent seat (data/opponents/<slug> or data/decks/<slug>); repeatable")
            cmd.add_argument("--games", type=int, default=None,
                             help="number of games (default SIM_DEFAULT_GAMES)")
            cmd.add_argument("--jobs", type=int, default=None,
                             help="JVMs to run in parallel (default: performance cores, "
                                  "i.e. forge.default_jobs())")
            cmd.add_argument("--clock", type=int, default=None,
                             help="seconds before Forge calls a game a draw (default SIM_GAME_CLOCK_SECONDS)")
            cmd.add_argument("--pod", default=None, metavar="NAME",
                             help="a named table from data/pods/ instead of "
                                  "repeating --vs. It expands to the SAME "
                                  "ordered slugs and therefore the same run id, "
                                  "and it carries each seat's archetype, bracket "
                                  "and AI profile (`manamap pilot pods`)")
            cmd.add_argument("--list", action="store_true", help="list this deck's runs")
            cmd.add_argument("--dry-run", action="store_true", dest="dry_run",
                             help="print the JVM commands and the run id; run nothing")
            cmd.add_argument("--seed", type=int, default=None,
                             help="RNG seed (default derives from the configuration, so the default REPLAYS; pass one for a new sample)")
            cmd.add_argument("--vs-profile", default=None, dest="vs_profile",
                             metavar="P",
                             help="AI profile for every OPPONENT seat (default: Experimental, "
                                  "the standard pod since 2026-08-30). "
                                  "The pod is part of the instrument: a win rate is "
                                  "relative to how well the table plays.")
            cmd.add_argument("--profile", default=None, metavar="P",
                             help="AI profile for YOUR seat — Forge's four, or any *.ai in res/ai/ (a per-deck mm-<slug> is compiled from pilot_policy.json). Forge validates the name and lists what it has; argparse was the only thing refusing a deck profile — the opponents' is "
                                  "--vs-profile (measured: aggro profiles make a "
                                  "hold-up deck worse, on six games, which this "
                                  "repo's own power function calls undetectable)")
            cmd.add_argument("--force", action="store_true",
                             help="replay an existing run id (it writes the same bytes)")
            cmd.add_argument("--analyze", default=None, metavar="RUN_ID",
                             help="re-derive a run's analysis from its kept logs (where the run was made)")
            cmd.add_argument("--detect", type=float, default=None, metavar="DELTA",
                             help="the change in win rate you want this run to be able to see "
                                  "against the pod's null; the run is REFUSED when its power "
                                  "for that is under 0.8 (see --anyway)")
            cmd.add_argument("--anyway", action="store_true",
                             help="run an underpowered --detect as a screen rather than a test")
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
        if name in ('fetch-deck', 'bracket-check', 'deck-combos', 'validate-deck-combos', 'mana-analysis', 'goldfish', 'deck-facts', 'deck-audit', 'deck-map', 'diagnose', 'candidates', 'assess', 'close', 'upgrades', 'mana-fit',
                    # The validators too: a branch's artifacts were gated by
                    # NOTHING, because no validator could be pointed at one.
                    'validate-deck', 'validate-deck-map',
                    'validate-goldfish-targets', 'validate-diagnostic',
                    'net-change', 'validate-net-change', 'validate-branch', 'validate-cast-proofs'):
            cmd.add_argument("--branch", default=None, metavar="NAME",
                             help="run against a branch (see `deck-branch <slug> list`) "
                                  "instead of the deck's own list")
        if name == "try":
            cmd.add_argument("--out", action="append", metavar="CARD",
                             help="a card leaving the list (repeat; pairs with --in in order)")
            cmd.add_argument("--in", dest="in_", action="append", metavar="CARD",
                             help="the card taking its place (repeat)")
            cmd.add_argument("--branch", default=None, metavar="NAME",
                             help="try the swaps on a branch's list instead of the deck's")
            cmd.add_argument("--each", action="store_true",
                             help="also measure every swap alone, to see which one moves what")
            cmd.add_argument("--stage", default=None, metavar="NAME",
                             help="after the screen, write the swaps to this branch (opened if new)")
            cmd.add_argument("--add", action="append", metavar="CARD",
                             help="a card going in with nothing coming out (repeat) — the same "
                                  "rules `edit` applies, so a lone add to a Commander 100 is refused")
            cmd.add_argument("--cut", action="append", metavar="CARD",
                             help="a card coming out with nothing going in (repeat)")
            cmd.add_argument("--set", action="append", metavar="NAME=COPIES",
                             help="set a card's copy count (0 cuts it)")
            cmd.add_argument("--side", action="store_true",
                             help="--add/--cut/--set act on the sideboard (60-card formats)")
            cmd.add_argument("--iterations", type=int, default=None)
            cmd.add_argument("--seed", type=int, default=None)
            cmd.add_argument("--json", action="store_true")
        if name == "edit":
            cmd.add_argument("action", nargs="?", default=None,
                             choices=("undo", "redo", "history"),
                             help="undo / redo the last change, or list the journal")
            cmd.add_argument("--add", action="append", metavar="CARD",
                             help="one copy in (repeat)")
            cmd.add_argument("--cut", action="append", metavar="CARD",
                             help="one copy out (repeat)")
            cmd.add_argument("--swap", action="append", metavar="OUT=IN",
                             help="one copy of OUT out, one of IN in (repeat)")
            cmd.add_argument("--set", action="append", metavar="NAME=COPIES",
                             help="a card's copy count (0 cuts it)")
            cmd.add_argument("--side", action="store_true",
                             help="the ops act on the sideboard (60-card formats only)")
            cmd.add_argument("--note", default=None, help="why — kept in the journal")
            cmd.add_argument("--dry-run", action="store_true", dest="dry_run",
                             help="validate and show the change; write nothing")
            cmd.add_argument("--no-chain", action="store_true", dest="no_chain",
                             help="write the list only; skip the rebuild")
            cmd.add_argument("--rebuild", action="store_true",
                             help="re-run the rebuild (alone: after an offline edit)")
            cmd.add_argument("--json", action="store_true")
        if name == "save-version":
            cmd.add_argument("--note", required=True,
                             help="the version's subject line: what changed and why")
            cmd.add_argument("--json", action="store_true")
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
                                      "propose", "withdraw", "reject", "merge", "delete"],
                             help="list branches / new: open one from a list / show it / "
                                  "diff it against the deck / stage: one card out, one in / "
                                  "source: where every added card comes from / "
                                  "propose: accept it as the deck's next version "
                                  "and wait for the cardboard / withdraw it / "
                                  "merge it into decklist.txt")
            cmd.add_argument("name", nargs="?", default=None, help="the branch name")
            cmd.add_argument("--from", dest="source", default=None,
                             help="new: the candidate list (a file, or `-` for stdin); default: the deck's own decklist.txt")
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
        if name == "watch":
            cmd.add_argument("verb", nargs="?", default="list", choices=["list", "mark", "note"])
            cmd.add_argument("rest", nargs="*", help="SET \"Card\" verdict, or SET \"Card\" \"note\"")
            cmd.add_argument("--note", default=None, help="mark: a note alongside the verdict")
        if name == "queue":
            cmd.add_argument("verb", nargs="?", default="list",
                             choices=["list", "waiting", "show", "add", "apply", "rank", "kill", "decide"])
            cmd.add_argument("rest", nargs="*", help="item ids, a draft path, or a decision")
            cmd.add_argument("--all", action="store_true", help="list: include closed and expired items")
            cmd.add_argument("--deck", default=None, help="list/waiting: one deck only; add: the deck it is about")
            cmd.add_argument("--json", action="store_true")
            # add: Sean's own claim. The same fields a pod hypothesis needs, so `check`
            # can hold it to the same rule; it still faces the Challenger.
            cmd.add_argument("--claim", default=None, help="add: the claim, specific enough to be wrong")
            cmd.add_argument("--expect", default=None, help="add: its expected effect")
            cmd.add_argument("--method", default=None,
                             help="add: the lightest test — context, argument, data, try, rules or strategy")
            cmd.add_argument("--how", default=None, help="add: how that method would settle it")
            cmd.add_argument("--why", default=None, help="add: where it came from (optional)")
            cmd.add_argument("--reason", default=None, help="kill: why")
            cmd.add_argument("--note", default=None, help="decide: a note on the call")
        if name == "context":
            cmd.add_argument("--slice", nargs="+", default=None, metavar="SECTION",
                             help="print only these sections (summary plays cards numbers "
                                  "pilot history questions changelog) — what an agent is sent")
            cmd.add_argument("--check", action="store_true", help="the gate; exit 1 on an error")
            cmd.add_argument("--scaffold", action="store_true",
                             help="write a fresh CONTEXT.md: generated blocks, prose placeholders")
            cmd.add_argument("--force", action="store_true", help="with --scaffold: overwrite")
            cmd.add_argument("--refresh", action="store_true",
                             help="rewrite the generated blocks only; prose kept byte for byte")
            cmd.add_argument("--all", action="store_true",
                             help="with --refresh/--check: every live deck that has a CONTEXT.md")
            cmd.add_argument("--install", default=None, metavar="DRAFT",
                             help="install the Context Keeper's draft through the strict gate")
            cmd.add_argument("--note", default=None,
                             help="with --install: one line for the Context changelog")
            cmd.add_argument("--who", default=None,
                             help="with --install: who wrote the pass (default Keeper)")
        if name == "deck-version":
            cmd.add_argument("action", nargs="?", default="list",
                             choices=["list", "show", "tag", "restore", "paper", "baseline"],
                             help="list versions / show one vs the working list / name one / "
                                  "write one back to decklist.txt (dry run without --write) / "
                                  "paper: mark the version you have SLEEVED (locked) / "
                                  "baseline: restart version numbering at the working list")
            cmd.add_argument("ref", nargs="?", default=None,
                             help="V4, a tag name, or a sha prefix. For `tag`: the new "
                                  "name — vMAJOR.MINOR.PATCH for a release "
                                  "(PATCH = a mana fix or a SINGLE-CARD swap; "
                                  "MINOR = a sizable change, several cards or a "
                                  "new capability; MAJOR = the strategy or the "
                                  "commander changed; every slug starts at "
                                  "v1.0.0), or any word not starting with a "
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
                             help="restore: actually write decklist.txt (default is a dry "
                                  "run), keeping decklist.txt.bak and running the check-in "
                                  "chain. Refused for an archived deck, a protected cut, or "
                                  "a sleeved deck unless the target is its paper version")
            cmd.add_argument("--no-chain", action="store_true", dest="no_chain",
                             help="restore: write the list but skip fetch-deck / goldfish "
                                  "/ mana-analysis (no corpus on this machine)")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "prescribe":
            cmd.add_argument("prompt", nargs="?", default=None,
                             help="the question; omit (or --list) to list prescriptions")
            cmd.add_argument("--list", action="store_true",
                             help="list this deck's prescriptions")
            cmd.add_argument("--merge", default=None, metavar="ID",
                             help="merge the doctor's (and skeptic's) handoff into prescription ID")
        if name == "experiment":
            cmd.add_argument("--detect", type=float, default=None, metavar="DELTA",
                             help="the win-rate change you care about, e.g. 0.10. "
                                  "The power preflight then says whether this run "
                                  "can see it and how many games it would take — "
                                  "before four hours are spent finding out")
            cmd.add_argument("--anyway", action="store_true",
                             help="run an underpowered --detect as a screen rather than a test")
            cmd.add_argument("--looks", type=int, default=1, metavar="K",
                             help="a group-sequential design with K equally spaced looks "
                                  "(1-4, O'Brien-Fleming): each look is whole rotated jobs "
                                  "from both arms, tested at its own boundary; the record "
                                  "is rewritten after every look and resumes with --resume")
            cmd.add_argument("--until-mde", type=float, default=None, dest="until_mde",
                             metavar="X", help="non-binding futility: stop a look whose "
                                               "boundary interval already excludes +X")
            cmd.add_argument("--boundary", choices=["obf", "pocock"], default="obf",
                             help="the boundary scheme (default O'Brien-Fleming)")
            cmd.add_argument("--aa", action="store_true",
                             help="an A/A: one list twice, arm B on a second seed base — "
                                  "the harness's noise floor, said out loud")
            cmd.add_argument("--resume", action="store_true",
                             help="continue an unfinished sequential run of the SAME "
                                  "command line from its next look")
            cmd.add_argument("--profile-b", dest="profile_b", default=None, metavar="P",
                             help="fly OUR seat on this AI profile on arm B only (policy-on "
                                  "vs policy-off on one list); arm A keeps --profile")
            cmd.add_argument("--a", dest="a", default=None, metavar="REF",
                             help="arm A: a version (V4 / tag / sha) or `working`")
            cmd.add_argument("--b", dest="b", default=None, metavar="REF",
                             help="arm B: a version ref or `working`")
            cmd.add_argument("--vs", action="append", default=[], metavar="SLUG",
                             help="an opponent seat; repeatable — the same table for both arms")
            cmd.add_argument("--pod", default=None, metavar="NAME",
                             help="a named table from data/pods/ instead of "
                                  "repeating --vs. It expands to the SAME "
                                  "ordered slugs and therefore the same run id, "
                                  "and it carries each seat's archetype, bracket "
                                  "and AI profile (`manamap pilot pods`)")
            cmd.add_argument("--games", type=int, default=None, help="games PER ARM (default SIM_DEFAULT_GAMES)")
            cmd.add_argument("--jobs", type=int, default=None)
            cmd.add_argument("--clock", type=int, default=None)
            cmd.add_argument("--seed", type=int, default=None,
                             help="seed base (default derives from both arms' lists; same seed replays)")
            cmd.add_argument("--profile", default=None, metavar="P",
                             help="AI profile for YOUR seat in BOTH arms")
            cmd.add_argument("--vs-profile", default=None, dest="vs_profile",
                             metavar="P",
                             help="AI profile for every OPPONENT seat in both "
                                  "arms (default: Experimental, the same pod "
                                  "`simulate` has used since 2026-08-30). This "
                                  "built the pod on Default and never read the "
                                  "standard, so the two commands measured "
                                  "different populations; pass Default to "
                                  "reproduce an experiment made before that fix")
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
        if name == "sim-findings":
            cmd.add_argument("--run", default=None, metavar="RUN_ID", help="just this run")
            cmd.add_argument("--boards", action="store_true",
                             help="add recurring board shapes (held / death / widest) from the "
                                  "finder — needs the logs, so only where the run was made")
            cmd.add_argument("--write", action="store_true",
                             help="write sim_findings.json (prose carried forward from the tracked file)")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "sim-boards":
            cmd.add_argument("run", help="the run id (see `simulate <slug> --list`)")
            cmd.add_argument("--criterion", required=True,
                             choices=["widest", "modal", "death", "held", "pre-wipe",
                                      "lethal-missed", "first-attack"],
                             help="which moment: the widest board / the modal board over own "
                                  "turns 4-10 / the last main before elimination / a cleanup "
                                  "with N+ lands untapped and nothing cast / the start of a "
                                  "wipe turn / an opponent at or under our untapped printed "
                                  "power with no attack / the first attack")
            cmd.add_argument("--n", type=int, default=None, help="for `held`: lands untapped (default 4)")
            cmd.add_argument("--top", type=int, default=1, help="how many shapes to lift")
            cmd.add_argument("--lift", action="store_true",
                             help="lift the top shapes' exemplars into sim/scenarios/ (gitignored)")
            cmd.add_argument("--stack", action="store_true",
                             help="lift into stacks/NNN-sim-….json (tracked; the resolve loop's input)")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "validate-lift":
            cmd.add_argument("--stack", default=None, metavar="NNN", help="only this stack")
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
            # One format ships, and the flag exists anyway: it is the seam PRD-v1
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
        if name == "validate-brief":
            cmd.add_argument("--themes", action="store_true",
                             help="also resolve `theme` against the commander's "
                                  "real EDHREC archetypes. Off by default: a "
                                  "gate that fails when the network is down is "
                                  "a gate that gets switched off")
        if name in ("promote", "demote"):
            cmd.add_argument("--to", default=None, choices=["dev", "bench", "sleeved"],
                             help="the rung to move to (default: one step). A "
                                  "promotion may not SKIP a rung — the gate for "
                                  "the one it skipped would never run")
            cmd.add_argument("--show", action="store_true",
                             help="report the gate and change nothing")
            cmd.add_argument("--force", action="store_true",
                             help="promote past an unmet requirement (needs --reason)")
            cmd.add_argument("--reason", default=None,
                             help="why a gate was waived — a waiver with no "
                                  "reason is a gate nobody will trust")
        if name == "forge-install":
            cmd.add_argument("--verify", action="store_true",
                             help="report what the engine carries against what the "
                                  "repo declares, and change nothing. This is the "
                                  "question a run record's `card_overrides.agrees` "
                                  "answers, asked on demand")
            cmd.add_argument("--revert", action="store_true",
                             help="restore Forge's pristine card scripts. Every "
                                  "install rebuilds from that copy, so this is also "
                                  "what makes installing twice identical to once")
            cmd.add_argument("--generate", action="store_true",
                             help="re-derive the override scripts from the shipped "
                                  "ones. Refuses any card whose targeting line is "
                                  "SP$ CopyPermanent, because CopyPermanentAi never "
                                  "reads AITgts$ — a hint the engine cannot act on")
        if name == "forge-telemetry":
            cmd.add_argument("--build", action="store_true",
                             help="compile data/forge_patches/GameLogFormatter.java "
                                  "against the pristine jar and write the patched copy "
                                  "beside it (needs javac 21+). A new class sha is "
                                  "registered in the manifest — commit it")
        if name == "decisions":
            cmd.add_argument("action", nargs="?", default="list",
                             choices=["list", "show", "outcome", "backfill", "adopt-policy"],
                             help="list the ledger / show one line / outcome: close every "
                                  "merge that has runs of the merged list at its pod / "
                                  "backfill: seed from branch.json / adopt-policy: record a "
                                  "piloting rule accepted from an experiment")
            cmd.add_argument("entry_id", nargs="?", default=None, help="for `show`")
            cmd.add_argument("--from-experiment", dest="from_experiment", default=None,
                             metavar="EID", help="for `adopt-policy`")
            cmd.add_argument("--reason", default=None)
            cmd.add_argument("--dry-run", action="store_true", dest="dry_run",
                             help="outcome: compute without appending")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "campaign":
            cmd.add_argument("name", nargs="?", default=None,
                             help="a campaign under data/campaigns/; omit to list them")
            cmd.add_argument("action", nargs="?", default="status",
                             choices=["plan", "run", "status"],
                             help="plan: resolve refs to shas, preflight, prepend an A/A, "
                                  "write `resolved`; run: in order, skip DONE and STALE, "
                                  "resume RUNNING, never merge; status: the derived states")
            cmd.add_argument("--only", action="append", default=None, metavar="ID",
                             help="run just this entry (repeatable)")
            cmd.add_argument("--dry-run", action="store_true", dest="dry_run",
                             help="plan without writing; run without starting Forge")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "pods":
            cmd.add_argument("name", nargs="?", default=None,
                             help="one pod; omit to list them all")
            cmd.add_argument("--calibration", action="store_true",
                             help="how this table actually divides its wins, "
                                  "from every tracked run that faced it — and "
                                  "therefore what the NULL is. A four-player "
                                  "win rate reads against 0.25 unless something "
                                  "says otherwise, and `standard` gives one seat "
                                  "0.572 and another 0.052")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "metrics":
            from manamap.metrics import GROUPS, STATUSES
            cmd.add_argument("--group", default=None, choices=sorted(GROUPS))
            cmd.add_argument("--status", default=None, choices=sorted(STATUSES),
                             help="published / opt_in / derivable / unavailable")
            cmd.add_argument("--verbose", "-v", action="store_true",
                             help="the definition, the source, and the caveat")
            cmd.add_argument("--problems", action="store_true",
                             help="PRD §2's six observed pod-night problems, "
                                  "with today's answer beside each")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "build":
            # The same inputs as `brew`, deliberately: `build` is `brew --build`
            # plus the four stages that come after it, and two commands that
            # take a library must not disagree about how a library is spelled.
            cmd.add_argument("--brief", default=None, metavar="TEXT",
                             help="what the deck should do, in a sentence. TWO "
                                  "things are read out of it — a `bracket N`, "
                                  "and a style, but only by matching against "
                                  "the commander's REAL EDHREC archetypes. The "
                                  "rest is stored and consumed by nothing, and "
                                  "the report says so")
            cmd.add_argument("--commander", default=None,
                             help="the commander's name. Omit it and three are "
                                  "proposed from --library/--from, then the "
                                  "build halts for you to pick one")
            cmd.add_argument("--partner", default=None,
                             help="the second commander of a Partner pair; identity is "
                                  "the union and the 99 becomes 98")
            cmd.add_argument("--theme", default=None,
                             help="an EDHREC archetype slug, overriding whatever "
                                  "--brief would have matched "
                                  "(`manamap pilot archetypes \"<commander>\"`)")
            cmd.add_argument("--library", nargs="*", default=[],
                             help="cards you are keeping — they become must_include")
            cmd.add_argument("--from", dest="from_file", default=None, metavar="FILE",
                             help="read the library from a file, or a brief.json "
                                  "exported from the Atlas ('-' for stdin)")
            cmd.add_argument("--bracket", type=int, default=None,
                             help="power bracket target. A HARD CONSTRAINT: the "
                                  "builder cuts and replaces to reach it and "
                                  "refuses if it cannot, rather than shipping a "
                                  "flagged overage")
            cmd.add_argument("--json", action="store_true", dest="as_json")
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
                                  "and print the Atlas URL that opens it (PRD-v1 §6.1 step 9)")
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
            # WHAT THE GOLDFISH CAN PRICE. A search that cannot answer this
            # hands you a card the model reads as a vanilla body, which measures
            # as nothing and looks exactly like a card that does not help.
            cmd.add_argument("--channel", action="append", default=[], metavar="CHANNEL",
                             help="goldfish channel the card feeds, repeatable (any): "
                                  "fodder / pump / draw / storm / per-cast-damage / "
                                  "magecraft")
            grp = cmd.add_mutually_exclusive_group()
            grp.add_argument("--modelled", action="store_false", dest="unmodelled",
                             default=None,
                             help="only cards the goldfish can price at all")
            grp.add_argument("--unmodelled", action="store_true", dest="unmodelled",
                             default=None,
                             help="only cards it CANNOT — the blind spots")
            cmd.add_argument("--cmc-max", type=float, default=None, dest="cmc_max")
            cmd.add_argument("--cmc-min", type=float, default=None, dest="cmc_min")
            cmd.add_argument("--set", action="append", default=[], metavar="CODE",
                             help="Scryfall set code of the CORPUS printing, repeatable "
                                  "(any) — so `--set fra` includes FRA's reprints")
            cmd.add_argument("--released-after", default=None, metavar="DATE",
                             dest="released_after",
                             help="first printed ON OR AFTER this date (YYYY[-MM[-DD]]); "
                                  "the card's FIRST printing, not the corpus printing")
            cmd.add_argument("--released-before", default=None, metavar="DATE",
                             dest="released_before",
                             help="first printed ON OR BEFORE this date (YYYY[-MM[-DD]])")
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
        if name == "scenario-ab":
            # NO slug positional: the spec names every seat's deck, ours included.
            cmd.add_argument("--spec", required=True,
                             help="the scenario spec JSON (the scenario-sim agent writes it)")
            cmd.add_argument("--check", action="store_true",
                             help="convert and show each arm's board per seat; play nothing")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "page-state":
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "similar-cards":
            # NO slug positional, like card-search: the positionals are the SEED cards,
            # and --deck only scopes the corpus (identity derived, the 99 excluded).
            cmd.add_argument("cards", nargs="*", metavar="CARD",
                             help="one or more seed cards; several rank against their centroid")
            cmd.add_argument("--deck", default=None,
                             help="scope to a deck: identity derived from its commander, "
                                  "and its own 99 excluded from the results")
            cmd.add_argument("--identity", default=None,
                             help="colour identity to stay within (e.g. URW); mutually "
                                  "exclusive with --deck, whose identity is derived")
            cmd.add_argument("--include-deck", action="store_true", dest="include_deck",
                             help="--deck: do NOT exclude cards already in the deck")
            cmd.add_argument("--no-game-changers", action="store_true", dest="no_game_changers",
                             help="drop Game Changers (they force bracket 4)")
            cmd.add_argument("--limit", type=int, default=None,
                             help="max results (default 15, at most 100)")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name == "scan-candidates":
            from manamap.pilot.candidate_scan import DEFAULT_LIMIT, DIMENSIONS
            cmd.add_argument("--dimension", default="all", choices=(*DIMENSIONS, "all"),
                             help="one dimension, or all (the default; --write needs all)")
            cmd.add_argument("--against-branch", default=None, metavar="NAME", dest="against_branch",
                             help="exclude and check infinites against a BRANCH's staged list rather "
                                  "than the deck's — Conqueror's departure is honoured")
            cmd.add_argument("--limit", type=int, default=None, help=f"rows per dimension (default {DEFAULT_LIMIT})")
            cmd.add_argument("--json", action="store_true", dest="as_json")
            cmd.add_argument("--write", action="store_true",
                             help="write the tracked candidate_scan.json (dated; the evidence a stage cites)")
            cmd.add_argument("--out", default=None, help="Also write JSON here (a view, never tracked; slug-scoped)")
            cmd.add_argument("--shortlist", action="append", default=[], metavar="CARD",
                             help="JOIN every source on these cards instead of scanning: the scan's dimensions and "
                                  "flags, the prescription's rank, the recon findings naming it, the EDHREC page, "
                                  "assess's read, and a predicted direction per Forge axis (a view)")
        if name == "forge-cast-check":
            from manamap.sim.cast_check import DEFAULT_CLOCK, DEFAULT_COPIES, DEFAULT_GAMES, DEFAULT_VS
            cmd.add_argument("--card", default=None, help="the card to prove (corpus name); or --adds")
            cmd.add_argument("--adds", action="store_true",
                             help="prove EVERY card the branch adds (needs --branch); writes cast_proofs.json with --write")
            cmd.add_argument("--jobs", type=int, default=None, help="shells at a time with --adds (default 2)")
            cmd.add_argument("--write", action="store_true", help="--adds: write the branch's tracked cast_proofs.json")
            cmd.add_argument("--force", action="store_true",
                             help="--adds: re-run cards already PLAYED under the same harness")
            cmd.add_argument("--copies", type=int, default=None, help=f"copies in the shell (default {DEFAULT_COPIES})")
            cmd.add_argument("--games", type=int, default=None, help=f"two-seat games (default {DEFAULT_GAMES})")
            cmd.add_argument("--vs", default=None, metavar="SEAT", help=f"the opposing seat (default {DEFAULT_VS})")
            cmd.add_argument("--clock", type=int, default=None, help=f"seconds per game (default {DEFAULT_CLOCK})")
            cmd.add_argument("--seed", type=int, default=None, help="Forge seed (default 4343)")
            cmd.add_argument("--branch", default=None, metavar="NAME", help="take the commander and filler from a branch's list")
            cmd.add_argument("--json", action="store_true", dest="as_json")
            cmd.add_argument("--out", default=None, help="Also write the JSON view and the game log here (never tracked; slug-scoped)")
        if name == "fetch-edhrec":
            cmd.add_argument("--theme", action="append", default=[], metavar="THEME",
                             help="an EDHREC theme page beside the base page, repeatable (e.g. aristocrats, lifedrain)")
            cmd.add_argument("--json", action="store_true", dest="as_json")
        if name in ("prices", "validate-prices"):
            # Its own `--branch` rather than a name in the shared tuple above: the
            # artifact is written beside whichever list it prices.
            cmd.add_argument("--branch", default=None, metavar="NAME",
                             help="price a branch's list instead of the deck's")
        if name == "prices":
            cmd.add_argument("--source", choices=("auto", "manapool", "scryfall"), default="auto",
                             help="auto: Mana Pool's public price feed (no token needed), "
                                  "falling back to Scryfall's prices.usd (the default)")
            cmd.add_argument("--write", action="store_true",
                             help="write prices.json beside the list (the deck's, or the branch's)")
            cmd.add_argument("--json", action="store_true", dest="as_json")
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
        if name == "proxies":
            # NOT slug-guarded and not `--out`: a print sheet is no deck artifact
            # (it lands on the Desktop), and `--out` means a swap-out elsewhere.
            cmd.add_argument("targets", nargs="*", metavar="SLUG@BRANCH",
                             help="branches whose ADDS to print (several -> one PDF)")
            cmd.add_argument("--have", action="append", default=[], metavar="NAME",
                             help="a card you already hold — not printed (repeatable)")
            cmd.add_argument("--have-file", dest="have_file", default=None, metavar="PATH",
                             help="cards you hold, one per line ('1 Name' is fine)")
            cmd.add_argument("--card", action="append", default=[], metavar="NAME",
                             help="a one-off card to print, from the corpus (repeatable)")
            cmd.add_argument("--paper", choices=("letter", "a4"), default="letter")
            cmd.add_argument("--dest", default=None, metavar="PATH",
                             help="where the PDF goes (default ~/Desktop/<slugs>-proxies.pdf)")
            cmd.add_argument("--no-open", dest="no_open", action="store_true",
                             help="do not open the PDF when it is written")
            cmd.add_argument("--dry-run", dest="dry_run", action="store_true",
                             help="list what would print; download and write nothing")
        if name == "deck-export":
            # Its own `--branch`/`--version`: a version is the deck's history out of
            # git, a branch its working list — `deck_export.source_text` refuses both.
            cmd.add_argument("--format", choices=("moxfield", "arena", "plain"),
                             default="moxfield",
                             help="moxfield: Moxfield's import box (the decklist.txt form); "
                                  "arena: MTG Arena's import; plain: `N Name`, sections kept")
            cmd.add_argument("--version", default=None, metavar="V",
                             help="a past version (number, tag or sha prefix) read out of git")
            cmd.add_argument("--branch", default=None, metavar="NAME",
                             help="export a branch's list instead of the deck's")
            cmd.add_argument("--out", default=None,
                             help="Write the text here (slug-scoped; a view, never tracked)")
        if name == "deck-link":
            cmd.add_argument("action", metavar="SERVICE|list",
                             help="`moxfield` to set or --remove its link; `list` to show them")
            cmd.add_argument("url", nargs="?", default=None,
                             help="the deck's URL, e.g. https://moxfield.com/decks/<id>")
            cmd.add_argument("--note", default=None, help="a line kept beside the link")
            cmd.add_argument("--remove", action="store_true",
                             help="drop this service's link (the file goes with the last)")
        if name == "buy-list":
            # `--branch` IS REQUIRED: a deck's own list has no "adds", so there
            # is nothing to buy for it — the bill is a property of a branch.
            cmd.add_argument("--branch", required=True, metavar="NAME",
                             help="the branch whose BUY rows to list")
            cmd.add_argument("--exact", action="store_true",
                             help="pin printings: `N Name (SET) CN` (Moxfield / Mana Pool "
                                  "exact-printings form) instead of `N Name`")
            cmd.add_argument("--json", action="store_true", dest="as_json",
                             help="{text, count, buy_cents|null, as_of|null}")
            cmd.add_argument("--out", default=None,
                             help="Write the list here (slug-scoped; a view, never tracked)")
        if name == "diagnosis-report":
            cmd.add_argument("--out", default=None,
                             help="Write markdown here instead of stdout (a view, never tracked)")
        if name == "deck-audit":
            cmd.add_argument("--write", action="store_true",
                             help="write the tracked audit.json (the handbook's "
                                  "Limitations section reads it; the manual may "
                                  "not compute at render time)")
            cmd.add_argument("--archetype", default=None,
                             help="aggro|control|combo|voltron — overrides what "
                                  "strategic_frame.json says; omit for the base targets")
        if name in ("impact", "deck-audit",
                    "deck-history"):
            cmd.add_argument("--json", action="store_true", dest="as_json")
            cmd.add_argument("--out", default=None,
                             help="Write JSON here instead of stdout (a view, never tracked)")


def run_pilot_step(args):
    """Dispatch a parsed pilot command to its module's main(args).

    A MALFORMED DECLARATION IS A REFUSAL, NOT A CRASH, and it is converted HERE
    so every command gets the same treatment. `goldfish.run` used to raise
    `SystemExit` for this — which is correct at a terminal and wrong in a
    library, because four commands call `run` IN PROCESS and one deck's bad
    `goldfish_targets.json` ended whatever sweep was running. It raises
    `DeclarationError` now; this puts the terminal behaviour back without
    putting the decision back into the library.
    """
    from manamap.pilot.goldfish import DeclarationError

    for name, module_path, _ in PILOT_STEPS:
        if name == args.pilot_command:
            try:
                importlib.import_module(module_path).main(args)
            except DeclarationError as bad:
                raise SystemExit(str(bad))
            return
    raise ValueError(f"Unknown pilot command: {args.pilot_command!r}")
