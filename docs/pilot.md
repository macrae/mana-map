# Pilot Subsystem — the bench, the evidence, and the loops

The deck side of Mana Map: everything under `manamap pilot`, the agents it spawns, and
the artifacts under `data/decks/<slug>/`. **Read `docs/vision.md` first** — it says who
this is for and what the bench does end to end; this document is the reference for how
each piece works and why it is shaped the way it is.

It turns a 100-card Commander decklist into a deck the pilot can **experiment on**
(`experiment`, the controlled A/B of two versions on one table), **measure** (a seeded
Forge run against the pod, a Monte Carlo goldfish against nobody), **prove** (rules-cited,
machine-checked stack resolutions), **understand** (the engine model, the 16-axis audit,
the diagnosis), **research** (dated `deck-recon`, `card-search` over the corpus),
**remember** (the captain's log and its debrief), **version** (from git, joined to the
games played), and **ask** (prescriptions).

**The experiment is the shape everything else serves.** A question becomes an arm, an arm
becomes a figure, and the figure travels with its interval, its N and its limits or it does
not travel. `viz/deck.html?deck=<slug>` is where the results are read; the magazine it used
to publish is a legacy renderer, described in the last section.

## The three-tier evidence contract

Every artifact and every figure carries its epistemic status, and the rest of the repo
exists to keep that honest:

| Tier | Badge | Content | Enforcement |
|---|---|---|---|
| Rules-verified | ✓ | Stack resolutions (v1 boards and v2 game states) | Citation contract + adversarial `rules-checker`; only a `pass` is a fact |
| Data-derived | ◆ | goldfish, Forge runs, the audit, mana analysis, versions, `deck-info` | Deterministic Python over committed artifacts; **seeded** where randomness exists, **sampled** said out loud where it cannot be |
| Coaching | ★ | the debrief, prescriptions' readings, decisions, the notes | Labelled judgment grounded in ✓/◆ artifacts; never wears a badge it was not granted |

## The citation contract

> The resolver is not allowed to make an uncited claim. Every effect it reports carries a Comprehensive Rules number pulled from the rules DB, and the checker's only job is verifying the cited rule text actually supports the claim.

Enforcement is layered:
1. **Form (code)** — `manamap pilot validate-stack`: every step has ≥1 citation; every rule ID matches `RULE_ID_RE` and exists in the index; every quote is a whitespace-normalized substring of real rule text. A resolution that fails form never reaches the checker.
2. **Meaning (agent)** — the `rules-checker` agent exact-fetches every cited rule and judges the *full* rule text against the claim (guards out-of-context quoting), and audits for missing steps (state-based actions, priority, triggers).
3. **Publication** — only stacks with `checker.verdict == "pass"` are facts anywhere downstream: the engine's solid lines, the doctor's `verified` claims, the deck page.

## Commands

**`manamap pilot deck-info <slug>` first** — the workbench view, with a derived *next*;
`deck-status` is its lifecycle half, and `/publish-deck` sequences the phases.

```bash
manamap pilot deck-status <slug> [--json]  # lifecycle completeness + STALENESS. Start here.
manamap pilot download-rules            # CR txt (idempotent; sha256 sidecar)
manamap pilot build-rules-db            # ~3.9K chunks → embeddings + index
manamap pilot query-rules "…" --json    # semantic top-k (resolver's discovery path)
manamap pilot lookup-rule 702.40a --json  # exact fetch (checker's verification path)
manamap pilot build-deck <slug> [--write-decklist]  # brief.json → build_plan.json (no agents)
manamap pilot validate-build <slug>     # form gate over a build plan
manamap pilot bracket-check <slug> [--target N] [--json]  # bracket floor → bracket_report.json
manamap pilot deck-facts <slug> [--out F]  # the deterministic brief agents read first
manamap pilot card-search [--deck <slug>] [--identity GU] [--oracle REGEX]…  # mine the corpus for candidates
manamap pilot commander-search <cards…> | --from FILE | --deck <slug>  # cards in, commanders out
manamap pilot archetypes "<commander>" [--theme SLUG]   # how it is actually built, and that style's role template
manamap pilot brew <slug> --commander "<name>" [--theme SLUG] [--from FILE] [--build]  # the cards you kept -> a deck on the bench
manamap pilot validate-recon <slug>                 # form-check deck_recon.json: cards real, legal, in identity
manamap pilot deck-history <slug> [--json]  # applied swaps (from git) + the pending ten
manamap pilot deck-notes <slug> add "…" [--result win|loss|draw] [--opponents N] [--tag T]
                                        #   the captain's log: AUTHORED, append-only, sha-stamped
manamap pilot deck-notes <slug> list [--since D] | show <id>
manamap pilot deck-info <slug> [--json] [--write]            # THE WORKBENCH VIEW: version · record · status · figures · what to do next
manamap pilot simulate <slug> --vs A [--vs B…] [--games N] [--jobs J]   # N seeded Commander games in Forge, headless; a ◆ run record
manamap pilot simulate <slug> --list | --dry-run | --analyze <run-id>
manamap pilot validate-sim <slug>                 # form + re-derive the analysis from logs where they exist
manamap pilot fetch-opponent "<commander>" [--as slug] [--note …] | --list   # a pod seat under data/opponents/ from EDHREC's average deck
manamap pilot experiment <slug> --a V4 --b working --vs <opp>… --games N   # A/B same table; one artifact with the delta
manamap pilot sim-scenario <slug> <run> --game G --turn T [--step "declare blockers"] [--stack]
                                        #   lift one board into a game_state v2 scenario (question left to you)
manamap pilot deck-version <slug> [list] [--json]   # every list this deck has been, from git; games per version
manamap pilot deck-version <slug> show V4 [--full] | tag <name> [--at V4] [--note …] | restore V4 [--write]
manamap pilot deck-branch <slug> [list] | new <name> --from <file> [--why …]
manamap pilot deck-branch <slug> show <name> | diff <name> | source <name>
manamap pilot deck-branch <slug> merge <name> [--write] [--proxy] [--force --reason …]
                                        #   a candidate 99 you cannot yet sleeve: diff it,
                                        #   price it against your boxes, MEASURE it
                                        #   (`--branch` on fetch-deck / bracket-check /
                                        #   mana-analysis / goldfish / deck-facts /
                                        #   deck-audit / deck-map), and merge only when
                                        #   every added card is sourced
manamap pilot diagnose <slug> [--branch N] [--vs main] [--iterations N] [--json] [--write]
                                        #   the vitals: engine online against what the deck
                                        #   DECLARES, stall risk, mana — every rate with its
                                        #   interval; `--vs` adds the interval on the DIFFERENCE
manamap pilot candidates <slug> --pool <file|library|-> [--axis A] [--cut CARD] [--limit N]
                                        #   rank a pool by what each card MEASURABLY does:
                                        #   substitute, re-run, report the delta. Anything
                                        #   under the MDE is marked as noise, not ranked.
                                        #   `--pool library` reads pool.txt, written by
                                        #   the Atlas's "consider these" on a pile
manamap pilot merge-debrief <slug>      # the debrief agent's annotations in, by entry id
manamap pilot validate-debrief <slug>   # the annotation held to the log and the 99
manamap pilot prescribe <slug> "<question>"   # open ONE question to the doctor (accumulates under prescriptions/)
manamap pilot prescribe <slug> --list | --merge <id>
manamap pilot validate-prescription <slug> [--id ID]  # the diagnosis contract, scoped; stale = form only
manamap pilot deck-audit <slug> [--archetype A] [--json] [--out D/]  # cited axis targets + engine activation
manamap pilot card-value <slug> [--metric M] [--iterations N] [--json] [--out D/]
                                        # what each card is WORTH: swap it for a blank, measure the loss
                                        # needs `model_combat`; invisible cards are EXCLUDED, not ranked last
manamap pilot validate-pending <slug> [--json]   # the queue: decided, not yet applied
manamap pilot deck-status --all         # THE FLEET VIEW: every deck, stale count, queued count
manamap pilot validate-diagnosis <slug>    # diagnosis form; axes re-derived, cuts checked against verified stacks
manamap pilot deck-map <slug>           # the constellation: local layout + cities/neighbourhoods
manamap pilot merge-deck-map <slug>     # cartographer's names in — `label`/`gloss` ONLY
manamap pilot validate-deck-map <slug>  # names distinct, membership untouched
manamap pilot engine-facts <slug> [--json] [--out D/]  # the deterministic engine brief
manamap pilot validate-engine <slug>    # stages, completeness, verified_by re-checked
manamap pilot pool-facts <paths…> [--exclude F] [--json] [--out F]  # a BOX OF CARDS → which deck to build
manamap pilot cache-rebless <slug>             # re-record every STALE_OK routine, zero spawns
manamap pilot impact <slug> [--json]           # card/figure/target/zone staleness report (free)
manamap pilot validate-strategic-frame <slug>  # frame form + candidate-line flags
manamap pilot check-in <slug> --from F  # a PAPER list → decklist.txt: diff, refuse, apply
manamap pilot targeting <slug>          # who the pod attacks, measured from sim logs
manamap pilot build-page <slug>         # the compact deck page (the Pilot's Manual)
manamap pilot fetch-deck <slug>         # decklist.txt → cards.json (Scryfall)
manamap pilot validate-deck <slug>      # 100/commander/singleton/color identity
manamap pilot validate-stack <slug> [--stack NNN]   # citation contract (stacks + decisions)
manamap pilot validate-stack <slug> --scenario-only # preflight BEFORE spawning a resolver
manamap pilot scaffold-targets <slug>   # a DRAFT goldfish_targets.json to EDIT — derived from
                                        #   contained combo lines and role axes, marked
                                        #   "scaffolded": true until a person rewrites it
manamap pilot goldfish <slug>           # seeded Monte Carlo metrics → goldfish_metrics.json
manamap pilot benchmark [<slug>|--all]  # THE STANDARD BENCHMARK (PRD §9): four measures
                                        #   under ONE frozen configuration — fixed seed,
                                        #   fixed iterations, UNIFORM model flags that
                                        #   override each deck's own opt-in, because
                                        #   comparability is the whole requirement.
                                        #   Reads the 99, never the declaration. Emits NO
                                        #   aggregate score: §14.1 is open and the fleet
                                        #   spread says speed is not archetype-neutral.
manamap pilot validate-goldfish-targets <slug>  # the DECLARATION itself: cards still in the 99,
                                        #   and any card in 2+ passing stacks with no component
manamap pilot mana-analysis <slug>      # the mana audit, deterministic — run AFTER goldfish (embeds its figures)
manamap pilot scenario-facts <slug> [--stack NNN]  # the deterministic brief for ONE scenario
manamap pilot validate-considering <slug>   # LEGACY gate on the frozen considering.json
manamap pilot validate-tutor-guide <slug>   # every fetch target is in this deck
manamap pilot diagnosis-report <slug>   # render diagnosis.json as readable markdown
manamap pilot artist-credits <slug> --json  # standout artists + art themes (legacy page)
manamap pilot short-list-art <slug>     # LEGACY: card art for a frozen considering.json
manamap pilot build-manual <slug>       # → manuals/<slug>.html (LEGACY magazine renderer, until manual-v5)
manamap pilot build-index               # → manuals/index.html + data/decks/index.json (the deck manifest the viz reads)
manamap pilot issue-length <slug> [--rendered]  # LEGACY: words + screens per section of the rendered page
manamap pilot validate-issue <slug>     # LEGACY gate: issue.json + the frozen issue_plan.json
manamap pilot cache-status <slug>       # have an agent routine's inputs changed?
manamap pilot cache-record <slug> --routine R   # record what produced an artifact
manamap pilot cache-clear <slug>        # drop cache records
manamap pilot cache-snapshot <slug>     # every routine's status BEFORE a cache-format change
manamap pilot cache-rerecord <slug>     # re-fingerprint what a FORMAT change invalidated (gated
                                        #   on a snapshot; never a way to make a red board green)
manamap pilot merge-prose <slug> --routine R  # an agent's .agent-out prose in, keys it owns ONLY
manamap pilot validate-strategy         # form-check strategy.md + CHANGELOG
manamap pilot build-strategy-db         # chunk + embed strategy.md
manamap pilot query-strategy "…" --json # semantic top-k strategy search
manamap pilot lookup-strategy <id> --json  # exact section fetch (strategy:tempo)
```

## Data layout

```
data/rules/                    gitignored (regenerable): comprehensive_rules.txt,
                               rules_index.json, rules_embeddings.npy, sidecars
data/strategy/                 strategy.md + CHANGELOG.md tracked;
                               strategy_index.json / strategy_embeddings.npy /
                               .strategy-db-meta.json gitignored (regenerable)
data/decks/<slug>/             all tracked:
                               brief.json            authored (build side only)
                               candidate_pool.json   deck-analyst
                               build_plan.json       build-deck (deterministic) + deck-architect ⇄ deck-critic merge
                               bracket_report.json   bracket-check (◆)
                               decklist.txt          authored, OR build-deck --write-decklist
                               cards.json            fetch-deck
                               goldfish_targets.json authored
                               goldfish_metrics.json goldfish
                               stacks/NNN-*.json     authored scenario + resolve loop
                               decisions/NNN-*.json  pilot-notes
                               strategic_frame.json  strategy-researcher (consult)
                               manual_prose.json     pilot-notes (five keys; card_roles/mana_base/upgrades/
                                                     editors_letter/pilots_log are FROZEN legacy, unowned)
                               pilot_feedback.md     authored, OPTIONAL (free-text pilot notes; the log supersedes it)
                               log.jsonl             AUTHORED, append-only — `deck-notes add` (the captain's log)
                               log_annotations.json  debrief agent, by entry id — `merge-debrief` / `validate-debrief`
                               deck_versions.json    AUTHORED tags on git-derived versions — `deck-version tag`
                               diagnostic.json       the VITALS — engine online against what the
                                                     deck declares, stall, mana. `diagnose --write`
                               pool.txt              CANDIDATES to consider, not a promise: written
                                                     by the Atlas's "consider these" on a library
                                                     pile, read by `candidates --pool library`
                               branches/<name>/      A CANDIDATE 99 you cannot yet sleeve —
                                 decklist.txt          the candidate list
                                 branch.json           name, opened, why, base version
                                 cards.json            `fetch-deck --branch`
                                 <measurements>.json   bracket / mana / goldfish / map,
                                                       run with `--branch`, written here
                                                       so they can never overwrite the
                                                       deck's own
                               mana_analysis.json    mana-analysis (deterministic, no agent)
                               tutor_guide.json      pilot-notes (one wish per tutor)
                               considering.json      FROZEN legacy (The Short List; its rule lives in prescriptions now)
                               prescriptions/<id>-*.json  AUTHORED question + deck-doctor ⇄ deck-skeptic answer — `prescribe`
                               diagnosis.json        deck-doctor ⇄ deck-skeptic (the improvement loop)
                               deck_recon.json       deck-doctor MODE recon (dated; perishable)
                               issue.json            authored (never generated)
                               issue_plan.json       LEGACY, frozen — the retired magazine-editor's packaging
                               .agent-cache.json     cache-record
data/opponents/<slug>/         tracked: the pod — decklist.txt + source.json (`fetch-opponent`)
manuals/<slug>.html            tracked; the LEGACY magazine render per deck + manuals/index.html
```

**Exact printings**: `fetch-deck` resolves a Moxfield export's `(SET) COLLECTOR [*F*]`
annotations against Scryfall's `/cards/collection` by set + collector number **first**,
falling back to name lookup only for unannotated lines. `cards.json` therefore carries
the physical card the pilot owns — artist, set, collector number, border, frame
effects, finishes, foil, plus `art_crop` for full-bleed art. Image URLs have
Scryfall's cache-busting query string stripped so re-fetches stay byte-stable, and the
run short-circuits entirely when the decklist hash is unchanged (`--force` to override
after an oracle update).

Deck slugs are kebab-case. Scenario files are `NNN-<kebab>.json`, zero-padded, authoring order. Card names use the full `" // "` form, matching the combo/synergy/obsolescence graph keys.

## Deck status — is this deck finished? (`deck-status`, tier ◆)

**Run this first on any deck.** The lifecycle is dozens of skills and subcommands and until
2026-08 nothing said what a COMPLETE deck looks like: each phase knew its own inputs, none
knew the sequence. So a capability added in one development cycle was reachable only by
somebody who remembered it existed, and a deck built the following month silently inherited
the old pipeline. Capabilities added in August (`ADDED_2026_08` — the map, the engine, the log) are named so a
deck built before them reports them MISSING rather than complete-by-omission.

`pilot/deck_status.py:STAGES` is the single machine-readable statement of what a deck can
have and in what order; `/publish-deck` sequences the work and reads the same list rather
than restating it. **When you add a phase to the lifecycle, add it to `STAGES`** or the next
person will not find it. That rule has been broken once already and by the file that states
it: two legacy stages shipped on all nine decks while `deck-status` reported nine complete
decks without them.

**A stage whose artifact exists for another reason cannot be checked by file presence.**
The retired `panel` stage wrote its keys INTO `manual_prose.json`, which the writer had
already created, so it was checked by KEY rather than by file; the stage and the keyed
mechanism left `STAGES` together when the magazine agents were retired (2026-08-19). The
lesson stands: bring the mechanism back with the first stage that needs it, not before.
The `log` stage is the current example of the same care — the authored log has no gate,
so the row runs the debrief's validator on the annotation beside it.

It separates two things that look alike. **INCOMPLETE is a state** — a half-built deck is
work in progress. **STALE is an error**: most artifacts stamp the `decklist_sha256` they were
derived from, and one whose stamp no longer matches `cards.json` is not incomplete but
CONFIDENT AND WRONG, which is worse and looks finished from every angle except this one.

## The workbench view (`deck-info`, tier ◆, computed on demand)

One deck, one screen: commander and identity, the current version and its tags, the
lifecycle status (with STALE/INVALID named), the bracket floor against target, the
record from the captain's log (games, W/L, last played, un-debriefed), the goldfish
headline figures, the engine's verified-line count and critic verdict, the audit's
under/over axes, the diagnosis verdict and skeptic, the prescriptions asked and
answered, and every open question across engine/diagnosis/debrief with its route. It
computes nothing new — every figure is read from the module or artifact that owns it —
so it cannot disagree with `deck-status`, `deck-version`, `deck-notes`, `prescribe` or
`deck-audit`. **The `next` block is the point**: each suggestion is derived from a
condition true right now (un-debriefed games → `/debrief`; an uncommitted working list →
commit it; a stale stage → regenerate; no games → play it; open prescriptions → run the
loop) and names the command. No judgment about the deck lives there. `--json` is the
shape a future UI reads.

**`--write` puts the same shape on disk as `info.json`, and that one IS committed.**
It is what `viz/deck.html` fetches — the only committed artifact composed from every
other one, which is exactly why it is staleness-gated by recomputation
(`tests/test_pilot_artifact_freshness.py`) rather than by a stamp: it goes stale when
ANY input moves and it stamps nothing. It **omits the version block** by construction
(`deck_info.fetchable`), because versions come from a git walk and the commit that
changes `decklist.txt` gets its sha after anything written in the same commit — a
committed version number is one commit behind forever, and a wrong version is worse
than an absent one when the captain's log stamps games against it. The page reads a
deploy-time `versions.json` instead, which CI can build because `deck_versions` needs
only git while `deck_audit` needs the gitignored corpus.

**A deck that no longer exists says so, and is not told to go and play itself.** The
`status` field on `issue.json` (`broken-down` / `retired` / `superseded`, absent = live)
is the deck's authored *existence*, distinct from the stage lifecycle above. It renders
as a banner under the header, and for the two statuses that mean there is no cardboard
to shuffle — `broken-down`, `retired` — the suggestions that end in "play it", "simulate"
or "run an experiment" are **withheld and said to be withheld**, because a silently
shorter list reads as "nothing to do here". `superseded` is deliberately not in that set:
that list is still sleeved. Everything a published record can still do — fix a failing
gate, regenerate a stale artifact, settle an open rules question — survives.

The vocabulary lives in `pilot/common.py` (`DECK_STATUSES`, `UNPLAYABLE_STATUSES`,
`deck_lifecycle`), not in `issue_spec`, where it started: the workbench has to read it and
`issue_spec` is the frozen renderer that gets deleted with the magazine. `issue_spec`
re-exports it under the old names so the legacy banner is unchanged.

## Mining the corpus (`card-search`, tier ◆, computed on demand)

Every other command on the bench measures a deck; this is the only one that answers the
question those measurements end in — **which cards would fix it**. `deck-audit` names an
under-filled axis, `goldfish` prices a thin component, `prescribe` asks the doctor for
adds, and until now all three needed a human or an agent to *think of* candidates. An
agent asked to think of candidates invents them; this hands it a list it did not author,
and a validator can then check membership.

```bash
manamap pilot card-search --deck kianne --oracle "additional combat phase"
manamap pilot card-search --identity GU --role ramp:rock --cmc-max 2 --no-game-changers
manamap pilot card-search --deck heliod --name "^Sword of " --include-owned
```

Filters: `--identity` / `--deck`, `--oracle` (regex, repeatable, ANY unless `--all`),
`--name`, `--type`, `--role`, `--cmc-min/max`, `--no-game-changers`, `--owned` /
`--unowned`, `--limit`. Three rules it enforces so a caller cannot get them wrong:

- **Identity is DERIVED, never authored.** `--deck` takes it from that deck's commander,
  the same rule `build_deck.load_brief` follows, and passing `--identity` alongside is a
  hard error rather than an override.
- **A candidate is a card you do not already have.** `--deck` excludes the deck's own 99
  (`--include-owned` turns that off) — a search that hands your own list back is the
  commonest way a tool like this wastes a reader's time.
- **Truncation is stated.** A silently cut list reads as "that is all of them".

`--owned` / `--unowned` read `pilot/collection.py`, so "owned" means **in a box OR
sleeved in a tracked deck** — unsleeving is a decision, not a purchase. They are exact
complements and a test asserts it; without the flag the `owned` field is `None`, because
"not asked" is not "no". `--unowned` is the buy list.

**`--identity` takes letters, and the compact form was broken.** `parse_identity_arg`
accepts `GU`, `gu`, `G,U` or `G, U`; a non-colour letter is an error, never a silent
narrowing. It exists because `analysis.common.parse_color_identity` splits on commas —
correct for `cards.csv`'s `"G, U"` — and returned `{"GU"}` for the compact form, one
two-character token that no coloured card's identity can be a subset of. `--identity GU`
therefore returned only *colourless* cards while printing "identity GU" in its header.
Same shape as the bug recorded in `card_pool._build_pool`.

`--oracle` searches rules text and `--name` searches names, deliberately separately:
`--oracle "Sol Ring"` looks like it should find Sol Ring and instead finds every card
whose rules text mentions one, which is a confusing empty result rather than an error.

It does **not** score fit. The repo has one scorer (`build_deck`) and one retrieval aid
(the synergy graph, whose own docstring says it "is a retrieval aid and not a scoring
function"); a second opinion here would be a third answer to "is this card good" that
nothing reconciles. Results rank by EDHREC rank, unranked last — a card with no rank is
usually just new, and a new set's answer to a problem is exactly what this should surface.

## Deck versions (`deck-version`, derived from git; `deck_versions.json`, authored tags)

Every change to the 99 is a commit (`decklist.txt` is tracked), so the list of lists
this deck has been is already on disk. `manamap pilot deck-version <slug>` numbers them
— `V1` the first tracked list, `V2` the first content change — reusing `deck-history`'s
git walk and parser, so a comment-only edit adds a byte-sha to the version it belongs
to and never a new version. The captain's log stamps each entry with the byte-sha of
`decklist.txt` as it stood, so the join is exact: each version reports its games and
W/L, and an entry played on an uncommitted working copy is reported **unmatched** rather
than guessed (commit the list and it resolves). `show V4` diffs a version against the
working list; `tag <name> [--at V4] [--note]` writes the authored `deck_versions.json`
(the one version datum a browser can read without git); `restore V4` is a **dry run**
unless `--write`, after which `fetch-deck` → `goldfish` → `mana-analysis` and a commit.
`deck-status` prints the current version in its header.

### The paper lock — three states, not two

`deck-version <slug> paper [--at V4] [--note …] [--clear]` asserts that one exact list is
the one **sleeved**. It is authored and it is the only claim in the repo about cardboard;
nothing derives it, because nothing can. Locking a version is also what makes **drift** free
— `report()` already knows the current version, so locked / in_sync / versions_behind falls
out, and the two sides are named `pull` and `add` because that is the physical instruction.

The state that had to be added is the third one:

| state | meaning | what the front door says |
|---|---|---|
| **locked** | the pilot says this exact list is sleeved | play it, log it, simulate it |
| dead (`broken-down` / `retired`) | it demonstrably is not | the play/measure loop is closed, **and says what it withheld** |
| **absent** | nobody has said either way | *not marked as built in paper* — with the command that settles it |

`deck-info` had two of the three and treated **absent as locked**, telling the pilot to go
and play a deck that may never have been built. That is the quiet half of the defect that
once had this same command recommending `hapatra` while hapatra's cards sat in yawgmoth's
sleeves: the loud half was caught because the cards were provably elsewhere, this one just
does not know and said nothing, which reads identically to knowing.

It **informs rather than withholds** — an unbuilt deck is not a closed deck, so the
play/measure suggestions stay and a line above them names the command. Withholding is
reserved for a deck we know is gone.

Only the **authored** half of the lock reaches `info.json` (version, date, note).
`paper_state`'s drift needs a git walk, and `info.json` is committed and omits everything
git-derived because the commit that changes `decklist.txt` receives its sha after anything
written alongside it — a stored drift would be one swap behind forever. `paper()` is a plain
file read, so it is on the safe side of that split, and a test asserts `in_sync` and `drift`
never appear there.

**A lock nobody checked is worse than no lock.** Three decks carried placeholder locks
written to exercise the drift display during a demo; two asserted "in sync" and one asserted
a two-card drift, and all three rendered identically to evidence on the surface whose job is
saying what you can play tonight. They were withdrawn rather than corrected — the note on a
lock should say who checked and when.

### What a version bump means

`deck-version` NUMBERS every list from git (V1, V2, V3…) and that numbering is
mechanical: it counts content changes and says nothing about their size. A **tag** is
where the pilot says how big a change was, and the vocabulary is semantic versioning
with this deck's own meanings:

| bump | what changed | examples |
|---|---|---|
| **PATCH** — `v1.0.1` | mana only. The deck does the same things; it does them more reliably. | a tapland for an untapped one, +1 land / −1 spell, a Signet for a dork, fixing a colour the audit says is short |
| **MINOR** — `v1.1.0` | the deck can now do something it could not do before. | a new engine component, a new combo line, a single card that adds a capability (Gleaming Splendor) |
| **MAJOR** — `v2.0.0` | it is a different deck wearing the same commander, or a different commander. | voltron → stax, a new commander, a large fraction of the 99 moving at once |

The line between patch and minor is **capability, not card count**: twelve land swaps
are a patch and one enchantment can be a minor. That is not a stylistic preference —
it matches what the bench can actually measure. A mana change moves `mana-analysis`
and `goldfish`'s resource curve and leaves `engine.json` alone; a capability change is
one that ought to move the engine model or add a line worth resolving. If a proposed
bump does not move the artifact its tier implies, the bump is probably the wrong tier.

Keep it beside the measured finding in CLAUDE.md that **lowering a deck's curve does
not make it faster; its mana base does** — twelve spells three mana cheaper moved
ur-dragon's turn-five mana by 0.02, while three lands and four accelerants moved it
+0.27. A patch is the tier that most often buys the most.

**Every slug starts at `v1.0.0`.** Zask → Blech → Hapatra is a lineage of *cardboard*,
not versions of one deck: each has its own slug, its own artifacts, its own captain's
log, and every key in the repo is the slug. A rebuild that takes a new commander takes
a new slug and starts again at 1.0.0; the old deck is marked `broken-down` or
`superseded` (see `DECK_STATUSES`) and keeps its own history intact.

Tags are authored — `manamap pilot deck-version <slug> tag v1.1.0 --at V4 --note "…"`.
A name that is nothing but digits and dots must be a well-formed release, so `v1.2` is
**refused** rather than filed as a nickname: a nickname sorts alphabetically and would
sit in the wrong place forever while looking exactly like a version. Releases sort
numerically and nicknames alphabetically after them, because plain lexical order puts
**`v1.10.0` before `v1.9.0`** — and a deck reaches its tenth minor bump in an ordinary
year. Re-tagging an existing name at a different version is refused without `--force`:
a tag is a claim about one exact list, and moving it silently re-points every artifact
that ever quoted it.

**Not yet automated, deliberately.** The tool should propose a tier and the pilot
confirm it, and that needs three things the repo does not have: a diff between two
*arbitrary* versions (every diff today is consecutive or against the working tree),
`quantity_changes` carried into `versions()` (`history()` computes it and `versions()`
drops it, so 36→37 Forests reads as no change at all), and a classifier that reports
**evidence and never intent** — `deck_history` is explicit that *why* a card moved is
not knowable from a commit. Until then the tier is the pilot's call, which is the
honest arrangement rather than a stopgap.

### check-in — a paper deck arrives

`manamap pilot check-in <slug> --from <file>` (or `-` for stdin) is how a rebuilt
paper deck enters the repo. It parses the pasted list with `fetch_deck.parse_decklist`
— the same parser the browser importer is fixture-locked to — diffs it against
`decklist.txt` at **copy** level, and reports **PULL** (leaves the sleeves) and **ADD**
(goes in). A dry run by default; `--write` applies it and then runs `fetch-deck` →
`goldfish` → `mana-analysis`, because those two artifacts stamp the decklist sha and
leaving them behind makes the deck read as stale forever.

**It refuses rather than guesses**, and that is the whole point. A paper list is typed
by a human reading sleeves, so it arrives with a card written twice, a name
misremembered, or ninety-nine cards where there should be a hundred — every one of
which `fetch-deck` would survive silently, leaving a repo list that is not the deck on
the table. Blocking: a non-basic card listed more than once (singleton forbids it;
basics are exempt), a total that is not 100, a name matching nothing in the corpus, no
commander. Warning-only: a changed commander (that is a different deck, and probably
wants a new slug) and an absent corpus, which cannot check names but must not stop a
fresh clone accepting a deck. `--force` overrides; you want it approximately never.

The written file is **canonical**, not verbatim — `Commander:` block, then `Deck:`
sorted by name, printings and foil markers carried through. Reformatting cannot
manufacture a version, because `deck-history` and `deck-version` compare parsed entries.

The last step is a commit, and it is not optional bookkeeping: `decklist.txt` is tracked,
so the commit is what `deck-version` numbers and what the captain's log stamps games
against. Check a deck in without committing and tonight's games attach to no version at
all. Then `deck-version <slug> paper` marks it as sleeved.

### build-page — the Pilot's Manual

`manamap pilot build-page <slug>` renders the compact deck page from
`pilot/page_spec.py:SECTIONS` — the plan, the roster, the mulligan, the lines, the table
read, the debrief, the numbers, the proof. It is the replacement for the magazine
(`docs/manual-v5-spec.md`), built alongside the frozen `build_manual.py` rather than over
it, and it writes to the same `manuals/<slug>.html` path.

**Measured, not estimated.** Radagast 71.3 screens → **15.5**; yawgmoth-swarm 88.4 →
**21.9**; edgar 16.1; goblin-storm 12.8. Visible words on radagast fall 34,653 → ~5,900,
because the folds do most of the work.

Two measurements changed the design mid-build, and both are worth knowing:

- **A page whose stylesheet is missing measures three times its real height.** The first
  render came in at 83.4 screens — *worse than the magazine* — because it was written to a
  scratch directory where the relatively-linked `magazine.css` 404s, so every hover-preview
  image rendered inline at full size. `build-page --out` now copies both sheets beside the
  page. Nothing about the renderer was wrong.
- **THE LINES folds the board WITH the theatre.** With the board open the section was 8.2
  screens against the spec's ~4, while every other section matched or beat its estimate —
  so ~4 was only ever reachable with the board folded. The argument stays open (question,
  intro, result); the evidence folds. The result is the authored `final_state.summary`
  rather than a second board block: that block is 266px, the same height as the board it
  followed, and `judges_desk_files` already renders it inside every case in THE RECORD, so
  the page was drawing the same board twice.

**Every section degrades to absent**, never to `[TODO]` — that was a magazine convention,
needed because the contents page indexed every department whether or not it had copy. The
nav rail here is generated from what actually rendered. Measured: of the nine, `kinnan` (a bare build
plan) renders two, `kianne` four and `radagast` eight — and none of them raise.
(Spelled out on purpose. `test_no_surface_states_a_wrong_section_count` cannot tell a sentence counting how many rendered from one stating how many the registry holds — no regex can — and a doc is cheaper to reword than a check is to make smarter and wronger.)

Rebuilds are byte-identical: no build date anywhere, and the authored `issue_date` is
printed only when there is one.

### targeting — who the pod actually attacks

`manamap pilot targeting <slug>` is the one measurement here about the OPPONENTS'
choices rather than our deck's development. It walks every sim run and experiment arm's
logs and asks, at each declare-attackers step with **at least two living opponents**,
which seat the attacker chose — then scores that against three rankings: the seat that
has dealt the most combat damage, the lowest-life seat, and the highest-life seat.

**The unit is a decision, not a game.** That is what makes the question answerable:
twelve games is nothing, but twelve games hold hundreds of targeting decisions. A forced
choice is not a choice, so a one-on-one run contributes zero by construction rather than
being filtered out by hand. It is not per attacking creature either — five creatures
almost always go at the same player, which would multiply the sample without adding one
independent choice.

**Ties count the whole tied set**, and the null expectation for that decision is the tied
set over the choice set. Early on every seat is at forty life; a rule that broke that tie
arbitrarily would manufacture signal out of nothing. Inference is a **seeded permutation
test** over each decision's own eligible set — the null is a sum of Bernoullis with
different probabilities, not one binomial, which is why it is simulated rather than looked
up. The pooled Wilson interval and a **game-clustered** mean are both reported, because
decisions cluster inside games and the pooled one is optimistic.

Measured on both decks that have logs, and they agree: the pod attacks the
biggest revealed threat **0.685** (radagast, 444 decisions) and **0.675** (kianne, 335)
against uniform baselines of 0.54 and 0.52, p = 0.0001. But on the 196 radagast decisions
where "biggest threat" and "easiest kill" name **different** seats, it is 0.541 against
0.403 with overlapping intervals — **not enough to say which it follows.** Both halves
ship in the artifact.

**Strength is REVEALED cumulative combat damage, not board power.** Forge prints printed
P/T on resolution and `bridge.py` parses it, but counters, anthems, auras, equipment and
token counts are invisible in a log — so a board-power ranking would be biased against
token, counter and anthem decks, which is most of this pod.

**What it can and cannot claim.** This is empirical **opponent modelling**: the input half
of a game-theoretic argument, and the first thing here about opponents' choices. It is not
an equilibrium, a solution concept, or human politics — no deals, no grudges, no table
talk, no player who remembers last turn. The artifact key is `forge_ai_targeting_policy`
rather than anything shorter precisely so the caveat cannot be trimmed off, and
`limits[]` carries `FORGE_AI_CAVEAT` as the imported constant.

**Why the version list is not a tracked file:** the commit that changes `decklist.txt`
gets its sha AFTER anything written in the same commit, so a generated `versions.json`
would be one behind forever. Computed on demand; the viz history viewer gets its copy
from a deploy-time step with git available.

## The captain's log (`log.jsonl`, authored) and the debrief (`log_annotations.json`, ★)

What happened when the deck was PLAYED — the one thing no other artifact records.
`manamap pilot deck-notes <slug> add "…" [--result win|loss|draw] [--opponents N]
[--tag T]` appends one JSON line (`id`, `at`, `decklist_sha256` of `decklist.txt` as it
stood, `result`, `opponents`, `tags`, `text`) and nothing ever rewrites it; `list
[--since]` and `show <id>` read it back, marking which entries the debrief has read.
Light structure on purpose: the point is that a note costs one sentence.

The `debrief` agent (`/debrief`, cheapest agent in the set) reads the un-debriefed ids
and writes, per entry: `summary`, `opponents[]` (each with a verbatim `evidence` phrase
of the note), `cards[]` (`read` ∈ over/under/as-expected/missed), `decisions[]`
(`worth_a_spread` → `/author-decision`), `takeaways[]`, `engine_stages[]` (names from
`engine.json`), `lines[]` (`verified` only with a checker-passed `stack_artifact`, else
`needs a stack scenario`) and `open_questions[]` routed to
`resolve-stack|goldfish|research-strategy|diagnose`. `merge-debrief` writes by id,
rejects ids the log lacks and carries earlier annotations; `validate-debrief` fails the
annotation on any of those contracts. The one rule is that the debrief may name nothing
the pilot and the 99 did not — it is a reader, not a witness.

## Prescriptions (`prescriptions/<id>-*.json`, the diagnosis scoped to a question)

`diagnosis.json` is deterministic over the deck and takes no prompt. The workbench asks
the doctor *questions* — "I keep getting wrathed on five", "make it faster", "should I
run Sol Ring" — so a question is an ARTIFACT, deterministic over (deck, question).
`manamap pilot prescribe <slug> "…"` writes the authored half (`prompt`, `id` = hash of
the normalized prompt, `as_of_decklist_sha256`); `deck-doctor` MODE prescribe writes the
answered half (`reading`, `log_entries_read`, optional `axes_engaged`, `cut_candidates`,
`add_candidates` RANKED and capped at ten — The Short List's rule, relocated —
`open_questions`, `gaps`); `deck-skeptic` reviews it like a diagnosis; `prescribe --merge
<id>` folds both in, answer keys only. The cache routine `prescription:<id>` digests only
the prompt (`prompt:self`), so merging never self-invalidates; `cache-record` refuses a
file without a passing skeptic. Prescriptions ACCUMULATE and are never overwritten: a
later decklist makes one stale (MISS; `validate-prescription` checks form only), never
wrong. Both doctor modes read `log_annotations.json`.

## Simulation — a table, not a goldfish (`simulate`, tier ◆ seeded)

Forge is the rules engine; this repo owns the harness, the parser and the bridge.
`manamap pilot simulate <slug> --vs <opponent>… --games N` runs N seeded Commander games
headless against seats from `data/opponents/` (your pod, via `fetch-opponent`) or your
other decks, and writes one tracked run record — win rate with a Wilson interval, who
eliminated whom and how (damage or life loss), combat damage by round, the token figures
counted two honest ways — with Forge's own AI caveat in its assumptions. `validate-sim`
re-derives the analysis from the logs where they exist; `sim-scenario` lifts one board at
a CR step into a **game state v2** scenario for `/resolve-stack`. The design, the spike
and the measured limits are in **`docs/simulation.md`**; the v2 schema is below.

## Goldfish metrics (`goldfish_metrics.json`, tier ◆)

### The Treasure model (opt-in per deck)

A Treasure is **not** a mana rock and modelling it as one is the trap: a rock
produces every turn forever, a Treasure produces once and is gone. `simulate_once`
keeps a **stockpile** spent only when lands and rocks fall short, which is both how
it is played and what makes a hoard-counting payoff measurable.

**Only triggers a goldfish can see are modelled** — upkeep, landfall, cast, Saga
chapters (recurring) and enters-the-battlefield (once). This simulation has no
combat and no opponents, so `whenever this creature deals combat damage` and
`whenever an opponent draws` produce **nothing**. That is a finding, not a
shortcoming: measured across the nine decks, **16 of 19 Treasure sources are
combat- or opponent-gated**, and a naive `create a Treasure token` match would have
handed eight decks free mana they never get — turning a deliberately conservative
model optimistic. Unmodelled sources are NAMED in
`meta.treasure_sources_not_modelled`, so a hoard of zero is legible rather than
mysterious. Ur-dragon's four Treasure Dragons are all combat-gated and all report
zero, which is the single most useful thing the model says about that deck.

**It is opt-in**, via `"model_treasures": true` in `goldfish_targets.json`, and for
the same reason `OPTIONAL_DEPARTMENTS` existed: a model that changes every deck's
numbers at once cannot be landed on one deck first. Turning it on fleet-wide moves
three decks (gishath, goblin-storm, radagast) and **gishath's `mean_cast_turn`
alone — 7.969 → 7.912 — is quoted sixteen times across seven tracked artifacts**,
including agent-authored prose and an `engine.json` carrying a critic verdict.
With the flag off the treasure keys and the two extra `model_assumptions` lines are
**absent rather than zeroed**, so a non-opted deck's artifact is byte-identical to
before the model existed. `goldfish` prints a WARNING naming the sources it is
ignoring, so a deck cannot sit un-opted by accident. Remove the flag once every
deck has been re-baselined — a permanently optional model is one nobody committed
to.

What it reports when on: `treasure.mean_treasures_in_hoard_by_turn` and
`treasure.engine_online_rate_by_turn`. The second is the consistency figure, and it
is unforgiving — a six-card Treasure package in a 99 measured **28.7% online by
turn six**.



`pilot/goldfish.py`: seeded Monte Carlo (seed 42, 10K iterations) simulating **resource development, not full games** — model assumptions are embedded in the artifact and rendered in the manual. Metrics: opening-hand/mulligan stats, land-drop and mana curves, commander-cast turn distribution, per-deck target-set assembly (`goldfish_targets.json`, `any_of` groups, drawn-by-turn semantics), bodies-by-turn (labeled crude). Deterministic: the data-gated test regenerates and compares byte-for-byte.

## Goldfish: two opening-hand distributions

`goldfish_metrics.json` reports **both** `first_seven_land_histogram` and
`kept_hand_land_histogram`, and they answer different questions. The first is the deck's
real land distribution and moves when you change the mana base. The second is that
distribution *after* the keep rule has filtered it, so it sits near 100% inside the 2–5
window for every deck — informative about the mulligan rule, useless as a fitness signal.

They replace a single `land_histogram` key that carried the second while being read as the
first, which made the metric nearly invariant to deck composition. `keep_first_seven_rate`
is unaffected, and by construction it equals the in-window share of the *first-seven*
histogram — if those two ever diverge, one of them is wrong.

## Scenario schema (`stacks/NNN-<kebab>.json`)

```
id, slug, deck, title, rules_version
scenario:   board, hand, mana_available, stack[] (pos 0 = bottom), extras, question
resolution: steps[] {n, action, effect, citations[] {rule, quote}}, final_state
checker:    verdict (pass|fail), iterations, findings[] {step, rule,
            status ∈ supported|unsupported|irrelevant|misquoted, note}
```

**Scenario format.** The keys above are the shape; the conventions are in
`.claude/skills/resolve-stack/SKILL.md` step 1 and enforced as far as they can be
by `validate-stack --scenario-only`. The four that cost real rounds when unwritten:
`hand` is a list and `[]` when empty (never prose — a placeholder sentence was
once read as a card name and shipped into the deck manifest); a permanent already
sacrificed to pay a cost stays LISTED with the `— already sacrificed to pay the
cost of the ability now on the stack` annotation and is NOT on the battlefield;
`mana_available` leads with symbols (`"{0}"` for none, never `""`); and every card
named must resolve against `cards.json` apart from tokens and opponents' permanents.
`extras` is non-normative scaffolding. Run `manamap pilot scenario-facts <slug>`
before authoring — it derives the board split, the per-opponent vs pod-total
arithmetic, deck membership, and which siblings are comparable.

Verdict `pass` requires all findings `supported` **and** the mechanical validator passing. Failed artifacts are saved (they document open questions) but never published.

## Game state v2 — seats that can act (consumed since S4, 2026-08-19)

**Status: consumed since S4 (2026-08-19).** `pilot/game_state.py` holds the vocabulary and
the form check; `validate-stack` runs it on any `version: 2` scenario (preflight and the
loop), `scenario-facts` reads `seats[]` and board objects, and `manamap pilot sim-scenario`
(the bridge, `sim/bridge.py`) WRITES v2 scenarios lifted from Forge games. The resolver and
checker charters carry a paragraph each. It was written down first because three things
already wanted to name a seat the same way (the resolver's
board, `debrief`'s `opponents[]`, `prescribe`'s pod) and the simulation branch will
need a state an *actor* can act on. The audit's finding was that the combat /
interaction gap is a **schema** gap before it is a prompt gap: v1's opponents are
`{life, board}` — furniture that cannot hold priority, cannot block, has no hand.

**Design rules.** (1) **Additive.** A v1 scenario is valid forever; v2 is opted into
per artifact with `"version": 2` inside `scenario`, and every v1 string form is still
accepted wherever v2 allows an object, so a board can be upgraded one entry at a
time. (2) **One `seat` object**, used for you and for every opponent — what differs
between seats is what is *known*, not what fields exist. (3) **CR names, verbatim.**
Phases and steps are the Comprehensive Rules' own words (CR 500.1), so a resolver
citing `506.1` and a state saying `"step": "declare blockers"` are talking about the
same thing. (4) **Hidden information is a count, never a guess.** A seat's hand is
either a list of names (known) or `{"unknown": n}`; an actor only sees what it
should. (5) **Actions are the unit of resolution.** A v1 scenario asks "what happens
when this stack resolves"; a v2 scenario may also ask "what happens when these
actions are taken in order" — and that is the whole difference between resolving a
stack and resolving a turn.

### The shape

```
scenario:
  version: 2
  turn: 4                          # game turn number, 1-based
  active_seat: "you"               # whose turn it is — a seat id
  phase: "combat"                  # CR 500.1: beginning | precombat main | combat |
                                   #           postcombat main | ending
  step: "declare blockers"         # the step within the phase (CR 500.1), or null
                                   #   beginning: untap | upkeep | draw
                                   #   combat: beginning of combat | declare attackers |
                                   #           declare blockers | combat damage | end of combat
                                   #   ending: end | cleanup
  priority: "seat-2"               # the seat that currently holds priority, or null
  seats:                           # ordered in turn order, starting from the active seat
    - seat: "you"                  # id; "you" is the pilot, opponents are "seat-2".."seat-N"
      archetype: null              # free text for opponents ("Dimir control"); the debrief
                                   #   and prescribe use the same field and the same words
      commander: {name, zone: "battlefield|command|library|graveyard|exile",
                  casts: 1}        # casts so far → the tax is derived, never stated
      life: 40
      poison: 0
      hand: ["Craterhoof Behemoth"]          # known — a list of names
      # or   {"unknown": 4}                   # hidden — a count, never a guess
      # or   {"known": ["Counterspell"], "unknown": 3}   # partially revealed
      library: {"count": 61}                  # a count; "known_top": [...] when revealed
      graveyard: []                           # names; v1 already carries this for "you"
      exile: []
      mana:
        available: "{G}{G}{G}{G}{G}{G}"       # v1's mana_available string, per seat
        open: 6                               # untapped sources — the number an opponent
                                              #   actually reads across the table
        pool: "{0}"                           # floating mana, if any
      board: [ <board entry>, ... ]
  stack: [ {pos: 0, item: "…"} | <stack entry> ]   # pos 0 = bottom, unchanged
  actions: [ <action>, ... ]       # OPTIONAL; absent = "resolve the stack" (v1 semantics)
  extras: {…}                      # non-normative scaffolding, unchanged
  question: "…"
```

**Board entry.** A string (v1, still fully accepted — including the house annotation
`— already sacrificed to pay the cost of the ability now on the stack`, which means
LISTED and NOT on the battlefield) **or** an object:

```
{name: "Scute Swarm", controller: "you",
 tapped: false, summoning_sick: false, pt: "1/1", token: false,
 counters: {"+1/+1": 2}, attached_to: null, face_down: false,
 annotations: ["already sacrificed to pay the cost of the ability now on the stack"]}
```

`annotations` is where v1's prose qualifiers go, verbatim, so `scenario-facts`'
reading of "already paid" survives the upgrade unchanged. A v2 `board_bodies` reads
`pt`/`token` from the object instead of regexing the string; the split it reports is
the same split.

**Stack entry.** `{pos, item}` (v1) **or** `{pos, object, controller, source, targets:
[...], kind: "spell|ability|triggered"}` — `item` may be kept alongside as the
human-readable line.

**Action** — what an actor does, in order. Each carries `seat` and one `kind`:

```
{seat: "you",    kind: "cast",      card: "Craterhoof Behemoth", paying: "{3}{G}{G}{G}"}
{seat: "you",    kind: "activate",  source: "Castle Garenbrig", ability: "…", paying: "…"}
{seat: "you",    kind: "play_land", card: "Forest"}
{seat: "you",    kind: "attack",    attackers: [{attacker: "Scute Swarm", defending: "seat-2"}]}
{seat: "seat-2", kind: "block",     blocks: [{blocker: "Wall of Omens", blocking: "Scute Swarm"}]}
{seat: "seat-2", kind: "pass"}                    # passes priority
{seat: "seat-2", kind: "cast", card: "Counterspell", targets: ["Craterhoof Behemoth"]}
{seat: "you",    kind: "special",   text: "turn Hidden Face up"}   # anything else, in words
```

Triggers are never actions — they are consequences the resolver puts on the stack,
which is exactly what the checker's missing-steps audit (triggers, SBA, priority)
already looks for. An `actions` list is resolved left to right, each action followed
by the priority round it implies; the resolver narrates, the checker audits the
steps it skipped, and nothing about the citation contract changes.

### Who reads what

| consumer | reads | today | under v2 |
|---|---|---|---|
| `stack-resolver` / `rules-checker` | board, hand, mana, stack, question | prose board, static opponents | **now**: `seats[]`, `actions[]`; the checker's missing-steps list names combat steps (506–511) by the `step` vocabulary |
| `scenario-facts` | `board_bodies`, `opponents_of` | regex over strings | **now**: the object fields (`pt`, `token`, `annotations`, `type`), falling back to the string regex per entry |
| `sim-scenario` (the bridge) | a Forge game's events up to a cut | — | **now**: writes a v2 scenario — life exact, lands exact (tapped since the controller's last untap), cast permanents from resolve lines, tokens from first use, commander exit read as `command`, hand as `{unknown: n, estimate: true}`; every approximation in `extras.reconstruction_notes`; `question` empty on purpose |
| `goldfish` (`model_combat`) | one opponent, 40 life, does nothing | internal | could emit a v2 `seats[]` snapshot at turn N |
| `debrief` | `opponents[].seat / archetype / commander` | free text | the same three words; `seat` ids are the vocabulary for "the Dimir player" |
| `prescribe` | pod description in the prompt | free text | may carry `seats[]` with `archetype` only — the doctor reasons about a pod, not a board |
| `opponent` (post-MVP) | everything its seat may see | — | the actor: given a state with its own hand known and others `{unknown: n}`, emits the next `action` |

### What it deliberately does not do

- **No rules engine.** The state is a description; the rules are in the CR and the
  checker. A v2 scenario is resolved by the same loop with the same citation contract,
  and the checker's verdict is still atomic over the artifact — so **one rules domain
  per scenario** still governs, and a five-action turn across combat, triggers and
  layers is still five chances to fail in one file. Split it.
- **No probabilities.** A v2 state is one board. What happens *on average* is the
  goldfish's job; what happens *here* is the resolver's.
- **No migration of the 49 passing stacks.** They are evidence; their scenario blocks
  are cache fingerprint inputs; touching them would MISS every stack routine to change
  nothing a reader can see (the same argument that left `object`/`item` both accepted).
- **The first writer was the bridge, not a hand.** The consumers (`validate-stack`,
  `scenario-facts`) landed the same commit as `sim-scenario`, tested against a board lifted
  from a real Forge game rather than a fixture nobody authored for a reason. The opponent
  actor is still last.

## Scenario scope, and why it is the loop's main cost lever

The checker's verdict is atomic over the whole artifact, so every citation is another
chance for all of it to fail. Measured across three published decks: every artifact at
**≤32 citations passed in 1–2 rounds**; every one at **≥59 needed 4 rounds or failed**.
goblin-storm's five narrow scenarios produced 5 verified lines in 6 rounds; sisay's three
broad ones produced 1 in 9, and sisay 003's answers (a)–(d) were verified correct three
times before being discarded with the file.

`RESOLVE_SCOPE_BUDGET` (config.py, and actually imported) warns above 12 steps, 40
citations, or 3 lettered sub-questions. `validate-stack --scenario-only` runs the
sub-question check **before** a resolver spawn, for free. The rule: **one rules domain per
scenario**; split multi-part questions into separate artifacts so they fail independently.

## The resolve loop (agents)

Run via the `resolve-stack` skill: `stack-resolver` agent drafts → `validate-stack` (mechanical gate, short-circuits on form errors) → `rules-checker` agent verdict → re-spawn resolver with findings while iterations < `RESOLVE_MAX_ITERATIONS` (3). Agents are read-only; the orchestrating session writes files. Batch scale-out (many scenarios in parallel) is a Workflow-tool upgrade path.

**Definition of done**: run `/resolve-stack` on a scenario; confirm the saved artifact passes `manamap pilot validate-stack` and the golden-artifact test (`tests/test_pilot_validate_stack.py::test_all_committed_stacks_validate_and_pass`) unskips and passes.

## Rules DB

One chunk per numbered CR rule — **chunk ID = rule number = citation ID** — plus `glossary:<term>` chunks. `Example:` and continuation lines attach to the owning rule, so quotes from examples satisfy the contract. Embedded text is prefixed with `id + section title` (helps MiniLM find "storm" for 702.40a, whose text never says storm); stored text is verbatim CR. Embeddings are L2-normalized MiniLM (reuses `compute_text_embeddings`); row i ↔ `order[i]`.

**CR refresh** (each set release): get the current TXT link from https://magic.wizards.com/en/rules, update `CR_RULES_URL` in `src/manamap/config.py`, run `download-rules` + `build-rules-db`. Artifacts record their `rules_version`.

## Strategy DB (`data/strategy/`, tier ★ grounding)

The strategic counterpart to the rules DB: `strategy.md` is a tracked, sourced
companion doc of expert theory (resource pillars, role assignment, information
play, combat math, Commander multiplayer dynamics, schools of thought), chunked
and embedded exactly like the CR — **section ID = citation ID**
(`strategy:<slug>[.<child>]`, `STRATEGY_ID_RE` in `common.py`). Heading format
`## strategy:<id> — Title`; every section ends with a `Sources:` block
(`- Author, "Title" — URL`, URL verified or `(print)`). `CHANGELOG.md` logs every
amendment (`added|amended|renamed|deprecated strategy:<id>` bullets, mechanically
checked). Enforcement mirrors the citation contract: `validate-strategy` enforces
form in code; substance is founder-reviewed via `git diff data/strategy/`. The
index records the doc's sha256 — `load_strategy_db` refuses a stale DB, so
rebuild after any doc edit. Derived index/embeddings are gitignored; the doc and
changelog are tracked.

Strategy content is **curated grounding for tier ★**, not a fourth tier: coach
and writer prose may reference `strategy:<id>` sections, and decision-branch
citations may cite them under the same verbatim-quote contract (`validate-stack`
dispatches on the `strategy:` prefix), but a strategy citation never makes a
claim rules-verified.

**The strategy-researcher agent** (two modes, stated in its prompt):
- `MODE: research` — the only write-scoped pilot agent (strictly
  `data/strategy/` only). Searches online sources (articles, reddit,
  transcripts — video only via transcript), verifies every URL it cites,
  amends the doc, appends one changelog entry per pass. Run via the
  `research-strategy` skill: spawn → scope guard (`git status --porcelain`,
  revert strays) → `validate-strategy` (≤3 iterations) → `build-strategy-db`
  → founder reviews the diff.
- `MODE: consult` — read-only strategic feedback on board states, cards,
  combos, and decks; must RAG-query before answering and cite `strategy:<id>`
  for every framework claim. Produces the **strategic frame**
  (`data/decks/<slug>/strategic_frame.json`, tracked): archetype, schools,
  role assignment, engine map, candidate missing lines (flagged "needs a stack
  scenario", feeding the resolve-stack queue), matchup frames, gaps (feeding
  the next research pass). The write-manual pipeline generates it after the
  evidence pull; pilot-notes and the engineer consume it.

## Deck facts — the brief agents read instead of re-deriving

`manamap pilot deck-facts <slug>` composes existing primitives (`extract.get_colors`,
`manabase.count_pips`/`land_colors`, `bracket.combos_in_deck`, `card_roles.json`) into
one deterministic answer. Computed on demand and printed to stdout, **never committed** —
same rule as `artist-credits`: a second copy of facts already in `cards.json` could only
desync.

It reports counts (entries *and* copies), the mana-value curve, per-card colours resolved
correctly for multi-face cards (both the card's union and the face-up permanent's),
per-colour pip load and source targets, role coverage plus the cards the taxonomy has no
pattern for, every combo line fully contained in the deck — and a `notes[]` block that
pre-answers the traps agents kept rediscovering:

- how many synergy edges actually fall **inside** this deck (0 on sisay, 213 on
  edgar-vampires — it is a global top-10 shortlist, so report the number rather than
  assuming either way)
- which cards have no `card_roles.json` entry, with the standing caveat that absence of
  a role is not absence of the function
- **restricted mana, classified precisely.** "Spend this mana only" means three different
  things: `spells_only` (Delighted Halfling, Unclaimed Territory — cannot pay an
  activated ability, because an ability is not a spell), `pays_abilities` (Secluded
  Courtyard, whose clause says "or activate an ability"), and
  `has_unrestricted_coloured_mode` (Plaza of Heroes). An unrestricted `{T}: Add {C}`
  does **not** count — colourless pays no coloured pip. Getting this wrong is worse than
  saying nothing: sisay's strategic frame asserted Secluded Courtyard was dead to its own
  commander, and it isn't.

**The baseline builds a curve, and finishes a combo line it half-holds.** `fill_slots`
fills a per-ROLE quota (`DECK_ROLE_BUDGET`) crossed with a per-MANA-VALUE quota derived
from `DECK_AXIS_TARGETS["curve"]` — the cited target `deck_audit` already measured
against. Without the second quota the builder scored every card independently and took
the top N, and since `curve_fit` penalises every point above mana value 3 the top N were
always cheap: kinnan's first baseline had **nothing above 3** across 64 nonland cards,
29 of them mana producers, and `validate_build` passed it because form is not substance.
A role whose shape-fitting candidates run out still gets its slots — a legal deck of the
right size beats a perfect curve.

`complete_combos` then swaps in the ONE missing card of a combo line the deck already
half-holds, bounded by `DECK_BUILD_COMBO_COMPLETIONS`. It reads real lines from
`combo_details`, never the flat `combo_partners` map: that map is co-occurrence, so
"partners with something present" is true of a hundred cards once the commander is on
the list and cannot tell a completion from a coincidence. A completion is allowed to
score *lower* than what it replaces, because the one-shot score is exactly the thing
that cannot see a pair — kinnan's baseline held 23 Kinnan partners and zero completions.
The swaps are surfaced in `build_plan.json` under `combo_completions`, like
`cut_for_bracket`, because a swap made for a reason the score does not show must be
readable.

## Pool facts — building from parts

`manamap pilot pool-facts <paths…>` answers the question a physical collection asks:
*what deck should I build from these cards?* It takes files or directories rather than a
slug, on purpose — a collection is not a deck, and putting one in `data/decks/<slug>/`
would place it in reach of validators that assume a legal 100. `deck-facts` on a 764-card
box reports hypergeometrics against a 99-card library and `validate-deck` emits roughly a
thousand errors; both answer a question nobody asked. Output is **computed on demand,
never committed**, the same rule `deck-facts` and `artist-credits` follow.

It reports per-source contribution (what a box supplies that nothing else does), name
resolution including the front-face → `" // "` translation, every legal commander in the
box, per-identity depth **and** castable sources, role coverage against `DECK_ROLE_BUDGET`,
fully-contained combo lines, the bracket floor, in-box upgrades from
`obsolescence_index.json`, and mechanical-tag concentrations.

**The commander list is ranked by DEPTH, and `edhrec_card_rank` is not a quality
signal.** `cards.csv`'s `edhrec_rank` is a card's popularity across the whole format in
every role, so a legend played mostly in other people's 99s outranks a genuine commander —
Selvala, Heart of the Wilds is card rank 430 and commander rank **#448**, while Atraxa is
commander rank **#4**. It is the shortlist's *tiebreak* only (every commander in one
identity sees the same pool, so they tie on depth), the key is named `edhrec_card_rank` so
the label cannot lie, and `build_notes` emits the caveat into `notes` so it travels with
the JSON rather than only the printed report. Depth is what the box holds, not what the
deck would be worth.

Three things it exists to get right, each learned the expensive way on the first real box:

- **Depth is not castability.** Depth — owned cards inside a commander's colour identity —
  ranked Atraxa first at 663, the deepest in the box. Her W and U have 10 sources each
  against B's 44. Ranking a shortlist on depth alone recommends a deck that cannot cast
  its own spells, and does it confidently. Both numbers are reported per commander, and
  `notes[]` names any identity where they disagree.
- **Count sources with `manabase.land_colors`.** A hand-rolled count — `{U}` appears in the
  oracle text, or the type line says Island — put that same box at **1 blue source**. The
  real figure is **10**: a bulk collection's fixing is overwhelmingly generic (Command
  Tower, City of Brass, Exotic Orchard, Path of Ancestry, Ash Barrens, tri-lands), and
  none of those name a colour. A factor-of-ten error pointing the wrong way, killing an
  archetype that was live. `land_colors` is also restriction-aware, so a Dragon-only land
  is not counted as five sources.
- **Dedupe combo containment.** `combo_details.json` carries several records per
  interaction, so a straight containment read double-counts — 33 lines where the box holds
  31. Dedupe on `frozenset(cards)` and keep the most popular record.

Every line it reports carries `verified: false`. Containment is not verification:
Commander Spellbook is format-agnostic, its bracket tags are not gospel, and a line only
becomes evidence after a resolve-stack run.

## Deck audit — is this deck any good? (`deck-audit`, tier ◆)

Five commands measure a deck and nothing joined them. `deck-facts` reports composition,
`mana-analysis` castability, `goldfish` speed, `bracket-check` power
what is better out there. Ask "is my card draw enough" and nothing answered.

`deck-audit` is the join, and it is **computed on demand, never committed** — it embeds
goldfish and bracket figures, so a tracked copy would be a second source of truth that
goes stale the moment the decklist moves. Two blocks:

**Sixteen axes**, each `{measured, target, verdict, gap}`. The point is not the arithmetic
— every figure already existed somewhere — but that each target carries the **verbatim
quote** from `strategy.md` that supports it, so an agent cites a number instead of
inventing one. `DECK_AXIS_TARGETS` in `config.py` holds them, and
`tests/test_pilot_deck_audit.py` fails if any quote drifts out of the doc. That is the
gap `DECK_ROLE_BUDGET` was built to have: one flat uncited budget handed to every deck,
its own comment calling it "PROVISIONAL", `upgrade_facts` printing its shortfalls as
"Context, not evidence". `DECK_ARCHETYPE_BUDGETS` varies the targets per archetype from
`strategy:deckbuilding.archetype-selection`'s own spread, and the archetype is taken from
`strategic_frame.json` or `--archetype` — **never guessed from the cards**, because a
budget silently attributed to the wrong archetype is worse than no budget.

Three details that cost a fleet survey to find:

- **Burgess's land formula budgets *sources*, not lands.** Applied to the land count
  alone it asks a five-colour deck with a nine-mana commander for 45 lands. So
  `mana-base` takes the conventional 36–38 band and `mana-sources` takes Burgess,
  counting lands plus persistent producers (rocks, dorks, land ramp — rituals and
  Treasures are one-shot and are not sources).
- **Aggro's "26-32" is a creature count**, not a finisher count. Overriding
  `threat-density` with it told edgar-vampires it was thirteen finishers short.
- **An axis count is a floor, and the audit says which cards make it one.** Oracle-text
  probes name cards showing an axis's function that the taxonomy filed elsewhere —
  `card_roles.json` calls Yawgmoth, Thran Physician `removal:debuff` and his ability
  draws a card per activation. The probes never change a count; they stop an agent
  reading UNDER as a finding when it is a question.

**Engine activation.** `goldfish_targets.json` is already a machine-readable declaration
of what the deck is trying to assemble and nothing had ever read it as one. Its
`need: [{any_of: […]}]` groups ARE the engine's components, and a group's size IS that
component's redundancy — priced through `manabase.hypergeometric_at_least` (which
reproduces `strategy:deckbuilding.redundancy-vs-tutors`'s cited 31% / 41% / 54%, asserted
by test), set beside the rate the simulation measured. The thinnest group is where the
deck fails first, and "what would activate the engine" becomes "which pool cards would
join that group": by shared role signature, or — when the component is a named combo half
with no shared role — through `combo_details.by_card`. **The role route needs a SHARED
role, not a modal one**: run off one card's roles, a component holding only Blowfly
Infestation returns Massacre Wurm and Dismember, because the roles describe the card
rather than the group's job.

Reported honestly and never papered over: each target is an AND of ORs, so the schema
cannot express the UNION of several independent kills. A deck with four kills has no
single assembled rate, and averaging them would invent a number the simulation never
measured.

## The diagnosis (`diagnosis.json`, tiers ◆ + ★)

`deck-doctor` ⇄ `deck-skeptic`, bounded at `DIAGNOSE_MAX_ITERATIONS = 3` like the other
two loops, driven by the `/diagnose-deck` skill. The doctor is adversarial toward the
deck; the skeptic is adversarial toward the doctor. Output: an axis-by-axis reading, the
engine's single points of failure, `lean_into`, a ranked `add_candidates`, an argued
`cut_candidates`, and `open_questions` carrying a `settled_by` that routes each one back
into `/resolve-stack`, `/research-strategy` or a goldfish-target edit. **Analysis-only** —
nothing in the loop edits a decklist.

`deck-doctor` has two modes. **MODE recon** is the only place in this subsystem that
touches the web: it fills a hole `docs/history/deck-builder-v2.md` names outright — there are no
per-commander inclusion rates in any bulk data we have, and inclusion rate is the real
staples signal. Its `deck_recon.json` is dated (`as_of`) and deliberately kept **out of
`strategy.md`**: durable theory and perishable meta claims must invalidate differently,
the lesson recorded when `meta-analyst` was traded away. Its cache routine `deck-recon`
is therefore the one routine in the registry whose staleness is **time**, not inputs —
its declared input is the brief, and `RECON_MAX_AGE_DAYS` is judged by the skill, because
`deck_audit` is deterministic and never reads the clock. **MODE diagnose** is strictly
read-only and artifact-grounded; recon is evidence there, never authority.

`validate_diagnosis.py` recomputes rather than trusts: every `axes[].measured.value` is
re-derived from `deck-audit`, every citation goes through the shared verbatim gate, every
`bracket_delta` is recomputed through `bracket.assess()`, and — the check nothing else in
the repo performs — **`orphans_stack` is computed**. If a proposed cut names a card that
appears in a checker-passed stack's scenario, the entry must list those stack ids. That is
the Ophiomancer / South Wind Avatar class of finding made mechanical: a cut list will
otherwise propose the one card a verified line rests on, in a confident sentence. The
probe reads the **scenario block only** — a checker note may discuss a card the board
never held, and a discussion is not a dependency.

No L10 rule applies, deliberately: the diagnosis is a working artifact and is never
rendered into an issue. It may name a weakness plainly, which is the one thing
every-issue-is-the-reader's-first would forbid.

## The engine (`engine.json`, tiers ✓ ◆ ★)

The constellation's own limit, found by the cartographers and then measured: **a card is
clustered by what it SAYS, and an engine is what cards DO TO EACH OTHER.** On radagast only
**4 of 10** declared components sit in a single city — the metronome class and the flash
traps span five each — so a city name is the wrong address for a component.

`manamap pilot engine-facts <slug>` is the deterministic brief: `deck_audit.engine_activation`
(components already priced hypergeometrically), the verified pairings from checker-passed
stacks via `build_index.line_cards`, the contained combo lines deduped on `frozenset`, and a
**scatter table** so the agent starts from the disagreement rather than discovering it.
Computed on demand, never committed.

`/analyze-engine` runs `deck-engineer` ⇄ `engine-critic`, gated by `validate-engine`, into an
eight-stage model: `mana · ignition · fuel · fodder · conversion · output · protection ·
wincon`. Not every deck has all eight — radagast has no `fodder` because nothing in the 99
sacrifices, and saying so is a finding.

**The evidence ladder is the whole job.** A checker-passed stack is the only fact. A contained
combo line is a candidate stamped "needs a stack scenario". A role is a property, not an
interaction. The synergy graph is retrieval only and is deliberately absent from the brief.
`lines[].verified_by` is nullable for exactly this reason, and the renderer draws a null one
**dashed**.

**The figure is a schematic, not a block diagram.** Every arrow is labelled with what it
carries — the line's own `carries` when the engineer authored one, and otherwise DERIVED from
the source stage (`design.STAGE_CARRIES`). Deriving costs no schema change and no respawn of
this loop; derived labels render italic and the caption counts them, because an inference
wearing an authored label is exactly what the dashed line exists to prevent. A forward arrow
arcs above the rail and a backward one arcs below as a feedback loop, which is what makes an
engine an engine rather than a list of steps. Each stage also carries a plain-language job
(`STAGE_ROLE`), and each card in The 99 wears its stage as a chip inked from the same
`ENGINE_STAGE_INK` — the chip annotates the grid and never regroups it, because the engine is
measurably not the clusters.

**What the gate cannot see, stated because it matters:** `validate-engine` checks that a
cited stack NAMES a line's cards; it can never check that the stack SUPPORTS the line. Two
real radagast lines passed every mechanical check while citing a stack that showed the
opposite — one claimed Castle Garenbrig paid for Craterhoof, citing a stack that leaves
Garenbrig untapped. Both rendered as solid green, the mark for proof. A passing stack is
evidence a BOARD resolved a certain way; reading it as causation is inference, and inference
is the critic's job. Do not close it with string matching — the same wrong line survives a
rephrase.

## The constellation (`deck_map.json`, tier ◆ + ★ names)

`manamap pilot deck-map <slug>` re-lays-out ONE deck's cards from `embeddings_ability.npy` —
the FUNCTION space; the layout space knows only colour and type, so a mono-green deck
clusters there into a green blob and a land pile — and cuts two levels of cities and
neighbourhoods. It is `viz/js/drill.js`'s argument applied to a decklist: a hundred cards
scattered across the 34,890-card atlas are dust, because the structure that matters is
exactly what a global projection compressed out.

**Tracked**, because the embeddings are gitignored and a fresh clone must still render
manuals — the same argument the projections are committed under. **Positions are LOCAL** and
are not atlas positions; everything that draws it says so.

Three parameters were measured rather than assumed:

- **Ward, not average linkage.** Average linkage on cosine distance chains: on radagast it
  put 37 of 71 cards in one city and 1 in another.
- **The city count is chosen by BALANCE**, not by a cards-per-city divisor — that divisor
  still put 54% in one city. Grow k until the largest holds under 35%, stop at seven, because
  past seven regions a reader consults a key instead of seeing a shape. (Ward on radagast:
  k=4 54%, k=6 42%, k=7 32%, k=9 14%.)
- **Territories draw per NEIGHBOURHOOD.** A spread-out city's convex hull covered every other
  city and the map read as one continent with labels floating on it.

`deck-cartographer` then names each region for the job its cards do, and `merge-deck-map`
writes **`label` and `gloss` and nothing else** — positions and membership are a measurement,
and a whole-file copy from `.agent-out/` would let a model's paraphrase silently replace the
map. `validate-deck-map` checks names are distinct within a level and that membership still
totals.

## Decision scenarios (`decisions/NNN-<kebab>.json`, tier ★)

`kind: "decision"` artifacts: archetypal board + table state, a decision question, 2–4 branches each with `choice`, `line`, `signals`, `coalition_risk`, `coaching`, optional `citations` (same verbatim-quote contract), and a `recommendation` matching a branch. Mechanically form-checked by `validate-stack`; substantively reviewed by humans — the tracked JSON is the red-line surface. Authored via the `pilot-coach` agent (`author-decision` skill).

## The tutor guide (`tutor_guide.json`, tier ★)

One wish per tutor. `pilot-notes` authors an entry for every maindeck library-search
tutor — scenario → the exact card to fetch → why — and `validate-tutor-guide` holds each
one to the deck and to that tutor's own search constraint, **per clause**: a DFC or
chapter card can carry several search clauses (Huatli's front face fetches a basic land;
Roar III fetches Dinosaurs), so a fetch is legal if any clause permits it. Pure land ramp
(Cultivate, Nature's Lore) is excluded — that is the mana analysis's business. A deck with
no tutors reports `N/A` and the legacy page prints standing copy.

## The mana analysis (`mana_analysis.json`, tier ◆)

The mana audit — rendered as *Sources Say* by the legacy page — and the one with **no agent at all**: `manamap pilot
mana-analysis <slug>` computes it deterministically, reusing the deck-builder's own
hypergeometric kit (`manabase.py`). Land classes, per-colour land and nonland sources,
pip share vs source share, on-curve probability with and without rocks, the ramp census,
and a stated-assumptions block. (The legacy `mana_base` prose key that narrated it is retired and frozen.)

**Count copies, not decklist entries.** `cards.json` stores basics as one entry with
`quantity: N`, and counting entries once published "18 lands" for a 33-land deck and
understated every colour's sources fleet-wide. `common.expand_copies()` is the shared
primitive; `lands.total` is copies and `lands.entries` is distinct cards, both reported so
they can never be confused again. Three guards: a unit fixture (11 Islands = 11 blue
sources), a staleness test recomputing every tracked artifact, and a legacy `validate-issue` lint
rejecting reader-facing copy that quotes the entry count as a land count.

**The trap this exists to catch.** Sazacap's Brew is tagged `buff:pump` because its text
contains "+2/+0", and Vol. 001 shipped advice to test it in the Witch's Mark slot. Both are wrong: the Brew's first target is a *player*, so
Zada — which copies instants targeting **only** Zada — never copies it, while Witch's Mark
targets a creature and is copyable. Reading the card rather than its role tag inverted the
recommendation, and the published prose was corrected to match. That is the whole value of
the pass.

## Agent invocation cache

Subagent spawns are the only real cost here (the renderer is free and deterministic —
there are **no LLM calls in Python at all**). A full manual regeneration is ~330k
tokens across four serially-dependent agents, so every skill that spawns one checks
first:

```
check → (miss) spawn → write → validate → record
```

`manamap pilot cache-status <slug>` reports per routine — `HIT`/`EDITED` exit 0 (don't
spawn), `MISS` exits 1 (spawn), a missing required input exits 2 (stop). Records live
in `data/decks/<slug>/.agent-cache.json` (**tracked**, so a `git pull` transfers
someone else's regeneration as a cache hit, and `git log` answers "which inputs
produced this prose?"). `record()` refuses artifacts that are missing, lack their
routine's keys, or have no checker block — a failed run can't poison the cache.

Routines (10 static): `candidate-pool`, `deck-build`, `deck-diagnosis`, `deck-recon`,
`deck-engine`, `deck-map-names`, `debrief` (N/A until something is logged),
`strategic-frame`, `pilot-notes` (five keys of `manual_prose.json`), `tutor-guide`
(the tutor guide — `N/A` for a deck with no library-search tutors, via the applicability
gate in agent_cache), plus `prescription:<id>` (one question to the doctor; `prompt:self` digests only the authored question) and `stack:<NNN>` and `decision:<NNN>` discovered
from disk. Declared in `config.AGENT_ROUTINES`.

The two build routines take **no `cards:semantic`** — it digests a `cards.json`
that by definition doesn't exist before a build, so the authored `brief.json` is
their root input instead. Conversely a hand-built deck has no `brief.json`, so
those routines report **`N/A`** in the all-routines scan rather than aborting it;
an explicit `--routine` still exits 2, because there you asked about that routine
specifically and a missing input means fix it, don't spawn.

`validate-build` checks the role budget **per role**, not just in total — a budget that
sums correctly while every line is wrong is not a budget — and cross-checks the plan's
self-reported bracket floor against `bracket_report.json`, the `lands` array against
`land_counts`, and the mana base's `spell_slots` stamp against the current slot count so
diagnostics computed for a deck you no longer run are rejected.

Four semantics worth knowing: agent prompts are inputs (editing
`.claude/agents/*.md` invalidates that agent's routines by design); `issue-plan`
hashes prose *structure* not wording, so a typo fix is free but a new section
re-plans; `strategy:doc` hashes `strategy.md` bytes so `build-strategy-db` never
invalidates anything; and `stack:<NNN>` hashes only its own scenario slice so the
resolver/checker loop can't self-invalidate. Full sizing and rationale:
`docs/agent-cost.md`.

`build-manual` is deliberately **uncached** — already $0 and deterministic.

## Agent handoff

Deck agents write their JSON to `data/decks/<slug>/.agent-out/<agent>.json` (gitignored)
and return only that path plus a short summary. The orchestrator reads, validates, and
merges into the tracked artifact. `candidate_pool.json` reaches 133 KB — returning it
inline costs ~35k tokens of orchestrator context for nothing, and the agent's tools are
unchanged either way.
## Tests

`tests/test_pilot_*.py` — 42 files, the largest group in the suite. **The inventory lives
in `docs/testing.md`** (what each file covers) and so do the counts, which that file
declares itself the only home for.

This section used to restate both, and ignoring that rule is exactly how it drifted: it
claimed 42 cases for `test_pilot_build_manual` (91), 29 for `test_pilot_validate_issue`
(51) and 57 for `test_pilot_agent_cache` (83), while omitting twenty-odd files entirely.
A number restated in two places is a number that will disagree with itself.

Data-gated tests use `requires_rules` / `requires_deck` / `requires_strategy` /
`requires_roles` markers from `tests/conftest.py`.

## LEGACY — the magazine renderer (frozen; replaced by `docs/manual-v5-spec.md`)

Until 2026-08-19 each deck was published as an **issue** of a magazine, *Pilot's Manual*.
The renderer (`build_manual.py`, `issue_spec.py`, `design.py`, `validate_issue.py`) still
runs and still renders the nine decks from the artifacts it reads — its constitution,
`STYLEv3.md`, was deleted on 2026-08-25 and lives in git (`git show 23e8cec:STYLEv3.md`), and those artifacts — `issue_plan.json`, the panel keys and `card_roles` /
`mana_base` / `upgrades` in `manual_prose.json`, `considering.json` + its art sidecar —
are **frozen**: no agent regenerates them (`magazine-editor`, `pilot-panel`,
`manual-writer`, `pilot-coach` and `short-list-analyst` are retired), the cache has no
routine for them, and the compact deck page in `docs/manual-v5-spec.md` replaces the whole
layer. Everything below is kept because it is an accurate account of that code and of the
lessons it cost to learn — the length measurements, the theatre, the voice lint — not
because any of it is the product.

### The magazine layer (STYLEv3)

Each deck is a complete **issue** of *Pilot's Manual* — a fixed set of sections in a
fixed order (see `issue_spec.DEPARTMENTS`; never transcribe the list or its count into
a prompt), grouped into five acts that ramp from what to do, through tactics and the
long game, into the numbers and the proof. Readers learn the publication once and
navigate it forever. Every section is signed by one of three columnists — `"Ledger"
Lin Marginal` (◆), `Counselor Vera Dictum` (✓), `Coach Sunny Brightside` (★) — and
STYLEv3 L10 holds that every issue is the reader's first: no version numbers, no
changelog voice, enforced by `validate_issue.validate_self_containment()`. The design
authority was `STYLEv3.md` (editorial laws, the Commander Mandate, section specs, voice,
component library), with `STYLE-v1-visual-research.md` and `-v2-editorial-method.md` as
its sources. All three were deleted on 2026-08-25 — `git show 23e8cec:STYLEv3.md` and
`git show 23e8cec:docs/history/<file>`. The `STYLEv3 §N` citations left in the renderer's
own comments still say which clause each piece implements; they simply resolve through git
now, which is the right place for the constitution of something nobody is allowed to
extend.

- **`src/manamap/pilot/issue_spec.py`** — the canonical department system: ids, order,
  promises, evidence tiers, rhythm tags, component library. Changing it changes every
  issue; treat it like `config.py`.
- **`issue.json`** (tracked, **authored by a human**) — volume, issue_date, cover_price,
  deck_name, commander, cover_tagline, next_issue. Never generated: a generated date
  would break byte-identical rebuilds.
  - Optional **`status`** — one of `issue_spec.ISSUE_STATUSES` (`broken-down`,
    `superseded`, `retired`). An issue is a **published record**: when the deck it
    describes stops existing, it is MARKED, never edited or deleted. §5.1's rule
    against editing a passing artifact post-hoc applies to a whole magazine as much
    as to a stack, and every figure in a retired issue was true when it shipped.
    Set, it renders a banner above the cover and mutes-but-keeps the newsstand card;
    absent (the default) it renders **nothing**, so live issues stay byte-identical.
    An unknown value renders nothing and is reported by `validate-issue` — a typo
    must not be able to take a magazine offline, nor silently read as live.
    First use: `hapatra` (Vol. 002), broken down for parts so its aristocrats shell
    could be sleeved into `yawgmoth-swarm`; the two lists share **27** nonbasics.
- **`pending.json`** (tracked, **hand-authored**, optional) — the queue of changes
  DECIDED but not yet applied. The repo had two homes for a swap and neither could
  hold an intention: applied swaps are derived from git, and `considering.json` is
  fixed at exactly ten entries, forbids a pick already in the deck, and is
  regenerated wholesale on any decklist edit. A three-land swap decided in
  conversation fell through that gap and was lost, which is why this exists.
  - Entries carry `id` / `decided` / `why` plus **list-valued `in` and `out`**, so a
    three-for-three swap is one decision rather than three picks that each look
    wrong alone. `settled_by` names the routine that closes it, reusing
    `open_questions`' routing vocabulary.
  - **Closure is DERIVED, never declared** — there is no `applied: true` field,
    because a hand-set flag is exactly how `HISTORY.md` became append-only and
    append-forgotten. `state_of()` reads the deck: **the cuts decide it, not the
    additions**, since a card the deck already runs cannot prove it just arrived.
    An APPLIED entry is deleted rather than ticked; git owns it from then on.
  - **PARTIAL is a signal, not an error.** A cut no longer in the deck means either
    "applied" or "left for another reason" and nothing separates them, so the
    validator declines to guess. There is deliberately no stranded-cut check.
  - **Not a cache input.** Declaring it would MISS agent routines for a decision
    nobody has acted on; intent must not invalidate content.
  - Report-only, and never read by the renderer — a queue of unmade changes is the
    "previous build" framing STYLEv3 L10 bans from print.
- **`issue_plan.json`** (tracked, human-editable) — the packaging layer from the
  `magazine-editor` agent: the issue's angle, cover lines, per-department
  kicker/headline/dek, captions, PILOT TIPs, callouts, pull quotes, roster grouping,
  threat boxes, sample hands. `manual_prose.json` remains the body-copy layer; the
  renderer merges them.
- **`validate-issue`** — the mechanical gate: identity block complete (including a
  `decklist_sha256` that must match `cards.json`), every section present in canonical
  order, copy completeness, components from the fixed library, **tier costume never
  overridden**, every PILOT TIP / caption / roster card name real, no two dense
  sections adjacent unless a breather is declared (`BREATHER_AFTER`), no changelog
  voice (L10), and no reader-facing copy quoting `lands.entries` as a land count.
- **`magazine-editor` agent and `design-issue` skill** — RETIRED 2026-08-19; the nine
  `issue_plan.json` files are frozen inputs the renderer still reads.

The Kill renders feature spreads with dossier pointers; **Judge's Desk** carries the
complete resolutions with every citation verbatim (the renderer may not summarize proof)
as a **case index** — one scannable row per case, expanding to the unchanged record. The
proof is printed in exactly ONE place: the theatre prints a citation COUNT and points at
the case, because shipping it with the quotes inline put the identical 120 citations into
both departments.

**`the-kill.features` decides which lines get a theatre.** An ordered list of stack ids;
everything else prints under *Also on the record*, keeping its whole authored intro and
its result and losing only the staging. Omit the key and every presentable stack features,
which is right up to about seven and wrong past it — yawgmoth-swarm has eleven and its
Kill reached **44,119 words, 42% of the issue**, since its loops run 11–14 steps and each
was staged. Featuring four took it to 19,104 words and 20.4% of scroll, the same share The
Kill takes on a seven-stack issue.

Two measurements set that design. A rendered stack is ~4,000 words and its authored intro
is **77–144**, so the intro costs nothing and an index that dropped it would keep the
department's title while cutting the thing it names. And **word count is a bad proxy for
scroll here**: −25,015 words bought only −8.1 screens, because the theatre stacks its
plates in Z and is word-heavy, pixel-light.

`validate-issue` fails a `features` entry naming a non-presentable stack, a repeat, and a
list naming every presentable stack (that is what omitting the key does, and it rots the
first time a stack is added). The renderer instead **skips** an unknown id, because a crash
there turns a copy mistake into a missing magazine. An indexed row carries the `line-<id>`
anchor Judge's Desk links back to, or every case's *↩ Back to this line in The Kill*
becomes a dead jump.

#### Length is measured (`issue-length`, `PROSE_BUDGET`)

`manamap pilot issue-length <slug> [--rendered]` reports words and visible words per
section — visible excludes anything inside a collapsed `<details>`. The gap is the
point: Judge's Desk was 21% of Vol. 009's words and 2.4% of its scroll, so a single
number sends you to cut the wrong department half the time.

`issue_spec.PROSE_BUDGET` caps each prose key at a length at least one deck already
achieves. `validate-issue` reports breaches; **`--strict` fails on them**, so the gate
is real for new work without turning eight pre-budget artifacts red. The two
deliberate exceptions — `threat_assessment` and `matchups` at 2,500 where the fleet's
shortest are 3,821 and 4,129 — take their number from
`validate_engine.MAX_WHAT_IT_DOES` instead, and are the debt the Act III merge left when
it combined three departments' headers without touching their prose. That merge is now
complete on all nine decks and the three ids are deleted, so the debt is the prose and
nothing else.

**The Kill's stack theatre** (`design.stack_theatre`) renders a resolution as a
receding stack of plates on a vanishing-point grid — one plate per step, hover to
lift, a tab to bring one forward with its action, effect and citations. It is
**CSS-only**: an issue is a standalone printable file with no scripts, so the
mechanism is radio inputs and `:checked ~` selectors, the depth is
`transform-style: preserve-3d`, and step 1 is `checked` in the markup so CSS-off,
print and screen-reader readers open on a valid view rather than a blank stage.
Per-index rules are generated into the stylesheet (`_theatre_rules`, bounded by
`THEATRE_MAX_STEPS`), because per-instance CSS would put a `<style>` block inside
every case. It does not replace Judge's Desk and must not: the theatre is a way
*through* the proof, and §5.1 forbids the renderer summarising proof. The Command Zone department is mandatory and format-specific — the
tax ladder, color identity, the 21-damage clock — and is what makes this a Commander
magazine rather than a Magic one.

### The legacy render pipeline

The `write-manual` skill still drives it — goldfish → `deck-analyst` evidence pull → **strategic frame** (`strategy-researcher` MODE consult → `strategic_frame.json`; its `candidate_missing_lines` feed the resolve-stack queue, its `gaps` feed the next research pass) → `pilot-notes` (the five prose keys + decisions + the tutor guide, receives the frame and `engine.json`; since 2026-08-19 one agent in one voice replaces the coach + writer pair) (zero-guessing: combo lines only from verified stacks, claims trace to graphs/oracle text; receives the frame) → `manual_prose.json` (tracked, human-editable) → `manamap pilot build-manual <slug>` + `build-index` (deterministic, byte-identical rebuilds, `[TODO]` placeholders for missing prose, only checker-passed stacks render).

### The front of the book (`editors_letter`, `pilots_log`, tiers — and ★)

Two departments bracket Act I — the Editor's Letter opens it and the Pilot's Log closes
it, behind The 99. Both arrived through `issue_spec.OPTIONAL_DEPARTMENTS`, piloted on
radagast and then rolled to the fleet; **all nine now carry them and that set is empty
again**, which is the state it should be found in. See CLAUDE.md for why the concept
exists and why an id should not stay in it.

**The Editor's Letter** is signed by Editor-in-Chief Margot Stet, the masthead's
only unbadged name. Each columnist owns exactly one evidence tier, so a fourth
badge would make four tiers out of three and a shared one would put two names on
one. She therefore may not make a claim that needs a badge — `validate-issue`
fails a bare percentage in her copy — and names the columnist who established a
figure instead, which is how a real editor's letter reads anyway.

**The Pilot's Log** was a three-way conversation written by `pilot-panel` (retired 2026-08-19; the nine keys are frozen). Its
`pilots_log` key is a LIST of turns, not prose: a turn carries the voice that
speaks it, so the renderer can label and colour it and a reader can follow who is
answering whom. Handed a string it renders TODO — an unlabelled panel is prose
with quotation marks.

**Its tier is `("coach",)` and not all three.** A department's tier is what it
GRANTS, not what its speakers mention: Vera cites a ruling and Ledger a rate, but
both earned those badges in The Kill and By the Numbers. Give the panel all three
and a conversation becomes a place where a new verified claim can arrive wearing
three voices at once.

**The panel opens on a HOT TAKE and runs behind The 99.** Turn 0 carries
`"kind": "hot-take"` and Sunny's voice; a later turn carries
`"responds_to": "hot-take"`. `validate-issue` checks those three things and no
semantic ones — whether a take is genuinely counter-intuitive, correct and
insightful is the charter's problem and an editor's, not a regex's. The department
moved to the end of Act I because the panel is the densest thing in the issue and
every move it makes refers to material the reader must already have met.

**The rule that outranks the rest: a line `engine.json` draws DASHED is a line the
panel may not assert — and that includes the hot take.** A `lines[]` entry with a `verified_by` rests on a
checker-passed stack and Vera may state it flatly; a null is the analyst's reading
and the panel may discuss it, argue about it, or say nobody has checked. That is
the evidence contract reaching past the picture into the copy.

**The per-byline voice lint** covers the panel (each turn carries its voice) and
every other prose key, whose voice is derived from its department's byline via
`issue_spec.voices_for`. A shared department flags only what both voices are
barred from. The bans were cut twice by measuring against the fleet — see
CLAUDE.md — and what remains is six evaluative adjectives with no hedging reading
plus Sunny's consulting vocabulary.

### The Short List (`considering.json`) — RETIRED 2026-08-19, frozen on the nine decks

**Exactly ten cards**, ranked, that the pilot should be thinking about — one artifact and
one routine (`the-ten`, retired with the `short-list-analyst` agent; its rule — ten ranked
adds, ownership not a criterion — now lives in `/prescribe`'s `add_candidates`) for every deck, replacing the retired `sideboard_analysis.json` /
`upgrade_watch.json` pair — and, once the sideboard itself was retired, the last artifact
standing on the question "what else could this deck play".

**Ownership is not a criterion.** Picks are scouted from the whole card pool and the list
carries no `source`. Ranking owned cards first turns an inventory question into a
selection rule: a card is on the list because it is worth knowing about, or it is not on
the list. **Analysis-only** — `cards.json` is never
rewritten by this routine.

`validate_considering.py` enforces the count and every claim: no pick may already be in
the deck, no duplicate picks or duplicate `natural_cut`s, a cut
that is a real maindeck card and never the commander, combo-line status vocabulary
(`needs a stack scenario` unless a checker-passed artifact is named), obsolescence claims
re-checked against `obsolescence_index.json`, synergy partners re-checked against the
pick's own graph shortlist **and** the deck, and every claimed bracket delta recomputed
through `bracket.assess()`. `deck-facts` and `deck-audit` are its deterministic pre-agent
briefs.

Rendered as **The Short List**, straight from the artifact with no prose key — a new key
would change `prose:shape` and invalidate both prose routines for no gain. The writer's
`upgrades` key is the section's opening copy and is cached separately. Tiers are marked
inline: computed evidence ◆, every ranking and verdict ★.

