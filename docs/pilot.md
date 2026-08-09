# Pilot Subsystem

Turns a locked 100-card Commander decklist into a **pilot's manual** — a coaching zine whose combo lines are backed by rules-cited, machine-verified stack resolutions, whose numbers come from seeded simulations, and whose coaching is labeled as coaching.

## The three-tier evidence contract (manual v2)

Every section of a manual wears a badge declaring its epistemic status:

| Tier | Badge | Content | Enforcement |
|---|---|---|---|
| Rules-verified | ✓ green | Stack resolutions | Citation contract + adversarial checker |
| Data-derived | ◆ blue | Goldfish metrics, upgrade data | Seeded, reproducible artifacts (byte-identical re-runs) |
| Coaching | ★ gold | Threat assessment, matchups, decision trees | Labeled judgment grounded in tier 1/2 artifacts; founder review via tracked JSON |

## The citation contract

> The resolver is not allowed to make an uncited claim. Every effect it reports carries a Comprehensive Rules number pulled from the rules DB, and the checker's only job is verifying the cited rule text actually supports the claim.

Enforcement is layered:
1. **Form (code)** — `manamap pilot validate-stack`: every step has ≥1 citation; every rule ID matches `RULE_ID_RE` and exists in the index; every quote is a whitespace-normalized substring of real rule text. A resolution that fails form never reaches the checker.
2. **Meaning (agent)** — the `rules-checker` agent exact-fetches every cited rule and judges the *full* rule text against the claim (guards out-of-context quoting), and audits for missing steps (state-based actions, priority, triggers).
3. **Publication** — `build-manual` renders only stacks with `checker.verdict == "pass"`.

## Commands

```bash
manamap pilot download-rules            # CR txt (idempotent; sha256 sidecar)
manamap pilot build-rules-db            # ~3.9K chunks → embeddings + index
manamap pilot query-rules "…" --json    # semantic top-k (resolver's discovery path)
manamap pilot lookup-rule 702.40a --json  # exact fetch (checker's verification path)
manamap pilot build-deck <slug> [--write-decklist]  # brief.json → build_plan.json (no agents)
manamap pilot validate-build <slug>     # form gate over a build plan
manamap pilot bracket-check <slug> [--target N] [--json]  # bracket floor → bracket_report.json
manamap pilot deck-facts <slug> [--out F]  # the deterministic brief agents read first
manamap pilot deck-history <slug> [--json]  # applied swaps (from git) + the pending ten
manamap pilot deck-audit <slug> [--archetype A] [--json] [--out D/]  # cited axis targets + engine activation
manamap pilot validate-diagnosis <slug>    # diagnosis form; axes re-derived, cuts checked against verified stacks
manamap pilot pool-facts <paths…> [--exclude F] [--json] [--out F]  # a BOX OF CARDS → which deck to build
manamap pilot cache-rebless <slug>             # re-record every STALE_OK routine, zero spawns
manamap pilot impact <slug> [--json]           # card/figure/target/zone staleness report (free)
manamap pilot validate-strategic-frame <slug>  # frame form + candidate-line flags
manamap pilot fetch-deck <slug>         # decklist.txt → cards.json (Scryfall)
manamap pilot validate-deck <slug>      # 100/commander/singleton/color identity
manamap pilot validate-stack <slug> [--stack NNN]   # citation contract (stacks + decisions)
manamap pilot validate-stack <slug> --scenario-only # preflight BEFORE spawning a resolver
manamap pilot goldfish <slug>           # seeded Monte Carlo metrics → goldfish_metrics.json
manamap pilot artist-credits <slug> --json  # standout artists + art themes (Featured Artist)
manamap pilot build-manual <slug>       # → manuals/<slug>.html
manamap pilot build-index               # → manuals/index.html gallery
manamap pilot validate-issue <slug>     # form-check issue.json + issue_plan.json
manamap pilot cache-status <slug>       # have an agent routine's inputs changed?
manamap pilot cache-record <slug> --routine R   # record what produced an artifact
manamap pilot cache-clear <slug>        # drop cache records
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
                               decisions/NNN-*.json  pilot-coach
                               strategic_frame.json  strategy-researcher (consult)
                               manual_prose.json     pilot-coach + manual-writer
                               pilot_feedback.md     authored, OPTIONAL (free-text pilot notes)
                               mana_analysis.json    mana-analysis (deterministic, no agent)
                               tutor_guide.json      pilot-coach (Fetch Quests)
                               considering.json      short-list-analyst (The Short List — ten)
                               diagnosis.json        deck-doctor ⇄ deck-skeptic (the improvement loop)
                               deck_recon.json       deck-doctor MODE recon (dated; perishable)
                               issue.json            authored (never generated)
                               issue_plan.json       magazine-editor
                               .agent-cache.json     cache-record
manuals/<slug>.html            tracked; manuals/index.html gallery tracked
```

**Exact printings**: `fetch-deck` resolves a Moxfield export's `(SET) COLLECTOR [*F*]`
annotations against Scryfall's `/cards/collection` by set + collector number **first**,
falling back to name lookup only for unannotated lines. `cards.json` therefore carries
the physical card the pilot owns — artist, set, collector number, border, frame
effects, finishes, foil, plus `art_crop` for full-bleed magazine art. Image URLs have
Scryfall's cache-busting query string stripped so re-fetches stay byte-stable, and the
run short-circuits entirely when the decklist hash is unchanged (`--force` to override
after an oracle update).

Deck slugs are kebab-case. Scenario files are `NNN-<kebab>.json`, zero-padded, authoring order. Card names use the full `" // "` form, matching the combo/synergy/obsolescence graph keys.

## Rules DB

One chunk per numbered CR rule — **chunk ID = rule number = citation ID** — plus `glossary:<term>` chunks. `Example:` and continuation lines attach to the owning rule, so quotes from examples satisfy the contract. Embedded text is prefixed with `id + section title` (helps MiniLM find "storm" for 702.40a, whose text never says storm); stored text is verbatim CR. Embeddings are L2-normalized MiniLM (reuses `compute_text_embeddings`); row i ↔ `order[i]`.

**CR refresh** (each set release): get the current TXT link from https://magic.wizards.com/en/rules, update `CR_RULES_URL` in `src/manamap/config.py`, run `download-rules` + `build-rules-db`. Artifacts record their `rules_version`.

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

## Goldfish metrics (`goldfish_metrics.json`, tier ◆)

`pilot/goldfish.py`: seeded Monte Carlo (seed 42, 10K iterations) simulating **resource development, not full games** — model assumptions are embedded in the artifact and rendered in the manual. Metrics: opening-hand/mulligan stats, land-drop and mana curves, commander-cast turn distribution, per-deck target-set assembly (`goldfish_targets.json`, `any_of` groups, drawn-by-turn semantics), bodies-by-turn (labeled crude). Deterministic: the data-gated test regenerates and compares byte-for-byte.

## Decision scenarios (`decisions/NNN-<kebab>.json`, tier ★)

`kind: "decision"` artifacts: archetypal board + table state, a decision question, 2–4 branches each with `choice`, `line`, `signals`, `coalition_risk`, `coaching`, optional `citations` (same verbatim-quote contract), and a `recommendation` matching a branch. Mechanically form-checked by `validate-stack`; substantively reviewed by humans — the tracked JSON is the red-line surface. Authored via the `pilot-coach` agent (`author-decision` skill).

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
  evidence pull; pilot-coach and manual-writer consume it.

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
`strategic-frame`, `coach-prose`,
`writer-prose`, `the-ten` (The Short List — applies to every deck), `tutor-guide`
(Fetch Quests — `N/A` for a deck with no library-search tutors, via the applicability
gate in agent_cache), `issue-plan`, plus `stack:<NNN>` and `decision:<NNN>` discovered
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

## The resolve loop (agents)

Run via the `resolve-stack` skill: `stack-resolver` agent drafts → `validate-stack` (mechanical gate, short-circuits on form errors) → `rules-checker` agent verdict → re-spawn resolver with findings while iterations < `RESOLVE_MAX_ITERATIONS` (3). Agents are read-only; the orchestrating session writes files. Batch scale-out (many scenarios in parallel) is a Workflow-tool upgrade path.

**Manual DoD verification**: run `/resolve-stack` on a scenario; confirm the saved artifact passes `manamap pilot validate-stack` and the golden-artifact test (`tests/test_pilot_validate_stack.py::test_all_committed_stacks_validate_and_pass`) unskips and passes.

## The magazine layer (STYLEv3)

Each deck is a complete **issue** of *Pilot's Manual* — a fixed set of sections in a
fixed order (see `issue_spec.DEPARTMENTS`; never transcribe the list or its count into
a prompt), grouped into five acts that ramp from what to do, through tactics and the
long game, into the numbers and the proof. Readers learn the publication once and
navigate it forever. Every section is signed by one of three columnists — `"Ledger"
Lin Marginal` (◆), `Counselor Vera Dictum` (✓), `Coach Sunny Brightside` (★) — and
STYLEv3 L10 holds that every issue is the reader's first: no version numbers, no
changelog voice, enforced by `validate_issue.validate_self_containment()`. The design
authority is `STYLEv3.md` (editorial laws, the Commander Mandate, section specs,
voice, component library); `docs/history/STYLE-v1-visual-research.md` and
`-v2-editorial-method.md` are its archived sources.

- **`src/manamap/pilot/issue_spec.py`** — the canonical department system: ids, order,
  promises, evidence tiers, rhythm tags, component library. Changing it changes every
  issue; treat it like `config.py`.
- **`issue.json`** (tracked, **authored by a human**) — volume, issue_date, cover_price,
  deck_name, commander, cover_tagline, next_issue. Never generated: a generated date
  would break byte-identical rebuilds.
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
- **`magazine-editor` agent** — reads STYLEv3 and every artifact, returns the plan as
  JSON. It never writes HTML: determinism, mechanical validation, and the citation
  contract all depend on the renderer staying deterministic.
- **`design-issue` skill** — the loop: gather → plan → validate → build → review.

The Kill renders combo lines as feature spreads with dossier pointers; **Judge's Desk**
carries the complete resolutions with every citation verbatim (the renderer may not
summarize proof). The Command Zone department is mandatory and format-specific — the
tax ladder, color identity, the 21-damage clock — and is what makes this a Commander
magazine rather than a Magic one.

## Manual generation

Content pipeline (`write-manual` skill) — goldfish → `deck-analyst` evidence pull → **strategic frame** (`strategy-researcher` MODE consult → `strategic_frame.json`; its `candidate_missing_lines` feed the resolve-stack queue, its `gaps` feed the next research pass) → `pilot-coach` coaching (threat/matchups + decisions, receives the frame) → `manual-writer` prose (zero-guessing: combo lines only from verified stacks, claims trace to graphs/oracle text; receives the frame) → `manual_prose.json` (tracked, human-editable) → `manamap pilot build-manual <slug>` + `build-index` (deterministic, byte-identical rebuilds, `[TODO]` placeholders for missing prose, only checker-passed stacks render).

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

## Tests

`tests/test_pilot_*.py` — the largest group in the suite; see `docs/testing.md` for
the per-file inventory.

**Build side:** deck builder pool/scoring/slot-filling/emergent-combo pass (`test_pilot_build_deck`, 42), hypergeometric mana math and land selection (`test_pilot_manabase`, 36), bracket floor + drivers + the goblin-storm golden checks (`test_pilot_bracket`, 35), build-plan form gate (`test_pilot_validate_build`, 37).

**Publish side:** agent cache incl. N/A scan semantics and memoized loaders (`test_pilot_agent_cache`, 57), renderer determinism/escaping/TOC (`test_pilot_build_manual`, 42), issue form gate incl. the decklist_sha256 stamp (`test_pilot_validate_issue`, 29), artist analysis (`test_pilot_artist_credits`, 24), mocked Scryfall ingestion (`test_pilot_fetch_deck`, 24), citation contract incl. strategy-citation dispatch (`test_pilot_validate_stack`, 18), strategy form validator + changelog (`test_pilot_validate_strategy`, 18), goldfish determinism and the two opening-hand distributions (`test_pilot_goldfish`, 16), strategic-frame form (`test_pilot_validate_strategic_frame`, 15), deck facts (`test_pilot_deck_facts`, 14), CR chunker edge cases (`test_pilot_rules_db`, 12), strategy chunker + real-DB checks (`test_pilot_strategy_db`, 9), rules queries (`test_pilot_query_rules`, 5).

Data-gated tests use `requires_rules` / `requires_deck` / `requires_strategy` / `requires_roles` markers from `tests/conftest.py`.

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
touches the web: it fills a hole `docs/deck-builder-v2.md` names outright — there are no
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

## The Short List (`considering.json`, tiers ◆ + ★)

**Exactly ten cards**, ranked, that the pilot should be thinking about — one artifact and
one routine (`the-ten`) for every deck, replacing the retired `sideboard_analysis.json` /
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

## Fetch Quests (`tutor_guide.json`, tier ★)

One wish per tutor. `pilot-coach` authors an entry for every maindeck library-search
tutor — scenario → the exact card to fetch → why — and `validate-tutor-guide` holds each
one to the deck and to that tutor's own search constraint, **per clause**: a DFC or
chapter card can carry several search clauses (Huatli's front face fetches a basic land;
Roar III fetches Dinosaurs), so a fetch is legal if any clause permits it. Pure land ramp
(Cultivate, Nature's Lore) is excluded — that belongs to Sources Say. A deck with no
tutors keeps the section and prints standing copy; the routine reports `N/A`.

## Sources Say (`mana_analysis.json`, tier ◆)

The mana audit, and the one section with **no agent at all**: `manamap pilot
mana-analysis <slug>` computes it deterministically, reusing the deck-builder's own
hypergeometric kit (`manabase.py`). Land classes, per-colour land and nonland sources,
pip share vs source share, on-curve probability with and without rocks, the ramp census,
and a stated-assumptions block. The writer's `mana_base` key narrates it.

**Count copies, not decklist entries.** `cards.json` stores basics as one entry with
`quantity: N`, and counting entries once published "18 lands" for a 33-land deck and
understated every colour's sources fleet-wide. `common.expand_copies()` is the shared
primitive; `lands.total` is copies and `lands.entries` is distinct cards, both reported so
they can never be confused again. Three guards: a unit fixture (11 Islands = 11 blue
sources), a staleness test recomputing every tracked artifact, and a `validate-issue` lint
rejecting any reader-facing copy that quotes the entry count as a land count.

**The trap this exists to catch.** Sazacap's Brew is tagged `buff:pump` because its text
contains "+2/+0", and Vol. 001 shipped advice to test it in the Witch's Mark slot. Both are wrong: the Brew's first target is a *player*, so
Zada — which copies instants targeting **only** Zada — never copies it, while Witch's Mark
targets a creature and is copyable. Reading the card rather than its role tag inverted the
recommendation, and the published prose was corrected to match. That is the whole value of
the pass.

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

## Agent handoff

Deck agents write their JSON to `data/decks/<slug>/.agent-out/<agent>.json` (gitignored)
and return only that path plus a short summary. The orchestrator reads, validates, and
merges into the tracked artifact. `candidate_pool.json` reaches 133 KB — returning it
inline costs ~35k tokens of orchestrator context for nothing, and the agent's tools are
unchanged either way.
