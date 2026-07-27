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
                               build_plan.json       deck-architect ⇄ deck-critic
                               bracket_report.json   bracket-check (◆)
                               decklist.txt          authored, OR build-deck --write-decklist
                               cards.json            fetch-deck
                               goldfish_targets.json authored
                               goldfish_metrics.json goldfish
                               stacks/NNN-*.json     authored scenario + resolve loop
                               decisions/NNN-*.json  pilot-coach
                               strategic_frame.json  strategy-researcher (consult)
                               manual_prose.json     pilot-coach + manual-writer
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

Routines (6 static): `candidate-pool`, `deck-build`, `strategic-frame`,
`coach-prose`, `writer-prose`, `issue-plan`, plus `stack:<NNN>` and
`decision:<NNN>` discovered from disk. Declared in `config.AGENT_ROUTINES`.

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

Each deck is a complete **issue** of *Pilot's Manual* — fifteen fixed departments in
a fixed order, so readers learn the publication once and navigate it forever. The
design authority is `STYLEv3.md` (editorial laws, the Commander Mandate, department
specs, voice, component library); `docs/history/STYLE-v1-visual-research.md` and
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
- **`validate-issue`** — the mechanical gate: identity block complete, all fifteen
  departments present in canonical order, copy completeness, components from the fixed
  library, **tier costume never overridden**, every PILOT TIP / caption / roster card
  name real, and no two dense departments adjacent.
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

`tests/test_pilot_*.py` — 372 tests across 15 files.

**Build side:** deck builder pool/scoring/slot-filling/emergent-combo pass (`test_pilot_build_deck`, 42), hypergeometric mana math and land selection (`test_pilot_manabase`, 36), bracket floor + drivers + the goblin-storm golden checks (`test_pilot_bracket`, 35), build-plan form gate (`test_pilot_validate_build`, 37).

**Publish side:** agent cache incl. N/A scan semantics (`test_pilot_agent_cache`, 42), renderer determinism/escaping/TOC/sideboard (`test_pilot_build_manual`, 30), issue form gate (`test_pilot_validate_issue`, 25), artist analysis (`test_pilot_artist_credits`, 24), mocked Scryfall ingestion (`test_pilot_fetch_deck`, 19), citation contract incl. strategy-citation dispatch (`test_pilot_validate_stack`, 18), strategy form validator + changelog (`test_pilot_validate_strategy`, 18), goldfish determinism and the two opening-hand distributions (`test_pilot_goldfish`, 16), CR chunker edge cases (`test_pilot_rules_db`, 12), strategy chunker + real-DB checks (`test_pilot_strategy_db`, 9), rules queries (`test_pilot_query_rules`, 5).

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
