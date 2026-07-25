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
manamap pilot fetch-deck <slug>         # decklist.txt → cards.json (Scryfall)
manamap pilot validate-deck <slug>      # 100/commander/singleton/color identity
manamap pilot validate-stack <slug> [--stack NNN]   # citation contract (stacks + decisions)
manamap pilot goldfish <slug>           # seeded Monte Carlo metrics → goldfish_metrics.json
manamap pilot build-manual <slug>       # → manuals/<slug>.html
manamap pilot build-index               # → manuals/index.html gallery
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
data/decks/<slug>/             tracked: decklist.txt, cards.json,
                               stacks/NNN-<kebab>.json, decisions/NNN-<kebab>.json,
                               goldfish_targets.json, goldfish_metrics.json,
                               strategic_frame.json, manual_prose.json
manuals/<slug>.html            tracked; manuals/index.html gallery tracked
```

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

## The resolve loop (agents)

Run via the `resolve-stack` skill: `stack-resolver` agent drafts → `validate-stack` (mechanical gate, short-circuits on form errors) → `rules-checker` agent verdict → re-spawn resolver with findings while iterations < `RESOLVE_MAX_ITERATIONS` (3). Agents are read-only; the orchestrating session writes files. Batch scale-out (many scenarios in parallel) is a Workflow-tool upgrade path.

**Manual DoD verification**: run `/resolve-stack` on a scenario; confirm the saved artifact passes `manamap pilot validate-stack` and the golden-artifact test (`tests/test_pilot_validate_stack.py::test_all_committed_stacks_validate_and_pass`) unskips and passes.

## Manual generation

`write-manual` skill (v2.1 order): goldfish → `deck-analyst` evidence pull → **strategic frame** (`strategy-researcher` MODE consult → `strategic_frame.json`; its `candidate_missing_lines` feed the resolve-stack queue, its `gaps` feed the next research pass) → `pilot-coach` coaching (threat/matchups + decisions, receives the frame) → `manual-writer` prose (zero-guessing: combo lines only from verified stacks, claims trace to graphs/oracle text; receives the frame) → `manual_prose.json` (tracked, human-editable) → `manamap pilot build-manual <slug>` + `build-index` (deterministic, byte-identical rebuilds, `[TODO]` placeholders for missing prose, only checker-passed stacks render).

## Tests

`tests/test_pilot_*.py` — 103 tests across 8 files: CR chunker edge cases (`test_pilot_rules_db`, 12), rules queries (`test_pilot_query_rules`, 5), mocked Scryfall ingestion (`test_pilot_fetch_deck`, 11), citation-contract fixtures incl. strategy-citation dispatch (`test_pilot_validate_stack`, 18), goldfish determinism (`test_pilot_goldfish`, 12), renderer determinism/escaping/TOC/sideboard (`test_pilot_build_manual`, 18), strategy chunker + real-DB checks (`test_pilot_strategy_db`, 9), strategy form validator + changelog (`test_pilot_validate_strategy`, 18). Data-gated tests use `requires_rules` / `requires_deck` / `requires_strategy` markers from `tests/conftest.py`.
