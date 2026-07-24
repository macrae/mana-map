# Pilot Subsystem

Turns a locked 100-card Commander decklist into a **pilot's manual** — zine HTML whose combo lines are backed by rules-cited, machine-verified stack resolutions.

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
manamap pilot validate-stack <slug> [--stack NNN]   # citation contract
manamap pilot build-manual <slug>       # → manuals/<slug>.html
```

## Data layout

```
data/rules/                    gitignored (regenerable): comprehensive_rules.txt,
                               rules_index.json, rules_embeddings.npy, sidecars
data/decks/<slug>/             tracked: decklist.txt, cards.json,
                               stacks/NNN-<kebab>.json, manual_prose.json
manuals/<slug>.html            tracked
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

## The resolve loop (agents)

Run via the `resolve-stack` skill: `stack-resolver` agent drafts → `validate-stack` (mechanical gate, short-circuits on form errors) → `rules-checker` agent verdict → re-spawn resolver with findings while iterations < `RESOLVE_MAX_ITERATIONS` (3). Agents are read-only; the orchestrating session writes files. Batch scale-out (many scenarios in parallel) is a Workflow-tool upgrade path.

**Manual DoD verification**: run `/resolve-stack` on a scenario; confirm the saved artifact passes `manamap pilot validate-stack` and the golden-artifact test (`tests/test_pilot_validate_stack.py::test_all_committed_stacks_validate_and_pass`) unskips and passes.

## Manual generation

`write-manual` skill: optional `deck-analyst` evidence pull → `manual-writer` agent (zero-guessing: combo lines only from verified stacks, claims trace to graphs/oracle text) → `manual_prose.json` (tracked, human-editable) → `manamap pilot build-manual <slug>` (deterministic, byte-identical rebuilds, `[TODO]` placeholders for missing prose).

## Tests

`tests/test_pilot_*.py` — 41 tests: chunker edge cases, DB queries, mocked Scryfall ingestion, citation-contract fixtures, renderer determinism/escaping. Data-gated tests use `requires_rules` / `requires_deck` markers from `tests/conftest.py`.
