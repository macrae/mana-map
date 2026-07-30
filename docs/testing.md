# Testing

```bash
.venv/bin/python -m pytest          # full suite (discovery via testpaths = tests/)
```

932 tests in `tests/`: 331 card-pipeline + 601 pilot-subsystem. Three categories:

**Card-pipeline unit tests (281) — no data files needed, run anywhere:**

| File | Tests | Covers |
|------|-------|--------|
| `test_extract.py` | 53 | Multi-face cards, derived columns, supertype classification |
| `test_preprocess.py` | 23 | Vocab building, encoding, normalization, multi-hot |
| `test_mechanical_tags.py` | 45 | All 33 tag regexes, removal edge cases, multi-hot |
| `test_synergy.py` | 19 | Rule matching, bidirectionality, combo exclusion, ranking |
| `test_power_creep.py` | 36 | Strictly-better detection, tiered similarity gate, stat parsing |
| `test_combos.py` | 30 | Combo extraction, graph building, dedup |
| `test_cluster_regions.py` | 31 | Region naming (color/type/guild/TF-IDF), geometry, dedup |
| `test_card_roles.py` | 27 | Role classification, type-line mana disambiguation, coverage floors |
| `test_analysis_common.py` | 17 | Colour-identity masks, name index, vectorized top-k |

**Card-pipeline data-dependent tests (42) — need artifacts from a pipeline run:**

| File | Tests | Covers |
|------|-------|--------|
| `test_pipeline_integration.py` | 30 | Cross-artifact count consistency, output quality checks |
| `test_find_similar.py` | 12 | Binary format fidelity, L2 normalization, 128D vs 2D ranking |

Both are skip-guarded: `test_pipeline_integration.py` skips per-file via `requires_file(...)`; `test_find_similar.py` uses the module-level `requires_data` marker from `tests/conftest.py` (gates on `embeddings.npy` existing).

**Pilot-subsystem tests (601) — mostly pure-function with inline fixtures; data-gated ones behind markers:**

| File | Tests | Covers | Data gate |
|------|-------|--------|-----------|
| `test_pilot_rules_db.py` | 12 | CR chunker edge cases (TOC, subrules, examples, glossary) | 2 behind `requires_rules` |
| `test_pilot_query_rules.py` | 5 | Semantic top-k, exact lookup, suggestions | all behind `requires_rules` |
| `test_pilot_fetch_deck.py` | 24 | Decklist parsing, mocked Scryfall, exact printings, decklist-hash short-circuit | 1 behind `requires_deck` |
| `test_pilot_validate_stack.py` | 18 | Citation contract, decision form, strategy-citation dispatch, golden artifacts | golden test behind `requires_deck` **and** `requires_rules` |
| `test_pilot_goldfish.py` | 16 | Seeded determinism, mulligan rule, target assembly | 1 behind `requires_deck` |
| `test_pilot_build_manual.py` | 42 | Department completeness, contract integrity, furniture rendering, determinism, escaping | — |
| `test_pilot_strategy_db.py` | 9 | Strategy chunker (IDs, sources, parents), real-DB alignment | 3 behind `requires_strategy` |
| `test_pilot_validate_strategy.py` | 18 | Doc form errors, changelog contract, strategy citations through `validate_citations` | — |
| `test_pilot_validate_issue.py` | 29 | Issue identity incl. the decklist_sha256 stamp, department completeness/order, tier-costume integrity, card-name accuracy | — |
| `test_pilot_artist_credits.py` | 24 | Standout detection, per-entry counting, drop runs, roster overlap | 1 behind `requires_deck` |
| `test_pilot_agent_cache.py` | 57 | Fingerprint stability/order-independence, prose-shape semantics, staleness diffs, record guards, N/A scan semantics (incl. sideboard gating) and exit codes, memoized loaders | 5 behind `requires_deck` |
| `test_pilot_build_deck.py` | 48 | Pool hard filters (bracket, identity, bans), scoring components, slot filling with alternates, emergent-combo pass, decklist naming | — |
| `test_pilot_manabase.py` | 40 | Hypergeometric source counts, pip counting incl. hybrid, effective-pip quorum, greedy land selection, land quality | — |
| `test_pilot_bracket.py` | 35 | Floor drivers, commander-assumption exclusion (A-004), two-card infinites, tutors-never-scored, goblin-storm golden checks | 3 behind `requires_deck` + `requires_roles` |
| `test_pilot_validate_build.py` | 37 | Card count, singleton, identity, per-role budget arithmetic, bracket cross-check, manabase staleness, critic verdict consistency | — |
| `test_pilot_deck_facts.py` | 14 | Deterministic deck brief: DFC colours, curve, restricted-mana classes, notes | 4 behind `requires_deck` |
| `test_pilot_sideboard_facts.py` | 14 | Board split, accessory exclusion, lines-opened set difference, bracket-if-added | 2 behind `requires_deck` |
| `test_pilot_validate_sideboard.py` | 22 | Swap form (in/out/why/when), recomputed bracket deltas, verdict closed set | — |
| `test_pilot_validate_strategic_frame.py` | 15 | Frame form, engine strategy_refs, candidate-line status, shared validator tail | — |

## conftest.py

- `requires_data` — skipif marker gating on `embeddings.npy` (card pipeline)
- `requires_rules` — gates on `rules_index.json` (build the rules DB first)
- `requires_deck` — gates on `data/decks/goblin-storm/cards.json`
- `requires_strategy` — gates on `strategy_index.json` (run `manamap pilot build-strategy-db`)
- `requires_roles` — gates on `card_roles.json` (run `manamap card-roles`)
- `data_dir` — session fixture returning the resolved `config.DATA_DIR`

Each pilot marker gates on the *last* artifact of its stage so a partially populated directory still skips cleanly.

Paths come from `manamap.config` (never hardcode `Path("data")`), so the suite runs from any CWD and honors `MANAMAP_DATA_DIR`:

```bash
MANAMAP_DATA_DIR=/nonexistent .venv/bin/python -m pytest   # data tests skip cleanly
```

## Notes for writing tests

- Unit tests build inline DataFrames/dicts — keep it that way (no fixture files)
- `test_synergy.py` patches `manamap.analysis.synergy.load_combo_partners` to stub combo I/O
- Integration count assertions enforce the index-alignment invariant — if you change the card count (new Scryfall data), re-run the full pipeline before expecting green
