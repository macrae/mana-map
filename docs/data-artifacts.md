# Data Artifacts

Everything lives in `data/`. Most files are gitignored (regenerable via `manamap run`); the ones the deployed viz fetches are **git-tracked** via explicit `.gitignore` exceptions.

**Do NOT move these to Git LFS.** The viz is served by GitHub Pages straight from the repo, and Pages serves LFS pointer files, not content — LFS would silently break every data fetch on the deployed site.

| File | Producer (step) | Shape / size | Git | Consumed by |
|------|-----------------|--------------|-----|-------------|
| `oracle-cards.json` | download (1) | ~200MB | ignored | extract |
| `.download-meta.json` | download (1) | tiny | ignored | download (idempotency) |
| `cards.csv` | extract (2) | ~34,300 rows | ignored | preprocess, train×2, embed, synergy, power_creep, cluster_regions |
| `text_embeddings.npy` | preprocess (3) | (N, 384) ~50MB | ignored | train×2, embed |
| `card_features.npz` | preprocess (3) | ~13MB | ignored | train×2, embed |
| `color_vectors.npy` | preprocess (3) | (N, 5) | ignored | viz metadata build |
| `mechanical_tags.npy` | preprocess (3) | (N, 33) | ignored | train_ability |
| `model.pt` | train (4a) | ~711KB | ignored | embed |
| `model_ability.pt` | train_ability (4b) | ~711KB | ignored | embed |
| `embeddings.npy` | embed (5) | (N, 128) ~17MB | ignored | reduce, export, synergy (fallback) |
| `embeddings_ability.npy` | embed (5) | (N, 128) ~17MB | ignored | reduce, export, synergy, power_creep |
| `card_metadata.csv` | embed (5) | ~3MB | ignored | reduce |
| `projection_2d.json` | reduce (6) | ~13MB | **tracked** | viz (Color+Type map) |
| `projection_2d_ability.json` | reduce (6) | ~13MB | **tracked** | viz (Abilities map) |
| `combos_raw.json` | download-combos (7) | **~430MB** | ignored | process-combos |
| `combo_graph.json` | process-combos (8) | ~4.5MB, `{"partners": {name: [names]}}` **only** | **tracked** | viz deck builder, synergy (exclusions) |
| `combo_details.json` | process-combos (8) | ~25.7MB, `{combos, by_card, meta}`; per-combo `bracket` (1–4), `mana_value_needed`, `popularity`, banned flag | **tracked** | `pilot/bracket.py`, `pilot/build_deck.py`, `deck-analyst` — **never fetched by the viz** |
| `embeddings.bin` | export (9) | ~17MB | **tracked** | viz (Find Similar, deck builder) |
| `embeddings_ability.bin` | export (9) | ~17MB | **tracked** | viz (Abilities map similarity) |
| `synergy_graph.json` | synergy (10) | ~8–27MB | **tracked** | viz (Find Synergies, deck builder) |
| `obsolescence_index.json` | power-creep (11) | ~5–8MB | **tracked** | viz (obsolescence panels) |
| `regions_default.json` | cluster-regions (12) | ~212KB, `{meta, regions, membership}`; 15 L0 + 110 L1, each with `cx/cy/span/w/h/count/top_tags` | **tracked** | viz (region labels, drill-by-region) |
| `regions_ability.json` | cluster-regions (12) | ~195KB, same shape; 12 L0 + 73 L1 | **tracked** | viz (region labels, drill-by-region) |

`membership` is two positional arrays (`l0`, `l1`), one entry per card in `cards.csv` row order, `-1` for noise — so it inherits the index-alignment invariant and `membership.l0[i]` describes `cards.csv[i]`. Cluster id *n* at level *L* is the region with `id == "lL_n"`. This is the only thing in the repo that can answer *which region is this card in*; before it existed the viz could draw a region's name but never its members. **Noise is a real answer, not a gap**: 29% of cards on the default map belong to no L0 region, and they are left at `-1` rather than snapped to a nearest centroid they were never clustered into.

`w` and `h` are the bounding box beside `span` (which stays `max(w, h)` and still drives label culling). Collapsing them discarded aspect ratio, the one signal distinguishing a filament from a blob — a 20×1 streak and a 20×20 cloud serialised identically. With both axes kept, the map's roads are measurable: `White Enchantments — Auras — ETB` is 209 cards at 1.6 × 0.1, a 16:1 streak.
| `card_roles.json` | card-roles (13) | ~1.9MB, `{roles, meta}`; 30,563 of 34,322 cards classified (31,622 Commander-legal), **53 roles in 19 families**, coverage 89.5% / 73.2% specific | **tracked** | `pilot/build_deck.py`, `pilot/bracket.py` (tutor density), `deck-analyst`, **and `viz/js/deck-map.js`** — the Deck Lens colours the 99 by role family |
| `viz_index.json` | viz-index (14) | 3.4 MB / **0.56 MB gzipped**, one slim record per card: name, supertype, colour, rarity, CMC, role tags. Deliberately **no oracle text** — the Scryfall card image already shows it | **tracked** | the discovery landing: random pick, coarse filters, name→row resolution for imports |
| `neighbours.bin` | viz-index (14) | 2.3 MB / **1.7 MB gzipped**, uint16 row ids: 12 similar + 10 synergy + 5 obsoleted-by per card, uint8 quantised similarity, sha256 of the source embeddings in the header. **Pre-sorted — never re-sort client-side** | **tracked** | synchronous branching without the 16.8 MB embedding matrix |
| `eval/similarity_golden.json` | **hand-authored** (never generated) | ~6KB, 40 groups / 163 cards of functional equivalents, `dev`/`test` split | **tracked** | `analysis/eval_embeddings.py` (step 14), `tests/test_embedding_quality.py` — must stay independent of tags/roles/synergy/combos, which training mines for positives |

N = card count, ~34,300 as of July 2026; grows as Scryfall adds sets.

**"Tracked" no longer means "the viz fetches it."** It did until Deck Building v2 added
`combo_details.json` and `card_roles.json`, which are tracked because the deck builder and
the agents need them on a fresh clone, but which the browser never touches.

**And "the viz" is now two pages with two registries.** The card map (`viz/index.html`)
fetches the nine files in the `DATA` map in `viz/js/mana-map.js`. The deck dossier
(`viz/deck.html`) fetches a disjoint set through its own `FILES` map in
`viz/js/deck-view.js`: `data/decks/index.json` first, then up to eight per-deck artifacts.
Nine is no longer the total — see `docs/viz.md`.

Note also that **`cards.csv` is gitignored but load-bearing for the builder**: it is the
only home of the `game_changer` column (WotC's Game Changers list, via Scryfall), so
`bracket-check` needs a pipeline run even though rendering a manual doesn't.

## Pilot-subsystem artifacts (see docs/pilot.md for semantics)

| Path | Producer | Git | Notes |
|------|----------|-----|-------|
| `data/rules/*` | `pilot download-rules` + `build-rules-db` | ignored | CR text, 3,888-chunk index + embeddings, sha/meta sidecars — fully regenerable |
| `data/strategy/strategy.md`, `CHANGELOG.md` | authored / `strategy-researcher` agent | **tracked** | The strategy companion — curated source of truth; founder-reviewed via diffs |
| `data/strategy/strategy_index.json`, `strategy_embeddings.npy`, `.strategy-db-meta.json` | `pilot build-strategy-db` | ignored | Derived RAG DB; index records the doc's sha256 (staleness handshake) |
| `data/decks/<slug>/*` | build-deck / resolve-stack / write-manual / design-issue loops, `pilot goldfish`, agents | **tracked** | Curated per-deck artifacts. **Build side:** `brief.json` (authored), `candidate_pool.json` (deck-analyst), `build_plan.json` (deck-architect ⇄ deck-critic), `bracket_report.json` (◆, from `bracket-check`; `validate-build` cross-checks the plan against it), `decklist.txt` (authored *or* generated by `build-deck --write-decklist`). **Publish side:** `cards.json`, verified stacks, decisions, goldfish metrics/targets, `mana_analysis.json` (◆, from `pilot mana-analysis`), `tutor_guide.json` (Fetch Quests), `considering.json` (The Short List — exactly ten), `strategic_frame.json`, `manual_prose.json`, `issue.json` (authored), `issue_plan.json`, plus `HISTORY.md` (append-only decklist ledger) and optional `pilot_feedback.md` (authored, a cached agent input). **Retired:** `sideboard_analysis.json` and `upgrade_watch.json`, both folded into `considering.json` |
| `data/decks/<slug>/.agent-cache.json` | `pilot cache-record` | **tracked** | Which inputs produced each agent artifact. Tracked so a `git pull` transfers someone else's regeneration as a cache hit. No timestamps — every diff line is a content fact. See `docs/agent-cost.md` |
| `data/decks/index.json` | `pilot build-index` | **tracked** | The deck manifest for `viz/deck.html`: slug, volume, deck name, commander, coverline, verified/decision counts, and each deck's **passing** stack filenames. Exists because a browser can list neither `data/decks/` nor `stacks/`. `tests/test_pilot_deck_manifest.py` asserts it matches the artifacts |
| `manuals/*.html` | `pilot build-manual` + `build-index` | **tracked** | Deterministic renders; deployed by GitHub Pages |
| `manuals/magazine.css` | `pilot build-manual` (from `pilot/design.py`) | **tracked** | Content-addressed (`?v=<sha8>`), so a token change obligates rebuilding every page |

The `.gitignore` mechanics matter here: `data/*` blanket-ignores, `!data/decks/` and `!data/strategy/` re-include those directories (trailing slash load-bearing), and then two sets are re-ignored individually — the three derived strategy-DB files, and `data/decks/*/.agent-out/` (agent scratchpads, deliberately transient).

## Consistency invariant

`projection[i]`, `embeddings[i]`, and `cards.csv[i]` all refer to the same card by **position**. Never partially regenerate after the card count changes — re-run the pipeline from the changed step onward (see `docs/pipeline.md`). The integration tests (`tests/test_pipeline_integration.py`) assert cross-artifact count consistency.

## Paths

All paths are defined in `src/manamap/config.py`, anchored to the repo root via `__file__` (CWD-independent). Override with `MANAMAP_DATA_DIR` / `MANAMAP_VIZ_DIR` env vars for sandboxed runs.
