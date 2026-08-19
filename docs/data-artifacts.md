# Data Artifacts

Everything lives in `data/`. Most files are gitignored (regenerable via `manamap run`); the ones the deployed viz fetches are **git-tracked** via explicit `.gitignore` exceptions.

**Do NOT move these to Git LFS.** The viz is served by GitHub Pages straight from the repo, and Pages serves LFS pointer files, not content — LFS would silently break every data fetch on the deployed site.

| File | Producer (step) | Shape / size | Git | Consumed by |
|------|-----------------|--------------|-----|-------------|
| `oracle-cards.json.gz` | download (1) | ~22MB gz (172MB raw) | ignored | extract |
| `.download-meta.json` | download (1) | tiny | ignored | download (idempotency) |
| `cards.csv` | extract (2) | ~34,900 rows | ignored | preprocess, train×2, embed, synergy, power_creep, cluster_regions |
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
| `combos_raw.json.gz` | download-combos (7) | **~34MB gz** (428MB raw) | ignored | process-combos |
| `combo_graph.json` | process-combos (8) | ~4.5MB, `{"partners": {name: [names]}}` **only** | **tracked** | viz deck builder, synergy (exclusions) |
| `combo_details.json` | process-combos (8) | ~25.7MB, `{combos, by_card, meta}`; per-combo `bracket` (1–4), `mana_value_needed`, `popularity`, banned flag | **tracked** | `pilot/bracket.py`, `pilot/build_deck.py`, `deck-analyst` — **never fetched by the viz** |
| `embeddings.bin` | export (9) | ~17MB | **tracked** | viz (Find Similar, deck builder) |
| `embeddings_ability.bin` | export (9) | ~17MB | **tracked** | viz (Abilities map similarity) |
| `synergy_graph.json` | synergy (10) | ~8–27MB | **tracked** | viz (Find Synergies, deck builder) |
| `obsolescence_index.json` | power-creep (11) | ~2.9MB | **tracked** | viz (obsolescence panels) |
| `regions_default.json` | cluster-regions (12) | ~579KB, `{meta, regions, membership}`; 19 L0 + 106 L1, each with `cx/cy/span/w/h/count/top_tags` | **tracked** | viz (region labels, drill-by-region) |
| `regions_ability.json` | cluster-regions (12) | ~502KB, same shape; 16 L0 + 43 L1 | **tracked** | viz (region labels, drill-by-region) |

`membership` is two positional arrays (`l0`, `l1`), one entry per card in `cards.csv` row order, `-1` for noise — so it inherits the index-alignment invariant and `membership.l0[i]` describes `cards.csv[i]`. Cluster id *n* at level *L* is the region with `id == "lL_n"`. This is the only thing in the repo that can answer *which region is this card in*; before it existed the viz could draw a region's name but never its members. **Noise is a real answer, not a gap**: 29% of cards on the default map belong to no L0 region, and they are left at `-1` rather than snapped to a nearest centroid they were never clustered into.

`w` and `h` are the bounding box beside `span` (which stays `max(w, h)` and still drives label culling). Collapsing them discarded aspect ratio, the one signal distinguishing a filament from a blob — a 20×1 streak and a 20×20 cloud serialised identically. With both axes kept, the map's roads are measurable: `White Enchantments — Auras — ETB` is 209 cards at 1.6 × 0.1, a 16:1 streak.
| `card_roles.json` | card-roles (13) | ~1.9MB, `{roles, meta}`; 28,513 of 34,890 cards classified, 23,313 with a *specific* role, **53 roles in 19 families**, coverage 89.5% / 73.2% specific | **tracked** | `pilot/build_deck.py`, `pilot/bracket.py` (tutor density), `deck-analyst`, **and the viz** — `build.js` colours the 99 by role family, and `MM.GROUPINGS.role` makes it a map overlay. Fetched **lazily**, only when the Role grouping is selected: 0.39 MB gzipped is not something to spend inside the 1.83 MB discovery boot |
| `viz_index.json` | viz-index (14) | 3.4 MB / **0.56 MB gzipped**, one slim record per card: name, supertype, colour, rarity, CMC, role tags. Deliberately **no oracle text** — the Scryfall card image already shows it | **tracked** | the discovery landing: random pick, coarse filters, name→row resolution for imports |
| `neighbours.bin` | viz-index (14) | 2.6 MB / **1.27 MB gzipped** (format v2 — smaller than v1's 1.70 MB despite the extra block, because playability-ranked partners repeat across anchors and compress), uint16 row ids: 12 similar + 10 synergy + 5 obsoleted-by per card, uint8 quantised similarity, **uint8 synergy-reason codes plus the 24-entry vocabulary appended**, sha256 of the source embeddings in the header. **Pre-sorted — never re-sort client-side** | **tracked** | synchronous branching without the 16.8 MB embedding matrix |
| `eval/similarity_golden.json` | **hand-authored** (never generated) | ~6KB, 40 groups / 163 cards of functional equivalents, `dev`/`test` split | **tracked** | `analysis/eval_embeddings.py` (step 15), `tests/test_embedding_quality.py` — must stay independent of tags/roles/synergy/combos, which training mines for positives |

N = card count, ~34,900 as of August 2026; grows as Scryfall adds sets.

**"Tracked" no longer means "the viz fetches it."** It did until Deck Building v2 added
`combo_details.json` and `card_roles.json`, which are tracked because the deck builder and
the agents need them on a fresh clone, but which the browser never touches.

**And "the viz" is now two pages with two registries.** The card map (`viz/index.html`)
fetches from the twelve-entry `DATA` map in `viz/js/mana-map.js` (only three land on the discovery boot: `viz_index.json`, `neighbours.bin` and the projection; `card_roles.json` is lazy). The deck dossier
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
| `data/opponents/<slug>/` | `fetch-opponent` (EDHREC average deck) or authored | **tracked** | The pod: an opponent seat for `simulate --vs <slug>` — `decklist.txt` in the repo's format (commander `*CMDR*`, basics as one quantity line) and `source.json` (commander, URL, fetch date, card count, note). A representative list, not a specific one; paste the real list over it when you know it. Resolved by the harness BEFORE `data/decks/` for the same name. |
| `data/decks/<slug>/*` | the bench loop (`deck-notes`, `deck-version`, `prescribe`, `simulate`), the build and resolve loops, `goldfish`, the agents | **tracked** | Curated per-deck artifacts, grouped by what they are for. **Identity:** `decklist.txt` (authored *or* generated by `build-deck --write-decklist` — every change is a commit, which is what `deck-version` numbers), `cards.json` (`fetch-deck`; Scryfall resolution with exact printings), `deck_versions.json` (AUTHORED tags on git-derived versions — `{tags: {name: {version, sha, decklist_sha256, at, note}}}`; the version LIST is never tracked because a commit's sha is not known inside the commit). **The table:** `log.jsonl` (the captain's log — AUTHORED, append-only, one game per line via `deck-notes add`, each stamped with the sha of `decklist.txt` as it stood; no agent or command rewrites it), `log_annotations.json` (DERIVED — the `debrief` agent's structured reading keyed by entry id, merged by `merge-debrief`, held to the log by `validate-debrief`: it may name nothing the note and the 99 do not), `experiments/<id>.json` (a controlled A/B of two versions against the same table — both arms' decklist text and analysis, the delta, and whether the win-rate intervals overlap; logs beside it gitignored and exactly regenerable), `sim/<run-id>.json` (a Forge simulation run — ◆ SEEDED since `-s` was found, SAMPLED on the earliest records: outcomes, per-game rows, `analysis` with Wilson/normal intervals and its `limits`, every seat's decklist sha, engine versions, and the assumptions incl. Forge's AI caveat; `sim/logs/` and `sim/scenarios/` beside it are gitignored and `validate-sim` re-derives the analysis from the logs where they exist), `prescriptions/<id>-*.json` (one question to the doctor — AUTHORED prompt + the doctor ⇄ skeptic answer in one file; accumulating, never overwritten; `validate-prescription` form-checks stale ones). **Measurement and proof:** `goldfish_targets.json` (the engine DECLARATION — `validate-goldfish-targets` checks it), `goldfish_metrics.json` (◆ seeded), `mana_analysis.json` (◆, `mana-analysis`), `bracket_report.json` (◆, `bracket-check`), `stacks/NNN-*.json` (authored scenario — v1 board or v2 game state, incl. boards lifted by `sim-scenario` — + the cited resolution + the checker's verdict; only a `pass` is a fact), `decisions/NNN-*.json` (★, `pilot-notes`), `engine.json` (the eight-stage model from `deck-engineer` ⇄ `engine-critic`; a `critic.verdict` of `fail` is SAVED and never cache-recorded), `deck_map.json` (◆ positions + membership from `deck-map`, ★ optional city names via `merge-deck-map` — tracked because `embeddings_ability.npy` is gitignored and a fresh clone must still render), `strategic_frame.json` (strategy-researcher consult), `diagnosis.json` (deck-doctor ⇄ deck-skeptic), `deck_recon.json` (MODE recon — **dated and perishable**, kept out of `strategy.md` on purpose), `tutor_guide.json` (one wish per tutor, `pilot-notes`), `manual_prose.json` (the five `pilot-notes` keys + frozen legacy keys, see below), `pending.json` (hand-authored queue of decided-but-unapplied changes; closure derived from the deck, never a flag), `.agent-cache.json` (below). **Build side:** `brief.json` (authored), `candidate_pool.json` (deck-analyst), `build_plan.json` (deck-architect ⇄ deck-critic; `validate-build` cross-checks it against the bracket report), and for a pool-constrained build the pool listing itself (`pool-green-bulk.txt` on radagast). **LEGACY, frozen on the nine published decks** (read only by the magazine renderer; no agent regenerates them; replaced by `docs/manual-v5-spec.md`): `issue.json` (authored identity — stays, the renderer needs it), `issue_plan.json` (the retired magazine-editor's packaging), the `card_roles`/`mana_base`/`upgrades`/`editors_letter`/`pilots_log` keys of `manual_prose.json`, `considering.json` (The Short List — its rule lives in prescriptions now) and its art sidecar `considering_art.json`. Also `HISTORY.md` (append-only decklist ledger, superseded by `deck-version`), optional `pilot_feedback.md` (authored; superseded by the log), an optional `README.md`. **Retired earlier:** `sideboard_analysis.json` / `upgrade_watch.json` (folded into `considering.json`, itself now frozen); `is_sideboard` no longer exists |
| `data/collection/*.txt` | authored (a human's card boxes) | **tracked** | A PHYSICAL card collection, one decklist-format file per box or colour. `deck-history` reads it to tell a proposed swap you already own from one you would have to buy — **the only ownership question left in the repo**, and it is about cardboard, not about a deck. Configurable via `MANAMAP_COLLECTION_DIR`; an absent directory means no ownership claim rather than an error. It lived in a top-level `share/` with no config entry and no mention in the layout diagram, which meant every clone answered "do you own this?" from one person's boxes |
| `data/decks/<slug>/.agent-cache.json` | `pilot cache-record` | **tracked** | Which inputs produced each agent artifact. Tracked so a `git pull` transfers someone else's regeneration as a cache hit. No timestamps — every diff line is a content fact. See `docs/agent-cost.md` |
| `data/decks/index.json` | `pilot build-index` | **tracked** | The deck manifest for `viz/deck.html`: slug, deck name, commander, verified/decision counts, each deck's **passing** stack filenames (plus legacy volume/coverline fields the renderer still emits). Exists because a browser can list neither `data/decks/` nor `stacks/`. `tests/test_pilot_deck_manifest.py` asserts it matches the artifacts |
| `manuals/*.html` | `pilot build-manual` + `build-index` | **tracked** | Deterministic renders of the LEGACY magazine page per deck (until `docs/manual-v5-spec.md`); deployed by GitHub Pages |
| `manuals/magazine.css` | `pilot build-manual` (from `pilot/design.py`) | **tracked** | The legacy page's stylesheet, content-addressed (`?v=<sha8>`), so a token change obligates rebuilding every page |

The `.gitignore` mechanics matter here: `data/*` blanket-ignores, `!data/decks/` and `!data/strategy/` re-include those directories (trailing slash load-bearing), and then two sets are re-ignored individually — the three derived strategy-DB files, and the transient per-deck dirs: `data/decks/*/.agent-out/` (agent scratchpads), `sim/logs/` (raw Forge games, exactly regenerable when seeded) and `sim/scenarios/` (lifted boards awaiting a question). `!data/opponents/` re-includes the pod.

## Consistency invariant

`projection[i]`, `embeddings[i]`, and `cards.csv[i]` all refer to the same card by **position**. Never partially regenerate after the card count changes — re-run the pipeline from the changed step onward (see `docs/pipeline.md`). The integration tests (`tests/test_pipeline_integration.py`) assert cross-artifact count consistency.

## Paths

All paths are defined in `src/manamap/config.py`, anchored to the repo root via `__file__` (CWD-independent). Override with `MANAMAP_DATA_DIR` / `MANAMAP_VIZ_DIR` env vars for sandboxed runs.
