# CLAUDE.md — Mana Map

## What This Project Is

Mana Map is an MTG card embedding pipeline that downloads all ~33,700 oracle cards from Scryfall, trains lightweight neural networks to embed them into a 128-dim vector space, then projects them to 2D for interactive visualization. It supports **two maps** (Color+Type and Abilities), **synergy detection**, and **power creep analysis**. The entire pipeline runs locally on a Mac.

## Environment

- **Python 3.10** via conda `py310` env — PyTorch does NOT have wheels for Python 3.14
- **Venv**: `.venv` in project root (created from the conda py310 base)
- **Device**: MPS (Apple Silicon) for training; falls back to CUDA then CPU
- **Key version constraints**: `sentence-transformers<4`, `numpy<2` (required for PyTorch 2.2.2 compat)
- Install deps: `pip install -r requirements.txt`
- Run everything: `python pipeline.py`

## Pipeline Steps

| Step | File | What it does | Output |
|------|------|-------------|--------|
| 1 | `download.py` | Fetches Scryfall oracle_cards bulk JSON | `data/oracle-cards.json` |
| 2 | `extract.py` | Parses JSON → flat CSV with derived columns (supertype, primary_color, mechanical_tags, embedding_text) | `data/cards.csv` (~33,700 rows) |
| 3 | `preprocess.py` | Sentence embeddings (all-MiniLM-L6-v2, frozen), categorical encoding, keyword multi-hot, mechanical tag multi-hot | `data/text_embeddings.npy`, `data/card_features.npz`, `data/color_vectors.npy`, `data/mechanical_tags.npy` |
| 4a | `train.py` | Triplet margin loss training — positives by (supertype, primary_color) groups | `data/model.pt` |
| 4b | `train_ability.py` | Triplet margin loss training — positives by mechanical tag overlap (>=2 shared tags) | `data/model_ability.pt` |
| 5 | `embed.py` | Runs all cards through both trained models, builds metadata CSV | `data/embeddings.npy`, `data/embeddings_ability.npy`, `data/card_metadata.csv` |
| 6 | `reduce.py` | PaCMAP dimensionality reduction (128D → 2D), exports both projections | `data/projection_2d.json`, `data/projection_2d_ability.json` |
| 7 | `download_combos.py` | Paginates Commander Spellbook API for combo data | `data/combos_raw.json` |
| 8 | `process_combos.py` | Builds combo partner graph from raw combos + cards.csv | `data/combo_graph.json` |
| 9 | `export_embeddings.py` | Converts both embeddings.npy to raw Float32 binary for JS | `data/embeddings.bin`, `data/embeddings_ability.bin` |
| 10 | `synergy.py` | Builds synergy partner graph using complementary mechanical tags | `data/synergy_graph.json` |
| 11 | `power_creep.py` | Detects strictly-better card replacements (power creep) | `data/obsolescence_index.json` |

`pipeline.py` orchestrates all steps in order. Steps 10-11 are wrapped in try/except ImportError for forward compatibility.

## Architecture

### Model (`model.py`)

**CardEmbeddingModel**: parameterized fusion MLP, 128-dim L2-normalized output.

Constructor accepts optional parameters to customize for different training objectives:
- `ci_emb_dim` — Color identity embedding dim (default: 32 from config)
- `keyword_emb_dim` — If > 0, keywords pass through nn.Linear+ReLU learned projection; if 0/None, raw multi-hot passthrough
- `mechanical_tag_dim` — Number of mechanical tag inputs (0 = no tag input)
- `mechanical_tag_emb_dim` — If > 0 with mechanical_tag_dim > 0, learns tag embeddings

Forward pass: categorical embeddings + text embedding + continuous features + keywords (+ optional tags) → concat → 3-layer MLP (hidden→ReLU→dropout→128→ReLU→dropout→128) → L2 normalize.

**Default model (Color+Type):**
- Inputs: text(384) + supertype(16) + rarity(8) + ci(32) + layout(16) + continuous(2) + keywords(50) = 508-dim
- ~181K trainable params
- No mechanical tag input

**Ability model:**
- Inputs: text(384) + supertype(16) + rarity(8) + ci(**8**, shrunk) + layout(16) + continuous(2) + keywords(**50→32** learned) + tags(**33→32** learned) = 498-dim
- ~180K trainable params
- Color identity deliberately shrunk (32→8 dim) so the model *can't* rely on color
- Tags get a full 32-dim learned projection, giving the model rich ability representation
- Result: cards cluster by function (all blink cards together regardless of color)

Text encoder (all-MiniLM-L6-v2) is **frozen** — only the MLP trains.

### Training

**`train.py`** (Color+Type):
- Positives = same (supertype, primary_color) group
- Negatives = different supertype AND different color (rejection sampled, 50 attempts)
- Fallback chain for positives: same group → same supertype → same color → random

**`train_ability.py`** (Abilities):
- Positives = share >=2 mechanical tags (`MIN_SHARED_TAGS_POSITIVE`)
- Negatives = share 0 mechanical tags
- Fallback chain for positives: >=2 shared → >=1 shared → random

**Both models:**
- Triplet margin loss (margin=0.3), batch size 256, Adam lr=1e-3, 90/10 train/val split
- Early stopping with patience=5 (`EARLY_STOPPING_PATIENCE`)
- Color+Type: up to 40 epochs (`NUM_EPOCHS`), typically stops ~epoch 7 (converges fast, near-zero loss)
- Abilities: up to 100 epochs (`ABILITY_NUM_EPOCHS`), typically stops ~epoch 16 (tag groups are fuzzier)
- Best model saved on val_loss improvement

### Supertype Classification (`extract.py`)

`SUPERTYPE_PRIORITY` in `config.py` determines classification for multi-typed cards. The first match wins:
```
Planeswalker > Battle > Land > Creature > Instant > Sorcery > Enchantment > Artifact
```
**Land is before Creature** so "Land Creature" cards (Dryad Arbor, Jasconian Isle) classify as Land, not Creature. This prevents false positives in power creep detection (land creatures being compared against real creatures).

### Mechanical Tags (`mechanical_tags.py`)

33 regex-based mechanical tags extracted from oracle text, defined in `config.py` `MECHANICAL_TAGS` dict. Patterns applied case-insensitive. `MECHANICAL_TAG_DIM = len(MECHANICAL_TAGS) = 33`.

**Triggers (5):** `etb`, `death_trigger`, `attack_trigger`, `damage_trigger`, `upkeep_trigger`
**Effects (9):** `sacrifice`, `draw`, `removal`, `bounce`, `counterspell`, `blink`, `reanimate`, `tutor`, `discard`
**Generators (6):** `tokens`, `counters_plus`, `counters_minus`, `ramp`, `lifegain`, `mill`
**Modifiers (8):** `anthem`, `cost_reduction`, `copy`, `protection`, `evasion_flying`, `evasion_trample`, `evasion_menace`, `evasion_unblockable`
**Permanents (3):** `equipment`, `aura`, `tap_ability`
**Graveyard (1):** `graveyard_matters`
**Storm (1):** `storm`

Evasion is split into 4 granular tags (not one combined "evasion" tag) so that power creep doesn't treat flying as equivalent to trample.

Coverage: >=70% of non-land cards (actual ~80%). Used for: ability model training signal, synergy detection, power creep comparison.

`MECHANICAL_TAG_NAMES = sorted(MECHANICAL_TAGS.keys())` — tags are always in sorted order for consistent multi-hot encoding.

### Synergy Detection (`synergy.py`)

Synergies are **complementary** — finds cards that *complete* each other, NOT cards that do the same thing. Different from "Find Similar" which finds embedding neighbors.

14 rules in `config.py` `SYNERGY_RULES`, each a `(tag_A, tag_B, label)` tuple:

| Card A has | Card B has | Label |
|-----------|-----------|-------|
| `blink` | `etb` | Blink + ETB |
| `sacrifice` | `death_trigger` | Sac + Death Trigger |
| `tokens` | `anthem` | Tokens + Anthem |
| `reanimate` | `death_trigger` | Reanimate + Death Trigger |
| `tokens` | `sacrifice` | Tokens + Sacrifice |
| `draw` | `discard` | Draw + Discard |
| `mill` | `graveyard_matters` | Mill + Graveyard |
| `counters_plus` | `tokens` | Counters + Tokens |
| `copy` | `etb` | Copy + ETB |
| `storm` | `cost_reduction` | Storm + Cost Reduction |
| `sacrifice` | `etb` | Sac + ETB |
| `reanimate` | `etb` | Reanimate + ETB |
| `lifegain` | `death_trigger` | Lifegain + Death Trigger |
| `anthem` | `tokens` | Anthem + Tokens |

Rules are applied bidirectionally (both forward and reverse). Known combo partners (from `combo_graph.json`) are excluded to surface NEW synergies. Top 5 partners per card, ranked by rule count + embedding similarity tiebreaker.

### Power Creep (`power_creep.py`)

Detects "strictly-better" card replacements. Card B obsoletes Card A if **all** hold:
1. Same supertype (grouped for efficiency)
2. B.cmc <= A.cmc
3. B's color requirement same or easier (pip comparison via `parse_color_requirement`)
4. B has all of A's mechanical tags (superset)
5. B has same or better power/toughness (creatures)
6. B has at least one concrete advantage (lower CMC, better stats, or additional tags)
7. B was printed later than A

**Exclusions (cards skipped entirely):**
- Supertypes "Land" and "Unknown" are never compared
- Vanilla cards (no tags) are skipped — nothing meaningful to compare
- Cards with empty/NaN `mana_cost` are skipped (augments, tokens, some special cards)
- Modifier stats (`+2`, `-1`) return None from `parse_stat` — augment/host cards don't get real stat values

Up to 5 replacements per card, sorted by number of advantages.

### Visualization (`viz/`)

- `viz/index.html` — HTML structure with toolbar, plot, detail panel, deck panel
- `viz/css/mana-map.css` — All CSS (~294 lines): explore + deck builder + synergy + obsolescence + responsive
- `viz/js/mana-map.js` — Explore mode (~1159 lines): multi-map selector, rendering, search, toggles, detail panel, multi-card selection, find similar, find synergies, obsolescence display, keyboard nav, pinch zoom. Exposes shared state on `window.MM`.
- `viz/js/deck-builder.js` — Deck builder (~1299 lines): state, UI, recommendation algorithm (with synergy scoring), mana base generator, analytics, export. Exposes API on `window.DeckBuilder`.
- Plotly.js 2.35.2 from CDN (`scattergl` for WebGL)

**Two modes**: Explore (default) and Build Deck, toggled via toolbar dropdown.

Dark theme (#1a1a2e background, #c4a747 gold accents). Serves via `python -m http.server 8000` → `http://localhost:8000/viz/index.html` (must serve from project root — viz fetches `../data/` relative paths).

#### Explore Mode (`mana-map.js`)

**Map System:**
- **Map Selector**: Switch between Color+Type and Abilities maps (projections + embeddings cached for instant switching)
- **Color By**: 3 palettes — Primary Color (W/U/B/R/G/Colorless/Multicolor), Supertype (9 types), Rarity (common/uncommon/rare/mythic/bonus/special)
- **Supertype Toggles**: 9 filter buttons (Creature/Instant/Sorcery/Enchantment/Artifact/Land/Planeswalker/Battle/Unknown) — active toggles show purple bg with gold border, inactive hides those cards from the plot

**Search (4-tier fallback):**
1. Exact name match
2. Name starts with query
3. Name includes query
4. Oracle text search (capped at 200 results, white diamond markers with orange outline)

**Multi-Card Selection (up to 8):**
- Click: select single card (replaces selection)
- Shift+Click: toggle card in/out of selection stack
- Shift+Drag: box select up to 8 cards (visual "⇧ Multi-select" hint when Shift held)
- Stack tabs in detail panel show name, mana, P/T, type, CMC badge per card
- "Top card" concept — detail panel shows the top card's full info

**Keyboard Shortcuts:**
- `←` `→` `↑` `↓`: navigate through card stack
- `1`-`8`: jump to stack position
- `Delete`/`Backspace`: remove current card from stack
- `Escape`: clear selection (or exit deck builder in build mode)
- `/`: focus search input

**Card Detail Panel (350px, 280px on mobile):**
- Card image (Scryfall API, lazy load, fallback on error)
- Quick stats line: mana symbols, P/T or loyalty/defense, rarity pill (color-coded)
- Type line
- Obsolescence section (if applicable): up to 3 strictly-better replacements with clickable names and advantage badges (red themed)
- Oracle text (pre-wrap, splits "//" faces)
- Keyword badges
- Details: color identity, CMC, EDHREC rank
- Format legality: 8 formats (standard/modern/legacy/vintage/commander/pioneer/pauper/historic)
- **Find Similar** button (gold): 20 nearest in 128D cosine similarity, orange diamonds, auto-populates stack with reference + top 7
- **Find Synergies** button (magenta): complementary tag matches, magenta (#E040FB) diamonds with synergy labels, auto-populates stack with reference + top 7

**Mana Symbol Rendering:**
- Inline colored circles (20px, 16px small, 14px deck cards)
- Supports: {W}, {U}, {B}, {R}, {G}, {C}, {X}, numeric {1}{2}{3}, hybrid {W/U}
- Used throughout: card viewer, stack tabs, deck builder

**Mobile Support:**
- Custom 2-finger pinch-to-zoom (Plotly scattergl doesn't support natively)
- `touch-action: none` on plot div
- Responsive: deck panel fullscreen on <768px, detail panel shrinks to 280px, toolbar wraps

#### Deck Builder (`deck-builder.js`)

**Format Support:** 8 formats (standard/modern/legacy/vintage/commander/pioneer/pauper/historic). Commander: 100 cards, singleton, 0 sideboard. 60-card formats: 60 cards, 4x max, 15 sideboard.

**Workflow:**
1. Select format, optionally set commander (legendary creature autocomplete with 200ms debounce)
2. Click cards on map to add as seeds (gold star markers)
3. Set type distribution targets (editable per-type with progress bars, "Defaults" button for format presets)
4. Generate Recommendations → top 20 scored cards

**Recommendation Algorithm (5-factor weighted scoring):**
- **40% embedding similarity** (cosine to deck centroid)
- **20% combo bonus** (from `combo_graph.json`)
- **20% synergy bonus** (from `synergy_graph.json`, normalized: 5+ matches = max)
- **10% EDHREC popularity** (normalized rank)
- **10% keyword overlap** (Jaccard similarity)

Filters: format legal, color identity subset, type distribution not full, singleton (commander), not rejected.

**Recommendation UI:** Expandable cards with accept (✓) / reject (✗) buttons. Expanded view shows score breakdown chips (similarity/combo/synergy/popularity/keyword), synergy labels, combo partners, keywords, oracle text. "Accept All" button at top.

**Mana Base Generator:**
- Auto-generates lands based on deck's color pip requirements
- Greedy set cover: scores lands by colors covered × 10 + basic subtype bonus + EDHREC × 3 - ETB tapped penalty
- Commander extras: auto-adds Command Tower if 2+ colors
- Fills to distribution target (36 commander, 24 sixty-card)

**Deck Analytics:**
- Mana curve: 7 buckets (0-6+), gold bars
- Color distribution: pip percentages with mana symbols (W/U/B/R/G)

**Deck List:** Grouped by type, per-group progress headers, alphabetical within groups. Remove buttons for seeds/accepted (not auto-lands).

**Export:** Text format (`1 Card Name` per line, commander first), copy to clipboard.

**Plot Overlay Traces:**
- Seeds: gold stars (size 12)
- Commander: large gold star (size 16, double outline)
- Accepted: green circles (size 8)
- Recommendations: orange diamonds (size 9)
- Lands: blue squares (size 7)
- Dimming: format-illegal and CI-violating cards fade to 0.08 opacity

**Persistence:** Deck state saved to LocalStorage (`manamap-deck` key).

## Configuration (`config.py`)

All constants live here — paths, hyperparameters, vocab sizes, embedding dims, mechanical tag patterns, synergy rules. No magic numbers in other files.

**Key paths:**
- `DATA_DIR = Path("data")` — all data artifacts
- `VIZ_DIR = Path("viz")` — visualization files
- `PROJECTION_PATH = DATA_DIR / "projection_2d.json"`
- `ABILITY_PROJECTION_PATH = DATA_DIR / "projection_2d_ability.json"`

**Key hyperparameters:**
- `BATCH_SIZE = 256`, `NUM_EPOCHS = 40`, `ABILITY_NUM_EPOCHS = 100`
- `LEARNING_RATE = 1e-3`, `TRIPLET_MARGIN = 0.3`, `VAL_SPLIT = 0.1`
- `EARLY_STOPPING_PATIENCE = 5`
- `MIN_SHARED_TAGS_POSITIVE = 2` (for ability model positive mining)
- `MLP_HIDDEN_DIM = 256`, `MLP_DROPOUT = 0.1`, `FINAL_EMBEDDING_DIM = 128`

**Ability model overrides:**
- `ABILITY_CI_EMBEDDING_DIM = 8` (vs default 32)
- `ABILITY_KEYWORD_EMBEDDING_DIM = 32` (vs default: raw 50-dim passthrough)
- `ABILITY_MECHANICAL_TAG_EMBEDDING_DIM = 32`

## Source Files

| File | Purpose |
|------|---------|
| `config.py` | All constants: paths, hyperparameters, vocab sizes, tag patterns, synergy rules |
| `download.py` | Step 1: Fetch Scryfall bulk data with idempotent `.download-meta.json` sidecar |
| `extract.py` | Step 2: Parse JSON → flat CSV, derive supertype/primary_color/mechanical_tags/embedding_text |
| `mechanical_tags.py` | `tag_oracle_text(text)` → sorted list of tags; `encode_tags_multihot(df, tag_names)` → numpy array |
| `preprocess.py` | Step 3: Sentence embeddings, categorical encoding, keyword/tag multi-hot, color vectors |
| `model.py` | `CardEmbeddingModel` — parameterized fusion MLP, 128-dim L2-normalized output |
| `train.py` | Step 4a: Color+Type triplet training. Also exports `get_device()`, `run_epoch()`, `collate_triplets()` |
| `train_ability.py` | Step 4b: Ability triplet training. Imports `get_device`, `collate_triplets` from `train.py` |
| `embed.py` | Step 5: Run both models → embeddings + card_metadata.csv |
| `reduce.py` | Step 6: PaCMAP 128D → 2D for both projections |
| `download_combos.py` | Step 7: Paginate Commander Spellbook API |
| `process_combos.py` | Step 8: Build combo partner graph |
| `export_embeddings.py` | Step 9: Convert .npy → raw Float32 .bin for JS |
| `synergy.py` | Step 10: Build synergy graph from complementary tag rules |
| `power_creep.py` | Step 11: Detect strictly-better replacements |
| `pipeline.py` | Orchestrator: runs all 11 steps in order |

## Data Artifacts (all in `data/`, gitignored unless noted)

| File | Shape / Size | Description |
|------|-------------|-------------|
| `oracle-cards.json` | ~200MB | Raw Scryfall bulk data |
| `cards.csv` | ~33,700 rows | Extracted flat CSV (includes `mechanical_tags` column) |
| `text_embeddings.npy` | (~33700, 384) ~49MB | Frozen sentence embeddings |
| `card_features.npz` | varies | Categorical indices, continuous features, keywords, mechanical_tags |
| `mechanical_tags.npy` | (~33700, 33) | Multi-hot mechanical tag vectors |
| `color_vectors.npy` | (~33700, 5) | WUBRG binary vectors |
| `model.pt` | ~711KB | Color+Type model checkpoint |
| `model_ability.pt` | ~711KB | Ability model checkpoint |
| `embeddings.npy` | (~33700, 128) ~16MB | Color+Type embeddings |
| `embeddings_ability.npy` | (~33700, 128) ~16MB | Ability embeddings |
| `card_metadata.csv` | ~33,700 rows | oracle_id, name, supertype, colors, cmc, rarity |
| `projection_2d.json` | ~13MB | Color+Type 2D PaCMAP projection |
| `projection_2d_ability.json` | ~13MB | Ability 2D PaCMAP projection |
| `combos_raw.json` | ~50-100MB | Raw Commander Spellbook data (gitignored) |
| `combo_graph.json` | ~23MB | Combo partner graph (git tracked) |
| `synergy_graph.json` | ~8MB | Synergy partner graph (git tracked) |
| `obsolescence_index.json` | ~8MB | Strictly-better replacements (git tracked) |
| `embeddings.bin` | ~16MB | Color+Type Float32 binary for JS (git tracked) |
| `embeddings_ability.bin` | ~16MB | Ability Float32 binary for JS (git tracked) |

**Note:** Card count grows as Scryfall adds new sets. Exact count as of latest run: 33,682.

## Tests

```bash
# Unit + integration tests (193 total, no data files required for unit tests)
pytest test_extract.py test_preprocess.py test_combos.py test_mechanical_tags.py test_synergy.py test_power_creep.py test_pipeline_integration.py -v

# Embedding quality tests (requires data files from pipeline run)
pytest test_find_similar.py -v
```

| File | Tests | What it covers |
|------|-------|----------------|
| `test_extract.py` | 51 | Multi-face card handling, derived column logic, supertype classification |
| `test_preprocess.py` | 23 | Vocab building, categorical encoding, normalization, keyword multi-hot, color vector parsing |
| `test_combos.py` | 15 | Combo data extraction, graph building, filtering, deduplication |
| `test_mechanical_tags.py` | 42 | All 33 tag regex patterns, edge cases, multi-hot encoding |
| `test_synergy.py` | 14 | Synergy rule matching, bidirectionality, combo exclusion, ranking |
| `test_power_creep.py` | 26 | Strictly-better detection, stat parsing (incl. modifier rejection), empty mana cost exclusion, edge cases |
| `test_pipeline_integration.py` | 22 | End-to-end validation of all pipeline outputs (requires `data/` artifacts) |
| `test_find_similar.py` | 12 | Embedding binary format, L2 normalization, cosine similarity, 128D vs 2D ranking divergence (requires `data/` artifacts) |

**Total: 205 tests** (193 standard + 12 embedding quality).

## Common Tasks

**Run the full pipeline:**
```bash
python pipeline.py
```

**Run individual pipeline steps:**
```bash
python download.py        # Step 1 (requires internet)
python extract.py         # Step 2
python preprocess.py      # Step 3 (slow — computes sentence embeddings)
python train.py           # Step 4a (Color+Type model)
python train_ability.py   # Step 4b (Ability model)
python embed.py           # Step 5 (both models)
python reduce.py          # Step 6 (both projections)
python download_combos.py # Step 7 (requires internet, ~2.5 min)
python process_combos.py  # Step 8
python export_embeddings.py # Step 9 (both .bin files)
python synergy.py         # Step 10
python power_creep.py     # Step 11
```

**Serve the visualization:**
```bash
python -m http.server 8000
# http://localhost:8000/viz/index.html
```

**Re-run just synergy + power creep (no retraining needed):**
```bash
python synergy.py && python power_creep.py
```

## Gotchas

- `pacmap` requires `numba` + `llvmlite`. On macOS, install specific versions with pre-built wheels: `pip install llvmlite==0.41.1 numba==0.58.1` before `pip install pacmap`
- The `data/` directory is mostly gitignored. Run `python pipeline.py` to regenerate from scratch (requires internet for Steps 1 and 7)
- The viz fetches `../data/projection_2d.json` relative to `viz/index.html` — must serve from project root
- Color+Type model converges to near-zero triplet loss very quickly (epoch 2-3); this is expected since the margin (0.3) is easily satisfied with color/type groups
- The ability model uses tag-overlap mining — it converges slower since tag groups are fuzzier (best val_loss ~0.05)
- Synergy detection is **complementary** (blink finds ETB), not **similar** (ETB finds ETB). These are fundamentally different algorithms.
- Changing `MECHANICAL_TAGS` in config.py changes `MECHANICAL_TAG_DIM`, which makes existing `model_ability.pt` checkpoints incompatible. Must retrain (steps 3-5) after any tag change.
- `PLAN.md` in root is a historical planning document from the deck builder feature. It contains outdated numbers and is not maintained.
