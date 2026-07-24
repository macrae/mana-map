# Architecture

## Overview

Mana Map embeds every Magic: The Gathering oracle card (~34,300 and growing) into a 128-dim vector space using two lightweight fusion MLPs, then projects to 2D with PaCMAP for the interactive map. On top of the embeddings sit three analysis layers: synergy detection, power-creep detection, and region clustering.

## Models (`src/manamap/training/model.py`)

**CardEmbeddingModel**: parameterized fusion MLP, 128-dim L2-normalized output.

Constructor parameters customize the model for different training objectives:
- `ci_emb_dim` — color identity embedding dim (default 32)
- `keyword_emb_dim` — if > 0, keywords pass through a learned Linear+ReLU projection; if 0/None, raw multi-hot passthrough
- `mechanical_tag_dim` — number of mechanical tag inputs (0 = no tag input)
- `mechanical_tag_emb_dim` — if > 0 with `mechanical_tag_dim` > 0, learns tag embeddings

Forward pass: categorical embeddings + text embedding + continuous features + keywords (+ optional tags) → concat → 3-layer MLP (hidden→ReLU→dropout→128→ReLU→dropout→128) → L2 normalize.

**Color+Type model** (`model.pt`):
- Inputs: text(384) + supertype(16) + rarity(8) + ci(32) + layout(16) + continuous(2) + keywords(50) = 508-dim
- ~181K trainable params, no mechanical tag input

**Ability model** (`model_ability.pt`):
- Inputs: text(384) + supertype(16) + rarity(8) + ci(**8**, shrunk) + layout(16) + continuous(2) + keywords(**50→32** learned) + tags(**33→32** learned) = 498-dim
- ~180K trainable params
- Color identity deliberately shrunk (32→8) so the model *can't* lean on color
- Result: cards cluster by function — all blink cards together regardless of color

The text encoder (all-MiniLM-L6-v2) is **frozen** — only the MLP trains.

## Training (`src/manamap/training/`)

Shared utilities (`get_device`, `collate_triplets`, `run_epoch`) live in `training/common.py`.

**`train.py`** (Color+Type):
- Positives = same (supertype, primary_color) group
- Negatives = different supertype AND different color (rejection sampled, 50 attempts)
- Fallback chain for positives: same group → same supertype → same color → random

**`train_ability.py`** (Abilities):
- Positives = share >= 2 mechanical tags (`MIN_SHARED_TAGS_POSITIVE`)
- Negatives = share 0 mechanical tags
- Fallback chain: >= 2 shared → >= 1 shared → random

**Both:** triplet margin loss (margin 0.3), batch 256, Adam lr 1e-3, 90/10 split, early stopping patience 5. Color+Type converges near-zero loss by ~epoch 7 (expected — color/type groups easily satisfy the margin); the ability model typically stops ~epoch 16 (tag groups are fuzzier, best val_loss ~0.05).

## Supertype classification (`src/manamap/ingest/extract.py`)

`SUPERTYPE_PRIORITY` in `config.py` — first match wins:

```
Planeswalker > Battle > Land > Creature > Instant > Sorcery > Enchantment > Artifact
```

**Land is before Creature** so "Land Creature" cards (Dryad Arbor, Jasconian Isle) classify as Land, preventing power-creep false positives against real creatures.

## Mechanical tags (`src/manamap/mechanical_tags.py`)

33 regex-based tags extracted from oracle text, defined in `config.py` `MECHANICAL_TAGS`. Case-insensitive matching. Coverage ~80% of non-land cards.

- **Triggers (5):** `etb`, `death_trigger`, `attack_trigger`, `damage_trigger`, `upkeep_trigger`
- **Effects (9):** `sacrifice`, `draw`, `removal`, `bounce`, `counterspell`, `blink`, `reanimate`, `tutor`, `discard`
- **Generators (6):** `tokens`, `counters_plus`, `counters_minus`, `ramp`, `lifegain`, `mill`
- **Modifiers (8):** `anthem`, `cost_reduction`, `copy`, `protection`, `evasion_flying`, `evasion_trample`, `evasion_menace`, `evasion_unblockable`
- **Permanents (3):** `equipment`, `aura`, `tap_ability`
- **Graveyard (1):** `graveyard_matters` · **Storm (1):** `storm`

Evasion is split into 4 granular tags so power creep doesn't treat flying as equivalent to trample. `MECHANICAL_TAG_NAMES = sorted(MECHANICAL_TAGS.keys())` — always sorted for consistent multi-hot encoding.

**Changing `MECHANICAL_TAGS` changes `MECHANICAL_TAG_DIM`, invalidating `model_ability.pt`. Retrain steps 3–5 after any tag change.**

## Synergy detection (`src/manamap/analysis/synergy.py`)

Synergies are **complementary** — cards that *complete* each other (blink finds ETB), NOT cards that do the same thing (that's "Find Similar", which uses embedding neighbors).

24 rules in `config.py` `SYNERGY_RULES`, each `(tag_A, tag_B, label)`, covering 27/33 tags. Examples: blink+etb, sacrifice+death_trigger, tokens+anthem, mill+graveyard_matters, storm+cost_reduction, aura+protection, counterspell+draw. Rules apply bidirectionally. Known combo partners (from `combo_graph.json`) are excluded to surface NEW synergies. Top `SYNERGY_MAX_PARTNERS` (10) per card, ranked by rule count with ability-embedding cosine similarity as tiebreaker (falls back to color+type embeddings).

The graph is **format-agnostic by design** — filtering happens at consumption time (deck builder filters by legality + color identity; explore mode shows everything).

Shared helpers (`parse_tag_set`, `cosine_similarity`, `load_first_embeddings`) live in `analysis/common.py`.

## Power creep (`src/manamap/analysis/power_creep.py`)

Card B obsoletes Card A only if **all** hold:
1. Same supertype (Land and Unknown never compared)
2. Cosine similarity gate in ability embedding space — **tiered**: 1-tag cards need >= 0.98 (`OBSOLESCENCE_SINGLE_TAG_THRESHOLD`), 2+-tag cards >= 0.75 (`OBSOLESCENCE_SIMILARITY_THRESHOLD`). Keeps iconic single-tag upgrades (Doom Blade → Fatal Push, sim 0.999) while filtering false positives.
3. B.cmc <= A.cmc
4. B's color requirement same or easier (pip comparison)
5. B has all of A's mechanical tags (superset)
6. B has same or better power/toughness (creatures)
7. B has at least one concrete advantage
8. B printed later than A

Exclusions: cards with < `OBSOLESCENCE_MIN_TAGS` (1) tags, empty/NaN mana cost (augments, tokens), modifier stats (`+2`) parse to None. Up to `OBSOLESCENCE_MAX_REPLACEMENTS` (5) per card, sorted by similarity then advantage count. Batch matrix multiply per supertype group for performance.

## Region clustering (`src/manamap/analysis/cluster_regions.py`)

HDBSCAN names geographic-style regions on both 2D maps at two zoom levels:
- **L0 mega-regions**: `min_cluster_size=800`, `min_samples=50` → ~10–15 regions (zoomed out)
- **L1 sub-regions**: `min_cluster_size=100`, `min_samples=15` → ~40–120 regions (zoomed in)

Naming — Color+Type map: dominant color (>= 40%) + type (>= 30%), guild names for 2-color pairs (>= 50%), fallback to top tag. Abilities map: TF-IDF-like scoring (cluster tag freq / global freq), top 1–2 overrepresented tags, minimum presence threshold `REGION_MIN_TAG_PRESENCE=0.10`. Label dedup: tag suffixes (max 2 for L0, 3 for L1), then spatial direction (N/S/E/W).

**Index-alignment invariant**: `projection[i]` corresponds exactly to `cards.csv[i]` (maintained through embed → reduce). Tags are looked up by direct index, never by name (duplicate card names exist).
