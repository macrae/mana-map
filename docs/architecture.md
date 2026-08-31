# Architecture

## Overview

Mana Map embeds every Magic: The Gathering oracle card (~34,900 and growing) into a 128-dim vector space using two lightweight fusion MLPs, then projects to 2D with PaCMAP for the interactive map. On top of the embeddings sit four analysis layers: synergy detection, power-creep detection, region clustering, and the deckbuilding role taxonomy.

## Models (`src/manamap/training/model.py`)

*A reader-facing overview of the same material — feature decomposition, the loss
functions, the eval harness — is in README.md under "The embedding models". This
section is the implementation reference; keep the two consistent.*

**CardEmbeddingModel**: parameterized fusion MLP, 128-dim L2-normalized output.

Constructor parameters customize the model for different training objectives:
- `ci_emb_dim` — color identity embedding dim (default 32)
- `keyword_emb_dim` — if > 0, keywords pass through a learned Linear+ReLU projection; if 0/None, raw multi-hot passthrough
- `mechanical_tag_dim` — number of mechanical tag inputs (0 = no tag input)
- `mechanical_tag_emb_dim` — if > 0 with `mechanical_tag_dim` > 0, learns tag embeddings
- `structured_dim` — width of the `[power/toughness(3), mana pips(6), colour(6)]` block. 0 keeps the pre-Phase-1 input
- `text_passthrough` — carve a fixed-weight projection of the frozen text out of the output (below). The layout model leaves this **off**: its job is colour/type organisation and a text floor would fight it

Forward pass: categorical embeddings + text embedding + continuous features + keywords
(+ optional tags, + optional structured) → concat → 3-layer MLP
(hidden 256 → ReLU → dropout 0.1 → 128 → ReLU → dropout 0.1 → out) → L2 normalize.

**Layout model** (`model.pt`) — colour and type, feeds `projection_2d.json` only:
- Inputs: text(384) + supertype(16) + rarity(8) + ci(32) + layout(16) + continuous(2) + keywords(50) = **508-dim**
- **181,272** trainable params; no mechanical tag input, no structured block
- Trunk emits the full 128 and returns `F.normalize(x)`

**Function model** (`model_ability.pt`) — what a card DOES, the sole source of similarity:
- Inputs: text(384) + supertype(16) + rarity(8) + ci(**8**, shrunk) + layout(16) + continuous(2) + keywords(**50→32** learned) + tags(**33→32** learned) + structured(**15**) = **513-dim**
- **192,672** trainable params
- Colour identity deliberately shrunk (32→8) so the model *can't* lean on colour
- Trunk emits **96**, not 128 — see the output split below
- Result: cards cluster by function — all blink cards together regardless of colour

### The output split (`text_passthrough=True`)

The function model's 128 dims are **two independently L2-normalized halves**: 96 learned,
plus 32 from a `Linear(384, 32)` projection of the frozen text. Each is scaled so the
squared weights sum to 1 (`√(1−W)` and `√W`, `W = TEXT_PASSTHROUGH_WEIGHT = 0.3`), which
makes the concatenation unit-norm and the dot product of two cards **exactly**

```
sim(a, b) = 0.7 · cos_learned(a, b) + 0.3 · cos_text(a, b)
```

The scales are `register_buffer`s, not plain floats, so they move with `.to(device)` and are
saved in the checkpoint — the weights are meaningless without them.

Why it exists: the previous function model scored **0.093 recall@10 against 0.187 for the
frozen text it was built from**, using 5.97 of its 128 dimensions. Training was subtractive
— a model free to discard the text did exactly that. This makes discarding it structurally
impossible and the floor arithmetic rather than hope. `text_proj` has **no ReLU** on
purpose: this half exists to preserve the text geometry and a rectifier folds half of it away.

The text encoder (all-MiniLM-L6-v2) is **frozen** — only the MLP trains.

## Training (`src/manamap/training/`)

Shared utilities (`get_device`, `collate_triplets`, `run_epoch`) live in `training/common.py`.

**`train.py`** (Color+Type):
- Positives = same (supertype, primary_color) group
- Negatives = different supertype AND different color (rejection sampled, 50 attempts)
- Fallback chain for positives: same group → same supertype → same color → random

`train.py` uses a **triplet margin loss** (margin 0.3): the layout task is "same colour,
same type", the supervision is exact, and a margin is the right shape for it.

**`train_ability.py`** (Function) uses a different objective, because the function task is
harder and a margin loss is the wrong instrument for it:

- **In-batch InfoNCE** (`INFONCE_TEMPERATURE = 0.05`), symmetric, over L2-normalised
  embeddings. A margin loss stops teaching the moment it is satisfied; a contrastive
  softmax keeps ranking the whole batch. There is no explicit negative — every other item
  in the batch is one.
- **Positives mined from `card_roles.json`, rarest shared role first**, so the pairs that
  teach most are the ones a common role would have drowned out. `ROLE_BODY_FALLBACK` is
  excluded, because it labels every creature and would make "creature" the lesson.
- **A fixed-weight text passthrough** makes similarity exactly
  `0.7·cos_learned + 0.3·cos_text`. The weight is structural, not learned, so the model
  *cannot* discard the frozen text it was built from — which is what the previous
  architecture allowed and what made it lose to its own input.

**Both:** batch 256, Adam lr 1e-3, 90/10 split, early stopping patience 5.

### The two spaces do different jobs

**Layout** (`train.py` → `embeddings.npy`) organises the map by colour and type. That is what
a *picture* wants, and it is all this model is asked for — it feeds `projection_2d.json` and
nothing else. It still converges to near-zero loss in a few epochs, and that is fine for a
task whose whole content is "same colour, same type".

**Function** (`train_ability.py` → `embeddings_ability.npy`) answers whether two cards do the
same job. Every "find similar" answer in the product comes from here, on either map.

### The function model, measured (`manamap eval-embeddings`, step 15)

Held-out `test` split of `data/eval/similarity_golden.json`:

Held-out `test` split: 28 groups, **308 queries** (`dev` is 12 groups / 218 queries and
was consumed while diagnosing — always quote `test`). The query counts were stated as
107/56 until 2026-08-31; those predate the 2026-08-12 corpus refresh.

| space | dim | effective dim | 1st→50th gap | r@10 | r@50 | median rank |
|---|---|---|---|---|---|---|
| layout | 128 | 3.89 | 0.0061 | 0.086 | 0.139 | 1148 |
| frozen MiniLM text (the input) | 384 | 51.39 | 0.1341 | **0.244** | 0.414 | 126 |
| **function** | 128 | **27.31** | 0.0323 | 0.232 | **0.464** | **76** |

Against the model it replaced — 5.97 effective dims, r@10 0.093, median rank 995 — the
rebuild is a large win, and the function space is clearly better at depth than the frozen
text it is built from: recall@50 +0.050 and median rank 126 → 76.

**The `-0.012` at r@10 is NOT a loss. It is a tie, and the table above cannot show that**
— a marginal number has no interval, and a comparison needs one on the DIFFERENCE.

### The candidate pool decides the answer (2026-08-31)

Every figure above ranks each golden card against all **34,890** cards. Nothing in the
product does that: commander search ranks against **79**, Find Similar shows **12**
neighbours, `build-deck` ranks within a colour identity and a pool. So the pool is an
axis, and `eval-embeddings` now reports it — candidates are each group's own targets plus
N most-played distractors, so all 28 groups appear at every size, with a paired bootstrap
over **groups** (queries inside a group are correlated; the group is the unit).

| distractors | function | text | gap | 95% CI on the difference |
|---|---|---|---|---|
| **100** | 0.964 | 0.819 | **+0.145** | **[+0.053, +0.235]** excludes 0 |
| **500** | 0.794 | 0.629 | **+0.165** | **[+0.052, +0.289]** excludes 0 |
| 2,000 | 0.562 | 0.446 | +0.115 | [−0.018, +0.255] |
| 10,000 | 0.363 | 0.311 | +0.052 | [−0.045, +0.162] |
| corpus | 0.227 | 0.240 | −0.013 | [−0.083, +0.058] **spans zero** |

`dev` agrees in sign at every pool size and never reaches significance, which is what 12
groups buys. So: **at corpus scale the two spaces are indistinguishable, and at the pool
sizes the product actually uses the trained space wins** — the opposite of the conclusion
the corpus-wide row supported on its own.

**The design is load-bearing.** Restricting to the top-N and keeping only groups that fit
inside it changes *which groups qualify* as the pool narrows — a selection effect wearing a
pool effect's clothes. Tried first: a clean monotonic +0.200 at pool 500 that collapsed to
5 test / 2 dev groups disagreeing in sign once the groups were genuinely held constant.

**The two halves are complementary, and that is the load-bearing result:**

| | r@10 | r@50 | median rank |
|---|---|---|---|
| learned half alone (96d) | 0.136 | 0.312 | 224 |
| text half alone (32d) | 0.214 | 0.361 | 138 |
| combined (128d) | **0.232** | **0.464** | **76** |

The learned half is *worse alone* than the text half, yet the combination beats both on
every metric — and beats the full 384-dim frozen text at depth. The 32-dim projection also
retains most of the 384-dim text's r@10 (0.214 vs 0.244), which is what makes it cheap
enough to spend a quarter of the output on. This is why positives are deliberately **not**
gated on text similarity: selecting pairs the text already scores highly would make the
learned half a copy of the text half instead of a complement to it.

**Do not tune the mixing weight on this golden set.** Swept on the pre-refresh artifact,
everything in W ∈ [0.15, 0.6] fell inside noise, and the two splits disagreed about the
optimum — selecting on `test` picked 0.45, selecting on `dev` picked 0.15. At 56 dev and
107 test queries that is a sample-size problem, not a tuning opportunity, and re-running the
sweep on today's artifact would not change the argument. The shipped 0.3 was chosen a priori
and fitted to neither split.

**Open:** neighbour spread is 0.0323 against a 0.05 target — the top-50 sit tighter than a
well-separated space would put them. `tests/test_embedding_quality.py` keeps that gate
failing as `xfail(strict=True)` rather than lowering the threshold to match the result.
Hard-negative mining is the obvious next lever, and needs a similarity ceiling: 39% of
cards have a text neighbour above 0.75, so unfiltered mining would label true synonyms as
negatives.

### The input text (`ingest/extract.py:build_embedding_text`)

`"{type}. Cost {mana_cost}. {P}/{T}. {oracle}. Keywords: …"` — **no card name.**

**The name is excluded on purpose.** Including it buys similarity off shared *words*
rather than shared function — *Rhystic Study* → *White Rhystic Study* scores 0.951 that
way, and *Sol Ring* → *Sisay's Ring*. It is also a large fraction of a short card: `Sol
Ring`'s entire text is eleven words, three of them its name. Measured on the held-out
split:

Measured when the decision was taken, on the artifact of the day — a three-way A/B needs
three sets of text embeddings, so these are kept as the decision record rather than re-run
against every corpus refresh:

| text | r@10 | r@50 | median rank |
|---|---|---|---|
| name-led | 0.187 | 0.362 | 159 |
| no name | **0.248** | 0.407 | 129 |
| no name + cost + P/T (shipped) | 0.244 | **0.414** | **124** |

Dropping the name is the win. Cost and P/T are a wash in the *text* (the r@10 difference is
inside noise) and earn their place as structured features instead — they are in the string
because it costs nothing and helps the frozen-text fallback, not because they moved this table.

Note that effective dimensionality *fell* here, 81.0 → ~51, while quality rose. Names added
variance, and variance is not quality: the participation ratio is a collapse **detector**, good
for telling 3 from 50, not a score to maximise. The evaluation is deliberately independent of every training signal (see
`data/eval/similarity_golden.json`), and its `test` split was written after the diagnosis, so
these numbers are not self-graded.

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

## Deckbuilding roles (`src/manamap/analysis/card_roles.py`)

**Roles ≠ mechanical tags**, and the distinction is as load-bearing as "Synergy ≠ Similar" below. `MECHANICAL_TAGS` is a *retrieval* vocabulary — it answers "what is this card like" and it clusters well. Roles answer "what job does this card do in a 99", and the answers have to be **countable**.

The canonical difference: `MECHANICAL_TAGS` has one `ramp` tag matching everything that adds mana or fetches a land, so a Sol Ring, a Llanowar Elves, a Rampant Growth and a Dark Ritual are indistinguishable. `ROLE_PATTERNS` splits that into `ramp:rock` / `ramp:dork` / `ramp:land` / `ramp:ritual` / `ramp:cost-reduction`, **disambiguated by the type line** — an artifact that taps for mana is a rock, a creature is a dork, an instant is a ritual. They cost differently, they die differently, and a curve model that conflates them is wrong.

40 regex patterns in `config.py` `ROLE_PATTERNS`, plus type-derived `land:*`, `ramp:*` and the `threat:body` fallback — **53 emitted roles in 19 families**. Families: `ramp:*`, `draw:*`, `removal:*`, `tutor:*`, `buff:*`, `wincon:*`, `protection:*`, `land:*`, `doubler:*`, `payoff:*`, `threat:*`, plus `recursion`, `counterspell`, `sac-outlet`, `sac-cost`, `stax`, `hate:graveyard`, `value:etb`, `utility:activated`.

**Patterns match templating, not meaning, and that is where they fail.** Every hole found so far has been one card saying the same thing a different way. `removal:edict` demanded `each|target` adjacent to `opponent|player`, so Grave Pact's "each **other** player sacrifices" missed by two words. `wincon:drain` demanded a literal numeral, so Sanguine Bond's "loses **that much** life" and Exquisite Blood's "whenever **an** opponent loses life" both missed — the two halves of the format's most famous loop. `counterspell` demanded the literal "counter target spell", which blanked half of Sisay's interaction. Mondrak reverses Parallel Lives' word order ("tokens **would be created**") and was classified on its ward clause instead. When adding a pattern, test it against the *other* templates for the same effect.

Two rules keep it honest:

- **Lands never carry spell roles.** A land is evaluated only for land quality (fetch / tapped / untapped-dual / utility / basic / mdfc).
- **Coverage is published, not assumed.** The `meta` block reports `coverage` (89.5% of Commander-legal cards) *and* `specific_coverage` (73.2%, excluding cards carrying only the `threat:body` fallback), because labelling every creature a body and stopping would tell a slot filler nothing. `ROLE_COVERAGE_TARGET` / `ROLE_SPECIFIC_COVERAGE_TARGET` are regression floors asserted by the test suite. 100% is explicitly not the goal — the stragglers are genuinely miscellaneous one-off effects (Silence, Teferi Time Raveler, Chaos Warp), and looser regexes would buy coverage with false positives, which is worse than an honest null.

`ROLE_PATTERNS` is a **separate dict from `MECHANICAL_TAGS` on purpose**: roles are not model-facing, so editing them costs only `manamap card-roles` (step 13) rather than a retrain.

## Commander brackets (`src/manamap/pilot/bracket.py`)

Computes a bracket **floor** — the lowest WotC bracket a deck's contents are consistent with — from three mechanically checkable signals, each of which names the specific card or line responsible:

- **Game Changers**, from the `game_changer` column in `cards.csv` (Scryfall carries WotC's list).
- **Combo content.** Commander Spellbook tags every variant, and `COMBO_BRACKET_TAGS` maps its letters onto the ladder (`E`→1, `C`/`O`→2, `P`/`S`→3, `R`→4; `B` is banned and excluded). A separate **two-card-infinite test** runs independently of that tag — which matters, because Spellbook tags a real Hapatra two-card infinite as bracket 1 and the engine returns 4 anyway.
- **Mass land denial**, a curated `MASS_LAND_DENIAL` name list — the one signal not derived from data, and the report says so.

Two deliberate refusals. It never scores **tutor density**: WotC removed that guardrail on 2025-10-21 and now relies on Game Changers membership, so a tutor budget would encode a rule that no longer exists. And it never returns a verdict — brackets are a conversation, so the report says what a deck *contains* and what that is consistent with.

It also excludes lines that assume one of their own pieces is your commander (`"Infinite commander casts"` in `produces` is the tell), encoding Judge's Desk A-004 where CR 903.9a refuted a promised infinite.

## Mana base math (`src/manamap/pilot/manabase.py`)

Colour-source counts are **computed hypergeometrically** (callers must pass `common.expand_copies()` output — `manabase.py` counts whatever list it is handed, and handing it decklist *entries* silently halves every basic-land colour), not quoted from a table: the fewest sources k such that P(at least `pips` among the cards seen by turn T) ≥ 90%, over a 99-card library. Three calibrations keep the output usable. **Conditional sources count only at their unconditional value**: "add one mana of any color" is a five-colour source only when unrestricted, so Haven of the Spirit Dragon ("spend this mana only to cast a Dragon creature spell") taps for `{C}` in a Vampire deck. Understating a source is recoverable; overstating one produces a deck that cannot cast its spells, and a greedy selector reaches for those lands precisely because they *look* like they cover everything. Then — requirements are sized against the pip weight a 20% quorum of a colour's cards actually demand (`PIP_WEIGHT_QUORUM`), so one `{B}{B}{B}` bomb doesn't demand 48 sources; and planning never targets earlier than turn 3 (`MIN_PLANNING_TURN`), since sizing to turn 1 asks for more sources than a 36-land base can hold. Both the target and the *achieved* on-curve probability are reported, so a shortfall is information rather than a failure.

## Mana audit (`src/manamap/pilot/mana_analysis.py`)

The same hypergeometric kit, pointed at a *finished* deck instead of a pool, and the one
measurement with **no agent at all**: `manamap pilot mana-analysis <slug>` writes
`mana_analysis.json` deterministically. It classifies each land (basic / snow / mdfc /
fetch / tapped / untapped-dual / utility, reusing `ROLE_LAND_PATTERNS` deck-locally),
counts land and nonland sources per colour, computes pip share against source share, and
reports on-curve probability both from lands alone and with rocks and dorks.

**It counts copies, not decklist entries.** Basics live in `cards.json` as one entry with
`quantity: N`, so the artifact reports `lands.total` (copies — the real answer) beside
`lands.entries` (distinct cards). Counting entries once published "18 lands" for a 33-land
deck and understated every colour's sources across the whole fleet; `validate-issue` now
lints reader-facing copy that quotes the entry count as a land count.

## Synergy detection (`src/manamap/analysis/synergy.py`)

Synergies are **complementary** — cards that *complete* each other (blink finds ETB), NOT cards that do the same thing (that's "Find Similar", which uses embedding neighbors).

24 rules in `config.py` `SYNERGY_RULES`, each `(tag_A, tag_B, label)`, covering 29/33 tags. Examples: blink+etb, sacrifice+death_trigger, tokens+anthem, mill+graveyard_matters, storm+cost_reduction, aura+protection, counterspell+draw. Rules apply bidirectionally. Known combo partners (from `combo_graph.json`, the partners map) are excluded to surface NEW synergies. Top `SYNERGY_MAX_PARTNERS` (10) per card, ranked by rule count with ability-embedding cosine similarity as tiebreaker (falls back to color+type embeddings).

The graph is **format-agnostic by design** — filtering happens at consumption time (deck builder filters by legality + color identity; explore mode shows everything).

Shared helpers (`parse_tag_set`, `top_k_similar`, `load_first_embeddings`) live in `analysis/common.py`.

### Partners are ranked by playability, not similarity

Ranking is `(-score, -playability)`. Breaking the tie by **embedding similarity would be
backwards**: synergy is a *complementary* relation, so a similarity tiebreak surfaces cards
that resemble the anchor rather than cards that play with it.

The tiebreak decides almost everything, because a score tier is usually large — median 70
cards, p90 1,529. Measured over 4,000 cards:

| | median partner EDHREC rank | in the top 2,000 most-played |
|---|---|---|
| similarity tiebreak | 10,713 | 7.0% |
| **playability tiebreak** | **1,472** | **60.2%** |

Skullclamp went from recommending *Playable Delusionary Hydra* to *Yawgmoth, Thran Physician*.
`tests/test_synergy.py:test_synergy_partners_are_playable` is the gate.

Playability is `1 - log1p(rank)/log1p(EDHREC_RANK_SCALE)`, clipped strictly below 1 so it can
never outrank a full score step — a 2-rule match always beats a 1-rule match. Unranked cards
score 0. It is popularity, not quality, and will bias toward staples.

**Known limit:** a card whose top tier is genuinely small cannot be rescued by re-ranking.
Skullclamp's holds 3 cards, so its answer barely moves — that is coarseness in the 24 rules,
not in the ordering.

## Power creep (`src/manamap/analysis/power_creep.py`)

Writes `data/obsolescence_index.json`: for each anchor card A, up to
`OBSOLESCENCE_MAX_REPLACEMENTS` (5) cards that do the same job, each with a
**`strength` from 0.0 to 1.0**. It is a comparison, not a verdict — the schema key
is `compare_with`, and the pilot decides what a difference is worth.

**Retrieval — which pairs are considered at all.** Both cards must share a
supertype (Land and Unknown are never compared), clear a cosine gate in the
*ability* embedding space — **tiered**: 1-tag cards need >= 0.98
(`OBSOLESCENCE_SINGLE_TAG_THRESHOLD`), 2+-tag cards >= 0.75
(`OBSOLESCENCE_SIMILARITY_THRESHOLD`), which keeps Doom Blade → Fatal Push
(sim 0.999) while filtering false positives — and B must have all of A's
mechanical tags, cost no more (`effective_cost`, which discounts Phyrexian
pips), have a same-or-easier colour requirement, same-or-better power/toughness,
and be printed later than A.

**`newer` is a gate; rank is not.** Printed-later is what makes the relation
antisymmetric — drop it and every pair appears twice. It predicts "actually
played more" only 67.7% of the time, so it is weak evidence of strength and is
never presented as any. EDHREC rank is *reported* and never filtered on:
Storm Crow is genuinely outclassed by a card nobody plays.

**Judgement — two hard gates, everything else priced.**

1. **Legality** (`legal_commander`, combined so any legal printing wins). An
   illegal card is not a candidate in any degree.
2. **Nothing to compare on.** Tags are valenced through `config.TAG_VALENCE`
   into `gain` / `cost` / `context`; a pair whose only claimed advantage was a
   COST (`discard`, `sacrifice`, `mill`) has no gains left and is dropped.
   `DEFAULT_TAG_VALENCE = "context"`, so an unvalenced tag is a *difference* and
   never an advantage.

Everything else multiplies into `strength` via `OBSOLESCENCE_PENALTIES` —
tribal gate 0.30, added restriction 0.55, ability costs more 0.45, a cost tag
0.75, played less 0.90 — starting from a base that rises with how much cheaper
B is and how many distinct gains it has. Multiplicative, so two problems
compound rather than averaging out.

**The entry is symmetric**: `{strength, gains, costs, narrows, edhrec_rank,
played_more, released_at, name, similarity}` — what the card costs you as well
as what it gains.

Exclusions: cards with < `OBSOLESCENCE_MIN_TAGS` (1) tags, empty/NaN mana cost
(augments, tokens), modifier stats (`+2`) parse to None. Batch matrix multiply
per supertype group for performance; every oracle-text scan is precomputed into
the per-card record, because the comparison itself is O(n²).

**`manamap eval-obsolescence` scores the index against how it is known to be
wrong** — the failure classes, the strength histogram, the separation between
detectably-bad and clean pairs, and the retrieval check. The full audit that
produced it, and the four fixture failures, are in `docs/gotchas-analysis.md`.

## Region clustering (`src/manamap/analysis/cluster_regions.py`)

HDBSCAN names geographic-style regions on both 2D maps at two zoom levels:
- **L0 mega-regions**: `min_cluster_size=800`, `min_samples=50` → ~10–15 regions (zoomed out)
- **L1 sub-regions**: `min_cluster_size=100`, `min_samples=15` → ~40–120 regions (zoomed in)

Naming — Color+Type map: dominant color (>= 40%) + type (>= 30%), guild names for 2-color pairs (>= 50%), fallback to top tag. Abilities map: TF-IDF-like scoring (cluster tag freq / global freq), top 1–2 overrepresented tags, minimum presence threshold `REGION_MIN_TAG_PRESENCE=0.10`. Label dedup: tag suffixes (max 2 for L0, 3 for L1), then spatial direction (N/S/E/W).

**L0 and L1 are two independent flat clusterings, not a tree.** L1 is a separate HDBSCAN run over the same 2D coordinates, not a subdivision of L0, and `parent` is assigned by nearest L0 centroid with **no containment test** — an L1 region can be parented to an L0 region it does not overlap. Any UI that presents them as a hierarchy is claiming more than the data supports.

**Membership is stored, and noise is a real answer.** `regions_*.json` carries `membership.l0` / `membership.l1`: positional arrays over `cards.csv` row order, `-1` for noise. 29% of cards on the default map are L0 noise and belong to no region at all — they stay `-1` rather than being snapped to a nearest centroid they were never clustered into. Regions also record `w`/`h` beside `span`, because `span = max(w, h)` alone cannot distinguish a filament from a blob.

**Index-alignment invariant**: `projection[i]` corresponds exactly to `cards.csv[i]` (maintained through embed → reduce), and `membership.l0[i]` describes the same card. Tags are looked up by direct index, never by name (duplicate card names exist).


## The masked-imputation VAE — first run, 2026-08-31

A shadow experiment replacing a contrastive objective whose positives are mined
from the repo's own regexes with **masked imputation**: hide a block of the card,
reconstruct it. No labels — the label is the input.

**Configuration**: frozen MiniLM (`unfreeze=0`, 0.94M trainable against 2.2M
corpus tokens), 20 epochs, ~4.2 min/epoch on a quiet machine, split by TEXT HASH
(1,031 duplicate-text families, 0 crossing). Val loss 0.0134 → 0.0121.

### Scored against the space it would replace

| target | function | VAE | |
|---|---|---|---|
| function @ pool 500 | 0.794 | 0.483 | **−0.311**, excludes 0 |
| theme @ pool 500 | 0.152 | 0.326 | **+0.174**, excludes 0 |
| hard negatives (mean 1−cos) | 0.0133 | **0.0064** | worse, halved |
| centroid headroom | 0.019 | **0.092** | 4.8×, and beats text's 0.075 |
| effective dimensionality | 27.31 | **5.71** | far worse |

**Two of four, and the two it won are the two the contrastive objective fails by
design.** The theme win is the hypothesis working exactly as argued: a model
trained to impute the type line must carry tribe. Intervals exclude zero at every
pool size.

**The hard-negative loss is the design risk landing where it was flagged.** §6.3
names "cards with similar phrasing that do meaningfully different things" as the
failure mode, and the plan said a reconstruction objective was the version most
likely to amplify it. It did: rewarding a model for predicting wording stacks
`Seachrome Coast` and `Deserted Beach` — one word inverts the game stage —
closer together.

### Two instruments that misled, and are now fixed

**`FREE_BITS = 0.5` disabled the regulariser.** The run settled at 0.19–0.31
nats/dim, entirely under the floor, so `clamp(per_dim − 0.5, min=0)` was exactly
zero for all 20 epochs. Beta annealed 0 → 0.25 against a structurally dead term.
**What trained was a denoising autoencoder, not a VAE.** A floor above the KL a
model reaches is not a floor, it is an off switch. Now 0.1.

**`active_units` reported 128/128 "no collapse" on a space with a participation
ratio of 5.71** — barely above the layout space's 3.89, which is deliberately
trivial. It cannot catch that: with free bits disengaged nothing pressures the
latent, so the alarm is guaranteed silent. `train_vae` now gates on
**effective dimensionality** as well, floor 10.0.

### What it says about the next run

The objective moves what it was designed to move, and the two spaces now have
**measurably complementary failures** — function/hard-negatives against
theme/centroid-geometry. That argues for combining the objectives rather than
substituting one for the other, with the KL actually engaged.


## The VAE sweep, 2026-08-31 — and why the objective, not the hyperparameters, is the problem

Caching the frozen encoder's output turned an 85-minute run into **66 seconds**
(`manamap vae-cache`, 268 MB, one 13-minute build). Seven configurations, twelve
epochs each, in the time one epoch used to take.

| config | val | KL/dim | active | effdim | effdim trajectory |
|---|---|---|---|---|---|
| no-KL (autoencoder) | 0.0131 | 0.230 | 128 | 2.63 | 21.9 → 6.3 → 3.6 → 2.9 |
| fb0.5 β0.25 (run 1) | 0.0131 | 0.229 | 128 | 2.63 | identical to no-KL |
| fb0.1 β0.25 (run 2) | 0.0131 | 0.049 | 128 | 8.37 | 22.8 → 15.7 → 11.8 → 10.0 |
| fb0.1 β0.01 (run 3) | 0.0131 | 0.071 | 128 | 5.36 | 22.5 → 10.2 → 7.0 → 5.8 |
| no floor, β0.001 | 0.0131 | 0.002 | 0 | 20.50 | 22.2 → 15.5 → 18.7 → 18.5 |
| no floor, β0.01 | 0.0131 | 0.000 | 0 | 64.29 | 23.3 → 49.9 → 63.4 → 63.9 |
| no floor, β0.05 | 0.0131 | 0.000 | 0 | **65.84** | 21.8 → 67.3 → 67.5 → 67.1 |

### Every number in the `effdim` column is a trap except by accident

`no floor, β0.05` has the best effective dimensionality in the table — better
than the frozen text baseline's 51.39 — and the **worst** latent. Measured raw
posterior-mean norms:

    fb0.1 β0.01        ‖mu‖ = 4.01     real signal
    no floor β0.05     ‖mu‖ = 0.129    ≈ zero

μ→0 collapses the posterior onto the prior, and L2-normalising near-zero vectors
amplifies floating-point noise into something isotropic. **A high participation
ratio over noise is still noise.**

### Three instruments, three different lies

| instrument | how it failed |
|---|---|
| `active_units` | reported 128/128 on run 1's degenerate space — it cannot fire when free bits disengage |
| `effective_dim` | **highest for the emptiest latent**, as above |
| validation loss | **0.0131 for all seven configs**, to four decimals |

### Which is the actual finding

Validation loss is identical across configurations whose latents differ by a
factor of **31 in norm**. The objective cannot distinguish them at all.

The bag-of-tokens target is **9.5 positive columns out of 2,048 — a 0.47%
positive rate** — so the BCE is dominated by easy negatives. Predicting the base
rate everywhere scores 0.0296; every configuration reaches 0.0131 and stops. The
objective *is* learning something (a 56% reduction), but it has no gradient left
to say which latent is better.

**So three runs and a sweep of hyperparameters were tuning a loss that does not
rank them.** The next change is to the objective — weighted positives, a ranking
loss over tokens, or a denser target — not to β.

### And the geometry decays

Every configuration starts near effdim 22 and falls. The `5.71` from the first
run was not a property of the objective; it was where a monotonic decay had
reached by epoch 20. Reconstruction quality and latent spread move in opposite
directions here, which is the thing worth attacking.

### A Python bug worth naming, caught by a control

The first sweep produced **byte-identical results for all seven configs**.
`kl_with_free_bits(mu, logvar, free_bits=FREE_BITS)` binds its default ONCE at
definition time, so setting `model_vae.FREE_BITS` between configs changed the
module attribute and nothing else. Both knobs now take `None` and resolve inside.

It was caught only because the sweep re-ran the three slow configurations as
controls. Without them the three novel configs would have read as "these knobs do
not matter" — wrong, and entirely plausible.


## The finding: validation loss and space quality are ANTI-CORRELATED

Scored on the real eval (pool 500, test split), everything else held fixed, with
only the epoch count varying:

| epochs | val loss | function | theme | hard_neg | effdim |
|---|---|---|---|---|---|
| **1** | 0.0215 | **0.618** | 0.387 | 0.0026 | **34.19** |
| 2 | 0.0151 | 0.604 | 0.384 | 0.0024 | 32.97 |
| 3 | 0.0142 | 0.599 | 0.382 | 0.0024 | 31.39 |
| 5 | 0.0134 | 0.585 | 0.387 | 0.0024 | 28.31 |
| 8 | 0.0130 | 0.588 | 0.387 | 0.0027 | 24.86 |
| 12 | 0.0131 | 0.567 | 0.381 | 0.0031 | 20.64 |

Validation loss falls monotonically (0.0215 → 0.0130) while function recall falls
monotonically with it (0.618 → 0.567) and effective dimensionality collapses
(34.19 → 20.64).

**`train_vae` early-stops on validation loss, so it was selecting the worst space
it produced.** One epoch — eight seconds — beats the 85-minute twenty-epoch run
on every axis: function 0.618 against 0.483, theme 0.387 against 0.326, effdim
34.19 against 5.71.

That also explains the first run's poor showing. It was not a bad configuration;
it was twenty epochs deep into a degradation that starts immediately.

### What the best VAE configuration actually achieves

| | function | theme | hard_neg | effdim |
|---|---|---|---|---|
| function space (incumbent) | **0.794** | 0.152 | **0.0133** | 27.31 |
| text baseline | 0.629 | **0.523** | 0.0197 | **51.39** |
| **VAE, 1 epoch** | 0.618 | 0.387 | 0.0026 | 34.19 |

It **loses function** (0.618 vs 0.794) and **loses hard negatives** (0.0026 vs
0.0133 — the reconstruction objective stacking similar phrasings, as predicted).
It **beats the incumbent on theme** (0.387 vs 0.152) and on effective
dimensionality (34.19 vs 27.31).

So: not a replacement, and the trade is now measured at three points instead of
argued about. `pos_weight` was tested and rejected — it buys hard-negative
separation (0.0031 → 0.0110) and pays for it on both relations.

### The methodological lesson

Every training-time signal in this work has now been shown to mislead:
`active_units` (silent when free bits disengage), `effective_dim` (highest for an
empty latent), validation loss (identical across configs, and anti-correlated
with quality once it does move). **A sweep must score the downstream task.** The
cached encoder makes that affordable — 8 to 73 seconds per configuration, eval
included — which is the only reason any of this was findable in an afternoon.
