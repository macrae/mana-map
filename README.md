# Mana Map

[![tests](https://github.com/macrae/mana-map/actions/workflows/test.yml/badge.svg)](https://github.com/macrae/mana-map/actions/workflows/test.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![the map](https://img.shields.io/badge/live-the%20card%20map-8b5cf6)](https://macrae.github.io/mana-map/viz/index.html)
[![the newsstand](https://img.shields.io/badge/live-nine%20issues-e11d48)](https://macrae.github.io/mana-map/manuals/index.html)

Two things live here, sharing one data layer and one CLI:

**A card discovery tool** — every Magic: The Gathering oracle card (~34,900) embedded by
two small neural nets. It opens on **one card**: hover it, click a relation, and its
neighbours join a force-directed graph you grow by clicking. Load one of your own decks and
it lights up with its commander ringed, so you can see where it sits in card space and walk
outward from it. The 34,890-point atlas is still there, one click away — and it drifts
slowly at altitude, settling to a stop as you zoom in to read.

Three relations, each precomputed so a click is instant: **similar** (embedding neighbours),
**synergy** (rule-based complements, each edge labelled with the rule), and **outclassed by**
(strictly-better replacements). Boot costs 1.8 MB.

**Pilot's Manual** — a magazine generator. Point it at one Commander deck and it produces
a self-contained web issue: combo lines whose every step cites the Comprehensive Rules and
survived an adversarial checker, resource curves from a seeded simulation, and coaching
that says out loud when it's coaching.

An issue opens on an editor's letter and then a **panel** — three columnists arguing about
how to fly the deck, in the vocabulary of its engine model, where a connection the model
draws *dashed* is one the panel may discuss and may not assert. Then two pictures of it. The **constellation** re-lays-out its cards from
the embeddings and clusters them into named cities — what shape it is. The **engine flow**
shows how it runs, stage by stage, with each connection drawn solid when a rules-verified
line proves it and dashed when it is the analyst's reading. Those are different relations: a
card clusters by what it *says*, and an engine is what cards *do to each other*.

The fastest way to understand the second one is to read an issue:
**Nine issues live** — [001 Goblin Storm](https://macrae.github.io/mana-map/manuals/goblin-storm.html) · [002 Hapatra](https://macrae.github.io/mana-map/manuals/hapatra.html) · [003 Sisay](https://macrae.github.io/mana-map/manuals/sisay.html) · [004 Heliod](https://macrae.github.io/mana-map/manuals/heliod.html) · [005 Ur-Dragon](https://macrae.github.io/mana-map/manuals/ur-dragon.html) · [006 Edgar](https://macrae.github.io/mana-map/manuals/edgar-vampires.html) · [007 Gishath](https://macrae.github.io/mana-map/manuals/gishath.html) · [008 Yawgmoth](https://macrae.github.io/mana-map/manuals/yawgmoth-swarm.html) · [009 Radagast](https://macrae.github.io/mana-map/manuals/radagast.html)
· [the newsstand](https://macrae.github.io/mana-map/manuals/index.html)

Vol. 001 was hand-built. **Vol. 002 was not** — the deck was generated from a three-line
brief naming a commander and a power bracket, and the issue's headline finding is that the
deck's own combo count was wrong.

---

# For users

## Explore the card map

**Two commands. No install, no pipeline, no API keys.** Every data file the frontend
needs is committed.

```bash
git clone git@github.com:macrae/mana-map.git && cd mana-map
python3 -m http.server 8000
```

Open <http://localhost:8000/viz/index.html>.

*Contributing? [CONTRIBUTING.md](CONTRIBUTING.md) is two commands and a list of landmines.
[docs/README.md](docs/README.md) sorts the documentation into current reference and
historical design records, which saves reading 12,000 lines of the latter by accident.*

Three things to know:

- **Serve from the repo root.** The page fetches `../data/*`, so `viz/` and `data/` must
  stay top-level siblings. Opening `viz/index.html` as a `file://` URL fails on CORS.
- The clone carries **149 MB of tracked data** (and `git clone` transfers rather more
  than that), but **discovery boots on 1.8 MB** — a slim
  card index plus a precomputed neighbour table. The heavy artifacts load only if you ask
  for what needs them: the 2.9 MB projection when you open the atlas, the 16.8 MB embedding
  matrix never on the discovery path at all. That the *clone* is large is deliberate — see
  [Landmines](#landmines). (Two of the tracked files,
  `combo_details.json` and `card_roles.json`, are for the deck builder and the agents; the
  browser never fetches them.)
- d3 loads from a CDN, so the page needs internet even though the data doesn't.

What you get: two maps (one clustered by color and type, one by what cards *do*), three
relations on every card — **similar** via embedding neighbours, **synergy** via rule-based
complementarity, **outclassed by** via the obsolescence index (these are different
algorithms, see `docs/architecture.md`) — a **Build** mode that lights up a deck or a pool inside the 34K atlas, and obsolescence
badges for cards with strictly-better replacements.

Build shows a deck's footprint in card space — its role histogram, mana curve segmented by
the current overlay, colour load and verified lines — and hands work back out as a
`brief.json` the deterministic builder reads. It does **not** score cards: evaluation comes
from the pipeline through the agent loop, so there is exactly one scorer.

The atlas is a launchpad: clicking a relation there carries you into the walk seeded on that
card, rather than doing something subtly different because you happened to be in a different
mode.

## Generate a manual for your own deck

This half is more involved, and honest about why: **the manual pipeline needs
[Claude Code](https://claude.com/claude-code).** The Python in this repo makes *zero* LLM
calls — it is deterministic infrastructure (fetching, simulating, validating, rendering)
that AI agents drive from the outside. A full generation runs ~700k tokens across six
serially-dependent routines, nearly 40% of it the engine loop; an invocation cache is what
makes iterating on it affordable. `docs/agent-cost.md` has the breakdown.

### 1. Environment

```bash
make setup
```

That is the whole step. It checks for **Python 3.10 exactly** — PyTorch publishes no wheels
for 3.13+, and the pins (`sentence-transformers<4`, `numpy<2`) target torch 2.2.2 — then
installs in the one order that works and downloads chromium for the browser tests.

If your 3.10 is not called `python3.10` on `PATH` (a conda or pyenv build, say):

```bash
make setup PYTHON310=$(pyenv which python3.10)
```

By hand it is three commands, and the **order is not cosmetic**: pacmap pulls numba, and
installing it afterwards triggers a source build of LLVM.

```bash
python3.10 -m venv .venv
.venv/bin/pip install llvmlite==0.41.1 numba==0.58.1   # must come first
.venv/bin/pip install -e ".[dev]"
```

Developed on macOS/arm64. Linux should work apart from MPS device selection; Windows is
untested.

### 2. One-time databases

```bash
.venv/bin/manamap pilot download-rules      # the Comprehensive Rules text
.venv/bin/manamap pilot build-rules-db      # ~3.9K chunks, chunk ID = rule number
.venv/bin/manamap pilot build-strategy-db   # the strategy companion's RAG index
```

All three outputs are gitignored and must be built locally. `CR_RULES_URL` in `config.py` is pinned to
a specific rules release — update it when Wizards ships a new one.

### 3. Your deck

```bash
mkdir -p data/decks/<slug>/{stacks,decisions}
# write data/decks/<slug>/decklist.txt
.venv/bin/manamap pilot fetch-deck <slug>
.venv/bin/manamap pilot validate-deck <slug>
```

**Use a Moxfield export.** Lines like `1 Zada, Hedron Grinder (SLD) 2406 *F*` carry the
set, collector number and foil marker, and `fetch-deck` resolves those *first* — so the
manual shows the actual cards in your deck, with the right artist and the right art. A
name-only list works but yields default reprints and a visibly weaker issue.

`fetch-deck` short-circuits when the decklist hasn't changed; use `--force` after oracle
errata.

### 4. The files you write by hand

Nothing scaffolds these. `data/decks/goblin-storm/` is the worked reference.

| File | Why it's manual |
|---|---|
| `decklist.txt` | It's your deck — **unless you let the builder write it**, see below |
| `issue.json` | Volume, date, cover price, next issue. **The build hard-exits without it** — a *generated* date would break byte-identical rebuilds |
| `goldfish_targets.json` | Which key-piece sets are worth simulating is a judgment call — though `/build-deck` derives it from the plan's declared engines |

### 4b. Or don't write a decklist at all

If you have a commander in mind rather than a list, the builder makes one:

```bash
# author data/decks/<slug>/brief.json: commander, bracket (1-5), playstyle
.venv/bin/manamap pilot build-deck <slug> --write-decklist   # deterministic, no agents
.venv/bin/manamap pilot fetch-deck <slug>
```

That alone produces a legal, tier-conditioned, goldfishable 99. Running the `/build-deck`
skill on top adds the agent loop, which is what makes it *good* rather than merely legal —
and the whole path from brief to published issue is proven: see Vol. 002.

### 5. The agent phases

Run these from Claude Code in the repo root. Each is a skill in `.claude/skills/`.
**`/publish-deck` sequences all of them** — start there, and run
`manamap pilot deck-status <slug>` at any point to see what a deck still needs.

| Step | Produces | Pure CLI? |
|---|---|---|
| `/build-deck` | A 99 from a brief: pool → architect ⇄ critic, bracket-gated | no |
| `manamap pilot bracket-check <slug>` | Computed bracket floor + its evidence | **yes** |
| `/resolve-stack` | A verified combo line: resolver → validator → adversarial checker | no |
| `manamap pilot goldfish <slug>` | Resource curves from 10k seeded games | **yes** |
| `manamap pilot deck-map <slug>` | The constellation: local layout + clusters | **yes** |
| `/analyze-engine` | The engine: stages, lines, what a stack actually proves | no |
| `/write-manual` | Strategic frame, coaching, body prose | no |
| `manamap pilot validate-issue <slug>` | Form gate over the legacy plan (published decks) | **yes** |

### 6. Build and read

```bash
.venv/bin/manamap pilot build-manual <slug>
.venv/bin/manamap pilot build-index          # run after build-manual
open manuals/<slug>.html
```

Rendering is free, deterministic and repeatable — the same artifacts always produce the
same bytes.

---

# For developers

## The shape

Two pipelines, same pattern: each stage writes artifacts the next one reads, and every
stage is independently runnable and testable.

```
Card map   download → extract → preprocess → train ×2 → embed → reduce
                    → combos → export → synergy → power-creep → regions → card-roles → viz

Build      brief.json → build-deck → bracket-check → architect ⇄ critic → decklist

Manual     fetch-deck → goldfish + RAG DBs → agents author JSON
                    → validators gate → renderer builds → GitHub Pages
```

`manamap run` drives the first (15 steps, ~40–60 min, internet at two of them).
`manamap pilot <cmd>` drives the second (52 pilot subcommands). All constants live in
`src/manamap/config.py`; both CLIs are registry-driven with lazy imports.

Two lightweight fusion MLPs (~180K params each) produce the 128-dim embeddings; the text
encoder stays frozen. They answer different questions and are not interchangeable. The
**layout** model organises the map by colour and type and feeds the projection only. The
**function** model answers whether two cards do the same job, and is the sole source of
similarity — the *similar* relation, the walk and drill all read it whichever map is on screen.

That split exists because the alternative was measured and was bad: when similarity followed
the displayed map, the colour/type space was using 3.9 of its 128 dimensions and *Doubling
Season*'s nearest neighbours came back as arbitrary green enchantments. `manamap
eval-embeddings` (step 15) scores every space against a hand-authored golden set so a claim
like that is a number rather than an opinion.

## The embedding models

*A technical overview. Code: `src/manamap/training/{model,train,train_ability}.py`,
`src/manamap/ingest/{extract,preprocess}.py`, `src/manamap/analysis/eval_embeddings.py`.
Every constant named below lives in `config.py`.*

### The problem

34,890 Magic cards, each a short piece of natural language plus a dozen structured
attributes, into a metric space where "these two cards do the same job" is a nearest-neighbour
query. There is no click-stream and no relevance judgements — supervision has to be
manufactured from the cards themselves, which is most of what makes this interesting.

### How a card is decomposed

One card becomes nine parallel inputs. Nothing is learned end-to-end from raw text; the
sentence encoder is frozen and everything else is a small learned table over an explicit
feature.

| block | shape | how it is built |
|---|---|---|
| frozen text | 384 | `all-MiniLM-L6-v2` over a synthesised sentence (below) |
| supertype | 1 → 16 | `nn.Embedding(10, 16)` — Creature, Land, Instant… |
| rarity | 1 → 8 | `nn.Embedding(7, 8)` |
| colour identity | 1 → 32 / 8 | `nn.Embedding(33, ·)` over the 32 observed WUBRG subsets |
| layout | 1 → 16 | `nn.Embedding(18, 16)` — normal, split, transform, adventure… |
| continuous | 2 | normalised CMC; normalised EDHREC rank |
| keywords | 50 | multi-hot over the 50 most frequent keywords |
| mechanical tags | 33 | multi-hot, regex over oracle text (function model only) |
| structured | 15 | power/toughness (3) + mana pips (6) + colour features (6) |

Two details that are load-bearing rather than incidental:

**The sentence is synthesised, and the card's name is deliberately excluded.**
`build_embedding_text` emits `"{type_line}. Cost {mana_cost}. {P}/{T}. {oracle_text}.
Keywords: {…}"`. The name used to lead the string and was buying similarity off shared
tokens rather than shared function — *Rhystic Study* matched *White Rhystic Study* at 0.951,
*Sol Ring* matched *Sisay's Ring*. A name is also a large fraction of a short card: *Sol
Ring*'s entire text is eleven words, three of them the name. Dropping it moved held-out
recall@10 from 0.187 to 0.248 and median rank from 159 to 129.

**Continuous features use fixed scales, never per-run min-max.** EDHREC rank divides by a
hardcoded 50,000, power/toughness by 15. A per-run normalisation makes the same card's
features differ between pipeline runs, which silently destroys comparability between two
runs' embeddings — this was a real bug.

### Architecture

A fusion MLP. Categorical blocks go through embedding tables, the two high-cardinality
multi-hot blocks (keywords, mechanical tags) through `Linear + ReLU`, and everything is
concatenated into one wide vector fed to a three-layer trunk:

```
concat[…] → Linear(d_in, 256) → ReLU → Dropout(0.1)
          → Linear(256, 128)  → ReLU → Dropout(0.1)
          → Linear(128, d_out)
```

**181,272 trainable parameters** for the layout model, **192,672** for the function model.
That is small on purpose: the frozen 384-dim MiniLM output is the only component that needed
scale, and it is amortised across every run.

### The output split — the one piece of real design

The layout model returns `F.normalize(x)`, 128 dims, and that is the whole story.

The function model does something else. Its trunk emits 96 dims and a **separate
`Linear(384, 32)` with no ReLU** projects the frozen text into the remaining 32. Each half
is L2-normalised independently, then scaled by `√(1−W)` and `√W` with `W = 0.3`:

```python
learned = F.normalize(x) * sqrt(1 - W)
text    = F.normalize(text_proj(text_emb)) * sqrt(W)
return torch.cat([learned, text], dim=1)      # already unit-norm
```

Because the squared weights sum to 1, the concatenation is unit-norm and the dot product of
two cards is **exactly**

```
sim(a, b) = 0.7 · cos_learned(a, b) + 0.3 · cos_text(a, b)
```

This exists because of a measured failure. The previous function model scored **0.093
recall@10 against 0.187 for the frozen text it was built from** — training was subtractive,
and the model had quietly learned to discard the only signal that was working. The split
makes discarding it *structurally impossible*: the text's contribution is a fixed fraction
set by architecture, not a fraction the optimiser is free to drive to zero. The rectifier is
omitted from `text_proj` for the same reason — this half exists to preserve the text
geometry, and a ReLU folds half of it away.

### Objectives

**Layout model — `TripletMarginLoss`, margin 0.3.** Positives are drawn from the same
`(supertype, primary_colour)` group with two fallbacks; negatives must differ on *both*.
This task is nearly trivial, which is the point — its only job is to give PaCMAP something
with legible colour/type structure to project. Its effective dimensionality of 3.9/128 is
not a defect for that job; it *is* the job.

**Function model — symmetric in-batch InfoNCE, τ = 0.05.** The dataset yields
`(anchor, positive)` pairs only; the batch supplies negatives.

```python
scores = (anchors @ positives.T) / temperature
labels = torch.arange(len(scores))
loss   = 0.5 * (cross_entropy(scores, labels) + cross_entropy(scores.T, labels))
```

At batch 256 that is **255 negatives per anchor for one forward pass**, against the old
triplet loss's single mined negative. The replacement was diagnostic, not fashionable: a
margin loss stops producing gradient the moment it is satisfied, which for a task this easy
was around epoch 3, so nothing pressured the model to preserve structure *within* a class.

### Positive mining

With no labels, the positive-selection rule is effectively the loss function. Three tiers,
per anchor:

1. **Rarest specific role first.** A 53-role taxonomy (`ROLE_PATTERNS`) covers 72.6% of
   cards at 1.62 specific roles each. Roles are sorted by group size ascending — two cards
   sharing `doubler:tokens` (11 cards) say far more about each other than two sharing
   `value:etb` (5,580), so the positive is spent on the anchor's most specific claim.
2. **≥2 shared mechanical tags** (the old rule, now a fallback). It covered only 46.9% of
   the corpus, so for most cards it *was* the random tier wearing a better name.
3. **Random.**

`ROLE_BODY_FALLBACK` is excluded deliberately: it labels all 19,050 creatures, so "shares
this role" would be barely narrower than "is a creature" and would rebuild the trivial task
this design exists to escape. Mining is scoped to each split's own indices — a validation
positive drawn from the training set is leakage.

**No hard-negative mining, stated as a non-change.** Random in-batch negatives are safe
here: measured on this corpus, a random pair at batch 256 has a 0.004% chance of being a
true near-duplicate, about 0.01 false negatives per anchor. That would *not* survive hard
mining, which selects nearest-non-positives by construction while 39% of cards have a text
neighbour above 0.75 — it needs a similarity ceiling, and that is a second variable.

Optimiser: Adam, lr 1e-3, batch 256, 10% validation, early stopping on patience 5, seed 42.

### Similarity ranking, and how it is served

Embeddings are L2-normalised at build time, so cosine is a plain dot product and top-k is
`argpartition` over one matrix-vector product — no index structure, no approximation, 34,890
rows is small.

Serving is the constrained part, because the browser must branch **synchronously** mid-gesture
and cannot download a 16.8 MB float matrix. `neighbours.bin` precomputes, per card, 12
similar + 10 synergy + 5 obsoleted-by row ids: `uint16` ids, similarities quantised to
`uint8` — **2.4 MB gzipped for the whole discovery boot, against 18.4 MB before.**

The quantised value is used for edge length only, and **ordering is array order.** Re-sorting
client-side by the lossy value changes the top-10 for roughly two thirds of cards, because
the space is a narrow cone — median pairwise cosine 0.714, so 8 bits over the observed range
is coarser than the gaps being ranked. It would read as a model regression rather than a
precision bug. The header carries a SHA-256 of the embeddings it was built from and a test
fails if they diverge, because a stale table parses fine and answers confidently.

The 2D map is a separate artifact: PaCMAP, `n_components=2, random_state=42`, over the
**layout** embeddings only. Nothing reads the projection for similarity.

### Evaluation

`manamap eval-embeddings` scores every space against `data/eval/similarity_golden.json` — 40
hand-authored groups, 12 dev / 28 test. It must **stay** hand-authored: training mines its
positives from roles and tags, so an eval derived from those would only measure whether
training memorised its own supervision.

Three metrics, deliberately including two geometric ones, because recall alone hid the
collapse for months:

- **recall@k / median rank** against the golden groups.
- **effective dimensionality** — participation ratio of the PCA spectrum, `(Σλ)²/Σλ²`. Reads
  as "how many of the nominal dimensions are actually in use"; equals *d* for an isotropic
  *d*-dimensional cloud and 1 for a line.
- **neighbour spread** — mean cosine gap between the 1st and 50th neighbour. Near zero means
  the top neighbours are indistinguishable and whichever one is returned first is an artefact
  of float ordering.

Current shipped artifacts, test split:

| space | dim | eff. dim | spread | r@10 | r@50 | med. rank |
|---|---:|---:|---:|---:|---:|---:|
| layout (colour+type) | 128 | 3.89 | 0.0061 | 0.086 | 0.139 | 1148 |
| function (ability) | 128 | 27.31 | 0.0323 | 0.232 | 0.464 | **76** |
| text baseline (frozen MiniLM) | 384 | 51.39 | 0.1341 | **0.244** | 0.414 | 126 |

**Read that table honestly.** Against the previous function model — 5.97 effective dims,
0.093 recall@10, median rank 995 — the rebuild is a large win, and the current space is
clearly better at depth: r@50 0.464 vs 0.414, median rank 76 vs 126. But it is still **0.012
behind the frozen text at r@10**, and the eval prints a warning saying so on every run. It
is not fixed. `tests/test_embedding_quality.py` also holds a deliberately still-failing
`xfail(strict=True)` gate on neighbour spread — 0.0323 against a 0.05 target — whose
threshold was not lowered to match the result.

Two standing rules around this harness:

- **Do not tune hyperparameters on it.** Sweeping the text weight looked like a win (0.258
  r@10) until selecting on `dev` picked a different value and the two splits disagreed. At
  this sample size those differences are noise.
- **Quote the test split.** `dev` was consumed while diagnosing.

## Extension points

| To add… | Touch |
|---|---|
| A pipeline step | `STEPS` in `pipeline.py`; the module exposes `main()` |
| A pilot command | `PILOT_STEPS` + `_DECK_COMMANDS` + argparse in `pilot/registry.py`; module exposes `main(args)` |
| A magazine department | `DEPARTMENTS` in `pilot/issue_spec.py` — changes every issue; treat it like `config.py`. Add to `OPTIONAL_DEPARTMENTS` to pilot it on one deck first, and remove it once every deck carries it |
| A columnist, or a voice rule | `MASTHEAD_COLUMNISTS` + `_VOICE_BANS` in `pilot/validate_issue.py`. Measure a proposed ban against all nine decks before keeping it |
| A deck lifecycle phase | `STAGES` in `pilot/deck_status.py`, or the next person will not find it |
| A data file the viz reads | The `DATA` map in `viz/js/mana-map.js`, plus a `.gitignore` negation |
| A synergy rule, tag, or threshold | `config.py`, nowhere else |
| A deckbuilding role | `ROLE_PATTERNS` in `config.py`, then re-run `manamap card-roles` |
| A bracket rule | `BRACKETS` / `COMBO_BRACKET_TAGS` / `MASS_LAND_DENIAL` in `config.py` |
| Builder tuning | `DECK_ROLE_BUDGET` / `DECK_BUILD_WEIGHTS` in `config.py` |
| An agent cache routine | `AGENT_ROUTINES` in `config.py` |

**One thing that is someone's, not the project's.** `data/collection/*.txt` lists a
physical card collection — `deck-history` reads it to tell a swap you already own from
one you would have to buy, and it is the only ownership question left in the repo. The
tracked files are the maintainer's, kept as a worked example. Point
`MANAMAP_COLLECTION_DIR` at your own boxes, or at nothing: an absent directory means no
ownership claim rather than an error.

## Three contracts worth understanding

**Evidence tiers.** Every section of a manual wears a badge: ✓ rules-verified,
◆ data-derived, ★ coaching. `validate-issue` enforces that a department cannot claim a
tier the system didn't grant it — costume never earns the badge.

**Validation gates — form in code, meaning by agent, publication by verdict.** A mechanical
validator checks structure: does every step cite a rule, does that rule exist, is the quote
a verbatim substring. A *separate adversarial agent* then fetches each full rule and judges
whether it actually supports the claim. Only `pass` renders. Failed artifacts are kept —
they document open questions.

**Determinism.** Agents return JSON and never write HTML. That keeps the renderer a pure
function of committed artifacts, byte-identical on rebuild and enforced by tests — and it
is *why* the issue date is authored, why goldfish is seeded, and why image URLs get their
cache-busters stripped.

## Testing

```bash
make test          # the inner loop: non-browser, parallel, cached   ~22 s
make test-fresh    # same, nothing served from the cache             ~29 s
make test-browser  # the playwright suite                            ~4 min
```

A bare `pytest` is `make test`. Four test files recompute an artifact and compare it to
the tracked copy — 90,000 seeded goldfish simulations among them — so those are cached on
a hash of their inputs *and* of the code that produces them, recorded only on a pass, and
kept in gitignored `.pytest_cache/` where they cannot reach another machine or CI. The run
prints how many it skipped. `make test-fresh` is the one to trust before a PR.

A fresh clone runs **1,360 of the 1,488 fast tests** green in 20 s; the 129 it skips
gate on artifacts that are gitignored and built locally, and each says which command
would enable it.

Counts and the per-file inventory live in `docs/testing.md` — they move on almost every
commit, so restating them here would be one more thing to drift. Five skip markers in `tests/conftest.py` gate on the last
artifact of each stage, so **skips on a fresh clone are expected and correct**. Unit tests
build inline fixtures — no fixture files. Paths always come from `manamap.config`, so the
suite is CWD-independent and honours `MANAMAP_DATA_DIR`.

## Landmines

- **`python -m manamap.pipeline` starts the full 40–60 minute run** with no arguments and
  no confirmation, overwriting trained models. Use the `manamap` CLI.
- **Never put `data/` on Git LFS.** GitHub Pages serves LFS pointer files, not content —
  it would silently break every fetch on the deployed site. The 149 MB of tracked data
  is deliberate.
- **Index alignment**: `projection[i]` ≡ `cards.csv[i]` ≡ `embeddings[i]`, positionally
  (card names duplicate). Never partially regenerate after a card-count change.
- **Cache ordering**: check → spawn → write → validate → **record last**. Recording before
  validating poisons the cache.
- **A cache MISS is information, not a bug.** `cache-status` reports MISS for a routine
  that has never run as well as one whose inputs moved, and the two need opposite
  responses. Check the `changed` list before spawning: an empty one means the routine was
  never recorded and there is nothing to re-bless. Never `cache-record` to make the board
  green — the record is the claim that a human read the artifact and agreed it holds.
- **Bump `?v=N`** on the script and CSS tags in `viz/index.html` after any frontend edit.
- **The combo graph is format-agnostic.** Commander Spellbook lines may quietly assume a
  card is *your commander* — `"Infinite commander casts"` in `produces` is the tell.
  Verify before publishing; Judge's Desk Case A-004 is the cautionary tale.
- **Rebuild the strategy DB after editing `strategy.md`** — the loader hard-errors on a
  sha256 mismatch.

## Deployment

GitHub Pages serves the repo directly. There is no root index; the two entry points are
`/viz/index.html` and `/manuals/index.html`. Pushing to `main` deploys.

## Where to read next

| Doc | Covers |
|---|---|
| `CLAUDE.md` | Orientation, environment, gotchas — the densest single page |
| `PLAN.md` | Current state and what's next |
| `/publish-deck` | The deck lifecycle: every phase in order, with its gate |
| `STYLEv3.md` | The magazine's editorial and design constitution |
| `docs/architecture.md` | Models, mechanical tags, synergy rules, power creep, regions |
| `docs/pipeline.md` | All 15 steps: inputs, outputs, runtimes, when to re-run |
| `docs/data-artifacts.md` | Every `data/` file: producer, size, git status, consumers |
| `docs/viz.md` | Frontend structure, the `window.MM` API, Pages layout |
| `docs/testing.md` | Test layout, skip markers, conventions |
| `docs/pilot.md` | Evidence contract, rules and strategy DBs, the magazine layer |
| `docs/deck-builder-v2.md` | The deck builder: bracket engine, role taxonomy, architect ⇄ critic loop |
| `docs/agent-cost.md` | Where LLM spend lives, per-routine costs, the cache |

## Non-goals

**Lint and formatting** are intentionally absent. Match the surrounding style; the test
suite is the gate. Adding a formatter to 20,000 lines would produce one enormous diff and
no information.

CI *was* on this list, on the grounds that a single-author project does not need it. It is
here now — `.github/workflows/test.yml` runs the fast suite and checks that every manual
still rebuilds byte-identically — because a pull request from someone else arrives
unverified otherwise.

---

Card images and card text are property of Wizards of the Coast. This is unofficial fan
content permitted under the Wizards of the Coast Fan Content Policy, not approved or
endorsed by Wizards.
