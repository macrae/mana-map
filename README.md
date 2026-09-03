# Mana Map

[![tests](https://github.com/macrae/mana-map/actions/workflows/test.yml/badge.svg)](https://github.com/macrae/mana-map/actions/workflows/test.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![the map](https://img.shields.io/badge/live-the%20card%20map-8b5cf6)](https://macrae.github.io/mana-map/viz/index.html)

**A workbench for crafting, experimenting, researching and analysing Commander decks.**
It is built around one idea: **a claim about a deck is worth what the experiment behind it
is worth.** `docs/vision.md` is the page everything else is written against.

At the centre is **simulation** — two engines answering different questions:

- **Forge**, the real rules engine, run headless and **seeded**, playing your list against
  your pod's actual decks. Same inputs, same games, byte for byte.
- **A seeded Monte Carlo goldfish** — 10,000 games of resource development against nobody,
  for the questions that are about a curve rather than a table.

Around them sit the things that make an experiment mean something: a deterministic builder,
a rules-citation loop for lines that must be *proven* rather than measured, dated web
reconnaissance, deterministic card mining over 34,890 cards, and a frontend that surfaces
the results.

Optimised for one player; open-sourced so anyone can stand up their own bench, not so
anyone else is supported.

## The hypothesis loop

```
  a question                     →  an experiment              →  a result you can cite
  "does it want more lands?"        experiment --a V1 --b V2      +0.27 mana on t5
  "is this line lethal?"            /resolve-stack                ✓ or refuted, with CR cites
  "how fast does it go off?"        goldfish                      mean t4.19, 89% by t6
  "what do strong lists run?"       deck-recon, /prescribe        ranked, cited, skeptic-checked
  "what would fix this axis?"       deck-audit + card-search      candidates that move the number
  "is this change worth buying?"    net-change, then propose      a trade, priced, and a pull list
```

**`experiment` is the flagship.** Two versions of a deck, the same table, the same N, the
same seeds, one artifact carrying both arms, the delta, and the sentence people skip —
whether the intervals overlap. Same seeds are **not** paired games (a changed list changes
every shuffle), so seeds buy per-arm replayability and the control is N. An A/A is refused
with the reason.

## Four pages over one data layer

**The workbench** (`viz/workbench.html`) — **the landing page**: every deck you own, in
racks by whether it is sleeved, **waiting on cardboard**, on the bench or history; or as
one fleet table across record, stages, evidence, table and open work — sortable by
*recently played*, *needs game logs*, *needs analysis*, *optimisations identified* and
*waiting on cardboard*. Each deck carries a derived **next**, and three named links: its
Pilot's Manual, its dossier, and where it sits on the map.

**The card atlas** — every Magic oracle card (~34,900) embedded by two small neural nets.
It opens on **one card**: hover it, click a relation, and its neighbours join a
force-directed graph you grow by clicking. Load one of your own decks and it lights up with
its commander ringed. The 34,890-point atlas is one click away, and drifts slowly at
altitude, settling as you zoom in to read. Three relations, each precomputed so a click is
instant: **similar** (embedding neighbours), **synergy** (rule-based complements, each edge
labelled with its rule), **outclassed by** (strictly-better replacements). Boot costs 1.8 MB.

**The deck page** (`viz/deck.html?deck=<slug>`) — the workbench surface: what to do next,
where the deck stands, every list it has been, what limits it, the engine, **the
experiments and simulation runs with their intervals**, prescriptions, the captain's log,
open questions, and the deck's own constellation. It renders `info.json` — the shape
`deck-info` composes — rather than re-deriving anything, so it cannot disagree with the
command that owns each figure.

**The branch workbench** (`viz/branch.html?deck=<slug>&branch=<name>`) — one candidate 99
and the decision about it: what it would become and what it was accepted on, the verdict,
every measured row with its definition and a plain-language reading, reward / risk / cost,
and the bill. A branch is a deck that does not exist, so it gets its own page rather than
growing into the dossier.

## The commands behind it

- `simulate <slug> --vs <pod> --games N` — N seeded Forge games: win rate with its interval,
  who kills you and how, the kill curve, **commander damage per defender**, token pay-off.
- `experiment <slug> --a <ref> --b <ref> --vs <pod>` — the controlled A/B.
- `goldfish` — seeded Monte Carlo resource development; Treasure and combat opt-in.
- `build-deck` — a legal 99 from a brief: role budget crossed with a **cited curve target**,
  combo lines completed, bracket-gated. No agents required.
- `deck-audit` — 16 axes, each carrying the verbatim `strategy.md` quote that sets its target.
- `card-search` — deterministic mining over the corpus: identity, oracle/name regex, role,
  cmc, and `--owned` against your physical boxes.
- `deck-info <slug>` — the whole join, and a derived **next**.
- `deck-branch` — a candidate 99 you cannot yet sleeve: stage swaps, measure it against the
  deck with `net-change`, then **`propose`** it as the next version and wait for the
  cardboard. The decision is frozen; the blocker is recomputed from your boxes on every
  read, so a proposal un-blocks itself when a card lands in one.
- `deck-version` — every list the deck has been, from git, joined to the games played on it.
- `deck-notes add` → `/debrief` → `/prescribe` — the table, structured, then answered.
- `/resolve-stack` — a board (authored, or **lifted from a simulated game**) resolved with
  Comprehensive Rules citations and adversarially checked. The ✓ tier.
- `analyze-engine` — the deck's machine as eight stages, solid where a stack proves a line.

Under it all is a **three-tier evidence contract** that never moves: ✓ rules-verified, ◆
data-derived (seeded where randomness is involved), ★ coaching — labelled judgment, never
disguised as measurement. **A figure travels with its interval, its N and its limits, or it
does not travel** — enforced in code, not by convention. Every agent returns JSON a
validator checks; the Python makes zero LLM calls; the deployed site and your machine run
the same code.

**Three things this is honest about.** Forge's AI pilots every seat *including yours*, and
rates itself "poor to ok in control, pretty bad for combo" — a sentence quoted verbatim in
every run record, which makes a control deck's win rate a lower bound on the pilot. **Two
decks have a real game logged** (Edgar and Ur-Dragon, both losses, both debriefed) — which
is two, not a sample: the log, debrief and prescription surfaces are built and tested, and
barely used. And **most decks are not marked as built in paper**: whether a deck exists as
cardboard is an assertion only the pilot can make, so an unlocked deck says it is unlocked
rather than being assumed playable.

**The Pilot's Manual** (`manuals/p/<slug>.html`, from `manamap pilot build-page`) — each
deck's self-contained printable page: the game plan, the mulligan, the verified lines
argued, the engine, the numbers with their intervals. **No `<script>` anywhere**, so it
rebuilds byte-identically and is trustworthy offline and in print — which is exactly why
live map embeds go on the dossier instead. The *legacy magazine* (nine frozen issues;
[the rack](https://macrae.github.io/mana-map/manuals/index.html)) still renders from
`build-manual` and is no longer linked from any live surface.

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

## Set up the bench for your own decks

This half is more involved, and honest about why: **the agent phases need
[Claude Code](https://claude.com/claude-code).** The Python in this repo makes *zero* LLM
calls — it is deterministic infrastructure (fetching, simulating, validating, rendering)
that AI agents drive from the outside, and most of the bench (`deck-info`, `deck-version`,
`deck-notes`, `goldfish`, `simulate`, every validator) is pure CLI. The agent routines —
the doctor, the resolver and checker, the engineer and critic, the notes writer, the
debrief — are what cost tokens; an invocation cache is what makes iterating on them
affordable. `docs/agent-cost.md` has the breakdown.

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
deck page shows the actual cards in your deck, with the right art. A name-only list works
but yields default reprints.

`fetch-deck` short-circuits when the decklist hasn't changed; use `--force` after oracle
errata.

### 4. The files you write by hand

Nothing scaffolds these. `data/decks/goblin-storm/` is the worked reference.

| File | Why it's manual |
|---|---|
| `decklist.txt` | It's your deck — **unless you let the builder write it**, see below |
| `issue.json` | The deck page's authored identity (name, commander, status). **The legacy build hard-exits without it** — a *generated* date would break byte-identical rebuilds |
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
and the whole path from brief to a finished deck is proven (hapatra was built this way).

### 5. The lifecycle, then the loop

Run the skills from Claude Code in the repo root (each is in `.claude/skills/`).
**`/publish-deck` sequences the lifecycle** and `manamap pilot deck-info <slug>` tells you
where a deck stands and what to do next; start with both.

| Step | Produces | Pure CLI? |
|---|---|---|
| `/build-deck` | A 99 from a brief: pool → architect ⇄ critic, bracket-gated | no |
| `manamap pilot bracket-check <slug>` | Computed bracket floor + its evidence | **yes** |
| `manamap pilot goldfish <slug>` | Resource curves from 10k seeded games | **yes** |
| `manamap pilot deck-map <slug>` | The constellation: local layout + clusters | **yes** |
| `/analyze-engine` | The engine: stages, lines, what a stack actually proves | no |
| `/resolve-stack` | A verified line: resolver → validator → adversarial checker | no |
| `/write-manual` | The pilot's notes: game plan, mulligan, line intros, threats, matchups | no |
| `manamap pilot build-page <slug>` + `build-index` | The Pilot's Manual (`manuals/p/`, deterministic, no `<script>`) | **yes** |
| `manamap pilot build-manual <slug>` | The legacy magazine issue (frozen renderer) | **yes** |

Then the loop the bench exists for — all CLI except the two agents:

| | | |
|---|---|---|
| `manamap pilot deck-branch <slug> new/stage/…` → `net-change --branch` | a candidate 99, measured against the deck | **yes** |
| `manamap pilot deck-branch <slug> propose <name> --as v1.0.2` | accept it, and wait for the cardboard | **yes** |
| `manamap pilot deck-version <slug> [tag …]` | commit the list; every version numbered from git | **yes** |
| `manamap pilot fetch-opponent "<commander>"` / `simulate <slug> --vs <pod> --games N` | your table, in Forge, seeded | **yes** |
| `manamap pilot deck-notes <slug> add "…" --result win\|loss` | the captain's log | **yes** |
| `/debrief <slug>` | the note, structured and routed | no |
| `/prescribe <slug> "<question>"` | the doctor's answer, priced and skeptic-checked | no |
| `manamap pilot sim-scenario <slug> <run> --game G --turn T --stack` → `/resolve-stack` | a simulated board, proven | mixed |

---

# For developers

## The shape

Two pipelines, same pattern: each stage writes artifacts the next one reads, and every
stage is independently runnable and testable.

```
Card map    download → extract → preprocess → train ×2 → embed → reduce
                     → combos → export → synergy → power-creep → regions → card-roles → viz

Build       brief.json → build-deck → bracket-check → architect ⇄ critic → decklist

Experiment  decklist → fetch-deck → goldfish ─┐
            pod       → fetch-opponent ───────┼→ simulate / experiment → parse → analysis
                                              └→ sim-scenario → /resolve-stack → ✓

Change      decklist → deck-branch new → stage → net-change → propose → merge → a version
                                                        └ blocked on cardboard, derived

Surface     artifacts → deck-info --write → info.json ─┐
            build-index → index.json ──────────────────┴→ viz/deck.html
```

`manamap run` drives the first (15 steps, ~40–60 min, internet at two of them).
`manamap pilot <cmd>` drives the rest (95 pilot subcommands). All constants live in
`src/manamap/config.py`; both CLIs are registry-driven with lazy imports.

## Simulation — the centre of the bench

*Code: `src/manamap/sim/{forge,parse,experiment,bridge,opponents,validate_sim}.py`.
Design, the spike and the verdict: `docs/simulation.md`.*

**Forge was chosen by measurement, not preference.** Three things were checked before
committing to it: every log line parses, 4-seat Commander runs headless, and `-s` makes a
run byte-replayable. Writing a rules engine was shelved for one narrow deterministic case.

**A run is seeded.** `simulate` converts each seat's `decklist.txt` to a Forge `.dck`
*through the repo's own parser* — so a deck analysed and a deck simulated can never
disagree about what is in it — then runs N Commander games across J JVMs. The default seed
derives from the configuration, so **the default replays**; `--seed` asks for a new sample.
Job *i* runs `seed_base + i`, and a same-id re-run is refused without `--force`.

**The record is one tracked JSON.** Outcomes, per-game rows, every seat's decklist sha,
Forge and Java versions, wall time, the seeds, and an `analysis` block with Wilson
intervals for rates and normal intervals for means. `validate-sim` **re-derives that
analysis from the kept logs** where they exist and form-checks where they do not, so a
figure in the record is not merely asserted.

**Two turn counts, and they are not the same.** `round` is the winner's own turn count
(Forge's `Game Outcome: Turn N`); `global_turn` is the game's last `Turn:` line. In a
4-seat game round 8 is global turn ~32.

**Three things the parser gets right on purpose.** Tokens are reported two honest ways —
`token_resolutions` (creation abilities that resolved; blind to X and doubling) and
`tokens_observed` (distinct ids seen acting) — because Forge names a token on first *use*,
never on creation. Damage figures see **damage only**: a drain kill shows in `life_by_turn`
and `eliminated_how`, never in a damage total. And **commander damage is per defender**,
because CR 903.10a asks for 21 from one commander on one *player* — 60 damage spread over
three seats wins nothing.

**Every aggregate carries median, min, max and an interval.** A mean over a skewed sample
is a true number that describes no game: one measured arm read mean 17.42 with a **median
of 0**, the whole difference being two games out of twelve.

**The bridge closes the loop.** `sim-scenario <slug> <run> --game G --turn T --step S`
lifts a board out of a simulated game into a `game_state` v2 scenario, which
`/resolve-stack` then proves with rules citations. Life and lands are exact; hand size is
an estimate; every approximation is written into `extras.reconstruction_notes`. This has
run for real: radagast stack 008 is a board lifted from a simulated game, resolved and
checker-passed in three iterations, and the checker caught two triggers the author missed
that the log confirms.

**The AI caveat is not a footnote.** Every seat is a Forge AI including yours, and Forge's
own rating — "poor to ok in control, pretty bad for combo" — is quoted verbatim in every
run record's `assumptions`. Measured: no AI profile flies a hold-up deck better than
Default (Default 3/6, Experimental 2/6, Reckless 2/6 over seeded games), so Default stays
the default.

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
| A deck lifecycle phase | `STAGES` in `pilot/deck_status.py`, or the next person will not find it |
| A pod seat | `manamap pilot fetch-opponent "<commander>"`, or a `decklist.txt` under `data/opponents/<slug>/` |
| A figure the sim reports | `game_facts` + `aggregate` in `sim/parse.py`, then re-derive every run with `--analyze` — the record is compared against the logs, so an added key must be backfilled |
| A panel on the deck page | a `*Panel(d)` function in `viz/js/deck-view.js` returning `''` when its artifact is absent, plus the artifact's filename in `build_index.gather_entries` if a browser cannot list it |
| A field the deck page reads | `deck_info.compose`, then `deck-info <slug> --write` for every deck — `info.json` is committed and staleness-gated |
| A section of the (legacy) deck page | **don't** — the magazine renderer (`issue_spec.DEPARTMENTS`, `design.py`) is frozen; the compact page is `docs/manual-v5-spec.md` |
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

**Evidence tiers.** Every artifact and every figure carries a tier: ✓ rules-verified,
◆ data-derived (seeded where randomness is involved; *sampled* said out loud where it cannot
be), ★ coaching. A validator per artifact enforces that nothing claims a tier it was not
granted — costume never earns the badge.

**Validation gates — form in code, meaning by agent, publication by verdict.** A mechanical
validator checks structure: does every step cite a rule, does that rule exist, is the quote
a verbatim substring. A *separate adversarial agent* then fetches each full rule and judges
whether it actually supports the claim. Only `pass` renders. Failed artifacts are kept —
they document open questions.

**Determinism.** Agents return JSON and never write HTML. That keeps the renderer a pure
function of committed artifacts, byte-identical on rebuild and enforced by tests — and it
is *why* the deck page's date is authored, why goldfish and Forge runs are seeded, and why
image URLs get their cache-busters stripped.

## Testing

```bash
make test          # the inner loop: non-browser, parallel, cached   ~2.5 min
make test-fresh    # same, nothing served from the cache             ~2.3 min
make test-browser  # the playwright suite                            ~7 min
```

A bare `pytest` is `make test`. Four test files recompute an artifact and compare it to
the tracked copy — 90,000 seeded goldfish simulations among them — so those are cached on
a hash of their inputs *and* of the code that produces them, recorded only on a pass, and
kept in gitignored `.pytest_cache/` where they cannot reach another machine or CI. The run
prints how many it skipped. `make test-fresh` is the one to trust before a PR.

A fresh clone skips the cases that gate on gitignored artifacts built locally, and each
one says which command would enable it — so a clone runs green and faster than a developed
checkout. Current counts and the measured timings live in `docs/testing.md`.

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
  Verify with a stack before stating it as a line; goblin-storm stack 004 is the cautionary tale.
- **Rebuild the strategy DB after editing `strategy.md`** — the loader hard-errors on a
  sha256 mismatch.
- **A mean is not a result.** Report the median and the interval beside it or the number
  lies on a skewed sample. `mean_ci` emits all of them; do not unpack only `mean`.
- **Scryfall leaves `mana_cost` EMPTY on transform and MDFC layouts** and puts the cost on
  each face — and holds *both halves* on adventure and split layouts. Read
  `common.front_field(card, "mana_cost")`, never `card["mana_cost"]`, or you under-count
  one class of card and double-count another.
- **`--identity` takes letters, and `parse_color_identity` splits on commas.** It is right
  for `cards.csv`'s `"G, U"` and returns `{"GU"}` — one token nothing can be a subset of —
  for the compact form a human types. `card_search.parse_identity_arg` is the CLI parser.
- **Ownership means a BOX.** `data/decks/` holds build plans as well as sleeved decks and
  nothing tells them apart, so `collection.owned_names()` deliberately does not count deck
  membership.
- **`VALIDATED` and `STAGES` are different lists.** An artifact with a gate but no
  lifecycle stage still has to be reported, or `deck-status` says green while the gate is
  red — which it did, fleet-wide, for three artifacts.

## Deployment

GitHub Pages serves the repo directly. There is no root index; the entry points are
**`/viz/workbench.html` (the landing page — start here)**, `/viz/index.html` (the card
atlas), `/viz/deck.html?deck=<slug>` (one deck's dossier) and `/manuals/p/<slug>.html`
(its Pilot's Manual). `/manuals/index.html` is the legacy magazine rack, still rendered
and no longer linked from any live surface. Pushing to `main` deploys.

One artifact the deployed site cannot carry yet: **the version list**. `deck-version`
derives it by walking git, and the commit that changes `decklist.txt` receives its sha
*after* anything written in the same commit — so a committed copy is one version behind
forever. It needs a deploy-time step (a Pages workflow checking out with `fetch-depth: 0`),
and until that exists the deck page's version panel simply does not render, which is what
every panel there does when its artifact is absent.

## Where to read next

| Doc | Covers |
|---|---|
| `docs/vision.md` | **Start here.** Who this is for, what the bench does, what is live / legacy / next |
| `CLAUDE.md` | Orientation, environment, gotchas — the densest single page |
| `PLAN.md` | Current state and what's next |
| `/publish-deck` | The deck lifecycle: every phase in order, with its gate |
| `docs/simulation.md` | The Forge engine: the spike, the harness, the parser, the pod, the bridge |
| `docs/manual-v5-spec.md` | The compact deck page that replaces the magazine (spec) |
| `docs/agent-audit-2026-08-19.md` | The agent audit behind the pivot |
| ~~`STYLEv3.md`~~ | The legacy magazine's constitution, **deleted 2026-08-25** with the rest of the magazine era. `git show 23e8cec:STYLEv3.md` |
| `docs/architecture.md` | Models, mechanical tags, synergy rules, power creep, regions |
| `docs/pipeline.md` | All 15 steps: inputs, outputs, runtimes, when to re-run |
| `docs/data-artifacts.md` | Every `data/` file: producer, size, git status, consumers |
| `docs/viz.md` | Frontend structure, the `window.MM` API, Pages layout |
| `docs/testing.md` | Test layout, skip markers, conventions |
| `docs/pilot.md` | Evidence contract, the bench's commands and artifacts, rules and strategy DBs |
| `docs/agent-cost.md` | Where LLM spend lives, per-routine costs, the cache |

## Non-goals

**Lint and formatting** are intentionally absent. Match the surrounding style; the test
suite is the gate. Adding a formatter to 20,000 lines would produce one enormous diff and
no information.

CI *was* on this list, on the grounds that a single-author project does not need it. It is
here now — `.github/workflows/test.yml` runs the fast suite and checks that every deck page
still rebuilds byte-identically — because a pull request from someone else arrives
unverified otherwise.

---

Card images and card text are property of Wizards of the Coast. This is unofficial fan
content permitted under the Wizards of the Coast Fan Content Policy, not approved or
endorsed by Wizards.
