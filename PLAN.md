# PLAN — current state and what's next

*The resume-here doc. `docs/vision.md` says what this is for; `CLAUDE.md` carries the
gotchas; this says what exists and what is open. The magazine era's plan is archived
verbatim in git at `git show 23e8cec:docs/history/PLAN-2026-08-magazine-era.md`.*

Last updated **2026-09-01**. Everything below is committed and pushed to `main` except
where marked. Every figure was derived from the repo at write time — **do not quote one
from memory**; the command that prints it is named beside it.

## What this is

A **workbench for crafting, experimenting, researching and analysing Commander decks**,
built around one idea: a claim about a deck is worth what the experiment behind it is
worth. **Simulation is the centre** — Forge for the real rules against a real pod, a
seeded Monte Carlo goldfish for the questions that are about a curve rather than a table —
and `experiment` is the flagship: two versions, one table, one artifact, the delta and the
overlap sentence. Around it sit a deterministic builder, a rules-citation loop for lines
that must be proven, dated reconnaissance, deterministic card mining, and a frontend that
surfaces the results.

Optimised for one player (the maintainer, in Orinda); open-sourced, not externally
supported. The magazine that used to be the product is a frozen legacy renderer until the
compact deck page replaces it. The card atlas in `viz/` is unchanged and live; the **deck
page** (`viz/deck.html?deck=<slug>`) is new and is the workbench surface.

Scale (derived; `tests/test_docs_counts.py` polices these): 85 pilot subcommands,
27 top-level subcommands, 15 agents, 19 skills, 10 static cache routines
(plus `stack:`/`decision:`/`prescription:` per artifact). Test counts live in
`docs/testing.md` only.

## Start here on any deck

```bash
manamap pilot deck-info <slug>        # where it stands, and a derived NEXT
manamap pilot deck-status <slug>      # lifecycle: present / missing / STALE / INVALID, gates run
```

Staleness is an ERROR; incompleteness is a state. A half-built deck is work in progress.
A deck whose artifacts disagree about which decklist they describe is confident and wrong.

## The evidence contract

| | tier | granted by |
|---|---|---|
| ✓ | rules-verified | a stack artifact whose every step cites a real CR rule verbatim (`validate_stack`), then survives the adversarial `rules-checker`. Only a `pass` publishes. |
| ◆ | data-derived | deterministic Python over committed artifacts; **seeded** where randomness is involved (goldfish, Forge runs), **sampled** said out loud where it was not (the first Forge run). |
| ★ | coaching | labelled judgment. Useful, never disguised as measurement. |

Agents return JSON and validators check it; no agent writes prose that claims a tier it
was not granted; a figure travels with its interval, its N and its limits.

## The decks

`deck-info <slug>` per deck for the live picture; `deck-status --all` for the fleet. As of
this writing (derived from the stack, bracket, engine, sim and log artifacts):

| deck | stacks ✓/total | bracket floor/target | engine | sim runs | logged | status |
|---|---|---|---|---|---|---|
| `goblin-storm` | 5/5 | 4/4 | pass | 0 | **1** | v1.0.0 ◆ SLEEVED; 0W 1L |
| `hapatra` | 1/1 | 4/4 | pass | 0 | 0 | `broken-down` (cards live in yawgmoth) |
| `sisay` | 1/3 | 4/4 | pass | 0 | 0 | `retired` — **not the pilot's deck** |
| `heliod` | 6/6 | 4/4 | pass | 0 | **1** | v1.0.0 PLACEHOLDER, not paper-locked; 0W 1L; needs PROTECTION |
| `ur-dragon` | 6/6 | 4/4 | pass | 1 | **2** | **v1.0.1 ◆ SLEEVED** (what is being played); v1.0.2 paper on the way; 1W 1L |
| `edgar-vampires` | 11/11 (9 presentable) | 4/4 | pass | 1 | **3** | v1.0.0 ◆ SLEEVED; 0W 3L; **direction changed 08-28** |
| `gishath` | 5/5 | 4/4 | pass | 0 | **1** | v1.0.0 PLACEHOLDER, not paper-locked; **1W 0L** — the fleet's first win |
| `yawgmoth-swarm` | 14/14 (11 presentable) | 4/4 | pass | 0 | 0 | paper rebuild in progress |
| `radagast` | 8/8 | 1/3 | pass | **2** | 0 | `broken-down` (2026-08-21) |
| `kianne` | 0/0 | 4/4 | — | **2** | 0 | **concept abandoned** — voltron is 10 decks of 970; the shell is sound, the win condition is not |
| `kinnan` | 0/0 | 4/4 | — | 0 | 0 | deterministic baseline built; recon done; commander not owned |

**Eight Forge runs and six experiments exist**, across four decks — edgar (3 runs, 2
experiments), kianne (2, 2), radagast (2, 2), ur-dragon (1). The other seven have not been
simulated at all.

**Five decks now have a real table logged — eight games, 2W 6L.** edgar (3), ur-dragon
(2), goblin-storm (1), heliod (1), gishath (1). **Six of the eight are NOT yet
debriefed**, deliberately: the pilot's order is modelling first, `/debrief` after.

The night of **2026-08-28** at Alex's (Moraga Way, Orinda; three-player pod with Alex and
Stuart) put four of those on the board in one sitting — goblin-storm, edgar, heliod and
gishath, in that order, **1W 3L with the Dinosaurs taking it**. Three of those decks had
never been logged at all.

**Three decks are not marked as built in paper** (yawgmoth-swarm, kianne, kinnan). That is
a third state, distinct from the three that no longer exist as cardboard: nobody has said
either way, and `deck-info` now says so instead of assuming.

**heliod and gishath came off that list by being PLAYED**, which is a stronger fact than
the lock records. Both are tagged **v1.0.0 as a PLACEHOLDER** and deliberately NOT
paper-locked: the lock asserts that the exact committed list is what is sleeved, and that
is precisely what a check-in establishes. The pilot will supply both lists and will not
change either deck before then, so the committed lists are stable in the meantime and the
log entries' decklist stamps hold.

Three decks no longer exist as cardboard (`hapatra`, `sisay`, `radagast`). Their
artifacts stay exactly as published; `deck-info` states the status and withholds the
suggestions that would need a deck to shuffle.

## Where it stands (2026-08-25)

| | | where |
|---|---|---|
| Agent audit + Sprint 0 | 18 → 15 agents; shared contract (`.claude/agents-common.md`); L10 repealed; magazine editor/panel/short-list retired; writer + coach → `pilot-notes`; `debrief` new; doctor MODE prescribe | `docs/agent-audit-2026-08-19.md` |
| MVP Sprints 1–3 | `deck-version`, `deck-notes` + `/debrief`, `prescribe`, `deck-info` | `docs/pilot.md` |
| Simulation S0–S5 | Forge spike + verdict; seeded harness; parser with CIs; `validate-sim`; the pod; the bridge `sim-scenario` → `game_state` v2; the doctor reads the table | `docs/simulation.md` |
| The chain, once for real | stack 008 — a board lifted from a simulated game, resolved + checker-passed in 3 iterations; matched Forge's log line for line | `docs/simulation.md` |
| `experiment` + AI profiles | the controlled A/B, one accumulating artifact; `--profile` on both commands (Default stays default, measured) | `docs/simulation.md` |
| Commander damage | per DEFENDER through the parser, the record and `experiment`'s delta; all runs migrated | `sim/parse.py` |
| The distribution | `mean_ci` carries median/min/max beside the mean — a skewed arm read mean 17.42, median 0 | `sim/parse.py` |
| `card-search` | deterministic corpus mining: identity, oracle/name regex, role, cmc, `--owned` | `docs/pilot.md` |
| The collection | `pilot/collection.py`, the one reader of `COLLECTION_DIR`, memoized | `docs/pilot.md` |
| `validate-recon` | the gate `deck_recon.json` never had; its first catch was a data gap, not an agent error | `pilot/validate_recon.py` |
| Builder curve + combos | role quota crossed with a **cited** mana-value target; `complete_combos` finishes a line the deck half-holds | `pilot/build_deck.py` |
| **The deck page** | `viz/deck.html` — nine workbench panels over `info.json`, sim figures with intervals, lifecycle flag | `docs/viz.md` |
| **The Pilot's Manual** | `manuals/p/<slug>.html` from `build-page` — the compact technical page, no `<script>`, rebuilds byte-identically. The magazine is unlinked from every live surface | `docs/manual-v5-spec.md` |
| **The first real games** | edgar v1.0.0 and ur-dragon v1.0.0, one logged game each, both debriefed, both feeding a prescription | `data/decks/*/log.jsonl` |
| **The workbench landing page** | `viz/workbench.html` — racks by whether a deck is SLEEVED, plus a fleet table sorted four ways over every `info.json`. Three labelled links per deck | `docs/viz.md` |
| **The fusion** | an open verified line prints its prose (50 of 50 covered); engine arrows on the 5 of 196 edges whose direction is a fact; a curve bar is a control | `docs/viz.md` |
| **Seed a walk from named cards** | textarea + `?cards=`; the enumeration separates, never the comma (9.2% of names contain one); *Start here* vs *Add to walk* | `docs/viz.md` |
| **The version policy** | PATCH/MINOR/MAJOR by capability, every slug from v1.0.0; releases sort numerically, near-misses refused, re-tagging needs `--force` | `docs/pilot.md` |
| **The paper lock's third state** | UNLOCKED is not dead. Four of eleven decks are unlocked and now say so; three rehearsal locks withdrawn | `docs/pilot.md` |

## Open work

### DONE — the speed sprint (2026-08-30/31)

**The complaint was that iteration had become heavy: questions slow, fleet
regeneration manual, and fidelity surprises discovered after the run.** Three
audits (tests, simulation, interactive path) said the Python simulation was
never the bottleneck — the fan-out around it was. The whole fleet regenerates in
**78 seconds of CPU**; the same regeneration used to cost **6-9 MILLION agent
tokens**, and that ratio was the entire problem.

**The single most expensive line in the repo was a provenance stamp.**
`goldfish.model_version()` is a sha over the whole of `goldfish.py`, and ten
`AGENT_ROUTINES` declarations hashed the file it is stamped into. A COMMENT edit
moved the digest on every deck and hard-MISSed strategic-frame, pilot-notes,
tutor-guide, deck-diagnosis, every decision and every prescription. Measured over
four real goldfish commits: **45 artifacts stamped stale, 31 with figures that
actually moved — 31% of the spend bought nothing**, and `deb711e` changed one
docstring line and invalidated the fleet. Excluded from the fingerprint; the
stamp stays in the artifact and `model_staleness` still reports it. The next
commit proved the point — a 30% goldfish speedup that moved no figure at all and
cost nothing.

| | before | after |
|---|---|---|
| `query-rules` / `query-strategy` | 6.93s | **0.16s** (43x) |
| `deck-facts` | 1.44s | **0.14s** |
| `deck-audit` | 2.26s | **0.59s** |
| `deck-info` | 7.8s | **1.25s** |
| `mde_proportion(0.25, 200)` | 2.17s | **0.21s** |
| goldfish (edgar) | 5.70s | **3.96s** |
| whole-fleet regen | a hand-written shell loop | **78s** |
| `make test` (warm) | ~101s | **~74s** |

**`manamap serve` is a warm worker.** Every CLI invocation was a cold process and
every memo is per-process — including the frozen MiniLM behind `query-rules`,
~8s to build and thrown away each time, while `rules-lookup` tells the agent to
"try several phrasings". `/api/cli` runs read-only pilot commands in the warm
process behind an allow-list; the terminal routes to it when one is listening and
**fails open** on any error. It holds the modules it started with, so restart it
after a code change.

**`manamap pilot regen`** rebuilds the fleet in dependency order, parallel across
targets — 72 targets, 78s, **bit-identical** (`git status data/` empty after).
Parallel across DECKS, never across games: one `random.Random(seed)` is threaded
through all 10,000 games, so splitting them would re-base every figure.

**`manamap pilot model-coverage`** answers the fidelity question in the other
direction — not "what did the channel miss" but "what would this deck need, and
is it switched on". **236 DARK cards across the fleet**; gishath is a Dinosaur
deck with 33 cards whose combat the model was told not to look at. `goldfish` and
`net-change` print it as a PREFLIGHT, so it arrives before the games.

**Forge's `-c` clock ends a game's accounting, not its AI thread.** Two tracked
20-game runs took **3.7 and 4.2 hours** with 95% of the wall claimed by no game.
Jobs are capped now; checked against all 18 tracked runs, the two pathological
ones die and **all sixteen others survive** — **7.1 hours** on that set.

**Three statements that were false, now corrected in place:** `forge.ASSUMPTIONS`
claimed a clock-hit game is recorded as a draw (it carries a winner — 75 of
edgar's 400 games, 19%, with zero recorded draws); `SEEDED_NOTE` claims
byte-for-byte replay (it diverges at game 1 on the 400-game run); and
`docs/agent-cost.md` claimed no Python spawns a subprocess (`serve.py`'s `ask`
shells out to `claude -p`).

Full record: `docs/gotchas-bench.md`.

### IN FLIGHT — the embedding architecture (2026-08-31 → 09-01)

Replacing a contrastive objective whose positives are mined from the repo's own
regexes. **The approach changed twice, and both changes were forced by a
measurement rather than an argument.**

#### Phase 1 (the eval) — DONE, and it moved the target

The eval measured one relation against one candidate pool and reported bare
differences. It now measures three relations, a geometry, and carries an interval
on every gap. Two findings from building it: **the `-0.012` that named issue #12
is a TIE** (interval [−0.088, +0.060], never computed), and **the
commander-search contradiction is settled** — text's advantage is entirely
thematic (0.470 against the function space's **0.005** on tribal commanders),
because `train_ability` mines positives from roles and tags and "Vampire" is
neither.

#### The VAE was built, measured, and abandoned — by its own control

`card_serialize` + `model_vae` + `train_vae` shipped and trained. The control is
the whole finding:

    frozen MiniLM 384d          function 0.629   theme 0.523   effdim 51.39
    PCA 128d of it              function 0.648   theme 0.494   effdim 42.62
    RANDOM 128d projection      function 0.602   theme 0.444   effdim 37.69
    the TRAINED VAE             function 0.618   theme 0.387   effdim 34.19

**A random projection beat the trained model on theme and PCA beat it on
everything.** Training bought less than a matrix multiply with random numbers,
because there was nothing to learn that was not already MiniLM. The artifact is
kept as a scored baseline; nothing depends on it.

#### The redirect: this is TABULAR data with some text columns

Set by the pilot, 2026-08-31: *"we went too literal and too far down the path of
language modeling… an input could just be CMC (int) or color identity (one-hot
array)… we are treating this like a language problem instead of a tabular data
problem (with some language inputs)."*

The serialiser flattened every card into one string and pushed it through a
sentence encoder, so CMC never existed as a number and colour identity never
existed as a set. Rebuilt as typed fields:

| module | what it is |
|---|---|
| `training/card_source.py` | the corpus as the MODEL sees it — `cards.csv` plus the two things the CSV threw away: oracle newlines (the ability boundary) and `produced_mana` |
| `training/card_fields.py` | **73 typed fields, 623 columns**, three states (PRESENT / ABSENT / MASKED). Numeric, Binary, Categorical, SetOf |
| `training/span_encoder.py` | **6 maskable text slots** over a frozen-MiniLM cache of 75,178 distinct spans |
| `training/masking.py` | correlated-group masking; `GROUPS` + `COMPANION` |
| `training/model_cardbert.py` | BERT where the tokens are FIELDS |
| `training/loss_cardbert.py` | one loss per field kind; InfoNCE for spans; VICReg |
| `analysis/recoverability.py` | which fields a lookup table already solves |
| `analysis/project_spaces.py` | every space projected side by side, to LOOK at |

**Absent is not zero, and it earned its keep on a case nobody predicted.**
Scryfall OMITS `produced_mana` for the 32,190 cards that make no mana rather than
writing an empty list, so a field reading a missing key as False would report the
whole corpus as making no mana — plausibly and silently. Command Tower went from
5 populated fields to 38 once `produces_*` existed.

**The name comes out of the rules text.** 4,401 cards (12.6%) say their own name
in their own abilities, so the `name` slot and an ability slot shared a literal
string. Split on commas and ` // `, never on spaces (a card named `Food Fight`
must not redact *Food* from "create a Food token"); possessives keep their `'s`.

#### The recoverability audit gates the objective

`manamap recoverability` fits a ridge probe per field from every other field,
held out by TEXT hash. **19 of 73 are solved by a linear probe** — `cmc` is the
pips added up (R² 0.96), `supertype` is the type flags (0.998), `color_identity`
is the coloured pips (0.956). Meanwhile `kw_deathtouch`, `kw_lifelink` and
`kw_trample` score NEGATIVE lift: the probe does worse than always guessing
false, so those are the informative targets.

**So masking one field is arithmetic, not a task.** `masking.GROUPS` hides
correlated blocks, and `COMPANION` hides the keyword TEXT alongside the keyword
flags because 99% of keywords appear verbatim in oracle text.

#### CardBERT, and the bug the eval caught

A card is 79 positions (73 fields + 6 spans) plus `[CLS]`; masking hides fields;
bidirectional attention predicts them; `[CLS]` is the product.

**The embedding was never trained.** `to_latent.weight.grad` came back **None**
after a full backward pass — every head reads its own field's position and
nothing reads `[CLS]`, so what shipped was a random projection of an untrained
state and scored like one (r@10 0.093, effdim 5.53). This is the textbook BERT
result reached from first principles: a raw `[CLS]` is a poor sentence embedding,
which is why SBERT exists. BERT survives it because it is always fine-tuned
downstream; here the embedding IS the product.

Fixed by making **masking the augmentation** — two independent maskings of one
card are two views, NT-Xent between their latents. 40 epochs, never early-stopped:

    space                              dim  effdim  spread   r@10   r@50  medRank
    layout (color+type)                128    3.89  0.0061  0.086  0.139     1148
    cardbert (masked fields)           128   16.72  0.1347  0.103  0.262      323
    vae (masked imputation)            128    5.71  0.0454  0.167  0.247      374
    function (ability)                 128   27.31  0.0323  0.232  0.464       76
    text baseline (frozen MiniLM)      384   51.39  0.1341  0.244  0.414      126

#### The result is a SPLIT, not a win — and the split is legible

Against the space it would replace, 95% CI on the DIFFERENCE:

    FUNCTION (28 groups)          THEME (55 groups, EDHREC tribes)
     100  0.759 vs 0.964  -0.205    100  0.537 vs 0.443  +0.094  excludes 0
     500  0.519 vs 0.794  -0.275    500  0.303 vs 0.152  +0.151  excludes 0
    2000  0.317 vs 0.562  -0.245   2000  0.127 vs 0.053  +0.074  excludes 0

It LOSES function at every size and WINS theme at every size. Not surprising once
stated: the function space mines positives from role and tag regexes, so function
is what it was built for; CardBERT reads types, subtypes, keywords and ability
spans, so tribe is legible to it in a way it never was to a role regex. **At pool
500 it doubles the baseline on tribe** — and theme was the function space's known
weakness, recorded when the commander-search contradiction was settled.

Two diagnostics agree. Hard-negative separation on the fastland/slowland cycle —
the canonical "should NOT look alike" failure — is **0.0377 against 0.0133**,
2.8x. Centroid headroom is **0.976 against 0.019**, the metric that explains why
centroid queries have nothing to rank on today.

**Commander search** (a centroid operation, 79 candidates): CardBERT is the best
TRAINED space and the only one with a perfect top20 — but the top1 ranges overlap
so that difference is the draw, and frozen text still wins outright.

    function (ability)     top1 0.410   top5 0.811   top20 0.967   MRR 0.587
    cardbert               top1 0.458   top5 0.908   top20 1.000   MRR 0.642
    text baseline          top1 0.584   top5 0.962   top20 0.996   MRR 0.746

**Nothing is cut over.** It is complementary, and strongest exactly where the
incumbent is weakest.

#### RULE — visual inspection is part of evaluation

Set by the pilot, 2026-09-01. `eval-embeddings` asks one question — are the k
nearest cards right — and its numbers do not describe a MAP. The VAE retrieves
better than CardBERT (0.167 against 0.103) with a third the spread and a tenth
the headroom: **a space can win recall@10 by CONCENTRATING and lose everything
that makes an atlas navigable.** `manamap project-spaces` projects every space
side by side, coloured by facts none of them optimised directly (colour identity,
card type, EDHREC tribe), so the question is "did this structure emerge" rather
than "was it supplied".

`--components 3` emits 3D. The pilot's framing: the Atlas as a UNIVERSAL map —
galaxies, solar systems, planets and satellites — which is a third scale on top
of the two `cluster_regions` already runs (HDBSCAN L0 at 800, L1 at 100). The
coordinates are a one-line change; **the cost is entirely the frontend**, since
`viz/render/canvas.js` is 2D through hit-testing, labels and the force graph. A
rotatable 3D→2D camera is the cheap path and gives most of the exploration feel.

#### RUNNING — the `VIEW_WEIGHT` ablation, and what comes after

Testing whether the function gap is the contrastive WEIGHT or the NEGATIVES.
`vw025` finished 40 epochs; `vw050` is mid-run; `vw100` is preserved.

**The early read is that the weight is not the lever.** A 4x change moved view
agreement 0.953 → 0.922 and left imputation untouched (`kw_flying` 0.956 →
0.954), with the two trajectories almost superimposed. Instance discrimination
is easy — telling one card from another needs few bits — so the model solves it
early at any weight.

If that holds, the objective is next, not the weight: **VICReg is built, tested
and ready** (`--objective vicreg`). InfoNCE makes every other card in the batch a
negative, so two cards that ramp the same way are pushed apart however the term
is weighted; VICReg has no negatives at all — invariance, variance (which does
the anti-collapse job), covariance. Applied to the SHIPPED latent rather than to
a discarded expander as the paper does, because decorrelating dimensions is
exactly what this space is worst at (16.72 of 128 against text's 51.39 of 384).

**Two traps NOT taken, both of which would have invalidated the measurement:**
using `ROLE_PATTERNS`/`MECHANICAL_TAGS` as a similarity label is the bootstrapped
supervision this rebuild exists to escape, arriving through the denominator
instead of the positives; and **EDHREC co-occurrence is what
`eval_embeddings.theme_groups` builds the theme eval FROM**, so training on it
would turn the +0.151 theme win into "the model learned its test set".

#### Artifacts and their gates

`data/span_vectors.npy` (gitignored, 115 MB), `data/embeddings_cardbert*.npy`,
`data/eval/recoverability.json`, `data/eval/space_projections.json`.
`--tag` keeps a sweep's runs apart: every artifact path was a fixed constant, so
two configurations run back to back would silently overwrite each other — the
`--out is slug-scoped` lesson in a new place, proved with a 1-epoch smoke run
before spending hours on the assumption. All three artifacts are written together
on each improving epoch and stamped with it, so an interrupted run is usable
rather than a trap.

### OPEN — what the speed sprint deliberately did not do

- **The five stale `diagnosis.json` files** — edgar-vampires, gishath,
  goblin-storm, heliod, yawgmoth-swarm. Their colour-source figure moved when
  fetchlands started resolving; two were already stale before that. Each needs a
  `/diagnose-deck` run, and prose is never hand-patched to green a gate. This is
  the only thing failing `make test` (plus one `deck_info` test downstream of
  heliod's).
- **Forge job count and work stealing.** `jobs` defaults to `cpu_count - 1` = 7
  on a machine with 4 performance cores, with a static split and no stealing —
  straggler tails of +4061s / +6734s / +1304s on the three biggest runs. Left
  alone because `run_id` does not encode `jobs`, so changing the default changes
  the SAMPLE a given run id produces.
- **`CODE = (SRC,)` over-invalidation.** An edit to any of ~130 files under
  `src/manamap/` invalidates the regenerate-and-compare cache for **236
  parametrized cases** across three files. The comment prices this at 20s; it is
  now 5-8x that. A per-subpackage key would keep the conservative property.
- **`card_value` and `candidates` sweeps are embarrassingly parallel** and still
  serial — each `_measure` builds a fresh generator, so a pool would be
  bit-identical. `candidates` runs one full 10k-game goldfish PER CANDIDATE.
- **Digest-based staleness.** Freshness tests still re-run the real producer;
  stamping an input digest would make them O(1) with one canary per artifact.
- **Agent fan-out.** The `open_questions` work queue dispatches `/resolve-stack`
  one at a time, `/write-manual` runs analyst then researcher serially, and
  `/diagnose-deck` runs recon then diagnose serially — all independent work.
  `resolve-stack/SKILL.md` already names this under "Scale-out note".
- **A resolved-path memo survives `monkeypatch` teardown** in `test_serve.py`, so
  a later test reading a real deck sees a tmp directory. Worked around by keeping
  `test_serve_cli.py` separate; the memo is the actual defect.
- **`stats.mde_proportion` overflows** above n ~ 1000 (exact binomial). A
  multiplicative PMF recurrence would fix it and is not bit-identical, so it was
  not done alongside the hoist.


### RULE — a branch is graded on what the deck PRODUCES, never on an authored file

**Set 2026-08-28, by measurement.** `net-change` carried an ENGINE LIFT: kill
rate in the games where every component marked `required` in
`goldfish_targets.json` was assembled by T3, minus the rate where it was not,
with a Newcombe interval on the difference. The statistics were right. The input
was not: **that file is authored, and the same hand writes the declaration and
reads the verdict.** Three defensible declarations of one Ur-Dragon list, same
10,000 games, same seed, against kill-by-T8 — **+0.007 (spans zero), −0.036
(REAL), +0.014 (REAL)**. One of them says at an interval excluding zero that
assembling the engine makes the deck win less.

**Deleted, with the guard in the same commit.** `deck_branch.MEMBERSHIP_AXES`
refuses `engine_online_*` and `any_route_*` as branch objectives and names the
output axes instead; a test walks every tracked `branch.json` so an old one
cannot survive it. `goldfish_targets.json` stays — it drives the `*_assisted`
figures and the target table, which are hypergeometric and real. It just does
not get to grade anything. Full record: `docs/gotchas-bench.md`.

Ur-Dragon's objective moved from `engine_online_5 >= 0.22` (met **4.4x over**
while the lift spanned zero) to **`damage_8 >= 40.0`** — the opponent's starting
life, a number with meaning outside this branch. v1.0.1 reads 30.81 and misses.

### DONE — the merge request: `propose`

**2026-08-28.** A branch is now `OPEN` / `PROPOSED · BLOCKED` / `READY` / `STALE`
/ `OUTRUN` / `MERGED`, all derived and none stored, so a proposal un-blocks
itself as cards land in a box. `deck-branch <slug> propose <name> --as v1.0.2`
freezes the decision (which list, which report, which grade) and hands over a
**pull list** split by what each bucket costs. `validate-branch` gates it —
`branch.json` was the last tracked pilot artifact without one.

Ur-Dragon's `eminence-v3` is proposed as **v1.0.2**, blocked on **6 cards**
(down from 12: four sat in decks that do not physically exist, and five more the
pilot had already agreed to proxy). Full record: `docs/gotchas-bench.md`.

### DONE — a granted mana ability belongs to whoever received it (2026-08-31)

**Found by the mana sweep for the encoder's `mana_repeatable` field, which is the
cross-pollination working: a change in `training/` audited a function in
`pilot/`.** `goldfish.produced_mana` counted every quoted ability as the card's
own — **145 corpus cards, 8 of them sleeved across five decks, five in kinnan**.
Leyline Immersion, an Aura, read as a five-mana rock.

**THE OBVIOUS FIX IS WRONG AND THE SWEEP IS WHAT SAYS SO.** Stripping quoted text
zeroes fifteen cards that are correct: Citanul Hierophants grants `{T}: Add {G}`
to "creatures you control" and IS a creature, as are Gemhide Sliver, Enduring
Vitality, Inga and Esika, Katilda, Sachi and seven more; Dryad Arbor, Jasconian
Isle and Gobland carry theirs in reminder text about themselves. The question is
not "is it quoted" but **is this card a member of the class it grants to** —
`produced_mana` takes `type_line` to answer it and defaults to reading every
grant as foreign, because overcounting tells the model it can cast things it
cannot.

Two bugs found while fixing it, both by the sweep: the backward window **crossed
a clause** (Sachi opens "OTHER Snake creatures…" then grants to "Shamans you
control", which she is), and `it has` was **too loose** (in Jiang Yanggu the "it"
is the recipient; in Llanowar Mentor and The Bus Runner it is a token created a
sentence earlier). **And one guard deleted**: a second, wider window written for
those four cards changed ZERO readings across all 34,890 — a bug probe caught
that it could not fail, and a guard that guards nothing is worse than none.

Sweep: 133 readings changed, 15 quoted grants kept as the card's own (each read
individually), 34,742 untouched. Corpus nonzero 1,975 → 1,848. Fleet regenerated
(72 targets, 92.7s); **gishath's commander cast-by-turn-6 drops 0.189 → 0.170**,
the honest direction once phantom mana stops counting.

### OPEN — five diagnoses are stale and need the doctor, not a patch

`edgar-vampires`, `gishath`, `goblin-storm`, `heliod` and `yawgmoth-swarm` fail
`validate-diagnosis` on `axes[].colour-sources.measured.value`: the `manabase`
correctness fixes moved the audit's figure underneath a diagnosis whose PROSE
names the old number ("Green 26, red 20, white 20 against a 36 target"; "Four
sources above the Karsten yardstick"). goblin-storm's reading even describes the
colour-identity fallback that no longer exists — the number moved *because* the
doctor's complaint was fixed.

**Patching `measured.value` is forbidden** — it would leave prose contradicting
its own figure, which is "a fresh claim under an old byline". These need a
`/diagnose-deck` re-run each, which is an agent spend and the pilot's call.
Until then `make test` is 5 red on exactly these five, plus one `deck_info`
test downstream of heliod's. **gishath's is now doubly stale** — the granted-mana
fix moved its figures again on 2026-08-31. **`diagnosis.json`
carries `as_of_decklist_sha256` but no AUDIT stamp**, which is why a code change
leaves it looking current — the same class `meta.model_version` solved for the
goldfish, unsolved here.

### RULE — a retired deck is not a downstream target

**Set by the pilot, 2026-08-27:** *"if a deck is deprecated, broken down,
exclude it from these downstream tasks."*

A `manabase` correctness fix moved the colour-source figure on six decks, and
three of them — `hapatra` and `radagast` (broken-down), `sisay` (retired, and
not the pilot's deck) — got an agent re-run each before the rule was stated.
That is real tokens spent regenerating a document about a deck nobody will play.

**A retired deck's artifacts are HISTORY, NOT CLAIMS.** Nothing derives from
them and nothing plays the list, so holding them to today's model is the "gate
that reddens history" `validate_prescription` already refused to be.
`deck_info.STATE_RETIRED` buckets broken-down, superseded and retired together
and is the one place that decides; `tests/test_pilot_tracked_artifacts_validate`
now skips a deck with a `lifecycle` block, and any fleet fan-out should do the
same.

**The related rule, same day:** every optimisation happens in a BRANCH. Nothing
touches a main `decklist.txt` — measurements of the existing list are not
optimisations, and regenerating them after a model change is required, but the
list itself moves only through `deck-branch merge`.

### ISSUE — `land_colors` credits mana it cannot actually make

**Opened 2026-08-27. Found by the `yawgmoth-swarm` doctor mid-run, which is the
loop working: an agent re-derived a figure and disbelieved it.**

Two stacked defects, both the class the reminder-text fix already closed once —
**text describing ANOTHER object's ability read as this card's own.**

1. **A quoted token ability counts as a mana source.** `Pawn of Ulamog` creates
   an Eldrazi Spawn with `"Sacrifice this token: Add {C}."` —
   `nonland_producer_kind` matches the quoted clause and calls it `ramp:dork`.
   It is also one-shot and self-sacrificing, which is the Jeweled Lotus rule
   `goldfish._CONSUMING_COST` already enforces and `manabase` does not.
2. **The colour then comes from COLOUR IDENTITY, not from the ability.**
   `land_colors` ends `if not produced and not restricted: produced.update(
   color_identity)`. `{C}` is not a coloured symbol, so nothing parses and the
   fallback credits Pawn of Ulamog with **B — because it is a black card.**

**AND THE FALLBACK IS THE BIG ONE — it is systematically wrong for LANDS.**
Measured across the corpus: **60 land entries** get their colours only from that
fallback, and the pattern is always the same — the land taps for `{C}` and its
colour identity comes from an ACTIVATION COST it has nothing to do with
producing. `Goblin Burrows` is `{T}: Add {C}` plus `{1}{R}, {T}: pump a Goblin`,
and reads as a **red source**. So does `Kher Keep`. `Kor Haven` reads white off
`{1}{W}, {T}: prevent damage`; `Blighted Woodland` reads green off a `{3}{G}`
sacrifice. None of them makes a single coloured mana. Two doctors found this
independently on two different decks in the same batch, which is the corroboration
— goblin-storm's true repeatable red land count is **31, not 33**.

Whether the fallback has any legitimate case is the open question: it exists for
a land whose production text does not parse, and nobody has enumerated those. The
fix must not be "delete the fallback" until that set is known, which is the same
discipline that kept the quoted-text class from being a blanket strip.

**Scope, measured:** 8 further cards read mana colours only from text inside
quotation marks. They are NOT uniformly wrong — `Worldknit` and `Paradise
Mantle` grant `{T}: Add one mana of any color` to permanents you control, which
is real fixing. The wrong ones are token abilities. So the fix is not "strip
quotes"; it is to separate *granting an ability to something you control* from
*describing a token you may create*.

**Not fixed on discovery, on purpose.** Six `deck-doctor` runs were in flight
against the current model. Changing `land_colors` mid-batch would have made
every one of them stale on arrival and wasted the spend. A model change and a
regeneration of the artifacts that depend on it belong in one commit — which is
the same rule that made this batch necessary in the first place.

### ISSUE — unit tests must not depend on an experimental deck

**Opened 2026-08-27, by the pilot, and it is a rule rather than a chore.**

> *"we shouldn't be writing tests for decks that are experimental/feature
> branches — we can have tests to assert the functions that support that work,
> but we don't need to test the actual deck via unit tests. Stats, math, etc. is
> what we will apply to those decks. Not until they are merged and pinned do
> they get tests."*

**What happened.** Twenty tests across nine files hardcoded
`ur-dragon/treasure-v2` as a data fixture — `close`'s component search, `assess`'s
triage, the diagnostic's magnitude series, the branch-scoping controls, the
net-change report. The treasure refactor was measured, found worse, and deleted,
which is the tool working exactly as designed. All twenty failed. Deleting a
branch is a first-class pilot action and the suite punished it.

**The rule.** A branch is a candidate 99 that is *supposed* to change and
*supposed* to be thrown away. Only a **merged and pinned** deck is stable enough
to assert against. What a test may assert about branch machinery is the
FUNCTION — that a branched write lands in the branch directory, that reads fall
back and writes never do, that a swap is one-for-one — and those need a
**synthetic branch in a `tmp_path` sandbox**, which
`tests/test_pilot_branch_lifecycle.py::sandbox` already demonstrates.

**What is done and what is not.** `conftest.requires_branch` / `A_BRANCH` now
take *whichever* branch exists rather than one name, so the suite no longer
names a specific experiment — that is a stopgap, not the fix. The fix is to
rebuild these tests on synthetic fixtures and delete the assertions that were
only ever about the treasure deck's contents (the `MULTIPLIER` component, the
treasure cards `assess` was triaging). Until then they are gated and some of
them will skip.

**Related invariant** (below): a tracked artifact needs a gate in the same
commit. That is still true — but the gate belongs on the artifact's SHAPE and
its producing function, never on one experimental deck's numbers.

### THE ORDER OF TASKS — set by the pilot, 2026-09-01

**Modelling first, agents after.** Six logged games are un-debriefed and stay that
way until the embedding work clears. This is a deliberate sequencing decision, not
a backlog: `/debrief` is an agent spend against a model that is actively moving,
and a debrief written now would be annotating figures that the next commit
changes.

1. **Finish the `VIEW_WEIGHT` ablation** — `vw025` done, `vw050` running, `vw100`
   preserved. Report the three-way table.
2. **`project-spaces` on all five spaces** and LOOK at them — visual inspection is
   part of evaluation, and a space can win recall@10 by concentrating while losing
   everything that makes an atlas navigable.
3. **`train-cardbert --objective vicreg`** — built, tested, waiting on 1.
4. **Then decide what, if anything, cuts over.** The honest current answer is that
   CardBERT is a THEME-and-CENTROID space complementary to the function space, not
   a replacement for it.
5. **`/debrief` the six un-debriefed entries** — edgar 002/003, ur-dragon 002,
   goblin-storm 001, heliod 001, gishath 001. One batch, five decks.
6. **Edgar's direction change**, which the 08-28 log records and which supersedes
   the token-conversion axis: swarm typal → LORDS AND PAYOFFS, tokens demoted from
   win condition to dig engine, a mill package as a second route through the deck,
   W/B lifegain into lifedrain. This is a branch, never a `decklist.txt` edit.
7. **Heliod: protection first.** Losing the commander takes the engine with it and
   there is no answer today to a counter war plus redirected removal.

### Next, in order (the standing backlog)

1. **Paper check-ins, one deck at a time** — **only the pilot can do this**, and it is
   the highest-value thing available. **heliod and gishath lists are promised** and hold
   v1.0.0 placeholders until they land; three more (yawgmoth-swarm, kianne, kinnan) are
   still **not marked as built in paper**, so nothing knows whether they exist as
   cardboard. **ur-dragon v1.0.2 paper is on the way**; v1.0.1 is what is being played. `check-in`
   takes a typed list and refuses rather than guesses, then `deck-version paper` locks
   it and drift is computed on every swap from then on. Two decks (edgar, ur-dragon)
   have one logged game each — which is two, not a sample.
2. **The versions deploy-time step** — a Pages workflow checking out with
   `fetch-depth: 0` and running `deck-version list --json` per deck into `versions.json`.
   The producer already exists and the deck page's panel is already written; it renders
   nothing until the artifact does. Needs the Pages source flipped from "branch" to
   "GitHub Actions" in repo settings, which only the maintainer can do.
3. **An agent in the pilot's seat** for a handful of seeded games on one question — the
   evidence for it is in: no Forge AI profile flies a hold-up deck better than Default,
   so the AI is the thing limiting the measurement rather than the configuration.
4. **The version-bump classifier** — `deck-version bump` measuring a diff and PROPOSING
   major/minor/patch with its evidence, for the pilot to confirm. The policy is written
   (`docs/pilot.md`); the classifier needs three things that do not exist: a diff between
   two *arbitrary* versions (every diff today is consecutive or against the working tree),
   `quantity_changes` carried into `versions()` — `history()` computes it and `versions()`
   drops it, so 36 to 37 Forests reads as no change at all — and a classifier reporting
   **evidence, never intent**, since `deck_history` is explicit that *why* a card moved is
   not knowable from a commit.
5. **Content-addressed cache busting for `deck.html`** — it went from nine artifact
   fetches to fifteen, and `manuals/magazine.css`'s `?v=<sha8>` is the pattern to copy.

**Done since the last revision:** the Pilot's Manual (`build-page`) and the magazine
unlinked from every live surface; edgar and ur-dragon pinned at v1.0.0 with a real game
each, debriefed and prescribed; the workbench landing page and its fleet table;
verified-line prose, engine arrows and clickable group bars in Build; seeding a walk from
named cards and `?cards=`; the semver policy and its three tag guards; the paper lock's
third state and the withdrawal of three rehearsal locks (2026-08-23 → 25). Before that:
`experiment` and AI profiles (2026-08-19); `card-search`,
commander damage per defender, the collection primitive, `validate-recon`, deck lifecycle
in `deck-info` (2026-08-21); the builder's curve quota and combo completion, `mean_ci`'s
distribution, `deck-info --write`, the manifest's instanced files, the deck page, the DFC
pip fix, and `deck-status`'s gate blind spot (2026-08-22).

### Known gaps, named in the artifacts

- **Forge's AI pilots the deck** — "poor to ok in control, pretty bad for combo" (its own
  words, quoted in every run record). radagast 0/20 vs the pod at 12.5 combat damage a
  game against 45.6 among its stablemates. A lower bound on the pilot; a true picture of
  the table's clock.
- **Bridge approximations**: token types unknown from the log (tokens filed as
  `other_permanents` by `scenario-facts`), hand sizes are estimates, continuous effects
  (a Craterhoof pump) are not tracked and must be authored into the scenario. Every one is
  written into `extras.reconstruction_notes`.
- **Parser**: damage figures see damage only; drain kills show in `life_by_turn` and
  `eliminated_how`, never in a damage total. **Commander damage is now measured**
  (2026-08-21) — per DEFENDER, because CR 903.10a asks for 21 from one commander on one
  player and `combat_damage_dealt_to_players` sums every source and every seat at once.
  The commander names ride IN the record (`seats[].commander`), never looked up from
  disk at validate time; a record without the field re-derives exactly as before, and
  `simulate <slug> --analyze <run>` is the migration.
- ~~**The deterministic builder cannot produce a curve SHAPE.**~~ **FIXED 2026-08-22.**
  It scored every card independently and took the top N while `curve_fit` penalised
  each point above `DECK_CURVE_SWEET_SPOT = 3`, so the top N were always cheap: the
  first kinnan baseline was 64 nonland cards with curve `{0:1, 1:11, 2:28, 3:24}`,
  **nothing above mana value 3**, and 29 of them mana producers — a legal deck that
  ramps into nothing, which `validate_build` passed because it checks form. The role
  quota in `fill_slots` is now crossed with a mana-value quota derived from
  `DECK_AXIS_TARGETS["curve"]`, the target `deck_audit` already measured against and
  the builder never read, so no new uncited constant. Rebuilt: `{0:1, 1:9, 2:15,
  3:16, 4:10, 5:6, 6:4, 7:1, 8:1}`. Combo blindness went with it — `complete_combos`
  reads real lines from `combo_details` (not the flat `combo_partners` map, which
  cannot tell a completion from a coincidence) and swaps in the one missing card of a
  line the deck half-holds. kinnan went from 23 partners and 0 completions to **4
  contained combos and 2 two-card infinites**, including Kinnan + Pili-Pala +
  Enduring Vitality. And `build()` now has end-to-end tests: there were none.
- ~~**DFC pips**~~ — **FIXED 2026-08-22**, and it was two defects rather than one.
  `pip_requirements` read `card["mana_cost"]`, which Scryfall leaves EMPTY on
  transform/MDFC layouts (counting zero pips) and which holds BOTH halves on
  adventure/split layouts (counting double). `common.front_field` is the one shared
  reader now, replacing `deck_facts._front`, which had solved this for colours and never
  for pips. It produced a real finding: **heliod's commander is `{2}{W}{W}` and that
  second pip was invisible**, so its white target read 22 when it should read 36 — short
  by 16 against 20 sources, not by 5, with white rather than blue the binding colour.
  Seven `mana_analysis.json` regenerated. Same-class defect still unfixed in
  `build_deck.castability` (`build_deck.py`, `getattr(row, "mana_cost", "")`).
- **hapatra's `bracket_report.json`** contradicts a verified stack: two of its three
  `drivers` cite lines that stack 001 refuted or explicitly declined to rest on, and the
  "19 two-card infinites" figure is inflated by six. The contradiction is duplicated
  verbatim into `build_plan.json`. Blocked on a schema question — the refutation is prose
  inside `resolution.final_state.summary`, and there is no machine-readable `refutes`
  field for a gate to read.
- **`build_plan.json` is not reproducible** from today's data, and is now *further* from
  it: the builder gained a mana-value quota and `complete_combos` on 2026-08-22, so
  re-running produces a different 99 by design. Open question whether historical build
  plans should be reproducible at all — they are records of a build that happened, like
  `log.jsonl`.
- **A diagnosis can go stale without its decklist moving.** The DFC fix changed the audit
  underneath heliod's `diagnosis.json`, which had cited the old figures correctly when
  written. `validate_diagnosis` re-derives every axis, so it failed — right, but there is
  no staleness *class* for "the measurement code moved", the way there is for "an older
  decklist". The only route is a re-spawn.

### Verification backlog (✓ work)

- **Sisay 001** (the tutor chain) — highest-value promotion from ★ to ✓ in the fleet.
- **Grafdigger's Cage** — three of yawgmoth's kills rest on an oracle reading no checker has settled.
- **Hapatra's Mikaeus +1/+1 anthem** vs its token loops — if it switches them off, its two engines are mutually exclusive.
- **Radagast's `open_questions`** — six from the engineer; only Craterhoof has a stack. **Boards for these can now be lifted from sims** (`sim-scenario`).
- Queued: Roaming Throne × Zada; the Past in Flames rebuild; sisay's other Najeela pairs.

### Still owed

Sized 2026-08-22; each is small and independent unless noted.

**QUEUED 2026-08-24, both ready to run, both precisely specified.** They came out of
ur-dragon's terminal rounds and were deliberately NOT acted on there: a change made after
the last adversary has finished is unreviewed by construction.

- **`/resolve-stack ur-dragon 007`** — the scenario is written, preflighted
  (`validate-stack --scenario-only`: OK) and committed as
  `stacks/queued/007-cascade-without-panharmonicon.json` — in `queued/` rather than
  `stacks/`, because `validate-stack`'s glob is non-recursive and the citation-contract
  test requires every tracked stack to carry a PASSING resolution, so a staged scenario in
  `stacks/` turns the suite red. **Move it up one directory to run it.** That tension is
  real and worth noticing: `--scenario-only` exists to preflight in place, and a preflighted
  scenario cannot then be committed where it was preflighted. It is stack **002's exact board minus
  Panharmonicon**, derived from 002 rather than invented, and answers the three things
  `diagnosis.json`'s `open_questions[0]` asks: how many token copies and how much damage
  without the doubler; whether it is still lethal to the 32-life seat 002 kills with 64;
  and — the deck's central question — whether ANY version kills a fresh seat at **40**,
  which neither proven board has been shown to do (002 leaves the 40-life seat alive at 8).
  It prices `cut_candidates[5]`, which currently rests on a mana argument alone with its
  decisive evidence *named and absent*: nothing on the record says whether cutting
  Panharmonicon costs 002's **lethality** or only its **margin**. One rules domain, no
  combat; 002's nontoken finding is inherited and stated in `extras.context`.

- **Scourge of the Throne in THE COMBAT KILL's multiplier leg** — `engine-critic`'s
  terminal round found the two-card leg (Atarka, Thrakkus) arguably omits it: an additional
  combat phase doubles the turn's Dragon combat damage, and `engine.json` itself calls it
  that kill's mechanism, which is an internal contradiction. It differs from the other two
  in being **conditional** (it must attack the player with the most life), which is why it
  is a judgement rather than an obvious omission. Recorded in `goldfish_targets.json`'s
  note. **Run it with a critic attached** — this declaration moved three times on
  2026-08-24 and every move cascaded into `engine.json` and `diagnosis.json`.

- **`diagnosis.json` for ur-dragon carries a terminal `fail`** and is NOT cache-recorded,
  per the rule that a fail is never recorded. Two text-level defects, both discharged,
  neither moving a swap. It clears on the next diagnose pass, which the two items above
  should precede — both change figures the diagnosis quotes.

- **Six of nine `strategic_frame.json` are unstamped** and now say so in `deck-status`
  ("unstamped — staleness cannot be checked"). That is a third state, not a softer STALE:
  they may be current and simply not say so. Each is one `strategy-researcher` MODE consult
  to stamp, and worth doing opportunistically rather than as a sweep — ur-dragon's turned
  out to be asserting proof that had left the deck.

- **`merge_deck_map` / `engine_facts` have ZERO test coverage** — no test file imports
  either. ~300 lines, blocked on nothing. `merge_deck_map`'s whole reason to exist is the
  `OWNED = ("label", "gloss")` whitelist, and nothing asserts it.
- **No regression floor on the balance bound.** Nothing reads the nine tracked
  `deck_map.json`. Note edgar-vampires sits at **35.05%** against a 35% bound, so the
  floor must encode the `MAX_CITIES` escape the synthetic test already uses.
- **`deck-recon` on four LIVE decks** — gishath, goblin-storm, heliod, ur-dragon. It is
  *absence*, not staleness: nothing is stale by `RECON_MAX_AGE_DAYS` (oldest 19 days
  against 120). hapatra and sisay are dead cardboard and should be skipped — a perishable
  meta artifact for a deck nobody can shuffle is ~100k spent on nothing.
- **`deck-history pending` should read prescriptions' adds.** Blocked twice over: "open"
  is the wrong predicate (an open prescription has *no* adds; the wanted state is
  answered-but-unapplied), and **zero prescription files exist fleet-wide**.
- **`supersedes`** — no scaffolding at all; the status half exists (`DECK_STATUSES`) and
  the pointer does not. Blocked on a decision: it lives in the frozen magazine layer.
- **Strategy-DB gaps** — 49 across the frames; aristocrats/sacrifice first.
- **Ur-Dragon two-engine rebuild** — proposed, measured, not applied.
- **`build_deck.castability` reads `getattr(row, "mana_cost", "")`** — the same defect the
  DFC fix just closed in `manabase`, still open one module over.

### Legacy, frozen — and what unfreezes it

The magazine renderer (`build_manual`, `issue_spec`, `design`, `validate_issue`, STYLEv3)
still renders the nine pages from artifacts nothing regenerates (`issue_plan.json`, the
panel keys, `card_roles`/`mana_base`/`upgrades`, `considering.json`). They are edited by
nobody and deleted in one commit when the compact page lands (`docs/manual-v5-spec.md`
§"What gets unfrozen"). Code and docs about it carry a LEGACY banner and are otherwise
left accurate.

## Decisions that bind

### The frontend stays LLM-free
The deployed site and the local checkout run the same code. The agent loop stays in
Claude Code, reached by commands and a brief. No local bridge.

### Forge is the engine; we build the harness, the parser and the bridge
Measured before chosen: (a) every log line parses, (b) 4-seat Commander runs, (c) `-s`
makes it byte-replayable. Writing our own rules engine is shelved for one narrow
deterministic case. The goldfish stays as the seeded resource model.

### The cache board is red fleet-wide, and deliberately not re-recorded
The shared-contract commit MISSed every routine; charter edits disqualify STALE_OK by
construction. Artifacts are gated by validators and tests, not by the cache; each routine
clears on its next real spawn. Never `cache-record` to make a board green.

### Versions are derived from git and never committed; tags are authored
A commit's sha is unknown inside the commit, so a generated version list would be one
behind forever. `deck_versions.json` (tags) is the one version datum a browser can read.

### A prescription accumulates; stale is not wrong
One file per question, keyed by the prompt's hash; an older-decklist prescription is
form-checked only.

### Similarity comes from the function space, always
`embeddings_ability.npy` is the sole source of similarity (Find Similar, the walk, drill,
the deck map). `embeddings.npy` feeds `projection_2d.json` only. Do not tune on the golden set.

### Synergy is complementary, not similar
24 rules over mechanical tags, ranked by playability. "Anti-cards" do not exist.

### The clusters are an input to engine analysis, never the analysis
A city name is the wrong address for a component; a disagreement between the map and the
engine is a finding.

## Invariants that must not erode

- Only checker-passed stacks publish; failed artifacts are kept as open questions.
- Agents return JSON and never write HTML.
- `issue.json`, `log.jsonl`, `deck_versions.json`, a prescription's prompt: **authored**,
  never regenerated. A derived artifact may be regenerated; an authored one may not be rewritten.
- Costume never earns the badge.
- Record the cache **after** validation, never before; never record a `fail`.
- Charter edits invalidate before they inform — make them **before** `cache-record`.
- A bracket **floor** is what the contents are consistent with, never a verdict.
- The deterministic builder must always produce a complete legal 99 with no agent involved.
- **Count copies, not decklist entries**, for anything the shuffler would see.
- `--out` on a per-deck command must be slug-scoped.
- **A validator that fires on correct data is worse than none** — measure a proposed check
  against the whole fleet before keeping it.
- **A critic's findings become mechanical checks**, or its work is re-spent every run.
- **Name what a gate cannot see** rather than papering it with string matching.
- A sim figure travels with its interval, its N and its limits — or it does not travel.
- **A measure computed from an authored file is not evidence, however tight its
  interval.** A branch objective names an OUTPUT the deck produces.
- **Every figure carries its definition in the report that prints it** — a number a
  reader has to look up gets guessed at, and the guesses go one way.
- **When you delete a producer, grep the validator** for the only place its contract
  was enforced. Removing the engine lift silently took "an unavailable block owes a
  reason" with it.
- **A model's embedding must be in its own objective.** `[CLS]` received no
  gradient for a whole 40-epoch run and shipped as a random projection; nothing
  failed, because every other metric was measuring the imputation heads.
- **A guard that cannot fail is a claim, not a guard.** A bug probe that MISSES
  is a finding: either the test is vacuous or the code is dead. Both have
  happened, one commit apart.
- **Never train on what the eval measures.** EDHREC co-occurrence is genuinely
  external supervision AND is what `theme_groups` builds the theme eval from.
- **Visual inspection is part of evaluation.** A space can win recall@10 by
  concentrating and lose everything that makes an atlas navigable.
