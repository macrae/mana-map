# PLAN — current state and what's next

*The resume-here doc. Read `README.md` for orientation, `CLAUDE.md` for gotchas, this for
what shipped and what's still open. Completed plans live in `docs/history/`.*

Last updated 2026-07-30. All work below is committed, pushed, and deployed.
Every figure here was derived from the repo at write time, not remembered.

## What this is now

Two products in one repo, sharing a data layer and a CLI.

**The card tool** was rebuilt this cycle and is no longer "a scatter plot": it opens on a
single card and grows a graph as you click (see *Shipped — discovery-first* below). Under it,
the function embedding was retrained after being measured as collapsed, and the map itself
moved off Plotly onto canvas.

**Pilot's Manual** turns one Commander deck into a self-contained web issue with a three-tier
evidence contract — and, since Deck Building v2, a **deck builder** produces the deck in the
first place.

**Seven issues live**, all rebuilt 2026-07-29 under the v3.3 process:
[001 Goblin Storm](https://macrae.github.io/mana-map/manuals/goblin-storm.html) ·
[002 Hapatra](https://macrae.github.io/mana-map/manuals/hapatra.html) ·
[003 Sisay](https://macrae.github.io/mana-map/manuals/sisay.html) ·
[004 Heliod](https://macrae.github.io/mana-map/manuals/heliod.html) ·
[005 Ur-Dragon](https://macrae.github.io/mana-map/manuals/ur-dragon.html) ·
[006 Edgar](https://macrae.github.io/mana-map/manuals/edgar-vampires.html) ·
[007 Gishath](https://macrae.github.io/mana-map/manuals/gishath.html) ·
[newsstand](https://macrae.github.io/mana-map/manuals/index.html)

1,119 tests (1,042 fast + 77 browser). 33 `manamap pilot` subcommands. 12 agents, 15 skills.

> ### ⚠ OPEN: 23 agent-cache routines are deliberately MISSed
>
> The embedding rebuild regenerated `synergy_graph.json` and `obsolescence_index.json`, which
> are declared inputs of `writer-prose`, `the-ten` and `issue-plan` — so **every deck's copy of
> those three is stale**, plus `candidate-pool` and `deck-build` on hapatra.
>
> | decks | routines MISSed |
> |---|---|
> | edgar-vampires, gishath, goblin-storm, heliod, sisay, ur-dragon | `writer-prose`, `the-ten`, `issue-plan` |
> | hapatra | those three **plus** `candidate-pool`, `deck-build` |
>
> `strategic-frame`, `coach-prose`, `tutor-guide` and every `stack:NNN` still HIT — the rules
> and strategy inputs did not move.
>
> **This was left MISSed on purpose.** Re-spawning all 23 costs roughly **2.46M tokens**
> (~1.74M for the six decks, ~725k for hapatra). The published issues are not wrong: the graphs
> changed underneath the prose, but the prose's claims were validated when written. The choice
> is deliberate and belongs to a human:
>
> - **re-bless** — `manamap pilot cache-record <slug> --routine <name>` per routine, after
>   reading the prose and agreeing it still holds. Free. This is what `docs/pipeline.md`
>   recommends for a graph refresh.
> - **re-spawn** — regenerate for real, at the cost above. Warranted if the new synergy and
>   obsolescence data would actually change what the sections say.
>
> Do not `cache-record` blindly to make the status board green: the record is the claim that
> someone checked. Until this is resolved, expect `cache-status` to report MISS on any deck.


## Shipped

| Layer | What |
|---|---|
| **Rules DB** | CR chunked one-per-rule (~3.9K), chunk ID = citation ID; semantic query + exact lookup |
| **Citation contract** | Form enforced in `validate_stack.py`; meaning by adversarial `rules-checker`; only `pass` publishes |
| **Goldfish** | Seeded Monte Carlo (seed 42, 10K iters), resource development *not* full games, assumptions rendered |
| **Strategy DB** | 45 sourced strategy sections / 14 pillars; `strategy:<id>` = citation ID; `strategy-researcher` (research + consult modes) |
| **Magazine layer** | **17 sections in five acts**, `issue_spec.py` as the single source of truth, `magazine-editor`, deterministic renderer, newsstand |
| **The three columnists** | Every section is signed. `"Ledger" Lin Marginal` (◆), `Counselor Vera Dictum` (✓), `Coach Sunny Brightside` (★) — `MASTHEAD_COLUMNISTS`, per-section `byline`, promises written in the signing voice. Personas are presentation only; costume never earns the badge (STYLEv3 §7.7, §10) |
| **L10 — every issue is the reader's first** | No version numbers, no HISTORY.md, no "previous build", no swap-wave narration in print. Enforced in code: `validate_issue.validate_self_containment()` lints the plan, prose and every decision file |
| **Agent cache** | Fingerprints declared inputs per routine into a tracked `.agent-cache.json`; check → spawn → validate → record |
| **Incremental regeneration** | Card-scoped cache: per-card digest map + conservative `card_refs`; `STALE_OK`/`cache-rebless`; `manamap pilot impact` reports reference/figure/target/zone staleness; per-key fingerprints (`PROSE_KEY_INPUTS`) + charter "Partial revision mode" scope spawns to stale keys |
| **Exact printings** | Moxfield `(SET) COLLECTOR *F*` resolved first, so the manual shows the pilot's physical cards |
| **Featured Artist** | `artist_credits.py` auto-detects standout artists, clusters, drop runs; counts per card, never per copy |
| **Role taxonomy** | `analysis/card_roles.py` (step 13) → `card_roles.json`; 43 roles, *separate* from `MECHANICAL_TAGS` |
| **Bracket engine** | `pilot/bracket.py` computes a **floor** from Game Changers, combo tags, two-card infinites and mass land denial; names the driving card or line; `validate-build` cross-checks the plan's claim |
| **Deterministic builder** | `brief.json` → `build_plan.json` → `decklist.txt` with no agent involvement; `manabase.py` sizes colour sources hypergeometrically |
| **Deck facts** | `deck-facts` — the deterministic brief every agent reads instead of re-deriving. Computed on demand, never committed |
| **Fetch Quests** | `tutor-guide` routine → `tutor_guide.json` (all 7 decks), authored by `pilot-coach`, gated by `validate-tutor-guide`. One wish per maindeck tutor; per-clause search-constraint checking (a DFC's two faces carry two different clauses) |
| **Sources Say** | `mana_analysis.py` + `manamap pilot mana-analysis` → `mana_analysis.json` (all 7 decks). Land classes, pips vs sources, hypergeometric on-curve probability, ramp census. No agent — deterministic Python, narrated by the writer's `mana_base` key |
| **The Short List** | `considering.json` (all 7 decks) — exactly ten cards, bench-first, pool-filled, `validate-considering` enforcing the count and every evidence claim. Replaces both `sideboard_analysis.json` and `upgrade_watch.json` |
| **Card links** | Every card mention in body copy links to its tile in The 99 (commander → The Command Zone) with a CSS-only hover preview. Renderer-provided navigation; agents write plain names |
| **Deck versioning** | `HISTORY.md` per deck (append-only: date · sha12 · floor · reason) on 4 decks; `decklist_sha256` stamped in `issue.json` **and asserted against `cards.json`** by `validate_issue.validate_identity()` |
| **Deck dossier** | `viz/deck.html` — every deck's committed artifacts rendered as data: bracket floor + named driver, Sources Say, goldfish, the Short List, Fetch Quests, verified case files with citations, and the builder's record where one exists. Nothing recomputed in the browser, nothing hardcoded; `build-index` emits `data/decks/index.json` as the manifest. `viz/css/tokens.css` ports design.py's tokens to a dark register with the ✓/◆/★ colours fixed. Each issue's Back Page links to its dossier and back |
| **Loop economics** | Scenario preflight (`validate-stack --scenario-only`); `RESOLVE_SCOPE_BUDGET`; `RESOLVE_MAX_ITERATIONS` enforced in `cache-record`; agents hand off by path via `.agent-out/` |

## The 17 sections, in five acts

`issue_spec.DEPARTMENTS` is the only authority — never transcribe the list into a prompt
(`tests/test_docs_section_count.py` enforces this).

| Act | Sections |
|---|---|
| — | The Cover · The Flight Plan |
| **In the Cockpit** | The Game Plan · Keep or Ship · What's Your Play? |
| **At the Table** | Table Manners · Know Your Enemy · Fetch Quests |
| **The Long Game** | The Command Zone · The 99 · The Short List |
| **Show Your Work** | Sources Say · *(declared art break)* · By the Numbers · The Kill |
| **The Appendix** | Judge's Desk · Featured Artist · The Back Page |

Depth rises monotonically: what to do, then tactics, then the zoomed-out game, then the
numbers, then the proof. `BREATHER_AFTER` declares the one place two dense sections sit
adjacent, and the renderer emits a full-bleed art break between them.

## The agent roster (12)

All read-only except where noted. Definitions in `.claude/agents/`.

| Agent | Role |
|---|---|
| `stack-resolver` | Cite-or-decline stack resolutions |
| `rules-checker` | Adversarial citation verification + missing-step audit |
| `manual-writer` | Body prose (7 keys incl. `mana_base`), per-key persona voice |
| `pilot-coach` | Threat/matchups/decisions (★) **and the tutor guide** |
| `magazine-editor` | The issue plan: cover, sections, headlines, furniture |
| `strategy-researcher` | Strategy doc research (**write-scoped to `data/strategy/`**) + consulting |
| `deck-analyst` | ◆ data layer; emits `candidate_pool.json` |
| `deck-architect` | Improves the deterministic plan; every ratio cites `strategy:<id>` |
| `deck-critic` | Adversarial verifier for build plans; report-only |
| `sideboard-analyst` | **The Short List** — ten cards, bench-first, pool-filled, analysis-only |
| `pipeline-runner` | Runs and diagnoses card-pipeline steps |
| `viz-dev` | Frontend work (write-scoped to `viz/`) |

## The pipelines

```
Build     brief.json → build-deck → validate-build → bracket-check
                     → /build-deck   (deck-analyst → deck-architect ⇄ deck-critic, ≤3 iters)
                     → decklist.txt

Publish   fetch-deck → validate-deck → goldfish → mana-analysis
                     → /resolve-stack per line   (resolver → validate-stack → rules-checker, ≤3)
                     → /write-manual             (frame → coach → writer)
                     → /analyse-sideboard        (the Short List)
                     → /design-issue             (magazine-editor → validate-issue)
                     → build-manual → build-index
```

Cache-gated across **8 static routines** — `candidate-pool`, `deck-build`,
`strategic-frame`, `coach-prose`, `writer-prose`, `the-ten`, `tutor-guide`, `issue-plan` —
plus the dynamic `stack:NNN` / `decision:NNN` families. `cache-status` before spawning,
`cache-record` after validating. Exit 0 = don't spawn, 1 = spawn, 2 = fix the input first;
`tutor-guide` reports `N/A` for a deck with no library-search tutors.

## Decks

All seven published. Floor 4 across the board.

| Deck | Vol | Stacks | Dec | GC | Ten | Tutors | Lands | State |
|---|---|---|---|---|---|---|---|---|
| `goblin-storm` | 001 | 5/5 | 2 | 0 | 1b/9p | 2 | 36 | Hand-built. Verified true infinite (Haze of Rage + Storm-Kiln); stack 004 **refutes** the combo-graph Krenko infinite |
| `hapatra` | 002 | 1/1 | 0 | 11 | 0b/10p | 8 | 36 | **Built by the v2 loop** — the only deck with `brief.json` + `build_plan.json` + `candidate_pool.json`. Stack 001 showed the engine's 19 two-card infinites is inflated. The fleet's cleanest mana: zero tapped, 96.2% on curve both colours |
| `sisay` | 003 | **1/3** | 2 | 4 | 0b/10p | 1 | 40 | The first issue to publish with `fail` artifacts on the record. Stack 002 **corrected the bracket engine in print** (Najeela + Derevi is a real loop but not two cards) |
| `heliod` | 004 | 6/6 | 2 | 10 | 7b/3p | 3 | 33 | **v5** — five HISTORY entries. Verified Aetherflux table-kill (stack 006). v5 is the mana-base correction: blue 21 → 26 sources, on-curve 61% → 74.2% |
| `ur-dragon` | 005 | 6/6 | 2 | 3 | 0b/10p | 4 | 34 | v4. Includes a refutation (Sneak Attack ≠ commander-cheat) and the Throne + Bloodletter 14-per-opponent bound |
| `edgar-vampires` | 006 | 8/8 | 2 | 4 | 8b/2p | 3 | 37 | "THE KILL MATRIX" — a fully verified **2×2 loop matrix** (Vito/Bond × Exquisite/Conqueror) plus a partial refutation. Most verified lines in the fleet |
| `gishath` | 007 | 5/5 | 2 | 2 | 4b/6p | 5 | 35 | v3. The headline is a refutation: Marauding Raptor + Polyraptor is a whole-table **draw** machine. Floor conversation on the cover — PRESENT 4, ARGUE 3 |

Every deck carries: `cards.json`, `bracket_report.json`, `goldfish_metrics.json`,
`goldfish_targets.json`, `mana_analysis.json`, `considering.json`, `tutor_guide.json`,
`manual_prose.json`, `issue_plan.json`, `issue.json`, `strategic_frame.json`, `stacks/`,
`decklist.txt`, `.agent-cache.json`. Only hapatra has `build_plan.json`; only hapatra
lacks `decisions/`.

## The copies-vs-entries land bug (2026-07-29)

Worth keeping because of how it was found and what it changed. `cards.json` stores basics
as one entry with `quantity: N`, and both `mana_analysis.py` and `deck_facts.mana_facts()`
counted **entries**. Heliod published "18 lands" for a 33-land deck; goblin-storm published
15 for 36. Every per-colour source count and every hypergeometric on-curve probability was
understated fleet-wide. A second bug fell out of the same audit: `upgrade_facts`'s role
budget compared entry counts against a copies-based 36-land budget, briefing the pool
scouts that decks were ~20 lands short.

Caught by the pilot reading his own manual. Fixed at the root (`common.expand_copies()`),
repaired across all seven issues, and guarded three ways: a fixture pinning the shape
(11 Islands = 11 blue sources), a staleness test recomputing every tracked
`mana_analysis.json`, and a `validate-issue` lint rejecting any reader-facing copy that
quotes `lands.entries` as a land count. The lint found two survivors the greps had missed.

`goldfish.py` was already correct (it expands `quantity` when building the library), so no
simulation figure was ever wrong.

## Ongoing — what is in flight right now

Nothing is mid-edit: the working tree is clean, all seven issues validate, every deck's
cache exits 0, and the fleet rebuilds byte-identical. "Ongoing" here means the threads
that are live and have an obvious next move, not work left half-done.

| Thread | State | Next move |
|---|---|---|
| **Frontend v2** | The dossier (`viz/deck.html`) and the **Deck Lens** (map mode 3) shipped; the three surfaces now form a link cycle. The engine port has not started. | `viz_index.json` — see *Future* below |
| **Deck versioning** | `HISTORY.md` + a validated `decklist_sha256` shipped; `supersedes` and `build:<NNN>` did not | Add `supersedes` to `issue.json` so a second issue for one deck is expressible |
| **Verification backlog** | 32 verified lines across 7 decks; ~30 named candidate lines unresolved, several one clause from passing | Sisay 001 — highest value in the fleet |
| **Strategy DB** | 45 strategy sections / 14 pillars; four named gaps, one of them now backing a whole section with no pillar behind it | A tutor-sequencing pillar for Fetch Quests |
| **Back pages** | `heliod` and `edgar-vampires` carry `next_issue: TO BE ANNOUNCED` | Pilot names the next deck |

### The frontend, precisely

`viz/deck.html` shipped 2026-07-29. Before it, the two products shared exactly one link
(newsstand → map) and the viz knew nothing of the pilot subsystem; now every issue's Back
Page opens its deck's dossier and the dossier links back. It also introduced the first URL
state this frontend has had (`?deck=<slug>`) and forced the design-token port into
existence against real content.

**The Deck Lens** (`viz/js/deck-map.js`) shipped 2026-07-30 as the map's third mode, and
closes the remaining gap: the map plotted 34,322 cards while seven decks sat in tracked
JSON one directory over with nothing connecting them. Pick a deck and its 99 light up
while the rest dim, so the deck's *footprint in card space* becomes legible — a claim
about archetype no decklist can make. It needed no new pipeline step: `card_roles.json` is
tracked, and all seven decks' card names resolve against `projection_2d.json` exactly.

It also generalised the overlay contract. `getOverlayTraces()` / `getDimmedIndices()` were
private to the deck builder; `render()` now dispatches to whichever mode owns the panel, so
mode four costs two methods. And `index.html?deck=<slug>` now enters the Lens directly, so
the three surfaces form a cycle rather than a chain.

Two deliberate honesty choices are worth keeping when this is extended: the panel states
that **bars count copies and dots count distinct cards** rather than letting the two
numbers disagree silently (the land bug's lesson, applied before it could recur), and a
verified line naming fewer than two deck cards **stays in the list, greyed** instead of
vanishing — so the panel's count always agrees with the manifest's `verified`.

**The renderer port is at Phase 3 of 4.** The map now draws on `<canvas>` + d3 behind
`?renderer=canvas`, as a strangler behind the `MM` contract — both renderers are live at
once and the *layer format is the trace format*, so there is no adapter to delete later.
Phase 3 took the last four things Plotly still owned (region labels → DOM, contours →
`d3.contourDensity`, the legend → a `<div>`, box-select → the quadtree) plus per-point
opacity and an `updateLayerBy` restyle path. Measured: `render()` 30 ms → 15 ms, box-select
138 ms → 4.5 ms per mousemove. `docs/viz.md` has the table and the reasoning.

**Phase 4 is the deletion** — drop the Plotly CDN tag, the `keepX`/`keepY` camera
preservation, the `_is*` trace flags, and the four resize timers; make canvas the only
path. Held back deliberately: the two renderers should stay comparable on identical data
until the canvas has run against real use, and the four source-assertion suites
(`test_viz_{camera,drill,viewer,deck_lens}.py`) need porting or retiring first — several
of their docstrings carry reasoning worth keeping.

**The 39 browser tests are the real gate here.** A perf commit once stripped a variable
declaration and left its use behind, breaking drill mode on every render; all 13
source-assertion drill tests passed and it shipped. Playwright caught it in both
directions when pointed at that revision. Nothing that renders should be verified by
grepping source again.

Still true, and still the two live defects in `viz/js/deck-builder.js`:

- its six-factor scorer **diverges from `config.DECK_BUILD_WEIGHTS`** — different weights
  on every shared factor, and its sixth factor is keyword Jaccard where Python uses
  castability. Two implementations of one algorithm, documented in two places with no
  cross-reference;
- saved decks persist **raw projection row indices**, so any pipeline refresh that changes
  card ordering silently reinterprets a saved deck as a different set of cards. A
  correctness bug, and the one item worth doing on its own schedule.

`docs/frontend-v2.md` keeps its analysis but carries an audit header: its M1 → M2 → M6
sequencing was wrong (the dossier had no prerequisites; the engine port is blocked on
`data/cards.csv` being gitignored) and its M3 premise predates the 17-section magazine.

## Shipped — the embedding rebuild

Find Similar returned neighbours that looked arbitrary. Measured, not guessed: **both trained
embedding spaces had collapsed, and the frozen MiniLM text they were built from beat both of
them 2:1.** The training stage was subtractive.

Held-out `test` split of a 40-group hand-authored golden set:

| space | dim | effective dim | 1st→50th gap | r@10 | r@50 | median rank |
|---|---|---|---|---|---|---|
| layout (colour+type) | 128 | 3.20 | 0.0041 | 0.090 | 0.142 | 1651 |
| frozen MiniLM text *(the input)* | 384 | 50.41 | 0.1411 | 0.244 | 0.414 | 124 |
| function — **before** | 128 | 5.97 | 0.0236 | 0.093 | 0.190 | 995 |
| function — **after** | 128 | **27.87** | 0.0315 | **0.245** | **0.455** | **78** |

Against the model it replaces: 2.6× recall@10, median rank cut 12.8×. Against the frozen text
it is built from: **recall@10 is a tie**; the real gains are recall@50 and median rank.

What it looks like in the product — *Doubling Season* went from *Gift of the Woods, Super
Strength, Naturalize the Phyresis* to *Primal Vigor, Parallel Lives, Branching Evolution,
Halving Season, Anointed Procession*.

| Phase | What |
|---|---|
| **0 — measure** | `manamap eval-embeddings` (step 14, the first reporting step) + `data/eval/similarity_golden.json` (40 groups, `dev`/`test` split) + `tests/test_embedding_quality.py`. Nothing here was falsifiable before it |
| **1 — inputs** | Card name out of the embedding text (0.187 → 0.248 alone — it bought similarity off shared *words*: *Rhystic Study* ↔ *White Rhystic Study* at 0.951); cost and P/T in; empty-string keyword slot fixed; EDHREC rank on a fixed rather than per-run scale; vocab capacity asserted |
| **2 — objective** | In-batch InfoNCE replaces a triplet margin that stopped teaching once satisfied; positives from roles rarest-first instead of a ≥2-tag rule that fell back to random for most of the corpus; fixed-weight text passthrough so similarity is exactly `0.7·cos_learned + 0.3·cos_text` and the text can no longer be discarded |
| **3 — consumers** | Similarity decoupled from the displayed map; the duplicate k-NN inside `findSimilarCards` deleted; duplicate names excluded; the `Math.max(0, dot)` clamp documented on both sides |
| **4 — refresh** | `manamap run --from preprocess`; eight tracked artifacts regenerated |

Two results worth keeping:

**The halves are complementary.** The learned half alone scores 0.136 recall@10 and the text
half alone 0.219, yet combined they reach 0.245 with median rank 78 — better than either half
and better than the full 384-dim frozen text. That is why positives are deliberately *not*
gated on text similarity: it would have made the learned half a copy rather than a complement.

**The golden set is too small to tune on.** Sweeping the text weight showed W=0.45 at 0.258
recall@10 — but that was selected by reading the held-out split. Selecting on `dev` picks
W=0.15, the two disagree, and everything in W ∈ [0.15, 0.6] is inside noise at ~50 dev and
~160 test queries. Shipped W=0.3, chosen a priori and fitted to neither split.

**Still open:** neighbour spread is 0.0315 against a 0.05 target — better than 0.0236 but the
top-50 remain tight. Held as a failing `xfail(strict=True)` gate rather than lowering the
threshold to match the result. Hard-negative mining was scoped out of this pass deliberately
(random in-batch negatives are safe here — 0.004% false-negative rate — but mined ones need a
similarity ceiling, since 39% of cards have a text neighbour above 0.75), and is the obvious
next lever.

## Shipped — ManaMap as an experience (discovery-first)

The product was reframed from the builder's view to the user's. It opens on **one card**:
hover it, click a relation, and its neighbours join a graph you grow by clicking. The 34K
scatter survives as a mode you go to (`?mode=explore`), not the thing you arrive at.

The useful surprise: **~70% already existed** inside The Walk — physics, drag-and-fling,
hover popup, click-to-branch, cumulative growth, card detail, no persistence. This was a
front door and some plumbing, not a rebuild.

**What it costs to use.** Boot is `viz_index.json` (0.56 MB gz) + `neighbours.bin` (1.27 MB
gz) = **1.83 MB**, against the **18.4 MB** it used to take to reach a first branch (12.9 MB
projection, then 16.8 MB of incompressible float32 embeddings on the first click). Branching
is **synchronous** — median 0.4 ms, no await inside the gesture, which is what makes
click-to-grow feel physical rather than laggy.

| | what it does |
|---|---|
| **Landing** | weighted random card, `?card=` / `?seed=` for a reproducible one, coarse filters, *Feeling lucky* |
| **Relations** | Similar / Synergy / Outclassed by, counts stated **before** the click, rendered in every panel via `MM.relate` |
| **Graph** | branch to grow, drag to fling, cross-links so it is a graph and not a tree, relation-inked edges, synergy edges labelled with their rule |
| **Decks** | load any of the seven by slug (commander ringed) or paste a Moxfield export; deck cards read differently from cards you found |
| **Tray** | keep cards, export a brief for the pilot loop in Claude Code — the site stays static |

**The synergy graph was recommending near-random cards**, which no interface would have
fixed. Partners were tie-broken by embedding *similarity* — backwards for a complementary
relation, since it surfaces cards resembling the anchor rather than cards that play with it.
Ranking by playability instead moved the median partner from EDHREC rank 10,713 to 1,472, and
top-2,000 share from 7.0% to 60.2%. Skullclamp went from *Playable Delusionary Hydra* to
*Yawgmoth, Thran Physician*. `tests/test_synergy.py:test_synergy_partners_are_playable` is
the gate.

**Findings worth keeping:**

- **"Anti-cards" do not exist.** Across 4.5M pairs, zero fall below cosine 0 (min +0.344,
  median +0.714) — the space is a narrow positive cone, so "orthogonal" is not a place. The
  complementary relation the vision wanted is the rule-based synergy graph, not distance.
- **Three live relation lookups would have made the product 5× heavier**, not lighter (~48 MB
  of lazy fetches with an await per click). Hence the precomputed table.
- **Coverage is uneven and the UI says so:** similar 100%, synergy 76.1%, obsolescence 22.5%,
  and **23.6% of cards have nothing but similar**. Doubling Season has no synergy partners at
  all — a real hole in the rules, now visible rather than buried.
- **A card whose top synergy tier is small cannot be rescued by re-ranking.** Skullclamp's
  holds 3 cards. That is coarseness in the 24 rules, not in the ordering.

**Open, deliberately:** the synergy rule set is coarse (24 rules) and leaves holes like
Doubling Season; mobile is undesigned (`force.js` registers only `mousemove` and `click`).

Full record: `/Users/michellemacrae/.claude/plans/wondrous-gliding-hoare.md`.

## Future — what is not started

### Frontend, in order

1. ~~**`viz_index.json`**~~ — **shipped** as pipeline step 14, though for discovery rather
   than the deck builder: name, supertype, colour, rarity, CMC and role tags per row, 0.56 MB
   gzipped. It still does **not** carry `game_changer`, `mechanical_tags`, `layout`, or
   `legal_commander` as a tri-state, which is what the engine port below actually needs — so
   that gap is real, just smaller and no longer blocking.
2. **Engine port to a Worker** — `manabase` is trivial (pure math, no deps), `bracket` and
   `goldfish` easy, `build_deck` hardest (pandas is load-bearing in pool filtering).
   `viz/js/engine/constants.js` must be **generated from `config.py`**, never hand-edited,
   with a parity test asserting both builders emit identical `build_plan.json` — otherwise
   the scorer divergence above simply recurs in a new file. Goldfish determinism needs an
   MT19937 port matching `random.shuffle`; the honest fallback is labelling
   browser-computed goldfish an *estimate* that never overwrites a committed ◆ artifact.
3. **`build.html`** — the deck starts complete and you review and swap slots, because the
   builder already fills 63 slots by role budget and keeps scored alternates for each. A
   score delta of 0.01 is the signal that the scorer was nearly indifferent and the
   pilot's judgment is cheap. Every mutation incremental; no `innerHTML` rebuild.
4. **Handoff** — emit `brief.json` + `decklist.txt` and the command to run.

Opportunistic, none blocking: ~~hover tooltips~~ **done** — hovering shows the card image at
the cursor in every mode, without Plotly's per-point text (`showCardPopup`); the detail panel
hides in build mode, exactly when you're deciding whether a card belongs; int8-quantising
`embeddings.bin` takes 17.6 MB → 4.4 MB.

**Agents run in Claude Code, not the browser**, and none of the above changes that. The
integration model is: render what agents produced (shipped), hand a brief back (step 4),
and deep-link cards between map and magazine. A browser-triggered agent run would need a
server this project deliberately does not have.

### Deck versioning — the remaining third

`HISTORY.md` (step 1) and the validated `decklist_sha256` (half of step 2) shipped. Still
open: **`supersedes` in `issue.json`**, so "Vol. 008 corrects Vol. 004" is expressible and
`build_index` has something to key on besides the slug — it currently emits at most one
entry per deck, so a second issue for the same deck would silently overwrite
`manuals/<slug>.html`. And **`build:<NNN>`** as a third dynamic cache-routine family.
Still explicitly avoid a hand-incremented `deck_version` int; `volume` already demonstrates
that failure mode.

### Queued verification work

- **Sisay's two failed stacks are one or two named clauses from passing.** 001 (the tutor
  chain) is the highest-value fix in the fleet — it would promote the ladder arithmetic and
  the summoning-sickness answer from ★ to ✓ across three sections at once. 003 needs a
  fresh run, not a patch.
- **Sisay's other Najeela pairs** (Faeburrow Elder, Esika, Selvala) — unlike Derevi, each
  produces mana, so the break-even arithmetic that refuted the two-card claim may resolve
  differently.
- **Hapatra: eleven unresolved combo lines**, headed by whether Mikaeus's `+1/+1` anthem
  switches off the deck's own token loops (if so, its two flagship engines are mutually
  exclusive), and the Blowfly Infestation family that goldfish actually tracks.
- **Hapatra has no decision spreads** — What's Your Play? renders a visible `[TODO]`. The
  editor specified both boards precisely enough to author straight into artifacts.
- **From the Short List batch**: Blight-Priest's would-be 2×3 matrix (edgar), the Seething
  Anger loop arithmetic (goblin-storm), Hullbreaker + Mana Vault and the Mentor loop
  (heliod), Ballista + Heliod and Staff/Assault + Selvala (sisay), Reaver Cleaver + Assault
  and Gnawbone + Charger (ur-dragon), Wrathful Raptors × the ping web (gishath).
- **Queued stacks**: Roaming Throne × Zada (genuinely unsettled — Throne makes Zada's copy
  trigger fire twice; whether that yields a second full copy set is the question) and the
  Past in Flames ritual rebuild.

### Known-wrong artifacts

- **`bracket_report.json` contradicts a verified stack in print** on hapatra: it still
  carries the inflated 19 two-card infinites and names a refuted pairing as its example
  driver. The floor of 4 holds on the 11 Game Changers alone. Either annotate the artifact
  with verified refutations or teach the engine to consult passing stacks.
- **`heliod` and `edgar-vampires` have `next_issue: TO BE ANNOUNCED`** — the Back Page is
  thin until the pilot names the next deck.

### Strategy-DB gaps

Four, all flagged by strategic frames and all load-bearing there: **auditing a combo list**
(how to price an unverified count; why pairwise combo data silently drops third pieces) —
the single most load-bearing idea in Vol. 002; **aristocrats/sacrifice engines** (outlet,
fodder, converter taxonomy); **counters-matter** (annihilation as a resource, the
anthem-versus-shrink anti-synergy); and **tutor sequencing** — now doubly wanted, since
Fetch Quests is a whole section with no strategy pillar behind it.

### Deferred by decision

**`meta-analyst`** — the v2 design called for a meta-awareness agent with its own
`data/meta/` corpus. Traded away to get the loop working, and the loop works without it.
The design point worth keeping: **meta claims perish and strategy theory doesn't**, so they
want separate corpora with different invalidation, and every meta section needs an `as_of`.

## Standing notes — decided, not open

### Judgment calls surfaced by audit, deliberately not changed

- `goldfish.py` — `mean_cast_turn` divides by games where the commander was cast;
  `cast_by_turn_6_rate` divides by all games. They sit side by side in the Commander File.
- `goldfish.py` — `cast_by_turn_6_rate` hardcodes turn 6 while `GOLDFISH_MAX_TURN` is
  configurable.
- `power_creep.py` — the docstring numbers eight criteria; the inline comments number seven.
- `download_combos.is_up_to_date` checks file existence only, so combo data never refreshes.

## Invariants that must not erode

- Only checker-passed stacks publish; failed artifacts are kept as open questions.
- Agents return JSON and never write HTML — that's what keeps rebuilds byte-identical.
- `issue.json` is authored, never generated.
- Costume never earns the badge: a section cannot claim a tier it wasn't granted.
- Record the cache **after** validation, never before.
- A bracket **floor** is what the contents are consistent with, never a verdict.
- The deterministic builder must always produce a complete legal 99 with no agent involved.
- **Never transcribe the section list or its count into a prompt** — read
  `issue_spec.DEPARTMENTS`. A list in prose goes stale; the spec cannot.
- **Count copies, not decklist entries**, for anything the shuffler would see.
