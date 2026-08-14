# PLAN — current state and what's next

*The resume-here doc. `README.md` orients, `CLAUDE.md` carries the gotchas, this says what
exists and what is open. Superseded plans live in `docs/history/`.*

Last updated **2026-08-14**. Everything below is committed and pushed except where marked.
Every figure was derived from the repo at write time — **do not quote one from memory**;
the commands that print them are named beside each.

## What this is

Two products in one repo, sharing a data layer and a CLI.

**The card tool** opens on a single card and grows a graph as you click. Underneath it,
34,890 oracle cards are embedded twice — once for layout, once for function — projected to
2D, and served as a static site from `viz/`. The atlas drifts at altitude and settles as you
zoom in.

**Pilot's Manual** turns one Commander deck into a self-contained web issue under a
three-tier evidence contract. The deck can be produced by a deterministic builder from a
written brief, and the issue now opens on two pictures of the deck: its **constellation**
(what shape it is) and its **engine flow** (how it runs).

**Nine issues published**, one per deck:
[001 Goblin Storm](https://macrae.github.io/mana-map/manuals/goblin-storm.html) ·
[002 Hapatra](https://macrae.github.io/mana-map/manuals/hapatra.html) ·
[003 Sisay](https://macrae.github.io/mana-map/manuals/sisay.html) ·
[004 Heliod](https://macrae.github.io/mana-map/manuals/heliod.html) ·
[005 Ur-Dragon](https://macrae.github.io/mana-map/manuals/ur-dragon.html) ·
[006 Edgar](https://macrae.github.io/mana-map/manuals/edgar-vampires.html) ·
[007 Gishath](https://macrae.github.io/mana-map/manuals/gishath.html) ·
[008 Yawgmoth](https://macrae.github.io/mana-map/manuals/yawgmoth-swarm.html) ·
[009 Radagast](https://macrae.github.io/mana-map/manuals/radagast.html) ·
[newsstand](https://macrae.github.io/mana-map/manuals/index.html)

Scale: 46 `manamap pilot` subcommands, 18 top-level subcommands, 17 agents, 19 skills,
13 cache-gated routines, 20 magazine sections (6 of them optional while two migrations are mid-flight). Test counts live in `docs/testing.md`.

## Start here on any deck

```bash
manamap pilot deck-status <slug>
```

It reports every lifecycle stage as present, missing or **STALE**, and flags stages added
recently that an older deck will not have. `/publish-deck` is the runbook that sequences the
whole thing; `pilot/deck_status.py:STAGES` is the machine-readable version, and **a new
phase belongs there** or the next person will not find it.

Staleness is an ERROR; incompleteness is a state. A half-built deck is work in progress. A
deck whose artifacts disagree about which decklist they describe is confident and wrong.

## The evidence contract

| | Tier | Granted by |
|---|---|---|
| ✓ | rules-verified | A stack artifact whose every step cites a real CR rule verbatim (`validate_stack.py`), then survives the adversarial `rules-checker`. Only a `pass` publishes. |
| ◆ | data-derived | Deterministic Python over committed artifacts. Same inputs, same bytes, no LLM. |
| ★ | coaching | Labelled judgment. Useful, and never disguised as measurement. |

**Agents return JSON and the renderer emits HTML**, so regenerating a manual from unchanged
artifacts is byte-identical. That is why a refactor cannot disturb a published issue.

## The decks

Run `deck-status` per deck for the live picture. As of this writing:

| Deck | Vol | Stacks | Bracket | Map | Engine | Front of book |
|---|---|---|---|---|---|---|
| `goblin-storm` | 001 | 5/5 | 4 | named | — | — |
| `hapatra` | 002 | 1/1 | 4 | named | — | — |
| `sisay` | 003 | 1/3 | 4 | named | — | — |
| `heliod` | 004 | 6/6 | 4 | named | — | — |
| `ur-dragon` | 005 | 6/6 | 4 | named | — | — |
| `edgar-vampires` | 006 | 7/9 | 4 | named | — | — |
| `gishath` | 007 | 5/5 | 4 | named | — | — |
| `yawgmoth-swarm` | 008 | 11/14 | 4 | named | — | — |
| `radagast` | 009 | 7/7 | 3 | named | **yes, `pass`** | **yes** |

*Stacks* is checker-passed / total; a failed artifact is kept as an open question and never
publishes. **Every deck now carries a bracket target and a named constellation.** Only
radagast has an engine model and the front-of-book departments — that is the current
frontier, not an oversight, and §2 is the rollout.

## What shipped in the 2026-08-13 cycle

Recorded here because a capability nobody knows about does not propagate. All of it is in
`deck_status.STAGES` and the gates, so new decks inherit it without anyone remembering.

**The constellation.** `deck-map` re-lays-out one deck's cards from the ability embeddings
and cuts two levels of cities and neighbourhoods; `deck-cartographer` names them for the job
their cards do. Radagast's read THE WIDE BOARD, MANA ON LEGS, THE TRAPS, THE BODYGUARDS,
GREEN ON CURVE, THE SEARCH PARTY, A CARD PER BODY. The 99 groups by city and shares the
map's ink by construction.

**The engine model.** The cartographer's own notes exposed the limit that motivated it: a
card is clustered by what it SAYS and an engine is what cards DO TO EACH OTHER, and on
radagast only **4 of 10** declared components sit in a single city. `engine-facts` joins
everything that already answers part of the question; `deck-engineer` ⇄ `engine-critic`
reasons over it into `engine.json` — eight closed stages, drawn as a flow whose arrows are
solid when a checker-passed stack proves them and dashed when they are a reading.

**Two magazine lints and two components.** Internal taxonomy ids in reader copy (68
occurrences across all eight published issues, now 0) and deks that open by asking the
reader a question (14, now 0). `coach-gauge` renders a ★ judgment as stars rather than a
percentage; `stat-slab` runs the issue's signature number once, full width.

**Ambient motion in the atlas**, in the projection rather than the data, with an exact
inverse so hit-testing keeps working.

**`deck-status` + `/publish-deck`** — the lifecycle, checked and sequenced.

## Open work

### 1. Phase 3 — DONE. Radagast carries the whole v4 shape.

All four parts shipped. The magazine now opens on **The Editor's Letter** (Margot Stet,
the masthead's only unbadged name) and **The Pilot's Log** (three columnists arguing in
the engine model's stage names), then the engine flow, then the constellation and a 99
grouped by its cities.

| | |
|---|---|
| Engine loop closed | critic `pass`, `deck-status radagast` **17/17** |
| Fourth persona | Margot Stet — no tier, no glyph, and `badge()` still raises |
| Two departments | radagast 19, the other eight 17, all nine validate |
| Voice separation | fleet-wide **0**; the blind-attribution test passes |

**What it cost to learn, in one line each** — the detail is in CLAUDE.md:

- A field is only as revisable as it is short. One stage failed four consecutive
  revisions at 2,554 characters; the cap is 1,800 and the artifact failed it on arrival.
- Run a proposed check against the whole fleet before keeping it. The voice lint shipped
  matching `"very "` inside `"every "` — 13 false hits — and then lost three more bans to
  the hedge/intensifier problem.
- A critic's findings become mechanical checks or its work is re-spent every run.
- The dashed line reaches the prose: the panel may discuss an unverified line, never
  assert it.

### 2. Roll the v4 shape to the other eight decks · **next**

Radagast proves every loop; the rest is repetition, in this order per deck:
`analyze-engine` (engineer ⇄ critic, the expensive one) → opt the two departments into the
plan → `pilot-panel` → `merge-prose` → `validate-issue` → `build-manual`. `deck-status`
says what each deck still needs; `/publish-deck` sequences it.

Budget honestly: `deck-engine` is the most expensive routine in the repo (~120k per
engineer pass, ~140k per critic; radagast took four spawns over three iterations plus two
scoped partial revisions), and `panel-prose` is ~134k. **Pay §9's remaining test debt
first** — it is small and it guards exactly what eight repetitions would stress.

### 3. The Coach-department merge — DONE on radagast, queued for the other eight

`politics-table` + `know-your-enemy` + `fetch-quests` → one **At the Table**, live on
radagast. The four ids are all in `OPTIONAL_DEPARTMENTS` — a **two-way** migration, since
neither the originals nor their replacement may be required while both shapes exist. Each
of the other eight moves when it is next re-planned, and the three originals get deleted
from the spec when the last one does.

**It did not shorten the issue, and the measurement is the point.** Words went 22,713 →
22,618 and rendered height went 65,977px → 66,278px — the two dropped openers bought about
800px and the taller constellation spent it back. What the merge actually fixed is
editorial: the reader met the same byline, colour and page furniture three times before the
argument had moved once. It is one destination now, and it reads as one argument that turns
twice.

If length is still the complaint, this is where it lives, measured at 1280px:

| section | px | words |
|---|---:|---:|
| The Kill | 10,651 | 3,228 |
| At the Table | 9,150 | 4,487 |
| The 99 | 7,930 | 3,268 |
| What's Your Play? | 6,308 | 3,393 |
| *(the other twelve, together)* | 27,786 | 8,006 |

Four sections are 55% of the issue. No further department merge reaches that — the next
cut is inside those four, and it is a cut to CONTENT, which is an editorial call and not a
renderer change.

### 4. Verification backlog

- **Sisay 001** (the tutor chain) is the highest-value fix in the fleet: it would promote the
  ladder arithmetic and the summoning-sickness answer from ★ to ✓ across three sections at
  once. 003 needs a fresh run, not a patch.
- **Grafdigger's Cage** — three of yawgmoth's four kills rest on an oracle reading no checker
  has settled.
- **Hapatra's eleven unresolved combo lines**, headed by whether Mikaeus's `+1/+1` anthem
  switches off the deck's own token loops. If it does, its two flagship engines are mutually
  exclusive.
- **Radagast's `open_questions`** — six, emitted by the engineer with a `settled_by` each.
  Whether the other three finishers also carry the kill is the load-bearing one; only
  Craterhoof has a stack.
- **Queued**: Roaming Throne × Zada, the Past in Flames ritual rebuild, sisay's other
  Najeela pairs.

### 5. Stack artifact staleness

Twelve stack artifacts carry preambles that violate L10 and name retired concepts, and
edgar's presentable stacks cross-reference a non-presentable one. **Do not regex it** — a dry
run destroyed substance in four files. Needs a per-file read, and it invalidates
`stacks:passing` fleet-wide, so do it in one pass and re-record. ~200k.

### 6. `deck-recon` on six decks

Never run there. Its staleness is **time, not inputs** — a decklist edit does not change what
strong lists for a commander run — so it needs dated web passes judged against
`RECON_MAX_AGE_DAYS` (120). ~600k.

### 7. Known-wrong artifacts

- **hapatra's `bracket_report.json` contradicts a verified stack in print**: an inflated
  two-card-infinite count naming a refuted pairing as its driver. The floor of 4 holds on the
  11 Game Changers alone. Either annotate the artifact or teach the engine to consult passing
  stacks.
- **`build_plan.json` is not reproducible from today's data** — re-running `build-deck` on
  hapatra yields a different 99, because the embeddings, roles and synergy graph it scored
  against have been regenerated. Nothing tests this.
- **Five decks carry `next_issue: TO BE ANNOUNCED`.**

### 8. Strategy-DB gaps

Four, all flagged by strategic frames and all load-bearing there: **auditing a combo list**;
**aristocrats/sacrifice engines** (outlet, fodder, converter — now partly answered by the
engine model's stage vocabulary, which is the better home for it); **counters-matter**; and
**tutor sequencing**, doubly wanted since Fetch Quests is a whole section with no pillar.

### 9. Unit tests for the new subsystems · **partly paid**

**Done (sprint step 0):** `deck_map.json` and `engine.json` are now gated by
`test_pilot_tracked_artifacts_validate.py` (45 → 55 cases), and both gates were proven to
fire by sabotaging a real artifact and watching them fail. `tests/test_pilot_deck_map.py`
adds determinism, orientation, the unit box, the balance bound, completeness and unique
naming. Suite 1,530 → 1,547.

**One correction came out of writing them, and it matters more than the tests.** The record
credited Ward linkage with shrinking radagast's oversized city. It did not: at the k a
divisor picks, Ward is 38/71 and average linkage 37/71 — *average is better*. Ward only
leads once k grows (32% vs 49% at k=7), so the win belongs to **the balance bound**. Ward's
other apparent advantage — no one-card cities — does not generalise either; on
outlier-bearing data it strands strays exactly as average does. The test file asserts the
bound and documents why it **refuses** to assert the linkage: three attempts each came back
narrower than the last, and a test asserting it would look like a fact about the algorithm
while being a fact about one deck. CLAUDE.md is corrected too.

**A second correction, from closing the engine loop.** `stages[4].what_it_does` failed
**four consecutive revisions** — each fixed the defect it was sent to fix and introduced a
new one. The cause was not the agent: the field had reached 2,554 characters by accreting
narration of its own previous drafts, against 836–1,645 for the six stages that were never
revised at all. **A field is only as revisable as it is short.** `validate-engine` now caps
`what_it_does` at 1,800 characters, the charter says why, and the fifth revision — cut to
1,704 by deleting draft narration rather than findings — passed. The limit is measured, not
fitted: the data failed it when it was added.

The step-0 gate earned itself the same day: `test_pilot_tracked_artifacts_validate.py`
caught the tracked `engine.json` violating the new limit before anything else did.

**Paid since:** the constellation's two rendering rules are gated
(`test_pilot_deck_map.py` — the diffuse-lobe suppression proven against a compact-lobe
control, and the card-labelling completeness contract), the Act III merge has two
(`test_pilot_build_manual.py`, including the pre-merge fallback), and both browser failures
are fixed and green on three consecutive full runs. `test_docs_section_count.py` grew three
inventory guards — step numbers against `pipeline.STEPS`, the pilot command list against
`PILOT_STEPS`, and tracked per-deck files against the docs. That last one immediately found
`data/decks/radagast/None`, 25 KB of superseded deck map written by an early run that handed
`resolve_out_path` a literal `None`; it had been tracked for weeks because nothing ever
compared the directory against the docs. Deleted.

**Still owed:**

- `merge_deck_map`, `engine_facts` and `deck_status` have no direct unit tests. The
  tracked-artifact gate exercises the first indirectly; the other two do not appear in any
  test.
- No regression floor on the balance bound's *effect* — the test asserts the invariant holds
  on a synthetic fixture, not that real decks stay balanced.

Do the rest before rolling `analyze-engine` to the other eight decks.

### 10. Codebase hygiene

Still open and genuinely worth doing:

- **Browser-suite runtime**: ~73s of unconditional `wait_for_timeout`, and most tests
  re-parse the projection because pages are function-scoped. Replace the sleeps with
  condition waits first; only consider page reuse if that misses the budget, since cross-test
  state is this repo's known enemy.
- ~~`test_the_spotlight_actually_dims_the_canvas` fails on `main`~~ — **FIXED 2026-08-14,
  and the feature was never broken.** The test counted green over the whole canvas, but
  clearing a spotlight rests the line you were looking at *and* un-mutes the other eleven
  verified lines; on goblin-storm those cancel (868 px spotlit vs 833 cleared, ratio 0.96)
  while the line's own box goes 1024 → 301. It reads `Force.spotlitRows` and measures near
  the line now, with a threshold measured on both sides. The sibling flake,
  `test_canvas_draws_density_contours` (~1 run in 3), was a `canvas_page` race: `setCamera`
  no-ops while `baseFit` is null, so the test measured the fitted view believing it had
  zoomed. Fixed in the fixture — every test on it had the same hole. Full browser suite
  green on three consecutive runs.
- **Leave `.git` alone** — history rewriting breaks every clone to reclaim ~76 MB.

Closed deliberately: **`config.py` is NOT split** (most-imported file in the repo; the
frozen/mutable boundary is a rule, not a filing system), **int8 embeddings** measured at
97.3% top-10 agreement with no first-paint win to weigh against it — a user-facing quality
trade belongs to a person — and **five flagged "duplications" were false**.

### 11. Deck versioning — the remaining third

`HISTORY.md` and a validated `decklist_sha256` exist. Still open: **`supersedes` in
`issue.json`**, so "Vol. 009 corrects Vol. 004" is expressible and `build_index` has
something to key on besides the slug — it emits at most one entry per deck today, so a second
issue for one deck would silently overwrite `manuals/<slug>.html`.

### 12. Frontend engine port — not started

`manabase` is trivial, `bracket` and `goldfish` are easy, `build_deck` is hardest because
pandas is load-bearing in pool filtering. `viz/js/engine/constants.js` must be **generated
from `config.py`**, never hand-edited, with a parity test. Goldfish determinism needs an
MT19937 port; the honest fallback is labelling a browser-computed goldfish an *estimate* that
never overwrites a ◆ artifact.

## Decisions that bind

### The frontend stays LLM-free

The deployed static site and the local checkout run the same code. The viz is exploration
plus artifact reads; the agent loop stays in Claude Code, reached by an exported brief. No
local bridge, because a bridge means the deployed site and your machine run different code
and only one of them is the one you test.

The costing behind it: the deterministic layer is 45 subcommands answering in 0.2–2.2s with
JSON out and zero LLM calls. What genuinely needs an agent is artifact-shaped and expensive —
the cheapest routine is `coach-prose` at ~54.5k tokens, `candidate-pool` is ~235k.

### Similarity comes from the function space, always

`embeddings.npy` is the **layout** space (colour and type) and feeds `projection_2d.json`
only. `embeddings_ability.npy` is the **function** space and is the sole source of similarity
— Find Similar, the walk, drill and the deck map all read it. Held-out: 27.87 effective
dimensions, recall@10 0.245, median rank 78. Similarity is exactly
`0.7·cos_learned + 0.3·cos_text` by fixed weight, so the model cannot discard the text.

**Do not tune on the golden set.** At ~50 dev / ~160 test queries everything in
W ∈ [0.15, 0.6] is inside noise and the splits disagree. W=0.3 was chosen a priori.

Still open: neighbour spread 0.0315 against a 0.05 target, held as a failing
`xfail(strict=True)` rather than lowering the threshold to match the result.

### Synergy is complementary, not similar

24 rules over mechanical tags (blink → ETB), ranked by **playability**, not embedding
similarity. Coverage is uneven and the UI says so: similar 100%, synergy 76.1%, obsolescence
22.5%. **"Anti-cards" do not exist** — across 4.5M pairs the minimum cosine is +0.344, median
+0.714, so "orthogonal" is not a place.

### The clusters are an input to engine analysis, never the analysis

Measured before it was built and re-measured after: only 4 of 10 of radagast's declared
components sit in a single city. A city name is the wrong address for a component, and a
disagreement between the map and the engine is a *finding* — it means that part of the engine
is not visible in card text.

## Invariants that must not erode

- Only checker-passed stacks publish; failed artifacts are kept as open questions.
- Agents return JSON and never write HTML — that is what keeps rebuilds byte-identical.
- `issue.json` is authored, never generated.
- Costume never earns the badge: a section cannot claim a tier it was not granted.
- Record the cache **after** validation, never before. Never `cache-record` to make a board
  green, and never record a routine whose critic verdict is `fail`.
- Charter edits invalidate before they inform — make them **before** `cache-record`.
- A bracket **floor** is what the contents are consistent with, never a verdict.
- The deterministic builder must always produce a complete legal 99 with no agent involved.
- **Never transcribe the section list or its count into a prompt** — read
  `issue_spec.DEPARTMENTS`.
- **Count copies, not decklist entries**, for anything the shuffler would see.
- `--out` on a per-deck command must be slug-scoped.
- **A validator that fires on correct data is worse than none** — measure a proposed check
  against the whole fleet before keeping it. Four were scoped down or deleted this cycle.
- **A critic's findings become mechanical checks**, or its work is re-spent every run.
- **Name what a gate cannot see** rather than papering it with string matching.
