# PLAN — current state and what's next

*The resume-here doc. `README.md` orients, `CLAUDE.md` carries the gotchas, this says what
exists and what is open. Superseded plans live in `docs/history/`.*

Last updated **2026-08-13**. Everything below is committed and pushed except where marked.
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

Scale: 45 `manamap pilot` subcommands, 18 top-level subcommands, 16 agents, 19 skills,
12 cache-gated routines, 17 magazine sections. Test counts live in `docs/testing.md`.

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

| Deck | Vol | Stacks | Bracket | Map | Engine |
|---|---|---|---|---|---|
| `goblin-storm` | 001 | 5/5 | 4 | named | — |
| `hapatra` | 002 | 1/1 | 4 | named | — |
| `sisay` | 003 | 1/3 | 4 | named | — |
| `heliod` | 004 | 6/6 | 4 | named | — |
| `ur-dragon` | 005 | 6/6 | 4 | named | — |
| `edgar-vampires` | 006 | 7/9 | 4 | named | — |
| `gishath` | 007 | 5/5 | 4 | named | — |
| `yawgmoth-swarm` | 008 | 11/14 | 4 | named | — |
| `radagast` | 009 | 7/7 | 3 | named | **yes** |

*Stacks* is checker-passed / total; a failed artifact is kept as an open question and never
publishes. **Every deck now carries a bracket target and a named constellation.** Only
radagast has an engine model — that is the current frontier, not an oversight.

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

### 1. Phase 3 — the Editor's Letter and the Pilot's Log · **next**

The engine stages exist; this is where they become a vocabulary three voices argue in. The
founder's framing is recorded verbatim in `docs/magazine-feedback-2026-08-13.md` §9 — the
engine doc is scaffolding each columnist reads and interprets in character, never printed as
prose. Four parts, in order:

1. **Close the engine loop.** Radagast's `engine.json` carries a `critic: fail` — two
   clauses in one sentence of `stages[4]` (it cites CR 117.1a where stack 007 cites 304.1,
   and says the window is "only through Radagast", contradicting its own Yeva argument).
   The loop is at 3/3, so this is a scoped partial revision. **Not cache-recorded**, correctly.
2. **A fourth persona** — an editor-in-chief with **no tier and no badge**; the three
   columnists own ◆ ✓ ★ and a fourth that owned one would break §10.
3. **Two departments**: `editors-letter` (cheap, the magazine-editor writes it) and
   `pilots-log` (an agentic three-way conversation, ~33% each, opening on a play moment).
   Both go into `DEPARTMENTS` **and** `OPTIONAL_DEPARTMENTS`.
4. **Voice separation** — sentence-level register rules in STYLEv3 §7.7 and a
   banned-construction list per voice (Sunny may not write "posture", "prescribes",
   "framework"; Ledger takes no adjectives; Vera keeps the legalese because she is the only
   one who earns it), backed by a per-byline lint in `validate_issue`. The test is the
   editor's: **cover the bylines and attribute three paragraphs.** If a reader cannot, the
   phase is not done. The 2026-08 record predicted this round by leaving it open; do not
   leave it open again.

The Pilot's Log's requirements, from the founder's own words: it opens on a concrete play
moment tied to a primary win line, segues off what the last voice said, touches three or four
topics at roughly a third each, and every voice reads the same two inputs — `engine.json` and
`strategic_frame.json`. **The tie worth building: a line drawn DASHED in the engine flow is a
line the panel may not assert.** That is the evidence contract reaching into the prose
instead of stopping at the picture.

**The blocker is already removed.** `issue_spec.OPTIONAL_DEPARTMENTS` (empty today) lets a
department be piloted on one deck instead of arriving on nine at once. Optional means "an
older plan without it stays valid", NOT "the editor may skip it" — remove an id from the set
once every deck has it.

### 2. Roll the new subsystems to the other eight decks

Radagast proves the loop; the rest is repetition. `deck-map` and its naming pass are cheap
and already done fleet-wide. **`analyze-engine` is not** — eight decks × (engineer + critic)
is the expensive item, and it should follow Phase 3 so the engine model is being written for
a consumer that exists.

### 3. The deferred Coach-department merge

`politics-table` + `know-your-enemy` + `fetch-quests` → one **At the Table**. Three
consecutive Coach sections; the founder approved the merge. Now cheaper than when it was
deferred, because `OPTIONAL_DEPARTMENTS` makes a staged department change possible.

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

### 9. The 2026-08-13 cycle shipped with NO unit tests · **owed**

Recorded plainly because it is the kind of debt that becomes invisible: five new Python
modules — `deck_map`, `merge_deck_map`, `engine_facts`, `validate_engine`, `deck_status` —
are referenced by **zero** test files. The suite grew 1,524 → 1,530 this cycle and all six
are the browser motion tests.

They are not unexercised — every one was run repeatedly against all nine real decks, and the
validators were each proven to fire by hand-breaking an artifact. But "I ran it" is not
coverage, and the specific risks are known:

- **`test_pilot_tracked_artifacts_validate.py` covers five artifacts and neither new one.**
  Adding `deck_map.json → validate_deck_map` and `engine.json → validate_engine` to its map
  is the single highest-value line of test code available, and it is one line each.
- `deck_map`'s determinism (`_orient` pinning rotation, reflection and scale) is asserted by
  nothing. It was verified once by rerunning and diffing bytes; a regression there silently
  redraws every map.
- The measured clustering rules — Ward over average linkage, the 35% balance bound — have no
  regression floor, so a linkage change would pass silently.

Do this before rolling `analyze-engine` to the other eight decks.

### 10. Codebase hygiene

Still open and genuinely worth doing:

- **Browser-suite runtime**: ~73s of unconditional `wait_for_timeout`, and most tests
  re-parse the projection because pages are function-scoped. Replace the sleeps with
  condition waits first; only consider page reuse if that misses the budget, since cross-test
  state is this repo's known enemy.
- **`test_the_spotlight_actually_dims_the_canvas` fails on `main`** and predates this cycle —
  verified by stashing. Build's verified-line spotlight is not dimming (green pixels 1116 vs
  1117 where healthy is ~0.6).
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
