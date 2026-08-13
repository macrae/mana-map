# PLAN — current state and what's next

*The resume-here doc. `README.md` orients, `CLAUDE.md` carries the gotchas, this says what
exists and what is open. Superseded plans live in `docs/history/`.*

Last updated 2026-08-09. Everything below is committed, pushed and deployed. Every figure
was derived from the repo at write time.

## What this is

Two products in one repo, sharing a data layer and a CLI.

**The card tool** opens on a single card and grows a graph as you click. Underneath it,
34,322 oracle cards are embedded twice — once for layout, once for function — projected to
2D, and served as a static site from `viz/`.

**Pilot's Manual** turns one Commander deck into a self-contained web issue under a
three-tier evidence contract, and the deck itself can be produced by a deterministic
builder from a written brief.

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
10 cache-gated routines, 17 magazine sections. Test counts live in `docs/testing.md`,
which states the command that prints them — they change too often to restate elsewhere.

## The evidence contract

Everything the magazine prints carries one of three marks, and the mark is granted by a
mechanism rather than claimed by an author.

| | Tier | Granted by |
|---|---|---|
| ✓ | rules-verified | A stack artifact whose every step cites a real CR rule verbatim (`validate_stack.py`), then survives the adversarial `rules-checker`. Only a `pass` publishes. |
| ◆ | data-derived | Deterministic Python over committed artifacts. Same inputs, same bytes, no LLM. |
| ★ | coaching | Labelled judgment. Useful, and never disguised as measurement. |

This is why a refactor cannot disturb a published issue: **agents return JSON and the
renderer emits HTML**, so regenerating a manual from unchanged artifacts is byte-identical.

## Capabilities

| Layer | What it does |
|---|---|
| **Rules DB** | The Comprehensive Rules chunked one-per-rule (~3.9K), chunk id = citation id. Semantic query plus exact lookup, local MiniLM, no network at query time. |
| **Strategy DB** | 45 sourced sections across 14 pillars; `strategy:<id>` is a citation id under the same verbatim contract as a CR rule. |
| **Goldfish** | Seeded Monte Carlo (seed 42, 10K iterations) measuring resource development — not full games. Its nine assumptions are rendered in the issue. |
| **Bracket engine** | Computes a **floor** from Game Changers, contained combos, two-card infinites and mass land denial, and names the card or line that drove it. |
| **Deterministic builder** | `brief.json` → `build_plan.json` → `decklist.txt` with no agent involved, so a cache miss degrades to a worse deck rather than no deck. `manabase.py` sizes colour sources hypergeometrically. |
| **Deck audit** | 16 axes, each carrying the verbatim `strategy.md` quote that sets its target, joined with engine-activation rates. A target nobody can quote is not a target. |
| **Diagnosis loop** | `deck-doctor` ⇄ `deck-skeptic`, ≤3 iterations, gated by `validate-diagnosis`, which re-derives every axis figure rather than trusting it. |
| **Magazine** | 17 sections in five acts; `issue_spec.DEPARTMENTS` is the only authority. Three signing columnists, one per tier. |
| **L10** | Every issue reads as the reader's first: no version numbers, no swap-wave narration. Enforced in `validate_issue.validate_self_containment()`. |
| **Agent cache** | Fingerprints each routine's declared inputs into a tracked `.agent-cache.json`. Check → spawn → validate → **record last**. |
| **Incremental regeneration** | Per-card digests plus conservative `card_refs`, so a one-card swap invalidates only what references that card. `manamap pilot impact` reports what a change touches. |
| **Deck dossier** | `viz/deck.html?deck=<slug>` renders every committed artifact as data. Nothing recomputed in the browser, nothing hardcoded. |

## The decks

All eight published, floor 4 across the board.

| Deck | Vol | Stacks | Decisions | GC | Tutors | Lands | Notes |
|---|---|---|---|---|---|---|---|
| `goblin-storm` | 001 | 5/5 | 2 | 0 | 2 | 36 | Verified true infinite (Haze of Rage + Storm-Kiln); stack 004 **refutes** a combo-graph Krenko infinite |
| `hapatra` | 002 | 1/1 | 2 | 11 | 8 | 36 | Built by the deterministic loop — `brief` + `build_plan` + `candidate_pool`. Cleanest mana in the fleet: zero tapped, 96.2% on curve both colours |
| `sisay` | 003 | **1/3** | 2 | 4 | 1 | 40 | Publishes with `fail` artifacts on the record. Stack 002 corrected the bracket engine in print |
| `heliod` | 004 | 6/6 | 2 | 10 | 3 | 33 | Verified Aetherflux table-kill; blue at 26 sources, 74.2% on curve |
| `ur-dragon` | 005 | 6/6 | 2 | 3 | 4 | 34 | Includes a refutation (Sneak Attack ≠ commander-cheat) and the Throne + Bloodletter bound |
| `edgar-vampires` | 006 | 9/9 | 2 | 4 | 4 | 37 | A fully verified 2×2 loop matrix (Vito/Bond × Exquisite/Conqueror) |
| `gishath` | 007 | 5/5 | 2 | 2 | 5 | 35 | Headline is a refutation: Marauding Raptor + Polyraptor is a whole-table **draw** machine |
| `yawgmoth-swarm` | 008 | 13/13 | 2 | 4 | 5 | 36 | Most verified lines in the fleet; built from a brief |

*Stacks* counts checker-passed / total. *GC* is Game Changers; *Tutors* is `tutor_guide.json`
entries (one wish per maindeck tutor).

Every deck carries `cards.json`, `decklist.txt`, `bracket_report.json`, `considering.json`,
`diagnosis.json`, `goldfish_metrics.json`, `goldfish_targets.json`, `issue.json`,
`issue_plan.json`, `mana_analysis.json`, `manual_prose.json`, `strategic_frame.json`,
`tutor_guide.json`, `decisions/`, `stacks/` and `.agent-cache.json`. Uneven by design:
`build_plan.json` + `brief.json` on hapatra and yawgmoth (the two brief-built decks),
`candidate_pool.json` on hapatra, `deck_recon.json` on edgar and yawgmoth, `HISTORY.md` on
the four decks that have been revised.

## The agent roster (14)

All read-only except where noted. Definitions in `.claude/agents/`.

| Agent | Role |
|---|---|
| `stack-resolver` | Cite-or-decline stack resolutions |
| `rules-checker` | Adversarial citation verification + missing-step audit |
| `manual-writer` | Body prose (6 keys incl. `mana_base`), per-key persona voice |
| `pilot-coach` | Threat, matchups, decisions (★) and the tutor guide |
| `magazine-editor` | The issue plan: cover, sections, headlines, furniture |
| `strategy-researcher` | Strategy-doc research (**write-scoped to `data/strategy/`**) and consulting |
| `deck-analyst` | ◆ data layer; emits `candidate_pool.json` |
| `deck-architect` | Improves the deterministic plan; every ratio cites `strategy:<id>` |
| `deck-critic` | Adversarial verifier for build plans; report-only |
| `deck-doctor` | Diagnoses a finished deck (modes: `recon`, `diagnose`) |
| `deck-skeptic` | Adversarial verifier for diagnoses |
| `short-list-analyst` | The Short List — ten cards worth knowing about, pool-scouted |
| `pipeline-runner` | Runs and diagnoses card-pipeline steps |
| `viz-dev` | Frontend work (write-scoped to `viz/`) |

## The pipelines

```
Build     brief.json → build-deck → validate-build → bracket-check
                     → /build-deck   (deck-analyst → deck-architect ⇄ deck-critic, ≤3)
                     → decklist.txt

Publish   fetch-deck → validate-deck → goldfish → mana-analysis
                     → /resolve-stack per line   (resolver → validate-stack → rules-checker, ≤3)
                     → /write-manual             (frame → coach → writer → merge-prose)
                     → /short-list               (the Short List)
                     → /design-issue             (magazine-editor → validate-issue)
                     → build-manual → build-index

Diagnose  deck-audit → /diagnose-deck  (doctor ⇄ skeptic, ≤3) → validate-diagnosis
```

Ten cache-gated routines — `candidate-pool`, `deck-build`, `strategic-frame`,
`coach-prose`, `writer-prose`, `the-ten`, `tutor-guide`, `issue-plan`, `deck-recon`,
`deck-diagnosis` — plus the dynamic `stack:NNN` / `decision:NNN` families. `cache-status`
before spawning, `cache-record` after validating. Exit 0 = don't spawn, 1 = spawn, 2 = fix
the input first.

## The cache board

**Do not quote a number here from memory — run `cache-status` per deck.** The count moves
whenever a *shared* input is edited, and the surfaces that do that are easy to forget:
`STYLEv3.md`, `issue_spec.py`, `deck_audit.py` and every `.claude/agents/*.md` are all
declared inputs. A doc pass that touches `STYLEv3.md` MISSes `issue-plan` on every deck in
the fleet, and there is no warning at edit time.

As of 2026-08-09, 29 non-HIT:

| Routine | n | Why |
|---|---|---|
| `issue-plan` | 8 | `STYLEv3.md` edited in the documentation pass |
| `the-ten` | 8 | `bracket_report.json` gained `target` / `within_target` |
| `deck-recon` | 6 | Never run on those decks — absent, not stale |
| `deck-diagnosis` | 6 | Same `bracket_report.json` change |
| `candidate-pool` | 1 | Never run (yawgmoth) |

None of these is a stale *figure*: no floor moved, no measurement changed. They are the
cache correctly reporting that a declared input's bytes differ.

**Never `cache-record` to make a board green** — the record is the claim that a human read
the artifact and agreed it holds. Where an input changed but no figure did, the honest
route is `cache-snapshot` → change → `cache-rerecord`, or a per-routine re-bless with the
reasoning committed.

## Open work

### 1. The six withheld diagnoses — start here

Six decks have a `diagnosis.json` whose skeptic verdict is `fail`, several judged against
figures since corrected: gishath's was killed over an enrage figure now measured at 60.8%,
and hapatra's prescription was sized against a Mikaeus loop that its own checker-passed
stack refutes. `/diagnose-deck` per deck, ~250k each, ~1.5M for six. Worth being the
session's goal rather than its tail — it has been inherited at the end of a long session
twice and has not happened either time.

### 2. Six bracket targets — a decision, not work

Six of eight decks compute a floor but never answer "is this deck inside its bracket",
because a target bracket is the pilot's statement of intent and must not be inferred from
a floor. hapatra and yawgmoth declare 4. The other six need a human to say.

### 3. Verification backlog

- **Sisay 001** (the tutor chain) is the highest-value fix in the fleet: it would promote
  the ladder arithmetic and the summoning-sickness answer from ★ to ✓ across three
  sections at once. 003 needs a fresh run, not a patch.
- **Sisay's other Najeela pairs** — Faeburrow Elder, Esika, Selvala. Unlike Derevi each
  produces mana, so the break-even arithmetic may resolve differently.
- **Grafdigger's Cage** — three independent passes named it the fleet's highest-value
  unresolved line. Three of yawgmoth's four kills rest on an oracle reading no checker has
  settled.
- **Hapatra's eleven unresolved combo lines**, headed by whether Mikaeus's `+1/+1` anthem
  switches off the deck's own token loops. If it does, its two flagship engines are
  mutually exclusive.
- **From the Short List batch**: Blight-Priest's 2×3 matrix (edgar), the Seething Anger
  loop (goblin-storm), Hullbreaker + Mana Vault and the Mentor loop (heliod), Ballista +
  Heliod and Staff/Assault + Selvala (sisay), Reaver Cleaver + Assault and Gnawbone +
  Charger (ur-dragon), Wrathful Raptors × the ping web (gishath).
- **Queued stacks**: Roaming Throne × Zada (Throne makes Zada's copy trigger fire twice;
  whether that yields a second full copy set is the question) and the Past in Flames
  ritual rebuild.

### 4. Stack artifact staleness

Twelve stack artifacts carry preambles that violate L10 and name retired concepts, and
edgar's presentable stacks cross-reference a non-presentable one. **Do not regex it** — a
dry run destroyed substance in four files. Needs a per-file read, and it invalidates
`stacks:passing` fleet-wide, so do it in one pass and re-record. ~200k.

### 5. `deck-recon` on six decks

Never run there. Its staleness is **time, not inputs** — a decklist edit does not change
what strong lists for a commander run — so it needs dated web passes and a judgement
against `RECON_MAX_AGE_DAYS` (120). ~600k.

### 6. Known-wrong artifacts

- **hapatra's `bracket_report.json` contradicts a verified stack in print**: it carries an
  inflated two-card-infinite count and names a refuted pairing as its example driver. The
  floor of 4 holds on the 11 Game Changers alone. Either annotate the artifact with
  verified refutations or teach the engine to consult passing stacks.
- **`build_plan.json` is not reproducible from today's data.** Re-running `build-deck` on
  hapatra yields a different 99 than the committed plan, because the embeddings, roles and
  synergy graph it scored against have since been regenerated. Nothing tests this — the
  freshness suite covers `bracket_report`, `goldfish_metrics`, `mana_analysis` and the
  rendered manual, but not the build plan.
- **Five decks carry `next_issue: TO BE ANNOUNCED`** — the Back Page stays thin until the
  pilot names the next deck.

### 7. Strategy-DB gaps

Four, all flagged by strategic frames and all load-bearing there: **auditing a combo list**
(how to price an unverified count; why pairwise combo data silently drops third pieces);
**aristocrats/sacrifice engines** (outlet, fodder, converter taxonomy); **counters-matter**
(annihilation as a resource, the anthem-versus-shrink anti-synergy); and **tutor
sequencing**, doubly wanted since Fetch Quests is a whole section with no pillar behind it.

### 8. Codebase hygiene — what is left

Phases 1–4 are done. **Phase 3 shipped** (70 dead CSS classes, `mana-map.css` 621 → 482
lines) and **Phase 5 is closed with two items deliberately declined.**

**`config.py` is NOT split, and should not be.** The plan recommended it; the
recommendation was wrong. It is the most-imported file in the repo (70 modules), so a
re-export façade means two files to keep in sync forever for no behaviour change. And the
frozen/mutable boundary is a **rule**, not a filing system: changing `MECHANICAL_TAGS` or
a model-facing dim invalidates `model_ability.pt`, and that is easier to enforce with
those constants under one loud warning than spread across modules. It now carries a
table-of-contents docstring stating that boundary explicitly, which addresses the real
complaint — navigation — at no risk.

**int8 embeddings: measured, and the decision is the pilot's.** Quantising takes each file
17.6 MB → 4.4 MB, dropping 26.4 MB from the tracked set and the Pages payload. Measured
cost over 400 sampled cards: **97.3% top-10 neighbour agreement**, so roughly one card in
37 moves in a Find Similar result. The fetch is **lazy** — `embeddings` loads only for
Find Similar, never on the discovery boot — so there is no first-paint win to weigh
against it. A user-facing quality trade belongs to a person.

**Five flagged "duplications" were false and none was touched**: `setCommander` ×3
(deliberate layers), `esc` in `deck-view.js` (required — `deck.html` loads it alone, so
`window.MM` does not exist there), discovery's tray trio (they add the repaint),
`build_manual._card_probes` vs `card_refs.name_probes` (link insertion vs invalidation
detection), and `build_name_index` ×2 (opposite tie-breaks per consumer). Name collision
is not duplication, and this codebase collides a lot.

Still open, and genuinely worth doing:

- **Browser-suite runtime**: ~73s of unconditional `wait_for_timeout` in 24 tests, and 62
  of 118 tests re-parse the 12.9 MB projection because pages are function-scoped. Replace
  the sleeps with condition waits first; only consider page reuse if that misses the
  budget, since cross-test state is this repo's known enemy.
- **Leave `.git` alone** — history rewriting breaks every clone to reclaim ~76 MB.

### 9. Deck versioning — the remaining third

`HISTORY.md` and a validated `decklist_sha256` exist. Still open: **`supersedes` in
`issue.json`**, so "Vol. 009 corrects Vol. 004" is expressible and `build_index` has
something to key on besides the slug — it emits at most one entry per deck today, so a
second issue for one deck would silently overwrite `manuals/<slug>.html`. And
**`build:<NNN>`** as a third dynamic cache-routine family. Avoid a hand-incremented
`deck_version` int; `volume` already demonstrates that failure mode.

### 10. Frontend, not started

**Engine port to a Worker.** `manabase` is trivial (pure maths), `bracket` and `goldfish`
are easy, `build_deck` is hardest because pandas is load-bearing in pool filtering.
`viz/js/engine/constants.js` must be **generated from `config.py`**, never hand-edited,
with a parity test asserting both builders emit identical `build_plan.json`. Goldfish
determinism needs an MT19937 port matching `random.shuffle`; the honest fallback is
labelling a browser-computed goldfish an *estimate* that never overwrites a ◆ artifact.

`viz_index.json` ships but does not carry `game_changer`, `mechanical_tags`, `layout` or a
tri-state `legal_commander`, which is what the port actually needs.

## Decisions that bind

### The frontend stays LLM-free

The deployed static site and the local checkout run the same code. The viz is exploration
plus artifact reads; the agent loop stays in Claude Code, reached by an exported brief. No
local bridge, because a bridge means the deployed site and your machine run different code
and only one of them is the one you test.

The costing behind it: the deterministic layer is 39 subcommands answering in 0.2–2.2s with
JSON out and zero LLM calls, so most of what feels like "ask the system a question" needs
no agent. What genuinely needs one is artifact-shaped and expensive — the cheapest routine
is `coach-prose` at ~54.5k tokens, `candidate-pool` is ~235k. Those are jobs, not chat
turns.

### Similarity comes from the function space, always

`embeddings.npy` is the **layout** space (colour and type) and feeds `projection_2d.json`
only. `embeddings_ability.npy` is the **function** space and is the sole source of
similarity — Find Similar, the walk and drill all read it regardless of which map is
displayed. Held-out scores for the function space: 27.87 effective dimensions,
recall@10 0.245, recall@50 0.455, median rank 78.

Similarity is exactly `0.7·cos_learned + 0.3·cos_text` by fixed weight, so the model cannot
discard the text it was built from. The two halves are complementary: alone they score
0.136 and 0.219 recall@10, together 0.245 — better than either, and better than the full
384-dim frozen text.

**Do not tune on the golden set.** At ~50 dev and ~160 test queries, everything in
W ∈ [0.15, 0.6] is inside noise, and the two splits disagree about the optimum. W=0.3 was
chosen a priori and fitted to neither.

Still open: neighbour spread is 0.0315 against a 0.05 target, held as a failing
`xfail(strict=True)` rather than lowering the threshold to match the result. Hard-negative
mining is the obvious next lever and needs a similarity ceiling, since 39% of cards have a
text neighbour above 0.75.

### Synergy is complementary, not similar

The synergy graph is 24 rules over mechanical tags (blink → ETB). Partners are ranked by
**playability**, not embedding similarity — ranking by similarity surfaces cards resembling
the anchor rather than cards that play with it.

Coverage is uneven and the UI says so: similar 100%, synergy 76.1%, obsolescence 22.5%, and
23.6% of cards have nothing but similar. Doubling Season has no synergy partners at all —
a real hole in the rules, visible rather than buried.

Two measured findings worth keeping: **"anti-cards" do not exist** (across 4.5M pairs the
minimum cosine is +0.344, median +0.714 — the space is a narrow positive cone, so
"orthogonal" is not a place), and **a card whose top synergy tier is small cannot be
rescued by re-ranking** — that is coarseness in the rules, not in the ordering.

### `meta-analyst` is deferred

The design point worth keeping: **meta claims perish and strategy theory does not**, so
they want separate corpora with different invalidation, and every meta section needs an
`as_of`. `deck_recon.json` follows that rule already — it stays out of `strategy.md`.

## Judgment calls, deliberately not changed

- `goldfish.py` — `mean_cast_turn` divides by games where the commander was cast;
  `cast_by_turn_6_rate` divides by all games. They sit side by side in the Commander File.
- `goldfish.py` — `cast_by_turn_6_rate` hardcodes turn 6 while `GOLDFISH_MAX_TURN` is
  configurable.
- `power_creep.py` — the docstring numbers eight criteria; the inline comments number seven.
- `download_combos.is_up_to_date` checks file existence only, so combo data never refreshes.

## Invariants that must not erode

- Only checker-passed stacks publish; failed artifacts are kept as open questions.
- Agents return JSON and never write HTML — that is what keeps rebuilds byte-identical.
- `issue.json` is authored, never generated.
- Costume never earns the badge: a section cannot claim a tier it was not granted.
- Record the cache **after** validation, never before.
- Charter edits invalidate before they inform — make them **before** `cache-record`.
- A bracket **floor** is what the contents are consistent with, never a verdict.
- The deterministic builder must always produce a complete legal 99 with no agent involved.
- **Never transcribe the section list or its count into a prompt** — read
  `issue_spec.DEPARTMENTS`. A list in prose goes stale; the spec cannot.
- **Count copies, not decklist entries**, for anything the shuffler would see.
- `--out` on a per-deck command must be slug-scoped; concurrent agents share one scratch
  directory and a generic filename silently swaps one deck's numbers for another's.
