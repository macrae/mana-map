# PLAN — current state and what's next

*The resume-here doc. Read `README.md` for orientation, `CLAUDE.md` for gotchas, this for
what shipped and what's still open. Completed plans live in `docs/history/`.*

Last updated 2026-07-29. All work below is committed, pushed, and deployed.
Every figure here was derived from the repo at write time, not remembered.

## What this is now

Two products in one repo. The card map is stable and complete. The active work is
**Pilot's Manual** — a magazine generator that turns one Commander deck into a
self-contained web issue with a three-tier evidence contract — and, since Deck Building v2,
a **deck builder** that produces the deck in the first place.

**Seven issues live**, all rebuilt 2026-07-29 under the v3.3 process:
[001 Goblin Storm](https://macrae.github.io/mana-map/manuals/goblin-storm.html) ·
[002 Hapatra](https://macrae.github.io/mana-map/manuals/hapatra.html) ·
[003 Sisay](https://macrae.github.io/mana-map/manuals/sisay.html) ·
[004 Heliod](https://macrae.github.io/mana-map/manuals/heliod.html) ·
[005 Ur-Dragon](https://macrae.github.io/mana-map/manuals/ur-dragon.html) ·
[006 Edgar](https://macrae.github.io/mana-map/manuals/edgar-vampires.html) ·
[007 Gishath](https://macrae.github.io/mana-map/manuals/gishath.html) ·
[newsstand](https://macrae.github.io/mana-map/manuals/index.html)

926 tests. 33 `manamap pilot` subcommands. 12 agents, 15 skills.

## Shipped

| Layer | What |
|---|---|
| **Rules DB** | CR chunked one-per-rule (~3.9K), chunk ID = citation ID; semantic query + exact lookup |
| **Citation contract** | Form enforced in `validate_stack.py`; meaning by adversarial `rules-checker`; only `pass` publishes |
| **Goldfish** | Seeded Monte Carlo (seed 42, 10K iters), resource development *not* full games, assumptions rendered |
| **Strategy DB** | 45 sourced sections / 14 pillars; `strategy:<id>` = citation ID; `strategy-researcher` (research + consult modes) |
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

## Open

### Frontend

**Frontend v2 is 0% started** and `docs/frontend-v2.md` predates the v3.2/v3.3 magazine
work, so its M3 targets a component set and a 15-department layout that no longer exist.
Audited state: no `viz_index.json`, no `viz/js/engine/`, no Worker, no `build.html` or
`deck.html`, no URL state anywhere, no CSS custom properties. The viz is four files,
~3,000 lines, and knows nothing of the pilot subsystem — the two products share exactly
one link, newsstand → map.

Two live defects in `viz/js/deck-builder.js`, both confirmed:
- its six-factor scorer **diverges from `config.DECK_BUILD_WEIGHTS`** — different weights
  on every shared factor, and its sixth factor is keyword Jaccard where Python uses
  castability;
- saved decks persist **raw projection row indices**, so any pipeline refresh silently
  reinterprets a saved deck as a different set of cards. A correctness bug.

**Resequenced from the original M1 → M2 → M6.** The deck artifacts are already tracked and
servable (~3.6 MB, uniform across seven decks), so `deck.html` has **no prerequisites**,
while the engine port is blocked on `data/cards.csv` being gitignored. Order:

**`viz/deck.html` shipped 2026-07-29** — the dossier is the surface the rest builds on:
it forced the design-token port into existence against real content and introduced the
first URL state this frontend has had (`?deck=<slug>`). What remains, in order:

1. **`viz_index.json`** — the next step. The browser is missing four card fields
   (`game_changer`, `mechanical_tags`, `layout`, and `legal_commander` as a tri-state);
   a positional file in cards.csv row order closes every gap. ~476 KB gzipped.
2. **Engine port to a Worker** — `manabase` is trivial (pure math, no deps), `bracket` and
   `goldfish` easy, `build_deck` hardest (pandas is load-bearing). `constants.js` must be
   **generated from `config.py`**, with a parity test, or the scorer divergence recurs.
3. **`build.html`** — slots as the primitive, the deck starts complete, incremental DOM.
4. **Handoff** — emit `brief.json` + `decklist.txt` + the command to run.

**Agents run in Claude Code, not the browser.** The integration model is: render what
agents produced, hand a brief back, and deep-link cards between map and magazine. A
browser-triggered agent run would need a server this project deliberately doesn't have.

Opportunistic: hover tooltips are built for all 34,322 points every render and thrown away
(13 traces set `hoverinfo:'none'`); the detail panel hides in build mode, exactly when
you're deciding; int8-quantising `embeddings.bin` takes 17.6 MB → 4.4 MB.

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

### Deferred

**`meta-analyst`** — the v2 design called for a meta-awareness agent with its own
`data/meta/` corpus. Traded away to get the loop working, and the loop works without it.
The design point worth keeping: **meta claims perish and strategy theory doesn't**, so they
want separate corpora with different invalidation, and every meta section needs an `as_of`.

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
