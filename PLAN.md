# PLAN — current state and what's next

*The resume-here doc. Read `README.md` for orientation, `CLAUDE.md` for gotchas, this for
what shipped and what's still open. Completed plans live in `docs/history/`.*

Last updated 2026-08-09. All work below is committed, pushed, and deployed.
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

1,289 tests (1,179 fast + 110 browser). 35 `manamap pilot` subcommands. 12 agents, 15 skills.

> ### The cache board reads 15 MISS, and every one is honest
>
> *(Resolved 2026-08-09. This callout previously described 23 MISSes from the embedding
> rebuild, then 73 from a charter edit. Both are settled; the history is in the hygiene
> plan's D1 below.)*
>
> **15 non-HIT across 8 decks**, and none of them is a bookkeeping artefact:
>
> | routine | n | why |
> |---|---|---|
> | `deck-recon` | 6 | never run — absent, not stale |
> | `deck-diagnosis` | 5 | never run — the withheld diagnoses |
> | `deck-diagnosis` | 3 | inputs genuinely moved (`goldfish_targets`, `mana_analysis`, `deck_audit.py`) |
> | `candidate-pool` | 1 | never run (yawgmoth) |
>
> Everything else HITs. The 58 that missed on a **charter edit alone** were re-recorded
> with the reasoning committed: `af7ded9`'s diff replaced "name your scratch file after
> the deck" with "use `--out <dir>/`", which changes no analytical instruction, no
> threshold and no schema, and therefore cannot move a figure. All eight validators pass
> on all eight decks, checked *before* recording.
>
> **The rule is unchanged and was not bent:** never `cache-record` to make a board green.
> The three `deck-diagnosis` entries whose real inputs moved were deliberately left MISS,
> and the twelve never-recorded routines have no artifact to claim.


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
| **The Short List** | `considering.json` (all 8 decks) — exactly ten cards scouted from the whole pool, ownership-free, `validate-considering` enforcing the count and every evidence claim. The last survivor of the retired sideboard apparatus |
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
| `short-list-analyst` | **The Short List** — ten cards worth knowing about, pool-scouted, analysis-only |
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
                     → /short-list               (the Short List)
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

## NEXT SESSION — tighten what 2026-08-09 exposed

That session set out to re-run seven withheld diagnoses and finished that in its first
stretch (1 pass, 6 withheld). Everything after was scope approved a step at a time: the
tooling bugs the run surfaced, the sideboard retirement, 46 re-authored prose routines,
and a scratch-clobber fix. It is all shipped and green — this section is the tail it
left, ordered by value, with the reason each one is worth doing rather than a wish list.

**Start here — re-run the six withheld diagnoses.** They were judged against numbers
this branch has since corrected, so several were arguably failed on bad inputs:
gishath's skeptic killed its diagnosis over an 82.4% enrage figure that is now
**60.8%** measured against the five cards that actually convert damage; hapatra's
prescription was sized against a 9.5% Mikaeus loop that checker-passed stack 001
refutes outright. The loop is `/diagnose-deck` per deck, ~250k each, ~1.5M for six.
Do it as the session's GOAL, not as a tail — the last one inherited it at the end of a
long session and that is why it never happened.

| Thread | Why it is worth doing | Size |
|---|---|---|
| **Six withheld diagnoses** | judged against figures since corrected | ~1.5M |
| **Stack artifact staleness** | 12 files carry `VERIFIED PRE-SWAP … HISTORY.md` preambles that violate L10, reference the deleted `sideboard-facts`, and say "benched" for a dead concept. edgar's presentable stacks 004/006 cite non-presentable 001 and Exquisite Blood 11 times, and it reaches the page. **DO NOT REGEX IT** — a dry run ate substance on four files (edgar 006 loses "with maindeck Bloodthirsty Conqueror — the third gain/loss"). Needs a per-file read, and it invalidates `stacks:passing` fleet-wide, so do it in one pass and re-record. | ~200k |
| **Grafdigger's Cage stack** | three independent agents named it the fleet's highest-value unresolved line. Three of yawgmoth's four kills and most of its political advice rest on an oracle reading no checker has settled. | ~120k |
| **Six bracket targets** | seven of eight decks compute a floor but never answer "is this deck inside its bracket". Not lost — never declared. yawgmoth's was recovered from its own `brief.json`. **This is a decision, not work**: a target bracket is the pilot's statement of intent and must not be inferred from a floor. | you |
| **gishath's unmeasured conjunction** | its frame flagged that the enrage web is *five converters × a damage source* and no target measures the joint. Unlike the three win lines declared on 2026-08-09, no checker-passed stack names it, so declaring it is an authoring judgement — it belongs in a diagnosis pass, not a data edit. | with the diagnosis |
| **`deck-recon` × 8** | never run on any deck. Absent, not stale — a feature nobody is waiting on. Needs dated web passes and `RECON_MAX_AGE_DAYS` judgement. | ~600k |

**Three validator gaps are still open** and each caused a real defect, so each is
mechanically checkable and cheap:

- an add's `closes` must move the axis — FIXED for `(cards, roles)` axes; `colour-sources`
  and `mana-base` still cannot be checked because they need `mana_analysis`;
- `upgrades` prose must name the current ten. Regenerating `considering.json` orphaned
  that paragraph on five of eight decks — hapatra's narrated a completely different list
  above the one on the page. The check that works is *card names from the corpus in the
  copy that are in neither the current ten nor the 99*; overlap counting and numeric
  checks both missed it;
- a routine's staleness check must look at **labels and members**, not only figures. A
  grep for moved numbers called gishath's frame clean while it discussed the "Enrage
  engine" four times and Temple Altisaur seven, with that group's membership just
  changed.

## THE HYGIENE PLAN — optimize, clean, document (drafted 2026-08-09, not yet started)

Three parallel surveys (Python health, docs drift, tests/data/frontend) plus direct
verification produced this. Every figure was measured at draft time; where a survey claim
was checked and found wrong, the correction is stated. **Zero agent spend** — all phases
are deterministic work, roughly two focused sessions.

### Ground truth at draft time

18,646 lines Python · 8,541 lines JS · tests 18,598 lines (~1:1 with source) ·
**1,469 tests** (1,346 fast @ 58s single-proc, 123 browser) · 38 pilot subcommands ·
14 agents · 16 skills · 10 cached routines · 8 published decks · 131.8 MB tracked data ·
`.git` 208 MB · cache board 73 MISS *(now 15 — see D1 and the callout at top)*.

### The defects, ranked

1. ~~**Cache board at 73 MISS**~~ — **fixed.** My charter-edit-after-record sequencing
   error; 58 re-recorded, 15 left honestly MISS. See D1.
2. ~~**`impact --out` is accepted and silently ignored**~~ — **fixed** in `a831162`.
3. **Docs state four different test totals across five files**; the board is called
   "green" in 2 places and "23-MISSed" in 4 — none matches reality; `cli.py --help` says
   "13-step pipeline" against a 15-step registry; **yawgmoth (Vol. 008) is absent from
   this file's issue list, deck table and inventory**; "hapatra has no decision spreads"
   and "deck-recon never run" (both above) are false.
4. **Seven independent `cards.csv` parsers in the pilot layer** (24.65 MB/parse):
   `card_pool` (unmemoized full parse), `bracket`, `pool_facts`, `validate_build`
   (unmemoized), `validate_stack`, `validate_diagnosis`, `build_deck` (full 35-column
   frame, twice in one process; build→validate = 3 parses). The `card_flags` dict is
   built by identical comprehensions in 3 files; `pool_facts.py:358` carries a comment
   begging for the shared loader.
5. **Two unkeyed memos are stale-data hazards** — `validate_stack._CORPUS_NAMES`,
   `validate_diagnosis._ORACLE_MEMO` lack the `(mtime_ns, size)` key every other memo
   uses. Five hand-rolled implementations of the same memo pattern exist.
6. **~73s floor of unconditional `wait_for_timeout` sleeps** in 24 browser tests; 62 of
   118 browser tests re-parse the 12.9 MB projection per test (function-scoped pages).
7. **Dead weight**: 6 zero-caller functions (`analysis/common.cosine_similarity`,
   `viz_index.similar_rows`, `mechanical_tags.tag_oracle_text_from_row`,
   `deck_audit._roles_for`, `design.map_key`, `agent_cache.file_sha256`); 14 unused
   imports (one costs a pandas import); **63 of 212 CSS classes (~30%) referenced
   nowhere** (remains of three deleted features); 5 duplicate test names;
   `mana-map.js`'s docstring says it exists to serve deleted `deck-builder.js`;
   `build_deck.py:261` cites `deck-builder.js:embeddingSim` (deleted);
   `common.py:108` cites deleted `validate_sideboard.py`; permanently-no-op probes for
   `sideboard_analysis.json`/`upgrade_watch.json` in `impact.py:94` +
   `validate_issue.py:281`; 179 stale `.agent-out` files (3.4 MB); one merged branch.
8. **Pattern fragmentation**: 20 DFC face-split sites in 3 semantics with no helper
   (one layout-aware, its twin not); 5 JSON-writing styles; canonical-JSON implemented
   twice ("local copy avoids a circular import"); `validate_stack` is the only validator
   not using `report_errors`; out-path guarding covers 3 of 8 `--out` commands
   (`diagnosis_report` hand-rolls a guardless copy).

### The sequencing constraint that shapes everything

The cache hashes three source files — `issue_spec.py`, `deck_audit.py`, `STYLEv3.md` —
plus every `.claude/agents/*.md`. Therefore: **all edits to cache-hashed files batch
into ONE phase** closed by exactly one snapshot → verify-artifacts-unchanged →
re-record; **docs are written LAST** so they describe the end state once. `common.py`,
`card_pool.py`, `config.py`, `build_manual.py`, `viz/**` are not hashed — free anytime.

### Phase 0 — three decisions (pilot, before any work)

All three decided 2026-08-09: **take the recommendation in each case.**

- **D1 — the 73-MISS board** ✅ **DONE.** Re-recorded with the reason documented — but
  only **58 of the 73**. The board was never one thing, and framing it as one was this
  plan's error. Classified by the cache's own declared cause: **58 charter-only** (the
  only changed input is a `.claude/agents/*.md`), **12 never recorded** (`changed=[]`,
  so there is nothing to re-record and no artifact to claim), and **3 real input
  changes** where `goldfish_targets.json`, `mana_analysis.json`, `deck_audit.py` and
  `cards_semantic` genuinely moved. Only the 58 were recorded. The 15 that remain are
  honest: `deck-recon` ×6 and `deck-diagnosis` ×5 have never run, and the last three
  `deck-diagnosis` missed on inputs this cleanup itself changed — re-recording those is
  exactly the act the repo forbids.
- **D2 — count-drift guard**: build `tests/test_docs_counts.py` on the proven
  `test_docs_section_count.py` pattern. **Lands in Phase 4 by necessity** — a test that
  fails on any doc stating a wrong count would fail instantly today, so the docs must be
  true before the guard goes up.
- **D3 — `config.py` split** (1,235 lines, 8 unrelated concerns: paths, a binary format
  spec, ML hyperparams, three regex rulebooks, UI display strings, editorial citations,
  the routine graph): split into a `config/` package with `config.py` as a pure
  re-export façade so no import breaks. Own commit; full suite as gate. Phase 5.

### Phase 1 ✅ DONE (2026-08-09) — correctness + dead weight

Shipped in `a831162` and `e5e6b38`. `impact --out`, `diagnosis-report` and
`scenario-facts` now go through `resolve_out_path`, which grew an `ext` argument
because diagnosis-report emits markdown and a directory auto-naming its report `.json`
would lie about the bytes; `cache-snapshot` and `pool-facts` stay unguarded on purpose
and now say so in the registry · `common.mtime_memo` replaces five hand-rolled copies,
two of which were truthiness-gated and therefore blind to a rewrite · five dead
functions gone (`deck_audit._roles_for` waits for Phase 2 — cache-hashed) · `cli.py`
derives its step count from the highest declared step NUMBER, not `len(STEPS)`, since
4a/4b make 16 entries 15 steps · `merge-prose` promoted, verified by reproducing
heliod's tracked `manual_prose.json` byte-for-byte from its two handoffs · merged
branch deleted.

**1,369 fast tests (+23) and 123 browser, green; no tracked artifact moved.**

Two departures from the plan as written, both deliberate:

- **42 unused imports, not 14** — the survey scanned only `src/`. Three were a trap:
  `page`, `discover_page` and `canvas_page` in the browser suite are pytest FIXTURES,
  invisible to a static check and load-bearing for 118 tests. They keep their `noqa`
  and now carry the *reason*, since a bare `noqa` does not stop the next reader.
- **The 179 `.agent-out` files were NOT deleted.** The plan called them stale working
  files; they are the provenance of `manual_prose.json`, and `merge-prose`'s strongest
  test reads them to prove the merge reproduces the tracked artifact exactly. Deleting
  3.4 MB of gitignored files to lose a verification is a bad trade. Revisit only with
  a rule that distinguishes a current handoff from debris.

### Phase 2 — shared infrastructure (the perf phase; the ONE phase touching `deck_audit.py`)

Grow `pilot/card_pool.py` into the single `cards.csv` reader — `(mtime_ns, size)`-
memoized, union of needed columns, serving the three consumed shapes (name-keyed dicts /
a DataFrame for `build_deck`'s positional alignment / flat name-set with DFC faces);
port all seven pilot readers; delete the three `card_flags` comprehensions; kill
`build_deck`'s second same-process parse (pipeline steps stay — one parse per one-shot
process is correct) · DFC helpers in `common`: `front_face(type_line)`,
`front_face_name(name, layout)` (fixing `pool_facts`'s layout-unaware twin),
`expand_faces(name)`; port the ~20 sites · `common.write_json(path, doc, tracked=True)`
(canonical tracked style vs compact pipeline style); move canonical-JSON to `common`,
deleting `card_refs`' circular-import copy · fix `impact.py`'s double parses ·
`validate_stack` adopts `report_errors` (exit semantics kept) · `deck_history`'s
`DECKS_DIR.parent.parent` → `_REPO_ROOT` · `build_name_index` ×2 with opposite
tie-breaks is **deliberate per consumer** — cross-reference, don't unify.
**Gate**: fast-suite time (baseline 58s — the validate cluster should drop);
`build-deck`+`validate-build` wall time before/after; artifacts byte-identical; then the
single snapshot → re-record.

### Phase 3 — frontend + browser suite

Delete the 63 dead CSS classes (browser suite before/after as the gate) · dedup: `esc`
(deck-view.js → `MM.escHtml`), the tray trio (Session is the owner; discovery.js keeps
wrappers or loses them), the colour-ordering literal · **`setCommander` ×3 needs
investigation before touching** — CLAUDE.md's gotcha says it must be written wherever a
commander is learned; the three may be deliberate layers (Session=truth, Force=ring) ·
fix comments naming dead files as live (`mana-map.js:4`, `discovery.js:485/629`); KEEP
the "Plotly is gone" tombstones — they are load-bearing warnings · replace the 48 hard
sleeps with condition waits where a real condition exists, measured per test; consider a
module-scoped page for the 62 projection-loading tests only if that alone misses the
budget (page reuse risks cross-test state, this repo's known enemy) · dedupe the two
cache-bust assertions.
**Gate**: browser suite green and timed; visual spot-check after CSS deletion.

### Phase 4 — the documentation truth pass + guards (LAST, one commit per cluster)

**This file**: the Decks table gains yawgmoth's row and drops the bench-split "Ten"
column; roster 12→14 (+`deck-doctor`, +`deck-skeptic`); "seven issues"→8 at all 9 sites;
artifact inventory (both `build_plan.json` holders; all 8 have `decisions/`; add
`diagnosis.json`); "cache board is green" §rewritten; 32→46 verified lines; the false
"hapatra has no decision spreads" and "deck-recon never run" rows; "8 static routines"→10.
**CLAUDE.md**: counts; layout tree +`card_pool.py` +`validate_goldfish_targets.py`; fix
the DIAGNOSE/PUBLISH heading boundary. **docs/testing.md**: full re-table (14 drifted
counts, 19 missing files, the internal 80-vs-123 contradiction). **docs/pilot.md**:
30→38 commands, 8→10 routines, test inventory. **docs/pipeline.md**: "thirteen"→15,
triplet→InfoNCE at step 4b, 5→6 invalidated routines. **docs/architecture.md**: the
:40-45 triplet description contradicts its own :77-86 — fix the former.
**docs/data-artifacts.md**: region sizes/counts, obsolescence 2.9 MB, eval step 15,
DATA map 9→12. **README.md**: 8 issues + Vol. 008 link, 26→38 subcommands, delete the
"six-factor recommender" claim (that scorer is deleted), test counts. **docs/viz.md**:
six-of-eight dossiers, 8-deck picker. **docs/agent-cost.md**: board state, 7→8 spawns,
5.9→2.9 MB. **docs/frontend-v2.md**: extend the audit header — `deck-builder.js` no
longer exists. Ship the D2 guard test. Correct this file's own SentenceTransformer
claim: it **is** memoized per-process (`preprocess._MODEL_CACHE`); the cost is
per-invocation because each CLI call is a fresh process — interactive RAG needs a
long-lived process, a decision against the no-server stance, not a memoization task.
Also: `show_progress_bar=True` leaks a progress bar into `--json` output — off for
single-text queries.

### Phase 5 — optional, each with its own go/no-go

`config.py` package split (D3) · `build_manual.py` untangling — it contains a second,
independently written card-name matcher beside `card_refs.py` plus a module-global
mutable link registry; unify on `card_refs`, and collapse the
`render_upgrade_watch`/`render_short_list` double name (the department **id** stays
pinned) · int8-quantise `embeddings*.bin` (17.6→~4.4 MB each — the two largest tracked
blobs and the Pages payload) · `.git` at 208 MB (four ~27 MB revisions of
`synergy_graph.json` alone): **leave it**; history rewriting breaks every clone for
~76 MB · the `embeddings.bin`/`.npy` duality is intentional (browser export vs pipeline
working format) — document, keep.

### Do-not-touch (verified during drafting)

**`share/` is load-bearing** — `deck_history._owned_index` reads it for ownership
derivation; a survey called it an orphan and was wrong (the path is built via
`DECKS_DIR.parent.parent`, which greps miss) · frozen config (`MECHANICAL_TAGS`,
model-facing dims) · stack preambles (separately planned per-file work; a regex dry-run
ate substance on four files) · `SIDEBOARD_SECTION_MARKERS` (both parsers must keep
consuming the marker) · `build_name_index` tie-breaks (opposite by design) ·
`upgrade-watch` department id (pinned by `validate_issue`, the act table, every manual).

### Ordering against the diagnosis thread

The six withheld diagnoses (NEXT SESSION above) are orthogonal. If they run first,
Phase 2's `deck_audit.py` edits wait until after their re-record; if hygiene runs
first, the diagnoses inherit a faster, cleaner audit. Either order works — what is
forbidden is interleaving Phase 2 with a diagnosis run mid-flight.

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

**The renderer port is done — Plotly is deleted.** The map draws on `<canvas>` + d3 and
there is no second path: no CDN tag, no `?renderer=` flag, no `_fullLayout`. The port ran as
a strangler behind the `MM` contract with *the layer format as the trace format*, so there
was no adapter to delete at the end — `render()` still builds one structure, it just has one
consumer. Measured across the port: `render()` 30 ms → 15 ms, box-select 138 ms → 4.5 ms per
mousemove.

The last phase was mostly deletion: the `keepX`/`keepY` camera-preservation dance (a layout
with no explicit range silently autoranged, so filtering and zooming were mutually
destructive — the hazard left with the renderer rather than being ported),
`getRegionAnnotations` (40 lines duplicating `refreshCanvasLabels`, still computed every
render and thrown away unread at the fork), `refreshLabelsOnZoom` and its re-entry guard,
~80 lines of hand-rolled pinch-zoom written because `scattergl` has no native pinch (`d3.zoom`
does touch itself), the four scattered 260 ms resize timers, and `drillTraceIndex`.

**Keeping it behind a flag cost three real bugs**, all invisible because canvas was opt-in:
`updateSelectionHighlight` opened by reading `plotDiv.data` — Plotly's trace array — so on
canvas it returned at the first line and selecting a card never repainted the highlight; the
region-label click emitted `{regionId, row:null}` into a handler that read only `ev.row`, so
it called `addToSelection(null)` and threw; and `deck-builder.js` carried two unguarded
`Plotly.Plots.resize` calls that would have thrown the moment the tag went. A renderer kept
behind a flag is a renderer nobody is testing.

`tests/test_viz_camera.py` was **retired rather than ported** — all three of its assertions
were about the Plotly layout hazard, and the invariant they protected (the camera survives a
re-render, except on a map switch) was already covered behaviourally in
`test_viz_behaviour.py`, where it always belonged. Its reasoning moved there as a comment.
The other three source-assertion suites were ported; one of them failed the canvas handler
for using `if/else` where Plotly used an early `return` — the same guarantee, and a good
illustration of a source check failing a correct refactor.

**The 39 browser tests are the real gate here.** A perf commit once stripped a variable
declaration and left its use behind, breaking drill mode on every render; all 13
source-assertion drill tests passed and it shipped. Playwright caught it in both
directions when pointed at that revision. Nothing that renders should be verified by
grepping source again.

**Both live defects in `viz/js/deck-builder.js` are closed, by deleting the file.** The
six-factor scorer diverged from `config.DECK_BUILD_WEIGHTS` on five of six factors — two
had no counterpart at all — and cost ~50 MB of lazy downloads to be wrong differently from
the pipeline. Saved decks persisted raw projection row indices with no schema version, so a
refresh that reordered `cards.csv` silently reinterpreted a saved deck as different cards.
Evaluation now comes from the sub-agent routine via the exported brief; suggestions come
from the precomputed relations. See *Shipped — three modes* below.

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
| **Relations** | Similar / Synergy / Outclassed by, counts stated **before** the click, rendered in every panel via `MM.relate` — always growing the graph, so a click in Explore carries you into the walk seeded on that card |
| **Graph** | branch to grow, drag to fling, cross-links so it is a graph and not a tree, relation-inked edges, synergy edges labelled with their rule |
| **Decks** | load any of the seven by slug (commander ringed) or paste a Moxfield export; deck cards read differently from cards you found |
| **Tray** | keep cards, export a brief for the pilot loop in Claude Code — the site stays static |

**Explore became a launchpad, not a second mode.** `MM.relate` used to fork — graph modes
branched, Explore opened a linear browse set — on the reasoning that a scatter plot cannot
grow. Sound reasoning, wrong result: one control meant two things, so the atlas felt dead
next to the walk. It now always grows, switching modes to do it. The **Keep** control moved
into the same shared card HTML for the same reason. Box-select still opens a browse set,
because "what did I just lasso?" is a genuinely different question.

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

## Shipped — three modes, and the loop closed

The frontend went from five modes to three: **Discover**, **Explore**, **Build**.

**The Walk was deleted.** It was Discover with different chrome — four `chrome ===` reads
amounting to two behaviours and a status string. But its *panel* was the only home for six
things, so this was a port, then a delete: the scoreboard, Fit, Reheat, Start over, the
trail, the truncation notice (never shown in Discover, so a >500-card import was silently
cut), and **region seeding** — `renderEmptyState` was the only region→graph path in the
app, since `Drill.enterRegion` re-embeds into the atlas and never seeds the graph.

**Deck Lens and Build Deck merged into `viz/js/build.js`.** They were halves of one
activity. `build.js` is deck-map.js grown, not deck-builder.js kept: the Lens half owned
`card_roles.json` in the browser, the role histogram, the Short List, verified lines, the
copies-vs-dots discipline and `dimsAll()`'s scalar fast path. From the builder it took only
colour identity, format legality, the curve and the colour load. **`?deck=<slug>` now lands
in Build** — an inbound contract from the dossier and every published manual.

Build opens on the **graph**; the map is a view toggle. Measured before committing to that
default: a real 251-card pool renders at 8.6% ink coverage with legible cluster structure,
and `Force.fit`'s 1.6× cap does not bite at that size (fit lands at k≈0.19).

**The brief IS `brief.json`.** `Discovery.brief()` emits the exact shape
`pilot/build_deck.py:load_brief` reads, round-trip verified through Python. Colour identity
rides in a `_manamap` block as information because it is *derived* from the commander,
never authored; budget says it is unsupported rather than approximating it.

Net ~1,900 lines deleted. Six bugs found and fixed on the way, four of them introduced
during the work itself — the pattern in every case was **two places responsible for one
decision**: mode CSS naming Plotly elements after Plotly was gone; the quadtree signature
blind to positions `updateLayerBy` mutates; `clearSelection` meaning both "clear" and
"peel"; panel ownership collapsed to one owner the moment a second appeared.

## Decided — the frontend stays LLM-free (2026-08-01)

A conversational agent UI inside Build was scoped and **declined**. The reasoning is worth
keeping, because the question will come back.

**The decision:** the deployed static site and the local checkout run the same code. The
frontend is exploration plus artifact reads; the agent loop stays in Claude Code, reached by
the exported brief. No local bridge, no LLM provider in the browser.

**What the costing found:**

- The repo has **two disjoint layers with a human as the only bridge**. The deterministic
  layer is 35 CLI subcommands, 1.5–9s, JSON out, zero LLM — `deck-facts`, `bracket-check`,
  `manabase`, `goldfish`, `impact`, and `query-rules`/`query-strategy` as pure local RAG over
  a MiniLM index. The judgment layer is 12 agent charters that only a Claude Code session can
  spawn.
- **Agent spawns are not chat.** Cheapest routine is `coach-prose` at 54,515 tokens;
  `candidate-pool` is 235,579; stack resolution is 2–6 serial spawns with one measured
  outlier near 600k. They write whole artifacts and return a path, and the cache refuses to
  record anything without a validated artifact — `docs/agent-cost.md` already says ad-hoc
  consults have nothing to key against.
- So the "fast, conversational" half of the ask was mostly answerable **without an LLM**, and
  the half that needed one was minutes-long and artifact-shaped. Those are two different
  products, and only one of them belongs in a static page.

**If it is revisited**, the honest version is: make the deterministic commands the interactive
surface (memoise the SentenceTransformer first — it is reconstructed per call, ~8.6s cold, and
is the single change that makes RAG interactive), and keep agents on the brief handoff.

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
3. ~~**`build.html`**~~ — **superseded.** Build is a mode, not a separate page, and its
   premise died with the in-browser scorer: "review the scored alternates" needs a scorer,
   and that one diverged from the pipeline on five of six factors. Reviewing and swapping
   slots is worth building, but against the *pipeline's* scores arriving through the agent
   loop, not a second implementation in the browser.
4. ~~**Handoff**~~ — **shipped.** `Discovery.brief()` emits the exact shape `load_brief`
   reads, round-trip verified through Python. `decklist.txt` is still not emitted; the
   brief plus `/build-deck` produces one.

Opportunistic, none blocking: ~~hover tooltips~~ **done** — hovering shows the card image at
the cursor in every mode, without Plotly's per-point text (`showCardPopup`); the detail panel
stays open in Build, because clicking a lit card to read it is the whole interaction (it
used to hide, which is part of why adding a card felt like filing); int8-quantising
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

### The cache board, and why it is green

The 23 MISSes left by the embedding rebuild were **re-blessed**, not re-spawned. The
reasoning, so the record is a judgement rather than a shrug:

- What changed: `synergy_graph.json` and `obsolescence_index.json` were regenerated.
- What MISSed: `writer-prose`, `the-ten`, `issue-plan` (×7), plus hapatra's
  `candidate-pool` and `deck-build`. Prose and packaging.
- What did **not** miss: every `stack:NNN`, `strategic-frame`, `coach-prose`. Nothing
  rules-verified depends on those graphs, which is the whole reason the tiers exist.
- Re-spawning was ~2.46M tokens to re-derive body copy that cites no synergy rank.

`the-ten` is the one with a real claim on it — a Short List *is* a ranking, and the synergy
graph's median partner rank moved 9,397 → 737. It was blessed on the judgement that the
published tens were curated against evidence the analyst cited individually, not against
the ranking wholesale. **If that turns out wrong, `the-ten` is the routine to re-spawn
first**, and `--force` is how.

### Known-wrong, found and left alone deliberately

~~Verified-line edges string-matched from scenario prose~~ — **fixed.**
`build_index.py:line_cards` derives them from the scenario's structure and the manifest
carries them per stack file, so the browser stopped guessing. The rule that gets both
shapes right: **if the stack and hand name the line, `board` is context; if they name
nothing, the line is on the board.** heliod's Approach scenario has its line in hand and
its furniture on the board (Ancient Tomb, Howling Mine — both were being drawn as
"verified" edges while Swan Song, the real interaction, was cut by a cap that truncated in
name-length order). edgar's combo loops are the mirror image: the pieces are already
resolved permanents and the stack object is a prose sentence about a trigger. Deriving in
Python also left the tracked stack artifacts untouched — adding a field to them would have
changed their digests and invalidated every agent-cache routine on the deck. Result: 29 of
32 published lines now draw, with the right cards.

~~`Force.fit` caps zoom at 1.6×~~ — **fixed.** The cap is now the zoom behaviour's own
ceiling (12), so a fit may go wherever a drag could. It bit hardest on the commonest state
in Discover: a landing card plus one branch, 7 nodes spanning 20.7×26 world units, wanting
k≈19 and allowed 1.6 — so the graph drew ~33px wide and every label collided. The original
reasoning ("blown up is not more readable, just bigger") was wrong here because node radius
and label text are drawn in SCREEN space; zooming in enlarges nothing, it only spreads.

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
