# PLAN — current state and what's next

*The resume-here doc. Read `README.md` for orientation, `CLAUDE.md` for gotchas, this for
what shipped and what's still open. Completed plans live in `docs/history/`.*

Last updated 2026-07-26. All work below is committed, pushed, and deployed.

## What this is now

Two products in one repo. The card map is stable and complete. The active work is
**Pilot's Manual** — a magazine generator that turns one Commander deck into a
self-contained web issue with a three-tier evidence contract — and, since Deck Building v2,
a **deck builder** that produces the deck in the first place.

Live: [Vol. 001 — Goblin Storm](https://macrae.github.io/mana-map/manuals/goblin-storm.html)
· [Vol. 002 — Hapatra](https://macrae.github.io/mana-map/manuals/hapatra.html)
· [Vol. 003 — Sisay](https://macrae.github.io/mana-map/manuals/sisay.html)
· [newsstand](https://macrae.github.io/mana-map/manuals/index.html)

## Shipped

| Layer | What |
|---|---|
| **Rules DB** | CR chunked one-per-rule (~3.9K), chunk ID = citation ID; semantic query + exact lookup |
| **Citation contract** | Form enforced in `validate_stack.py`; meaning by adversarial `rules-checker`; only `pass` publishes |
| **Goldfish** | Seeded Monte Carlo (seed 42, 10K iters), resource development *not* full games, assumptions rendered |
| **Strategy DB** | 45 sourced sections / 14 pillars, including a `strategy:deckbuilding` pillar; `strategy:<id>` = citation ID; `strategy-researcher` agent (research + consult modes) |
| **Magazine layer** | 15 fixed departments, `issue_spec.py` as the single source of truth, `magazine-editor` agent, deterministic renderer, newsstand |
| **Agent cache** | Fingerprints declared inputs per routine into a tracked `.agent-cache.json`; check → spawn → validate → record |
| **Exact printings** | Moxfield `(SET) COLLECTOR *F*` resolved first, so the manual shows the pilot's physical cards |
| **Featured Artist** | 15th department; `artist_credits.py` auto-detects standout artists, clusters, drop runs |
| **Role taxonomy** | `analysis/card_roles.py` (step 13) → `card_roles.json`; 43 deckbuilding roles, *separate* from `MECHANICAL_TAGS`; coverage published (86.5%, 68.1% excluding the body fallback) |
| **Bracket engine** | `pilot/bracket.py` computes a bracket **floor** from Game Changers, per-combo Spellbook bracket tags, two-card infinites and mass land denial; names the card or line driving it; writes `bracket_report.json`, which `validate-build` cross-checks against the plan's own claim |
| **Deterministic builder** | `brief.json` → `build_plan.json` → `decklist.txt` with no agent involvement; `manabase.py` sizes colour sources hypergeometrically |
| **Build loop** | `deck-analyst` → `deck-architect` ⇄ `deck-critic`, gated by `validate_build.py` and the bracket engine |
| **Deck facts** | `pilot/deck_facts.py` (`manamap pilot deck-facts <slug>`) — the deterministic brief every agent reads instead of re-deriving: DFC-correct colours, curve, pip load, role coverage + holes, contained combos, and a `notes[]` block naming the traps. Computed on demand, never committed |
| **Sideboard analysis** | `sideboard_facts.py` + `validate_sideboard.py` + `sideboard-analyst`; the agent's pool is **that deck's sideboard only** — no pool search, no rebuild, no new decklist. Deltas recomputed by the validator, never trusted. Renders as a section in Upgrade Watch |
| **Sideboard analysis** | `sideboard_facts.py` + `validate_sideboard.py` + `sideboard-analyst`; the agent's pool is **that deck's sideboard only** — no pool search, no rebuild, no new decklist. Bracket deltas recomputed by the validator, never trusted. Renders as a section in Upgrade Watch |
| **Loop economics** | Scenario preflight (`validate-stack --scenario-only`) before any spawn; `RESOLVE_SCOPE_BUDGET` warns on oversized artifacts; `RESOLVE_MAX_ITERATIONS` enforced in `cache-record` rather than quoted in markdown; agents hand off by path via `.agent-out/` |

## The agent roster (12)

All read-only except where noted. Definitions in `.claude/agents/`.

| Agent | Role |
|---|---|
| `stack-resolver` | Cite-or-decline stack resolutions |
| `rules-checker` | Adversarial citation verification + missing-step audit |
| `manual-writer` | Body prose under the zero-guessing rule |
| `pilot-coach` | Threat/matchups/decision coaching (★) |
| `magazine-editor` | The issue plan: cover, departments, headlines, furniture |
| `strategy-researcher` | Strategy doc research (**write-scoped to `data/strategy/`**) + strategic consulting |
| `deck-analyst` | ◆ data layer; emits `candidate_pool.json` under a schema |
| `deck-architect` | Improves the deterministic plan; every ratio cites `strategy:<id>` |
| `deck-critic` | Adversarial verifier for build plans; report-only, never edits |
| `sideboard-analyst` | Sideboard swaps, cuts and long-term defaults; **pool = that deck's sideboard only** |
| `pipeline-runner` | Runs and diagnoses card-pipeline steps |
| `viz-dev` | Frontend work (write-scoped to `viz/`) |

## The pipelines

```
Build     brief.json → build-deck → validate-build → bracket-check
                     → /build-deck   (deck-analyst → deck-architect ⇄ deck-critic, ≤3 iters)
                     → decklist.txt

Publish   fetch-deck → validate-deck → goldfish
                     → /resolve-stack per line   (resolver → validate-stack → rules-checker, ≤3)
                     → /write-manual             (deck-analyst → strategic frame → coach → writer)
                     → /design-issue             (magazine-editor → validate-issue)
                     → build-manual → build-index
```

Every agent step is cache-gated across **7 static routines** (`candidate-pool`,
`deck-build`, `strategic-frame`, `sideboard-analysis`, `coach-prose`, `writer-prose`,
`issue-plan`) plus the
dynamic `stack:NNN` / `decision:NNN` families. `cache-status <slug>` before spawning,
`cache-record` after validating. Exit 0 = don't spawn, 1 = spawn, 2 = fix the input first;
routines that don't apply to a deck report `N/A` and never abort the scan.

## Verified stacks — goblin-storm

| # | Line | Verdict |
|---|---|---|
| 001 | Zada + Crimson Wisps engine (1 mana → 5 cards + haste) | pass, 1 iter |
| 002 | Empty the Warrens storm turn (copies ≠ casts) | pass, 1 iter |
| 003 | Fists of Flame exact lethal | pass, **2 iters** — checker caught a missing SBA step |
| 004 | Krenko/Prospector — combo-graph infinite **refuted** for this deck (903.9a) | pass, 1 iter |
| 005 | Haze of Rage + Storm-Kiln — true infinite (k+3 Treasures per 4 mana) | pass, 1 iter |

Stack 004's finding is now enforced in code: `bracket.py` excludes combos whose `produces`
mentions the command zone when none of their pieces is actually this deck's commander.

## Decks

| Deck | State |
|---|---|
| `goblin-storm` | Hand-built, **published** as Vol. 001 — 5 verified stacks, 2 decision spreads |
| `hapatra` | **Built by the v2 loop and published** as Vol. 002 — bracket 4, 1 verified stack, 0 decisions |
| `sisay` | Hand-built, **published** as Vol. 003 — five-colour Sisay, Weatherlight Captain toolbox, bracket 4, **1 verified stack of 3 attempted**, 2 decision spreads |
| `edgar-vampires` | **Built** by the deterministic builder (Mardu, bracket 3, floor 1); no brief-driven agent pass, not published |
| `heliod` | **Published** as Vol. 004 — v3 build, five verified stacks, five sideboard swaps applied, zero swaps left |
| `ur-dragon` | Hand-built, **published** as Vol. 005 — five-colour dragon tribal, bracket floor 3, 3 verified stacks (003 lives in the sideboard: Ventmaw + Aggravated Assault, floor 3→4 if swapped in), 2 decision spreads, full 38-card sideboard analysis (Deflecting Swat + Korlessa promoted) |

## Vol. 003 — Sisay, and the first issue to ship with a `fail`

The deck is a toolbox whose currency is **access**, not cards: Sisay is 2/2 plus one per distinct
colour among *other* legendary permanents, and her gate is mana value **strictly less than** her
power. The ladder is **3 / 10 / 23 / 28 / 33 / 33** targets at power 2–7, and it tops out at four
colours because nothing in the list exceeds mana value 5 — the fifth colour buys stats and zero
access. The structural flaw is the headline: **26 of the 99 are non-legendary and permanently
outside her reach**, including three of the six cards that can pay her `{W}{U}{B}{R}{G}`, all six
counterspells, and all four Game Changers. She finds payoffs, never fuel or protection.

**Three stacks attempted, one passed.** This is the first issue to publish with failures on the
record, and Judge's Desk names them rather than hiding them:

| # | Line | Verdict |
|---|---|---|
| 001 | The escalating tutor chain | **fail** at 3 iterations — one clause: a −X/−X death case priced at exactly −6/−6 when power can go below zero. Everything else verified, including the ladder arithmetic and the summoning-sickness answer |
| 002 | Najeela + Derevi combat loop | **pass** at 3 iterations, 116 citations |
| 003 | Relic of Legends vs Esika's granted ability | **fail** at 3 iterations — the mana ceiling is capped by trigger count, not tapped-source count, because a permanent can be re-tapped in the priority window between resolutions |

**Stack 002 corrected the bracket engine in print.** The engine calls Najeela + Derevi a two-card
infinite and uses it to justify floor 4. Verified verdict: **partially accurate**. A real,
deterministic, unbounded loop — but not two cards, because the pair produces no mana and each
iteration owes five pips again. Attackers follow `A' = 2A − 1` (3, 5, 9, 17, 33); combat 1 must be
pre-funded, **combat 2 breaks even at exactly 5**, combat 3 runs away. It also established the loop
costs **0 life, not 2** — Command Tower plus Plaza of Heroes cover all five colours, so City of
Brass and Mana Confluence never need tapping (CR 118.3c: activating a mana ability is never
mandatory). The floor of 4 holds regardless, forced by the four Game Changers alone.

**Process notes worth keeping.** Three errors originated in *my* prompts, not the agents:
a checker status vocabulary I invented (`partial`, which the validator rejects — the real set is
`supported`/`unsupported`/`irrelevant`/`misquoted`, and collapsing them lost diagnostic
resolution); a prose key the renderer does not read (`combo_line_intros` vs `combo_lines`, which
would have rendered a TODO where the only verified line goes); and a wrong card fact propagated
into the strategic frame — **Secluded Courtyard *can* pay the activation**, because it carries an
"or activate an ability of a creature source of the chosen type" clause that Unclaimed Territory
lacks. The writer caught the last one by preferring oracle text over the frame it was handed.

Also: both decision spreads were authored before 002 cleared, and both asserted the deck had no
checker-passed resolution. The magazine-editor caught the contradiction against its own cover at
plan time and flagged it as blocking. Fixed in the artifacts, not papered over in the plan.

## Verified stack — hapatra

**Verified stack — hapatra.** One line, and it rewrote the deck. Stack 001 asked whether
Hapatra's −1/−1 counters reset undying under Mikaeus; the answer is no, and the bracket
engine's **19 two-card infinites is inflated**. Eight route through Mikaeus and six of those
are refuted: a Snake token yields *zero* iterations (it ceases to exist before the trigger
is even placed on the stack) and a non-token creature yields exactly *one*. What survives is
Mikaeus + Devoted Druid (a true two-card infinite of deaths and Snakes, **not mana** — the
returned Druid is summoning-sick), Mikaeus + Walking Ballista (real, three cards) and
Yawgmoth (the only free repeatable −1/−1 source, library-bounded by its own draw).

Verdict `pass` at **4 iterations** — `RESOLVE_MAX_ITERATIONS` is 3 and was deliberately
overridden, recorded in `checker.iteration_bound_override`. See the loop note below.

## The yield collapse, and what was done about it

Three decks through the pipeline, and the per-round yield of verified lines fell off a
cliff: goblin-storm **0.83** lines/round (5 in 6), hapatra 0.25 (1 in 4), sisay **0.11**
(1 in 9). The cause is measurable and authored, not inherent — **citation count predicts
iterations**, because the checker's verdict is atomic over an artifact whose size the
author controls:

| Artifact | steps | citations | outcome |
|---|---|---|---|
| goblin-storm 001–005 | 8–12 | **18–32** | pass@1 ×4, pass@2 ×1 |
| hapatra 001 | 16 | 59 | pass@**4** |
| sisay 001 / 003 | 15 / 16 | 84 / 82 | **fail**@3 |
| sisay 002 | 24 | 116 | pass@3 |

Sisay 003's answers (a)–(d) were independently verified correct in **all three** passes
and were discarded because sub-question (e) failed in the same file. The rule now stated
in `resolve-stack/SKILL.md`: **one rules domain per scenario**, split multi-part
questions. `RESOLVE_SCOPE_BUDGET` warns, and re-running the budget over the committed
artifacts flags every problem file and none of goblin-storm's.

**Four live bugs found by auditing the loops, each with correct code already in-repo:**

- **DFC colours were `[]` in every deck's `cards.json`.** `fetch_deck.py` read the
  top-level field, which Scryfall omits for transform/modal DFCs, and `_shape_face`
  dropped face colours so nothing could recover them. `extract.get_colors()` — correct
  and tested — was never imported. 7 cards across 3 decks; `colors` is in
  `CARD_SEMANTIC_FIELDS`, so the wrong value was agent-facing.
- **Every synergy chip in every manual rendered empty.** `build_manual.py` read an
  `entry["rule"]` key that has never existed on `synergy_graph.json` (`partner`/`score`/
  `synergies`). Two published issues had **zero** chips; now 125/120/110. The test
  fixture had invented the same wrong key, which is why nothing caught it.
- **`cache-record` used `any()`, not `all()`** — a 1-of-6-key artifact recorded cleanly
  and froze as a permanent HIT. On its first run the fix caught hapatra's `deck-build`
  recorded with **5 of 9 keys**: the critic block and the architect's `engines`/`keep`/
  `gaps` were never merged, and the cache had been saying "current, don't re-spawn".
- **The `cover` prose key was dead work** — declared, written on every deck, cached, and
  rendered nowhere (`issue_plan.json` owns the cover).

Also: `RESOLVE_MAX_ITERATIONS` was never imported by any Python file — the bound was
enforced by a model reading a number in markdown, which is why hapatra ran to 4. It is
now enforced in `cache-record`, with `iteration_bound_override` schematized rather than
invented ad hoc (it accepts the bare string hapatra actually stored).

## Open

**Deck versioning is the next thing this needs, and nothing exists for it.** The
sideboard analyst *proposes* swaps; the stated next step is an agent that **applies** one
and publishes a new deck version without publishing a new manual each time — the UI shows
only the current list. There is no versioning primitive anywhere today: no `version`
field in any deck artifact, no per-deck changelog, and `issue.json.volume` is a magazine
issue number, not a deck version (nothing checks it for uniqueness or monotonicity).
`build_index.gather_entries()` keys on **slug** and emits at most one entry per deck, so a
second issue for the same deck is currently inexpressible and would silently overwrite
`manuals/<slug>.html`.

`decklist_sha256` (`fetch_deck.py`) is already the right identity primitive: content-
addressed, tracked, stable, scoped to the decklist rather than to Scryfall data, and
already printed in the colophon. It is an identity, not yet a version — it carries no
ordering and nothing records what came before. Three steps, increasing cost:

1. **`data/decks/<slug>/HISTORY.md`** — append-only, one line per decklist change
   (`decklist_sha256`, date, bracket, one-sentence reason). Modelled on
   `data/strategy/CHANGELOG.md` + `validate_strategy.py`, the repo's only precedent for a
   tracked artifact plus a validated append-only log. Ordering comes from append
   position, identity from a hash that already exists. Zero new computation.
2. **`supersedes` in `issue.json`**, plus stamping `decklist_sha256` at publication, so
   "Vol. 004 corrects Vol. 002" becomes expressible and `build_index` has something to
   key on besides the slug.
3. **`build:<NNN>` as a third dynamic cache-routine family** — `docs/deck-builder-v2.md`
   already names the three edits, and `artifact_subdir` supports a `builds/` directory
   with no new code.

Explicitly **avoid a hand-incremented `deck_version` int**; `volume` already demonstrates
that failure mode.


**Sisay's two failed stacks are one or two named clauses from passing**, and the magazine-editor
ranks 001 as the highest-value fix in the deck: resolving it would promote the ladder arithmetic
and the summoning-sickness answer from ★ to ✓ across three departments at once. 003 needs a fresh
resolve run rather than a patch — its defect is load-bearing, not clerical. Both were declined a
fourth iteration deliberately; `RESOLVE_MAX_ITERATIONS` is 3 and the bound was left standing.

**Nine of Sisay's ten combo lines are unverified.** The three worth resolving next are the other
Najeela pairs — Faeburrow Elder, Esika, Selvala — all claimed as two-card infinites at bracket 3.
Case A-002 hands them a ready-made test: unlike Derevi, each of those *does* produce mana, so the
break-even arithmetic that refuted the two-card claim may resolve differently. Interesting either
way. Also queued, from the writer: does a fetched Mikaeus arrive as a 0/0 (decides whether the
cold-start pool is three cards or two), and which face an Esika put onto the battlefield arrives on.

**Sisay's `goldfish_targets.json` has no target for the line the cover sells.** The issue can prove
the loop works and cannot say how often it is available — the obvious reader question. A
"Najeela + Derevi + five colour sources" target would close it.

**Eleven unresolved combo lines on hapatra**, none of which may be stated as fact. Highest
value first, per the magazine-editor's own gaps list:
- **Does Mikaeus's `+1/+1` anthem switch off the deck's own token loops?** Snakes, Insects
  and Elf Warriors are non-Human, so under Mikaeus they are 2/2 and a single −1/−1 counter
  no longer kills them. If so, the deck's two flagship engines are mutually exclusive.
  Stack 001 step 15 already established this exact layer-7c math for Walking Ballista.
- **The Blowfly Infestation family** — the line goldfish actually tracks (16.3% assembled),
  and the deck's headline two-card kill on paper. Never verified.
- **Yawgmoth + Hapatra without Mikaeus** — very likely the line the deck wins with most
  often, and verified only *inside* the Mikaeus context.
- **The Ivy Lane Denizen family.** Six of its seven two-card entries pair it with sacrifice
  outlets, but Ivy Lane only triggers when a green creature *enters* and none of those
  outlets makes one — the same projection artifact that inflated the Mikaeus count.
- Hapatra + Host of the Hereafter; Necroskitter + Black Sun's Zenith; and whether Heroic
  Intervention answers a −X/−X effect that kills by toughness rather than destruction.

**hapatra has no decision spreads**, so What's Your Play? renders a visible TODO. The
magazine-editor specified both boards precisely enough to author straight into artifacts —
a counted-pivot board and a default-fetch board — see `issue_plan.json` gaps.

**`bracket_report.json` contradicts the verified stack in print.** It still carries the
inflated 19 and names a refuted pairing as its example driver. The floor of 4 holds on the
11 Game Changers alone, but the artifact should either carry a verified-refutation
annotation or the engine should consult passing stacks.

**edgar-vampires is built but not architected or published.** It has `brief.json`,
`build_plan.json`, `cards.json` and a bracket report from the deterministic path only; no
`candidate_pool.json`, no agent pass, no `issue.json`.

**The resolve loop's iteration bound needs a structural fix.** Stack 001 took four resolver
passes and four checks — roughly 600k tokens for one line, against five lines for less in
Vol. 001. The override was justified (the checker had confirmed convergence; the blockers
were a stale count and a citation about casting spells doing work that belongs to activating
abilities) but a bound lifted whenever the checker sounds confident is not a bound. Either
raise `RESOLVE_MAX_ITERATIONS` to 4 or add a mechanical-defects-only exit that doesn't
consume a full iteration.

**A prompt is an evidence surface.** A figure I stated in a prompt ("99.7% by turn 6", when
goldfish reports 0.999) propagated into two prose keys and was caught only by the
magazine-editor at plan time. Figures passed to agents need the same citation discipline as
figures in artifacts.

**`meta-analyst` — deferred, not dropped.** The v2 design called for a meta-awareness agent
with its own `data/meta/` corpus (`validate_meta`, `build_meta_db`, `query_meta`,
`META_ID_RE`). It was traded away to get the loop working and the loop demonstrably works
without it. The design point worth keeping: **meta claims perish and strategy theory
doesn't**, so they want separate corpora with different invalidation, and every meta section
needs an `as_of` date.

**Frontend v2 — planned, not started.** Full plan in `docs/frontend-v2.md`; scope and the
reasoning behind it below. Decisions taken: built for one expert user (depth over
onboarding, no mobile or a11y push), a new builder surface with the map kept as-is,
everything deterministic client-side with a handoff to Claude Code for the judgment layer,
and the magazine's design system as the single design language.

Three findings reframe it from "modernise the deck builder" to "the builder already exists,
it's just in the wrong language":

- **The whole deterministic core ports to JS.** `build_deck.py`, `bracket.py`,
  `manabase.py`, `goldfish.py` and `validate_build.py` use numpy/pandas as convenience over
  arrays and dicts, and every constant is a literal in `config.py`. Bracket floors would be
  byte-identical; goldfish runs 10K iterations in ~200–500 ms in a Worker.
- **The blocker is 476 KB.** The browser is missing exactly four card fields —
  `game_changer`, `mechanical_tags`, `layout`, and `legal_commander` as a tri-state
  (`reduce.py` collapses `banned` and `not_legal`, so the browser can't reproduce
  `bracket.py`'s banned note). A positional `viz_index.json` in cards.csv row order closes
  every gap.
- **~1 MB of finished JSON per deck is committed and unrendered.** `build_plan.json` carries
  every slot's six-component score and its runners-up with deltas; `bracket_report.json`
  carries display-ready driver sentences. `combo_details.json` and `card_roles.json` are
  tracked, served by Pages, and never fetched by anything.

**The problem worth solving is the missing middle gear.** A curated 100-card deck costs
~250 clicks today, each triggering a full `innerHTML` rebuild that resets scroll and
collapses every expanded row; the alternative is ~12 clicks and no agency. The fix is a
change of primitive — **the deck starts complete** and you review and swap slots, because
the builder already fills 63 slots by role budget and already keeps scored alternates for
each. A score delta of 0.01 is the signal that the scorer was nearly indifferent and the
pilot's judgment is cheap. Two things give the loop consequence, both newly possible: live
goldfish on every swap, and a live bracket floor that **names its driver** ("Bracket 4 —
Mikaeus + Devoted Druid"), which nothing else can do without a rules-verified combo corpus.

Milestones, sequenced **M1 → M2 → M6 → M3 → M4 → M5**:

| | Scope |
|---|---|
| **M1** | `viz_index.json`; fetch `combo_details.json` + `card_roles.json`; **int8-quantise `embeddings.bin`** (16.3 MB gz → 4.3 MB, and it's currently loaded *twice*) |
| **M2** | Port the five deterministic modules into a Worker |
| **M3** | Port `design.py` tokens + component library to CSS — `power_meter`→bracket floor, `fast_facts`→spec sheet, `badge`→tier/role, `threat_box`→matchups; dark register, same tokens |
| **M4** | `build.html` — slots as the primitive, incremental DOM, always-live diagnostics, map demoted from *only* input to *one* input |
| **M5** | Handoff: emit `brief.json` + `decklist.txt` and the command to run |
| **M6** | `deck.html` — render the already-committed artifacts. No pipeline work at all |

**M6 sits before M4 deliberately**: it's low-risk, needs no new data, and forces the
component library into existence against real content before anything depends on it. If the
plan stalls it should stall *after* M6, not before.

**The rule that keeps it honest.** Two implementations of one algorithm is the bug we
already have — `deck-builder.js` carries its own six-factor scorer with different weights
and a different sixth factor than `DECK_BUILD_WEIGHTS`, documented in two places with no
cross-reference. So `viz/js/engine/constants.js` is **generated from `config.py`**, never
hand-edited, and a parity test asserts both builders emit identical `build_plan.json`.
Divergence becomes a failing test rather than a documentation problem.

**The one genuinely fiddly part is goldfish determinism.** The manual renders those numbers
as ◆ reproducible evidence, so a browser producing different values would quietly break the
tier contract. It needs an MT19937 port matching `random.shuffle`'s reverse Fisher–Yates
with `_randbelow`. If that proves unreasonable, the honest fallback is labelling
browser-computed goldfish an *estimate* and never letting it overwrite a committed artifact.

Carried into M7 (opportunistic, not blocking): hover tooltips are **built for all 34,322
points on every render and thrown away** — all 13 traces set `hoverinfo: 'none'`, ~275K
regex ops producing text nobody sees, and turning them on is the single biggest available
UX win; the detail panel is hidden in build mode, exactly when you're deciding whether a
card belongs; there is no URL state anywhere (`location`, `history`, `URLSearchParams` do
not appear), so a map whose value is "look at this region" can't link to a region; and
`deck-builder.js` persists **raw row indices** into `projection_2d.json`, so any pipeline
refresh that changes card ordering silently reinterprets every saved deck as a different
set of cards — a correctness bug, and the one item worth doing on its own schedule.

**Four strategy-DB gaps the Vol. 002 pipeline routed around**, all flagged by the strategic
frame and all load-bearing there: no section on **auditing a combo list** (how to price an
unverified count, why pairwise combo data silently drops third pieces) — the single most
load-bearing idea in that issue; no **aristocrats/sacrifice-engine** section (outlet, fodder
and converter taxonomy); no **counters-matter** section (annihilation as a resource, the
anthem-versus-shrink anti-synergy); and no **tutor-sequencing** section for pilots.

**Queued stacks.** Roaming Throne × Zada is the interesting one: Throne makes Zada's copy
trigger fire twice, and whether that produces a second full copy set is genuinely unsettled.
Past in Flames ritual rebuild is the other. The 005 follow-on kills are prose-flagged as
pilot-assembled and could be promoted to ✓.

**goblin-storm's cache reports all-MISS**, invalidated by the strategy corpus growing and
`combo_graph.json` splitting — not by anything about the deck. Its prose is still correct.
Re-record rather than re-spawn (~330k tokens saved), but do it as a stated decision.

**Judgment calls surfaced by audit, deliberately not changed:**
- `goldfish.py` — `mean_cast_turn` divides by games where the commander was actually cast;
  `cast_by_turn_6_rate` divides by all games. They sit side by side in the Commander File.
- `goldfish.py` — `cast_by_turn_6_rate` hardcodes turn 6 while `GOLDFISH_MAX_TURN` is
  configurable.
- `power_creep.py` — the docstring numbers eight criteria; the inline comments number seven.
- `download_combos.is_up_to_date` checks file existence only, so combo data never refreshes.

**Editorial gaps flagged by the magazine-editor:**
- The per-set dispersion story (63 of 80 cards from The List across 53 artists) could carry
  a paragraph.
- Three Secret Lair cards have no role group in The 99 and fall into the unlabelled bucket.
- Strategy DB has no commander-damage/clock section to ground The Command Zone's 21-damage
  answer.

## Invariants that must not erode

- Only checker-passed stacks publish; failed artifacts are kept as open questions.
- Agents return JSON and never write HTML — that's what keeps rebuilds byte-identical.
- `issue.json` is authored, never generated.
- Costume never earns the badge: a department cannot claim a tier it wasn't granted.
- Record the cache **after** validation, never before.
- A bracket **floor** is what the contents are consistent with, never a verdict — brackets
  are a conversation, and tutor density is reported but never scored.
- The deterministic builder must always produce a complete legal 99 with no agent involved.
