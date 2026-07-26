# PLAN — current state and what's next

*The resume-here doc. Read `README.md` for orientation, `CLAUDE.md` for gotchas, this for
what shipped and what's still open. Completed plans live in `docs/history/`.*

Last updated 2026-07-25. All work below is committed, pushed, and deployed.

## What this is now

Two products in one repo. The card map is stable and complete. The active work is
**Pilot's Manual** — a magazine generator that turns one Commander deck into a
self-contained web issue with a three-tier evidence contract — and, since Deck Building v2,
a **deck builder** that produces the deck in the first place.

Live: [Vol. 001 — Goblin Storm](https://macrae.github.io/mana-map/manuals/goblin-storm.html)
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
| **Role taxonomy** | `analysis/card_roles.py` (step 13) → `card_roles.json`; deckbuilding roles, *separate* from `MECHANICAL_TAGS`; coverage published (86.3%, 67.6% excluding the body fallback) |
| **Bracket engine** | `pilot/bracket.py` computes a bracket **floor** from Game Changers, per-combo Spellbook bracket tags, two-card infinites and mass land denial; names the card or line driving it; writes `bracket_report.json` |
| **Deterministic builder** | `brief.json` → `build_plan.json` → `decklist.txt` with no agent involvement; `manabase.py` sizes colour sources hypergeometrically |
| **Build loop** | `deck-analyst` → `deck-architect` ⇄ `deck-critic`, gated by `validate_build.py` and the bracket engine |

## The agent roster (11)

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

Every agent step is cache-gated across **6 static routines** (`candidate-pool`,
`deck-build`, `strategic-frame`, `coach-prose`, `writer-prose`, `issue-plan`) plus the
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
| `goblin-storm` | Built by hand, **published** as Vol. 001 |
| `hapatra` | **Built** by the v2 loop (bracket 3, floor 3); not yet published |
| `edgar-vampires` | Scaffold only — no brief, no decklist |

## Open

**Publish hapatra.** It has `brief.json`, `candidate_pool.json`, `build_plan.json`,
`cards.json`, `goldfish_targets.json` (derived from the plan's engines) and
`goldfish_metrics.json`. It needs, in order: `issue.json` authored → at least one
checker-passed stack → strategic frame → coach + writer prose → issue plan → `build-manual`.

The first stack is already specified. `build_plan.gaps` names it: CR 122.6 establishes that
counters put on a permanent *as it enters* count as counters being put on it, but not **who
put them** when the counter arrives via persist or the creature's own enters-with
replacement effect. Hapatra reads "whenever *you* put one or more −1/−1 counters on a
creature". Resolving it once validates or kills four persist slots.

**Build edgar-vampires.** No longer blocked on a decklist — the builder produces one. Its
three-colour mana base is a harder test of `manabase.py` than hapatra's two.

**`meta-analyst` — deferred, not dropped.** The v2 design called for a meta-awareness agent
with its own `data/meta/` corpus (`validate_meta`, `build_meta_db`, `query_meta`,
`META_ID_RE`). It was traded away to get the loop working and the loop demonstrably works
without it. The design point worth keeping: **meta claims perish and strategy theory
doesn't**, so they want separate corpora with different invalidation, and every meta section
needs an `as_of` date.

**Phase 4, the frontend.** `viz/js/deck-builder.js` persists **raw row indices** into
`projection_2d.json` — any pipeline refresh that changes card ordering silently
reinterprets every saved deck as a different set of cards. That is a correctness bug and the
one Phase 4 item worth doing on its own schedule. The viz was otherwise untouched by v2 and
its build-mode payload dropped ~19 MB for free when `combo_graph.json` was split.

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
