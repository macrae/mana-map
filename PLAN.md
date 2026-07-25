# PLAN — current state and what's next

*The resume-here doc. Read `README.md` for orientation, `CLAUDE.md` for gotchas, this for
what shipped and what's still open. Completed plans live in `docs/history/`.*

Last updated 2026-07-25. All work below is committed, pushed, and deployed.

## What this is now

Two products in one repo. The card map is stable and complete. The active work is
**Pilot's Manual** — a magazine generator that turns one Commander deck into a
self-contained web issue with a three-tier evidence contract.

Live: [Vol. 001 — Goblin Storm](https://macrae.github.io/mana-map/manuals/goblin-storm.html)
· [newsstand](https://macrae.github.io/mana-map/manuals/index.html)

## Shipped

| Layer | What |
|---|---|
| **Rules DB** | CR chunked one-per-rule (~3.9K), chunk ID = citation ID; semantic query + exact lookup |
| **Citation contract** | Form enforced in `validate_stack.py`; meaning by adversarial `rules-checker`; only `pass` publishes |
| **Goldfish** | Seeded Monte Carlo (seed 42, 10K iters), resource development *not* full games, assumptions rendered |
| **Strategy DB** | 32 sourced sections / 13 pillars; `strategy:<id>` = citation ID; `strategy-researcher` agent (research + consult modes) |
| **Magazine layer** | 15 fixed departments, `issue_spec.py` as the single source of truth, `magazine-editor` agent, deterministic renderer, newsstand |
| **Agent cache** | Fingerprints declared inputs per routine into a tracked `.agent-cache.json`; check → spawn → validate → record |
| **Exact printings** | Moxfield `(SET) COLLECTOR *F*` resolved first, so the manual shows the pilot's physical cards — artist, border, foil, art crop |
| **Featured Artist** | 15th department; `artist_credits.py` auto-detects standout artists, clusters, drop runs, treatments |

## The agent roster (9)

All read-only except where noted. Definitions in `.claude/agents/`.

| Agent | Role |
|---|---|
| `stack-resolver` | Cite-or-decline stack resolutions |
| `rules-checker` | Adversarial citation verification + missing-step audit |
| `manual-writer` | Body prose under the zero-guessing rule |
| `pilot-coach` | Threat/matchups/decision coaching (★) |
| `magazine-editor` | The issue plan: cover, departments, headlines, furniture |
| `strategy-researcher` | Strategy doc research (**write-scoped to `data/strategy/`**) + strategic consulting |
| `deck-analyst` | Read-only data layer over graphs and embeddings |
| `pipeline-runner` | Runs and diagnoses card-pipeline steps |
| `viz-dev` | Frontend work (write-scoped to `viz/`) |

## The build pipeline

```
fetch-deck → goldfish
          → /resolve-stack per line     (resolver → validate-stack → rules-checker, ≤3 iters)
          → /write-manual               (deck-analyst → strategic frame → coach → writer)
          → /design-issue               (magazine-editor → validate-issue)
          → build-manual → build-index
```

Every agent step is cache-gated: `cache-status <slug> --routine R` before spawning,
`cache-record` after validating. Exit 0 = don't spawn, 1 = spawn, 2 = stop.

## Verified stacks — goblin-storm

| # | Line | Verdict |
|---|---|---|
| 001 | Zada + Crimson Wisps engine (1 mana → 5 cards + haste) | pass, 1 iter |
| 002 | Empty the Warrens storm turn (copies ≠ casts) | pass, 1 iter |
| 003 | Fists of Flame exact lethal | pass, **2 iters** — checker caught a missing SBA step |
| 004 | Krenko/Prospector — combo-graph infinite **refuted** for this deck (903.9a) | pass, 1 iter |
| 005 | Haze of Rage + Storm-Kiln — true infinite (k+3 Treasures per 4 mana) | pass, 1 iter |

## Open

**Blocked on decklists** — `hapatra` and `edgar-vampires` are scaffolded and waiting.
Each needs a Moxfield export, then `goldfish_targets.json` and `issue.json` authored.

**Queued stacks.** Roaming Throne × Zada is the interesting one: Throne makes Zada's copy
trigger fire twice, and whether that produces a second full copy set is genuinely
unsettled — good checker fodder, and it would give The Kill a sixth line. Past in Flames
ritual rebuild is the other. The 005 follow-on kills (Fists trample exit, Treasure-funded
Empty the Warrens under Impact Tremors, storm-count Grapeshot) are prose-flagged as
pilot-assembled and could be promoted to ✓.

**A real data question.** `goldfish_metrics.json` reports `keep_first_seven_rate` of
0.791, but its own `land_histogram` puts 9,899 of 10,000 first sevens at 2–5 lands — under
the stated keep rule those two figures cannot describe the same test. The 79.1% is quoted
in three departments. Either the keep rule checks something undocumented, or the histogram
counts a different population.

**Judgment calls surfaced by the doc audit, deliberately not changed:**
- `goldfish.py` — `mean_cast_turn` divides by games where the commander was actually cast;
  `cast_by_turn_6_rate` divides by all games. They sit side by side in the Commander File,
  and the natural reading of the first understates the deck.
- `goldfish.py` — `cast_by_turn_6_rate` hardcodes turn 6 while `GOLDFISH_MAX_TURN` is
  configurable.
- `power_creep.py` — the docstring numbers eight criteria; the inline comments number
  seven, off by one from item 4.
- `download_combos.is_up_to_date` checks file existence only, so combo data never refreshes
  once present.

**Editorial gaps flagged by the magazine-editor:**
- The per-set dispersion story (63 of 80 cards from The List across 53 artists) is one row
  in the Art File and could carry a paragraph.
- Three Secret Lair cards — Moggcatcher, Broadside Bombardiers, Roaming Throne — have no
  role group in The 99 and fall into the unlabeled Depth bucket. They're in the deck
  because they came in the box, which is a legitimate roster fact the roster doesn't state.
- Strategy DB has no commander-damage/clock section to ground The Command Zone's
  21-damage answer.

## Next: Deck Building v2

**Full spec: [`docs/deck-builder-v2.md`](docs/deck-builder-v2.md).** Proposed, not started.

Give the system a commander, a target bracket, and any cards you already want; it returns a
legal, functional 99 with a justification per slot, a mechanically-proven bracket claim, and
a simulated proof the mana works — then publishes its own manual. The same evidence contract
that governs how we *describe* a deck governs how we *choose* it.

Four things set the shape:

1. A prototype builder already ships in `viz/js/deck-builder.js`, but it doesn't model
   Commander (color identity only enforced when a commander is set, no partners, no
   "build me a deck"), every constant is an untested JS literal, and it persists raw row
   indices that a pipeline refresh silently corrupts.
2. **The power-level signal is already on disk and we discard it.** `combos_raw.json` carries
   a per-combo `bracketTag` (`E`xhibition/`C`ore/`O`ddball/`P`owerful/`S`picy/`R`uthless/
   `B`anned — read from the Spellbook backend source), plus `popularity`, `manaValueNeeded`,
   prerequisites and prices. `process_combos.py` drops all of it. ~30 lines to recover, and
   it makes a **computed bracket floor** possible: we can name the exact combo that pushes a
   deck out of Bracket 2.
3. `MECHANICAL_TAGS` is a retrieval vocabulary, not a deckbuilding role taxonomy — `ramp`
   conflates rocks, dorks, land ramp and rituals; there is no `wincon` tag; 20% of cards are
   untagged. Slot-filling needs a new classifier.
4. **The gate:** of 32 strategy sections, zero are about construction, and tutors are never
   mentioned once. Since citations are verbatim-substring checked, a builder cannot cite
   "37 lands" because that number exists nowhere in the prose. Phase 0 is a deckbuilding
   research pass (~14–18 sections, ~3 passes) and everything else waits on it.

New agents: `deck-architect` ⇄ `deck-critic` (adversarial pair, `rules-checker` pattern),
`meta-analyst` (own dated corpus — meta claims perish, theory doesn't), and `deck-analyst`
upgraded with a schema and an artifact so it's finally cacheable.

Hard requirement: **the deterministic core produces a complete legal 99 with no agent
involvement.** Agents improve a build; they're never required to make one.

## Invariants that must not erode

- Only checker-passed stacks publish; failed artifacts are kept as open questions.
- Agents return JSON and never write HTML — that's what keeps rebuilds byte-identical.
- `issue.json` is authored, never generated.
- Costume never earns the badge: a department cannot claim a tier it wasn't granted.
- Record the cache **after** validation, never before.
