# PLAN — Pilot's Manuals: Tiered Evidence + Strategy Research

*Active plan (unlike `docs/history/PLAN.md`, which is archived). Manual v2 drafted
2026-07-24 from the founder conversation about piloting-focused manuals for
MagicCon; strategy subsystem added the same day. This file is the resume-here
document: current state, how it all fits, and what's next.*

## Goal

Coaching manuals from a world-champion perspective, with every claim wearing its
epistemic status visibly, backed by a growing strategy knowledge base ("the
brains of the operation"). Three decks: `goblin-storm` (LIVE, deployed),
`hapatra`, `edgar-vampires` (scaffolded, blocked on decklists). Shareable at the
MagicCon release via the manuals gallery.

## Current state (2026-07-24, all pushed & deployed)

- **goblin-storm manual v2.1 is live**: https://macrae.github.io/mana-map/manuals/goblin-storm.html
  (gallery at /manuals/index.html). 5 verified stacks, 2 decision spreads,
  goldfish section, threat/matchups coaching grounded in the strategy DB.
- **Strategy subsystem is live**: `data/strategy/strategy.md` (32 sourced
  sections after two research passes) + RAG DB + `strategy-researcher` agent
  (research/consult modes) + `strategic_frame.json` per deck. See docs/pilot.md
  "Strategy DB".
- 364 tests green. Branch `main` level with `origin/main`.

## The three-tier evidence contract (the brand)

| Tier | Badge | Content | Enforcement |
|---|---|---|---|
| **Rules-verified** | ✓ green | Stack resolutions | Citation contract (`validate_stack.py`) + adversarial `rules-checker` agent |
| **Data-derived** | ◆ blue | Goldfish metrics, curve/upgrade data | Seeded reproducible artifacts (byte-identical re-runs) |
| **Coaching** | ★ gold | Politics, threats, matchups, decisions | Labeled judgment grounded in tier 1/2 artifacts **and the strategy DB**; founder review of tracked JSON |

The strategy DB is *not* a fourth tier — it is curated, sourced, human-reviewed
grounding for tier ★. `strategy:<id>` citations in decision branches pass the
same verbatim-quote contract as CR citations.

## The agent roster (`.claude/agents/`)

| Agent | Role | Writes files? |
|---|---|---|
| `stack-resolver` | Cite-or-decline stack resolutions | No (orchestrator writes) |
| `rules-checker` | Adversarial citation verification + missing-step audit | No |
| `manual-writer` | Manual prose under the zero-guessing rule | No |
| `pilot-coach` | Threat/matchups/decision coaching (★) | No |
| `deck-analyst` | Read-only data layer over graphs/embeddings | No |
| `strategy-researcher` | Strategy doc research (online) + strategic consulting (RAG) | **Yes — scoped to `data/strategy/` only** (research mode); consult mode read-only |

## The v2.1 manual pipeline (write-manual skill)

1. `cards.json` (build-deck-db) + ≥1 verified stack (resolve-stack)
2. Goldfish (◆): `manamap pilot goldfish <slug>`
3. Evidence pull: `deck-analyst`
4. **Strategic frame** (★): `strategy-researcher` MODE consult →
   `data/decks/<slug>/strategic_frame.json`; its `candidate_missing_lines` feed
   the resolve-stack queue, its `gaps` feed the next research-strategy pass
5. Coaching (★): `pilot-coach` (receives the frame) → threat/matchups + decisions
6. Prose: `manual-writer` (receives the frame) → manual_prose.json
7. Build: `manamap pilot build-manual <slug>` + `build-index`
8. Founder review: manual HTML + tracked JSON diffs

The frame→stacks feedback loop is proven: goblin-storm's frame queued 4
candidate lines; resolving the two strongest produced one confirmed infinite
(005 Haze of Rage + Storm-Kiln) and one **refutation** (004: the Spellbook
Krenko/Prospector infinite requires Krenko as commander — 903.9a — which Zada
occupies; verified as a one-shot burst instead).

## Verified stacks — goblin-storm

| # | Line | Verdict |
|---|---|---|
| 001 | Zada + Crimson Wisps engine (1 mana → 5 cards + haste) | pass, 1 iter |
| 002 | Empty the Warrens storm turn (copies ≠ casts; Grapeshot count) | pass, 1 iter |
| 003 | Zada + Fists of Flame exact lethal (checker caught missing SBA on iter 1) | pass, 2 iters |
| 004 | Krenko/Prospector burst — combo-graph infinite REFUTED for this deck | pass, 1 iter |
| 005 | Haze of Rage + Storm-Kiln true infinite (k+3 Treasures per 4 mana) | pass, 1 iter |

Remaining queued candidates (from the frame + manual-writer flags): Past in
Flames ritual rebuild; Roaming Throne doubling Zada's trigger (needs a stack
scenario to settle whether the second trigger makes a second full copy set);
the 005 follow-on kills (Fists trample exit, Treasure-funded Empty the Warrens
under Impact Tremors, storm-count Grapeshot) are prose-flagged as
pilot-assembled, promotable to ✓ via resolve-stack if wanted.

## Strategy subsystem status

- Doc: 32 sections / 11 pillars, every section sourced with fetched-and-verified
  URLs. Two research passes logged in `data/strategy/CHANGELOG.md`.
- Known source constraints: TCGplayer Infinite unfetchable (JS shell), archive.org
  blocked, Command Zone video-only (not citable); PVDDR via Substack, Karsten via
  PDF mirror.
- Open research topics (from pass 2's report): a storm-hands keep/ship source
  (The EPIC Storm blocks fetches); quantitative racing-under-tax math; folding
  new authors (Girten, McGuinness, Cullen) into `strategy:schools`.

## Commit sequence

| # | Scope | Status |
|---|---|---|
| C1–C7 | Manual v2 (goldfish, decisions, coach agent, v2 build, Zada regen, docs) | **done** (2026-07-24) |
| S1 | Strategy KB infrastructure (validator, RAG DB, CLI, strategy citations) | **done** |
| S2 | Strategy doc seed + 2 research passes + researcher agent + skills + pipeline integration | **done** |
| S3 | Stacks 004+005 verified; manual v2.1 regenerated + deployed | **done** |
| C6 | hapatra + edgar-vampires full runs | **BLOCKED on Sean's decklists** |

## Next up (in rough priority order)

1. **C6**: when decklists land → `build-deck-db` → full v2.1 pipeline per deck
   (goldfish targets need per-deck curation in `goldfish_targets.json`).
2. Resolve remaining queued stacks (Past in Flames; Roaming Throne — genuinely
   unsettled rules question, good checker fodder; optionally the 005 exits).
3. Third research-strategy pass on the open topics above; consider a
   deck-agnostic "how to read a strategic frame" section.
4. Longer-term (from memory): commander build functionality in the viz deck
   builder — deck-analyst + strategic frames are the seed data layer.

## Risks / honesty notes

- Goldfish v1 simulates resource development, not games — assumptions render in
  the manual; overclaiming undercuts the brand.
- Decision trees + strategy doc are tier ★: founder review of tracked diffs is
  the quality mechanism, not machine verification.
- The combo graph is format-agnostic (Spellbook): "Infinite commander casts" in
  `produces` means the combo may assume that card is your commander — stack 004
  is the precedent for checking before believing.
- Agent-registry gotcha: a newly created `.claude/agents/*.md` is not spawnable
  in the same session that created it (registry loads at session start) — inline
  the definition via a general-purpose agent as fallback.
