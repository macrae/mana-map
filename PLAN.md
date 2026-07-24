# PLAN — Pilot's Manual v2: Coaching Manuals with a Tiered Evidence Contract

*Active plan (unlike `docs/history/PLAN.md`, which is archived). Drafted 2026-07-24 from the founder conversation about piloting-focused manuals for MagicCon.*

## Goal

Evolve the pilot's manual from "verified combo lines + card reference" into a **coaching manual from a world-champion perspective**: goldfish metrics, matchup heuristics, threat assessment, and the political dimension (archetypal board states with decision trees covering signaling, information management, and coalition dynamics). Three decks: `goblin-storm` (live), `hapatra`, `edgar-vampires`. Built to be read and shared at the MagicCon release.

## The core architecture: a three-tier evidence contract

Coaching content is judgment and cannot pass a citation contract. Instead of weakening "zero guessing" or refusing coaching content, every claim wears its epistemic status visibly:

| Tier | Badge | Content | Enforcement |
|---|---|---|---|
| **Rules-verified** | ✓ green | Stack resolutions, rules interactions | Citation contract (`validate_stack.py`): ≥1 citation per step, rule exists, quote verbatim; adversarial checker verdict |
| **Data-derived** | ◆ blue | Goldfish metrics, curve stats, combo/synergy/upgrade data | Must trace to a committed artifact produced by a seeded, reproducible script |
| **Coaching** | ★ gold | Politics, signaling, threat assessment, matchups, decision trees | Labeled as judgment; may reference tier 1/2 artifacts; embedded rules claims still require citations |

The tier legend appears on every manual's cover. This is the shareable differentiator: readers always know what is proven, what is simulated, and what is coaching.

## Components

### 1. Goldfish simulator — `src/manamap/pilot/goldfish.py` (`manamap pilot goldfish <slug>`)
Monte Carlo over shuffled decks, seeded RNG (reproducible → tier 2). v1 models **resource development, not full games** — assumptions stated in the output and rendered in the manual:
- Opening-hand land distribution + keepable-hand rate (uses a stated simple mulligan rule)
- Land-drop hit probability by turn
- Available-mana curve by turn (lands + simple persistent producers detected from oracle text)
- **Commander-cast turn distribution** (first affordable turn)
- Target-set assembly turns: per-deck `goldfish_targets.json` defines named piece sets (with `any_of` groups); metric = distribution of first turn the set is in hand (+ commander affordable when flagged)
- Bodies-by-turn (greedy casting with a token-count heuristic; explicitly labeled crude)

Output: `data/decks/<slug>/goldfish_metrics.json` (tracked; includes seed, iterations, model assumptions, decklist sha). Deterministic: same seed → byte-identical.

### 2. Decision-tree scenarios — `data/decks/<slug>/decisions/NNN-<kebab>.json`
The political dimension as reviewable artifacts. Each: archetypal board + **table** state (who's ahead, what's open, what's been signaled), a decision point, 2–4 branches each carrying: the line, signals sent, coalition risk, information given up/gained, coaching rationale, optional citations (same rule/quote contract when a branch makes a rules claim). Plus a recommendation.
`validate_stack.py` grows a `kind: "stack" | "decision"` switch (missing kind = stack, existing artifacts unchanged); mechanical checks: ≥2 branches, required fields, recommendation matches a branch, citations (when present) pass the existing contract. Tracked JSON = red-linable coaching film.

### 3. `pilot-coach` agent (world-champion voice)
Fourth agent: writes matchup heuristics, threat assessment ("when does the table flip on you"), authors decision scenarios. Evidence rules: every judgment grounds in a goldfish metric, verified stack, graph entry, or an explicitly stated archetypal assumption. Separate from `manual-writer` so voice and evidence rules stay distinct.

### 4. Manual v2 — `build_manual.py`
Section order: Cover (+ tier legend) → How It Wins → **Goldfish Numbers** (◆) → Combo Lines (✓) → **Playing the Table** (threat assessment + decision spreads, ★) → **Matchups** (★) → Card Roles → Mulligan (citing goldfish numbers) → Upgrades. Per-section badges, OG/social meta tags, `manuals/index.html` gallery page for one shareable MagicCon link.

### 5. Three decks
`goblin-storm` regenerates under v2. Scaffold `hapatra` and `edgar-vampires`; full runs blocked on Sean's decklists. write-manual skill becomes the v2 pipeline: goldfish → deck-analyst evidence → pilot-coach → manual-writer → build.

## Commit sequence

| # | Scope | Status |
|---|---|---|
| C1 | Goldfish simulator + goblin-storm metrics artifact + tests | in progress |
| C2 | Decision-scenario schema, validator extension, renderer badges + spreads + goldfish section | in progress |
| C3 | pilot-coach agent; write-manual v2 pipeline skill; author-decision skill | pending |
| C4 | Manual v2 build: matchups/threat sections, OG tags, manuals index gallery | pending |
| C5 | Regenerate Zada manual v2 end-to-end; resolve 1–2 more queued combo lines | pending |
| C6 | hapatra + edgar-vampires scaffolds; full runs when decklists land | blocked on decklists |
| C7 | docs/pilot.md v2 (tier contract headline), memory updates | pending |

## Risks / honesty notes

- Goldfish v1 simulates resource development, not games — assumptions render in the manual; overclaiming here would undercut the whole brand
- Decision trees are pure coaching: the review loop (tracked JSON, founder red-lines) is the quality mechanism, not machine verification
- Queued verified-line candidates (from the manual-writer): Fists of Flame lethal math, Empty the Warrens storm turn, Prospector+Krenko loops, Storm-Kiln+Haze buyback, Past in Flames rebuild, Roaming Throne doubling
