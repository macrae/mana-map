# Deck Building v2 — design plan

*Status: proposed, not started. Written 2026-07-25. The active summary lives in `PLAN.md`;
this is the full spec.*

## What we're building

Give the system a commander, a target bracket, and optionally a handful of cards you
already know you want. It returns a legal, functional, strategically coherent 99 — with a
written justification for every slot, a mechanically-proven bracket claim, and a simulated
proof that the mana actually works.

Then it publishes its own pilot's manual.

That last sentence is the point. We already turn a deck into a magazine issue with verified
combo lines, seeded simulations, and labelled coaching. Deck Building v2 closes the loop at
the front: **the same evidence contract that governs how we describe a deck should govern
how we choose it.** A builder that improvises "run 37 lands" is exactly the kind of
confident guess this whole subsystem was built to refuse.

## What already exists

A working Commander deck builder already ships in `viz/js/deck-builder.js` (~1,370 lines).
It is a genuine prototype, not a stub: embedding-centroid similarity, a six-factor score,
combo/synergy graph lookups, a pip-weighted mana base, plot overlays, localStorage
persistence. It should be respected and then largely replaced, for reasons that are
structural rather than aesthetic:

- **It doesn't model Commander.** Color identity is only enforced when a commander is set,
  and never re-checked against cards added earlier. There is no partner slot, no
  planeswalker or Background commander, no companion. The 4-copy rule for the other seven
  formats is never implemented at all — every format is accidentally singleton.
- **There is no "build me a deck."** Setting a commander only sets a color filter. The loop
  is seed → recommend 20 → accept, driven entirely by clicks. Filling a 99 means pressing
  Generate and Accept All about five times.
- **Every constant is a JS literal with no derivation and no test.** The six weights, the
  target curve `[0.03, 0.12, 0.25, 0.22, 0.18, 0.12, 0.08]`, the type distribution
  `{Creature: 30, Land: 36, …}` — invented inline, tuned against nothing.
- **The mana base is the weakest part.** Land scoring is `colors.size * 10 + …`, so any
  "add one mana of any color" land scores 50 and gets taken first regardless of what the
  deck needs; a five-color land decrements five color requirements at once, so the loop
  terminates early and backfills with basics of a single color.
- **The saved format is a list of raw row indices** into `projection_2d.json`, unversioned.
  A Scryfall refresh that changes card ordering silently reinterprets every saved deck as a
  different set of cards.
- **51 MB of JSON is parsed synchronously on the main thread** when build mode opens.

Reusable as-is: the CSS design system, `computeCentroid`/`embeddingSim`, the plot-overlay
seam, `countPips`, `renderManaCurve`, `renderColorDist`, and the positional-index card ID
convention (internally — just never persisted raw).

## Four findings that shape the design

**1. The signal we have is about *relatedness*. It says nothing about *quality*.**
Three well-formed graphs (83K combos, 26K synergy adjacency lists, 14K obsolescence
upgrades), two aligned 128-dim L2-normalized embedding spaces, and a clean 34,322-row card
table with hard filters for color identity and Commander legality — `legal_commander` even
carries the 83 banned cards correctly, so the ban list is free. What's missing is any
notion of how *good* or how *fast* a card is. `edhrec_rank` is a uniform global popularity
rank with no scale, no commander-conditioning, and 2,566 nulls; it pushes every
recommendation toward generic staples, which is the wrong bias for a synergy deck and
doubly wrong for a low bracket.

**2. The power-level signal we need is already on disk, and we throw it away.**
`data/combos_raw.json` carries per-variant fields that `process_combos.py` discards:
`bracketTag`, `popularity`, `manaValueNeeded`, `legalities.commander`, `easyPrerequisites`,
`notablePrerequisites`, a prose `description` of the line, and `prices`. The enum, read from
the Spellbook backend source:

| Tag | Label | Count | Maps to bracket |
|---|---|---|---|
| `E` | Exhibition | 71,165 | 1 |
| `C` | Core | 243 | 2 |
| `O` | Oddball | 204 | 2 |
| `P` | Powerful | 2,024 | 3 |
| `S` | Spicy | 2,782 | 3 |
| `R` | Ruthless | 5,468 | 4 |
| `B` | Banned | 1,375 | excluded |

85% being Exhibition is the expected shape, not a defect: most Spellbook variants are
trivial three-card "Infinite ETB" interactions that aren't win conditions. The ~10,000
Powerful/Spicy/Ruthless variants are the lines that actually decide a bracket. Recovering
these fields is roughly a 30-line change to one file and it is the highest
value-per-line work in the whole plan.

**3. The role taxonomy is not adequate for slot-filling.** `MECHANICAL_TAGS` is 33 regexes
built for retrieval and clustering, and it does that job well. For deckbuilding it fails in
specific ways: `ramp` conflates mana rocks, mana dorks, land ramp and rituals behind one
regex, so a Signet and a Dark Ritual score identically; there is no `wincon` tag at all; no
board wipe, sacrifice outlet, recursion, or stax distinction; `evasion_flying` fires on
removal that destroys flyers; and 6,875 cards (20%) carry no tag. You cannot ask "does this
deck need more interaction" from what's there.

**4. The strategy corpus has no deckbuilding theory, and the citation contract makes that
fatal.** Of 32 sections, roughly 10 contain a deckbuilding *clause* and **zero are about
construction**. There is nothing on land counts, ramp/draw/removal ratios, curve design,
tutor density, threat count, archetype selection, power-level matching, or budget. Karsten's
land-count work is cited three times as a *method* and not one number from it appears in the
doc. Since citations are validated as verbatim whitespace-normalized substrings, a builder
claiming "37 lands" literally cannot cite anything today — the number has to physically
exist in the prose. Under the repo's own zero-guessing rule, an architect agent would be
forced to flag essentially every construction decision as ungrounded.

**This is the gate.** Phase 0 exists because of finding 4.

## Architecture

Same shape as the manual pipeline, for the same reasons: a deterministic Python core that
earns the ◆ badge by being reproducible and tested, with a thin agent layer on top that
earns ★ by labelling itself as judgment and ✓ only by surviving the rules checker.

```
                 ┌─ deterministic, tested, zero LLM calls ──────────────┐
brief.json  ───► │ card_roles → candidate pool → slot fill → manabase   │ ───► build_plan
                 │ bracket_check · validate_build · validate_deck       │
                 └──────────────────────────────────────────────────────┘
                                        ▲   │
                          cited ★/✓ ────┘   └──── ◆ evidence
                 ┌─ agents, cache-gated, JSON only ─────────────────────┐
                 │ deck-architect ⇄ deck-critic                         │
                 │ deck-analyst · strategy-researcher · meta-analyst    │
                 │ pilot-coach · stack-resolver ⇄ rules-checker         │
                 └──────────────────────────────────────────────────────┘
```

The deterministic core must be able to produce a complete, legal, goldfish-passing 99 **with
no agent involvement at all.** That is a hard requirement, not a nicety. It gives us a free
baseline to A/B the agents against, it keeps the expensive path optional, and it means a
cache miss degrades to a worse deck rather than no deck.

### New deterministic modules

| Module | Does |
|---|---|
| `analysis/card_roles.py` | The deckbuilding role taxonomy. Deterministic rules over type line + oracle text, emitting `data/card_roles.json`. Reports its own coverage and an explicit `unclassified` bucket. |
| `analysis/similar.py` | The vectorized top-k with a color-identity mask that exists today only in JavaScript. ~30 lines; `argpartition(emb @ emb[i] * ci_mask, -k)`. |
| `pilot/bracket.py` | The bracket engine — see below. |
| `pilot/manabase.py` | Karsten source-count math: pip-weighted color requirements, untapped-source targets by turn, ETB-tapped budget, utility-land tax. Replaces the JS greedy set cover wholesale. |
| `pilot/build_deck.py` | Brief → candidate pool → role-slot fill under hard constraints → `build_plan.json`. Seeded and reproducible. |
| `pilot/validate_build.py` | Form gate over a build plan, in the house style: pure `validate(plan) -> list[str]`. |

Proposed role vocabulary — deliberately finer than `MECHANICAL_TAGS` where deckbuilding
needs it and coarser where it doesn't:

```
ramp:rock  ramp:dork  ramp:land  ramp:ritual  ramp:cost-reduction
draw:burst  draw:engine  draw:impulse  draw:wheel
removal:spot  removal:sweeper  removal:edict  removal:tax
tutor:unrestricted  tutor:narrow  recursion  protection:self  protection:granted
wincon:combat  wincon:combo  wincon:drain  wincon:alt
sac-outlet  stax  land:untapped-dual  land:tapped  land:fetch  land:utility  land:mdfc
```

A card carries several. Coverage is a published number with a target (say ≥90% of
Commander-legal non-lands carrying at least one role), and the shortfall is reported rather
than hidden — the same honesty the artist analysis applies to its `notes[]`.

### The bracket engine

This is the differentiated part, and it falls out of finding 2 almost for free.

WotC's bracket ladder is defined by deck-construction restrictions, and we can check nearly
all of them mechanically:

| Bracket | Name | Game Changers | Two-card infinites | Mass land denial | Tutors |
|---|---|---|---|---|---|
| 1 | Exhibition | 0 | none | none | minimal |
| 2 | Core | 0 | none | none | few |
| 3 | Upgraded | ≤3 | not early-game | none | some |
| 4 | Optimized | unlimited | unlimited | allowed | unlimited |
| 5 | cEDH | unlimited | unlimited | allowed | unlimited |

`bracket.py` computes a **bracket floor** for any 99:

- **Game Changer count** — set membership against a new tracked `data/game_changers.json`.
- **Combo content** — intersect the deck's name set against `combo_graph.combos`, take the
  maximum `bracketTag` present, and report the specific lines. This is the piece nobody
  else has: we can name the exact combo that pushes a deck out of Bracket 2.
- **Two-card infinite detection** — combos of length 2 whose `produces` contains an
  `Infinite …` feature, cross-checked against `manaValueNeeded` for the "early game"
  qualifier in Bracket 3.
- **Tutor density** — from the role classifier's `tutor:unrestricted` count.
- **Mass land denial** — a small curated name/text list; the honest weak spot, and it should
  be a reviewed list rather than a regex.

The output is a ◆ artifact: a claimed bracket, a computed floor, and the evidence for each.
When the two disagree, that is a finding, not a rounding error. It also runs on *existing*
decks — pointing it at goblin-storm on day one is the cheapest possible validation.

Two honest caveats to carry into implementation:

- The Game Changers list must be fetched from the **official WotC page**, not an aggregator.
  Sources currently disagree on the count (53 vs 56 depending on date and site), the list is
  revised periodically, and the artifact needs a `source_url` + `as_of` + `sha256` sidecar
  exactly like `.rules-meta.json`. A stale Game Changers list produces a confidently wrong
  bracket claim, which is worse than no claim.
- "Intent of play is the most important part" is WotC's own framing — the brackets are
  explicitly *not* a calculator. We compute a floor and say so. We never tell a player what
  bracket their deck is; we tell them what their deck contains.

## New agents

The repo's pattern is specialist pairs — `stack-resolver` ⇄ `rules-checker`,
`strategy-researcher` in two modes. Three new agents, plus one existing one upgraded.

**`deck-architect`** — proposes the build. Read-only, returns JSON. Given the brief, the
candidate pool, the strategic frame and the meta frame, it emits a role budget (how many
ramp, draw, interaction, threats, and *why*, each with a `strategy:<id>` citation), the
engine packages, the must-includes, and a per-slot justification. It never picks cards the
deterministic pool didn't surface, so it cannot hallucinate a card that doesn't exist.

**`deck-critic`** — adversarial, mirroring `rules-checker`, with a closed status set
(`supported | unjustified | miscounted | off-bracket | off-identity | unverified-line`) and
a `pass`/`fail` verdict that `validate_build` cross-checks for consistency. Its job is to
try to break the plan: does the role budget match the declared archetype and cite real
sections; does the bracket claim survive `bracket.py`; do the claimed engines actually work
or do they need a stack scenario; does the mana base support the pips. It never
rubber-stamps, and a `fail` artifact is saved as an open question rather than deleted.

**`meta-analyst`** — the "hyper aware of strategy metas and playstyles" role. Research-mode,
write-scoped to `data/meta/` with a CHANGELOG, mirroring `strategy-researcher`'s two-layer
write guard (prompt declares scope, skill reverts anything else via `git status`
snapshotting). Separate corpus from `strategy.md` for one specific reason: **strategy theory
is durable and meta claims are perishable.** Every meta section carries an `as_of` date, and
a build that cites a meta section older than some threshold gets a staleness warning. What a
Bracket 3 pod looks like this month is a fact with a shelf life; "who's the beatdown" is not.

**`deck-analyst`** — keep, upgrade. It is already the correct evidence role and the only
agent that knows the index-alignment gotcha and the synergy-versus-similarity distinction.
Today it has no output schema, no artifact, and therefore no cache routine — every manual
regeneration pays for it. Give it a JSON schema, a `candidate_pool.json` artifact, and a
cache entry.

Not a new agent: the mana base. It is a solved math problem and belongs in Python, where it
can be tested. An agent picking utility lands is a taste call we can add later if the
deterministic version proves boring.

## The loop

```
brief.json                          (authored: commander, bracket, budget, must-include/exclude, playstyle)
  ↓
deck-analyst        → candidate_pool.json     ◆  role-bucketed shortlists with scores
strategy-researcher → build_frame.json        ★  archetype, role budget, cited
meta-analyst        → meta_frame.json         ★  pod expectations, as_of dated
  ↓
deck-architect      → build_plan.json         ★  the 99, per-slot justification
  ↓
validate-build                                   form gate
build-manabase                                ◆  folded into the plan
bracket-check                                 ◆  floor vs claim
  ↓
deck-critic         → findings                   ≤3 iterations, then stop
  ↓
write decklist.txt → fetch-deck → validate-deck   the real legality gate (100 cards, singleton, CI)
  ↓
goldfish                                      ◆  targets auto-derived from the plan's engines
pilot-coach         → build_review.json       ★  does it actually play?
  ↓
/resolve-stack per claimed win line           ✓  the lines get verified like any other
  ↓
... straight into the existing manual pipeline
```

Two details worth flagging.

**Goldfish targets stop being hand-authored.** Today `goldfish_targets.json` is a judgment
call someone writes by hand — which key-piece sets are worth simulating. A builder that
declared its own engines already knows the answer. Deriving targets from `build_plan.engines`
removes one of the three manual files from the deck-onboarding path and makes the fitness
function a genuine closed loop: the deck states its plan, the simulator tests that plan, and
a low assembly rate is a build defect rather than a mystery.

**The critic loop is bounded at 3, like `resolve-stack`.** If a plan can't satisfy the critic
in three passes, that's a finding about the brief, not a reason to keep spending tokens.

## Cache wiring

The existing routines all take `cards:semantic`, which digests a `cards.json` that by
definition does not exist before a build. `resolve_inputs` would raise `MissingInput` → exit
2 → "stop, don't spawn". So the build routines need a different input vocabulary:

```python
"deck-build": {
    "agent": "deck-architect+deck-critic",
    "artifact": "build_plan.json",
    "inputs": ["deck:brief.json", "deck:candidate_pool.json", "deck:build_frame.json",
               "deck:meta_frame.json?", "global:COMBO_GRAPH_PATH", "global:SYNERGY_GRAPH_PATH",
               "global:CARD_ROLES_PATH", "global:GAME_CHANGERS_PATH",
               "strategy:doc", "meta:doc"],
},
```

Three additions to `agent_cache.py`: a `meta:doc` token (bytes of the meta corpus, same
treatment as `strategy:doc`), a `roles:version` token so a reclassification invalidates
builds, and — if we want iterative revisions as `build:001`, `build:002` — an extension to
`_DYNAMIC_RE`, `routine_spec` and `discover_routines`, which is the same three-edit change
the stack/decision families already made. `global:CARDS_CSV_PATH` works mechanically but is
a 23 MB hash on every status call; the existing `(path, mtime_ns, size)` memoization covers
it.

`AGENT_CACHE_VERSION` only bumps if hashing semantics change. Adding routines doesn't
require it.

## Phasing

Each phase ends in something usable. Nothing after Phase 0 is blocked on agent work.

**Phase 0 — ground truth.** *Gate for everything else.*
Strategy research pass on deckbuilding: ~14–18 new sections under a `strategy:deckbuilding.*`
namespace, roughly three research passes at the measured ~100k tokens each. Priority order:
mana base (the one thing cited three times and quoted zero of, with Karsten's numbers
physically in the prose so they can be cited), ratios, redundancy-vs-tutors (a total blank
today — tutors are never mentioned once in 32 sections), power-level and brackets, curve,
threat density, archetype selection, cutting, budget. Plus: fetch the Game Changers list into
a tracked artifact with a meta sidecar, and recover the discarded Spellbook fields.

Design constraint to honour up front: `STRATEGY_SECTION_WARN_CHARS` is 1200 and MiniLM
truncates around 256 tokens, so **numeric tables do not belong in `strategy.md`.** Split it —
short dense prose carrying the specific citable numbers in the doc, computed tables in code
as ◆ artifacts. The goldfish precedent is the model.

Section ids are append-mostly and never reused, so pick the `strategy:deckbuilding.*`
namespace deliberately in one go; renaming later costs a changelog bullet per id and
invalidates every routine taking `strategy:doc`.

**Phase 1 — deterministic core.** `card_roles.py`, `similar.py`, `bracket.py`,
`manabase.py`, `build_deck.py`, `validate_build.py`, all tested, all registered in
`PILOT_STEPS`. Milestone: `manamap pilot build-deck <slug>` produces a legal, goldfishable
99 with zero agent involvement, and `manamap pilot bracket-check goblin-storm` returns a
defensible floor for a deck we already understand.

**Phase 2 — the agent loop.** `deck-architect`, `deck-critic`, `meta-analyst`; the
`/build-deck` skill; `deck-analyst` upgraded with a schema and artifact; cache routines and
the new input tokens. Milestone: a built deck whose every ratio traces to a citation and
whose bracket claim survives the critic.

**Phase 3 — close the loop.** Goldfish targets derived from the plan; `pilot-coach` review;
the hand-off that turns a built deck into a magazine issue. Milestone: brief in, published
manual out.

**Phase 4 — the frontend.** The browser stops being the builder and becomes the
inspector/editor for a Python-built plan. This is also where v1's structural liabilities get
paid down, in priority order: the unversioned raw-index localStorage schema (a correctness
bug, not a nicety — it silently corrupts saved decks on any pipeline refresh), the 51 MB
synchronous parse, and the full-`innerHTML`-rebuild render model with its unbounded document
listener leak.

## Risks and open decisions

**Decisions I'd make without asking, flagged so they can be overruled:**

- *Tier means WotC bracket, not a custom scale.* It's the format's actual shared vocabulary,
  it's mechanically checkable against data we have, and inventing a parallel power scale
  would be the kind of unverifiable claim this repo exists to avoid.
- *The deterministic core must stand alone.* Agents improve a build; they are never required
  to produce one.
- *CLI and artifacts first, UI last.* It matches the pilot subsystem, it's testable, and the
  frontend work is genuinely independent.
- *Meta gets its own corpus, not a section of `strategy.md`*, because dated claims and
  durable theory have different shelf lives and should invalidate differently.

**Real risks:**

- **The role classifier is the highest-risk deterministic component.** If coverage lands at
  70% the slot filler is guessing. Mitigation: measure coverage first on a sample, publish
  the number, and treat `unclassified` as a first-class reported bucket rather than a silent
  fallback.
- **`edhrec_rank` is a bad signal we're currently weighting at 0.10.** It's a uniform rank
  with no scale, and it biases toward generic staples — precisely wrong for a synergy deck
  and for low brackets. At minimum log-transform it; consider dropping it from the
  deterministic score and letting the architect use it as context.
- **No prices and no per-commander inclusion rates.** Budget conditioning needs a new fetch;
  Scryfall carries per-card prices we currently strip in two places, and `combos_raw.json`
  carries per-combo prices. Inclusion rates — the real staples signal — aren't in any bulk
  data we have. Budget is a Phase 3+ concern, and the plan should say "budget unsupported"
  rather than approximate it.
- **A stale Game Changers list produces confidently wrong bracket claims.** Sidecar it,
  date it, and fail loudly rather than serve stale, the same way `load_strategy_db` hard-errors
  on a sha256 mismatch.
- **`bracket.py` will find things about goblin-storm.** Running it on a published deck may
  contradict something already in Vol. 001. That's the system working, and it should be
  published as a correction rather than quietly reconciled.

**Genuinely open, worth a decision before Phase 2:**

1. How much of the 99 does the system *fill* versus *propose*? Filling all 99 is the
   "automagic" ask; proposing packages with alternates is more honest about how deckbuilding
   actually works. My lean: fill completely, but every slot carries 2–3 named alternates so
   the plan reads as a starting point rather than a verdict.
2. Does a built deck get its own `data/decks/<slug>/` directory immediately, or a staging
   area until someone commits to it? Leaning: same directory, with `brief.json` as the
   authored root — it makes the hand-off to the manual pipeline free.
3. Whether `deck-critic` can *edit* the plan or only report findings. The `rules-checker`
   precedent says report only, and I'd follow it — the architect owning its own revisions is
   what makes the iteration count meaningful.

## Verification

- Every new module has unit tests in the house style — inline fixtures, no fixture files,
  paths from `manamap.config`.
- `build-deck` twice on the same brief → byte-identical `build_plan.json`.
- A built deck passes `validate-deck` (100 cards, singleton, color identity, legality) —
  the same gate goblin-storm passes.
- `bracket-check` on goblin-storm and on the two scaffolded decks, reviewed by hand against
  what we know about them.
- Goldfish on a built deck reaches a stated assembly threshold, or the build reports why not.
- `card_roles` coverage published as a number, with the unclassified tail sampled and
  eyeballed.
- No agent claim without a citation that `validate_build` can verify as a verbatim substring.

## Prerequisite worth doing regardless

`PLAN.md` records a real open data question: `goldfish_metrics.json` reports
`keep_first_seven_rate` of 0.791 while its own `land_histogram` puts 9,899 of 10,000 first
sevens at 2–5 lands, and under the stated keep rule those two figures cannot describe the
same test. Goldfish becomes this feature's fitness function. **Resolve that discrepancy
before Phase 3 depends on it.**
