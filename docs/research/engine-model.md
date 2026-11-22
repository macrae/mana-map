# The engine model, rebuilt: a computed event graph

*A research memo for the pilot, 2026-11-22. Every figure here is either measured in this repo
(the command or script is named) or sourced (the link is given). Nothing in this memo changes a
file; it is the argument the code will be written against.*

## 0. The question, and the answer in one paragraph

What is a Commander deck's **engine**, how do we know, and how should the bench model it so that
Jarvis, the pages, the goldfish and the agents all read one thing? Today the bench has four
answers that do not agree — an agent's eight-stage prose, a hand-written list of card groups, a
strategic frame's prose, and a brief's `resources` field — and a card map built from an embedding
that was trained to see theme, not function. The answer this memo argues for: **an engine is a
graph of typed events between cards.** Each card's abilities say what events they consume
(triggers, costs) and produce (effects); an edge runs from a producer to a consumer of the same
event; stages are positions in that flow, lines are paths, redundancy and single points of
failure are graph properties. The graph is **computed** from Forge's card scripts — a typed
grammar we already ship for 99.7% of the fleet's cards — and from the goldfish's own readers where
Forge is silent. Every edge carries its evidence tier. The pilot's and the agents' words sit on
top as a stamped `reading`, never inside the computed record. Card similarity then falls out of
the same signatures, and the deck map draws the engine instead of disagreeing with it.

## 1. What the bench has today, measured

### 1.1 Four engine models

**`engine.json`** is written by the `deck-engineer` agent and attacked by `engine-critic`
through `/analyze-engine`: a closed set of eight stages (`mana, ignition, fuel, fodder,
conversion, output, protection, wincon`; `validate_engine.STAGES`), every card placed in exactly
one, `lines[{from, to, via, verified_by}]` where the only fact tier is a checker-passed stack
scenario naming every `via` card. It costs about 120k tokens per engineer pass and 140k per critic
pass, and it goes stale on any list change (`cards:semantic` is a cache input).

| deck | lines | verified | critic | map disagreements | list stamp |
|---|---|---|---|---|---|
| radagast | 9 | 9 | pass | 5 | no |
| yawgmoth-swarm | 15 | 11 | pass | 5 | no |
| goblin-storm | 18 | 10 | pass | 5 | no |
| ur-dragon | 9 | 6 | pass | 6 | yes |
| sharknado | 8 | 5 | pass | 5 | no |
| hapatra | 18 | 3 | pass | 5 | no |
| sisay | 11 | 3 | pass | 5 | no |
| keen-engineering | 12 | 2 | pass | 5 | no |
| edgar-vampires | 10 | 1 | pass | 6 | no |
| heliod | 10 | 1 | pass | 5 | yes |
| gishath | 8 | 0 | pass | 7 | yes |

Eleven files; 51 of 128 lines verified; eight carry no decklist stamp, so `deck-status` cannot see
them age; `cache-status --routine deck-engine` reports nine of eleven stale; four were last
touched 2026-08-14. The schema has no slot for an *event*: heliod's own `notes` say "the closed set
of eight has no slot for the most characteristic thing about it." Direction lives in an
undocumented `carries` key that seven of eleven files happen to have.

**`goldfish_targets.json`** is hand-written: channel flags (`model_draw`, `model_combat`, …) and
`targets[{label, need:[{any_of:[names]}], required?, route?}]`. It is the source of **every
computed engine number** in the repo — `diagnostic._engine` (`engine_online` = P(all required
targets assembled by turn N), `any_route`), `deck_audit.engine_activation` (hypergeometric per
component, single points of failure), the cover's `engine_health` word, the handbook's "Engine
assembly" chart, `sim/engine_casts`' engine set. None of them reads `engine.json`. Fourteen
decks declare targets; `required` is set on five, `route` on four. On sharknado the two models
disagree about the one stage the charter calls "the one to get right": the declaration says
ignition is *a wheel drawn — nothing else in the deck starts itself*; `engine.json` says ignition
is Shabraz. `engine_lift` — a lift measured over these declared groups — was deleted on
2026-08-28 after three defensible declarations of one Ur-Dragon list read +0.007, −0.036 and
+0.014 against the same games: *a measure computed from an authored file is not evidence*, and
`deck_branch.MEMBERSHIP_AXES` still refuses `engine_online` as a branch objective for that reason.

**`strategic_frame.json`** carries `engines[{piece, engine, evidence, strategy_ref}]` in prose.
Edgar's names Bloodthirsty Conqueror as "the deck's kill" with "8 two-card infinites"; the card is
not in the list and `combos.json` holds zero included lines. **`brief.json`** carries a
`resources` field on two decks (edgar, meren), read by `resource-scan`.

### 1.2 The map is the wrong instrument for this, by construction

`deck_map.py` lays a deck out from **CardBERT** — the embedding chosen for the pages on
2026-10-10 because it wins on theme: against pools of 100 and 500 it scores +0.09 / +0.15 on
theme and −0.21 / −0.28 on function (`docs/architecture.md`, the re-measure of 10-10). Its
clusters are Ward agglomerative (the file's `meta.clustering` says average-linkage; a mislabel),
four to seven "cities". The engine is about what cards do *to each other*, so
`engine_facts.component_scatter` finds three of nine declared components in one city on
sharknado, and every `engine.json` carries five to seven `map_disagreements` saying the same
thing in different words: "draw sources are clustered on WHAT ELSE their text says". The
charter's framing is right and its conclusion is the problem: "the clusters are an input to engine
analysis, not the analysis" — a map that has to be argued against on every deck is not an input.

Roles are regex (`ROLE_PATTERNS`, 53 roles in 19 families); the synergy graph is 26 tag rules,
retrieval only; the ability space is InfoNCE over role/tag positives (r@10 0.232) and reads tribe
as nothing by design. **Nothing in the bench models an interaction between two specific cards
except a hand-written stack scenario.** That is the gap.

### 1.3 What is already in the house

- **Forge's card scripts.** `~/.mana-map/forge/res/cardsfolder/cardsfolder.zip.orig` holds
  33,587 scripts in a typed grammar. Measured over the archive (script in §7): 16,925 `T:`
  trigger lines over 137 trigger modes — ChangesZone 7,511, Phase 2,360, Attacks 1,599,
  SpellCast 1,401, DamageDone 873, Drawn 149, Sacrificed 115, LifeGained 90, Discarded 83 — 18,414
  `A:` activated/spell lines, 7,080 `S:` statics, 1,692 `R:` replacements; 191 effect APIs
  (ChangeZone 6,599, Pump 4,969, Draw 3,714, Token 3,555, PutCounter 3,275, DealDamage 2,850,
  Mana 2,497, Effect 1,912, GainLife 1,770, Destroy 1,542, LoseLife 1,211, Discard 1,097). Forge's
  own deck-building hints ride on 7,478 scripts (`DeckHas:Ability$…` 7,314), with `DeckHints`
  on 4,081 and `DeckNeeds` on 1,210; 4,955 carry `AI:RemoveDeck`. Blood Artist is literally

  ```
  T:Mode$ ChangesZone | Origin$ Battlefield | Destination$ Graveyard | ValidCard$ Card.Self,Creature.Other | Execute$ TrigLoseLife
  SVar:TrigLoseLife:DB$ LoseLife | ValidTgts$ Player | LifeAmount$ 1 | SubAbility$ DBGainLife
  SVar:DBGainLife:DB$ GainLife | Defined$ You | LifeAmount$ 1
  ```

  — *consumes a creature death, produces a life loss and a life gain*. Shabraz consumes `Drawn`
  by `Card.YouCtrl` and produces `PutCounter P1P1` on itself plus `GainLife`. Edgar consumes
  `SpellCast` of `Card.Vampire+Other` by `You` with `TriggerZones$ Command` and produces a
  `Token b_1_1_vampire`; the token's own script (`res/tokenscripts/`, 836 files) says
  `Colors:black, Types:Creature Vampire, PT:1/1`, which is exactly what Teysa's or Sanctum Seeker's
  filter needs to know. `sim/forge_cards.py` already opens the zip.
- **The goldfish's vocabulary.** `model_coverage.CHANNELS` (mana, bodies, tutor, target, treasure,
  combat, draw, sacrifice, drain, attack_enabler, discard, recursion, discover, cascade),
  `channels_for(profile)` and `card_state` (seen / partial / dark / invisible) — a per-card
  coverage statement the new graph should mirror, so `try`'s coverage line and the engine agree.
- **Pricing and stamps.** `deck_audit._component` (hypergeometric availability of an `any_of`
  group by turn), `deck_branch.stacks_naming` (which checker-passed stacks name a card),
  `sim/engine_casts` (cast / held-while-castable / idle per card at a real table), `deck-notes
  tally` (the pilot's own over / under / as-expected reads per card), `deck_combos`
  (Commander Spellbook lines contained or one short), and the strategy companion's
  `conversion.*` family (payoff-density, yield, availability, durability, substitution…).

### 1.4 Fleet coverage of the scripts (measured)

22 decks, 1,504 card rows, 1,072 distinct names. Four names have no script at all — Edgar,
Ancient Bloodlord; Ingris Stingerquill; Mishra, Lost to Phyrexia; The Echoverse Fulcrum — all
recent printings. Per deck:

| deck | cards | no script | no T/A/S/R line | loose effects¹ | board-X² | AI:RemoveDeck | with a trigger |
|---|---|---|---|---|---|---|---|
| sharknado | 99 | 0 | 2 | 6 | 12 | 12 | 17 |
| edgar-vampires | 97 | 2 | 2 | 7 | 11 | 13 | 33 |
| heliod | 85 | 0 | 3 | 4 | 14 | 17 | 21 |
| gishath | 93 | 0 | 3 | 8 | 17 | 6 | 27 |
| goblin-storm | 79 | 0 | 1 | 8 | 15 | 5 | 21 |
| ur-dragon | 97 | 0 | 4 | 6 | 9 | 11 | 29 |
| keen-engineering | 67 | 0 | 3 | 2 | 13 | 10 | 24 |
| meren-recursion | 97 | 0 | 1 | 6 | 14 | 17 | 37 |
| emiel-blink | 88 | 0 | 3 | 7 | 20 | 8 | 56 |
| valgavoth | 88 | 2 | 3 | 8 | 13 | 15 | 40 |

¹ the card's script leans on `Effect`, `Animate`, `Charm`, `GenericChoice` or `Branch`, which
Forge models loosely. ² the script carries a `Count$`/`TriggerCount$`/`ReplaceCount$` amount,
i.e. the quantity depends on the board. The "no T/A/S/R line" cards are vanilla or keyword-only
(a Sol Ring is `A:AB$ Mana`; a Glorious Anthem is `S:`; a vanilla 2/2 has nothing to parse, and
that is a correct reading).

The pilot's own evidence available for validation: 122 per-card reads in the captain's log
debriefs — ur-dragon 35, edgar 32, sharknado 21, goblin-storm 14, gishath 11, heliod 9 — each
`over | under | as-expected | missed` with its game id.

## 2. How decks are modelled elsewhere, and what each lens cannot see

**Theory.** Card advantage and tempo are the oldest frame: Chapin's *Next Level Magic* preview
pieces separate raw card advantage from virtual card advantage (cards that are live against the
opponent's plan) and set tempo against both as mana spent efficiently
([SCG, card advantage](https://articles.starcitygames.com/articles/next-level-magic-preview-card-advantage-and-virtual-card-advantage/);
[SCG, tempo](https://articles.starcitygames.com/articles/next-level-magic-preview-tempo/)). Flores's
*Who's the Beatdown* (1999) assigns roles per matchup and is the pivot-point logic the strategy
companion already carries under `strategy:whos-the-beatdown`
([SCG, eight core principles](https://articles.starcitygames.com/articles/eight-core-principles-of-whos-the-beatdown/)).
Resource conversion — a deck turns one resource into another at a rate, and the rate is what
matters — is in-house as `strategy:conversion.*` (payoff-density, yield, availability,
durability, entries-vs-deaths, substitution). Enabler/payoff and setup/payoff ratios are the
Commander community's working vocabulary; EDHRECast 117 poses "how many cards should enable the
deck and how many should help it win" as the question, and no retrievable source gives a
number ([EDHRECast 117](https://edhrec.com/articles/?p=52473)). Critical mass and redundancy vs
tutors are deckbuilding heuristics the companion holds under `deckbuilding.*`. Karsten's
hypergeometric mana math is the one piece of this that is a measurement rather than an opinion:
how many sources make a colour castable on curve at 90% ([Karsten, sources update, via Library of
Leng](https://library-of-leng.com/articles/tcgplayer-dc23a7d2-0a16-4c0b-ad36-586fcca03ad8); the
original ChannelFireball URLs now 404). **What this lens sees:** one resource or one ratio at a
time. **What it cannot see:** which card feeds which. "Ten card-draw spells" says nothing about
whether any of them is read by the card that pays off a draw.

**Category templates.** The Command Zone's template (ramp 10–12, card draw 10, removal 10–12,
mass removal 3–4; [episode 658](https://edhrec.com/articles/the-command-zone-commander-deckbuilding-template-for-the-new-era-the-command-zone-658-mtg-edh-magic-gathering))
and the 8x8 and 7x9 variants are hypergeometric reasoning about *categories*; episode 659 is the
template's own critique — that counting categories flattens the difference between a card that
is one of nine and a card that is the only one that does its job
([episode 659](https://edhrec.com/articles/the-command-zone-the-problem-with-deckbuilding-templates-the-command-zone-659-mtg-edh-magic-gathering)).
`deck_audit`'s sixteen axes are this lens, done properly with intervals. It cannot see an engine
either; a category is a property of a card, not a relation between two.

**Co-occurrence.** EDHREC replaced its synergy percentage with **lift** in 2025: lift(A,B) =
P(A∩B) / (P(A)·P(B)) over the "colour universe" of decks that could hold both, displayed as
log-lift so that 2 means a hundred times more likely than chance and −1 means a tenth; it stays
near 1 for ubiquitous cards (over 50% inclusion: Sol Ring, Command Tower, Swords); the median
across card pairs is 1.58, which the author attributes to the non-randomness of Commander
deckbuilding ([EDHREC, From Synergy to Lift](https://edhrec.com/articles/from-synergy-to-lift-the-math-behind-edhrecs-new-era)).
Ward et al. build a synergy graph for draft from co-pick ratios and project it with
multidimensional scaling; its clusters are mostly colour ([arXiv 2009.00655](https://arxiv.org/abs/2009.00655)).
Xue & Hosobe score synergy from co-occurrence networks of elite decklists for a generic card game
([SciTePress 2025](https://scitepress.org/Papers/2025/132569/132569.pdf)). **What this lens
sees:** what players actually run together, including interactions no parser can read (politics,
sequencing, a card that is simply good). **What it cannot see:** why. Lift between Blood Artist
and Viscera Seer is high; lift does not know that one produces the death the other consumes. It
is also a statement about the field's decks, not this one.

**Learned card representations.** Bertram, Fürnkranz and Müller learn representations from card
attributes, images and third-party usage data to predict human deck-building picks, and reach
55% of human choices on cards never seen in training ([arXiv 2407.05879](https://arxiv.org/abs/2407.05879)).
Rigaux & Kashima learn set-contextualised card embeddings from draft sequences to predict a
drafted deck's win rate ([arXiv 2607.04782](https://arxiv.org/abs/2607.04782)). Our own CardBERT
and ability spaces are this family. **What it sees:** resemblance, at the granularity its
training signal chose — theme for CardBERT, role/tag for the ability space. **What it cannot
see:** direction. An embedding is symmetric; "feeds" is not.

**Simulation.** Westendorp's EDHREC piece simulates available mana with ramp and draw packages
against a baseline and notes that "most repeated card draw effects are board-state related and
cannot currently be modelled" ([EDHREC 2021](https://edhrec.com/articles/simultaing-available-mana-beyond-the-hypergeometric-distribution)).
Our goldfish is the same instrument grown large: ten thousand seeded games, every channel opt-in,
no opponents, no blockers, no removal. Forge is the opposite end: a real table with an AI that
will not cast what it has no logic for. **What this lens sees:** rates with intervals — how often
a declared component assembles, how many cards a wheel draws. **What it cannot see:** anything it
was not told to read; a card the model cannot read looks exactly like a card that does not help.

**Forge's own hints.** `DeckHas` / `DeckHints` / `DeckNeeds` are the engine's drafting
heuristics (a card that *has* tokens raises cards that *need* tokens). They ride on 22% of
scripts and Forge's developers note they cannot express counter synergies. A hint, not a model.

**The bound on all of it.** Churchill, Biderman and Herrick show Magic is Turing-complete
([arXiv 1904.09828](https://arxiv.org/abs/1904.09828)): whether a given board wins is not
computable in general. That is why the bench's ✓ tier stays with a human-written stack scenario
and a rules checker, and why no graph property below is ever called a proof.

## 3. The model

### 3.1 Flows and stocks

A card's abilities do one of three things. A **trigger** consumes an *event* ("whenever a
creature dies"). A **cost** consumes a *stock* ("sacrifice a creature", "{2}{B}", "discard a
card"). An **effect** produces events and moves stocks ("create a token" produces an
enters-the-battlefield event and adds to the BODIES stock; "draw a card" produces a draw event and
adds to HAND). Replacement effects and statics **scale** events (Teferi's Ageless Insight: `R:Event$
Draw … ReplaceWith$ DrawTwo`; Torbran: `R:Event$ DamageDone … DmgPlus2`; Lae'zel: `R:Event$
AddCounter … AddOneMoreCounters`) or **protect** nodes (hexproof, counters, regeneration, blink).

So the deck is a directed multigraph. Nodes are the ninety-nine, the commander(s) in the command
zone (available from turn one, availability 1), the tokens the deck can make (resolved from their
token scripts), and a few pseudo-nodes: TURN (the source of phase events), OPPONENT (a stationary
producer of the events opponents cause — their casts, their draws — at authored rates), the typed
stocks (HAND, BODIES, MANA, LIFE, LIBRARY, GRAVEYARD), and the terminal sink (opponent life to
zero, ten poison, "you win the game"). Edges:

- **feeds**: A produces an event that matches B's trigger filter. Subject filters form a lattice
  (Creature ⊃ Vampire; `Other`, `Self`, `YouCtrl`, `OppCtrl`; zones from/to).
- **supplies**: A's output stock matches B's cost (Edgar's tokens supply Viscera Seer's
  `Sac<1/Creature>`).
- **scales**: A's replacement or static matches an event kind some node produces.
- **protects**: A's target filter matches a node.

**An engine is a cycle that closes through a stock.** Edgar: a Vampire spell is cast → Edgar makes
a token → the token is a body → an outlet eats the body → the death triggers Blood Artist → an
opponent loses life, you gain it → Vito turns the gain into more loss; and, through the second
stock, Ashnod's Altar turns the body into two mana that casts the next Vampire. A **pure event
cycle** with no stock decrement is an infinite — Spellbook's domain, and ✓ only when a stack
scenario resolves it. A **DAG** from ignition to a terminal is a line, not an engine; a deck can
be all lines (many Commander decks are) and the model says so instead of inventing a loop.

### 3.2 The typed event vocabulary

A closed registry in `config.py`, one home. Forge's names are kept beside ours so a reader can
check any mapping against the script.

| kind | parameters | produced by (Forge API) | consumed by (Mode / Cost / R:Event) |
|---|---|---|---|
| ZoneChange (ETB, DIES, LTB, MILL, RECURSION as named views) | subject filter, from, to, controller | ChangeZone, ChangeZoneAll, Destroy, Sacrifice, Mill | ChangesZone(Origin/Destination), ChangesZoneAll, Destroyed, Sacrificed, Exploited |
| Token (TREASURE as a view) | type(s), P/T, count, controller, token script | Token, CopyPermanent, Amass, Investigate | as ZoneChange→Battlefield; R:CreateToken |
| SpellCast | spell filter (type/subtype/mv), caster, from zone | HAND stock → cast | SpellCast, SpellCastOrCopy, AbilityCast; R:Counter |
| Draw | player, count | Draw, Dig | Drawn; R:Draw |
| Discard / Cycle | player, filter, count | Discard, Cost$ Discard | Discarded, DiscardedAll, Cycled |
| LifeGain / LifeLoss | player, amount | GainLife, LoseLife, lifelink | LifeGained, LifeLost; R:LifeReduced / LifeGained |
| Damage / CombatDamage | source, target, amount, combat? | DealDamage, DamageAll, Fight, any body | DamageDone, DamageDoneOnce, DamageDealtOnce; R:DamageDone |
| CounterAdd / CounterRemove | type, object filter, count | PutCounter, PutCounterAll, Proliferate | CounterAdded, CounterAddedOnce; R:AddCounter |
| Attack / Block | attacker filter, count | combat | Attacks, AttackersDeclared, Blocks, AttackerBlocked |
| Tap / Untap / TapsForMana | object filter | Tap, Untap | Taps, Untaps, TapsForMana |
| Mana / CostReduction | amount, colours, restriction | Mana (Produced$, Amount$); S:ReduceCost | Cost$ mana — a STOCK |
| LandPlayed | land filter | — | LandPlayed |
| Tutor | zone from, filter | ChangeZone Origin=Library, Dig, Discover | — |
| Removal / Counterspell | target filter | Destroy, ChangeZone→Exile on OppCtrl, Counter | BecomesTarget |
| PhaseStep | step, whose turn | TURN | Phase |
| Terminal | opp life → 0, poison ≥ 10, WinsGame | LoseLife, DealDamage, PoisonCounter, WinsGame | OPPONENT sink |
| Scales (edge) | event kind, transform, filter | R:Event$ Draw / CreateToken / AddCounter / DamageDone / LifeReduced; S:Continuous; ReduceCost | — |
| Protects (edge) | target filter | Pump KW$ Hexproof / Indestructible, Counter, Regenerate, blink | — |
| Stocks (pseudo-nodes) | HAND (typed), BODIES (typed), MANA, LIFE, LIBRARY, GRAVEYARD | the effects above | Cost$ Sac / Discard / PayLife / tapXType / ExileFromGrave |
| **Unmapped** | the Forge name, verbatim | — | — |

The fleet uses 28 of the 137 trigger modes and 60 of the 191 effect APIs. Anything outside the
registry is **reported with its Forge name and never guessed**; the edge it would have made is
absent, and the card's coverage says why. Every `model_coverage.CHANNELS` key maps to at least
one registry row, so the goldfish's channels and the graph's events are one vocabulary.

### 3.3 Derivation rules

1. **Nodes**: the 99 + commander(s) (`TriggerZones$ Command` recognised; availability 1) + TURN,
   OPPONENT, the typed stocks, the terminal sink.
2. **Edges**: `feeds` on an event match that passes every filter atom the parser understands;
   `supplies` on a stock match; `scales` on a kind match; `protects` on a target match. An
   unparsed filter atom (`IsRemembered`, `Valid$ Card.cmcGE4` and the like) makes the edge
   *unproven* and listed, never silently dropped and never silently kept.
3. **Mana**: a node that produces only Mana / CostReduction / LandDrop and consumes only Tap /
   PhaseStep is the substrate — in the graph, excluded from cycle computation. Ashnod's Altar
   consumes a sacrifice, so it is not mana; it is conversion.
4. **Ignition**: a node with in-degree 0 on feeds/supplies (ignoring TURN and mana) and a path to a
   cycle or a terminal; the commander in the command zone always qualifies. This settles the
   sharknado disagreement in the declaration's favour: a wheel starts the deck; Shabraz converts.
5. **Fuel**: in-edges only from OPPONENT / TURN / ignition, out-edges into conversion (Rhystic
   Study, Welcoming Vampire).
6. **Fodder**: produces a stock that a `supplies` edge consumes, and nothing else (token makers
   under an outlet; Bloodghast).
7. **Conversion**: in-edge and out-edge on feeds/supplies; members of any strongly connected
   component computed with the untyped HAND removed (Shabraz, Edgar, Viscera Seer, Blood Artist).
8. **Output**: out-edge is a Terminal, or `scales` a Terminal event (Brallin, Torbran, Bloodletter).
9. **Wincon**: an Output whose single resolution is lethal (amount ≥ 40, or X-scaled with a
   verified stack), a WinsGame effect, or a pure event cycle Spellbook names and a stack proves.
   Otherwise the deck wins *by accumulation*, and the model says so rather than promoting a card.
10. **Protection**: `protects` out-edges only; not in the flow.
11. **One primary stage per card**, by priority wincon > ignition > conversion > output > fuel >
    fodder > protection > mana, plus `also[]`. Edgar is ignition *and* conversion; the current
    one-stage rule cannot hold and the renderer will show both.
12. **Lines**: simple paths ignition → terminal over feeds/supplies/scales, deduped on card set,
    ranked ✓ first then by the product of hypergeometric availabilities; `verified_by` when a
    passing stack's cards ⊇ the path.
13. **Redundancy**: producers per consumed kind on each line. **Single points of failure**:
    articulation points of the engine subgraph, plus any consumed kind with one producer; a
    command-zone node is exempt from availability but flagged when its `protects` in-degree is 0.
14. **Activation**: `deck_audit._component` over the producers of each required event, by turn;
    the goldfish's `engine_online` printed beside it where the deck declares one, with the gap.
15. **Yield**: from the goldfish only where `model_coverage.card_state` is `seen`; otherwise absent
    with the channel named. Forge's `engine_casts` stamps cast / held per node; `deck-notes tally`
    stamps the pilot's over / under reads; Spellbook stamps an infinite.

### 3.4 Evidence tiers on the graph

- ◆ **computed**: nodes, edges, stages, lines, redundancy, cut vertices, hypergeometric activation.
  Derived from the scripts and the list; recomputed byte-identically on every build; a freshness
  gate holds the committed file to a fresh run, as `combos.json` is held today.
- ✓ **verified**: an edge or a line whose cards a checker-passed stack names (keeping the
  documented gap: a stack that names a line's cards does not thereby *support* the line — the
  critic's job stays), a node Forge's AI actually cast or held in a logged game, a Spellbook line
  a stack resolved.
- ★ **judgment**: the `reading` — thesis, named lines, stage overrides with a why, open questions,
  the critic's verdict. Authored by the engineer and the pilot, stamped with the list it was
  written for, marked stale when the list moves, never recomputed and never deleted.

The record and the reading live in one file for one reason: a page that shows the graph without
the pilot's words is a wiring diagram, and a page that shows the words without the graph is what
we have now.

## 4. Similarity, from the same signatures

Each card has two signatures: the multiset of events it **produces** and the multiset it
**consumes**, each with its key parameters, plus one token per ability for the conversion it
performs ("DIES → LIFE_LOSS"). Two relations fall out, and they must stay apart:

- **alike(A, B)**: cosine over IDF-weighted signature vectors (ZoneChange is 44% of all triggers
  in the archive and must count for less than Cycled). Two death payoffs are alike; a death
  payoff and a sacrifice outlet are not.
- **feeds(A → B)**: the overlap between what A produces and what B consumes — directed. Ashnod's
  Altar feeds Blood Artist; Blood Artist does not feed Ashnod's Altar.

EDHREC lift is the second column, dated evidence beside either number, never blended into it.
The disagreements are the finding: high lift and no edge means an interaction the parser cannot
see (or a card that is simply good); an edge and low lift means the field does not play it. The
deck map then lays the deck out by `alike`, with `feeds` edges drawn as arrows and the cities
being the graph's components — a map that *is* the engine rather than one the engineer has to
contradict on every deck. CardBERT keeps the atlas, where theme is the question.

## 5. Two worked examples (from the scripts; the code will reproduce these)

**Sharknado.** Wheel of Fortune is `A:SP$ Discard | Mode$ Hand | Defined$ Player | SubAbility$
DBEachDraw` → `DB$ Draw | Defined$ Player | NumCards$ 7`: it produces Discard(each player) and
Draw(each player, 7). Shabraz consumes `Drawn | ValidCard$ Card.YouCtrl` and produces
CounterAdd(P1P1, self) + LifeGain(you, 1). Lae'zel `R:Event$ AddCounter … AddOneMoreCounters`
scales CounterAdd by +1. Teferi's Ageless Insight `R:Event$ Draw | ValidPlayer$ You |
NotFirstCardInDrawStep$ True` scales Draw ×2. Fireshrieker `S:Mode$ Continuous | Affected$
Creature.EquippedBy | AddKeyword$ Double Strike` scales CombatDamage. Brallin consumes
Discard(you) and produces Damage(each opponent, 1); Torbran `R:Event$ DamageDone | ValidSource$
Card.RedSource+YouCtrl | ValidTarget$ Player.Opponent` scales it +2. Rhystic Study consumes
OPPONENT.SpellCast and produces Draw(you). The expected graph: wheels and standing draw are
ignition and fuel into Shabraz (conversion) whose counters reach the terminal through combat,
scaled by Fireshrieker; the discard half of each wheel reaches the terminal through Brallin,
scaled by Torbran and Gratuitous Violence. There is no pure event cycle — the loop closes through
HAND, because draws refill the wheels. Today's `engine.json` reproduces the verified stacks 006,
007 and 009 as paths (right), places Shabraz as ignition where rule 4 says a wheel is (wrong),
lumps ten wheels and fourteen standing engines into one 24-card "fuel" stage (the graph separates
them by what each consumes), and files Akroma's Will under wincon where `protects` dominates.

**Edgar.** Edgar consumes SpellCast(Vampire, you, from Command) and produces Token(black 1/1
Vampire) — the token script is where its colour comes from, and Teysa's filter cares. Viscera
Seer `A:AB$ Scry | Cost$ Sac<1/Creature>` and Ashnod's Altar `A:AB$ Mana | Cost$ Sac<1/Creature>
| Produced$ C | Amount$ 2` consume the BODIES stock and produce DIES (and, for the Altar, MANA);
Blood Artist consumes DIES(creature, self or other) and produces LifeLoss(target) + LifeGain(you);
Vito consumes LifeGained(you) and produces LifeLoss(opponent, X); Bloodletter `R:Event$ LifeReduced
| ValidPlayer$ Opponent | PlayerTurn$ True` scales LifeLoss ×2 on your turn; Sanctum Seeker
consumes Attacks(Vampire you control) and produces LifeLoss(each opponent, 1) + LifeGain(1). The
expected graph has the stock cycle described in §3.1 and a second route through Sanctum Seeker
into Vito. Today's `engine.json` has one verified line in ten, names two 2-drops as ignition where
the graph says the ignition class is every castable Vampire in hand (thirteen at two mana or
less), and the frame still points at the cut Conqueror loop.

## 6. Limits, and how the model will be validated

**Limits.** Forge's correctness becomes the graph's: a script error is a graph error, which is
one more reason ✓ never derives from the graph. 6.4% of the fleet's cards lean on effects Forge
models loosely (`Effect`, `Animate`, `Charm`); they parse to what the script says and the
coverage block says "partial". 14.8% carry a board-dependent amount, so yield is absent unless
the goldfish reads the channel. 14.2% are `AI:RemoveDeck`, so Forge games cannot stamp them.
Opponents are a stationary producer at authored rates and never enter activation probability.
Combat, politics and sequencing stay dark: the graph says *what can feed what*, not when.

**Validation** (reported as measurements, not pass/fail, in a regression-tier test over the
tracked fleet):
1. Per deck, the Jaccard overlap between graph-derived component members and the hand-declared
   `any_of` groups, and between graph-derived and declared `required` sets.
2. Spearman correlation of Forge cast counts (`engine_casts`) against node betweenness, over
   non-`RemoveDeck` nodes, with N stated.
3. Whether the 122 captain's-log card reads land on graph hubs more than on leaves — counts.
4. Every Spellbook included line should appear as a pure event cycle (null on sharknado and
   edgar, which hold none; yawgmoth-swarm holds 37, emiel-blink 10).
5. A permutation null: shuffle event kinds across cards and show the hub structure collapses, so
   that what we see is not an artefact of vocabulary granularity.

**Open questions the code must settle** (each is a decision, not a research gap): typed stocks
(an untyped HAND connects every draw to every cast — type it by the spell filter of the consumer);
the filter whitelist (about thirty properties; anything else makes an unproven edge); replacement
compounding (reuse the goldfish's `(n + bonus) × multiplier`); strongly connected components with
or without the typed HAND; how the engineer's loop collapses to a naming pass once the critic's
checks are validators.

## 7. Reproducing the numbers

The archive and fleet figures in §1 come from one read-only script over
`cardsfolder.zip.orig` and `data/decks/*/{cards,engine,goldfish_targets,log_annotations,combos}.json`
(kept in this session's record; it will ship as a test fixture with Phase 1). The per-deck engine
table reads `engine.json` directly; staleness is `manamap pilot cache-status <slug> --routine
deck-engine`; the embedding figures are `docs/architecture.md`'s own.
