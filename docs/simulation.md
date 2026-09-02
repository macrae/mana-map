# Simulation — the centre of the workbench

*What this subsystem is for, what was measured before building it, the verdict, and how it
grew. The `game_state` v2 schema it consumes is in `docs/pilot.md`. Last revised
2026-08-22.*

**This is the thing the rest of the bench serves.** A claim about a deck is worth what the
experiment behind it is worth, and this is where experiments run: `simulate` for a deck
against a real pod, `experiment` for two versions of a deck against the same pod, and the
seeded goldfish for the questions that are about a curve rather than a table. Everything
downstream — the audit's targets, the doctor's prescriptions, the deck page's figures —
either feeds an experiment or reads one.


## THE 2026-09-02 AUDIT: one real bug, and a table that was never fair

The pilot said the simulations looked broken. They were, in one specific way,
and three others turned out to be method rather than code.

### The bug: a clock-out was awarded to the last seat

Forge's `-c` clock does not end a game, it **abandons** one. Decompiled: Forge
catches its own timeout, prints `"Stopping slow match as draw"`, calls
`setGameOver(GameEndReason.Draw)` — and then prints `has won because all
opponents have lost` for **every seat still alive**. Both parsers (`forge.py`
and `parse.py` carried separate copies) assigned `winner` on each such line, so
the **last** one won: the highest-numbered survivor.

**Our deck is always `Ai(1)`.** Across 121 truncated games it was credited with
**zero**, while surviving to the clock in 93 of them. `baylen-tokens`, always
the final seat, took **73 of its 85 recorded wins** that way.

It is also the entire "win rate falls as N grows" signature — the clock-hit
share runs 0% at n=20, 9% at n=100, **18% at n=400**.

Fixed: a truncated game is `truncated: true` with **no winner**, excluded from
the win rate. `summary.truncated` and `summary.decided` state the denominator.
All 16 tracked records were re-derived and re-proven from their own logs.

| seat | before | after |
|---|---|---|
| vito | 0.433 | 0.447 |
| giada-angels | 0.358 | 0.405 |
| our seat (pooled) | 0.114 | **0.132** |
| baylen-tokens | 0.094 | **0.015** |

**Why nothing caught it.** `validate-sim`'s invariant was `wins + draws == n`,
and it held *through* the bug — the parser REASSIGNED wins rather than losing
them, so the books balanced while the attribution was wrong. An accounting check
cannot see a misattribution that conserves the total. It is now
`wins + draws + truncated == n`.

### The table was never fair: vito is the only combo deck in it

| pod deck | bracket | contained combos | two-card infinites |
|---|---|---|---|
| **vito** | **4** | **13** | **13** |
| giada-angels | 3 | 0 | 0 |
| baylen-tokens | 3 | 0 | 0 |
| abaddon | 3 | 0 | 0 |

Vito's thirteen lines come from about seven interchangeable pieces — `Exquisite
Blood` or `Bloodthirsty Conqueror`, plus any of `Sanguine Bond` / `Enduring
Tenacity` / `Marauding Blight-Priest` / `Aetherflux Reservoir` — so it assembles
nearly every game, and it wins by LIFE LOSS, which the AI does not block and the
damage parser cannot see.

**The standard pod is now `giada-angels`, `baylen-tokens`, `abaddon`** — three
bracket-3 decks with zero combos between them, within one bracket of each other
and of the fleet. Vito remains fetched and is a legitimate opponent to name
deliberately; it is no longer the default table.

**0.25 was never the null.** Two seats took 85% of decided games. Every run
before this date was measured against a table where a perfectly average deck in
the subject seat could not have scored 0.25.

### Seats now rotate, and the old intervals were not intervals

`GameAction.determineFirstTurnPlayer` picks, from game 2 onward, the
**lowest-indexed seat that did not win the previous game** — and all N games of
a job run inside one `Match` carrying `lastOutcome` forward. Our deck started
**323 of 400** games in one tracked run, and the games are a Markov chain rather
than independent draws.

Every `win_rate_ci95` written before 2026-09-02 therefore assumes an
independence the data does not have. The `-d` order now rotates per job;
`_seat_label` and `record_commanders` were made position-independent so
attribution follows the deck rather than the chair.

### The control: the subject seat is not handicapped

The obvious suspicion, once the pod turned out to be lopsided, is that the fault
is the seat rather than the decks — that whatever sits in the `-d` first position
loses. It does not, and the way to know is a control rather than an argument.

`pod-control` is **abaddon's own EDHREC average deck** run in the subject seat
against giada / vito / baylen — a deck with no relationship to the fleet, so any
handicap in the seat shows up as a handicap on it. 100 games, seeds rotated,
600 s clock:

| seat | wins | rate | 95% CI |
|---|---:|---:|---|
| **vito** | 47 | **0.516** | [0.415, 0.616] |
| giada-angels | 34 | 0.374 | [0.281, 0.476] |
| **pod-control** (subject seat) | 9 | **0.099** | [0.053, 0.177] |
| baylen-tokens | 1 | 0.011 | [0.002, 0.060] |

To reproduce it — the control is a FIXTURE, not one of the pilot's decks, so it
does not live in `data/decks/` and its record is not tracked:

```bash
mkdir -p data/decks/pod-control
cp data/opponents/abaddon/decklist.txt data/decks/pod-control/decklist.txt
manamap pilot simulate pod-control --vs giada-angels --vs vito --vs baylen-tokens --games 100
manamap pilot validate-sim pod-control      # re-proves it from the logs
rm -rf data/decks/pod-control               # a fixture deck on the bench is a lie
```

100 games, 91 decided, 9 truncated. **The neutral deck reads 0.099 in the subject
seat and the fleet pools at 0.132 above it** — so the seat is not the problem and
never was. What the control does show is the other half: vito alone takes more
than half of all decided games and vito + giada take **89%** between them. The
pod was doing the deciding.

### The rotation broke the per-seat analysis block, and the validator caught it

Rotating the `-d` order gave every deck FOUR Forge labels — `Ai(1)-mm-vito`
through `Ai(4)-mm-vito`. `forge.tally_wins` was made position-independent for
exactly this; `parse.aggregate` was not, and it assigned into a dict keyed by
deck name inside a loop over raw seat labels. **Each rotation overwrote the
last.** Every per-seat figure in `analysis` — wins, combat damage, eliminations,
interaction — was computed from only the games where that deck happened to sit at
whichever index was processed last. On the control run the analysis block
reported vito with **6** wins against the summary's **47**.

Two smaller faults travelled with it, both found by tests rather than by reading:

- `win_rate` in `analysis` still divided by ALL games. The truncation fix reached
  `summary` and never reached here, so one record disagreed with itself.
- `commanders.get(s)` read a variable LEAKED from the grouping loop, so every
  seat was published under the LAST seat's commander.

`validate-sim` flagged the disagreement before anyone read the record, which is
the whole reason it re-derives from logs instead of trusting the file. All 18
tracked records were re-derived.

`bridge.build_scenario` had the same shape of bug one layer up: it zipped
`_seat_label`'s keys — a CROSS PRODUCT, N x N — against the record's seat list by
position, giving each seat the wrong decklist and commander, and it called seat
index 0 `"you"`, which under rotation is whichever deck sat first. A board lifted
for `/resolve-stack` could therefore be argued from an opponent's side of the
table. It now reads the game's own `seat_order`.

### The clock is 600 s, and the run id carries it

The distribution, over 1023 games: median 111 s, p75 173 s, p90 227 s, p95 257 s
— and then **12.6% piled up AT the 300 s wall against 6.4% in the 60 s bucket
before it**. A wall truncating twice the mass of the bucket preceding it is
cutting through a second population, not the tail of the first.

Two things followed. `run_id` now carries a non-baseline clock (`-c600`), because
without it a 600 s run writes to the exact path a 300 s run already occupies —
the same silent overwrite `profile_tag` exists to stop, and worse here, since the
clock decides which games are truncated and pooling two clocks mixes populations.
`SIM_CLOCK_ID_BASELINE = 300` is frozen so no record on disk is renamed.

And **the default JVM count is 4, not `cpu_count() - 1` = 7**. This machine has 4
performance and 4 efficiency cores; a JVM on an E-core runs the same game at
roughly half speed, and `-c` is WALL time, so those seats hit the clock and were
recorded as truncated. That is a property of the scheduler wearing the name of a
property of the decks. It matches the censoring exactly: **every 4-JVM run
truncated 0%, every 7-JVM run truncated 5-18%**.

### What the record measures, and the two things it cannot

An audit of the telemetry, against the question "are these games diagnosable".
26 per-seat measures per game, including tokens (six), counters and proliferate,
commander damage per defender, elimination cause, and combat / non-combat damage
split.

**Added 2026-09-02: `interaction_cast` / `interaction_received`.** The targets
were in the log the whole time — `Add To Stack: SEAT cast X targeting [...]`,
captured by the regex and then discarded before reaching a fact. Resolved through
the same learned `owner` map the elimination attribution uses. It answers what
`creatures_lost` could not: that figure counts a creature leaving the
battlefield and cannot separate a removal spell from a chump block from a
sacrifice outlet, which is the entire distinction on Edgar and Yawgmoth.

It is deliberately **not** called removal. The log records that a spell targeted
something and never what the spell did, so a Swords, an edict, a drain and a pump
spell aimed at an opponent's creature are identical to it. Coverage is **59%**: a
cast line carries no permanent id, so ownership comes only from lands, attacks
and blocks, and 43 of 105 targets in one 15-game log were unattributable. It is a
FLOOR on interaction, and `limits` says so.

**Card advantage, tutoring and recursion are ABSENT and cannot be added.** Forge
logs exactly two zone transitions, `Battlefield -> Graveyard` and
`Battlefield -> Exile`. Measured on a 100-game pod run: **zero `from Library`
lines of any kind**. No parser change recovers them. The goldfish is the only
place card advantage is measured — and the goldfish has no blockers, so a Forge
result must never be read as a verdict on a draw engine.

### Still open

- **Our seat's AI profile rests on six games.** `mde_proportion(0.5, 6, 6)`
  returns `None` — the repo's own power function says no difference is
  detectable at that N. The pod's profile was chosen on 100 games.
- **A real draw was a latent landmine.** Forge prints `ended in a Draw! Took N
  ms.` and the pattern matched only `ended in N ms.`; since a game closes only
  on that line, a genuine draw would have merged two games into one. It had
  never fired *because* the clock-outs were being handed to a survivor. Fixed
  with the truncation work, which is what would otherwise have armed it.


## What it is for

The goldfish measures resource development against nobody. The workbench needs what
happens **against a table**: blockers, removal, counterspells, wraths, an opponent who
holds up two; and the figures that only exist under those conditions — how often the
token plan actually converts, what a board of four bodies is worth when one gets
chumped, whether the deck's kill turn survives a seat that answers the first finisher.
That is "token generation pay-off" and "deeper interaction" in one sentence: **a
distribution over many games with real rules and real opponents, plus the ability to
pull one board out and ask the resolver about it.**

## The spike: wrap Forge, or build our own?

"Full and complete interaction" is a rules engine, and Magic's rules are not a weekend.
Before writing one, the question was whether **Forge** — the open-source rules engine
with an AI — would run headless on this machine with Commander decks from this repo and
hand back logs we can measure. Three criteria were set in advance:

| criterion | result | evidence |
|---|---|---|
| **(a) parseable per-game events** — damage by turn, tokens made, blocks, casts, life, winner | **YES** | every line carries a stable prefix (`Turn:`, `Phase:`, `Land:`, `Mana:`, `Add To Stack:`, `Resolve Stack:`, `Combat:`, `Damage:`, `Life:`, `Zone Change:`, `Replacement Effect:`, `Game Outcome:`); a 60-line parser produced winner, turns, casts per seat, combat damage per seat, life trajectory, blocks, and token-creating resolutions (incl. doubling replacement effects, with `Activator:`/`Zone Changer:` for attribution) |
| **(b) 3–4 seat Commander** | **YES** | `sim -d radagast edgar-vampires yawgmoth-swarm heliod -f commander -n 2`: two full 4-seat games, 39 and 37 global turns, different winners; a wrath (Supreme Verdict) cast into a 4-seat board; blocks and a stack response (Tyvar's Stand answering Stroke of Midnight, fizzle) in the 2-seat run |
| **(c) reproducible** | **YES with `-s`** (found in Forge's source after the spike; absent from its wiki). Byte-identical logs on two runs of the same seed, including a two-game sequence under one seed. Without `-s` — the first runs — NO: identical runs diverged | `forge.sh sim … -s 42` twice → `diff` empty; the log prints `seed 42`. The harness seeds every job (`seed_base + i`, recorded), so a run is ◆ **seeded** and game g of job j replays as `-n g -s seed_j`. Runs made before this are recorded SAMPLED and stay valid |

Also measured: **throughput** ~6 s per 2-seat game, ~30 s per 4-seat game on this Mac
(8 CPUs, one JVM) — 500 four-seat games ≈ 4 h serial, ≈ 35 min across 8 JVMs.
**Setup**: Java 21 present; Forge 2.0.14 unpacks to ~470 MB at `~/.mana-map/forge/`
(outside the repo); decks must sit in `~/Library/Application Support/Forge/decks/commander/`
— the documented `-D` override did not take effect, and meta names (`-d radagast`) work.
`.dck` format: `[metadata] Name=…` / `[Commander] 1 Name` / `[Main] 1 Name`, generated
from `decklist.txt` through the repo's own `parse_decklist` so it cannot disagree with
`fetch-deck`. 33,617 cards load; no card in the four decks failed to resolve.

## The verdict on the AI, measured rather than quoted (2026-08-26)

Asked directly whether the simulation is legitimate, the logs were read. **The rules
engine is correct and the pilot is poor, and those are separable** — which is why this
subsystem stays and why its win rate is demoted rather than trusted.

**Rules — sound.** `Whenever an artifact you control enters, Reckless Fireweaver deals 1
damage to each opponent. [Zone Changer: Treasure Token (427)]` is a treasure deck's whole
thesis firing correctly. Treasures are sacrificed for mana, combat triggers resolve, and
Revel in Riches won four games outright without anyone piloting toward it.

**Piloting — poor, and NOT TUNABLE.** 0.67 land drops per own turn, 9.2 casts a game,
first attack on turn 17, keystone cast in 27 of 100 games. Forge's four `res/ai/*.ai`
profiles carry ~200 knobs whose land-related entries are all Strip Mine, Scry, Explore and
Momir edge cases — there is no knob for making a land drop or for sequencing, that is Java.
The aggro profiles were already measured (2026-08-19) to make a hold-up deck worse.

**But the weakness is UNIFORM, and that is the finding.** Every run contains its own
control — the other seats, same games, same engine. Our seat came in at **90–97% of the
pod's rate**, and the AI played the champion and a very different branch alike (lands 5.5
vs 6.0, casts 9.9 vs 9.2). A uniform weakness leaves an A/B between two of your own lists
against one pod substantially intact, both played equally badly; it rescues no absolute win
rate.

**So `sim/pilot_quality.py` measures it on every run** rather than repeating a caveat, and
`info.json` carries the reading so **a win rate never appears on any surface without it**.

---

**Forge's own caveat, verbatim from its `docs/AI.md`:** the AI "is *not* trained", is
"best with aggro and midrange decks, poor to ok in control decks, pretty bad for most combo
decks". One run printed an `AI eval thread at timeout` trace (its think-time cap; the game
continued).

### Verdict: Road A — Forge is the engine. We build the harness, the parser, and the bridge.

Building our own rules engine (Road B) would spend the next month reaching a fraction of
what already runs in six seconds with every card in print. What we own instead is
everything *around* the engine, which is where this repo's value has always been:

- the **harness** (`manamap pilot simulate`) — deck conversion, opponent selection, N games
  across JVMs, the run recorded with Forge version, deck shas, N and wall time;
- the **parser** — logs → an event model → per-game facts → aggregates with confidence
  intervals, tier ◆ with *"sampled, not seeded"* stated in the artifact;
- the **bridge** — a board at turn N lifted out of a log into a **`game_state` v2**
  scenario, handed to `resolve-stack` for the ✓ tier on the interactions the sample
  surfaces.

Road B stays on the shelf for one narrow case: a deterministic, seeded, pattern-tiered
model of a *specific* question Forge's AI answers badly (a combo turn, say). The goldfish
already is that model for resource development, and it stays.

## Evidence tiers under simulation

| | tier | why |
|---|---|---|
| a Forge aggregate (win rate vs. a pod, kill-turn distribution, token damage share) | ◆ **sampled** | deterministic *parser* over non-deterministic *games*; the artifact states N, the CI, and that no game is replayable |
| a single Forge game's narrative | ★ at best | one sample; useful as a story, never as a figure |
| a v2 scenario lifted from a game and resolved | ✓ | the citation contract, unchanged |
| a goldfish figure | ◆ seeded | unchanged |

The AI caveat is **stated in every artifact's assumptions**: a control deck's win rate
under Forge's AI is a lower bound on a competent pilot's, and a combo deck's is not a
measurement at all. The harness runs anyway and writes the caveat into the artifact's
assumptions, keyed off `strategic_frame.archetype` — a number with a stated limit beats
a refusal.

## What the LLM does, and does not do

| does | does not |
|---|---|
| author **opponent decks** for your pod from recon (`data/opponents/<slug>/decklist.txt`, fetched like a deck) | play a seat — 500 games is not an agent's job |
| read a run's aggregate and write the **debrief** of a simulated campaign (same agent, `kind: "sim"`, separate file from the captain's log — the log is what *you* played) | invent a figure the parser did not produce |
| turn a surfaced board into a **v2 scenario** and hand it to the resolve loop | resolve anything without the checker |
| in `prescribe`, cite a sim aggregate as evidence with its CI | cite a single game |

## What the first runs say, and do not

**The pod (S3, 20 seeded games):** radagast 0, Giada 11, Vito 9, Baylen 0. Against three
of its own stablemates it dealt the most damage at the table and lost; against an anthem
deck and a drain deck it deals 12.5 a game and is eliminated by turn 30. Both are Forge's
AI flying a flash-creature control plan; the second says more about the *table* — Giada's
79 a game is the clock everyone else is racing, and Vito's drain is invisible to a damage
parser. Both records carry the caveat.


With S2's analysis on the same eight games: radagast deals the **most** combat damage per
game of the four seats (45.6 mean; edgar 22.0) and wins none — it is eliminated latest on
average (global turn 43.5) by edgar twice, yawgmoth once, heliod once, and its token damage
share is 0.12 against edgar's 0.30. Its cumulative damage curve is 6.6 → 21.6 → 38.6 across
rounds 5–9: the deck *does* develop the kill the goldfish measured; what it lacks at an
AI-piloted four-seat table is the last fifteen points before the table closes on it.
Eight games; every interval is wide; `win_rate_ci95` is [0, 0.324].

`radagast` 0 of 8 against three of its own stablemates. Read with the record's assumptions:
every seat is Forge's AI, which it rates "poor to ok" on control and radagast's frame
calls the deck control; the deck's plan is flash bodies held across opponents' turns and
an AI that taps out on its own turn is not flying it. The figure is a lower bound on a
pilot and an upper bound on nothing. What the run *is* good for already: the pod's
clock (mean round 21.8 with three AIs trading), and which seats win by what (edgar by
damage, heliod by Approach) — the shape of the table, before S2 reads the events.

## The chain, run once for real (2026-08-19)

**simulate → parse → lift → pose → resolve → check → ✓**, end to end, on a board no
one authored: game 1 of the first tracked run, global turn 33, the start of declare
attackers — radagast resolves Craterhoof with eleven creatures and swings everything at
yawgmoth at 16 life; yawgmoth blocks with six and sacrifices one of Craterhoof's two
blockers to Ayara before damage. `sim-scenario … --stack` lifted the board into
`stacks/008-sim-g1-t33-declare-attackers.json` (v2); the author added the two Zombie tokens
the log shows blocking (tokens that had not yet acted are invisible to the bridge — the
note said so), the attack/block/sacrifice actions exactly as logged, the continuous effects
in force (Craterhoof X = 11, Saryth's deathtouch), and ONE question in one rules domain:
combat damage assignment with trample and deathtouch, one blocker removed before damage.

**Three iterations, six spawns, ~570k tokens, verdict pass.** The resolver's damage
assignment matched Forge's actual log line for line — Craterhoof 15 through + 1 lethal to
Grave Titan, Saryth 12 through + 2 to her Zombie, 136 total — a ✓ on the engine's play as
well as on the rules. What the loop found that the author had not: the resolver read the
oracle text and corrected the authored scenario twice (Ayara's ability draws and does not
gain life; Saryth grants deathtouch to OTHER tapped creatures, so she assigns 2, which is
exactly what Forge did); the checker found two missed triggers on boards the scenario
carried — seat-4's Scrawling Crawler on Ayara's draw (which is why Forge's log has yawgmoth
at 15, not 16, when damage hit) and seat-2's Bloodthirsty Conqueror on the 136-point loss
(the 136 life edgar gained in the log) — then held the artifact on three one-sentence slips
until they were fixed. Round 3 passed with 62 citations, which re-confirms the repo's
measured rule that an artifact past ~59 citations takes three or four rounds: the cut was
right, the question was one domain, and the board was simply full.

`scenario-facts` files the four Insect tokens under `other_permanents` because the bridge
cannot know a token's type from the log — a nit the checker read past via the attack list;
fix when the bridge learns token types from the creating card's text.

## The controlled experiment (`experiment`, one artifact)

`manamap pilot experiment <slug> --a <ref> --b <ref> --vs <opp>… --games N [--profile P]`
runs two versions of one deck — a version ref (`V4`, a tag, a sha) or `working` — against
the SAME table, N games per arm, and writes one accumulating artifact under
`data/decks/<slug>/experiments/`: each figure for both arms (win rate with intervals,
elimination turn, damage dealt and taken, first attack, the token figures), the delta,
and — on EVERY figure — a `ci95_diff`, an interval on the DIFFERENCE (Newcombe for
proportions, Welch plus a permutation p for means, a bootstrap on skewed ones) with
`excludes_zero` beside it, plus a `power` block giving the design's minimum detectable
difference so an uninformative result says so instead of reading as no effect. It used to
report whether the two arms' MARGINAL intervals overlapped; that key is deleted rather than
deprecated, because non-overlap implies a difference while overlap implies nothing at all.
Arms run under their own Forge meta names and never touch the deck directory; each arm's
decklist text rides IN the artifact, so the gitignored logs are exactly regenerable.
**Same seeds are not paired games** — a changed list changes every shuffle; the control
is same table, same N, same profile, same engine, and the assumptions say so. An A/A is
refused with the reason (it measures the noise floor; pass different lists knowingly).

First real one (2026-08-19, tracked): radagast **V1 vs V5**, 10/arm vs giada + vito —
win 0 → 0 (overlap: noise), but Δ combat damage **+27.6/game**, Δ eliminated turn
**+5.4**, token damage share **0 → 0.19**: the four swap waves measurably improved the
deck's table presence even where the AI cannot convert it.

**AI profiles** (`--profile`, also on `simulate`): Forge ships Default / Cautious /
Reckless / Experimental. Measured on radagast's seat vs a Default edgar, 6 seeded games
each: Default 3/6, Experimental 2/6, Reckless 2/6 — the aggro profiles make a hold-up
deck worse, so Default stays the default and the AI caveat stands. Also learned: a game
that hits the `-c` clock still declares a winner.

## Every figure carries its median, not just its mean

`mean_ci` reports `{mean, median, min, max, ci95, n}`. The median is there because a
mean over a skewed sample is a true number that describes no game. Measured on the
kianne V1-vs-V2 experiment, arm B's per-game commander damage was
`0 0 0 0 0 0 0 0 0 0 31 178` — **mean 17.42 against V1's 2.25**, which reads as a
sevenfold improvement and was nearly reported as one. The **median is 0 in both arms**:
the entire difference is two games, one of them a 178-damage blowout, and the deck
actually connected in FEWER games after the change (2 of 12 against 4).

The `ci95` of `[-11.64, 46.47]` already spanned zero, so the record was honest and the
repo's interval discipline worked — but it took sorting the per-game values in a
throwaway script to see it. `compact()` had been writing those per-game scalars into
`doc["games"]` all along; the distribution was on disk and merely unsurfaced.

## Commander damage (CR 903.10a), per defender

A player dealt 21 combat damage by the same commander over a game loses — for some decks
it is the *only* win condition, and until 2026-08-21 the parser could not see it.
`combat_damage_dealt_to_players` sums every source and every defender at once, so a
commander that hit three seats for 20 each looked identical to one that hit a single seat
for 60 and killed them. Each seat's analysis now carries a `commander_damage` block:
`dealt_total`, `max_on_one_defender` (**the number the win condition reads**),
`best_single_game_max`, `games_reaching_21` and `games_dealing_any`.

Three decisions worth keeping:

- **Per defender, not per game.** Spreading 60 across three seats wins nothing, and the
  two numbers are reported separately so no reader can confuse them.
- **Combat only.** 903.10a asks for combat damage, so a commander that pings for
  noncombat damage does not count here. A Purphoros deck must not read as closing on
  commander damage it can never deal.
- **The names ride IN the record** (`seats[].commander`), read from the decklists once
  when the run is made. Re-derivation depends on the record and its logs alone: looking
  the commander up from disk at validate time would make a later commander swap read as
  parser drift on a run that was correct when it was made, and would turn every record
  written before the field existed red at once. A record without the field re-derives
  exactly as it always did; `simulate <slug> --analyze <run>` backfills it.

Measured on the first deck that needed it — kianne, whose single win condition is 21
commander damage: over 12 pod games she dealt 12.25 a game, reached 21 on one seat in
**1 of 12**, and in the game she won she finished baylen 24 / giada 34 / vito 22, killing
the whole table through the command zone on round 18. The 1v1 run against the same list
never got there, best 20 in a single game — one short. As a control, Vito (a drain deck)
reads 0.35 commander damage a game across 20 games, which is what a correct measurement
of a deck that does not attack should say.

## Artifacts and where they live

```
data/opponents/<slug>/decklist.txt, cards.json     authored lists for your pod (fetch-deck works)
data/decks/<slug>/sim/<run-id>.json                 TRACKED: the aggregate + meta (forge version,
                                                     deck + opponent shas, N, wall, assumptions)
data/decks/<slug>/sim/logs/<run-id>/*.log           gitignored: the raw games (exactly regenerable when seeded)
data/decks/<slug>/sim/scenarios/*.json               gitignored: lifted boards awaiting a question; `--stack` promotes
~/.mana-map/forge/                                  the engine, outside the repo
```

A run id is `<opponents>-<N>-<short sha of all decklists>`. Re-running the same configuration
after a swap is a new run; the old one stays — it is history, like a prescription.

## Phases

| | | ships |
|---|---|---|
| **S0** | this document + the spike (done) | — |
| **S1** ✅ | `src/manamap/sim/forge.py` — `.dck` conversion, run, N across JVMs, log capture; `manamap pilot simulate <slug> --vs <opp>… --games N [--jobs J] [--clock S] [--list] [--dry-run]` | **done**: the harness, a `forge` pytest marker (opt-in, one real game), and the first tracked run — `data/decks/radagast/sim/edgar-vampires-vs-yawgmoth-swarm-vs-heliod-n8-dfd75e54.json`: 8 four-seat games, 404 s on 4 JVMs (~50 s/game under contention, not the solo 30 s), radagast 0 · edgar 4 · yawgmoth 2 · heliod 2, mean round 21.8 (global turn 43.1). **Two things the first run corrected**: Forge's `Game Outcome: Turn N` is the winner's own turn count (a ROUND), not the global turn — the record carries both; and an alternate win condition prints `has won due to effect of '…'`, not `has won because` — two Approach of the Second Sun wins read as draws until matched. The record carries `won_by` now |
| **S2** ✅ | `src/manamap/sim/parse.py` — events → per-game facts → aggregates with CIs (Wilson for rates, normal for means); the run record gains `analysis` + compact per-game rows; `simulate --analyze <run>` re-derives from kept logs; `validate-sim` re-proves the tracked analysis against the logs where they exist and form-checks where they do not; `deck-info` gets a `simulated` panel | **done.** Token figures are two, each with its limit named in `analysis.limits`: `token_resolutions` (creation abilities that resolved — blind to X and doubling) and `tokens_observed` (distinct ids that attacked/blocked/dealt combat damage — a token that sat is invisible), plus `token_damage_share`, `tokens_chumped`, and our seat's **cumulative combat damage by round** — the shape of the kill. Seat attribution is learned from assignment/land lines, never assumed; `eliminated_by` is the controller of the last damage source before the life line that crossed zero, null when never seen acting. **One bug the fixture caught**: the seat pattern `\S+` swallowed the comma after an `Activator:` tag and mis-attributed every token resolution to the active seat |
| **S3** ✅ | `data/opponents/<slug>/` + `manamap pilot fetch-opponent "<commander>" [--as slug]` (`sim/opponents.py`, EDHREC's average deck through its JSON endpoint, `source.json` for provenance); the harness resolves an opponent before a deck of the same name | **done.** The pod as dictated: **giada-angels** (Giada, Font of Hope — angels + anthems), **abaddon** (read from dictation as "Abigail"; best guess, flagged in its `source.json`), **baylen-tokens** (Baylen, the Haymaker), **vito** (Vito, Thorn of the Dusk Rose). All four load in Forge. First tracked table: `giada-angels-vs-baylen-tokens-vs-vito-n20-…-s1451665738` — 20 seeded games, 487 s on 4 JVMs: **radagast 0 · Giada 11 · Vito 9 · Baylen 0**, `win_rate_ci95` [0, 0.161], mean round 17.6. Giada deals 79 combat damage per game and eliminates radagast 9 of 20; radagast's cumulative curve flatlines at 12.5 by round 9 (45.6 against its own stablemates). **A limit the run exposed**: Vito wins 9 on 7.0 combat damage per game — his kills are life LOSS, which the damage parser cannot see (only `Life:` lines do); `eliminated_by` still attributes through the last damage source and is wrong for a drain kill. Named in `analysis.limits`, fixed in S5 |
| **S4** ✅ (see below) | `sim/bridge.py` + `manamap pilot sim-scenario <slug> <run> --game G --turn T [--step S] [--stack]`; `pilot/game_state.py` (the v2 vocabulary + form check); `validate-stack` and `scenario-facts` take v2; `--seed` in the harness (found after the spike) | **done.** A board lifted at a CR step: life exact, lands exact with tapped-since-last-untap, cast permanents from their resolve lines (creature `X - Creature P / T`, permanent bare `X`, spell `X (id) - …` is not one, a countered cast never enters), removal by id, Morph → the card it was on `has unmorphed`, tokens from first use with `tokens_unobserved_resolutions` sizing the gap, a commander's logged exit read as `command` (Forge prints the exit before the CZ replacement; later casts confirm), hand as `{unknown: n, estimate: true}`; every approximation in `extras.reconstruction_notes`; `question` empty on purpose and the preflight says so until the pilot poses one and a stack/action. Measured on the real game: the AI unmorphed in its upkeep, not its main — the test had the cut wrong, the bridge did not |
| **S5** ✅ | `eliminated_how` (damage vs life loss) and drain attribution in the parser; `sim:runs` cache input on `deck-diagnosis` and `prescription:<id>`; the doctor and skeptic charters read run records with interval, N and the AI caveat | **done, minimal by choice.** A separate prose "sim debrief" was not built: the run record's `analysis` already IS the debrief of a simulated campaign, `deck-info` shows it, and reading it into advice is the doctor's job under `/prescribe` — a fourth agent writing prose about an aggregate would be the magazine coming back. Measured after the fix: radagast eliminated by Vito 5 times (was 2), 6 of 16 attributed eliminations by life loss |

S1 and S2 are one session each. The first real question to put through the whole chain is
the one the log will have raised by then.
