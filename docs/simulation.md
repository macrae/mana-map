# Simulation — both engines

*Two engines, two jobs. The **goldfish** (`src/manamap/pilot/goldfish*.py`) is the
decision instrument: seeded, paired, about ten seconds a question. **Forge** is the probe:
real rules and a real AI, asked narrow questions. The `game_state` v2 schema is in
`docs/pilot.md`. Rewritten at the top 2026-10-05; everything from "The 2026-09-02 audit"
down is the Forge record, still how a probe works.*

## The goldfish is the decision instrument (2026-10-04)

The pilot ruled overnight pod runs out of the decision loop: a day per answer, runs killed
by a sleeping laptop and the two-hour background cap, an MDE of ~0.14 at 200 games, and a
Forge AI that mis-pilots some decks so their rates are floors. A swap is now answered by
`manamap pilot try <slug> --out A --in B` in about ten seconds, and a branch is graded by
`net-change`. What makes the goldfish fit for that job:

**It is PAIRED.** Every game takes its own seed, `random.Random(f"{seed}:{i}")`
(`diagnostic.HARNESS` version 2), and `diagnostic.align` lines two lists up slot for slot,
so game *i* of the champion and game *i* of the candidate deal the same shuffle wherever
the lists agree. `net_change._paired` takes the per-game differences (`keep_games=True`)
and reports a paired t interval, MDE = 2.8016 × se, a call only when the ROUNDED bounds
exclude zero. Pairing roughly halved interval widths; a list against itself reads exactly
zero and makes no call.

**It reads what it claims to.** `model-coverage <slug>` names per card the channel it feeds
and whether that channel is on — seen, DARK (its channel is off) or invisible. `try` prints
it for every card in and out. A card the model cannot read looks exactly like a card that
does not help, so read that line before believing a null.

**It obeys attack restrictions.** `goldfish_profiles.attack_gate` reads Defender (never
attacks), "can't attack unless …" (held back, with the reason) and the two hand conditions
it can evaluate (Kefnet the Mindful, seven or more cards; Hazoret, one or fewer). Before
this, Kefnet was a free 5/5 flyer every turn.

**It seats three opponents for what WE gain off them, and one for damage.**
`GOLDFISH_OPPONENTS = 3` multiplies triggers on opponents' draws — Faerie Mastermind's card,
Smothering Tithe's Treasure — because a four-player table has three opponents
drawing. Damage is still measured against one seat at 40 life: the clock is "how fast does
this kill one player", and nobody blocks or removes anything.

**Its authored rates are stated, in `config.py`, and `try` names them** when a card on
either side depends on one:

| constant | value | what it stands in for |
|---|---:|---|
| `GOLDFISH_OPPONENTS` | 3 | opponents whose draws trigger our payoffs |
| `TITHE_PAY_RATE` | 0.5 | share of opponents who pay {2} to Smothering Tithe rather than give a Treasure |
| `GEYSER_TAPPED_SHARE` | 0.5 | share of an opponent's lands tapped when Mana Geyser resolves |
| `OPPONENT_HAND` | 4 | cards in a target opponent's hand (Jeska's Will) |

**Rituals are cast when they are the difference** (2026-10-05). `ritual_profile` reads a
fixed ritual (Dark Ritual), per-creature (Battle Hymn), per-type (Brightstone Ritual),
per-opponent-hand (Jeska's Will, with its impulse mode) and per-tapped-land (Mana Geyser);
it refuses a ritual with an additional cost or restricted mana rather than reading it as
free. A ritual is cast in the main phase only when the mana it adds reaches a spell, or the
commander, that the turn could not otherwise cast; its mana is gone at end of turn.

**What it still cannot see** is in the channel table's named gaps below, and the largest is
structural: no blockers, no removal, no interaction. Its verdict on board QUALITY is not
evidence — a go-wide refactor it preferred lost 31/400 to 50/400 in Forge because 1/1
tokens do not connect. That is what a Forge probe is for.

## Forge is the probe

`forge-cast-check <slug> --card "<name>"` asks the one question Forge answers better than
anything: **does the AI play this card?** A two-seat shell, about a minute. Every add that
must be cast or activated for a branch's objective is proven this way before it enters an
arm (`--branch B --adds --write`). A pod run (`simulate`, `experiment`) is optional
evidence; in `net-change` a Forge loss whose interval excludes zero is a `forge_warning`
beside the verdict, never a block. Read `engine_casts` and HELD-WHILE-CASTABLE before any
Forge rate, and no Forge rate is graded without an A/A at the same N.

> **2026-09-03 — A COMMANDER-DAMAGE KILL IS A GO-WIDE KILL WEARING A CROWN, and
> the goldfish cannot judge either.** Zur's V6 engine is *"Zur attacks, fetches
> an aura, connects with lifelink, Vito drains"* — every step gated on a 1/4
> commander connecting. The goldfish reported that route assembled by turn six
> in **23.6%** of games. Forty Forge games against the pilot's own table:
> **0.35 commander damage a game, best single game 2, and 0 of 39 reaching 21.**
> Total combat damage 2.33 a game against edgar's 32.3 on the same table; first
> attack on turn 21.05 against edgar's 16.69. The route was never wrong about
> the DRAW and was never evidence about the KILL.
>
> Same class as the Edgar go-wide refactor the model preferred and Forge scored
> 31/400 against 50/400, one turn further along: there it was 1/1 tokens that do
> not connect, here it is one creature that has to connect twenty-one times.
> `validate-goldfish-targets` notes a combat-dependent route now — measured
> across the fleet first, it fires on 2 of 10 decks and both are genuine — and
> says sharply more when `model_combat` is off, which is what Zur had.

> **2026-09-03 — THE NULL IS NOT 0.25, AND THE POD THAT REPLACED THE UNFAIR ONE
> IS ALSO UNEVEN.** `pods <name> --calibration` pools every tracked run that
> faced a table and reports how it actually divides its wins. `vito-era` was
> dropped on 2026-09-02 for being unfair — vito and giada took 85% between them.
> Measured the same way, **`standard` gives giada-angels 0.572 (2.3x its fair
> share) and baylen-tokens 0.052 (a fifth of one)**, and the subject chair
> **0.159**. The dominant seat moved; the table did not level out. `baylen-tokens`
> is the floor in BOTH pods, which is a fact about that deck under Forge's AI
> rather than about either table.
>
> So a deck scoring 0.16 on `standard` is AT the typical subject rate, not two
> thirds below a quarter — and `simulate --list` now prints the null beside the
> rate. Two nulls exist and they are not the same: this one pools OUR decks and
> describes the fleet as much as the table, while `pod-control` (an opponent's
> own average deck in the subject chair, 0.099) is the neutral one and has been
> run once. Neither is 1/n.

> **2026-09-03 — WHO WENT FIRST IS NOT THE `-d` ORDER, and the difference
> reports backwards.** PRD §14 asks for win rate by turn order position, and the
> obvious source is `outcomes[].seat_order` — which is the order the decks were
> passed to Forge, not the order they play. `determineFirstTurnPlayer` gives the
> first turn, from game 2 of a job onward, to the lowest-indexed seat that DID
> NOT WIN the previous game, so the deck that loses most starts most: over the
> 400-game run **edgar-vampires started 81% of games**. A figure built from `-d`
> position would have credited the losing seat with a turn-order advantage.
> `Turn: Turn 1 (seat)` names who actually started, and `started_rate` travels
> beside the split so the confound cannot be read past. `placement` lands in the
> same pass — 1 is the survivor, the rest ordered by how late they died, and a
> clock-out is excluded rather than filed as a first place. All 19 records
> re-derived; no other figure moved.

> **2026-09-03 — the standard pod is an artifact, not a sentence.** It was
> described here and nowhere in the code — `grep STANDARD_POD` found an AI
> profile and no roster — so the most load-bearing configuration in this layer
> lived in shell history. `data/pods/*.json` is tracked and carries each seat's
> archetype and bracket, which is what lets B-2's "report by pod composition" be
> more than a slug list. `--pod standard` expands to the same ordered slugs and
> therefore **the same run id**, asserted through `run_id_for`, the function
> `run` itself calls. `vito-era` is kept so every pre-2026-09-02 record stays
> reproducible. A seat may carry its own AI profile — the Forge seam always
> existed (`-a` is index-aligned, `command()` took a list) and nothing could
> reach it; a table whose seats disagree gets `-podMixed<8hex>` so two pods that
> play differently cannot share a path.

> **2026-09-03 — FORGE GIVES THE FIRST MULLIGAN FREE.** A rules-coverage gap,
> found by wiring up a measurement that had been taken and thrown away: Forge
> emits `has mulliganed down to N cards.` once per mulligan and `has kept a hand
> of N cards` for the final size, and the parser matched only the second while
> `compact()` dropped even that. Measured across all 130 tracked logs and
> **5,056 seat-hands with zero exceptions**: 0 mulligans keeps 7, **one mulligan
> also keeps 7**, two keeps 6, three keeps 5 — so `kept = 7 - max(0, taken - 1)`.
> Under the London mulligan a single mulligan draws seven and bottoms one, for a
> hand of SIX. **A deck that mulligans is flattered by one card here.** Both
> figures are now reported per seat because neither is derivable from the other
> under real rules, `analysis.limits` carries the gap, and all 19 tracked records
> were re-derived from their logs in the same commit — **no other figure moved.**

> **2026-09-03 — four defects fixed in `experiment`, and one of them re-bases the
> table.** Its final print read `ci95_a`/`ci95_b`, keys `delta()` has never
> emitted, so **every real run raised `KeyError` after writing its artifact** and
> exited 1; the measurement survived and only the exit code said anything. It
> lived because the tests exercised `--dry-run` and `--analyze` and the branch
> that runs had no seam — it is `_print_result` now. Alongside it: `_run_arm` had
> **no `timeout=`**, so the 3.7-/4.2-hour runaway class was unguarded on the
> flagship while being capped on `simulate`; and the pod ran on **Default** while
> `simulate` has run it on `STANDARD_POD_PROFILE = "Experimental"` since
> 2026-08-30, so the two commands measured different populations. The pod default
> now matches, `--vs-profile Default` reproduces the old one, and
> `experiment_id` carries `profile_tag` so the two cannot share a path. **Both
> tracked experiments predate this and were measured against the Default pod**;
> their ids are unchanged and still mean what they said.

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

### 2026-09-10: the round robin, and `standard-v3`

That table lasted eight days. Over 549 decided games giada-angels took **0.537
(2.15x fair)** and baylen-tokens **0.080**, so the subject null was 0.180 and no
deck of ours could read well at it. `standard-v2` (nekusar / muldrotha / abaddon)
was tried on 09-06 and was worse: nekusar 0.637. Both tables were chosen on
POOLED SHARES from tables the seats happened to sit at, which the v2 note already
said does not transfer.

The measurement that had never been made was a **round robin among candidate
opponents with none of our decks seated** — eight seats, ten tables of eight,
every seat on Experimental, 74 decided games (`data/opponents/*/sim/`). It
became expressible when `seat_home` let an opponent be the subject (09-09):

| seat | share | x fair | note |
|---|---|---|---|
| giada-angels | 0.528 [0.370, 0.680] | 2.11x | genuinely too strong, not just fed by baylen |
| nekusar-discard | 0.436 | 1.74x | |
| muldrotha-value | 0.351 | 1.41x | floor 4, Hermit Druid + Jace — OUT by the zero-combo rule |
| sythis-enchantress | 0.235 | 0.94x | |
| jarad-graveyard | 0.189 | 0.76x | |
| abaddon | 0.175 | 0.70x | |
| baylen-tokens | 0.029 | 0.11x | a floor at every table |
| talrand-spells | 0.026 | 0.11x | |

**`standard-v3` is sythis-enchantress / jarad-graveyard / abaddon**, the three
nearest fair that are bracket 3 with zero combos. Then the part v2 skipped: five
of our decks sat at it, 185 decided games, default profiles.

| subject | W/N | rate | ci95 | clocked out | sythis | abaddon | jarad |
|---|---|---|---|---|---|---|---|
| edgar-vampires | 19/46 | 0.413 | [0.283, 0.557] | 14 | 0.348 | 0.174 | 0.065 |
| ur-dragon | 12/35 | 0.343 | [0.208, 0.508] | 5 | 0.314 | 0.200 | 0.114 |
| gishath | 12/36 | 0.333 | [0.202, 0.497] | 4 | 0.361 | 0.167 | 0.111 |
| heliod | 10/36 | 0.278 | [0.158, 0.440] | 4 | 0.250 | 0.278 | 0.167 |
| goblin-storm | 1/32 | 0.031 | [0.006, 0.157] | 8 | 0.562 | 0.250 | 0.156 |
| zur-enchantress † | 4/47 | 0.085 | [0.034, 0.199] | 12 | 0.660 | 0.191 | 0.064 |

† killed at 59/60 by hand, no record; `sim-progress` reads the logs.

Pooled (`pods standard-v3 --calibration`): **sythis 0.362 [0.296, 0.434] 1.45x,
subject 0.292, abaddon 0.211, jarad 0.119 (floor)**. Not level, but the most even
table measured, and the default from this date.

**The ranking did not predict the calibration, and that is the finding.** Sythis
was 0.94x on the level field and is 1.45x here — and the per-subject column is
the reason: 0.25 against heliod, 0.31–0.36 against the three that attack with
real bodies, 0.56 against goblin-storm, 0.66 against zur. Sphere of Safety
behind forty enchantments stops a deck that has to CONNECT and barely slows one
that swings every turn. **A pod's null is a property of the table with the
subject in it.** Report every rate against this table beside that spread, and
re-read the calibration after each new subject. The clock-out column matters
too: edgar's 14 were mostly games it was ahead in at the 600s limit.


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
lines of any kind**. No parser change recovers them — but a one-method patch to
Forge's log FORMATTER does, and the spike below measured it (2026-09-30). Until it
is productionised the goldfish is the only
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

## What the goldfish models, channel by channel (2026-09-26)

Every channel below is **OFF unless its flag is set** in the deck's
`goldfish_targets.json`, and the default figures on a deck that declares nothing are
mana, bodies and land drops. `model-coverage <slug>` names, per card, which channel it
feeds and whether that channel is on — **a card the model cannot read looks exactly like
a card that does not help**, and that confusion has cost this project a whole branch.

| flag | what it switches on |
|---|---|
| `model_draw` | a card's own ETB draw, a spell's draw, a recurring draw, an arrival draw, the draw you buy, draw doublers. `meta.card_advantage.draw_not_modelled` names what is still unread, and is **absent** rather than empty when nothing is |
| `model_combat` | attacks, board power, the damage clock, tokens as attackers |
| `model_treasures` | Treasure creation and spending, including Treasure off opponents' draws (Smothering Tithe at `TITHE_PAY_RATE`) |
| `model_sacrifice` / `model_deaths` / `model_drain` | outlets, deaths and the damage they convert into |
| `model_discard` | wheels, loots, and the discard half of a draw you pay for |
| `model_colors` | on by default; a colourless mana model is simply wrong |
| `model_commander_animate`, `model_commander_attack_tutor`, `model_commander_combat_reveal`, `model_commander_copy` | commander abilities only one corpus card has, declared per deck |

**The copy channel is the one that needs no authored rate**, and that is why it can be
trusted more than its neighbours. `model_commander_attack_tutor` reported 5.70 fires a
game against Forge's 1.22 and cost half a day's measured gains when corrected. The copy
count IS the number of other creatures you control, which the model already measures —
there is no number to write down, so none can be wrong. The flag is checked against the
commander's own text, so a deck cannot declare it on a commander without the ability.

### What a copied spell does

A single-target spell copied across the board is read per effect, each ONE TURN unless
noted. `copy_fodder` decides what qualifies: the spell must target one creature and
nothing else, and its effect must be one you want on every body you control.

| effect | read as | notes |
|---|---|---|
| its draw | multiplied by the board | the archetype's whole engine |
| a pump | power only | this model has no blockers, so toughness changes no damage |
| a **+1/+1 counter** | PERMANENT | measured as a *wash* against a bigger temporary pump on a board of 1/1s |
| a Treasure | one per copy | the mana for the follow-up |
| double strike | doubles the board's combat damage | |
| doubled power | scales with the board | LOSES to `+3/+3` below average power 3 |
| damage each creature deals to each opponent | one seat's worth | Chandra's Ignition |
| an additional combat phase | **not** multiplied by the copy count | there is no way to know which creatures untapped; see below |
| an untap | untaps the whole board | held for BETWEEN combats, never cast on curve |

### Storm, magecraft and per-cast damage

The spell count is tracked — before 2026-09-26 there was no spells-cast-this-turn
variable at all, so **storm was unreadable** and Grapeshot scored as a 1-damage ping
however many spells preceded it. Every cast routes through one helper, guarded by a
structural test that fails if any of the twelve cast sites bypasses it.

**Magecraft fires on a cast OR A COPY; per-cast damage fires on a cast only.** That is a
rules fact and the two flags are separate because of it: with a copy commander out, one
cantrip is eight magecraft triggers and ONE Guttersnipe trigger. Collapsing them would
hand every such deck a burn kill it does not have.

### Creatures tap; lands do not

Creatures tap when they attack (CR 508.1f) and untap at the start of your turn, and each
combat phase re-selects its attackers. Before this the swing was computed once and
multiplied by the phase count, so **an additional combat was free damage from a board
that had already attacked** — ur-dragon's damage at turn ten fell 68.798 → 59.499 when it
was corrected, and it was the only deck in the fleet claiming an extra combat.

An extra combat is therefore worth nothing on its own. Something must untap the team,
which is why an untapper is HELD for between combats rather than cast in the main phase —
cast early it untaps creatures that are already untapped, the same mistake a pilot makes
with Seize the Day.

**LANDS still enter untapped, always.** That assumption is unchanged, is about lands, and
is why `mana-analysis` and `mana-fit` remain the whole of the evidence for a land swap.

### Named gaps

- **Vigilance is not modelled.** No deck in the fleet grants it; a deck that did would
  make every extra-combat card live again.
- **X-based pumps and X-based counters are not read** — X is a count this reader has no
  board to resolve, the same refusal `land_colors` makes for a fetchland without a pool.
- **A ritual with an additional cost or restricted mana is refused**, not read as free
  mana (Infernal Plunge, Geosurge); the unconditional shapes are read (see the top).
- **Toughness is tracked but nothing reads it yet.** It exists for a line that damages
  your own board and cashes the deaths; that channel is not built.

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
**In Forge, same seeds are not paired games** — a changed list changes every shuffle (the
goldfish pairs; Forge cannot); the control is same table, same N, same profile, same
engine, and the assumptions say so. `--aa` runs one list against itself on purpose: it
measures the noise floor, and no Forge rate is graded without one at the same N.

**Looks (2026-09-29): a group-sequential A/B, so an overnight queue can stop early
honestly.** `experiment … --looks K` (K ≤ 4, O'Brien–Fleming by default, `--boundary
pocock` the alternative) splits N into K equal waves; each look is WHOLE ROTATED JOBS FROM
BOTH ARMS, never a partial job, because a job's early games differ systematically from its
late ones (Forge gives turn 1 to the previous loser) and a look cut inside one would be the
biased slice. Each look tests the primary — wins over DECIDED games — with the Newcombe
interval at that look's own boundary z (`stats.OBF_BOUNDARIES`, K=4: 4.049, 2.863, 2.337,
2.024; pinned like `T975`), so an early stop needs a huge effect and the final look pays
about 3% of half-width over a fixed design. `--until-mde X` adds NON-BINDING futility: stop
when the boundary interval already excludes +X on the favourable side, which is the "large
effect absent" reading formalised and touches nothing of the design's alpha. The record is
rewritten after every wave with `design {looks, boundary, critical, schedule, until_mde}`,
`status` (running / stopped_efficacy / stopped_futility / complete) and `looks[]` (each
with its jobs, seeds, seat orders, counts, boundary z and decision); it is the crash-resume
unit — the same command line with `--resume` re-reads the completed waves' logs and
continues the global job index, so the seeds and rotations are the ones the design
planned. A record with no `design` cannot be resumed: a look added after the fact is
optional stopping. A stopped run's `reading` names the look and its boundary, and calls the
1.96 interval beside it descriptive. `stats.sequential_power` (seeded Monte Carlo) prices a
design; at K=1 it equals the exact grid. `experiment.validate` form-checks a record against
its design and a test sweeps the tracked ones.

**The overnight queue (`campaign`, 2026-09-29).** A powered A/B is ten hours an arm on
this machine, so the loop cannot be a person typing `experiment` at eleven at night; it is
a queue, and a queue written down BEFORE the games is the only kind that counts as
pre-registration. `data/campaigns/<name>.json` is TRACKED (like `data/pods/`) and lists
entries — slug, two arm refs, pod, N, looks, the one registered endpoint (`win_rate`), a
hypothesis, optional `detect` / `until_mde` / `profile_b` / `aa`. `campaign <name> plan`
pins each ref to a sha, computes the experiment id the entry will write (so the entry IS
the run id), preflights every entry against the pod's null and refuses an underpowered
`detect` unless the entry says `anyway`, prepends an A/A for every harness fingerprint
(engine build, override sha, pod, clock) that has none — the standing noise-floor check —
and writes `resolved` back, the one time the command writes the file. `run` goes in order:
DONE and STALE are skipped, RUNNING is resumed, PENDING is run; a harness that changed
since planning is refused (another harness is another measurement); each terminal record
is written to the deck's decision ledger; NOTHING MERGES, and an AST test holds the module
to that. State is DERIVED, never stored — DONE when the record is terminal, RUNNING when it
says so, STALE when a pinned `working` or `@branch` list has moved underneath the plan
(the games would measure a list nobody holds), PENDING otherwise. A gitignored
`<name>.state.json` carries the entry in flight for `sim-progress`. The first campaign,
`2026-10-standard-v3`, ships unplanned: `plan` is the pilot's act, because it pins the
engine's fingerprint at the moment it runs.

**Four instrument fixes shipped with the looks.** The experiment did not rotate seats
(`simulate` had since the seat-bias finding; the A/B kept our seat at index 0 in every job
of every arm) — `_run_wave` rotates per GLOBAL job index with the profiles rotated
alongside, and the label map scores our seat at every index. The id omitted the clock, the
override sha and the AI-profile sha, each empty at its default so every record keeps its
name (`-k{K}`, `-aa` and `-bme{P}` are new). The record had no `pod` block, so
`net_change.forge` could never bucket an experiment with the run records at the same table.
And one list twice was refused outright: `--aa` runs it as the harness's noise floor on a
second seed base for arm B, and `--profile-b P` flies our seat on another AI profile on arm
B only — policy-on against policy-off on one list, the A/B every piloting change needs and
nothing could express. The win-rate interval also divided by every game PLAYED while the
rate beside it divided by decided games; one denominator now.

**The preflight, on both commands (2026-09-29).** `experiment` had printed its power
arithmetic since 2026-09-10; `simulate` — the command that runs most — never did. Both now
print it before a JVM starts: the baseline (the pod's subject null from `pods
<name> --calibration`, with its game count as the other side of the test; for a table with
no null, the deck's last run there), the MDE at this N, and a four-row table of what
+0.05 / +0.10 / +0.15 / +0.20 would need. The hours are per ARM — one for `simulate`, whose
comparison arm is the null and costs nothing to play again. **`--detect X` turns the
preflight into a refusal**: a run whose power for X is under 0.8 exits with the games that
would reach it, and `--anyway` runs it on the record as a screen. Without `--detect`
nothing is refused; a smoke test made no claim. `sim/power.py` is the one home for the
null (`null_rate`), the preflight and the refusal; `net-change` prints, beside an
UNDERPOWERED Forge block, the games per arm that would resolve the observed delta.
`stats.mde_proportion` is exact below `EXACT_MDE_MAX_N` games (1,000; `diagnostic` passes
400) and the normal approximation above, labelled — the grid is O(n²) and overflows a
double near 4,000, which is why `games_for_difference` used to answer ">1000" where it now
answers "1,199".

First real one (2026-08-19, tracked): radagast **V1 vs V5**, 10/arm vs giada + vito —
win 0 → 0 (overlap: noise), but Δ combat damage **+27.6/game**, Δ eliminated turn
**+5.4**, token damage share **0 → 0.19**: the four swap waves measurably improved the
deck's table presence even where the AI cannot convert it.

**AI profiles** (`--profile`, also on `simulate`): Forge ships Default / Cautious /
Reckless / Experimental. Measured on radagast's seat vs a Default edgar, 6 seeded games
each: Default 3/6, Experimental 2/6, Reckless 2/6 — the aggro profiles make a hold-up
deck worse, so Default stays the default **for our seat**; the pod's three seats have
been Experimental since 2026-08-30 (`forge.STANDARD_POD_PROFILE`, and `--vs-profile`
defaults to it), and the AI caveat stands. Also learned: **Forge's raw log** declares a
winner even for a game that hits the `-c` clock. **The harness does not.** A clock-out is
recorded `truncated: true` with NO winner and is excluded from the rate — it used to be
awarded to the last seat, which our deck can never be. Every record carries
`summary.truncated` / `summary.decided`; see the clock section above.

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

## The board series — an estimate from the log the parser said could not carry one (2026-09-30)

`parse.py` records, correctly, that Forge logs a permanent LEAVING the battlefield and
never one arriving — 0 `to Battlefield` lines in a 100-game run — and concluded that a
board count "would be a series of zeros wearing the name of a measurement". But
`bridge.reconstruct` has been building exactly that board for `sim-scenario` since S4: a
cast permanent enters on its `Resolve Stack` line, a token at the resolution that creates
it, a land on its `Land:` line, and each leaves on its zone change. `sim/board_series.py`
builds what the bridge builds at ONE cut at EVERY cut — the start of each turn's cleanup
step, the PRD's "end of each turn" — one `reconstruct` per global turn (~0.4 ms a cut, a
second and a half per 100-game run), and the record carries it top-level as `board_series`
beside `engine_casts`: validated only where present, so no older record reddens, and
re-derived from the logs by `validate-sim` where they exist. Our seat only, per OWN turn:
`bodies`, `printed_power` with `power_unknown` beside it (a `*/*` is a body, never 0
power), `permanents`, `tokens`, `lands`, `open_lands` (untapped at cleanup — mana left
unused), `rocks_tapped` (a floor on mana sources), and `commander_uptime` (own turns after
the first with the commander on the battlefield, share still there — heliod's open
question, "losing the commander takes the engine with it", has a figure). `limits` names
every floor: a body that entered without being cast is seen only when it first acts, an
X-count token is not on the board, printed power ignores counters and anthems, removal by
name takes the first holder, the hand is never logged. The catalog moves `bodies by turn`
to both engines, `commander uptime` and `creature power distribution` to PUBLISHED (Forge,
as estimates that say so), and `post-wipe recovery` and `threat-to-lethal gap` to
DERIVABLE — the data is in `per_game` and the subtraction is not written. The old policy
test ("everything about a board arrival is off Forge") became "a board figure may claim
Forge only as an estimate that names `board_series` and calls itself one".

## The board finder, and the gate a lifted board never had (2026-09-30)

A board is worth a procedure only if it RECURS; one game is an anecdote, and the two
boards proven so far (goblin-storm 011 "the modal board", 012 "the widest") were chosen by
reading logs. `sim-boards <slug> <run> --criterion C` reads every game of a run through the
board series and names, per criterion, a CUT (game, global turn, CR step) and a SHAPE —
what must be the same for two hits to count as one kind of moment: `widest` (our main with
the most bodies; shape = bodies bucket + commander on the battlefield), `modal` (every own
turn 4–10 at precombat main; shape = the exact bodies/lands/commander tuple, so the most
common one IS the modal board), `death` (our last main before elimination; how, by whom,
turn bucket), `held` (a cleanup of our turn with N+ lands untapped and nothing cast or
activated — a held hand; open-lands and hand-estimate buckets), `pre-wipe` (the start of a
wipe turn), `lethal-missed` (an opponent at or under our untapped PRINTED power and no
attack — a floor on lethal, never a proof, and its row says so) and `first-attack`. The
shortlist is ranked by games reaching the shape with a Wilson interval on the share, and
`--lift` / `--stack` hand the exemplar to `sim-scenario` with `extras.finder` (criterion,
shape, recurrence, rank) — the provenance a handbook proposal will cite instead of "I picked
this game". Measured on the first run: `held` at four lands fires in 24 of 100 goblin-storm
games, a rate worth a procedure and not on every game.

**The gate.** Every lift now stamps `source.lift_sha` (over the canonical board: per seat,
life, commander zone, sorted name/tapped/pt/token) and `source.bridge_sha` (over
`bridge.py`, the `model_version` idea). `validate-lift` re-lifts every committed scenario
from its own cut where the logs are and diffs the canonical boards — FAIL while the
artifact has no checker verdict yet (the board it will be argued from is wrong: re-lift and
supersede), a NOTE once the loop has finished on it, pass or fail (finished work is not
condemned by a later bridge fix, the `unknown_cards` rule), and where the logs are absent, a NOTE when the bridge has moved
since. `validate-stack` carries the same result as a warning on every lifted scenario. That
is the gate CLAUDE.md said was missing; goblin-storm 010 is the case it would have caught.

## One primary, twelve exploratory (`net-change`, 2026-09-29)

`experiment` pre-registers `win_rate` and calls its other ten figures descriptive, because
eleven intervals at α = 0.05 is roughly one that excludes zero by chance every two
experiments. `net-change` had the identical exposure and no such rule: twelve goldfish
rows, each with an independent `better / worse / noise` verdict by MDE threshold, no
interval on any row's difference, and no correction across the family. Now the objective
— declared when the branch was opened, before anything was measured — is the one primary,
graded on its own; the rows are a FAMILY, each carrying the interval on its own difference
(Welch from the cells' `{rate, sd, n}` on a mean, Newcombe on a rate — `stats.
diff_means_summary` is `diff_means` from summary statistics, held equal by a test), and a
row's verdict needs both the MDE and Holm's step-down across the family (`stats.holm`,
`stats.z_two_sided`, no scipy). `noise` keeps its meaning: unresolved, never "no change".
Measured over the 42 tracked reports before shipping: **Holm flips zero of 438 verdicts**
and no recommendation moves — the MDE rule (2.8016·se) was already within 2% of the
Bonferroni-12 threshold (2.865·se) at the top rank. So this changes what a verdict MEANS,
which `design` now states in every report, not which rows carry one today.

**The real table is in the rule (2026-09-29).** `recommend()` read twelve goldfish rows and
the objective, and appended Forge as a `note` — on copy-burst-v1 the goldfish said MERGE
while 73 real games read −0.061. Two changes. First, a branch may aim at the table:
`forge.win_rate >= 0.25 @standard-v3` is an objective (`candidates.FORGE_OBJECTIVE_AXES`:
win rate, commander resolved rate, combat damage, and the two CONDITIONAL means, first
attack turn and eliminated turn, whose `n` is the games in which the event happened). A
Forge axis is a property of the list AT A TABLE, so it names its pod or is refused, and it
is graded on the branch's pooled reading at that pod — never at whichever table held the
most branch games — with the interval on the champion-to-branch difference and the table's
null carried in the grade. Second, above every other row of the stated rule: **a Forge
win-rate loss whose interval excludes zero at the same pod and harness is "do not merge"
whatever the goldfish said**; one that spans zero is "cannot tell" and leaves the goldfish
verdict alone. The Forge block now STORES the null beside the figure it scales (it was
printed and never written) and carries `endpoints`, every axis with both arms, the
interval on the difference (Newcombe / Welch, a bootstrap on the median for combat
damage), the MDE and the run ids. `deck-branch new` on a Forge objective prints the
champion's reading at that pod, the null and the MDE, where the line is chosen. Swept: the
eight tracked reports with a Forge block regenerated; no recommendation moved.

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

## The telemetry patch: one method, and what the spike measured (2026-09-30)

The two zone transitions the log carries are not the engine's limit. `javap` on the
shipped jar (2.0.14) put the filter in `forge.game.GameLogFormatter.visit(GameEventCardChangeZone)`,
which returns `null` unless the move is Battlefield → Graveyard or Exile; the engine
fires the event for every move. The spike patched THAT ONE METHOD — drop the filter,
keep the Ante exclusion, same `ZONE_CHANGE` type and caption — compiled it with
`javac --release 21` against the fat jar (no Maven), and put the class into a COPY
named `forge-mm-telemetry.jar`. The pristine jar is untouched and the patch lives in the
session scratchpad until productionised.

**What a patched line looks like.** `Zone Change: Sol Ring (123) was put into Hand from
Library.` — the card NAME and id, in sim mode. `parse.RX["zone"]` already matches it.
Over B's 20 games: 2,157 Library→Hand, 911 Hand→Stack, 906 Stack→Battlefield, 633
Hand→Battlefield, 381 Library→Exile, 252 Library→Graveyard, 139 Hand→Library, 122
Command→Stack, 115 Hand→Graveyard. Draws, tutors, mills, wheels, discards, ramp
landing from the library, and the command zone, every one of them named.

**Measurement 1 — does the patch change a game?** The first design was wrong and
is recorded because the number is useful: one seeded 20-game table (goblin-storm at
standard-v3, seed 990990, 4 JVMs), shipped jar twice (A1, A2) and patched once (B).
**A1 and A2 differed in 18 of 20 games.** Same seed, same jar. The mechanism is the
one this document already names — `AI eval thread at timeout` fired 130, 105 and 140
times in the three runs — so at this table two replays of one seed share the SHUFFLE
and nothing else, and a game-level comparison of patched against pristine cannot
speak. (A1's wall clock was 22,303 s against A2's 1,802 s: A1 ran under the test
suite. Load is not a nuisance variable here; it is the variable.) The aggregate is
consistent with noise and says nothing either way: B 3/13 decided against the pooled
shipped-jar 2/28, difference −0.16 [−0.44, +0.06].

The decisive control is the one this document's reproducibility claim actually
covers: a SHORT game on a QUIET machine. One two-seat game (goblin-storm vs
giada-angels, seed 4242, clock 600), pristine jar twice and patched once, run one at a
time with nothing else on the machine, **zero AI timeouts in any of the three**.
Comparing every line the pristine formatter would have written: pristine-vs-pristine
differs on exactly one line, `Game Result: Game 1 ended in N ms`, and pristine-vs-
patched differs on exactly the same one line. Same winner, same turn, same 349 old
lines; the patched log carries 87 new zone lines beside them. The patch is purely
observational where the control can see, and the control saw everything.

**Measurement 2 — do the old readers survive the new lines?** B's 20 logs parsed in
full, then parsed again with every new zone line stripped (anything `was put into X
from Y` where Y is not Battlefield): the two `analysis` blocks are identical at every
key, and our seat's facts are identical key by key. Nothing that exists today counts
a line it should not.

**Verdict: GO.** Names are visible (not counts), the patch does not move a game, and
the existing parser is blind to the new lines rather than confused by them. What GO
buys, in the catalog's terms: cards drawn per turn, turns with an empty hand, missed
land drops BY NAME, draw-engine uptime, tutors resolved, `engine_casts.held` MEASURED
for every card rather than inferred from expected draws, and a hand the bridge no
longer has to estimate — which is what `line-finder` (Phase D2) needs to reason from a
known hand.

**The owner, added before it shipped.** The zone line names the card and not its
owner, and the parser's owner map is learned from cast and land lines — blind to
exactly the cards that are drawn and never cast. So the patch appends ` owner
Ai(1)-mm-goblin-storm` to every zone line (a suffix; the shipped caption is intact and
`parse.RX["zone"]` takes it as an optional group). The same single-game control was
re-run on that build: the same one differing line, and 87 of 87 zone lines carry an owner.

**Productionised the same day** (`sim/telemetry.py`, `manamap pilot forge-telemetry`).
`data/forge_patches/` tracks the `.java` and a manifest — the Forge version, the pristine
class sha, the source sha, and every registered patched class sha (javac output differs
between JDKs, so a build elsewhere registers its own sha rather than failing against this
one's). `telemetry.installed()` reads the class out of the patched jar beside the
pristine one and answers with the three states card scripts answer with: no jar or
Forge's own class → a plain run; a registered sha → the fingerprint; anything else →
`simulate` and `experiment` REFUSE before a JVM starts. A run under the patched jar
carries `-tl<sha8>` in its id and a `telemetry {sha, class, source_sha, jar, lines}` block
in its record, where `lines` counts the zone lines the shipped formatter would not have
written and `validate_sim` refuses a block whose count is zero. It is NOT a bucket axis
in `net_change.forge` and NOT excluded from a pod's null: the control showed the games
are the same games, and the tag exists so two runs of one configuration under two jars
are two paths and a reader knows whether a record's draw facts are measured or absent.
**The hand, read exactly** (`parse.hand_facts`, same day). A record played under the
patch carries `analysis.seats[*].hand`: `library_to_hand` from turn 1 on (the deal and
the mulligans are before turn 1 and excluded — the control game reads 6 for the seat on
the play and 7 for the draw, one per own turn), `end_of_turn_size` per own turn,
`empty_own_turns`, `own_turns_without_land_drop`, `missed_land_drops_with_land_in_hand`
(a FLOOR: a card in hand is a land only if some seat played it as one in the run), and
`hand_at_end` with the turn each card arrived. With our deck's mana values
(`forge.cmc_map`) every card also gets `castable_uncast` — own turns it ended in hand
with at least that many lands on the battlefield, lands only, a floor — merged into
`engine_casts.by_card` beside `cast`, and `engine_casts.held_while_castable` lists the
cards held on two or more such turns and cast at most once: `engine_casts`'s inferred
"held and never cast" made exact, and printed as HELD WHILE CASTABLE (MEASURED): goblin-storm kept Goblin Bombardment and Great Train
Heist in its opening hand and still held both at turn fourteen. A plain record has no
`hand` key and its `limits` sentence about card advantage is unchanged, so every record
on disk re-analyses to the same bytes; a patched record's `limits` says what the hand
facts are and what the floor is. Still not answered: draw-engine UPTIME, which needs a
predicate for "repeatable draw source" over the arrivals now logged. The spike's scripts and logs are in the session scratchpad (`telemetry/noise_floor.py`,
`telemetry/one_game.py`, `noise_floor.json`).

## Artifacts and where they live

```
data/opponents/<slug>/decklist.txt, cards.json     authored lists for your pod (fetch-deck works)
data/decks/<slug>/sim/<run-id>.json                 TRACKED: the aggregate + meta (forge version,
                                                     deck + opponent shas, N, wall, assumptions)
data/decks/<slug>/sim/logs/<run-id>/*.log           gitignored: the raw games (exactly regenerable when seeded)
data/decks/<slug>/sim/scenarios/*.json               gitignored: lifted boards awaiting a question; `--stack` promotes
~/.mana-map/forge/                                  the engine, outside the repo
```

A run id is
`<opponents>-n<N>-<short sha of all decklists>-s<seed>[-me<profile>]-pod<profile>-c<clock>`,
built by `forge.run_id_for` — a real one on disk:
`giada-angels-vs-baylen-tokens-vs-abaddon-n120-996adb84-s573917060-podExperimental-c600.json`
(`me<profile>` is omitted when our seat is Default, which it normally is).
**The seed, both AI profiles and the clock are in the id on purpose**: each one changes
what the games ARE, so two runs that differ in any of them must not collide. Re-running the
same configuration after a swap is a new run; the old one stays — it is history, like a
prescription.

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


## The patch set grows an `ai` class: `SacOutlet` for a sacrifice-cost mill ability (2026-09-30)

The formatter patch was observational. The second patch is not, and the pipeline says so:
`data/forge_patches/` now registers a SET of classes, each with a `kind` — `log` for
`GameLogFormatter.java`, `ai` for `MillAi.java` — the jar is built from all of them, the
run id's `-tl<sha8>` is over the set, the record's `telemetry.classes` lists them with
their kinds, and `net_change.forge` buckets on the set sha, because a set with an `ai`
class changes how the AI plays.

**Why an engine patch rather than a hint.** Altar of Dementia in Edgar is a free outlet:
its job is to turn a doomed body into a Blood Artist trigger, and the mill is incidental.
Forge's `MillAi` has no aristocrat path (only `PumpAi` and `CountersPutAi` do), and its
own targeting gate computes `X` — the sacrificed creature's power — BEFORE any creature is
chosen, reads 0, and refuses every target. Measured on the hinted 20-game replay: **32 of
our own permanents went to the graveyard while the Altar sat on the battlefield across six
games, and it was activated 0 times.** A card hint cannot reach that; a preference alone
measured nothing.

**What `AILogic$ SacOutlet` does.** Fires when a creature of ours other than the source
is marked `SacMe` or is one the engine's own `ComputerUtil.shouldSacrificeThreatenedCard`
says will die this turn and is not dangerous to sacrifice in combat; aims the mill at the
legal opponent with the thinnest library; any phase, because it is reactive. The cost is
paid through the engine's normal `SacCost` path, which prefers the THREATENED creature
before the profile's default fodder — so the Altar hint carries no `AIPreference` on
purpose. Ten lines of Java, compiled beside the formatter.

**Measured** (same seed, twenty games at standard-v3): see the record tagged
`-tlbeb0c66d` and the sacrifice table in `docs/gotchas-bench.md`.


## Two more `ai` classes: `Deflect` and `Protection` (2026-09-30)

The pilot: "Teferi's Protection is a legit amazing card that can save boards and games —
the fact both of these are 0 is a HUGE red flag." The measurement agreed: over Edgar's four
telemetry passes (80 games) Deflecting Swat was held while castable on 59 own turns and
cast 0 times, with **17 opposing spells or abilities aimed at our seat or our permanents**
while it sat in hand; Teferi's Protection 29 turns, cast 0. Neither is a hint problem.

- **Deflecting Swat.** `ChangeTargetsAi` "can't otherwise play this ability" (its only
  logic is the Spellskite magnet), and worse, `PlayerControllerAi.chooseNewTargetsFor`
  returned `null` — "AI currently can't do this" — so a redirect the AI did cast would have
  changed nothing. Two patches: `AILogic$ Deflect` fires when the top of the stack is an
  opponent's targeted spell or ability aimed at us or at something we control and there is
  somewhere else legal to send it; the new-target chooser sends it AWAY from us — the
  caster's best permanent first, then the caster, then another opponent's best, never back
  onto anything of ours — carrying a divided allocation across when the spell has one.
- **Teferi's Protection.** `EffectAi` casts an Effect spell only under a logic it
  implements, and none fit a protection spell (the earlier `Fog` hint covered lethal combat
  only). `AILogic$ Protection`: cast against an opponent's board wipe on the stack
  (`DestroyAll`, `DamageAll`, `ChangeZoneAll`, `SacrificeAll`) when we have two or more
  creatures to lose; against removal on the stack aimed at our commander; against a spell
  that would take our life to zero; or at the opponent's declare-blockers when combat is
  lethal.

`data/forge_patches/` registers five classes now — one `log`, four `ai` — one jar, one
patch-set sha in every run id (`-tl43a8b062`), and `net_change.forge` buckets on it. The
jar carries every class javac emits for a patched source (siblings ride with their
primary), and a two-game smoke under it ran clean before anything was measured.

## The cast preflight: `forge-cast-check` (2026-10-01)

The drain-v1 branch arm (200 games) was the measurement that made this a tool: Toxic
Deluge, unflagged on 2026-09-30 and staged as the deck's one sweeper, was **drawn 28
times and cast 0**, held at game end, never discarded. The scan row carried the AI flag
and "unflagged" was read as "castable". It is not: `AI:RemoveDeck:All` is one filter, and
each API's AI class can refuse a card for reasons of its own. Deluge's script has no
`IsCurse$`, so `PumpAllAi` treats a -X/-X sweep as a pump of OUR creatures, and its X is
`Count$xPaid` — priced at 0 before the life is paid, so even the curse branch would read
"-0/-0 kills nothing". The same two defects Vish Kal had, on a different API, found after
a night of games instead of before.

`forge-cast-check <slug> --card NAME` builds a two-seat shell — the deck's own commander
over `--copies` of the card, the deck's cheapest spells as filler (a sweeper needs
creatures to see, an outlet needs fodder) and basics of the card's colours — and plays
`--games` short games against one named seat under the jar and the pilot profile
`simulate` would use. It counts from the telemetry hand facts: drawn, cast, activated,
discarded, castable-and-uncast own turns, held at game end, and prints one verdict:
PLAYED (with the counts), HELD (castable in N games, cast 0 — a hint or a patch before any
branch), or NOT DRAWN / inconclusive. It measures nothing about the card's value. The rule
it enforces: every add a branch objective depends on being cast or activated runs this
first, and the result goes in the stage `--why`.

**The gate (2026-10-01, the same day, after Bastion of Remembrance read CAST-LATE in the
same arm — cast in 10 of 30 games it was drawn, castable on 48 turns).** The pilot: "no
more finding out after hours of running that the cards aren't firing … measure twice cut
once." Four pieces:
- `forge-cast-check <slug> --branch B --adds --write` runs one shell per card the branch
  ADDS (`deck_branch.diff`, copies that rose included), `--jobs` at a time, and writes
  `branches/B/cast_proofs.json`: per card the counts (`triggered` joined them — "cast" is
  not enough for a card whose job is its trigger), one of five verdicts (PLAYED /
  CAST-LATE / HELD / UNPLAYED / NOT DRAWN), the CLASS read off the card's installed script
  (`cast_check.diagnose`: `removedeck-all`, `x-priced-before-cost`, `no-iscurse`,
  `no-ai-logic`, `permanent-cast-priority`, or `unknown` — never a guess; a Java patch the
  installed jar carries is credited, so Vish Kal reads clean under PumpAi) and its remedy.
  The file is stamped with the harness — overrides sha, pilot profile and its content sha,
  patch set, jar — and a row PLAYED under the same tuple is kept rather than re-run. The
  validator (`validate-cast-proofs`) holds it to form; `deck_status.VALIDATED` and the
  fleet sweep gate it.
- `simulate <slug>@B` REFUSES before a JVM starts when any add is HELD, CAST-LATE or
  unproven under the harness the run would stamp — a proof under another tuple is
  unproven, like a goldfish figure under another `model_version` — naming the card, the
  class and the command; `--anyway` runs it and the record's `cast_proofs` block names the
  slots as FLOORS (`validate-sim` refuses a record that ran unproven without it).
- `net-change` prints CAST PROOFS in its preflight and on THE REAL TABLE, reads the
  branch's own arms post hoc too (`_cast_proofs_from_runs`: a record's gate block, else
  an add in hand and never cast in `engine_casts`), marks the objective's RESULT `FLOOR
  (n add(s) held)`, and freezes the reading in `harness.cast_proofs` so `propose` copies
  it into the decision. `deck-info` NEXT names the check ahead of `simulate`; `deck-branch
  stage` warns, never refuses — the gate is at the expensive step.
- The Deluge remedy itself (an `IsCurse$` hint plus a PumpAllAi patch pricing X) and the
  cast-priority hint Bastion / Anointed Procession / Sanguine Bond need are harness
  changes and wait for drain-v1 to decide; each is proven with the same shell first.

## A sixth `ai` class and a third hint kind: Vish Kal's -X/-X (2026-09-30, evening)

`kills_by_ability` read 0.00–0.05 a game on every Edgar record while the sacrifice half
of Vish Kal activated 29–30 times a pass: the AristocratCounters logic fed him counters
and nothing ever spent them. Two causes, read from the 2.0.14 source and both needed:

- **X is priced before it exists.** The ability is `NumAtt$ -X | NumDef$ -X` with
  `SVar:X:SVar$CostCountersRemoved`, and the engine sets `CostCountersRemoved` only when
  the cost is PAID (`CostRemoveCounter.payAsDecided`). `PumpAi.checkApiLogic` evaluates X
  first, reads 0, and refuses any X pump whose amount is zero. `data/forge_patches/
  PumpAi.java` prices X as the counters a `SubCounter<All/…>` (or numeric) cost paid from
  the source would remove right now — the number `AiCostDecision` pays for `All` — and
  leaves the rest of the logic to decide: `getCurseCreatures` keeps only creatures the
  -X/-X would kill, `useRemovalNow` decides now or later.
- **The script never says it is a curse.** `SpellAbility.isCurse()` is `hasParam("IsCurse")`
  and the card author left it off (72 of the 95 -X/-X Pump lines in the corpus carry it),
  so even with X priced the AI would look for one of OUR creatures to "pump". The third
  hint kind, `ability_params` (`forge_pilot.HINT_KEYS`), appends `| IsCurse$ True` to the
  named ability line; one hint per (card, ability), so Vish Kal carries two.

Measured on an eight-game two-seat shell (Vish Kal commanding 62 cheap W/B creatures
against giada-angels, seed 4343, profile mm-edgar-vampires, zero AI timeouts): under the
patched jar the removal fired **3 times, each one killing an Angel** (Resplendent Angel
-7/-7, Emeria Shepherd -5/-5, Archangel of Tithes -6/-6); under the pristine jar with the
same hint, **0**. The patch set is `c22ecaf3328d` and the card overrides `8c347642bcf0`
(Toxic Deluge unflagged in the same pass); every Edgar record before this is a different
bucket, which is the point of the harness pin in the drain-v1 plan.


## The drain axis and the threat axis, measured (2026-09-30)

The pilot's brief for Edgar is two sentences: life gain / life drain as the ENGINE, and big
scary vampires with every way to hurt people. Until today the record measured the cost side
of that plan and none of the identity: a Blood Artist trigger is life LOSS, which no damage
total sees, and a 12/12 lifelinker connecting reads the same in `combat_damage_dealt_to_players`
as twelve 1/1 tokens. Five figures now ride in every seat block, from lines the shipped log
already carries, each with its definition and its floor in `analysis.limits`:

| figure | what it counts | attribution | floor / ceiling |
|---|---|---|---|
| `drain_dealt` | an opponent's life loss that was not damage | the seat whose life-loss ability resolved last this turn — the elimination attribution's own rule | FLOOR: an unread resolve line or a later ability credits nobody |
| `life_gained_by_source` | life gained, split lifelink / trigger / other | the line before the gain: our combat damage → lifelink, our resolve → trigger | attribution by adjacency, the total exact |
| `combat_damage_by_keyword`, `evasive_damage_share` | our combat damage to players by the source's printed evasion keyword | `cards.json` keywords, so OUR seat only; absent elsewhere | printed keywords only — a granted flying reads as ground |
| `biggest_hit` | the largest single-source combat damage event to a player, and its card | exact | one event, one source |
| `kills_by_ability` | opposing permanents that left the battlefield directly after one of our activated abilities resolved | the last resolved activation this turn; any event in between breaks it | CEILING per card: a permanent dying anyway is counted |
| `extra_draw_per_turn` (the draw axis, same day) | Library → Hand moves beyond one natural draw per own turn (none for turn 1 on the play), over own turns | exact, from the telemetry patch's owner-bearing zone lines | TELEMETRY RECORDS ONLY — a plain record has no `hand` and the axis is absent, never zero |
| `empty_hand_turns` | own turns that ended with zero cards in hand | exact, same source | telemetry records only; LOWER is better |

The last two are the pilot's oldest Edgar complaint — "running out of steam post turn 7/8
... trying to rebuild with 1 or 2 cards (or zero) in hand" — read from `hand_facts` per
game: the champion at the pinned harness holds 1–2 cards at the end of its own turns from
turn 5 on. Hand size at a fixed turn is deliberately NOT an axis: only 3–6 of 20 games
reach our eighth own turn, so it would be read over the games the deck already survived.
Every one is a Forge objective axis a branch can be aimed at — `forge.drain_dealt`,
`forge.life_gained`, `forge.biggest_hit`, `forge.evasive_damage_share`,
`forge.kills_by_ability`, `forge.extra_draw_per_turn`, `forge.empty_hand_turns` — with the
interval on the difference and an MDE in `net_change`'s endpoint table, so drain-v1's objective can finally be stated on the axis the pilot means
instead of win rate standing in. Every tracked record with logs on disk was re-derived to
carry them (the catalog marks them PUBLISHED, and the catalog test holds every record to it);
a record whose logs are gone keeps its old block and says nothing about them.
