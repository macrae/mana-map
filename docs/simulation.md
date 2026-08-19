# Simulation — design (S0) and the Forge spike, 2026-08-19

*Branch `simulation`. What this subsystem is for, what was measured before building it,
the verdict, and the phases. The `game_state` v2 schema it consumes is in `docs/pilot.md`.*

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

## What the first run says, and does not

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
| **S3** | opponents — `data/opponents/`, `deck-recon` → authored pod lists, `simulate --vs pod` | your table, not your other decks |
| **S4** ✅ | `sim/bridge.py` + `manamap pilot sim-scenario <slug> <run> --game G --turn T [--step S] [--stack]`; `pilot/game_state.py` (the v2 vocabulary + form check); `validate-stack` and `scenario-facts` take v2; `--seed` in the harness (found after the spike) | **done.** A board lifted at a CR step: life exact, lands exact with tapped-since-last-untap, cast permanents from their resolve lines (creature `X - Creature P / T`, permanent bare `X`, spell `X (id) - …` is not one, a countered cast never enters), removal by id, Morph → the card it was on `has unmorphed`, tokens from first use with `tokens_unobserved_resolutions` sizing the gap, a commander's logged exit read as `command` (Forge prints the exit before the CZ replacement; later casts confirm), hand as `{unknown: n, estimate: true}`; every approximation in `extras.reconstruction_notes`; `question` empty on purpose and the preflight says so until the pilot poses one and a stack/action. Measured on the real game: the AI unmorphed in its upkeep, not its main — the test had the cut wrong, the bridge did not |
| **S5** | the sim debrief + `prescribe` reading sim aggregates | the doctor sees the table |

S1 and S2 are one session each. The first real question to put through the whole chain is
the one the log will have raised by then.
