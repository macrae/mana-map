# Known issues — the inventory

**The board of what is red and why.** `make test` does not pass on `main`, and it
has not for some days. That is a deliberate state for most of what is on this
page and an unowned one for the rest — the distinction is the point of the file.

A red test here is not a bug report to be triaged later. Each row says what it
is measuring, why it fails, and **who or what unblocks it**. A row leaves this
page when the test is green, never because the test was changed to suit the
artifact.

Last verified against `make test`: **2026-09-08**, 9 failing / 3217 passing /
208 skipped. Two of the eleven failures on that run were fixed in the same
commit that created this file (`sim-progress` was undocumented and the pilot
subcommand count read 101 in four surfaces); they are not listed below.

---

## 1. ur-dragon: five artifacts describe a deck that is not on disk — BLOCKED, deliberately

| test | measuring |
|---|---|
| `test_pilot_tracked_artifacts_validate[ur-dragon/engine.json]` | every card the engine model names is in the 99 |
| `test_pilot_artifact_freshness::test_versions_json_matches_a_fresh_run[ur-dragon]` | the tracked version history matches one derived from git |
| `test_pilot_manual_freshness::test_tracked_manual_matches_a_fresh_render[ur-dragon]` | the manual is a pure function of the artifacts |
| `test_pilot_poh::test_a_rebuild_is_byte_identical` | the handbook re-renders to the same bytes |
| `test_pilot_poh::test_a_procedure_page_names_only_cards_the_deck_runs` | the procedures name only cards in the 99 |

**Cause.** `validate-engine ur-dragon` reports it exactly:

```
stages[0]: names 'Shivan Reef', which is not in the 99
stages[0]: names 'Stormcarved Coast', which is not in the 99
2 card(s) in the 99 appear in no stage and in no unassigned entry:
  Sunbaked Canyon, Turbulent Springs
```

Two of those four are the 2026-09-03 patch (Plateau and Volcanic Island, sleeved
as proxies, swapped for Turbulent Springs and Sunbaked Canyon). **The other two
are not.** Shivan Reef and Stormcarved Coast are named by the engine model and
are in no version of the list — which is why the pilot's own read is that *the
deck definition is in question*, and why the paper lock was withdrawn (`22e3743`).

**What unblocks it.** The pilot checks the paper list in — `check-in ur-dragon
--from <file>` — and only then does the artifact chain get rebuilt against it.
A `deck-engineer` pass is already sitting unmerged in `.agent-out/` and is
**deliberately not applied**: regenerating the engine model against a list
nobody has confirmed would put a fresh byline under an unverified deck, which is
this repo's oldest rule.

Fixing these five before the check-in means writing artifacts that describe a
guess. Leave them red.

---

## 2. heliod: the engine model has never been criticised — needs one agent run

| test | measuring |
|---|---|
| `test_pilot_deck_info::test_a_real_deck_composes_every_panel` | a sleeved deck's dossier has an engine panel with a critic verdict |

**Cause, as of 2026-09-08:** the critic has now RUN, and returned **`fail`** —
16 findings, 5 of them `supported`. The panel stays unverified because the
verdict is `fail`, not because it is missing, which is the test working
correctly. Round 2 (engineer answers the findings, critic re-judges) is the
thing that turns it green; the loop allows three rounds and this was one.

Nothing here is cache-recorded. A `fail` model is kept because it documents what
could not be grounded, and recording it would say the opposite.

**The finding worth reading even if you never touch the engine model.** The model
implied Heliod's flip carries the deck. Cross-tabbing
`per_seat.heliod.commander_transformed_turn` against `winner` over the 120-game
standard-pod run says the reverse, and it re-derives exactly:

| | games | wins | rate | Approach wins |
|---|---:|---:|---:|---:|
| Heliod NEVER flipped | 85 | 23 | 0.271 | 18 |
| Heliod flipped | 35 | 4 | 0.114 | 2 |

> **RETRACTED 2026-09-09. This did not replicate and was never a finding.** A
> fresh 120-game run on v1.2.1 reads 0.165 against 0.171 — flat, difference
> +0.007 with a 95% interval of [−0.123, +0.175]. The engineer and the doctor
> found the non-replication independently, and so did a direct re-derivation.
>
> It was "verified" at the time by re-deriving the cross-tab from the same run's
> logs, which proves the ARITHMETIC and cannot prove the finding — a number
> re-derived from the run that produced it agrees with itself by construction.
> No interval was ever put on the split, in a commit that quoted the rule that
> every rate carries one. Four wins in thirty-five games has an interval about
> twenty points wide.
>
> Kept on the page rather than deleted, because a retraction that removes the
> claim removes the lesson with it. See `docs/gotchas-bench.md`, "re-deriving a
> number is not replicating a finding".

The critic's instruction is to rebut rather than weaken, and this is what that
looks like: the correction makes the model's own thesis stronger, not softer.
Whether flipping is *causally* bad or merely marks the longer games it survives
into is open — the split is a fact, the mechanism is not.

Four other figures a reader would take as measured are wrong: the "four gift
pieces" enumeration is six inside the stage and more outside it; a proposed
goldfish edit argues for dropping two Medallions that were already removed at
`c879dcd`, the very commit that shipped this `engine.json`, so its cited 0.823
is a pre-edit rate against a live 0.732; a roleless-card count reads five where
six is right; and "whose every threat flies" is 30 of 32.

**edgar-vampires and ur-dragon are in the same state** and are not asserted by
any test, which is its own gap: three of nine decks have an uncriticised engine
model and one test notices.

**What unblocks it.** An `engine-critic` run on heliod (the `/analyze-engine`
loop, after `validate-engine` passes), then the same for edgar-vampires.
ur-dragon waits on §1.

*A note on the test.* It names heliod. Four tests were repointed this month for
naming a deck and inheriting its decisions, and this one is a candidate — but
not yet: the deck it names is genuinely unverified, so the test is currently
right for the right reason. Repoint it only once heliod is green, and give it
`deck_status`'s sleeved set rather than another single slug.

---

## 3. Three goldfish fidelity findings — real, unowned, and each worth a session

These are model bugs, not stale artifacts. Every one of them was found by a test
written for a *different* defect, which is the only reason they are known at all.
None is fixed by regeneration.

### 3a. The runaway guard fires on a real deck

```
test_pilot_goldfish_drain_and_draw::test_the_runaway_guard_holds
  the cap fired on a real deck — either a loop exists or the policy is
  eating more than a board can hold
  assert 0.003 == 0.0
```

Three games in a thousand on edgar-vampires hit the iteration cap. The test
asserts **zero**, correctly: a cap that ever fires means either the sacrifice
policy has a cycle, or it is consuming more permanents than a board can hold.
0.3% is small enough to have been ignored and large enough that the figures for
those games are whatever the cap left behind rather than a played-out result.

**Do not raise the cap.** Find which games hit it and what the board looked like.

### 3b. The arrival channel depends on a flag it should not

```
test_..._the_arrival_channel_does_not_secretly_require_the_combat_model
  draw-only reads 1.325 against 2.671 with combat
  assert 1.325 > (2.671 * 0.7)
```

Bodies arriving on the battlefield is not a combat question, so the arrival
count must not halve when `model_combat` is off. It does — it reads **half**.
Something in the arrival path is gated behind the combat flag, which means every
draw-only deck's board figures are understated by a factor near two, and two
decks measured under different flags are not comparable on board at all.

This is the `model_combat` sibling of the class `model_coverage.never_cast` was
built for: an effect the model computes and then does not apply.

### 3c. Colour screw does not bite a five-colour deck

```
test_pilot_goldfish_colors::test_a_mono_colour_deck_is_barely_affected_and_a_five_colour_one_is
  five-colour deck did not move: [0.22, 0.323]
  assert 0.323 < 0.22
```

`model_colors` is supposed to make a five-colour manabase *worse* than a
colour-blind one. On ur-dragon it makes it **better** — 0.323 against 0.220.
A mono-colour deck is correctly indifferent, so the flag is doing something; it
is doing the wrong thing at five colours, which is the only place it matters.

The land-colour reader is the first suspect: `land_colors` has been wrong twice
before in a way that flattered a greedy manabase (fetchlands reading as
colourless, gated any-colour lands counted at full value), and both were found
by a sweep rather than by reasoning about the code.

---

## 4. The parser drops 88% of all noncombat damage — CONFIRMED, unfixed

Found 2026-09-08 by following up a finding from heliod's engine critic, which
claimed 1,002 noncombat damage from three cards against a record reporting 202.
The first read was that the critic had counted pings at creatures alongside
pings at players. **That was wrong.** All of it is damage to players.

Measured over the 120-game standard-pod run
(`giada-angels-vs-baylen-tokens-vs-abaddon-n120-996adb84`):

| | |
|---|---:|
| noncombat damage to players **in the logs** | 4,420 |
| **attributed to a seat by the record** | 521 |
| dropped — source permanent has no owner | **3,901 (88%)** |

**Cause.** `parse.py` learns its `owner` map from exactly three line kinds:
`land` (line 244), `attack` (263) and `block` (269). A permanent that is neither
a land nor ever attacks or blocks **never enters the map**. At line 510
`src_seat = owner.get(src_id)` is then `None`, and line 524's guard —
`elif ev["noncombat"] and src_seat in per:` — drops the damage in silence.

So every artifact and enchantment that deals damage is invisible to the seat
tallies. The largest unattributed sources in that one run:

    Descent into Avernus   880     Impact Tremors        333
    Warleader's Call       697     Viseling              265
    Iron Maiden            381     Delayed Blast Fireball 250

**What it costs.** heliod's punisher axis — Viseling, Iron Maiden, Ebony Owl
Netsuke, 1,003 damage between them — reads as 202. That is the deck's entire
secondary win condition understated FIVE-FOLD, in the artifact the engine model
and the dossier both quote. It is not a heliod problem: any deck whose damage
comes off noncreature permanents is understated, and the four seats in this pod
are wrong by different factors, so seats are not comparable on the axis either.

**Why it is not fixed yet.** The fix is to teach the owner map from the lines
that put a permanent onto the battlefield, not just the three that happen to
name a controller. That changes `analyze` output for every stored run, and
`validate-sim` re-derives `analysis` from the logs — so every tracked sim record
must be re-derived in the same commit, and every figure quoted from one
re-checked. It is a parser change with a fleet-wide blast radius and wants its
own session, its own corpus sweep over the log grammar, and a test that fails
first on a stored log.

**And what survives the drop is pointed the wrong way.** Re-derived over the
same run, the 201 the record *does* credit heliod breaks down as:

    Viseling         117      (of its true 382)
    Adarkar Wastes    82      aimed at HELIOD'S OWN SEAT
    Walking Ballista   2

**41% of the deck's measured "noncombat damage dealt to players" is a painland
hurting its own controller.** That is not a coding error — Adarkar Wastes really
does deal noncombat damage to a player, and the metric never said *to opponents*
— but every consumer reads the figure as offence, and on this deck it is closer
to a tax than a threat. Found independently by `deck-engineer` in round 2 and by
this repo's own re-derivation; the two agree on the mechanism.

So the metric is wrong twice over: it misses seven-eighths of what the table
deals, and a large minority of the remainder is self-inflicted. Fixing the owner
map addresses the first. The second needs a decision — either split the figure
by whether the target is an opponent, or rename it so it stops being read as
damage the deck *did to someone else*.

**UPDATED 2026-09-09, and the case is now much stronger.** With the three
punishers cut in skies-v1, heliod's v1.2.1 run reads:

    at ITSELF    128   Adarkar Wastes 83 · Talisman of Progress 43 · two more
    at OTHERS     20   Walking Ballista, all of it

**Eighty-six percent of the deck's measured "noncombat damage dealt to players"
is the deck hurting itself**, and twenty points across 120 games is what reaches
an opponent. The figure is now not merely polluted — for this deck it is
predominantly a measure of its own manabase.

That also settles which fix matters more. The owner-map bug was the bigger
number and it is fixed; the naming decision is the one still doing damage,
because the metric's current definition makes a painland indistinguishable from
a win condition. The two candidates remain: split by whether the target is an
opponent, or rename it so it stops being read as offence.

**Do not quote a `noncombat_damage_dealt_to_players` figure until both land.**

---

# The rest of the inventory

Sections 1–4 are the red tests and the one confirmed engine bug. Everything
below is a real gap that **no test is currently failing on**, which is why it
needs writing down: a problem nothing asserts is a problem nobody is reminded of.

Taken 2026-09-08 by sweeping `deck-status` across the live fleet, the strict-
xfail registries, the engine models' `open_questions`, the branch states, and
the captain's logs.

## 5. Fleet artifact state

| deck | OK | STALE | FAIL | gated | unstamped |
|---|---:|---:|---:|---:|---:|
| edgar-vampires | 14 | 0 | 0 | 7 | 2 |
| gishath | 14 | 0 | 0 | 5 | 3 |
| goblin-storm | 14 | 0 | 0 | 6 | 4 |
| heliod | 13 | 0 | **1** | 5 | 2 |
| radagast | 14 | 0 | **1** | 4 | 4 |
| sisay | 13 | 0 | **1** | 2 | 4 |
| ur-dragon | 9 | **4** | **1** | 6 | 0 |
| zur-enchantress | 8 | 0 | 0 | 3 | 0 |

Two FAILs are listed here — **radagast and sisay both fail `diagnosis.json` on
the same axis**, `axes[3] (colour-sources)`, where the diagnosis's
`measured.value` disagrees with what `deck-audit` computes (9 against audit for
radagast, **−15** for sisay). A negative colour-source count is not a plausible
measurement, so this is one bug in one place rather than two decks drifting.

**CORRECTED 2026-09-08: neither deck is live.** `radagast` is BROKEN DOWN FOR
PARTS and `sisay` is RETIRED, both since August. The first version of this table
called them live, because the sweep that built it read
`deck_versions.json`'s lifecycle block as `.get("state")` when the key is
`status` — so the `.get(..., "living")` default answered for every deck in the
fleet and the column meant nothing.

That changes what the row is worth. A wrong figure in an archived deck's
diagnosis is a published error, so it stays on this page — but it is a deck that
no longer physically exists, and the rule about not manufacturing artifacts for
a pile of cards applies to fixing it as much as to regenerating it. Priority:
below everything else here.

## 6. Staleness is undecidable on 19 artifacts

`unstamped — staleness cannot be checked` appears **19 times across six decks**.
An agent-authored artifact that carries no decklist sha cannot be told from a
current one, so the gate reports OK and means "no opinion". ur-dragon and
zur-enchantress are the only decks at zero, because they are the two whose
artifacts were most recently rebuilt.

This is the quiet version of the staleness problem the `meta.model_version`
stamp was introduced to solve for computed figures. The authored side never got
it. Until it does, "OK" on those rows is not evidence.

## 7. Two engine models have never been criticised

`critic` is null on **edgar-vampires** and **ur-dragon** (heliod's is now `fail`
— see §2). No test notices, because only heliod is named by one. Three of eight
live decks were in this state this morning and exactly one test could see it.

## 8. Fifty open questions, and no queue

The engine models carry **50 `open_questions`** — 37 on live decks — routed
`resolve-stack` 30, `goldfish` 11, `research-strategy` 9. `analyze-engine`
step 7 says the orchestrator dispatches these, because subagents cannot spawn
subagents. Nothing tracks which have been dispatched. They accumulate.

The `resolve-stack` thirty are the cheapest real evidence available: the fleet's
verified-line counts are as thin as 1-of-11 (heliod), and each of those thirty
is a scenario waiting to be written.

## 9. Eleven games logged, zero debriefed

| deck | logged | debriefed |
|---|---:|---:|
| edgar-vampires | 4 | 0 |
| ur-dragon | 3 | 0 |
| goblin-storm | 2 | 0 |
| gishath | 1 | 0 |
| heliod | 1 | 0 |

No log entry on any deck carries a `debrief` key. The captain's log is the only
artifact in the bench sourced from a real table rather than a simulation, and
none of it has been read back. `/debrief` is the cheapest agent in the set.

## 10. Branch hygiene

Twenty-five branch directories exist. **Eight are MERGED and still on disk** —
seven on zur-enchantress, one on heliod (`archangel-v1`, merged into v1.2.1).
Seventeen are open experiments on zur-enchantress alone.

Every merged branch records its target as **`into_version_before: null`**, which
is why `branch_state` prints "merged … into the list after **VNone**" for all
eight. `base_version` is written correctly (9 on archangel-v1); the merge record
simply never fills the field it prints. Cosmetic today, wrong in an artifact
that exists to say what a change was measured against.

## 11. Two artifacts are held by retired agents

`STALE_XFAIL` and `ISSUE_XFAIL` in `tests/test_pilot_tracked_artifacts_validate.py`
carry three entries between them, all strict:

- `heliod/considering.json` — the Short List, part of the frozen magazine
  renderer whose editor was retired 2026-08-19
- `edgar-vampires/issue.json` — prose predating THE LOCK's 12 swaps
- `ur-dragon/issue.json` — quotes "31 lands" where the deck runs 36 copies;
  the copies-vs-entries defect, live in tracked prose

**There is no agent left to re-run for any of them.** The honest options are to
delete the artifacts or leave them marked. They are marked, which keeps the gate
live for the other seven decks — but it is a permanent xfail, and a permanent
xfail is a decision deferred rather than made.

Note that **both heliod entries came OFF this list on 2026-09-08** when the
rebuilt `engine.json` and `tutor_guide.json` XPASSed — the strict marker working
exactly as designed.

## 12. Fixed since the last sweep — do not re-investigate

Six bugs recorded during the PRD exploration were checked again today and five
are gone: `experiment`'s `KeyError: 'ci95_a'` on the final print; `experiment`
running a different pod profile from `simulate` (it reads `STANDARD_POD_PROFILE`
now); `_run_arm` having no timeout (it passes `per_job_cap`); the `manuals/p/`
three-writer collision (`build_page` has no default output any more, so `poh`
owns it); and `mulligan` being parsed then discarded (`mulligans_taken` and
`mulligan_kept` are both aggregated, with the London-mulligan derivation
recorded).

**One is still live:** `experiment` does not rotate seats. Both arms sit at
`Ai(1)` (`experiment.py:416-417`), while `forge.py` rotates per job for
`simulate` and is careful to rotate the AI profiles alongside the decks. So an
A/B carries whatever seat-1 bias the table has, and the two commands are still
not measuring under the same conditions — a narrower version of the pod-profile
bug that was fixed.

## 13. Every Forge figure for heliod describes a deck that was replaced

The 120-game standard-pod run is dated 2026-09-07. `skies-v1` and
`archangel-v1` both merged on **2026-09-08**. The run's own record says which
list it played — `seats[0].decklist_sha256` is `decc32b0…` against a current
`214af30…` — and **Viseling, Iron Maiden and Ebony Owl Netsuke were all cut**
in that merge.

So the entire punisher axis argued over in two rounds of the engine loop, the
1,003-against-202 damage discrepancy, the 45% commander-removal rate, the
39-of-53 eliminations from the air, and the flip cross-tab
(23/85 against 4/35 — since RETRACTED, see §2) all describe **v1.0.0**, not the v1.2.1
list that is sleeved. `engine.json`, `diagnosis.json` and `strategic_frame.json`
all quote from it.

Nothing here is *wrong* — every figure is a true statement about the list it
measured, and the record names that list. What is missing is a reader who knows
that. The parser defect the punisher argument uncovered is entirely unaffected:
it is about attribution, not about which cards were in the deck, and it was
measured across 33 runs and 1,943 games.

**What unblocks it:** a Forge run on v1.2.1. Until then, a Forge figure quoted
for heliod needs the version said out loud beside it.

*How it was found:* the pilot, reading a claim about his own deck — "viseling was
pulled from the recent deck list". Two agent rounds and a fleet-wide parser sweep
had all reasoned about those three cards without one of them checking the list.

## What is NOT on this page

- **`make test-browser`** is a local pre-push gate and is deliberately outside
  CI. It is not covered by the counts here.
- **hapatra and yawgmoth-swarm** are archived as broken down for parts
  (2026-09-08). Their artifacts sit at a model version four changes old, and
  that is correct — regenerating a deck that no longer physically exists
  manufactures figures about a pile of cards. `regen` skips them by design.
- **radagast and sisay** are live but unpinned, and `regen --slug` reports "no
  tracked artifacts matched" for both: their `goldfish_metrics.json` is tracked
  but no stage claims them, so a model change leaves them silently stale. No
  test fails on it today. It is the same shape as the gap `regen.BOOTSTRAP` was
  written for, one rung down — worth closing before it costs something.
