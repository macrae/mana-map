# Known issues

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

**Cause.** `data/decks/heliod/engine.json` has `critic: null`. The engineer was
re-run on 2026-09-03 *with the critic's findings*, and the resulting document was
never put back in front of the critic — so the model that answered the critique
carries no verdict saying it answered it. `deck_info.py:701` reads
`critic.verdict != "pass"` and the panel comes back unverified.

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
