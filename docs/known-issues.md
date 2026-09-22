# Known issues — the inventory

**The board of what is red and why.** Each row says what it is measuring, why it
fails, and **who or what unblocks it**. A row leaves this page when the test is
green, never because the test was changed to suit the artifact.

Last verified against **`make test-fresh`**: **2026-09-21**, **3,695 passing /
0 failing** / 8 skipped / 3 xfailed, 617 s.

## THE SUITE IS GREEN. `corpus-gates` is red ON PURPOSE (§9c).

**CI, as of 2026-09-22.** The `tests` workflow has two jobs and they mean
different things:

| job | state | |
|---|---|---|
| `test` (every push) | **GREEN** | must stay green — this is the signal |
| `corpus-gates` (weekly) | **RED, expected** | the corpus is 266 cards ahead of the tracked artifacts; refresh after FRA on 2026-10-02 (§9c) |

The `test` job went green on 2026-09-22 for the first time since **2026-08-25**,
and the byte-diff determinism gate ran for the first time in that whole period —
it had been short-circuited by the failing `Test` step, exactly as its own
comment predicted. It caught the suite writing into `data/decks/` on every run
(`docs/testing.md`).

## The gaps on this page are the ones no test fails on.

That is a change of kind, not of degree, and it is why this page was almost
entirely wrong when it was audited on 2026-09-21. **Every red it listed had been
fixed and it still said "`make test` does not pass on `main`."** A page that
inventories failures has no signal when there are none, and it drifts fastest
exactly when the board is clean, because nothing forces anybody to open it.

So the sections below are now in two groups, and the second is the real content:

- **What FAILS** — currently nothing in `make test`. Four archived decks fail
  their own validator (§5) and `deck-status --all` exits non-zero on them.
- **What no test can see** — staleness nothing stamps (§6), engine models
  nobody criticised (§7), open questions nobody dispatched (§8), sleeved decks
  no simulation describes (§13), and the debt list.

**VERIFY WITH `test-fresh`, NEVER WITH `make test`.** The two still disagree, and
this audit re-confirmed it with a fresh instance: warm, `make test` reported
3,216 passing and served 481 from the regenerate-and-compare cache; uncached, the
same commit failed on **five heliod branch `goldfish_metrics.json`** that the
cache had served as passing. The cause is the one filed as **#49** — the cache
keys a branch's case on the branch's own inputs, and a branch has **no
`goldfish_targets.json` of its own** (see `docs/gotchas-bench.md`), so editing
the DECK's declaration stales every branch under it while looking like a no-op.
Until #49 is fixed, a warm run cannot establish this page.

### Cleared since the last sweep

| was | what actually closed it |
|---|---|
| ur-dragon's three artifacts describing a list nobody checked in (§1) | the pilot checked the paper list in; V6 is sleeved as v1.3.0 and `validate-engine ur-dragon` is OK |
| the arrival channel depending on `model_combat` (§3b) | re-derived 2026-09-10 — the remaining gap is the cast-token channel and is legitimate; the threshold moved 0.7 → 0.4 |
| two artifacts held by retired agents (§11) | the magazine renderer was deleted 2026-09-13, so nothing gates them; `STALE_XFAIL` is now empty and `ISSUE_XFAIL` does not exist |
| `model_colors` conflating a constraint with a bonus (#35) | fixed 2026-09-13 — see §3c, which is kept for the mechanism |
| the sacrifice runaway guard firing on edgar (#34) | fixed 2026-09-13 — see §3a |

---

## 1. ur-dragon: THREE artifacts describe a deck that is not on disk — **RESOLVED 2026-09-20**

*The pilot checked the paper list in. V6 is sleeved as v1.3.0
(`decklist_sha256` `99419d5b91…`, built 2026-09-20), the artifact chain was
rebuilt against it, and `validate-engine ur-dragon` now reports
`OK — 7 stage(s), 97 card(s) placed, 0 unassigned, 5/8 line(s) verified`. All
three tests below are green. Kept for the rule it demonstrates, which held:
**the engineer pass sitting unmerged in `.agent-out/` was deliberately not
applied** until a human confirmed the list, and that is why nothing was
published under a guess.*

*Below is the state as it stood.*

*Corrected 2026-09-12. This grouped FIVE tests and was wrong about two of them.
`versions_json_matches_a_fresh_run` and `poh::a_rebuild_is_byte_identical` had
nothing to do with the phantom cards: the version file was missing the paper
block and the handbook's revision line read `UNBOUND — no version tag` instead
of `Applies to the sleeved list, v1.2.1`. Both were `adfad9e5` setting the lock
without regenerating what derives from it, both were cleared by `make manuals`,
and neither needed a check-in or an agent. Grouping them here made the check-in
look like it would fix five things when it fixes three.*

| test | measuring |
|---|---|
| `test_pilot_tracked_artifacts_validate[ur-dragon/engine.json]` | every card the engine model names is in the 99 |
| `test_pilot_agent_stamps::test_no_agent_artifact_names_a_card_the_deck_does_not_run` | the same claim, swept across every sleeved deck |
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

Fixing these before the check-in would have meant writing artifacts that
describe a guess. (The correction note above says the grouping "fixes three",
and this sentence said "these five" — the two never agreed, which is its own
small lesson about editing a page in layers.)

---

## 2. Four sleeved decks have no criticised engine model

*Retitled and re-measured 2026-09-21. It named one deck and reported a `fail`
verdict; **heliod's critic block is now GONE**, not failing — `engine.json` was
rebuilt for v1.4.0 and the critic never re-ran over it. Losing a `fail` is not
an improvement: a saved `fail` documents what could not be grounded, and absence
documents nothing.*

**The live state**, from `(engine.json).critic.verdict` across the six sleeved
decks:

| deck | critic | findings |
|---|---|---:|
| edgar-vampires | `pass` | 18 |
| goblin-storm | `pass` | 9 |
| **gishath** | **absent** | 0 |
| **heliod** | **absent** | 0 |
| **sharknado** | **absent** | 0 |
| **ur-dragon** | **absent** | 0 |

Four of six, and three of those four are the decks whose models were rebuilt
this month — gishath v1.1.0, heliod v1.4.0, ur-dragon v1.3.0, plus sharknado
which was modelled and never criticised at all. **Rebuilding an engine model
drops its critic block, and nothing re-runs the critic.** That is the mechanism
worth naming: the `/analyze-engine` loop ends at the critic, so a regeneration
that stops after `validate-engine` passes leaves the model uncriticised and the
board looks the same as one that was never modelled.

**The test that watches this is weaker than its docstring claims.**
`test_every_sleeved_deck_has_a_criticised_engine` says it is
`xfail(strict=True)` and so "goes red the moment the fleet is clean and somebody
has to delete this". **There is no marker.** It calls `pytest.xfail()`
imperatively inside `if uncriticised:` — so a clean fleet makes it pass silently
and nobody is ever told. Its guard, `assert len(uncriticised) < 5`, is at **4**:
one more deck and it fails with the message "no sleeved deck has a passing
engine critic — has the key moved?", which would be wrong.

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

**What unblocks it.** An `engine-critic` run on each of gishath, heliod,
sharknado and ur-dragon — the `/analyze-engine` loop, after `validate-engine`
passes, which it does on all four. edgar-vampires and goblin-storm are done.

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

### 3a. The runaway guard fires on a real deck — **FIXED 2026-09-13**

The guard asserted zero and read 0.001–0.007 on edgar-vampires. The advice at
the time was right — "do not raise the cap; find which games hit it and what the
board looked like" — and the board turned out to be the answer.

**IT WAS GUARDING BOARD WIDTH AND CALLING THAT A LOOP.** The check was
`n_sac >= SAC_LIMIT_PER_TURN` with the limit at twenty, so a turn converting
more than twenty tokens was TRUNCATED and the truncation counted as a runaway.
Edgar reaches that under eminence with Anointed Procession and Mondrak: measured,
a busy game converts **thirty-one on turn ten**.

There was no loop to guard. The sweep iterates a SNAPSHOT of a list it never
appends to, and no death payoff creates a battlefield entry — they drain, draw
and make Treasure. Lifting the cap entirely took the hit rate to zero with the
per-turn series unchanged, which is what proved the cap was guarding nothing.

**A SECOND DEFECT HID THE FIRST.** `sacrifices_by_turn` was appended BEFORE the
combat step that does the sacrificing, so the series was shifted one turn and
the final turn's sacrifices were never recorded at all. That is how a game could
report

```
sac_cap_hits: 1   beside   sacrifices_by_turn: [0,0,0,0,0,0,0,0,0,0]
```

— the board went wide and converted on turn ten, and the only field that could
have shown it had already been written. Anyone looking at the series would have
concluded the sacrifice channel was not firing at all.

**The fix.** `SAC_LIMIT_PER_TURN` is retired. The guard now asserts the
battlefield does not GROW during the sweep, which is the loop stated as itself:
if a death payoff ever starts creating a creature, that is caught by what it
actually is rather than by a number somebody tuned. The series is recorded at
the end of the turn.

**What moved, and only on edgar** — the one deck that declares `model_sacrifice`:

| figure | was | now |
|---|---:|---:|
| `mean_sacrifices_by_turn["10"]` | 1.996 | 2.966 |
| `sac_cap_hit_rate` | 0.001 | 0.0 |
| `combat.mean_board_power_by_turn["10"]` | 21.469 | 21.467 |

The sacrifice figure was understating by half because the busiest turn was both
truncated and unrecorded. The test is fleet-scoped now rather than named on the
one deck that happened to trip it.

### 3b. The arrival channel depends on a flag it should not — **RESOLVED 2026-09-10**

*Resolved three days BEFORE this page's previous "last verified" date, and the
page still carried it as open and unowned. The gap that remains is legitimate
and was re-derived rather than fixed: the cast-token channel (`9d2efd9`) rides
on `model_combat`, and on edgar that is eminence — a body on every other Vampire
cast, each a real arrival that fires the arrival draws. So combat-on reads 2.671
against 1.325 draw-only and the difference IS the tokens. The threshold moved
0.7 → 0.4 with that reasoning recorded in the test. The original defect — every
call to `creature_entered` sitting inside `if model_combat:` — is fixed.*

*Below is the state as it stood.*

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

### 3c. Colour screw does not bite a five-colour deck — **FIXED 2026-09-13**

`model_colors` is supposed to make a five-colour manabase *worse* than a
colour-blind one. On ur-dragon it made it **better** — 0.319 against 0.219, a
move of +0.100 against an SE of 0.022, so z ≈ 4.5 and not noise. A mono-colour
deck was correctly indifferent.

**THE FLAG WAS NOT A CONSTRAINT. It added a penalty and unlocked a bonus at the
same time.** `sources` — the colours on the battlefield — was appended to only
under the flag:

```python
if model_colors:
    sources.append(played["colors"])     # playing a land
```

and colour-scaling producers READ it, in both arms, to size their own output:

```python
if card["scales_with_colors"]:
    colors = frozenset().union(*sources) if sources else frozenset()
    made = max(1, min(len(colors), 5))
```

So with the flag off `sources` was empty, and every `scales_with_colors`
producer fell to `max(1, 0)` — **one mana instead of up to five**. Bloom Tender
and Faeburrow Elder are both in ur-dragon. On a mono deck the bonus is worth
nothing and the constraint is nearly free, which is why only the five-colour
half of the test failed.

`sources` is tracked unconditionally now and read only under the flag — which
is what `goldfish_library.classify`'s own comment had claimed all along:
"both ride along always and are READ only under `model_colors`". The turn loop
broke that rule and the comment was the tell.

**Measured after, on the same seed:**

| deck | flag on | flag off | delta |
|---|---:|---:|---:|
| goblin-storm (mono-ish) | 0.904 | 0.905 | −0.001 |
| heliod (two colours) | 0.892 | 0.931 | −0.039 |
| ur-dragon (five colours) | 0.324 | 0.357 | −0.033 |

A constraint on all three, and indifferent on mono.

**NO PUBLISHED FIGURE WAS EVER WRONG.** `model_colors` defaults to TRUE and no
deck declares it false, so the defect lived entirely in the CONTROL arm — the
one the test and any blind-arm sweep uses. The fleet re-measured byte-identical
apart from `meta.model_version`. The land-colour reader, this section's first
suspect, was not involved.

---

## 4. The parser drops 88% of all noncombat damage — **the owner map is FIXED; the naming split is not**

*Header corrected 2026-09-12: it said "unfixed" while this section's own update
at the end recorded the owner map landing in `parse.py:257-285`. What remains is
the naming split, and it is filed.*

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

*Dated table, not re-swept. **`zur-enchantress` was broken down on 2026-09-10**
as a failed experiment, so its row describes a deck that is now a pile — the
same status as radagast and sisay below. `sharknado` and `ingris-infect`, the
two decks on the bench, were never in this sweep.*

## 5. Fleet artifact state — four archived decks fail their own validator

*Re-measured 2026-09-21. The old table listed 8 of what are now 14 decks, showed
ur-dragon at 4 STALE when the fleet is at ZERO, and named two FAILs when there
are four. Its stated CAUSE was also wrong — see below.*

`deck-status --all`: **14 decks, 0 stale, 4 failing their own gate, 1 queued
change.** The four are the same artifact on the same axis:

| deck | lifecycle | `diagnosis.json` axes[3] (colour-sources) |
|---|---|---|
| hapatra | `broken-down` | says 15, audit computes 11 |
| radagast | `broken-down` | says 9, audit computes 8 |
| sisay | `retired` | says −15, audit computes −8 |
| yawgmoth-swarm | `broken-down` | says 2, audit computes 1 |

**THE OLD DIAGNOSIS OF THE CAUSE WAS WRONG, and it is worth saying how.** This
page argued: *"A negative colour-source count is not a plausible measurement, so
this is one bug in one place rather than two decks drifting."* But negative is
the axis's own unit. `deck-audit` reports
`"unit": "sources above target (worst colour)"`, so sisay's −8 means **eight
sources SHORT on its worst colour** (B: 28 have against 36 target) — a perfectly
plausible measurement of a deck that is short on black.

What is actually wrong is simpler and less interesting: **each diagnosis carries
a figure from an older audit**, and `validate-diagnosis` re-derives the axis and
compares. Four stale numbers in four archived artifacts, not one bug.

That reading was reached by looking at a number and finding it implausible
rather than by reading the unit printed beside it — the same shape as a mean
read as a rate, which `docs/gotchas-bench.md` already records twice.

**All four decks are archived** (three `broken-down`, one `retired`). The rule
about not manufacturing artifacts for a pile of cards applies to fixing these as
much as to regenerating them: a wrong figure in an archived deck's diagnosis is
a published error and stays on this page, but it is below everything else here.

## 6. Staleness is undecidable on 26 artifacts

*Re-counted 2026-09-21: 19 across six decks has become **26 across ten**, which
is the direction this goes on its own — every new deck and every agent artifact
adds to it.*

`unstamped — staleness cannot be checked` appears **26 times across ten decks**:
goblin-storm, radagast, sisay, hapatra and yawgmoth-swarm at 4 each, gishath 2,
and edgar-vampires, heliod, sharknado and ur-dragon at 1.
An agent-authored artifact that carries no decklist sha cannot be told from a
current one, so the gate reports OK and means "no opinion". The decks nearest zero are the ones most recently rebuilt; the archived ones
are worst, because nothing has regenerated them since stamping shipped.

This is the quiet version of the staleness problem the `meta.model_version`
stamp was introduced to solve for computed figures. The authored side never got
it. Until it does, "OK" on those rows is not evidence.

## 7. Engine models nobody criticised — **MERGED INTO §2**

*This said `critic` is null on edgar-vampires and ur-dragon with heliod at
`fail`. Every part of that has moved: edgar now PASSES, heliod's block is GONE
rather than failing, and the real set is four decks. §2 carries the live table,
the mechanism (a rebuilt model drops its critic block and nothing re-runs it),
and the defect in the test that watches it.*

## 8. Sixty open questions, and no queue

*Re-counted 2026-09-21: fifty has become **sixty**, which is the whole point of
the section — nothing dispatches them, so they only ever go up.*

The engine models carry **60 `open_questions`**, routed `resolve-stack` 37,
`goldfish` 13, `research-strategy` 10. By deck: hapatra 9, heliod 8, sharknado
8, radagast 7, and 5 each on edgar-vampires, gishath, goblin-storm and
ur-dragon, 4 each on sisay and yawgmoth-swarm. `analyze-engine`
step 7 says the orchestrator dispatches these, because subagents cannot spawn
subagents. Nothing tracks which have been dispatched. They accumulate.

The `resolve-stack` thirty-seven are the cheapest real evidence available: the fleet's
verified-line counts are as thin as 1-of-11 (heliod), and each of those thirty
is a scenario waiting to be written.

## 9. Eleven games logged, zero debriefed — **RESOLVED 2026-09-11**

| deck | logged | annotated |
|---|---:|---:|
| ur-dragon | 6 | 6 |
| edgar-vampires | 4 | 4 |
| goblin-storm | 2 | 2 |
| heliod | 2 | 2 |
| gishath | 1 | 1 |

**Fifteen games, every one annotated.** The `log_annotations.json` files were
written 2026-09-11. Kept as a row rather than deleted because the argument
survives the fix: the captain's log is still the only artifact in the bench
sourced from a real table rather than a simulation, so it is the one that
silently stops being read.

What is still open is the other half of **#39** — Edgar's direction change is
recorded in the log and has not been built.

## 9a. A model change stales the bench, `regen` will not touch it, and a test fails on it

*Found 2026-09-21 by changing the goldfish and running the suite.*

Three behaviours that each make sense alone and disagree with each other:

| | |
|---|---|
| `regen` (fleet pass) | covers **sleeved decks only** — 33 goldfish targets across the six with a paper lock |
| `regen --slug <bench deck>` | works fine, finds the targets, regenerates them |
| `test_goldfish_metrics_match_a_fresh_run` | covers **every non-archived deck**, bench included |

So a model-version bump stales `emiel-blink` and `meren-recursion`, the fleet
regen deliberately skips them, and the suite goes red on two decks the
documented workflow says not to rebuild automatically.

**The rule the fleet pass follows is right** (CLAUDE.md: *"Building it
automatically would manufacture artifacts for a list that will be different
tomorrow"*) — but it does not fit this case. These artifacts **already exist**;
regenerating one is not manufacturing it, and the test's own premise is that a
tracked artifact must match a fresh run. The archived decks are handled
correctly and consistently: `regen` skips them AND the freshness test skips them.
Bench decks are the gap — skipped by one and not the other.

Two honest fixes, and it is a policy call rather than a bug:

1. **Fleet regen covers an artifact that already exists**, wherever it lives,
   and keeps `BOOTSTRAP` (creating a missing one) restricted to sleeved decks.
   That is already the distinction `regen.is_pinned` was written to draw.
2. **The freshness test skips bench decks** the way it skips archived ones,
   accepting that a bench deck's figures may be stale.

(1) is the better answer — a stale tracked artifact is a published error whether
or not the deck is sleeved. Until it is decided, **regenerate bench decks by hand
after any model change**: `regen --slug emiel-blink --slug meren-recursion`.

## 9b. The axis-independence gate is failing, and it is not in the default suite

*Found 2026-09-21 by running the suite with the `fleet` marker included.*

`test_metric_hygiene.test_no_two_axes_measure_the_same_thing` is the check
CLAUDE.md describes — *"Adding a metric requires re-running the independence
check. Three magnitude axes shipped that were one axis at r = 0.92–0.98."* It
is over the line again on two pairs, against a `MAX_AXIS_CORRELATION` of 0.90:

```
damage_8 ~ hoard_10:  r=+0.91
hoard_10 ~ hoard_6:   r=+0.96
```

**Nothing has been reporting this.** `pyproject.toml` sets
`addopts = "-m 'not browser and not forge and not fleet'"`, so the gate is
excluded from `make test` AND from CI, by design and for time — the test's own
docstring says *"this is a gate you must ask for"*. Nobody has asked.

`hoard_6 ~ hoard_10` at 0.96 is the easy one to read: the same measure at two
turns is one measure, and a branch aimed at both gets two confirmations of one
fact. `damage_8 ~ hoard_10` at 0.91 is the interesting one and is the shape the
original finding had.

**This is a judgement about the metrics catalog, not a test fix.** Raising the
threshold would defeat the check; deleting an axis is a decision about what a
pilot may aim a branch at. `candidates.OBJECTIVE_AXES` currently offers 18, and
O3 above argues for ADDING one (drain) — which should be weighed against this.

Run it with `pytest -m fleet`.

## 9c. `corpus-gates` is RED ON PURPOSE until Reality Fracture — expected, not ignored

*Decided 2026-09-22. Revisit after FRA releases 2026-10-02.*

The weekly `corpus-gates` job downloads a FRESH Scryfall corpus and runs the
gates a per-push job cannot reach. It is failing, and **all four failures are
one fact**: the corpus has moved and the tracked artifacts have not.

```
TestCardCountConsistency.test_projection_count      assert 34890 == 35156
TestCardCountConsistency.test_embeddings_bin_size   assert 17863680 == 17999872
…is_unit_norm_and_the_right_length[cardbert]        expected 35156 x 128 x 4
…is_unit_norm_and_the_right_length[function]        expected 35156 x 128 x 4
```

Every tracked matrix is at **34,890** — `projection_2d.json`, `embeddings.bin`,
`embeddings_ability.bin`, `embeddings_cardbert.bin`, all 17,863,680 bytes —
against a live Scryfall at **35,156**. Exactly 266 cards behind, and the other
3,239 tests in that job pass.

**This is the gate working.** It is not a bug in the artifacts, the pipeline or
the tests: it is the index-alignment invariant (`projection[i] == cards.csv[i]
== embeddings[i]`) correctly reporting that the corpus has drifted from what was
trained on.

### Why it is not being fixed today

**Reality Fracture releases 2026-10-02**, ten days out, and the standing rule is
*refresh after a set release, never during a Forge run*. A refresh now means a
retrain — the ability model is ~1h on MPS — and it would be superseded almost
immediately by FRA. So the honest move is to refresh once, after FRA, and take
the alignment forward in one step. `.claude/skills/refresh-corpus/SKILL.md` is
the runbook.

### THE RISK THIS ENTRY EXISTS TO MANAGE

A deliberate red is one bad habit away from an ignored red, and **this repo has
just spent four weeks proving it.** The `tests` workflow failed on every push
from 2026-08-25 to 2026-09-22 — 143 failures against 21 successes over its life
— and in that time the byte-diff determinism gate never ran once, because the
`Test` step short-circuited it. When it finally ran it caught the suite writing
into `data/decks/` on every run (see `docs/testing.md`). Nobody decided to
ignore CI; it just stopped meaning anything, one push at a time.

So this red is **scoped, dated and owned**, and it has a closing condition:

- **Scope.** ONLY `corpus-gates`, and only these four card-count assertions. The
  per-push `test` job is green and must stay green — that is the signal.
- **Close it by:** `manamap run --from download` after 2026-10-02, then commit
  the refreshed artifacts. The four assertions go green on their own; nothing
  needs editing.
- **If a FIFTH failure appears in this job, that is a real finding** and does not
  belong under this entry.

Do not raise the thresholds, do not mark the tests xfail, and do not add
`corpus-gates` to a list of jobs that are allowed to be red. The count check is
the only thing standing between a retrain and a silently misaligned index.

## 10. Branch hygiene

**Thirty-nine** branch directories exist (twenty-five when this was written).
**Eight are MERGED and still on disk**, unchanged — seven on zur-enchantress,
one on heliod (`archangel-v1`, merged into v1.2.1).

Every merged branch records its target as **`into_version_before: null`**, which
is why `branch_state` prints "merged … into the list after **VNone**" for all
eight. `base_version` is written correctly (9 on archangel-v1); the merge record
simply never fills the field it prints. Cosmetic today, wrong in an artifact
that exists to say what a change was measured against.

## 11. Two artifacts are held by retired agents — **RESOLVED 2026-09-13**

*`STALE_XFAIL` is now EMPTY and `ISSUE_XFAIL` does not exist. The magazine
renderer and its validators were deleted on 2026-09-13, so `considering.json`
and `issue.json` are no longer gated by anything and an xfail has nothing to
attach to. The permanent-xfail question this section posed — "a decision
deferred rather than made" — was answered by deleting the renderer, which is
the decision.*

*Below is the state as it stood.*

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
three-writer collision (`build_page` was DELETED with the magazine renderer on
2026-09-13, so `poh` owns the path outright — this said "has no default output
any more", which was the state for eleven days); and `mulligan` being parsed then discarded (`mulligans_taken` and
`mulligan_kept` are both aggregated, with the London-mulligan derivation
recorded).

**One is still live:** `experiment` does not rotate seats. Both arms sit at
`Ai(1)` (`src/manamap/sim/experiment.py:416-417` — re-verified 2026-09-21), while `forge.py` rotates per job for
`simulate` and is careful to rotate the AI profiles alongside the decks. So an
A/B carries whatever seat-1 bias the table has, and the two commands are still
not measuring under the same conditions — a narrower version of the pod-profile
bug that was fixed.

## 13. Every Forge figure describes a deck that was replaced — **REOPENED, AND IT IS FLEET-WIDE**

**Marked RESOLVED on 2026-09-12 for heliod, and the condition came straight
back — on five of the six sleeved decks.** That is the finding: this is not a
heliod problem that was fixed, it is what happens to every deck every time it is
sleeved at a new version, and nothing notices.

Measured 2026-09-21, each deck's `cards.json` sha against
`seats[0].decklist_sha256` on every run under `sim/`:

| deck | sleeved | runs | runs describing the sleeved list |
|---|---|---:|---:|
| goblin-storm | V1 | 4 | **4** |
| edgar-vampires | v1.1.1 | 9 | **0** |
| gishath | v1.1.0 | 3 | **0** |
| heliod | v1.4.0 | 4 | **0** |
| sharknado | v1.0.1 | 1 | **0** |
| ur-dragon | v1.3.0 | 4 | **0** |

Twenty-one Forge runs on the fleet, and **only goblin-storm's four describe the
list in its sleeves** — because goblin-storm is the one deck that has not
changed since V1. Every figure quoted for the other five is a true statement
about a list nobody is holding.

**Nothing here is wrong and nothing warns.** Each record names the list it
played (`seats[0].decklist_sha256`), which is exactly the honesty this bench is
built on; what is missing is a READER that joins the two. `deck-status` reports
stale ARTIFACTS and has no notion of a stale RUN, so a dossier prints a win rate
beside a version it does not belong to and says nothing.

**This is cheap to close and has been open twice.** The join is one comparison —
`sim_record.seats[0].decklist_sha256` against `cards.json`'s — and it has the
same shape as the freshness stamp every computed artifact already carries. Until
it exists, the rule from the first time round stands and is now fleet-wide: **a
Forge figure quoted for any deck but goblin-storm needs the version said out
loud beside it.**

*Below is the original heliod-only section, kept for the lesson and for how it
was found.*

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

---

# Debt, gaps and opportunities

Not red, not blocking, and worth writing down before they are rediscovered.
Sections 1–14 above are things that FAIL; these are things that are merely
wrong, wasteful, or half-built. Added 2026-09-09 while rewriting the captain's
log — most were found by reading code adjacent to the change rather than by
looking for them, which is the usual way.

| # | what | where | why it matters |
|---|---|---|---|
| D1 | **A backfilled game is stamped with TODAY's decklist** | `deck_notes.append_entry` — `--at` overrides the date, never the sha | ur-dragon entry 004 was played 3 Sep on V3 and joins to V4. The log-to-version join is by sha, so a correctly-logged game attaches to the wrong list. Now VISIBLE in `captains_log.read_meta`, which is how it was confirmed |
| D1a | **A backfilled entry can claim a list it was never played on — and gishath proves it** | `deck_notes.append_entry`; `data/decks/gishath/log.jsonl` entry 001 | The game was played 2026-08-28 and logged on 09-01, so `--at` set the date and stamped the CURRENT sha. That sha **matches `decklist.txt` exactly today**, and the note says the game was won by drawing **Enlarge — which has never appeared in any committed version of the list.** So the entry asserts it was played on a deck it demonstrably was not. Found by the `debrief` agent, not by any check. **CONFIRMED BY THE PILOT 2026-09-09: the paper deck has Enlarge and he won with it.** So the tracked 99 differs from the sleeved deck by at least one card in each direction — the repo list is 100 cards WITHOUT Enlarge, so something in it is not in paper either. Every artifact under `gishath/` is computed against a list he does not play. Needs `check-in`, exactly like ur-dragon |
| D2 | **`supplementals` is silently dropped by the merge** | `merge_captains_log._sections` whitelists `SECTION_KEYS`, which never contained it | The validator and the renderer both handle the key; the merge cannot deliver it. Latent — no deck has a second game by one deck on one night yet |
| ~~D3~~ | ~~`stations_for_deck` is built, tested, and consumed by nothing~~ **DELETED 2026-09-09** with the register that needed it | `captains_log.py:254`, `STATION_ROLES`, `UNSTATIONED_ROLES` | Its docstring promises "the validator holds it to this roster" and no code calls it. The equivalent guard IS implemented for `validate_debrief`. Dead weight or an unfinished check — decide which |
| D4 | **Two POH sections are registered with no renderer** | `poh_spec.SECTIONS` declares **TEN** (§0–§9); `poh.RENDERERS` holds **seven**, and §0 front-matter renders through the page shell, so **eight of ten** appear | §8 `matchups` and §9 `appendices` have never rendered on any deck since the book shipped 2026-09-02, and `validate-poh` has printed `NOTE section(s) not rendered: 8, 9` on every deck ever since. "Seven of nine" was wrong in both numbers. **It is a RENDERER gap, not a data gap** — edgar has a full `matchups` key in `manual_prose.json` and eleven passing stacks, and still stops at §7; that distinction cost an hour on 2026-09-21 when the missing pages were read as missing prose and agent passes were nearly spent refreshing copy that would render nowhere |
| D5 | **`compose` computes the log-to-version join and discards it** | `deck_info.compose` calls `deck_versions.report(slug)` and keeps only `current_version` and a count | Per-version records exist inside that call. `info.record` is a flat all-time roll-up, so the dossier cannot say "3–1 on the current list" without recomputing |
| D6 | **Game-derived prose renders with no version predicate** | `deck-view.js logPanel`, and `build_page.render_debrief` in the frozen renderer | Every night renders regardless of which list it was played on, while versions, diagnoses, sims and experiments all carry staleness. The join (`versionOfSha`) is right there and unused |
| D7 | **`agent_cache._SHA_MEMO` never evicts** | `agent_cache.py:64` — keyed `(path, mtime, size)`, `if key not in _SHA_MEMO` | Fine in a CLI process that exits in seconds; an unbounded leak in `manamap serve`, which Sven now keeps alive for days. Every other memo in the repo replaces on signature change |
| ~~D8~~ | ~~CLAUDE.md's deck-page section order is stale~~ **FIXED 2026-09-09** | `CLAUDE.md` ~line 155 | It listed "case file / log / next / status …" against a page that had been reorganised into the nine-section dossier. Now names the nine and notes that a test locks the JS to `page_spec` — the prose was the only unlocked copy |

## The thing that worked: the paper lock refused to lie

Worth recording beside D1a, because it is the system behaving correctly under a
condition nobody designed for.

gishath's tracked decklist does not match the deck the pilot owns. Twenty-odd
artifacts are computed from it. And yet nothing in the repo *claims* the two
match — because `deck_versions.paper` is absent on gishath, and the paper lock
is the ONE assertion that the tracked 99 is the cardboard 99. Nobody had made
it, so nothing was lying.

`docs/data-artifacts.md` says why the lock exists: it "is a fact about cardboard"
that "no artifact can derive". This is the case that proves it. The lock is not
a convenience for the rack; it is the only place the repo can be wrong about
reality, which is exactly why it is authored and why `deck-state` withdraws it
rather than letting it drift.

What DID assert something false was the log entry's `decklist_sha256`, stamped
by a backfill — see D1a. The difference between the two is instructive: the lock
is a claim a human makes on purpose, and the sha is a claim a command makes on
that human's behalf, silently, at the wrong moment.

## O2 — A pod deck's best cards are graded at one-third, and no axis grades them at all

Found on zur-enchantress 2026-09-09, while trying to settle its list.

**The under-credit.** `goldfish` tracks ONE opponent at 40 life, and says so at
`goldfish.py:1123`: *"'target opponent loses that much' credits the full amount
and 'each opponent loses N' credits N — the other seats' losses are real but do
not help kill the one being tracked."* That is honest and deliberate. The
consequence for CARD CHOICE is not recorded anywhere, and it is sharp:

| clause | model credits | worth at the pilot's four-player table |
|---|---|---|
| `target opponent loses that much life` | full | full, against ONE seat |
| `each opponent loses N life` | N | **3N** |

zur-enchantress runs **eight** each-opponent drain sources (Balemurk Leech,
Bastion of Remembrance, Grim Guardian, Marauding Blight-Priest, Northern Air
Temple, Sanctum of Stone Fangs, The Meathook Massacre, Underworld Coinsmith)
against **two** target-opponent ones (Vito, Enduring Tenacity). The deck is
already four-fifths built on the kind that scales to a pod, and every one of
those eight is graded at a third of its table value. A branch that trades an
each-opponent source for anything else measures as neutral-or-better when it is
strictly worse at the table — `oil-v1` does exactly that, cutting Sanctum of
Stone Fangs and Northern Air Temple for mana rocks.

This is the same SHAPE as the bug the drain pillar already fixed once. The
comment at `goldfish.py:1118` records it: `kill_by_turn_rate` was combat-only,
so *"any change trading a body for a drain effect could ONLY ever measure as a
loss"*. The pillar is read now; its POD SCALING still is not.

**Not proposing a 3x multiplier.** An authored constant applied to a headline is
exactly what the engine-lift deletion was about. The honest fix is to REPORT the
split — how much of `mean_cumulative_drain_by_turn` comes from each-opponent
sources — and let the pilot read the pair, the same way `commander_ability_band`
reports a ceiling and a floor rather than picking one.

**And the corpus says there is no card-shaped way out.** A sweep of all 34,900
oracle cards for `each opponent loses that much life` returns THREE: Vizkopa
Guildmage, Caustic Bronco, Lulu. Only the Guildmage ties it to lifegain, and it
is an ACTIVATION — which `simulate` shows this deck's pilot never presses
(median 0 activations across 60 games). Sanguine Bond, Vito and Defiant
Bloodlord all read `target opponent`. Full-amount table-wide lifegain drain
essentially does not exist, so a pod drain plan is built by COUNTING
each-opponent triggers, not by finding a better payoff.

## O3 — The deck's own win condition has no objective axis

`candidates.OBJECTIVE_AXES` offers 13 axes: combat (`kill_by_*`, `damage_8`,
`board_power_6`), mana, steam and treasure. **There is no drain axis**, and
`diagnostic.run()`'s `output` block carries no drain series for one to read.

So on a deck whose drain is 16.77 of ~37 damage by turn ten and fires in 84.7%
of games, every branch objective has to be stated in combat. Twenty-two branches
on zur-enchantress were, and `simulate` says that deck deals a MEDIAN OF 2
combat damage and has never once reached 21 commander damage in 39 games.

`candidates.py` already argues the case for adding this, for `extra_cards_8`:
*"a correlated measure is a perfectly good thing to aim at even when it is a
useless thing to sort by... the pilot says 'draw more cards' and means it."* The
pilot says "drain the table" and means it. An axis is AIMABLE without being
RANKABLE, and this one would live in `OBJECTIVE_AXES` alone.

`kill_by_turn` DOES count drain (`opponent_life` is one pool that both `dealt`
and `drained` reduce), so the newer branches were graded fairly — but on a
BINARY at a fixed turn, for a deck that survives to global turn 34. `damage_8`
is combat-only and blind to 45% of this deck's output.

## O4 — `net_change.json` records neither the model nor the champion it measured against

It stamps the branch's own `decklist_sha256` and nothing else. It does NOT stamp
`meta.model_version`, which `goldfish.model_version()` computes and
`model_staleness.py` exists to check, and it does NOT stamp the champion's sha.

Twenty-two of them exist on zur-enchantress, written across three days during
which the goldfish model demonstrably changed. Their champion columns are
byte-identical, which is good evidence they were all re-measured against the
same list — but NOTHING IN THE FILE SAYS SO, and the next reader has to infer it
from a coincidence. Two lines in `net_change.build` close it.

## O5 — A colour's target can jump 14 sources because a rounding boundary moved

`manabase.effective_pips` returns "the heaviest pip requirement that at least
`PIP_WEIGHT_QUORUM` of that colour's cards share", with the threshold computed as
`ceil(n * 0.2)`. That makes the reported target a STEP FUNCTION of the number of
cards in a colour, and the step can be crossed by a card that has nothing to do
with the requirement.

Measured on zur-enchantress/drain-v1, 2026-09-09. The branch cuts The Spirit
Oasis — a `{2}{U}` Shrine, one blue pip — and touches no other blue card:

```
                blue cards   threshold      3 UU cards meet it?   effective_pips
champion            16       ceil(16*.2)=4          no                  1
drain-v1            15       ceil(15*.2)=3          yes                 2
```

Reported consequence: U target `22 -> 36`, gap `-1 -> -15`, on-curve
`0.895 -> 0.610`, and `net-change` printed **"colour sources went backwards"**
as a paid cost of the branch.

**Nothing about blue castability changed.** The same three UU cards —
Counterspell, Muddle the Mixture, Enduring Curiosity — sit in both lists and are
exactly as hard to cast. The branch's 0.610 is the HONEST figure for them; the
champion's 0.895 was reporting the single-pip case on a deck that contains UU
cards, and has been doing so all along.

Two separable problems, and the second is the worse one:

1. A branch can be charged for a regression it did not cause, and a pilot reading
   "colour sources went backwards, U gap -1 -> -15" has no way to tell this from
   a real one. `net-change` should say WHICH card set `effective_pips` and note
   when the quorum threshold moved between the two arms.
2. A deck sitting just under the quorum reports the easy colour requirement and
   hides the hard one. zur-enchantress has been reporting blue at 0.895 while its
   counterspells sit at 0.610. That is not a branch problem at all — it is the
   CHAMPION's figure being optimistic, found only because a branch tipped it.

The quorum itself is defensible: one `{U}{U}` card in a 40-card blue deck should
not size the whole manabase. A step function with no reported provenance is the
part that is not.

## O6 — Two thirds of a deck's card draw can be invisible to the metric that grades card draw

zur-enchantress, 2026-09-09. `goldfish_metrics.meta.card_advantage` reads:

```
cards_that_draw: 12   modelled: 4
not modelled:  Black Market Connections · Dawn of Hope · Enduring Curiosity
               Lunar Convocation · Mesa Enchantress · Mystic Remora
               Rhystic Study · Tocasia's Welcome
```

The file states this honestly and names the cards, which is the repo working. What
is NOT stated anywhere is the consequence: `extra_cards_8` — a legal branch
objective, and a row `net-change` prints as a paid cost — is computed over the
four cards the model can price, on a deck whose card advantage is twelve cards
deep. Three branch iterations were partly steered by that row before anyone
checked what it covered, and the two conclusions drawn from it were both wrong:

- drain-v3/v4 "lost draw by cutting token-makers, because Enduring Curiosity
  draws on combat damage". Enduring Curiosity is UNMODELLED. Whatever moved that
  row, it was not that.
- drain-v5 added Mesa Enchantress — the single most-included card in the meta at
  72.6% — to fix the draw hole, and the row went DOWN. Mesa Enchantress is
  UNMODELLED. The add was worth zero to the measurement by construction.

**The fix is not to model everything.** It is that a row whose coverage is 4 of
12 must say so where it is printed. `net-change` already carries `limits` and
`blind_spots`; the per-deck coverage fraction belongs in them, and `deck-branch
new` should refuse — or at least warn on — an `extra_cards_8` objective on a deck
whose modelled fraction is below some line. The same question should be asked of
every other axis: `damage_8` is combat-only on a deck that drains, which is the
same defect in a different channel (O3).

### FIXED 2026-09-09 — and the narrow diagnosis was wrong

The first theory was that Mesa Enchantress's "you may draw a card" tripped
`_DRAW_CONDITIONAL_RE`, which rejects "you may". **Deleting "you may" from the
oracle text and re-profiling still read nothing.** There was no cast-trigger
channel in the file at all: not the ETB pattern (needs "enters"), not the
recurring one (needs an upkeep), not the arrival one (needs "you control …
enters"), and not the spell one, which only runs on instants and sorceries.

`_CAST_DRAW_RE` now reads the family. Sweep: 68 cards, and **only 23 are
modelled** — the four gates whose spells are permanents, so this model sees
every cast of them. `(any)`, `noncreature` and `instant or sorcery` are refused
DESPITE trivial regexes, because this model casts few instants and would
under-report those engines by an unknown amount; `legendary`, `historic` and
twelve tribal or mechanic gates are refused because a type line does not settle
them. An absent figure beats a wrong one.

`cast_draw` went into the casting predicate in the same commit, which is the
rule this file has broken six times.

Fleet impact, measured rather than assumed: **zero live decks.** The only card
the tracked fleet held was Beast Whisperer in radagast, which is `broken-down`
and correctly not measured — the first sweep counted it anyway, which is its own
small lesson about using `common.deck_is_apart`.

### The narrower thing underneath it

Mesa Enchantress is unmodelled for one clause: it says "you may draw a card", and
`_DRAW_CONDITIONAL_RE` treats a bare "you may" identically to "you may pay {2}.
If you do, draw a card". A corpus sweep of `you may draw`:

```
 80 cards  the draw is FREE and unconditional in its own sentence
 60 cards  the same sentence carries a real cost or gate — correctly unmodelled
```

So the rule is over-conservative on 80 cards. Most of those 80 trigger on things
this model does not simulate anyway (combat damage, being targeted, blocking,
creatures dying), so widening it would change little — EXCEPT where the trigger
IS simulated, and "whenever you cast an enchantment spell" on a 44-enchantment
deck is exactly that case. NOT changed here: it is a real widening, it needs its
own measurement across the fleet, and this session had already changed the model
once.

## O7 — An aborted run is returned as a run

`forge.list_runs` globs `sim/*.json` and applies **no guard at all**. A record
whose `nonzero_exit_jobs` is 4 and whose `games_completed` is 77 of a requested
120 comes back looking exactly like a finished one, and consumers that read
`summary.win_rate` — which IS populated on a partial — get a figure with no
signal that the run was killed.

Found 2026-09-09 by killing a 120-game run on zur-enchantress@drain-v2 at 77
games. Two things saved it from being quoted:

- The **writer is honest**. It set `games_requested` 120 against
  `games_completed` 77, `nonzero_exit_jobs` 4, `summary.games` null, and
  crucially `win_rate_ci95` **null** — it declined to compute an interval on a
  partial sample, which is the right call and the reason a careful reader would
  notice.
- The record was quarantined by hand before anything read it.

Neither is a control. The gap is in the READER:

1. `list_runs` should carry a `complete` flag, or filter, or at minimum surface
   `games_completed`/`games_requested` to every caller.
2. `summary.win_rate` on an incomplete run is the same defect as
   `win_rate_ci95` would have been, one field over. If the interval is withheld
   for being unsound, the rate it centres on is unsound too. **Absent means
   absent** — that rule already exists in this repo and this field breaks it.
3. **The run id lies.** The filename says `n120` and the run is 77 games. A run
   id is the most-quoted string in this subsystem — it appears in every report,
   every `--analyze` invocation and every commit message — and this one is false
   on its face. The id is fixed at launch from the REQUESTED n, so any killed or
   crashed run inherits a wrong name.

The partial figures are preserved in the quarantine's `WHY.md` rather than
deleted, with the reason they are not a result: the convergence trace fell
monotonically from 0.250 at n=20 to 0.134 at n=67, so stopping early would have
flattered the deck by seven points. That is optional stopping, and it is the
failure mode `sim-progress` was written to refuse.

## O1 — You are paying for games that produce no decision (and the clock experiment has already run)

Regenerating every dossier surfaced the clock-out rate, which nothing had ever
put in one place. **A clock-out has no winner and is excluded from the rate,
correctly. It is not excluded from the wall clock.**

That matters wherever a sample size is quoted: the power arithmetic says ~329
games per arm to resolve a 0.10 difference, and at a 10% clock-out rate that is
~366 games to PLAY. Any estimate that multiplies decided games by the per-game
clock is short by that fraction.

### The comparison this section said nobody had run

*Added 2026-09-21.* This closed with *"raising `-c` might convert clock-outs into
decisions, or might just make each one cost longer. Nobody has run that
comparison, and it is cheap."* **It has been run** — not deliberately, but
`SIM_GAME_CLOCK_SECONDS` moved 300 → 600 and the fleet now holds runs at both,
so the answer is sitting in the tracked records and costs nothing to read:

| clock | clock-outs | games | rate |
|---|---:|---:|---:|
| `-c300` | 118 | 889 | **13.3%** |
| `-c600` | 119 | 1,232 | **9.7%** |

Per deck, where both exist:

| deck | `-c300` | `-c600` |
|---|---:|---:|
| ur-dragon | 14% | **5%** |
| goblin-storm | 10% | **7%** |
| edgar-vampires | 15% | **11%** |
| heliod | 15% | 13% |
| gishath | 10% | 10% |

**Doubling the clock roughly halves the clock-out rate on the decks where it
moves at all, and never raises it.** Read it as evidence, not as the
experiment: these are not seed-matched arms, the `-c300` and `-c600` runs are
against different pods and different list versions, and gishath's 20 games at
300 cannot say anything. What it does settle is the direction — the "or might
just make each one cost longer" branch is not what the data looks like — and it
says the controlled version is worth running rather than speculated about.

**The lesson is the one this page keeps relearning.** A question filed as "cheap,
and nobody has run it" sat open for twelve days while the answer accumulated in
`sim/` as a side effect of an unrelated config change. Nothing joins a filed
question to the data that would close it; §8's sixty undispatched open questions
are the same gap at a different scale.

## What is NOT on this page

- **`make test-browser`** is a local pre-push gate and is deliberately outside
  CI. It is not covered by the counts here.
- **hapatra and yawgmoth-swarm** are archived as broken down for parts
  (2026-09-08). Their artifacts sit at a model version four changes old, and
  that is correct — regenerating a deck that no longer physically exists
  manufactures figures about a pile of cards. `regen` skips them by design.
- **radagast and sisay are NOT live** — radagast is `broken-down` and sisay is
  `retired`, which §5 of this same page had already corrected on 2026-09-08
  while this bullet went on saying otherwise. One file contradicting itself
  thirteen days apart is the clearest argument there is for re-deriving a fleet
  table rather than editing it.

  The half that IS still true: `regen --slug` reports "nothing to regenerate —
  no tracked artifacts matched" for both. Their `goldfish_metrics.json` is
  tracked but no stage claims them, so a model change leaves them silently
  stale. No test fails on it. Same shape as the gap `regen.BOOTSTRAP` was
  written for, one rung down — but on two archived decks, so it is worth
  closing for the NEXT deck that lands in this state rather than for these.
