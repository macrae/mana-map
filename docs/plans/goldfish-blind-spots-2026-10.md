# Goldfish blind spots — a two-session plan (proposed 2026-10-11)

*Status: PROPOSED, not started. Written from the sharknado `torbran-v1` tune-up, where the
goldfish under-read most of a 17-swap package. Nothing here is implemented; every line
number is as of `fb628829` and will drift.*

## Why

The `torbran-v1` screen (sharknado, 17 swaps staged) read **−6.4 damage @T10 and −4.7 pts
killed-by-T6** against the pre-package branch. Measured swap by swap (`try --each`, reverted
on the branch), most of that is the model, not the cards:

| What the model does today | Card(s) | Effect on the screen |
|---|---|---|
| Reads only the body, so the +2 damage is missing (`state: seen`, so the screen does not flag it) | Torbran, Thane of Red Fell (a 2/4) | its whole value is missing |
| Reads only the body, so the +1 counter is missing | Lae'zel, Vlaakith's Champion (a 3/3) | its whole value is missing |
| DARK: the life-gain trigger is not parsed, the drain channel is off, and Shabraz's own life gain is not read | Archangel of Thune | −2.5 vs Chasm Skulker |
| INVISIBLE: escalate wheel unparsed | Collective Defiance | rows silent |
| Refused on purpose (paid trigger) | Mind's Eye | rows silent |
| Never cast (opponent tax unmodelled) | Rhystic Study | rows silent |
| **BUG: free 8/8 on cast** | Ominous Seas (and any "Remove N counters: create …") | Seas reads +1.2 damage it has not earned |
| Mode not read | Boros Charm's 4 to a player, Fateful Showdown's hand-size damage | rows understate |
| Out of scope by design (no opponent removal) | Campfire, Command Beacon, Boros Charm's indestructible, Swords, counters | correctly zero; Forge and scenario slices answer these |

Session 1 fixes what is cheap and matters most (one fleet-wide bug and three multipliers).
Session 2 adds two new kinds of event, life gain and modal spells.

## Rules every change here obeys (from CLAUDE.md and `docs/gotchas-bench.md`)

1. **A model change makes every derived artifact stale.** `meta.model_version` hashes all
   four goldfish files byte for byte, so each session ends with
   `manamap pilot regen --jobs 8 && make manuals`. Do not edit source while the regen runs.
2. **Widening a pattern needs a corpus sweep in the same commit**: list the cards newly
   matched and newly dropped, and read the extreme tail card by card. Put the sweep's numbers
   in the comment beside the pattern, the way `event_payoffs` and `cast_token_profile` already do.
3. **A card can be read correctly and never cast.** Teach the casting predicate
   (`model_coverage.never_cast` and the loop that casts it) in the SAME commit as the ability.
4. **A flag the model sets is a claim the model must act on** (`tests/test_metric_hygiene.py`).
   A field nothing applies is the silent zero this file exists to stop.
5. **A test that re-derives the rule is testing itself.** Drive `goldfish.classify` and
   `goldfish.run`; prove each test by re-introducing the bug.
6. **An authored rate must name where it came from.** Any new rate goes in `config.py` and is
   printed by `try` (as `TITHE_PAY_RATE` is).
7. **Turning a channel ON for a deck changes its baseline.** A branch may not declare a
   channel its deck does not, so a declaration change goes on the deck itself (the
   sleeved `goldfish_targets.json`), which takes the pilot's OK.

---

## Session 1 — the bug, the multipliers, the cheap reads

Order matters: A first, because it moves every deck that carries such a card.

### A. Free tokens from counter-cost activations (BUG, fleet-wide)

- **Cause.** `goldfish_profiles._inside_activation` (≈ l.2161) decides "is this token bought?"
  with `_ACTIVATION_COST_RE` (≈ l.2157). That regex knows only mana costs (`{…}:`) and loyalty
  costs (`[−N]:`). **"Remove eight foreshadow counters from this enchantment:"** is neither,
  so `combat_profile` (≈ l.3443) credits Ominous Seas' 8/8 Kraken as a token made *on cast*.
- **Fix A1 (the guard).** Widen the cost head to non-mana costs: `Remove … counters …:`,
  `Sacrifice …:`, `Discard …:`, `Exile … from your graveyard:`, `Tap … untapped …:`,
  `Pay N life:`. The colon test stays; only the head widens.
- **Fix A2 (the accrual, optional in S1).** A counter-threshold token: Ominous Seas gains a
  counter per card drawn (the draw loop already counts `drawn_this_turn`). At N counters it
  makes the token and spends N. New profile fields `threshold_token: {per: "draw", n: 8,
  power: 8}`, applied in the event block (≈ `goldfish_turn` l.2723) and gated `model_draw`.
- **Sweep.** Every corpus card whose token sentence now reads as inside an activation. Expect
  the delta to be removals of free bodies only. Any *gained* body is a regression.
- **Tests** (`tests/test_goldfish_cast_token.py`, or a new
  `test_goldfish_activation_costs.py`): Ominous Seas has `token_bodies == 0`; a plain
  "When this enters, create a 2/2" still reads 1. With A2, the Kraken arrives only after 8
  draws in a seeded run. Re-introduce the old regex and watch both fail.
- **Expect.** Sharknado's Ominous Seas loses its free 8/8. Any fleet deck with such a card
  moves; `regen` will show which.

### C. Counter bonus (Lae'zel; also Hardened Scales-class)

- **Read.** A new profile field `counter_bonus` from
  `If you would put one or more counters on a creature … that many plus one` (Lae'zel,
  Hardened Scales scoped to +1/+1, Branching Evolution as ×2, so a multiplier variant).
  Check first that the token doublers (`goldfish_profiles` ≈ l.1909, Doubling Season) do not
  already own counter doubling. If they do, extend that path and do not add a second one.
- **Apply.** At the point where counters are placed, add `+counter_bonus` **per placing
  event**, not per counter:
  - the event block `_ctr` (≈ `goldfish_turn` l.2743):
    `per_draw_counter * drawn` becomes `(per_draw_counter + bonus) * drawn`, because every
    Shabraz draw trigger is its own event. Same for `per_discard_counter`.
  - `spell_counters`, ETB "enters with N counters", and Opal Palace if the model reads it.
- **Cast it.** A static with no body value would be never-cast. Lae'zel is a creature, so it
  is cast anyway; Hardened Scales is not. Add `counter_bonus` to `never_cast`'s
  "cast this" set in the same commit.
- **Tests.** Seeded sharknado: Shabraz's `commander_counters_by_turn` with Lae'zel on board
  is exactly 2× the per-event count. Hardened Scales on an unrelated deck changes nothing
  when no counter event fires.

### B. Damage bonus per instance (Torbran; Artist's Talent level 3 later)

- **Read.** A new profile field `damage_bonus: {amount: 2, scope: "red_source" |
  "noncombat" | "any", to: "opponents"}` from
  `If a red source you control would deal damage to an opponent … that much damage plus 2`.
- **Apply per damage instance, before multipliers.** The player orders replacement effects
  (CR 616.1); adding before multiplying is the better order and the one a pilot picks.
  - **Event damage** (≈ `goldfish_turn` l.2725): each payoff's `per_discard_damage * _disc_t`
    becomes `(per_discard_damage + bonus_if_red(payoff)) * _disc_t`. This needs the payoff's
    **source colour**, so attach `colors` when payoffs are appended (≈ l.841, 1377, 2455).
    Brallin is red, so 1 becomes 3 per discard.
  - **Cast damage** (`per_cast_damage`, Khorvath's Fury, Fateful Showdown once E lands): +2
    per red spell instance.
  - **Combat**: +2 per red creature that connects. This needs colour on `creature_entered`.
    If that is a larger change than the rest, ship event and cast damage first and name
    combat as a stated gap in `MODEL_ASSUMPTIONS`.
- **Approximation to state.** Magmakin's "deals that much damage" is ONE instance per batch,
  not one per card. Today it is read per card, so mark it (`per_discard_batch: True`) or
  state it.
- **Tests.** Brallin + Torbran + one Windfall in a seeded game: event damage that turn =
  3 × cards discarded. Without Torbran = 1 ×. Order test: Torbran plus a ×2 multiplier gives
  (1+2)×2.

### F. Paid opponent-draw triggers (Mind's Eye) and the Rhystic tax

- **F1, Mind's Eye.** Refused today on purpose (`_EVENT_PAID_RE`, ≈ l.324): unbudgeted, it
  would give 7 free cards per wheel. Model it with a budget: our draws off their draws are
  capped by the mana left untapped at end of turn, each costing its `{N}`. Read the cost
  into `per_opponent_draw_our_draw_cost`. Stated approximation: opponents' draws happen on
  their turns, and we pay from lands left untapped on ours.
- **F2, Rhystic Study** ("unless that player pays {1}"). The same shape as Smothering Tithe,
  priced with the SAME authored `TITHE_PAY_RATE`, which `try` already names. Today it is
  never cast.
- **Sweep.** All 31 "whenever an opponent/player draws" cards from the 2026-09-14 sweep
  (comment at ≈ l.304). Make sure no paid trigger reads unbudgeted.
- **Tests.** A seeded wheel with Mind's Eye and 2 mana open yields at most 2 extra draws.
  With 0 open, it yields 0.

### H. Honest coverage: a "partial" state

`model-coverage` and `try` said **seen** for Torbran and Lae'zel when only the body was read.
Add a `partial` state: the card has a replacement or static clause (`would … instead`,
`plus N`, `put that many plus`) that set no profile field. That surfaces exactly this class
before anyone trusts a row. Card-scoped, cheap, and it would have caught Torbran today.
Check it against the whole fleet first ("a validator that fires on correct data is worse
than none").

### Session 1 close

1. `make test` (unit tier), then `make check-deck SLUG=sharknado`.
2. `manamap pilot regen --jobs 8 && make manuals`. Commit the regen as its own commit, as
   `59788763` did.
3. `make prepush` scoped to python.
4. Re-run the sharknado screen and compare against this baseline from 2026-10-11:
   `try sharknado --branch torbran-v1 --each` (reverted), with undo-all at
   **damage@T10 60.30 → 66.66, killed-by-T6 0.188 → 0.235**.
   Expect Torbran and Lae'zel to read positive, Ominous Seas to lose its 8/8, and
   Mind's Eye and Rhystic to show rows.
5. Update `docs/simulation.md`'s channel table and `docs/known-issues.md`.

**Estimate:** about half a day of work plus one fleet regen. A and C are small. B is the
largest because of source colour, and combat may slip to Session 2.

---

## Session 2 — life gain as an event, and modal spells

### D. Life gain as a counted event (Shabraz, Archangel of Thune)

- **Today.** `event_payoffs` says, in so many words, "you gain N life is NOT read: this
  model has no own life total". The drain pillar (`drain_profile`, ≈ l.2682) reads
  gain-to-damage converters, but nothing counts **gain events**. Sharknado declares
  `model_drain: false`.
- **Read.**
  - `per_draw_gain` / `per_discard_gain` on `event_payoffs` (Shabraz: 1 per draw).
  - lifelink connects as one event per creature.
  - a new payoff field `on_gain`, with three shapes:
    - `team_counters: 1` — Archangel of Thune.
    - `team_counters: "amount"`, once a turn — Nykthos Paragon.
    - `draw_paid: {2}` — Dawn of Hope, budgeted the way F1 is.
- **Apply.** A `gain_events_this_turn` count. Archangel adds +1 counter to **each creature
  on the battlefield** per gain event. Feed it through C's `counter_bonus`, so with Lae'zel
  that is +2 per creature per draw. Archive's "gain twice that much" changes the amount,
  not the event count. Say so in the profile docstring.
- **Declaration.** Gate on `model_drain` (or a new `model_lifegain` if drain carries too much
  else). Switching it on for sharknado changes its baseline, so ask the pilot first. Look at
  every other deck the switch would move, heliod above all: it is a life-gain deck and will
  shift the most.
- **Sweep.** "Whenever you gain life" is about 100 corpus cards. Read the tail card by card.
  Re-run the metric independence check (`pytest -m fleet`) if a new row appears.
- **Tests.** Shabraz + Archangel + a 7-card wheel: each creature on board gains 7 counters,
  or 14 with Lae'zel. Heliod's existing drain tests still pass unchanged.

### E. Modal and escalate spells, and damage modes

- **Wheels behind a mode or a target.**
  - "Target player discards all the cards in their hand, then draws that many" → a
    self-wheel (Collective Defiance).
  - "Discard up to two cards, then draw that many" → a rummage (Cathartic Pyre).
  - Hex Magic's exile-and-draw → a Shabraz-only wheel (`wheel_shuffles`-like, no discard).
- **Mode choice.** Choose the modes that score in a goldfish: the wheel plus face damage.
  Escalate pays `{1}` per extra mode when mana allows. State the choice as a policy in
  `MODEL_ASSUMPTIONS`.
- **Damage modes.**
  - Boros Charm: 4 to a player. Mark it a **ceiling**, because the pilot will often hold it
    for indestructible.
  - Fateful Showdown: hand size to a player.
  - Incendiary Command: 4 to a player plus the wheel, and refuse "2 to each creature" (it
    hits our own board).
  - All of these go through B's +2.
- **Sweep.** Modal and escalate red and blue spells. Every newly read wheel needs its
  casting predicate in the same commit (rule 3).
- **Tests.** Collective Defiance on a 5-card hand: 5 discards and 5 draws (pays both
  commanders) plus 3 to the face, or 5 with Torbran.

### G. The rest of the "unmodelled" wheels (optional, card by card)

The Elder Dragon War chapter II, Colossus of the Blood Age's death wheel (moot: nothing dies),
Umaro's random mode, Ajani Unrelenting −2. Low value; do it only for a card in a list.

### Artist's Talent (stretch)

Its level 3 is B's `damage_bonus{scope: "noncombat"}`. The hard part is the **Class
level-up as a mana sink** (`room_profile` handles rooms; Classes need a "buy the next level
when spare mana" loop). Level 1 (loot on a noncreature cast, which pays both commanders) is
cheap and worth reading even without the levels.

### Session 2 close

As Session 1, plus the pilot's OK on sharknado's `model_drain` switch before regen, and a
note in `docs/simulation.md` on what changed for heliod.

**Estimate:** about half a day plus one fleet regen. D is the largest because of the sweep
and the cross-deck effect.

---

## What this will NOT do

- Price protection, recursion or removal (Campfire, Command Beacon, Boros Charm's
  indestructible, Swords). The goldfish has no opponent removal, so those stay at zero by
  design. Forge (`forge-cast-check`, `simulate`) and scenario slices (`scenario-ab`) answer
  them.
- Model opponents refilling off symmetric wheels (Incendiary Command, Day's Undoing, Snort).
  The goldfish has no opponent hands that act. That is a stated limit, not a bug.
- Make a one-card swap's ~1-point move a verdict. The pilot's rule, 2026-10-11: judgment and
  play lead; the goldfish is a guardrail.

## Open questions for the pilot

1. Session 1 in full, or A + C + B (event and cast damage) only, with F and H next?
2. OK to switch sharknado's `model_drain` on in Session 2, knowing its baseline moves?
3. A2 (real foreshadow accrual) now, or just the guard?
