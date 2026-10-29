# Next session (written 2026-10-29, end of the cleanup + test-speed session)

Read CLAUDE.md first. Sean is the pilot. Never push without asking. One heavy job at a time: the
PreToolUse guard refuses a second one, and `make jobs` / `make reap` show and clear leftovers.

## Done 2026-10-29 (all on main, NOT PUSHED)

- **38562067, edf4a3cd: the job guard.** `src/manamap/jobs.py` is a PreToolUse hook on Bash
  (`.claude/settings.json`). It refuses a suite, retrain, regen or Forge run beside another
  one, including from subagents. `make jobs` lists what is running; `make reap` kills
  orphans. `--help` passes. The cause was the last session's 8 h retrain plus stacked
  prepush runs (load 47).
- **4ebf1afa: names and UI fixes.**
  - Every deck is named in `issue.json` (Sharknado, Goblin Storm, Edgar Vampires,
    Ur-Dragon…).
  - All-caps names draw in title case.
  - Ask Jarvis is styled before the first click.
- **e01e4d2e: the wheel payoff tags.**
  - New tags `draw_payoff` (135 cards) and `discard_payoff` (81), with two synergy rules.
    Windfall's partners are now 7/10 real payoffs.
  - The ability model was retrained. Function-space change, MIXED: r@10 0.258 → 0.262,
    median rank 66 → 76 (worse).
  - Region names carried across by card membership; 8 new ability-map regions named.
  - DATA_VERSION 15.
- **Test speed:**
  - 704a2c53: handbook phantom-card check 213 s → 5 s; the simulation validator's board
    replay is linear now (sharknado 171 s → 24 s, every tracked run reproduces exactly).
  - d59ef3ec: net-change parses each record once, and the bootstrap draws the same random
    stream faster (70 s → 43 s).
  - 37f80491: a source-only prepush no longer re-runs the unit tier as an isolation proof.
  - 3edcf911: `MM.whenIdle()` replaces 151 fixed sleeps in `test_viz_behaviour.py`
    (246 s → ~183 s, green twice). The full browser suite took 270 s.

## Test state

- Unit 3409 green. Regression green on e01e4d2e. Browser: 397 passed + 1 xfail.
- `test_small_selection_uses_the_accordion` is xfail: intermittent, cause unknown. It passes
  alone; in full runs it has read exactly 207 px. It is not a loading problem.
- goblin-storm's lifted stack 009 drifts (one Abundant Growth). This predates this session
  and isn't gated.

## Next, in order

1. Ask Sean before pushing the commits above.
2. Remaining test speed:
   - Convert the other 22 browser-file sleeps, the 67 short in-page ones and the 93 Python
     `wait_for_timeout` calls. Some are deliberate: debounces, or proving that nothing
     happens.
   - Five behaviour tests still need timed waits (search/slug debounces, the
     first-expansion fit, the follow camera). Each needs its own condition.
   - Regression's next heaviest: `test_embedding_pool_eval` (113 s),
     `test_pilot_sim_findings` (98 s), `test_sim_experiment` (89 s), `test_pilot_deck_info`
     (86 s), `test_sim_seat_audit` (81 s).
   - Measure with the cache ON. `-p no:cacheprovider` turns off the suite's `unchanged`
     cache and inflates every figure.
3. Still open from the UI sprint:
   - `sets.json` set_type for the set lens.
   - Region subtitle data work.
   - Prove the Sharknado stacks (Gratuitous Violence + Needlehead; Brallin + Needlehead).
   - The heliod and ur-dragon engine critics fail.

## Standing rules

- Never start Forge, regen, a retrain or any long job unasked. Say what a "go ahead" costs
  before starting it.
- Never cut the pilot's keeps. Report the whole profile by axis.
- List cards non-lands then lands, each alphabetical. Cache-bust every page together.
- Never edit source while a suite or job runs. Commit to main; push only when asked.
