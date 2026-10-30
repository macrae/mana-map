# Next session (written 2026-10-30, after the CI fix, the worktree prune and the UI walkthrough)

Read CLAUDE.md first. Sean is the pilot. Never push without asking. One heavy job at a time: the
PreToolUse guard refuses a second one, and `make jobs` / `make reap` show and clear leftovers.

## Done 2026-10-30 (all pushed; CI green on 32d49dd6)

- **32d49dd6: the CI browser flake, fixed at the cause.** `idle()` could resolve in the
  middle of the force graph's 520 ms fit transition, because the page's busy probe only
  watched the map camera and the layout ticks. `Force.lastZoom()` now feeds the probe
  ("the graph camera is moving"); the zoom test waits with `graph=True` before and after
  the fit. Busts at v=272 on all seven pages — `tests/test_viz_cache_busts.py` holds all
  seven to one number, and CLAUDE.md's gotcha now says so.
- **The 23 agent worktrees under `.claude/worktrees` (8 GB) are removed** and their
  branches deleted; every patch was already on main. One worktree remains under
  `~/.claude/jobs`, which is Claude Code's own.
- **The UI overhaul is VERIFIED in Chrome by Sean (2026-10-30): "it all looks great".**
  Explore on My decks, the New: FRA pill, Filters and ⚙ Map, region subtitles,
  Windfall's real payoffs, Sharknado in title case, Ask Jarvis styled, the Build tabs,
  the landing page's names. No fix list came out of it.
- **The payoff-tag retrain's mixed read is ACCEPTED by Sean (2026-10-30): "no need to
  retrain". Not a decision to reopen.**
- **The UI overhaul's leftovers are done (2026-10-30, later).** `extract` keeps Scryfall's
  `set_type` (cards.csv is 37 columns; re-extracted from the same dump, every other column
  byte-identical), `sets.json` carries it, and the set lens picks by release class instead
  of a 100-card floor. A neighbourhood takes the first of its tags not already used under
  its parent: numbered L2 names 71/72/72 -> 16/41/16 across the three maps, membership and
  L0/L1 names untouched. The "subtitles from the full card_roles.json" idea is moot — the
  index's roles ARE those roles — and `regions_cardbert.json` had already been re-clustered
  in e01e4d2e. DATA_VERSION 16, busts v=273.
- **The decks (2026-10-30, later).** Sharknado stacks 013 (Needlehead + Gratuitous
  Violence: 14 per opponent) and 014 (Brallin + Needlehead on one wheel: 10 per opponent,
  Brallin 6/6) are checker-PASSED. Q011 answered from the game logs. heliod's engine
  model PASSED its critic (round 3; recorded); ur-dragon's FAILED round 3 (saved, not
  recorded). Two CI browser flakes fixed at the cause (89bce1f3).
- **The browser suite on this Mac is not reliable with Chrome open** (load 13: two full
  runs failed on `Page.goto` and boot timeouts, different tests each time, every one
  passing alone). CI is the proof of a full run; locally, close Chrome first.

## Done 2026-10-29 (pushed through f2b44e85)

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
  - 815e1c2f: the rest of the waits. `mmWhenIdle` moved to `api.js` (every page);
    `idle(page)` in `conftest_viz.py`. 129 more waits converted; full browser suite
    219 s / 223 s, green twice.

## Test state

- Unit 3409 green. Regression green on e01e4d2e. Browser: 397 passed + 1 xfail.
- `test_small_selection_uses_the_accordion` is xfail: intermittent, cause unknown. It passes
  alone; in full runs it has read exactly 207 px. It is not a loading problem.
- goblin-storm's lifted stack 009 drifts (one Abundant Growth). This predates this session
  and isn't gated.

## Next, in order

1. DECISIONS FOR SEAN, from today's deck work:
   - Q011 is TESTED (`manamap pilot queue show Q011`): stage, drop, watch or more.
   - heliod `proposed_goldfish_edits[6]`: switch on `model_treasures` and `model_discard`
     (seven cards dark on them, five in conversion). Re-baselines every open branch; the
     Tithe and Rhystic figures would rest on the authored TITHE_PAY_RATE 0.5.
   - ur-dragon: drop Solphim, Mayhem Dominus (not a Dragon) from two Dragon legs of
     goldfish_targets.json (the engineer's proposed edits).
2. Both engine models PASS (2026-10-30, later; recorded). heliod passed its critic in round 5:
   stages[1] now lists every card that gives an opponent a card on YOUR turn, Faerie Mastermind
   and Arcane Denial included (round 4 missed Mastermind). ur-dragon passed in round 4:
   Atsushi is counted beside Gadrak as a death-to-resource card. One non-blocking finding
   remains: Crux's non-Dragon mode also kills Solphim, and Faeburrow Elder can survive
   Desolation, and absent_stages[0].why (2,576 chars) wants a trim. Its open question
   (Sneak Attack's sacrifice ordered before Gadrak's Treasure, CR 603.3b) is still ready for
   /resolve-stack.
3. Remaining test speed:
   - What stays timed is deliberate: negative checks, hover delays, poll periods, and
     waits under 200 ms (cheaper than `idle`'s 150 ms quiet window).
   - Five behaviour tests still need timed waits (search/slug debounces, the
     first-expansion fit, the follow camera). Each needs its own condition.
   - Regression's next heaviest: `test_embedding_pool_eval` (113 s),
     `test_pilot_sim_findings` (98 s), `test_sim_experiment` (89 s), `test_pilot_deck_info`
     (86 s), `test_sim_seat_audit` (81 s).
   - Measure with the cache ON. `-p no:cacheprovider` turns off the suite's `unchanged`
     cache and inflates every figure.

## Standing rules

- Never start Forge, regen, a retrain or any long job unasked. Say what a "go ahead" costs
  before starting it.
- Never cut the pilot's keeps. Report the whole profile by axis.
- List cards non-lands then lands, each alphabetical. Cache-bust every page together.
- Never edit source while a suite or job runs. Commit to main; push only when asked.
