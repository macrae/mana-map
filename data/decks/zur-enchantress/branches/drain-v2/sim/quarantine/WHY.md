# Why these runs are here

A run in `quarantine/` must never be read as a result. `forge.list_runs` globs
`sim/*.json`, so moving a record one directory down is what takes it out of
every consumer at once — the same mechanism the champion's six wrong-commander
runs use in `data/decks/zur-enchantress/sim/quarantine/`.

## giada-angels-vs-baylen-tokens-vs-abaddon-n120-1662420c-s375538188-podExperimental-c600

**Killed by the pilot at 77 of 120 games, 2026-09-09.** Not a failed run — a
deliberately abandoned one.

The record is internally honest and says so in four places: `games_requested`
120 against `games_completed` 77, `nonzero_exit_jobs` 4, `summary.games` null,
and `win_rate_ci95` null — it declined to compute an interval, which is exactly
right.

It is quarantined anyway, for two reasons:

1. **`summary.win_rate` IS populated** — 0.125 from 72 decided games — and
   `forge.list_runs` applies no guard, so any consumer that reads `win_rate`
   without also reading `games_completed` gets a partial figure that looks like
   a finished one.
2. **The FILENAME claims `n120`.** A run id is the most-quoted string in this
   subsystem and this one is false on its face.

The partial reading, recorded here so the compute is not simply lost: 9 wins of
72 decided, and a convergence trace that fell monotonically 0.250 (n=20) ->
0.134 (n=67). giada-angels took 45 of 77 at 0.612, above the 0.597 the pod file
records for that seat. **None of that is a result about drain-v2** — optional
stopping is precisely what makes a killed run unusable, and the decline means
stopping early would have flattered the deck by seven points.
