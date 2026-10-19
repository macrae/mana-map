---
name: rate-fleet
description: Rate every live Commander deck within its bracket by running /rate-deck's loop deck by deck, then refresh every rank, the fleet artifact and the manifest, and print the leaderboard (`manamap pilot rating --all`). Use when the pilot asks to rank the fleet, or after a scorecard change.
---

# Rate the fleet

`/rate-deck` for every live Commander deck, in slug order, then one pass to settle the
ranks. The leaderboard is `manamap pilot rating --all`; the fleet page draws it.

## Steps

0. **The fleet artifact first**, so every skeleton reads the same bands:

   ```bash
   .venv/bin/manamap pilot fleet-analytics --write
   .venv/bin/manamap pilot validate-fleet-analytics
   ```

1. **The decks**: every deck the fleet artifact lists with `apart: false`, `format`
   null and a bracket group other than `unbracketed`, in slug order. For each, run
   `/rate-deck <slug>` steps 0–5 (the cache gate skips the ones whose judgment is
   current). Do not run step 6 per deck; the ranks settle below.

2. **Settle the ranks.** A member's score moves its neighbours' positions, so once every
   judgment has landed, rewrite each skeleton (judgments carry forward) and re-validate:

   ```bash
   for slug in <the decks above>; do
     .venv/bin/manamap pilot rating $slug --write && .venv/bin/manamap pilot validate-rating $slug
   done
   ```

3. **The surfaces**:

   ```bash
   .venv/bin/manamap pilot fleet-analytics --write
   .venv/bin/manamap pilot build-index
   .venv/bin/manamap pilot rating --all
   .venv/bin/manamap pilot rating --calibrate        # NOT ANSWERABLE below calibrate.MIN_DECKS; say so
   ```

4. **Report**: the leaderboard by bracket as `rating --all` prints it (grade, score,
   interval, skeleton or judged, withheld), the calibration verdict verbatim, and the
   decks whose skeptic verdict is `fail`. Commit `data/decks/*/rating.json`,
   `data/decks/*/.agent-cache.json`, `data/fleet/analytics.json` and
   `data/decks/index.json` together; the message carries the leaderboard.

## Notes

- Every rating is ★ coaching with its n; a bracket of three is three numbers.
- Forge's component is weight zero by the pilot's rule; it is never a reason to run a
  simulation here.
