---
name: rate-deck
description: Rate one Commander deck 0-100 within its bracket — the bench's scorecard of measured components plus one agent judgment, verified by an adversarial skeptic — producing data/decks/<slug>/rating.json (★ coaching over ◆ inputs). Use when the pilot asks how strong a deck is, or to refresh a stale rating.
---

# Rate a deck (the rating loop)

Writes `data/decks/<slug>/rating.json`: a score, a five-band grade and a rank **within
the deck's bracket**, labelled ★ coaching. The bench computes every measured component
(`manamap pilot rating`); the `deck-rater` adds the one judgment component; the
`rating-skeptic` attacks it; `validate-rating` holds the whole thing to the record.
Reference: `docs/pilot.md` → *The rating*.

**The scorecard always runs and always answers on its own.** `manamap pilot rating
<slug> --write` costs nothing but a few seconds and names every component it could not
score. If you skip the agents you still have an honest skeleton. Never present an agent
pass as the thing that produced the numbers.

**The score never gates anything.** The floor (`bracket_report.json`) is the bracket gate;
the rating is a labelled judgment with its n and no significance claim.

## Loop (max RATING_MAX_ITERATIONS = 3, from config.py)

0. **The fleet, then the skeleton** — free:

   ```bash
   .venv/bin/manamap pilot validate-fleet-analytics        # WARN "stale" → rebuild it:
   .venv/bin/manamap pilot fleet-analytics --write
   .venv/bin/manamap pilot rating <slug> --write           # the skeleton; carries a current judgment forward
   ```

   Read `withheld[]` out loud before spawning anything. What the components could not
   score is the brief: it is what the rater is filling in for.

1. **Freshness** — `manamap pilot deck-audit <slug>`'s `freshness` block; if
   `benchmark.json` or `goldfish_metrics.json` is not current, re-run them before the
   agents (`benchmark <slug> --write`, `goldfish <slug>`, then step 0 again).

2. **Cache gate**: `.venv/bin/manamap pilot cache-status <slug> --routine deck-rating`
   - **exit 0** (`HIT`/`EDITED`) — the judgment is current for this list, this fleet
     artifact and this scorecard; report `rating.json` and **do not spawn**. `--force`
     to override.
   - **exit 1** — run the loop.
   - **exit 2** — an input is missing (usually `rating.json` itself: run step 0).

3. **Rate**: spawn `deck-rater` with **`MODE: rate`** and the slug (by its own
   `subagent_type`, so the band times it). It writes `.agent-out/deck-rater.json`. Then:

   ```bash
   .venv/bin/manamap pilot merge-rating <slug>              # recomputes the skeleton, validates, writes
   .venv/bin/manamap pilot validate-rating <slug>
   ```

   A refused merge prints every error. Re-spawn the rater with them — **never hand-patch
   the hand-off** to get it through.

4. **Verify**: spawn `rating-skeptic` with the slug. It writes
   `.agent-out/rating-skeptic.json`. Then `merge-rating <slug> --verifier`. If the
   verdict is `fail` and `iterations < 3`, re-spawn `deck-rater` with **`MODE: revise`**
   and the findings, and repeat from the merge. A `fail` at three is saved as a `fail`
   and reported plainly.

5. **Record**, last, only after `validate-rating` passes:
   `.venv/bin/manamap pilot cache-record <slug> --routine deck-rating`.

6. **The fleet page**: the leaderboard reads each `rating.json` directly, and the group's
   ranks move when a member's score lands:

   ```bash
   .venv/bin/manamap pilot fleet-analytics --write
   .venv/bin/manamap pilot build-index
   ```

7. **Report**: the grade with its group ("B in Bracket 3, 2 of 5"), the score with its
   interval, the no-significance line, every withheld component and why, the skeptic's
   verdict and iteration count, and the open questions as dispatched work
   (`resolve-stack` → `/resolve-stack`; `research-strategy` → price it first;
   `goldfish` → a targets edit; `diagnose` → `/diagnose-deck`; `rate` → this loop again).

## Notes

- **Tier discipline.** Every component is ◆ and recomputed by the validator; the
  judgment and the grade are ★. The page draws the badge.
- **Rank is relative to the group's current scores**, so rating one deck can move its
  neighbours' positions. `/rate-fleet` refreshes every member after a pass.
- **A decklist edit, a fleet rebuild or a scorecard edit invalidates the judgment**
  (`judgment.against`); `rating --write` drops it into `withheld[]` with the reason and
  the cache MISSes. That is the loop asking to be run again, not a defect.
