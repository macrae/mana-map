---
name: release-branch
description: Release a sleeved deck's branch end to end — measure it, propose it as a version, merge it, re-place the Deck Context, re-price, commit, tag and lock the version, push the list to DeckCheck and record the rescan, then re-rate the deck and settle its bracket. Use when the pilot says a staged branch is going into the sleeves (on cardboard or on proxies); `/publish-deck` phase 13 is the long form, this is the one sequence with every gate the first two releases missed.
---

# Release a branch

The whole release of one branch on a SLEEVED deck, in the order the gates require.
Written from sharknado's torbran-v1 (v1.3.0, 2026-10-19) after goblin-storm's meaner-v1
(v1.1.0, 2026-10-18) skipped the tag, the paper lock and the rating. Every step names
what it writes; nothing here is optional unless it says so.

**Before anything: the pilot's word.** A sleeved deck's list changes only through a branch,
and the branch merges only when he says the cards are going into the sleeves — cardboard or
printed proxies (`/print-proxies`). Ask which, and whether the unowned cards are ordered:
that becomes the proposal's note and the merge's reason.

## Steps

0. **Preflight (read-only).** `deck-branch <slug> show <branch>`: the OUT list against the
   keep list (`protected.json`; and the pilot's standing rules — never Vish Kal, never
   Marauding Mako); `git status` clean under `data/decks/<slug>`; `deck-info <slug>` for
   anything already red (engine, tutors, prose, build_plan) so it is not blamed on the merge.

1. **Measure.** `net-change <slug> --branch <branch> --write` — the paired goldfish on the
   pre-registered objective plus twelve Holm rows. Report the primary row with its MDE; a
   Forge loss is a warning. `forge-cast-check <slug> --branch <branch> --adds --write` only if
   the harness moved since `cast_proofs.json`. `propose` refuses without `net_change.json`.

2. **Propose as a version.** `deck-branch <slug> propose <branch> --as vX.Y.Z --why "…"
   [--proxy] [--ordered "…"]`. The semver rule (`docs/pilot.md`): PATCH = mana only, MINOR =
   a new capability, MAJOR = a different deck — capability, not card count. `--proxy` counts
   cards sleeved in his OTHER decks as sourced; it does not cover unowned cards.

3. **Merge.** `deck-branch <slug> merge <branch> --write [--proxy] [--force --reason "played
   on printed proxies from <date>; the unowned cards are on the order list"]`. The sourcing
   gate blocks unowned cards; `--force` needs the reason and the ledger keeps it. The merge
   runs the chain itself (fetch-deck → goldfish → mana-analysis → regen → deck-map →
   build-poh → build-index) and prints STALE, INVALID, CONTEXT STALE and the commit line.
   Then **`validate-goldfish-targets <slug>`**: the declaration is hand-written and a swap
   strands names in it — the merge's first goldfish ran with seven stranded names on
   sharknado, which silently deflates a group's rate. Repair it by oracle text (remove the
   cuts from every group, place each add in the group its text earns, multipliers in no
   group), add a dated `_note`, then **`regen --slug <slug> --jobs 4`** so every measurement
   reads the repaired declaration. Since 2026-10-21 regen never touches the MERGED branch
   (or any decided one): its `net_change.json` is the reading the decision rested on, and
   the merge froze the same figures into the ledger (`decisions.jsonl`) before its chain
   ran — quote either, and the commit message. (Before that date regen overwrote the
   merged branch's file with zero deltas; a file dated earlier may read that way, and the
   ledger is the record then.) The old text that follows is kept for those files: the
   branch file after the merge (known-issues §28).

4. **The Deck Context.** Spawn `context-keeper` MODE deck-change with the out/in lists the
   merge printed; `context <slug> --install data/decks/<slug>/.agent-out/context-keeper.md
   --note "deck change: <branch> merged as vX.Y.Z"`; `context <slug> --refresh`;
   `context <slug> --check` says current.

5. **The deck's own artifacts.** `prices <slug> --write` (the merge leaves the old list's
   prices, INVALID); `deck-info <slug> --write`; `build-poh <slug>`; `build-index`.
   `deck-status <slug>` should now list only what was red before plus the agent files the
   merge genuinely stales (engine, tutors, prose) — those wait for `/analyze-engine` and
   pilot-notes, and the commit message names them.

6. **Check, then the merge commit.** `make check-deck SLUG=<slug>` (~1 min).
   `test_versions_json_matches_a_fresh_run` FAILS here by design: the list is uncommitted,
   and the versions ledger is a separate commit. Commit the deck's paths plus
   `data/decks/index.json`, `data/fleet/analytics.json` and `manuals/p/<slug>.html`, with
   the swaps and the net-change primary row in the message. **This commit mints the version.**

7. **Tag and lock.** `deck-version <slug> list --write` (regenerates `versions.json` against
   the new commit; it walks git and can take minutes — run it in the background);
   `deck-version <slug> tag vX.Y.Z --note "…"`; `deck-version <slug> paper --note "proxies in
   sleeves <date>; cards to order"` if the pilot said the list is in the sleeves (proxies
   count: the captain's log stamps tonight's games against it). `deck-version <slug> list`
   must read SLEEVED V<n> [vX.Y.Z]. Commit `deck_versions.json` + `versions.json`.

8. **DeckCheck.** `deck-export <slug> --format moxfield --out <scratch>/<slug>-vX.Y.Z.txt`;
   mark the commanders the way their Bulk Edit expects (`… (Commander)`, `… (Partner)`);
   open `https://deckcheck.co/app/builder/<id>` (the id is in `links.json`) → More Tools →
   Bulk Edit → replace the Mainboard → Apply Changes; confirm "Cards 100". `deck-link
   <slug> deckcheck <url> --note "vX.Y.Z pasted …"`. **The rescan spends his credits — ask
   first**, then Rescan → Continue (leave "Add context for the AI" alone; the saved context
   stays). Wait for "Scan complete" (minutes), then `deckcheck <slug> --write` (it fetches
   their API; the response cache is 15 minutes, so a write inside that window after an
   earlier read returns the old answer — wait it out or delete the cache entry under
   `data/cache/deckcheck/`). The reading is stamped with the CURRENT `cards.json` sha, so
   this step must come after the merge. `deck-info <slug> --write && build-index`; commit
   `deckcheck.json`, `links.json`, `info.json`, `index.json`.

9. **Re-rate.** `fleet-analytics --write && validate-fleet-analytics`; `rating <slug>
   --write` (the judgment drops: written against the old sha); then `/rate-deck <slug>`
   steps 2–5 (rater → merge → validate → skeptic → merge --verifier, ≤3 iterations; a `fail`
   is saved). Settle the bracket — `rating <m> --write && validate-rating <m>` for EVERY
   member, since the group's simulation is joint — `cache-record <slug> --routine
   deck-rating`, `fleet-analytics --write`, `build-index`, `rating --all`. Commit the
   `rating.json` files, `.agent-cache.json`, the fleet artifact and the manifest.

10. **Report and push.** The net-change primary row, the version, the DTI reading
    (old → new), the rating (old → new, with its concordance and what moved), what is still
    stale and who fixes it, and the order list if cards are unowned. `PLAN.md`'s fleet
    section and `NEXT_SESSION.md` get one line each. Push.

## Notes

- A merge that cannot run the Keeper or the rating is not finished; say so rather than
  leaving a stamped list under stale prose and a stale grade.
- Never `cache-record` to green a board; never hand-patch an agent's prose.
- The engine model, tutor guide and prose go stale on every merge and are agent work
  (`/analyze-engine`, pilot-notes) — name them in the merge commit, do not block on them.
