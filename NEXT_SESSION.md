# Next session (written 2026-10-29, end of the Atlas UI overhaul)

Continue the Mana Map Atlas UI overhaul (the pilot's own UI review, 2026-10-28). Read
CLAUDE.md, docs/gotchas-viz.md and docs/viz.md first. Sean is the pilot; never push
without asking.

## Done (all on main)

- **67888828 P0.**
  - Discover: the relations (Similar / Synergy / Outclassed) sit under the card. Combos
    fold to 3, sorted your decks → your colours → the rest. An empty relation reads
    "none" with the reason.
  - Build: the review grid opens only when cards are left to review, otherwise it's a
    chip. The header badges come from `DeckFacts` in api.js: version, rung, bracket,
    grade, W-L, DTI.
  - The stale "Loading embeddings…" status is fixed.
- **8038cf94:** `Force.refit()`, so an automatic refit never claims the camera.
- **3af2bff5 P1 Build.**
  - Tabs: Change / Candidates / Deck / Numbers. Change is the default for a sleeved deck,
    Deck otherwise. `?tab=` deep-links, and the tab is kept per deck in sessionStorage.
  - Goldfish rows are grouped Better / Worse / No call; there is no single CALL line.
  - Live branches are listed via `window.BranchRows`.
  - The settings popover holds Graph/Map, the Show toggles and Fit.
  - "Ask the bench" was removed.
  - A clicked card opens as an overlay; Escape peels it first.
  - A branch's objective is optional (serve, `deck-branch new`, `validate_branch`, docs).
- **adb7cfb3 P2.**
  - "Ask Jarvis" is on the card panel, the Change tray and branch rows (Build and
    branch.html). It copies `/jarvis <q>` and posts ONE pending `ask` to page-state.
    `manamap pilot page-state` prints it; `--clear-ask` removes it. The Jarvis skill
    answers it, then clears it.
  - Load speed: `loadDeck` fetches in parallel (Build `?deck=sharknado` 10.1 s → 0.62 s
    software-rendered, 860 → 746 ms GPU). The embeddings prefetch on `?deck=` and on the
    picker. serve sends `Cache-Control: no-cache`. Figures are in docs/viz.md.
- **2eec8a21 P1 Explore (NOT PUSHED).**
  - Opens coloured by "My decks": sleeved decks lit, a card in 2+ decks white, the rest
    texture, bench & brewing a legend toggle.
  - A "New: <set>" lens is derived from the data (FRA today).
  - Set, date and type filters sit behind a Filters button with a count.
  - Topo/Labels/Motion are in a "⚙ Map" popover.
  - Region labels carry a functional subtitle.
  - Tests: `tests/test_viz_explore_lens.py`.
- **c64927f8 Tags (NOT PUSHED).**
  - The `draw` and `discard` regexes were widened (sweep: draw +535, discard +768,
    0 dropped; "discard this card" excluded).
  - The ability model was retrained: function r@10 0.232 → 0.258, median rank 76 → 66.
  - The wheels now get synergy partners.
  - `region_names.json` was re-keyed by membership overlap.
  - `DATA_VERSION` is 14 and the busts are at 267.

## Test state

- **Unit:** 3362 passed. Separately,
  `test_serve_cli_threads::test_stdout_survives_concurrent_cli_calls` passes alone but
  times out under load (it failed 3× on 10-28/29).
- **Regression:** green after the fixes in c64927f8.
- **Browser:** NOT run on c64927f8. On 2eec8a21 it was 396/397;
  `test_hovering_names_the_card_under_the_cursor` failed under load and passes 4/4 alone.
- The machine was heavily loaded (Chrome at ~140–200% CPU; load 20–47). Close Chrome
  before a run.

## Next, in order

1. Run `make prepush` on a quiet machine (Chrome closed). Then ask Sean before pushing
   2eec8a21 + c64927f8.
2. Fix the concurrency flake in `tests/test_serve_cli_threads.py` (raise the join timeout
   or move it out of `-n auto`), with a reason in the test.
3. Verify in Chrome with serve running (`.venv/bin/manamap serve`; hard reload for v=267):
   - Explore opens on My decks.
   - The New set pill works.
   - Filters and ⚙ Map work.
   - Region subtitles show.
   - Windfall in Discover shows Synergy.
4. **DECISION FOR SEAN:** the wheels' synergy partners are generic. All four wheels get
   the same 10 (Three Steps Ahead, Refute, Kozilek…), because the rules only pair tags.
   - Real payoffs (Nekusar, Waste Not, Brallin, Bone Miser — "whenever a player
     draws/discards") have no tag.
   - Adding a payoff tag means a new frozen tag + a corpus sweep + retraining steps 3–15
     (about 1 h on a quiet machine).
   - Ask before doing it.
5. Smaller UI follow-ups:
   - Sharknado has no authored deck name: the page shows the slug, and there is no
     `issue.json`. Ask Sean for the name.
   - Some legend names are all caps (EDGAR MARKOV, THE UR-DRAGON), straight from the
     manifest.
   - Optional data work suggested by the Explore agent:
     - `cluster-regions` writes the subtitle from the full `card_roles.json`;
     - drop "(North)" and the "Auras 6" names;
     - `sets.json` carries Scryfall's `set_type`, so the set lens needn't use a 100-card
       floor.
   - `regions_cardbert.json` (the pages' map) was not re-clustered after the tag change.
     Its `top_tags`/`mechanical` lines predate the new tags. Re-running needs the same
     membership re-key: match old→new regions by Jaccard ≥ 0.5 on `membership`, then
     carry the name to the new signature.
6. Not UI, still open:
   - Prove the Sharknado stacks: Gratuitous Violence + Needlehead, and Brallin +
     Needlehead on one wheel.
   - The engine critics fail for heliod and ur-dragon.

## Standing rules

- Never start Forge, regen or long jobs unasked; regen is `--slug` only.
- Never cut the pilot's keeps.
- Report a whole profile by axis, never one headline metric.
- List cards non-lands then lands, each alphabetical.
- Cache-bust every page together.
- Never edit source while a suite or job runs.
- Commit to main; push only when asked.
