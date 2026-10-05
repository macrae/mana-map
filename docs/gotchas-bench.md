# The bench

The pilot bench: agents and the invocation cache, the simulator and the goldfish model, branches, the diagnostic layer, `deck-audit`, versions, the captain's log. Read before touching anything under `src/manamap/pilot/` or `src/manamap/sim/`.

Extracted verbatim from `CLAUDE.md` — every measurement here was in that file and none was reworded. `CLAUDE.md` loads into every session; this does not, so the rules that bite regardless of what you are touching stayed there and the full record moved here.

- **(History, 2026-08) The cache board was green once and the 23 MISSes of the embedding rebuild were re-blessed rather than re-spawned — with the reason.** It is red fleet-wide now, deliberately (see PLAN.md §Decisions); the rule below is unchanged. The embedding rebuild regenerated `synergy_graph.json` and `obsolescence_index.json`, which MISSed `writer-prose`, `the-ten` and `issue-plan` on all seven decks (plus hapatra's `candidate-pool` and `deck-build`). Re-spawning was ~2.46M tokens. **What did NOT miss is the load-bearing part**: every `stack:NNN` stayed HIT, as did `strategic-frame` and `coach-prose` — nothing rules-verified depends on those graphs. What missed is prose and packaging, and none of it quotes a synergy rank numerically. The record is still a claim someone read the artifact and agreed it holds; that claim was made deliberately, and this is its reasoning. **The rule itself is unchanged: never `cache-record` to make a board green.** If a future MISS touches a routine whose output cites the changed artifact, re-spawn it.
- **REVERSED 2026-08-25 — the frontend now reaches a LOCAL server, and the note below is kept because its reasoning still binds.** `manamap serve` exposes `/api/` beside the static files; GitHub Pages does not. The owner's reason: *"I AM the sole user … having a build process produces higher quality code, so we will check in and build what we can, but at the end of the day I want the software to work how I need it to … and that's have sub-agents accessible for questions, resolutions, etc. in the Build page."* So **the static build is HYGIENE, not a second audience** — it keeps CI honest and the artifacts deterministic, and those disciplines are why the code is worth trusting. The old objection ("two products, and only one is the one you test") is answered rather than ignored: **the difference is a feature of the page**. `Api.probe()` runs ONCE, every agent affordance is gated on `Api.ready`, and a page that cannot reach a server renders its static half and **names the command that would start one** — absent, never broken, never silent. **The browser still makes no model calls**: it asks the local server to run a NAMED command from an allow-list, exactly as a terminal would. Commands are a `{name: (function, {arg: coercion})}` table — nothing reaches a shell, an unknown name is a 404 rather than an attempt, a bad argument is a 400, and a missing required argument is checked at the endpoint (it once reached `edhrec_slug(None)` and surfaced as a 500, a server fault for a caller's mistake). Bound to `127.0.0.1`, deliberately not configurable. **Agent jobs are async and PRICED before they are spent** — `claude -p` with the repo's own `.claude/agents/*.md` charters, so the answer in the page is the answer in the terminal; the cheapest measured routine is 54.5k tokens and `candidate-pool` is 235k, and a button that spends a quarter of a million tokens without saying so is the one thing this must not become. And the probe's own bug is worth keeping: a **404 is the DEPLOYED shape**, not a malfunction, so reporting it as "the API answered oddly" would have shown a fault message to every visitor of the published site.
- **A DRAFT is a deck that is a brief and no more, and it gets its OWN list in the manifest.** A partial build is work you must be able to put down and find again, so the Workbench shows it first (`IN PROGRESS`, above `SLEEVED`) — a deck you started and cannot navigate back to is lost the moment the tab closes. It is `manifest.drafts`, NOT `manifest.decks`: every consumer of `decks` assumes a 99, and a draft has none, so the deck picker, the dossier and `deck-info` would each offer something they cannot load. Putting one in `decks` crashed the manifest projection outright, which was the honest early warning. `POST /api/build/save` is **idempotent and read-modify-write** — the page saves on every change, so a save must not be a replace, and `{slug, bracket}` must not drop the commander or the library somebody spent ten minutes gathering. **A required field is checked on the MERGED brief**, not the incoming patch: validating the patch refused exactly that update for lacking a field it already had on disk. A draft writes `brief.json` and nothing else — no `cards.json`, no `paper` block, because it claims no cardboard and no 99. **The manifest grew a key, so its fetch is now cache-busted** (`index.json?v=2` in both `workbench.js` and `build.js`): "no drafts" versus "one draft" is exactly the different-conclusion case, and a browser holding the old shape would show an empty bench and be quietly wrong about your own work. Adding that query string then broke a test's `page.route("**/data/decks/index.json")` glob, which is worth knowing before the next one.
- **A MENU IS A PROMISE — `FormatSpec.buildable`, and the bug that produced it.** The new-deck picker offered all five formats while `build_deck` builds ONE, so choosing Standard and pressing Build did nothing: reported as *"I tried to build a standard deck and nothing happened"*. TWO defects stacked. (1) `newDeckBuild` opened `if (!newDeck.slug) return;` — **a silent early return, which is literally that sentence**; a control that declines must say why or it reads as broken software rather than as a limit. (2) The builder genuinely cannot build a 60-card deck, and that is not a missing flag: it is **anchored on a commander at every step** — colour identity gates the candidate pool, the similarity score is seeded from the commander's name, its mechanical tags drive synergy, the bracket engine reads it, and `manabase` sizes against a 99-card library. A constructed deck has no such anchor (you build around an archetype and a colour pair), so it is a different build STRATEGY. So `buildable` is on the spec, the picker offers only what it can keep and **names what is missing rather than hiding it**, and `build/run` refuses with the reason and with what CAN be done instead — it used to come back "brief.json has no commander", which is true, useless, and blames the brief for a limitation of the builder. **Buildable is not the same question as legal**: Standard is fully validated and searched, just not built.
- **A COMMANDER DECK CAN BE BUILT FROM SCRATCH IN THE BROWSER, and the last terminal step is `build/finish`.** Format → commander → style → a legal, bracket-checked 99 → `cards.json` → on the bench, no CLI. **The commander field is a PICKER, not a text box that refuses**: typing "zur" used to produce a 400 with suggestions nobody could click, write no brief, and then report a missing file when Build was pressed — three messages for one unfinished thought, the first overwritten by the third because both wrote the same slot. Two letters, real commanders, click one; ranked start-of-name then shortest, legendary creatures only, quiet under two characters. **`build/finish` does NOT commit unless asked**: `decklist.txt` is tracked, so the commit is what `deck-version` NUMBERS and what the captain's log stamps games against — load-bearing enough that a button must not do it by surprise, so the exact command is returned instead. **Finish appears only once there is a 99 to finish** — beside Build it would read as a second route to the same place rather than the second half of one thing. Two error classes were leaking on this path and both are fixed at the source: `load_brief`'s **"author it first"** told a browser to go and write a JSON file, and **EDHREC answers an unknown theme with a different SHAPE** (a bare list, not `{deck: …}`), so `deck.get("cards")` sent `'list' object has no attribute 'get'` into the page as the reason a role budget fell back.
- **A RETRY LOOP THAT INSPECTS A STATUS CODE CANNOT SEE A DROPPED CONNECTION.** `fetch_deck._post_collection` retried 429 and 5xx through `resp.status_code`, so it could only ever survive a failure Scryfall was well enough to describe; a closed keep-alive socket raises INSIDE `SESSION.post`, where there is no response to inspect, and sailed past four retries written for exactly this blip — reaching a browser mid-build as `ConnectionError: ('Connection aborted.', RemoteDisconnected(…))`. The comment above the loop recorded fixing the 503 case and never the transport one, which is how the two halves of one bug drift apart. Three things the fix needs: catch `requests.exceptions.RequestException` (Connection/Timeout/ChunkedEncoding are one class of event), **`SESSION.close()` before retrying** — a dead pooled socket is reused and the retry fails identically, so the loop looks like it ran without doing anything — and raise a **sentence**, since exhausting the retries is an ordinary operating condition and "the deck is unchanged, run it again" is both true and the whole remedy. `serve.py` answers a `RuntimeError` with **502 and `str(exc)` alone**: the class-name prefix is a useful clue for an unexpected TYPE and pure noise in front of a message somebody wrote on purpose.
- **A PAGE LEFT OPEN CAN WRITE TO THE REPO.** The draft autosave is debounced 600ms and fires on every field change, so a tab still sitting on `?draft=<slug>` re-created a deck directory that had just been deleted — and the manifest test caught it as "the tracked manifest does not match disk". A real consequence of the local bridge existing: the browser is now a writer, not only a reader. Close the tab before cleaning up, and treat an unexpected `data/decks/<slug>/` as a hint that something is still open.
- **The frontend never calls an LLM, and deployed == local.** Decided 2026-08-01 after costing the alternative. The viz is exploration plus *artifact* reads: it renders what the pipeline and the agents already committed, and hands work back out through `Discovery.brief()`. It does not spawn agents, and there is no local-only bridge — because a local bridge means the deployed site and your machine run different code, and only one of them is the one you test. The handoff stays a brief you paste into Claude Code.
  - Why it is not merely a limitation: the *deterministic* layer is already 123 pilot subcommands with JSON output and zero LLM calls (Sven routes to them; he does not replace them) (`deck-facts` 1.5s, `bracket-check`, `manabase`, `goldfish`, `query-rules`/`query-strategy` as pure local RAG). Almost everything that feels like "ask the system a question" is answerable without an agent at all. What genuinely needs one is artifact-shaped and expensive — cheapest routine was `coach-prose` at 54.5k tokens (now part of `pilot-notes`), `candidate-pool` is 235k, and `docs/agent-cost.md` notes ad-hoc consults "produce no artifact, so there is nothing to key against". Those are jobs, not chat turns.
- **Swap history is DERIVED from git, never hand-kept.** Three places recorded it before `deck-history` and none could be trusted: comment blocks in `decklist.txt` (prose, and the deck moves out from under them), `HISTORY.md` on half the decks (append-only and append-forgotten), and `considering.json`, which is replaced wholesale on every regeneration — so an applied ten leaves no trace of having existed. `decklist.txt` is tracked, so every change to the 99 IS a commit; diffing the parsed list across `git log` cannot drift and needs no maintenance, the same argument `build-index` makes for the deck manifest. What it cannot know is *why* a card moved — the commit subject is reported verbatim as `reason`, which is an argument for good commit subjects rather than for a second file that would disagree. **Ownership in the pending block is derived when `acquisition` is absent**: one regeneration dropped that field and every pending swap then read as "buy". An absent field is not evidence a card is unowned — the boxes are checked first.
Ownership is `COLLECTION_DIR` = `data/collection/`, read ONLY through `pilot/collection.py`;
a hardcoded top-level `share/` was the original bug and that directory does not exist. This is the ONLY ownership question left in the repo, and it is about a physical collection, not a deck.
- **`deck-audit` is the join nothing else performs, and its targets are CITED.** Four commands measure a deck (`deck-facts` composition, `mana-analysis` castability, `goldfish` speed, `bracket-check` power) and nothing joined them, so "is my card draw enough" had no answer. `deck-audit` emits 16 axes each carrying the **verbatim `strategy.md` quote** that sets its target — a test fails if any quote drifts out of the doc, because a target nobody can quote is not a target. That is the defect `DECK_ROLE_BUDGET` was built with: one flat budget for every deck, its own comment calling it "PROVISIONAL", `upgrade_facts` printing its shortfalls as "Context, not evidence". Three things a fleet survey caught: **Burgess's formula budgets sources, not lands** (applied to the land count it asks a 5-colour deck with a 9-mana commander for 45 lands, so `mana-base` takes the conventional 36–38 and `mana-sources` takes Burgess); **aggro's "26-32" is a creature count**, and overriding `threat-density` with it told edgar it was thirteen finishers short; and **an axis count is a floor** — oracle probes name cards showing the function that the taxonomy filed elsewhere, because `card_roles.json` calls Yawgmoth `removal:debuff` and his ability draws a card. Computed on demand, **never committed** — it embeds goldfish and bracket figures.
- **`goldfish_targets.json` was already a machine-readable engine declaration and nothing read it as one.** Its `any_of` groups ARE the engine's components and a group's SIZE is that component's redundancy — priced through `hypergeometric_at_least`, which reproduces `strategy:deckbuilding.redundancy-vs-tutors`'s cited 31%/41%/54% (asserted by test), and set beside the rate the simulation measured. The thinnest group is where the deck fails first, and "what would activate the engine" becomes "which pool cards would join it". **The role route needs a SHARED role, not a modal one**: run off one card's roles, a component holding only Blowfly Infestation returns Massacre Wurm and Dismember — the roles describe the card, not the group's job. Two members sharing an *axis* (Sol Ring `ramp:rock` + Dark Ritual `ramp:ritual`) is the coarser fallback. **`manamap pilot validate-goldfish-targets <slug>` checks the declaration itself**, because a fleet survey found it wrong on six of eight decks — a component every member of which a passing stack refutes, a group declaring two cards where the deck holds four, and *twice* a **primary win line with no target at all**, so the simulator never measured how the deck actually wins (heliod's Hullbreaker Horror, ur-dragon's Aggravated Assault). Two checks only: declared cards must still be in the 99, and a card in ≥2 checker-passed stacks belonging to no component is reported — commanders and basic lands excluded, or four commanders false-positive.
- **A goldfish component must name the leg every proof OPENS on, and a payoff group must be split by trigger EVENT.** Edgar's headline number was wrong twice in one direction and it took `validate-goldfish-targets`' cheapest complaint to find either. **THE GO-WIDE KILL by turn six: 0.382 → 0.327 → 0.285.** (1) Every checker-passed proof of that kill opens by CASTING a cheap Vampire to fire eminence and no group named that leg — the validator had been saying so ("Gifted Aetherborn in 3 passing stacks and no component") and it reads as bookkeeping. The leg is 13 cards and is **not free**: 13 of 99 is 84.8% by turn six, so the model had been assuming a card it should have been drawing. (2) The payoff leg was wrong in BOTH directions: it counted Blood Artist and Cruel Celebrant, which fire when a creature **dies** and cannot fire on the entry event the kill runs on, and omitted **Mirkwood Bats**, which keys on token **creation** rather than on "enters" — which is exactly why the declaration AND an oracle sweep for `enters` both walked past it. Split the group by trigger event rather than relabelling it, and **keep the excluded half as its own component** unless you have shown the deck cannot reach it: `engine-critic` refuted the claim that the death drains were stranded (Skullclamp kills a 1/1 token for `{1}`; Indulgent Aristocrat's outlet has no tap symbol and no timing restriction), so they are a second line, not dead weight. Three independent agents found the same defect from three directions, which is the argument for asking more than one.
- **A stack CITED for a line must name the line's cards in the SCENARIO, and no shape test can enforce it — measured, 2026-08-24.** `engine-critic` caught a third instance of `validate_engine`'s documented blind spot (an arrow through Cordial Vampire citing a stack whose scenario says twice, in the negative, that it never triggers), and two formulations of a closing check were then RUN over all nine tracked engines and their passing stacks — 120 lines, 57 verified — by three parties independently, agreeing to the line. As-worded: **46 fires of 57**, 9 of radagast's 9. The refined acting-stage variant: **1 fire fleet-wide, and it is a FALSE POSITIVE on a critic-passed model** (ur-dragon `lines[3]`). It cannot be fixed because edgar's defect and ur-dragon's correct line are *structurally identical* — a line whose `from` stage holds one card on the stack and one on the battlefield, `via` naming the battlefield one — and what separates them is a rules fact. Scoping by `carries` fails independently: it is free prose the engineer writes, so a rename evades the check silently. Full record and numbers in `validate_engine.py`'s docstring; do not re-derive them.
- **A validator that fires on correct data is worse than no validator, and the only way to know is to measure it against the whole fleet first.** Three proposed checks were prototyped and *rejected* on that basis. "Group members should share a role axis" flagged Howling Mine and Font of Mythos as outliers in heliod's draw engine, which they are, and agreed 1-of-6 with goblin-storm's cantrip group where every member is right — because `ROLE_PATTERNS` answers "what job in a 99" while a goldfish group is a **deck-specific functional set** with no taxonomy axis (there is no `cantrip` role). "Engine component roles must be unique" would have fired on four of seven decks: duplicate roles are normal (hapatra runs `payoff`×2 and `piece`×3, edgar `piece`×4) because a role is a KIND, not a key. SPOF-to-component agreement died with it — `component` is free prose ("Mikaeus, the Unhallowed"), not a role reference.
- **An add's `closes` must move the axis it names, and the frame is MARGINAL — not isolation.** Four adds across three decks named an axis the axis does not credit them for: Nature's Claim doubled two covered classes, Walking Ballista carries no `wincon:*` so threat-density could not see it, Bojuka Bog is only `land:tapped` so the breadth function skips it before reading its text. Each card does the thing in Magic terms and the *measure* does not move, so the prescription was sized against a number buying it would not change. Isolation misses the commonest shape: Nature's Claim alone takes hapatra's breadth 1→3 and looks fine, but Assassin's Trophy is bought in the same package and already covers both classes, so the pair and the trio are both 4. `validate_diagnosis` checks only axes computable from `(cards, roles)` — the seven role counts and `interaction-breadth`; `colour-sources` and `mana-base` need `mana_analysis`, and a half-recomputed axis is a worse answer than none. `_named_axis` matches **longest-first**, because `interaction` and `interaction-breadth` are both axes and a shortest-first scan silently checks the wrong one. The paired-swap floor check applies only `(add, natural_cut)` PAIRS: **a cut list is a ranked list, not a mandated set** — sisay deliberately lists a `painful` cut it simultaneously holds, and applying every listed candidate failed it for a swap it does not prescribe.
- **`bracket-check <slug>` with no `--target` inherits the recorded one and is idempotent.** A deck's target bracket is a property of the DECK, not of the invocation, so a bare re-run must not strip `target`, `within_target` or `cut_candidates` from the tracked report — that silently turns "is this deck inside its bracket" into a question the file no longer answers.
- **Charter edits invalidate before they inform: make them BEFORE `cache-record`, not after.** Editing `.claude/agents/*.md` MISSes that agent's routines by design, and a charter edit **disqualifies STALE_OK by construction** — so it cannot be re-blessed, and `cache-record`ing to make the board green is the thing this repo forbids. A purely operational note (per-slug scratch filenames, which cannot change any figure) cost edgar's freshly-recorded `deck-diagnosis` entry.
- **A cut list will propose the one card a verified line rests on.** `validate-diagnosis` computes `orphans_stack` rather than reading it: if a proposed cut names a card appearing in a checker-passed stack's **scenario block**, the entry must list those ids. Scenario only — a checker note may discuss a card the board never held, and a discussion is not a dependency. Nothing else in the repo performs this check; `validate_considering` verifies a `natural_cut` is a real card and stops there.
- **The diagnosis is a working artifact, not a page.** `diagnosis.json` feeds `/prescribe` and the pilot; the renderer never reads it. It may say a card is underperforming and compare the deck to what it could be — the page's self-containment rule (the repealed magazine law L10) never applied to it, and the workbench has no such rule anywhere now.
- **A deck's EXISTENCE is a fact about cardboard, and it was authored where only the renderer could see it.** `status` on `issue.json` (`broken-down` / `retired` / `superseded`; absent = live) says whether the deck is still sleeved. It was defined in `issue_spec` and read by the magazine alone, so `deck-info` — the START HERE command — spent the whole pivot telling the pilot to go and play `hapatra`, whose cards had been pulled and sleeved into yawgmoth-swarm months earlier. The vocabulary now lives in `pilot/common.py` (`DECK_STATUSES`, `UNPLAYABLE_STATUSES`, `deck_lifecycle`) with `issue_spec` re-exporting the old names, because **a live command importing a module scheduled for deletion is a break with a date on it** — `issue_spec` goes when the compact page lands. `deck-info` prints the banner and withholds the suggestions that need cardboard (log a game, simulate, experiment) — and **says that it withheld them**, since a silently shorter list reads as "nothing to do here". `superseded` is deliberately NOT unplayable: that list is still sleeved, it is just no longer the best version of itself. Marking a deck obliges a `build-manual` + `build-index` rebuild, which is what puts the banner on the tracked page.
- **The captain's log is AUTHORED; the debrief is DERIVED — and the validator's only job is keeping the second inside the first.** `manamap pilot deck-notes <slug> add "…" [--result win|loss|draw] [--opponents N] [--tag T]` appends one JSON line to `log.jsonl` stamped with the sha of `decklist.txt` *as it stood* (not `cards.json`'s stamp — the text file is what moves when a card is swapped, whether or not `fetch-deck` has run), and nothing ever rewrites it; the same rule `issue.json` lives under. The `debrief` agent reads un-debriefed ids and writes `log_annotations.json` by id (`merge-debrief` rejects ids the log lacks and carries earlier entries; `validate-debrief` fails an opponent reading without a verbatim phrase of the note, a card that is neither in the 99 nor written in the note, a line asserted without a checker-passed stack, a route outside `resolve-stack|goldfish|research-strategy|diagnose`, and a stage `engine.json` does not declare). No mood field and no superlative check, on purpose: neither is mechanically decidable, and a validator that fires on correct data is worse than none. `debrief` is N/A on a deck with nothing logged rather than a permanent MISS. This is the bridge from play to the lab: `open_questions` route to the loops, `decisions[].worth_a_spread` feeds `/author-decision`, and the doctor reads the log (step 5).
- **"Locked" is an ASSERTION and nothing in the repo could make it before.** `DECK_STATUSES` holds three values and all three are obituaries (`broken-down`, `superseded`, `retired`); `deck_status_of` returns `None` for a healthy deck, so "live" meant *not explicitly killed* — an absence. The workbench's front door filters on "can I play this tonight", which no artifact derives, because it is a fact about cardboard. So `deck_versions.json` gains an authored `paper` block, and it hangs off a **version**, not a deck: what is sleeved is one exact 99. That is also what makes **drift** free — `report()` already knew `current_version`, so locked/in_sync/versions_behind falls out, and `diff_vs_working` already named the cards. The two sides are `pull` and `add`, the hands' side rather than the diff's, because that is the physical instruction. Verified on edgar: locking V5 recovers THE LOCK's twelve swaps exactly. `paper_state` walks git, so `build_index` computes it ONLY for decks that carry a lock — unlocked costs one file read.
- **UNLOCKED is a THIRD state, and `deck-info` had only two — so it treated "nobody has said" as "yes, it is sleeved".** LOCKED means the pilot asserted this exact list is on the table; a dead `status` means it demonstrably is not; ABSENT means nobody has claimed either way, which is where **four of eleven decks** sit and where every build plan sits. The START HERE command read `deck_versions.json` for nothing and told the pilot to go and play decks that may never have been built. That is the quiet half of the defect that had this same command recommending `hapatra` while hapatra's cards were in yawgmoth's sleeves — the loud half was caught because the cards were provably elsewhere; this one just does not know and said nothing, which reads identically to knowing. It **informs rather than withholds** (an unbuilt deck is not a closed deck, so the play/measure lines stay and a line above them names the command); withholding stays reserved for a deck we know is gone. Only the AUTHORED half reaches `info.json` — `paper()` is a plain file read, while `paper_state`'s drift needs git and `info.json` is committed, so a stored drift would be one swap behind forever; a test asserts `in_sync`/`drift` never appear there and the whole composition runs in a fixture with no git repo at all.
- **A lock nobody checked is worse than no lock.** Three decks carried placeholder `paper` blocks written to exercise the drift display during a demo — two asserting "in sync", one asserting a two-card drift — and all three rendered identically to evidence on the surface whose entire job is saying what you can play tonight. They were WITHDRAWN rather than corrected, because a lock's note should say who checked and when. Fabricating a claim to demonstrate the machinery that displays claims is the same failure as `cache-record`ing to make a board green.
- **`check-in` refuses rather than guesses, and the refusals are the feature.** A paper list is typed by a human reading sleeves: a card written twice, a name misremembered, ninety-nine where there should be a hundred. Every one of those is silently survivable — `fetch-deck` resolves what it can and moves on — and every one leaves a repo list that is not the deck on the table, after which everything downstream measures a deck nobody owns. Blocking: a non-basic listed twice (basics exempt), a total that is not 100, a name matching nothing in the corpus, no commander. Warning only: a changed commander (a different deck; probably wants a new slug) and an absent corpus, since a fresh clone must still be able to accept a deck. The write is **canonical, not verbatim**, and reformatting cannot manufacture a version because `deck-history` compares parsed entries. **The commit is load-bearing**: `decklist.txt` is tracked, so the commit is what `deck-version` numbers and what the log stamps games against — check a deck in without committing and tonight's games attach to no version at all.
- **THE ROSTER IS THE CONSTELLATION'S OTHER FORM, and the dossier had neither the cards nor a way to read them.** The page could show a PICTURE of the deck, what it is trying to be and what it costs, and never the list — `cards.json` was dropped from its fetches years ago as "read by nothing" and nothing replaced it. `rosterPanel` renders `deck_map.json` — already in hand for the constellation — as a list grouped by city, ordered by size exactly as `render_the_99` does, with the heading taking `CITY_INK` **at the same index** so the roster is the picture's legend rather than a second taxonomy (the whole of `design.py:city_head`'s argument). **A card the map could not place still gets a seat**, the way `render_the_99` ends with a `stray` group; a roster that silently drops a card is worse than none. **The hover preview is CSS-ONLY**, ported from `design.py:163` — an anchor with a nested `.card-pop` — so there is no positioning code and no tooltip layer, and it opens DOWNWARD because a roster is a column of names and an upward preview sits off the top of every group's first entry (the manual's `.swap-list` learned this on one list; this has it throughout). Images route through `Shell.cardImageUrl`, which already carries the DFC front-face retry. **No engine-stage chips**, unlike the printed `card_tile`: a branch inherits the DECK's `engine.json`, which describes the old list, so a stage on a card the model never placed is a claim nobody made.
- **A DECKLIST AND A `cards.json` NAME A DOUBLE-FACED CARD DIFFERENTLY, AND BOTH ARE RIGHT.** `cards.json` and everything derived from it key the joined `A // B` form; a decklist names the front face, and for a transform card it HAS to — Scryfall answers the joined name with a **404** and resolves the front face alone, so a decklist carrying `A // B` cannot be fetched at all. Unreconciled, the seam is silent: the roster joined `deck_map`'s names to `deck_branch`'s and lost exactly two cards, under-marking the additions **30 against 32** and the buy list **19 against 21**. Nothing errored — two rows simply rendered unmarked. `deck_branch._canonical` maps every face to the resolved name through `common.expand_faces`, which already existed for this ("decklists name whichever face the writer had in front of them"), and ownership lookups match on either face too. **The fix landed in `source()` and silently missed `diff()`**, because a `.replace` without an `assert` did not match a body that had gained a comment — which is why buy/box agreed while `add` stayed wrong, and why every edit in this file asserts its anchor.
- **FORGE IS A CORRECT RULES ENGINE AND A BAD PILOT, AND THOSE ARE SEPARABLE.** Asked whether the simulation is legitimate, the logs answer yes on rules — `Whenever an artifact you control enters, Reckless Fireweaver deals 1 damage to each opponent. [Zone Changer: Treasure Token (427)]` is the branch's whole thesis firing correctly, treasures are sacrificed for mana, Revel in Riches won four games outright. And no on piloting: **0.67 land drops per own turn, 9.2 casts a game, first attack on turn 17, the keystone (Academy Manufactor) cast in 27 of 100 games.** **THE AI IS NOT TUNABLE FOR THIS.** Its four `res/ai/*.ai` profiles carry ~200 knobs of which the land-related ones are all Strip Mine, Scry, Explore and Momir edge cases — there is no knob for "make your land drop" or for sequencing, that is Java — and the repo had already measured (2026-08-19) that the aggro profiles make a hold-up deck WORSE. So the answer to "can we pilot better" is no, and the answer to "is it therefore noise" is: only the win rate is.
- **THE VERDICT RESTS ON LAND DROPS ALONE, BECAUSE CASTS-PER-TURN IS A CURVE MEASURE WEARING A PILOTING ONE.** Across every tracked run `corr(mean mana value, casts ratio) = **-0.50**`: an expensive deck casts fewer spells while being played perfectly. Scored on it, the gate flagged radagast NOT COMPARABLE at 0.84 against a 0.85 line on a twenty-game run whose only fault is a mean mana value of 2.97 — **a check firing on correct data, which this repo has now rejected four times**. A land drop is not confounded that way: every deck wants its land every turn whatever it costs. Casts are still reported, with the confound named. And **a single game yields no verdict** (`MIN_GAMES = 8`) — the n=1 smoke run reads 0.60, which is a shuffle. A test runs the gate over every tracked run and fails on any positive, so a false one cannot teach its reader to ignore the flag.
- **EVERY RUN CONTAINS ITS OWN CONTROL, WHICH IS WHAT MAKES THE VERDICT MEASURABLE RATHER THAN A CAVEAT.** `sim/pilot_quality.py` asks not "did the AI play well" — unanswerable, nothing calibrates it — but **"was our seat played about as well as the pod"**, using the other seats in the same games under the same engine. Measured on the treasure branch: our seat 0.67 land drops per turn against a pod mean of 0.72, casts 1.04 against 1.11, so **90–97% of the pod's rate: COMPARABLE**. Every seat misses about a third of its drops, and **a UNIFORM weakness is a completely different finding from a weakness at our archetype** — it leaves an A/B between two of your own lists against one pod substantially intact while rescuing no absolute win rate. Checked directly: the AI played the champion and the branch alike (lands 5.5 vs 6.0, casts 9.9 vs 9.2), so **my first reading — "biased against the branch by construction" — was wrong**, and the 11-vs-12 result is a fair comparison badly played by both sides rather than a rigged one. Computed at print time from the RECORD, never written into it and never reading the gitignored logs, so no tracked run moved and the reading works from a fresh checkout.
- **A BRANCH IS A SEAT — `<slug>@<branch>` sits down at the table like any list — AND THE FLATTENED NAME SCORED IT ZERO.** Forge gets `mm-ur-dragon-treasure-v2` because `@` has no business in a deck registry, so an outcome names the flattened form while `seats` holds the slug: `o["winner"] == s` matched every OTHER seat and silently counted ours at nought. **The run printed `wins 0` for a list that had won ELEVEN of a hundred, with the other three seats' counts correct beside it** — which is exactly the shape that gets believed, and it was in `summary` while `analysis.seats` held the right number all along. `--analyze` could not repair it either, because it rewrote `analysis` and `games` and never `summary`. Both fixed, and a test asserts the two blocks of a tracked run agree. **Branch runs are filed under `branches/<name>/sim/`** via `_out_dir`, so a branch's win rate can never land under the champion's name; `validate-sim` and `bridge` resolve the same way. `experiment` takes `@<branch>` as an arm, which makes the most useful A/B in the system sayable at last: the list you are considering against the one you are playing, same pod, same seed.
- **THE GOLDFISH AND THE TABLE DISAGREED, AND THAT IS THE POINT OF HAVING BOTH.** ur-dragon's treasure branch assembles its declared engine **five times as often** as the champion (0.099 → 0.593 by turn three, interval on the difference excluding zero) and won **11 of 100** against the pod where the champion won 12 — a difference of −0.01 whose interval `[-0.101, +0.081]` spans zero, so at 100 games a side the two are indistinguishable. **A goldfish measures whether a deck does what it says; only a table measures whether that wins.** The one thing the table saw that no other layer could: **Revel in Riches actually fired, closing 4 of the branch's 11 wins** — an alternate win the goldfish can price the assembly of and never observe paying out.
- **THE READING IS WHERE A TOOL STARTS INVENTING THINGS, so every line is derived by a stated rule and carries the measure it came from.** `diagnostic.interpret` turns a comparison into plain language and does exactly three things beyond restating numbers: it knows which DIRECTION is an improvement per measure (without that, a lower stall reads as a loss), it converts a rate into a frequency, and it names the bottleneck. **Nothing weighs one axis against another and nothing calls a trade good** — the trade is shown and ruling on it is the pilot's. **THE DISTINCTION THAT DOES THE MOST WORK IS "did not change" VERSUS "could not be seen"**: they print identically and they are opposite findings. Below the MDE it says *evidence of NOTHING, not evidence of no change*; above it, with an interval spanning zero, it says flat — because the run COULD have resolved it and did not.
- **A FREQUENCY IS A PRESENTATION AID AND MUST NEVER ERASE A DIFFERENCE THE INTERVAL SHOWS.** "One game in twenty" is easier to hold at the table than 0.053 — but at an absolute tolerance both 0.039 and 0.053 rounded to *the same words* while the interval on their difference excluded zero, so the phrasing hid a real cost. The tolerance is RELATIVE now (`max(0.008, rate * 0.10)`), fractions reduce (`2 games in 50` is `1 game in 25`), and `_pair` falls back to percentages the moment two numbers would print alike. The numbers win over the phrasing, always.
- **CALIBRATION KILLED TWO OF THE PRD'S OWN IDEAS, AND ONLY RUNNING THE FLEET COULD SHOW IT.** (1) **The PRD's `P(stall) > 0.15 -> red` fires on ZERO of 13 decks** — the highest reading anywhere is 0.079. A red line that can never go red is as useless as one that always does, and it would have shipped looking rigorous; this repo had already rejected three checks for the OPPOSITE failure (one fired on 27% of correct data), and this is the same mistake pointing the other way. (2) **Three of the mana readings are one measurement**: missed-drop-by-five vs the all-turn drop rate r = **+0.994**, vs mulligan rate **+0.968**, and those two **+0.958** — all driven by land count. `benchmark.py` had recorded the two-way version at 0.97 and refused to sum them; the third member is new. They are reported together with the correlation stated so three confirmations of one fact cannot read as three findings. **Stall is the one INDEPENDENT reading** (r = −0.26 to −0.41 against that family), which is what makes it a real second dimension rather than `consistency`'s mistake again. `diagnostic.FLEET` holds the observed bands as CONTEXT for placing a reading — never as the grade, which is strategy-relative — and `pytest -m fleet` re-derives them so they cannot outlive their evidence.
- **TRIAGE BEFORE MEASUREMENT — `assess` is the reading that has to happen first, and it exists because doing it by hand found three things no simulation would say.** On a 21-card pile brought from the Atlas library: **half was combat-gated** for a deck built to win without an attack step, **one card needed a creature type the deck runs none of** (Magda wants Dwarves), and **the two best cards were invisible to every model here**. Each is cheap to check and expensive to miss, and `candidates` — which substitutes and re-measures — would have reported all of them as noise, correctly and uselessly. The order is the one that kills candidates fastest: real card → legal in identity → cost and job → **what is it GATED on** → does it feed a DECLARED component → **can any model here see it** → is something in the list already doing that job for less. **Nothing scores a card**; every line carries its reason so a pilot can disagree with it.
- **A TYPE NAME IN RULES TEXT IS AMBIGUOUS AND THE OBVIOUS READING IS WRONG TWICE.** `Artifact Creature — Treasure Dog` exists, so the corpus honestly reports **Treasure as a creature type** — but "Treasures you control" means the TOKEN every time, and a deck's `cards.json` never lists a token, so the tribe check called a perfectly castable Alchemist's Talent dead. `TOKEN_NAMES` excludes the artifact-token names for that reason. Before that, the scan tested `"Creature" in line` across a DOUBLE-FACED type line and then read the FRONT face, harvesting artifact subtypes off cards whose back is a creature — it is per-face now. And the first cut used a capital-letter regex instead of the corpus at all, which produced "needs Treasuress": a confident sentence about a card with no tribal text.
- **`--as` ASKS THE HYPOTHETICAL A CANDIDATE SWEEP CANNOT OTHERWISE ASK, and the override has to reach the SIMULATION.** A card the declaration does not name cannot move an engine axis except by displacing something, so "would this widen my thinnest component?" reads as noise every time, correctly. `--as <target>` counts the candidate toward that group FOR THE MEASUREMENT ONLY, never written back. **The first cut passed the modified declaration to the reporting layer while `goldfish` still read the file** — `target_turns` stayed indexed by the file's targets and the override changed nothing; it is `targets_override=` on `goldfish.run` now. **The tell was eight different candidates returning the identical 0.501, which is the third time in this subsystem that identical numbers were the signal.**
- **IDENTICAL READINGS CAN BE THE ANSWER RATHER THAN THE FAULT, and the tool has to say which.** With the override working, all eight multiplier candidates read **+0.039** — the same, because a goldfish target asks whether a card was DRAWN, so the seventh member of a group raises its assembly rate by the same amount whichever card it is. That is a real answer to "what is one more member worth" and no answer at all to "which member". Eight identical rows would otherwise read as a broken tool, so `all_identical` is detected and said out loud.
- **THE CHEAPEST WIN WAS A DECLARATION FIX, NOT A PURCHASE.** ur-dragon's treasure branch bottlenecked on a six-card multiplier group; **Panharmonicon was already in the deck and the declaration had missed it**, because the scan looked for token-doubling language ("twice that many") and it doubles ETB TRIGGERS instead. Naming it moved engine-online by turn three **0.459 → 0.508** for nothing. In the same edit Peregrin Took was added and REFUSED by `declaration_fits` — it is not in that list, and a declaration naming a card the deck does not run measures a different deck. Check the declaration before costing a swap.
- **A VOCABULARY IS NOT SOMETHING TO ASK A PILOT TO RECALL IN A `prompt()`.** `deck-branch new` demands `<measure> <op> <number>` and refuses without one, which is right — a branch that cannot be falsified gets graded on whether it did what it does. But producing one from memory means guessing at `OBJECTIVE_AXES` **and** at what number this deck could plausibly reach, and an objective set past what any list of this shape achieves is not ambition, it is a report that will read `not met` whatever the pilot does. So `deck-doctor` grew a fourth mode, `MODE: objective`: it reads `deck-info` and `diagnose --json`, does NOT run the audit, and proposes one axis **with the current reading beside it** — "hoard_8 >= 6.0" means nothing until a reader knows the deck is at 3.9. **It proposes and writes nothing**; the pilot confirms, and only then does `branch/new` land anything. **The mode string is built server-side**, because a page that could name its own mode could ask for a full diagnosis under an objective's price, and stating a cost before spending it only means something if the cost is the real one. **A REFUSAL IS AN ANSWER AND MUST NOT BE A DEAD END**: *"make it better"* names no axis, the charter returns `axis: null` with the alternatives rather than guessing, and the page falls through to the manual prompt **showing the doctor's reasoning** — which is more than the pilot had before they asked.
- **THE PILE → `pool.txt` PIPE HAD NEVER ONCE WORKED, AND NOTHING TESTED EITHER HALF.** `Shell.consider()` read `store.zoneNames()` — `zoneNames` is a **getter** on `Session.library` (`session.js`), not a method — so every press of "Consider for a deck…" raised `TypeError: store.zoneNames is not a function` before it reached the server check. The comment two lines above it warns about a *different* shadowing bug on the same line and notes that `node --check` passes it happily, which is precisely why the second one survived beside it. `pool/save` had **no test at all** and `consider()` had none either, so an audit found this rather than a use. Two rules come out of it: **a getter and a method are not interchangeable and the failure is silent until the click**, and **an endpoint with no test is an endpoint with no evidence it is reachable** — the Python side was correct the whole time.
- **`Api.probe()` IS NOT CALLED FOR YOU.** It runs in `mana-map.js` (index) and `deck-view.js` (deck), and nowhere else — `branch.html` loaded `api.js` and never probed, `workbench.html` did not load it at all. The library drawer mounts on **every** surface, so on two of four its own buttons reported "This needs a local server" while a server was running. A shared component that reaches the API makes the probe a per-page obligation; probe in the page's `boot`, and never block the first paint on it, because every artifact panel renders without a server.
- **THE VERB LAYER, AND WHAT DELIBERATELY STAYS IN THE TERMINAL.** `branch.html` rendered a decision beautifully and could cause none of it: every branch verb lived in the CLI, so the page was a report about work you had to go elsewhere to do. `branch/new`, `branch/upgrades`, `branch/stage` and `branch/net-change` are allow-listed; **`branch/merge` and `branch/delete` are not, and a test asserts their absence** — a merge rewrites the tracked `decklist.txt`, runs the regeneration chain and is what `deck-version` numbers, and a button that spends cardboard is the one thing this bridge must not become. **`net-change` is ~15s, which is neither an agent nor a request that should block**, so it borrows the `JOBS` machinery with a wall-clock price string and the page polls it with the code it already had. **Snapshot the job record BEFORE starting the thread**: reading it back afterwards is a race that hands the caller `done` for a job it was told to poll — an agent takes minutes and never shows it, a 40ms local job shows it every time.
- **A MEASUREMENT MUST BE OF THE LIST AS IT STANDS.** Staging a swap rewrites `decklist.txt` and leaves the branch's `cards.json` describing the list *before* the swap, and a figure computed from that is a real measurement of a deck nobody has. The `branch/net-change` job compares the sha through `fetch_deck.is_up_to_date` and re-resolves only when it must. **`upgrades` sidesteps the question entirely by reading `decklist.txt` rather than `cards.json`**: everything it needs about a card — cost, identity, type, oracle text — is in the CORPUS keyed by name, and the only thing the deck supplies is which names it runs. Reading `cards.json` meant a freshly opened branch could not be looked at until somebody ran `fetch-deck` against Scryfall, which is how the end-to-end smoke test failed at step two.
- **IDENTITY IS THE COMMANDER'S, NEVER THE UNION OF WHAT THE LIST RUNS.** A union is narrower whenever a colour sits in the identity and no card in the 99 uses it, and it then refuses a legal candidate in exactly that colour — the one question a colour filter exists to answer.
- **THE OBSOLESCENCE INDEX PROPOSES EFFICIENCY, NOT IMPACT, and expecting a measurable delta from it is expecting the wrong thing.** It pairs cards that do the SAME JOB more cheaply, so the similarity gate and the tag-superset gate exclude by construction the card that would actually move a number: across **149 rows on six decks it proposed a Game Changer exactly zero times**, and a test holds that claim to the data rather than to prose. **THE MEASUREMENT IS DECK-LEVEL: swap a handful of cards and measure the lift.** A 100-card singleton dilutes any one card below what a run can usually resolve — a one-swap branch of ur-dragon returned noise on all nine rows of `net-change` — but that is a statement about the TYPICAL card and not a law: a Game Changer or a table-warper moves a number on its own, and some cards are. So a blank table on a barely-changed branch is arithmetic, not a verdict on the swaps, and `recommend` names the staged count, states the exception, and says to stage the rest first. The first draft of that note claimed one card is unmeasurable "by construction", which is the over-claim the exception exists against. Efficiency frees mana and slots; impact is what `prescribe` and `close` are for.
- **THE MODEL COULD NOT SEE EMINENCE, AND THE DECK IS BUILT ON IT.** `Cost reducers and rituals are not modeled (conservative)` was a stated assumption, and for a deck built ON a cost reducer that is not conservative — it is wrong about the thesis. The Ur-Dragon's eminence takes `{1}` off every Dragon spell **from the command zone**, live from turn one and unremovable, across 22 of its 24 creatures: mean Dragon MV **5.73 → 4.73**. Four more reducers sat in the 99 reading as vanilla bodies. Measured on the tracked list: commander cast by T6 **0.094 → 0.180**, mean bodies by T5 **1.38 → 1.92**, kill by T8 **0.299 → 0.505**. Every figure this bench published about that deck understated it. Third of the class, after "could not see 65% of the fleet's mana rocks" and "the mana model was colourless". **THE CONSISTENCY CHECK IS WHAT SAYS IT IS REAL**: every target ASSEMBLY rate is unchanged to the digit, because a goldfish target asks whether a card was DRAWN and drawing does not care what anything costs. A change that moved both would mean the discount had leaked into the shuffle. Rules-correct: the discount pays GENERIC only and is floored at the coloured pip count, and eminence says "OTHER Dragon spells" so it never pays for its own commander — but a Servant on the battlefield does, which is where a nine-drop feels it. **A cost reducer is neither a rock, a tutor nor a body**, so Urza's Incubator and Herald's Horn fell through every cast loop and sat in hand for ten turns while being the deck's stated curve fixer — the third card to find that hole after Aggravated Assault and Primal Vigor.
- **A REDUCTION THAT SCALES IS NOT A RATE, and the sweep is what caught it.** 93 corpus cards matched the first cut of the regex; 45 survive. `Creature spells you cast cost {1} less to cast FOR EACH …` — Animar counts counters and starts at zero, Rakdos counts life the opponents lost this turn which is zero forever in a solitaire model, Hamza counts +1/+1 counters. The regex stops at "to cast" and would have reported a flat 1 for all three: the Jeweled Lotus failure exactly. And the capture group was taking `Noncreature` on 7 cards, `Artifact` on 7, `Equipment` on 5, `Enchantment` on 4 and six colour words on 14 — none a creature subtype, all silently matching nothing. Refused against the corpus's own 383 creature types via `analysis.common.creature_types`, the same scan `assess` and `power_creep` use.
- **A DORK WHOSE OUTPUT IS THE BOARD READ AS ZERO.** `_TAP_ADD_RE` wants `{T}: Add <symbols>`; Bloom Tender and Faeburrow Elder say *"for each color among permanents you control, add one mana of that color"*, so the two best dorks a five-colour deck can run produced NOTHING while the conditional rocks they replace counted as five sources each. Of **34,084 cards, exactly FIVE** have a `{T}` mana ability the old regex misses and only these two are this shape — the other three are correctly excluded (Charmed Pendant pays with a mill, Idol of False Gods makes a token that sacrifices itself, Rainbow Dash is an acorn card). Priced by **snapshot at cast** from the colours actually in play, deliberately the conservative end: an Elder cast on two colours and living to see five is understated, and understating is recoverable. On ur-dragon's branch this moved mean mana at T4 from a measured **+0.14 to +0.61**.
- **REMINDER TEXT IS NOT THIS CARD'S ABILITY.** `Prosperous Innkeeper` has no mana ability at all — it creates a Treasure, and the Treasure's reminder text `(It's an artifact with "{T}, Sacrifice this token: Add one mana of any color.")` was being read as the creature's own, making it a five-colour source AND a `ramp:dork`. **24 corpus cards** read colours from reminder text alone, every one a Treasure-maker, and `Goldvein Pick` and `Prying Blade` are in zur-enchantress today. Parentheses in oracle text are always reminder text, so stripping them is exact rather than a heuristic — and it must be done in BOTH `land_colors` and `nonland_producer_kind`, because that second one is the GATE and without it a Treasure-maker still counts as a dork producing nothing. **The consequence is the expensive part**: six agent-authored `diagnosis.json` files quoted colour figures that were correct when written, and the remedy is re-running the doctor, never hand-patching the prose. → see PLAN.md for the sibling defect this exposed.
- **`mana-fit` — THE STEP THAT WAS MISSING FROM EVERY REBUILD.** `mana-analysis` measured the gap and stopped, so every refactor ended with a shortfall and no answer — and the shortfall MOVES the moment a spell changes: three counterspells out takes blue pips with them, six dorks in puts sources back. Ranked by **how many SHORT colours each card covers at once**, because a five-colour land is five fixes in one slot; basics rank last, since a five-colour deck that fixes its colours one basic at a time cannot cast its own spells. **It COMPOSES `mana-analysis` rather than recomputing it** — the first cut recomputed and reported **53 red sources against that module's 27**, because `land_colors` applied to a spell answers a question nobody asked and because that module gates on colour identity and on `nonland_producer_kind`. Two modules that can disagree about one number is the divergence this repo keeps paying for. It also names a **SPLASH**: a Karsten target of 30 sources driven by a single `{B}{B}` spell reads identically to one driven by thirty black cards, and cutting the card is usually cheaper than buying the sources.
- **A POOL IS NOT A PROMISE.** `build/save` sends the Atlas library to a brief's `must_include`, which says *these cards are in the 99*. `pool/save` writes `data/decks/<slug>/pool.txt`, which says *consider these* — the input to `candidates`, which substitutes each one and measures. They must not share a slot: afterwards nothing could tell a card you committed to from one you were only weighing. One PILE, not the whole library, matching the scope rule a brief already keeps. `--pool library` reads it.
- **TWO ARTIFACTS ONE LETTER APART, AND THE SECOND ASSIGNMENT WON SILENTLY.** `deck_info` already bound `diag` to `diagnosis.json`; `diagnostic.json` reused the name, so the dossier rendered **"not measured" over a file full of data** and nothing errored. It is `vitals` now. Related, in the same panel: **`facts()` ESCAPES its values** — correctly, since its twelve other callers pass plain text — so a figure built as HTML arrived as the literal string `<b>10.3%</b>`. The vitals build their own rows rather than widening `facts` into a place markup can arrive from data, and the browser test asserts `textContent` does not contain `<b>`, because markup-as-text is invisible to a does-it-render check.
- **THE DIAGNOSTIC IS STRATEGY-RELATIVE, AND THAT IS WHAT MAKES A GRADE POSSIBLE AT ALL.** `benchmark` freezes a harness so twelve decks can sit beside each other and deliberately IGNORES the declaration (a benchmark that read one would rank decks partly on how well their pilot writes JSON). `diagnose` reads the declaration and asks the other question — is this deck doing what IT says. The two must not merge. **The aggregate refusal is unchanged**: `speed` spans 400x and ranks a combo deck last for not attacking, `consistency` was `speed` at r=0.78, two of four inputs correlate at 0.97, and `benchmark.json`'s `"score": None` stays. Grading against the deck's OWN target dissolves that — heliod killing slowly is a fault only if heliod claims to kill fast. **Runtime is not the constraint anyone expects**: the whole existing stack is 12.6s on one deck and a 10,000-game diagnostic is 3.8s, against a 5–10 MINUTE budget. That headroom is the design — a candidate can be judged by SUBSTITUTING IT AND RE-MEASURING rather than by a score over its properties, which is why `candidates` is a measurement and not the six-factor scorer this repo already deleted once.
- **`engine_online` IS COUNTED, NEVER MULTIPLIED — measured 1.74x apart.** Components share cards, so P(A and B) ≠ P(A)P(B): on ur-dragon the joint by turn three is 0.1010 against a product of marginals of 0.0582, so composing from marginals would understate the engine by 42% and look entirely plausible doing it. `goldfish.run(with_results=True)` returns per-iteration `target_turns`, so the joint is a count over rows with the correlations intact; a test fails if the two ever agree. The declaration gains **`required`** (cannot function without) and **`route`** (one of several ways to close, counted as a UNION), which is the honest resolution of the older refusal that "a deck with four kills has no single assembled_rate" — a deck should not score near zero for having options. **Absent marking ⇒ absent figure, never 0.0**, same contract as `model_treasures`.
- **A BRANCH INHERITS THE DECK'S DECLARATION, AND FOR A REBUILD THAT MEASURES A DIFFERENT DECK.** Right for a swap, wrong for ur-dragon's treasure branch, where all ten inherited targets name cards the branch cut. `diagnostic.declaration_fits` detects it and WITHHOLDS the engine figure with the reason and the fix, because the number it would have produced is a real measurement of the wrong list and looks ordinary. Related: when two readings use different declarations the comparison prints **"THE TWO ENGINE FIGURES ANSWER DIFFERENT QUESTIONS"** — the branch's +0.36 is each list against ITS OWN intent, not one list being better at the same thing, and nothing on the row would tell you which.
- **A STALL IS A TURN WITH NOTHING CASTABLE, NOT A TURN WITH NOTHING CAST — the obvious definition measures the MODEL.** The goldfish is a resource simulation: it casts rocks, tutors, extra-combat permanents and bodies, and never a wipe, counterspell or targeted removal. Scored as "nothing was cast" ur-dragon reads **6.4 dead turns in ten while its hand grows to eleven cards**, which describes what the model declines to represent. Castability needs only mana value and available mana, so it is true of cards the model would never pick up: P(stall) then reads 0.629 on turn one and 0.040 by turn three — **which is why the headline excludes turn one**, since one mana and no one-drops is a structural fact about Commander rather than a fault. The cause is split too: on ur-dragon 906 stall turns, **none of them an empty hand**, so every one is mana and none is flood.
- **A SWEEP MUST NOT CUT AN ENGINE PIECE, AND MOST SINGLE-CARD SWAPS ARE INVISIBLE.** `candidates` substitutes each card and re-measures; the first version auto-cut the most expensive card, which on ur-dragon is Utvara Hellkite — named in a declared target — so every candidate came back "no reading" as the engine correctly refused a declaration that no longer described the list. **The sweep was testing its own cut.** It skips declared cards now and reports which cut it made, because the choice moves the answer. Every row carries the MDE and anything under it is marked as noise rather than ranked: measured, Jeweled Lotus, Mana Crypt and Rhystic Study returned *the identical* 0.0833 against a 0.125 baseline, because none is named in the declaration and the only thing that changed was the card that came out. **`stats.mde_proportion` is EXACT and overflows above ~400 per arm** (`math.comb(4000, k)` exceeds a double); the normal approximation is not a compromise at 10,000 games but the regime it is valid in, so the method switches and says so.
- **A BRANCH IS A CANDIDATE 99 THE PILOT CANNOT YET SLEEVE, and it is a deck-SHAPED DIRECTORY for one reason.** `decklist.txt` is tracked, so writing it MINTS A VERSION and the captain's log stamps games against versions — which makes a version you cannot physically play a version that lies. The Ur-Dragon treasure refactor was designed, measured, briefed and unappliable: 32 in, 32 out, 21 to buy. So `branches/<name>/` holds its own `decklist.txt`, `cards.json` and measurements, and `deck_dir(slug, branch=…)` resolves it — **every read and write in the package already goes through that one function, so scoping is STRUCTURAL rather than per-command**. A single `branches/<name>.txt` plus per-command write plumbing is the silent-overwrite class `resolve_out_path` documents, one level up: a command taking `--branch` for reading and writing through an un-branched path corrupts the tracked artifact with a list nobody is playing, under the deck's own name. **NOT a git branch** — versions derive from `git log` over `decklist.txt`, so one would mint numbers that never shipped, and checking it out moves every other deck's artifacts too. **READS FALL BACK, WRITES DO NOT** (`common.deck_file`): a branch owns its measurements but inherits AUTHORED inputs like `goldfish_targets.json`, because measuring against no engine declaration reports a different deck rather than a different list. **`source` answers FOUR states and the fourth is the one nothing computed before** — in the deck / in a box / **sleeved in another deck** / buy. The third is a trade-off rather than a purchase and carries the holder's lock: three of this prototype's cards sit in `goblin-storm`, which is finished. Ownership is still `collection.py` and still means a BOX; deck membership rides alongside as information, never folded in. **`elsewhere` IS A LOGISTICS PROBLEM, NOT AN OWNERSHIP ONE, and the first cut got it backwards.** Counting a card sleeved in another deck as unsourced reads as "buy a second copy" — advice to spend money on something already in the house: on ur-dragon's branch the pilot **already owns 74 of 95** while the report led with 25 unsourced. It keeps its own state, because unsleeving a locked deck is a real trade-off worth seeing (Anointed Procession and Mondrak sit in `edgar-vampires`, which carries a genuine paper lock and was played on the 22nd), and **`--proxy`** says the pilot will proxy across their own decks and makes it sourced. **`buy` is never proxiable** — a claim about a card nobody owns is a different decision and not one to make quietly. The report leads with "you already own N of M". **`merge` refuses on unsourced cards**, reuses `check_in.analyze`'s refusals verbatim, needs `--reason` with `--force`, does NOT commit, and never touches the `paper` block. **`counts` are DISTINCT NAMES, `size` is COPIES** — 36 basics are one line on a buy list and 36 cards to the shuffler. **The control is `test_a_branch_run_never_touches_the_decks_own_artifacts`, and it must call `main(args)` and not `analyze(slug, branch)`**: analyze RETURNS and main WRITES, so the first version tested the layer that cannot hold the bug and passed against a deliberately broken write path. An AST sweep for `branch` referenced-but-unbound caught three functions a blanket edit had rewritten without giving them the parameter.
- **Versions are derived from git and cannot be committed; tags are authored and can.** `manamap pilot deck-version <slug>` numbers every content-distinct `decklist.txt` (V1, V2…) by walking `deck-history`'s git log — a comment-only edit adds a byte-sha to its version, never a version — and joins the captain's log to them by the stamped sha, reporting games and W/L per list and an uncommitted working copy as **unmatched** rather than guessed. The list is computed on demand because **the commit that changes `decklist.txt` receives its sha after anything written in the same commit**, so a generated `versions.json` would be one behind forever (the viz history viewer will get its copy from a deploy-time step). Tags live in authored `deck_versions.json`. `restore V4` is a dry run without `--write`.
- **A version NUMBER is mechanical; a version TAG is the judgement, and the tiers are about capability rather than card count** (`docs/pilot.md`, *What a version bump means*). PATCH = mana only, MINOR = the deck can do something it could not, MAJOR = a different strategy or commander — so twelve land swaps are a patch and one enchantment can be a minor. It matches what the bench measures: a mana change moves `mana-analysis` and the goldfish curve and leaves `engine.json` alone, and **a bump that does not move the artifact its tier implies is probably the wrong tier**. **Every slug starts at `v1.0.0`** — Zask → Blech → Hapatra is a lineage of cardboard, not versions of one deck, and every key in the repo is the slug. Three guards, each of which was a real defect: releases sort NUMERICALLY (plain lexical puts **`v1.10.0` before `v1.9.0`**, and ten minor bumps is an ordinary year), a name that is nothing but digits and dots must be a well-formed release (`v1.2` is refused rather than filed as a nickname that sorts alphabetically among the real ones), and re-tagging a name at a different version needs `--force` because a tag is a claim about one exact list. The near-miss rule reads the WHOLE name — a first cut matched `^v?\d` and refused `3rd-rebuild`. The tier stays the pilot's call: proposing it needs an arbitrary-version diff, `quantity_changes` carried into `versions()` (`history()` computes it, `versions()` drops it, so 36→37 Forests reads as no change), and a classifier reporting **evidence, never intent**.
- **Forge is the simulation engine; a run is SEEDED (the first one was SAMPLED, and says so).** `manamap pilot simulate <slug> --vs <opp>… --games N` converts each seat's `decklist.txt` to a Forge `.dck` through the repo's own parser, runs N Commander games across J JVMs headless, and writes one tracked record (`sim/<run-id>.json`: outcomes, `round` AND `global_turn` — Forge's `Game Outcome: Turn N` is the winner's own turn count, not the game's — `won_by`, Forge/Java versions, every seat's decklist sha, wall time, and the assumptions incl. Forge's own AI caveat verbatim). Forge has no seed: identical runs diverge, so the tier is ◆ *sampled* and a second identical sample is a second file. Logs are gitignored. The engine lives at `~/.mana-map/forge/` (`MANAMAP_FORGE_HOME`), decks in Forge's userdata Commander folder (`MANAMAP_FORGE_DECKS_DIR`) because the documented `-D` override did not take effect. `pytest -m forge` plays one real game; the default suite does not. **`-s` exists** (found in Forge's source after the spike): identical seeds reproduce logs byte for byte, so runs are ◆ *seeded* — the default seed derives from the configuration (the default REPLAYS; `--seed` is a new sample), job *i* runs `seed_base + i`, and a same-id re-run is refused without `--force`. **`experiment <slug> --a <ref> --b <ref> --vs …` is the controlled A/B** — same table, N per arm, one artifact with both arms' figures, the delta, and a **`ci95_diff` on every figure** with `excludes_zero` beside it — an interval on the DIFFERENCE (Newcombe for proportions, Welch for means), never a comparison of two marginal intervals. `intervals_overlap` was deleted: non-overlap implies a difference, but overlap implies nothing at all, because two marginal intervals can overlap while the interval on their difference excludes zero. A `power` block reports the design's MDE, so an uninformative result says so instead of reading as no effect; same seeds are NOT paired games (a changed list changes every shuffle — the control is N) and an A/A is refused with the reason. **`sim-scenario <slug> <run> --game G --turn T --step S` is the bridge**: it lifts one board at a CR step into a `game_state` v2 scenario (`pilot/game_state.py` holds the vocabulary; `validate-stack` and `scenario-facts` read v2) — life and lands exact, cast permanents from resolve lines, tokens from first use, a commander's logged exit read as `command`, hand an estimate, every approximation in `extras.reconstruction_notes`, `question` empty on purpose. **The doctor reads the table** — `sim:runs` is an input to `deck-diagnosis` and `prescription:<id>`, and the charter says cite a sim figure WITH its interval and N and the AI caveat or not at all. **The chain has run once for real** — radagast stack 008 is a board lifted from a simulated game, resolved and checker-passed in three iterations; the resolver's damage assignment matched Forge's log line for line, and the checker caught two triggers (Scrawling Crawler, Bloodthirsty Conqueror) the author missed that the log confirms. **The parser (`sim/parse.py`) reports tokens two honest ways** — `token_resolutions` (creation abilities that resolved; blind to X and doubling) and `tokens_observed` (distinct ids that attacked/blocked/dealt combat damage; a token that sat on the board is invisible to the log) — because Forge names a token on first USE, never on creation; seat attribution is learned from assignment/land lines and `eliminated_by` is null when the source was never seen acting. `validate-sim` re-derives `analysis` from the logs where they exist and form-checks where they do not. `docs/simulation.md` has the spike, the verdict and S1–S5.
- **A mean is not a result — `mean_ci` carries median, min and max beside it.** A mean over a skewed sample is a true number that describes no game. Measured on kianne's V1-vs-V2 experiment: arm B's per-game commander damage was `0 0 0 0 0 0 0 0 0 0 31 178` — **mean 17.42 against V1's 2.25**, which reads as a sevenfold win and was one reading away from being reported as one. The **median is 0 in both arms**; the whole difference is two games, one a blowout, and the deck connected in FEWER games after the change. The `ci95` of `[-11.64, 46.47]` already spanned zero, so the record was honest and the interval discipline worked — but it took sorting the per-game values in a throwaway script to see it, and `compact()` had been writing those scalars into `doc["games"]` all along. Adding a key to the analysis means re-deriving every run with `--analyze`, because `validate-sim` compares the record against its logs.
- **Scryfall leaves `mana_cost` EMPTY on transform/MDFC and holds BOTH halves on adventure/split — so `card["mana_cost"]` is wrong in two opposite directions.** `common.front_field(card, key)` is the one reader; it replaced `deck_facts._front`, which had solved this for COLOURS and never for PIPS, which is exactly how the two halves of one bug drift apart. Before the fix `manabase.pip_requirements` counted **zero** pips for every double-faced spell and **double** for every adventure — gishath netted to zero on green because Huatli gained a pip while Monster Manual lost a phantom one, which is why the fleet deltas do not all move one way. The finding it produced: **heliod's commander is `{2}{W}{W}` and that second pip was invisible**, so its Karsten target read 22 when it should read 36 — short by 16 against 20 sources, not by 5, with white rather than blue the binding colour. Seven `mana_analysis.json` regenerated, and a diagnosis that had cited the old figures correctly then failed its own gate. **`build_deck.castability` still has the same defect.**
- **A diagnosis can go stale without its decklist moving.** `validate_diagnosis` re-derives every axis figure from `deck_audit`, so when the DFC fix changed the audit underneath heliod's `diagnosis.json` the gate failed — correctly. But there is no staleness *class* for "the measurement code moved" the way there is for "an older decklist" (`validate_prescription`'s form-only path), and the artifact records no code version, so the two are indistinguishable from the file alone. The only honest route is a re-spawn. Do not hand-patch an agent's prose to make a gate green.
- **`VALIDATED` and `STAGES` are different lists, and `deck-status` must walk both.** Three artifacts — `diagnosis.json`, `build_plan.json`, `deck_recon.json` — have validators but no lifecycle stage, so the status loop never reached them: the fleet view reported **"0 failing a gate" across 11 decks in the same second** `validate-diagnosis heliod` was failing. That is the precise divergence the `VALIDATED` map was extracted from the test suite to end, reappearing through the other door. They report as `GATE` rows now, **excluded from the stage count** (counting them made a deck with more evidence read as less finished — 13/15 became 13/17), and a failure names the artifact rather than `—`.
- **A browser cannot list a directory, so `data/decks/index.json` names the files.** `stacks/` was named there from the start for exactly that reason and four other directories of keyed instances were not — which is why the deck page could show eight panels and no simulation, no experiment, no prescription. Not because those artifacts were missing or gitignored: every one is tracked and fetchable, and nothing told the page their names. `build_index.gather_entries` now emits `sim_runs`, `experiments`, `prescriptions`, `decision_files` and a `has` block of presence flags — the flags matter because `getJSON` swallows a 404 to `null`, so "no artifact" and "request failed" were otherwise indistinguishable.
- **`info.json` is the deck page's data model and the one committed artifact composed from every other one.** `deck-info --write` emits it. That reverses the module's "never committed" rule, so it is staleness-gated by RECOMPUTATION rather than by a stamp — it goes stale when any input moves and it stamps nothing. It **omits the version block**: versions are a git walk, and the commit that changes `decklist.txt` gets its sha after anything written in the same commit, so a committed version number is one behind forever and a wrong version is worse than an absent one when the log stamps games against it. That split is forced rather than chosen — `deck_versions` imports only git, `deck_audit` reads the gitignored corpus, so CI can build one and not the other.
- **AN ABSENT SECTION MUST SAY WHAT IT IS AND HOW TO GET IT — a new deck was indistinguishable from a broken one.** Every dossier panel opened `if (!x) return ''`, so a missing artifact made the whole section VANISH: `zur-enchantress` rendered nine fewer panels than `radagast` and said nothing about the difference, on the surface whose job is telling you where a deck stands. The fix is NOT a stage→command lookup in JavaScript — that sequence is the thing `deck_status.STAGES` exists to be the single statement of, and a copy in the frontend is a second one free to drift. **`how` is a field on the registry**, carried through `deck-info --write` as `status.todo` (`{stage, what, how}`), and both surfaces render what they are given: the dossier draws a muted panel with a copyable command, the workbench draws a **new — N of 15, nothing measured yet** chip *only when a card would otherwise be blank*, since a deck built this afternoon and one nobody has touched in six months both rendered as a title and no chips. An agent-advanced stage names its SKILL (`/analyze-engine`) and an authored one says AUTHORED, because naming a subcommand that cannot write an engine model is worse than admitting there isn't one. **`sim` is a todo but NOT a stage**: adding it to `STAGES` would change the denominator for eleven decks at once and mark nine newly incomplete for a measurement that is optional and costs 45 minutes. Adding the block made every tracked `info.json` stale — the freshness gate caught it, and all thirteen were regenerated.
- **A BUTTON MAY RUN WHAT COSTS NOTHING TO THINK ABOUT — `serve.MEASURES` is the allow-list, and what it EXCLUDES is the argument for it.** `deck/measure` runs `bracket-check`, `deck-map`, `goldfish` and `mana-analysis` from the dossier: deterministic, no model call, one artifact each. Absent by design: `simulate` (45–62 min of Forge), every agent loop (cheapest routine 54.5k tokens, `candidate-pool` 235k), and the AUTHORED files (`goldfish_targets.json`, `issue.json`) — those are judgements a person makes, which is the whole reason a command travels with a stage: some of them are for reading, not for pressing. A test asserts the list by MODULE PATH, since a key can be renamed into innocence. **The `info.json` refresh is part of the endpoint, not a nicety**: `info.json` is composed from every other artifact, so a measurement leaves it stale by construction and the page renders `info.json` — without the re-emit the pilot presses a button, the command succeeds, and the dossier goes on saying the thing is missing. **`needs` is reported rather than assumed** (mana needs goldfish; goldfish needs the authored declaration), because a button that fails on an unstated dependency is worse than one that explains itself. The page RELOADS on success rather than patching the live object — two seconds of work is not the place to invent a cache-coherence problem against a fifteen-artifact composition. Measured end to end, including the refresh: bracket 2.3s, map 1.9s, of which `deck-info --write` is 1513ms — **the refresh is the larger half**, worth knowing before anyone tries to make this feel instant by trimming the command. And the first version of that comment quoted ~200ms from a timing loop whose runs had all FAILED: a measurement of a command that did not do the work is not a measurement of the command.
- **A PAGE MAY DRAFT AN AUTHORED FILE; IT MAY NOT AUTHOR ONE — `manamap pilot scaffold-targets <slug>` and `serve.SCAFFOLDS`.** `goldfish_targets.json` stays authored (a component is a claim about what the deck is trying to do), but the dossier's note was a path and a schema nobody had seen, with the goldfish and mana panels blocked behind it — so the pilot's first act was inventing a JSON shape from nothing. The draft derives targets from **contained combo lines** (`combo_details`, one `any_of` leg per card, because `need` is an AND of ORs and one group of two would say either card alone assembles it) and from **role axes**, which `validate_goldfish_targets` already records are NOT what a component is. **It never names a WIN LINE** — that is the defect the validator exists to catch (heliod's Hullbreaker Horror and ur-dragon's Aggravated Assault are each in two passing stacks and no component), and a machine guessing at one manufactures exactly the claim being demanded; finishers are stated as a GROUP, never a LINE, asserted by test. The `DECK_ROLE_BUDGET` trap is designed around rather than hoped away: the file carries `"scaffolded": true`, every group carries `_from`, and **`validate-goldfish-targets` reports an unedited draft on every run** — reported, never failed, because a gate that reddens a legitimate intermediate state teaches its reader to ignore it. **`BROAD_GROUP = 21` is measured and replaced its own opposite**: across 113 authored groups on 10 decks the median is 5 and the **max is 20**, so 21 means "wider than anything anyone has declared" — while the THIN warning at 3 that was written first would have fired on **31 of 113, 22 of them deliberate size-1 declarations** that a component has no backup. A test re-derives both from the fleet, so the constant cannot outlive its evidence. **`issue` is deliberately undraftable**: its live keys are a deck's NAME and whether it is SLEEVED, which is a fact about cardboard no command can derive — the class of claim the rehearsal locks were withdrawn for. Measured on zur-enchantress: **3/15 → 8/15**, draft → goldfish → mana, each step unblocking the next from the page.
- **RESUMING A DRAFT REPLACES YOUR LIBRARY, so it must read before it clears.** `resumeDraft` ran `Session.library.clear()` and only then looped `(brief && brief.must_include) || []`. A 404 is safe by luck — `build.js`'s `getJSON` **throws** on `!res.ok` so `.catch` fires first — but a brief that LOADS with no `must_include` is not, and that is the documented minimum shape (`{slug, commander, bracket}`, which `brew --commander` scaffolds). It emptied the library and reported **"0 card(s) kept"** as though that were the draft's content. Two things this cost: the first diagnosis blamed `getJSON` swallowing a 404 to `null` — true of `deck-view.js`'s copy, false of `build.js`'s, and **two helpers with one name and opposite failure semantics is its own hazard**; and the negative control passed three times against the bug because `page.route("**/…/brief.json")` does not match a cache-busted `brief.json?v=150`, so the stub never applied and the test measured the 404 path while believing it measured the other — **the same glob trap already on this list**. `Discovery.library.clear` was removed in the same pass: its only caller was the tray's unconfirmed Clear button, and leaving a one-call wipe exported is how the confirming version gets bypassed by the convenient one.
- **THE OPPONENT-GATED BLIND SPOT IS NARROWER THAN I CLAIMED, AND CLOSING IT REVERSED A RECOMMENDATION.** A solitaire goldfish has no opponents, so a card taxing what THEY do is worth zero in every figure — that is a boundary, not a gap, and building an opponent inside the goldfish would make it a worse Forge. But the question a pilot actually asks is narrower and IS answerable: **how often would this have triggered against my table?** Forge played 13,456 opponent turns and kept the record, so `sim/pod_behaviour.py` reads the tracked runs instead of inventing behaviour. Measured: the pod casts **1.09 spells per turn** and reaches a SECOND spell on **23.2%** of them — so a second-spell trigger fires **~0.7 times a round** against three opponents where a per-draw trigger fires **3.0** by rule (one draw step each, no measurement needed). **Four times apart, and the cheap card is the weak one**: I had recommended Monologue Tax (mv3) and Gleaming Splendor (mv2) over Smothering Tithe's effect on price, and no amount of reading the cards gets to that ratio. A SECOND DRAW is **bounded rather than estimated** — Forge logs no draw events, so `rate_for` returns a stated bound and never a number, the absent-not-zeroed rule one module over. The constants are re-derived from the logs by a test where they exist and skipped where they do not, since logs are gitignored; the estimate is a FREQUENCY and never a value, because what a trigger is worth is the pilot's call and the model has no business having an opinion about it. `assess`'s verdict keeps the disclaimer AND carries the rate — dropping the disclaimer would let a Forge-derived frequency read as a goldfish figure. **RUNNING IT ON THE REAL PILE FOUND TWO DEFECTS IN IT, AND THE FIRST IS THIS REPO'S OWN RECURRING ONE.** (1) The opponent gate matched `each opponent` ANYWHERE — the wording every drain payoff in the game uses — so it classed **8 of the branch's 95 cards** as gated on the pod and **5 were wrong**, including Reckless Fireweaver, which is the branch's OWN drain (gated on your artifacts entering) and Exotic Orchard, which is a land. A card is opponent-gated only when THEY act first; the pattern requires agency now, and the branch reads 2 of 95, both real. (2) **A FREQUENCY IS NOT THROUGHPUT when one firing touches every seat**: Master of Ceremonies fires on one upkeep against Smothering Tithe's three draw steps and read as a third the card, when each firing resolves against each opponent and the throughput is comparable — `scales_with_opponents` rides with the number and the verdict says it. The `basis` string is per-pattern for the same reason: quoting the measured spell rate under an upkeep trigger cites evidence that had no part in the answer, so a reader cannot tell which figures are load-bearing.
- **`--identity` takes letters; `parse_color_identity` splits on commas.** It is correct for `cards.csv`'s `"G, U"` and returns `{"GU"}` — one two-character token no coloured card's identity can be a subset of — for the compact form a human types and the help text advertises. `card_search --identity GU` therefore returned only COLOURLESS cards while printing "identity GU" in its header. `parse_identity_arg` is the CLI parser; a non-colour letter is a hard error, because a typo must not narrow a search invisibly. Same shape as the bug `card_pool._build_pool` already records in a comment, reintroduced one module over.
- **Ownership means a BOX. `pilot/collection.py` is the only reader of `COLLECTION_DIR`.** It shipped defaulting to "in a box OR sleeved in a tracked deck" — unsleeving is a decision, not a purchase — and `validate-recon` refuted that within the hour: `data/decks/` holds BUILD PLANS as well as assembled decks and nothing tells them apart, so counting kinnan's whole-format baseline made 99 unowned cards read as owned. `include_decks=True` survives for a caller who knows what it contains; no gate uses it. The module replaced two incompatible parsers over the same nine files (`deck_history._owned_index`, membership and both faces and unresolved; `pool_facts.read_sources`, copies and front-face-mapped and resolved), and is memoized on a signature over every file AND the directory listing — `mtime_memo` keys on one path, and a NEW `.txt` changes the answer while every existing file's mtime is untouched.
- **What a deck IS is a parameter now — `pilot/formats.py`** (PRD-v1 §13: constraints as parameters, "cheap if designed in now, expensive later"). One format ships and the point is not the second one; it is that the rules have a name and a home. **FOUR places independently said how big a Commander deck is**: `config.DECK_SIZE`, `check_in`'s OWN `DECK_SIZE` shadowing it, `manabase.DECK_SIZE_AFTER_COMMANDER` (the same fact minus one), and a bare `if total != 100` in `validate_deck` that no shared constant could reach. None was wrong — that is what makes it the expensive kind of duplication, since nothing breaks until one has to change and three do not. **`library_size` is DERIVED**, because `100 - 1` written twice is two things to change and one will be missed. **The split is legality vs shape**: size, singleton, commander count and colour identity moved to `formats`; `DECK_ROLE_BUDGET` and the curve targets stay in `config`, because a 60-card deck is legal at any curve and mixing "what is allowed" with "what is good" turns a format table into a strategy opinion. The layering settles ownership — everything imports `config`, including `pilot`, so `config` cannot import `formats`. **Pendragon is deliberately absent**: the PRD flags its own description as unverified, and a spec that encodes a guess is worse than one that omits it, because the guess goes invisible the moment it is in a table. An unknown format name is an ERROR, never a fallback to Commander. The literal ban is asserted through the **AST**, not a regex — a first cut flagged "100" inside a docstring, which is prose about the format and exactly what those modules should say.
- **FIVE formats, and three things the obvious reading gets wrong.** Commander plus Standard/Modern/Pioneer/Pauper. The pool cost nothing — `extract` already writes eight `legal_<format>` columns from Scryfall, so a format's pool is a column lookup and not a rule to reimplement (Standard 4,887 legal, Pauper 10,793, Pioneer 14,817, Modern 22,450, Commander 31,830). **60 is a MINIMUM** — "at least sixty cards", so a 63-card Modern deck is legal and an exact check would reject legal decks while looking rigorous (`exact_size`). **Pauper is NOT commons-only**, which the PRD-v1 §13 says it is: Scryfall disagrees for **373 cards** because a card printed at common anywhere qualifies (Theft of Dreams is uncommon and legal), so a rarity filter would look stricter and be wrong 373 times. **A PROMO PRINTING MADE A STAPLE ILLEGAL**: adding the legality check failed ur-dragon and radagast on their own tracked decklists over Savage Lands, because `cards.csv` holds two rows — an `fmsc` promo marked `not_legal` sorting before the legal `msc` one — and first-printing-wins is right for IDENTITY and wrong for legality. The combining rule was measured: **16 names disagree across printings and every disagreement is exactly {legal, not_legal}**, `banned` never co-occurring, so "any legal printing wins" is unambiguous; `banned` is still tested first (unreachable today, correct if it stops being) and a test fails if the assumption breaks. Legality is read ON DEMAND through `card_pool` and memoized with `mtime_memo` — widening `CORPUS_COLUMNS` would erode its ~30%-of-parse-time argument a column at a time, and a bare dict never invalidates after a corpus refresh.
- **Harvesting from a reference deck is `?ref=<slug>`, and it must NOT load as your deck.** PRD-v1 §6.1 steps 9–10, and almost no new machinery: `commander-search … --open N` writes `data/reference/<slug>.json` (gitignored — someone else's list, scratch for one brew, re-fetchable in a command) and prints the Atlas URL; the page seeds the graph from it exactly as `?cards=` does, and the Keep button that every card panel has carried since the library shipped does step 10. **The distinction is the whole point**: passing `opts.deck` would ring a commander, ink the cards as yours and put all eighty in your library — the opposite of harvesting a few out of another brew. The CLI fetches because the page cannot: the deployed site is static and makes no service calls, so a missing reference file **says so and names the command that writes one**, rather than doing nothing.
- **Archetypes are DATA, not research — and the raw role histograms are too similar to act on.** PRD-v1 §7.2 describes an agent researching how a commander is built; EDHREC's `taglinks` panel already IS that list with a deck count each, and it answers Zur with exactly the split the PRD names as its worked example (Enchantress 1201, Auras 736, Stax 542, Control 529, Combo 380, Voltron 361). `average-decks/<commander>/<theme>` then gives that style's own deck, so §7.3's role template is DERIVED — a role histogram over the decks people actually built in that style — rather than authored. That fixes the `DECK_ROLE_BUDGET` defect already on the record: one flat budget for every deck, its own comment calling it PROVISIONAL. **But the histograms barely differ: 0.955–0.978 cosine across Zur's styles**, because every build runs the same signets, lands and removal and that shared bulk swamps the part that varies. So the template ships WITH a delta against the commander's own baseline, and the delta is the half worth acting on — voltron comes back **+7 `buff:attached`, +3 `protection:self`, +2 `wincon:combat`, −2 `wincon:alt`**, which is voltron described exactly, and stax comes back **+2 `stax`, +1 `removal:tax`, −4 `value:etb`**. §7.2's no-ranking rule is kept mechanically: EDHREC's order is preserved (re-sorting would BE a ranking), the count travels as data, and a test scans the payload for recommendation language — **scanning the whole document instead fired on the module's own disclaimer** ("nothing here recommends a style"), which is a check failing on correct output. A style under 50 decks is still computed and flagged, because a histogram over eleven decks describes eleven decks. And `--limit` is a DISPLAY concern: resolving `--theme` against the truncated list made `--theme voltron --limit 3` report that Zur has no voltron decks — false, confidently phrased, and a display flag silently changing a lookup's answer.
- **A STYLE shapes the role budget, and the shaping is renormalised, not copied.** PRD-v1 §7.4: `brew <slug> --commander … --theme voltron --build` scaffolds `brief.json` from the cards you kept (they become `must_include` — the promise that they are in the 99), and `build_deck.role_budget_for` reads the archetype's role histogram instead of the flat `DECK_ROLE_BUDGET`. It collapses through **`role_group`, the builder's OWN mapping**, so the budget and the filling cannot disagree about what counts as ramp. **Shaped, not copied**: the measured histogram is renormalised onto the provisional budget's total, because that total is what land counts, the curve quota and the bracket pass are all sized against — taking EDHREC's raw counts would change the deck's SIZE as a side effect of choosing a style, and a test asserts the totals match. Result on Zur: voltron protection 6 against the flat 3, stax ramp 12, and all three styles want ~5 draw where the flat budget says 10 — Zur decks draw off the engine, not off ten dedicated spells. **The bug worth remembering: lands were being counted into the SPELL pool.** `role_group` has no land line, so every `land:basic`/`land:tapped`/`land:utility` fell into `flex` — 32 of voltron's 135 role-copies — inflating flex and deflating every real line in one stroke, while `land_counts` was budgeting the mana base separately. A test now fails if flex exceeds 55% of the spells. An unfetchable theme **falls back to the flat budget with a reason in `role_budget_grounding`** rather than raising, and the grounding string is how a reader tells a measured budget from a provisional one. **A new deck is NOT sleeved**: no `paper` block is written, because 0.x is the version of a list that exists only digitally and reaching 1.0.0 is the act of sleeving it — writing one here would claim cardboard that does not exist, which is what the rehearsal locks were withdrawn for. `scaffold_brief` uses `DECKS_DIR / slug` rather than `deck_dir(slug)`: that helper is a READER and refuses a directory that does not exist yet, which is right for every other caller and exactly wrong for the one creating a deck.
- **The deterministic builder fills a curve SHAPE and finishes combo lines.** It used to score every card independently and take the top N, and `curve_fit` penalises each point above `DECK_CURVE_SWEET_SPOT = 3` — so the top N were always cheap: kinnan's first baseline was 64 nonland cards with **nothing above mana value 3**, 29 of them mana producers, and `validate_build` passed it because it checks form. N independent maxima are not a distribution. `fill_slots` now crosses its role quota with a mana-value quota derived from `DECK_AXIS_TARGETS["curve"]` — the cited target `deck_audit` already measured against and the builder never read, so no new uncited constant — with a second pass on score alone when a role's shape-fitting candidates run out, because a legal deck of the right SIZE beats a perfect curve. And `complete_combos` reads REAL LINES from `combo_details`, never the flat `combo_partners` map: that is co-occurrence, so "partners with something present" is true of a hundred cards once the commander is on the list and cannot tell a completion from a coincidence. kinnan went from 23 partners and 0 completions to 4 contained combos and 2 two-card infinites.
- **Commander damage is measured PER DEFENDER, and the commander's name rides in the record.** CR 903.10a asks for 21 combat damage from one commander on one *player*, which is the only win condition some decks have — and `combat_damage_dealt_to_players` sums every source and every seat at once, so a commander that hit three opponents for 20 each was indistinguishable from one that hit a single seat for 60 and killed them. `analysis.seats[*].commander_damage` reports `dealt_total` and `max_on_one_defender` separately because they answer different questions. **Combat only** — a commander dealing noncombat damage does not count toward 903.10a, or a Purphoros deck reads as closing on damage it can never deal. The Forge log never names a commander, so the names come from the decklists **once**, at run or `--analyze` time, and are written into `seats[].commander`: re-derivation must depend on the record and its logs alone, because a disk lookup at validate time makes a later commander swap read as parser drift on a run that was correct when it was made, and gives every pre-existing record a block its stored analysis lacks — turning the whole fleet red at once. A record without the field re-derives exactly as before; that is what made this migration safe, and `--analyze` is the migration. Unknown commander ⇒ the block is **absent, not zeroed** (same contract as `model_treasures`): "dealt 0" is a measurement nobody made.
- **A prescription is a diagnosis scoped to ONE question, and it accumulates.** `manamap pilot prescribe <slug> "…"` opens `prescriptions/<id>-….json` with the authored half (`prompt`, `id` = sha of the normalized prompt, `as_of_decklist_sha256`); `deck-doctor` MODE prescribe + `deck-skeptic` fill the answered half through `prescribe --merge <id>` (answer keys only — the prompt is never touched); `validate_prescription` REUSES `validate_diagnosis`'s cut/add/bracket/axis-movement/citation functions rather than copying them. `add_candidates` is ranked and capped at ten — The Short List's rule relocated; `short-list-analyst` and `the-ten` are retired and the surviving `considering.json` are frozen legacy — unregenerable, since the agent is gone and the validator demands exactly ten entries, so a deck whose pilot ACTS on the advice can only delete the file. Edgar did: Mirkwood Bats was one of its ten and is now in the 99. Do not restate how many remain; the count only goes down. The cache routine `prescription:<id>` digests only the prompt (`prompt:self`, the same trick as `scenario:self`) and `cache-record` refuses a file without a passing skeptic. **Stale is not wrong**: one written against an older decklist is form-checked only, because a cut since applied would read "not in the maindeck" forever and a gate that reddens history teaches its reader to ignore it. Both doctor modes now read `log_annotations.json` (`deck-diagnosis` declares it optional).
- **`targeting` is the first measurement here about the OPPONENTS' choices, and its NAME carries its caveat.** Everything else measures our deck; this measures the pod's target selection, which is the input half of a game-theoretic argument rather than game theory itself. **The unit is a decision, not a game** — one declare-attackers step with ≥2 living opponents — which is what makes twelve games answerable: they hold hundreds of decisions. A forced choice is not a choice, so a 1v1 run contributes zero BY CONSTRUCTION rather than by being filtered; and it is not per attacking creature, since five creatures go at the same player and would multiply the sample without adding one independent choice. **Ties count the whole tied set** and raise that decision's null accordingly — every seat is at 40 early, and breaking that tie arbitrarily manufactures signal from nothing. Both decks with logs agree: biggest revealed threat 0.685 (radagast, 444 decisions) and 0.675 (kianne, 335) against uniform 0.54/0.52, p=0.0001 — **but** on the 196 decisions where "biggest threat" and "easiest kill" name different seats it is 0.541 vs 0.403 with overlapping intervals, and that honest null ships beside the result. **Strength is REVEALED combat damage, never board power**: printed P/T is parseable (`bridge.py:73`) but counters, anthems, auras, equipment and token counts are invisible, which would bias a power ranking against token, counter and anthem decks — most of this pod. The artifact key is `forge_ai_targeting_policy`, not `archenemy_tax`, because a key cannot be paraphrased away by a slide; `limits[]` carries `FORGE_AI_CAVEAT` as the imported constant. It lands BESIDE `parse.aggregate`, never inside, so no tracked run record moves.
- **`deck-recon` is the one routine whose staleness is TIME, not inputs.** A decklist edit does not change what strong lists for that commander run, so hashing `cards.json` there buys a web pass on every swap. Its declared input is `deck:brief.json?`; `RECON_MAX_AGE_DAYS` is judged by the SKILL, because `deck_audit` is deterministic and never reads the clock. `deck_recon.json` stays **out of `strategy.md`** — durable theory and perishable meta claims must invalidate differently, the lesson recorded when `meta-analyst` was traded away.
- **Agent cache**: subagent spawns are the only LLM cost that lands in this cache (no *pilot command* calls a model; `serve.py`'s `ask` bridge and `mm ask` are opt-in exceptions and are not cached here). Always `cache-status` before spawning, `cache-record` **after** validating. Editing a `.claude/agents/*.md` prompt invalidates that agent's routines by design, and editing **`.claude/agents-common.md`** — the shared contract every pilot charter opens by reading (read-only, `deck-facts` first, `--out`, the evidence ladder, enumerate-before-superlative, partial revision, output handoff) — invalidates every routine, because it is hashed with every agent prompt. It replaced ~1,000 lines pasted across twelve charters; it lives *beside* `agents/`, not inside, because Claude Code loads `.claude/agents/*.md` as agents and a test counts them; `build-manual` is deliberately uncached. Costs and per-routine sizing: `docs/agent-cost.md`.
- **Strategy DB staleness**: any edit to `data/strategy/strategy.md` requires `manamap pilot build-strategy-db` — `load_strategy_db` hard-errors on a sha256 mismatch. Doc + CHANGELOG are tracked; the derived index/embeddings are gitignored.
- **The combo data is two files**: `combo_graph.json` is `{"partners": {...}}` **only** — it's what the viz fetches on the main thread, so nothing else belongs in it. The per-combo records live in `combo_details.json` (`{combos, by_card, meta}`, Python/agents only) with `bracket`, `mana_value_needed`, `popularity`. If you remember a `combos` key on the graph, that moved. Sizes in `docs/data-artifacts.md`.
- **Combo data is format-agnostic**: Commander Spellbook combos may assume a card is your commander ("Infinite commander casts" in `produces` is the tell) — verify lines with a resolve-stack run before presenting them as fact (stack 004 refuted one this way; `bracket.py` now excludes such lines automatically). Their per-combo `bracket` tag is also not gospel: it tags a real Hapatra two-card infinite as bracket 1, which is why the engine runs its own infinite test.
- **Building from a box of cards is `pool-facts`, not `deck-facts`.** A collection is not a deck: `deck-facts` on 764 cards reports hypergeometrics against a 99-card library (`manabase` is hardwired to one) and `validate-deck` emits ~1000 errors — both answer a question nobody asked. `manamap pilot pool-facts <paths>` takes files or directories instead of a slug, deliberately, so a collection never lands in `data/decks/<slug>/` where the deck validators could reach it. **Do not commit its output**, same rule as `deck-facts`.
  - **Depth is not castability, and only one of them is obvious.** Depth (owned cards inside a commander's colour identity) ranked Atraxa first at 663 on the first real collection — and Atraxa's W and U have 10 sources each against B's 44. A shortlist ranked on depth recommends a deck that cannot cast its spells, confidently. `pool_facts` reports both and its `notes[]` names any identity where the two disagree.
  - **Count sources with `manabase.land_colors`, never by grepping the oracle text.** A hand-rolled count (`{U}` appears in the text, or the type line says Island) put the same collection at **1 blue source**; the real number is **10**. In a bulk collection the fixing is overwhelmingly generic — Command Tower, City of Brass, Exotic Orchard, Path of Ancestry, Ash Barrens, tri-lands — none of which name a colour. The error is a factor of ten and it points the wrong way: it kills an archetype that was actually live. `land_colors` is also restriction-aware, so it won't count Haven of the Spirit Dragon as five sources.
  - **`combo_details.json` holds several records per interaction**, so containment counts double: dedupe on `frozenset(cards)` or a box reports 33 lines where it has 31.
- **The two raw dumps are stored gzipped.** `oracle-cards.json.gz` and `combos_raw.json.gz` were 600 MB of the 871 MB in `data/`; JSON compresses ~8.5x, so they are now 56 MB and `data/` is 326 MB. Each is written by one downloader and read by one extractor — so this is `ingest/common.open_dump` and nothing else. **Scryfall's bulk files became gzipped JSONL in Aug 2026** (`jsonl_download_uri`; the single-array `download_uri` is gone): the downloader writes already-gzipped payloads verbatim rather than double-gzipping, and `extract` sniffs the first character ('[' = legacy array, else JSONL), so both generations of dump parse. **Reading falls back to an uncompressed sibling**, so a repo that still has the plain file keeps working with no migration and no re-download; **writing deletes the sibling**, or a re-download would leave both and cost space instead of saving it. RAM is deliberately unchanged: gzip streams the decompress, `json.load` still materialises the whole document, and the peak is identical.
- **`bracket-check` needs a pipeline run**: the Game Changers signal is the `game_changer` column in `cards.csv`, which is gitignored. A fresh clone can render manuals but cannot compute a bracket floor until `manamap extract` has run.
- **`deck-facts` first, always**: `manamap pilot deck-facts <slug>` is the deterministic brief — DFC-correct colours, curve, pip load, role coverage and holes, contained combos, and a `notes[]` block naming the traps. It is computed on demand and never committed (same rule as `artist-credits`). Every deck agent is told to run it before deriving anything; re-deriving by hand costs tokens and has produced wrong answers.
- **One rules domain per scenario.** The checker's verdict is atomic over the whole artifact, so citation count predicts iterations — small scenarios pass, big ones fail and take correct answers down with them. `RESOLVE_SCOPE_BUDGET` warns; `validate-stack --scenario-only` preflights for free before any spawn. The measured evidence is in `docs/pilot.md`.
- **`--out` on a per-deck command is slug-scoped, and that is enforced.** Concurrent deck agents share one scratchpad directory and were writing views to generic names (`audit.json`, `aud.json`, `audit2.json`); the second write silently replaced the first, so an agent working heliod read goblin-storm's numbers **under a heliod invocation** and reasoned from them. It hit seven agents across two sessions and every catch was someone noticing an implausible figure — which is not a control, because the wrong deck's numbers are exactly the shape the reader expects. `common.resolve_out_path` now takes a **directory** and auto-names `<command>-<slug>.json`, and **refuses** an explicit path whose filename omits the slug. A shell redirect (`> audit.json`) cannot be policed and must not be used for per-deck data. The CLI itself is concurrency-safe — 8 decks audited in parallel to unique paths return 8 correct slugs; the bug was always the filename.
- **Agents hand off by path, not inline JSON**: deck agents write `data/decks/<slug>/.agent-out/<agent>.json` (gitignored) and return the path plus a summary; the orchestrator validates and merges. Returning a large artifact inline burns context for nothing.
- **Count copies, not decklist entries**: `cards.json` stores basics as one entry with `quantity: N`, so anything the shuffler would see (land totals, colour sources, hypergeometric draws) must go through `common.expand_copies()`. Counting entries once published "18 lands" for a 33-land deck and understated every colour fleet-wide. `mana_analysis` reports `lands.total` (copies) beside `lands.entries` (distinct cards); `validate-issue` lints prose that quotes the entry count as a land count. `artist_credits` counts entries **on purpose** — authorship is per card.

- **THE STANDARD BENCHMARK RUNS ITS OWN CONFIGURATION, and the reason is that the fleet's own goldfish figures are NOT comparable.** PRD-v1 §9.2: "uncontrolled sim output cannot be aggregated into a ranking." Of twelve decks with a 99, **exactly one** opts into `model_combat`/`model_treasures` — so ranking off each deck's tracked `goldfish_metrics.json` would compare one deck measured with a kill clock against eleven measured without, and `mean_bodies_by_turn` does not even mean the same thing across the two. Two decks have no declaration at all, a third's is an unedited scaffold. So `pilot/benchmark.py` freezes seed, iterations, max turn and **uniform model flags that override the declaration**, and **reads the 99 rather than the declaration** — otherwise a deck would be ranked partly on how well its pilot writes JSON, and the two decks without one could not be scored at all. `goldfish.run` grew `model_treasures`/`model_combat` overrides (None = read the declaration, the tracked behaviour) and `with_results=True` for the raw per-iteration rows; **both default off, and the rows leaking unconditionally turned two freshness tests red immediately** — they compare `run()` against the tracked artifact byte for byte, which is what they are for. Whole fleet: **30 seconds**.
- **§14.1 IS ANSWERED WITH A REFUSAL, and the refusal is the finding: DO NOT publish an aggregate score yet.** Three defects, all found by running the fleet and looking rather than by reasoning. (1) **`speed` is not archetype-neutral.** `kill_by_turn_8` ranges **0.001 to 0.405** — a 400x spread — and the bottom is heliod and hapatra, whose declared kills are "win condition access" and a two-card combo. The goldfish's combat model cannot see either, so a weighted sum including speed ranks a combo deck last **for not attacking**: an archetype filter wearing a ranking's clothes. (2) **`consistency` was speed under another name**, r = 0.78, because the first version took the spread of the kill-turn histogram — computed over the games that KILLED, so a deck killing in 0.1% of games contributed ten clustered late kills and scored as supremely consistent. Exactly backwards. It measures mana spread now (every deck, every game, nothing censored) and reads −0.08 against speed. (3) **`missed_land_drop_rate` and `mulligan_rate` correlate at 0.97** — one measurement reported twice, and two of those in one total is that quantity counted twice. `consistency` sits at 0.78 with the mana LEVEL, checked rather than assumed: a coefficient of variation moves it only to 0.78 from 0.90, so the relationship is substantive (ramp drawn and ramp not drawn are different games) and it is left as a plain stdev rather than dressed up. **`benchmark.json` is tracked and freshness-gated** — deterministic under a fixed seed, verified identical across runs — so the workbench can read it on a static host.

## RE-DERIVING A NUMBER IS NOT REPLICATING A FINDING

On 2026-09-08 heliod's engine critic reported that the deck wins more when its
commander never flips. It was checked before being believed — the cross-tab was
re-derived straight from the run's own logs and matched the critic exactly:

    never flipped   85 games   23 wins   0.271
    flipped         35 games    4 wins   0.114

It was committed as a headline finding, called "the correction that strengthens
the model", and written into the engine document.

The next day, a fresh 120-game run on the current list:

    never flipped   85 games   14 wins   0.165
    flipped         35 games    6 wins   0.171
    difference +0.0067  ci95 [-0.1230, +0.1748]  SPANS ZERO

**It was noise.** Flat, and the interval on the difference is wider than the
effect first claimed.

The verification that was done proved the ARITHMETIC — that 23/85 and 4/35 were
correctly computed from those logs. It could not prove the finding, because it
used the same sample. A number re-derived from the run that produced it will
always agree with itself; that is what re-derivation means.

The tell was available and unread: 4 wins in 35 games is a rate with an interval
about twenty points wide, and no interval was ever put on the split. Every rate
carries its interval is the rule this bench states in five places, and it was
broken on the very figure being celebrated for correcting somebody else.

So: **a cross-tab discovered inside one run is a HYPOTHESIS.** It earns the word
finding when a second sample agrees, or when its own interval excludes zero —
and this one's never did.

## A branch aimed at a MECHANISM still has to be graded on a sample that can see it

heliod's `skies-v1` and `archangel-v1` merges were built for one purpose: the
deck was dying to fliers, giada-angels having taken **39 of 53** attributed
eliminations. Fifteen swaps went in, cheap flying and reach bodies and taxers.

Two 120-game runs against the same pod, one per list:

    win rate      0.250 -> 0.200   difference -0.050  ci95 [-0.161, +0.064]  SPANS ZERO
    giada's share
    of our deaths 74%   -> 70%     difference -0.031  ci95 [-0.190, +0.134]  SPANS ZERO

**Neither moved measurably, and the second one is the interesting failure.** The
win rate not moving is unsurprising — fifteen swaps rarely shift a rate by
enough to see. But the *mechanism* did not move either, and the mechanism is
what the branch was for.

The arithmetic says why, and it is worse than it looks. Resolving a 4-point
share change needs **more than 1000 eliminations per arm** at 80% power; 259 for
10 points, 50 for 20. We observed **53 and 61**. The effect being hunted was
roughly an order of magnitude below what the sample could see, and it was
already an order of magnitude below that when the branch was opened.

So the rule is not "grade a branch on a mechanism rather than an outcome" —
that much was already known, and `deck_branch.MEMBERSHIP_AXES` exists because of
it. The rule is that **the mechanism needs its own power calculation, in the
units the mechanism is counted in.** Eliminations are not games: a 120-game run
produced fifty-odd of them, so a mechanism measured per-elimination has a
quarter of the sample the headline rate does, and nobody had noticed because
nobody had computed it.

Absolute eliminations by giada went UP, 39 to 43, while the share went down.
Both facts are inside the noise and neither is evidence. Reporting either as a
result would be the overlap fallacy one layer down from where this repo already
refuses it.

## The net change report, 2026-08-28

- **A MEASURE COMPUTED FROM AN AUTHORED FILE IS NOT EVIDENCE, HOWEVER TIGHT ITS INTERVAL — and it sat in the block a spending decision reads first.** `engine_lift` split the 10,000 iterations by whether every component marked `required` in `goldfish_targets.json` had been assembled by turn three, and compared kill rates with a Newcombe interval on the difference. Statistically it was correct. The problem is upstream of the statistics: **`goldfish_targets.json` is authored, and the same hand writes the declaration and reads the verdict.** Measured on one Ur-Dragon list, one seed, the same 10,000 games, three defensible declarations of the same 99, graded against kill-by-T8:

  | `required` set | lift | interval | |
  |---|---|---|---|
  | ramp + a loosely worded payoff | +0.007 | [−0.003, +0.017] | spans zero |
  | discount + ramp + burn, all three | **−0.036** | [−0.052, −0.020] | **REAL** |
  | ramp + burn | **+0.014** | [+0.005, +0.023] | **REAL** |

  Same cards, same games, **opposite signs**, and the middle row says at an interval excluding zero that assembling the engine makes the deck win LESS. That row is not a bug either — a second cost reducer is a card and two-to-three mana that deals no damage when eminence already pays the discount for free from the command zone, and drawing it early genuinely correlates with a slower kill. Each stage split on its own, online by T5 against kill by T8: **BURN +0.033 [+0.015, +0.051] REAL; FLOOD −0.006 [−0.027, +0.017]; DISCOUNT −0.028 [−0.044, −0.013] REAL.** Every one of those is a true statement about the games. None of them is a fact about the deck that survives someone rewording a label.

  Deleted. `deck_branch.MEMBERSHIP_AXES` now **refuses** `engine_online_3/5/8` and `any_route_8` as branch objectives, with an error naming the output axes to use instead, and a test walks every tracked `branch.json` so an old one cannot survive the guard. `goldfish_targets.json` stays — it is a good description of a deck and it drives the `*_assisted` figures and the target table, which are hypergeometric and real. What it may not do is be the thing a branch is graded on.

  **The counter-argument is real and is recorded rather than answered:** card advantage is not modelled at all, so a cost reducer is charged its card and its mana and credited only with the discount, never with the extra spell a real turn casts. The split is also correlational within one fixed list — it says games that drew a reducer early killed slower, NOT that cutting reducers makes the deck faster. `candidates` is what would say that.

- **Ur-Dragon's objective moved from `engine_online_5 >= 0.22` to `damage_8 >= 40.0` for exactly this reason.** The old one was met **4.4x over** while the lift spanned zero. The new threshold is the opponent's starting life — a number with meaning outside this branch — and v1.0.1 reads 30.81 and misses it while the branch reads 46.44. An objective should name something the deck PRODUCES.

- **Deleting a block took a validator rule with it, and nothing noticed.** "An unavailable block owes a reason" was checked on `engine_lift` alone, so removing it left `mana` and `forge` free to report `available: false` with no explanation — a blank section a reader cannot tell from a measured nothing. The rule is stated per-block now. **When you delete a producer, grep the validator for the only place its contract was enforced.**

- **Every figure carries its definition, in the report that prints it.** `net_change.METRICS` is the registry: what each row measures, why a pilot should care, the unit, and the yardstick it is only meaningful against. A test asserts it matches `ROWS` exactly **in both directions** — a rendered row with no definition prints a bare number under a heading promising definitions, and a definition for a row nobody renders is how the registry rots without failing. Two entries are load-bearing: `killed by T*` is tagged *"a CLOCK against one unblocking opponent, never a win rate"* (reading 0.318 as a win rate overestimates the deck by the size of a pod), and `noise` is defined as *"whatever happened is smaller than this run can see"* — no answer, not no change.

- **The risk block separates four kinds of bad news, because they read alike as prose and are not the same claim.** `paid` is a row that measurably fell. `unresolved` is a row inside the MDE. `unmeasured` is a swap the harness is structurally blind to, derived from the staged cards' own role heads — on the Ur-Dragon branch that is 3 counterspells, 3 protection, 1 removal, 1 draw and 6 land swaps, **eight cards whose value is in none of the nine figures above them**. `structural` is a caveat about the whole goldfish. Nine rows with intervals *read* as a full accounting and are silent about the rest; silence rendered beside a confidence interval looks like a measured zero. **The `unmeasured` line is scoped to the EFFECT, never the card** — Solphim is a `protection:self` body AND a damage doubler the combat model prices at +7 damage, and filing the whole card under "unmeasured" would understate the branch the line exists to warn about.

- **`elsewhere` in the bill was two different costs wearing one integer.** A card in a RETIRED or BROKEN-DOWN deck is loose cardboard and costs nothing to take; a card in a deck that is still sleeved costs that deck the card. Reported as one number, six read as six decks to disturb when one of them was in `sisay`, already apart. The split first shipped with `net_change.FREE_TO_RAID`, a local list of the statuses that mean "already apart" — which was itself a fourth copy of `common.UNPLAYABLE_STATUSES` and is gone; see the entry below.

## The manabase, 2026-08-28

- **A CONDITION IS SCOPED TO THE CLAUSE IT ATTACHES TO.** `enters_tapped_unconditionally` tested `"unless" in text` over the whole oracle text, so **Archway Commons** — *"This land enters tapped. When this land enters, sacrifice it unless you pay {1}"* — read as an untapped five-colour source, and `mana-fit` offered it as a tempo-free fix. The "unless" belongs to the sacrifice; paying {1} stops it dying and does nothing about the tapping. **Eleven lands in the corpus share the wording** (Rupture Spire, Transguild Promenade, Gateway Plaza, Command Bridge, Public Thoroughfare and the five Karoo bounce-lands); the sweep over all 1,266 lands flags exactly those eleven and drops none.

  **The obvious fix is worse, and the sweep is the only thing that says so.** Scoping the test to the SENTENCE containing "enters tapped" flags **all ten shocklands** plus Multiversal Passage and The Black Gate, because the shockland idiom spans two sentences — *"As this land enters, you may pay 2 life. If you don't, it enters tapped."* That is the exact overstatement the function exists to prevent. Strip `sacrifice … unless …` and keep the whole-text test for everything else.

- **THE GOLDFISH CANNOT RANK TWO LANDS THAT MAKE THE SAME COLOURS, and it never could.** It plays the **first land in hand** and credits its colours the **same turn**: LANDS have no tapped state, and there is no choice of which land to play. (Creatures gained one on 2026-09-26 — they tap to attack and untap on your turn — which changed what an extra combat phase is worth and left the land question exactly where it was.) A twelve-land `candidates` sweep against ur-dragon returned exactly **two distinct readings** — 45.304 for every unrestricted land and 44.027 for every restricted one — with always-tapped **Grand Coliseum tying never-tapped Forbidden Orchard**. That is the byte-identical tell this repo already uses to catch a flag nothing acts on, and here it caught a whole missing dimension. Named in `MODEL_ASSUMPTIONS` rather than fixed quietly: modelling it would slow every deck's early turns and restate every published figure on the fleet. **`mana-analysis` and `mana-fit` are deterministic for exactly this reason and are the whole of the evidence for a land swap.**

## The proposal — a decided branch, 2026-08-28

- **A BRANCH HAD TWO OBSERVABLE STATES AND NEEDED SIX.** The directory exists, or `merged` is present — and `delete` was the only code in the repo that read `merged`. So a branch the pilot had **accepted** was byte-identical to a half-finished experiment nobody had looked at, and `deck-info` said the same sentence about both: *"branch `eminence-v3` needs 12 card(s) sourced."* The gap had been named twice in the source and never given a home — `commit`: *"a commit says 'this is the deck I am committed to running'; a merge says 'this is the deck'. The gap between them is CARDBOARD"*; and `deck_info._next`, directly above the line it printed: *"a branch that is fully sourced is a decision waiting to be taken; one that is not is a shopping list."*

  `propose` records the decision. **`branch_state` derives all six states and stores none**, which is the rule `validate_pending` earned — *"closure is DERIVED, never declared … a hand-set flag is precisely how HISTORY.md became append-only and append-forgotten."* The consequence is the point: a proposal **un-blocks itself**. Drop a card into `data/collection/` and the blocker shrinks with nobody touching the branch.

- **`base_version` had been written since branches shipped and no code had ever compared it to anything.** `new()` records what the branch was cut from; nothing read it back. That comparison is `PROPOSED · OUTRUN` — another branch merged first, so the version this proposal claims is taken — and it is the merge-conflict analogue the stage needed. A field written for a check nobody wrote is indistinguishable from a field nobody needs, right up until the check exists.

- **`--proxy` was a per-invocation flag and was never persisted**, so `list`, `deck-info` and the web roster all showed the non-proxy verdict *however the pilot had decided*, and `merge` asked again at the last gate. A proposal records the **cards**, not a boolean: `source(proxy=…)` now takes either `True` (all of them) or an iterable of names. A decision about specific cardboard is recorded as specific cardboard.

- **`procurement` is a NOTE and nothing else** — no date parsed, no state derived, and the blocker does not move. A real `ordered` state was considered and refused: ownership means a box (`collection.py:20-34`), and a fifth sourcing state would make `owned_names()` return cards that are not in the house. It exists so the pilot does not buy the same six cards twice.

- **ONE PREDICATE WHERE FOUR MODULES HAD GROWN THEIR OWN.** `common.UNPLAYABLE_STATUSES`, `deck_info.STATE_RETIRED`, `net_change.FREE_TO_RAID` and `deck_branch._deck_holders` — which carried each holder's `status` and did nothing with it — all answered "is this deck in a pile". None disagreed yet, and it was **already costing something**: `deck-branch merge` refused Ur-Dragon on twelve cards, of which Faeburrow Elder (`sisay`, retired) and three more in `hapatra` (broken down) sit in decks that do not physically exist. The pilot was being told to unsleeve a deck already in a pile. `common.deck_is_apart` decides now and `source()` derives `free`/`apart` per row; everything downstream reads the row. Ur-Dragon's blocker went **12 → 6**, and the six are exactly the cards nobody owns.

  **`superseded` is deliberately not in the set**, which is why this reuses `UNPLAYABLE_STATUSES` rather than naming its own: a superseded list can still be sleeved and played, so its cards are still spoken for. "Cannot be played" and "its cards are free" are the same question asked twice.

- **`source()["cards"][].where` was a STRING for a box row and a LIST OF DICTS for an elsewhere row**, so all five consumers had to know which state they were in before they could read the field. One shape now, every row, carrying `kind`.

- **THE MERGE POST-AMBLE HAD NEVER RUN ONCE.** It called `deck_status.report(slug)` — a function that does not exist; the module defines `status`, `fleet`, `_fleet_main` and `main` — inside a bare `except`, so `out["stale"]` was unconditionally `[]` and the *"written against the previous list"* warning has never printed. **Two more bugs sat behind it**: rows key their name as `stage`, not `key`, and `status()` returns a list rather than `{"stages": [...]}`. Three wrong assumptions stacked, and none of them could surface, because **no branch in this repo has ever been merged** and a bare `except` around a chain cannot fail loudly enough to be noticed. The `except` stays — a missing corpus genuinely makes some gates unrunnable and a merge must not die for it — but it is narrow enough now that a typo cannot hide in it, and `test_deck_status_exposes_what_merge_reads_from_it` asserts the interface rather than re-deriving it.

- **`branch.json` was the last tracked pilot artifact with no gate**, and it holds the objective a branch is graded against. `validate-branch` closes it, and folds in the `MEMBERSHIP_AXES` ban that had been living as a loose test. It also gates `objective_history` — a key nothing writes or reads, hand-added when Ur-Dragon's objective moved off an authored axis. An ungated key that only ever arrives by hand is the shape a typo lives in forever.

## The simulation could not see the deck, 2026-08-29

One day's work on edgar-vampires, and every finding here was paid for by
measuring rather than reading. The order matters: each fix exposed the next,
and the last one invalidated the conclusions the first five had produced.

- **THE COMMANDER'S EMINENCE MINTED NOTHING, AND IT WAS THE WHOLE AXIS.**
  `command_zone_reduction` reads a commander for COST REDUCTION — The Ur-Dragon
  discounts Dragons — and there was no channel for the other kind. Edgar
  Markov's eminence says *"whenever you cast another Vampire spell, if Edgar is
  in the command zone or on the battlefield, create a 1/1 black Vampire creature
  token"*: live from turn one, unremovable, and the deck's entire token engine.
  Unmodelled, every one of those tokens was absent — the bodies, the
  arrival-damage payoffs they fire, the arrival draw they fire, and the fuel the
  sacrifice model eats. **`mean_bodies_by_turn` at ten was understated by 50%
  (9.3 against 14.0).** `deck-audit`'s engine brief described it in prose the
  whole time. 92 corpus cards carry the shape; exactly one is a commander on
  this bench, so the fix moves one deck.

- **THE TOKEN DOUBLERS DOUBLED ONLY TREASURES.** `treasure_doubler` shipped with
  the Treasure model and its own comment calls the shape *"Procession-style xN"* —
  and it was never applied to creature tokens. Anointed Procession, Elspeth
  Storm Slayer and Mondrak all sat in this 99 doubling nothing that fights. The
  deck's engine brief reads *"eminence mints a free body every time you cast a
  Vampire and the doublers turn one mint into four"*, and **neither half of that
  sentence was in the simulation.** Nine corpus doublers; the six TRIPLERS are
  left alone rather than read as x2, and two conditional matches are excluded —
  Kaya, Geist Hunter doubles only *"until end of turn"* off a -2, and Hosting
  Season is gated on a calendar date.

- **NOTHING DIED, SO A FIFTH OF THE DECK SCORED ZERO.** No blockers, no removal,
  no sacrifice outlets — so every death-triggered card was priced at exactly
  nothing. On this list that is **20 of 99 cards**: Blood Artist, Zulaport
  Cutthroat, Cruel Celebrant, Bastion of Remembrance, Viscera Seer, Ashnod's
  Altar, Phyrexian Tower, Woe Strider, Skullclamp. `model_sacrifice` converts
  creature TOKENS to a free outlet after combat has swung. **The policy is a
  choice and is stated rather than tuned**: a real pilot sacrifices in response
  to a wipe or for lethal and this model has neither, so keeping every token and
  sacrificing every token BRACKET the truth — the run without the flag is the
  floor, the run with it the ceiling.

- **`(?!lands?\b)` BLOCKED THE WORD, NOT THE FIVE TYPE NAMES.** The ETB trigger
  matched *"Whenever a Mountain you control enters"*, so a landfall payoff named
  by basic type read as a payoff for CREATURES entering. Fourteen corpus cards,
  two of them actively scoring: Dread Presence billed 2 damage per creature
  arrival off a Swamp trigger, and **Koth, Fire of Resistance — a PLANESWALKER —
  billed 4** off an emblem's Mountain trigger. Both surfaced in a candidate
  search for the very channel they were corrupting. The sweep only narrows: 438
  matches to 424, nothing newly matched.

- **AN ARRIVAL-DRAW ENGINE SAW ITS OWN ARRIVAL**, and the code comment claimed
  the opposite of what the code did. Welcoming Vampire is a 2/3 that draws
  *"whenever one or more OTHER creatures you control with power 2 or less
  enter"* — its own power is 2, so registering the engine before its own entry
  passed its own gate. Worth 22% of the turn-eight figure. And the same channel
  **silently required `model_combat`**: every call to the one door onto the
  battlefield sat inside `if model_combat:`, so a deck opting into `model_draw`
  alone lost three quarters of its arrival draws — 1.264 extra cards by turn ten
  against 0.323 — and reported the smaller number without a word.

- **AND THEN THE TABLE OVERRULED ALL OF IT.** With eminence, the doublers, the
  sacrifice engine, the death triggers and four draw channels all modelled, the
  goldfish preferred a go-wide refactor on damage, kill rate and card advantage.
  Forge, 400 games per arm against the pilot's own pod at one seed, said the
  refactor won **31/400 against the champion's 50/400** — a difference of
  -0.0475 with a Newcombe interval of [-0.0899, -0.0056] that **excludes zero**.
  The mechanism is one number: **combat damage dealt to players fell from 29.07
  to 18.20**. The goldfish has NO BLOCKERS, so it rewards a wide board of 1/1s
  that a real table simply blocks, and the refactor had cut every lord and
  anthem — the cards that make a small body connect.
  **That single missing assumption outweighed every gap closed above.** A
  go-wide or token strategy must be judged in Forge from the start; the goldfish
  is a resource model and its verdict on board QUALITY is not evidence.

- **A 100-GAME FORGE RUN IS NOT A RESULT.** The champion read 18/100 at a
  hundred games and **50/400 — 12.5% — at four hundred**, so the first estimate
  of the refactor's cost (-0.103) was more than double the properly powered one
  (-0.0475). `damage_on_wipe` told the same story: 0.0 over 9 wipes became 0.32
  over 37. MDE against an 0.18 baseline is 42 points at 20 games per arm, 17.5
  at 100, and 8.5 at 400. The deck's one historical experiment ran at 20 per arm
  and returned an interval spanning zero, which is the design and not the deck.

## The Forge AI will not press a sacrifice button, 2026-08-30

Measured across one 400-game pod run on `edgar-vampires@bloodline`, with a
cast-count control so a zero cannot be mistaken for a card that never arrived:

| card | cast | activated | per cast |
|---|---|---|---|
| Ashnod's Altar | 46 | **0** | 0.00 |
| Indulgent Aristocrat | 76 | 5 | 0.07 |
| Immersturm Predator | 72 | 5 | 0.07 |
| Yahenni, Undying Partisan | 65 | 4 | 0.06 |
| **Skullclamp** | 130 | **162** | **1.25** |

**Forge casts Ashnod's Altar forty-six times and activates it zero.** Every
sacrifice outlet sits between 0.00 and 0.07 activations per casting, free ones
included — this is not about paying mana. `Skullclamp` at 1.25 is the control
that makes it a finding rather than a parsing fault: the AI uses EQUIPMENT
happily and simply will not press a sacrifice button.

**THE CONSEQUENCE IS THAT EVERY FORGE RESULT ON A SACRIFICE DECK IS A FLOOR,
AND NOT A SMALL ONE.** `bloodline` was built on Viscera Seer, Ashnod's Altar,
Altar of Dementia, Woe Strider and Skullclamp; its measured 0.142 win rate came
almost entirely from COMBAT, because the engine the branch exists for barely
ran. A human pilot cracks Viscera Seer every turn. The simulation never does.

It also re-explains two earlier results that were read as deck failures:

- **`Culling the Weak` appeared 0 times in 208 games** (bloodline-v4). Read at
  the time as "a ritual whose additional cost the AI will not pay". The wider
  measurement says it is the same defect: a sacrifice, in any position, is a
  button this AI does not press.
- **The sacrifice model added to `pilot/goldfish.py` the same week** brackets
  the truth from the other side — it converts tokens at a free outlet after
  combat, which is closer to correct play than Forge manages.

**WHAT TO DO ABOUT IT, in order of cost:**

1. A card whose value is behind an ACTIVATED ability is under-measured by Forge.
   Check `activated <card>` against `cast <card>` before reading any result that
   rests on one. A trigger is safer than an activation for the same effect —
   `Lightless Evangel` grows on any sacrifice with no button, where
   `Bloodflow Connoisseur` needs the AI to choose.
2. ~~`simulate --profile` already exposes Forge's AI personalities (Default,
   Cautious, Reckless, Experimental) and every run in this repo has used
   Default for every seat. Whether another profile activates more is UNTESTED
   and is the cheapest experiment available.~~ **SUPERSEDED — the experiment was
   run; see AMENDED below.** `forge.STANDARD_POD_PROFILE = "Experimental"` and
   `--vs-profile` defaults to it, so the pod's three seats are Experimental and
   ours is Default. Heliod's record carries
   `profiles: ['Default', 'Experimental', 'Experimental', 'Experimental']`.
3. Forge's AI profiles are property files in the engine install. Tuning them is
   possible and would need its own harness change plus a fresh baseline on
   every deck, because it changes what "the pod" means.

**AMENDED 2026-08-30, AND THE FIRST VERSION OF THIS RULE WAS WRONG.** Running
the same list, pod and seed at `--profile Experimental` for 100 games against
the 400-game Default baseline:

| card | Experimental /cast | Default /cast |
|---|---|---|
| Indulgent Aristocrat | **0.41** | 0.07 |
| Immersturm Predator | 0.22 | 0.07 |
| Yahenni, Undying Partisan | 0.18 | 0.06 |
| Skullclamp (equipment control) | 1.19 | 1.25 |
| **Ashnod's Altar** | **0.00** | **0.00** |

All three sacrifice CREATURES moved together, roughly three- to six-fold, while
the equipment control stayed flat — a pattern that is hard to read as noise.
So the profile IS a lever and "the AI will not press a sacrifice button" is not
the rule. The rule is:

> **THE AI WILL NOT SACRIFICE FOR A BENEFIT ITS EVALUATOR CANNOT PRICE.**

`Indulgent Aristocrat` puts +1/+1 COUNTERS on the board — visible and scoreable,
and it moves freely once the AI is willing. `Ashnod's Altar` produces COLOURLESS
MANA with nothing queued to spend it on and is 0 for 59 castings across BOTH
profiles; `Viscera Seer` offers SCRY and was cast 0 times in 500 games across
both. Cost is not the discriminator — the Aristocrat costs {2} and the Altar is
free.

FOR DECKBUILDING: when a list will be judged in Forge, prefer an outlet whose
payoff is a visible BOARD CHANGE over one that makes mana or filters cards, and
prefer a TRIGGER to an ACTIVATION for the same effect.

AND THE OUTCOME DID NOT FOLLOW THE MECHANISM. Win rate 0.142 -> 0.130 and total
damage 38.37 -> 39.78, both inside noise at 100 games. The sacrifice engine
firing six times as often did not produce a better result, which is worth
holding onto before treating activation rate as a proxy for deck quality.

**THE STANDARD POD IS NOW `Experimental`, CHANGED 2026-08-30 ON A MEASUREMENT.**
Neither AI change moves our own result: me-Experimental reads -0.0125 and
pod-Experimental +0.0275 against the 400-game baseline, both intervals spanning
zero. So today's branch rankings are ROBUST TO AI COMPETENCE, which is the thing
worth knowing. But the pod SHARES move a lot:

| seat | Default pod | Experimental pod |
|---|---|---|
| vito | 0.422 | **0.330** |
| baylen-tokens | 0.130 | **0.190** |

The Default AI misplays a token deck badly — more decisions per turn, more it
fumbles. So the table was never uniformly weak, it was weak UNEVENLY, and that
quietly flattered whichever seat Default happened to pilot competently. The tag
rule is deliberately unchanged, so a record made before this date keeps its
untagged id and still means the Default pod: **every Forge record predating
2026-08-30 was measured against the old table and is not directly comparable to
one made after it.**

**AND A NAMING COLLISION, INTRODUCED AND CAUGHT THE SAME DAY.** The first cut
tagged these `-ai<P>` and `-vsai<P>` — and `-aiExperimental` is a SUBSTRING of
`-vsaiExperimental`, so a glob for one matched both. The first comparison written
against those ids read the same directory twice and reported two configurations
as byte-identical, which is exactly how it was caught: identical numbers are not
a coincidence. Now `-me<P>` and `-pod<P>`, neither a substring of the other, with
a test asserting they cannot collide in either direction. This was a SECOND
silent-overwrite defect created while fixing the first one.

**AND THE POD IS PART OF THE INSTRUMENT.** giada-angels, vito and baylen-tokens
are three EDHREC average decks played by the same Default AI. A result is
relative to that table and to that AI's competence, not to a metagame.

## A fetchland's colours are a property of the DECK, not of the card

**2026-08-30.** `manabase.land_colors(card)` credits a land for the basic types in
its type line and for the coloured symbols in an `add` clause. A fetchland has
neither. Wooded Foothills reads:

    {T}, Pay 1 life, Sacrifice this land: Search your library for a Mountain or
    Forest card, put it onto the battlefield, then shuffle.

No type, no `add`. It counted as **zero sources** — as did all sixteen true
fetches in the corpus — and `goldfish.py` built every land's `colors` from the
same call, so four fetches modelled as four lands that never produce anything.
`model_colors` defaults to **True**, so this was on for every deck in the fleet.

### The champion under the overrides: the hints did nothing for it, and two harness bugs came back (2026-09-29)

The copy-burst-v1 comparison had been cross-harness since the overrides shipped: the branch
had run under them, the champion never had. So the champion ran under the same eleven —
same pod, seed 909090, 100 games, `--jobs 4`, Default on our seat — and for the first time
both arms sat in one harness bucket.

| | plain | overridden |
|---|---|---|
| champion (v1.0.x) | 7/94 = 0.074 | **6/83 = 0.072** |
| copy-burst-v1 | 1/73 = 0.014 | 9/82 = 0.110 |

**THE OVERRIDE DID NOTHING FOR THE CHAMPION.** -0.002, ci95 [-0.083, +0.083]. The same hints
that lifted the branch from 6% of par to 47% left the champion exactly where it was — and the
lever is weaker there by construction: the champion carries **6** of the eleven hinted spells,
the branch all eleven, because the branch is the list the override was built around. Zada still
fired more (0.52/game against 0.35–0.39 pre-override), so the hints were active; they just did
not convert. That is a property of the lists, not a confound, and it is what copy-burst-v1 IS.

**THE WITHIN-HARNESS A/B, finally:** champion 6/83 against branch 9/82, **+0.037, ci95
[-0.054, +0.132], spans zero**, MDE 0.155. Whether copy-burst-v1 is the better list is an open
question at this sample size — the honest state, and the one the plan's own arithmetic
predicted: a +0.05 gain is unreachable at 1,000 games per arm.

**AND TWO HARNESS BUGS CAME BACK, both mine, both within twelve hours of the fix they undid.**

`net_change.forge` printed `branch 12/150`. The bucket key was `(pod, override_sha)`, and the
branch's Default-override run (9/82) and Experimental-override run (3/68) share the sha. The
profile was not in the key — one day after measuring that Default -> Experimental doubles
clock-outs on the same list. The same defect as the 10/155 it had been written to fix, one
axis over. The key is `(pod, overrides, profile)` now, the held-out row is labelled `ours
Experimental`, and the remedy note names the axis that actually differs instead of telling a
reader to install overrides the champion already had.

`pods.calibration` pooled the overridden champion into the subject null and moved it
**0.233 -> 0.206** — a 12% shift in the yardstick every MDE is scaled against, from one run
that changed what the AI may target. `limits` had said "nothing excludes one"; it was true and
it was a warning I wrote and did not act on. The null is the PLAIN harness now, overridden
runs are counted and printed beside it, and the figure is back at 0.233.

The record carries one never-cast card not seen before: **Witch's Mark**, "you may discard a
card, if you do draw two" — the same loot shape as Faithless Looting, and the AI will not
discard its own hand. No `AI:RemoveDeck` flag on it, so that entry rests on the documented
class alone.


### Our seat on the pod's profile: the land ratio did not move, the clock-outs doubled, and the early slice lied twice (2026-09-28)

`pilot_quality`'s argument for using the pod as a control is that the other seats are "the
same AI, in the same games, under the same engine". They are not the same AI: our seat runs
`Default` and every opponent runs `Experimental` (`STANDARD_POD_PROFILE`), which differ in 33
properties including the sacrifice master toggle. So the cheapest experiment nobody had run
was to put our seat on the pod's profile and see whether the 0.845 land ratio — the number
that makes every goblin-storm run read NOT COMPARABLE — was a profile artifact.

Same deck (copy-burst-v1), same pod, same seed (909090), 100 games, `--jobs 4`, same eleven
overrides installed, same 600s clock. Only our seat's profile changed. The prediction was
written down before the run: a profile artifact moves the ratio toward 1.0; genuine
mispiloting leaves it near 0.845.

| | Default | Experimental |
|---|---|---|
| land ratio | 0.845 | **0.852** |
| cast ratio | 0.863 | 0.902 |
| win rate | 9/82 = 0.110 | 3/68 = 0.044 |
| clock-outs of 100 | 17 | **32** |

**The land ratio is not a profile artifact.** +0.007 is inside the fleet's within-deck noise
(median spread 0.037). The seat drops fewer lands than the pod on either profile, the WITHHELD
band stands, and the 0.85 line is about the deck.

**Matching the pod's profile is not a free fix.** Clock-outs nearly doubled and the
difference EXCLUDES ZERO (+0.150, ci95 [+0.031, +0.264]). Win rate moved the wrong way and
spans zero (-0.066, [-0.156, +0.027]) — but it is a comparison across two censoring regimes,
82 decided against 68, so even that null is softer than it looks. A profile that changes how
often games reach the clock changes the denominator of every rate it is compared on. The
Default/Experimental asymmetry is a real flaw in the control claim and NOT what hurts land
drops; putting our seat on Experimental trades a cosmetic fix for a worse instrument.

**THE EARLY SLICE WAS WRONG TWICE, which answers "can we stop once we see signal".** At 11
games the land ratio read **0.805** and the cast ratio **1.009** — and I reported the second
as "a real profile effect, casts brought to parity". The full run says 0.852 and 0.902. Both
moved by more than the effect being looked for. `sim-progress`'s footer says why: Forge gives
the first turn to the previous game's loser, so a job's early games differ systematically from
its late ones and a partial slice is biased. "Large effect absent" is the one early reading
that survives — the 0.805 was already not heading to 1.0 — but a positive early reading on a
rate is not signal, it is the bias. Stopping there would have produced two wrong conclusions
from one run.

Also: `forge.run` writes the record at the END. Killing a run leaves logs and no record —
nothing `net-change` can bucket, nothing `validate-sim` can check. That is a second, purely
practical reason a run's N is fixed at launch.


### The warm worker serves the code it started with (2026-09-28)

`manamap serve` is also a warm worker: every read-only `manamap pilot <cmd>` routes through
`/api/cli` and skips the cold start, and CLAUDE.md says to restart it after a code change
because it holds the old modules until you do. That sentence was true and I read past it.

The daemon had been up since early in a session that changed `sim/progress.py`, and
`sim-progress` kept printing **"held and never cast: Smoldering Crater, Castle Embereth,
Forgotten Cave"** — lands — after the fix that filters lands was on disk and the function
was verified correct by direct import. The command was being served the pre-fix module.
`MANAMAP_NO_DAEMON=1` on the same command showed the fix immediately.

The exposure is wider than one command. **A NEW subcommand is safe** — the daemon's old
registry does not know it, the route fails open, and it runs locally with current code.
**An EXISTING subcommand is not**: `validate-sim`, `net-change`, `pods`, `validate-stack`
all predate the session and every CLI verification of them could have been stale. Re-run
with the bypass flag after the fact, all five were fine — but that was luck about which
modules the worker happened to reload, not a property of the design.

So: after a code change, either restart the worker or verify with `MANAMAP_NO_DAEMON=1`.
A direct `python -c` import bypasses it and is the honest check; a `manamap pilot` call is
only a check of the code once you know which code is answering.


### Per-deck AI profiles: Forge's own lever, and the knob that was never switched on (2026-09-28)

`res/ai/*.ai` are plain KEY=VALUE files of 121 `AiProps` knobs.
`AiProfileUtil.getAvailableProfiles()` is `new File(res/ai).list()`, so ANY file dropped
there becomes a name `-a` accepts, and `loadProfile()` falls back to `AiProps.getDefault()`
per key — so a PARTIAL file holding only the keys you change is legal. `forge.command()`
already passed `-a` per-seat in `-d` order. **Forge's own supported lever for piloting, no
engine patching, and the only thing refusing it was `argparse`**: four `choices=[...]` lists
in `registry.py` naming the four shipped profiles, duplicating a check Forge already does
better (it validates the name and prints what it has).

**THE KNOB THAT WAS NEVER SWITCHED ON.** `SACRIFICE_DEFAULT_PREF_ENABLE` is **false** in
`Default.ai`. A policy declaring `SACRIFICE_DEFAULT_PREF_MAX_CMC: 3` therefore compiles into
a legal profile, runs 400 games, returns no difference, and gets recorded as UNPROVEN —
"the sacrifice knob does not help", about a knob that was never on. That is
CLAUDE.md's *A CARD THE MODEL CANNOT READ LOOKS EXACTLY LIKE A CARD THAT DOES NOT HELP*, one
layer up, and it would have been the first thing this work measured. Forge documents FIVE
master toggles in its own comments ("Master toggle for the following options", "If disabled,
the following three options do nothing"); four are encoded with their families and a gated
key now requires its master.

`PLAY_AGGRO` is deliberately NOT one of them: `CHANCE_TO_ATTACK_INTO_TRADE` "works even if
not playing all-out aggro, e.g. PLAY_AGGRO disabled", while
`ATTACK_INTO_TRADE_WHEN_TAPPED_OUT` is "ignored if PLAY_AGGRO is globally **enabled**" — an
INVERSE relation. Encoding it as a master would refuse a correct policy.

**THE RUN ID NEEDED THE PROFILE'S CONTENT, NOT ITS NAME.** `profile_tag` carries the name,
which is enough for four fixed profiles and useless for one `mm-<slug>` whose content changes
every experiment: iterating knob VALUES would write one id and the second run would be
refused as an existing measurement. That is the fourth instance in this project of a
configuration axis missing from the id — `profile_tag`, `clock_tag`, `overrides_tag`, and now
`ai_profile_tag`. Forge's own four are deliberately NOT content-hashed: they are named
already and hashing them would rename every historical record.

**A MODEL STAMP IS A SHA OVER A WHOLE FILE, SO WHAT LIVES IN THAT FILE MATTERS.** The 121
constants went into `pilot_policy.py`, which is in `goldfish._MODEL_FILES` — and every knob
edit then staled all 37 goldfish artifacts, for constants the goldfish does not read and
cannot read. It took one commit: the stamp moved `c42ed8ea2ec3` -> `e41cebc10cf0` and ~25
fleet tests went red. Split into `forge_ai.py`, with a test asserting `pilot_policy.py` IS in
the stamp, `forge_ai.py` is NOT, and neither `goldfish` nor `goldfish_turn` imports the Forge
half — because the stamp only tells the truth while the dependency really is one-way.

**A REASON AND A DIRECTION CANNOT BE CHECKED AGAINST EACH OTHER, so print both.** The first
trial policy set `MIN_COUNT_FOR_STORM_SPELLS: 2` under a `why` that said "lowering it so the
AI casts sooner". The default is 1, so the value RAISED the threshold and would have made the
AI hold storm spells longer — the opposite of its stated reason, in a file whose whole purpose
is to make a piloting claim arguable. Nothing mechanical catches that. The compiled profile
now prints `Default.ai has 1; this sets 2` beside the knob, which makes it visible to a human
reading the file the engine loaded.

Also refused, each because it produces an unmeasurable experiment: a key Forge does not read,
a wrong type, and **restating a default** — a policy line that changes nothing gives the A/B
two identical arms and reports noise as a finding.

NO DECK HAS A POLICY. The mechanism exists and nothing uses it, which is the honest place to
stop: the one rule ever written was CR-proven correct and measured monotonically harmful.


### The three omissions a run id had, and the power the screen never had (2026-09-28)

**A RUN ID MUST CARRY THE HARNESS, AND THIS WAS THE THIRD TIME.** `profile_tag` and
`clock_tag` each exist because a configuration axis was missing from the id and a run
silently overwrote a different measurement at the same name. Card-script overrides were the
third, and the worst, because they change how the AI PLAYS: goblin-storm/copy-burst-v1 reads
**1/73 with Forge's own scripts and 9/82 with eleven `AITgts$` hints installed** — same
deck, same pod, same clock, same profile, so at the same seed the two would have written
THE SAME PATH. The second would have been refused as an existing measurement, or replaced
the first with `--force`. `overrides_tag` closes it.

The two records that produced that +0.096 escaped the collision **by accident** — they were
made at different seeds, which is itself a confound on the headline. Their decided counts
also differ, 82 against 73, so the denominators are not the same population either. The
interval is still reported, because the piloting handicap is common to both arms (lands
0.316 vs 0.333), but "same seed, same everything" was never true of it.

`overrides_tag` TAKES the fingerprint rather than reading the engine. The first cut called
`forge_pilot.installed()` inside it, so the id depended on what happened to be installed on
the machine asking — which broke the two things an id is for: the pod test could no longer
find a tracked record by recomputing its name, and the same lookup resolved to different
paths on two machines. **A run id is a function of a configuration, never of an
environment.** Six tests caught it.

**A POLICY CHANGE STALED NOTHING.** `pilot_policy.py` shipped the same day and was not added
to `goldfish._MODEL_FILES`, which is verbatim the failure that constant's own docstring
exists to prevent — "every derived artifact would have read as current while the model
underneath it changed". The policy layer decides whether a card is cast at all. Added;
`model_version` moved `7d7fbdfb76cf` -> `c42ed8ea2ec3` and the fleet regenerated.

Separately, `meta` gained `pilot_policy` — the per-deck policy DATA, which is a different
question from the policy ENGINE, the same split as `card_overrides` (the harness) against
`decklist_sha256` (the list). Its fingerprint is over the RULES and not the file bytes, the
opposite call from `model_version`: there a comment sits beside code that runs, so coarse is
right; here the `why` genuinely is prose and the thresholds genuinely are the model, so
editing a reason must not stale a figure it cannot have changed.

**AND THE 100-GAME SCREEN CANNOT SEE ANYTHING WORTH SEEING.** An MDE is meaningless without
the baseline it is computed against, and the plan for the piloting work quoted **0.115 at
100 games** — computed against `p_a = 0.014`, the BROKEN arm, a week after the null-scaling
lesson was learned the hard way one layer up. Against the baseline that now applies,
`p_a = 0.110`, through this repo's own `sim/stats.py`:

| n/arm | detectable win-rate gain | needed rate | % of the 0.233 null |
|---|---|---|---|
| 100 | +0.155 | 0.265 | **114%** |
| 200 | +0.105 | 0.215 | 92% |
| 400 | +0.070 | 0.180 | 77% |
| 1000 | +0.045 | 0.155 | 67% |

`games_for_difference(0.110, 0.05)` returns **None at max_n=1000**: a plausible +0.05 policy
gain is not reachable. +0.08 needs 308 games/arm, +0.12 needs 152. So at 100 games the only
detectable improvement is one that lifts the deck ABOVE par, and a "screen" there can
produce false positives and nothing else.

A COUNT METRIC IS BETTER AND STILL NOT ENOUGH, which is worth knowing before anyone reaches
for it. Per-game Zada triggers over the 100-game run: mean 0.440, sd 0.795, and **71 of 100
games have none at all**. That zero-inflation is what costs the power — the detectable
difference in means is +0.315 at n=100, 72% of the current mean, against the win rate's
141%. Twice as sensitive, in the same unusable range.

The conclusion the plan has to absorb: **100 games is a smoke test, never a verdict.** It
answers "did the mechanism fire at all", which is a real question with a visible answer —
`part-03.log:810` shows one Reckless Ransacking copied 29 times, +3/+2 each. It cannot
answer "is this better", and an adaptive ladder that treats it as a screen is spending
5-8 hours per candidate to learn nothing.


### What it cost

`ur-dragon/landbase-v1` swapped two basic-only fetches and four coloured lands
for four true fetches. `mana-fit` reported it as worse on **every colour**:

| | W | U | B | R | G |
|---|---|---|---|---|---|
| eminence-v3, fetch-blind | −5 | −4 | −12 | −8 | −1 |
| landbase-v1, fetch-blind | **−6** | **−7** | **−14** | **−10** | **−3** |
| landbase-v1, fetch-aware | −2 | −3 | −10 | −6 | **+1** |

The true delta is **W +1, U −1, B +2, R 0, G 0** — colour access flat — while the
recurring life tax **halved, 8 → 4 per tap-cycle**, and the last always-tapped
land left the deck. A tool that scores a strict improvement as a five-colour
regression will talk a pilot out of it, and there is no figure on the page that
looks wrong.

### The sweep — 54 lands matching `ROLE_LAND_PATTERNS["land:fetch"]`

The load-bearing split is **one word**:

| | | finds a dual? |
|---|---|---|
| **16** | TRUE FETCH — `a Mountain or Forest card` | **yes** |
| **20** | BASIC-of-TYPE — `a basic Forest, Plains, or Island card` | **no** |
| 15 | ANY BASIC — `a basic land card` | no |
| 1 | ANY LAND — Urza's Cave | yes |

The Panoramas and Landscapes read almost identically to a real fetch and cannot
touch a shockland. Resolving those two groups the same way is the error the
taxonomy exists to prevent, and it is the bug the test suite catches first.

Three more are excluded by the gate rather than the taxonomy, each for a
different reason, and each was read card by card:

- **`Ash Barrens`, `Thaumatic Compass`** — put the land into your **HAND**. The
  clause must reach `onto the battlefield`. (Ash Barrens is also parenthesised
  reminder text, so `_REMINDER_RE` already dropped it.)
- **`Flagstones of Trokair`** — searches on a **death trigger**, not at will. The
  `:` in the pattern is the gate.
- **`Demolition Field`** — its search sits in a later sentence, behind
  "destroy target nonbasic land an opponent controls". Excluded by the
  sentence boundary, and correctly: a conditional source is counted at its
  unconditional value, because understating is recoverable and overstating
  builds a deck that cannot cast its spells.

### The shape of the fix

`land_colors(card, pool=None)`. **Without `pool` it is byte-identical** — proven
over all 1266 corpus lands, zero divergences — which is what keeps a caller with
no deck to offer reproducible. Callers that HAVE a deck pass one: `mana_analysis`,
`mana_fit` (so a *candidate* fetch is scored on what this list could give it),
`deck_facts`, and `goldfish.build_library`.

Resolution is **one level deep**, and the reason is not colour. Union is
idempotent, so one hop and two hops agree on every real board; the failure mode
is no answer at all — two `a land card` fetches each find the other and recurse
until the stack ends. Targets are priced with a bare `land_colors`.

### Four of the first five tests were VACUOUS, and only re-introducing the bug said so

All five passed. Removing the guard each was written for changed nothing in four
of them:

- *a fetch cannot fetch itself* — going through `land_colors` prices the self-target
  with a bare `land_colors`, which for a fetch is the empty set. Union unchanged.
  It has to drive `fetch_targets` directly.
- *resolution is one level deep* — asserted on colours, which cannot show it.
  Rewritten as **termination** on two mutually-fetching lands.
- …and that one only closes the loop when **the card is in its own pool**, which is
  how every caller passes it. Leave it out and cave finds twin, twin self-excludes,
  done — no recursion, test green, guard untested.
- *every corpus land is unchanged without a pool* — asserted
  `land_colors(card) == land_colors(card, pool=None)`. Both arguments take the same
  branch. It holds however the fetch layer is wired. Rewritten to assert the property
  the branch protects: **a fetch handed no deck produces nothing.**

The rule this pays for again: a test that re-derives the rule is testing itself.
Drive the production function, then prove the test by putting the bug back.


## Forge's `-c` clock ends a game's ACCOUNTING, not its AI thread

**2026-08-31, and the constant has since moved.** `SIM_GAME_CLOCK_SECONDS` is passed
to Forge as `-c`; it was **300** when this was measured and is **600** now. `300`
survives as `SIM_CLOCK_ID_BASELINE`, frozen so the run ids written under the old clock
still resolve — every tracked record on disk now ends `-c600`. The mechanism below is
unchanged by the value. It
fires a `FutureTask` timeout, Forge writes `Game Result: Game 1 ended in 300122
ms` and reports **a winner** — and the AI thread carries on running. Nothing
bounded the subprocess, so a job could run until somebody noticed.

Measured across the eighteen tracked runs, wall time against the per-game `ms`
the run itself recorded:

| deck | n | wall | accounted by its own games |
|---|---|---|---|
| edgar-vampires | 20 | 509s | 95% |
| gishath | 20 | 564s | 94% |
| edgar-vampires | 400 | 11243s | 90% |
| ur-dragon | 100 | 3720s | 61% |
| **yawgmoth-swarm** | **20** | **13372s** | **3%** |
| **zur-enchantress** | **20** | **15220s** | **5%** |

Eleven of thirteen are healthy. **Two burned nearly four hours each on twenty
games, and 95% of that time is claimed by no game at all.** They were not run
concurrently, so it is not heap contention between runs.

A job is now capped at `clock x games_in_job x 1.5 + 120s`. Checked against
every tracked run: the two pathological ones are killed, **all sixteen others
survive** — including the 400-game arm, which is the real control, since a cap
that also killed a legitimate long run would be useless. **7.1 hours** across
the tracked set. A killed job's finished games are kept and parsed normally;
`truncated_jobs` names the rest, and is an empty list rather than a missing key.

### Two things the record was saying that were not true

- **`ASSUMPTIONS` claimed "a game past the clock is recorded as a draw, not
  dropped."** It is not. `summary.draws` is **0** on every tracked run,
  *including* the two with three clock-hit games each, and `edgar-vampires`
  n=400 carries **75 clock-hit games — 19% of the run — all with winners**. A
  deck that trips the clock has its rate scored off truncated games with nothing
  marking them.
- **`SEEDED_NOTE` claims identical inputs reproduce the logs byte for byte.**
  Tested on three tracked runs with identical seat shas, seed base, Forge build
  and clock, differing only in `-n`: n20 and n100 agree on job 0 for all three
  shared games and diverge at game 3 on jobs 1 and 2; n400 diverges from game 1.
  The logs say why — 197 `TimeoutException` in one part file. **The AI is on a
  wall clock, and wall clocks are not reproducible.** Seeded replay holds for a
  game or two, which is exactly as far as the note's own evidence went.

### Also measured, not yet changed

**FIXED — this is what `forge.default_jobs()` exists for.** `jobs` used to default to
`os.cpu_count() - 1` = **7** on this machine, which has **4 performance cores** and 4
efficiency cores; a job that lands on an E-core sets the wall for everyone, and 5–18% of
games came back `truncated`. `default_jobs()` now returns `max(1, (os.cpu_count() or 2)
// 2)` = **4**, the performance-core count.

Still measured and still left alone: `split_games` is a static even split with no work
stealing, and the straggler tail was **+4061s / +6734s / +1304s** on the three biggest
runs — 12% to 50% of the wall spent with idle cores waiting on one job. Work stealing
would change the SAMPLE a given run id produces, because `run_id` does not encode how the
games were split.

---

## 2026-09-04/05 — A card read correctly and never played

The single most expensive class this bench has produced, found **five times in
one session** and only caught as a class on the fourth.

Every casting loop in `goldfish.simulate_once`'s main phase selects on a
CHANNEL: cards that draw, cards that ramp, cards that make Treasure, cards with
a body. **A card matching none of them sits in hand for ten turns while its
profile says exactly what it would have done.** The figure that results is not
low — it is a number about a different deck.

`goldfish.py` already carried a comment about patching this once, for damage
doublers: *"Gratuitous Violence and Dictate of the Twin Gods are enchantments,
read correctly and never cast."* It was patched case by case and nothing looked
for the next one.

| found | cards | what it cost |
|---|---|---|
| drain permanents | Sanctum of Stone Fangs, Northern Air Temple | two Shrines measured as EXACTLY nothing; casting them nearly doubled the deck's drain and moved kill-by-t8 +0.051 |
| lifelink Auras | Steel of the Godhead, Sheltered by Ghosts | the gain side of the drain engine |
| free sacrifice outlets | Ashnod's Altar, Altar of Dementia | a SLEEVED deck ran its engine on 2 of its 4 outlets |
| death engines | The Meathook Massacre | went straight back into the bucket the moment its death triggers started working, because its DRAIN profile is empty and that was all the predicate checked |
| attack enablers | Aqueous Form, Reconnaissance, Aether Tunnel, Spirit Mantle | **a DEADLOCK** — see below |

The fleet sweep that found the rest: **228 of 873 cards (26%) are never cast**,
heliod 48%. Most of that is CORRECT and the number alone means nothing — a
counterspell should never be cast in a goldfish. The defect is only ever a card
that is never cast **and feeds a channel the deck switched ON**. Four, on two
decks, and both classes were real.

`model_coverage.never_cast` and `silent_losses` put the predicate in production
and a fleet test asserts the list is EMPTY on every deck. It MIRRORS the casting
loops rather than replacing them, which is a real risk of drift — and the drift
happened within the hour, when `model_deaths` was taught to goldfish and not to
the mirror. The test caught it. That is what it is for.

### The deadlock

Four of zur-enchantress's six attack enablers — including BOTH one-mana ones —
matched no casting loop. The only route to an enabler was Zur's tutor, which
needs an attack, which needs an enabler. **The model had built a deck that could
not start its own engine, and then reported the result as a fact about the
deck.** `enabled_share` 0.132 -> 0.581 on the fix.

## A rate that nothing had ever asked about

`model_commander_attack_tutor` fired ONCE A TURN from the turn after the
commander landed, and reported **5.70 fires a game**. Forge — 60 games, piloting
confirmed COMPARABLE — resolved that search **73 times. 1.22 a game.**

    tutor fires   5.70 -> 0.877        board power t10   27.30 -> 16.64
    kill by t8    0.501 -> 0.173       drain t10         17.47 ->  9.43
    kill by t10   0.924 -> 0.732

**More than half of one day's measured gains were that assumption.** Five
branches were merged against a tutor firing five times more often than evidence
describes. The branches were compared against EACH OTHER under the same wrong
assumption, so the directions mostly survive; the levels do not.

`fires_per_turn` and `source` are now REQUIRED, the same contract `model_deaths`
keeps. **A rate driving a figure must name where it was measured**, or it is the
deleted engine lift wearing a new hat. And where a rate is a CEILING — 1.0 when
an enabler is out, because once a turn is the most that can happen — the record
says so, because the `basis` string had described the old model as a ceiling
while the figure was being read as a forecast.

## Six lands counted as coloured sources that cannot pay a coloured cost

`land_colors` counts a land whose ONLY coloured mode is `{1}, {T}: Add one mana
of any color` at FULL VALUE, in every colour it makes. Karsten's targets are
about casting ON CURVE, and a land needing an extra generic cannot pay a
`{1}{B}{B}` cost on turn three the way a Swamp can.

    colour   sources   gated   real   target
    W             24       6     18       22
    U             24       6     18       22
    B             31       6     25       36

Every colour short; the headline read 24/24/31 and only black looked wrong.

**NOT DISCOUNTED — REPORTED.** Choosing a fraction to divide a gated land by
would be an authored number driving a headline. `sources.gated` names them and
`on_curve_probability.lands_only_ungated` is the FLOOR, so the figure travels as
a band.

Two things the sweep bought. The obvious fix is backwards: cutting those lands
for basics made every colour WORSE, because they really are counted for all
three and cutting the fixing loses more than it gains. And the free-mode escape
is load-bearing — City of Brass and Mana Confluence carry a cost clause AND a
free coloured tap. The first pattern used a lookbehind for `}` and failed on
`{1}, {T}`, where a SPACE sits between them, so the whole check returned zero.

`Fetid Heath` is a FILTER (`{W/B}, {T}`) and is a different class; the sweep
covered numeric costs only and the pattern was not widened past it.

## Two regexes in the parser, both wrong about a game's outcome

- **`^Game Outcome: .*draw` matched a DECKING LOSS.** Forge writes `has lost
  trying to draw cards from empty library`. Every one of the 14 lines matching
  the loose pattern across every log on disk was that line and ZERO were a
  genuine draw. Twelve games in one 60-game run were recorded as draws while
  still carrying a winner, `tally_wins` counted those winners anyway, and the
  run's accounting came to **72 of 60**.
- **A game nobody won was DROPPED.** `analyze` rebuilds its summary from games
  carrying a winner, a draw or a truncation, so a game with none of the three
  vanished — and a simultaneous loss (all four seats at 0 life on the same turn,
  a draw the rules recognise and Forge does not announce) is exactly that game.
  Derived from the `lost` map because there is no line to match. The flag then
  had to survive three hops — settle, `game_facts`, `compact` — and was dropped
  at two of them.

`test_every_tracked_run_accounts_for_all_its_games` found both. A balance test
over a tracked artifact is worth more than it looks.

## `OK` from a validator that checked nothing but the file's shape

`validate-goldfish-targets` caught a bare `Exception` around `load_deck_cards`
and returned an empty error list, which `main` printed as `OK … ◆`. Every
membership and win-line check was skipped in silence. Caught when `deck-branch
new` (which writes `decklist.txt` and no `cards.json`) produced a clean OK over
a declaration naming two cards the swap had just removed.

The headline word now carries how much was checked — `PARTIAL`, with the note
naming what did not run and the exact command that fixes it. **One existing test
had been asserting `OK` on a run that checked nothing, for months, which is how
the blind spot stayed invisible.**

## 2026-09-06 — The commander was doing nothing, twice over

zur-enchantress changed commander after Forge measured the old one's engine
firing **1.08–1.22 times a game** across three 60-game runs — and a direct test
showed that at that rate the deck performed IDENTICALLY to one with the trigger
removed entirely (kill-by-t8 0.135 against 0.145). Zur the Enchanter is a 1/4
who lands on own-turn 5, attacks on 6, and dies to the first sweeper in 27 of 52
games. Three versions all read below the table's null with the interval
excluding zero. **The payoffs were never the problem; the deck was converting a
resource it did not receive.**

The replacement, `Zur, Eternal Schemer`, has two abilities and the model read
NEITHER:

    Enchantment creatures you control have deathtouch, lifelink, and hexproof.
    {1}{W}: Target non-Aura enchantment you control becomes a creature … with
            base power and base toughness each equal to its mana value.

    kill by t8   0.153 (neither) -> 0.214 (grant) -> 0.327 (both)
    drain t10     8.01           -> 14.96        -> 17.62

**The grant is a whole-type effect and worth far more than any one lifelink
card.** Corpus sweep: 7 cards grant lifelink to a named type. The animation is
UNIQUE — one card in the corpus — so it is DECLARED per deck, the same contract
`model_commander_attack_tutor` keeps. A pattern fitted to a single card is not a
pattern.

And the synergy only works because "becomes a creature IN ADDITION TO ITS OTHER
TYPES" means an animated enchantment is an ENCHANTMENT CREATURE, so the grant
covers it. That required the type line to travel with the body.

### The zip that paired two different lists

The first cut of the grant read

    for tl_, (pw_, …) in zip(battlefield_types, battlefield)

`battlefield_types` holds every nonland permanent; `battlefield` holds only
creatures. The zip matched a creature's power against an unrelated permanent's
type line, and the grant figure was computed from garbage — it overstated drain
by 16%. Fixed by appending the type line at `creature_entered`, the one door
onto the battlefield, so the two cannot drift.

### Starfield of Nyx, and the sixth never-cast

`Starfield of Nyx` — "as long as you control five or more enchantments, each
other non-Aura enchantment you control is a creature … equal to its mana value"
— fed NO channel, so it was never cast and its static mass-animation did
nothing. Sixth instance of the never-cast class.

Its measured value once fixed was small, and the reason is worth recording: the
model lets the commander activate `{1}{W}` **unlimited times per turn**, mana
permitting, so it had already animated everything Starfield would have. That
probably overstates the commander and understates Starfield, whose real
advantage is being free and simultaneous. Treat that flat reading as suspect.

## The drain angle, measured against Forge rather than argued

The pilot asked whether the lifegain-drain plan was actually performing. It was
not, and the numbers are unambiguous:

    life gained per game, six runs:  MEDIAN 0.0 in four of them
    noncombat damage:                1.3 – 2.5 per game, 9–17% of our damage
    the goldfish claimed:            26.4 per game, 29% of damage

A tenfold overstatement, and the mechanism is that the payoffs had no fuel —
`Vito`, `Enduring Tenacity` and `Marauding Blight-Priest` all read "whenever you
gain life", and the deck gained none.

    seat              wins  combat  noncombat
    giada-angels        40   98.13       0.00
    zur-enchantress      5   14.38       2.22
    a four-player table needs 120 life removed

**Combat wins these games by a factor of four**, and the deck was dealing
one-seventh of what a kill requires. What changed the answer was not cutting the
drain but the new commander granting lifelink to seventeen bodies: cutting the
payoffs afterwards made the deck WORSE (kill-by-t8 0.400 -> 0.309), because the
fuel finally existed.

## The Shrine count was always a fiction

Six Shrines in a 99 are drawn **1.06 times by turn ten** — 30% of games see
zero. "For each Shrine you control" was multiplying by one. The whole cycle,
twelve Shrines, only reaches 1.9.

That explains every earlier measurement: eight Shrines was worse, four was
worse, six was "best" by a margin that was noise. It was not a curve with a
peak; the count did nothing and individual cards drove the results. **A flat
line was read as an optimum.**

Under the new commander they are worth running for a different reason entirely:
mana value is now POWER, so the MV4–5 Shrines that were unfetchable liabilities
are 4/4s and 5/5s with deathtouch, lifelink and hexproof.

## A ROOM'S `cmc` IS BOTH DOORS, AND THE MODEL CHARGES IT TO CAST THE FIRST ONE

Measured 2026-09-06 on `zur-enchantress/rooms-v1`, which read 0.3108 against the
champion's 0.3540 on kill-by-t8 and **is not evidence about Rooms at all**.

Scryfall reports a Room's `cmc` as the combined mana value of both halves, per
CR 202.3d. `goldfish.py` builds its casting cost straight from that field —
`"cmc": int(card.get("cmc") or 0)`, spent through `spend(card["cmc"], ...)` —
so the model charged:

| card | left door | model charged |
|---|---|---|
| Bottomless Pool // Locker Room | `{U}` = 1 | **6** |
| Underwater Tunnel // Slimy Aquarium | `{U}` = 1 | **5** |
| Grand Entryway // Elegant Rotunda | `{1}{W}` = 2 | **5** |
| Surgical Suite // Hospital Room | `{1}{W}` = 2 | **6** |
| Unholy Annex // Ritual Chamber | `{2}{B}` = 3 | **8** |

A one-mana enchantment was priced at six. This is the `never_cast` class again —
the branch measured five cards the model could rarely afford, and the reading
says nothing about the line.

**Three separate blind spots stack here, and only the first is a pricing error:**

1. **Cost.** You cast ONE half. `cmc` is both.
2. **The unlock trigger.** `goldfish.py` contains no concept of unlocking.
   `_ENCHANTMENT_ENTERS` reads the Eerie idiom as a plain enchantment-ETB, so
   the "and whenever you fully unlock a Room" half of four cards already in the
   deck — Balemurk Leech, Entity Tracker, Fear of Infinity, Gremlin Tamer —
   has never been modelled, and has never been able to fire in paper either
   because the deck owns no Rooms.
3. **Animation size.** CR 709.5 says a permanent does not have the mana cost of
   a locked half, so by CR 202.3d a FULLY UNLOCKED Room's mana value is both
   doors combined — and `model_commander_animate` sets power from mana value.
   `Unholy Annex // Ritual Chamber` is an 8/8 with deathtouch, lifelink and
   hexproof once both doors are open. The model sees none of that.

**FIXED 2026-09-06, all three together** — deliberately, because charging the
left door while still ignoring the unlock and the animation would have made
Rooms cheap AND empty, and the model would then have recommended them for the
wrong reason. `rooms-v1` moved **0.3108 -> 0.3842** on the strength of the
pricing alone, which is the size of the error.

The cheaper door is cast, NOT the front one, and the corpus sweep is what
settles it: the front door is cheaper-or-equal on 24 of the 30 Rooms, so a
front-door rule passes casual inspection — and is a fourfold error on `Defiled
Crypt // Cadaver Lab`, whose doors are `{3}{B}` and `{B}`. The pips come from the
same door; `front_field` cannot be reused because it always answers with the
left half, which is the wrong half exactly when the cost is.

**WHAT IS STILL WRONG, AND IT IS NO LONGER A CLEAN FLOOR.** Scryfall
concatenates both halves into `oracle_text`, so the model reads text off a
LOCKED door:

* **Over-credited.** `Unholy Annex // Ritual Chamber` is charged 3 for its
  `{2}{B}` front half and handed the 6/6 Demon printed on the `{3}{B}{B}` back
  half at the same moment. Two of five Rooms in `rooms-v1` gain a body this way.
* **Under-credited.** The "When you unlock this door, …" effect itself is
  modelled for none of the 26 Rooms that have one. `Unholy Annex` is wrong in
  BOTH directions on one card: its front half drains 2 from each opponent every
  end step *only if you control a Demon* — `drain_profile` flags the whole card
  `unmodelled` — while the Demon that would switch it on is credited free.

So a Room reading is now a mix of an over-credit and an under-credit, not a
bound in either direction. Do not describe it as a floor.

## THREE PARALLEL LISTS, TWO CASTING LOOPS, AND ONLY ONE OF THEM APPENDED

Found 2026-09-06 while wiring Rooms, and it is the larger of the two bugs.

`battlefield_pips`, `battlefield_types` and `battlefield_mv` are index-aligned by
convention and nothing enforced it. The `_engine_permanent` loop — which casts
drain payoffs, sac outlets, death engines and attack enablers — appended to the
first two and not the third. So `zip(battlefield_types, battlefield_mv)` in the
animate scan paired a type line with ANOTHER card's mana value, and then
truncated at the shorter list, hiding every later permanent from animation.

It bites exactly the deck that has both halves — `model_drain` AND
`model_commander_animate` — which is zur-enchantress and nothing else. The
champion had been understated all along:

    kill by T8   0.354 -> 0.381   (+0.027)
    kill by T10  0.864 -> 0.888
    no kill      0.137 -> 0.112

**IT INVALIDATED A PROPOSAL THAT WAS ALREADY MADE.** `bodies-v4` had been
proposed as v3.1.0 on the reading that it was FLAT (0.3549 against 0.3540) while
gaining board power. Corrected, the champion is 0.3810 and the branch is 0.3683
— not flat, 0.0127 BEHIND — and the proposal was withdrawn. Five card purchases
were riding on a number produced by a `zip` over misaligned lists.

This is the THIRD instance of the same class in this file. `zip(battlefield_types,
battlefield)` paired two different lists; `merge_deck_map` measured membership
rather than reading names; and now this. The lesson that keeps not sticking is
that **parallel lists maintained by more than one writer will drift**, and `zip`
hides it by truncating instead of raising. A fourth list was added here for
Rooms, which is a reason to be suspicious of the shape rather than a defence
of it.

## THE UNLOCK DOOR, MODELLED — AND IT MADE ROOMS WORSE

Completed 2026-09-06, and the direction is the finding.

`room_profile` now splits `oracle_text` on the same ` // ` separator as the cost,
in the same order, so each door is paid for and credited separately. `classify`
rebinds `card` and `text` to the ENTRY face before every text-derived profile
runs — bodies, draw, drain, combat and anything added later inherit the fix by
construction — and stores the other door's profiles as `room["on_unlock"]`,
applied in the unlock block and nowhere earlier.

    rooms-v1, locked door read free      0.3842
    rooms-v1, correctly attributed       0.3192
    champion                             0.3810

**THE OVER-CREDIT WAS LARGER THAN THE UNDER-CREDIT**, which is the opposite of
what the earlier note in this file predicted, and it is why that note said not to
describe a Room reading as a floor. `Unholy Annex // Ritual Chamber` was handing
the model a 6/6 flying Demon the instant its `{2}{B}` front half resolved. Making
that Demon cost the `{3}{B}{B}` it actually costs drops board power at T6 from
7.70 to 7.00, and the twenty-six "When you unlock this door" clauses the same
commit taught the model to read do not make it back.

**THE MECHANISM IS ARITHMETIC AND IT IS NOT ABOUT THE CARDS.**

    entry + unlock, the five Rooms in rooms-v1   30 mana
    mean available mana at turn 10                8.2

One door opens, maybe two. The 8/8 animated Room and the 6/6 Demon are real and
they are on the far side of a mana wall, in a deck whose objective is a kill by
turn eight. Rooms want a slower deck than this one.

**WHAT SURVIVES IS A ONE-ROOM ARGUMENT, NOT A PACKAGE.** rooms-v1 is still best
in the fleet on stall (0.026 against 0.043) and interaction affordable at T6
(0.627 against 0.577), because a cheap front door is a real enchantment that
fires four Eerie cards for one mana. That is a case for a single `{U}` Room, and
it is not the case that was tested.

TWO IMPLEMENTATION NOTES THAT COST SOMETHING TO GET RIGHT:

* `combat_profile(card)["power"]` is the CARD's power, which is 0 for a Room, so
  a hand-rolled body loop entered the 6/6 Demon as a 1/1. The model's own token
  convention is `token_bodies` and `token_power`, where the power is the TOTAL
  across bodies — (6, 1), never (6, 6).
* The draw key is `etb_draw`. An invented `on_etb` silently read nothing, which
  no test would have caught because absent draw looks exactly like no draw.

---

## The goldfish asks whether the mana was there; nothing asked whether the commander STICKS

**2026-09-07, heliod.** Every deck in the fleet had a goldfish figure for its
commander and every one of them answered the same question: *could this deck
afford it by turn N.* Heliod's read **80% by turn six**. Then the Forge logs:

```
  Heliod RESOLVED in        6/20 games   (30%)
  transformed to Eclipse    3/20 games   (15%)
  first cast, global turn   21 25 30 32 34 41      (median 31 — the fleet's latest)
```

The solitaire model has no counterspells, no removal, no commander tax and no
table that is doing something else. It is not wrong — it answers its own
question correctly — but nothing in the repo was asking the other one, and the
two had been read as the same number.

`commander_access` in `sim/parse.py` is the measurement. Across the whole fleet
on the runs already on disk:

| deck | n | casts/game | resolved | ci95 |
|---|---|---|---|---|
| hapatra | 20 | 1.70 | 1.000 | [0.84, 1.00] |
| yawgmoth-swarm | 20 | 1.20 | 0.950 | [0.76, 0.99] |
| goblin-storm | 100 | 1.55 | 0.910 | [0.84, 0.95] |
| radagast | 20 | 1.10 | 0.900 | [0.70, 0.97] |
| zur-enchantress | 60 | 1.57 | 0.867 | [0.76, 0.93] |
| sisay | 20 | 0.95 | 0.850 | [0.64, 0.95] |
| edgar-vampires | 400 | 0.82 | 0.657 | [0.61, 0.70] |
| gishath | 20 | 0.60 | 0.550 | [0.34, 0.74] |
| ur-dragon | 60 | 0.42 | 0.350 | [0.24, 0.48] |
| **heliod** | **20** | **0.30** | **0.300** | **[0.14, 0.52]** |

Heliod is last on both columns and its deck is named for the back face of a card
it casts 0.30 times a game. Read beside `mana-analysis` — {2}{W}{W} against a
white on-curve probability of 0.446 on lands alone, in a base of **14 Islands to
3 Plains** — that is not an AI-piloting story, it is a colour-source story, and
it is the first thing to fix.

### The id suffix is what separates a cast from an arrival

Forge writes a resolving spell and an ability of a permanent already on the
battlefield with the same leading text:

```
Resolve Stack: Heliod, the Radiant Dawn - Creature 4 / 4
Resolve Stack: Heliod, the Warped Eclipse (100) - Transform Heliod, the Warped Eclipse (100).
```

`Heliod, the Warped Eclipse` is a face of the commander card and `_is_commander`
matches faces on purpose, so the **obvious-looking normalisation — strip a
trailing `(nnn)` before matching — is the bug.** With it, this deck's commander
arrives on the turn it TRANSFORMS, which is later than it arrived and earlier
than the next cast. Swept across every tracked log: 77 transform resolutions in
the two shapes Forge emits, all 77 carrying the id.

An earlier cut paired each resolution to a PENDING cast instead. It was
defensible and it was deleted, because no real log distinguishes it from the id
rule and a mechanism nobody can write a failing test for is a mechanism nobody
is testing.

`commander_casts` counts SPELLS: a countered one counts and a recast after
removal counts. The gap between casts and arrivals is what tax and interaction
cost the deck, and it is the figure, not an error in it.

## Forge models the card correctly and the AI barely uses it

**2026-09-07, heliod.** *Heliod, the Warped Eclipse* grants flash to every spell
and reduces each by {1} for each card the opponents drew this turn. The pilot's
question was whether Forge could model it. It can, exactly:

```
S:Mode$ CastWithFlash | ValidCard$ Card | ValidSA$ Spell | Caster$ You
S:Mode$ ReduceCost | Type$ Spell | Activator$ You | Amount$ X
SVar:X:PlayerCountOpponents$CardsDrawn
```

All 100 cards in the deck exist in Forge's `cardsfolder`, every punisher trigger
is scripted, and `AILogic$ Always` on the transform means the AI does flip him.
**What the AI does not do is sequence for the ability.** Over 20 games our seat
cast 226 spells: 196 on its own turn and 30 on an opponent's — and 24 of the 30
were counterspells and removal, which are instants that would have been cast in
that window with or without the commander. Non-instants cast in the flash window
the deck is built around, in twenty games: **two** (a Talisman and a Loran).

The symmetric-draw half fares better than expected and is worth recording
because it contradicts the obvious guess: the AI **does** pull a lever that
helps opponents when the payoff is on its own board. Temple Bell was activated
14 times off 4 castings, Kwain 9 times off 5, and Iron Maiden / Ebony Owl
Netsuke / Viseling landed 5 / 6 / 5 times and dealt real damage.

Six cards in the deck carry Forge's own `AI:RemoveDeck` marker — `RemoveDeck:All`
on Psychosis Crawler and Skyscribing, `RemoveDeck:Random` on Forced Fruition,
Prosperity, Mystic Remora and Loran of the Third Path. That flag governs deck
GENERATION, not a supplied `.dck`, so the cards stay in and their triggers fire;
what it marks is that the AI has no logic for them. **Psychosis Crawler was cast
0 times in 20 games** — a {5} artifact creature with `SVar:NeedsToPlayVar:X GE3`
that the evaluator prices as a `*/*`. Same class as the Ashnod's Altar finding:
read correctly, never played.

**So a Forge win rate for this deck is a FLOOR**, in the sense `simulate`'s
PILOTING block already reports — and the floor is the honest number for a
comparison between two of your own lists, which is what `experiment` is for.

### What the goldfish cannot be taught here

The deck's clock is *opponent state*: Iron Maiden and Viseling read the number of
cards in an opponent's hand minus four; Ebony Owl Netsuke wants seven; the
commander's discount counts what the opponents drew. The goldfish has **one
opponent at 40 life who does nothing** and no opponents' draws at all, so none of
it is reachable — and a model of one of the five punishers would produce a figure
shaped like the clock that is not the clock.

The one thing that *is* reachable was swept and rejected: `whenever you draw a
card, each opponent loses N life` matches **exactly one card in 34,890**
(Psychosis Crawler). Of the 49 cards using `whenever you draw a card`, about
twenty put +1/+1 counters on something and three drain. That is a
one-card pattern in a shared model, and the card Forge never casts.

Absent, with the reason stated, is the answer. Forge measures the clock.

---

## The goldfish had no reason to cast 24 of a deck's 28 instants and sorceries

**2026-09-07, heliod.** The deck's entire plan is drawing cards.
`mean_extra_cards_drawn_by_turn` read **0.428 by turn eight**. The cause was
not the draw model — it was the CASTING side:

```
  instants/sorceries in the 99            28
    the model had a reason to cast         4   (3 tutors + Arcane Denial's cantrip)
    INVISIBLE to every casting loop       24   Braingeyser, Stroke of Genius,
                                               Prosperity, Skyscribing, Mathemagics,
                                               every counterspell, every sweeper
```

Removal and interaction are *correctly* invisible — the model has no opponent to
aim them at. What was not correct is that every X-cost draw spell was invisible
too, and those are the deck's engine.

**A magecraft channel was scoped and rejected on this ground.** The sweep found
only **8 cards** in 34,890 that draw off a `whenever you cast an instant or
sorcery` trigger, and — much worse — the trigger they need almost never happens
here: with four castable spells in 28, Archmage Emeritus would have fired a
handful of times a game and the number would have been driven by which spells
happened to carry a readable draw profile rather than by the deck. **A payoff
whose trigger the model cannot produce is worse than an unmodelled payoff: it
looks like a measurement.**

### X-spell draw: 33 in the corpus, 28 credited

```
  13  Draw X cards                     6  Target player draws X cards
   3  Each player draws X cards       11  the same with a rider
```

All four shapes give the CASTER the cards, so one rule covers them.

**The fixed part is already `cmc`** — Scryfall counts `{X}` as zero, so Stroke of
Genius at `{X}{2}{U}` has cmc 3 and Braingeyser at `{X}{U}{U}` has 2.
`x_draw_multiplier` is the count of `{X}` symbols, because `{X}{X}` buys one card
per two mana.

Two things had to be true, and each was a bug first:

* **`draw_profile` returned before the block ever ran.** Its `_DRAW_RE` guard
  wants a written-out quantity ("draw two cards") and does not recognise "draws
  X cards", so the whole family returned an all-zero profile **with `unmodelled`
  still None** — the one value that means "there is nothing here to model".
  Braingeyser read as a card with no draw on it.
* **The spell must be cast LAST.** Because `cmc` is only the fixed part, every
  cheapest-first loop in the module would have fired Stroke of Genius on turn
  three for X=0 — drawing nothing and burning the card. `X_DRAW_MIN = 2` is the
  floor and the only authored number in the channel; casting after every other
  loop also makes the figure conservative, since the spell can only ever spend
  mana nothing else wanted.

Result on heliod, and the size of what was missing:

```
  mean extra cards drawn   @T6    0.197 -> 1.214
                           @T8    0.428 -> 2.329      5.4x
                           @T10   0.722 -> 3.471
  board power @T6                 2.444 -> 2.547      it did NOT starve the board
```

**FIVE REFUSALS, each read card by card, each with its own test.** Expansion //
Explosion (CR 202.3d — a split card's mana value is both halves, so the fixed
part is wrong by the other half); Ingenious Mastery (an alternative cost under
which X is 0); Occult Epiphany and Read the Runes (draw X, discard X — a filter,
and Read the Runes has the *cheapest* fixed cost in the family, so crediting it
would have made the worst card read as the best); Skeletal Scrying (X is cards
exiled from a graveyard this model does not have). The rule divides the
remaining **mana pool**, so it is only correct where X is bought with mana.

The opt-in contract held on the fleet: edgar-vampires, goblin-storm and ur-dragon
each moved by **three lines** — the `model_version` sha and the assumption text —
and not one figure.

### A test that names a deck inherits that deck's declarations

`test_a_deck_that_has_not_opted_into_draw_has_no_draw_series` used **heliod** as
its example of a deck with `model_draw` off. Heliod declared `model_draw` the
same morning, so the test began asserting the opposite of the deck's own targets
file and failed — correctly, and not because of the model change it failed
alongside. It now reads `ur-dragon` **and asserts that stand-in has not opted
in**, so the day ur-dragon declares draw this fails loudly instead of passing on
a premise that stopped being true.

---

## A pre-registered threshold read off a baseline that has since moved

**2026-09-07, heliod. Twice in one session, the same way.** `--objective` takes
a NUMBER, and the number has to come from somewhere. Both times it came from a
figure printed by a different run than the one that would grade it:

| branch | objective | champion when the line was set | champion when it was graded |
|---|---|---|---|
| `teeth-v1` | `board_power_6 >= 2.80` | 2.438 — a bare `goldfish` run | **2.366** — `net_change`'s own seed |
| `survive-v1/v2` | `interaction_6 >= 0.78` | 0.795 — a net-change run at 09:xx | **0.661** — after the X-draw model landed |

The first is a seed difference: `goldfish` runs the deck's own seed and
`net_change` runs 20260826, so the same list reads 2.438 and 2.366 and neither
is wrong. The second is worse and more instructive — **a model change moved the
control by 13 points between the branch being opened and being graded.**
Teaching the goldfish X-spell draw made big X spells eat the whole remaining
pool, which is correct play and correctly leaves less mana for interaction. The
control fell 0.795 → 0.661 and the branch's line did not.

Both objectives stand as written and both read NOT MET, because a
pre-registration that gets rewritten once it is inconvenient is not one. But the
reading changes completely: against the control it was actually graded on,
`survive-v2` **improves** the axis its objective was guarding, +0.027.

**Take the number from a `net-change` run of the CURRENT champion, immediately
before opening the branch** — not from `goldfish`, not from a table printed
earlier in the session, and never from before a model change. `net_change` is
the harness that grades it, so it is the only harness whose baseline means
anything.

### The third instance, and the first where the objective cannot fail (2026-09-13)

**sharknado / `recon-v1`.** Same failure, opposite direction, worse consequence.
The branch was opened 2026-09-10 with `extra_cards_8 >= 1.2` against a champion
that read **0.632** that morning (`c46a3952`) — a pre-registration of roughly
double the measured figure, which is a defensible line to draw.

Then the model moved twice, both times correctly:

| when | what landed | champion `extra_cards_8` |
|---|---|---|
| 2026-09-10 | the discard channel — wheels became visible at all | 0.632 → **7.509** |
| 2026-09-13 | the activated wheel — Jace's Archivist and two others | 7.509 → **10.087** |

The line did not move, so it now reads **MET at 8.334** — and would read MET at
any value the branch could plausibly produce. The first two instances were
objectives that became unfairly hard. This one became one that **cannot fail**,
which is the same defect wearing the answer the pilot wanted to hear.

It stands as written, by the rule above. What it costs is that `recon-v1` has no
live grade: the row that was supposed to be able to say no says yes by
construction, and the honest reading has to be taken off the measured table
instead, where the branch is **−1.75 extra cards by T8 (−17%)** and **−5.27
damage @T10 (−11%)** against **+26 games per 100 interaction-affordable @T6**
and **−12 per 100 on the double stall**. That is a trade, and it is the pilot's
call, not the objective's.

**The branch's `why` text is now false in its own terms as well**, and this is
the part worth carrying forward: it reads *"THE GOLDFISH READS THE SHARK HALF:
Brallin's discard trigger has no channel and a wheel's 'draws that many' is not
'draw N', so the draw axis is a proxy."* Both halves of that sentence were true
when written and neither is true now. **A branch note describes the model that
existed when it was opened.** Re-read it against `meta.model_version` before
trusting a word of it.

One thing the branch got right without being able to measure it: of the four
wheels `recon-v1` cuts, **all four are SHUFFLE wheels** (Molten Psyche, Time
Reversal, Whirlpool Warrior, Winds of Change) and it keeps every DISCARD wheel.
The pilot's recon call and the measurement that only became possible three days
later agree exactly.

And the deeper point, which is why this cost nothing real here: on this deck the
goldfish objective can only ever be a RISK check. It has no opponent that
attacks, so a taxer and a sweeper are invisible; it reads the draw payoffs on
`teeth-v1` as `unmodelled`; and every card in `deckout-v1` is invisible to it.
The reward is decided in Forge for all three. An objective that cannot see the
upside should be chosen to grade the branch's specific DOWNSIDE and read as
nothing more.

---

## teeth-v1, called at 36 of 100 — and the drift report that would have sent the pilot to the wrong sleeves

**2026-09-07, heliod.** The A/B of v1.0.1 against `teeth-v1` was stopped by the
pilot at arm B 36/100. No experiment record exists; the logs are gitignored, so
this is the record.

```
  arm A  v1.0.1     100/100   82 decided   heliod 0.244 [0.16, 0.35]
  arm B  teeth-v1    36/100   35 decided   heliod 0.147 [0.06, 0.30]
```

Arm B's trace ran 0.200 → 0.150 → 0.133 → 0.147 and did not wander after the
first ten games. The intervals still overlap and **no interval on the difference
was ever computed**, so this is a pilot's judgment call on a consistent
direction, not a measured refutation. The gap is ~9.7 points against a run whose
MDE at the full 100 was ~17.5 — it was heading for INCONCLUSIVE even completed,
which is the more useful thing to record than the direction.

**A finding that looked real for twenty minutes and was not.** Arm B was
clocking out 0/24 where arm A clocked out 17/100, and that read as "teeth-v1
makes games end" — the branch doing what its hypothesis says. The second
measurement killed it:

```
  rounds      A 21.19   B 19.79   Δ 1.40    95% [-0.70, +3.50]   contains zero
  seconds     A 246.6   B 125.3   Δ 121.3   95% [+69.9, +172.7]  EXCLUDES ZERO
  clocked out A 0.170   B 0.000   Δ 0.170   95% [+0.019, +0.256] EXCLUDES ZERO
```

The games are not shorter. They take **half the wall time at the same number of
rounds** — and `-c` is a wall clock. Arm A ran while the same eight cores were
building `sim-progress`, running test suites and sweeping the corpus; arm B ran
while the machine was quiet. That is oversubscription censoring games (the test that once re-derived it from tracked runs was deleted 2026-10-05: its 2x claim stopped holding across tables)
happening live, inside one experiment, between its two arms: the arm played on
the busier machine was piloted worse and had 17% of its games censored out of
the denominator.

**Run nothing else on the machine while an experiment is running.** The
confound here pointed against the observed direction — arm B had the easier
conditions and still lost — so the reading survives, but only by luck.

### The drift report counted entries, not copies

Found immediately after, while confirming what cardboard to move:

```
  the sleeved list is V6; the repo is at V7 — pull 0, add 0 to bring the
  cardboard level
```

v1.0.1 IS five Islands becoming five Plains. `diff_vs_working` compared name
MEMBERSHIP, and both names are in both lists, so the deck's largest measured fix
was invisible to the one report whose entire job is telling the pilot what to
physically move. `deck_history._entries` has always returned name → copies and
nothing used the second half of it.

The same defect the magazine era already paid for — counting entries once
published "18 lands" for a 33-land deck — in a different file, six weeks later.
The naive fix is wrong too and has its own test: `10 Island → 8 Island` is a
two-card swap, not a ten-card one.

---

## The win rate is a low-power endpoint, and the arithmetic was always available

**2026-09-07, heliod.** A 100-game-per-arm A/B was launched against a 0.244
baseline. It ran four hours. Its probability of detecting a real ten-point
improvement was **0.34** — more likely to miss than to find, and knowable in a
millisecond from functions `stats` has carried since the statistics went in.

```
  games/arm at 80% power, baseline 0.244
    +0.05   >1000/arm       —
    +0.10     324/arm    15.4 h
    +0.15     149/arm     7.1 h

  what 100/arm actually resolves:  +0.190 or larger
  power at 400/arm vs +0.10:       0.88   (24 hours)
```

Three things make the win rate expensive here and no care fixes any of them: it
is BINARY, it is RARE (0.244), and **a clocked-out game has no winner so it is
DISCARDED** — 17% of one run's games were paid for and thrown out of the
denominator.

Mechanism endpoints are better but not enough better. Commander uptime
(mean 3.22, sd 4.01), the removal rate per resolution (0.451), the share of
games the commander never lands (0.368) all still want 100–500 games per arm.
**Forge cannot price a three-card swap.** Reserve it for changes of fifteen
points or more, and for the final gate on a list about to be sleeved.

### Decompose the failure instead of A/B-ing the fix

The cheapest rigorous answer is usually not a faster experiment — it is a
smaller question, answered from logs already on disk.

Asked of the protection branch: hexproof and shroud stop TARGETED removal and
nothing else, so the only empirical question is what share of the problem that
is. 78 departures across 220 games already recorded:

```
  39   50%   MASS — three or more permanents left the same turn
  24   31%   TARGETED — hexproof/shroud stops this
  15   19%   other / unattributed

  removal rate now            78/173 resolutions = 0.451
  with hexproof              ~0.312
```

Greaves and Boots buy about **fourteen points of commander survival**, derived
from the rules plus a decomposition of games already paid for, with no new
simulation at all. Half the problem is sweepers, which no amount of hexproof
touches — which is also the argument against loading up on protection.

Measured beside it: **0 of 173 commander casts were countered.** Eight
counterspells have never once protected the engine.

### The preflight

`experiment` now prints what the run can see before it launches, and takes
`--detect DELTA` for the change the pilot actually cares about. It PRINTS and
never refuses: an underpowered A/B is still legitimate as a noise floor, a smoke
test, or the first half of a bigger sample, and a gate blocking it would be a
validator firing on correct use.

The baseline is the deck's largest measured run **against the same table**, and
is ABSENT when there is none — a preflight computed from a default rate would be
a number about nothing that looked exactly like a number about something.

---

## `regen` is not the fleet, and a model change needs the fleet

**2026-09-08.** One model change — teaching the goldfish X-spell draw — left
**54 tests failing**, and it was not discovered for several hours because the
command whose whole job is "regenerate after a model change" does not do that.

```
  manamap pilot regen              22 targets: SLEEVED decks only
  the live fleet                   7 decks, and zur-enchantress alone has
                                   14 branches with tracked goldfish_metrics
```

`regen.targets()` is documented and deliberate: a sleeved deck is played so its
figures are kept current automatically; a bench deck is malleable and rebuilding
it on a sweep measures a list that will be different tomorrow. That rule is
right for a **decklist** change, which is local to one deck.

**It is exactly wrong for a MODEL change, which invalidates every deck at
once** — sleeved, benched, and every branch of both. CLAUDE.md says
"regenerate the fleet after any model change" and the command that sounds like
it does that quietly covers a third of it.

What it actually took:

```
  for d in edgar-vampires gishath goblin-storm heliod radagast ur-dragon zur-enchantress
      manamap pilot regen --slug $d          # NAMED = manual = this deck, sleeved or not
```

`zur-enchantress` alone was 69 targets and 173 seconds.

**Two further artifacts no `regen` stage touches at all**, both found the same
way:

* `versions.json` — rebuilt by `deck-version <slug> list --write`, and it must
  be a SEPARATE COMMIT from the decklist that changes it, because a version's
  sha is not knowable inside the commit that creates it. Its freshness test says
  so in its own failure message.
* `info.json` on decks the sweep skipped, which drifts on any change to
  `deck_info` — and three landed today (the drift block, the run ordering, the
  archetype fix).

### The shape of it

A staleness rule keyed on **who plays the deck** cannot protect artifacts
invalidated by **what the code computes**. Those are different blast radii and
the tool models only the first. Until `regen` grows a model-change sweep, the
loop above is the fleet, and it belongs in the same commit as any change to
`goldfish.py`, `parse.py`, `diagnostic.py` or `deck_info.py`.

The tests caught all of it. Nothing shipped wrong — but they caught it hours
late, on a full `make test` nobody runs between commits, and the failures
pointed at zur-enchantress branches rather than at the one line in `goldfish.py`
that caused them.

---

## A test that names a deck inherits that deck's decisions

**2026-09-08. Four instances in one day**, all the same shape and none of them
caught by anything except a full suite run hours later.

| test | named | assumed | what changed |
|---|---|---|---|
| `..._has_no_draw_series` | heliod | declares no `model_draw` | it declared one |
| `test_heliod_primary_win_line` | heliod | runs Hullbreaker Horror | the card was cut |
| `..._names_the_flag_rather_than_reading_zero` | heliod | declares NO flag | it declared two |
| `..._byte_identical_with_the_flag_absent` | the un-opted POOL | at least three decks | heliod opted in, leaving two |

Every one of them was a good test. Each encoded a real invariant — absent means
absent, an undeclared win line is measured by nothing, the opt-in contract holds
across decks. What failed was the EXAMPLE, not the rule.

**The fix is the same every time and it is not "stop naming decks".** These
invariants are about real decks and cannot be tested on fixtures alone. The fix
is that a test naming a deck must ASSERT THE PREMISE IT DEPENDS ON, so the day
that premise dies it fails saying so instead of failing on the conclusion:

```python
    assert not _d.get("model_draw"), (
        "gishath has opted into a model — this test needs a deck that declares "
        "NOTHING, or it proves nothing")
```

Without that line the failure reads as a broken model. With it, it reads as a
test pointing at the wrong deck, which is what it is.

The fourth one is different and worth separating: the un-opted pool SHRINKS as
the fleet adopts a model, and that is adoption rather than decay. The floor
moved 3 → 2 rather than the invariant being abandoned — two decks still prove it
ACROSS decks, which is what the guard is for, and one would not.

**Cost of not doing this:** a decklist change on 2026-09-07 left four tests
asserting the old deck, and they were found on a `make test` nobody runs between
commits, pointing at zur-enchantress branches and a "broken" diagnostic model
rather than at the check-in that caused them.

---

## The paper lock is a claim, and drift only runs one way

**2026-09-08, ur-dragon.** The repo was two cards behind the cardboard for two
days and nothing could see it.

`landbase-v1`, opened 2026-08-30, proposed six land swaps and was **never
merged** — no merge commit, the branch directory gone, every "out" card still in
`decklist.txt` and no "in" card ever arrived. The pilot sleeved two of the six as
proxies. The lock was then set on 2026-09-01 at V3, note "sleeved 2026-09-01",
asserting that the committed list is what is in the sleeves.

```
  repo V3 · lock V3 · drift report: nothing to do
  actual: Volcanic Island and Plateau in the sleeves, Shivan Reef and
          Stormcarved Coast in the file
```

**`deck-version <slug> paper` records a CLAIM; nothing verifies it, and nothing
can.** The drift report compares the lock's version against the repo's, so it
fires only when the REPO moves ahead. Paper moving ahead of the repo is
invisible by construction — there is no sensor on the other side.

Consequences, all real:

* every Forge run, every audit and every goldfish figure for two days measured a
  list the pilot was not playing;
* `deck-branch merge` refused cards as "unsourced" that were physically in a
  sleeve;
* `regen.is_pinned` kept the whole chain current for a list that did not exist.

### What to do about it

**A lock is only as good as the check-in behind it.** Set it from a
`check-in --from <the paper list>`, which diffs and refuses, and not from
"I think that's what I built". The two-minute version is `deck-version <slug>
show <ref> --full` read against the sleeves before pinning.

**And when a deck's definition comes into question, WITHDRAW the lock** —
`deck-version <slug> paper --clear` — rather than leaving it asserting. It went
back to ON THE BENCH here, which is the honest state for a list nobody has
verified, and it stops `regen` maintaining figures for a deck whose contents are
unknown.

The two-card fix is committed as v1.2.1 and is correct as far as it goes. What
is NOT established is the other 98, and a lock claiming otherwise is worse than
no lock: it is the difference between "unknown" and "wrong".

---

## Zur: the goldfish and Forge were not measuring the same deck, twice over

**2026-09-08.** zur-enchantress reports `kill_by_10` of **0.888** in the goldfish
and a **0.118** win rate in Forge. That is not a modelling gap. It is two
separate defects, and between them almost none of this deck's evidence describes
the list a pilot would sleeve.

### 1. Six of eight Forge runs played the WRONG COMMANDER

```
  2026-09-06  n=59   Zur, Eternal Schemer   sha a17b0f4a   <- the current list
  2026-09-06  n=60   Zur, Eternal Schemer   sha a17b0f4a   <- the current list
  2026-09-05  n=60   Zur the Enchanter      sha 734250b0
  2026-09-04  n=60   Zur the Enchanter      sha 9354852a
  2026-09-04  n=60   Zur the Enchanter      sha 7cdbafc9
  2026-09-04  n=60   Zur the Enchanter      sha 9354852a
  2026-09-03  n=40   Zur the Enchanter      sha c63a265e
  2026-08-27  n=20   Zur the Enchanter      sha e71580a7
```

They are different cards. **Zur the Enchanter** attacks and tutors a
mana-value-3 enchantment onto the battlefield. **Zur, Eternal Schemer** grants
deathtouch, lifelink and hexproof to enchantment creatures and carries
`{1}{W}: target non-Aura enchantment becomes a creature with power and toughness
equal to its mana value`. The deck was rebuilt around the Schemer; 300 of its
419 Forge games were played by the Enchanter, who cannot animate anything.

Nothing flagged it. The runs are correctly stamped with their own decklist shas
and `deck-info` marked them stale — but "stale" and "different commander" render
identically, and a reader comparing eight runs sees eight runs.

### 2. Forge's AI does not use the ability the deck is built on

Isolating the two runs that DID play Zur, Eternal Schemer:

```
  standard pod  n=60   Zur cast 94, resolved 97
                       ANIMATE fired 3 times, in 3 of 60 games   (5%)
  value pod     n=59   Zur cast 74, resolved 76
                       ANIMATE fired 2 times, in 2 of 59 games   (3%)
```

**The commander lands and then does nothing.** Five activations across 119
games, on a deck whose entire kill is animated enchantments.

It is not AI paralysis: the same seat activated Profane Procession 37 times,
Thassa 8, Caretaker's Talent 8. It uses activated abilities freely. It will not
use THIS one, because the evaluator cannot price "a 3-mana enchantment becomes a
3/3" as worth `{1}{W}`.

Third instance of this class recorded here, after Psychosis Crawler (0 casts in
120 games) and Ashnod's Altar (0 for 59 castings). **THE FORGE AI WILL NOT PAY
FOR A BENEFIT ITS EVALUATOR CANNOT SEE**, and an animate is the purest case: it
spends real mana and produces no immediate board change the evaluator counts.

### What this invalidates

`goldfish` applies `model_commander_animate` every turn it can afford. Forge
applies it in 5% of games. So the goldfish is a CEILING and Forge is a FLOOR,
and the truth is between them where neither instrument reaches.

**All 24 zur branches were graded on `kill_by_8`** — a figure produced almost
entirely by that ability. The ranking measured how well each list feeds an engine
that, at a table Forge plays, does not run. `deck_branch.MEMBERSHIP_AXES` already
refuses authored engine axes because the same hand sets the target and reads the
verdict; this is the same failure through a different door — an axis whose value
depends on an ability one instrument fires and the other does not.

**Do not open a 25th zur branch graded on `kill_by_8`.**

## Haste was read per card, so the enabler the pilot asked for was worth nothing

**2026-09-11.** The Ur-Dragon captain's log asked for haste by name — a Dragon
that swings the turn it lands — and the doctor's first-ranked add was Rhythm of
the Wild. The goldfish could not measure the ask. `combat_profile` read
`\bhaste\b` anywhere in the oracle text and stamped it on the card itself, so:

- **Temur Ascendancy and Dragon Tempest, both in the 99, granted nothing.** An
  enchantment with `haste: True` is not a creature; the flag was never read.
- **A creature that GIVES haste attacked on arrival.** Regisaur Alpha ("Other
  Dinosaurs you control have haste"), Ogre Battledriver, and every creature
  whose ETB reads "it gains haste" (Hellkite Courser, Puppeteer Clique). The
  corpus count of "hasty" creatures was 1,250; the keyword count is **705**.
- **Riot's reminder text made Spider-Punk hasty**, and "has haste as long as"
  (Markov Crusader) was unconditional.

Now one profile key, `team_haste`, says WHO a card grants haste to — `"all"`,
`"nontoken"` (riot, read as always choosing haste), `"flying"` (Dragon
Tempest) or a creature type (Karrthus: Dragon) — and it is read at the attack
step against the type line and keyword flying that ride beside each body.
Sweep: **49 grants** in the corpus, 29 team-wide, 18 typed across 14 types.
Three class grants ("artifact creatures", "multicolored creatures", "equipped
creatures") are left unread and named rather than widened to the team; so
are Anger (from the graveyard), Crashing Drawbridge (an activation), and the
two lands whose mana carries haste (Hall of the Bandit Lord, Arena of Glory),
because this model taps no particular land. A haste enabler is in the
combat-payoff casting predicate in the same commit, as the rule requires.

**What it is worth on an unopposed table: about 1.5 points.** ur-dragon's two
grants, blinded and restored on the same seed: kill-by-T8 0.779 → 0.795,
kill-by-T6 0.259 → 0.276. The clock is set by mana, not summoning sickness,
which is the reading a haste branch has to be measured against before it is
sold on the pilot's instinct. The fleet drift: ur-dragon up (the grants),
ingris-infect and sharknado down a point (Skithiryx and Ingris buy haste with
an activation the model does not price; the Locust God's haste belongs to its
tokens, which carry no keywords here), zur-enchantress up on **drain** —
which is the second finding.

### The parallel lists did not follow a sacrifice or a death

`creature_types` is index-aligned with `battlefield` at the one door, and two
sites rebuilt or popped the battlefield without touching it: the sacrifice
site (`battlefield[:] = kept`) and the deaths channel (`sort` then `pop(0)`).
After either, `zip(creature_types, battlefield)` paired a creature's power
with another creature's type line, so the typed lifelink grant — and, had it
shipped on top, the typed haste grant — read garbage. Both sites now carry the
parallel lists with them; zur's drain figure moved because the lifelink it
grants by type is finally credited to the right bodies. The lesson is the one
the file already states at the door: a parallel list is a liability at every
site that mutates the list it parallels, and there were two such sites the
comment did not know about.

### A record with two definitions of "decided"

Found by the skeptic on the same prescription: the ur-dragon standard-v3 record
said 0.353 on 34 decided in `analysis.seats` and 0.343 on 35 in `summary`. The
35th game was a simultaneous loss — every seat at 0 on one turn, `draw: true`
without `truncated` — which `summary` counted as decided (games minus
clock-outs) and the seat block did not (games with a winner). Six records
carried the split. One definition now, in `run()`, `analyze()` and
`validate_sim`: **a decided game has a winner**, and `summary.decided` equals
the sum of `summary.wins`. The six records were re-derived from their own
`games` block, which is enough because the block carries `draw`, `truncated`
and `winner` per game.

## Gishath: Forge saw the win condition and the goldfish saw a 7-power body

**2026-09-11.** The pilot checked in the paper Gishath list and asked whether
the bench could model the deck's win condition: Gishath connects for seven,
reveals seven, and every Dinosaur among them enters free. The audit split
cleanly by instrument.

**Forge read it.** On the V3 record at standard-v3 the trigger resolved 42
times in 40 games, Gishath was assigned to attack 60 times (0.70 connects per
attack), and the log shows the AI taking the Dinosaurs: Vaultborn Tyrant,
Regal Behemoth and Trumpeting Carnosaur entering off one trigger and their own
triggers firing. Mirari's Wake's mana trigger resolved 55 times, Ghalta and
Mavren's 52. Forge is a sound gate for this deck.

**The goldfish read none of it.** Gishath was a 7-power body with the trigger
flagged unreadable; Mirari's Wake produced nothing; Rishkar's Expertise (draw
equal to the greatest power) drew nothing — and so did Return of the
Wildspeaker in Ur-Dragon, which nobody had noticed; Earthshaker Dreadmaw drew
ONE card instead of one per Dinosaur; Ghalta and Mavren was unreadable; and
the deck had never declared `model_combat`, so 33 cards were DARK. Every
goldfish figure on the deck was a resource curve with the plan missing.

What was taught, each with a corpus sweep locked in
`tests/test_pilot_goldfish_gishath.py`:

- **The commander's combat-damage reveal, DECLARED per deck**
  (`model_commander_combat_reveal`: type, connects-per-attack, source) the way
  Zur's animate and the attack tutor are, because one card in the corpus has
  it. The rate is REQUIRED with its source — this model has no blockers and
  would connect every attack — and the record reports fires and bodies per
  game beside the declared rate. A revealed creature enters through the one
  door with every registration a cast body gets, minus what a cast is.
- **A land-mana bonus** (4 cards): one extra mana per land from the turn after.
- **Draw equal to the greatest power** (3 instants and sorceries): resolved
  against the board at cast, held while the board is empty.
- **Draw a card per type on entry** (10 cards): counted on the board it joins;
  the one-card read is suppressed so it is not double-counted.
- **An attack token as big as the best other attacker** (Ghalta alone): the
  second-largest swing.

Named and left: Etali, Primal Storm (casting other players' cards is outside
this model) and Hunter's Insight (draw equal to one creature's combat damage).

Gishath on the sleeved list, before and after, same seed: kill by turn 8 0.56 →
0.62, mean kill turn 8.2 → 8.0, the reveal firing 1.95 times a game for 4.1
free Dinosaurs. Forge's AI put the same trigger to work 1.05 times a game.

### The second batch: an entry trigger fires on its type, and a cast fires what listens

Same day, same deck, from the pilot's cart. Six more shapes, each swept and
locked in `tests/test_pilot_goldfish_gishath.py`:

- **A typed entry trigger fires on its type.** `_ETB_TRIGGER_RE` threw the
  subject noun away, so Dragon Tempest fired on a Bird of Paradise and Lathliss
  minted a Dragon for a mana dork — the `.*`-where-the-noun-lives trap from
  `docs/gotchas-analysis.md`, one file over. 144 cards in the corpus name a type
  in their entry trigger. Tokens carry no type line in this model and PASS the
  gate, so Lathliss's Dragons still fire Tempest (stated). Molten Echoes is the
  same gate keyed to the deck's chosen type, copying every nontoken entry.
- **A cast fires what listens for a cast, at every door, the commander's
  included.** Draw on a spell of mana value N or more (Up the Beanstalk, 3
  cards), draw for a paid mana on a creature cast (Lifecrafter's Bestiary, paid
  only when the pool has it), damage on casting a creature of power N or more
  (Sarkhan's Unsealing). The commander's cast used to fire nothing at all.
- **A spell that has a creature deal its power** (Chandra's Ignition) deals the
  biggest body's power once; the wipe half has nothing to hit here.
- **A tutor onto the battlefield** (Savage Order, Natural Order, 3 spells)
  sacrifices the smallest 4-power nontoken body and fetches the highest-power
  creature of the named type the library holds through the free-entry door.
  `is_tutor` had refused Savage Order because its text opens with the
  additional cost, not the search.

The typed gate is the one that moves other decks: ur-dragon's damage falls
where Tempest and Valkas had been paid for dorks entering. That is a
correction, and every branch on the deck was measured under the same
over-credit in both arms, so the comparisons stand.


## 2026-09-11 — The real table pooled five tables, and counted clock-outs as losses

`net_change.forge` said "pooled within one pod only" in its docstring and globbed
every record under `sim/`. On edgar-vampires the champion arm was **136/840 across
vito-era, standard, standard-v2, playgroup and standard-v3** against a fear-v1 arm
played at standard-v3 alone, and the block reported the branch **+0.063** on that.
It also divided both arms by `analysis.games`, so thirteen clock-outs — which have
NO winner and are excluded from the record's own `win_rate` — counted as losses:
the block read the champion at 0.20 while the record beside it said 0.296.

Restricted to the one table both arms sat at, over decided games, the same branch
reads **champion 27/73 (0.370), branch 9/33 (0.273), delta −0.097, CI [−0.265,
+0.101]**. The sign flipped. Both readings are underpowered (MDE 0.29), so nothing
was decided on the wrong one, but a reader of the branch page had a control from
tables the branch never sat at, with no name on it.

The block now carries `pod`, `basis`, `all_games` beside `games`, and
`other_tables` naming what was NOT pooled; the print and the branch page show the
table. No common table is `available: false` with the reason, not a pool across
two nulls. `tests/test_pilot_net_change.py` re-introduces both defects on a fixture
that reads 58/355 or 8/40 under either regression. Six of the seven branch reports
with a Forge record changed bytes on regeneration (sharknado's has no champion run);
zur-enchantress's two now read UNAVAILABLE, because the champion has never sat at
standard-v3 and the pooled block had been comparing against `standard` and
`standard-v2` without saying so. None of the seven had a decision resting on the
Forge line.


## The speed sprint, 2026-08-30/31

*Moved verbatim out of `PLAN.md` on 2026-09-12 — measurements with no other home.*

### What was measured

**The complaint was that iteration had become heavy: questions slow, fleet
regeneration manual, and fidelity surprises discovered after the run.** Three
audits (tests, simulation, interactive path) said the Python simulation was
never the bottleneck — the fan-out around it was. The whole fleet regenerates in
**78 seconds of CPU**; the same regeneration used to cost **6-9 MILLION agent
tokens**, and that ratio was the entire problem.

**The single most expensive line in the repo was a provenance stamp.**
`goldfish.model_version()` is a sha over the whole of `goldfish.py`, and ten
`AGENT_ROUTINES` declarations hashed the file it is stamped into. A COMMENT edit
moved the digest on every deck and hard-MISSed strategic-frame, pilot-notes,
tutor-guide, deck-diagnosis, every decision and every prescription. Measured over
four real goldfish commits: **45 artifacts stamped stale, 31 with figures that
actually moved — 31% of the spend bought nothing**, and `deb711e` changed one
docstring line and invalidated the fleet. Excluded from the fingerprint; the
stamp stays in the artifact and `model_staleness` still reports it. The next
commit proved the point — a 30% goldfish speedup that moved no figure at all and
cost nothing.

| | before | after |
|---|---|---|
| `query-rules` / `query-strategy` | 6.93s | **0.16s** (43x) |
| `deck-facts` | 1.44s | **0.14s** |
| `deck-audit` | 2.26s | **0.59s** |
| `deck-info` | 7.8s | **1.25s** |
| `mde_proportion(0.25, 200)` | 2.17s | **0.21s** |
| goldfish (edgar) | 5.70s | **3.96s** |
| whole-fleet regen | a hand-written shell loop | **78s** |
| `make test` (warm) | ~101s | **~74s** |

**`manamap serve` is a warm worker.** Every CLI invocation was a cold process and
every memo is per-process — including the frozen MiniLM behind `query-rules`,
~8s to build and thrown away each time, while `rules-lookup` tells the agent to
"try several phrasings". `/api/cli` runs read-only pilot commands in the warm
process behind an allow-list; the terminal routes to it when one is listening and
**fails open** on any error. It holds the modules it started with, so restart it
after a code change.

**`manamap pilot regen`** rebuilds the fleet in dependency order, parallel across
targets — 72 targets, 78s, **bit-identical** (`git status data/` empty after).
Parallel across DECKS, never across games: one `random.Random(seed)` is threaded
through all 10,000 games, so splitting them would re-base every figure.

**`manamap pilot model-coverage`** answers the fidelity question in the other
direction — not "what did the channel miss" but "what would this deck need, and
is it switched on". **236 DARK cards across the fleet**; gishath is a Dinosaur
deck with 33 cards whose combat the model was told not to look at. `goldfish` and
`net-change` print it as a PREFLIGHT, so it arrives before the games.

**Forge's `-c` clock ends a game's accounting, not its AI thread.** Two tracked
20-game runs took **3.7 and 4.2 hours** with 95% of the wall claimed by no game.
Jobs are capped now; checked against all 18 tracked runs, the two pathological
ones die and **all sixteen others survive** — **7.1 hours** on that set.

**Three statements that were false, now corrected in place:** `forge.ASSUMPTIONS`
claimed a clock-hit game is recorded as a draw (it carries a winner — 75 of
edgar's 400 games, 19%, with zero recorded draws); `SEEDED_NOTE` claims
byte-for-byte replay (it diverges at game 1 on the 400-game run); and
`docs/agent-cost.md` claimed no Python spawns a subprocess (`serve.py`'s `ask`
shells out to `claude -p`).

Full record: `docs/gotchas-bench.md`.



## A granted mana ability belongs to whoever received it (2026-08-31)

*Moved verbatim out of `PLAN.md` on 2026-09-12.*

### What was measured

**Found by the mana sweep for the encoder's `mana_repeatable` field, which is the
cross-pollination working: a change in `training/` audited a function in
`pilot/`.** `goldfish.produced_mana` counted every quoted ability as the card's
own — **145 corpus cards, 8 of them sleeved across five decks, five in kinnan**.
Leyline Immersion, an Aura, read as a five-mana rock.

**THE OBVIOUS FIX IS WRONG AND THE SWEEP IS WHAT SAYS SO.** Stripping quoted text
zeroes fifteen cards that are correct: Citanul Hierophants grants `{T}: Add {G}`
to "creatures you control" and IS a creature, as are Gemhide Sliver, Enduring
Vitality, Inga and Esika, Katilda, Sachi and seven more; Dryad Arbor, Jasconian
Isle and Gobland carry theirs in reminder text about themselves. The question is
not "is it quoted" but **is this card a member of the class it grants to** —
`produced_mana` takes `type_line` to answer it and defaults to reading every
grant as foreign, because overcounting tells the model it can cast things it
cannot.

Two bugs found while fixing it, both by the sweep: the backward window **crossed
a clause** (Sachi opens "OTHER Snake creatures…" then grants to "Shamans you
control", which she is), and `it has` was **too loose** (in Jiang Yanggu the "it"
is the recipient; in Llanowar Mentor and The Bus Runner it is a token created a
sentence earlier). **And one guard deleted**: a second, wider window written for
those four cards changed ZERO readings across all 34,890 — a bug probe caught
that it could not fail, and a guard that guards nothing is worse than none.

Sweep: 133 readings changed, 15 quoted grants kept as the card's own (each read
individually), 34,742 untouched. Corpus nonzero 1,975 → 1,848. Fleet regenerated
(72 targets, 92.7s); **gishath's commander cast-by-turn-6 drops 0.189 → 0.170**,
the honest direction once phantom mana stops counting.


## A wheel on a permanent is an ACTIVATED ability, and the model read three of them as vanilla bodies (2026-09-13)

`draw_profile` credits `wheel_draws` on an **Instant or a Sorcery only**, and
that gate is right — a permanent carrying *"each player discards their hand,
then draws…"* carries it as the effect of an **activated ability**, paid for on
every use, not as a spell you cast once. But nothing else read that sentence, so
a permanent carrying it read as a body with no text on it.

Found on **sharknado** (Shabraz / Brallin), the deck whose entire plan is
wheeling. Twelve cards in the 99 are wheel-shaped; the model saw nine:

| card | what the model saw before |
|---|---|
| Jace's Archivist | a 2/2 Vedalken Wizard |
| Magus of the Wheel | a 3/3 Human Wizard |
| Whirlpool Warrior | `draw: {unmodelled: 'Whirlpool Warrior'}` |

Jace's Archivist is the one that mattered: `{U}, {T}` wheels **every turn,
forever, for one blue mana**, and the pilot's instruction was to model it as
exactly that.

### The sweep, and the split that is the whole design

34,814 cards. **43** say a player empties their hand; **8** do it from an
activated ability; **7** are modelled and **1** is refused:

```
REPEATABLE   Jace's Archivist          {U}, {T}
             Queen Kayla bin-Kroog     {4}, {T}
ONE-SHOT     Magus of the Wheel        {1}{R}, {T}, Sacrifice this creature
             Whirlpool Warrior         {R}, Sacrifice this creature
             Vindictive Flamestoker    {6}{R}, Sacrifice this creature
             Jack of Hearts            Power-up — {4}{R}{R}  ("only once")
             Immortus                  Power-up — {5}{U}{U}  ("only once")
REFUSED      Runehorn Hellkite         {5}{R}, Exile this card from your GRAVEYARD
```

Runehorn is refused because the zone is one this model does not have. Firing it
off the battlefield would be a seven-card refill the card cannot give, and it is
**named** in `meta.draw_not_modelled` rather than returned as an all-zero
profile — that being the one state that means "nothing to see here". Before this
commit it was in exactly that state, silently.

The two Jar effects (Magus of the Jar, Memory Jar) **exile** a hand rather than
discard it, so `_WHEEL_RE` never matches them and no discard payoff is credited.
That falls out of the pattern; it is not special-cased.

### The anchor was the bug the sweep would have baked in

The first draft anchored the cost segment to a line break or a sentence end.
Scryfall separates abilities with newlines, **`cards.json` keeps them and
`cards.csv` does not** — so Runehorn read one way inside a deck and the other
way inside a corpus sweep, which is precisely the disagreement a sweep exists to
prevent. The anchor is gone; the token set (mana symbols, `Sacrifice this …`,
`Exile this card from your graveyard`, commas) does the work, and a word is not
in it, so a sentence that merely contains a colon cannot be read as a cost.

### Three positions doing three jobs

The fire site sits **before the draw spells** (what a wheel finds is castable
this turn), **after `pool` is set** (unlike an upkeep trigger this one costs
mana and competes for it) and **before every casting loop** — which gives a
`{T}` ability its summoning sickness for free, since a permanent joins
`wheel_engines` inside this turn's loops and is first seen next turn. No tapped
state is tracked and none is needed.

A cost that says "Sacrifice this creature" fires once and **takes the body off
the board**. Leaving a 3/3 standing after it has been sacrificed is the same
over-credit the sacrifice channel already learned to avoid, and it is invisible
in the draw figures — which is why the test asserts it on board power instead.

### Measured

sharknado, 4,000 games, seed 3, `model_draw` + `model_combat` + `model_discard`.
Mean cumulative EXTRA cards drawn, and mean cards discarded:

| turn | drawn before | drawn after | discarded before | discarded after |
|---|---|---|---|---|
| 4 | 1.190 | 1.194 | 0.625 | 0.627 |
| 6 | 4.187 | 4.826 | 1.992 | 2.432 |
| 8 | 7.506 | 9.940 | 3.409 | 5.043 |
| 10 | 10.317 | **14.780** | 4.456 | **7.580** |

Turn four is flat because none of the three is on the table yet; the whole
delta is the back half, which is the shape a repeatable engine has.

Attributed by blinding one card at a time, at turn ten: **Archivist +2.09
drawn** (+2.11 discarded), **Magus +1.55** (+1.07, and board power 17.79 → 17.42
as the sacrificed body leaves), **Whirlpool +0.92** (+0.03 — it *shuffles*, so
no discard trigger fires, which is `wheel_shuffles` doing its job).

**The control**: blinding `activated_wheel` returns every figure to the
pre-change value **exactly**, so the delta is these three cards and nothing else,
and every deck in the fleet without an activated wheel is byte-identical.

### THE FIGURE IS A FLOOR, AND THE FLOOR IS LOW

Jace's Archivist draws *"cards equal to the greatest number of cards a player
discarded this way"*. In a real game that number is usually an **opponent's**
hand, which is the entire reason the card is good — you dump two and draw seven.
This model has no opponents holding cards, so it draws what **our** hand held,
and the hand-size gate only lets it fire when our hand is thin. The correction
is not made, because the correction would be an authored opponent hand size
driving a headline — the `engine_online_*` failure one channel over. Read it as
a lower bound and settle the question in Forge, where sharknado's seat has its
own problem: it **cast Wheel of Fortune once and Windfall never in 60 games**
while discarding them.

## sharknado is a deck Forge cannot pilot, and the AI profile does not fix it (2026-09-14)

`CLAUDE.md` already says a Forge result on a deck whose engine the AI never cast
is a floor, and names sharknado as the case. Two runs now put numbers on it and
**close the obvious escape hatch.**

**Run 1 — 120 games, our seat on `Default`.** 13 wins, 0.124 [0.074, 0.200]
against a table null of 0.257. Over 1,141 own turns the seat cast 95 wheels and
**discarded 105**, and nine of the seventeen were never cast once. The split is
not random:

| the AI casts | the AI refuses |
|---|---|
| Wheel of Misfortune 23, Khorvath's Fury 20 — damage on resolution | Wheel of Fortune 1, Windfall 0, Reforge the Soul 0 — symmetrical draw |
| Burning Inquiry 14 — one mana | Jace's Archivist 0, Magus of the Wheel 0 — activated |
| Arjun 17, Teferi's Puzzle Box 14 — permanents it wants for the body | Molten Psyche 0, Winds of Change 0, Whirlpool Warrior 0 |

**Run 2 — `--profile Experimental` on our seat, same pod, same seeds.** The
hypothesis was the sacrifice-outlet one: the repo records Indulgent Aristocrat
activating 0.41/cast under Experimental against 0.07 under Default, so a
symmetrical draw spell looked like the same shape of problem — an effect the
evaluator cannot price.

**It is not.** Abandoned at 37 games because the answer was already in:
**Jace's Archivist was drawn and discarded three times and activated zero
times**, Wheel of Misfortune was cast 10 times exactly as before, and the rate
was 0.062 [0.02, 0.20] — no better, and the games ran markedly slower.

So the refusal is a limit of the evaluator, not a setting. **Do not re-run this
hoping the profile will fix it.** No record was written for run 2; only the
120-game Default run is tracked.

### What this means for how the deck is judged

- **The win rate is out.** Not "wide", out. The engine is off for most of it.
- **`experiment` still works where the misplay is SYMMETRIC** — a land change, a
  protection package, an interaction suite. The refusal is a constant on both
  arms and cancels.
- **`experiment` does NOT work on any change that touches the wheels**, because
  the refusal is card-by-card, not uniform. `recon-v1` cuts four wheels and
  cannot be graded in Forge: it would measure which wheels the AI likes.
- **`engine_casts` is the thing Forge is still good for here.** It is the only
  measurement in the bench that looks at real rules against a real pod, and it
  is what diagnosed this.
- **The goldfish is the output metric of record for this deck** — extra cards
  drawn, cards discarded, per-opponent ping damage — read as a floor, with
  `_shuffle_note`'s caveat that a "greatest number discarded" wheel draws only
  what OUR hand held because this model has no opponents holding cards.
- **The arbiter is the table.** The deck has zero games in the captain's log. For
  a deck whose plan is a symmetrical draw spell, the missing evidence is a person
  choosing to cast it.

## An axis with no population to check it against, and the null it found (2026-09-14)

`candidates.AXES` carries a standing rule: a new axis ships with an independence
check across the fleet, because three combat axes once shipped that were one
axis at r = 0.92–0.98. `both_online_6` — the turn BOTH commanders of a partner
pair are on the battlefield — **cannot have that check.** It reads
`diagnostic.commanders`, which exists only for a deck with a partner, and
sharknado is the only one in the fleet. n = 1.

**The first draft of its comment invented five r values rather than say so.**
That is the failure this page is about, committed inside the commit adding the
guard against it. A test now greps the block for `r = <number>` and fails, and
the block states the limitation instead.

### Proved by sensitivity instead

The question an independence check answers is "does this axis see something the
others do not". With no population, ask instead whether it sees the thing it was
built for. sharknado, 4,000 games, seed 7, every white-producing land replaced
with a Mountain:

| | both @6 | mean joint | Shabraz | Brallin |
|---|---:|---:|---:|---:|
| as printed | 0.820 | 5.464 | 5.391 | 4.254 |
| no white at all | **0.553** | 6.223 | 6.203 | 4.195 |

−0.267 on the rate, far outside any MDE, **and it moves the right commander**:
Shabraz is `{3}{W}{U}` and slips 0.81 turns; Brallin is `{3}{R}` and does not
move at all. The axis reads colour access to the second commander, which is what
it claims to. `land_drop` cannot do this — it stops at turn five and has no
notion of a five-drop needing two specific colours.

**One test of this shape was designed backwards first and is worth recording.**
The obvious experiment — swap six non-white lands for Plains, matching
`mana-fit`'s "six white sources short" — made the deck **worse** (0.820 → 0.794).
The six lands cut were duals feeding U and R, which the same two commanders also
need. `mana-fit` says this outright in its own tail: *"No land is safe to cut on
colour grounds: every one feeds a colour this list is still short of."* A colour
experiment that adds one colour by removing another measures the trade, not the
colour.

### The null, which is worth more than the axis

The first sweep: nine accelerants, 3,000 games each, MDE 0.0279, with **Mind
Stone planted as a control** — it is ramp, and it makes only `{C}`, so an axis
that measures "can I cast a `{3}{W}{U}` five-drop" must rank it below the
`{W}{U}` fixers or it is measuring mana count.

Every card landed between +0.019 and +0.027. Mind Stone (+0.021) was
indistinguishable from Azorius Signet (+0.027), whose pips are Shabraz's
exactly. Nothing cleared the MDE.

**ONE TWO-MANA ROCK DOES NOT MOVE THIS DECK'S IGNITION TURN**, and the
sensitivity test says that is a true null rather than a blind axis. `mana-fit`'s
six-source shortfall is a Karsten 90%-on-curve target, not a cliff: the 21 white
sources already in the list are doing the work. The three fixers sitting in the
Zur Parts box — Azorius Signet, Talisman of Progress, Commander's Sphere — are
free to add and should not be expected to land the commanders sooner.

That is the bench doing the job it exists for: the cheapest available change was
measured and found not to matter, before anything was bought or unsleeved.

## Two branches on one deck, measured on one harness, and the cheap one won (2026-09-14)

sharknado carries two open branches. Neither was ever proposed, so neither was
ever formally on the table — `OPEN` in `branch_state` means "no proposal — this
is an experiment, not a decision", which is the honest description of both. This
records which one the evidence favours, so the comparison is not re-derived.

| | `recon-v1` | `tax-v1` |
|---|---|---|
| opened | 2026-09-10 | 2026-09-14 |
| size | +37 −37 | +3 −3 |
| **to buy** | **18** | **2** |
| sleeved in other decks | 15, across four LOCKED decks | 1 (heliod, LOCKED) |
| damage @T10 | **−5.27 (−11%)** | **+5.88 (+12%)** |
| extra cards by T8 | **−1.75 (−17%)** | **+0.85 (+8%)** |
| board power @T6 | noise | +0.49 |
| killed by T6 | +0.048 | +0.043 |
| stall, two in a row | −0.123 | −0.035 |
| interaction affordable @T6 | **+0.265** | noise |
| objective | `extra_cards_8 >= 1.2`, **cannot fail** | `damage_8 >= 31.5`, **met at 32.40** |
| verdict | A TRADE | **MERGE** |

Same harness, same seed, 10,000 games each.

**`recon-v1` is not a bad branch and this is not a reason to delete it.** It buys
a real thing — interaction affordable at turn six goes from 17 games per 100 to
44, and the double stall falls by 12 per 100 — and it made one call that today's
measurement independently confirms: of the four wheels it cuts, all four are
SHUFFLE wheels, which pay Brallin nothing. It was written before the model could
see this deck's engine at all, and its own `why` text says so in terms that are
now false.

What decides it is the price. **18 purchases plus unsleeving 15 cards out of four
working decks, for a measured loss on the deck's own axis — against two
purchases for a gain on every row that clears its MDE.**

Its objective also cannot fail (recorded above, "An axis with no population…"),
so it has no live grade and the reading has to come off the measured table. That
stands as written; a pre-registration rewritten once it is inconvenient is not
one.

**Kept, not deleted.** `deck_branch.delete`'s own docstring is the argument: a
branch holds measurements that cost real time, and removing an unmerged one
throws away the evidence for a decision nobody recorded. This page is that
record.

## A doubler that was never there, and four cards the model could not read (2026-09-14)

**THE MOST EXPENSIVE HOUR OF THE SESSION WAS THE ONE SPENT TRUSTING A NULL.**
Four cards in a row — Mind Stone, Jaws Relentless Predator, Bard King of Dale,
Teferi's Ageless Insight — measured as "no effect" in a `candidates` sweep that
had never priced any of them. Each null was reported beside a confidence
interval, which is what made it dangerous: a card the model cannot read looks
exactly like a card that does not help.

The pilot caught it twice by refusing the answer. That is the only reason it
came out.

### The phantom

Cutting Elesh Norn // The Argent Etchings from sharknado measured **−2.2 damage
at turn eight**, far past the 0.66 MDE, and would have been reported as a reason
to keep a card that does nothing. It carried `team_damage_multiplier: 2` — the
model believed it doubled all damage, permanently — off **chapter II of its Saga
BACK FACE**: "creatures you control get +1/+1 and gain double strike until end
of turn". One turn, on a face reachable only by paying {2}{W} and sacrificing
three other creatures.

`_TEAM_DOUBLE_STRIKE_RE` matched the clause anywhere in the oracle. The token
doubler eight lines away had scoped this correctly for a year —
`token_doubler` takes a window around the match and refuses it if the window
says "until end of turn" — and the damage multiplier simply never had.

**THE FIRST FIX WAS WRONG IN THE OTHER DIRECTION, and an existing test is what
said so.** Testing only for "until end of turn" also dropped Atarka, World
Render and Thrakkus the Butcher, and `test_three_wordings_one_effect` went red.
Those two are real: *"whenever a Dragon you control attacks, it gains double
strike until end of turn"* wears off every turn and is PUT BACK every turn, so
to a model that attacks every turn it is permanent. The distinction is not
temporary-versus-permanent, it is **one-shot versus re-applied** — an instant, a
sorcery or a Saga chapter fires once; a trigger the permanent carries does not.

The sweep, which is the whole reason the rule exists:

| | cards reading as a permanent damage multiplier |
|---|---|
| before | 75 |
| after the blunt fix | 37 — dropped Atarka and Thrakkus, which are real |
| after the right fix | **53 — 22 were phantom** |

The 22: four Saga chapters, seven instants and sorceries (Cleaver Riot, Savage
Beating, Double Trouble), and ETB one-shots like God-Eternal Rhonas and Terror
of Mount Velus. **sharknado's own damage@8 fell 31.45 → 29.47** when the phantom
went, which means every figure quoted off that deck before this date was two
points high.

### The four channels

None of these existed, and all four returned a confident zero instead:

| channel | what it reads | corpus |
|---|---|---|
| `activated_draw` | "{1}, {T}, Sacrifice this artifact: Draw a card" | **405 cards carried a sacrifice-gated draw and `draw_profile` read zero for 400 of them — 99%** |
| `blood` | a Blood token, which is that ability with a discard in the cost | 44, every one read by hand |
| `artifact_sac` | a payoff when an artifact leaves the battlefield | 48, in three phrasings |
| `draw_multiplier` | "if you would draw a card … draw two instead" | 8, five of them Jeskai |

**A CHANNEL PLACED WHERE IT CANNOT FIRE IS NOT A CHANNEL.** The Blood crack was
put last, on leftover mana, by analogy with the X spells — and a goldfish spends
its whole pool casting, so `spend(1)` failed almost every turn: **49 of 300
games had Blood standing on the board uncracked at end of turn**. A resource the
model can never spend is a resource it cannot price, and the reading was a
confident zero either way. Moved before the casting loops, one crack per turn,
stated in `MODEL_ASSUMPTIONS` as the authored floor it is.

### The blind-spot list was lying, in the dangerous direction

`net_change.BLIND["draw"]` read *"extra card draw is not modelled — one card per
turn, always"* and keyed off the card's ROLE, which only says that the card
draws. It named Teferi's Ageless Insight as unmeasured **in the same report that
measured it at +0.89 extra cards**. The sentence predates the entire `model_draw`
channel.

A wrong entry on that list is worse than a missing one: a missing entry hides a
limit, a wrong one tells the pilot that a measured gain was never measured and
invites cutting a card the figures already credited. It asks `draw_profile` now,
which has always known — that function sets `unmodelled` to the card's own name
when the draw is through a channel there is no event for.

### What the ceiling run is for

The pilot's objection to Jaws was mechanically correct: connect for five, make
five Blood, crack them for five discards and five draws, and both commanders
charge for every one while Jaws pings the table for each artifact sacrificed.
The model said −0.43, and the model was blind to the whole line.

So the blinkers came off as a **labelled sensitivity**, never committed: combat
Blood credited, the one-per-turn cap lifted, tokens granted on ARRIVAL rather
than on connecting — more generous than the card. Even then, *as if he connects
for five*, damage@8 moved **+0.09**. Floor −0.43, ceiling +0.09.

The answer did not change, and that is the point: **the card is bracketed rather
than blind**, and the argument is settled instead of being a standing doubt about
the instrument. The volume is what kills it — the deck makes ~1 Blood and cracks
**0.463 artifact sacrifices per game**, against an engine already dealing 30 by
turn eight.

The first attempt at that ceiling measured nothing at all and said so loudly:
all three arms returned byte-identical numbers, because the Blood count pattern
took number-words only to "four" and no digits, so "create 3", "create 5" and
"create 1 Blood token" all failed to match and fell into the same bucket. **Three
identical arms is the shape of a sensitivity that measured nothing.**

## A ROLE-GATED CHECK CANNOT SEE A CARD DOING A SECOND JOB (2026-09-21)

`deck-audit`'s `interaction-breadth` reads sharknado at **ZERO of five classes
answered** — a deck that cannot answer a single class of permanent. The engine
model enumerated the answers the check cannot see and the true figure is **four
of five**; only LAND is genuinely unanswered.

    Commit // Memory        any nonland permanent — creature, artifact, enchantment
    Irencrag Pyromancer     "any target" damage
    Niv-Mizzet, Parun       "any target" damage
    Talon Gates of Madara   phase out
    Echo of Eons / Time Reversal / Memory   a graveyard

`_interaction_breadth` is **role-gated**: it reads `card_roles` and Commit //
Memory's only role is `draw:burst`. The card removes any nonland permanent and
the check never looks at its oracle text. This is the **third recorded instance
of the same failure** — a card doing two jobs is scored on its loudest one, and
the analysis follows the count rather than the card.

And it is wrong in BOTH directions on the same deck: the interaction COUNT
reads 7, but Brallin, Glint-Horn Buccaneer and Magmakin Artillerist can never
point at a permanent, so the real suite is **five**. Under-counting breadth
while over-counting copies.

**Before calling an axis a weakness, enumerate the cards.** `deck-audit` prints
probe notes under exactly this kind of reading and they are there to be read.

## A TUTOR-ASSISTED FIGURE IS TYPE-BLIND (2026-09-21)

`goldfish_library._target_met` lets any cast tutor fill any missing `any_of`
group **without asking whether it could legally find a member**. On gishath the
assisted "THE ONE-SIDED APOCALYPSE" row reads 39.0% by crediting a wish for
Blasphemous Act, and "Protection held for the board" reads 87.5% by crediting
five instants no tutor in the deck can reach.

The unassisted figures are sound; the assisted ones are an upper bound that no
deck achieves. `gishath/tutor_guide.json` names the two rows to read unassisted.
**Prefer the unassisted figure whenever a target's members are not all tutorable
by the same card.**

## `combo_graph.partners` IS CO-MEMBERSHIP, NOT A TWO-CARD COMBO (2026-09-21)

`process_combos.build_combo_graph` adds **every card in a combo as a partner of
every other**, so a four-card line contributes six "pairs". Set-covering
`partners` to break a pilot's "no two-card infinites" rule therefore answers a
question nobody asked.

Measured on emiel-blink: `partners` reported **65 pairs and demanded 29
exclusions**, a third of the nonland cards, including three of the eight
`must_include` entries. Counting TRUE two-card combos out of
`combo_details.json` — `len(c["cards"]) == 2` and both inside the 99 — gives
**EIGHT**, and **six exclusions break all of them**.

The tell was a reported pair of `Ashnod's Altar + Command Tower`, which is
obviously not an infinite. **Use `combo_details.json` and filter on card count;
`partners` is a retrieval index, not a combo classifier.**

meren-recursion was re-checked with the correct measure and is genuinely clean
at zero, so its twelve exclusions reached the right answer by the wrong route —
six of them were unnecessary and freeing them changed the built list not at all.

## A RUN RECORD IS JUDGED AGAINST TODAY'S DECLARATION (2026-09-21)

`test_no_kept_record_has_an_uncast_engine_by_accident` compares a kept Forge
record's `engine_casts` against the CURRENT `goldfish_targets.json`. When a deck
gains a card, every record that predates it flags as "the AI never cast part of
the engine" — the AI could not cast a card the deck did not contain.

Three records flagged this way on one day. The tell is that they flag
*together*, and all predate the version that added the cards. The honest fix is
`KNOWN_UNCAST` with the reason stated; the record is honest, the declaration is
honest, and it is the JOIN between them that is stale.

## FORGE CANNOT LOAD A RECENT CARD, AND SAYS SO ONLY IN THE LOG (2026-09-21)

A 177-game run on ur-dragon reported Marang River Regent and Whirlwing
Stormbrood at **cast 0**, and that was read as a finding about the deck. The
logs said otherwise, 28 times:

    An unsupported card was requested: "Marang River Regent // Coil and Catch"
    An unsupported card was requested: "Whirlwing Stormbrood // Dynamic Soar"

Forge's card database did not have them. **The run simulated a 98-card deck**,
and the two missing cards were two of the seven the branch was proposing to buy.
A zero cast count is indistinguishable from a card the engine never held.

The tell was available before the logs: the deck contained exactly TWO
double-faced cards and they were exactly the two zero-cast ones. **Grep the logs
for `unsupported card` before reading any per-card cast figure**, and treat a
zero on a recent card as unproven rather than measured.


## A BRANCH HAS NO DECLARATION OF ITS OWN, so editing the deck's stales every branch

**2026-09-21.** Five heliod branch artifacts were stale and nothing had touched a branch.
`git log` on each `branches/*/decklist.txt` points at the commit that created it, weeks
earlier; the culprit was `a38371a3` — *"heliod: procedures rewritten for v1.4.0, and its
declaration repaired"* — which edited **`data/decks/heliod/goldfish_targets.json`** and
re-ran `goldfish heliod`.

**A branch directory holds `branch.json`, `cards.json`, `decklist.txt`,
`goldfish_metrics.json`, `net_change.json` and `sim/`. It does NOT hold
`goldfish_targets.json`** — there is one declaration per deck and every branch is measured
against it, which is the whole point: two candidate 99s are only comparable if the same
questions are asked of both. The consequence is that **a declaration edit is a fleet-wide
event for that deck**, and it looks like a no-op because no list moved.

Exactly two of eleven targets had been reworded — *"Heliod down with an asymmetric card
engine"* and *"Heliod protected, or a flash grant that does not need him"* — and the other
nine came back byte-identical in all five branches, which is what made it legible at all.

**`regen --slug <slug>` already covers this**; the branches are in its target list
(`regen.targets()` walks `branches/*` for any artifact that already exists there).
Running `goldfish <slug>` by hand after a declaration edit does not, and that is the whole
bug. **After editing `goldfish_targets.json`, run `regen --slug <slug>`, never the bare
command** — the same rule the model-change gotcha states, one scope smaller.

`tests/test_pilot_artifact_freshness.py` is what caught it, and only on `make test-fresh`:
the regenerate-and-compare cache had served all five as passing, because the cache keys on
the simulator and the branch's own inputs and the deck's declaration is neither.

## The AI will not target its own commander to switch on a copy ability (2026-09-25)

The sibling of the sacrifice rule above, found on goblin-storm, and the sharper
case because here the AI plays the deck *well* and is blind to exactly one thing.

Zada, Hedron Grinder reads "whenever you cast an instant or sorcery spell that
targets only Zada, copy that spell for each other creature you control that the
spell could target." The whole deck is that sentence. Measured over 60 games
against `standard-v3`, with our seat on `Experimental`:

```
Zada cast                    100 times
Zada's trigger fired          21 times        0.35 per game
spells cast per game       ~11.9
```

**The trigger is implemented correctly and the AI almost never sets it up.**
When it does fire the log is unambiguous and the line is devastating — one red
mana for +3/+3 on eight creatures:

```
cast Brute Force targeting [Zada, Hedron Grinder (403)]
  triggered Zada, Hedron Grinder
  Resolve: "copy the spell for each other creature you control…"
cast Brute Force targeting [Goblin Token (1708)]     (… seven copies in all)
```

THE MECHANISM IS THE SAME AS THE ALTAR'S. Forge's evaluator prices `Brute Force`
on Zada as +3/+3 on one creature — identical to `Brute Force` on any other
creature. Nothing in the target-selection heuristic can see that this target
multiplies the spell across the board, so it picks whatever it normally likes
(the biggest body, an unblocked attacker) and Zada is a 3/3 among Goblins. It is
not declining a good play; it cannot see that the play differs.

WHAT THIS DOES TO A RESULT. It is worse than a floor. A floor is the same deck
played timidly; this is **a different deck** — a Zada list played without Zada's
ability. Three runs are affected and all three are void as deck measurements:

```
champion                 0.031  (1/40)
branch zada-v1 Default   0.040  (2/60)
branch zada-v1 Exper.    0.065  (3/60)
diff branch vs champion  -0.008  ci95 [-0.091, +0.098]
```

The baseline is invalid by the same mechanism, so the A/B is not rescued by
comparing them: neither arm ran the engine. `--profile Experimental` does not
help, because willingness to activate abilities is not the problem — the
observations were nearly unchanged (`tokens_observed` 1.98 → 1.95,
`token_attackers` 1.70 → 1.67) while only the win count moved by one game.

HOW TO GET EVIDENCE ANYWAY, and it is the right answer rather than a consolation:
the 21 fires are 21 real boards. `sim-scenario --stack` lifts one and
`/resolve-stack` proves the line by CITATION, which is the ✓ tier — strictly
better than a rate. Stack 007 is that artifact.

THE PREFLIGHT. Before spending hours on a deck whose engine needs a deliberate,
unusual play, grep the logs for the commander's own trigger:

```bash
grep -rc "triggered <Commander Name>" data/decks/<slug>/sim/logs/<run>/*.log
```

A count near zero means the run is not measuring the deck. This is cheaper than
`engine_casts`, which reports CASTS and said the plan was played 134-160% of
expected natural draws here — true, and beside the point: every card was cast,
just never at Zada.

## A BRANCH AND ITS CHAMPION MUST BE MEASURED BY THE SAME SIMULATOR (2026-09-26)

goblin-storm's `zada-v1` had run for a day as "the refactor that did not convert".
It had tripled the fodder count, added per-cast damage, three body factories and
the storm rituals, and the report said: two rows better, two worse, one noise, and
an objective it missed. The pilot remembered a measurement showing +40% damage and
could not find it in the report.

It was in `393228a7`'s own commit message — *"MEASURED, champion vs branch zada-v1,
10,000 games, seed 11, **all channels live**: damage @10 15.512 -> 21.735"*. That
run forced the channels on for both arms. **Only the branch's declaration was
committed.** `data/decks/goblin-storm/goldfish_targets.json` still declared nothing
at all, untouched since the goldfish shipped in July, so the deck built on *"whenever
you cast an instant or sorcery that targets only Zada, copy it for each other
creature you control"* was measured with Zada's ability, combat and card draw all
switched off.

**What that does to a report is not a smaller number, it is a different table.**
`_cell` returns None when a block is absent and the row loop `continue`s, so:

    output block, champion arm   available: false — "this deck opts into neither
                                 model_treasures nor model_combat, so there is no
                                 hoard and no clock to read"
    rows dropped                 7 of 12 — damage @T10, killed by T6, killed by
                                 T10, board power @T6, hoard @T6, hoard @T10,
                                 extra cards by T8
    rows surviving               5, all opening-hand consistency

And the two surviving rows that MOVED were the two the asymmetry flattered:

    missed drop by T5            -0.083 "better"  ->  -0.011 noise (MDE 0.019)
    interaction affordable @T6   -0.037 "worse"   ->  +0.095 better

The first is a land-finding row and only the branch was allowed to draw extra
cards. The second charged the branch for mana spent on cantrips the champion was
not permitted to cast. **One of the two reported gains was an artifact and one of
the two reported costs was backwards, in the same direction: the asymmetry paid
the branch on consistency and billed it on tempo.**

Declared identically on both arms — every channel the deck's cards feed, so the
choice is not a knob — the deck's DARK count went 7 -> 0 and the branch won ten of
twelve rows and lost none: damage @T10 **16.013 -> 27.726 (+73%)**, hoard @T10
+222%, killed by T6 1 -> 5 per 100, board power @T6 +14%.

**THE RULE, and it is narrower than "one declaration per deck".** `common.deck_file`
prefers a branch's copy of an authored file over the deck's, and its docstring
asserted that nobody writes a second `goldfish_targets.json`. Two branches do.
One of them is legitimate: meren-recursion/drain-density-v1 adds Bastion of
Remembrance and Cauldron of Essence to four `any_of` groups, because **a target
names CARDS and the branch has those cards** — that asks the same question of a
different list, which is the entire point of a branch. What a branch may never
change is a `model_*` flag. That is the instrument, not the question, and two lists
measured under two instruments are not an A/B whatever their intervals say.
`test_a_branch_never_declares_a_model_channel_its_deck_does_not` is the gate, and
it reproduces this exact state when the channels are stripped back off the deck.

The preflight is one command, and it was available the whole time:
`model-coverage <slug>` prints the DARK count per channel. It is champion-only,
which is why a branch-side asymmetry survived it.

## A FORGE RECORD DESCRIBES THE LIST IT PLAYED, NOT THE LIST ON DISK (2026-09-26)

Found in the same session, on the same branch. `net_change.forge` reported 120
games as zada-v1's rate. Both records carried
`seats[].decklist_sha256 = 57725742` — the branch's FOURTH commit — against
`e01b366c` on disk, its seventh. Eight cards had come in and nine gone out
since, including Hanweir Garrison, the largest single gain the branch claimed
(+9.91 damage @10). The stamp is written into every record beside the seat and
**nothing had ever read it.**

Swept across the fleet: **29 of 41 branches, 10,120 games.** Mostly champion
arms, because a merged branch rewrites the champion's `decklist.txt` and every
run made before the merge then describes the pre-merge deck.

**IT FLAGS, IT DOES NOT SUPPRESS, and the sweep is what decided that.**
`decklist_sha256` is over the file's bytes. edgar-vampires' list changed
`Gifted Aetherborn (AER) 61` to `Gifted Aetherborn` in the same commit as two
real swaps — so a cosmetic edit trips the gate identically to a card swap, and
the only remedy for a false positive is a multi-hour Forge batch. A gate that
silences 10,120 measured games to prevent a misreading destroys more than it
saves. So the rate stays, `forge.list_mismatch` names the played sha, the game
count and the sha on disk, and both the CLI and `branch-view.js` print it ABOVE
the number — underneath, it reads as a footnote to a figure the eye has already
taken. Runs are still preferred when a current-list run exists; the fallback is
only for an arm that has nothing else.

**`engine_casts` IS strict, because it makes a claim about particular cards.**
It named Hanweir Garrison, Legion Warboss, Assault Strobe, Reckless Ransacking
and Great Train Heist as *"held and never cast"* — the log's own statement that
the AI drew a card and passed it over, and the strongest reading this report
offers. **None of the five was in the list those games were played with.** All
five were added afterwards. "Held and never cast" and "not in the deck" are
different facts and it could not tell them apart, because it read the newest
record against the current list. It now reads only current-list records, and
names the run and sha it read so the claim can be checked against the games.

This is the sibling of *"A RUN RECORD IS JUDGED AGAINST TODAY'S DECLARATION
(2026-09-21)"* above, which found the same join stale from the declaration side.
Same defect, second surface: **the record, the declaration and the list are three
things, and every figure has to name which ones it joined.**

## The pilot loop found six reconstruction defects in one session (2026-09-28)

The pilot asked for an LLM to fly the deck, because Forge's AI cannot fly Zada and
patching goldfish channels one card at a time does not scale. `docs/simulation.md`
had already drawn that line — *"play a seat — 500 games is not an agent's job"* —
and offered the alternative in the same table: **turn a surfaced board into a v2
scenario and hand it to the resolve loop.** That loop existed and was unused,
because nothing pointed it at a board worth the spend.

Two boards were lifted from one 100-game run and handed to
`stack-resolver` ⇄ `rules-checker`. **The scenarios were not the return on the
exercise. The bugs the checkers found in `bridge.py` were**, and every one of them
had been silently corrupting every lifted board since the bridge shipped.

### 1. A creature with a characteristic-defining power never reached the board

`_CREATURE` demanded `\d+`, and Forge prints `Lord of Extinction - Creature * / *`
verbatim. The pattern did not match, `text == name` did not match, and the cast
**stayed in `pending_casts` forever** — so the creature was absent and the lift
described a battlefield it was not on.

Stack 008 said seat-2 held "Ripples of Undeath and three tapped lands". The log
had **Splinterfright** there too. The kill that artifact proved survived only
because Splinterfright happened to be tapped, an argument the artifact could not
make because it could not see the creature. ONE untapped blocker turns that kill
into a survival, so **this gap points one way: it flatters every lethal claim.**

Sweep, one run: 62 `* / *`, 17 `* / *+N`, 28 `* / * (X=N)` — ~107 resolutions over
six creatures (Boneyard Wurm, Lord of Extinction, Mortivore, Old Stickfingers,
Souls of the Lost, Splinterfright), **all of them opponents'**. The P/T is now kept
as the literal `*/*`: the log carries no value, and "a real creature whose size I
cannot give you" is what a resolver must reason about. A fabricated number would be
worse than the absence it replaced.

### 2. Every Aura was discarded as an instant

An Aura enters attached and resolves as `Rancor (203) -  Attach to Sythis (12)`,
which matches `_SPELL` — so the branch meaning *"an instant or sorcery resolved"*
threw it away. **Every Aura was missing from every lift.**

Found on stack 009: seat-4's Sphere of Safety taxes attackers `{X}` where X counts
its controller's enchantments, and an Aura the lift had dropped made the real tax
`{3}` against the `{2}` the artifact reasoned from. Sweep, same run: Rancor 38,
Whip Silk 26, All That Glitters 25, Overgrowth 24, Ancestral Mask 23, Strength of
the Harvest 17. Equipment is deliberately unaffected — it enters as a bare name
when cast, so it is no longer pending when it later attaches.

### 3. A token existed only from the moment it ACTED

`_bind` registered a token the first time it attacked, blocked or dealt damage, so
a token that was made and left standing was absent — and the artifact said so in an
annotation, *"tokens that only sat are not listed"*, as though that were a footnote
rather than the whole board.

**MEASURED over 804 of our precombat mains: the lift listed 187 tokens where 1843
were alive. It saw 10%.** On a Goblin token deck that is not a gap, it is the deck.
A conclusion was published from it and was wrong by ~18x:

    other bodies beside the commander     as lifted     with tokens
    >= 3 others                             17.3%          67.1%
    >= 6 others                              2.0%          36.5%
    >= 8 others                              0.0%          24.9%

"The board your line needs happens 2% of the time" became "about a fifth of
commander-out turns". Tokens now register at CREATION, unbound until one acts, and
binding an id **consumes** the placeholder rather than adding a second copy.

### 4. The number words stopped at "seven"

`_WORDS` had no `eight`, and Krenko, Mob Boss prints *"creates eight 1/1 red Goblin
creature tokens"* — a doubler takes it to sixteen or thirty. **The largest boards
were therefore the ones most wrongly reported.** Sweep: a 1114, two 120, X 39,
three 17, eight 5, six 3, four 3, eighteen 2, twelve 1, thirty 1, sixteen 1,
fourteen 1. `X` stays absent on purpose and is reported as unreadable, never
guessed — and the count group is `\w+` rather than a list of the words it knows,
because listing them made `create X …` match NOTHING and vanish in silence.

### 5. A token death was charged to a seat on a guess

`owner` is learned only from lines that name a controller outright, so a token that
never acted never entered it: **705 of 857 token deaths (82%) had no owner.** They
were being applied to whichever seat happened to hold a matching token — which can
delete AN OPPONENT'S BLOCKER, the direction that flatters a kill.

Three layers now, most reliable first, following the discipline `parse.py` already
uses for its name fallback — consulted only where `owner` is silent, and a tie left
unattributed rather than guessed:

1. `owner` said so. A fact.
2. Exactly one seat currently HOLDS an unlisted one of that name.
3. Exactly one seat ever MAKES that token name.
4. Otherwise removed from NOBODY, with a note that the board may overstate a token.

The layers were first ordered 3-before-2 and that was needlessly conservative: seat
A makes two that both die, seat B makes one and holds it — both "make" them, so the
maker layer calls a tie the holder layer can settle. A genuine tie now survives in
24% of games, and it errs toward overstating a token rather than deleting a blocker.

### 6. The graveyard is battlefield deaths only — NOT fixed, documented

`parse.py` emits zone events only for `Battlefield -> Graveyard` and
`Battlefield -> Exile`, so **a card milled or discarded into the graveyard never
becomes an event at all.** On jarad-graveyard, a mill deck, the real graveyard is
large and the lift's is nearly empty — which taken literally makes Splinterfright
`0/0` and already dead.

Found by the resolver on stack 010, which flagged it and declined to resolve it.
Fixing it properly means teaching `parse.py` to emit mill and discard events, which
feeds the whole analysis layer. Until then the bridge STATES the limit:
`graveyard_is_a_floor: true` with a reason, plus a board-level note whenever a `*/*`
creature is present saying its size is unknown and cannot be computed from the
artifact. On 010 the answer does not depend on it, which is why the proof holds.

### The fleet audit, and the one artifact that matters

Six lifted scenarios exist. goblin-storm/008 and 009 are superseded by 010 and 011,
which were lifted after the fixes. goblin-storm/007 predates them and is a `fail`.

**radagast/008 predates them and is a `pass` — the only lifted scenario in the repo
with a passing checker verdict. IT IS ALSO ON A DECK THAT IS BROKEN DOWN FOR PARTS
(since 2026-08-21), so it is out of scope: an archived deck's artifacts stay as
published and are not re-verified. Scoped to LIVE decks the exposure is zero.** Re-lifting its board from
the same run (edgar-vampires-vs-yawgmoth-swarm-vs-heliod-n8, game 1, turn 33; life
totals 12/43/16/13 match exactly) gives OUR seat **11 creatures against the 7 it was
resolved with**, and seat-2 **5 against 2**. Seven creatures missing from a board
carrying a ✓. It has not been re-resolved and will not be — the deck is apart. The figures are
kept only to size what the defect would have cost on a live deck.

### What this says about the instrument, not the deck

Every one of the six was found by pointing an adversarial rules-checker at a real
board and making it argue. None would have been found by a test, because each was a
pattern that silently matched nothing — and a filter that matches nothing reads
exactly like a fact about the deck. That is the same shape as the goldfish's
"a card the model cannot read looks exactly like a card that does not help", one
layer down, in the reconstruction rather than the model.

## Two goldfish channels, and the sweep that found the cards (2026-09-27)

The pilot asked why Molten Duplication and Heat Shimmer were not in a branch built
around "lean into Zada". They were in no sweep's shortlist because they were in no
CHANNEL: `model-coverage` read them as invisible and `candidates` ranked them at
exactly zero, which is indistinguishable from a card that does not help. **Seven
passes of candidate sweeps built a deck around a commander whose best card the
instrument could not see.**

### `spell_token_copy` — a spell that token-copies one target creature

38 instants and sorceries in the corpus; 4 castable on a mono-red identity — Molten
Duplication `{1}{R}`, Electroduplicate `{2}{R}` (flashback `{2}{R}{R}`), Heat
Shimmer `{2}{R}`, Kindle the Inner Flame `{3}{R}`. Excluded: activated abilities
(Kiki-Jiki, Mirrorpool, The Fire Crystal — not cast spells, so no commander-copy
trigger sees them), graveyard copies (Feldon), and mass copies (Kindred Charge — not
a single target, so a Zada-style ability never copies it).

Under Zada each copy targets a different creature, so a board of N gets N token
copies of ITSELF, all with haste. The effect goes through `creature_entered`, the ONE
DOOR, so Impact Tremors and every arrival payoff fire per token without that site
knowing they exist. The copies are TEMPORARY and removed after combat, or a board
doubling would become permanent on every cast.

**`copy_fodder` had to be widened too, and that is the half that was invisible.** It
required the literal `target creature`, and Molten Duplication reads "target artifact
or creature you control" — so the single best card for a Zada deck in mono-red was
structurally excluded from being copied at all. Sweep: 855 -> 872 matches, nothing
dropped; all 17 gained read card by card, and the three "gain control" hits (Hijack,
Sibling Rivalry, Systems Override) are kept deliberately because what the model
credits is the *"untap it, it gains haste"* clause after the no-op control change.

### `death_damage` — a death trigger whose payoff is damage

`death_drain` read only "each opponent loses N life" / "target player loses N life",
the Blood Artist idiom. A death trigger that DEALS DAMAGE had no field, so on a
Goblin deck whose whole conversion is bodies dying, **Pashalik Mons and Boggart
Shenanigans contributed nothing to any figure.**

THREE IDIOM GAPS, none found by guessing — all 45 damage clauses following a death
trigger were enumerated first, because three attempts at the pattern each found too
little. `deals` alone misses *"have this enchantment DEAL"*. "each opponent|any
target" alone misses *"target player or planeswalker"*, which is five cards. And the
TRIGGER missed two forms: a card that names ITSELF where others say "this creature"
(Pashalik Mons), and *"is put into a graveyard from the battlefield"* — which is what
**CR 700.4 says "dies" means**, so a trigger's AGE decided whether the model could
read it.

Sweep: 81 -> 87 death engines, nothing lost. The first pass read 91 and **four of
those ten gains were wrong** — Wicked Visitor, Urza's Miter, Ashiok's Reaper and
Femeref Enchantress trigger on an ENCHANTMENT or ARTIFACT reaching the graveyard, not
a creature. Reading the longhand admits every permanent type, so a
`_subject_is_a_creature` guard sits beside the pattern rather than inside it: the
subject slot holds anything from "nontoken creature" to a bare "Goblin", and a
lookahead tight enough to reject "enchantment" rejected those too.

`death_damage` stays a SEPARATE FIELD from `death_drain`. Identical to a goldfish with
one opponent at 40 life; not identical at a table, where life loss ignores damage
prevention and hits through a Platinum Angel. Folded together they could not be
unfolded later without re-reading every card.

### The rate is measured, and the measurement is the whole design

`model_deaths` requires a `source`, so goblin-storm's was read off the Forge run that
had just finished: **own 0.3529 and opponent 0.7894 creature deaths per own turn**,
from 253 and 566 deaths over 717 own turns across 75 games. The flag refuses to be
set without that string, which is why it can be trusted at all — an authored death
rate driving a damage figure is the deleted engine lift.

### What the channels then measured, and the three errors on the way

Measured on zada-v1 + the four cards, 10,000 to 40,000 games per arm:

    one card at a time, damage@8      +0.78  ->  -0.18 as N grew   NOISE
    one card at a time, damage@10     +2.57 "REAL" -> +0.77        NOISE
    all four, cuts chosen by the harness   +1.54 damage, kill@T10 -0.041
    all four, paid with NON-BODY cards     +4.64 damage, kill flat  REAL

Three errors, each mine, each instructive:

1. **Singleton instead of redundancy.** One copy in a 99 fires in 16% of games, so
   the MEAN buries a burst. Four copies is a different deck. An effect whose value is
   conditional on assembly must be measured as a package.
2. **The wrong turn.** `candidates` offers `damage_8` and the line pays at ten,
   because it needs a wide board to copy and at turn eight there is not one. Aiming
   the branch at `damage_8` would have failed it for paying off late. `damage_10` is
   now an OBJECTIVE axis — and only an objective axis: the hygiene gate correctly
   refused it in `AXES`, where three combat magnitudes already correlate at
   r = 0.92-0.98.
3. **The harness chose the cut and it sold the engine.** The default cut is the most
   expensive non-declared card, which here meant **Krenko, Mob Boss** and
   **Siege-Gang Commander** — the deck's two best body factories. Buying "copy your
   whole board" by selling the board measured +1.5 damage for 4 fewer kills per 100.
   That is the documented Edgar failure in miniature, and the fix was four non-body
   cuts.

And the model change moved the baseline it was measured against: with `model_deaths`
live the champion's own damage@10 fell **27.7 -> 21.0**, because the deck now loses
creatures. The one reading that had cleared its MDE did not survive making the model
more accurate — it was an artifact of a simulation in which nothing ever died.

### Forge cast the cards 84 times and could not use them

On copy-burst-v1, 100 games at standard-v3: the AI cast Molten Duplication 17 times,
Heat Shimmer 25, Electroduplicate 27, Kindle the Inner Flame 15 — **84 casts — while
Zada triggered 39 times in total.** Most of those casts did not target her, so each
made ONE token instead of N. The run reads 1 win in 73 decided against the champion's
7 in 94, and it trips both gates (`OUR SEAT WAS HANDLED WORSE THAN THE POD`, and
`THE AI NEVER CAST PART OF THE ENGINE: Haze of Rage`).

**That is not evidence the cards are bad. It is the documented Zada ceiling, now
quantified per card.** A deck whose payoff requires targeting your own commander
cannot be measured by an AI that will not do it, and 84 wasted casts is the number
that says so.

## The policy layer, and the first rule it killed (2026-09-28)

The pilot's instruction was blunt: Forge cannot fly the deck and patching goldfish
channels one card at a time does not scale, so **make our own AI**.

The useful observation is that the goldfish ALREADY has one. Its piloting decisions
were written in Python, one per channel, by whoever added that channel:

    "Cast LAST in the main phase and only with attackers already out"
    "Most expensive pump first"
    "AN UNTAPPER IS HELD, NOT CAST ON CURVE"

Every line is from a pilot's manual — hardcoded, generic, and per-channel. Meanwhile
the Pilot's Operating Handbook carries the pilot's OWN version in prose under
*normal procedures*, and the two had never been connected: a policy the pilot wrote
could not move a figure, and a policy the model followed could not be read.

`pilot_policy.py` + a per-deck `pilot_policy.json` is the join. Properties, each
deliberate:

- **ABSENT MEANS ABSENT.** No policy file, identical figures — VERIFIED across four
  decks, every figure byte-identical with only the `model_version` stamp moving,
  which is correct for a source change. Same opt-in contract as every `model_*` flag.
- **A branch inherits the deck's policy**, through `deck_file`, exactly as it
  inherits `goldfish_targets.json`. A branch with its own policy is the "two models,
  not two lists" defect that cost a day on 2026-09-27.
- **`why` is MANDATORY**, because a piloting rule is a CLAIM about how the deck is
  flown and one with no reason cannot be argued with.
- **One verb, one channel, one counter.** The vocabulary grows one PROVEN verb at a
  time so it cannot outrun the evidence.

### WHY NOT AN AGENT PLAYING THE GAMES — checked, not assumed

Forge's sim mode has **no external-seat hook**: `-a` sets AI profiles and stops
there (`docs/AI.md`, and the flag list carries nothing else). Nothing on this bench
can legally advance a game state either — the goldfish is stochastic and
`validate_stack` adjudicates one frozen board. An LLM deciding 10,000 games is
neither seeded nor affordable, so its figures leave the ◆ tier entirely. A DECLARED
policy keeps seeded reproducibility AND makes every rule a measurable A/B; agents
author and attack rules rather than execute them.

(Forge's card scripts DO expose `AILogic$`, `AIPreference$` and `AITgts$` — 367, 34
and 16 uses in the first 4,000 files — and `AITgts$` narrows what the AI may target
while leaving legal targets alone (`apprentice_necromancer.txt`:
`ValidTgts$ Creature.YouOwn | AITgts$ Card.cmcGE5`). An override on our own copies of
the pump and copy spells would make Forge target Zada. Declared, that is a real
lever; undeclared it silently stops measuring Forge. An earlier grep here returned
ZERO AI hooks and was reported as fact before being rechecked — the tokens are
there.)

### THE FIRST RULE WAS PROVEN AND STILL WRONG

Stack 011 established, with CR citations and an adversarial pass, that Zada copies
NOTHING when she is the only creature (707.10d) and that **no single card fixes it**,
because the trigger resolves before the spell that caused it (603.3 scoped by
603.3b) — the minimum is two cards, a body then the spell. So a token-copy spell
into a thin board is two mana for one token, and the obvious policy is to hold it.

MEASURED on copy-burst-v1, 20,000 games per arm, same seed:

    policy          kill@T6    delta            kill@T10   delta
    no policy        0.0384        —              0.7761       —
    hold until 1     0.0377   -0.0007 noise       0.7680  -0.0081 noise
    hold until 3     0.0340   -0.0044 noise       0.7507  -0.0254 REAL
    hold until 5     0.0286   -0.0098 REAL        0.7434  -0.0327 REAL
    hold until 7     0.0266   -0.0118 REAL        0.7417  -0.0344 REAL
    kill@T6 MDE 0.0054 · kill@T10 MDE 0.0117

**Monotonically harmful.** Holding the card costs more tempo than the wasted copy
costs mana. The rule was withdrawn the same hour it was written, and goblin-storm
carries no policy file.

**A TRUE RULES FACT DOES NOT IMPLY A GOOD POLICY.** That is the whole reason this
layer stores a rule as DATA with a `why` and measures it, instead of a channel author
encoding a plausible heuristic in Python where nobody ever prices it.

### AND THE SWEEP THAT JUDGED IT WAS ITSELF MISREAD FIRST

The sweep printed one MDE — the DAMAGE mean's, **3.1782**, wide because the damage
distribution is — and used it to label all five arms `noise`. The kill figures are
RATES with MDEs of 0.0054 and 0.0117, and they were printed BARE. Reading them
against the damage yardstick said "the policy does nothing"; reading each rate
against its own said "it costs 2.5 kills per 100 and the interval excludes zero".
Every rate carries its interval, and a comparison carries the interval on the
DIFFERENCE — broken here in the very table used to judge new work.


## `AI:RemoveDeck:All` is never-cast, and the repo's own counter-example was a counterspell (2026-09-30)

**The claim that was wrong.** `sim/forge_cards.py` said for a month that Forge's
`AI:RemoveDeck` marker "governs deck GENERATION, not whether a supplied deck may
contain it — Swan Song carries `RemoveDeck:All` and was cast 28 times in 160 games".
`assess` printed it to the pilot on every flagged card. The measurement was real and the
inference was not: Swan Song is a counterspell, and the AI casts counterspells through a
reactive path that never asks the question.

**How it was found.** The pilot's 2026-09-29 log: Vish Kal, Blood Arbiter "is a win
condition in itself… if it's never cast, that's a red flag." Three tracked runs, 220
games, 0 casts, 6 discards. The first exact read under the telemetry patch (20 games at
standard-v3 on the current list): in hand in six games, seven-plus lands on the
battlefield while held in three of them, cast in none — while Crossway Troublemakers and
Liliana were cast when castable and The Haunt of Hightower was not.

**The isolation.** A constructed shell — 28 basics, 12 rocks, cheap bodies — with four
copies of the card under test against a slow wall deck, twelve games, seed 777, zero AI
timeouts, telemetry jar so every draw is on the log:

| variant | drawn (games) | castable and uncast | cast |
|---|---|---|---|
| Vish Kal, shipped script | 11 | 7 | **0** |
| control: Serra Avatar in the same slot | 8 | 1 | 4 |
| Vish Kal, no activated abilities | — | 1 | 10 |
| Vish Kal, sacrifice ability only | — | 1 | 10 |
| Vish Kal, pump ability only | — | 1 | 10 |
| Vish Kal, full script minus `AI:RemoveDeck:All` | — | 1 | **10** |

The abilities are innocent; the flag is the whole effect. Loose files under
`res/cardsfolder/<letter>/` load over the zip (which is how the variants ran, and a thing
to know: a loose file changes the engine while `forge_pilot.installed()` reads only the
zip, so the bisect deleted its file before it ended).

**The mechanism, in the bytecode.** `forge/ai/AiController.getSpellAbilityToPlay`'s
candidate filter (`lambda$getSpellAbilityToPlay$8`) drops every ability that is not a
land play whose host card `ComputerUtilCard.isCardRemAIDeck` — which reads
`CardRules.getAiHints().getRemAIDecks()`, the parsed `AI:RemoveDeck:All`. `Random` sets
a different hint and is not read here. Nothing in `PermanentAi`, `PermanentCreatureAi` or
`ComputerUtilCost` names a sacrifice cost.

**The fleet sweep, before the fix.** Every `RemoveDeck:All` card in a live deck, joined
to `engine_casts` over every tracked record:

| deck | `All` permanents (games on record) | cast |
|---|---|---|
| edgar-vampires (860) | Altar of Dementia, Bloodflow Connoisseur, Viscera Seer, Vish Kal | 0, 0, 0, 0 |
| goblin-storm (435) | Goblin Bombardment | 0 |
| heliod (300) | Azorius Signet, Isochron Scepter, Psychosis Crawler | 0, 0, 0 |
| sharknado (120) | Improbable Alliance, Jace's Archivist, Magus of the Wheel, Psychosis Crawler | 0, 0, 0, 0 |

14 of 14. The `All` sorceries the same: Vampiric Tutor 0/860, Windfall 0/120, Tolarian
Winds 0/120, Hunter's Insight 0/100, Long-Term Plans 0/300, Skyscribing 0/300; Faithless
Looting 3/435 and Past in Flames 2/435 (both castable from the graveyard by another path).
Every `Random` card — Captivating Vampire 112, Indulgent Aristocrat 135, Sanguine Bond 63,
Mystic Remora 96, Sneak Attack 50 — cast normally. **So goblin-storm's sacrifice outlet,
sharknado's wheels, heliod's Scepter and Edgar's whole sacrifice suite were uncastable in
every Forge figure this repo has published**, and the "held and never cast" finding on
sharknado (below) was this flag, not the AI's judgement.

**The fix, and what it costs.** `data/forge_overrides/unflag.txt` lists 23 stems — every
`All` non-land card in a live deck on 2026-09-30 — and `forge_pilot.generate_unflag`
writes each one's override as the shipped script minus that line (layered on an AITgts
override where both apply). `forge-install --generate` regenerates the set and prints any
flagged card a live deck has acquired since; a fleet test refuses one. The engine
fingerprint moved from `60636e9e` (eleven AITgts scripts) to **`8bf04cfcdb21`**, so every
run from now on is a different instrument from every record on disk, and `net_change.forge`
will not pool them — correctly. The cost is the null: `pods.calibration` counts PLAIN-harness
runs only, so until the standard table is re-measured under `8bf04cfcdb21` no fresh run
feeds a null, and a rate read against 0.233 is read against an instrument that no longer
exists. Re-measuring it is the calibration campaign, and `pods.calibration` needs to learn
to bucket by harness rather than exclude — `docs/known-issues.md` §17.

**What an unflagged card is.** Not a card the AI plays well: `RemoveDeck:All` was put
there because the AI has no logic for it, and an unflagged Altar of Dementia may mill its
controller or a Windfall may wheel at the wrong time. It is a card the AI will CAST, whose
activations `engine_casts` can count, which makes it a floor with a number instead of a
card the instrument cannot see at all.


## The sacrifice suite, activated: four passes of one seed (2026-09-30)

The pilot: "Vish Kal … if it's never cast, that's a red flag." Then: "get the sac ability
activating." Then, of Altar of Dementia: "0 is a possibility, but it is a flag; we need
to confirm." Twenty games, seed 1664213641, standard-v3, four passes of the same deal —
each pass one lever more, each lever a fingerprint the record carries:

| card | pre-fix | unflagged (`ov8bf04cfc`) | + hints, sac profile (`ov260c7a72 aif7c3b6a8`) | + `SacOutlet` patch (`tlbeb0c66d`) |
|---|---|---|---|---|
| Vish Kal, Blood Arbiter | cast 0 | cast 3 · act 0 / 9 board turns | cast 1 · act 2 / 5 | cast 3 · act 30 / 8 |
| Viscera Seer | cast 0 | cast 2 · act 0 / 5 | cast 4 · act 4 / 5 | cast 3 · act 2 / 4 |
| Altar of Dementia | cast 0 | cast 3 · act 0 / 17 | cast 6 · act 0 / 23 | cast 5 · **act 5 / 19** |
| Bloodflow Connoisseur | cast 0 | cast 3 · act 5 / 3 | cast 3 · act 14 / 14 | cast 5 · act 35 / 12 |
| Ashnod's Altar | cast 3 · act 0 | cast 1 · act 0 | cast 4 · act 0 | cast 4 · act 0 |
| win rate (decided) | 5/17 | 4/16 | 2/17 | 8/17 |
| noncombat damage / life gained per game | 3.65 / 25.25 | 5.75 / 22.65 | 5.2 / 17.85 | 4.55 / 43.55 |

Two more passes of the same seed, once the two instants were patched (`tl0acbd08b`, then
`tl56cd8561` with Protection scanning the whole stack):

| card | fifth pass | sixth pass |
|---|---|---|
| Altar of Dementia | cast 6 · act 11 / 44 board turns | cast 4 · act 34 / 40 |
| Vish Kal | cast 3 · act 29 / 11 | cast 0 · act 6 / 5 |
| Viscera Seer | cast 1 · act 1 / 2 | cast 6 · act 10 / 19 |
| Bloodflow Connoisseur | cast 1 · act 0 / 3 | cast 4 · act 23 / 12 |
| Deflecting Swat | cast 3 — a Grasp of Fate trigger, a destroy-target-permanent spell, a Warstorm Surge trigger at Vampire Cutthroat | cast 1 |
| Teferi's Protection | cast 0 — sat through a Blasphemous Act (see below) | cast 2 — a four-attacker Spirit Cleric swing, and Rakdos + three at 5 life |
| win rate (decided) | 5/14 | 4/16 |

The fifth pass's Teferi's zero was a bug: with Blasphemous Act on the stack, three creatures
of ours out and the mana open, the AI answered with the Altar first — correctly — which put
OUR ability on top of the stack, and a logic that looked only at the top saw no opponent's
spell. It walks every stack instance now, and the sixth pass cast it twice, both times on
the swing that would have ended the game.

Read the columns as levers, not as a trend: twenty games cannot separate 2/17 from 8/17
(the MDE at this table is 42 points), and the last column carries every lever at once.
What IS settled is the mechanism, per card, measured from the telemetry hand and board:

- **`AI:RemoveDeck:All` is never-cast** (the RemoveDeck entry above): four cards the
  deck was built around had never been cast in 860 games.
- **Cast is not activated.** Unflagged, Vish Kal sat 9 own turns on the battlefield and
  Altar 17, with zero activations: the shipped scripts give the AI no logic and no
  preference for a sacrifice cost, and `Default.ai` has `SACRIFICE_DEFAULT_PREF_ENABLE`
  off. `forge_hints.json` (the aristocrat logic and a `SacCost` preference, Forge's own
  shape from `carrion_feeder.txt`) plus the profile knob moved Vish Kal to 2, the Seer
  to 4 and Bloodflow to 14 — and Altar to 0 over 23 turns, because `MillAi` has no
  aristocrat path and its targeting gate computes X from the sacrificed creature's power
  before any creature is chosen, reads 0, and refuses every target.
- **Altar's 0 was not effective play**: 32 of our own permanents went to the graveyard
  while it sat on the battlefield across six games — each a free sacrifice declined, each
  at minimum a Blood Artist trigger where Artist was out. `AILogic$ SacOutlet`
  (`data/forge_patches/MillAi.java`, the engine's own `shouldSacrificeThreatenedCard`,
  thinnest library as the target, no preference so the threatened creature is chosen
  first) took it to 5 activations over 19 turns and the declined deaths from 32 to 14.
- **Ashnod's Altar reads 0 and always will here**: a mana ability never touches the stack,
  so the log cannot see it. Unmeasurable, not idle — and `idle_on_battlefield` must not
  be read for it.
- **Vish Kal at 30 activations over 8 turns is the next question**, not an answer: the
  aristocrat logic with a token-first preference feeds it everything it can. Whether that
  wins is exactly what `data/campaigns/2026-10-edgar-sac-policy.json` pre-registers.

The instruments this took, all of them now on every telemetry run: `castable_uncast` and
`turns_on_battlefield` per card in `engine_casts.by_card`, `held_while_castable` and
`idle_on_battlefield` in the reading, and the patch set with kinds in the run id.


## Deflecting Swat and Teferi's Protection, from 0 to fired (2026-09-30)

Over Edgar's four telemetry passes: Swat held while castable on 59 own turns, cast 0 in 80
games, with **17 opposing spells or abilities aimed at our seat or our permanents** while
it sat in hand; Teferi's Protection 29 turns, cast 0. Not hints — Forge's redirect AI has
no general logic, its new-target chooser returned `null` ("AI currently can't do this"), and
its effect AI has no logic that fits a protection spell. Two `ai` patches
(`docs/simulation.md`, "Two more ai classes") and an eight-game constructed shell —
four of each against a deck of Murders, Doom Blades and three kinds of wrath, seed 4343,
zero AI timeouts:

| card | in hand (games) | cast | what it answered |
|---|---|---|---|
| Deflecting Swat | 11 | 4 | every one a `Murder` on our creature, resolved onto the caster's own Nighthawk or Serra Angel — the log shows the opponent's creature going to the graveyard under the opponent's own spell |
| Teferi's Protection, first cut | 12 | 11 | two `Day of Judgment`s, and NINE ordinary unblocked attacks — Forge's Fog threshold (`lifeInDanger`) fires on any unblocked flier |
| Teferi's Protection, `lifeThatWouldRemain <= 2` | 15 | 9 | one `Damnation` at life 4, and combats where the declared attackers were lethal or left us at two (life 10 facing Angel + two Nighthawks; life 6 facing eight) |

The first cut is the lesson: a Fog is cheap and a protection spell is not, and the
engine's own danger threshold is calibrated for the cheap one. The rule that ships spends
the card on the swing that would otherwise end the game.


## Vish Kal's -X/-X: X priced before the cost exists, and a curse that never said so (2026-09-30)

`kills_by_ability` was ≈ 0 on every Edgar record — 0.00 to 0.05 a game across six passes —
while the same card's sacrifice half activated 29–30 times a pass. Not a hint problem and
not one problem: `SVar:X:SVar$CostCountersRemoved` is set by the engine only when the cost
is paid, so `PumpAi` read X = 0 before every activation and refused it (`attack == 0`); and
the ability line has no `IsCurse$`, so even a priced X would have been aimed at OUR
creatures. The fix is one `ai` patch (`PumpAi.java`: X = the counters the cost would remove
from the source now) plus a new hint kind (`ability_params: {"IsCurse": "True"}`), and the
shell that proves it is eight two-seat games, seed 4343:

| jar | Pump activations | what died | kills_by_ability |
|---|---|---|---|
| patched + hint | 3 | Resplendent Angel (-7/-7), Emeria Shepherd (-5/-5), Archangel of Tithes (-6/-6) | 2 credited (the third resolved inside combat, where a phase line sits between the resolve and the death) |
| pristine + hint | 0 | — | 0 |

The attribution floor shows in the same table: `kills_by_ability` credits a death only
when it follows the activation's resolve directly, and one of the three kills landed
inside a combat step. The log is exact; the figure is a floor, as its definition says.
Vish Kal needed TWO fixes, and the first alone would have measured as "still never
fires" — check `activated <card> targeting` in the log, not just the activation count.

## A Forge run launched from the agent's background shell is cut at two hours (2026-09-30)

The drain-v1 champion baseline — `simulate edgar-vampires --pod standard-v3 --games 200
--jobs 4`, launched as a Claude Code background task — stopped at **84 of 200 games,
wall 7198.9 s**: all four JVMs ended together, exit code 0, no `[manamap] job killed`
line (the per-job cap was 45,120 s), three of them mid-way through an ordinary AI
timeout trace, and the harness wrote a complete record for the games that had finished.
Every earlier tracked run with longer JVMs completed (goblin-storm 16,184 s at 100/100,
edgar 11,243 s at 400/400), so Forge has no two-hour limit; the background task does.
The record is valid for what it holds — 84 games under the pinned harness pool with any
later run at the same tuple — and the remainder was relaunched in its own session
(`python -c "os.setsid(); subprocess.call([... simulate ...])"` under `nohup`; macOS has
no `setsid`) with a FRESH `--seed`, because the default seed is a digest of the
configuration and a second run under it would replay the same games. Rule: a Forge run
expected to pass two hours is launched detached or by the pilot in a terminal, never as
a tool background task, and a short run with a stray `games_completed < games_requested`
is read as this before anything else.


## Unflagged is not castable: three cards in two days, each found after the games (2026-10-01)

| card | the scan said | the games said | the class, read from the script |
|---|---|---|---|
| Vish Kal, Blood Arbiter (−X/−X) | unflagged, hinted | 0 activations across six 20-game passes | `SVar:X:SVar$CostCountersRemoved` priced at 0 before payment; no `IsCurse$` |
| Toxic Deluge | unflagged | drawn 28, cast 0, discarded 0 in a 200-game branch arm | `Count$xPaid` + `PayLife<X>` priced at 0; no `IsCurse$` on `SP$ PumpAll` |
| Bastion of Remembrance | clean | cast in 10 of 30 games it was drawn, castable and uncast on 48 turns | a three-mana enchantment that does nothing the turn it lands — `PermanentAi` puts it behind every creature in hand |

`AI:RemoveDeck:All` is ONE filter, applied before any API is asked. Each API's AI class
then refuses for its own reasons, and none of them is visible from the flag. The pilot:
"we need to make sure the cards we add are actually testing trigger!!!! this is a waste!"
and then "measure twice cut once." The gate is `forge-cast-check --adds` plus a refusal
in `simulate` on a branch seat (`docs/simulation.md`, "The gate"); the rule is that a
card's standing under the Forge AI is a MEASUREMENT taken in a shell before the arm,
stamped with the harness, never an inference from its script — the script only names
the class and the remedy once the shell has said HELD.

## THE A/A, 2026-10-02: THE HARNESS PRODUCED A SIGNIFICANT RESULT FROM NO CHANGE AT ALL

The bench ran experiments for months without ever running the one control that says whether the
instrument can see anything: **one list against itself**. `experiment --aa` has existed the whole
time. No A/A record existed for any deck. The first one, on edgar-vampires' champion at
standard-v3 under the pinned harness, at its first look:

| | games | decided | wins | win rate |
|---|---|---|---|---|
| arm A | 50 | 42 | 7 | **0.167** |
| arm B | 50 | 41 | 15 | **0.366** |

**Identical decklist. Identical table. A gap of 0.199, and the plain Newcombe 95% interval on
the difference is [+0.0086, +0.3734] — IT EXCLUDES ZERO.** At 50 games per arm this harness
manufactures a statistically significant difference between a deck and itself.

For scale: the treasury-v1 arm read 0.217 against the champion's 0.310, a difference of 0.093 —
**less than half the noise the instrument generates on its own.** Four nights of branch verdicts
were read against that.

**THE SEQUENTIAL BOUNDARY CAUGHT IT AND NOTHING ELSE DID.** O'Brien-Fleming demands z = 4.049 at
look 1 of 4, and by that standard the result correctly does NOT clear: `excludes_zero` is false at
the boundary. The protection was built, works, and is the right answer. What fails is the PLAIN
interval — and the plain interval is what `net_change` prints, what a branch objective is graded
on, and what every verdict so far has used. The lesson is not "add a control"; it is **route the
grading through the boundary that already exists.**

The other ten axes all span zero at this look, and one shows the width problem plainly: combat
damage dealt to players read 48.76 against 44.98 with an interval of [−31.9, +24.3]. The interval
is wider than the quantity.

**HOW STRONG IS THIS? WEAKER THAN I FIRST WROTE IT, and the pilot asking "sure?" is what got
it checked.** The firing is marginal: at 42 and 41 decided games, one game either way flips it —
8/42 against 15/41 gives [−0.018, +0.353] and SPANS zero, as does 7/42 against 14/41. So a single
marginal firing of a 95% interval on a null comparison is also just what a 95% interval does
one time in twenty. It is suggestive, not a demonstration, and the first draft of this entry
called it "damning", which it is not.

**The confound WAS checked and is clean**: both arms run job indices [0,1,2,3], so seat rotation —
turn order, which matters enormously in Commander — is matched between them. The only difference
is the seed base (+100,000). And `--aa` refuses to run if the two refs resolve to different lists.

**What the fragility check sharpened.** The two arms are the same deck, so a gap that PERSISTS as
n grows cannot be noise; it would be a systematic asymmetry between the arms, which would be a bug
in the A/A itself. If it shrinks toward zero, the naive test merely fired on a one-in-twenty event.
Looks 2-4 distinguish two genuinely different problems, which is a better reason to let the run
finish than "quantify the noise".

**THE ROBUST EVIDENCE IS NOT THE A/A AT ALL — it predates it and needs no test.** The champion
measuring its own `life_removed_total`: **74.39 at 84 games, 48.77 at 116, 59.53 at 200.** Three
independent samples of ONE list, spanning 26 points, against branch objectives set 19.5 points
above the mean. No significance machinery is required to see that a threshold inside that spread
cannot grade anything.

**What this invalidates.** Every branch verdict on this bench that rests on a win rate or a
per-game mean compared across two separately-run `simulate` records, at 200 games per arm or
fewer, with no A/A beneath it. That is drain-v1, boss-v1, entry-v1 and the killed treasury-v1 arm
on edgar-vampires, and by construction the same reading on every other deck. The measurements are
real; the VERDICTS are not evidence.

**What it does not invalidate.** Mechanism counts from the logs — casts, triggers, activations —
which are counted events and not estimated rates: Vish Kal activated 141 times against 0 before
the PumpAi patch is a fact, as is Marshland Bloodcaster cast 26 times and activated 0. The
deterministic tools are untouched: `mana-analysis`, `mana-fit`, the bracket engine, the citation
contract. And the cast-proof gate is untouched, because "did the AI cast this card" is a count.

**Open, and the reason the run continues.** Look 1 says 50 per arm is hopeless, which was
guessable. The number the bench needs is the gap at **200** per arm, because 200 is what every
branch arm used, and looks 2, 3 and 4 give the ladder at 100, 150 and 200. That ladder is what
sizes every future experiment here. Killing the run after look 1 would leave the instrument known
broken and un-fixed.

**The rule until that lands.** No branch is graded on a Forge win rate or a per-game mean without
an A/A at the same N and the same harness beneath it. A mechanism endpoint — did the card get
cast, did the trigger fire, how many times — is still readable, and is usually the cheaper
question anyway, which the power preflight has been saying all along.

## CLAUDE.md's rule digest, verbatim as of 2026-10-05

> Moved here VERBATIM when CLAUDE.md was compacted on 2026-10-05: CLAUDE.md now carries
> each rule as one line and points here. Several of these bullets had grown past the
> sections above them (the A/A's completion, the `AI:RemoveDeck:All` sweep, the lifted
> board's five omissions), so nothing below was reworded and no number was dropped.
> Arrows (→) inside it point at the page the bullet originally pointed at.

Each of these cost something to learn. The full record — the measurement, the
wrong first attempt, the number — is in the page named beside it.

**Evidence**
- **A validator that fires on correct data is worse than no validator, and the only way to know is to MEASURE IT AGAINST THE WHOLE FLEET FIRST.** Six proposed checks have been prototyped and rejected on this ground; one fired on 27% of correct authored data, another on 29 of 91 components. → `docs/gotchas-evidence.md`
- **Absent means ABSENT, never zero.** A figure nobody measured must be a missing key with a stated reason. `0.0` is a measurement, and a reader cannot tell it from one. → `docs/gotchas-bench.md`
- **Every rate carries its interval, and a comparison carries the interval on the DIFFERENCE.** Two marginal intervals overlapping implies nothing at all. **And a FAMILY of comparisons carries its correction**: `net-change`'s twelve rows are exploratory and Holm-corrected; the objective is the one pre-registered primary, as `win_rate` is in `experiment`. → `docs/gotchas-bench.md`, `docs/simulation.md`
- **Never `cache-record` to make a board green**, and never hand-patch an agent's prose to make a gate pass. Editing prose to satisfy a check puts a fresh claim under an old byline. → `docs/gotchas-bench.md`
- **NO BRANCH IS GRADED ON A FORGE RATE OR MEAN WITHOUT AN A/A AT THE SAME N, because the bench had never run one.** `experiment --aa` is one list against itself and no A/A record existed for ANY deck until 2026-10-02. The robust half of what it showed needs no significance test at all: **the champion measuring its OWN total life removed read 74.39 at 84 games, 48.77 at 116 and 59.53 at 200** — a 26-point spread on one list across independent samples, against branch objectives set 19.5 points above the mean. That alone makes drain-v1, boss-v1, entry-v1 and treasury-v1's verdicts unreadable; the measurements are real, the VERDICTS are not evidence. The A/A's first look agrees but only suggestively — arm A 7/42, arm B 15/41, gap 0.199, plain interval [+0.009, +0.373] excluding zero, and ONE GAME EITHER WAY FLIPS THAT (8/42 vs 15/41 spans zero), so it is a marginal firing at small n and not a demonstration. **Completed 2026-10-03 at 200/arm: 40/176 (0.227) against 48/171 (0.281), gap 0.054, plain interval [−0.038, +0.144] — spans zero, and every one of the eleven endpoints spans zero too.** The look-1 firing shrank as the sample grew, as noise does, so there is no arm asymmetry to chase. What the A/A FIXES is scale: at 200 games/arm ONE list reads 5.4 points apart from itself and the interval is ±0.09 wide, so a branch effect under ~0.14 cannot be seen. Worth noting which guard worked: the O'Brien-Fleming boundary (z 4.049 at look 1) refuses it and the PLAIN interval does not — and the plain interval is what `net_change` prints. Mechanism endpoints stay readable throughout, because a cast, a trigger and an activation are COUNTS rather than estimated rates. → `docs/gotchas-bench.md`
- **A mean is not a result.** Carry median, min and max: a mean of 17.42 against 2.25 read as a sevenfold win when the median was 0 in both arms and two games were the whole difference. → `docs/gotchas-bench.md`
- **THE GOLDFISH HAS NO BLOCKERS, so its verdict on board QUALITY is not evidence.** With eminence, the token doublers, the sacrifice engine and four draw channels all finally modelled, it still preferred a go-wide Edgar refactor on damage, kill rate and card advantage — and Forge, 400 games per arm against the pilot's own pod, gave the refactor **31/400 against the champion's 50/400**, a difference whose interval EXCLUDES ZERO. The mechanism is one number: combat damage dealt to players fell **29.07 → 18.20**, because 1/1 tokens do not connect and the refactor had cut every lord. That one missing assumption outweighed every other gap closed the same day. Judge a go-wide or token strategy in FORGE from the start. → `docs/gotchas-bench.md`
- **THE FORGE AI WILL NOT SACRIFICE FOR A BENEFIT ITS EVALUATOR CANNOT PRICE**, so a Forge result on a sacrifice deck is a FLOOR. `Indulgent Aristocrat` puts +1/+1 COUNTERS on the board and activates 0.41/cast under `--profile Experimental` against 0.07 under Default; `Ashnod's Altar` makes colourless MANA and is **0 for 59 castings under BOTH**, while `Viscera Seer` (scry) was cast 0 times in 500 games. Cost is not the discriminator -- the Aristocrat costs {2} and the Altar is free. Prefer an outlet with a visible BOARD payoff, and a TRIGGER over an ACTIVATION. Check `activated <card>` against `cast <card>` before trusting any result that rests on one. -> `docs/gotchas-bench.md`
- **`AI:RemoveDeck:All` MEANS THE AI NEVER CASTS THE CARD, and this file said the opposite for a month.** `AiController.getSpellAbilityToPlay` filters every non-land ability whose host `isCardRemAIDeck` (2.0.14 bytecode, read 2026-09-30). Measured first: in a 26-land shell with four copies against a slow opponent, Vish Kal was castable in **7 of 12 games and cast in none**; the same script with only that line removed was cast **10 times**; a plain seven-drop control was cast 4 of 5. The fleet sweep agreed — **every `All` permanent in a live deck with games on record was cast 0 times, 14 of 14** (Viscera Seer, Goblin Bombardment, Altar of Dementia, Isochron Scepter, Magus of the Wheel, Vish Kal in 860 games…), while every `Random` one was cast normally. The earlier belief rested on Swan Song, 28 casts in 160 games: a COUNTERSPELL, cast through the reactive path, which does not filter. The sharknado Windfall finding below was THIS. `data/forge_overrides/unflag.txt` lists the 23 cards whose override is the shipped script minus that line; `forge-install --generate` prints any a live deck has since acquired, and a fleet test refuses one. Every run since carries `-ov8bf04cfc`, and THE NULL MUST BE RE-MEASURED under it — `pods.calibration` counts plain-harness runs only, so until then no fresh run feeds a null. → `docs/gotchas-bench.md`
- **A FORGE RESULT ON A DECK WHOSE ENGINE THE AI NEVER CAST IS A FLOOR, and the record now says so.** sharknado's seat cast Wheel of Fortune once and Windfall never in 60 games while DISCARDING Windfall three times and Faithless Looting six -- the log's own statement that the card was held and passed over. `record["engine_casts"]` carries per-card cast / activated / discarded for our seat (MEASURED, top-level, validated where present); `sim/engine_casts.py` reads it at print time against the deck's declaration and prints "held and never cast" at the `simulate` tail, in `deck-info` and live in `sim-progress`. Under the telemetry jar it is no longer an inference: every card carries `castable_uncast` (own turns it ended in hand with the lands to cast it) and `held_while_castable` names the cards held on two or more such turns — HELD WHILE CASTABLE (MEASURED). Check it before reading any rate, the way the piloting gate is checked. → `docs/gotchas-bench.md`
- **THE AI WILL NOT TARGET ITS OWN COMMANDER TO SWITCH ON A COPY ABILITY, and that makes a run measure a DIFFERENT DECK rather than a floor.** Zada, Hedron Grinder's whole deck is "an instant or sorcery that targets only Zada, copy it for each other creature you control". Over 60 games she was **cast 100 times and her trigger fired 21** — 0.35 per game against ~11.9 spells cast. Forge implements the card correctly; its evaluator prices `Brute Force` on Zada as +3/+3 on one creature, identical to any other target, so the targeting heuristic cannot see that this one multiplies. `engine_casts` said the plan was played at 134-160% of expected natural draws and was right: every card was cast, just never at Zada. So goblin-storm's 0.031 baseline and the branch's 0.040 / 0.065 are all void as deck measurements, and the A/B is not rescued by comparing them. `--profile Experimental` does not help — `tokens_observed` moved 1.98 → 1.95. THE PREFLIGHT is one grep: `grep -rc "triggered <Commander>" .../sim/logs/<run>/*.log`. The fix for evidence is `sim-scenario --stack` plus `/resolve-stack` — one of the 21 real boards, proven by CITATION (✓) instead of a rate. → `docs/gotchas-bench.md`
- **UNFLAGGED IS NOT CASTABLE. MEASURE TWICE: NO ADD ENTERS A FORGE ARM UNPROVEN.** Three cards cleared by the scan were found NOT PLAYED only after a night of games — Vish Kal's −X/−X (0 activations, six passes), Toxic Deluge (drawn 28, cast 0 across a 200-game arm) and Bastion of Remembrance (cast in 10 of 30 games it was drawn, castable on 48 turns). `AI:RemoveDeck:All` is one filter; each API's AI class refuses for its own reasons — X priced before its cost is paid (`Count$xPaid` + `PayLife<X>`, `SVar$CostCountersRemoved`), a −X/−X with no `IsCurse$`, no `AILogic$` for the shape, a cheap do-nothing-now permanent cast behind every creature. `forge-cast-check <slug> --branch B --adds --write` proves every add in a two-seat shell and names the class and the remedy; `simulate` on a branch seat REFUSES an add without a PLAYED proof under the current harness. → `docs/gotchas-bench.md`
- **A LIFTED BOARD IS AN INFERENCE, AND IT WAS OMITTING PERMANENTS IN FIVE WAYS.** `sim-scenario` reconstructs a board from an event stream, and every gap in that reconstruction reads as a fact about the game. Found in ONE session (2026-09-28) by pointing `rules-checker` at two real boards: a creature with a characteristic-defining power (`Creature * / *`) never reached the board because `_CREATURE` demanded digits — ~107 resolutions in one run, **all opponents', so it flattered every lethal claim**; every AURA was discarded as an instant (`Rancor (203) - Attach to X` matches `_SPELL`), which made Sphere of Safety's tax read {2} instead of {3}; a TOKEN existed only from the moment it ACTED, so **the lift listed 187 of 1843 live tokens — 10%**, and a published board-width figure was wrong by 18x (">= 6 other bodies" 2.0% against ~21%); `_WORDS` stopped at "seven" so Krenko's "creates eight" was unreadable, losing the BIGGEST boards; and **705 of 857 token deaths (82%) had no owner**, so a death was charged to whichever seat held a match — which can delete an opponent's blocker. All five fixed and tested. A sixth is DOCUMENTED, not fixed: the graveyard is battlefield-deaths-only (mills and discards are not zone events), so a `*/*` power cannot be computed from it — `graveyard_is_a_floor` says so. The repo's only PASSING lifted scenario is radagast/008, which predates all of this — re-lifting gives our seat 11 creatures against the 7 its ✓ was resolved with — but radagast is BROKEN DOWN FOR PARTS, so scoped to live decks the exposure is **zero**. The gate exists since 2026-09-30: `validate-lift` re-lifts every committed board from its own cut and `validate-stack` carries the result as a NOTE — FAIL while it has no verdict yet, NOTE once the loop finished. → `docs/gotchas-bench.md`
- **A FORGE RECORD DESCRIBES THE LIST IT PLAYED, NOT THE LIST ON DISK.** Every record stamps `seats[].decklist_sha256` and nothing read it, so `net-change` reported 120 games as goblin-storm/zada-v1's rate that were played on its FOURTH commit against its seventh on disk — eight cards in and nine out since, including the branch's largest claimed gain. Fleet sweep: **29 of 41 branches, 10,120 games**, mostly champion arms, because a merged branch rewrites the champion's list. It FLAGS rather than suppresses (`forge.list_mismatch`, printed ABOVE the rate in the CLI and on `branch.html`) because the sha is over file BYTES: edgar's list dropped a set code from `Gifted Aetherborn (AER) 61` in the same commit as two real swaps, so a cosmetic edit trips it identically. `engine_casts` IS strict — it named five cards "held and never cast" that were not in the simulated list at all. → `docs/gotchas-bench.md`
- **A 100-GAME FORGE RUN IS NOT A RESULT.** The same champion read 18/100 and then **50/400**, so the first estimate of a refactor's cost was more than double the powered one. MDE against an 0.18 baseline: **42 points at 20 games/arm, 17.5 at 100, 8.5 at 400**. → `docs/gotchas-bench.md`
- **A COMMANDER'S ABILITY IS NOT AUTOMATICALLY MODELLED.** `command_zone_reduction` reads a commander for COST REDUCTION only; Edgar Markov's eminence MINTS A TOKEN on every other Vampire cast and was absent entirely — the deck's whole axis, understating bodies at turn ten by 50%. `deck-audit`'s engine brief had described it in prose the whole time. Before trusting a figure on a deck, check that the model reads the commander. → `docs/gotchas-bench.md`
- **A MEASURE COMPUTED FROM AN AUTHORED FILE IS NOT EVIDENCE, however tight its interval.** The engine lift split games by the `required` flags in `goldfish_targets.json` — which the same hand writes. Three defensible declarations of one Ur-Dragon list, same 10,000 games, same seed, gave **+0.007 (spans zero), −0.036 (REAL) and +0.014 (REAL)** against kill-by-T8; one of them said, at an interval excluding zero, that assembling the engine made the deck win LESS. Deleted 2026-08-28, and `deck_branch.MEMBERSHIP_AXES` now refuses `engine_online_*` and `any_route_*` as branch objectives. Aim a branch at an OUTPUT the deck produces. → `docs/gotchas-bench.md`
- **Every figure carries its definition, in the report that prints it.** A number a reader has to look up elsewhere gets guessed at, and the guesses go one way: a mean read as a rate, a clock read as a win rate, a hoard read as mana. All three have happened. `net_change.METRICS` is the registry and a test asserts it matches `ROWS` exactly, in both directions. → `docs/pilot.md`

**Changing a matcher or a model**
- **Widening a pattern needs a CORPUS SWEEP in the same commit** — newly matched, newly dropped, and the extreme tail read card by card. Skipped once, it billed Jeweled Lotus three mana every turn forever and counted `Add {R}, {G}, or {W}` as three. → `docs/gotchas-bench.md`
- **A CONDITION IS SCOPED TO THE CLAUSE IT ATTACHES TO.** `enters_tapped_unconditionally` searched the whole oracle text for "unless", so Archway Commons — *"This land enters tapped. When this land enters, sacrifice it unless you pay {1}"* — read as an UNTAPPED five-colour source and `mana-fit` offered it as one. Eleven lands share the wording. The obvious fix is worse and the sweep is what says so: scoping to the SENTENCE flags all ten shocklands, whose idiom spans two. → `docs/gotchas-bench.md`
- **A FETCHLAND'S COLOURS ARE A PROPERTY OF THE DECK, NOT OF THE CARD**, so a function
  that takes only a card cannot answer the question and must not pretend to. `land_colors`
  credits basic types in the type line and symbols in an `add` clause; a fetch has neither,
  so all sixteen true fetches in the corpus read as producing NOTHING — and `goldfish` built
  every land's colours from the same call, modelling four fetches as four colourless lands.
  Measured on ur-dragon/landbase-v1: `mana-fit` reported **every colour worse** on a change
  that left colour access flat (W +1, U −1, B +2, R 0, G 0) and **halved the recurring life,
  8 → 4 per tap-cycle**. `land_colors(card, pool=…)` takes the deck; without `pool` it is
  byte-identical, which is what keeps a caller that has no deck reproducible. The sweep's
  load-bearing split is one word: **`a Mountain card` finds a shockland, `a basic Mountain
  card` cannot** — 16 true fetches against 20 Panorama-shaped ones that read almost
  identically. → `docs/gotchas-bench.md`
- **The goldfish CANNOT rank two lands that make the same colours.** It plays the first land in hand and credits its colours the same turn — LANDS have no tapped state and there is no choice of which land to play. (CREATURES do tap when they attack, added 2026-09-26; that is a different question and does not help a land swap.) A twelve-land `candidates` sweep returned exactly two distinct readings, with always-tapped Grand Coliseum tying never-tapped Forbidden Orchard. `mana-analysis` and `mana-fit` are deterministic for exactly this reason and are the whole of the evidence for a land swap. → `docs/gotchas-bench.md`
- **A flag the model sets is a claim the model must ACT ON.** `treasure_doubler` shipped set-and-unread; fifteen candidates returned byte-identical −0.026. `tests/test_metric_hygiene.py` checks this now.
- **A ONE-TURN GRANT IS NOT A PERMANENT DOUBLER, and a channel placed where it cannot fire is not a channel.** `team_damage_multiplier` matched "creatures you control gain double strike" anywhere in an oracle, so Elesh Norn // The Argent Etchings doubled all of sharknado's damage off a SAGA BACK FACE chapter lasting one turn, behind a three-creature sacrifice — cutting it measured **−2.2 damage** and read as a reason to keep it. **75 corpus cards read as permanent doublers, 53 after the fix: 22 were phantom**, and sharknado's damage@8 fell 31.45 → 29.47. The rule is ONE-SHOT versus RE-APPLIED, not temporary versus permanent: Atarka's identical clause fires every combat and is real, which an existing test caught when the first fix dropped it. Separately, the Blood crack was placed on leftover mana and a goldfish spends its pool casting — **49 of 300 games ended with Blood uncracked**, a confident zero from a channel that never ran. → `docs/gotchas-bench.md`
- **A CARD THE MODEL CANNOT READ LOOKS EXACTLY LIKE A CARD THAT DOES NOT HELP.** Four in a row measured "no effect" from a sweep that had never priced them — 400 of 405 sacrifice-gated draws read as zero, and Blood, artifact-sacrifice payoffs and draw DOUBLERS had no channel at all. Before trusting a null on a swap, check `draw_profile`'s `unmodelled` and `model-coverage`. When a card's value is contingent on something the model omits, measure the CEILING with the omission reversed and LABEL it: Jaws came back floor −0.43, ceiling +0.09, which settles the card instead of leaving a standing doubt about the instrument. → `docs/gotchas-bench.md`
- **A model change makes every derived artifact stale.** `meta.model_version` (a sha over `goldfish.py`, `goldfish_profiles.py`, `goldfish_library.py` and `goldfish_turn.py` — `_MODEL_FILES`, so splitting the module did not blind the stamp) makes that decidable; the three prose validators REPORT it and never fail on it. Regenerate the fleet after any model change. The 39 figures already stale predate stamping and report as unknown, not stale. → `docs/gotchas-bench.md`
- **Adding a metric requires re-running the independence check.** Three magnitude axes shipped that were one axis at r = 0.92–0.98. → `tests/test_metric_hygiene.py`
- **THE COMMANDER'S OWN TEXT IS NOT MODELLED UNTIL SOMEBODY MODELS IT.** zur-enchantress was rebuilt around Zur, Eternal Schemer and the goldfish read NEITHER of his abilities — the static grant of deathtouch/lifelink/hexproof to every enchantment creature, nor the `{1}{W}` that animates an enchantment into a body whose power is its mana value. Modelling them took kill-by-t8 from 0.153 to 0.327 on an unchanged 99. A commander ability that only one card in the corpus has is DECLARED per deck (`model_commander_animate`, `model_commander_attack_tutor`); one a handful share is parsed after a sweep. → `docs/gotchas-bench.md`
- **A CARD CAN BE READ CORRECTLY AND NEVER PLAYED.** Every casting loop in the goldfish selects on a CHANNEL — draws, ramps, makes Treasure, has a body — and a card matching none of them sits in hand for ten turns while its profile says exactly what it would have done. Found FIVE times in one session and only caught as a class on the fourth: the Shrines measured as exactly nothing, a SLEEVED deck ran its sacrifice engine on 2 of its 4 outlets, and four of six attack enablers were uncastable, which made the model unable to start its own engine. `model_coverage.never_cast` / `silent_losses` are the predicate and a fleet test asserts no deck computes an effect it never applies. **Teach the casting predicate in the SAME commit as the ability.** → `docs/gotchas-bench.md`
- **A RATE DRIVING A FIGURE MUST NAME WHERE IT WAS MEASURED.** `model_commander_attack_tutor` fired every turn and reported 5.70 fires a game; Forge resolved the search 1.22 times. Correcting it took kill-by-t8 from 0.501 to 0.173 and undid more than half of one day's measured gains. `fires_per_turn`, `model_deaths` and their `source` keys are REQUIRED for exactly this reason, and a CEILING (1.0 when the attack is free) is labelled as one in the record rather than read as a forecast. → `docs/gotchas-bench.md`
- **A land whose only coloured mode costs extra mana is not a coloured source on curve.** `land_colors` counts `{1}, {T}: Add one mana of any color` at full value; six such lands made every colour in zur-enchantress read at or above target when all three were short. Reported (`sources.gated`, `on_curve_probability.lands_only_ungated`) rather than discounted, because a fraction to divide by would be an authored number driving a headline. The obvious fix is BACKWARDS — cutting them for basics makes every colour worse. → `docs/gotchas-bench.md`

**Branches, paths and artifacts**
- **A branched write needs a branched READ.** Three instances now, the third committed inside the commit fixing the class: `goldfish.main` measured the champion and filed it under the branch, understating turn-10 hoard by 4×. Every branch measurement must record the branch's own `decklist_sha256`. → `docs/gotchas-bench.md`
- **A BRANCH MAY NOT DECLARE A MODEL CHANNEL ITS DECK DOES NOT — that is two simulators, not an A/B.** `common.deck_file` prefers a branch's copy of an authored file, and its docstring asserted nobody writes a second `goldfish_targets.json`. goblin-storm/zada-v1 did: it declared `model_draw`, `model_combat` and `model_commander_copy` while the deck declared NOTHING, so Zada's whole ability was modelled on one arm only. `net_change` then dropped **7 of its 12 rows** (no champion `output` block at all) and the two surviving rows that moved were the two the asymmetry flattered — missed-drop-by-T5 −0.083 "better" → noise, interaction-affordable-@T6 −0.037 "worse" → **+0.095 better**. Declared identically, DARK went 7 → 0 and the branch won ten of twelve rows: damage @T10 **16.01 → 27.73**. A branch MAY name its own new cards in a target's `any_of` (meren-recursion does, legitimately — a target names cards); it may never change the instrument. → `docs/gotchas-bench.md`
- **`--out` on a per-deck command is slug-scoped, and a shell redirect cannot be policed.** Concurrent agents overwrote each other's views seven times across two sessions. → `docs/gotchas-bench.md`
- **A new tracked artifact needs a gate in the same commit** — a validator, a freshness test, or both — and a `deck_status.VALIDATED` entry so the status command sees what the tests see. → `docs/gotchas-evidence.md`
- **ONE PREDICATE, ONE HOME.** Four modules had grown their own answer to "is this deck in a pile" — `common.UNPLAYABLE_STATUSES`, `deck_info.STATE_RETIRED`, `net_change.FREE_TO_RAID` and `deck_branch._deck_holders`, which carried the status and did nothing with it. None disagreed yet and it was already costing something: `deck-branch merge` refused Ur-Dragon on 12 cards, 4 of which sit in decks that do not physically exist. `common.deck_is_apart` decides; everything else reads the row. → `docs/gotchas-bench.md`
- **A DECIDED BRANCH IS NOT AN EXPERIMENT, and until `propose` shipped they rendered identically.** A branch had two observable states — the directory exists, or `merged` is present — and `delete` was the only reader of `merged`. `deck_branch.branch_state` derives six and stores none, so a proposal un-blocks itself when a card lands in a box. `base_version` had been written since branches shipped and **no code had ever compared it to anything**; that comparison is `PROPOSED · OUTRUN`. → `docs/pilot.md`
- **Count COPIES, not decklist entries.** `cards.json` stores basics as one entry with `quantity: N`; counting entries once published "18 lands" for a 33-land deck. Use `common.expand_copies()`. → `docs/gotchas-bench.md`

**Tests**
- **A test that re-derives the rule is testing itself.** Drive the production function, and prove the test by RE-INTRODUCING the bug it was written for. Four such tests shipped, one guarding the flagship metric. → `docs/testing.md`
- **A loop over a possibly-empty collection needs `assert checked >= N`.** Fourteen lacked it; several passed by iterating zero times.
- **A control can be blind to the class it exists for.** The branch control proved the WRITE landed correctly and could not see a read from the wrong place.

**The frontend**
- **Cache-bust `?v=N` on every script and CSS tag in `viz/index.html` AND `viz/deck.html` after any JS/CSS change**; `index.html`'s nine busts move together. Bump `DATA_VERSION` whenever a consumer would draw a DIFFERENT CONCLUSION from the bytes — a retrain qualifies, a content refresh does not.
- **`viz/` and `data/` must stay top-level siblings**; every fetch is `../data/<file>`. Serve from the repo root.
- **A renderer kept behind a flag is a renderer nobody is testing.** → `docs/gotchas-viz.md`

## CLAUDE.md's command commentary, verbatim as of 2026-10-05

> The annotated command block CLAUDE.md carried until the 2026-10-05 compaction, kept
> whole because its comments hold measurements (the pod calibration, the telemetry
> patch, `regen`'s timings, the cast-check record). CLAUDE.md now lists the commands
> with one line each; `docs/pilot.md` is the reference.

```bash
manamap run                   # full 15-step pipeline (steps 1 & 7 need internet)
manamap run --from STEP       # resume from a step
manamap <step>                # single step; `manamap --help` lists all 28 top-level subcommands
manamap synergy && manamap power-creep && manamap cluster-regions && manamap card-roles
                              # fast analysis-only refresh (no retrain)
manamap pilot <cmd>           # the bench (123 pilot subcommands); `manamap pilot --help`

manamap pilot deck-info <slug>                          # START HERE: where a deck stands + a derived NEXT
manamap pilot try <slug> --out "A" --in "B" [--out C --in D …] [--each] [--stage NAME]
                              # THE SWAP LOOP (2026-10-04): an idea to an answer in ~10 s, one
                              # screen, nothing written. Every card in and out with its roles,
                              # declared target, what the goldfish can SEE of it, and what
                              # Forge's AI did with it on record (AI behaviour, NEVER a reason
                              # to cut); the keep list; colour sources before/after; the
                              # net-change rows (same harness, measured identical to a staged
                              # and fetched branch); one line. `--stage` writes a branch only
                              # after the screen. Through `manamap serve` it skips the cold start.
data/decks/<slug>/protected.json                        # THE PILOT'S KEEP LIST, hand-written only:
                              # stage / new / propose / merge, try, the build's must-include,
                              # the candidates auto-cut and the diagnosis/prescription gates all
                              # refuse to cut a card it names. `validate-protected` gates it.
                              # Born of draw-v1 cutting Vish Kal unread.
manamap pilot build <slug> --commander "<name>" [--brief "…"] [--from FILE]
                              # THE ONE COMMAND (PRD Epic A): brief -> a legal, MEASURED 99
                              # on the bench, six stages in ~10s. Omit --commander and it
                              # proposes three and halts. The dev batch is the GOLDFISH,
                              # not Forge: a 12-minute Forge batch is ~20 games, whose MDE
                              # is 42 points. Forge is a TARGETED PROBE since 2026-10-04 (`forge-cast-check`);
                              # a pod run is optional and never gates a merge.
manamap pilot validate-brief <slug> [--themes]          # the gate brief.json never had
manamap pilot check-in <slug> --from <file>             # a PAPER list -> decklist.txt: diff, refuse, apply
manamap pilot deck-version <slug> [list|show|tag|restore|paper]  # every list from git, joined to the log;
                                                        #   `paper` marks the version you have SLEEVED
manamap pilot deck-state <slug> [archive|retire|supersede|revive] --reason "…"
                              # IS THIS STILL A DECK OR A PILE OF CARDS. Writes
                              # deck_versions.json's `lifecycle`, WITHDRAWS the paper
                              # lock (the two contradict), and rewrites info.json —
                              # which it must, since `regen` skips archived decks
manamap pilot deck-delete <slug>                        # only a deck that was never sleeved,
                              # never played and never published; git rm, staged not committed
manamap pilot deck-notes <slug> add "…" --result win|loss --cause <code>
                              # the captain's log (authored). `--cause` is a CLOSED
                              # vocabulary (deck_notes.CAUSES) so the dossier's priors
                              # table can COUNT how games end; it lands in the sidecar
                              # log_causes.json because log.jsonl is append-only
manamap pilot deck-notes <slug> cause <id> --cause <code>   # file one after the fact
manamap pilot simulate <slug> --pod standard-v3 --games N
                              # Forge, seeded, against THE STANDARD TABLE — three
                              # bracket-3 decks with ZERO combos between them,
                              # chosen by a ROUND ROBIN with none of our decks
                              # seated, then CALIBRATED with five of them (185
                              # decided games): sythis 1.45x, subject null 0.292,
                              # jarad a 0.48x floor. `standard` (giada 2.15x) and
                              # `vito-era` (13 two-card infinites, 0.447) are kept so
                              # old records resolve; naming either is deliberate.
                              # The table is `vito-era`; `vito` alone is an
                              # opponent SEAT under data/opponents/, not a pod.
                              # A POD'S NULL IS A PROPERTY OF THE TABLE WITH THE
                              # SUBJECT IN IT: sythis reads 0.25 against heliod
                              # and 0.66 against zur. Read `pods <name> --calibration`.
                              # A clock-out is `truncated`, has NO winner, and is
                              # excluded from the rate — it used to be awarded to
                              # the last seat, which our deck can never be.
                              # THE PREFLIGHT PRINTS FIRST: the null, the MDE at N,
                              # and what +0.05..+0.20 would need. `--detect X`
                              # REFUSES a run that cannot see X at 80% power;
                              # `--anyway` runs it as a screen (same on `experiment`).
                              # `--list` labels every run with the VERSION it
                              # played and `NOT the current list` where it is not.
manamap pilot fetch-opponent "<commander>" --as <slug>  # a pod seat under data/opponents/
manamap pilot validate-forge-hints <slug>               # forge_hints.json: per-card AILogic / AIPreference
                              # hints derived onto the shipped scripts (the shape Forge's own
                              # aristocrat cards use), plus a `forge` rule in pilot_policy.json
                              # for the AiProps knobs. `forge-install --generate` installs
                              # both; the record's card_overrides / ai_profile shas say so.
                              # Measured 2026-09-30: an unflagged outlet is CAST and then sits
                              # idle — `idle_on_battlefield` names it — until it is hinted.
manamap pilot forge-telemetry [--build]                 # THE PATCHED LOG FORMATTER. Forge's shipped log
                              # keeps two zone transitions; one patched method logs EVERY
                              # zone change by name and owner (draws, tutors, mills, wheels,
                              # arrivals). Measured purely observational 2026-09-30: pristine
                              # twice and patched once on a quiet machine differ on the
                              # millisecond line only. The jar is a COPY beside the pristine
                              # one; `simulate`/`experiment` use it when it is there and
                              # stamp `-tl<sha8>` + a `telemetry` block; a class the manifest
                              # does not register REFUSES the run. THE PATCH SET HAS KINDS:
                              # `log` (the formatter, observational) and `ai` (MillAi's
                              # `AILogic$ SacOutlet`, 2026-09-30 — a sacrifice-cost mill
                              # ability fires when a creature of ours is about to die anyway);
                              # a set with an `ai` class changes play and `net-change`
                              # buckets on it like a card override.
manamap pilot sim-scenario <slug> <run> --game G --turn T --stack   # lift a board -> /resolve-stack
manamap pilot sim-findings <slug> --write   # THE SIM DEBRIEF'S SKELETON (sim_findings.json):
                              # per run, findings with ids, intervals and sources. Prose is
                              # the sim-debrief agent's and may cite only finding ids; the
                              # merge recomputes the skeleton. The captain's log is NEVER
                              # written from a Forge game — it has no pilot.
manamap pilot sim-boards <slug> <run> --criterion held --lift --stack   # WHICH board: a criterion names
                              # a cut and a SHAPE, the shortlist is ranked by how many games
                              # RECUR to it (one game is an anecdote), and `extras.finder`
                              # carries the provenance a handbook proposal cites.
                              # `validate-lift` re-lifts every committed board from its own
                              # cut: FAIL while it has no verdict yet, NOTE once finished — the
                              # gate CLAUDE.md:400 said was missing.
manamap pilot prescribe <slug> "<question>"             # open a question to the doctor (then /prescribe)
manamap pilot experiment <slug> --a V1 --b working --pod <name> --games N [--looks K]
                              # THE CONTROLLED A/B. `--looks K` (<=4, O'Brien-Fleming):
                              # each look is WHOLE ROTATED JOBS from both arms at its own
                              # boundary z, never a partial job; the record is rewritten
                              # after every look and `--resume` continues it; `--until-mde X`
                              # is non-binding futility. `--aa` is one list twice (the noise
                              # floor); `--profile-b P` is policy-on vs policy-off on one list.
                              # Seats rotate per global job like `simulate`; the id carries
                              # clock, overrides and AI-profile shas, empty at their defaults.
manamap pilot campaign <name> plan|run|status   # THE OVERNIGHT QUEUE. data/campaigns/<name>.json
                              # is a TRACKED pre-registration of A/Bs; `plan` pins refs to
                              # shas, preflights, prepends an A/A per harness; `run` skips
                              # DONE/STALE, resumes RUNNING, NEVER merges; state is derived
                              # from the records, never stored.
manamap pilot net-change <slug> --branch <name> --write  # what a branch costs and buys.
                              # ONE PRIMARY (the objective), TWELVE EXPLORATORY rows,
                              # Holm-corrected. THE REAL TABLE IS IN THE RULE: a Forge
                              # win-rate loss whose interval excludes zero at the same
                              # pod blocks a merge whatever the goldfish said; one that
                              # spans zero changes nothing. The block stores the pod's
                              # null and every Forge endpoint with its interval.
manamap pilot deck-branch <slug> new <name> --objective "forge.win_rate >= 0.25 @standard-v3"
                              # a Forge objective NAMES ITS TABLE or is refused; graded on
                              # the branch's pooled rate there, with the interval on the
                              # difference and the null in the grade. THE AXES INCLUDE THE
                              # DECK'S IDENTITY (2026-09-30): forge.drain_dealt (life loss
                              # that was not damage), forge.life_gained, forge.biggest_hit,
                              # forge.evasive_damage_share, forge.kills_by_ability — each
                              # with its floor named in analysis.limits
manamap pilot deck-branch <slug> propose <name> --as v1.0.2   # accept it; wait for cards
manamap pilot deck-branch <slug> withdraw|reject <name> --reason "…"  # the reason goes in the LEDGER
manamap pilot decisions <slug> [outcome|backfill]   # THE DECISION LEDGER (decisions.jsonl,
                              # append-only): every propose/withdraw/reject/merge with the
                              # report's prediction frozen; `outcome` joins the merged list's
                              # own runs at the same pod+harness back to the merge —
                              # predicted beside realised, inside the interval or not.
                              # `deck-info` says when a merge can be closed.
manamap pilot card-search --deck <slug> --oracle REGEX [--owned]         # mine the corpus
manamap pilot scan-candidates <slug> [--dimension drain|gain|threat|outlet|sweeper|draw] [--against-branch B] --write
                              # ONE PASS along the deck's DIMENSIONS: every row names the
                              # predicate that admitted it (oracle id / role / tag / printed
                              # keyword); a converter or a two-card infinite with the (staged)
                              # 99 is FLAGGED and sorted last, never ranked or dropped; death-
                              # draw splits on `nontoken`. Retrieval, not judgement — sorted by
                              # EDHREC rank. Writes the dated candidate_scan.json (validated)
manamap pilot forge-cast-check <slug> --card "Toxic Deluge" [--games 8 --copies 4 --vs giada-angels]
                              # PROVE THE AI PLAYS A CARD BEFORE A NIGHT IS SPENT ON IT: a
                              # two-seat shell (the deck's commander, N copies, its own cheap
                              # spells as filler, basics), counted from the telemetry hand
                              # facts — drawn / cast / activated / HELD while castable.
                              # Unflagged is NOT castable: Toxic Deluge was drawn 28 times
                              # and cast 0 in a 200-game branch arm (2026-10-01) because its
                              # script lacks IsCurse$ and X is priced before the life is paid.
                              # Every add that must be cast or activated for a branch's
                              # objective runs this first; a HELD card is a piloting item.
                              # `--branch B --adds --write` proves EVERY add and writes the
                              # branch's cast_proofs.json (validated, stamped with the
                              # harness); `simulate <slug>@<branch>` REFUSES an add that is
                              # not PLAYED under the current harness (--anyway runs it with
                              # the slots recorded as FLOORS), net-change prints CAST PROOFS
                              # and marks the primary FLOOR, deck-info NEXT names the check
manamap pilot fetch-edhrec <slug> [--theme aristocrats] # EDHREC's commander page(s) as dated per-card
                              # synergy / inclusion (edhrec_cards.json, ★ evidence, validated);
                              # cards newer than the corpus are listed apart, not failed
manamap pilot model-coverage <slug>                     # WHAT THE MODEL CANNOT SEE, before the games:
                              # seen / DARK (feeds a channel that is OFF) / invisible.
                              # 236 DARK cards across the fleet when it shipped; goldfish
                              # and net-change now print the headline as a PREFLIGHT.
manamap pilot regen [--only STAGE] [--slug S] [--jobs N] [--dry-run]
                              # REBUILD THE FLEET after a model change, in dependency
                              # order (goldfish -> mana-analysis -> net-change ->
                              # diagnose -> benchmark -> deck-info), parallel across
                              # TARGETS. MEASURED 2026-09-04 at 72 targets: 109s at
                              # --jobs 8, the goldfish stage alone 83.6s -> 23.7s. The
                              # fleet is 96 targets now, so that is a ratio, not a
                              # runtime. BIT-IDENTICAL: games inside
                              # one run are never split, only decks are.
                              # A MISSING artifact is CREATED, not skipped -- but only
                              # on a SLEEVED deck (`regen.BOOTSTRAP` + `is_pinned`).
                              # REFRESH IS EVERY LIVE DECK; BOOTSTRAP IS SLEEVED ONLY.
                              # Two questions, and one gate used to answer both: the
                              # sweep was sleeved-only while the freshness tests check
                              # every deck that is not RETIRED, so a model change left
                              # emiel-blink and meren-recursion stale and the board red
                              # (2026-09-26). An artifact that EXISTS is tracked and
                              # already gated, so it is rebuilt wherever it lives; an
                              # artifact that is MISSING is still only created on a
                              # SLEEVED deck, because minting a tracked figure for a
                              # list that changes daily is the pilot's call.
                              # `manamap pilot regen --jobs 8 && make manuals` is now
                              # the WHOLE recipe after a model change.
manamap pilot deck-info <slug> --write                  # write info.json for the deck page
manamap pilot build-poh <slug> && manamap pilot build-index    # the HANDBOOK + the manifest
# agents (Claude Code skills): /publish-deck sequences the lifecycle; then
# /build-deck /analyze-engine /resolve-stack /write-manual /poh-procedures
# /debrief /sim-debrief /captains-log /prescribe /diagnose-deck /research-strategy /refresh-corpus.
# 22 skills in .claude/skills/, 18 charters in .claude/agents/

make test                     # THE INNER LOOP — non-browser, -n auto, cached.
make test-fresh               # same with nothing cached; trust this one.
                              # RUNTIMES LIVE IN docs/testing.md, not here. This
                              # line said ~22s/~29s for weeks while the real
                              # figure was 772s — a number nobody re-measured
                              # after the suite tripled.
make test-browser             # the playwright suite
.venv/bin/pytest -n0 -k NAME  # one test, no worker startup
.venv/bin/pytest -m forge     # ONE real Forge game; needs ~/.mana-map/forge
.venv/bin/pytest -m ""        # literally everything, browser included

# .mcp.json registers an MCP SERVER (`manamap.mcp_server`) exposing seven read-only
# tools to Claude Code: deck_state, fleet, search_docs, search_code, stats,
# run_command, command_help. Structured data from the warm daemon instead of parsed prose —
# `deck-status heliod` is 2.9s cold, 0.003s warm, byte-identical. No SDK: MCP is
# JSON-RPC over stdio and the subset a tool server needs is ~150 lines, the same
# reasoning that keeps scipy out of sim/stats.py. It CANNOT write; the gate is
# `serve._cli`, imported rather than restated.

manamap serve                 # viz + a LOCAL /api the deployed site does not have
                              # ALSO A WARM WORKER: with it running, every read-only
                              # `manamap pilot <cmd>` routes through /api/cli and skips
                              # the cold start. query-rules 6.93s -> 0.16s (43x),
                              # deck-facts 1.44s -> 0.14s, deck-audit 2.26s -> 0.59s;
                              # output byte-identical. Fails OPEN — no server, or any
                              # error at all, and the command runs locally as before.
                              # MANAMAP_NO_DAEMON=1 opts out; MANAMAP_DAEMON=host:port
                              # points elsewhere. Restart the server after a code change:
                              # it holds the old modules until you do.
python -m http.server 8000    # or plain static, FROM REPO ROOT (no Build agents)
# http://localhost:8000/viz/workbench.html          THE LANDING PAGE — start here
# http://localhost:8000/viz/index.html              the card map (3 modes)
# http://localhost:8000/viz/index.html?cards=1)%20Sol%20Ring,%202)%20Zur%20the%20Enchanter
#                                                   a walk seeded from cards you name
# http://localhost:8000/viz/deck.html?deck=heliod   a deck's dossier
# http://localhost:8000/viz/branch.html?deck=ur-dragon&branch=eminence-v3
#                                                   a candidate 99 and its net change
# http://localhost:8000/manuals/p/heliod.html       its Pilot's Operating Handbook (printable, no JS)
```
