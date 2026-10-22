# PLAN — current state and what's next

*The resume-here doc. `docs/vision.md` says what the bench is; `CLAUDE.md` carries the
rules; this says where things stand and what is open. The plan as it stood until
2026-10-05 — every dated block, and the pilot's PUNCHLIST of 2026-09-16 — is
[`docs/history/plan-to-2026-10-05.md`](docs/history/plan-to-2026-10-05.md).*

Last updated **2026-10-19**. Do not quote a figure from here without re-running the
command beside it.

## Where things stand

**PRD v2 is built except for the deletions.** One living Deck Context per deck
(`data/decks/<slug>/CONTEXT.md`, `docs/deck-context.md`), `/jarvis` as the one entry
point, the lean agent roster with response-time targets, the hypothesis queue
(`docs/queue.md`), Forge scenario slices (`manamap pilot scenario-ab`), the live job graph
in the job band and the page-state beacon all shipped 2026-10-07..08. Sharknado's context
is approved; the other seven are in review. The engine model, the handbook (`poh*.py`,
`manuals/p/`) and the stack-resolution chain are still in the repo, because nothing is
deleted until every context is approved.

**Card and game support shipped 2026-10-08..09.** Banned vs not-legal and Game Changers on
the card panel; known combos per card and per deck (`deck-combos` -> `combos.json`); prices
as dated evidence (`prices` -> `prices.json`, from Mana Pool's public feed — no token —
with Scryfall as the fallback; `docs/integrations.md`); a printings picker and a Mana Pool
buy list; 60-card formats (`pilot/formats.py`), with the first Modern deck, `elves`;
Moxfield export, link and import by paste (`deck-export`, `deck-link`).

**The fleet is rated, and the rating is honest about what it cannot see (2026-10-19).**
Rating v2 (`src/manamap/pilot/rating.py`, `rating_mc.py`; `docs/pilot.md` "The rating";
known-issues §27): the weights are a Dirichlet prior, the inputs are drawn from their
posteriors, DeckCheck's read is a component, a win the goldfish cannot see halves speed's
weight, and `concordance` publishes how far a deck's instruments disagree — a divergent deck
prints its grade as a range. The fleet analytics artifact (`data/fleet/analytics.json`,
`viz/fleet.html`) sits under it with profiles and bands and never a score. The pilot's 1–5
survey rides on `deck-notes add --feel`. Every judgment sits at its measured score today
because no logged game is on a current list; `rating --calibrate` refuses at six decks.

**The fleet.** Six decks are sleeved — edgar-vampires, gishath, goblin-storm, heliod,
sharknado, ur-dragon (`deck-version <slug>`); emiel-blink, meren-recursion and elves are
on the bench; the rest are retired or broken down. Every live deck and branch is priced.

**The push gate** (`make prepush`, `docs/testing.md`) runs only the tiers a diff can break
and leaves the fleet regen to CI. CI's browser job runs two workers (a hosted runner has
four cores); the corpus job runs weekly or by hand.

## The fleet, rated — the reading and the pilot's calls (2026-10-19)

`manamap pilot rating --all` as it stood after the third `/rate-fleet` pass (★ coaching, 90%
intervals, within bracket, no significance claim; re-run the command before quoting):

| bracket | deck | grade | score | P(1st) | concordance | skeptic | the low component |
|---|---|---|---|---|---|---|---|
| B3 | gishath | B | 76.7 | 95% | mixed | pass | resilience 70 |
| B3 | ur-dragon | C–B | 69.0 | 5% | divergent | pass | dti 50, construction 56 |
| B3 | goblin-storm | D–C | 57.1 | 0% | pass | pass | resilience 33 |
| B4 | heliod | C–B | 65.1 | 77% | divergent | fail at cap | speed 0 (Approach) |
| B4 | sharknado | D–B | 59.0 | 23% | divergent | pass | resilience 13 |
| B4 | edgar-vampires | C | 50.6 | 0% | mixed | fail at cap | speed 22, record 24 |

What it says, deck by deck (`sensitivity.leave_one_out` is the figure behind each line):

- **Gishath is the best Bracket 3 deck and the only deck the instruments agree on.** Every
  instrument ranks it first or second; the low end of its interval is still a B; an A would
  need speed at over half the weight, which no honest prior gives it. **Play it, don't
  change it.**
- **Heliod is the best Bracket 4 deck on paper, Sharknado by the one instrument built for
  Bracket 4 speed**, and the two are one evening of logged games apart. Heliod's first place
  rests on DeckCheck's card-quality read and the Approach exception, not on anything measured
  at a table; its instruments disagree completely (dti and resilience first, the clock,
  land drops and construction last).
- **Sharknado has the most upside in the fleet.** One component holds it at D–B: resilience
  13.3 on eleven answers covering one class. Without it the score is 67, a B; everything
  else about the deck is at the top of its bracket (kill by turn eight in 89% of goldfish
  games, mana placed first). The staged **torbran-v1** branch already carries the fix —
  Silence, Snapback, March of Swirling Mist are what that component counts.
- **Goblin-storm is second for upside**: resilience 33 (thirteen answers, zero sweepers)
  and no game yet on v1.1.0. The thirteen cards arriving are bodies, not interaction, so
  the lift comes from logged games first and a sweeper or two later.
- **Edgar has the most ground to gain in absolute terms** (speed 22, no wins in five) and
  needs the mean-v1 rebuild, not a slot or two. Its upside is real and expensive.
- **Ur-dragon has a land pass in it** (construction and the land-drop miss rate are its low
  components; the Bracket 3 clock is already there).
- **Heliod has the least**: speed is zero and cannot move while Approach is the win, and its
  other components sit near their anchors. Its upside is games, not cards.

**Sharknado released the same day** (`/release-branch`, its first run): torbran-v1 merged as
**v1.3.0** on printed proxies, the paper lock moved, DeckCheck re-pasted and rescanned
(31/96 B4 → 33/96 B3 on their read; our floor 3, and the pilot's TARGET IS 3 — "optimized high bracket 3", corrected 2026-10-19; the earlier 4 was wrong), the goldfish declaration repaired
for the new list, the deck re-rated: resilience 13 → 40, score 59.0 → ~65 C–B, now ranked in Bracket 3 beside Gishath, Ur-Dragon and Goblin Storm (`rating --all` for the settled figure; Bracket 4 is Edgar and Heliod, n=2, so their placements are withheld). The order list is
the branch's 18 unowned adds (`deck-branch sharknado show torbran-v1`). Known-issues §28:
regen zeroes a merged branch's `net_change.json`; the pre-merge reading lives in the ledger.

**Branches are findable (2026-10-19).** The workbench chip links the live branch with its
verdict; the deck page's Branches rack sits after NEXT, newest first, every state badged;
the branch page says its state, the goldfish's blind spots, what Forge's AI did with each
add, the stacks proven on it, the deck's own goldfish beside the branch's, and the
siblings; `deck-info` prints a BRANCHES block; Jarvis has a "Review a branch" intent;
`/release-branch` ships one. Edgar's eleven stale experiments are REJECTED on the ledger
with their reasons; mean-v1 is its one open branch (`viz/branch.html?deck=edgar-vampires&branch=mean-v1`).

**Later the same day.** Sharknado's proxies went in (v1.3.0 in the sleeves). Goblin-storm's v1.1.0 was tagged and locked (eleven of thirteen adds in; Purphoros to find, Fable in the mail). The stale branches across the sleeved fleet were REJECTED with their reasons: goblin-storm's zada-v1 and copy-burst-v1, gishath's cart-v1 (absorbed by mana-v1), ur-dragon's six-branch 2026-09-11 chain (absorbed by final-v2), heliod's splendor-v1 and blue-v1 (absorbed by splendor-v2) — twenty-two in all with Edgar's eleven. Three open branches remain, each a real call of the pilot's: edgar mean-v1 (HALTED — see NEXT_SESSION.md), gishath lands-v1 (Savannah, Taiga, Sacred Foundry for three unfetchable lands), ur-dragon protection-v1 (Radiant Destiny for Whirlwing Stormbrood). Edgar's mean-v1 was restaged as option C: Elspeth, Storm Slayer and Crossway Troublemakers STAY — both had been cut off DeckCheck's list and the instruments' silence, and a card the goldfish cannot read is never a cut on that ground.

**A seventh sleeved deck (2026-10-19): keen-engineering**, the Foundations Commander mono-blue precon (Sai, Master Thopterist) as printed — the pilot's Bracket 2 / low-3 power-down deck, v1.0.0, paper-locked, Context seeded, measured and rated. It is the only Bracket 2 member, so its rating is anchors only with placements withheld until two more decks join the group; Sai's Thopter trigger and sacrifice-draw are not modelled, so its goldfish figures are a floor. DeckCheck read the same evening (30 of 96, their Bracket 2) and the deck was re-rated B 79.3. Every agent phase ran that night: engine model (critic pass), strategic frame, three checker-passed stacks, the handbook's procedures, the pilot notes and one decision spread — the first deck to go from a printed list to a complete bench record in a day. What it lacks is games: the captain's log is empty and every procedure says so.

**Edgar v2.1.0 (2026-10-20).** The sleeved list was scanned and checked in first (three cards of drift, v2.0.1), then mean-v2 — the profile package, four proactive-denial Vampires (Anowon, Drana and Linvala, Captivating Vampire; Charismatic Conqueror was already sleeved), Bloodghast, a land pass — merged on printed proxies via `/release-branch`. DeckCheck's read moved 28 → 46 of 96 (Proactive Denial F → B, Critical Onset B → A); the rating stayed C (61.8) at the code score, because no game has been played on the list and the goldfish cannot see a lock, a tax, an edict or a steal. Rule proposed and applied here: a paper lock is written only after a scan is checked in.

**Sharknado's harness (2026-10-21).** Game 002 (Oliver's, a bracket-3 list at a bracket-4
table, lost to an infinite) brought the first piloting read on v1.3.0: too many wheels, a
kept hand of one creature and wheels, wheels as FINISHERS once a commander is down. The
goldfish had been firing a wheel on any thin hand, so the pilot's rule is now a declared
policy (`pilot_policy.json`: hold a wheel until a commander is out; a keep has at most one
wheel and a body or ramp, reported never enforced), the wheels' timing is a figure (finisher
wheel by T8, wheels cast by T10, dead wheels in hand @T5), and a branch aims at the finisher
row by his choice, with the whole profile watched. The policy moved under 2% of wheels: the
complaint lives in what the goldfish cannot see (the table's removal, the politics) and in
the hand it is dealt (28% of keeps fail his rule). Forge is a later probe, on his word. No
handbook rebuild while the deck is iterating.

**Measuring finisher-v1 without the goldfish (2026-10-21, later).** The pilot: "fuck the
goldfish … we need a way to measure it, and goldfish isn't it". Three instruments, built
cheapest first: his MARKS (per-game counts logged after a game — when the commanders
landed, when the first wheel fired and onto what, what the kept seven held — on the deck
page and the branch page as "At the table"); scenario SLICES on the branch list against
Oliver's Araumi and Ryan's Teval (the finisher wheel with both commanders out vs one in
the zone; his one-creature-plus-wheels keep vs a rocks hand); and Forge COUNTS at
`olivers-table` (board series reads both partners now; a `wheel_timing` block; axes for
commander uptime, wheels fired with a commander out, first wheel turn, mulligans; never a
win-rate verdict). The seven-swap package stands in words, unstaged, until he calls the
slots; the Forge runs wait on his word per run.

**The finisher-v1 A/B (launched 2026-10-21, on his word).** "We need a true A/B test of the
previous deck against this deck … enough games to get a statistically significant delta …
controlling for how the AI pilots it." `experiment sharknado --a working --b @finisher-v1
--pod olivers-table --games 300 --looks 4 --detect 0.10` (O'Brien-Fleming at 75/150/225/300
games per arm, ~14 h; 81% power for +0.10 on a 0.20 baseline; smaller is INCONCLUSIVE, not
"no effect"). Both arms on ONE harness: every wheel in either list holds for a commander,
every rock in either list is cast in Main 1, all eleven adds PLAYED in a shell. It ran
`--anyway` against an audit one game short on commander access (15/20); arm A's 300 games
write the table's dynamic audit afterwards. PRE-REGISTERED before any look: the PRIMARY is
win rate (the design's alpha); the PROFILE is read whole, Holm-corrected, never a gate —
faster (commander resolved rate, first attack turn, first wheel turn), stronger (life
removed per game, biggest hit), answers (`interaction_cast`, and `sim/answers.py`'s answers
/ counterspells / protection cast, wired in after the run and re-derived with `--analyze`),
resilience (commander uptime, the turn eliminated, `interaction_received` beside it),
consistency (mulligans, empty-hand turns, extra draw per turn), and the branch's own
mechanism (wheels with a commander out, dead wheels never fired). An A/A at 300 is the
rule's price for grading a Forge figure and follows on his word.

**The finisher-v1 result (2026-10-22).** 300/arm, complete: win rate 45/299 -> 33/297, delta
-0.039 [-0.094, +0.015]; under Holm across 21 Forge rows the branch draws less (extra draw per
turn 2.08 -> 1.69) and attacks sooner (first attack ~2 global turns earlier). It traded the
draw/discard/burn engine for a smoother start, and the table did not pay it back.

**The voltron-v1 A/B (launched 2026-10-22, on his word).** "I want the main win cons in this
deck to be 21 commander damage." `experiment sharknado --a working --b @voltron-v1 --pod
olivers-table --games 300 --looks 4 --detect 0.10`, both arms replayed under ovdf27290367
(Embercleave unflagged), so last night's arm A is NOT this one's control. Static seat audit
PILOTABLE under that harness; arm A's logs write the dynamic half afterwards. All 29 adds
proven (28 PLAYED, Loran's Escape CAST-LATE). PRE-REGISTERED before any look: the design's
stopping rule is on win rate (the experiment's one primary); the BRANCH OBJECTIVE, graded by
net-change, is `forge.commander_damage_max_on_one_defender >= 10` (deck 5.69 under the old
harness); beside it, games reaching 21 commander damage, and the same Holm-corrected profile
as finisher-v1 (answers, interaction, resilience, consistency, wheels with a commander out).

**The pilot's calls (2026-10-19):** print the sharknado torbran-v1 proxies, playtest, and
possibly order the cards for the staged branch; sleeve goblin-storm's v1.1.0 cards as they
arrive and play it; then Edgar gets the next round of attention (mean-v1, discussed slot by
slot first). Log every game with `--feel`: the survey and the table record are the two
components withheld on every deck, and the rating cannot move until they land.

## Next, in order

0. **The fleet's own next moves** (above): sharknado torbran-v1 at the table and, on the
   pilot's word, `net-change` then merge; goblin-storm `deck-version paper` when all
   thirteen cards are sleeved, then `/analyze-engine` and pilot-notes to clear its stale
   `engine.json` / `tutor_guide.json` (the ten red regression tests on main since the v1.1.0
   merge are these); Edgar's mean-v1 discussion. `/rate-fleet` again once games are logged.
1. **Finish the context reviews** (the PRD's Step 4). Verdicts go through a review page; a
   context marked for changes goes back to the Context Keeper.
2. **The deletions** (Step 6): the engine model, the handbook and its agents, the
   stack-resolution chain that fed them. The carve-outs are mapped: `info.json` and
   `deck_info.compose` stay, `validate_citations` moves out of `validate_stack`,
   `sim/forge.py` stays.
3. **Response-time logging for every agent.** `manamap pilot sla-report` has one recorded
   agent run; the PRD's "90% within target" cannot be measured until every agent logs.
4. **A corpus refresh** (`/refresh-corpus`): Scryfall has moved past the tracked corpus
   (Reality Fracture released 2026-10-02), which is what the weekly corpus job reports.
   A model change follows it, so `regen` and `make manuals` too — on the pilot's word.
5. **The PRD's open questions**: the context length limit before pruning, whether a pilot
   log starts the incubation pod or Jarvis asks, how the queue is prioritised, and the
   final per-agent targets.

Still open from 2026-10-05 (`docs/history/plan-to-2026-10-05.md` has the detail):
edgar-vampires' diagnosis re-run after the static-lord fix, sharknado's Forge piloting,
the hoard_6/hoard_10 axis overlap (`docs/known-issues.md` 9b), a goldfish switch for
goblin-storm's ritual-into-commander, and edgar-vampires' Treasure flag.

## Decisions that bind

- **The frontend stays LLM-free**; the pipeline and pilot commands make zero LLM calls.
- **The goldfish decides, paired; Forge probes.** No Forge rate is graded without an A/A
  at the same N.
- **Never cut Vish Kal**, and nothing on a deck's keep list. Read every OUT list before any
  compute.
- **A flat +1/+1 anthem is too slow**; a card earns its slot by multiplying or scaling. No
  filler slots.
- **Versions come from git; tags are authored.** The paper lock is the one claim about
  cardboard.
- **Similarity comes from the function space; synergy is complementary, not similar.**
- **A retired deck is not a downstream target.**
