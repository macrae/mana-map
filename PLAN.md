# PLAN — current state and what's next

*The resume-here doc. `docs/vision.md` says what the bench is; `CLAUDE.md` carries the
rules; this says where things stand and what is open. Earlier plans are kept whole:
[`docs/history/plan-to-2026-11-23.md`](docs/history/plan-to-2026-11-23.md) (the October
ratings, Sharknado's Forge A/Bs, PRD v2's reviews and deletions) and
[`docs/history/plan-to-2026-10-05.md`](docs/history/plan-to-2026-10-05.md).*

Last updated **2026-11-23**. Do not quote a figure from here without re-running the
command beside it.

## Where things stand

**The decision is a study; Jarvis is the entry point (2026-11-21).** Every call is a
tracked study (`manamap pilot study`, `docs/study.md`): the facts from every source, the
agents' answers kept verbatim, the call in the pilot's words and who decided. The goldfish
is one labelled instrument, never the sole decider and never one on a card it cannot read;
`propose` requires a study. Forge is a probe, never a gate.

**The engine is a computed graph (2026-11-22).** `engine-graph` builds `engine.json` v2
from Forge's card scripts (typed events, components, lines as event shapes, single points
of failure); agents write only the `reading` over it (`/engine-reading`). The interaction
space and `similar-cards --space/--feeds` exist; the default space did not move because
the evaluation did not say it wins (`data/eval/interaction_eval.json`). Memo:
`docs/research/engine-model.md`.

**One place to read a deck (2026-11-22).** The deck page is five chapters, one screen
open; the rendered handbook (`manuals/p/`) is retired and its data kept. The bench keeps
no notion of card ownership: a branch is a list and a study, merged on the pilot's word.

**The test suite is pruned and scoped (2026-11-22).** A `core` of must-never-break
contracts (`make smoke`), impact-selected unit tests per commit, prepush by area, the
full run nightly in CI (`docs/testing.md`).

**The fleet.** Sleeved: edgar-vampires, gishath, goblin-storm, heliod, keen-engineering,
sharknado, ur-dragon (`deck-version <slug>`). On the bench: meren-recursion (V30, nine
studies: the yard-fill pass), elves, emiel-blink; the teach-* decks are brewing. Open
branches, each waiting on the pilot's call: edgar-vampires `threat-v1`, heliod `b3-v1`,
sharknado `jaws-v1` (`deck-branch <slug> list`).

## Next, in order

1. **The open branches.** edgar threat-v1, heliod b3-v1 (a stack scenario is owed after
   its merge), sharknado jaws-v1 — each merges on the pilot's word via `/release-branch`,
   then the DeckCheck paste (his) and `/rate-deck`.
2. **The goldfish register (the Jarvis refactor's last commit).** The goldfish's blind
   spots as a register, not a footnote: `model_coverage` gives every printed figure its
   asterisk from one place, and the channel registry is the single list `try`,
   `net-change` and the study read.
3. **The engine model, Phase 3's remainder.** The deck map clustered on `feeds`, an
   EDHREC-lift column, the permutation null for the engine set; and, deck by deck on the
   pilot's word, switching a declaration to `targets_from: engine.json`.
4. **Opponents' libraries in the goldfish — Triple Triad and Etali (2026-11-23).** The
   goldfish cannot read any card that exiles or casts from OPPONENTS' libraries: Triple
   Triad (your upkeep: each player exiles their top card; you play yours free and any
   opponent's card of lesser mana value free) is INVISIBLE to `try` on Gishath, and
   Gishath's own Etali, Primal Storm is the same shape. Three pieces, cheapest first:
   (a) our half — exile our top card at upkeep and play it through cascade's and
   discover's free-cast door, a land through the land drop; a floor, about a day;
   (b) opposing libraries — deal three seat decks from `data/opponents/` on the same
   paired seeds and flip their tops; (c) what a stolen card is worth — first an honest
   count (free spells stolen, mana value stolen by turn eight, with intervals), then,
   larger, the stolen card through the normal dispatcher, crediting only what the model
   reads and naming the rest as dark, never zero. Still blind to blockers, removal and
   how a table plays around it. `forge-cast-check gishath --card "Triple Triad"` answers
   "does the AI play it" today.
5. **Self-mill in the goldfish, for Meren (the pilot, 2026-11-24: "teach the goldfish
   self-mill (add in plan), not yet").** The goldfish has no self-mill reader: Satyr
   Wayfinder, Stitcher's Supplier, Six's mill, Nyx Weaver and Primal Witchstalker read as
   their bodies alone (study 013), so a deck built to fill its graveyard is measured as if
   it did not, and Meren's end-step return draws from a yard the model under-fills. Teach
   it as the Edgar passes were taught: milled creature cards join `yard_creatures` (Meren's
   and the reanimation spells' targets), milled lands are landfall fuel and Witchstalker's
   land-back target, a card's mill read from its text with a corpus sweep, and a fill row
   (creature cards in the graveyard by T4) for net-change; every reading's test proved red
   with its bug put back. (`study install` refreshing the Context and the stamps is DONE,
   2026-11-23/24.)
6. **The goldfish test refactor** (deferred from the 2026-11-22 test overhaul, by the
   pilot's call). The goldfish's tests are about 43 files named by date or by card
   (`test_goldfish_audit_2026_11.py`, `test_pilot_goldfish_gishath.py`, …) rather than by
   mechanic. Reorganise them by what the simulator does, for example mana, draw, combat,
   deaths, tokens and discard. Fold the 94 audit tests into those files and share one
   stacked-library helper. Merge overlapping draw coverage (`test_goldfish_cast_draw`,
   `_mana_into_cards`, `_drain_and_draw`) only after reading the three side by side.
   Rules: no test deleted unless a kept test guards the same behaviour; the core entries
   in `tests/core_contracts.py` are repointed in the same commit; `make smoke` and the
   unit tier stay green at every step. There is no speed gain. The point is that a
   mechanic's tests can be found in one place. Also worth a pass: `goldfish_turn.py` is
   the largest simulator module (5.7k lines) at about 67% unit coverage.
7. **Agent response times.** `manamap pilot sla-report` now logs every agent; the
   data-analyst misses its 30 s target on nearly every run and card-scout on a third —
   trim their briefs or move the targets. The job band's installed version is stale
   (the report says how to update it).

**Deferred by the pilot:** a corpus refresh (2026-11-02; the weekly corpus job's count
drift is expected until then). **Known, no test fails on it:** the hoard_6/hoard_10 axis
overlap (`docs/known-issues.md` 9b).

## Decisions that bind

- **The frontend stays LLM-free**; the pipeline and pilot commands make zero LLM calls.
- **The study decides; the goldfish reads, paired; Forge probes (2026-11-21).** A call names
  what decided it and the goldfish is never that alone. No Forge rate is graded without an
  A/A at the same N.
- **Only clear, obvious upgrades (2026-11-22).** A swap is pitched when it is clearly
  advantageous; "modest" or "close to a tie" is NO CHANGE, and an open slot is fine.
- **Never cut a card on a deck's keep list** (Vish Kal first among them). Read every OUT
  list before any compute.
- **A flat +1/+1 anthem is too slow**; a card earns its slot by multiplying or scaling. No
  filler slots.
- **Versions come from git; tags are authored.** The paper lock is the one claim about
  cardboard; the bench tracks no ownership.
- **Lazy evaluation.** Deck-scoped regen only; a fleet regen or `make artifacts` on the
  pilot's word.
- **Similarity comes from the function space; synergy is complementary, not similar.**
- **A retired deck is not a downstream target.**
