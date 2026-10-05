# PLAN — current state and what's next

*The resume-here doc. `docs/vision.md` says what the bench is; `CLAUDE.md` carries the
rules; this says where things stand and what is open. The plan as it stood until
2026-10-05 — every dated block, and the pilot's PUNCHLIST of 2026-09-16 — is
[`docs/history/plan-to-2026-10-05.md`](docs/history/plan-to-2026-10-05.md).*

Last updated **2026-10-05**. Do not quote a figure from here without re-running the
command beside it.

## Where things stand

**The workbench changed its decision loop on 2026-10-04.** The pilot ruled overnight
Forge runs out of it — too slow, too brittle, an MDE of ~0.14 at 200 games. A swap is now
answered by `manamap pilot try <slug> --out A --in B` in about ten seconds: the goldfish
with a seed per game, the two lists aligned slot for slot, and a paired interval on the
difference. Forge is a probe (`forge-cast-check`), and a Forge loss in `net-change` is a
warning, never a gate. In the same two days the goldfish learned attack restrictions
(Defender, Kefnet), three seats for what we gain off opponents' draws, Smothering Tithe's
Treasure, and rituals. The pilot's keep list (`protected.json`) exists because a branch
cut Vish Kal unread; every cut path refuses a card it names.

**The fleet.** Six decks are sleeved — edgar-vampires, gishath, goblin-storm, heliod,
sharknado, ur-dragon (`deck-version <slug>`); emiel-blink and meren-recursion are on the
bench; six are retired or broken down. Nineteen games are logged across five decks.

**Proposed and waiting on cardboard:** sharknado `momentum-v1` as **v1.2.0** (wheels,
rituals, Teferi's Protection; `deck-branch sharknado list`).

**Diagnoses, re-run 2026-10-05.** goblin-storm's passed the skeptic at iteration 3.
edgar-vampires' is saved with a skeptic **FAIL** after three iterations (six findings
open), and its largest open finding is about the model, not the prose — see below.

**The docs and tests were overhauled 2026-10-05.** Two test tiers (`make test` ~3 min,
`make test-fleet` before a push; `docs/testing.md`), CLAUDE.md compacted to one line per
rule, module headers and file maps, three finished records archived.

## Next, in order

1. **Re-run edgar-vampires' diagnosis** now that the goldfish reads static lords (fixed
   2026-10-05, `docs/known-issues.md` §22; the file maps landed in the same model change).
   Its open findings rest on it: Legion Lieutenant -> Vampire Interloper reads −1.088
   damage@T10 where it read +0.022, so the lord cuts it proposed now cost something.
   That is in tension with the pilot's rule that a flat anthem is too slow — the model
   says the lord earns damage; whether it earns the SLOT is the pilot's call. Also open:
   the win-rate pool that mixed two AI profiles, and what fills a cut lord's slot.
2. **sharknado's piloting in Forge** — the AI mis-plays its wheels, so its Forge rates are
   floors. A piloting item (hints, `forge-cast-check`), not a deck change.
3. **The axis catalog** (`docs/known-issues.md` 9b): hoard_6 and hoard_10 are one
   measurement (r=+0.93). Which axis a branch may aim at is the pilot's call.
4. **Games at a table, logged.** All nineteen logged games predate this month's model work.

## Open questions held for the pilot

- goblin-storm: whether to add a goldfish switch for ritual-into-commander, so the ramp
  reading can be split (its diagnosis `open_questions[1]`).
- edgar-vampires' Treasure flag (`model_treasures`), and the input-versus-refuel reading
  the diagnosis leaves unranked.

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
