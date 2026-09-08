# Lines this deck can no longer play

Five checker-passed stack resolutions, retired 2026-09-08 when the pilot's paper
check-in and the branches that followed removed the cards they run on.

**They are not wrong.** Every one passed the citation contract — each step cites
a real Comprehensive Rules number and quotes it verbatim — and each was verified
against the deck as it stood. What changed is the deck.

| stack | needs, and no longer has |
|---|---|
| 001 Hullbreaker Horror + Sol Ring | Hullbreaker Horror |
| 003 Forced Fruition feeds the punishers | Iron Maiden, Ebony Owl Netsuke |
| 004 Hullbreaker three-card engine | Hullbreaker Horror |
| 005 Displacer Kitten, mana-positive | Aetherflux Reservoir, Displacer Kitten, Hullbreaker Horror |
| 006 Aetherflux storm lethal | Aetherflux Reservoir, Grand Abolisher, Hullbreaker Horror |

Four of the five are the Aetherflux Reservoir kill, which the pilot removed
deliberately — they hold a standing constraint against two-card infinite combos.
003 is the hand-size punisher line, cut after 116 castings across 120 Forge
games produced a MEDIAN of zero noncombat damage.

## Why moved rather than deleted

`build_index` and the handbook glob `stacks/*.json`, which is not recursive, so
a subdirectory takes them out of the live set without destroying the work. They
would otherwise have been PUBLISHED: `validate_stack` checks the citation
contract and reports a card outside the 99 as a WARNING, not an error, so all
six passed their gate while five described lines nobody could execute. A
handbook is read at a table under pressure; a rules-verified line you cannot
play is worse there than no line at all.

`002-approach-second-sun-tutor-win.json` STAYS. Its board state names Ancient
Tomb, which is no longer in the 99 — that is the same warning — but the line
itself is the deck's actual win condition (cast Approach, tutor it back, cast it
again) and it is live. It is left exactly as its author wrote it; the honest
repair is a re-spawn, not an edit under someone else's byline.
