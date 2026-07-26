# hapatra

**Built by the deck builder, not by hand.** There was never a decklist to wait for —
`brief.json` names the commander and a target bracket, and the builder produced the 99.

- Commander: Hapatra, Vizier of Poisons (BG)
- Target bracket 3 (Upgraded); computed floor 3
- Archetype: −1/−1 counter aristocrats — every counter placement is a Snake, and the
  Snakes convert to cards and drain

Regenerate or revise:

```
manamap pilot build-deck hapatra --write-decklist   # deterministic, no agents
manamap pilot validate-build hapatra
manamap pilot bracket-check hapatra --target 3
manamap pilot fetch-deck hapatra
manamap pilot validate-deck hapatra
manamap pilot goldfish hapatra
```

Run the `/build-deck` skill for the agent loop (deck-analyst → deck-architect ⇄
deck-critic) on top of the deterministic baseline — check `cache-status` first, both
build routines are recorded.

**Not yet published.** To turn this into a magazine issue it needs `issue.json`
authored, then at least one checker-passed stack, then `/write-manual` and
`/design-issue`. The first stack is already specified in `build_plan.json`'s `gaps`:
whether counters arriving via persist count as counters *you* put on a creature
(CR 122.6 settles that entering counters count; it does not settle the attribution).
