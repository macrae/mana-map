# goblin-storm

**Published as Vol. 001** — the hand-built reference deck, and the worked example for
every per-deck artifact in the subsystem.
<https://macrae.github.io/mana-map/manuals/goblin-storm.html>

Commander: Zada, Hedron Grinder. Five verified stacks, two decision spreads, a full
15-department issue.

It predates the deck builder, so it has no `brief.json` / `build_plan.json` — the
`candidate-pool` and `deck-build` routines report `N/A` for it, which is correct rather
than a gap.

Regenerate the derived artifacts:

```
manamap pilot fetch-deck goblin-storm      # short-circuits unless decklist.txt changed
manamap pilot validate-deck goblin-storm
manamap pilot goldfish goblin-storm
manamap pilot build-manual goblin-storm && manamap pilot build-index
```

`decklist.txt` is a Moxfield export — `1 Card Name (SET) 123 *F*` lines carry the exact
printing, which is what puts the pilot's own Secret Lair art in the issue. Also accepted:
`1x Card Name`, bare names, a `Commander:` header or trailing `*CMDR*`, `#`/`//` comments.
