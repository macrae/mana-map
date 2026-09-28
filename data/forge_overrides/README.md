# Forge card-script overrides — THE INSTRUMENT, CHANGED ON PURPOSE

Forge's AI implements every card here correctly and **chooses its targets with a
generic heuristic**. On a deck whose whole engine is "target your own commander",
that heuristic is the deck: measured over 100 games on `copy-burst-v1`, the AI cast
the four token-copy spells **84 times while Zada's ability triggered 39 times in
total**, so most casts made ONE token instead of N. The cards are fine; the aim is
not. Forge's own docs say so — *"pretty bad for most combo decks"*, and
*"sometimes there is hardcoded logic for single cards"*.

Forge card scripts expose `AITgts$`, which narrows what **the AI** may target while
leaving legal targets untouched — `apprentice_necromancer.txt` ships
`ValidTgts$ Creature.YouOwn | AITgts$ Card.cmcGE5`. These overrides add one clause
to the spells Zada copies:

    AITgts$ Ally.YouCtrl

`Ally` names Zada because her Forge script is `Types:Legendary Creature Goblin Ally`
and no other Ally is in the 99. **The selector syntax cost two failed runs**, and both
failures are the same lesson: a target class is read, not guessed.

- `Creature.YouCtrl+namedZada, Hedron Grinder` — the **comma separates alternatives**,
  as `agency_outfitter.txt` shows (`namedMagnifying Glass,Card.namedThinking Cap`), so
  this parsed as `namedZada` OR ` Hedron Grinder` and matched nothing.
- `Creature.YouCtrl+Ally` — a **subtype is the HEAD of a target class**, never a `+`
  clause. Across the corpus the head carries the subtype (`Zombie` 8 uses, `Goblin` 4,
  `Ally` 2) while the `+` position takes only `YouCtrl` (132), `Other` (68), `YouOwn`
  (44), `OppCtrl` (30) and colours. `+Ally` is not a clause.

## ELEVEN SPELLS, NOT FIFTEEN — WHAT CANNOT BE STEERED

The deck's four token-copy spells — **Molten Duplication, Heat Shimmer,
Electroduplicate, Kindle the Inner Flame** — are `SP$ CopyPermanent`, and
`forge.ai.ability.CopyPermanentAi` **never reads `AITgts`**. That is from the 2.0.14
bytecode, not from a failed run: the class references `AILogic` and the sixteen logic
names it implements (`DuplicatePerms`, `MimicVat`, `Saheeli`, …), and `AITgts` is not
among its strings. The corpus agrees — `AITgts$` appears beside `Destroy` 22 times,
`ChangeZone` 12, `Pump` 7, and `CopyPermanent` **zero**.

So the generator REFUSES to write a script for them rather than shipping a hint the
engine cannot act on: a flag nothing reads is the failure `tests/test_metric_hygiene.py`
exists for. Those four cards are priceable in the goldfish only, where
`spell_token_copy` models them, and the Zada-targeting CEILING on them is not
measurable in Forge at all without patching Forge itself.

## THE COST, STATED BEFORE THE FIRST RUN

**A run made with these loaded does NOT measure Forge. It measures Forge plus this
directory.** That is legitimate only while it is declared, so:

- every file here is TRACKED and reviewable — nothing lives only in Forge's user dir
- `sim/forge.py` fingerprints this directory into every run record
  (`overrides.sha` / `overrides.files`), so a record made with overrides can never be
  mistaken for one made without, and a comparison across the boundary is refused
- the 11 spells are OURS. Verified before the first run: **none of the 14 seats under
  `data/opponents/` runs any of them**, so the pod is unaffected and remains a control
- `ValidTgts$` is never touched, so nothing here changes what is LEGAL — only what
  the AI will choose. A human pilot's options are identical.

## WHAT IT DOES NOT FIX

The AI still decides *whether* to cast, *when*, and what to attack with. This narrows
one decision, on eleven of the fifteen spells that wanted it — and not on the four the
engine cannot be told about (above). A rate from an overridden run is still Forge's AI flying the deck — just
no longer throwing the deck's best spells at the wrong creature.
