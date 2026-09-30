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

## `unflag.txt` — THE CARDS THE AI WOULD NEVER CAST (2026-09-30)

`AI:RemoveDeck:All` on a Forge script is read at play time: the AI's candidate filter
drops every ability of such a card, so it is never cast proactively. Measured (Vish Kal
castable in 7 of 12 games, cast in 0; the same script minus that line, cast 10) and read
from the 2.0.14 bytecode — `docs/gotchas-bench.md` has the record. `unflag.txt` lists,
one script stem per line, every such non-land card a live deck runs; each one's override
is the shipped script with that single line removed, layered on an `AITgts$` override
where a card has both. `manamap pilot forge-install --generate` regenerates the set and
prints any flagged card a live deck has since acquired; a fleet test refuses one. These
overrides change the fingerprint like any other: a run under them is `-ov<sha>` and is
never pooled with a run made without.

## `forge_hints.json` — PER-DECK CARD HINTS (2026-09-30)

A deck may declare, in `data/decks/<slug>/forge_hints.json`, the two hint kinds Forge's
own aristocrat scripts use: `ai_logic` (appended as `| AILogic$ …` to the one ability
line whose API prefix the hint names — `carrion_feeder.txt` puts `AristocratCounters` on
its sacrifice ability) and `ai_preference` (`SVar:AIPreference:<kind>$<selector>`, placed
before the Oracle line — `SacCost$Creature.token,Creature.Other+cmcLE2` says what the AI
may feed the cost). Each hint says `why` and what it `cites`. `validate-forge-hints`
refuses a card not in the 99, a hint with nothing to hint, or an ability prefix that
matches zero or several lines; `forge-install --generate` derives the override onto the
shipped script (or its unflag override) and refuses a line that already carries a
different `AILogic$`. Idempotent over its own output. These land in the same fingerprint
as everything else. Measured first on Edgar: under the unflagged engine Vish Kal, Viscera
Seer and Altar of Dementia were cast and then sat 9, 5 and 17 own turns on the battlefield
with zero activations — the shipped scripts give the AI no logic and no preference for the
sacrifice cost. The knob half lives in `pilot_policy.json` (`SACRIFICE_DEFAULT_PREF_ENABLE`,
off in `Default.ai`), as a rule with a `forge` verb that explains the key it sets.

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
  (`card_overrides.sha` / `.cards` / `.n`), so a record made with overrides can never be
  mistaken for one made without, and `net_change.forge` **buckets runs by table AND
  fingerprint**, so the two never pool and a held-out harness is named with the command
  that would make it comparable

  **This bullet was ASPIRATIONAL when first written and it cost a wrong number the same
  day.** The fingerprint shipped; nothing read it. `net-change` pooled the override run
  with the plain one and printed `branch 10/155 (0.065)` for a list that is **1/73 =
  0.014 unpiloted and 9/82 = 0.110 piloted** — a rate describing neither deck, sitting
  under a README that claimed the comparison was refused. It is the same defect as a
  Forge record describing a list it never played, one layer over, shipped hours after
  that one was fixed. A written guard is not a guard; the gate is
  `test_forge_never_pools_runs_made_under_different_card_overrides`, proven by
  re-introducing the pooling
- the 11 spells are OURS. Verified before the first run: **none of the 14 seats under
  `data/opponents/` runs any of them**, so the pod is unaffected and remains a control
- `ValidTgts$` is never touched, so nothing here changes what is LEGAL — only what
  the AI will choose. A human pilot's options are identical.

## WHAT IT DOES NOT FIX

The AI still decides *whether* to cast, *when*, and what to attack with. This narrows
one decision, on eleven of the fifteen spells that wanted it — and not on the four the
engine cannot be told about (above). A rate from an overridden run is still Forge's AI flying the deck — just
no longer throwing the deck's best spells at the wrong creature.
