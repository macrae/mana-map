# Preview cards — a set Forge does not ship yet

Reality Fracture releases 2026-10-02. The installed Forge is 2.0.14, built 2026-08-08: its
`editions/Reality Fracture.txt` lists the 43 cards previewed by then and ships a script for
none of the ones the pilot wants. The corpus is a dated Scryfall dump, so it has them
neither — `/refresh-corpus` after release fixes that half.

This directory is the other half: a hand-written Forge card script per preview card, so a
card that hits the pilot's feed can be CAST-CHECKED and measured before the engine catches
up. Each script is derived from Forge's own templates (named in the commit that adds it) and
is a CLAIM about the card that the gate then tests: `forge-cast-check <slug> --card "<name>"`
says whether the AI plays it, and the usual rules-citation loop is what would prove it
resolves correctly.

## What is here

- `cards/edgar_ancient_bloodlord.txt` — {W}{B} 2/3 Legendary Vampire Noble. Death trigger
  templated on `cruel_celebrant` (ChangesZone / Creature.Other+YouCtrl), the sacrifice
  ability on `bartolome_del_presidio`'s two-type Sac cost plus `indulgent_aristocrat`'s
  `AILogic$ AristocratCounters` and `AIPreference:SacCost`, and the menace grant as a
  `Pump` sub-ability. No new token, so this one is complete.

## Not here yet, and why

- **Ingris Stingerquill** ({B}{R}{R} 1/4 Elder Sphinx) needs a token Forge does not have: a
  2/2 **colourless, non-artifact** Wizard Soldier named Cadet. Token scripts live in the
  ENGINE's `res/tokenscripts/`, not in a card override, so the preview lane needs a second
  install step for them — `Riptide Replicator`'s `TokenScript$ base + TokenTypes$/TokenPower$`
  overrides is the alternative and every colourless 2/2 base in the engine is an ARTIFACT
  creature, which would change the card. Designing that install step is the open work; the
  card's other two abilities are ordinary (`Attacks` trigger -> `DealDamage | Defined$
  Player.Opponent`, templated on `hellrider`/`agate_instigator`; the haste grant on
  `fires_of_yavimaya`).

## THE INSTALL IS A HARNESS CHANGE

A preview script is a card script, so installing one moves `card_overrides.sha` and
re-buckets every Forge comparison (`net_change.forge` keys on it). Nothing here is installed
while a branch arm is pending: the pin is the whole point. Install, then re-baseline.
