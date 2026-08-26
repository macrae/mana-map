# Naya Treasure: a shell

**Status:** a design shell for discussion. No deck directory, no `brief.json`, nothing applied.
**Date:** 2026-08-26. **Supersedes nothing** — this is branch B3 of `docs/ur-dragon-fork.md`.

The axis: **treasure is the engine, not the ramp.** Big creatures stay as payoffs; they stop
being the plan. No creature typal.

---

## 1. The commander, and why it is not a compromise

**Baylen, the Haymaker** — `{R}{G}{W}`, mv3, Rabbit Warrior, non-typal.

> Tap two untapped tokens you control: Add one mana of any color.
> Tap three untapped tokens: **Draw a card.**
> Tap four untapped tokens: Put three +1/+1 counters on Baylen. Trample until end of turn.

Three things make him the right answer rather than the available one:

1. **Treasures are tokens, and tapping is not sacrificing.** Every other treasure payoff eats
   its own fuel. Baylen taps them, they untap next turn, and they do it again. That is the
   difference between ramp and an engine.
2. **He is the card-draw requirement, on the commander.** §8 of the debrief asks that treasure
   convert into cards rather than mana. Three untapped tokens draw a card, every turn, at no
   cost and with no combat.
3. **mv3, and he is not a Dragon.** He resolves on turn three into a wipe-heavy table and
   costs almost nothing to redeploy.

**The tension, named:** Baylen wants token *width* (three or four untapped), and treasure
arrives one or two at a time. This is the design problem of the deck, and it is why the
multiplier package below is not optional.

**Also true:** `baylen-tokens` is already a modelled seat in your pod — an EDHREC average
list, which is the *creature*-token build. Same commander, different axis. Worth knowing
before you sit down with it.

---

## 2. The engine

**23 treasure generators.** Against a 99-card library that is:

| | probability |
|---|---|
| ≥1 in the opening seven | **85.3%** |
| ≥1 by turn three | **91.8%** |
| ≥2 by turn three | 66.7% |

That answers "every hand has a treasure thing" at the level the maths actually permits.
For contrast, **≥2 in the opening hand at 80% would take 37 cards** — 37% of the deck — so
that version of the goal is a different deck, not a tuning target.

**The multipliers are the fix for Baylen's width problem** — Academy Manufactor turns one
Treasure trigger into Treasure + Clue + Food, which is three untapped tokens from one
trigger, and the Clue draws. Xorn adds a Treasure to every Treasure. Doubling Season and
Parallel Lives double everything.

**The damage engine is the same engine.** Reckless Fireweaver and Weftstalker Ardent turn
every artifact entering into damage to each opponent. Treasure production *is* the clock,
with no combat step and nothing for a wrath to kill. That is the debrief's §5 meta thesis
built into the mana base rather than bolted on.

**Hellkite Tyrant is an on-axis alternate win** — twenty artifacts and you win at upkeep —
and it is already in the Ur-Dragon list.

---

## 3. The list (62 spells + 37 lands + commander)

**Commander** — Baylen, the Haymaker

**Treasure generators (23)** — Academy Manufactor · Xorn · Smothering Tithe · Smuggler's
Share · Monologue Tax · Professional Face-Breaker · Ragavan, Nimble Pilferer · Captain
Lannery Storm · Jolene, the Plunder Queen · Gadrak, the Crown-Scourge · The Reaver Cleaver ·
Tireless Provisioner · Bootleggers' Stash · Old Gnawbone · Goldspan Dragon · Rapacious
Dragon · Ancient Copper Dragon · Atsushi, the Blazing Sky · Smaug the Magnificent · Monument
to Endurance · Collector's Vault · Treasure Map · Big Score · Unexpected Windfall

**Token multipliers (5)** — Doubling Season · Parallel Lives · Peregrin Took · Xorn ·
Academy Manufactor *(the last two double-counted above; they are generators and multipliers)*

**Non-combat damage (4)** — Reckless Fireweaver · Weftstalker Ardent · Jaya's Immolating
Inferno · Crackle with Power · *(plus Chandra's Ignition)*

**Payoffs (7)** — Hellkite Tyrant · Terror of the Peaks · Drakuseth, Maw of Flames · Utvara
Hellkite · Glorybringer · Lathliss, Dragon Queen · Atarka, World Render

**Draw (5)** — Dragon's Hoard · Garruk's Uprising · Return of the Wildspeaker · Rishkar's
Expertise · Idol of Oblivion

**Interaction (9)** — Swords to Plowshares · Teferi's Protection · Generous Gift · Heroic
Intervention · Flawless Maneuver · Boros Charm · Chaos Warp · Deflecting Swat · Beast Within
· Vandalblast

**Ramp & fixing (6)** — Sol Ring · Arcane Signet · Mox Jasper · Farseek · Nature's Lore ·
Skyshroud Claim · Panharmonicon

**Tutors (2)** — Enlightened Tutor · Worldly Tutor

---

## 4. What it measures

| | five-colour V2 | this shell |
|---|---|---|
| source demand | 132 | **80** |
| colours per producer | 2.9 | **1.7** |
| mean mana value | — | **3.48** |
| median mana value | — | **3.0** |
| curve | top-heavy | 1:6 · 2:10 · **3:20** · 4:9 · 5:7 · 6:5 · 7:3 · 8:1 |

Pip load: R 61.6% (30 cards) · G 26.0% (17) · W 12.3% (9). White is a genuine splash at
nine cards, which is the honest shape — see §6.

**37 land slots**, which is more than the current deck and now affordable, because demand
fell by 39%.

---

## 5. What it costs

| | count |
|---|---|
| already in Ur-Dragon — free when you break it down | **34** |
| in a box — free | 4 |
| sleeved in **goblin-storm**, which is locked | 2 |
| **to buy** | **24** (26 with the two above) |

**53% of the shell comes out of the deck it replaces.** The two blocked cards (Idol of
Oblivion, Vandalblast) are in the one deck §15 says is finished and not to be modified —
buy them or cut them, do not unsleeve goblin-storm.

**No dollar figure.** Prices are stripped from the corpus and this repo cannot compute one.
Ragavan, Nimble Pilferer is the line item most likely to dominate the total and should be
priced before anything is ordered; the deck functions without it.

---

## 6. Open questions

1. **Is white worth nine cards?** It carries Smothering Tithe (the best treasure generator
   in the format), Smuggler's Share, Monologue Tax — all wipe-proof enchantments that tax
   opponents — plus Swords, Teferi's Protection, Generous Gift, Flawless Maneuver. Cutting
   to Gruul drops demand from 80 to ~58. The case for white is interaction plus three
   enchantments, not treasure volume.
2. **Do the Dragons stay at seven?** They are payoffs now, not a tribe. Seven may still be
   too many for a deck whose engine wins without them.
3. **Anointed Procession and Mondrak are in Edgar**, which v1.1 also wants them for. Two
   doublers, two decks, one copy each.
4. **Does Baylen actually get to four tokens?** The multiplier package says yes; nothing has
   simulated it. This is the first thing to goldfish.

---

## 7. Not done here

No deck directory, no `brief.json`, no `paper` block — this deck does not exist in cardboard
and nothing should claim it does. The next step, if the shell survives discussion, is
`manamap pilot brew` with a proper brief, then the deterministic builder, then a goldfish
against the question in §6.4.
