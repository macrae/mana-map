# Ur-Dragon Treasure: an RGW shell under the five-colour commander

**Status:** a design shell. Nothing applied — no brief, no `paper` block, no version bump.
**Date:** 2026-08-26. Supersedes the Baylen draft of the same date.

---

## 0. A correction to `docs/ur-dragon-fork.md`

**That document conflated the commander's colour identity with the mana base's requirement.
They are different things.** Colour identity is a *deckbuilding permission*; it imposes no
mana cost of its own. The "132 source demand" figure was a property of the **old
Dragon-typal card list**, not of The Ur-Dragon.

The fork doc computed each branch by filtering the *existing* 63 cards by identity. That
correctly answers "what if I keep this list and cut colours". It is the wrong question when
the list is being rebuilt anyway.

Measured directly:

| | demand | colours per producer |
|---|---|---|
| The Ur-Dragon + an **RGW-only 99** | **80** | **1.7** |
| Any Naya commander + the same 99 | 80 | 1.7 |
| the old Dragon-typal list | 132 | 2.9 |

**Identical.** The mana base serves the cards, not the command zone. Keeping The Ur-Dragon
costs nothing, and the fork's conclusion — that branch A was unavailable — was wrong for a
rebuild. It stands only for the "keep this list" question it actually asked.

---

## 1. Why the five-colour commander is now an asset

1. **Eminence works from the command zone.** *"As long as The Ur-Dragon is in the command
   zone or on the battlefield, other Dragon spells you cast cost {1} less."* Thirteen
   Dragons in this shell, each a mana cheaper, **from turn zero, without ever casting him.**
2. **He is castable off two Treasures.** He costs `{4}{W}{U}{B}{R}{G}`. An RGW mana base
   already supplies W, R and G — only U and B need covering, and a Treasure taps for **any**
   colour. Commander's Sphere and Chromatic Lantern cover it too. **A treasure deck is the
   one archetype that can reliably cast a five-colour commander.**
3. **The attack trigger is the payoff, not the body.** Whenever one or more Dragons attack,
   draw that many cards and put a permanent from hand onto the battlefield.
4. **It stays `ur-dragon`.** A version bump, not a new slug — the captain's log, the version
   history and both logged games stay attached to the deck.

**Not a Dragon-typal deck.** The typal payload is cut (§4). Dragons are payoffs the engine
buys; eminence is a discount you get for free rather than a strategy you build around.

**Verified while researching:** there is **no Naya legendary Dragon with treasure
generation.** Of eight legendary Dragons that mention Treasure, all are mono-G, mono-R,
mono-B or B/R. The card being looked for does not exist, and The Ur-Dragon is the closest
thing to it that does.

---

## 2. The engine

**24 treasure generators.** ≥1 in the opening seven **86.7%**; ≥1 by turn three **92.7%**.

(For contrast: ≥2 in the opening hand at 80% would take **37 cards** — 37% of the deck. That
half of the goal is a different deck, not a tuning target.)

**Non-creature by design.** The pod wipes constantly, so the engine is built where a wrath
cannot reach: enchantments (Smothering Tithe, Smuggler's Share, Monologue Tax), artifacts
(Academy Manufactor, Treasure Vault, Monument to Endurance, Collector's Vault) and lands
(Fountainport, Mines of Moria). Twelve of the top twenty wipe-proof treasure generators in
these colours are **colourless**, which is also why the mana stays cheap.

**Academy Manufactor is the keystone** — every Treasure becomes Treasure + Clue + Food, and
the Clue draws a card by itself. Xorn and Jolene each add a Treasure to every Treasure;
Doubling Season and Parallel Lives double all of it.

**The engine is also the clock.** Reckless Fireweaver and Weftstalker Ardent turn every
artifact entering into damage to each opponent. Treasure production kills, with no combat
step and nothing for a wrath to remove. **Hellkite Tyrant** wins outright at twenty
artifacts and is already in the deck.

---

## 3. The list — 63 spells + 36 lands + The Ur-Dragon

`[BUY]` = to acquire · `[box]` = loose in the collection · `[in <deck>]` = sleeved elsewhere
· unmarked = already in the current Ur-Dragon list.

**TREASURE GENERATORS (26)** — Academy Manufactor [BUY] · Ancient Copper Dragon [BUY] · Atsushi, the Blazing Sky · Big Score [box] · Bootleggers' Stash [BUY] · Captain Lannery Storm [BUY] · Collector's Vault [BUY] · Currency Converter [BUY] · Fountainport [in goblin-storm] · Gadrak, the Crown-Scourge [BUY] · Goldspan Dragon · Mines of Moria [BUY] · Monologue Tax [BUY] · Monument to Endurance [BUY] · Old Gnawbone · Professional Face-Breaker [BUY] · Ragavan, Nimble Pilferer [BUY] · Rapacious Dragon · Smaug the Magnificent · Smothering Tithe · Smuggler's Share [BUY] · The Reaver Cleaver [BUY] · Tireless Provisioner [box] · Treasure Map // Treasure Cove [BUY] · Treasure Vault [BUY] · Unexpected Windfall [BUY]

**MULTIPLIERS (5)** — Doubling Season [box] · Jolene, the Plunder Queen [BUY] · Parallel Lives [BUY] · Peregrin Took [BUY] · Xorn [BUY]

**NON-COMBAT DAMAGE (5)** — Blasphemous Act [in gishath] · Drakuseth, Maw of Flames · Jaya's Immolating Inferno [BUY] · Reckless Fireweaver [BUY] · Weftstalker Ardent [BUY]

**DRAGON PAYOFFS (5)** — Glorybringer · Hellkite Tyrant · Lathliss, Dragon Queen · Terror of the Peaks · Utvara Hellkite

**DRAW (6)** — Cleansing Wildfire [BUY] · Commander's Sphere [BUY] · Garruk's Uprising · Idol of Oblivion [in goblin-storm] · Return of the Wildspeaker · Skullclamp [box]

**INTERACTION (8)** — Beast Within [box] · Boros Charm · Flawless Maneuver · Generous Gift · Heroic Intervention · Swords to Plowshares · Teferi's Protection · Vandalblast [in goblin-storm]

**RAMP & FIXING (12)** — Arcane Signet · Chaos Warp · Chromatic Lantern · Crackle with Power [BUY] · Deflecting Swat · Enlightened Tutor · Farseek · Mox Jasper · Nature's Lore · Panharmonicon · Skyshroud Claim · Sol Ring

---

## 4. What comes out

Thirty-two cards leave, and they are the typal payload plus the blue/black splashes:

**The Dragon-typal engine** — Dragonlord's Servant · Dragonspeaker Shaman · Urza's Incubator
· Herald's Horn · Sarkhan's Triumph · Encroaching Dragonstorm · Breaching Dragonstorm ·
Desolation of Smaug · Dragon Tempest · Radiant Destiny · Sarkhan, Fireblood · Sneak Attack

**Blue and black** — Counterspell · Swan Song · Dovin's Veto · Roiling Dragonstorm ·
Korlessa, Scale Singer · Miirym, Sentinel Wyrm · Temur Ascendancy · Crux of Fate

**Surplus Dragons** — Ancient Gold Dragon · Atarka, World Render · Hellkite Courser ·
Moltensteel Dragon · Scourge of Valkas · Scourge of the Throne · Thrakkus the Butcher ·
Wrathful Red Dragon

**Other** — Dragon's Hoard · Relic of Legends · Rishkar's Expertise · Worldly Tutor

**Crux of Fate is the only real loss** and it is the deck's one black card. Blasphemous Act
replaces it as an asymmetric wipe the treasure engine survives.

---

## 5. Cost

| | count |
|---|---|
| already in the deck | 31 |
| loose in a box | 5 |
| sleeved in **goblin-storm**, which is locked | 3 |
| to buy | 28 |

**54% reuse.** The three blocked cards (Fountainport, Idol of Oblivion, Vandalblast) sit in
the one deck that is finished — buy or cut them; do not unsleeve goblin-storm.

**No dollar figure.** Prices are stripped from the corpus. **Ragavan, Nimble Pilferer is the
line item most likely to dominate the total and the deck works without him** — price it
before ordering.

---

## 6. Open questions

1. **Is white worth its nine cards?** Smothering Tithe, Smuggler's Share and Monologue Tax
   are wipe-proof enchantments that tax opponents; Swords, Teferi's Protection, Generous
   Gift and Flawless Maneuver are the interaction. Cutting to Gruul takes demand 80 → 58.
2. **Five Dragon payoffs, or fewer?** The engine wins without them; they are what eminence
   is for.
3. **Does the deck reach twenty artifacts** often enough for Hellkite Tyrant to be a real
   second win, or is it a trap? Goldfish it.
4. **How often is The Ur-Dragon actually cast?** The claim is "two Treasures cover U and B".
   That is a hypergeometric question nobody has run.

---

## 7. Not done here

No `brief.json`, no version bump, no `paper` block. Next step if the shell survives: a brief,
the deterministic builder, `mana-analysis` on the real list, and a goldfish against §6.3–6.4.
