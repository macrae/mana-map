# Ur-Dragon v2: the treasure refactor

**Report only. The decklist is untouched.** `brief.json` is written; nothing else has moved.
**Measured:** 2026-08-26 against the tracked V2 list.

**34 out · 35 in · 29 kept.** Commander unchanged: The Ur-Dragon, for Eminence.

---

## 1. The axis

Treasure is the engine. Everything is bought with it: **cards**, **damage**, and **Dragons
at a discount Eminence grants free from the command zone**. Dragons are payoffs, not a tribe.

The design rule the whole list is filtered through: **a card that does two jobs beats a card
that does one.** Thirteen cards in the 63 do three or more.

| | current | refactor |
|---|---|---|
| treasure sources | 6 | **28** |
| wipe-proof treasure sources | 2 | **10** |
| cards doing ≥3 jobs | — | **13** |
| mean mana value | 3.63 | **3.38** |
| RGW source demand | 132 | **80** |
| alternate win conditions | 1 | **1** (plus a sink) |

**P(a treasure source in the opening seven) = 91.1%.** By turn three, **95.7%**.
**P(two by turn four) = 83.7%** — which is what makes the off-colour splashes castable.

---

## 2. The treasure engine — 28 sources, built where a wrath cannot reach

Ten are non-creature, because this pod wipes constantly and an engine on bodies resets to
zero. That is the debrief's §5 thesis as a construction rule.

**The multipliers (4).** Every generator is worth more than once:
**Academy Manufactor** (Treasure → Treasure + Clue + Food; the Clue draws by itself) ·
**Xorn** (a Treasure on every Treasure) · **Jolene, the Plunder Queen** (the same, on a body,
and she pays the table for attacking *each other*) · **Doubling Season** + **Parallel Lives**.

**The wipe-proof core (10).** Smothering Tithe · Smuggler's Share · Revel in Riches ·
Monument to Endurance · Collector's Vault · Treasure Map · Reckoner Bankbuster ·
Sword of Wealth and Power · The Reaver Cleaver · Bootleggers' Stash.

**The bodies that also make treasure (8).** Ragavan · Charming Scoundrel ·
Professional Face-Breaker · Tireless Provisioner · Xorn · Jolene · Gadrak · Academy Manufactor.

**Free slots in the land base.** Treasure-making *lands* cost no spell slot at all:
**Treasure Vault · Fountainport · Mines of Moria · Volatile Fault · The Gold Saucer**.
Volatile Fault is the standout — it taps for mana, kills an opponent's nonbasic, **and**
makes a Treasure, from a slot the deck was spending anyway.

---

## 3. Engine synergies — what actually chains

1. **Manufactor → Clue → card.** Every treasure trigger becomes a card. This is the deck's
   draw engine and it costs no dedicated slots.
2. **Any generator → Xorn/Jolene → doubled → Doubling Season → doubled again.** Four
   multipliers on 24 generators; the ceiling is not linear.
3. **Artifact enters → Reckless Fireweaver / Weftstalker Ardent → damage to each opponent.**
   The engine *is* the clock. No combat step, nothing for a wrath to kill. This is the single
   most important structural change in the refactor.
4. **Fable of the Mirror-Breaker III → Kiki-Jiki → copy Academy Manufactor** every turn.
5. **Panharmonicon** doubles every artifact- and creature-ETB treasure trigger.
6. **Opponents' wipes feed you.** Revel in Riches makes a Treasure whenever an *opponent's*
   creature dies. In a pod that wipes constantly, their removal is your ramp.
7. **Gadrak** does the same for *your* dead creatures — a Treasure per nontoken creature that
   died this turn, at end step.

---

## 4. Combos and alternate wins

**Zero contained two-card infinites, and that is deliberate.** The deck is a bracket-3 engine,
not a combo deck.

**ONE alternate win is keyed to the treasure engine, not three.** An earlier draft of this
report claimed three and that was wrong:

- **Revel in Riches** — 10 Treasures, win at your upkeep, and it *makes* them off opponents'
  creatures dying. This is the only win the engine builds toward on its own.
- **Jaya's Immolating Inferno** — X damage to *each* opponent. Not an alternate win: a
  **treasure sink** that converts the engine into lethal. Legendary sorcery, and the
  commander is legendary, so it is always castable.
- **Hellkite Tyrant** — **a combat-gated steal, and the win clause is a rider on it.** The
  trigger reads *"whenever THIS CREATURE deals combat damage to a player"*, so it needs a
  six-mana 6/5 to resolve, survive to your next turn on a wipe-heavy table, and connect.
  What it is really for is the theft: measured against the tracked pod, **abaddon runs 12
  artifacts, giada-angels 12, vito 11, baylen-tokens 7** — a connect takes ~7–12 permanents
  including their mana rocks, which is also what would carry you past twenty. Strong, and
  **off the deck's own non-combat axis**; keep it as one of twelve Dragons and a threat that
  demands an answer, not as a line the deck is building toward.

**Near-misses worth knowing** (one card away from a Spellbook line): **Aggravated Assault**
(4 lines — and it was cut from this deck in July), **Mechanized Production** (a second
treasure alt-win: enchant a Treasure, copy it each upkeep, win at eight same-named artifacts),
**Hellkite Charger**, **Time Sieve**.

---

## 5. Ramp

Deliberately thin and cheap, because **treasure is the ramp**. Sol Ring · Arcane Signet ·
Mox Jasper · Chromatic Lantern · Commander's Sphere · Farseek · Nature's Lore · Skyshroud Claim.

Chromatic Lantern and Commander's Sphere are doing double duty: they are the **fixers that
make The Ur-Dragon castable** without a five-colour mana base. Sphere taps for any colour in
the commander's identity — all five.

Cut from the old ramp: the entire Dragon **cost-reduction** package (Dragonlord's Servant,
Dragonspeaker Shaman, Urza's Incubator, Herald's Horn). Eminence already discounts Dragons,
from the command zone, for free — those four cards were paying twice for one effect.

---

## 6. Dragon payoffs — five, down from fourteen

**Hellkite Tyrant** (alt-win, steals artifacts) · **Terror of the Peaks** (damage on every
creature ETB — non-combat) · **Drakuseth, Maw of Flames** (non-combat damage on attack) ·
**Utvara Hellkite** (makes 6/6s) · **Ancient Copper Dragon** (treasure) ·
**Bonehoard Dracosaur** (a Dragon that draws two and makes a body *and* a treasure every upkeep).

All discounted by Eminence. None of them is the plan.

**Nine Dragons leave** because they were tribe rather than payoff: Ancient Gold Dragon, Atarka,
Glorybringer, Hellkite Courser, Lathliss, Moltensteel, Scourge of Valkas, Scourge of the
Throne, Thrakkus, Wrathful Red Dragon.

---

## 7. Other payoffs

**Non-combat damage** — Reckless Fireweaver · Weftstalker Ardent · Jaya's Immolating Inferno ·
Crackle with Power · Terror of the Peaks · Drakuseth. This is where the deck's damage now
comes from, and it is the direct answer to the debrief's §5.

**Card draw** — Academy Manufactor's Clues · Monument to Endurance · Collector's Vault ·
Treasure Map · Reckoner Bankbuster · Smuggler's Share · Rhystic Study · Idol of Oblivion ·
Skullclamp · Garruk's Uprising · Return of the Wildspeaker · Bonehoard Dracosaur.

**Interaction (10)** — Swords to Plowshares · Teferi's Protection · Generous Gift ·
Flawless Maneuver · Heroic Intervention · Beast Within · Chaos Warp · Vandalblast ·
Deflecting Swat · Blasphemous Act, plus Crux of Fate and Toxic Deluge as the splash wipes.

---

## 8. The off-colour splashes, and the rule they obey

Treasures tap for any colour, so a splash is a **timing** question, not a mana-base one.

> **Splash sorcery-speed one-shots. Never instants.**

An instant held up means Treasures kept untapped across every opponent's turn — Treasures the
engine is not spending. That is precisely the structural failure the debrief found in Edgar:
*"mana was spent casting vampires, so there was never open mana to hold up."*

**In:** Crux of Fate `{3}{B}{B}` (asymmetric wipe you survive) · Toxic Deluge `{2}{B}` ·
Demonic Tutor `{1}{B}` · Rhystic Study `{2}{U}` · Revel in Riches `{4}{B}`.
Every one is sorcery-speed. Rhystic Study and Revel in Riches are also **enchantments that tax
opponents** — the same wipe-proof cluster as Smothering Tithe.

**Rejected by the rule, not by colour:** Counterspell, Swan Song, Cyclonic Rift, Vampiric Tutor.

---

## 9. The second pass — what the review changed

Every slot was re-checked for a card doing more for the same or marginally more mana.

| out | in | why |
|---|---|---|
| Big Score, Unexpected Windfall | **Reckoner Bankbuster** mv2 | one-shot draw-two-plus-treasure → *recurring* three cards, then a Treasure **and** a body |
| Peregrin Took mv2 | **Fable of the Mirror-Breaker** mv3 | Food only → body + loot + treasure + Kiki-Jiki copying Manufactor |
| Currency Converter mv1 | **Monument to Endurance** mv3 | +2 mana, recurring draw *and* treasure off every discard |
| Lathliss mv6 | **Bonehoard Dracosaur** mv5 | cheaper, still a Dragon for Eminence, and does three jobs |
| Monologue Tax mv3 | **Revel in Riches** mv5 | +2 mana, and it is a **win condition** that eats opponents' wipes |
| Captain Lannery Storm mv3 | **Charming Scoundrel** mv2 | cheaper, not combat-gated, three modes |
| Cleansing Wildfire mv2 | **Volatile Fault** (land) | same effect from a slot the deck already spends |
| Boros Charm, Glorybringer | Toxic Deluge, Demonic Tutor | a wipe and a tutor beat a 4-damage modal and a fifth Dragon |

---

## 10. Cost

**29 already in the deck · 6 in a box · 6 sleeved elsewhere · 23 to buy. 55% reuse.**

Sleeved elsewhere, each a trade-off rather than a purchase: Bonehoard Dracosaur and
Blasphemous Act (gishath) · Idol of Oblivion and Vandalblast (**goblin-storm, which is
locked — buy or cut, do not unsleeve it**) · Rhystic Study (sisay) · Sword of Wealth and Power
(zur-enchantress).

**No dollar figure — prices are stripped from the corpus.** Ragavan is the line item most
likely to dominate and the deck functions without him. Price before ordering.

---

## 11. Open questions for the goldfish

1. **Does it reach 10 Treasures** for Revel in Riches — the deck's one real alternate win?
2. **Does Hellkite Tyrant ever connect?** Its value is the theft, and the theft needs combat.
   If the answer is "rarely", it is a 6-mana 6/5 flier and should be judged as one.
3. **Are two Treasures actually available** when a splash is wanted? 83.7% is the chance of
   having *drawn* two sources by turn four — not of having two Treasures on board. Only a
   simulation separates those.
4. **Is 36 lands right** now that five treasure-lands are in the mana base?

---

## 12. Addendum — the commander question, settled

**The objection was right and sharper than "weak":** with twelve Dragons, Eminence saves ~12
mana across a game, which is real — but the *second* ability reads *"whenever Dragons you
control **attack**, draw that many cards"*, and this deck was rebuilt specifically to stop
needing an attack step. The commander's payoff asked for the one thing the refactor removed.

**Three alternatives were costed against the 63:**

| commander | keeps | loses | what the command zone does |
|---|---|---|---|
| Mahadi, Emporium Master `{1}{B}{R}` mv3 | 43 | **21** | a Treasure per creature that died **table-wide** |
| Ziatora, the Incinerator `{3}{B}{R}{G}` mv6 | 56 | 8 | sac a creature → damage + 3 Treasures |
| **The Ur-Dragon** — kept | **64** | **0** | five-colour access |

**No treasure commander exists that keeps both green and white.** Every three-plus-colour
option including white is combat-gated or outlaw-typal. Mahadi is the best fit for a
wipe-heavy pod and costs Doubling Season, Parallel Lives, Old Gnawbone and all the white.

**Decision: keep The Ur-Dragon, and treat five-colour access as the axis** — which is only
worth something if it is *spent*. Treasures tap for any colour, so the deck takes the best
treasure card in every colour instead of staying inside three.

**Measured, that means black.** Blue is deliberately left unspent: its best cards for this
deck are colourless ones that need no splash at all. Two black additions:

- **Black Market Connections** `{2}{B}` — *treasure, a card, and a 3/2 body, every turn,
  choose one or more.* An enchantment, one pip, three jobs. The single best expression of
  the brief's own "a card that does two jobs beats a card that does one".
- **Pitiless Plunderer** `{3}{B}` — a Treasure whenever another creature you control dies,
  so your own losses to the pod's wipes become fuel.

Out: Rhystic Study (blue, and sleeved in sisay) and Crackle with Power (redundant beside
Jaya's Immolating Inferno).

**Final: 63 spells · 36 lands · 30 treasure sources · 92.8% to open with one · 14 cards doing
three or more jobs · mean mana value 3.41 · colours used W B R G.**
