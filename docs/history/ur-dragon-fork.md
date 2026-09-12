# The Ur-Dragon: costing the commander fork

**Status:** a costed comparison for the pilot to decide from. Nothing here is applied.
**Measured:** 2026-08-26, against V2 (the 36-land list, `decklist_sha256 f77e320ebea0…`).
**Prompted by:** the pod debrief of 25 August (§8, "OPEN DECISION — commander"), and
`log.jsonl` entry 002 — a win the deck did not earn, on a mana base that never arrived.

---

## 1. The finding that reframes the question

The deck does not have a land-count problem. It has an **allocation** problem, and the
allocation is inverse to demand.

| colour | pip share | source share | sources | target | short | on-curve |
|---|---|---|---|---|---|---|
| **R** | **54.3%** | **30.4%** | 29 | 36 | **7** | 0.804 |
| G | 19.6% | 24.6% | 24 | 22 | −2 | 0.927 |
| W | 13.0% | 15.9% | 18 | 22 | 4 | 0.849 |
| U | 9.8% | 15.9% | 18 | 22 | 4 | 0.849 |
| **B** | **3.3%** | **13.0%** | 16 | 30 | 14 | 0.563 |

Red is **54% of the deck's 92 pips and gets 30% of its sources**. Black is 3.3% of pips —
*two cards*, earliest relevant turn 5 — and consumes 13% of the mana base.

**Q1 from the debrief is answered: 36 is not the problem, and a 37th land will not help.**
The V2 swap added five genuinely good fixing lands (Cavern of Souls, Mana Confluence,
Prismatic Vista, Sacred Foundry, Shivan Reef) and did not touch the ratio.

> **On the targets.** These come from `manabase.sources_needed` at 90%, which is a
> *stricter question* than Karsten's published table — raw unconditional hypergeometric,
> where Karsten conditions on hands you would keep. It runs 40–55% above his numbers
> (27 vs 19 for one pip on turn 1). The repo's own notes already call it "a yardstick, not
> a possibility, in a 36-land deck". **Every comparison below is by ratio**, which is
> robust to where the threshold sits.

Also worth correcting: `enters_tapped` reads 15 of 36, but `enters_tapped_always` is **5** —
the classifier counts shocklands as tapped. The tapped-land problem is small.

---

## 2. The four branches, measured

Survival is by **colour identity**, not by pips, so a card is lost if any symbol in its
rules text falls outside the branch.

| branch | spells kept | lost | source demand | colours per producer | treasure kept |
|---|---|---|---|---|---|
| **A** WUBRG — keep The Ur-Dragon | 63 | 0 | 132 | **2.9** | 6/6 |
| **B1** Jund (BRG) | 47 | 16 | 88 | 1.9 | 5/6 |
| **B2** Gruul (RG) | 46 | 17 | **58** | **1.3** | 5/6 |
| **C** mono-red | 34 | 29 | 36 | 0.8 | 4/6 |

Producers = 36 lands + 10 rocks and dorks = **46**. **The current base achieves 2.3 colours
per producer; branch A needs 2.9.** That is the whole of "mana never arrived", in one number
— and note six of the ten rocks produce all five colours, so the base is already leaning on
the most generic fixing available and still falls short.

### What each colour actually buys

- **B → 1 card.** Crux of Fate, and nothing else. Black exists in this deck for one card,
  at 13% of the mana and a 30-source demand. It is the worst ratio in the list.
- **U → 7 cards.** Counterspell, Dovin's Veto, Swan Song, Roiling Dragonstorm, Korlessa,
  Miirym, Temur Ascendancy.
- **W → 10 cards**, and they are the interaction suite: Swords to Plowshares, Teferi's
  Protection, Smothering Tithe, Enlightened Tutor, Flawless Maneuver, Generous Gift.
- **G → 15 cards**: the ramp (Farseek, Nature's Lore, Skyshroud Claim), the draw (Return of
  the Wildspeaker, Rishkar's Expertise), Heroic Intervention — and **Old Gnawbone**, the
  deck's best treasure engine and directly on §8's axis.

---

## 3. Branch A is not actually available

The obvious cheap move — keep the commander, reallocate sources toward red — does not exist
as a pure mana fix.

Red is short 7. Taking 7 from white and blue puts **both from 4 short to 8 short**, on
colours carrying 10 and 7 cards respectively. You cannot starve a colour you are still
casting ten spells in.

**So feeding red means cutting colours, which is branch B or C under another name.** The
branches are a continuum, not four discrete options: the real question is *how far down it
to go*, and the honest minimum is "cut black, trim white and blue hard".

---

## 4. Reading the table

**B2 (Gruul) is the value pick.** It loses one card more than Jund — exactly Crux of Fate —
and sheds **30 source-demand** doing it, because black's demand is 30 for that single card.
It keeps 5 of 6 treasure sources including Old Gnawbone, keeps all the green ramp, keeps
Heroic Intervention (which matters against §5's wipe-heavy table), and lands at 1.3 colours
per producer, which a 36-land base satisfies comfortably.

**C (mono-red) is the clean pick** and costs 29 cards. Demand 36 against 36 lands: satisfied
by basics alone. It loses Old Gnawbone and Smothering Tithe, and all the ramp and draw that
green was carrying — so the rebuild is much larger than the swap list suggests.

**B1 (Jund) is dominated.** It keeps black's 30-source demand to buy back one card.

### The commander that resolves §8's tension

**Magda, Brazen Outlaw** — `{1}{R}`, legendary Dwarf, EDHREC #1305:

> Other Dwarves you control get +1/+0. Whenever a Dwarf you control becomes tapped, create
> a Treasure token. **Sacrifice five Treasures: Search your library for an artifact or
> Dragon card, put that card onto the battlefield**, then shuffle.

She converts treasure directly into dragons *onto the battlefield*, which is §8's entire
thesis in one card, and she resolves on turn two rather than turn nine. She also explains
why §8 flagged Dwarves: her treasure trigger keys on a Dwarf becoming tapped.

The cost is that she is a 2-drop, not the 4–6 the pilot asked for — and that the treasure
engine has to be built around Dwarves to fire.

---

## 5. What this would cost to build

The physical collection is **1,003 distinct cards**, and it is green, black, vampire and
Hobbit boxes. Almost none of the red treasure package is in it: of twelve candidate
staples checked, **two are owned** (Tireless Provisioner, Big Score).

**Prices are stripped from the corpus, so this repo cannot compute a dollar figure.** §11.3
of the debrief asks for a cost delta in dollars and nothing here can honestly produce one —
that is a manual check against a price source, and it is the largest unquantified risk in
branches B and C.

Note on ownership: cards reading "not owned" that are already in the deck (Goldspan Dragon)
are not missing — ownership means *a box of loose cards*, never deck membership.

---

## 6. The mechanical consequence of choosing B or C

**A commander change is a new slug, not a version bump.** Every key in this repo is the
slug, and the rule is already written down: "Zask → Blech → Hapatra is a lineage of
cardboard, not versions of one deck."

So branch B or C means a new `data/decks/<slug>/` starting at `v1.0.0`, with `ur-dragon`
marked `superseded` — still sleeved, no longer the best version of itself. The captain's
log, the version history and the two logged games stay with `ur-dragon` and do not follow
the cards.

Branch A — or the honest continuum version of it — is a genuine **minor** bump on
`ur-dragon`: the deck would be able to do something it could not.

---

## 7. What is still unverified

- The mono-red dragon/treasure precon the pilot recalled as "Atlas". Not checked; not
  guessed at here.
- Whether the treasure package is affordable. See §5.
- Whether a Dwarf sub-theme is worth the deck slots it costs, if Magda is the commander.
