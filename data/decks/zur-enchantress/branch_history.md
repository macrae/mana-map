# zur-enchantress — retired branch record

Twenty-four branches were opened on this deck between 2026-09-04 and 2026-09-06
and every one of them was measured. **Not one beat the champion.** The branch
directories were removed 2026-09-09 to consolidate the deck to a single
definition; this file is what they were for and what they found, so the
measurements are not re-bought.

The full records remain in git history under `data/decks/zur-enchantress/branches/`.

| branch | state | damage @T10 vs champion | objective | verdict |
|---|---|---|---|---|
| bodies-v3 | open | — | — | — |
| bodies-v4 | open | 20.49 → 18.07 (-2.42) | `kill_by_8 >= 0.38` → 0.3683 (not resolvable) | worse |
| bodies-v5 | open | 20.49 → 16.75 (-3.73) | `kill_by_8 >= 0.38` → 0.3082 (not met) | worse |
| constellation-v1 | open | 20.49 → 15.75 (-4.73) | `kill_by_8 >= 0.4` → 0.2054 (not met) | worse |
| constellation-v2 | merged | 20.49 → 18.05 (-2.44) | `kill_by_8 >= 0.5` → 0.2425 (not met) | worse |
| graveyard-audit-v1 | merged | 20.49 → 17.87 (-2.62) | `kill_by_8 >= 0.51` → 0.2405 (not met) | worse |
| graveyard-v1 | open | 20.49 → 13.06 (-7.42) | `kill_by_10 >= 0.84` → 0.6826 (not met) | worse |
| grey-havens-out | merged | 20.49 → 17.60 (-2.88) | `kill_by_8 >= 0.26` → 0.2388 (not met) | worse |
| mana-v1 | merged | 20.49 → 17.66 (-2.83) | `kill_by_8 >= 0.27` → 0.2408 (not met) | worse |
| manabase-v1 | open | 20.49 → 13.44 (-7.04) | `kill_by_10 >= 0.79` → 0.706 (not met) | worse |
| oil-v1 | open | 20.49 → 16.64 (-3.84) | `kill_by_8 >= 0.38` → 0.301 (not met) | worse |
| pillars-v1 | open | 20.49 → 9.17 (-11.32) | `interaction_6 >= 0.8` → 0.6403 (not met) | worse |
| pillars-v2 | open | 20.49 → 8.17 (-12.31) | `interaction_6 >= 0.8` → 0.644 (not met) | worse |
| ramp-v2 | open | 20.49 → 13.46 (-7.02) | `kill_by_10 >= 0.84` → 0.713 (not met) | worse |
| ramp-v3 | open | 20.49 → 13.29 (-7.19) | `kill_by_10 >= 0.84` → 0.6964 (not met) | worse |
| reach-v1 | open | 20.49 → 16.58 (-3.90) | `kill_by_8 >= 0.38` → 0.3229 (not met) | worse |
| rooms-v1 | open | 20.49 → 17.15 (-3.33) | `kill_by_8 >= 0.38` → 0.3192 (not met) | worse |
| shrine-anthem-only | open | — | — | — |
| shrines-v1 | merged | 20.49 → 12.98 (-7.51) | `kill_by_10 >= 0.84` → 0.7054 (not met) | worse |
| shrines-v2 | merged | 20.49 → 20.68 (+0.20) | `kill_by_8 >= 0.28` → 0.2386 (not met) | noise |
| shrines-v4 | open | 20.49 → 20.30 (-0.19) | `kill_by_8 >= 0.36` → 0.3261 (not met) | noise |
| shrines-v5 | open | 20.49 → 19.98 (-0.51) | `kill_by_8 >= 0.36` → 0.3387 (not met) | worse |
| toolbox-v1 | open | 20.49 → 11.21 (-9.28) | `kill_by_10 >= 0.75` → 0.5738 (not met) | worse |
| toolbox-v2 | merged | 20.49 → 13.36 (-7.13) | `kill_by_10 >= 0.79` → 0.7065 (not met) | worse |

## What each branch argued, and what it found

### bodies-v3

ZUR FETCHES BODIES, NOT EFFECTS. The pilot's read, and it is the first plan
that answers the arithmetic: the deck removes 4.15 life a game and needs 160,
while surviving to turn 32 — the defence works and there is no offence. Zur's
trigger puts the card ONTO THE BATTLEFIELD, so fetching an enchantment
CREATURE is a free body every attack; fetching a Propaganda is not. Eleven of
the twelve adds are Zur-fetchable enchantment creatures, including a 5/5 flier
(Master of the Feast) and two indestructible Gods whose devotion this deck can
actually reach — measured: Heliod needs ~8 nonland permanents on board and
Athreos ~7, against 55 in the list. Thassa needs ~17 and stays a utility
enchantment rather than a body. Enduring Tenacity is the pilot's own find:
Vito on a body that returns as an enchantment when it dies, which is
redundancy for the payoff that already resolves most often. NO one-card win
conditions and no two-card infinites — the kill is a board. MODEL NOTE: this
is the first Zur list to declare `model_combat`, because until now the
goldfish reported `output.available: false` — "no magnitude series ... that is
an absent measurement, not a zero" — and could not see a clock at all. THE
GOLDFISH HAS NO BLOCKERS, so its kill turn is a CLOCK against one 40-life
opponent who never blocks and it OVERSTATES a creature plan by construction.
It is used here because Forge's AI cannot pilot the attack-to-tutor line at
all, and because the error is in a known direction and is the same for every
list it compares.

### bodies-v4

THE COMMANDER'S GRANT IS FREE AND DOES NOT CARE WHAT ANYTHING COSTS, so the
cheapest enchantment creature is the most efficient carrier of deathtouch,
lifelink and hexproof. This is the hypothesis the Shrine failure pointed at:
shrines-v4 and v5 both raised the curve and both lost board power at T6 (7.00
-> 6.43 / 6.66). Five one- and two-drops instead. Every one is a BUY -- the
owned pool holds only three cheap enchantment creatures and all three are
marginal -- so this branch is also the price check on that hypothesis.

### bodies-v5

bodies-v4 plus the draw the model can actually see. bodies-v4 was the best of
three hypotheses -- the only one to raise board power at T6 (7.00 -> 7.49) and
killed-by-T10 (0.855 -> 0.869) -- but every branch lost card draw, and model-
coverage shows why in part: Wavebreak Hippocamp and Hateful Eidolon both have
draw the parser does NOT read, so two of the five adds were vanilla bodies to
the model. Those two are replaced by CANTRIP EVASION AURAS. Angelic Gift and
Feather of Flight are {1}{W}, grant FLYING and draw a card, so they cost no
card; Unquestioned Authority grants protection from creatures, which is
unblockable. On a body the commander has already made hexproof, deathtouch and
lifelink, a flier is close to unanswerable, and each aura is also one more
animation target. This tests evasion and bodies TOGETHER, which the three
single-axis branches could not.

### constellation-v1

ZUR PUTS AN ENCHANTMENT ONTO THE BATTLEFIELD EVERY TIME HE ATTACKS, and that
is a constellation trigger the deck was barely collecting. Only three cards in
the 100 read it. The multiplier is free and recurring, so every constellation
payoff added is another effect per attack. Mesa Enchantress -> Entity Tracker
is the sharpest swap in the deck and is a like-for-like on the draw group:
Mesa says 'whenever you CAST an enchantment spell', and Zur's fetch puts them
onto the battlefield WITHOUT casting -- the deck's own engine never turned it
on. Entity Tracker draws on ENTER, so it draws off every attack. Balemurk
Leech is a second Grim Guardian at mana value 2: each opponent loses 1 life
per enchantment entering. Archon of Sun's Grace makes a 2/2 FLYING LIFELINK
Pegasus per enchantment -- bodies, evasion, and life gained, which this list
converts to damage three ways. The cuts are the cast-trigger that never fired,
an aura-tutor body the commander outclasses, and a one-shot ritual.

### constellation-v2

Two more payoffs on the axis that multiplies, and both cuts pay a second time
in the mana. Gremlin Tamer mints a 1/1 per enchantment at mana value 2;
Ajani's Chosen mints a 2/2. Zur makes an enchantment enter on every attack, so
each of these is another body per turn on top of Archon's Pegasus. THE CUTS
LOWER TWO COLOUR REQUIREMENTS: Fear of Impostors is the deck's remaining
{U}{U} card and Tymaret its cheapest {B}{B} one, and a source target is driven
by the earliest double pip -- black has been 7 sources short since manabase-v1
precisely because Tymaret wants {B}{B} on turn two. Cutting them should move
the targets down as well as the payoffs up. Cost stated: a counterspell and a
body leave, and Tymaret was also a graveyard answer.

### graveyard-audit-v1

The Master of Keys is a 3-mana 3/3 flier whose whole text box is dead here,
and the audit says why. Its escape grant needs enchantment cards IN YOUR
GRAVEYARD and three more to exile, and this deck has exactly TWO cards that
put anything into its own graveyard -- Conduit Pylons (surveil 1, once) and
Master of Keys itself. Worse, the deck's main way of deploying enchantments is
Zur's fetch, and X is 0 outside the stack: fetched, it enters with NO counters
and mills NOTHING, so the one card that could have fuelled the engine does not
even fuel itself. Turning it on means hard-casting at X=4 for seven mana in a
deck whose curve tops at four. Fear of Infinity is the same mana value, also
Zur-fetchable, and every word of it works here: FLYING and LIFELINK, and
lifelink is damage in this list because three cards turn life gained into life
lost; Eerie, so it is another payoff reading the enchantment-enters trigger
Zur hands over free; and it returns ITSELF from the graveyard off that same
trigger. Cost stated: three coloured pips become two, so devotion drops by
one, and it cannot block.

### graveyard-v1

Graveyard breadth without the self-harm, plus a recurring evasive body. The
audit reports interaction-breadth at 3 classes against a target of 5, with
graveyard named as unanswered. The obvious answer is Rest in Peace and it is
WRONG HERE: it blanks Nighthowler, a {1}{B}{B} creature whose entire stat line
is creatures in graveyards, and hurts Tymaret and The Master of Keys.
Summoner's Sending is the one-sided version -- YOU choose the graveyard -- and
it also makes a 1/1 FLYING Spirit at every end step, which is a recurring
evasive body in a deck that kills with a board. It is an enchantment, so Zur
fetches it and it counts for constellation. The cut is Light-Paws, Emperor's
Voice, which reads strong and is not: it triggers on an Aura entering only IF
YOU CAST IT, and Zur's fetch puts auras onto the battlefield WITHOUT casting
them -- the deck's own engine turns its trigger off. Same mana cost, same
single white pip, so devotion does not move.

### grey-havens-out

The Grey Havens taps for {C} and, separately, for one mana of any colour AMONG
LEGENDARY CREATURE CARDS IN YOUR GRAVEYARD. This deck runs six legendary
creatures and can put none of them there: Zur goes to the command zone,
Athreos and Heliod and Thassa are INDESTRUCTIBLE, and nothing in the 100
mills, surveils or discards. That leaves Daxos and Vito, both of which have to
die and stay dead. In this deck it is a Wastes that scries 1 -- and
land_colors counts it as a source of all five colours. It is a leftover from
the build that ran Tymaret and The Master of Keys, both now cut. Shattered
Sanctum is a real white-black source in a deck whose black is the tightest
colour.

### mana-v1

The oil. Six of this deck's lands are counted as coloured sources and CANNOT
PAY A COLOURED COST ON CURVE: their only coloured mode costs extra mana on top
of the tap. Excluding them, every colour is short -- white 18 against a target
of 22, blue 18 against 22, black 25 against 36 -- where the headline figures
read 24/24/31 and only black looked wrong. The four cut here have no
compensating utility; Talon Gates of Madara phases out a creature and
Darkwater Catacombs is a real filter once a land is down, so both stay. The
three adds are UNTAPPED DUALS, each covering two colours, plus a Swamp for the
fourth slot. A first attempt at this went the other way -- cutting the gated
lands for basics -- and mana-fit refused it: those lands ARE counted for all
three colours, so cutting them without replacing the fixing made every gap
worse.

### manabase-v1

The worst axis in the audit and the only one that is deterministic. Black
wants 36 sources and has 24 -- a -12 gap driven by six double-black cards,
four of which ARE the drain engine -- and black is on-curve in 66.7% of games
against 91.8% for white and blue. One game in three cannot cast its black
cards on time. The cuts are mana-fit`s OWN cut candidates (Adarkar Wastes and
Skycloud Expanse make only colours already at target, and Skycloud costs {1}
to use) plus two Islands -- blue is the LEAST-pipped colour at 21.5% and had
the MOST basics at five -- and Valgavoth`s Lair, the deck`s only always-tapped
land. Every add makes black while keeping a second colour, so black rises
without white or blue falling below target. Measured by mana-analysis, which
is deterministic: this is arithmetic over the decklist, not a simulation, and
the goldfish cannot rank two lands that make the same colours.

### oil-v1

THE AUDIT NAMES THE HOLES AND NONE OF THEM ARE THREATS. deck-audit reports
threat-density 11 against a target of 3, and 26 creatures; it reports UNDER on
exactly three axes -- colour-sources (black is 30 against the 36 a 90% on-
curve rate wants, on-curve 0.718 ungated), ramp (9 against 10-12) and
interaction-breadth. The goldfish agrees from the other side: missed land drop
by T5 is 0.405, two games in five. Board power at T6 is 7.00 and the bodies
already exist; what is missing is the mana to deploy them. Any-colour rocks
buy ramp and the black gap in one card.

### pillars-v1

ABANDON commander damage as the win condition. Measured over 40 Forge games on
the pilot's own table: 0 of 39 games reached 21 commander damage, 0.35
commander damage a game, 2.33 total combat damage against edgar's 32.3 — and
Zur itself attacked 0.33 times a game, so the engine barely fired and the kill
never did. Rebuilt on the three pillars the pilot named: (1) counter target
spell, (2) stax, (3) enchantment-creature trickery for ASYMMETRIC table
damage. The four swing-enablers stay but now serve the ENGINE rather than the
kill — Zur attacking is what fetches an enchantment, and the trigger is on
ATTACK, not on connect. OBJECTIVE CAVEAT: the goldfish cannot grade this
deck's output at all ('no magnitude series ... that is an absent measurement,
not a zero'), so the objective is the CONTROL half — of the games where an
answer is in hand on turn six, the share where the turn also ends with the
mana to cast it, currently 0.728. The kill itself is settleable only by
simulate, which is also where the recon's contested finding about voltron at a
five-player table belongs.

### pillars-v2

pillars-v1 PROVED THE ENGINE FIX AND EXPOSED THE NEXT CONSTRAINT. Measured
over 40 games a side against the playgroup table: Zur attacked 0.20 -> 0.60 a
game and the tutor fired 0.28 -> 1.00 a game, a 3x and 3.6x improvement from
the evasion package alone. The win rate did not move (0/40 both arms) because
the engine now runs and finds a deck it cannot convert: every pillar piece
resolves in roughly one game in four or five (Propaganda 0.20, Ghostly Prison
0.23, Doomwake Giant 0.25, Vito 0.47, Counterspell 0.07), which is what a
singleton with no tutoring does. v2 attacks CONSISTENCY rather than adding win
conditions: three more Zur-fetchable enablers, and four ways to FIND one
(Idyllic Tutor finds any enchantment; Open the Armory, Light-Paws and Heliod's
Pilgrim find an Aura, which is what every enabler is). The cuts are the
expensive non-fetchable one-ofs the measurement showed resolving least.

### ramp-v2

ramp-v1 with Hypnotic Siren kept. v1 cut it as a weak {U} body and that ONE
card was holding blue`s requirement down: with it gone the {U}{U} spells
(Counterspell, Muddle the Mixture, Fear of Impostors) defined the blue pip
profile, blue`s source target jumped 22 -> 36 and blue on-curve COLLAPSED
89.5% -> 61% -- worse than the black problem this whole line of work started
from. A cheap single-pip card is load-bearing for a colour requirement in a
way nothing about the card itself suggests. Three white cuts for three rocks;
blue is untouched.

### ramp-v3

ramp-v2 plus the pilot's line, which defeats the objection this whole branch
was built around. Artifacts accelerate but Zur cannot fetch them -- except
Urza's Saga IS an enchantment card with mana value 0, so Zur puts it ONTO THE
BATTLEFIELD off an attack: a free land drop, and three turns later chapter III
fetches an artifact costing {0} or {1} onto the battlefield as well. Sol Ring
arrives through the tutor engine. Lotus Petal comes in as the second chapter-
III target, because Arcane Signet and Talisman of Dominance both cost {2} and
the Saga cannot find them -- with Sol Ring alone the chapter is live in far
fewer games. The Saga is also an ENCHANTMENT on the battlefield for
constellation. It costs a Plains (white is +2 over target) and Nyx-Fleece Ram,
a 0/4 that contributes nothing to board power. Stated cost: the Saga
sacrifices itself after chapter III and makes only {C}, so it does nothing for
the black shortfall.

### reach-v1

SIXTEEN BODIES THAT CANNOT CONNECT DEAL NO DAMAGE. Damage at T10 is 18.99
against the 40 a kill takes, and this is the exact mechanism that sank the
Edgar go-wide refactor: the goldfish has no blockers, so it scores a board the
table would simply wall off. Deathtouch plus FIRST STRIKE (Archetype of
Courage) kills a blocker before it deals damage, which turns every enchantment
creature into an unprofitable block; Flowering of the White Tree is an anthem
plus protection; Aqueous Form and Whispersilk Cloak make one body unblockable
outright. Tests evasion and pump rather than more or cheaper bodies.

### rooms-v1

THE DECK ALREADY HOLDS FOUR CARDS WHOSE TEXT NAMES A ROOM AND OWNS NO ROOMS.
Balemurk Leech, Entity Tracker, Fear of Infinity and Gremlin Tamer all read
'Eerie -- whenever an enchantment you control enters AND WHENEVER YOU FULLY
UNLOCK A ROOM'; the second clause has never once been able to trigger. Five
more cards carry constellation. A Room pays this deck three separate ways from
one card: the ETB triggers all nine payoffs, the unlock triggers the four
Eerie ones a SECOND time, and CR 709.5 says a permanent does not have the mana
cost of a locked half -- so by CR 202.3d a FULLY UNLOCKED Room's mana value is
both doors combined, and the commander's {1}{W} animates it into a body whose
power is that sum. Unholy Annex // Ritual Chamber is a 8/8 once both doors are
open. Every pick has a one- to three-mana first half, because the cheap half
is what buys the constellation trigger. THE MEASUREMENT IS A FLOOR:
goldfish.py contains no concept of unlocking, so it will score each Room as
one enchantment entering and see none of the second trigger or the animation
size.

### shrine-anthem-only

Probe: are the two mana-value-4 Shrines worth their slot? Zur fetches MV<=3,
so Honden of Cleansing Fire and Southern Air Temple must be HARD-CAST in a
deck whose nonland mean is 2.37, and the eight-Shrine test already showed the
curve breaking. shrine-cheap drops both; shrine-anthem-only keeps Southern Air
Temple, whose anthem was the single card that flipped the package's verdict,
and drops only the Honden. Go-Shintai of Lost Wisdom is the last Esper Shrine
at MV<=3 and keeps the count up.

### shrines-v1

The pilot's shrine idea, taken at its two strongest cards and NOT as a
subtheme. The tension is real: a Shrine scales with the number of Shrines, so
one Shrine is normally close to blank, and building the count needs six or
eight of them -- most of which are mana value 4 and therefore OUTSIDE Zur's
fetch. But two of them resolve that tension in THIS deck specifically. Sanctum
of Stone Fangs and Northern Air Temple both read 'each opponent loses X life
AND YOU GAIN X', and this list already turns life gain into damage three
separate ways -- Vito, Marauding Blight-Priest and Enduring Tenacity. So a
lone Shrine here is not a blank: gaining one life is a drain trigger, and the
Shrine count is a multiplier on top of a floor that already pays. Both are
single black pips at mana value 1 and 2, so Zur fetches them, they raise no
new double-black requirement, and they add black devotion for Athreos. The
cuts are the two narrowest toolbox answers, keeping Journey to Nowhere and
Reprobation, which are the real removal. NOT A SHRINE DECK -- two cards that
happen to be Shrines and would be good here if the word never appeared on
them.

### shrines-v2

SHRINES ARE NOT A SECOND AXIS -- every Shrine is an ENCHANTMENT ENTERING, and
nine cards in this deck already read that event. So a Shrine costs no new
axis: it fires Grim Guardian and Balemurk Leech for a drain each, Underworld
Coinsmith for a life which Vito and Enduring Tenacity and Marauding Blight-
Priest each convert to more drain, Entity Tracker for a card, and Archon of
Sun's Grace, Ajani's Chosen and Gremlin Tamer for a body each. THREE SHRINES
THEN READ IT BACK: Northern Air Temple drains and gains on every other Shrine,
The Spirit Oasis draws on every other Shrine, and Southern Air Temple puts a
+1/+1 counter on EVERY creature on every other Shrine. Shrines 2 -> 6. The
cuts are the cards this deck could least defend: Master of the Feast hands
three opponents a card every upkeep in a deck that wins slowly and nothing
here punishes a draw; Smuggler's Share needs an opponent to draw TWO in a turn
and Master never gave them two; Dreadhorde Invasion is slow tokens that cost
life; Silent Hallcreeper is one unblockable body on an axis the deck no longer
needs. Stated cost: curve up two mana, black devotion down three pips, and
four of the twelve Esper Shrines are mana value 4 so they are hard-cast rather
than fetched.

### shrines-v4

SHRINES AS BODIES, NOT AS A COUNT. Under the old commander these were
unfetchable liabilities at mana value 4 and 5; under Zur, Eternal Schemer
POWER EQUALS MANA VALUE, so Honden of Seeing Winds is a 5/5 and Sanctum of
Calm Waters a 4/4 -- and once animated they are enchantment creatures, so the
commander's own grant gives them deathtouch, LIFELINK and HEXPROOF. Go-Shintai
of Shared Purpose is already an enchantment creature and gets the grant
without being animated at all. The count is NOT the argument and is not
claimed: six Shrines are drawn 1.06 times by turn ten and the whole twelve-
card cycle only reaches 1.9, so 'for each Shrine you control' multiplies by
about one either way. THIS ALSO SURVIVES A BAD PILOT, which is the point:
Forge activated the commander's {1}{W} twice in twenty-two games, but the
static grant needs no activation at all, and a hexproof deathtouch lifelink
body does its work whether or not anybody presses a button. All three adds are
in the pilot's own boxes. Cuts are a 1/1 bestow creature that is weak at both
ends, a colourless rock, and a 2/2 whose graveyard hate the deck no longer
leans on.

### shrines-v5

shrines-v4 without the self-inflicted wound. That branch added three Shrines
and paid for one of them with Talisman of Dominance -- a MANA ROCK -- while
raising the curve from 2.67 to 2.79. Cutting acceleration to add five-drops is
the error, not the Shrines: it measured -0.068 on kill-by-t8 with the interval
excluding zero. This keeps all three rocks and adds two Shrines instead of
three. Same thesis as v4: under this commander POWER IS MANA VALUE, so Honden
of Seeing Winds is a 5/5 and Go-Shintai of Shared Purpose is already an
enchantment creature that takes the grant natively. The Shrine COUNT is still
not claimed as an argument.

### toolbox-v1

Four situational answers Zur can FETCH, against four cards it was drawing. The
tutor searches MV<=3 enchantments on attack, so a narrow answer costs nothing
in card economy -- you do not draw it, you go and get it when the problem
appears. Cuts are the redundant fourth evasion aura (Aether Tunnel, Spirit
Mantle), the weakest enchantment-creature body (Glyph Elemental) and a
hexproof effect Solitary Confinement overlaps (Aegis of the Gods). Adds are
Ghostly Prison, Propaganda, Tainted Remedy, Solitary Confinement. THE PIP
COUNT IS DELIBERATELY UNCHANGED -- four one-pip cards out, four one-pip cards
in -- so devotion is held constant and the cost lands where it can be read:
curve and bodies. This branch is measured for its COST. The goldfish has no
opponents and CANNOT price a hate card; the benefit is argued, not measured,
and no number will be attached to it.

### toolbox-v2

toolbox-v1 bought four answers and paid seven points of kill-by-t10 for them.
This branch goes after that price directly. The four adds are MV1-2 rather
than MV3, and TWO OF THEM REPLACE ANSWERS ALREADY IN THE LIST, so the ANSWER
redundancy group stays at eleven members rather than shrinking. Nothing is cut
from the way-through group, which toolbox-v1 took from seven members to five.
No creature is cut, so bodies are flat. The pip count is again held constant
at four white pips out, four white pips in, so devotion does not move. Net
curve: eleven mana out, six mana in -- five cheaper. Same caveat as toolbox-v1
and it is the whole point: the goldfish has no opponents and CANNOT price a
hate card. This measures the COST only.

