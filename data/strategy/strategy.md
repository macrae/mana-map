# Mana Map Strategy Companion

The strategic counterpart to the Comprehensive Rules: how expert players think.
The manuals cite rules for what *happens*; this document grounds coaching in the
established schools of thought for what to *do*. Every section carries sources;
section IDs (`strategy:<slug>`) are stable citation targets for the pilot-coach,
manual-writer, and strategy-researcher agents. Tier ★ grounding — curated and
human-reviewed, not machine-verified.

## strategy:card-advantage — Card Advantage

Card advantage is generating more usable cards than your opponents: drawing
extra cards, trading one for two or more of theirs, or deploying permanents that
persist and keep producing value. Reid Duke calls it "very likely, the single
most important concept in competitive Magic." Count every exchange: removal that
answers a two-card threat is a two-for-one; a draw-two is +1 net card. The
player who sees more cards makes more real decisions per game and can afford
worse individual exchanges — small consistent edges in exchange rate decide long
games. Duke's balance rule: card advantage governs the late game, when mana is
abundant and both players can deploy everything they draw; in the mana-scarce
early turns tempo often matters more (strategy:tempo). The expert-level mistake
is counting raw cards without weighing impact — see virtual card advantage
below. Commander correction: with three opponents, one-for-one trades are net
losses to the table, and engines that out-draw three players are the format's
defining advantage (strategy:multiplayer.asymmetry).

Sources:
- Reid Duke, "The Basics of Card Advantage" — https://magic.wizards.com/en/articles/archive/level-one/basics-card-advantage-2015-07-13
- Reid Duke, "Tempo & Card Advantage: A Delicate Balance" — https://magic.wizards.com/en/articles/archive/level-one/tempo-card-advantage-delicate-balance-2014-11-17
- Jason Rice, "Unified Theory of Commander: Card Advantage" — https://brainstormbrewery.com/unified-theory-of-commander-card-advantage/
- Patrick Chapin, "Next Level Magic" (print)

### strategy:card-advantage.virtual — Virtual Card Advantage

Card count doesn't tell the whole story: a card that blanks several opposing
cards generates advantage without drawing anything. Duke's example is impact —
one high-impact draw (a Shivan Dragon) can outweigh several marginal ones. A big
blocker against an aggro deck turns their small creatures into dead cards; a
graveyard hoser blanks half a dredge hand; protection from red turns a burn
player's removal suite into blanks. Flores' role analysis runs on the same idea:
the misassigned deck's situational cards go virtually dead. The practical habit
is counting *live* cards, not cards — ask which of the opponent's likely cards
your permanent invalidates, and which of yours are dead in this matchup;
mulligan and sideboard decisions are largely virtual-advantage decisions.
Commander correction: virtual advantage triples — one tax or pillow-fort effect
can blank attack plans from three players at once, the cheapest way to "answer"
~120 life worth of aggression (strategy:multiplayer.asymmetry).

Sources:
- Reid Duke, "The Basics of Card Advantage" — https://magic.wizards.com/en/articles/archive/level-one/basics-card-advantage-2015-07-13
- Mike Flores, "Who's the Beatdown?" — https://articles.starcitygames.com/premium/whos-the-beatdown/
- Patrick Chapin, "Next Level Magic" (print)

### strategy:card-advantage.plan-draw — Draw on the Plan: Not Running Out After Turn 7

The ten draw slots in strategy:deckbuilding.ratios say nothing about shape,
and shape decides turn eight. Steady draw — Phyrexian Arena, Black Market
Connections — pays a card a turn whatever the board does, best cast early.
Zaccagnino: "if games are ending between turns 6-9, this slow drip card
advantage just isn't all that impressive". On-plan draw pays only when the
deck does what it does — on death (Midnight Reaper, Grim Haruspex, Species
Specialist), on cast (Vanquisher's Banner), on gain (Well of Lost Dreams, Dawn
of Hope) — so it draws nothing in the engine-less hand and in bursts once the
engine runs, the wipe-insurance property (
strategy:aristocrats.wipe-insurance). The trade is correlation: steady draw
diversifies, on-plan draw fails with the plan and refills with it. A typal or
aristocrat midrange that empties its hand by turn 7 wants the slots split, not
stacked — steady draw live in the enabler-less keep
(strategy:mulligans.engine-hands), the rest on- plan so a sacrifice or a swing
draws the next wave; no primer gives the split. Price life-costed draw against
the drain that refunds it: Reaper's point per card is negligible beside Blood Artist gains (GenoDoak777).

Sources:
- Timothy Zaccagnino, "The 20 Most Overrated and Overplayed Cards in Commander Ranked" — https://draftsim.com/mtg-overrated-commander-cards/
- Steve Vrooman, "The 52 Best Black Card Draw Cards in Magic Ranked" — https://draftsim.com/black-card-draw-mtg/
- GenoDoak777, "Cards That Draw a Card When Creatures Die in Commander - Top 10" — https://cardmystic.com/articles/5773cd5d-ac55-43dd-b674-bfb869cf35e7
- Reid Duke, "The Basics of Card Advantage" — https://magic.wizards.com/en/articles/archive/level-one/basics-card-advantage-2015-07-13

## strategy:tempo — Tempo

Tempo is board position bought with mana efficiency: Duke defines it as board
presence — how your permanents match up against your opponent's — and it accrues
to the player who spends each turn's mana on purpose. A tempo play trades
resources unevenly to buy time or position: bouncing a threat, deploying a cheap
threat plus protection, forcing a full-turn answer to a half-turn investment.
Initiative is the engine — the proactive player forces reactive defense and
dictates when tricks and removal get cast. Tempo and card advantage sit in
deliberate tension, and Duke warns no equivalency formula holds in every
situation: the skill is reading which resource the current game rewards.
Mana-scarce early turns and snowball positions are tempo games; stalls and
mana-abundant late games are card-advantage games. The expert marker: if you
regularly end turns with unspent mana and no plan for it, your curve or your
sequencing is wrong. Commander correction: pure tempo devalues against three
untap steps — bounce and taxes buy turns, not wins; convert tempo into engines,
or spend it timing the pivot window (strategy:multiplayer.pivot-window).

Sources:
- Reid Duke, "Tempo" — https://magic.wizards.com/en/articles/archive/level-one/tempo-2015-07-20
- Reid Duke, "Tempo & Card Advantage: A Delicate Balance" — https://magic.wizards.com/en/articles/archive/level-one/tempo-card-advantage-delicate-balance-2014-11-17
- Patrick Chapin, "Next Level Magic" (print)

### strategy:tempo.sequencing — Sequencing & Spell Timing

Duke's first rule: think through the entire turn before doing anything. Then
order actions so information-gathering comes first and information-revealing
comes last — cast the draw spell before choosing how to spend the rest of the
mana; delay the land drop when a scry could change which land you want. Default
spell timing is to wait until the last possible moment, with two standing
exceptions: cast early when the spell informs the rest of your turn, and strike
in the window when opponents are tapped out and cannot respond. Combat tricks
want the opponent's mana already spent; removal sometimes wants casting just
before their untap, when protection mana is down. The payoff compounds
invisibly: opponents forced to decide before you reveal make slightly worse
decisions all game. Commander correction: you are sequencing around three
responders — count each seat's open mana separately, and prefer committing at
the end step before your own turn so the table's chances to respond are spent
(strategy:multiplayer.pod-management).

Sources:
- Reid Duke, "Sequencing" — https://magic.wizards.com/en/articles/archive/level-one/sequencing-2015-02-16
- Reid Duke, "When to Cast Your Spells" — https://magic.wizards.com/en/articles/archive/level-one/when-cast-your-spells-2015-08-31

## strategy:life-as-resource — Life as a Resource

Your life total is currency, not a score. Only the last point matters: every
other point can be spent — absorbing damage instead of chump blocking, paying
life for cards or effects, shaving turns off your own clock by tapping out. Duke
puts it directly: "your life total is a resource too, though with some important
differences" — its points appreciate as they dwindle, so early points are cheap
and late points are priceless. Experts price life in cards and turns: taking 5
to keep a blocker back for a bigger threat is often buying a card for 5 life, a
fine rate. The discipline cuts both ways — knowing when your life total has
become real (combo decks in range, burn reach, lethal on board next turn) and
switching from spending to defending one turn before it matters, not one after.
Commander correction: 40 life makes the currency deeper — aggressive life
payments price in at a discount — but the total is attacked from three
directions and shortcut by commander damage (21 from any one commander) and
infect; track each opponent's clock on you separately
(strategy:multiplayer.pod-management).

Sources:
- Reid Duke, "Damage Racing" — https://magic.wizards.com/en/articles/archive/level-one/damage-racing-2015-05-04
- Mike Flores, "Life and Cards I: Philosophy of Fire" — https://magic.wizards.com/en/news/making-magic/life-and-cards-i-philosophy-fire-2014-04-28
- Patrick Chapin, "Next Level Magic" (print)

### strategy:life-as-resource.philosophy-of-fire — The Philosophy of Fire

Adrian Sullivan's 1999 frame, canonized by Mike Flores: treat cards as quanta of
damage. If Shock is one card for 2 damage, ten Shocks are a dead opponent — a
burn deck is a machine for converting a hand into twenty damage, and every mana,
card, and turn must translate into damage for it to win. The frame prices
aggression: it tells you what a point of damage is worth in cards, and therefore
when racing beats trading. It also arms the defender: every extra card you force
the aggressor to spend — a blocker they must burn away, incidental lifegain —
comes from a finite stock, and burn decks lose when their damage-per-card rate
is pushed below lethal. The classic misapplication is throwing burn at the face
while the board kills you, when removal was the higher-value cast. Commander
correction: raw cards-to-damage math collapses against ~120 combined life —
reach must scale (each-opponent drains, commander damage, damage doublers) or be
held as targeted finishing rather than the plan
(strategy:multiplayer.asymmetry).

Sources:
- Mike Flores, "Life and Cards I: Philosophy of Fire" — https://magic.wizards.com/en/news/making-magic/life-and-cards-i-philosophy-fire-2014-04-28
- Patrick Chapin, "Next Level Magic" (print)

### strategy:life-as-resource.lifegain-engine — Life Gain as an Engine

Life gain is a buffer until something converts it; the engine is sources,
converters and draw (Dunn). Gain-to-loss: Sanguine Bond and Vito, Thorn of the
Dusk Rose turn each gain into "target opponent loses that much life" — Vito
"is Sanguine Bond on a cheaper, albeit legendary body" — and Cliffhaven
Vampire drains each opponent 1 per gain event. Loss-to-gain: Exquisite Blood,
which with either of the first two is a two-card infinite (
strategy:life-as-resource.lifegain-engine.bond-blood). Read a converter's
scaling before its source: Vito, Bond and Well of Lost Dreams scale with the
AMOUNT gained, so a six-power lifelink body connecting is six aimed drain or six
cards; Cliffhaven and Dawn of Hope scale with the COUNT of gain events, so Soul
Warden shapes feed them and one big hit does not. Hurst's test — "The amount of
life doesn't have to be large, just consistent" — is a count engine's; a
lifelink-body deck is an amount engine and should pick converters to match. Draw
is the same split: Well is "a one-to-one ratio of mana-to-card-draw" on the
amount, Dawn "still 2 mana for each card" per event (Dunn); Zucchetti's Liesa
list runs all three. Lifelink is a source, not a payoff.

Sources:
- Jeff Dunn, "The 54 Best Payoffs for Gaining Life in Magic Ranked" — https://draftsim.com/mtg-lifegain-payoffs/
- Andy Hurst, "An EDH Deck Tech - Pridemates" — https://andythurst.substack.com/p/an-edh-deck-tech-pridemates
- Jonathan Zucchetti, "Recross the Paths - Building a Vampire Deck with Liesa, Shroud of Dusk" — https://edhrec.com/articles/recross-the-paths-building-a-vampire-deck-with-liesa-shroud-of-dusk

### strategy:life-as-resource.lifegain-engine.bond-blood — The Bond + Blood Infinite in Bracket Terms

Sanguine Bond or Vito with Exquisite Blood loops any gain, or any opponent's
loss, until the table is dead: Spellbook's record reads "Infinite lifeloss.
Infinite lifegain", estimated "Spicy (Bracket 3-4+)", and Furtado's Edgar
guide names the Vito half outright — "gain some life and then just have them
trigger back and forth forever". It is the shape Bracket 3's barometer names:
decks there should not have two-card infinites that "happen cheaply and in
about the first six or so turns" (
strategy:deckbuilding.power-level.barometers). Ten mana across two enchantments
rarely does, but the test is intent, not turn count: holding both halves is a
declaration the pregame conversation has to carry, and a list that wants to stay
at 3 runs one half as a fair engine — Bond or Vito alone is a drain doubler on
every gain, Exquisite Blood alone a buffer that refunds every drain. Price the
pair by what it costs the bracket, not by how often it assembles (
strategy:deckbuilding.redundancy-vs-tutors).

Sources:
- Commander Spellbook, "Sanguine Bond | Exquisite Blood" (combo record 690-3966) — https://commanderspellbook.com/combo/690-3966/
- Pedro Furtado, "Edgar Markov Commander Deck Guide" — https://draftsim.com/edgar-markov-edh-deck/
- Gavin Verhey, "Introducing Commander Brackets Beta" — https://magic.wizards.com/en/news/announcements/introducing-commander-brackets-beta

## strategy:threat-assessment — Threat Assessment

Threat assessment is distinguishing what actually threatens your path to victory
from what is merely annoying. Answers are scarce — Duke's asymmetry: "while
there are wrong answers, there are no wrong threats" — and an answer only counts
when it matches a threat that actually beats you. Rank threats on three axes:
clock (how fast it kills you), inevitability (who wins if the game goes long —
strategy:whos-the-beatdown.metagame-clock), and interaction cost (what answering
it later costs versus now). The assessment is always relative to *your* win
condition: a value engine is lethal against your control plan and irrelevant
against your combo kill. Misspent removal is the quiet way good players lose.
Commander correction: assessment becomes mutual and continuous — the table is
assessing you back — and the ranking axes widen: draw engines, ramp bursts,
combo proximity, and board *quality* outrank board quantity; re-rank every turn
instead of fixing on one player (strategy:multiplayer.threat-deflection).

Sources:
- Reid Duke, "Threats and Answers" — https://magic.wizards.com/en/articles/archive/lo/threats-and-answers-2014-09-08
- A.L. Walser, "The Ultimate Guide to Threat Assessment in Commander" — https://draftsim.com/mtg-commander-threat-assessment/
- Jason Krell, "Threat Assessment in Commander" — https://blog.cardkingdom.com/threat-assessment-in-commander/

### strategy:threat-assessment.answer-economy — Answer Economy

Removal is a budget; spend it where it buys wins. The classic leaks: killing the
scariest creature instead of the one that beats you, spending removal on sight
instead of at the last profitable moment, and answering a threat someone else
was about to answer for you. Rice's Commander rule generalizes cleanly: unless
politics demands otherwise, hold single-target removal until the threat is
pointed at you — every one-for-one you cast in multiplayer is a net gift to the
bystanders, so the bar is "ends the game or ends my deck's function," not "looks
scary" (strategy:multiplayer.asymmetry). Walser's mistake list applies in every
format: short-term focus (spending now the answer you'll need for the real
threat), vindictive targeting (punishing history instead of board state), and
linear assessment (never re-ranking as the board changes). The countervailing
risk: held answers rot — against combo, or lethal already on board, the last
profitable moment is now.

Sources:
- Jason Rice, "Unified Theory of Commander: Card Advantage" — https://brainstormbrewery.com/unified-theory-of-commander-card-advantage/
- A.L. Walser, "The Ultimate Guide to Threat Assessment in Commander" — https://draftsim.com/mtg-commander-threat-assessment/
- Reid Duke, "Threats and Answers" — https://magic.wizards.com/en/articles/archive/lo/threats-and-answers-2014-09-08

### strategy:threat-assessment.resource-denial — Playing Under Resource Denial

Resource denial comes in three flavors — LaPage's taxonomy: stax strips
resources already amassed (sacrifice effects, wraths, land destruction), taxes
raise the price of converting resources (Sphere and Thalia effects), and locks
stop new resources arriving at all (Winter Orb, Contamination). Decks built on
many cheap spells per turn are hit hardest: taxes attack the velocity math
directly, so recount your hand under the tax — a two-spell turn becoming a
one-spell turn halves the engine. Role assignment is forced: you rarely own the
long game under an assembling lock, so you are the beatdown, racing the lock's
completion (strategy:whos-the-beatdown.metagame-clock). Two clocks favor you:
layering — one piece is an inconvenience; the compounding second and third are
the lock — and conversion: McGuinness's cEDH reading is that stax must still
turn "how do I not lose" into winning, and that conversion is slow. Name your
outs early — the removal or velocity that breaks the piece that beats *you*,
not the scariest piece — and sequence to protect them
(strategy:resource-hedging.playing-to-outs). Johnson's frame: stax wins by
breaking parity; break it back, or win before it breaks.

Sources:
- James LaPage, "The Metaworker: Stax, Tax, and Resource Denial" — https://themanabase.com/the-metaworker-stax-tax-and-resource-denial/
- Harvey McGuinness, "Let's Talk About Stax in cEDH" — https://commandersherald.com/lets-talk-about-stax-in-cedh/
- Stephen Johnson, "What Is Stax in Commander?" — https://www.coolstuffinc.com/a/stephenjohnson-01302023-what-is-stax-in-commander

## strategy:whos-the-beatdown — Role Assignment: Who's the Beatdown?

Mike Flores, 1999: "The most common (yet subtle, yet disastrous) mistake I see
in tournament Magic is the misassignment of who is the beatdown deck and who is
the control deck." In every matchup one deck is the beatdown (racing to end the
game) and one is the control (aiming for the long game) — and the roles are set
by matchup and game state, not your deck's label. The aggro deck gets forced
into the control role by a faster deck; a control mirror makes someone the
beatdown. Duke's operational tests: who has inevitability? — whoever loses the
long game must be the beatdown; who is faster? — the faster clock should force
the action; and when decks are comparable, this game's draws, play/draw, and
sideboard bombs settle it game by game. Misassignment loses slowly and
invisibly: playing control when you must race concedes the only axis you could
have won on. Ask every game, re-ask after every board change. Commander
correction: role is seat-relative — you can be the beatdown against one opponent
and the control against another in the same turn cycle; assign per-threat, not
per-game (strategy:multiplayer.pod-management).

Sources:
- Mike Flores, "Who's the Beatdown?" — https://articles.starcitygames.com/premium/whos-the-beatdown/
- Reid Duke, "Role Assignment" — https://magic.wizards.com/en/articles/archive/level-one/role-assignment-2015-01-05
- Reid Duke, "Inevitability" — https://magic.wizards.com/en/articles/archive/level-one/inevitability-2014-12-08

### strategy:whos-the-beatdown.metagame-clock — Inevitability & the Metagame Clock

Inevitability is Duke's name for ownership of the long game: if one player is
virtually guaranteed to win a game that goes long enough, that player has it —
and knowing whether it's you decides your role (strategy:whos-the-beatdown).
Its sources: superior late-game card quality, an engine that scales, an
unanswerable finisher, or simply more answers than the opponent has threats.
The metagame extends the clock reading to preparation: know the field's clocks
and where your deck's speed sits among them — Duke channels Karsten's
"winner's circle" method of weighting recent tournament finishes, and relays
Chapin's information cascades: popularity is self-reinforcing, so the popular
deck is not always the best-positioned one. Practical table test: picture turn
15 with both players on their best draws — whoever likes that picture less
should be attacking now. Commander correction: four-player games trend long, so
inevitability compounds — engines are the default winning plan and aggro must
instead compress the game before the table's late game arrives
(strategy:multiplayer.asymmetry).

Sources:
- Reid Duke, "Inevitability" — https://magic.wizards.com/en/articles/archive/level-one/inevitability-2014-12-08
- Reid Duke, "The Metagame" — https://magic.wizards.com/en/articles/archive/level-one/metagame-2015-06-01
- Mike Flores, "Who's the Beatdown?" — https://articles.starcitygames.com/premium/whos-the-beatdown/

## strategy:pivot-point — The Pivot Point

Role assignment is not static — the pivot point is the moment the correct role
flips, and recognizing it a turn early is a defining expert skill. A midrange
deck that spent five turns developing pivots to beatdown the turn its board
outclasses the table; a racing deck pivots to control the moment its opponent
stabilizes. Duke calls the defender's version "turning the corner": once you
stabilize, every extra passive turn is a free draw step for the opponent, so the
flip to offense should be immediate. The pivot is usually triggered by a
concrete event — a resolved engine, a wrath, a key removal spell spent, a life
total crossing a clock threshold. Plan for it before it happens: hold the
protection spell for the turn you commit, not after; stop deploying the turn
attrition starts favoring you. Posture follows role — from ahead, simplify the
game and shrink variance; from behind, complicate it and keep outs alive
(strategy:resource-hedging.playing-to-outs). Commander correction: the pivot
must clear three defenses at once, which makes the window narrower and the
timing more decisive — see strategy:multiplayer.pivot-window.

Sources:
- Mike Flores, "Who's the Beatdown?" — https://articles.starcitygames.com/premium/whos-the-beatdown/
- Reid Duke, "Role Assignment" — https://magic.wizards.com/en/articles/archive/level-one/role-assignment-2015-01-05
- Reid Duke, "Damage Racing" — https://magic.wizards.com/en/articles/archive/level-one/damage-racing-2015-05-04
- Reid Duke, "Playing From Ahead, Playing From Behind" — https://magic.wizards.com/en/news/feature/playing-ahead-playing-behind-2015-03-30

## strategy:information — Information Inference & Bluffing

Magic is a hidden-information game; experts treat every opponent action — and
inaction — as data. Land drops missed, mana held open, the spell they chose not
to cast, the speed of a block decision: each narrows the range of hands they can
hold. If a player with open mana lets a game-ending engine resolve, infer the
interaction isn't there and sequence accordingly. Bluffing is the mirror image:
attack into the bigger blocker as if holding the trick, hold two lands open with
nothing, play at instant-speed pace. The bluff's value isn't the single stolen
exchange — it's that opponents who must respect tricks play slower and worse.
Give up as little as possible in return: Duke's sequencing rule — reveal-last
ordering — means the earliest plays each turn should be the least informative
(strategy:tempo.sequencing). Commander correction: three hands to read, and
three players reading yours; table talk is both data and deliberate
disinformation, and visibly open mana taxes the whole table's plays, not one
opponent's (strategy:multiplayer.politics, strategy:multiplayer.pod-management).

Sources:
- Reid Duke, "Sequencing" — https://magic.wizards.com/en/articles/archive/level-one/sequencing-2015-02-16
- Reid Duke, "When to Cast Your Spells" — https://magic.wizards.com/en/articles/archive/level-one/when-cast-your-spells-2015-08-31
- Patrick Chapin, "Next Level Magic" (print)

### strategy:information.range-tells — Ranges, Tells & Representing

Work in ranges, not guesses: every action prunes the set of hands an opponent
can hold. Held lands plus instant-speed pace means interaction until proven
otherwise; a missed land drop caps what they can represent; an instant no-block
means either a trick or nothing worth protecting. Weight inaction heaviest —
what they did NOT do with mana available is the strongest signal (they let the
engine resolve: the counterspell isn't there). Representing is the offensive
half: keep open the mana for the trick you don't have, sequence so your possible
holdings stay wide, and keep your pace constant — the classic tells are the snap
play when the decision should have been hard and the long tank when it should
have been easy. Decide bluffs in advance so your tempo never betrays them.
Commander correction: your image persists — a pod remembers which bluffs you
run and which fears you exploit, so tells and credibility are multi-game
resources to manage deliberately (strategy:multiplayer.threat-deflection,
strategy:multiplayer.politics).

Sources:
- Reid Duke, "Sequencing" — https://magic.wizards.com/en/articles/archive/level-one/sequencing-2015-02-16
- Reid Duke, "When to Cast Your Spells" — https://magic.wizards.com/en/articles/archive/level-one/when-cast-your-spells-2015-08-31
- Patrick Chapin, "Next Level Magic" (print)

## strategy:combat-math — Combat Math

Expert combat is modeled at least one full turn cycle ahead: project not just
this combat's exchange but what your attack does to the opponent's options on
the crack-back. An attack that looks risky often forces a passive posture — if
counter-attacking would leave them dead to your return swing, your swing was
safe. Duke's baselines: creatures are for combat ("the value of your creatures
plummets if you're unwilling to put them into combat"); when equivalent
creatures face off, usually attack and usually block — trade early, before
tricks and removal can break the exchange; and cast your own tricks when the
opponent's mana is spent, not into a full grip of open mana. Count damage in
windows: their outs, your outs, and what each block or no-block line does to
both clocks. The common leaks are fear-based — chump blocking early to save
spendable points, and declining profitable attacks to dodge cards the
statistics say aren't there (strategy:combat-math.probability).
Commander correction: combat is also diplomacy — every attack chooses an enemy
and opens your defenses to two bystanders; incremental combat damage rarely
scales against three life totals (strategy:multiplayer.asymmetry).

Sources:
- Reid Duke, "Attacking and Blocking" — https://magic.wizards.com/en/articles/archive/level-one/attacking-and-blocking-2015-07-27
- Reid Duke, "Damage Racing" — https://magic.wizards.com/en/articles/archive/level-one/damage-racing-2015-05-04
- Reid Duke, "Playing Safe and Playing Scared" — https://magic.wizards.com/en/articles/archive/level-one/playing-safe-and-playing-scared-2015-08-24

### strategy:combat-math.racing — Damage Racing

A race is the state where neither player can take firm control — both are
counting turns to lethal. Count both clocks exactly, then optimize yours: attack
early even from a defensive posture, because races arrive unannounced and banked
damage wins the ones that do (Duke: defense "doesn't mean that you can't attack
when the opportunity presents itself"). Spend life freely above the threshold
where it buys tempo or cards, and price every chump block: blocking early wastes
a creature that still had attacking value; blocking late risks the trick that
makes lethal arrive a turn early. The racing pivot is turning the corner — the
moment you stabilize, flip to offense to shrink the window in which they can
draw out of it (strategy:pivot-point). Initiative favors the racer: forcing
blocks converts even your worst attackers into removal. Commander correction:
you cannot race three life totals — races there are two-player subgames run
while the rest of the table develops, so spend the minimum that wins yours and
keep watching the other seats (strategy:multiplayer.pod-management).

Sources:
- Reid Duke, "Damage Racing" — https://magic.wizards.com/en/articles/archive/level-one/damage-racing-2015-05-04
- Reid Duke, "Attacking and Blocking" — https://magic.wizards.com/en/articles/archive/level-one/attacking-and-blocking-2015-07-27

### strategy:combat-math.probability — Outs & Probability

Put numbers on "playing around": with N live outs among K unseen cards, each
draw hits at roughly N/K — a playset they've shown none of, forty cards unseen,
is about a 10% draw, and declining profitable attacks for three turns to dodge a
10% trick costs more than it ever saves. Frank Karsten's hypergeometric work is
the reference for the deck-side numbers: his land-count tables, and more
importantly the method — count successes in the deck and cards seen, read the
cumulative probability — answer any "will I have it by turn X" question your
deckbuilding or line depends on. Play around a card only when respecting it is
cheap or the card is actually likely (they kept seven holding up mana; the
format wraths on four). Duke's playing-scared test is the discipline: if you
would still lose the games where they have it, stop paying to play around it.
Commander correction: singleton decks flatten card probabilities — any specific
card is a rounding error, so play around *classes* of effects (a wrath by turn
six, some counterspell behind blue mana) rather than named cards.

Sources:
- Frank Karsten, "How Many Lands Do You Need to Consistently Hit Your Land Drops?" (mirror of the 2017 ChannelFireball article) — https://orkerhulen.dk/onewebmedia/How%20Many%20Lands%20Do%20You%20Need%20to%20Consistently%20Hit%20Your%20Land%20Drops.pdf
- Reid Duke, "Playing Safe and Playing Scared" — https://magic.wizards.com/en/articles/archive/level-one/playing-safe-and-playing-scared-2015-08-24

## strategy:resource-hedging — Resource Hedging & Playing to Win

Loss aversion is the enemy: "playing not to lose" prioritizes delaying defeat
over maximizing win probability, and it is mathematically wrong. Duke's
diagnostic pair: playing *safe* is correct risk-reduction from a winning
position; playing *scared* is paying real equity to dodge specific cards —
handing a control deck free draw steps by refusing to commit — and it loses
games the opponent never had the card in. Know your own bias, aggressive or
conservative, and correct against it. When behind, find the line that wins if an
assumption holds — they don't have the wrath, the top card is a land — and
commit to it, because the hedged line loses to everything anyway
(strategy:resource-hedging.playing-to-outs). When ahead, hedging is correct:
deploy only what you need, simplify, bank protection. The rule of thumb: hedge
from ahead, gamble from behind. Commander correction: with three opponents
someone can usually punish overcommitment, so the hedged default is stronger —
but the pivot window still demands full commitment the turn it opens
(strategy:multiplayer.pivot-window).

Sources:
- Reid Duke, "Playing Safe and Playing Scared" — https://magic.wizards.com/en/articles/archive/level-one/playing-safe-and-playing-scared-2015-08-24
- Reid Duke, "Playing From Ahead, Playing From Behind" — https://magic.wizards.com/en/news/feature/playing-ahead-playing-behind-2015-03-30
- Paulo Vitor Damo da Rosa, "PVDDR's Articles" (Substack archive) — https://pvddr.substack.com/archive
- Patrick Chapin, "Next Level Magic" (print)

### strategy:resource-hedging.playing-to-outs — Playing to Your Outs

When no line beats their best case, optimize for the game states you can still
win. Duke: identify the card or event that saves you and play as if it's coming
— "if it doesn't come up, you lose in either case; if it does come up, then
you've given yourself the best chance." The out isn't always a draw: it can be
an opponent's mistake, so complicate the board, keep decisions in front of them,
and never concede a game still being played. Severa's refinements: aggressive
lines often *create* outs that passive lines close off; know what you can't beat
and stop paying for it ("don't play around something you can't beat anyway");
and separate calculated risk from recklessness by asking what fraction of
realistic game states each line actually wins. The discipline cuts both ways —
from ahead, count THEIR outs and close them: simplify, hold up the answer, take
the line that loses to nothing. Commander correction: outs multiply — a third
player's interaction, a political deal, or the archenemy drawing the table's
fire all count; choose lines that any of them rescues
(strategy:multiplayer.politics).

Sources:
- Reid Duke, "Playing From Ahead, Playing From Behind" — https://magic.wizards.com/en/news/feature/playing-ahead-playing-behind-2015-03-30
- Matt Severa, "Learning To Truly Play To Your Outs" — https://articles.starcitygames.com/articles/learning-to-truly-play-to-your-outs/

### strategy:resource-hedging.wrath-math — Sweeper Math & Sizing Commitment

The canonical hedge: how much board to commit under a possible sweeper. Duke's
rules: don't overextend into an unknown sweeper without a reason; commit exactly
as much as wins before it arrives, no further; and buy resilience instead of
restraint where you can — high toughness, indestructible and regeneration,
token rebuilders, threats that replace themselves. The math is a two-line
comparison: what you lose if the wrath comes now versus what holding a threat
back costs in clock and position — and against a control deck, holding back
often IS the loss, because buying time is exactly their plan. After the wrath,
initiative goes to whoever rebuilds cheapest: redeploy worst-first, keeping the
best threat in hand for their second sweeper. Commander correction: assume the
wrath — over a four-player game one is near-certain — so deploy in waves, hold a
rebuild package, and when possible let another player's board be the reason it
gets cast (strategy:multiplayer.pivot-window).

Sources:
- Reid Duke, "Board Sweepers" — https://magic.wizards.com/en/articles/archive/level-one/board-sweepers-2015-06-22
- Reid Duke, "Playing Safe and Playing Scared" — https://magic.wizards.com/en/articles/archive/level-one/playing-safe-and-playing-scared-2015-08-24

## strategy:critical-mass — Linear & Critical-Mass Strategy

Duke's linear-strategies frame: a linear deck is "entirely focused on one goal
or theme" — every card feeds one plan, and the plan is reaching the threshold
where the synergies take over. Below critical mass the deck is a pile of
undersized cards; past it, linear decks are "designed to spiral out of
control" and outclass fair decks. The tradeoff is stated flatly — "linear
strategies give up flexibility in exchange for power" — interaction slots are
the price of the engine, so the deck's defense is speed, redundancy, and
resilience, not answers. Play one by protecting the threshold:
mulligan to enablers (strategy:mulligans.engine-hands), sequence so one removal
spell can't break the chain, and know your number — the count of bodies, spells,
or mana at which the deck actually goes over the top
(strategy:critical-mass.storm-math). Play against one by denying the threshold
before it snowballs: hit enablers, not payoffs. Commander correction: three
players' interaction punishes the naked all-in — the linear deck needs
redundancy that shrugs off three answers, or a deflection posture that keeps
them unspent (strategy:multiplayer.threat-deflection,
strategy:multiplayer.pivot-window).

Sources:
- Reid Duke, "Linear Strategies" — https://magic.wizards.com/en/articles/archive/level-one/linear-strategies-2014-12-29
- Mike Flores, "Who's the Beatdown?" — https://articles.starcitygames.com/premium/whos-the-beatdown/

### strategy:critical-mass.storm-math — Storm Velocity & the Go Decision

Girten names the storm balance: "a balance of mana to cast all the
spells you need, enough cards in hand to cast sufficient spells in one turn,
AND the right combination of spells to actually do something that impacts the
game." Velocity is those three multiplied — floating mana, castable spells,
cantrip draws — and the go decision is arithmetic done *before* the first
spell: the chain you can bank on versus the storm count, bodies, or damage the
payoff needs for lethal. Count only hand and board; expected draws are a bonus,
not a plan. Enabler density is the deckbuilding half: Karsten's hypergeometric
method answers "how many rituals and cantrips until the engine assembles by
turn X" just as it answers land counts —
successes in deck, cards seen, cumulative probability
(strategy:combat-math.probability). Separate ceiling from median: the full
storm kill is the ceiling; the median game is the value engine under it — pace
for the median, hold the ceiling as opportunism. A forced half-go
(telegraphed sweeper, assembling lock) changes the payoff math: go for value or
a wide board now rather than holding for a lethal that never gets cheaper
(strategy:threat-assessment.resource-denial).

Sources:
- Jeff Girten, "Muerra, Trash Tactician: Storm is for Trash Pandas (Plot Twist #14)" — https://commandersherald.com/muerra-trash-tactician-storm-is-for-trash-pandas-plot-twist-14/
- Frank Karsten, "How Many Lands Do You Need to Consistently Hit Your Land Drops?" (mirror of the 2017 ChannelFireball article) — https://orkerhulen.dk/onewebmedia/How%20Many%20Lands%20Do%20You%20Need%20to%20Consistently%20Hit%20Your%20Land%20Drops.pdf
- Reid Duke, "Linear Strategies" — https://magic.wizards.com/en/articles/archive/level-one/linear-strategies-2014-12-29

## strategy:aristocrats — Aristocrats: Outlet & Death-Trigger Economics

An aristocrats turn needs three things — fodder, an outlet, a payoff — and the
outlet is the piece that gives the pilot "control over when things die"
(Ullman). Price outlets by activation first. A free one (Viscera Seer; Warren
Soultrader's one life is "essentially free in our 40-life format", Gottfried)
fires with no mana open, in response to removal, and loops while
you have bodies; a mana-gated one (Indulgent Aristocrat's {2}) converts one
body per two mana and competes with the curve — a payoff with an outlet
attached. Commander Theory's frame: an outlet makes every creature
answer-proof. Gregory's correction is the cost: a free outlet is "a
fragile, highly suspicious permanent" that "signif[ies] a checkmate scenario",
so it eats the removal your payoffs would draw. Fodder comes three
ways (Gottfried): tokens, cheap bodies, recursion; a real threat is
fodder only when about to die anyway — sacrificed in response, a one-
for-one becomes a death trigger plus a wasted spell. Drain per death is the
exchange rate: converters live × deaths you can manufacture this turn, against
the lowest total (strategy:combat-math.racing) — closing is a separate question
(strategy:conversion).

Sources:
- Cooper Gottfried, "EDHREC Guide to Aristocrats in Commander" — https://edhrec.com/guides/edhrec-guide-to-aristocrats-in-commander
- Alex Ullman, "Treasured Finds: The Four Types of Aristocrat Commanders" — https://nerdtothecore.com/2023/11/15/treasured-finds-the-four-types-of-aristocrat-commanders/
- Kristen Gregory, "There's No Such Thing as Free in Commander" — https://blog.cardkingdom.com/theres-no-such-thing-as-free-in-commander/
- Commander Theory, "Sacrifice Outlets" (unattributed blog post) — https://commandertheory.com/post/58081860785/sacrifice-outlets/amp

### strategy:aristocrats.drain-scope — Drain Scope in a Pod: Target vs Each Opponent

Read the trigger and the drain separately. Blood Artist triggers on every
creature that dies, anyone's, and drains "target player" for 1; Zulaport
Cutthroat, Cruel Celebrant and Bastion of Remembrance trigger only on your own
creatures and drain each opponent for 1. In a four-player pod the each-
opponent shape moves 3 life per death against Blood Artist's 1 — the three-to-
one that strategy:multiplayer.asymmetry asks of a scaling payoff — so ten
deaths through two each-opponent payoffs take 60 of the table's ~120 life, and
the same ten through Blood Artist take 20, aimed. What Blood Artist buys is
two different things. Trigger scope: a table's chump blocks, trades and
someone else's sweeper all pay you (alexis: it triggers "from your opponents'
creatures, too", where Cutthroat "cares specifically about your creatures").
And a choice of target — a finisher and a political lever: every drain at the
player on 4 while the rest of the table watches (Royale: Blood Artist can
"target just one opponent unlike the other cards in the deck with similar
effects"). EDHREC's Edgar lifedrain theme carries Blood Artist in 89% of 484
decks. Rule: each-opponent for the race, targeted for the kill.

Sources:
- alexis, "Ygra, Eater of All Commander Deck Guide" — https://draftsim.com/ygra-eater-of-all-edh-deck/
- David Royale, "Teysa Karlov Commander Deck Guide" — https://draftsim.com/mtg-teysa-karlov-edh-deck/
- EDHREC, "Edgar Markov — Lifedrain" (theme page, inclusion rates) — https://edhrec.com/commanders/edgar-markov/lifedrain
- Jason Rice, "Unified Theory of Commander: Card Advantage" — https://brainstormbrewery.com/unified-theory-of-commander-card-advantage/

### strategy:aristocrats.wipe-insurance — Rebuilding After a Wipe: Death-Triggered Draw

With a free outlet and a death-draw body on board, a wrath on the stack is
answered by sacrificing the whole board in response — each body a card, the
drains firing on the way out — so the wipe resolves against an empty table and
a full hand. Vrooman: Midnight Reaper "gives you plenty of cards as you sac
your way through your board"; GenoDoak777 says it directly — repeatable death-
triggered draw keeps a sacrifice deck from "running out of gas" after a wipe.
The four shapes differ on what counts. Midnight Reaper and Grim Haruspex read
nontoken creatures (Haruspex only yours, Reaper a life per card); Species
Specialist reads a chosen type, tokens included — asymmetric in a typal deck;
Liliana's Standard Bearer has flash and draws one per creature that died under
your control this turn — held, it IS the insurance, the way Vrooman keeps
Haruspex face-down via morph as a wipe surprise. Against steady draw: Arena
pays a card a turn regardless; death draw pays nothing until bodies die, then
in a burst, exactly when the hand is empty (
strategy:card-advantage.plan-draw). Hold the second outlet and one draw body
back for the rebuild (strategy:resource-hedging.wrath-math).

Sources:
- Steve Vrooman, "The 52 Best Black Card Draw Cards in Magic Ranked" — https://draftsim.com/black-card-draw-mtg/
- GenoDoak777, "Cards That Draw a Card When Creatures Die in Commander - Top 10" — https://cardmystic.com/articles/5773cd5d-ac55-43dd-b674-bfb869cf35e7
- David Royale, "Teysa Karlov Commander Deck Guide" — https://draftsim.com/mtg-teysa-karlov-edh-deck/
- Reid Duke, "Board Sweepers" — https://magic.wizards.com/en/articles/archive/level-one/board-sweepers-2015-06-22

## strategy:conversion — Conversion: Turning a Board Into Dead Opponents

Production is not conversion. John puts the whole of it in two sentences: "The
point of your deck's engine is not to make resources. The point of your engine
is to win the game." A deck that makes bodies, cards or mana has a
numerator; winning needs a channel that turns each unit into life lost, and in a
pod that channel must clear three 40-life totals, or 21 commander damage each
(John). Anderson gives the shape of the failure: trading creatures for
immediate damage
"won't go far enough to bring down multiple 40-life opponents, especially with
whoever we don't kill first taking free shots at us while we're tapped out".
Collins names it from inside a sacrifice list — without a good output "all our
deck will do is spin its wheels". Gregory's is sharpest: most strategies take
actions "at least positive to your board state" while aristocrats "throw away
those incremental accruals", so the best such decks "sacrifice things
incidentally, and are built around another strategy entirely". Diagnose a
non-closing deck as three numbers, not one — units produced, the fraction a
converter is live for, and the life each converted unit removes
(strategy:deckbuilding.threat-density).

Sources:
- John, "How to Win in Commander? Attack Your Opponents Until They Die" (airza.net) — https://airza.net/2025/03/13/how-to-win-in-commander-attack-your-opponents-until-they-die
- Tom Anderson, "Commander Archetype Inspection: Go Wide" — https://blog.cardkingdom.com/commander-archetype-inspection-go-wide/
- Kristen Gregory, "Aristocrats Decks are a Trap in Commander" — https://blog.cardkingdom.com/aristocrats-decks-are-a-trap-in-commander/
- Ciel Collins, "Dueling Deck Techs: Aristocrats" — https://commandersherald.com/dueling-deck-techs-aristocrats/

### strategy:conversion.payoff-density — Payoff Density Is Not the Lever

Density is not the lever, and this bench has the control: yawgmoth-swarm runs
nine drain payoffs against edgar-vampires' four, cheaper on average (mean mana
value 2.78 against 3.00), and fires FEWER times a game — 1.40 against 1.76. No
Commander primer states an enabler-to-payoff ratio either; that is a gap in the
literature, not a number to look up, and the fetchable floors sit one question
over (Zupke's 3-5 finishers, Nicol's 7-8 enablers against 10-12 enhancers —
strategy:deckbuilding.threat-density). What exists is revealed preference:
across EDHREC's 1,496 Edgar decks tagged aristocrats, Blood Artist appears in
93%, Cruel Celebrant 79%, Vito 72%, Bloodthirsty Conqueror 58%. Read the rate as
a product instead — units produced × P(a payoff is available when the unit
resolves) × life per fire — and note that only the first term is about slots.
Edgar: 9.64 of its own creatures die a game and its death-drains fire 1.90
times, a 20% conversion. Buy availability (strategy:conversion.availability) and
yield (strategy:conversion.yield) before buying copies.

Sources:
- EDHREC, "Edgar Markov — Aristocrats" (theme page, 1,496 decks) — https://edhrec.com/commanders/edgar-markov/aristocrats
- Andy Zupke, "Building a Commander Deck - Part Two: Structure" — https://blog.cardsphere.com/building-a-commander-deck-part-two-structure/
- Benjamin Nicol, "Solve the Equation - How to Tell if a Strategy Has Enough Support by Using Deck Templates" — https://edhrec.com/articles/solve-the-equation-how-to-tell-if-a-strategy-has-enough-support-using-deck-templates
- Mana Map, "edgar-vampires Forge run records (seeded, one pinned harness)" — https://github.com/macrae/mana-map/tree/main/data/decks/edgar-vampires/sim

### strategy:conversion.yield — Yield per Fire: Scope × Scaling

Yield per fire is scope × scaling, and the two axes cross. Scope: a targeted
drain takes 1 life off a four-player table where an each-opponent drain takes 3
(strategy:aristocrats.drain-scope). Dunn states the axis once, cleanly —
"Vizkopa's ability hits each opponent, instead of just one target", at "a
similar mana investment to Sanguine Bond". Scaling: AMOUNT-scaled converters pay
the size of the event (Vito, Sanguine Bond — "target opponent loses that much
life"), COUNT-scaled ones pay once per event whatever its size (Cruel Celebrant,
Epicure of Blood, Marauding Blight-Priest) —
strategy:life-as-resource.lifegain-engine. GAP: no writer states the crossover,
so here is the arithmetic. Per event an amount-scaled targeted converter removes
A life from the table where a count-scaled each-opponent one removes 3, so
targeted wins only when the mean gain event exceeds 3. Measure that mean: a deck
whose gain arrives as 1-life drain refunds sits near 1, which collapses Vito and
Bond to the lowest yield of the four quadrants — one life a fire. Measured
per-fire yields here: 3.75 life where the payoff is each-opponent, 1.83 where
the highest-firing payoff is single-target.

Sources:
- Jeff Dunn, "The 54 Best Payoffs for Gaining Life in Magic Ranked" — https://draftsim.com/mtg-lifegain-payoffs/
- Timothy Zaccagnino, "The Best Cards Like Blood Artist In MTG" — https://www.thegamer.com/magic-the-gathering-mtg-best-cards-like-blood-artist/
- Mana Map, "edgar-vampires Forge run records (seeded, one pinned harness)" — https://github.com/macrae/mana-map/tree/main/data/decks/edgar-vampires/sim

### strategy:conversion.availability — Putting the Payoff Where It Cannot Be Missed

Availability is bought with LOCATION, not with copies. Timm's test for an engine
is "repeatable, recoverable, affordable", and his ranking of locations is
explicit: your commander "is the key to your greatest advantage engine as it is
the most reliably repeatable card in the 100 cards", and inside the 99 he would
"favour permanents with abilities rather than instants/sorceries". The command
zone is the strongest form of that — a payoff there cannot be kept off the
table, only taxed (strategy:multiplayer.commander-insurance,
strategy:multiplayer.zonal-commander) — then an enchantment or artifact
(strategy:conversion.durability), then recursion, and only then more copies,
which climb slowly: five copies of an effect is 31% of opening hands, seven 41%
(strategy:deckbuilding.redundancy-vs-tutors). The bench's spread is the whole
argument. A deck whose payoff is its commander had one available in 80.8% of
games; a deck running four of them as two-mana creatures managed 36-39%. Count
the fraction of games the payoff reaches the battlefield before counting the
payoffs.

Sources:
- Keir Timm, "Sophisticated Synergies and EDH Engines" — https://www.intothe99.com/post/sophisticated-synergies-and-edh-engines
- Scott Cullen, "Choosing the Right Protection for Your Commander" — https://blog.cardkingdom.com/choosing-the-right-protection-for-your-commander/
- Mana Map, "edgar-vampires Forge run records (seeded, one pinned harness)" — https://github.com/macrae/mana-map/tree/main/data/decks/edgar-vampires/sim

### strategy:conversion.durability — Converters That Outlive Their Fodder

A converter's type line is part of its rate. Zaccagnino reads Bastion of
Remembrance as "a three-mana Zulaport Cutthroat on an enchantment" where "being
an enchantment keeps it safe from a larger swathe of removal and board wipes";
Cullen's case for noncreature anthems is the same trade one slot over
(strategy:deckbuilding.typal-density.lord-exposure). The sharpening no primer
states is a correlation: a creature converter dies to the same sweeper that
kills the fodder, so a death-drain deck's conversion rate collapses at the exact
moment its death count spikes — the largest conversion event of the game is the
one the converter is absent for. Three fixes, cheapest first
(strategy:conversion.availability): put the converter on a type the wipe does
not read; put it where removal cannot reach it
(strategy:multiplayer.zonal-commander); or hold a converter and a death-draw
body back for the rebuild (strategy:aristocrats.wipe-insurance,
strategy:resource-hedging.wrath-math). Price a two-mana creature converter as a
cheap body with high variance in uptime, not as a rate — this bench's Edgar kept
its most-included one on the battlefield 0.6 turns a game.

Sources:
- Timothy Zaccagnino, "The Best Cards Like Blood Artist In MTG" — https://www.thegamer.com/magic-the-gathering-mtg-best-cards-like-blood-artist/
- Scott Cullen, "Tribal Cards You Need for Commander" — https://blog.cardkingdom.com/tribal-cards-you-need-for-commander/
- Mana Map, "edgar-vampires Forge run records (seeded, one pinned harness)" — https://github.com/macrae/mana-map/tree/main/data/decks/edgar-vampires/sim

### strategy:conversion.entries-vs-deaths — Converting Entries, Not Deaths

A death costs a permanent; an entry costs nothing, because the entry already
happened. For a deck whose production channel is tokens, that makes entry
conversion the cheaper half of the same engine. Furtado's frame: when every
creature you play damages opponents on arrival "it's like they're
pseudo-unblockable, hasted threats", and that "gets better in go-wide token
strategies" — the per-trigger rate doing the rest (Purphoros' 2 against Impact
Tremors' 1). Scope multiplies it again: an each-opponent trigger moves three
life per unit where a targeted one moves one — Zaccagnino's "a single creature
dying under your control is worth three total damage in situations where Blood
Artist would only deal one" (strategy:aristocrats.drain-scope). This is
Gregory's prescription made mechanical: sacrifice incidentally, convert what the
deck does anyway. It is also the under-adopted lane — of EDHREC's 51,341 Edgar
lists, Blood Artist appears in 82% against Impact Tremors' 20% and Warleader's
Call's 18%, and the Call is also an anthem, doing the job that makes a wide
board connect (strategy:deckbuilding.typal-density). Noncombat-damage doublers
scale this channel the way anthems scale combat.

Sources:
- Pedro Furtado, "The 24 Best Impact Tremors Effects in Magic Ranked" — https://draftsim.com/mtg-impact-tremors-effects/
- Timothy Zaccagnino, "The Best Cards Like Blood Artist In MTG" — https://www.thegamer.com/magic-the-gathering-mtg-best-cards-like-blood-artist/
- Kristen Gregory, "Aristocrats Decks are a Trap in Commander" — https://blog.cardkingdom.com/aristocrats-decks-are-a-trap-in-commander/
- EDHREC, "Edgar Markov" (commander page, 51,341 decks) — https://edhrec.com/commanders/edgar-markov

### strategy:conversion.combat-channel — Why Combat Converts Badly in a Pod

Combat is the channel with three taxes. Three sets of blockers divide your width
instead of multiplying it; an attack chooses one enemy and opens you to the two
seats you did not choose (strategy:multiplayer.politics,
strategy:multiplayer.threat-deflection); and the step comes once a turn however
much you built. Anderson states the result — trading creatures for immediate
damage "won't go far enough to bring down multiple 40-life opponents, especially
with whoever we don't kill first taking free shots at us while we're tapped out"
— and his fix is multiplicative: effects reading "each creature you control",
which is why the go-wide combat channel lives or dies on anthems
(strategy:deckbuilding.typal-density; this bench measured combat damage fall
29.07 → 18.20 when a refactor cut every lord). None of this abandons combat:
Duke's creatures are still for combat, and a wide board under an anthem is the
format's cheapest twenty points. It costs it. GAP: no writer prices the
damage-per-mana of an attack step against that of a trigger; the comparison in
the literature is qualitative only, so a deck that needs the answer must measure
both channels on one harness and report both totals.

Sources:
- Tom Anderson, "Commander Archetype Inspection: Go Wide" — https://blog.cardkingdom.com/commander-archetype-inspection-go-wide/
- Reid Duke, "Attacking and Blocking" — https://magic.wizards.com/en/articles/archive/level-one/attacking-and-blocking-2015-07-27
- Mana Map, "docs/gotchas-bench.md" (this bench's Forge measurement of the Edgar go-wide refactor) — https://github.com/macrae/mana-map/blob/main/docs/gotchas-bench.md

### strategy:conversion.passive-tax — Passive Each-Opponent Taxation

The shape this bench has measured carrying a total is narrow: cheap,
each-opponent, PASSIVE — an upkeep trigger, no attack and no activation —
non-creature, and scaling on a state the deck already produces. Three four-mana
artifacts reading "at the beginning of each opponent's upkeep, deal damage for
cards in hand above four" fired 3.86 times a game and carried 9.45 of one deck's
noncombat damage; cutting them took its total life removed from 15.62 to 7.88.
Passive each-opponent taxation is a named archetype — group slug, which Ignacio
frames by its symmetry, "we're all gonna be having a bad time together" — and
the hand-size family (Black Vise, Viseling, Iron Maiden, Misers' Cage, Ebony Owl
Netsuke) is collected as a family in forum writing rather than in any primer.
Two prices. The trigger reads an opponent's state, which they can change; and
taxing the whole table makes you everyone's problem at once
(strategy:multiplayer.threat-deflection). GAP: nobody prices a passive upkeep
trigger against an attack step or a death trigger, which leaves the comparison
to the harness (strategy:conversion.combat-channel).

Sources:
- Ignacio (Iñaki), "The 21 Best Group Slug Commanders in Magic Ranked" — https://draftsim.com/mtg-group-slug-commander/
- MTG Salvation, "Cards that punish people for holding cards?" (forum thread; the hand-size family collected in one post) — https://www.mtgsalvation.com/forums/the-game/commander-edh/199278-cards-that-punish-people-for-holding-cards
- Mana Map, "heliod Forge run records (seeded, same pod and harness)" — https://github.com/macrae/mana-map/tree/main/data/decks/heliod/sim

### strategy:conversion.substitution — Substitution: Every Measured Swap Has Been a Decrease

Adding a payoff is a swap, and on this bench no swap has raised a total. Across
22 same-pod same-harness version pairs, four differences exclude zero and all
four are DECREASES — observational, spanning several merges and seeds rather
than an A/B, which is why it is read as a pattern and not an effect. The
clearest case cut three working passive pingers and took total life removed from
15.62 to 7.88, −7.73 [−12.11, −3.36]; noncombat damage fell 9.45 to 1.23, which
is the whole of it (strategy:conversion.passive-tax). SaffronOlive's rule is the
discipline that predicts the asymmetry: ask not "Can this card work in my deck?"
but "Is this the best option for this slot in my deck?" — a card rating 7 out of
10 going in over an 8 or a 9 is a net loss however well it fits the theme. GAP:
nothing this pass reached names the asymmetry itself. Its practical form: a CUT
is a reliable experiment and an ADDITION is not, so prove a card is dead by
removing it, and when it is the ceiling you want, change an input rather than a
slot (strategy:conversion.raising-the-ceiling).

Sources:
- SaffronOlive, "Brewer's Minute: Opportunity Cost in Deck Building" — https://www.mtggoldfish.com/articles/brewer-s-minute-opportunity-cost-in-deck-building
- Mana Map, "heliod and edgar-vampires Forge run records (seeded, same pod and harness)" — https://github.com/macrae/mana-map/tree/main/data/decks/heliod/sim

### strategy:conversion.raising-the-ceiling — What Raises the Ceiling When Swaps Do Not

Since no measured swap on this bench has raised a total, the ceiling sits
upstream of the payoffs — in how many INPUT events the deck generates and how
fast it deploys them. The literature's levers, roughly ordered for a
cast-triggered deck. Mana, where John is blunt: "for a 6 mana commander,
investing in cards and ramp is the only thing that lets us cast our commander at
all". Cost reduction, which Walser frames as ramp changing only "the amount of
mana you pay" and "encourages playing bundles of spells in a manner that Storm
decks in particular can use" — more casts per turn off the same lands, and on a
cast-triggered commander every qualifying cast is another trigger. Velocity
(strategy:critical-mass.storm-math). Extra combats, which re-run the
whole combat channel for one card (Sison). Damage doublers, and free spells,
which buy a cast without a land. And consistency, which Karsten prices directly
(strategy:deckbuilding.mana-base, strategy:deckbuilding.curve).
The test separating a payoff problem from an input problem: hold the 99 fixed
and ask what the deck does with two more mana, or one fewer missed land drop. If
the answer is "the same thing, twice", stop swapping spells.

Sources:
- John, "How to Win in Commander? Attack Your Opponents Until They Die" (airza.net) — https://airza.net/2025/03/13/how-to-win-in-commander-attack-your-opponents-until-they-die
- A.L. Walser, "The 50 Best Cost Reducers in Magic Ranked" — https://draftsim.com/mtg-cost-reducer/
- Julian Sison, "Seize the Day: Commander Staples That Unlock Extra Combat Power" — https://www.coolstuffinc.com/a/juliansison-seo-10212024-seize-the-day-commander-staples-that-unlock-extra-combat-power
- Frank Karsten, "How Many Lands Do You Need to Consistently Hit Your Land Drops?" (mirror of the 2017 ChannelFireball article) — https://orkerhulen.dk/onewebmedia/How%20Many%20Lands%20Do%20You%20Need%20to%20Consistently%20Hit%20Your%20Land%20Drops.pdf

### strategy:conversion.quiet-contributors — Pricing a Slot That Does Nothing Now

Cutting the card that was doing nothing visible is how an aggressive deck loses
a channel nobody was watching. Sherwood's quadrant theory is the pre-cut test:
grade the card in Developing, Winning, Losing and Parity — the last being where
"all the players have the resources to win, but no one has the advantage", whose
goal is to "break parity and move on to winning" — and that a card "crosses the
line into win more when the player was winning anyway and the new card only
increases the margin of victory". katydee's definition is the sharper half: a
win-more card "works very well, but only if you're already winning", so "it
never helps turn a loss into a win", and his caveat is the one that bites a cut
list — the concept makes players worse, because the easy error is labelling a
genuinely good card win-more and never testing it again. An enchantment that
converts and a body that attacks are not the same slot: the cut card's quiet
contributions (evasion, a blocker, a draw trigger) surface in rows you did not
aim at. Rule: before cutting, name the channel the card feeds and the figure
that would fall without it; after cutting, read every channel. And free is never free (Gregory).

Sources:
- John Sherwood, "The EDHREC Guide to Quadrant Theory in Commander" — https://edhrec.com/guides/the-edhrec-guide-to-quadrant-theory-in-commander
- katydee, "The Problem of \"Win-More\"" — https://www.lesswrong.com/posts/Dfi7Snzs4pGSAu52b/the-problem-of-win-more
- Kristen Gregory, "There's No Such Thing as Free in Commander" — https://blog.cardkingdom.com/theres-no-such-thing-as-free-in-commander/

### strategy:conversion.measure-choice — Which Output Number Predicts Winning

Pick the output number before the experiment, because efficiency and winning
come apart. On this bench the deck with the best life-removed-per-fire has the
worst win rate, and the deck with the highest TOTAL life removed has the best —
a ratio can be maximised by firing rarely, since the deck sets its own
denominator. The literature never states which output statistic predicts a
multiplayer win, and that is a gap worth naming; it is consistent, though, about
the quantities it reasons in. Flores' Philosophy of Fire counts cards as quanta
of damage against a fixed lethal total
(strategy:life-as-resource.philosophy-of-fire). Duke's racing counts CLOCKS —
turns until a named player dies, both ways (strategy:combat-math.racing). John's
pod arithmetic is a total: three 40-life seats, or 21 commander damage each.
None of the three is a rate. The rule here is this doc's own: aim a branch at a
quantity monotone in winning — a total, or a clock on one seat — never at a
quotient whose denominator the deck controls, and print the total beside any
efficiency figure (strategy:aristocrats.drain-scope — each-opponent for the race, targeted for the kill).

Sources:
- Mike Flores, "Life and Cards I: Philosophy of Fire" — https://magic.wizards.com/en/news/making-magic/life-and-cards-i-philosophy-fire-2014-04-28
- Reid Duke, "Damage Racing" — https://magic.wizards.com/en/articles/archive/level-one/damage-racing-2015-05-04
- John, "How to Win in Commander? Attack Your Opponents Until They Die" (airza.net) — https://airza.net/2025/03/13/how-to-win-in-commander-attack-your-opponents-until-they-die

### strategy:conversion.kill-pattern — The Kill Pattern a Vampire List Actually Closes With

Ask a primer how its Edgar wins and the answer is combat. Furtado: "Your typical
wincon in this deck is going to come from combat damage. Your board state will
get very large very quickly so swinging with 10+ medium-powered creatures won't
be uncommon", with a mass-drain body as the alternate — Malakir Bloodwitch
"becomes an instant win card if your board state is big enough". Inclusion
agrees: across EDHREC's 51,341 Edgar lists the two most-played cards are a lord
and a converter (Captivating Vampire 82%, Blood Artist 82%), and only 1,496
lists are tagged aristocrats against 2,144 tokens and 1,587 aggro — the
aristocrat Edgar is a minority build, and inside it the compact closer is the
gain-to-loss loop (Bloodthirsty Conqueror 58%, Exquisite Blood 53%, Sanguine
Bond 43% — strategy:life-as-resource.lifegain-engine.bond-blood). GAP: no
fetchable primer states a kill turn or a combat-versus-trigger split for this
archetype; the high-power Edgar writing lives on Moxfield and Archidekt, which
refuse automated retrieval. The literature supports the *mechanism* claim, not
the *timing* one — treat any turn number for this deck as the bench's own measurement.

Sources:
- Pedro Furtado, "Edgar Markov Commander Deck Guide" — https://draftsim.com/edgar-markov-edh-deck/
- EDHREC, "Edgar Markov" (commander page, 51,341 decks) — https://edhrec.com/commanders/edgar-markov
- EDHREC, "Edgar Markov — Aristocrats" (theme page, 1,496 decks) — https://edhrec.com/commanders/edgar-markov/aristocrats

## strategy:mulligans — Mulligan Theory

Duke's frame: the mulligan question is only "which choice gives you better odds
of winning" — answered coldly, because "starting the game with fewer cards is
bad" and every ship spends a real card. The beginner baseline — keep two to
five lands, ship the rest — is right roughly 90% of the time; expertise is
knowing the overrides. Archetype pace: fast decks mulligan slow hands
aggressively, since their edge dies in long games; control keeps marginal hands
rather than start down a card. Matchup context rules Constructed: the same hand
is a keep in a slow mirror and a ship against mono-red, and sideboard cards
brought in for the matchup tip borderline hands toward shipping. Compounding
dependencies: a hand that needs the right land AND the right spell AND a timely
draw to function is worse than it looks — the probabilities multiply. Below six
cards, standards collapse: any hand with a realistic route to victory is a
keep. PVDDR's drill sets the habit: judge the hand as a plan against the
expected matchup, not as a collection of good cards. Engine and combo decks get
their own calculus — strategy:mulligans.engine-hands.

Sources:
- Reid Duke, "Mulligans" — https://magic.wizards.com/en/articles/archive/level-one/mulligans-2015-01-26
- Reid Duke, "Mulligans Part II: Limited" — https://magic.wizards.com/en/articles/archive/level-one/mulligans-part-ii-limited-2015-06-15
- Reid Duke, "Mulligans Part III: Constructed" — https://magic.wizards.com/en/articles/archive/level-one/mulligans-part-iii-constructed-2015-06-29
- Paulo Vitor Damo da Rosa, "Keep or Mulligan #1" — https://pvddr.substack.com/p/keep-or-mulligan-1

### strategy:mulligans.engine-hands — Engine-Deck Keeps: Enablers vs Payoffs

Duke's rule for linear decks: mulligan to the key card. Bogles ships every
seven- and six-card hand without a hexproof creature, because no other hand
plays Magic — and the inverse hand fails too: all payoff, no enabler (his
example: four lands, three expensive creatures, zero acceleration) is a
mulligan despite its card quality. The engine-deck test: classify the hand's
cards as enablers, payoffs, and glue, then ask whether the hand contains a
route to the engine. Payoffs without the enabler are blanks until the engine
exists; enablers without payoffs at least dig toward them — an asymmetry that
usually favors keeping the enabler-heavy hand. Conversely, a hand holding the
key card is keepable even when imperfect, to avoid cascading mulligans. Speed
bounds it all — Duke: "coming out too slowly is an easy way to lose a game" —
ship hands that assemble after the field's clock arrives
(strategy:whos-the-beatdown.metagame-clock, strategy:critical-mass). Commander
correction: the customary free first mulligan makes the first ship near-free —
use it ruthlessly — and 100-card singleton means you mulligan to enabler
*classes*, not named cards; build the redundancy your keeps depend on.

Sources:
- Reid Duke, "Mulligans Part III: Constructed" — https://magic.wizards.com/en/articles/archive/level-one/mulligans-part-iii-constructed-2015-06-29
- Paulo Vitor Damo da Rosa, "Keep or Mulligan #1" — https://pvddr.substack.com/p/keep-or-mulligan-1

## strategy:multiplayer — Commander & Multiplayer Dynamics

Multiplayer Commander changes the resource mathematics of every 1v1 framework:
three opponents, ~120 combined life, and a table of political actors.
One-for-one answers and incremental damage lose value; asymmetric effects,
repeatable engines, and other people's removal gain it. The frameworks above
still apply, but each needs a multiplayer correction — role assignment becomes
seat-relative, threat assessment becomes mutual (the table is assessing you),
and information inference now includes table talk. Commander theory is younger
than 1v1 theory and article-borne: EDHREC and Commander's Herald carry most of
the written threat-assessment and politics work. The subsections below carry the
corrections.

Sources:
- Jason Rice, "Unified Theory of Commander: Card Advantage" — https://brainstormbrewery.com/unified-theory-of-commander-card-advantage/
- A.L. Walser, "The Ultimate Guide to Threat Assessment in Commander" — https://draftsim.com/mtg-commander-threat-assessment/
- Jason Krell, "Threat Assessment in Commander" — https://blog.cardkingdom.com/threat-assessment-in-commander/
- EDHREC, "Articles" — https://edhrec.com/articles

### strategy:multiplayer.asymmetry — The 1-to-3 Disadvantage

Every single-target card you cast is a net loss against the table: you spend a
card, one opponent loses a card, and two bystanders profit from the exchange for
free. Rice's rule follows: unless politics demands otherwise, hold removal until
the threat is pointed at you, and let opponents spend their one-for-ones on each
other. Winning consistently means preferring asymmetric board wipes, repeatable
value engines that out-scale three players, and effects that tax or drain every
opponent at once. Aggro's math is worst hit — ~120 combined life behind three
defenses makes incremental combat damage a losing plan
(strategy:life-as-resource.philosophy-of-fire); wide alpha strikes, loops,
commander-damage kills, and each-opponent drains are the payoffs that actually
scale. Threat evaluation follows the same shape: board quality beats board
quantity (Walser), because a single scaling permanent out-values a fair board
against three players. Your cards should do something to all three seats or win
the game; anything else is someone else's job.

Sources:
- Jason Rice, "Unified Theory of Commander: Card Advantage" — https://brainstormbrewery.com/unified-theory-of-commander-card-advantage/
- A.L. Walser, "The Ultimate Guide to Threat Assessment in Commander" — https://draftsim.com/mtg-commander-threat-assessment/

### strategy:multiplayer.politics — Politics & Table Negotiation

Politics is the fourth resource pillar of multiplayer: attention, favors, and
perceived threat are spendable currencies. Deals, nudges ("their board is
scarier than mine"), and shared-enemy framing redirect removal you would
otherwise eat. Political cards are engineered dilemmas — Nicol's game-theory
reading: they work because opponents facing unknown choices accept bad terms
rather than risk falling behind, so offer deals where every possible answer
profits you. Craft rules: never lie outright about game state (credibility is a
multi-game resource); keep deals small, specific, and expiring — Hinds' case
studies show vague terms get exploited and unspoken mercy debts don't get
repaid; and read the table's tone, because the same play is diplomacy in one pod
and kingmaking in another. Watch who benefits from every suggestion — including
yours. Position yourself so eliminating you is never anyone's most profitable
line until you're winning that turn.

Sources:
- Benjamin Nicol, "Solve the Equation – Game Theory Basics: Political Cards in Commander" — https://edhrec.com/articles/solve-the-equation-game-theory-basics-political-cards-in-commander
- Cas Hinds, "Should We Politic and Show Mercy in Commander" — https://edhrec.com/articles/should-we-politic-and-show-mercy-in-commander
- EDHREC, "Articles" — https://edhrec.com/articles

### strategy:multiplayer.threat-deflection — Threat Deflection

The table assesses threats mutually, and the deflection skill is managing your
*perceived* threat level below your actual one. Downplay the board ("it's just a
3/3"), deploy engines that read as defensive, sequence scary pieces after the
table's removal is spent, and keep the spotlight on the player whose board
looks — or can be made to look — scarier. Krell's warning is your playbook read
in reverse: combo decks look non-threatening while advancing in secret, which is
exactly why experienced tables hit the quiet ramp player — so expect your own
"I haven't done anything" to be discounted by anyone tracking mana and draw
engines (Walser files falling for it under political exploitation). Forcing
opponents to exhaust interaction on each other early is worth more than any
single card you could cast. Know your deck's tells — the cards everyone fears —
and time them for the turn they win, not the turn they impress. The deflection
budget spends down as your engine shows; plan the pivot before it runs out
(strategy:multiplayer.pivot-window).

Sources:
- Jason Krell, "Threat Assessment in Commander" — https://blog.cardkingdom.com/threat-assessment-in-commander/
- A.L. Walser, "The Ultimate Guide to Threat Assessment in Commander" — https://draftsim.com/mtg-commander-threat-assessment/

### strategy:multiplayer.pivot-window — The Pivot Window

The multiplayer version of the pivot point (strategy:pivot-point): the exact
window to flip from value/defensive posture to an all-out win attempt. Go too
early and the table unites to stop you with three players' worth of interaction;
too late and someone else takes their window first. The window opens when your
win attempt either can't be profitably answered — protection up, table tapped
out, interaction counted and accounted for — or doesn't need to resolve fully to
leave you ahead. Krell's endgame frame sets the bar: late-game threat assessment
collapses to "does this card win the game if it resolves," and that is exactly
the standard the table applies to YOUR attempt — assume every held answer
appears the moment you present lethal, and count the table's open mana before
committing. Prefer win attempts that leave a rebuilt position if they fail, and
sequence the scariest piece last so the deflection budget
(strategy:multiplayer.threat-deflection) lasts until the turn you take it.

Sources:
- Jason Krell, "Threat Assessment in Commander" — https://blog.cardkingdom.com/threat-assessment-in-commander/
- A.L. Walser, "The Ultimate Guide to Threat Assessment in Commander" — https://draftsim.com/mtg-commander-threat-assessment/

### strategy:multiplayer.pod-management — Pod Management & Pacing

Reading and pacing the pod as a system: who is the archenemy this turn, whose
clock is fastest, who is the table's answer-holder, and what game length each
deck wants. Krell's stage model structures the read: openers are assessed from
commanders and reputations before a card is played; early game watches who is
"setting up for the win" with ramp and draw; midgame asks who accomplishes the
most per turn and whether their win takes out one player or the table; endgame
collapses to counting held interaction. Manage pace toward the game length your
deck wins — engines want the game slow until the pivot window; aggro wants
pressure spread so nobody stabilizes into inevitability. Track attention like a
resource: after any flashy play, the table's assessment updates (Walser:
recalibrate continuously, and expect the table to), so spend a turn or two under
the radar. Leaving instant-speed mana open forces three players, not one, to
respect your possible interaction.

Sources:
- Jason Krell, "Threat Assessment in Commander" — https://blog.cardkingdom.com/threat-assessment-in-commander/
- A.L. Walser, "The Ultimate Guide to Threat Assessment in Commander" — https://draftsim.com/mtg-commander-threat-assessment/
- EDHREC, "Articles" — https://edhrec.com/articles

### strategy:multiplayer.commander-insurance — Commander Protection & the Recast Tax

When the engine routes through your commander, protection is insurance priced
against the recast tax: each return trip to the command zone adds {2} per
prior command-zone cast (Miljkovac) — and, the bigger price, a tempo turn spent
re-casting instead of advancing. Cullen's taxonomy sets the options:
constant protection (boots-style equipment and enchantments — always on, cheap
to re-arm, useless against wraths), single-use instants (cost held mana all
game but protect at the decisive moment and can cover the board), and flexible
modal slots that are never dead. His decision test: how critical is the
commander to the plan, and what removal this table actually runs. A
value commander can eat removal and re-buy — one recast of a cheap commander is
a fine rate; the engine's hub justifies constant protection down plus a held
response on the pivot turn — assume every held answer appears the moment you
commit (strategy:multiplayer.pivot-window). The
racing alternative is honest math: protection slots and held mana cost you
every game, removal only costs you some — against light-interaction pods, skip
the insurance, pay the tax, rebuild (strategy:resource-hedging).

Sources:
- Scott Cullen, "Choosing the Right Protection for Your Commander" — https://blog.cardkingdom.com/choosing-the-right-protection-for-your-commander/
- Ilija Miljkovac, "How Does the Commander Tax Work in EDH?" — https://draftsim.com/mtg-commander-tax-edh/

### strategy:multiplayer.zonal-commander — The Commander You Need Not Cast

Some commanders are a static ability with an optional body. Macready on
eminence: it "grants them an ability that affects the Battlefield from the
Command Zone, so they can shape games even if they are never played" —
untouchable there, which is why its designer Verhey called it (as Macready
quotes him) "The Magic Mechanic I Shouldn't Have Made!". Iñaki's rules framing
is the test: the ability "works while that creature is in the command zone or on
the battlefield", and against a pilot who never casts it "there's little you can
do". Casting it therefore buys three things at full price: a body, its
battlefield-only text, and a removal magnet. Furtado says the quiet part for the
six-drop case: "Edgar can get expensive quickly as a 6-drop, but even leaving it
in the command zone reasonably grows your board." Then the tax compounds at {2}
a trip (strategy:multiplayer.commander-insurance), "prohibitively expensive"
past three or four casts (Bockman). GAP: nobody gives the threshold at which the
body is worth the cast. Make it a comparison, not a habit — cast it when the
body closes or protects, else spend the mana on something nobody can tax (strategy:conversion.raising-the-ceiling).

Sources:
- Ben Macready, "The most powerful mechanic in MTG's Commander format returns, except now it hurts you" — https://www.wargamer.com/magic-the-gathering/eminence-sahir
- Ignacio (Iñaki), "Eminence in MTG: Rules, History, and All 6 Cards Ranked" — https://draftsim.com/eminence-mtg/
- Pedro Furtado, "Edgar Markov Commander Deck Guide" — https://draftsim.com/edgar-markov-edh-deck/
- Robert Bockman, "What is the point of Commander Tax? Rules Explained" — https://www.coolstuffinc.com/a/what-is-commander-tax-rules-06252026
- Ilija Miljkovac, "How Does the Commander Tax Work in EDH?" — https://draftsim.com/mtg-commander-tax-edh/

## strategy:deckbuilding — Deck Construction

The construction frame: 99 slots are a budget, and every inclusion is a cut.
The question is never "is this card good" but "is this the best card for this
slot given what the other 98 already do" — a card that duplicates a job you
have covered is worse than a mediocre card that covers a job you don't. Build
for the distribution, not the list: you play the hands your deck deals you, so
a deck is a probability distribution over openings (strategy:mulligans), and
the strong card you cast in a third of games loses to the fine card you cast in
all of them. Consistency is bought with slots, so price it: every tutor, ramp
piece and cantrip is a slot spent making other slots reachable. Commander
correction: singleton means you cannot build toward named cards, only toward
*classes* of effects — ramp, draw, interaction, protection, payoff — so count
functions first and fill them second (Hinds' categories survive even where his
numbers don't). Unsummoned Skull's lesson holds: decks that "do a few things,
and do them well" beat multi-purpose piles, and a synergy deck still needs a
clock it controls.

Sources:
- Cas Hinds, "Everything You Need To Know About Commander Deck Building" — https://www.coolstuffinc.com/a/everything-you-need-to-know-about-commander-deckbuilding-05292026
- Unsummoned Skull, "Top Lessons Learned About Deckbuilding" — https://commandersherald.com/top-lessons-learned-about-deckbuilding/
- Reid Duke, "The Basics of Mana" — https://magic.wizards.com/en/articles/archive/level-one/basics-mana-2015-07-06
- Patrick Chapin, "Next Level Deckbuilding" (print)

### strategy:deckbuilding.mana-base — Mana Base: Land Count & the Source Budget

Budget mana *sources*, not lands. Karsten's regression over 110 top-performing
60-card lists fits "the number of lands in a deck is given by 16 plus 3.14
times the average converted mana cost of its nonland spells" — 24 lands at a
2.40–2.72 average cost, 26 at 3.04–3.36 — and he scales to Commander by
multiplying by 99/60, which makes a 60-card deck's 25 lands worth 41.25 here.
Commander's usual 36–38 lands only works because rocks, dorks and land-ramp pay
the remainder. Dana Roach's EDHREC pull found the average deck at just over 29
lands plus 4.15 mana rocks, goldfished 100 of them, and had 26% miss the turn-3
land drop with no mana source at all — "that is terrible, and it will make you
lose games before they've even begun"; precons ship over 37 lands. His quoted
starting point (Nate Burgess): lands = 31 + colours in the commander's identity
+ the commander's mana value, counting 0-mana rocks as lands. Everything else
taxes the same budget: a utility land that can't produce the colour you need is
not a source, and Duke treats entering tapped as a real drawback — around eight
taplands is comfortable in a slow deck, near zero in an aggressive one.

Sources:
- Frank Karsten, "How Many Lands Do You Need to Consistently Hit Your Land Drops?" (mirror of the 2017 ChannelFireball article) — https://orkerhulen.dk/onewebmedia/How%20Many%20Lands%20Do%20You%20Need%20to%20Consistently%20Hit%20Your%20Land%20Drops.pdf
- Dana Roach, "Superior Numbers - Land Counts" — https://edhrec.com/articles/superior-numbers-land-counts
- Reid Duke, "Building a Mana Base" — https://magic.wizards.com/en/articles/archive/level-one/building-mana-base-2014-11-24
- David Royale, "How to Build a Great Commander Deck (4 Steps)" — https://draftsim.com/build-commander-deck-mtg/

### strategy:deckbuilding.mana-base.color-sources — Colour Sources per Pip

Karsten's numbers, C meaning an arbitrary coloured pip; his 99-card column
assumes 40 lands, on the play, casting on curve, "consistently" ≈ 90%. Single
pip: 23 sources for {C}, 21 for 1{C}, 19 for 2{C}, 17 for 3{C}, 15 for 4{C}.
Double pip: 33 for {C}{C}, 29 for 1{C}{C}, 26 for 2{C}{C}, 23 for 3{C}{C}, 22
for 4{C}{C}. Triple pip: 37 for {C}{C}{C}, 33 for 1{C}{C}{C}, 30 for 2{C}{C}{C},
28 for 3{C}{C}{C}. (60-card equivalents for {C} / {C}{C} / {C}{C}{C}: 14 / 20 /
23.) Read them as a budget, not a table: 37 sources of one colour is a
mono-coloured mana base, so triple pips and three colours do not coexist, and
the cheap pips are the expensive ones — the requirement falls fast as the spell
gets later. Only untapped sources count for turn one, and Karsten caps pure
taplands at about four before you should "rethink your deck or simply add more
lands". Duke's judgment layer, in 60-card terms: 17–18 sources for a colour you
won't keep a hand without, 14–16 for a main colour, 10–13 for a secondary, 4–7
for a splash.

Sources:
- Frank Karsten, "How Many Colored Mana Sources Do You Need to Consistently Cast Your Spells? A Guilds of Ravnica Update" (ChannelFireball, archived) — https://web.archive.org/web/20230331165535/https://strategy.channelfireball.com/all-strategy/mtg/channelmagic-articles/how-many-colored-mana-sources-do-you-need-to-consistently-cast-your-spells-a-guilds-of-ravnica-update/
- Reid Duke, "Building a Mana Base" — https://magic.wizards.com/en/articles/archive/level-one/building-mana-base-2014-11-24

### strategy:deckbuilding.ratios — Slot Ratios & the Template Genre

Templates exist to stop you from building 99 payoffs. The 8x8 method takes a
commander, 35 land slots, and "8 different kinds of effects... 8 individual
cards for each", yielding 64 spells — offered explicitly as "an initial
jumping-off point", not rules. Cas Hinds' "11 9s" partitions the 99 into 36
lands, 9 ramp, 9 draw, 9 removal and 45 theme. The Command Zone template (Wong
and Lee Kwai) is the format's default: 36–38 lands, 10–12 ramp, 10 card draw,
10–12 targeted removal, 3–4 board wipes, the rest strategy — the updated
version having roughly doubled targeted removal and cut wipes. Draftsim's
Royale lands nearby: 36–40 lands, ~10 ramp, ~10 card-advantage-or-tutor slots,
10–15 removal split between spot and sweepers. Where they go wrong: they are
counts of *functions*, not cards, so a card doing two jobs fills two slots
(Hinds counts self-mill as draw, reanimation as interaction) and a graveyard
deck legitimately runs 5 draw. Hinds' own verdict on his template is that it is
"too 'one size fits all'". Take the categories, derive the counts from the
deck's actual failure modes. For draw, shape as well as count
(strategy:card-advantage.plan-draw).

Sources:
- The 8x8 Theory, "What is the 8x8 Theory?" — https://the8x8theory.tumblr.com/what-is-the-8x8-theory
- Cas Hinds, "Everything You Need To Know About Commander Deck Building" — https://www.coolstuffinc.com/a/everything-you-need-to-know-about-commander-deckbuilding-05292026
- Commander Deck Maker, "The Command Zone Template" (written write-up of Wong & Lee Kwai's template) — https://commanderdeckmaker.com/learn/deckbuilding/command-zone-template
- David Royale, "How to Build a Great Commander Deck (4 Steps)" — https://draftsim.com/build-commander-deck-mtg/

### strategy:deckbuilding.curve — Mana Curve & the Four-Player Clock

Duke's baseline: lands are "a touch over 40% of a deck", 24–25 in 60 cards, and
"there's no 'master formula'" for the spell costs above them — the curve is
whatever your plan needs, steep only if you can pay for it. The Commander shape
is measurable: MtGDS' EDHREC data puts the modal mana value at 2, with 15.7
two-drops and 15.4 three-drops in the average deck and only ~1.5 cards at mana
value 8+; the same data shows builders run *more* cards at their commander's
mana value than the global curve (9.7 four-drops for four-mana commanders vs
9.5), not fewer. The clock is what disciplines the top end: Commander games
average 7–10 turns, so "unless the majority of your deck is mana ramp... you're
probably only going to be able to cast ONE big-mana spell per game" (Commander
Mechanic). Build to hit 2, 4 and 6 mana reliably, because four-player turn
cycles are expensive — a turn spent off-curve is three opponents' worth of
development, and one missed key turn puts the plan two to three turns behind.
Archetype shaping: aggro and critical-mass decks buy cheap enablers
(strategy:critical-mass.storm-math); ramp decks buy a payoff tier and must
count the ramp itself as curve.

Sources:
- Reid Duke, "The Basics of Mana" — https://magic.wizards.com/en/articles/archive/level-one/basics-mana-2015-07-06
- MtGDS, "Paradigm Shift - How Your Commander's Mana Value Alters Your Curve" — https://edhrec.com/articles/paradigm-shift-how-your-commanders-mana-value-alters-your-curve
- Commander Mechanic, "Dangerous Curves Ahead" — https://commandersherald.com/dangerous-curves-ahead/

### strategy:deckbuilding.redundancy-vs-tutors — Redundancy vs Tutors

Singleton kills the 4-of, so consistency has two purchases: redundancy (more
cards doing the job) or tutoring (cards that fetch it). Hypergeometric
arithmetic on 99 cards: a named card is 7.1% of your opening seven, 10.1% by
ten cards seen. An effect-class climbs slowly — 5 copies is 31% of openers, 7
is 41% (WitchPHD's 7-of, the singleton 4-of, 41.1%), 10 is 54%, and 90% by
ten cards seen costs about 20 copies. A tutor is "essentially an additional
copy of any card in our deck" (Commander Mechanic): k tutors plus the card act
like k+1 copies of *every* card. The trade: redundancy is cheap per slot and
live every game; tutoring is fewer slots but costs board impact and a turn of
tempo — Lowry, "the more tutors you play, the less actual cards you get to
play" — and Sheldon's design cost: tutors "reduce the diversity of the games
you play", making tutor count a power-level lever. So stack the classes you
want every game (ramp, draw, removal, enablers), tutor only the irreplaceable;
Nicol's start is 7-8 enablers, 10-12 enhancers. Keep rates track enabler-class
density, not tutor count (strategy:mulligans.engine-hands) — a tutor is an
enabler only if you can cast it and still deploy.

Sources:
- WitchPHD, "7x9, Every Time" — https://witchphd.substack.com/p/7x9-every-time
- Commander Mechanic, "Mechanical Engineering – Learning from Tutors" — https://commandersherald.com/mechanical-engineering-learning-from-tutors/
- Anthony Lowry, "How Many Tutors Is Too Many Tutors?" — https://www.hipstersofthecoast.com/2023/06/how-many-tutors-is-too-many-tutors/
- Kieran Sheldon, "Brew like a Game Designer: Fixing the Tutor Problem (A Defense of Toolbox Decks)" — https://commandersherald.com/brew-like-a-game-designer-fixing-the-tutor-problem-a-defense-of-toolbox-decks/
- Benjamin Nicol, "Solve the Equation - How to Tell if a Strategy Has Enough Support by Using Deck Templates" — https://edhrec.com/articles/solve-the-equation-how-to-tell-if-a-strategy-has-enough-support-using-deck-templates

### strategy:deckbuilding.threat-density — Threat Density: Finishers & Engine Pieces

Value is not a win. Gregory's diagnosis of the deck that ramps, draws and never
closes is focus — unsynergistic win conditions dilute the list — and a
four-player seat wins ~25% of games by default, so conversion is where decks
differ. Countable floor, Zupke's structure: 3-5 finishers, "at least 3
finishers in your deck", 5-7 protection slots, 20-25 flexible strategy cards.
cEDH compresses that, its finishers being combos: Eisenherz's rule is two
primary combos, "a Plan A and Plan B", never more than two *unconnected* ones —
cut the least efficient — a third only when the pieces are "exceptionally
compact". Layering reconciles them: overlap pieces, so five cards support four
combos (his sans-blue Pod example), keeping dead cards down and surviving
disruption. Engine density — Nicol asks 7-8 enablers and 10-12 enhancers
against ~25 standalone cards, and the enabler slots fail first
(strategy:mulligans.engine-hands). Separate ceiling from median: the ceiling
build runs one spectacular finisher and loses when it's answered; the median
runs three cheap ones and closes from any board. Know your number
(strategy:critical-mass.storm-math), then count how many of the 99 produce it.

Sources:
- Andy Zupke, "Building a Commander Deck - Part Two: Structure" — https://blog.cardsphere.com/building-a-commander-deck-part-two-structure/
- Learn cEDH, "How Many Combos Are Too Many?" (written lesson from Eisenherz's video) — https://learncedh.com/intermediate-course/how-many-combos-are-too-many
- Kristen Gregory, "5 Reasons Your Commander Deck Isn't Winning Games" — https://blog.cardkingdom.com/5-reasons-your-commander-deck-isnt-winning-games/
- Benjamin Nicol, "Solve the Equation - How to Tell if a Strategy Has Enough Support by Using Deck Templates" — https://edhrec.com/articles/solve-the-equation-how-to-tell-if-a-strategy-has-enough-support-using-deck-templates

### strategy:deckbuilding.typal-density — Typal Density: Lords, Bodies & the Creature Floor

Walser's floor: typal decks want 30+ creatures "because typal payoffs and
lords work best with a high saturation of creatures"; Furtado's Edgar list
runs 28, all Vampires. A lord is an anthem on a body, and both halves scale
with the bodies on board — the +1/+1 is worth the number of creatures it
touches, so the creature floor comes first and the lord count second. The
anthem half is what makes small bodies connect, and this bench measured it: a
go-wide Edgar refactor that cut every lord and anthem dropped Forge's combat
damage to players from 29.07 to 18.20 per game and won 31/400 against the
champion's 50/400, an interval excluding zero — a wide board of 1/1s is simply
blocked. DougY's lord survey adds the axes a bare +1/+1 misses: cost reduction
is "one of the best things you can get" from a lord, and keyword grants
(evasion, haste, deathtouch) turn a wide board into one that connects;
Walser's anthem rule is the same scaling read from the other side — an anthem
is "great on turn 3 and fine on turn 10" because it buffs whatever you
control. Count bodies, then count what pumps them, then read the wipe cost of
each pump (strategy:deckbuilding.typal-density.lord-exposure).

Sources:
- A.L. Walser, "Here's Exactly How Many Creatures to Put in Your Commander Deck" — https://draftsim.com/how-many-creatures-in-a-commander-deck/
- Pedro Furtado, "Edgar Markov Commander Deck Guide" — https://draftsim.com/edgar-markov-edh-deck/
- DougY, "The Top 10 Lords in Commander" — https://edhrec.com/articles/the-top-10-lords-in-commander
- A.L. Walser, "The 59 Best Anthem Effects in Magic Ranked" — https://draftsim.com/mtg-anthem/
- Mana Map, "docs/gotchas-bench.md" (this bench's Forge measurement of the Edgar go-wide refactor) — https://github.com/macrae/mana-map/blob/main/docs/gotchas-bench.md

### strategy:deckbuilding.typal-density.lord-exposure — Lords Under the Wrath: Bodies vs Noncreature Anthems

The body half of a lord is its cost. Creature lords make a typal deck "more
susceptible to board wipes and removal" (Cullen), and the lord is the target
the removal picks — a Legion Lieutenant eats the spell the payoff would have
drawn. Cullen's fix is the noncreature anthem: Icon of Ancestry, Vanquisher's
Banner, which survive the wrath — Icon "lets you convert excess mana into more
gas"; Walser's Legion's Initiative is the anthem you "cash in for some
protection against board wipes". No primer gives the lord count at which an
anthem beats a body; the trade is explicit, though — every lord past the
creature floor is +1 on every body and one more body under the wrath — so
price the marginal slot on the sweeper you assume (
strategy:resource-hedging.wrath-math), and buy the rebuild with it (
strategy:aristocrats.wipe-insurance). Menery's framing holds for the typal deck
too: with sweepers "you're either playing them or you're building your deck to
work around them – possibly both"; Furtado accepts his own because the Edgar
deck "can get back online pretty quickly after one".

Sources:
- Scott Cullen, "Tribal Cards You Need for Commander" — https://blog.cardkingdom.com/tribal-cards-you-need-for-commander/
- A.L. Walser, "The 59 Best Anthem Effects in Magic Ranked" — https://draftsim.com/mtg-anthem/
- Sheldon Menery, "Mechanics Under The Commander Hood: Board Wipes, Part One" — https://articles.starcitygames.com/articles/mechanics-under-the-commander-hood-board-wipes-part-one/
- Pedro Furtado, "Edgar Markov Commander Deck Guide" — https://draftsim.com/edgar-markov-edh-deck/

### strategy:deckbuilding.interaction-suite — Interaction Suite: Breadth, Depth & Answers to Answers

Count first. Walser's baseline is "at least 8-10 removal spells" inside a
"15- to 20-card interactive suite", built expecting only 2-3 to resolve, wipes
"one or two max" (Zupke caps at 3) — the drift is real: the Command Zone
template cut wipes 5→3-4 as the format sped up
(strategy:deckbuilding.ratios), so date any wipe count you inherit. Hinds'
observed spread is 7-15, counting counters and bounce. Breadth before depth:
three opponents mean every permanent type appears, so cover the classes —
creature, artifact, enchantment, graveyard, land — before doubling any;
catch-all removal earns its premium. Depth is a meta call: copy the answer your
table beats you with. Ceiling: McGuinness reports three mana as "essentially
the most any player should pay for interaction" in cEDH, wipes out of favour as
games speed up. Answers to answers are a separate budget — Commander Deck
Maker: 2-4 protection generic, 4-6 combo, 6-8 Voltron; counterspells do both
jobs. Stax is the hardest breadth test: a counter answers
a lock piece only on the way in, so some answers must be castable under the tax
(strategy:threat-assessment.resource-denial). Spending it:
strategy:threat-assessment.answer-economy.

Sources:
- A.L. Walser, "How Much Removal Should You Really Play in Commander?" — https://draftsim.com/edh-how-much-removal/
- Cas Hinds, "The Problem with Removal in Commander" — https://www.coolstuffinc.com/a/cashinds-seo-10232024-the-problem-with-removal-in-commander
- Harvey McGuinness, "EDHREC Guide To Interaction in cEDH" — https://edhrec.com/guides/edhrec-guide-to-interaction-in-cedh
- Commander Deck Maker, "Interaction and Protection" — https://commanderdeckmaker.com/learn/card-roles/interaction-and-protection
- Andy Zupke, "Building a Commander Deck - Part Two: Structure" — https://blog.cardsphere.com/building-a-commander-deck-part-two-structure/

### strategy:deckbuilding.archetype-selection — Commander & Archetype Selection

Choose three things at once: a commander, a plan, a bracket
(strategy:deckbuilding.power-level) — Walser puts the bracket first, since it
decides which cards are candidates. Zupke's entry points: commander-first,
strategy-first (Voltron, wheels, aristocrats, alt-win), flavour-first. They
converge, but the commander is the only card you always have access to, so it
licenses a plan rather than being one. What it licenses is narrow: colour
identity fixes the source budget
(strategy:deckbuilding.mana-base.color-sources), its mana value bends the curve
(strategy:deckbuilding.curve). Walser's distinction carries the weight: built
around your commander is not reliant on it. A deck whose only payoff sits in
the command zone hands three opponents one target; buy it back with redundant
copies (strategy:deckbuilding.redundancy-vs-tutors) or protection
(strategy:multiplayer.commander-insurance). Archetype then sets every count,
each in its own section: Commander Deck Maker's spread runs
aggro 26-32 creatures / 5-6 removal / 34-36 lands, control 12-15 removal / 5-7
wipes / 37-39 lands, combo 4-8 tutors and 4-6 protection, Voltron 12-16
equipment and auras. Bend those there, not here.

Sources:
- A.L. Walser, "9 Critical Tips for Building a Better Commander Deck" — https://draftsim.com/edh-deck-tips/
- Andy Zupke, "Building a Commander Deck - Part One: The Adventure" — https://blog.cardsphere.com/building-a-commander-deck-part-one-the-adventure/
- Brian Cain, "How to Choose Your Commander" — https://edhrec.com/articles/how-to-choose-your-commander
- Commander Deck Maker, "Ratios by Archetype" — https://commanderdeckmaker.com/learn/deckbuilding/ratios-by-archetype

### strategy:deckbuilding.power-level — Power Level: Brackets, Game Changers & Rule 0

WotC's Commander Format Panel sorts decks into five brackets, and the sharpest
line between them is expected game length: 1 Exhibition, theme over power, at
least nine turns before you win or lose; 2 Core, unoptimized and telegraphed,
at least eight; 3 Upgraded, strong synergy and one-big-turn kills off accrued
resources, at least six; 4 Optimized, "lethal, consistent, and fast", at least
four; 5 cEDH, metagame-driven, where games "could end on any turn". The one
hard gate is the Game Changers list — 53 cards as of July 2026, still beta:
Brackets 1-2 exclude them, Bracket 3 allows up to three, Brackets 4-5 are
unlimited. The rest is intent, not arithmetic. Verhey grants he "can easily
build a deck that technically meets all the rules of Core (Bracket 2) and plays
at the power level of Optimized (Bracket 4)", says of the Moxfield and
Archidekt estimators that "any estimate is just an estimate", and the panel
frames brackets as "a tool to guide pregame conversations—not an ultimate
arbiter of who can play against whom". Contents give a floor, never a verdict.
Rule zero — permission by pregame discussion — stays live at every bracket
except cEDH.

Sources:
- Wizards of the Coast, "MTG Commander Format" (official brackets and Game Changers pages) — https://magic.wizards.com/en/formats/commander
- Gavin Verhey, "Introducing Commander Brackets Beta" — https://magic.wizards.com/en/news/announcements/introducing-commander-brackets-beta
- Gavin Verhey, "Commander Brackets Beta Update – April 22, 2025" — https://magic.wizards.com/en/news/announcements/commander-brackets-beta-update-april-22-2025
- Gavin Verhey, "Commander Brackets Beta Update – October 21, 2025" — https://magic.wizards.com/en/news/announcements/commander-brackets-beta-update-october-21-2025
- Gavin Verhey, "Commander Brackets Beta Update – February 9, 2026" — https://magic.wizards.com/en/news/announcements/commander-brackets-beta-update-february-9-2026

### strategy:deckbuilding.power-level.barometers — Bracket Barometers: Combos, Extra Turns, Land Denial, Tutors

Three content barometers, and one that was deleted. Mass land denial: not
anywhere in Brackets 1-3, defined as cards that "regularly destroy, exile, and
bounce other lands, keep lands tapped, or change what mana is produced by four
or more lands per player without replacing them" — Armageddon, Ruination,
Sunder, Winter Orb, Blood Moon. Two-card infinite combos: none intended in 1-2
and none early in 3, restated in October 2025 as the turn floor, so the test is
whether a line tends to happen inside the bracket's turn count; holding it back
doesn't launder it, since "if a combo could frequently come up, it's not the
best fit for that bracket". Extra turns: none in 1, low quantities in 2-3 and
"not intended to be chained in succession or looped". Tutors: no restriction at
all any more. The original "few"/"sparse" guidance was never given a number and
was dropped in October 2025 to "rely on Game Changers to catch the most
efficient tutors" — which is why Demonic Tutor and Imperial Seal are gated and
Diabolic Tutor isn't. Brackets 4-5 restrict nothing but the banned list.

Sources:
- Gavin Verhey, "Introducing Commander Brackets Beta" — https://magic.wizards.com/en/news/announcements/introducing-commander-brackets-beta
- Gavin Verhey, "Commander Brackets Beta Update – October 21, 2025" — https://magic.wizards.com/en/news/announcements/commander-brackets-beta-update-october-21-2025
- Wizards of the Coast, "MTG Commander Format" (official brackets and Game Changers pages) — https://magic.wizards.com/en/formats/commander

### strategy:deckbuilding.cutting — The Last Ten Cards: Cutting & Iteration

Gregory locates the pressure: 36 lands plus the commander, Sol Ring, Arcane
Signet and Command Tower leave "about 58-60 slots", and a brewer's pile
routinely arrives at 90 or 100 nonland candidates. Her cuts: use a price budget
purely as a forcing device, cut the most-played "staples" first and make the
synergy pick argue for itself, brew only with what's at home. Milan's mechanic:
fill a template to the letter, set the lands aside untouched, then change one
in, one out so you never lose count. Then stop building and play — he cut ~150
candidates to reach 100, and reckons seven or eight of those survive six months
later. The math says late cuts are cheap: an 11th copy of an effect buys about
four points of opening-hand probability over the 10th, 54% to 57%
(strategy:deckbuilding.redundancy-vs-tutors), so cut duplicated function before
any uncovered class, and cut the ceiling card before the median one
(strategy:deckbuilding.threat-density). Iterate against play, not paper —
Gregory: "it's far better to playtest your rough draft first". Keep the
near-misses as a sideboard and swap them in between games.

Sources:
- Kristen Gregory, "5 Ways to Cut Cards Easier in Commander" — https://blog.cardkingdom.com/5-ways-to-cut-cards-easier-in-commander/
- Roman Milan, "New Player Guide - How to Cut Cards From Your Commander Decks" — https://edhrec.com/articles/new-player-guide-how-to-cut-cards-from-your-commander-decks
- Commander Deck Maker, "The Command Zone Template" (its build loop ends "build, playtest, adjust") — https://commanderdeckmaker.com/learn/deckbuilding/command-zone-template

### strategy:deckbuilding.budget — Budget, Proxies & Where Money Actually Binds

Budget binds in one place: the mana base. Zupke built a five-colour Sisay deck
to a $50 total with no card over $1 and found the fixing, not the spells, was
the constraint — his tier is Command Tower and Exotic Orchard, ~$1 pain lands,
BFZ duals carrying basic types (Farseek finds them), and sub-dollar tri-lands
you pay for in tempo rather than dollars (strategy:deckbuilding.mana-base). His
hard rule: "You should never have to pay mana to play your lands" — Rupture
Spire and its clones aren't budget, just bad. Bucks prices the same tier at "$1
or less", rating a condition to enter untapped paramount. For spells, Levin
filters Scryfall at two dollars or less against the rules text he wants:
"budget replacements aren't going to be as good as their non-budget
counterparts, but rarely are they going to be as bad as their difference in
price might suggest." Gregory uses a budget as a cutting device — it makes you
less precious about staples. Price is not power: proxies are normalized, and
Carrozza's point is that "the issue... is the power level and that's the
conversation that should be had" — a proxied Game Changer still raises your
bracket floor (strategy:deckbuilding.power-level).

Sources:
- Andy Zupke, "Now Is the Best Time There's Ever Been for Budget Commander" — https://blog.cardsphere.com/why-now-is-the-best-time-theres-ever-been-for-budget-edh/
- Benjamin Levin, "Shower Thoughts: Budget Deck Building Guide" — https://commandersherald.com/shower-thoughts-budget-deck-building-guide/
- Tyler Bucks, "The Big List of Budget Dual Lands in Commander" — https://edhrec.com/articles/the-big-list-of-budget-dual-lands-in-commander
- Kristen Gregory, "5 Ways to Cut Cards Easier in Commander" — https://blog.cardkingdom.com/5-ways-to-cut-cards-easier-in-commander/
- Mike Carrozza, "Am I The Bolas? – Moving to EDHREC (A Last One about Proxies and Power Level)" — https://commandersherald.com/am-i-the-bolas-moving-to-edhrec-a-last-one-about-proxies-and-power-level/

## strategy:sixty-card — Sixty-Card Constructed: The 1v1 Baseline

The pod sections above are corrections to this baseline, which is what a
Standard deck is audited against. The shape: one opponent at 20 life, a
seven-card hand, four copies of a card, best-of-three with fifteen cards that
change the deck after game one, and a play/draw choice where playing first is
correct "in such an overwhelming majority of cases" (Duke). The clock is Zvi's
Fundamental Turn: "For beatdown or combination decks, the FT is the turn you
kill your opponent"; for control, the turn "the deck's strategy begins to work
and you make up for any early disadvantage" — he put Standard's at four in
2000. Every count below follows: fewer turns, a lower curve; one opponent,
answers without breadth; 20 life, a tapped land is a real tax; a sideboard,
the sixty need not answer everything. Quotable target: a 60-card deck is built
for one opponent at 20 life, on a fundamental turn of three to five, with four
copies of its best cards and fifteen sideboard slots to change shape after
game one. The mana maths are not restated: Karsten's regression and source
counts and Duke's per-colour bands live in strategy:deckbuilding.mana-base and
strategy:deckbuilding.mana-base.color-sources.

Sources:
- Zvi Mowshowitz, "Clear The Land And The Fundamental Turn" — https://articles.starcitygames.com/articles/clear-the-land-and-the-fundamental-turn/
- Reid Duke, "Play or Draw" — https://magic.wizards.com/en/articles/archive/level-one/play-or-draw-2015-03-16
- Reid Duke, "The Sideboard" — https://magic.wizards.com/en/articles/archive/level-one/sideboard-2015-08-10
- Reid Duke, "Building a Mana Base" — https://magic.wizards.com/en/articles/archive/level-one/building-mana-base-2014-11-24

### strategy:sixty-card.interaction — Interaction by Archetype

Count lists, not slogans. Duke's Level One examples: the W/B aggro list runs
six instant removal spells and four Thoughtseize among 39 nonland cards, his
mono-red ten burn spells; the Abzan midrange list thirteen (four Thoughtseize,
four Abzan Charm, three Hero's Downfall, two Utter End) among 36; the W/U
control list is "a mix of permission spells and removal spells" over two
creatures. Soorani's control rule: "8-10 interactive spells early on to set up
for our big haymakers", and the "six to seven removal spells" era is over;
Walser likes "10-12 counterspells in most of my 60-card control decks".
Vrooman's midrange read: "8-10 pieces of spell-based creature removal main
deck". Why lower than Commander's 10-15 removal plus 3-4 wipes
(strategy:deckbuilding.ratios): one opponent is one board, so Duke's rule
holds — "For every job that needs to be done, choose the cheapest (in terms of
mana) option available" — and breadth is bought in the sideboard. Quotable
target (judgement from the lists above): an aggressive 60-card deck runs 8 to
12 cheap interactive spells, a midrange deck 10 to 14, and a control deck 16
to 22 counting counterspells, removal and sweepers together.

Sources:
- Reid Duke, "Aggro Decks" — https://magic.wizards.com/en/articles/archive/lo/aggro-decks-2014-09-29
- Reid Duke, "Midrange Decks" — https://magic.wizards.com/en/articles/archive/lo/midrange-decks-2014-10-27
- Reid Duke, "Control Decks" — https://magic.wizards.com/en/articles/archive/lo/control-decks-2014-10-06
- Shaheen Soorani, "How To Build A Control Deck" — https://articles.starcitygames.com/articles/how-to-build-a-control-deck/
- A.L. Walser, "The Complete Guide to Control Decks in Magic: the Gathering" — https://draftsim.com/mtg-control-deck/
- Steve Vrooman, "The Complete Guide to Midrange Decks in Magic: the Gathering" — https://draftsim.com/mtg-midrange/

### strategy:sixty-card.creatures — Creature Count by Archetype

Duke's W/B aggro list runs 29 creatures with "a full 22 spells" at one mana
and nothing above three; Bartholdi's rule is "An aggro deck usually needs 8–12
one-drops and 8–12 two-drops"; Royale's sample is 28 creatures; Lax notes "Tom
Ross was successful with Boss Sligh by playing almost all one-drops" and asks
for threats that are "sticky or redundant" rather than ones that trade
one-for-one into a midrange deck drawing the same number of cards. Midrange
halves it — Duke's Abzan list has 16 creatures beside 7 planeswalkers — and
control nearly zeroes it: two creatures in Duke's W/U list, and Soorani's "Win
conditions cannot be a huge portion of the deck." The count does not define
the archetype: PVDDR's 31-creature mono-white is, in his words, "not a
traditional aggro deck, it's a midrange deck." What the count buys is in
strategy:sixty-card.threat-density. Quotable target (judgement from the lists
above): an aggressive 60-card deck runs 24 to 30 creatures, 16 to 24 of them
at one or two mana; a midrange deck 12 to 18 creatures plus noncreature
threats; a control deck 0 to 4.

Sources:
- Reid Duke, "Aggro Decks" — https://magic.wizards.com/en/articles/archive/lo/aggro-decks-2014-09-29
- Maria Bartholdi, "How to Build: Aggro" — https://magic.wizards.com/en/news/feature/how-build-aggro-2017-02-02
- David Royale, "The Ultimate Guide to Aggro Decks in Magic: The Gathering" — https://draftsim.com/mtg-aggro-deck/
- Ari Lax, "Building Good Aggro Decks" — https://articles.starcitygames.com/articles/building-good-aggro-decks/
- Reid Duke, "Midrange Decks" — https://magic.wizards.com/en/articles/archive/lo/midrange-decks-2014-10-27
- Reid Duke, "Control Decks" — https://magic.wizards.com/en/articles/archive/lo/control-decks-2014-10-06
- Shaheen Soorani, "How To Build A Control Deck" — https://articles.starcitygames.com/articles/how-to-build-a-control-deck/
- Paulo Vitor Damo da Rosa, "Mono-White Aggro Deck Guide (Alchemy)" — https://pvddr.substack.com/p/mono-white-aggro-deck-guide-alchemy

### strategy:sixty-card.threat-density — Threat Density: The Second Wave After a Sweeper

The threat count is set by the sweeper, not the curve. In-game, commit no more
than wins before the wrath — Mullen: "you don't want to expose more than one
resilient threat at a time to a Settle the Wreckage"; Royale: "apply steady
pressure with just a couple of high-damage creatures and keep the rest of your
hand safe in case someone plays a sweeper" — and Duke's sweeper article has
Whisperwood Elemental "as insurance" and threats diversified into
"Planeswalkers, dash creatures". The construction consequence (the doc's
reading): if you deploy two or three, eat the wrath, and must redeploy two or
three to keep the clock, the second wave must be *in hand*, and only density
puts it there — 24+ bodies in sixty, where 12 leaves the rebuild in the
library. Spend the slots on "sticky or redundant" threats (Lax), and hold the
best threat for the second sweeper (strategy:resource-hedging.wrath-math).
Commander buys the rebuild with draw instead
(strategy:aristocrats.wipe-insurance). Quotable target: an aggressive 60-card
deck must keep a clock after one sweeper — enough creatures that two waves of
two or three threats are in hand by the midgame, a count in the mid-twenties,
not the teens.

Sources:
- Collins Mullen, "The Guide To W/B Aggro" — https://articles.starcitygames.com/articles/the-guide-to-w-b-aggro/
- David Royale, "The Ultimate Guide to Aggro Decks in Magic: The Gathering" — https://draftsim.com/mtg-aggro-deck/
- Reid Duke, "Board Sweepers" — https://magic.wizards.com/en/articles/archive/level-one/board-sweepers-2015-06-22
- Ari Lax, "Building Good Aggro Decks" — https://articles.starcitygames.com/articles/building-good-aggro-decks/

### strategy:sixty-card.curve — The Curve for a Twenty-Life, Seven-Card Game

Build the curve to the Fundamental Turn, then count lands back
(strategy:sixty-card.curve.lands). Shvartsman's original Sligh breakdown: 9-13
one-drops, 6-8 two-drops, 3-5 threes, 1-3 fours, 2-3 X spells, 8-10 burn —
every mana spent from turn one, because a 20-life opponent dies by turn five
if it is. Duke on play/draw: play first "in such an overwhelming majority of
cases", and on the draw "you're drawing *first,* not necessarily drawing
*extra*" — so the seven-card hand must hold the turn-one and turn-two plays,
which is why aggro stacks one- and two-drops (strategy:sixty-card.creatures)
and caps at three. Midrange centres on two to four, topping at five or six;
control inverts it, Soorani's "big haymakers" sitting above "8-10 interactive
spells early on". Duke's "no 'master formula'" holds
(strategy:deckbuilding.curve); the 60-card difference is that every slot above
the FT must win on its own. Quotable target (judgement): an aggressive 60-card
deck puts 16 to 24 cards at one or two mana and nothing above four; a midrange
deck centres on two to four with a top end of five or six; a control deck runs
its interaction at one to three and its finishers above its opponent's FT.

Sources:
- Alex Shvartsman, "Famous Red Decks in Magic History" — https://magic.wizards.com/en/articles/archive/feature/famous-red-decks-magic-history-2004-07-22-0
- Zvi Mowshowitz, "Clear The Land And The Fundamental Turn" — https://articles.starcitygames.com/articles/clear-the-land-and-the-fundamental-turn/
- Reid Duke, "Play or Draw" — https://magic.wizards.com/en/articles/archive/level-one/play-or-draw-2015-03-16
- Reid Duke, "Aggro Decks" — https://magic.wizards.com/en/articles/archive/lo/aggro-decks-2014-09-29
- Shaheen Soorani, "How To Build A Control Deck" — https://articles.starcitygames.com/articles/how-to-build-a-control-deck/
- Reid Duke, "The Basics of Mana" — https://magic.wizards.com/en/articles/archive/level-one/basics-mana-2015-07-06

### strategy:sixty-card.curve.lands — Land Count by Archetype

Lands follow the turn you must hit on time, Lax's rule: "If you are going to
hit your fourth land on time for some big whammy ... you are going to want 23
or 24 lands"; capping at three mana, "20 to 22"; a deck that "can run off one
mana sometimes" sits in "the seventeen to nineteen range". The bands agree:
Bartholdi, "An aggro deck usually plays between 20–23 lands"; Royale's
Standard read, "Aggressive decks tend to run from 23 to 24 lands, midrange
decks run from 24 to 25 lands and control decks run at least 26", with Pioneer
a land lighter (tempo ~18, aggro 22-23, midrange ~24, control 26-27); Duke's
control list "26!"; Soorani, "All control decks operate on 26-27 lands";
Vrooman's midrange "25-26 lands in slower 60-card formats"; Walser's 27-29 is
the high outlier. Karsten's regression (strategy:deckbuilding.mana-base) is
the check on the band, not the start — his 2017 fit gives 24 at a 2.4-2.7
average cost and 26 at 3.0-3.4. Quotable target (judgement from the sources
above): an aggressive 60-card deck runs 20 to 24 lands, a midrange deck 24 to
26, a control deck 26 to 28, and the count is set by the turn the deck must
hit on time, not by the average mana value alone.

Sources:
- Ari Lax, "Building Good Aggro Decks" — https://articles.starcitygames.com/articles/building-good-aggro-decks/
- Maria Bartholdi, "How to Build: Aggro" — https://magic.wizards.com/en/news/feature/how-build-aggro-2017-02-02
- David Royale, "How Many Lands Should You Really Play in Magic?" — https://draftsim.com/mtg-how-many-lands/
- Reid Duke, "Control Decks" — https://magic.wizards.com/en/articles/archive/lo/control-decks-2014-10-06
- Shaheen Soorani, "How To Build A Control Deck" — https://articles.starcitygames.com/articles/how-to-build-a-control-deck/
- Steve Vrooman, "The Complete Guide to Midrange Decks in Magic: the Gathering" — https://draftsim.com/mtg-midrange/
- A.L. Walser, "The Complete Guide to Control Decks in Magic: the Gathering" — https://draftsim.com/mtg-control-deck/
- Frank Karsten, "How Many Lands Do You Need to Consistently Hit Your Land Drops?" (mirror of the 2017 ChannelFireball article) — https://orkerhulen.dk/onewebmedia/How%20Many%20Lands%20Do%20You%20Need%20to%20Consistently%20Hit%20Your%20Land%20Drops.pdf

### strategy:sixty-card.taplands — Taplands at Twenty Life

A tapped land costs a turn's mana once; on a fundamental turn of four that is
a quarter of the game, so the 60-card band sits far below Commander's. Duke
draws it by archetype: "In a super-aggressive Red-White Weenie deck in
Standard, you'd hope to play with no lands that enter the battlefield tapped
(or at least a very small number)", while "a slow Abzan deck in Standard ...
can play eight tapped lands pretty comfortably (and ten or twelve a little
uncomfortably)". Karsten caps pure taplands at about four before you rethink
the deck or add lands (strategy:deckbuilding.mana-base.color-sources). The
dodge is the untapped land with upside — PVDDR runs three Eiganjo because "the
second and third copies of Eiganjo rarely hurt you" — so audit the class, not
the count. Commander's "around eight taplands is comfortable in a slow deck"
(strategy:deckbuilding.mana-base) is 40 life and a 7-10 turn game refunding
the tempo. Quotable target (judgement from Duke and Karsten): an aggressive
60-card deck runs zero to four lands that enter tapped, a midrange deck four
to eight, a control deck eight to twelve, and each one is a turn of the
opponent's clock that twenty life does not refund.

Sources:
- Reid Duke, "Building a Mana Base" — https://magic.wizards.com/en/articles/archive/level-one/building-mana-base-2014-11-24
- Frank Karsten, "How Many Colored Mana Sources Do You Need to Consistently Cast Your Spells? A Guilds of Ravnica Update" (ChannelFireball, archived) — https://web.archive.org/web/20230331165535/https://strategy.channelfireball.com/all-strategy/mtg/channelmagic-articles/how-many-colored-mana-sources-do-you-need-to-consistently-cast-your-spells-a-guilds-of-ravnica-update/
- Paulo Vitor Damo da Rosa, "Mono-White Aggro Deck Guide (Alchemy)" — https://pvddr.substack.com/p/mono-white-aggro-deck-guide-alchemy
- Zvi Mowshowitz, "Clear The Land And The Fundamental Turn" — https://articles.starcitygames.com/articles/clear-the-land-and-the-fundamental-turn/

### strategy:sixty-card.sideboard — The Fifteen: Answers to the Metagame

The sideboard is "at most fifteen cards" brought in "to tailor their deck
toward winning the next game" (Duke) — answers to the metagame, not more
maindeck. Sam Black's taxonomy: hate cards, "generally narrow but highly
effective bullets"; efficient answers, cheap and against classes that recur;
tuning and juking cards. His frequencies are the ratio: "If you expect to
bring hate cards in around 10% of the time, you can often expect to bring
efficient answers in 30% of the time." Duke's split is "versatile" cards that
are "minor upgrades in a lot of matchups" against "hate cards—single cards
that are extremely effective at beating a particular deck, color, or
strategy". Black's 2009 warning stands: trimming the maindeck is not a plan,
because in games two and three "you're presenting a deck". Build it as 75
(Duke): ideal lists for every matchup, then make the unique cards add up. Per
matchup: strategy:sixty-card.sideboard.plan. Quotable target (judgement from
Black's frequencies): most of a 15-card sideboard is cheap answers that come
in about a third of matches, a minority is hate cards that come in about a
tenth, and every card has a named matchup and a named cut.

Sources:
- Reid Duke, "The Sideboard" — https://magic.wizards.com/en/articles/archive/level-one/sideboard-2015-08-10
- Sam Black, "The Art Of Sideboarding" — https://articles.starcitygames.com/articles/the-art-of-sideboarding/
- Sam Black, "Black Magic – Sideboarding" — https://articles.starcitygames.com/articles/black-magic-sideboarding/

### strategy:sixty-card.sideboard.plan — The Plan per Matchup

A sideboard plan is written in advance and sized by the maindeck's weak cards,
not the sideboard's strong ones. Duke's count: "go through your main deck and
count the number of cards that are dispensable and that you're unhappy with.
This is a good target for how many sideboard cards you ought to have" — per
matchup, so three dead cards against aggro means three removal spells, not
six. What comes out: "Cut the weakest cards or the cards that most depend on
synergy with other cards" (Black), because "sideboarded games are slower, more
interactive" and a card that "only really does something if the rest of what
you're doing is working perfectly" is a liability. What stays: "your creature
count, your mana curve" (Duke) — boarding out the curve changes the deck's
role (strategy:whos-the-beatdown). Then "reevaluate your sideboarding between
Games 2 and 3" (Duke): beat their post-board deck, not their game-one list.
Quotable target: every sideboard card names the matchup it comes in for and
the maindeck card it replaces, the count per matchup equals the maindeck's
dispensable cards in that matchup, and the swap preserves the creature count
and curve.

Sources:
- Reid Duke, "Sideboard Plans" — https://magic.wizards.com/en/articles/archive/level-one/sideboard-plans-2015-03-09
- Reid Duke, "The Sideboard" — https://magic.wizards.com/en/articles/archive/level-one/sideboard-2015-08-10
- Sam Black, "The Art Of Sideboarding" — https://articles.starcitygames.com/articles/the-art-of-sideboarding/

## strategy:schools — Schools of Thought & Canonical Literature

The canon this document synthesizes. Mike Flores' "Who's the
Beatdown?" (1999, The Dojo) founded role-assignment theory; Adrian Sullivan's
Philosophy of Fire (1999, canonized by Flores) founded resource-conversion
theory. Reid Duke's Level One course (Wizards, 2014-2015) is the consensus
foundation for the resource pillars — card advantage, tempo, combat,
sequencing, risk — and most 1v1 grounding here cites it. Patrick Chapin's "Next
Level Magic" systematized information play and metagaming; Paulo Vitor Damo da
Rosa's corpus (archived at his Substack) is the playing-to-win reference. Deck
construction has a different shape: Karsten's regressions and per-pip source
tables are the mathematical spine, above which sits a template genre — 8x8, the
Command Zone template, Hinds' 11 9s — counts as priors, not law, plus EDHREC's
data pulls. Commander theory is younger and article-borne: EDHREC, Commander's
Herald, Card Kingdom and Cardsphere carry its threat-assessment, politics and
construction work. Power level alone has a primary: WotC's Commander Format
Panel. The Command Zone podcast popularized much of it but is cited only
through written sources; episodes are not directly citable.

Sources:
- Mike Flores, "Who's the Beatdown?" — https://articles.starcitygames.com/premium/whos-the-beatdown/
- Reid Duke, "Level One: The Full Course" — https://magic.wizards.com/en/news/feature/level-one-full-course-2015-10-05
- Mike Flores, "Life and Cards I: Philosophy of Fire" — https://magic.wizards.com/en/news/making-magic/life-and-cards-i-philosophy-fire-2014-04-28
- Patrick Chapin, "Next Level Magic" (print)
- Paulo Vitor Damo da Rosa, "PVDDR's Articles" (Substack archive) — https://pvddr.substack.com/archive
- Frank Karsten, "How Many Lands Do You Need to Consistently Hit Your Land Drops?" (mirror of the 2017 ChannelFireball article) — https://orkerhulen.dk/onewebmedia/How%20Many%20Lands%20Do%20You%20Need%20to%20Consistently%20Hit%20Your%20Land%20Drops.pdf
- Commander Deck Maker, "The Command Zone Template" (written write-up of Wong & Lee Kwai's template) — https://commanderdeckmaker.com/learn/deckbuilding/command-zone-template
- Wizards of the Coast, "MTG Commander Format" (official brackets and Game Changers pages) — https://magic.wizards.com/en/formats/commander
- EDHREC, "Articles" — https://edhrec.com/articles
- The Command Zone (YouTube channel; episodes not directly citable) — https://www.youtube.com/@commandzone
