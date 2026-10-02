# Strategy Companion Changelog

Every research pass appends one dated entry. Bullets are mechanically validated
(`manamap pilot validate-strategy`): each starts `added|amended|renamed|deprecated
strategy:<id>` and added/amended IDs must exist in strategy.md.

## 2026-07-24 — initial seed from founder baseline

- added strategy:card-advantage — four-pillars baseline
- added strategy:tempo — four-pillars baseline
- added strategy:life-as-resource — four-pillars baseline
- added strategy:threat-assessment — four-pillars baseline
- added strategy:whos-the-beatdown — Flores role-assignment framework
- added strategy:pivot-point — role flips and timing
- added strategy:information — inference and bluff management
- added strategy:combat-math — turn-cycle-ahead combat modeling
- added strategy:resource-hedging — loss aversion and playing to win
- added strategy:multiplayer — Commander corrections to the 1v1 frameworks
- added strategy:multiplayer.asymmetry — the 1-to-3 disadvantage
- added strategy:multiplayer.politics — table negotiation norms
- added strategy:multiplayer.threat-deflection — managing perceived threat
- added strategy:multiplayer.pivot-window — timing the win attempt
- added strategy:multiplayer.pod-management — pacing the pod as a system
- added strategy:schools — canonical literature index

Note: seeded only with sources confidently attributable (Flores, Duke, Chapin,
PVDDR, Command Zone, EDHREC); several titles from the baseline conversation
could not be confirmed and were omitted pending the first research pass, which
must verify or repair every URL.

## 2026-07-24 — first research pass: URL verification and deep expansion

- amended strategy:card-advantage — exchange-counting habit, Duke's game-stage balance rule, Commander correction; sources upgraded to verified individual Level One lessons plus Rice
- added strategy:card-advantage.virtual — virtual card advantage: blanked cards, live-card counting, Commander tripling (Duke, Flores)
- amended strategy:tempo — Duke's board-presence definition, initiative, tempo-vs-CA game-stage reading, Commander correction
- added strategy:tempo.sequencing — whole-turn planning, gather-first/reveal-last ordering, last-possible-moment timing with its two exceptions (Duke)
- amended strategy:life-as-resource — point-appreciation framing, Commander 40-life/commander-damage/infect correction
- added strategy:life-as-resource.philosophy-of-fire — Sullivan/Flores cards-as-damage frame, defender's inversion, Commander collapse of raw burn math
- amended strategy:threat-assessment — Duke's threats-vs-answers asymmetry, three ranking axes, Commander mutual-assessment correction
- added strategy:threat-assessment.answer-economy — removal-budget discipline, Rice's hold-until-pointed-at-you rule, Walser's mistake list, rotting-answers caveat
- amended strategy:whos-the-beatdown — Flores' opening quote, Duke's operational role tests, seat-relative Commander correction
- added strategy:whos-the-beatdown.metagame-clock — inevitability defined, winner's-circle metagame reading, information cascades, turn-15 table test
- amended strategy:pivot-point — turning the corner, simplify-from-ahead/complicate-from-behind posture, Commander cross-ref
- amended strategy:information — reveal-last ordering tied to sequencing, Commander table-talk correction; sources verified
- added strategy:information.range-tells — range pruning, weighting inaction, representing, pace tells, multi-game table image
- amended strategy:combat-math — Duke's attack/block/trick baselines, fear-based leaks, combat-as-diplomacy Commander correction; unverifiable TCGplayer archive URL removed
- added strategy:combat-math.racing — clock counting, chump-block pricing, banked damage, two-player-subgame Commander correction
- added strategy:combat-math.probability — N/K outs arithmetic, Karsten's hypergeometric method (working mirror URL), playing-scared test, singleton flattening
- amended strategy:resource-hedging — Duke's safe-vs-scared diagnostic, bias correction; PVDDR attribution repaired to live Substack archive
- added strategy:resource-hedging.playing-to-outs — Duke's play-as-if-it's-coming rule, Severa's created outs and can't-beat discipline, closing THEIR outs from ahead
- added strategy:resource-hedging.wrath-math — commit-exactly-enough sizing, resilience over restraint, worst-first rebuild, assume-the-wrath Commander default
- amended strategy:multiplayer — Command Zone channel citation replaced with verified written sources (Rice, Walser, Krell, EDHREC)
- amended strategy:multiplayer.asymmetry — Rice's bystander-profit math and hold-removal rule, board quality over quantity; sourced to verified articles
- amended strategy:multiplayer.politics — Nicol's game-theory reading of political cards, Hinds' deal-craft case studies; sourced to verified EDHREC articles
- amended strategy:multiplayer.threat-deflection — Krell's quiet-combo warning read in reverse, Walser's political-exploitation label; verified sources
- amended strategy:multiplayer.pivot-window — Krell's does-it-win-if-it-resolves standard applied to your own attempt; verified sources
- amended strategy:multiplayer.pod-management — Krell's stage-by-stage read, Walser's continuous recalibration; verified sources
- amended strategy:schools — added Sullivan, Karsten, PVDDR Substack; all URLs verified live; Command Zone marked not directly citable (video)

Verification note: all Wizards Level One lesson URLs, both StarCityGames
articles, both EDHREC articles, Draftsim, Card Kingdom, and Brainstorm Brewery
fetched live this pass. Repaired: PVDDR's TCGplayer Infinite archive URL
(301-redirects to a JS shell, content unverifiable) replaced with his Substack
archive; Karsten's ChannelFireball original is offline, cited via a working PDF
mirror; The Command Zone channel URL replaced by written sources per the
no-video citation rule.

## 2026-07-24 — second research pass: goblin-storm strategic-frame gaps

- added strategy:threat-assessment.resource-denial — stax/tax/lock taxonomy (LaPage), racing the lock's assembly and its slow win conversion (McGuinness), parity framing (Johnson), naming and protecting narrow outs
- added strategy:critical-mass — Duke's linear-strategies frame: threshold effects, flexibility-for-power tradeoff, protecting vs denying critical mass, Commander redundancy/deflection correction
- added strategy:critical-mass.storm-math — Girten's mana/hand/payoff storm balance, go-decision arithmetic on banked resources, Karsten hypergeometric enabler density, ceiling-vs-median pacing, forced half-go
- added strategy:mulligans — Duke's three-part mulligan course: odds-only question, 2-5 lands baseline with archetype/matchup/dependency overrides, below-six collapse, PVDDR hand-as-a-plan drill
- added strategy:mulligans.engine-hands — key-card mulligans, payoff-no-enabler failure, enabler/payoff/glue classification, enabler-side asymmetry, free-first-mulligan and enabler-class Commander correction
- added strategy:multiplayer.commander-insurance — Cullen's protection taxonomy and criticality test, Miljkovac's recast-tax math, protect-vs-race-and-recast as a priced decision

Verification note: all six new sections' URLs fetched live this pass (three
Level One mulligan lessons, Linear Strategies, PVDDR Substack post, two
Commander's Herald articles, The Mana Base, CoolStuffInc, Card Kingdom,
Draftsim); Karsten citation reuses the already-verified PDF mirror. Sought but
unusable: theepicstorm.com theory archive (HTTP 403 bot block), EDH wiki Storm
page (HTTP 402), Goonhammer "Commander 102" (page fetched empty) — all omitted
rather than cited blind.

## 2026-07-25 — third research pass: opening the deck-construction pillar

- added strategy:deckbuilding — construction frame: slots as a budget, best-card-for-this-slot over is-this-card-good, deck as a probability distribution, consistency priced in slots, Commander's build-toward-classes-of-effects correction (Hinds, Unsummoned Skull, Duke, Chapin)
- added strategy:deckbuilding.mana-base — land count as a mana-source budget: Karsten's 16 + 3.14×avg-MV regression, his ×99/60 Commander scaling (25 lands → 41.25), Roach's EDHREC 29-lands/4.15-rocks average and 26% turn-3 miss rate, Burgess's 31+colours+commander-MV formula, tapland and dead-utility-land tax
- added strategy:deckbuilding.mana-base.color-sources — Karsten's 99-card per-pip source counts (23/33/37 for C/CC/CCC and the whole curve between), stated assumptions, the four-tapland cap, Duke's essential/main/secondary/splash tiers; split out of mana-base to stay inside the embedding window
- added strategy:deckbuilding.ratios — the template genre with its actual numbers (8x8's 8×8+35, Hinds' 11 9s, the Command Zone 36-38/10-12/10/10-12/3-4, Draftsim's 36-40/10/10/10-15), plus the failure modes: counts are functions not cards, and Hinds' own "too one size fits all" verdict
- added strategy:deckbuilding.curve — Duke's 40%-lands/no-master-formula baseline, MtGDS' EDHREC curve data (mode 2, 15.7 two-drops, ~1.5 at MV 8+, commander-MV overweighting), the 7-10 turn Commander clock and one-big-spell limit, four-player turn-cycle cost of an off-curve turn

Verification note: every URL in the five new sections was fetched this pass.
Karsten's colored-sources article is offline at ChannelFireball (the live
channelfireball.com/tcgplayer URLs return a JS shell with no article text), so
the 99-card table is cited to a Wayback snapshot fetched and parsed directly;
the land-drops numbers were re-extracted from the already-verified PDF mirror
rather than trusted to summary. Sought but unusable: MTGGoldfish "Brewer's
Minute: Opportunity Cost in Deck Building" (HTTP 403), EDH Wiki's Command Zone
Template page (Cloudflare interstitial), the EDHREC page for Command Zone
episode 658 (show notes only, no written numbers — the video rule forbids
citing the episode itself), manabased.substack.com "30 lands is enough"
(satire, no data).

## 2026-07-25 — fourth research pass: consistency, threat count, interaction budget

- added strategy:deckbuilding.redundancy-vs-tutors — first treatment of tutors in the corpus: the two purchases of singleton consistency, hypergeometric costs on 99 cards (1 copy = 7.1% of openers, 5 = 31%, 7 = 41%, 10 = 54%, ~20 for 90% by ten cards seen), WitchPHD's 7-of-as-a-4-of, the tutor-as-extra-copy identity (k tutors ⇒ k+1 copies of every card), Lowry's fewer-real-cards/tempo cost, Sheldon's game-diversity cost as a power-level lever, Nicol's 7-8 enablers / 10-12 enhancers, and the tie-back to strategy:mulligans.engine-hands
- added strategy:deckbuilding.threat-density — the countable answer to "know your number": Zupke's 3-5 finishers (min 3), 5-7 protection, 20-25 flex; Eisenherz's two-primary-combos rule and layering (5 cards → 4 combos); Nicol's engine-piece counts; Gregory's focus diagnosis and the ~25% default four-player win share; ceiling-vs-median build split, extending strategy:critical-mass.storm-math
- added strategy:deckbuilding.interaction-suite — slot counts for answers: Walser's 8-10 removal inside a 15-20-card interactive suite and 2-3 resolving per game, Zupke's 3-wipe cap, Hinds' observed 7-15 spread counting counters/bounce, breadth-by-permanent-class before depth, McGuinness's three-mana efficiency ceiling in cEDH, Commander Deck Maker's 2-4/4-6/6-8 protection split, and the stax-resistance corollary; cross-references strategy:threat-assessment.answer-economy (spending) and .resource-denial rather than repeating them

Verification note: every URL in the three new sections was fetched this pass
(Substack, Commander's Herald ×2, Hipsters of the Coast — permalink resolved to
its canonical /2023/06/ URL, EDHREC article + cEDH guide, Cardsphere, Card
Kingdom, Draftsim, CoolStuffInc, Commander Deck Maker, Learn cEDH). The Learn
cEDH lesson is a written write-up of Eisenherz's video and is cited as the
write-up, per the no-video rule. The per-copy percentages in
redundancy-vs-tutors are hypergeometric arithmetic computed over a 99-card
library (Karsten's method, already cited in the pillar) and corroborated
against WitchPHD's published 41.1% figure for a 7-of. Sought but unusable:
Commander Deck Maker's "Interaction and Protection" page carries no author or
date (cited to the site, as with its Command Zone Template page); the "how many
removal spells" search space is dominated by unattributed SEO/AI content
(tappeddecks, grimdeck, krakenopus, geekydomain, cultureofgaming, proxyking,
abyssproxyshop, manacove, mtg-agents) — all omitted deliberately; no
"Superior Numbers" instalment on removal or tutor counts appears to exist, so
there is still no EDHREC-scale observed-average number for interaction, only
prescriptive ones.

## 2026-07-25 — fifth research pass: closing the deckbuilding namespace

- added strategy:deckbuilding.archetype-selection — commander/plan/bracket chosen together: Zupke's commander-first, strategy-first and flavour-first entry points, the commander as the only card you always have access to, Walser's built-around vs reliant-on distinction, and Commander Deck Maker's per-archetype spreads (aggro 26-32 creatures / 5-6 removal / 34-36 lands, control 12-15 removal / 5-7 wipes / 37-39 lands, combo 4-8 tutors + 4-6 protection, Voltron 12-16 equipment and auras); absorbs the archetype-varies clauses by cross-referencing .curve, .threat-density and .interaction-suite rather than restating them
- added strategy:deckbuilding.power-level — WotC's bracket system from the official pages and the Verhey announcements: the five brackets with their expected-turn floors (9/8/6/4/any), the Game Changers gate (0 in Brackets 1-2, up to 3 in Bracket 3, unlimited in 4-5; 53 cards as of July 2026, system still labelled beta), Verhey's "any estimate is just an estimate" on third-party calculators, the panel's "tool to guide pregame conversations—not an ultimate arbiter" framing, and rule zero being live at every bracket except cEDH
- added strategy:deckbuilding.power-level.barometers — the deck-contents barometers split out of the parent to stay inside the embedding window: WotC's mass-land-denial definition ("four or more lands per player without replacing them") and its absence from Brackets 1-3, the two-card-infinite-combo rule as restated in October 2025 in terms of the bracket's turn floor, the extra-turn "not intended to be chained in succession or looped" clause, and the October 2025 removal of tutor restrictions entirely ("rely on Game Changers to catch the most efficient tutors")
- added strategy:deckbuilding.cutting — the last-ten-cards problem: Gregory's "about 58-60 slots" arithmetic and the 90-100-card moodboard, cut-the-staples-first / budget-as-forcing-device / brew-with-what's-at-home heuristics, Milan's template-then-one-in-one-out mechanic and 7-8-of-150 survival rate, and the cut rule derived from the hypergeometric numbers already in .redundancy-vs-tutors (the 11th copy of an effect buys ~4 points of opening-hand probability, 54% → 57%) rather than re-derived
- added strategy:deckbuilding.budget — where money actually binds: Zupke's $50 / $1-per-card five-colour build showing the mana base as the constraint plus his "never have to pay mana to play your lands" rule and the sub-dollar fixing tier, Bucks' "$1 or less" threshold, Levin's two-dollar Scryfall filter and price-vs-quality quote, Gregory's budget-as-cutting-device, and the modern proxy norm (Carrozza: the argument is about power level, not cost) with the reminder that a proxied Game Changer still raises the bracket floor
- amended strategy:deckbuilding.interaction-suite — reconciled the removal-count tension with .ratios: added the dated drift (the Command Zone template cut wipes 5→3-4 as the format got faster; Walser's one-or-two-max is the continuation) and the instruction to date any wipe count you inherit; tightened surrounding prose to stay inside the 1200-char window
- amended strategy:schools — brought the corpus description current: deck construction now named as a separate shape (Karsten's regressions and per-pip tables as its mathematical spine, the template genre above them as priors rather than law, EDHREC's data pulls), Card Kingdom and Cardsphere added to the article-borne Commander canon, and WotC's Commander Format Panel named as the one primary source the doc has for power level

Verification note: every URL added this pass was fetched and parsed this
session. The bracket material comes from primaries only — the live
magic.wizards.com/en/formats/commander page (parsed out of its Nuxt payload;
the tab content is not in the rendered HTML) for the current bracket copy, the
Game Changers gate and the 53-card list, plus Verhey's four announcement
articles for the wording and the change history. Commander's Herald blocks
plain HTTP clients (406), so its two sources were fetched through the
article-reading fetcher instead. Tutor density, specifically: there is **no**
official number, and as of the October 21, 2025 update there is no tutor
restriction at all — the panel judged "few" to be unclear ("not all Tutors are
created equal... is Expedition Map a tutor?") and deleted the guiderail,
delegating it to the Game Changers list. `src/manamap/pilot/bracket.py` is
therefore correct to keep tutor count advisory, though its note that "'few
tutors' was never given a number" now understates the case: the restriction
itself is gone. Sought but unusable: the WotC pages magic.wizards.com/en/
formats/commander-brackets and /en/gamechangers (both 404 — the content lives
under the #brackets and #gamechangers anchors of the format page); the
Nitpicking Nerds' final-cuts piece (video, no written write-up); and the
first page of "how many board wipes"/"budget commander" search results, which
is dominated by affiliate SEO (farseek, geekydomain, scrollvault, spellweave,
tcgprotectors, orbsportscards) — all omitted deliberately.

## 2026-09-30 — sixth research pass: aristocrats, the lifegain engine, typal density, on-plan draw

- added strategy:aristocrats — outlet + death-trigger economics: free vs mana-gated outlets priced by activation (Gottfried's "essentially free in our 40-life format", Ullman's control-over-timing, Commander Theory's answer-proofing), Gregory's correction that a free outlet is itself the removal magnet, fodder's three sources, a threat as fodder only in response, drain-per-death as the exchange rate
- added strategy:aristocrats.drain-scope — trigger scope vs drain scope: Blood Artist's any-creature trigger and single target against the each-opponent shape (Zulaport Cutthroat, Cruel Celebrant, Bastion of Remembrance), the three-to-one in a pod (60 vs 20 life over ten deaths), targeting as finisher and political lever, EDHREC's 89%-of-484 Edgar lifedrain inclusion; rule: each-opponent for the race, targeted for the kill
- added strategy:aristocrats.wipe-insurance — sacrifice-in-response as the wipe answer, the four death-draw shapes read off oracle text (nontoken / own nontoken / chosen type incl. tokens / flash count-this-turn), Vrooman's and GenoDoak777's running-out-of-gas framing, held insurance vs steady draw, the rebuild package
- added strategy:life-as-resource.lifegain-engine — sources/converters/draw; gain-to-loss (Vito, Sanguine Bond, Cliffhaven Vampire) vs loss-to-gain (Exquisite Blood); AMOUNT-scaled (Vito, Bond, Well of Lost Dreams) vs COUNT-scaled (Cliffhaven, Dawn of Hope) converters and which source feeds which; lifelink as source never payoff; Dunn's draw-rate quotes
- added strategy:life-as-resource.lifegain-engine.bond-blood — Bond/Vito + Exquisite Blood as a two-card infinite per Spellbook ("Spicy (Bracket 3-4+)") and Furtado, read against the Bracket 3 six-turn wording; intent over assembly odds; one half alone as the fair engine
- added strategy:deckbuilding.typal-density — Walser's 30+ creature floor, lord = anthem on a body with both halves scaling on board width, this bench's Forge measurement of cutting every lord (29.07 → 18.20 combat damage, 31/400 vs 50/400), DougY's cost-reduction and keyword axes, Walser's turn-3/turn-10 anthem rule
- added strategy:deckbuilding.typal-density.lord-exposure — Cullen's wipe-exposure cost and noncreature-anthem fix (Icon of Ancestry, Vanquisher's Banner), Legion's Initiative as cash-in protection, Menery's play-them-or-build-around-them, Furtado's quick rebuild; the lord count at which an anthem beats a body is explicitly unsourced and left as the trade-off
- added strategy:card-advantage.plan-draw — steady draw (Phyrexian Arena, Black Market Connections; Zaccagnino's turn 6-9 objection) vs on-plan draw (death / cast / gain), correlation as the trade, split-not-stack for a midrange that empties by turn 7 (the split itself flagged unsourced), life-costed draw priced against drain refunds
- amended strategy:deckbuilding.ratios — one cross-reference to strategy:card-advantage.plan-draw: draw has a shape as well as a count

Verification note: every URL added this pass was fetched this session
(Draftsim ×7, EDHREC guide + two articles + the Edgar lifedrain theme page,
Card Kingdom ×2, nerdtothecore, Commander Spellbook, Hurst's Substack,
cardmystic, Commander Theory via its /amp path, the Verhey announcement, Star
City Games, and the bench's own gotchas-bench.md on GitHub, which renders
publicly). Commander Theory's "Sacrifice Outlets" post carries no byline and
is cited to the site, as the Commander Deck Maker pages were; its Grim
Haruspex post redirects to a Tumblr login and was not used. Sought but
unusable: wiki.edhrec.com (DNS failure); both TCGplayer articles ("How to
Build a Lifegain Commander Deck", "Exiting the Phyrexian Arena") render only
a header to a plain fetcher; Goonhammer's "Getting Started: Commander" fetched
blank; the MTG Salvation "Zulaport Cutthroat is broken" thread has no post
comparing drain scope; no reddit thread on lord counts surfaced through four
query phrasings (results were eBay listings); turnzerohq's tribal guide is
unattributed SEO and was omitted with the gamertagmythras / grimdeck /
krakenthemeta / geekydomain tier. Two figures in the new sections are the
doc's own arithmetic from oracle text, not a primer's: the 60-vs-20 drain
over ten deaths, and the AMOUNT / COUNT split of converters. Two
prescriptions are explicitly unsourced and flagged in-text: the lord count
past which an anthem beats a body, and the steady / on-plan draw split.

## 2026-10-01 — seventh research pass: the conversion layer, conserved output, the zonal commander

Opened a new pillar for the problem class behind edgar-vampires: a deck that
produces units reliably and cannot turn them into dead opponents. Three of the
ten new sections name a hole in the literature rather than filling it.

- added strategy:conversion — the pillar: production is not conversion. John's "the point of your engine is not to make resources. The point of your engine is to win the game", three 40-life totals (or 21 commander damage each), Anderson's "won't go far enough to bring down multiple 40-life opponents... while we're tapped out", Collins' "spin its wheels", Gregory's aristocrats-are-a-trap argument ("throw away those incremental accruals"; the best such decks "sacrifice things incidentally, and are built around another strategy entirely"); the three-number diagnosis (units × converter presence × life per unit) is the doc's own framing, not a primer's
- added strategy:conversion.payoff-density — SOURCED GAP: no Commander primer states an enabler-to-payoff ratio, so the section says so and substitutes revealed preference — EDHREC's Edgar aristocrats theme, n=1,496: Blood Artist 93%, Cruel Celebrant 79%, Vito 72%, Bloodthirsty Conqueror 58%, Sanguine Bond 43% (fetched from EDHREC's own JSON, counts over potential_decks). The rule "count converter-turns, not slots" and the 20% conversion figure (9.64 deaths, 1.90 fires, converter cast in 17% of games) are this bench's Forge arithmetic
- added strategy:conversion.durability — Zaccagnino's Bastion of Remembrance reading ("a three-mana Zulaport Cutthroat on an enchantment"; "being an enchantment keeps it safe from a larger swathe of removal and board wipes") and Cullen's noncreature-anthem case. The CORRELATION — a creature converter dies to the same sweeper that makes the deaths, so the rate collapses exactly when the death count spikes — is explicitly flagged in-text as stated by no primer; the 0.6 battlefield-turns figure is ours
- added strategy:conversion.entries-vs-deaths — Furtado's on-entry damage theory ("it's like they're pseudo-unblockable, hasted threats", better "in go-wide token strategies") and Zaccagnino's each-opponent arithmetic ("worth three total damage in situations where Blood Artist would only deal one"); EDHREC's whole-commander page (n=51,341) for the adoption gap, Blood Artist 82% against Impact Tremors 20% and Warleader's Call 18%. The claim that an entry is free where a death costs a permanent is the doc's own, offered as the mechanical form of Gregory's "sacrifice incidentally"
- added strategy:conversion.combat-channel — Anderson on why trading creatures for damage does not scale to three seats, plus the existing lord measurement (29.07 → 18.20). UNSOURCED GAP named in-text: nobody prices the damage-per-mana of an attack step against that of a trigger, so the comparison must be measured on one harness
- deprecated strategy:conversion.output-conservation — ADDED IN THIS PASS AND SUPERSEDED THE SAME DAY by strategy:conversion.substitution once the fleet control arrived (see the eighth pass below); the verb on this bullet was changed from `added` so the entry still resolves against the doc. SaffronOlive's opportunity-cost rule ("Is this the best option for this slot in my deck?"; a 7-out-of-10 going in over an 8 or 9 is a net loss). UNSOURCED: "conserved output" is not a named phenomenon in Magic writing — the nearest framings are opportunity cost per slot and diminishing returns on redundancy, and neither predicts a total, so the section instructs reading conservation as evidence about the bottleneck. The four totals (59.5 / 47.5 / 47.2 / 57.9) are ours
- added strategy:conversion.raising-the-ceiling — the levers the literature does name for an input-limited deck: John on a six-mana commander ("investing in cards and ramp is the only thing that lets us cast our commander at all"), Walser on cost reduction as ramp that changes only "the amount of mana you pay" and "encourages playing bundles of spells", Sison on extra combats, Karsten on consistency. The two-more-mana / one-fewer-missed-land-drop test is ours
- added strategy:conversion.quiet-contributors — Sherwood's quadrant theory (the four quadrants quoted, plus "crosses the line into win more when the player was winning anyway") and katydee's win-more definition with the caveat that matters when cutting ("it never helps turn a loss into a win"; the concept makes players worse because good cards get mislabelled). The name-the-channel-before-you-cut rule is ours
- added strategy:conversion.kill-pattern — Furtado's stated Edgar wincon quoted in full ("going to come from combat damage... swinging with 10+ medium-powered creatures"), EDHREC's theme counts (aristocrats 1,496 against tokens 2,144 and aggro 1,587; Conqueror 58%, Exquisite Blood 53%, Sanguine Bond 43%). SOURCED GAP: no fetchable primer gives a kill turn or a combat-versus-trigger split — the high-power Edgar writing is on Moxfield (403) and Archidekt (402), so the mechanism claim is sourced and the timing claim is not
- added strategy:multiplayer.zonal-commander — Macready on eminence affecting the battlefield from the command zone "even if they are never played" and Verhey's "The Magic Mechanic I Shouldn't Have Made!" as Macready quotes it, Iñaki's rules framing and "there's little you can do", Furtado's "Edgar can get expensive quickly as a 6-drop, but even leaving it in the command zone reasonably grows your board", Bockman on the tax being "prohibitively expensive" past three or four casts. UNSOURCED GAP: no writer gives the threshold at which the body is worth the cast
- amended strategy:aristocrats — one cross-reference out to strategy:conversion (the three legs say nothing about whether the deck closes), with the surrounding prose tightened to hold the section under the size cap; "payoffs on board" reworded to "converters live" for consistency with the new pillar's vocabulary

Verification note: every URL added this pass was fetched this session and the
load-bearing quotes were re-checked against raw HTML, not a summariser —
cardkingdom ×2 (Anderson's Go Wide, Gregory's Aristocrats-are-a-trap),
commandersherald (Collins), thegamer (Zaccagnino, byline read from the page's
JSON-LD), draftsim ×4 (Furtado's Edgar guide and Impact Tremors list, Iñaki's
eminence piece, Walser's cost reducers), wargamer (Macready), mtggoldfish
(SaffronOlive), lesswrong (katydee), edhrec (Sherwood's quadrant guide, plus
json.edhrec.com for the Edgar commander and aristocrats-theme inclusion
counts), coolstuffinc ×1 (Bockman), airza.net (John), and the bench's own
sim-record directory and gotchas-bench.md on GitHub, both of which return 200
publicly. Sought and unusable: Moxfield primers (403) and Archidekt guides
(402), which is where the high-power Edgar writing lives, so the kill-pattern
section cites the mechanism and not the turn; edh.fandom.com's Group Slug and
Aristocrats pages (402); MTGNexus' "Archetypes: Enablers and Payoffs" thread
(403), which search snippets claimed carried a 2:1 enabler-to-payoff ratio —
omitted rather than cited unfetched; the Out of the Box MTG substack episode
page (transcript not published); Draftsim's group-slug list, which turned out
to contain no multiplayer arithmetic; tcgplaymatpros (402). The SEO tier
(manacove, wubrgapp, tappeddecks, mtgapp, cultureofgaming, krakenthemeta,
gamertagmythras, grimdeck, gamequery) was read and omitted, including one page
whose "10-15 token producers, 2-3 payoffs, 1-2 doublers" ratio was the only
numeric answer found to the payoff-density question — unattributable, so the
section records the gap instead. Four claims in the new sections are the doc's
own arithmetic or framing and are labelled as such in-text: the
units × presence × life-per-unit product, the sweeper/converter correlation,
the entry-costs-nothing asymmetry, and the two-more-mana test.

## 2026-10-01 — eighth pass: the fleet control refutes conserved output; yield, availability, the passive tax

A fleet control landed hours after the seventh pass committed and refuted the
frame that pass was built on. Output is NOT conserved under substitution: across
22 same-pod same-harness version pairs, four differences exclude zero and ALL
FOUR ARE DECREASES — no swap on this bench has ever raised a total. The same
control retired payoff COUNT as the lever (yawgmoth-swarm's nine drain payoffs,
cheaper on average, fire FEWER times a game than edgar-vampires' four) in favour
of two axes the doc did not have: yield per fire and availability. This pass
corrects the superseded section, adds the two axes, and adds the shape the bench
has actually measured carrying a total.

- renamed strategy:conversion.output-conservation — superseded by strategy:conversion.substitution. The old id asserted conservation, which the control refutes in a specific direction: substitution is not symmetric, it is downward. The id is retired rather than re-pointed because the claim changed, not the wording
- added strategy:conversion.substitution — the control stated as the finding: 22 version pairs, four differences excluding zero, all four DECREASES, labelled observational-across-merges-and-seeds rather than an A/B in the text itself. The clearest case (three passive pingers cut; total life removed 15.62 → 7.88, −7.73 [−12.11, −3.36]; noncombat 9.45 → 1.23) carries the whole of the loss. SaffronOlive's opportunity-cost rule is the discipline that predicts the asymmetry ("Is this the best option for this slot in my deck?"; a 7-out-of-10 going in over an 8 or a 9 is a net loss). UNSOURCED GAP: nothing this pass reached names the cut/addition asymmetry; the practical form — prove a card is dead by removing it, and change an input rather than a slot when the ceiling is what you want — is the doc's own
- added strategy:conversion.yield — the scope × scaling 2×2, which is where the pilot's breadth finding belongs. Dunn states the scope axis once and cleanly ("Vizkopa's ability hits each opponent, instead of just one target", at "a similar mana investment to Sanguine Bond"); the AMOUNT / COUNT axis was already in strategy:life-as-resource.lifegain-engine and is cross-referenced, not restated. UNSOURCED GAP: no writer states the crossover, so the section derives it — per event an amount-scaled targeted converter removes A life from the table where a count-scaled each-opponent one removes 3, so targeted wins only when the mean gain event exceeds 3, which is directly measurable. Verified against oracle text in data/decks/edgar-vampires/cards.json: three of the five drain payoffs are single-target (Blood Artist "target player loses 1", Sanguine Bond and Vito "target opponent loses that much life"), Cruel Celebrant is the only each-opponent DEATH drain, Sanctum Seeker's each-opponent trigger is gated on attacking and so inherits the combat channel's taxes, and Edgar's eminence token is a plain 1/1 with no lifelink — which is what collapses the amount-scaled converters to one life a fire. The 3.75 / 1.83 per-fire yields are the bench's
- added strategy:conversion.availability — Timm's engine test ("repeatable, recoverable, affordable"), his ranking of locations (the commander "is the key to your greatest advantage engine as it is the most reliably repeatable card in the 100 cards"; inside the 99, "favour permanents with abilities rather than instants/sorceries"), and the doc's existing hypergeometric figures for the alternative of buying availability with copies (5 copies = 31% of openers, 7 = 41%). The ordering — command zone, then enchantment or artifact, then recursion, then copies — is the doc's own synthesis of Timm with strategy:conversion.durability. The 80.8%-against-36-39% spread is the bench's
- added strategy:conversion.passive-tax — the shape measured to carry a total: cheap, each-opponent, PASSIVE (an upkeep trigger, no attack and no activation), non-creature, scaling on a state the deck already produces. heliod's three four-mana hand-size artifacts fired 3.86 times a game and carried 9.45 of its noncombat damage. Group slug is the named archetype (Ignacio's "we're all gonna be having a bad time together"); the hand-size family (Black Vise, Viseling, Iron Maiden, Misers' Cage, Ebony Owl Netsuke) is collected in forum writing rather than any primer, and the section says so. UNSOURCED GAP: nobody prices a passive upkeep trigger against an attack step or a death trigger
- added strategy:conversion.measure-choice — the objective-choice warning the pilot asked for: on this bench the best per-fire yield has the WORST win rate and the highest total has the best, because a ratio can be maximised by firing rarely. The literature never names a predictive output statistic — a gap, stated as one — but it reasons in totals (Flores' quanta of damage against a fixed lethal; John's three 40-life seats) and CLOCKS (Duke's racing), never in rates. The rule "aim at a quantity monotone in winning, never at a quotient whose denominator the deck controls, and print the total beside any efficiency figure" is the doc's own
- amended strategy:conversion.payoff-density — retitled "Payoff Density Is Not the Lever" and re-led with the control (yawgmoth-swarm nine payoffs at mean mana value 2.78 firing 1.40 a game against edgar's four at 3.00 firing 1.76) instead of with the literature's missing ratio, which is still named as a gap. The conversion rate is now stated as a product — units × availability × yield per fire — with the note that only the first term is about slots, and the section hands off to the two new ones
- amended strategy:conversion.durability — its first fix now routes through strategy:conversion.availability, since location is the lever and the type line is only the cheapest form of it
- amended strategy:conversion.raising-the-ceiling — opening rewritten from "if output is conserved under substitution" to "since no measured swap on this bench has raised a total", and free spells added to the lever list beside cost reduction and extra combats
- amended strategy:card-advantage.plan-draw — repaired a cross-reference that line-wrapping had split mid-id ("strategy:aristocrats.wipe-" + "insurance"), which left it unresolvable to anything reading the doc programmatically
- amended strategy:aristocrats.wipe-insurance — same repair (strategy:card-advantage.plan-draw)
- amended strategy:life-as-resource.lifegain-engine — same repair (strategy:life-as-resource.lifegain-engine.bond-blood)
- amended strategy:life-as-resource.lifegain-engine.bond-blood — same repair, two references
- amended strategy:deckbuilding.typal-density.lord-exposure — same repair, two references

One edit reaches backwards, and it is flagged here rather than done quietly: the
seventh pass's committed bullet for strategy:conversion.output-conservation had
to stop saying `added`, because the validator requires an added id to exist and
this pass retired it. Its verb is now `deprecated`, its text is otherwise
untouched, and it names its successor; the seventh pass's own description of what
it added is preserved word for word.

Verification note: the new URLs this pass were fetched and their load-bearing
quotes re-checked against raw HTML — intothe99 (Timm), draftsim's lifegain
payoff list (Dunn's Vizkopa note) and draftsim's group-slug list (Ignacio). One
source is cited WITHOUT a quoted sentence on purpose: MTG Salvation's "Cards
that punish people for holding cards?" thread returns content to a browser-like
fetch but 403s a plain one behind Cloudflare, so its wording could not be
re-verified to the standard every other quote here meets;
strategy:conversion.passive-tax therefore names the hand-size family — whose
membership is a fact about five cards' oracle text — and attributes the
collecting to the thread without quoting it. A second MTG Salvation thread, on
early elimination, was fetched, read and then DROPPED for the same reason:
strategy:conversion.measure-choice rests on Flores, Duke and John instead, all
of which verify. Searched and found empty: no article compares an amount-scaled
targeted drain against a count-scaled each-opponent one (four phrasings), and no
article prices a passive upkeep trigger against an attack step — both recorded
in-text as gaps rather than filled.

The fleet-control figures folded in here were supplied by the orchestrator and
are NOT re-derived: the 22 version pairs and their four decreases, heliod's
15.62 → 7.88 and 9.45 → 1.23 with 3.86 fires a game, yawgmoth-swarm's nine
payoffs at mean mana value 2.78 firing 1.40 against edgar's four at 3.00 firing
1.76, the 80.8% / 36-39% availability spread, and the 3.75 / 1.83 per-fire
yields. The one thing this pass checked for itself is the oracle text behind the
breadth claim, read from the deck's own cards.json and quoted in the yield
bullet above. Five sections changed in whitespace only, from re-wrapping around
the repaired ids and nothing else: strategy:multiplayer,
strategy:conversion.entries-vs-deaths, strategy:conversion.combat-channel,
strategy:conversion.kill-pattern and strategy:multiplayer.zonal-commander.
