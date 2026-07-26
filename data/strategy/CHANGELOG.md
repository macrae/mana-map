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
