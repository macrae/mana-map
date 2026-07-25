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
