# STYLE.md — The Pilot's Manual Design Language

*A researched style guide for restyling the pilot's manuals in the visual and
editorial language of late-1990s video game magazines — Nintendo Power
(1996–2001, Volumes ~80–140) as the structural spine, with The Duelist,
InQuest, and GamePro supplying the TCG-specific and page-furniture vocabulary.
Research provenance: primary-source OCR of Nintendo Power V100 (Sept 1997),
V103 (Dec 1997), and V140 (Jan 2001) plus The Duelist #9 (Feb 1996), backed by
design retrospectives and magazine-history sources. Claims below marked
"(evocation)" are period-consistent art direction where no primary source
confirms the specific detail; everything else traces to the research.*

---

## 1. The premise

We have the horsepower and the rigor — machine-verified combo lines, seeded
simulations, an adversarial checker. What we're adding is the window dressing
of the golden age of game magazines: the feeling of opening a fat glossy issue
in 1997 and knowing, before reading a word, that the people who made this
*loved the game* and were about to make you better at it.

**The manual becomes an issue.** Each deck's manual is presented as a numbered
volume of an ongoing magazine, not a standalone report. Zada is Vol. 001.
Hapatra and Edgar will be Vols. 002 and 003. Spines line up.

**The evidence contract survives the makeover — completely.** The three-tier
badges (✓ rules-verified / ◆ data-derived / ★ coaching) are the brand's
credibility and they stay on every section. What changes is their costume:
each tier gets a period-authentic visual identity (see §4). Nothing in this
guide ever trades verifiability for vibes.

**Two registers existed in the era — we pick per-section, never blend mushily:**
- **"Extreme" register** (early/mid-90s: GamePro, NP features): neon clash,
  starbursts, comic energy, punny all-caps tip boxes. → Use for strategy
  features, tips, decision spreads.
- **"Techno" register** (late-90s/Y2K: PSM, OPM): Eurostile Extended, chrome,
  blue/purple/silver, spec-sheet precision. → Use for data (goldfish, stat
  boxes) and the appendix dossiers.

---

## 2. Editorial architecture — the department map

Nintendo Power's book had a fixed anatomy (verified from V100/V103 TOCs):
letters + charts up front → fat middle of strategy features → service
departments (codes, Q&A) → reviews and previews in back → "Next Issue" teaser.
Departments were *branded institutions* with persistent mastheads, mascot
framing, and their names printed in the page folios ("NOW PLAYING | 131").

Our manual sections map onto that anatomy:

| Manual section | Department treatment | Tier | Model |
|---|---|---|---|
| Cover | Magazine cover: masthead, hero art, coverlines, violators | — | NP cover conventions (§3) |
| Tier legend | "HOW TO READ THE METERS" box on the contents page | — | NP's ESRB legend + Power Meter explainer, reprinted every issue |
| Table of contents | "IN THIS ISSUE" with STRATEGY / EVERY ISSUE buckets | — | V103 TOC's three-bucket structure |
| How It Wins | Feature splash spread: big deck logo, second-person hype dek | ★ | NP feature openers ("Take the wheel of one of the hottest…") |
| Goldfish Numbers | **"POWER CHARTS"-style data department**: meters, turn-by-turn chart tables, fast-facts box | ◆ | Power Charts + Now Playing fast-facts sidebar |
| Combo lines | **Strategy feature spreads** — the fat middle. Play sequence as numbered map-style callouts ("1. GET WET" energy), punny tip boxes, MAP KEY legend | ✓ | NP strategy features; full citations move to the Appendix (§2.1) |
| Playing the Table | **"COUNSELORS' CORNER"-style advice department**: the coach as a named counselor persona, Q&A rhythm | ★ | NP Counselors' Corner (terse Q:/A: blocks, hotline framing) |
| Decision spreads | **"ARENA"-style challenge pages**: "WHAT'S YOUR PLAY?" — board state as the challenge, branches as entries, recommendation as the counselor's answer | ★ | NP Arena (reader challenges) + puzzle culture (The Duelist's "Magic: The Puzzling" — board state, find the line) |
| Matchups | **"VS." pages** with matchup mini-mastheads and a threat meter per archetype | ★ | EGM Review Crew multi-panel + boss-box treatment |
| Card Roles | **"CARD SPOTLIGHT" grid** — trading-card tiles with role labels as category chips | ★ | The Duelist's "Card Spotlight" column; price-guide grid density |
| Mulligan Guide | **"KEEP OR SHIP?"** — sample hands as quiz items with verdicts | ★/◆ | Counselors' Corner Q&A + goldfish numbers cited inline |
| Upgrade Paths | **"PAK WATCH"-style previews department**: "UPGRADE WATCH — the inside source on future card slots" | ◆ | NP Pak Watch (chatty preview voice, fast-facts per card) |
| **Appendix: stack dossiers** | **"CLASSIFIED INFORMATION"** — see §2.1 | ✓ | NP Classified Information (espionage dossier styling) |
| Back page | "NEXT ISSUE" teaser (the other decks in the series) + colophon | — | NP back matter |

### 2.1 The appendix restructure (the one structural change)

Today the full rules-cited resolutions render inline. In the restyle:

- **In the body**: each combo line becomes a *strategy feature spread* — the
  scene, the play sequence as numbered callouts with card images, the payoff,
  and the coaching read. Fun first. Each spread carries its ✓ badge and a
  dossier pointer styled as a Classified Information cross-reference:
  **"FULL DOSSIER: APPENDIX A-004 →"**.
- **In the appendix**: the complete step-by-step resolution with every CR
  citation, checker verdict, and iteration count — styled as declassified
  case files (manila folder tints, TOP SECRET / VERIFIED stamps, typewriter
  face for rule quotes, redaction-bar section dividers as decoration).
  This is where the rigor lives for anyone who wants to dig in — exactly
  NP's move of quarantining dense code lists in a themed department instead
  of interrupting features with them.
- Contract note: the appendix is generated from the same checker-passed
  artifacts; the builder must not summarize away any citation. Anchor IDs
  stay stable (`#stack-004` → appendix anchor) so existing links keep working.

---

## 3. The cover

Verified NP conventions to adopt:
- **Masthead**: big, dimensional, top of page, never moves. Ours: **MANA MAP**
  set in chrome-gradient beveled caps (the 1995–2001 NP logo era was a
  3D-rendered chrome red logo), with "PILOT'S MANUAL" as the fixed series
  slug beneath. Volume number + deck name as the issue identity:
  "VOL. 001 — GOBLIN STORM".
- **One dominant hero image** (the commander — we have Scryfall art), with a
  short stack of secondary teases down one side (V103 listed 3–4 secondary
  games along the cover edge): tease the verified lines ("THE HAZE LOOP —
  VERIFIED INFINITE INSIDE!").
- **Coverlines and violators**: starburst/violator badges with era copy
  energy — "5 VERIFIED LINES!", "EXCLUSIVE: THE KRENKO FILE", "GOLDFISH
  CHARTS INSIDE". (Violator = the industry term for a burst that
  intentionally violates the grid.)
- **Tagline** riffing on NP's "THE ONLY INSIDE SOURCE FOR ALL NINTENDO NEWS":
  ours — **"THE INSIDE SOURCE FOR YOUR COMMAND ZONE"**.
- The tier legend moves just inside (contents page), styled like NP's
  standing Power Meter / ESRB explainer boxes.
- The Duelist precedent for art-forward TCG covers: commissioned painting
  energy — lean on the commander's card art large, not a collage.

---

## 4. The three tiers, in costume

The badges keep their glyphs and meanings; each gains a department-consistent
visual identity:

| Tier | Costume | Devices |
|---|---|---|
| ✓ rules-verified | **Classified/dossier**: manila tint, file-tab corners, a rubber-stamp "VERIFIED — RULES CHECKED" seal (stamp rotation ~-8°), typewriter face for quoted rule text | Stamp badge; "APPENDIX A-NNN" file numbers; checker iteration count printed like a case status ("CLEARED: 1 REVIEW CYCLE") |
| ◆ data-derived | **Spec-sheet/techno**: Eurostile-class extended caps, meter bars, chart grids on dark or grid-paper panels | "POWER METER" style horizontal bars for rates (79% keepable = a filled meter); fast-facts data boxes (seed, iterations, sha) styled like NP's megabits/save-type boxes |
| ★ coaching | **Counselor**: warm gold, speech-balloon and Q&A furniture, the counselor persona byline | "FROM THE COUNSELOR'S DESK" slugs; GamePro-style **PILOT TIP:** boxes (bold slug + one-sentence imperative tip attached to a card image) |

The cover legend explains all three in one standing box, reprinted verbatim
in every volume (NP reprinted its scoring explainer every issue).

---

## 5. Voice

Verified NP voice rules, adapted:

1. **Second person, imperative, reader-as-hero.** "Our turn-by-turn charts
   will put you back at the head of the pod." Never "the pilot should" —
   always "you."
2. **Pun-dense headlines at every scale.** Feature titles, tip boxes, letters
   — NP punned relentlessly ("Stop eating Diddy dust!", "ROCKS FOR BRAINS",
   "TONGUE LOOP"). Ours: "GOBLIN UP THE COMPETITION", "A TREASURE-BLE
   DECISION", "WISP-ER NETWORK". Tip-box titles are ALL CAPS and 2–4 words.
3. **Kicker + headline + dek structure.** Eyebrow line above ("VERIFIED
   INFINITE"), big headline ("THE HAZE LOOP"), then a 1–2 sentence
   second-person dek ("Four mana in, nine Treasures out — and it only gets
   better from here.").
4. **Clubby insiderism, never parental.** NP's editorial rule (Gail Tilden):
   peer-to-peer voice — "No reader wants their mom running their magazine."
   We/us newsroom framing: "We ran ten thousand goldfish games so you don't
   have to."
5. **Named personas.** NP reviews carried recurring first-name evaluator
   comments ("Scott P.: Play before you pay."). Our agent roster maps
   naturally: the coach signs coaching sections as a counselor persona; the
   checker is "the Judge" whose stamp appears on dossiers; the goldfish
   simulator gets a lab-tech persona for chart captions. Personas are
   *presentation only* — the tier badges still tell the truth about
   provenance.
6. **Authoritative on strategy, cornball in banter.** Strategy copy stays
   precise and stepwise; department intros get the "Hey there, joystick
   jockeys" energy — sparingly.
7. **Honesty as a feature.** NP printed corrections with self-deprecating
   charm ("Okay, we admit it. We goofed!"). Our Krenko refutation (stack 004)
   is *exactly* this energy — "the database said infinite; the rules said
   no" is a proud headline, not a footnote. InQuest's credibility died on
   hype misses; ours lives on the checker.
8. **The Duelist register for rules content.** Rules text stays literate and
   exact (The Duelist was the in-house authority: real rulings by the actual
   rules manager). Dossiers quote the CR verbatim, always.

---

## 6. Typography

No primary source documents NP's pre-2005 body faces — so these are honest
evocations of documented era trends, chosen from freely embeddable faces
(manuals are standalone HTML on Pages; fonts must be bundled `@font-face`
assets or system stacks — the CSP-free static site can carry WOFF2 files in
`manuals/fonts/`):

| Role | Era model | Free equivalent (bundle) | Fallback stack |
|---|---|---|---|
| Masthead / techno display | Eurostile Bold Extended ("near-monopoly on sci-fi type through 2000") | **Michroma** or **Jura** (Eurostile-class) | `"Michroma", "Arial Black", sans-serif` |
| Feature headlines | Helvetica Compressed / Compacta / Futura Extra Bold — condensed bold caps, often obliqued 8–15° | **Archivo Black** / **Oswald** (condensed bold) | `"Oswald", "Arial Narrow", sans-serif` |
| Tip-box slugs / comic register | Comic hand-lettering (GamePro balloon captions) | **Bangers** or **Luckiest Guy** | cursive fallback |
| Dossier / rule quotes | Typewriter (classified files) | **Special Elite** or **Courier Prime** | `"Courier New", monospace` |
| Body copy | 9–10pt grotesque, tight leading | **Inter** or system | system-ui stack |
| Data tables / stats | Eurostile-class small caps or tabular grotesque | Michroma small sizes / Inter tabular-nums | monospace for numbers |

Display treatments (all documented era trends): ALL-CAPS display; hard offset
drop shadows (`text-shadow: 3px 3px 0 #000`); thick contrasting keylines
(webkit-text-stroke or layered shadows); gradient fills via
`background-clip: text` (yellow→orange for extreme register, silver chrome
ramp for techno); oblique transforms on score numbers and section logos;
type punched out of starbursts.

---

## 7. Color system

Era-documented palettes, organized as department color-coding (NP branded
each department persistently; folios carried the department name):

```css
/* Base "paper" */
--paper:        #F4EFE4;  /* cheap-glossy cream, not pure white */
--ink:          #1A1714;  /* rich black, slight warmth */

/* Extreme register (features, tips, decisions) */
--power-red:    #E4002B;  /* masthead red (NP chrome-logo era) */
--burst-yellow: #FFD800;  /* violator fill */
--radical-purple:#7B2D8B;
--slime-green:  #3FBF3F;
--hot-magenta:  #E4007C;

/* Techno register (data, appendix chrome) */
--chrome-hi:    #E8ECF0;
--chrome-lo:    #7A8699;
--y2k-blue:     #1B4FD8;
--y2k-violet:   #5B2E9E;

/* Tier identities (existing semantics, new values) */
--tier-verified:#2E7D32;  /* ✓ green — stamped-ink green */
--tier-data:    #1B4FD8;  /* ◆ blue — techno blue */
--tier-coach:   #C8A03C;  /* ★ gold — counselor gold */

/* Dossier */
--manila:       #E8D9A8;
--stamp-red:    #C41E1E;
```

Rules: gradients are period staples (orange→yellow, blue→purple) — use them
on display type and meter fills, never on body text panels. Neon-on-black is
reserved for the appendix dossier section dividers (evocation — the
"Classified" energy). Section color-coding: each department keeps one accent
across all its pages and its folio tab.

---

## 8. Page furniture catalog

Every device below is research-verified unless marked (evocation):

- **Folios with department names**: page footer alternates
  "MANA MAP | VOL. 001" and "COUNSELORS' CORNER | 12" — NP literally put the
  department name in the folio. In HTML: a sticky-bottom strip per section,
  or a repeated section-end bar.
- **Violators/starbursts**: 8–12 point bursts with rotated ALL-CAPS copy,
  used max 1–2 per spread ("NEW!", "VERIFIED!"). CSS: `clip-path` polygon
  star + rotation.
- **PILOT TIP boxes** (GamePro ProTip, verbatim formula): card image +
  bold slug + one imperative sentence. "PILOT TIP: Never sac Krenko to
  Prospector — the rules have confirmed he does not come back."
- **MAP KEY legend boxes** (NP maps): our combo spreads get a legend mapping
  icons → resources (⚡ mana floated, 🜲 storm count, ⛃ Treasures, ♥ life).
  Consistent icon set across all volumes.
- **Numbered callouts** with mini-headlines in caps: play sequences as
  "1. LIGHT THE FUSE — cast Haze with buyback…" tied to card images the way
  NP tied callouts to map positions.
- **Boss boxes → THREAT BOXES**: matchup archetypes get the boss treatment —
  name, "arena notes" (what their board looks like), tactics, and a threat
  meter (evocation of NP boss panels + power meters).
- **Fast-facts boxes** (NP review sidebar, verbatim structure): per-deck spec
  sheet — commander, colors, curve peak, goldfish seed/iterations, decklist
  sha, rules version. Data presented like megabits/save-type/players.
- **Power Meters**: horizontal segmented bars for every rate the goldfish
  reports; corner dial/thermometer for the deck's overall "assembly speed"
  (evocation — NP's meter was for review scores; ours reads from real ◆ data).
- **Speech balloons on card images** (GamePro register): sparingly, decisions
  section only — the table talking ("He's got the storm deck!").
- **Stamps and file tabs** (Classified evocation): VERIFIED stamps on
  dossiers, tabbed appendix pages A-001…A-005, "CASE STATUS: CLEARED".
- **Reader-culture furniture** (NP's membership loop, adapted): a "SUBMIT
  YOUR LINE" box inviting scenario suggestions (→ resolve-stack queue), and
  decision spreads framed as reader challenges with the counselor's answer.
- **Tinted sidebars with heavy borders**: 10–20% tint panels, 2–4pt borders,
  slightly rotated (-1° to 1.5°) for the pasted-on look (evocation).
- **Angled cutout card images** with thick keylines and hard drop shadows —
  the era's screenshot treatment applied to card scans. Never more than ±6°.

---

## 9. Print-artifact effects (CSS recipes)

Documented techniques for the paper feel (apply globally, subtly):

- **Halftone dots**: pure-CSS recipe — `radial-gradient` dot matrix +
  `mix-blend-mode: screen` + `filter: contrast(999)` threshold; use on hero
  image backgrounds and section-divider art, ~4–6px dots. CMYK variant:
  stacked cyan/magenta/yellow dot layers with `background-blend-mode:
  multiply` and staggered positions.
- **CMYK misregistration**: 1–2px per-channel offset on display type only
  (duplicated text-shadow in cyan/magenta) — "real-world printing always has
  slight misregistration." Body text stays clean for readability.
- **Paper**: `--paper` cream base; a faint noise/texture overlay (tiny SVG
  data-URI, ≤3% opacity); images very slightly desaturated (`saturate(.95)`)
  to sit on the stock.
- **Ink spread**: 0.2px blur on display type in the dossier section only.
- **Boxed-page frame**: fixed max-width (~1040px) "trim" with the paper
  running edge-to-edge behind it — the screen is the desk, the div is the
  magazine.
- Accessibility floor: all body text ≥ 4.5:1 contrast on its panel; effects
  never applied to rule-quote text; `prefers-reduced-motion` disables any
  animated furniture.

---

## 10. Layout grid

- Base: 12-col CSS grid rendering as a 3-column magazine grid on desktop,
  broken *deliberately and often* — the NP/Japanese-anthology look is a
  lattice of map + callouts + boxed sidebars, body copy short and
  interstitial. Single column stack on mobile with furniture intact.
- Kicker/headline/dek on every feature opener.
- Captions: bold colored lead-in + roman body ("THE PAYOFF: nine Treasures
  and a 20/2 paymaster.") — the ProTip caption grammar.
- Decklists (era-verified format): counts + names **grouped by card type**,
  sideboard as its own labeled list, attributed to the builder — exactly how
  The Duelist printed R&D decks. No mana-curve bar charts in the body (not
  attested in 90s print — our curve data lives in the ◆ goldfish department
  where charts are the point).
- Tables: heavy 2px rules, tinted header rows, tabular numerals.

---

## 11. Authenticity and legal guardrails

- **Inspired by, never imitating.** No Nintendo marks, no "Nintendo Power"
  name, no Nester, no Power Meter™ label verbatim where it reads as trade
  dress. Department names are ours (Counselors' Corner → "The Counselor's
  Desk" if we want extra distance — decide at implementation; the generic
  words are fine, the ensemble shouldn't photocopy one magazine).
- Card images/art: Scryfall imagery as already used; follow WotC Fan Content
  Policy framing (non-commercial fan work, no implication of endorsement) —
  worth a colophon line on the back page.
- **The tier contract is not costume.** A restyle PR that weakens a badge,
  drops a citation from the appendix, or lets a ★ section wear ✓ styling
  fails review. The stamps are earned by the checker, not the CSS.
- Verified-vs-evocation ledger: this document marks its evocations; when in
  doubt during implementation, prefer the documented device over an invented
  one.

---

## 12. Implementation roadmap (when we build it)

1. **R1 — Tokens + typography**: CSS custom properties (§7), bundled WOFF2
   fonts in `manuals/fonts/`, base paper/trim frame. `build_manual.py` CSS is
   inline today — move to a template constant, keep builds deterministic.
2. **R2 — Furniture components**: badge costumes, PILOT TIP box, fast-facts
   box, meters, violators, folios, tinted sidebars. Renderer helpers, each
   with tests (the renderer suite already covers determinism/escaping).
3. **R3 — Appendix restructure**: combo spreads become feature layouts with
   dossier pointers; full resolutions render in Appendix A with Classified
   styling. Stable anchors; `test_full_render_contains_all_sections` grows
   accordingly.
4. **R4 — Cover + TOC + department mastheads**: masthead art, coverlines,
   IN THIS ISSUE, tier-legend box, NEXT ISSUE back page.
5. **R5 — Gallery**: `manuals/index.html` becomes the newsstand — covers as
   issues on a rack.

Each phase regenerates goblin-storm and ships behind founder review of the
rendered HTML. Determinism, OG tags, and the citation contract are invariant
throughout.

---

## Appendix: research source highlights

- Nintendo Power V100/V103/V140 OCR (archive.org) — department order, review
  system history, verbatim voice samples, folio conventions.
- Game Developer, Vice/Motherboard, Complex retrospectives — V-Design/Work
  House Japanese layout DNA, map production, Classified Information's manila
  design, Tilden's voice doctrine.
- The Duelist #9 (Feb 1996) OCR — real column names (Magic: The Puzzling,
  Card Spotlight, Insider Trading, Magic Rulings & Errata), decklist format,
  Pro Tour announcement.
- InQuest/Scrye histories — top-10 set review format, price-guide culture,
  the purple-card hoax, casual-vs-tournament framing.
- Typography/design: Fonts In Use (Helvetica Compressed; NP's 2005 FF Dax),
  Typeset in the Future (Eurostile), Envato/Y2K retrospectives (palettes,
  treatments), leanrada.com (pure-CSS halftone), STUDIO·ITY (CMYK
  misregistration), The Bedroom Coder (90s layout in modern CSS).
