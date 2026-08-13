# Pilot's Manual — editor + founder feedback, 2026-08-13

> **STATUS: input, not decision.** This is the record of what was asked for, in
> the words it was asked in. `STYLEv3.md` and `issue_spec.DEPARTMENTS` are changed
> against it in the working pass that follows, and what actually shipped against
> each thread gets recorded at the bottom — the same shape as
> `docs/magazine-feedback-2026-08.md`.

Captured after Vol. 009 (Radagast) published. Two voices in this round: a **chief
editor** reading the issue cold, and the **founder** responding. Where they
disagree, the founder's call is marked.

The framing sentence, from the editor, which most of the rest follows from:

> This is a beautifully written magazine for a 42-year-old lawyer. My 12-year-old
> reader bounces off page one.

And the founder, agreeing and naming himself as the problem:

> The tone is just awful and rotten. Or it's, like, for a forty-two-year-old
> lawyer, data scientist — right — myself.

---

## 0. This round was predicted by the last one

The 2026-08-05 record closed with a **Still open** list. Three entries on it are
the direct cause of three complaints here, which is worth stating plainly rather
than rediscovering:

| Left open in August | Arrived as, this round |
|---|---|
| "`magazine-editor`'s per-columnist guidance is still a name-check… the packaging layer is still monovocal." | Every dek and headline reads in one voice; six departments open with the same rhetorical-question construction. |
| "`pilot-coach` was deliberately not touched." | Coach Sunny writes *"the deflection posture the strategic frame prescribes."* |
| "The founder raised adding columnists beyond the three. Not acted on." | The founder now wants a fourth — an editor-in-chief who writes the letter. |

The August pass made the three personas *mechanically separable* in the prose
agents. It did not touch the two places the reader actually meets them first.

---

## 1. The three bylines are one byline

The editor's test, which is the one to keep:

> If I can't tell who wrote a paragraph with the byline covered, the personas
> aren't doing work.

Sample sentences quoted back from Vol. 009:

- "One candour the record keeps…"
- "The holding is negative and it is the whole department."
- "…the deflection posture the strategic frame prescribes."

The third is signed **Coach Sunny Brightside** — the persona whose bio is "has
never once believed you're going to lose." The editor:

> That's not a coach, that's a McKinsey deck. Sunny should sound like he's
> yelling. Ledger should be all numbers and no adjectives. Vera gets to be the
> dry one — she's the only one earning it.

## 2. A straight bug: taxonomy tags in body prose

> "(strategy:multiplayer.pod-management)" — taxonomy tags leaking into body
> prose, three times. That last one's a straight bug.

Measured across the fleet, not just Vol. 009: **68 occurrences reaching the
rendered HTML in all eight published issues** (hapatra 18, sisay 10, ur-dragon 10,
goblin-storm 8, yawgmoth-swarm 8, heliod 7, edgar 5, radagast 2). Source is
`manual_prose.json`. Nothing validates against it.

## 3. Kill the rhetorical-question openers

> Every department starts with one. Six departments, six questions. Open on a
> moment instead: "Turn five. Dave has six Forests open and everyone at the table
> has decided he's the ramp guy. He is not the ramp guy."

Founder: *"A hundred percent. You're reading Commander magazine. We don't need to
describe what a commander is and cite the rules every single time."*

They are in the **deks**, written by `magazine-editor` — not in the prose. 14
live: heliod 5, sisay 5, radagast 3, edgar 1.

## 4. The best fact in the issue, sanded flat

> The 36-vs-40 math appears six times… In print you run it once, huge, as a
> full-page number, and cross-reference everywhere else.

## 5. The 99 is 70 essays

> Nobody reads 70 paragraphs. Make it a grid — card art, name, and a 12-word
> callout. "Hornet Nest: 0/2. Touch it and it makes deathtouch fliers. Nobody sane
> attacks." Move the paragraph-length reads to the ten cards that deserve them.

Confirmed: `card_roles` in `manual_prose.json` is a dict of **71 paragraphs**.

## 6. Unearned numbers

> "Threat level 60%" against a section that admits "Zero games played." A kid
> doesn't parse that hedge; an adult sees a fake number. Either derive them or
> make them stars out of five and call them the Coach's gut.

## 7. Draw the metronome

> Four windows a round is the entire deck, and it exists only as sentences —
> repeated in six different departments. It's a clock face. Nintendo Power drew
> the map; you're describing the map.

Founder: *"It needs more visuals."*

## 8. The big one — the magazine opens on the deck's constellation

This is the founder's, and it is the structural idea the rest reorganises around.

> I have something called a Mana Map, which uses neural networks to create an
> embedding space for all the Magic cards… you can load up any one of these decks
> and see the constellation of that deck. You see hard clusters and groupings, and
> it gives you a structure for the deck… I feel like that visualization of the
> card structure, that map, should actually be in this magazine.

Mechanics, in his words:

> We'd use the trained embeddings, and then we would just sort of cluster those
> points, name those — into a two-level hierarchy maybe — and then that becomes
> the name of the cards.

And on what it unlocks downstream:

> That also sort of informs then the engine. So based on that clustering, when we
> talk about the deck, what the coach and all of the other analysts…

The editor's read on why this solves several threads at once:

> You lead with the constellation… Then the prose points: "See those five
> clustered way out? Those are your traps." The metronome becomes a small
> repeating graphic anchoring the cluster names. The 99 becomes the legend… the
> magazine becomes visual-first instead of prose-first — exactly the 1999
> instinct.

**Founder's answers to the three questions the editor put:**

1. **Interactive?** *"A hundred percent. A reader can hover over a point and see
   the card. They can see the text. They can click. We can filter. We can
   highlight."*
2. **Static or per-deck clustering?** Per deck, from the trained embeddings, named
   after the fact. Cluster naming register: **functional, with wit** (founder's
   call this session — "THE TRAPS", not a place name).
3. **Verified lines marked on the map?** Yes.

## 9. Two new front-of-book pieces

**The Editor's Letter.** A fourth persona, and explicitly not one of the three:

> I think that we're gonna have the editor, the main editor — which is neither of
> these writers — write the intro to the magazine, as most magazine writers do…
> This is the deck, this is your commander, this is why you would play it and what
> it does, and you're gonna have a good time if this is what you like.

**The Pilot's Log** — a three-way panel conversation replacing the prose-heavy
opening:

> It's a conversation between Coach and Vera and whoever else… this is now, like,
> a conversation that's occurring on a panel. This is how you pilot this thing.

Requirements the founder stated for it:

- **Agentic, not scripted.** *"They should be kind of responding to each other in
  real time… It should feel new, interesting, and dynamic when we generate it, but
  we should have those guideposts on it to keep it on point."*
- **Opens on a play moment** tied to a primary mechanic or win condition — *"I
  played with this deck recently, and this was something that I found
  interesting."*
- **Real segues**: *"finding segues and looking at what somebody else said, being
  like, well, that's interesting, and this is how it relates to some other thing
  in this deck."*
- **Three or four topics, split ~33% each.**
- Each voice reads the same engine structure and interprets it in character —
  Coach on maneuvering and reps, Vera on *"how to think about the exact way where
  most people will approach it like this, but based on the rules we can do it like
  this."*

**The engine doc is scaffolding, not copy.** Ignition, engine, fuel, fodder,
output, damage, removal — *"structures all that together into a strategy doc
that's given to the coaches"* — and never printed.

## 10. Judge's Desk is appendix material sitting in the magazine

> The "Verified line 001" blocks run full board states and result prose inline.
> Front of book gets the verdict in one line; the dossier lives in the back with a
> page reference.

## 11. The battle report

> 1999 magazines lived on them — a real game, four named players, a board photo
> with numbered callouts, and the turn where it broke open. You've got seven
> verified stacks that are almost that, minus the humans.

**Founder's call this session:** derived from a real seed. Board states come from
an actual goldfish seed or a checker-passed stack so every position is
reproducible; the three opponents and their table talk are costume, and the page
says so. A fully invented board next to a rules-verified one is the one thing this
magazine cannot afford.

---

## What the editor said to protect

Recorded because a redesign is exactly when good things get thrown out:

- **The headlines.** "36 IS NOT 40." "THE HOOF STAYS HOME." "NEVER TAP OUT." "TWO
  LANDS AND A DORK."
- **The cover blurb list.** *"'A 1/1 kills a 6/6 and draws two cards' — that's the
  whole pitch in eleven words."*
- **What's Your Play?** — *"the best department in the issue, and it's buried at
  #5. In 1999 that's your centerfold."*
- **The rules-verified conceit.** *"Nobody else was doing receipts."*

---

## Founder decisions taken this session

| Question | Call |
|---|---|
| Sequencing | **Three phases, ship each** — bugs/cuts, then the constellation, then the voices |
| Battle report | **Derived from a real seed**; opponents are costume and labelled |
| Cluster names | **Functional, with wit** |
| Length budget | **All three cuts**: the 99 becomes a grid, Judge's Desk shrinks to verdicts, and the three Coach table departments merge into one |

---

## Needs reconciling before Vol. 009 re-renders

The decklist pasted to the editor **does not match the published 99**, four cards
each way. The repo is self-consistent — Whiptongue Hydra, Indrik Stomphowler,
Yeva and Glademuse are in `cards.json` (four of the nine buys), and Abundant
Growth, Hardened Scales, Old Fat Spider and Unforgiving Aim are in no artifact at
all. So the pasted list is either an older export or the physical build diverging.
It matters because Know Your Enemy's whole flier plan rests on a Whiptongue Hydra
that list does not run.

---

## What this touches, if it is acted on

- **Phase 1** is renderer, validator and charter work: `issue_spec.py`,
  `validate_issue.py`, `build_manual.py`, `design.py`, STYLEv3 §5/§7.2/§8.4. The
  department merge changes `DEPARTMENTS`, so every `issue_plan.json` regenerates.
- **Phase 2** is a new subsystem: a per-deck local layout from
  `embeddings_ability.npy`, clustered and agent-named, emitted as a **tracked**
  `deck_map.json` — tracked because the embeddings are gitignored and a fresh
  clone must still render manuals.
- **Phase 3** is charters and one new routine. Charter edits invalidate by design
  and must land **before** any `cache-record`, never after.
