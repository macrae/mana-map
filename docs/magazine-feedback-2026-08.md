# Pilot's Manual — founder feedback, 2026-08-05

> **STATUS: shipped 2026-08-05** in `fc2277b`, `25f7378`, `233722a`, `9a7ff81`.
> What landed against each thread is recorded at the bottom of this file. Kept as
> the record of what was asked for, in the founder's own words, and of what was
> traded away to deliver it.

Captured from the pilot after reading through the published issues. This is a
**feedback record, not a decision** — it is the input to a working pass with the
`magazine-editor`, and nothing in `STYLEv3.md` or `issue_spec.DEPARTMENTS` has
been changed on the strength of it yet.

The framing sentence, which most of the rest follows from:

> Imagine as a reader — this should feel just like somebody handing you their deck
> and saying *here's my deck, check it out*. The first thing you do is look at the
> commander. You read his abilities, and then you start shuffling through the
> cards and organising things in your head.

---

## 1. It is too long

First reaction, before anything else. The issue does not feel like something you
read; it feels like something you get through. No page target was named — the
observation is about felt length, and it compounds with §2 (things arrive in an
order that makes you wait for the part you wanted).

Related and stated separately: **there are a lot of coaching sections.** Six of
the seventeen departments are signed by Coach Sunny Brightside alone, and they
run consecutively at positions 3–8.

## 2. The running order is wrong

The strongest and most specific piece of feedback. Current order against what the
pilot wants:

| | current | proposed |
|---|---|---|
| 3 | The Game Plan | **The Command Zone** |
| 4 | Keep or Ship | **The Game Plan** |
| 5 | What's Your Play? | **The 99** |
| 6 | Table Manners | **Keep or Ship** |
| … | | **The Kill** |
| 9 | The Command Zone | |
| 10 | The 99 | |
| 14 | The Kill | |

The reasoning, in the pilot's terms:

- **Lead with the commander.** "All commanders are built around a commander. When
  you ask somebody what deck are you playing, they lead with who the commander
  is." The Command Zone currently sits at position 9. It should open the issue —
  and it does not need to be long: *this is your commander, this is how you should
  think about them.*
- **The Game Plan segues from it** and is fine where it is *relative to the
  commander* — it simply cannot come first.
- **The 99 comes next, and it is one of the pilot's favourite sections.** "It
  gives it to you already organised, with all these cards already flagged in terms
  of who they are in the deck, how you should think about this card, what it gives
  you, grouping them." It is currently at position 10, behind six coaching
  departments.
- **Mulligans are too early.** Keep or Ship is at position 4, before the reader
  has seen a single card. "Once somebody has actually had the opportunity to look
  through the 99, they have an idea for the distribution, and those mulligans make
  a lot more sense."
- **Pull the combo lines up**, to just after the 99 — "here's how you pilot it,
  and here's some combo lines." The Kill is currently at 14, Judge's Desk at 15.

## 3. The combo lines need to be cut down and grounded

Two distinct problems.

**Too much text.** "Way too much text… less words, and just: this is the combo,
these are how the cards interact, boom, boom, boom, boom."

**The board state is not established.** The lines "jump in with board states that
are unknown — it references creatures, it doesn't really describe the board
state." The fix named explicitly, as a fixed order:

1. This is what the combo is
2. This is the card on the stack
3. The existing board state is like this
4. Here's what the opponent has
5. Here's what you have
6. Here's what the stack is

Note this is a **presentation** problem, not an evidence problem — the underlying
scenario JSON already carries `board.you`, `board.opponents`, `hand` and `stack`
as structured fields. The renderer is not surfacing them as the reader's entry
point.

**Keep:** linking to the rules, and the accordions. Both called out as good.

## 4. Cover and titles

- **Drop the words "verified" and "bounded" from the cover furniture.** "You don't
  need to delete the verified/bounded — everything should be verified. If
  something's bounded, we say it's bounded." I.e. verification is the baseline
  promise and does not need announcing; boundedness is a claim about a specific
  line and belongs on that line.
- **The date is the same on every issue**, which reads as odd across the rack.
- **"Mana Map / Pilot's Manual" is fine. The volume number is fine.**
- **Titles that land and should set the register:** *The Seven-Token Verdict* ·
  *Seven damage, then the ground shakes* · *One Ping, Five Payoffs* · *The Enrage
  Web* · *Ten Thousand Goldfish Hands* · *The Flight Plan*.
- **The problem with the weaker ones is that they read as generated after the
  fact** — as the output of a rules-confirmation run rather than as an opening.
  The complaint is about how the issue *kicks off*, not about wordcraft.

## 5. The three voices are not distinct

"It still all feels very much like one voice." The masthead already names three
columnists and `issue_spec` already assigns bylines per department:

- **Coach Sunny Brightside** — ★ coaching (6 solo bylines, 2 shared)
- **"Ledger" Lin Marginal** — ◆ data (4 solo bylines)
- **Counselor Vera Dictum** — ✓ rules (2 solo bylines, 1 shared)

The ask is that each have a genuinely distinct voice on the page: "if it's Coach
Sunny Brightside talking, we want Coach Sunny Brightside." The pilot also raised
the possibility of **other members of the team** beyond the current three.

---

## What this touches, if it is acted on

Nothing here has been changed. For sizing:

- Reordering departments is `issue_spec.DEPARTMENTS`, which is the single source
  of truth — `STYLEv3.md`'s table and `tests/test_docs_section_count.py` follow it.
- A reorder changes every issue's `issue_plan.json`, so it MISSes `issue-plan`
  fleet-wide (8 decks).
- Voice differentiation is a charter edit to `magazine-editor` and the prose
  agents, which invalidates their routines by design.
- Combo-line presentation is largely a **renderer** change (`build_manual.py` +
  `design.py`) working off scenario fields that already exist — the cheapest item
  on this list, and the one with no agent cost at all.


---

## What shipped, thread by thread

**§1 Too long** — The Kill is **27,872 → 22,033 words fleet-wide (-21%)**. Two
causes, one per phase: combo-line prose roughly halved once the renderer showed the
board, and `scenario.question` moved out of the read-through (below). A single
gishath Kill entry runs 393 words against a 426 pre-work baseline — shorter than
before, while now showing a board it never showed. Judge's Desk was not shortened;
it is collapsed by default and its weight is opt-in.

**§2 Running order** — shipped whole as STYLEv3 **v3.4**. The Command Zone opens
the book, The Game Plan segues, The 99 follows, and Keep or Ship is read after the
roster. The Kill closes Act II. *Traded away:* v3.2's "stop at any act boundary and
get a complete, shallower book" property — the depth ramp no longer rises
monotonically. *Bought:* the first three sections answer the question a player asks
first, and Acts III/IV became single-voice.

**§3 Combo lines** — the board block renders in the order asked for: what you have
→ what each opponent has → mana → the stack, top-first. It reuses `scenario_facts`,
which already parsed all of it. Rules links and accordions kept, untouched.
*Additional finding:* `scenario.question` was a **resolver brief** being published
verbatim ("confirm each is a Dinosaur creature card…"), so it moved to Judge's Desk.

**§4 Cover and titles** — "VERIFIED"/"BOUNDED" removed from all kickers by deletion
only; the rule is now in STYLEv3 §5.1 and the editor charter. Dates run Vol. 004
November 2026 → Vol. 008 March 2027. Coverlines were not touched — the ones named
as landing are the register the charter now points at.

**§5 Three voices** — the cause was structural: `manual-writer` writes under all
three bylines in one pass, and its charter contradicted itself. Fixed by making the
personas mechanically separable — chiefly **what each does with a number** — and all
eight decks re-voiced.

## Still open

- **`pilot-coach` was deliberately not touched.** It owns four routines and would
  cost roughly 1M; Coach is the one voice already authored by a single persona.
- **`magazine-editor`'s per-columnist guidance is still a name-check.** It writes
  every kicker, headline, dek and caption in the issue, so the packaging layer is
  still monovocal.
- **Nothing validates that prose matches its byline** and nothing mechanically can.
- **The founder raised adding columnists beyond the three.** Not acted on.
- **edgar-vampires** still has four stale non-prose artifacts from the swap pass.
