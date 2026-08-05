# Pilot's Manual — founder feedback, 2026-08-05

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
