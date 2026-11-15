# The teach series' cards

The pilot's own cards for the five mono-colour teaching decks (`data/decks/teach-*`):
the deck pickers a newcomer chooses from (each with a How to Play back that names the deck's own cards — update it if the list changes), and the rules references that sit beside the
table. Art and layout are the pilot's (made from prompts written 2026-11-13/14); the text
on each card is what the decks actually teach, from their Deck Contexts.

| File | Card |
|---|---|
| `01-teach-red-the-rush.jpg` | TEACH RED: The Rush — difficulty ●○○ |
| `01b-teach-red-how-to-play.jpg` | TEACH RED: How to Play (the back — the plan and the deck's key cards) |
| `02-teach-white-the-shield-wall.jpg` | TEACH WHITE: The Shield Wall — ●○○ |
| `02b-teach-white-how-to-play.jpg` | TEACH WHITE: How to Play (the back — the plan and the deck's key cards) |
| `03-teach-green-the-wild-growth.jpg` | TEACH GREEN: The Wild Growth — ●●○ |
| `03b-teach-green-how-to-play.jpg` | TEACH GREEN: How to Play (the back — the plan and the deck's key cards) |
| `04-teach-black-the-grave-bargain.jpg` | TEACH BLACK: The Grave Bargain — ●●○ |
| `04b-teach-black-how-to-play.jpg` | TEACH BLACK: How to Play (the back — the plan and the deck's key cards) |
| `05-teach-blue-the-clever-trick.jpg` | TEACH BLUE: The Clever Trick — ●●● |
| `05b-teach-blue-how-to-play.jpg` | TEACH BLUE: How to Play (the back — the plan and the deck's key cards) |
| `10-a-turn-step-by-step.jpg` | A Turn, Step by Step |
| `11-combat-in-four-beats.jpg` | Combat in Four Beats |
| `12-how-to-win.jpg` | How to Win |
| `13-your-opening-hand.jpg` | Your Opening Hand |
| `14-card-types.jpg` | Card Types |
| `15-paying-mana.jpg` | Paying Mana |
| `16-the-stack.jpg` | The Stack (blue's lesson) |
| `17-best-matchups.jpg` | Best Matchups (the colour wheel: allies, rivals, five lesson pairings) |
| `18-keywords.jpg` | Keywords (ten evergreen abilities, each with a definition) |
| `19-the-zones.jpg` | The Zones (library, hand, battlefield, graveyard, exile, the stack) |
| `20-gone-or-stopped.jpg` | Gone or Stopped (destroy, exile, sacrifice, can't attack or block, doesn't untap) |
| `21-reading-your-cards.jpg` | Reading Your Cards (when, whenever, at the beginning of, until end of turn, tokens, +1/+1 counters) |

JPEG at quality 92, 1061 x 1483 — the 63:88 shape of a real card, so they print true
to size. Superseded art lives in `v1/` and is not tracked.

## Printing

```bash
manamap pilot proxies --image data/teach/cards/10-a-turn-step-by-step.jpg \
                      --image data/teach/cards/11-combat-in-four-beats.jpg …
```

Name a file twice to print two copies. Print at 100% / Actual size, never "fit to page",
and cut on the crop marks.
