# Shared decklist fixtures

One input, one hand-authored expectation, **two** parsers checked against it:
`src/manamap/pilot/fetch_deck.py:parse_decklist` (Python) and
`viz/js/decklist.js` (the browser). See `tests/test_decklist_parity.py`.

**The expected files are hand-written on purpose.** Generating them from the
Python parser would make Python the oracle and the parity property would be
theatre — both sides would agree with each other's bugs.

**The contract is a projection, not full equality.** Only
`{name, quantity, is_commander, board}` is compared. Python additionally
resolves printings against Scryfall (`set`, `collector_number`, `foil`); the viz
has no use for those and deliberately strips-and-discards the annotation. That
matters because the printing regex is exactly where the one real hazard lives —
see `printings.txt`.

## The hazard

`_PRINTING_RE` is anchored to `$`, so `*F*` and `*CMDR*` must come off the end of
the line **before** it runs. Reverse that order and every foil line silently keeps
its printing suffix inside the card name. `printings.txt` line 2 is that case, and
`tricky_names.txt` guards the inverse: an inline ` // ` in a double-faced card name
must not be mistaken for a comment.

## Comment-style section headers

`comment_markers.txt` is the Moxfield/Archidekt export shape: the commander sits
under a `// COMMANDER` header and there is **no matching `// DECK`** — a blank
line is what ends the section. Both parsers used to strip `//` lines before ever
testing for a marker, so the header was swallowed and the whole list imported
with no commander.

Two traps the fixture pins, and they pull in opposite directions:

- A comment is a marker only when its **whole** text is one. Lines 5 and 7 both
  contain the word "commander" in prose and must stay comments — otherwise a
  note about the deck silently re-sections it.
- A blank line closes a section **only** when that section was entered from a
  comment. `basic.txt` has the identical commander/blank/deck shape with an
  explicit `Deck` marker, and must keep parsing exactly as it always has.

The trailing `// SIDEBOARD` proves a comment-style sideboard header is read
too — and that a blank line inside it does NOT close it, the way a blank closes
a `// COMMANDER` section. A sideboard is the last section of every export, so
`Containment Priest`, after the blank, is still `side`.

## The sideboard is read, not skipped (2026-10-09)

Both parsers used to stop at a `Sideboard:` line, which is why no 60-card list
could be imported whole. Now the section switches, a `Deck:` after it returns to
the mainboard, and every entry carries `board: main | side`. `basic.txt`,
`comment_markers.txt` and `comments_and_aliases.txt` keep their trailing
sideboard lines and now expect them as `side` entries (a Moxfield `Maybeboard`
counts as the sideboard section, as it always did in the marker set).
`sixty_with_sideboard.txt` is the Standard shape: no commander, four-ofs, sixty
in the library and fifteen under `Sideboard:`. Note `Urabrask's Forge` sits on
both boards — two entries, never a quantity of four.
