# Shared decklist fixtures

One input, one hand-authored expectation, **two** parsers checked against it:
`src/manamap/pilot/fetch_deck.py:parse_decklist` (Python) and
`viz/js/decklist.js` (the browser). See `tests/test_decklist_parity.py`.

**The expected files are hand-written on purpose.** Generating them from the
Python parser would make Python the oracle and the parity property would be
theatre — both sides would agree with each other's bugs.

**The contract is a projection, not full equality.** Only
`{name, quantity, is_commander}` is compared. Python additionally
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
