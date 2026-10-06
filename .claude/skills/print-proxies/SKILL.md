---
name: print-proxies
description: Print a proxy sheet for the cards a pilot is waiting on — a branch's adds (several branches into one PDF) minus the cards already in hand, laid out 3x3 at true card size (63x88 mm) with crop marks, written to the Desktop and opened. Use when the pilot has ordered cards for a staged branch and wants to play the new list before the mail arrives, or asks to print proxies for any cards.
---

# Print proxies for the cards in the mail

One command does the work: `manamap pilot proxies`. It takes a branch's ADDS
(`deck_branch.diff`, never basics), drops what the pilot holds, prints both
faces of a double-faced card, fetches Scryfall's 745x1040 PNG render of each
card (cached in `data/cache/proxies/`), and writes a Letter (or `--paper a4`)
PDF at 300 dpi.

1. **Ask, don't assume.** Which branches is the pilot collecting for, and which
   of their adds are already in hand? Take the in-hand list in their words —
   pasted names, or a file (`1 Card Name` lines are fine).
2. **Dry run and confirm the list with the pilot** before anything downloads:

   ```bash
   .venv/bin/manamap pilot proxies <slug>@<branch> [<slug>@<branch> …] \
       --have "Card A" --have "Card B" [--have-file in_hand.txt] --dry-run
   ```

   Read the count back. A `--have` that matches nothing prints a WARNING —
   usually a typo, and the card it meant would be printed; fix it and re-run.
   One-off cards outside a branch: `--card "Name"` (repeatable).
3. **Print it.** The same command without `--dry-run` writes
   `~/Desktop/<slugs>-proxies.pdf` and opens it (`--dest PATH` to put it
   elsewhere, `--no-open` to skip opening).
4. **Tell the pilot how to print**: scale **100% / Actual size**, never "fit to
   page" (the cards come out too small for sleeves); cut on the crop marks
   outside the grid — the cards abut, so one cut separates two; sleeve each in
   front of a basic land.

Never report which other deck holds a card or build a pull list — the pilot
names what they have; the sheet is what they don't.
