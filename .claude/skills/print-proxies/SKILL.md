---
name: print-proxies
description: Print a proxy sheet — a branch's adds (several branches into one PDF), minus any cards the pilot says to leave out, laid out 3x3 at true card size (63x88 mm) with crop marks, written to the Desktop and opened. Use when the pilot wants to play a staged branch's list before the cards arrive, or asks to print proxies for any cards.
---

# Print proxies

One command does the work: `manamap pilot proxies`. It takes a branch's ADDS
(`deck_branch.diff`, never basics), drops whatever the pilot names with `--have`,
prints both faces of a double-faced card, fetches Scryfall's 745x1040 PNG render
of each card (cached in `data/cache/proxies/`), and writes a Letter (or `--paper
a4`) PDF at 300 dpi. The bench keeps no inventory (2026-11-22): `--have` is the
pilot's own word and nothing else, and `/jarvis` may run this command when he
asks for the proxies of a branch — dry run, the list read back, then the print.

1. **Ask, don't assume.** Which branches is the pilot printing for, and which
   of their adds should be left off the sheet? Take that list in their words —
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

Never report which other deck holds a card, and never build a pull list or an
order list — the pilot names what to leave off; the sheet is the rest.
