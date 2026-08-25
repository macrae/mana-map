"""`manamap pilot brew` — the cards you kept become a deck. PRD §7.4.

The last step of the brew flow:

    library -> archetype -> role template -> candidates -> BUILD-OUT -> the bench

Everything upstream already existed by the time this was written, which is what
the PRD means about the program being mostly glue. What this adds is the
scaffold: a `brief.json` naming the commander, the cards the pilot kept, and —
new — the STYLE, which is what makes the role budget the archetype's rather than
the one flat provisional set every deck used to get.

    manamap pilot brew zur-voltron --commander "Zur the Enchanter" \\
        --theme voltron --from library.txt --build

Three ways in, and they are the three ways a library actually arrives: named on
the command line, from a file, or from the `brief.json` the Atlas exports — the
shape `Discovery.brief()` already emits, so a walk in the browser continues here
without a translation step.

A NEW DECK LANDS ON THE BENCH, NOT SLEEVED. No `paper` block is written and none
should be: §7.4 says v0.1.0, and 0.x is the version of a list that exists only
digitally. Reaching 1.0.0 is the act of sleeving it, which only the pilot can do
and which `deck-version paper` already proposes when they do.
"""

import json
import sys

from manamap.pilot import build_deck
from manamap.pilot.common import deck_dir


def _library_from_file(path):
    """Cards from a file — a plain list, a decklist, or an exported brief.

    The Atlas exports `brief.json` with `must_include` already resolved, so that
    is read as-is rather than re-parsed; anything else goes through the repo's
    ONE decklist parser, which handles quantities and `*CMDR*` for free.
    """
    text = sys.stdin.read() if path == "-" else open(path, encoding="utf-8").read()
    stripped = text.lstrip()
    if stripped.startswith("{"):
        doc = json.loads(text)
        return list(doc.get("must_include") or []), doc.get("commander")

    from manamap.pilot.fetch_deck import parse_decklist

    entries = parse_decklist(text)
    commander = next((e["name"] for e in entries if e.get("is_commander")), None)
    return [e["name"] for e in entries if not e.get("is_commander")], commander


def main(args):
    library = list(args.library)
    from_commander = None
    if args.from_file:
        found, from_commander = _library_from_file(args.from_file)
        library.extend(n for n in found if n not in library)

    commander = args.commander or from_commander
    if not commander:
        raise SystemExit("no commander — pass --commander, or a file that names one")

    path, brief = build_deck.scaffold_brief(
        args.slug, commander, library=library,
        theme=args.theme, bracket=args.bracket)

    print(f"Wrote {path}")
    print(f"  {commander} · {len(library)} card(s) kept · bracket {brief['bracket']}")
    if args.theme:
        print(f"  style: {args.theme} — its role histogram will shape the budget")
    else:
        print("  no style: the budget falls back to the flat provisional one "
              "(`archetypes \"<commander>\"` lists the styles)")

    if not args.build:
        print(f"\n  next: manamap pilot build-deck {args.slug} --write-decklist")
        return

    build_deck.main(type("A", (), {"slug": args.slug, "write_decklist": True})())
    print(f"\n  ON THE BENCH — not sleeved. Commit decklist.txt to make it V1, then:")
    print(f"      manamap pilot deck-version {args.slug} tag v0.1.0 --at V1 "
          f"--note \"first build\"")
    print("  v0.x is a list; sleeving it is what makes it v1.0.0.")
