"""`manamap pilot commander-search` — the CLI over `analysis/commander_search.py`.

Thin on purpose. Every decision that could change a ranking lives in the
analysis module, which the eval also imports; this file only decides where the
seed came from and how to print the answer.

Three ways to hand it a seed, because three are the ways cards actually arrive:

    commander-search "Sol Ring" "Rhystic Study" …    named on the command line
    commander-search --from basket.txt               a file, or `-` for stdin
    commander-search --deck heliod                   an existing deck's own 99

The file form takes a decklist as-is — quantities and `*CMDR*` markers and all —
because the thing people have lying around is a decklist, and asking them to
strip it first is asking them to do the parser's job.
"""

import json
import sys

from manamap.analysis import commander_search as cs
from manamap.pilot.common import deck_dir, load_json


def _names_from_file(path):
    """Seed names out of a file or stdin, through the repo's ONE decklist parser.

    `fetch_deck.parse_decklist` is fixture-locked in parity with the browser's
    reader, and it already handles quantity prefixes, `*CMDR*`, printing
    suffixes and section markers. A bare list of names parses through it
    unchanged, so there is no second code path for "just names".
    """
    from manamap.pilot.fetch_deck import parse_decklist

    text = sys.stdin.read() if path == "-" else open(path, encoding="utf-8").read()
    return [e["name"] for e in parse_decklist(text)]


def _names_from_deck(slug):
    doc = load_json(deck_dir(slug) / "cards.json")
    if not doc:
        raise SystemExit(f"{slug}: no cards.json — `manamap pilot fetch-deck {slug}`")
    return [c["name"] for c in doc.get("cards", [])]


def main(args):
    if args.from_file:
        names = _names_from_file(args.from_file)
    elif args.deck:
        names = _names_from_deck(args.deck)
    elif args.cards:
        names = list(args.cards)
    else:
        raise SystemExit(
            "no seed — pass card names, `--from <file>` (or `-` for stdin), "
            "or `--deck <slug>`")

    result = cs.search(
        names,
        space=args.space,
        controlled=not args.no_type_control,
        per_identity=args.per_identity,
        limit=args.limit,
    )

    # §6.1 steps 9-10: open one of the results in the Atlas and harvest from it.
    #
    # The URL is the whole mechanism. `?ref=` seeds the graph from a reference
    # deck the same way `?cards=` seeds it from names, and every card panel has
    # carried a Keep button since the library shipped — so "select cards out of
    # that deck into your library" is a feature that already existed, pointed at
    # a list it could not previously see.
    rank = getattr(args, "open_rank", None)
    if rank is not None:
        if not 1 <= rank <= len(result["results"]):
            raise SystemExit(f"--open {rank}: there are {len(result['results'])} results")
        chosen = result["results"][rank - 1]["commander"]
        ref = cs.write_reference(chosen)
        print(f"\n{chosen} — {ref['resolved']} of {ref['cards']} cards on the map")
        if ref["unresolved"]:
            print(f"  not in the corpus: {', '.join(ref['unresolved'][:5])}")
        print(f"  wrote {ref['path']}")
        print(f"  open  http://localhost:8000/viz/index.html?ref={ref['slug']}")
        print("  then Keep the cards you want — they go to your library.")
        return

    # stdout is the answer; the progress went to stderr on the way here.
    if getattr(args, "as_json", False):
        print(json.dumps(result, indent=2))
    else:
        print(cs.format_report(result))
