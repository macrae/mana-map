"""Pilot: check-in — a paper deck arrives, and the repo learns what it now is.

WHY THIS EXISTS. The pilot rebuilds decks in cardboard and the repo finds out
afterwards, by hand. The recipe was real and it was written down in three
places: diff the pasted list against `decklist.txt` with the repo's own parser,
report PULL/ADD as COPIES, apply, `fetch-deck`, `goldfish`, `mana-analysis`,
commit. Every step of that is mechanical and every step of it was being done
from memory, which is how a list gets applied with a card counted once that the
paper holds twice.

It is one command because of what step four buys. `decklist.txt` is tracked, so
the commit that carries a new list is what `deck-version` numbers and what the
captain's log stamps its games against — the whole git-log history of a deck is
a side effect of checking it in properly. Do it by hand and skip the commit, and
the games you play tonight attach to no version at all.

WHAT IT REFUSES, AND WHY IT REFUSES RATHER THAN GUESSES. A paper list is typed
by a human reading sleeves, so it arrives with the errors that come from that: a
card written twice, a name misremembered, ninety-nine cards where there should
be a hundred. Every one of those is silently survivable — `fetch-deck` would
resolve what it could and move on — and every one produces a repo list that is
not the deck on the table. That is worse than no check-in, because everything
downstream then measures a deck nobody owns. So the diff is a REPORT by default
and `--write` is refused while anything is wrong.

DFC names normalise on ` // ` and quantities are counted as copies, both because
`parse_decklist` is the shared contract the browser importer is fixture-locked
to. Counting entries instead of copies is the mistake this repo has made and
documented before: it published "18 lands" for a 33-land deck.
"""

import argparse
import hashlib
import shutil
import sys
from collections import Counter
from types import SimpleNamespace

from manamap.pilot.card_pool import corpus_names
from manamap.pilot import formats
from manamap.pilot.common import deck_dir
from manamap.pilot.fetch_deck import parse_decklist

# This module used to declare its own `DECK_SIZE = 100`, shadowing the one in
# `config.py` — a name that resolved locally to something a reader would swear
# came from the shared constant. The format spec is the one place now.
DECK_SIZE = formats.DEFAULT.deck_size


def _copies(entries):
    """name -> copies. The shuffler's view, never the line-count view."""
    c = Counter()
    for e in entries:
        c[e["name"]] += int(e.get("quantity") or 1)
    return c


def _is_basic(name):
    return name in {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}


def read_list(path_or_dash):
    if str(path_or_dash) == "-":
        return sys.stdin.read()
    with open(path_or_dash, encoding="utf-8") as f:
        return f.read()


def render_decklist(entries):
    """Entries back to the repo's canonical `decklist.txt` form.

    Canonical rather than verbatim, so the tracked file stays diffable and two
    check-ins of the same 99 produce the same bytes. That costs nothing:
    `deck-history` and `deck-version` compare PARSED entries, so reformatting
    can never manufacture a version — only a real change to the 99 does.

    Printing annotations and foil markers ride through when the pasted list
    carried them, because `fetch-deck` resolves exact printings from them and
    dropping them would silently re-resolve a Secret Lair to its cheapest
    reprint.
    """
    def line(e):
        s = f"{int(e.get('quantity') or 1)} {e['name']}"
        if e.get("set") and e.get("collector_number"):
            s += f" ({str(e['set']).upper()}) {e['collector_number']}"
        if e.get("foil"):
            s += " *F*"
        return s

    cmds = [e for e in entries if e.get("is_commander")]
    deck = sorted((e for e in entries if not e.get("is_commander")),
                  key=lambda e: e["name"])
    out = []
    if cmds:
        out.append("Commander:")
        out.extend(line(e) for e in cmds)
        out.append("")
    out.append("Deck:")
    out.extend(line(e) for e in deck)
    return "\n".join(out) + "\n"


def analyze(slug, text):
    """The diff, plus everything wrong with the pasted list.

    `blocking` is the half that stops `--write`. `warnings` is the half worth
    seeing and not worth refusing over — a check-in that cannot be applied until
    the corpus is rebuilt would make a fresh clone unable to accept a deck.
    """
    entries = parse_decklist(text)
    path = deck_dir(slug) / "decklist.txt"
    before = parse_decklist(path.read_text(encoding="utf-8")) if path.exists() else []

    new, old = _copies(entries), _copies(before)
    total = sum(new.values())
    commanders = sorted({e["name"] for e in entries if e.get("is_commander")})
    was_commander = sorted({e["name"] for e in before if e.get("is_commander")})

    blocking, warnings = [], []

    if not entries:
        blocking.append("the pasted list parsed to nothing — wrong file, or a format "
                        "`parse_decklist` does not read")
    if not commanders:
        blocking.append("no commander: put it under a `Commander:` header or mark the "
                        "line `*CMDR*`")
    if total != DECK_SIZE:
        blocking.append(f"{total} cards, not {DECK_SIZE} — a {formats.DEFAULT.name} deck is the "
                        f"commander plus 99")

    # A name written twice is the characteristic paper-list error: you read the
    # sleeve, write it down, and meet it again forty cards later. Singleton makes
    # every one of them illegal, and applying it silently would put a card in the
    # repo that the table cannot legally hold.
    dupes = sorted(n for n, k in new.items() if k > 1 and not _is_basic(n))
    if dupes:
        blocking.append(f"{len(dupes)} non-basic card(s) listed more than once, which "
                        f"singleton forbids — check the box: {', '.join(dupes)}")

    known = corpus_names()
    if known is None:
        warnings.append("no card corpus on this machine, so names were not checked "
                        "against it — run `manamap extract` to enable that")
    else:
        unknown = sorted(n for n in new if n not in known)
        if unknown:
            blocking.append(f"{len(unknown)} name(s) match no card in the corpus — a typo "
                            f"here becomes a card the deck does not have: "
                            f"{', '.join(unknown)}")

    # Only when the new list HAS a commander. With none parsed, the blocking error
    # above already says so, and this rendered as "Edgar Markov -> ." — an empty
    # arrow that reads as a data bug rather than as the missing header it is.
    if was_commander and commanders and commanders != was_commander:
        warnings.append(f"the commander changed: {', '.join(was_commander)} -> "
                        f"{', '.join(commanders)}. That is a different deck; consider a "
                        f"new slug rather than a new version of this one")

    pull = {n: old[n] - new.get(n, 0) for n in old if old[n] > new.get(n, 0)}
    add = {n: new[n] - old.get(n, 0) for n in new if new[n] > old.get(n, 0)}
    return {
        "slug": slug,
        "entries": entries,
        "cards": total,
        "commanders": commanders,
        # PULL leaves the sleeves, ADD goes in. Named for the hands, same as the
        # paper-lock drift, because that is what the pilot does with the answer.
        "pull": dict(sorted(pull.items())),
        "add": dict(sorted(add.items())),
        "unchanged": sum(min(old.get(n, 0), k) for n, k in new.items()),
        "blocking": blocking,
        "warnings": warnings,
        "decklist_sha256_before": (hashlib.sha256(path.read_bytes()).hexdigest()
                                   if path.exists() else None),
    }


def apply(slug, entries, run_chain=True):
    """Write the list, then re-derive what depends on it.

    The chain is not optional in spirit: `goldfish_metrics.json` and
    `mana_analysis.json` stamp the decklist sha, so leaving them behind makes the
    deck read as stale forever and every downstream figure describe a list that
    is gone. `--no-chain` exists for the case where the corpus is absent.
    """
    path = deck_dir(slug) / "decklist.txt"
    if path.exists():
        shutil.copy(path, path.with_suffix(".txt.bak"))
    path.write_text(render_decklist(entries), encoding="utf-8")
    ran = []
    if run_chain:
        from manamap.pilot import fetch_deck, goldfish, mana_analysis
        for name, mod in (("fetch-deck", fetch_deck), ("goldfish", goldfish),
                          ("mana-analysis", mana_analysis)):
            mod.main(SimpleNamespace(slug=slug))
            ran.append(name)
    return ran


def _print(d, write):
    print(f"CHECK-IN — {d['slug']}  ({d['cards']} cards, commander: "
          f"{', '.join(d['commanders']) or 'NONE'})\n")
    if not d["pull"] and not d["add"]:
        print("  the paper list and the repo's already agree — nothing to apply\n")
    else:
        print(f"  PULL {sum(d['pull'].values())} · ADD {sum(d['add'].values())} · "
              f"unchanged {d['unchanged']}\n")
        for n, k in d["pull"].items():
            print(f"    - {n}" + (f"  x{k}" if k > 1 else ""))
        for n, k in d["add"].items():
            print(f"    + {n}" + (f"  x{k}" if k > 1 else ""))
        print()
    for w in d["warnings"]:
        print(f"  warning: {w}")
    for b in d["blocking"]:
        print(f"  REFUSED: {b}")
    if d["blocking"]:
        print("\n  nothing was written. Fix the list and run it again; --force applies "
              "anyway, which you want approximately never.")
    elif not write:
        print("  dry run — add --write to apply, run the chain, and make it a version.")


def main(args):
    text = read_list(args.source)
    d = analyze(args.slug, text)
    if getattr(args, "as_json", False):
        import json
        print(json.dumps({k: v for k, v in d.items() if k != "entries"},
                         indent=2, ensure_ascii=False))
        return
    write = getattr(args, "write", False)
    _print(d, write)
    if not write:
        return
    if d["blocking"] and not getattr(args, "force", False):
        raise SystemExit(1)
    ran = apply(args.slug, d["entries"], run_chain=not getattr(args, "no_chain", False))
    print(f"\n  WROTE decklist.txt" + (f" · ran {' → '.join(ran)}" if ran else ""))
    print(f"  next: commit it — that is what makes it a version the log can stamp:")
    print(f"    git add data/decks/{args.slug} && git commit")
    print(f"    manamap pilot deck-version {args.slug} paper   # mark it as sleeved")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot check-in <slug> --from <file>`.")
