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
import re
import shutil
import sys
from collections import Counter
from types import SimpleNamespace

from manamap.pilot.card_pool import corpus_names
from manamap.pilot import formats
from manamap.pilot import common as _common
from manamap.pilot.common import deck_dir
from manamap.pilot.fetch_deck import parse_decklist

# This module used to declare its own `DECK_SIZE = 100`, shadowing the one in
# `config.py` — a name that resolved locally to something a reader would swear
# came from the shared constant. Then it held `formats.DEFAULT.deck_size` at
# module scope, which is the same shadow one step removed: a constant bound at
# import cannot follow the deck. There is no module-level size now; `analyze`
# asks the deck's own spec (`formats.for_deck`) and nothing else.


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


def render_line(e, cmdr_marker=False):
    """One entry as one `decklist.txt` line: `N Name [(SET) CN] [*F*] [*CMDR*]`.

    THE ONE WRITER of the line form, shared by `render_decklist`, `set_printing`
    and `buy_list` (the Moxfield / Mana Pool "exact printings" paste is this same
    line), so a printing written by any of them parses back through
    `parse_decklist` to the same entry — `_PRINTING_RE` is `$`-anchored and the
    markers come off first, which is why the order here is printing, foil,
    commander and nothing may come after. A second copy is how a set code ends
    up upper-cased on one surface and not the other.
    """
    s = f"{int(e.get('quantity') or 1)} {e['name']}"
    if e.get("set") and e.get("collector_number"):
        s += f" ({str(e['set']).upper()}) {e['collector_number']}"
    if e.get("foil"):
        s += " *F*"
    if cmdr_marker:
        s += " *CMDR*"
    return s


# `buy_list`'s name for the same writer.
decklist_line = render_line


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
    line = render_line

    main = [e for e in entries if e.get("board", "main") == "main"]
    side = sorted((e for e in entries if e.get("board", "main") == "side"),
                  key=lambda e: e["name"])
    cmds = [e for e in main if e.get("is_commander")]
    deck = sorted((e for e in main if not e.get("is_commander")),
                  key=lambda e: e["name"])
    out = []
    if cmds:
        out.append("Commander:")
        out.extend(line(e) for e in cmds)
        out.append("")
    out.append("Deck:")
    out.extend(line(e) for e in deck)
    if side:
        # Written only when there is one, so a Commander list renders exactly
        # as it did before sideboards were read. A `*CMDR*` on a sideboard
        # line is carried through rather than dropped: the file keeps saying
        # what was pasted, and `validate_deck` is where it is called wrong.
        out.append("")
        out.append("Sideboard:")
        out.extend(line(e, cmdr_marker=bool(e.get("is_commander"))) for e in side)
    return "\n".join(out) + "\n"


def _boards(entries):
    """(mainboard, sideboard). An entry with no `board` key is mainboard — every
    caller that built entries by hand before the key existed."""
    main = [e for e in entries if e.get("board", "main") == "main"]
    side = [e for e in entries if e.get("board", "main") == "side"]
    return main, side


def analyze(slug, text, spec=None):
    """The diff, plus everything wrong with the pasted list — PER THE DECK'S FORMAT.

    `blocking` is the half that stops `--write`. `warnings` is the half worth
    seeing and not worth refusing over — a check-in that cannot be applied until
    the corpus is rebuilt would make a fresh clone unable to accept a deck.

    `spec` is the format every rule here reads: size (exact or at-least),
    copies, whether a commander is required, how big a sideboard may be. It
    defaults to `formats.for_deck(slug)` — the deck says what it is — and
    `main` passes `--format` through so a NEW deck declared Modern is analysed
    as Modern before any brief exists to resolve from.

    THE SIDEBOARD IS READ AND HELD TO THE SPEC. A constructed list's fifteen are
    kept, reported beside the main count, and refused past `sideboard_size`.
    On a format with none (Commander) the side entries are DROPPED with a
    warning rather than refused, because a Moxfield export of a Commander deck
    routinely carries a Maybeboard — which the parser files as `side` — and a
    pilot pasting their real list should not be blocked by cards they were
    only thinking about. Copies are counted main and side TOGETHER (CR 100.4a:
    the limit is across both), and the diff, the size and `cards` are the
    mainboard alone, since a sideboard card is not in the deck.
    """
    spec = spec or formats.for_deck(slug)
    entries = parse_decklist(text)
    path = deck_dir(slug) / "decklist.txt"
    before = parse_decklist(path.read_text(encoding="utf-8")) if path.exists() else []

    blocking, warnings = [], []

    main, side = _boards(entries)
    was_main, was_side = _boards(before)
    if side and not spec.sideboard_size:
        dropped = sum(int(e.get("quantity") or 1) for e in side)
        warnings.append(f"{dropped} card(s) after Sideboard:/Maybeboard dropped — "
                        f"{spec.name} has no sideboard")
        entries, side = main, []

    new, old = _copies(main), _copies(was_main)
    new_side, old_side = _copies(side), _copies(was_side)
    total = sum(new.values())
    side_total = sum(new_side.values())
    commanders = sorted({e["name"] for e in main if e.get("is_commander")})
    was_commander = sorted({e["name"] for e in was_main if e.get("is_commander")})

    if not entries:
        blocking.append("the pasted list parsed to nothing — wrong file, or a format "
                        "`parse_decklist` does not read")
    if spec.commanders and not commanders:
        blocking.append("no commander: put it under a `Commander:` header or mark the "
                        "line `*CMDR*`")
    size_problem = spec.size_error(total)
    if size_problem:
        blocking.append(f"{size_problem} ({spec.name})")
    if side_total > spec.sideboard_size:
        blocking.append(f"{side_total} sideboard cards — {spec.name} allows at most "
                        f"{spec.sideboard_size}")

    # A name written twice is the characteristic paper-list error: you read the
    # sleeve, write it down, and meet it again forty cards later. Singleton makes
    # every one of them illegal, and applying it silently would put a card in the
    # repo that the table cannot legally hold. Where the limit is four, the
    # count is main and sideboard together — that is the rule's own scope.
    combined = new + new_side
    over = sorted(n for n, k in combined.items()
                  if k > spec.max_copies and not _is_basic(n))
    if over and spec.singleton:
        blocking.append(f"{len(over)} non-basic card(s) listed more than once, which "
                        f"singleton forbids — check the box: {', '.join(over)}")
    elif over:
        blocking.append(f"{len(over)} non-basic card(s) with more than {spec.max_copies} "
                        f"copies, main and sideboard together (CR 100.4a): "
                        f"{', '.join(f'{n} x{combined[n]}' for n in over)}")

    known = corpus_names()
    if known is None:
        warnings.append("no card corpus on this machine, so names were not checked "
                        "against it — run `manamap extract` to enable that")
    else:
        # A NAME THE DECK ALREADY HOLDS IS KNOWN. ingris-infect was built by
        # hand around a commander the corpus will not carry until the set
        # releases; every branch of it was refused on its own commander's
        # name, which this list has carried since the deck existed.
        held = set(old) | set(old_side)
        unknown = sorted(n for n in combined if n not in known and n not in held)
        if unknown:
            blocking.append(f"{len(unknown)} name(s) match no card in the corpus — a typo "
                            f"here becomes a card the deck does not have: "
                            f"{', '.join(unknown)}")

    # Only when the new list HAS a commander. With none parsed, the blocking error
    # above already says so, and this rendered as "Edgar Markov -> ." — an empty
    # arrow that reads as a data bug rather than as the missing header it is.
    if (spec.commanders and was_commander and commanders
            and commanders != was_commander):
        warnings.append(f"the commander changed: {', '.join(was_commander)} -> "
                        f"{', '.join(commanders)}. That is a different deck; consider a "
                        f"new slug rather than a new version of this one")

    pull = {n: old[n] - new.get(n, 0) for n in old if old[n] > new.get(n, 0)}
    add = {n: new[n] - old.get(n, 0) for n in new if new[n] > old.get(n, 0)}
    return {
        "slug": slug,
        "format": spec.name.lower(),
        "entries": entries,
        "cards": total,
        "sideboard": side_total,
        "commanders": commanders,
        # PULL leaves the sleeves, ADD goes in. Named for the hands, same as the
        # paper-lock drift, because that is what the pilot does with the answer.
        # Mainboard only: a sideboard edit is not a swap and never a version
        # (`deck_history._entries` reads the same board), so it is a flag here.
        "pull": dict(sorted(pull.items())),
        "add": dict(sorted(add.items())),
        "unchanged": sum(min(old.get(n, 0), k) for n, k in new.items()),
        "sideboard_changed": new_side != old_side,
        "blocking": blocking,
        "warnings": warnings,
        # ONE DEFINITION. This was the second `read_bytes` caller, so a CRLF
        # decklist would have made the "before" sha disagree with every version
        # number derived from the same file.
        "decklist_sha256_before": (_common.list_sha256(
            path.read_text(encoding="utf-8")) if path.exists() else None),
    }


def chain_plan(spec):
    """What `apply` re-derives for a deck of this format, and what it skips.

    Returns `(stages, skipped)`: the stage names in order, and `{stage: why}`
    for each one the format cannot have. The goldfish is Commander-only —
    its seats, its commander in the command zone, its authored rates all
    assume the table (docs/simulation.md) — so a Standard deck's chain is
    `fetch-deck -> mana-analysis` and the report SAYS the goldfish was not
    run rather than leaving a reader to infer it from a missing file.
    """
    if spec.commanders:
        return ["fetch-deck", "goldfish", "mana-analysis"], {}
    return (["fetch-deck", "mana-analysis"],
            {"goldfish": f"not modelled for {spec.name} — the goldfish is Commander-only "
                         f"(docs/simulation.md)"})


def apply(slug, entries, run_chain=True, spec=None):
    """Write the list, then re-derive what depends on it.

    The chain is not optional in spirit: `goldfish_metrics.json` and
    `mana_analysis.json` stamp the decklist sha, so leaving them behind makes the
    deck read as stale forever and every downstream figure describe a list that
    is gone. `--no-chain` exists for the case where the corpus is absent.

    Returns `{"ran": [...], "skipped": {stage: why}}` — the stages that ran
    (none under `run_chain=False`) and the ones the deck's format has no use
    for, by `chain_plan`.
    """
    spec = spec or formats.for_deck(slug)
    path = deck_dir(slug) / "decklist.txt"
    if path.exists():
        shutil.copy(path, path.with_suffix(".txt.bak"))
    path.write_text(render_decklist(entries), encoding="utf-8")
    _, skipped = chain_plan(spec)
    return {"ran": _run_chain(slug) if run_chain else [], "skipped": skipped}


_CHAIN_MODULES = {"fetch-deck": "fetch_deck", "goldfish": "goldfish",
                  "mana-analysis": "mana_analysis"}


def _run_chain(slug, branch=None):
    """The deck's chain, per `chain_plan` of its format, on the deck or on one
    of its branches. Returns the names it ran. One definition, because `apply`
    and `set_printing` must re-derive the same artifacts."""
    import importlib
    stages, _ = chain_plan(formats.for_deck(slug, branch))
    ran = []
    for name in stages:
        mod = importlib.import_module(f"manamap.pilot.{_CHAIN_MODULES[name]}")
        mod.main(SimpleNamespace(slug=slug, branch=branch))
        ran.append(name)
    return ran


# `(SET) CN` as the CLI takes it — the parens optional, the set 2-6 letters or
# digits, the collector number anything `_PRINTING_RE` accepts (`123`, `123a`,
# `A-123`).
_PRINTING_ARG_RE = re.compile(r"^\(?\s*([A-Za-z0-9]{2,6})\s*\)?\s+([\w-]+)$")


def parse_printing_arg(text):
    """`"(SLD) 1234"` (or `"sld 1234"`) -> `("sld", "1234")`; SystemExit otherwise."""
    m = _PRINTING_ARG_RE.match(str(text or "").strip())
    if not m:
        raise SystemExit(f"{text!r} is not a printing — write it as `(SET) CN`, "
                         f"e.g. `(SLD) 1234`")
    return m.group(1).lower(), m.group(2)


def _matches(entry, name):
    """The entry IS this card: the full name, or the front face of a DFC — the
    corpus and `cards.json` key `A // B`, a pasted list often carries `A`."""
    want = str(name or "").strip().lower()
    have = entry["name"].lower()
    if want == have:
        return True
    return want.split(" // ")[0] == have.split(" // ")[0]


def set_printing(slug, name, set_code, collector_number, foil=False, run_chain=True,
                 branch=None):
    """Point ONE line of `decklist.txt` at an exact printing, and re-derive.

    The pilot sleeves a particular card — the borderless Secret Lair, the foil
    from the precon — and `cards.json` is supposed to show it (`fetch-deck`
    resolves `(SET) CN` first, by name only as the fallback). Until now the only
    way to say which one was to re-paste the whole list through `check-in`.
    This writes the annotation on the one line and leaves every other byte of
    the file alone, so the diff that lands in git is one line, and then runs
    the same chain `apply` runs: fetch-deck resolves the printing into
    cards.json, goldfish and mana-analysis re-stamp the new decklist sha.

    WHAT IT DOES NOT DO. It is not a new deck version — `deck_versions` keys a
    version on `(name, copies, commander)`, never on the printing — and it
    re-runs no agent: `agent_cache.CARD_SEMANTIC_FIELDS` excludes set, collector
    number and image on purpose (docs/agent-cost.md), so the prose about the
    card is as true after the swap as before. Idempotent: the printing already
    on the line is `changed: False` and runs nothing.

    `branch` scopes it to `branches/<name>/decklist.txt`, which is its own file
    with its own chain. Refuses a name the list does not hold, and a name it
    holds on more than one line (basics split across printings): say which
    line by naming the one you mean.
    """
    set_code = str(set_code or "").strip().lower()
    collector_number = str(collector_number or "").strip()
    if not set_code or not collector_number:
        raise SystemExit("a printing is a set code and a collector number: `(SLD) 1234`")
    path = (deck_dir(slug, branch) if branch else deck_dir(slug)) / "decklist.txt"
    if not path.exists():
        raise SystemExit(f"{path} not found")
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    # LINE BY LINE THROUGH THE SHARED PARSER. A header line parses to nothing; a
    # card line parses to one entry; the section it sits in is irrelevant to
    # whether it names this card. `*CMDR*` on the line itself is the one thing a
    # lone-line parse keeps that the rewrite must put back.
    hits = []
    for i, raw in enumerate(lines):
        parsed = parse_decklist(raw)
        if len(parsed) == 1 and _matches(parsed[0], name):
            hits.append((i, parsed[0], raw.strip().upper().endswith("*CMDR*")))
    if not hits:
        raise SystemExit(f"{slug}{' @' + branch if branch else ''}: {name!r} is not in "
                         f"decklist.txt — a printing is set on a card the list holds")
    if len(hits) > 1:
        raise SystemExit(f"{slug}: {name!r} is on {len(hits)} lines (lines "
                         f"{', '.join(str(i + 1) for i, _, _ in hits)}) — merge them, "
                         f"or edit the file: this sets one line")
    i, entry, cmdr_marker = hits[0]
    same = (entry.get("set") == set_code
            and entry.get("collector_number") == collector_number
            and bool(entry.get("foil")) == bool(foil))
    entry["set"], entry["collector_number"], entry["foil"] = (
        set_code, collector_number, bool(foil))
    line = render_line(entry, cmdr_marker=cmdr_marker)
    if same:
        return {"changed": False, "line": line, "ran": []}
    shutil.copy(path, path.with_suffix(".txt.bak"))
    lines[i] = line
    path.write_text("\n".join(lines), encoding="utf-8")
    ran = _run_chain(slug, branch) if run_chain else []
    return {"changed": True, "line": line, "ran": ran}


def _print(d, write):
    spec = formats.get(d.get("format"))
    head = f"{d['cards']} cards"
    if d.get("sideboard"):
        head += f" + {d['sideboard']} side"
    head += f", {spec.name}"
    if spec.commanders:
        head += f", commander: {', '.join(d['commanders']) or 'NONE'}"
    print(f"CHECK-IN — {d['slug']}  ({head})\n")
    if not d["pull"] and not d["add"] and not d.get("sideboard_changed"):
        print("  the paper list and the repo's already agree — nothing to apply\n")
    elif not d["pull"] and not d["add"]:
        print("  the mainboard agrees with the repo's; only the sideboard moved "
              "(not a version)\n")
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


def _main_set_printing(args):
    name, printing = args.set_printing
    set_code, cn = parse_printing_arg(printing)
    r = set_printing(args.slug, name, set_code, cn, foil=getattr(args, "foil", False),
                     run_chain=not getattr(args, "no_chain", False),
                     branch=getattr(args, "branch", None))
    if getattr(args, "as_json", False):
        import json
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return
    if not r["changed"]:
        print(f"  {args.slug}: already `{r['line']}` — nothing written, nothing run")
        return
    print(f"  WROTE decklist.txt: `{r['line']}`"
          + (f" · ran {' → '.join(r['ran'])}" if r["ran"] else " · chain skipped"))
    print(f"  not a new version (the 99 did not move) and no agent re-runs "
          f"(cards.json is hashed on its rules text, not its art)")
    print(f"  next: git add data/decks/{args.slug} && git commit")


def set_brief_format(slug, name):
    """Record the deck's format in `brief.json`, creating a minimal brief if
    there is none. Returns the brief written.

    THE FORMAT'S HOME IS THE BRIEF, and `fetch-deck` reads it from there, so a
    `check-in --format standard --write` is one declaration followed by the
    ordinary chain rather than a flag that has to be repeated on every later
    fetch. A brief with no commander is a legal brief for a format with none
    (`validate_brief` reads the same spec). `format_key` keeps a Commander
    declaration out of the file for the same reason it keeps it out of
    `cards.json`: absent is the default, and the fourteen tracked briefs must
    not change for having been checked in again.
    """
    import json
    spec = formats.get(name)
    path = deck_dir(slug) / "brief.json"
    brief = _common.load_json(path, None)
    key = formats.format_key(spec)
    if not isinstance(brief, dict):
        # A NEW brief says what the pilot said, the default included — there
        # is no tracked file to keep identical. Only an existing brief is left
        # alone when the declaration is the default it already implies.
        brief = {"slug": slug, "format": spec.name.lower()}
    elif key:
        brief["format"] = key
    elif brief.get("format") and formats.get(brief["format"]) is not formats.DEFAULT:
        # A declared non-default format being moved back to the default is
        # written explicitly (the Atlas's `brew` writes `"commander"` too); a
        # brief that already says the default, or says nothing, is left as is.
        brief["format"] = spec.name.lower()
    path.write_text(json.dumps(brief, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return brief


def main(args):
    if getattr(args, "set_printing", None):
        return _main_set_printing(args)
    if not getattr(args, "source", None):
        raise SystemExit("check-in needs --from <file> (a paper list) or "
                         "--set-printing \"Name\" \"(SET) CN\"")
    fmt = getattr(args, "format", None)
    # Refuse an unknown name before any read. The spec is threaded from here so
    # a NEW deck declared `--format modern` is analysed as Modern now, before
    # any brief exists for the resolver to read it from.
    spec = formats.get(fmt) if fmt else None
    text = read_list(args.source)
    d = analyze(args.slug, text, spec=spec)
    if getattr(args, "as_json", False):
        import json
        print(json.dumps({k: v for k, v in d.items() if k != "entries"},
                         indent=2, ensure_ascii=False))
        return
    write = getattr(args, "write", False)
    _print(d, write)
    if fmt and not write:
        print(f"  format: {formats.get(fmt).name} — recorded in brief.json on --write")
    if not write:
        return
    if d["blocking"] and not getattr(args, "force", False):
        raise SystemExit(1)
    if fmt:
        # BEFORE the list and the chain: `fetch-deck` resolves the format from
        # the brief, so the brief has to say it first or cards.json is written
        # as the default and the flag did nothing.
        set_brief_format(args.slug, fmt)
        print(f"  WROTE brief.json: format {formats.get(fmt).name}")
    r = apply(args.slug, d["entries"], run_chain=not getattr(args, "no_chain", False),
              spec=spec)
    ran = r["ran"]
    print(f"\n  WROTE decklist.txt" + (f" · ran {' → '.join(ran)}" if ran else ""))
    for stage, why in r["skipped"].items():
        print(f"  skipped {stage}: {why}")
    from manamap.pilot import deck_context
    deck_context.print_list_change(deck_context.list_change(args.slug))
    print(f"  next: commit it — that is what makes it a version the log can stamp:")
    print(f"    git add data/decks/{args.slug} && git commit")
    print(f"    manamap pilot deck-version {args.slug} paper   # mark it as sleeved")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot check-in <slug> --from <file>`.")
