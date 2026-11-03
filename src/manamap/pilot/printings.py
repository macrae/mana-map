"""Pilot: which physical card each line of a deck is — `manamap pilot art`.

The art is the pilot's. `decklist.txt`'s `N Name (SET) CN *F*` has always been able
to say it; this is the command that sets it, for a whole deck at once, on any deck
that is not archived:

    art <slug>                                   every card, its printing, pinned or not
    art <slug> --set "Name=SET CN [*F*]" ...     batch: one write, one rebuild
    art <slug> --split "Mountain=SLD 2418:8,SLD 2419:7,SLD 2420:7"
    art <slug> --pin-shown [--dry-run]           pin every unpinned card to what it shows
    art <slug> undo | redo

NOT A VERSION AND NOT A SWAP. `deck_versions` keys a version on names and copies, so
the commit joins the version it belongs to and a paper lock holds. That is why a
SLEEVED deck may do this when `edit` refuses it: the cardboard is unchanged, and the
digital deck is catching up with it. Nothing here changes a card's identity, and
`agent_cache.CARD_SEMANTIC_FIELDS` excludes the printing, so no agent re-runs.

The file keeps every other byte: `rewrite` touches only the lines of the cards it
pins. The write goes through `deck_edit.replace_list`, so it journals and undoes.
`check_in.set_printing` stays for one card from a page or a paper check-in.
"""

import json

from manamap.pilot import common, deck_edit
from manamap.pilot.common import deck_dir
from manamap.pilot.fetch_deck import fetch_printings, parse_decklist

SOURCE = "printing"
_key = deck_edit._key


def guard(slug):
    """Any deck but an archived one: it has no cardboard to describe."""
    deck_dir(slug)
    if common.deck_is_apart(slug):
        life = common.deck_lifecycle(slug)
        raise SystemExit(f"{slug} is archived ({life[0] if life else 'archived'}) — revive it "
                         f"first: `manamap pilot deck-state {slug} revive --reason \"…\"`")


# ── what a pin is ────────────────────────────────────────────────────────

def parse_printing(text):
    """`SLD 2406`, `(SLD) 2406`, either with `*F*` -> `{set, collector_number, foil}`."""
    s = " ".join(str(text or "").replace("(", " ").replace(")", " ").split())
    foil = False
    for marker in common.FOIL_MARKERS:
        if s.upper().endswith(marker):
            foil, s = True, s[: s.upper().rfind(marker)].strip()
            break
    parts = s.split(" ")
    if len(parts) != 2 or not parts[0].isalnum():
        raise SystemExit(f"{text!r} is not a printing — write it as `SET CN`, e.g. `SLD 2406`")
    return {"set": parts[0].lower(), "collector_number": parts[1], "foil": foil,
            "quantity": None}


def parse_set(arg):
    """`Name=SET CN [*F*]` -> `(name, [pin])`."""
    if "=" not in str(arg):
        raise SystemExit(f"--set takes \"Name=SET CN\", not {arg!r}")
    name, printing = str(arg).rsplit("=", 1)
    return name.strip(), [parse_printing(printing)]


def parse_split(arg):
    """`Mountain=SLD 2418:8,SLD 2419 *F*:7` -> `(name, [pin, pin])`."""
    if "=" not in str(arg):
        raise SystemExit(f"--split takes \"Name=SET CN:N,SET CN:N\", not {arg!r}")
    name, rest = str(arg).rsplit("=", 1)
    pins = []
    for part in rest.split(","):
        if ":" not in part:
            raise SystemExit(f"--split {part.strip()!r}: each printing needs its copies, `SET CN:N`")
        printing, n = part.rsplit(":", 1)
        pin = parse_printing(printing)
        if not n.strip().isdigit() or int(n) < 1:
            raise SystemExit(f"--split {part.strip()!r}: {n!r} is not a copy count")
        pin["quantity"] = int(n)
        pins.append(pin)
    return name.strip(), pins


def pins_from_ops(ops):
    """`deck_edit`'s printing ops as `{name: [pin]}`."""
    return {o["name"]: [{"set": o["set"], "collector_number": o["collector_number"],
                         "foil": bool(o.get("foil")), "quantity": None}]
            for o in ops if o["op"] == "printing"}


# ── the file ─────────────────────────────────────────────────────────────

def _card_lines(text):
    """Every card line: `(index, entry, has_cmdr_marker)`. Parsed cumulatively so a
    line reads under its own section header, which a lone-line parse cannot see."""
    lines = text.split("\n")
    out, seen = [], 0
    for i, raw in enumerate(lines):
        parsed = parse_decklist("\n".join(lines[: i + 1]))
        if len(parsed) > seen:
            out.append((i, parsed[-1], raw.strip().upper().endswith("*CMDR*")))
            seen = len(parsed)
    return out


def rewrite(text, pins):
    """`text` with each pinned card's lines replaced, every other byte alone.

    A card on several lines gathers onto its first (per board); a `--split` needs
    copies that sum to what the list holds, because art never changes a count.
    SystemExit names a card the list does not hold.
    """
    from manamap.pilot.check_in import render_line
    lines = text.split("\n")
    hits = _card_lines(text)
    replace, drop = {}, set()
    for name, wanted in pins.items():
        mine = [h for h in hits if _key(h[1]["name"]) == _key(name)]
        if not mine:
            raise SystemExit(f"{name!r} is not in this list — a printing is set on a card "
                             f"the list holds")
        for board in sorted({h[1].get("board", "main") for h in mine}):
            group = [h for h in mine if h[1].get("board", "main") == board]
            total = sum(int(h[1]["quantity"]) for h in group)
            said = [p["quantity"] for p in wanted if p.get("quantity") is not None]
            if said and sum(said) != total:
                raise SystemExit(f"{name}: the list holds {total}, and the split says "
                                 f"{sum(said)} — art never changes a count")
            first, entry, cmdr = group[0]
            replace[first] = [
                render_line({"name": entry["name"], "quantity": p.get("quantity") or total,
                             "set": p["set"], "collector_number": p["collector_number"],
                             "foil": p.get("foil")},
                            cmdr_marker=cmdr) for p in wanted]
            drop |= {h[0] for h in group[1:]}
    out = []
    for i, raw in enumerate(lines):
        if i in drop:
            continue
        out += replace.get(i, [raw])
    return "\n".join(out)


def check_pins(slug, pins):
    """Ask Scryfall about every pin. Returns the pins corrected, or SystemExit.

    Refuses a printing that does not exist, is another card, or was never on paper:
    the art would be wrong, or unbuyable. Foil follows the printing: a foil pin on a
    printing with no foil is refused, and a foil-only printing is corrected to foil
    and said so, since the copy cannot be anything else.
    """
    wanted = sorted({(p["set"], p["collector_number"]) for ps in pins.values() for p in ps})
    found = fetch_printings(wanted) if wanted else {}
    errors, notes = [], []
    for name, ps in pins.items():
        for p in ps:
            label = f"{name} ({p['set'].upper()}) {p['collector_number']}"
            sc = found.get((p["set"], p["collector_number"]))
            if sc is None:
                errors.append(f"{label}: Scryfall has no such printing")
                continue
            if _key(sc.get("name")) != _key(name):
                errors.append(f"{label}: that printing is {sc.get('name')!r}")
                continue
            games = sc.get("games")
            if games and "paper" not in games:
                errors.append(f"{label}: exists only on {', '.join(games)}, never on paper")
                continue
            finishes = sc.get("finishes") or []
            if p.get("foil") and finishes and not {"foil", "etched"} & set(finishes):
                errors.append(f"{label}: was never printed in foil")
            elif not p.get("foil") and finishes and "nonfoil" not in finishes:
                p["foil"] = True
                notes.append(f"{label}: foil only, so marked *F*")
    if errors:
        raise SystemExit("Refusing that art:\n  - " + "\n  - ".join(errors))
    return notes


# ── the reading ──────────────────────────────────────────────────────────

def _is_land(card):
    return "Land" in str((card or {}).get("type_line") or "")


def rows(slug, branch=None):
    """One row per line of the list, non-lands then lands, each alphabetical."""
    path = deck_dir(slug, branch) / "decklist.txt"
    if not path.exists():
        raise SystemExit(f"{path} not found")
    doc = common.load_json(deck_dir(slug, branch) / "cards.json") or {}
    by_name = {_key(c["name"]): c for c in (doc.get("cards") or []) + (doc.get("sideboard") or [])}
    out = []
    for e in parse_decklist(path.read_text(encoding="utf-8")):
        card = by_name.get(_key(e["name"])) or {}
        pinned = bool(e.get("set"))
        arts = {(a.get("set"), a.get("collector_number")): a for a in card.get("printings") or []}
        art = arts.get((e.get("set"), e.get("collector_number"))) or card
        out.append({"name": e["name"], "quantity": e["quantity"], "pinned": pinned,
                    "board": e.get("board", "main"), "is_commander": e.get("is_commander", False),
                    "set": e.get("set") or card.get("set"),
                    "collector_number": e.get("collector_number") or card.get("collector_number"),
                    "foil": bool(e.get("foil")), "artist": art.get("artist"),
                    "image": art.get("image"), "land": _is_land(card)})
    out.sort(key=lambda r: (r["land"], r["name"].lower(), str(r["collector_number"])))
    return out


def unpinned_pins(slug, branch=None):
    """`(pins, skipped)`: what `--pin-shown` would write, and the cards it will not.

    What the deck SHOWS is `cards.json`. A digital-only row is skipped and named:
    pinning it would order art nobody can buy. A card with some lines pinned and
    some not is the pilot's half-finished split, which only the pilot can finish.
    """
    table = rows(slug, branch)
    lines = {}
    for r in table:
        lines.setdefault(_key(r["name"]), []).append(r)
    pins, skipped = {}, []
    for rs in lines.values():
        if all(r["pinned"] for r in rs):
            continue
        if any(r["pinned"] for r in rs):
            skipped.append((rs[0]["name"], "some lines are pinned and some are not"))
            continue
        r = rs[0]
        if not (r["set"] and r["collector_number"]):
            skipped.append((r["name"], "cards.json shows no printing — rebuild the deck"))
            continue
        pins[r["name"]] = [{"set": r["set"], "collector_number": r["collector_number"],
                            "foil": False, "quantity": None}]
    # A digital-only printing cannot be pinned; everything else may need foil fixed.
    found = fetch_printings(sorted({(p[0]["set"], p[0]["collector_number"])
                                    for p in pins.values()})) if pins else {}
    for name, ps in list(pins.items()):
        sc = found.get((ps[0]["set"], ps[0]["collector_number"]))
        games = (sc or {}).get("games")
        if sc is None:
            skipped.append((name, "Scryfall does not know the printing shown"))
            del pins[name]
        elif games and "paper" not in games:
            skipped.append((name, f"shown on a {'/'.join(games)}-only printing — pick a paper one"))
            del pins[name]
        elif (sc.get("finishes") and "nonfoil" not in sc["finishes"]):
            ps[0]["foil"] = True
    return pins, sorted(skipped)


# ── the write ────────────────────────────────────────────────────────────

def apply(slug, pins, branch=None, note=None, run=True, validate=True, echo=None):
    """Write `pins`, then bring the deck's derived files along. Returns a report."""
    guard(slug)
    notes = check_pins(slug, pins) if validate else []
    path = deck_dir(slug, branch) / "decklist.txt"
    if not path.exists():
        raise SystemExit(f"{path} not found")
    before = path.read_text(encoding="utf-8")
    after = rewrite(before, pins)
    report = {"slug": slug, "branch": branch, "changed": after != before,
              "cards": sorted(pins), "notes": notes, "refresh": None}
    if after == before:
        return report
    label = note or "art: " + ", ".join(sorted(pins)[:6]) + (" …" if len(pins) > 6 else "")
    if branch:
        import shutil
        shutil.copy(path, path.with_suffix(".txt.bak"))
        path.write_text(after, encoding="utf-8")
    else:
        deck_edit.replace_list(slug, after, source=SOURCE, note=label, path=path)
    if run:
        report["refresh"] = refresh(slug, branch, echo=echo)
    return report


def refresh(slug, branch=None, echo=None):
    """Everything that quotes this deck's list or art, this deck only.

    `deck_edit.rebuild` covers the deck's own measurements. It is the handbook,
    the prices, the manifest and the fleet file that a list's sha also reaches,
    and `make manuals` compares those in CI — teach-red's V2 went red on exactly
    that, 2026-11-03. A failure is reported rather than raised: the art is written,
    and the rest can be run again.
    """
    import contextlib
    import importlib
    import io
    from types import SimpleNamespace
    from manamap import net
    from manamap.pilot import check_in, prices
    failures, ran = [], []
    if branch:
        ran += check_in._run_chain(slug, branch)
    else:
        got = deck_edit.rebuild(slug, echo=echo)
        ran += got.get("ran") or []
        failures += got.get("failures") or []
    if (deck_dir(slug, branch) / prices.ARTIFACT).exists():
        try:
            prices._write(deck_dir(slug, branch) / prices.ARTIFACT, prices.build(slug, branch))
            ran.append("prices")
        except (net.Offline, SystemExit) as exc:
            failures.append(f"prices: {exc} — `manamap pilot prices {slug} --write` when online")
    steps = [] if branch else [("build-index", "manamap.pilot.deck_manifest", {}),
                               ("fleet-analytics", "manamap.pilot.fleet_analytics",
                                {"write": True})]
    if not branch and (deck_edit._repo_root() / "manuals" / "p" / f"{slug}.html").exists():
        steps.insert(0, ("build-poh", "manamap.pilot.poh", {"slug": slug}))
    for name, dotted, kwargs in steps:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                importlib.import_module(dotted).main(SimpleNamespace(**kwargs))
            ran.append(name)
        except Exception as exc:                            # noqa: BLE001 - reported
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return {"ran": ran, "failures": failures}


# ── the command ──────────────────────────────────────────────────────────

def _print_table(table):
    pinned = sum(1 for r in table if r["pinned"])
    print(f"{pinned} of {len(table)} lines pinned")
    for land in (False, True):
        part = [r for r in table if r["land"] == land]
        if not part:
            continue
        print("LANDS" if land else "NON-LANDS")
        for r in part:
            where = f"({str(r['set']).upper()}) {r['collector_number']}" if r["set"] else "—"
            print(f"  {r['quantity']:>2} {r['name'][:38]:38} {where:18} "
                  f"{'foil' if r['foil'] else '    '} {'pinned' if r['pinned'] else 'shown '} "
                  f"{r['artist'] or ''}")


def main(args):
    slug = args.slug
    branch = getattr(args, "branch", None)
    action = getattr(args, "action", None)
    as_json = getattr(args, "as_json", False)
    chain = not getattr(args, "no_chain", False)
    dry = getattr(args, "dry_run", False)

    if action:
        if branch:
            raise SystemExit("a branch's list has no journal — `deck-branch log` shows its history")
        rec = (deck_edit.undo if action == "undo" else deck_edit.redo)(slug, art_only=True)
        got = refresh(slug) if chain else None
        print(f"{action.upper()} on {slug}: {rec.get('note') or 'art'}")
        for f in (got or {}).get("failures") or []:
            print(f"  FAILED {f}")
        return

    pins, skipped = {}, []
    for arg in getattr(args, "set", None) or []:
        name, ps = parse_set(arg)
        pins[name] = ps
    for arg in getattr(args, "split", None) or []:
        name, ps = parse_split(arg)
        pins[name] = ps
    if getattr(args, "pin_shown", False):
        shown, skipped = unpinned_pins(slug, branch)
        pins = {**shown, **pins}

    if not pins and not getattr(args, "pin_shown", False):
        table = rows(slug, branch)
        if as_json:
            print(json.dumps({"slug": slug, "branch": branch, "rows": table}, indent=2))
        else:
            _print_table(table)
        return

    if dry:
        path = deck_dir(slug, branch) / "decklist.txt"
        rewrite(path.read_text(encoding="utf-8"), pins)     # refuses what apply would
        doc = {"slug": slug, "branch": branch, "would_pin": sorted(pins),
               "skipped": [{"name": n, "why": w} for n, w in skipped]}
        if as_json:
            print(json.dumps(doc, indent=2))
        else:
            print(f"DRY RUN {slug}: {len(pins)} card(s) would be pinned; nothing written")
            for name in sorted(pins):
                p = pins[name][0]
                print(f"  {name} ({p['set'].upper()}) {p['collector_number']}"
                      + (" *F*" if p.get("foil") else ""))
            for name, why in skipped:
                print(f"  SKIPPED {name}: {why}")
        return

    report = apply(slug, pins, branch=branch, note=getattr(args, "note", None), run=chain)
    report["skipped"] = [{"name": n, "why": w} for n, w in skipped]
    if as_json:
        print(json.dumps(report, indent=2))
        return
    if not report["changed"]:
        print(f"{slug}: already on that art — nothing written")
    else:
        print(f"ART {slug}{' @' + branch if branch else ''}: {len(report['cards'])} card(s) "
              f"pinned. Not a version; a paper lock holds. Undo: `manamap pilot art {slug} undo`")
    for note in report["notes"]:
        print(f"  note: {note}")
    for name, why in skipped:
        print(f"  SKIPPED {name}: {why}")
    for f in (report["refresh"] or {}).get("failures") or []:
        print(f"  FAILED {f}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot art <slug>`.")
