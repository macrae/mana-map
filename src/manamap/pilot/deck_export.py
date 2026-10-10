"""A deck's list as text another tool's import box takes — Moxfield, Arena, or plain.

`manamap pilot deck-export <slug> [--version V] [--branch B]
                                  [--format moxfield|arena|plain] [--out F]`

WHY TEXT AND NOTHING ELSE. Moxfield has no public API and no write API, and its
Cloudflare front answers every server-side request with a 403 (docs/integrations.md,
"Moxfield"). So nothing in this package talks to Moxfield: publishing a deck there is
this command's output, pasted by the pilot into Moxfield's import box, and the deck's
Moxfield URL is recorded afterwards by hand with `deck-link`. The command reads one
list and prints it; it writes nothing unless `--out` names where.

THE THREE FORMS, and what each is for:

  moxfield  `check_in.render_decklist` byte for byte — `Commander:` / `Deck:` /
            `Sideboard:` headers, `N Name (SET) CN`, `*F*` for a foil. Moxfield's
            import box ("Import" on a new deck, or Bulk Edit on an existing one)
            accepts this form as it stands, printings and finishes included, and it
            is also the form `check-in --from -` reads back: `parse_decklist` of it
            returns the deck's own entries, boards and commander included.
  arena     MTG Arena's import form: `N Name (SET) CN` lines with NO section
            headers — the commander first, then the deck — and, when there is one,
            a blank line and the sideboard (that blank line is how Arena tells the
            two apart). Foil markers are dropped: Arena has no finish to import.
  plain     `N Name` only, sections kept — for a forum post, a message, or any
            importer that chokes on printing annotations.

WHICH LIST. The deck's `decklist.txt`; `--branch B` a branch's; `--version V` (a
number, a tag or a sha prefix, as `deck-version` resolves it) that version's text
out of git, via `deck_versions.blob_at`, so the list a past game was played on can be
exported without restoring it. `--version` and `--branch` together is refused: a
version is the deck's history, and a branch has none of its own.
"""

from manamap.pilot import check_in
from manamap.pilot.common import deck_dir, resolve_out_path
from manamap.pilot.fetch_deck import parse_decklist

FORMATS = ("moxfield", "arena", "plain")


def source_text(slug, version=None, branch=None):
    """The decklist text to export, and a label saying which list it is."""
    if version and branch:
        raise SystemExit("deck-export: --version reads the deck's history out of git and "
                         "--branch reads a branch's working list — pass one, not both")
    if version:
        from manamap.pilot import deck_versions
        v = deck_versions.resolve(slug, version)
        if v is None:
            raise SystemExit(f"{slug}: no version {version!r} — "
                             f"`manamap pilot deck-version {slug} list` names them")
        text = deck_versions.blob_at(slug, v)
        if text is None:
            raise SystemExit(f"{slug}: version {version!r} resolved to "
                             f"{v['first_sha'][:10]}, which git cannot show")
        return text, f"{slug} V{v['version']}"
    path = deck_dir(slug, branch) / "decklist.txt"
    if not path.exists():
        raise SystemExit(f"{path} not found")
    return path.read_text(encoding="utf-8"), (f"{slug}@{branch}" if branch else slug)


def _bare(e):
    """The entry with its printing and finish stripped — `N Name` survives."""
    return {k: v for k, v in e.items() if k not in ("set", "collector_number", "foil")}


def render(entries, fmt="moxfield"):
    """Entries as the text `fmt`'s import box takes. One trailing newline."""
    if fmt == "moxfield":
        return check_in.render_decklist(entries)
    if fmt == "plain":
        return check_in.render_decklist([_bare(e) for e in entries])
    if fmt == "arena":
        main = [e for e in entries if e.get("board", "main") == "main"]
        side = sorted((e for e in entries if e.get("board", "main") == "side"),
                      key=lambda e: e["name"])
        cmds = [e for e in main if e.get("is_commander")]
        deck = sorted((e for e in main if not e.get("is_commander")),
                      key=lambda e: e["name"])
        line = lambda e: check_in.render_line({**e, "foil": False})  # noqa: E731
        out = [line(e) for e in cmds + deck]
        if side:
            out.append("")
            out.extend(line(e) for e in side)
        return "\n".join(out) + "\n"
    raise SystemExit(f"deck-export: unknown format {fmt!r} — one of {', '.join(FORMATS)}")


def export(slug, fmt="moxfield", version=None, branch=None):
    """(text, label): the rendered list and which list it was."""
    text, label = source_text(slug, version=version, branch=branch)
    entries = parse_decklist(text)
    if not entries:
        raise SystemExit(f"{label}: the decklist parsed to nothing — nothing to export")
    return render(entries, fmt), label


def main(args):
    slug = args.slug
    fmt = getattr(args, "format", None) or "moxfield"
    text, label = export(slug, fmt=fmt, version=getattr(args, "version", None),
                         branch=getattr(args, "branch", None))
    out = getattr(args, "out", None)
    if out:
        # SLUG-SCOPED like every per-deck view: a generic name in a shared scratch
        # directory is how one deck's list silently becomes another's.
        path = resolve_out_path(out, slug, f"deck-export-{fmt}", ext=".txt")
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path} ({label}, {fmt})")
        return
    print(text, end="")
