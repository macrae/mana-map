"""Pilot: rewrite — or check — every COMPOSED status file in one pass (2026-11-12).

`info.json` (each live deck and open branch), `data/fleet/analytics.json` and
`data/decks/index.json` are composed from everything else, so they go stale when ANY
input moves — a goldfish change (every deck's `older_model` flips), a new deck, an art
pin — and nothing rewrites them unless asked. On 2026-11-12 that cost three 15-minute
gate runs: nine `info.json`, the fleet file and the manifest were behind, and the
freshness tests said so only at the end of the regression tier.

    manamap pilot restamp            # rewrite them all (seconds per deck)
    manamap pilot restamp --check    # report what is behind, write nothing (exit 1)

`check_scope` runs `--check` FIRST, before any tier, so a stale stamp fails in about a
minute with the one command that fixes it. The targets are exactly the freshness
test's (`targets`, which that test imports): live decks and their OPEN branches;
retired decks and decided branches are history and are never rewritten.
"""
import contextlib
import io
import sys

from manamap import config


def targets(artifact="info.json"):
    """Every `(slug, branch)` where `artifact` is tracked and a fresh run must match.

    ONE ENUMERATION, ONE HOME: `tests/test_pilot_artifact_freshness._slugs` reads this.
    """
    from manamap.pilot import deck_branch
    from manamap.pilot.regen import is_retired
    out = []
    if not config.DECKS_DIR.is_dir():
        return out
    for d in sorted(config.DECKS_DIR.iterdir()):
        if not d.is_dir() or is_retired(d.name):
            continue
        if (d / artifact).exists():
            out.append((d.name, None))
        for b in sorted((d / "branches").glob("*")):
            if not (b.is_dir() and (b / artifact).exists()):
                continue
            if deck_branch.is_decided(d.name, b.name):
                continue
            out.append((d.name, b.name))
    return out


def _info_path(slug, branch):
    base = config.DECKS_DIR / slug
    return (base / "branches" / branch if branch else base) / "info.json"


def _rewrite(path, write):
    """Run `write()`, return whether `path` changed. In check mode the old bytes are
    put back whatever happened, so a check never leaves the tree different."""
    before = path.read_bytes() if path.is_file() else None
    try:
        write()
    finally:
        after = path.read_bytes() if path.is_file() else None
    return before, after


def run(check=False, quiet=False):
    """Restamp everything; returns the paths that were (or, checking, would be) rewritten."""
    from manamap.pilot import deck_info, deck_manifest, fleet_analytics
    jobs = []
    for slug, branch in targets("info.json"):
        args = type("Args", (), {"slug": slug, "branch": branch, "write": True})()
        jobs.append((_info_path(slug, branch), lambda a=args: deck_info.main(a)))
    # AFTER every info.json: both read it.
    jobs.append((fleet_analytics.ARTIFACT, lambda: fleet_analytics.write()))
    jobs.append((config.DECKS_DIR / "index.json",
                 lambda: deck_manifest.write_manifest(deck_manifest.gather_entries())))
    changed, restore = [], []
    for path, write in jobs:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            before, after = _rewrite(path, write)
        if before != after:
            changed.append(path)
            if check:
                restore.append((path, before))
    # Restore only at the end: later stamps read earlier ones, and a check must see
    # the whole fresh chain before putting the tree back.
    for path, before in reversed(restore):
        if before is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(before)
    if not quiet:
        rel = [str(p.relative_to(config.DECKS_DIR.parent.parent)) for p in changed]
        if not changed:
            print(f"restamp: all {len(jobs)} composed file(s) current")
        elif check:
            print(f"restamp --check: {len(changed)} of {len(jobs)} composed file(s) behind:\n  "
                  + "\n  ".join(rel)
                  + "\n  fix: `manamap pilot restamp`, then commit them")
        else:
            print(f"restamp: rewrote {len(changed)} of {len(jobs)}:\n  " + "\n  ".join(rel))
    return changed


def main(args=None):
    check = bool(getattr(args, "check", False))
    changed = run(check=check)
    if check and changed:
        sys.exit(1)
