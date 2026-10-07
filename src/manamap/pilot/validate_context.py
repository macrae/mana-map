"""Pilot: the gate on `CONTEXT.md` — `deck_context.check`, as a validator.

`deck_status.VALIDATED` runs every gate through `module.main(args)` with only a
slug, and `deck_context.main` with only a slug PRINTS the document. So the gate is
its own module, the same shape as every other `validate_*`.
"""
import sys

from manamap.pilot import deck_context


def main(args=None):
    slug = getattr(args, "slug", None)
    if not deck_context.path(slug).exists():
        print(f"{slug}: no {deck_context.ARTIFACT} — absent means absent")
        return
    errors, warnings = deck_context.check(slug)
    if errors:
        print(f"FAIL {slug} — {deck_context.ARTIFACT} ({len(errors)} error(s)):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"OK   {slug} — {deck_context.ARTIFACT}" + "".join(f"\n  · {w}" for w in warnings))
