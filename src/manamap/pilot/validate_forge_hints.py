"""Pilot: form-check `forge_hints.json`, a deck's per-card hints to Forge's AI.

The gate in the same commit as the artifact. A hint is a piloting decision written onto
a card script — `AILogic$ AristocratCounters` on a sacrifice ability, `SVar:AIPreference:
SacCost$…` for what to feed it — and one that names a card the deck does not run, or
hints nothing, or cannot be placed on exactly one line, is refused here and by
`forge-install --generate` alike (`sim/forge_pilot.validate_hints`, `generate_hints`).
"""
import sys

from manamap.pilot.common import deck_dir, load_json, report_errors
from manamap.sim import forge_pilot


def validate(slug, doc):
    return forge_pilot.validate_hints(slug, doc)


def main(args=None):
    slug = getattr(args, "slug", None)
    path = deck_dir(slug) / forge_pilot.HINTS_FILE
    if not path.is_file():
        print(f"{slug}: no {forge_pilot.HINTS_FILE} — Forge plays the deck's cards with its own "
              f"logic (absent means absent; this is the normal state)")
        return
    doc = load_json(path) or {}
    errors = validate(slug, doc)
    report_errors(f"{slug} — {forge_pilot.HINTS_FILE}", errors)
    if not errors:
        hints = doc.get("hints") or []
        print(f"OK   {slug} — {len(hints)} hint(s): "
              + ", ".join(f"{h['card']}" + (f" [{h['ai_logic']}]" if h.get("ai_logic") else "")
                          for h in hints))
    sys.exit(1 if errors else 0)
