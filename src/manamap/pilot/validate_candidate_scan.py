"""Pilot: form-check `candidate_scan.json` — the dated corpus scan a staging decision cites.

The gate in the same commit as the artifact. What it holds the scan to: every candidate
is a real, Commander-legal card inside the deck's colour identity; no candidate is a
Game Changer (the scan drops them — the deck is held at bracket 3); every `infinite_with`
partner is a real card that forms a two-card infinite with the candidate in
`combo_details`; `as_of` is a date; the dimensions are the known set. What it does NOT
fail on, and why (the `validate_recon` rule — a check that fires on correct data is worse
than no check): a candidate that has since joined the 99, or a `decklist_sha256` that no
longer matches, because the scan is DATED and the decklist moves under it. Both are WARN
lines, so a reader knows the scan predates the list.
"""
import sys
from datetime import date

from manamap.pilot.card_pool import load_pool
from manamap.pilot.card_search import commander_identity, deck_names
from manamap.pilot.common import (
    deck_dir,
    decklist_sha256,
    expand_faces,
    load_combo_details,
    load_json,
    report_errors,
)

ARTIFACT = "candidate_scan.json"
REQUIRED = ("slug", "as_of", "decklist_sha256", "identity", "dimensions", "excluded", "limits")


def validate(slug, doc):
    from manamap.pilot.candidate_scan import DIMENSIONS
    errors, warns = [], []
    for k in REQUIRED:
        if k not in doc:
            errors.append(f"missing required key {k!r}")
    if errors:
        return errors, warns
    if doc["slug"] != slug:
        errors.append(f"slug is {doc['slug']!r} but the artifact lives in {slug}/")
    try:
        date.fromisoformat(str(doc["as_of"]))
    except (TypeError, ValueError):
        errors.append(f"as_of {doc['as_of']!r} is not an ISO date — the scan is dated evidence")
    pool = load_pool()
    if pool is None:
        return ["cards.csv is absent — the gate cannot resolve names"], warns
    ident = set(doc.get("identity") or [])
    try:
        deck_ident = commander_identity(slug)
        if ident != deck_ident:
            errors.append(f"identity {sorted(ident)} is not the commander's {sorted(deck_ident)}")
    except FileNotFoundError:
        warns.append("no cards.json to check identity against")
    present = set()
    try:
        present = deck_names(slug)
    except FileNotFoundError:
        pass
    details = load_combo_details()
    combos = details["combos"]
    for dim, block in (doc.get("dimensions") or {}).items():
        if dim not in DIMENSIONS:
            errors.append(f"dimensions.{dim}: not a known dimension {list(DIMENSIONS)}")
            continue
        rows = block.get("candidates")
        if not isinstance(rows, list):
            errors.append(f"dimensions.{dim}: no candidates list")
            continue
        seen_flagged = False
        for i, r in enumerate(rows):
            where = f"dimensions.{dim}[{i}]"
            name = r.get("name")
            rec = pool.get(name)
            if rec is None:
                errors.append(f"{where}: {name!r} is not in the corpus")
                continue
            if not rec["legal"]:
                errors.append(f"{where}: {name} is not Commander-legal")
            if not rec["color_identity"] <= ident:
                errors.append(f"{where}: {name} is outside the identity {sorted(ident)}")
            if rec["game_changer"]:
                errors.append(f"{where}: {name} is a Game Changer — the scan drops them (bracket 3)")
            if present and expand_faces(name) & present:
                warns.append(f"{where}: {name} is now in the 99 — the scan predates the list")
            inf = (r.get("combos") or {}).get("infinite_with") or []
            if inf:
                seen_flagged = True
            elif seen_flagged:
                errors.append(f"{where}: {name} is unflagged but sorts after a flagged row — "
                              f"infinite_with rows sort LAST")
            for other in inf:
                if other not in pool:
                    errors.append(f"{where}: infinite_with names {other!r}, not in the corpus")
                    continue
                ok = any(len(combos[j]["cards"]) == 2 and other in combos[j]["cards"]
                         and any(str(p).lower().startswith("infinite") for p in combos[j].get("produces", []))
                         for j in details["by_card"].get(name, []))
                if not ok:
                    errors.append(f"{where}: {name} + {other} is not a two-card infinite in combo_details")
    try:
        live = decklist_sha256(slug, doc.get("against_branch"))
        if live != doc.get("decklist_sha256"):
            warns.append("decklist_sha256 no longer matches the list — the scan predates it; re-run "
                         "`scan-candidates --write` before citing a row")
    except FileNotFoundError:
        pass
    return errors, warns


def main(args=None):
    slug = getattr(args, "slug", None)
    path = deck_dir(slug) / ARTIFACT
    if not path.is_file():
        print(f"{slug}: no {ARTIFACT} — nothing scanned yet (absent means absent)")
        return
    doc = load_json(path) or {}
    errors, warns = validate(slug, doc)
    for w in warns:
        print(f"WARN {slug} — {ARTIFACT}: {w}")
    report_errors(f"{slug} — {ARTIFACT}", errors)
    if not errors:
        dims = doc.get("dimensions") or {}
        print(f"OK   {slug} — {ARTIFACT} as_of {doc.get('as_of')}: "
              + ", ".join(f"{d} {len(b.get('candidates') or [])}/{b.get('matched')}" for d, b in dims.items()))
    sys.exit(1 if errors else 0)
