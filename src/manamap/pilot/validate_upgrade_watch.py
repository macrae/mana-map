"""Pilot: mechanically enforce the contract on an upgrade-watch scout report.

The upgrade scout is the empty-sideboard counterpart of the sideboard analyst:
its pool is the whole card database, so the constraint that keeps it honest is
not "only these cards" but "only these claims":

  * a lookout card must NOT already be in the deck — a scout that recommends a
    card the pilot runs has read neither list;
  * an obsolescence claim ("upgrades X") must exist in obsolescence_index.json
    under that deck card, naming this replacement;
  * a synergy-partner claim must appear in the candidate's own
    synergy_graph.json shortlist;
  * a combo line is a candidate until a stack artifact passes the checker —
    the same status vocabulary validate_sideboard enforces, imported from it
    rather than duplicated.

Cross-checks degrade gracefully: when a reference artifact is absent (fresh
clone), the corresponding check is skipped, never failed — the same posture
validate_sideboard takes toward bracket recomputation.
"""

import json

from manamap.config import OBSOLESCENCE_INDEX_PATH, SYNERGY_GRAPH_PATH
from manamap.pilot.common import (
    deck_dir,
    load_deck_cards,
    load_json,
    mainboard,
    report_errors,
)
from manamap.pilot.validate_sideboard import (
    UNVERIFIED_STATUS,
    VERIFIED_STATUS,
    _check_verified_line,
)

REQUIRED_TOP_KEYS = {"slug", "assessment", "lookout", "gaps"}
REQUIRED_ENTRY_KEYS = {"card", "why"}
MAX_LOOKOUT = 10


def _validate_entries(doc, main_names, deck_path):
    errors = []
    lookout = doc.get("lookout", [])
    if not isinstance(lookout, list) or not lookout:
        return ["lookout must be a non-empty list (1-10 entries)"]
    if len(lookout) > MAX_LOOKOUT:
        errors.append(f"lookout has {len(lookout)} entries — the contract is a "
                      f"top {MAX_LOOKOUT}, not a catalogue")
    seen = {}
    for i, entry in enumerate(lookout):
        label = f"lookout {i} ({entry.get('card')})"
        missing = REQUIRED_ENTRY_KEYS - set(entry)
        if missing:
            errors.append(f"{label}: missing keys {sorted(missing)}")
            continue
        name = entry["card"]
        if name in main_names:
            errors.append(f"{label}: already in the deck — a scout that "
                          f"recommends a card the pilot runs has read neither list")
        if name in seen:
            errors.append(f"{label}: duplicate of lookout {seen[name]}")
        else:
            seen[name] = i
        if not str(entry.get("why", "")).strip():
            errors.append(f"{label}: `why` is empty — a pick without a reason is a diff")
        errors += _validate_entry_lines(i, entry, deck_path)
    return errors


def _validate_entry_lines(i, entry, deck_path):
    """Combo-line claims stay candidates until a stack artifact passes."""
    errors = []
    for j, line in enumerate(entry.get("combo_lines_opened") or
                             entry.get("evidence", {}).get("combo_lines_opened") or []):
        status = line.get("status")
        if status == VERIFIED_STATUS:
            errors += [e.replace("opens_lines", f"lookout {i} line {j}")
                       for e in _check_verified_line(j, line, deck_path)]
        elif status != UNVERIFIED_STATUS:
            errors.append(
                f"lookout {i} ({entry.get('card')}) line {j}: status must be "
                f"{UNVERIFIED_STATUS!r} (or {VERIFIED_STATUS!r} with a "
                f"checker-passed `stack_artifact`), got {status!r}"
            )
    return errors


def _validate_obsolescence(doc):
    """`upgrades`/`obsoletes` claims re-checked against the index, never trusted."""
    index = load_json(OBSOLESCENCE_INDEX_PATH, default=None)
    if index is None:
        return []  # reference artifact absent: skip, do not fail
    errors = []
    for i, entry in enumerate(doc.get("lookout", [])):
        claims = (entry.get("obsoletes") or
                  entry.get("evidence", {}).get("obsoletes") or [])
        for deck_card in claims:
            reps = {r["name"] for r in (index.get(deck_card) or {}).get("obsoleted_by", [])}
            if entry.get("card") not in reps:
                errors.append(
                    f"lookout {i} ({entry.get('card')}): claims to obsolete "
                    f"'{deck_card}', but obsolescence_index.json lists no such "
                    f"replacement — the index is the authority on strictly-better"
                )
    return errors


def _validate_synergy(doc, main_names):
    """Partner claims re-checked against the candidate's own shortlist."""
    graph = load_json(SYNERGY_GRAPH_PATH, default=None)
    if graph is None:
        return []  # reference artifact absent: skip, do not fail
    errors = []
    for i, entry in enumerate(doc.get("lookout", [])):
        claims = (entry.get("synergy_partners_in_deck") or
                  entry.get("evidence", {}).get("synergy_partners_in_deck") or [])
        shortlist = {e.get("partner") for e in graph.get(entry.get("card"), [])}
        for claim in claims:
            partner = claim.get("partner") if isinstance(claim, dict) else claim
            if partner not in shortlist:
                errors.append(
                    f"lookout {i} ({entry.get('card')}): claims synergy partner "
                    f"'{partner}' but the synergy graph's shortlist for this card "
                    f"does not include it"
                )
            elif partner not in main_names:
                errors.append(
                    f"lookout {i} ({entry.get('card')}): synergy partner "
                    f"'{partner}' is not in this deck"
                )
    return errors


def validate(doc, deck_doc, deck_path=None):
    """Return a list of error strings (empty = the contract holds)."""
    errors = []
    missing = REQUIRED_TOP_KEYS - set(doc)
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")
        return errors
    if not str(doc.get("assessment", "")).strip():
        errors.append("assessment is empty — say what the pool offers this deck")

    main_names = {c["name"] for c in mainboard(deck_doc.get("cards", []))}
    errors += _validate_entries(doc, main_names, deck_path)
    errors += _validate_obsolescence(doc)
    errors += _validate_synergy(doc, main_names)
    return errors


def main(args):
    base = deck_dir(args.slug)
    path = base / "upgrade_watch.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run the analyse-sideboard skill (empty-sideboard "
            f"branch) for {args.slug} first."
        )
    with open(path) as f:
        doc = json.load(f)
    deck_doc = load_deck_cards(args.slug)
    errors = validate(doc, deck_doc, deck_path=base)
    report_errors(
        path.name, errors,
        f"OK   {path.name} — {len(doc.get('lookout', []))} lookout card(s); "
        f"evidence ◆, picks ★")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-upgrade-watch <slug>`.")
