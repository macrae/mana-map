"""Pilot: mechanically enforce the contract on The Short List (considering.json).

One artifact replaces the sideboard-analysis / upgrade-watch pair: the ten
cards most worth the pilot's sleeves, bench-first, pool-filled. The checks are
the union of both retired validators' contracts:

  * exactly TEN entries, no duplicates — ten is the section, not a budget;
  * `source: "sideboard"` → the card is really on this deck's bench (and not a
    table accessory); `source: "pool"` → the card is NOT in the deck at all;
  * a bench larger than ten must have every non-listed card either covered by
    a `bench_verdicts` line or implicitly cut — the assessment carries the cut
    rationale (editorial, not mechanical);
  * combo-line claims stay candidates until a stack artifact passes — the
    status vocabulary is imported from validate_sideboard, not duplicated;
  * obsolescence / synergy-partner claims are re-checked against the indexes;
  * claimed bracket deltas are recomputed via bracket.assess();
  * `natural_cut` names a real maindeck card (never the commander), and no two
    entries claim the same cut.

Cross-checks degrade gracefully when a reference artifact is absent (fresh
clone): skipped, never failed.
"""

import json

from manamap.config import OBSOLESCENCE_INDEX_PATH, SYNERGY_GRAPH_PATH
from manamap.pilot import bracket as bracket_mod
from manamap.pilot.artist_credits import is_accessory
from manamap.pilot.common import (
    deck_dir,
    load_deck_cards,
    load_json,
    mainboard,
    report_errors,
    sideboard,
)
from manamap.pilot.validate_sideboard import (
    UNVERIFIED_STATUS,
    VERIFIED_STATUS,
    _check_verified_line,
)

REQUIRED_TOP_KEYS = {"slug", "assessment", "ten", "gaps"}
REQUIRED_ENTRY_KEYS = {"card", "source", "why"}
SHORT_LIST_SIZE = 10
SOURCES = {"sideboard", "pool"}


def _validate_entries(doc, main_names, bench_names, commander_names, deck_path):
    errors = []
    ten = doc.get("ten", [])
    if not isinstance(ten, list):
        return ["`ten` must be a list"]
    if len(ten) != SHORT_LIST_SIZE:
        errors.append(
            f"`ten` has {len(ten)} entries — The Short List is exactly "
            f"{SHORT_LIST_SIZE}: prune a big bench to its best ten, top a small "
            f"one up from the pool")
    seen, cuts = {}, {}
    for i, entry in enumerate(ten):
        label = f"ten[{i}] ({entry.get('card')})"
        missing = REQUIRED_ENTRY_KEYS - set(entry)
        if missing:
            errors.append(f"{label}: missing keys {sorted(missing)}")
            continue
        name, source = entry["card"], entry["source"]
        if source not in SOURCES:
            errors.append(f"{label}: source must be one of {sorted(SOURCES)}, "
                          f"got {source!r}")
        elif source == "sideboard" and name not in bench_names:
            errors.append(f"{label}: claims to be in the sideboard, but the "
                          f"bench holds no such card (accessories don't count)")
        elif source == "pool" and (name in main_names or name in bench_names):
            errors.append(f"{label}: claims to be a pool scout, but the card "
                          f"is already in the deck")
        if name in seen:
            errors.append(f"{label}: duplicate of ten[{seen[name]}]")
        else:
            seen[name] = i
        if not str(entry.get("why", "")).strip():
            errors.append(f"{label}: `why` is empty — a pick without a reason "
                          f"is a diff")
        cut = entry.get("natural_cut")
        if cut:
            if cut not in main_names:
                errors.append(f"{label}: natural_cut {cut!r} is not in the "
                              f"maindeck")
            elif cut in commander_names:
                errors.append(f"{label}: natural_cut may not be the commander")
            if cut in cuts:
                errors.append(f"{label}: natural_cut {cut!r} already claimed "
                              f"by ten[{cuts[cut]}]")
            else:
                cuts[cut] = i
        errors += _validate_entry_lines(i, entry, deck_path)
    return errors


def _validate_entry_lines(i, entry, deck_path):
    errors = []
    for j, line in enumerate((entry.get("evidence") or {})
                             .get("combo_lines_opened") or []):
        status = line.get("status")
        if status == VERIFIED_STATUS:
            errors += [e.replace("opens_lines", f"ten[{i}] line {j}")
                       for e in _check_verified_line(j, line, deck_path)]
        elif status != UNVERIFIED_STATUS:
            errors.append(
                f"ten[{i}] ({entry.get('card')}) line {j}: status must be "
                f"{UNVERIFIED_STATUS!r} (or {VERIFIED_STATUS!r} with a "
                f"checker-passed `stack_artifact`), got {status!r}")
    return errors


def _validate_obsolescence(doc):
    index = load_json(OBSOLESCENCE_INDEX_PATH, default=None)
    if index is None:
        return []
    errors = []
    for i, entry in enumerate(doc.get("ten", [])):
        for deck_card in (entry.get("evidence") or {}).get("obsoletes") or []:
            reps = {r["name"]
                    for r in (index.get(deck_card) or {}).get("obsoleted_by", [])}
            if entry.get("card") not in reps:
                errors.append(
                    f"ten[{i}] ({entry.get('card')}): claims to obsolete "
                    f"{deck_card!r}, but obsolescence_index.json lists no such "
                    f"replacement — the index is the authority on strictly-better")
    return errors


def _validate_synergy(doc, deck_names):
    graph = load_json(SYNERGY_GRAPH_PATH, default=None)
    if graph is None:
        return []
    errors = []
    for i, entry in enumerate(doc.get("ten", [])):
        shortlist = {e.get("partner") for e in graph.get(entry.get("card"), [])}
        for claim in (entry.get("evidence") or {}).get(
                "synergy_partners_in_deck") or []:
            partner = claim.get("partner") if isinstance(claim, dict) else claim
            if partner not in shortlist:
                errors.append(
                    f"ten[{i}] ({entry.get('card')}): claims synergy partner "
                    f"{partner!r} but the synergy graph's shortlist for this "
                    f"card does not include it")
            elif partner not in deck_names:
                errors.append(
                    f"ten[{i}] ({entry.get('card')}): synergy partner "
                    f"{partner!r} is not in this deck")
    return errors


def _validate_bracket_deltas(doc, main_names, commander_names):
    """Recompute every claimed delta. Trusting one would be laundering ◆."""
    claimed = [(i, e) for i, e in enumerate(doc.get("ten", []))
               if isinstance(e.get("bracket_delta"), dict)]
    if not claimed:
        return []
    try:
        card_flags, roles, details = bracket_mod.load_reference()
    except (FileNotFoundError, SystemExit):
        return []  # fresh clone: skip, never fail
    errors = []
    base = bracket_mod.assess(sorted(main_names), card_flags, roles, details,
                              sorted(commander_names))
    for i, entry in claimed:
        after_names = set(main_names) | {entry["card"]}
        cut = entry.get("natural_cut")
        if cut:
            after_names.discard(cut)
        after = bracket_mod.assess(sorted(after_names), card_flags, roles,
                                   details, sorted(commander_names))
        got = entry["bracket_delta"]
        for key, value in (("before", base["floor"]), ("after", after["floor"])):
            if key in got and got[key] != value:
                errors.append(
                    f"ten[{i}] ({entry['card']}): bracket_delta.{key} = "
                    f"{got[key]!r}, recomputed {value!r}")
    return errors


def validate(doc, deck_doc, deck_path=None):
    """Return a list of error strings (empty = the contract holds)."""
    errors = []
    missing = REQUIRED_TOP_KEYS - set(doc)
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")
        return errors
    if not str(doc.get("assessment", "")).strip():
        errors.append("assessment is empty — say what these ten do for the deck")

    cards = deck_doc.get("cards", [])
    main_names = {c["name"] for c in mainboard(cards)}
    bench_names = {c["name"] for c in sideboard(cards) if not is_accessory(c)}
    commander_names = {c["name"] for c in cards if c.get("is_commander")}
    deck_names = main_names | commander_names

    errors += _validate_entries(doc, main_names, bench_names, commander_names,
                                deck_path)
    errors += _validate_obsolescence(doc)
    errors += _validate_synergy(doc, deck_names)
    errors += _validate_bracket_deltas(doc, main_names, commander_names)
    return errors


def main(args):
    base = deck_dir(args.slug)
    path = base / "considering.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run the analyse-sideboard skill (The Short "
            f"List) for {args.slug} first.")
    with open(path) as f:
        doc = json.load(f)
    deck_doc = load_deck_cards(args.slug)
    errors = validate(doc, deck_doc, deck_path=base)
    bench = sum(1 for e in doc.get("ten", []) if e.get("source") == "sideboard")
    report_errors(
        path.name, errors,
        f"OK   {path.name} — the ten holds ({bench} from the bench, "
        f"{len(doc.get('ten', [])) - bench} scouted); evidence ◆, verdicts ★")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-considering <slug>`.")
