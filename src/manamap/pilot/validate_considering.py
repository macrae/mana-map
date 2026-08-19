"""Pilot: mechanically enforce the contract on The Short List (considering.json).

LEGACY (2026-08-19): the magazine renderer. It still renders the nine frozen issues from
artifacts nothing regenerates any more (issue_plan.json, the panel keys,
card_roles/mana_base/upgrades, considering.json), and it is replaced by the compact deck
page in docs/manual-v5-spec.md. Do not extend it; internals below are accurate for what it
does.

Ten cards worth knowing about that could play well with this deck. One artifact,
one question — and deliberately NOT "do you own it". The list used to carry
`source: "sideboard" | "pool"` and rank bench cards first, which made ownership
a selection rule; a card is now on the list because it is worth knowing about or
it is not on the list. Whether it is already in a box is the reader's business.

  * exactly TEN entries, no duplicates — ten is the section, not a budget;
  * a pick is NOT already in the deck — the ten are cards to consider, not cards
    you already run;
  * combo-line claims stay candidates until a stack artifact passes — the status
    vocabulary lives in `common`, shared with the diagnosis validator;
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
from manamap.pilot.common import (
    UNVERIFIED_STATUS,
    VERIFIED_STATUS,
    check_verified_line,
    deck_dir,
    load_deck_cards,
    load_json_memo,
    report_errors,
)

REQUIRED_TOP_KEYS = {"slug", "assessment", "ten", "gaps"}
REQUIRED_ENTRY_KEYS = {"card", "why"}
SHORT_LIST_SIZE = 10


def _validate_entries(doc, main_names, commander_names, deck_path):
    errors = []
    ten = doc.get("ten", [])
    if not isinstance(ten, list):
        return ["`ten` must be a list"]
    if len(ten) != SHORT_LIST_SIZE:
        errors.append(
            f"`ten` has {len(ten)} entries — The Short List is exactly "
            f"{SHORT_LIST_SIZE} — ten is the section, not a budget")
    seen, cuts = {}, {}
    for i, entry in enumerate(ten):
        label = f"ten[{i}] ({entry.get('card')})"
        missing = REQUIRED_ENTRY_KEYS - set(entry)
        if missing:
            errors.append(f"{label}: missing keys {sorted(missing)}")
            continue
        name = entry["card"]
        if name in main_names:
            errors.append(f"{label}: already in the deck — the ten are cards to "
                          f"consider, not cards you already run")
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
                       for e in check_verified_line(j, line, deck_path)]
        elif status != UNVERIFIED_STATUS:
            errors.append(
                f"ten[{i}] ({entry.get('card')}) line {j}: status must be "
                f"{UNVERIFIED_STATUS!r} (or {VERIFIED_STATUS!r} with a "
                f"checker-passed `stack_artifact`), got {status!r}")
    return errors


def _big_graph(path):
    """A large SHARED graph, parsed once per process.

    `load_json` is documented "for small per-deck artifacts" and does not memoize.
    These two are the format-wide graphs — obsolescence at 3 MB and synergy at
    27.8 MB — and every validate() call re-parsed both. The suite noticed before a
    human did: nine tests validating ten synthetic card names cost 7.8 seconds,
    almost all of it re-reading 31 MB of JSON that never changes within a run.
    """
    try:
        return load_json_memo(path)
    except FileNotFoundError:
        return None


def _validate_obsolescence(doc):
    index = _big_graph(OBSOLESCENCE_INDEX_PATH)
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
    graph = _big_graph(SYNERGY_GRAPH_PATH)
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
    main_names = {c["name"] for c in cards}
    commander_names = {c["name"] for c in cards if c.get("is_commander")}
    deck_names = main_names | commander_names

    errors += _validate_entries(doc, main_names, commander_names,
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
            f"{path} not found — run the short-list skill (The Short "
            f"List) for {args.slug} first.")
    with open(path) as f:
        doc = json.load(f)
    deck_doc = load_deck_cards(args.slug)
    errors = validate(doc, deck_doc, deck_path=base)
    report_errors(
        path.name, errors,
        f"OK   {path.name} — the ten holds; evidence ◆, verdicts ★")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-considering <slug>`.")
