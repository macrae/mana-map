"""Pilot: mechanically enforce the contract on Fetch Quests (tutor_guide.json).

The coach writes scenario → fetch → why for every tutor in the 99. Code
enforces form; the checks that keep a ★ artifact honest:

  * every `card` is a maindeck NONLAND whose oracle text actually searches the
    library — fetch lands belong to Sources Say, not here;
  * every maindeck tutor appears exactly once — a tutor with no entry is a
    hole in the promise ("one wish per tutor");
  * every `fetch` target is a real card in the deck (commander included — a
    reveal can hit it) with at least one scenario and a non-empty why;
  * best-effort type legality: when the tutor's oracle names a target type
    (creature/land/artifact/…), the fetched card's type line must carry it;
  * citations, when present, obey the shared citation contract.
"""

import json
import re

from manamap.pilot.common import (
    deck_dir,
    is_land,
    load_deck_cards,
    mainboard,
    report_errors,
    try_load_rules_db,
)
from manamap.pilot.validate_stack import load_strategy_sections, validate_citations

REQUIRED_TOP_KEYS = {"slug", "assessment", "tutors", "gaps"}
REQUIRED_TARGET_KEYS = {"scenario", "fetch", "why"}
SEARCH_RE = re.compile(r"search your library", re.IGNORECASE)
# A search clause that only finds lands is mana, not a wish — Cultivate and
# Nature's Lore belong to Sources Say. Tutors with broader clauses stay.
_LAND_ONLY_RE = re.compile(
    r"search your library for [^.]*?"
    r"(?:land cards?|plains|island|swamp|mountain|forest|wastes)",
    re.IGNORECASE)

# Target-type words a tutor's oracle can constrain on, checked against the
# fetched card's type line. Deliberately coarse: subtypes (Dinosaur, Aura)
# are checked the same way since Scryfall type lines carry them verbatim.
_TYPE_WORDS = ("creature", "land", "artifact", "enchantment", "instant",
               "sorcery", "planeswalker", "battle")


def deck_tutors(cards):
    """Maindeck nonland cards whose oracle searches the library for nonland
    targets — the cards Fetch Quests owes an entry."""
    names = []
    for c in mainboard(cards):
        text = str(c.get("oracle_text", "") or "")
        if is_land(c) or not SEARCH_RE.search(text):
            continue
        clauses = re.findall(r"search your librar(?:y|ies)[^.]*", text,
                             re.IGNORECASE)
        if clauses and all(_LAND_ONLY_RE.search(cl) for cl in clauses):
            continue  # pure land ramp — Sources Say territory
        names.append(c["name"])
    return sorted(names)


def _tutor_constraint_types(oracle):
    """Per-clause type constraints. A DFC or chapter card can carry several
    search clauses (Huatli's front face fetches a basic land; Roar III fetches
    Dinosaurs) — a fetch is legal if ANY clause permits it, so each clause's
    constraint list is returned separately. A clause with no recognizable
    type words yields [] (unconstrained)."""
    constraints = []
    for match in re.finditer(r"search your librar(?:y|ies) for ([^.]*)",
                             oracle, re.IGNORECASE):
        clause = match.group(1).lower()
        if " card" not in clause + " ":
            constraints.append([])
            continue
        constraints.append([t for t in _TYPE_WORDS if t in clause])
    return constraints


def validate(doc, deck_doc, rules=None, strategy_sections=None):
    errors = []
    missing = REQUIRED_TOP_KEYS - set(doc)
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")
        return errors
    if not str(doc.get("assessment", "")).strip():
        errors.append("assessment is empty — say how this deck tutors")

    cards = deck_doc.get("cards", [])
    by_name = {c["name"]: c for c in cards}
    deck_names = {c["name"] for c in mainboard(cards)}
    expected = deck_tutors(cards)

    entries = doc.get("tutors", [])
    covered = [e.get("card") for e in entries]
    for name in expected:
        if name not in covered:
            errors.append(f"maindeck tutor {name!r} has no entry — the promise "
                          f"is one wish per tutor")
    for i, entry in enumerate(entries):
        name = entry.get("card")
        label = f"tutors[{i}] ({name})"
        if name not in expected:
            errors.append(
                f"{label}: not a maindeck library-search tutor (lands belong "
                f"to Sources Say; the reveal isn't a search)")
            continue
        if covered.count(name) > 1:
            errors.append(f"{label}: duplicate entry")
        targets = entry.get("targets") or []
        if not targets:
            errors.append(f"{label}: no targets — a tutor with nothing to "
                          f"fetch is a hole in the section")
        oracle = str(by_name[name].get("oracle_text", "") or "")
        clause_constraints = _tutor_constraint_types(oracle)
        for j, target in enumerate(targets):
            tlabel = f"{label} target {j}"
            missing_keys = REQUIRED_TARGET_KEYS - set(target)
            if missing_keys:
                errors.append(f"{tlabel}: missing keys {sorted(missing_keys)}")
                continue
            fetch = target["fetch"]
            fetched = by_name.get(fetch)
            if fetch not in deck_names and not (fetched or {}).get("is_commander"):
                errors.append(f"{tlabel}: fetch {fetch!r} is not in the deck")
            elif clause_constraints and fetched is not None:
                type_line = str(fetched.get("type_line", "")).lower()
                legal = any(
                    not constraint
                    or any(t in type_line for t in constraint)
                    for constraint in clause_constraints)
                if not legal:
                    wanted = " or ".join(
                        "/".join(c) for c in clause_constraints if c)
                    errors.append(
                        f"{tlabel}: {name!r} searches for a {wanted} card, "
                        f"but {fetch!r} is {fetched.get('type_line')!r}")
            if not str(target.get("why", "")).strip():
                errors.append(f"{tlabel}: `why` is empty")
            if not str(target.get("scenario", "")).strip():
                errors.append(f"{tlabel}: `scenario` is empty")
            citations = target.get("citations") or []
            if citations:
                validate_citations(citations, rules or {}, tlabel, errors,
                                   strategy_sections)
    return errors


def main(args):
    base = deck_dir(args.slug)
    path = base / "tutor_guide.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — spawn pilot-coach for the Fetch Quests "
            f"artifact first (write-manual skill, tutor-guide routine).")
    with open(path) as f:
        doc = json.load(f)
    deck_doc = load_deck_cards(args.slug)
    errors = validate(doc, deck_doc, try_load_rules_db(),
                      load_strategy_sections())
    n_targets = sum(len(e.get("targets") or []) for e in doc.get("tutors", []))
    report_errors(
        path.name, errors,
        f"OK   {path.name} — {len(doc.get('tutors', []))} tutor(s), "
        f"{n_targets} wish(es); coaching ★")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-tutor-guide <slug>`.")
