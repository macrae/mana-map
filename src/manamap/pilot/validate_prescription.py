"""Pilot: form-check a prescription — the doctor's answer to one question.

A prescription reaches a decklist the same way a diagnosis does, so it is held to
the same contract, by the same code: `validate_diagnosis`'s cut, add, bracket-delta,
axis-movement, open-question, skeptic and citation checks are imported and run over
the `add_candidates` / `cut_candidates` / `open_questions` / `axes_engaged` blocks.
Two diagnoses' worth of findings are encoded in those functions (a cut that orphans a
verified stack, an add whose named axis does not move when it is bought); a second
copy here would drift from the first.

What is different, and why:

- **The prompt is authored and the id is derived from it.** `id` must equal
  `prescription_id(prompt)` — a hand-edited prompt under an old id would make the
  cache's `prompt:self` digest and the filename disagree about which question this
  answers.
- **`add_candidates` is ranked and capped at `MAX_ADDS` (10).** The retired Short List's
  rule, relocated: ten is the section, not a budget, and the eleventh is the one you
  rank harder to leave off.
- **Stale is not wrong.** Prescriptions accumulate. One written against an older
  decklist is checked for FORM only — keys, closed sets, the prompt/id pair, verbatim
  citations — because every deck-membership check ("not in the maindeck") would fail
  on a cut that was since applied, and a gate that reddens history teaches its
  reader to ignore it. The cache routine's MISS is what says "ask again".
- **`log_entries_read`** names the captain's-log ids the doctor leaned on; each must
  exist. The log is how a prescription knows what happened at the table.
"""

from manamap.pilot import deck_audit as audit_mod
from manamap.pilot.common import (
    deck_dir, load_card_roles, load_deck_cards, load_json, load_rules_db,
    report_errors)
from manamap.pilot.deck_notes import read_log
from manamap.pilot.prescribe import DIR, MAX_ADDS, find, prescription_id
from manamap.pilot.validate_diagnosis import (
    ADD_SOURCES, CUT_DIFFICULTIES, REQUIRED_ADD_KEYS, REQUIRED_CUT_KEYS,
    _validate_adds, _validate_all_citations, _validate_axes,
    _validate_bracket_deltas, _validate_cuts, _validate_prescription_moves,
    _validate_questions, _validate_skeptic)
from manamap.pilot.validate_stack import load_strategy_sections

REQUIRED_AUTHORED_KEYS = {"slug", "id", "prompt", "as_of_decklist_sha256"}
REQUIRED_ANSWER_KEYS = {"reading", "cut_candidates", "add_candidates", "open_questions"}


def is_answered(doc):
    return "add_candidates" in doc


def validate(doc, deck_doc, deck_path=None, measured_axes=None, rules=None,
             strategy_sections=None, log_ids=None):
    """Error strings (empty = holds). Form-only when the stamp is not the deck."""
    errors = []
    missing = REQUIRED_AUTHORED_KEYS - set(doc)
    if missing:
        return [f"missing authored keys {sorted(missing)} — create the file with "
                f"`prescribe <slug> \"<question>\"`, never by hand"]
    if doc["id"] != prescription_id(doc["prompt"]):
        errors.append(f"id {doc['id']!r} is not the hash of the prompt "
                      f"({prescription_id(doc['prompt'])}) — the question was edited "
                      f"under an old id; ask it as a new prescription instead")
    if not is_answered(doc):
        return errors                      # an open question is a valid file

    for k in sorted(REQUIRED_ANSWER_KEYS - set(doc)):
        errors.append(f"missing {k!r} — the doctor did not finish")
    if not str(doc.get("reading") or "").strip():
        errors.append("reading is empty — say what the question asks and what the "
                      "evidence says before prescribing")
    adds = doc.get("add_candidates") or []
    if isinstance(adds, list) and len(adds) > MAX_ADDS:
        errors.append(f"add_candidates has {len(adds)} entries; the cap is {MAX_ADDS} — "
                      f"ten is the section, not a budget; rank harder")
    for i, eid in enumerate(doc.get("log_entries_read") or []):
        if log_ids is not None and eid not in log_ids:
            errors.append(f"log_entries_read[{i}]: no log entry {eid!r}")

    stamped, current = doc.get("as_of_decklist_sha256"), deck_doc.get("decklist_sha256")
    stale = bool(stamped and current and stamped != current)
    cards = deck_doc.get("cards", [])
    main_names = {c["name"] for c in cards}
    commander_names = {c["name"] for c in cards if c.get("is_commander")}

    # The diagnosis validator's checks, reused. `axes_engaged` is the optional
    # subset of axes the answer leans on — same shape as a diagnosis's `axes`.
    shim = dict(doc, axes=doc.get("axes_engaged") or [])
    if not stale:
        errors += _validate_axes(shim, measured_axes)
        errors += _validate_cuts(shim, main_names, commander_names, deck_path)
        errors += _validate_adds(shim, main_names, commander_names)
        errors += _validate_bracket_deltas(shim, main_names, commander_names)
        errors += _validate_prescription_moves(shim, deck_doc, _roles())
    else:
        errors += _form_only(shim)
    errors += _validate_questions(shim)
    errors += _validate_skeptic(shim)
    errors += _validate_all_citations(shim, rules or {}, strategy_sections)
    return errors


def _roles():
    """card_roles.json is tracked, so a fresh clone has it; a scratch data dir may
    not, and the moves check already skips when roles are absent — it must skip,
    not crash, the same way it treats a missing corpus."""
    try:
        return load_card_roles()
    except FileNotFoundError:
        return None


def _form_only(doc):
    """Required keys and closed sets, and nothing that asks the current deck a
    question — a stale prescription's cuts may already be applied and its adds
    already sleeved, and that is history, not an error."""
    errors = []
    for i, cut in enumerate(doc.get("cut_candidates") or []):
        missing = REQUIRED_CUT_KEYS - set(cut)
        if missing:
            errors.append(f"cut_candidates[{i}]: missing keys {sorted(missing)}")
        elif cut["difficulty"] not in CUT_DIFFICULTIES:
            errors.append(f"cut_candidates[{i}]: difficulty {cut['difficulty']!r} not in "
                          f"{sorted(CUT_DIFFICULTIES)}")
    for i, add in enumerate(doc.get("add_candidates") or []):
        missing = REQUIRED_ADD_KEYS - set(add)
        if missing:
            errors.append(f"add_candidates[{i}]: missing keys {sorted(missing)}")
        elif add["source"] not in ADD_SOURCES:
            errors.append(f"add_candidates[{i}]: source {add['source']!r} not in "
                          f"{sorted(ADD_SOURCES)}")
    return errors


def main(args):
    slug = args.slug
    base = deck_dir(slug)
    pid = getattr(args, "id", None)
    if pid:
        path = find(slug, pid)
        if path is None:
            raise SystemExit(f"{slug}: no prescription {pid!r} under {DIR}/")
        docs = [(path, load_json(path))]
    else:
        docs = [(p, load_json(p)) for p in sorted((base / DIR).glob("*.json"))] \
            if (base / DIR).is_dir() else []
        if not docs:
            print(f"OK   {slug} — no prescriptions")
            return
    deck_doc = load_deck_cards(slug)
    try:
        rules, _, _ = load_rules_db()
    except (FileNotFoundError, ValueError):
        rules = {}
    sections = load_strategy_sections()
    try:
        measured = {a["axis"]: a for a in audit_mod.analyze(slug)["axes"]}
    except FileNotFoundError:
        measured = None
    log_ids = {e["id"] for e in read_log(slug)}
    current = deck_doc.get("decklist_sha256")

    errors, answered, stale = [], 0, 0
    for path, doc in docs:
        errs = validate(doc, deck_doc, deck_path=base, measured_axes=measured,
                        rules=rules, strategy_sections=sections, log_ids=log_ids)
        errors += [f"{path.name}: {e}" for e in errs]
        answered += is_answered(doc)
        stale += bool(doc.get("as_of_decklist_sha256") != current)
    report_errors(f"{slug} prescriptions", errors,
                  ok_line=f"OK   {slug} — {len(docs)} prescription(s), {answered} answered"
                          f"{f', {stale} stale (older decklist; form-checked only)' if stale else ''}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-prescription <slug> [--id ID]`.")
