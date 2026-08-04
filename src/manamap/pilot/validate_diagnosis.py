"""Pilot: mechanically enforce the contract on a deck diagnosis (diagnosis.json).

The diagnosis is the deck-doctor's artifact: an axis-by-axis reading of what a
finished deck is good at and what actually limits it, a ranked add list, a ranked
cut list whose painful entries are argued rather than hidden, and a queue of open
questions routed back into the other skills.

Everything here follows the house rule — **recompute, never trust**:

  * every `axes[].measured.value` is re-derived from `deck-audit` and compared.
    A diagnosis that quotes a figure it did not measure is laundering ◆ into ★.
  * every citation goes through the shared verbatim gate (`validate_citations`),
    so a `strategy:<id>` quote that has drifted out of `strategy.md` fails here
    rather than in a reviewer's head;
  * `cut_candidates[].orphans_stack` is **computed**, not read. If cutting the
    card would strand a checker-passed stack and the doc says otherwise, that is
    an error. Nothing else in the repo performs this check, and it is the
    Ophiomancer / South Wind Avatar class of finding made mechanical: a cut list
    that quietly proposes the one card a verified line rests on.
  * `add_candidates[].bracket_delta` is recomputed via `bracket.assess()`, the
    same treatment `validate_considering` gives The Short List;
  * combo-line claims are restricted to the shared status vocabulary;
  * `skeptic.verdict == "pass"` requires every finding to be `supported`.

No L10 lint, deliberately: the diagnosis is a working artifact and is never
rendered into an issue. L10 exists so a magazine reads as the reader's first;
applying it to a candid weakness audit would forbid the audit from describing
what it is for.

Cross-checks degrade gracefully when a reference artifact is absent (fresh
clone): skipped, never failed.
"""

import json

from manamap.config import DECK_AXIS_TARGETS
from manamap.pilot import bracket as bracket_mod
from manamap.pilot import deck_audit as audit_mod
from manamap.pilot.common import (
    checker_passed,
    deck_dir,
    load_deck_cards,
    load_json,
    load_rules_db,
    mainboard,
    report_errors,
    sideboard,
)
from manamap.pilot.validate_sideboard import UNVERIFIED_STATUS, VERIFIED_STATUS
from manamap.pilot.validate_stack import load_strategy_sections, validate_citations

REQUIRED_TOP_KEYS = {"slug", "verdict", "axes", "engine", "lean_into",
                     "cut_candidates", "add_candidates", "open_questions", "gaps"}
REQUIRED_AXIS_KEYS = {"axis", "verdict", "measured", "reading"}
REQUIRED_CUT_KEYS = {"card", "why", "cost_of_cutting", "difficulty"}
REQUIRED_ADD_KEYS = {"card", "closes", "source", "why"}
REQUIRED_QUESTION_KEYS = {"question", "settled_by", "why_it_matters"}

AXIS_VERDICTS = {"strength", "adequate", "weakness", "liability"}
CUT_DIFFICULTIES = {"easy", "contested", "painful"}
ADD_SOURCES = {"pool", "sideboard", "recon"}
SETTLED_BY = {"resolve-stack", "research-strategy", "goldfish"}
SKEPTIC_STATUSES = {"supported", "unjustified", "miscounted", "mis-cited",
                    "over-claimed", "unverified-line", "contradicts-artifact"}


def _citations_of(obj):
    return obj.get("citations") or []


# ── Axes ─────────────────────────────────────────────────────────────────

def _validate_axes(doc, measured_axes):
    """Every claimed measurement must equal what deck-audit computes today."""
    errors = []
    axes = doc.get("axes")
    if not isinstance(axes, list):
        return ["`axes` must be a list"]
    seen = {}
    for i, axis in enumerate(axes):
        label = f"axes[{i}] ({axis.get('axis')})"
        missing = REQUIRED_AXIS_KEYS - set(axis)
        if missing:
            errors.append(f"{label}: missing keys {sorted(missing)}")
            continue
        name = axis["axis"]
        if name in seen:
            errors.append(f"{label}: duplicate of axes[{seen[name]}]")
        else:
            seen[name] = i
        if axis["verdict"] not in AXIS_VERDICTS:
            errors.append(f"{label}: verdict must be one of "
                          f"{sorted(AXIS_VERDICTS)}, got {axis['verdict']!r}")
        if not str(axis.get("reading", "")).strip():
            errors.append(f"{label}: `reading` is empty — an axis without a "
                          f"reading is a number the reader already had")
        if measured_axes is None:
            continue
        if name not in measured_axes:
            errors.append(
                f"{label}: deck-audit measures no axis by that name "
                f"(it measures {', '.join(sorted(measured_axes))})")
            continue
        claimed = (axis.get("measured") or {}).get("value")
        real = measured_axes[name]["measured"]["value"]
        if claimed is not None and claimed != real:
            errors.append(
                f"{label}: measured.value = {claimed!r}, but deck-audit computes "
                f"{real!r} — the diagnosis must carry the audit's figure, not "
                f"its own")
    return errors


# ── Cuts ─────────────────────────────────────────────────────────────────

def stacks_naming(deck_path, name):
    """Checker-passed stack ids whose scenario names this card.

    Read from the scenario block only. A resolution or a checker note may
    *discuss* a card the board never held, and a discussion is not a dependency.
    """
    hits = []
    for path in sorted((deck_path / "stacks").glob("*.json")):
        try:
            doc = load_json(path, default=None)
        except json.JSONDecodeError:
            continue
        if not doc or not checker_passed(doc):
            continue
        blob = json.dumps(doc.get("scenario", {}), ensure_ascii=False)
        if name in blob:
            hits.append(path.stem.split("-", 1)[0])
    return hits


def _validate_cuts(doc, main_names, commander_names, deck_path):
    errors = []
    cuts = doc.get("cut_candidates")
    if not isinstance(cuts, list):
        return ["`cut_candidates` must be a list"]
    seen = {}
    for i, cut in enumerate(cuts):
        label = f"cut_candidates[{i}] ({cut.get('card')})"
        missing = REQUIRED_CUT_KEYS - set(cut)
        if missing:
            errors.append(f"{label}: missing keys {sorted(missing)}")
            continue
        name = cut["card"]
        if name not in main_names:
            errors.append(f"{label}: not in the maindeck — a cut list may only "
                          f"name cards the deck actually runs")
        if name in commander_names:
            errors.append(f"{label}: the commander cannot be cut")
        if name in seen:
            errors.append(f"{label}: duplicate of cut_candidates[{seen[name]}]")
        else:
            seen[name] = i
        if cut["difficulty"] not in CUT_DIFFICULTIES:
            errors.append(f"{label}: difficulty must be one of "
                          f"{sorted(CUT_DIFFICULTIES)}, got {cut['difficulty']!r}")
        if not str(cut.get("cost_of_cutting", "")).strip():
            errors.append(f"{label}: `cost_of_cutting` is empty — every cut costs "
                          f"something, and naming it is the job")
        if deck_path is None:
            continue
        real = stacks_naming(deck_path, name)
        claimed = cut.get("orphans_stack")
        claimed = [] if claimed in (None, False) else (
            claimed if isinstance(claimed, list) else [claimed])
        if real and sorted(claimed) != sorted(real):
            errors.append(
                f"{label}: cutting this card touches checker-passed stack(s) "
                f"{', '.join(real)}, but orphans_stack says {cut.get('orphans_stack')!r}. "
                f"A verified line resting on a card the cut list proposes must be "
                f"priced, not omitted")
        elif claimed and not real:
            errors.append(
                f"{label}: orphans_stack claims {claimed}, but no checker-passed "
                f"stack's scenario names this card")
    return errors


# ── Adds ─────────────────────────────────────────────────────────────────

def _validate_adds(doc, main_names, bench_names, commander_names):
    errors = []
    adds = doc.get("add_candidates")
    if not isinstance(adds, list):
        return ["`add_candidates` must be a list"]
    seen, claimed_cuts = {}, {}
    for i, add in enumerate(adds):
        label = f"add_candidates[{i}] ({add.get('card')})"
        missing = REQUIRED_ADD_KEYS - set(add)
        if missing:
            errors.append(f"{label}: missing keys {sorted(missing)}")
            continue
        name = add["card"]
        if name in main_names:
            errors.append(f"{label}: already in the maindeck")
        if name in seen:
            errors.append(f"{label}: duplicate of add_candidates[{seen[name]}]")
        else:
            seen[name] = i
        if add["source"] not in ADD_SOURCES:
            errors.append(f"{label}: source must be one of {sorted(ADD_SOURCES)}, "
                          f"got {add['source']!r}")
        elif add["source"] == "sideboard" and name not in bench_names:
            errors.append(f"{label}: claims to come from the bench, but the "
                          f"sideboard holds no such card")
        if not str(add.get("closes", "")).strip():
            errors.append(f"{label}: `closes` is empty — an add that closes "
                          f"nothing named is a preference, not a prescription")
        cut = add.get("natural_cut")
        if cut:
            if cut not in main_names:
                errors.append(f"{label}: natural_cut {cut!r} is not in the maindeck")
            elif cut in commander_names:
                errors.append(f"{label}: natural_cut may not be the commander")
            if cut in claimed_cuts:
                errors.append(f"{label}: natural_cut {cut!r} already claimed by "
                              f"add_candidates[{claimed_cuts[cut]}]")
            else:
                claimed_cuts[cut] = i
        for j, line in enumerate(add.get("combo_lines_opened") or []):
            status = line.get("status")
            if status not in (UNVERIFIED_STATUS, VERIFIED_STATUS):
                errors.append(
                    f"{label} line {j}: status must be {UNVERIFIED_STATUS!r} or "
                    f"{VERIFIED_STATUS!r}, got {status!r}")
    return errors


def _validate_bracket_deltas(doc, main_names, commander_names):
    """Recompute every claimed delta. Trusting one would be laundering ◆."""
    claimed = [(i, a) for i, a in enumerate(doc.get("add_candidates", []))
               if isinstance(a, dict) and isinstance(a.get("bracket_delta"), dict)]
    if not claimed:
        return []
    try:
        card_flags, roles, details = bracket_mod.load_reference()
    except (FileNotFoundError, SystemExit):
        return []  # fresh clone: skip, never fail
    errors = []
    base = bracket_mod.assess(sorted(main_names), card_flags, roles, details,
                              sorted(commander_names))
    for i, add in claimed:
        after = set(main_names) | {add["card"]}
        if add.get("natural_cut"):
            after.discard(add["natural_cut"])
        got = add["bracket_delta"]
        recomputed = bracket_mod.assess(sorted(after), card_flags, roles, details,
                                        sorted(commander_names))
        for key, value in (("before", base["floor"]), ("after", recomputed["floor"])):
            if key in got and got[key] != value:
                errors.append(
                    f"add_candidates[{i}] ({add['card']}): bracket_delta.{key} = "
                    f"{got[key]!r}, recomputed {value!r}")
    return errors


# ── Open questions and the skeptic ───────────────────────────────────────

def _validate_questions(doc):
    errors = []
    questions = doc.get("open_questions")
    if not isinstance(questions, list):
        return ["`open_questions` must be a list"]
    for i, q in enumerate(questions):
        label = f"open_questions[{i}]"
        missing = REQUIRED_QUESTION_KEYS - set(q)
        if missing:
            errors.append(f"{label}: missing keys {sorted(missing)}")
            continue
        if q["settled_by"] not in SETTLED_BY:
            errors.append(
                f"{label}: settled_by must be one of {sorted(SETTLED_BY)}, got "
                f"{q['settled_by']!r} — the value routes the question to a skill, "
                f"so an unroutable one is a question nobody will answer")
    return errors


def _validate_skeptic(doc):
    """`pass` is only available when every finding is `supported`."""
    block = doc.get("skeptic")
    if block is None:
        return []
    if not isinstance(block, dict):
        return ["`skeptic` must be an object"]
    errors = []
    verdict = block.get("verdict")
    if verdict not in {"pass", "fail"}:
        errors.append(f"skeptic.verdict must be 'pass' or 'fail', got {verdict!r}")
    findings = block.get("findings") or []
    unsupported = []
    for i, finding in enumerate(findings):
        status = finding.get("status")
        if status not in SKEPTIC_STATUSES:
            errors.append(f"skeptic.findings[{i}]: status must be one of "
                          f"{sorted(SKEPTIC_STATUSES)}, got {status!r}")
        elif status != "supported":
            unsupported.append(f"{status} on {finding.get('where', '?')}")
    if verdict == "pass" and unsupported:
        errors.append(
            "skeptic.verdict is 'pass' but findings are not all supported: "
            + "; ".join(unsupported)
            + " — a pass alongside an open finding is an inconsistency, not a "
              "judgment call")
    return errors


# ── Citations ────────────────────────────────────────────────────────────

def _validate_all_citations(doc, rules, sections):
    errors = []
    blocks = (
        ("axes", doc.get("axes") or []),
        ("lean_into", doc.get("lean_into") or []),
        ("cut_candidates", doc.get("cut_candidates") or []),
        ("add_candidates", doc.get("add_candidates") or []),
    )
    for key, entries in blocks:
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            where = f"{key}[{i}]"
            validate_citations(_citations_of(entry), rules, where, errors, sections)
    engine = doc.get("engine")
    if isinstance(engine, dict):
        for i, spf in enumerate(engine.get("single_points_of_failure") or []):
            validate_citations(_citations_of(spf), rules,
                               f"engine.single_points_of_failure[{i}]",
                               errors, sections)
    return errors


# ── Entry point ──────────────────────────────────────────────────────────

def validate(doc, deck_doc, deck_path=None, measured_axes=None, rules=None,
             strategy_sections=None):
    """Return a list of error strings (empty = the contract holds)."""
    errors = []
    missing = REQUIRED_TOP_KEYS - set(doc)
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")
        return errors
    if not str(doc.get("verdict", "")).strip():
        errors.append("verdict is empty — say what this deck is good at and what "
                      "actually limits it")

    # A diagnosis states the decklist it described. If that is not the decklist
    # on disk, its figures are answers about a different deck — and re-deriving
    # them one by one produces a wall of mismatches that buries the actual
    # problem. Yawgmoth's post-swap run emitted 26 figure errors for what is one
    # fact. Say the fact, and stop re-deriving.
    stamped = doc.get("as_of_decklist_sha256")
    current = deck_doc.get("decklist_sha256")
    if stamped and current and stamped != current:
        errors.append(
            f"diagnosis describes decklist {stamped[:12]} but cards.json is now "
            f"{current[:12]} — every axis figure in it answers a question about a "
            f"different deck. Re-run the diagnose-deck loop; axis re-derivation is "
            f"skipped below because comparing the two would report noise.")
        measured_axes = None

    cards = deck_doc.get("cards", [])
    main_names = {c["name"] for c in mainboard(cards)}
    bench_names = {c["name"] for c in sideboard(cards)}
    commander_names = {c["name"] for c in cards if c.get("is_commander")}

    errors += _validate_axes(doc, measured_axes)
    errors += _validate_cuts(doc, main_names, commander_names, deck_path)
    errors += _validate_adds(doc, main_names, bench_names, commander_names)
    errors += _validate_bracket_deltas(doc, main_names, commander_names)
    errors += _validate_questions(doc)
    errors += _validate_skeptic(doc)
    errors += _validate_all_citations(doc, rules or {}, strategy_sections)
    return errors


def main(args):
    base = deck_dir(args.slug)
    path = base / "diagnosis.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run the diagnose-deck skill for {args.slug} first.")
    with open(path) as f:
        doc = json.load(f)
    deck_doc = load_deck_cards(args.slug)

    try:
        rules, _, _ = load_rules_db()
    except (FileNotFoundError, ValueError):
        rules = {}
    sections = load_strategy_sections()
    try:
        measured = {a["axis"]: a for a in audit_mod.analyze(args.slug)["axes"]}
    except FileNotFoundError:
        measured = None
        print("WARN deck-audit could not run — axis figures were NOT re-derived")

    errors = validate(doc, deck_doc, deck_path=base, measured_axes=measured,
                      rules=rules, strategy_sections=sections)
    weak = sum(1 for a in doc.get("axes", [])
               if a.get("verdict") in {"weakness", "liability"})
    report_errors(
        path.name, errors,
        f"OK   {path.name} — {len(doc.get('axes', []))} axes ({weak} weak), "
        f"{len(doc.get('cut_candidates', []))} cut / "
        f"{len(doc.get('add_candidates', []))} add candidate(s), "
        f"{len(doc.get('open_questions', []))} open question(s); "
        f"measurements ◆, verdicts ★")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-diagnosis <slug>`.")
