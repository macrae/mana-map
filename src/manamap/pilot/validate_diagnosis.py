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
    UNVERIFIED_STATUS,
    VERIFIED_STATUS,
    checker_passed,
    deck_dir,
    load_card_roles,
    load_deck_cards,
    load_json,
    load_rules_db,
    mtime_memo,
    report_errors,
)
from manamap.pilot.validate_stack import load_strategy_sections, validate_citations

REQUIRED_TOP_KEYS = {"slug", "verdict", "axes", "engine", "lean_into",
                     "cut_candidates", "add_candidates", "open_questions", "gaps"}
REQUIRED_AXIS_KEYS = {"axis", "verdict", "measured", "reading"}
REQUIRED_CUT_KEYS = {"card", "why", "cost_of_cutting", "difficulty"}
REQUIRED_ADD_KEYS = {"card", "closes", "source", "why"}
REQUIRED_QUESTION_KEYS = {"question", "settled_by", "why_it_matters"}

AXIS_VERDICTS = {"strength", "adequate", "weakness", "liability"}
CUT_DIFFICULTIES = {"easy", "contested", "painful"}
ADD_SOURCES = {"pool", "recon"}
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

def _validate_adds(doc, main_names, commander_names):
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


# ── Does the prescription do what it says? ───────────────────────────────
#
# An add's `closes` names what the card is bought for, and four adds across three
# decks in one fleet run named an axis the axis does not credit them for: Nature's
# Claim doubled two covered classes, Walking Ballista carries no `wincon:*` role so
# threat-density could not see it, Bojuka Bog is only `land:tapped` so the breadth
# function skips it before reading its text. In every case the card does the thing
# in Magic terms and the MEASURE does not move — so the prescription was sized
# against a number buying it would not change.
#
# Only axes computable from (cards, roles) alone are checked. colour-sources and
# mana-base need mana_analysis and goldfish, and a half-recomputed axis would be a
# worse answer than no answer. `closes` is free prose on most entries, so an entry
# that does not name a known axis is skipped rather than failed — the field is a
# sentence, not an enum, and forcing it into one would be a schema change.
_BREADTH_AXIS = "interaction-breadth"


def _computable_axes():
    return set(audit_mod.AXIS_ROLES) | {_BREADTH_AXIS}


def _axis_value(axis, cards, roles):
    """Recompute one axis from (cards, roles), or None if it needs more."""
    if axis in audit_mod.AXIS_ROLES:
        copies, _ = audit_mod._count_copies(cards, roles, audit_mod.AXIS_ROLES[axis])
        return copies
    if axis == _BREADTH_AXIS:
        breadth = audit_mod._interaction_breadth(cards, roles)
        return sum(1 for names in breadth.values() if names)
    return None


def _named_axis(closes):
    """The axis `closes` names, if it names one. Prose entries return None.

    LONGEST match wins. `interaction` is an axis and so is `interaction-breadth`,
    and a shortest-first scan resolves the latter to the former — which silently
    checks the wrong axis and reports the wrong number in the error text.
    """
    text = str(closes or "").strip()
    for axis in sorted(_computable_axes(), key=len, reverse=True):
        if text == axis or text.startswith(axis):
            return axis
    return None


def _corpus_card(name, roles):
    """A minimal card record for a card outside the deck, or None if unknown."""
    oracle = _corpus_oracle().get(name)
    if oracle is None:
        return None
    return {"name": name, "oracle_text": oracle, "quantity": 1,
            "type_line": ""}


def _read_corpus_oracle():
    import pandas as pd

    from manamap.config import OUTPUT_CSV_PATH
    df = pd.read_csv(OUTPUT_CSV_PATH, usecols=["name", "oracle_text"])
    return {n: (t if isinstance(t, str) else "")
            for n, t in zip(df["name"], df["oracle_text"])}


def _corpus_oracle():
    """{name: oracle_text} from cards.csv, once per (mtime, size). {} if absent."""
    from manamap.config import OUTPUT_CSV_PATH
    try:
        return mtime_memo(OUTPUT_CSV_PATH, "validate_diagnosis:oracle",
                          _read_corpus_oracle, absent={}) or {}
    except Exception:                      # pragma: no cover — unreadable corpus
        return {}


def _validate_prescription_moves(doc, deck_doc, roles):
    """Each add's named axis must move, and the whole swap set must clear floors."""
    if not roles or not _corpus_oracle():
        return []                          # no roles or no corpus: skip, never fail
    errors = []
    main_cards = list(deck_doc.get("cards", []))
    adds = [a for a in doc.get("add_candidates") or [] if isinstance(a, dict)]

    # (a) MARGINAL contribution: does this card move its axis GIVEN the rest of the
    # prescription? Isolation is the wrong frame and misses the commonest shape of
    # the defect — Nature's Claim alone takes hapatra's breadth 1 -> 3, so it looks
    # fine tested by itself, but Assassin's Trophy is bought in the same package and
    # already covers both classes, so the pair is 4 and the trio is also 4. The card
    # earns a slot only if the axis it names is different with it than without it.
    records = {}
    for add in adds:
        rec = _corpus_card(add.get("card"), roles)
        if rec is not None:
            records[add.get("card")] = rec
    for i, add in enumerate(adds):
        axis = _named_axis(add.get("closes"))
        name = add.get("card")
        if not axis or name not in records:
            continue                       # prose `closes`, or unknown to the corpus
        others = [r for n, r in records.items() if n != name]
        without = _axis_value(axis, main_cards + others, roles)
        with_it = _axis_value(axis, main_cards + others + [records[name]], roles)
        if without is None or with_it is None or with_it != without:
            continue
        alone = _axis_value(axis, main_cards + [records[name]], roles)
        base = _axis_value(axis, main_cards, roles)
        detail = (f" (it moves the axis {base} -> {alone} on its own, so the "
                  f"other adds in this prescription already cover what it covers)"
                  if others and alone != base else "")
        errors.append(
            f"add_candidates[{i}] ({name}): closes {axis!r}, but with the rest of "
            f"this prescription applied that axis is {without} with or without it"
            f"{detail} — the slot buys no movement on the axis it names")

    # (b) the PAIRED SWAPS together, against the floors the doc itself cites.
    # sisay's swaps each looked sound and landed threat-density at 2 because Beast
    # Within's single removal:spot was spent covering two different cuts.
    #
    # Only (add, natural_cut) pairs are applied, NOT every listed candidate. Two
    # skeptics adjudicated that distinction independently: cut_candidates is a
    # RANKED LIST, not a mandated set, and sisay deliberately lists a `painful` cut
    # it simultaneously holds — stated three times with a lift condition, and with
    # no natural_cut pointing at it. Applying every listed cut would fail that
    # document for a swap it does not prescribe, which is the check firing on
    # correct data.
    pairs = [(a, a.get("natural_cut")) for a in adds if a.get("natural_cut")]
    if not pairs:
        return errors
    paired_cuts = {cut for _, cut in pairs}
    swapped = [c for c in main_cards if c["name"] not in paired_cuts]
    for add, _ in pairs:
        record = _corpus_card(add.get("card"), roles)
        if record is not None:
            swapped.append(record)
    cited = {a.get("axis") for a in doc.get("axes") or [] if isinstance(a, dict)}
    for axis in sorted(cited & _computable_axes()):
        low = (DECK_AXIS_TARGETS.get(axis) or {}).get("low")
        if low is None:
            continue
        before = _axis_value(axis, main_cards, roles)
        after = _axis_value(axis, swapped, roles)
        if before is None or after is None:
            continue
        if after < low <= before:
            errors.append(
                f"applying every cut and add together takes {axis} from {before} "
                f"to {after}, under the floor of {low} this diagnosis cites for it "
                f"— each swap may be sound alone and the set still not be")
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
    main_names = {c["name"] for c in cards}
    commander_names = {c["name"] for c in cards if c.get("is_commander")}

    errors += _validate_axes(doc, measured_axes)
    errors += _validate_cuts(doc, main_names, commander_names, deck_path)
    errors += _validate_adds(doc, main_names, commander_names)
    errors += _validate_bracket_deltas(doc, main_names, commander_names)
    errors += _validate_prescription_moves(doc, deck_doc, load_card_roles())
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
