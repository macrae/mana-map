"""Pilot: mechanically enforce the contract on a sideboard analysis.

The sideboard analyst proposes swaps out of a deliberately tiny pool — the cards
sitting next to the deck, and nothing else. That constraint is the whole feature,
so it is checked here rather than trusted: `in` must be a real sideboard card,
`out` must be a maindeck card, and neither may be the commander.

Two rules are worth naming because their absence has already cost something:

  * **`why` must say something.** `validate_build` never reads `plan["swaps"]` at
    all, and hapatra's build plan carries 56 swaps with `why` set to the empty
    string on every one. Nothing rejected it. A swap without a reason is not a
    recommendation, it is a diff.
  * **Bracket deltas are recomputed, never trusted.** A claim that a swap does or
    does not change the deck's power level is the one ◆ fact in an otherwise ★
    section, and an agent asserting it is not evidence. `bracket.assess()` takes a
    plain name list, so verifying costs one call.

Citations go through the same validator the stack and build contracts use, so
there is no second citation implementation to drift.
"""

import json

from manamap.pilot import bracket as bracket_mod
from manamap.pilot.artist_credits import is_accessory
from manamap.pilot.common import (
    deck_dir,
    load_deck_cards,
    mainboard,
    report_errors,
    sideboard,
    try_load_rules_db,
)
from manamap.pilot.validate_stack import load_strategy_sections, validate_citations

REQUIRED_TOP_KEYS = {"slug", "assessment", "swaps", "opens_lines", "long_term_defaults"}
REQUIRED_SWAP_KEYS = {"in", "out", "role", "when", "why"}
# A card either earns a maindeck slot or it does not. Anything vaguer than these
# two is an opinion with nowhere to go.
DEFAULT_VERDICTS = {"promote", "keep-in-sideboard"}
# The magic string five agent prompts already use for an unproven line. Kept
# identical so one grep finds every candidate claim in the repo.
UNVERIFIED_STATUS = "needs a stack scenario"
# An opened line may claim this only by pointing at a stack artifact whose
# checker verdict is `pass` and which actually names the line's cards — the
# claim is re-checked against the file, never trusted.
VERIFIED_STATUS = "verified"


def deck_split(doc):
    """(maindeck names, real sideboard names, accessory names, commander names)."""
    cards = doc.get("cards", [])
    side_all = sideboard(cards)
    main = {c["name"] for c in mainboard(cards)}
    side = {c["name"] for c in side_all if not is_accessory(c)}
    accessories = {c["name"] for c in side_all if is_accessory(c)}
    commanders = {c["name"] for c in cards if c.get("is_commander")}
    return main, side, accessories, commanders


def _validate_swaps(doc, main, side, accessories, commanders):
    errors = []
    seen_out = {}
    for i, swap in enumerate(doc.get("swaps", [])):
        label = f"swap {i} ({swap.get('in')} -> {swap.get('out')})"
        missing = REQUIRED_SWAP_KEYS - set(swap)
        if missing:
            errors.append(f"{label}: missing keys {sorted(missing)}")
            continue

        incoming, outgoing = swap["in"], swap["out"]
        if incoming in accessories:
            errors.append(
                f"{label}: '{incoming}' is a table accessory with no rules text — "
                f"it cannot be played"
            )
        elif incoming not in side:
            errors.append(
                f"{label}: '{incoming}' is not in this deck's sideboard. The analyst "
                f"may only propose cards from the sideboard: {sorted(side) or '(empty)'}"
            )
        if outgoing not in main:
            errors.append(f"{label}: '{outgoing}' is not a maindeck card")
        if outgoing in commanders:
            errors.append(f"{label}: you cannot cut your commander")
        if outgoing in seen_out:
            errors.append(
                f"{label}: '{outgoing}' is already cut by swap {seen_out[outgoing]} — "
                f"two swaps cannot claim the same slot"
            )
        else:
            seen_out[outgoing] = i

        if not str(swap.get("why", "")).strip():
            errors.append(f"{label}: `why` is empty — a swap without a reason is a diff")
        if not str(swap.get("when", "")).strip():
            errors.append(f"{label}: `when` is empty — say the condition that makes this right")
    return errors


def _validate_bracket_deltas(doc, main, commanders):
    """Recompute every claimed bracket delta. Trusting one would be laundering ◆."""
    claimed = [(i, s) for i, s in enumerate(doc.get("swaps", []))
               if isinstance(s.get("bracket_delta"), dict)]
    if not claimed:
        return []
    try:
        card_flags, roles, details = bracket_mod.load_reference()
    except SystemExit:
        return []  # reference artifacts absent: skip, do not fail

    errors = []
    base = bracket_mod.assess(sorted(main), card_flags, roles, details, commanders)
    for i, swap in claimed:
        after_names = (main - {swap.get("out")}) | {swap.get("in")}
        after = bracket_mod.assess(sorted(after_names), card_flags, roles, details, commanders)
        truth = {"before": base["floor"], "after": after["floor"]}
        got = swap["bracket_delta"]
        for key, value in truth.items():
            if key in got and got[key] != value:
                errors.append(
                    f"swap {i} ({swap.get('in')} -> {swap.get('out')}): claims "
                    f"bracket_delta.{key} = {got[key]!r}, recomputed {value!r}"
                )
    return errors


def _validate_lines(doc, deck_path=None):
    errors = []
    for i, line in enumerate(doc.get("opens_lines", [])):
        if not line.get("cards"):
            errors.append(f"opens_lines {i}: no cards named")
        status = line.get("status")
        if status == VERIFIED_STATUS:
            errors += _check_verified_line(i, line, deck_path)
        elif status != UNVERIFIED_STATUS:
            errors.append(
                f"opens_lines {i}: status must be {UNVERIFIED_STATUS!r} (or "
                f"{VERIFIED_STATUS!r} with a checker-passed `stack_artifact`), got "
                f"{status!r} — a line the sideboard opens is a candidate "
                f"until a stack artifact passes the checker"
            )
        if not str(line.get("why_plausible", "")).strip():
            errors.append(f"opens_lines {i}: `why_plausible` is empty")
    return errors


def _check_verified_line(i, line, deck_path):
    """A `verified` claim is re-checked against the stack artifact, never trusted."""
    rel = line.get("stack_artifact")
    if not rel:
        return [f"opens_lines {i}: status {VERIFIED_STATUS!r} requires a "
                f"`stack_artifact` path to a checker-passed stack"]
    if deck_path is None:
        return [f"opens_lines {i}: status {VERIFIED_STATUS!r} cannot be confirmed "
                f"without the deck directory"]
    path = deck_path / rel
    if not path.exists():
        return [f"opens_lines {i}: stack_artifact {rel!r} does not exist"]
    with open(path) as f:
        artifact = json.load(f)
    if artifact.get("checker", {}).get("verdict") != "pass":
        return [f"opens_lines {i}: stack_artifact {rel!r} has checker verdict "
                f"{artifact.get('checker', {}).get('verdict')!r}, not 'pass' — "
                f"only a passing stack verifies a line"]
    body = json.dumps(artifact, ensure_ascii=False)
    missing = [c for c in line.get("cards", []) if c not in body]
    if missing:
        return [f"opens_lines {i}: stack_artifact {rel!r} never mentions "
                f"{missing} — it cannot verify this line"]
    return []


def _validate_defaults(doc, side, accessories):
    errors = []
    for i, entry in enumerate(doc.get("long_term_defaults", [])):
        name = entry.get("card")
        if name not in side:
            where = "a table accessory" if name in accessories else "not in the sideboard"
            errors.append(f"long_term_defaults {i}: '{name}' is {where}")
        if entry.get("verdict") not in DEFAULT_VERDICTS:
            errors.append(
                f"long_term_defaults {i} ({name}): verdict must be one of "
                f"{sorted(DEFAULT_VERDICTS)}, got {entry.get('verdict')!r}"
            )
        if not str(entry.get("why", "")).strip():
            errors.append(f"long_term_defaults {i} ({name}): `why` is empty")
    return errors


def validate(doc, deck_doc, rules=None, strategy_sections=None, deck_path=None):
    """Return a list of error strings (empty = the contract holds)."""
    errors = []
    missing = REQUIRED_TOP_KEYS - set(doc)
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")
        return errors

    if not str(doc.get("assessment", "")).strip():
        errors.append("assessment is empty — say what this sideboard is for")

    main, side, accessories, commanders = deck_split(deck_doc)
    errors += _validate_swaps(doc, main, side, accessories, commanders)
    errors += _validate_lines(doc, deck_path)
    errors += _validate_defaults(doc, side, accessories)
    errors += _validate_bracket_deltas(doc, main, commanders)

    for i, swap in enumerate(doc.get("swaps", [])):
        citations = swap.get("citations") or []
        if citations:
            validate_citations(citations, rules or {}, f"swap {i}", errors, strategy_sections)
    return errors


def main(args):
    base = deck_dir(args.slug)
    path = base / "sideboard_analysis.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run the analyse-sideboard skill for {args.slug} first."
        )
    with open(path) as f:
        doc = json.load(f)
    deck_doc = load_deck_cards(args.slug)

    rules = try_load_rules_db()

    errors = validate(doc, deck_doc, rules, load_strategy_sections(), deck_path=base)
    report_errors(
        path.name, errors,
        f"OK   {path.name} — {len(doc.get('swaps', []))} swap(s), "
        f"{len(doc.get('opens_lines', []))} candidate line(s); coaching tier")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-sideboard <slug>`.")
