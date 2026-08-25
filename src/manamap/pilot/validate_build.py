"""Pilot: mechanically enforce the contract on a build plan.

Code enforces **form**, humans and the critic agent judge **substance**. This
module answers only the questions with objective answers: is the deck 100
cards, is every name real, is every card inside the commander's colour
identity, does the role budget add up, and does the declared bracket survive
the computed floor.

Citations are optional here in a way they are not for a stack. The
deterministic builder makes no claims, so it carries none. The architect agent
does make claims — every ratio it asserts must trace to a strategy section —
and those go through the *same* validator the stack contract uses, so there is
no second citation implementation to drift.
"""

import json


from manamap.config import (
    BRACKETS,
    DECKS_DIR,
    OUTPUT_CSV_PATH,
)
from manamap.analysis.common import parse_color_identity
from manamap.pilot import formats
from manamap.pilot.card_pool import load_frame
from manamap.pilot.common import (
    deck_dir, mtime_memo, report_errors, try_load_rules_db)
from manamap.pilot.validate_stack import load_strategy_sections, validate_citations

REQUIRED_TOP_KEYS = {"slug", "commander", "color_identity", "bracket", "slots", "land_counts"}
CRITIC_STATUSES = {
    "supported", "unjustified", "miscounted", "off-bracket", "off-identity", "unverified-line",
}


def deck_card_names(plan):
    """Every card in the plan: commander, slots, and lands with quantities."""
    names = [plan.get("commander")] if plan.get("commander") else []
    names += [s.get("name") for s in plan.get("slots", []) if s.get("name")]
    for name, count in (plan.get("land_counts") or {}).items():
        names += [name] * int(count)
    return names


def validate(plan, cards=None, rules=None, strategy_sections=None, bracket_report=None):
    """Return a list of human-readable error strings (empty = the form holds).

    `cards` maps name -> {color_identity, legal_commander, type_line}. None
    means "skip the card-reality checks" — not "the deck has no cards".
    """
    errors = []

    missing = REQUIRED_TOP_KEYS - set(plan)
    if missing:
        errors.append(f"missing required key(s): {', '.join(sorted(missing))}")
        return errors

    names = deck_card_names(plan)
    # Legality, so it reads the format rather than a build constant.
    spec = formats.DEFAULT
    if len(names) != spec.deck_size:
        errors.append(f"plan has {len(names)} cards, expected exactly {spec.deck_size}")

    # Singleton, basics excepted.
    seen = {}
    for name in names:
        seen[name] = seen.get(name, 0) + 1
    for name, count in sorted(seen.items()):
        if count > 1 and cards is not None:
            type_line = cards.get(name, {}).get("type_line", "")
            if "Basic" not in type_line:
                errors.append(f"singleton violation: {name} x{count}")

    identity = set(plan.get("color_identity") or [])

    if cards is not None:
        for name in sorted(set(names)):
            card = cards.get(name)
            if card is None:
                errors.append(f"unknown card: {name!r} is not in cards.csv")
                continue
            if card.get("legal_commander") != "legal":
                errors.append(f"{name} is not legal in Commander "
                              f"({card.get('legal_commander')})")
            outside = parse_color_identity(card.get("color_identity", "")) - identity
            if outside:
                errors.append(
                    f"color identity violation: {name} is {sorted(outside)} outside "
                    f"commander identity {sorted(identity) or ['C']}"
                )

    errors.extend(_validate_pool(plan, cards))
    errors.extend(_validate_bracket(plan, bracket_report))
    errors.extend(_validate_budget(plan))
    errors.extend(_validate_land_list(plan))
    errors.extend(_validate_slots(plan))
    errors.extend(_validate_manabase(plan))
    errors.extend(_validate_critic(plan))

    for i, slot in enumerate(plan.get("slots", [])):
        for citation_list, where in (
            (slot.get("citations", []), f"slot {i} ({slot.get('name')})"),
        ):
            if citation_list:
                validate_citations(citation_list, rules or {}, where, errors, strategy_sections)
    for key in ("role_budget_citations", "gameplan_citations"):
        if plan.get(key):
            validate_citations(plan[key], rules or {}, key, errors, strategy_sections)

    return errors


def _validate_pool(plan, cards):
    """If the build declared a collection, prove it stayed inside it.

    Re-derives the pool from the decklist files the plan names rather than
    trusting a count written by the thing being validated. Basics are exempt:
    `build_deck` deliberately draws them from the format, because you always own
    another Swamp and a pool that happens to list four should not cap the mana
    base at four.
    """
    pool_spec = plan.get("pool")
    if not pool_spec or not pool_spec.get("files"):
        return []
    try:
        from manamap.pilot.pool_facts import collect_paths, load_cards, read_sources

        per_file, _ = read_sources(collect_paths(pool_spec["files"], []), load_cards())
    except SystemExit as exc:
        return [f"pool files named by the plan could not be read: {exc}"]

    owned = set()
    for counts in per_file.values():
        owned.update(counts)

    # The brief's explicit `pool` list is part of the declared pool too —
    # `resolve_pool` merges it with the files (it exists for cards whose
    # acquisition is the build's premise, like a just-released commander).
    # Read it from the BRIEF, the authored input, never from the plan being
    # validated.
    brief_path = DECKS_DIR / plan.get("slug", "") / "brief.json"
    if brief_path.exists():
        with open(brief_path) as f:
            owned.update(json.load(f).get("pool") or [])

    outside = []
    for name in sorted(set(deck_card_names(plan))):
        if name in owned:
            continue
        if "Basic" in (cards or {}).get(name, {}).get("type_line", ""):
            continue
        outside.append(name)
    return [f"card outside the declared pool: {n}" for n in outside]


def _validate_bracket(plan, bracket_report=None):
    """The plan's bracket claim, cross-checked against the engine's own artifact.

    A plan self-reporting its floor is a claim, not evidence. When
    bracket_report.json is present it is the ◆ record and the plan must agree
    with it — a stale floor left behind by a later swap is exactly the failure
    the deck-critic had to catch by hand the first time.
    """
    errors = []
    block = plan.get("bracket")
    if not isinstance(block, dict):
        return ["bracket must be an object with target/computed_floor"]
    target = block.get("target")
    floor = block.get("computed_floor")
    if target not in BRACKETS:
        errors.append(f"bracket.target must be 1-{max(BRACKETS)}, got {target!r}")
        return errors
    if floor is None:
        errors.append("bracket.computed_floor is missing — run the bracket engine")
    elif floor > target:
        errors.append(
            f"bracket floor {floor} exceeds target {target} ({BRACKETS[target]['name']}) — "
            f"the deck is out of tier"
        )

    if bracket_report is not None and floor is not None:
        measured = bracket_report.get("floor")
        if measured is not None and measured != floor:
            errors.append(
                f"bracket.computed_floor is {floor} but bracket_report.json measured "
                f"{measured} — re-run `manamap pilot bracket-check`"
            )
    return errors


def _validate_budget(plan):
    """Budget arithmetic, per role as well as in total.

    Checking only the total lets an author re-label slots until the sum lands
    while every individual line is wrong — the deck-critic caught exactly that
    on the first real build (declared 9 ramp against 10 ramp slots, 9 draw
    against 6, 10 removal against 12; net zero). A budget that doesn't describe
    its own deck is not a budget.
    """
    budget = plan.get("role_budget") or {}
    if not budget:
        return ["role_budget is missing"]
    counts = {}
    for slot in plan.get("slots", []):
        counts[slot.get("role")] = counts.get(slot.get("role"), 0) + 1
    lands = sum((plan.get("land_counts") or {}).values())
    errors = []

    if budget.get("lands") and lands != budget["lands"]:
        errors.append(f"land_counts total {lands}, role_budget says {budget['lands']}")

    declared = sum(v for k, v in budget.items() if k != "lands")
    actual = sum(counts.values())
    if declared != actual:
        errors.append(f"role_budget declares {declared} nonland slots, plan has {actual}")

    for role, want in sorted(budget.items()):
        if role == "lands":
            continue
        have = counts.get(role, 0)
        if have != want:
            errors.append(
                f"role_budget says {want} {role}, plan labels {have} slot(s) {role}"
            )
    for role in sorted(set(counts) - set(budget)):
        errors.append(f"{counts[role]} slot(s) labelled {role!r}, which the budget never declares")

    return errors


def _validate_land_list(plan):
    """`lands` and `land_counts` must describe the same 36 cards."""
    counts = plan.get("land_counts") or {}
    listed = plan.get("lands")
    if listed is None:
        return []
    if sorted(set(listed)) != sorted(set(counts)):
        only_listed = sorted(set(listed) - set(counts))
        only_counted = sorted(set(counts) - set(listed))
        return [
            "lands and land_counts disagree — "
            f"only in lands: {only_listed or '(none)'}; "
            f"only in land_counts: {only_counted or '(none)'}"
        ]
    return []


def _validate_manabase(plan):
    """The mana base must describe the spells the plan actually runs.

    Diagnostics are computed against a spell list; if slots change afterwards
    and nobody recomputes, the plan reports a mana base for a deck it no longer
    is. That is not a cosmetic staleness — the deck-critic found a build whose
    swaps pushed black from one pip to two, moving the source target from 22 to
    36 while the plan still claimed no shortfall.
    """
    diag = plan.get("manabase")
    if not isinstance(diag, dict):
        return []
    slots = len(plan.get("slots", []))
    counted = diag.get("spell_slots")
    if counted is not None and counted != slots:
        return [
            f"manabase diagnostics were computed against {counted} spell slot(s) "
            f"but the plan now has {slots} — re-run the mana base"
        ]
    return []


def _validate_slots(plan):
    errors = []
    for i, slot in enumerate(plan.get("slots", [])):
        if not slot.get("name"):
            errors.append(f"slot {i} has no name")
        if not slot.get("role"):
            errors.append(f"slot {i} ({slot.get('name')}) has no role")
    return errors


def _validate_critic(plan):
    """Same verdict-consistency rule the stack contract uses."""
    critic = plan.get("critic")
    if critic is None:
        return []
    errors = []
    if critic.get("verdict") not in {"pass", "fail"}:
        errors.append(f"critic.verdict must be pass|fail, got {critic.get('verdict')!r}")
    for finding in critic.get("findings", []):
        if finding.get("status") not in CRITIC_STATUSES:
            errors.append(
                f"critic finding for {finding.get('card') or finding.get('claim')}: "
                f"invalid status {finding.get('status')!r}"
            )
    if critic.get("verdict") == "pass":
        bad = [f for f in critic.get("findings", []) if f.get("status") != "supported"]
        if bad:
            errors.append(
                f"critic verdict is 'pass' but {len(bad)} finding(s) are not 'supported'"
            )
    return errors


def _build_reference():
    frame = load_frame()
    return {
        name: {"color_identity": ci, "legal_commander": legal, "type_line": tl}
        for name, ci, legal, tl in zip(
            frame["name"], frame["color_identity"],
            frame["legal_commander"], frame["type_line"])
    }


def load_cards_reference():
    """name -> {color_identity, legal_commander, type_line} from cards.csv.

    Shares `card_pool`'s single parse: `validate-build` used to read the CSV
    here and again inside `pool_facts.load_cards` when the plan named a pool.
    """
    if not OUTPUT_CSV_PATH.exists():
        return None
    return mtime_memo(OUTPUT_CSV_PATH, "validate_build:reference", _build_reference)


def main(args):
    path = deck_dir(args.slug) / "build_plan.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run `manamap pilot build-deck {args.slug}` first."
        )
    with open(path) as f:
        plan = json.load(f)

    cards = load_cards_reference()
    rules = try_load_rules_db()
    strategy_sections = load_strategy_sections()

    # Best-effort: absent bracket_report.json only means the cross-check is
    # skipped, not that the plan is wrong.
    bracket_path = deck_dir(args.slug) / "bracket_report.json"
    bracket_report = json.loads(bracket_path.read_text()) if bracket_path.exists() else None

    errors = validate(plan, cards, rules, strategy_sections, bracket_report)
    report_errors(path.name, errors)
    block = plan["bracket"]
    print(
        f"OK   {path.name} — {plan['commander']}, {formats.DEFAULT.deck_size} cards, "
        f"bracket {block['target']} ({BRACKETS[block['target']]['name']}), "
        f"floor {block.get('computed_floor')}"
    )


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-build <slug>`.")
