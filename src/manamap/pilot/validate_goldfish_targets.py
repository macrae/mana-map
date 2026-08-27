"""Pilot: mechanically check a deck's goldfish target declaration.

`goldfish_targets.json` is a machine-readable ENGINE DECLARATION — its `any_of`
groups are the engine's components and a group's SIZE is that component's
redundancy. `deck_audit` prices those groups and quotes the rates in the engine
block, and a diagnosis then sizes its prescription off them. Nothing checked the
declaration itself, so the engine block could report an assembly rate with
complete confidence for a component that cannot assemble.

A seven-deck diagnosis run found the declaration wrong on six of eight decks, in
four different shapes: a component every one of whose members a checker-passed
stack refutes; a group declaring two cards where the deck holds four; a group
whose members do not all perform the declared function; and — twice — a deck
whose PRIMARY WIN LINE, verified by a passing stack, has no target at all. The
last is the one this module is really for: heliod's Hullbreaker Horror line and
ur-dragon's Aggravated Assault are each named in two passing stacks and in no
component, so the simulator has never once measured how those decks actually win.

Three checks, each measured against the whole fleet before being written:

  * every declared card must still be in the 99 — the staleness guard, because a
    swap silently strands the name it removed;
  * a card appearing in TWO OR MORE checker-passed stacks and in no component is
    reported as a likely omission. Commanders and basic lands are excluded: both
    are on every board and neither says anything about the engine.
  * no card may sit in TWO `any_of` legs of the same target — an AND of ORs
    satisfied twice over by one draw. See `_validate_shared_leg`; it fires on
    2 of 9 decks and both are genuine.

A check deliberately NOT implemented: "members of a group should share a role
axis". Prototyped against all eight decks, it fired hardest on the most correct
groups — goblin-storm's cantrip group agreed 1/6 with every member right, and it
flagged Howling Mine and Font of Mythos as outliers in heliod's draw engine,
which they ARE. The taxonomy is the reason: `ROLE_PATTERNS` answers "what job
does this card do in a 99", while a goldfish group is a deck-specific functional
set with no taxonomy axis (there is no `cantrip` role). A validator that fires on
correct data trains its reader to ignore it, which is worse than not having it.
"""

import json

from manamap.pilot.agent_cache import passing_stacks
from manamap.pilot.common import (
    scenario_game_state,
    deck_dir,
    load_deck_cards,
    load_json,
    report_errors,
)

# A card must appear in at least this many checker-passed stacks before its
# absence from every component is worth reporting. One passing stack is a line;
# two is a pattern, and both real omissions the fleet survey found clear it.
WIN_LINE_QUORUM = 2


def _declared_names(doc):
    """Every card named anywhere in the declaration."""
    names = set()
    for target in doc.get("targets", []):
        for need in target.get("need", []) or []:
            names.update(need.get("any_of", []) or [])
    return names


def _validate_shape(doc):
    errors = []
    targets = doc.get("targets")
    if not isinstance(targets, list) or not targets:
        return ["targets must be a non-empty list — a deck with no declared "
                "target has no engine block and no measured assembly rate"]
    seen_labels = {}
    for i, target in enumerate(targets):
        label = target.get("label")
        if not str(label or "").strip():
            errors.append(f"targets[{i}]: label is empty — the engine block "
                          f"prints it, and an unnamed component cannot be cited")
            label = f"<targets[{i}]>"
        if label in seen_labels:
            errors.append(f"targets[{i}] ({label}): duplicate label, first used "
                          f"at targets[{seen_labels[label]}]")
        else:
            seen_labels[label] = i

        need = target.get("need")
        if not isinstance(need, list) or not need:
            errors.append(f"targets[{i}] ({label}): need must be a non-empty list")
            continue
        for j, group in enumerate(need):
            members = group.get("any_of")
            if not isinstance(members, list) or not members:
                errors.append(f"targets[{i}] ({label}) need[{j}]: any_of must be "
                              f"a non-empty list of card names")
                continue
            dupes = sorted({m for m in members if members.count(m) > 1})
            if dupes:
                errors.append(
                    f"targets[{i}] ({label}) need[{j}]: {dupes} listed twice — a "
                    f"group's size is its redundancy, so a duplicate overstates it")
    return errors


def _validate_shared_leg(doc):
    """No card may sit in two `any_of` legs of the SAME target.

    A multi-leg target is an AND of ORs, and `goldfish._target_met` marks a need
    met if ANY card it names is in hand. So a card appearing in two legs of one
    target satisfies both from a single draw, and the target reports an assembly
    rate it has not earned — silently, because every name in it is a real card
    doing a real job.

    Measured against all nine tracked declarations before being kept, per this
    module's own rule. It fires on exactly TWO decks and both are genuine:
    ur-dragon's THE COMBAT KILL (Thrakkus the Butcher in both legs, worth 5.8
    points of by-turn-six rate) and edgar's THE GO-WIDE KILL (Charismatic
    Conqueror, worth 1.6). Zero false positives — which is why this one shipped
    and the two checks proposed for `validate_engine` the same day did not, at
    50% and 1.4% precision respectively.

    The size of the error tracks the SCARCER leg, not the shared card: edgar's
    cost little because the leg it left is thirteen deep, ur-dragon's cost four
    times as much because its leg is four. The fix is always to keep the card in
    the leg where its job is rarer.
    """
    errors = []
    for i, target in enumerate(doc.get("targets", []) or []):
        label = target.get("label", f"target {i}")
        legs = [set(g.get("any_of") or []) for g in (target.get("need") or [])]
        for a in range(len(legs)):
            for b in range(a + 1, len(legs)):
                for name in sorted(legs[a] & legs[b]):
                    errors.append(
                        f"targets[{i}] ({label}): '{name}' is in BOTH need[{a}] "
                        f"and need[{b}] — one drawn card satisfies both, so this "
                        f"target's rate is an upper bound on itself. Keep it in "
                        f"the leg where its job is scarcer")
    return errors


def _validate_membership(doc, main_names, commander_names):
    """Every declared card must still be in the 99."""
    errors = []
    known = main_names | commander_names
    for name in sorted(_declared_names(doc)):
        if name not in known:
            errors.append(
                f"'{name}' is declared in a target but is not in the deck — a swap "
                f"stranded the name, so this group's size overstates its redundancy")
    return errors


def _validate_win_line_coverage(doc, slug, main_names, commander_names, base):
    """A card carrying two or more passing stacks belongs to some component."""
    declared = _declared_names(doc)
    deck_doc = None
    basics = set()
    try:
        deck_doc = load_deck_cards(slug)
        basics = {c["name"] for c in deck_doc.get("cards", [])
                  if "Basic Land" in str(c.get("type_line", ""))}
    except Exception:                      # pragma: no cover — fresh clone
        pass

    counts = {}
    stacks = list(passing_stacks(base))
    if not stacks:
        return []                          # nothing verified yet; nothing to say
    for path in stacks:
        try:
            blob = json.dumps(scenario_game_state(
                load_json(path).get("scenario", {})))
        except Exception:                  # pragma: no cover
            continue
        for name in main_names:
            if name in blob:
                counts[name] = counts.get(name, 0) + 1

    errors = []
    for name, n in sorted(counts.items()):
        if n < WIN_LINE_QUORUM or name in declared:
            continue
        if name in commander_names or name in basics:
            continue                       # on every board; says nothing
        errors.append(
            f"'{name}' appears in {n} checker-passed stacks and in no target — "
            f"the simulator never measures it, so the engine block reports rates "
            f"for a plan this card is not part of")
    return errors


def validate(doc, slug, base):
    """Return a list of error strings (empty = the declaration holds)."""
    errors = _validate_shape(doc)
    if errors:
        return errors                      # shape first; the rest would cascade

    try:
        deck_doc = load_deck_cards(slug)
    except Exception:                      # pragma: no cover — fresh clone
        return errors
    cards = deck_doc.get("cards", [])
    main_names = {c["name"] for c in cards}
    commander_names = {c["name"] for c in cards if c.get("is_commander")}

    errors += _validate_shared_leg(doc)
    errors += _validate_membership(doc, main_names, commander_names)
    errors += _validate_win_line_coverage(doc, slug, main_names,
                                          commander_names, base)
    return errors


def main(args):
    base = deck_dir(args.slug)
    path = base / "goldfish_targets.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — a deck with no declared targets has no engine "
            f"block. Author it before running `manamap pilot goldfish {args.slug}`.")
    with open(path) as f:
        doc = json.load(f)
    errors = validate(doc, args.slug, base)
    groups = sum(len(t.get("need", []) or []) for t in doc.get("targets", []))

    # AN UNEDITED SCAFFOLD IS REPORTED, NOT FAILED.
    #
    # `scaffold-targets` writes a starting file so a new deck is not a blank
    # page. Its groups are role axes, which this module's own docstring records
    # are NOT what a goldfish component is — so a scaffold that nobody rewrites
    # would have the simulator reporting assembly rates for generic buckets
    # while every reader believes it is measuring the engine. That is the
    # `DECK_ROLE_BUDGET` failure exactly: provisional, labelled provisional, and
    # left in place for months.
    #
    # It is not an ERROR, because a scaffold is a legitimate intermediate state
    # and a gate that reddens one teaches its reader to ignore the gate. It is
    # said on every run instead, so the state cannot go quiet.
    scaffold_note = ""
    if doc.get("scaffolded"):
        derived = sum(1 for t in doc.get("targets", [])
                      if str(t.get("_from", "")).startswith("role:"))
        scaffold_note = (
            f"\n     SCAFFOLD — never edited. {derived} of "
            f"{len(doc.get('targets', []))} target(s) are role axes rather than "
            f"this deck's components, and no win line is declared. Rewrite the "
            f"labels, regroup, then delete \"scaffolded\".")

    # NO `required` MARKING SILENTLY DISABLES THE FLAGSHIP METRIC, and the
    # validator used to print a clean OK over it. `diagnostic.engine` needs to
    # know which components the deck cannot do without; absent that it withholds
    # the figure — correctly, and out of sight of anyone running this. Measured
    # 2026-08-26: **1 of 13 decks** carries the marking, so `engine_online` and
    # every axis built on it were validated on a sample of one.
    #
    # REPORTED, NOT FAILED — same reasoning as the scaffold note above. A
    # declaration without it is a legitimate older file, not a broken one, and a
    # gate that reddens twelve correct artifacts teaches its reader to ignore
    # the gate. Saying it on every run is what keeps the state from going quiet.
    required_note = ""
    targets = doc.get("targets", [])
    if targets and not any(t.get("required") for t in targets):
        routes = sum(1 for t in targets if t.get("route"))
        required_note = (
            f"\n     NO `required` MARKING — `diagnose` cannot report an engine "
            f"figure for this deck and withholds it silently. Mark the "
            f"component(s) the deck cannot function without with "
            f"\"required\": true; the alternative kills take \"route\": "
            f"\"<name>\" and are counted as a union"
            + (f" ({routes} already carry a route)." if routes else "."))

    report_errors(
        path.name, errors,
        f"OK   {path.name} — {len(doc.get('targets', []))} target(s), "
        f"{groups} component group(s); sizes are redundancy claims ◆"
        + required_note + scaffold_note)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-goldfish-targets <slug>`.")
