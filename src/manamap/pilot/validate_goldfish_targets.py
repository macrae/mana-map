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
import re

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


#: A route whose LABEL says the kill lands in combat. Matched on the label
#: because that is where the pilot writes what the route IS; a route's members
#: are cards, and a card cannot say whether the plan needs to connect.
#:
#: Measured across the fleet before it was written: it fires on 2 of 10 decks
#: (ur-dragon and zur-enchantress) and both are genuine combat routes. That is
#: the entry criterion here — a check that fires on correct data is worse than
#: no check — and this one is a NOTE rather than a failure, because a combat
#: route is a legitimate thing to declare. What is not legitimate is reading the
#: goldfish's number for it as evidence that the kill lands.
#: Route values whose kill has to CONNECT, so the goldfish's assembly rate is
#: about the DRAW and never about the kill. `drain` and `entry` are absent
#: deliberately — a drain does not care about blockers and the model grades it
#: honestly. Kept as a set rather than folded into `_COMBAT_ROUTE` because a
#: route is a declared token and a label is prose; matching them with one regex
#: is what let a rename switch the note off.
#:
#: HELD TO THE VALUES THE FLEET ACTUALLY USES. The first draft of this also
#: listed "commander" and "voltron", which sound obviously right and appear
#: nowhere — and adding them broke a fixture whose default route is "commander",
#: which is how the speculation announced itself. Widening a matcher past what
#: was swept is the mistake this file's own history is made of. Add a value when
#: a deck declares it, not before.
_COMBAT_ROUTES = frozenset({"board", "combat"})

_COMBAT_ROUTE = re.compile(
    r"commander damage|combat|attack|swing|voltron|lord or anthem", re.I)


def _combat_route_notes(doc):
    """Routes the goldfish structurally cannot grade, and why.

    THE GOLDFISH HAS NO BLOCKERS. It reports whether the PIECES WERE DRAWN, and
    for a kill that has to connect against three or four opponents that is a
    different question from whether the kill happens — the distinction the
    Edgar go-wide refactor already paid for, where the model preferred a list
    Forge then scored 31/400 against the champion's 50/400 because 1/1 tokens
    do not connect.

    Zur is the same class one turn further along and cost 39 games to find. Its
    V6 engine is *"Zur attacks, fetches an aura, connects with lifelink, Vito
    drains"* — every step gated on a 1/4 commander connecting. The goldfish
    reported the route assembled by turn six in 23.6% of games. Forge, on the
    pilot's own table: **0.35 commander damage a game, best single game 2, and
    0 of 39 games reaching 21.** The route was never wrong about the draw and
    was never evidence about the kill.

    Sharper when the combat model is OFF, because then the figure is purely
    "the cards arrived" — there is not even a modelled swing behind it.
    """
    notes = []
    combat_on = doc.get("model_combat") is True
    for target in doc.get("targets", []):
        label = target.get("label") or ""
        route = target.get("route")
        # THE STRUCTURED FIELD FIRST, the label prose second. `_COMBAT_ROUTE`
        # greps the label, and a label is free text: zur-enchantress renamed its
        # kill from "KILL — commander damage: a buff aura on Zur" to "KILL — a
        # BOARD: a real body plus a way through" and the note stopped firing on
        # a route that still has to CONNECT — the exact thing the note exists to
        # say. The rename was correct and the matcher was reading the wrong
        # field.
        #
        # Swept across all 13 declarations on 2026-09-04 before this was kept:
        # 4 newly firing, all of them the same board route on zur-enchantress
        # and its three branches, all genuinely combat kills. Zero decks that
        # were correct started reporting. The whole route vocabulary in the
        # fleet is {entry, combat, board, drain}; `drain` and `entry` are graded
        # fine by a model with no blockers, which is why they are not here.
        if not route or not (route in _COMBAT_ROUTES or _COMBAT_ROUTE.search(label)):
            continue
        head = f"\n     COMBAT ROUTE — \"{label[:56]}\"."
        if combat_on:
            notes.append(
                head + " The goldfish models a swing but has NO BLOCKERS, so "
                "its rate for this route says the pieces were drawn and not "
                "that the kill lands. Judge it in `simulate` against a pod.")
        else:
            notes.append(
                head + " `model_combat` is OFF, so this route's rate is purely "
                "\"the cards arrived\" — there is not even a modelled swing "
                "behind it, and the goldfish has no blockers either way. "
                "Zur declared exactly this and read 23.6% by t6; Forge returned "
                "0 of 39 games reaching 21 commander damage. Judge it in "
                "`simulate`.")
    return notes


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


def _validate_win_line_coverage(doc, slug, main_names, commander_names, base,
                                branch=None):
    """A card carrying two or more passing stacks belongs to some component."""
    declared = _declared_names(doc)
    deck_doc = None
    basics = set()
    try:
        deck_doc = load_deck_cards(slug, branch)
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


def validate(doc, slug, base, branch=None, notes=None):
    """Return a list of error strings (empty = the declaration holds).

    `notes` collects things the caller must SAY but that are not failures. The
    one that matters is a missing `cards.json`: every check below the shape pass
    needs the deck, and without it this used to `return errors` — an empty list,
    which `main` printed as a clean `OK`. A guard that cannot fail is a claim,
    not a guard, and this one made the claim in the validator's own voice.

    Caught live on 2026-09-04: `deck-branch new` writes `decklist.txt` and no
    `cards.json`, so a branch validated between `new` and `fetch-deck` reported
    OK on a declaration with two stranded card names. The goldfish found them a
    minute later, which is the only reason it was noticed at all.
    """
    errors = _validate_shape(doc)
    if errors:
        return errors                      # shape first; the rest would cascade

    try:
        deck_doc = load_deck_cards(slug, branch)
    except Exception as exc:
        # NOT an error — a fresh clone has no card data and reddening it would
        # teach the reader to ignore this gate. Said instead, every run.
        if notes is not None:
            notes.append(
                f"MEMBERSHIP AND WIN-LINE CHECKS DID NOT RUN — {type(exc).__name__}: "
                f"{exc}. Only the SHAPE of this file was checked. A stranded "
                f"card name, a group whose size overstates its redundancy, and "
                f"an undeclared win line would all pass unseen. Run "
                f"`manamap pilot fetch-deck {slug}"
                f"{' --branch ' + branch if branch else ''}` and validate again.")
        return errors
    cards = deck_doc.get("cards", [])
    main_names = {c["name"] for c in cards}
    commander_names = {c["name"] for c in cards if c.get("is_commander")}

    errors += _validate_shared_leg(doc)
    errors += _validate_membership(doc, main_names, commander_names)
    errors += _validate_win_line_coverage(doc, slug, main_names,
                                          commander_names, base, branch)
    return errors


def main(args):
    branch = getattr(args, "branch", None)
    base = deck_dir(args.slug, branch)
    path = base / "goldfish_targets.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — a deck with no declared targets has no engine "
            f"block. Author it before running `manamap pilot goldfish {args.slug}`.")
    with open(path) as f:
        doc = json.load(f)
    notes = []
    errors = validate(doc, args.slug, base, branch, notes=notes)
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
        if not derived:
            # THE FLAG OUTLIVED THE SCAFFOLD. No target is a role axis any
            # more, so somebody did the rewrite and left the marker — and the
            # old wording ("0 of 6 target(s) are role axes") read as an
            # accusation of the opposite. `validate-goldfish-targets` reports an
            # unedited draft on every run by design; saying it about an edited
            # one is the same failure one door over.
            scaffold_note = (
                "\n     The \"scaffolded\" flag outlived the scaffold — no "
                "target is a role axis any more, so this file WAS rewritten. "
                "Delete the flag; while it is set every run reports a draft.")
        else:
            scaffold_note = (
                f"\n     SCAFFOLD — never edited. {derived} of "
                f"{len(doc.get('targets', []))} target(s) are role axes rather "
                f"than this deck's components, and no win line is declared. "
                f"Rewrite the labels, regroup, then delete \"scaffolded\".")

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

    # THE HEADLINE WORD CARRIES HOW MUCH WAS ACTUALLY CHECKED. "OK" over a run
    # that only checked the file's shape is the validator asserting something it
    # did not look at, which is worse than saying nothing.
    headline = "OK  " if not notes else "PARTIAL"
    note_block = "".join("\n     " + n for n in notes)
    report_errors(
        path.name, errors,
        f"{headline} {path.name} — {len(doc.get('targets', []))} target(s), "
        f"{groups} component group(s); sizes are redundancy claims ◆"
        + note_block + required_note + scaffold_note
        + "".join(_combat_route_notes(doc)))


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-goldfish-targets <slug>`.")
