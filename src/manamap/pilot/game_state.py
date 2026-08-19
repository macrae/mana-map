"""Pilot: the `game_state` v2 vocabulary and its form check.

The spec is `docs/pilot.md` → *Game state v2*; this is the code that three consumers
share so they cannot drift: `validate_stack` (preflight + the citation loop),
`scenario_facts` (board_bodies / opponents_of over `seats[]`), and `sim/bridge.py`
(which writes v2 scenarios lifted from Forge games). v1 scenarios are untouched —
`is_v2(scenario)` is the only switch, and a v1 artifact never reaches this code.

Design rules carried from the spec: one `seat` object for every seat; phase and step
names are the Comprehensive Rules' own (CR 500.1); a hand is a list of names or
`{"unknown": n}`; a board entry is a v1 string or an object with at least `name`;
actions are a closed set and triggers are never actions.
"""

import re

PHASES = ("beginning", "precombat main", "combat", "postcombat main", "ending")
STEPS = {
    "beginning": ("untap", "upkeep", "draw"),
    "precombat main": (),
    "combat": ("beginning of combat", "declare attackers", "declare blockers",
               "combat damage", "end of combat"),
    "postcombat main": (),
    "ending": ("end", "cleanup"),
}
ACTION_KINDS = ("cast", "activate", "play_land", "attack", "block", "pass", "special")
COMMANDER_ZONES = ("battlefield", "command", "library", "graveyard", "exile", "hand", "stack")
SEAT_REQUIRED = ("seat", "life", "board")
_YOU = "you"
_ANNOT = re.compile(r"\s*[—(].*$")


def is_v2(scenario):
    return isinstance(scenario, dict) and scenario.get("version") == 2


def entry_name(entry):
    """The card a board entry names — v1 string (annotations stripped) or v2 object."""
    if isinstance(entry, dict):
        return str(entry.get("name") or "").strip()
    name = str(entry or "")
    return _ANNOT.sub("", name.split("—")[0]).strip()


def entry_is_token(entry):
    if isinstance(entry, dict):
        return bool(entry.get("token"))
    return "token" in str(entry).lower()


def entry_annotations(entry):
    if isinstance(entry, dict):
        return list(entry.get("annotations") or [])
    s = str(entry)
    return [s.split("—", 1)[1].strip()] if "—" in s else []


def our_seat(scenario):
    for s in scenario.get("seats") or []:
        if isinstance(s, dict) and s.get("seat") == _YOU:
            return s
    return None


def opponent_seats(scenario):
    return [s for s in (scenario.get("seats") or [])
            if isinstance(s, dict) and s.get("seat") != _YOU]


def validate_v2(scenario):
    """Form errors for a v2 scenario block (empty = holds). Semantics are the resolver's."""
    errors = []
    seats = scenario.get("seats")
    if not isinstance(seats, list) or len(seats) < 2:
        return ["v2: `seats` must be a list of at least two seat objects"]
    ids = []
    for i, s in enumerate(seats):
        if not isinstance(s, dict):
            errors.append(f"v2: seats[{i}] must be an object"); continue
        for k in SEAT_REQUIRED:
            if k not in s:
                errors.append(f"v2: seats[{i}] missing {k!r}")
        ids.append(s.get("seat"))
        if not isinstance(s.get("board", []), list):
            errors.append(f"v2: seats[{i}].board must be a list")
        hand = s.get("hand")
        if hand is not None and not (isinstance(hand, list) or
                                     (isinstance(hand, dict) and ("unknown" in hand or "known" in hand))):
            errors.append(f"v2: seats[{i}].hand must be a list of names or {{unknown: n}}")
        cmd = s.get("commander")
        if isinstance(cmd, dict) and cmd.get("zone") not in (None, *COMMANDER_ZONES):
            errors.append(f"v2: seats[{i}].commander.zone {cmd.get('zone')!r} not in {COMMANDER_ZONES}")
    if _YOU not in ids:
        errors.append('v2: one seat must be "you"')
    if len(set(ids)) != len(ids):
        errors.append(f"v2: seat ids repeat: {ids}")
    active = scenario.get("active_seat")
    if active is not None and active not in ids:
        errors.append(f"v2: active_seat {active!r} is not a seat")
    prio = scenario.get("priority")
    if prio is not None and prio not in ids:
        errors.append(f"v2: priority {prio!r} is not a seat")
    phase = scenario.get("phase")
    if phase is not None and phase not in PHASES:
        errors.append(f"v2: phase {phase!r} not in {PHASES} (CR 500.1 names)")
    step = scenario.get("step")
    if step is not None:
        allowed = STEPS.get(phase, ()) if phase else sum(STEPS.values(), ())
        if step not in allowed:
            errors.append(f"v2: step {step!r} is not a step of phase {phase!r} ({allowed})")
    turn = scenario.get("turn")
    if turn is not None and (not isinstance(turn, int) or turn < 1):
        errors.append("v2: turn must be a positive integer")
    actions = scenario.get("actions") or []
    if not isinstance(actions, list):
        errors.append("v2: actions must be a list")
    else:
        for i, a in enumerate(actions):
            if not isinstance(a, dict) or a.get("kind") not in ACTION_KINDS:
                errors.append(f"v2: actions[{i}].kind must be one of {ACTION_KINDS}")
            elif a.get("seat") not in ids:
                errors.append(f"v2: actions[{i}].seat {a.get('seat')!r} is not a seat")
    stack = scenario.get("stack")
    if not isinstance(stack, list):
        errors.append("v2: stack must be a list (pos 0 = bottom; [] when empty)")
    if not (stack or actions):
        errors.append("v2: nothing to resolve — give a non-empty `stack` or `actions`")
    return errors


def our_named_cards(scenario):
    """Card names on YOUR board and in your hand (if known), tokens excluded —
    the set `unknown_cards` checks against cards.json."""
    me = our_seat(scenario) or {}
    names = [entry_name(e) for e in me.get("board") or [] if not entry_is_token(e)]
    hand = me.get("hand")
    if isinstance(hand, list):
        names += [str(h) for h in hand]
    elif isinstance(hand, dict):
        names += [str(h) for h in hand.get("known") or []]
    return [n for n in names if n]
