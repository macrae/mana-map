"""`manamap pilot scenario-ab --spec FILE [--check]` — a scenario slice A/B from one spec.

The command Jarvis runs after Sean has OKed the board (PRD v2 Step 5; the
`scenario-sim` agent writes the spec, it never runs this itself). The spec:

    {"seats":   {"you": "sharknado", "opp1": "giada-angels"},     # who plays each seat
     "board":   {game_state v2 board, `you` among its seats},
     "arms":    {"greaves": [],                                     # A: the board as given
                 "signet":  [{"seat": "you", "zone": "hand",
                              "out": "Lightning Greaves", "in": "Arcane Signet"}]},
     "primary": "commander_out",                                    # named BEFORE the run
     "seeds":   10, "rounds": 1,
     "question": "Does Greaves keep Brallin on the board better than a Signet?"}

An arm is a list of EDITS to the board (`zone` is hand, board, graveyard or exile; `out`
and/or `in`) or a whole v2 board. Edits are what an agent should write: a full second
board is a second chance to get the first one wrong.

`--check` converts both arms for the first seed and prints the board per seat, with every
note, WITHOUT playing — the structured board Sean OKs before any run (his call,
2026-10-07: every time). Without it the A/B runs and prints the answer line and the table.
"""

import copy
import json

from manamap.pilot import game_state as gs

ZONES = {"hand": "hand", "board": "board", "graveyard": "graveyard", "exile": "exile"}
DEFAULT_SEEDS = 10
MAX_SEEDS = 40


def _seat(board, sid):
    for s in board.get("seats") or []:
        if s.get("seat") == sid:
            return s
    raise SystemExit(f"arm edit names seat {sid!r}, which the board does not have")


def apply_edits(board, edits):
    """A copy of the board with the arm's edits made. Refuses an `out` that is not there."""
    out = copy.deepcopy(board)
    for e in edits:
        zone = ZONES.get(e.get("zone"))
        if not zone:
            raise SystemExit(f"arm edit zone {e.get('zone')!r} is not one of {', '.join(ZONES)}")
        seat = _seat(out, e.get("seat", "you"))
        cards = seat.get(zone)
        if zone == "hand" and isinstance(cards, dict):
            cards = cards.setdefault("known", [])
        elif cards is None:
            cards = seat.setdefault(zone, [])
        if e.get("out"):
            names = [gs.entry_name(c) for c in cards]
            if e["out"] not in names:
                raise SystemExit(f"arm edit: {e['out']!r} is not in {seat['seat']}'s {zone}")
            cards.pop(names.index(e["out"]))
        if e.get("in"):
            cards.append(e["in"])
    return out


_MARKS = {"IsCommander": "commander", "Tapped": "tapped", "SummonSick": "summoning sick",
          "FaceDown": "face down", "Transformed": "transformed"}


def _plain(spec):
    """`Brallin|Tapped|IsCommander` -> `Brallin (tapped, commander)`: Forge's markers
    in words, for the board Sean reads."""
    name, *mods = spec.split("|")
    words = [_MARKS.get(m, m.replace("Counters:", "counters ")) for m in mods]
    return name + (f" ({', '.join(words)})" if words else "")


def load_spec(path):
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    for k in ("seats", "board", "arms", "primary"):
        if k not in spec:
            raise SystemExit(f"the spec needs `{k}`")
    if len(spec["arms"]) != 2:
        raise SystemExit("the spec needs exactly two arms")
    board = spec["board"]
    board.setdefault("version", 2)
    board.setdefault("stack", [])
    board.setdefault("actions", [])
    arms = {k: (v if isinstance(v, dict) else apply_edits(board, v)) for k, v in spec["arms"].items()}
    seeds = int(spec.get("seeds") or DEFAULT_SEEDS)
    if not 2 <= seeds <= MAX_SEEDS:
        raise SystemExit(f"seeds must be 2..{MAX_SEEDS} (a paired interval needs two)")
    return spec, board, arms, seeds


def check_text(spec, arms):
    """What Sean OKs: each arm's board per seat as Forge will get it, first seed."""
    from manamap.sim import slice_state as ss

    out = []
    if spec.get("question"):
        out.append(f"QUESTION  {spec['question']}")
    b = spec["board"]
    out.append(f"turn {b.get('turn')} · {b.get('phase')}{' / ' + b['step'] if b.get('step') else ''} · "
               f"active {b.get('active_seat', 'you')} · {int(spec.get('rounds') or 1)} round(s) after "
               f"this turn · primary {spec['primary']} · {int(spec.get('seeds') or DEFAULT_SEEDS)} seeds")
    for label, board in arms.items():
        text, notes = ss.to_forge_state(board, spec["seats"], 1)
        kv = dict(line.split("=", 1) for line in text.strip().splitlines())
        out.append(f"\n{label}:")
        seats = [gs.our_seat(board)] + gs.opponent_seats(board)
        for i, s in enumerate(seats):
            p = f"p{i}"
            def show(z, kv=kv, p=p):
                v = kv.get(f"{p}{z}", "")
                return ", ".join(_plain(x) for x in v.split(";")) if v else "—"
            lib = kv.get(f"{p}library", "")
            out.append(f"  {s['seat']} ({spec['seats'][s['seat']]}) life {kv.get(p + 'life')}")
            out.append(f"    battlefield: {show('battlefield')}")
            out.append(f"    hand:        {show('hand')}")
            if kv.get(f"{p}graveyard"):
                out.append(f"    graveyard:   {show('graveyard')}")
            if kv.get(f"{p}command"):
                out.append(f"    command:     {show('command')}")
            out.append(f"    library:     {len(lib.split(';')) if lib else 0} cards, dealt from the list per seed")
        for n in notes:
            out.append(f"  note: {n}")
    return "\n".join(out)


def main(args):
    from manamap.sim import slice_ab as ab

    spec, board, arms, seeds = load_spec(args.spec)
    if getattr(args, "check", False):
        print(check_text(spec, arms))
        print("\nnot played — OK the board, then run without --check")
        return
    rep = ab.compare(board, spec["seats"], arms, list(range(1, seeds + 1)), spec["primary"],
                     rounds=int(spec.get("rounds") or 1))
    rep["question"] = spec.get("question")
    if getattr(args, "as_json", False):
        print(json.dumps(rep, indent=1, default=str))
    else:
        if spec.get("question"):
            print(f"QUESTION  {spec['question']}\n")
        print(ab.format_report(rep))
