"""A BOARD SERIES FROM THE LOG THE PARSER SAID COULD NOT CARRY ONE.

`parse.py` records, correctly, that Forge logs a permanent LEAVING the
battlefield and never one arriving — 0 `to Battlefield` lines in a 100-game
run — and concluded that a board count "would be a series of zeros wearing the
name of a measurement". But `bridge.reconstruct` has been building exactly
that board for `sim-scenario` since S4: a cast permanent enters on its
`Resolve Stack` line (`X - Creature P / T`, a bare `X`, an aura's `Attach to`),
a token at the resolution that creates it, a land on its `Land:` line, and each
leaves on its zone change. What the bridge builds at ONE cut, this builds at
EVERY cut — the start of each turn's cleanup step, which is the PRD's "end of
each turn" — and reports as what it is: an ESTIMATE, with every floor named.

WHAT IS A FLOOR AND WHY. A token whose count the resolution states as X or
"that many" is counted in `token_counts_unread` and not on the board; a
creature that entered without being cast (reanimated, flashed in by an
ability, put onto the battlefield by another card) is seen only when it first
acts, so a body that sat is invisible; a `*/*` creature is a body whose power
is unknown and is counted in `power_unknown`, never as 0; counters, anthems,
auras and equipment do not change a printed power; removal by name where two
seats hold the card takes the first; the hand is never logged. So `bodies` is
a floor on creatures, `printed_power` a floor on power, and `open_lands` is
lands untapped at the cleanup snapshot — mana left unused that turn.

Our seat only, in the record, for size; the per-game rows carry every turn so
a finder (B2) can ask "how often did the board look like THIS".
"""

from manamap.sim import parse as sim_parse
from manamap.sim.bridge import reconstruct
from manamap.sim.parse import mean_ci

#: Records made before this block existed carry no `board_series`; a catalog
#: entry that names the block is checked from this date on.
SINCE = "2026-09-30"

#: Own turns reported in the aggregate. Past this the sample thins to nothing
#: on most tables; the per-game rows carry every turn regardless.
MAX_OWN_TURN = 16

LIMITS = [
    "AN ESTIMATE FROM RESOLVE LINES, NOT AN ARRIVAL LOG. Forge never logs a "
    "permanent entering the battlefield; a cast permanent is placed on its "
    "resolution line, a token at the resolution that creates it, a land on its "
    "`Land:` line. A body that entered WITHOUT being cast (reanimated, flashed "
    "in by an ability, put onto the battlefield by another card) is seen only "
    "when it first attacks, blocks or deals damage; one that sat is invisible.",
    "A token whose count the resolution states as X or 'that many' is not on "
    "the board; `token_counts_unread` says how many resolutions were unread.",
    "`printed_power` is PRINTED: counters, anthems, auras and equipment are "
    "invisible to the log. A `*/*` creature is a body in `bodies` and counted "
    "in `power_unknown`, never as 0 power.",
    "`open_lands` is lands untapped at the start of the cleanup step — mana "
    "left unused that turn. `rocks_tapped` is nonland permanents tapped for "
    "mana this turn, a FLOOR on mana sources: a rock never tapped is invisible.",
    "Removal by name where more than one seat controls the card takes the first "
    "seat; the bridge's reconstruction notes record each such case.",
    "Summoning sickness, vigilance and the hand are not in the log.",
]


def snapshot(st):
    """The figures for one seat from one reconstructed state. Pure, so a test
    can hand it a state and check every rule."""
    bodies = power = unknown = 0
    for p in list(st["perms"].values()) + list(st["tokens"].values()):
        pt = p.get("pt")
        if not pt:
            continue                     # a Treasure token, a rock, an aura: not a body
        bodies += 1
        pw = str(pt).split("/")[0].strip()
        if pw.isdigit():
            power += int(pw)
        else:
            unknown += 1
    lands = st["lands"]
    return {"permanents": len(st["perms"]) + len(st["tokens"]) + len(lands),
            "bodies": bodies, "printed_power": power, "power_unknown": unknown,
            "tokens": len(st["tokens"]),
            "lands": len(lands),
            "open_lands": sum(1 for l in lands.values() if not l.get("tapped")),
            "rocks_tapped": sum(1 for p in st["perms"].values() if p.get("tapped") and not p.get("pt")),
            "commander_on_battlefield": st.get("commander_zone") == "battlefield",
            "token_counts_unread": st.get("token_counts_unread", 0)}


def own_turns(game):
    """{seat: [global turns on which that seat took its untap step]}."""
    out = {s: [] for s in game["seats"]}
    for ev in game["events"]:
        if ev.get("kind") == "phase" and ev.get("text") == "Untap step" and ev.get("seat") in out:
            t = ev["turn"]
            if not out[ev["seat"]] or out[ev["seat"]][-1] != t:
                out[ev["seat"]].append(t)
    return out


def series(game, commanders):
    """Per seat, per OWN turn: the board at the start of that turn's cleanup.

    One `reconstruct` per global turn — measured at ~0.4 ms a cut, so a
    100-game run costs about a second and a half. Returns
    `{seat: [{"own_turn": k, "global_turn": G, **snapshot}, ...]}`.
    """
    turns = own_turns(game)
    max_turn = max((ev["turn"] for ev in game["events"]), default=0)
    boards = {}
    for g in range(1, max_turn + 1):
        states, _notes, _a, _p, _s = reconstruct(game, g, "ending", "cleanup", commanders)
        boards[g] = {s: snapshot(st) for s, st in states.items()}
    out = {}
    for seat, gts in turns.items():
        out[seat] = [{"own_turn": k + 1, "global_turn": g, **boards[g][seat]}
                     for k, g in enumerate(gts) if g in boards and seat in boards[g]]
    return out


def commander_uptime(rows):
    """Share of a seat's own turns AFTER the first one with the commander on
    the battlefield, on which it is still there — a per-game figure; None
    when the commander never resolved in that game."""
    first = next((i for i, r in enumerate(rows) if r["commander_on_battlefield"]), None)
    if first is None:
        return None
    after = rows[first:]
    return sum(1 for r in after if r["commander_on_battlefield"]) / len(after)


def from_logs(log_texts, commanders_by_label, ours):
    """The record block for our seat from a run's logs.

    `commanders_by_label` is the map `forge.run` builds — every seat index for
    every deck, a SET of names per label; `reconstruct` wants one name per
    label, so the first is taken (a partner pair is two names, and the bridge
    already reads the commander as one card).
    """
    # ONE NAME PER LABEL, AND THE FRONT FACE. `record_commanders` carries a
    # double-faced commander as `Front // Back`, and Forge's resolve line
    # prints the front face — so heliod's uptime read None on every run until
    # the name was split. A partner pair is two names; the bridge reads the
    # commander as one card and the first is taken, which `limits` should say
    # if a partner deck ever joins the fleet.
    def _front(v):
        if isinstance(v, (set, list, tuple)):
            v = sorted(v)[0] if v else None
        return v.split(" // ")[0].strip() if isinstance(v, str) else v
    cmd = {k: _front(v) for k, v in (commanders_by_label or {}).items()}
    per_game = []
    for text in log_texts:
        for game in sim_parse.parse_games(text):
            ser = series(game, cmd)
            mine = next((s for s in ser if s.endswith(f"-{ours}")), None)
            if mine is None:
                continue
            rows = ser[mine]
            per_game.append({"seat": mine, "turns": len(rows),
                             "commander_uptime": commander_uptime(rows),
                             "by_turn": {r["own_turn"]: {k: r[k] for k in
                                         ("bodies", "printed_power", "power_unknown", "permanents",
                                          "tokens", "lands", "open_lands", "rocks_tapped",
                                          "commander_on_battlefield")}
                                         for r in rows}})
    return aggregate(per_game, ours)


def aggregate(per_game, ours):
    by_turn = {}
    for key in ("bodies", "printed_power", "permanents", "tokens", "lands", "open_lands"):
        by_turn[key] = {}
        for t in range(1, MAX_OWN_TURN + 1):
            vals = [g["by_turn"][t][key] for g in per_game if t in g["by_turn"]]
            if len(vals) >= 2:
                by_turn[key][str(t)] = mean_ci(vals)
    up = [g["commander_uptime"] for g in per_game if g["commander_uptime"] is not None]
    return {"seat": ours, "games": len(per_game), "since": SINCE,
            "by_turn": by_turn,
            "commander_uptime": ({"mean_ci": mean_ci(up), "games_with_commander": len(up),
                                  "basis": "per game: own turns after the first on which the "
                                           "commander was on the battlefield, share still there; "
                                           "mean over games where it resolved at all"}
                                 if len(up) >= 2 else
                                 {"mean_ci": None, "games_with_commander": len(up),
                                  "basis": "fewer than two games saw the commander resolve"}),
            "per_game": [{"turns": g["turns"], "commander_uptime": g["commander_uptime"],
                          "by_turn": {str(t): v for t, v in g["by_turn"].items()}}
                         for g in per_game],
            "limits": LIMITS}


def validate(block, games_completed):
    """Form only; the logs re-derive it where they exist."""
    errors = []
    want = {"seat", "games", "since", "by_turn", "commander_uptime", "per_game", "limits"}
    if not isinstance(block, dict) or set(block) != want:
        return ["board_series: wrong shape"]
    if block["games"] != games_completed:
        errors.append(f"board_series.games {block['games']} != games_completed {games_completed}")
    if len(block["per_game"]) != block["games"]:
        errors.append("board_series.per_game does not hold one row per game")
    for g in block["per_game"]:
        for t, row in (g.get("by_turn") or {}).items():
            if row["bodies"] > row["permanents"]:
                errors.append(f"board_series: bodies {row['bodies']} exceed permanents {row['permanents']} at own turn {t}")
                break
        u = g.get("commander_uptime")
        if u is not None and not 0 <= u <= 1:
            errors.append(f"board_series: commander_uptime {u} outside [0, 1]")
    return errors
