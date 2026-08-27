"""What your pod actually does on its turns, measured from the runs you have.

WHY THIS IS NOT A GOLDFISH FEATURE. A solitaire model has no opponents by
construction, so a card that taxes what THEY do is worth exactly zero in it —
that is a boundary, not a gap, and building an opponent inside the goldfish would
make it a worse Forge. But the question a pilot actually asks about such a card
is narrower and IS answerable: **how often would this have triggered against my
table?** Forge already played hundreds of those turns and kept the record.

So this reads the tracked sim runs rather than inventing behaviour, and it is a
frequency estimate rather than a value: it says a trigger fires about N times a
round against THIS pod, and leaves what that is worth to the pilot.

WHAT IT CHANGED. Reading the cards alone, "whenever an opponent draws their
second card each turn" (2 mana) looked like better value than "whenever an
opponent draws a card" (4 mana). Measured across 1,143 opponent-games: the pod
casts **1.09 spells per turn** and reaches a SECOND spell on only **23.2%** of
them — so a second-thing trigger fires about **0.7 times a round** against three
opponents, where a per-draw trigger fires **3.0** by rule. Four times apart, and
the cheap card is the weak one. No amount of reading the cards gets there.
"""

import collections
import glob
import json
import re

#: A draw step is one card per turn, by rule. Nothing to measure.
DRAWS_PER_OPPONENT_TURN = 1.0

#: Commander's default table minus you.
DEFAULT_OPPONENTS = 3

_CAST = re.compile(r"^Add To Stack: (Ai\(\d\)-mm-[\w@-]+) cast ")


def observed(logs=None):
    """Opponent behaviour per own turn, from real games.

    Reads LOGS where they exist because a per-turn rate is not in the record —
    the record aggregates per game. Returns None where they do not, and the
    caller falls back to the measured constants below, which came from this.
    """
    paths = logs if logs is not None else glob.glob(
        "data/decks/**/sim/logs/*/part-*.log", recursive=True)
    if not paths:
        return None
    turns, second, casts = (collections.Counter() for _ in range(3))
    for path in paths:
        cur, n = None, 0
        try:
            fh = open(path, errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                if line.startswith("Turn: "):
                    if cur:
                        turns[cur] += 1
                        casts[cur] += n
                        if n >= 2:
                            second[cur] += 1
                    who = re.search(r"(Ai\(\d\)-mm-[\w@-]+)", line)
                    cur, n = (who.group(1) if who else None), 0
                    continue
                m = _CAST.match(line)
                if m and cur and m.group(1) == cur:
                    n += 1
    if not turns:
        return None
    total_t = sum(turns.values())
    return {"turns": total_t,
            "casts_per_turn": round(sum(casts.values()) / total_t, 3),
            "second_spell_rate": round(sum(second.values()) / total_t, 3)}

#: MEASURED 2026-08-26 across 1,143 opponent-games of tracked Forge play. Used
#: when the logs are absent — they are gitignored and only exist where the run
#: was made, and a checkout still deserves the estimate.
POD = {"casts_per_turn": 1.09, "second_spell_rate": 0.232, "n_seat_games": 1143}

#: A second DRAW in a turn needs a draw spell, and Forge does not log draws. This
#: is bounded rather than measured: it cannot exceed the rate of casting a second
#: spell at all, and most second spells are not draw spells. Reported AS a bound,
#: never as a figure — the honest thing a missing measurement can say.
SECOND_DRAW_BOUND = "at most the second-spell rate, and in practice well below it"

TRIGGERS = (
    ("per_opponent_draw", re.compile(
        r"whenever an opponent draws a card|whenever a player draws a card", re.I),
     lambda pod, n: n * DRAWS_PER_OPPONENT_TURN),
    ("second_draw", re.compile(r"draws their second card", re.I), None),
    ("second_spell", re.compile(
        r"casts their second spell|second spell each turn", re.I),
     lambda pod, n: n * pod["second_spell_rate"]),
    ("per_opponent_cast", re.compile(r"whenever an opponent casts", re.I),
     lambda pod, n: n * pod["casts_per_turn"]),
    ("each_upkeep", re.compile(r"at the beginning of your upkeep", re.I),
     lambda pod, n: 1.0),
)


def rate_for(text, opponents=DEFAULT_OPPONENTS, pod=None):
    """Estimated triggers per round against this pod, or a stated bound."""
    pod = pod or POD
    for name, pat, fn in TRIGGERS:
        if pat.search(text or ""):
            if fn is None:
                return {"pattern": name, "per_round": None,
                        "bound": SECOND_DRAW_BOUND,
                        "basis": f"{opponents} opponents; Forge does not log draws"}
            return {"pattern": name, "per_round": round(fn(pod, opponents), 2),
                    "basis": (f"{opponents} opponents at "
                              f"{pod['casts_per_turn']} spells/turn, a second on "
                              f"{pod['second_spell_rate']:.1%} of turns "
                              f"(measured, n={pod.get('n_seat_games', '?')})")}
    return None
