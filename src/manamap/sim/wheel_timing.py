"""WHEN THE WHEELS FIRED, AND ONTO WHAT — from a run's logs, our seat only (2026-10-21).

The pilot's thesis on sharknado (captain's log 002): a wheel is a FINISHER dropped onto
Brallin or Shabraz, never early tempo, and the deck runs too many of them. The goldfish
measures that under a declared policy; he does not trust the goldfish. Forge's AI fires
wheels on its own rule, so at a table the question becomes an observation about the
LIST — do the commanders land soon enough that the wheels the AI fires land on them —
and this module is that observation, as counts a reader can check against the log.

Per game, our seat:
    first_wheel_turn        own turn the first wheel resolved (None = never)
    wheels_cast             wheel spells cast (activated wheels are not casts and are
                            not counted here — `engine_casts` has the activations)
    wheels_with_commander   of those, cast on an own turn whose CLEANUP snapshot shows a
                            commander on the battlefield (`board_series.series`, both
                            partners): "the wheel landed and a commander was there to be
                            paid" is a floor, since a commander cast and removed inside
                            the same turn reads as absent and one cast AFTER the wheel
                            reads as present — the log has no mid-turn board

THE KEPT HAND IS NOT HERE. The telemetry's zone lines begin with turn 1's draw — the
opening deal and the mulligans are not in the log (checked on sharknado's V6 logs
2026-10-21: the first `Hand` arrival of every seat is a turn-1 `Library` draw) — so the
seven the AI kept is unreadable from a run. The pilot's marks (`kept_wheels`,
`kept_bodies`) and the goldfish's keep-rule row carry that question.

`from_logs` returns None when the deck's list holds no card the readers call a wheel:
absent means absent, and a block of zeros on a deck with no wheels would read as "never
wheels". Validated on form only where present (`validate`), like `board_series`.
"""

from manamap.sim import board_series, parse as sim_parse, stats
from manamap.sim.parse import mean_ci

SINCE = "2026-10-21"

LIMITS = [
    "A wheel is a SPELL the goldfish's readers call one (wheel_draws; a split card "
    "by the face that wheels); a discard wheel those readers cannot parse is not "
    "counted. Activated wheels (Kayla, Magus, Archivist) are not casts and are in "
    "engine_casts.",
    "`wheels_with_commander` reads the CLEANUP board of the cast's own turn: a commander "
    "cast after the wheel on the same turn reads as present, one removed on the same "
    "turn reads as absent — the log has no mid-turn board. A floor, labelled one.",
    "The kept seven is NOT in the log: the telemetry's zone lines start with turn 1's "
    "draw. The pilot's marks carry the keep question.",
    "Forge's AI fires a wheel on its own rule, so these are counts about the LIST at "
    "this table, never about the pilot's hold.",
]


def wheel_names(deck_doc):
    """The names the goldfish's own readers call a wheel SPELL in a deck's cards.json,
    so the two instruments cannot disagree about what a wheel is. An activated wheel
    (Queen Kayla, Magus, Jace's Archivist) is excluded: casting the creature is not
    the wheel, and the eight-game test read Kayla's turn-3 cast as a wheel with no
    commander out. A SPLIT CARD counts
    only the face whose own text wheels: Forge logs `cast Memory` and `cast Commit`
    as two names, and Commit — a bounce — was read as a wheel on 2026-10-21's
    eight-game test until this read the faces apart."""
    from manamap.pilot.goldfish_library import classify
    from manamap.pilot.goldfish_profiles import _WHEEL_RE
    out = set()
    for card in (deck_doc or {}).get("cards") or []:
        try:
            c = classify(card)
            if c["is_land"] or not c["draw"]["wheel_draws"]:
                continue                         # an activated wheel is not a CAST wheel
        except Exception:                        # pragma: no cover - a malformed entry
            continue
        name = card["name"]
        faces = card.get("card_faces") or []
        if " // " in name and len(faces) >= 2:
            for face in faces:
                if _WHEEL_RE.search(face.get("oracle_text") or ""):
                    out.add(str(face.get("name") or "").split(" // ")[0].strip())
            if not any(_WHEEL_RE.search(f.get("oracle_text") or "") for f in faces):
                out.add(name.split(" // ")[0].strip())
        else:
            out.add(name.split(" // ")[0].strip())
    return out


def _clean(name):
    return sim_parse._CARD_ID_TAIL.sub("", name or "").split(" (")[0].strip()


def per_game(game, commanders, label, wheels):
    """One row for our seat in one parsed game."""
    rows = board_series.series(game, commanders).get(label) or []
    own = {r["global_turn"]: r for r in rows}
    own_turn = {r["global_turn"]: r["own_turn"] for r in rows}
    cast_turns = [ev["turn"] for ev in game["events"]
                  if ev.get("kind") == "cast" and ev.get("seat") == label
                  and _clean(ev.get("what")) in wheels]
    with_cmd = sum(1 for t in cast_turns if own.get(t, {}).get("commander_on_battlefield"))
    first = min(cast_turns) if cast_turns else None
    return {"first_wheel_turn": own_turn.get(first) if first is not None else None,
            "wheels_cast": len(cast_turns),
            "wheels_with_commander": with_cmd}


def from_logs(log_texts, commanders_by_label, ours, deck_doc):
    """The record block for our seat, or None when the list holds no wheel."""
    wheels = wheel_names(deck_doc)
    if not wheels:
        return None
    cmd = {k: (sorted(x.split(" // ")[0].strip() for x in v) if isinstance(v, (set, list, tuple))
               else v) for k, v in (commanders_by_label or {}).items()}
    games = []
    for text in log_texts:
        for game in sim_parse.parse_games(text):
            label = next((s for s in game["seats"] if s.endswith(f"-{ours}")), None)
            if label is None:
                continue
            games.append(per_game(game, cmd, label, wheels))
    return aggregate(games, ours, sorted(wheels))


def _rate(k, n):
    if not n:
        return {"k": k, "n": n, "rate": None, "ci95": None}
    lo, hi = stats.wilson_bounds(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def aggregate(games, ours, names):
    firsts = [g["first_wheel_turn"] for g in games if g["first_wheel_turn"] is not None]
    cast = sum(g["wheels_cast"] for g in games)
    paid = sum(g["wheels_with_commander"] for g in games)
    return {"seat": ours, "games": len(games), "since": SINCE,
            "wheel_names": names,
            "first_wheel_turn": {"mean_ci": mean_ci(firsts) if len(firsts) >= 2 else None,
                                 "games_with_a_wheel": len(firsts),
                                 "never": len(games) - len(firsts)},
            "wheels_cast": {"mean_ci": mean_ci([g["wheels_cast"] for g in games])
                            if len(games) >= 2 else None, "total": cast},
            "wheels_with_commander": _rate(paid, cast),
            "per_game": games,
            "limits": LIMITS}


def validate(block, games_completed):
    """Form only; the logs re-derive it where they exist."""
    want = {"seat", "games", "since", "wheel_names", "first_wheel_turn", "wheels_cast",
            "wheels_with_commander", "per_game", "limits"}
    if not isinstance(block, dict) or set(block) != want:
        return ["wheel_timing: wrong shape"]
    errors = []
    if block["games"] != games_completed:
        errors.append(f"wheel_timing.games {block['games']} != games_completed {games_completed}")
    if len(block["per_game"]) != block["games"]:
        errors.append("wheel_timing.per_game does not hold one row per game")
    for g in block["per_game"]:
        if g["wheels_with_commander"] > g["wheels_cast"]:
            errors.append("wheel_timing: more wheels with a commander than wheels cast")
            break
    w = block["wheels_with_commander"]
    if w["n"] != block["wheels_cast"]["total"]:
        errors.append("wheel_timing: the with-commander denominator is not the wheels cast")
    return errors
