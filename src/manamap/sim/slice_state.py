"""`game_state` v2 -> Forge's puzzle `[state]` text, for a scenario slice.

The reverse of `bridge.lift`, and the step that makes a board a slice can play. The rules
are Sean's (2026-10-07, PRD v2 Step 5 scoping):

  * **Seats.** `you` is p0; the opponents follow in the order the scenario lists them.
    The caller names the deck each seat plays (`seat_slugs`), because a slice is played
    by AI decks and the hidden cards come from them.
  * **Hidden cards come from the seat's own decklist, dealt per seed** by a KEYED
    shuffle (`keyed_order`), so two arms that differ by one card deal alike. A hand given
    as names is used as given; `{"unknown": n}` (or `{"known": [...], "unknown": n}`)
    draws n from what the deck has left; a library is everything left, shuffled, with
    any named `library.top` cards first (the override). "What is left" removes, copy by
    copy, every card the seat already has somewhere on the board.
  * **Nothing is silently dropped.** Every card name is checked against the corpus
    BEFORE Forge sees it — Forge skips a name it does not know with one stderr line and
    plays on, which would answer a question about a different board. A token without a
    `forge_token` script, a non-empty stack, an unread field: each is a NOTE the caller
    must show, never a quiet omission.

`to_forge_state(scenario, seat_slugs, seed)` -> `(text, notes)`.
"""

import hashlib

from manamap.pilot import game_state as gs

#: CR 500.1 phase/step names -> Forge's PhaseType names.
PHASE = {
    ("beginning", "untap"): "UNTAP", ("beginning", "upkeep"): "UPKEEP",
    ("beginning", "draw"): "DRAW", ("beginning", None): "UPKEEP",
    ("precombat main", None): "MAIN1",
    ("combat", "beginning of combat"): "COMBAT_BEGIN", ("combat", None): "COMBAT_BEGIN",
    ("combat", "declare attackers"): "COMBAT_DECLARE_ATTACKERS",
    ("combat", "declare blockers"): "COMBAT_DECLARE_BLOCKERS",
    ("combat", "combat damage"): "COMBAT_DAMAGE",
    ("combat", "end of combat"): "COMBAT_END",
    ("postcombat main", None): "MAIN2",
    ("ending", "end"): "END_OF_TURN", ("ending", None): "END_OF_TURN",
    ("ending", "cleanup"): "CLEANUP",
}
#: Forge applies a board mid-combat only between two players (GameState.handleCombat).
COMBAT_1V1 = {"COMBAT_DECLARE_ATTACKERS", "COMBAT_DECLARE_BLOCKERS"}
DEFAULT_LIFE = 40


def keyed_order(cards, seed):
    """The pile in a seed's order, where a card's place depends only on the seed, its
    name and which copy it is — NOT on what else is in the pile. So two arms whose piles
    differ by the card being tested deal the same order for everything else: seed k of
    arm A and seed k of arm B are the same draws. A plain shuffle of two different lists
    deals two unrelated libraries and throws the pairing away."""
    seen, keyed = {}, []
    for c in cards:
        k = seen.get(c, 0)
        seen[c] = k + 1
        keyed.append((hashlib.sha256(f"{seed}:{c}:{k}".encode()).hexdigest(), c))
    return [c for _, c in sorted(keyed)]


class StateError(ValueError):
    """The board cannot be played as given; the message says what to fix."""


def _deck_copies(slug):
    """Every copy in the seat's list, commanders included, by name."""
    from manamap.pilot.common import expand_copies
    from manamap.pilot.fetch_deck import parse_decklist
    from manamap.sim import forge

    text = (forge.seat_dir(slug) / "decklist.txt").read_text(encoding="utf-8")
    entries = parse_decklist(text)
    commanders = [e["name"] for e in entries if e.get("is_commander")]
    return [e["name"] for e in expand_copies(entries)], commanders


def _front(spec):
    """The FRONT face, always. Measured 2026-10-08: Forge's state loader creates a split
    card from "Commit // Memory" OR "Commit", but a modal double-faced card only from its
    front face — "Hengegate Pathway // Mistgate Pathway" was skipped with one stderr line
    and the board played on without it. The front face works for every shape."""
    head, sep, mods = spec.partition("|")
    return head.split(" // ")[0] + (sep + mods if sep else "")


def _entry(e, notes, seat_id):
    """One board entry -> a Forge card spec, or None (with a note) when it cannot be."""
    name = gs.entry_name(e)
    if gs.entry_is_token(e):
        script = e.get("forge_token") if isinstance(e, dict) else None
        if not script:
            notes.append(f"{seat_id}: token {name!r} left off — give it a `forge_token` "
                         f"script name to include it")
            return None, None
        spec = f"t:{script}"
    else:
        spec = name
    if isinstance(e, dict):
        if e.get("tapped"):
            spec += "|Tapped"
        if e.get("summoning_sick"):
            spec += "|SummonSick"
        counters = e.get("counters") or {}
        if counters:
            spec += "|Counters:" + ",".join(f"{k}={int(v)}" for k, v in counters.items())
        if e.get("face_down"):
            spec += "|FaceDown"
        if e.get("transformed"):
            spec += "|Transformed"
    return spec, (None if gs.entry_is_token(e) else name)


def to_forge_state(scenario, seat_slugs, seed, known_names=None):
    """The `[state]` text for one replicate, and the notes the caller must show.

    `seat_slugs` maps each seat id to the deck slug that plays it. `known_names` is the
    corpus name set (`card_pool.corpus_names()`), injectable for tests."""
    if not gs.is_v2(scenario):
        raise StateError("a slice needs a game_state v2 board (`version: 2`)")
    errors = [e for e in gs.validate_v2(dict(scenario, stack=scenario.get("stack") or [],
                                              actions=scenario.get("actions") or [{"kind": "pass",
                                              "seat": "you"}]))]
    if errors:
        raise StateError("; ".join(errors))
    seats = [gs.our_seat(scenario)] + gs.opponent_seats(scenario)
    missing = [s["seat"] for s in seats if s["seat"] not in seat_slugs]
    if missing:
        raise StateError(f"no deck named for seat(s) {missing} — every seat is played by a deck")
    if known_names is None:
        from manamap.pilot.card_pool import corpus_names
        known_names = corpus_names() or set()

    notes, lines, unknown = [], [], []
    pid = {s["seat"]: f"p{i}" for i, s in enumerate(seats)}

    phase = PHASE.get((scenario.get("phase") or "precombat main", scenario.get("step")))
    if phase is None:
        raise StateError(f"no Forge phase for {scenario.get('phase')!r} / {scenario.get('step')!r}")
    if phase in COMBAT_1V1 and len(seats) != 2:
        raise StateError("a board mid-combat (attackers or blockers declared) plays only "
                         "between two seats — Forge's limit; start at beginning of combat")
    lines += [f"activeplayer={pid[scenario.get('active_seat') or 'you']}",
              f"activephase={phase}",
              f"turn={int(scenario.get('turn') or 1)}"]
    if scenario.get("stack"):
        notes.append("the stack is not played into a slice — the board starts with it empty")

    for s in seats:
        p, sid = pid[s["seat"]], s["seat"]
        deck, commanders = _deck_copies(seat_slugs[sid])
        left = list(deck)

        def take(name):
            if name in left:
                left.remove(name)

        zones = {}
        board, placed = [], []
        for e in s.get("board") or []:
            spec, name = _entry(e, notes, sid)
            if spec is None:
                continue
            if name in commanders:
                spec += "|IsCommander"
            board.append(spec)
            if name:
                placed.append(name)
        zones["battlefield"] = board
        for z in ("graveyard", "exile"):
            zones[z] = [gs.entry_name(x) for x in s.get(z) or []]
            placed += zones[z]

        cmd = (s.get("commander") or {})
        in_command = [c for c in commanders if c not in placed]
        if cmd.get("zone") not in (None, "command"):
            in_command = []                     # the scenario says where it is
        zones["command"] = [f"{c}|IsCommander" for c in in_command]
        placed += in_command

        hand = s.get("hand")
        known = hand if isinstance(hand, list) else list((hand or {}).get("known") or [])
        placed += known
        for n in placed:
            take(n)
        left = keyed_order(left, seed)
        n_unknown = int((hand or {}).get("unknown") or 0) if isinstance(hand, dict) else 0
        if hand is None:
            n_unknown = 0
            notes.append(f"{sid}: no hand given — playing with an empty hand")
        drawn, left = left[:n_unknown], left[n_unknown:]
        if n_unknown:
            notes.append(f"{sid}: {n_unknown} unknown hand card(s) drawn from "
                         f"{seat_slugs[sid]}'s list")
        zones["hand"] = known + drawn

        lib = s.get("library") or {}
        top = list(lib.get("top") or []) if isinstance(lib, dict) else []
        for n in top:
            take(n)
        count = lib.get("count") if isinstance(lib, dict) else None
        rest = left if count is None else left[:max(0, int(count) - len(top))]
        zones["library"] = top + rest
        if count is not None and count > len(top) + len(left):
            notes.append(f"{sid}: library asked for {count}, the list has {len(top) + len(left)} left")

        for z in ("hand", "library", "graveyard", "exile"):
            for n in zones[z]:
                if n.split("|")[0] not in known_names:
                    unknown.append(f"{sid} {z}: {n}")
        for spec in zones["battlefield"]:
            n = spec.split("|")[0]
            if not n.startswith("t:") and n not in known_names:
                unknown.append(f"{sid} battlefield: {n}")

        lines.append(f"{p}life={int(s.get('life', DEFAULT_LIFE))}")
        for z, key in (("hand", "hand"), ("library", "library"), ("graveyard", "graveyard"),
                       ("battlefield", "battlefield"), ("exile", "exile"), ("command", "command")):
            lines.append(f"{p}{key}=" + ";".join(_front(x) for x in zones[z]))
    if unknown:
        raise StateError("not in the corpus (Forge would skip them silently): " + "; ".join(unknown[:12]))
    return "\n".join(lines) + "\n", notes
