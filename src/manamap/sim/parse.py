"""Simulation S2: the parser — Forge logs → events → per-game facts → aggregates.

The deterministic half of a SAMPLED artifact. Forge's games are not reproducible; this
module is, and a run record's `analysis` block is re-derivable from its logs by anyone
who has them (`validate-sim` does exactly that). What it reads is the log's own
vocabulary — every line a stable prefix — and what it refuses to do is infer what the
log does not say.

TOKENS ARE THE HONEST CASE OF THAT. A token enters the log on first USE: Forge names it
(`Cat Token (414)`) when it attacks, blocks, deals or takes damage, dies, or is targeted,
and a creation resolution (`…create a 1/1 white Cat creature token.`) states neither
the count (doubling is a replacement effect printed separately) nor the ids. So the
parser reports TWO token figures per seat, each with its limit named in the artifact:

  token_resolutions  — token-creating abilities that resolved for that seat (a floor on
                       creation events, blind to doubling and to X)
  tokens_observed    — distinct token ids that DID something for that seat (attacked or
                       blocked; the seat is learned from the assignment line). A token
                       that sat on the board is invisible here.

From the observed set come the figures the workbench wanted: token share of the seat's
combat damage to players, and tokens that died the turn they blocked (chumped).

Seat attribution is learned, never assumed: a permanent's controller is whoever assigned
it to attack or block, or whose `Land:`/`cast` line named it. The final blow on a seat is
the damage line immediately before the life line that crossed zero, attributed through
that map; when the source was never seen acting, `eliminated_by` is null and says so.

Turn numbers: `Turn: Turn N` lines are GLOBAL; `Game Outcome: Turn N` is the winner's own
turn count (`round`). Both are carried, as in S1.
"""

import math
import re
from collections import Counter, defaultdict

from manamap.sim import stats

# Seat labels are `Ai(k)-<meta name>` and meta names are kebab (`mm-<slug>`); `\S+`
# swallowed the comma after an `Activator:` tag and mis-attributed every token resolution.
_SEAT = r"(Ai\(\d+\)-[\w-]+)"
_PERM = r"(.+?) \((\d+)\)"                    # "Cat Token (414)" → name, id

RX = {
    "turn":      re.compile(r"^Turn: Turn (\d+) \(" + _SEAT + r"\)"),
    "phase":     re.compile(r"^Phase: " + _SEAT + r"'s? (.*)$"),
    "mana":      re.compile(r"^Mana: " + _PERM + r" - (.*)$"),
    "mulligan":  re.compile(r"^Mulligan: " + _SEAT + r" has kept a hand of (\d+) cards"),
    # THE SECOND MULLIGAN LINE, which was never matched. Forge emits one of
    # these per mulligan TAKEN and then one `kept` line with the final hand
    # size, and only the second was read — so the count of mulligans was in
    # every log and in no measurement.
    "mulligan_down": re.compile(r"^Mulligan: " + _SEAT + r" has mulliganed down to (\d+) cards\."),
    "land":      re.compile(r"^Land: " + _SEAT + r" played " + _PERM),
    "stack":     re.compile(r"^Add To Stack: " + _SEAT + r" (cast|triggered|activated) (.+?)(?: targeting \[(.*)\])?$"),
    "resolve":   re.compile(r"^Resolve Stack: (.*)$"),
    "attack":    re.compile(r"^Combat: " + _SEAT + r" assigned (.+) to attack " + _SEAT + r"\.?$"),
    "block":     re.compile(r"^Combat: " + _SEAT + r" assigned (.+) to block " + _PERM + r"\.?$"),
    "noblock":   re.compile(r"^(?:Combat: )?" + _SEAT + r" didn't block " + _PERM),
    "damage":    re.compile(r"^Damage: " + _PERM + r" deals (\d+) (combat |non-combat )?damage(?: \((\w+)\))? to (.+?)\.?$"),
    "life":      re.compile(r"^Life: Life: " + _SEAT + r" (-?\d+) > (-?\d+)"),
    "zone":      re.compile(r"^Zone Change: " + _PERM + r" was put into (\w+) from (\w+)"),
    "outcome_t": re.compile(r"^Game Outcome: Turn (\d+)"),
    "won":       re.compile(r"^Game Outcome: " + _SEAT + r" has won (?:because|due to) (.*)$"),
    "lost":      re.compile(r"^Game Outcome: " + _SEAT + r" has lost because (.*)$"),
    # Two endings, two format strings — see `forge._RESULT`. A real draw reads
    # "ended in a Draw! Took N ms." and was matched by neither pattern.
    "result":    re.compile(
        r"^Game Result: Game (\d+) ended in (?:a Draw! Took )?(\d+) ms"),
}
_TOKEN = re.compile(r"\bToken$")
_CREATE = re.compile(r"\bcreates?\b.*\btokens?\b", re.I)
_LOSES_LIFE = re.compile(r"\b(loses?|lose) (that much |\d+ |X )?life\b|\bdrain", re.I)
#: COUNTERS ARE NOT AN EVENT IN THIS LOG. Forge emits no `Counter:` line; a
#: +1/+1 counter only ever appears inside the ABILITY TEXT of a `Resolve Stack`
#: line — 5,519 of them in one 400-game run, and the parser read none of them.
#: So what is counted here is COUNTER-PLACING EVENTS, not counters: "put a
#: +1/+1 counter on each Vampire you control" is ONE line whether it lands one
#: counter or eight, and the log does not say which.
#:
#: The `on each` form is split out for exactly that reason — it is the one whose
#: size scales with the board, and it is what makes proliferate worth playing.
_COUNTER_PLACED_RE = re.compile(r"put (?:a|an|one|two|three|X|that many) "
                                r"\+1/\+1 counters? on", re.IGNORECASE)
_COUNTER_MASS_RE = re.compile(r"\+1/\+1 counters? on each", re.IGNORECASE)
_PROLIFERATE_RE = re.compile(r"\bproliferate\b", re.IGNORECASE)
_ACTIVATOR = re.compile(r"\[(?:Card: .*?, )?Activator: " + _SEAT)
_PLAYER_TAG = re.compile(r"\[(?:Player|Phase): " + _SEAT + r"\]")


_SEP = re.compile(r"^(?:,\s*|\s*and\s+)+")


def _perms(text):
    """Every `Name (id)` in a combat assignment list ('A (1) and B (2)' or 'A (1), B (2)').
    The non-greedy name starts right after the previous id, so the list separator is
    stripped — measured: a lifted board read ', Insect Token' and 'and Insect Token'."""
    return [(_SEP.sub("", m.group(1)).strip(), m.group(2)) for m in re.finditer(_PERM, text)]


def _is_token(name):
    return bool(_TOKEN.search(name))


# CR 903.10a: a player dealt 21 combat damage by the same commander over the course of
# the game loses. It is the ONLY win condition some decks have, and until now the parser
# could not see it: `combat_damage_dealt_to_players` sums every source and every
# defender at once, so a commander that hit three seats for 20 each looked identical to
# one that hit a single seat for 60 and killed them.
COMMANDER_DAMAGE_LETHAL = 21


def _is_commander(source_name, names):
    """Does this damage source name the seat's commander?

    Matched on the face as well as the joined `A // B` form, because Forge logs a
    transformed or melded permanent under the face that is on the battlefield while a
    decklist names the card. Falls back to False when the seat's commander is unknown,
    which is what keeps the whole block absent rather than wrong.
    """
    if not names or not source_name:
        return False
    src = str(source_name).strip()
    for n in names:
        if src == n or src in {f.strip() for f in str(n).split(" // ")}:
            return True
    return False


def _new_game():
    return {"seats": [], "turn": 0, "active": None, "events": [], "owner": {},
            "mulligan": {}, "mulligans_taken": defaultdict(int), "started": None, "outcome": {"winner": None, "won_by": None, "round": None,
                                        "global_turn": None, "draw": False, "lost": {},
                                        "ms": None,
                                        # Every seat Forge declared a winner. One
                                        # is a result; more than one means the
                                        # clock stopped the game — see `_settle`.
                                        "_won": [], "truncated": False}}


def parse_games(text):
    """Split one JVM log into games and parse each into an event list + outcome."""
    games, g, last_turn = [], None, None
    for raw in text.splitlines():
        line = raw.rstrip()
        m = RX["mulligan_down"].match(line)
        if m:
            if g is None or g["outcome"]["ms"] is not None or g["outcome"]["winner"]:
                g = _new_game(); games.append(g)
            g["mulligans_taken"][m.group(1)] += 1
            if m.group(1) not in g["seats"]:
                g["seats"].append(m.group(1))
            continue
        m = RX["mulligan"].match(line)
        if m:
            if g is None or g["outcome"]["ms"] is not None or g["outcome"]["winner"]:
                g = _new_game(); games.append(g)
            g["mulligan"][m.group(1)] = int(m.group(2))
            if m.group(1) not in g["seats"]:
                g["seats"].append(m.group(1))
            continue
        if g is None:
            continue
        m = RX["turn"].match(line)
        if m:
            g["turn"] = int(m.group(1)); g["active"] = m.group(2); last_turn = g["turn"]
            # WHO ACTUALLY WENT FIRST, which is a different fact from the `-d`
            # order the run record calls `seat_order`. Forge's
            # `determineFirstTurnPlayer` picks, from game 2 of a job onward, the
            # lowest-indexed seat that did not win the previous game — so the
            # deck that loses most starts most. Measured on one 25-game job:
            # edgar started 21 of 25. A seat-effect figure computed from `-d`
            # position would have reported the opposite of the truth.
            if g["turn"] == 1 and g.get("started") is None:
                g["started"] = m.group(2)
            continue
        ev = _event(line, g)
        if ev:
            ev["turn"] = g["turn"]; ev["active"] = g["active"]
            g["events"].append(ev)
            continue
        o = g["outcome"]
        m = RX["outcome_t"].match(line)
        if m:
            o["round"] = int(m.group(1)); o["global_turn"] = last_turn; continue
        m = RX["won"].match(line)
        if m:
            o["_won"].append((m.group(1), m.group(2).rstrip("."))); continue
        m = RX["lost"].match(line)
        if m:
            o["lost"][m.group(1)] = m.group(2).rstrip("."); continue
        m = RX["result"].match(line)
        if m:
            o["ms"] = int(m.group(2)); continue
    for g in games:
        _settle_outcome(g["outcome"])
    return [x for x in games
            if x["events"] or x["outcome"]["winner"] or x["outcome"]["truncated"]]


def _settle_outcome(o):
    """One winner, or none at all — the same rule `forge.parse_outcomes` uses.

    THIS FILE CARRIED ITS OWN COPY OF THE BUG, which is why nothing ever caught
    it: `validate-sim` re-derives a tracked record through `analyze_logs`, so the
    re-derivation contained the identical overwrite and agreed with the record
    forever. Two implementations of one rule, wrong in the same direction.

    When Forge's `-c` clock fires it calls `setGameOver(GameEndReason.Draw)` —
    the game is a DRAW — and then prints `has won because all opponents have
    lost` for every seat still alive. Taking the last such line credited the
    highest-numbered survivor. Our deck is always `Ai(1)`, so across 121
    truncated games it was credited with ZERO while surviving to the clock in 93
    of them, and `baylen-tokens` (always last) took 73 of its 85 recorded wins
    that way.
    """
    won = o.pop("_won", [])
    if len({name for name, _why in won}) > 1:
        o["truncated"] = True
        o["draw"] = True          # Forge's own verdict: `GameEndReason.Draw`
        o["winner"] = o["won_by"] = None
    elif won:
        o["winner"], o["won_by"] = won[0]


def _event(line, g):
    m = RX["phase"].match(line)
    if m:
        return {"kind": "phase", "seat": m.group(1), "text": m.group(2).rstrip()}
    m = RX["mana"].match(line)
    if m:
        return {"kind": "mana", "perm": (m.group(1), m.group(2)), "ability": m.group(3)}
    m = RX["land"].match(line)
    if m:
        g["owner"][m.group(3)] = m.group(1)
        return {"kind": "land", "seat": m.group(1), "card": m.group(2), "id": m.group(3)}
    m = RX["stack"].match(line)
    if m:
        return {"kind": m.group(2), "seat": m.group(1), "what": m.group(3),
                "targets": m.group(4)}
    m = RX["resolve"].match(line)
    if m:
        text = m.group(1)
        seat = None
        a = _ACTIVATOR.search(text) or _PLAYER_TAG.search(text)
        if a:
            seat = a.group(1)
        return {"kind": "resolve", "seat": seat or g["active"], "text": text,
                "creates_token": bool(_CREATE.search(text))}
    m = RX["attack"].match(line)
    if m:
        perms = _perms(m.group(2))
        for _, pid in perms:
            g["owner"][pid] = m.group(1)
        return {"kind": "attack", "seat": m.group(1), "attackers": perms, "defender": m.group(3)}
    m = RX["block"].match(line)
    if m:
        perms = _perms(m.group(2))
        for _, pid in perms:
            g["owner"][pid] = m.group(1)
        return {"kind": "block", "seat": m.group(1), "blockers": perms,
                "blocked": (m.group(3), m.group(4))}
    m = RX["noblock"].match(line)
    if m:
        return {"kind": "noblock", "seat": m.group(1), "attacker": (m.group(2), m.group(3))}
    m = RX["damage"].match(line)
    if m:
        target = m.group(6)
        tm = re.match(_SEAT + r"$", target)
        return {"kind": "damage", "source": (m.group(1), m.group(2)), "amount": int(m.group(3)),
                "combat": (m.group(4) or "").strip() == "combat",
                "noncombat": (m.group(4) or "").strip() == "non-combat",
                "tag": m.group(5), "to_player": tm.group(1) if tm else None,
                "to_perm": None if tm else target}
    m = RX["life"].match(line)
    if m:
        return {"kind": "life", "seat": m.group(1), "from": int(m.group(2)), "to": int(m.group(3))}
    m = RX["zone"].match(line)
    if m:
        return {"kind": "zone", "card": m.group(1), "id": m.group(2), "to": m.group(3), "from": m.group(4)}
    return None


# ── Per-game facts ──────────────────────────────────────────────────────────

def _aim(per, owner, ev, seats):
    """Attribute one targeted stack object to the seats it was aimed AT.

    A target group holds either seat labels (`targeting [Ai(3)-mm-baylen]`) or
    permanents (`targeting [Demon's Horn (131)]`); a permanent resolves through
    `owner`, which is learned from the `Land:` and `Add To Stack:` lines that
    named it. A permanent nobody was ever seen playing is UNATTRIBUTED and
    counted for neither side — the alternative is guessing, and guessing here
    would systematically credit the active seat.

    One stack object counts ONCE per opposing seat it touched, not once per
    target: a wrath naming four of one player's creatures is one act of
    interaction against that player, and counting four would make a
    board-targeting deck look four times as interactive as a spot-removal one.
    """
    src = ev["seat"]
    raw = ev.get("targets")
    if not raw or src not in per:
        return
    hit = set()
    for name in [t.strip() for t in raw.split(",")]:
        if name in per:                       # a seat, targeted directly
            hit.add(name)
            continue
        for _n, pid in _perms(name):          # a permanent, via the owner map
            who = owner.get(pid)
            if who in per:
                hit.add(who)
    hit.discard(src)                          # aiming at your own board is not interaction
    if hit:
        per[src]["interaction_cast"] += 1
        for seat in hit:
            per[seat]["interaction_received"] += 1


def game_facts(g, commanders=None):
    """Per-game facts. `commanders` maps a Forge seat label to that seat's commander
    name(s); pass it and the seat gains a per-defender commander-damage tally, omit it
    and the key is ABSENT rather than zero — the log does not name a commander anywhere,
    so an unknown one must not be reported as "dealt 0"."""
    seats = list(g["seats"])
    commanders = commanders or {}
    per = {s: {"lands": 0, "casts": 0, "activations": 0, "triggers": 0,
               "combat_damage_dealt_to_players": 0, "combat_damage_taken": 0,
               "noncombat_damage_dealt_to_players": 0,
               "token_resolutions": 0, "tokens_observed": set(), "token_attackers": set(),
               "token_blockers": set(), "token_combat_damage_to_players": 0,
               "tokens_chumped": 0, "creatures_lost": 0,
               "life_by_turn": {}, "eliminated_turn": None, "eliminated_by": None,
               "eliminated_how": None,
               "first_attack_turn": None, "damage_to_players_by_turn": defaultdict(int),
               # PER-TURN LOSSES — what a wipe is measured against.
               # `creatures_lost` above is a whole-game total and cannot say
               # whether eleven permanents left one at a time or all at once,
               # which is the only thing that separates attrition from a wipe.
               #
               # THERE IS NO MATCHING BOARD SERIES AND THERE CANNOT BE. Forge
               # logs a `Zone Change` when a permanent LEAVES the battlefield
               # for a graveyard and emits nothing when one arrives — measured
               # on a 100-game pod run: 0 `to Battlefield` lines in the entire
               # log. A board count reconstructed from this would be a series of
               # zeros wearing the name of a measurement, so it is ABSENT.
               "permanents_lost_by_turn": defaultdict(int),
               # See _COUNTER_PLACED_RE: EVENTS, not counters, and attributed
               # to the tagged seat when the line carries one and to the ACTIVE
               # seat otherwise, which is the same approximation the drain
               # attribution already makes.
               "counter_events": 0,
               "mass_counter_events": 0,
               "proliferate_events": 0,
               # INTERACTION, and deliberately NOT called removal. The log gives
               # `Add To Stack: SEAT cast X targeting [...]`, and the target
               # group is resolved through the same `owner` map the elimination
               # attribution uses. What that yields is "this seat aimed a spell
               # or ability at something ANOTHER seat controlled" — which covers
               # a Swords, an edict and a drain, and also covers a Giant Growth
               # cast on an opponent's creature. The log does not say what the
               # spell DID, so naming this `removal_cast` would be a claim the
               # evidence cannot carry.
               #
               # It answers the question `creatures_lost` could not: that figure
               # counts a creature leaving the battlefield and cannot separate a
               # removal spell from a chump block from a sacrifice outlet — the
               # distinction that decides whether an Edgar or Yawgmoth result
               # means the deck was dismantled or was doing its job.
               #
               # SELF-TARGETING IS EXCLUDED, which is most of the traffic: the
               # equip that prompted this check was Lightning Greaves on the
               # caster's own commander.
               "interaction_cast": 0,
               "interaction_received": 0,
               "commander_damage_by_defender": defaultdict(int)}
           for s in seats}
    owner = g["owner"]
    # The last thing that could have cost a seat life: a DAMAGE line (source's controller)
    # or a resolved ability whose text says the victim LOSES life (its controller, from the
    # Activator/Player tag or the active seat). Measured on the pod run: Vito wins 9 of 20
    # on 7 combat damage a game — every one a drain, and damage-only attribution got them
    # all wrong. Whichever came later decides.
    last_cause = None                      # (seat, turn, kind)
    blockers_this_turn = {}          # token id -> (seat, turn) for chump detection
    for ev in g["events"]:
        k = ev["kind"]
        if k == "land":
            per[ev["seat"]]["lands"] += 1
        elif k == "cast":
            per[ev["seat"]]["casts"] += 1
            _aim(per, owner, ev, seats)
        elif k == "activated":
            per[ev["seat"]]["activations"] += 1
            _aim(per, owner, ev, seats)
        elif k == "triggered":
            per[ev["seat"]]["triggers"] += 1
        elif k == "resolve":
            if ev["creates_token"] and ev["seat"] in per:
                per[ev["seat"]]["token_resolutions"] += 1
            if _LOSES_LIFE.search(ev["text"]) and ev["seat"] in per:
                last_cause = (ev["seat"], ev["turn"], "life loss")
            # THE COUNTER CHANNEL LIVES HERE AND NOT IN A SECOND `resolve`
            # BRANCH. The first cut added one lower down the chain, where it was
            # unreachable — this `elif` had already claimed every resolve event
            # — and it read a flat zero on 400 games while the regex matched
            # seven times in the first log alone. A dead branch and a channel
            # that does not fire look identical from the outside.
            if ev["seat"] in per:
                t = ev["text"] or ""
                if _COUNTER_PLACED_RE.search(t):
                    per[ev["seat"]]["counter_events"] += 1
                    if _COUNTER_MASS_RE.search(t):
                        per[ev["seat"]]["mass_counter_events"] += 1
                if _PROLIFERATE_RE.search(t):
                    per[ev["seat"]]["proliferate_events"] += 1
        elif k == "attack":
            p = per[ev["seat"]]
            if p["first_attack_turn"] is None:
                p["first_attack_turn"] = ev["turn"]
            for name, pid in ev["attackers"]:
                if _is_token(name):
                    p["tokens_observed"].add(pid); p["token_attackers"].add(pid)
        elif k == "block":
            p = per[ev["seat"]]
            for name, pid in ev["blockers"]:
                if _is_token(name):
                    p["tokens_observed"].add(pid); p["token_blockers"].add(pid)
                    blockers_this_turn[pid] = (ev["seat"], ev["turn"])
        elif k == "damage":
            src_name, src_id = ev["source"]
            src_seat = owner.get(src_id)
            if ev["to_player"]:
                tgt = ev["to_player"]
                if tgt in per:
                    if ev["combat"]:
                        per[tgt]["combat_damage_taken"] += ev["amount"]
                        if src_seat in per:
                            per[src_seat]["combat_damage_dealt_to_players"] += ev["amount"]
                            per[src_seat]["damage_to_players_by_turn"][ev["turn"]] += ev["amount"]
                            if _is_commander(src_name, commanders.get(src_seat)):
                                per[src_seat]["commander_damage_by_defender"][tgt] += ev["amount"]
                            if _is_token(src_name):
                                per[src_seat]["token_combat_damage_to_players"] += ev["amount"]
                                per[src_seat]["tokens_observed"].add(src_id)
                    elif ev["noncombat"] and src_seat in per:
                        per[src_seat]["noncombat_damage_dealt_to_players"] += ev["amount"]
                last_cause = (src_seat, ev["turn"], "damage")
        elif k == "life":
            p = per.get(ev["seat"])
            if p is None:
                continue
            p["life_by_turn"][ev["turn"]] = ev["to"]
            if ev["to"] <= 0 and p["eliminated_turn"] is None:
                p["eliminated_turn"] = ev["turn"]
                if last_cause and last_cause[1] == ev["turn"] and last_cause[0] and last_cause[0] != ev["seat"]:
                    p["eliminated_by"] = last_cause[0]
                    p["eliminated_how"] = last_cause[2]
        elif k == "zone" and ev["from"] == "Battlefield":
            s = owner.get(ev["id"])
            if ev["to"] == "Graveyard" and s in per:
                per[s]["creatures_lost"] += 1      # anything the seat was seen acting with
                per[s]["permanents_lost_by_turn"][ev["turn"]] += 1
                if ev["id"] in blockers_this_turn and blockers_this_turn[ev["id"]][1] == ev["turn"]:
                    per[s]["tokens_chumped"] += 1
    # finalise
    for s, p in per.items():
        p["tokens_observed"] = len(p["tokens_observed"])
        p["token_attackers"] = len(p["token_attackers"])
        p["token_blockers"] = len(p["token_blockers"])
        p["damage_to_players_by_turn"] = dict(sorted(p["damage_to_players_by_turn"].items()))
        p["permanents_lost_by_turn"] = dict(sorted(p["permanents_lost_by_turn"].items()))
        if commanders.get(s):
            cd = dict(sorted(p["commander_damage_by_defender"].items()))
            p["commander_damage_by_defender"] = cd
            p["commander_damage_max"] = max(cd.values(), default=0)
            p["commander_damage_lethal"] = p["commander_damage_max"] >= COMMANDER_DAMAGE_LETHAL
        else:
            p.pop("commander_damage_by_defender")
        p["life_by_turn"] = dict(sorted(p["life_by_turn"].items()))
        d = p["combat_damage_dealt_to_players"]
        p["token_damage_share"] = round(p["token_combat_damage_to_players"] / d, 3) if d else None
    # THE OPENING HAND, WHICH WAS MEASURED AND THROWN AWAY. `g["mulligan"]` has
    # been parsed since the parser existed and `compact()` drops it, so the
    # figure reached the game dict and no further. Both halves ride in `per_seat`
    # now, where the aggregate and the compact rows already look.
    #
    # TWO FIGURES, NOT ONE, AND FORGE'S OWN RELATION BETWEEN THEM IS A RULES
    # GAP. Measured across all 130 tracked logs and 5,056 seat-hands, with ZERO
    # exceptions: 0 mulligans keeps 7, ONE mulligan also keeps 7, two keeps 6,
    # three keeps 5 — so `kept = 7 - max(0, taken - 1)` and **Forge gives the
    # first mulligan free**. That is not the London rule, under which one
    # mulligan keeps 7 and bottoms 1 for a hand of six. Reporting only "mean
    # mulligans" would hide it; reporting both makes it visible, and
    # `analysis.limits` names it.
    for seat in seats:
        per[seat]["mulligans_taken"] = int(g["mulligans_taken"].get(seat, 0))
        per[seat]["mulligan_kept"] = g["mulligan"].get(seat)

    o = g["outcome"]
    return {"seats": seats, "started": g.get("started"),
            "winner": o["winner"], "won_by": o["won_by"], "round": o["round"],
            "truncated": bool(o.get("truncated")),
            "global_turn": o["global_turn"], "ms": o["ms"], "lost": dict(o["lost"]),
            "mulligan": dict(g["mulligan"]), "per_seat": per}


# ── Board wipes and what happens after one ──────────────────────────────────
#
# THE METRIC THE GOLDFISH CANNOT HAVE. Nothing dies in `pilot/goldfish.py` — no
# blockers, no removal, no sacrifice — so every death-triggered card in a deck
# contributes exactly zero there, and "neither of my decks generates value on the
# way down" (edgar-vampires, captain's log 002) was unanswerable. Forge kills
# things, so this is where the question lives.
#
# WHAT COUNTS AS A WIPE, and why it is not "a turn creatures died". Creatures die
# in combat every turn of every game; that is attrition, not a wipe. The
# signature of mass removal is BREADTH — several seats losing permanents on the
# same turn, which a single attack cannot do because one attacker hits one
# defender. So a wipe is a turn on which at least `WIPE_MIN_PERMANENTS` leave the
# battlefield for a graveyard across at least `WIPE_MIN_SEATS` seats.
#
# IT IS A HEURISTIC OVER A LOG, NOT A READ OF THE CARD. A huge multi-block turn
# can trip it and an Edict cannot. The rate it fires at is reported beside every
# figure so a reader can see how often it thought it saw one, and the threshold
# is stated rather than tuned to make a number look good.
WIPE_MIN_PERMANENTS = 5
WIPE_MIN_SEATS = 2
#: How many turns after the wipe to watch. Two, because that is the window the
#: log entry describes ("rebuild, wipe, rebuild") and because a longer one starts
#: measuring the rest of the game instead of the recovery.
WIPE_RECOVERY_TURNS = 2


def wipes(fact, seat):
    """Every wipe in one game, and what `seat` got paid on and after it.

    VALUE ON THE WAY DOWN IS THE MEASURE, not board recovery. The obvious
    version — board before, board two turns later — is not available: Forge
    logs a permanent LEAVING the battlefield and never one arriving, so there
    is no board series to difference. What the log does carry is damage and
    life, and for a drain deck that is the better question anyway. The
    captain's log describes exactly this play: "a board wipe going off with the
    drain package online — eleven creatures died, every opponent lost 11".

    `damage_on_wipe` is the payout as the board dies; `damage_after` is the two
    turns of rebuilding behind it. A wipe the game did not outlive is marked
    `truncated` rather than averaged in beside full windows.
    """
    per = fact["per_seat"]
    if seat not in per:
        return []
    def _dmg(p, t):
        d = p.get("damage_to_players_by_turn", {})
        return d.get(t, d.get(str(t), 0))
    turns = sorted({int(t) for p in per.values()
                    for t in p.get("permanents_lost_by_turn", {})})
    last = fact.get("global_turn") or (max(turns) if turns else 0)
    mine = per[seat]
    out = []
    for t in turns:
        lost = {s: p["permanents_lost_by_turn"].get(t, 0) for s, p in per.items()}
        total = sum(lost.values())
        hit = sum(1 for v in lost.values() if v)
        if total < WIPE_MIN_PERMANENTS or hit < WIPE_MIN_SEATS:
            continue
        end = t + WIPE_RECOVERY_TURNS
        out.append({
            "turn": t, "permanents_lost_table": total, "seats_hit": hit,
            "permanents_lost_mine": lost.get(seat, 0),
            "damage_on_wipe": _dmg(mine, t),
            "damage_after": sum(_dmg(mine, x) for x in range(t + 1, end + 1)),
            "truncated": end > last,
        })
    return out


def wipe_recovery(facts, seat):
    """Aggregate `wipes` over a run. Absent, never zero, when none were seen."""
    per_game = [wipes(f, seat) for f in facts]
    seen = [w for g in per_game for w in g]
    full = [w for w in seen if not w["truncated"]]
    n = len(facts)
    if not seen:
        return {"available": False,
                "why": (f"no turn in {n} game(s) lost {WIPE_MIN_PERMANENTS}+ "
                        f"permanents across {WIPE_MIN_SEATS}+ seats. That is an "
                        f"absent measurement, not a claim that the pod runs no "
                        f"wipes.")}
    return {
        "available": True,
        "definition": (f"a turn on which {WIPE_MIN_PERMANENTS}+ permanents left "
                       f"the battlefield for a graveyard across {WIPE_MIN_SEATS}+ "
                       f"seats; recovery is the next {WIPE_RECOVERY_TURNS} turns"),
        "games": n,
        # A LIST, NOT THE TUPLE `wilson` RETURNS. `validate_sim` re-derives a
        # record and compares it to the stored JSON, where a tuple has become a
        # list — so a tuple here can never match itself after a round trip. The
        # validator caught it on the first run; the rest of this module already
        # writes `[lo, hi]` for exactly this reason.
        "games_with_a_wipe_rate": round(sum(1 for g in per_game if g) / n, 3) if n else None,
        "games_with_a_wipe_ci95": list(wilson(sum(1 for g in per_game if g), n)),
        "wipes_seen": len(seen),
        "wipes_with_a_full_window": len(full),
        "permanents_lost_mine": mean_ci([w["permanents_lost_mine"] for w in seen]),
        # THE TWO FIGURES THE THESIS TURNS ON. A deck that "generates value on
        # the way down" bills the table as its own board dies; one that does not
        # is only rebuilding.
        "damage_on_wipe": mean_ci([w["damage_on_wipe"] for w in seen]),
        "damage_after": mean_ci([w["damage_after"] for w in full]) if full else None,
        "limits": [
            "A HEURISTIC OVER A LOG, not a read of the card that did it. A large "
            "multi-block combat can trip the threshold and a one-sided Edict "
            "cannot; the rate it fired at is reported above so the reader can "
            "judge it.",
            "Board size before and after is NOT here and cannot be: Forge logs "
            "a permanent leaving the battlefield and never one arriving.",
            "Turn numbers are Forge's, which advance once per PLAYER turn — "
            "turn 24 of a four-handed game is round 6.",
        ],
    }


# ── Aggregates over a run ───────────────────────────────────────────────────

def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion, rounded; (None, None) when n == 0.

    The arithmetic lives in `sim.stats.wilson_bounds`, unrounded, because
    Newcombe's interval on a DIFFERENCE is built out of these bounds and rounding
    them first puts that error into the difference. One implementation, two
    presentations: this one reports, that one composes.
    """
    lo, hi = stats.wilson_bounds(k, n, z)
    if lo is None:
        return None, None
    return round(lo, 3), round(hi, 3)


def _turn_order(facts, labels, decided):
    """Did going first help this seat — and how often did it go first at all?

    PRD §14 calls this "win rate by turn order position" and the obvious
    implementation reads `-d` position, which is WRONG HERE and would report
    backwards. Forge's `determineFirstTurnPlayer` picks, from game 2 of a job
    onward, the lowest-indexed seat that DID NOT WIN the previous game — so the
    deck that loses most starts most. Measured on one 25-game job:
    edgar-vampires started 21 of 25 while winning almost none of them.

    So the figure is built from `Turn: Turn 1 (seat)`, which names who actually
    went first, and it reports `started_rate` beside the split — because a seat
    that starts 84% of the time is not being measured on turn order at all, it
    is being measured on its own losing streak. THE CONFOUND TRAVELS WITH THE
    FIGURE; it is not a caveat someone has to look up.
    """
    mine = [f for f in facts if f.get("started") in labels]
    seen = [f for f in facts if f.get("started")]
    if not seen:
        # Absent, not zero. An older log the parser never read a turn line from
        # cannot report this, and 0.0 would be a measurement.
        return {"turn_order": None}

    first_w = sum(1 for f in mine if f["winner"] in labels)
    rest = [f for f in seen if f.get("started") not in labels]
    rest_w = sum(1 for f in rest if f["winner"] in labels)

    def _rate(k, n):
        """A rate never travels without its N and its interval."""
        if not n:
            return None
        lo, hi = wilson(k, n)
        return {"rate": round(k / n, 3), "wins": k, "games": n, "ci95": [lo, hi]}

    return {"turn_order": {
        "games_with_a_starter": len(seen),
        "games_started": len(mine),
        "started_rate": round(len(mine) / len(seen), 3),
        "win_rate_when_starting": _rate(first_w, len(mine)),
        "win_rate_when_not": _rate(rest_w, len(rest)),
        "basis": "who is active on turn 1, from the log — NOT the `-d` seat "
                 "order, which Forge overrides from game 2 of a job onward by "
                 "giving the first turn to the lowest-indexed seat that did not "
                 "win the previous game. Read `started_rate` first: a seat that "
                 "starts far more than its share is being measured on its own "
                 "losing streak, not on turn order.",
    }}


def _placement(facts, labels):
    """Finishing position: 1 is the survivor, then latest elimination first.

    A game where nobody was eliminated (a clock-out) has no ordering and is
    excluded rather than filed as a draw at position 1.
    """
    counts = Counter()
    ranked = 0
    for f in facts:
        if f.get("truncated"):
            continue
        turns = {s: p.get("eliminated_turn") for s, p in f["per_seat"].items()}
        if all(v is None for v in turns.values()):
            continue
        # Survivors first (None sorts last by turn), then latest death upward.
        order = sorted(turns, key=lambda s: (turns[s] is not None,
                                             -(turns[s] or 0)))
        mine = next((i for i, s in enumerate(order, 1) if s in labels), None)
        if mine:
            counts[mine] += 1
            ranked += 1
    if not ranked:
        return {"placement": None}
    return {"placement": {
        "games_ranked": ranked,
        "by_position": {str(k): v for k, v in sorted(counts.items())},
        "mean_position": round(
            sum(k * v for k, v in counts.items()) / ranked, 3),
        "basis": "1 is the seat still standing; the rest are ordered by how "
                 "late they were eliminated. Clock-outs have no ordering and "
                 "are excluded, so `games_ranked` is below the game count.",
    }}


def mean_ci(xs):
    """`{mean, median, min, max, ci95, n}` — the mean never travels alone.

    The median is here because a mean over a skewed sample is a true number that
    describes no game. Measured on the V1-vs-V2 kianne experiment, arm B's per-game
    commander damage was `0 0 0 0 0 0 0 0 0 0 31 178`: mean 17.42 against a V1 mean
    of 2.25, which reads as a sevenfold improvement. The median is 0 in BOTH arms —
    the whole difference is two games, one of them a blowout, and the deck connected
    in FEWER games after the change. The ci95 of [-11.64, 46.47] did span zero, so
    the record was already honest and the interval discipline worked; it took
    sorting the per-game values by hand to see it. Now it does not.
    """
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "n": 0}
    n = len(xs)
    mean = sum(xs) / n
    ordered = sorted(xs)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    out = {"mean": round(mean, 2), "median": round(median, 2),
           "min": round(ordered[0], 2), "max": round(ordered[-1], 2), "n": n}
    if n < 2:
        return out
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    half = 1.96 * sd / math.sqrt(n)
    out["ci95"] = [round(mean - half, 2), round(mean + half, 2)]
    return out


def _life_delta(per_seat, sign):
    """Total life GAINED (sign +1) or LOST (sign -1) across the game.

    `life_by_turn` records the seat's total at each turn and nothing summed the
    movement, so a deck with seventeen lifelink and lifegain cards published no
    figure for any of it. Gains and losses are kept apart because they are
    different facts: a deck can gain 40 and lose 60 and end where it started.
    """
    series = per_seat.get("life_by_turn") or {}
    vals = [v for _, v in sorted(series.items(), key=lambda kv: int(kv[0]))]
    total = 0
    for a, b in zip(vals, vals[1:]):
        d = b - a
        if (d > 0) == (sign > 0) and d:
            total += abs(d)
    return total


def aggregate(facts, slug_label, label, commanders=None):
    """`facts` is a list of game_facts; `slug_label` the Forge seat label of OUR deck;
    `label` maps Forge seat labels to slugs; `commanders` seat label -> commander name(s)."""
    commanders = commanders or {}
    n = len(facts)
    seats = sorted({s for f in facts for s in f["seats"]})
    wins = Counter(f["winner"] for f in facts if f["winner"])

    # THE DECKS ROTATE THROUGH THE SEATS, so one deck has SEVERAL Forge labels:
    # `Ai(1)-mm-vito` through `Ai(4)-mm-vito` are all the same deck. Group by the
    # slug and aggregate over every label it ever sat at.
    #
    # THIS BUG ARRIVED WITH THE ROTATION. Before it, each deck sat at one fixed
    # index, `label` was 1:1, and `out["seats"][name] = ...` in a loop over raw
    # seats was correct. The moment a deck could occupy four labels, that
    # assignment started OVERWRITING: the last index processed won, and every
    # figure for the deck was computed from only the games where it happened to
    # sit there. Measured on the 100-game control: the analysis block reported
    # vito with 6 wins and pod-control with 3, against the summary's tally of 47
    # and 9 over the same games. Same class as the truncation bug — a per-seat
    # figure quietly computed over a quarter of the sample — and it is worse
    # than a wrong win count, because combat damage, interaction and elimination
    # were all being averaged over that same quarter.
    by_name = {}
    for s in seats:
        by_name.setdefault(label.get(s, s), []).append(s)

    # A GAME NOBODY WON IS NOT A GAME SOMEBODY LOST. `win_rate` divides by the
    # DECIDED games, matching `summary`; dividing by `n` counts every clock-out
    # against every seat. The summary was fixed when truncation was; this second
    # tally was not, so the two disagreed inside one record.
    decided = sum(1 for f in facts if f["winner"])
    out = {"games": n, "decided": decided, "seats": {name: {} for name in by_name}}
    # WHAT THIS DECK DID WHEN THE BOARD DIED. Attached for OUR seat only: it is
    # a question about the pilot's deck, and computing it for three AI seats
    # would put three numbers on the page that nobody asked and nobody reads.
    if slug_label:
        out["wipe_recovery"] = wipe_recovery(facts, slug_label)
    for name, raw in by_name.items():
        k = sum(wins.get(s, 0) for s in raw)
        lo, hi = wilson(k, decided)
        ps = [f["per_seat"][s] for f in facts for s in raw if s in f["per_seat"]]
        out["seats"][name] = {
            "wins": k,
            "win_rate": round(k / decided, 3) if decided else None,
            "win_rate_ci95": [lo, hi],
            "games_played": len(ps),
            "eliminated_turn": mean_ci([p["eliminated_turn"] for p in ps]),
            "eliminated_by": dict(Counter(label.get(p["eliminated_by"], p["eliminated_by"])
                                          for p in ps if p["eliminated_by"])),
            "eliminated_how": dict(Counter(p["eliminated_how"] for p in ps if p["eliminated_how"])),
            "lands": mean_ci([p["lands"] for p in ps]),
            "casts": mean_ci([p["casts"] for p in ps]),
            # THE OPENING HAND. Parsed since the parser existed, dropped by
            # `compact()` and never read by this function — a live measurement
            # thrown away, and one of the metrics the PRD's catalog asks for.
            #
            # BOTH FIGURES, because Forge's relation between them is a rules
            # gap rather than arithmetic: across all 130 tracked logs and 5,056
            # seat-hands with zero exceptions, one mulligan still keeps SEVEN
            # (`kept = 7 - max(0, taken - 1)`), so Forge gives the first
            # mulligan free. Under the London rule one mulligan keeps 7 and
            # bottoms 1 for a hand of six. A single "mean mulligans" would hide
            # that; `limits` names it.
            "mulligans_taken": mean_ci([p["mulligans_taken"] for p in ps]),
            **_turn_order(facts, raw, decided),
            **_placement(facts, raw),
            "mulligan_kept": mean_ci([p["mulligan_kept"] for p in ps
                                      if p.get("mulligan_kept") is not None]),
            "combat_damage_dealt_to_players": mean_ci([p["combat_damage_dealt_to_players"] for p in ps]),
            "combat_damage_taken": mean_ci([p["combat_damage_taken"] for p in ps]),
            # TOTAL IMPACT, AND THE SEVEN FIELDS THIS FUNCTION USED TO COMPUTE
            # AND THROW AWAY.
            #
            # `game_facts` has always accumulated `noncombat_damage_dealt_to_players`,
            # `life_by_turn`, `creatures_lost`, `activations`, `triggers` and the
            # per-turn series — and `aggregate` published NONE of them. So a
            # record for a deck whose stated engine is DRAIN reported only its
            # combat half. Audited card by card on edgar-vampires: of seven
            # output channels the 99 actually produces, exactly TWO reached the
            # page — combat damage and tokens. Drain (9 cards), lifegain (17),
            # +1/+1 counters (15), card draw (11) and removal (7) were invisible,
            # and four branches were designed and killed against that view.
            "noncombat_damage_dealt_to_players": mean_ci(
                [p["noncombat_damage_dealt_to_players"] for p in ps]),
            "damage_dealt_total": mean_ci(
                [p["combat_damage_dealt_to_players"]
                 + p["noncombat_damage_dealt_to_players"] for p in ps]),
            "life_gained": mean_ci([_life_delta(p, +1) for p in ps]),
            "life_lost": mean_ci([_life_delta(p, -1) for p in ps]),
            "creatures_lost": mean_ci([p["creatures_lost"] for p in ps]),
            # THE COUNTER CHANNEL. Fifteen cards in edgar-vampires put +1/+1
            # counters and NONE of it reached a record until 2026-08-29, which
            # made the whole counter-and-proliferate thesis unjudgeable.
            "counter_events": mean_ci([p["counter_events"] for p in ps]),
            "mass_counter_events": mean_ci([p["mass_counter_events"] for p in ps]),
            "proliferate_events": mean_ci([p["proliferate_events"] for p in ps]),
            # See `_aim`: acts of interaction AIMED AT another seat's board or
            # face, and acts aimed at this one. `interaction_received` is the
            # pod's answer to "does the table take this deck seriously",
            # measured at the table rather than modelled by threat.py.
            "interaction_cast": mean_ci([p["interaction_cast"] for p in ps]),
            "interaction_received": mean_ci([p["interaction_received"] for p in ps]),
            "activations": mean_ci([p["activations"] for p in ps]),
            "triggers": mean_ci([p["triggers"] for p in ps]),
            "first_attack_turn": mean_ci([p["first_attack_turn"] for p in ps]),
            "tokens": {
                "token_resolutions": mean_ci([p["token_resolutions"] for p in ps]),
                "tokens_observed": mean_ci([p["tokens_observed"] for p in ps]),
                "token_attackers": mean_ci([p["token_attackers"] for p in ps]),
                "token_blockers": mean_ci([p["token_blockers"] for p in ps]),
                "token_combat_damage_to_players": mean_ci([p["token_combat_damage_to_players"] for p in ps]),
                "token_damage_share": mean_ci([p["token_damage_share"] for p in ps]),
                "tokens_chumped": mean_ci([p["tokens_chumped"] for p in ps]),
            },
        }
        # Present only where the seat's commander is known — see game_facts.
        #
        # UNION ACROSS THE DECK'S SEAT LABELS, not a lookup on one. `commanders`
        # is keyed by Forge label and the deck rotates, so `Ai(1)-mm-x` and
        # `Ai(3)-mm-x` are both this deck and both carry its commander. The first
        # cut of the grouping read a bare `s` here, which Python resolved to the
        # variable LEAKED from the loop that built `by_name` — so every seat was
        # given the LAST seat's commander, and the fixture caught radagast's
        # commander damage being published under Edgar Markov's name.
        cmd = sorted({c for lbl in raw for c in commanders.get(lbl, ())})
        if cmd and any("commander_damage_max" in p for p in ps):
            maxes = [p.get("commander_damage_max", 0) for p in ps]
            lethal = sum(1 for p in ps if p.get("commander_damage_lethal"))
            dealt = [sum((p.get("commander_damage_by_defender") or {}).values()) for p in ps]
            out["seats"][name]["commander_damage"] = {
                "commander": cmd,
                "dealt_total": mean_ci(dealt),
                "max_on_one_defender": mean_ci(maxes),
                "best_single_game_max": max(maxes, default=0),
                "games_reaching_21": lethal,
                "games_reaching_21_rate": round(lethal / n, 3) if n else None,
                "games_dealing_any": sum(1 for d in dealt if d),
            }
    # Our seat's clock: mean CUMULATIVE combat damage to players by ROUND (global turn
    # divided by the seat count), rounds 1..16. The shape of the kill, not one number.
    if slug_label and facts:
        nseats = max(len(f["seats"]) for f in facts) or 1
        curve = []
        for r in range(1, 17):
            vals = []
            for f in facts:
                p = f["per_seat"].get(slug_label)
                if not p:
                    continue
                vals.append(sum(v for t, v in p["damage_to_players_by_turn"].items()
                                if t <= r * nseats))
            curve.append({"round": r, **mean_ci(vals)})
        out["our_cumulative_combat_damage_by_round"] = curve
    out["round"] = mean_ci([f["round"] for f in facts])
    out["global_turn"] = mean_ci([f["global_turn"] for f in facts])
    out["won_by"] = dict(Counter(f["won_by"] for f in facts if f["won_by"]))
    out["limits"] = [
        "tokens_observed counts distinct token ids that attacked or blocked (or dealt combat "
        "damage) for the seat; a token that only sat on the board is invisible to the log.",
        "token_resolutions counts token-creating abilities that resolved; it is blind to how "
        "many tokens each made (X, doubling) — a floor on creation events, not a token count.",
        "eliminated_by is the controller of the last thing that could have cost the seat life "
        "before the life line that crossed zero — a damage source (its controller learned from "
        "attack/block/land/cast lines) or a resolved ability whose text says the victim loses "
        "life (its Activator/Player tag, else the active seat); eliminated_how says which. Null "
        "when neither was seen.",
        "Damage figures see DAMAGE only. Life LOSS that is not damage (Vito's drain, Blood "
        "Artist, 'each opponent loses 1 life') shows in life_by_turn, eliminated_turn and "
        "eliminated_how='life loss', never in a damage total. Measured: Vito wins 9 of 20 on "
        "7.0 combat damage a game.",
        "interaction_cast counts stack objects a seat aimed at ANOTHER seat's permanent "
        "or face, resolved through the learned owner map, once per opposing seat touched. "
        "It is not a removal count: the log records that a spell targeted something, never "
        "what the spell did, so a Swords, an edict, a drain and a pump spell aimed at an "
        "opponent's creature are one and the same to it. A permanent nobody was seen "
        "playing is unattributed and counted for neither side.",
        "FORGE GIVES THE FIRST MULLIGAN FREE, which is not the London rule and is "
        "a fidelity gap rather than a parser one. Measured across all 130 tracked "
        "logs and 5,056 seat-hands with ZERO exceptions: 0 mulligans keeps 7, ONE "
        "mulligan also keeps 7, two keeps 6, three keeps 5 -- so "
        "kept = 7 - max(0, taken - 1). Under the London mulligan a single "
        "mulligan draws seven and bottoms one, for a hand of SIX. So "
        "mulligans_taken and mulligan_kept are both reported and neither is "
        "derivable from the other under real rules; a deck that mulligans is "
        "being flattered by one card here.",
        "Card advantage, tutoring and recursion are ABSENT and cannot be added. Forge logs "
        "exactly two zone transitions -- Battlefield to Graveyard and Battlefield to Exile. "
        "Measured on a 100-game pod run: ZERO `from Library` lines of any kind. Draw, "
        "tutor and graveyard recursion are invisible to every record here, and no parser "
        "change recovers them; the goldfish is the only place those are measured.",
        "ci95 is a Wilson interval for rates and a normal interval for means; both are "
        "meaningless below ~10 games. A seed fixes the SHUFFLE, not the games — one configuration replayed twice gave four different games and one different winner, because Forge aborts its AI's evaluation on a WALL-CLOCK budget. The interval is the claim; the stored logs, not the seed, are the receipt.",
        "commander_damage is per DEFENDER (CR 903.10a asks for 21 from the same commander "
        "on one player), so max_on_one_defender is the number the win condition reads and "
        "dealt_total is not — a commander that hit three seats for 20 each dealt 60 and "
        "killed nobody. It counts COMBAT damage by a permanent whose logged name matches "
        "the seat's commander; a commander dealing noncombat damage does not count toward "
        "903.10a and is excluded here too. The block is ABSENT when the seat's commander "
        "is unknown, because the log never names one.",
    ]
    return out


def compact(fact, label):
    """A per-game row small enough to track: outcome + per-seat scalars, no curves."""
    return {"winner": label.get(fact["winner"], fact["winner"]), "won_by": fact["won_by"],
            "round": fact["round"], "global_turn": fact["global_turn"], "ms": fact["ms"],
            # CARRIED THROUGH, or `--analyze` cannot tell a game that finished
            # from one the clock stopped, and re-deriving a record would quietly
            # reintroduce the bug it exists to repair.
            "truncated": bool(fact.get("truncated")),
            "per_seat": {label.get(s, s): {k: v for k, v in p.items()
                                           if k not in ("life_by_turn", "damage_to_players_by_turn")}
                         for s, p in fact["per_seat"].items()}}


def analyze_logs(log_texts, label, commanders=None):
    """All games across a run's JVM logs → (facts, aggregate). `label` maps Forge seat
    labels (Ai(k)-<name>) to slugs; the first seat in `label` order is ours.
    `commanders` maps the same labels to commander name(s); omit it and every
    commander-damage key is absent rather than zero."""
    facts = []
    for text in log_texts:
        for g in parse_games(text):
            facts.append(game_facts(g, commanders))
    slug_label = next(iter(label)) if label else None
    return facts, aggregate(facts, slug_label, label, commanders)
