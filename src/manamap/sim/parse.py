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

# Seat labels are `Ai(k)-<meta name>` and meta names are kebab (`mm-<slug>`); `\S+`
# swallowed the comma after an `Activator:` tag and mis-attributed every token resolution.
_SEAT = r"(Ai\(\d+\)-[\w-]+)"
_PERM = r"(.+?) \((\d+)\)"                    # "Cat Token (414)" → name, id

RX = {
    "turn":      re.compile(r"^Turn: Turn (\d+) \(" + _SEAT + r"\)"),
    "phase":     re.compile(r"^Phase: " + _SEAT + r"'s? (.*)$"),
    "mana":      re.compile(r"^Mana: " + _PERM + r" - (.*)$"),
    "mulligan":  re.compile(r"^Mulligan: " + _SEAT + r" has kept a hand of (\d+) cards"),
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
    "result":    re.compile(r"^Game Result: Game (\d+) ended in (\d+) ms"),
}
_TOKEN = re.compile(r"\bToken$")
_CREATE = re.compile(r"\bcreates?\b.*\btokens?\b", re.I)
_LOSES_LIFE = re.compile(r"\b(loses?|lose) (that much |\d+ |X )?life\b|\bdrain", re.I)
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


def _new_game():
    return {"seats": [], "turn": 0, "active": None, "events": [], "owner": {},
            "mulligan": {}, "outcome": {"winner": None, "won_by": None, "round": None,
                                        "global_turn": None, "draw": False, "lost": {}, "ms": None}}


def parse_games(text):
    """Split one JVM log into games and parse each into an event list + outcome."""
    games, g, last_turn = [], None, None
    for raw in text.splitlines():
        line = raw.rstrip()
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
            o["winner"], o["won_by"] = m.group(1), m.group(2).rstrip("."); continue
        m = RX["lost"].match(line)
        if m:
            o["lost"][m.group(1)] = m.group(2).rstrip("."); continue
        m = RX["result"].match(line)
        if m:
            o["ms"] = int(m.group(2)); continue
    return [x for x in games if x["events"] or x["outcome"]["winner"]]


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

def game_facts(g):
    seats = list(g["seats"])
    per = {s: {"lands": 0, "casts": 0, "activations": 0, "triggers": 0,
               "combat_damage_dealt_to_players": 0, "combat_damage_taken": 0,
               "noncombat_damage_dealt_to_players": 0,
               "token_resolutions": 0, "tokens_observed": set(), "token_attackers": set(),
               "token_blockers": set(), "token_combat_damage_to_players": 0,
               "tokens_chumped": 0, "creatures_lost": 0,
               "life_by_turn": {}, "eliminated_turn": None, "eliminated_by": None,
               "eliminated_how": None,
               "first_attack_turn": None, "damage_to_players_by_turn": defaultdict(int)}
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
        elif k == "activated":
            per[ev["seat"]]["activations"] += 1
        elif k == "triggered":
            per[ev["seat"]]["triggers"] += 1
        elif k == "resolve":
            if ev["creates_token"] and ev["seat"] in per:
                per[ev["seat"]]["token_resolutions"] += 1
            if _LOSES_LIFE.search(ev["text"]) and ev["seat"] in per:
                last_cause = (ev["seat"], ev["turn"], "life loss")
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
        elif k == "zone" and ev["to"] == "Graveyard" and ev["from"] == "Battlefield":
            s = owner.get(ev["id"])
            if s in per:
                per[s]["creatures_lost"] += 1      # anything the seat was seen acting with
                if ev["id"] in blockers_this_turn and blockers_this_turn[ev["id"]][1] == ev["turn"]:
                    per[s]["tokens_chumped"] += 1
    # finalise
    for s, p in per.items():
        p["tokens_observed"] = len(p["tokens_observed"])
        p["token_attackers"] = len(p["token_attackers"])
        p["token_blockers"] = len(p["token_blockers"])
        p["damage_to_players_by_turn"] = dict(sorted(p["damage_to_players_by_turn"].items()))
        p["life_by_turn"] = dict(sorted(p["life_by_turn"].items()))
        d = p["combat_damage_dealt_to_players"]
        p["token_damage_share"] = round(p["token_combat_damage_to_players"] / d, 3) if d else None
    o = g["outcome"]
    return {"seats": seats, "winner": o["winner"], "won_by": o["won_by"], "round": o["round"],
            "global_turn": o["global_turn"], "ms": o["ms"], "lost": dict(o["lost"]),
            "mulligan": dict(g["mulligan"]), "per_seat": per}


# ── Aggregates over a run ───────────────────────────────────────────────────

def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion; (None, None) when n == 0."""
    if not n:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


def mean_ci(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "n": 0}
    n = len(xs)
    mean = sum(xs) / n
    if n < 2:
        return {"mean": round(mean, 2), "n": n}
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    half = 1.96 * sd / math.sqrt(n)
    return {"mean": round(mean, 2), "ci95": [round(mean - half, 2), round(mean + half, 2)], "n": n}


def aggregate(facts, slug_label, label):
    """`facts` is a list of game_facts; `slug_label` the Forge seat label of OUR deck;
    `label` maps Forge seat labels to slugs."""
    n = len(facts)
    seats = sorted({s for f in facts for s in f["seats"]})
    wins = Counter(f["winner"] for f in facts if f["winner"])
    out = {"games": n, "seats": {label.get(s, s): {} for s in seats}}
    for s in seats:
        name = label.get(s, s)
        k = wins.get(s, 0)
        lo, hi = wilson(k, n)
        ps = [f["per_seat"][s] for f in facts if s in f["per_seat"]]
        out["seats"][name] = {
            "wins": k, "win_rate": round(k / n, 3) if n else None, "win_rate_ci95": [lo, hi],
            "eliminated_turn": mean_ci([p["eliminated_turn"] for p in ps]),
            "eliminated_by": dict(Counter(label.get(p["eliminated_by"], p["eliminated_by"])
                                          for p in ps if p["eliminated_by"])),
            "eliminated_how": dict(Counter(p["eliminated_how"] for p in ps if p["eliminated_how"])),
            "lands": mean_ci([p["lands"] for p in ps]),
            "casts": mean_ci([p["casts"] for p in ps]),
            "combat_damage_dealt_to_players": mean_ci([p["combat_damage_dealt_to_players"] for p in ps]),
            "combat_damage_taken": mean_ci([p["combat_damage_taken"] for p in ps]),
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
        "ci95 is a Wilson interval for rates and a normal interval for means; both are "
        "meaningless below ~10 games. Seeded runs replay exactly; the interval is still the claim.",
    ]
    return out


def compact(fact, label):
    """A per-game row small enough to track: outcome + per-seat scalars, no curves."""
    return {"winner": label.get(fact["winner"], fact["winner"]), "won_by": fact["won_by"],
            "round": fact["round"], "global_turn": fact["global_turn"], "ms": fact["ms"],
            "per_seat": {label.get(s, s): {k: v for k, v in p.items()
                                           if k not in ("life_by_turn", "damage_to_players_by_turn")}
                         for s, p in fact["per_seat"].items()}}


def analyze_logs(log_texts, label):
    """All games across a run's JVM logs → (facts, aggregate). `label` maps Forge seat
    labels (Ai(k)-<name>) to slugs; the first seat in `label` order is ours."""
    facts = []
    for text in log_texts:
        for g in parse_games(text):
            facts.append(game_facts(g))
    slug_label = next(iter(label)) if label else None
    return facts, aggregate(facts, slug_label, label)
