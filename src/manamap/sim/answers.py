"""WHAT OUR SEAT ANSWERED WITH — answers, counterspells and protection, per game, from a
run's logs (2026-10-21). The CLASS of card cast, by corpus role; `parse`'s
`interaction_cast` / `interaction_received` are the other half — what was AIMED at an
opponent — and this module does not duplicate them.

The pilot asked for an A/B of sharknado against finisher-v1 that proves the branch "has
answers, interactions, resolve" as well as speed and power. Forge's record carried
`kills_by_ability` (activated abilities only) and `interaction_cast` (targeting), and
targeting cannot see a counterspell's class, a sweeper's or a
protection spell's. A branch that adds Prismari Command and Loran's Escape needs both. This module is that reading, as counts a reader can check against the log.

A card's class comes from the corpus roles (`data/card_roles.json`, the analysis step's
artifact — the same vocabulary `deck-audit` reads), never from a list written here:

    answer       removal:spot / bounce / sweeper / edict / fight / tax, counterspell,
                 and removal:damage ON A NONCREATURE CARD (Collective Defiance is an
                 answer; Brallin's discard ping is a payoff, not an answer)
    counter      counterspell (also an answer; counted apart because it is the one
                 class whose use is a stack decision)
    protection   protection:* (Boros Charm, Greaves, Boots, Loran's Escape)

Per game, our seat:  answers_cast, counters_cast, protection_cast — CASTS and
ACTIVATIONS of a card in the class (an equipment's equip is an activation of a
protection card and counts: that is the AI choosing to protect).

A cast is a choice to interact, not a proof the answer hit; whether it removed the right
threat is not in the log in a form this reads. `from_logs` returns None when the list
holds no card of any class: absent means absent.
"""

import json

from manamap.sim import parse as sim_parse
from manamap.sim.parse import mean_ci

SINCE = "2026-10-21"

ANSWER_ROLES = {"removal:spot", "removal:bounce", "removal:sweeper", "removal:edict",
                "removal:fight", "removal:tax", "counterspell"}

LIMITS = [
    "A class is the corpus role (data/card_roles.json): answer = spot, bounce, sweeper, "
    "edict, fight, tax removal or a counterspell, plus damage removal on a noncreature "
    "card; protection = any protection:* role. A card the role patterns do not classify "
    "is in no class.",
    "Counts CASTS and ACTIVATIONS — the AI choosing to interact — never whether the "
    "answer hit the right threat.",
    "An equip activation of Greaves or Boots counts as protection: it is the AI moving "
    "its shroud onto a creature.",
]


def _roles():
    from manamap.config import DATA_DIR
    path = DATA_DIR / "card_roles.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("roles") or {}


def classes(deck_doc, roles=None):
    """{name: set of classes} for every card in a deck's cards.json that has one."""
    roles = _roles() if roles is None else roles
    if roles is None:
        return None
    out = {}
    for card in (deck_doc or {}).get("cards") or []:
        name = card["name"]
        rr = set(roles.get(name) or roles.get(name.split(" // ")[0]) or [])
        creature = "Creature" in (card.get("type_line") or "")
        got = set()
        if rr & ANSWER_ROLES or ("removal:damage" in rr and not creature):
            got.add("answer")
        if "counterspell" in rr:
            got.add("counter")
        if any(r.startswith("protection") for r in rr):
            got.add("protection")
        if got:
            out[name.split(" // ")[0].strip()] = got
    return out


def _clean(name):
    return sim_parse._CARD_ID_TAIL.sub("", name or "").split(" (")[0].strip()


def per_game(game, label, cls):
    row = {"answers_cast": 0, "counters_cast": 0, "protection_cast": 0}
    for ev in game["events"]:
        if ev.get("kind") not in ("cast", "activated") or ev.get("seat") != label:
            continue
        got = cls.get(_clean(ev.get("what")))
        if not got:
            continue
        row["answers_cast"] += "answer" in got
        row["counters_cast"] += "counter" in got
        row["protection_cast"] += "protection" in got
    return row


def from_logs(log_texts, ours, deck_doc, roles=None):
    """The record block for our seat, or None when no card in the list has a class
    (or the roles artifact is not on disk — said in the block's absence, not as zeros)."""
    cls = classes(deck_doc, roles)
    if not cls:
        return None
    games = []
    for text in log_texts:
        for game in sim_parse.parse_games(text):
            label = next((s for s in game["seats"] if s.endswith(f"-{ours}")), None)
            if label is not None:
                games.append(per_game(game, label, cls))
    return aggregate(games, ours, cls)


def aggregate(games, ours, cls):
    def block(key):
        vals = [g[key] for g in games]
        return {"mean_ci": mean_ci(vals) if len(vals) >= 2 else None, "total": sum(vals)}
    return {"seat": ours, "games": len(games), "since": SINCE,
            "cards": {k: sorted(v) for k, v in sorted(cls.items())},
            "answers_cast": block("answers_cast"),
            "counters_cast": block("counters_cast"),
            "protection_cast": block("protection_cast"),
            "per_game": games, "limits": LIMITS}


def validate(block, games_completed):
    """Form only; the logs re-derive it where they exist."""
    want = {"seat", "games", "since", "cards", "answers_cast", "counters_cast",
            "protection_cast", "per_game", "limits"}
    if not isinstance(block, dict) or set(block) != want:
        return ["answers: wrong shape"]
    errors = []
    if block["games"] != games_completed:
        errors.append(f"answers.games {block['games']} != games_completed {games_completed}")
    if len(block["per_game"]) != block["games"]:
        errors.append("answers.per_game does not hold one row per game")
    for key in ("answers_cast", "counters_cast", "protection_cast"):
        if block[key]["total"] != sum(g[key] for g in block["per_game"]):
            errors.append(f"answers.{key}.total is not the sum of its games")
    if any(g["counters_cast"] > g["answers_cast"] for g in block["per_game"]):
        errors.append("answers: more counterspells than answers in a game")
    return errors
