"""WHY the commander leaves the battlefield, from the games already played.

The cheapest rigorous answer is usually not a faster experiment — it is a
smaller question, asked of logs already on disk. Hexproof and shroud stop
TARGETED removal and nothing else, so the only empirical part of "is Lightning
Greaves worth a slot" is what share of the problem targeting is. That is a
decomposition, not an A/B, and it costs no games at all.

Measured on heliod, 220 games, 78 departures:

    39  50%  MASS      three or more permanents left the same turn
    24  31%  TARGETED  hexproof and shroud stop this
    15  19%  other / unattributed

    removal rate 78/173 resolutions = 0.451; with hexproof ~0.312

MASS IS DETECTED BY COINCIDENCE, NOT BY NAME. A first cut matched sweeper card
names and left 35% "unattributed" — and reading those showed several permanents
leaving in the same window, which is a wipe whatever named it. Counting how many
OTHER permanents left on the same turn catches the wipe the name list has never
heard of, and it is the property that actually decides whether hexproof helps.

Every rate carries its interval. `other` is reported and never redistributed:
attributing it proportionally would turn nineteen percent of unknown into
confident percentage points.
"""

import glob
import re

from manamap.config import DECKS_DIR
from manamap.sim.parse import wilson

#: Three or more permanents leaving on one turn is a wipe. Two can be a trade in
#: combat; the threshold is the smallest number that is not ordinary combat.
MASS_THRESHOLD = 3

_TURN = re.compile(r"^Turn: Turn (\d+) ")
_ANY_LEAVES = re.compile(r"^Zone Change: .+ was put into (?:Graveyard|Exile) from Battlefield")
_TARGETED = re.compile(r"targeting \[[^\]]*\b%s", re.I)
_COMBAT = re.compile(r"deals \d+ combat damage to ")


def _logs(slug):
    out = []
    for root in (DECKS_DIR / slug / "sim" / "logs",
                 DECKS_DIR / slug / "experiments" / "logs"):
        # `Path / "*/"` normalises the trailing separator away, so the glob
        # returned directories without one and the join below produced
        # `.../rundirpart-*.log`. Zero logs, silently.
        for d in sorted(root.glob("*")):
            if not d.is_dir():
                continue
            out += [str(f) for f in sorted(d.glob("*part-*.log"))
                    if "b-part" not in f.name]
    return out


def commander_departures(slug, commander):
    """Decompose every departure of `commander` from the battlefield.

    `b-part-*.log` is EXCLUDED: it is an experiment's arm B, a different list,
    and folding it in would decompose two decks as one.
    """
    face = re.escape(str(commander).split(" // ")[0])
    faces = "|".join(re.escape(f.strip()) for f in str(commander).split(" // "))
    resolved_re = re.compile(rf"^Resolve Stack: (?:{faces}) - Creature")
    left_re = re.compile(rf"^Zone Change: (?:{faces}) \(\d+\) was put into (\w+) from Battlefield")
    targeted_re = re.compile(rf"targeting \[[^\]]*(?:{faces})", re.I)
    combat_re = re.compile(rf"deals \d+ combat damage to (?:{faces})")

    counts = {"mass": 0, "targeted": 0, "combat": 0, "other": 0}
    zones = {"Graveyard": 0, "Exile": 0}
    resolutions = games = 0
    # TWO DENOMINATORS, AND THEY ARE DIFFERENT QUESTIONS. A commander can
    # resolve twice in one game and be removed twice, so "how often does he get
    # removed when he lands" (EVENTS) and "in how many games is he removed at
    # least once" (GAMES) are not the same number. Measured here: 0.454 of
    # resolutions against 0.412 of games he resolved in.
    #
    # Both are reported because a protection spell answers the EVENT — each time
    # he lands, what are the odds — while a pilot planning a game wants the
    # GAME. Reporting one under the other's label is what happened: "removed in
    # 45% of the games he lands" was the event rate wearing the game rate's
    # words, and a subagent caught it.
    games_landed = set()
    games_lost_him = set()
    game_no = 0
    for path in _logs(slug):
        on_bf = False; window = []; leaves = 0
        for line in open(path, errors="ignore"):
            line = line.rstrip()
            m = _TURN.match(line)
            if m:
                if int(m.group(1)) == 1:
                    games += 1; game_no += 1
                leaves = 0; window = []
                continue
            if _ANY_LEAVES.match(line):
                leaves += 1
            if resolved_re.match(line):
                resolutions += 1; on_bf = True; window = []
                games_landed.add((path, game_no))
                continue
            if on_bf:
                m = left_re.match(line)
                if m:
                    on_bf = False
                    games_lost_him.add((path, game_no))
                    zones[m.group(1)] = zones.get(m.group(1), 0) + 1
                    w = " | ".join(window[-14:])
                    if leaves >= MASS_THRESHOLD:
                        counts["mass"] += 1
                    elif targeted_re.search(w):
                        counts["targeted"] += 1
                    elif combat_re.search(w):
                        counts["combat"] += 1
                    else:
                        counts["other"] += 1
                    continue
                window.append(line)
    total = sum(counts.values())
    out = {"slug": slug, "commander": commander, "games": games,
           "resolutions": resolutions, "departures": total, "zones": zones,
           "causes": {}}
    for k, v in counts.items():
        lo, hi = wilson(v, total) if total else (None, None)
        out["causes"][k] = {"n": v,
                            "share": round(v / total, 3) if total else None,
                            "ci95": [lo, hi]}
    if resolutions:
        lo, hi = wilson(total, resolutions)
        out["removal_rate"] = {
            "rate": round(total / resolutions, 3), "ci95": [lo, hi],
            "n": resolutions,
            "basis": "EVENTS: departures per resolution. He can land and be "
                     "removed more than once in a game."}
    if games_landed:
        k, n = len(games_lost_him), len(games_landed)
        lo, hi = wilson(k, n)
        out["games_losing_him"] = {
            "rate": round(k / n, 3), "ci95": [lo, hi], "n": n, "k": k,
            "basis": "GAMES: he was removed at least once, over games he landed in."}
    return out


#: WHAT A CARD CAN ACTUALLY STOP, and the four answers are not interchangeable.
#:
#: PHASING is the strongest: phase out in response and a targeted spell fizzles
#: for want of a legal target, and a wipe finds nothing to destroy. Both classes.
#: PROTECTION FROM EVERYTHING covers both, for the same reason.
#: INDESTRUCTIBLE covers DESTROY only — nothing about exile, bounce or
#: sacrifice, and 33 of heliod's 119 departures were EXILE. Its share is
#: discounted by the graveyard fraction rather than credited in full. Getting
#: this wrong would price Mithril Coat identically to Guardian of Faith.
#: HEXPROOF and SHROUD stop targeting and nothing else. A wipe does not target.
#:
#: THE KEYWORD MUST BE GRANTED TO A CREATURE YOU CONTROL. The first cut matched
#: the word anywhere in the text, and on a real 46-card pile that scored:
#:
#:   Thassa, God of the Sea    "Indestructible"           — the GOD is, not yours
#:   Aegis of the Gods         "You have hexproof"        — the PLAYER, not a creature
#:   Dragonlord Ojutai         "Ojutai has hexproof"      — itself
#:   Darksteel Sentinel        "Indestructible", 6 mana   — itself
#:   Defender of Law           "Protection from red"      — itself
#:   Seht's Tiger              "you gain protection"      — the player again
#:
#: Six of thirteen top-ranked cards were protecting something other than the
#: commander, in a ranking about to choose what to buy. A keyword is only an
#: answer here if it lands on a creature you control.
#: PERMANENTS, not just creatures. Teferi's Protection — the card already IN
#: this deck and the reference case for the whole column — says "All permanents
#: you control phase out", and a creature-only grant list scored the deck's own
#: best protection card at nothing. Caught by checking the new rule against a
#: card whose answer was already known, which is the only way that class of
#: false NEGATIVE ever surfaces.
_GRANT = (r"(?:target |another target |other |each |all )?"
          r"(?:creature|permanent)s? you control"
          r"|equipped creature|enchanted creature|commanders? you control"
          r"|target creature")
_SELF_OR_PLAYER = re.compile(r"^\s*(?:you (?:have|gain)|players? (?:have|gain))", re.I)

_PHASES = re.compile(r"phase(?:s)? out|phasing", re.I)
_PROT_ALL = re.compile(r"protection from everything", re.I)
_INDESTRUCTIBLE = re.compile(r"indestructible", re.I)
_UNTARGETABLE = re.compile(r"hexproof|shroud|can't be the target|protection from", re.I)


def _sentences(text):
    return [x.strip() for x in re.split(r"(?<=[.;])\s+|\n", str(text or "")) if x.strip()]


def _granted(text, keyword_re):
    """Does this card put `keyword_re` onto a creature YOU CONTROL?

    Sentence-scoped, because a card can carry a keyword itself in one line and
    grant a different one in another — Spectacular Spider-Man has Flash on line
    one and grants hexproof and indestructible on line three.
    """
    for sent in _sentences(text):
        if not keyword_re.search(sent):
            continue
        if _SELF_OR_PLAYER.match(sent):
            continue                                  # "You have hexproof"
        if re.search(_GRANT, sent, re.I):
            return True
    return False


def what_a_card_answers(text):
    """(stops_targeted, stops_mass, destroy_only) from the card's own text."""
    t = str(text or "")
    if _granted(t, _PHASES) or _granted(t, _PROT_ALL):
        return True, True, False
    if _granted(t, _INDESTRUCTIBLE):
        return True, True, True
    if _granted(t, _UNTARGETABLE):
        return True, False, False
    return False, False, False
def coverage(text, decomposition):
    """The SHARE of this deck's measured commander losses a card could address.

    AN UPPER BOUND, and labelled as one in the payload. That a card CAN stop a
    class of removal is not a claim that it was on the battlefield, untapped and
    held up at the moment it was needed. What it does is rank a pile against a
    MEASURED failure instead of against a taste.
    """
    tgt, mass, destroy_only = what_a_card_answers(text)
    if not (tgt or mass):
        return None
    c = decomposition["causes"]
    share = (c["targeted"]["share"] or 0) * tgt + (c["mass"]["share"] or 0) * mass
    note = None
    if destroy_only:
        z = decomposition.get("zones") or {}
        total = sum(z.values()) or 1
        keep = z.get("Graveyard", 0) / total
        share *= keep
        note = (f"indestructible answers DESTROY only; {z.get('Exile', 0)} of "
                f"{total} departures were exile, so the share is discounted to "
                f"the {keep:.0%} that went to the graveyard")
    kinds = [k for k, on in (("targeted", tgt), ("mass", mass)) if on]
    out = {"stops": kinds, "share_of_losses": round(share, 3),
           "basis": f"{decomposition['departures']} departures over "
                    f"{decomposition['games']} games",
           "is_upper_bound": True}
    if note:
        out["caveat"] = note
    return out
