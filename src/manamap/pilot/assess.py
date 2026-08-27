"""Pilot: triage a pile of cards against one deck, before spending anything.

    manamap pilot assess <slug> [--branch N] --pool <file|library|->

THE READING THAT HAS TO HAPPEN BEFORE THE MEASUREMENT. `candidates` substitutes
each card and re-measures, which is the right answer for a card the model can
see — and most cards a pilot brings are not that. Done by hand on a 21-card pile
the triage took four passes and turned up three things no simulation would have
said: half the pile was combat-gated for a deck built to win without an attack
step, one card needed a creature type the deck does not run, and the two genuinely
best cards were invisible to every model here. Each of those is cheap to check
and expensive to miss.

WHAT IT ASKS, IN THE ORDER THAT KILLS CANDIDATES FASTEST:

  1. Is it a real card, and is it already in the list?
  2. Is it LEGAL — inside the commander's colour identity?
  3. What does it cost, and what does it do?
  4. Is it GATED on something this deck does not do? A combat trigger in a deck
     whose thesis is winning without attacking is an efficient card on the wrong
     axis, and no amount of simulation will say so.
  5. Is it ON-AXIS — does it feed a component the deck DECLARES it needs?
  6. CAN ANY MODEL HERE SEE IT? A card that taxes what opponents do is worth
     exactly zero in a solitaire goldfish, by construction. Recommending it is
     legitimate; pretending the recommendation is measured is not.
  7. Is there something already in the list doing the same job for less?

NOTHING HERE SCORES A CARD. The output is a reading with the reason attached, so
a pilot can disagree with any line of it. The measured half stays in
`candidates`, which this names as the next step for the cards worth measuring.
"""

import json
import re

from manamap.pilot.common import (
    deck_file, expand_faces, load_deck_cards, load_json)
from manamap.pilot import goldfish
from manamap.sim import pod_behaviour

#: What a card is gated on — the question "will this deck ever turn it on".
COMBAT, OPPONENT, DEATH, ACTIVATED, RECURRING, ONESHOT = (
    "combat", "opponent", "death", "activated", "recurring", "one-shot")

_GATES = (
    (COMBAT, re.compile(r"combat damage|attacks|becomes? blocked|deals damage to a player", re.I)),
    # OPPONENT MEANS OPPONENT AGENCY, NEVER OPPONENT AS A TARGET. The first cut
    # matched "each opponent" anywhere, which is the wording every drain payoff
    # in the game uses — it classed 8 of this branch's 95 cards as gated on the
    # pod, and 5 were wrong: Reckless Fireweaver (the branch's OWN drain, gated
    # on your artifacts entering), Terror of the Peaks, Revel in Riches,
    # Weftstalker Ardent, and Exotic Orchard, which is a land. A card is gated on
    # the opponents only when THEY have to do something first.
    (OPPONENT, re.compile(
        r"whenever an opponent|whenever a player|whenever another player|"
        r"each opponent (?:may|chooses|who)|an opponent (?:casts|draws|plays|"
        r"sacrifices|attacks|gains|searches)|opponent who (?:drew|had|controls)",
        re.I)),
    (DEATH, re.compile(r"\bdies\b|would die|leaves the battlefield|put into a graveyard", re.I)),
    (ACTIVATED, re.compile(r"\{\d*\}?\s*,?\s*\{T\}|Sacrifice this|as an additional cost", re.I)),
    (RECURRING, re.compile(r"[Ww]henever|[Aa]t the beginning")),
)

# TRIAGE IS DELIBERATELY BROADER THAN THE MODEL, and the two shared halves are
# IMPORTED so they cannot drift again. This pattern once matched five wordings
# while `goldfish._TRE_EXTRA_RE` matched one, which is how the goldfish came to
# price 2 of the 8 multipliers ur-dragon's treasure branch declares. The extra
# terms here are on purpose: Panharmonicon and Academy Manufactor ARE
# multipliers for a deck and are not ones this model can price, so triage should
# surface them and the simulation should name them blind.
_MULTIPLIER = re.compile(
    "|".join((goldfish.TREASURE_BONUS_RE.pattern,
              goldfish.TOKEN_DOUBLER_RE.pattern,
              r"twice that many", r"those tokens plus",
              r"additional Treasure token", r"plus a Treasure, Clue",
              r"triggers an additional time")), re.I)
_TREASURE = re.compile(r"[Cc]reate.{0,45}Treasure")


#: Real creature types, learned from the corpus rather than listed here — the
#: list changes every set and a stale copy would silently stop catching a tribe.
#: Artifact token names that are also creature types. In rules text they mean the
#: token, so they are never read as a tribal requirement.
TOKEN_NAMES = {"Treasure", "Clue", "Food", "Blood", "Powerstone", "Map",
               "Junk", "Incubator", "Gold", "Shard", "Lander"}


def _creature_types():
    """The corpus's real creature types. The scan lives in `analysis/common` so
    this and `power_creep`'s tribal gate cannot drift — the triage that warns a
    pilot and the index that ranks cards for them must agree on what a tribe is.
    """
    from manamap.analysis.common import creature_types
    from manamap.pilot import card_pool
    return creature_types(card_pool.load_frame())


class _Lazy(set):
    """The type set is a corpus scan; only pay for it if a card mentions a tribe."""

    def __contains__(self, item):
        if not self:
            self.update(_creature_types())
        return set.__contains__(self, item)


CREATURE_TYPES = _Lazy()

#: English plurals for the handful of tribes that are not `+s`.
_IRREGULAR = {"Dwarf": "Dwarves", "Elf": "Elves", "Wolf": "Wolves",
              "Thief": "Thieves", "Scout": "Scouts"}
#: Creature types whose plural is the singular. "needs Fishs" is a small thing
#: that makes a correct verdict read like broken software, which is the whole
#: cost of a sentence a tool says out loud.
_INVARIANT = {"Fish", "Sheep", "Moose", "Aetherborn", "Efreet", "Djinn"}


def _plural(t):
    if t in _INVARIANT:
        return t
    if t.endswith(("s", "x", "z", "ch", "sh")):
        return t + "es"
    return _IRREGULAR.get(t, t + "s")


def gate_of(text):
    for name, pat in _GATES:
        if pat.search(text or ""):
            return name
    return ONESHOT


def job_of(text, roles):
    if _MULTIPLIER.search(text or ""):
        return "multiplier"
    if _TREASURE.search(text or ""):
        return "treasure"
    for r in roles or []:
        head = r.split(":", 1)[0]
        if head in ("removal", "draw", "ramp", "protection", "tutor", "wincon"):
            return head
    return "other"


def assess(slug, pool, branch=None):
    from manamap.pilot import card_pool, candidates as _cand, deck_branch
    from manamap.pilot.common import load_card_roles
    frame = card_pool.load_frame()
    by = {}
    for _, r in frame.iterrows():
        if r["name"] not in by:
            by[r["name"]] = r
    roles = load_card_roles()
    doc = load_deck_cards(slug, branch)
    held = {c["name"] for c in doc["cards"]}
    identity = {c for x in doc["cards"] for c in (x.get("color_identity") or [])}
    types = {t for c in doc["cards"] for t in str(c.get("type_line") or "").split()}

    declared = _cand._declared_cards(slug, branch)
    targets = (load_json(deck_file(slug, "goldfish_targets.json", branch)) or {}).get("targets") or []
    # What the deck SAYS it needs, so "on axis" is the deck's own claim rather
    # than a taste of mine.
    axes = {t["label"]: {c for g in (t.get("need") or []) for c in (g.get("any_of") or [])}
            for t in targets}
    # The cheapest card already in the list doing each job, for the value question.
    cheapest = {}
    for c in doc["cards"]:
        if "Land" in (c.get("type_line") or ""):
            continue
        j = job_of(c.get("oracle_text"), roles.get(c["name"]))
        mv = float(c.get("cmc") or 0)
        if j not in cheapest or mv < cheapest[j][1]:
            cheapest[j] = (c["name"], mv)

    try:
        src = {r["name"]: r for r in deck_branch.source(slug, branch)["cards"]}
    except Exception:
        src = {}

    rows = []
    for name in pool:
        r = by.get(name)
        if r is None:
            # A DOUBLE-FACED CARD IS NAMED TWO WAYS AND BOTH ARE CORRECT: the
            # library holds `A // B`, a pasted list may hold either face. The
            # same seam `deck_branch._canonical` closes, one module over.
            hit = next((k for k in by
                        if name in expand_faces(k) or k in expand_faces(name)), None)
            r = by.get(hit) if hit else None
            if r is not None:
                name = hit
        if r is None:
            rows.append({"card": name, "verdict": "not in the corpus — check the name"})
            continue
        text = str(r.get("oracle_text") or "")
        ci = r.get("color_identity")
        ci = {x.strip() for x in ci.split(",") if x.strip()} if isinstance(ci, str) else set()
        mv = float(r.get("cmc") or 0)
        job = job_of(text, roles.get(name))
        gate = gate_of(text)
        row = {
            "card": name, "mv": int(mv), "job": job, "gate": gate,
            "in_list": name in held,
            "legal": ci <= identity,
            "on_axis": [lbl for lbl, cards in axes.items() if name in cards],
            "declared": name in declared,
            "source": (src.get(name) or {}).get("state"),
        }
        # Does it need a creature type the deck does not run? (Magda wants
        # Dwarves.) MATCHED AGAINST REAL CREATURE TYPES, not a capital-letter
        # pattern: "Treasures you control" is not a tribe, and the first cut
        # reported Alchemist's Talent as needing "Treasures" the deck runs none
        # of — a confident sentence about a card that has no tribal text at all.
        need = {t for t in re.findall(r"\b([A-Z][a-z]+)s?\b(?= you control)", text)
                if t in CREATURE_TYPES}
        row["needs_type"] = sorted(need - types)
        # HOW OFTEN WOULD THIS HAVE FIRED AGAINST THE POD? A goldfish cannot
        # answer it, but Forge already played those turns — see pod_behaviour.
        row["pod_rate"] = pod_behaviour.rate_for(text) if gate == OPPONENT else None
        # WHICH CHANNELS OF THE MODEL CAN SEE THIS CARD AT ALL. Step 6 of the
        # order above, and until now it only covered opponent-gated cards. It
        # is the same question for every card: a sweep can price what the
        # goldfish simulates and can price everything else ONLY by displacement,
        # which reads as noise and costs a full run per card to say so. Measured
        # on a 29-card pool `close` proposed: 14 were invisible — Mana
        # Reflection doubles MANA not tokens, Oath of Lieges and Greener
        # Pastures are land-matters cards the centroid pulled in on similar
        # phrasing. Naming them here is 14 runs not spent.
        row["model_sees"] = _channels(
            {"name": name, "oracle_text": text, "cmc": mv,
             "type_line": str(r.get("type_line") or ""),
             "mana_cost": str(r.get("mana_cost") or ""),
             "power": r.get("power"), "toughness": r.get("toughness")})
        eq = cheapest.get(job)
        row["cheaper_than_ours"] = bool(eq and mv < eq[1])
        row["ours_cheapest"] = f"{eq[0]} (mv{int(eq[1])})" if eq else None
        rows.append(dict(row, verdict=_verdict(row, slug, branch)))
    return {"slug": slug, "branch": branch, "cards": rows,
            "axes": sorted(axes), "identity": sorted(identity)}


def _channels(card):
    """The goldfish's own reading of a card, as a list of channel names."""
    try:
        got = goldfish.classify(card)
    except Exception:                              # pragma: no cover - defensive
        return None
    out = []
    if got.get("treasure_doubler"):
        out.append("treasure doubler")
    if got.get("treasure_bonus"):
        out.append("treasure adder")
    if got.get("treasure_trigger") not in (None, "unmodelled"):
        out.append(f"treasure on {got['treasure_trigger']}")
    if got.get("produces"):
        out.append("mana")
    if got.get("bodies"):
        out.append("body")
    if got.get("tutor"):
        out.append("tutor")
    return out


def _verdict(row, slug, branch):
    if row["in_list"]:
        return "already in the list"
    if not row["legal"]:
        return "OUTSIDE the commander's colour identity — cannot be played here"
    if row["needs_type"]:
        return (f"needs {' or '.join(_plural(t) for t in row['needs_type'])} "
                f"and this deck runs none — dead as written")
    if row["gate"] == COMBAT:
        return ("combat-gated: it needs a creature to connect. Efficient in a "
                "vacuum, wrong axis for a deck that wins without attacking")
    if row["gate"] == OPPONENT:
        # The goldfish still cannot see it. But the QUESTION — how often would
        # this have fired at my table — is answerable from the sim runs, and a
        # measured frequency beats "unmeasurable" by the whole width of the
        # decision. It reversed a recommendation the first time it was asked.
        est = row.get("pod_rate")
        head = ("opponent-gated: no goldfish figure can price it "
                "(a solitaire model has no opponents)")
        if not est:
            return head + " — judge it by reading the card, and say so"
        if est["per_round"] is None:
            return (f"{head}, but the trigger is bounded: {est['bound']} "
                    f"({est['basis']})")
        # A per-round frequency is not throughput when one firing touches every
        # seat — saying only "1.0" ranks an upkeep tax below a per-draw one that
        # it actually matches. The scaling rides with the number, never instead.
        each = (" — and each firing resolves against EACH opponent, so at this "
                "table one trigger is worth up to 3"
                if est.get("scales_with_opponents") else "")
        return (f"{head}, but it would fire about {est['per_round']} times a "
                f"round against your pod{each} — {est['basis']}")
    if row.get("model_sees") == []:
        return ("NO CHANNEL OF THE MODEL CAN SEE THIS. It is legal and on-axis "
                "by name, but the goldfish simulates none of what it does, so a "
                "sweep can only price it by displacement — which reads as noise "
                "and costs a full run to say so. Judge it by reading the card")
    if row["job"] in ("treasure", "multiplier") and row["cheaper_than_ours"]:
        return (f"cheaper than anything in the list doing that job "
                f"({row['ours_cheapest']}) — worth measuring")
    return "on-colour and playable — measure it if it feeds a declared component"


MEASURABLE = ("cheaper than", "measure it")


def main(args):
    from manamap.pilot.candidates import read_pool
    pool = read_pool(getattr(args, "pool", None), args.slug)
    if not pool:
        raise SystemExit("no pool — `--pool <file>`, `--pool library`, or `--pool -`")
    doc = assess(args.slug, pool, branch=getattr(args, "branch", None))
    if getattr(args, "json", False):
        print(json.dumps(doc, indent=1)); return
    where = doc["slug"] + (f"/{doc['branch']}" if doc.get("branch") else "")
    rows = doc["cards"]
    print(f"\nASSESSMENT — {len(rows)} card(s) against {where}\n")
    order = {"already in the list": 3}
    rows = sorted(rows, key=lambda r: (order.get(r["verdict"], 0), r.get("mv", 99)))
    for r in rows:
        if "mv" not in r:
            print(f"  {r['card'][:30]:30}   {r['verdict']}"); continue
        tag = "·".join(x for x in (r["job"], r["gate"]) if x)
        own = ("" if r["in_list"] else
               f"  [{r['source']}]" if r.get("source") else "")
        print(f"  mv{r['mv']:<2} {r['card'][:30]:30} {tag:22}{own}")
        print(f"       {r['verdict']}")
        if r["on_axis"]:
            print(f"       feeds: {', '.join(x[:44] for x in r['on_axis'])}")
    worth = [r for r in rows if any(m in r.get("verdict", "") for m in MEASURABLE)]
    print(f"\n  {len(worth)} of {len(rows)} worth measuring:")
    for r in worth:
        print(f"    manamap pilot candidates {args.slug}"
              + (f" --branch {doc['branch']}" if doc.get("branch") else "")
              + f" --pool library --axis engine_online_3")
        break
    if not worth:
        print("    none — the pile is off-axis or unmeasurable here")
