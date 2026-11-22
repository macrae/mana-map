"""Pilot: the engine as a COMPUTED graph of typed events between cards.

    manamap pilot engine-graph <slug> [--branch B] [--json] [--write] [--abilities [CARD …]]

`docs/research/engine-model.md` is the argument; this is the code. Every card's
abilities are read by `ability_graph` (Forge's script, else the goldfish's readers);
an edge runs from a card that PRODUCES an event to a card whose trigger or cost
CONSUMES it, when the consumer's filter accepts the producer's subject; stocks
(HAND, BODIES, MANA, LIFE, GRAVEYARD) are nodes a card fills and a cost draws on;
replacements and statics SCALE an event kind or PROTECT a node. Stages are positions
in that flow (`STAGE_RULES`, first match wins, `also[]` keeps the rest); lines are
simple paths from an ignition to the terminal; redundancy and single points of
failure are graph properties; activation is `deck_audit._component`'s hypergeometric
odds over a consumed event's producers.

THREE TIERS, never blended. ◆ everything `build()` computes, byte-identical on
every run (`validate-engine` holds the committed file to a fresh one). ✓ a line whose
cards a checker-passed stack names (`engine_facts.verified_pairings`), a node Forge's
AI cast. ★ the `reading` — thesis, named lines, stage overrides with a why, the
critic's verdict — authored by the engineer and the pilot, stamped with the list it
was written for, COPIED FORWARD by `--write` and never recomputed. A card the parser
cannot read is named in `coverage`, never a silent leaf. Zero LLM calls.
"""
import json
from collections import Counter, OrderedDict
from itertools import combinations

from manamap.config import (ABILITY_GRAPH_VERSION, ENGINE_EVENTS, ENGINE_LINE_MAX_DEPTH,
                            ENGINE_MAX_LINES, ENGINE_STAGE_RULES, ENGINE_TERMINALS,
                            ENGINE_THIN_GROUP)
from manamap.pilot import ability_graph as ag
from manamap.pilot.common import (decklist_sha256, deck_dir, deck_file, load_deck_cards,
                                  load_json)

STAGE_LABEL = {"mana": "Mana", "ignition": "Ignition", "fuel": "Fuel", "fodder": "Fodder",
               "conversion": "Conversion", "output": "Output", "protection": "Protection", "wincon": "Win condition"}
TERMINAL = "TERMINAL"
OPPONENT = "OPPONENT"
TURN = "TURN"
STOCKS = ("HAND", "BODIES", "MANA", "LIFE", "GRAVEYARD")
#: What fills each stock and what draws on it (an event -> stock).
_FILLS = {"DRAW": "HAND", "TUTOR": "HAND", "ETB": "BODIES", "TOKEN": "BODIES", "MANA": "MANA",
          "TREASURE": "MANA", "LIFE_GAIN": "LIFE", "DIES": "GRAVEYARD", "MILL": "GRAVEYARD",
          "DISCARD": "GRAVEYARD"}
_DRAWS = {"CAST": "HAND", "RECURSION": "GRAVEYARD"}      # a consumer of these reads the stock
_COST_STOCK = {"BODIES": "DIES", "HAND": "DISCARD", "LIFE": "LIFE_LOSS"}
#: Keywords on a Pump or static that PROTECT rather than pump.
PROTECTIVE = ("Hexproof", "Indestructible", "Shroud", "Protection", "Ward", "Phasing")
#: Events the OPPONENT pseudo-node is assumed to produce at a table (authored, at no
#: rate; they enter no activation figure — docs/gotchas-bench.md's engine_lift rule).
OPPONENT_EVENTS = ("CAST", "DRAW", "ATTACK", "ETB", "DIES", "LIFE_GAIN", "LAND_DROP", "DAMAGE",
                   "TUTOR", "DISCARD", "COUNTER_ADD")
MANA_ONLY = {"MANA", "COST_REDUCTION", "LAND_DROP", "TAP", "UNTAP", "SELECTION"}
FUEL_EVENTS = {"DRAW", "TUTOR", "RECURSION", "SELECTION", "DISCOVER", "CASCADE", "TREASURE"}


# ── nodes ────────────────────────────────────────────────────────────────────

def _nodes(slug, branch):
    """The 99 + commanders (zone `command`), the tokens they make (zone `token`),
    then the pseudo-nodes. Returns (nodes, cards_doc)."""
    doc = load_deck_cards(slug, branch)
    rows = doc.get("cards") or []
    abilities = ag.deck_abilities(slug, branch)
    nodes = OrderedDict()
    for c in rows:
        name = c["name"]
        if name in nodes:
            nodes[name]["copies"] += int(c.get("quantity") or 1)
            continue
        ab = abilities[name]
        is_cmd = bool(c.get("is_commander"))
        types = ab["types"] or [t.strip() for t in (c.get("type_line") or "").split("—")[0].split()]
        node = {
            "id": name, "zone": "command" if is_cmd else "library", "copies": int(c.get("quantity") or 1),
            "types": [t for t in types if t not in ("Legendary", "Basic", "Snow", "Token")],
            "subtypes": ab["subtypes"] or [t.strip() for t in ((c.get("type_line") or "").split("—") + [""])[1].split()],
            "colors": ab["colors"] or _colors_from_cost(c.get("mana_cost")),
            "pt": ab["pt"] or (f"{c.get('power')}/{c.get('toughness')}" if c.get("power") else None),
            "mana_cost": ab["mana_cost"] or c.get("mana_cost"),
            "abilities": list(ab["abilities"]),
            "coverage": {"reader": ab["reader"], "partial": ab["partial"], "unmapped": ab["unmapped"],
                         "loose": ab["loose"], "ai_remove_deck": ab.get("ai_remove_deck")},
            "keywords": ab["keywords"],
        }
        _implicit(node)
        nodes[name] = node
    _token_nodes(nodes)
    return nodes, doc


def _colors_from_cost(cost):
    out = []
    for ch in str(cost or ""):
        if ch in "WUBRG" and ch not in out:
            out.append(ch)
    return out


def _implicit(node):
    """What every card does without a line of script saying so: a nonland is CAST
    (its own subject — Edgar reads the Vampire type off it); a permanent ENTERS; a
    land is a LAND_DROP; a creature that may attack ATTACKS and deals COMBAT
    damage to an opponent. Marked `implicit` so stage rules can set them aside."""
    types = node["types"]
    imp = []
    if "Land" not in types:
        imp.append({"event": "CAST", "actor": "you", "self": True})
    if any(t in types for t in ("Creature", "Artifact", "Enchantment", "Planeswalker", "Battle", "Land")):
        imp.append({"event": "ETB", "actor": "you", "self": True})
    if "Land" in types:
        imp.append({"event": "LAND_DROP", "actor": "you", "self": True})
    if "Creature" in types and not any(k.startswith("Defender") for k in node.get("keywords") or []):
        imp.append({"event": "ATTACK", "actor": "you", "self": True})
        imp.append({"event": "COMBAT_DAMAGE", "actor": "opponent", "self": True})
    if imp:
        node["abilities"].append({"kind": "implicit", "consumes": None, "costs": [], "produces": imp,
                                  "optional": False, "zone": "battlefield", "text": ""})


def _token_nodes(nodes):
    from manamap.sim import forge_cards
    stems = []
    for n in list(nodes.values()):
        for a in n["abilities"]:
            for p in a.get("produces") or []:
                if p.get("token") and p["token"] not in stems:
                    stems.append(p["token"])
    for stem in stems:
        text = forge_cards.token_script(stem)
        ab = ag.card_abilities(stem, text) if text else None
        name = f"token:{stem}"
        node = {"id": name, "zone": "token", "copies": 0, "is_token": True,
                "types": [t for t in (ab["types"] if ab else []) if t not in ("Legendary", "Token")],
                "subtypes": ab["subtypes"] if ab else [], "colors": ab["colors"] if ab else [],
                "pt": ab["pt"] if ab else None, "mana_cost": None,
                "abilities": list(ab["abilities"]) if ab else [],
                "coverage": {"reader": "forge" if ab else "none", "partial": bool(ab and ab["partial"]),
                             "unmapped": ab["unmapped"] if ab else [], "loose": ab["loose"] if ab else [],
                             "ai_remove_deck": None},
                "keywords": ab["keywords"] if ab else []}
        # A token is never cast; it enters, and a creature token attacks.
        imp = [{"event": "ETB", "actor": "you", "self": True}]
        if "Creature" in node["types"]:
            imp += [{"event": "ATTACK", "actor": "you", "self": True},
                    {"event": "COMBAT_DAMAGE", "actor": "opponent", "self": True}]
        node["abilities"].append({"kind": "implicit", "consumes": None, "costs": [], "produces": imp,
                                  "optional": False, "zone": "battlefield", "text": ""})
        nodes[name] = node


def _subject(node):
    return ag.subject_of({"name": node["id"], "types": node["types"], "subtypes": node["subtypes"],
                          "colors": node["colors"], "mana_cost": node.get("mana_cost"),
                          "is_token": node.get("is_token", False), "pt": node.get("pt")}) | {"pt": node.get("pt")}


# ── edges ────────────────────────────────────────────────────────────────────

def _productions(nodes):
    """Every (event, producer, subject, actor, via, ability index) in the deck. A
    produced token's subject is the TOKEN node; a cost's event (a sacrifice) is
    produced once per body the cost can eat, with that body as the subject."""
    out = []
    bodies = [n for n in nodes.values() if "Creature" in n["types"]]
    for n in nodes.values():
        for i, a in enumerate(n["abilities"]):
            for p in a.get("produces") or []:
                ev = p["event"]
                if p.get("token") and f"token:{p['token']}" in nodes:
                    subj = _subject(nodes[f"token:{p['token']}"])
                    via = nodes[f"token:{p['token']}"]["id"]
                elif p.get("self") or p.get("actor") == "self":
                    subj, via = _subject(n), None
                else:
                    subj, via = None, None               # a draw, a life loss: no card subject
                out.append({"event": ev, "from": n["id"], "subject": subj, "actor": p.get("actor"),
                            "via": via, "ability": i, "amount": p.get("amount"),
                            "implicit": a["kind"] == "implicit", "keywords": p.get("keywords")})
            for k in a.get("costs") or []:
                if k.get("stock") == "BODIES":
                    for b in bodies:
                        if b["id"] == n["id"]:
                            continue
                        if ag.filter_accepts(k.get("filters") or [], _subject(b), source=n["id"]) is not False:
                            out.append({"event": "DIES", "from": n["id"], "subject": _subject(b), "actor": "you",
                                        "via": b["id"], "ability": i, "amount": k.get("amount"),
                                        "implicit": False, "cost": True})
                    # The outlet itself as its own fodder (Sac<1/CARDNAME>): a DIES of self.
                    if any(alt.get("self") for alt in k.get("filters") or []):
                        out.append({"event": "DIES", "from": n["id"], "subject": _subject(n), "actor": "you",
                                    "via": None, "ability": i, "amount": k.get("amount"), "implicit": False,
                                    "cost": True})
                elif k.get("stock") == "HAND":
                    out.append({"event": "DISCARD", "from": n["id"], "subject": None, "actor": "you",
                                "via": None, "ability": i, "amount": k.get("amount"), "implicit": False, "cost": True})
    for ev in OPPONENT_EVENTS:
        out.append({"event": ev, "from": OPPONENT, "subject": {"controller": "opponent", "types": [], "subtypes": [],
                                                              "colors": [], "name": None},
                    "actor": "opponent", "via": None, "ability": None, "amount": None, "implicit": False})
    out.append({"event": "TURN_PHASE", "from": TURN, "subject": None, "actor": "you", "via": None,
                "ability": None, "amount": None, "implicit": False})
    return out


def _consumptions(nodes):
    out = []
    for n in nodes.values():
        for i, a in enumerate(n["abilities"]):
            c = a.get("consumes")
            if c:
                out.append({"event": c["event"], "to": n["id"], "filters": c.get("filters") or [],
                            "actor": c.get("actor"), "ability": i, "zone": a.get("zone")})
            for k in a.get("costs") or []:
                if k.get("stock") in ("BODIES", "HAND", "LIFE"):
                    out.append({"event": k["stock"], "to": n["id"], "filters": k.get("filters") or [],
                                "actor": "you", "ability": i, "zone": "battlefield", "stock": True})
    return out


def _subjectless_accepts(alts):
    """A draw, a life gain, a tutor of an unnamed card: the event has no card
    subject. A player or Card-only alternative accepts it; a `Self` alternative
    can never (another card's event is not this card); anything that names a type,
    a colour or a subtype is UNPROVEN — the card could be anything."""
    if not alts:
        return True
    verdicts = []
    for alt in alts:
        if alt.get("self"):
            verdicts.append(False)
        elif set(alt) <= {"any", "controller", "player", "other"}:
            verdicts.append(True)
        else:
            verdicts.append(None)
    if True in verdicts:
        return True
    return None if None in verdicts else False


def _actor_ok(consumer_actor, producer_actor):
    if consumer_actor in (None, "player"):
        return True
    if producer_actor in (None, "player", "target"):
        return True
    if producer_actor == "self":
        return consumer_actor == "you"
    return consumer_actor == producer_actor


def _edges(nodes, productions, consumptions):
    """`feeds` edges: producer -> consumer on an event the consumer's filter accepts
    (`proven: false` when a filter atom is outside the grammar); `supplies`: a body
    -> the stock BODIES -> an outlet whose cost accepts it; `fills`: an event that
    adds to a stock. Deduped on (from, to, event)."""
    edges = OrderedDict()

    def add(kind, frm, to, event, proven=True, via=None, implicit=False, checked=None):
        key = (frm, to, event)
        row = edges.get(key)
        if row is None:
            row = {"kind": kind, "from": frm, "to": to, "event": event, "proven": proven,
                   "via": [], "implicit": implicit}
            edges[key] = row
        if via and via not in row["via"]:
            row["via"].append(via)
        if proven and not row["proven"]:
            row["proven"] = True
        if not implicit:
            row["implicit"] = False

    by_event = {}
    for p in productions:
        by_event.setdefault(p["event"], []).append(p)
    for c in consumptions:
        if c.get("stock"):
            stock = c["event"]
            for n in nodes.values():
                if n["id"] == c["to"]:
                    continue
                if stock == "BODIES" and "Creature" in n["types"]:
                    v = ag.filter_accepts(c["filters"], _subject(n), source=c["to"])
                    if v is not False:
                        add("supplies", n["id"], stock, stock, proven=True)
                        add("supplies", stock, c["to"], stock, proven=v is True, via=n["id"])
            if stock in ("HAND", "LIFE"):
                add("supplies", stock, c["to"], stock)
            continue
        for p in by_event.get(c["event"], ()):
            if p["from"] == c["to"]:
                continue                                  # a card's own event is not an edge
            if not _actor_ok(c["actor"], p["actor"]):
                continue
            if p["from"] == OPPONENT and c["actor"] not in ("opponent", "player", None):
                continue
            if p["from"] != OPPONENT and c["actor"] == "opponent":
                continue
            if p["subject"] is None:
                v = _subjectless_accepts(c["filters"])
            else:
                v = ag.filter_accepts(c["filters"], p["subject"], source=c["to"])
            if v is False:
                continue
            add("feeds", p["from"], c["to"], c["event"], proven=v is True, via=p.get("via"),
                implicit=p.get("implicit", False))
    # A maker reaches its token: the token's own abilities and combat are the
    # maker's reach (Shark Typhoon's shark is the payoff, not Shark Typhoon).
    for p in productions:
        if p.get("via") and str(p["via"]).startswith("token:") and p["event"] == "TOKEN" \
                and p["from"] not in (OPPONENT, TURN):
            add("makes", p["from"], p["via"], "TOKEN")
    # Stocks filled.
    for p in productions:
        stock = _FILLS.get(p["event"])
        if stock and p["from"] not in (OPPONENT, TURN) and not p.get("cost") and not p.get("implicit"):
            if p.get("actor") in ("you", None, "self", "player"):
                add("fills", p["from"], stock, p["event"])
        if (p["event"] == "ALT_WIN" or (p["event"] in ENGINE_TERMINALS
                                        and p.get("actor") in ("opponent", "player", "target", None))) \
                and p["from"] not in (OPPONENT, TURN):
            add("ends", p["from"], TERMINAL, p["event"], implicit=p.get("implicit", False))
    # A stock drawn on by an event's consumers: HAND -> every CAST consumer's ignition
    # class is the deck's castable spells, so HAND feeds the CAST producers, not the
    # consumer; GRAVEYARD -> a RECURSION producer (it returns what the yard holds).
    for p in productions:
        if p["event"] == "RECURSION" and p["from"] not in (OPPONENT, TURN):
            add("supplies", "GRAVEYARD", p["from"], "GRAVEYARD")
    return list(edges.values())


def _modifiers(nodes):
    scales, protects = [], []
    for n in nodes.values():
        for i, a in enumerate(n["abilities"]):
            sc = a.get("scales")
            if sc:
                kws = sc.get("keywords") or []
                if sc["event"] == "DENIAL":
                    # A tax or a ban on attacking, blocking, casting or targeting: it
                    # protects the table side it names and feeds nothing.
                    protects.append({"card": n["id"], "filters": sc.get("filters") or [],
                                     "keywords": [a.get("text", "")[:60] or "denial"], "kind": "denial"})
                elif sc["event"] == "PUMP" and any(any(k.startswith(p) for p in PROTECTIVE) for k in kws):
                    protects.append({"card": n["id"], "filters": sc.get("filters") or [], "keywords": kws,
                                     "kind": "static"})
                elif sc["event"] == "PUMP":
                    scales.append({"card": n["id"], "event": "COMBAT_DAMAGE", "kind": "pump",
                                   "filters": sc.get("filters") or [], "keywords": kws})
                else:
                    scales.append({"card": n["id"], "event": sc["event"], "kind": a["kind"],
                                   "actor": sc.get("actor"), "your_turn": sc.get("your_turn", False),
                                   "filters": sc.get("filters") or []})
            for p in a.get("produces") or []:
                kws = p.get("keywords") or []
                if p["event"] == "PUMP" and any(any(k.startswith(q) for q in PROTECTIVE) for k in kws):
                    protects.append({"card": n["id"], "filters": p.get("targets") or [], "keywords": kws,
                                     "kind": a["kind"]})
                if p["event"] == "COUNTERSPELL":
                    protects.append({"card": n["id"], "filters": [], "keywords": ["Counter"], "kind": a["kind"]})
    return scales, protects


# ── components, stages, lines, spof ─────────────────────────────────────────

def _deck(nodes):
    return [n for n in nodes.values() if n["zone"] in ("command", "library")]


def _components(nodes, edges, deck_names, roles, library_size):
    from manamap.pilot.deck_audit import _component
    consumed = {}
    for e in edges:
        if e["kind"] != "feeds" or e["to"] not in nodes or nodes[e["to"]]["zone"] == "token":
            continue
        c = consumed.setdefault(e["event"], {"producers": set(), "consumers": set(), "unproven": set(),
                                             "deck_fed": set(), "opponent": False, "turn": False})
        c["consumers"].add(e["to"])
        if e["from"] == OPPONENT:
            c["opponent"] = True
        elif e["from"] == TURN:
            c["turn"] = True
        else:
            c["deck_fed"].add(e["to"])                    # a consumer the DECK feeds
            if nodes.get(e["from"], {}).get("zone") != "token":
                c["producers"].add(e["from"])
                if not e["proven"]:
                    c["unproven"].add(e["from"])
    scales, _ = _modifiers(nodes)
    out = []
    for ev in sorted(consumed):
        c = consumed[ev]
        producers = sorted(c["producers"])
        consumers = sorted(c["consumers"])
        comp = _component({"any_of": [p for p in producers if nodes[p]["zone"] == "library"]},
                          deck_names, roles, library_size)
        out.append({
            "id": f"ev:{ev}", "event": ev, "producers": producers, "consumers": consumers,
            "deck_fed_consumers": sorted(c["deck_fed"]),
            "unproven_producers": sorted(c["unproven"]),
            "from_opponent": c["opponent"], "from_turn": c["turn"],
            "modifiers": [m["card"] for m in scales if m["event"] == ev],
            "redundancy": {"producers": len(producers), "consumers": len(consumers)},
            "activation": {"producers": {"size": comp["size"], "odds": comp["odds"]},
                           "consumers": _component({"any_of": [x for x in consumers if nodes[x]["zone"] == "library"]},
                                                   deck_names, roles, library_size)["odds"]},
            "thin": len(producers) <= ENGINE_THIN_GROUP and not c["opponent"] and not c["turn"],
            "channel": ENGINE_EVENTS.get(ev, {}).get("channel"),
        })
    return out


def _degrees(nodes, edges):
    """Per node: explicit produced events, consumed events, feeds in/out by source class."""
    deg = {n: {"in_deck": set(), "in_opp": False, "in_turn": False, "out_feeds": set(), "out_events": set(),
               "out_explicit": set(), "ends": set(), "supplies_bodies": False, "fills": set(), "stock_in": set()}
           for n in nodes}
    for e in edges:
        if e["kind"] == "feeds" and e["to"] in deg:
            if e["from"] == OPPONENT:
                deg[e["to"]]["in_opp"] = True
            elif e["from"] == TURN:
                deg[e["to"]]["in_turn"] = True
            elif e["from"] in deg:
                deg[e["to"]]["in_deck"].add(e["from"])
                deg[e["from"]]["out_feeds"].add(e["to"])
                deg[e["from"]]["out_events"].add(e["event"])
                if not e.get("implicit"):
                    deg[e["from"]]["out_explicit"].add(e["event"])
        elif e["kind"] == "ends" and e["from"] in deg:
            deg[e["from"]]["ends"].add(e["event"])
            if nodes[e["from"]].get("is_token"):
                for maker in _makers_of(nodes, e["from"]):
                    deg[maker]["ends"].add(e["event"])
                    deg[maker]["via_token"] = True
        elif e["kind"] == "supplies" and e["to"] == "BODIES" and e["from"] in deg:
            deg[e["from"]]["supplies_bodies"] = True
        elif e["kind"] == "supplies" and e["from"] in STOCKS and e["to"] in deg:
            deg[e["to"]]["stock_in"].add(e["from"])
        elif e["kind"] == "fills" and e["from"] in deg:
            deg[e["from"]]["fills"].add(e["to"])
    return deg


def _makers_of(nodes, token_id):
    stem = token_id.split(":", 1)[1]
    return [n["id"] for n in nodes.values()
            if any(p.get("token") == stem for a in n["abilities"] for p in a.get("produces") or [])]


def _self_growth(node):
    """COUNTER_ADD or PUMP the card puts on ITSELF off something it consumed:
    Shabraz grows a counter per draw and cashes it in combat."""
    for a in node["abilities"]:
        if a.get("consumes"):
            for p in a.get("produces") or []:
                if p["event"] in ("COUNTER_ADD", "PUMP") and p.get("actor") == "self":
                    return True
    return False


def _explicit_produces(node):
    return {p["event"] for a in node["abilities"] if a["kind"] != "implicit" for p in a.get("produces") or []}


def _explicit_consumes(node):
    out = set()
    for a in node["abilities"]:
        if a.get("consumes"):
            out.add(a["consumes"]["event"])
        for k in a.get("costs") or []:
            if k.get("stock"):
                out.add(k["stock"])
    return out


def _terminal_amounts(node):
    out = []
    for a in node["abilities"]:
        if a["kind"] == "implicit":
            continue
        for p in a.get("produces") or []:
            if p["event"] == "ALT_WIN" or (p["event"] in ENGINE_TERMINALS
                                           and p.get("actor") in ("opponent", "player", "target", None)):
                out.append(p.get("amount"))
    return out


def _stage_of(node, deg, nodes, edges, components, protects_by_card, scales_by_card):
    """Every rule the node satisfies, in `ENGINE_STAGE_RULES` order; the first is
    its stage. Each rule's test is spelled out here so the printout can say WHY."""
    prod = _explicit_produces(node)
    cons = _explicit_consumes(node)
    d = deg[node["id"]]
    deck_consumed = {c["event"] for c in components}
    # An implicit CAST feeding a "whenever you cast" card is a CLASS, not a start:
    # it is reported in the CAST component's producers, never as ignition. Filling
    # a stock some outlet draws on (bodies under a sacrifice outlet) IS an out.
    drawn_stocks = {e["from"] for e in edges if e["kind"] == "supplies" and e["from"] in ("BODIES", "HAND", "GRAVEYARD")}
    outs_to_deck = set(d["out_explicit"]) | (d["fills"] & drawn_stocks)
    self_pain = {"DAMAGE", "LIFE_LOSS"}
    has_outlet = any(c["event"] == "BODIES" for c in components) or any(
        e["kind"] == "supplies" and e["from"] == "BODIES" for e in edges)
    terminals = _terminal_amounts(node)
    is_cmd = node["zone"] == "command"
    hits, why = [], {}

    def mark(stage, reason):
        hits.append(stage)
        why[stage] = reason

    scaled = scales_by_card.get(node["id"]) or []
    if "Land" in node["types"]:
        mark("mana", "a land" + (f" that also feeds {sorted(outs_to_deck)}" if outs_to_deck else ""))
    elif prod and (prod - self_pain) <= MANA_ONLY and cons <= {"TAP", "TURN_PHASE", "MANA", "LIFE"} \
            and not terminals and not outs_to_deck:
        mark("mana", f"produces only {sorted(prod)}")
    if node["id"] in protects_by_card and not (outs_to_deck - {"PUMP"}) and not terminals:
        mark("protection", f"protects: {', '.join(protects_by_card[node['id']])}")
    starts_nothing_before = not d["in_deck"] and not d["in_opp"] and not d["in_turn"]
    if is_cmd and outs_to_deck:
        mark("ignition", f"in the command zone, producing {sorted(outs_to_deck)} the deck consumes")
    elif starts_nothing_before and outs_to_deck and not (prod <= MANA_ONLY and prod):
        mark("ignition", f"nothing in the deck starts it; it produces {sorted(outs_to_deck)}")
    if any(a == "X" or (isinstance(a, int) and a >= 40) for a in terminals):
        mark("wincon", "a terminal with a variable or lethal amount")
    elif "ALT_WIN" in prod:
        mark("wincon", "wins the game")
    elif terminals and d["in_deck"]:
        feeders = max((c["redundancy"]["producers"] for c in components
                       if node["id"] in c["consumers"]), default=0)
        if feeders >= ENGINE_THIN_GROUP + 1:
            mark("wincon", f"a terminal fed by {feeders} deck producers")
    fed = d["in_deck"] or d["in_opp"] or d["in_turn"] or d["stock_in"]
    if fed and (outs_to_deck - {"CAST"}):
        mark("conversion", f"consumes {sorted(cons)}, produces {sorted(outs_to_deck - {'CAST'})}")
    elif fed and d["in_deck"] and (d["fills"] & {"HAND", "MANA", "GRAVEYARD"}):
        mark("conversion", f"turns {sorted(cons)} into {sorted(d['fills'] & {'HAND', 'MANA', 'GRAVEYARD'})}")
    elif fed and _self_growth(node):
        mark("conversion", f"grows itself off {sorted(cons)} and cashes it in combat")
    elif scaled and not (set(scaled) & set(ENGINE_TERMINALS)) and not prod:
        mark("conversion", f"scales {sorted(set(scaled))}")
    fuel_prod = (prod & FUEL_EVENTS) | (prod & {"MANA", "TREASURE"} if cons else set())
    if fuel_prod and not (d["in_deck"] - {node["id"]}):
        mark("fuel", f"{sorted(fuel_prod)} off {'the opponents' if d['in_opp'] else 'the turn' if d['in_turn'] else 'a cast'}")
    vanilla = all(a["kind"] in ("implicit", "keyword") for a in node["abilities"])
    if ("TOKEN" in prod or ("Creature" in node["types"] and vanilla)) and has_outlet and not terminals and not scaled:
        mark("fodder", "bodies for an outlet")
    if terminals or (d["ends"] - ({"COMBAT_DAMAGE"} if not (_self_growth(node) or d.get("via_token")) else set())) \
            or ("REMOVAL" in prod) or ("COUNTERSPELL" in prod):
        mark("output", f"reaches the terminal by {sorted(d['ends'] or set(prod & {'REMOVAL', 'COUNTERSPELL'}))}"
             + (" through its tokens" if d.get("via_token") and not terminals else ""))
    elif scaled and (set(scaled) & set(ENGINE_TERMINALS)) and not prod:
        mark("output", f"scales {sorted(set(scaled) & set(ENGINE_TERMINALS))}")
    elif "Creature" in node["types"] and d["ends"]:
        mark("output", "a body that attacks")
    ordered = [s for s in ENGINE_STAGE_RULES if s in hits]
    if not ordered:
        unread = (node.get("coverage") or {}).get("unmapped") or []
        return "unplaced", [], {"unplaced": "no rule matched: produces "
                                + (", ".join(sorted(prod)) or "nothing the registry names")
                                + (f" (unmapped: {', '.join(unread[:3])})" if unread else "")}
    return ordered[0], ordered[1:], why


def _stages(nodes, edges, components):
    deg = _degrees(nodes, edges)
    scales, protects = _modifiers(nodes)
    pb = {}
    for p in protects:
        pb.setdefault(p["card"], []).extend(p["keywords"])
    sb = {}
    for s in scales:
        sb.setdefault(s["card"], []).append(s["event"])
    members = {s: [] for s in ENGINE_STAGE_RULES}
    unassigned = []
    placed = {}
    for n in _deck(nodes):
        stage, also, why = _stage_of(n, deg, nodes, edges, components, pb, sb)
        placed[n["id"]] = {"stage": stage, "also": also, "why": why.get(stage) or why.get("unplaced")}
        if stage == "unplaced":
            unassigned.append({"card": n["id"], "why": placed[n["id"]]["why"]})
        else:
            members[stage].append(n["id"])
    # The LIST shape every reader of engine.json already knows (the handbook's
    # Systems, the dossier, engine_casts, the build page's arrows): one block per
    # stage that has members, `what_it_does` COMPUTED from the rules that placed
    # them — the reading may say it better, never here.
    stages = []
    for s in ENGINE_STAGE_RULES:
        if not members[s]:
            continue
        whys = Counter(placed[c]["why"] for c in members[s])
        what = "; ".join(f"{w} ({k})" if k > 1 else w for w, k in whys.most_common(3))
        stages.append({"stage": s, "label": STAGE_LABEL.get(s, s.title()), "cards": members[s],
                       "what_it_does": f"{len(members[s])} card(s), by rule: {what}",
                       "also": {c: placed[c]["also"] for c in members[s] if placed[c]["also"]},
                       "single_point_of_failure": None})
    return stages, unassigned, placed


def _lines(nodes, edges, placed, components, by_stack, library_size):
    """Lines are SHAPES: distinct event sequences from an ignition to the terminal,
    each with the cards that stand at every hop. A deck with fifteen token makers
    and twelve drains has one `ETB › LIFE_LOSS` line with fifteen starts and twelve
    ends, not a hundred and eighty two-card lines. Walked as simple paths over
    feeds / supplies / fills edges (a stock is a hop), depth-capped, an implicit
    CAST allowed only as the first hop and a bare body's combat never the end;
    ranked ✓ first (a passing stack names every card of SOME instance), then by the
    product of hop availabilities by turn three."""
    from manamap.pilot.manabase import cards_seen, hypergeometric_at_least
    adj = {}
    links = {(e["from"], e["to"]) for e in edges if e["kind"] in ("feeds", "supplies", "fills", "makes")}
    for e in edges:
        if e["kind"] in ("feeds", "supplies", "fills", "ends", "makes") and e["from"] not in (OPPONENT, TURN) \
                and e["to"] not in ("LIFE", "MANA") and e["from"] not in ("LIFE", "MANA"):
            adj.setdefault(e["from"], []).append(e)
    ignition = [c for c, p in placed.items() if p["stage"] == "ignition"]
    # A FRONTIER EXPANSION, NOT A PATH ENUMERATION (2026-11-22). Enumerating simple
    # paths took ten seconds on edgar (one shape had 4,800 of them) and every
    # status row paid it. A shape is defined by its event sequence, so paths with
    # the same events are merged as they go: a state is (node, events so far) and
    # carries the SET of cards seen at every position; two paths reaching the same
    # state union their sets. States are bounded by nodes x distinct event prefixes.
    shapes = {}
    states = {}

    def seed(path, events):
        key = (path[-1], tuple(events))
        sets = states.setdefault(key, [set() for _ in path])
        for i, c in enumerate(path):
            sets[i].add(c)

    for ig in ignition:
        seed([ig], [])
    # A commander that consumes an event is reached by its producers: the line
    # "cast a Vampire -> Edgar -> a body -> ..." starts at the Vampires, not at Edgar.
    for cmd in [n["id"] for n in nodes.values() if n["zone"] == "command" and n["id"] in ignition]:
        for e in edges:
            if e["kind"] == "feeds" and e["to"] == cmd and e["from"] in nodes \
                    and nodes[e["from"]]["zone"] == "library":
                seed([e["from"], cmd], [e["event"]])

    frontier = list(states.items())
    for depth in range(ENGINE_LINE_MAX_DEPTH + 1):
        nxt_frontier = []
        for (last, events), sets in frontier:
            for e in adj.get(last, ()):
                nxt = e["to"]
                # No revisit of a card the path definitely holds (a singleton
                # position), and never the node we stand on.
                if nxt == last or any(len(st) == 1 and nxt in st for st in sets):
                    continue
                if e.get("implicit") and len(sets) > 1 and nxt != TERMINAL:
                    continue                              # only the ignition's own cast, entry or attack is a hop
                label = nxt if nxt in STOCKS else (None if last in STOCKS else e["event"])
                evs = list(events) + ([label] if label else [])
                if nxt == TERMINAL:
                    if e.get("implicit") and not (nodes[last].get("is_token") or _self_growth(nodes[last])):
                        continue                          # a bare body's combat is not a line's end
                    if len(evs) < 2:
                        continue                          # one hop is a spell, not a line
                    key = tuple(evs) + ("+",) * max(0, len(sets) - len(evs))
                    sh = shapes.setdefault(key, {"at": [set() for _ in sets]})
                    if len(sh["at"]) != len(sets):        # the same events over a different hop count
                        key = key + ("+",)
                        sh = shapes.setdefault(key, {"at": [set() for _ in sets]})
                    for i, st in enumerate(sets):
                        sh["at"][i] |= st
                    continue
                if depth >= ENGINE_LINE_MAX_DEPTH:
                    continue
                skey = (nxt, tuple(evs))
                if skey in states:
                    tgt = states[skey]
                    if len(tgt) != len(sets) + 1:
                        continue                          # same events, different hop count: rare; dropped
                    grew = False
                    for i, st in enumerate(sets):
                        if not st <= tgt[i]:
                            tgt[i] |= st
                            grew = True
                    if nxt not in tgt[-1]:
                        tgt[-1].add(nxt)
                        grew = True
                    if grew:
                        nxt_frontier.append((skey, tgt))
                else:
                    tgt = [set(st) for st in sets] + [{nxt}]
                    states[skey] = tgt
                    nxt_frontier.append((skey, tgt))
        frontier = nxt_frontier
        if not frontier:
            break
    out = []
    for evs, sh in shapes.items():
        hops = []
        avail = 1.0
        for i, members in enumerate(sh["at"]):
            cards = sorted(m for m in members if m in nodes and nodes[m]["zone"] == "library")
            cmd = sorted(m for m in members if m in nodes and nodes[m]["zone"] == "command")
            tokens = sorted(m for m in members if m in nodes and nodes[m].get("is_token"))
            stocks = sorted(m for m in members if m in STOCKS)
            copies = sum(nodes[c]["copies"] for c in cards)
            # A stock, a token or the commander is where the line passes, not a draw.
            p3 = 1.0 if (cmd or stocks or (tokens and not cards)) else \
                (hypergeometric_at_least(copies, cards_seen(3), 1, library_size) if copies else 0.0)
            avail *= p3
            hops.append({"cards": cards + cmd, "tokens": tokens, "stocks": stocks, "n": len(cards) + len(cmd),
                         "by_turn_three": round(p3, 3)})
        # A stack stamps the shape when a CONNECTED instance of it sits inside the
        # scenario: at every card-bearing hop the stack names a card, and each
        # named card at hop i has an edge to a named card at hop i+1 (through the
        # stock between them when there is one). Naming a card per hop alone
        # stamped 38 of yawgmoth's 40 shapes; connection is what the stack proves.
        stacks = sorted(sid for sid, st in by_stack.items()
                        if len(set(st["cards"])) >= 2 and _stack_supports(hops, set(st["cards"]), links))
        real = [h for h in hops if h["cards"]]
        bottleneck = min(real, key=lambda h: h["n"]) if real else None
        events = [e for e in evs if e != "+"]

        def stage_of_hop(h):
            votes = Counter(placed.get(c, {}).get("stage") for c in h["cards"] if c in placed)
            return votes.most_common(1)[0][0] if votes else None
        via = sorted({c for h in real for c in h["cards"]} if sum(h["n"] for h in real) <= 6
                     else {c for h in real if h["n"] <= ENGINE_THIN_GROUP for c in h["cards"]})
        out.append({"events": events, "terminal": events[-1],
                    "from": stage_of_hop(real[0]) if real else None,
                    "to": stage_of_hop(real[-1]) if real else None,
                    "via": via, "carries": " › ".join(events),
                    "hops": hops, "starts": len(sh["at"][0]),
                    "cards": sorted({c for h in hops for c in h["cards"]}),
                    "bottleneck": {"position": hops.index(bottleneck), "n": bottleneck["n"],
                                   "cards": bottleneck["cards"]} if bottleneck else None,
                    "availability_by_turn_three": round(avail, 4),
                    "verified_by": stacks, "stamp": "✓" if stacks else "◆"})
    out.sort(key=lambda l: (0 if l["verified_by"] else 1, -l["availability_by_turn_three"], len(l["events"]), l["events"]))
    out = out[:ENGINE_MAX_LINES]
    for i, l in enumerate(out, 1):
        l["id"] = f"L{i:02d}"
    return out


def _stack_supports(hops, named, links):
    """Does the scenario hold a connected instance of this shape? Walk the hops;
    at each card-bearing hop keep the named cards reachable from the previous
    hop's kept cards (a stock or a token between two hops is crossed when both
    sides link to it)."""
    reach = None
    pending_bridge = None                                 # a stock/token hop between two card hops
    for h in hops:
        if not h["cards"]:
            pending_bridge = (h["stocks"] or h["tokens"] or [None])[0]
            continue
        here = [c for c in h["cards"] if c in named]
        if not here:
            return False
        if reach is None:
            reach = set(here)
        else:
            if pending_bridge:
                nxt = {c for c in here if any((a, pending_bridge) in links and (pending_bridge, c) in links
                                              for a in reach)}
            else:
                nxt = {c for c in here if any((a, c) in links for a in reach)}
            if not nxt:
                return False
            reach = nxt
        pending_bridge = None
    return reach is not None and len(reach) >= 1


def _spof(nodes, edges, placed, components, lines):
    """Articulation points of the engine subgraph (deck nodes outside mana and
    protection, undirected over feeds/supplies), single-producer consumed events,
    and an unprotected commander."""
    engine = {c for c, p in placed.items() if p["stage"] not in ("mana", "protection", "unplaced")}
    und = {n: set() for n in engine}
    for st in ("BODIES", "HAND", "GRAVEYARD"):
        und[st] = set()                                   # a stock joins the sides it connects
    for e in edges:
        a, b = e["from"], e["to"]
        if e["kind"] in ("feeds", "supplies", "fills") and a in und and b in und:
            und[a].add(b)
            und[b].add(a)
    cut = {c for c in _articulation_points(und) if c not in STOCKS}
    out = []
    for c in sorted(cut):
        severs = [l["id"] for l in lines if any(c in h["cards"] and h["n"] == 1 for h in l["hops"])]
        out.append({"card": c, "kind": "cut_vertex", "severs": severs})
    for comp in components:
        if comp["redundancy"]["producers"] == 1 and not comp["from_opponent"] and not comp["from_turn"]:
            only = comp["producers"][0]
            if not any(r["card"] == only for r in out):
                out.append({"card": only, "kind": "single_producer", "event": comp["event"],
                            "severs": [l["id"] for l in lines if any(only in h["cards"] and h["n"] == 1 for h in l["hops"])]})
        if len(comp["deck_fed_consumers"]) == 1 and comp["producers"]:
            only = comp["deck_fed_consumers"][0]
            if not any(r["card"] == only for r in out):
                out.append({"card": only, "kind": "single_consumer", "event": comp["event"],
                            "severs": [l["id"] for l in lines if any(only in h["cards"] and h["n"] == 1 for h in l["hops"])]})
    _, protects = _modifiers(nodes)
    for n in nodes.values():
        if n["zone"] == "command":
            covered = any(ag.filter_accepts(p["filters"], _subject(n), source=p["card"]) is not False
                          for p in protects if p["card"] != n["id"])
            if not covered:
                out.append({"card": n["id"], "kind": "unprotected_commander", "severs": [l["id"] for l in lines if any(n["id"] in h["cards"] and h["n"] == 1 for h in l["hops"])]})
    return out


def _articulation_points(adj):
    """Tarjan over an undirected adjacency dict."""
    index = {}
    low = {}
    cut = set()
    counter = [0]

    def dfs(u, parent):
        index[u] = low[u] = counter[0]
        counter[0] += 1
        children = 0
        for v in sorted(adj[u]):
            if v not in index:
                children += 1
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if parent is not None and low[v] >= index[u]:
                    cut.add(u)
            elif v != parent:
                low[u] = min(low[u], index[v])
        if parent is None and children > 1:
            cut.add(u)

    for u in sorted(adj):
        if u not in index:
            dfs(u, None)
    return cut


# ── the record ───────────────────────────────────────────────────────────────

def build(slug, branch=None):
    from manamap.pilot import engine_facts, formats
    from manamap.pilot.goldfish import model_version
    from manamap.sim import forge_cards
    nodes, cards_doc = _nodes(slug, branch)
    deck_names = {n["id"] for n in _deck(nodes)}
    library_size = formats.for_deck(slug, branch).library_size
    productions = _productions(nodes)
    consumptions = _consumptions(nodes)
    edges = _edges(nodes, productions, consumptions)
    components = _components(nodes, edges, deck_names, {}, library_size)
    stages, unassigned, placed = _stages(nodes, edges, components)
    pairings = engine_facts.verified_pairings(slug, deck_names) if branch is None else {"by_stack": {}}
    lines = _lines(nodes, edges, placed, components, pairings.get("by_stack") or {}, library_size)
    spof = _spof(nodes, edges, placed, components, lines)
    for st in stages:                                   # the first cut vertex among a stage's cards
        st["single_point_of_failure"] = next((x["card"] for x in spof if x["kind"] == "cut_vertex"
                                              and x["card"] in st["cards"]), None)
    scales, protects = _modifiers(nodes)
    readers = Counter(n["coverage"]["reader"] for n in _deck(nodes))
    unmapped = sorted({u for n in _deck(nodes) for u in n["coverage"]["unmapped"]})
    targets = load_json(deck_file(slug, "goldfish_targets.json", branch), default={}) or {}
    flags = {k: bool(v) for k, v in targets.items() if k.startswith("model_")}
    return {
        "slug": slug, "branch": branch,
        "decklist_sha256": cards_doc.get("decklist_sha256") or decklist_sha256(slug, branch),   # deck-status reads it here
        "_meta": {"v": 2, "decklist_sha256": cards_doc.get("decklist_sha256") or decklist_sha256(slug, branch),
                  "forge_digest": forge_cards.cardsfolder_digest(), "ability_graph_version": ABILITY_GRAPH_VERSION,
                  "model_version": model_version(), "readers": dict(sorted(readers.items())),
                  "library_size": library_size},
        "graph": {
            "nodes": [{k: v for k, v in n.items()} for n in nodes.values()],
            "edges": [e for e in edges if e["kind"] in ("feeds", "supplies", "fills", "ends", "makes")],
            "scales": scales, "protects": protects,
        },
        "components": components,
        "stages": stages,
        "unassigned": unassigned,
        "placed": placed,
        "lines": lines,
        "spof": spof,
        "coverage": {"readers": dict(sorted(readers.items())),
                     "partial": sorted(n["id"] for n in _deck(nodes) if n["coverage"]["partial"]),
                     "unmapped_forge_names": unmapped,
                     "ai_remove_deck": sorted(n["id"] for n in _deck(nodes) if n["coverage"]["ai_remove_deck"])},
        "stamps": {"stacks": {sid: s["cards"] for sid, s in (pairings.get("by_stack") or {}).items()},
                   "goldfish_flags": flags},
        "reading": None,
    }


def empty_reading(sha):
    return {"written_for": sha, "stale": False, "names_departed": [], "thesis": None, "lines": [],
            "stage_overrides": [], "open_questions": [], "critic": None, "decided_by": [], "stamped": []}


def migrate_reading(old, fresh):
    """A v1 `engine.json`'s authored parts -> a v2 `reading`, stamped with the list
    they were written for and marked stale when that is not the current list. The
    v1 line notes are keyed to the nearest graph line by card overlap, else kept
    under `unmatched_v1_lines`. Nothing authored is dropped."""
    if not old:
        return empty_reading(fresh["_meta"]["decklist_sha256"])
    if (old.get("_meta") or {}).get("v") == 2 and isinstance(old.get("reading"), dict):
        r = dict(old["reading"])
    else:
        written_for = old.get("decklist_sha256") or (old.get("_meta") or {}).get("decklist_sha256") \
            or (old.get("stamps") or {}).get("decklist_sha256")
        r = empty_reading(written_for)
        r["thesis"] = old.get("thesis")
        r["open_questions"] = list(old.get("open_questions") or [])
        r["critic"] = old.get("critic")
        r["notes"] = old.get("notes")
        r["decided_by"] = ["deck-engineer", "engine-critic"] if old.get("critic") else ["deck-engineer"]
        lines, unmatched = [], []
        for ol in old.get("lines") or []:
            cards = set((ol.get("via") or []) + [ol.get("from"), ol.get("to")]) - {None}
            best = max(fresh["lines"], key=lambda l: len(cards & set(l["cards"])), default=None)
            if best and len(cards & set(best["cards"])) >= 2:
                lines.append({"line": best["id"], "name": f"{ol.get('from')} → {ol.get('to')}",
                              "note": ol.get("note"), "v1_via": ol.get("via"), "v1_verified_by": ol.get("verified_by")})
            else:
                unmatched.append(ol)
        r["lines"] = lines
        if unmatched:
            r["unmatched_v1_lines"] = unmatched
        v1_stages = old.get("stages") or []
        if isinstance(v1_stages, dict):                  # an early hand-written shape
            v1_stages = [{"stage": k, "cards": v} for k, v in v1_stages.items()]
        if v1_stages:
            r["v1_stages"] = [{"stage": st.get("stage"), "label": st.get("label"),
                               "what_it_does": st.get("what_it_does"), "cards": st.get("cards")}
                              for st in v1_stages]
        if old.get("map_disagreements"):
            r["v1_map_disagreements"] = old["map_disagreements"]
    now = fresh["_meta"]["decklist_sha256"]
    r["stale"] = bool(r.get("written_for")) and r["written_for"] != now or not r.get("written_for")
    deck_names = {n["id"] for n in fresh["graph"]["nodes"] if n["zone"] in ("command", "library")}
    named = set()
    for key in ("thesis",):
        pass
    for l in r.get("lines") or []:
        for c in (l.get("v1_via") or []):
            named.add(c)
    for o in r.get("stage_overrides") or []:
        named.add(o.get("card"))
    r["names_departed"] = sorted(c for c in named if c and c not in deck_names)
    return r


def write(slug, branch=None):
    fresh = build(slug, branch)
    path = deck_dir(slug, branch) / "engine.json"          # a write never falls back to the deck
    old = load_json(path, default=None)
    fresh["reading"] = migrate_reading(old, fresh)
    path.write_text(json.dumps(fresh, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # ONE MODEL: a deck that opted in (`targets_from: engine.json`) gets its goldfish
    # targets regenerated from the components in the same write; every other deck's
    # declaration is untouched.
    from manamap.pilot import scaffold_targets
    scaffold_targets.regenerate_targets(slug, branch, engine=fresh)
    return fresh, path


def computed_part(doc):
    """Everything but the authored block — what `validate-engine` compares."""
    return {k: v for k, v in (doc or {}).items() if k != "reading"}


# ── the printout ─────────────────────────────────────────────────────────────

def _print(doc, abilities_for=None):
    m = doc["_meta"]
    cov = doc["coverage"]
    print(f"ENGINE GRAPH — {doc['slug']}{'@' + doc['branch'] if doc['branch'] else ''}  "
          f"list {m['decklist_sha256'][:12]}  forge {m['forge_digest']}  parser v{m['ability_graph_version']}")
    print(f"  read by Forge {cov['readers'].get('forge', 0)}, by the goldfish {cov['readers'].get('goldfish', 0)}, "
          f"by nothing {cov['readers'].get('none', 0)}; partial {len(cov['partial'])}; "
          f"AI:RemoveDeck {len(cov['ai_remove_deck'])}")
    if cov["unmapped_forge_names"]:
        print(f"  unmapped Forge names: {', '.join(cov['unmapped_forge_names'][:12])}"
              + (" …" if len(cov["unmapped_forge_names"]) > 12 else ""))
    print("\nSTAGES (computed; the reading may override a card with a why)")
    for st in doc["stages"]:
        members = st["cards"]
        print(f"  {st['stage']:11} {len(members):3}  " + ", ".join(members[:10]) + (" …" if len(members) > 10 else ""))
    if doc["unassigned"]:
        print(f"  {'unplaced':11} {len(doc['unassigned']):3}  "
              + "; ".join(f"{u['card']} ({u['why']})" for u in doc["unassigned"][:6]))
    print("\nCOMPONENTS (one per consumed event; odds = a producer in the opening seven / by turn three)")
    for c in doc["components"]:
        od = c["activation"]["producers"]["odds"]
        src = " +opp" if c["from_opponent"] else ""
        src += " +turn" if c["from_turn"] else ""
        thin = "  THIN" if c["thin"] else ""
        print(f"  {c['event']:14} producers {c['redundancy']['producers']:2}{src:6} consumers {c['redundancy']['consumers']:2}  "
              f"{od['opening_seven']:.2f} / {od['by_turn_three']:.2f}{thin}"
              + (f"  scaled by {', '.join(c['modifiers'])}" if c["modifiers"] else ""))
    print(f"\nLINES ({len(doc['lines'])}; ✓ a passing stack names every card, ◆ computed)")
    for l in doc["lines"][:12]:
        bn = l["bottleneck"]
        hop_words = []
        for h in l["hops"]:
            if h["stocks"]:
                hop_words.append(h["stocks"][0])
            elif h["n"] == 1:
                hop_words.append(h["cards"][0])
            elif h["cards"]:
                hop_words.append(f"{h['n']} cards ({h['cards'][0]} …)")
            else:
                hop_words.append(", ".join(h["tokens"]) or "?")
        print(f"  {l['id']} {l['stamp']} {' › '.join(l['events'])}: " + " → ".join(hop_words)
              + f"  [{l['starts']} start{'s' if l['starts'] != 1 else ''}, P3 {l['availability_by_turn_three']:.2f}]"
              + (f"  stacks {','.join(l['verified_by'])}" if l["verified_by"] else "")
              + (f"  thinnest hop {bn['n']}: {', '.join(bn['cards'][:3])}" if bn and bn["n"] <= ENGINE_THIN_GROUP else ""))
    if doc["spof"]:
        print("\nSINGLE POINTS OF FAILURE")
        for s in doc["spof"]:
            print(f"  {s['card']:32} {s['kind']:22} " + (f"{s.get('event', '')} " if s.get("event") else "")
                  + (f"severs {', '.join(s['severs'])}" if s["severs"] else ""))
    r = doc.get("reading")
    if r and r.get("thesis"):
        wf = str(r.get("written_for"))[:12] if r.get("written_for") else "an unstamped list"
        print(f"\nREADING ★ {'STALE — ' if r.get('stale') else ''}written for {wf}: {r['thesis'][:200]}")
    if abilities_for is not None:
        want = set(abilities_for) if abilities_for else None
        print("\nABILITIES")
        for n in doc["graph"]["nodes"]:
            if n["zone"] == "token" or (want and n["id"] not in want):
                continue
            c = n["coverage"]
            print(f"\n  {n['id']}  [{c['reader']}{' · partial' if c['partial'] else ''}]  stage {doc['placed'].get(n['id'], {}).get('stage')}")
            for a in n["abilities"]:
                if a["kind"] == "implicit":
                    continue
                cons = a.get("consumes")
                left = (f"{cons['event']}" + (f"({cons.get('actor')})" if cons.get("actor") else "")) if cons else \
                    " ".join(f"{k['event']}<{k.get('amount', '')}>" for k in a.get("costs") or []) or \
                    (f"scales {a['scales']['event']}" if a.get("scales") else "")
                prod = ", ".join(f"{p['event']}" + (f"×{p['amount']}" if p.get("amount") is not None else "")
                                 + (f"→{p['actor']}" if p.get("actor") else "") for p in a.get("produces") or [])
                print(f"    {a['kind']:11} {left:30} -> {prod or '—'}")
            if c["unmapped"] or c["loose"]:
                print(f"    unmapped: {', '.join(c['unmapped'] + ['loose:' + x for x in c['loose']])}")


def main(args):
    slug = args.slug
    branch = getattr(args, "branch", None)
    if getattr(args, "write", False):
        doc, path = write(slug, branch)
        print(f"wrote {path}")
        tdoc = load_json(deck_file(slug, "goldfish_targets.json", branch), default=None) or {}
        if tdoc.get("targets_from") == "engine.json":
            n = sum(1 for t in tdoc.get("targets") or [] if t.get("generated"))
            print(f"regenerated {n} goldfish target(s) from the components (targets_from: engine.json); "
                  f"the goldfish's next run reads them — `manamap pilot regen --slug {slug}`")
    else:
        doc = build(slug, branch)
        old = load_json(deck_file(slug, "engine.json", branch), default=None)
        doc["reading"] = migrate_reading(old, doc) if old else None
    if getattr(args, "as_json", False) or getattr(args, "json", False):
        print(json.dumps(doc, indent=1, ensure_ascii=False))
        return 0
    abilities = getattr(args, "abilities", None)
    _print(doc, abilities_for=abilities if abilities is not None else None)
    return 0


# ── AGREEMENT: the computed graph against three other witnesses (memo §6) ────────
#
# Reported as MEASUREMENTS, never pass/fail: (1) the hand-written goldfish
# declaration — Jaccard of its card set against the graph's engine set, and per
# declared group its best Jaccard against any component's producers or consumers;
# (2) Forge — Spearman of a node's feeds degree against its casts plus activations
# per game in the deck's largest logged run, over cards Forge's AI is allowed to
# play, with N; (3) the pilot — how the captain's-log card reads fall on hubs
# (degree at or above the engine's median), leaves, and cards outside the engine.

def _jaccard(a, b):
    a, b = set(a), set(b)
    return round(len(a & b) / len(a | b), 3) if (a | b) else None


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(xs, ys):
    """Spearman's rho with average ranks for ties; None under three points or a constant side."""
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None


def agreement(slug, doc=None):
    from manamap.pilot.scaffold_targets import _SUBSTRATE_EVENTS
    doc = doc or build(slug)
    base = deck_dir(slug)
    deck_nodes = {n["id"]: n for n in doc["graph"]["nodes"] if n["zone"] in ("command", "library")}
    commanders = {k for k, n in deck_nodes.items() if n["zone"] == "command"}
    comps = [c for c in doc["components"] if c["event"] not in _SUBSTRATE_EVENTS]
    engine_set = {x for c in comps for x in c["producers"] + list(c.get("deck_fed_consumers") or [])} - commanders
    out = {"slug": slug, "engine_cards": len(engine_set)}

    targets = load_json(base / "goldfish_targets.json") or {}
    groups = [g.get("any_of") or [] for t in targets.get("targets") or [] if not t.get("generated")
              for g in t.get("need") or []]
    declared = {x for g in groups for x in g if x in deck_nodes} - commanders
    if groups:
        comp_sets = [set(c["producers"]) for c in comps] + [set(c.get("deck_fed_consumers") or []) for c in comps]
        best = [max((_jaccard(set(g) - commanders, cs) or 0) for cs in comp_sets) if comp_sets else 0 for g in groups]
        out["declared"] = {"cards": len(declared), "jaccard": _jaccard(declared, engine_set),
                           "groups": len(groups), "group_best_jaccard_median": sorted(best)[len(best) // 2],
                           "declared_outside_graph": len(declared - engine_set),
                           "graph_outside_declared": len(engine_set - declared)}
    else:
        out["declared"] = None

    degree = {k: 0 for k in deck_nodes}
    for e in doc["graph"]["edges"]:
        if e["kind"] != "feeds":
            continue
        for end in (e["from"], e["to"]):
            if end in degree:
                degree[end] += 1
    runs = []
    for p in sorted((base / "sim").glob("*.json")) if (base / "sim").is_dir() else []:
        ec = (load_json(p) or {}).get("engine_casts") or {}
        if ec.get("by_card") and ec.get("games"):
            runs.append((ec["games"], p.name, ec))
    if runs:
        games, name, ec = max(runs)
        pts = [(degree[c], (row.get("cast", 0) + row.get("activated", 0)) / games)
               for c, row in ec["by_card"].items()
               if c in degree and c not in commanders and not deck_nodes[c]["coverage"].get("ai_remove_deck")
               and "Land" not in deck_nodes[c]["types"]]
        out["forge"] = {"run": name, "games": games, "n": len(pts),
                        "spearman_degree_vs_plays": spearman([a for a, _ in pts], [b for _, b in pts])}
    else:
        out["forge"] = None

    ann = load_json(base / "log_annotations.json") or {}
    eng_deg = sorted(degree[c] for c in engine_set if c in degree)
    median = eng_deg[len(eng_deg) // 2] if eng_deg else 0
    table = {}
    n_reads = 0
    for e in (ann.get("entries") or {}).values():
        for r in e.get("cards") or []:
            c, read = r.get("card"), r.get("read")
            if c not in deck_nodes or read is None:
                continue
            where = "hub" if c in engine_set and degree[c] >= median else "leaf" if c in engine_set else "outside"
            table.setdefault(where, {}).setdefault(read, 0)
            table[where][read] += 1
            n_reads += 1
    out["log_reads"] = {"n": n_reads, "hub_degree_at_least": median, "by_position": table} if n_reads else None
    return out
