"""A card's abilities as typed events — the engine graph's parser (no subcommand of
its own: `engine-graph <slug> --abilities` prints its reading).

A Forge card script says, in typed fields, what a card consumes and produces:

    T:Mode$ ChangesZone | Origin$ Battlefield | Destination$ Graveyard | ValidCard$ Card.Self,Creature.Other | Execute$ TrigLoseLife
    SVar:TrigLoseLife:DB$ LoseLife | ValidTgts$ Player | LifeAmount$ 1 | SubAbility$ DBGainLife
    SVar:DBGainLife:DB$ GainLife | Defined$ You | LifeAmount$ 1

Blood Artist CONSUMES a creature death and PRODUCES a life loss and a life gain. This
module reads that into `abilities[]` over the closed vocabulary `config.ENGINE_EVENTS`
(the memo `docs/research/engine-model.md` §3.2 is the argument for the vocabulary):

    {"kind": "trigger|activated|spell|static|replacement|keyword",
     "consumes": {"event": "DIES", "filters": [{"types": ["Creature"], "other": true}, {"self": true}]},
     "costs": [{"event": "DIES", "stock": "BODIES", "filters": [...]}],     # a Sac<…> cost
     "produces": [{"event": "LIFE_LOSS", "actor": "opponent", "amount": 1}, …],
     "scales": {"event": "DRAW", "actor": "you"},                           # an R: or S: line
     "optional": false, "zone": "battlefield|command|…", "text": "Whenever …"}

THREE RULES, and the second is the whole point. (1) The vocabulary is closed: a trigger
Mode, an effect API or a static Mode outside the registry is recorded under `unmapped`
with its Forge name and makes no event — never a guess. (2) A filter property outside
`ENGINE_FILTER_PROPERTIES` is kept under `unparsed` on that filter, so the edge it would
make downstream is UNPROVEN rather than silently kept or silently dropped (the same
posture `model_coverage` takes: a card the model cannot read is named, never a leaf).
(3) Where Forge has no script, the goldfish's own readers (`goldfish_library.classify`
→ `model_coverage.channels_for`) stand in, coarsely and labelled `reader: "goldfish"`;
no reader at all is `reader: "none"`.

Computed, never stored on its own: `deck_abilities()` is what `engine_graph` builds from,
and `card_events.json` (the fleet-wide cache the similarity space reads) is written by
the pipeline step that comes with it. Zero LLM calls.
"""
import json
import re

from manamap.config import (ABILITY_GRAPH_VERSION, ENGINE_CARD_TYPES, ENGINE_EVENTS,
                            ENGINE_FILTER_PROPERTIES)

#: The Forge parameter that names an effect's recipient, in the order it is trusted.
_ACTOR_KEYS = ("Defined", "ValidTgts", "ValidPlayer", "TokenOwner")
#: Amount parameters by effect API: the number a reader wants beside the event.
_AMOUNT_KEYS = ("NumCards", "LifeAmount", "NumDmg", "CounterNum", "Amount", "TokenAmount", "NumTokens")
_PARAM_RE = re.compile(r"\s*\|\s*")
_KV_RE = re.compile(r"^([A-Za-z]+)\$\s*(.*)$")
_COST_SAC_RE = re.compile(r"Sac<(\d+|X)/([^>/]+)(?:/[^>]*)?>")
_COST_DISCARD_RE = re.compile(r"Discard<(\d+|X)/([^>/]+)(?:/[^>]*)?>")
_COST_PAYLIFE_RE = re.compile(r"PayLife<(\d+|X)>")
_COLOR_WORDS = {"White": "W", "Blue": "U", "Black": "B", "Red": "R", "Green": "G"}
_PLAYER_WORDS = {"Player": "player", "You": "you", "Opponent": "opponent",
                 "Targeted": "player", "TriggeredActivator": "player", "TriggeredPlayer": "player"}
#: Single-capital property words that are NOT subtypes; they leave a filter unproven.
_SINGLE_CAP_PROPERTIES = frozenset((
    "Attacking", "Blocking", "Blocked", "Tapped", "Untapped", "Enchanted", "Equipped", "Kicked",
    "Remembered", "Imprinted", "Targeted", "Triggered", "Legendary", "Basic", "Snow", "Historic",
    "Monstrous", "Renowned", "Suspected", "Transformed", "Flipped", "Modal", "Face", "Phased",
    "Chosen", "Named", "Sacrificed", "Destroyed", "Exiled", "Returned", "Countered", "Copied",
    "Commander", "Companion", "Outlast", "Unblocked", "Paired", "Soulbond", "Goaded", "Suspended",
))
#: Effect APIs Forge itself models loosely; a card leaning on one is PARTIAL.
LOOSE_APIS = ("Charm", "Effect", "Branch", "GenericChoice", "Animate", "AnimateAll", "Repeat",
              "RepeatEach", "ChooseCard", "ChooseType", "ImmediateTrigger", "DelayedTrigger")
_SILENT_APIS = ("Cleanup", "StoreSVar", "Reveal", "Shuffle", "Pump", "PumpAll")


# ── the registry, inverted ──────────────────────────────────────────────────

def _matchers(field):
    """{forge name: [(event, required params)]} for one registry field."""
    out = {}
    for event, row in ENGINE_EVENTS.items():
        for m in row.get(field) or ():
            name, params = (m, {}) if isinstance(m, str) else (m[0], dict(m[1]))
            out.setdefault(name, []).append((event, params))
    return out


_TRIGGERS = _matchers("triggers")
_EFFECTS = _matchers("effects")
_REPLACEMENTS = _matchers("replacements")
_STATICS = _matchers("statics")
_KEYWORDS = _matchers("keywords")
_COSTS = _matchers("costs")


def _events_for(table, forge_name, params, top_only=False):
    """Every event the registry maps `forge_name` to, given the line's params,
    most specific first. `top_only` keeps the most specific tier alone: a trigger
    on `ChangesZone | Origin$ Battlefield | Destination$ Graveyard` is DIES, not
    DIES and LTB (an effect keeps every tier — a Token IS an ETB and a TOKEN)."""
    hits = []
    for event, need in table.get(forge_name, ()):
        if all(_param_has(params, k, v) for k, v in need.items()):
            hits.append((len(need), event))
    if not hits:
        return []
    hits.sort(key=lambda h: -h[0])
    if top_only:
        hits = [h for h in hits if h[0] == hits[0][0]]
    return [e for _, e in hits]


def _param_has(params, key, value):
    got = params.get(key)
    if got is None:
        return False
    return value.lower() in str(got).lower()


# ── the script's lines ───────────────────────────────────────────────────────

def _params(rest):
    """`Mode$ X | Origin$ Y | …` -> {"Mode": "X", "Origin": "Y", …}."""
    out = {}
    for part in _PARAM_RE.split(rest.strip()):
        m = _KV_RE.match(part.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _lines(text):
    """(prefix, body) for every script line that is not a comment or blank."""
    for raw in (text or "").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        if ":" not in raw:
            continue
        prefix, body = raw.split(":", 1)
        yield prefix.strip(), body


def _svars(text):
    out = {}
    for prefix, body in _lines(text):
        if prefix == "SVar" and ":" in body:
            name, value = body.split(":", 1)
            out[name.strip()] = value.strip()
    return out


# ── filters -> predicates ────────────────────────────────────────────────────

def parse_filter(expr):
    """A Forge `ValidCard$` expression -> a list of alternatives (comma = OR), each
    a predicate dict (plus = AND). `Creature.Black+Other+YouCtrl,Card.Self` ->
    [{types:["Creature"], colors:["B"], other:True, controller:"you"}, {self:True}].
    A property outside the whitelist lands in `unparsed` and the alternative is
    kept, so a consumer can say the edge is unproven rather than absent."""
    alts = []
    for alt in str(expr or "").split(","):
        alt = alt.strip()
        if not alt:
            continue
        head, _, props = alt.partition(".")
        pred = {}
        if head in ENGINE_CARD_TYPES:
            if head not in ("Card", "Permanent", "Spell"):
                pred["types"] = [head]
            else:
                pred["any"] = head
        elif head in _PLAYER_WORDS:
            pred["player"] = _PLAYER_WORDS[head]
        elif head in ("CARDNAME", "Self"):
            pred["self"] = True
        elif head:
            pred["subtypes"] = [head]
        for prop in [p for p in props.split("+") if p]:
            _apply_property(pred, prop)
        alts.append(pred)
    return alts


def _apply_property(pred, prop):
    if prop.startswith("!"):
        inner = prop[1:]
        if inner == "token":
            pred["token"] = False
        elif inner in ENGINE_CARD_TYPES:
            pred.setdefault("not_types", []).append(inner)
        elif inner in ("YouCtrl", "YouOwn"):
            pred["controller"] = "opponent"
        elif inner in ("OppCtrl", "OppOwn"):
            pred["controller"] = "you"
        else:
            pred.setdefault("unparsed", []).append(prop)
        return
    if "player" in pred and prop in _PLAYER_WORDS:
        pred["player"] = _PLAYER_WORDS[prop]
    elif prop in ("Self", "StrictlySelf", "CARDNAME"):
        pred["self"] = True
    elif prop == "Other":
        pred["other"] = True
    elif prop in ("YouCtrl", "YouOwn"):
        pred["controller"] = "you"
    elif prop in ("OppCtrl", "OppOwn"):
        pred["controller"] = "opponent"
    elif prop == "token":
        pred["token"] = True
    elif prop == "nonToken":
        pred["token"] = False
    elif prop in _COLOR_WORDS:
        pred.setdefault("colors", []).append(_COLOR_WORDS[prop])
    elif prop == "Colorless":
        pred["colors"] = []
    elif prop.startswith("non") and prop[3:] in ENGINE_CARD_TYPES:
        pred.setdefault("not_types", []).append(prop[3:])
    elif prop in ENGINE_CARD_TYPES:
        pred.setdefault("types", []).append(prop)
    elif re.match(r"^(power|toughness|cmc)(LE|GE|LT|GT|EQ)(\d+)$", prop):
        m = re.match(r"^(power|toughness|cmc)(LE|GE|LT|GT|EQ)(\d+)$", prop)
        pred.setdefault("bounds", []).append([m.group(1), m.group(2), int(m.group(3))])
    elif prop in ENGINE_FILTER_PROPERTIES:
        pred.setdefault("flags", []).append(prop)
    elif prop and prop.isalpha() and prop[0].isupper() and sum(c.isupper() for c in prop) == 1 \
            and prop not in _SINGLE_CAP_PROPERTIES:
        # One capital, letters only, not a known property word: a subtype
        # (Creature.Vampire, Card.Shark). Forge's own property names are CamelCase
        # with two capitals or more (IsRemembered, ControllerControls) or start in
        # lower case (inZoneBattlefield, powerLE2), and land in `unparsed` below.
        pred.setdefault("subtypes", []).append(prop)
    else:
        pred.setdefault("unparsed", []).append(prop)


def subject_of(card):
    """A card's abilities row -> the SUBJECT a filter is checked against: its
    types, subtypes, colours and controller (`you` for everything in the deck)."""
    return {"types": [t for t in card.get("types") or [] if t not in ("Legendary", "Basic", "Snow")],
            "subtypes": list(card.get("subtypes") or []), "colors": _colors_of(card),
            "controller": "you", "token": bool(card.get("is_token")), "name": card.get("name")}


def _colors_of(card):
    if card.get("colors"):
        return list(card["colors"])
    cost = card.get("mana_cost") or ""
    out = []
    for sym in cost.replace("/", " ").split():
        for ch in sym:
            if ch in "WUBRG" and ch not in out:
                out.append(ch)
    return out


def filter_accepts(alts, subject, *, source=None):
    """Does a parsed filter (one or more alternatives) accept this subject? An
    alternative with an `unparsed` property answers None — UNPROVEN, neither yes
    nor no — and the caller keeps the edge dashed. `source` is the name of the
    card whose filter this is, for `Self` / `Other`. An empty filter accepts."""
    if not alts:
        return True
    unproven = False
    for alt in alts:
        verdict = _alt_accepts(alt, subject, source)
        if verdict is True:
            return True
        if verdict is None:
            unproven = True
    return None if unproven else False


def _alt_accepts(alt, subject, source):
    if alt.get("player"):
        return False                                  # a player filter never accepts a card
    name = subject.get("name")
    if alt.get("self") and not (source and name == source):
        return False
    if alt.get("other") and source and name == source:
        return False
    for t in alt.get("types") or []:
        if t == "Token":
            if not subject.get("token"):
                return False
        elif t not in (subject.get("types") or []):
            return False
    for t in alt.get("not_types") or []:
        if t in (subject.get("types") or []):
            return False
    for st in alt.get("subtypes") or []:
        if st not in (subject.get("subtypes") or []):
            return False
    if "colors" in alt:
        want = alt["colors"]
        have = subject.get("colors") or []
        if not want and have:
            return False                              # Colorless wanted
        if want and not any(c in have for c in want):
            return False
    if alt.get("controller") and subject.get("controller") and alt["controller"] != subject["controller"]:
        return False
    if "token" in alt and bool(subject.get("token")) != alt["token"]:
        return False
    if alt.get("bounds"):
        pt = subject.get("pt")
        if not pt:
            return None
        try:
            power, tough = (int(x) for x in pt.split("/"))
        except ValueError:
            return None
        for stat, op, n in alt["bounds"]:
            v = {"power": power, "toughness": tough}.get(stat)
            if v is None:
                return None
            ok = {"LE": v <= n, "GE": v >= n, "LT": v < n, "GT": v > n, "EQ": v == n}[op]
            if not ok:
                return False
    if alt.get("unparsed") or alt.get("flags"):
        return None
    return True


def _actor(params):
    """Who an effect is aimed at: you / opponent / player / self / target."""
    for key in _ACTOR_KEYS:
        v = params.get(key)
        if not v:
            continue
        low = v.lower()
        if "opponent" in low:
            return "opponent"
        if low in ("you", "player.you"):
            return "you"
        if low in ("self", "card.self"):
            return "self"
        if "player" in low or "triggeredcardcontroller" in low or "triggeredactivator" in low:
            return "player"
        return "target"
    return "you"


def _amount(params, svars):
    for key in _AMOUNT_KEYS:
        v = params.get(key)
        if v is None:
            continue
        if re.fullmatch(r"-?\d+", v):
            return int(v)
        # `X` names an SVar; a Count$ there is a board-dependent amount.
        ref = svars.get(v, v)
        return "X" if ("Count$" in ref or v in ("X", "Y", "Z") or "Count" in ref) else v
    return None


# ── effects, walked through their SubAbility chain ───────────────────────────

def _effect_chain(svar_name, svars, seen=None):
    """Every effect API reached from an SVar through SubAbility$/Choices$/Charm."""
    seen = seen or set()
    out = []
    if not svar_name or svar_name in seen:
        return out
    seen.add(svar_name)
    body = svars.get(svar_name)
    if body is None:
        return out
    params = _params(body)
    api = params.get("DB") or params.get("SP") or params.get("AB")
    if api:
        out.append((api, params))
    for nxt in (params.get("SubAbility"), params.get("Execute")):
        if nxt:
            out.extend(_effect_chain(nxt, svars, seen))
    if params.get("Choices"):
        for choice in params["Choices"].split(","):
            out.extend(_effect_chain(choice.strip(), svars, seen))
    return out


def _produced(api, params, svars, unmapped):
    """One effect line -> the events it produces (possibly several: a Token is an
    ETB and a TOKEN and, by its script, maybe a TREASURE)."""
    events = _events_for(_EFFECTS, api, params)
    if api in LOOSE_APIS:
        unmapped.append(f"loose:{api}")
    if not events:
        if api not in LOOSE_APIS and api not in _SILENT_APIS:
            unmapped.append(f"effect:{api}")
        return []
    out = []
    amount = _amount(params, svars)
    actor = _actor(params)
    for ev in events:
        row = {"event": ev, "actor": actor}
        if amount is not None:
            row["amount"] = amount
        if api == "Token" and params.get("TokenScript"):
            row["token"] = params["TokenScript"]
        if api in ("PutCounter", "PutCounterAll", "RemoveCounter") and params.get("CounterType"):
            row["counter"] = params["CounterType"]
        if api in ("Pump", "PumpAll", "Animate") and params.get("KW"):
            row["keywords"] = [k.strip() for k in params["KW"].split("&")]
        for key in ("ValidTgts", "ValidCards", "Defined"):
            if params.get(key) and key != "Defined":
                row["targets"] = parse_filter(params[key])
                break
        out.append(row)
    return out


def _costs(cost, unmapped):
    """`Sac<1/Creature>` consumes a BODIES stock and produces a DIES; `Discard<1/Card>`
    produces a DISCARD; `T` is a TAP of self; mana is the MANA stock."""
    out = []
    if not cost:
        return out
    for m in _COST_SAC_RE.finditer(cost):
        out.append({"event": "DIES", "stock": "BODIES", "amount": m.group(1),
                    "filters": parse_filter(m.group(2))})
    for m in _COST_DISCARD_RE.finditer(cost):
        out.append({"event": "DISCARD", "stock": "HAND", "amount": m.group(1),
                    "filters": parse_filter(m.group(2))})
    for m in _COST_PAYLIFE_RE.finditer(cost):
        out.append({"event": "LIFE_LOSS", "stock": "LIFE", "amount": m.group(1), "actor": "you"})
    if re.search(r"(^|\s)T(\s|$)", cost):
        out.append({"event": "TAP", "stock": None, "filters": [{"self": True}]})
    mana = [t for t in cost.split() if re.fullmatch(r"[0-9WUBRGCSXP/]+", t) and re.search(r"[0-9WUBRGCX]", t)]
    if mana:
        out.append({"event": "MANA", "stock": "MANA", "amount": " ".join(mana)})
    return out


# ── a whole script ───────────────────────────────────────────────────────────

def parse_script(text):
    """A Forge script -> {name, types, subtypes, colors, mana_cost, abilities,
    unmapped, keywords}. Pure; the same text always gives the same dict."""
    svars = _svars(text)
    doc = {"name": None, "types": [], "subtypes": [], "colors": [], "mana_cost": None,
           "pt": None, "keywords": [], "abilities": [], "unmapped": [], "loose": [], "deck_hints": {}}
    for prefix, body in _lines(text):
        body = body.strip()
        if prefix == "Name":
            doc["name"] = body
        elif prefix == "AI" and body.startswith("RemoveDeck"):
            doc["ai_remove_deck"] = body.split(":", 1)[-1]
        elif prefix == "ManaCost":
            doc["mana_cost"] = body
        elif prefix == "PT":
            doc["pt"] = body
        elif prefix == "Types":
            words = body.split()
            doc["types"] = [w for w in words if w in ENGINE_CARD_TYPES or w in ("Legendary", "Basic", "Snow")]
            doc["subtypes"] = [w for w in words if w not in doc["types"]]
        elif prefix == "Colors":
            doc["colors"] = [_COLOR_WORDS.get(w.capitalize(), w) for w in body.split(",")]
        elif prefix == "K":
            doc["keywords"].append(body)
            _keyword_ability(doc, body)
        elif prefix == "T":
            _trigger(doc, body, svars)
        elif prefix == "A":
            _activated(doc, body, svars)
        elif prefix == "S":
            _static(doc, body)
        elif prefix == "R":
            _replacement(doc, body, svars)
        elif prefix in ("DeckHas", "DeckHints", "DeckNeeds"):
            doc["deck_hints"][prefix] = body
    doc["loose"] = sorted({u.split(":", 1)[1] for u in doc["unmapped"] if u.startswith("loose:")})
    doc["unmapped"] = sorted({u for u in doc["unmapped"] if not u.startswith("loose:")})
    return doc


def _trigger(doc, body, svars):
    params = _params(body)
    mode = params.get("Mode", "")
    events = _events_for(_TRIGGERS, mode, params, top_only=True)
    produces = []
    for api, p in _effect_chain(params.get("Execute"), svars):
        produces.extend(_produced(api, p, svars, doc["unmapped"]))
    if not events:
        doc["unmapped"].append(f"trigger:{mode}")
        return
    filt = params.get("ValidCard") or params.get("ValidCards") or params.get("ValidPlayer") \
        or params.get("ValidSource") or ""
    actor = None
    for key in ("ValidActivatingPlayer", "ValidPlayer"):
        if params.get(key):
            actor = "opponent" if "Opponent" in params[key] else ("you" if "You" in params[key] else "player")
    filters = parse_filter(filt)
    if actor is None:
        # `Card.OppOwn` / `Card.YouCtrl` on a Drawn or Discarded trigger name whose event it is.
        for alt in filters:
            if alt.get("controller"):
                actor = alt["controller"]
    zone = (params.get("TriggerZones") or "Battlefield").lower()
    for ev in events:
        doc["abilities"].append({
            "kind": "trigger",
            "consumes": {"event": ev, "filters": filters, "actor": actor,
                         **({"origin": params["Origin"]} if params.get("Origin") else {}),
                         **({"destination": params["Destination"]} if params.get("Destination") else {})},
            "produces": produces,
            "optional": bool(params.get("OptionalDecider")),
            "zone": zone,
            "text": params.get("TriggerDescription", "")[:200],
        })


def _activated(doc, body, svars):
    params = _params(body)
    api = params.get("AB") or params.get("SP")
    kind = "activated" if params.get("AB") else "spell"
    produces = list(_produced(api, params, svars, doc["unmapped"])) if api else []
    for nxt in (params.get("SubAbility"),):
        for a2, p2 in _effect_chain(nxt, svars):
            produces.extend(_produced(a2, p2, svars, doc["unmapped"]))
    if api == "Charm":
        for choice in (params.get("Choices") or "").split(","):
            for a2, p2 in _effect_chain(choice.strip(), svars):
                produces.extend(_produced(a2, p2, svars, doc["unmapped"]))
    costs = _costs(params.get("Cost"), doc["unmapped"]) if kind == "activated" else []
    doc["abilities"].append({
        "kind": kind, "consumes": None, "costs": costs, "produces": produces,
        "optional": False, "zone": "battlefield" if kind == "activated" else "hand",
        "text": params.get("SpellDescription", "")[:200],
        **({"charm": True} if api == "Charm" else {}),
    })


def _static(doc, body):
    params = _params(body)
    mode = params.get("Mode", "")
    events = _events_for(_STATICS, mode, params)
    if not events and mode == "Panharmonicon" and params.get("ValidMode"):
        # Forge's generic "triggers an additional time": it scales whatever event
        # the named trigger Mode consumes (Teysa Karlov: ChangesZone; Roaming
        # Throne: by type). Read through the trigger table, never guessed.
        events = _events_for(_TRIGGERS, params["ValidMode"].split(",")[0].strip(), params, top_only=True)
    if not events:
        doc["unmapped"].append(f"static:{mode}")
        return
    row = {"kind": "static", "consumes": None, "produces": [],
           "scales": {"event": events[0]}, "zone": "battlefield",
           "text": params.get("Description", "")[:200]}
    if params.get("Affected"):
        row["scales"]["filters"] = parse_filter(params["Affected"])
    if params.get("AddKeyword"):
        row["scales"]["keywords"] = [k.strip() for k in params["AddKeyword"].split("&")]
    if params.get("AddPower") or params.get("AddToughness"):
        row["scales"]["power"] = params.get("AddPower")
    if mode in ("ReduceCost", "AlternativeCost"):
        row["scales"]["event"] = "COST_REDUCTION"
    doc["abilities"].append(row)


def _replacement(doc, body, svars):
    params = _params(body)
    ev_name = params.get("Event", "")
    events = _events_for(_REPLACEMENTS, ev_name, params)
    if not events:
        doc["unmapped"].append(f"replacement:{ev_name}")
        return
    actor = None
    for key in ("ValidPlayer", "ValidSource", "ActivePlayer"):
        if params.get(key):
            actor = "opponent" if "Opponent" in params[key] else ("you" if "You" in params[key] else "player")
    row = {"kind": "replacement", "consumes": None, "produces": [],
           "scales": {"event": events[0], "actor": actor}, "zone": "battlefield",
           "text": params.get("Description", "")[:200]}
    for key in ("ValidCard", "ValidToken", "ValidObject", "ValidTarget", "ValidSource"):
        if params.get(key):
            row["scales"]["filters"] = parse_filter(params[key])
            break
    if params.get("PlayerTurn"):
        row["scales"]["your_turn"] = True
    doc["abilities"].append(row)


def _keyword_ability(doc, body):
    word = body.split(":")[0].strip()
    events = _events_for(_KEYWORDS, word, {})
    for ev in events:
        doc["abilities"].append({"kind": "keyword", "consumes": None, "produces": [{"event": ev, "actor": "you"}],
                                 "optional": False, "zone": "battlefield", "text": body[:80]})


# ── the goldfish's readers as the second reader ──────────────────────────────

_CHANNEL_EVENT = {"mana": "MANA", "bodies": "ETB", "tutor": "TUTOR", "treasure": "TREASURE",
                  "combat": "COMBAT_DAMAGE", "draw": "DRAW", "sacrifice": "DIES", "drain": "LIFE_LOSS",
                  "discard": "DISCARD", "recursion": "RECURSION", "attack_enabler": "ATTACK",
                  "discover": "TUTOR", "cascade": "CAST", "target": None}


def _goldfish_abilities(card):
    """Coarse abilities from the goldfish's own card readers, for a card Forge has no
    script for. Labelled, never confused with a parsed script."""
    try:
        from manamap.pilot import goldfish_library, model_coverage
        profile = goldfish_library.classify(card)
        channels = model_coverage.channels_for(profile)
    except Exception:                                # noqa: BLE001 - the reader is optional
        return []
    out = []
    for ch in sorted(channels):
        ev = _CHANNEL_EVENT.get(ch)
        if ev:
            out.append({"kind": "goldfish", "consumes": None, "produces": [{"event": ev, "actor": "you"}],
                        "optional": False, "zone": "battlefield", "text": f"the goldfish's {ch} reader"})
    return out


# ── a card, a deck ───────────────────────────────────────────────────────────

def card_abilities(name, script_text, card=None):
    """One card's typed abilities with its coverage. `reader` is `forge` when a
    script was read, `goldfish` when only the goldfish's readers could, `none`
    otherwise; `partial` is set when the script carried lines the registry does
    not map, which are listed under `unmapped`."""
    if script_text:
        doc = parse_script(script_text)
        return {"name": name, "reader": "forge", "partial": bool(doc["unmapped"] or doc["loose"]),
                "types": doc["types"], "subtypes": doc["subtypes"], "colors": doc["colors"],
                "mana_cost": doc["mana_cost"], "pt": doc["pt"], "keywords": doc["keywords"],
                "abilities": doc["abilities"], "unmapped": doc["unmapped"], "loose": doc["loose"],
                "ai_remove_deck": doc.get("ai_remove_deck"), "deck_hints": doc["deck_hints"]}
    abilities = _goldfish_abilities(card) if card else []
    return {"name": name, "reader": "goldfish" if abilities else "none", "partial": False,
            "types": [], "subtypes": [], "colors": [], "mana_cost": None, "pt": None, "keywords": [],
            "abilities": abilities, "unmapped": [], "loose": [], "ai_remove_deck": None, "deck_hints": {}}


def deck_abilities(slug, branch=None):
    """{name: card_abilities} for a deck's whole list, commanders included, read
    through `forge_cards.scripts` (one zip open) with the goldfish as the fallback."""
    from manamap.pilot.common import load_deck_cards
    from manamap.sim import forge_cards
    doc = load_deck_cards(slug, branch)
    rows = doc.get("cards") or []
    names = [c["name"] for c in rows]
    texts = forge_cards.scripts(names)
    by_name = {c["name"]: c for c in rows}
    return {n: card_abilities(n, texts.get(n), by_name.get(n)) for n in names}


def coverage(cards):
    """How much of a list the parser read, in the goldfish's own words: counts per
    reader, the partial ones, and every unmapped Forge name seen."""
    readers = {"forge": 0, "goldfish": 0, "none": 0}
    partial, unmapped = [], set()
    for name, row in cards.items():
        readers[row["reader"]] += 1
        if row.get("partial"):
            partial.append(name)
        unmapped.update(row.get("unmapped") or [])
    return {"readers": readers, "partial": sorted(partial), "unmapped": sorted(unmapped),
            "version": ABILITY_GRAPH_VERSION}

