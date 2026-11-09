"""A deck's ENGINE RESOURCE, and every card that makes or multiplies it.

WHY THIS EXISTS (2026-11-08). For a whole session the Meren search was steered by what
the goldfish could read: it pays experience on deaths and nothing else, so every
"double her experience" search looked for deaths, and Toph, Earthbending Master
(landfall experience) and the proliferate family never came up. The pilot found them
himself. The rule this module enforces: **the search is driven by the deck's declared
resource and the card text, never by the model's vocabulary**, and the model's reading
is reported BESIDE each card as a label — never used to drop one.

A deck declares its resources in `brief.json`:

    "resources": [{"name": "experience counters", "kind": "player_counter",
                   "makes": ["you get an? (?:\\w+ )?experience counters?"],
                   "spends": "Meren's end step: return a creature with MV <= experience"}]

`kind` picks a FIXED multiplier vocabulary (`MULTIPLIERS`), so nobody has to remember
that proliferate doubles experience: the kind remembers it.

READ OR BLIND is computed, never declared. A card's resource text is READ only when the
goldfish has a reader for it: the deck's commander when the deck declares a
`model_commander_*` channel, and — for a player counter under that channel — every card
whose experience profile the turn loop applies (`model_coverage._experience_read`: landfall
experience, Constrictor/Vorinclex replacements, a landfall or ETB proliferate; 2026-11-08).
Activated and cast proliferates (Yawgmoth, Drown in Ichor) stay BLIND. A creature the bodies channel reads is still BLIND here when
its resource line is not read: `model_coverage.card_state` calls Toph "seen" because she
is a 2/4, which is exactly the hole this closes.
"""

import json
import re

from manamap.pilot.common import deck_dir, load_deck_cards

KINDS = ("player_counter", "permanent_counter", "token", "trigger", "land_drop")

_COUNTER_MULT = (
    r"\bproliferate\b",
    r"\b(?:one|an) additional\b[^.]*\bcounters?\b",
    r"\bthat many plus one\b[^.]*\bcounters?\b|\bcounters?\b[^.]*\bthat many plus one\b",
    r"\btwice that many\b[^.]*\bcounters?\b|\bcounters?\b[^.]*\btwice that many\b",
    r"\bdouble the number of\b[^.]*\bcounters?\b",
)

#: COUNTERS ON A PLAYER (experience, energy, poison) are multiplied by proliferate and by
#: the few effects that name a PLAYER getting counters — never by "an additional +1/+1
#: counter" (Hardened Scales), which puts counters on permanents: the first sweep listed 136
#: cards with the permanent vocabulary, most of which do nothing for experience.
_PLAYER_COUNTER_MULT = (
    r"\bproliferate\b",
    r"\b(?:you|a player) would get one or more counters\b",
    r"\bcounters? on a permanent or player\b[^.]*\btwice that many\b",
)

#: What multiplies a resource of each kind, from card text. Fixed per kind on purpose:
#: a deck's brief names what MAKES its resource, and the bench remembers what multiplies it.
MULTIPLIERS = {
    "player_counter": _PLAYER_COUNTER_MULT,
    "permanent_counter": _COUNTER_MULT,
    "token": (
        r"\btwice that many\b[^.]*\btokens?\b|\btokens?\b[^.]*\btwice that many\b",
        r"\btokens? would be created\b[^.]*\binstead\b",
        r"\bthat many plus one\b[^.]*\btokens?\b|\btokens?\b[^.]*\bthat many plus one\b",
    ),
    "trigger": (
        r"\btriggers? an additional time\b",
    ),
    "land_drop": (
        r"\bplay (?:an|two|three) additional lands?\b",
        r"\bplay any number of lands\b",
        r"\bsearch your library for\b[^.]*\blands?\b[^.]*\bonto the battlefield\b",
        r"\breturn\b[^.]*\bland cards?\b[^.]*\bfrom your graveyard to the battlefield\b",
    ),
}

#: Multiplier patterns the goldfish's turn loop actually applies. EMPTY TODAY: there is no
#: proliferate, no counter doubler on players, no general landfall event (2026-11-08).
#: `tests/test_pilot_resources.py` greps the simulator for "proliferat" and fails when a
#: reader lands without this set being updated — a reader nobody told the gate about.
READ_MULTIPLIERS = frozenset()

_REMINDER = re.compile(r"\([^)]*\)")


def _strip(text):
    """Card text with reminder text removed: proliferate's reminder says "another counter
    of each kind", which would make every proliferate card a counter-doubler twice over and
    every keyword with counters in its reminder a hit."""
    return _REMINDER.sub(" ", text or "")


def validate(resources):
    """Errors for a brief's `resources` value (a list), as strings — never raises."""
    errors = []
    if not isinstance(resources, list) or not resources:
        return ["resources must be a non-empty list of {name, kind, makes, spends}"]
    for i, r in enumerate(resources):
        where = f"resources[{i}]"
        if not isinstance(r, dict):
            errors.append(f"{where} must be an object")
            continue
        if not str(r.get("name") or "").strip():
            errors.append(f"{where} has no name")
        if r.get("kind") not in KINDS:
            errors.append(f"{where}.kind must be one of {', '.join(KINDS)}, got {r.get('kind')!r}")
        makes = r.get("makes")
        if not isinstance(makes, list) or not makes:
            errors.append(f"{where}.makes must be a non-empty list of regexes")
            continue
        for p in makes:
            if not isinstance(p, str) or not p.strip():
                errors.append(f"{where}.makes has an empty pattern")
                continue
            try:
                re.compile(p)
            except re.error as exc:
                errors.append(f"{where}.makes {p!r} does not compile: {exc}")
    return errors


def declared(slug):
    """The deck's declared resources, or [] — read from its brief."""
    path = deck_dir(slug) / "brief.json"
    if not path.exists():
        return []
    return list((json.loads(path.read_text(encoding="utf-8")) or {}).get("resources") or [])


def _commander_reader(slug):
    """Commander names whose ability the goldfish reads: the deck declares a
    `model_commander_*` channel. Computed from the declaration, never authored."""
    from manamap.pilot.common import deck_file
    path = deck_file(slug, "goldfish_targets.json", None)
    doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not any(k.startswith("model_commander_") and v for k, v in (doc or {}).items()):
        return set()
    cards = load_deck_cards(slug).get("cards") or []
    return {c["name"] for c in cards if c.get("is_commander")}


def match(resource, text):
    """`(makes_hits, multiplier_hits)` — the patterns this card text satisfies."""
    body = _strip(text)
    mk = [p for p in resource.get("makes") or [] if re.search(p, body, re.I)]
    mu = [p for p in MULTIPLIERS.get(resource.get("kind"), ()) if re.search(p, body, re.I)]
    return mk, mu


def reading(resource, name, makes_hits, mult_hits, readers, card=None):
    """READ when the goldfish applies this card's resource text, else BLIND.

    `readers` is the commander set `_commander_reader` returns — non-empty exactly when the
    deck declares the commander channel the experience economy runs under. Beyond the
    commander herself, a player-counter card is READ when `model_coverage._experience_read`
    says the turn loop applies its text (Toph, Constrictor, Evolution Sage), computed from
    its profile, never declared."""
    if makes_hits and name in readers:
        return "read"
    if readers and card is not None and resource.get("kind") == "player_counter":
        from manamap.pilot import goldfish, model_coverage
        if model_coverage._experience_read(goldfish.classify(card)):
            return "read"
    if mult_hits and set(mult_hits) <= READ_MULTIPLIERS and not makes_hits:
        return "read"
    return "blind"


def engine_unread(slug, card, resources=None, readers=None):
    """The resources whose text this card carries and the goldfish does NOT read —
    `[]` when there are none. What `try` gates on beside `card_state`."""
    resources = declared(slug) if resources is None else resources
    if not resources:
        return []
    readers = _commander_reader(slug) if readers is None else readers
    out = []
    for r in resources:
        mk, mu = match(r, card.get("oracle_text") or "")
        if (mk or mu) and reading(r, card["name"], mk, mu, readers, card) == "blind":
            out.append(r["name"])
    return out


def scan(slug, only=None, limit=400):
    """Every corpus card in the deck's identity that makes or multiplies each declared
    resource, marked in-deck or not, with the goldfish's reading of its resource text
    and its `model_coverage.card_state`. Read-only."""
    from manamap.pilot import card_search, model_coverage

    resources = declared(slug)
    if not resources:
        raise SystemExit(
            f"{slug} declares no engine resource — add `resources` to "
            f"data/decks/{slug}/brief.json (see docs/pilot.md, resource-scan)")
    errors = validate(resources)
    if errors:
        raise SystemExit("brief.json resources: " + "; ".join(errors))
    if only:
        resources = [r for r in resources if r["name"].lower() == only.lower()]
        if not resources:
            raise SystemExit(f"no declared resource named {only!r}")
    identity = card_search.commander_identity(slug)
    in_deck = card_search.deck_names(slug)
    readers = _commander_reader(slug)
    flags, named = model_coverage.declaration(slug)
    out = []
    for r in resources:
        pats = list(r["makes"]) + list(MULTIPLIERS[r["kind"]])
        rows, meta = card_search.search(identity=identity, oracle=pats, limit=limit)
        cards = []
        for row in rows:
            mk, mu = match(r, row["oracle_text"])
            if not (mk or mu):                  # the hit was only in reminder text
                continue
            card = {"name": row["name"], "type_line": row["type_line"],
                    "oracle_text": row["oracle_text"]}
            cards.append({
                "name": row["name"], "mana_cost": row["mana_cost"], "cmc": row["cmc"],
                "type_line": row["type_line"],
                "land": "Land" in row["type_line"].split("//")[0],
                "group": "makes" if mk else "multiplies",
                "in_deck": row["name"] in in_deck,
                "reading": reading(r, row["name"], mk, mu, readers, card),
                "state": model_coverage.card_state(card, flags, named)["state"],
                "game_changer": row["game_changer"],
                "oracle_text": row["oracle_text"],
            })
        cards.sort(key=lambda c: (c["group"] != "makes", c["land"], c["name"]))
        out.append({"resource": r["name"], "kind": r["kind"], "spends": r.get("spends"),
                    "cards": cards, "truncated": meta.get("truncated", 0)})
    return {"slug": slug, "identity": sorted(identity), "resources": out}


def format_report(doc):
    L = [f"RESOURCE SCAN — {doc['slug']} · identity {''.join(doc['identity']) or 'colourless'}",
         "  READ = the goldfish applies the card's resource text; BLIND = it does not "
         "(judge it, model it, or play-test it — a BLIND card is never a cut or a pass)", ""]
    for r in doc["resources"]:
        L.append(f"{r['resource'].upper()} ({r['kind']})" + (f" — spent by: {r['spends']}" if r.get("spends") else ""))
        if r["truncated"]:
            L.append(f"  {r['truncated']} further match(es) not shown — raise --limit")
        for group in ("makes", "multiplies"):
            for land in (False, True):
                block = [c for c in r["cards"] if c["group"] == group and c["land"] == land]
                if not block:
                    continue
                L.append(f"  {group.upper()} — {'lands' if land else 'non-lands'} ({len(block)})")
                for c in block:
                    tag = "IN DECK" if c["in_deck"] else ""
                    gc = " ★GC" if c["game_changer"] else ""
                    L.append(f"    {c['name']}{gc}  [{c['reading'].upper()}"
                             f"{', ' + tag if tag else ''}]")
        L.append("")
    return "\n".join(L)


def main(args):
    doc = scan(args.slug, only=getattr(args, "resource", None),
               limit=getattr(args, "limit", None) or 400)
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(format_report(doc))
