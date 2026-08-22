"""Pilot: what deck can I build from a box of cards?

Every other command in this subsystem is keyed on a deck slug and assumes a legal
Commander 100. That is the wrong shape for the question a pile of cardboard asks.
`deck-facts` on a 764-card collection reports hypergeometrics computed against a
99-card library (`manabase` is hardwired to one) and `validate-deck` emits roughly
a thousand errors — both are answering a question nobody asked. Note also that
"pool" everywhere else in this repo means the *format-wide* 34K pool; here it
means the cards you physically own.

The question this answers is "what should I build", and it is decided by four
numbers, only one of which is obvious:

  depth        how many owned cards fall inside a commander's colour identity
  castability  how many owned SOURCES those colours actually have
  roles        whether the owned cards cover a 99's jobs, or leave holes
  combos       which lines are fully contained in what you own

**Depth without castability is the trap this module exists to close.** On the
collection it was written for, Atraxa showed 663 playable cards — the deepest in
the box — and Sultai showed 587 including the single most popular two-card win in
the format. Both are illusions: the box holds exactly one blue source. A shortlist
ranked on depth alone recommends a deck that cannot cast its own spells, and it
does so confidently.

Every combo line reported here is a **candidate, not a fact**. Commander Spellbook
is format-agnostic and its lines can quietly assume a piece is your commander;
`bracket.assumes_other_commander` filters the ones that name the tell, but a line
only becomes evidence after a resolve-stack run. `verified: false` is on every one
of them, and it is not decoration.

**Computed on demand, never committed** — same rule as `deck-facts` and
`artist-credits`. The output is a view of files you already have.

Composes primitives that already exist; no new analysis lives here:

    parsing        pilot/fetch_deck.py     parse_decklist   (quantities, *F*, *CMDR*)
    commanders     pilot/common.py         commander_rejection
    identity       analysis/common.py      parse_color_identity, parse_tag_set
    sources        pilot/manabase.py       land_colors      (restriction-aware)
    combos         pilot/bracket.py        assess, is_infinite
    role budget    pilot/build_deck.py     role_group
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from manamap.analysis.common import WUBRG, parse_color_identity, parse_tag_set
from manamap.config import (
    DECK_ROLE_BUDGET,
    OBSOLESCENCE_INDEX_PATH,
    OUTPUT_CSV_PATH,
)
from manamap.pilot.bracket import assess, is_infinite
from manamap.pilot.build_deck import role_group
from manamap.pilot.card_pool import load_frame
from manamap.pilot.common import (
    commander_rejection,
    is_land,
    load_card_roles,
    load_combo_details,
    load_json_memo,
    mtime_memo,
)
from manamap.pilot.fetch_deck import parse_decklist
from manamap.pilot.manabase import land_colors

# Columns the analysis actually reads. cards.csv is 24.7 MB across 35 columns and
# this is a CLI people run repeatedly while deciding what to build.
_COLUMNS = [
    "name", "layout", "type_line", "oracle_text", "color_identity", "cmc",
    "edhrec_rank", "game_changer", "legal_commander", "mechanical_tags",
]

# Report the deepest lines per identity rather than all of them; a five-colour
# box contains hundreds and the tail is noise. Truncation is always announced.
COMBO_REPORT_LIMIT = 25
ARCHETYPE_REPORT_LIMIT = 18


def _read_cards():
    # Slice to this module's columns rather than handing out the full union:
    # these records are passed around as "a card row" and widening them would
    # quietly change what downstream code sees in a `.keys()` or a dump.
    return {r["name"]: r
            for r in load_frame()[_COLUMNS].to_dict("records")}


def load_cards():
    """The cards.csv slice this module reads, as {name: row-dict}.

    Built from `card_pool.load_frame()`, so a `pool-facts` run parses the 24 MB
    CSV once for this, the bracket engine and everything else together — it used
    to be parsed here and again inside `bracket.load_reference`, with
    `build_deck` calling in through `resolve_pool` for a third. Treat the result
    as read-only; callers share one parse.
    """
    if not OUTPUT_CSV_PATH.exists():
        raise SystemExit(
            f"{OUTPUT_CSV_PATH} not found — run `manamap extract` first. "
            f"A pool cannot be analysed without the card database."
        )
    return mtime_memo(OUTPUT_CSV_PATH, "pool_facts:cards", _read_cards)


def front_face_map(cards):
    """{front face → joined name} for multi-face cards.

    cards.csv and every global graph key on the joined `" // "` form; decklists
    carry the front face alone. Skipping this translation silently drops ~2.5%
    of the corpus from every join, and drops it *quietly* — the card is simply
    absent from the results rather than reported as unresolved.
    """
    return {name.split(" // ")[0]: name for name in cards if " // " in name}


def read_sources(paths, cards):
    """Parse decklist files into {file: {name: copies}} plus the unresolved names."""
    front = front_face_map(cards)
    per_file = {}
    unresolved = defaultdict(list)
    for path in paths:
        counts = Counter()
        for entry in parse_decklist(Path(path).read_text()):
            name = entry["name"]
            resolved = name if name in cards else front.get(name)
            if resolved is None:
                unresolved[name].append(Path(path).name)
                continue
            counts[resolved] += int(entry.get("quantity") or 1)
        per_file[Path(path).name] = dict(counts)
    return per_file, {n: sorted(set(f)) for n, f in unresolved.items()}


def pool_printings(paths, cards):
    """{name: {set, collector_number, foil}} — the PRINTING you actually own.

    `parse_decklist` already captures set, collector number and foil from a
    decklist's `(SET) 123 *F*` suffix; `read_sources` deliberately drops all of it
    because counting copies does not care. Building a deck from a physical
    collection does care, and dropping it here is how a 100-card list of cards you
    own became a Scryfall default for every one of them — the legacy page
    credited Marvel Super Heroes Commander art for a Sol Ring that is nothing of
    the sort, and Featured Artist counts artists per printing.

    First occurrence wins. A card in two boxes is one card; which sleeve it came
    out of is not a decision this function should invent.
    """
    front = front_face_map(cards)
    out = {}
    for path in paths:
        for entry in parse_decklist(Path(path).read_text()):
            name = entry["name"]
            resolved = name if name in cards else front.get(name)
            if resolved is None or resolved in out:
                continue
            if not entry.get("set"):
                continue
            out[resolved] = {
                "set": entry["set"],
                "collector_number": entry.get("collector_number"),
                "foil": bool(entry.get("foil")),
            }
    return out


def collect_paths(targets, exclude):
    """Expand files and directories into a sorted list of decklists to read."""
    excluded = {Path(e).resolve() for e in exclude or []}
    found = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            found.extend(sorted(path.glob("*.txt")))
        elif path.exists():
            found.append(path)
        else:
            raise SystemExit(f"No such file or directory: {target}")
    kept = [p for p in found if p.resolve() not in excluded]
    if not kept:
        raise SystemExit(f"No decklist files found in {', '.join(map(str, targets))}")
    return kept


# ── The four numbers ────────────────────────────────────────────────────


def identity_key(identity):
    """A stable WUBRG-ordered label for a colour identity set ('C' if colourless)."""
    return "".join(c for c in WUBRG if c in identity) or "C"


def sources_for(pool, cards, identity):
    """Owned mana sources per colour, counting COPIES, restriction-aware.

    Uses `manabase.land_colors`, which refuses to count restricted mana ("spend
    this only to cast a Dragon") as a general source. It understates rather than
    overstates, which is the correct direction: an understated source is a deck
    that runs one land too many, an overstated one is a deck that cannot cast
    its spells.
    """
    counts = Counter()
    for name, copies in pool.items():
        row = cards[name]
        if not is_land(row) or not (card_identity(row) <= identity):
            continue
        for colour in land_colors(row) & identity:
            counts[colour] += copies
    return {c: counts.get(c, 0) for c in sorted(identity, key=WUBRG.index)}


def card_identity(row):
    return parse_color_identity(row.get("color_identity", ""))


def in_identity(pool, cards, identity):
    """The owned cards a commander of this identity could legally play."""
    return [n for n in pool if card_identity(cards[n]) <= identity]


def role_coverage(names, cards, roles):
    """Role-group histogram over the non-land cards, the budget's holes, and the
    size of the blind spot underneath both.

    The budget is `DECK_ROLE_BUDGET`, the same one `build_deck` fills against, so
    a hole here is a hole the deterministic builder would hit too.

    **A hole is only as trustworthy as the taxonomy's coverage, so report the
    coverage next to it.** `card_roles.json` classifies ~89% of the corpus and
    omissions are silent — a card simply has no entry and lands in `flex`.
    Measured on a real deck: Necropotence classifies as `stax`, and Sylvan
    Library, Ad Nauseam and Painful Truths have no entry at all, so a deck with
    six genuine draw engines reported `draw -8`. "Your deck has no card draw" is
    exactly the kind of confidently wrong answer this subsystem exists to avoid.
    Absence of a role is not absence of the function.

    `role_group` also resolves multi-role cards by `DECK_ROLE_GROUPS` priority,
    not by importance: Deadly Dispute holds `draw:burst` *and* `ramp:ritual`, and
    ramp is checked first, so it counts as ramp. That ordering is `build_deck`'s
    frozen contract and is deliberately not second-guessed here.
    """
    histogram = Counter()
    unclassified = 0
    for name in names:
        if is_land(cards[name]):
            continue
        card_roles = roles.get(name)
        if not card_roles:
            unclassified += 1
        histogram[role_group(card_roles or [])] += 1
    holes = {
        group: budget - histogram.get(group, 0)
        for group, budget in DECK_ROLE_BUDGET.items()
        if group != "lands" and histogram.get(group, 0) < budget
    }
    return dict(sorted(histogram.items())), dict(sorted(holes.items())), unclassified


def contained_combos(pool, cards, details):
    """Every combo line fully contained in the pool, deduplicated.

    The raw artifact holds distinct records for the same set of cards (variants
    of one interaction), so a straight read double-counts: Springheart Nantuko +
    Tireless Provisioner appears twice in the collection this was written for.
    Dedupe on the card set and keep the most popular record.
    """
    combos = details["combos"]
    candidates = set()
    for name in pool:
        candidates.update(details["by_card"].get(name, []))

    best = {}
    for i in candidates:
        combo = combos[i]
        if not set(combo["cards"]) <= pool.keys():
            continue
        key = frozenset(combo["cards"])
        if key not in best or (combo.get("popularity") or 0) > (best[key].get("popularity") or 0):
            best[key] = combo

    lines = []
    for combo in best.values():
        identity = set().union(*(card_identity(cards[c]) for c in combo["cards"]))
        lines.append({
            "cards": sorted(combo["cards"]),
            "produces": combo.get("produces", []),
            "bracket": None if combo.get("banned") else combo.get("bracket"),
            "banned": bool(combo.get("banned")),
            "popularity": combo.get("popularity") or 0,
            "mana_value_needed": combo.get("mana_value_needed"),
            "identity": identity_key(identity),
            "infinite": is_infinite(combo),
            # Never verified by containment alone. Spellbook is format-agnostic
            # and a line here is a candidate for resolve-stack, not evidence.
            "verified": False,
        })
    lines.sort(key=lambda c: (-c["popularity"], c["cards"]))
    return lines


def obsolete_in_pool(pool, index):
    """Owned cards a *different owned card* outclasses — candidate free upgrades.

    CANDIDATES, not verdicts. `obsolescence_index.json` compares tags, mana value,
    colour, stats and date; it does not read restriction clauses, so a card whose
    ability is gated on a creature type reads as a superset of one that isn't.
    Four of the first eight it produced on a real box were wrong in that exact
    way — the damaging one offered Boggart Mischief (drains only when a *Goblin*
    dies, in a box with almost no Goblins) as a replacement for Bastion of
    Remembrance, which drains on any creature. Two more inverted the direction
    for a sacrifice deck, and one compared Dismember on mana value while ignoring
    that Phyrexian mana makes it a one-mana spell.

    Same failure shape as counting mana sources by grepping oracle text: the
    index answers a structural question and gets read as a functional one.
    """
    out = []
    for name in sorted(pool):
        entry = index.get(name)
        if not entry:
            continue
        better = [e for e in entry.get("obsoleted_by", []) if e["name"] in pool]
        if better:
            out.append({
                "card": name,
                "replaced_by": [
                    {"name": e["name"], "advantages": e.get("advantages", [])}
                    for e in sorted(better, key=lambda e: -e.get("similarity", 0))[:3]
                ],
            })
    return out


# ── Assembly ────────────────────────────────────────────────────────────


def analyze(targets, exclude=None):
    """Everything a strategist needs to pick a commander from a box of cards."""
    cards = load_cards()
    paths = collect_paths(targets, exclude)
    per_file, unresolved = read_sources(paths, cards)

    pool = Counter()
    for counts in per_file.values():
        pool.update(counts)
    pool = dict(pool)
    if not pool:
        raise SystemExit("Every card was unresolved — nothing to analyse.")

    roles = load_card_roles()
    details = load_combo_details()
    # `bracket.load_reference()` would give the same three things, but it re-reads
    # cards.csv for a three-column slice of what `load_cards` already holds — a
    # second 24 MB scan per run for no new information.
    card_flags = {
        name: {"game_changer": bool(row["game_changer"]),
               "legal_commander": row["legal_commander"]}
        for name, row in cards.items()
    }
    obsolescence = (
        load_json_memo(OBSOLESCENCE_INDEX_PATH) if OBSOLESCENCE_INDEX_PATH.exists() else {}
    )

    # Which file supplies what: a card in three lists is owned once, and the
    # interesting number is what would be LOST by putting a box back on the shelf.
    owners = defaultdict(set)
    for filename, counts in per_file.items():
        for name in counts:
            owners[name].add(filename)
    sources = [
        {
            "file": filename,
            "distinct": len(counts),
            "copies": sum(counts.values()),
            "unique": sum(1 for n in counts if len(owners[n]) == 1),
        }
        for filename, counts in sorted(per_file.items())
    ]

    commanders = []
    for name in sorted(pool):
        if commander_rejection(cards[name]) is not None:
            continue
        identity = card_identity(cards[name])
        playable = in_identity(pool, cards, identity)
        colour_sources = sources_for(pool, cards, identity)
        rank = cards[name].get("edhrec_rank")
        commanders.append({
            "name": name,
            "color_identity": identity_key(identity),
            # NOT a commander rating. `cards.csv`'s edhrec_rank is the card's
            # popularity across the WHOLE format in every role, so a legend played
            # mostly IN other people's 99s outranks a genuine commander. Named
            # `edhrec_card_rank` because the short name invited exactly one reading
            # and it was the wrong one: it put Selvala, Heart of the Wilds (card
            # rank 430, commander rank #448) above Atraxa, Praetors' Voice
            # (commander rank #4) on a real shortlist. `card_search` and
            # `deck_audit` use the same column correctly, on cards-as-candidates.
            "edhrec_card_rank": None if pd.isna(rank) else int(rank),
            "depth": len(playable),
            "sources": colour_sources,
            # A colour with no sources is a colour you cannot cast. Reported per
            # commander because that is where the decision is made.
            "starved": sorted(c for c, n in colour_sources.items() if n < 2),
        })
    # Depth first, then card popularity as a TIEBREAK only — every commander in one
    # identity sees the same pool and so ties on depth, which makes this the
    # effective order within a colour. It is a stable, defensible tiebreak and it is
    # not a quality ranking; `build_notes` says so out loud.
    commanders.sort(key=lambda c: (-c["depth"], c["edhrec_card_rank"] or 10**9, c["name"]))

    # One entry per identity a commander in the box actually licenses — the real
    # menu, rather than a hardcoded list of guild names.
    identities = []
    for key in sorted({c["color_identity"] for c in commanders}):
        identity = set(key) if key != "C" else set()
        playable = in_identity(pool, cards, identity)
        lands = [n for n in playable if is_land(cards[n])]
        histogram, holes, unclassified = role_coverage(playable, cards, roles)
        identities.append({
            "key": key,
            "depth": len(playable),
            "copies": sum(pool[n] for n in playable),
            "lands": {"distinct": len(lands), "copies": sum(pool[n] for n in lands)},
            "sources": sources_for(pool, cards, identity),
            "commanders": [c["name"] for c in commanders if c["color_identity"] == key],
            "roles": histogram,
            "role_holes": holes,
            # The denominator a hole must be read against: cards the taxonomy
            # has no pattern for all land in `flex` and can inflate any hole.
            "role_unclassified": unclassified,
        })
    identities.sort(key=lambda i: -i["depth"])

    combos = contained_combos(pool, cards, details)
    by_identity = Counter(c["identity"] for c in combos)

    tags = Counter()
    for name in pool:
        for tag in parse_tag_set(cards[name].get("mechanical_tags")):
            tags[tag] += 1

    facts = {
        "meta": {
            "files": [str(p) for p in paths],
            "excluded": [str(e) for e in (exclude or [])],
            "distinct": len(pool),
            "copies": sum(pool.values()),
        },
        "sources": sources,
        "resolution": {
            "resolved": len(pool),
            "unresolved": [{"name": n, "files": f} for n, f in sorted(unresolved.items())],
        },
        "commanders": commanders,
        "identities": identities,
        "combos": {
            "total": len(combos),
            "by_identity": dict(sorted(by_identity.items())),
            "reported": min(len(combos), COMBO_REPORT_LIMIT),
            "lines": combos[:COMBO_REPORT_LIMIT],
        },
        # The whole box, so the floor is the floor of anything built from it.
        "bracket": assess(list(pool), card_flags, roles, details),
        "archetypes": [
            {"tag": tag, "cards": n, "share": round(n / len(pool), 3)}
            for tag, n in tags.most_common(ARCHETYPE_REPORT_LIMIT)
        ],
        "obsolescence": obsolete_in_pool(pool, obsolescence),
    }
    facts["notes"] = build_notes(facts)
    return facts


def build_notes(facts):
    """The traps, said out loud, so no agent has to rediscover them."""
    notes = []
    if facts.get("commanders"):
        notes.append(
            "`commanders[].edhrec_card_rank` is the CARD's popularity across the whole "
            "format in every role — NOT a rating of the card as a commander, and not a "
            "power ranking. A legend played mostly in other people's 99s outranks a "
            "genuine commander: Selvala, Heart of the Wilds is card rank 430 and "
            "commander rank #448, while Atraxa, Praetors' Voice is commander rank #4. "
            "The list is ordered by DEPTH, with card rank as a tiebreak only, because "
            "every commander in one identity sees the same pool and so ties on depth. "
            "Depth is what the box holds, not what the deck would be worth."
        )
    if facts["resolution"]["unresolved"]:
        notes.append(
            f"{len(facts['resolution']['unresolved'])} name(s) did not resolve against "
            f"cards.csv and are absent from every count below."
        )

    illusions = [
        i for i in facts["identities"]
        if any(n < 2 for n in i["sources"].values()) and i["depth"] > 100
    ]
    for i in illusions:
        starved = ", ".join(f"{c}={n}" for c, n in i["sources"].items() if n < 2)
        notes.append(
            f"{i['key']} shows {i['depth']} playable cards but is not castable: {starved}. "
            f"Depth counts what a commander LICENSES, not what the mana can pay for."
        )

    blind = [
        i for i in facts["identities"]
        if i.get("role_holes") and i.get("role_unclassified")
    ]
    if blind:
        worst = max(blind, key=lambda i: i["role_unclassified"])
        notes.append(
            f"Role holes are computed over a taxonomy with gaps: {worst['key']} has "
            f"{worst['role_unclassified']} card(s) with no card_roles.json entry, all of "
            f"which fall into flex. Absence of a role is not absence of the function — "
            f"Necropotence classifies as stax and Sylvan Library not at all, so a 'draw' "
            f"hole in particular is understated coverage before it is a real gap."
        )

    if facts["combos"]["total"] > facts["combos"]["reported"]:
        notes.append(
            f"{facts['combos']['total']} contained lines found; the "
            f"{facts['combos']['reported']} most popular are listed."
        )
    if facts["combos"]["total"]:
        notes.append(
            "Every combo line is UNVERIFIED. Commander Spellbook is format-agnostic "
            "and its bracket tags are not gospel — run `/resolve-stack` before "
            "presenting any of these as fact."
        )

    if facts.get("obsolescence"):
        notes.append(
            f"The {len(facts['obsolescence'])} in-box upgrade(s) are CANDIDATES. "
            f"obsolescence_index.json compares tags, mana value, colour and stats — it "
            f"does not read restriction clauses, so an ability gated on a creature type "
            f"looks like a superset of one that isn't. Read the two oracle texts before "
            f"acting on any of them."
        )

    floor = facts["bracket"].get("floor")
    if floor:
        notes.append(
            f"Bracket floor {floor} for the whole box — any deck built from it inherits "
            f"only the floor of the cards actually chosen, not this one."
        )
    return notes


# ── Report ──────────────────────────────────────────────────────────────


def format_report(facts):
    meta = facts["meta"]
    out = [
        f"POOL  {meta['distinct']} distinct / {meta['copies']} copies "
        f"from {len(meta['files'])} list(s)",
        "",
        "  sources:",
    ]
    for s in facts["sources"]:
        out.append(f"    {s['file']:28s} {s['distinct']:4d} distinct  "
                   f"{s['copies']:4d} copies  {s['unique']:4d} found nowhere else")

    out += ["", f"  identities ({len(facts['identities'])} licensed by a commander in the box):"]
    for i in facts["identities"]:
        src = " ".join(f"{c}={n}" for c, n in i["sources"].items()) or "colourless"
        holes = ", ".join(f"{g} -{n}" for g, n in i["role_holes"].items()) or "none"
        out.append(f"    {i['key']:6s} depth={i['depth']:4d}  lands={i['lands']['copies']:3d}  "
                   f"sources: {src}")
        out.append(f"           {len(i['commanders'])} commander(s); role holes: {holes}"
                   f"  ({i['role_unclassified']} unclassified)")

    out += ["", f"  deepest commanders (of {len(facts['commanders'])} legal in the box) — "
                f"ranked by DEPTH, not by quality:"]
    for c in facts["commanders"][:15]:
        rank = c["edhrec_card_rank"] or "—"
        warn = f"  ⚠ starved: {','.join(c['starved'])}" if c["starved"] else ""
        out.append(f"    {c['color_identity']:6s} depth={c['depth']:4d}  card-rank={str(rank):>6s}  "
                   f"{c['name']}{warn}")

    out += ["", f"  combo lines contained ({facts['combos']['total']} total, all UNVERIFIED):"]
    for c in facts["combos"]["lines"][:12]:
        tag = "BAN" if c["banned"] else f"b{c['bracket']}"
        out.append(f"    {tag:>4s} {c['identity']:5s} pop={c['popularity']:>7d}  "
                   f"{' + '.join(c['cards'])}")
        out.append(f"         -> {'; '.join(c['produces'][:3])}")

    if facts["obsolescence"]:
        out += ["", f"  in-box upgrades ({len(facts['obsolescence'])}):"]
        for o in facts["obsolescence"][:10]:
            best = o["replaced_by"][0]
            out.append(f"    {o['card']} -> {best['name']} ({', '.join(best['advantages'])})")

    out += ["", f"  archetype signal (top {len(facts['archetypes'])} mechanical tags):"]
    out.append("    " + ", ".join(f"{a['tag']} {a['cards']}" for a in facts["archetypes"]))

    out += ["", f"  bracket floor: {facts['bracket'].get('floor')}"]
    for note in facts["notes"]:
        out.append(f"  note: {note}")
    return "\n".join(out)


def main(args):
    facts = analyze(args.targets, getattr(args, "exclude", None))
    if getattr(args, "as_json", False):
        print(json.dumps(facts, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(format_report(facts))
    out = getattr(args, "out", None)
    if out:
        # A pool has no slug, so `resolve_out_path`'s slug contract doesn't apply
        # here — but a directory argument must still auto-name rather than crash
        # (the other --out commands accept a directory, and callers expect parity).
        if Path(out).is_dir():
            out = str(Path(out) / "pool-facts.json")
        with open(out, "w") as f:
            json.dump(facts, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {out}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot pool-facts <path>...`.")
