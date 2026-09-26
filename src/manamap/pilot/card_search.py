"""Pilot: deterministic card mining over the corpus.

The bench could measure a deck from nine directions and could not answer the
question every one of those measurements ends in — *which cards would fix it*.
`deck-audit` names an under-filled axis, `goldfish` prices a thin component,
`prescribe` asks the doctor for adds; all three then needed a human or an agent
to think of candidates, and an agent asked to think of candidates invents them.
This is the deterministic half: a filter over `cards.csv`, no model, no ranking
opinion beyond EDHREC's, so an agent proposing a card can be handed a list it
did not author and a validator can check membership.

Three rules it enforces so callers cannot get them wrong:

  * **Colour identity is DERIVED, never authored** — `--deck <slug>` takes it
    from that deck's commander, the same rule `build_deck.load_brief` follows.
    `--identity` exists for exploring without a deck and is the only way to
    state one by hand.
  * **A candidate is a card you do NOT already have.** `--deck` excludes the
    deck's own 99 by default, because a search that returns your own list back
    is the commonest way this kind of tool wastes a reader's time.
  * **Legality is a filter, not a note.** Commander-illegal cards are dropped
    before ranking rather than flagged afterwards.

It deliberately does NOT score fit. The repo has exactly one scorer
(`build_deck`) and one retrieval aid (the synergy graph, which its own docstring
says "is a retrieval aid and not a scoring function"); a second opinion here
would be a third answer to "is this card good" that nothing reconciles.
"""

import json
import re
from pathlib import Path

from manamap.pilot import collection
from manamap.pilot.card_pool import corpus_oracle, load_pool
from manamap.pilot.common import (
    deck_dir,
    expand_faces,
    load_card_roles,
    load_deck_cards,
)

# A card with no EDHREC rank has never been played enough to have one. It sorts
# last rather than being dropped: the corpus carries plenty of unranked cards
# that are simply new, and a brand-new set's answer to a problem is exactly the
# kind of thing a search like this should be able to surface.
UNRANKED = 10 ** 9

MAX_RESULTS = 50


#: THE GOLDFISH CHANNELS A CARD FEEDS, as a searchable filter.
#:
#: THIS EXISTS BECAUSE OF A MEASURED FAILURE. Mining goblin-storm for cards that
#: would convert its card advantage into damage returned Guttersnipe, Firebrand
#: Archer and Kessig Flamebreather; all three were staged, and all three read as
#: VANILLA BODIES — the model had no spell count, so "whenever you cast" fired on
#: nothing. The branch measured WORSE and the decline was an artifact of the
#: instrument, not a verdict on the cards. A search that cannot say "the model
#: cannot price this" hands you that trap every time.
#:
#: Each entry is (label, predicate) over a card dict. `channels_for` annotates
#: every result; `--channel` filters on them.
def _channel_table():
    from manamap.pilot import goldfish_profiles as gp

    def _fodder(c):
        return gp.copy_fodder(c)

    def _pump(c):
        return gp.spell_pump(c) != (0, 0)

    def _draw(c):
        d = gp.draw_profile(c)
        return bool(d["spell_draw"] or d["etb_draw"] or d["recurring_draw"]
                    or d["arrival_draw"] or d["cast_draw"])

    def _storm(c):
        return gp.spell_count_profile(c)["storm"]

    def _percast(c):
        return bool(gp.spell_count_profile(c)["per_cast_damage"])

    def _magecraft(c):
        return gp.spell_count_profile(c)["magecraft"]

    return {
        "fodder": ("copied by a Zada-style ability (targets ONE creature)", _fodder),
        "pump": ("a one-turn pump from a spell", _pump),
        "draw": ("draws cards through a channel the model reads", _draw),
        "storm": ("has storm — copies scale with the spell count", _storm),
        "per-cast-damage": ("damage to each opponent per CAST (never a copy)", _percast),
        "magecraft": ("fires on cast OR COPY — Zada multiplies it", _magecraft),
    }


CHANNELS = None          # built lazily; importing profiles is not free


def channels_for(card):
    """Every goldfish channel this card feeds. Empty means the model reads it as
    a body and a mana cost and nothing else."""
    global CHANNELS
    if CHANNELS is None:
        CHANNELS = _channel_table()
    return sorted(k for k, (_desc, pred) in CHANNELS.items() if pred(card))


def parse_identity_arg(value):
    """`--identity` as a set of single-letter colours, from either spelling.

    `analysis.common.parse_color_identity` splits on commas because that is how
    `cards.csv` stores the column ("G, U"). Handed the COMPACT form a human types —
    `GU` — it returns `{"GU"}`, a single two-character token, and `{"U"} <= {"GU"}`
    is False for every coloured card. The filter then silently returned only
    COLOURLESS cards while reporting "identity GU": `--identity GU --oracle
    "additional combat phase"` found Genji Glove (colourless) and dropped
    Illusionist's Gambit (mono-blue). Same shape as the bug recorded in
    `card_pool._build_pool` — an identity set that can never be a superset, failing
    quietly rather than loudly.

    Accepts `GU`, `gu`, `G,U`, `G, U`. Anything that is not a WUBRGC letter is an
    error rather than a silent drop: a typo must not narrow a search invisibly.
    """
    raw = str(value or "").upper()
    letters = [c for c in raw if not c.isspace() and c != ","]
    bad = sorted({c for c in letters if c not in "WUBRGC"})
    if bad:
        raise SystemExit(f"--identity: {', '.join(bad)} is not a colour. "
                         f"Use WUBRG letters, e.g. GU or 'G, U'.")
    # C means colourless, which as an IDENTITY is the empty set, not a sixth colour.
    return {c for c in letters if c != "C"}


def commander_identity(slug):
    """The colour identity of a deck, derived from its commander(s).

    Never read from a brief or a config: `build_deck` derives it the same way,
    and two derivations of one fact are two chances to disagree.
    """
    # load_deck_cards returns the whole cards.json document, not the list.
    cards = load_deck_cards(slug).get("cards") or []
    ident = set()
    for c in cards:
        if c.get("is_commander"):
            ident |= set(c.get("color_identity") or [])
    return ident


def deck_names(slug):
    """Every name in a deck, including both faces, for exclusion."""
    out = set()
    for c in load_deck_cards(slug).get("cards") or []:
        out |= expand_faces(c["name"])
    return out


def search(identity=None, oracle=None, names=None, types=None, roles=None, cmc_max=None,
           cmc_min=None, exclude=(), limit=MAX_RESULTS, allow_game_changers=True,
           require_all=False, owned=None, channels=None, unmodelled=None):
    """Filter the corpus. Returns (rows, meta) — rows already ranked and capped.

    `oracle` is a list of regexes: a card matches when ANY of them hits, or ALL
    when `require_all`. ANY is the default because the question that brings
    someone here is usually "how do I do X" with several phrasings of X, and a
    card that says "additional combat phase" and one that says "untap all
    creatures" are alternative answers, not a conjunction.
    """
    pool = load_pool()
    if pool is None:
        raise SystemExit(
            "cards.csv is absent — card-search reads the corpus. Run `manamap extract` "
            "first (a fresh clone can render and validate, but cannot mine cards).")
    oracle_text = corpus_oracle()
    try:
        roles_map = load_card_roles()
    except FileNotFoundError:
        roles_map = {}

    ident = set(identity or [])
    pats = [re.compile(p, re.IGNORECASE) for p in (oracle or [])]
    # Name search is separate from oracle search on purpose: `--oracle "Sol Ring"`
    # looks like it should find Sol Ring and finds every card whose RULES TEXT says
    # "Sol Ring" instead, which is a confusing empty result rather than an error.
    name_pats = [re.compile(p, re.IGNORECASE) for p in (names or [])]
    type_pats = [re.compile(t, re.IGNORECASE) for t in (types or [])]
    want_roles = set(roles or [])
    exclude = set(exclude or [])
    # `pool-facts` knew the box but could not filter by oracle text; this could
    # filter by oracle text but could not see the box. Every "what could I add that I
    # already have" question needed both, so it was answered by hand every time.
    # Boxes only — deck membership is not ownership; see `pilot.collection`.
    have = collection.owned_names() if owned is not None else set()

    rows, skipped_illegal = [], 0
    for name, rec in pool.items():
        if name in exclude:
            continue
        if not rec["legal"]:
            skipped_illegal += 1
            continue
        if identity is not None and not rec["color_identity"] <= ident:
            continue
        if not allow_game_changers and rec["game_changer"]:
            continue
        cmc = rec["cmc"]
        if cmc_max is not None and cmc > cmc_max:
            continue
        if cmc_min is not None and cmc < cmc_min:
            continue
        if name_pats and not any(p.search(name) for p in name_pats):
            continue
        if type_pats and not any(p.search(rec["type_line"]) for p in type_pats):
            continue
        text = oracle_text.get(name, "")
        if pats:
            hits = [p.pattern for p in pats if p.search(text)]
            if (len(hits) != len(pats)) if require_all else (not hits):
                continue
        else:
            hits = []
        card_roles = roles_map.get(name) or []
        if want_roles and not (want_roles & set(card_roles)):
            continue
        is_owned = bool(expand_faces(name) & have) if owned is not None else None
        if owned is not None and is_owned is not owned:
            continue
        # WHAT THE MODEL CAN PRICE. Annotated on every row, never only filtered:
        # an empty list is the answer to "why did this measure as nothing".
        card_for_profile = {"name": name, "type_line": rec["type_line"],
                            "oracle_text": text}
        chans = channels_for(card_for_profile)
        if channels and not (set(channels) & set(chans)):
            continue
        if unmodelled is True and chans:
            continue
        if unmodelled is False and not chans:
            continue
        rows.append({
            "name": name,
            "mana_cost": rec["mana_cost"],
            "cmc": cmc,
            "type_line": rec["type_line"],
            "color_identity": sorted(rec["color_identity"]),
            "edhrec_rank": rec["edhrec_rank"],
            "game_changer": rec["game_changer"],
            "roles": sorted(card_roles),
            "owned": is_owned,
            "matched": hits,
            "channels": chans,
            "oracle_text": text,
        })

    rows.sort(key=lambda r: (r["edhrec_rank"] if r["edhrec_rank"] is not None else UNRANKED,
                             r["name"]))
    meta = {"matched": len(rows), "returned": min(len(rows), limit),
            "commander_illegal_skipped": skipped_illegal}
    if owned is not None:
        meta["ownership_filter"] = "owned" if owned else "unowned"
        meta["collection"] = collection.summary()["distinct_in_boxes"]
    if len(rows) > limit:
        # Say what was dropped. A silently truncated list reads as "that is all
        # of them", which is the claim this tool must never make by accident.
        meta["truncated"] = len(rows) - limit
    return rows[:limit], meta


def analyze(args):
    identity = None
    exclude = set()
    derived_from = None
    if getattr(args, "deck", None):
        deck_dir(args.deck)                       # fail early with the good message
        identity = commander_identity(args.deck)
        derived_from = args.deck
        if not getattr(args, "include_owned", False):
            exclude = deck_names(args.deck)
    if getattr(args, "identity", None) is not None:
        if derived_from:
            raise SystemExit(
                "--identity and --deck both given: a deck's identity is DERIVED from its "
                "commander and may not be overridden. Drop one.")
        identity = parse_identity_arg(args.identity)
    rows, meta = search(
        identity=identity,
        oracle=getattr(args, "oracle", None) or [],
        names=getattr(args, "name", None) or [],
        types=getattr(args, "type", None) or [],
        roles=getattr(args, "role", None) or [],
        channels=list(getattr(args, "channel", None) or []) or None,
        unmodelled=getattr(args, "unmodelled", None),
        cmc_max=getattr(args, "cmc_max", None),
        cmc_min=getattr(args, "cmc_min", None),
        exclude=exclude,
        limit=getattr(args, "limit", None) or MAX_RESULTS,
        allow_game_changers=not getattr(args, "no_game_changers", False),
        require_all=getattr(args, "require_all", False),
        owned=(True if getattr(args, "owned", False)
               else False if getattr(args, "unowned", False) else None),
    )
    return {
        "identity": sorted(identity) if identity is not None else None,
        "identity_derived_from": derived_from,
        "excluded_deck_cards": len(exclude),
        "query": {"oracle": getattr(args, "oracle", None) or [],
                  "name": getattr(args, "name", None) or [],
                  "type": getattr(args, "type", None) or [],
                  "role": getattr(args, "role", None) or [],
                  "cmc_min": getattr(args, "cmc_min", None),
                  "cmc_max": getattr(args, "cmc_max", None),
                  "require_all": bool(getattr(args, "require_all", False))},
        "meta": meta,
        "results": rows,
    }


def format_report(doc):
    m, out = doc["meta"], []
    ident = "".join(doc["identity"]) if doc["identity"] else "any"
    src = f" (derived from {doc['identity_derived_from']})" if doc["identity_derived_from"] else ""
    out.append(f"CARD SEARCH — identity {ident}{src} · {m['matched']} match(es), "
               f"showing {m['returned']}")
    if doc["excluded_deck_cards"]:
        out.append(f"  excluding {doc['excluded_deck_cards']} name(s) already in the deck")
    if m.get("truncated"):
        out.append(f"  {m['truncated']} further match(es) not shown — raise --limit")
    out.append("")
    for r in doc["results"]:
        rank = r["edhrec_rank"]
        gc = " ★GC" if r["game_changer"] else ""
        own = "" if r.get("owned") is None else ("  ✓owned" if r["owned"] else "  ·buy")
        out.append(f"  {str(r['mana_cost']) or '—':<12} {r['name']}{gc}{own}")
        out.append(f"    {r['type_line']}  ·  edhrec "
                   f"{rank if rank is not None else 'unranked'}"
                   + (f"  ·  roles {', '.join(r['roles'])}" if r["roles"] else ""))
        # THE LINE THAT WOULD HAVE SAVED A WHOLE BRANCH. "model: —" means the
        # goldfish reads this as a body and a mana cost: measure it and it will
        # come back as nothing, which is not the same as not helping.
        out.append(f"    model: {', '.join(r['channels']) if r.get('channels') else '—'}")
        text = " | ".join(str(r["oracle_text"]).splitlines())
        out.append(f"    {text[:240]}")
        out.append("")
    return "\n".join(out)


def main(args):
    doc = analyze(args)
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(format_report(doc))
    out = getattr(args, "out", None)
    if out:
        # A search is not per-deck data even when --deck scoped it (the results are
        # corpus rows, not this deck's numbers), so `resolve_out_path`'s slug rule
        # does not bind. A directory still auto-names, for parity with the others.
        if Path(out).is_dir():
            stem = f"card-search-{args.deck}.json" if getattr(args, "deck", None) \
                else "card-search.json"
            out = str(Path(out) / stem)
        with open(out, "w") as f:
            json.dump(doc, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {out}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot card-search ...`.")
