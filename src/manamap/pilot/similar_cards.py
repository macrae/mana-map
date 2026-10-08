"""Pilot: cards that DO what a named card does — nearest neighbours in the ability space.

`card-search` answers "which cards say X"; it needs the question phrased as a regex,
and a pilot holding a card they like does not have a regex, they have the card. This
is the other half: name one card (or several), get the cards whose FUNCTION sits
nearest it in `embeddings_ability.npy`, filtered by the same three rules card-search
enforces so callers cannot get them wrong:

  * **Colour identity is DERIVED** — `--deck <slug>` takes it from the commander;
    `--identity` is the hand-stated form for exploring without a deck.
  * **A candidate is a card you do NOT already have** — `--deck` excludes the 99.
  * **Legality is a filter, not a note** — Commander-illegal cards never rank.

Several seeds rank against their CENTROID (`commander_search.centroid`), so
"cards like Windfall and Wheel of Fortune" asks for the shared function, not
whichever seed happens to be closer.

**The ability space, not the layout space.** `embeddings_ability.npy` is FUNCTION and
is the sole source of similarity in this repo (CLAUDE.md, "Two embedding spaces");
`embeddings.npy` would return cards of the same colour and type.

**Proximity is a discovery aid, not a verdict.** The score is a cosine between
learned vectors, and cards with similar phrasing can do different jobs. It prints
the score, the roles and what the goldfish can price so the reader can judge; it
never says a card is good for a deck. The one scorer stays `build_deck`.

Index alignment is the contract: row i of the embeddings is row i of `cards.csv`
(`card_pool.load_frame`). A length mismatch is a refusal, never a silent offset.
"""

import json
import urllib.parse

import numpy as np

from manamap import config
from manamap.pilot.card_pool import corpus_oracle, load_frame, load_pool
from manamap.pilot.card_search import (
    channels_for,
    commander_identity,
    deck_names,
    parse_identity_arg,
)
from manamap.pilot.common import deck_dir, load_card_roles, mtime_memo

DEFAULT_LIMIT = 15
MAX_LIMIT = 100

#: The deployed Atlas; the same `?cards=` form the Deck Context links with, because
#: `?card=` falls back to a random card when a name does not resolve.
ATLAS_HREF = "https://manamap.seanmacrae.com/viz/index.html?cards="


def _load_embeddings():
    return np.load(config.ABILITY_EMBEDDINGS_PATH)


def ability_embeddings():
    """The ability space, parsed once per process (and again only if it changes)."""
    if not config.ABILITY_EMBEDDINGS_PATH.exists():
        raise SystemExit(
            "embeddings_ability.npy is absent — similar-cards reads the ability space. "
            "Run the pipeline (`manamap run --from embed`) first.")
    return mtime_memo(config.ABILITY_EMBEDDINGS_PATH, "corpus:ability", _load_embeddings)


def _index(names):
    """Lower-cased name AND each DFC face -> first corpus row. First printing wins,
    as everywhere (`cards.csv` carries 76 repeated names)."""
    out = {}
    for i, n in enumerate(names):
        out.setdefault(n.lower(), i)
        if " // " in n:
            for face in n.split(" // "):
                out.setdefault(face.strip().lower(), i)
    return out


def resolve_seeds(seeds, names):
    """Seed names -> corpus rows, or a refusal naming what did not resolve and the
    closest real names. A typo must not quietly drop a seed and shift the centroid."""
    import difflib

    index = _index(names)
    rows, missing = [], []
    for s in seeds:
        key = s.strip().lower()
        row = index.get(key)
        if row is None:
            # "Brallin" for "Brallin, Skyshark Rider": a prefix is accepted only when
            # it names exactly ONE card, so it can never pick between two.
            prefixed = {r for k, r in index.items() if k.startswith(key)}
            row = prefixed.pop() if len(prefixed) == 1 else None
        if row is None:
            missing.append(s)
        elif row not in rows:
            rows.append(row)
    if missing:
        hints = []
        for m in missing:
            prefixed = sorted({names[r] for k, r in index.items() if k.startswith(m.strip().lower())})
            if prefixed:
                more = f" (+{len(prefixed) - 5} more)" if len(prefixed) > 5 else ""
                hints.append(f"  {m!r} — names {len(prefixed)} cards: "
                             f"{', '.join(prefixed[:5])}{more}")
                continue
            close = difflib.get_close_matches(m.lower(), list(index), n=3, cutoff=0.75)
            hints.append(f"  {m!r}" + (f" — did you mean: {', '.join(names[index[c]] for c in close)}?"
                                       if close else " — no close match in the corpus"))
        raise SystemExit("could not resolve:\n" + "\n".join(hints))
    return rows


def similar(seeds, identity=None, exclude=(), limit=DEFAULT_LIMIT,
            allow_game_changers=True):
    """Rank the corpus by ability-space proximity to the seeds. Returns (rows, meta)."""
    from manamap.analysis.common import top_k_similar
    from manamap.analysis.commander_search import centroid

    frame = load_frame()
    pool = load_pool()
    if frame is None or pool is None:
        raise SystemExit("cards.csv is absent — similar-cards reads the corpus. "
                         "Run `manamap extract` first.")
    emb = ability_embeddings()
    names = frame["name"].tolist()
    if len(emb) != len(names):
        raise SystemExit(
            f"index alignment broken: embeddings_ability.npy has {len(emb)} rows and "
            f"cards.csv {len(names)} — re-run the pipeline from the step that changed.")

    seed_rows = resolve_seeds(seeds, names)
    exclude = set(exclude or [])
    ident = set(identity or [])

    # One eligible row per name (first printing), legal, in identity, not excluded,
    # not a seed. Built as a mask so the ranking is one matrix product.
    mask = np.zeros(len(names), dtype=bool)
    seen, skipped_illegal = set(), 0
    seed_names = {names[r] for r in seed_rows}
    for i, n in enumerate(names):
        if n in seen:
            continue
        seen.add(n)
        rec = pool.get(n)
        if rec is None or n in exclude or n in seed_names:
            continue
        if not rec["legal"]:
            skipped_illegal += 1
            continue
        if identity is not None and not rec["color_identity"] <= ident:
            continue
        if not allow_game_changers and rec["game_changer"]:
            continue
        mask[i] = True

    limit = max(1, min(int(limit), MAX_LIMIT))
    if len(seed_rows) == 1:
        ranked = top_k_similar(emb, seed_rows[0], limit, mask=mask)
    else:
        # A centroid is not a corpus row, so rank it the way top_k_similar would:
        # one dot product, then the top slice of the eligible rows.
        c = centroid(emb, seed_rows)
        scores = emb @ c
        eligible = np.flatnonzero(mask)
        k = min(limit, len(eligible))
        top = eligible[np.argpartition(scores[eligible], -k)[-k:]] if k else eligible
        ranked = [(int(i), float(scores[i])) for i in top[np.argsort(-scores[top])]]

    oracle = corpus_oracle()
    try:
        roles_map = load_card_roles()
    except FileNotFoundError:
        roles_map = {}
    seed_roles = set()
    for r in seed_rows:
        seed_roles |= set(roles_map.get(names[r]) or [])

    rows = []
    for i, score in ranked:
        n = names[i]
        rec = pool[n]
        text = oracle.get(n, "")
        roles = sorted(roles_map.get(n) or [])
        rows.append({
            "name": n,
            "score": round(score, 4),
            "mana_cost": rec["mana_cost"],
            "cmc": rec["cmc"],
            "type_line": rec["type_line"],
            "color_identity": sorted(rec["color_identity"]),
            "edhrec_rank": rec["edhrec_rank"],
            "game_changer": rec["game_changer"],
            "roles": roles,
            # The reason a reader can check: which of the seed's jobs it shares.
            "shared_roles": sorted(seed_roles & set(roles)),
            "channels": channels_for({"name": n, "type_line": rec["type_line"],
                                      "oracle_text": text}),
            "oracle_text": text,
            "atlas": ATLAS_HREF + urllib.parse.quote_plus(n),
        })
    meta = {"eligible": int(mask.sum()), "returned": len(rows),
            "commander_illegal_skipped": skipped_illegal,
            "space": "ability (embeddings_ability.npy)",
            "seeds": [names[r] for r in seed_rows],
            "seed_roles": sorted(seed_roles),
            "ranked_against": "the seed" if len(seed_rows) == 1 else "the seeds' centroid"}
    return rows, meta


def analyze(args):
    identity, exclude, derived_from = None, set(), None
    if getattr(args, "deck", None):
        deck_dir(args.deck)
        identity = commander_identity(args.deck)
        derived_from = args.deck
        if not getattr(args, "include_deck", False):
            exclude = deck_names(args.deck)
    if getattr(args, "identity", None) is not None:
        if derived_from:
            raise SystemExit(
                "--identity and --deck both given: a deck's identity is DERIVED from its "
                "commander and may not be overridden. Drop one.")
        identity = parse_identity_arg(args.identity)
    if not args.cards:
        raise SystemExit("name at least one card: `similar-cards \"Windfall\" --deck <slug>`")
    rows, meta = similar(
        args.cards, identity=identity, exclude=exclude,
        limit=getattr(args, "limit", None) or DEFAULT_LIMIT,
        allow_game_changers=not getattr(args, "no_game_changers", False))
    return {
        "identity": sorted(identity) if identity is not None else None,
        "identity_derived_from": derived_from,
        "excluded_deck_cards": len(exclude),
        "meta": meta,
        "results": rows,
    }


def format_report(doc):
    m, out = doc["meta"], []
    ident = "".join(doc["identity"]) if doc["identity"] is not None else "any"
    src = f" (derived from {doc['identity_derived_from']})" if doc["identity_derived_from"] else ""
    out.append(f"SIMILAR CARDS — like {' + '.join(m['seeds'])} · identity {ident}{src}")
    out.append(f"  ability space, ranked against {m['ranked_against']}; "
               f"{m['eligible']} eligible, showing {m['returned']}")
    if doc["excluded_deck_cards"]:
        out.append(f"  excluding {doc['excluded_deck_cards']} name(s) already in the deck")
    out.append("  proximity is a discovery aid, not a verdict — read the text")
    out.append("")
    for n, r in enumerate(doc["results"], 1):
        gc = " ★GC" if r["game_changer"] else ""
        out.append(f"{n:>3}. {r['score']:.3f}  {r['mana_cost'] or '—':<12} {r['name']}{gc}")
        rank = r["edhrec_rank"]
        out.append(f"       {r['type_line']}  ·  edhrec {rank if rank is not None else 'unranked'}"
                   + (f"  ·  shares {', '.join(r['shared_roles'])}" if r["shared_roles"]
                      else (f"  ·  roles {', '.join(r['roles'])}" if r["roles"] else "")))
        out.append(f"       model: {', '.join(r['channels']) if r['channels'] else '—'}")
        text = " | ".join(str(r["oracle_text"]).splitlines())
        out.append(f"       {text[:240]}")
        out.append(f"       {r['atlas']}")
        out.append("")
    return "\n".join(out)


def main(args):
    doc = analyze(args)
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(format_report(doc))
