"""Pilot: what in this list has a cheaper card doing its job, and is your pile it.

    manamap pilot upgrades <slug> [--branch N] [--pool <file|library|->]

IT PROPOSES. IT DOES NOT SCORE AND IT DOES NOT MEASURE. Ranking is by the
obsolescence index's own published `strength` — not a new weighting. Four modules
(`candidates`, `close`, `card_search`, `assess`) each record that the last second
scorer in this repo was deleted for disagreeing with the first; this is not a
seventh. `candidates` is what measures a swap, by substituting it and re-running
the diagnostic, and every row here names that as the next step.

TWO DIRECTIONS OVER ONE INDEX, because they are different questions:

  INWARD    for each card in the 99, what does its job for less. This is the
            efficiency question, and the deck is the anchor.
  THE PILE  which of those candidates are cards you already picked up. A pile
            card the index cannot compare to anything in the list produces no
            row at all — SAID OUT LOUD rather than silently dropped, because
            "the index has nothing to say about this" is a real answer and an
            empty report is not.

WHAT THE STRENGTH CANNOT SEE, AND WHAT CATCHES IT. `strength` prices how strongly
B outclasses A *given* that they do the same job. It cannot tell you they do not:
Leaden Myr (a mana rock) against H.E.R.B.I.E. (a flying creature) scores 0.405 at
similarity 0.944, and that is a retrieval failure no comparison score repairs.

TWO FLAGS, BOTH MEASURED OVER ALL 20,827 PAIRS BEFORE BEING WRITTEN, because a
check that fires on correct data is worse than no check:

  `roles_disjoint`      both cards are role-classified and share NO role head.
                        1,202 pairs, and 5.68% of the 10,926 at strength >= 0.6:
                        Phyrexian Broodstar (ramp) -> Aerial Doombot (buff) at
                        0.83, Soul of the Rapids (protection) -> Eon Frolicker
                        (value) at 0.75. `threat:body` is excluded from the
                        comparison for the reason `eval_obsolescence` excludes
                        it — "is a creature" is not a job, and counting it drops
                        the catch rate to 2.58% by making any two creatures look
                        related.

  `newly_combat_gated`  the replacement needs a creature to connect and the card
                        it replaces does not. 7.62% of strong pairs: Vedalken
                        Heretic (draws on a cast) -> Flitterwing Nuisance
                        (combat) at 0.75. This is `assess`'s own strongest
                        verdict — efficient in a vacuum, wrong axis.

WHAT WAS TRIED AND REJECTED, both on the same ground. `assess.job_of` collapses a
role LIST to one label by first match over an alphabetically sorted list, so
Atarka, World Render read as `wincon` and Super-Adaptoid as `protection` while
the two actually share `wincon:combat` — and Skyshroud Claim (`tutor`) against
Three Visits (`ramp`) is a perfectly good ramp swap. Both were false positives on
the first cut of this file. And flagging ANY change of gate fires on 38.1% of all
pairs, which is one row in three.

Neither flag GATES anything. A disjoint role pair means the SEARCH failed, which
a pilot sees at a glance and no threshold can decide; and 14.3% of pairs have one
side with no classified role at all, where the question is simply unanswerable
and is reported as such rather than as agreement.
"""

import json

from manamap.pilot import assess as _assess
from manamap.pilot.common import deck_file, load_deck_cards, load_json

#: Below this, `pool_facts` already tells a pilot the claim is weak. Same line,
#: said once, so the two surfaces cannot drift.
DEFAULT_MIN_STRENGTH = 0.4

#: Reported, never silent. A truncated list reads as "these are all of them".
DEFAULT_LIMIT = 40

#: "Is a creature" is not a job. `eval_obsolescence` excludes it from its own
#: role-agreement figure for the same reason: counting it makes any two creatures
#: look related and drops this file's catch rate from 5.68% to 2.58%.
GENERIC_ROLE = "threat:body"

#: The index's own ceiling. base 0.45 + 0.30 cheaper + 0.15 + 0.10, capped —
#: there is no 1.0 in the data and a pilot waiting for one waits forever.
MAX_STRENGTH_IN_DATA = 0.95


def _index():
    from manamap.config import OBSOLESCENCE_INDEX_PATH
    if not OBSOLESCENCE_INDEX_PATH.exists():
        raise SystemExit(
            f"{OBSOLESCENCE_INDEX_PATH} not found — run `manamap power-creep` "
            f"(step 11) first. It is what this command reads.")
    return json.loads(OBSOLESCENCE_INDEX_PATH.read_text(encoding="utf-8"))


def _entries(record):
    """The comparisons on one anchor. Reads the pre-2026-08 key as a fallback.

    A reader that silently answers zero against an old artifact is worse than
    one that fails, and worse still than one that just reads both shapes.
    """
    if not isinstance(record, dict):
        return []
    return record.get("compare_with") or record.get("obsoleted_by") or []


def role_heads(roles_for_card):
    """The KINDS of job a card does — `ramp:dork` and `ramp:land` are one head.

    Heads, not full roles, because the index pairs cards that do the same job and
    two flavours of ramp are the same job. And a SET, not `assess.job_of`'s
    single label: that collapses the list by first match over an alphabetically
    sorted array, which read Super-Adaptoid as `protection` while it shares
    `wincon:combat` with the card it was set against.
    """
    return {r.split(":", 1)[0] for r in (roles_for_card or []) if r != GENERIC_ROLE}


def propose(slug, branch=None, pool=None, min_strength=DEFAULT_MIN_STRENGTH,
            limit=DEFAULT_LIMIT, owned_only=False):
    from manamap.pilot import card_pool, candidates as _cand, collection, deck_branch
    from manamap.pilot.common import load_card_roles

    index = _index()
    corpus = card_pool.load_pool()
    if corpus is None:
        raise SystemExit("no card corpus — run the pipeline through `extract`.")
    oracle = card_pool.corpus_oracle()
    roles = load_card_roles()

    doc = load_deck_cards(slug, branch)
    held = {c["name"] for c in doc["cards"]}
    identity = {c for x in doc["cards"] for c in (x.get("color_identity") or [])}
    by_name = {c["name"]: c for c in doc["cards"]}

    declared = _cand._declared_cards(slug, branch)
    targets = (load_json(deck_file(slug, "goldfish_targets.json", branch))
               or {}).get("targets") or []
    axes = {t["label"]: {c for g in (t.get("need") or []) for c in (g.get("any_of") or [])}
            for t in targets}

    owned = collection.owned_names()
    # WHERE WOULD EACH CARD COME FROM — the four states, and `buy` is the one
    # that costs money. `deck_branch.source` is the only answer to this in the
    # repo and it walks the whole list, so a name it does not know is simply
    # absent rather than guessed at.
    try:
        src = {r["name"]: r.get("state")
               for r in deck_branch.source(slug, branch)["cards"]}
    except Exception:
        src = {}

    pool_set = set(pool or [])
    rows = []
    for out_name, card in by_name.items():
        for entry in _entries(index.get(out_name)):
            in_name = entry.get("name")
            if not in_name or in_name in held:
                continue
            strength = float(entry.get("strength") or 0.0)
            if strength < min_strength:
                continue
            info = corpus.get(in_name)
            if info is None:
                continue                       # not in this corpus; say nothing
            if not (info.get("color_identity") or set()) <= identity:
                continue
            is_owned = in_name in owned
            if owned_only and not is_owned:
                continue

            out_text, in_text = oracle.get(out_name, ""), oracle.get(in_name, "")
            heads_out, heads_in = role_heads(roles.get(out_name)), role_heads(roles.get(in_name))
            gate_out, gate_in = _assess.gate_of(out_text), _assess.gate_of(in_text)
            rows.append({
                "out": out_name,
                "in": in_name,
                "strength": strength,
                # BOTH SIDES, ALWAYS. The pre-repair schema carried `advantages`
                # alone, so a card that CHARGED you something read as pure
                # upside. Anything that renders one of these must render all.
                "gains": entry.get("gains") or entry.get("advantages") or [],
                "costs": entry.get("costs") or [],
                "narrows": entry.get("narrows") or [],
                "also_differs": entry.get("also_differs") or [],
                "similarity": entry.get("similarity"),
                "edhrec_rank": entry.get("edhrec_rank"),
                "played_more": entry.get("played_more"),
                "mv_out": int(float(card.get("cmc") or 0)),
                "mv_in": int(float(info.get("cmc") or 0)),
                "source": src.get(in_name) or ("box" if is_owned else None),
                "owned": is_owned,
                "in_pool": in_name in pool_set,
                "roles_out": sorted(heads_out),
                "roles_in": sorted(heads_in),
                # THE SEARCH FAILED — not the comparison. Only when BOTH sides
                # are classified: 14.3% of pairs have one side with no role but
                # `threat:body`, and there the question is unanswerable rather
                # than answered yes.
                "roles_disjoint": bool(heads_out and heads_in
                                       and not (heads_out & heads_in)),
                "roles_unclassified": not (heads_out and heads_in),
                "gate_out": gate_out,
                "gate_in": gate_in,
                # `assess`'s strongest verdict, one module over: efficient in a
                # vacuum, wrong axis. Only the ONSET of a combat gate — any
                # change of gate at all fires on 38.1% of pairs.
                "newly_combat_gated": (gate_in == _assess.COMBAT
                                       and gate_out != _assess.COMBAT),
                "on_axis": [lbl for lbl, cards in axes.items() if out_name in cards],
                "declared": out_name in declared,
                "model_sees": _assess._channels({
                    "name": in_name, "oracle_text": in_text,
                    "cmc": float(info.get("cmc") or 0),
                    "type_line": info.get("type_line") or "",
                    "mana_cost": info.get("mana_cost") or "",
                    "power": None, "toughness": None}),
            })

    rows.sort(key=lambda r: (-r["strength"], r["out"], r["in"]))
    considered, dropped = rows[:limit], rows[limit:]

    # A PILE CARD WITH NO ROW IS AN ANSWER, NOT AN OMISSION. The index compares
    # a card against cards that do its job; one it has never paired with
    # anything in this list is a card this command cannot speak to, and saying
    # so is the difference between "nothing to report" and "nothing found".
    matched = {r["in"] for r in rows}
    unmatched = sorted(n for n in pool_set if n not in matched and n not in held)

    return {
        "slug": slug, "branch": branch,
        "min_strength": min_strength,
        "identity": sorted(identity),
        "swaps": considered,
        "not_considered": [{"out": r["out"], "in": r["in"],
                            "strength": r["strength"]} for r in dropped],
        "pool": sorted(pool_set) or None,
        "pool_unmatched": unmatched if pool_set else None,
        "notes": build_notes(considered, unmatched, bool(pool_set)),
    }


def build_notes(rows, unmatched, had_pool):
    """The traps said out loud, so no reader has to rediscover them."""
    notes = [
        "THESE ARE COMPARISONS, NOT VERDICTS. `strength` says how strongly the "
        "replacement outclasses the card it is set against, GIVEN that the two "
        "do the same job. Whether the difference is worth taking depends on "
        "this deck, which the index cannot see.",
        f"The scale tops out at {MAX_STRENGTH_IN_DATA} in the shipped data, not "
        f"1.0 — the base and its bonuses cap there. Nothing under "
        f"~{DEFAULT_MIN_STRENGTH} is a claim worth acting on, which is why it "
        f"is the floor.",
        "Every row carries what the card COSTS you as well as what it gains. A "
        "`narrows` entry is a gate the replacement adds and the original does "
        "not — a creature type it needs, or a restriction on when it works.",
    ]
    dis = [r for r in rows if r["roles_disjoint"]]
    if dis:
        notes.append(
            f"{len(dis)} row(s) pair cards that share NO job — the search "
            f"failed, not the comparison, and no strength figure can see it. "
            f"Read those two cards before anything else.")
    combat = [r for r in rows if r["newly_combat_gated"]]
    if combat:
        notes.append(
            f"{len(combat)} replacement(s) need a creature to CONNECT where the "
            f"card they replace does not. Efficient in a vacuum, wrong axis for "
            f"a deck that does not attack — check the thesis before the number.")
    unk = [r for r in rows if r["roles_unclassified"]]
    if unk:
        notes.append(
            f"{len(unk)} row(s) have a card with no classified role, so the "
            f"do-they-do-the-same-job check could not run at all. That is not "
            f"agreement — it is silence, and those rows carry one fewer guard.")
    unseen = [r for r in rows if r.get("model_sees") == []]
    if unseen:
        notes.append(
            f"{len(unseen)} replacement(s) are invisible to every model here, "
            f"so `candidates` can price them only by displacement — which reads "
            f"as noise and costs a full run to say so. Judge those by reading "
            f"the card.")
    if had_pool and unmatched:
        notes.append(
            f"{len(unmatched)} card(s) in your pile have no comparison against "
            f"anything in this list. That is not a verdict on them: the index "
            f"only pairs cards that do the same job, and it has never paired "
            f"these with one you run. `assess` is the reading for those.")
    notes.append(
        "NOTHING HERE IS MEASURED. A swap is priced by substituting it and "
        "re-running the diagnostic — `candidates <slug> --pool <file> --cut "
        "<card>` — and most single-card swaps are smaller than the run's "
        "minimum detectable difference.")
    return notes


def format_report(doc):
    where = doc["slug"] + (f"/{doc['branch']}" if doc.get("branch") else "")
    out = [f"\nUPGRADES — {len(doc['swaps'])} comparison(s) against {where}",
           f"  strength floor {doc['min_strength']} · "
           f"identity {''.join(doc['identity']) or 'colourless'}\n"]
    if not doc["swaps"]:
        out.append("  Nothing in this list has a candidate above the floor. "
                   "That is a real answer.\n")
    for r in doc["swaps"]:
        band = ("strong" if r["strength"] >= 0.65
                else "mild" if r["strength"] >= 0.4 else "weak")
        own = f"  [{r['source']}]" if r.get("source") else "  [buy]"
        out.append(f"  {r['strength']:.2f} {band:6} "
                   f"mv{r['mv_out']} {r['out'][:26]:26} -> "
                   f"mv{r['mv_in']} {r['in'][:26]:26}{own}")
        if r["gains"]:
            out.append(f"        + {', '.join(r['gains'])}")
        if r["costs"]:
            out.append(f"        - {', '.join(r['costs'])}")
        if r["narrows"]:
            out.append(f"        narrower: {', '.join(r['narrows'])}")
        if r["roles_disjoint"]:
            out.append(f"        !! NO SHARED JOB: {'/'.join(r['roles_out'])} -> "
                       f"{'/'.join(r['roles_in'])} — the search failed, read "
                       f"both cards")
        if r["newly_combat_gated"]:
            out.append(f"        !! newly COMBAT-gated ({r['gate_out']} -> "
                       f"combat): it has to connect")
        if r["played_more"] is False:
            out.append(f"        played less than what it replaces "
                       f"(rank {r['edhrec_rank']})")
        if r["on_axis"]:
            out.append(f"        the card it replaces feeds: "
                       f"{', '.join(x[:40] for x in r['on_axis'])}")
    if doc["not_considered"]:
        out.append(f"\n  {len(doc['not_considered'])} more above the floor, not "
                   f"shown — raise --limit to see them.")
    if doc.get("pool_unmatched"):
        out.append(f"\n  {len(doc['pool_unmatched'])} pile card(s) with no "
                   f"comparison here: "
                   f"{', '.join(doc['pool_unmatched'][:6])}"
                   + (" …" if len(doc["pool_unmatched"]) > 6 else ""))
    out.append("")
    for n in doc["notes"]:
        out.append(f"  · {n}")
    return "\n".join(out) + "\n"


def main(args):
    from manamap.pilot.candidates import read_pool
    from manamap.pilot.common import resolve_out_path
    pool = read_pool(getattr(args, "pool", None), args.slug)
    doc = propose(
        args.slug,
        branch=getattr(args, "branch", None),
        pool=pool,
        min_strength=getattr(args, "min_strength", None) or DEFAULT_MIN_STRENGTH,
        limit=getattr(args, "limit", None) or DEFAULT_LIMIT,
        owned_only=bool(getattr(args, "owned", False)))
    if getattr(args, "as_json", False) or getattr(args, "json", False):
        print(json.dumps(doc, indent=1))
    else:
        print(format_report(doc))
    if getattr(args, "out", None):
        path = resolve_out_path(args.out, args.slug)
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")
        print(f"  Wrote {path}")


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit("Run via `manamap pilot upgrades <slug>`.")
