"""Pilot: right-size the mana to the spells that are actually in the list.

    manamap pilot mana-fit <slug> [--branch N] [--owned]

THE STEP THAT WAS MISSING FROM EVERY REBUILD. `mana-analysis` measures the gap
between the colour sources a list HAS and the ones Karsten's tables say it NEEDS,
and then stops. Every refactor therefore ended with a shortfall and no answer,
and the shortfall MOVES the moment a spell changes: cutting three counterspells
takes blue pips out, adding six dorks puts sources in, and a manabase fitted to
the list before those changes is fitted to a list nobody is playing.

So this runs whenever the nonland half moves. It reads the pip distribution,
prices the gap per colour, and proposes lands, rocks and dorks that close it —
ranked by how many SHORT colours each one covers at once, because a five-colour
land is five fixes in one slot and a basic is one.

IT PROPOSES. IT DOES NOT MEASURE AND IT DOES NOT APPLY. The same doctrine
`close` and `upgrades` keep: `candidates` substitutes and re-measures, and
`net-change` is what a decision rests on. A source count is a hypergeometric
claim about opening hands, not a claim about winning.

WHAT IT DELIBERATELY WILL NOT DO IS ADD A BASIC TO FIX ONE COLOUR when a
multicolour land fixes four. A Swamp is one black source and nothing else; a
Command Tower is a black source AND the four others this deck is also short of.
Basics are still offered — they are free, always untapped and never dead — but
they rank last, because a five-colour deck that fixes its colours one basic at a
time ends up with a land base that cannot cast its own spells.
"""

import json

#: A land that always enters tapped costs a turn. It is not disqualifying — the
#: repo's own `enters_tapped_unconditionally` exists because "unless you control
#: two or more opponents" is always true in Commander and must NOT count — but
#: it ranks below an untapped source that fixes the same colours.
TAPPED_PENALTY = 0.5

#: How many proposals to show per bucket before saying how many were dropped.
DEFAULT_LIMIT = 10

#: Below this share of the pip weight, a colour is a SPLASH and its Karsten
#: target is a claim about one or two cards rather than about the deck. Reported
#: with that said out loud, because a target of 30 sources driven by a single
#: {B}{B} spell reads identically to one driven by thirty black cards.
SPLASH_PIP_SHARE = 0.08


def shortfall(slug, branch=None):
    """Per colour: what the list demands, what it has, and the gap.

    COMPOSED FROM `mana_analysis`, never recomputed. That module already owns
    every figure here — it gates production by the commander's colour identity
    and classifies a non-land producer through `nonland_producer_kind` rather
    than reading mana text off any card that happens to mention it. The first
    cut of this function recomputed the counts and reported 53 red sources
    against its 27, because `land_colors` applied to a spell answers a question
    nobody asked. Two modules that can disagree about one number is the
    divergence this repo keeps paying for.
    """
    from manamap.pilot import mana_analysis

    doc = mana_analysis.analyze(slug, branch)
    have = doc["sources"]["total"]
    targets = doc["source_targets"]
    rows = {}
    for c in "WUBRG":
        share = (doc["shares"].get(c) or {}).get("pip_share", 0.0)
        req = doc["pips"].get(c) or {}
        tgt = targets.get(c, 0)
        rows[c] = {
            "pip_cards": req.get("cards", 0),
            "effective_pips": req.get("effective_pips", 0),
            "pip_share": share,
            "target": tgt,
            "have": have.get(c, 0),
            "short": have.get(c, 0) - tgt,
            "splash": bool(share < SPLASH_PIP_SHARE and req.get("cards", 0)),
        }
    return {"slug": slug, "branch": branch, "colours": rows,
            "lands": doc["lands"]["total"],
            "tapped_always": doc["lands"]["enters_tapped_always"],
            "land_rows": doc["lands"]["list"],
            "on_curve": doc["on_curve_probability"]}


def propose(slug, branch=None, owned_only=False, limit=DEFAULT_LIMIT):
    """Lands, rocks and dorks that close the gap — ranked by colours covered."""
    from manamap.pilot import card_pool, collection, mana_analysis, manabase
    from manamap.pilot.common import deck_dir
    from manamap.pilot.fetch_deck import parse_decklist

    facts = shortfall(slug, branch)
    short = {c: -v["short"] for c, v in facts["colours"].items() if v["short"] < 0}
    over = sorted(c for c, v in facts["colours"].items() if v["short"] > 0)

    pool = card_pool.load_pool()
    oracle = card_pool.corpus_oracle()
    held = {e["name"] for e in parse_decklist(
        (deck_dir(slug, branch) / "decklist.txt").read_text(encoding="utf-8"))}
    identity = set()
    for e in parse_decklist((deck_dir(slug, branch) / "decklist.txt").read_text()):
        if e.get("is_commander"):
            identity |= set((pool.get(e["name"]) or {}).get("color_identity") or set())
    owned = collection.owned_names()
    # THE DECK'S OWN LANDS, so a CANDIDATE fetchland is scored on what it could
    # actually go and get in this list. Without it every fetch covers nothing
    # and is never offered — the one land class a five-colour deck most wants.
    deck_lands = [dict(pool[n], name=n, oracle_text=oracle.get(n, ""))
                  for n in held
                  if n in pool and "Land" in ((pool[n] or {}).get("type_line") or "")]

    adds = {"land": [], "rock": [], "dork": []}
    for name, info in pool.items():
        if name in held or not info.get("legal"):
            continue
        if not (info.get("color_identity") or set()) <= identity:
            continue
        if owned_only and name not in owned:
            continue
        card = dict(info, name=name, oracle_text=oracle.get(name, ""))
        colours = set(manabase.land_colors(card, pool=deck_lands))
        covers = sorted(colours & set(short))
        if not covers:
            continue
        type_line = card.get("type_line") or ""
        is_land = "Land" in type_line
        if is_land:
            kind = "land"
        else:
            # THE SAME GATE `mana_analysis` COUNTS THROUGH. Classifying by type
            # line alone offered Giant's Boulder as a five-colour rock — it is
            # an artifact that makes a Treasure, and the Treasure's reminder
            # text was being read as its own ability. Proposing a source the
            # counter would not count is a report that argues with itself.
            role = mana_analysis.nonland_producer_kind(card)
            if role == "ramp:dork":
                kind = "dork"
            elif role == "ramp:rock":
                kind = "rock"
            else:
                continue
        tapped = bool(is_land and manabase.enters_tapped_unconditionally(card))
        # THE SCORE IS COVERAGE, NOT QUALITY. How many colours this list is
        # actually short of does this one card answer — a five-colour land is
        # five fixes in one slot. Ties break on the tempo tax, then on how
        # widely the format plays it, which is the only quality signal here
        # that was not invented in this file.
        score = len(covers) - (TAPPED_PENALTY if tapped else 0)
        adds[kind].append({
            "name": name, "covers": covers, "colours": sorted(colours),
            "cmc": int(float(info.get("cmc") or 0)),
            "tapped": tapped, "owned": name in owned,
            "edhrec_rank": info.get("edhrec_rank"), "score": round(score, 2),
        })

    for kind in adds:
        adds[kind].sort(key=lambda r: (-r["score"], r["cmc"],
                                       r["edhrec_rank"] or 10 ** 9, r["name"]))

    # A CUT IS A LAND THAT ONLY FEEDS A COLOUR YOU ALREADY HAVE ENOUGH OF, or
    # one that always enters tapped while covering nothing scarce. Never a land
    # that touches a short colour, however slow it is.
    cuts = []
    for row in facts["land_rows"]:
        cols = set(row.get("produces") or [])
        if cols & set(short) or not cols:
            continue
        tapped = "tapped" in " ".join(row.get("classes") or [])
        if cols <= set(over):
            cuts.append({"name": row["name"], "colours": sorted(cols),
                         "quantity": row.get("copies", 1), "tapped": tapped,
                         "why": "only makes colours already at target"})

    trimmed = {k: v[:limit] for k, v in adds.items()}
    return {
        **facts,
        "short": short, "over": over,
        "add": trimmed,
        "not_shown": {k: max(0, len(adds[k]) - limit) for k in adds},
        "cut": cuts,
        "notes": build_notes(facts, short, over, cuts),
    }


def build_notes(facts, short, over, cuts):
    notes = [
        "A SOURCE COUNT IS A HYPERGEOMETRIC CLAIM ABOUT OPENING HANDS, not a "
        "claim about winning. Karsten's targets are the count that casts a spell "
        "on curve 90% of the time; missing one is a real cost and not a defect.",
        "THESE ARE PROPOSALS. Nothing here is measured — `candidates <slug> "
        "--pool <file> --cut <card>` substitutes and re-measures, and "
        "`net-change` is what a purchase rests on.",
        "RANKED BY COLOURS COVERED, so a five-colour land outranks a basic that "
        "fixes one. Basics are still offered — free, always untapped, never "
        "dead — but a five-colour deck that fixes its colours one basic at a "
        "time ends up unable to cast its own spells.",
    ]
    splash = [c for c, v in facts["colours"].items() if v["splash"]]
    if splash:
        named = ", ".join(
            f"{c} ({facts['colours'][c]['pip_cards']} card(s), "
            f"{facts['colours'][c]['pip_share']:.1%} of pips, target "
            f"{facts['colours'][c]['target']})" for c in splash)
        notes.append(
            f"SPLASH COLOURS, and their targets are the loudest number here for "
            f"the smallest reason: {named}. A target driven by one or two cards "
            f"reads identically to one driven by thirty. Cutting the card is "
            f"usually cheaper than buying the sources — decide that first, "
            f"because everything below is sized against it.")
    if facts["tapped_always"]:
        notes.append(
            f"{facts['tapped_always']} land(s) always enter tapped. That is a "
            f"tempo tax on every one of them, and it is why an untapped source "
            f"outranks a tapped one covering the same colours here.")
    if not short:
        notes.append("Every colour is at or above its target. The mana is not "
                     "what is limiting this list.")
    if not cuts:
        notes.append("No land is safe to cut on colour grounds: every one feeds "
                     "a colour this list is still short of.")
    return notes


def format_report(doc):
    where = doc["slug"] + (f"/{doc['branch']}" if doc.get("branch") else "")
    out = [f"\nMANA FIT — {where}",
           f"  {doc['lands']} land(s), {doc['tapped_always']} always tapped\n",
           f"  {'':2} {'cards':>6} {'pips':>5} {'share':>7} "
           f"{'target':>7} {'have':>5} {'gap':>6}"]
    for c in "WUBRG":
        r = doc["colours"][c]
        mark = "  SPLASH" if r["splash"] else ""
        out.append(f"  {c:2} {r['pip_cards']:>6} {r['effective_pips']:>5} "
                   f"{r['pip_share']:>6.1%} {r['target']:>7} {r['have']:>5} "
                   f"{r['short']:>+6}{mark}")
    if doc["short"]:
        worst = sorted(doc["short"].items(), key=lambda kv: -kv[1])
        out.append("\n  SHORT: " + ", ".join(f"{c} by {n}" for c, n in worst))
    if doc["over"]:
        out.append(f"  AT OR OVER TARGET: {', '.join(doc['over'])}")

    for kind, title in (("land", "LANDS"), ("rock", "ROCKS"), ("dork", "DORKS")):
        rows = doc["add"][kind]
        if not rows:
            continue
        out.append(f"\n  {title} that cover what is short")
        for r in rows:
            tag = "tapped" if r["tapped"] else ""
            own = "OWN" if r["owned"] else "   "
            out.append(f"    {own} mv{r['cmc']} {r['name'][:30]:32} "
                       f"+{''.join(r['covers']):5} {tag:6} "
                       f"#{r['edhrec_rank'] or '-'}")
        if doc["not_shown"][kind]:
            out.append(f"    ({doc['not_shown'][kind]} more not shown)")

    if doc["cut"]:
        out.append("\n  CUT CANDIDATES (they fix nothing this list is short of)")
        for r in doc["cut"]:
            out.append(f"    - {r['name'][:32]:34} {''.join(r['colours']):6} "
                       f"{r['why']}")
    out.append("")
    for n in doc["notes"]:
        out.append(f"  · {n}")
    return "\n".join(out) + "\n"


def main(args):
    doc = propose(args.slug, branch=getattr(args, "branch", None),
                  owned_only=bool(getattr(args, "owned", False)),
                  limit=getattr(args, "limit", None) or DEFAULT_LIMIT)
    if getattr(args, "json", False) or getattr(args, "as_json", False):
        print(json.dumps(doc, indent=1))
    else:
        print(format_report(doc))


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit("Run via `manamap pilot mana-fit <slug>`.")
