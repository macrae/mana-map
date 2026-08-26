"""Pilot: rank a pool of cards by what each one MEASURABLY does to the deck.

    manamap pilot candidates <slug> [--branch N] --pool <file|library|-> \
        --axis engine_online_3 [--cut <card>] [--limit N]

NOT A SCORER. This repo deleted a six-factor card scorer once already — its
weights disagreed with the pipeline's, and the note left behind says evaluation
belongs to the routine that actually measures something. So nothing here scores a
card on its properties. Each candidate is SUBSTITUTED INTO THE LIST, the
diagnostic is re-run, and what is reported is the difference with an interval on
it. That is a measurement, which is a different kind of object from a heuristic.

IT IS ONLY POSSIBLE BECAUSE THE RUN IS CHEAP. The whole diagnostic stack is 12.6s
on one deck, and a reduced-iteration pass is under two seconds — so forty
candidates is minutes, not hours, and the loop of change-and-remeasure closes
inside one sitting.

THE COMPARISON IS UNPAIRED, and that is not a shortcut. Changing one card
reshuffles every game, so a shared seed buys replayability and never pairing —
the same fact `experiment`'s assumptions block states. Newcombe on the
difference; never two marginal intervals side by side, because they can overlap
while the interval on the difference excludes zero.

AND MOST SINGLE-CARD SWAPS ARE INVISIBLE. At any workable iteration count the
minimum detectable difference is a real number, and a ranking of deltas smaller
than it is a ranking of noise. Every row carries the MDE and anything under it is
marked, rather than being silently ordered.
"""

import copy
import json

from manamap.pilot import diagnostic
from manamap.pilot.common import deck_dir, load_json
from manamap.sim import stats as st

#: A sweep is N+1 diagnostic runs, so the per-run cost is the whole budget.
#: 2000 games keeps a run near 1.5s while holding the MDE around 0.03 — enough to
#: see a swap that matters and honest about one that does not.
SWEEP_ITERATIONS = 2000

#: Reported, never silent. A truncated list reads as "these are all of them".
DEFAULT_LIMIT = 40

AXES = {
    "engine_online_3": ("engine", "online_by_turn", "3"),
    "engine_online_5": ("engine", "online_by_turn", "5"),
    "engine_online_8": ("engine", "online_by_turn", "8"),
    "any_route_8": ("engine", "any_route_by_turn", "8"),
    "stall": ("stall", "two_in_a_row", None),
    "land_drop": ("mana", "missed_land_drop_by_five", None),
}
#: Axes where DOWN is better, so the ranking does not reward a worse deck.
LOWER_IS_BETTER = {"stall", "land_drop"}


def _read(doc, axis):
    block, key, sub = AXES[axis]
    got = ((doc.get(block) or {}).get(key)) or {}
    if sub:
        got = got.get(sub) or {}
    return got or None


def read_pool(spec, slug=None):
    """Names to consider. A file of card names, or a decklist, or `-` for stdin."""
    if spec in (None, ""):
        return []
    if spec == "-":
        import sys
        text = sys.stdin.read()
    else:
        from pathlib import Path
        text = Path(spec).read_text(encoding="utf-8")
    # The same reader the rest of the bench uses, so a pool pasted from the Atlas
    # and a pool exported as a decklist cannot disagree about what is in it.
    from manamap.pilot.fetch_deck import parse_decklist
    names = [e["name"] for e in parse_decklist(text)]
    if not names:
        names = [l.strip() for l in text.splitlines()
                 if l.strip() and not l.strip().startswith("#")]
    return list(dict.fromkeys(names))


def sweep(slug, pool, axis="engine_online_3", branch=None, cut=None,
          iterations=SWEEP_ITERATIONS, limit=DEFAULT_LIMIT, progress=None):
    """Baseline, then one substituted run per candidate."""
    if axis not in AXES:
        raise SystemExit(f"unknown axis {axis!r} — pick one of {', '.join(AXES)}")
    base = diagnostic.run(slug, branch=branch, iterations=iterations, quiet=True)
    b = _read(base, axis)
    if not b:
        raise SystemExit(
            f"the baseline has no reading for {axis!r}"
            + (f" — {(base.get('engine') or {}).get('why', '')}"
               if axis.startswith(("engine", "any_route")) else ""))
    considered, dropped = pool[:limit], pool[limit:]
    declared = _declared_cards(slug, branch)
    rows = []
    for i, name in enumerate(considered):
        if progress:
            progress(i + 1, len(considered), name)
        got = _with_swap(slug, branch, name, cut, axis, iterations)
        if got and "rate" in got:
            got["in_declaration"] = name in declared
        if got is None:
            rows.append({"card": name, "skipped": "already in the list"})
            continue
        rows.append(got)
    rows.sort(key=lambda r: (r.get("delta") is None,
                             (r.get("delta") or 0) * (1 if axis in LOWER_IS_BETTER else -1)))
    # A CARD THE DECLARATION DOES NOT NAME CANNOT MOVE AN ENGINE AXIS except by
    # displacing something, and the reading will be a small delta that looks like
    # a finding. Measured: Jeweled Lotus, Mana Crypt and Rhystic Study all came
    # back at exactly 0.0833 against a 0.125 baseline — the same number, because
    # the only thing that changed was the card that came out. Say so rather than
    # letting three identical rows read as three weak results.
    off = [r["card"] for r in rows
           if "rate" in r and not r.get("in_declaration")]
    note = None
    if axis.startswith(("engine", "any_route")) and off:
        note = (f"{len(off)} candidate(s) are not named in the engine "
                f"declaration, so on this axis they can only move the reading by "
                f"displacing another card. To test a card AS an engine piece, add "
                f"it to the relevant `any_of` group first, or measure it on an "
                f"axis it can reach (stall, land_drop).")
    return {"slug": slug, "branch": branch, "axis": axis, "note": note,
            "lower_is_better": axis in LOWER_IS_BETTER,
            "baseline": b, "cut": cut, "iterations": iterations,
            "candidates": rows,
            "considered": len(considered),
            # SAID OUT LOUD. A silently truncated list reads as the whole pool.
            "not_considered": dropped,
            "mde": diagnostic._mde(b["rate"], b["n"], b["n"])}


def _with_swap(slug, branch, name, cut, axis, iterations):
    """Substitute one card into the list and re-measure. Returns the row."""
    from manamap.pilot import goldfish
    doc = _load_cards(slug, branch)
    held = {c["name"] for c in doc["cards"]}
    if name in held:
        return None
    swapped = copy.deepcopy(doc)
    if cut:
        swapped["cards"] = [c for c in swapped["cards"] if c["name"] != cut]
    else:
        # No cut named: drop the highest-mana non-commander spell, so the size
        # holds. Stated in the output, because the choice of cut moves the
        # answer and a sweep that picked one silently would hide half the
        # experiment.
        #
        # IT MUST NOT EAT AN ENGINE PIECE. The first version took the most
        # expensive card outright, which on ur-dragon is Utvara Hellkite — named
        # in a declared target — so every candidate came back "no reading":
        # `declaration_fits` correctly refused to measure an engine whose
        # declaration no longer described the list. The sweep was testing its own
        # cut, not the candidates.
        declared = _declared_cards(slug, branch)
        pool_cards = [c for c in swapped["cards"]
                      if not c.get("is_commander")
                      and "Land" not in (c.get("type_line") or "")
                      and c["name"] not in declared]
        if pool_cards:
            victim = max(pool_cards, key=lambda c: float(c.get("cmc") or 0))
            swapped["cards"] = [c for c in swapped["cards"]
                                if c["name"] != victim["name"]]
            cut = victim["name"]
    add = _resolve(name)
    if add is None:
        return {"card": name, "skipped": "not in the corpus"}
    swapped["cards"].append(add)
    got = diagnostic.run_on(swapped, slug, branch=branch,
                            iterations=iterations, quiet=True)
    r = _read(got, axis)
    if not r:
        why = (got.get("engine") or {}).get("why", "")
        return {"card": name, "skipped": "no reading", "why": why[:120]}
    row = _row(name, r)
    row["cut"] = cut
    return row


def _in_declaration(name, declared):
    return name in declared


def _declared_cards(slug, branch):
    """Every card the engine declaration names — never an automatic cut."""
    from manamap.pilot.common import deck_file
    doc = load_json(deck_file(slug, "goldfish_targets.json", branch)) or {}
    return {c for t in (doc.get("targets") or [])
            for g in (t.get("need") or []) for c in (g.get("any_of") or [])}


def _row(name, r):
    return {"card": name, "rate": r["rate"], "n": r["n"], "_r": r}


def _load_cards(slug, branch):
    from manamap.pilot.common import load_deck_cards
    return load_deck_cards(slug, branch)


def _resolve(name):
    """A corpus row shaped like a cards.json entry — enough for the goldfish."""
    from manamap.pilot import card_pool
    frame = card_pool.load_frame()
    hit = frame[frame["name"] == name]
    if not len(hit):
        return None
    r = hit.iloc[0]
    return {"name": name, "quantity": 1, "is_commander": False,
            "cmc": float(r.get("cmc") or 0),
            "type_line": str(r.get("type_line") or ""),
            "oracle_text": str(r.get("oracle_text") or ""),
            "mana_cost": str(r.get("mana_cost") or ""),
            "power": r.get("power"), "toughness": r.get("toughness")}


# ── CLI ──────────────────────────────────────────────────────────────────

def main(args):
    pool = read_pool(getattr(args, "pool", None), args.slug)
    if not pool:
        raise SystemExit(
            "no pool — pass `--pool <file>` (card names, one per line, or a "
            "decklist) or `--pool -` to read from stdin.")
    axis = getattr(args, "axis", None) or "engine_online_3"
    branch = getattr(args, "branch", None)

    def tick(i, n, name):
        print(f"  [{i}/{n}] {name}", flush=True)

    doc = sweep(args.slug, pool, axis=axis, branch=branch,
                cut=getattr(args, "cut", None),
                iterations=getattr(args, "iterations", None) or SWEEP_ITERATIONS,
                limit=getattr(args, "limit", None) or DEFAULT_LIMIT,
                progress=None if getattr(args, "json", False) else tick)
    if getattr(args, "json", False):
        print(json.dumps(doc, indent=1, default=str)); return
    _print(doc)


def _print(doc):
    b = doc["baseline"]
    where = doc["slug"] + (f"/{doc['branch']}" if doc.get("branch") else "")
    arrow = "lower is better" if doc["lower_is_better"] else "higher is better"
    print(f"\nCANDIDATES — {where}   axis {doc['axis']} ({arrow})")
    print(f"  baseline {b['rate']:.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]"
          f"   {doc['iterations']} games each")
    print(f"  smallest difference this many games can see: {doc['mde']}\n")
    print(f"  {'card':32} {'reading':>8} {'delta':>8}   note")
    for r in doc["candidates"]:
        if "rate" not in r:
            print(f"  {r['card'][:32]:32} {'—':>8} {'—':>8}   {r.get('skipped','')}")
            continue
        d = r["rate"] - b["rate"]
        mark = "" if abs(d) >= (doc["mde"] or 0) else "under the MDE — noise"
        if not r.get("in_declaration"):
            mark = (mark + "; " if mark else "") + "not in the declaration"
        print(f"  {r['card'][:32]:32} {r['rate']:>8.3f} {d:>+8.3f}   {mark}")
    if doc.get("note"):
        print(f"\n  ! {doc['note']}")
    if doc["not_considered"]:
        print(f"\n  {len(doc['not_considered'])} card(s) beyond --limit were NOT "
              f"considered: {', '.join(doc['not_considered'][:6])}"
              + (" …" if len(doc["not_considered"]) > 6 else ""))
