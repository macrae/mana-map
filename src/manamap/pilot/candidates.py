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

# TWO KINDS OF AXIS, AND ONLY ONE CAN RANK WITHIN A COMPONENT.
# A MEMBERSHIP axis (`engine_online_*`, `any_route_*`) asks whether a card was
# DRAWN, so the ninth member of a group moves it by the same amount whichever
# card it is — measured on ur-dragon's treasure branch, all eight declared
# multipliers returned the identical +0.039. That is a true answer to "what is
# one more member worth" and no answer at all to "which member".
# A MAGNITUDE axis reads what the deck actually produced, so it can separate
# them. The goldfish has emitted these series all along and nothing exposed
# them; they arrive through `diagnostic.output`.
AXES = {
    "engine_online_3": ("engine", "online_by_turn", "3"),
    "engine_online_5": ("engine", "online_by_turn", "5"),
    "engine_online_8": ("engine", "online_by_turn", "8"),
    "any_route_8": ("engine", "any_route_by_turn", "8"),
    "stall": ("stall", "two_in_a_row", None),
    "land_drop": ("mana", "missed_land_drop_by_five", None),
    # magnitude
    "hoard_6": ("output", "hoard_by_turn", "6"),
    "hoard_10": ("output", "hoard_by_turn", "10"),
    # ONE COMBAT AXIS, BECAUSE THERE IS ONE COMBAT DIMENSION. Shipped as three
    # and measured across 13 decks at a uniform harness the same day:
    # power@6 vs damage@8 r=+0.97, power@6 vs kill@10 r=+0.98, damage@8 vs
    # kill@8 r=+0.92. Three axes that rank identically are a trap rather than a
    # redundancy — sweep on one, sweep on another, get the same order, and read
    # it as confirmation. That is the "three confirmations of one fact reading
    # as three findings" failure this repo already documents for the mana
    # family. `damage_8` is the survivor because it has no floor effect:
    # heliod's kill@8 is 0.001 and cannot discriminate anything, while damage
    # still separates decks that never close. All the series are still REPORTED
    # in `diagnostic.output` — they are free and a reader wants them — they are
    # simply not three things to rank on.
    "damage_8": ("output", "damage_by_turn", "8"),
    # STEAM — added 2026-08-28 WITH the independence check this comment block
    # demands, run across the eleven decks that carry a declaration:
    #
    #   interaction_6 vs mana@5   r = +0.68   (correlated, and not the same thing)
    #   interaction_6 vs stall    r = -0.20
    #   interaction_6 vs land_drop r = +0.17
    #   keep_t3       vs everything |r| <= 0.41
    #
    # Both clear the 0.90 bar this file rejects three combat axes for missing.
    # `interaction_6` is CONDITIONAL on holding an answer, which is what keeps it
    # from collapsing into "how much interaction do you run".
    #
    # THE SAME SWEEP FOUND land_drop vs mulligan AT r = +0.982 — two axes already
    # in `net_change.ROWS` that are one axis wearing two names. Recorded here
    # rather than fixed silently; see docs/gotchas-bench.md.
    "interaction_6": ("steam", "castable_given_in_hand_by_turn", "6"),
    "keep_t3": ("steam", "keep_can_act_by_t3", None),
}
#: Which axes need a deck to have opted into a model, and the flag to name when
#: it has not. A bare "no reading" would send the pilot looking for a bug.
AXIS_NEEDS = {
    "hoard_6": "model_treasures", "hoard_10": "model_treasures",
    "damage_8": "model_combat",
}
MAGNITUDE_AXES = tuple(AXIS_NEEDS)
#: WHAT A BRANCH MAY AIM AT — WIDER THAN WHAT A SWEEP MAY RANK ON, on purpose.
#:
#: `AXES` is deliberately narrow: three combat measures correlate at r = 0.92-0.98,
#: so ranking on more than one is three confirmations of one fact wearing three
#: names. That argument is about RANKING and does not transfer to STATING A GOAL.
#: A pilot's objective is "kill by turn eight", not "8.4 damage at turn eight" —
#: and a correlated measure is a perfectly good thing to aim at even when it is a
#: useless thing to sort by.
#:
#: Same one-concept-two-questions split as `cast_pips` vs `manabase.count_pips`
#: and `bodies` vs `creature_bodies`. Forcing them into one vocabulary would make
#: the honest goal unsayable to protect a ranking nobody is doing here.
#: AIMABLE BUT NOT RANKABLE, and the split is the one this file already draws.
#: `extra_cards_8` cannot have its independence checked: exactly ONE deck opts
#: into `model_draw`, so there is no fleet to correlate it against, and an axis
#: nobody can show to be independent must not be allowed to SORT a candidate
#: list. Stating it as a goal is a different act — the pilot says "draw more
#: cards" and means it — so it lives in OBJECTIVE_AXES alone until a second deck
#: opts in and the check can actually be run.
OBJECTIVE_AXES = dict(AXES, **{
    "extra_cards_8": ("steam", "extra_cards_by_turn", "8"),
    "kill_by_6": ("output", "kill_by_turn", "6"),
    "kill_by_8": ("output", "kill_by_turn", "8"),
    "kill_by_10": ("output", "kill_by_turn", "10"),
    "board_power_6": ("output", "board_power_by_turn", "6"),
    "hoard_8": ("output", "hoard_by_turn", "8"),
})

#: Axes where DOWN is better, so the ranking does not reward a worse deck.
LOWER_IS_BETTER = {"stall", "land_drop"}


def _read(doc, axis):
    block, key, sub = AXES[axis]
    got = ((doc.get(block) or {}).get(key)) or {}
    if sub:
        got = got.get(sub) or {}
    return got or None


def read_pool(spec, slug=None):
    """Names to consider: a file, a decklist, `-` for stdin, or `library`.

    `library` reads what the Atlas handed over — `data/decks/<slug>/pool.txt`,
    written by the `pool/save` endpoint when you press "consider these" on a
    pile. It is deliberately NOT the same slot as a brief's `must_include`: that
    one promises the cards are in the 99, and a card you are considering has made
    no such promise.
    """
    if spec in (None, ""):
        return []
    if spec == "library":
        from manamap.pilot.common import deck_dir
        path = deck_dir(slug) / "pool.txt"
        if not path.exists():
            raise SystemExit(
                f"{path} not found — open the Atlas library, pick a pile and "
                f"press 'consider these', or pass a file to --pool.")
        text = path.read_text(encoding="utf-8")
        from manamap.pilot.fetch_deck import parse_decklist
        got = [e["name"] for e in parse_decklist(text)]
        return got or [l.strip() for l in text.splitlines() if l.strip()]
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
          iterations=SWEEP_ITERATIONS, limit=DEFAULT_LIMIT, progress=None,
          join=None):
    """Baseline, then one substituted run per candidate."""
    if axis not in AXES:
        raise SystemExit(f"unknown axis {axis!r} — pick one of {', '.join(AXES)}")
    base = diagnostic.run(slug, branch=branch, iterations=iterations, quiet=True)
    # ABSENT ⇒ SAY WHICH FLAG, never a bare "no reading". The figure is missing
    # because the deck declined to model that half, which is a fact about the
    # declaration and not about the card being weighed.
    out = base.get("output") or {}
    if axis in AXIS_NEEDS and not out.get("available"):
        raise SystemExit(
            f"{axis!r} needs a magnitude reading and this list has none: "
            f"set \"{AXIS_NEEDS[axis]}\": true in goldfish_targets.json, "
            f"re-run `goldfish`, then ask again. "
            f"(An absent figure is not a zero.)")
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
        got = _with_swap(slug, branch, name, cut, axis, iterations, join=join)
        if got and "rate" in got:
            got["in_declaration"] = (name in declared) or bool(join)
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
    # IDENTICAL READINGS ARE A FINDING, NOT A FAULT — and this is the third time
    # in this file's history that identical numbers were the tell.
    #
    # A goldfish target asks whether a card was DRAWN. So when `--as` widens a
    # group, the SEVENTH member raises the draw probability by exactly the same
    # amount whichever card it is: Anointed Procession and Primal Vigor both read
    # +0.044 because the model counts draws, not effects. The answer is real —
    # widening this component by one buys 4.4 points — and it is an answer to
    # "how much is one more member worth", never to "which of these is best".
    # Eight identical rows would otherwise read as a broken tool.
    seen_rates = {r["rate"] for r in rows if "rate" in r}
    identical = len(seen_rates) == 1 and len([r for r in rows if "rate" in r]) > 1
    off = [r["card"] for r in rows
           if "rate" in r and not r.get("in_declaration")]
    note = None
    if axis.startswith(("engine", "any_route")) and off:
        note = (f"{len(off)} candidate(s) are not named in the engine "
                f"declaration, so on this axis they can only move the reading by "
                f"displacing another card. To test a card AS an engine piece, add "
                f"it to the relevant `any_of` group first, or measure it on an "
                f"axis it can reach (stall, land_drop).")
    if identical:
        note = ((note + " ") if note else "") + (
            "EVERY CANDIDATE READ THE SAME. A goldfish target asks whether a card "
            "was DRAWN, so widening a group by one raises its assembly rate by the "
            "same amount whichever card you add. This measures what ONE MORE "
            "MEMBER is worth — not which member. Choose between them on what they "
            "do once resolved, which this model cannot see.")
    return {"slug": slug, "branch": branch, "axis": axis, "note": note, "join": join,
            "all_identical": identical,
            "lower_is_better": axis in LOWER_IS_BETTER,
            "baseline": b, "cut": cut, "iterations": iterations,
            "candidates": rows,
            "considered": len(considered),
            # SAID OUT LOUD. A silently truncated list reads as the whole pool.
            "not_considered": dropped,
            "mde": diagnostic.mde(b)}


def _with_swap(slug, branch, name, cut, axis, iterations, join=None):
    """Substitute one card into the list and re-measure. Returns the row.

    `join` NAMES A DECLARED TARGET THE CANDIDATE SHOULD COUNT TOWARD, and without
    it a whole class of question is unaskable. A card the declaration does not
    name cannot move an engine axis except by displacing something — so asking
    "would this widen my thinnest component?" reads as noise, every time, for the
    right reason. Measured on ur-dragon's treasure branch: the bottleneck is a
    six-card multiplier group, and of twelve cards in the pilot's own library
    exactly one is a multiplier and it is already in the deck.

    So `join` adds the candidate to that group FOR THE MEASUREMENT ONLY. It is a
    hypothetical — "if this card were a multiplier, how much would the engine
    move" — and it is never written to the declaration, which stays authored.
    """
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
    targets = None
    if join:
        targets = _joined_targets(slug, branch, join, name)
        if targets is None:
            return {"card": name, "skipped": f"no target matching {join!r}"}
    got = diagnostic.run_on(swapped, slug, branch=branch, targets=targets,
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


def _joined_targets(slug, branch, label, name):
    """The declaration with one card added to one group — for this run only."""
    import copy
    from manamap.pilot.common import deck_file
    doc = load_json(deck_file(slug, "goldfish_targets.json", branch)) or {}
    targets = copy.deepcopy(doc.get("targets") or [])
    hit = [t for t in targets if label.lower() in t["label"].lower()]
    if not hit:
        return None
    for g in (hit[0].get("need") or []):
        if name not in g.get("any_of", []):
            g.setdefault("any_of", []).append(name)
        break
    return targets


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
                join=getattr(args, "join", None),
                progress=None if getattr(args, "json", False) else tick)
    if getattr(args, "json", False):
        print(json.dumps(doc, indent=1, default=str)); return
    _print(doc)


def _print(doc):
    b = doc["baseline"]
    where = doc["slug"] + (f"/{doc['branch']}" if doc.get("branch") else "")
    arrow = "lower is better" if doc["lower_is_better"] else "higher is better"
    print(f"\nCANDIDATES — {where}   axis {doc['axis']} ({arrow})")
    if doc.get("join"):
        print(f"  counted toward '{doc['join']}' for the measurement — a "
              f"hypothetical, not written to the declaration")
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
