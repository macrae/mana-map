"""The net change: what a branch would cost, what it would buy, and whether it met
what it set out to do.

THIS IS THE DOCUMENT A SPENDING DECISION RESTS ON. It was assembled by hand once —
eight commands and a page of HTML — to decide whether to buy 21 cards for the
Ur-Dragon treasure refactor. The answer was no, and the report is why the money
stayed in the bank. Doing that by hand again is how the next one gets skipped.

IT COMPOSES AND IT MEASURES NOTHING OF ITS OWN. Every figure here comes from a
command that already owns it — `diagnostic.compare` for the delta table with its
intervals and per-row MDE, `mana_analysis` for the colour half, `deck_branch.source`
for the bill, the tracked `sim/*.json` for the real table.

THE ENGINE LIFT WAS HERE AND WAS DELETED 2026-08-28, and the reason is the only
thing worth keeping from it. It split the games by whether the components marked
`required` in `goldfish_targets.json` had been drawn, and compared kill rates. But
that file is AUTHORED: the same hand writes the declaration and reads the verdict.
Measured on one Ur-Dragon list, one seed, 10,000 games, three defensible
declarations of the same 99 graded against kill-by-T8:

    ramp + a loosely worded payoff        +0.007  [-0.003, +0.017]  spans zero
    discount + ramp + burn, all required  -0.036  [-0.052, -0.020]  REAL
    ramp + burn                           +0.014  [+0.005, +0.023]  REAL

Same cards, same games, opposite signs — one of them saying at an interval
excluding zero that assembling the engine makes the deck win LESS. A figure whose
sign a JSON edit can flip is not evidence, however tight its interval, and it was
sitting in the block a spending decision reads first. What is left is arithmetic:
sampled rates with Newcombe intervals on the difference, deterministic
hypergeometric source counts, and a bill.

WHAT IT WILL AND WILL NOT DO. It grades the objective the branch was opened
with, names both sides of the trade, prices it, and ends on one of five stated
words — `recommend` is a RULE, written out in that function so it can be argued
with, not a score. What it will not do is collapse the axes into a number: they
move in both directions, weighting them would invent the weights, and this repo
deleted a six-factor card scorer for exactly that. Which side of a trade is
worth taking stays the pilot's call; the report's job is that the call is made
on figures whose meaning is on the page beside them.

EVERY FIGURE CARRIES ITS DEFINITION. `METRICS` is the registry — what each row
measures, why a pilot should care, and the unit it is in — and it renders with
the table rather than living in a doc nobody has open. A number a reader has to
go and look up is a number they will guess at instead, and the guesses are
wrong in a consistent direction: a mean read as a rate, a clock read as a win
percentage, a hoard read as mana. All three have happened here.
"""

import glob
import json

from manamap.pilot import deck_branch
from manamap.pilot.common import deck_dir, load_deck_cards, load_json
from manamap.sim import stats

ARTIFACT = "net_change.json"

#: What the report shows, and which direction is an improvement. Direction is
#: load-bearing: without it a lower stall reads as a loss. Same table
#: `diagnostic.interpret` keeps, and for the same reason.
ROWS = (
    ("hoard @T10", "output", "hoard_by_turn", "10", +1),
    ("hoard @T6", "output", "hoard_by_turn", "6", +1),
    ("damage @T10", "output", "damage_by_turn", "10", +1),
    ("board power @T6", "output", "board_power_by_turn", "6", +1),
    ("killed by T6", "output", "kill_by_turn", "6", +1),
    ("killed by T10", "output", "kill_by_turn", "10", +1),
    ("stall, 2 in a row", "stall", "two_in_a_row", None, -1),
    ("missed drop by T5", "mana", "missed_land_drop_by_five", None, -1),
    ("mulliganed", "mana", "mulliganed", None, -1),
)


#: WHAT EACH ROW MEANS AND WHY IT IS ON THE PAGE. Keyed by the label in `ROWS`.
#:
#: `unit` drives the plain-language reading beside each row: a `rate` is stated
#: as games per hundred because "0.318" and "32 games in 100" are the same fact
#: and only one of them is arguable at a table; a `mean` keeps its own units and
#: says what they are. `scale` is the yardstick the number is only meaningful
#: against — 40 life for damage, the fleet band for the three consistency rows.
METRICS = {
    "hoard @T10": {
        "unit": "mean",
        "what": "Treasures sitting in the hoard at the end of turn 10, averaged "
                "over every game.",
        "why": "The second axis, and the only one that is not combat. It is "
               "near-uncorrelated with damage, power and kill (r = 0.08-0.25 "
               "across the fleet), so it can fall while they rise and mean it. "
               "For this deck Treasures are incidental ramp, never an engine.",
    },
    "hoard @T6": {
        "unit": "mean",
        "what": "The same count at the end of turn 6 — the hoard you actually "
                "have when the deck wants to deploy, rather than the one you "
                "end up with.",
        "why": "A big turn-10 hoard that was not there on turn 6 arrived too "
               "late to cast anything that mattered.",
    },
    "damage @T10": {
        "unit": "mean",
        "what": "Cumulative damage dealt to the single goldfish opponent by the "
                "end of turn 10, averaged over every game.",
        "why": "The headline output figure and the one a doubler moves. It has "
               "no ceiling, so unlike the kill rate it keeps separating two "
               "lists after both of them already win.",
        "scale": "the opponent starts at 40 life, so 40.0 is exactly lethal once",
    },
    "board power @T6": {
        "unit": "mean",
        "what": "Total power on the battlefield at the end of turn 6.",
        "why": "The deck's turn-6 threat, before any of it has connected. It is "
               "what a table sees and decides to answer, and it moves earlier "
               "than damage does.",
    },
    "killed by T6": {
        "unit": "rate",
        "what": "The share of games in which cumulative damage reaches 40 by "
                "the end of turn 6.",
        "why": "THE CLOCK, at the turn a real pod is still setting up. This is "
               "the figure a faster list is bought for.",
        "scale": "a CLOCK against one unblocking opponent, never a win rate",
    },
    "killed by T10": {
        "unit": "rate",
        "what": "The same, by the end of turn 10 — the harness's last turn.",
        "why": "Whether the deck closes at all. It saturates near 1.0, so on a "
               "list that already wins it stops discriminating and damage @T10 "
               "is the honest axis instead.",
        "scale": "a CLOCK against one unblocking opponent, never a win rate",
    },
    "stall, 2 in a row": {
        "unit": "rate",
        "what": "The share of games with two consecutive turns, from turn 2 on, "
                "where nothing in hand could be cast with the mana available.",
        "why": "Two dead turns in a row is how a deck loses without ever being "
               "interacted with. Curve and colour problems both surface here.",
        "fleet": "stall_two_in_a_row",
    },
    "missed drop by T5": {
        "unit": "rate",
        "what": "The share of games that miss at least one land drop across "
                "turns 1 to 5.",
        "why": "The single best predictor of a slow start, and the row most "
               "sensitive to a land count change.",
        "fleet": "missed_land_drop_by_five",
    },
    "mulliganed": {
        "unit": "rate",
        "what": "The share of games that took at least one mulligan under the "
                "harness rule: keep a 7 with 2-5 lands, up to two redraws.",
        "why": "A proxy for whether the land count fits the curve. It should "
               "barely move unless the land count did.",
        "fleet": "mulliganed",
    },
}

#: The figures that are not rows. Same contract — a reader should never have to
#: leave the page to find out what a number is.
DERIVED = (
    ("THE OBJECTIVE",
     "One falsifiable threshold, written when the branch was OPENED and before "
     "anything was measured.",
     "It is the only thing here that can fail. Everything else is a "
     "description of what changed; this is the claim the branch made about "
     "itself, graded against its own minimum detectable difference."),
    ("THE MANA",
     "Karsten's hypergeometric source targets against the pip distribution "
     "each list actually has, and the on-curve probability the base achieves.",
     "Deterministic, and the nine sampled rows cannot see it. A branch that "
     "changes its spells changes its pip distribution, so the target moves "
     "underneath the base — which is why the GAP is the figure and not the "
     "source count."),
    ("MDE",
     "The minimum detectable difference: the smallest gap this many games "
     "could resolve at 95% confidence.",
     "A row marked `noise` is NOT a row where nothing happened. It is a row "
     "where whatever happened is smaller than this run can see, which rules "
     "out a large effect and says nothing about a small one."),
    ("THE BILL",
     "Every card in the branch, sorted by where it physically is: already in "
     "the deck, in a box, in another deck, or not owned.",
     "The only cost figure in the report. `buy` is money; `elsewhere` is a "
     "card that has to come out of a deck that is currently sleeved."),
)


def _cell(doc, block, key, turn=None):
    got = (doc.get(block) or {}).get(key)
    if turn and isinstance(got, dict):
        got = got.get(turn)
    return got if isinstance(got, dict) and "rate" in got else None


def mana(slug, branch):
    """The colour half, which the nine goldfish rows cannot see.

    THE REPORT DECIDED A PURCHASE WITHOUT IT. `ROWS` is derived from the
    goldfish, and the goldfish measures development, not castability by colour —
    so a branch that cut three counterspells (blue pips) and added six dorks
    changed its whole pip distribution and the report said nothing. Every
    figure here is `mana_analysis`'s, composed rather than recomputed, for the
    reason `mana_fit` composes it too: two modules that can disagree about one
    number is the divergence this repo keeps paying for.

    NOT IN `table`, and deliberately. Those rows carry a Newcombe interval on
    the difference; a source count is a deterministic hypergeometric claim with
    no sampling error at all, and giving it a `verdict` alongside them would
    make a different KIND of number look like the same kind.
    """
    from manamap.pilot import mana_analysis
    try:
        a = mana_analysis.analyze(slug)
        b = mana_analysis.analyze(slug, branch)
    except Exception as exc:                     # noqa: BLE001 - reported
        return {"available": False, "why": f"mana-analysis could not run: {exc}"}

    rows = []
    for c in "WUBRG":
        ta, tb = a["source_targets"].get(c, 0), b["source_targets"].get(c, 0)
        ha, hb = a["sources"]["total"].get(c, 0), b["sources"]["total"].get(c, 0)
        # THE GAP IS THE FIGURE, not the count. A colour whose target moved
        # because the pips moved is the whole point of running this after a
        # spell change, and `have` alone hides it.
        rows.append({"colour": c, "target": [ta, tb], "have": [ha, hb],
                     "gap": [ha - ta, hb - tb], "delta": (hb - tb) - (ha - ta)})
    return {
        "available": True,
        "colours": rows,
        "lands": [a["lands"]["total"], b["lands"]["total"]],
        "enters_tapped_always": [a["lands"]["enters_tapped_always"],
                                 b["lands"]["enters_tapped_always"]],
        "on_curve": {c: [a["on_curve_probability"]["with_rocks_and_dorks"].get(c),
                         b["on_curve_probability"]["with_rocks_and_dorks"].get(c)]
                     for c in "WUBRG"},
        "note": ("Deterministic, not sampled — Karsten's tables against the pip "
                 "distribution each list actually has. No interval, because "
                 "there is no sampling error to carry."),
    }


def changes(slug, branch):
    """THE SWAPS THEMSELVES, which the report used to state only as a count.

    "21 staged" is not a description of a treatment. A reader deciding whether
    to spend money needs to see WHICH cards moved and why each one moved, and
    the `why` was written at the moment the swap was staged — before any of the
    figures below existed, so it cannot have been fitted to them.

    Split into lands and spells because they answer different questions and are
    measured by different halves of this report: a spell swap moves the nine
    sampled rows, a land swap moves only the deterministic mana block. Mixing
    them lets a land pass borrow credit from a spell pass.
    """
    from manamap.pilot import card_pool
    meta = deck_branch.meta(slug, branch) or {}
    pool = card_pool.load_pool()

    def is_land(name):
        return "Land" in ((pool.get(name) or {}).get("type_line") or "")

    lands, spells = [], []
    for row in meta.get("staged") or []:
        entry = {"out": row.get("out"), "in": row.get("in"),
                 "why": row.get("why"), "at": row.get("at")}
        (lands if is_land(entry["in"]) or is_land(entry["out"])
         else spells).append(entry)
    return {"spells": spells, "lands": lands,
            "count": len(lands) + len(spells),
            "opened": meta.get("opened"), "why": meta.get("why")}


#: A role head this model is structurally blind to, and the sentence that says
#: so. Keyed to `load_card_roles`' vocabulary.
BLIND = {
    "removal": "the goldfish has no opponents, so a removal spell has nothing "
               "to remove and reads as a dead card",
    "counterspell": "the goldfish has no opponents, so a counterspell has "
                    "nothing to counter and reads as a dead card",
    "protection": "the goldfish is never attacked or targeted, so protection "
                  "reads as a dead card",
    "draw": "extra card draw is not modelled — one card per turn, always",
    "recursion": "nothing dies in a goldfish, so recursion has no target",
    "stax": "there is no opponent to tax",
    "hate": "there is no opponent to hate out",
}


def blind_spots(slug, branch, change_doc):
    """Which of THIS branch's swaps land in a hole in the model.

    THE MOST IMPORTANT PARAGRAPH IN THE REPORT AND THE ONE NOBODY WOULD THINK
    TO ASK FOR. Nine rows of figures with intervals read as a full accounting,
    and they are not: a branch that cut three counterspells and added a
    protection package changed nine cards the goldfish is structurally unable
    to price. The rows are not wrong — they are silent, and silence rendered
    beside a confidence interval reads as a measured zero.

    Derived from the swaps themselves rather than declared, so it cannot go
    stale when the treatment changes.
    """
    from manamap.pilot.common import load_card_roles
    roles = load_card_roles()
    found = {}
    for entry in change_doc["spells"]:
        for side in ("out", "in"):
            name = entry.get(side)
            for role in roles.get(name) or []:
                head = role.split(":", 1)[0]
                if head in BLIND:
                    found.setdefault(head, set()).add(name)
    out = [{"class": head, "why": BLIND[head], "cards": sorted(names),
            "headline": f"{len(names)} card(s) carrying a {head} effect"}
           for head, names in sorted(found.items())]
    if change_doc["lands"]:
        out.append({
            "class": "land",
            "headline": f"{len(change_doc['lands'])} land swap(s)",
            "why": "this model plays the first land in hand and credits its "
                   "colours the same turn — there is no tapped state, so it "
                   "cannot rank two lands that make the same colours. The "
                   "deterministic mana block is the whole of the evidence for "
                   "a land swap",
            "cards": sorted({e["in"] for e in change_doc["lands"]}
                            | {e["out"] for e in change_doc["lands"]}),
        })
    return out


def reads_as(row):
    """The row restated in a sentence a pilot can repeat at a table.

    A rate becomes games per hundred; a mean keeps its units and names them.
    Same numbers, and the only ones that survive being read aloud.
    """
    spec = METRICS.get(row["measure"]) or {}
    a, b, d = row["champion"], row["branch"], row["delta"]
    if row["verdict"] == "noise":
        return (f"no call — the gap of {abs(d):.3f} is smaller than the "
                f"{row['mde']:.3f} this run can resolve")
    if spec.get("unit") == "rate":
        return (f"{a * 100:.0f} games in 100 -> {b * 100:.0f} in 100, "
                f"a swing of {abs(d) * 100:.0f} games per 100")
    if abs(a) > 1e-9:
        return f"{a:.2f} -> {b:.2f}, a change of {d:+.2f} ({d / a:+.0%})"
    return f"{a:.2f} -> {b:.2f}, a change of {d:+.2f}"


def deck_file_or_none(slug, branch):
    from manamap.pilot.common import deck_file
    return deck_file(slug, "goldfish_targets.json", branch)


def forge(slug, branch):
    """The real table, if it has been played. Pooled within one pod only."""
    def rows_for(pattern, want):
        wins = games = 0
        by_route = {}
        for path in sorted(glob.glob(pattern)):
            if "logs" in path:
                continue
            doc = json.load(open(path))
            a = doc.get("analysis") or {}
            n = a.get("games") or 0
            for seat, v in (a.get("seats") or {}).items():
                if seat != want or v.get("wins") is None:
                    continue
                wins += v["wins"]
                games += n
            for g in (doc.get("games") or []):
                if g.get("winner") and want.split("@")[0] in str(g["winner"]):
                    by_route[g.get("won_by") or "unstated"] = \
                        by_route.get(g.get("won_by") or "unstated", 0) + 1
        return wins, games, by_route

    a_w, a_n, a_r = rows_for(f"data/decks/{slug}/sim/*.json", slug)
    b_w, b_n, b_r = rows_for(
        f"data/decks/{slug}/branches/{branch}/sim/*.json",
        deck_branch_seat(slug, branch))
    if not (a_n and b_n):
        return {"available": False,
                "why": ("no Forge run on " +
                        ("the branch" if a_n else "the deck") +
                        " — `manamap pilot simulate` puts both at a table")}
    d = stats.diff_proportions(a_w, a_n, b_w, b_n)
    m = stats.mde_proportion(a_w / a_n, a_n, b_n) or {}
    return {"available": True,
            "champion": {"wins": a_w, "games": a_n, "rate": round(a_w / a_n, 4),
                         "won_by": a_r},
            "branch": {"wins": b_w, "games": b_n, "rate": round(b_w / b_n, 4),
                       "won_by": b_r},
            "delta": round(b_w / b_n - a_w / a_n, 4),
            "ci95": d["ci95"], "excludes_zero": d.get("excludes_zero"),
            "mde": m.get("minimum_detectable_difference"),
            "caveat": ("Forge's AI is a weak pilot; the comparison is fair only "
                       "because both seats were played at a comparable rate — see "
                       "`sim/pilot_quality`.")}


def deck_branch_seat(slug, branch):
    from manamap.sim.forge import deck_meta_name
    return deck_meta_name(f"{slug}@{branch}")


def build(slug, branch, iterations=None, seed=None):
    from manamap.pilot import candidates, diagnostic
    it = iterations or diagnostic.HARNESS["iterations"]
    sd = seed if seed is not None else diagnostic.HARNESS["seed"]
    a = diagnostic.run(slug, iterations=it, seed=sd, quiet=True)
    b = diagnostic.run(slug, branch=branch, iterations=it, seed=sd, quiet=True)

    table = []
    for label, blk, key, turn, want in ROWS:
        ca, cb = _cell(a, blk, key, turn), _cell(b, blk, key, turn)
        if not (ca and cb):
            continue
        delta = round(cb["rate"] - ca["rate"], 4)
        mde = max(diagnostic.mde(ca), diagnostic.mde(cb))
        good = (delta > 0) == (want > 0)
        spec = METRICS.get(label) or {}
        row = {
            "measure": label, "champion": ca["rate"], "branch": cb["rate"],
            "delta": delta, "mde": round(mde, 4),
            # THE DEFINITION TRAVELS WITH THE FIGURE. `deck.html` renders this
            # artifact and had no way to say what a row meant; a reader who has
            # to leave the page to find out guesses instead, and the guesses go
            # one way — a mean read as a rate, a clock read as a win rate.
            "what": spec.get("what"), "why_we_care": spec.get("why"),
            "unit": spec.get("unit"), "scale": spec.get("scale"),
            "better_is": "higher" if want > 0 else "lower",
            "verdict": (("better" if good else "worse") if abs(delta) > mde
                        else "noise")}
        row["reads_as"] = reads_as(row)
        table.append(row)

    doc_meta = deck_branch.meta(slug, branch) or {}
    objective = doc_meta.get("objective")
    change_doc = changes(slug, branch)
    staged = len(doc_meta.get("staged") or [])
    grade = None
    if objective:
        block, key, turn = candidates.OBJECTIVE_AXES.get(
            objective["axis"], (None, None, None))
        cell = _cell(b, block, key, turn) if block else None
        grade = deck_branch.grade_objective(
            objective, cell["rate"] if cell else None,
            mde=diagnostic.mde(cell) if cell else None)

    doc = {
        "slug": slug, "branch": branch,
        "harness": {"iterations": it, "seed": sd},
        "decklist_sha256": (b.get("decklist_sha256")),
        "objective": objective,
        "objective_grade": grade,
        "staged": staged,
        "changes": change_doc,
        "blind_spots": blind_spots(slug, branch, change_doc),
        "definitions": {"rows": METRICS, "derived": [
            {"name": n, "what": w, "why": y} for n, w, y in DERIVED]},
        "table": table,
        "mana": mana(slug, branch),
        "forge": forge(slug, branch),
        "bill": deck_branch.source(slug, branch),
        "limits": [
            "The goldfish has no opponents and nothing blocks: its kill turn is a "
            "CLOCK, not a win rate, and it cannot see interaction, removal or any "
            "alternate win.",
            "Nothing here reads `goldfish_targets.json`'s `required` flags. The "
            "engine lift did, and was deleted for it: the declaration is "
            "authored, so the same hand set the target and read the verdict.",
            "Card advantage is measured nowhere in this suite.",
        ],
    }
    # Derived from the finished document, so it can never disagree with the rows
    # it summarises — the same reason `deck_info` composes and computes nothing.
    doc["recommendation"] = recommend(doc)
    return doc


def main(args):
    branch = getattr(args, "branch", None)
    if not branch:
        raise SystemExit(
            f"net-change compares a BRANCH against the deck. "
            f"`--branch <name>`; `manamap pilot deck-branch {args.slug} list` "
            f"shows what there is.")
    doc = build(args.slug, branch,
                iterations=getattr(args, "iterations", None),
                seed=getattr(args, "seed", None))
    if getattr(args, "as_json", False) or getattr(args, "json", False):
        print(json.dumps(doc, indent=1))
    else:
        _print(doc)
    if getattr(args, "write", False):
        out = deck_dir(args.slug, branch) / ARTIFACT
        out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(f"\n  Wrote {out}")


#: The states a recommendation may be in. A merge decision is not a scalar, and
#: five words is the whole vocabulary.
STATES = ("merge", "a trade", "do not merge", "inconclusive", "no objective")


def recommend(doc):
    """Sort the table into what rose, what fell, and what the run cannot tell.

    THE AXES MOVE IN BOTH DIRECTIONS ON PURPOSE, so a single number would have to
    weight them and every weight here would be invented — this repo deleted a
    six-factor card scorer for exactly that. What a pilot needs instead is the
    LEDGER plus a rule stated plainly enough to argue with:

        objective met, nothing fell      -> merge
        objective met, something fell    -> a trade; name both sides and the bill
        objective not met                -> do not merge
        objective stated but unreadable  -> inconclusive, and say WHY it is
        no objective at all              -> no objective; the ledger still stands

    THE LAST TWO ARE DIFFERENT AND WERE ONE STATE IN THE FIRST DRAFT. A branch
    that stated a goal the run could not read has been falsifiable all along and
    simply was not measured; a branch that stated none never could be. Collapsing
    them would let the second borrow the credibility of the first, which is the
    Ur-Dragon treasure branch's exact failure — it hit "treasure is the engine"
    4.4x over and missed the purpose nobody wrote down.

    NOTHING HERE RE-MEASURES. Every row's `verdict` was set against its own MDE in
    `build`; this reads them.
    """
    table = doc.get("table") or []
    rose = [r["measure"] for r in table if r["verdict"] == "better"]
    fell = [r["measure"] for r in table if r["verdict"] == "worse"]
    no_call = [r["measure"] for r in table if r["verdict"] == "noise"]

    objective, grade = doc.get("objective"), doc.get("objective_grade") or {}
    state = grade.get("state")
    if not objective:
        out = ("no objective",
               "This branch never stated what it was for, so nothing here can "
               "say whether it worked — only what changed.")
    elif state == "met":
        if fell:
            out = ("a trade",
                   f"You buy {_and(rose)} and pay {_and(fell)}.")
        else:
            out = ("merge",
                   f"The objective is met and nothing measured here got worse"
                   + (f"; {_and(rose)} improved." if rose else "."))
    elif state == "not met":
        out = ("do not merge",
               f"The objective is not met"
               + (f" — {grade.get('why')}" if grade.get("why") else ".")
               + (f" {_and(rose).capitalize()} improved anyway, which is a "
                  f"different branch's case." if rose else ""))
    elif state == "not resolvable":
        out = ("inconclusive",
               f"The miss is smaller than this run can see. "
               f"{grade.get('why', '')} A larger N is the only thing that "
               f"settles it.")
    else:                                            # "not measured", or absent
        out = ("inconclusive",
               f"The objective names {objective.get('axis')}, and this list has "
               f"no reading for it. That is a missing measurement, not a failure "
               f"— the axis may need a model flag set in goldfish_targets.json.")

    got = {"state": out[0], "because": out[1],
           "rose": rose, "fell": fell, "no_call": no_call,
           "bill": (doc.get("bill") or {}).get("counts") or {},
           "reward": reward(doc), "risk": risk(doc), "cost": cost(doc)}

    notes = []
    # THE MEASUREMENT IS DECK-LEVEL: swap a handful of cards, measure the lift.
    # One card USUALLY will not register — a 100-card singleton dilutes it below
    # what the run can resolve — but that is a statement about the typical card,
    # not a law. A Game Changer or a table-warper moves a number on its own, and
    # some cards are. So a blank table on a barely-changed branch is arithmetic
    # rather than a verdict on the swaps, and reading it as "these did nothing"
    # is the wrong lesson from a correct measurement. Measured: a one-swap
    # branch of ur-dragon returned noise on all nine rows.
    staged = doc.get("staged")
    if table and not rose and not fell:
        head = (f"Nothing moved: all {len(no_call)} measures came back inside "
                f"this run's minimum detectable difference.")
        if staged is not None and 0 < staged <= 3:
            notes.append(
                f"{head} With {staged} swap(s) staged this branch is nearly the "
                f"deck. A 100-card singleton dilutes one card below what this "
                f"run can resolve unless it is a Game Changer or a table-warper "
                f"— so this is not a verdict on the swap(s). Stage the rest of "
                f"the treatment and measure the lift on the whole thing.")
        else:
            notes.append(
                f"{head} The change is real and smaller than this run can "
                f"resolve — an answer about its SIZE, not a failure to measure.")

    # THE REAL TABLE IS EVIDENCE THE RULE DOES NOT USE, and hiding it because the
    # rule ignores it would be the worse error. Named beside the verdict, never
    # folded into it.
    f = doc.get("forge") or {}
    if f.get("available") and f.get("ci95"):
        lo, hi = f["ci95"]
        notes.append(
            f"Forge, against a real pod: {f['delta']:+.3f} win rate, "
            f"CI [{lo:+.3f}, {hi:+.3f}] — "
            + ("this run cannot separate the two lists."
               if (lo <= 0 <= hi) else "the difference excludes zero."))
    got["notes"] = notes
    return got


def reward(doc):
    """What the branch BUYS, each line stated in the row's own units.

    Composed from the table's verdicts, never re-measured — the same discipline
    `recommend` keeps. A row only appears here if it beat its own MDE.
    """
    out = []
    for r in doc.get("table") or []:
        if r["verdict"] == "better":
            out.append({"measure": r.get("measure"),
                        "reads_as": r.get("reads_as"),
                        "why_we_care": r.get("why_we_care")})
    return out


def risk(doc):
    """WHAT COULD BE WRONG WITH TAKING THIS, in four kinds, all derived.

    A report that lists only what improved is an argument, not a document. Each
    entry names its own kind so a reader can tell a measured loss from a thing
    the harness structurally cannot see — they read alike on a page and are not
    remotely the same claim:

      `paid`        a row that got measurably worse. A real, sized cost.
      `unresolved`  a row inside the MDE. Not "no change" — no answer.
      `unmeasured`  a swap the model is structurally blind to (`blind_spots`).
      `structural`  a caveat about the whole harness, not about this branch.
    """
    out = []
    for r in doc.get("table") or []:
        if r.get("verdict") == "worse":
            out.append({"kind": "paid", "what": r.get("measure"),
                        "detail": r.get("reads_as"),
                        "why_it_matters": r.get("why_we_care")})

    noise = [r for r in doc.get("table") or []
             if r.get("verdict") == "noise"]
    if noise:
        worst = max(noise, key=lambda r: abs(r.get("delta") or 0)
                    / (r.get("mde") or 1))
        out.append({
            "kind": "unresolved",
            "what": f"{len(noise)} row(s) returned no call",
            "detail": (f"the largest is {worst.get('measure')} at "
                       f"{(worst.get('delta') or 0):+.3f} against an MDE "
                       f"of {(worst.get('mde') or 0):.3f}. This run rules "
                       f"out a difference bigger than that and says "
                       f"nothing about a smaller one."),
            "why_it_matters": "an unresolved row is an open question, not a "
                              "settled zero"})

    for b in doc.get("blind_spots") or []:
        out.append({"kind": "unmeasured",
                    "what": b["headline"],
                    "detail": b["why"],
                    "cards": b["cards"],
                    # SCOPED TO THE EFFECT, NOT THE CARD, and the distinction is
                    # not pedantic: Solphim is a `protection:self` body AND a
                    # damage doubler the combat model prices at +7 damage. A
                    # line reading "3 protection cards are unmeasured" would
                    # file a measured card under unmeasured and understate the
                    # branch it is warning about.
                    "why_it_matters": "it is the EFFECT that no figure above "
                                      "can price, not the whole card — a body "
                                      "or a trigger on the same card is still "
                                      "measured"})

    m = doc.get("mana") or {}
    lost = [r for r in (m.get("colours") or []) if (r.get("delta") or 0) < 0]
    if lost:
        out.append({
            "kind": "paid",
            "what": "colour sources went backwards",
            "detail": ", ".join(f"{r['colour']} gap {r['gap'][0]:+d} -> "
                                f"{r['gap'][1]:+d}" for r in lost),
            "why_it_matters": "a source count is a hypergeometric claim about "
                              "opening hands; a widening gap is a real cost "
                              "even though no sampled row can see it"})

    f = doc.get("forge") or {}
    out.append({
        "kind": "structural",
        "what": ("no game has been played at a real table"
                 if not f.get("available") else "Forge is a weak pilot"),
        "detail": (f.get("why") or f.get("caveat") or ""),
        "why_it_matters": "every figure above is a goldfish: no blockers, no "
                          "removal, one opponent at 40 life who does nothing"})
    return out


#: A deck in one of these states is not competing for cardboard: its cards are
#: already loose. `deck_status` owns the vocabulary; this is the subset that
#: means "nothing has to be unsleeved to take this card".
FREE_TO_RAID = ("retired", "broken-down")


def cost(doc):
    """The bill, said in money and in sleeves rather than in four integers.

    THE `elsewhere` COUNT IS TWO DIFFERENT COSTS WEARING ONE NUMBER, and the
    difference is the whole of what a pilot needs to know before pulling
    sleeves. A card sitting in a RETIRED or BROKEN-DOWN deck is loose cardboard
    — taking it costs nothing and breaks nothing. A card sitting in a deck that
    is currently sleeved and played costs that deck the card. Reported as one
    integer, the six here read as six decks to disturb; three of them are in
    hapatra and sisay, which are already apart.
    """
    bill = doc.get("bill") or {}
    c = bill.get("counts") or {}
    total = sum(c.values()) or 0

    buy, loose, contested = [], [], []
    for row in bill.get("cards") or []:
        if row.get("state") == "buy":
            buy.append(row["name"])
        elif row.get("state") == "elsewhere":
            homes = row.get("where") or []
            live = sorted({h["slug"] for h in homes
                           if (h.get("status") not in FREE_TO_RAID)})
            (contested if live else loose).append(
                {"name": row["name"],
                 "decks": live or sorted({h["slug"] for h in homes})})

    parts = [f"{len(buy)} to buy"]
    if contested:
        parts.append(f"{len(contested)} to pull out of a deck that is still "
                     f"together")
    if loose:
        parts.append(f"{len(loose)} sitting in a retired or broken-down deck, "
                     f"which cost nothing")
    parts.append(f"{total - c.get('buy', 0)} of {total} already owned")

    return {
        "counts": c,
        "buy": len(buy), "buy_cards": sorted(buy),
        "must_unsleeve": contested,
        "free_to_raid": loose,
        "owned": total - c.get("buy", 0),
        "total": total,
        # THE PHYSICAL GATE, and it is not the same question as whether the
        # branch is a good idea. A branch can be worth merging and impossible
        # to merge today.
        "mergeable": bill.get("mergeable"),
        "reads_as": "; ".join(parts),
    }


def _and(names):
    if not names:
        return "nothing"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _wrap(text, width=74, indent="        "):
    """Prose wrapped to a terminal, because a definition nobody can read is not
    a definition. `textwrap` with an explicit width beats relying on the tty."""
    import textwrap
    return "\n".join(textwrap.wrap(str(text or ""), width=width,
                                   initial_indent=indent,
                                   subsequent_indent=indent))


def _print(doc):
    h = doc["harness"]
    print(f"\nNET CHANGE — {doc['slug']} vs branch {doc['branch']}"
          f"   ({h['iterations']:,} games each, seed {h['seed']})")

    rec = doc.get("recommendation") or {}

    # ---------------------------------------------------------- the verdict
    if rec:
        print(f"\n  ==> {rec['state'].upper()}")
        print(_wrap(rec["because"], indent="      "))
        for n in rec.get("notes") or []:
            print(_wrap(n, indent="      "))

    # ---------------------------------------------------------- the change
    ch = doc.get("changes") or {}
    if ch.get("count"):
        print(f"\n  THE CHANGE   {ch['count']} swap(s)"
              + (f", branch opened {ch['opened']}" if ch.get("opened") else ""))
        for title, rows in (("spells", ch.get("spells") or []),
                            ("lands", ch.get("lands") or [])):
            if not rows:
                continue
            print(f"\n    {title.upper()}  ({len(rows)})")
            for r in rows:
                print(f"      - {str(r['out'])[:30]:32} + {str(r['in'])[:30]}")
                if r.get("why"):
                    print(_wrap(r["why"], indent="          "))

    # ---------------------------------------------------------- objective
    o, g = doc.get("objective"), doc.get("objective_grade")
    print("\n  THE OBJECTIVE   the one thing here that can FAIL")
    if not o:
        print("    NONE — this branch predates the requirement and cannot be graded.")
    else:
        print(f"    {o['axis']} {o['op']} {o['value']}")
        state = (g or {}).get("state", "?").upper()
        print(f"    RESULT   {(g or {}).get('reading', '—')}   ->   {state}")
        if (g or {}).get("why"):
            print(_wrap(g["why"], indent="             "))
        if o.get("why"):
            print(_wrap("Written when the branch was opened: " + o["why"],
                        indent="    "))

    # ---------------------------------------------------------- the table
    print("\n  MEASURED   10,000 goldfish games per list, same seed, same harness")
    print(f"    {'measure':20} {'v1.0.1':>9} {'branch':>9} {'delta':>9}  verdict")
    for r in doc["table"]:
        print(f"    {r['measure']:20} {r['champion']:>9.3f} {r['branch']:>9.3f} "
              f"{r['delta']:>+9.3f}  {r['verdict']}"
              + (f" (MDE {r['mde']:.3f})" if r["verdict"] == "noise" else ""))
        print(f"      {r['reads_as']}")

    # ---------------------------------------------------------- definitions
    print("\n  WHAT THESE MEASURE, AND WHY THEY ARE ON THE PAGE")
    for r in doc["table"]:
        if not r.get("what"):
            continue
        print(f"\n    {r['measure']}   ({r['unit']}, {r['better_is']} is better"
              + (f"; {r['scale']}" if r.get("scale") else "") + ")")
        print(_wrap(r["what"]))
        print(_wrap("WHY: " + str(r["why_we_care"])))
    for d in (doc.get("definitions") or {}).get("derived") or []:
        print(f"\n    {d['name']}")
        print(_wrap(d["what"]))
        print(_wrap("WHY: " + d["why"]))

    # ---------------------------------------------------------- mana
    m = doc.get("mana") or {}
    print("\n  THE MANA — deterministic, and the nine rows above cannot see it")
    if not m.get("available"):
        print(f"    {m.get('why', 'not measured')}")
    else:
        print(f"    {'':2} {'target':>13} {'have':>13} {'gap':>13}   on curve")
        for r in m["colours"]:
            t, hv, gp = r["target"], r["have"], r["gap"]
            oc = m["on_curve"][r["colour"]]
            arrow = "" if gp[1] == gp[0] else ("  " + f"{r['delta']:+d}")
            print(f"    {r['colour']:2} {t[0]:>6} -> {t[1]:<5} {hv[0]:>6} -> {hv[1]:<5} "
                  f"{gp[0]:>+6} -> {gp[1]:<+5}  "
                  f"{(oc[0] or 0):.3f} -> {(oc[1] or 0):.3f}{arrow}")
        print(f"    lands {m['lands'][0]} -> {m['lands'][1]} "
              f"({m['enters_tapped_always'][0]} -> {m['enters_tapped_always'][1]} "
              f"always tapped)")
        print(_wrap("`on curve` is the probability the base casts a spell of "
                    "that colour on the turn Karsten's table sizes it for. "
                    "The GAP is the figure, not the source count: a branch "
                    "that changes its spells moves the target underneath the "
                    "base.", indent="    "))

    # ---------------------------------------------------------- real table
    f = doc["forge"]
    print("\n  THE REAL TABLE")
    if not f.get("available"):
        print(f"    {f['why']}")
    else:
        print(f"    champion {f['champion']['wins']}/{f['champion']['games']} "
              f"({f['champion']['rate']:.3f})   "
              f"branch {f['branch']['wins']}/{f['branch']['games']} "
              f"({f['branch']['rate']:.3f})")
        print(f"    delta {f['delta']:+.3f}  CI [{f['ci95'][0]:+.3f}, "
              f"{f['ci95'][1]:+.3f}]  MDE {f['mde']}")
        if f["mde"] and abs(f["delta"]) < f["mde"]:
            print(_wrap(f"UNDERPOWERED — this run could only resolve a "
                        f"difference of {f['mde']}; it rules out a large "
                        f"effect and cannot say which list is better.",
                        indent="    "))

    # ---------------------------------------------------------- the ledger
    print("\n  THE REWARD")
    for r in rec.get("reward") or []:
        print(f"    + {r['measure']}")
        print(_wrap(r["reads_as"]))
    if not rec.get("reward"):
        print("    nothing beat its own MDE.")

    print("\n  THE RISK")
    for r in rec.get("risk") or []:
        print(f"    [{r['kind']}] {r['what']}")
        if r.get("detail"):
            print(_wrap(r["detail"]))
        if r.get("cards"):
            print(_wrap("cards: " + ", ".join(r["cards"])))

    c = doc["bill"]["counts"]
    cst = rec.get("cost") or {}
    print("\n  THE COST   " + "   ".join(f"{k}={v}" for k, v in c.items()))
    print(_wrap(cst.get("reads_as", ""), indent="    "))
    if cst.get("buy_cards"):
        print(f"\n    BUY ({len(cst['buy_cards'])})")
        print(_wrap(", ".join(cst["buy_cards"]), indent="      "))
    if cst.get("must_unsleeve"):
        print(f"\n    UNSLEEVE ({len(cst['must_unsleeve'])}) — these come out "
              f"of a deck that is still together")
        for r in cst["must_unsleeve"]:
            print(f"      {r['name'][:34]:36} {', '.join(r['decks'])}")
    if cst.get("free_to_raid"):
        print(f"\n    FREE ({len(cst['free_to_raid'])}) — only in a retired or "
              f"broken-down deck, so nothing has to be disturbed")
        for r in cst["free_to_raid"]:
            print(f"      {r['name'][:34]:36} {', '.join(r['decks'])}")
    if cst.get("mergeable") is False:
        print(_wrap("NOT MERGEABLE YET — `deck-branch merge` refuses while any "
                    "card is unsourced. That is a question about cardboard, "
                    "not about whether the branch is right.", indent="    "))

    for line in doc["limits"]:
        print()
        print(_wrap(line, indent="  · ").replace("\n  · ", "\n    "))


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot net-change <slug> --branch <name>`.")
