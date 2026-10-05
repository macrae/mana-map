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
import math
import hashlib
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
    # STEAM — the three the captain's log kept asking for and nothing measured.
    ("extra cards by T8", "steam", "extra_cards_by_turn", "8", +1),
    ("interaction affordable @T6", "steam", "castable_given_in_hand_by_turn", "6", +1),
    ("hand can act by T3", "steam", "keep_can_act_by_t3", None, +1),
)


#: WHAT EACH ROW MEANS AND WHY IT IS ON THE PAGE. Keyed by the label in `ROWS`.
#:
#: `unit` drives the plain-language reading beside each row: a `rate` is stated
#: as games per hundred because "0.318" and "32 games in 100" are the same fact
#: and only one of them is arguable at a table; a `mean` keeps its own units and
#: says what they are. `scale` is the yardstick the number is only meaningful
#: against — 40 life for damage, the fleet band for the three consistency rows.
METRICS = {
    "extra cards by T8": {
        "unit": "mean",
        "scale": "against the 8 cards every deck draws for free by then",
        "what": "Cards drawn BEYOND the one-a-turn draw step, cumulative to the "
                "end of turn 8, averaged over every game. Counts only draw the "
                "model can price — a card's own ETB draw, an instant or sorcery "
                "that draws, an upkeep trigger, and a trigger that draws when "
                "other creatures or tokens enter. Activated, X-based, "
                "sacrifice-gated and death-triggered draw are NOT counted and "
                "the cards are named in the goldfish artifact.",
        "why": "\"Vampires on board, nothing in hand, no way to rebuild\" is the "
               "most-repeated line in this deck's log, and until 2026-08-28 the "
               "model could not see it: it drew one card a turn whatever the "
               "list said. Read it against the count of draw cards, not alone — "
               "a deck listing twelve draw sources whose modelled figure is one "
               "has almost no UNCONDITIONAL card advantage, and that is a fact "
               "about the deck rather than a limit of the parser.",
    },
    "interaction affordable @T6": {
        "unit": "rate",
        "scale": "conditional on holding one — the denominator is games where "
                 "an answer was in hand, not all games",
        "what": "Of the games where an answer was in hand on turn 6, the share "
                "where the turn also ended with enough unspent mana to cast it. "
                "The suite is removal + sweepers + protection, the same set "
                "`deck-audit` counts.",
        "why": "The pilot's own diagnosis of a lost game: \"Deflecting Swat and "
               "Teferi's Protection sat in my hand uncast for the entire game "
               "... my mana was spent casting vampires, so there was never open "
               "mana to hold up.\" This is that claim as a number. It is "
               "CONDITIONAL on purpose: the raw castable rate falls when you "
               "simply cut interaction, which answers a different question. "
               "Measured at the END OF THE MAIN PHASE, before combat is paid "
               "for. And it is a FLOOR — the model casts everything it can "
               "afford every turn, so a pilot choosing to hold up scores higher.",
    },
    "hand can act by T3": {
        "unit": "rate",
        "scale": "against the keep rate itself — this is a property of the "
                 "seven cards kept, assuming no draws",
        "what": "The share of kept opening hands containing a nonland card the "
                "hand's own lands can pay for by turn three, counting at most "
                "three land drops and no draws.",
        "why": "A stricter threshold than the 2-5 lands rule the model actually "
               "mulligans by, and it is REPORTED rather than enforced, so it "
               "cannot restate a figure on any deck. It separates \"this hand "
               "had lands\" from \"this hand could do something\" — the "
               "distinction behind a two-land keep that never got going.",
    },
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
        # NOT CUMULATIVE, AND IT SAID SO HERE FOR MONTHS. The source is
        # `combat.mean_damage_by_turn["10"]`, and `goldfish_turn` resets `dealt`
        # to 0 INSIDE the turn loop before appending it — so the series is what
        # was dealt ON each turn, never a running total. The file's own naming
        # settles it: the event-damage pillar carries BOTH
        # `mean_event_damage_by_turn` (2.288 / 2.978 / 3.87 at T8/9/10) and
        # `mean_cumulative_event_damage_by_turn` (6.372 / 9.35 / 13.22), and
        # this row has only the first shape and no cumulative twin.
        #
        # Found by the deck-engineer on sharknado, which noticed that 53.904 at
        # turn ten sits right on top of board power 18.334 plus 32.765 commander
        # counters — one swing, not ten turns of them. The old `scale` line made
        # it worse by anchoring the reader to a 40-life total, which is the
        # cumulative reading.
        "what": "Damage dealt to the single goldfish opponent ON turn 10, "
                "averaged over every game — the clock at its tenth turn, not a "
                "running total. The cumulative series does not exist for this "
                "pillar; `mean_cumulative_event_damage_by_turn` is the only "
                "running total the model keeps, and it covers event damage "
                "alone.",
        "why": "The headline output figure and the one a doubler moves. It has "
               "no ceiling, so unlike the kill rate it keeps separating two "
               "lists after both of them already win.",
        "scale": "one turn's damage, so 40.0 would be lethal in a single swing "
                 "from an opponent at full life — not a total across the game",
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
    ("PRIMARY AND EXPLORATORY",
     "The objective is the one pre-registered test. The twelve rows are a "
     "family of looks, each with an interval on its difference, and a row's "
     "verdict is Holm-corrected across the family.",
     "Twelve rows at alpha 0.05 with no correction is one false 'better' in "
     "every other report. `experiment.py` pre-registers `win_rate` for the "
     "same reason. Measured over 42 tracked reports before shipping: Holm "
     "flips no verdict — the MDE rule was already near Bonferroni at the top "
     "rank — so this is what a verdict means, not which rows carry one."),
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


#: The per-game readers, borrowed from `experiment` so that a Forge mean here
#: is the figure `experiment` would report — one definition, two commands.
from manamap.sim.experiment import PER_GAME as _EXP_PER_GAME  # noqa: E402
_PER_GAME = {k: _EXP_PER_GAME[k] for k in
             ("combat_damage_dealt_to_players", "first_attack_turn", "eliminated_turn",
              "drain_dealt", "biggest_hit", "evasive_damage_share", "kills_by_ability", "life_gained",
              "extra_draw_per_turn", "empty_hand_turns", "noncombat_damage_dealt_to_players",
              "life_removed_total")}


def _null_block(pod):
    """The pod's subject null as `pods.calibration` states it, with the pod's
    name on it — or (None, why). Absent means absent."""
    try:
        from manamap.sim import pods
        cal = pods.calibration(pod) or {}
        row = cal.get("subject_null")
    except Exception as exc:                    # noqa: BLE001
        return None, f"no calibration for {pod}: {exc.__class__.__name__}"
    if not row or not isinstance(row.get("rate"), (int, float)) or row["rate"] <= 0:
        return None, f"{pod} has no measured subject null"
    return {"pod": pod, "rate": row["rate"], "ci95": row.get("ci95"),
            "wins": row.get("wins"), "games": row.get("games"),
            "runs": cal.get("runs"), "decks": len(cal.get("decks") or []),
            "excluded_overridden_runs": cal.get("excluded_overridden_runs"),
            "basis": "what our decks score in seat 0 at this table, plain "
                     "harness, deck-level runs"}, None


def _arm_summary(row):
    """What one pooled arm reads on every Forge endpoint, with n."""
    from manamap.pilot import candidates
    out = {}
    n = row["decided"]
    out["forge.win_rate"] = {"k": row["wins"], "n": n,
                             "value": round(row["wins"] / n, 4) if n else None}
    rg = row.get("resolved_games") or 0
    out["forge.commander_resolved_rate"] = (
        {"k": row["resolved"], "n": rg, "value": round(row["resolved"] / rg, 4)}
        if rg else {"k": None, "n": 0, "value": None})
    for axis, spec in candidates.FORGE_OBJECTIVE_AXES.items():
        if spec["kind"] != "mean":
            continue
        vals = (row.get("per_game") or {}).get(spec["per_game"]) or []
        if len(vals) >= 2:
            mean, sd = stats._mean_sd(vals)
            srt = sorted(vals)
            out[axis] = {"n": len(vals), "value": round(mean, 4), "sd": round(sd, 4),
                         "median": srt[len(srt) // 2], "min": srt[0], "max": srt[-1]}
        else:
            out[axis] = {"n": len(vals), "value": None}
    return out


def _endpoints(a, b):
    """Every Forge endpoint for the chosen bucket: both arms, the interval on
    the difference (Newcombe on a proportion, Welch on a mean, a bootstrap on
    the median where the sample is mostly zeros), and the MDE."""
    from manamap.pilot import candidates
    sa, sb = _arm_summary(a), _arm_summary(b)
    out = {}
    for axis, spec in candidates.FORGE_OBJECTIVE_AXES.items():
        ca, cb = sa[axis], sb[axis]
        ep = {"champion": ca, "branch": cb, "kind": spec["kind"],
              "lower_is_better": spec["lower_is_better"],
              "conditional": spec["conditional"]}
        if ca.get("value") is None or cb.get("value") is None:
            ep.update({"delta": None, "ci95": None, "excludes_zero": None,
                       "mde": None, "method": None,
                       "why": "no reading on one arm (absent, not zero)"})
            out[axis] = ep
            continue
        if spec["kind"] == "proportion":
            d = stats.diff_proportions(ca["k"], ca["n"], cb["k"], cb["n"])
            m = stats.mde_proportion(ca["k"] / ca["n"], ca["n"], cb["n"]) or {}
            ep.update({"delta": d["diff"], "ci95": d["ci95"],
                       "excludes_zero": d["excludes_zero"], "method": d["method"],
                       "mde": m.get("minimum_detectable_difference")})
        else:
            d = stats.diff_means_summary(ca["value"], ca["sd"], ca["n"],
                                         cb["value"], cb["sd"], cb["n"])
            se = math.sqrt(ca["sd"] ** 2 / ca["n"] + cb["sd"] ** 2 / cb["n"])
            ep.update({"delta": d["diff"], "ci95": d["ci95"],
                       "excludes_zero": d["excludes_zero"], "method": d["method"],
                       "mde": round(stats.Z_MDE_80 * se, 4)})
            va = (a.get("per_game") or {}).get(spec["per_game"]) or []
            vb = (b.get("per_game") or {}).get(spec["per_game"]) or []
            if va and vb and spec["per_game"] == "combat_damage_dealt_to_players":
                ep["median"] = stats.diff_medians(va, vb)
        out[axis] = ep
    return out


def champion_at(slug, pod):
    """What the champion reads at `pod` today — the anchor `deck-branch new`
    prints for a Forge objective. The bucket with the most decided games at
    that table under one harness; None when there is none."""
    from manamap.pilot.common import decklist_sha256
    try:
        live = decklist_sha256(slug)
    except FileNotFoundError:
        live = None
    dropped = []
    rows = _rows_for_public(f"data/decks/{slug}/sim/*.json", slug, live, dropped)
    at = {k: v for k, v in rows.items() if k[0] == pod and v["decided"]}
    if not at:
        return None
    key = max(at, key=lambda k: at[k]["decided"])
    row = at[key]
    summ = _arm_summary(row)
    eps = {}
    for axis, arm in summ.items():
        ep = {"value": arm.get("value"), "n": arm.get("n")}
        if arm.get("value") is not None and arm.get("n"):
            if "k" in arm:
                m = stats.mde_proportion(arm["value"], arm["n"], arm["n"]) or {}
                ep["mde"] = m.get("minimum_detectable_difference")
            else:
                ep["mde"] = round(stats.Z_MDE_80 * arm["sd"] * math.sqrt(2 / arm["n"]), 4)
        eps[axis] = ep
    return {"pod": pod, "label": _label(key), "games": row["decided"],
            "runs": row["runs"], "endpoints": eps}


def _row_difference(ca, cb):
    """The interval on one row's difference, and the z that ranks it.

    A MEAN cell carries `sd` and `n`, so Welch from summary statistics is the
    interval the per-game series would give (`stats.diff_means_summary`); a
    RATE cell carries only `rate` and `n`, so Newcombe on the reconstructed
    counts, as `diagnostic._diff_rate` does. The z is the Wald statistic — at
    10,000 games a cell the normal is the regime — used only to ORDER the
    family for Holm; the interval a reader sees is the better one.
    """
    na, nb = ca.get("n"), cb.get("n")
    if not na or not nb:
        return None, None
    ra, rb = ca["rate"], cb["rate"]
    if ca.get("sd") is not None and cb.get("sd") is not None:
        diff = stats.diff_means_summary(ra, ca["sd"], na, rb, cb["sd"], nb)
        se = math.sqrt(ca["sd"] ** 2 / na + cb["sd"] ** 2 / nb)
    else:
        diff = stats.diff_proportions(round(ra * na), na, round(rb * nb), nb)
        se = math.sqrt(max(ra * (1 - ra), 0) / na + max(rb * (1 - rb), 0) / nb)
    d = rb - ra
    z = (d / se) if se > 0 else (0.0 if d == 0 else float("inf"))
    return diff, z


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
        # THE PRICE, read from mana_analysis rather than recomputed here — the
        # rule the mana block already follows, and the one a test asserts.
        # It is on the page because NOTHING ELSE ON IT CAN SEE THIS: the
        # goldfish models no life, so a base that stops charging 3 a turn looks
        # identical to one that does not.
        "life": [a["lands"].get("life") or {}, b["lands"].get("life") or {}],
        "on_curve": {c: [a["on_curve_probability"]["with_rocks_and_dorks"].get(c),
                         b["on_curve_probability"]["with_rocks_and_dorks"].get(c)]
                     for c in "WUBRG"},
        "note": ("Deterministic, not sampled — Karsten's tables against the pip "
                 "distribution each list actually has. No interval, because "
                 "there is no sampling error to carry."),
    }


def _pool(slug, branch):
    """The corpus pool, or — with NO CORPUS (a fresh clone, CI's push job) — the
    deck's and the branch's own cards.json. `card_pool.load_pool` returns None there
    by contract and both readers below crashed on it (2026-10-03). Every name a
    branch diff can produce is in one of those two files, and they carry the type
    line and mana value the split and the rows need."""
    from manamap.pilot import card_pool
    pool = card_pool.load_pool()
    if pool is not None:
        return pool
    pool = {}
    for b in (branch, None):
        try:
            doc = load_deck_cards(slug, b)
        except FileNotFoundError:
            continue
        for c in (doc.get("cards") if isinstance(doc, dict) else doc) or []:
            pool.setdefault(c.get("name"), {"type_line": c.get("type_line"), "cmc": c.get("cmc")})
    return pool


def card_diff(slug, branch, bill=None):
    """THE MERGE-REQUEST DIFF: every card out, every card in, against the deck.

    WHY THIS IS NOT `changes()`. That function reads `branch.json`'s `staged`
    list, which only holds swaps made with `deck-branch stage`. A branch opened
    with `new --from <list>` sets its whole 99 at once and stages NOTHING, so a
    17-for-17 refactor rendered as "The change (0)" while the report underneath
    it measured all 34 cards. The page showed the incoming cards on the bill and
    the outgoing cards NOWHERE — a diff that omits the deletions.

    So the diff is derived from the two lists, the way git derives one, and the
    staged `why` is joined onto it where there is one. A card with no recorded
    reason is REPORTED AS UNEXPLAINED rather than left blank: the count is the
    honest measure of how much of this branch is argued for card by card.

    Lands and spells stay separated for the reason `changes()` already gives —
    a spell swap moves the sampled rows, a land swap moves only the
    deterministic mana block, and mixing them lets one borrow the other's credit.
    """
    pool = _pool(slug, branch)
    d = deck_branch.diff(slug, branch)
    meta = deck_branch.meta(slug, branch) or {}
    # PAIRING SURVIVES THE TWO COLUMNS. A staged swap is one decision about two
    # cards, and a diff laid out as two lists loses that: rendered naively the
    # `why` prints once on each side, so the reader sees the same sentence twice
    # and still cannot tell which removal paid for which addition. Each row
    # carries the name of its opposite number instead, and the reason is printed
    # once against the card that was ARGUED FOR.
    why_out, why_in, pair = {}, {}, {}
    for row in meta.get("staged") or []:
        if row.get("out"):
            why_out[row["out"]] = row.get("why")
        if row.get("in"):
            why_in[row["in"]] = row.get("why")
        if row.get("out") and row.get("in"):
            pair[row["out"]] = row["in"]
            pair[row["in"]] = row["out"]
    state = {r["name"]: r.get("state")
             for r in ((bill or {}).get("cards") or [])}

    # THROUGH `_named`, NOT RAW `_entries`. `d` above comes from `deck_branch.diff`,
    # which canonicalises both lists through the resolver's vocabulary; `_entries`
    # returns the literal decklist text. Those agree on 99 cards in 100 and disagree
    # on a DFC: decklist.txt carries "The Restoration of Eiganjo" (Scryfall rejects
    # the joined form on fetch) while cards.json — and therefore `d["out"]` — carries
    # "The Restoration of Eiganjo // Architect of Restoration". Cutting any DFC then
    # raised KeyError here and took the whole net-change down.
    base, cand = deck_branch._named(
        slug, branch,
        deck_branch._entries(deck_branch._list_text(slug)),
        deck_branch._entries(deck_branch._list_text(slug, branch)))

    def row(name, why_map, with_state):
        info = pool.get(name) or {}
        tl = info.get("type_line") or ""
        out = {"name": name,
               "cmc": int(float(info.get("cmc") or 0)),
               "type_line": tl,
               "kind": "land" if "Land" in tl else "spell",
               "why": why_map.get(name),
               "pair": pair.get(name)}
        # HOW MANY, because a name is not a card count. Every row the renderer
        # sums has to carry its own copies or the group totals silently assume
        # one apiece — which is true of 96 of these 99 and wrong about the ones
        # that matter.
        out["copies"] = (cand if with_state else base).get(name, 1)
        if with_state:
            # Where the card physically is, straight off the bill — so the diff
            # and the bill can never disagree about what has to be bought.
            out["state"] = state.get(name)
        return out

    outs = [row(n, why_out, False) for n in d["out"]]
    ins = [row(n, why_in, True) for n in d["add"]]

    # A COPY CUT IS A CARD CUT, and a name-level diff cannot see it. Cutting one
    # of two Plains and two of four Swamps removes three cards and no NAMES, so
    # the panel read "18 out, 21 in" for a change that was 21 for 21 — three
    # removals invisible on the page. That is the same defect as the one this
    # function was written for (a diff that omits its deletions), one scale down,
    # and it is exactly the trap `common.expand_copies` exists for.
    changed = []
    for name in sorted(set(base) & set(cand)):
        if base[name] != cand[name]:
            info = pool.get(name) or {}
            tl = info.get("type_line") or ""
            changed.append({"name": name, "from": base[name], "to": cand[name],
                            "delta": cand[name] - base[name],
                            "kind": "land" if "Land" in tl else "spell",
                            "cmc": int(float(info.get("cmc") or 0))})
    copies_out = sum(base[r["name"]] for r in outs) + sum(
        -c["delta"] for c in changed if c["delta"] < 0)
    copies_in = sum(cand[r["name"]] for r in ins) + sum(
        c["delta"] for c in changed if c["delta"] > 0)
    key = lambda r: (r["kind"] != "spell", r["cmc"], r["name"])
    outs.sort(key=key)
    ins.sort(key=key)
    return {
        "out": outs, "in": ins, "changed": changed,
        "counts": {
            # COPIES first, because that is what a pilot sleeves and what makes
            # the two sides balance. The name counts stay beside them so the
            # difference between the two is legible rather than a discrepancy.
            "out_copies": copies_out, "in_copies": copies_in,
            "out": len(outs), "in": len(ins),
            "spells_out": sum(1 for r in outs if r["kind"] == "spell"),
            "spells_in": sum(1 for r in ins if r["kind"] == "spell"),
            "lands_out": sum(1 for r in outs if r["kind"] == "land"),
            "lands_in": sum(1 for r in ins if r["kind"] == "land"),
        },
        "size": d["size"], "base_size": d["base_size"],
        "names": d["names"], "base_names": d["base_names"],
        # THE HONEST NUMBER. How much of this branch nobody wrote a reason for,
        # card by card. A branch opened from a list starts at 100%.
        "unexplained": {"out": sum(1 for r in outs if not r["why"]),
                        "in": sum(1 for r in ins if not r["why"])},
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
    meta = deck_branch.meta(slug, branch) or {}
    pool = _pool(slug, branch)

    def is_land(name):
        return "Land" in ((pool.get(name) or {}).get("type_line") or "")

    # THE NET DIFF, NOT THE STAGING LOG — and reading the log was a real defect.
    #
    # `staged` is every swap ever staged on this branch, superseded ones
    # included. goblin-storm/zada-v1 staged 29 and its NET change is 16: nine
    # cards were staged in and later staged back out, and this function
    # published all nine as ADDS. Three names appeared on BOTH sides at once.
    # The page's own header said "16 out and 16 in" from `deck_branch.diff`
    # while the list below it showed 29 pairs — a reader following it would
    # have bought cards the branch does not run.
    #
    # ONE PREDICATE, ONE HOME: `deck_branch.diff` already answers "what actually
    # changed", counts copies rather than names, and is what the header prints.
    # This reads it instead of keeping a second, wronger answer.
    #
    # The `why` is still the staged one — written when the swap was made, before
    # any figure in this report existed, so it cannot have been fitted to them.
    # A card whose PARTNER was superseded keeps its own reason and loses only the
    # arrow, because the pairing is the part that stopped being true.
    # A MERGED BRANCH IS A RECORD, NOT A SHOPPING LIST. Its swaps are already in
    # the deck, so the diff against the current list is empty BY DEFINITION —
    # and reporting nothing would erase what the branch did. Three merged
    # branches (gishath@mana-v1, ur-dragon@final-v2, heliod@splendor-v2) would
    # have gone blank on the first version of this fix. For those the staging
    # log IS the change; for an open branch it is a superset of it.
    merged = bool((meta or {}).get("merged"))
    d = deck_branch.diff(slug, branch)
    if merged and not (d.get("add") or d.get("out")):
        staged = meta.get("staged") or []
        lands, spells = [], []
        for row in staged:
            entry = {"out": row.get("out"), "in": row.get("in"),
                     "why": row.get("why"), "at": row.get("at")}
            (lands if is_land(entry["in"]) or is_land(entry["out"])
             else spells).append(entry)
        rows = lands + spells
        return {"spells": spells, "lands": lands,
                "count": len(rows),
                "counts": {"in": sum(1 for r in rows if r["in"]),
                           "out": sum(1 for r in rows if r["out"]),
                           "rows": len(rows)},
                "staged_count": len(staged), "merged": True,
                "opened": meta.get("opened"), "why": meta.get("why")}
    net_in = list(d.get("add") or [])
    net_out = list(d.get("out") or [])
    why_in, why_out = {}, {}
    for row in meta.get("staged") or []:
        if row.get("in"):
            why_in.setdefault(row["in"], row)
        if row.get("out"):
            why_out.setdefault(row["out"], row)

    lands, spells, paired_out = [], [], set()
    for name in net_in:
        row = why_in.get(name) or {}
        partner = row.get("out")
        if partner not in net_out:          # the pair was superseded
            partner = None
        else:
            paired_out.add(partner)
        entry = {"out": partner, "in": name,
                 "why": row.get("why"), "at": row.get("at")}
        (lands if is_land(name) or (partner and is_land(partner))
         else spells).append(entry)
    for name in net_out:
        if name in paired_out:
            continue
        row = why_out.get(name) or {}
        entry = {"out": name, "in": None,
                 "why": row.get("why"), "at": row.get("at")}
        (lands if is_land(name) else spells).append(entry)
    # THE HEADER IS NOT THE ROW COUNT. A row need not be a pair, so
    # `len(rows)` overstates what is coming in and understates nothing — it
    # read "22 swap(s)" on a branch bringing in 16 cards, six of the rows
    # being cuts whose partner had been superseded. `in` and `out` are the
    # figures a spending decision rests on; `rows` and `staged` are how the
    # list got here, and all four are now printed with their own names.
    return {"spells": spells, "lands": lands,
            "count": len(lands) + len(spells),
            "counts": {"in": len(net_in), "out": len(net_out),
                       "rows": len(lands) + len(spells)},
            "staged_count": len(meta.get("staged") or []),
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
    # NOT A BLANKET CLAIM ANY MORE, and it was one for a year after it stopped
    # being true. `model_draw` prices ETB draw, spell draw, recurring draw, cast
    # draw, X spells, wheels, activated draw and draw DOUBLERS — a deck that has
    # opted in has most of its card advantage measured. Left as a flat sentence
    # it told a reader that a measured +0.89 extra cards had not been measured,
    # which is a liar in the more dangerous direction: it invites cutting a card
    # the figures already credited. `_draw_is_blind` decides per CARD.
    "draw": "this card's draw is not one of the shapes the model reads — see "
            "`meta.draw_not_modelled` for what it does instead",
    "recursion": "nothing dies in a goldfish, so recursion has no target",
    "stax": "there is no opponent to tax",
    "hate": "there is no opponent to hate out",
}


def _draw_is_blind(slug, branch, name):
    """Is THIS card's draw outside what the model prices?

    A ROLE IS NOT AN ANSWER. `card_roles.json` says "this card draws"; whether
    the model can READ that draw is a different question, and `draw_profile`
    already answers it — it sets `unmodelled` to the card's own name when the
    draw is through a channel there is no event for, and leaves it None when the
    draw is priced. Asking the profile rather than the role is what stops the
    report claiming a measured figure was never measured.

    Blind if the deck never opted into `model_draw` at all, since then every
    draw really is one a turn.
    """
    from manamap.pilot.common import load_deck_cards
    from manamap.pilot.goldfish_profiles import draw_profile

    try:
        targets = json.loads(deck_file_or_none(slug, branch).read_text())
    except Exception:                                # pragma: no cover - defensive
        targets = {}
    if not targets.get("model_draw"):
        return True
    try:
        doc = load_deck_cards(slug, branch)
    except Exception:                                # pragma: no cover - defensive
        return True
    for card in doc.get("cards") or []:
        if card.get("name") == name:
            return bool(draw_profile(card)["unmodelled"])
    # Not in this list — it is the card leaving, so read it from the champion.
    try:
        for card in load_deck_cards(slug).get("cards") or []:
            if card.get("name") == name:
                return bool(draw_profile(card)["unmodelled"])
    except Exception:                                # pragma: no cover - defensive
        pass
    return True


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
                if head not in BLIND:
                    continue
                if head == "draw" and not _draw_is_blind(slug, branch, name):
                    continue
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
            # EITHER SIDE MAY BE ABSENT since `changes()` reports the net diff:
            # a card whose partner was superseded is a row with one side only.
            "cards": sorted({e["in"] for e in change_doc["lands"] if e["in"]}
                            | {e["out"] for e in change_doc["lands"] if e["out"]}),
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


def _seat_sha(doc, want):
    """The list THIS run actually played for `want`, or None on a record that
    predates the stamp.

    `doc["seats"]` is the run's own manifest and carries the sha; `analysis.seats`
    carries the outcomes and is keyed differently — `goblin-storm-zada-v1` there
    against `goblin-storm@zada-v1` and `mm-goblin-storm-zada-v1` in the manifest.
    `want` is the analysis key, so both manifest spellings are normalised to it
    rather than a caller being asked to know which one a record used.
    """
    for seat in (doc.get("seats") or []):
        names = {(seat.get("forge_name") or "").removeprefix("mm-"),
                 (seat.get("slug") or "").replace("@", "-")}
        if want in names - {""}:
            return seat.get("decklist_sha256")
    return None


def _label(key):
    """A bucket's name for a reader: the table, and every harness axis that is
    not the plain one. `('standard-v3', '', 'Default') -> 'standard-v3'`."""
    pod, ov, prof, tl = (key + ("",))[:4]
    bits = []
    if ov:
        bits.append(f"overrides {ov[:12]}")
    if prof and prof != "Default":
        bits.append(f"ours {prof}")
    if tl:
        bits.append(f"patches {tl[:12]}")
    return f"{pod} ({', '.join(bits)})" if bits else pod


def forge(slug, branch, pod=None):
    """The real table, if it has been played. POOLED WITHIN ONE POD ONLY, over
    DECIDED games only.

    Two things this used to get wrong, found 2026-09-11 on edgar-vampires/fear-v1:
    the champion arm pooled EVERY record under `sim/` — 840 games across five
    tables (vito-era, standard, standard-v2, playgroup, standard-v3) — against a
    branch played at standard-v3 alone, while the docstring said "within one pod
    only"; and both arms divided by `analysis.games`, so thirteen clock-outs
    counted as losses and the block's 0.20 disagreed with the record's own 0.296.
    A pod's null is a property of the table with the subject in it (CLAUDE.md), so
    a champion rate from another table is not this branch's control. The pod is
    named in the block so a reader can see which table decided it.
    """
    rows_for = _rows_for_public
    return _forge(slug, branch, pod, rows_for)


def _rows_for_public(pattern, want, live_sha, dropped, strict=True):
        """A RUN DESCRIBES THE LIST IT PLAYED, NOT THE LIST ON DISK.

        MEASURED, on goblin-storm/zada-v1: both branch records carried
        `seats[].decklist_sha256 = 57725742` — the branch's fourth commit — while
        the list on disk was `e01b366c`, its seventh. Eight cards had come in and
        nine had gone out since, including the largest single gain the branch
        claimed. The block reported those 120 games as the branch's rate and
        nothing said otherwise, because nothing read the sha the record puts
        right there beside the seat.

        A record made before the stamp existed carries None and is KEPT — it is
        older evidence, not wrong evidence, and dropping it would silently empty
        the block on every historical run.
        """
        by_pod = {}
        for path in sorted(glob.glob(pattern)):
            if "logs" in path:
                continue
            doc = json.load(open(path))
            a = doc.get("analysis") or {}
            pod = doc.get("pod")
            pod = (pod.get("name") if isinstance(pod, dict) else pod) or "unnamed"
            seat = (a.get("seats") or {}).get(want) or {}
            if seat.get("wins") is None:
                continue
            # A RUN ALSO DESCRIBES THE HARNESS IT WAS PLAYED UNDER. `data/
            # forge_overrides/` changes what the AI may TARGET, and on a deck
            # whose engine is "target your own commander" that is the difference
            # between 1/73 and 9/82 on one unchanged list. Pooling the two gave
            # 10/155 = 0.065 — a rate describing neither deck. The fingerprint has
            # been written into every record since the overrides shipped and
            # NOTHING READ IT, which is the same defect as a record describing a
            # list it never played, one layer over and shipped the same day.
            ov = (doc.get("card_overrides") or {}).get("sha") or ""
            # AND OUR SEAT'S PROFILE, because a profile is a harness too. The key was
            # `(pod, ov)` for eleven hours and it pooled the branch's Default-override
            # run (9/82) with its Experimental-override run (3/68) into `12/150` — the
            # same defect as the 10/155 it was written to fix, one axis over. That the
            # profile changes the instrument is MEASURED: on one list, same seed, same
            # overrides, Default -> Experimental took clock-outs from 17 to 32 of 100
            # with an interval excluding zero. `profiles` is the `-a` list or null;
            # null is every seat on Default, which is what a pre-profile record means.
            prof = ((doc.get("profiles") or ["Default"])[0]) or "Default"
            # THE PATCH SET, when the record carries one — `-tl` in the id. Observational
            # while it held only the log formatter; since 2026-09-30 it may hold an `ai`
            # class (MillAi's SacOutlet), which changes play, so it is an axis like `ov`.
            tl = (doc.get("telemetry") or {}).get("sha") or ""
            ran = _seat_sha(doc, want)
            if ran and live_sha and ran != live_sha:
                dropped.append({"run": doc.get("run_id") or path.split("/")[-1],
                                "pod": pod, "played": ran[:12],
                                "games": a.get("games") or 0})
                if strict:
                    continue
            row = by_pod.setdefault((pod, ov, prof, tl), {"wins": 0, "decided": 0,
                                                 "games": 0, "runs": 0,
                                                 "by_route": {},
                                                 "per_game": {n: [] for n in _PER_GAME},
                                                 "resolved": 0, "resolved_games": 0,
                                                 "run_ids": []})
            row["wins"] += seat["wins"]
            # A clock-out has NO winner and is excluded from the rate — the same
            # definition the record's own `win_rate` uses.
            row["decided"] += a.get("decided", a.get("games") or 0)
            row["games"] += a.get("games") or 0
            row["runs"] += 1
            row["run_ids"].append(doc.get("run_id") or path.split("/")[-1])
            # THE OTHER ENDPOINTS a branch may aim at, pooled the same way as the
            # wins: per-game values for the means (read with `experiment`'s own
            # readers, so the two commands cannot disagree about a figure) and
            # the commander's games-resolved count. Absent on a record that
            # predates the block, never zero.
            ca = seat.get("commander_access") or {}
            if ca.get("games_resolved") is not None:
                row["resolved"] += ca["games_resolved"]
                row["resolved_games"] += a.get("games") or 0
            per_seat_key = want
            for g in (doc.get("games") or []):
                if g.get("winner") and want.split("@")[0] in str(g["winner"]):
                    k = g.get("won_by") or "unstated"
                    row["by_route"][k] = row["by_route"].get(k, 0) + 1
                ps = (g.get("per_seat") or {}).get(per_seat_key)
                if ps is None:
                    continue
                for name, fn in _PER_GAME.items():
                    v = fn(ps)
                    if v is not None:
                        row["per_game"][name].append(v)
        return by_pod


def _cast_proofs_from_runs(slug, branch, run_ids):
    """What the branch's pooled arms say about the adds: the gate's block where a record
    carries one, and the post-hoc read — an add `held_while_castable` or never cast in an
    arm — where it does not. Returns None when the branch has no pooled runs."""
    if not run_ids or not branch:
        return None
    from manamap.sim import cast_check as _cc, engine_casts as _ec, forge as _forge_mod
    try:
        adds = set(_cc.adds(slug, branch))
    except Exception:                              # noqa: BLE001
        adds = set()
    held, late, gated, anyway = set(), set(), set(), False
    try:
        runs = _forge_mod.list_runs(f"{slug}@{branch}")
    except (FileNotFoundError, SystemExit):        # a synthetic slug has no seat on disk
        return None
    for rec in runs:
        if rec.get("run_id") not in run_ids:
            continue
        cp = rec.get("cast_proofs")
        if cp:
            gated.add(rec["run_id"]); held |= set(cp.get("held") or []); late |= set(cp.get("late") or [])
            anyway = anyway or bool(cp.get("anyway"))
        by = ((rec.get("engine_casts") or {}).get("by_card") or {})
        for name in adds:
            row = by.get(name) or by.get(name.split(" // ")[0]) or {}
            if row.get("in_hand_games") and not row.get("cast") and not row.get("activated"):
                held.add(name)
            elif row.get("castable_uncast", 0) >= 2 * max(1, row.get("in_hand_games") or 1) and \
                    (row.get("cast", 0) + row.get("activated", 0)) * 2 < (row.get("in_hand_games") or 0):
                late.add(name)
    return {"adds": sorted(adds), "held": sorted(held), "late": sorted(late - held),
            "gated_runs": sorted(gated), "anyway": anyway,
            "floor": bool(held or late),
            "reads_as": ("every branch figure on this page is a FLOOR: " + ", ".join(sorted(held)) + " never played"
                         + (f"; {', '.join(sorted(late - held))} cast late" if (late - held) else "")
                         if (held or late) else "every add the branch stages was played in its arms")}


def _forge(slug, branch, pod, rows_for):
    from manamap.pilot.common import decklist_sha256
    # FLAG, DO NOT SUPPRESS — and the reason is the ONE COSMETIC EDIT in the
    # sweep. `decklist_sha256` is over the file's bytes, so dropping the set code
    # from `Gifted Aetherborn (AER) 61` changes it while the deck is identical:
    # edgar-vampires' list did that in the same commit as two real swaps. A gate
    # that silences a run on a whitespace change, whose only remedy is a
    # multi-hour Forge batch, destroys evidence to prevent a misreading — so the
    # rate stays and the mismatch is stated beside it. `engine_casts` IS strict,
    # because "held and never cast" names particular cards and is simply false
    # about a list that did not contain them.
    def live_sha(b):
        # No deck directory (a synthetic slug, a deck not yet checked in) means
        # nothing to compare against, so the gate is simply off for that arm.
        try:
            return decklist_sha256(slug, b)
        except FileNotFoundError:
            return None

    cham_sha = live_sha(None)
    br_sha = live_sha(branch)
    br_seat = deck_branch_seat(slug, branch)
    # THE VERSION BESIDE THE SHA. A mismatch printed as `57725742 on disk` gave a
    # reader nothing to place; "champion runs describe V4 (deck is V5)" is what
    # they can act on. Versions come from git; a deck outside a repo, or a sha
    # that matches no committed list, simply has none — never "V0".
    by_version = _versions_by_sha(slug)
    cham_glob = f"data/decks/{slug}/sim/*.json"
    br_glob = f"data/decks/{slug}/branches/{branch}/sim/*.json"
    superseded = {"champion": [], "branch": []}
    champ = rows_for(cham_glob, slug, cham_sha, superseded["champion"])
    br = rows_for(br_glob, br_seat, br_sha, superseded["branch"])
    # An arm with nothing left after the gate falls back to every run it has,
    # and the block says so rather than going blank.
    mismatch = {}
    if not champ and superseded["champion"]:
        champ = rows_for(cham_glob, slug, cham_sha, [], strict=False)
        mismatch["champion"] = superseded["champion"]
    if not br and superseded["branch"]:
        br = rows_for(br_glob, br_seat, br_sha, [], strict=False)
        mismatch["branch"] = superseded["branch"]
    if not (champ and br):
        # A BRANCH CAN BE PUT AT A REAL TABLE, and the seat grammar is how.
        # `simulate` takes no `--branch` FLAG, which is what made this look
        # unreachable — but `sim/forge.py` splits a seat on `@`, so
        # `simulate edgar-vampires@drain-engine --vs …` resolves the branch's
        # own decklist and files the run under `branches/<name>/sim/`, which is
        # exactly the path this function reads.
        where = f"{slug}@{branch}" if (branch and champ) else slug
        return {"available": False,
                "superseded": {k: v for k, v in superseded.items() if v},
                "why": (f"no Forge run on {'the branch' if champ else 'the deck'}"
                        f" — `manamap pilot simulate {where} --vs <pod>` puts "
                        f"it at a table and writes where this reads")}
    common = [k for k in br if k in champ and champ[k]["decided"]
              and br[k]["decided"]]
    # THE OBJECTIVE'S TABLE DECIDES when there is one. A Forge objective names
    # its pod, and grading it at whichever table happened to hold the most
    # branch games would be grading it against a different null.
    if pod:
        at_pod = [k for k in common if k[0] == pod]
        if not at_pod:
            return {"available": False,
                    "superseded": {k: v for k, v in superseded.items() if v},
                    "why": (f"the objective is measured at {pod} and "
                            f"{'neither arm has' if not common else 'the arms have no common harness with'} "
                            f"a run there — `manamap pilot simulate "
                            f"{slug} --pod {pod} --games N` and the same for "
                            f"`{slug}@{branch}`")}
        common = at_pod
    if not common:
        return {"available": False,
                "superseded": {k: v for k, v in superseded.items() if v},
                "why": (f"the branch sat at {sorted(_label(k) for k in br)} "
                        f"and the champion at "
                        f"{sorted(_label(k) for k in champ)} — no table in "
                        f"common under the same harness, and a rate from "
                        f"another table is not this branch's control. Run "
                        f"`manamap pilot simulate {slug} --pod {sorted(br)[0]}`")}
    # The table with the most branch games decides; the others are named.
    key = max(common, key=lambda k: br[k]["decided"])
    pod, ovsha, prof, tl = (key + ("",))[:4]
    a, b = champ[key], br[key]
    a_w, a_n, b_w, b_n = a["wins"], a["decided"], b["wins"], b["decided"]
    d = stats.diff_proportions(a_w, a_n, b_w, b_n)
    m = stats.mde_proportion(a_w / a_n, a_n, b_n) or {}
    null, null_why = _null_block(pod)
    return {"available": True,
            "pod": pod,
            # WHICH ADDS THE AI PLAYED IN THE BRANCH'S OWN ARMS (2026-10-01): the gate's
            # block on each branch record, and `held_while_castable` read post hoc from
            # the same records — so an arm run before the gate existed is labelled too.
            # A held add makes every branch figure here a FLOOR, and the page says so.
            "cast_proofs": _cast_proofs_from_runs(slug, branch, b.get("run_ids") or []),
            # THE NULL, STORED BESIDE THE FIGURE IT SCALES. It was printed by
            # `_print_real_table` and absent from the artifact, so the JSON and
            # the terminal were not the same document.
            "null": null, **({"null_why": null_why} if null is None else {}),
            # EVERY ENDPOINT A BRANCH MAY AIM AT, each with the interval on the
            # difference and its own MDE; `forge.win_rate` here is the same
            # figure as the top-level delta / ci95 / mde.
            "endpoints": _endpoints(a, b),
            "run_ids": {"champion": a["run_ids"], "branch": b["run_ids"]},
            # WHICH HARNESS DECIDED IT. Absent means no card-script overrides
            # were loaded; a sha means both arms were played under exactly that
            # directory, because they are bucketed together or not at all.
            "card_overrides": ovsha or None,
            # And the profile that flew our seat in this bucket — absent when Default,
            # so a plain record's block is unchanged.
            **({"ai_profile": prof} if prof and prof != "Default" else {}),
            # WHAT THIS BLOCK LEFT OUT AND WHY. A run excluded in silence is
            # indistinguishable from a run that was never made.
            "superseded": {k: v for k, v in superseded.items() if v},
            # AND WHERE IT HAD NOTHING ELSE TO USE. An arm listed here is being
            # reported from runs made on a DIFFERENT list — the figure describes
            # that list, not this one.
            "list_mismatch": {
                arm: {"played": sorted({r["played"] for r in rows}),
                      "games": sum(r["games"] for r in rows),
                      "on_disk": (br_sha if arm == "branch" else cham_sha)[:12],
                      # Champion versions resolve from git; a branch's commits are
                      # not versions, so its entry names none rather than guessing.
                      **({"played_versions": sorted({
                              f"V{by_version[p]}" for p in {r["played"] for r in rows}
                              if p in by_version}),
                          "on_disk_version": (f"V{by_version[cham_sha[:12]]}"
                                              if cham_sha and cham_sha[:12] in by_version
                                              else None)}
                         if arm == "champion" else {}),
                      "reads_as": (
                          f"every Forge run on the {arm} was made with a "
                          f"different list, so this rate describes that list. "
                          f"A card swap and a cosmetic edit to decklist.txt both "
                          f"land here. Re-run: `manamap pilot simulate "
                          f"{slug + '@' + branch if arm == 'branch' else slug} "
                          f"--pod <name> --games N`")}
                for arm, rows in mismatch.items()} or None,
            "basis": ("wins over DECIDED games (clock-outs have no winner and are "
                      "excluded), each arm pooled across every run of it at this "
                      "one table"),
            "other_tables": {_label(k): {
                                 "champion_runs": champ.get(k, {}).get("runs", 0),
                                 "branch_runs": br.get(k, {}).get("runs", 0)}
                             for k in sorted(set(champ) | set(br)) if k != key},
            "champion": {"wins": a_w, "games": a_n, "all_games": a["games"],
                         "runs": a["runs"], "rate": round(a_w / a_n, 4),
                         "won_by": a["by_route"]},
            "branch": {"wins": b_w, "games": b_n, "all_games": b["games"],
                       "runs": b["runs"], "rate": round(b_w / b_n, 4),
                       "won_by": b["by_route"]},
            "delta": round(b_w / b_n - a_w / a_n, 4),
            "ci95": d["ci95"], "excludes_zero": d.get("excludes_zero"),
            "mde": m.get("minimum_detectable_difference"),
            "caveat": ("Forge's AI is a weak pilot; the comparison is fair only "
                       "because both seats were played at a comparable rate — see "
                       "`sim/pilot_quality`."),
            # WHETHER EITHER ARM'S ENGINE WAS EVER CAST. A record made before
            # 2026-09-10 carries no measurement and reads None here.
            "engine_casts": _engine_casts_caveat(slug, branch)}


def _engine_casts_caveat(slug, branch):
    """The never-cast list on the latest CURRENT-LIST record of each arm, or None.

    "HELD AND NEVER CAST" AND "NOT IN THE DECK" ARE DIFFERENT FACTS, and this
    read the latest record against the list on disk, so it could not tell them
    apart. On goblin-storm/zada-v1 it named Hanweir Garrison, Legion Warboss,
    Assault Strobe, Reckless Ransacking and Great Train Heist as held and passed
    over. None of the five was in the list those games were played with: they
    were added afterwards. The strongest claim this function makes — that the AI
    saw the deck's engine and declined it — was being made about cards the AI had
    never been dealt.
    """
    from manamap.pilot.common import decklist_sha256
    from manamap.sim import engine_casts as ec
    out = {}
    for arm, pattern, b in (("champion", f"data/decks/{slug}/sim/*.json", None),
                            ("branch", f"data/decks/{slug}/branches/{branch}/sim/*.json", branch)):
        want = deck_branch_seat(slug, b) if b else slug
        live = decklist_sha256(slug, b)
        paths = [p for p in sorted(glob.glob(pattern)) if "logs" not in p]
        # A record with no stamp predates it and is still readable; one stamped
        # with another list is not.
        current = [p for p in paths
                   if (_seat_sha(json.load(open(p)), want) or live) == live]
        if not current:
            out[arm] = None
            continue
        rec = json.load(open(current[-1]))
        try:
            names = ec.nonland_names(load_deck_cards(slug, b))
        except Exception:
            names = None
        q = ec.from_record(rec, names, ec.engine_set(slug, b))
        # THE CLAIM NAMES THE RUN IT IS MADE ABOUT. "Held and never cast" is the
        # strongest reading this report offers, and until it said which games it
        # came from there was no way to check it against them.
        out[arm] = None if not q else {
            "covered": q["covered"],
            "never_cast": [r["card"] for r in q["never_cast"]][:8],
            "run": rec.get("run_id"),
            "played": (_seat_sha(rec, want) or live)}
    return out


def deck_branch_seat(slug, branch):
    from manamap.sim.forge import deck_meta_name
    return deck_meta_name(f"{slug}@{branch}")


def measured_list_is_current(slug, branch):
    """`(ok, stamp, live)` — is `cards.json` built from the list on disk?

    THE REPORT MEASURES `cards.json`; THE LIST OF RECORD IS `decklist.txt`.
    `diagnostic.run` hands the goldfish `load_deck_cards(slug, branch)`, so a
    swap staged and committed into `decklist.txt` without a `fetch-deck
    --branch` is invisible: the arms are measured on the PREVIOUS list, the
    report is written, and its `decklist_sha256` is the previous list's.

    That happened on `heliod/splendor-v1` on 2026-09-12 (#45). Nothing refused
    it — `validate-net-change` passed the file, and the only thing that noticed
    was `deck-branch propose`, whose message said to re-run `net-change`, which
    reproduces the same stale file. The sequence that actually repairs it is
    three commands, and this is the one place that can say so.

    This is the "a branched write needs a branched read" class one level up: the
    write is correctly branched, and its INPUTS are older than the list.
    """
    from manamap.pilot.common import deck_dir, load_json

    root = deck_dir(slug, branch)
    live = hashlib.sha256((root / "decklist.txt").read_bytes()).hexdigest()
    stamp = (load_json(root / "cards.json") or {}).get("decklist_sha256")
    return (stamp == live), stamp, live


def _refuse_a_stale_measurement(slug, branch):
    """Refuse before measuring, naming all three commands in order."""
    ok, stamp, live = measured_list_is_current(slug, branch)
    if ok:
        return
    where = f"{slug} --branch {branch}"
    raise SystemExit(
        f"{slug}/{branch}: cards.json was built from a different list than "
        f"decklist.txt.\n"
        f"  cards.json   {(stamp or 'absent')[:12]}\n"
        f"  decklist.txt {live[:12]}\n"
        f"Measuring now would report figures for the PREVIOUS list and stamp "
        f"them with its sha. Re-derive first, in this order:\n"
        f"  manamap pilot fetch-deck {where}\n"
        f"  manamap pilot goldfish {where}\n"
        f"  manamap pilot net-change {where} --write")


def _death_limit(slug, branch):
    """The death limitation, or the rate that replaced it.

    THIS SENTENCE WENT STALE THE DAY THE CHANNEL SHIPPED. It read "Death-
    triggered DRAIN is not modelled at all — nothing dies in this simulation",
    and on 2026-09-27 that stopped being true: `model_deaths` carries a per-turn
    rate and `death_damage` reads the "deals N damage" idiom `death_drain` never
    did. A report that states a limitation the model no longer has is worse than
    one that states none — the reader discounts a figure that is now sound.
    #
    So the sentence is DERIVED from the deck's own declaration rather than
    asserted. A deck that has not declared a death rate gets the warning it has
    earned; one that has gets the rate and its source, which is the thing a
    reader actually needs in order to judge the figure.
    """
    from manamap.pilot.common import deck_file, load_json

    decl = load_json(deck_file(slug, "goldfish_targets.json", branch)) or {}
    deaths = decl.get("model_deaths") or None
    if not (deaths and decl.get("model_drain")):
        return ["Death-triggered drain and damage are NOT modelled on this deck: "
                "it declares no `model_deaths` rate, so nothing dies in this "
                "simulation and Blood Artist, Zulaport Cutthroat, Pashalik Mons "
                "and Bastion of Remembrance contribute nothing to any damage or "
                "kill row. A branch built on them is understated by however much "
                "that line is worth, and `simulate` against a real pod is the "
                "only place it can be measured."]
    return [f"Deaths ARE modelled here, at a MEASURED rate: "
            f"{deaths['own_per_turn']} of our creatures and "
            f"{deaths['opponent_per_turn']} of theirs per own turn, read off "
            f"{deaths['source']}. Both the life-loss idiom (`death_drain`) and "
            f"the damage idiom (`death_damage`) are counted. The rate is an "
            f"AVERAGE, so it cannot represent a turn where eight bodies die at "
            f"once — a sacrifice burst is still understated."]


def _paired(a, b, block, key, turn):
    """`(diff, z, mde)` from GAME-BY-GAME differences, or None.

    Needs both readings taken with `keep_games` on the same seed and game count,
    so game i of each list dealt the same shuffle (`goldfish.run` seeds every game
    on its own). The interval is then on the mean of the per-game differences — the
    noise the two lists SHARE cancels, which an unpaired interval at the same n
    cannot do. Games where either side has no value (a series that ended) drop out
    of both."""
    ga, gb = (a.get("_games") or {}), (b.get("_games") or {})
    k = f"{block}|{key}|{turn}"
    xa, xb = ga.get(k), gb.get(k)
    if not xa or not xb or len(xa) != len(xb):
        return None
    d = [y - x for x, y in zip(xa, xb) if x is not None and y is not None]
    if len(d) < 30 or len(d) < 0.95 * len(xa):
        return None
    n = len(d)
    mean = sum(d) / n
    var = sum((v - mean) ** 2 for v in d) / (n - 1)
    se = math.sqrt(var / n)
    half = stats.t_crit(n - 1) * se if se else 0.0
    # Decided on the ROUNDED bounds, the ones a reader sees: a bound of -0.00002
    # prints as -0.0, and `validate_net_change` rightly calls "[-0.02, -0.0]
    # excludes zero" a contradiction (gishath@lands-v1, 2026-10-04).
    lo, hi = round(mean - half, 4), round(mean + half, 4)
    diff = {"diff": round(mean, 4), "ci95": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "method": f"paired t interval on {n:,} game-by-game differences (same seed per game)"}
    z = (mean / se) if se > 0 else (0.0 if mean == 0 else float("inf"))
    return diff, z, round(2.8016 * se, 4)


def compare_readings(a, b):
    """The exploratory family: every ROW read on two diagnostic readings, each with
    the interval on its own difference, its MDE, Holm across the family, and a
    verdict that needs both. Shared by `build` (two lists on disk) and `try` (a
    list held in memory), so the two can never grade a row differently."""
    from manamap.pilot import diagnostic
    table = []
    pending = []
    for label, blk, key, turn, want in ROWS:
        ca, cb = _cell(a, blk, key, turn), _cell(b, blk, key, turn)
        if not (ca and cb):
            continue
        delta = round(cb["rate"] - ca["rate"], 4)
        good = (delta > 0) == (want > 0)
        spec = METRICS.get(label) or {}
        pr = _paired(a, b, blk, key, turn)
        if pr:
            diff, z, mde = pr
        else:
            mde = max(diagnostic.mde(ca), diagnostic.mde(cb))
            diff, z = _row_difference(ca, cb)
        row = {
            "measure": label, "champion": ca["rate"], "branch": cb["rate"],
            "delta": delta, "mde": round(mde, 4),
            # THE INTERVAL ON THE DIFFERENCE, per row — Welch from the cells'
            # {rate, sd, n} on a mean, Newcombe on a rate. The MDE said what the
            # run could see; this says what it saw.
            "ci95_diff": diff.get("ci95") if diff else None,
            "excludes_zero": diff.get("excludes_zero") if diff else None,
            "method": diff.get("method") if diff else None,
            "z": round(z, 3) if z is not None else None,
            "role": "exploratory",
            # THE DEFINITION TRAVELS WITH THE FIGURE. `deck.html` renders this
            # artifact and had no way to say what a row meant; a reader who has
            # to leave the page to find out guesses instead, and the guesses go
            # one way — a mean read as a rate, a clock read as a win rate.
            "what": spec.get("what"), "why_we_care": spec.get("why"),
            "unit": spec.get("unit"), "scale": spec.get("scale"),
            "better_is": "higher" if want > 0 else "lower"}
        pending.append((row, good, mde))
        table.append(row)
    for row, h in zip(table, stats.holm([r["z"] or 0.0 for r in table])):
        row["holm"] = h
    for row, good, mde in pending:
        # A verdict needs both: a difference the run could resolve (the MDE)
        # AND one that survives being one of twelve looks (Holm). `noise` keeps
        # its meaning — "unresolved", never "no change".
        clears = abs(row["delta"]) > mde and row["holm"]["significant"]
        row["verdict"] = ("better" if good else "worse") if clears else "noise"
        row["reads_as"] = reads_as(row)
    return table


def build(slug, branch, iterations=None, seed=None):
    from manamap import console
    from manamap.pilot import candidates, diagnostic, goldfish

    _refuse_a_stale_measurement(slug, branch)
    it = iterations or diagnostic.HARNESS["iterations"]
    sd = seed if seed is not None else diagnostic.HARNESS["seed"]
    # TWO 10,000-GAME RUNS, ~15 s, and until now it printed nothing at all until
    # both had finished. A command that is silent for fifteen seconds is
    # indistinguishable from one that has hung, and this is the one the pilot
    # runs on every branch.
    #
    # The bar counts ARMS, not games, because that is the honest unit: `run`
    # does not report progress inside itself, and a bar that crept while a
    # simulation was actually blocked would be worse than none —
    # `console.py`'s third rule, never fake a percentage.
    with console.task(f"Measuring {slug} vs {branch}", total=2, unit="arms") as bar:
        bar.state("champion")
        a = diagnostic.run(slug, iterations=it, seed=sd, quiet=True, keep_games=True)
        bar.advance(1, state="branch")
        # PAIRED: the branch in the champion's slots, one seed per game, so each
        # row's interval is on game-by-game differences (`_paired`).
        b = diagnostic.run(slug, branch=branch, iterations=it, seed=sd, quiet=True,
                           keep_games=True, align=True)
        bar.advance(1)

    # ONE PRIMARY, TWELVE EXPLORATORY. Twelve rows each given an independent
    # verdict at alpha = 0.05 is roughly one false "better" every two reports,
    # and `experiment.py` had already solved the identical problem by
    # pre-registering `win_rate` and calling the other ten descriptive. Here the
    # primary is THE OBJECTIVE (declared when the branch was opened, graded
    # below), and the rows are a FAMILY: each carries the interval on its own
    # difference, and a verdict needs BOTH the MDE and Holm's step-down
    # correction across the family. Measured before shipping: over the 42
    # tracked reports and their 438 rows, Holm flips ZERO verdicts — the MDE
    # rule (2.8016·se) was already within 2% of the Bonferroni-12 threshold
    # (2.865·se) at the top rank — so this changes what a verdict MEANS, not
    # which rows carry one today.
    table = compare_readings(a, b)

    doc_meta = deck_branch.meta(slug, branch) or {}
    objective = doc_meta.get("objective")
    change_doc = changes(slug, branch)
    bill_doc = deck_branch.source(slug, branch)
    change_doc["diff"] = card_diff(slug, branch, bill_doc)
    staged = len(doc_meta.get("staged") or [])
    grade = None
    forge_block = forge(slug, branch, pod=(objective or {}).get("pod"))
    if objective and objective["axis"] in candidates.FORGE_OBJECTIVE_AXES:
        # THE REAL TABLE AS THE PRIMARY. Graded on the branch's pooled reading
        # at the objective's pod, against the MDE at those game counts, and the
        # grade carries the interval on the champion-to-branch difference and
        # the table's null, so the page shows what the number was read against.
        ep = ((forge_block.get("endpoints") or {}).get(objective["axis"])
              if forge_block.get("available") else None)
        if ep and (ep.get("branch") or {}).get("value") is not None:
            grade = deck_branch.grade_objective(
                objective, ep["branch"]["value"], mde=ep.get("mde"),
                difference={"delta": ep.get("delta"), "ci95": ep.get("ci95"),
                            "excludes_zero": ep.get("excludes_zero"),
                            "method": ep.get("method"),
                            "n_a": (ep.get("champion") or {}).get("n"),
                            "n_b": (ep.get("branch") or {}).get("n")},
                null=forge_block.get("null"))
        else:
            grade = deck_branch.grade_objective(
                objective, None,
                why_unmeasured=(forge_block.get("why") if not forge_block.get("available")
                                else f"{objective['axis']} has no reading on one arm at "
                                     f"{objective['pod']} — "
                                     f"{(ep or {}).get('why') or 'absent, not zero'}"))
    elif objective:
        block, key, turn = candidates.OBJECTIVE_AXES.get(
            objective["axis"], (None, None, None))
        cell = _cell(b, block, key, turn) if block else None
        grade = deck_branch.grade_objective(
            objective, cell["rate"] if cell else None,
            mde=diagnostic.mde(cell) if cell else None)

    doc = {
        "slug": slug, "branch": branch,
        # A SEED WITHOUT A MODEL VERSION REPRODUCES NOTHING. `harness` recorded
        # the two reproducibility inputs the goldfish takes — game count and seed
        # — and left out the third, which is the model those games were played
        # under. Same seed, different model, different figures: goblin-storm's
        # damage @T10 read 27.73 and then 21.02 across one commit that touched no
        # decklist, because the model learned to see creatures dying.
        #
        # It matters here rather than in `goldfish_metrics.json` (which has
        # stamped it since the model_version work) because THIS is the report a
        # decision is taken on, and `deck_branch.propose` copies `harness` whole
        # into `accepted_on` — the only record of what the evidence said at the
        # moment the pilot said yes. Nine branches carry a decision taken before
        # the token-copy and death-damage channels existed, and `regen` has since
        # rewritten every figure beside them, so the artifact cannot say the
        # evidence was replaced underneath the decision. This is what lets it.
        "harness": {"iterations": it, "seed": sd,
                    "model_version": goldfish.model_version(),
                    # WHICH ADDS THE AI PLAYED, frozen with the decision (2026-10-01).
                    "cast_proofs": ((forge_block.get("cast_proofs") or None)
                                    if forge_block.get("available") else None)},
        "decklist_sha256": (b.get("decklist_sha256")),
        # THE DESIGN, STATED: which figure is the test and which are the looks.
        "design": {"primary": (objective or {}).get("axis"),
                   "alpha": 0.05,
                   "exploratory_rows": len(table),
                   "correction": "Holm step-down over the exploratory rows",
                   "rule": ("a row is better/worse only if its delta clears the "
                            "MDE AND its |z| survives Holm across the family; "
                            "the objective is graded on its own, uncorrected, "
                            "because it was declared before the measurement")},
        "objective": objective,
        "objective_grade": grade,
        "staged": staged,
        "changes": change_doc,
        "blind_spots": blind_spots(slug, branch, change_doc),
        "definitions": {"rows": METRICS,
                        "forge_endpoints": candidates.FORGE_OBJECTIVE_AXES,
                        "derived": [
            {"name": n, "what": w, "why": y} for n, w, y in DERIVED]},
        "table": table,
        "mana": mana(slug, branch),
        "forge": forge_block,
        "bill": bill_doc,
        "limits": [
            "The goldfish has no opponents and nothing blocks: its kill turn is a "
            "CLOCK, not a win rate, and it cannot see interaction, removal or any "
            "alternate win.",
            "Nothing here reads `goldfish_targets.json`'s `required` flags. The "
            "engine lift did, and was deleted for it: the declaration is "
            "authored, so the same hand set the target and read the verdict.",
            "Card advantage is measured only where the model can price it. "
            "Activated, X-based, sacrifice-gated and death-triggered draw are "
            "unmodelled and named per deck in `goldfish_metrics.json`; on a "
            "sacrifice-based list that is most of it.",
        ] + _death_limit(slug, branch),
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
    # 20,000 simulated games are about to be spent comparing two lists. If the
    # model cannot see a third of either one, say so first — every expensive
    # fidelity surprise on this bench was found after the run, not before it.
    try:
        from manamap.pilot import model_coverage

        for scope in (None, branch):
            line = model_coverage.headline(
                model_coverage.analyze(args.slug, scope))
            if line:
                print(f"  {'branch' if scope else 'deck  '}  {line}")
    except Exception:                              # noqa: BLE001 - never block
        pass
    # MEASURE TWICE, said here too: which of the branch's adds the Forge AI has been
    # PROVEN to play under the current harness. The gate is in `simulate`; this is the
    # same reading at the report, so a reader sees it before the figures.
    try:
        from manamap.sim import cast_check as _cc
        _cp = _cc.status(args.slug, branch, _cc.current_harness(args.slug, branch))
        n = sum(len(_cp[k]) for k in ("proven", "held", "late", "unproven"))
        if n:
            print(f"  branch  CAST PROOFS — {len(_cp['proven'])}/{n} adds PLAYED under the current harness"
                  + (f"; HELD: {', '.join(_cp['held'])}" if _cp["held"] else "")
                  + (f"; CAST-LATE: {', '.join(_cp['late'])}" if _cp["late"] else "")
                  + (f"; unproven: {', '.join(_cp['unproven'])}" if _cp["unproven"] else ""))
    except (Exception, SystemExit):                # noqa: BLE001 - never block; no Forge exits
        pass
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
    # THE REAL TABLE CAN SAY NO, ABOVE EVERY OTHER ROW OF THE RULE. The goldfish
    # has no blockers, and on copy-burst-v1 it said MERGE while Forge read the
    # branch at -0.061 against the champion. A Forge win-rate loss whose
    # interval EXCLUDES ZERO at the same pod and harness is the one figure here
    # with external validity, and a rule that files it as a footnote is the
    # August lesson repeating. A Forge result that spans zero changes nothing:
    # that is "cannot tell", and the goldfish's verdict stands on its own terms.
    f = doc.get("forge") or {}
    wr = (f.get("endpoints") or {}).get("forge.win_rate") or {}
    real_table_no = bool(f.get("available") and wr.get("excludes_zero")
                         and (wr.get("delta") or 0) < 0)
    if real_table_no:
        lo, hi = wr["ci95"]
        null = (f.get("null") or {}).get("rate")
        out = ("do not merge",
               f"The real table says no: at {f.get('pod')} the branch wins "
               f"{wr['delta']:+.3f} less than the deck, CI [{lo:+.3f}, {hi:+.3f}], "
               f"which excludes zero"
               + (f" (the table's null is {null:.3f})" if null else "")
               + (f". The goldfish objective read {state!r}; the goldfish has no "
                  f"blockers, and its verdict on a table is not evidence against "
                  f"one." if objective and state else "."))
    elif not objective:
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
            # `free` AND `apart` ARE THE BRANCH'S ANSWER, NOT A SECOND ONE HERE.
            # This block used to carry `FREE_TO_RAID = ("retired", "broken-down")`
            # — a fourth copy of a set `common.UNPLAYABLE_STATUSES` already held.
            # `deck_branch.source` now derives both and this reads them.
            homes = row.get("where") or []
            live = sorted({h["slug"] for h in homes if not h.get("apart")})
            (loose if row.get("free") else contested).append(
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


def _print_changes(doc):
    """The change section, apart from `_print` so a test can state the row mix
    that exposed the header bug without standing up a whole report."""
    ch = doc.get("changes") or {}
    if ch.get("count"):
        c = ch.get("counts") or {}
        n_in, n_out = c.get("in"), c.get("out")
        if n_in is None:                       # a report written before counts
            head = f"{ch['count']} row(s)"
        else:
            head = f"{n_in} in, {n_out} out"
            aside = [f"{n} {w}" for n, w in
                     ((c.get("rows"), "rows"), (ch.get("staged_count"), "staged"))
                     if n and n != n_in]
            if aside:
                head += "  (" + ", ".join(aside) + ")"
        print(f"\n  THE CHANGE   {head}"
              + (f", branch opened {ch['opened']}" if ch.get("opened") else ""))
        cuts = []
        for title, rows in (("spells", ch.get("spells") or []),
                            ("lands", ch.get("lands") or [])):
            # A ROW WITH NOTHING COMING IN IS A CUT, NOT A SWAP, and printed
            # among the swaps as `- Card + None` it reads as one.
            swaps = [r for r in rows if r["in"]]
            cuts += [r for r in rows if not r["in"]]
            if not swaps:
                continue
            print(f"\n    {title.upper()}  ({len(swaps)} in)")
            for r in swaps:
                print(f"      - {str(r['out'] or '')[:30]:32} + {str(r['in'])[:30]}"
                      if r["out"] else f"      {'':34} + {str(r['in'])[:30]}")
                if r.get("why"):
                    print(_wrap(r["why"], indent="          "))
        if cuts:
            print(f"\n    CUT, NOTHING IN ITS SLOT  ({len(cuts)})")
            # WHY A ROW HAS NO PARTNER DEPENDS ON HOW THE BRANCH WAS BUILT, and
            # asserting the staging reason on a branch with no staging log is
            # simply false. A branch opened with `new --from <list>` has never
            # staged a swap, so EVERY row is unpaired and there is no superseded
            # partner to point at — the pairing is unknown, not lost.
            print(_wrap("Their partner was staged back out later, so the slot "
                        "is paid for elsewhere in this list."
                        if ch.get("staged_count") else
                        "This branch was opened from a whole list rather than "
                        "staged swap by swap, so no row has a recorded partner "
                        "— the 20 cards above and the 20 below are the same "
                        "change, unpaired.", indent="      "))
            for r in cuts:
                print(f"      - {str(r['out'])[:30]}")
                if r.get("why"):
                    print(_wrap(r["why"], indent="          "))


def _versions_by_sha(slug):
    """{12-char decklist sha prefix: version number} for a deck, from git.
    Empty when the deck has no history (a synthetic slug, a clone with no git),
    which every caller treats as "unknown", not as V0."""
    try:
        from manamap.pilot import deck_versions
        vers = deck_versions.versions(slug)
    except Exception:                    # noqa: BLE001 — a label, never a gate
        return {}
    return {s[:12]: v["version"] for v in vers
            for s in (v.get("decklist_sha256s") or [])}


def _pod_null(pod):
    """The table's subject null, or None when it has none.

    ABSENT MEANS ABSENT: a table nothing has been measured against has no null,
    and a default here would be an invented figure standing exactly where a
    measured one belongs. Every failure mode — no pod name, no calibration file,
    an untracked table, a zero rate that would make a ratio meaningless — returns
    None and the caller prints nothing.
    """
    from manamap.sim import power
    return power.null_rate(pod)[0]


def _games_to_resolve(rate, delta):
    """Games per arm to resolve `delta` against `rate` at 80% power, with the
    hours for two arms — None when the delta is zero or beyond reach."""
    if not delta or rate is None:
        return None
    from manamap.sim import power
    need = stats.games_for_difference(rate, abs(delta))
    if not need:
        return None
    return {"games": need, "hours": 2 * need / power.GAMES_PER_MINUTE / 60,
            "per_minute": power.GAMES_PER_MINUTE}


def _print_real_table(doc):
    """The Forge section, apart from `_print` for the same reason as
    `_print_changes`: a test can then state the one shape that matters —
    an arm whose only runs were made on a list the deck no longer is."""
    f = doc["forge"]
    print("\n  THE REAL TABLE")
    # THE MISMATCH GOES FIRST, ABOVE THE RATE IT IS ABOUT. Printed underneath,
    # it reads as a footnote to a number the eye has already taken.
    for arm, m in (f.get("list_mismatch") or {}).items():
        print(f"    !! {arm.upper()} MEASURED ON A DIFFERENT LIST — "
              f"{m['games']} game(s) on {_and(m['played'])}, "
              f"{m['on_disk']} on disk")
        if m.get("played_versions") or m.get("on_disk_version"):
            print(f"       those runs describe {_and(m.get('played_versions') or ['an uncommitted list'])}"
                  f"; the deck is {m.get('on_disk_version') or 'an uncommitted list'}")
        print(_wrap(m["reads_as"], indent="       "))
    for arm, rows in (f.get("superseded") or {}).items():
        if (f.get("list_mismatch") or {}).get(arm):
            continue          # already stated, in stronger terms, just above
        print(f"    not counted ({arm}, superseded list): "
              f"{sum(r['games'] for r in rows)} game(s) on "
              f"{_and(sorted({r['played'] for r in rows}))}")
    if not f.get("available"):
        print(f"    {f['why']}")
    else:
        print(f"    at {f.get('pod', '?')} — {f.get('basis', '')}")
        _cpf = f.get("cast_proofs")
        if _cpf:
            print(f"    CAST PROOFS  {'FLOOR — ' if _cpf['floor'] else ''}{_cpf['reads_as']}")
        print(f"    champion {f['champion']['wins']}/{f['champion']['games']} "
              f"({f['champion']['rate']:.3f})   "
              f"branch {f['branch']['wins']}/{f['branch']['games']} "
              f"({f['branch']['rate']:.3f})")
        other = {k: v for k, v in (f.get("other_tables") or {}).items()
                 if v.get("champion_runs") or v.get("branch_runs")}
        if other:
            print("    not pooled (another table or harness): " + ", ".join(
                f"{k} ({v['champion_runs']} champion / {v['branch_runs']} branch run(s))"
                for k, v in other.items()))
            # THE ACTIONABLE CASE, called out rather than left in a list. A bucket
            # at THIS table under a different harness is not a foreign table whose
            # null does not apply — it is the same table, and the only reason it
            # cannot be compared is that the other arm has never been run under
            # those overrides. Saying which command fixes that is the difference
            # between a held-out run and a wasted one.
            here = {k: v for k, v in other.items()
                    if k.startswith(f"{f.get('pod')} (overrides")}
            for k, v in here.items():
                need = "champion" if not v["champion_runs"] else "branch"
                # FROM THE DOCUMENT, NOT A CLOSURE. `_print_real_table` takes only
                # `doc`; reaching for `slug`/`branch` here raised NameError and the
                # line silently did not print, which is how this was found.
                _slug = doc.get("slug")
                seat = _slug if need == "champion" else f"{_slug}@{doc.get('branch')}"
                # NAME THE AXIS THAT ACTUALLY DIFFERS. This said "never been played
                # under those card-script overrides" for every held-out row, and the
                # day the champion WAS played under them it was still saying so about
                # a row that differed only by profile. A reader was told the wrong
                # reason and the wrong remedy.
                is_prof = "ours " in k
                why = ("under that AI profile" if is_prof
                       else "under those card-script overrides")
                what = ("A profile changes how the AI plays — measured: Default -> "
                        "Experimental took clock-outs from 17 to 32 of 100"
                        if is_prof else
                        "They change what the AI may TARGET")
                flag = (" --profile " + k.split("ours ", 1)[1].rstrip(")")
                        if is_prof else "")
                print(_wrap(
                    f"{k} has {v['branch_runs']} branch and "
                    f"{v['champion_runs']} champion run(s) — the SAME table, held "
                    f"out only because the {need} has never been played {why}. "
                    f"{what}, so pooling them would average two different "
                    f"harnesses. To compare: `manamap pilot simulate {seat} "
                    f"--pod {f.get('pod')}{flag} --games N`"
                    f"{'' if is_prof else ' with data/forge_overrides/ in place'}.",
                    indent="      "))
        print(f"    delta {f['delta']:+.3f}  CI [{f['ci95'][0]:+.3f}, "
              f"{f['ci95'][1]:+.3f}]  MDE {f['mde']}")
        # AN MDE MEANS NOTHING WITHOUT THE NULL IT IS SCALED AGAINST, and this
        # block printed the delta, the interval and the MDE while the null lived
        # in a different command (`pods <name> --calibration`).
        #
        # The cost, first-person, 2026-09-28: copy-burst-v1 reads 0.014 against
        # standard-v3 and the MDE at ~75 games is 0.115, so the detectable rate is
        # 0.129. Scaled against the BASELINE that is "a ninefold improvement" and
        # sounds unreachable, which is how I read it and concluded the run could
        # not answer its own question. Scaled against the NULL — 0.233, what our
        # decks actually score in seat 0 here — 0.129 is 55% of par: still a
        # losing deck, and an ordinary thing for a fixed engine to reach. The run
        # was well powered and I had argued myself out of it on a ratio.
        #
        # So the null is printed here, beside the figure it scales, which is the
        # same rule as every other number in this report.
        null = (f.get("null") or {}).get("rate")
        if null is None and "null" not in f:          # a report from before it was stored
            null = _pod_null(f.get("pod"))
        if null is not None:
            print(f"    the table's null is {null:.3f} (what our decks score in "
                  f"seat 0 at {f.get('pod')}) — champion "
                  f"{f['champion']['rate'] / null:.0%} of it, branch "
                  f"{f['branch']['rate'] / null:.0%}, and the MDE is "
                  f"{f['mde'] / null:.0%} of it")
        if f["mde"] and abs(f["delta"]) < f["mde"]:
            print(_wrap(f"UNDERPOWERED — this run could only resolve a "
                        f"difference of {f['mde']}; it rules out a large "
                        f"effect and cannot say which list is better.",
                        indent="    "))
            # WHAT IT WOULD TAKE, in games and hours, so "underpowered" is a
            # figure to budget against rather than a verdict to shrug at.
            need = _games_to_resolve(f["champion"]["rate"], f["delta"])
            if need:
                print(f"    to resolve the observed {f['delta']:+.3f} at 80% power: "
                      f"{need['games']}/arm (~{need['hours']:.0f} h at "
                      f"{need['per_minute']} games/min)")
        # THE OTHER ENDPOINTS, one line each, so a branch aimed at damage or at
        # the commander's access can read its figure where the win rate is.
        for axis, ep in (f.get("endpoints") or {}).items():
            if axis == "forge.win_rate" or ep.get("delta") is None:
                continue
            print(f"    {axis:38} {ep['champion']['value']:>8} -> {ep['branch']['value']:>8}"
                  f"  {ep['delta']:+.3f}  CI [{ep['ci95'][0]:+.3f}, {ep['ci95'][1]:+.3f}]"
                  f"  MDE {ep['mde']}{'  (conditional)' if ep.get('conditional') else ''}")


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
    _print_changes(doc)

    # ---------------------------------------------------------- objective
    o, g = doc.get("objective"), doc.get("objective_grade")
    print("\n  THE OBJECTIVE   the one thing here that can FAIL")
    if not o:
        print("    NONE — this branch predates the requirement and cannot be graded.")
    else:
        print(f"    {o['axis']} {o['op']} {o['value']}")
        state = (g or {}).get("state", "?").upper()
        _cpf = (doc.get("forge") or {}).get("cast_proofs") or {}
        floor = (f"   FLOOR ({len(_cpf['held'])} add(s) held" + (f", {len(_cpf['late'])} cast late" if _cpf.get("late") else "") + ")"
                 if _cpf.get("floor") else "")
        print(f"    RESULT   {(g or {}).get('reading', '—')}   ->   {state}{floor}")
        if (g or {}).get("why"):
            print(_wrap(g["why"], indent="             "))
        if o.get("why"):
            print(_wrap("Written when the branch was opened: " + o["why"],
                        indent="    "))

    # ---------------------------------------------------------- the table
    print("\n  MEASURED   10,000 goldfish games per list, same seed, same harness")
    dz = doc.get("design") or {}
    if dz:
        print(f"    {dz.get('exploratory_rows')} exploratory rows, Holm-corrected; "
              f"the primary is {dz.get('primary') or 'NOT DECLARED (no objective)'}")
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
        la, lb = (m.get("life") or [{}, {}])[0], (m.get("life") or [{}, {}])[1]
        if la or lb:
            def _life(key):
                x, y = la.get(key), lb.get(key)
                if x is None or y is None:
                    return "     — not measured"
                return f"{x:>6} -> {y:<5}{'' if y == x else f'  {y - x:+d}'}"
            print("\n    WHAT THE BASE CHARGES IN LIFE")
            print(f"      every tap-cycle   {_life('recurring_per_tap_cycle')}")
            print(f"      once, on entry    {_life('one_time_on_entry')}")
            print(_wrap("Two figures, never summed: a painland charges again "
                        "on EVERY activation and a fetch or a shock charges "
                        "ONCE, so a list that trades the first for the second "
                        "pays less the longer the game runs. Nothing else on "
                        "this page can see it — the goldfish models no life, "
                        "so a base that stops charging 3 a turn reads "
                        "identically to one that does not.", indent="      "))

    # ---------------------------------------------------------- real table
    _print_real_table(doc)

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
