"""Score the obsolescence index against the ways it is known to be wrong.

WHY THIS EXISTS. Nothing measured this index. It shipped 22,753 claimed
replacements, `viz/js/mana-map.js` published them to users as **"Obsoleted By"**,
and `pool_facts.py` carried a hand-written caveat recording that four of the first
eight results on a real box were wrong — with no way to tell whether a change to
the gates made that better or worse.

The audit it was written from, on the index as it stood in 2026-08:

    15.5%  a COST reported as an advantage (`Additional: discard`)
    22.9%  the replacement ADDS a restriction the original lacks
    29.9%  of ability-pairs, the replacement's ability costs MORE
     8.2%  the replacement is commander-illegal
    30.8%  the replacement is played LESS than the card it "outclasses"
    36.5%  fail at least one purely mechanical check
    82.0%  share a real functional role — the RETRIEVAL half works

THE POINT OF THE LAST LINE. The embedding is mostly finding cards that do the same
job; what fails is the judgement laid on top. A change that improves precision by
throwing away recall has not fixed this, and the role-agreement figure is what says
so.

MODELLED ON `eval_embeddings` (step 15) and deliberately NOT a pipeline step: it
scores an artifact the pipeline built rather than building one, and `manamap run`
should not pay for it.

EVERY CLASS HERE IS A LOWER BOUND. These are regexes over oracle text; they cannot
see semantics, and the sampled survivors are worse than the mechanical rate implies.
A future gate that drives a class to zero has not necessarily fixed the card — read
the sample.
"""

import collections
import json
import math
import re

from manamap.config import OBSOLESCENCE_INDEX_PATH

#: The RESTRICTION classes, kept here rather than imported from `power_creep` on
#: purpose: an eval that shares its subject's definition of a defect cannot see a
#: defect in that definition. This is the one place in the repo where duplicating
#: a pattern is the right call, and it is duplicated deliberately.
GATES = {
    "timing": re.compile(r"only (?:during|as a sorcery|any time you could|if)", re.I),
    "conditional": re.compile(r"\bas long as\b|\bonly if\b", re.I),
    "additional cost": re.compile(r"as an additional cost", re.I),
    "sacrifice cost": re.compile(r"sacrifice (?:a|an|another|two)\b", re.I),
    "discard cost": re.compile(r"discard (?:a|two|\d+)\b", re.I),
    "life cost": re.compile(r"pay \d+ life", re.I),
}

#: Tags that name a COST or a drawback far more often than a gain. The index
#: reports every tag B has and A lacks as `Additional: <tag>`, which is how
#: "discard a card for hexproof" became an upgrade over unconditional hexproof.
COST_TAGS = {"discard", "sacrifice", "counters_minus", "mill", "tap_ability",
             "upkeep_trigger", "death_trigger"}

#: A role 62.4% of the classified corpus carries. Two cards sharing only this
#: share "is a creature", which is not a job.
GENERIC_ROLE = "threat:body"

_ACTIVATION = re.compile(r"(^|\n)([^:\n]{0,60}?):", re.M)
_GENERIC = re.compile(r"\{(\d+)\}")
_COLOURED = re.compile(r"\{[WUBRGC]\}")
PHYREXIAN = re.compile(r"\{[WUBRG]/P\}")


def _rank(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def activation_mana(text):
    """Cheapest mana in an activation cost, or None if the card has no ability.

    Bartolome del Presidio's sac ability is FREE and the index's "upgrade"
    charges {2} for the same thing, reported as `Better Toughness`. An ability's
    cost is invisible to a comparison that only reads mana value.
    """
    best = None
    for match in _ACTIVATION.finditer(text or ""):
        cost = match.group(2)
        if "{" not in cost and "Sacrifice" not in cost and "Discard" not in cost:
            continue
        total = len(_COLOURED.findall(cost)) + sum(
            int(g) for g in _GENERIC.findall(cost))
        best = total if best is None else min(best, total)
    return best


def collect(index=None, frame=None, pool=None, roles=None):
    """Every failure class, counted over every pair the index publishes."""
    from manamap.pilot import card_pool
    from manamap.pilot.common import load_card_roles

    index = index if index is not None else json.loads(
        OBSOLESCENCE_INDEX_PATH.read_text())
    frame = frame if frame is not None else card_pool.load_frame()
    if frame is None:
        return None
    frame = frame.drop_duplicates("name")
    pool = pool if pool is not None else (card_pool.load_pool() or {})
    roles = roles if roles is not None else load_card_roles()

    text = dict(zip(frame["name"], frame["oracle_text"].fillna("")))
    rank = dict(zip(frame["name"], frame["edhrec_rank"]))

    # The schema moved: `compare_with` is the current key, `obsoleted_by` the
    # one the audit was taken against. Reading both is what lets a before/after
    # comparison mean anything.
    def entries(v):
        return v.get("compare_with") or v.get("obsoleted_by") or []

    pairs = [(a, r) for a, v in index.items() for r in entries(v)]
    n = len(pairs) or 1
    counts = collections.Counter()
    ability_pairs = ability_worse = 0
    role_both = role_shared = role_generic_only = 0
    newer_and_played_more = newer_total = 0

    for a, r in pairs:
        b = r["name"]
        ta, tb = text.get(a, ""), text.get(b, "")

        if b in pool and not pool[b]["legal"]:
            counts["replacement is not legal"] += 1
        for label, pat in GATES.items():
            if pat.search(tb) and not pat.search(ta):
                counts[f"adds a restriction: {label}"] += 1
        advantages = r.get("advantages") or []   # pre-repair key only
        for adv in advantages:
            if not str(adv).startswith("Additional: "):
                continue
            tags = [t.strip() for t in str(adv)[len("Additional: "):].split(",")]
            if any(t in COST_TAGS for t in tags):
                counts["a cost counted as an advantage"] += 1
                break
        if PHYREXIAN.search(ta):
            counts["original has Phyrexian mana (cmc overstates its cost)"] += 1

        ra, rb = _rank(rank.get(a)), _rank(rank.get(b))
        if rb is None:
            counts["replacement has no EDHREC rank"] += 1
        elif ra is not None and rb > ra:
            counts["replacement is played less than the original"] += 1

        ca, cb = activation_mana(ta), activation_mana(tb)
        if ca is not None and cb is not None:
            ability_pairs += 1
            if cb > ca:
                ability_worse += 1

        if roles.get(a) and roles.get(b):
            role_both += 1
            shared = set(roles[a]) & set(roles[b])
            specific = {x for x in shared if x != GENERIC_ROLE}
            if specific:
                role_shared += 1
            elif shared:
                role_generic_only += 1

        released = str(r.get("released_at") or "")
        if released:
            newer_total += 1
            if ra is not None and rb is not None and rb < ra:
                newer_and_played_more += 1

    # THE STRENGTH DISTRIBUTION IS NOW THE MAIN OUTPUT. The index stopped
    # publishing a verdict: it publishes a degree, and the pilot sets the line.
    # So the question this eval asks changed with it — not "how many claims are
    # wrong" but "does the score SEPARATE the wrong ones from the right ones".
    buckets = collections.Counter()
    flagged_strength, clean_strength = [], []
    for a, r in pairs:
        st = r.get("strength")
        if st is None:
            continue
        buckets[min(9, int(st * 10)) / 10] += 1
        tb = text.get(r["name"], "")
        ta = text.get(a, "")
        problem = (any(p.search(tb) and not p.search(ta) for p in GATES.values())
                   or bool(r.get("costs")) or bool(r.get("narrows")))
        (flagged_strength if problem else clean_strength).append(st)

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    return {
        "anchors": len(index),
        "pairs": len(pairs),
        "strength": {
            "histogram": {f"{k:.1f}": v for k, v in sorted(buckets.items())},
            "mean_where_a_problem_is_detectable": mean(flagged_strength),
            "mean_where_none_is": mean(clean_strength),
            "separation": (round(mean(clean_strength) - mean(flagged_strength), 3)
                           if flagged_strength and clean_strength else None),
        },
        "classes": {k: {"pairs": v, "share": round(v / n, 4)}
                    for k, v in counts.most_common()},
        "ability": {"pairs": ability_pairs,
                    "replacement_costs_more": ability_worse,
                    "share": round(ability_worse / (ability_pairs or 1), 4)},
        "role_agreement": {
            "both_classified": role_both,
            "share_a_specific_role": role_shared,
            "share_only_a_body": role_generic_only,
            "share": round(role_shared / (role_both or 1), 4)},
        "newer_predicts_played_more": {
            "pairs": newer_total,
            "share": round(newer_and_played_more / (newer_total or 1), 4)},
    }


def format_report(got):
    out = [f"    {got['pairs']:,} claimed replacement(s) across "
           f"{got['anchors']:,} anchor(s)", ""]
    for name, cell in got["classes"].items():
        out.append(f"      {cell['share']:>6.1%}  {cell['pairs']:>7,}  {name}")
    a = got["ability"]
    out += ["",
            f"      {a['share']:>6.1%}  {a['replacement_costs_more']:>7,}  "
            f"the replacement's ACTIVATED ABILITY costs more "
            f"(of {a['pairs']:,} ability-pairs)"]
    st = got.get("strength") or {}
    if st.get("histogram"):
        out += ["", "    STRENGTH — the index publishes a degree, not a verdict."]
        for k, v in st["histogram"].items():
            bar = "#" * max(1, round(40 * v / max(st["histogram"].values())))
            out.append(f"      {k}  {v:>6,}  {bar}")
        if st.get("separation") is not None:
            out += ["",
                    f"      mean where a problem is detectable: "
                    f"{st['mean_where_a_problem_is_detectable']}",
                    f"      mean where none is:                 "
                    f"{st['mean_where_none_is']}",
                    f"      SEPARATION: {st['separation']:+.3f}  — the score must "
                    f"rank the bad pairs BELOW the good ones, or it is decoration."]
    r = got["role_agreement"]
    out += ["",
            f"    RETRIEVAL — {r['share']:.1%} of pairs share a functional role "
            f"other than '{GENERIC_ROLE}'",
            f"      ({r['share_only_a_body']:,} share only 'is a creature', "
            f"which is not a job)"]
    nw = got["newer_predicts_played_more"]
    out += ["",
            f"    THE `newer` GATE — it predicts 'played more' "
            f"{nw['share']:.1%} of the time.",
            "      Printed later is not evidence of stronger; it is evidence of "
            "a larger card pool."]
    out += ["",
            "    Every class above is a LOWER BOUND: these are regexes over "
            "oracle text and",
            "    cannot see semantics. Read the sample before believing a class "
            "reached zero."]
    return "\n".join(out)


def main(args=None):
    if not OBSOLESCENCE_INDEX_PATH.exists():
        raise SystemExit(
            f"{OBSOLESCENCE_INDEX_PATH} not found — run `manamap power-creep`.")
    got = collect()
    if got is None:
        raise SystemExit("requires the card corpus (run `manamap extract`).")
    print("\n  OBSOLESCENCE INDEX — scored against how it is known to be wrong\n")
    print(format_report(got))
    print()


if __name__ == "__main__":
    main()
