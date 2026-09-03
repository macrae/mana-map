"""Spike S1, made answerable: can the embedding rank commanders?

PRD-v1 gates commander search (§6) on one question — *does embedding proximity
track functional similarity well enough to rank commanders?* — and the existing
`eval-embeddings` cannot answer it. That eval asks whether functionally
interchangeable CARDS are neighbours, over 40 hand-authored groups. It is the
right instrument for the wrong question, and at ~50 dev / ~160 test queries it
is too small to steer on: `CLAUDE.md` records a text-weight sweep that looked
like a win until dev and test picked different values.

So this measures **the decision the product actually makes**, end to end:

    a seed of ~20 cards
      -> seed centroid
      -> rank every candidate commander by centroid proximity
      -> where did the commander the seed CAME FROM land?

Top-1, top-5, top-20 and MRR against a candidate pool of N. That is a number
that means something to a user: "I gave it twenty cards out of a Shorikai deck
and it put Shorikai third."

Three things this gets right that a card-similarity eval cannot:

**The seed is HELD OUT.** The cards drawn for the seed are removed from that
commander's reference centroid before ranking. Leave them in and the true
commander wins by containing its own query — a leak that would report near-
perfect accuracy for an embedding that had learned nothing.

**Type control is measured, not assumed.** §6.1 step 6 argues that a seed of
instants and sorceries embeds nowhere near a creature-heavy centroid, so the
reference must be restricted to the seed's type composition or the ranking
measures deck composition rather than deck identity. That is a *claim*. This
runs both ways and prints both, so it is a finding rather than an assertion.

**Ground truth is frozen.** EDHREC rankings move; §6.1 step 4 says not to freeze
them for the product, and that is right for the product and exactly wrong for a
benchmark. `ingest/edhrec.py` caches to disk, so a change in the number here is
a change in the model and never a change in the metagame.

Run: `manamap eval-commander-search`
"""

import json
import random

import numpy as np
import pandas as pd

from manamap.config import (ABILITY_EMBEDDINGS_PATH, EMBEDDINGS_PATH, DATA_DIR,
                            TEXT_EMBEDDINGS_PATH)
from manamap import console
# THE PRODUCT OWNS THE MATHS AND THE EVAL IMPORTS IT — never the other way, and
# never a second copy. An eval with its own centroid measures code nobody ships,
# and the two drift the first time one of them is tuned. Same argument the deck
# builder makes for having exactly one scorer.
from manamap.analysis.commander_search import (BASIC_LANDS, Corpus, centroid,
                                               composition, normalized as _normalized,
                                               primary_type as _primary_type,
                                               type_controlled_rows)

#: The frozen candidate pool. Written by `--refresh`, read by everything else.
POOL_PATH = DATA_DIR / "eval" / "commander_pool.json"


SEED_SIZE = 20          # §6.1 step 1: "user supplies ~20 cards"
EVAL_SEED = 20260825    # fixed, so a re-run is a re-run

#: How many independent held-out draws to average over.
#:
#: ONE DRAW IS NOISE, and this constant exists because the first version of this
#: module reported one and I read a finding out of it. On a single draw, type
#: control appeared to buy the text space +12.7 points of top-1; over ten draws
#: it buys roughly zero and is as often negative. The pool is 79 commanders, so
#: one query per commander per draw, and a single pass swings ±5 points on the
#: draw alone. `CLAUDE.md` already records this lesson about the card-level
#: golden set — "do not tune hyperparameters on it, those differences are
#: noise" — and the fix there was to distrust the number. The fix here is to
#: make the number trustworthy: report the mean and the SPREAD, so a reader can
#: see whether a gap survives resampling.
REPEATS = 10




def load_corpus():
    """The corpus views this eval needs, through the product's own reader."""
    c = Corpus()
    return c.names, c, c.types


def refresh_pool(identities, per_identity, limit_decks=None):
    """Fetch and freeze the candidate pool. A deliberate act with its own commit."""
    from manamap.ingest import edhrec

    pool = []
    seen = set()
    with console.task("Fetching commanders", total=len(identities), unit="identities") as t:
        for ident in identities:
            t.state(ident)
            for name in edhrec.top_commanders(ident, limit=per_identity):
                if name not in seen:
                    seen.add(name)
                    pool.append({"commander": name, "identity": ident})
            t.advance()

    wanted = pool if limit_decks is None else pool[:limit_decks]
    kept = []
    with console.task("Fetching average decks", total=len(wanted), unit="decks") as t:
        for entry in wanted:
            t.state(entry["commander"])
            try:
                deck = edhrec.average_deck(entry["commander"])
            except Exception as exc:                  # a missing deck is data, not a crash
                t.advance()
                console.err(f"    no average deck for {entry['commander']}: {exc}")
                continue
            if deck["cards"]:
                kept.append({**entry, "cards": [n for n, _ in deck["cards"]]})
            t.advance()

    POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    POOL_PATH.write_text(json.dumps({
        "_comment": [
            "Frozen candidate pool for `manamap eval-commander-search` (PRD-v1 spike S1).",
            "FETCHED ONCE AND COMMITTED. EDHREC's rankings move, which is right for the",
            "product (PRD-v1 §6.1 step 4 says do not freeze them) and wrong for a benchmark:",
            "ground truth that shifts underneath cannot tell a model change from a",
            "metagame change. Refresh with --refresh, deliberately, in its own commit.",
        ],
        "identities": identities, "per_identity": per_identity,
        "commanders": kept,
    }, indent=1) + "\n", encoding="utf-8")
    return kept


def load_pool():
    if not POOL_PATH.exists():
        raise SystemExit(f"no candidate pool at {POOL_PATH} — run "
                         f"`manamap eval-commander-search --refresh` once to fetch it")
    return json.loads(POOL_PATH.read_text(encoding="utf-8"))["commanders"]


def _rows(cards, corpus):
    """Deck names -> rows, through the product's resolver."""
    rows, _missing = corpus.rows(cards)
    return rows




def evaluate(embeddings, pool, corpus, types, controlled, rng):
    """Rank every commander for every held-out seed. Returns the metric block."""
    decks = []
    for entry in pool:
        rows = _rows(entry["cards"], corpus)
        if len(rows) >= SEED_SIZE + 10:               # enough to hold out AND still describe
            decks.append((entry["commander"], rows))

    ranks = []
    for i, (name, rows) in enumerate(decks):
        seed_rows = rng.sample(rows, SEED_SIZE)
        seed_set = set(seed_rows)
        seed_vec = centroid(embeddings, seed_rows)
        if seed_vec is None:
            continue

        comp = composition(seed_rows, types)      # the product's own function

        scores = []
        for j, (other, other_rows) in enumerate(decks):
            # HOLD OUT. Without this the true commander contains its own query
            # and wins on every space, including a random one.
            ref = [r for r in other_rows if r not in seed_set] if i == j else other_rows
            if controlled:
                ref = type_controlled_rows(ref, types, comp, rng)
            vec = centroid(embeddings, ref)
            scores.append(-1.0 if vec is None else float(seed_vec @ vec))

        truth = scores[i]
        rank = sum(1 for s in scores if s > truth) + 1
        ranks.append(rank)

    n = len(ranks)
    if not n:
        return {"queries": 0}
    return {
        "queries": n,
        "candidates": len(decks),
        "top1": sum(1 for r in ranks if r == 1) / n,
        "top5": sum(1 for r in ranks if r <= 5) / n,
        "top20": sum(1 for r in ranks if r <= 20) / n,
        "mrr": float(np.mean([1 / r for r in ranks])),
        "median_rank": float(np.median(ranks)),
        "random_top1": 1 / len(decks),
    }


def repeated(embeddings, pool, corpus, types, controlled, repeats=REPEATS):
    """`evaluate` over K independent draws: mean, min and max of each metric.

    The spread is not decoration. It is what separates "this space is better"
    from "this draw was easier", and the two are indistinguishable from a single
    number — which is how the first reading of this eval produced a confident,
    wrong claim about type control.
    """
    runs = [evaluate(embeddings, pool, corpus, types, controlled,
                     random.Random(EVAL_SEED + i))
            for i in range(repeats)]
    runs = [r for r in runs if r.get("queries")]
    if not runs:
        return {"queries": 0}
    out = {"queries": runs[0]["queries"], "candidates": runs[0]["candidates"],
           "repeats": len(runs), "random_top1": runs[0]["random_top1"]}
    for key in ("top1", "top5", "top20", "mrr", "median_rank"):
        vals = [r[key] for r in runs]
        out[key] = float(np.mean(vals))
        out[key + "_min"] = float(min(vals))
        out[key + "_max"] = float(max(vals))
    return out


def main(args=None):
    identities = ["w", "u", "b", "r", "g", "wu", "ub", "br", "rg", "gw"]
    if args is not None and getattr(args, "refresh", False):
        refresh_pool(identities, per_identity=getattr(args, "per_identity", 8),
                     limit_decks=getattr(args, "limit", None))

    pool = load_pool()
    names, corpus, types = load_corpus()

    # CardBERT is here because this eval is a CENTROID operation and centroid
    # headroom is where it differs most: 0.976 against the function space's
    # 0.019. `eval-embeddings` measures single-card retrieval, which is a
    # different question and one CardBERT loses.
    from manamap.training.train_cardbert import EMBEDDINGS_PATH as CARDBERT_PATH

    spaces = {
        "function (ability)": ABILITY_EMBEDDINGS_PATH,
        "layout (color+type)": EMBEDDINGS_PATH,
        "text baseline (frozen MiniLM)": TEXT_EMBEDDINGS_PATH,
        "cardbert (masked fields)": CARDBERT_PATH,
    }

    rows = []
    for label, path in spaces.items():
        try:
            emb = _normalized(path)
        except FileNotFoundError:
            console.err(f"    skipping {label}: {path.name} not found")
            continue
        for controlled in (False, True):
            # The SAME rng seed for every space and both control settings, so
            # every row answers the same held-out seeds. A per-row rng would let
            # an easier draw look like a better embedding.
            rows.append((label, controlled,
                         repeated(emb, pool, corpus, types, controlled)))

    head = next((m for _, _, m in rows if m.get("queries")), {})
    print(f"\nCOMMANDER SEARCH — {SEED_SIZE}-card held-out seed, "
          f"{head.get('candidates', 0)} candidates, "
          f"mean of {head.get('repeats', 0)} draws\n")
    print(f"{'space':32s} {'control':>8s} {'top1':>16s} {'top5':>7s} "
          f"{'top20':>7s} {'MRR':>7s} {'medRank':>8s}")
    print("-" * 92)
    for label, controlled, m in rows:
        if not m.get("queries"):
            continue
        # top1 carries its own range, because that is the column people quote.
        span = f"{m['top1']:.3f} [{m['top1_min']:.2f}-{m['top1_max']:.2f}]"
        print(f"{label:32s} {'type' if controlled else 'none':>8s} "
              f"{span:>16s} {m['top5']:7.3f} {m['top20']:7.3f} "
              f"{m['mrr']:7.3f} {m['median_rank']:8.1f}")
    if head:
        print(f"\n  random baseline top1 = {head['random_top1']:.3f} "
              f"({head['candidates']} candidates), {head['queries']} held-out seeds "
              f"x {head['repeats']} draws")
        print("  top1 ranges overlap => the difference is the draw, not the space.")
    return rows
