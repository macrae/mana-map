"""Evaluate the interaction space before any default moves (2026-11-22; memo §4, plan commit 10).

    python -m manamap.analysis.eval_interaction [--write]

THE BENCH'S OWN QUESTION FIRST. Positives are the card pairs a checker-passed stack
names together (`engine_facts.verified_pairings`, 89-odd stacks fleet-wide): two cards
a rules checker saw cooperate on one board. Within each deck, for each card of a pair,
the deck's other cards are ranked by each method and we ask whether the partner lands
in the top 10. Methods: `feeds` (max of the two directions — a pairing is undirected),
`interaction` (alike cosine), `function` (the ability space, the incumbent), `cardbert`
(the pages' space), `synergy` (1 when the tag-rule graph lists the pair), and `random`
(the expected recall, 10 / (n − 1)). The incumbent cannot answer this question by
design — it was trained to put cards that do the SAME job together, and a sacrifice
outlet and a death payoff do different jobs — so this measure favours `feeds`, and
the report says so.

THE INCUMBENT'S QUESTION SECOND. Recall@10 on the 40-group golden set
(`data/eval/similarity_golden.json`, `eval_embeddings.recall_metrics`) and separation
on the accepted hard negatives — both favour a space built for "does the same job",
which is the incumbent's training signal.

`data/eval/interaction_eval.json` is the record, written with `--write`; the flip of
`spaces.DEFAULT` cites it or does not happen.
"""
import json
import random

import numpy as np

from manamap import config, spaces

EVAL_PATH = config.DATA_DIR / "eval" / "interaction_eval.json"
K = 10
SEED = 0


def _decks():
    return sorted(p.parent.name for p in config.DECKS_DIR.glob("*/stacks") if p.is_dir())


def pair_recall(methods, k=K):
    """Per method: recall@k over every (card, partner) direction of every verified
    pairing, ranked within the card's own deck. Returns (summary, per_deck)."""
    from manamap.pilot.common import load_deck_cards
    from manamap.pilot.engine_facts import verified_pairings
    hits = {m: 0 for m in methods}
    total = 0
    per_deck = {}
    for slug in _decks():
        try:
            deck = sorted({c["name"] for c in load_deck_cards(slug)["cards"]})
        except FileNotFoundError:
            continue
        pairs = verified_pairings(slug, set(deck))["pairs"]
        if not pairs:
            continue
        d_hits = {m: 0 for m in methods}
        d_total = 0
        for pair in pairs:
            a, b = pair["cards"]
            for q, t in ((a, b), (b, a)):
                others = [x for x in deck if x != q]
                d_total += 1
                for m, score in methods.items():
                    sc = score(q, others)
                    if sc is None:
                        continue
                    top = [x for x, _ in sorted(zip(others, sc), key=lambda z: (-z[1], z[0]))[:k]]
                    if t in top:
                        d_hits[m] += 1
        per_deck[slug] = {"pairs": len(pairs), "queries": d_total,
                          **{m: round(d_hits[m] / d_total, 3) for m in methods}}
        for m in methods:
            hits[m] += d_hits[m]
        total += d_total
    return {m: round(hits[m] / total, 3) if total else None for m in methods} | {"queries": total}, per_deck


def build_methods():
    """Score functions `(query, others) -> [score]` over the corpus artifacts."""
    from manamap.analysis import interaction as ix
    from manamap.pilot.card_pool import load_frame
    from manamap.pilot.common import load_synergy_graph
    names = load_frame()["name"].tolist()
    row = {}
    for i, n in enumerate(names):
        row.setdefault(n, i)
    mats = {}
    for slug in (spaces.INTERACTION, "function", "cardbert"):
        if spaces.get(slug).exists():
            mats[slug] = spaces.load(slug)
    events = (ix.load_card_events() or {}).get("cards") or {}
    syn = load_synergy_graph() or {}

    def cosine(mat):
        def f(q, others):
            if q not in row:
                return None
            v = mat[row[q]]
            return [float(mat[row[o]] @ v) if o in row else -1.0 for o in others]
        return f

    def feeds(q, others):
        sq = (events.get(q) or {}).get("sig") or []
        return [max(ix.feeds(sq, (events.get(o) or {}).get("sig") or []),
                    ix.feeds((events.get(o) or {}).get("sig") or [], sq)) for o in others]

    def synergy(q, others):
        near = {e["partner"] for e in syn.get(q) or []}
        back = {o for o in others if any(e["partner"] == q for e in syn.get(o) or [])}
        return [1.0 if (o in near or o in back) else 0.0 for o in others]

    rng = random.Random(SEED)

    def rand(q, others):
        return [rng.random() for _ in others]

    methods = {"feeds": feeds}
    for slug, mat in mats.items():
        methods[slug] = cosine(mat)
    methods["synergy"] = synergy
    methods["random"] = rand
    return methods, mats, names


def golden(mats, names):
    from manamap.analysis.eval_embeddings import (hard_negative_separation, load_golden,
                                                  recall_metrics, resolve_groups)
    groups, missing = resolve_groups(load_golden(), names)
    out = {}
    for slug, mat in mats.items():
        r = recall_metrics(mat, groups).get("all", {})
        hn = hard_negative_separation(mat, names)
        out[slug] = {"recall@10": round(r.get("recall@10", 0), 3), "median_rank": r.get("median_rank"),
                     "queries": r.get("queries"), "hard_negative_separation": round(hn["mean_separation"], 3),
                     "hard_negative_pairs": hn["n"]}
    return out, missing


def run():
    methods, mats, names = build_methods()
    summary, per_deck = pair_recall(methods)
    gold, missing = golden(mats, names)
    return {"k": K, "verified_pairs": {"recall_at_10": summary, "per_deck": per_deck,
                                      "note": "positives are cards a checker-passed stack names together; "
                                              "favours feeds by construction — said, not hidden"},
            "golden": gold, "golden_missing": missing,
            "note": "golden recall and hard-negative separation favour a space trained for 'does the same job', "
                    "the incumbent's signal"}


def main(args=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true")
    a = p.parse_args([] if args is None else args)
    doc = run()
    s = doc["verified_pairs"]["recall_at_10"]
    print(f"VERIFIED PAIRS — recall@{K} within the deck, {s['queries']} queries:")
    for m, v in s.items():
        if m != "queries":
            print(f"  {m:12} {v}")
    print("GOLDEN SET (recall@10, median rank) and HARD NEGATIVES (mean separation):")
    for m, v in doc["golden"].items():
        print(f"  {m:12} {v['recall@10']}  rank {v['median_rank']}  sep {v['hard_negative_separation']} ({v['hard_negative_pairs']})")
    if a.write:
        EVAL_PATH.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {EVAL_PATH}")
    return doc


if __name__ == "__main__":
    main(None if __import__("sys").argv[1:] == [] else __import__("sys").argv[1:])
