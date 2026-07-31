"""Step 14: Measure whether an embedding actually represents similarity.

A reporter, not a producer — the only step that writes no artifact. It exists
because every other test in this repo passes against a randomly initialized
model: `test_find_similar.py` checks L2 norms, `.bin`/`.npy` fidelity, byte
sizes and 2D-vs-128D divergence, all of which are structural. None of them can
tell you that `Doubling Season`'s nearest neighbours are arbitrary green
enchantments, and for a long time none of them did.

Three families of number, each answering a different question:

**Recall on the golden set** — can the space find cards a player would call
interchangeable? This is the one that matters, and the only one grounded in
human judgement rather than in the model's own supervision.

**Effective dimensionality** (participation ratio of the PCA spectrum) — how
much of the space is actually used. A collapsed model scores near 1 no matter
how many dimensions it nominally has, and collapse is invisible to recall until
it is severe.

**Neighbour spread** (cosine gap between the 1st and 50th neighbour) — whether
the ranking carries information. A spread of 0.004 means the top 50 are a
numerical tie and their order is float noise, which is a different failure from
"the wrong cards are near", and it looks fine in a top-5 list.
"""

import json

import numpy as np
import pandas as pd

from manamap.analysis.common import top_k_similar
from manamap.config import (
    ABILITY_EMBEDDINGS_PATH,
    EMBEDDINGS_PATH,
    EVAL_GEOMETRY_SAMPLE,
    EVAL_SEED,
    EVAL_SPREAD_PROBES,
    OUTPUT_CSV_PATH,
    SIMILARITY_GOLDEN_PATH,
    TEXT_EMBEDDINGS_PATH,
)

RECALL_KS = (10, 50)
SPREAD_K = 50


def load_golden(path=SIMILARITY_GOLDEN_PATH):
    """Load the golden set. Returns [{id, split, cards}] with the comment dropped."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["groups"]


def resolve_groups(groups, names):
    """Map card names to row indices, reporting any that no longer exist.

    Names are resolved rather than assumed because the golden set is tracked and
    hand-edited while `cards.csv` is regenerated — a renamed or removed card must
    surface as a warning, not as a silently smaller evaluation.
    """
    name_to_row = {}
    for row, name in enumerate(names):
        name_to_row.setdefault(name, row)  # first printing wins; 51 names duplicate

    resolved, missing = [], []
    for group in groups:
        rows = []
        for card in group["cards"]:
            if card in name_to_row:
                rows.append(name_to_row[card])
            else:
                missing.append(f"{group['id']}/{card}")
        if len(rows) >= 2:
            resolved.append({"id": group["id"], "split": group["split"], "rows": rows})
    return resolved, missing


def recall_metrics(embeddings, groups):
    """Recall@k and median rank of each group-mate, per split and overall.

    For every card in a group, the other members are the targets. Recall@k asks
    how many of them land in the top k; median rank says how far away they are
    when they do not, which is what distinguishes "close but not top-10" from
    "nowhere".
    """
    per_split = {}
    for group in groups:
        bucket = per_split.setdefault(group["split"], {"recalls": {k: [] for k in RECALL_KS},
                                                       "ranks": []})
        for query in group["rows"]:
            targets = [r for r in group["rows"] if r != query]
            if not targets:
                continue

            top = [row for row, _ in top_k_similar(embeddings, query, k=max(RECALL_KS))]
            for k in RECALL_KS:
                hits = sum(1 for t in targets if t in top[:k])
                bucket["recalls"][k].append(hits / len(targets))

            # Rank without a full sort: how many rows outscore the target.
            scores = embeddings @ embeddings[query]
            for target in targets:
                bucket["ranks"].append(int((scores > scores[target]).sum()) - 1)

    def summarise(bucket):
        return {
            **{f"recall@{k}": float(np.mean(v)) for k, v in bucket["recalls"].items()},
            "median_rank": float(np.median(bucket["ranks"])),
            "queries": len(bucket["ranks"]),
        }

    out = {split: summarise(b) for split, b in per_split.items()}
    if per_split:
        merged = {"recalls": {k: [] for k in RECALL_KS}, "ranks": []}
        for b in per_split.values():
            for k in RECALL_KS:
                merged["recalls"][k].extend(b["recalls"][k])
            merged["ranks"].extend(b["ranks"])
        out["all"] = summarise(merged)
    return out


def effective_dimensionality(embeddings, sample=EVAL_GEOMETRY_SAMPLE, seed=EVAL_SEED):
    """Participation ratio of the PCA spectrum: (Σλ)² / Σλ².

    Reads as "how many dimensions is this space really using". Equals d for an
    isotropic d-dimensional cloud and 1 for a line, so it is directly comparable
    against the nominal dimension in a way that a variance-explained curve is not.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(embeddings), min(sample, len(embeddings)), replace=False)
    centred = embeddings[idx] - embeddings[idx].mean(axis=0)
    eigenvalues = np.linalg.svd(centred, compute_uv=False) ** 2
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    return float(total**2 / (eigenvalues**2).sum())


def neighbour_spread(embeddings, probes=EVAL_SPREAD_PROBES, seed=EVAL_SEED):
    """Mean cosine gap between the 1st and SPREAD_K-th neighbour.

    Near zero means the top neighbours are indistinguishable from each other, so
    whichever one is returned first is an artefact of float ordering.
    """
    rng = np.random.default_rng(seed)
    gaps = []
    for query in rng.choice(len(embeddings), min(probes, len(embeddings)), replace=False):
        top = top_k_similar(embeddings, int(query), k=SPREAD_K)
        if len(top) >= 2:
            gaps.append(top[0][1] - top[-1][1])
    return float(np.mean(gaps)) if gaps else 0.0


def evaluate(embeddings, groups):
    """Every metric for one embedding space."""
    return {
        "shape": list(embeddings.shape),
        "effective_dim": effective_dimensionality(embeddings),
        "neighbour_spread": neighbour_spread(embeddings),
        "recall": recall_metrics(embeddings, groups),
    }


def _normalized(path):
    """Load an embedding artifact, L2-normalizing defensively.

    Model output is already normalized (`model.py`'s forward ends in F.normalize),
    but the text baseline comes straight from the sentence transformer, and this
    function is the one place both are treated identically.
    """
    array = np.load(path)
    norms = np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)
    return array / norms


def collect(paths=None):
    """Evaluate every available embedding space. Returns {label: metrics}."""
    if paths is None:
        paths = {
            "function (ability)": ABILITY_EMBEDDINGS_PATH,
            "layout (color+type)": EMBEDDINGS_PATH,
            "text baseline (frozen MiniLM)": TEXT_EMBEDDINGS_PATH,
        }

    names = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)["name"].tolist()
    groups, missing = resolve_groups(load_golden(), names)
    if missing:
        print(f"    WARNING: {len(missing)} golden cards not in cards.csv: "
              f"{', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}")

    results = {}
    for label, path in paths.items():
        try:
            embeddings = _normalized(path)
        except FileNotFoundError:
            print(f"    skipping {label}: {path.name} not found")
            continue
        results[label] = evaluate(embeddings, groups)
    return results, groups


def format_report(results):
    """The table, ordered worst-first so a regression is the first thing read."""
    lines = [
        "",
        f"{'space':32s} {'dim':>5s} {'effdim':>7s} {'spread':>7s} "
        f"{'r@10':>6s} {'r@50':>6s} {'medRank':>8s}   (test split)",
        "-" * 92,
    ]
    ordered = sorted(results.items(),
                     key=lambda kv: kv[1]["recall"].get("test", {}).get("recall@10", 0.0))
    for label, m in ordered:
        test = m["recall"].get("test", {})
        lines.append(
            f"{label:32s} {m['shape'][1]:5d} {m['effective_dim']:7.2f} "
            f"{m['neighbour_spread']:7.4f} "
            f"{test.get('recall@10', 0):6.3f} {test.get('recall@50', 0):6.3f} "
            f"{test.get('median_rank', 0):8.0f}"
        )
    lines.append("")
    lines.append("  effdim  = participation ratio; how many of `dim` dimensions are in use")
    lines.append("  spread  = cosine gap 1st->50th neighbour; ~0 means the ranking is noise")
    lines.append("  test    = golden groups written after the diagnosis, never tuned against")
    return "\n".join(lines)


def main():
    results, groups = collect()
    if not results:
        print("    No embedding artifacts found — run the pipeline first.")
        return

    splits = {}
    for group in groups:
        splits[group["split"]] = splits.get(group["split"], 0) + 1
    print(f"    Golden set: {len(groups)} groups "
          f"({', '.join(f'{n} {s}' for s, n in sorted(splits.items()))})")
    print(format_report(results))

    baseline = results.get("text baseline (frozen MiniLM)")
    function = results.get("function (ability)")
    if baseline and function:
        gap = (function["recall"]["test"]["recall@10"]
               - baseline["recall"]["test"]["recall@10"])
        if gap < 0:
            print(f"    ** The trained function space is {abs(gap):.3f} recall@10 WORSE than "
                  f"the frozen text it is built from. Training is destroying information. **")
        else:
            print(f"    Function space beats the frozen-text baseline by {gap:+.3f} recall@10.")


if __name__ == "__main__":
    main()
