"""Step 15: Measure whether an embedding actually represents similarity.

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
    EVAL_BOOTSTRAP_RESAMPLES,
    EVAL_GEOMETRY_SAMPLE,
    EVAL_POOL_SIZES,
    EVAL_SEED,
    EVAL_THEME_GROUP_SIZE,
    EVAL_THEME_MAX_MEMBERS,
    EVAL_THEME_MIN_MEMBERS,
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



# ── Theme: a second relation, measured objectively ──────────────────────────


def edhrec_archetype_slugs(path=None):
    """Tribes EDHREC's own `tag_counts` treat as something people build.

    The gate that keeps this from being "shares a word in the type line".
    Returns an empty set when the cache is absent, and `theme_groups` then
    reports nothing rather than silently measuring a different relation.
    """
    from manamap.config import DATA_DIR

    root = path or (DATA_DIR / "edhrec")
    slugs = set()
    if not root.is_dir():
        return slugs
    for file in sorted(root.glob("average-*.json")):
        try:
            doc = json.loads(file.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        for tag in (doc.get("tag_counts") or []):
            if isinstance(tag, dict) and tag.get("slug"):
                slugs.add(tag["slug"])
    return slugs


def _is_archetype(subtype, slugs):
    """EDHREC pluralises its slugs — `vampires`, `dragons`, `elves`."""
    low = subtype.lower()
    return (low in slugs or low + "s" in slugs or low + "es" in slugs
            or (low.endswith("f") and low[:-1] + "ves" in slugs))


def theme_groups(frame, slugs=None, seed=EVAL_SEED):
    """`[{id, split, rows}]` — one group per buildable creature tribe.

    OBJECTIVE, unlike the golden set: membership is the type line, and the only
    judgement is the member-count band, which exists because `Human` (4,840) is a
    body type and `Phoenix` (41) is a theme. Independent of ROLE_PATTERNS and
    MECHANICAL_TAGS, which is what issue #12 requires of an eval — training mines
    positives from those, so a groups file derived from them would measure only
    whether training memorised its own supervision.

    Split is by a stable hash of the tribe name, so it does not move when the
    corpus does.
    """
    import hashlib

    slugs = edhrec_archetype_slugs() if slugs is None else slugs
    if not slugs:
        return []
    members = {}
    for row, line in enumerate(frame["type_line"].fillna("")):
        if "Creature" not in line or "—" not in line:
            continue
        for subtype in line.split("—")[1].split("//")[0].split():
            members.setdefault(subtype, []).append(row)
    rng = np.random.default_rng(seed)
    out = []
    for subtype, rows in sorted(members.items()):
        if not (EVAL_THEME_MIN_MEMBERS <= len(rows) <= EVAL_THEME_MAX_MEMBERS):
            continue
        if not _is_archetype(subtype, slugs):
            continue
        take = rows if len(rows) <= EVAL_THEME_GROUP_SIZE else [
            rows[i] for i in rng.choice(len(rows), EVAL_THEME_GROUP_SIZE, replace=False)]
        digest = hashlib.sha256(subtype.encode()).hexdigest()
        out.append({"id": f"tribe:{subtype}",
                    "split": "test" if int(digest[:8], 16) % 10 >= 3 else "dev",
                    "rows": sorted(int(r) for r in take)})
    return out


def centroid_collapse(embeddings, size=20, samples=600, seed=EVAL_SEED):
    """How much averaging costs this space: mean pairwise cosine, cards vs centroids.

    THE NUMBER THAT EXPLAINS COMMANDER SEARCH. `commander-search`, `build-deck`'s
    commander scoring and any deck-gap query are CENTROID operations, and a narrow
    cone cannot survive averaging. Measured 2026-08-31 at size 20:

        space      card-card   centroid-centroid   headroom
        function       0.721               0.981      0.019
        text           0.379               0.925      0.075
        layout         0.183               0.853      0.147

    A centroid figure near 1.0 means every deck looks alike in that space, so a
    centroid query has almost no signal left to rank on. Isolated on the golden
    set, swapping a single-card query for a centroid cut the function space's
    advantage from +0.191 (excluding zero) to +0.099 (spanning it).
    """
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    a, b = rng.integers(0, n, samples * 5), rng.integers(0, n, samples * 5)
    card = float(np.mean(np.sum(embeddings[a] * embeddings[b], axis=1)))
    centroids = np.array([
        embeddings[rng.integers(0, n, size)].mean(axis=0) for _ in range(samples)])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)
    ca, cb = rng.integers(0, samples, samples * 5), rng.integers(0, samples, samples * 5)
    cent = float(np.mean(np.sum(centroids[ca] * centroids[cb], axis=1)))
    return {"card": card, "centroid": cent, "headroom": 1.0 - cent}



# ── The candidate pool, and the interval on the difference ──────────────────


def playability_order(frame):
    """Row indices, most-played first. Unranked cards sort last.

    `edhrec_rank` is the only popularity signal in the corpus and it is missing
    for 8% of cards — those are, by construction, cards nobody plays, so sorting
    them to the back is the honest placement rather than a convenience.
    """
    rank = pd.to_numeric(frame["edhrec_rank"], errors="coerce").to_numpy(dtype=float)
    return np.argsort(np.where(np.isnan(rank), np.inf, rank))


def recall_by_group(embeddings, groups, order, pool=None, k=10, split=None):
    """recall@k for each group SEPARATELY, against `pool` distractors.

    Returns one number per group, not one per query, because queries inside a
    group are not independent — every card in `mana-dorks` is asked about the
    same five targets. Pooling them inflates the sample fivefold and shrinks
    every interval built on it. The group is the unit.

    CANDIDATES ARE THE GROUP'S OWN TARGETS PLUS `pool` DISTRACTORS, so a group is
    present at every pool size and the comparison across sizes is like for like.
    See `config.EVAL_POOL_SIZES` for what the alternative design did.
    """
    distractors = order if pool is None else order[:pool]
    out = []
    for group in groups:
        if split is not None and group.get("split") != split:
            continue
        rows = np.asarray(group["rows"], dtype=int)
        if len(rows) < 2:
            continue
        candidates = np.unique(np.concatenate([rows, distractors]))
        where = {int(c): j for j, c in enumerate(candidates)}
        block = embeddings[candidates]
        members = set(int(r) for r in rows)
        hits = total = 0
        for row in rows:
            scores = block @ embeddings[row]
            scores[where[int(row)]] = -np.inf
            top = {int(candidates[j]) for j in np.argpartition(-scores, k)[:k]}
            hits += len(top & (members - {int(row)}))
            total += len(rows) - 1
        out.append(hits / total if total else 0.0)
    return np.asarray(out, dtype=float)


def paired_bootstrap(a, b, resamples=EVAL_BOOTSTRAP_RESAMPLES, seed=EVAL_SEED):
    """Mean of `a - b` and a 95% interval, resampling GROUPS with replacement.

    PAIRED, because both spaces are scored on the identical groups — the
    variance that matters is in the groups, not between two independent samples.
    An unpaired interval here would be wider and would not answer the question.

    This is the piece the eval never had. `-0.012 recall@10` was reported as
    "training is destroying information" for months; the interval on that
    difference is [-0.088, +0.060] and spans zero.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return {"gap": 0.0, "lo": 0.0, "hi": 0.0, "n": 0, "excludes_zero": False}
    diff = a - b
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(diff), size=(resamples, len(diff)))
    means = diff[picks].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(means, [2.5, 97.5]))
    return {"gap": float(diff.mean()), "lo": lo, "hi": hi, "n": int(len(diff)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def pool_curve(spaces, groups, order, split="test", k=10):
    """`{pool: {label: per-group recall array}}` across `EVAL_POOL_SIZES`."""
    return {
        pool: {label: recall_by_group(emb, groups, order, pool, k, split)
               for label, emb in spaces.items()}
        for pool in EVAL_POOL_SIZES
    }


def format_pool_report(curve, challenger, baseline, split="test", relation="function"):
    """The table that decides the question, with an interval on every gap."""
    if challenger not in next(iter(curve.values())) or baseline not in next(iter(curve.values())):
        return ""
    n = len(next(iter(curve.values()))[challenger])
    lines = [
        "",
        f"  CANDIDATE POOL — relation: {relation.upper()} — {challenger} vs "
        f"{baseline}, {split} split, {n} groups at every size",
        f"    {'distractors':>12s} {'challenger':>11s} {'baseline':>9s} {'gap':>8s}   "
        f"95% CI on the difference",
    ]
    for pool, per_space in curve.items():
        stat = paired_bootstrap(per_space[challenger], per_space[baseline])
        mark = "  excludes 0" if stat["excludes_zero"] else ""
        lines.append(
            f"    {('corpus' if pool is None else pool):>12} "
            f"{per_space[challenger].mean():>11.3f} {per_space[baseline].mean():>9.3f} "
            f"{stat['gap']:>+8.3f}   [{stat['lo']:+.3f}, {stat['hi']:+.3f}]{mark}")
    lines += [
        "",
        "    Candidates are each group's own targets PLUS N most-played distractors,",
        "    so every group appears at every size. Bootstrap resamples GROUPS, which",
        "    is the unit of independence; queries inside a group are correlated.",
        "    NOTHING IN THE PRODUCT RANKS AGAINST THE WHOLE CORPUS: commander search",
        "    ranks against 79, Find Similar shows 12, build-deck ranks within a pool.",
    ]
    return "\n".join(lines)


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


def pool_section(challenger="function (ability)",
                 baseline="text baseline (frozen MiniLM)", split="test",
                 relation="function"):
    """The pool table for one RELATION. Returns (text, curve).

    `relation="function"` is the hand-authored golden set — "do these do the same
    job". `relation="theme"` is the tribe groups — "are these the same deck
    theme". They are different questions and the spaces answer them differently;
    reporting only the first is how the function space's total failure on theme
    (0.005 top-1 on tribal commanders) went unmeasured until 2026-08-31.

    Deliberately does its own loading rather than widening `collect`, whose
    two-tuple return is unpacked by `tests/test_embedding_quality.py:57`.
    """
    frame = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
    if relation == "theme":
        groups = theme_groups(frame)
    else:
        groups, _missing = resolve_groups(load_golden(), frame["name"].tolist())
    if not groups:
        return "", {}
    spaces = {}
    for label, path in (("function (ability)", ABILITY_EMBEDDINGS_PATH),
                        ("text baseline (frozen MiniLM)", TEXT_EMBEDDINGS_PATH),
                        ("layout (color+type)", EMBEDDINGS_PATH)):
        try:
            spaces[label] = _normalized(path)
        except FileNotFoundError:
            continue
    if challenger not in spaces or baseline not in spaces:
        return "", {}
    curve = pool_curve(spaces, groups, playability_order(frame), split=split)
    return format_pool_report(curve, challenger, baseline, split, relation), curve


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


def format_collapse(spaces):
    """What averaging costs each space — the number that explains commander search."""
    lines = ["", "  CENTROID COLLAPSE — mean pairwise cosine, 20-card centroids",
             f"    {'space':32s} {'card-card':>10s} {'centroid':>10s} {'headroom':>10s}"]
    for label, emb in spaces.items():
        c = centroid_collapse(emb)
        lines.append(f"    {label:32s} {c['card']:>10.3f} {c['centroid']:>10.3f} "
                     f"{c['headroom']:>10.3f}")
    lines += ["",
              "    Headroom near zero means every deck looks alike, so a CENTROID query",
              "    has nothing to rank on. commander-search, build-deck's commander",
              "    scoring and any deck-gap query are all centroid operations."]
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
    spaces = {}
    for label, path in (("function (ability)", ABILITY_EMBEDDINGS_PATH),
                        ("text baseline (frozen MiniLM)", TEXT_EMBEDDINGS_PATH),
                        ("layout (color+type)", EMBEDDINGS_PATH)):
        try:
            spaces[label] = _normalized(path)
        except FileNotFoundError:
            continue
    if spaces:
        print(format_collapse(spaces))

    # A BARE DIFFERENCE IS NOT A FINDING. This block used to print
    # "** Training is destroying information **" off `-0.012 recall@10`, with no
    # interval anywhere. Measured 2026-08-31, the interval on that difference is
    # [-0.088, +0.060] and spans zero: it was a TIE reported as a loss for
    # months, and the repo's own rule already said so —
    #   "a comparison carries the interval on the DIFFERENCE."
    text, curve = pool_section(relation="function")
    theme_text, _theme_curve = pool_section(relation="theme")
    if theme_text:
        print(theme_text)
        print("    THEME groups are objective — a creature subtype EDHREC treats as an")
        print("    archetype — so they are independent of the roles and tags training")
        print("    mines its positives from. 87 tribes, against the golden set's 40.")
    if text:
        print(text)
        full = curve.get(None, {})
        if full:
            stat = paired_bootstrap(full["function (ability)"],
                                    full["text baseline (frozen MiniLM)"])
            verdict = ("BEATS" if stat["gap"] > 0 else "trails")
            if not stat["excludes_zero"]:
                print(f"    At corpus scale the two are INDISTINGUISHABLE: "
                      f"{stat['gap']:+.3f} recall@10, 95% CI "
                      f"[{stat['lo']:+.3f}, {stat['hi']:+.3f}] spans zero over "
                      f"{stat['n']} groups. Neither space wins this comparison.")
            else:
                print(f"    At corpus scale the function space {verdict} the frozen text by "
                      f"{abs(stat['gap']):.3f} recall@10, 95% CI "
                      f"[{stat['lo']:+.3f}, {stat['hi']:+.3f}].")


if __name__ == "__main__":
    main()
