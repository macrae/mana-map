"""Step 12: Cluster 2D projections into named regions using HDBSCAN."""

import json
from collections import Counter

import hdbscan
import numpy as np
import pandas as pd

from manamap.config import (
    OUTPUT_CSV_PATH,
    PROJECTION_PATH,
    ABILITY_PROJECTION_PATH,
    REGIONS_DEFAULT_PATH,
    REGIONS_ABILITY_PATH,
    REGION_L0_MIN_CLUSTER_SIZE,
    REGION_L0_MIN_SAMPLES,
    REGION_L1_MIN_CLUSTER_SIZE,
    REGION_L1_MIN_SAMPLES,
    REGION_L2_MIN_CLUSTER_SIZE,
    REGION_L2_MIN_PARENT_SIZE,
    REGION_L2_MIN_SAMPLES,
    REGION_COLOR_DOMINANCE,
    REGION_TYPE_DOMINANCE,
    REGION_TAG_DISPLAY_NAMES,
    REGION_COLOR_DISPLAY_NAMES,
    REGION_GUILD_NAMES,
    REGION_MIN_TAG_PRESENCE,
    REGION_NAMES_PATH,
)


# ── Naming helpers ──────────────────────────────────────────────────────


PLURAL_OVERRIDES = {
    "Sorcery": "Sorceries",
    "Unknown": "Unknown",
}


def _pluralize_type(supertype):
    """Pluralize a supertype name correctly."""
    if supertype in PLURAL_OVERRIDES:
        return PLURAL_OVERRIDES[supertype]
    if supertype.endswith("s"):
        return supertype
    return supertype + "s"


def _count_cluster_tags(tags_list):
    """Count tag occurrences in a list of comma-separated tag strings.

    Returns (Counter, int) — tag counts and total number of items.
    """
    counts = Counter()
    n = len(tags_list)
    for tags_str in tags_list:
        if not tags_str or (isinstance(tags_str, float) and np.isnan(tags_str)):
            continue
        for tag in str(tags_str).split(","):
            tag = tag.strip()
            if tag:
                counts[tag] += 1
    return counts, n


def name_cluster_colortype(colors, supertypes):
    """Name a cluster from the Color+Type map using dominant color/type.

    Args:
        colors: list of primary_color values for cards in this cluster
        supertypes: list of supertype values for cards in this cluster

    The default map names by color and type only. Tag descriptors are the
    ability map's job (name_cluster_ability) and appear here solely as
    disambiguating suffixes, added later by _deduplicate_labels.

    Returns:
        (label, short) tuple
    """
    n = len(colors)
    if n == 0:
        return "Unknown", "Unknown"

    color_counts = Counter(colors)
    type_counts = Counter(supertypes)

    top_color, top_color_n = color_counts.most_common(1)[0]
    top_type, top_type_n = type_counts.most_common(1)[0]

    color_frac = top_color_n / n
    type_frac = top_type_n / n

    color_display = REGION_COLOR_DISPLAY_NAMES.get(top_color, top_color)
    type_display = _pluralize_type(top_type)

    has_color = color_frac >= REGION_COLOR_DOMINANCE
    has_type = type_frac >= REGION_TYPE_DOMINANCE

    # Single dominant color wins first (before guild check)
    if has_color and has_type:
        return f"{color_display} {type_display}", color_display
    if has_color:
        return color_display, color_display

    # Check for guild (2-color pair >= 50% combined, no single dominant)
    single_colors = [c for c in color_counts if c in ("W", "U", "B", "R", "G")]
    if len(single_colors) >= 2:
        sorted_colors = sorted(single_colors, key=lambda c: color_counts[c], reverse=True)
        top2 = sorted_colors[:2]
        pair_frac = (color_counts[top2[0]] + color_counts[top2[1]]) / n
        if pair_frac >= 0.50:
            guild_key = frozenset(top2)
            guild_name = REGION_GUILD_NAMES.get(guild_key)
            if guild_name:
                if has_type:
                    return f"{guild_name} {type_display}", guild_name
                return guild_name, guild_name

    if has_type:
        return type_display, type_display

    # Fallback: top color + top type even if not dominant
    return f"{color_display} {type_display}", color_display


def name_cluster_ability(tags_list, global_tag_freq, colors=None, supertypes=None):
    """Name a cluster from the Abilities map using TF-IDF-like tag scoring.

    Only considers tags present in >= REGION_MIN_TAG_PRESENCE of the cluster.
    Falls back to color/type naming if no tag meets the threshold.

    Args:
        tags_list: list of tag strings (comma-separated) for cards in this cluster
        global_tag_freq: dict of tag -> global frequency fraction
        colors: optional list of colors for fallback naming
        supertypes: optional list of supertypes for fallback naming

    Returns:
        (label, short) tuple
    """
    n = len(tags_list)
    if n == 0:
        return "Unknown", "Unknown"

    cluster_tag_counts, _ = _count_cluster_tags(tags_list)

    if not cluster_tag_counts:
        if colors and supertypes:
            return name_cluster_colortype(colors, supertypes)
        return "Vanilla", "Vanilla"

    # TF-IDF scoring, but only consider tags with >= minimum presence
    scored = {}
    for tag, count in cluster_tag_counts.items():
        cluster_freq = count / n
        if cluster_freq < REGION_MIN_TAG_PRESENCE:
            continue
        global_freq = global_tag_freq.get(tag, 0.01)
        scored[tag] = cluster_freq / max(global_freq, 0.01)

    # No tag meets presence threshold → fall back to color/type
    if not scored:
        if colors and supertypes:
            return name_cluster_colortype(colors, supertypes)
        return "Vanilla", "Vanilla"

    ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    top_tag, top_score = ranked[0]
    top_display = REGION_TAG_DISPLAY_NAMES.get(top_tag, top_tag.replace("_", " ").title())

    if len(ranked) >= 2:
        second_tag, second_score = ranked[1]
        if second_score >= top_score * 0.50:
            second_display = REGION_TAG_DISPLAY_NAMES.get(
                second_tag, second_tag.replace("_", " ").title()
            )
            return f"{top_display} & {second_display}", top_display

    return top_display, top_display


def _deduplicate_labels(regions, max_suffixes=1):
    """Post-process region labels: append tag descriptors where labels collide.

    Uses tag suffixes first, then falls back to spatial direction (N/S/E/W)
    when clusters share identical tags.

    Args:
        max_suffixes: max number of " — Tag" suffixes to append (2 for L0, 3 for L1)
    """
    # Pass 1: tag-based disambiguation
    for pass_num in range(max_suffixes):
        label_groups = {}
        for r in regions:
            label_groups.setdefault(r["label"], []).append(r)

        changed = False
        for label, group in label_groups.items():
            if len(group) <= 1:
                continue
            for r in group:
                top_tags = r.get("top_tags", [])
                if not top_tags:
                    continue
                for tag in top_tags:
                    tag_display = REGION_TAG_DISPLAY_NAMES.get(tag)
                    if tag_display and tag_display not in r["label"]:
                        r["label"] = f"{r['label']} — {tag_display}"
                        if r.get("level") == 1 and " — " in r["label"]:
                            parts = r["label"].split(" — ")
                            r["short"] = parts[-1]
                        changed = True
                        break
        if not changed:
            break

    # Pass 2: spatial direction for any remaining collisions
    label_groups = {}
    for r in regions:
        label_groups.setdefault(r["label"], []).append(r)
    for label, group in label_groups.items():
        if len(group) <= 1:
            continue
        # Compute group centroid for relative positioning
        gcx = sum(r["cx"] for r in group) / len(group)
        gcy = sum(r["cy"] for r in group) / len(group)
        for r in group:
            dx = r["cx"] - gcx
            dy = r["cy"] - gcy
            # Use primary axis direction
            if abs(dx) > abs(dy):
                direction = "East" if dx > 0 else "West"
            else:
                direction = "North" if dy > 0 else "South"
            r["label"] = f"{r['label']} ({direction})"


# ── Core clustering ─────────────────────────────────────────────────────


def compute_centroid(xs, ys):
    """Compute the centroid of a set of points."""
    return float(np.mean(xs)), float(np.mean(ys))


def compute_span(xs, ys):
    """Bounding-box width, height, and the max of the two.

    `span` alone drives the viz's label culling and is what callers usually want.
    Width and height are returned beside it because collapsing them threw away the
    one thing that distinguishes a filament from a blob: a 20x1 streak and a 20x20
    cloud used to serialise identically, so nothing downstream could tell a road
    from a region.
    """
    width = float(np.max(xs) - np.min(xs))
    height = float(np.max(ys) - np.min(ys))
    return max(width, height), width, height


def assign_parents(l0_regions, l1_regions):
    """Assign each L1 region a parent L0 region based on centroid proximity."""
    if not l0_regions or not l1_regions:
        return

    for l1 in l1_regions:
        cx, cy = l1["cx"], l1["cy"]
        # Find closest L0 centroid
        best_dist = float("inf")
        best_parent = None
        for l0 in l0_regions:
            dx = cx - l0["cx"]
            dy = cy - l0["cy"]
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best_parent = l0["id"]
        l1["parent"] = best_parent


def cluster_within(coords, labels_parent, min_cluster_size, min_samples, min_parent_size):
    """Sub-cluster INSIDE each parent cluster. Returns a label array over all points.

    Clustering globally at a smaller `min_cluster_size` and hoping the result lands
    inside the parents is not the same thing: it nests by luck. Running the clusterer
    separately on each parent's own points makes containment structural — a child
    cannot span two parents because it never sees the other parent's points.

    Parents below `min_parent_size` are left alone: a 100-card region is already a
    neighbourhood, and splitting it produces names nobody needs.
    """
    out = np.full(len(coords), -1, dtype=int)
    next_id = 0
    for parent in sorted(set(labels_parent)):
        if parent == -1:
            continue
        idx = np.where(labels_parent == parent)[0]
        if len(idx) < min_parent_size:
            continue
        sub = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples,
        ).fit_predict(coords[idx])
        for local in sorted(set(sub)):
            if local == -1:
                continue
            out[idx[sub == local]] = next_id
            next_id += 1
    return out


def fill_nearest(coords, labels, centroids):
    """A copy of `labels` with every noise point snapped to the nearest centroid.

    Kept SEPARATE from the real membership on purpose. The truthful answer to "which
    region is this card in" is sometimes "none" — a quarter of the cards sit in the
    thin space between clusters, and this file has always said so rather than inventing
    a home for them. But a country/state/neighbourhood *address* has to be total: you
    cannot zoom in and be told you are nowhere.

    So both are published. `membership` is what HDBSCAN found; `nearest` is where a card
    would post its letters. A consumer that conflates them is claiming more than the
    clustering supports.
    """
    if not centroids:
        return [int(v) for v in labels]
    ids = sorted(centroids)
    pts = np.array([centroids[c] for c in ids], dtype=np.float64)
    out = np.asarray(labels).copy()
    noise = np.where(out == -1)[0]
    if len(noise):
        d = ((coords[noise][:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
        out[noise] = np.array(ids)[d.argmin(axis=1)]
    return [int(v) for v in out]


def load_region_names():
    """Hand-authored names, keyed by content signature. Absent file is not an error."""
    if not REGION_NAMES_PATH.exists():
        return {}
    with open(REGION_NAMES_PATH) as f:
        return json.load(f).get("names", {})


def apply_region_names(regions, map_type, names):
    """Swap in the hand-authored name where the signature matches.

    The mechanical label is kept as `mechanical` — it is what the signature is built
    from, so losing it would make the names file unmaintainable, and it is the honest
    description of what the cluster actually contains.

    Returns the regions that found no name, so the caller can report them. A silent
    fallback is how a re-cluster quietly reverts the whole map to machine names.
    """
    unmatched = []
    for r in regions:
        key = f"{map_type}|{r['level']}|{r['label']}"
        hit = names.get(key)
        if not hit:
            unmatched.append(r)
            continue
        r["mechanical"] = r["label"]
        r["label"] = hit[0]
        r["short"] = hit[1] if len(hit) > 1 else hit[0]
    return unmatched


def name_neighbourhoods(l2_regions, by_id):
    """Neighbourhoods are named FROM THEIR PARENT, procedurally.

    410 of them across both maps. Hand-authoring that many produces a long tail of bad
    jokes; deriving them keeps every name true and lets the parent's authored character
    carry down — 'Swolesville' subdivides into 'Swolesville — Counters', not into a
    fresh invention that has to be remembered separately.
    """
    # Deduped HERE rather than by `_deduplicate_labels`, which runs before naming and so
    # sees the mechanical labels, not these. Two neighbourhoods of one parent can easily
    # share a top tag; without this the map shows the same name twice a few pixels apart.
    used = Counter()
    for r in l2_regions:
        parent = by_id.get(r.get("parent"))
        base = parent["short"] if parent else r["short"]
        tag = (r.get("top_tags") or [None])[0]
        detail = REGION_TAG_DISPLAY_NAMES.get(tag, tag) if tag else None
        label = f"{base} — {detail}" if detail else base
        used[label] += 1
        if used[label] > 1:
            label = f"{label} {used[label]}"
        r["mechanical"] = r["label"]
        r["label"] = label
        r["short"] = f"{detail} {used[label.rsplit(' ', 1)[0]]}" if False else (detail or base)
        r["short"] = label.split(" — ", 1)[-1] if " — " in label else label


def build_regions(level, labels, xs, ys, projection_data, proj_tags,
                  map_type, global_tag_freq):
    """Region records for one clustering level. One loop, three levels.

    This body existed twice, character for character apart from the level and the id
    prefix, and adding neighbourhoods would have made it three. Every field a region
    carries — centroid, span, count, top tags, the generated name — is derived the same
    way whatever level it sits at.
    """
    out = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        mask = labels == cluster_id
        cluster_xs, cluster_ys = xs[mask], ys[mask]
        cx, cy = compute_centroid(cluster_xs, cluster_ys)
        span, width, height = compute_span(cluster_xs, cluster_ys)
        count = int(mask.sum())

        indices = np.where(mask)[0]
        cluster_colors = [projection_data[i]["c"] for i in indices]
        cluster_types = [projection_data[i]["s"] for i in indices]
        cluster_tags = [proj_tags[i] for i in indices]

        if map_type == "default":
            label, short = name_cluster_colortype(cluster_colors, cluster_types)
        else:
            label, short = name_cluster_ability(
                cluster_tags, global_tag_freq, cluster_colors, cluster_types
            )

        tag_counts, _ = _count_cluster_tags(cluster_tags)
        top_tags = [
            tg for tg, c in tag_counts.most_common()
            if c / count >= REGION_MIN_TAG_PRESENCE
        ][:3]

        out.append({
            "id": f"l{level}_{cluster_id}",
            "level": level,
            "label": label,
            "short": short,
            "cx": round(cx, 2),
            "cy": round(cy, 2),
            "span": round(span, 1),
            "w": round(width, 1),
            "h": round(height, 1),
            "count": count,
            "top_tags": top_tags,
        })
    return out


def cluster_map(projection_data, cards_df, map_type, output_path):
    """Run HDBSCAN clustering on a single map projection.

    Args:
        projection_data: list of dicts from projection JSON (with x, y, n, c, s, etc.)
        cards_df: DataFrame from cards.csv (for mechanical_tags column)
        map_type: 'default' or 'ability'
        output_path: Path to write the output JSON
    """
    # Build arrays
    xs = np.array([d["x"] for d in projection_data], dtype=np.float64)
    ys = np.array([d["y"] for d in projection_data], dtype=np.float64)
    coords = np.column_stack([xs, ys])

    # Direct index alignment: projection[i] corresponds to cards_df row i
    # (both built in the same order through embed.py → reduce.py)
    tags_col = cards_df["mechanical_tags"].fillna("").values
    proj_tags = [str(tags_col[i]) for i in range(len(projection_data))]

    # Global tag frequencies — the IDF half of the TF-IDF used to name ability-map
    # regions. Recomputed per map; deduplication does not use it.
    global_tag_freq = {}
    total_cards = len(proj_tags)
    tag_counts = Counter()
    for tags_str in proj_tags:
        if not tags_str:
            continue
        for tag in tags_str.split(","):
            tag = tag.strip()
            if tag:
                tag_counts[tag] += 1
    for tag, count in tag_counts.items():
        global_tag_freq[tag] = count / total_cards

    # Run HDBSCAN at two levels
    print(f"  Clustering L0 (min_cluster_size={REGION_L0_MIN_CLUSTER_SIZE})...")
    clusterer_l0 = hdbscan.HDBSCAN(
        min_cluster_size=REGION_L0_MIN_CLUSTER_SIZE,
        min_samples=REGION_L0_MIN_SAMPLES,
    )
    labels_l0 = clusterer_l0.fit_predict(coords)
    n_l0 = len(set(labels_l0)) - (1 if -1 in labels_l0 else 0)
    print(f"  L0: {n_l0} clusters ({(labels_l0 == -1).sum()} noise points)")

    print(f"  Clustering L1 (min_cluster_size={REGION_L1_MIN_CLUSTER_SIZE})...")
    clusterer_l1 = hdbscan.HDBSCAN(
        min_cluster_size=REGION_L1_MIN_CLUSTER_SIZE,
        min_samples=REGION_L1_MIN_SAMPLES,
    )
    labels_l1 = clusterer_l1.fit_predict(coords)
    n_l1 = len(set(labels_l1)) - (1 if -1 in labels_l1 else 0)
    print(f"  L1: {n_l1} clusters ({(labels_l1 == -1).sum()} noise points)")

    print(f"  Clustering L2 within each L1 (min_cluster_size={REGION_L2_MIN_CLUSTER_SIZE})...")
    labels_l2 = cluster_within(
        coords, labels_l1,
        REGION_L2_MIN_CLUSTER_SIZE, REGION_L2_MIN_SAMPLES, REGION_L2_MIN_PARENT_SIZE,
    )
    n_l2 = len(set(labels_l2)) - (1 if -1 in labels_l2 else 0)
    print(f"  L2: {n_l2} neighbourhoods ({(labels_l2 == -1).sum()} not in one)")

    # Build region data — one loop, three levels (see build_regions).
    args = (xs, ys, projection_data, proj_tags, map_type, global_tag_freq)
    l0_regions = build_regions(0, labels_l0, *args)
    l1_regions = build_regions(1, labels_l1, *args)
    l2_regions = build_regions(2, labels_l2, *args)
    regions = list(l0_regions)

    # Assign parents. L1→L0 is a proximity match; L2→L1 is exact by construction,
    # since a neighbourhood was clustered from its parent's own points.
    assign_parents(l0_regions, l1_regions)
    assign_parents(l1_regions, l2_regions)
    regions.extend(l1_regions)
    regions.extend(l2_regions)

    # Deduplicate labels — append tag descriptors where names collide.
    # These mutate the region dicts in place, and `regions` holds the same
    # objects, so the renames propagate without reassignment. L0 and L1 are
    # deduped separately by design: a level-0 and a level-1 region may share
    # a label because they are drawn at different zooms.
    _deduplicate_labels(l0_regions, max_suffixes=2)
    _deduplicate_labels(l1_regions, max_suffixes=3)
    _deduplicate_labels(l2_regions, max_suffixes=3)

    # Names LAST, after dedup — the signature is built from the deduplicated mechanical
    # label, so "Blue Creatures — Flyers — ETB (East)" and its West twin are two distinct
    # keys rather than one ambiguous one.
    names = load_region_names()
    unmatched = apply_region_names(l0_regions, map_type, names)
    unmatched += apply_region_names(l1_regions, map_type, names)
    by_id = {r["id"]: r for r in l0_regions + l1_regions}
    name_neighbourhoods(l2_regions, by_id)
    if unmatched:
        print(f"  {len(unmatched)} region(s) have no hand-authored name and kept the "
              f"machine one — add them to {REGION_NAMES_PATH.name}:")
        for r in unmatched[:6]:
            print(f"      {map_type}|{r['level']}|{r['label']}")
        if len(unmatched) > 6:
            print(f"      … and {len(unmatched) - 6} more")

    # Build output.
    #
    # `membership` is the pair of HDBSCAN label arrays, one entry per card in
    # cards.csv row order, -1 for noise. They were being computed and thrown
    # away, which left nothing anywhere in the repo able to answer "which region
    # is this card in" — the viz could draw a region's name but not its members.
    # ~34K small ints; the file goes from ~25 KB to ~100 KB and stays tracked.
    #
    # Cluster ids index the regions by `id`: label 3 at L0 is the region with
    # id "l0_3". Noise is a real answer, not a gap — 29% of cards belong to no
    # L0 region at all, and the honest thing is to say so rather than snap them
    # to a nearest centroid they were never clustered into.
    # `nearest` is NOT membership and must never be read as it. See `fill_nearest`:
    # membership is what the clusterer found, noise included, because "this card sits in
    # the thin space between clusters" is a true and useful answer. But an ADDRESS has to
    # be total — you cannot zoom in and be told you are nowhere — so the snapped version
    # is published beside it under a name that cannot be mistaken for the real one.
    centroids_l0 = {int(r["id"].split("_")[1]): (r["cx"], r["cy"]) for r in l0_regions}
    centroids_l1 = {int(r["id"].split("_")[1]): (r["cx"], r["cy"]) for r in l1_regions}

    output = {
        "meta": {
            "map": map_type,
            "card_count": len(projection_data),
            "l0_count": n_l0,
            "l1_count": n_l1,
            "l2_count": n_l2,
        },
        "regions": regions,
        "membership": {
            "l0": [int(v) for v in labels_l0],
            "l1": [int(v) for v in labels_l1],
            "l2": [int(v) for v in labels_l2],
        },
        "nearest": {
            "l0": fill_nearest(coords, labels_l0, centroids_l0),
            "l1": fill_nearest(coords, labels_l1, centroids_l1),
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_kb = output_path.stat().st_size / 1024
    print(f"  Wrote {output_path} ({size_kb:.1f} KB, "
          f"{n_l0} L0 + {n_l1} L1 + {n_l2} L2 regions)")


def main(space=None):
    if space is not None:
        from manamap import spaces as space_registry

        target = space_registry.get(space)
        if target.regions is None or target.projection is None:
            raise SystemExit(f"the {target.slug!r} space has no regions")
        if not target.projection.exists():
            raise SystemExit(
                f"{target.projection} not found — run `manamap reduce "
                f"--space {target.slug}` first")
        cards_df = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
        print(f"\n  Processing {target.label} map...")
        with open(target.projection, "r") as fh:
            projection_data = json.load(fh)
        # The map NAME is the space slug for a new space; `default` and `ability`
        # are kept for the two that already have named regions on disk.
        cluster_map(projection_data, cards_df, target.slug, target.regions)
        return
    return _main_all()


def _main_all():
    cards_df = pd.read_csv(OUTPUT_CSV_PATH)
    print(f"  Loaded {len(cards_df)} cards from {OUTPUT_CSV_PATH}")

    # Default (Color+Type) map
    if PROJECTION_PATH.exists():
        print(f"\n  Processing Color+Type map...")
        with open(PROJECTION_PATH, "r") as f:
            projection_data = json.load(f)
        cluster_map(projection_data, cards_df, "default", REGIONS_DEFAULT_PATH)
    else:
        print(f"  Skipping Color+Type map ({PROJECTION_PATH} not found)")

    # Ability map
    if ABILITY_PROJECTION_PATH.exists():
        print(f"\n  Processing Abilities map...")
        with open(ABILITY_PROJECTION_PATH, "r") as f:
            projection_data = json.load(f)
        cluster_map(projection_data, cards_df, "ability", REGIONS_ABILITY_PATH)
    else:
        print(f"  Skipping Abilities map ({ABILITY_PROJECTION_PATH} not found)")


if __name__ == "__main__":
    main()
