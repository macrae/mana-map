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
    REGION_COLOR_DOMINANCE,
    REGION_TYPE_DOMINANCE,
    REGION_TAG_DISPLAY_NAMES,
    REGION_COLOR_DISPLAY_NAMES,
    REGION_GUILD_NAMES,
    REGION_MIN_TAG_PRESENCE,
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
    """Compute the span (max of width, height) of a bounding box."""
    width = float(np.max(xs) - np.min(xs))
    height = float(np.max(ys) - np.min(ys))
    return max(width, height)


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

    # Build region data
    regions = []

    # Process L0 clusters
    for cluster_id in sorted(set(labels_l0)):
        if cluster_id == -1:
            continue
        mask = labels_l0 == cluster_id
        cluster_xs = xs[mask]
        cluster_ys = ys[mask]
        cx, cy = compute_centroid(cluster_xs, cluster_ys)
        span = compute_span(cluster_xs, cluster_ys)
        count = int(mask.sum())

        # Gather metadata for naming
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

        # Top tags for this cluster (only tags with >= minimum presence)
        tag_counts, _ = _count_cluster_tags(cluster_tags)
        top_tags = [
            t for t, c in tag_counts.most_common()
            if c / count >= REGION_MIN_TAG_PRESENCE
        ][:3]

        regions.append({
            "id": f"l0_{cluster_id}",
            "level": 0,
            "label": label,
            "short": short,
            "cx": round(cx, 2),
            "cy": round(cy, 2),
            "span": round(span, 1),
            "count": count,
            "top_tags": top_tags,
        })

    # Process L1 clusters
    l0_regions = [r for r in regions if r["level"] == 0]
    l1_regions = []

    for cluster_id in sorted(set(labels_l1)):
        if cluster_id == -1:
            continue
        mask = labels_l1 == cluster_id
        cluster_xs = xs[mask]
        cluster_ys = ys[mask]
        cx, cy = compute_centroid(cluster_xs, cluster_ys)
        span = compute_span(cluster_xs, cluster_ys)
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
            t for t, c in tag_counts.most_common()
            if c / count >= REGION_MIN_TAG_PRESENCE
        ][:3]

        region = {
            "id": f"l1_{cluster_id}",
            "level": 1,
            "label": label,
            "short": short,
            "cx": round(cx, 2),
            "cy": round(cy, 2),
            "span": round(span, 1),
            "count": count,
            "top_tags": top_tags,
        }
        l1_regions.append(region)

    # Assign parents
    assign_parents(l0_regions, l1_regions)
    regions.extend(l1_regions)

    # Deduplicate labels — append tag descriptors where names collide.
    # These mutate the region dicts in place, and `regions` holds the same
    # objects, so the renames propagate without reassignment. L0 and L1 are
    # deduped separately by design: a level-0 and a level-1 region may share
    # a label because they are drawn at different zooms.
    _deduplicate_labels(l0_regions, max_suffixes=2)
    _deduplicate_labels(l1_regions, max_suffixes=3)

    # Build output
    output = {
        "meta": {
            "map": map_type,
            "card_count": len(projection_data),
            "l0_count": n_l0,
            "l1_count": n_l1,
        },
        "regions": regions,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_kb = output_path.stat().st_size / 1024
    print(f"  Wrote {output_path} ({size_kb:.1f} KB, {n_l0} L0 + {n_l1} L1 regions)")


def main():
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
