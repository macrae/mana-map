"""Pilot: a deck's own constellation — its 100 cards, laid out and clustered.

THE POINT. The 34,890-card atlas answers "where does this card sit in Magic". It
cannot answer "what shape is my deck", because a hundred cards scattered across
the whole map are dust: the structure that matters — which cards do the same job,
which cluster is thin, which card lives on its own out at the edge — is exactly
the structure a global projection compressed out to preserve global shape.

So this re-lays-out the deck ALONE, from the 128-d ability embeddings, and
clusters the result into cities and neighbourhoods. It is `viz/js/drill.js`'s
argument applied to a decklist, and it produces the picture a deck page can open
on: the engine, visible before a word of prose.

WHICH EMBEDDING SPACE, AND WHY IT IS NOT A CHOICE. `embeddings_ability.npy` — the
FUNCTION space. `embeddings.npy` is the layout space and knows only colour and
type, so clustering a mono-green deck in it produces one green blob and a land
pile, which is a true statement about nothing. Every similarity question in this
repo reads the ability space; this is one more.

DETERMINISM. The deck page rebuilds byte-identically, so the layout may not wobble
between runs. Classical MDS gives a fixed starting configuration from the
eigendecomposition, SMACOF refines it, and the result is then canonically
oriented (see `_orient`) — because a rotation or a reflection is free in MDS and
would otherwise redraw the whole map on a rerun that changed nothing.

TRACKED OUTPUT, on purpose. `embeddings_ability.npy` is gitignored, so a fresh
clone cannot compute this. `deck_map.json` is committed for the same reason the
projections are: the renderer must work without a pipeline run.

A DECK MAP POSITION IS LOCAL. It is not the card's position on the atlas, and the
two coordinate systems mean different things — the same honesty constraint
drill.js carries at the top of its file. Anything that draws this must say so.
"""

import json
import math

import numpy as np

from manamap.config import ABILITY_EMBEDDINGS_PATH
from manamap.pilot.build_index import line_cards
from manamap.pilot.card_pool import load_frame
from manamap.pilot.common import (
    deck_dir,
    expand_faces,
    load_card_roles,
    load_deck_cards,
    presentable,
    report_errors,
    resolve_out_path,
)

ARTIFACT = "deck_map.json"

# Cities and neighbourhoods, sized for a printed deck page rather than for a
# statistic. Five to seven named regions is what a reader can hold; twelve is a
# legend they skip. Deck sizes here run 60–100 distinct cards (basics collapse to
# one entry, which is right for a map — eleven Forests are one place).
CARDS_PER_CITY = 14
MIN_CITIES, MAX_CITIES = 4, 7
CARDS_PER_NEIGHBOURHOOD = 8
# No city may hold more than this share of the deck, or the legend stops
# describing anything. Measured against Ward on real decks — see `cluster`.
MAX_CITY_SHARE = 0.35
MAX_NEIGHBOURHOODS = 4

# Role → the plain word a city gets called before an agent gives it a better one.
# Deliberately coarse: this is the deterministic fallback, and a fallback that
# guesses precisely is worse than one that is obviously a placeholder.
ROLE_FAMILY = {
    "ramp": "Ramp", "draw": "Card Flow", "removal": "Interaction",
    "counter": "Interaction", "protection": "Insurance", "recursion": "Recursion",
    "tutor": "Tutors", "wincon": "Finishers", "threat": "Bodies",
    "payoff": "Payoffs", "piece": "Engine Pieces", "land": "Mana",
    "cost": "Discounts", "value": "Value", "sac": "Sacrifice",
    "token": "Width", "anthem": "Anthems", "stax": "Taxes",
}


# ── Resolving the deck into the corpus ──────────────────────────────────


def _name_index(frame):
    """corpus name → row. DFC faces resolve to their full `A // B` row, because a
    decklist writes "Bala Ged Recovery" and cards.csv carries the joined name."""
    index = {}
    for row, name in enumerate(frame["name"].tolist()):
        index.setdefault(name, row)
        for face in expand_faces(name):
            index.setdefault(face, row)
    return index


def resolve_rows(cards, frame):
    """(rows, names, unresolved). One point per distinct card — copies collapse."""
    index = _name_index(frame)
    rows, names, missing = [], [], []
    for card in cards:
        name = card["name"]
        row = index.get(name)
        if row is None:
            row = index.get(name.split(" // ")[0].strip())
        if row is None:
            missing.append(name)
            continue
        rows.append(row)
        names.append(name)
    return rows, names, missing


# ── Layout ──────────────────────────────────────────────────────────────


def cosine_distances(vectors):
    unit = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)
    similarity = np.clip(unit @ unit.T, -1.0, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return distance


def _classical_mds(distance, dims=2):
    """PCoA — the deterministic starting configuration.

    Double-centre the squared distances, take the top eigenvectors. No iteration,
    no seed, and the same input always gives the same output up to sign, which
    `_orient` then pins down.
    """
    n = distance.shape[0]
    squared = distance ** 2
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ squared @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:dims]
    keep = np.clip(values[order], 0, None)
    return vectors[:, order] * np.sqrt(keep)


def _orient(points):
    """Pin rotation, reflection and scale, so a rerun draws the same picture.

    MDS is indifferent to all three: the same deck can come back mirrored, and the
    deck page would render a different map from identical inputs. Principal axis
    to +x, the half with more mass to the right, the half with more mass on top,
    then scale to a unit box.
    """
    centred = points - points.mean(axis=0)
    _u, _s, basis = np.linalg.svd(centred, full_matrices=False)
    rotated = centred @ basis.T
    if rotated[:, 0].sum() < 0 or (rotated[:, 0].sum() == 0 and rotated[0, 0] < 0):
        rotated[:, 0] *= -1
    if rotated[:, 1].sum() < 0 or (rotated[:, 1].sum() == 0 and rotated[0, 1] < 0):
        rotated[:, 1] *= -1
    span = np.abs(rotated).max()
    return rotated / (span if span > 1e-9 else 1.0)


def layout(distance):
    """Classical MDS, refined by SMACOF, oriented. Deterministic end to end."""
    start = _classical_mds(distance)
    try:
        from sklearn.manifold import smacof
        points, _stress = smacof(
            distance, n_components=2, init=start, n_init=1,
            random_state=0, normalized_stress=False, eps=1e-9, max_iter=400,
        )
    except Exception:                       # sklearn absent or signature drift
        points = start                      # PCoA alone is a usable layout
    return _orient(np.asarray(points, dtype=float))


# ── Cities and neighbourhoods ───────────────────────────────────────────


def _cut(unit, members, n_clusters):
    """Ward-linkage agglomerative cut over a subset of the UNIT-NORMALISED vectors.

    Clustered in the 128-d space rather than in the 2-D picture: the projection is
    a compromise, and clustering its compromises would name the compromise. The
    picture is where clusters are DRAWN, not where they are decided.

    WARD, and this was measured rather than assumed. Average linkage on cosine
    distance chains: on radagast it returned one city holding 37 of 71 cards and
    another holding exactly 1 — a map whose legend says "half the deck" is not a
    legend. Ward minimises within-cluster variance, so it splits the mass instead
    of growing one blob, and on unit-normalised vectors squared euclidean distance
    is a monotone function of cosine distance — the same geometry, in the metric
    Ward is defined for.

    Agglomerative rather than the HDBSCAN the atlas uses: at 34,890 points density
    estimation is meaningful and "noise" is a real answer; at 71 points it labels
    most of the deck noise, which on a printed page is a map with a hole in it.
    Every card belongs to a city here — that is the contract this page rests on.
    """
    if len(members) <= 1 or n_clusters <= 1:
        return [0] * len(members)
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(
        n_clusters=min(n_clusters, len(members)), linkage="ward")
    return model.fit_predict(unit[members]).tolist()


def cluster(unit, n):
    """(city_of[i], neighbourhood_of[i]) — a two-level partition, every card placed.

    The city count is chosen by BALANCE, not by a divisor. Ward on radagast, by k:

        k=4  largest 38 (54%)      k=7  largest 23 (32%)
        k=5  largest 38 (54%)      k=8  largest 19 (27%)
        k=6  largest 30 (42%)      k=9  largest 10 (14%)

    A fixed cards-per-city rule picked k=5 and produced a map whose biggest region
    was over half the deck — technically a partition, useless as a legend. So: grow
    k until the largest city is at most `MAX_CITY_SHARE`, and stop at MAX_CITIES
    because past seven regions a reader is consulting a key instead of seeing a
    shape. Both bounds bind on real decks, which is why both exist.
    """
    lo = max(MIN_CITIES, min(MAX_CITIES, round(n / CARDS_PER_CITY)))
    cities = _cut(unit, list(range(n)), min(lo, n))
    for k in range(lo + 1, min(MAX_CITIES, n) + 1):
        largest = max(cities.count(c) for c in set(cities))
        if largest <= MAX_CITY_SHARE * n:
            break
        cities = _cut(unit, list(range(n)), k)

    hoods = [0] * n
    for city in sorted(set(cities)):
        members = [i for i, c in enumerate(cities) if c == city]
        want = max(1, min(MAX_NEIGHBOURHOODS,
                          math.ceil(len(members) / CARDS_PER_NEIGHBOURHOOD)))
        for local, label in zip(members, _cut(unit, members, want)):
            hoods[local] = label
    return cities, hoods


def name_cluster(names, roles_by_name):
    """The deterministic placeholder name: the dominant role family, or Mixed.

    An agent replaces this with something functional and witty. Until it does, the
    label says what the cluster IS rather than pretending to a voice — a
    placeholder that reads like finished copy is how a placeholder ships.
    """
    tally = {}
    for name in names:
        for role in roles_by_name.get(name, []):
            family = ROLE_FAMILY.get(role.split(":", 1)[0])
            if family:
                tally[family] = tally.get(family, 0) + 1
    if not tally:
        return ["Mixed"]
    # RANKED, not just the winner. Two cities in a creature deck both come back
    # "Bodies" from the top role alone, and a map with the same name printed twice
    # is worse than one with a vaguer name — the reader assumes they misread it.
    # The caller walks this list to find a name nothing else has taken.
    return [k for k, _ in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))]


def unique_names(ranked_lists):
    """Assign each cluster the best name no earlier cluster already took.

    Greedy in emit order, which is descending size — so the biggest city keeps its
    truest name and a smaller one moves to its second role. Falls back to
    "<name> II" only when a cluster has no unclaimed role at all, which is honest:
    the two really are the same kind of place.
    """
    out, taken = [], set()
    for ranked in ranked_lists:
        pick = next((r for r in ranked if r not in taken), None)
        if pick is None:
            base = ranked[0] if ranked else "Mixed"
            pick, n = f"{base} II", 2
            while pick in taken:
                n += 1
                pick = f"{base} {'I' * n}"
        taken.add(pick)
        out.append(pick)
    return out


# ── Evidence overlay ────────────────────────────────────────────────────


def near_edges(distance, k=2):
    """Each card's k nearest neighbours IN THIS DECK, deduped and undirected.

    This is the "graph structure" half of the picture, and it is computed here
    rather than read from `neighbours.bin` for one reason: that table's neighbours
    are the nearest cards in the whole 34,890-card corpus, and for a deck card they
    are almost never other cards in the deck. The interesting question on this page
    is which cards in YOUR NINETY-NINE do the same job — a within-deck k-NN, which
    only exists once the deck has been isolated.

    k=2 on purpose. At k=4 a 71-card deck draws ~200 edges and the cities disappear
    under them; at k=2 the edges trace the spine of each cluster and the shape
    survives. The renderer is free to draw fewer.
    """
    n = distance.shape[0]
    seen, edges = set(), []
    order = np.argsort(distance, axis=1)
    for i in range(n):
        for j in order[i][1:k + 1]:
            j = int(j)
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            edges.append({"a": key[0], "b": key[1],
                          "d": round(float(distance[i][j]), 4)})
    return edges


def verified_cards(slug, deck_names):
    """Cards a checker-passed stack's scenario actually names.

    Reuses `build_index.line_cards`, which already encodes the hard-won rule for
    when a `board` entry is a line piece and when it is furniture. Re-deriving
    that here would be a second answer to a question this repo has settled once.
    """
    found = set()
    directory = deck_dir(slug) / "stacks"
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*.json")):
        doc = json.loads(path.read_text())
        if not presentable(doc):
            continue
        found.update(line_cards(doc.get("scenario") or {}, deck_names))
    return found


# ── Build ───────────────────────────────────────────────────────────────


def build(slug):
    deck = load_deck_cards(slug)
    cards = deck["cards"]
    frame = load_frame()
    rows, names, missing = resolve_rows(cards, frame)
    if len(rows) < 4:
        raise SystemExit(
            f"{slug}: only {len(rows)} of {len(cards)} cards resolved against the "
            f"corpus — run `manamap extract` first, or check for renamed cards. "
            f"A map of four points is not a map.")

    vectors = np.load(ABILITY_EMBEDDINGS_PATH)[rows]
    unit = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)
    distance = cosine_distances(vectors)
    points = layout(distance)
    cities, hoods = cluster(unit, len(rows))

    roles_by_name = load_card_roles()
    deck_names = {c["name"] for c in cards}
    verified = verified_cards(slug, deck_names)
    # The flag lives on the CARD, not on the deck header — cards.json's top level
    # is just {deck, decklist_sha256}. Reading a "commander" key off the header
    # returns None silently and the map loses its centre.
    commanders = {c["name"] for c in cards if c.get("is_commander")}

    entries = []
    for i, name in enumerate(names):
        entries.append({
            "name": name,
            "x": round(float(points[i][0]), 5),
            "y": round(float(points[i][1]), 5),
            "city": int(cities[i]),
            "hood": int(hoods[i]),
            "roles": roles_by_name.get(name, []),
            "verified": name in verified,
            "commander": name in commanders,
        })

    regions = []
    for city in sorted(set(cities)):
        members = [i for i, c in enumerate(cities) if c == city]
        member_names = [names[i] for i in members]
        centroid = points[members].mean(axis=0)
        regions.append({
            "id": f"city-{city}",
            "level": 0,
            "label": None,                       # an agent names these
            "fallback": None,          # filled by unique_names below
            "cx": round(float(centroid[0]), 5),
            "cy": round(float(centroid[1]), 5),
            "count": len(members),
            "cards": sorted(member_names),
            "verified_count": sum(1 for n in member_names if n in verified),
        })
        for hood in sorted({hoods[i] for i in members}):
            sub = [i for i in members if hoods[i] == hood]
            sub_names = [names[i] for i in sub]
            sub_centroid = points[sub].mean(axis=0)
            regions.append({
                "id": f"city-{city}-hood-{hood}",
                "level": 1,
                "parent": f"city-{city}",
                "label": None,
                "fallback": None,
                "cx": round(float(sub_centroid[0]), 5),
                "cy": round(float(sub_centroid[1]), 5),
                "count": len(sub),
                "cards": sorted(sub_names),
                "verified_count": sum(1 for n in sub_names if n in verified),
            })

    # Names are assigned ACROSS clusters, not per cluster, so no two share one.
    # Cities and neighbourhoods are namespaced separately: a neighbourhood called
    # "Ramp" inside a city called "Mana" is fine and reads correctly.
    for level in (0, 1):
        peers = [r for r in regions if r["level"] == level]
        peers.sort(key=lambda r: (-r["count"], r["id"]))
        ranked = [name_cluster(r["cards"], roles_by_name) for r in peers]
        for region, chosen in zip(peers, unique_names(ranked)):
            region["fallback"] = chosen

    return {
        "slug": slug,
        "decklist_sha256": deck.get("decklist_sha256"),
        "meta": {
            "space": "embeddings_ability.npy",
            "projection": "classical MDS + SMACOF, canonically oriented",
            "clustering": "average-linkage agglomerative on cosine distance, 2 levels",
            "cards": len(entries),
            "unresolved": missing,
            "note": ("Positions are LOCAL to this deck and are not atlas positions — "
                     "the deck is re-laid-out from its own cards, which is the whole "
                     "point. Anything that draws this must say so."),
        },
        "cards": entries,
        "regions": regions,
        "edges": near_edges(distance),
    }


def main(args):
    doc = build(args.slug)
    errors = []
    if doc["meta"]["unresolved"]:
        errors.append(f"unresolved card names: {doc['meta']['unresolved']}")
    # `resolve_out_path` stringifies whatever it is handed, so passing None writes a
    # file literally called "None" into the deck directory. Only ask it when the
    # caller actually chose a path; the default is the tracked artifact.
    requested = getattr(args, "out", None)
    out = (resolve_out_path(requested, args.slug, "deck-map") if requested
           else deck_dir(args.slug) / ARTIFACT)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")

    cities = [r for r in doc["regions"] if r["level"] == 0]
    hoods = [r for r in doc["regions"] if r["level"] == 1]
    print(f"OK   {args.slug} — {doc['meta']['cards']} cards, {len(cities)} cities, "
          f"{len(hoods)} neighbourhoods -> {out}")
    for city in cities:
        kids = [h for h in hoods if h.get("parent") == city["id"]]
        print(f"  {city['id']:10s} {city['count']:3d} cards  "
              f"[{city['fallback']}]  {len(kids)} neighbourhood(s)"
              + (f"  ✓{city['verified_count']}" if city["verified_count"] else ""))
    report_errors(f"deck map for {args.slug}", errors)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot deck-map <slug>`.")
