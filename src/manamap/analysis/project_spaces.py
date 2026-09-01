"""Project every embedding space to 2D (and 3D) so they can be LOOKED AT.

## WHY THIS IS PART OF EVALUATION AND NOT A GARNISH

`eval-embeddings` asks one question — are the k nearest cards the right ones —
and the numbers it produces do not describe a MAP. Measured on the same five
spaces, the disagreement is stark:

    space       spread   effdim   centroid headroom   hard-neg sep    r@10
    cardbert    0.1347    16.72               0.976         0.0377   0.103
    vae         0.0454     5.71               0.092         0.0064   0.167

The VAE retrieves better and maps worse: a third the spread, a tenth the
headroom, everything piled into a narrow cone. A space can win recall@10 by
concentrating and lose everything that makes an atlas navigable. So the
projection is run for every space, side by side, and looked at.

## WHAT THE COLOURING IS FOR

A projection coloured by the thing a space was TRAINED on flatters it. Each space
is therefore coloured by facts none of them optimised directly — colour identity,
card type, and EDHREC tribe — so the question becomes "did this structure emerge"
rather than "was this structure supplied".

Tribe is the interesting one: CardBERT beats the function space on theme at every
pool size, and if that is real it should be VISIBLE as separated tribal islands
rather than a number in a table.

## 3D IS CHEAP HERE AND THE RENDERER IS WHERE THE COST IS

PaCMAP takes `n_components=3` without complaint, so producing the coordinates is
a one-line change. What 3D actually costs is the frontend: `viz/render/canvas.js`
is 2D throughout — hit-testing, labels, the force graph. This module emits the 3D
coordinates so that decision can be made against a real artifact rather than in
the abstract.
"""

import json

import numpy as np

from manamap.config import DATA_DIR, OUTPUT_CSV_PATH

OUT_PATH = DATA_DIR / "eval" / "space_projections.json"

#: How many cards to project. The full corpus is 34,890 and PaCMAP handles it,
#: but five spaces at 3D each is the wall-clock cost — and a scatter of 12,000
#: points already shows every structure a scatter of 34,890 does.
SAMPLE = 12000
SEED = 42


def spaces():
    from manamap.analysis.eval_embeddings import spaces_on_disk

    return spaces_on_disk()


def project(matrix, components=2, seed=SEED):
    import pacmap

    reducer = pacmap.PaCMAP(n_components=components, random_state=seed)
    return reducer.fit_transform(matrix.astype(np.float32))


def _quantise(points):
    """2D floats -> int16 on a shared scale. 34,890 x 2 floats is 280 KB of JSON
    and 140 KB of int16, and a scatter plot cannot resolve more than that anyway."""
    points = np.asarray(points, dtype=np.float32)
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    scaled = (points - lo) / span * 2000.0 - 1000.0
    return scaled.round().astype(np.int16).tolist()


def card_facts(frame, rows):
    """The labels each projection is coloured by — none of them trained on."""
    from manamap.training import card_fields as CF

    facts = {"name": [], "identity": [], "type": [], "tribe": []}
    subtype_counts = {}
    for i in rows:
        card = frame.iloc[i].to_dict()
        for sub in CF.subtypes_of(card):
            subtype_counts[sub] = subtype_counts.get(sub, 0) + 1
    common = {s for s, n in sorted(subtype_counts.items(),
                                   key=lambda kv: -kv[1])[:14]}
    for i in rows:
        card = frame.iloc[i].to_dict()
        identity = "".join(CF.color_identity_of(card)) or "C"
        types = CF.card_types_of(card)
        primary = next((t for t in ("Land", "Creature", "Instant", "Sorcery",
                                    "Artifact", "Enchantment", "Planeswalker")
                        if t in types), "Other")
        tribe = next((s for s in CF.subtypes_of(card) if s in common), "")
        facts["name"].append(str(card.get("name")))
        facts["identity"].append(identity if len(identity) <= 2 else "multi")
        facts["type"].append(primary)
        facts["tribe"].append(tribe)
    return facts


def main(args=None):
    import pandas as pd

    from manamap.training.common import say

    components = getattr(args, "components", None) or 2
    sample = getattr(args, "sample", None) or SAMPLE

    frame = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
    rng = np.random.default_rng(SEED)
    rows = np.sort(rng.choice(len(frame), size=min(sample, len(frame)),
                              replace=False))
    say(f"  {len(rows):,} cards, {components}D")

    out = {"cards": len(rows), "components": components,
           "facts": card_facts(frame, rows), "spaces": {}}
    for label, path in spaces().items():
        matrix = np.load(path)
        if matrix.shape[0] != len(frame):
            say(f"    skipping {label}: {matrix.shape[0]} rows, corpus has {len(frame)}")
            continue
        say(f"    projecting {label} ({matrix.shape[1]}d)…")
        points = project(matrix[rows], components=components)
        out["spaces"][label] = {"dim": int(matrix.shape[1]),
                                "points": _quantise(points)}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out) + "\n")
    say(f"  Wrote {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB, "
        f"{len(out['spaces'])} spaces)")
