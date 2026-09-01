"""Which fields does a lookup table already solve?

## WHY THIS GATES THE OBJECTIVE

An imputation loss is a sum over fields, and a field that is a deterministic
function of another field drives its own term to zero while teaching the model
nothing. `cmc` is the mana pips added up. `supertype` is the type flags. Left
unweighted they dominate the gradient with arithmetic, and the loss curve looks
excellent while the latent learns nothing about what a card DOES.

`train_ability.py` records the same lesson from the other direction: an objective
free to take the easy route takes it.

## HOW IT MEASURES

For each field, fit a RIDGE PROBE from every other field's columns — everything
unmasked, which is the most generous case the model will ever see — and score it
against the trivial baseline (the mean for a number, the majority class for a
flag). What is reported is the LIFT over that baseline, on a held-out split.

A linear probe understates a genuinely nonlinear dependency, so a high score is
strong evidence of triviality and a low one is weak evidence of difficulty. That
asymmetry is the right way round: this is used to DEMOTE fields, never to promote
them.

The whole sweep runs off one Gram matrix. `X'X` is computed once at 623x623 and
each field's probe is a submatrix slice, so 73 probes cost one pass over the
corpus rather than 73.
"""

import json

import numpy as np

from manamap.training import card_fields as CF

#: Ridge penalty. Large enough that a collinear block stays solvable, small
#: enough not to be doing the work itself.
RIDGE = 1e-3

#: Held-out fraction, split on a hash of the card's oracle text — NOT on the row.
#: 2,705 cards share an exact oracle text with another card, so a row split puts
#: the same text on both sides and every score comes back inflated.
TEST_FRACTION = 0.2


def _split(cards):
    """Boolean mask: True == held out. Split by TEXT, seeded, reproducible."""
    import hashlib

    out = np.zeros(len(cards), dtype=bool)
    for i, card in enumerate(cards):
        key = str(card.get("oracle_text") or card.get("name") or i)
        digest = hashlib.sha1(key.encode("utf-8")).digest()
        out[i] = (digest[0] / 256.0) < TEST_FRACTION
    return out


def encode_corpus(cards, schema):
    """`(matrix, offsets)` — every card encoded UNMASKED."""
    rows = np.zeros((len(cards), sum(f.total_width for f in schema)), dtype=np.float32)
    offsets = None
    for i, card in enumerate(cards):
        rows[i], offsets = CF.encode(card, schema)
    return rows, offsets


def _value_columns(field, offsets):
    """The columns carrying the VALUE, excluding the two state flags.

    The flags must be excluded from both sides: as a target they are not the
    question, and as a predictor `is_present` for a field is a perfect predictor
    of itself.
    """
    lo, hi = offsets[field.name]
    return list(range(lo, hi - 2))


def probe(matrix, offsets, schema, held_out):
    """`{field: {score, baseline, lift, kind}}` for every field."""
    columns_of = {f.name: _value_columns(f, offsets) for f in schema}
    flags_of = {f.name: [offsets[f.name][1] - 2, offsets[f.name][1] - 1] for f in schema}

    train = ~held_out
    X = matrix[train]
    gram = X.T @ X
    n_cols = matrix.shape[1]

    results = {}
    for field in schema:
        target_cols = columns_of[field.name]
        if not target_cols:
            continue
        # Everything except this field's own columns AND its state flags.
        drop = set(target_cols) | set(flags_of[field.name])
        keep = np.array([c for c in range(n_cols) if c not in drop])

        A = gram[np.ix_(keep, keep)] + RIDGE * np.eye(len(keep), dtype=np.float32)
        Y = matrix[train][:, target_cols]
        B = X[:, keep].T @ Y
        try:
            weights = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:                 # pragma: no cover - singular
            continue

        predicted = matrix[held_out][:, keep] @ weights
        actual = matrix[held_out][:, target_cols]
        results[field.name] = _score(field, predicted, actual,
                                     matrix[train][:, target_cols])
    return results


def _score(field, predicted, actual, train_actual):
    """Lift over the trivial baseline, on the held-out split."""
    if field.kind == "numeric":
        mean = float(train_actual[:, 0].mean())
        residual = float(((predicted[:, 0] - actual[:, 0]) ** 2).sum())
        total = float(((actual[:, 0] - mean) ** 2).sum())
        r2 = 1.0 - residual / total if total > 1e-9 else 0.0
        return {"kind": "numeric", "score": round(r2, 4),
                "baseline": 0.0, "lift": round(max(0.0, r2), 4)}

    if field.kind in ("binary", "set"):
        hit = ((predicted > 0.5) == (actual > 0.5)).mean(axis=0)
        majority = np.maximum(train_actual.mean(axis=0), 1 - train_actual.mean(axis=0))
        accuracy, base = float(hit.mean()), float(majority.mean())
        return {"kind": field.kind, "score": round(accuracy, 4),
                "baseline": round(base, 4),
                # How much of the headroom above the baseline it closed.
                "lift": round((accuracy - base) / max(1e-9, 1 - base), 4)}

    # categorical: argmax over the one-hot block
    hit = (predicted.argmax(axis=1) == actual.argmax(axis=1)).mean()
    base = float(train_actual.mean(axis=0).max())
    return {"kind": "categorical", "score": round(float(hit), 4),
            "baseline": round(base, 4),
            "lift": round((float(hit) - base) / max(1e-9, 1 - base), 4)}


#: A field the probe solves this well is arithmetic, not knowledge. Demote it in
#: the loss rather than deleting it — it is still a useful INPUT, and a model
#: that cannot reproduce `cmc` from the pips has a different problem.
TRIVIAL_LIFT = 0.95


def report(results, echo=print):
    rows = sorted(results.items(), key=lambda kv: -kv[1]["lift"])
    echo(f"{'field':26} {'kind':12} {'score':>7} {'base':>7} {'lift':>7}")
    for name, r in rows:
        mark = "  TRIVIAL" if r["lift"] >= TRIVIAL_LIFT else ""
        echo(f"{name:26} {r['kind']:12} {r['score']:>7.4f} "
             f"{r['baseline']:>7.4f} {r['lift']:>7.4f}{mark}")
    trivial = [n for n, r in rows if r["lift"] >= TRIVIAL_LIFT]
    echo(f"\n{len(trivial)} of {len(rows)} fields are solved by a linear probe: {trivial}")
    return trivial


RESULTS_PATH = None      # set in main; kept out of import to avoid a config cycle


def main(args=None):
    import pandas as pd

    from manamap.config import OUTPUT_CSV_PATH
    from manamap.training import card_source
    from manamap.training.common import say

    cards = card_source.enriched(
        pd.read_csv(OUTPUT_CSV_PATH, low_memory=False).to_dict("records"))
    schema = CF.build_schema(CF.vocabularies(cards))
    say(f"  {len(cards):,} cards, {len(schema)} fields")
    matrix, offsets = encode_corpus(cards, schema)
    held = _split(cards)
    say(f"  held out {int(held.sum()):,} by TEXT hash; probing…")
    results = probe(matrix, offsets, schema, held)
    report(results, echo=say)

    from manamap.config import DATA_DIR

    out = DATA_DIR / "eval" / "recoverability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "cards": len(cards), "fields": len(schema),
        "held_out": int(held.sum()), "trivial_lift": TRIVIAL_LIFT,
        "results": results,
    }, indent=1) + "\n")
    say(f"  Wrote {out}")
