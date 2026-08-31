"""Cache the frozen encoder's output so a sweep costs minutes instead of hours.

## WHY THIS EXISTS

At `unfreeze=0` the encoder NEVER CHANGES. Three runs spent ~250s per epoch
pushing 34,890 cards through 22.7M frozen parameters, twenty times each, in
order to train a 0.94M head — and every one of those forward passes computed
exactly the same numbers as the run before it.

    20-epoch run, recomputing the encoder     ~85 min
    20-epoch run over a cache                  ~2 min   (after a ~9 min one-off)

That is the difference between one hyperparameter per evening and a sweep in a
coffee break, and the first three runs made the case for it: run 1 established
the free-bits floor was an off switch, run 2 that beta 0.25 collapses the
latent, run 3 that beta 0.01 holds but drifts back under the floor. Three
findings, four hours, and not one of them needed the encoder recomputed.

## WHAT IS CACHED, AND WHAT IS NOT

One pooled 384-d vector per (card, masked-block) pair, for the FOUR single-block
masks — 139,560 forwards, 214 MB. The two-block masks are dropped: they were 15%
of training draws, and caching all ten combinations would cost 2.5x the space for
a case the sweep does not need to resolve.

**A cache is only valid for the encoder that built it.** The header records the
model name and the serialisation's block list, and `load_features` refuses a
mismatch rather than silently training a head against vectors from a different
encoder — the failure that would look like a bad hyperparameter.

**It is useless for `unfreeze > 0`**, where the encoder is what is learning.
That configuration keeps the slow path, and `train_vae` picks by looking at
`unfreeze` rather than by a flag somebody has to remember.
"""

import json
import time

import numpy as np
import torch

from manamap.config import DATA_DIR, OUTPUT_CSV_PATH, TEXT_MODEL_NAME
from manamap.training.card_serialize import BLOCKS, serialize
from manamap.training.common import get_device, say

FEATURES_PATH = DATA_DIR / "vae_encoder_features.npy"
FEATURES_META_PATH = DATA_DIR / "vae_encoder_features.json"

#: Also cache the UNMASKED view, which `write_embeddings` needs and which would
#: otherwise be the one forward pass still done the slow way.
VIEWS = ("none",) + BLOCKS


def build(batch_size=128, echo=say):
    """Encode every (card, view) pair once. Returns the (N, V, 384) matrix."""
    import pandas as pd
    from transformers import AutoModel, AutoTokenizer

    device = get_device()
    frame = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
    cards = frame.to_dict("records")
    name = f"sentence-transformers/{TEXT_MODEL_NAME}"
    tok = AutoTokenizer.from_pretrained(name)
    encoder = AutoModel.from_pretrained(name).to(device).eval()

    out = np.zeros((len(cards), len(VIEWS), encoder.config.hidden_size), dtype=np.float32)
    started = time.time()
    for v, view in enumerate(VIEWS):
        mask = () if view == "none" else (view,)
        texts = [serialize(card, mask) for card in cards]
        for i in range(0, len(texts), batch_size):
            batch = tok(texts[i:i + batch_size], truncation=True, max_length=128,
                        padding="max_length", return_tensors="pt").to(device)
            with torch.no_grad():
                hidden = encoder(**batch).last_hidden_state
            weights = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1e-9)
            out[i:i + batch_size, v] = pooled.cpu().numpy()
        echo(f"    view {view:>6}: {time.time() - started:.0f}s elapsed")

    np.save(FEATURES_PATH, out)
    FEATURES_META_PATH.write_text(json.dumps({
        "encoder": name, "views": list(VIEWS), "blocks": list(BLOCKS),
        "cards": len(cards), "dim": int(out.shape[2]),
        # THE GUARD: a cache built from a different corpus is not a cache.
        "corpus_rows": len(frame),
    }, indent=1) + "\n")
    return out


def load_features():
    """`(matrix, meta)`, or `(None, why)` when the cache cannot be trusted.

    Refuses rather than warns. A head trained against vectors from a different
    encoder produces a plausible loss curve and a meaningless space, which is
    the failure mode this whole session has been learning to catch early.
    """
    if not (FEATURES_PATH.exists() and FEATURES_META_PATH.exists()):
        return None, "no cache"
    meta = json.loads(FEATURES_META_PATH.read_text())
    expected = f"sentence-transformers/{TEXT_MODEL_NAME}"
    if meta.get("encoder") != expected:
        return None, f"cache built from {meta.get('encoder')}, not {expected}"
    if list(meta.get("blocks") or []) != list(BLOCKS):
        return None, f"cache blocks {meta.get('blocks')} != {list(BLOCKS)}"
    matrix = np.load(FEATURES_PATH, mmap_mode="r")
    if matrix.shape[0] != meta.get("cards"):
        return None, "cache length disagrees with its own header"
    return matrix, meta


def view_index(view):
    return VIEWS.index(view)


def main(args):
    echo = say
    matrix, why = load_features()
    if matrix is not None and not getattr(args, "force", False):
        echo(f"  cache present: {matrix.shape} — pass --force to rebuild")
        return
    echo(f"  building encoder feature cache ({why})")
    started = time.time()
    out = build(echo=echo)
    echo(f"  Wrote {FEATURES_PATH}: {out.shape} "
         f"({out.nbytes/1e6:.0f} MB) in {time.time()-started:.0f}s")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap vae-cache`.")
