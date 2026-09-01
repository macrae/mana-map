"""Train CardBERT: masked field imputation over the whole corpus.

## THE SPLIT IS BY TEXT, NOT BY ROW

**2,705 cards share an exact oracle text with another card.** A row split puts
the same text on both sides and every held-out number comes back inflated — the
one control this whole architecture rests on, quietly broken. `_split` hashes the
oracle text, so duplicates land together whichever side they fall.

## MASKING IS APPLIED TO A PRECOMPUTED MATRIX

Encoding 34,890 cards through `card_fields.encode` takes ~25s, and doing it once
per epoch would dominate training. Both matrices are built ONCE unmasked, and
masking is column arithmetic on the tensor: zero the value columns of the chosen
groups, set their `is_masked` flag. That is the same transformation `encode`
performs — asserted against it by a test, because two implementations of one rule
is the divergence this repo has paid for more than once.

## WHAT EARLY STOPPING WATCHES

Held-out imputation loss, not training loss. At `d_model=128, layers=3` the model
has ~2 field-observations per parameter, which is generous — the previous
architecture had 0.94M parameters and lost to a random projection of frozen
MiniLM. Overfitting is the expected failure and the val curve is the instrument.
"""

import hashlib
import json
import time

import numpy as np
import torch

from manamap.config import DATA_DIR, OUTPUT_CSV_PATH
from manamap.training import card_fields as CF
from manamap.training import card_source, masking
from manamap.training import span_encoder as SE
from manamap.training.common import get_device, say
from manamap.training.loss_cardbert import (accuracy, field_present, field_targets,
                                            imputation_loss, span_recall)
from manamap.training.model_cardbert import CardBERT

EMBEDDINGS_PATH = DATA_DIR / "embeddings_cardbert.npy"
MODEL_PATH = DATA_DIR / "model_cardbert.pt"
HISTORY_PATH = DATA_DIR / "model_cardbert_history.json"
RECOVERABILITY_PATH = DATA_DIR / "eval" / "recoverability.json"

TEST_FRACTION = 0.15
BATCH_SIZE = 256
EPOCHS = 40
PATIENCE = 5
LR = 3e-4


def split_mask(cards, fraction=TEST_FRACTION):
    """True == held out. Hashed on TEXT so duplicate oracle texts stay together."""
    out = np.zeros(len(cards), dtype=bool)
    for i, card in enumerate(cards):
        key = str(card.get("oracle_text") or "") + "|" + str(card.get("type_line") or "")
        out[i] = (hashlib.sha1(key.encode("utf-8")).digest()[0] / 256.0) < fraction
    return out


def build_matrices(cards, schema, cache, echo=say):
    """Encode every card ONCE, unmasked. Returns tensors and their offsets."""
    started = time.time()
    tab_width = sum(f.total_width for f in schema)
    span_width = (cache.dim + 2) * len(SE.SPAN_SLOTS)
    tabular = np.zeros((len(cards), tab_width), dtype=np.float32)
    spans = np.zeros((len(cards), span_width), dtype=np.float32)
    tab_offsets = span_offsets = None
    for i, card in enumerate(cards):
        tabular[i], tab_offsets = CF.encode(card, schema)
        spans[i], span_offsets = cache.encode(card, card.get("oracle_text"))
        if i and i % 8000 == 0:
            echo(f"    encoded {i:,}/{len(cards):,}")
    echo(f"    {len(cards):,} cards in {time.time()-started:.0f}s "
         f"({tabular.nbytes/1e6:.0f} + {spans.nbytes/1e6:.0f} MB)")
    return tabular, spans, tab_offsets, span_offsets


def group_columns(schema, tab_offsets, span_offsets):
    """`{group: (value columns, flag column)}` for both matrices."""
    known = {f.name for f in schema}
    tab, span = {}, {}
    for name, fields in masking.GROUPS.items():
        value, flag = [], []
        for field in fields:
            if field not in known:
                continue
            lo, hi = tab_offsets[field]
            value.extend(range(lo, hi - 2))
            flag.append(hi - 1)                       # the is_masked column
        tab[name] = (np.array(value, dtype=np.int64), np.array(flag, dtype=np.int64))
    for name, slots in masking.SPAN_GROUPS.items():
        value, flag = [], []
        for slot in slots:
            lo, hi = span_offsets[slot]
            value.extend(range(lo, hi - 2))
            flag.append(hi - 1)
        span[name] = (np.array(value, dtype=np.int64), np.array(flag, dtype=np.int64))
    # COMPANION: hiding the keyword block must also hide the keyword TEXT, or the
    # answer is sitting in the span the model can still read.
    for name, slots in masking.COMPANION.items():
        value, flag = [], []
        for slot in slots:
            lo, hi = span_offsets[slot]
            value.extend(range(lo, hi - 2))
            flag.append(hi - 1)
        span[name] = (np.array(value, dtype=np.int64), np.array(flag, dtype=np.int64))
    return tab, span


def to_device_columns(columns, device):
    """Move the per-group column indices to the device ONCE, not per batch."""
    return {name: (torch.as_tensor(value, device=device),
                   torch.as_tensor(flag, device=device))
            for name, (value, flag) in columns.items()}


def mask_batch(tabular, spans, chosen, tab_cols, span_cols):
    """Apply each example's groups. Returns masked copies."""
    tabular, spans = tabular.clone(), spans.clone()
    for group in set(g for groups in chosen for g in groups):
        rows = torch.tensor([i for i, groups in enumerate(chosen) if group in groups],
                            device=tabular.device).unsqueeze(1)
        if not len(rows):
            continue
        if group in tab_cols:
            value, flag = tab_cols[group]
            tabular[rows, value] = 0.0
            tabular[rows, flag] = 1.0
        if group in span_cols:
            value, flag = span_cols[group]
            spans[rows, value] = 0.0
            spans[rows, flag] = 1.0
    return tabular, spans


def group_indicator(schema, slots):
    """`(groups, field matrix, slot matrix)` — which fields each group hides.

    Precomputed ONCE. The first version wrote `by_field[name][i] = 1.0` for every
    example and every field it hid: 256 x 79 scalar writes per batch, each one a
    device synchronisation on MPS. It cost **355 ms per batch**, more than the
    forward and backward passes together.
    """
    groups = masking.all_groups()
    names = [f.name for f in schema]
    field_matrix = np.zeros((len(groups), len(names)), dtype=np.float32)
    slot_matrix = np.zeros((len(groups), len(slots)), dtype=np.float32)
    for g, group in enumerate(groups):
        fields, hidden = masking.resolve([group])
        for name in fields:
            if name in names:
                field_matrix[g, names.index(name)] = 1.0
        for slot in hidden:
            slot_matrix[g, list(slots).index(slot)] = 1.0
    return groups, field_matrix, slot_matrix


def field_mask(chosen, schema, slots, device, indicator):
    """`({field: 0/1}, {slot: 0/1})` — what was hidden, per example.

    One matmul: an (examples x groups) indicator against the precomputed
    (groups x fields) matrix, clamped because two groups can hide one field.
    """
    groups, field_matrix, slot_matrix = indicator
    index = {g: i for i, g in enumerate(groups)}
    picked = np.zeros((len(chosen), len(groups)), dtype=np.float32)
    for i, names in enumerate(chosen):
        for name in names:
            picked[i, index[name]] = 1.0
    fields = torch.from_numpy(np.clip(picked @ field_matrix, 0, 1)).to(device)
    hidden = torch.from_numpy(np.clip(picked @ slot_matrix, 0, 1)).to(device)
    return ({f.name: fields[:, i] for i, f in enumerate(schema)},
            {s: hidden[:, i] for i, s in enumerate(slots)})


def load_weights(schema):
    if not RECOVERABILITY_PATH.exists():
        say("  no recoverability audit on disk — every field weighted 1.0. "
            "Run `manamap recoverability` first; 19 of 73 fields are arithmetic.")
        return {f.name: 1.0 for f in schema}
    results = json.loads(RECOVERABILITY_PATH.read_text())["results"]
    return masking.loss_weights(results)


def run_epoch(model, order, tabular, spans, tab_offsets, span_offsets,
              tab_cols, span_cols, weights, schema, slots, rng, device,
              optimiser=None, indicator=None):
    training = optimiser is not None
    model.train(training)
    totals, seen = 0.0, 0
    field_scores, span_scores = {}, {}
    for start in range(0, len(order), BATCH_SIZE):
        rows = order[start:start + BATCH_SIZE]
        if len(rows) < 8:
            continue
        clean_tab, clean_span = tabular[rows].to(device), spans[rows].to(device)
        chosen = [masking.draw(rng) for _ in rows]
        in_tab, in_span = mask_batch(clean_tab, clean_span, chosen, tab_cols, span_cols)
        by_field, by_slot = field_mask(chosen, schema, slots, device, indicator)

        with torch.set_grad_enabled(training):
            out = model(in_tab, in_span, tab_offsets, span_offsets)
            targets = field_targets(clean_tab, tab_offsets, schema)
            present = field_present(clean_tab, tab_offsets, schema)
            span_targets = {s: clean_span[:, span_offsets[s][0]:span_offsets[s][1] - 2]
                            for s in slots}
            loss, _ = imputation_loss(out["predictions"], targets, present, by_field,
                                      weights, schema, span_targets, by_slot, slots)
        if training:
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        else:
            for name, value in accuracy(out["predictions"], targets, present,
                                        by_field, schema).items():
                field_scores.setdefault(name, []).append(value)
            for slot in slots:
                sel = by_slot[slot].bool()
                if int(sel.sum()) > 1:
                    span_scores.setdefault(slot, []).append(span_recall(
                        out["predictions"][f"span:{slot}"][sel], span_targets[slot][sel]))
        totals += float(loss.detach()) * len(rows)
        seen += len(rows)
    return (totals / max(1, seen),
            {k: float(np.mean(v)) for k, v in field_scores.items()},
            {k: float(np.mean(v)) for k, v in span_scores.items()})


def main(args=None):
    import pandas as pd

    epochs = getattr(args, "epochs", None) or EPOCHS
    d_model = getattr(args, "d_model", None) or 128
    layers = getattr(args, "layers", None) or 3

    device = get_device()
    say(f"  device {device}")
    cards = card_source.enriched(
        pd.read_csv(OUTPUT_CSV_PATH, low_memory=False).to_dict("records"))
    schema = CF.build_schema(CF.vocabularies(cards))
    cache, why = SE.load()
    if cache is None:
        raise SystemExit(f"span cache unusable ({why}) — run `manamap span-cache`")

    tabular, spans, tab_offsets, span_offsets = build_matrices(cards, schema, cache)
    tab_cols, span_cols = group_columns(schema, tab_offsets, span_offsets)
    tab_cols = to_device_columns(tab_cols, device)
    span_cols = to_device_columns(span_cols, device)
    weights = load_weights(schema)
    demoted = sorted(n for n, w in weights.items() if w < 0.2)
    say(f"  {len(demoted)} fields demoted by the audit: {demoted[:6]}…")

    held = split_mask(cards)
    say(f"  {int((~held).sum()):,} train / {int(held.sum()):,} held out, split by TEXT")

    tabular = torch.from_numpy(tabular)
    spans = torch.from_numpy(spans)
    train_rows = np.where(~held)[0]
    val_rows = np.where(held)[0]

    model = CardBERT(schema, SE.SPAN_SLOTS, span_dim=cache.dim,
                     d_model=d_model, layers=layers,
                     heads=max(4, d_model // 32)).to(device)
    say(f"  CardBERT d_model={d_model} layers={layers}: "
        f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = np.random.default_rng(0)

    indicator = group_indicator(schema, SE.SPAN_SLOTS)
    history, best, bad = [], float("inf"), 0
    for epoch in range(1, epochs + 1):
        started = time.time()
        rng.shuffle(train_rows)
        train_loss, _, _ = run_epoch(
            model, train_rows, tabular, spans, tab_offsets, span_offsets,
            tab_cols, span_cols, weights, schema, SE.SPAN_SLOTS,
            np.random.default_rng(epoch), device, optimiser, indicator)
        val_loss, field_scores, span_scores = run_epoch(
            model, val_rows, tabular, spans, tab_offsets, span_offsets,
            tab_cols, span_cols, weights, schema, SE.SPAN_SLOTS,
            np.random.default_rng(10_000 + epoch), device, None, indicator)
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                        "fields": field_scores, "spans": span_scores})
        flag = ""
        if val_loss < best - 1e-4:
            best, bad = val_loss, 0
            torch.save({"state": model.state_dict(), "d_model": d_model,
                        "layers": layers, "fields": [f.name for f in schema]},
                       MODEL_PATH)
            flag = "  *"
        else:
            bad += 1
        say(f"  epoch {epoch:3}  train {train_loss:7.4f}  val {val_loss:7.4f}  "
            f"kw_flying {field_scores.get('kw_flying', float('nan')):.3f}  "
            f"span/triggered {span_scores.get('triggered', float('nan')):.3f}  "
            f"{time.time()-started:5.1f}s{flag}")
        if bad >= PATIENCE:
            say(f"  early stop: {PATIENCE} epochs without improvement")
            break

    HISTORY_PATH.write_text(json.dumps(history, indent=1) + "\n")
    write_embeddings(model, tabular, spans, tab_offsets, span_offsets, device)


@torch.no_grad()
def write_embeddings(model, tabular, spans, tab_offsets, span_offsets, device):
    """Every card, UNMASKED and L2-normalised, in corpus order."""
    model.eval()
    out = np.zeros((tabular.shape[0], model.latent_dim), dtype=np.float32)
    for start in range(0, tabular.shape[0], 512):
        stop = min(start + 512, tabular.shape[0])
        out[start:stop] = model.embed(
            tabular[start:stop].to(device), spans[start:stop].to(device),
            tab_offsets, span_offsets).cpu().numpy()
    np.save(EMBEDDINGS_PATH, out)
    say(f"  Wrote {EMBEDDINGS_PATH}: {out.shape}")
