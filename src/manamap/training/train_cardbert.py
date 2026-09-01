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

## THE THREE ARTIFACTS ARE WRITTEN TOGETHER OR NOT AT ALL

`write_embeddings` used to run once, after the loop. A run killed at epoch 19
therefore left a checkpoint from epoch 18 sitting beside an embeddings file from
a two-epoch run forty minutes earlier and a history describing neither —
**three artifacts in three different states, with nothing on disk saying so**.
That is the branched-write bug in a new place: the thing that was measured and
the thing that was filed are not the same thing.

So the checkpoint, the embeddings and the history are all written on each new
best, stamped with the epoch and the val loss they came from. It costs **16.6s
against a ~110s epoch** — measured, not guessed; the first estimate in this
paragraph said 4s — and only on epochs that improve, which grow rarer as a run
goes on. That buys an interrupted run being a usable result rather than a trap,
and `--embed-only` regenerates the embeddings from a checkpoint without
retraining, which is what makes a killed run recoverable at all.

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
from manamap.training.loss_cardbert import (VIEW_WEIGHT, accuracy, field_present,
                                            field_targets, imputation_loss,
                                            span_recall, vicreg, vicreg_parts,
                                            view_agreement, view_contrastive)
from manamap.training.model_cardbert import CardBERT

EMBEDDINGS_PATH = DATA_DIR / "embeddings_cardbert.npy"
MODEL_PATH = DATA_DIR / "model_cardbert.pt"
HISTORY_PATH = DATA_DIR / "model_cardbert_history.json"
EMBEDDINGS_META_PATH = DATA_DIR / "embeddings_cardbert.json"


def paths_for(tag=None):
    """Artifact paths for one run. A TAGGED run never touches the untagged ones.

    Every path here is a fixed constant, so two configurations run back to back
    would silently overwrite each other and the second would be read as the
    first. That is the `--out is slug-scoped` lesson in a new place: a sweep that
    cannot keep its runs apart produces one result wearing several names.
    """
    if not tag:
        return EMBEDDINGS_PATH, MODEL_PATH, HISTORY_PATH, EMBEDDINGS_META_PATH
    return (DATA_DIR / f"embeddings_cardbert_{tag}.npy",
            DATA_DIR / f"model_cardbert_{tag}.pt",
            DATA_DIR / f"model_cardbert_history_{tag}.json",
            DATA_DIR / f"embeddings_cardbert_{tag}.json")
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
              optimiser=None, indicator=None, view_weight=VIEW_WEIGHT,
              objective="infonce"):
    training = optimiser is not None
    model.train(training)
    totals, seen, agree, parts = 0.0, 0, [], []
    field_scores, span_scores = {}, {}
    for start in range(0, len(order), BATCH_SIZE):
        rows = order[start:start + BATCH_SIZE]
        if len(rows) < 8:
            continue
        clean_tab, clean_span = tabular[rows].to(device), spans[rows].to(device)
        # TWO INDEPENDENT MASKINGS OF THE SAME CARDS. The second view is what
        # trains `[CLS]`; without it the projection that produces the embedding
        # gets no gradient at all.
        chosen = [masking.draw(rng) for _ in rows]
        other = [masking.draw(rng) for _ in rows]

        with torch.set_grad_enabled(training):
            targets = field_targets(clean_tab, tab_offsets, schema)
            present = field_present(clean_tab, tab_offsets, schema)
            span_targets = {s: clean_span[:, span_offsets[s][0]:span_offsets[s][1] - 2]
                            for s in slots}
            loss = None
            latents = []
            for view in (chosen, other):
                in_tab, in_span = mask_batch(clean_tab, clean_span, view,
                                             tab_cols, span_cols)
                by_field, by_slot = field_mask(view, schema, slots, device, indicator)
                out = model(in_tab, in_span, tab_offsets, span_offsets)
                latents.append(out["latent"])
                term, _ = imputation_loss(out["predictions"], targets, present,
                                          by_field, weights, schema, span_targets,
                                          by_slot, slots)
                loss = term if loss is None else loss + term
            # THE VIEW TERM, two ways. `infonce` uses every other card in the batch
        # as a negative — which is the mechanism under suspicion, since two
        # cards that ramp the same way are pushed apart. `vicreg` has no
        # negatives at all: variance stops collapse, covariance decorrelates.
        if objective == "vicreg":
            view_term = vicreg(latents[0], latents[1])
        else:
            view_term = view_contrastive(latents[0], latents[1])
        loss = 0.5 * loss + view_weight * view_term
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
            agree.append(view_agreement(latents[0], latents[1]))
            if objective == "vicreg":
                parts.append(vicreg_parts(latents[0], latents[1])["live_dims"])
        totals += float(loss.detach()) * len(rows)
        seen += len(rows)
    return (totals / max(1, seen),
            {k: float(np.mean(v)) for k, v in field_scores.items()},
            {k: float(np.mean(v)) for k, v in span_scores.items()},
            float(np.mean(agree)) if agree else float("nan"),
            float(np.mean(parts)) if parts else float("nan"))


def embed_only(args=None):
    """Regenerate the embeddings from the saved checkpoint. No training."""
    import pandas as pd

    device = get_device()
    if not MODEL_PATH.exists():
        raise SystemExit(f"no checkpoint at {MODEL_PATH} — train first")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    cards = card_source.enriched(
        pd.read_csv(OUTPUT_CSV_PATH, low_memory=False).to_dict("records"))
    schema = CF.build_schema(CF.vocabularies(cards))
    names = [f.name for f in schema]
    if names != list(checkpoint.get("fields") or []):
        raise SystemExit(
            "the schema has changed since this checkpoint was trained "
            f"({len(names)} fields now, {len(checkpoint.get('fields') or [])} then) — "
            "position embeddings are learned BY INDEX, so the weights no longer "
            "mean what they meant. Retrain.")
    cache, why = SE.load()
    if cache is None:
        raise SystemExit(f"span cache unusable ({why})")

    tabular, spans, tab_offsets, span_offsets = build_matrices(cards, schema, cache)
    model = CardBERT(schema, SE.SPAN_SLOTS, span_dim=cache.dim,
                     d_model=checkpoint["d_model"], layers=checkpoint["layers"],
                     heads=max(4, checkpoint["d_model"] // 32)).to(device)
    model.load_state_dict(checkpoint["state"])
    say(f"  checkpoint from epoch {checkpoint.get('epoch')} "
        f"(val {checkpoint.get('val_loss')})")
    write_embeddings(model, torch.from_numpy(tabular), torch.from_numpy(spans),
                     tab_offsets, span_offsets, device,
                     epoch=checkpoint.get("epoch"),
                     val_loss=checkpoint.get("val_loss"))


def main(args=None):
    import pandas as pd

    if getattr(args, "embed_only", False):
        return embed_only(args)
    epochs = getattr(args, "epochs", None) or EPOCHS
    d_model = getattr(args, "d_model", None) or 128
    layers = getattr(args, "layers", None) or 3
    view_weight = getattr(args, "view_weight", None)
    view_weight = VIEW_WEIGHT if view_weight is None else float(view_weight)
    objective = getattr(args, "objective", None) or "infonce"
    if objective == "vicreg" and getattr(args, "view_weight", None) is None:
        # VICReg's own coefficients (25/25/1) already set the scale, so the outer
        # weight stays at 1.0 rather than multiplying them again.
        view_weight = 1.0
    tag = getattr(args, "tag", None)
    emb_path, model_path, history_path, meta_path = paths_for(tag)
    say(f"  objective={objective}  view_weight={view_weight}  "
        f"tag={tag or '(none)'}")

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
        train_loss, _, _, _, _ = run_epoch(
            model, train_rows, tabular, spans, tab_offsets, span_offsets,
            tab_cols, span_cols, weights, schema, SE.SPAN_SLOTS,
            np.random.default_rng(epoch), device, optimiser, indicator,
            view_weight, objective)
        val_loss, field_scores, span_scores, agreement, live = run_epoch(
            model, val_rows, tabular, spans, tab_offsets, span_offsets,
            tab_cols, span_cols, weights, schema, SE.SPAN_SLOTS,
            np.random.default_rng(10_000 + epoch), device, None, indicator,
            view_weight, objective)
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                        "fields": field_scores, "spans": span_scores,
                        "view_agreement": agreement, "live_dims": live})
        flag = ""
        if val_loss < best - 1e-4:
            best, bad = val_loss, 0
            torch.save({"state": model.state_dict(), "d_model": d_model,
                        "layers": layers, "fields": [f.name for f in schema],
                        "epoch": epoch, "val_loss": val_loss,
                        "view_weight": view_weight,
                        "objective": objective}, model_path)
            # ALL THREE, TOGETHER. See the module docstring: writing embeddings
            # only after the loop meant a killed run left them describing a
            # different model than the checkpoint did.
            history_path.write_text(json.dumps(history, indent=1) + "\n")
            write_embeddings(model, tabular, spans, tab_offsets, span_offsets,
                             device, epoch=epoch, val_loss=val_loss, quiet=True,
                             out_path=emb_path, meta_path=meta_path,
                             view_weight=view_weight)
            flag = "  *"
        else:
            bad += 1
        extra = f"live {live:.0f}  " if objective == "vicreg" else ""
        say(f"  epoch {epoch:3}  train {train_loss:7.4f}  val {val_loss:7.4f}  "
            f"kw_flying {field_scores.get('kw_flying', float('nan')):.3f}  "
            f"span/trig {span_scores.get('triggered', float('nan')):.3f}  "
            f"views {agreement:.3f}  {extra}"
            f"{time.time()-started:5.1f}s{flag}")
        if bad >= PATIENCE:
            say(f"  early stop: {PATIENCE} epochs without improvement")
            break

    say(f"  best val {best:.4f}; artifacts are from that epoch")


@torch.no_grad()
def write_embeddings(model, tabular, spans, tab_offsets, span_offsets, device,
                     epoch=None, val_loss=None, quiet=False, out_path=None,
                     meta_path=None, view_weight=None):
    """Every card, UNMASKED and L2-normalised, in corpus order.

    The sidecar records WHICH model these came from. Without it the only way to
    tell a fresh embeddings file from a stale one is the mtime, and an mtime says
    when a file was written, not what wrote it.
    """
    was_training = model.training
    model.eval()
    out = np.zeros((tabular.shape[0], model.latent_dim), dtype=np.float32)
    for start in range(0, tabular.shape[0], 512):
        stop = min(start + 512, tabular.shape[0])
        out[start:stop] = model.embed(
            tabular[start:stop].to(device), spans[start:stop].to(device),
            tab_offsets, span_offsets).cpu().numpy()
    out_path = out_path or EMBEDDINGS_PATH
    meta_path = meta_path or EMBEDDINGS_META_PATH
    np.save(out_path, out)
    meta_path.write_text(json.dumps({
        "epoch": epoch, "val_loss": val_loss, "cards": int(out.shape[0]),
        "dim": int(out.shape[1]), "latent_dim": model.latent_dim,
        "view_weight": view_weight,
    }, indent=1) + "\n")
    model.train(was_training)
    if not quiet:
        say(f"  Wrote {out_path}: {out.shape} (epoch {epoch})")
    return out
