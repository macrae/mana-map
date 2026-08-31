"""Train the masked-imputation VAE, and say loudly if the latent collapsed.

Self-supervised: there are no mined positives anywhere in this file, which is
the point of the architecture. `card_serialize` hides a block, the model
reconstructs it, and the latent is whatever had to be carried to do that.

## THE SPLIT IS BY TEXT HASH, NEVER BY ROW

The corpus holds **2,705 exact duplicate oracle texts** across 1,031 families —
reprints and functional reprints. A row split puts `Llanowar Elves` in train and
`Fyndhorn Elves` in test and calls the memorised answer generalisation. Hashing
the oracle text keeps a family whole; measured, 0 of the 1,031 families cross.

## WHAT IS REPORTED EVERY EPOCH, AND WHY

`active_units` and per-dimension KL, because the latent IS the product. A run
that collapses still produces a loss curve that looks fine — reconstruction from
the visible blocks alone — and every downstream metric would then be measuring a
constant without saying so. A run under `MIN_ACTIVE_UNITS` exits non-zero.

## THE ARTIFACT IS A SHADOW, NOT A REPLACEMENT

Writes `data/embeddings_function_vae.npy`, alongside the current space rather
than over it. Nothing downstream changes until `eval-embeddings` says it should,
which is the whole point of measuring before switching.
"""

import collections
import hashlib
import json
import time

import numpy as np
import torch

from manamap.config import (
    DATA_DIR,
    EVAL_SEED,
    OUTPUT_CSV_PATH,
)
from manamap.training.card_serialize import BLOCKS, blocks_for, sample_mask, serialize
from manamap.training.common import get_device
from manamap.training.model_vae import (
    CardVAE,
    DECODER_VOCAB,
    MIN_ACTIVE_UNITS,
    MIN_EFFECTIVE_DIM,
    active_units,
    beta_at,
    embeddings_from,
    kl_with_free_bits,
    reconstruction_loss,
)

VAE_EMBEDDINGS_PATH = DATA_DIR / "embeddings_function_vae.npy"
VAE_MODEL_PATH = DATA_DIR / "model_vae.pt"

MAX_TOKENS = 128          # covers 99.41% of cards uncut; p99 is 120
BATCH_SIZE = 32
LEARNING_RATE = 3e-4      # the head is what trains by default; the encoder is frozen
ENCODER_LR = 2e-5         # …and a thawed layer moves far more slowly than the head
EPOCHS = 20
PATIENCE = 4
TEST_FRACTION = 2         # of 10, by text-hash bucket


def text_hash_split(frame, test_of_ten=TEST_FRACTION):
    """Boolean train mask, split on a hash of the ORACLE TEXT.

    Not the row, not the name: 2,705 cards share an oracle text with another
    card, and a row split leaks every one of those pairs.
    """
    digest = frame["oracle_text"].fillna("").map(
        lambda text: int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
    return (digest % 10 >= test_of_ten).to_numpy()


def build_vocab(token_lists, size=DECODER_VOCAB):
    """`{token_id: column}` for the most frequent `size` wordpieces.

    Measured: 7,507 distinct wordpieces over 2,235,677 occurrences, and the top
    2,000 cover 99.31%. The map is saved with the checkpoint because a decoder
    head is meaningless without the vocabulary its columns refer to.
    """
    counts = collections.Counter(t for row in token_lists for t in row)
    return {int(tok): i for i, (tok, _n) in enumerate(counts.most_common(size))}


class MaskedCards(torch.utils.data.Dataset):
    """One masked view per card per epoch, re-sampled each time it is drawn."""

    def __init__(self, cards, tokenizer, vocab, seed=EVAL_SEED):
        self.blocks = [blocks_for(card) for card in cards]
        self.cards = cards
        self.tok = tokenizer
        self.vocab = vocab
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, index):
        mask = sample_mask(self.rng)
        text = serialize(self.cards[index], mask)
        encoded = self.tok(text, truncation=True, max_length=MAX_TOKENS,
                           padding="max_length", return_tensors="pt")
        targets = torch.zeros(len(BLOCKS), len(self.vocab))
        masked = torch.zeros(len(BLOCKS), dtype=torch.bool)
        for i, block in enumerate(BLOCKS):
            if block not in mask:
                continue
            masked[i] = True
            body = self.blocks[index][block]
            for token in self.tok(body, add_special_tokens=False)["input_ids"]:
                column = self.vocab.get(int(token))
                if column is not None:
                    targets[i, column] = 1.0
        return (encoded["input_ids"][0], encoded["attention_mask"][0], targets, masked)


def _split_targets(targets, masked):
    """(B, len(BLOCKS), V) -> the per-block dicts the loss expects."""
    return ({block: targets[:, i] for i, block in enumerate(BLOCKS)},
            {block: masked[:, i] for i, block in enumerate(BLOCKS)})


def run_epoch(model, loader, device, optimizer=None, step=0, total_steps=1):
    """One pass. Returns (mean loss, mean per-dim KL, steps taken)."""
    train = optimizer is not None
    model.train(train)
    losses, kls = [], []
    for input_ids, attention, targets, masked in loader:
        input_ids, attention = input_ids.to(device), attention.to(device)
        targets, masked = targets.to(device), masked.to(device)
        with torch.set_grad_enabled(train):
            out = model(input_ids, attention)
            per_block, per_mask = _split_targets(targets, masked)
            recon = reconstruction_loss(out["logits"], per_block, per_mask)
            kl, per_dim = kl_with_free_bits(out["mu"], out["logvar"])
            loss = recon + beta_at(step, total_steps) * kl
        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
        losses.append(float(loss.detach()))
        kls.append(per_dim.detach().cpu().numpy())
    return float(np.mean(losses)), np.mean(kls, axis=0), step


def _say(*parts):
    """Print and FLUSH.

    Python block-buffers stdout when it is not a terminal, so a redirected
    training run showed an empty log for its entire length — a 36-minute job
    with no evidence it was alive, which is the one thing this repo's CLI is not
    allowed to be. Caught on the first real run.
    """
    print(*parts, flush=True)


def train(unfreeze=0, epochs=EPOCHS, batch_size=BATCH_SIZE, seed=EVAL_SEED, echo=_say):
    import pandas as pd
    from transformers import AutoTokenizer

    from manamap.config import TEXT_MODEL_NAME

    torch.manual_seed(seed)
    device = get_device()
    frame = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
    cards = frame.to_dict("records")
    is_train = text_hash_split(frame)
    echo(f"  {is_train.sum():,} train / {(~is_train).sum():,} test, split by TEXT HASH")

    tok = AutoTokenizer.from_pretrained(f"sentence-transformers/{TEXT_MODEL_NAME}")
    vocab = build_vocab(tok([serialize(c) for c in cards],
                            add_special_tokens=False)["input_ids"])
    echo(f"  decoder vocabulary: {len(vocab)} wordpieces")

    train_set = MaskedCards([c for c, keep in zip(cards, is_train) if keep], tok, vocab, seed)
    val_set = MaskedCards([c for c, keep in zip(cards, is_train) if not keep], tok, vocab, seed + 1)
    loaders = {
        name: torch.utils.data.DataLoader(data, batch_size=batch_size,
                                          shuffle=(name == "train"), num_workers=0)
        for name, data in (("train", train_set), ("val", val_set))}

    model = CardVAE(unfreeze=unfreeze, vocab_size=len(vocab)).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    echo(f"  unfreeze={unfreeze}: {trainable/1e6:.2f}M trainable "
         f"({2235677/max(trainable,1):.2f} corpus tokens per parameter)")
    # A THAWED ENCODER LAYER MOVES FAR MORE SLOWLY THAN A FRESH HEAD. One LR for
    # both either cooks the pretrained weights or starves the decoder.
    head = [p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("encoder.")]
    body = [p for n, p in model.named_parameters()
            if p.requires_grad and n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": head, "lr": LEARNING_RATE}] +
        ([{"params": body, "lr": ENCODER_LR}] if body else []))

    total_steps = epochs * max(1, len(loaders["train"]))
    best, best_epoch, step, history = float("inf"), -1, 0, []
    for epoch in range(epochs):
        started = time.time()
        loss, kl_dim, step = run_epoch(model, loaders["train"], device,
                                       optimizer, step, total_steps)
        val_loss, val_kl, _ = run_epoch(model, loaders["val"], device)
        units = active_units(torch.tensor(val_kl))
        history.append({"epoch": epoch, "train_loss": loss, "val_loss": val_loss,
                        "active_units": units, "kl_mean": float(val_kl.mean()),
                        "beta": beta_at(step, total_steps)})
        echo(f"    epoch {epoch:>2}  train {loss:.4f}  val {val_loss:.4f}  "
             f"KL/dim {val_kl.mean():.3f}  ACTIVE {units}/{model.latent_dim}  "
             f"beta {beta_at(step, total_steps):.3f}  {time.time()-started:.0f}s")
        if val_loss < best - 1e-5:
            best, best_epoch = val_loss, epoch
            torch.save({"state": model.state_dict(), "vocab": vocab,
                        "unfreeze": unfreeze, "epoch": epoch}, VAE_MODEL_PATH)
        elif epoch - best_epoch >= PATIENCE:
            echo(f"    early stop: no improvement in {PATIENCE} epochs")
            break

    model.load_state_dict(torch.load(VAE_MODEL_PATH, weights_only=False)["state"])
    final_units = history[best_epoch]["active_units"] if history else 0
    return model, tok, vocab, history, final_units


def write_embeddings(model, tok, cards, device, path=VAE_EMBEDDINGS_PATH):
    """Every card, unmasked, in corpus order. A SHADOW artifact."""
    class Plain(torch.utils.data.Dataset):
        def __len__(self):
            return len(cards)

        def __getitem__(self, i):
            enc = tok(serialize(cards[i]), truncation=True, max_length=MAX_TOKENS,
                      padding="max_length", return_tensors="pt")
            return enc["input_ids"][0], enc["attention_mask"][0]

    loader = torch.utils.data.DataLoader(Plain(), batch_size=128, shuffle=False)
    matrix = embeddings_from(model, loader, device)
    np.save(path, matrix)
    return matrix


def main(args):
    import pandas as pd

    unfreeze = int(getattr(args, "unfreeze", 0) or 0)
    epochs = int(getattr(args, "epochs", None) or EPOCHS)
    model, tok, _vocab, history, units = train(unfreeze=unfreeze, epochs=epochs)

    frame = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
    matrix = write_embeddings(model, tok, frame.to_dict("records"), get_device())
    _say(f"  Wrote {VAE_EMBEDDINGS_PATH}: {matrix.shape} "
         f"(shadow artifact — nothing downstream reads it yet)")
    (DATA_DIR / "model_vae_history.json").write_text(
        json.dumps({"history": history, "unfreeze": unfreeze}, indent=1) + "\n")

    # A COLLAPSED RUN IS A FAILED RUN AND MUST SAY SO. The loss curve looks fine
    # either way — the model reconstructs from the visible blocks — and every
    # downstream metric would then be measuring a constant.
    # EFFECTIVE DIMENSIONALITY IS THE HONEST GATE, and the first run is why.
    # `active_units` reported 128/128 while the participation ratio was 5.71 —
    # the space was nearly as degenerate as the layout space and the alarm was
    # silent, because free bits never engaged so nothing was pressuring it.
    from manamap.analysis.eval_embeddings import effective_dimensionality

    effdim = effective_dimensionality(matrix)
    _say(f"  effective dimensionality {effdim:.2f}/{model.latent_dim} "
         f"(function space 27.31, layout 3.89, floor {MIN_EFFECTIVE_DIM})")
    if effdim < MIN_EFFECTIVE_DIM:
        raise SystemExit(
            f"DEGENERATE LATENT: participation ratio {effdim:.2f}, floor is "
            f"{MIN_EFFECTIVE_DIM}. The space uses a fraction of its dimensions "
            f"however many carry KL. Lower FREE_BITS so the regulariser engages, "
            f"or the objective is not shaping the geometry at all.")

    if units < MIN_ACTIVE_UNITS:
        raise SystemExit(
            f"POSTERIOR COLLAPSE: {units} active units of {model.latent_dim}, "
            f"floor is {MIN_ACTIVE_UNITS}. The latent is carrying almost nothing "
            f"and every figure derived from it would be measuring a constant. "
            f"Raise FREE_BITS, lower BETA, or lengthen the anneal.")
    _say(f"  active units {units}/{model.latent_dim} — no collapse")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap train-vae`.")
