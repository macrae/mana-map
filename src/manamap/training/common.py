"""Shared training utilities: device selection, triplet collation, epoch loop."""

import numpy as np
import torch


def say(*parts, **kw):
    """`print`, but FLUSHED — the default for anything that runs for minutes.

    THE THIRD TIME THIS BUG WAS FIXED AS AN INSTANCE. `print` buffers when stdout
    is not a tty, so a long job launched into a log or a pipe shows nothing at all
    until it exits: a 36-minute training run printed its first line at the end,
    and `vae_cache` reproduced it immediately afterwards because the fix had gone
    into `train_vae` alone. The pilot works from the terminal and a long job must
    show it is alive, so the flush belongs in one shared place rather than in
    whichever call site last got caught.
    """
    kw.setdefault("flush", True)
    print(*parts, **kw)


def get_device():
    """Pick best available device: MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collate_triplets(batch):
    """Collate list of (anchor, pos, neg) feature tuples into batched tensors."""
    anchors, positives, negatives = zip(*batch)

    def stack_features(feats_list):
        text = torch.tensor(np.array([f[0] for f in feats_list]), dtype=torch.float32)
        supertype = torch.tensor([f[1] for f in feats_list], dtype=torch.long)
        rarity = torch.tensor([f[2] for f in feats_list], dtype=torch.long)
        ci = torch.tensor([f[3] for f in feats_list], dtype=torch.long)
        layout = torch.tensor([f[4] for f in feats_list], dtype=torch.long)
        continuous = torch.tensor(np.array([f[5] for f in feats_list]), dtype=torch.float32)
        keywords = torch.tensor(np.array([f[6] for f in feats_list]), dtype=torch.float32)
        return text, supertype, rarity, ci, layout, continuous, keywords

    return stack_features(anchors), stack_features(positives), stack_features(negatives)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one epoch. Pass optimizer=None for validation (no grad)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_batches = 0

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for anchor, positive, negative in loader:
            # Move to device
            a_args = [t.to(device) for t in anchor]
            p_args = [t.to(device) for t in positive]
            n_args = [t.to(device) for t in negative]

            a_emb = model(*a_args)
            p_emb = model(*p_args)
            n_emb = model(*n_args)

            loss = criterion(a_emb, p_emb, n_emb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)
