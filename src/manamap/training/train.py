"""Step 4: Train the CardEmbeddingModel with triplet margin loss."""

import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from manamap.config import (
    BATCH_SIZE,
    CARD_FEATURES_PATH,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MODEL_PATH,
    NUM_EPOCHS,
    OUTPUT_CSV_PATH,
    TEXT_EMBEDDINGS_PATH,
    TRIPLET_MARGIN,
    VAL_SPLIT,
)
from manamap.training.common import collate_triplets, get_device, run_epoch
from manamap.training.model import CardEmbeddingModel


class TripletMiningDataset(Dataset):
    """Mines triplets on-the-fly based on (supertype, primary_color) groups."""

    def __init__(self, indices, text_embs, features, supertypes, primary_colors):
        self.indices = indices
        self.text_embs = text_embs
        self.supertype = features["supertype"]
        self.rarity = features["rarity"]
        self.color_identity = features["color_identity"]
        self.layout = features["layout"]
        self.continuous = features["continuous"]
        self.keywords = features["keywords"]
        self.supertypes = supertypes
        self.primary_colors = primary_colors

        # Build group index: (supertype, primary_color) → list of dataset indices
        self.groups = defaultdict(list)
        self.supertype_groups = defaultdict(list)
        self.color_groups = defaultdict(list)
        for idx in indices:
            key = (supertypes[idx], primary_colors[idx])
            self.groups[key].append(idx)
            self.supertype_groups[supertypes[idx]].append(idx)
            self.color_groups[primary_colors[idx]].append(idx)

        self.group_keys = list(self.groups.keys())

    def __len__(self):
        return len(self.indices)

    def _get_features(self, idx):
        return (
            self.text_embs[idx],
            self.supertype[idx],
            self.rarity[idx],
            self.color_identity[idx],
            self.layout[idx],
            self.continuous[idx],
            self.keywords[idx],
        )

    def __getitem__(self, i):
        anchor_idx = self.indices[i]
        anchor_st = self.supertypes[anchor_idx]
        anchor_pc = self.primary_colors[anchor_idx]
        anchor_key = (anchor_st, anchor_pc)

        # ── Positive: same (supertype, primary_color) group ───────────────
        group = self.groups[anchor_key]
        if len(group) > 1:
            pos_idx = anchor_idx
            while pos_idx == anchor_idx:
                pos_idx = random.choice(group)
        elif len(self.supertype_groups[anchor_st]) > 1:
            # Fallback: same supertype
            pos_idx = anchor_idx
            while pos_idx == anchor_idx:
                pos_idx = random.choice(self.supertype_groups[anchor_st])
        elif len(self.color_groups[anchor_pc]) > 1:
            # Fallback: same primary color
            pos_idx = anchor_idx
            while pos_idx == anchor_idx:
                pos_idx = random.choice(self.color_groups[anchor_pc])
        else:
            # Last resort: random
            pos_idx = random.choice(self.indices)

        # ── Negative: different supertype AND different primary_color ─────
        neg_idx = anchor_idx
        for _ in range(50):
            candidate = random.choice(self.indices)
            if (self.supertypes[candidate] != anchor_st and
                    self.primary_colors[candidate] != anchor_pc):
                neg_idx = candidate
                break
        else:
            # If rejection sampling fails, just pick any different card
            neg_idx = random.choice(self.indices)

        anchor_feats = self._get_features(anchor_idx)
        pos_feats = self._get_features(pos_idx)
        neg_feats = self._get_features(neg_idx)

        return anchor_feats, pos_feats, neg_feats


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Load features
    print("Loading features...")
    text_embs = np.load(TEXT_EMBEDDINGS_PATH)
    features = dict(np.load(CARD_FEATURES_PATH))
    df = pd.read_csv(OUTPUT_CSV_PATH)
    supertypes = df["supertype"].fillna("Unknown").values
    primary_colors = df["primary_color"].fillna("Colorless").values

    n = len(df)
    print(f"  {n:,} cards")

    # Train/val split
    rng = np.random.RandomState(42)
    all_indices = list(range(n))
    rng.shuffle(all_indices)
    split = int(n * (1 - VAL_SPLIT))
    train_indices = all_indices[:split]
    val_indices = all_indices[split:]
    print(f"  Train: {len(train_indices):,}  Val: {len(val_indices):,}")

    # Datasets and loaders
    train_ds = TripletMiningDataset(train_indices, text_embs, features, supertypes, primary_colors)
    val_ds = TripletMiningDataset(val_indices, text_embs, features, supertypes, primary_colors)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_triplets, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_triplets, num_workers=0
    )

    # Model
    model = CardEmbeddingModel().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    criterion = torch.nn.TripletMarginLoss(margin=TRIPLET_MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience_counter = 0
    print(f"\nTraining for {NUM_EPOCHS} epochs (early stopping patience={EARLY_STOPPING_PATIENCE})...")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            marker = " ← saved"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{marker}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping: no improvement for {EARLY_STOPPING_PATIENCE} epochs")
            break

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
