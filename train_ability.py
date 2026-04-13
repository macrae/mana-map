"""Step 4b: Train an ability-focused embedding model with tag-overlap triplet mining."""

import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    ABILITY_CI_EMBEDDING_DIM,
    ABILITY_KEYWORD_EMBEDDING_DIM,
    ABILITY_MECHANICAL_TAG_EMBEDDING_DIM,
    ABILITY_MODEL_PATH,
    ABILITY_NUM_EPOCHS,
    BATCH_SIZE,
    CARD_FEATURES_PATH,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MECHANICAL_TAG_DIM,
    MECHANICAL_TAGS_PATH,
    MIN_SHARED_TAGS_POSITIVE,
    OUTPUT_CSV_PATH,
    TEXT_EMBEDDINGS_PATH,
    TRIPLET_MARGIN,
    VAL_SPLIT,
)
from model import CardEmbeddingModel
from train import collate_triplets, get_device, run_epoch


class AbilityTripletMiningDataset(Dataset):
    """Mines triplets based on mechanical tag overlap instead of color/supertype."""

    def __init__(self, indices, text_embs, features, mechanical_tags):
        self.indices = indices
        self.text_embs = text_embs
        self.supertype = features["supertype"]
        self.rarity = features["rarity"]
        self.color_identity = features["color_identity"]
        self.layout = features["layout"]
        self.continuous = features["continuous"]
        self.keywords = features["keywords"]
        self.mechanical_tags = mechanical_tags

        # Build tag-set index for each card
        self.tag_sets = {}
        for idx in indices:
            self.tag_sets[idx] = set(np.where(mechanical_tags[idx] > 0)[0])

        # Group by tag for finding positives — cards sharing >=2 tags
        self.tag_groups = defaultdict(list)
        for idx in indices:
            for tag_idx in self.tag_sets[idx]:
                self.tag_groups[tag_idx].append(idx)

        # Pre-compute cards with no tags (for negative mining)
        self.no_tag_cards = [idx for idx in indices if len(self.tag_sets[idx]) == 0]

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
            self.mechanical_tags[idx],
        )

    def _find_positive(self, anchor_idx):
        """Find a card sharing >= MIN_SHARED_TAGS_POSITIVE tags with anchor."""
        anchor_tags = self.tag_sets[anchor_idx]
        if len(anchor_tags) < 1:
            return random.choice(self.indices)

        # Pick a random anchor tag, get candidates from that group
        anchor_tag_list = list(anchor_tags)
        random.shuffle(anchor_tag_list)

        for tag_idx in anchor_tag_list:
            candidates = self.tag_groups[tag_idx]
            random.shuffle(candidates)
            for candidate in candidates[:50]:
                if candidate == anchor_idx:
                    continue
                shared = len(anchor_tags & self.tag_sets[candidate])
                if shared >= MIN_SHARED_TAGS_POSITIVE:
                    return candidate

        # Fallback: any card sharing at least 1 tag
        for tag_idx in anchor_tag_list:
            for candidate in self.tag_groups[tag_idx]:
                if candidate != anchor_idx:
                    return candidate

        return random.choice(self.indices)

    def _find_negative(self, anchor_idx):
        """Find a card sharing 0 mechanical tags with anchor."""
        anchor_tags = self.tag_sets[anchor_idx]

        if len(anchor_tags) == 0:
            # Anchor has no tags — pick any card that has tags
            for _ in range(50):
                candidate = random.choice(self.indices)
                if len(self.tag_sets[candidate]) > 0:
                    return candidate
            return random.choice(self.indices)

        # Try rejection sampling for card with 0 shared tags
        for _ in range(50):
            candidate = random.choice(self.indices)
            if candidate == anchor_idx:
                continue
            shared = len(anchor_tags & self.tag_sets[candidate])
            if shared == 0:
                return candidate

        # Fallback: pick from no-tag cards
        if self.no_tag_cards:
            return random.choice(self.no_tag_cards)

        return random.choice(self.indices)

    def __getitem__(self, i):
        anchor_idx = self.indices[i]
        pos_idx = self._find_positive(anchor_idx)
        neg_idx = self._find_negative(anchor_idx)

        return (
            self._get_features(anchor_idx),
            self._get_features(pos_idx),
            self._get_features(neg_idx),
        )


def collate_ability_triplets(batch):
    """Collate triplets that include mechanical tags (8 features per card)."""
    anchors, positives, negatives = zip(*batch)

    def stack_features(feats_list):
        text = torch.tensor(np.array([f[0] for f in feats_list]), dtype=torch.float32)
        supertype = torch.tensor([f[1] for f in feats_list], dtype=torch.long)
        rarity = torch.tensor([f[2] for f in feats_list], dtype=torch.long)
        ci = torch.tensor([f[3] for f in feats_list], dtype=torch.long)
        layout = torch.tensor([f[4] for f in feats_list], dtype=torch.long)
        continuous = torch.tensor(np.array([f[5] for f in feats_list]), dtype=torch.float32)
        keywords = torch.tensor(np.array([f[6] for f in feats_list]), dtype=torch.float32)
        mech_tags = torch.tensor(np.array([f[7] for f in feats_list]), dtype=torch.float32)
        return text, supertype, rarity, ci, layout, continuous, keywords, mech_tags

    return stack_features(anchors), stack_features(positives), stack_features(negatives)


def run_ability_epoch(model, loader, criterion, device, optimizer=None):
    """Run one epoch with mechanical tags. Pass optimizer=None for validation."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_batches = 0

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for anchor, positive, negative in loader:
            a_args = [t.to(device) for t in anchor]
            p_args = [t.to(device) for t in positive]
            n_args = [t.to(device) for t in negative]

            # Pass first 7 positional + mechanical_tags as keyword
            a_emb = model(*a_args[:7], mechanical_tags=a_args[7])
            p_emb = model(*p_args[:7], mechanical_tags=p_args[7])
            n_emb = model(*n_args[:7], mechanical_tags=n_args[7])

            loss = criterion(a_emb, p_emb, n_emb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Load features
    print("Loading features...")
    text_embs = np.load(TEXT_EMBEDDINGS_PATH)
    features = dict(np.load(CARD_FEATURES_PATH))
    mechanical_tags = np.load(MECHANICAL_TAGS_PATH)
    df = pd.read_csv(OUTPUT_CSV_PATH)

    n = len(df)
    print(f"  {n:,} cards")
    print(f"  Mechanical tags shape: {mechanical_tags.shape}")

    # Tag coverage
    tagged = (mechanical_tags.sum(axis=1) > 0).sum()
    print(f"  Cards with tags: {tagged:,}/{n:,} ({tagged/n*100:.1f}%)")

    # Train/val split
    rng = np.random.RandomState(42)
    all_indices = list(range(n))
    rng.shuffle(all_indices)
    split = int(n * (1 - VAL_SPLIT))
    train_indices = all_indices[:split]
    val_indices = all_indices[split:]
    print(f"  Train: {len(train_indices):,}  Val: {len(val_indices):,}")

    # Datasets and loaders
    train_ds = AbilityTripletMiningDataset(train_indices, text_embs, features, mechanical_tags)
    val_ds = AbilityTripletMiningDataset(val_indices, text_embs, features, mechanical_tags)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_ability_triplets, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_ability_triplets, num_workers=0
    )

    # Ability model with reduced color embedding, learned keywords & tags
    model = CardEmbeddingModel(
        ci_emb_dim=ABILITY_CI_EMBEDDING_DIM,
        keyword_emb_dim=ABILITY_KEYWORD_EMBEDDING_DIM,
        mechanical_tag_dim=MECHANICAL_TAG_DIM,
        mechanical_tag_emb_dim=ABILITY_MECHANICAL_TAG_EMBEDDING_DIM,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    criterion = torch.nn.TripletMarginLoss(margin=TRIPLET_MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience_counter = 0
    print(f"\nTraining ability model for {ABILITY_NUM_EPOCHS} epochs (early stopping patience={EARLY_STOPPING_PATIENCE})...")

    for epoch in range(1, ABILITY_NUM_EPOCHS + 1):
        train_loss = run_ability_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_ability_epoch(model, val_loader, criterion, device)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ABILITY_MODEL_PATH)
            marker = " <- saved"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch:3d}/{ABILITY_NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{marker}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping: no improvement for {EARLY_STOPPING_PATIENCE} epochs")
            break

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Ability model saved to {ABILITY_MODEL_PATH}")


if __name__ == "__main__":
    main()
