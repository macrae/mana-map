"""Step 4b: Train the *function* model — what a card does, for similarity.

This is the space every "find similar" answer comes from. The layout model
(`train.py`) organises the map by colour and type; this one answers whether two
cards do the same job, and the two questions want different geometry.

**Why this was rewritten.** The previous objective was a `TripletMarginLoss` over
"shares >=2 mechanical tags" positives and "shares 0 tags" negatives. Measured on
the shipped artifact, it used **5.97 of its 128 dimensions**, its top-50
neighbours spanned 0.024 cosine, and it scored 0.093 recall@10 against the frozen
MiniLM text it was built from at 0.187. The training stage was subtractive.

Three causes, three fixes, each isolated so the harness can attribute the result:

1. *A margin loss stops teaching once satisfied.* Replaced with in-batch InfoNCE:
   every anchor is ranked against all B-1 other positives, so the gradient does
   not vanish when the easy pairs are solved.
2. *The positives were mostly fallbacks.* Only 46.9% of cards have the two tags
   the old rule required. Roles cover 72.6% at 1.62 specific roles per card, and
   the rarest shared role is searched first.
3. *Nothing stopped the model discarding the text.* The output now carries a
   fixed-weight projection of the frozen text (see `model.py`), so the
   frozen-text score is a floor rather than an aspiration.

Explicit non-change: **no hard-negative mining yet.** Random in-batch negatives
are safe here — measured, the chance a random pair at batch 256 is a true
near-duplicate is 0.004%, about 0.01 false negatives per anchor. Hard mining
selects nearest-non-positives by construction, and 39% of cards have a text
neighbour above 0.75, so it needs a similarity ceiling to avoid training against
true positives. That is a second variable; this run changes the loss, the
positives and the output shape, and the harness says whether that is enough.
"""

import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from manamap.config import (
    ABILITY_CI_EMBEDDING_DIM,
    ABILITY_KEYWORD_EMBEDDING_DIM,
    ABILITY_MECHANICAL_TAG_EMBEDDING_DIM,
    ABILITY_MODEL_PATH,
    ABILITY_NUM_EPOCHS,
    BATCH_SIZE,
    CARD_FEATURES_PATH,
    CARD_ROLES_PATH,
    EARLY_STOPPING_PATIENCE,
    INFONCE_TEMPERATURE,
    LEARNING_RATE,
    MECHANICAL_TAG_DIM,
    MECHANICAL_TAGS_PATH,
    MIN_SHARED_TAGS_POSITIVE,
    OUTPUT_CSV_PATH,
    ROLE_BODY_FALLBACK,
    ROLE_POSITIVE_CANDIDATES,
    STRUCTURED_FEATURE_DIM,
    TEXT_EMBEDDINGS_PATH,
    VAL_SPLIT,
)
from manamap.training.common import get_device
from manamap.training.model import CardEmbeddingModel


def load_role_rows(names):
    """{row index: [specific roles]} for the current CSV ordering.

    `ROLE_BODY_FALLBACK` is dropped: it tags all 19,050 creatures, so a positive
    drawn from it would mean little more than "also a creature" — which is the
    shape of the trivial task that collapsed the layout model.
    """
    with open(CARD_ROLES_PATH, encoding="utf-8") as fh:
        by_name = json.load(fh)["roles"]
    rows = {}
    for row, name in enumerate(names):
        roles = [r for r in by_name.get(name, []) if r != ROLE_BODY_FALLBACK]
        if roles:
            rows[row] = roles
    return rows


def build_structured(features):
    """Concatenate the Phase 1 numeric blocks into one (N, 15) matrix."""
    return np.concatenate(
        [features["power_toughness"], features["mana_pips"], features["color_features"]],
        axis=1,
    ).astype(np.float32)


class FunctionPairDataset(Dataset):
    """Yields (anchor, positive) pairs. Negatives come from the batch itself.

    Dropping the explicit negative is the point of moving to InfoNCE: the old
    dataset spent a rejection-sampling loop per item to find one easy negative,
    and got one comparison out of it. Here each anchor is scored against every
    other positive in the batch, so a batch of 256 does 255 comparisons per anchor
    for the same forward pass.
    """

    def __init__(self, indices, text_embs, features, mechanical_tags, structured, role_rows):
        self.indices = indices
        self.text_embs = text_embs
        self.supertype = features["supertype"]
        self.rarity = features["rarity"]
        self.color_identity = features["color_identity"]
        self.layout = features["layout"]
        self.continuous = features["continuous"]
        self.keywords = features["keywords"]
        self.mechanical_tags = mechanical_tags
        self.structured = structured

        # Mining is scoped to this split's indices only — a val positive drawn
        # from the train set would leak.
        self.roles = {i: role_rows[i] for i in indices if i in role_rows}
        self.role_groups = defaultdict(list)
        for idx, roles in self.roles.items():
            for role in roles:
                self.role_groups[role].append(idx)

        self.tag_sets = {i: set(np.where(mechanical_tags[i] > 0)[0]) for i in indices}
        self.tag_groups = defaultdict(list)
        for idx in indices:
            for tag in self.tag_sets[idx]:
                self.tag_groups[tag].append(idx)

        self.stats = defaultdict(int)

    def __len__(self):
        return len(self.indices)

    def _features(self, idx):
        return (
            self.text_embs[idx],
            self.supertype[idx],
            self.rarity[idx],
            self.color_identity[idx],
            self.layout[idx],
            self.continuous[idx],
            self.keywords[idx],
            self.mechanical_tags[idx],
            self.structured[idx],
        )

    def _positive(self, anchor):
        """Rarest specific role first, then shared tags, then random.

        Rarest-first is what makes a role positive worth anything: two cards
        sharing `doubler:tokens` (11 cards) tell the model far more than two
        sharing `value:etb` (5,580). The fallbacks are counted in `self.stats`
        because "how often did we actually fall back" is the number that
        condemned the previous rule.
        """
        roles = self.roles.get(anchor)
        if roles:
            for role in sorted(roles, key=lambda r: len(self.role_groups[r])):
                group = self.role_groups[role]
                if len(group) < 2:
                    continue
                for candidate in random.sample(
                    group, min(len(group), ROLE_POSITIVE_CANDIDATES)
                ):
                    if candidate != anchor:
                        self.stats["role"] += 1
                        return candidate

        anchor_tags = self.tag_sets[anchor]
        if anchor_tags:
            for tag in sorted(anchor_tags, key=lambda t: len(self.tag_groups[t])):
                for candidate in random.sample(
                    self.tag_groups[tag],
                    min(len(self.tag_groups[tag]), ROLE_POSITIVE_CANDIDATES),
                ):
                    if candidate != anchor and (
                        len(anchor_tags & self.tag_sets[candidate])
                        >= MIN_SHARED_TAGS_POSITIVE
                    ):
                        self.stats["tag"] += 1
                        return candidate

        self.stats["random"] += 1
        return random.choice(self.indices)

    def __getitem__(self, i):
        anchor = self.indices[i]
        return self._features(anchor), self._features(self._positive(anchor))


def collate_pairs(batch):
    anchors, positives = zip(*batch)

    def stack(feats):
        return (
            torch.tensor(np.array([f[0] for f in feats]), dtype=torch.float32),
            torch.tensor([f[1] for f in feats], dtype=torch.long),
            torch.tensor([f[2] for f in feats], dtype=torch.long),
            torch.tensor([f[3] for f in feats], dtype=torch.long),
            torch.tensor([f[4] for f in feats], dtype=torch.long),
            torch.tensor(np.array([f[5] for f in feats]), dtype=torch.float32),
            torch.tensor(np.array([f[6] for f in feats]), dtype=torch.float32),
            torch.tensor(np.array([f[7] for f in feats]), dtype=torch.float32),
            torch.tensor(np.array([f[8] for f in feats]), dtype=torch.float32),
        )

    return stack(anchors), stack(positives)


def infonce_loss(anchors, positives, temperature=INFONCE_TEMPERATURE):
    """Symmetric in-batch InfoNCE over L2-normalized embeddings.

    `scores[i][j]` is anchor i against positive j, so the diagonal is the true
    pair and every off-diagonal entry is a negative that cost nothing to obtain.
    Symmetrising (both directions) uses the same matrix twice and is slightly
    better behaved than the one-directional MNRL formulation.

    No false-negative masking: measured on this corpus, a random in-batch pair
    has a 0.004% chance of being a true near-duplicate — about 0.01 per anchor
    per batch, which is noise. That would NOT hold for mined hard negatives.
    """
    scores = (anchors @ positives.T) / temperature
    labels = torch.arange(len(scores), device=scores.device)
    return 0.5 * (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels))


def run_pair_epoch(model, loader, device, optimizer=None):
    """One epoch. Pass optimizer=None for validation."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_batches, total_acc = 0.0, 0, 0.0
    with torch.enable_grad() if is_train else torch.no_grad():
        for anchor, positive in loader:
            a = [t.to(device) for t in anchor]
            p = [t.to(device) for t in positive]
            a_emb = model(*a[:7], mechanical_tags=a[7], structured=a[8])
            p_emb = model(*p[:7], mechanical_tags=p[7], structured=p[8])

            loss = infonce_loss(a_emb, p_emb)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Retrieval accuracy within the batch: did the anchor rank its own
            # positive first out of B candidates? Far more legible than the loss
            # value, and it is the thing the loss is a proxy for.
            with torch.no_grad():
                scores = a_emb @ p_emb.T
                correct = scores.argmax(dim=1) == torch.arange(len(scores), device=device)
                total_acc += correct.float().mean().item()

            total_loss += loss.item()
            n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, total_acc / n


def main():
    device = get_device()
    print(f"Using device: {device}")

    print("Loading features...")
    text_embs = np.load(TEXT_EMBEDDINGS_PATH)
    features = dict(np.load(CARD_FEATURES_PATH))
    mechanical_tags = np.load(MECHANICAL_TAGS_PATH)
    df = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)

    missing = [k for k in ("power_toughness", "mana_pips", "color_features")
               if k not in features]
    if missing:
        raise SystemExit(
            f"card_features.npz is missing {missing}. It predates the current "
            f"preprocess step — run `manamap preprocess` before training."
        )

    structured = build_structured(features)
    role_rows = load_role_rows(df["name"].tolist())

    n = len(df)
    print(f"  {n:,} cards · structured features {structured.shape}")
    print(f"  cards with a specific role: {len(role_rows):,} ({len(role_rows)/n*100:.1f}%)")

    rng = np.random.RandomState(42)
    all_indices = list(range(n))
    rng.shuffle(all_indices)
    split = int(n * (1 - VAL_SPLIT))
    train_indices, val_indices = all_indices[:split], all_indices[split:]
    print(f"  Train: {len(train_indices):,}  Val: {len(val_indices):,}")

    train_ds = FunctionPairDataset(train_indices, text_embs, features, mechanical_tags,
                                   structured, role_rows)
    val_ds = FunctionPairDataset(val_indices, text_embs, features, mechanical_tags,
                                 structured, role_rows)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_pairs, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_pairs, num_workers=0, drop_last=True)

    model = CardEmbeddingModel(
        ci_emb_dim=ABILITY_CI_EMBEDDING_DIM,
        keyword_emb_dim=ABILITY_KEYWORD_EMBEDDING_DIM,
        mechanical_tag_dim=MECHANICAL_TAG_DIM,
        mechanical_tag_emb_dim=ABILITY_MECHANICAL_TAG_EMBEDDING_DIM,
        structured_dim=STRUCTURED_FEATURE_DIM,
        text_passthrough=True,
    ).to(device)
    print(f"  Trainable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss, patience_counter = float("inf"), 0
    print(f"\nTraining function model — in-batch InfoNCE (T={INFONCE_TEMPERATURE}), "
          f"batch {BATCH_SIZE} = {BATCH_SIZE - 1} negatives per anchor")

    for epoch in range(1, ABILITY_NUM_EPOCHS + 1):
        train_loss, train_acc = run_pair_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_pair_epoch(model, val_loader, device)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            torch.save(model.state_dict(), ABILITY_MODEL_PATH)
            marker = " <- saved"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch:3d}/{ABILITY_NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f}{marker}")

        if epoch == 1:
            total = sum(train_ds.stats.values()) or 1
            print(f"    positives mined: "
                  + ", ".join(f"{k} {v/total:.1%}" for k, v in sorted(train_ds.stats.items())))

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping: no improvement for {EARLY_STOPPING_PATIENCE} epochs")
            break

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Function model saved to {ABILITY_MODEL_PATH}")
    print("Run `manamap embed` then `manamap eval-embeddings` — the frozen-text "
          "baseline is the bar.")


if __name__ == "__main__":
    main()
