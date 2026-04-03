"""CardEmbeddingModel: lightweight fusion MLP that produces 128-dim embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    COLOR_IDENTITY_EMBEDDING_DIM,
    COLOR_IDENTITY_VOCAB_SIZE,
    CONTINUOUS_DIM,
    FINAL_EMBEDDING_DIM,
    KEYWORD_DIM,
    LAYOUT_EMBEDDING_DIM,
    LAYOUT_VOCAB_SIZE,
    MLP_DROPOUT,
    MLP_HIDDEN_DIM,
    RARITY_EMBEDDING_DIM,
    RARITY_VOCAB_SIZE,
    SUPERTYPE_EMBEDDING_DIM,
    SUPERTYPE_VOCAB_SIZE,
    TEXT_EMBEDDING_DIM,
)


class CardEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Categorical embedding tables
        self.supertype_emb = nn.Embedding(SUPERTYPE_VOCAB_SIZE, SUPERTYPE_EMBEDDING_DIM)
        self.rarity_emb = nn.Embedding(RARITY_VOCAB_SIZE, RARITY_EMBEDDING_DIM)
        self.ci_emb = nn.Embedding(COLOR_IDENTITY_VOCAB_SIZE, COLOR_IDENTITY_EMBEDDING_DIM)
        self.layout_emb = nn.Embedding(LAYOUT_VOCAB_SIZE, LAYOUT_EMBEDDING_DIM)

        # Input dim: text(384) + supertype(16) + rarity(8) + ci(32) + layout(16)
        #            + continuous(2) + keywords(50) = 508
        input_dim = (
            TEXT_EMBEDDING_DIM
            + SUPERTYPE_EMBEDDING_DIM
            + RARITY_EMBEDDING_DIM
            + COLOR_IDENTITY_EMBEDDING_DIM
            + LAYOUT_EMBEDDING_DIM
            + CONTINUOUS_DIM
            + KEYWORD_DIM
        )

        # Fusion MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(MLP_HIDDEN_DIM, FINAL_EMBEDDING_DIM),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(FINAL_EMBEDDING_DIM, FINAL_EMBEDDING_DIM),
        )

    def forward(self, text_emb, supertype_idx, rarity_idx, ci_idx, layout_idx,
                continuous, keywords):
        """
        Args:
            text_emb:      (B, 384)  float  — pre-computed sentence embeddings
            supertype_idx: (B,)      long   — supertype index
            rarity_idx:    (B,)      long   — rarity index
            ci_idx:        (B,)      long   — color_identity index
            layout_idx:    (B,)      long   — layout index
            continuous:    (B, 2)    float  — [normalized_cmc, normalized_edhrec_rank]
            keywords:      (B, 50)   float  — multi-hot keyword vector
        Returns:
            (B, 128) L2-normalized embeddings
        """
        sup = self.supertype_emb(supertype_idx)
        rar = self.rarity_emb(rarity_idx)
        ci = self.ci_emb(ci_idx)
        lay = self.layout_emb(layout_idx)

        x = torch.cat([text_emb, sup, rar, ci, lay, continuous, keywords], dim=1)
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=1)
        return x
