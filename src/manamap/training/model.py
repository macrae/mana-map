"""CardEmbeddingModel: lightweight fusion MLP that produces 128-dim embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from manamap.config import (
    COLOR_IDENTITY_EMBEDDING_DIM,
    COLOR_IDENTITY_VOCAB_SIZE,
    CONTINUOUS_DIM,
    FINAL_EMBEDDING_DIM,
    FUNCTION_LEARNED_DIM,
    FUNCTION_TEXT_DIM,
    KEYWORD_DIM,
    LAYOUT_EMBEDDING_DIM,
    LAYOUT_VOCAB_SIZE,
    MECHANICAL_TAG_DIM,
    MLP_DROPOUT,
    MLP_HIDDEN_DIM,
    RARITY_EMBEDDING_DIM,
    RARITY_VOCAB_SIZE,
    SUPERTYPE_EMBEDDING_DIM,
    SUPERTYPE_VOCAB_SIZE,
    TEXT_EMBEDDING_DIM,
    TEXT_PASSTHROUGH_WEIGHT,
)


class CardEmbeddingModel(nn.Module):
    def __init__(self, ci_emb_dim=None, keyword_emb_dim=None,
                 mechanical_tag_dim=0, mechanical_tag_emb_dim=0,
                 structured_dim=0, text_passthrough=False):
        """
        Args:
            ci_emb_dim: Color identity embedding dim. None = use config default.
            keyword_emb_dim: If > 0, pass keywords through nn.Linear + ReLU.
                             If None/0, use raw multi-hot (original behavior).
            mechanical_tag_dim: Number of mechanical tag inputs. 0 = no tag input.
            mechanical_tag_emb_dim: If > 0 and mechanical_tag_dim > 0, learn tag embeddings.
            structured_dim: Width of the concatenated [power/toughness, mana pips,
                            colour features] block. 0 keeps the pre-Phase-1 input.
            text_passthrough: Split the output into a learned half and a projected
                            frozen-text half (see config's output-split block). The
                            layout model leaves this off — its job is colour/type
                            organisation for the map, and a text floor would fight it.
        """
        super().__init__()
        self.text_passthrough = text_passthrough
        self.structured_dim = structured_dim

        ci_dim = ci_emb_dim if ci_emb_dim is not None else COLOR_IDENTITY_EMBEDDING_DIM

        # Categorical embedding tables
        self.supertype_emb = nn.Embedding(SUPERTYPE_VOCAB_SIZE, SUPERTYPE_EMBEDDING_DIM)
        self.rarity_emb = nn.Embedding(RARITY_VOCAB_SIZE, RARITY_EMBEDDING_DIM)
        self.ci_emb = nn.Embedding(COLOR_IDENTITY_VOCAB_SIZE, ci_dim)
        self.layout_emb = nn.Embedding(LAYOUT_VOCAB_SIZE, LAYOUT_EMBEDDING_DIM)

        # Optional learned keyword projection
        self.keyword_proj = None
        kw_out_dim = KEYWORD_DIM
        if keyword_emb_dim and keyword_emb_dim > 0:
            self.keyword_proj = nn.Sequential(
                nn.Linear(KEYWORD_DIM, keyword_emb_dim),
                nn.ReLU(),
            )
            kw_out_dim = keyword_emb_dim

        # Optional mechanical tag projection
        self.tag_proj = None
        tag_out_dim = 0
        if mechanical_tag_dim > 0 and mechanical_tag_emb_dim > 0:
            self.tag_proj = nn.Sequential(
                nn.Linear(mechanical_tag_dim, mechanical_tag_emb_dim),
                nn.ReLU(),
            )
            tag_out_dim = mechanical_tag_emb_dim

        # Input dim
        input_dim = (
            TEXT_EMBEDDING_DIM
            + SUPERTYPE_EMBEDDING_DIM
            + RARITY_EMBEDDING_DIM
            + ci_dim
            + LAYOUT_EMBEDDING_DIM
            + CONTINUOUS_DIM
            + kw_out_dim
            + tag_out_dim
            + structured_dim
        )

        # The learned trunk. Its width shrinks when a text half is carved out of
        # the 128, so the total embedding size is unchanged either way.
        out_dim = FUNCTION_LEARNED_DIM if text_passthrough else FINAL_EMBEDDING_DIM
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(MLP_HIDDEN_DIM, FINAL_EMBEDDING_DIM),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(FINAL_EMBEDDING_DIM, out_dim),
        )

        # A plain linear projection of the frozen text, no ReLU: this half exists to
        # preserve the text geometry, and a rectifier would fold half of it away.
        self.text_proj = (
            nn.Linear(TEXT_EMBEDDING_DIM, FUNCTION_TEXT_DIM) if text_passthrough else None
        )
        # Buffers, not plain floats, so they move with .to(device) and are saved
        # in the checkpoint — the weights are meaningless without them.
        self.register_buffer(
            "_text_scale", torch.tensor(TEXT_PASSTHROUGH_WEIGHT**0.5), persistent=True
        )
        self.register_buffer(
            "_learned_scale", torch.tensor((1.0 - TEXT_PASSTHROUGH_WEIGHT) ** 0.5),
            persistent=True,
        )

    def forward(self, text_emb, supertype_idx, rarity_idx, ci_idx, layout_idx,
                continuous, keywords, mechanical_tags=None, structured=None):
        """
        Args:
            text_emb:      (B, 384)  float  — pre-computed sentence embeddings
            supertype_idx: (B,)      long   — supertype index
            rarity_idx:    (B,)      long   — rarity index
            ci_idx:        (B,)      long   — color_identity index
            layout_idx:    (B,)      long   — layout index
            continuous:    (B, 2)    float  — [normalized_cmc, normalized_edhrec_rank]
            keywords:      (B, 50)   float  — multi-hot keyword vector
            mechanical_tags: (B, T)  float  — multi-hot mechanical tag vector (optional)
            structured:    (B, 15)   float  — [power/toughness(3), pips(6), colour(6)]
        Returns:
            (B, 128) L2-normalized embeddings
        """
        sup = self.supertype_emb(supertype_idx)
        rar = self.rarity_emb(rarity_idx)
        ci = self.ci_emb(ci_idx)
        lay = self.layout_emb(layout_idx)

        kw = self.keyword_proj(keywords) if self.keyword_proj is not None else keywords

        parts = [text_emb, sup, rar, ci, lay, continuous, kw]

        if self.tag_proj is not None and mechanical_tags is not None:
            parts.append(self.tag_proj(mechanical_tags))
        if self.structured_dim and structured is not None:
            parts.append(structured)

        x = self.mlp(torch.cat(parts, dim=1))

        if self.text_proj is None:
            return F.normalize(x, p=2, dim=1)

        # Normalize each half to unit length, then scale so the squared weights sum
        # to 1. The concatenation is then already unit-norm, and the dot product of
        # two cards is exactly
        #     (1 - W) * cos_learned  +  W * cos_text
        # so the frozen text's contribution is a fixed, known fraction rather than
        # something the optimizer can quietly drive to zero — which is what the
        # previous model did with the only signal that was working.
        learned = F.normalize(x, p=2, dim=1) * self._learned_scale
        text = F.normalize(self.text_proj(text_emb), p=2, dim=1) * self._text_scale
        return torch.cat([learned, text], dim=1)
