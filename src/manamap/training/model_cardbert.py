"""BERT, where the tokens are FIELDS.

## THE ANALOGY, AND WHERE IT STOPS

BERT reads a sequence of word tokens, hides some, and predicts them from the rest
using bidirectional attention. Here the sequence is a CARD: 73 typed fields and 6
text spans, 80 positions counting `[CLS]`. Masking hides fields; the heads
predict them; attention is bidirectional because a card has no reading order.

    BERT                             here
    word token                       one field  (cmc, kw_flying, produces_U…)
    position embedding               FIELD embedding — identity, not order
    WordPiece vocabulary             one input projection per field, since a
                                     numeric field is 4 columns and a subtype
                                     set is 258
    softmax over 30k words           one head per field, typed to what it holds
    [CLS] for classification         [CLS] IS THE PRODUCT — the card embedding

Two things do not carry over, and both come from the same fact: a card has 80
fields where a sentence has hundreds of tokens.

**Masking a single field is not a task.** The recoverability audit found 19 of 73
solved by a linear probe from the rest — `cmc` is the pips added up. So
`masking.GROUPS` hides correlated blocks, not independent positions.

**Order carries no information.** BERT's positions are a sequence; ours are a
set, and the embedding is there to tell the model WHICH field it is looking at,
not where it sits. The consequence is narrower than it first looks and the first
draft of this paragraph overstated it: shuffling the schema DOES change the
model, because each position's embedding is learned by index. What holds is that
attention is permutation-equivariant, so a field carries no information about
what sits next to it — and the schema order must therefore stay FIXED once a
model is trained, which is why `model_cardbert.pt` stores the field list it was
trained against.

## WHY THE TEXT SPANS ARE FROZEN VECTORS

The previous architecture fine-tuned MiniLM end to end and its control said what
that bought: a RANDOM 128-d projection of frozen MiniLM beat the trained model on
theme (0.444 vs 0.387), and PCA beat it on everything. There is not enough text
here to learn a better text encoder — 1.4M tokens against BERT-base's billions —
so the sentence vectors stay frozen and cached, and what gets learned is how a
card's PARTS RELATE, which is the thing 34,890 examples can actually support.

## THE SPAN HEAD IS CONTRASTIVE, NEVER REGRESSIVE

A masked span's target is a 384-d vector, and MSE against it has a trivial
minimum: predict the corpus mean. The loss falls, every prediction is the same
vector, and nothing has been learned — the exact failure the VAE runs kept
producing under a different name. The head scores its prediction against the true
span AND every other span in the batch, so being right means being CLOSER TO THIS
SPAN THAN TO 255 OTHERS.
"""

import torch
from torch import nn

CLS = 0


def _head_for(kind, d_model, width):
    """One output head per field, typed to what the field holds."""
    if kind == "numeric":
        return nn.Linear(d_model, 1)            # the scaled value; flags are inputs
    return nn.Linear(d_model, width)            # logits: binary 1, categorical/set V


class CardBERT(nn.Module):
    """Fields in, one card embedding out, plus a prediction for every field."""

    def __init__(self, schema, span_slots, span_dim=384, d_model=256,
                 layers=4, heads=8, latent=128, dropout=0.1):
        super().__init__()
        self.schema = schema
        self.span_slots = list(span_slots)
        self.span_dim = span_dim
        self.latent_dim = latent
        self.d_model = d_model

        # ONE PROJECTION PER FIELD. A shared projection would have to accept a
        # single width, and these run from 3 columns to 258.
        self.field_in = nn.ModuleList(
            [nn.Linear(f.total_width, d_model) for f in schema])
        # +2 for the span's own present/masked flags, the same contract the
        # tabular fields carry.
        self.span_in = nn.ModuleList(
            [nn.Linear(span_dim + 2, d_model) for _ in self.span_slots])

        n_positions = 1 + len(schema) + len(self.span_slots)
        self.position = nn.Embedding(n_positions, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm_in = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

        self.to_latent = nn.Linear(d_model, latent)
        # `nn.ModuleDict` REJECTS THE KEY "type" — it collides with an attribute
        # on Module. Prefixing every head is the fix, and it cost an afternoon
        # once already.
        self.heads = nn.ModuleDict({
            f"head_{f.name}": _head_for(f.kind, d_model, f.width) for f in schema})
        self.span_heads = nn.ModuleDict({
            f"head_{slot}": nn.Linear(d_model, span_dim) for slot in self.span_slots})

    # ── input assembly ────────────────────────────────────────────────

    def _tokens(self, tabular, spans, tab_offsets, span_offsets):
        batch = tabular.shape[0]
        parts = [self.cls.expand(batch, -1, -1)]
        for i, field in enumerate(self.schema):
            lo, hi = tab_offsets[field.name]
            parts.append(self.field_in[i](tabular[:, lo:hi]).unsqueeze(1))
        for i, slot in enumerate(self.span_slots):
            lo, hi = span_offsets[slot]
            parts.append(self.span_in[i](spans[:, lo:hi]).unsqueeze(1))
        stacked = torch.cat(parts, dim=1)
        index = torch.arange(stacked.shape[1], device=stacked.device)
        return self.drop(self.norm_in(stacked + self.position(index).unsqueeze(0)))

    def forward(self, tabular, spans, tab_offsets, span_offsets):
        hidden = self.encoder(self._tokens(tabular, spans, tab_offsets, span_offsets))
        latent = self.to_latent(hidden[:, CLS])
        predictions = {}
        for i, field in enumerate(self.schema):
            predictions[field.name] = self.heads[f"head_{field.name}"](hidden[:, 1 + i])
        base = 1 + len(self.schema)
        for i, slot in enumerate(self.span_slots):
            predictions[f"span:{slot}"] = self.span_heads[f"head_{slot}"](
                hidden[:, base + i])
        return {"latent": latent, "hidden": hidden, "predictions": predictions}

    @torch.no_grad()
    def embed(self, tabular, spans, tab_offsets, span_offsets):
        """The card embedding, L2-NORMALISED.

        `analysis/common.py:60` computes cosine as a raw dot product, so a space
        written unnormalised is silently scored on magnitude at eight call sites.
        Normalising here is cheaper than fixing all eight and safer than
        remembering to.
        """
        latent = self.forward(tabular, spans, tab_offsets, span_offsets)["latent"]
        return latent / latent.norm(dim=-1, keepdim=True).clamp(min=1e-9)
