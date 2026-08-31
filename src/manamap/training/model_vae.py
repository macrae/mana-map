"""A variational bottleneck over masked card imputation.

Replaces a contrastive objective whose positives are mined from the repo's own
regexes (`train_ability._positive`: 53 roles, 33 tags, then random). That model
needs LABELS; this one needs none, because **the label is the input** — hide a
block of the card, reconstruct it. `card_serialize` builds the masked view.

## WHAT IT HAS TO BEAT, measured 2026-08-31 before a line of this was written

    relation / property        current function space   target
    function (pool 500)                        +0.165   hold
    theme    (pool 500)                        -0.371   close
    hard negatives (mean 1-cos)                 0.0133  raise
    centroid headroom (1 - cos)                 0.019   raise

Three of the four are failures the contrastive objective caused: it mines
positives from roles and tags, "Vampire" is neither, so the space discards tribe
by design and scores **0.005** against text's 0.470 on tribal commanders. Masked
imputation of the TYPE block attacks exactly that — the latent cannot reconstruct
"Legendary Creature — Dinosaur Avatar" without carrying tribe.

## HOW MUCH OF THE ENCODER TO UNFREEZE — the pilot, and why the default is 0

WALL TIME, measured on this machine (MPS, batch 32, seq 128, 27,848 train rows).
The encoder dominates it; the decoder head is cheap to compute:

    unfrozen layers   s/epoch   20 epochs   10 epochs
    none (head only)      109         36m         18m
    top 2                 208         69m         35m
    top 4                 276         92m         46m
    all 6                 362        121m         60m

PARAMETERS are a different question and the pilot got them wrong at first: it
timed a toy `Linear(384, 256)` and recorded "head only = 0.1M". The real model's
trainable count, against 2,235,677 serialised corpus tokens:

    unfreeze=0    0.94M trainable    2.39 tokens/param
    unfreeze=2    4.49M trainable    0.50 tokens/param
    unfreeze=6   11.58M trainable    0.19 tokens/param

Chinchilla-optimal is ~20 tokens/param. It is a from-scratch rule and this is a
fine-tune, so it does not bind directly — but the ORDERING is the point, and the
corpus holds 2,705 EXACT duplicate oracle texts waiting to be memorised.
Freezing most of the encoder is not a compromise for speed; it is the
regularisation the data size demands.

So `unfreeze` defaults to 0 and the knob exists so the claim can be re-measured
rather than argued about. Split by TEXT HASH, never by row.

## POSTERIOR COLLAPSE IS THE SHIP-STOPPER, SO IT IS INSTRUMENTED

The latent IS the product here. A VAE whose decoder is strong enough to ignore
`z` will happily drive KL to zero and reconstruct from the visible blocks alone,
and every downstream metric would then be measuring a constant. Three defences,
all of them standard and none of them optional:

  * **The decoder never sees the encoder.** Heads read `z` and nothing else — no
    skip connection, no cross-attention. If `z` is uninformative the loss cannot
    go down.
  * **Free bits**: each dimension is allowed `FREE_BITS` nats of KL for nothing,
    so the optimiser has no incentive to switch a dimension off to save budget.
  * **KL annealing**: beta ramps from 0, so reconstruction establishes itself
    before the prior starts pulling.

`active_units` is reported every epoch and a run that falls below the floor is a
FAILED run that says so, rather than a quiet one that ships a constant.

## THE DECODER IS A BAG OF TOKENS, AND THAT IS A REAL LIMITATION

Reconstructing the masked block as an ordered sequence needs an autoregressive
decoder — the most expensive piece in the design, and the one the corpus can
least afford. This predicts the block's tokens as a SET (multi-label BCE over a
restricted vocabulary) instead.

That is enough to force `z` to carry the content: you cannot name the tokens of
"Legendary Creature — Dinosaur Avatar" without encoding the tribe. It is NOT
enough to generate a plausible card, so the interpolation product stays a
retrieval over real cards until a sequence decoder earns its cost. Said here
because a bag-of-tokens decoder quietly described as "reconstruction" would
overclaim.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from manamap.config import FINAL_EMBEDDING_DIM, TEXT_MODEL_NAME
from manamap.training.card_serialize import BLOCKS

#: Nats per dimension the KL term is not charged for.
#:
#: MEASURED AND CORRECTED. At 0.5 the term NEVER ENGAGED: the first full run
#: settled at a KL of 0.19-0.31 nats/dim, entirely under the floor, so
#: `clamp(per_dim - 0.5, min=0)` was exactly zero for all 20 epochs. Beta
#: annealed from 0 to 0.25 against a term that was structurally 0. What trained
#: was a denoising autoencoder with noise on the latent — not a VAE.
#:
#: The floor has to sit BELOW the KL the model naturally reaches or it is not a
#: floor, it is an off switch. 0.1 leaves the regulariser live while still
#: removing the incentive to zero a dimension outright.
FREE_BITS = 0.1

#: Weight on the KL after annealing. Below 1.0 because the latent's job here is
#: to be a good METRIC SPACE, not to be a faithful generative prior.
BETA = 0.25

#: Fraction of training spent ramping beta from 0. Reconstruction first.
ANNEAL_FRACTION = 0.3

#: A run whose active units fall below this has collapsed and is reported failed.
MIN_ACTIVE_UNITS = 32

#: …and the measure that actually caught the degeneracy `active_units` missed.
#: Participation ratio of the latent's PCA spectrum, the same statistic
#: `eval_embeddings.effective_dimensionality` reports. Reference points on the
#: shipped artifacts: function 27.31, text 51.39, layout 3.89 (deliberately
#: trivial), and the first VAE run **5.71**. A floor of 10 is above the layout
#: space and well below the space this is trying to replace.
MIN_EFFECTIVE_DIM = 10.0

#: Vocabulary for the bag-of-tokens decoder. MEASURED over the serialised corpus:
#: 7,507 distinct wordpieces, 2,235,677 token occurrences, and the most frequent
#: 2,000 cover **99.31%** of them (1,000 covers 98.32%, 4,096 covers 99.79%).
#: The head is the largest trainable block in the model, so the last 0.5% of
#: coverage is not worth doubling it for.
DECODER_VOCAB = 2048


class CardVAE(nn.Module):
    """MiniLM encoder -> variational bottleneck -> per-block bag-of-tokens heads."""

    def __init__(self, latent_dim=FINAL_EMBEDDING_DIM, unfreeze=0,
                 vocab_size=DECODER_VOCAB, encoder_name=None, hidden=384):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(encoder_name or
                                                 f"sentence-transformers/{TEXT_MODEL_NAME}")
        self.set_unfrozen(unfreeze)
        width = self.encoder.config.hidden_size
        self.to_mu = nn.Linear(width, latent_dim)
        self.to_logvar = nn.Linear(width, latent_dim)
        # THE DECODER READS `z` AND NOTHING ELSE, and there is ONE of it.
        #
        # Four separate [latent -> hidden -> vocab] heads came to 6.6M trainable
        # parameters with the encoder entirely frozen — the heads, not the
        # encoder, were the model. Against 1.4M corpus tokens that is the
        # over-parameterisation this design is supposed to be avoiding, and the
        # pilot's "head only = 0.1M" figure had measured a toy `Linear(384, 256)`
        # rather than these.
        #
        # One shared head, conditioned on a learned BLOCK EMBEDDING added to `z`,
        # gets the same per-block specialisation for a quarter of the parameters.
        # Sharing is also the honest structure: reconstructing a type line and
        # reconstructing oracle text are the same operation over the same
        # vocabulary, differing in what is being asked for.
        self.block_embedding = nn.Embedding(len(BLOCKS), latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, vocab_size))
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

    def set_unfrozen(self, unfreeze):
        """Freeze everything, then thaw the top `unfreeze` transformer layers."""
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        layers = self.encoder.encoder.layer
        for layer in (layers[len(layers) - unfreeze:] if unfreeze else []):
            for param in layer.parameters():
                param.requires_grad_(True)
        self.unfrozen = unfreeze

    def encode(self, input_ids, attention_mask):
        """Mean-pooled encoder output -> (mu, logvar). Masked-token aware pooling."""
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.to_mu(pooled), self.to_logvar(pooled)

    def reparameterize(self, mu, logvar):
        """Sample at train time; return the MEAN at eval.

        The artifact is `mu`, not a draw — a sampled embedding would make every
        downstream neighbour list non-reproducible, and this repo's whole
        evidence contract rests on a seeded rerun producing the same bytes.
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar.clamp(-10.0, 10.0))
        return mu + std * torch.randn_like(std)

    def forward(self, input_ids, attention_mask):
        mu, logvar = self.encode(input_ids, attention_mask)
        z = self.reparameterize(mu, logvar)
        return {"mu": mu, "logvar": logvar, "z": z,
                "logits": {block: self.decoder(z + self.block_embedding(
                    torch.tensor(i, device=z.device)))
                    for i, block in enumerate(BLOCKS)}}


def kl_with_free_bits(mu, logvar, free_bits=FREE_BITS):
    """Per-dimension KL, floored at `free_bits`. Returns (loss, per-dim KL).

    THE FLOOR IS APPLIED PER DIMENSION, NOT TO THE TOTAL, and that distinction is
    the whole mechanism. A total floor lets the optimiser switch most dimensions
    off and spend the whole allowance on a few; a per-dimension floor removes the
    reward for switching any of them off.
    """
    logvar = logvar.clamp(-10.0, 10.0)
    per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)   # (B, D)
    per_dim_mean = per_dim.mean(dim=0)                          # (D,)
    return torch.clamp(per_dim_mean - free_bits, min=0.0).sum(), per_dim_mean


def active_units(per_dim_kl, threshold=0.01):
    """Dimensions carrying more than `threshold` nats.

    A WEAK ALARM, and the first run proved it. It reported 128/128 "no collapse"
    on a space whose PARTICIPATION RATIO was 5.71 of 128 — barely above the
    layout space's 3.89, which this repo calls deliberately trivial. Every
    dimension carried a trickle of KL while the data occupied about six of them.
    It cannot be otherwise when free bits never engage: with no pressure toward
    the prior there is nothing to collapse, so the alarm is guaranteed silent.

    Keep it — a genuinely collapsed KL is worth catching — but `effective_dim`
    is the honest measure of whether the space is USED, and `train_vae` gates on
    both.
    """
    return int((per_dim_kl > threshold).sum())


def beta_at(step, total_steps, beta=BETA, anneal=ANNEAL_FRACTION):
    """Linear ramp from 0, so reconstruction establishes before the prior pulls."""
    if total_steps <= 0 or anneal <= 0:
        return beta
    return float(beta * min(1.0, step / max(1.0, total_steps * anneal)))


def reconstruction_loss(logits, targets, masked):
    """Multi-label BCE over the masked blocks ONLY.

    Scoring an unmasked block would reward copying a value the model can already
    see, which is the failure the recoverability audit exists to prevent — the
    same shape as predicting `cmc` while `mana_cost` is visible.
    """
    total = logits[BLOCKS[0]].new_zeros(())
    counted = 0
    for block in BLOCKS:
        rows = masked[block].nonzero(as_tuple=True)[0]
        if not len(rows):
            continue
        total = total + F.binary_cross_entropy_with_logits(
            logits[block][rows], targets[block][rows])
        counted += 1
    return total / max(counted, 1)


def embeddings_from(model, loader, device):
    """`mu`, L2-normalised, in corpus order — the artifact every consumer reads.

    NORMALISED AT WRITE TIME because `analysis/common.top_k_similar` computes
    cosine as a raw dot product with no normalisation of its own, and a VAE
    latent is not unit-norm. The alternative is fixing eight call sites.
    """
    model.eval()
    out = []
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            mu, _ = model.encode(input_ids.to(device), attention_mask.to(device))
            out.append(mu.cpu().numpy())
    matrix = np.concatenate(out, axis=0).astype(np.float32)
    norms = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    return matrix / norms
