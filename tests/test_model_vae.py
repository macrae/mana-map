"""The variational bottleneck: the guards that keep the latent from being a constant.

The latent IS the product here, so posterior collapse is not a quality problem,
it is a ship-stopper — every downstream metric would be measuring a constant and
none of them would say so. These tests are the alarm.
"""

import numpy as np
import pytest
import torch

from manamap.training import model_vae as MV
from manamap.training.card_serialize import BLOCKS


def test_free_bits_charges_nothing_at_the_prior():
    """A dimension sitting exactly on the prior must be free, or the optimiser
    is paying to keep dimensions it has already switched off."""
    loss, per_dim = MV.kl_with_free_bits(torch.zeros(64, 8), torch.zeros(64, 8))
    assert float(loss) == pytest.approx(0.0, abs=1e-9)
    assert per_dim.shape == (8,)


def test_free_bits_is_per_dimension_not_a_total_budget():
    """THE WHOLE MECHANISM, and the version that looks equivalent and is not.

    A TOTAL floor lets the model switch most dimensions off and spend the entire
    allowance on a few — which is posterior collapse wearing a healthy total.
    A PER-DIMENSION floor removes the reward for switching any of them off.

    One loud dimension and seven silent ones must still be charged for the loud
    one; under a total floor of 8 x 0.5 = 4.0 nats it would be free.
    """
    mu = torch.zeros(64, 8)
    mu[:, 0] = 4.0                       # one dimension far from the prior
    loss, per_dim = MV.kl_with_free_bits(mu, torch.zeros(64, 8), free_bits=0.5)
    assert float(loss) > 7.0, "a loud dimension must be charged"
    assert MV.active_units(per_dim) == 1


def test_active_units_detects_a_collapsed_latent():
    collapsed = torch.full((128,), 1e-6)
    healthy = torch.full((128,), 0.9)
    assert MV.active_units(collapsed) == 0
    assert MV.active_units(healthy) == 128
    assert MV.MIN_ACTIVE_UNITS > 0, "a floor of 0 would never fire"


def test_beta_anneals_from_zero():
    """Reconstruction has to establish before the prior starts pulling, or the
    model learns to satisfy the KL by ignoring z."""
    assert MV.beta_at(0, 1000) == 0.0
    assert 0 < MV.beta_at(150, 1000) < MV.BETA
    assert MV.beta_at(900, 1000) == pytest.approx(MV.BETA)
    assert MV.beta_at(0, 0) == MV.BETA, "a degenerate schedule must not divide by zero"


def test_the_decoder_never_sees_the_encoder():
    """No skip connection, no cross-attention. If `z` is uninformative the loss
    cannot go down — which is what makes collapse visible instead of survivable."""
    import inspect

    source = inspect.getsource(MV.CardVAE.forward)
    assert "self.decoder(z" in source
    assert "last_hidden_state" not in source, "the decoder is reading the encoder"
    model = MV.CardVAE(unfreeze=0)
    names = [n for n, _ in model.decoder.named_parameters()]
    assert names, "decoder has no parameters"


def test_eval_returns_the_mean_and_is_deterministic():
    """The artifact is `mu`, not a draw. A sampled embedding makes every
    neighbour list non-reproducible, and this repo's evidence contract rests on
    a seeded rerun producing the same bytes."""
    model = MV.CardVAE(unfreeze=0)
    ids = torch.randint(0, 20000, (4, 16))
    mask = torch.ones(4, 16, dtype=torch.long)
    model.eval()
    assert torch.equal(model(ids, mask)["mu"], model(ids, mask)["mu"])
    mu, logvar = model.encode(ids, mask)
    assert torch.equal(model.reparameterize(mu, logvar), mu)
    model.train()
    assert not torch.equal(model.reparameterize(mu, logvar),
                           model.reparameterize(mu, logvar)), "training must sample"


def test_reconstruction_scores_only_the_masked_blocks():
    """Scoring a visible block rewards copying a value the model can already
    see — the failure the recoverability audit exists to prevent."""
    logits = {b: torch.zeros(4, MV.DECODER_VOCAB) for b in BLOCKS}
    targets = {b: torch.zeros(4, MV.DECODER_VOCAB) for b in BLOCKS}
    none = {b: torch.zeros(4, dtype=torch.bool) for b in BLOCKS}
    assert float(MV.reconstruction_loss(logits, targets, none)) == 0.0
    one = {b: torch.zeros(4, dtype=torch.bool) for b in BLOCKS}
    one["text"][0] = True
    assert float(MV.reconstruction_loss(logits, targets, one)) > 0.0


def test_each_block_gets_its_own_prediction():
    """One shared head conditioned on a block embedding — if the embedding did
    nothing, every block would predict identically and the model could not be
    asked for a type line specifically."""
    model = MV.CardVAE(unfreeze=0)
    model.eval()
    out = model(torch.randint(0, 20000, (2, 8)), torch.ones(2, 8, dtype=torch.long))
    assert set(out["logits"]) == set(BLOCKS)
    assert not torch.equal(out["logits"]["type"], out["logits"]["text"])


def test_unfreezing_is_off_by_default_and_controls_the_trainable_count():
    """MEASURED against 2,235,677 corpus tokens: 0.94M trainable frozen, 4.49M
    at two layers, 11.58M at six. The default is 0 because the corpus holds
    2,705 exact duplicate oracle texts and cannot support the alternative."""
    counts = {}
    for unfreeze in (0, 2):
        model = MV.CardVAE(unfreeze=unfreeze)
        counts[unfreeze] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert MV.CardVAE().unfrozen == 0, "the default must be the defensible one"
    assert counts[0] < counts[2], "unfreeze must actually thaw layers"
    assert counts[0] < 2e6, f"the frozen config is {counts[0]/1e6:.2f}M — heads too big"


def test_written_embeddings_are_l2_normalised():
    """`analysis/common.top_k_similar` computes cosine as a RAW DOT PRODUCT with
    no normalisation of its own, and a VAE latent is not unit-norm. Normalise at
    write time or fix eight call sites."""
    model = MV.CardVAE(unfreeze=0)
    loader = [(torch.randint(0, 20000, (3, 8)), torch.ones(3, 8, dtype=torch.long))
              for _ in range(2)]
    out = MV.embeddings_from(model, loader, torch.device("cpu"))
    assert out.shape == (6, model.latent_dim) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_free_bits_sits_below_the_kl_a_run_actually_reaches():
    """RE-INTRODUCING THE BUG THAT WASTED A RUN. At FREE_BITS = 0.5 the first
    full run settled at 0.19-0.31 nats/dim — entirely under the floor — so
    `clamp(per_dim - free_bits, min=0)` was EXACTLY ZERO for all 20 epochs. Beta
    annealed 0 -> 0.25 against a term that was structurally zero, and what
    trained was a denoising autoencoder, not a VAE.

    A floor above the KL the model reaches is not a floor. It is an off switch.
    """
    observed = torch.full((128,), 0.25)          # the first run's steady state
    loss, _ = MV.kl_with_free_bits(torch.zeros(1, 128), torch.zeros(1, 128),
                                   free_bits=MV.FREE_BITS)
    assert MV.FREE_BITS < 0.19, (
        f"FREE_BITS={MV.FREE_BITS} is at or above the 0.19-0.31 nats/dim a real "
        f"run reached; the KL term would be dead again")
    charged = torch.clamp(observed - MV.FREE_BITS, min=0.0).sum()
    assert float(charged) > 0.0, "the regulariser must actually charge something"


def test_a_degenerate_latent_is_caught_even_when_every_unit_is_active():
    """THE INSTRUMENT THAT LIED. The first run reported active_units 128/128 —
    "no collapse" — on a space whose participation ratio was 5.71 of 128, barely
    above the layout space's 3.89. Every dimension carried a trickle of KL while
    the data occupied about six of them.

    `active_units` cannot catch that: with free bits disengaged there is no
    pressure toward the prior, so the alarm is guaranteed silent. Effective
    dimensionality is the honest measure.
    """
    from manamap.analysis.eval_embeddings import effective_dimensionality

    rng = np.random.default_rng(5)
    # a latent that is "fully active" by KL but lies in ~3 dimensions
    core = rng.normal(size=(2000, 3))
    degenerate = np.concatenate([core, rng.normal(size=(2000, 125)) * 0.001], axis=1)
    degenerate /= np.maximum(np.linalg.norm(degenerate, axis=1, keepdims=True), 1e-8)
    assert MV.active_units(torch.full((128,), 0.25)) == 128, "KL says fully active"
    assert effective_dimensionality(degenerate) < MV.MIN_EFFECTIVE_DIM, \
        "…and the participation ratio says it is not"
    assert MV.MIN_EFFECTIVE_DIM > 3.89, "the floor must sit above the layout space"
