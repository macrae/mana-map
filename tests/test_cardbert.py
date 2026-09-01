"""The masking strategy, the loss, and the two places this has a second
implementation of a rule that already exists somewhere else."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from manamap.training import card_fields as CF
from manamap.training import masking
from manamap.training import span_encoder as SE
from manamap.training.loss_cardbert import imputation_loss, info_nce, span_recall


@pytest.fixture(scope="module")
def schema():
    card = {"cmc": 1, "power": "1", "toughness": "1", "type_line": "Creature — Elf",
            "mana_cost": "{G}", "supertype": "Creature", "rarity": "common",
            "layout": "normal", "color_identity": "G", "keywords": "Flying",
            "edhrec_rank": 500, "oracle_text": "Flying", "produced_mana": []}
    return CF.build_schema(CF.vocabularies([card]))


# ── the masking strategy ────────────────────────────────────────────────


def test_every_field_belongs_to_exactly_one_group(schema):
    """A field in no group is never masked and never a target — it rides along
    as an input forever and nothing says so. A field in TWO groups is masked more
    often than the draw implies, quietly skewing the objective."""
    seen = {}
    for group, names in masking.GROUPS.items():
        for name in names:
            assert name not in seen, f"{name} is in both {seen.get(name)} and {group}"
            seen[name] = group
    assert masking.unassigned(schema) == []


def test_masking_a_keyword_block_also_hides_the_keyword_TEXT(schema):
    """99% of keywords appear verbatim in the oracle text. Hiding `kw_flying`
    while the keyword span still reads "Flying" is not a task, it is a lookup."""
    fields, slots = masking.resolve(["keywords"])
    assert "kw_flying" in fields
    assert "keyword" in slots, "the keyword span must be hidden with the flags"

    fields, slots = masking.resolve(["mana"])
    assert "keyword" not in slots, "only the keyword group carries that companion"


def test_the_mana_group_hides_everything_that_reconstructs_a_cost():
    """`cmc` is the pips added up — the audit measures R^2 0.96 — so hiding it
    alone is arithmetic. The whole block goes together or none of it is a task."""
    fields, _ = masking.resolve(["mana"])
    for name in ("cmc", "generic_pips", "pips_W", "pips_G", "color_identity"):
        assert name in fields, name


def test_a_draw_never_returns_nothing():
    """An example with nothing hidden contributes no imputation gradient and
    costs a full forward pass."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        assert masking.draw(rng), "empty draw"


def test_derived_fields_are_never_targets():
    """Masking `is_artifact_creature` while `is_artifact` and `is_creature` stay
    visible hides nothing, so scoring the model on it measures nothing."""
    weights = masking.loss_weights({n: {"lift": 0.0} for n in CF.DERIVED_FIELDS})
    assert all(weights[n] == 0.0 for n in CF.DERIVED_FIELDS)


def test_a_trivial_field_is_demoted_but_not_deleted():
    """It stays in the loss as a consistency term. Zeroing it outright removes
    the only signal that the model has forgotten how to add."""
    weights = masking.loss_weights({"cmc": {"lift": 0.9647},
                                    "kw_deathtouch": {"lift": -0.2592}})
    assert 0.0 < weights["cmc"] < 0.1
    assert weights["kw_deathtouch"] == pytest.approx(1.0)


# ── the loss ────────────────────────────────────────────────────────────


def test_infonce_punishes_the_mean_vector_that_mse_rewards():
    """THE LOAD-BEARING CHOICE. A regression head's cheapest win is to predict
    the corpus mean: the loss falls, every prediction is identical, nothing is
    learned. This is the failure the previous architecture kept producing."""
    torch.manual_seed(0)
    true = torch.randn(64, 384)
    mean = true.mean(0, keepdim=True).expand(64, -1)

    # MSE ACTIVELY PREFERS the degenerate answer to an honest random guess.
    assert F.mse_loss(mean, true) < F.mse_loss(torch.randn(64, 384), true)

    # InfoNCE does not: the mean vector is equidistant from everything, so it
    # scores at chance.
    assert info_nce(mean, true) > 4.0
    assert info_nce(true, true) < 0.01
    assert span_recall(mean, true) < 0.05          # chance is 1/64
    assert span_recall(true, true) == 1.0


def test_the_batched_loss_agrees_with_scoring_one_field_at_a_time():
    """48 binary fields scored in one op instead of 48 cost 221ms -> a few. Two
    implementations of one rule is the divergence this repo has paid for, so the
    fast path is held to the slow one."""
    class Field:
        def __init__(self, name, kind):
            self.name, self.kind, self.width = name, kind, 1

    torch.manual_seed(0)
    n = 64
    schema = ([Field(f"b{i}", "binary") for i in range(20)]
              + [Field(f"n{i}", "numeric") for i in range(5)])
    predictions = {f.name: torch.randn(n, 1) for f in schema}
    targets = {f.name: (torch.rand(n, 1) > 0.5).float() if f.kind == "binary"
               else torch.randn(n, 1) for f in schema}
    present = {f.name: (torch.rand(n) > 0.1).float() for f in schema}
    mask = {f.name: (torch.rand(n) > 0.5).float() for f in schema}
    weights = {f.name: 0.3 + 0.7 * torch.rand(1).item() for f in schema}

    batched, _ = imputation_loss(predictions, targets, present, mask, weights, schema)

    reference = torch.zeros(())
    for field in schema:
        selected = mask[field.name] * present[field.name]
        count = selected.sum()
        if float(count) == 0:
            continue
        per = (F.huber_loss(predictions[field.name][:, 0], targets[field.name][:, 0],
                            reduction="none") if field.kind == "numeric"
               else F.binary_cross_entropy_with_logits(
                   predictions[field.name][:, 0], targets[field.name][:, 0],
                   reduction="none"))
        reference = reference + weights[field.name] * ((per * selected).sum() / count)

    assert float(batched) == pytest.approx(float(reference), abs=1e-5)


def test_an_absent_field_is_not_scored(schema):
    """A land has no power. Scoring the model on it is scoring a question with
    no answer, and it would reward predicting whatever the padding happens to be."""
    field = next(f for f in schema if f.name == "power")
    predictions = {field.name: torch.zeros(4, 1)}
    targets = {field.name: torch.zeros(4, 1)}
    mask = {field.name: torch.ones(4)}
    present = {field.name: torch.zeros(4)}          # absent on every card
    total, detail = imputation_loss(predictions, targets, present, mask,
                                    {field.name: 1.0}, [field])
    assert float(total) == 0.0
    assert field.name not in detail["counts"]


# ── the fast masking path against the encoder that defines it ───────────


def test_masking_the_matrix_matches_encoding_with_masked():
    """`train_cardbert.mask_batch` zeroes columns on a precomputed matrix rather
    than re-encoding, because encoding 34,890 cards per epoch would dominate
    training. That makes it a SECOND implementation of what `encode(masked=…)`
    already does, so it is held to the original."""
    from manamap.training import train_cardbert as T

    card = {"cmc": 3, "power": "2", "toughness": "3", "mana_cost": "{1}{G}{G}",
            "type_line": "Creature — Elf Druid", "supertype": "Creature",
            "rarity": "rare", "layout": "normal", "color_identity": "G",
            "keywords": "Flying, Trample", "edhrec_rank": 120,
            "oracle_text": "Flying\n{T}: Add {G}.", "produced_mana": ["G"]}
    schema = CF.build_schema(CF.vocabularies([card]))
    cache = SE.SpanCache(
        np.random.default_rng(0).normal(size=(40, 8)).astype(np.float32),
        SE.unique_spans([card]))

    tabular, tab_offsets = CF.encode(card, schema)
    spans, span_offsets = cache.encode(card, card["oracle_text"])
    tab_cols, span_cols = T.group_columns(schema, tab_offsets, span_offsets)

    for group in ("mana", "keywords", "types", "body"):
        fields, slots = masking.resolve([group])
        want_tab, _ = CF.encode(card, schema, masked=[f for f in fields
                                                      if f in {x.name for x in schema}])
        want_span, _ = cache.encode(card, card["oracle_text"], masked=slots)
        got_tab, got_span = T.mask_batch(
            torch.from_numpy(tabular).unsqueeze(0),
            torch.from_numpy(spans).unsqueeze(0), [[group]], tab_cols, span_cols)
        assert np.allclose(got_tab[0].numpy(), want_tab), f"{group}: tabular differs"
        assert np.allclose(got_span[0].numpy(), want_span), f"{group}: spans differ"


def test_the_split_keeps_duplicate_oracle_texts_together():
    """2,705 cards share an exact oracle text with another card. A row split puts
    the same text on both sides and every held-out number comes back inflated —
    the one control this architecture rests on, quietly broken."""
    from manamap.training import train_cardbert as T

    cards = ([{"oracle_text": "Flying", "type_line": "Creature — Bird"}] * 50
             + [{"oracle_text": "Draw a card.", "type_line": "Instant"}] * 50)
    held = T.split_mask(cards)
    assert len(set(held[:50])) == 1, "identical cards landed on both sides"
    assert len(set(held[50:])) == 1


# ── the model ───────────────────────────────────────────────────────────


def test_a_forward_pass_predicts_every_field(schema):
    from manamap.training.model_cardbert import CardBERT

    model = CardBERT(schema, SE.SPAN_SLOTS, span_dim=8, d_model=32, layers=1, heads=2)
    tabular = torch.zeros(2, sum(f.total_width for f in schema))
    spans = torch.zeros(2, (8 + 2) * len(SE.SPAN_SLOTS))
    _, tab_offsets = CF.encode({"type_line": "Creature"}, schema)
    span_offsets = {s: (i * 10, i * 10 + 10) for i, s in enumerate(SE.SPAN_SLOTS)}

    out = model(tabular, spans, tab_offsets, span_offsets)
    assert out["latent"].shape == (2, model.latent_dim)
    for field in schema:
        assert field.name in out["predictions"]
    for slot in SE.SPAN_SLOTS:
        assert out["predictions"][f"span:{slot}"].shape == (2, 8)


def test_the_written_embedding_is_unit_norm(schema):
    """`analysis/common.py:60` computes cosine as a RAW DOT PRODUCT, so a space
    written unnormalised is silently scored on magnitude at eight call sites."""
    from manamap.training.model_cardbert import CardBERT

    model = CardBERT(schema, SE.SPAN_SLOTS, span_dim=8, d_model=32, layers=1, heads=2)
    tabular = torch.randn(5, sum(f.total_width for f in schema))
    spans = torch.randn(5, (8 + 2) * len(SE.SPAN_SLOTS))
    _, tab_offsets = CF.encode({"type_line": "Creature"}, schema)
    span_offsets = {s: (i * 10, i * 10 + 10) for i, s in enumerate(SE.SPAN_SLOTS)}

    norms = model.embed(tabular, spans, tab_offsets, span_offsets).norm(dim=-1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-5)


# ── the embedding must actually be trained ─────────────────────────────


def test_the_latent_projection_receives_gradient(schema):
    """THE BUG THIS TEST EXISTS FOR SHIPPED AND WAS CAUGHT BY THE EVAL.

    Every imputation head reads its own field's position; nothing reads `[CLS]`.
    So `to_latent.weight.grad` came back **None** after a full backward pass, and
    the embedding written to disk was a random projection of an untrained state.
    It scored like one: r@10 0.093 against the function space's 0.232, with 5.53
    of 128 dimensions in use.

    This is the textbook BERT result — a raw `[CLS]` is a poor sentence
    embedding, which is why SBERT exists — reached from first principles. BERT
    survives it because it is always fine-tuned downstream. Here the embedding IS
    the product.
    """
    from manamap.training.loss_cardbert import (field_present, field_targets,
                                                imputation_loss, view_contrastive)
    from manamap.training.model_cardbert import CardBERT

    model = CardBERT(schema, SE.SPAN_SLOTS, span_dim=8, d_model=32, layers=1, heads=2)
    tabular = torch.randn(8, sum(f.total_width for f in schema))
    spans = torch.randn(8, (8 + 2) * len(SE.SPAN_SLOTS))
    _, tab_offsets = CF.encode({"type_line": "Creature"}, schema)
    span_offsets = {s: (i * 10, i * 10 + 10) for i, s in enumerate(SE.SPAN_SLOTS)}

    # IMPUTATION ALONE leaves it untrained — the failure, pinned.
    out = model(tabular, spans, tab_offsets, span_offsets)
    loss, _ = imputation_loss(
        out["predictions"], field_targets(tabular, tab_offsets, schema),
        field_present(tabular, tab_offsets, schema),
        {f.name: torch.ones(8) for f in schema},
        {f.name: 1.0 for f in schema}, schema)
    loss.backward()
    assert model.to_latent.weight.grad is None, (
        "imputation now reaches [CLS] — if that is deliberate, this test should "
        "say so instead of asserting the gap it was written for")

    # THE VIEW TERM trains it.
    model.zero_grad()
    a = model(tabular, spans, tab_offsets, span_offsets)["latent"]
    b = model(tabular * 0.9, spans, tab_offsets, span_offsets)["latent"]
    view_contrastive(a, b).backward()
    assert model.to_latent.weight.grad is not None
    assert float(model.to_latent.weight.grad.norm()) > 0


def test_two_views_of_one_card_are_pulled_together():
    """Masking IS the augmentation: two draws hide different parts of the same
    card, and agreeing across them is the invariance a similarity space wants."""
    from manamap.training.loss_cardbert import view_agreement, view_contrastive

    torch.manual_seed(0)
    identical = torch.randn(32, 64)
    scrambled = torch.randn(32, 64)

    import math

    # Thresholds MEASURED, not guessed: at temperature 0.2 a perfect pair scores
    # 0.223 rather than ~0, because 31 negatives at cosine ~0 still carry weight
    # in the softmax. The first version of this test asserted < 0.05 and failed.
    chance = math.log(len(identical))                    # 3.466 for 32
    assert float(view_contrastive(identical, identical)) < 0.3
    assert float(view_contrastive(identical, scrambled)) == pytest.approx(chance, abs=0.2)
    assert view_agreement(identical, identical) == 1.0
    assert view_agreement(identical, scrambled) < 0.2
