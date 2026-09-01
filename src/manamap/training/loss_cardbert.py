"""The imputation objective: one loss per field kind, scored only where masked.

## SCORED ONLY WHERE MASKED

BERT's MLM loss is computed at masked positions and nowhere else, and the reason
is the same here: a field the model can SEE is not a prediction, and including it
turns most of the objective into an autoencoder that learns to copy its input.
Every term below is multiplied by the per-example mask.

## ONE LOSS PER KIND

    numeric      Huber on the scaled value. Not MSE: `edhrec_rank` and `cmc` both
                 have long tails, and a squared penalty lets a handful of
                 20-drops dominate a batch.
    binary       BCE with logits.
    categorical  cross-entropy over the field's vocabulary.
    set          multi-label BCE — a card has several subtypes, not one.
    span         InfoNCE. NEVER MSE — see below.

## THE SPAN LOSS IS CONTRASTIVE, AND THIS IS THE LOAD-BEARING CHOICE

A masked span's target is a 384-d frozen sentence vector. MSE against it has a
trivial minimum: **predict the corpus mean**. The loss falls smoothly, every
prediction converges to the same vector, and the model has learned nothing — the
failure the VAE runs produced repeatedly under other names, and the reason a
random projection beat a trained model on the previous architecture.

InfoNCE removes that minimum. The prediction is scored against the true span AND
every other span in the batch, so the mean vector is the WORST possible answer:
it is equally close to all of them. Being right means being closer to this span
than to the other 255.

## WEIGHTS COME FROM THE AUDIT, NOT FROM TASTE

`masking.loss_weights` turns the recoverability probe into a weight per field.
19 of 73 fields are solved by a linear probe from the others, and left at weight
1.0 they would contribute the bulk of a very satisfying-looking loss curve while
teaching the model arithmetic. They keep a floor rather than a zero, so they
still fail loudly if the model forgets how to add.
"""

import torch
import torch.nn.functional as F

#: InfoNCE temperature. 0.07 is the CLIP/SimCLR default and is not tuned here —
#: if it is ever swept, sweep it against the eval, not against the loss.
TEMPERATURE = 0.07


def field_targets(unmasked, offsets, schema):
    """`{field: tensor}` — the TRUE value of every field, from the unmasked view.

    Targets are read from the unmasked encoding rather than from the card, so
    the loss can never disagree with the encoder about what a field holds. If
    `card_fields` changes how it scales a number, the target changes with it.
    """
    out = {}
    for field in schema:
        lo, hi = offsets[field.name]
        values = unmasked[:, lo:hi - 2]                 # drop the two state flags
        if field.kind == "numeric":
            out[field.name] = values[:, :1]
        elif field.kind == "categorical":
            out[field.name] = values.argmax(dim=1)
        else:
            out[field.name] = values                    # binary (1) / set (V)
    return out


def field_present(unmasked, offsets, schema):
    """`{field: 0/1}` — was the field PRESENT on this card?

    An ABSENT field has no value to predict, so it is dropped from the loss even
    when it was drawn for masking. Scoring a model on the power of a land is
    scoring it on a question with no answer.
    """
    return {f.name: unmasked[:, offsets[f.name][1] - 2] for f in schema}


def imputation_loss(predictions, targets, present, mask, weights, schema,
                    span_targets=None, span_mask=None, slots=()):
    """`(total, {term: value})` — the objective and its parts, for reporting.

    `mask[field]` is 1 where that field was hidden for that example.

    BINARY AND NUMERIC FIELDS ARE SCORED IN ONE OP EACH. There are 48 binary
    fields and 11 numeric ones, and a separate tiny kernel per field cost **221
    ms per batch** — more than the forward pass. Stacking them changes no
    arithmetic: each field still carries its own weight and its own mask, and a
    test asserts the batched path agrees with the per-field one to 1e-6.
    """
    total = predictions[schema[0].name].new_zeros(())
    parts, counts = {}, {}

    stacked = {"binary": [], "numeric": []}
    for field in schema:
        weight = float(weights.get(field.name, 1.0))
        if weight <= 0.0:
            continue
        selected = mask[field.name] * present[field.name]
        if field.kind in stacked:
            stacked[field.kind].append((field, weight, selected))
            continue
        n = selected.sum()
        if float(n) == 0.0:
            continue
        logits, target = predictions[field.name], targets[field.name]
        if field.kind == "categorical":
            per = F.cross_entropy(logits, target, reduction="none")
        else:
            per = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none").mean(dim=1)
        term = (per * selected).sum() / n
        total = total + weight * term
        parts[field.name] = float(term.detach())
        counts[field.name] = int(n)

    for kind, entries in stacked.items():
        if not entries:
            continue
        logits = torch.stack([predictions[f.name][:, 0] for f, _, _ in entries], dim=1)
        target = torch.stack([targets[f.name][:, 0] for f, _, _ in entries], dim=1)
        select = torch.stack([s for _, _, s in entries], dim=1)
        weight = logits.new_tensor([w for _, w, _ in entries])
        per = (F.huber_loss(logits, target, reduction="none") if kind == "numeric"
               else F.binary_cross_entropy_with_logits(logits, target, reduction="none"))
        # Per FIELD: sum over the examples that had it masked, divided by how
        # many those were — identical to scoring each field on its own.
        n = select.sum(dim=0)
        term = (per * select).sum(dim=0) / n.clamp(min=1)
        total = total + (term * weight * (n > 0)).sum()
        for i, (field, _, _) in enumerate(entries):
            if float(n[i]) > 0:
                parts[field.name] = float(term[i].detach())
                counts[field.name] = int(n[i])

    for slot in slots:
        key = f"span:{slot}"
        selected = span_mask[slot]
        n = selected.sum()
        if float(n) < 2:            # InfoNCE needs negatives to be a task at all
            continue
        rows = selected.bool()
        term = info_nce(predictions[key][rows], span_targets[slot][rows])
        total = total + term
        parts[key] = float(term.detach())
        counts[key] = int(n)

    return total, {"parts": parts, "counts": counts}


#: Weight on the card-level contrastive term. Large enough that `[CLS]` is
#: genuinely shaped by it, small enough that imputation stays the main task.
VIEW_WEIGHT = 1.0

#: A card-level view pair is a WEAKER signal than a span pair — two maskings of
#: one card are far more alike than two different spans — so it gets a warmer
#: temperature, which is the standard SimCLR range rather than CLIP's 0.07.
VIEW_TEMPERATURE = 0.2


def view_contrastive(latent_a, latent_b, temperature=VIEW_TEMPERATURE):
    """Two maskings of the SAME card should agree; different cards should not.

    ## WHY THIS TERM EXISTS AT ALL

    Without it `[CLS]` receives NO GRADIENT. Measured, not suspected:
    `to_latent.weight.grad` came back **None** after a full backward pass,
    because every imputation head reads its own field's position and nothing
    reads `[CLS]`. The shipped embedding was therefore a random projection of an
    untrained state, and it scored like one — r@10 0.093 against the function
    space's 0.232, with a participation ratio of 5.53 of 128 dimensions.

    This is the textbook BERT problem, arrived at from first principles: a raw
    `[CLS]` is a poor sentence embedding, which is the entire reason SBERT
    exists. BERT gets away with it because it is always fine-tuned downstream.
    Here the embedding IS the product, so it has to be trained on purpose.

    ## MASKING IS THE AUGMENTATION

    SimCLR needs two views of one example and usually crops or colour-jitters an
    image to get them. Here the augmentation is already the objective: draw two
    independent maskings of one card and the model sees two genuinely different
    subsets of the same object. Agreeing across them is exactly the invariance a
    similarity space wants — a card is the same card whether or not you can see
    its mana cost.
    """
    a = F.normalize(latent_a, dim=-1)
    b = F.normalize(latent_b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.T, labels))


@torch.no_grad()
def view_agreement(latent_a, latent_b):
    """Fraction of cards whose nearest neighbour across views is themselves."""
    a = F.normalize(latent_a, dim=-1)
    b = F.normalize(latent_b, dim=-1)
    nearest = (a @ b.T).argmax(dim=1)
    return float((nearest == torch.arange(len(nearest), device=nearest.device))
                 .float().mean())


def info_nce(predicted, actual, temperature=TEMPERATURE):
    """Contrastive loss against every other span in the batch.

    Both sides are L2-normalised, so the score is a cosine and the loss cannot be
    reduced by inflating magnitudes — which is the other way a regression head
    finds a shortcut.
    """
    predicted = F.normalize(predicted, dim=-1)
    actual = F.normalize(actual, dim=-1)
    logits = predicted @ actual.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    # Symmetric: predicting the span from the card AND the card from the span.
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.T, labels))


@torch.no_grad()
def accuracy(predictions, targets, present, mask, schema):
    """Per-field held-out accuracy at MASKED positions. The number that matters.

    Reported alongside the loss because a loss is not interpretable across kinds:
    a Huber of 0.3 and a BCE of 0.3 are not comparable, and neither says whether
    the model can actually tell a flier from a ground creature.
    """
    out = {}
    for field in schema:
        selected = (mask[field.name] * present[field.name]).bool()
        if int(selected.sum()) == 0:
            continue
        logits, target = predictions[field.name][selected], targets[field.name][selected]
        if field.kind == "numeric":
            out[field.name] = float((logits[:, 0] - target[:, 0]).abs().mean())
        elif field.kind == "binary":
            out[field.name] = float(((logits[:, 0] > 0) == (target[:, 0] > 0.5)).float().mean())
        elif field.kind == "categorical":
            out[field.name] = float((logits.argmax(dim=1) == target).float().mean())
        else:
            out[field.name] = float(((logits > 0) == (target > 0.5)).float().mean())
    return out


@torch.no_grad()
def span_recall(predicted, actual, k=1):
    """Is the true span the nearest of the batch? The span head's real score."""
    predicted = F.normalize(predicted, dim=-1)
    actual = F.normalize(actual, dim=-1)
    ranks = (predicted @ actual.T).argsort(dim=1, descending=True)
    labels = torch.arange(ranks.shape[0], device=ranks.device).unsqueeze(1)
    return float((ranks[:, :k] == labels).any(dim=1).float().mean())
