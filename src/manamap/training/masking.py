"""What to hide from the model, and why hiding one field at a time does not work.

## BERT MASKS INDEPENDENT TOKENS. THESE FIELDS ARE NOT INDEPENDENT.

The recoverability audit measured it: **19 of 73 fields are solved by a linear
probe from the others**, everything unmasked. `cmc` is the pips added up
(R^2 0.96). `supertype` is the type flags (0.998). `color_identity` is the
coloured pips (0.956). `is_artifact_creature` is an AND of two visible flags.

So masking `cmc` on its own is not a task — it is arithmetic, and a model that
scores well on it has learned addition. Masking the whole MANA BLOCK is a task:
with every pip, the cost, the colour identity and the X flag hidden together, the
only remaining evidence is the rules text and the card's other behaviour.

    GROUP                 what stays visible when it is hidden
    mana                  the text, the types, the body
    body                  the text, the cost
    types                 the text, the cost, the body
    keywords              the text (keywords appear verbatim in oracle text —
                          which is exactly why they are worth predicting)
    production            the text
    spans/<slot>          every other slot and every tabular field

## THE KEYWORD CASE IS THE INTERESTING ONE

99% of keywords appear verbatim in the oracle text, so with the text visible,
predicting `kw_flying` is近 a string match. That is not a defect: the point of
the objective is a latent that knows what a card DOES, and "this text implies
flying" is real knowledge cheaply supervised. But it means the keyword block must
sometimes be masked TOGETHER WITH the keyword span slot, or the task degenerates.
`GROUPS` and `COMPANION` encode that.

## RATES

BERT masks 15% of tokens. Here a draw picks one or two GROUPS, which hides
anywhere from 3 to 30 fields — a far higher fraction than 15%, deliberately: a
card has 79 fields against a sentence's hundreds of tokens, and hiding one field
of 79 leaves the answer over-determined.
"""

import numpy as np

from manamap.training import card_fields as CF
from manamap.training import span_encoder as SE

#: Fields grouped by what makes them mutually recoverable. Every tabular field
#: belongs to exactly one group — asserted by a test, because a field that falls
#: out of every group is never a target and never says so.
GROUPS = {
    "mana": ["cmc", "generic_pips", "pips_W", "pips_U", "pips_B", "pips_R",
             "pips_G", "is_x_spell", "has_hybrid", "has_phyrexian",
             "color_identity"],
    "body": ["power", "toughness", "loyalty"],
    "types": (["supertype", "subtypes", "layout"]
              + [f"is_{t.lower()}" for t in CF.TYPE_FLAGS + CF.ROLE_SUBTYPES]
              + list(CF.DERIVED_FIELDS)),
    "keywords": ([f"kw_{k.lower().replace(' ', '_')}" for k in CF.EVERGREEN_KEYWORDS]
                 + ["keywords_other"]),
    "production": [f"produces_{s}" for s in CF.PRODUCIBLE]
                  + ["mana_repeatable", "mana_one_shot"],
    "playability": ["rarity", "edhrec_rank"],
}

#: A span slot that must be hidden alongside a tabular group, or the group is
#: readable straight off the text. The keyword flags are the whole reason this
#: exists: `Flying` is the keyword slot's entire content on most creatures.
COMPANION = {"keywords": ["keyword"]}

#: Span slots are their own maskable groups, one each.
SPAN_GROUPS = {f"span:{slot}": [slot] for slot in SE.SPAN_SLOTS}

#: Never a TARGET, however it is masked. `DERIVED_FIELDS` are an AND of two other
#: visible flags, so scoring a model on them measures nothing — they stay as
#: inputs and are dropped from the loss.
NEVER_A_TARGET = frozenset(CF.DERIVED_FIELDS)


def all_groups():
    return list(GROUPS) + list(SPAN_GROUPS)


def draw(rng, n_groups=(1, 2)):
    """Pick the groups to hide for one training example.

    One or two groups, never zero: an example with nothing hidden contributes no
    imputation gradient and only costs a forward pass.
    """
    groups = all_groups()
    count = int(rng.integers(n_groups[0], n_groups[1] + 1))
    chosen = list(rng.choice(len(groups), size=min(count, len(groups)), replace=False))
    return [groups[i] for i in chosen]


def resolve(chosen):
    """Group names -> `(tabular field names, span slot names)`.

    Applies `COMPANION`, so asking for the keyword block also hides the keyword
    text it is written on.
    """
    fields, slots = [], []
    for name in chosen:
        if name.startswith("span:"):
            slots.extend(SPAN_GROUPS[name])
            continue
        fields.extend(GROUPS[name])
        slots.extend(COMPANION.get(name, []))
    return sorted(set(fields)), sorted(set(slots))


def loss_weights(results, floor=0.05):
    """`{field: weight}` from the recoverability audit.

    A field a linear probe already solves keeps a floor rather than a zero: it
    stays in the loss as a consistency term, contributing almost nothing to the
    gradient while still failing loudly if the model forgets how to add. Zeroing
    it outright would remove the only signal that something has gone wrong.
    """
    weights = {}
    for name, row in results.items():
        if name in NEVER_A_TARGET:
            weights[name] = 0.0
            continue
        weights[name] = max(floor, 1.0 - max(0.0, float(row.get("lift", 0.0))))
    return weights


def apply(card, schema, cache, chosen, oracle_text=None):
    """One masked example: `(tabular vector, span vector, targets)`."""
    fields, slots = resolve(chosen)
    known = {f.name for f in schema}
    fields = [f for f in fields if f in known]
    tabular, tab_offsets = CF.encode(card, schema, masked=fields)
    spans, span_offsets = cache.encode(card, oracle_text, masked=slots)
    return {
        "tabular": tabular, "spans": spans,
        "tab_offsets": tab_offsets, "span_offsets": span_offsets,
        "masked_fields": [f for f in fields if f not in NEVER_A_TARGET],
        "masked_slots": slots,
    }


def coverage(rng, draws=4000):
    """How often each group is hidden. A group that is never drawn is untrained."""
    seen = {g: 0 for g in all_groups()}
    for _ in range(draws):
        for name in draw(rng):
            seen[name] += 1
    return {k: v / draws for k, v in sorted(seen.items())}


def unassigned(schema):
    """Fields in the schema that belong to no group — never masked, never a target."""
    grouped = {name for names in GROUPS.values() for name in names}
    return sorted({f.name for f in schema} - grouped)
