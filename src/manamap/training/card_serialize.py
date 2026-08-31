"""A card as a masked token sequence: the input to the imputation model.

The contrastive model this replaces needs LABELS — which pairs are alike — and
mines them from the repo's own regexes (53 `ROLE_PATTERNS`, 33 `MECHANICAL_TAGS`
in `train_ability._positive`). It is therefore largely learning to reproduce a
vocabulary we authored, which is the bootstrapping this work exists to remove.

Masked imputation needs no labels at all: **the label is the input.** Hide the
mana cost, the target is the mana cost you hid. That is the whole argument for
the change, and everything below exists to keep it honest.

## THE MASKING UNIT IS A MODALITY, NOT A FIELD

Measured 2026-08-31, held-out accuracy of a single-field lookup table built on
the train split (split by TEXT HASH — 1,031 duplicate-text families, 0 of them
crossing the split):

    target            best single-field source   held-out   verdict
    has_pt                         type_line        0.995   TRIVIAL
    supertype                      type_line        0.993   TRIVIAL
    layout                         type_line        0.984   TRIVIAL
    cmc                            mana_cost        0.982   TRIVIAL
    color_identity                 mana_cost        0.931   weak
    subtypes                       type_line        0.917   weak
    rarity                         type_line        0.445   keep

The pattern is not "some fields are easy". **`supertype`, `subtypes` and
`has_pt` are PARSED FROM the type line; `cmc` and `color_identity` are PARSED
FROM the mana cost.** Asking the model to predict one from its own source is
string parsing wearing a prediction task's clothes, and it would dominate the
loss while teaching nothing — the same failure `train_ability.py:1-33` records
for the previous model, which learned to discard the only signal that worked.

So blocks are masked WHOLE. Hide the entire type block and it must come from the
oracle text; hide the text and it must come from type, cost and stats. No single
surface is present to copy from, which is the only version of this objective
that can encode function rather than phrasing.

## TAGS AND ROLES MAY BE AN INPUT, NEVER A TARGET

`MECHANICAL_TAGS` and `ROLE_PATTERNS` are regexes over the oracle text. Training
the model to predict them re-imports the exact bootstrapping this architecture
removes — the model would be learning our regexes again, one layer down. They
are deliberately absent from `BLOCKS`, and `tests/test_card_serialize.py` says
so, because it is the kind of thing that gets added back for looking useful.

## THE ABILITY-LINE BOUNDARY IS RESTORED

`extract.py:157` flattens newlines when it builds `embedding_text`, so
"{T}: Add {G}." and a second unrelated ability become one run-on string. That is
fine for a pooled sentence vector and wrong for a model that should learn what
an ability IS. `[LINE]` puts the boundary back. The CSV column is untouched —
it feeds the frozen baseline this model has to beat, and moving it would confound
the comparison.
"""

import re

#: Field sentinels. Ordered so the cheap, highly-structured blocks come first and
#: a truncated sequence loses oracle text rather than losing the type line.
BLOCKS = ("type", "cost", "stats", "text")

MASK = "[MASK]"
LINE = "[LINE]"

_SENTINEL = {"type": "[TYPE]", "cost": "[COST]", "stats": "[PT]",
             "text": "[TEXT]"}


def _clean(value):
    """Absent is EMPTY, never the string "nan".

    `str(float("nan"))` is `"nan"`, and pandas hands every missing cell through
    as a float NaN — so the first cut of this module emitted `[PT] nan/nan` for
    a Forest and `[COST] nan` for every land. The model would have learned "nan"
    as a token meaning absent, which is a vocabulary item standing in for a fact
    the sentinel already carries.
    """
    if value is None:
        return ""
    if isinstance(value, float) and value != value:      # NaN
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def blocks_for(card):
    """`{block: text}` for one card row. Empty blocks are kept, not dropped.

    An absent block is information — 409 cards have no oracle text at all, and a
    model that never sees an empty `[TEXT]` cannot represent a vanilla creature.
    """
    type_line = _clean(card.get("type_line"))
    power, toughness = _clean(card.get("power")), _clean(card.get("toughness"))
    stats = f"{power}/{toughness}" if power or toughness else ""
    text = _clean(card.get("oracle_text"))
    # THE BOUNDARY extract.py FLATTENS. Scryfall separates abilities with \n.
    text = re.sub(r"\s*\n+\s*", f" {LINE} ", text)
    return {
        "type": type_line,
        "cost": _clean(card.get("mana_cost")),
        "stats": stats,
        "text": text,
    }


def serialize(card, mask=()):
    """The card as one string, with `mask` blocks replaced by `[MASK]`.

    The sentinel STAYS when a block is masked — the model must know that a type
    line was hidden rather than that the card has none. Those are different
    facts and collapsing them is how a masked model learns to predict "empty".
    """
    mask = {mask} if isinstance(mask, str) else set(mask)
    unknown = mask - set(BLOCKS)
    if unknown:
        raise ValueError(f"not maskable blocks: {sorted(unknown)}. "
                         f"Tags and roles are deliberately absent — see the "
                         f"module docstring.")
    parts = blocks_for(card)
    out = []
    for block in BLOCKS:
        body = MASK if block in mask else parts[block]
        out.append(f"{_SENTINEL[block]} {body}".rstrip())
    return " ".join(out)


def targets_for(card, mask):
    """What the model must reconstruct: `{block: the hidden text}`."""
    parts = blocks_for(card)
    mask = {mask} if isinstance(mask, str) else set(mask)
    return {block: parts[block] for block in mask}


#: KEYWORDS ARE NOT A BLOCK, and were one until measured. Scryfall's `keywords`
#: column is a canonical list of the abilities ALREADY PRINTED in the oracle
#: text: 99.0% of a card's keywords appear verbatim in its own text, and 98.1%
#: of cards have every one of them there. Gishath carried `[KW] Vigilance Haste
#: Trample` beside `[TEXT] Vigilance, trample, haste`.
#:
#: That makes the block worthless in both directions — masking it is a copy from
#: `[TEXT]`, and leaving it visible LEAKS the text when the text is what is
#: hidden. Same tautology the recoverability audit caught for supertype and cmc,
#: found one layer down.
#:
#: How often each remaining block is hidden. Weighted toward `text` and `type`
#: because the audit shows those are the two not recoverable from a single other
#: field, and toward masking one block at a time because a median card is 37
#: subword tokens and hiding several leaves nothing to reason from.
MASK_WEIGHTS = {"text": 0.40, "type": 0.28, "cost": 0.22, "stats": 0.10}


def sample_mask(rng, multi=0.15):
    """Choose which block(s) to hide for one training example.

    `multi` is the chance of hiding a SECOND block, which makes the task harder
    without making it impossible; three at once leaves a median card with almost
    nothing to condition on.
    """
    blocks = list(MASK_WEIGHTS)
    weights = [MASK_WEIGHTS[b] for b in blocks]
    first = blocks[int(rng.choice(len(blocks), p=weights))]
    if rng.random() >= multi:
        return (first,)
    rest = [b for b in blocks if b != first]
    second = rest[int(rng.integers(len(rest)))]
    return tuple(sorted((first, second), key=BLOCKS.index))
