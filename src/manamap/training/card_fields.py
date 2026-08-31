"""A card as TYPED FIELDS with three states, and the encoder that vectorises them.

The architecture this replaces flattened every card into one sentence, so a
random projection of MiniLM beat the trained model on theme and PCA beat it on
everything. Here a number is a number, a set is a set, and a span of text is one
span among several rather than a blob.

## THREE STATES, AND THE DISTINCTION THE WHOLE SCHEMA TURNS ON

    PRESENT   the card has this value
    ABSENT    the card CANNOT have it — a land has no power, a creature no loyalty
    MASKED    the card has it and we have hidden it; impute it

**ABSENT and MASKED are different facts and must never share an encoding.** A
model that cannot tell "this card has no power" from "the power was hidden"
learns to predict absence, which is the cheapest way to be right. Every field
therefore carries explicit `is_present` and `is_masked` indicators alongside its
value, and a masked field's value slot is ZEROED so nothing leaks through it.

## VOCABULARY SIZES, MEASURED

Coverage of occurrences in the corpus, which is what decides where a tail stops
being worth a column:

    subtypes        539 distinct   top 256 covers 97.79%   (128 -> 89.65%)
    keywords        879 distinct   top 128 covers 85.36%   (64 -> 75.94%)
    mana symbols     61 distinct   top  32 covers 99.88%   (16 -> 98.51%)

Everything past the cap lands in an explicit OTHER column rather than vanishing,
so "this card has a subtype I do not model" stays distinguishable from "this
card has no subtypes".

Note the keyword tail is genuinely long — 128 columns still miss 15% of
occurrences. The existing pipeline caps at 50 (~70%), so this is already a
material widening, and the residue is real rather than a rounding error.

## NUMERIC CLIPPING, MEASURED

    cmc          p99 = 8, max = 1,000,000   ONE card above 16 (Gleemax)
    power        p99 = 8, max = 99          one card above 20
    toughness    p99 = 8, max = 99          two cards above 20

A single 1,000,000 destroys any scale computed from the data, so the bounds are
FIXED constants rather than min/max over the corpus — the same rule
`preprocess.py` already follows for `EDHREC_RANK_SCALE`.

## THE NAME IS NOT A FIELD, AND THAT IS MEASURED

`extract.py:99-106` records why it was dropped from the embedding text:
`Sol Ring` matched *Sisay's Ring*, `Llanowar Elves` matched *Llanowar Tribe*,
and recall@10 went 0.187 -> 0.248 when it left.

For an IMPUTATION model the argument is stronger. There are 34,814 distinct
names across 34,890 cards, so the name is a near-unique key: with it visible,
imputing any other field collapses to memorisation. It is the recoverability
audit's lesson at its limit. It may become a target-only field later — imputing
a name from a card is a real task — but it can never be a visible input.
"""

import re

import numpy as np

PRESENT, ABSENT, MASKED = "present", "absent", "masked"

#: Fixed bounds, never computed from the corpus — one 1,000,000 would otherwise
#: set the scale for everything.
CMC_CLIP = (0.0, 16.0)
PT_CLIP = (-1.0, 20.0)
LOYALTY_CLIP = (0.0, 20.0)
EDHREC_CLIP = 32000.0

SUBTYPE_VOCAB_SIZE = 256
KEYWORD_VOCAB_SIZE = 128
MANA_SYMBOL_VOCAB_SIZE = 32

_PIP = re.compile(r"\{([^}]+)\}")
_NUMERIC = re.compile(r"^-?\d+(\.\d+)?$")


def _missing(value):
    return value is None or (isinstance(value, float) and value != value)


def _clean(value):
    """`""` for anything absent. `str(float('nan'))` is `"nan"`, which shipped once."""
    if _missing(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


class Field:
    """One typed slot. Subclasses define `width` and `_encode_value`."""

    kind = "abstract"

    def __init__(self, name):
        self.name = name

    def read(self, card):
        """`(state, value)` — never raises on a malformed card."""
        raise NotImplementedError

    def encode(self, card, masked=False):
        """Fixed-width vector: [value…, is_present, is_masked].

        A MASKED field's value slots are ZEROED. Leaving the true value in place
        behind a flag is the bug that makes an imputation task trivial, and it
        would not show up as anything but suspiciously good numbers.
        """
        state, value = self.read(card)
        body = np.zeros(self.width, dtype=np.float32)
        if state == PRESENT and not masked:
            body = self._encode_value(value).astype(np.float32)
        present = 0.0 if state == ABSENT else 1.0
        return np.concatenate([body, [present, 1.0 if masked else 0.0]])

    @property
    def total_width(self):
        return self.width + 2

    def _encode_value(self, value):
        raise NotImplementedError


class Numeric(Field):
    """A number, plus a flag for the values Magic writes as `*` or `1d4+1`.

    253 power values and 191 toughness values are non-numeric (`*`, `*²`, `1+*`),
    and loyalty carries `X` and `1d4+1`. Casting throws them away; a `variable`
    flag keeps "this card's power is defined by something else" as a fact.
    """

    kind = "numeric"
    width = 2          # [scaled value, is_variable]

    def __init__(self, name, clip, column=None):
        super().__init__(name)
        self.clip = clip
        self.column = column or name

    def read(self, card):
        raw = card.get(self.column)
        if _missing(raw) or _clean(raw) == "":
            return ABSENT, None
        text = _clean(raw)
        if _NUMERIC.match(text):
            return PRESENT, float(text)
        return PRESENT, text                      # variable: '*', '1d4+1', 'X'

    def _encode_value(self, value):
        if isinstance(value, str):
            return np.array([0.0, 1.0])           # variable, no usable magnitude
        low, high = self.clip
        scaled = (min(max(float(value), low), high) - low) / (high - low)
        return np.array([scaled, 0.0])


class Categorical(Field):
    """One-of-N with an explicit OTHER column."""

    kind = "categorical"

    def __init__(self, name, vocab, column=None):
        super().__init__(name)
        self.vocab = list(vocab)
        self.index = {v: i for i, v in enumerate(self.vocab)}
        self.column = column or name
        self.width = len(self.vocab) + 1          # + OTHER

    def read(self, card):
        text = _clean(card.get(self.column))
        return (PRESENT, text) if text else (ABSENT, None)

    def _encode_value(self, value):
        out = np.zeros(self.width)
        out[self.index.get(value, self.width - 1)] = 1.0
        return out


class SetOf(Field):
    """Multi-hot over a capped vocabulary, with OTHER for the tail.

    An empty set is ABSENT, not a zero vector: "this card has no subtypes" and
    "this card's subtypes were hidden" must not encode identically, which is the
    same rule the three states exist for.
    """

    kind = "set"

    def __init__(self, name, vocab, extract):
        super().__init__(name)
        self.vocab = list(vocab)
        self.index = {v: i for i, v in enumerate(self.vocab)}
        self.extract = extract
        self.width = len(self.vocab) + 1

    def read(self, card):
        try:
            items = [i for i in self.extract(card) if i]
        except Exception:                          # noqa: BLE001 - malformed card
            items = []
        return (PRESENT, items) if items else (ABSENT, None)

    def _encode_value(self, value):
        out = np.zeros(self.width)
        for item in value:
            out[self.index.get(item, self.width - 1)] = 1.0
        return out


# ── the extractors, kept out of the classes so they are testable alone ──


def color_identity_of(card):
    return [c for c in _clean(card.get("color_identity")) if c in "WUBRG"]


def subtypes_of(card):
    line = _clean(card.get("type_line"))
    if "—" not in line:
        return []
    return line.split("—", 1)[1].split("//")[0].split()


def card_types_of(card):
    line = _clean(card.get("type_line"))
    return line.split("—", 1)[0].replace("//", " ").split()


def keywords_of(card):
    return [k.strip() for k in _clean(card.get("keywords")).split(",") if k.strip()]


def mana_symbols_of(card):
    return _PIP.findall(_clean(card.get("mana_cost")))


def build_schema(vocabs):
    """The field registry. `vocabs` comes from `vocabularies()` over the corpus."""
    return [
        Numeric("cmc", CMC_CLIP),
        Numeric("power", PT_CLIP),
        Numeric("toughness", PT_CLIP),
        Numeric("loyalty", LOYALTY_CLIP),
        Numeric("edhrec_rank", (0.0, EDHREC_CLIP)),
        Categorical("supertype", vocabs["supertype"]),
        Categorical("rarity", vocabs["rarity"]),
        Categorical("layout", vocabs["layout"]),
        SetOf("color_identity", list("WUBRG"), color_identity_of),
        SetOf("card_types", vocabs["card_types"], card_types_of),
        SetOf("subtypes", vocabs["subtypes"], subtypes_of),
        SetOf("keywords", vocabs["keywords"], keywords_of),
        SetOf("mana_symbols", vocabs["mana_symbols"], mana_symbols_of),
    ]


def vocabularies(cards, subtypes=SUBTYPE_VOCAB_SIZE, keywords=KEYWORD_VOCAB_SIZE,
                 mana=MANA_SYMBOL_VOCAB_SIZE):
    """Most-frequent-first vocabularies, built ONCE and saved with the model.

    A head trained against one vocabulary and scored against another is
    meaningless in a way that looks like a bad hyperparameter.
    """
    import collections

    counts = collections.defaultdict(collections.Counter)
    for card in cards:
        counts["supertype"][_clean(card.get("supertype"))] += 1
        counts["rarity"][_clean(card.get("rarity"))] += 1
        counts["layout"][_clean(card.get("layout"))] += 1
        counts["card_types"].update(card_types_of(card))
        counts["subtypes"].update(subtypes_of(card))
        counts["keywords"].update(keywords_of(card))
        counts["mana_symbols"].update(mana_symbols_of(card))
    caps = {"subtypes": subtypes, "keywords": keywords, "mana_symbols": mana}
    return {name: [v for v, _n in counter.most_common(caps.get(name)) if v]
            for name, counter in counts.items()}


def encode(card, schema, masked=()):
    """`(vector, offsets)` — the card as one row, and where each field sits."""
    masked = {masked} if isinstance(masked, str) else set(masked)
    unknown = masked - {f.name for f in schema}
    if unknown:
        raise ValueError(f"not fields in this schema: {sorted(unknown)}")
    parts, offsets, at = [], {}, 0
    for field in schema:
        vector = field.encode(card, masked=field.name in masked)
        offsets[field.name] = (at, at + len(vector))
        at += len(vector)
        parts.append(vector)
    return np.concatenate(parts).astype(np.float32), offsets


def describe(card, schema):
    """`{field: (state, value)}` — the struct, for inspection and for tests."""
    return {field.name: field.read(card) for field in schema}
