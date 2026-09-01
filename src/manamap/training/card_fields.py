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


class DerivedNumeric(Numeric):
    """A number COMPUTED from the card rather than read from a column.

    The mana fields are all of this kind, and the first cut got it wrong: they
    were built as plain `Numeric`, which reads `card["pips_W"]` — a column that
    does not exist — so every one of them read ABSENT on every card. Gishath's
    `{5}{R}{G}{W}` produced six absent mana fields.

    Caught by `test_every_field_is_populated_somewhere`, which exists precisely
    because a field that is never present is a column of zeros wearing a name.
    """

    def __init__(self, name, clip, extract, applies=None):
        super().__init__(name, clip)
        self.extract = extract
        self.applies = applies

    def read(self, card):
        if self.applies is not None and not self.applies(card):
            return ABSENT, None
        try:
            return PRESENT, float(self.extract(card))
        except Exception:                          # noqa: BLE001 - malformed card
            return ABSENT, None


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


def _as_list(value):
    """A cell that means "several things" -> a list, whatever shape it arrived in.

    `cards.csv` comma-joins these columns (`"Vigilance, Haste, Trample"`) but
    Scryfall's JSON — which this encoder should also accept, since it is the
    upstream source — gives real lists. `keywords_of` handled only the first, so
    a genuine `["Flying", "Trample"]` was stringified and comma-split into the
    tokens `"['Flying'"` and `"'Trample']"`.

    That is the `nan`-as-a-literal-token bug wearing a different hat, and it was
    live in the TEST FIXTURE — `_card()` passes a real list, so every keyword
    assertion in this suite was reading garbage that a set field accepted without
    complaint. Nothing failed until a keyword became its own field with a value
    to check.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in _clean(value).split(",") if part.strip()]


def color_identity_of(card):
    # Correct for both shapes ON PURPOSE. The character filter made the list case
    # work by luck, which is not the same as working.
    return [c for c in "".join(_as_list(card.get("color_identity"))) if c in "WUBRG"]


def subtypes_of(card):
    line = _clean(card.get("type_line"))
    if "—" not in line:
        return []
    return line.split("—", 1)[1].split("//")[0].split()


def card_types_of(card):
    line = _clean(card.get("type_line"))
    return line.split("—", 1)[0].replace("//", " ").split()


def keywords_of(card):
    return _as_list(card.get("keywords"))


def mana_symbols_of(card):
    return _PIP.findall(_clean(card.get("mana_cost")))


class Binary(Field):
    """One yes/no fact with its own mask.

    THE GRANULARITY THAT MAKES MASKING USEFUL. `keywords` as a single 131-wide
    set hides flying AND trample AND lifelink together, so the only question it
    can ask is "what abilities does this card have" — never "does this card have
    FLYING". The second is a far better imputation task and the one a player
    would recognise, and it needs the keyword to be its own field.

    `present` is always 1: a card definitively does or does not have flying, so
    there is no ABSENT state here — unlike power, which a land genuinely lacks.
    """

    kind = "binary"
    width = 1

    def __init__(self, name, predicate, applies=None):
        super().__init__(name)
        self.predicate = predicate
        self.applies = applies

    def read(self, card):
        # `applies` separates "no" from "the question does not apply". A card
        # without a mana cost is not a non-X-spell; there is no cost to look at.
        if self.applies is not None and not self.applies(card):
            return ABSENT, None
        try:
            return PRESENT, bool(self.predicate(card))
        except Exception:                          # noqa: BLE001 - malformed card
            return PRESENT, False

    def _encode_value(self, value):
        return np.array([1.0 if value else 0.0])


#: Keywords that get their OWN maskable field, by corpus frequency. The tail
#: (879 distinct, this covers the head) stays in a `keywords_other` set — a
#: keyword appearing on 40 cards does not earn a column of its own, and folding
#: it into OTHER keeps "has a keyword I do not model" representable.
EVERGREEN_KEYWORDS = (
    "Flying", "Trample", "Vigilance", "Haste", "Flash", "Menace", "First strike",
    "Double strike", "Lifelink", "Deathtouch", "Reach", "Defender", "Hexproof",
    "Indestructible", "Ward", "Protection", "Equip", "Enchant", "Cycling",
    "Kicker", "Flashback", "Scry", "Mill", "Surveil", "Crew", "Regenerate",
)

#: Mana symbols that are a QUESTION rather than a count.
_X_RE = re.compile(r"\{X\}")
_HYBRID_RE = re.compile(r"\{[^}]*/[^}]*\}")
_PHYREXIAN_RE = re.compile(r"\{[^}]*/P\}", re.IGNORECASE)


def _has_cost(card):
    """Does this card have a mana cost AT ALL?

    THE ABSENT-IS-NOT-ZERO LINE, and the first cut was on the wrong side of it.
    Command Tower has no mana cost; Ornithopter costs `{0}`. Both read
    `generic_pips = 0.0, PRESENT` until this guard existed, which tells the model
    a land costs zero generic mana — a MEASUREMENT — when the truth is that the
    question does not apply. 1,760 cards have no mana cost.

    Caught by `test_every_field_is_populated_somewhere_and_absent_somewhere`,
    which asserts every field is absent SOMEWHERE. `generic_pips` never was.
    """
    return bool(_clean(card.get("mana_cost")).strip())


def _pip_count(card, colour):
    """How many pips of one colour. Hybrid counts toward BOTH halves.

    `manabase.count_pips` splits a hybrid half-and-half because it is sizing a
    mana base and half a source is the honest answer there. Here the question is
    "can this card want blue", and a hybrid card genuinely can — so it counts
    whole, and the two functions answer two different questions on purpose.
    """
    total = 0
    for symbol in _PIP.findall(_clean(card.get("mana_cost"))):
        if colour in symbol.upper().split("/"):
            total += 1
    return total


def _generic_count(card):
    for symbol in _PIP.findall(_clean(card.get("mana_cost"))):
        if symbol.isdigit():
            return int(symbol)
    return 0


#: The mana a card can MAKE, which is a different question from what it costs and
#: is the one `cards.csv` cannot answer at all. Command Tower's row is `Land`, no
#: mana cost, no power, no subtypes — before these fields the second-most-played
#: card in Commander encoded as almost entirely ABSENT while a vanilla French
#: creature encoded richly. Source is Scryfall's `produced_mana`, merged by
#: `card_source.enrich`; deriving it from oracle text would mean re-deriving what
#: an authoritative field already states.
#:
#: `C` is colourless, and it earns a flag: Sol Ring making {C}{C} is a fact about
#: the card, not an absence of colour. (One card, Unfinity's `Sole Performer`,
#: produces `{T}` tickets and correctly matches none of these.)
PRODUCIBLE = ("W", "U", "B", "R", "G", "C")


def _enriched(card):
    """Was this record put through `card_source.enrich`?

    The three-state contract doing real work: a record that never saw the dump
    reports ABSENT — nobody measured this — rather than False, which would claim
    every card in the corpus makes no mana and look entirely plausible doing it.
    """
    return "produced_mana" in card


def produces(card, symbol):
    return symbol in {str(s).upper() for s in (card.get("produced_mana") or [])}


#: THE CARD TYPES, one flag each, replacing a 38-wide set.
#:
#: A set field could say "this card is an Artifact and a Creature" but could not
#: be asked "is this a creature" on its own — masking hid every type at once.
#: 38 distinct values is exactly the size that should have been flags from the
#: start; `subtypes` stays a set because there are 539 of them.
#:
#: `Legendary` is here and matters more than its 4,686 suggests: in Commander it
#: is the difference between a card that can lead a deck and one that cannot.
TYPE_FLAGS = ("Creature", "Artifact", "Enchantment", "Instant", "Sorcery", "Land",
              "Planeswalker", "Battle", "Kindred", "Legendary", "Snow", "Basic")

#: Subtypes that change HOW a card is played rather than what it is thematically.
#: An Aura attaches and dies with its host; Equipment survives and re-attaches; a
#: Vehicle needs crew; a Saga advances and sacrifices itself. Buried among 539
#: creature types they were unmaskable and, for a model, nearly unfindable.
#: These are REMOVED from the `subtypes` set — otherwise masking `is_aura` would
#: leave the answer sitting in plain sight, which is the leak the keyword split
#: already had to fix once.
ROLE_SUBTYPES = ("Aura", "Equipment", "Vehicle", "Saga")

#: Fields derived ENTIRELY from other visible fields. They are legitimate inputs —
#: an explicit `is_artifact_creature` saves a linear probe from having to learn a
#: conjunction — but they are NOT honest imputation targets: masking one while
#: `is_artifact` and `is_creature` stay visible hides nothing at all. The masking
#: harness excludes them, and a test asserts the exclusion.
DERIVED_FIELDS = ("is_artifact_creature", "is_enchantment_creature", "is_artifact_land")


#: HOW MUCH, not just which. `produces_C` fires for Sol Ring and for Lotus Petal
#: alike; the quantity is the whole difference between them and it lived only in
#: the ability text. Sol Ring and Arcane Signet were byte-identical on every mana
#: field until these two.
#:
#: TWO NUMBERS, NOT ONE, and mixing them would be dishonest: Sol Ring's 2 arrives
#: every turn forever, Dark Ritual's 3 arrives once. Same split, same reason, as
#: `mana_analysis.life_cost`'s `{recurring, one_time}`.
#:
#: `mana_repeatable` REUSES `goldfish.produced_mana` rather than growing a fourth
#: add-clause matcher. That function has already paid for three lessons this
#: schema would otherwise re-learn: a consuming cost is not a rate (Jeweled Lotus
#: read as three mana every turn forever), alternatives are a CHOICE so
#: `Add {R}, {G}, or {W}` is one and not three, and the word forms count.
#:
#: WHAT THE SWEEP SAYS THEY GET WRONG, measured over all 34,890 cards:
#:
#:   repeatable  1,975 nonzero (max 5). **145 cards read a granted ability as
#:               their own** — and the obvious fix is worse. Citanul Hierophants
#:               grants `{T}: Add {G}` to "creatures you control" and IS a
#:               creature, so its 1 is correct; Gemhide Sliver, Enduring Vitality
#:               and Inga and Esika likewise, and Dryad Arbor's sits in reminder
#:               text about itself. Stripping quoted text would break all five.
#:               The genuinely wrong ones are the card that cannot be a member of
#:               the class it grants to: **Cryptolith Rite** (an enchantment
#:               granting to creatures), **Thranduil** ("OTHER Elves"),
#:               **Leyline Immersion** (an Aura, reads 5) and **Liliana of the
#:               Dark Realms** (an emblem, reads 4). 8 of the 145 are sleeved
#:               across five decks, five of them in kinnan — so this is a live
#:               overcount in the GOLDFISH, not only here.
#:   one_shot    735 nonzero (max 10: Ramos and Meeting of the Five, both real).
#:               Reads reminder text for a GRANTED keyword — Sozin's Comet's
#:               firebending 5 — as the card's own burst.
#:
#: Both are GROSS: `{5}, {T}: Add {W}{U}{B}{R}{G}` counts 5, not net zero.
MANA_CLIP = (0.0, 10.0)

#: An `Add` clause anywhere, for mana the repeatable reading does not claim —
#: rituals, sacrifice-for-mana, and one-shot triggers. Black Lotus, Dark Ritual
#: and Jeweled Lotus all read 0 repeatable, and a similarity space in which Black
#: Lotus makes no mana is not describing Magic.
_ADD_CLAUSE = re.compile(r"\bAdd ([^.\n]+)", re.IGNORECASE)
_MANA_RUN = re.compile(r"(?:\{[WUBRGC0-9]\})+")
_MANA_SYMBOL = re.compile(r"\{[WUBRGC0-9]\}")
_MANA_WORD = re.compile(r"\s*(one|two|three|four|five|six|seven|eight)\b", re.IGNORECASE)
_MANA_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4,
               "five": 5, "six": 6, "seven": 7, "eight": 8}


def _largest_add(text):
    """The most mana any single `Add` clause yields.

    LARGEST RUN, never the sum — `Add {R}, {G}, or {W}` is a choice of one and
    counting symbols gives three. This is the same rule `goldfish.produced_mana`
    applies, for the same reason, and getting it wrong overcounts every
    dual-choice rock in the corpus.
    """
    best = 0
    for match in _ADD_CLAUSE.finditer(_clean(text)):
        body = match.group(1)
        runs = [len(_MANA_SYMBOL.findall(run)) for run in _MANA_RUN.findall(body)]
        if runs:
            best = max(best, max(runs))
            continue
        word = _MANA_WORD.match(body)
        if word:
            best = max(best, _MANA_WORDS[word.group(1).lower()])
    return best


def mana_repeatable(card):
    """Mana a persistent producer yields per turn. Delegates — ONE PREDICATE, ONE HOME."""
    from manamap.pilot.goldfish import produced_mana

    # `type_line` decides whether a QUOTED ability is this card's own — without
    # it every granted ability reads as foreign, which is the conservative
    # direction but wrong for the 15 cards that grant to a class they belong to.
    return produced_mana(_clean(card.get("oracle_text")), _clean(card.get("type_line")))


def mana_one_shot(card):
    """Mana from a clause the repeatable reading does not claim (0 if none)."""
    return max(0, _largest_add(card.get("oracle_text")) - mana_repeatable(card))


#: Fields that are never ABSENT anywhere in the corpus — a MEASUREMENT over
#: ENRICHED records, checked in both directions by
#: `test_every_field_is_populated_somewhere`. Three reasons a field lands here,
#: and they are different:
#:
#:   * **By construction.** Every `kw_*` binary: a card definitively has flying or
#:     does not, so there is no third state to represent. (The mana binaries are
#:     NOT here — they carry `_has_cost`, because a land is not a non-X-spell.)
#:   * **By corpus.** `cmc`, `supertype`, `rarity`, `layout`, `card_types` could
#:     in principle be missing and simply never are across all 34,890 rows.
#:   * **By enrichment.** Every `produces_*` is present precisely BECAUSE the
#:     records went through `card_source.enrich`. Hand the schema a bare CSV row
#:     and all six read ABSENT — which is the point of `_enriched`, and the
#:     reason this set is a claim about enriched records rather than about cards.
#:
#: Declaring it rather than exempting it is the point: a hand-kept exemption list
#: only ever grows, and silently. This set is asserted in BOTH directions, so a
#: field that starts or stops being always-present fails the suite either way.
ALWAYS_PRESENT = frozenset({
    "cmc", "supertype", "rarity", "layout",
} | {f"kw_{k.lower().replace(' ', '_')}" for k in EVERGREEN_KEYWORDS}
  | {f"produces_{s}" for s in PRODUCIBLE}
  | {"mana_repeatable", "mana_one_shot"}
  | {f"is_{t.lower()}" for t in TYPE_FLAGS + ROLE_SUBTYPES}
  | set(DERIVED_FIELDS))


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
        *[Binary(f"is_{t.lower()}",
                 (lambda name: lambda c: name in card_types_of(c))(t))
          for t in TYPE_FLAGS],
        *[Binary(f"is_{t.lower()}",
                 (lambda name: lambda c: name in subtypes_of(c))(t))
          for t in ROLE_SUBTYPES],
        # DERIVED — see DERIVED_FIELDS. Explicit because the combination is a
        # different object from either part: an artifact creature answers to both
        # removal types, an artifact land is a land that Shatter kills.
        Binary("is_artifact_creature",
               lambda c: {"Artifact", "Creature"} <= set(card_types_of(c))),
        Binary("is_enchantment_creature",
               lambda c: {"Enchantment", "Creature"} <= set(card_types_of(c))),
        Binary("is_artifact_land",
               lambda c: {"Artifact", "Land"} <= set(card_types_of(c))),
        SetOf("subtypes",
              [t for t in vocabs["subtypes"] if t not in ROLE_SUBTYPES],
              lambda c: [t for t in subtypes_of(c) if t not in ROLE_SUBTYPES]),
        # MANA, DECOMPOSED. One `mana_symbols` set could not be asked "how many
        # blue pips" or "is this an X spell" — both of which are facts a player
        # reads off a card at a glance and a model should be able to impute.
        DerivedNumeric("generic_pips", (0.0, 12.0), _generic_count, _has_cost),
        *[DerivedNumeric(f"pips_{c}", (0.0, 5.0),
                         (lambda col: lambda card: _pip_count(card, col))(c),
                         _has_cost)
          for c in "WUBRG"],
        Binary("is_x_spell",
               lambda c: bool(_X_RE.search(_clean(c.get("mana_cost")))), _has_cost),
        Binary("has_hybrid",
               lambda c: bool(_HYBRID_RE.search(_clean(c.get("mana_cost")))), _has_cost),
        Binary("has_phyrexian",
               lambda c: bool(_PHYREXIAN_RE.search(_clean(c.get("mana_cost")))), _has_cost),
        # HOW MUCH IT MAKES. Gated on enrichment like `produces_*`, and for a
        # second reason: `_TAP_ADD_RE` refuses to cross a newline so a `{T}` in
        # one ability cannot bind to an `: Add` in another — a guarantee the
        # FLATTENED CSV text destroys.
        DerivedNumeric("mana_repeatable", MANA_CLIP, mana_repeatable, _enriched),
        DerivedNumeric("mana_one_shot", MANA_CLIP, mana_one_shot, _enriched),
        # WHAT IT TAPS FOR. Each colour on its own, so "can this make blue" is
        # askable — and maskable — separately from "can this make green".
        *[Binary(f"produces_{sym}",
                 (lambda s: lambda c: produces(c, s))(sym), _enriched)
          for sym in PRODUCIBLE],
        # KEYWORDS, ONE FIELD EACH, so "does this have flying" is a question the
        # model can be asked. The tail stays in a set.
        *[Binary(f"kw_{k.lower().replace(' ', '_')}",
                 (lambda name: lambda c: name in keywords_of(c))(k))
          for k in EVERGREEN_KEYWORDS],
        SetOf("keywords_other",
              [k for k in vocabs["keywords"] if k not in EVERGREEN_KEYWORDS],
              lambda c: [k for k in keywords_of(c) if k not in EVERGREEN_KEYWORDS]),
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
