"""What a legal deck IS, as a parameter rather than an assumption.

PRD §13: the constraints should be **parameters, not assumptions**, because
retrofitting format-awareness after the fact is expensive. This module is that
parameter — Commander plus the four 60-card constructed formats.

The parameters that vary, and all five differ on at least one: deck size and
whether it is exact, the singleton rule, whether a commander is required,
colour-identity enforcement, and the legal pool.

BEFORE THIS, four places independently decided how big a Commander deck is:

    config.DECK_SIZE = 100              read by validate_build
    check_in.DECK_SIZE = 100            its OWN constant, shadowing config's
    manabase.DECK_SIZE_AFTER_COMMANDER  = 99, the same fact minus one
    validate_deck.py:15                 `if total != 100`, a bare literal

None of them was wrong. That is what makes it the interesting kind of
duplication: nothing is broken, nothing fails, and the cost only arrives when
one of the four has to change and three of them do not. `check_in` shadowing
`config` is the shape that would have bitten first — a name that resolves,
locally, to something a reader would swear came from the shared constant.

**The library size is DERIVED, not stored.** `DECK_SIZE_AFTER_COMMANDER = 99`
was a second literal for `100 - 1`, and a format where that arithmetic differs
(a commander that starts on the battlefield, a format with none) would have had
to remember to change both. `FormatSpec.library_size` is a property.

WHAT IS DELIBERATELY NOT HERE:

- **Pendragon.** The PRD lists it and flags its own description as unverified
  ("name and rules unverified, confirm before scoping"). A format spec that
  encodes a guess is worse than one that omits it, because the guess is
  invisible once it is in a table.
- **A pool-filtering layer.** There is none and there does not need to be:
  `extract` already writes eight `legal_<format>` columns into `cards.csv`
  straight from Scryfall, so a format's pool is a column lookup rather than a
  rule to reimplement. Sizes today — Standard 4,887, Pauper 10,793, Pioneer
  14,817, Modern 22,450, Commander 31,830.
- **Deck-construction ratios.** `DECK_ROLE_BUDGET` and the curve targets stay in
  `config.py`. They are tuning, not legality — a 60-card Modern deck is legal
  with any curve, and mixing "what is allowed" with "what is good" is how a
  format table turns into a strategy opinion.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FormatSpec:
    """The rules a deck must satisfy to be legal, and nothing about whether it
    is any good."""

    name: str
    #: Total cards including any commander — the number a pilot counts.
    deck_size: int
    #: Whether `deck_size` is EXACT or a MINIMUM, which is a real rules
    #: distinction and not a nicety. Commander is exactly 100; constructed says
    #: "at least sixty cards", and a 63-card Modern deck is legal. Enforcing an
    #: exact 60 would reject legal decks while looking rigorous.
    exact_size: bool
    #: At most one copy of any non-basic card.
    singleton: bool
    #: How many commanders the format requires. 0 for the 60-card formats.
    commanders: int
    #: Whether the deck's cards must stay inside the commander's colour identity.
    colour_identity: bool
    #: Whether the singleton rule exempts basic lands. It always does where
    #: singleton applies at all, but a format that banned that would be a
    #: silent-wrong-answer bug rather than an obvious one.
    basics_exempt: bool
    #: The key in Scryfall's `legalities` map, and the suffix of the
    #: `legal_<key>` column `extract` already writes into `cards.csv`.
    legality_key: str
    #: Whether `build_deck` can actually BUILD this format, as opposed to
    #: validating a list somebody else built.
    #:
    #: Only Commander, and the gap is real rather than a missing flag. The
    #: builder is anchored on a commander at every step: colour identity comes
    #: from it and gates the whole candidate pool, the similarity score is
    #: seeded from its name, its mechanical tags drive synergy, the bracket
    #: engine reads it, and `manabase` sizes against a 99-card library. A
    #: constructed deck has no such anchor — you build around an archetype and
    #: a colour pair — so this is a different build strategy, not a parameter.
    #:
    #: It exists because the UI offered a five-format picker in front of a
    #: builder that only builds one, and "I tried to build a Standard deck and
    #: nothing happened" is what that costs. A format the tool cannot build is
    #: now something the tool SAYS it cannot build.
    buildable: bool = False

    @property
    def library_size(self):
        """Cards that start in the library — what a hypergeometric draws from.

        DERIVED, because `100 - 1` written down twice is two things to change
        and one of them will be missed. A format whose commander starts in the
        library rather than the command zone would override this property
        instead of editing a constant somewhere else.
        """
        return self.deck_size - self.commanders

    @property
    def max_copies(self):
        """Copies of one non-basic card. Singleton is a 1; everything else is 4."""
        return 1 if self.singleton else 4

    @property
    def legality_column(self):
        """The `cards.csv` column carrying this format's legality."""
        return f"legal_{self.legality_key}"

    def size_error(self, total):
        """Why `total` cards is illegal, or None."""
        if self.exact_size:
            return (None if total == self.deck_size
                    else f"Deck has {total} cards, expected exactly {self.deck_size}")
        return (None if total >= self.deck_size
                else f"Deck has {total} cards, expected at least {self.deck_size}")


COMMANDER = FormatSpec(
    name="Commander", deck_size=100, exact_size=True, singleton=True, commanders=1,
    colour_identity=True, basics_exempt=True, legality_key="commander",
    buildable=True,
)


def _constructed(name, key):
    """A 60-card constructed format. Everything but the legal pool is shared.

    `deck_size=60` with `exact_size=False`, because the rule is "at least
    sixty" — a 63-card Modern deck is legal, and enforcing an exact 60 would
    reject legal decks while looking rigorous. No commander, so no colour
    identity: a constructed deck may play any colours it can cast.

    PAUPER IS NOT FILTERED BY RARITY, and that is measured rather than assumed.
    The PRD describes it as "commons only" (§13) and Scryfall's own
    `legal_pauper` disagrees with that reading for **373 cards** — a card
    printed at common ANYWHERE is pauper-legal even where this printing is not.
    Consulting the legality column is both simpler and correct; a rarity filter
    would look stricter and be wrong 373 times.
    """
    return FormatSpec(name=name, deck_size=60, exact_size=False, singleton=False,
                      commanders=0, colour_identity=False, basics_exempt=True,
                      legality_key=key)


STANDARD = _constructed("Standard", "standard")
MODERN = _constructed("Modern", "modern")
PIONEER = _constructed("Pioneer", "pioneer")
PAUPER = _constructed("Pauper", "pauper")

#: Commander is what the bench BUILDS and SIMULATES; the others validate and
#: filter. Default rather than only, and every caller reads this rather than a
#: literal, so a 60-card deck is a parameter away rather than a rewrite.
#:
#: Sideboards are not modelled. Constructed allows fifteen and the bench has no
#: concept of one — the sideboard was deleted from this repo deliberately. A
#: `sideboard_size` here would be a field nothing reads, which is the kind of
#: speculative completeness this module is trying to avoid.
DEFAULT = COMMANDER

FORMATS = {
    "commander": COMMANDER,
    "standard": STANDARD,
    "modern": MODERN,
    "pioneer": PIONEER,
    "pauper": PAUPER,
}


def get(name=None):
    """A format by name, or the default. Unknown names are an error, not a
    fallback — silently building Commander because a name was misspelled is the
    class of bug this module exists to prevent."""
    if name is None:
        return DEFAULT
    key = str(name).strip().lower()
    if key not in FORMATS:
        raise SystemExit(f"unknown format {name!r} — known: {', '.join(sorted(FORMATS))}")
    return FORMATS[key]
