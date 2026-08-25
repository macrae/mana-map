"""What a legal deck IS, as a parameter rather than an assumption.

PRD §13: v1 is Commander, and the constraints should be **parameters, not
assumptions**, because retrofitting format-awareness after the fact is
expensive. This module is that parameter. It ships exactly one format and the
point is not the second one — it is that the rules now have a name and a home.

They did not. Four places independently decided how big a Commander deck is:

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
- **The legal card pool.** `legality_key` names the Scryfall field to consult
  and nothing here reads it yet: legality is per-card data, the corpus already
  carries it, and inventing a pool-filtering layer before a second format needs
  one is the speculative half of this work.
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
    #: The key in Scryfall's `legalities` map. Named, not consulted — see the
    #: module docstring on why the pool filter is not built yet.
    legality_key: str

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


COMMANDER = FormatSpec(
    name="Commander",
    deck_size=100,
    singleton=True,
    commanders=1,
    colour_identity=True,
    basics_exempt=True,
    legality_key="commander",
)

#: The only format the bench builds, validates and simulates today. Every
#: caller reads this rather than a literal, so the day a second one arrives the
#: work is threading a parameter and not finding the assumptions.
DEFAULT = COMMANDER

FORMATS = {"commander": COMMANDER}


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
