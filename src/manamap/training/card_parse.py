"""A card as TYPED FIELDS, not as a sentence.

## WHY THIS REPLACES `card_serialize`

The first architecture serialised every card into one string —
`[TYPE] … [COST] … [PT] … [TEXT] …` — and pushed it through a sentence encoder.
It looked structured and collapsed to text at the first step: CMC never existed
as a number, colour identity never existed as a set, and the model's only input
was MiniLM's opinion of a sentence.

The control says exactly what that cost. Scored on the real eval:

    frozen MiniLM 384d          function 0.629   theme 0.523   effdim 51.39
    PCA 128d of it              function 0.648   theme 0.494   effdim 42.62
    RANDOM 128d projection      function 0.602   theme 0.444   effdim 37.69
    the TRAINED model           function 0.618   theme 0.387   effdim 34.19

**A random projection beat the trained model on theme, and PCA beat it on
everything.** Training bought less than a matrix multiply with random numbers,
because there was nothing to learn from that was not already MiniLM.

This is a tabular problem with some text-valued columns. Each field gets its own
type, its own encoder and its own reconstruction head; a text span gets a frozen
sentence vector. Masking hides a FIELD.

## THE ABILITY CLASSES, and what the corpus sweep changed about them

34,388 cards, 64,430 ability lines, 1.87 per card. The first cut had three
classes and the sweep rejected two of its judgements:

1. **Ability words hide the trigger.** `Landfall — Whenever a land you control
   enters…` read as static because the classifier saw `Landfall` first. Stripping
   the `Word —` prefix moved **1,084 lines** from static to triggered.

2. **A sorcery's text is not a static ability.** `Destroy target artifact` was
   filed beside `Equipped creature gets +1/+1`, which tells the model those are
   the same kind of object. An instant or sorcery has no persistent abilities at
   all — its text IS the spell — so the class is decided by the TYPE LINE, not by
   the sentence.

3. **Keyword lines are their own thing.** `Flying`, `Ward {2}`, `Equip {3}` carry
   no clause structure, and two of those have costs with no colon to find them
   by. They are a vocabulary, not a sentence, and belong in a categorical field.
"""

import re

#: Ability words: `Landfall — `, `Metalcraft — `. Cosmetic, and they hide the
#: trigger word the classifier keys on.
_ABILITY_WORD = re.compile(r"^[A-Z][A-Za-z' ]{2,24}\s+—\s+")
_REMINDER = re.compile(r"\([^)]*\)")
_QUOTED = re.compile(r'"[^"]*"')
_TRIGGER = re.compile(r"^(When|Whenever|At )", re.IGNORECASE)
_MODAL = re.compile(r"^[•▪]\s*")

#: A line that is only keywords, optionally with a cost: `Flying`, `Flying,
#: haste`, `Ward {2}`, `Equip {3}`, `Morph {5}{G}`. Deliberately conservative —
#: anything with a verb clause falls through to `static`.
_KEYWORD_LINE = re.compile(
    r"^[A-Z][A-Za-z' ]{1,22}"                       # a keyword
    r"(\s*\{[^}]+\})*"                              # an optional cost
    r"(\s*,\s*[a-z][A-Za-z' ]{1,22}(\s*\{[^}]+\})*)*"   # more, comma-joined
    r"\.?$")

ABILITY_KINDS = ("spell", "activated", "triggered", "keyword", "static")


def ability_lines(oracle_text):
    """Oracle text -> its ability lines, reminder text kept.

    The RAW Scryfall text separates abilities with newlines. `extract.py:157`
    flattens them for `embedding_text`, which is right for a pooled vector and
    wrong here — the line boundary IS the ability boundary.
    """
    return [line.strip() for line in str(oracle_text or "").split("\n") if line.strip()]


def classify_line(line, type_line=""):
    """One of `ABILITY_KINDS`.

    `type_line` is not optional in spirit: an instant or sorcery has no
    persistent abilities, so its text is a SPELL EFFECT however it is worded.
    Deciding that from the sentence instead is how `Destroy target artifact`
    ends up filed as a static ability.
    """
    front = str(type_line).split("//")[0]
    body = _MODAL.sub("", _ABILITY_WORD.sub("", _REMINDER.sub(" ", line).strip())).strip()
    if _TRIGGER.match(body):
        return "triggered"
    # A colon OUTSIDE quotes. `Elves you control have "{T}: Add {G}{G}."` grants
    # an activated ability to something else; the card's own is static.
    if ":" in _QUOTED.sub("", body):
        return "activated"
    if "Instant" in front or "Sorcery" in front:
        return "spell"
    if _KEYWORD_LINE.match(body):
        return "keyword"
    return "static"


_COST_SPLIT = re.compile(r"^(?P<cost>[^:]{1,80}?):\s*(?P<effect>.+)$", re.DOTALL)


def split_activated(line):
    """`(cost, effect)` for an activated ability, else `(None, line)`.

    The cost is the half that says what the ability is worth — `{T}` against
    `{3}{B}, Sacrifice a creature` — and it is structured text the model should
    see as its own field rather than as the first few words of a sentence.
    """
    body = _ABILITY_WORD.sub("", _REMINDER.sub(" ", line).strip()).strip()
    match = _COST_SPLIT.match(_QUOTED.sub("", body))
    if not match:
        return None, line
    return match.group("cost").strip(), match.group("effect").strip()


def parse(card):
    """A card -> `{abilities: [{kind, text, cost}], counts: {kind: n}}`."""
    lines = ability_lines(card.get("oracle_text"))
    type_line = card.get("type_line") or ""
    abilities, counts = [], {kind: 0 for kind in ABILITY_KINDS}
    for line in lines:
        kind = classify_line(line, type_line)
        cost = split_activated(line)[0] if kind == "activated" else None
        abilities.append({"kind": kind, "text": line, "cost": cost})
        counts[kind] += 1
    return {"abilities": abilities, "counts": counts}
