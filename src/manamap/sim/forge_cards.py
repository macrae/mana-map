"""What Forge's own card scripts say about a card — read from the engine, not guessed.

`AI:RemoveDeck` is Forge's marker for a card its AI has no logic for. `RemoveDeck:All`
IS "never cast" for anything the AI casts proactively: `AiController.getSpellAbilityToPlay`
filters every non-land ability whose host `isCardRemAIDeck` (2.0.14 bytecode, verified
2026-09-30), and the fleet sweep agrees — every `All` permanent in a live deck with games
on record was cast 0 times, 14 of 14. This file said the opposite for a month on the
strength of Swan Song, cast 28 times in 160 games: a COUNTERSPELL, cast through the AI's
reactive path, which does not filter. `RemoveDeck:Random` is deck-generation only and
such cards are cast normally.

So this reports the flag, and `forge_pilot.unflag.txt` is the tracked list of cards
whose override removes it (see `forge_pilot.generate_unflag`). A flagged card that is
NOT unflagged is a card no Forge experiment can price at all; an unflagged one is played
by an AI that has no special logic for it, which is a floor and a measurable one.

Read straight out of `cardsfolder.zip` under FORGE_HOME so it cannot drift from
the engine actually running the games. Absent when Forge is not installed.
"""

import functools
import re
import zipfile

from manamap.config import FORGE_HOME

_NAME_RE = re.compile(r"^Name:(.+)$", re.M)
_AI_RE = re.compile(r"^AI:RemoveDeck:(\w+)", re.M)


@functools.lru_cache(maxsize=1)
def _flags():
    """{card name: RemoveDeck kind}. Empty when Forge is not installed."""
    # THE SHIPPED SCRIPTS, not the installed ones: `forge-install` overlays the repo's
    # overrides onto `cardsfolder.zip` (an unflag override REMOVES the line this reads),
    # and the pristine copy it keeps beside it is what Forge ships. A flag is a fact about
    # the card as Forge wrote it; whether we unflagged it is `forge_pilot.unflag_list`.
    zips = list(FORGE_HOME.glob("res/cardsfolder/cardsfolder.zip.orig")) \
        or list(FORGE_HOME.glob("res/cardsfolder/cardsfolder.zip"))
    if not zips:
        return {}
    out = {}
    with zipfile.ZipFile(zips[0]) as z:
        for info in z.infolist():
            if not info.filename.endswith(".txt"):
                continue
            try:
                text = z.read(info).decode("utf-8", errors="replace")
            except Exception:                        # pragma: no cover - defensive
                continue
            a = _AI_RE.search(text)
            if not a:
                continue
            n = _NAME_RE.search(text)
            if n:
                out[n.group(1).strip()] = a.group(1)
    return out


def stem(name):
    """The card-script stem Forge uses for a name: lowercase, apostrophes and commas
    dropped, every other non-alphanumeric run an underscore. `Vish Kal, Blood Arbiter`
    -> `vish_kal_blood_arbiter`; a DFC uses its front face."""
    front = name.split(" // ")[0]
    s = re.sub(r"[',]", "", front.lower())
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def installed():
    return bool(_flags()) or bool(list(FORGE_HOME.glob("res/cardsfolder/cardsfolder.zip")))


def ai_flag(name):
    """`RemoveDeck` kind for this card, or None. Matches either face of a DFC."""
    f = _flags()
    if not f:
        return None
    if name in f:
        return f[name]
    for face in str(name).split(" // "):
        if face.strip() in f:
            return f[face.strip()]
    return None


def _zip(installed_copy=False):
    """The shipped archive (`.orig`, what Forge wrote) or the INSTALLED one (with the
    repo's overrides laid over it) — two different questions about one card."""
    if installed_copy:
        zips = list(FORGE_HOME.glob("res/cardsfolder/cardsfolder.zip"))
    else:
        zips = list(FORGE_HOME.glob("res/cardsfolder/cardsfolder.zip.orig")) \
            or list(FORGE_HOME.glob("res/cardsfolder/cardsfolder.zip"))
    return zips[0] if zips else None


def script(name, installed_copy=False):
    """A card's Forge script text, or None when Forge is absent or the card has no
    script. `installed_copy=True` reads what the engine PLAYS (overrides included);
    the default reads what Forge SHIPPED. A DFC is its front face's stem."""
    z = _zip(installed_copy)
    if z is None:
        return None
    s = stem(name)
    entry = f"{s[0]}/{s}.txt"
    try:
        with zipfile.ZipFile(z) as zf:
            return zf.read(entry).decode("utf-8", errors="replace")
    except KeyError:
        return None
