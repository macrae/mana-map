"""What Forge's own card scripts say about a card — read from the engine, not guessed.

`AI:RemoveDeck` is Forge's marker for a card its AI has no logic for. It governs
deck GENERATION, not whether a supplied deck may contain it — Swan Song carries
`RemoveDeck:All` and was cast 28 times in 160 games, which is the measurement
that killed an earlier claim in this repo that the flag meant "never cast".

So this reports the flag and says what it does and does not mean. A flagged card
is a card whose behaviour under the AI is UNRELIABLE, which is a fact about the
INSTRUMENT rather than about the card: it may be a fine card in paper and a card
no Forge experiment can price.

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
    zips = list(FORGE_HOME.glob("res/cardsfolder/cardsfolder.zip"))
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
