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


def cardsfolder_digest(installed_copy=False):
    """A short digest of the archive's CONTENTS — the central directory's member
    names, CRCs and sizes — with no decompression. It moves when a member changes
    and stays when the zip is merely touched or re-copied, which is what a cache
    key and an artifact stamp (`engine.json._meta.forge_digest`) both need. None
    when Forge is absent."""
    z = _zip(installed_copy)
    if z is None:
        return None
    return _digest_of(str(z), z.stat().st_mtime_ns, z.stat().st_size)


@functools.lru_cache(maxsize=4)
def _digest_of(path, _mtime_ns, _size):
    import hashlib
    h = hashlib.sha256()
    with zipfile.ZipFile(path) as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if not info.filename.endswith(".txt"):
                continue
            h.update(f"{info.filename}:{info.CRC}:{info.file_size}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def _index_path(z):
    return FORGE_HOME / f".name_index-{cardsfolder_digest(False)}.json"


@functools.lru_cache(maxsize=1)
def _name_index():
    """{`Name:` line -> member path} over the SHIPPED archive. The filename guess
    in `scripts` misses a file Forge misspelled (Fallaji Archaeologist is
    `fallaji_archeologist.txt`) and a card filed under both faces when the list
    names one (Ojer Axonil, Deepest Might); both were read as "no script" on
    2026-10-21 and would have failed a seat audit for our own lookup's sake.

    BUILT ONLY ON A MISS AND KEPT ON DISK (2026-11-22). Building it decompresses all
    33,619 members, and `script()` consulted it on EVERY call, so a 1,235-card survey
    took 222 s. It is now read from `FORGE_HOME/.name_index-<digest>.json` when the
    archive's digest matches, rebuilt and written otherwise, and asked for only
    when a stem lookup fails. Empty when Forge is absent."""
    z = _zip(False)
    if z is None:
        return {}
    cache = _index_path(z)
    if cache.exists():
        try:
            import json
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:                            # pragma: no cover - a torn write
            pass
    out = {}
    with zipfile.ZipFile(z) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".txt"):
                continue
            try:
                head = zf.read(info).decode("utf-8", errors="replace")[:400]
            except Exception:                        # pragma: no cover - defensive
                continue
            for m in _NAME_RE.finditer(head):
                out.setdefault(m.group(1).strip(), info.filename)
    try:
        import json
        cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    except OSError:                                  # pragma: no cover - read-only home
        pass
    return out


def _stems_for(name):
    # A transform DFC is filed under its FRONT face; a split or `prepare` card under
    # its WHOLE name (`bloodline_recollector_ancestral_craving.txt`). Reading only the
    # front reported every prepare card as unscripted (2026-10-02).
    stems = [stem(name)]
    if " // " in name:
        whole = re.sub(r"[^a-z0-9]+", "_", re.sub(r"[',]", "", name.lower())).strip("_")
        stems.append(whole)
    return [f"{s[0]}/{s}.txt" for s in stems if s]


def scripts(names, installed_copy=False):
    """{name: script text or None} for many cards, THE ZIP OPENED ONCE. Stems are
    tried first for every name; the `Name:` index is consulted only for the names
    the stems missed, so the common case never pays for it. `installed_copy=True`
    reads what the engine PLAYS (overrides included); the default reads what Forge
    SHIPPED. Every name is a key of the result, None when there is no script or
    no Forge."""
    names = list(dict.fromkeys(names))
    z = _zip(installed_copy)
    if z is None:
        return {n: None for n in names}
    out = {}
    with zipfile.ZipFile(z) as zf:
        def read(member):
            try:
                return zf.read(member).decode("utf-8", errors="replace")
            except KeyError:
                return None
        missed = []
        for n in names:
            text = None
            for member in _stems_for(n):
                text = read(member)
                if text is not None:
                    break
            out[n] = text
            if text is None:
                missed.append(n)
        if missed:
            idx = _name_index()
            for n in missed:
                for face in [n] + [f.strip() for f in str(n).split(" // ")]:
                    member = idx.get(face)
                    if member:
                        out[n] = read(member)
                        if out[n] is not None:
                            break
    return out


def script(name, installed_copy=False):
    """One card's Forge script text, or None — `scripts([name])` for a single name."""
    return scripts([name], installed_copy=installed_copy)[name]


def token_script(token_stem):
    """A token's script (`res/tokenscripts/<stem>.txt`, the same grammar as a card's),
    or None. `Token | TokenScript$ b_1_1_vampire` names one; its `Colors:` and `Types:`
    are what a filter like `Creature.Black+token` is checked against."""
    p = FORGE_HOME / "res" / "tokenscripts" / f"{token_stem}.txt"
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return None
    # A transform DFC is filed under its FRONT face; a split or `prepare` card under
    # its WHOLE name (`bloodline_recollector_ancestral_craving.txt`). Reading only the
    # front reported every prepare card as unscripted (2026-10-02).
    stems = [stem(name)]
    if " // " in name:
        whole = re.sub(r"[^a-z0-9]+", "_", re.sub(r"[',]", "", name.lower())).strip("_")
        stems.append(whole)
    members = [f"{s[0]}/{s}.txt" for s in stems]
    idx = _name_index()
    for face in [name] + [f.strip() for f in str(name).split(" // ")]:
        if face in idx and idx[face] not in members:
            members.append(idx[face])
    with zipfile.ZipFile(z) as zf:
        for member in members:
            try:
                return zf.read(member).decode("utf-8", errors="replace")
            except KeyError:
                continue
    return None
