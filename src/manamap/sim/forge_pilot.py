"""THE INSTALL/VERIFY BOUNDARY between this repo and the Forge engine.

`data/forge_overrides/` changes what Forge's AI may TARGET, and on a deck whose engine
is "target your own commander" that is the difference between 1/73 and 9/82 on one
unchanged list. So a run made under those scripts is a different measurement from one
made without, and every record says which.

IT SAID SO WITHOUT CHECKING. `forge.card_overrides()` shipped 2026-09-28 hashing the
REPO WORKING TREE — it never opened the engine. Three ways that lies:

  1. Restore `cardsfolder.zip.orig` and the next run still stamps the same sha while
     the AI misaims every spell. A FALSE provenance stamp is worse than none: the
     bucketing in `net_change.forge` trusts it to keep two harnesses apart.
  2. The path was RELATIVE, so `simulate` from any directory but the repo root read
     `null` with the overrides fully installed.
  3. Nothing installed them. The 11 scripts were patched into the zip by a script in
     a scratchpad, and the revert lived in a commit message.

This module is the answer to all three. `declared()` is what the repo holds,
`installed()` is what the engine carries, and `verify()` compares them — so the
question "was this run actually overridden" has an answer that reads the engine.

WHY A ZIP PATCH AT ALL, when `res/ai/*.ai` profiles are Forge's own supported lever and
need no patching: a profile cannot express a TARGET. `AiProps` is 121 knobs about
aggression, holding, sacrificing and mulliganing, and not one of them says which
creature to aim a pump spell at. `AITgts$` on the card is the only expression of that,
so targeting is the one thing that still costs an engine edit. Everything else belongs
in a profile (see `compile_profile`).
"""

import hashlib
import pathlib
import re
import shutil
import zipfile

from manamap import config

#: The tracked source of truth. Anchored on `config.DATA_DIR`, which is `__file__`-anchored,
#: so the answer does not depend on the process CWD — the second of the three lies above.
OVERRIDE_DIR = config.DATA_DIR / "forge_overrides" / "cards"

#: The engine's card scripts, and the pristine copy every install rebuilds from.
CARDSFOLDER = config.FORGE_HOME / "res" / "cardsfolder" / "cardsfolder.zip"
PRISTINE = CARDSFOLDER.with_suffix(".zip.orig")

#: Where a per-deck AI profile goes. Forge resolves this relative to the JVM's working
#: directory, which `forge.run` sets to the jar's parent — so this is the same tree the
#: games will read. `AiProfileUtil.getAvailableProfiles()` is `new File(dir).list()`, so
#: any file dropped here becomes a name `-a` accepts.
AI_DIR = config.FORGE_HOME / "res" / "ai"

#: The clause an override adds, and the only edit it makes.
#:
#: `Ally` names Zada because her shipped script is `Types:Legendary Creature Goblin Ally`.
#: THE SELECTOR GRAMMAR COST TWO FAILED RUNS and both failures were the same lesson — a
#: target class is read out of the corpus, never guessed:
#:
#:   `Creature.YouCtrl+namedZada, Hedron Grinder`  the COMMA separates alternatives
#:       (`agency_outfitter.txt` ships `namedMagnifying Glass,Card.namedThinking Cap`),
#:       so this parsed as `namedZada` OR ` Hedron Grinder` and matched nothing.
#:   `Creature.YouCtrl+Ally`  a SUBTYPE is the HEAD of a target class, never a `+`
#:       clause. Across the corpus the head carries the subtype (`Zombie` 8 uses,
#:       `Goblin` 4, `Ally` 2) while `+` takes only `YouCtrl` (132), `Other` (68),
#:       `YouOwn` (44), `OppCtrl` (30) and colours.
AITGTS = "AITgts$ Ally.YouCtrl"


class EngineMismatch(RuntimeError):
    """The engine does not carry what the repo declares.

    Raised rather than reported, because the alternative is a run that stamps a
    fingerprint it did not earn — and `net_change.forge` buckets on that fingerprint to
    keep two harnesses from pooling into a rate describing neither deck.
    """


def _digest(pairs):
    """Fingerprint `(relative path, bytes)` pairs. One function, so `declared()` and
    `installed()` cannot drift into hashing the same content two different ways —
    which is the only way a comparison between them could report a false mismatch."""
    h = hashlib.sha256()
    for name, blob in sorted(pairs):
        h.update(name.encode())
        h.update(blob)
    return h.hexdigest()[:12]


def _override_files():
    """The tracked override scripts as `(letter/name.txt, bytes)`."""
    if not OVERRIDE_DIR.is_dir():
        return []
    return [(p.relative_to(OVERRIDE_DIR).as_posix(), p.read_bytes())
            for p in sorted(OVERRIDE_DIR.rglob("*.txt"))]


def declared():
    """What the REPO says should be installed, or None when it declares nothing.

    None rather than an empty fingerprint, so a checkout with no overrides is
    byte-indistinguishable from the world before this module existed — the same
    absent-means-absent contract every `model_*` channel keeps.
    """
    files = _override_files()
    if not files:
        return None
    return {"sha": _digest(files), "n": len(files),
            "cards": sorted(pathlib.PurePosixPath(n).stem for n, _ in files)}


def installed():
    """What the ENGINE actually carries, read out of `cardsfolder.zip`.

    Compares each declared script against the entry of the same basename, so the answer
    is about the bytes the JVM will load rather than about anything this repo holds. A
    missing engine, a missing zip or a missing entry all read as not installed, because
    "I could not check" and "it is not there" have the same consequence for a rate.
    """
    files = _override_files()
    if not files or not CARDSFOLDER.is_file():
        return None
    try:
        zf = zipfile.ZipFile(CARDSFOLDER)
    except (zipfile.BadZipFile, OSError):
        return None
    with zf:
        by_base = {}
        for entry in zf.namelist():
            if entry.endswith(".txt"):
                by_base.setdefault(entry.rsplit("/", 1)[-1], entry)
        present = []
        for name, _ in files:
            base = pathlib.PurePosixPath(name).name
            entry = by_base.get(base)
            if entry is None:
                continue
            present.append((name, zf.read(entry)))
    if not present:
        return None
    return {"sha": _digest(present), "n": len(present),
            "cards": sorted(pathlib.PurePosixPath(n).stem for n, _ in present)}


def verify():
    """`(agrees, declared, installed)` — the whole provenance question, answered.

    `agrees` is True only when the repo declares nothing and the engine carries nothing,
    or when the two fingerprints match. Anything else is a disagreement, including the
    case that motivated this module: the repo declaring 11 scripts the engine does not
    have.
    """
    d, i = declared(), installed()
    if d is None and i is None:
        return True, None, None
    if d is None or i is None:
        return False, d, i
    return d["sha"] == i["sha"], d, i


def require_agreement():
    """Raise unless the engine matches the repo. Called before a run starts.

    A run that discovers the mismatch afterwards has already spent the JVM time and
    written a record; one that refuses has cost nothing and says what to do.
    """
    agrees, d, i = verify()
    if agrees:
        return
    what = (f"repo declares {d['n']} override(s) ({d['sha']}), "
            f"engine carries {i['n'] if i else 0}"
            + (f" ({i['sha']})" if i else " — none"))
    raise EngineMismatch(
        f"{what}.\n"
        f"A run made now would stamp a fingerprint it did not earn, and "
        f"`net_change` buckets on that fingerprint to keep two harnesses apart.\n"
        f"  install: manamap pilot forge-install\n"
        f"  revert:  manamap pilot forge-install --revert")


def install():
    """Rebuild `cardsfolder.zip` from the pristine copy, overlaying the tracked scripts.

    ALWAYS FROM PRISTINE, never from the current zip, so installing twice is identical
    to installing once and an override can never be layered onto an older override. That
    is the property that makes this safe to run from a fleet sweep.

    On a first install with no `.orig` beside it, the current zip becomes the pristine
    copy — but only after checking it carries none of our scripts, because baking an
    override into the baseline would make the distortion permanent and invisible.
    """
    if not CARDSFOLDER.is_file():
        raise FileNotFoundError(
            f"no card scripts at {CARDSFOLDER} — is Forge installed under "
            f"{config.FORGE_HOME}?")
    files = _override_files()
    if not PRISTINE.is_file():
        # "ALREADY OVERRIDDEN" IS NOT "THE CARD EXISTS". `installed()` fingerprints
        # whatever the engine carries for each declared basename, so it is non-None on a
        # perfectly pristine engine — the first cut of this guard read that as "already
        # patched" and refused every first install. The dangerous case is narrower: the
        # engine already carries OUR bytes and there is no baseline to recover.
        if files and verify()[0]:
            raise EngineMismatch(
                f"{CARDSFOLDER.name} already carries override scripts and there is no "
                f"{PRISTINE.name} to rebuild from. Baking them into the baseline would "
                f"make the distortion permanent and unmeasurable. Reinstall Forge's "
                f"card scripts, then run this again.")
        shutil.copy2(CARDSFOLDER, PRISTINE)
    if not files:
        return uninstall()
    src = zipfile.ZipFile(PRISTINE)
    with src:
        by_base = {}
        for entry in src.namelist():
            if entry.endswith(".txt"):
                by_base.setdefault(entry.rsplit("/", 1)[-1], entry)
        replace = {}
        for name, blob in files:
            base = pathlib.PurePosixPath(name).name
            entry = by_base.get(base)
            if entry is None:
                raise EngineMismatch(
                    f"{name} overrides a card the engine does not ship ({base}). An "
                    f"override is derived from a shipped script; one with no original "
                    f"cannot be.")
            replace[entry] = blob
        tmp = CARDSFOLDER.with_suffix(".zip.new")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
            for info in src.infolist():
                blob = replace.get(info.filename)
                out.writestr(info, blob if blob is not None else src.read(info.filename))
    shutil.move(str(tmp), str(CARDSFOLDER))
    return len(replace)


def uninstall():
    """Restore the pristine card scripts. The revert that lived in a commit message."""
    if not PRISTINE.is_file():
        return 0
    shutil.copy2(PRISTINE, CARDSFOLDER)
    return 0


def generate_overrides(cards, out_dir=None):
    """Write override scripts DERIVED from the shipped ones, adding exactly one clause.

    Derived rather than hand-written so an override can never silently diverge from the
    card it overrides — the failure mode would be a script that changes a cost or drops
    a mode while claiming to change only the AI's aim. `ValidTgts$` is never touched, so
    nothing LEGAL changes: a human pilot's options are identical and only the AI's
    choice narrows.

    Returns `(written, unsteerable)`. A card is UNSTEERABLE when its targeting line is
    `SP$ CopyPermanent`, because `forge.ai.ability.CopyPermanentAi` never reads
    `AITgts` — verified in the 2.0.14 bytecode, where the class references `AILogic` and
    the sixteen logic names it implements and `AITgts` is not among its strings. The
    corpus agrees: `AITgts$` sits beside `Destroy` 22 times, `ChangeZone` 12, `Pump` 7,
    and `CopyPermanent` zero.

    Writing one anyway would be a flag the engine cannot act on, which is the failure
    `tests/test_metric_hygiene.py` exists for — so it is refused and named instead.
    """
    out_dir = pathlib.Path(out_dir) if out_dir else OVERRIDE_DIR
    source = PRISTINE if PRISTINE.is_file() else CARDSFOLDER
    if not source.is_file():
        raise FileNotFoundError(f"no card scripts to derive from at {source}")
    zf = zipfile.ZipFile(source)
    with zf:
        by_stem = {n.rsplit("/", 1)[-1][:-4]: n
                   for n in zf.namelist() if n.endswith(".txt")}
        written, unsteerable = [], []
        for stem in cards:
            entry = by_stem.get(stem)
            if entry is None:
                raise KeyError(f"{stem}: no such card script in {source.name}")
            lines, touched, skip = [], 0, False
            for line in zf.read(entry).decode("utf-8", "replace").splitlines():
                if "CopyPermanent" in line and "ValidTgts$" in line:
                    skip = True
                    lines.append(line)
                    continue
                if "ValidTgts$" in line and "Creature" in line and "AITgts$" not in line:
                    # After the ValidTgts clause, so the file reads like a shipped one.
                    # An existing AITgts$ is the card author's decision and is left be.
                    line = re.sub(r"(ValidTgts\$[^|]*)", r"\1| " + AITGTS + " ",
                                  line, count=1)
                    line = re.sub(r"\s*\|\s*", " | ", line).strip()
                    touched += 1
                lines.append(line)
            if skip:
                unsteerable.append(stem)
                continue
            if touched != 1:
                raise ValueError(
                    f"{stem}: expected exactly one creature-targeting line, patched "
                    f"{touched}. Refusing rather than guessing which line meant it.")
            dest = out_dir / stem[0] / f"{stem}.txt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
            written.append(stem)
    return written, unsteerable


#: The cards the fleet currently overrides. A LIST, not a glob over the deck, because an
#: override is a deliberate distortion of the instrument and the set of them is a thing
#: somebody decided — `--generate` re-derives exactly these and nothing else.
#:
#: Fifteen were tried; four are absent and stay absent. Molten Duplication, Heat Shimmer,
#: Electroduplicate and Kindle the Inner Flame are `SP$ CopyPermanent`, which is the one
#: API whose AI never reads the hint. They are the cards this whole effort started from,
#: and the goldfish is the only instrument that can price them.
OVERRIDDEN = ("ancestors_aid", "ancestral_anger", "assault_strobe", "crimson_wisps",
              "expedite", "fists_of_flame", "impolite_entrance", "reckless_ransacking",
              "renegade_tactics", "sudden_breakthrough", "wild_ride")


def render():
    """The provenance question, as a reader sees it."""
    agrees, d, i = verify()
    out = []
    if d is None and i is None:
        return ["no card-script overrides — Forge is playing its own cards, and a rate "
                "from this engine is Forge's AI as shipped"]
    out.append(f"repo declares  {d['n'] if d else 0} script(s)"
               + (f"  sha {d['sha']}" if d else ""))
    out.append(f"engine carries {i['n'] if i else 0} script(s)"
               + (f"  sha {i['sha']}" if i else ""))
    if agrees:
        out.append("AGREES — a run's `card_overrides.sha` describes what played the games")
        out.append(f"  cards: {', '.join(d['cards'])}")
    else:
        out.append("DISAGREES — `simulate` will refuse rather than stamp a fingerprint "
                   "it did not earn")
        out.append("  fix: manamap pilot forge-install")
    return out


def main(args=None):
    """`manamap pilot forge-install [--verify] [--revert] [--generate]`."""
    verify_only = bool(getattr(args, "verify", False))
    revert = bool(getattr(args, "revert", False))
    generate = bool(getattr(args, "generate", False))

    if generate:
        written, unsteerable = generate_overrides(OVERRIDDEN)
        print(f"derived {len(written)} override(s) from "
              f"{(PRISTINE if PRISTINE.is_file() else CARDSFOLDER).name}")
        if unsteerable:
            print(f"  REFUSED as unsteerable (CopyPermanentAi never reads AITgts$): "
                  f"{', '.join(sorted(unsteerable))}")

    if revert:
        uninstall()
        print(f"reverted {CARDSFOLDER.name} to the pristine card scripts")
        for line in render():
            print(f"  {line}")
        return

    if not verify_only:
        try:
            n = install()
        except (EngineMismatch, FileNotFoundError) as exc:
            raise SystemExit(str(exc)) from exc
        print(f"installed {n} override(s) into {CARDSFOLDER.name}")

    for line in render():
        print(f"  {line}")
    # EXIT NONZERO ON A DISAGREEMENT, so this is usable as a gate in a script rather
    # than only as something a person reads.
    if not verify()[0]:
        raise SystemExit(1)
