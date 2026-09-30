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


def _engine_bytes(zip_path, names):
    """The engine's bytes for each declared basename, or None when unreadable."""
    if not zip_path.is_file():
        return None
    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError):
        return None
    with zf:
        by_base = {}
        for entry in zf.namelist():
            if entry.endswith(".txt"):
                by_base.setdefault(entry.rsplit("/", 1)[-1], entry)
        out = []
        for name in names:
            entry = by_base.get(pathlib.PurePosixPath(name).name)
            if entry is None:
                continue
            out.append((name, zf.read(entry)))
    return out or None


def installed():
    """The override fingerprint the ENGINE carries, or None when it carries none.

    THERE ARE THREE STATES HERE AND THE FIRST CUT OF THIS COLLAPSED THEM TO TWO, which
    made `simulate` refuse to start on a clean checkout — for every deck, including the
    seven that must never carry an override. `data/forge_overrides/` is tracked, so a
    fresh clone declares eleven scripts; the engine ships its own; and reporting the
    SHIPPED bytes as "installed, and not what you declared" is a check firing on correct
    data. That is the failure class this repo has rejected six validators for.

    So the engine is compared against BOTH sides:

      PRISTINE   its bytes are Forge's own      -> None. Nothing is installed, which is
                 a legitimate state and the one a clone starts in.
      INSTALLED  its bytes are the repo's       -> the override fingerprint.
      NEITHER    something else entirely        -> a fingerprint that matches neither,
                 which is the only genuine mismatch: a stale or partial install, and
                 the state worth refusing a run over.
    """
    files = _override_files()
    if not files:
        return None
    names = [n for n, _ in files]
    live = _engine_bytes(CARDSFOLDER, names)
    if live is None:
        return None
    live_sha = _digest(live)
    fingerprint = {"sha": live_sha, "n": len(live),
                   "cards": sorted(pathlib.PurePosixPath(n).stem for n, _ in live)}
    if live_sha == _digest(files):
        return fingerprint              # the engine carries OUR bytes: installed.
    pristine = _engine_bytes(PRISTINE, names)
    if pristine is None:
        # NO BASELINE, SO NO CLAIM. Without `.orig` there is nothing to tell Forge's own
        # script from a third party's hint, and asserting an install that cannot be
        # verified is the original sin of this module — the first cut hashed the repo and
        # called it provenance. Report not-installed and let the run proceed as plain.
        return None
    if live_sha == _digest(pristine):
        return None                     # Forge's own cards. Nothing of ours installed.
    return fingerprint                  # neither: a stale or partial install.


def verify():
    """`(agrees, declared, installed)` — the provenance question, answered.

    Agreement is not "the engine matches the repo". It is "the engine is in a state this
    repo can name", which is true in three of the four combinations:

      declares nothing, carries nothing   agrees — a plain engine, a plain run
      declares some,    carries nothing   agrees — the overrides are OFFERED, not
                                          required. This is the policy-OFF arm of every
                                          A/B, and a clean checkout, and making it an
                                          error is what bricked `simulate` for the fleet.
      declares some,    carries the same  agrees — the policy-ON arm
      declares some,    carries something agrees NOT — a stale or partial install, where
                        else              the record could describe neither arm

    A run is refused only in the last case, and `card_overrides()` returns None in the
    second — so a pristine run is byte-indistinguishable from one made before any of this
    existed, and it buckets with the tracked baseline record in `net_change.forge`.
    """
    d, i = declared(), installed()
    if i is None:
        return True, d, None
    if d is None:
        # The engine carries overrides this repo does not declare — unnameable.
        return False, None, i
    return d["sha"] == i["sha"], d, i


def require_agreement():
    """Raise only when the engine is in a state no record could describe.

    A run that discovers a mismatch afterwards has already spent the batch and published
    the claim; one that refuses has cost nothing. But refusing a VALID state costs the
    whole fleet, which is what the first cut of this did.
    """
    agrees, d, i = verify()
    if agrees:
        return
    raise EngineMismatch(
        f"the engine's card scripts are neither Forge's own nor the "
        f"{d['n'] if d else 0} this repo declares"
        + (f" ({d['sha']})" if d else "")
        + f" — it carries {i['sha']}.\n"
        f"A run now would stamp a fingerprint describing neither arm, and "
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
        # "ALREADY OVERRIDDEN" NEEDS A DIRECT COMPARISON, not `installed()` and not
        # `verify()`. Both of those answer None without a baseline — deliberately, since
        # they must not claim an install they cannot verify — so asking either one here is
        # circular and refused every first install. The dangerous case is narrow and
        # checkable on its own: the engine already carries OUR EXACT BYTES and there is no
        # `.orig` to recover, so copying the current zip would bake the distortion into
        # the baseline permanently and invisibly.
        live = _engine_bytes(CARDSFOLDER, [n for n, _ in files])
        if live is not None and _digest(live) == _digest(files):
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


#: THE CARDS THE AI WILL NOT CAST, and the list that lets it. `AI:RemoveDeck:All` is
#: read at PLAY time: `AiController.getSpellAbilityToPlay` filters every non-land
#: ability whose host `ComputerUtilCard.isCardRemAIDeck` — verified in the 2.0.14
#: bytecode on 2026-09-30, and measured first: in a 26-land shell with four copies
#: against a slow opponent, Vish Kal was castable in 7 of 12 games and cast in NONE;
#: the same script with only the `AI:RemoveDeck:All` line removed was cast 10 times.
#: The fleet sweep agrees — every `All` PERMANENT in a live deck with games on
#: record was cast 0 times (14 of 14: Viscera Seer, Goblin Bombardment, Altar of
#: Dementia, Isochron Scepter, Windfall's Magus…), while every `Random` one was cast
#: normally. This repo said for a month that the flag was deck-generation only,
#: measured on Swan Song — a COUNTERSPELL, cast through the AI's reactive path, which
#: does not filter. The claim was true of one path and false of the one that casts.
#:
#: `UNFLAG_FILE` lists, one script stem per line, the cards whose override is the
#: shipped script minus that line. A tracked list rather than a glob over the fleet
#: because the set is part of the instrument's fingerprint: derived from the decks it
#: would move on every swap and split every bucket. `unflag_candidates()` is the sweep
#: that says what the list is missing.
UNFLAG_FILE = config.DATA_DIR / "forge_overrides" / "unflag.txt"


def unflag_list():
    """The stems `unflag.txt` names, in order; empty when the file is absent."""
    if not UNFLAG_FILE.is_file():
        return []
    return [ln.strip() for ln in UNFLAG_FILE.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def generate_unflag(stems, out_dir=None):
    """Write override scripts that are the shipped script MINUS its `AI:RemoveDeck` line.

    Derived like `generate_overrides`, and layered on top of it: a card that is also in
    `OVERRIDDEN` starts from its AITgts override so both edits land in one file. Refuses a
    stem whose shipped script carries no flag (an unflag with nothing to remove would be a
    listed edit that changes nothing — the failure `test_metric_hygiene` exists for).
    """
    out_dir = pathlib.Path(out_dir) if out_dir else OVERRIDE_DIR
    source = PRISTINE if PRISTINE.is_file() else CARDSFOLDER
    if not source.is_file():
        raise FileNotFoundError(f"no card scripts to derive from at {source}")
    written = []
    with zipfile.ZipFile(source) as zf:
        by_stem = {n.rsplit("/", 1)[-1][:-4]: n for n in zf.namelist() if n.endswith(".txt")}
        for stem in stems:
            entry = by_stem.get(stem)
            if entry is None:
                raise KeyError(f"{stem}: no such card script in {source.name}")
            dest = out_dir / stem[0] / f"{stem}.txt"
            base = dest.read_text(encoding="utf-8") if dest.is_file() \
                else zf.read(entry).decode("utf-8", "replace")
            lines = base.splitlines()
            kept = [ln for ln in lines if not ln.startswith("AI:RemoveDeck:")]
            if len(kept) == len(lines):
                raise ValueError(f"{stem}: the shipped script carries no AI:RemoveDeck line — "
                                 f"nothing to unflag, so it does not belong in {UNFLAG_FILE.name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("\n".join(kept) + "\n", encoding="utf-8")
            written.append(stem)
    return written


def unflag_candidates():
    """Every `RemoveDeck:All` non-land card in a LIVE deck that `unflag.txt` does not
    name — the sweep behind the list. `[(slug, card name, stem)]`."""
    import json
    from manamap.pilot.common import deck_is_apart
    from manamap.sim import forge_cards
    listed = set(unflag_list())
    out = []
    for d in sorted(config.DECKS_DIR.iterdir()) if config.DECKS_DIR.is_dir() else []:
        cj = d / "cards.json"
        if not cj.is_file() or deck_is_apart(d.name):
            continue
        for c in json.loads(cj.read_text(encoding="utf-8")).get("cards", []):
            if forge_cards.ai_flag(c["name"]) != "All" or "Land" in (c.get("type_line") or ""):
                continue
            stem = forge_cards.stem(c["name"])
            if stem not in listed:
                out.append((d.name, c["name"], stem))
    return out


#: PER-DECK CARD HINTS. `data/decks/<slug>/forge_hints.json`:
#:
#:   {"hints": [{"card": "Vish Kal, Blood Arbiter",
#:               "ability": "AB$ PutCounter",              # the line to touch, by its API prefix
#:               "ai_logic": "AristocratCounters",         # appended as `| AILogic$ …` (optional)
#:               "ai_preference": {"SacCost": "Creature.token,Creature.Other+cmcLE2"},  # optional
#:               "why": "…", "cites": ["stacks/004", "log 006"]}]}
#:
#: The two hint kinds Forge's own aristocrat scripts use (`carrion_feeder.txt`: `AILogic$
#: AristocratCounters` on the sacrifice ability plus `SVar:AIPreference:SacCost$…`), derived
#: onto the shipped script the way every override here is. A hint on a card the deck does
#: not run, an `ability` prefix that matches no line or several, or an `AILogic$` already on
#: that line, is refused rather than guessed. `validate-forge-hints` is the gate; the
#: override lands in the same fingerprint as everything else.
HINTS_FILE = "forge_hints.json"
HINT_KEYS = frozenset({"card", "ability", "ai_logic", "ai_preference", "why", "cites"})


def hints_for(slug):
    """The deck's hints, or []."""
    import json
    from manamap.pilot.common import deck_dir
    path = deck_dir(slug) / HINTS_FILE
    if not path.is_file():
        return []
    return list((json.loads(path.read_text(encoding="utf-8")) or {}).get("hints") or [])


def validate_hints(slug, doc):
    """Form: every hint names a card in the 99, says why, and carries at least one hint."""
    import json
    from manamap.pilot.common import deck_dir
    errors = []
    hints = (doc or {}).get("hints")
    if not isinstance(hints, list) or not hints:
        return ["no `hints` list — a file that hints nothing should not exist"]
    names = set()
    cj = deck_dir(slug) / "cards.json"
    if cj.is_file():
        names = {c["name"] for c in json.loads(cj.read_text(encoding="utf-8")).get("cards", [])}
        names |= {n.split(" // ")[0] for n in names}
    seen = set()
    for i, h in enumerate(hints):
        where = f"hints[{i}]"
        if not isinstance(h, dict):
            errors.append(f"{where}: not an object"); continue
        unknown = set(h) - HINT_KEYS
        if unknown:
            errors.append(f"{where}: unknown key(s) {sorted(unknown)}")
        card = h.get("card")
        if not card:
            errors.append(f"{where}: no card"); continue
        if names and card not in names:
            errors.append(f"{where}: {card!r} is not in the 99")
        if card in seen:
            errors.append(f"{where}: {card!r} hinted twice — one hint per card")
        seen.add(card)
        if not h.get("why"):
            errors.append(f"{where}: {card}: no `why` — a hint is a piloting decision and says so")
        if not (h.get("ai_logic") or h.get("ai_preference")):
            errors.append(f"{where}: {card}: neither ai_logic nor ai_preference — nothing to hint")
        if h.get("ai_logic") and not h.get("ability"):
            errors.append(f"{where}: {card}: ai_logic needs `ability`, the API prefix of the line it goes on")
        pref = h.get("ai_preference")
        if pref is not None and (not isinstance(pref, dict) or not pref
                                 or not all(isinstance(k, str) and isinstance(v, str) for k, v in pref.items())):
            errors.append(f"{where}: {card}: ai_preference is a mapping of kind -> selector, e.g. {{\"SacCost\": \"Creature.token\"}}")
    return errors


def generate_hints(slug, out_dir=None):
    """Write the deck's hint overrides, layered on any override the card already has.
    Returns the stems written. Refuses an `ability` prefix matching zero or several
    lines, or a line already carrying `AILogic$`, or a preference the script already
    states — the same 'never guess' rule `generate_overrides` keeps."""
    from manamap.sim import forge_cards
    out_dir = pathlib.Path(out_dir) if out_dir else OVERRIDE_DIR
    hints = hints_for(slug)
    if not hints:
        return []
    source = PRISTINE if PRISTINE.is_file() else CARDSFOLDER
    if not source.is_file():
        raise FileNotFoundError(f"no card scripts to derive from at {source}")
    written = []
    with zipfile.ZipFile(source) as zf:
        by_stem = {n.rsplit("/", 1)[-1][:-4]: n for n in zf.namelist() if n.endswith(".txt")}
        for h in hints:
            stem = forge_cards.stem(h["card"])
            entry = by_stem.get(stem)
            if entry is None:
                raise KeyError(f"{h['card']}: no card script {stem}.txt in {source.name}")
            dest = out_dir / stem[0] / f"{stem}.txt"
            base = dest.read_text(encoding="utf-8") if dest.is_file() \
                else zf.read(entry).decode("utf-8", "replace")
            lines = base.splitlines()
            if h.get("ai_logic"):
                prefix = "A:" + h["ability"]
                idx = [i for i, ln in enumerate(lines) if ln.startswith(prefix)]
                if len(idx) != 1:
                    raise ValueError(f"{h['card']}: `{h['ability']}` matches {len(idx)} line(s) of {stem}.txt; "
                                     f"a hint goes on exactly one")
                if "AILogic$" in lines[idx[0]]:
                    raise ValueError(f"{h['card']}: that line already carries an AILogic$ — the card "
                                     f"author's decision, left be")
                lines[idx[0]] = lines[idx[0]] + f" | AILogic$ {h['ai_logic']}"
            for kind, selector in (h.get("ai_preference") or {}).items():
                key = f"SVar:AIPreference:{kind}$"
                if any(ln.startswith(key) for ln in lines):
                    raise ValueError(f"{h['card']}: {stem}.txt already states {key}… — left be")
                # before the Oracle line, where Forge's own scripts put it
                at = next((i for i, ln in enumerate(lines) if ln.startswith("Oracle:")), len(lines))
                lines.insert(at, key + selector)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
            written.append(stem)
    return written


def hinted_decks():
    """Every live deck with a hints file."""
    from manamap.pilot.common import deck_is_apart
    out = []
    for d in sorted(config.DECKS_DIR.iterdir()) if config.DECKS_DIR.is_dir() else []:
        if (d / HINTS_FILE).is_file() and not deck_is_apart(d.name):
            out.append(d.name)
    return out


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


def install_all_profiles():
    """Install every live deck's profile. `{slug: fingerprint or None}`.

    Withdrawn policies are cleaned up here too, which is why it runs over every deck rather
    than only the ones with a policy today: a deck whose rule was deleted still has a stale
    `.ai` in the engine, and a stale profile steers every later run while nothing reports it
    — the same hazard `install()` rebuilds from pristine to avoid.
    """
    from manamap import config
    from manamap.pilot.common import deck_is_apart

    out = {}
    for deck in sorted(config.DECKS_DIR.iterdir()):
        if not (deck / "decklist.txt").is_file():
            continue
        if deck_is_apart(deck.name):
            continue
        out[deck.name] = install_profile(deck.name)
    return out


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
        unflagged = generate_unflag(unflag_list())
        print(f"unflagged {len(unflagged)} card(s) the AI would never cast ({UNFLAG_FILE.name})")
        # PER-DECK HINTS LAST, so they layer on an unflag or an aim override.
        for slug in hinted_decks():
            errs = validate_hints(slug, {"hints": hints_for(slug)})
            if errs:
                raise SystemExit(f"{slug}/{HINTS_FILE}: " + "; ".join(errs))
            hinted = generate_hints(slug)
            print(f"hinted {len(hinted)} card(s) for {slug} ({HINTS_FILE}): {', '.join(hinted)}")
        missing = unflag_candidates()
        if missing:
            print(f"  NOT YET UNFLAGGED — RemoveDeck:All cards in live decks the AI will not cast:")
            for slug, name, stem in missing:
                print(f"    {slug}: {name}  ({stem})")

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
        # AND EVERY DECK'S PROFILE, because a policy that is declared and not installed is
        # a run refused (`forge.run` checks agreement) — so the one command that prepares
        # the engine prepares all of it.
        for slug, fp in sorted(install_all_profiles().items()):
            if fp:
                print(f"  profile {profile_name(slug)}.ai — {fp['n']} knob(s): "
                      f"{', '.join(fp['keys'])}")

    for line in render():
        print(f"  {line}")
    # EXIT NONZERO ON A DISAGREEMENT, so this is usable as a gate in a script rather
    # than only as something a person reads.
    if not verify()[0]:
        raise SystemExit(1)


# ─────────────────────────────────────────────── the per-deck AI profile

def profile_name(slug):
    """What `-a` calls this deck's profile. `@` flattens, as Forge's registry does."""
    return "mm-" + str(slug).replace("@", "-")


def profile_path(slug):
    return AI_DIR / f"{profile_name(slug)}.ai"


def declared_profile(slug, branch=None):
    """`(text, fingerprint)` for the deck's policy, or `(None, None)`.

    A BRANCH INHERITS THE DECK'S PROFILE, for the reason `pilot_policy.load` already
    inherits the policy: two candidate 99s are only comparable if the same hand pilots
    both, and a branch with its own piloting would be the "two models, not two lists"
    defect that cost this project a day on 2026-09-27.
    """
    from manamap.pilot import forge_ai, pilot_policy

    doc = pilot_policy.load(slug, branch)
    if not (doc.get("forge") or {}):
        return None, None
    return (forge_ai.compile_profile(doc, name=profile_name(slug)),
            forge_ai.forge_fingerprint(doc))


def installed_profile(slug):
    """The profile text the ENGINE has for this deck, or None.

    The same question `installed()` asks of the card scripts, for the same reason: what a
    record claims must be what the JVM loaded, and the JVM reads `res/ai/`.
    """
    path = profile_path(slug)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def install_profile(slug, branch=None):
    """Write the deck's compiled profile into the engine. Returns its fingerprint or None.

    Removes a stale profile when the policy no longer declares one, for the same reason
    `install()` rebuilds the card scripts from pristine: a WITHDRAWN piloting rule that
    stays in the engine keeps steering every later run while nothing reports it.
    """
    text, fp = declared_profile(slug, branch)
    path = profile_path(slug)
    if text is None:
        if path.is_file():
            path.unlink()
        return None
    AI_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return fp


def profile_agrees(slug, branch=None):
    """Is the engine's profile the one this deck's policy declares?

    Three states again, and the same reasoning as `verify()`: no policy and no file agree;
    a policy whose file is missing or stale does not. A run in the third state would be
    flown by a profile nobody declared.
    """
    text, _ = declared_profile(slug, branch)
    live = installed_profile(slug)
    if text is None and live is None:
        return True
    return text == live


def profile_content_sha(name):
    """A sha over the `.ai` file the ENGINE holds under `name`, or None.

    Over the file's KEY=VALUE lines rather than its bytes, so the generated header and the
    `why` comments — which explain the policy and do not change how a game is played —
    cannot move a run id. The same call `forge_ai.forge_fingerprint` makes over the
    declaration; this one asks the engine, which is what actually plays.

    Returns None for Forge's own four profiles: they are named in the id already by
    `profile_tag`, they do not change, and hashing them would rename every historical record.
    """
    import hashlib

    if not name or not str(name).startswith("mm-"):
        return None
    path = AI_DIR / f"{name}.ai"
    if not path.is_file():
        return None
    body = sorted(ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
                  if ln.strip() and not ln.lstrip().startswith("#"))
    return hashlib.sha256("\n".join(body).encode()).hexdigest()[:12]
