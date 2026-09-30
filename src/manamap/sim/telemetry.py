"""Forge's log formatter, patched to log EVERY zone change — and the fingerprint that
says which jar a run was played under.

THE SHIPPED LOG STOPS AT TWO ZONE TRANSITIONS, Battlefield -> Graveyard and
Battlefield -> Exile, and `docs/simulation.md` said for a month that card advantage,
tutoring and recursion were "ABSENT and cannot be added". The engine fires
`GameEventCardChangeZone` for every move; the FORMATTER throws the rest away —
`forge.game.GameLogFormatter.visit(GameEventCardChangeZone)` returns null unless the
move is one of those two. One method, patched, and the log carries `Zone Change:
Sol Ring (123) was put into Hand from Library. owner Ai(1)-mm-goblin-storm` for every
draw, tutor, mill, wheel, discard and arrival, by name, with the owner appended so a
card that is drawn and never cast can still be charged to a seat (the parser's owner map
is learned from cast and land lines and is blind to exactly those cards).

MEASURED BEFORE IT SHIPPED (2026-09-30, the record is in docs/simulation.md): one short
two-seat game on a quiet machine, the pristine jar twice and the patched jar once, zero
AI timeouts, and the three logs differ on exactly one line each — `ended in N ms`. The
patch is purely observational. Every existing reader parses a patched log to
byte-identical facts with the new lines stripped.

WHAT THIS MODULE IS. `data/forge_patches/` tracks the `.java` and a manifest naming the
Forge version it was written against, the PRISTINE class sha, and every PATCHED class
sha a build has produced (javac output differs between JDKs, so a build on another
machine registers its own sha rather than failing a comparison against this one's).
`installed()` reads the class out of the jar `forge.run` would use and answers with the
same three states `forge_pilot.installed()` answers with for card scripts:

    PRISTINE   no telemetry jar, or one whose class is Forge's own  -> None (a plain run)
    INSTALLED  the class sha is one the manifest registers          -> the fingerprint
    NEITHER    a class nobody registered                             -> refuse the run

The run id carries `-tl<sha8>` and the record a `telemetry` block, so a record says which
log it has. It is NOT a bucket axis in `net_change.forge` and NOT excluded from a pod's
null, because the control above showed the games are the same games; the tag exists so
two runs of one configuration under two jars are two paths, and so a reader knows
whether the draw facts in a record were measured or are absent.

The patched jar is a COPY beside the pristine one (`forge-gui-desktop-<v>-mm-telemetry.jar`)
and the pristine jar is never touched, so `--plain-jar` is always available and the
pristine sha is always on disk to compare against.
"""

import datetime as _dt
import hashlib
import json
import pathlib
import shutil
import subprocess
import zipfile

from manamap import config

PATCH_DIR = config.DATA_DIR / "forge_patches"
SOURCE = PATCH_DIR / "GameLogFormatter.java"
MANIFEST = PATCH_DIR / "manifest.json"
CLASS_ENTRY = "forge/game/GameLogFormatter.class"
JAR_SUFFIX = "-mm-telemetry.jar"
_PRISTINE_SUFFIX = "-jar-with-dependencies.jar"


class EngineMismatch(RuntimeError):
    """The jar carries a formatter nobody registered. Refuse the run, say why."""


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()[:12]


def manifest():
    """What the repo declares, or None when it declares no patch."""
    if not MANIFEST.is_file():
        return None
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def source_sha():
    return _sha(SOURCE.read_bytes()) if SOURCE.is_file() else None


def class_sha(jar_path):
    """The formatter class's sha inside a jar, or None when the jar or entry is absent."""
    jar_path = pathlib.Path(jar_path)
    if not jar_path.is_file():
        return None
    try:
        with zipfile.ZipFile(jar_path) as zf:
            return _sha(zf.read(CLASS_ENTRY))
    except (zipfile.BadZipFile, KeyError, OSError):
        return None


def pristine_jar(home=None):
    from manamap.sim import forge
    return forge.forge_jar(home)


def telemetry_jar(home=None):
    """Where the patched copy lives: beside the pristine jar, suffixed."""
    p = pristine_jar(home)
    return p.with_name(p.name.replace(_PRISTINE_SUFFIX, JAR_SUFFIX))


def installed(home=None):
    """The telemetry fingerprint the ENGINE carries, or None when it carries none.

    Raises `EngineMismatch` for the third state — a jar whose formatter is neither
    Forge's own nor a build the manifest registers — because that is the only state a
    run must not start in: its log would have a shape nothing on disk describes.
    """
    man = manifest()
    if not man:
        return None
    jar = telemetry_jar(home)
    live = class_sha(jar)
    if live is None:
        return None                              # no patched copy: a plain run
    if live in man.get("patched_shas", []):
        return {"sha": live, "class": CLASS_ENTRY, "forge": (man.get("forge") or {}).get("version"),
                "source_sha": man.get("source_sha"), "jar": jar.name}
    if live == man.get("pristine_sha") or live == class_sha(pristine_jar(home)):
        return None                              # a copy that was never patched
    raise EngineMismatch(
        f"{jar.name} carries a {CLASS_ENTRY} (sha {live}) that {MANIFEST.name} "
        f"does not register — a stale build, or somebody else's patch. Its log would have a "
        f"shape nothing describes. Rebuild it (`manamap pilot forge-telemetry --build`) or "
        f"delete it to run plain.")


def jar_for_run(home=None, plain=False):
    """The jar `forge.run` and `experiment.run` should launch: the telemetry copy when
    it is installed and registered, the pristine jar otherwise or on `plain`."""
    if plain:
        return pristine_jar(home), None
    fp = installed(home)
    return (telemetry_jar(home), fp) if fp else (pristine_jar(home), None)


def count_new_lines(texts):
    """How many zone lines a set of logs carries that the shipped formatter would not
    have written — the receipt that the telemetry actually reached the log."""
    from manamap.sim import parse as sim_parse
    n = 0
    for t in texts:
        for line in t.splitlines():
            m = sim_parse.RX["zone"].match(line)
            if m and m.group(4) != "Battlefield":
                n += 1
    return n


def build(home=None, javac="javac"):
    """Compile the tracked source against the pristine jar and write the patched copy.

    Registers the resulting class sha in the manifest when it is new — commit that —
    and never touches the pristine jar. Needs a JDK on PATH (javac + jar); the spike
    used 21 and the class is compiled with `--release 21`.
    """
    if not SOURCE.is_file():
        raise FileNotFoundError(f"no patch source at {SOURCE}")
    pristine = pristine_jar(home)
    out = telemetry_jar(home)
    work = out.parent / ".mm-telemetry-build"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    subprocess.run([javac, "--release", "21", "-cp", str(pristine), "-d", str(work), str(SOURCE)],
                   check=True, capture_output=True, text=True)
    tmp = out.with_suffix(".jar.building")
    shutil.copy(pristine, tmp)
    subprocess.run(["jar", "uf", str(tmp), "-C", str(work), CLASS_ENTRY],
                   check=True, capture_output=True, text=True)
    tmp.replace(out)
    shutil.rmtree(work)
    live = class_sha(out)
    man = manifest() or {}
    man.setdefault("forge", None)
    man["pristine_sha"] = class_sha(pristine)
    man["source_sha"] = source_sha()
    shas = man.setdefault("patched_shas", [])
    registered = live in shas
    if not registered:
        shas.append(live)
        man.setdefault("builds", []).append({"sha": live, "at": _dt.date.today().isoformat(),
                                             "javac": _javac_version(javac)})
        MANIFEST.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    return {"jar": out, "sha": live, "registered_now": not registered}


def _javac_version(javac="javac"):
    try:
        r = subprocess.run([javac, "-version"], capture_output=True, text=True)
        return (r.stdout or r.stderr).strip() or None
    except OSError:
        return None


def render(home=None):
    man = manifest()
    lines = []
    if not man:
        lines.append("telemetry: the repo declares no formatter patch (data/forge_patches/ absent)")
        return lines
    lines.append(f"declared: {SOURCE.name} (source {man.get('source_sha')}), Forge "
                 f"{(man.get('forge') or {}).get('version')}, pristine class {man.get('pristine_sha')}, "
                 f"{len(man.get('patched_shas', []))} registered build(s)")
    try:
        fp = installed(home)
    except EngineMismatch as exc:
        lines.append(f"engine:   MISMATCH — {exc}")
        return lines
    if fp:
        lines.append(f"engine:   {fp['jar']} carries build {fp['sha']} — every zone change is "
                     f"logged with its owner; runs stamp -tl{fp['sha'][:8]}")
    else:
        jar = telemetry_jar(home)
        lines.append(f"engine:   no patched jar at {jar.name} — runs are plain (two zone "
                     f"transitions). Build one: manamap pilot forge-telemetry --build")
    return lines


def main(args=None):
    """`manamap pilot forge-telemetry [--build]` — report, or build and report."""
    if getattr(args, "build", False):
        try:
            r = build()
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"build failed:\n{exc.stderr or exc.stdout}") from exc
        print(f"built {r['jar'].name} — class {r['sha']}"
              + (" (NEW build registered in the manifest — commit it)" if r["registered_now"] else ""))
    for line in render():
        print(f"  {line}")
    try:
        installed()
    except EngineMismatch:
        raise SystemExit(1)
