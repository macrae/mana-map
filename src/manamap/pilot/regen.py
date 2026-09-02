"""Pilot: regenerate the fleet's derived artifacts, in dependency order.

THERE WAS NO COMMAND FOR THIS. A model change stales every deck's figures, and
the only way to rebuild them was a hand-written shell loop per artifact — one
was written during the fetchland fix on 2026-08-30 and broke twice on zsh
globbing before it ran. A loop nobody keeps is a loop that is wrong when it
matters, and it ran serially: ~125s of goldfish alone.

PARALLEL ACROSS TARGETS, NEVER WITHIN ONE. Each deck's `goldfish.run` threads a
single `random.Random(seed)` through all 10,000 games in sequence, so chunking
the games would move every published figure. Chunking the DECKS moves nothing:
each target constructs its own generator from the same seed and writes only its
own directory. The output is bit-identical to the serial run, and this module
has a test that says so.

STAGES ARE A BARRIER, and the order is not decoration. `mana_analysis` embeds
goldfish figures, `net_change` reads the deck's diagnostic, `deck-info` composes
everything. Running a later stage against an earlier stage's stale output is
exactly the "measured against the wrong list" class this repo has already paid
for, so each stage completes before the next begins.

RETIRED DECKS ARE SKIPPED, matching the freshness gates: their artifacts are
history, not claims, and regenerating a document about a deck nobody will
shuffle only churns the tree.
"""

import concurrent.futures
import contextlib
import importlib
import io
import time

from manamap.config import DECKS_DIR
from manamap.pilot.common import deck_lifecycle


class _Args:
    """A parsed-args stand-in; every producer's `main` takes one of these."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, name):        # unset flags read as absent, not error
        return None


#: (stage, artifact, module, extra kwargs for main's Args), in DEPENDENCY ORDER.
#: `branch_only` marks a stage that exists for candidate lists and not for the
#: deck itself — `net-change` compares a branch AGAINST the deck, so there is
#: nothing for it to say about the deck alone.
STAGES = (
    ("goldfish", "goldfish_metrics.json", "manamap.pilot.goldfish", {}, False),
    ("mana-analysis", "mana_analysis.json", "manamap.pilot.mana_analysis", {}, False),
    ("net-change", "net_change.json", "manamap.pilot.net_change", {"write": True}, True),
    ("diagnose", "diagnostic.json", "manamap.pilot.diagnostic", {"write": True}, False),
    ("benchmark", "benchmark.json", "manamap.pilot.benchmark", {}, False),
    ("deck-info", "info.json", "manamap.pilot.deck_info", {"write": True}, False),
)

STAGE_NAMES = tuple(s[0] for s in STAGES)


def is_retired(slug):
    """Reuses `common.deck_lifecycle` — the authored identity, one home.

    It returns None for a deck with no `issue.json`, which is most of them now
    that the magazine is legacy. None means "nothing was declared", which is not
    retired.
    """
    life = deck_lifecycle(slug)
    return bool(life and life[0])


#: An artifact a LIVE deck should have even when it does not yet, keyed to the
#: file that proves the deck is real enough to measure.
#:
#: WHY THIS EXISTS. `targets()` used to return only places the artifact ALREADY
#: existed, which makes `regen` a refresher that can never bootstrap: a deck
#: missing `diagnostic.json` was skipped by the `diagnose` stage forever, in
#: silence, and `deck-status` reported the file as a bare GATE with nothing
#: saying why it never arrived. Two of the three PINNED decks were in that state
#: — goblin-storm and edgar-vampires — so the dossier's vitals section and the
#: cover sheet's engine-health word were absent on decks that are sleeved and
#: played. A refresher that cannot create is a refresher that hides its own
#: gaps.
#:
#: Deliberately narrow in two directions.
#:
#: ONLY PURELY-DERIVED ARTIFACTS. Nothing authored, nothing an agent writes,
#: nothing that needs a branch — creating one of those would be the tool
#: inventing a claim rather than recomputing one.
#:
#: AND ONLY ON A SLEEVED DECK. A deck that is pinned in paper is one the pilot
#: plays, and its measurements are built automatically and kept complete — that
#: is what pinning MEANS. A deck on the bench is malleable: it changes daily,
#: nobody has said it exists in cardboard, and its stages are triggered by hand
#: when the pilot wants them. Bootstrapping there would manufacture artifacts
#: for a list that will be different tomorrow, and would put a freshness gate on
#: work in progress.
BOOTSTRAP = {"diagnostic.json": "goldfish_metrics.json"}


def is_pinned(slug):
    """Is this deck SLEEVED — built in paper, and therefore built automatically?

    Reads the paper lock, the one authored claim about cardboard, through the
    module that owns it. Not a second predicate: `deck_versions.paper` is where
    "this exact 99 is in sleeves" lives.
    """
    from manamap.pilot import deck_versions

    return bool(deck_versions.paper(slug))


def targets(artifact, slug=None):
    """`(slug, branch)` for every place this artifact is tracked, branches too.

    SLEEVED DECKS ONLY, unless a slug is NAMED. This is the bench's rule and it
    is structural rather than a convention somebody has to remember:

        a deck that is SLEEVED is played, so its measurements are kept current
        automatically — that is what pinning means;

        a deck ON THE BENCH is malleable. It changes daily, nobody has claimed
        it exists in cardboard, and rebuilding its figures on a sweep measures a
        list that will be different tomorrow;

        a deck in the ARCHIVE is history. Its artifacts are frozen as published.

    Naming a slug is the manual trigger — `regen --slug heliod` does exactly what
    it says on any deck, sleeved or not. A bare `regen` is the automatic pass and
    touches only what is in sleeves.

    A `BOOTSTRAP` artifact is also returned where it is MISSING but its
    precondition is present, so the stage creates it rather than skipping it.
    """
    if not DECKS_DIR.is_dir():
        return []
    needs = BOOTSTRAP.get(artifact)
    out = []
    for deck in sorted(DECKS_DIR.iterdir()):
        if not deck.is_dir() or is_retired(deck.name):
            continue
        # NAMED = manual, and it means THIS DECK ONLY. Unnamed = the automatic
        # sweep, which sees sleeved decks only.
        if slug is not None:
            if deck.name != slug:
                continue
        elif not is_pinned(deck.name):
            continue
        bootstrappable = needs and (deck / needs).exists() and is_pinned(deck.name)
        if (deck / artifact).exists() or bootstrappable:
            out.append((deck.name, None))
        for branch in sorted((deck / "branches").glob("*")):
            # NOT bootstrapped on a branch. A branch is a candidate list, and
            # creating a measurement it never asked for is work nobody ordered.
            if branch.is_dir() and (branch / artifact).exists():
                out.append((deck.name, branch.name))
    return out


def plan(only=None, slug=None):
    """`[(stage, module, kwargs, [(slug, branch), …]), …]` — what would run.

    `slug` is passed THROUGH to `targets`, not just used to filter its result.
    That is the whole of the sleeved-only rule: without a slug the sweep is
    automatic and sees sleeved decks only; with one it is a manual trigger and
    sees whatever deck was named. Filtering afterwards would have made
    `regen --slug heliod` return nothing at all on a bench deck, which is the
    opposite of what naming a deck means.
    """
    rows = []
    for stage, artifact, module, kwargs, branch_only in STAGES:
        if only and stage not in only:
            continue
        found = [t for t in targets(artifact, slug=slug)
                 if not (branch_only and t[1] is None)]
        if found:
            rows.append((stage, module, kwargs, found))
    return rows


def _one(job):
    """Run one producer. Module-level and picklable, for the process pool."""
    module, kwargs, slug, branch = job
    started = time.time()
    try:
        args = _Args(slug=slug, branch=branch, **kwargs)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            importlib.import_module(module).main(args)
        return (slug, branch, None, time.time() - started)
    except BaseException as exc:                    # noqa: BLE001 - reported
        return (slug, branch, f"{type(exc).__name__}: {exc}", time.time() - started)


def run(only=None, slug=None, jobs=None, dry_run=False, echo=print):
    """Regenerate, stage by stage. Returns `{"failures": [...], "seconds": n}`."""
    rows = plan(only, slug)
    if not rows:
        echo("  nothing to regenerate — no tracked artifacts matched")
        return {"failures": [], "seconds": 0.0, "ran": 0}

    total = sum(len(t) for _s, _m, _k, t in rows)
    echo(f"REGEN — {total} target(s) across {len(rows)} stage(s)"
         + (f", deck {slug}" if slug else "")
         + (f", {jobs} job(s)" if jobs and not dry_run else ""))
    if dry_run:
        for stage, _module, _kwargs, found in rows:
            echo(f"\n  {stage}  ({len(found)})")
            for s, b in found:
                echo(f"    {s}" + (f"@{b}" if b else ""))
        echo("\n  --dry-run: nothing was written")
        return {"failures": [], "seconds": 0.0, "ran": 0}

    failures, began = [], time.time()
    for stage, module, kwargs, found in rows:
        echo(f"\n  {stage}  ({len(found)})")
        jobs_list = [(module, kwargs, s, b) for s, b in found]
        started = time.time()
        # A pool of ONE runs in-process: cheaper for a single target, and it
        # keeps `--slug` debuggable because a traceback is not pickled.
        if jobs == 1 or len(jobs_list) == 1:
            results = [_one(j) for j in jobs_list]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
                results = list(pool.map(_one, jobs_list))
        for s, b, error, seconds in results:
            name = s + (f"@{b}" if b else "")
            if error:
                failures.append((stage, name, error))
                echo(f"    {name:34} FAILED  {error}")
            else:
                echo(f"    {name:34} ok      {seconds:5.1f}s")
        echo(f"    {'':34} stage    {time.time() - started:5.1f}s")

    seconds = time.time() - began
    echo(f"\n  {total} target(s) in {seconds:.1f}s"
         + (f" — {len(failures)} FAILED" if failures else ""))
    if failures:
        echo("\n  FAILURES")
        for stage, name, error in failures:
            echo(f"    {stage:16} {name:28} {error}")
    return {"failures": failures, "seconds": seconds, "ran": total}


def main(args):
    only = getattr(args, "only", None)
    only = [only] if isinstance(only, str) else only
    if only:
        unknown = sorted(set(only) - set(STAGE_NAMES))
        if unknown:
            raise SystemExit(
                f"--only: unknown stage(s) {', '.join(unknown)}. Pick from: "
                + ", ".join(STAGE_NAMES))
    result = run(only=only,
                 slug=getattr(args, "slug", None),
                 jobs=getattr(args, "jobs", None),
                 dry_run=bool(getattr(args, "dry_run", False)))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot regen`.")
