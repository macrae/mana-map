"""What Sven can reach, and what each reach depends on.

TWO TIERS, because eighteen commands' worth of argparse flags would be most of
a context window before he has read a single fact:

  TIER 1  A handful of shaped tools with small schemas, covering the questions
          actually asked in a day — where does this deck stand, what is blocking
          it, find me a card, what do the rules say, is that difference real.

  TIER 2  One generic `run_readonly`, whose description is the `CLI_READONLY`
          command names with their one-line `PILOT_STEPS` descriptions. Eighteen
          cheap lines instead of eighteen parsers, and `--help` on any of them
          returns its own flags. Self-describing, and it costs nothing to keep
          current.

**The allow-list is not re-declared here.** `serve.CLI_READONLY` and
`serve._CLI_WRITE_ATTRS` are the gates, and `serve._cli` is the dispatcher; this
module imports all three. A second copy of "which commands are safe" is a second
place to forget, and this repo has a name for that failure — ONE PREDICATE, ONE
HOME. `test_sven_tools.py` asserts the tool surface equals `CLI_READONLY`, so
adding a read-only command grants Sven the capability automatically and removing
one revokes it.

WHAT SVEN CANNOT DO, structurally rather than by instruction: anything that
writes. `_cli` refuses a command outside the allow-list, and refuses any parsed
namespace carrying `write`, `force`, `apply`, `record` or `anyway`. He does not
have a tool that edits a decklist, merges a branch or promotes a deck, and he
cannot obtain one by phrasing a request cleverly — the refusal is in the
dispatcher, not in his charter.
"""

from pathlib import Path

from manamap.config import DATA_DIR
from manamap.pilot.common import deck_dir

#: Corpus-level artifacts a command may lean on, for dependency signatures.
#: Deliberately coarse — see `cache.signature`.
_CORPUS = {
    "cards": DATA_DIR / "cards.csv",
    "roles": DATA_DIR / "card_roles.json",
    "combos": DATA_DIR / "combo_details.json",
    "rules": DATA_DIR / "rules",
    "strategy": DATA_DIR / "strategy",
    "decks": DATA_DIR / "decks",
    "pods": DATA_DIR / "pods",
}

#: command -> the corpora it reads, beyond its own deck directory. A command
#: absent from this map is assumed to read the whole `decks/` tree, which
#: over-invalidates and cannot be wrong.
_DEPS = {
    "query-rules": ("rules",), "lookup-rule": ("rules",),
    "query-strategy": ("strategy",), "lookup-strategy": ("strategy",),
    "card-search": ("cards", "roles"),
    "pool-facts": ("cards", "roles"),
    "bracket-check": ("cards", "combos"),
    "deck-audit": ("cards", "roles", "combos"),
    "mana-fit": ("cards",),
    "deck-info": (), "deck-facts": (), "deck-status": (),
    "engine-facts": (), "scenario-facts": (), "deck-history": (),
    "deck-version": (), "impact": (), "cache-status": (),
}


#: Commands whose answer is TRUE ONLY AT THE MOMENT IT IS ASKED. A turn that
#: touches one of these may never be served from the answer cache, whatever its
#: file signature says.
#:
#: The signature machinery cannot catch these, and that is the point of naming
#: them by hand:
#:   `sim-progress`  reads the mtimes of logs a running JVM rewrites continuously
#:                   — and `cache.signature` deliberately SKIPS `logs/`, so the
#:                   signature is stable while the answer changes every second.
#:   `deck-history`  shells out to git; the working tree is not in any signature.
#:   `cache-status`  reads `.agent-cache.json`, which `_SKIP_FILES` excludes.
#:
#: Each of these is excluded from the signature for a good reason, and each
#: exclusion is exactly what makes its answer uncacheable. Naming them is the
#: honest resolution — the alternative is a cache that serves "the run is 40%
#: done" an hour after it finished.
VOLATILE = frozenset({"sim-progress", "deck-history", "cache-status"})


def readonly_commands():
    """`[(name, description)]` for every command Sven may run, from the registry.

    Reads `PILOT_STEPS` rather than restating the descriptions, so a reworded
    help string reaches Sven without anyone remembering this file.
    """
    from manamap.pilot.registry import PILOT_STEPS
    from manamap.serve import CLI_READONLY

    return [(name, desc) for name, _mod, desc in PILOT_STEPS if name in CLI_READONLY]


def depends_on(argv):
    """The paths a command's answer depends on — its deck, plus its corpora.

    Over-inclusive by design. The cost of naming one file too many is a cache
    miss; the cost of naming one too few is an answer that is confidently stale,
    which is the failure this whole bench is built to refuse.
    """
    argv = [str(a) for a in (argv or [])]
    if argv and argv[0] == "pilot":
        argv = argv[1:]
    if not argv:
        return [_CORPUS["decks"]]
    name, rest = argv[0], argv[1:]
    paths = []

    slug = next((a for a in rest if not a.startswith("-")), None)
    if slug:
        candidate = deck_dir(slug)
        paths.append(candidate if Path(candidate).exists() else _CORPUS["decks"])
    else:
        paths.append(_CORPUS["decks"])

    for key in _DEPS.get(name, tuple(_CORPUS)):
        path = _CORPUS.get(key)
        if path is not None:
            paths.append(path)
    return [p for p in paths if Path(p).exists()]


def run(argv, facts=None):
    """One read-only pilot command, through the server's own gates.

    Returns `{"stdout", "exit", "cached"}`. Raises `ValueError` for anything the
    allow-list refuses — which Sven is told to report rather than work around.
    """
    from manamap import serve

    argv = [str(a) for a in (argv or [])]
    key = ("cli", tuple(argv))
    if facts is None:
        result = serve._cli(argv)
        result["cached"] = False
        return result
    before = facts.hits
    result = facts.get_or_call(key, depends_on(argv), lambda: serve._cli(argv))
    return {**result, "cached": facts.hits > before}


# ── Tier 1 — the shaped tools ─────────────────────────────────────────────
#
# Each returns plain data, never rendered text, so the caller decides how to say
# it. `promote.gate` already draws this line — pure rows out of `gate()`,
# presentation isolated in `format_gate()` — and these follow it.

def deck_state(slug, facts=None):
    """Where one deck stands: rung, unmet gates, and the derived next action."""
    from manamap.pilot import deck_info, promote

    def build():
        stage = promote.stage(slug)
        rows = promote.gate(slug, "sleeved") if stage else []
        info = deck_info.compose(slug)
        return {
            "slug": slug,
            "stage": stage,
            "gates_total": len(rows),
            "gates_met": len(rows) - len(promote.blockers(rows)),
            "blockers": [{"label": r["label"], "state": r["state"], "how": r.get("how")}
                         for r in promote.blockers(rows)],
            "next": info.get("_next"),
            "version": info.get("version"),
        }

    if facts is None:
        return build()
    return facts.get_or_call(("deck_state", slug), depends_on(["deck-info", slug]), build)


def fleet(facts=None):
    """Every deck, one row each — the answer to "what should I work on"."""
    from manamap.pilot import deck_status, promote

    def build():
        rows = []
        for row in deck_status.fleet():
            slug = row.get("slug")
            rows.append({**row, "stage": promote.stage(slug) if slug else None})
        return rows

    if facts is None:
        return build()
    return facts.get_or_call(("fleet",), [_CORPUS["decks"]], build)


def stat_test(kind, **kw):
    """A statistical answer straight out of `sim.stats`, unparaphrased.

    Sven's charter forbids him restating an interval in his own words; he calls
    this and prints what it returns. Misreading an interval is a failure this
    bench has paid for more than once, and the fix is to remove the opportunity.
    """
    from manamap.sim import stats

    fns = {
        "wilson": stats.wilson_bounds,
        "diff_proportions": stats.diff_proportions,
        "diff_means": stats.diff_means,
        "diff_medians": stats.diff_medians,
        "permutation_p": stats.permutation_p,
        "power": stats.power_for,
        "mde": stats.mde_proportion,
        "games_needed": stats.games_for_difference,
    }
    if kind not in fns:
        raise ValueError(f"stat_test: {kind!r} is not one of {sorted(fns)}")
    return {"test": kind, "result": fns[kind](**kw)}


#: The Tier-1 surface, named once so the tool schema and the tests read the same
#: list. Tier 2 is `run` plus `readonly_commands()`.
TIER1 = {
    "deck_state": deck_state,
    "fleet": fleet,
    "stat_test": stat_test,
}
