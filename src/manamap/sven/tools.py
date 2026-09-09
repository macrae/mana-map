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
    "docs": DATA_DIR / "docs_index",
    "code": DATA_DIR / "code_index",
}

#: command -> the corpora it reads, beyond its own deck directory. A command
#: absent from this map is assumed to read the whole `decks/` tree, which
#: over-invalidates and cannot be wrong.
_DEPS = {
    "query-rules": ("rules",), "lookup-rule": ("rules",),
    "query-strategy": ("strategy",), "lookup-strategy": ("strategy",),
    # The docs and code indexes are rebuilt by hand, so they move rarely — but
    # naming them means an answer citing a passage invalidates when that passage
    # is re-indexed, rather than quoting a chunk id that no longer exists.
    "query-docs": ("docs",), "lookup-doc": ("docs",), "query-code": ("code",),
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
        # `deck_dir` RAISES for an unknown slug rather than returning a path
        # that does not exist — which is right for a command and wrong here,
        # where this is only working out what to hash. It raised before
        # `deck_state` could resolve "zur" to "zur-enchantress", so a
        # dependency calculation decided a deck did not exist.
        try:
            candidate = deck_dir(slug)
        except Exception:                      # noqa: BLE001
            candidate = None
        paths.append(candidate if candidate and Path(candidate).exists()
                     else _CORPUS["decks"])
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

def resolve_slug(name):
    """`"zur"` -> `"zur-enchantress"`. Raises with candidates when it cannot tell.

    THE PILOT SAYS "ZUR", NOT "ZUR-ENCHANTRESS", and so does anyone repeating
    the pilot's words back. The first time Sven reached this tool he passed the
    slug straight out of the question, got a FileNotFoundError, and concluded
    the deck did not exist — offering to create one. A fluent, helpful, wrong
    answer, produced by a lookup that could only do exact matches.

    Exact wins, then a unique prefix, then a unique substring. AMBIGUITY IS AN
    ERROR, never a guess: two decks starting "sh" must produce a question, not a
    coin flip, because the wrong deck's figures are indistinguishable from the
    right deck's until someone notices they describe another list. The same
    reasoning `retrieve.fetch` uses for a rule id, which suggests and refuses
    rather than falling back to semantic search.
    """
    from manamap.config import DECKS_DIR

    name = (name or "").strip()
    known = sorted(d.name for d in DECKS_DIR.iterdir()
                   if (d / "decklist.txt").exists()) if DECKS_DIR.is_dir() else []
    if name in known:
        return name
    for match in (
        [k for k in known if k.startswith(name)],
        [k for k in known if name.lower() in k.lower()],
    ):
        if len(match) == 1:
            return match[0]
        if len(match) > 1:
            raise ValueError(
                f"{name!r} matches {len(match)} decks: {', '.join(match)}. "
                f"Say which one.")
    raise ValueError(
        f"No deck matches {name!r}. On the bench: {', '.join(known)}")


def deck_state(slug, facts=None):
    """Where one deck stands: rung, unmet gates, and the derived next action."""
    from manamap.pilot import deck_info, promote

    slug = resolve_slug(slug)

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
            # THE MEASUREMENTS, not just the gate that says they exist.
            #
            # Asked "is heliod's win rate actually good", Sven checked four
            # commands and concluded the deck "has no meaningful game record" —
            # while this very tool was telling him the full-sim-batch gate was
            # MET. It reported that a measurement EXISTED and never what it SAID,
            # so the only honest thing he could do with it was go looking, and
            # what he found was a one-game version history from July.
            #
            # `simulation` carries `stale` and `ran_on_decklist_sha256`, which
            # is the fact the charter requires him to state before quoting any
            # Forge figure — heliod's run describes v1.0.0 and the sleeved list
            # is v1.2.1. A rule the tools make unfollowable is not a rule.
            "simulation": _sim_with_comparison(info.get("simulation"), slug),
            "goldfish": info.get("goldfish"),
            "band": _band(slug),
            "record": info.get("record"),
            # ORIENTATION. Sven did not know a deck's own COMMANDER, which is
            # why he could not notice that six of zur's eight Forge runs were
            # played by the wrong one. A tool that describes a deck without
            # naming it leaves the model with nothing to check against.
            "commander": info.get("commander"),
            "colour_identity": info.get("colour_identity"),
            "size": info.get("size"),
            "lands": info.get("lands"),
            "bracket": info.get("bracket"),
            "engine_health": info.get("engine_health"),
            "branches": _branch_summary(info.get("branches")),
            "open_questions": info.get("open_questions"),
            # THE ANTI-INVENTION CONTRACT. Every gap in a tool result is a place
            # the model will narrate from nothing: asked what was next for zur it
            # wrote "you died by turn 5-6", where the record says median 34. It
            # had elimination data in view nowhere and filled the hole.
            #
            # So the payload says what it does NOT carry, and which command does.
            # Absent means absent — the bench's oldest rule, applied to a tool's
            # own coverage rather than to a figure.
            "not_included": {
                "elimination timing and who killed us":
                    "run_command simulate <slug> --analyze <run-id>, or read "
                    "`simulation.eliminated_by` where present — do NOT estimate it",
                "the decklist itself": "run_command deck-facts <slug>",
                "per-card analysis": "run_command deck-audit <slug>",
                "the captain's log": "run_command deck-history <slug>",
                "why a rule works": "run_command query-rules \"...\"",
            },
        }

    if facts is None:
        return build()
    return facts.get_or_call(("deck_state", slug), depends_on(["deck-info", slug]), build)


def _branch_summary(branches):
    """Counts and names, not the whole branch documents.

    Seventeen open branches on zur is a fact about where the work is; seventeen
    branch documents is most of a context window.
    """
    if not branches:
        return None
    open_ = [b.get("branch") for b in branches if b.get("state") != "MERGED"]
    return {"total": len(branches), "open": len(open_), "names": open_[:12]}


def _band(slug):
    """The goldfish's ceiling/floor pair, when the deck declares an ability.

    A DECK WITH A BAND HAS NO SINGLE KILL NUMBER. zur declares
    `model_commander_animate`; the model fires it every turn it can afford and
    Forge's AI fired it in 5% of games, so `kill_by_8` is 0.381 at the ceiling
    and 0.219 at the floor. TWENTY-FOUR zur branches were graded on the ceiling
    before anyone measured the floor.

    Quoting either end alone is the documented mistake, so the pair travels
    together with a sentence saying so.
    """
    from manamap.pilot.common import deck_file, load_json

    doc = load_json(deck_file(slug, "goldfish_metrics.json")) or {}
    band = doc.get("commander_ability_band")
    if not band or not band.get("rows"):
        return None
    return {**band,
            "reading": (
                "THIS DECK HAS NO SINGLE KILL NUMBER. It declares "
                f"{', '.join(band['abilities'])}, which the goldfish fires every "
                "turn it can afford and Forge's AI fires far less often. Quote "
                "the PAIR — ceiling and floor — never one end. Twenty-four "
                "branches were graded on the ceiling before the floor existed.")}


def _sim_with_comparison(sim, slug=None):
    """The win rate, WITH the two comparisons already computed.

    THE MODEL MUST NOT DO THIS ARITHMETIC, and telling it so in a charter did not
    work. Asked whether heliod's 25.2% was good, it answered "well below
    functional in a four-player pod" — where par is 25% and the table's measured
    null is 14.4%, so the deck is at par and above the null. It reasoned about
    rates in prose because the tool handed it a bare number and an instruction.

    So the tool hands it the answer instead. `par` is 1/seats, the rate a deck
    that does nothing would post. `null` is what THIS bench's decks actually
    score in seat 0 at THIS table, which is the more honest comparison and is
    already measured. Both come back as Newcombe intervals ON THE DIFFERENCE,
    from `stats.diff_proportions` — never two marginal intervals side by side,
    which imply nothing.

    This is the repo's own shape: `promote.gate` returns rows and
    `power.preflight` returns lines, both leaving nothing to re-derive.
    """
    if not sim or not sim.get("games"):
        return sim
    from manamap.sim import stats

    out = dict(sim)
    # WINS AND DECIDED ARE READ, NEVER RECONSTRUCTED.
    #
    # This computed `round(win_rate * games)`, and the two have DIFFERENT
    # DENOMINATORS: `win_rate` is over decided games, `games` is the total.
    # heliod's run is 20 wins in 100 decided out of 120 played — 21 clocked out
    # and a clock-out has no winner, which the same payload says two fields
    # earlier. The product was 24: a count belonging to neither.
    #
    # Every comparison this function printed used it. The engine critic caught
    # it by reproducing the arithmetic and finding it matched nothing in the
    # record. A rate is not a count, and a count rebuilt from a rate is a guess
    # wearing the rate's authority.
    wins, decided = sim.get("wins"), sim.get("decided")
    if wins is None or not decided:
        # Absent means absent. An older `info.json` predates these fields, and a
        # comparison computed from a guess is worse than one not offered.
        out["comparisons_unavailable"] = (
            "this run record predates `wins`/`decided` — regenerate with "
            "`deck-info <slug> --write`. A comparison is not computed from a "
            "rate alone.")
        return out
    seats = len(sim.get("vs") or []) + 1
    comparisons = {}

    if seats > 1:
        par = 1.0 / seats
        comparisons["vs_par"] = {
            "par": round(par, 4),
            "what": f"an equal share of a {seats}-player pod",
            **(stats.diff_proportions(round(par * decided), decided,
                                      wins, decided) or {}),
        }
    null = _null_rate(sim)
    if null is not None:
        comparisons["vs_null"] = {
            "null": null,
            "what": "what this bench's decks score in seat 0 at this table",
            **(stats.diff_proportions(round(null * decided), decided,
                                      wins, decided) or {}),
        }
    for key, block in comparisons.items():
        block["reading"] = _reading(key, block)
    if comparisons:
        out["comparisons"] = comparisons
    if slug:
        out.update(_commander_check(slug, sim))
    return out


def _commander_check(slug, sim):
    """Did Forge play the commander this deck is built around?

    SIX OF ZUR'S EIGHT RUNS DID NOT. They were piloted by `Zur the Enchanter`
    while the deck is built on `Zur, Eternal Schemer` — a different card with
    different abilities — and the records are indistinguishable from the good
    ones at a glance. `docs/gotchas-bench.md` documents the incident; nothing
    detected it, and nothing would have, because the played commander was in the
    run record and the declared one was never in the same view.

    Putting both in one payload is the entire fix. A mismatch is not a
    subtlety to notice — it is a field that says the figure describes a
    different deck.
    """
    from manamap.pilot.common import deck_file, load_json
    from manamap.sim import forge

    declared = None
    cards = load_json(deck_file(slug, "cards.json")) or {}
    for card in cards.get("cards", []):
        if card.get("is_commander"):
            declared = card.get("name")
            break

    played, bad = None, []
    try:
        # `list_runs` returns the RECORDS themselves, newest last — the same
        # documents on disk, so no second read is needed.
        runs = forge.list_runs(slug) or []
        for doc in runs:
            names = (doc.get("seats") or [{}])[0].get("commander") or []
            name = names[0] if names else None
            if declared and name and not _same_commander(name, declared):
                bad.append({"run": (doc.get("run_id") or "")[:60],
                            "played": name,
                            "games": doc.get("games_completed")})
        if sim and sim.get("latest"):
            for doc in runs:
                if sim["latest"].startswith(doc.get("run_id") or "\0"):
                    names = (doc.get("seats") or [{}])[0].get("commander") or []
                    played = names[0] if names else None
                    break
        if played is None and runs:
            names = (runs[-1].get("seats") or [{}])[0].get("commander") or []
            played = names[0] if names else None
    except Exception:                              # noqa: BLE001
        return {}

    out = {"commander_declared": declared, "commander_forge_played": played}
    if declared and played and not _same_commander(played, declared):
        out["WARNING"] = (
            f"THIS RUN WAS PILOTED BY {played!r}, NOT {declared!r}. Every figure "
            f"from it describes a different deck. Say so before quoting any of "
            f"it, and do not compare it against a run that used the right one.")
    if bad:
        out["runs_with_the_wrong_commander"] = bad
        out["runs_warning"] = (
            f"{len(bad)} stored run(s) were piloted by a commander this deck is "
            f"not built around. They are indistinguishable from valid runs at a "
            f"glance and must never be mixed into a comparison.")
    return out


def _same_commander(a, b):
    """Front-face comparison, so a DFC's two names do not read as two decks."""
    norm = lambda n: (n or "").split(" // ")[0].strip().lower()
    return norm(a) == norm(b)


def _reading(key, block):
    """The comparison as a SENTENCE TO QUOTE, not a field to interpret.

    Handed `excludes_zero: false` and an interval, the model wrote "the interval
    is [-10.9%, +10.9%] — it excludes zero and so cannot be called resolved."
    The conclusion was right and the reason was the exact inverse of the truth,
    which is worse than being wrong outright: a reader skimming for "excludes
    zero" takes away the opposite of what the sample says.

    So it does not get a boolean to narrate. `power.preflight` already returns
    lines rather than numbers, and `simulation.piloting` already carries a
    `reading` — this is that pattern, applied to the figure this bench
    misreads most.
    """
    against = {"vs_par": "par for the pod", "vs_null": "this table's null"}[key]
    lo, hi = block.get("ci95") or (None, None)
    if lo is None:
        return f"not comparable against {against} — no interval"
    span = f"[{lo:+.3f}, {hi:+.3f}]"
    if not block.get("excludes_zero"):
        return (f"INDISTINGUISHABLE from {against}: the interval on the "
                f"difference is {span}, which SPANS ZERO. This sample cannot "
                f"tell them apart — that is not the same as saying they are "
                f"equal, and not the same as saying the deck is bad.")
    direction = "ABOVE" if block["diff"] > 0 else "BELOW"
    return (f"{direction} {against} by {block['diff']:+.3f}, interval {span}, "
            f"which EXCLUDES ZERO — a real difference at this sample size.")


def _null_rate(sim):
    """This table's measured null, or None. Absent is absent."""
    try:
        from manamap.sim import pods

        for name in (sim.get("pod"), "standard"):
            if not name:
                continue
            cal = pods.calibration(name)
            rate = ((cal or {}).get("subject_null") or {}).get("rate")
            if rate is not None:
                return rate
    except Exception:                              # noqa: BLE001
        pass
    return None


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

#: A tool result longer than this is truncated. One careless `card-search
#: --limit 500` would otherwise be the whole turn, and the model would have
#: spent its context on rows nobody asked for. The truncation SAYS SO and says
#: how to narrow, so it reads as a boundary rather than as the end of the data.
TOOL_RESULT_CAP = 8000

_STAT_SIGNATURES = """  wilson(k, n)                          a rate's 95% interval
  diff_proportions(k_a, n_a, k_b, n_b)  Newcombe: the interval ON THE DIFFERENCE
  diff_means(xs, ys)                    Welch
  diff_medians(xs, ys)                  bootstrap, seeded
  permutation_p(xs, ys)                 two-sided, seeded, 10k iterations
  power(p_a, p_b, n_a, n_b)             exact, not a normal approximation
  mde(p_a, n_a)                         smallest effect this n could detect
  games_needed(p_a, difference)         n per arm for 80% power"""


def tool_block():
    """Sven's tool list — rendered from `sven.api`, the single registry.

    This function used to BUILD the list, and `mcp_server._tool_defs` built a
    second one by hand. They drifted: the same capability was `run_command` here
    and `run_readonly` there, `stats` here and `stat_test` there, and two tools
    existed on one side only — so a question Claude Code could answer, Sven
    could not.
    """
    from manamap.sven import api

    return api.anthropic_block()


_STAT_FNS = ("wilson", "diff_proportions", "diff_means", "diff_medians",
             "permutation_p", "power", "mde", "games_needed")


def command_help(name):
    """One command's `--help`, for progressive disclosure."""
    import argparse
    import contextlib
    import io

    from manamap.pilot.registry import add_pilot_parser
    from manamap.serve import CLI_READONLY

    if name not in CLI_READONLY:
        raise ValueError(f"command_help: {name!r} is not a command Sven may run")
    parser = argparse.ArgumentParser(prog="manamap")
    add_pilot_parser(parser.add_subparsers(dest="command"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        parser.parse_args(["pilot", name, "--help"])
    return buf.getvalue()
