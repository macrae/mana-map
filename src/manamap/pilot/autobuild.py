"""`manamap pilot build` — a brief in, a measured 100-card list out. PRD §5.

    mm build zur-voltron --commander "Zur the Enchanter" \\
        --brief "esper enchantment tempo, bracket 3, board by t4-5"

THE PROGRAM IS GLUE, AND THIS IS THE GLUE. Every stage below already existed as
its own subcommand; what did not exist was one command that ran them in order,
said how far along it was, and reported what it could not do. The PRD's account
of the gap — "the pieces to automate it exist … but nothing composes them into a
single command" — was exactly right about the gap and generous about its size.

SIX STAGES, NOT THE PRD'S SEVEN, and the difference is honesty rather than
scope. §5 lists populate / balance / verify as three; `build_deck.build` does all
three inside one deterministic call and cannot report between them. Printing
three bars for one function would be theatre. They are one stage here and three
sections in the report, which is where the distinction is actually useful.

    [1/6] intent    brief.json — the commander, the kept cards, the style
    [2/6] anchor    the commander confirmed, or three proposed and a halt
    [3/6] build     populate + balance + verify (build_deck.build)
    [4/6] resolve   decklist.txt -> cards.json, with printings
    [5/6] measure   the goldfish and the standard benchmark
    [6/6] land      the manifest, the report, and the commands to commit

WHAT "MEASURE" MEANS HERE, AND WHY IT IS NOT FORGE. §5's stage 6 asks for a
"short simulation batch" inside a dev build. Measured before this was written: a
Forge game runs ~100s at the median on four parallel jobs, so a twelve-minute
build buys 15-20 games — against a minimum detectable difference of 42 points at
n=20 (`sim.stats.mde_proportion`, at the 0.18 baseline the fleet actually reads).
A dev-stage Forge batch cannot resolve any change a build would ever make. The
goldfish runs 10,000 seeded games in about four seconds and is byte-deterministic,
and `benchmark` adds ~2.3s under a frozen harness. So the dev batch is those two,
it costs about six seconds, and Forge belongs at the staging gate where 400 games
gives an MDE of 8.5. The report says so out loud rather than leaving a reader to
assume a win rate was measured.

A BRIEF'S FREE TEXT DRIVES TWO THINGS AND IS OTHERWISE INERT — see `_read_brief`.
Saying which two is the whole point; a description that silently changes nothing
is worse than no description field, because it reads exactly like one that works.
"""

import json
import re
import subprocess
import time
from types import SimpleNamespace

from manamap import console
from manamap.config import BRACKET_DEFAULT, BRACKET_MAX, DECKS_DIR
from manamap.pilot import build_deck
from manamap.pilot.common import deck_dir

#: The stage names, in order. One list so the bar, the report and the docstring
#: cannot disagree about how many there are or what they are called.
STAGES = ("intent", "anchor", "build", "resolve", "measure", "land")

#: Flags that SHAPE the brief. If `brief.json` already exists, passing any of
#: these is a refusal rather than an overwrite: `scaffold_brief` refuses for the
#: same reason, and a rebuild that silently ignored `--bracket 4` would be the
#: worst of the three available behaviours.
BRIEF_FLAGS = ("commander", "theme", "bracket", "library", "from_file", "brief")

#: `bracket 3`, `b3`, `bracket-3` — the one number a description can carry
#: unambiguously. Nothing else in free text is parsed; see `_read_brief`.
_BRACKET_RE = re.compile(r"\bb(?:racket)?[\s-]*([1-5])\b", re.I)

#: Words that match every archetype and therefore distinguish none of them. A
#: brief saying "an enchantment deck" must not match a style named "Decks".
_STOPWORDS = frozenset(
    "and the of to for with deck decks build builds list lists that this "
    "its it is are be into on at by or if we my me you your commander edh "
    "magic card cards game games turn turns play plays playing want wants "
    "should would could then than but not only just very more most".split())


class BuildError(RuntimeError):
    """A build that cannot proceed, with a sentence saying what to do next."""


def _words(text):
    """Content words, lowercased. The unit both sides of a theme match use."""
    return {w for w in re.findall(r"[a-z]+", (text or "").lower())
            if len(w) > 2 and w not in _STOPWORDS}


def match_theme(brief_text, themes):
    """Resolve a description to one of the commander's REAL styles, or None.

    §7.2's rule is that styles are presented side by side and never ranked, and
    this does not rank them: it resolves what the pilot already wrote against the
    list EDHREC returns, exactly as `--theme voltron` does. What it must not do
    is GUESS — a description overlapping nothing returns None, and the flat
    provisional budget is used with a reason, which is `role_budget_for`'s
    existing behaviour and is readable in `role_budget_grounding`.

    Ties break on EDHREC's own order, which `list_themes` preserves, because
    inventing a tiebreak here would BE the ranking §7.2 forbids.
    """
    words = _words(brief_text)
    if not words:
        return None
    best, best_key = None, None
    for i, theme in enumerate(themes):
        overlap = words & (_words(theme.get("name")) | _words(theme.get("slug")))
        if not overlap:
            continue
        # Bigger overlap wins; earlier in EDHREC's order breaks the tie.
        key = (len(overlap), -i)
        if best_key is None or key > best_key:
            best, best_key = theme, key
    return best


def _themes_for(commander):
    """The commander's styles, or an empty list if EDHREC is unreachable.

    A build must not fail because a theme lookup did not answer. The cost of an
    empty list is the flat role budget with its grounding string saying so.
    """
    try:
        from manamap.pilot import archetypes

        return archetypes.list_themes(commander)
    except Exception:                            # network, cache, or shape
        return []


def _min_decks():
    """`archetypes.MIN_DECKS_FOR_TEMPLATE`, imported rather than retyped.

    That module owns the number and the reasoning behind it. Lazy for the same
    reason every import in the registry is lazy: `--help` should not pay for it.
    """
    from manamap.pilot.archetypes import MIN_DECKS_FOR_TEMPLATE

    return MIN_DECKS_FOR_TEMPLATE


def _named_flags(passed):
    """`--commander`, `--from`, … for the refusal message. Names as typed."""
    spelling = {"from_file": "--from", "brief": "--brief"}
    return ", ".join(spelling.get(f, "--" + f) for f in passed)


def _read_brief(args):
    """Stage 1. Resolve the inputs into a `brief.json` on disk.

    TWO THINGS ARE READ OUT OF `--brief` FREE TEXT, AND BOTH ARE RESOLVED
    AGAINST REAL DATA rather than interpreted:

      * the **bracket**, when the text says `bracket 3` (`_BRACKET_RE`);
      * the **style**, when the text's words overlap one of the commander's own
        EDHREC archetypes (`match_theme`).

    EVERYTHING ELSE IS STORED AND CONSUMED BY NOTHING. `build_deck` reads
    `commander`, `bracket`, `must_include`, `must_exclude`, `pool`/`pool_files`
    and `theme`; the prose keys people actually write — `playstyle`, `notes`,
    `design_rules`, `win_conditions` — are echoed into `info.json` and read by no
    code. That is a real limit, and `format_report` states it on every build,
    because a description field that quietly does nothing looks exactly like one
    that works.
    """
    slug = args.slug
    path = DECKS_DIR / slug / "brief.json"

    if path.exists():
        passed = [f for f in BRIEF_FLAGS if getattr(args, f, None)]
        if passed:
            raise BuildError(
                f"{path} already exists, and {_named_flags(passed)} would "
                f"change it. Edit the file and re-run without those flags, or "
                f"pick another slug — a brief is authored, and a build that "
                f"silently overwrote one would lose the edits it was given.")
        return path, json.loads(path.read_text(encoding="utf-8")), None

    library = list(getattr(args, "library", None) or [])
    commander = getattr(args, "commander", None)
    if getattr(args, "from_file", None):
        from manamap.pilot.brew import _library_from_file

        found, from_commander = _library_from_file(args.from_file)
        library.extend(n for n in found if n not in library)
        commander = commander or from_commander

    if not commander:
        return None, None, library           # stage 2 proposes, then halts

    text = getattr(args, "brief", None)
    bracket = args.bracket
    if bracket is None and text:
        hit = _BRACKET_RE.search(text)
        bracket = int(hit.group(1)) if hit else None

    theme, matched = args.theme, None
    if not theme and text:
        matched = match_theme(text, _themes_for(commander))
        theme = matched["slug"] if matched else None

    path, brief = build_deck.scaffold_brief(
        slug, commander, library=library, theme=theme,
        bracket=bracket or BRACKET_DEFAULT)
    if text:
        # Stored, not consumed. `notes` is the key `deck_info` already surfaces,
        # and it is what §5's stage 7 means by "the brief stored as the version
        # message" — the sentence survives, in the deck's own file.
        brief["notes"] = text
        path.write_text(json.dumps(brief, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    # The count rides out so the report can say what the style is made of. A
    # histogram over 29 decks is a description of 29 decks, and `role_budget_for`
    # applies one exactly as readily as it applies the 1201-deck one —
    # `MIN_DECKS_FOR_TEMPLATE` is read by the `archetypes` REPORT and by nothing
    # in the build path. Naming it beats gating on it: the pilot wrote the word.
    return path, brief, matched


def propose_commanders(library, limit=3):
    """Stage 2 with no commander: rank real commanders near the kept cards.

    A-1 asks for three with a one-line rationale each, and a halt. The rationale
    is `shared` — the cards this commander's own reference deck has in common
    with the pile — which is a fact about two lists rather than a claim about a
    strategy, and is the only rationale available without an agent.
    """
    from manamap.analysis import commander_search as cs

    if not library:
        raise BuildError(
            "no commander, and nothing to infer one from — pass --commander, "
            "or --library/--from with the cards you kept")
    return cs.search(library, limit=max(limit, 10))


def _unowned(plan):
    """Names in the plan that no box in the collection holds. A-2's count.

    OWNERSHIP MEANS A BOX. `collection` is deliberately the only reader of
    `COLLECTION_DIR` and deliberately does not count deck membership, because a
    card sleeved in another deck is not one you can put in this deck without
    taking that one apart. An empty collection returns nothing rather than
    reporting all ninety-nine as unowned.
    """
    from manamap.pilot import collection

    owned = collection.owned_names()
    if not owned:
        return set()
    names = [s["name"] for s in plan.get("slots", [])]
    return {n for n in names if n not in owned}


def flagged(plan, brief, theme_decks=None):
    """What the build could not do, as sentences. A-1's last acceptance clause.

    "Any category the builder could not fill to depth is flagged explicitly
    rather than silently padded." Every figure here was already computed by
    `build_deck.build` and was, until now, readable only by opening the plan.
    """
    flags = []

    for group, row in sorted((plan.get("role_budget_deviation") or {}).items()):
        short = row["target"] - row["actual"]
        if short > 0:
            flags.append(f"{group} at depth {row['actual']}, budget wants "
                         f"{row['target']}")

    illegal = plan.get("must_include_illegal") or []
    if illegal:
        named = ", ".join(c["name"] if isinstance(c, dict) else str(c)
                          for c in illegal)
        flags.append(f"{len(illegal)} must-include outside the colour identity, "
                     f"dropped: {named}")

    short = (plan.get("manabase") or {}).get("shortfalls") or {}
    if short:
        flags.append("mana base short of its source target: "
                     + ", ".join(f"{c} by {n}" for c, n in sorted(short.items())))

    cut = plan.get("cut_for_bracket") or []
    if cut:
        flags.append(f"{len(cut)} cut to reach bracket "
                     f"{plan['bracket']['target']}: "
                     + ", ".join(c["name"] for c in cut))

    unowned = _unowned(plan)
    if unowned:
        shown = sorted(unowned)
        flags.append(f"{len(shown)} of the nonland cards are in no box — "
                     + ", ".join(shown[:6])
                     + (f" (+{len(shown) - 6})" if len(shown) > 6 else ""))

    if not brief.get("theme"):
        flags.append("no style resolved, so the role budget is the flat "
                     "provisional one — `manamap pilot archetypes "
                     "\"<commander>\"` lists the real ones")
    elif theme_decks is not None and theme_decks < _min_decks():
        # `archetypes.MIN_DECKS_FOR_TEMPLATE` is read by that module's REPORT
        # and by nothing in the build path, so `role_budget_for` shapes a budget
        # from a 29-deck histogram exactly as readily as from a 1201-deck one.
        # Flagged rather than refused: the pilot wrote the word, and a histogram
        # over 29 decks is a description of 29 decks rather than an error.
        flags.append(f"style \"{brief['theme']}\" has only {theme_decks} decks "
                     f"behind it — its role budget describes those {theme_decks}, "
                     f"not the archetype. --theme picks a different one")
    return flags


def composition(slug, plan):
    """Curve, category depth and pip-to-source. A-1's composition report.

    The curve comes from `cards.json` rather than the plan, because a plan slot
    carries no mana value and because COPIES ARE THE UNIT — `expand_copies` is
    the shared reader that stops thirty basics counting as one land, which is
    the defect that once published "18 lands" for a 33-land deck.
    """
    from manamap.pilot.common import expand_copies, front_field, is_land, load_deck_cards

    # `load_deck_cards` returns the whole document, not the list — the first cut
    # passed the dict straight to `expand_copies` and swallowed the resulting
    # AttributeError in a bare `except`, reporting an empty curve rather than a
    # bug. Not caught at all now: stage 4 wrote this file seconds earlier, so a
    # failure here is a real one and should say so.
    curve, nonlands = {}, 0
    for card in expand_copies(load_deck_cards(slug)["cards"]):
        if is_land(card):
            continue
        nonlands += 1
        mv = int(front_field(card, "cmc", 0) or 0)
        curve[mv] = curve.get(mv, 0) + 1
    total_mv = sum(mv * n for mv, n in curve.items())

    mana = plan.get("manabase") or {}
    return {
        "curve": dict(sorted(curve.items())),
        "nonlands": nonlands,
        "mean_mana_value": round(total_mv / nonlands, 2) if nonlands else None,
        "depth": dict(sorted((plan.get("role_budget") or {}).items())),
        "sources": mana.get("sources") or {},
        "source_targets": mana.get("source_targets") or {},
        "requirements": {c: r.get("total_pips")
                         for c, r in (mana.get("requirements") or {}).items()},
        "on_curve_probability": mana.get("on_curve_probability") or {},
    }


def _run(module, **kwargs):
    """Call another subcommand's `main` with an argparse-shaped namespace."""
    import importlib

    importlib.import_module(module).main(SimpleNamespace(**kwargs))


def _headline(slug, bench):
    """The figures §5's mock puts under the build, each from its own artifact."""
    out = {}
    gold = deck_dir(slug) / "goldfish_metrics.json"
    if gold.exists():
        doc = json.loads(gold.read_text(encoding="utf-8"))
        metrics = doc.get("metrics") or {}
        cmd = metrics.get("commander") or {}
        out["commander_by_t6"] = cmd.get("cast_by_turn_6_rate")
        out["mean_cast_turn"] = cmd.get("mean_cast_turn")
        out["targets"] = [
            {"label": t.get("label"), "by_turn_6": t.get("by_turn_6_rate")}
            for t in metrics.get("targets") or []]
    # `metrics`, not `measures`. The first cut read a key that does not exist
    # and reported an empty dict — the silent-empty failure this repo keeps
    # catching, and the reason the smoke build printed no mana figures at all.
    m = bench.get("metrics") or {}
    out["mana_screw"] = m.get("mana_screw") or {}
    out["response"] = m.get("response") or {}
    return out


def build(args, quiet=False):
    """Run the six stages. Returns the record; writes every artifact."""
    import contextlib
    import io

    slug = args.slug
    started = time.time()
    record = {"slug": slug, "stages": [], "flagged": []}

    # Each stage's own command narrates to stdout. The bar is the narration
    # here, so theirs is captured — and captured rather than silenced, so a
    # traceback still carries whatever the failing command had already said.
    said = io.StringIO()
    # Whether this deck already existed decides whether the build may set its
    # stage: rebuilding a deck the pilot has moved to the bench must not put it
    # back in dev.
    new_deck_existed = (deck_dir(slug) / "deck_versions.json").exists() \
        if (DECKS_DIR / slug).is_dir() else False

    with console.task(f"Building {slug}", total=len(STAGES), unit="stages") as bar:
        bar.state(STAGES[0])
        # The third value is the kept cards when there is no commander yet, and
        # the STYLE the description matched when there is. One return, two
        # shapes, because the halting branch never reaches a style.
        path, brief, extra = _read_brief(args)
        if brief is None:
            bar.state(STAGES[1])
            record["proposed"] = propose_commanders(extra)["results"][:3]
            record["stages"] = list(STAGES[:2])
            return record
        record["brief"] = str(path)
        bar.advance(1, state=STAGES[1])

        record["commander"] = brief["commander"]
        record["theme"] = brief.get("theme")
        record["theme_decks"] = (extra or {}).get("decks")
        bar.advance(1, state=STAGES[2])

        # [3/6] populate, balance and verify, in one deterministic call.
        # `enforce_bracket` REFUSES rather than shipping an overage: a bracket
        # floor is what the contents are consistent with, and a build handing
        # back an illegal-for-the-table list behind a flag would be trusting the
        # reader to notice.
        with contextlib.redirect_stdout(said):
            _run("manamap.pilot.build_deck", slug=slug, write_decklist=True,
                 space=getattr(args, "space", None))
        plan = json.loads((deck_dir(slug) / "build_plan.json")
                          .read_text(encoding="utf-8"))
        record["plan"] = {k: plan[k] for k in
                          ("commander", "color_identity", "bracket",
                           "land_counts")}
        bar.advance(1, state=STAGES[3])

        with contextlib.redirect_stdout(said):
            _run("manamap.pilot.fetch_deck", slug=slug, branch=None)
        bar.advance(1, state=STAGES[4])

        # [5/6] the dev batch — see the module docstring for why it is not Forge.
        from manamap.pilot import benchmark

        with contextlib.redirect_stdout(said):
            _run("manamap.pilot.goldfish", slug=slug, branch=None)
            _, bench = benchmark.write(slug)
        record["measured"] = _headline(slug, bench)
        bar.advance(1, state=STAGES[5])

        with contextlib.redirect_stdout(said):
            _run("manamap.pilot.build_index")
            # A NEW DECK LANDS IN DEV. PRD §3: "most decks here will be thrown
            # away — the environment is optimised for throughput, not rigour."
            # Only a build says this; nothing infers `dev` for the ten decks
            # that predate the ladder, because nobody has said they are brews.
            from manamap.pilot import promote as _promote

            if not new_deck_existed:
                _promote.set_stage(slug, _promote.DEV)
        record["stage"] = _promote.stage(slug)
        record["composition"] = composition(slug, plan)
        record["flagged"] = flagged(plan, brief, record.get("theme_decks"))
        bar.advance(1)

    record["stages"] = list(STAGES)
    record["seconds"] = round(time.time() - started, 1)
    return record


def format_report(record):
    """The build, as the pilot reads it."""
    lines = []
    slug = record["slug"]

    if "proposed" in record:
        lines.append("\nNo commander given. Three from the cards you kept:\n")
        for i, c in enumerate(record["proposed"], 1):
            shared = ", ".join(c.get("shared") or [])[:80] or "no shared cards"
            lines.append(f"  {i}. {c['commander']}   ({c['score']:.3f})")
            lines.append(f"       shares: {shared}")
        lines.append("")
        lines.append("  Proximity is a discovery aid, not a verdict — the "
                     "embedding is built on")
        lines.append("  oracle text, and text is fuzzy. Re-run with "
                     "--commander \"<name>\".")
        return "\n".join(lines)

    plan = record["plan"]
    b = plan["bracket"]
    lands = sum(plan["land_counts"].values())
    lines.append(f"\nBUILT {slug} in {record['seconds']}s — {plan['commander']}, "
                 f"{''.join(plan['color_identity']) or 'C'}")
    lines.append(f"  bracket {b['target']} ({b['target_name']}), computed floor "
                 f"{b['computed_floor']}")
    style = record.get("theme") or "none resolved"
    if record.get("theme_decks"):
        style += f" ({record['theme_decks']} decks)"
    lines.append(f"  {lands} lands · style {style}"
                 + (f" · {record['stage'].upper()}" if record.get("stage") else ""))

    # A-1's composition report: category depth, curve, land count, and
    # pip-to-source coverage per colour.
    comp = record.get("composition") or {}
    if comp.get("curve"):
        curve = " ".join(f"{mv}:{n}" for mv, n in comp["curve"].items())
        lines.append(f"  curve  {curve}   mean mv {comp['mean_mana_value']}")
    depth = comp.get("depth") or {}
    if depth:
        shown = " ".join(f"{g} {n}" for g, n in depth.items() if g != "lands")
        lines.append(f"  depth  {shown}")
    sources, targets = comp.get("sources") or {}, comp.get("source_targets") or {}
    if sources:
        pips = comp.get("requirements") or {}
        cells = []
        for colour in sorted(sources):
            want = targets.get(colour)
            cells.append(f"{colour} {sources[colour]}/{want}"
                         + (f" ({pips[colour]:g}p)" if colour in pips else ""))
        lines.append("  sources  " + "   ".join(cells)
                     + "     (have/Karsten target, pips in the 99)")

    m = record.get("measured") or {}
    if m.get("commander_by_t6") is not None:
        lines.append(f"  commander by t6    {m['commander_by_t6']:.0%}"
                     f"   (mean cast turn {m.get('mean_cast_turn')})")
    for t in m.get("targets") or []:
        if t.get("by_turn_6") is not None:
            lines.append(f"  {str(t['label'])[:17]:<17}  {t['by_turn_6']:.0%} by t6")
    screw = m.get("mana_screw") or {}
    if screw:
        lines.append(f"  missed land drops  {screw.get('missed_land_drop_rate', 0):.0%}"
                     f"   mulligan {screw.get('mulligan_rate', 0):.0%}"
                     f"   mana at t5 {screw.get('mana_at_turn_five')}")
    resp = m.get("response") or {}
    if resp:
        lines.append(f"  interaction  {resp.get('answer_cards')} cards, "
                     f"{resp.get('classes_covered')}/{resp.get('classes_possible')} "
                     f"permanent classes answered")

    if record["flagged"]:
        lines.append("")
        for flag in record["flagged"]:
            lines.append(f"  flagged  {flag}")

    lines += [
        "",
        "  Measured by the goldfish and the benchmark — 10,000 seeded solitaire",
        "  games and a frozen four-measure harness. NO POD AND NO INTERACTION, so",
        "  no win rate: a dev-budget Forge batch is ~20 games, which resolves",
        "  nothing (MDE 42 points against this fleet's baseline). The pod is the",
        "  staging gate — `manamap pilot simulate`.",
        "",
        "  A brief's prose is stored, not read. The builder consumes commander,",
        "  bracket, must_include/exclude, pool and theme; everything else in",
        "  brief.json is a note that reaches `info.json` and no algorithm.",
        "",
        f"  next:  git add data/decks/{slug} && git commit",
        f"         manamap pilot promote {slug}"
        "            # what it owes to reach the bench",
        f"         manamap pilot deck-version {slug} tag v0.1.0"
        "   # 0.x is a list; 1.0.0 is sleeved",
        f"         manamap pilot deck-info {slug}",
    ]
    return "\n".join(lines)


def _dirty(slug):
    """Whether this deck already has uncommitted changes. Reported, never acted on."""
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--", f"data/decks/{slug}"],
            capture_output=True, text=True, timeout=20)
        return bool(proc.stdout.strip())
    except Exception:
        return False


def main(args):
    if args.bracket is not None and args.bracket not in range(1, BRACKET_MAX + 1):
        raise SystemExit(f"--bracket must be 1-{BRACKET_MAX}, got {args.bracket}")
    try:
        record = build(args)
    except (BuildError, build_deck.BriefError) as exc:
        raise SystemExit(str(exc))

    if getattr(args, "as_json", False):
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return
    print(format_report(record))


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot build <slug>`.")
