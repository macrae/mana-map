"""Pilot: which ENVIRONMENT a deck is in, and what it owes to leave it.

PRD §3 calls the three environments the spine of the product: requirements
tighten at each promotion. Before this the ladder had two rungs and one of them
was computed in JavaScript — `workbench.js` reads
`living.filter(e => !e.locked)` and calls the result "on the bench". Nothing
stored it, nothing validated it, and nothing gated on it.

    DEV      brewing. Most of these get thrown away; the environment is
             optimised for throughput, not rigour.
    BENCH    a candidate earning its way to the table. Full analytical treatment.
    SLEEVED  in sleeves, playable tonight. Version pinned.

## Two of the three rungs are DERIVED, and that is deliberate

**SLEEVED is the paper lock and nothing else.** Whether a deck exists as
cardboard is the one claim no artifact can derive and only the pilot can make,
and `deck_versions.paper` already IS that claim. A stored `stage: "sleeved"`
would be a second source for one fact, free to disagree with the lock — and the
workbench has already been burned by exactly that shape, filtering on `locked`
before `status` and rendering a broken-down deck under SLEEVED.

**BENCH is the default**, which is what the frontend has computed all along.

**Only DEV is stored**, because nobody has ever said which decks are throwaway
brews and inferring it would be inventing an opinion about ten existing decks.
`manamap pilot build` lands a new deck there; everything else is on the bench
until someone says otherwise. So no file migrates and no deck changes meaning.

## The gate reports PER REQUIREMENT

`bracket.assess` sets the pattern and the reason: a single pass/fail tells you a
deck is not ready and nothing about what to do next, and A-3 asks for the same
thing one layer up. Every row names the artifact, its state, and the command
that produces it.

A gate row is not a validator. `deck_status` already runs those; this reads its
answer and decides whether the rung is earned, so the two cannot disagree about
whether an artifact is present, stale or invalid.
"""

from manamap.pilot import deck_versions
from manamap.pilot.common import deck_dir, deck_lifecycle

DEV = "dev"
BENCH = "bench"
SLEEVED = "sleeved"

#: The ladder, in order. Index is the rung, so a promotion is a step up and a
#: demotion is a step down — neither may skip, because the gate for the rung you
#: skipped would never be checked.
LADDER = (DEV, BENCH, SLEEVED)

#: The key inside `deck_versions.json`, owned by the module that serialises
#: that file — `_write_tags` orders every key and silently DROPS any it does
#: not know, which is what happened the first time `stage` was set. So the
#: name lives there and is read from there.
STAGE_KEY = deck_versions.STAGE_KEY

#: What each promotion requires, from PRD §7's deck-object table. Each row is
#: `(label, artifact-or-check, why)`. The artifact names are `deck_status`
#: rows, so presence, staleness and validity all come from one place.
GATES = {
    BENCH: (
        ("decklist", "decklist.txt",
         "the 99 itself — nothing below this is meaningful without it"),
        ("cards resolved", "cards.json",
         "Scryfall resolution with printings; every downstream figure reads it"),
        ("composition stats", "mana_analysis.json",
         "colour sources against the Karsten target, and the curve"),
        ("short sim batch", "goldfish_metrics.json",
         "10,000 seeded games of resource development — the dev batch"),
        ("combo audit", "bracket_report.json",
         "contained combos, two-card infinites and the computed bracket floor"),
        ("engine declared", "goldfish_targets.json",
         "what the deck is FOR, machine-readable, or `diagnose` withholds the "
         "engine figure silently"),
    ),
    SLEEVED: (
        ("full sim batch", "sim/",
         "at least one Forge run against a pod — the goldfish has no blockers "
         "and its verdict on board quality is not evidence"),
        ("mulligan + tutor guidance", "tutor_guide.json",
         "what to keep and what to wish for, at the table"),
        ("benchmark score", "benchmark.json",
         "four measures under one frozen harness, so decks compare"),
        ("handbook", "manuals/p/<slug>.html",
         "the Pilot's Operating Handbook — procedures for when it goes wrong"),
        ("dossier", "info.json",
         "the composed view the deck page fetches"),
        ("ownership reconciled", "@ownership",
         "every card in a box, in another deck, or named as a buy"),
    ),
}


def stored(slug):
    """The stage written in `deck_versions.json`, or None. Only ever `dev`."""
    from manamap.pilot.common import load_json

    doc = load_json(deck_dir(slug) / deck_versions.TAGS_FILE) or {}
    block = doc.get(STAGE_KEY) or {}
    return block.get("name") or None


def stage(slug):
    """Which environment this deck is in, or None if it is not in one.

    A BROKEN-DOWN OR RETIRED DECK HAS NO STAGE. It is not in dev, on the bench
    or in sleeves — it is history, and reporting `bench` for it would put a pile
    of cards on a rack of things you could work on. Absent means absent, and
    `main` refuses to move such a deck for the same reason.

    Otherwise the paper lock wins outright: it is the claim, and a stored stage
    that could disagree with it would be the second source of truth this module
    exists to avoid.
    """
    if deck_lifecycle(slug):
        return None
    if deck_versions.paper(slug):
        return SLEEVED
    return stored(slug) or BENCH


def set_stage(slug, name):
    """Write the stage. Refuses anything the ladder does not name.

    ONLY `dev` IS STORED. Writing `sleeved` here would create a second claim
    about cardboard beside the lock; writing `bench` would store the default and
    make its absence mean something new on ten existing files. Both are refused
    with the command that actually does the job.
    """
    from datetime import date

    from manamap.pilot.common import load_json

    if name not in LADDER:
        raise SystemExit(f"{name!r} is not a stage — one of {', '.join(LADDER)}")
    if name == SLEEVED:
        raise SystemExit(
            f"SLEEVED is the paper lock, not a stage to write. "
            f"`manamap pilot deck-version {slug} paper` asserts that this exact "
            f"99 is in sleeves — the one fact no artifact can derive.")

    path = deck_dir(slug) / deck_versions.TAGS_FILE
    if name == BENCH and not path.exists():
        # NOTHING TO WRITE. Setting the default on a deck that has no versions
        # file would CREATE one containing only the default — and a round trip
        # through `dev` and back would leave a tracked artifact behind that was
        # never there. Found by doing exactly that to zur-enchantress.
        return stage(slug)

    doc = load_json(path) or {"slug": slug, "tags": {}}
    doc.setdefault("slug", slug)
    if name == BENCH:
        # The default. Storing it would make its absence mean something new.
        doc.pop(STAGE_KEY, None)
        # AND IF THAT WAS THE ONLY CLAIM, THE FILE GOES. `deck_versions.json`
        # holds assertions — the lifecycle, the baseline, the sleeve lock, the
        # tags — and one containing none of them asserts nothing. Leaving it
        # would mean a dev→bench round trip permanently adds a tracked artifact
        # to a deck that never had one, which is what it did to zur-enchantress
        # the first time.
        if not any(doc.get(k) for k in
                   (deck_versions.LIFECYCLE_KEY, deck_versions.BASELINE_KEY,
                    deck_versions.PAPER_KEY, "tags")):
            path.unlink(missing_ok=True)
            return stage(slug)
    else:
        doc[STAGE_KEY] = {"name": name, "at": date.today().isoformat()}
    deck_versions._write_tags(path, doc)
    return stage(slug)


def _ownership(slug):
    """Every card in a box, in another deck, or named as a buy. SLEEVED's gate.

    The one gate row that is a computation rather than a file, because there is
    no ownership artifact — `collection` answers "is this in a box" and
    `deck_branch._deck_holders` answers "which deck is it sleeved in", and
    nothing has ever joined them for a whole 99.
    """
    from manamap.pilot import collection, deck_branch
    from manamap.pilot.common import expand_copies, is_land, load_deck_cards

    try:
        cards = load_deck_cards(slug)["cards"]
    except Exception:
        return "missing", "no cards.json — run `fetch-deck` first", []
    owned = collection.owned_names()
    if not owned:
        return "unknown", ("no boxes under data/collection/, so ownership "
                           "cannot be reconciled at all"), []

    # C-3 asks for THREE buckets, and they must stay three. `collection` is
    # deliberately the only reader of the boxes and deliberately does NOT count
    # deck membership — a card sleeved in another deck is not one you can put in
    # this one without taking that deck apart. So "in another deck" is reported
    # BESIDE ownership and never folded into it.
    #
    # Basics are excluded: a build's Swamps are not a shopping list, and every
    # deck would otherwise fail on lands nobody counts.
    names = sorted({c["name"] for c in expand_copies(cards)
                    if not (is_land(c) and "Basic" in (c.get("type_line") or ""))})
    in_box = [n for n in names if n in owned]
    rest = [n for n in names if n not in owned]
    elsewhere, buy = [], []
    for name in rest:
        holders = [h for h in deck_branch._deck_holders(name, skip=slug)
                   if not h["apart"]]
        (elsewhere if holders else buy).append(
            f"{name} (in {holders[0]['slug']})" if holders else name)

    detail = (f"{len(in_box)} in a box, {len(elsewhere)} sleeved elsewhere, "
              f"{len(buy)} to buy")
    if not rest:
        return "present", f"all {len(names)} in a box", []
    return "missing", detail, elsewhere + buy


def gate(slug, to):
    """Every requirement for one rung, with its state. Never a bare pass/fail."""
    from manamap.pilot import deck_status

    if to not in GATES:
        raise SystemExit(f"nothing to check for {to!r} — "
                         f"gates exist for {', '.join(sorted(GATES))}")
    by_artifact = {row["artifact"]: row for row in deck_status.status(slug)}
    rows = []
    for label, artifact, why in GATES[to]:
        if artifact == "@ownership":
            state, detail, missing = _ownership(slug)
            rows.append({"label": label, "artifact": "the boxes", "why": why,
                         "state": state, "detail": detail, "how":
                         f"manamap pilot deck-branch {slug} source (or buy them)",
                         "missing": missing})
            continue
        if artifact.startswith("manuals/"):
            from manamap.config import MANUALS_DIR

            path = MANUALS_DIR / "p" / f"{slug}.html"
            rows.append({"label": label, "artifact": artifact, "why": why,
                         "state": "present" if path.exists() else "missing",
                         "detail": "", "how": f"manamap pilot build-poh {slug}"})
            continue
        row = by_artifact.get(artifact)
        if row is None:
            # `deck_status` does not track it, so presence is the only question
            # this can honestly answer.
            path = deck_dir(slug) / artifact
            rows.append({"label": label, "artifact": artifact, "why": why,
                         "state": "present" if path.exists() else "missing",
                         "detail": "", "how": ""})
            continue
        rows.append({"label": label, "artifact": artifact, "why": why,
                     "state": row["state"], "detail": row.get("detail", ""),
                     "how": row.get("how", "")})
    return rows


def blockers(rows):
    """The rows that are not satisfied. `present` is the only passing state."""
    return [r for r in rows if r["state"] != "present"]


def _next_rung(current, direction):
    i = LADDER.index(current)
    j = i + direction
    if not 0 <= j < len(LADDER):
        return None
    return LADDER[j]


def format_gate(slug, to, rows):
    lines = [f"\n{slug}: gate for {to.upper()}\n"]
    for row in rows:
        mark = {"present": "✓", "missing": "✗",
                "STALE": "~", "INVALID": "!"}.get(row["state"], "?")
        detail = f"  {row['detail']}" if row["detail"] else ""
        lines.append(f"  {mark} {row['label']:<26} {row['state']}{detail}")
    stuck = blockers(rows)
    if not stuck:
        lines.append(f"\n  every requirement met.")
        return "\n".join(lines)
    lines.append(f"\n  {len(stuck)} of {len(rows)} not met:")
    for row in stuck:
        lines.append(f"    {row['label']} — {row['why']}")
        if row.get("how"):
            lines.append(f"      {row['how']}")
        if row.get("missing"):
            shown = row["missing"][:6]
            more = (f" (+{len(row['missing']) - 6})"
                    if len(row["missing"]) > 6 else "")
            lines.append(f"      {', '.join(shown)}{more}")
    return "\n".join(lines)


def main(args):
    slug = args.slug
    if not deck_dir(slug).is_dir():
        raise SystemExit(f"{slug}: no such deck under data/decks/")

    life = deck_lifecycle(slug)
    if life:
        raise SystemExit(
            f"{slug} is {life[1]} — a deck in a pile does not move between "
            f"environments. `deck-state {slug} revive` first.")

    now = stage(slug)
    # The subcommand name IS the direction. `registry` dispatches both verbs to
    # this module, so the parser records which one was typed.
    action = getattr(args, "pilot_command", "promote")
    if getattr(args, "show", False):
        action = "show"

    if action == "show":
        print(f"{slug}: {now.upper()}")
        for rung in LADDER:
            mark = ">" if rung == now else " "
            print(f"  {mark} {rung}")
        nxt = _next_rung(now, +1)
        if nxt:
            rows = gate(slug, nxt)
            stuck = blockers(rows)
            print(f"\n  to {nxt}: {len(rows) - len(stuck)} of {len(rows)} "
                  f"requirement(s) met — `manamap pilot promote {slug}` to see them")
        return

    direction = +1 if action == "promote" else -1
    target = getattr(args, "to", None) or _next_rung(now, direction)
    if target is None:
        edge = "top" if direction > 0 else "bottom"
        raise SystemExit(f"{slug} is already at the {edge} of the ladder ({now})")
    if target not in LADDER:
        raise SystemExit(f"{target!r} is not a stage — one of {', '.join(LADDER)}")
    if LADDER.index(target) - LADDER.index(now) != direction:
        raise SystemExit(
            f"{slug} is {now} and {target} is not one step "
            f"{'up' if direction > 0 else 'down'} — a promotion may not skip a "
            f"rung, because the gate for the rung it skipped never runs.")

    if direction > 0:
        rows = gate(slug, target)
        print(format_gate(slug, target, rows))
        stuck = blockers(rows)
        if stuck and not getattr(args, "force", False):
            raise SystemExit(
                f"\n  NOT PROMOTED. Clear them, or `--force --reason \"…\"` to "
                f"promote anyway and record why.")
        if stuck:
            reason = getattr(args, "reason", None)
            if not reason:
                raise SystemExit("--force needs --reason: a gate waived without "
                                 "a stated reason is a gate nobody will trust")
            print(f"\n  FORCED past {len(stuck)} requirement(s): {reason}")

    if target == SLEEVED:
        raise SystemExit(
            f"\n  The gate is clear. Sleeving is the pilot's own act and is not "
            f"automated: `manamap pilot deck-version {slug} paper` asserts that "
            f"THIS exact 99 is in sleeves.")
    if now == SLEEVED:
        raise SystemExit(
            f"{slug} is SLEEVED. Coming off the table means withdrawing the "
            f"lock: `manamap pilot deck-version {slug} paper --clear`.")

    print(f"\n{slug}: {now} → {set_stage(slug, target)}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot promote <slug>`.")
