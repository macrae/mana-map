"""Pilot: mechanically check a deck's build brief.

`brief.json` is the input to every build — the commander, the cards the pilot
promised would be in the 99, the bracket the deck is aimed at, and the boxes it
may draw from. It is TRACKED, it is AUTHORED, and until now it was the only
tracked pilot artifact with no gate at all. `deck_status.STAGES[0]` lists it with
`sha=None` and no validator, so nothing ever checked it.

That silence had a cost with a name. `build_deck.role_budget_for` resolves
`theme` against EDHREC and **falls back to the flat provisional budget with a
reason** when the lookup returns nothing — correct behaviour, and invisible: a
typo'd style produces a legal 99 built to the wrong shape, and the only trace is
a string in `role_budget_grounding` that nothing reads.

EVERY CHECK BELOW WAS MEASURED AGAINST ALL FOUR BRIEFS ON DISK BEFORE IT WAS
WRITTEN, and every one of them fires on **zero** of the four. That is the whole
of the entry criterion here: a validator that fires on correct data is worse than
no validator, and six proposed checks have been rejected in this repo on exactly
that ground. The four are hapatra, radagast, yawgmoth-swarm and zur-enchantress.

  * `slug` matches the directory it sits in;
  * `commander` resolves against the corpus and passes `commander_rejection`;
  * every `must_include` / `must_exclude` name resolves against the corpus;
  * every `must_include` is inside the commander's colour identity;
  * every `pool_files` path exists;
  * `bracket` is 1-5 and `must_include` / `must_exclude` are lists of strings.

TWO THINGS ARE REPORTED AND NEVER FAILED, for the reason
`validate_goldfish_targets` states about its own scaffold note — a gate that
reddens correct artifacts teaches its reader to ignore the gate.

  * **The inert keys.** `partner`, `playstyle`, `notes`, `design_rules`,
    `win_conditions`, `commander_rationale`, `mana` and `targets` are written to
    briefs, echoed into `info.json`, and read by NO code: the builder consumes
    `commander`, `bracket`, `must_include`, `must_exclude`, `pool`/`pool_files`
    and `theme`, and nothing else. Three of the four briefs carry at least one,
    so failing on them would redden three correct files. They are named on every
    run instead, because a field that quietly does nothing looks exactly like one
    that works — which is the same reason `autobuild` prints it under a build.
  * **A thin or unresolvable `theme`.** Checking it needs EDHREC, and a gate that
    fails when the network is down is a gate that gets disabled. `--themes` opts
    into the lookup; without it the style is checked for shape only.
"""

import json
import pathlib

from manamap.config import BRACKET_MAX, DECKS_DIR
from manamap.pilot.common import commander_rejection, deck_dir, report_errors

#: Keys the BUILDER actually reads. Derived from `build_deck.load_brief` +
#: `resolve_pool` + `role_budget_for`, and asserted against them by a test, so
#: this list cannot quietly fall behind the code it describes.
CONSUMED = frozenset({
    "slug", "commander", "bracket", "must_include", "must_exclude",
    "pool", "pool_files", "theme", "format",
})

#: Keys that appear in real briefs and are read by nothing. Reported, never
#: failed — see the module docstring. `format` is NOT here: `serve` reads it.
INERT = frozenset({
    "partner", "playstyle", "notes", "design_rules", "win_conditions",
    "commander_rationale", "mana", "targets",
})


def _names():
    """The corpus's name index, front faces and joined forms both."""
    from manamap.pilot.card_pool import corpus_names

    return corpus_names()


def _rows():
    """`{name: row}` over the corpus frame, first printing winning.

    First-printing-wins is right for IDENTITY, which is all this uses it for.
    It is deliberately NOT used for legality — that combines across printings,
    and a promo `not_legal` row sorting first once failed two decks on their own
    tracked lists.
    """
    from manamap.pilot.build_deck import load_frame

    rows = {}
    for rec in load_frame().to_dict("records"):
        rows.setdefault(rec["name"], rec)
    return rows


def validate(doc, slug, rows=None, names=None, check_themes=False):
    """Return `(errors, notes)`. Errors fail the gate; notes are printed."""
    from manamap.analysis.common import parse_color_identity

    errors, notes = [], []

    if doc.get("slug") != slug:
        errors.append(
            f"slug is {doc.get('slug')!r} but the brief sits in {slug}/ — the "
            f"two must agree, or a build writes into the wrong deck")

    bracket = doc.get("bracket")
    if bracket is not None and bracket not in range(1, BRACKET_MAX + 1):
        errors.append(f"bracket must be 1-{BRACKET_MAX}, got {bracket!r}")

    for key in ("must_include", "must_exclude"):
        value = doc.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            errors.append(f"{key} must be a list of card names, got {type(value).__name__}")

    for key in ("pool_files",):
        for raw in doc.get(key) or []:
            path = pathlib.Path(raw)
            if not path.exists() and not (DECKS_DIR / slug / raw).exists():
                errors.append(f"{key} names {raw!r}, which is not on disk — "
                              f"`resolve_pool` raises on it at build time")

    commander = doc.get("commander")
    if not commander:
        errors.append("no commander — the builder cannot start without one")
        return errors, notes

    rows = rows if rows is not None else _rows()
    names = names if names is not None else _names()

    row = rows.get(commander)
    if row is None:
        errors.append(f"commander {commander!r} is not in the corpus — check the "
                      f"spelling, and use the full ' // ' form for a DFC")
        identity = set()
    else:
        rejection = commander_rejection(row)
        if rejection:
            errors.append(f"commander {commander!r} cannot be a commander: {rejection}")
        identity = parse_color_identity(row.get("color_identity", ""))

    for key in ("must_include", "must_exclude"):
        for name in doc.get(key) or []:
            if name not in names:
                errors.append(f"{key} names {name!r}, which is not in the corpus")
            elif key == "must_include" and row is not None:
                outside = parse_color_identity(
                    rows[name].get("color_identity", "")) - identity
                if outside:
                    # `legal_must_includes` DROPS these at build time and says
                    # so in the plan. Failing here is still right: a promise the
                    # builder silently cannot keep is a defect in the brief, and
                    # the plan's report is read after the build rather than
                    # before it.
                    errors.append(
                        f"must_include {name!r} is {''.join(sorted(outside))}, "
                        f"outside {commander}'s "
                        f"{''.join(sorted(identity)) or 'colourless'} identity — "
                        f"the builder drops it")

    inert = sorted(set(doc) & INERT)
    if inert:
        notes.append(
            f"read by nothing: {', '.join(inert)}. The builder consumes "
            f"commander, bracket, must_include/exclude, pool/pool_files and "
            f"theme; these reach `info.json` and no algorithm")

    unknown = sorted(set(doc) - CONSUMED - INERT - {"_pool"})
    if unknown:
        notes.append(f"unrecognised, and also read by nothing: {', '.join(unknown)}")

    theme = doc.get("theme")
    if theme and check_themes:
        notes.extend(_theme_notes(commander, theme))
    elif theme:
        notes.append(f"style {theme!r} checked for shape only — pass --themes to "
                     f"resolve it against EDHREC (a lookup, and a network call)")
    return errors, notes


def _theme_notes(commander, theme):
    """Resolve a style against the commander's real archetypes. Never fatal.

    A network failure returns a note saying the lookup did not happen, because a
    gate that fails when EDHREC is down is a gate that gets switched off.
    """
    from manamap.pilot import archetypes

    try:
        themes = archetypes.list_themes(commander)
    except Exception as exc:                        # network, cache, or shape
        return [f"style {theme!r} not resolved — EDHREC lookup failed ({exc})"]

    match = next((t for t in themes if t["slug"] == theme), None)
    if match is None:
        near = [t["slug"] for t in themes if theme in t["slug"] or t["slug"] in theme]
        return [f"style {theme!r} is NOT one of {commander}'s "
                f"{len(themes)} archetypes, so `role_budget_for` falls back to "
                f"the flat provisional budget and only says so in "
                f"`role_budget_grounding`"
                + (f". Did you mean {', '.join(near[:3])}?" if near else "")]
    if match["decks"] < archetypes.MIN_DECKS_FOR_TEMPLATE:
        return [f"style {theme!r} has {match['decks']} decks behind it "
                f"(under {archetypes.MIN_DECKS_FOR_TEMPLATE}) — its role budget "
                f"describes those {match['decks']} decks, not the archetype"]
    return [f"style {theme!r} — {match['decks']} decks"]


def main(args):
    path = deck_dir(args.slug) / "brief.json"
    if not path.exists():
        raise SystemExit(f"{path} not found — `manamap pilot brew {args.slug} "
                         f"--commander \"<name>\"` writes one")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report_errors(path.name, [f"not valid JSON: {exc}"])
        return

    errors, notes = validate(doc, args.slug,
                             check_themes=getattr(args, "themes", False))
    ok = (f"OK   {path.name} — {doc.get('commander')}, bracket "
          f"{doc.get('bracket')}, {len(doc.get('must_include') or [])} "
          f"must-include, {len(doc.get('must_exclude') or [])} must-exclude")
    for note in notes:
        ok += f"\n     {note}"
    report_errors(path.name, errors, ok)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-brief <slug>`.")
