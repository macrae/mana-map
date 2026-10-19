"""Pilot: the gate on `data/fleet/analytics.json` — form, re-adds, and no score.

A fleet file, so it takes no slug (the `validate-queue` shape). FORM is gated:
the keys, the schema version, sorted and unique slugs that are real deck
directories, every number carried as a `fig` or an `absent`, rates inside their
intervals, each row's group consistent with its bracket, the groups' bands
re-adding from the rows, `FLEET_MIN_GROUP` honoured, and — the one rule this
file exists to keep — NO COMPOSITE KEY anywhere on a row or a group. A numeric
`score`, `rating`, `index`, `power`, `grade` or `rank` on a deck row is an
error, whatever it was meant to mean.

FRESHNESS is warned, never failed: a row whose `built_from` sha is not the
deck's current `cards.json` sha is "stale: <slug> list moved", exit 0 — the
file is rebuilt by `regen`'s last stage and by `make manuals`, and a gate that
fails on time rather than form is a gate that gets switched off. Absent file:
"absent means absent", exit 0.
"""

import json

from manamap import config
from manamap.pilot import fleet_analytics as fa
from manamap.pilot.common import measured_sha, report_errors

REQUIRED = ("schema_version", "decks", "groups", "fleet_bands", "comparisons", "limits", "meta")
ROW_REQUIRED = ("slug", "format", "stage", "apart", "lifecycle", "member", "decklist_sha256", "bracket",
                "benchmark", "diagnostic", "own_goldfish", "dti", "table", "record", "survey",
                "price", "placements", "rating")
RATE_MEASURES = {"missed_land_drop_rate", "mulligan_rate", "kill_by_turn_6", "kill_by_turn_8",
                 "kill_by_turn_10", "never_kills_rate", "stall_two_in_a_row",
                 "missed_land_drop_by_five", "mulliganed", "win_rate"}


def _walk_numbers(node, path, errors, *, row):
    """Every number on a row must be inside a `fig` (as `value`, `n`, `ci95`) or
    a plain structural count; a bare composite-looking numeric key is an error."""
    if isinstance(node, dict):
        if fa.is_fig(node) or fa.is_absent(node):
            return
        for k, v in node.items():
            if k in fa.COMPOSITE_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool):
                errors.append(f"{row}: {path}.{k} is a bare composite number — this file publishes "
                              f"profiles, never a score")
            _walk_numbers(v, f"{path}.{k}", errors, row=row)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk_numbers(v, f"{path}[{i}]", errors, row=row)


def _check_fig(cell, where, errors):
    if fa.is_absent(cell):
        if not str(cell.get("absent_because") or "").strip():
            errors.append(f"{where}: absent without a reason")
        return
    if not fa.is_fig(cell):
        errors.append(f"{where}: not a fig or an absent")
        return
    v = cell["value"]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        errors.append(f"{where}: value is not a number")
        return
    ci = cell.get("ci95")
    if ci is not None:
        if not (isinstance(ci, list) and len(ci) == 2):
            errors.append(f"{where}: ci95 must be [lo, hi]")
        elif not (ci[0] <= v <= ci[1]):
            errors.append(f"{where}: value {v} outside its ci95 {ci}")
        if cell.get("n") is None:
            errors.append(f"{where}: an interval without its n")
    name = where.rsplit(".", 1)[-1]
    if name in RATE_MEASURES and not 0 <= v <= 1:
        errors.append(f"{where}: a rate outside [0, 1] ({v})")


def validate(doc, *, current_shas=None, deck_dirs=None):
    """`(errors, warnings)`. `current_shas` maps slug -> cards.json sha (None =
    don't check freshness); `deck_dirs` is the set of real deck directories."""
    errors, warnings = [], []
    if not isinstance(doc, dict):
        return ["not an object"], warnings
    for k in REQUIRED:
        if k not in doc:
            errors.append(f"missing top-level {k!r}")
    if errors:
        return errors, warnings
    if doc["schema_version"] != fa.SCHEMA_VERSION:
        errors.append(f"schema_version {doc['schema_version']!r} != {fa.SCHEMA_VERSION}")
    rows = doc["decks"]
    if not isinstance(rows, list):
        return ["decks is not a list"], warnings
    slugs = [r.get("slug") for r in rows if isinstance(r, dict)]
    if slugs != sorted(slugs):
        errors.append("decks are not sorted by slug")
    if len(set(slugs)) != len(slugs):
        errors.append("a slug appears twice")
    for r in rows:
        if not isinstance(r, dict):
            errors.append("a deck row is not an object")
            continue
        s = r.get("slug")
        at = f"{s}"
        for k in ROW_REQUIRED:
            if k not in r:
                errors.append(f"{at}: missing {k!r}")
        if deck_dirs is not None and s not in deck_dirs:
            errors.append(f"{at}: no such deck directory")
        if r.get("rating") is not None:
            errors.append(f"{at}: rating must be null here — the rating lives in rating.json")
        b = r.get("bracket") or {}
        want_group, want_basis = fa._group(b)
        if (b.get("group"), b.get("group_basis")) != (want_group, want_basis):
            errors.append(f"{at}: group {b.get('group')!r}/{b.get('group_basis')!r} does not follow "
                          f"from target {b.get('target')!r} and floor {b.get('floor')!r}")
        for block in ("benchmark", "diagnostic", "dti", "table", "price"):
            blk = r.get(block)
            if fa.is_absent(blk):
                _check_fig(blk, f"{at}.{block}", errors)
            elif isinstance(blk, dict):
                for k, cell in blk.items():
                    if isinstance(cell, dict) and (fa.is_fig(cell) or fa.is_absent(cell)):
                        _check_fig(cell, f"{at}.{block}.{k}", errors)
            else:
                errors.append(f"{at}.{block}: not an object")
        sv = r.get("survey")
        if isinstance(sv, dict) and not fa.is_absent(sv) and sv.get("tier") != "authored":
            errors.append(f"{at}.survey: must be labelled tier 'authored'")
        _walk_numbers({k: v for k, v in r.items() if k not in ("record",)}, at, errors, row=at)
        if current_shas is not None:
            cur = current_shas.get(s)
            built = (doc["meta"].get("built_from") or {}).get(s)
            if cur and built and cur != built:
                warnings.append(f"stale: {s} list moved since this was built — `manamap pilot fleet-analytics --write`")
    # Groups: membership from the rows, bands re-add.
    by_slug = {r["slug"]: r for r in rows if isinstance(r, dict)}
    want_members = {}
    for r in by_slug.values():
        if not fa.is_member(r):
            continue
        want_members.setdefault(r["bracket"]["group"], []).append(r["slug"])
        if r.get("member") is not True:
            errors.append(f"{r['slug']}: a sleeved, bracketed Commander deck must carry member: true")
    groups = doc["groups"]
    if set(groups) != set(want_members):
        errors.append(f"groups {sorted(groups)} do not match the rows' groups {sorted(want_members)}")
    for g, spec in groups.items():
        if g not in want_members:
            continue
        if spec.get("members") != want_members[g] or spec.get("n") != len(want_members[g]):
            errors.append(f"{g}: members {spec.get('members')} != rows' {want_members[g]}")
        want_bands = fa.bands(list(by_slug.values()), set(want_members[g]))
        if spec.get("bands") != want_bands:
            errors.append(f"{g}: bands do not re-add from the rows")
        for name, band in (spec.get("bands") or {}).items():
            if fa.is_absent(band):
                continue
            if band.get("n", 0) < fa.MIN_GROUP:
                errors.append(f"{g}.bands.{name}: a band on n={band.get('n')} < {fa.MIN_GROUP}")
        _walk_numbers({"bands": spec.get("bands")}, g, errors, row=g)
    fleet_members = {s for m in want_members.values() for s in m}
    if doc["fleet_bands"] != fa.bands(list(by_slug.values()), fleet_members):
        errors.append("fleet_bands do not re-add from the rows")
    meta = doc["meta"]
    if meta.get("no_composite") != fa.NO_COMPOSITE:
        errors.append("meta.no_composite is not the refusal this file is built on")
    if meta.get("min_group") != fa.MIN_GROUP:
        errors.append(f"meta.min_group {meta.get('min_group')} != {fa.MIN_GROUP}")
    sp = (doc["comparisons"] or {}).get("spearman") or {}
    for k, cell in sp.items():
        if cell.get("verdict") not in ("NOT ANSWERABLE", "measured"):
            errors.append(f"comparisons.spearman.{k}: verdict {cell.get('verdict')!r}")
        if cell.get("verdict") == "measured" and (cell.get("n") or 0) < meta.get("calibrate_min_decks", 10):
            errors.append(f"comparisons.spearman.{k}: measured on n={cell.get('n')} below the refusal line")
    return errors, warnings


def main(args=None):
    p = fa.ARTIFACT
    if not p.is_file():
        print(f"no {p.name} under {p.parent.name}/ — `manamap pilot fleet-analytics --write` "
              f"builds it (absent means absent)")
        return
    doc = json.loads(p.read_text(encoding="utf-8"))
    slugs = set(fa.live_slugs())
    shas = {s: measured_sha(s) for s in slugs}
    errors, warnings = validate(doc, current_shas=shas, deck_dirs=slugs)
    for w in warnings:
        print(f"WARN {w}")
    report_errors("fleet/analytics.json", errors)
    print(f"OK   fleet/analytics.json: {len(doc['decks'])} decks, {len(doc['groups'])} group(s), "
          f"{len(warnings)} warning(s); profiles ◆, never a score")
