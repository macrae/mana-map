"""Pilot: form-check `combos.json` — a deck's known lines and its near misses.

◆-tier like `bracket_report.json`, and recomputable, so the gate is about
CONSISTENCY rather than reality: the report must describe the list on disk
(`decklist_sha256`), every included line must sit inside the 99 plus the command
zone, every near miss must name a card that is real, Commander-legal, inside
the identity and NOT in the list, ids must be unique, and the stored `summary`
must equal the summary recomputed from the rows beneath it. The combo file the
report was built from must be the one on disk (`combo_data.combo_count`), which
is the check that fires after a Spellbook refresh.

Whether the rows are exactly what a fresh run would write is the freshness
test's question (`tests/test_pilot_artifact_freshness.py`), not this one's.
"""
import sys

from manamap.pilot.card_pool import load_pool
from manamap.pilot.card_search import commander_identity
from manamap.pilot.common import (
    decklist_sha256,
    deck_dir,
    load_combo_details,
    load_deck_cards,
    load_json,
    report_errors,
    sha_matches,
)
from manamap.pilot.deck_combos import ARTIFACT, summarize

REQUIRED = ("slug", "branch", "decklist_sha256", "combo_data", "summary", "included", "near")
INCLUDED_KEYS = ("id", "cards", "produces", "infinite", "bracket", "banned",
                 "mana_value_needed", "popularity", "assumes_other_commander")
NEAR_KEYS = ("id", "cards", "missing", "infinite", "bracket", "mana_value_needed", "popularity")


def validate(slug, doc, *, branch=None, present=None, identity=None, pool=None,
             live_sha=None, combo_count=None):
    """Every reason this document is not a combos report for `slug`.

    The deck-side facts are parameters so the checks can be driven on synthetic
    data: `present` is the deck's names plus its commanders, `identity` the
    commander's colours, `pool` the corpus view (`card_pool.load_pool`),
    `live_sha` the list's sha and `combo_count` the combo file's count. A
    parameter left None skips the check it feeds and says nothing — the CLI
    passes all of them.
    """
    errors = []
    for k in REQUIRED:
        if k not in doc:
            errors.append(f"missing required key {k!r}")
    if errors:
        return errors
    if doc["slug"] != slug:
        errors.append(f"slug is {doc['slug']!r} but the artifact lives in {slug}/")
    if doc["branch"] != branch:
        errors.append(f"branch is {doc['branch']!r} but the artifact lives in "
                      f"{'branches/' + branch if branch else 'the deck directory'}")
    if live_sha is not None and not sha_matches(doc.get("decklist_sha256"), live_sha):
        errors.append(f"decklist_sha256 {str(doc.get('decklist_sha256'))[:12]}… is not the "
                      f"list on disk ({live_sha[:12]}…) — stale: rerun `deck-combos --write`")
    stored_count = (doc.get("combo_data") or {}).get("combo_count")
    if combo_count is not None and stored_count != combo_count:
        errors.append(f"combo_data.combo_count {stored_count} is not the combo file's "
                      f"{combo_count} — built from a different Spellbook dump")

    included = doc.get("included")
    near = doc.get("near")
    if not isinstance(included, list) or not isinstance(near, list):
        errors.append("included and near must be lists")
        return errors

    for i, row in enumerate(included):
        where = f"included[{i}]"
        missing_keys = [k for k in INCLUDED_KEYS if k not in (row or {})]
        if missing_keys:
            errors.append(f"{where}: missing {', '.join(missing_keys)}")
            continue
        if not row["cards"]:
            errors.append(f"{where}: no cards")
        if present is not None:
            outside = [c for c in row["cards"] if c not in present]
            if outside:
                errors.append(f"{where}: {', '.join(outside)} not in the list — a line the "
                              f"deck does not contain")
    for i, row in enumerate(near):
        where = f"near[{i}]"
        missing_keys = [k for k in NEAR_KEYS if k not in (row or {})]
        if missing_keys:
            errors.append(f"{where}: missing {', '.join(missing_keys)}")
            continue
        name = row["missing"]
        if name not in row["cards"]:
            errors.append(f"{where}: missing card {name!r} is not one of the line's cards")
        if present is not None and name in present:
            errors.append(f"{where}: {name} is in the list — not a near miss")
        if present is not None:
            absent = [c for c in row["cards"] if c != name and c not in present]
            if absent:
                errors.append(f"{where}: {', '.join(absent)} also absent — more than one card short")
        if pool is not None:
            rec = pool.get(name)
            if rec is None:
                errors.append(f"{where}: {name!r} is not in the corpus")
            else:
                if not rec.get("legal"):
                    errors.append(f"{where}: {name} is not Commander-legal")
                if identity is not None and not set(rec.get("color_identity") or ()) <= set(identity):
                    errors.append(f"{where}: {name} ({''.join(sorted(rec.get('color_identity') or ()))}) "
                                  f"is outside the identity {''.join(sorted(identity))}")

    ids = [r.get("id") for r in included + near if isinstance(r, dict)]
    dupes = sorted({x for x in ids if ids.count(x) > 1})
    if dupes:
        errors.append(f"duplicate combo id(s) across included and near: {', '.join(map(str, dupes))}")

    if all(isinstance(r, dict) and all(k in r for k in INCLUDED_KEYS) for r in included):
        want = summarize(included, near, (doc.get("summary") or {}).get("near_total", len(near)))
        got = doc.get("summary") or {}
        if got.get("near_total") is not None and got["near_total"] < len(near):
            errors.append(f"summary.near_total {got['near_total']} is below the {len(near)} rows shown")
        for k, v in want.items():
            if got.get(k) != v:
                errors.append(f"summary.{k} is {got.get(k)!r}; the rows say {v!r}")
    return errors


def main(args=None):
    slug = getattr(args, "slug", None)
    branch = getattr(args, "branch", None)
    path = deck_dir(slug, branch) / ARTIFACT
    where = slug + (f"@{branch}" if branch else "")
    if not path.is_file():
        print(f"{where}: no {ARTIFACT} — run `manamap pilot deck-combos {slug}"
              f"{' --branch ' + branch if branch else ''} --write` (absent means absent)")
        return
    doc = load_json(path) or {}
    cards = load_deck_cards(slug, branch).get("cards") or []
    present = {c["name"] for c in cards}
    pool = load_pool()
    if pool is None:
        raise SystemExit(f"{where}: cannot validate {ARTIFACT} without the corpus — run `manamap extract`")
    details = load_combo_details()
    errors = validate(
        slug, doc, branch=branch, present=present,
        identity=commander_identity(slug, branch), pool=pool,
        live_sha=decklist_sha256(slug, branch),
        combo_count=(details.get("meta") or {}).get("combo_count", len(details["combos"])))
    report_errors(f"{where} — {ARTIFACT}", errors)
    if not errors:
        s = doc.get("summary") or {}
        print(f"OK   {where} — {ARTIFACT}: {s.get('included')} line(s), "
              f"{s.get('infinite')} infinite, {s.get('near_total')} one card short")
    sys.exit(1 if errors else 0)
