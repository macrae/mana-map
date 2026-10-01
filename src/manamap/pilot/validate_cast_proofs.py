"""Pilot: form-check `cast_proofs.json` — a branch's proofs that the Forge AI plays its adds.

The gate in the same commit as the artifact (2026-10-01). A proof is a MEASUREMENT under one
harness: the file names the overrides sha, the pilot profile and its content sha, the patch
set and the jar it was taken with, and `simulate` compares that stamp with the run it is
about to make — a proof under another harness is "unproven", never "played". What this
holds the file to: form, the slug and branch it lives under, `as_of` a date, the harness
stamp present, every row carrying the count keys and a verdict in
`cast_check.VERDICTS`. It does NOT fail a file whose rows no longer match the branch's
adds (the list moves under a dated file — the `validate_recon` rule); that is a WARN, and
the gate in `simulate` is what reads "every add proven" against the current list.
"""
import sys
from datetime import date

from manamap.pilot.common import deck_dir, load_json, report_errors
from manamap.sim import cast_check

ARTIFACT = cast_check.PROOFS
REQUIRED = ("slug", "branch", "as_of", "harness", "shell", "cards", "limits")
HARNESS_KEYS = ("overrides", "profile", "profile_sha", "patches", "jar", "forge")
ROW_KEYS = ("drawn_games", "drawn", "cast", "activated", "triggered", "discarded",
            "castable_uncast_turns", "held_castable_games", "held_at_end_games", "verdict", "verdict_word", "classes")


def validate(slug, branch, doc):
    errors, warns = [], []
    for k in REQUIRED:
        if k not in doc:
            errors.append(f"missing required key {k!r}")
    if errors:
        return errors, warns
    if doc["slug"] != slug or doc["branch"] != branch:
        errors.append(f"names {doc['slug']}@{doc['branch']} but lives under {slug}@{branch}")
    try:
        date.fromisoformat(str(doc["as_of"]))
    except (TypeError, ValueError):
        errors.append(f"as_of {doc['as_of']!r} is not an ISO date")
    h = doc.get("harness") or {}
    missing = [k for k in HARNESS_KEYS if k not in h]
    if missing:
        errors.append(f"harness stamp lacks {missing} — a proof that does not say what it was taken under cannot be matched to a run")
    cards = doc.get("cards")
    if not isinstance(cards, dict) or not cards:
        errors.append("cards is empty — a file that proves nothing should not exist")
        return errors, warns
    for name, r in cards.items():
        where = f"cards[{name!r}]"
        if not isinstance(r, dict):
            errors.append(f"{where}: not an object"); continue
        lack = [k for k in ROW_KEYS if k not in r]
        if lack:
            errors.append(f"{where}: lacks {lack}")
            continue
        if r["verdict_word"] not in cast_check.VERDICTS:
            errors.append(f"{where}: verdict {r['verdict_word']!r} not in {list(cast_check.VERDICTS)}")
        if not str(r["verdict"]).startswith(r["verdict_word"]):
            errors.append(f"{where}: verdict sentence does not start with its word")
        for k in ("drawn_games", "cast", "activated", "triggered", "castable_uncast_turns"):
            if not isinstance(r[k], int) or r[k] < 0:
                errors.append(f"{where}: {k} {r[k]!r} is not a count")
        if r["verdict_word"] == "PLAYED" and r["cast"] + r["activated"] == 0:
            errors.append(f"{where}: PLAYED with zero casts and activations")
        if r["verdict_word"] == "HELD" and r["cast"] + r["activated"] > 0:
            errors.append(f"{where}: HELD with {r['cast'] + r['activated']} play(s)")
    try:
        want = set(cast_check.adds(slug, branch))
        have = set(cards)
        if want - have:
            warns.append(f"adds without a row: {sorted(want - have)} — run `forge-cast-check {slug} --branch {branch} --adds --write`")
        if have - want:
            warns.append(f"rows for cards the branch no longer adds: {sorted(have - want)} (the list moved under a dated file)")
    except Exception as exc:                        # noqa: BLE001 - no branch list is the gate's problem, not form
        warns.append(f"could not read the branch's adds: {exc.__class__.__name__}")
    return errors, warns


def main(args):
    branch = getattr(args, "branch", None)
    if not branch:
        raise SystemExit(f"{ARTIFACT} lives on a branch — `--branch <name>`.")
    path = deck_dir(args.slug, branch) / ARTIFACT
    if not path.exists():
        raise SystemExit(f"{path} not found — `manamap pilot forge-cast-check {args.slug} --branch {branch} --adds --write` first.")
    doc = load_json(path) or {}
    errors, warns = validate(args.slug, branch, doc)
    for w in warns:
        print(f"WARN {ARTIFACT} for {args.slug}@{branch}: {w}")
    words = [r.get("verdict_word") for r in (doc.get("cards") or {}).values()]
    report_errors(f"{ARTIFACT} for {args.slug}@{branch}", errors,
                  f"OK   {ARTIFACT} for {args.slug}@{branch} — {len(words)} card(s): "
                  + ", ".join(f"{w} {words.count(w)}" for w in cast_check.VERDICTS if words.count(w)) + " ◆")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-cast-proofs <slug> --branch <name>`.")
