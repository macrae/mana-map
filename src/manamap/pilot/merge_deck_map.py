"""Pilot: merge the cartographer's names into a deck map — names ONLY.

`deck_map.json` holds two kinds of thing and they have different authors.
Positions, cities, neighbourhoods, membership and the verified flags are a
MEASUREMENT: `deck-map` computes them from the ability embeddings and they are
reproducible from the decklist. Labels and glosses are names someone chose.

A whole-file copy from `.agent-out/` would let an agent's paraphrase of the map
silently replace the map — a card moved to a different city because a language
model thought it belonged there, with every downstream reader then describing
a cluster the embeddings never found. So this writes exactly two keys per region
and refuses everything else, which is the same argument `merge_prose` makes about
two agents sharing one file, one layer down.

Deterministic, no LLM, no network. It reads what an agent already wrote.
"""

import json

from manamap.pilot.common import deck_dir, report_errors

AGENT_FILE = "deck-cartographer.json"
ARTIFACT = "deck_map.json"
OWNED = ("label", "gloss")


def merge(slug):
    """Returns (named, skipped, errors). Raises SystemExit if there is nothing to do."""
    base = deck_dir(slug)
    source = base / ".agent-out" / AGENT_FILE
    target = base / ARTIFACT
    if not source.exists():
        raise SystemExit(
            f"{source} not found — spawn deck-cartographer for {slug} first. "
            f"Agents hand off by path, so the artifact is the contract.")
    if not target.exists():
        raise SystemExit(
            f"{target} not found — run `manamap pilot deck-map {slug}` first. "
            f"There is no map to name.")

    payload = json.loads(source.read_text())
    regions = payload.get("regions")
    if not isinstance(regions, dict) or not regions:
        raise SystemExit(
            f"{source} carries no `regions` map — refusing to write. Merging "
            f"nothing and reporting success is how a map ships with placeholders.")

    doc = json.loads(target.read_text())
    by_id = {r["id"]: r for r in doc.get("regions", [])}

    named, skipped, errors = [], [], []
    for region_id, block in regions.items():
        target_region = by_id.get(region_id)
        if target_region is None:
            errors.append(f"names a region the map does not have: {region_id!r}")
            continue
        if not isinstance(block, dict):
            errors.append(f"{region_id}: expected an object, got {type(block).__name__}")
            continue
        label = (block.get("label") or "").strip()
        if not label:
            skipped.append(region_id)
            continue
        # The two owned keys, and only those. Anything else the agent returned —
        # a card list, a count, a "corrected" parent — is discarded on purpose.
        target_region["label"] = label
        gloss = block.get("gloss")
        target_region["gloss"] = gloss.strip() if isinstance(gloss, str) else None
        named.append(region_id)

    # Every city must end up named; a neighbourhood may keep its deterministic
    # word, which reads as a plain description rather than as a missing name.
    for region in doc.get("regions", []):
        if region["level"] == 0 and not region.get("label"):
            errors.append(f"city left unnamed: {region['id']} "
                          f"(fallback {region.get('fallback')!r})")

    if not named:
        raise SystemExit(f"{source} named no region that exists in the map.")

    doc.setdefault("meta", {})["named_by"] = "deck-cartographer"
    if payload.get("notes"):
        doc["meta"]["cartographer_notes"] = payload["notes"]
    target.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return named, skipped, errors


def main(args):
    named, skipped, errors = merge(args.slug)
    print(f"{args.slug}: named {len(named)} region(s)")
    doc = json.loads((deck_dir(args.slug) / ARTIFACT).read_text())
    for region in doc["regions"]:
        if region["level"] == 0:
            mark = "✓" if region.get("label") else " "
            print(f"  {mark} {region['id']:10s} {region['count']:3d}  "
                  f"{region.get('label') or region.get('fallback')}")
    if skipped:
        print(f"  left with the deterministic word: {', '.join(skipped)}")
    report_errors(f"deck map names for {args.slug}", errors)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot merge-deck-map <slug>`.")


# ── Validation, callable on its own ─────────────────────────────────────


def validate(slug, branch=None):
    """Form-check a named map. Returns a list of errors.

    Checks only what is mechanically checkable — a name's wit is not. What IS
    checkable is the failure that actually happened in review: two places with the
    same name, which makes a reader assume they misread the map, and a
    neighbourhood repeating its parent, which says the split found nothing.
    """
    doc = json.loads((deck_dir(slug, branch) / ARTIFACT).read_text())
    regions = doc.get("regions", [])
    errors = []

    def shown(region):
        return (region.get("label") or region.get("fallback") or "").strip().upper()

    for level, what in ((0, "city"), (1, "neighbourhood")):
        peers = [r for r in regions if r["level"] == level]
        seen = {}
        for region in peers:
            name = shown(region)
            if not name:
                errors.append(f"{region['id']}: {what} has no name at all")
                continue
            if name in seen:
                errors.append(
                    f"{region['id']} and {seen[name]} are both called {name!r} — "
                    f"a map with one name printed twice reads as a misprint")
            seen[name] = region["id"]

    by_id = {r["id"]: r for r in regions}
    siblings = {}
    for region in regions:
        if region["level"] == 1:
            siblings[region.get("parent")] = siblings.get(region.get("parent"), 0) + 1
    for region in regions:
        if region["level"] != 1:
            continue
        parent = by_id.get(region.get("parent"))
        # ONLY when the city actually split. A city with a single neighbourhood IS
        # that neighbourhood — same cards, so the same name is correct, and the
        # deterministic fallback produces it by construction. Flagging that fires on
        # five of radagast's seven cities while nothing is wrong, and a check that
        # fails on accurate data is how a suite teaches people to ignore red.
        # And only when a HUMAN-FACING NAME was chosen. The deterministic fallback
        # is a description, not a name, and describing a sub-cluster the same way
        # as its parent is simply accurate — "Bodies" inside "Bodies" is what the
        # role tags say. The obligation to be narrower falls on whoever names it.
        if not region.get("label"):
            continue
        if parent and siblings.get(parent["id"], 0) > 1 and shown(region) == shown(parent):
            errors.append(
                f"{region['id']} repeats its city's name ({shown(parent)!r}) while "
                f"sharing it with {siblings[parent['id']] - 1} sibling(s) — a "
                f"neighbourhood is narrower than its city or it is not a place")

    # The ids an agent may not invent, and the cards it may not move.
    total = sum(r["count"] for r in regions if r["level"] == 0)
    if total != len(doc.get("cards", [])):
        errors.append(f"cities hold {total} cards but the map has "
                      f"{len(doc.get('cards', []))} — membership was altered")
    return errors


def main_validate(args):
    branch = getattr(args, "branch", None)
    where = args.slug + (f"@{branch}" if branch else "")
    errors = validate(args.slug, branch)
    doc = json.loads((deck_dir(args.slug, branch) / ARTIFACT).read_text())
    cities = [r for r in doc["regions"] if r["level"] == 0]
    named = sum(1 for r in cities if r.get("label"))
    report_errors(f"deck map for {where}", errors,
                  f"OK   deck map for {where} — {len(cities)} cities "
                  f"({named} agent-named), {len(doc['cards'])} cards placed")
