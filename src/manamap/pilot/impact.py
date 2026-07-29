"""Pilot: what does the latest deck change actually touch? Report-only.

The manual regeneration sweep used to run on orchestrator judgment: grep the
prose for cut cards, eyeball which stacks mention a moved card, hand-compare
goldfish figures at three roundings. This module makes that judgment
mechanical — a deterministic, zero-cost report that names every artifact
piece the current deck state has invalidated, so regeneration can be scoped
to exactly those pieces. It NEVER edits anything; figures especially are
detect-and-report only (a number's surrounding sentence is prose, and prose
revision belongs to an agent under the citation discipline).

Sections:
  1. deck diff        — added/removed/changed/zone-moved cards vs the cache's
                        per-card baseline (agent_cache `cards_map`)
  2. reference impact — per artifact (and per owned key), which changed cards
                        it references (card_refs matcher, recomputed live)
  3. figure audit     — goldfish/bracket figures quoted in prose that no
                        longer match the current metrics, rounding-aware
  4. target audit     — goldfish_targets any_of members not in the maindeck
                        (the trap where a benched card silently deflates a rate)
  5. zone framing     — stacks/decisions referencing a zone-moved card, whose
                        scenario framing may need the VERIFIED PRE-SWAP note
"""

import json
import re
import subprocess

from manamap.pilot.agent_cache import (
    cards_semantic_card_map,
    diff_card_maps,
    load_cache,
    rel,
)
from manamap.pilot.card_refs import deck_card_names, text_refs
from manamap.pilot.common import deck_dir, load_json, load_json_memo, mainboard

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


# ── 1. deck diff ─────────────────────────────────────────────────────────


def deck_diff(base, cache):
    cards_path = base / "cards.json"
    current = cards_semantic_card_map(cards_path) or {}
    baseline = (cache.get("cards_map") or {}).get("cards")
    if baseline is None:
        return {"available": False,
                "reason": "no per-card baseline in .agent-cache.json — run "
                          "`manamap pilot cache-rebless <slug>` once to seed it"}
    changed = diff_card_maps(baseline, current)
    stale_records = sorted(
        r for r, e in (cache.get("routines") or {}).items()
        if (e.get("extra") or {}).get("cards_semantic")
        not in (None, (cache.get("cards_map") or {}).get("digest")))
    if not changed and stale_records:
        return {"available": False,
                "reason": ("baseline already advanced past "
                           f"{len(stale_records)} record(s) (a rebless ran before "
                           "impact — run impact FIRST next time): "
                           + ", ".join(stale_records[:6]))}
    zone_moved = sorted({
        key.split("\x00", 1)[0]
        for key in set(baseline) ^ set(current)
        if any(k.split("\x00", 1)[0] == key.split("\x00", 1)[0]
               for k in (set(baseline) ^ set(current)) - {key})
    })
    added = sorted({k.split("\x00", 1)[0] for k in set(current) - set(baseline)}
                   - set(zone_moved))
    removed = sorted({k.split("\x00", 1)[0] for k in set(baseline) - set(current)}
                     - set(zone_moved))
    return {"available": True, "changed": changed, "added": added,
            "removed": removed, "zone_moved": zone_moved}


# ── 2. reference impact ──────────────────────────────────────────────────

# artifact filename -> the manual_prose-style keys to scope by, or None for
# whole-document impact. issue_plan gets department-level treatment below.
_PROSE_KEYS = ("how_it_wins", "combo_lines", "card_roles", "mulligan",
               "upgrades", "threat_assessment", "matchups")


def _hits(doc, names):
    return sorted(text_refs(json.dumps(doc, ensure_ascii=False), names))


def reference_impact(base, changed_names):
    if not changed_names:
        return []
    impact = []

    for sub in ("stacks", "decisions"):
        for path in sorted((base / sub).glob("*.json")) if (base / sub).exists() else []:
            hits = _hits(load_json(path, {}), changed_names)
            if hits:
                impact.append({"artifact": rel(path), "cards": hits})

    for fname in ("strategic_frame.json", "considering.json",
                  "tutor_guide.json", "mana_analysis.json",
                  "sideboard_analysis.json", "upgrade_watch.json",
                  "goldfish_targets.json"):
        doc = load_json(base / fname)
        if doc is None:
            continue
        hits = _hits(doc, changed_names)
        if hits:
            impact.append({"artifact": rel(base / fname), "cards": hits})

    prose = load_json(base / "manual_prose.json")
    if prose:
        for key in _PROSE_KEYS:
            if key not in prose:
                continue
            hits = _hits(prose[key], changed_names)
            if hits:
                impact.append({"artifact": rel(base / "manual_prose.json"),
                               "key": key, "cards": hits})

    plan = load_json(base / "issue_plan.json")
    if plan:
        for dept in plan.get("departments", []):
            hits = _hits(dept, changed_names)
            if hits:
                impact.append({"artifact": rel(base / "issue_plan.json"),
                               "department": dept.get("id", "?"), "cards": hits})
        cover_hits = _hits(plan.get("cover", {}), changed_names)
        if cover_hits:
            impact.append({"artifact": rel(base / "issue_plan.json"),
                           "department": "cover", "cards": cover_hits})
    return impact


# ── 3. figure audit ──────────────────────────────────────────────────────


def canonical_figures(base):
    """Every number the prose might quote, from the deterministic artifacts."""
    figures = set()

    def add(value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        figures.add(round(v, 3))
        figures.add(round(v, 2))
        figures.add(round(v, 1))
        figures.add(round(v))
        if 0 <= v <= 1:  # rates are quoted as percentages at 0 or 1 dp
            figures.add(round(v * 100, 1))
            figures.add(round(v * 100))

    def walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            add(obj)

    goldfish = load_json(base / "goldfish_metrics.json")
    if goldfish:
        walk(goldfish.get("metrics", {}))
    bracket = load_json(base / "bracket_report.json")
    if bracket:
        for key in ("floor", "combo_bracket_floor", "combo_count",
                    "two_card_infinites"):
            add(bracket.get(key))
        add(len(bracket.get("game_changers") or []))
    return figures


def _previous_goldfish(base):
    """The goldfish metrics as last committed, via git — None when unavailable."""
    try:
        out = subprocess.run(
            ["git", "show", f"HEAD:{rel(base / 'goldfish_metrics.json')}"],
            capture_output=True, text=True, timeout=10, cwd=str(base))
        if out.returncode != 0:
            return None
        return json.loads(out.stdout)
    except Exception:
        return None


def figure_audit(base):
    """Prose literals that match figures which are no longer current.

    A literal is flagged when it appears in the OLD metrics' figure set (as
    last committed) but not in the CURRENT set — the mechanical version of
    "this sentence still says 7.932". Rounding-aware on both sides.
    Requires goldfish to have been re-run since the commit; silent otherwise.
    """
    current_doc = load_json(base / "goldfish_metrics.json")
    previous_doc = _previous_goldfish(base)
    if not current_doc or not previous_doc or previous_doc == current_doc:
        return []
    current = canonical_figures(base)

    old_figures = set()
    tmp = dict(previous_doc)

    def add_old(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return
        for r in (round(f, 3), round(f, 2), round(f, 1), round(f)):
            old_figures.add(r)
        if 0 <= f <= 1:
            old_figures.add(round(f * 100, 1))
            old_figures.add(round(f * 100))

    def walk_old(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                walk_old(v)
        elif isinstance(obj, list):
            for v in obj:
                walk_old(v)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            add_old(obj)

    walk_old(tmp.get("metrics", {}))
    stale_candidates = old_figures - current - {0.0, 0, 1, 1.0, 100, 40}

    findings = []
    surfaces = [("manual_prose.json", load_json(base / "manual_prose.json") or {}),
                ("strategic_frame.json", load_json(base / "strategic_frame.json") or {}),
                ("issue_plan.json", load_json(base / "issue_plan.json") or {})]
    for dec in sorted((base / "decisions").glob("*.json")) if (base / "decisions").exists() else []:
        surfaces.append((f"decisions/{dec.name}", load_json(dec, {})))

    for fname, doc in surfaces:
        items = doc.items() if isinstance(doc, dict) else [("", doc)]
        for key, value in items:
            text = json.dumps(value, ensure_ascii=False)
            for match in _NUMBER_RE.finditer(text):
                raw = match.group()
                # Bare 1-2 digit integers are hopelessly ambiguous in prose —
                # they match counts, versions, and version-labeled comparisons
                # ("up from 49%", "9->11 Forest cards"). Only decimal-bearing
                # or 3+-digit literals are unambiguous enough to flag.
                if "." not in raw and len(raw) < 3:
                    continue
                literal = float(raw)
                if literal in stale_candidates:
                    findings.append({"artifact": fname, "key": key,
                                     "stale_figure": match.group()})
                    break  # one flag per key is enough to scope a revision
    return findings


# ── 4. target audit ──────────────────────────────────────────────────────


def target_audit(base):
    doc = load_json(base / "goldfish_targets.json")
    cards = load_json(base / "cards.json")
    if not doc or not cards:
        return []
    main = {c["name"] for c in mainboard(cards.get("cards", []))}
    problems = []
    for target in doc.get("targets", []):
        for group in target.get("need", []):
            ghosts = [n for n in group.get("any_of", []) if n not in main]
            if ghosts:
                problems.append({"target": target.get("label", "?"),
                                 "not_in_maindeck": ghosts})
    return problems


# ── 5. zone framing ──────────────────────────────────────────────────────


def zone_framing(base, zone_moved):
    if not zone_moved:
        return []
    flags = []
    for sub in ("stacks", "decisions"):
        folder = base / sub
        for path in sorted(folder.glob("*.json")) if folder.exists() else []:
            hits = _hits(load_json(path, {}), zone_moved)
            if hits:
                flags.append({"artifact": rel(path), "zone_moved_cards": hits,
                              "note": "review scenario framing — a VERIFIED "
                                      "PRE-SWAP annotation may be needed"})
    return flags


# ── report ───────────────────────────────────────────────────────────────


def analyze(slug):
    base = deck_dir(slug)
    cache = load_cache(slug)
    diff = deck_diff(base, cache)
    changed = diff.get("changed", []) if diff.get("available") else []
    names = deck_card_names(load_json_memo(base / "cards.json")) \
        if (base / "cards.json").exists() else []
    changed_known = [n for n in changed if n in names] or changed
    return {
        "slug": slug,
        "deck_diff": diff,
        "reference_impact": reference_impact(base, changed_known),
        "figure_audit": figure_audit(base),
        "target_audit": target_audit(base),
        "zone_framing": zone_framing(base, diff.get("zone_moved", [])),
    }


def format_report(report):
    lines = [f"{report['slug']}: impact report (deterministic, report-only)"]
    diff = report["deck_diff"]
    if not diff.get("available"):
        lines.append(f"  deck diff: unavailable — {diff['reason']}")
    else:
        lines.append(f"  deck diff: {len(diff['changed'])} changed "
                     f"({len(diff['added'])} added, {len(diff['removed'])} removed, "
                     f"{len(diff['zone_moved'])} zone-moved)")
        for name in diff["changed"][:12]:
            lines.append(f"    ~ {name}")
    if report["reference_impact"]:
        lines.append(f"  reference impact ({len(report['reference_impact'])} artifact piece(s)):")
        for hit in report["reference_impact"]:
            where = hit.get("key") or hit.get("department")
            loc = f"{hit['artifact']}" + (f" [{where}]" if where else "")
            lines.append(f"    ! {loc}: {', '.join(hit['cards'][:5])}")
    else:
        lines.append("  reference impact: none — every artifact is clean of the changed cards")
    if report["figure_audit"]:
        lines.append(f"  figure audit ({len(report['figure_audit'])} stale key(s)):")
        for f in report["figure_audit"]:
            lines.append(f"    # {f['artifact']} [{f['key']}]: quotes {f['stale_figure']}, "
                         f"no longer a current figure")
    else:
        lines.append("  figure audit: clean (or goldfish unchanged since last commit)")
    if report["target_audit"]:
        lines.append("  goldfish targets: NAMES NOT IN MAINDECK (rates silently deflated):")
        for t in report["target_audit"]:
            lines.append(f"    X {t['target']}: {', '.join(t['not_in_maindeck'])}")
    else:
        lines.append("  goldfish targets: all members maindeck")
    for z in report["zone_framing"]:
        lines.append(f"    ? {z['artifact']}: zone-moved {', '.join(z['zone_moved_cards'])} "
                     f"— {z['note']}")
    return "\n".join(lines)


def main(args):
    report = analyze(args.slug)
    if getattr(args, "as_json", False):
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_report(report))


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot impact <slug>`.")
