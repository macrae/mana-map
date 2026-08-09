"""Pilot: mechanically enforce form on an issue plan (STYLEv3 §11).

The `magazine-editor` agent writes packaging decisions and copy as structured
data; this module is the gate that runs before the renderer. Same philosophy as
validate_stack.py: code enforces *form*, humans judge *substance*.

Checks:
- issue.json carries the full authored identity block (no generated dates)
- every department in the canonical list is present, in canonical order
- copy departments carry kicker + headline + dek
- the cover promises something specific (coverline + >=1 tease)
- components come from the fixed library
- tier costume is never overridden (a department can't claim a badge the
  department system doesn't grant it)
- pilot tips and captions name cards that actually exist in the deck
- a featured artist actually painted a card in the deck
"""

import json

from manamap.pilot.common import deck_dir, load_deck_cards, report_errors
from manamap.pilot.issue_spec import (
    BREATHER_AFTER,
    COMPONENTS,
    DENSE_MODES,
    FURNITURE_KEYS,
    NO_FURNITURE_DEPARTMENTS,
    DEPARTMENT_BY_ID,
    DEPARTMENT_IDS,
    MODE,
    REQUIRED_ISSUE_KEYS,
)

REQUIRED_PLAN_KEYS = {"slug", "angle", "cover", "departments"}
REQUIRED_COPY_KEYS = {"kicker", "headline", "dek"}
MAX_VIOLATORS_PER_SPREAD = 2


def validate_identity(issue, deck_sha256=None):
    """Check data/decks/<slug>/issue.json. Returns error strings.

    `deck_sha256` is cards.json's decklist hash when available: the issue's
    stamped hash must match it, or the manual is being rebuilt against a
    decklist this issue never described.
    """
    errors = []
    missing = REQUIRED_ISSUE_KEYS - set(issue)
    if missing:
        errors.append(f"issue.json missing keys: {sorted(missing)}")
    volume = issue.get("volume")
    if not isinstance(volume, int) or volume < 1:
        errors.append(f"issue.json volume must be a positive integer, got {volume!r}")
    stamped = issue.get("decklist_sha256")
    if deck_sha256 and stamped and stamped != deck_sha256:
        errors.append(
            f"issue.json decklist_sha256 {stamped[:12]} does not match cards.json "
            f"{deck_sha256[:12]} — the decklist changed after this issue was "
            f"published; stamp the new hash (and version the deck) deliberately"
        )
    return errors


def validate_plan(plan, card_names=None, artists=None):
    """Check an issue plan. Returns error strings (empty = form holds).

    `card_names` / `artists` of None mean *skip that class of check* — not
    "the deck has no cards". An empty set is the opposite: it makes every
    caption and every featured artist an error. main() passes None only when
    cards.json is unreadable.
    """
    errors = []

    missing = REQUIRED_PLAN_KEYS - set(plan)
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")
        return errors  # too broken to keep going

    if not plan.get("angle"):
        errors.append("angle is required — every issue is about one idea (STYLEv3 §11)")

    # Cover must promise something specific.
    cover = plan.get("cover") or {}
    if not cover.get("dominant_coverline"):
        errors.append("cover.dominant_coverline is required")
    teases = cover.get("teases") or []
    if not teases:
        errors.append("cover.teases must name at least one specific thing in the issue")
    violators = cover.get("violators") or []
    if len(violators) > MAX_VIOLATORS_PER_SPREAD:
        errors.append(
            f"cover has {len(violators)} violators — max {MAX_VIOLATORS_PER_SPREAD} (STYLEv3 §8.4)"
        )

    # Departments: complete and in canonical order.
    departments = plan.get("departments") or []
    seen = [d.get("id") for d in departments]
    unknown = [i for i in seen if i not in DEPARTMENT_BY_ID]
    if unknown:
        errors.append(f"unknown department id(s): {unknown}")
    absent = [i for i in DEPARTMENT_IDS if i not in seen]
    if absent:
        errors.append(
            f"missing department(s): {absent} — all {len(DEPARTMENT_IDS)} "
            f"render every issue")
    ordered = [i for i in seen if i in DEPARTMENT_BY_ID]
    if ordered != [i for i in DEPARTMENT_IDS if i in ordered]:
        errors.append("departments are out of canonical order (STYLEv3 §5)")

    for dept in departments:
        dept_id = dept.get("id")
        spec = DEPARTMENT_BY_ID.get(dept_id)
        if spec is None:
            continue
        where = f"department {dept_id}"

        if spec["needs_copy"]:
            missing_copy = REQUIRED_COPY_KEYS - {k for k in REQUIRED_COPY_KEYS if dept.get(k)}
            if missing_copy:
                errors.append(f"{where}: missing {sorted(missing_copy)}")

        for component in dept.get("components", []):
            if component not in COMPONENTS:
                errors.append(f"{where}: unknown component {component!r}")

        # Furniture the renderer would drop must be rejected, not accepted.
        if dept_id in NO_FURNITURE_DEPARTMENTS:
            carried = [k for k in FURNITURE_KEYS if dept.get(k)]
            if carried:
                errors.append(
                    f"{where}: has a bespoke layout and renders no department "
                    f"furniture — move {sorted(carried)} elsewhere "
                    f"(cover bursts belong in the plan's top-level cover block)"
                )

        # Costume never earns the badge (STYLEv3 §10).
        claimed = dept.get("tiers")
        if claimed is not None and tuple(claimed) != tuple(spec["tiers"]):
            errors.append(
                f"{where}: claims tiers {tuple(claimed)} but the department system "
                f"grants {spec['tiers']} — a department may not restyle its evidence tier"
            )

        if card_names is not None:
            for tip in dept.get("pilot_tips", []):
                card = tip.get("card")
                if card and card not in card_names:
                    errors.append(f"{where}: PILOT TIP names {card!r}, not in the deck")
                if not tip.get("text"):
                    errors.append(f"{where}: PILOT TIP for {card!r} has no text")
            for card in (dept.get("captions") or {}):
                if card not in card_names:
                    errors.append(f"{where}: caption names {card!r}, not in the deck")
            for group in dept.get("roster", []):
                for card in group.get("cards", []):
                    if card not in card_names:
                        errors.append(
                            f"{where}: roster group {group.get('role', '?')!r} names "
                            f"{card!r}, not in the deck"
                        )

        if artists is not None:
            named = [(dept.get("featured") or {}).get("artist")]
            named += [o.get("artist") for o in dept.get("also_worth_noting", [])]
            for artist in [a for a in named if a]:
                if artist not in artists:
                    errors.append(
                        f"{where}: names artist {artist!r}, who painted no card in "
                        f"this deck"
                    )

    # Rhythm: no two dense sections adjacent (STYLEv3 §6) — unless the
    # renderer emits a declared full-bleed breather spread between them.
    for a, b in zip(ordered, ordered[1:]):
        if a in BREATHER_AFTER:
            continue
        if MODE.get(a) in DENSE_MODES and MODE.get(b) in DENSE_MODES:
            errors.append(
                f"rhythm: {a} ({MODE[a]}) and {b} ({MODE[b]}) are both dense and adjacent — "
                f"insert a breather (STYLEv3 §6)"
            )

    return errors


# ── Self-containment (STYLEv3 L10): every issue is the reader's first ────

# Patterns that mark changelog voice. Deliberately narrow: "swap"/"wave"/
# "benched" are legitimate Commander vocabulary and are handled editorially,
# not mechanically. These four have no innocent reading in reader-facing copy.
import re

_CONTINUITY_RE = re.compile(
    r"\bv[1-9]\b"                       # "v2's answer", "V3 added"
    r"|HISTORY\.md"
    r"|\bprevious (?:version|build|list)\b"
    r"|\bearlier build\b"
    r"|\bsuperseded\b",
    re.IGNORECASE,
)

# Plan fields the reader never sees; everything else in a department is copy.
_EDITOR_ONLY_PLAN_KEYS = {"note", "components", "id", "tiers"}


def _walk_strings(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def _lint_strings(doc, label, skip_key=None):
    errors = []
    for path, text in _walk_strings(doc):
        if skip_key and skip_key(path):
            continue
        match = _CONTINUITY_RE.search(text)
        if match:
            errors.append(
                f"{label} [{path}]: changelog voice — {match.group()!r} "
                f"(STYLEv3 L10: every issue is the reader's first)"
            )
    return errors


# -- Land-count truth (the entries-vs-copies trap) -----------------------

# `mana_analysis.lands.entries` counts distinct land CARDS; `lands.total`
# counts physical copies. Eleven Islands are one entry and eleven lands. An
# issue once shipped claiming "18 lands" for a 33-land deck because prose read
# the wrong field, so the number that is never a land count gets linted.
_LAND_COUNT_RE = re.compile(r"\b(\d{1,3})[\s-]lands?\b", re.IGNORECASE)


def validate_land_counts(base, plan):
    """Reader-facing copy may not state the entry count as a land count."""
    path = base / "mana_analysis.json"
    if not path.exists():
        return []
    lands = json.loads(path.read_text()).get("lands", {})
    entries, total = lands.get("entries"), lands.get("total")
    if not entries or entries == total:
        return []  # nothing to confuse

    errors = []
    docs = [(plan, "issue_plan.json")]
    for fname in ("manual_prose.json", "considering.json", "tutor_guide.json"):
        extra = base / fname
        if extra.exists():
            docs.append((json.loads(extra.read_text()), fname))
    for doc, label in docs:
        for where, text in _walk_strings(doc):
            for match in _LAND_COUNT_RE.finditer(text):
                if int(match.group(1)) == entries:
                    errors.append(
                        f"{label} [{where}]: says {match.group()!r}, but "
                        f"{entries} is the count of distinct land CARDS - this "
                        f"deck runs {total} lands. Quote lands.total."
                    )
    return errors


def validate_self_containment(base, plan):
    """Reader-facing text must carry no memory of previous deck versions."""
    errors = []

    def plan_skip(path):
        last = path.rsplit(".", 1)[-1].split("[")[0]
        # departments[].note is editor-facing except under featured/also_worth_noting,
        # which build_manual renders. Keep the mechanical rule simple: skip bare
        # `note` only when it is a department-level key.
        return last in _EDITOR_ONLY_PLAN_KEYS and ".featured" not in path \
            and "also_worth_noting" not in path

    errors += _lint_strings(plan, "issue_plan.json", skip_key=plan_skip)

    path = base / "manual_prose.json"
    if path.exists():
        with open(path) as f:
            errors += _lint_strings(json.load(f), "manual_prose.json")

    decisions = base / "decisions"
    if decisions.exists():
        for dec in sorted(decisions.glob("*.json")):
            with open(dec) as f:
                errors += _lint_strings(json.load(f), f"decisions/{dec.name}")
    return errors


def main(args):
    base = deck_dir(args.slug)
    errors = []

    issue_path = base / "issue.json"
    if not issue_path.exists():
        raise SystemExit(
            f"{issue_path} not found — author the issue identity block first "
            f"(volume, issue_date, cover_price, deck_name, commander, "
            f"cover_tagline, next_issue). See STYLEv3 §4.1."
        )
    deck_sha256 = None
    cards_path = base / "cards.json"
    if cards_path.exists():
        with open(cards_path) as f:
            deck_sha256 = json.load(f).get("decklist_sha256")
    with open(issue_path) as f:
        errors += validate_identity(json.load(f), deck_sha256)

    plan_path = base / "issue_plan.json"
    if not plan_path.exists():
        raise SystemExit(
            f"{plan_path} not found — run the design-issue skill "
            f"(magazine-editor agent) to produce it."
        )
    with open(plan_path) as f:
        plan = json.load(f)

    try:
        deck_cards = load_deck_cards(args.slug)["cards"]
        card_names = {c["name"] for c in deck_cards}
        artists = {c["artist"] for c in deck_cards if c.get("artist")}
    except FileNotFoundError:
        card_names = artists = None
        print("WARN cards.json absent — skipping card-name checks")

    errors += validate_plan(plan, card_names, artists)
    errors += validate_self_containment(base, plan)
    errors += validate_land_counts(base, plan)

    report_errors(f"issue plan for {args.slug}", errors)
    print(
        f"OK   issue plan for {args.slug} — {len(plan['departments'])} departments, "
        f"form holds; angle: {plan['angle'][:60]}"
    )


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-issue <slug>`.")
