"""Pilot: content-addressed cache for subagent invocations.

The renderer is free; the agents are not. Regenerating a manual re-spawns four
serially-dependent agents (~330k tokens) even when nothing they read has
changed. Every agent's output is already a tracked artifact — this module
records *which inputs produced it*, so a skill can skip the spawn when nothing
moved.

Nothing here talks to an LLM. It answers one question for the orchestrating
session: "have this routine's declared inputs changed since the artifact was
recorded?" The contract is

    check → (miss) spawn → write → validate → record

Record last, after validation: an artifact that failed its validator must never
be recorded, so the next run re-spawns it. `record()` enforces that in code.

The cache never writes to or deletes an artifact. It only ever writes its own
sidecar, `data/decks/<slug>/.agent-cache.json` (tracked — a `git pull` should
transfer someone else's regeneration as a cache hit).
"""

import hashlib
import json
import re
import sys

from manamap import config
from manamap.config import (
    AGENT_CACHE_FILENAME,
    AGENT_CACHE_VERSION,
    AGENT_PROMPTS_DIR,
    AGENT_ROUTINE_DECISION_AGENT,
    AGENT_ROUTINE_DECISION_INPUTS,
    AGENT_ROUTINE_STACK_AGENT,
    AGENT_ROUTINE_STACK_INPUTS,
    AGENT_ROUTINES,
    CR_RULES_META_PATH,
    RESOLVE_MAX_ITERATIONS,
)
from manamap.pilot.common import deck_dir, load_json_memo

_REPO_ROOT = config.DATA_DIR.parent
_DYNAMIC_RE = re.compile(r"^(stack|decision):(\w+)$")


class UnknownRoutine(ValueError):
    """The routine id isn't in the registry and doesn't match stack:/decision:."""


class MissingInput(ValueError):
    """A required input is absent — the caller must fix that, not spawn."""


# ── Digests ─────────────────────────────────────────────────────────────

_SHA_MEMO = {}


def file_sha256(path):
    """sha256 of a file's bytes, or None if it doesn't exist."""
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached_file_sha256(path):
    """file_sha256 memoized on (path, mtime_ns, size) for this process.

    The three global graphs are ~56MB combined and feed several routines; a
    whole-deck status would otherwise hash them once per routine.
    """
    if not path.exists():
        return None
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _SHA_MEMO:
        _SHA_MEMO[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _SHA_MEMO[key]


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def json_sha256(obj):
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def rel(path):
    """Repo-relative posix path, so a tracked sidecar is machine-portable."""
    try:
        return path.resolve().relative_to(_REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def artifact_digest(path, keys=None):
    """Digest an artifact, or only the keys a routine owns.

    coach-prose and writer-prose write the same manual_prose.json but own
    disjoint key sets; digesting only the owned keys keeps them independent.
    """
    if not path.exists():
        return None
    with open(path) as f:
        doc = json.load(f)
    if keys is None:
        return json_sha256(doc)
    return json_sha256({k: doc.get(k) for k in sorted(keys)})


# The cards.json fields agents actually read. Printing metadata (art, artist,
# set, collector number, finishes) is presentation for the renderer — enriching
# it must not cost a 330k-token regeneration of prose that would be identical.
CARD_SEMANTIC_FIELDS = (
    "name", "quantity", "is_commander", "is_sideboard", "mana_cost", "cmc",
    "type_line", "oracle_text", "colors", "color_identity", "keywords",
    "power", "toughness", "loyalty", "layout",
)


def cards_semantic_digest(path):
    """Digest only the card facts agents reason about, not how cards look."""
    if not path.exists():
        return None
    doc = load_json_memo(path)
    cards = [
        {k: card.get(k) for k in CARD_SEMANTIC_FIELDS}
        for card in doc.get("cards", [])
    ]
    cards.sort(key=lambda c: (str(c.get("name")), bool(c.get("is_sideboard"))))
    return json_sha256(cards)


# How cards *look*. Only the magazine-editor reads this (Featured Artist), so
# it is a separate token rather than part of CARD_SEMANTIC_FIELDS — adding it
# there would needlessly invalidate coach, writer, stacks and decisions, none of
# which reason about art.
CARD_PRINTING_FIELDS = (
    "name", "artist", "set", "set_name", "collector_number",
    "border_color", "frame_effects", "finishes", "foil",
)


def cards_printing_digest(path):
    """Digest the printing identity of every card — artist, set, treatment."""
    if not path.exists():
        return None
    doc = load_json_memo(path)
    cards = [
        {k: card.get(k) for k in CARD_PRINTING_FIELDS}
        for card in doc.get("cards", [])
    ]
    cards.sort(key=lambda c: str(c.get("name")))
    return json_sha256(cards)


def prose_shape(path):
    """manual_prose.json's key skeleton with all leaf text dropped.

    The editor packages prose; it doesn't rewrite it. Rewording a paragraph
    must not cost a re-plan, but adding or removing a section must.
    """
    if not path.exists():
        return None
    doc = load_json_memo(path)
    return {k: (sorted(v) if isinstance(v, dict) else None) for k, v in sorted(doc.items())}


def agent_prompt_sha256(agent):
    """Hash the agent definition(s) — editing a prompt changes the output.

    `agent` may name a loop ("stack-resolver+rules-checker"); every part is
    hashed so editing either definition invalidates.
    """
    parts = {}
    for name in str(agent).split("+"):
        parts[name] = cached_file_sha256(AGENT_PROMPTS_DIR / f"{name}.md")
    return json_sha256(parts)


def scenario_block_digest(path):
    """Digest only {title, scenario} of a stack/decision artifact.

    The resolver and checker write `resolution`/`checker` back into the same
    file; fingerprinting the scenario slice alone keeps the loop idempotent.
    """
    if not path.exists():
        return None
    doc = load_json_memo(path)
    return json_sha256({"title": doc.get("title"), "scenario": doc.get("scenario")})


def rules_version():
    if not CR_RULES_META_PATH.exists():
        return None
    return load_json_memo(CR_RULES_META_PATH).get("effective_date")


def strategy_doc_digest():
    """Reuse the canonical helper; tolerate an absent doc."""
    from manamap.pilot.common import strategy_doc_sha256

    try:
        return strategy_doc_sha256()
    except FileNotFoundError:
        return None


# ── Routine specs ───────────────────────────────────────────────────────


def _artifact_for(base, kind, number):
    matches = sorted((base / f"{kind}s").glob(f"{number}-*.json"))
    return matches[0] if matches else None


def routine_spec(slug, routine):
    """Static registry entry, or a synthesized stack:/decision: spec."""
    if routine in AGENT_ROUTINES:
        return dict(AGENT_ROUTINES[routine])
    match = _DYNAMIC_RE.match(routine or "")
    if not match:
        raise UnknownRoutine(
            f"Unknown routine {routine!r}. Known: "
            f"{', '.join(sorted(AGENT_ROUTINES))}, stack:<NNN>, decision:<NNN>"
        )
    kind, number = match.groups()
    artifact = _artifact_for(deck_dir(slug), kind, number)
    if artifact is None:
        raise MissingInput(f"No {kind} artifact numbered {number} under {deck_dir(slug)}")
    if kind == "stack":
        return {"agent": AGENT_ROUTINE_STACK_AGENT, "artifact": artifact.name,
                "artifact_subdir": "stacks", "inputs": list(AGENT_ROUTINE_STACK_INPUTS)}
    return {"agent": AGENT_ROUTINE_DECISION_AGENT, "artifact": artifact.name,
            "artifact_subdir": "decisions", "inputs": list(AGENT_ROUTINE_DECISION_INPUTS)}


def artifact_path(slug, spec):
    base = deck_dir(slug)
    if spec.get("artifact_subdir"):
        return base / spec["artifact_subdir"] / spec["artifact"]
    return base / spec["artifact"]


def discover_routines(slug):
    """Static routines plus every stack:/decision: artifact on disk."""
    base = deck_dir(slug)
    routines = list(AGENT_ROUTINES)
    for kind in ("stack", "decision"):
        for path in sorted((base / f"{kind}s").glob("*.json")):
            number = path.name.split("-", 1)[0]
            routines.append(f"{kind}:{number}")
    return routines


def passing_stacks(base):
    """Only checker-passed stacks are inputs — a failed line can't be published."""
    out = []
    for path in sorted((base / "stacks").glob("*.json")):
        if (load_json_memo(path).get("checker") or {}).get("verdict") == "pass":
            out.append(path)
    return out


def resolve_inputs(slug, spec):
    """Resolve a spec's input tokens. Returns (entries, extra).

    entries: sorted [{"path", "sha256"}]; sha256 None means "absent, and that
    absence is part of the fingerprint".
    """
    base = deck_dir(slug)
    entries, extra = [], {}

    def add(path):
        entries.append({"path": rel(path), "sha256": cached_file_sha256(path)})

    for token in spec["inputs"]:
        if token.startswith("deck:"):
            name = token[len("deck:"):]
            optional = name.endswith("?")
            path = base / name.rstrip("?")
            if not path.exists() and not optional:
                raise MissingInput(f"{rel(path)} is required by this routine but missing")
            add(path)
        elif token == "stacks:passing":
            for path in passing_stacks(base):
                add(path)
        elif token == "decisions:all":
            for path in sorted((base / "decisions").glob("*.json")):
                add(path)
        elif token.startswith("global:") or token.startswith("repo:"):
            path = getattr(config, token.split(":", 1)[1])
            add(path)
        elif token == "scenario:self":
            path = artifact_path(slug, spec)
            if not path.exists():
                raise MissingInput(f"{rel(path)} is required by this routine but missing")
            entries.append({"path": f"{rel(path)}#scenario",
                            "sha256": scenario_block_digest(path)})
        elif token == "cards:semantic":
            path = base / "cards.json"
            if not path.exists():
                raise MissingInput(f"{rel(path)} is required by this routine but missing")
            extra["cards_semantic"] = cards_semantic_digest(path)
        elif token == "cards:printing":
            path = base / "cards.json"
            if not path.exists():
                raise MissingInput(f"{rel(path)} is required by this routine but missing")
            extra["cards_printing"] = cards_printing_digest(path)
        elif token == "strategy:doc":
            extra["strategy_doc_sha256"] = strategy_doc_digest()
        elif token == "prose:shape":
            extra["prose_shape"] = prose_shape(base / "manual_prose.json")
        elif token == "rules:version":
            extra["rules_version"] = rules_version()
        else:
            raise UnknownRoutine(f"Unknown input token {token!r}")

    entries.sort(key=lambda e: e["path"])
    return entries, extra


def fingerprint(routine, spec, entries, extra):
    """Stable, order-independent digest of everything that shapes the output."""
    return json_sha256({
        "cache_version": AGENT_CACHE_VERSION,
        "routine": routine,
        "agent": spec["agent"],
        "agent_prompt_sha256": agent_prompt_sha256(spec["agent"]),
        "inputs": sorted(entries, key=lambda e: e["path"]),
        "extra": extra,
    })


# ── Sidecar ─────────────────────────────────────────────────────────────


def cache_path(slug):
    return deck_dir(slug) / AGENT_CACHE_FILENAME


def load_cache(slug):
    path = cache_path(slug)
    if not path.exists():
        return {"cache_version": AGENT_CACHE_VERSION, "slug": slug, "routines": {}}
    with open(path) as f:
        return json.load(f)


def save_cache(slug, cache):
    """Write the sidecar, skipping identical bytes to keep git status clean."""
    path = cache_path(slug)
    payload = json.dumps(cache, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == payload:
        return False
    path.write_text(payload, encoding="utf-8")
    return True


def diff_inputs(old_entries, new_entries):
    """Name what changed — a MISS you can't explain is a bug report, not a cache."""
    old = {e["path"]: e.get("sha256") for e in old_entries or []}
    new = {e["path"]: e.get("sha256") for e in new_entries or []}
    changes = []
    for path in sorted(set(old) | set(new)):
        before, after = old.get(path, "\0missing"), new.get(path, "\0missing")
        if before == after:
            continue
        if before == "\0missing":
            note = "now passing" if "/stacks/" in path else ""
            changes.append({"path": path, "change": "added", "note": note})
        elif after == "\0missing":
            note = "no longer passing" if "/stacks/" in path else ""
            changes.append({"path": path, "change": "removed", "note": note})
        elif before is None:
            changes.append({"path": path, "change": "appeared", "note": ""})
        elif after is None:
            changes.append({"path": path, "change": "deleted", "note": ""})
        else:
            changes.append({"path": path, "change": "modified", "note": ""})
    return changes


def _extra_changes(old_extra, new_extra):
    changes = []
    labels = {"strategy_doc_sha256": "strategy.md changed",
              "rules_version": "rules version changed",
              "prose_shape": "prose structure changed"}
    for key in sorted(set(old_extra or {}) | set(new_extra or {})):
        if (old_extra or {}).get(key) != (new_extra or {}).get(key):
            changes.append({"path": key, "change": "changed",
                            "note": labels.get(key, "")})
    return changes


def _sideboard_applicable(slug):
    """A deck with no real sideboard card can never have a sideboard analysis.

    Mirrors what sideboard-facts reports as `available: false`: only accessories
    (or nothing) behind the SIDEBOARD marker. Checked here so a whole-deck scan
    reports the routine as N/A instead of a permanent MISS — without this, three
    of four decks could never reach a clean exit 0.
    """
    from manamap.pilot.artist_credits import is_accessory

    path = deck_dir(slug) / "cards.json"
    if not path.exists():
        return True  # let the normal missing-input error name the real problem
    cards = load_json_memo(path).get("cards", [])
    return any(c.get("is_sideboard") and not is_accessory(c) for c in cards)


def status(slug, routine, force=False, cache=None):
    """HIT / EDITED / MISS for one routine. Read-only.

    `cache` lets a whole-deck scan pass the sidecar in once instead of
    re-reading it per routine.
    """
    if routine == "sideboard-analysis" and not _sideboard_applicable(slug):
        raise MissingInput(
            "deck has no analysable sideboard (sideboard-facts reports "
            "available: false) — nothing to spawn or cache")
    spec = routine_spec(slug, routine)
    entries, extra = resolve_inputs(slug, spec)
    current = fingerprint(routine, spec, entries, extra)
    artifact = artifact_path(slug, spec)
    keys = spec.get("artifact_keys")
    result = {
        "routine": routine, "agent": spec["agent"], "artifact": rel(artifact),
        "artifact_keys": sorted(keys) if keys else None,
        "fingerprint": current, "changed": [],
    }
    if cache is None:
        cache = load_cache(slug)
    record_entry = (cache.get("routines") or {}).get(routine)

    if force:
        return {**result, "status": "MISS", "reason": "forced"}
    if record_entry is None:
        return {**result, "status": "MISS", "reason": "no record for this routine"}
    result["recorded_fingerprint"] = record_entry.get("fingerprint")
    if not artifact.exists():
        return {**result, "status": "MISS", "reason": "artifact missing"}
    if record_entry.get("fingerprint") != current:
        changed = diff_inputs(record_entry.get("inputs"), entries)
        changed += _extra_changes(record_entry.get("extra"), extra)
        if record_entry.get("agent_prompt_sha256") != agent_prompt_sha256(spec["agent"]):
            # Combined routines ("deck-architect+deck-critic") hash several
            # prompts; name the real files, not a nonexistent joined path.
            for agent_name in spec["agent"].split("+"):
                changed.append({"path": f".claude/agents/{agent_name}.md",
                                "change": "changed", "note": "agent prompt edited"})
        if not changed:
            changed = [{"path": "cache_version", "change": "changed",
                        "note": "cache format changed"}]
        return {**result, "status": "MISS", "reason": "inputs changed", "changed": changed}

    if artifact_digest(artifact, keys) != record_entry.get("artifact_sha256"):
        return {**result, "status": "EDITED",
                "reason": "artifact hand-edited since it was recorded"}
    return {**result, "status": "HIT", "reason": "inputs unchanged"}


def record(slug, routine):
    """Record the fingerprint that produced the artifact. Refuses bad states."""
    if routine == "sideboard-analysis" and not _sideboard_applicable(slug):
        raise MissingInput(
            "deck has no analysable sideboard (sideboard-facts reports "
            "available: false) — nothing to spawn or cache")
    spec = routine_spec(slug, routine)
    entries, extra = resolve_inputs(slug, spec)
    artifact = artifact_path(slug, spec)
    if not artifact.exists():
        raise MissingInput(f"{rel(artifact)} does not exist — nothing to record")
    keys = spec.get("artifact_keys")
    with open(artifact) as f:
        doc = json.load(f)
    if keys:
        # ALL, not any: a partial artifact must never be frozen as a HIT. The digest
        # below hashes missing keys as None, so recording a 1-of-6 output would make
        # the next run report "current" on a manual that is five sections short.
        missing = [k for k in keys if k not in doc]
        if missing:
            raise MissingInput(
                f"{rel(artifact)} is missing {len(missing)} of this routine's "
                f"{len(keys)} keys: {sorted(missing)} — did the agent finish? "
                f"Recording a partial artifact would freeze it as a permanent HIT."
            )
    entry = {
        "agent": spec["agent"],
        "agent_prompt_sha256": agent_prompt_sha256(spec["agent"]),
        "artifact": rel(artifact),
        "artifact_sha256": artifact_digest(artifact, keys),
        "fingerprint": fingerprint(routine, spec, entries, extra),
        "inputs": entries,
        "extra": extra,
    }
    if keys:
        entry["artifact_keys"] = sorted(keys)
    if routine.startswith("stack:"):
        checker = doc.get("checker")
        if not checker:
            raise MissingInput(
                f"{rel(artifact)} has no checker block — validate before recording"
            )
        entry["verdict"] = checker.get("verdict")
        entry["iterations"] = checker.get("iterations")

        # Make the bound real. RESOLVE_MAX_ITERATIONS lived in config purely so skill
        # markdown could quote it — no Python imported it, so it was enforced by a
        # model reading a number in a heading, and hapatra's stack 001 duly ran to 4.
        # Overriding is still allowed; it just has to be declared and reasoned, which
        # is the difference between a bound and a suggestion.
        iterations = checker.get("iterations")
        # The key was invented ad hoc on hapatra and written as free text, so accept a
        # bare string as well as {"reason": ...}. Either way it must actually say
        # something — an empty override is not a justification.
        override = checker.get("iteration_bound_override")
        if isinstance(override, str):
            override = {"reason": override}
        override = override or {}
        if isinstance(iterations, int) and iterations > RESOLVE_MAX_ITERATIONS:
            if not str(override.get("reason", "")).strip():
                raise MissingInput(
                    f"{rel(artifact)} records {iterations} iterations but "
                    f"RESOLVE_MAX_ITERATIONS is {RESOLVE_MAX_ITERATIONS}. To record it "
                    f"anyway, add checker.iteration_bound_override = "
                    f'{{"reason": "<why this line earned an extra pass>"}}. '
                    f"A bound lifted whenever the checker sounds confident is not a bound."
                )
            entry["iteration_bound_override"] = override

    cache = load_cache(slug)
    cache["cache_version"] = AGENT_CACHE_VERSION
    cache["slug"] = slug
    cache.setdefault("routines", {})[routine] = entry
    wrote = save_cache(slug, cache)
    return entry, wrote


def clear(slug, routine=None):
    cache = load_cache(slug)
    routines = cache.get("routines") or {}
    dropped = sorted(routines) if routine is None else ([routine] if routine in routines else [])
    for name in dropped:
        routines.pop(name, None)
    cache["routines"] = routines
    save_cache(slug, cache)
    return dropped


# ── CLI ─────────────────────────────────────────────────────────────────

_SYMBOL = {"added": "+", "removed": "-", "modified": "~", "appeared": "+",
           "deleted": "-", "changed": "~"}


def format_status(result, verbose=True):
    keys = result.get("artifact_keys")
    suffix = f"[{len(keys)} keys]" if keys else ""
    head = (f'{result["status"]:<6} {result["routine"]:<16} {result["agent"]:<24} '
            f'{result["artifact"].split("/")[-1]}{suffix}')
    if not verbose:
        extra = ""
        if result["status"] == "MISS" and result["changed"]:
            extra = f'   {len(result["changed"])} input(s) changed'
        elif result.get("verdict"):
            extra = f'   verdict: {result["verdict"]}'
        return head + extra
    lines = [head, f'       {result["reason"]}']
    for change in result["changed"]:
        note = f' ({change["note"]})' if change.get("note") else ""
        lines.append(f'       {_SYMBOL.get(change["change"], "~")} {change["path"]}'
                     f' {change["change"]}{note}')
    if result["status"] == "MISS":
        lines.append(f'       → spawn {result["agent"]}, validate, then: '
                     f'manamap pilot cache-record {result.get("slug", "<slug>")} '
                     f'--routine {result["routine"]}')
    elif result["status"] == "EDITED":
        lines.append("       → do NOT re-spawn; the hand edit wins. "
                     "cache-record to bless it.")
    return "\n".join(lines)


def main(args):
    slug = args.slug
    command = args.pilot_command

    if command == "cache-clear":
        dropped = clear(slug, getattr(args, "routine", None))
        print(f"Cleared {len(dropped)} routine record(s): {', '.join(dropped) or '(none)'}")
        return

    if command == "cache-record":
        try:
            entry, wrote = record(slug, args.routine)
        except (UnknownRoutine, MissingInput) as e:
            raise SystemExit(f"FAIL {e}")
        if wrote:
            print(f'Recorded {args.routine} ({entry["agent"]}) — '
                  f'fingerprint {entry["fingerprint"][:12]}, {len(entry["inputs"])} input(s).')
        else:
            print(f"{args.routine} already recorded at this fingerprint — skipping write.")
        return

    explicit = bool(getattr(args, "routine", None))
    routines = [args.routine] if explicit else discover_routines(slug)
    force = getattr(args, "force", False)
    results, not_applicable = [], []

    sidecar = load_cache(slug)
    for routine in routines:
        try:
            result = status(slug, routine, force=force, cache=sidecar)
        except (UnknownRoutine, MissingInput) as e:
            # An explicit --routine with a missing input is exit 2: the caller
            # asked about this routine and must fix the input, not spawn. But in
            # the all-routines scan a routine that simply doesn't apply to this
            # deck is not an error — a hand-built deck has no brief.json and
            # never will, and aborting there would hide every other routine.
            if explicit:
                if getattr(args, "as_json", False):
                    print(json.dumps({"slug": slug, "error": str(e)}, indent=2))
                else:
                    print(f"ERROR  {routine}: {e}")
                sys.exit(2)
            not_applicable.append({"routine": routine, "reason": str(e)})
            continue
        result["slug"] = slug
        results.append(result)

    any_miss = any(r["status"] == "MISS" for r in results)
    if getattr(args, "as_json", False):
        print(json.dumps({"slug": slug, "any_miss": any_miss, "routines": results,
                          "not_applicable": not_applicable},
                         indent=2, ensure_ascii=False))
    else:
        single = len(results) == 1 and not not_applicable
        for result in results:
            print(format_status(result, verbose=single))
        for skipped in not_applicable:
            print(f"N/A    {skipped['routine']:<16} {skipped['reason']}")
        if not single:
            misses = sum(1 for r in results if r["status"] == "MISS")
            print(f"\n{misses} of {len(results)} applicable routines need a spawn"
                  + (f"; {len(not_applicable)} not applicable to this deck." if not_applicable
                     else "."))
    sys.exit(1 if any_miss else 0)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot cache-status|cache-record|cache-clear`.")
