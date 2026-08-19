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
import pathlib
import re
import sys

from manamap import config
from manamap.config import (
    AGENT_CACHE_FILENAME,
    AGENT_CACHE_VERSION,
    AGENT_COMMON_PROMPT,
    AGENT_PROMPTS_DIR,
    AGENT_ROUTINE_DECISION_AGENT,
    AGENT_ROUTINE_DECISION_INPUTS,
    AGENT_ROUTINE_STACK_AGENT,
    AGENT_ROUTINE_STACK_INPUTS,
    AGENT_ROUTINES,
    CARD_REFS_VERSION,
    CR_RULES_META_PATH,
    PROSE_KEY_INPUTS,
    RESOLVE_MAX_ITERATIONS,
)
from manamap.pilot.common import (
    canonical_json, checker_passed, deck_dir, load_json_memo)

_REPO_ROOT = config.DATA_DIR.parent
_DYNAMIC_RE = re.compile(r"^(stack|decision):(\w+)$")


class UnknownRoutine(ValueError):
    """The routine id isn't in the registry and doesn't match stack:/decision:."""


class MissingInput(ValueError):
    """A required input is absent — the caller must fix that, not spawn."""


# ── Digests ─────────────────────────────────────────────────────────────

_SHA_MEMO = {}


def cached_file_sha256(path):
    """sha256 of a file's bytes (None if absent), memoized per (path, mtime, size).

    The global graphs are ~38MB combined and feed several routines; a
    whole-deck status would otherwise hash them once per routine. An
    unmemoized `file_sha256` sat beside this with an identical body and no
    callers — one hashing function, so there is one answer.
    """
    if not path.exists():
        return None
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _SHA_MEMO:
        _SHA_MEMO[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _SHA_MEMO[key]



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

    pilot-notes writes five keys of a manual_prose.json that also carries
    frozen legacy keys nobody owns; digesting only the owned keys means a
    legacy key is never what makes the routine read EDITED.
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
# Top-level artifact keys that are review apparatus rather than published content.
# Excluded from card_refs for the reason spelled out at the extraction site: a
# checker's notes ABOUT a card are not the artifact USING that card, and treating
# them as such let orchestrator-written audit prose pin cards it merely mentioned.
_REFS_EXCLUDED_KEYS = frozenset({"checker"})

CARD_SEMANTIC_FIELDS = (
    "name", "quantity", "is_commander", "mana_cost", "cmc",
    "type_line", "oracle_text", "colors", "color_identity", "keywords",
    "power", "toughness", "loyalty", "layout",
)


def file_digest_excluding(path, exclude):
    """Hash a JSON file with `exclude` dotted paths removed.

    `goldfish_metrics.json` embeds `meta.decklist_sha256`, so ANY decklist edit
    changed the file's bytes and MISSed every routine that declares it —
    strategic-frame, pilot-notes, tutor-guide and every
    decision — even when not one metric had moved. Observed directly: restoring
    comment lines in a decklist re-MISSed five prose routines whose figures were
    byte-identical. The provenance stamp is worth keeping in the artifact; it just
    is not an input to whether the prose about those figures is still true.
    """
    if not path.exists():
        return None
    doc = load_json_memo(path)
    doc = json.loads(json.dumps(doc))          # deep copy; never mutate the memo
    for dotted in exclude:
        node, *rest = dotted.split(".")
        cursor = doc
        while rest and isinstance(cursor, dict) and node in cursor:
            cursor = cursor[node]
            node, rest = rest[0], rest[1:]
        if isinstance(cursor, dict):
            cursor.pop(node, None)
    return json_sha256(doc)


def cards_semantic_digest(path):
    """Digest only the card facts agents reason about, not how cards look."""
    if not path.exists():
        return None
    doc = load_json_memo(path)
    cards = [
        {k: card.get(k) for k in CARD_SEMANTIC_FIELDS}
        for card in doc.get("cards", [])
    ]
    cards.sort(key=lambda c: str(c.get("name")))
    return json_sha256(cards)


def cards_semantic_card_map(path):
    """Per-card semantic digests: {"<name>": sha}.

    The same rows `cards_semantic_digest` hashes as one opaque value, hashed
    individually — the primitive that lets a MISS name WHICH cards changed
    and lets unreferencing routines report STALE_OK instead. One extra hash
    per card on data already parsed and memoized.

    Keys were once "<name>\\x00<0|1 sideboard>", because one name could occupy
    two zones at once. With no sideboard a name is a name.
    """
    if not path.exists():
        return None
    doc = load_json_memo(path)
    return {
        card.get("name"): json_sha256({k: card.get(k) for k in CARD_SEMANTIC_FIELDS})
        for card in doc.get("cards", [])
    }


def diff_card_maps(old_map, new_map):
    """Changed card NAMES between two per-card maps.

    Returns a sorted list of names. Keys are bare names now; the `\\x00`-suffix
    split is kept so a sidecar written before the sideboard was retired still
    diffs against a current one instead of reporting every card as changed.
    """
    changed_keys = set(old_map or {}).symmetric_difference(set(new_map or {}))
    changed_keys |= {k for k in set(old_map or {}) & set(new_map or {})
                     if (old_map or {})[k] != (new_map or {})[k]}
    return sorted({k.split("\x00", 1)[0] for k in changed_keys})


def agent_prompt_sha256(agent):
    """Hash the agent definition(s) — editing a prompt changes the output.

    `agent` may name a loop ("stack-resolver+rules-checker"); every part is
    hashed so editing either definition invalidates.

    `.claude/agents-common.md` is hashed with EVERY agent: each charter opens by
    reading it, so an edit there changes what every agent produces from identical
    artifacts. Hashing it here rather than listing it per routine means it cannot
    be forgotten on a new routine — a missed transitive edge does not fail, it
    serves a stale pass.
    """
    parts = {"_common": cached_file_sha256(AGENT_COMMON_PROMPT)}
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
    return [path for path in sorted((base / "stacks").glob("*.json"))
            if checker_passed(load_json_memo(path))]


def resolve_inputs(slug, spec):
    """Resolve a spec's input tokens. Returns (entries, extra).

    entries: sorted [{"path", "sha256"}]; sha256 None means "absent, and that
    absence is part of the fingerprint".
    """
    base = deck_dir(slug)
    entries, extra = [], {}

    def add(path, exclude=()):
        entries.append({
            "path": rel(path) + ("!" + ",".join(exclude) if exclude else ""),
            "sha256": (file_digest_excluding(path, exclude) if exclude
                       else cached_file_sha256(path)),
        })

    for token in spec["inputs"]:
        if token.startswith("deck:"):
            name = token[len("deck:"):]
            # `deck:<file>!<dotted.path>[,<dotted.path>]` hashes the file with those
            # paths removed. EXCLUSION, not inclusion, on purpose: an inclusion list
            # silently stops covering any field added to the artifact later, which is
            # a false-negative generator with a long fuse. Naming what does NOT matter
            # keeps everything new covered by default.
            name, _, excl = name.partition("!")
            exclude = tuple(p for p in excl.split(",") if p)
            optional = name.endswith("?")
            path = base / name.rstrip("?")
            if not path.exists() and not optional:
                raise MissingInput(f"{rel(path)} is required by this routine but missing")
            add(path, exclude)
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
        elif token == "strategy:doc":
            extra["strategy_doc_sha256"] = strategy_doc_digest()
        elif token == "rules:version":
            extra["rules_version"] = rules_version()
        else:
            raise UnknownRoutine(f"Unknown input token {token!r}")

    entries.sort(key=lambda e: e["path"])
    return entries, extra


def key_fingerprints(slug, spec, keys):
    """Per-owned-key fingerprints over each key's OWN declared inputs.

    Rides outside the routine fingerprint; consulted on a MISS to name which
    keys are actually stale so a re-spawn can be scoped to just those. Keys
    without a PROSE_KEY_INPUTS entry are omitted (whole-routine staleness).
    """
    out = {}
    for key in sorted(keys):
        tokens = PROSE_KEY_INPUTS.get(key)
        if tokens is None:
            continue
        entries, extra = resolve_inputs(slug, {"agent": spec["agent"],
                                               "artifact": spec["artifact"],
                                               "inputs": tokens})
        out[key] = json_sha256({"key": key,
                                "inputs": sorted(entries, key=lambda e: e["path"]),
                                "extra": extra})
    return out


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
              "rules_version": "rules version changed"}
    for key in sorted(set(old_extra or {}) | set(new_extra or {})):
        if (old_extra or {}).get(key) != (new_extra or {}).get(key):
            changes.append({"path": key, "change": "changed",
                            "note": labels.get(key, "")})
    return changes


def _tutor_guide_applicable(slug):
    """Fetch Quests exists only for decks that actually tutor.

    Zero maindeck library-search tutors → N/A: the renderer prints standing
    copy instead, and there is nothing for the coach to author. Missing
    cards.json defers to the normal missing-input error.
    """
    from manamap.pilot.validate_tutor_guide import deck_tutors

    path = deck_dir(slug) / "cards.json"
    if not path.exists():
        return True  # let the normal missing-input error name the real problem
    return bool(deck_tutors(load_json_memo(path).get("cards", [])))


# routine -> (predicate, reason shown when it does not apply). The whole-deck
# scan turns the raised MissingInput into an N/A row; an explicit --routine
# call exits 2 with the reason. The Short List (`the-ten`) has no gate: every
# deck gets exactly ten, bench-first, pool-filled.
def _debrief_applicable(slug):
    """A debrief needs a log. No entries → N/A rather than MISS: a deck nobody
    has played yet has nothing to annotate, and a permanent MISS there would
    teach the reader to ignore the board."""
    path = deck_dir(slug) / "log.jsonl"
    return path.exists() and any(line.strip() for line in path.read_text().splitlines())


_APPLICABILITY = {
    "debrief": (
        _debrief_applicable,
        "nothing in the captain's log — `deck-notes <slug> add` a game first; "
        "nothing to spawn or cache",
    ),
    "tutor-guide": (
        _tutor_guide_applicable,
        "deck runs zero library-search tutors — Fetch Quests renders its "
        "standing no-tutors copy; nothing to spawn or cache",
    ),
}


def _check_applicable(slug, routine):
    gate = _APPLICABILITY.get(routine)
    if gate and not gate[0](slug):
        raise MissingInput(gate[1])


def status(slug, routine, force=False, cache=None):
    """HIT / EDITED / MISS for one routine. Read-only.

    `cache` lets a whole-deck scan pass the sidecar in once instead of
    re-reading it per routine.
    """
    _check_applicable(slug, routine)
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
        input_changes = diff_inputs(record_entry.get("inputs"), entries)
        extra_changes = _extra_changes(record_entry.get("extra"), extra)
        prompt_changed = (record_entry.get("agent_prompt_sha256")
                          != agent_prompt_sha256(spec["agent"]))
        changed = input_changes + extra_changes

        # Name WHICH cards changed whenever the sidecar's per-card map lines
        # up with this record's deck state — and, when the record carries
        # refs, decide whether any changed card is actually referenced.
        changed_cards = None
        cards_map = (cache.get("cards_map") or {})
        if (any(c["path"] == "cards_semantic" for c in extra_changes)
                and cards_map.get("digest") == (record_entry.get("extra") or {}).get("cards_semantic")):
            cards_path = deck_dir(slug) / "cards.json"
            changed_cards = diff_card_maps(cards_map.get("cards"),
                                           cards_semantic_card_map(cards_path))
            for c in changed:
                if c["path"] == "cards_semantic":
                    shown = ", ".join(changed_cards[:6])
                    more = f" (+{len(changed_cards) - 6} more)" if len(changed_cards) > 6 else ""
                    c["note"] = f"cards changed: {shown}{more}"

        if prompt_changed:
            # Combined routines ("deck-architect+deck-critic") hash several
            # prompts; name the real files, not a nonexistent joined path.
            for agent_name in spec["agent"].split("+"):
                changed.append({"path": f".claude/agents/{agent_name}.md",
                                "change": "changed", "note": "agent prompt edited"})
        if not changed:
            changed = [{"path": "cache_version", "change": "changed",
                        "note": "cache format changed"}]

        # STALE_OK: the ONLY thing that moved is the deck digest, the per-card
        # diff is computable, the record declares its refs, and no changed card
        # is among them. Conservative by construction — the matcher that built
        # the refs over-triggers on purpose, and any other kind of change
        # (inputs, prompt, strategy doc, rules, printing) disqualifies.
        refs = record_entry.get("card_refs")
        if (not input_changes and not prompt_changed
                and extra_changes
                and all(c["path"] == "cards_semantic" for c in extra_changes)
                and refs is not None
                and changed_cards is not None
                and not set(changed_cards) & set(refs)):
            return {**result, "status": "STALE_OK",
                    "reason": ("cards changed but none this artifact references — "
                               "safe to re-bless"),
                    "changed": changed}

        # For keyed routines, name WHICH keys are stale so a re-spawn can be
        # scoped ("revise only these; copy the rest byte-identical"). A key
        # whose only stale input is the deck digest, and whose own refs are
        # disjoint from the changed cards, is not really stale — verify by
        # re-fingerprinting it with the record-time deck digest substituted.
        recorded_kfps = record_entry.get("key_fingerprints")
        if keys and recorded_kfps:
            stale_keys = []
            refs_by_key = record_entry.get("card_refs_by_key") or {}
            recorded_cards_digest = (record_entry.get("extra") or {}).get("cards_semantic")
            for key in sorted(keys):
                tokens = PROSE_KEY_INPUTS.get(key)
                if tokens is None or key not in recorded_kfps:
                    stale_keys.append(key)  # no per-key data — assume stale
                    continue
                k_entries, k_extra = resolve_inputs(
                    slug, {"agent": spec["agent"], "artifact": spec["artifact"],
                           "inputs": tokens})
                k_fp = json_sha256({"key": key,
                                    "inputs": sorted(k_entries, key=lambda e: e["path"]),
                                    "extra": k_extra})
                if k_fp == recorded_kfps[key]:
                    continue
                if (changed_cards is not None
                        and key in refs_by_key
                        and not set(changed_cards) & set(refs_by_key[key])
                        and "cards_semantic" in k_extra):
                    unchanged_fp = json_sha256({
                        "key": key,
                        "inputs": sorted(k_entries, key=lambda e: e["path"]),
                        "extra": {**k_extra, "cards_semantic": recorded_cards_digest},
                    })
                    if unchanged_fp == recorded_kfps[key]:
                        continue  # only the deck digest moved, no referenced card
                stale_keys.append(key)
            result["stale_keys"] = stale_keys

        return {**result, "status": "MISS", "reason": "inputs changed", "changed": changed}

    if artifact_digest(artifact, keys) != record_entry.get("artifact_sha256"):
        return {**result, "status": "EDITED",
                "reason": "artifact hand-edited since it was recorded"}
    return {**result, "status": "HIT", "reason": "inputs unchanged"}


def record(slug, routine):
    """Record the fingerprint that produced the artifact. Refuses bad states."""
    _check_applicable(slug, routine)
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

    # Card refs ride OUTSIDE the fingerprint: they refine invalidation (which
    # cards this artifact actually mentions) without changing what a HIT means.
    # Records without refs — anything written before this existed — keep the
    # classic full-MISS behavior until their next record.
    cards_path = deck_dir(slug) / "cards.json"
    if cards_path.exists() and "cards_semantic" in extra:
        from manamap.pilot.card_refs import (
            artifact_card_refs, artifact_card_refs_by_key, deck_card_names,
        )
        deck_names = deck_card_names(load_json_memo(cards_path))
        # Refs describe what the artifact SAYS, so they are taken over its published
        # content. `checker` is excluded: it is review apparatus, not output, and
        # `scenario_block_digest` already excludes it from the fingerprint for the
        # same reason. Leaving it in made the block that is explicitly untrusted for
        # invalidation the thing that DEFINED what the artifact references — and
        # orchestrator prose written into `checker.iteration_bound_override.reason`
        # named the very cards a swap had just changed, so those stacks could never
        # be STALE_OK for that swap. A note about a card is not a use of it.
        scoped = ({k: v for k, v in doc.items() if k in keys} if keys
                  else {k: v for k, v in doc.items() if k not in _REFS_EXCLUDED_KEYS})
        entry["card_refs"] = artifact_card_refs(scoped, deck_names)
        # Stamp the extraction rules these refs were computed under, so a later
        # fix to the extractor can re-seed them. Outside the fingerprint, like
        # the refs themselves.
        entry["card_refs_version"] = CARD_REFS_VERSION
        if keys:
            entry["card_refs_by_key"] = artifact_card_refs_by_key(doc, keys, deck_names)
    if keys:
        kfps = key_fingerprints(slug, spec, keys)
        if kfps:
            entry["key_fingerprints"] = kfps
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
    # One shared per-card map, stamped with the digest it corresponds to. A
    # record whose extra.cards_semantic matches this digest can have its
    # changed-card set computed later; older records simply can't and fall
    # back to a classic MISS.
    cards_path = deck_dir(slug) / "cards.json"
    if cards_path.exists():
        cache["cards_map"] = {
            "digest": cards_semantic_digest(cards_path),
            "cards": cards_semantic_card_map(cards_path),
        }
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


def rebless(slug):
    """Re-record every STALE_OK routine — and any HIT lacking refs — in one sweep.

    STALE_OK means the deck changed but nothing this artifact references did;
    re-recording refreshes the fingerprint (and refs) without any spawn.
    HIT-without-refs is the migration case: re-recording seeds `card_refs`
    so future card changes can be scoped. Never touches MISS or EDITED —
    those still mean what they always meant.
    """
    reblessed, skipped = [], []
    sidecar = load_cache(slug)

    # Classify EVERY routine against the pristine sidecar BEFORE recording any of
    # them. `record()` rewrites the deck-wide `cards_map` baseline (see the block
    # above it), and STALE_OK requires that baseline to still match the record's
    # own `extra.cards_semantic`. Classifying inside the record loop therefore made
    # the first record poison every routine after it: the baseline had moved on, so
    # `changed_cards` came back None and a genuinely-STALE_OK artifact fell to MISS
    # — permanently, since re-running could never restore the old baseline. The
    # symptom was a sweep that reblessed exactly one routine and reported "nothing
    # to rebless" on decks where six artifacts referenced nothing that had changed.
    # `status()` is read-only, so one pass to plan and one pass to write is safe.
    plan = []
    for routine in discover_routines(slug):
        try:
            result = status(slug, routine, cache=sidecar)
        except (UnknownRoutine, MissingInput):
            continue
        rec = sidecar.get("routines", {}).get(routine) or {}
        # Two migration cases, both pure re-fingerprints with no spawn: a record
        # written before refs existed, and one whose refs predate a change to the
        # extraction rules. Missing version == 0, i.e. older than any bump.
        needs_refs = result["status"] == "HIT" and (
            rec.get("card_refs") is None
            or rec.get("card_refs_version", 0) < CARD_REFS_VERSION)
        if result["status"] == "STALE_OK" or needs_refs:
            plan.append(routine)
        else:
            skipped.append((routine, result["status"]))

    for routine in plan:
        record(slug, routine)
        reblessed.append(routine)
    return reblessed, skipped


def snapshot(slug):
    """Every routine's status and artifact digest, right now.

    Taken BEFORE a change that will move fingerprints, so `rerecord` afterwards
    can tell "this went MISS because I changed the cache format" from "this was
    already stale and needs a real spawn". Without that distinction a bulk
    re-record would freeze genuinely stale artifacts as permanent HITs, which is
    the one failure mode of this tool that produces wrong published content
    rather than merely wasted tokens.
    """
    out, sidecar = {}, load_cache(slug)
    for routine in discover_routines(slug):
        try:
            result = status(slug, routine, cache=sidecar)
        except (UnknownRoutine, MissingInput):
            continue
        rec = (sidecar.get("routines") or {}).get(routine) or {}
        out[routine] = {
            "status": result["status"],
            # The artifact's own digest, not the fingerprint. A fingerprint moves
            # when INPUTS move — which is exactly what the format change does. The
            # artifact digest moves only when the artifact itself was rewritten,
            # which is the thing that must veto a re-record.
            "artifact_sha256": rec.get("artifact_sha256"),
        }
    return out


def rerecord(slug, snap, dry_run=False):
    """Re-fingerprint routines a format change invalidated. Never spawns.

    Two gates, both mandatory:
      1. the routine was HIT in `snap` — it was current before the change;
      2. its artifact is byte-identical to what `snap` recorded — nobody edited
         the content in between.

    A routine failing either gate is left MISS for a human. Re-recording is an
    attestation ("this artifact is still what I would accept"), so the tool
    refuses to make that claim on anything it cannot prove was already good.
    """
    planned, refused = [], []
    if not snap:
        raise MissingInput(
            f"no snapshot entry for {slug} — take one BEFORE the change with "
            f"`manamap pilot cache-snapshot {slug} --out <path>`. Re-recording "
            f"without a baseline cannot distinguish a format change from real staleness."
        )
    sidecar = load_cache(slug)
    for routine in discover_routines(slug):
        try:
            result = status(slug, routine, cache=sidecar)
        except (UnknownRoutine, MissingInput):
            continue
        was = snap.get(routine)
        if result["status"] != "MISS":
            continue                                    # nothing to do
        if was is None:
            refused.append((routine, "absent from the snapshot"))
        elif was.get("status") != "HIT":
            refused.append((routine, f'was {was.get("status")} before the change — real work'))
        else:
            rec = (sidecar.get("routines") or {}).get(routine) or {}
            if rec.get("artifact_sha256") != was.get("artifact_sha256"):
                refused.append((routine, "artifact changed since the snapshot"))
            else:
                planned.append(routine)

    if dry_run:
        return planned, refused, []
    done = []
    for routine in planned:
        record(slug, routine)
        done.append(routine)
    return planned, refused, done


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
        stale_keys = result.get("stale_keys")
        if stale_keys is not None:
            fresh = sorted(set(keys or []) - set(stale_keys))
            lines.append(f'       stale keys: {", ".join(stale_keys) or "(none)"}'
                         + (f' — scope the spawn; keep verbatim: {", ".join(fresh)}'
                            if fresh else ""))
        lines.append(f'       → spawn {result["agent"]}, validate, then: '
                     f'manamap pilot cache-record {result.get("slug", "<slug>")} '
                     f'--routine {result["routine"]}')
    elif result["status"] == "EDITED":
        lines.append("       → do NOT re-spawn; the hand edit wins. "
                     "cache-record to bless it.")
    elif result["status"] == "STALE_OK":
        lines.append(f'       → do NOT spawn; run: manamap pilot cache-rebless '
                     f'{result.get("slug", "<slug>")}')
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

    if command == "cache-snapshot":
        out = pathlib.Path(args.out)
        doc = json.loads(out.read_text()) if out.exists() else {}
        doc[slug] = snapshot(slug)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        hits = sum(1 for v in doc[slug].values() if v["status"] == "HIT")
        print(f"Snapshot {slug}: {hits}/{len(doc[slug])} HIT -> {out}")
        return

    if command == "cache-rerecord":
        snap_path = pathlib.Path(args.snapshot)
        if not snap_path.exists():
            raise SystemExit(f"FAIL snapshot {snap_path} does not exist")
        snap = (json.loads(snap_path.read_text()) or {}).get(slug)
        try:
            planned, refused, done = rerecord(slug, snap, dry_run=args.dry_run)
        except MissingInput as e:
            raise SystemExit(f"FAIL {e}")
        verb = "Would re-record" if args.dry_run else "Re-recorded"
        for routine in planned:
            print(f"  {verb} {routine}")
        if not planned:
            print(f"  {slug}: nothing to re-record")
        for routine, why in refused:
            print(f"  REFUSED {routine} — {why}")
        if refused:
            print(f"  {len(refused)} routine(s) left MISS on purpose — they need a human, "
                  f"not a re-fingerprint.")
        return

    if command == "cache-rebless":
        reblessed, skipped = rebless(slug)
        for routine in reblessed:
            print(f"Reblessed {routine}")
        if not reblessed:
            print("Nothing to rebless — no STALE_OK routines.")
        misses = [r for r, s in skipped if s == "MISS"]
        if misses:
            print(f"Still MISS (real work, spawn these): {', '.join(misses)}")
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
            stale = sum(1 for r in results if r["status"] == "STALE_OK")
            line = f"\n{misses} of {len(results)} applicable routines need a spawn"
            if stale:
                line += f"; {stale} STALE_OK (cache-rebless {slug} to clear)"
            if not_applicable:
                line += f"; {len(not_applicable)} not applicable to this deck."
            else:
                line += "."
            print(line)
    sys.exit(1 if any_miss else 0)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot cache-status|cache-record|cache-clear`.")
