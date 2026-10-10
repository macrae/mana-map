"""Pilot: edit a bench or brewing deck IN PLACE — with undo, a fast rebuild, and a save.

THE PILOT'S RULING (2026-10-10). A deck on the bench is where he experiments:
an add, a cut, a swap or a copy count lands STRAIGHT ON THE DECK, with live
before/after numbers, a full undo history, and the measurements rebuilt in
seconds. "Save version" commits the accumulated edits as one git version with a
note. A SLEEVED deck is never edited directly — its list is cardboard, and a
change to it is a branch that merges when the cards are in the sleeves. An
ARCHIVED deck is revived first.

    manamap pilot edit <slug> [--add N]… [--cut N]… [--swap OUT=IN]… [--set N=Q]…
                              [--side] [--note "…"] [--dry-run] [--no-chain] [--rebuild] [--json]
    manamap pilot edit <slug> undo|redo|history [--json]
    manamap pilot save-version <slug> --note "…" [--json]

ONE VALIDATOR. `plan` folds every op into one change and validates the FINAL
list, so a tray of two cuts and two adds is one legal edit even though no
intermediate state is. `try` (`try_swap.apply_ops`), the preview and `edit`
all read `plan`, so the three cannot disagree about what is legal.

ONE WRITER. `write_list` is the only code that writes a deck's main
`decklist.txt`: an fcntl lock beside it, an atomic temp-then-replace, the
`.txt.bak` that `deck_context.list_change` reads, and a line in the journal.
`check-in`, `--set-printing`, `deck-version restore`, `deck-branch merge` and
`build --overwrite` all write through it (each keeping its own guards), so each
of them can be undone like an edit.

THE JOURNAL (`edits.jsonl`, gitignored — git holds the versions) carries the
full before and after text of every change. Undo and redo REPLAY it under the
lock on every call; nothing is held in memory, so the CLI, `manamap serve` and
Jarvis always agree about where the stack stands. A change made outside the
editor (a hand edit, `git checkout`) is detected by sha and written as an
`external` barrier that undo stops at — the history before it is in git.
"""

import contextlib
import datetime
import fcntl
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from types import SimpleNamespace

from manamap import config
from manamap.pilot import common, formats
from manamap.pilot.common import deck_dir, load_json

JOURNAL = "edits.jsonl"
LOCK = ".edits.lock"
#: Held for the whole of a rebuild, so `save-version` can refuse to commit a
#: half-rebuilt deck. A separate file from LOCK on purpose: an edit takes
#: milliseconds and must not wait twelve seconds behind a rebuild of the edit
#: before it — the rebuild re-reads the list when it finishes instead.
REBUILD_LOCK = ".rebuild.lock"
JOURNAL_VERSION = 1
#: Past this many lines the journal is compacted (see `_rotate`).
ROTATE_AT = 500
#: How long an edit waits for another writer before it refuses. A writer holds
#: the lock for milliseconds; anything longer is a stuck process, named.
LOCK_TIMEOUT = 10.0
#: How long a rebuild waits for another rebuild of the same deck.
REBUILD_TIMEOUT = 300.0

OPS = ("add", "cut", "swap", "set")
BOARDS = ("main", "side")
SOURCES = ("cli", "ui", "jarvis", "check-in", "restore", "build", "merge", "printing")
KINDS = ("edit", "undo", "redo", "external", "save")

STALE = "the list moved since this page loaded — reload"
OUTSIDE = ("decklist.txt changed outside the editor — history before that is in git "
           "(`manamap pilot deck-version {slug} list`)")


def _key(name):
    return str(name or "").split(" // ")[0].strip().lower()


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


# ── the guard ────────────────────────────────────────────────────────────

def guard(slug):
    """Refuse a deck that may not be edited in place; return its stage otherwise.

    `promote.stage` is the one answer: `dev` (brewing) and `bench` pass;
    `sleeved` is refused with the branch path; a lifecycle status (broken
    down, retired, superseded) has no stage at all and is refused with the
    revive command.
    """
    from manamap.pilot import promote

    deck_dir(slug)                     # a missing deck is its own, clearer error
    st = promote.stage(slug)
    if st in (promote.DEV, promote.BENCH):
        return st
    if st == promote.SLEEVED:
        raise SystemExit(
            f"{slug} is SLEEVED — its list is cardboard, so it is never edited in "
            f"place. Change it on a branch: `manamap pilot deck-branch {slug} new "
            f"<name>` (or `manamap pilot try {slug} --out A --in B --stage <name>`), "
            f"measure it, and merge when the cards are in the sleeves.")
    life = common.deck_lifecycle(slug)
    status = life[0] if life else "archived"
    raise SystemExit(
        f"{slug} is archived ({status}) — revive it first: "
        f"`manamap pilot deck-state {slug} revive --reason \"…\"`")


# ── names ────────────────────────────────────────────────────────────────

_NAME_MEMO = {}


def _corpus_lookup():
    """lower-case name -> the corpus's spelling (faces included), or {}.

    Read through `check_in.corpus_names` so a test that patches the corpus
    there patches it here too — one corpus, whichever module asks.
    """
    from manamap.pilot import check_in
    names = check_in.corpus_names()
    if not names:
        return {}
    hit = _NAME_MEMO.get("lookup")
    if hit is None or hit[0] is not names:
        hit = (names, {n.lower(): n for n in names})
        _NAME_MEMO["lookup"] = hit
    return hit[1]


def canonical_name(name):
    """The corpus's spelling of a typed name (`sol ring` -> `Sol Ring`), else as typed.

    The face the pilot typed is kept: a transform card is fetched by its front
    face, and `fetch-deck` cannot resolve every joined `A // B` name.
    """
    name = " ".join(str(name or "").split())
    return _corpus_lookup().get(name.lower(), name)


def _pool_index(pool):
    hit = _NAME_MEMO.get("pool")
    if hit is None or hit[0] is not pool:
        index = {}
        for full in pool:
            index.setdefault(full.lower(), full)
            index.setdefault(_key(full), full)
        hit = (pool, index)
        _NAME_MEMO["pool"] = hit
    return hit[1]


def card_facts(names):
    """name -> {name, type_line, color_identity, cmc} from the corpus pool, for
    the names it knows. {} without a corpus — absent, never guessed."""
    from manamap.pilot import card_pool
    pool = card_pool.load_pool()
    if not pool:
        return {}
    index = _pool_index(pool)
    out = {}
    for n in names:
        full = index.get(str(n).lower()) or index.get(_key(n))
        if full:
            row = pool[full]
            out[n] = {"name": full, "type_line": row.get("type_line", ""),
                      "color_identity": sorted(row.get("color_identity") or []),
                      "cmc": row.get("cmc")}
    return out


def _find(entries, name):
    """The entry for this card, any DFC spelling, any case."""
    from manamap.pilot import deck_branch
    hit = deck_branch._resolve_in_list(entries, name)
    if hit is not None:
        return hit
    want = _key(name)
    return next((e for e in entries if _key(e["name"]) == want), None)


def _is_basic(name):
    from manamap.pilot.deck_manifest import _BASIC_LANDS
    return str(name).lower() in _BASIC_LANDS


# ── ops ──────────────────────────────────────────────────────────────────

def normalize_op(op):
    """One op as a dict with its defaults, or SystemExit naming what is wrong."""
    if not isinstance(op, dict) or op.get("op") not in OPS:
        raise SystemExit(f"an edit op is one of {', '.join(OPS)}: {op!r}")
    o = dict(op)
    o["board"] = o.get("board") or "main"
    if o["board"] not in BOARDS:
        raise SystemExit(f"board is main or side, not {o['board']!r}")
    if o["op"] == "swap":
        if not o.get("out") or not o.get("in"):
            raise SystemExit(f"a swap names the card OUT and the card IN: {op!r}")
        return o
    if not o.get("name"):
        raise SystemExit(f"{o['op']} needs a card name: {op!r}")
    if o["op"] == "set":
        try:
            o["qty"] = int(o.get("qty"))
        except (TypeError, ValueError):
            raise SystemExit(f"set needs a copy count: {op!r}")
        if o["qty"] < 0:
            raise SystemExit(f"a copy count is 0 or more: {op!r}")
        return o
    o["qty"] = int(o.get("qty") or 1)
    if o["qty"] < 1:
        raise SystemExit(f"{o['op']} moves at least one copy: {op!r}")
    return o


def ops_from_args(add=(), cut=(), swap=(), set_=(), side=False):
    """The CLI's flags -> ops, in the order cut, swap, add, set (the result is
    order-independent: `plan` validates the final list)."""
    board = "side" if side else "main"
    ops = [{"op": "cut", "name": n, "board": board} for n in cut or ()]
    for s in swap or ():
        if "=" not in s:
            raise SystemExit(f"--swap takes OUT=IN, not {s!r}")
        o, i = s.split("=", 1)
        ops.append({"op": "swap", "out": o.strip(), "in": i.strip(), "board": board})
    ops += [{"op": "add", "name": n, "board": board} for n in add or ()]
    for s in set_ or ():
        if "=" not in s:
            raise SystemExit(f"--set takes NAME=COPIES, not {s!r}")
        n, q = s.rsplit("=", 1)
        try:
            q = int(q)
        except ValueError:
            raise SystemExit(f"--set {s!r}: {q!r} is not a copy count")
        ops.append({"op": "set", "name": n.strip(), "qty": q, "board": board})
    return [normalize_op(o) for o in ops]


def _copies(entries):
    c = Counter()
    for e in entries:
        c[e["name"]] += int(e.get("quantity") or 1)
    return c


def _delta(before, after):
    out = {n: before[n] - after.get(n, 0) for n in before if before[n] > after.get(n, 0)}
    inn = {n: after[n] - before.get(n, 0) for n in after if after[n] > before.get(n, 0)}
    return dict(sorted(out.items())), dict(sorted(inn.items()))


def plan(slug, ops, branch=None):
    """THE ONE VALIDATOR: fold `ops` into the deck's list and judge the RESULT.

    Returns `{slug, branch, format, base_sha, ops, entries_after, text_after,
    diff: {out, in[, side]}, size, blocking, warnings, keep_list_hits}` and
    writes nothing. `blocking` empty means `edit` may write `text_after`.

    The rules are the ones the repo already has, read from where they live:
    names resolve through `deck_branch` (either face of a DFC); the commander
    cannot be cut (`deck_branch.COMMANDER_REFUSAL`); size, copies, sideboard
    and unknown names are `check_in.analyze` per the deck's `FormatSpec`; the
    keep list reads the COPY diff, so `set X=0` is caught as surely as a cut;
    identity and legality are `validate_deck.validate`, which `analyze` checks
    neither of. An identity or legality error on a card the edit ADDS blocks;
    the same error on a card already in the list is a warning — a ban
    announced last week should not make every edit to the deck impossible.
    """
    from manamap.pilot import check_in, deck_branch, protected

    ops = [normalize_op(o) for o in ops]
    spec = formats.for_deck(slug, branch)
    path = deck_dir(slug, branch) / "decklist.txt"
    if not path.exists():
        raise SystemExit(f"{path} not found — nothing to edit")
    text_before = path.read_text(encoding="utf-8")
    where = slug + (f"@{branch}" if branch else "")
    boards = {"main": [dict(e, board="main") for e in deck_branch._parsed(slug, branch)],
              "side": [dict(e, board="side") for e in deck_branch._sideboard(slug, branch)]}
    before = {b: _copies(boards[b]) for b in BOARDS}
    blocking, warnings, resolved = [], [], []

    for op in ops:
        board, kind = op["board"], op["op"]
        ents = boards[board]
        label = "" if board == "main" else " sideboard"
        if board == "side" and not spec.sideboard_size:
            blocking.append(f"{spec.name} has no sideboard — --side is for a 60-card deck")
            continue
        row = {"op": kind, "board": board}
        if kind in ("cut", "swap"):
            name = op["name"] if kind == "cut" else op["out"]
            q = op.get("qty", 1) if kind == "cut" else 1
            e = _find(ents, name)
            if e is None:
                blocking.append(f"{name!r} is not in {where}{label} — nothing to "
                                f"{'cut' if kind == 'cut' else 'swap out'}")
                continue
            if e.get("is_commander"):
                blocking.append(deck_branch.COMMANDER_REFUSAL.format(name=e["name"]))
                continue
            have = int(e.get("quantity") or 1)
            if q > have:
                blocking.append(f"{where}{label} holds {have} {e['name']} — cannot cut {q}")
                continue
            e["quantity"] = have - q
            row["out" if kind == "swap" else "name"] = e["name"]
            row["qty"] = q
        if kind in ("add", "swap"):
            name = canonical_name(op["name"] if kind == "add" else op["in"])
            q = op.get("qty", 1) if kind == "add" else 1
            e = _find(ents, name)
            if e is not None and e["quantity"] > 0 and spec.singleton and not _is_basic(e["name"]):
                blocking.append(f"{name!r} is already in {where} — {spec.name} is singleton")
                continue
            if e is not None:
                e["quantity"] = int(e.get("quantity") or 0) + q
            else:
                ents.append({"name": name, "quantity": q, "board": board})
            row["in" if kind == "swap" else "name"] = (e or {}).get("name") or name
            row["qty"] = q
        if kind == "set":
            q = op["qty"]
            e = _find(ents, op["name"])
            if e is not None and e.get("is_commander"):
                if q != 1:
                    blocking.append(deck_branch.COMMANDER_REFUSAL.format(name=e["name"]))
                continue
            if e is None and q == 0:
                blocking.append(f"{op['name']!r} is not in {where}{label} — nothing to set to 0")
                continue
            if e is not None:
                e["quantity"] = q
                row["name"] = e["name"]
            else:
                name = canonical_name(op["name"])
                ents.append({"name": name, "quantity": q, "board": board})
                row["name"] = name
            row["qty"] = q
        resolved.append(row)

    after_entries = {b: [e for e in boards[b] if int(e.get("quantity") or 0) > 0] for b in BOARDS}
    after = {b: _copies(after_entries[b]) for b in BOARDS}
    entries_after = after_entries["main"] + after_entries["side"]
    text_after = check_in.render_decklist(entries_after)
    out_main, in_main = _delta(before["main"], after["main"])
    out_side, in_side = _delta(before["side"], after["side"])
    diff = {"out": out_main, "in": in_main}
    if out_side or in_side:
        diff["side"] = {"out": out_side, "in": in_side}

    if not blocking:
        checked = check_in.analyze(slug, text_after, spec=spec)
        blocking += checked["blocking"]
        warnings += checked["warnings"]

    # THE KEEP LIST, on the copy diff: a cut, the OUT side of a swap and a
    # `set` to fewer copies all leave the deck the same way.
    keep_hits = protected.refusals(slug, list(out_main) + list(out_side))
    blocking += keep_hits

    # IDENTITY AND LEGALITY, which `analyze` does not check.
    if not blocking and (in_main or in_side):
        id_errors, id_warnings = _identity_and_legality(slug, branch, spec, after_entries,
                                                        set(in_main) | set(in_side))
        blocking += id_errors
        warnings += id_warnings

    size = {"before": sum(before["main"].values()), "after": sum(after["main"].values())}
    if spec.sideboard_size:
        size.update(side_before=sum(before["side"].values()),
                    side_after=sum(after["side"].values()))
    if not resolved and not blocking:
        blocking.append("no edit given — --add, --cut, --swap OUT=IN or --set NAME=COPIES")
    return {"slug": slug, "branch": branch, "format": spec.name.lower(),
            "base_sha": common.list_sha256(text_before), "ops": resolved,
            "entries_after": entries_after, "text_after": text_after,
            "diff": diff, "size": size, "blocking": blocking, "warnings": warnings,
            "keep_list_hits": keep_hits}


def _identity_and_legality(slug, branch, spec, after_entries, added):
    """(blocking, warnings) from `validate_deck.validate` on the list after."""
    from manamap.pilot import validate_deck
    doc = load_json(deck_dir(slug, branch) / "cards.json") or {}
    by_key = {_key(c["name"]): c for c in (doc.get("cards") or []) + (doc.get("sideboard") or [])}
    unknown = [e["name"] for b in BOARDS for e in after_entries[b] if _key(e["name"]) not in by_key]
    facts = card_facts(unknown)

    def record(e):
        c = by_key.get(_key(e["name"])) or facts.get(e["name"]) or {}
        return {"name": c.get("name") or e["name"], "quantity": int(e.get("quantity") or 1),
                "is_commander": bool(e.get("is_commander")),
                "type_line": c.get("type_line", ""),
                "color_identity": list(c.get("color_identity") or [])}

    cards = [record(e) for e in after_entries["main"]]
    side = [record(e) for e in after_entries["side"]]
    errors = validate_deck.validate({"cards": cards, "sideboard": side}, spec)
    errors = [x for x in errors if x.startswith("Color identity violation")
              or " legality: " in x or x.startswith("cannot check")]
    added_names = set()
    for n in added:
        added_names |= {n, (facts.get(n) or {}).get("name") or n}
    blocking, warnings = [], []
    for x in errors:
        if any(f"{n} is" in x for n in added_names):
            blocking.append(x)
        else:
            warnings.append(f"{x} (already in the list before this edit)")
    blind = sorted(n for n in added if n not in facts and _key(n) not in by_key)
    if blind:
        warnings.append(f"colour identity and legality not checked for {', '.join(blind)} — "
                        f"no corpus record on this machine")
    return blocking, warnings


# ── the lock ─────────────────────────────────────────────────────────────

class Busy(SystemExit):
    """Another process holds the deck's lock past the timeout."""


@contextlib.contextmanager
def locked(directory, what, name=LOCK, timeout=None):
    """An exclusive fcntl lock on `directory/name`, polled up to `timeout`.

    The lock file records who holds it (pid, what, when) so a refusal can say.
    `timeout=0` tries once.
    """
    timeout = LOCK_TIMEOUT if timeout is None else timeout
    path = directory / name
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    try:
                        holder = path.read_text().strip() or "another process"
                    except OSError:
                        holder = "another process"
                    raise Busy(f"{directory.name} is busy ({holder}) — try again when it "
                               f"finishes")
                time.sleep(0.05)
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()}: {what} since {_now()}".encode())
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def is_locked(directory, name=REBUILD_LOCK):
    """Is `name` held right now? Never blocks."""
    path = directory / name
    if not path.exists():
        return False
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


# ── the journal ──────────────────────────────────────────────────────────

def journal_path(slug):
    return deck_dir(slug) / JOURNAL


def read_journal(directory):
    """Every entry in `directory/edits.jsonl`, oldest first. A torn last line
    (a crash mid-append) is skipped, never fatal."""
    path = directory / JOURNAL
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _append(directory, entry):
    with open(directory / JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _new_id():
    return f"{int(time.time() * 1000):x}-{secrets.token_hex(2)}"


def stacks(entries):
    """(undo, redo) — lists of the `edit` entries each would apply, top LAST.

    REPLAYED, NEVER STORED. An edit pushes onto undo and clears redo; an undo
    moves its target to redo; a redo moves it back; an `external` barrier
    clears both, because nothing before it can be undone onto a list that
    moved outside the editor. A `save` changes neither: undo may go past a save.
    """
    by_id = {e["id"]: e for e in entries if e.get("kind") == "edit"}
    undo, redo = [], []
    for e in entries:
        k = e.get("kind")
        if k == "edit":
            undo.append(e)
            redo.clear()
        elif k == "undo":
            t = by_id.get(e.get("of"))
            if undo and t is not None and undo[-1]["id"] == t["id"]:
                redo.append(undo.pop())
        elif k == "redo":
            t = by_id.get(e.get("of"))
            if redo and t is not None and redo[-1]["id"] == t["id"]:
                undo.append(redo.pop())
        elif k == "external":
            undo.clear()
            redo.clear()
    return undo, redo


def _head_sha(entries):
    """The sha the journal says the list is at: the after-sha of the last entry
    that moved it. None for an empty journal."""
    for e in reversed(entries):
        if e.get("kind") in ("edit", "undo", "redo", "external"):
            return e.get("after_sha")
    return None


def _rotate(directory, entries):
    """Past ROTATE_AT lines, rewrite the journal as the SAME stacks, compacted.

    The undo stack (capped at half of ROTATE_AT) is re-written as edits, the
    redo stack as edits followed by the undos that put them there, and the last
    `save` is kept so the rebuild still knows the saved text. Replaying the
    compacted file gives the same `stacks` as replaying the old one, minus the
    oldest undo steps past the cap. The old file is kept as `edits.jsonl.1`.
    """
    if len(entries) < ROTATE_AT:
        return
    undo, redo = stacks(entries)
    undo = undo[-(ROTATE_AT // 2):]
    kept = list(undo) + list(reversed(redo))
    kept += [{"v": JOURNAL_VERSION, "id": _new_id(), "kind": "undo", "of": e["id"],
              "at": _now(), "source": "cli", "note": "compacted",
              "before_sha": e["after_sha"], "after_sha": e["before_sha"],
              "before": e.get("after"), "after": e.get("before")} for e in redo]
    saves = [e for e in entries if e.get("kind") == "save"]
    if saves:
        kept.append(saves[-1])
    path = directory / JOURNAL
    shutil.copy(path, path.with_name(JOURNAL + ".1"))
    _atomic_write(path, "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n"
                                for e in kept))


def _atomic_write(path, text):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _entry(kind, *, source, note=None, ops=None, before, after, of=None, **extra):
    rec = {"v": JOURNAL_VERSION, "id": _new_id(), "kind": kind, "at": _now(),
           "source": source, "note": note, "ops": ops or [],
           "diff": _text_diff(before, after),
           "before_sha": common.list_sha256(before) if before is not None else None,
           "after_sha": common.list_sha256(after) if after is not None else None,
           "before": before, "after": after}
    if of:
        rec["of"] = of
    rec.update(extra)
    return rec


def _text_diff(before, after):
    from manamap.pilot.fetch_deck import parse_mainboard
    if before is None or after is None:
        return {"out": {}, "in": {}}
    o, i = _delta(_copies(parse_mainboard(before)), _copies(parse_mainboard(after)))
    return {"out": o, "in": i}


def _barrier_if_moved(directory, entries, current_text):
    """Append an `external` barrier when the file is not where the journal left
    it. Returns the (possibly extended) entries."""
    head = _head_sha(entries)
    sha = common.list_sha256(current_text) if current_text is not None else None
    if entries and head != sha:
        rec = _entry("external", source="cli", note="decklist.txt changed outside the editor",
                     before=None, after=current_text)
        rec["before_sha"] = head
        _append(directory, rec)
        return entries + [rec]
    return entries


def write_list(slug, text, entry, *, path=None, expect_sha=None):
    """THE writer of a deck's main `decklist.txt`. Returns the journal entry, or
    None when `text` is already what the file holds (nothing written).

    Under the deck's lock: refuse a stale `expect_sha`; write an `external`
    barrier if the file moved outside the editor; copy `.txt.bak` (which
    `deck_context.list_change` reads); write atomically; append `entry` —
    `{kind, source, note, ops}`, filled in with the before and after text.

    `path` is the caller's own resolution of the deck's list — every caller
    already computed it, and a test that patches one module's `deck_dir` must
    not find its write landing in the real deck tree.
    """
    path = path or (deck_dir(slug) / "decklist.txt")
    directory = path.parent
    with locked(directory, f"writing {slug}'s list"):
        before = path.read_text(encoding="utf-8") if path.exists() else None
        before_sha = common.list_sha256(before) if before is not None else None
        if expect_sha and before_sha != expect_sha:
            raise SystemExit(STALE)
        if before == text:
            return None
        entries = _barrier_if_moved(directory, read_journal(directory), before)
        if before is not None:
            shutil.copy(path, path.with_suffix(".txt.bak"))
        _atomic_write(path, text)
        rec = _entry(entry.get("kind") or "edit", source=entry.get("source") or "cli",
                     note=entry.get("note"), ops=entry.get("ops"), before=before, after=text,
                     **{k: v for k, v in entry.items()
                        if k not in ("kind", "source", "note", "ops")})
        _append(directory, rec)
        _rotate(directory, entries + [rec])
        return rec


def replace_list(slug, text, *, source, note=None, guard_fn=None, ops=None, path=None,
                 expect_sha=None):
    """`write_list` behind a caller's own guard — the shared route for every
    writer that is not `edit`. Each keeps its own rules (check-in stays allowed
    on a sleeved deck; a merge on one is the publish path); the lock, the atomic
    write and the journal are the shared half."""
    if guard_fn is not None:
        guard_fn(slug)
    return write_list(slug, text, {"kind": "edit", "source": source, "note": note,
                                   "ops": ops or []}, path=path, expect_sha=expect_sha)


# ── edit / undo / redo ───────────────────────────────────────────────────

def edit(slug, ops, *, source="cli", note=None, expect_sha=None, dry_run=False):
    """Validate `ops` through `plan` and write the result. Returns
    `{plan, written, entry}`; raises SystemExit on any refusal (nothing written).
    The rebuild is the caller's next step (`rebuild`), so an edit returns in
    well under a second."""
    if source not in SOURCES:
        raise SystemExit(f"source is one of {', '.join(SOURCES)}")
    guard(slug)
    p = plan(slug, ops)
    if expect_sha and p["base_sha"] != expect_sha:
        raise SystemExit(STALE)
    if p["blocking"]:
        raise SystemExit("Refusing that edit:\n  - " + "\n  - ".join(p["blocking"]))
    if dry_run:
        return {"plan": p, "written": False, "entry": None}
    rec = write_list(slug, p["text_after"],
                     {"kind": "edit", "source": source, "note": note, "ops": p["ops"]},
                     expect_sha=p["base_sha"])
    return {"plan": p, "written": rec is not None, "entry": rec}


def _step(slug, kind, *, source="cli", expect_sha=None):
    guard(slug)
    path = deck_dir(slug) / "decklist.txt"
    directory = path.parent
    with locked(directory, f"{kind} on {slug}"):
        current = path.read_text(encoding="utf-8")
        sha = common.list_sha256(current)
        if expect_sha and sha != expect_sha:
            raise SystemExit(STALE)
        entries = read_journal(directory)
        moved = bool(entries) and _head_sha(entries) != sha
        entries = _barrier_if_moved(directory, entries, current)
        if moved:
            raise SystemExit(OUTSIDE.format(slug=slug))
        undo, redo = stacks(entries)
        stack = undo if kind == "undo" else redo
        if not stack:
            barrier = any(e.get("kind") == "external" for e in entries)
            raise SystemExit(f"nothing to {kind} on {slug}"
                             + (" — " + OUTSIDE.format(slug=slug) if barrier and kind == "undo"
                                else ""))
        target = stack[-1]
        new_text = target["before"] if kind == "undo" else target["after"]
        if new_text is None:
            raise SystemExit(f"{kind}: that entry has no list to return to "
                             f"(the deck did not exist before it)")
        shutil.copy(path, path.with_suffix(".txt.bak"))
        _atomic_write(path, new_text)
        rec = _entry(kind, source=source, note=target.get("note"), ops=target.get("ops"),
                     before=current, after=new_text, of=target["id"])
        _append(directory, rec)
        _rotate(directory, entries + [rec])
        return rec


def undo(slug, *, source="cli", expect_sha=None):
    """Put back the list before the last edit. Returns the journal entry."""
    return _step(slug, "undo", source=source, expect_sha=expect_sha)


def redo(slug, *, source="cli", expect_sha=None):
    """Re-apply the last undone edit. Returns the journal entry."""
    return _step(slug, "redo", source=source, expect_sha=expect_sha)


def history(slug, limit=30):
    """The journal, newest first, without the full texts, and where the stacks stand."""
    entries = read_journal(deck_dir(slug))
    undo_s, redo_s = stacks(entries)
    current = common.decklist_sha256(slug)
    rows = [{k: e.get(k) for k in ("id", "kind", "at", "source", "note", "diff", "of",
                                   "commit", "version", "before_sha", "after_sha")}
            for e in reversed(entries)][:limit]
    return {"slug": slug, "entries": rows, "undo": len(undo_s), "redo": len(redo_s),
            "in_sync": (not entries) or _head_sha(entries) == current,
            "decklist_sha256": current}


def saved_text(slug):
    """The list as of the last save — the journal's last `save`, else git HEAD's
    blob, else None. What the Deck Context's ins and outs are measured from."""
    for e in reversed(read_journal(deck_dir(slug))):
        if e.get("kind") == "save" and e.get("after") is not None:
            return e["after"]
    rel = _rel(deck_dir(slug) / "decklist.txt")
    r = _git("show", f"HEAD:{rel}", check=False)
    return r.stdout if r.returncode == 0 else None


# ── the rebuild ──────────────────────────────────────────────────────────

def _producer(name):
    """The `main` of one chain stage — one seam, so a test can stub them all."""
    import importlib
    from manamap.pilot import check_in
    return importlib.import_module(f"manamap.pilot.{check_in._CHAIN_MODULES[name]}").main


def _offline(exc):
    try:
        import requests
        if isinstance(exc, requests.exceptions.RequestException):
            return True
    except ImportError:                                    # pragma: no cover
        pass
    return isinstance(exc, (OSError, ConnectionError, TimeoutError))


#: What the rebuild's regen pass leaves out: the two stages its chain just ran,
#: and the Deck Context, which `list_change` refreshes as the last step.
REGEN_SKIP = ("goldfish", "mana-analysis", "context")

#: The regen stages that read nothing another rebuild stage writes, so they run
#: BESIDE the goldfish rather than after it. Proved from the code, 2026-10-10:
#:   deck-combos  reads cards.json + the corpus combo graph        -> combos.json
#:   diagnose     reads cards.json + goldfish_targets.json          -> diagnostic.json
#:                (`diagnostic.run_on`: its own goldfish, seed 20260826, never
#:                goldfish_metrics.json)
#:   benchmark    reads cards.json + card_roles.json (its own frozen-harness
#:                goldfish, seed 42; the module docstring says why it must not
#:                read goldfish_metrics.json)                       -> benchmark.json
#: and the goldfish chain reads cards.json + goldfish_targets.json, the try
#: warm the same plus the model version. Each writes only its own file, and
#: each constructs its own seeded generator, so running them side by side
#: moves no byte — the same argument as `regen`'s parallel-across-targets.
#: `mana-analysis` READS goldfish_metrics.json, so it stays chained behind the
#: goldfish in one worker; `deck-info` composes everything and runs after.
PARALLEL_STAGES = ("deck-combos", "diagnose", "benchmark")

#: Worker processes for the rebuild's middle phase: the goldfish chain, the
#: stages above and the try warm, ~8 s of CPU each on a Commander deck. Four is
#: the number of tasks there can be, and the performance cores of the machine
#: this was measured on. 1 runs them in order in this process (the tests).
REBUILD_JOBS = 4


def _task(job):
    """One producer of the middle phase. Module-level and picklable, for the pool.

    Returns `{"name", "ran", "error", "seconds", "stdout"}` and never raises:
    a failure is a result the parent reports, the way `regen._one` does.
    """
    import contextlib
    import io
    kind, slug, payload = job
    buf, ran, error = io.StringIO(), [], None
    started = time.time()
    with contextlib.redirect_stdout(buf):
        try:
            if kind == "chain":
                for name in payload:
                    _producer(name)(SimpleNamespace(slug=slug, branch=None))
                    ran.append(name)
            elif kind == "regen":
                from manamap.pilot import regen
                module, kwargs = payload
                got = regen._one((module, kwargs, slug, None))
                error = got[2]
            elif kind == "warm":
                from manamap.pilot import diagnostic, try_swap
                try_swap.champion_reading(slug, None, diagnostic.HARNESS["iterations"],
                                          diagnostic.HARNESS["seed"])
        except BaseException as exc:                       # noqa: BLE001 - reported
            error = f"{(payload[len(ran)] + ': ') if kind == 'chain' else ''}" \
                    f"{type(exc).__name__}: {exc}"
    return {"kind": kind, "ran": ran, "error": error,
            "seconds": time.time() - started, "stdout": buf.getvalue(), "pid": os.getpid()}


def _run_tasks(jobs, workers):
    """Every task's result, IN JOB ORDER whatever order they finish in.

    Spawned, not forked: `serve` runs a rebuild on a thread, and forking a
    threaded process can copy a held lock into the child. If the pool cannot
    start or dies, the tasks run here in order and stderr says so — slower,
    never wrong (each task rewrites its whole file, so a re-run is safe)."""
    if workers <= 1 or len(jobs) <= 1:
        return [_task(j) for j in jobs]
    import concurrent.futures
    import multiprocessing
    try:
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                                                    mp_context=ctx) as pool:
            return list(pool.map(_task, jobs))
    except (OSError, concurrent.futures.process.BrokenProcessPool) as exc:
        print(f"  rebuild: the process pool failed ({type(exc).__name__}: {exc}) — "
              f"running the {len(jobs)} task(s) in order instead", file=sys.stderr)
        return [_task(j) for j in jobs]


def rebuild(slug, *, before_text=None, warm=True, echo=None):
    """Re-derive what an edit made stale, in dependency order, this deck only.

    1. `fetch-deck` (`check_in.chain_plan`), alone: everything reads cards.json;
    2. IN PARALLEL (`REBUILD_JOBS` processes), each writing only its own file:
       `goldfish` → `mana-analysis` chained in one worker (a 60-card deck skips
       the goldfish and says why); the deck's EXISTING `deck-combos`,
       `diagnose` and `benchmark` (`PARALLEL_STAGES` says why each is
       independent); and `try`'s champion reading, so the next preview starts
       cached;
    3. `regen.run(slug=…, include_branches=False, only=…)` for the rest of the
       deck's existing artifacts (`deck-info`), which compose all of the above;
    4. `deck_context.list_change(before_text=<the last save>)`, which refreshes
       the context's generated blocks and returns the Keeper line — the
       Keeper itself is offered at save, never run here — then `deck-info`
       once more, because each of the two validates the other.

    Holds `.rebuild.lock`, so a second rebuild waits and `save-version`
    refuses. If the list moved while it ran (an edit landed meanwhile), it
    runs again on the new list, at most twice more.

    MEASURED on meren-recursion, one swap (docs/pilot.md has the table): 44.4 s
    before this sprint, every output byte-identical after.
    """
    from manamap.pilot import check_in, deck_context, regen
    from manamap.progress import Progress
    echo = echo or (lambda *a, **k: None)
    spec = formats.for_deck(slug)
    stages, skipped = check_in.chain_plan(spec)
    directory = deck_dir(slug)
    out = {"slug": slug, "ran": [], "skipped": dict(skipped), "failures": []}
    began = time.time()
    info_planned = False
    with locked(directory, f"rebuilding {slug}", name=REBUILD_LOCK, timeout=REBUILD_TIMEOUT):
        for _pass in range(3):
            sha = common.decklist_sha256(slug)
            ran = []
            # 1. fetch-deck, alone.
            try:
                _producer(stages[0])(SimpleNamespace(slug=slug, branch=None))
                ran.append(stages[0])
            except Exception as exc:                       # noqa: BLE001 - reported
                out["ran"] = ran
                out["behind"] = (
                    f"cards.json is behind the list — `manamap pilot edit {slug} "
                    f"--rebuild` when online"
                    + ("" if _offline(exc) else f" ({type(exc).__name__}: {exc})"))
                break
            # The Deck Context is skipped here because step 4 refreshes it
            # (`list_change` calls `refresh`) — once, not twice.
            kw = dict(slug=slug, include_branches=False, skip=REGEN_SKIP)
            rows = regen.plan(**kw)
            stages_run = [row[0] for row in rows]
            for st, _s, fmt in regen.skipped(slug=slug):
                if st not in REGEN_SKIP:
                    out["skipped"].setdefault(st, f"not modelled for {fmt}")
            # 2. the independent producers, side by side.
            jobs = [("chain", slug, tuple(stages[1:]))] if stages[1:] else []
            jobs += [("regen", slug, (module, kwargs))
                     for st, module, kwargs, _t in rows if st in PARALLEL_STAGES]
            labels = [" → ".join(stages[1:])] * bool(stages[1:]) + [
                st for st, *_ in rows if st in PARALLEL_STAGES]
            if warm and spec.commanders:
                jobs.append(("warm", slug, None))
                labels.append("try warm")
            progress = Progress(f"rebuild {slug}", total=len(jobs), unit="stages").start()
            progress.set(detail=", ".join(labels))
            t_mid = time.time()
            results = _run_tasks(jobs, REBUILD_JOBS)
            progress.advance(len(jobs), failed=sum(r["error"] is not None for r in results))
            progress.finish(ok=all(r["error"] is None for r in results))
            for label, r in zip(labels, results):
                if r["stdout"] and r["kind"] == "chain":
                    sys.stdout.write(r["stdout"])
                echo(f"    {label:34} {'FAILED  ' + r['error'] if r['error'] else 'ok'}"
                     f"      {r['seconds']:5.1f}s")
                if r["kind"] == "chain":
                    ran += r["ran"]
                    if r["error"]:
                        out["failures"].append(r["error"])
                elif r["kind"] == "warm":
                    out["warmed"] = True if r["error"] is None else r["error"]
                elif r["error"]:
                    out["failures"].append(f"{label} {slug}: {r['error']}")
            echo(f"    {'':34} parallel {time.time() - t_mid:5.1f}s")
            out["ran"] = ran
            # 3. anything else planned (nothing today: `net-change` is branch-only).
            # `deck-info` is NOT run here — see step 4.
            late = tuple(st for st in stages_run
                         if st not in PARALLEL_STAGES and st != "deck-info")
            got = regen.run(echo=echo, only=late, **kw) if late else {}
            info_planned = "deck-info" in stages_run
            out["regen"] = {"stages": stages_run, "ran": len(stages_run)}
            out["failures"] += [f"{st} {n}: {e}" for st, n, e in got.get("failures") or []]
            if common.decklist_sha256(slug) == sha:
                break
        if before_text is None:
            before_text = saved_text(slug)
        context = deck_context.list_change(slug, before_text=before_text)
        # 4. `deck-info` ONCE, after the context refresh. `info.json` validates the
        # Deck Context, so an info written before the refresh judged the OLD context
        # (it read "CONTEXT.md fails its gate" after an edit and an undo,
        # 2026-10-10). The refresh does not read `info.json` — its blocks come from
        # `deck_info.compose` directly — so the info the rebuild used to write
        # before it (and then overwrite) was 3.8 s of work no byte depended on.
        if "behind" not in out and (info_planned or (deck_dir(slug) / "info.json").exists()):
            regen.run(echo=echo, slug=slug, include_branches=False, only=("deck-info",))
    out["context"] = context
    out["keeper"] = (context or {}).get("keeper") if (context or {}).get("stale") else None
    out["seconds"] = round(time.time() - began, 1)
    return out


# ── save a version ───────────────────────────────────────────────────────

def _repo_root():
    """The git root — a sibling of `data/`, read from config AT CALL TIME the way
    `serve._build_finish` does, so a test's tmp tree is its own repo."""
    return config.DECKS_DIR.parent.parent


def _rel(path):
    return str(path.resolve().relative_to(_repo_root().resolve()))


def _git(*args, check=True):
    r = subprocess.run(["git", "-C", str(_repo_root()), *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args[:2])} failed: {(r.stderr or r.stdout).strip()}")
    return r


def _git_busy():
    """A sentence when git is mid-merge, -rebase or -cherry-pick, else None."""
    from pathlib import Path
    for marker, what in (("MERGE_HEAD", "merge"), ("REBASE_HEAD", "rebase"),
                         ("rebase-merge", "rebase"), ("rebase-apply", "rebase"),
                         ("CHERRY_PICK_HEAD", "cherry-pick")):
        p = _git("rev-parse", "--git-path", marker, check=False).stdout.strip()
        if p and (Path(p) if os.path.isabs(p) else _repo_root() / p).exists():
            return f"git is mid-{what} — finish or abort it before saving a version"
    return None


def _deck_name(slug):
    from manamap.pilot.fetch_deck import parse_mainboard
    text = (deck_dir(slug) / "decklist.txt").read_text(encoding="utf-8")
    cmd = [e["name"] for e in parse_mainboard(text) if e.get("is_commander")]
    return " & ".join(cmd) if cmd else slug


def _changed_paths(*pathspecs):
    r = _git("ls-files", "-m", "-o", "-d", "--exclude-standard", "--", *pathspecs)
    return sorted({ln for ln in r.stdout.splitlines() if ln.strip()})


def _consistency(slug, echo):
    """Merge's consistency tail, without the ledger. Returns
    `{ran, failures, invalid}`."""
    import importlib
    import io
    from manamap.pilot import deck_branch, regen
    out = {"ran": [], "failures": []}
    got = regen.run(slug=slug, echo=echo)
    out["ran"].append("regen")
    out["failures"] += [f"{st} {n}: {e}" for st, n, e in got.get("failures") or []]
    steps = []
    if (deck_dir(slug) / "deck_map.json").exists():
        steps.append(("deck-map", "manamap.pilot.deck_map", {"slug": slug}))
    if (_repo_root() / "manuals" / "p" / f"{slug}.html").exists():
        steps.append(("build-poh", "manamap.pilot.poh", {"slug": slug}))
    steps.append(("build-index", "manamap.pilot.deck_manifest", {}))
    for name, dotted, kwargs in steps:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                importlib.import_module(dotted).main(SimpleNamespace(**kwargs))
            out["ran"].append(name)
        except Exception as exc:                            # noqa: BLE001 - reported
            out["failures"].append(f"{name}: {type(exc).__name__}: {exc}")
    out["invalid"] = deck_branch._validate_after_merge(slug)
    return out


def _refresh_wanting_decks(slug, moved, echo):
    """Other live decks whose branches want a card that just moved in or out of
    this deck: their branch's sourcing (and so `net_change.json` and the deck's
    `info.json`) names this deck as a holder. Returns the slugs refreshed."""
    from manamap.pilot import deck_branch, regen
    if not moved or not config.DECKS_DIR.is_dir():
        return []
    want = {_key(n) for n in moved}
    touched = []
    for d in sorted(config.DECKS_DIR.iterdir()):
        if d.name == slug or not d.is_dir() or regen.is_retired(d.name):
            continue
        hits = []
        for b in deck_branch.names(d.name):
            try:
                adds = deck_branch.diff(d.name, b).get("add") or []
            except Exception:                               # noqa: BLE001 - a broken branch is not ours
                continue
            if want & {_key(n) for n in adds}:
                hits.append(b)
        if not hits:
            continue
        for b in hits:
            if (d / "branches" / b / "net_change.json").exists():
                regen._one(("manamap.pilot.net_change", {"write": True}, d.name, b))
        if (d / "info.json").exists():
            regen._one(("manamap.pilot.deck_info", {"write": True}, d.name, None))
        touched.append(d.name)
    return touched


def save_version(slug, note, *, echo=None):
    """Commit the deck's accumulated edits as ONE git version with a note.

    Refuses while a rebuild holds the deck, or while git is mid-merge/rebase.
    Runs merge's consistency tail (regen with branches, deck-map and build-poh
    where they exist, build-index, the post-merge validators) without the
    decisions ledger — nothing was proposed, so there is no prediction to
    freeze. Refreshes other decks whose branches want a card that moved. Then
    commits EXACTLY the paths it touched — `git commit --only -- <paths>` — so
    anything else the pilot has staged stays staged and out of this commit.
    The product's own commits carry no attribution trailer.

    Returns `{commit, version, paths, keeper, consistency, refreshed}`.
    """
    from manamap.pilot import deck_context, deck_versions
    echo = echo or (lambda *a, **k: None)
    note = " ".join(str(note or "").split())
    if not note:
        raise SystemExit("save-version needs --note \"…\": the note is the version's subject")
    guard(slug)
    directory = deck_dir(slug)
    if is_locked(directory, REBUILD_LOCK):
        raise SystemExit(f"{slug} is rebuilding — save when it finishes "
                         f"(the commit must carry the figures for this list)")
    busy = _git_busy()
    if busy:
        raise SystemExit(busy)
    with locked(directory, f"saving {slug}"):
        if is_locked(directory, REBUILD_LOCK):
            raise SystemExit(f"{slug} is rebuilding — save when it finishes")
        before_text = saved_text(slug)
        consistency = _consistency(slug, echo)
        list_path = directory / "decklist.txt"
        now = list_path.read_text(encoding="utf-8")
        since = _text_diff(before_text, now)
        refreshed = _refresh_wanting_decks(slug, set(since["out"]) | set(since["in"]), echo)
        context = deck_context.list_change(slug, before_text=before_text)
        deck_rel = _rel(directory)
        specs = [deck_rel, _rel(config.DECKS_DIR / "index.json")]
        poh = _repo_root() / "manuals" / "p" / f"{slug}.html"
        specs.append(str(poh.relative_to(_repo_root())))
        specs += [_rel(config.DECKS_DIR / s) for s in refreshed]
        paths = _changed_paths(*specs)
        if not paths:
            raise SystemExit(f"nothing to save on {slug} — the list and its artifacts "
                             f"match the last commit")
        head = _git("show", f"HEAD:{_rel(list_path)}", check=False)
        moved = _text_diff(head.stdout if head.returncode == 0 else "", now)
        o, i = moved["out"], moved["in"]
        moves = " / ".join(x for x in (
            "+in " + ", ".join(f"{n}" + (f" x{k}" if k > 1 else "") for n, k in i.items()) if i else "",
            "−out " + ", ".join(f"{n}" + (f" x{k}" if k > 1 else "") for n, k in o.items()) if o else "",
        ) if x) or "no change to the list (its artifacts only)"
        message = f"{_deck_name(slug)}: {note}\n\n{moves}\n"
        _git("add", "-A", "--", *paths)
        _git("commit", "--only", "-q", "-m", message, "--", *paths)
        commit = _git("rev-parse", "HEAD").stdout.strip()
        version = deck_versions.report(slug).get("current_version")
        rec = _entry("save", source="cli", note=note, before=before_text, after=now,
                     commit=commit, version=version)
        _append(directory, rec)
    return {"slug": slug, "commit": commit, "version": version, "paths": paths,
            "message": message, "consistency": consistency, "refreshed": refreshed,
            "keeper": (context or {}).get("keeper") if (context or {}).get("stale") else None,
            "context": context}


# ── preview (the instant tier and the goldfish tier live in `try_swap`) ──

def preview(slug, ops, branch=None, goldfish=True):
    """What an edit would do, before it is applied. See `try_swap.preview`."""
    from manamap.pilot import try_swap
    return try_swap.preview(slug, ops, branch=branch, goldfish=goldfish)


# ── CLI ──────────────────────────────────────────────────────────────────

def _fmt_moves(diff):
    parts = [f"- {n}" + (f" x{k}" if k > 1 else "") for n, k in (diff.get("out") or {}).items()]
    parts += [f"+ {n}" + (f" x{k}" if k > 1 else "") for n, k in (diff.get("in") or {}).items()]
    side = diff.get("side") or {}
    parts += [f"- {n} (side)" + (f" x{k}" if k > 1 else "") for n, k in (side.get("out") or {}).items()]
    parts += [f"+ {n} (side)" + (f" x{k}" if k > 1 else "") for n, k in (side.get("in") or {}).items()]
    return parts


def _print_rebuild(r):
    if not r:
        return
    print(f"  rebuilt in {r['seconds']}s: {' → '.join(r['ran']) or 'nothing'}"
          + (f" → regen ({r['regen']['ran']} target(s))" if r.get("regen") else ""))
    for st, why in (r.get("skipped") or {}).items():
        print(f"  skipped {st}: {why}")
    if r.get("behind"):
        print(f"  {r['behind']}")
    for f in r.get("failures") or []:
        print(f"  FAILED {f}")
    if r.get("keeper"):
        print(f"  CONTEXT: the prose is behind the list — offered at save: {r['keeper']}")


def _emit(obj, as_json, text_fn):
    if as_json:
        print(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    else:
        text_fn(obj)


def _main_save(args):
    r = save_version(args.slug, getattr(args, "note", None))

    def text(r):
        print(f"SAVED {args.slug} as V{r['version']} — {r['commit'][:12]}")
        print(f"  {r['message'].splitlines()[0]}")
        print(f"  {len(r['paths'])} path(s) committed, nothing else staged was touched")
        for f in r["consistency"].get("failures") or []:
            print(f"  FAILED {f}")
        for a, why in r["consistency"].get("invalid") or []:
            print(f"  INVALID {a}: {why}")
        if r["refreshed"]:
            print(f"  refreshed branches that want a moved card: {', '.join(r['refreshed'])}")
        if r["keeper"]:
            print(f"  next (Jarvis): {r['keeper']}")
    _emit(r, getattr(args, "json", False), text)


def main(args):
    if getattr(args, "pilot_command", None) == "save-version":
        return _main_save(args)
    slug = args.slug
    as_json = getattr(args, "json", False)
    action = getattr(args, "action", None)
    chain = not getattr(args, "no_chain", False)
    if action in ("undo", "redo"):
        rec = (undo if action == "undo" else redo)(slug)
        r = rebuild(slug) if chain else None

        def text(_):
            print(f"{action.upper()} on {slug}: " + ("; ".join(_fmt_moves(rec["diff"])) or "no change"))
            _print_rebuild(r)
        _emit({"entry": {k: v for k, v in rec.items() if k not in ("before", "after")},
               "rebuild": r}, as_json, text)
        return
    if action == "history":
        h = history(slug)

        def text(h):
            print(f"EDITS — {slug}: undo {h['undo']}, redo {h['redo']}"
                  + ("" if h["in_sync"] else " — the list moved outside the editor"))
            for e in h["entries"]:
                moves = "; ".join(_fmt_moves(e.get("diff") or {}))
                extra = f" V{e['version']} {str(e.get('commit') or '')[:10]}" if e["kind"] == "save" else ""
                print(f"  {e['at']}  {e['kind']:8} {e['source'] or '':9} {moves[:70]}{extra}"
                      + (f"  — {e['note']}" if e.get("note") else ""))
        _emit(h, as_json, text)
        return
    if action:
        raise SystemExit(f"edit {slug} {action}: the actions are undo, redo and history")
    ops_given = any(getattr(args, k, None) for k in ("add", "cut", "swap", "set"))
    if getattr(args, "rebuild", False) and not ops_given:
        guard(slug)
        r = rebuild(slug)
        _emit(r, as_json, lambda r: (print(f"REBUILD {slug}"), _print_rebuild(r)))
        return
    ops = ops_from_args(args.add, args.cut, args.swap, getattr(args, "set", None),
                        side=getattr(args, "side", False))
    dry = getattr(args, "dry_run", False)
    got = edit(slug, ops, source="cli", note=getattr(args, "note", None), dry_run=dry)
    p = got["plan"]
    r = rebuild(slug) if (chain and got["written"]) else None

    def text(_):
        head = "DRY RUN" if dry else ("EDITED" if got["written"] else "NO CHANGE")
        size = p["size"]
        print(f"{head} {slug}: {size['before']} -> {size['after']} cards"
              + (f", side {size['side_before']} -> {size['side_after']}" if "side_before" in size else ""))
        for line in _fmt_moves(p["diff"]):
            print(f"    {line}")
        for w in p["warnings"]:
            print(f"  warning: {w}")
        if dry:
            print("  nothing written — drop --dry-run to apply")
        elif got["written"]:
            print(f"  undo: `manamap pilot edit {slug} undo` · save: "
                  f"`manamap pilot save-version {slug} --note \"…\"`")
        _print_rebuild(r)
    _emit({"plan": {k: v for k, v in p.items() if k not in ("entries_after",)},
           "written": got["written"],
           "entry": ({k: v for k, v in got["entry"].items() if k not in ("before", "after")}
                     if got["entry"] else None),
           "rebuild": r}, as_json, text)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot edit <slug>`.")
