"""Pilot: a prescription — one question to the deck doctor, and its answer, in one file.

WHY THIS EXISTS. `diagnosis.json` answers "what limits this deck" deterministically:
same inputs, same diagnosis, no prompt. The workbench asks the doctor *questions* —
"I keep getting wrathed on five", "make it faster", "the Orinda pod is three aggro
decks, what do I cut" — and those are not deterministic over the deck alone. So a
question becomes an ARTIFACT, and each one is deterministic over (deck, question).

The shape is the stack's: an AUTHORED half (`prompt`, `as_of_decklist_sha256`) the
pilot writes with `manamap pilot prescribe <slug> "…"`, and an ANSWERED half
(`reading`, `cut_candidates`, `add_candidates`, `open_questions`, `skeptic`) the
deck-doctor ⇄ deck-skeptic loop fills. The cache routine `prescription:<id>` digests
only the prompt slice (`prompt:self`), so merging the answer never self-invalidates —
exactly what `scenario:self` does for a stack.

Prescriptions ACCUMULATE under `prescriptions/` and are never overwritten by a new
question: they are a record of what was asked and what the deck was told, with the
same history the captain's log has. A later decklist makes an old one *stale* (the
cache says MISS; the validator checks form only) — never wrong.

The id is a hash of the normalized prompt, so asking the same question twice finds
the same file, and asking it again after a swap is the cache's MISS, not a new file.
`add_candidates` is ranked and capped at ten: the retired Short List's "ten cards worth
knowing about, ownership not a criterion" rule lives here now.
"""

import hashlib
import json
import re

from manamap.pilot.common import deck_dir, load_json
from manamap.pilot.deck_notes import decklist_sha256

DIR = "prescriptions"
AGENT_FILE = "deck-doctor-prescribe-{id}.json"
SKEPTIC_FILE = "deck-skeptic-prescribe-{id}.json"
MAX_ADDS = 10
# The answered half: what the loop may write into the file. The prompt and the
# stamp are authored and are NOT in this set, so a merge cannot rewrite the question.
ANSWER_KEYS = ("reading", "axes_engaged", "log_entries_read", "cut_candidates",
               "add_candidates", "open_questions", "gaps")
_WS = re.compile(r"\s+")


def normalize_prompt(text):
    return _WS.sub(" ", str(text or "")).strip()


def prescription_id(prompt):
    """Twelve hex of the normalized prompt — stable across whitespace and retries."""
    return hashlib.sha256(normalize_prompt(prompt).lower().encode("utf-8")).hexdigest()[:12]


def _kebab(text, n=6):
    words = re.sub(r"[^a-z0-9 ]+", " ", normalize_prompt(text).lower()).split()
    return "-".join(words[:n]) or "question"


def find(slug, pid):
    matches = sorted((deck_dir(slug) / DIR).glob(f"{pid}-*.json"))
    return matches[0] if matches else None


def list_all(slug):
    base = deck_dir(slug) / DIR
    return [load_json(p) for p in sorted(base.glob("*.json"))] if base.is_dir() else []


def create(slug, prompt):
    """Write the authored half. Returns (path, created) — an existing file for the
    same question is returned unchanged, never overwritten."""
    text = normalize_prompt(prompt)
    if not text:
        raise SystemExit("a prescription needs a question — "
                         "`manamap pilot prescribe <slug> \"…\"`")
    pid = prescription_id(text)
    existing = find(slug, pid)
    if existing:
        return existing, False
    base = deck_dir(slug) / DIR
    base.mkdir(exist_ok=True)
    path = base / f"{pid}-{_kebab(text)}.json"
    doc = {"slug": slug, "id": pid, "prompt": text,
           "as_of_decklist_sha256": decklist_sha256(slug)}
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return path, True


def merge(slug, pid):
    """Merge the doctor's (and, if present, the skeptic's) handoff into the file —
    answer keys only. The prompt and the stamp are never touched."""
    path = find(slug, pid)
    if path is None:
        raise SystemExit(f"{slug}: no prescription {pid!r} — create it first with "
                         f"`prescribe {slug} \"<question>\"`")
    base = deck_dir(slug)
    source = base / ".agent-out" / AGENT_FILE.format(id=pid)
    if not source.exists():
        raise SystemExit(f"{source} not found — spawn deck-doctor in MODE prescribe "
                         f"for {slug} {pid} first; agents hand off by path.")
    payload = json.loads(source.read_text())
    doc = json.loads(path.read_text())
    merged = [k for k in ANSWER_KEYS if k in payload]
    if not merged:
        raise SystemExit(f"{source} holds none of {ANSWER_KEYS} — refusing to write")
    for k in merged:
        doc[k] = payload[k]
    skeptic = base / ".agent-out" / SKEPTIC_FILE.format(id=pid)
    if skeptic.exists():
        doc["skeptic"] = json.loads(skeptic.read_text())
        merged.append("skeptic")
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return path, merged


def main(args):
    slug = args.slug
    if getattr(args, "merge", None):
        path, merged = merge(slug, args.merge)
        print(f"{slug}: merged {', '.join(merged)} into {path.name}")
        print(f"  next: `manamap pilot validate-prescription {slug} --id {args.merge}`")
        return
    if getattr(args, "list", False) or not getattr(args, "prompt", None):
        docs = list_all(slug)
        if not docs:
            print(f"{slug}: no prescriptions — `manamap pilot prescribe {slug} \"<question>\"`")
            return
        current = decklist_sha256(slug)
        print(f"PRESCRIPTIONS — {slug} ({len(docs)})\n")
        for d in docs:
            answered = "answered" if d.get("add_candidates") is not None else "OPEN"
            verdict = ((d.get("skeptic") or {}).get("verdict") or "—")
            stale = "" if d.get("as_of_decklist_sha256") == current else "  (stale: older decklist)"
            print(f"{d['id']}  {answered:<8} skeptic:{verdict:<5} "
                  f"{len(d.get('add_candidates') or [])} add / "
                  f"{len(d.get('cut_candidates') or [])} cut{stale}")
            print(f"      {d['prompt'][:90]}{'…' if len(d['prompt']) > 90 else ''}")
        return
    path, created = create(slug, args.prompt)
    pid = json.loads(path.read_text())["id"]
    if created:
        print(f"{slug}: opened prescription {pid} → {path.relative_to(deck_dir(slug))}")
    else:
        print(f"{slug}: that question is already prescription {pid} "
              f"({path.name}); not overwritten")
    print(f"  next: `manamap pilot cache-status {slug} --routine prescription:{pid}`, "
          f"then spawn deck-doctor MODE prescribe (the /prescribe skill sequences it)")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot prescribe <slug> \"<question>\"`.")
