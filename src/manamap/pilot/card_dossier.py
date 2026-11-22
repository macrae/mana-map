"""EVERYTHING THE BENCH KNOWS ABOUT ONE CARD IN ONE DECK, from every source, each
line saying where it came from and what it is worth.

WHY. The judgment layer was unreachable. A swap question about Dawn's Truce had
answers in the captain's log (two games lost to removal), the debrief (a card read
"under" with the pilot's own words), two passing stacks, the engine model, the
recon, the queue and the decisions ledger — and `try` read none of them, printed
the goldfish rows, and the goldfish cannot see a protection spell at all. The
dossier is the query that joins those sources per card, so a study (`study.py`)
can be built from it and a swap screen can print it BEFORE any instrument's row.

WHAT A LINE IS. A fact with `kind`, `text`, `tier` (`verified` ✓ a checker-passed
stack, `data` ◆ a tracked record, `coach` ★ an agent's or a person's judgment),
`source` (the file or command it was read from), `as_of` where the source is dated,
and the `card`. Facts never say what to do. A source that is absent is ABSENT —
listed under `absent` with the reason — never a zero and never a guess.

THE GOLDFISH IS ONE SOURCE HERE, NOT THE FRAME. `coverage` says whether the model
reads the card at all (`seen` / `partial` / `dark` / `invisible`), and the
`coverage_block` every instrument prints first is built from the same call, so a
swap screen and a study can never disagree about what the simulator could see.
"""

import json
import re

from manamap.pilot.common import (
    deck_dir,
    deck_file,
    decklist_sha256,
    expand_faces,
    load_deck_cards,
    load_json,
)

#: The three tiers, the vocabulary `deck_model.TIERS` renders as ✓ ◆ ★.
VERIFIED, DATA, COACH = "verified", "data", "coach"

#: In-deck neighbours in the ability space worth listing (redundancy evidence).
NEIGHBOURS = 3
#: Log passages quoted per card (the pilot's own words, most recent first).
LOG_QUOTES = 4


def _fact(kind, text, tier, source, card, **extra):
    f = {"kind": kind, "text": text, "tier": tier, "source": source, "card": card}
    f.update({k: v for k, v in extra.items() if v is not None})
    return f


def _names_of(card):
    """Every string that names this card in prose: the whole name and each face."""
    return {n.lower() for n in expand_faces(card)} | {card.lower()}


def _mentions(text, card):
    low = (text or "").lower()
    return any(n in low for n in _names_of(card))


_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _sentences_naming(text, card, limit=2):
    """The sentences of a passage that name the card, trimmed; a markdown link
    reads as its text, so a CONTEXT line quotes card names, not URLs."""
    out = []
    text = _MD_LINK.sub(r"\1", text or "")
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        if _mentions(sent, card):
            s = " ".join(sent.split())
            out.append(s if len(s) <= 220 else s[:217] + "…")
            if len(out) >= limit:
                break
    return out


def _card_row(slug, card, branch):
    """The card's row from the branch's (else the deck's) cards.json, or None."""
    for b in ((branch,) if branch else ()) + (None,):
        try:
            doc = load_deck_cards(slug, b)
        except FileNotFoundError:
            continue
        for c in doc.get("cards") or []:
            if (c.get("name") or "").lower() in _names_of(card) or card.lower() in _names_of(c.get("name") or ""):
                return c, b
    return None, None


def _corpus_row(card):
    """The corpus's oracle row for a card the list does not run, or None."""
    try:
        from manamap.pilot.card_pool import corpus_oracle, load_pool
        pool, oracle = load_pool(), corpus_oracle()
    except Exception:                                   # noqa: BLE001 — no corpus here
        return None
    if not pool:
        return None
    for name, rec in pool.items():
        if name.lower() in _names_of(card):
            return {"name": name, "type_line": rec.get("type_line"), "oracle_text": oracle.get(name, ""),
                    "mana_cost": rec.get("mana_cost"), "game_changer": rec.get("game_changer")}
    return None


def dossier(slug, card, branch=None):
    """`{card, canonical, in_deck, in_branch, facts[], absent[]}` — every fact the
    bench holds about one card in one deck. Pure read; nothing is written."""
    base = deck_dir(slug)
    facts, absent = [], []
    row, found_in = _card_row(slug, card, branch)
    canonical = (row or {}).get("name") or (_corpus_row(card) or {}).get("name") or card
    in_deck = _card_row(slug, card, None)[0] is not None
    in_branch = bool(branch) and found_in == branch

    # ── the card itself ───────────────────────────────────────────────────────
    text_row = row or _corpus_row(card)
    if text_row:
        oracle = " | ".join(str(text_row.get("oracle_text") or "").splitlines())
        facts.append(_fact("oracle", f"{text_row.get('type_line') or '?'}"
                           + (f" · {text_row.get('mana_cost')}" if text_row.get("mana_cost") else "")
                           + (f" — {oracle[:300]}" if oracle else ""),
                           DATA, "cards.json" if row else "cards.csv", canonical,
                           game_changer=bool(text_row.get("game_changer")) or None))
    else:
        absent.append({"source": "oracle", "why": "not in the list and no corpus on this machine"})

    # ── the keep list ─────────────────────────────────────────────────────────
    prot = load_json(base / "protected.json") or {}
    for p in (prot.get("cards") or prot.get("protected") or []):
        name = p.get("name") if isinstance(p, dict) else p
        if name and name.lower() in _names_of(canonical):
            facts.append(_fact("protected", "on the pilot's keep list"
                               + (f": {p.get('why')}" if isinstance(p, dict) and p.get("why") else ""),
                               COACH, "protected.json", canonical))

    # ── the captain's log: the pilot's own words, then the debrief's reads ────
    try:
        from manamap.pilot import deck_notes
        log = deck_notes.read_log(slug)
        ann = deck_notes.annotations(slug)
    except SystemExit as bad:
        log, ann = [], {}
        absent.append({"source": "log.jsonl", "why": str(bad)[:120]})
    quoted = 0
    for e in reversed(log):
        if quoted >= LOG_QUOTES:
            break
        for s in _sentences_naming(e.get("text") or "", canonical, limit=1):
            facts.append(_fact("log", f"log {e['id']} ({str(e.get('at'))[:10]}, {e.get('result') or 'no result'}): "
                                      f"“{s}”", COACH, f"log.jsonl#{e['id']}", canonical,
                               as_of=str(e.get("at"))[:10], game=e["id"]))
            quoted += 1
    for gid in sorted(ann):
        for c in (ann[gid] or {}).get("cards") or []:
            if (c.get("card") or "").lower() in _names_of(canonical):
                facts.append(_fact("card_read", f"log {gid}: read {c.get('read')}"
                                   + (f" — {str(c.get('why'))[:200]}" if c.get("why") else ""),
                                   COACH, f"log_annotations.json#{gid}", canonical,
                                   read=c.get("read"), game=gid))
    if not log:
        absent.append({"source": "log.jsonl", "why": "no games logged"})

    # ── the stacks: proven lines that name it ─────────────────────────────────
    from manamap.pilot.deck_branch import stacks_naming
    for hit in stacks_naming(slug, [canonical]):
        tier = VERIFIED if hit.get("checker") == "pass" else COACH
        facts.append(_fact("stack", f"stack {hit['id']} — {hit.get('title') or '?'} "
                                    f"(checker: {hit.get('checker') or 'none'})",
                           tier, f"stacks/{hit['id']}", canonical, stack=hit["id"]))

    # ── the engine model ──────────────────────────────────────────────────────
    eng = load_json(base / "engine.json") or {}
    for st in eng.get("stages") or []:
        if any((n or "").lower() in _names_of(canonical) for n in (st.get("cards") or [])):
            facts.append(_fact("engine_stage", f"engine stage {st.get('stage')}: {st.get('label') or ''}".strip(),
                               COACH, "engine.json", canonical, stage=st.get("stage"),
                               as_of=(eng.get("decklist_sha256") or "")[:12] or None))
    if not eng:
        absent.append({"source": "engine.json", "why": "no engine model"})

    # ── the Deck Context: the prose that names it ─────────────────────────────
    ctx_path = base / "CONTEXT.md"
    if ctx_path.exists():
        from manamap.pilot import deck_context
        secs = deck_context.sections(ctx_path.read_text(encoding="utf-8"))
        for key in ("plays", "pilot", "cards"):
            for s in _sentences_naming(deck_context.authored(secs.get(key) or ""), canonical, limit=2):
                facts.append(_fact("context", f"CONTEXT {key}: {s}", COACH, f"CONTEXT.md#{key}", canonical))
    else:
        absent.append({"source": "CONTEXT.md", "why": "no Deck Context"})

    # ── the recon: other people's lists, dated ────────────────────────────────
    recon = load_json(base / "deck_recon.json") or {}
    for f in recon.get("findings") or []:
        named = any((n or "").lower() in _names_of(canonical) for n in (f.get("cards") or []))
        if named or _mentions(f.get("claim") or "", canonical):
            facts.append(_fact("recon", f"recon ({recon.get('as_of')}): {str(f.get('claim'))[:240]}",
                               COACH, "deck_recon.json", canonical, as_of=recon.get("as_of"),
                               confidence=f.get("confidence")))
    if not recon:
        absent.append({"source": "deck_recon.json", "why": "no dated recon"})

    # ── the queue and the ledger ──────────────────────────────────────────────
    try:
        from manamap.pilot import queue
        for r in queue.rows(queue.read(), deck=slug, everything=True):
            if _mentions(r.get("claim") or "", canonical) or _mentions((r.get("result") or {}).get("answer") or "", canonical):
                res = r.get("result") or {}
                facts.append(_fact("queue", f"{r['id']} {r['state']}: {str(r['claim'])[:160]}"
                                            + (f" → {res.get('verdict')}: {str(res.get('answer'))[:160]}" if res else ""),
                                   COACH, "queue.jsonl", canonical, as_of=r.get("last"), queue_id=r["id"]))
    except Exception as exc:                            # noqa: BLE001
        absent.append({"source": "queue.jsonl", "why": str(exc)[:120]})
    try:
        from manamap.pilot import decisions
        for e in decisions.read(slug):
            hay = json.dumps({k: v for k, v in e.items() if k in ("reason", "forced_reason", "pilot_words", "question", "swap")},
                             ensure_ascii=False)
            if _mentions(hay, canonical):
                facts.append(_fact("decision", f"decision {e['id']} {e['kind']} `{e.get('branch')}` ({str(e.get('at'))[:10]})"
                                               + (f": {str(e.get('reason') or e.get('pilot_words'))[:160]}" if (e.get("reason") or e.get("pilot_words")) else ""),
                                   COACH, "decisions.jsonl", canonical, as_of=str(e.get("at"))[:10]))
    except SystemExit as bad:
        absent.append({"source": "decisions.jsonl", "why": str(bad)[:120]})
    # a branch's own staging note naming the card — the pilot's words at the time
    if branch:
        bdoc = load_json(deck_dir(slug, branch) / "branch.json") or {}
        for s in bdoc.get("staged") or []:
            if (s.get("in") or "").lower() in _names_of(canonical) or (s.get("out") or "").lower() in _names_of(canonical):
                facts.append(_fact("staged", f"staged {s.get('at')}: {s.get('out')} → {s.get('in')}"
                                             + (f" — {str(s.get('why'))[:220]}" if s.get("why") else ""),
                                   COACH, f"branches/{branch}/branch.json", canonical, as_of=s.get("at")))

    # ── dated external evidence ───────────────────────────────────────────────
    prices = load_json(deck_file(slug, "prices.json", branch)) or {}
    pr = (prices.get("cards") or {}).get(canonical)
    if pr and pr.get("nm_cents") is not None:
        facts.append(_fact("price", f"${pr['nm_cents'] / 100:.2f} NM ({pr.get('printing')}) as of {prices.get('as_of')}",
                           DATA, "prices.json", canonical, as_of=prices.get("as_of"), figure=pr["nm_cents"] / 100))

    # ── the embedding space: its nearest in-deck neighbours ───────────────────
    try:
        from manamap.pilot import similar_cards
        from manamap.pilot.card_search import deck_names
        within = deck_names(slug) - _names_of(canonical) - set(expand_faces(canonical))
        rows, _meta = similar_cards.similar([canonical], within=within, limit=NEIGHBOURS)
        if rows:
            facts.append(_fact("neighbours", "nearest in-deck by function: "
                               + ", ".join(f"{r['name']} ({r['score']:.2f})" for r in rows),
                               DATA, "similar-cards (embeddings_ability.npy, within the deck)", canonical,
                               neighbours=[{"name": r["name"], "score": r["score"]} for r in rows]))
    except (SystemExit, Exception) as exc:              # noqa: BLE001 — no embeddings here
        absent.append({"source": "similar-cards", "why": str(exc).splitlines()[0][:120]})

    # ── the goldfish: can it even see the card ────────────────────────────────
    cov = None
    if text_row:
        try:
            from manamap.pilot import model_coverage
            flags, named = model_coverage.declaration(slug, branch)
            cov = model_coverage.card_state({"name": canonical, "type_line": text_row.get("type_line") or "",
                                             "oracle_text": text_row.get("oracle_text") or "",
                                             **({k: row[k] for k in row} if row else {})}, flags, named)
            facts.append(_fact("coverage", f"goldfish: {cov['state']}"
                               + (f" (channels off: {', '.join(cov['dark_channels'])})" if cov["dark_channels"] else "")
                               + (" — no channel reads it" if cov["state"] == "invisible" else "")
                               + (f"; unread: {cov['unread'][:80]}" if cov.get("unread") else ""),
                               DATA, "model_coverage.card_state", canonical, state=cov["state"],
                               channels=cov["active"]))
        except Exception as exc:                        # noqa: BLE001
            absent.append({"source": "model_coverage", "why": str(exc)[:120]})

    return {"slug": slug, "branch": branch, "card": card, "canonical": canonical,
            "in_deck": in_deck, "in_branch": in_branch,
            "decklist_sha256": decklist_sha256(slug, branch),
            "coverage": cov["state"] if cov else None,
            "facts": facts, "absent": absent}


# ── the instrument's own label ────────────────────────────────────────────────

ASTERISK = ("No blockers, no removal, no opponents: the rows are one input, never the call.")


def coverage_state(slug, card, branch=None):
    """One card's `model_coverage.card_state` state, read the way the goldfish
    reads the deck (branch declaration falling back to the deck's), or None when
    the card is in neither list and the corpus is not here. The cheap half of
    `dossier`, for an instrument's header over thirty changed cards."""
    row, _ = _card_row(slug, card, branch)
    text_row = row or _corpus_row(card)
    if not text_row:
        return None
    try:
        from manamap.pilot import model_coverage
        flags, named = model_coverage.declaration(slug, branch)
        return model_coverage.card_state({"name": text_row.get("name") or card,
                                          "type_line": text_row.get("type_line") or "",
                                          "oracle_text": text_row.get("oracle_text") or "",
                                          **({k: row[k] for k in row} if row else {})}, flags, named)["state"]
    except Exception:                                   # noqa: BLE001
        return None


def coverage(slug, cards, branch=None):
    """`{named, seen, partial, dark, invisible, unknown, share}` over the cards an
    instrument's reading is about — the figure every goldfish table carries first."""
    out = {"named": 0, "seen": [], "partial": [], "dark": [], "invisible": [], "unknown": []}
    for c in cards:
        out["named"] += 1
        st = coverage_state(slug, c, branch)
        out[st if st in ("seen", "partial", "dark", "invisible") else "unknown"].append(c)
    read = len(out["seen"]) + len(out["partial"])
    out["share"] = round(read / out["named"], 3) if out["named"] else None
    return out


def coverage_block(slug, cards, branch=None):
    """The one block every instrument prints before a row: what the goldfish could
    see of the cards in question. Returns `(text, coverage)`."""
    cov = coverage(slug, cards, branch)
    read = len(cov["seen"]) + len(cov["partial"])
    bits = []
    if cov["seen"]:
        bits.append("seen: " + ", ".join(cov["seen"]))
    if cov["partial"]:
        bits.append("partly read: " + ", ".join(cov["partial"]))
    if cov["dark"]:
        bits.append("DARK (a channel is off): " + ", ".join(cov["dark"]))
    if cov["invisible"]:
        bits.append("invisible (nothing reads them): " + ", ".join(cov["invisible"]))
    if cov["unknown"]:
        bits.append("unknown to the model: " + ", ".join(cov["unknown"]))
    text = (f"* ONE INSTRUMENT — the goldfish sees {read} of {cov['named']} card(s) in question"
            + (" (" + "; ".join(bits) + ")" if bits else "") + ". " + ASTERISK)
    return text, cov


# ── printing ──────────────────────────────────────────────────────────────────

_MARK = {VERIFIED: "✓", DATA: "◆", COACH: "★"}


def dossier_lines(doc, limit=None):
    """The dossier as lines a swap screen or a study can print, one fact each."""
    out = []
    facts = doc["facts"][:limit] if limit else doc["facts"]
    for f in facts:
        out.append(f"  {_MARK.get(f['tier'], '?')} {f['text']}  [{f['source']}]")
    if limit and len(doc["facts"]) > limit:
        out.append(f"  … {len(doc['facts']) - limit} more: `manamap pilot card-dossier {doc['slug']} "
                   f"\"{doc['canonical']}\"" + (f" --branch {doc['branch']}" if doc.get("branch") else "") + "`")
    return out


def main(args):
    doc = dossier(args.slug, args.card, getattr(args, "branch", None))
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2, ensure_ascii=False))
        return
    where = ("in the list" if doc["in_deck"] else "NOT in the list") + (
        f"; {'in' if doc['in_branch'] else 'not in'} branch {doc['branch']}" if doc["branch"] else "")
    print(f"CARD DOSSIER — {doc['canonical']} in {doc['slug']} ({where})")
    print(f"  {len(doc['facts'])} fact(s) from {len({f['source'].split('#')[0] for f in doc['facts']})} source(s); "
          f"✓ verified  ◆ data  ★ judgment")
    for line in dossier_lines(doc):
        print(line)
    if doc["absent"]:
        print("  absent: " + "; ".join(f"{a['source']} ({a['why']})" for a in doc["absent"]))
