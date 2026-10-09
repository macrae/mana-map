"""Pilot: the Deck Context — one living document per deck (PRD v2, 2026-10-07).

`data/decks/<slug>/CONTEXT.md` is the first place Jarvis looks for anything about a
deck. It replaces the engine model, the handbook and the dossier as the thing a
person reads, and it is two kinds of text in one file:

  PROSE, written and pruned by the Context Keeper agent — what the deck is, how it
  plays, its cards grouped by what they DO here, the pilot notes, the open
  questions. Each prose pass is stamped with the version it was written for
  (`<!-- ctx:written-for … -->`), so a reader can see when the list moved under it.

  GENERATED BLOCKS, between `<!-- ctx:gen NAME -->` and `<!-- /ctx:gen NAME -->` —
  the version and status, the numbers, the record, the change history. Code writes
  them from the artifacts that own each figure (`deck_info.compose`,
  `deck_versions.report`, the decisions ledger) and `regen` refreshes them, so a
  figure in a Deck Context is never stale by hand. The Keeper never writes one.

WHY THE CHECK NAMES CUT CARDS. The engine model it replaces rendered cards the deck
had stopped running, with nothing on the page saying so (sharknado, 2026-10-07).
Every card in authored prose is a ManaMap link, so `check` can read them back: a
card the 99 does not run is an ERROR while the prose claims to be current, and a
named STALE warning once the list has moved past it. Absent is never silent.

Commands (`manamap pilot context`):
  <slug>                 print it (or `--slice summary numbers …`, what agents get)
  <slug> --check         the gate: generated blocks current, cards in the 99, stamp
  <slug> --scaffold      a fresh file: generated blocks filled, prose placeholders
  <slug> --refresh       rewrite the generated blocks only, prose kept byte for byte
  --refresh --all        every live deck that has one (what regen runs)
  <slug> --install FILE  the Keeper's draft -> CONTEXT.md: links expanded, blocks
                         re-rendered, strict check, stamped, `.prev` kept, changelog
"""
import re
import sys
import urllib.parse
from datetime import date

from manamap import config
from manamap.pilot.common import deck_dir, load_deck_cards, load_json, report_errors

ARTIFACT = "CONTEXT.md"

#: The deployed ManaMap. Absolute, so a link works from an editor, from GitHub and
#: from Claude Code alike; `?cards=` (plural) because `?card=` falls back to a
#: random card when a name does not resolve.
SITE = "https://manamap.seanmacrae.com"
CARD_HREF = SITE + "/viz/index.html?cards="
DECK_HREF = SITE + "/viz/deck.html?deck="

#: The template, in PRD order: (key, heading, generated block or None, placeholder).
SECTIONS = (
    ("summary", "Summary", None,
     "What the deck is and what it is trying to do, in a short paragraph."),
    ("plays", "How it plays", None,
     "Game plan, the key turns, how it wins, what it fears."),
    ("cards", "Cards by role", None,
     "One `### <what these cards do here>` heading per role, each a list of "
     "[[Card]] links. Every card in the 99 under at least one role; a card may "
     "sit under several."),
    ("numbers", "Numbers", "numbers", None),
    ("pilot", "Pilot notes", "record",
     "A compact summary of the games: recurring patterns, links to log ids."),
    ("history", "Change history", "history",
     "What we tried, what happened, and why we think so — the history worth keeping."),
    ("questions", "Open questions", "queue",
     "What is still unsettled about this deck, in plain words. The block above is the "
     "queue's view of it; cite an item by its id (Q007)."),
    ("changelog", "Context changelog", None, None),
)
SECTION_KEYS = tuple(s[0] for s in SECTIONS)
GEN_BLOCKS = ("summary", "numbers", "record", "history", "queue")
PLACEHOLDER = "_(to be written)_"

#: The sections that describe the list on disk, so every card they name must be in
#: it. The others (pilot notes, change history, open questions) name cut cards and
#: candidates on purpose.
CURRENT_SECTIONS = ("summary", "plays", "cards")

#: A Deck Context is read in a few minutes. Past this the Keeper must prune.
MAX_LINES = 400

_GEN_RE = re.compile(r"<!-- ctx:gen (\w+) -->\n(.*?)<!-- /ctx:gen \1 -->", re.S)
_OPEN_RE = re.compile(r"<!-- (/?)ctx:gen (\w+) -->")
_STAMP_RE = re.compile(r"<!-- ctx:written-for (.*?) -->")
_WIKI_RE = re.compile(r"\[\[([^\]]+)\]\]")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(" + re.escape(CARD_HREF) + r"([^)]+)\)")


# ── the deck, as the gates see it ─────────────────────────────────────────────

def _deck_names(slug):
    """`{lowercased name: canonical name}` over the 99 and the commanders.

    Both halves of a double-faced card answer to the whole name, because prose
    writes "Treasure Map" and cards.json stores "Treasure Map // Treasure Cove".
    """
    doc = load_deck_cards(slug)
    rows = (doc.get("cards") if isinstance(doc, dict) else doc) or []
    out = {}
    for c in rows:
        name = c.get("name") or ""
        out[name.lower()] = name
        for face in name.split(" // "):
            out.setdefault(face.strip().lower(), name)
    return out


def card_link(name):
    return f"[{name}]({CARD_HREF}{urllib.parse.quote_plus(name)})"


def expand_links(text):
    """`[[Card]]` -> a ManaMap link. The authored form is short; the file is not."""
    return _WIKI_RE.sub(lambda m: card_link(m.group(1).strip()), text)


def named_cards(text):
    """Every card an authored passage links, in order, `[[…]]` included."""
    names = [urllib.parse.unquote_plus(m.group(2)) for m in _LINK_RE.finditer(text)]
    names += [m.group(1).strip() for m in _WIKI_RE.finditer(text)]
    return names


# ── the generated blocks ──────────────────────────────────────────────────────

def _pct(x):
    return f"{round(100 * x)}%"


def _ci(ci):
    return f" (95% CI {_pct(ci[0])}–{_pct(ci[1])})" if ci else ""


def _version_label(info):
    v = info.get("version") or {}
    tags = v.get("tags") or []
    label = (tags[-1] + f" (V{v.get('current')})") if tags else f"V{v.get('current')}"
    return label + (" + uncommitted edits" if v.get("uncommitted") else "")


def _measured_on(slug, sha, report):
    """`'v1.2.0 (V7), 2026-10-06'` for the list a figure was measured on."""
    if not sha:
        return "an unstamped list"
    by_tag = {t["version"]: name for name, t in (report.get("tags") or {}).items()}
    for v in report.get("versions") or []:
        if any(str(s).startswith(sha[:12]) for s in v.get("decklist_sha256s") or []):
            tag = by_tag.get(v["version"])
            return (f"{tag} (V{v['version']})" if tag else f"V{v['version']}") + f", {v['date']}"
    if report.get("working_decklist_sha256", "").startswith(sha[:12]):
        return "the working list (uncommitted)"
    return f"a list no version matches ({sha[:12]})"


def _block_summary(slug, info):
    commanders = info.get("commander") or []
    if isinstance(commanders, str):
        commanders = [commanders]
    stage = info.get("stage") or "bench"
    if info.get("lifecycle"):
        status = f"retired ({info['lifecycle'].get('headline', '')})"
    elif info.get("paper"):
        status = f"sleeved since {info['paper'].get('built_at')}"
    else:
        status = f"on the bench ({stage})"
    br = info.get("bracket") or {}
    if commanders:
        head = f"- **Commander:** {' + '.join(card_link(c) for c in commanders)}"
    else:
        # A 60-card deck (PRD Area C): no command zone, so the first line names
        # the FORMAT, the colours and the size — what a Modern pilot would say.
        from manamap.pilot import formats

        head = (f"- **Format:** {formats.get(info.get('format')).name} · "
                f"{''.join(info.get('colour_identity') or []) or 'colourless'} · "
                f"{info.get('size')} cards")
    lines = [
        head,
        f"- **Version:** {_version_label(info)} · **status:** {status}",
        f"- **Colours:** {''.join(info.get('colour_identity') or []) or 'colourless'}"
        f" · **lands:** {info.get('lands')} · **bracket floor:** "
        + (f"{br.get('floor')} ({br.get('floor_name')})" if br.get("floor") else "not checked"),
        f"- **Combos:** {_combos_line(slug, info)}",
        f"- **Links:** [deck page]({DECK_HREF}{slug}) · "
        f"[on the map]({SITE}/viz/index.html?deck={slug})",
    ]
    return "\n".join(lines)


def _combos_line(slug, info):
    """`'7 known lines (5 infinite, 1 two-card) · 312 one card short'`, from
    `info.combos` (`deck-combos --write`'s summary), or `not computed` — the
    same word the bracket floor uses when its artifact is missing, so Jarvis
    can answer "what combos does it have" from this line or say what to run."""
    cb = info.get("combos") or {}
    s = cb.get("summary") if isinstance(cb, dict) else None
    if not isinstance(s, dict):
        return f"not computed (`manamap pilot deck-combos {slug} --write`)"
    n, inf, two = s.get("included", 0), s.get("infinite", 0), s.get("two_card_infinite", 0)
    near = s.get("near_total", s.get("near", 0))
    return (f"{n} known line{'' if n == 1 else 's'} ({inf} infinite, {two} two-card)"
            f" · {near} one card short")


def _block_numbers(slug, info, report):
    """The goldfish headline, the engine's assembly rate, mana, and Forge — dated."""
    base = deck_dir(slug)
    gm = load_json(base / "goldfish_metrics.json") or {}
    meta = gm.get("meta") or {}
    out = []
    if info.get("format"):
        # Commander-only, and the reason is the model block's own — see
        # `deck_info._goldfish_block`. "Run goldfish" would refuse here.
        from manamap.pilot import formats

        out.append(f"- Goldfish: not modelled for {formats.get(info['format']).name} — "
                   "the goldfish is Commander-only (docs/simulation.md).")
    elif not gm:
        out.append("- No goldfish run yet (`manamap pilot goldfish " + slug + "`).")
    else:
        out.append(f"Goldfish, {meta.get('iterations', '?'):,} seeded games, measured on "
                   f"{_measured_on(slug, meta.get('decklist_sha256'), report)}. "
                   "Each group below is the share of games that have DRAWN it by turn six. "
                   "No blockers and no removal: read board quality with care.")
        out.append("")
        facts = ((info.get("model") or {}).get("goldfish") or {}).get("facts") or {}
        for key, f in facts.items():
            # Scalars only: the per-turn series and the model's assumptions live
            # in goldfish_metrics.json, and a Deck Context links rather than copies.
            if isinstance(f.get("value"), bool) or not isinstance(f.get("value"), (int, float)):
                continue
            label = key[len("target:"):] if key.startswith("target:") else f.get("definition", key)
            label = label.split(" — ")[0] if not key.startswith("target:") else label
            unit = f.get("unit") or ""
            value = f"{f['value']}{'%' if unit == '%' else ''}"
            if unit == "turn":
                value = f"turn {f['value']}"
            out.append(f"- {label[:200]}: **{value}**")
    eh = info.get("engine_health") or {}
    if eh.get("rate") is not None:
        out.append(f"- Engine online by turn {eh.get('turn')}: **{_pct(eh['rate'])}**"
                   f"{_ci(eh.get('ci95'))} — {eh.get('word', '').lower()}")
    diag = (info.get("diagnostic") or {}).get("engine") or {}
    for t in ("6", "8"):
        r = (diag.get("any_route_by_turn") or {}).get(t)
        if r:
            out.append(f"- A win route assembled by turn {t}: **{_pct(r['rate'])}**{_ci(r.get('ci95'))}")
    mana = load_json(base / "mana_analysis.json") or {}
    oc = (mana.get("on_curve_probability") or {}).get("with_rocks_and_dorks") or {}
    if oc:
        out.append("- Colour on curve (lands + rocks): "
                   + ", ".join(f"{c} {_pct(p)}" for c, p in oc.items()))
    sim = info.get("simulation") or {}
    if sim.get("runs"):
        stale = " — on an older list" if sim.get("stale") else ""
        rate = sim.get("win_rate")
        out.append(f"- Forge (an optional probe, not a gate): last run {sim.get('at')}, "
                   f"{sim.get('wins')}/{sim.get('games')} won"
                   + (f" ({_pct(rate)}{_ci(sim.get('win_rate_ci95'))})" if rate is not None else "")
                   + f" on V{sim.get('ran_on_version')}{stale}")
    return "\n".join(out)


def _block_record(info):
    r = info.get("record") or {}
    if not r.get("games"):
        return "- No games logged yet (`manamap pilot deck-notes " + info["slug"] + " add …`)."
    causes = ", ".join(f"{k} {v}" for k, v in sorted((r.get("cause_counts") or {}).items(),
                                                     key=lambda kv: (-kv[1], kv[0])))
    lines = [f"- **{r['games']} game(s):** {r.get('win', 0)} won, {r.get('loss', 0)} lost"
             + (f", {r['draw']} drawn" if r.get("draw") else "")
             + f" · {r.get('first_played')} to {r.get('last_played')}"]
    if causes:
        lines.append(f"- **How games ended:** {causes}")
    if r.get("undebriefed"):
        lines.append(f"- **Not yet read:** {', '.join(r['undebriefed'])}")
    return "\n".join(lines)


def _ledger_by_sha(slug):
    from manamap.pilot import decisions

    out = {}
    try:
        rows = decisions.read(slug)
    except Exception:                               # pragma: no cover - defensive
        rows = []
    for e in rows:
        if e.get("kind") not in ("merge", "propose"):
            continue
        sha = e.get("decklist_sha256") or ""
        out.setdefault(sha[:12], []).append(e)
    return out


def _block_history(slug, report, limit=8):
    """Newest first: what changed, the ledger's why, and what was predicted."""
    tags = {t["version"]: (name, t.get("note") or "") for name, t in (report.get("tags") or {}).items()}
    ledger = _ledger_by_sha(slug)
    out = []
    versions = list(reversed(report.get("versions") or []))
    for v in versions[:limit]:
        tag, note = tags.get(v["version"], (None, ""))
        head = f"- **{tag + ' · ' if tag else ''}V{v['version']}** ({v['date']})"
        ins, outs = v.get("in") or [], v.get("out") or []
        if v is report["versions"][0]:
            change = f"first list, {v.get('size', len(ins))} cards"
        else:
            change = (f"in: {', '.join(ins[:8])}{' …' if len(ins) > 8 else ''}"
                      f"; out: {', '.join(outs[:8])}{' …' if len(outs) > 8 else ''}")
        line = f"{head} — {change}"
        if note:
            line += f". _{note[:200]}_"
        out.append(line)
        for e in ledger.get(str(v.get("decklist_sha256") or "")[:12], []):
            pred = e.get("prediction") or {}
            grade = pred.get("grade")
            why = e.get("forced_reason") or (pred.get("objective") or {}).get("why") or ""
            bit = f"  - {e['kind']} `{e.get('branch')}`"
            if pred.get("endpoint"):
                bit += f": objective {pred['endpoint']}" + (f" read **{grade}**" if grade else "")
            if e.get("outcome"):
                bit += f"; realised: {str(e['outcome'].get('realised'))[:80]}"
            if why:
                bit += f" — {why[:160]}"
            out.append(bit)
        if v.get("games"):
            rec = v.get("record") or {}
            out.append(f"  - played {v['games']}: {rec.get('win', 0)}W {rec.get('loss', 0)}L")
    if len(versions) > limit:
        out.append(f"- … {len(versions) - limit} earlier version(s): "
                   f"`manamap pilot deck-version {slug} list`")
    return "\n".join(out)


def render_generated(slug):
    """`{block: text}` — every generated block, from the artifacts that own it."""
    from manamap.pilot import deck_info, deck_versions, queue

    info = deck_info.compose(slug, ladder=False)
    report = deck_versions.report(slug)
    return {
        "summary": _block_summary(slug, info),
        "numbers": _block_numbers(slug, info, report),
        "record": _block_record(info),
        "history": _block_history(slug, report),
        "queue": queue.deck_block(slug),
    }


# ── the document ──────────────────────────────────────────────────────────────

def _gen(name, body):
    return f"<!-- ctx:gen {name} -->\n{body.rstrip()}\n<!-- /ctx:gen {name} -->"


def _title(slug):
    try:
        doc = load_deck_cards(slug)
        rows = (doc.get("cards") if isinstance(doc, dict) else doc) or []
        names = [c["name"] for c in rows if c.get("is_commander")]
    except FileNotFoundError:
        names = []
    return f"# {' + '.join(names) or slug} — Deck Context"


def scaffold_text(slug, blocks):
    parts = [_title(slug), "", "<!-- ctx:written-for none -->", "", _gen("summary", blocks["summary"])]
    for key, heading, block, placeholder in SECTIONS:
        parts += ["", f"## {heading}", ""]
        if block and block != "summary":
            parts.append(_gen(block, blocks[block]))
            if placeholder:
                parts += ["", PLACEHOLDER]
        elif key == "changelog":
            parts.append(f"- {date.today().isoformat()} · scaffolded by `manamap pilot context`")
        else:
            parts.append(PLACEHOLDER)
    return "\n".join(parts) + "\n"


def check_markers(text):
    """Every block opened is closed, in order, and named once."""
    errors, stack, seen = [], [], set()
    for m in _OPEN_RE.finditer(text):
        closing, name = m.group(1) == "/", m.group(2)
        if not closing:
            if stack:
                errors.append(f"block '{name}' opens inside '{stack[-1]}'")
            if name in seen:
                errors.append(f"block '{name}' appears twice")
            seen.add(name)
            stack.append(name)
        elif not stack or stack[-1] != name:
            errors.append(f"block '{name}' closes without opening")
        else:
            stack.pop()
    errors += [f"block '{n}' is never closed" for n in stack]
    unknown = seen - set(GEN_BLOCKS)
    errors += [f"unknown block '{n}' — the blocks are {', '.join(GEN_BLOCKS)}" for n in sorted(unknown)]
    return errors


def replace_blocks(text, blocks):
    """Swap every generated block's body; everything else is kept byte for byte.

    A block the template gained after a file was written (`queue`, 2026-10-07) is
    inserted under its section's heading, so a refresh migrates the fleet rather
    than leaving old files a block short forever."""
    errors = check_markers(text)
    if errors:
        raise ValueError("; ".join(errors))
    text = _GEN_RE.sub(lambda m: _gen(m.group(1), blocks.get(m.group(1), m.group(2))), text)
    present = {m.group(1) for m in _GEN_RE.finditer(text)}
    for _key, heading, block, _p in SECTIONS:
        if block and block in blocks and block not in present:
            marker = f"\n## {heading}\n"
            if marker in text:
                text = text.replace(marker, f"{marker}\n{_gen(block, blocks[block])}\n", 1)
    return text


def sections(text):
    """`{key: text}` for each `## ` section, in file order; `_head` is the preamble."""
    by_heading = {heading.lower(): key for key, heading, _b, _p in SECTIONS}
    out, key, buf = {}, "_head", []
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            out[key] = "".join(buf)
            key = by_heading.get(line[3:].strip().lower(), "?" + line[3:].strip())
            buf = [line]
        else:
            buf.append(line)
    out[key] = "".join(buf)
    return out


def authored(text):
    """The text with every generated block removed: what the Keeper wrote."""
    return _GEN_RE.sub("", text)


def stamp(text):
    """`{version, sha, at}` from the header, or None for 'none' / missing."""
    m = _STAMP_RE.search(text)
    if not m or m.group(1).strip() == "none":
        return None
    return dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv)


def check_text(slug, text, blocks=None, strict=False):
    """`(errors, warnings)`. `strict` is the install gate: every card placed, no
    placeholder left, no warning at all."""
    from manamap.pilot import common

    errors, warnings = list(check_markers(text)), []
    secs = sections(text)
    for key in SECTION_KEYS:
        if key not in secs:
            errors.append(f"section '## {dict((k, h) for k, h, _b, _p in SECTIONS)[key]}' is missing")
    for key in secs:
        if key.startswith("?"):
            warnings.append(f"section '## {key[1:]}' is not in the template")
    if errors:
        return errors, warnings

    if blocks is not None:
        for name, body in _GEN_RE.findall(text):
            if name in blocks and body.rstrip() != blocks[name].rstrip():
                errors.append(f"generated block '{name}' is out of date — "
                              f"`manamap pilot context {slug} --refresh`")

    truth = common.decklist_sha256(slug)
    st = stamp(text)
    current = bool(st and truth and common.sha_matches(st.get("sha", ""), truth))
    if st is None:
        warnings.append("prose not written yet — the Context Keeper's `seed` pass writes it")
    elif not current:
        warnings.append(f"STALE: prose written for {st.get('version', '?')} "
                        f"({st.get('sha', '?')[:12]}); the deck has moved since")

    deck = _deck_names(slug)
    prose = authored(text)
    # Only the sections that describe the deck AS IT IS. History, the pilot's
    # notes and the open questions name cut cards and candidates on purpose.
    named = [n for k in CURRENT_SECTIONS for n in named_cards(authored(secs.get(k, "")))]
    cut = sorted({n for n in named if n.lower() not in deck})
    if cut:
        msg = f"names {len(cut)} card(s) the 99 does not run: {', '.join(cut)}"
        (errors if current else warnings).append(msg if current else "STALE: " + msg)

    placed = {deck[n.lower()] for n in _named_in(secs.get("cards", "")) if n.lower() in deck}
    unplaced = sorted(set(deck.values()) - placed)
    # `strict` looks whatever the stamp says: a Keeper's draft carries no stamp
    # until install writes one, and a self-check that skipped this read two
    # drafts clean that the install then refused (2026-10-07).
    if (st is not None or strict) and unplaced:
        warnings.append(f"{len(unplaced)} card(s) in the 99 under no role: "
                        f"{', '.join(unplaced[:12])}{' …' if len(unplaced) > 12 else ''}")
    if (st is not None or strict) and PLACEHOLDER in prose:
        warnings.append("a section still reads " + PLACEHOLDER)
    n_lines = text.count("\n")
    if n_lines > MAX_LINES:
        warnings.append(f"{n_lines} lines, over the {MAX_LINES}-line budget — prune")
    if strict:
        errors += [w for w in warnings if not w.startswith("STALE: prose written")]
        warnings = []
    return errors, warnings


def _named_in(section_text):
    return named_cards(authored(section_text))


def path(slug):
    return deck_dir(slug) / ARTIFACT


def check(slug, strict=False):
    p = path(slug)
    if not p.exists():
        return [f"no {ARTIFACT} — `manamap pilot context {slug} --scaffold`"], []
    return check_text(slug, p.read_text(), render_generated(slug), strict=strict)


def refresh(slug):
    """Rewrite the generated blocks in place. Returns True when the file changed."""
    p = path(slug)
    if not p.exists():
        return False
    before = p.read_text()
    after = replace_blocks(before, render_generated(slug))
    if after != before:
        p.write_text(after)
    return after != before


def list_change(slug):
    """What a merge or check-in owes the Deck Context, right after it wrote the list.

    THE HOOK (PRD v2 Step 2). The context's prose is stamped with the sha it was
    written for, so it reads STALE the moment `decklist.txt` moves — that much was
    always derived. What was missing is anyone SAYING so at the moment it happens:
    the generated blocks lagged until the next regen, and the Keeper's `deck-change`
    pass ran only if somebody remembered. A command cannot spawn an agent, so this
    refreshes what is deterministic and hands back the one step Jarvis runs next.

    Returns None when the deck has no context. The ins and outs are read from the
    `.txt.bak` both writers leave (copies, not entries), so the Keeper is told
    exactly what moved.
    """
    from collections import Counter

    from manamap.pilot import common
    from manamap.pilot.fetch_deck import parse_mainboard

    p = path(slug)
    if not p.exists():
        return None
    try:
        refresh(slug)
        refreshed = True
    except Exception as exc:                            # pragma: no cover - env
        refreshed = f"blocks not refreshed: {exc}"

    def copies(f):
        if not f.exists():
            return Counter()
        return Counter({e["name"]: e["quantity"] for e in parse_mainboard(f.read_text(encoding="utf-8"))})

    lst = deck_dir(slug) / "decklist.txt"
    before, after = copies(lst.with_suffix(".txt.bak")), copies(lst)
    outs, ins = sorted((before - after).elements()), sorted((after - before).elements())
    st = stamp(p.read_text())
    truth = common.decklist_sha256(slug)
    stale = not (st and truth and common.sha_matches(st.get("sha", ""), truth))
    keeper = (f"spawn context-keeper MODE deck-change for {slug}"
              f" — out: {', '.join(outs) or 'nothing'}; in: {', '.join(ins) or 'nothing'}")
    install = (f"manamap pilot context {slug} --install "
               f"data/decks/{slug}/.agent-out/context-keeper.md --note \"deck change: …\"")
    return {"stale": stale, "written_for": (st or {}).get("version"), "outs": outs,
            "ins": ins, "refreshed": refreshed, "keeper": keeper, "install": install}


def print_list_change(change):
    """The lines a merge or check-in prints. One home, so the two cannot drift."""
    if not change:
        return
    if not change["stale"]:
        print("\n  CONTEXT current — CONTEXT.md was already written for this list")
        return
    print(f"\n  CONTEXT STALE — the prose was written for {change['written_for'] or 'an older list'}; "
          f"the list moved under it" + ("" if change["refreshed"] is True else f" ({change['refreshed']})"))
    print(f"    next (Jarvis): {change['keeper']}")
    print(f"    then:          {change['install']}")


def scaffold(slug, force=False):
    p = path(slug)
    if p.exists() and not force:
        raise SystemExit(f"{p} exists — `--refresh` updates its numbers; `--install` "
                         "replaces its prose; --force overwrites it whole")
    p.write_text(scaffold_text(slug, render_generated(slug)))
    return p


def slice_text(slug, keys):
    """The header and only the sections asked for — what an agent is sent."""
    text = path(slug).read_text()
    secs = sections(text)
    bad = [k for k in keys if k not in SECTION_KEYS]
    if bad:
        raise SystemExit(f"unknown section(s) {', '.join(bad)} — choose from {', '.join(SECTION_KEYS)}")
    return "".join([secs["_head"]] + [secs[k] for k in keys if k in secs])


def install(slug, draft_path, note, who="Keeper"):
    """The Keeper's draft -> CONTEXT.md, through the strict gate. Enforced here,
    not in the charter: a draft that names a cut card or leaves a card unplaced
    is refused, and nothing is written."""
    from manamap.pilot import common, deck_info

    draft = open(draft_path).read()
    if not note or not note.strip():
        raise SystemExit("--note is required: one line on what this pass changed")
    blocks = render_generated(slug)
    text = expand_links(draft)
    if check_markers(text):
        raise SystemExit("FAIL the draft's generated-block markers are broken: "
                         + "; ".join(check_markers(text)))
    text = replace_blocks(text, blocks)
    info = deck_info.compose(slug, ladder=False)
    sha = (common.decklist_sha256(slug) or "")[:12]
    header = (f"<!-- ctx:written-for version={_version_label(info).split(' ')[0]} "
              f"sha={sha} at={date.today().isoformat()} -->")
    text = _STAMP_RE.sub(header, text, count=1) if _STAMP_RE.search(text) \
        else text.replace("\n", f"\n\n{header}\n", 1)
    line = f"- {date.today().isoformat()} · {who} · {note.strip()}"
    secs = sections(text)
    if "changelog" in secs:
        text = text.replace(secs["changelog"], secs["changelog"].rstrip("\n") + "\n" + line + "\n", 1)
    errors, _ = check_text(slug, text, blocks, strict=True)
    report_errors(f"{slug} — {ARTIFACT} draft refused", errors)
    p = path(slug)
    if p.exists():
        p.with_name(ARTIFACT + ".prev").write_text(p.read_text())
    p.write_text(text)
    return p


def live_slugs():
    from manamap.pilot import regen

    return [d.name for d in sorted(config.DECKS_DIR.iterdir())
            if d.is_dir() and (d / ARTIFACT).exists() and not regen.is_retired(d.name)]


def main(args):
    slug = getattr(args, "slug", None)
    if getattr(args, "all", False):
        if not getattr(args, "refresh", False) and not getattr(args, "check", False):
            raise SystemExit("--all goes with --refresh or --check")
        slugs = live_slugs()
    elif not slug:
        raise SystemExit("name a deck, or pass --all with --refresh/--check")
    else:
        slugs = [slug]

    if getattr(args, "scaffold", False):
        p = scaffold(slug, force=getattr(args, "force", False))
        print(f"wrote {p} — generated blocks filled, prose placeholders for the Keeper's seed pass")
        return
    if getattr(args, "install", None):
        p = install(slug, args.install, getattr(args, "note", None),
                    who=getattr(args, "who", None) or "Keeper")
        _, warnings = check(slug)
        print(f"OK   installed {p}" + "".join(f"\n  · {w}" for w in warnings))
        return
    if getattr(args, "refresh", False):
        for s in slugs:
            changed = refresh(s)
            print(f"{'updated' if changed else 'current'}  {s}")
        return
    if getattr(args, "check", False):
        failed = False
        for s in slugs:
            errors, warnings = check(s)
            if errors:
                failed = True
                print(f"FAIL {s} — {ARTIFACT} ({len(errors)} error(s)):")
                for e in errors:
                    print(f"  - {e}")
            else:
                print(f"OK   {s} — {ARTIFACT}")
            for w in warnings:
                print(f"  · {w}")
        if failed:
            sys.exit(1)
        return
    if not path(slug).exists():
        raise SystemExit(f"{slug} has no {ARTIFACT} yet — `manamap pilot context {slug} --scaffold`")
    keys = getattr(args, "slice", None)
    print(slice_text(slug, keys) if keys else path(slug).read_text(), end="")
