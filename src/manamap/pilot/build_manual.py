"""Pilot: render a deck's issue of Pilot's Manual as standalone magazine HTML.

LEGACY (2026-08-19): the magazine renderer. It still renders the nine frozen issues from
artifacts nothing regenerates any more (issue_plan.json, the panel keys,
card_roles/mana_base/upgrades, considering.json), and it is replaced by the compact deck
page in docs/manual-v5-spec.md. Do not extend it; internals below are accurate for what it
does.

Fully deterministic — no LLM calls, no dates, no randomness. The editorial layer
arrives as data (`issue.json` identity + `issue_plan.json` packaging from the
magazine-editor agent) and the body prose as `manual_prose.json`; this module
assembles them into the fixed sections of STYLEv3 §5 (issue_spec.DEPARTMENTS).

Contract invariants:
- Only checker-passed stacks render, and Judge's Desk reproduces every citation
  verbatim — the renderer may not summarize proof (STYLEv3 §5.1, docs/pilot.md).
- A department whose *prose* is missing renders a visible [TODO], never vanishes.
  (The 99 and Featured Artist are grid-driven and thin out gracefully instead.)
- Tier badges come from the department system, not the plan (STYLEv3 §10).
- Every department renders the plan's furniture. `validate_issue` accepts pilot
  tips and captions for any department, so a renderer that ignored them would
  silently drop validated content.

The only date here is `issue.json`'s authored `issue_date`; nothing is generated,
which is what keeps rebuilds byte-identical.
"""

import json
import re

from manamap.config import MANUALS_DIR, SYNERGY_GRAPH_PATH
from manamap.pilot.common import (
    checker_passed,
    presentable,
    withheld,
    deck_dir,
    load_deck_cards,
    load_json,
    load_synergy_graph,
)
from manamap.pilot.design import (
    card_tile,
    issue_status_banner,
    stylesheet_link,
    write_stylesheet,
    FONT_LINK,
    badge,
    barcode,
    callout,
    card_figure,
    esc,
    fast_facts,
    folio,
    hot_take,
    letterhead,
    stack_theatre,
    not_modelled_rail,
    pilot_tip,
    power_meter,
    stat_slab,
    constellation_figure,
    engine_figure,
    city_head,
    printing_credit,
    pull_quote,
    threat_box,
    violator,
)
from manamap.pilot.scenario_facts import board_bodies, opponents_of
from manamap.pilot.short_list_art import ARTIFACT as SHORT_LIST_ART
from manamap.pilot.issue_spec import (
    ACTS,
    issue_status,
    OPTIONAL_DEPARTMENTS,
    BREATHER_AFTER,
    DEPARTMENT_BY_ID,
    DEPARTMENT_IDS,
    MASTHEAD,
    MASTHEAD_COLUMNISTS,
    SERIES_SLUG,
    SERIES_TITLE,
    STANDING_TAGLINE,
)

# One accent per department — held across all its pages and its folio tab.
ACCENT = {
    "cover": "var(--power-red)", "contents": "var(--ink)",
    "editors-letter": "var(--ink)", "pilots-log": "var(--burst-yellow)",
    "first-turns": "var(--power-red)", "command-zone": "var(--y2k-violet)",
    "by-the-numbers": "var(--y2k-blue)", "the-kill": "var(--power-red)",
    "at-the-table": "var(--radical-purple)",
    "whats-your-play": "var(--hot-magenta)", "the-99": "var(--slime-green)",
    "sources-say": "var(--y2k-blue)",
    "featured-artist": "var(--hot-magenta)",
    "keep-or-ship": "var(--tier-coach)", "upgrade-watch": "var(--y2k-blue)",
    "judges-desk": "var(--stamp-red)", "back-page": "var(--ink)",
}

TODO = '<p><span class="todo">TODO</span> This section is awaiting content.</p>'

# Per-render holder for `deck_map.json`, set by render_issue. Module state for the
# same reason `_CARD_LINKS` is: the renderer is single-threaded and deterministic,
# and threading one artifact through every department signature to reach one
# furniture call is worse than a holder that is cleared on the way out.
_DECK_MAP = {"doc": None, "engine": None}


def engine_stage_of(engine=None):
    """`{card name: stage}` from the engine model, or `{}` when there is none.

    Takes the doc as an argument when given, and falls back to the per-render
    holder otherwise. The holder exists because threading one artifact through
    every department signature to reach one furniture call was worse than a
    module global; a second renderer that does not use the holder is a better
    reason to accept the parameter than to make it set the global.

    Built on demand from `stages[].cards` rather than stored, because the engine
    doc is the single source and a second copy is a second thing to keep true. A
    card in two stages takes the first in canonical order — the model treats a
    stage as a job and a card doing two jobs is led by the earlier one, which is
    also the order the schematic reads in.
    """
    engine = engine if engine is not None else (_DECK_MAP.get("engine") or {})
    from manamap.pilot.validate_engine import STAGES as ORDER
    out = {}
    for stage in sorted((engine.get("stages") or []),
                        key=lambda s: ORDER.index(s["stage"])
                        if s.get("stage") in ORDER else 99):
        for name in stage.get("cards") or []:
            out.setdefault(name, stage.get("stage"))
    return out


# ── Loading ─────────────────────────────────────────────────────────────


def load_verified_stacks(slug):
    """Presentable stacks only, in id order — the publication gate.

    Two conditions, not one: the checker must have passed it AND the board must
    still be one this deck can make. See `common.presentable`.
    """
    stacks = []
    for path in sorted((deck_dir(slug) / "stacks").glob("*.json")):
        with open(path) as f:
            doc = json.load(f)
        if presentable(doc):
            stacks.append(doc)
    return stacks


def load_withheld_stacks(slug):
    """Checker-PASSED stacks held back from publication, in id order.

    They are not failures. Each passed the citation contract and was then
    withheld because a card left the 99 — the rules finding stands, the board is
    one this deck can no longer make. `presentable_note` records which card.

    They are loaded because the published prose REFERS TO THEM. Nineteen times
    across edgar and yawgmoth, a presentable stack's resolution says "exactly as
    in stack 001" or "the same lock that refuted stack 001" — and stack 001 is
    nowhere in the issue, so the reader hunts for a case that was deliberately
    not printed. Judge's Desk gives each one an index row saying so.

    **Their resolutions are still not published, and that is the point.** Editing
    the referring prose instead was the obvious fix and is the wrong one: those
    are checker-PASSED artifacts, so their step text is evidence, and rewriting
    it post-hoc puts a ✓ over words no checker read.
    """
    out = []
    for path in sorted((deck_dir(slug) / "stacks").glob("*.json")):
        with open(path) as f:
            doc = json.load(f)
        if withheld(doc) and checker_passed(doc):
            out.append(doc)
    return out


def load_decisions(slug):
    decisions = []
    directory = deck_dir(slug) / "decisions"
    if not directory.is_dir():
        return decisions
    for path in sorted(directory.glob("*.json")):
        with open(path) as f:
            doc = json.load(f)
        if doc.get("presentable", True) is not False:
            decisions.append(doc)
    return decisions


# ── Plan access ─────────────────────────────────────────────────────────


def plan_dept(plan, dept_id):
    for dept in (plan or {}).get("departments", []):
        if dept.get("id") == dept_id:
            return dept
    return {}


_STACK_REF_RE = re.compile(r"\b([Ss]tacks?)\s+(\d{3})\b")
_CR_REF_RE = re.compile(r"\b(CR\s+\d+(?:\.\d+[a-z]?)?)")


def linkify(escaped_text):
    """Turn plain evidence references in ALREADY-ESCAPED text into links.

    "stack 003" → a link to its Judge's Desk case file; "CR 603.2h" → a link
    to the Judge's Desk section. Runs strictly AFTER esc() — on escaped text a
    digit-bearing pattern cannot sit inside a tag or an entity, so injection
    is safe (pinned by the escaping test). Never applied inside Judge's Desk
    itself or to `.cite` blocks — no self-links.
    """
    out = _STACK_REF_RE.sub(
        lambda m: f'<a class="xref" href="#case-{m.group(2)}">{m.group(1)} {m.group(2)}</a>',
        escaped_text)
    out = _CR_REF_RE.sub(
        lambda m: f'<a class="xref" href="#judges-desk">{m.group(1)}</a>', out)
    return out


# ── Card links (renderer-provided navigation, STYLEv3 §8.4) ─────────────
#
# Every card mention in reader-facing copy links to that card's tile in The 99
# (the commander, which has no tile, links to The Command Zone) and carries a
# CSS-only hover preview of the card image. Agents keep writing plain names;
# the map is per-render module state set by render_issue — single-threaded,
# deterministic, cleared after assembly.

_CARD_LINKS = {"regex": None, "meta": {}}
# name → tile anchor id for the current render; The 99 reads this when minting
# tile ids so the linker's hrefs and the tiles can never drift apart.
CARD_ANCHORS = {}


def card_slug(name):
    """Deterministic anchor slug for a card name."""
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]", "-", name.lower())).strip("-")


def card_anchor_ids(cards):
    """name → unique tile anchor id, stable across renders (sorted names;
    collisions get -2, -3 … in that order)."""
    anchors, used = {}, set()
    for name in sorted({c["name"] for c in cards}):
        slug = card_slug(name)
        candidate, n = slug, 2
        while candidate in used:
            candidate, n = f"{slug}-{n}", n + 1
        used.add(candidate)
        anchors[name] = f"card-{candidate}"
    return anchors


def _card_probes(cards):
    """Conservative probe set: full names, DFC faces, and unambiguous
    pre-comma short names ("Selvala"). Returns {probe: card_name}."""
    names = {c["name"] for c in cards}
    probes = {n: n for n in names}
    for n in names:
        for face in n.split(" // "):
            face = face.strip()
            if face and face not in probes:
                probes[face] = n
    shorts = {}
    for n in sorted(names):
        if "," in n:
            short = n.split(",", 1)[0].strip()
            if len(short) >= 5:
                shorts.setdefault(short, []).append(n)
    for short, owners in shorts.items():
        if len(owners) == 1 and short not in probes:
            probes[short] = owners[0]
    return probes


def set_card_links(cards, commander_name=None, offdeck=None):
    """Arm the card linker for one render. Probes are esc()-escaped so they
    match post-escape text; longest-first so full names beat short forms.

    `offdeck` is `considering_art.json`'s `cards` map — The Short List's ten,
    which are by definition NOT in the 99 and so had no image, no link and no
    hover preview. They were the only card names in the issue a reader could not
    look at, in the one department whose entire job is showing you cards you do
    not own. Their links point OUT (to Scryfall) because there is no tile to point
    at, and they are marked `.cardref.offdeck` so the page can say so.

    In-deck names always win: a card the analyst recommends and the deck already
    runs must resolve to its own tile, not to an external page about it.
    """
    anchors = card_anchor_ids(cards)
    by_name = {c["name"]: c for c in cards}
    meta = {}
    for name, card in sorted((offdeck or {}).items()):
        if name in by_name:
            continue
        meta[esc(name)] = (card.get("scryfall_uri") or "#upgrade-watch",
                           card.get("image") or "", True)
    for probe, name in _card_probes(cards).items():
        href = ("#command-zone" if name == commander_name
                else f"#{anchors[name]}")
        meta[esc(probe)] = (href, by_name[name].get("image") or "", False)
    pattern = "|".join(re.escape(p) for p in
                       sorted(meta, key=lambda p: (-len(p), p)))
    _CARD_LINKS["regex"] = re.compile(rf"(?<!\w)(?:{pattern})(?!\w)") if meta else None
    _CARD_LINKS["meta"] = meta
    CARD_ANCHORS.clear()
    CARD_ANCHORS.update(anchors)
    return anchors


def clear_card_links():
    _CARD_LINKS["regex"] = None
    _CARD_LINKS["meta"] = {}
    CARD_ANCHORS.clear()


def card_linkify(escaped_text):
    """Wrap card-name mentions in ALREADY-ESCAPED text with tile links and a
    hover-preview image. Runs before linkify(); card names cannot contain the
    stack/CR patterns, and the emitted markup contains nothing the evidence
    regexes match, so the passes compose safely."""
    if _CARD_LINKS["regex"] is None:
        return escaped_text

    def repl(m):
        href, image, offdeck = _CARD_LINKS["meta"][m.group(0)]
        pop = (f'<img class="card-pop" src="{esc(image)}" loading="lazy" alt="">'
               if image else "")
        cls = "cardref offdeck" if offdeck else "cardref"
        rel = ' target="_blank" rel="noopener"' if offdeck else ""
        return f'<a class="{cls}" href="{href}"{rel}>{m.group(0)}{pop}</a>'

    return _CARD_LINKS["regex"].sub(repl, escaped_text)


def esc_x(text):
    """esc() + card links + evidence linkification — the default for
    reader-facing copy."""
    return linkify(card_linkify(esc(text)))


def esc_x_paras(text):
    """esc_x that honors paragraph seams: '\\n\\n' splits into <p> blocks, so a
    multi-paragraph field never renders as one collapsed wall."""
    parts = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
    if len(parts) <= 1:
        return esc_x(text)
    return "".join(f"<p>{esc_x(p)}</p>" for p in parts)


def prose(prose_doc, key, sub=None):
    """Body copy from manual_prose.json; visible TODO when absent."""
    value = (prose_doc or {}).get(key)
    if sub is not None:
        value = (value or {}).get(sub)
    if not value:
        return TODO
    paragraphs = [p.strip() for p in str(value).split("\n\n") if p.strip()]
    return "".join(f"<p>{esc_x(p)}</p>" for p in paragraphs)


def caption_html(text):
    """Caption grammar: **bold lead-in**, then roman body."""
    if "**" in text:
        head, _, tail = text.partition("**")
        lead, _, rest = tail.partition("**")
        return f"{esc_x(head)}<b>{esc_x(lead)}</b>{esc_x(rest)}"
    return esc_x(text)


# ── Department frame ────────────────────────────────────────────────────


def dept_open(dept_id, plan, extra_badges=()):
    spec = DEPARTMENT_BY_ID[dept_id]
    dept = plan_dept(plan, dept_id)
    badges = "".join(badge(t) for t in spec["tiers"]) + "".join(extra_badges)
    kicker = (
        f'<div class="kicker">{esc(dept["kicker"])}</div>' if dept.get("kicker") else ""
    )
    headline = dept.get("headline") or spec["title"]
    dek = f'<p class="dek">{esc(dept["dek"])}</p>' if dept.get("dek") else ""
    byline = (
        f'<div class="byline">by {esc(spec["byline"])}</div>' if spec["byline"] else ""
    )
    return (
        f'<section class="dept" id="{dept_id}" style="--accent:{ACCENT[dept_id]}">'
        f'<div class="dept-head"><div>'
        f'<h2 class="dept-title">{esc(spec["title"])}</h2>{byline}</div>'
        f"<div>{badges}</div>"
        f'<div class="dept-promise">{esc(spec["promise"])}</div></div>'
        f"{kicker}<h1 class=\"feature\">{esc(headline)}</h1>{dek}"
    )


def dept_close(dept_id, issue):
    return "</section>" + folio(DEPARTMENT_BY_ID[dept_id]["title"], issue["volume"])


def dept_furniture(dept, cards_by_name):
    """Render the plan's furniture for a department: tips, callouts, pull quote."""
    out = []
    # The signature number leads the department that earns it, before any callout —
    # it is the thing the reader is meant to stop on, and furniture that follows it
    # reads as elaboration rather than competition.
    # The deck's own map, where the plan asks for it. Furniture rather than a
    # department of its own for now: a new department is a structural change that
    # re-plans every issue, and the picture is worth having in one before that.
    # The engine flow leads the constellation where both are asked for: HOW IT RUNS
    # before WHAT SHAPE IT IS. The shape is the easier question and answering it
    # first invites the reader to treat the clusters as the engine, which is the
    # exact confusion this subsystem exists to undo.
    if dept.get("engine_flow") and _DECK_MAP.get("engine"):
        out.append(engine_figure(
            _DECK_MAP["engine"],
            dept.get("engine_caption")
            or "How this deck runs, stage by stage."))

    if dept.get("constellation") and _DECK_MAP.get("doc"):
        out.append(constellation_figure(
            _DECK_MAP["doc"],
            dept.get("constellation_caption")
            or "The deck, re-laid-out from its own cards and clustered. "
               "Positions are LOCAL to this deck — they are not atlas positions."))

    slab = dept.get("stat_slab")
    if slab:
        out.append(stat_slab(slab.get("figure", ""), slab.get("label", ""),
                             slab.get("note", "")))
    for step in dept.get("callouts", []):
        out.append(callout(step.get("n", "•"), step.get("title", ""),
                           step.get("text", ""), esc_fn=esc_x))
    for tip in dept.get("pilot_tips", []):
        card = cards_by_name.get(tip.get("card", ""))
        out.append(pilot_tip(tip.get("card", ""), tip.get("text", ""),
                             (card or {}).get("image"), esc_fn=esc_x))
    if dept.get("pull_quote"):
        out.append(pull_quote(dept["pull_quote"]))
    return "".join(out)


def dept_captions(dept, cards_by_name, limit=3):
    """Card figures for captions the editor wrote, in plan order."""
    figures = []
    for name, text in list((dept.get("captions") or {}).items())[:limit]:
        card = cards_by_name.get(name)
        if not card:
            continue
        figures.append(card_figure(name, card.get("image"), caption_html(text),
                                   card.get("scryfall_uri"), card))
    return "".join(figures)


# ── Departments ─────────────────────────────────────────────────────────


def render_cover(issue, plan, commander):
    cover = (plan or {}).get("cover", {})
    volume = issue["volume"]
    teases = "".join(f"<li>{esc(t)}</li>" for t in cover.get("teases", []))
    violators = "".join(
        violator(v.get("text", "")) for v in cover.get("violators", [])[:2]
    )
    kicker = (
        f'<div class="kicker">{esc(cover["kicker"])}</div>' if cover.get("kicker") else ""
    )
    coverline = cover.get("dominant_coverline") or issue["deck_name"]
    art = ""
    if commander:
        # art_crop is borderless full-bleed art — magazine photography, not a
        # card scan. Falls back to the card image when a printing lacks it.
        image = commander.get("art_crop") or commander.get("image")
        img = f'<img src="{esc(image)}" alt="{esc(commander["name"])}">'
        if commander.get("foil"):
            img = f'<span class="foil">{img}</span>'
        art = f'<div class="hero-art">{img}{printing_credit(commander)}</div>'
    return f"""
<section class="cover" id="cover">
  <div class="cover-top">
    <div><h1 class="masthead">{esc(MASTHEAD)}</h1>
      <div class="series-slug">{esc(SERIES_SLUG)}</div></div>
    <div class="cover-meta">VOL. {volume:03d}<br>{esc(issue["issue_date"])}<br>
      {esc(issue["cover_price"])}<br>{esc(STANDING_TAGLINE)}</div>
  </div>
  <div class="cover-body">
    <div>{kicker}
      <div class="coverline">{esc(coverline)}</div>
      <p class="dek">{esc(issue["cover_tagline"])} · {esc(issue["commander"])}</p>
      <ul class="teases">{teases}</ul>
      {barcode(f"vol-{volume}")}
    </div>
    <div>{art}</div>
  </div>
  {violators}
</section>"""


def render_contents(issue, plan, stacks, decisions):
    spec_c = DEPARTMENT_BY_ID["contents"]
    acts = []
    # The SAME optional filter the body uses. Without it the Flight Plan lists a
    # department this issue does not carry and links to an anchor that is not on
    # the page — eight dead links, in the one department whose whole job is
    # telling the reader where everything is.
    planned = {d.get("id") for d in (plan or {}).get("departments", [])}
    for act_title, dept_ids in ACTS:
        rows = []
        for dept_id in dept_ids:
            if dept_id in OPTIONAL_DEPARTMENTS and dept_id not in planned:
                continue
            spec = DEPARTMENT_BY_ID[dept_id]
            badges = "".join(badge(t) for t in spec["tiers"])
            byline = (
                f'<span class="toc-byline">{esc(spec["byline"])}</span>'
                if spec["byline"] else ""
            )
            rows.append(
                f'<tr><td class="toc-title"><a href="#{dept_id}">'
                f'<b>{esc(spec["title"])}</b></a></td>'
                f'<td class="toc-promise">{esc(spec["promise"])}</td>'
                f"<td>{badges}{byline}</td></tr>"
            )
        acts.append(
            f'<div class="toc-act"><h3 class="toc-act-title">{esc(act_title)}</h3>'
            f'<table class="toc">{"".join(rows)}</table></div>'
        )
    # A columnist with no tier renders no badge — never a blank one. The absence
    # is the point (§10, and `issue_spec.MASTHEAD_COLUMNISTS`), so it gets a
    # spacer that holds the column rather than a placeholder that looks like a
    # badge failed to load.
    # A columnist with no tier renders no badge — never a blank one. The absence
    # is the point (§10, and `issue_spec.MASTHEAD_COLUMNISTS`), so it gets a
    # spacer that holds the column rather than a placeholder that looks like a
    # badge failed to load. Built outside the f-string: a backslash in an f-string
    # expression is a SyntaxError on Python 3.10, which this repo is pinned to and
    # which has now cost two separate edits.
    no_badge = '<span class="badge-none"></span>'
    masthead_rows = "".join(
        f'<div class="legend-row">{badge(c["tier"]) if c["tier"] else no_badge}'
        f'<div><b>{esc(c["name"])}</b> — {esc(c["bio"])}</div></div>'
        for c in MASTHEAD_COLUMNISTS
    )
    legend = f"""
<div class="legend"><h3>How to read this issue</h3>
  <div class="legend-row">{badge("verified")}<div>Every step cites the Comprehensive
    Rules, and an adversarial checker verified each citation against the full rule text.
    The complete case files are in Judge's Desk.</div></div>
  <div class="legend-row">{badge("data")}<div>Numbers from a seeded, reproducible
    simulation committed to the repository. Same seed, same answer, every time.</div></div>
  <div class="legend-row">{badge("coach")}<div>Judgment — grounded in the verified
    lines and the numbers, but a human call, and labeled as one.</div></div>
  <p class="small soft">Tap any <a class="xref" href="#judges-desk">stack reference</a>
    in the text to jump to its case file; the ☰ button returns you here.</p>
</div>
<div class="legend masthead-block"><h3>The masthead</h3>{masthead_rows}</div>"""
    return f"""
<section class="dept" id="contents" style="--accent:{ACCENT["contents"]}">
  <div class="dept-head"><div><h2 class="dept-title">{esc(spec_c["title"])}</h2></div>
    <div class="dept-promise">{esc(spec_c["promise"])}</div></div>
  <p class="dek">{len(stacks)} verified line(s) · {len(decisions)} decision spread(s)
    · {esc(issue["deck_name"])}</p>
  {"".join(acts)}
  {legend}
</section>""" + folio(spec_c["title"], issue["volume"])


def render_first_turns(issue, plan, prose_doc, goldfish, cards_by_name):
    """The thesis — and, beside it, the conditions the thesis is offered on.

    The rail is emitted by the RENDERER, not asked for by the plan, and that is the
    point. A department whose job is "why it's going to work" will not volunteer
    what it assumed away; radagast's read as "assemble five bodies, swing for
    forty", which is a correct sum over an empty table and silently models no
    blocker, no removal, no instant and three opponents who do nothing. Where the
    engine model exists, its own unsettled questions go here, because the analyst's
    admissions are the honest version of the same caveat and nothing else in the
    magazine prints them.
    """
    dept = plan_dept(plan, "first-turns")
    scope = None
    if goldfish:
        meta = goldfish.get("meta") or {}
        if meta.get("iterations"):
            scope = (f"{meta['iterations']:,} runs of resource development, not of "
                     f"games — no opponent acts in any of them.")
    rail = not_modelled_rail(
        dept.get("not_modelled") or [],
        (_DECK_MAP.get("engine") or {}).get("open_questions") or [],
        scope,
        esc_fn=esc_x,
    )
    return (
        dept_open("first-turns", plan)
        + f'<div class="body-copy">{prose(prose_doc, "how_it_wins")}</div>'
        + rail
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + dept_close("first-turns", issue)
    )


def render_command_zone(issue, plan, commander, goldfish, cards_by_name):
    """The Commander Mandate department (STYLEv3 §3.3)."""
    dept = plan_dept(plan, "command-zone")
    body = []
    if commander and (commander.get("art_crop") or commander.get("image")):
        image = commander.get("art_crop") or commander.get("image")
        img = f'<img src="{esc(image)}" alt="{esc(commander["name"])}" loading="lazy">'
        if commander.get("foil"):
            img = f'<span class="foil">{img}</span>'
        body.append(f'<div class="hero-art">{img}{printing_credit(commander)}</div>')
    if commander:
        cmc = int(commander.get("cmc") or 0)
        identity = "".join(commander.get("color_identity") or []) or "C"
        rows = "".join(
            f"<tr><td>{n}{'st' if n == 1 else 'nd' if n == 2 else 'rd' if n == 3 else 'th'} cast</td>"
            f"<td>{esc(commander.get('mana_cost', ''))}</td><td>+{{{2 * (n - 1)}}}</td>"
            f"<td>{cmc + 2 * (n - 1)}</td></tr>"
            for n in range(1, 5)
        )
        body.append(
            '<table class="tax-ladder"><tr><th>Cast</th><th>Printed cost</th>'
            "<th>Commander tax</th><th>Total mana</th></tr>" + rows + "</table>"
        )
        facts = [
            ("Commander", commander["name"]),
            ("Mana cost", commander.get("mana_cost", "—")),
            ("Color identity", identity),
            ("Type", commander.get("type_line", "—")),
        ]
        if goldfish:
            m = goldfish.get("metrics", goldfish)
            cast = m.get("commander", {})
            if cast:
                facts.append(("Mean cast turn", cast.get("mean_cast_turn", "—")))
                facts.append(("Out by turn 6", f"{cast.get('cast_by_turn_6_rate', 0):.0%}"))
        body.append(fast_facts("Commander File", facts))
    body.append(f'<div class="body-copy">{prose_or_todo(dept)}</div>')
    return (
        dept_open("command-zone", plan)
        + "".join(body)
        + dept_captions(dept, cards_by_name, limit=1)
        + dept_furniture(dept, cards_by_name)
        + dept_close("command-zone", issue)
    )


def prose_or_todo(dept):
    """Departments whose body copy the editor supplies inline via the plan."""
    body = dept.get("body")
    if not body:
        return TODO
    return "".join(f"<p>{esc(p.strip())}</p>" for p in str(body).split("\n\n") if p.strip())


def render_by_the_numbers(issue, plan, goldfish, cards_by_name):
    dept = plan_dept(plan, "by-the-numbers")
    if not goldfish:
        return dept_open("by-the-numbers", plan) + TODO + dept_close("by-the-numbers", issue)
    m = goldfish["metrics"]
    meta = goldfish["meta"]
    opening = m.get("opening_hand", {})
    commander = m.get("commander", {})

    meters = [power_meter("Keepable first sevens", opening.get("keep_first_seven_rate", 0))]
    if commander:
        meters.append(power_meter("Commander cast by turn 6",
                                  commander.get("cast_by_turn_6_rate", 0)))
    for target in m.get("targets") or []:
        meters.append(power_meter(target.get("label", "Target"),
                                  target.get("assembled_rate", 0)))

    turns = sorted(m["land_drop_hit_rate_by_turn"], key=int)
    def row(label, values):
        return f"<tr><th>{esc(label)}</th>" + "".join(f"<td>{esc(v)}</td>" for v in values) + "</tr>"
    table = (
        '<table class="data">'
        + row("Turn", turns)
        + row("Land drop hit", [f"{m['land_drop_hit_rate_by_turn'][t]:.0%}" for t in turns])
        + row("Mean mana", [f"{m['mean_available_mana_by_turn'][t]:.1f}" for t in turns])
        + row("Mean bodies", [f"{m['mean_bodies_by_turn'][t]:.1f}" for t in turns])
        + "</table>"
    )
    assumptions = "".join(f"<li>{esc(a)}</li>" for a in meta.get("model_assumptions", []))
    facts = fast_facts("Simulation File", [
        ("Iterations", f"{meta.get('iterations', 0):,}"),
        ("Seed", meta.get("seed", "—")),
        ("Decklist sha", str(meta.get("decklist_sha256", ""))[:12] or "—"),
    ])
    return (
        dept_open("by-the-numbers", plan)
        + "".join(meters) + table + facts
        + f'<div class="assumptions"><b>What this model does and does not do.</b> '
          f"These runs simulate resource development, not full games. Every assumption "
          f"is stated:<ul>{assumptions}</ul></div>"
        + dept_furniture(dept, cards_by_name)
        + dept_close("by-the-numbers", issue)
    )


def _rows(pairs):
    """`<dt>/<dd>` rows for the pairs that actually have a value."""
    return "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in pairs if v)


def _seat_label(seat):
    """`opponent_a` → "Opponent A". Seats the coach named keep their own name.

    `scenario_facts` emits machine keys because it is read by agents; the reader
    should not meet a snake_case identifier on the page.
    """
    seat = str(seat or "")
    if re.fullmatch(r"opponent[_ ]?\w*", seat, re.I):
        tail = seat.split("_", 1)[1] if "_" in seat else ""
        return f"Opponent {tail.upper()}".strip()
    return seat


def _as_text(value):
    """A board field is a list of entries or a prose blob. Render either."""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value if v)
    return str(value or "")


def stack_headline(title):
    """`(headline, subtitle)` from a stack's authored title.

    Stack titles are written for the RESOLVER, and they read like it: a median of
    74 characters and up to 157, because they carry the question the scenario was
    posed to answer. Set at feature size that runs three lines of display type
    before the reader has learned anything.

    They almost all share one shape, though — a real headline, a colon, then the
    question: *"The Frostfang trap: flashed in after blockers are declared, does
    deathtouch apply…"*. Splitting there is free and authored rather than invented:
    across the 49 presentable stacks the head runs a median of 36 characters, only
    two exceed 60, and the six titles with no colon keep their whole text. The
    question is not dropped — it becomes the deck under the headline.
    """
    title = str(title or "")
    head, sep, tail = title.partition(":")
    if not sep or len(head.strip()) < 8:
        return title, ""
    return head.strip(), tail.strip()


def _seat_line(head, pairs):
    """One seat, one line: who, then labelled runs of what they have.

    Replaces a bordered card holding a `<dl>`. Same fields, a fifth of the height
    — a board state is a spec sheet and belongs at spec-sheet weight, especially
    in The Kill where it is the preamble to the thing the reader came for.
    """
    runs = "".join(
        f'<span class="run">' +
        (f'<i>{esc(label)}</i> ' if label else "") +
        f'{esc(value)}</span>'
        for label, value in pairs if value)
    if not runs:
        return ""
    return (f'<div class="seat"><span class="seat-who">{esc(head)}</span>'
            f'{runs}</div>')


def stack_entry_text(entry):
    """What a `scenario.stack` entry says, under either key the corpus uses.

    53 entries carry `object`; 11 carry `item` (radagast 8, yawgmoth-swarm 3), and
    reading only `object` printed those eleven as an empty `<b></b>` on the
    published page — an unnamed thing on the stack, in the department whose whole
    job is showing what is on the stack.

    Both keys stay valid and the scenario files are NOT normalised: a scenario block
    is a cache fingerprint input, so tidying the corpus would MISS every stack
    routine and cost 42 respawns to change nothing a reader can see.
    """
    entry = entry or {}
    return entry.get("object") or entry.get("item") or ""


def render_board_block(scenario, label="The board"):
    """The board, stated before anything argues about it.

    Every stack file carries `board`, `hand`, `graveyard`, `mana_available` and an
    ordered `stack`, and until now the magazine rendered exactly one of them —
    `question` — so the reader met a hundred-word question about a board they had
    never been shown. Board parsing is NOT reimplemented here: `scenario_facts`
    already splits a board into bodies/permanents/lands/spent and already reads both
    corpus opponent shapes, and it was written because agents kept getting these
    same lookups wrong in prose.

    Two board shapes, both real: stack scenarios list `board.you` as entries, and
    decision scenarios write it as one prose string with a `table` beside it.
    Anything absent is omitted — never invented, never a placeholder.

    **Reference weight, not feature weight.** It first shipped as a grid of
    bordered cards each holding a `<dl>`, and on radagast that came to 3,782px
    across seven lines — MORE than the 4,835px of stack theatre it exists to
    introduce, in the issue's most narrative department. Every field it printed
    then it prints now; they are set as labelled inline runs, one line per seat,
    so the board reads as the spec sheet it is and the line that follows gets the
    room. `board_bodies` and `opponents_of` still do all the parsing.
    """
    scenario = scenario or {}
    board = scenario.get("board") or {}
    seats = []

    you = board.get("you")
    if isinstance(you, (list, tuple)):
        split = board_bodies(you)
        pairs = [
            ("Creatures", _as_text(split["creature_bodies"])),
            ("Permanents", _as_text(split["other_permanents"])),
            ("Lands", _as_text(split["lands"])),
            # Listed but NOT on the battlefield — folding it into either side
            # changes the body count, which is what these engines are bounded by,
            # so it keeps its own labelled run rather than joining Permanents.
            ("Already paid", _as_text(split["spent_paying_a_cost"])),
        ]
    else:
        pairs = [("Battlefield", _as_text(you))]
    pairs += [
        ("Hand", _as_text(scenario.get("hand"))),
        ("Graveyard", _as_text(scenario.get("graveyard"))),
        ("Mana", _as_text(scenario.get("mana_available"))),
    ]
    if any(v for _, v in pairs):
        seats.append(_seat_line("You", pairs))

    for opp in opponents_of(scenario):
        life = opp.get("life")
        head = _seat_label(opp["seat"]) + (f" — {life} life" if life is not None else "")
        seats.append(_seat_line(head, [("", _as_text(opp.get("board")))]))

    # Decision scenarios describe the rest of the pod as one prose `table` field.
    table = board.get("table")
    if table:
        seats.append(_seat_line("The table", [("", _as_text(table))]))

    # `pos` 0 is the BOTTOM of the stack (docs/pilot.md, validate_stack), so the
    # reader's first question — what resolves next — is the LAST entry.
    entries = [s for s in (scenario.get("stack") or []) if s]
    on_stack = ""
    if entries:
        items = []
        for i, obj in enumerate(reversed(entries)):
            # Stack scenarios carry {pos, object, controller, note}; decision
            # scenarios carry a bare string per entry. Both are in the corpus.
            if isinstance(obj, dict):
                what, note, who = stack_entry_text(obj), obj.get("note"), obj.get("controller")
            else:
                what, note, who = str(obj), None, None
            meta = " · ".join(x for x in (f"controlled by {who}" if who else "", note) if x)
            items.append(
                f'<li><b>{esc(what)}</b>'
                + (f'<div class="effect">{esc(meta)}</div>' if meta else "")
                + ("" if i else '<span class="lbl"> — resolves first</span>')
                + "</li>")
        on_stack = ('<span class="lbl">On the stack, top first</span>'
                    f'<ol>{"".join(items)}</ol>')

    if not seats and not on_stack:
        return ""
    return (f'<div class="scenario"><span class="lbl">{esc(label)}</span>'
            + (f'<div class="seats">{"".join(seats)}</div>' if seats else "")
            + on_stack + "</div>")


def render_after_block(final):
    """What the board looks like when the line is done.

    `final_state.you` is free-form — the corpus carries 80+ distinct keys across 48
    files — so this reads only the three that are common and well-typed and skips
    the rest rather than guessing at a schema that does not exist.
    """
    final = final or {}
    you = final.get("you") if isinstance(final.get("you"), dict) else {}
    rows = _rows((
        ("Your life", you.get("life")),
        ("Your battlefield", _as_text(you.get("battlefield"))),
    ))
    seats = []
    for opp in final.get("opponents") or []:
        if isinstance(opp, dict):
            name = _seat_label(opp.get("seat") or opp.get("name") or "Opponent")
            life = opp.get("life")
            if life is not None:
                seats.append(f"{name}: {life}")
    rows += _rows((("Opponents", ", ".join(str(s) for s in seats)),))
    if not rows:
        return ""
    return ('<div class="scenario"><span class="lbl">After the line</span>'
            f'<dl class="branch">{rows}</dl></div>')


"""A one-line HOLDING for the front of the book was tried here and removed.

The editor's note is right — the front of book should give the verdict and send
the argument to the back — and deriving that verdict from the first sentence of
`final_state.summary` looked free: no re-planning, no new authored field, and the
two could never contradict each other.

It does not survive contact with the corpus. Measured across radagast's seven
stacks, one is perfect ("The trap works."), and then: 003's first sentence is
"Beast Whisperer draws you a card" — true, and NOT the holding, which is that 36
damage is short of 40. 005 yields "VERDICTS. (1) LEGALITY: yes". 001, 006 and 007
run past 150 characters because the summaries are built on colons and semicolons
rather than sentences.

A wrong verdict in the one department that exists for correctness is worse than a
long right one, and the renderer may not summarise proof (STYLEv3 §5.1). A real
holding needs an AUTHORED field on the stack, written by the resolver alongside
the resolution it is a holding for — a schema change, not a rendering trick.
Recorded so the next attempt starts from the measurement rather than the idea.
"""


def render_the_kill(issue, plan, stacks, cards, prose_doc, cards_by_name):
    """The lines, argued — and each one as a stack you can walk through.

    The board is stated first (who has what, what is on the stack), then the
    THEATRE resolves it a step at a time. Judge's Desk still carries the verbatim
    record and is deliberately untouched: the theatre is a way through the proof,
    not a replacement for it, and §5.1 forbids the renderer summarising proof.

    **`features` decides which lines get a spread, and the rest get a row.** The
    department rendered a full theatre per passing stack, which is right at seven
    and wrong at eleven: yawgmoth-swarm's Kill reached **44,119 words — 42% of its
    issue** — because its loops run 11–14 steps each and every one was staged. The
    editors have been naming a feature set in prose since the first issue ("Three
    feature spreads: stack 003, 006, 005") and nothing read it; `features` is that
    note made machine-readable. Absent, every stack still features, so no existing
    issue moves without its plan asking.

    What is NOT featured is indexed rather than dropped. A silent cut reads as
    "that is all of them", and here it would also strand Judge's Desk's own
    `↩ Back to this line in The Kill` link — so an index row carries the same
    `line-<id>` anchor a spread would. It is a pointer, not a summary: the id, the
    authored head, the counts, and a link into the case.
    """
    dept = plan_dept(plan, "the-kill")
    featured, indexed = split_features(dept, stacks)
    spreads = []
    for stack in featured:
        sid = stack["id"]
        checker = stack.get("checker", {})
        intro = prose(prose_doc, "combo_lines", sid)
        headline, subtitle = stack_headline(stack["title"])
        resolution = stack.get("resolution", {})
        final = resolution.get("final_state", {})
        spreads.append(f"""
<article class="rule-top" id="line-{esc(sid)}">
  <div class="kicker">Verified line {esc(sid)}</div>
  <h3 class="line-head">{esc(headline)}</h3>
  {f'<p class="line-dek">{esc(subtitle)}</p>' if subtitle else ""}
  <div class="body-copy">{intro}</div>
  {render_board_block(stack.get("scenario"))}
  {stack_theatre(sid, resolution.get("steps"), cards, esc_fn=esc_x)}
  <p><b>Result.</b> {esc(final.get("summary", ""))}</p>
  <a class="dossier-pointer" href="#case-{esc(sid)}">
    Full dossier: Judge's Desk, Case A-{esc(sid)} →</a>
  <span style="margin-left:10px">{badge("verified")} cleared in
    {esc(checker.get("iterations", "?"))} review cycle(s)</span>
</article>""")
    return (
        dept_open("the-kill", plan)
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + ("".join(spreads) or TODO)
        + kill_index(indexed, len(stacks), prose_doc)
        + dept_close("the-kill", issue)
    )


def split_features(dept, stacks):
    """`(featured, indexed)` — the plan's `features`, in the order it wrote them.

    Unknown ids are skipped here rather than raising: `validate-issue` is where a
    bad id is an error, and a renderer that crashes on one turns a copy mistake
    into a missing magazine. Everything not featured is indexed, in id order,
    which is the order a reader scans a list of case numbers in.
    """
    named = [str(i) for i in (dept.get("features") or [])]
    if not named:
        return list(stacks), []
    by_id = {s["id"]: s for s in stacks}
    featured = [by_id[i] for i in named if i in by_id]
    chosen = {s["id"] for s in featured}
    return featured, [s for s in stacks if s["id"] not in chosen]


def kill_index(indexed, total, prose_doc):
    """The lines without a spread — **argued in full, just not staged.**

    The first cut here dropped the authored `combo_lines` intro and printed a
    bare pointer row, and measuring it showed that was throwing away the wrong
    thing. On yawgmoth-swarm a rendered stack is ~4,000 words and its intro is
    **77–144** — the board block and the theatre are essentially all of it. So
    the argument stays and the staging is what gets rationed, which is also the
    only reading that keeps the department's promise: The Kill is "the winning
    lines, ARGUED and affirmed", and an index that dropped the argument would be
    keeping the title and cutting the thing it names.

    The count sentence states the TRUE total, because a cut that does not say it
    cut reads as completeness. Empty when every line was featured, so an issue
    whose plan has no `features` renders byte-identically to before the key
    existed.
    """
    if not indexed:
        return ""
    rows = []
    for stack in indexed:
        sid = stack["id"]
        resolution = stack.get("resolution", {})
        n_steps = len(resolution.get("steps") or [])
        n_cites = sum(len(s.get("citations") or [])
                      for s in resolution.get("steps") or [])
        intro = prose(prose_doc, "combo_lines", sid)
        final = resolution.get("final_state", {})
        rows.append(f"""
<article class="kill-row" id="line-{esc(sid)}">
  <div class="kill-row-head">
    <span class="case-id">A-{esc(sid)}</span>
    <span class="case-title">{esc(stack_headline(stack["title"])[0])}</span>
    <span class="case-meta">{n_steps} step{"" if n_steps == 1 else "s"}
      · {n_cites} citation{"" if n_cites == 1 else "s"}</span>
  </div>
  <div class="body-copy">{intro}</div>
  <p class="kill-row-result"><b>Result.</b> {esc(final.get("summary", ""))}</p>
  <a class="xref" href="#case-{esc(sid)}">Full dossier: Judge's Desk, Case
    A-{esc(sid)} →</a>
</article>""")
    n = len(indexed)
    return (f'<div class="kill-index">'
            f'<h4 class="kill-index-head">Also on the record</h4>'
            f'<p class="kill-index-dek">{n} of this deck\'s {total} verified lines '
            f'are argued here without the stack theatre. They cleared the same '
            f'review; the step-by-step resolution and every citation are in '
            f'Judge\'s Desk.</p>{"".join(rows)}</div>')


def letter_teases(plan, limit=3):
    """`(title, line)` for the IN THIS ISSUE rail — authored, else derived.

    The rail is what makes the letter a preview of the magazine rather than a note
    stapled to the front of it, so it must never be empty. Authored teases win;
    absent them the editor's own department deks are already exactly this — one
    line each saying what a department is about — so they are borrowed in reading
    order, skipping the letter itself and the two pieces of furniture.

    Derived beats blank and authored beats derived; nothing here invents a line.
    """
    authored = (plan_dept(plan, "editors-letter").get("in_this_issue") or [])
    out = []
    for entry in authored[:limit]:
        dept_id = (entry or {}).get("department")
        title = DEPARTMENT_BY_ID.get(dept_id, {}).get("title") or dept_id or ""
        if title and (entry or {}).get("line"):
            out.append((title, entry["line"]))
    if out:
        return out
    planned = {d.get("id"): d for d in plan.get("departments") or []}
    for dept_id in DEPARTMENT_IDS:
        if dept_id in ("cover", "contents", "editors-letter"):
            continue
        dek = (planned.get(dept_id) or {}).get("dek")
        if dek:
            out.append((DEPARTMENT_BY_ID[dept_id]["title"], dek))
        if len(out) == limit:
            break
    return out


def render_editors_letter(issue, plan, commander, prose_doc, cards_by_name):
    """One page, unbadged. What this deck is and whether it is for you.

    The only department signed by someone who holds no evidence tier, so it is
    also the only one that may not make a claim needing one (STYLEv3 §7.7). The
    renderer cannot enforce that — it is a judgment about sentences — but the
    absent badge in the department head is what makes the difference visible.

    The `letterhead` carries the whole page. The card it opens on is the plan's
    `letter_card` where the editor named one and the commander otherwise, because
    a letter about a deck should be looking at something and the commander is the
    one card every issue is guaranteed to have.
    """
    dept = plan_dept(plan, "editors-letter")
    card = cards_by_name.get(dept.get("letter_card") or "") or commander
    spec = DEPARTMENT_BY_ID["editors-letter"]
    volume = f'Vol. {issue.get("volume", "—")} · {issue.get("deck_name", "")}'
    byline = spec["byline"] or ""
    # "Editor-in-Chief Margot Stet" → hand-signed name over the printed role, the
    # way a letter is actually signed. Split rather than restated: the masthead is
    # the single source for both halves and a retyped name drifts from it.
    name, _, _tail = byline.rpartition("Editor-in-Chief ")
    signed = _tail or byline
    return (
        dept_open("editors-letter", plan)
        + letterhead(MASTHEAD, "From the Editor", volume,
                     prose(prose_doc, "editors_letter"),
                     card=card, teases=letter_teases(plan),
                     signature=signed, role="Editor-in-Chief")
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + dept_close("editors-letter", issue)
    )


def render_pilots_log(issue, plan, prose_doc, cards_by_name):
    """Three pilots arguing about one deck — a conversation, not three essays.

    `pilots_log` is a LIST of turns, not a block of prose, and the shape is the
    contract: a turn carries the voice that speaks it, so the renderer can label
    each one and a reader can follow who is answering whom. Handed a string it
    renders nothing and says so, rather than printing an unattributed wall — an
    unlabelled panel is just prose with quotation marks.
    """
    dept = plan_dept(plan, "pilots-log")
    turns = (prose_doc or {}).get("pilots_log")
    if not isinstance(turns, list) or not turns:
        body = TODO
    else:
        blocks = []
        # Turn 0 is the HOT TAKE when it says so — the claim the rest of the
        # department is an argument with. It is still a turn in the list rather
        # than a sibling key, so the voice lint, the ordering and "who answers
        # whom" all keep working on one structure. A panel whose opener sat in a
        # different field would be a panel with a paragraph nobody is checking.
        if (turns[0] or {}).get("kind") == "hot-take":
            blocks.append(hot_take((turns[0] or {}).get("text", ""),
                                   (turns[0] or {}).get("voice", ""),
                                   esc_fn=esc_x_paras))
            turns = turns[1:]
        for turn in turns:
            voice = (turn or {}).get("voice", "")
            text = (turn or {}).get("text", "")
            # The speaker's own accent, so a reader tracks the argument by colour
            # before they read the name — the same trick the constellation uses.
            key = ("coach" if "Sunny" in voice else
                   "verified" if "Vera" in voice else
                   "data" if "Ledger" in voice else "none")
            blocks.append(
                f'<div class="turn turn-{key}">'
                f'<div class="turn-voice">{esc(voice)}</div>'
                f'<div class="turn-text">{esc_x_paras(text)}</div></div>')
        body = f'<div class="panel">{"".join(blocks)}</div>'
    return (
        dept_open("pilots-log", plan)
        + body
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + dept_close("pilots-log", issue)
    )


def render_at_the_table(issue, plan, prose_doc, tutor_guide, cards_by_name):
    """Act III as ONE department: who wants you dead, then what you go get.

    The three sections it replaces are still rendered by their own functions on
    the eight decks that carry them; this composes the same three BODIES under one
    opener, one byline and one folio. What it drops is duplication, not content —
    two department heads, two bylines, two promises and two folios, all of them
    saying "Coach Sunny Brightside" about the same act.

    The bodies are pulled from the plan entry for `at-the-table` itself: the threat
    boxes come from its `threats`, the prose from the same `threat_assessment` and
    `matchups` keys, and the tutor guide is unchanged — it was never department-
    scoped. An issue that plans the three separately is untouched by this function.
    """
    dept = plan_dept(plan, "at-the-table")

    # Subheads keep the editor's own headline and dek for the section they
    # replace. A merge that threw away three written headlines and substituted the
    # department titles would be cheaper prose than the issue already had — the
    # point was to stop repeating the CHROME, not to stop writing.
    subheads = dept.get("subheads") or {}

    def sub(key, fallback, body):
        if not body:
            return ""
        spec = subheads.get(key) or {}
        title = spec.get("headline") or fallback
        dek = f'<p class="dek">{esc(spec["dek"])}</p>' if spec.get("dek") else ""
        return f'<h2 class="act-sub">{esc(title)}</h2>{dek}{body}'

    boxes = []
    for entry in dept.get("threats", []):
        level = entry.get("level")
        if level is None:
            level = round(float(entry.get("rate", 0.5)) * 5)
        boxes.append(threat_box(
            entry.get("archetype", ""), entry.get("meter_label", "Threat"), level,
            f'<p>{esc_x(entry.get("read", ""))}</p>'
            f'<p><b>Your outs:</b> {esc_x(", ".join(entry.get("outs", [])))}</p>',
        ))

    politics = prose(prose_doc, "threat_assessment")
    matchups = prose(prose_doc, "matchups")
    return (
        dept_open("at-the-table", plan)
        + (f'<div class="body-copy">{politics}</div>' if politics else "")
        + sub("enemy", "Know Your Enemy", "".join(boxes)
              + (f'<div class="body-copy">{matchups}</div>' if matchups else ""))
        + sub("tutors", "Fetch Quests", tutor_bodies(tutor_guide, cards_by_name))
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + dept_close("at-the-table", issue)
    )


def render_whats_your_play(issue, plan, decisions, cards_by_name):
    dept = plan_dept(plan, "whats-your-play")
    spreads = []
    for decision in decisions:
        scenario = decision.get("scenario", {})
        branches = "".join(f"""
<div class="branch"><h4>{esc(b.get("choice", ""))}</h4>
  <dl><dt>The line</dt><dd>{esc_x_paras(b.get("line", ""))}</dd>
  <dt>Signals sent</dt><dd>{esc_x_paras(b.get("signals", ""))}</dd>
  <dt>Coalition risk</dt><dd>{esc_x_paras(b.get("coalition_risk", ""))}</dd>
  <dt>Read</dt><dd>{esc_x_paras(b.get("coaching", ""))}</dd></dl></div>"""
            for b in decision.get("branches", []))
        rec = decision.get("recommendation", {})
        spreads.append(f"""
<article class="rule-top" id="play-{esc(decision.get("id", ""))}">
  <h3 style="font-family:var(--display);font-size:1.4em">{esc(decision["title"])}</h3>
  {render_board_block(scenario)}
  <div class="scenario"><b>{esc(scenario.get("question", ""))}</b></div>
  <div class="branches">{branches}</div>
  <div class="verdict"><b>Our call: {esc(rec.get("choice", ""))}</b><br>
    {esc_x_paras(rec.get("rationale", ""))}</div>
</article>""")
    return (
        dept_open("whats-your-play", plan)
        + dept_furniture(dept, cards_by_name)
        + ("".join(spreads) or TODO)
        + dept_close("whats-your-play", issue)
    )


def render_the_99(issue, plan, cards, prose_doc, synergy, cards_by_name):
    """Ranked roster — load-bearing cards lead, depth reads as depth.

    The plan's optional `roster` groups cards by role; anything it doesn't name
    falls into a final "Depth" group so no card silently vanishes.
    """
    dept = plan_dept(plan, "the-99")
    roles = (prose_doc or {}).get("card_roles", {})
    # Every other department routes its copy through prose(), which makes a missing
    # key a visible TODO. card_roles is a dict, so it can't — and an absent one used
    # to render a full grid of blank blurbs, which reads as "these cards need no
    # explanation" rather than "nobody wrote this yet". Say it out loud instead.
    roles_todo = "" if roles else TODO
    main = cards
    by_name = {c["name"]: c for c in main}
    stage_of = engine_stage_of()

    # THE CITIES ARE THE ROSTER, when the plan asks for it. The deck map already
    # answers "which cards do the same job" from the embeddings; grouping the grid
    # any other way means the reader meets two taxonomies on facing pages and has
    # to reconcile them. Ordered by size, so the deck's centre of mass leads.
    if dept.get("group_by") == "city" and (_DECK_MAP.get("doc") or {}).get("regions"):
        doc = _DECK_MAP["doc"]
        city_of = {c["name"]: c["city"] for c in doc["cards"]}
        cities = sorted((r for r in doc["regions"] if r["level"] == 0),
                        key=lambda r: (-r["count"], r["id"]))
        sections = []
        for region in cities:
            index = int(region["id"].rsplit("-", 1)[-1])
            members = [c for c in main
                       if city_of.get(c["name"]) == index and not c["is_commander"]]
            if not members:
                continue
            tiles = "".join(
                card_tile(c, roles, synergy, anchor_id=CARD_ANCHORS.get(c["name"]),
                          stage=stage_of.get(c["name"]))
                for c in members)
            sections.append(
                city_head(index,
                          region.get("label") or region.get("fallback") or "",
                          len(members), region.get("verified_count", 0),
                          region.get("gloss"))
                + f'<div class="card-grid">{tiles}</div>')
        # Any card the map could not place (an unresolved name) still gets a seat.
        stray = [c for c in main
                 if c["name"] not in city_of and not c["is_commander"]]
        if stray:
            tiles = "".join(
                card_tile(c, roles, synergy, anchor_id=CARD_ANCHORS.get(c["name"]),
                          stage=stage_of.get(c["name"]))
                for c in stray)
            sections.append("<h3>Unmapped</h3>"
                            f'<div class="card-grid">{tiles}</div>')
        return (
            dept_open("the-99", plan)
            + roles_todo
            + dept_furniture(dept, cards_by_name)
            + "".join(sections)
            + dept_captions(dept, cards_by_name)
            + dept_close("the-99", issue)
        )

    groups, placed = [], set()
    for entry in dept.get("roster", []):
        named = [by_name[n] for n in entry.get("cards", []) if n in by_name]
        if not named:
            continue
        placed.update(c["name"] for c in named)
        groups.append((entry.get("role", "Roster"), named))
    remainder = [c for c in main if c["name"] not in placed and not c["is_commander"]]
    if remainder:
        groups.append(("Depth", remainder))
    if not groups:  # no roster in the plan — flat grid, decklist order
        groups = [("", [c for c in main if not c["is_commander"]])]

    sections = []
    for role, group_cards in groups:
        heading = f"<h3>{esc(role)}</h3>" if role else ""
        tiles = "".join(
            card_tile(c, roles, synergy, anchor_id=CARD_ANCHORS.get(c["name"]),
                      stage=stage_of.get(c["name"]))
            for c in group_cards)
        sections.append(f'{heading}<div class="card-grid">{tiles}</div>')
    tiles = "".join(sections)

    return (
        dept_open("the-99", plan)
        + roles_todo
        + tiles
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + dept_close("the-99", issue)
    )


def render_featured_artist(issue, plan, cards, cards_by_name):
    """Who painted the deck. Facts computed here, the choice authored in the plan.

    The department adapts: a real standout leads with a gallery; a deck of
    all-different artists tells a breadth story instead. Either way the honesty
    notes from the analysis are surfaced, so concentration is never dressed up
    as curation it wasn't (STYLEv3 §7.6).
    """
    from manamap.pilot.artist_credits import analyze

    dept = plan_dept(plan, "featured-artist")
    roster = plan_dept(plan, "the-99").get("roster")
    credits = analyze(cards, roster)
    totals = credits["totals"]

    featured = dept.get("featured") or {}
    # The plan may name the artist; otherwise the computed standout leads.
    name = featured.get("artist") or (credits["standout"] or {}).get("artist")
    entry = next((r for r in credits["ranking"] if r["artist"] == name), None)

    body = []
    if entry:
        # Lead with the commander when the featured artist painted them, else the
        # most load-bearing card they did paint, else whatever comes first.
        owned = [cards_by_name[c] for c in entry["cards"] if c in cards_by_name]
        roster_order = [n for g in (roster or []) for n in g.get("cards", [])]
        hero = next(
            (c for c in owned if c.get("is_commander")),
            next((cards_by_name[n] for n in roster_order
                  if n in cards_by_name and cards_by_name[n].get("artist") == name),
                 owned[0] if owned else None),
        )
        if hero:
            lead = f'{entry["entries"]} of {totals["entries"]} cards'
            body.append(card_figure(
                hero["name"], hero.get("art_crop") or hero.get("image"),
                f"<b>{esc(name)}</b> — {esc(lead)} in this deck.",
                hero.get("scryfall_uri"), hero))
        if featured.get("note"):
            paragraphs = "".join(
                f"<p>{esc_x(p.strip())}</p>"
                for p in str(featured["note"]).split("\n\n") if p.strip()
            )
            body.append(f'<div class="body-copy">{paragraphs}</div>')
        tiles = "".join(
            card_tile(cards_by_name[c], {}, {}, printing=True)
            for c in entry["cards"] if c in cards_by_name
        )
        body.append(f'<h3>Every {esc(name)} card in the deck</h3>'
                    f'<div class="card-grid">{tiles}</div>')

    # The whole table, because the story is the contrast: the groups this
    # artist dominates next to the ones where every card is a different hand.
    if credits["roster_overlap"]:
        rows = "".join(
            f'<tr><td>{esc(r["group"])}</td><td>{esc(r["artist"])}</td>'
            f'<td>{r["painted"]} of {r["of"]}</td>'
            f'<td>{r["distinct_artists"]}</td></tr>'
            for r in credits["roster_overlap"]
        )
        body.append('<h3>Against the roster</h3><table class="data">'
                    "<tr><th>Group</th><th>Most cards by</th><th>Painted</th>"
                    "<th>Artists in group</th></tr>" + rows + "</table>")

    others = dept.get("also_worth_noting") or []
    if others or credits["clusters"]:
        noted = {o.get("artist"): o.get("note", "") for o in others}
        items = []
        for cluster in credits["clusters"][:6]:
            note = noted.get(cluster["artist"], "")
            items.append(
                f'<div class="branch"><h4>{esc(cluster["artist"])} '
                f'({cluster["entries"]})</h4>'
                f'<p class="small">{esc(", ".join(cluster["cards"]))}</p>'
                f'{f"<p>{esc(note)}</p>" if note else ""}</div>'
            )
        body.append(f'<h3>Also worth noting</h3><div class="branches">{"".join(items)}</div>')

    treat = credits["treatments"]
    facts = [("Distinct artists", totals["artists"]),
             ("Cards counted", totals["entries"]),
             ("Borderless", treat["borderless"]),
             ("Foil", treat["foil"])]
    # Set dispersion is its own story: a reprint stream assembled one card at a
    # time reads very differently from a drop bought whole.
    for entry in credits["sets"][:2]:
        facts.append((entry["set_name"] or entry["set"] or "Set",
                      f'{entry["entries"]} cards · {entry["artists"]} artists'))
    for run in credits["drop_runs"][:2]:
        facts.append((f'Drop run ({run["set"]})',
                      f'#{run["from"]}–{run["to"]}, {run["entries"]} cards'))
    body.append(fast_facts("Art File", facts))

    if credits["notes"]:
        notes = "".join(f"<li>{esc(n)}</li>" for n in credits["notes"])
        body.append('<div class="assumptions"><b>How these numbers are counted.</b>'
                    f"<ul>{notes}</ul></div>")

    return (
        dept_open("featured-artist", plan)
        + "".join(body)
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + dept_close("featured-artist", issue)
    )


def render_keep_or_ship(issue, plan, prose_doc, goldfish, cards_by_name):
    dept = plan_dept(plan, "keep-or-ship")
    meter = ""
    if goldfish:
        rate = goldfish["metrics"].get("opening_hand", {}).get("keep_first_seven_rate", 0)
        meter = power_meter("Keepable first sevens (simulated)", rate)
    hands = "".join(f"""
<div class="branch"><h4>{esc(h.get("verdict", ""))}</h4>
  <p style="font-size:.92em">{esc(", ".join(h.get("cards", [])))}</p>
  <p class="small soft">{esc_x(h.get("why", ""))}</p></div>"""
        for h in dept.get("hands", []))
    hands_html = f'<div class="branches">{hands}</div>' if hands else ""
    return (
        dept_open("keep-or-ship", plan)
        + meter + hands_html
        + f'<div class="body-copy">{prose(prose_doc, "mulligan")}</div>'
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + dept_close("keep-or-ship", issue)
    )


def tutor_bodies(tutor_guide, cards_by_name):
    """The tutor guide, with no department chrome around it.

    It was Fetch Quests' whole body and is now At the Table's third subhead. It
    stayed a function through the merge rather than being inlined, because for one
    fleet regeneration both shapes rendered and two copies of a renderer is how
    the same section starts saying different things on different decks with
    nothing failing. Keeping it separate also keeps `render_at_the_table` legible
    as three arguments rather than one four-hundred-line function.
    """
    parts = []
    tutors = (tutor_guide or {}).get("tutors") or []
    if tutor_guide and tutor_guide.get("assessment"):
        parts.append(f'<div class="body-copy">'
                     f'{esc_x_paras(tutor_guide["assessment"])}</div>')
    for entry in tutors:
        name = entry.get("card", "")
        card = cards_by_name.get(name) or {}
        rows = "".join(
            f'<div class="callout"><div class="n">{i}</div><div>'
            f'<span class="t">{esc_x(t.get("scenario", ""))}</span>'
            f'<b>Fetch:</b> {esc_x(t.get("fetch", ""))}<br>'
            f'{esc_x(t.get("why", ""))}</div></div>'
            for i, t in enumerate(entry.get("targets", []), 1)
        )
        note = (f'<p class="small soft">{esc_x(entry["notes"])}</p>'
                if entry.get("notes") else "")
        image = (f'<img src="{esc(card["image"])}" alt="{esc(name)}" '
                 f'loading="lazy" style="max-width:180px">'
                 if card.get("image") else "")
        parts.append(
            f'<article class="rule-top">'
            f'<h3 style="font-family:var(--display);font-size:1.25em">'
            f'{esc_x(name)}</h3>{image}{rows}{note}</article>')
    if not tutors:
        parts.append(
            '<div class="body-copy"><p>No tutors in this 99. The deck finds '
            'its pieces the honest way — redundancy and card draw — so every '
            'game plays out a little differently, and the Coach is fine with '
            'that.</p></div>')
    return "".join(parts)


def render_sources_say(issue, plan, mana, prose_doc, cards_by_name):
    """The mana audit (◆), straight from mana_analysis.json — deterministic
    Python, no agent. Ledger's mana_base prose key narrates it."""
    dept = plan_dept(plan, "sources-say")
    if not mana:
        return (dept_open("sources-say", plan) + TODO
                + dept_close("sources-say", issue))
    lands = mana.get("lands", {})
    sources = mana.get("sources", {})
    pips = mana.get("pips", {})
    probs = mana.get("on_curve_probability", {})
    shares = mana.get("shares", {})
    targets = mana.get("source_targets", {})

    meters = "".join(
        power_meter(f"{colour} on curve (lands + ramp)",
                    probs.get("with_rocks_and_dorks", {}).get(colour, 0))
        for colour in sorted(pips)
    )
    colour_rows = "".join(
        f"<tr><th>{esc(colour)}</th>"
        f"<td>{pips[colour]['total_pips']:g}</td>"
        f"<td>{esc(pips[colour]['effective_pips'])}</td>"
        f"<td>{esc(sources.get('lands', {}).get(colour, 0))}</td>"
        f"<td>{esc(sources.get('total', {}).get(colour, 0))}</td>"
        f"<td>{esc(targets.get(colour, 0))}</td>"
        f"<td>{shares.get(colour, {}).get('pip_share', 0):.0%} / "
        f"{shares.get(colour, {}).get('source_share', 0):.0%}</td></tr>"
        for colour in sorted(pips)
    )
    colour_table = (
        '<table class="data"><tr><th>Colour</th><th>Total pips</th>'
        '<th>Heavy pip</th><th>Land sources</th><th>+ ramp</th>'
        '<th>90% yardstick</th><th>Pip / source share</th></tr>'
        f'{colour_rows}</table>'
    )
    class_rows = "".join(
        f"<tr><th>{esc(cls.replace('-', ' ').title())}</th><td>{esc(n)}</td></tr>"
        for cls, n in (lands.get("classes") or {}).items()
    )
    class_table = (
        f'<table class="data"><tr><th>Land class</th><th>Count</th></tr>'
        f'{class_rows}</table>' if class_rows else ""
    )
    ramp = mana.get("ramp", {})
    facts = fast_facts("Mana File", [
        ("Lands", lands.get("total", 0)),
        ("Enter tapped", lands.get("enters_tapped", 0)),
        ("Rocks", ramp.get("ramp:rock", 0)),
        ("Dorks", ramp.get("ramp:dork", 0)),
        ("Rituals", ramp.get("ramp:ritual", 0)),
        ("Land ramp", ramp.get("ramp:land", 0)),
        ("Cost reducers", ramp.get("ramp:cost-reduction", 0)),
    ])
    assumptions = "".join(f"<li>{esc(a)}</li>" for a in mana.get("assumptions", []))
    notes = "".join(f"<li>{esc_x(n)}</li>" for n in mana.get("notes", []))
    notes_html = (f'<h4>What the audit flags</h4><ul class="swap-list">{notes}</ul>'
                  if notes else "")
    return (
        dept_open("sources-say", plan)
        + f'<div class="body-copy">{prose(prose_doc, "mana_base")}</div>'
        + meters + colour_table + class_table + facts + notes_html
        + f'<div class="assumptions"><b>What this audit does and does not do.</b> '
          f"Hypergeometric draws, not games. Every assumption is stated:"
          f"<ul>{assumptions}</ul></div>"
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + dept_close("sources-say", issue)
    )


def render_art_break(commander, mana):
    """The declared breather between the two dense analysis spreads (STYLEv3
    §6, v3.3): a full-bleed art spread with one computed Ledger line."""
    if not commander:
        return ""
    image = commander.get("art_crop") or commander.get("image")
    if not image:
        return ""
    lands = (mana or {}).get("lands", {})
    line = (f"{lands.get('total', '—')} lands. "
            f"{lands.get('enters_tapped', '—')} arrive tapped. "
            f"The pips know the difference.") if lands else ""
    return (
        f'<section class="art-break">'
        f'<img src="{esc(image)}" alt="{esc(commander.get("name", ""))}">'
        f'<blockquote class="pull-quote">{esc(line)}</blockquote>'
        f'{printing_credit(commander)}</section>'
    )


def render_upgrade_watch(issue, plan, prose_doc, cards_by_name, considering=None):
    dept = plan_dept(plan, "upgrade-watch")
    body = render_short_list(considering) if considering else ""
    return (
        dept_open("upgrade-watch", plan)
        + f'<div class="body-copy">{prose(prose_doc, "upgrades")}</div>'
        + body
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + dept_close("upgrade-watch", issue)
    )


def render_short_list(analysis):
    """The Ten, straight from considering.json: the only ten cards worth the
    reader's sleeves — evidence ◆, verdicts ★, exactly ten by contract
    (validate_considering).

    No ownership chip. The list used to mark each pick "In the box" or "Scouted",
    which asked the reader a question the section does not exist to answer: these
    are ten cards worth knowing about, and whether one is already in a box is the
    reader's business, not the magazine's."""
    parts = []
    if analysis.get("assessment"):
        parts.append(f'<div class="body-copy">'
                     f'{esc_x_paras(analysis["assessment"])}</div>')
    rows = []
    for i, entry in enumerate(analysis.get("ten") or [], 1):
        evidence = entry.get("evidence") or {}
        ev_bits = []
        for line in evidence.get("combo_lines_opened") or []:
            ev_bits.append(
                f'<span class="tier-data">◆</span> completes '
                f'{esc_x(" + ".join(line.get("cards", [])))} '
                f'({esc(line.get("status", ""))})')
        for obs in evidence.get("obsoletes") or []:
            ev_bits.append(f'<span class="tier-data">◆</span> obsoletes '
                           f'{esc_x(obs)}')
        partners = evidence.get("synergy_partners_in_deck") or []
        if partners:
            ev_bits.append(f'<span class="tier-data">◆</span> synergy: '
                           f'{esc_x(", ".join(partners))}')
        rank = evidence.get("edhrec_rank")
        if rank is not None:
            shown = f"{rank:,}" if isinstance(rank, int) else str(rank)
            ev_bits.append(f'<span class="tier-data">◆</span> EDHREC rank '
                           f'{esc(shown)}')
        ev_html = ("<br>" + "<br>".join(ev_bits)) if ev_bits else ""
        when = entry.get("when") or entry.get("unlocks") or ""
        when_html = f'<br><em>When:</em> {esc_x(when)}' if when else ""
        cut = entry.get("natural_cut")
        cut_html = f'<br><em>Natural cut:</em> {esc_x(cut)}' if cut else ""
        rows.append(
            f'<li><b>{esc(i)}.</b> <strong>{esc_x(entry.get("card", "?"))}</strong>'
            f' <span class="chip">{esc(entry.get("role", ""))}</span>'
            f'{ev_html}'
            f'<br><span class="tier-coach">★</span> {esc_x(entry.get("why", ""))}'
            f'{when_html}{cut_html}</li>'
        )
    parts.append(f'<ul class="swap-list">{"".join(rows)}</ul>')
    return "".join(parts)


def judges_desk_files(stacks, back_label="The Kill", contents_link=True):
    """The case files themselves, with no department furniture around them.

    Extracted so the compact page can render the same record without
    `dept_open`/`dept_close`, which need an `issue["volume"]` that only a
    magazine has. `back_label` names the section the ↩ link returns to,
    because the two renderers call that section different things and a link
    that says "The Kill" on a page with no Kill is a small lie in the one
    department that exists for correctness.
    """
    contents = (' · <a class="xref" href="#contents">↑ Contents</a>'
                if contents_link else "")
    files = []
    for stack in stacks:
        sid = stack["id"]
        checker = stack.get("checker", {})
        # `scenario.question` is authored FOR THE RESOLVER, not for a reader —
        # it carries instructions like "confirm each is a Dinosaur creature card
        # and that Cultivate/Mountain/Path go to the bottom in random order".
        # That is case-file material, so it lives here, in the collapsed record,
        # rather than in the middle of the read-through.
        asked = (stack.get("scenario") or {}).get("question", "")
        question = (f'<div class="scenario"><span class="lbl">The question put to '
                    f'the record</span><br>{esc(asked)}</div>') if asked else ""
        steps = []
        for step in stack.get("resolution", {}).get("steps", []):
            cites = "".join(
                f'<div class="cite"><b>CR {esc(c["rule"])}</b> — “{esc(c["quote"])}”</div>'
                for c in step.get("citations", [])
            )
            steps.append(
                f'<li><b>{esc(step.get("action", ""))}</b>'
                f'<div class="effect">{esc(step.get("effect", ""))}</div>{cites}</li>'
            )
        resolution = stack.get("resolution", {})
        n_steps = len(resolution.get("steps") or [])
        n_cites = sum(len(s.get("citations") or [])
                      for s in resolution.get("steps") or [])
        files.append(f"""
<details class="dossier" id="case-{esc(sid)}">
  <summary>
    <span class="case-row">
      <span class="case-id">A-{esc(sid)}</span>
      <span class="case-title">{esc(stack_headline(stack["title"])[0])}</span>
      <span class="case-meta">✓ cleared, {esc(checker.get("iterations", "?"))} cycle(s)
        · {n_steps} step{"" if n_steps == 1 else "s"}
        · {n_cites} citation{"" if n_cites == 1 else "s"}</span>
    </span>
  </summary>
  <div class="case-sub">Rules version {esc(stack.get("rules_version", "—"))}</div>
  {question}
  <ol>{"".join(steps)}</ol>
  {render_after_block(stack.get("resolution", {}).get("final_state", {}))}
  <p class="small"><a class="xref" href="#line-{esc(sid)}">↩ Back to this line in
    {esc(back_label)}</a>{contents}</p>
</details>""")
    return "".join(files)


def render_judges_desk(issue, plan, stacks, cards_by_name, withheld_stacks=()):
    """The proof — a scannable case index, each row opening the full record.

    "Judge's Desk shrinks to verdicts" and "it may not summarize, truncate, or
    paraphrase a single citation" (§5.1) are both binding, and they only look
    contradictory if you read "shrinks" as "holds less". What shrinks is the
    FOOTPRINT: a reader meets a one-line row per case — number, title, status,
    and how much record is behind it — instead of a stack of tall headers. Open
    one and the complete resolution is there, every citation verbatim, unchanged.

    The row deliberately carries **no holding**. Deriving one from
    `final_state.summary` was tried, measured against the corpus and removed — the
    note above `render_the_kill` records what it produced. A wrong verdict in the
    one department that exists for correctness is worse than no verdict, and the
    renderer may not summarise proof. The title is authored, so the title is what
    the index shows.
    """
    dept = plan_dept(plan, "judges-desk")
    files_html = judges_desk_files(stacks)
    return (
        dept_open("judges-desk", plan)
        + '<p class="dek">Every claim the magazine made, with the rule text that backs '
          "it. Nothing here is paraphrased. Tap a case to open its full record.</p>"
        + (files_html or TODO)
        + withheld_cases(withheld_stacks)
        + dept_furniture(dept, cards_by_name)
        + dept_close("judges-desk", issue)
    )


def withheld_cases(stacks):
    """Cases the issue names and does not print, with the reason it does not.

    A published resolution says "exactly as in stack 001" nineteen times across
    two issues, and stack 001 is not in either of them — so the reader goes
    looking for a case that was deliberately withheld and finds a gap. A dead
    pointer in the department whose job is being findable.

    Each is a checker-PASSED artifact held back because a card left the 99: the
    rules finding stands, the board is one this deck can no longer make, and
    `presentable_note` says which card. So the honest row is the reason, not the
    record — the resolution stays unpublished, because it is about a board this
    deck cannot assemble, and the reader stops hunting.

    Nothing in the referring prose is touched. Those artifacts passed the
    checker; their step text is evidence, and rewriting it to remove a pointer
    would put a ✓ over words no checker read.
    """
    if not stacks:
        return ""
    rows = []
    for stack in stacks:
        sid = stack["id"]
        note = stack.get("presentable_note") or (
            "withheld from this issue; the resolution stands as a rules finding")
        rows.append(f"""
<div class="kill-row" id="case-{esc(sid)}">
  <div class="kill-row-head">
    <span class="case-id">A-{esc(sid)}</span>
    <span class="case-title">{esc(stack_headline(stack["title"])[0])}</span>
    <span class="case-meta">withheld</span>
  </div>
  <p class="kill-row-result">{esc(note)}</p>
</div>""")
    n = len(stacks)
    return (f'<div class="kill-index">'
            f'<h4 class="kill-index-head">Named in the issue, not printed here</h4>'
            f'<p class="kill-index-dek">{n} case{"" if n == 1 else "s"} cleared the same '
            f'review and {"is" if n == 1 else "are"} held back for the reason given: the '
            f'board is one this deck can no longer make. The finding stands; the line is '
            f'not one you can assemble, so it is not presented as one.</p>'
            f'{"".join(rows)}</div>')


def render_back_page(issue, plan, deck_doc, stacks, cards_by_name):
    dept = plan_dept(plan, "back-page")
    sha = str(deck_doc.get("decklist_sha256", ""))[:12]
    rules_version = stacks[0].get("rules_version", "—") if stacks else "—"
    return f"""
<section class="dept" id="back-page" style="--accent:{ACCENT["back-page"]}">
  <div class="dept-head"><div><h2 class="dept-title">The Back Page</h2></div>
    <div class="dept-promise">{esc(DEPARTMENT_BY_ID["back-page"]["promise"])}</div></div>
  <div class="kicker">Next issue</div>
  <h1 class="feature">{esc(issue["next_issue"])}</h1>
  <p class="dek">Another commander, another 99, the same contract: verified lines,
    seeded numbers, and coaching that says when it's coaching.</p>
  <p class="dek" style="margin-top:18px">Want the numbers without the prose?
    <a href="../viz/deck.html?deck={esc(deck_doc.get("deck", ""))}">Open this deck&rsquo;s
    dossier</a> — the same committed artifacts, rendered as data — or
    <a href="../viz/index.html">explore the card map</a>.</p>
  {fast_facts("Colophon", [
      ("Volume", f'{issue["volume"]:03d}'),
      ("Issue", issue["issue_date"]),
      ("Deck", issue["deck_name"]),
      ("Verified lines", len(stacks)),
      ("Rules version", rules_version),
      ("Decklist sha", sha or "—"),
  ])}
  <p style="font-size:.82em;color:var(--ink-soft);margin-top:18px">
    Card images and card text are property of Wizards of the Coast. Pilot's Manual is
    unofficial fan content permitted under the Wizards of the Coast Fan Content Policy,
    not approved or endorsed by Wizards. Portions of the materials used are property of
    Wizards of the Coast LLC.</p>
  {dept_furniture(dept, cards_by_name)}
</section>""" + folio("The Back Page", issue["volume"])


# ── Assembly ────────────────────────────────────────────────────────────


def render_issue(issue, plan, deck_doc, stacks, prose_doc, synergy,
                 goldfish=None, decisions=None,
                 considering=None, tutor_guide=None, mana=None,
                 short_list_art=None, withheld_stacks=()):
    """Assemble a complete issue. Deterministic for fixed inputs."""
    cards = deck_doc["cards"]
    cards_by_name = {c["name"]: c for c in cards}
    commanders = [c for c in cards if c["is_commander"]]
    commander = commanders[0] if commanders else None
    decisions = decisions or []
    volume = issue["volume"]

    # Masthead first, series second — the order printed on the cover. The tab and
    # the og:title were leading with the series slug while the page itself is
    # branded MANA MAP, so a shared link named a magazine the reader would not find
    # on the page. Both are built from `issue_spec`, never re-typed here.
    title = (f"{MASTHEAD} — {issue['deck_name']} · "
             f"{SERIES_TITLE} Vol. {volume:03d}")
    # art_crop for the same reason the cover uses it: a social preview is
    # magazine photography, not a card scan.
    og_image = (commander.get("art_crop") or commander.get("image")) if commander else ""
    description = (
        f"{issue['deck_name']}: {len(stacks)} rules-verified lines, seeded goldfish "
        f"numbers, and table coaching. Pilot's Manual Vol. {volume:03d}."
    )

    # Arm the card linker: every card mention in reader-facing copy becomes a
    # link to its tile in The 99 (commander → Command Zone) with a hover
    # preview. Cleared after assembly so renders never leak state into tests.
    # The Short List's ten are the one set of card names the linker cannot reach
    # from `cards.json`, because they are deliberately not in the deck.
    set_card_links(cards, commander["name"] if commander else None,
                   offdeck=(short_list_art or {}).get("cards"))

    # One renderer per section; issue_spec.DEPARTMENT_IDS is the only place
    # the STYLEv3 §5 five-act order lives — reordering the spec reorders the book.
    renderers = {
        "cover": lambda: render_cover(issue, plan, commander),
        "contents": lambda: render_contents(issue, plan, stacks, decisions),
        "first-turns": lambda: render_first_turns(issue, plan, prose_doc,
                                                  goldfish, cards_by_name),
        "keep-or-ship": lambda: render_keep_or_ship(issue, plan, prose_doc,
                                                    goldfish, cards_by_name),
        "whats-your-play": lambda: render_whats_your_play(issue, plan, decisions,
                                                          cards_by_name),
        "at-the-table": lambda: render_at_the_table(
            issue, plan, prose_doc, tutor_guide, cards_by_name),
        "sources-say": lambda: render_sources_say(issue, plan, mana, prose_doc,
                                                  cards_by_name),
        "command-zone": lambda: render_command_zone(issue, plan, commander,
                                                    goldfish, cards_by_name),
        "the-99": lambda: render_the_99(issue, plan, cards, prose_doc, synergy,
                                        cards_by_name),
        "upgrade-watch": lambda: render_upgrade_watch(issue, plan, prose_doc,
                                                      cards_by_name, considering),
        "by-the-numbers": lambda: render_by_the_numbers(issue, plan, goldfish,
                                                        cards_by_name),
        "the-kill": lambda: render_the_kill(issue, plan, stacks, cards,
                                            prose_doc, cards_by_name),
        "editors-letter": lambda: render_editors_letter(issue, plan, commander,
                                                        prose_doc, cards_by_name),
        "pilots-log": lambda: render_pilots_log(issue, plan, prose_doc,
                                                cards_by_name),
        "judges-desk": lambda: render_judges_desk(issue, plan, stacks,
                                                  cards_by_name, withheld_stacks),
        "featured-artist": lambda: render_featured_artist(issue, plan, cards,
                                                          cards_by_name),
        "back-page": lambda: render_back_page(issue, plan, deck_doc, stacks,
                                              cards_by_name),
    }
    try:
        sections = []
        planned = {d.get("id") for d in (plan or {}).get("departments", [])}
        for dept_id in DEPARTMENT_IDS:
            # An OPTIONAL department absent from this plan renders nothing at all.
            # Without this every issue that has not opted in yet prints an empty
            # department with a [TODO] in it — which is what "optional" would mean
            # if the renderer disagreed with the validator about it.
            if dept_id in OPTIONAL_DEPARTMENTS and dept_id not in planned:
                continue
            sections.append(renderers[dept_id]())
            if dept_id in BREATHER_AFTER:
                sections.append(render_art_break(commander, mana))
        # Before the cover, deliberately: a reader must learn the deck no longer
        # exists BEFORE they start reading its figures as current. Empty string
        # for a live issue, so nothing byte-shifts on the other eight.
        body = issue_status_banner(issue_status(issue)) + "".join(sections) + (
            '<a class="toc-float" href="#contents" '
            'title="Back to The Flight Plan">☰</a>')
    finally:
        clear_card_links()

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(description)}">
<meta property="og:type" content="article">
{f'<meta property="og:image" content="{esc(og_image)}">' if og_image else ""}
{FONT_LINK}
{stylesheet_link()}
</head><body><div class="trim">
{body}
</div></body></html>
"""


def main(args):
    slug = args.slug
    base = deck_dir(slug)
    issue = load_json(base / "issue.json")
    if issue is None:
        raise SystemExit(
            f"{base / 'issue.json'} not found — author the issue identity block "
            f"(volume, issue_date, cover_price, deck_name, commander, cover_tagline, "
            f"next_issue). See STYLEv3 §4.1."
        )
    plan = load_json(base / "issue_plan.json", {})
    if not plan:
        print("WARN issue_plan.json absent — rendering with department defaults "
              "(the magazine-editor is retired; a plan is a legacy input)")

    deck_doc = load_deck_cards(slug)
    stacks = load_verified_stacks(slug)
    withheld_stacks = load_withheld_stacks(slug)
    decisions = load_decisions(slug)
    prose_doc = load_json(base / "manual_prose.json", {})
    goldfish = load_json(base / "goldfish_metrics.json")
    synergy = load_synergy_graph() if SYNERGY_GRAPH_PATH.exists() else {}
    considering = load_json(base / "considering.json")
    short_list_art = load_json(base / SHORT_LIST_ART, {})
    tutor_guide = load_json(base / "tutor_guide.json")
    mana = load_json(base / "mana_analysis.json")
    _DECK_MAP["doc"] = load_json(base / "deck_map.json")
    _DECK_MAP["engine"] = load_json(base / "engine.json")

    try:
        html_out = render_issue(issue, plan, deck_doc, stacks, prose_doc, synergy,
                                goldfish, decisions, considering, tutor_guide,
                                mana, short_list_art, withheld_stacks)
    finally:
        _DECK_MAP["doc"] = _DECK_MAP["engine"] = None   # cleared like the card links
    MANUALS_DIR.mkdir(parents=True, exist_ok=True)
    sheet, wrote_sheet = write_stylesheet(MANUALS_DIR)
    if wrote_sheet:
        print(f"Wrote {sheet}")
    out = MANUALS_DIR / f"{slug}.html"
    out.write_text(html_out, encoding="utf-8")
    print(
        f"Wrote {out}: Vol. {issue['volume']:03d} · {len(stacks)} verified line(s), "
        f"{len(decisions)} decision spread(s), goldfish: {'yes' if goldfish else 'no'}, "
        f"plan: {'yes' if plan else 'defaults'}"
    )


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot build-manual <slug>`.")
