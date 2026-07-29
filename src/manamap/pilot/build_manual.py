"""Pilot: render a deck's issue of Pilot's Manual as standalone magazine HTML.

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
from manamap.pilot.artist_credits import is_accessory
from manamap.pilot.common import (
    checker_passed,
    deck_dir,
    load_deck_cards,
    load_json,
    load_synergy_graph,
    mainboard,
    sideboard,
)
from manamap.pilot.design import (
    card_tile,
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
    map_key,
    pilot_tip,
    power_meter,
    printing_credit,
    pull_quote,
    threat_box,
    violator,
)
from manamap.pilot.issue_spec import (
    ACTS,
    BREATHER_AFTER,
    DEPARTMENT_BY_ID,
    DEPARTMENT_IDS,
    MASTHEAD,
    MASTHEAD_COLUMNISTS,
    SERIES_SLUG,
    STANDING_TAGLINE,
)

# One accent per department — held across all its pages and its folio tab.
ACCENT = {
    "cover": "var(--power-red)", "contents": "var(--ink)",
    "first-turns": "var(--power-red)", "command-zone": "var(--y2k-violet)",
    "by-the-numbers": "var(--y2k-blue)", "the-kill": "var(--power-red)",
    "politics-table": "var(--radical-purple)", "whats-your-play": "var(--hot-magenta)",
    "know-your-enemy": "var(--radical-purple)", "the-99": "var(--slime-green)",
    "fetch-quests": "var(--tier-coach)", "sources-say": "var(--y2k-blue)",
    "featured-artist": "var(--hot-magenta)",
    "keep-or-ship": "var(--tier-coach)", "upgrade-watch": "var(--y2k-blue)",
    "judges-desk": "var(--stamp-red)", "back-page": "var(--ink)",
}

MAP_KEY_ENTRIES = [("⚡", "mana floated"), ("🜲", "storm count"),
                   ("⛃", "treasure"), ("♥", "life")]

TODO = '<p><span class="todo">TODO</span> This section is awaiting content.</p>'


# ── Loading ─────────────────────────────────────────────────────────────


def load_verified_stacks(slug):
    """Checker-passed stacks only, in id order — the publication gate."""
    stacks = []
    for path in sorted((deck_dir(slug) / "stacks").glob("*.json")):
        with open(path) as f:
            doc = json.load(f)
        if checker_passed(doc):
            stacks.append(doc)
    return stacks


def load_decisions(slug):
    decisions = []
    directory = deck_dir(slug) / "decisions"
    if not directory.is_dir():
        return decisions
    for path in sorted(directory.glob("*.json")):
        with open(path) as f:
            decisions.append(json.load(f))
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


def set_card_links(cards, commander_name=None):
    """Arm the card linker for one render. Probes are esc()-escaped so they
    match post-escape text; longest-first so full names beat short forms."""
    anchors = card_anchor_ids(cards)
    by_name = {c["name"]: c for c in cards}
    meta = {}
    for probe, name in _card_probes(cards).items():
        href = ("#command-zone" if name == commander_name
                else f"#{anchors[name]}")
        meta[esc(probe)] = (href, by_name[name].get("image") or "")
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
        href, image = _CARD_LINKS["meta"][m.group(0)]
        pop = (f'<img class="card-pop" src="{esc(image)}" loading="lazy" alt="">'
               if image else "")
        return f'<a class="cardref" href="{href}">{m.group(0)}{pop}</a>'

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
    for act_title, dept_ids in ACTS:
        rows = []
        for dept_id in dept_ids:
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
    masthead_rows = "".join(
        f'<div class="legend-row">{badge(c["tier"])}<div><b>{esc(c["name"])}</b> — '
        f'{esc(c["bio"])}</div></div>'
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


def render_first_turns(issue, plan, prose_doc, cards_by_name):
    dept = plan_dept(plan, "first-turns")
    return (
        dept_open("first-turns", plan)
        + f'<div class="body-copy">{prose(prose_doc, "how_it_wins")}</div>'
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


def render_the_kill(issue, plan, stacks, prose_doc, cards_by_name):
    dept = plan_dept(plan, "the-kill")
    spreads = []
    for stack in stacks:
        sid = stack["id"]
        checker = stack.get("checker", {})
        intro = prose(prose_doc, "combo_lines", sid)
        final = stack.get("resolution", {}).get("final_state", {})
        spreads.append(f"""
<article class="rule-top" id="line-{esc(sid)}">
  <div class="kicker">Verified line {esc(sid)}</div>
  <h3 style="font-family:var(--display);font-size:1.5em">{esc(stack["title"])}</h3>
  <div class="body-copy">{intro}</div>
  <div class="scenario"><span class="lbl">The question</span><br>
    {esc(stack["scenario"].get("question", ""))}</div>
  <p><b>Result.</b> {esc(final.get("summary", ""))}</p>
  <a class="dossier-pointer" href="#case-{esc(sid)}">
    Full dossier: Judge's Desk, Case A-{esc(sid)} →</a>
  <span style="margin-left:10px">{badge("verified")} cleared in
    {esc(checker.get("iterations", "?"))} review cycle(s)</span>
</article>""")
    return (
        dept_open("the-kill", plan)
        + map_key(MAP_KEY_ENTRIES)
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + ("".join(spreads) or TODO)
        + dept_close("the-kill", issue)
    )


def render_politics(issue, plan, prose_doc, cards_by_name):
    dept = plan_dept(plan, "politics-table")
    return (
        dept_open("politics-table", plan)
        + f'<div class="body-copy">{prose(prose_doc, "threat_assessment")}</div>'
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + dept_close("politics-table", issue)
    )


def render_whats_your_play(issue, plan, decisions, cards_by_name):
    dept = plan_dept(plan, "whats-your-play")
    spreads = []
    for decision in decisions:
        scenario = decision.get("scenario", {})
        board = scenario.get("board", {})
        bits = []
        for label, key in (("You", "you"), ("Table", "opponents"), ("Hand", "hand")):
            value = board.get(key) or scenario.get(key)
            if value:
                text = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
                bits.append(f'<span class="lbl">{label}</span> {esc(text)}')
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
  <div class="scenario">{"<br>".join(bits)}<br><br>
    <b>{esc(scenario.get("question", ""))}</b></div>
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


def render_know_your_enemy(issue, plan, prose_doc, cards_by_name):
    dept = plan_dept(plan, "know-your-enemy")
    boxes = []
    for entry in dept.get("threats", []):
        boxes.append(threat_box(
            entry.get("archetype", ""), entry.get("meter_label", "Threat"),
            float(entry.get("rate", 0.5)),
            f'<p>{esc_x(entry.get("read", ""))}</p>'
            f'<p><b>Your outs:</b> {esc_x(", ".join(entry.get("outs", [])))}</p>',
        ))
    return (
        dept_open("know-your-enemy", plan)
        + "".join(boxes)
        + f'<div class="body-copy">{prose(prose_doc, "matchups")}</div>'
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + dept_close("know-your-enemy", issue)
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
    main = mainboard(cards)
    by_name = {c["name"]: c for c in main}

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
            card_tile(c, roles, synergy, anchor_id=CARD_ANCHORS.get(c["name"]))
            for c in group_cards)
        sections.append(f'{heading}<div class="card-grid">{tiles}</div>')
    tiles = "".join(sections)

    side = sideboard(cards)
    side_html = ""
    if side:
        side_tiles = []
        for card in side:
            # is_accessory imported rather than re-implemented — this predicate used
            # to exist twice, here and in artist_credits, with nothing keeping them
            # in step.
            if is_accessory(card):
                blurb = roles.get(card["name"]) or (
                    "Table aid — no rules text. Use it to track game state on big turns.")
                image = (f'<img src="{esc(card["image"])}" alt="{esc(card["name"])}" loading="lazy">'
                         if card.get("image") else "")
                aid = (None if card["name"] in by_name
                       else CARD_ANCHORS.get(card["name"]))
                anchor = f' id="{esc(aid)}"' if aid else ""
                side_tiles.append(
                    f'<div class="card-tile"{anchor}>{image}<h4>{esc(card["name"])}</h4>'
                    f'<span class="chip">Table aid</span>'
                    f"<p>{esc(blurb)}</p></div>"
                )
                continue
            # A real sideboard card is a card: same tile, same synergy chips. It used
            # to bypass _card_tile entirely and render with an empty <p> when nobody
            # had written it a role blurb — silently, with no TODO.
            side_roles = dict(roles)
            side_roles.setdefault(
                card["name"],
                f"{card.get('type_line', 'Sideboard card')} — no role blurb written yet.")
            side_anchor = (None if card["name"] in by_name
                           else CARD_ANCHORS.get(card["name"]))
            side_tiles.append(card_tile(card, side_roles, synergy,
                                         anchor_id=side_anchor))
        side_html = ("<h3>Sideboard &amp; table aids</h3>"
                     f'<div class="card-grid">{"".join(side_tiles)}</div>')
    return (
        dept_open("the-99", plan)
        + roles_todo
        + tiles + side_html
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


def render_fetch_quests(issue, plan, tutor_guide, cards_by_name):
    """What to tutor (★): scenario -> fetch -> why, straight from
    tutor_guide.json. A deck with no tutors keeps the section (L8) with
    standing copy instead of a TODO — zero tutors is an answer, not a gap."""
    dept = plan_dept(plan, "fetch-quests")
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
    return (
        dept_open("fetch-quests", plan)
        + "".join(parts)
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + dept_close("fetch-quests", issue)
    )


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


def render_upgrade_watch(issue, plan, prose_doc, cards_by_name, sideboard=None,
                         lookout=None, considering=None):
    dept = plan_dept(plan, "upgrade-watch")
    if considering:
        body = render_short_list(considering)
    else:
        # Transitional: decks not yet regenerated under the Short List keep
        # rendering their legacy artifact until the batch pass retires it.
        body = (render_sideboard(sideboard) if sideboard
                else render_lookout(lookout))
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
    reader's sleeves — bench picks and pool scouts on one list, evidence ◆,
    verdicts ★, exactly ten by contract (validate_considering)."""
    parts = []
    if analysis.get("assessment"):
        parts.append(f'<div class="body-copy">'
                     f'{esc_x_paras(analysis["assessment"])}</div>')
    rows = []
    for i, entry in enumerate(analysis.get("ten") or [], 1):
        source = entry.get("source", "pool")
        chip = "In the box" if source == "sideboard" else "Scouted"
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
            f' <span class="chip">{chip}</span>'
            f' <span class="chip">{esc(entry.get("role", ""))}</span>'
            f'{ev_html}'
            f'<br><span class="tier-coach">★</span> {esc_x(entry.get("why", ""))}'
            f'{when_html}{cut_html}</li>'
        )
    parts.append(f'<ul class="swap-list">{"".join(rows)}</ul>')
    verdicts = analysis.get("bench_verdicts") or []
    if verdicts:
        items = "".join(
            f'<li><strong>{esc_x(v.get("card", "?"))}</strong> — '
            f'<span class="chip">{esc(v.get("verdict", ""))}</span> '
            f'<span class="tier-coach">★</span> {esc_x(v.get("why", ""))}</li>'
            for v in verdicts
        )
        parts.append(f'<h4>The rest of the bench</h4>'
                     f'<ul class="swap-list">{items}</ul>')
    return "".join(parts)


def render_sideboard(analysis):
    """The sideboard read, straight from sideboard_analysis.json.

    Rendered from the artifact rather than a prose key on purpose: a new
    manual_prose key would change prose:shape and invalidate both prose routines
    and every issue plan, for a section the renderer can build itself.

    Tiers are labelled inline because this section genuinely mixes them. The
    bracket delta is ◆ — recomputed by validate-sideboard, not taken on trust.
    The recommendation to make the swap is ★, and a line the sideboard opens is a
    candidate until a stack passes. The department badge stays ◆; nothing here
    wears costume it did not earn.
    """
    if not analysis:
        return ""  # a deck with no sideboard simply has no section

    parts = ['<h3>From the sideboard</h3>']
    assessment = analysis.get("assessment")
    if assessment:
        parts.append(f'<div class="body-copy"><p>{esc_x(assessment)}</p></div>')

    swaps = analysis.get("swaps") or []
    if swaps:
        rows = []
        for swap in swaps:
            delta = swap.get("bracket_delta") or {}
            if delta.get("before") is not None and delta.get("after") is not None:
                moved = delta["before"] != delta["after"]
                bracket = (f'<span class="tier-data">◆</span> bracket '
                           f'{delta["before"]}&nbsp;&rarr;&nbsp;{delta["after"]}' if moved
                           else f'<span class="tier-data">◆</span> bracket '
                                f'{delta["after"]}, unchanged')
            else:
                bracket = ""
            rows.append(
                '<li>'
                f'<strong>{esc(swap.get("in", "?"))}</strong> in, '
                f'<strong>{esc(swap.get("out", "?"))}</strong> out'
                f' <span class="chip">{esc(swap.get("role", ""))}</span>'
                f'<br><em>When:</em> {esc_x(swap.get("when", ""))}'
                f'<br><span class="tier-coach">★</span> {esc_x(swap.get("why", ""))}'
                + (f'<br>{bracket}' if bracket else "")
                + '</li>'
            )
        parts.append(f'<ul class="swap-list">{"".join(rows)}</ul>')
    else:
        parts.append('<div class="body-copy"><p>No swap in this sideboard earns a '
                     'maindeck slot as the deck currently stands.</p></div>')

    lines = analysis.get("opens_lines") or []
    if lines:
        items = "".join(
            f'<li><strong>{esc(" + ".join(line.get("cards", [])))}</strong> — '
            f'{esc_x(line.get("why_plausible", ""))} '
            f'<span class="chip">{esc(line.get("status", ""))}</span></li>'
            for line in lines
        )
        parts.append(
            '<h4>Lines this sideboard would open</h4>'
            f'<ul class="swap-list">{items}</ul>'
            '<div class="body-copy"><p>Unverified, every one: a line is fact only once a '
            'resolution has passed the checker. These are the next Judge&rsquo;s Desk '
            'candidates, not results.</p></div>'
        )

    defaults = analysis.get("long_term_defaults") or []
    if defaults:
        items = "".join(
            f'<li><strong>{esc(d.get("card", "?"))}</strong> — '
            f'<span class="chip">{esc(d.get("verdict", ""))}</span> '
            f'<span class="tier-coach">★</span> {esc_x(d.get("why", ""))}</li>'
            for d in defaults
        )
        parts.append(f'<h4>Belongs in the 99?</h4><ul class="swap-list">{items}</ul>')

    return "".join(parts)


def render_lookout(analysis):
    """The pool-scout read, straight from upgrade_watch.json.

    The empty-sideboard counterpart of render_sideboard, and the same tier
    discipline: the evidence lines (obsolescence, synergy partners, combo
    lines) are ◆ — re-checked by validate-upgrade-watch against the tracked
    indexes, never taken on trust — while the pick itself is ★. A combo line a
    candidate would open stays a candidate until a stack passes. Department
    badge stays ◆.
    """
    if not analysis:
        return ""  # no sideboard and no scout report: no section, same as ever

    parts = ['<h3>On the Lookout</h3>']
    assessment = analysis.get("assessment")
    if assessment:
        parts.append(f'<div class="body-copy"><p>{esc_x(assessment)}</p></div>')

    rows = []
    for entry in analysis.get("lookout") or []:
        ev = entry.get("evidence") or entry
        bits = []
        for deck_card in ev.get("obsoletes") or []:
            bits.append(f'<span class="tier-data">◆</span> straight upgrade over '
                        f'<strong>{esc(deck_card)}</strong>')
        partners = ev.get("synergy_partners_in_deck") or []
        if partners:
            names = ", ".join(esc(p.get("partner") if isinstance(p, dict) else p)
                              for p in partners)
            bits.append(f'<span class="tier-data">◆</span> synergy partners already '
                        f'in the 99: {names}')
        for line in ev.get("combo_lines_opened") or []:
            bits.append(
                f'<span class="tier-data">◆</span> completes '
                f'<strong>{esc(" + ".join(line.get("cards", [])))}</strong> '
                f'<span class="chip">{esc(line.get("status", ""))}</span>'
            )
        role = entry.get("role", "")
        rows.append(
            '<li>'
            f'<strong>{esc(entry.get("card", "?"))}</strong>'
            + (f' <span class="chip">{esc(role)}</span>' if role else "")
            + "".join(f'<br>{bit}' for bit in bits)
            + f'<br><span class="tier-coach">★</span> {esc(entry.get("why", ""))}'
            + (f'<br><em>Natural cut:</em> {esc(entry["natural_cut"])}'
               if entry.get("natural_cut") else "")
            + '</li>'
        )
    if rows:
        parts.append(f'<ul class="swap-list">{"".join(rows)}</ul>')
        parts.append(
            '<div class="body-copy"><p>Ten cards the deck does not own yet. Every '
            'combo line above is a candidate until a resolution passes the checker; '
            'every straight-upgrade and synergy claim traces to a tracked index.</p></div>'
        )
    return "".join(parts)


def render_judges_desk(issue, plan, stacks, cards_by_name):
    """The proof. Every citation reproduced verbatim — never summarized."""
    dept = plan_dept(plan, "judges-desk")
    files = []
    for stack in stacks:
        sid = stack["id"]
        checker = stack.get("checker", {})
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
        files.append(f"""
<details class="dossier" id="case-{esc(sid)}">
  <summary>
    <span class="file-tab">Case A-{esc(sid)}</span>
    <span class="dossier-head">
      <span><b>{esc(stack["title"])}</b><br>
        <span style="font-size:.85em;color:var(--ink-soft)">
          Rules version {esc(stack.get("rules_version", "—"))} ·
          status: cleared in {esc(checker.get("iterations", "?"))} review cycle(s) ·
          tap to open</span></span>
      <span class="stamp">Verified</span>
    </span>
  </summary>
  <ol>{"".join(steps)}</ol>
  <p class="small"><a class="xref" href="#line-{esc(sid)}">↩ Back to this line in
    The Kill</a> · <a class="xref" href="#contents">↑ Contents</a></p>
</details>""")
    return (
        dept_open("judges-desk", plan)
        + '<p class="dek">Every claim the magazine made, with the rule text that backs '
          "it. Nothing here is paraphrased. Tap a case to open its full record.</p>"
        + ("".join(files) or TODO)
        + dept_furniture(dept, cards_by_name)
        + dept_close("judges-desk", issue)
    )


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
                 goldfish=None, decisions=None, sideboard=None, lookout=None,
                 considering=None, tutor_guide=None, mana=None):
    """Assemble a complete issue. Deterministic for fixed inputs."""
    cards = deck_doc["cards"]
    cards_by_name = {c["name"]: c for c in cards}
    commanders = [c for c in cards if c["is_commander"]]
    commander = commanders[0] if commanders else None
    decisions = decisions or []
    volume = issue["volume"]

    title = f"{issue['deck_name']} — Pilot's Manual Vol. {volume:03d}"
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
    set_card_links(cards, commander["name"] if commander else None)

    # One renderer per section; issue_spec.DEPARTMENT_IDS is the only place
    # the STYLEv3 §5 five-act order lives — reordering the spec reorders the book.
    renderers = {
        "cover": lambda: render_cover(issue, plan, commander),
        "contents": lambda: render_contents(issue, plan, stacks, decisions),
        "first-turns": lambda: render_first_turns(issue, plan, prose_doc,
                                                  cards_by_name),
        "keep-or-ship": lambda: render_keep_or_ship(issue, plan, prose_doc,
                                                    goldfish, cards_by_name),
        "whats-your-play": lambda: render_whats_your_play(issue, plan, decisions,
                                                          cards_by_name),
        "politics-table": lambda: render_politics(issue, plan, prose_doc,
                                                  cards_by_name),
        "know-your-enemy": lambda: render_know_your_enemy(issue, plan, prose_doc,
                                                          cards_by_name),
        "fetch-quests": lambda: render_fetch_quests(issue, plan, tutor_guide,
                                                    cards_by_name),
        "sources-say": lambda: render_sources_say(issue, plan, mana, prose_doc,
                                                  cards_by_name),
        "command-zone": lambda: render_command_zone(issue, plan, commander,
                                                    goldfish, cards_by_name),
        "the-99": lambda: render_the_99(issue, plan, cards, prose_doc, synergy,
                                        cards_by_name),
        "upgrade-watch": lambda: render_upgrade_watch(issue, plan, prose_doc,
                                                      cards_by_name, sideboard,
                                                      lookout, considering),
        "by-the-numbers": lambda: render_by_the_numbers(issue, plan, goldfish,
                                                        cards_by_name),
        "the-kill": lambda: render_the_kill(issue, plan, stacks, prose_doc,
                                            cards_by_name),
        "judges-desk": lambda: render_judges_desk(issue, plan, stacks,
                                                  cards_by_name),
        "featured-artist": lambda: render_featured_artist(issue, plan, cards,
                                                          cards_by_name),
        "back-page": lambda: render_back_page(issue, plan, deck_doc, stacks,
                                              cards_by_name),
    }
    try:
        sections = []
        for dept_id in DEPARTMENT_IDS:
            sections.append(renderers[dept_id]())
            if dept_id in BREATHER_AFTER:
                sections.append(render_art_break(commander, mana))
        body = "".join(sections) + (
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
              "(run the design-issue skill for the full magazine treatment)")

    deck_doc = load_deck_cards(slug)
    stacks = load_verified_stacks(slug)
    decisions = load_decisions(slug)
    prose_doc = load_json(base / "manual_prose.json", {})
    goldfish = load_json(base / "goldfish_metrics.json")
    synergy = load_synergy_graph() if SYNERGY_GRAPH_PATH.exists() else {}
    # The Short List (considering.json) is the current contract; the legacy
    # sideboard/lookout pair renders only for decks not yet regenerated under
    # it (transitional — retired per-deck at batch time).
    considering = load_json(base / "considering.json")
    sideboard = load_json(base / "sideboard_analysis.json")
    lookout = load_json(base / "upgrade_watch.json")
    tutor_guide = load_json(base / "tutor_guide.json")
    mana = load_json(base / "mana_analysis.json")

    html_out = render_issue(issue, plan, deck_doc, stacks, prose_doc, synergy,
                            goldfish, decisions, sideboard, lookout,
                            considering, tutor_guide, mana)
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
