"""Pilot: render a deck's issue of Pilot's Manual as standalone magazine HTML.

Fully deterministic — no LLM calls, no dates, no randomness. The editorial layer
arrives as data (`issue.json` identity + `issue_plan.json` packaging from the
magazine-editor agent) and the body prose as `manual_prose.json`; this module
assembles them into the fourteen fixed departments of STYLEv3 §5.

Contract invariants:
- Only checker-passed stacks render, and Judge's Desk reproduces every citation
  verbatim — the renderer may not summarize proof (STYLEv3 §5.1, docs/pilot.md).
- A department with thin artifacts renders a visible [TODO], never vanishes.
- Tier badges come from the department system, not the plan (STYLEv3 §10).
"""

import json

from manamap.config import MANUALS_DIR, SYNERGY_GRAPH_PATH
from manamap.pilot.common import deck_dir, load_deck_cards
from manamap.pilot.design import (
    CSS,
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
    pull_quote,
    threat_box,
    violator,
)
from manamap.pilot.issue_spec import (
    DEPARTMENT_BY_ID,
    DEPARTMENT_IDS,
    MASTHEAD,
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
    "keep-or-ship": "var(--tier-coach)", "upgrade-watch": "var(--y2k-blue)",
    "judges-desk": "var(--stamp-red)", "back-page": "var(--ink)",
}

MAP_KEY_ENTRIES = [("⚡", "mana floated"), ("🜲", "storm count"),
                   ("⛃", "treasure"), ("♥", "life")]

TODO = '<p><span class="todo">TODO</span> This department is awaiting content.</p>'


# ── Loading ─────────────────────────────────────────────────────────────


def load_json(path, default=None):
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def load_verified_stacks(slug):
    """Checker-passed stacks only, in id order — the publication gate."""
    stacks = []
    for path in sorted((deck_dir(slug) / "stacks").glob("*.json")):
        with open(path) as f:
            doc = json.load(f)
        if (doc.get("checker") or {}).get("verdict") == "pass":
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


def prose(prose_doc, key, sub=None):
    """Body copy from manual_prose.json; visible TODO when absent."""
    value = (prose_doc or {}).get(key)
    if sub is not None:
        value = (value or {}).get(sub)
    if not value:
        return TODO
    paragraphs = [p.strip() for p in str(value).split("\n\n") if p.strip()]
    return "".join(f"<p>{esc(p)}</p>" for p in paragraphs)


def caption_html(text):
    """Caption grammar: **bold lead-in**, then roman body."""
    if "**" in text:
        head, _, tail = text.partition("**")
        lead, _, rest = tail.partition("**")
        return f"{esc(head)}<b>{esc(lead)}</b>{esc(rest)}"
    return esc(text)


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
    return (
        f'<section class="dept" id="{dept_id}" style="--accent:{ACCENT[dept_id]}">'
        f'<div class="dept-head"><div>'
        f'<h2 class="dept-title">{esc(spec["title"])}</h2></div>'
        f"<div>{badges}</div>"
        f'<div class="dept-promise">{esc(spec["promise"])}</div></div>'
        f"{kicker}<h1 class=\"feature\">{esc(headline)}</h1>{dek}"
    )


def dept_close(dept_id, volume):
    return "</section>" + folio(DEPARTMENT_BY_ID[dept_id]["title"], volume)


def dept_furniture(dept, cards_by_name):
    """Render the plan's furniture for a department: tips, callouts, pull quote."""
    out = []
    for step in dept.get("callouts", []):
        out.append(callout(step.get("n", "•"), step.get("title", ""), step.get("text", "")))
    for tip in dept.get("pilot_tips", []):
        card = cards_by_name.get(tip.get("card", ""))
        out.append(pilot_tip(tip.get("card", ""), tip.get("text", ""),
                             (card or {}).get("image")))
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
                                   card.get("scryfall_uri")))
    return "".join(figures)


# ── Departments ─────────────────────────────────────────────────────────


def render_cover(issue, plan, commander, stacks):
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
        image = commander.get("art_crop") or commander.get("image")
        artist = commander.get("artist")
        credit = f'<div class="art-credit">Art: {esc(artist)}</div>' if artist else ""
        art = (
            f'<div class="hero-art"><img src="{esc(image)}" '
            f'alt="{esc(commander["name"])}">{credit}</div>'
        )
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
    rows = []
    for dept_id in DEPARTMENT_IDS:
        if dept_id in ("cover", "contents"):
            continue
        spec = DEPARTMENT_BY_ID[dept_id]
        dept = plan_dept(plan, dept_id)
        headline = dept.get("headline") or spec["title"]
        badges = "".join(badge(t) for t in spec["tiers"])
        rows.append(
            f'<tr><td><a href="#{dept_id}"><b>{esc(spec["title"])}</b></a><br>'
            f'<span style="color:var(--ink-soft)">{esc(headline)}</span></td>'
            f'<td>{esc(spec["promise"])}</td><td>{badges}</td></tr>'
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
</div>"""
    return f"""
<section class="dept" id="contents" style="--accent:{ACCENT["contents"]}">
  <div class="dept-head"><div><h2 class="dept-title">In This Issue</h2></div>
    <div class="dept-promise">Where am I, and how do I read this?</div></div>
  <p class="dek">{len(stacks)} verified line(s) · {len(decisions)} decision spread(s)
    · {esc(issue["deck_name"])}</p>
  <table class="data"><tr><th>Department</th><th>The promise it keeps</th><th>Evidence</th></tr>
    {"".join(rows)}</table>
  {legend}
</section>""" + folio("In This Issue", issue["volume"])


def render_first_turns(issue, plan, prose_doc, cards_by_name):
    dept = plan_dept(plan, "first-turns")
    return (
        dept_open("first-turns", plan)
        + f'<div class="body-copy">{prose(prose_doc, "how_it_wins")}</div>'
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + dept_close("first-turns", issue["volume"])
    )


def render_command_zone(issue, plan, commander, goldfish, cards_by_name):
    """The Commander Mandate department (STYLEv3 §3.3)."""
    dept = plan_dept(plan, "command-zone")
    body = []
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
        + dept_close("command-zone", issue["volume"])
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
        return dept_open("by-the-numbers", plan) + TODO + dept_close("by-the-numbers", issue["volume"])
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
        + dept_close("by-the-numbers", issue["volume"])
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
<article style="border-top:3px solid var(--ink);padding-top:18px;margin-top:26px">
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
        + dept_close("the-kill", issue["volume"])
    )


def render_politics(issue, plan, prose_doc, cards_by_name):
    dept = plan_dept(plan, "politics-table")
    return (
        dept_open("politics-table", plan)
        + f'<div class="body-copy">{prose(prose_doc, "threat_assessment")}</div>'
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + dept_close("politics-table", issue["volume"])
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
  <dl><dt>The line</dt><dd>{esc(b.get("line", ""))}</dd>
  <dt>Signals sent</dt><dd>{esc(b.get("signals", ""))}</dd>
  <dt>Coalition risk</dt><dd>{esc(b.get("coalition_risk", ""))}</dd>
  <dt>Read</dt><dd>{esc(b.get("coaching", ""))}</dd></dl></div>"""
            for b in decision.get("branches", []))
        rec = decision.get("recommendation", {})
        spreads.append(f"""
<article style="border-top:3px solid var(--ink);padding-top:18px;margin-top:26px">
  <h3 style="font-family:var(--display);font-size:1.4em">{esc(decision["title"])}</h3>
  <div class="scenario">{"<br>".join(bits)}<br><br>
    <b>{esc(scenario.get("question", ""))}</b></div>
  <div class="branches">{branches}</div>
  <div class="verdict"><b>Our call: {esc(rec.get("choice", ""))}</b><br>
    {esc(rec.get("rationale", ""))}</div>
</article>""")
    return (
        dept_open("whats-your-play", plan)
        + dept_furniture(dept, cards_by_name)
        + ("".join(spreads) or TODO)
        + dept_close("whats-your-play", issue["volume"])
    )


def render_know_your_enemy(issue, plan, prose_doc, cards_by_name):
    dept = plan_dept(plan, "know-your-enemy")
    boxes = []
    for entry in dept.get("threats", []):
        boxes.append(threat_box(
            entry.get("archetype", ""), entry.get("meter_label", "Threat"),
            float(entry.get("rate", 0.5)),
            f'<p>{esc(entry.get("read", ""))}</p>'
            f'<p><b>Your outs:</b> {esc(", ".join(entry.get("outs", [])))}</p>',
        ))
    return (
        dept_open("know-your-enemy", plan)
        + "".join(boxes)
        + f'<div class="body-copy">{prose(prose_doc, "matchups")}</div>'
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + dept_close("know-your-enemy", issue["volume"])
    )


def _card_tile(card, roles, synergy):
    name = card["name"]
    labels = sorted({
        entry.get("rule", "").split(":")[0]
        for entry in (synergy or {}).get(name, [])[:3] if entry.get("rule")
    })
    chips = "".join(f'<span class="chip">{esc(l)}</span>' for l in labels[:2])
    image = (f'<img src="{esc(card["image"])}" alt="{esc(name)}" loading="lazy">'
             if card.get("image") else "")
    return (
        f'<div class="card-tile">{image}<h4>{esc(name)}</h4>{chips}'
        f'<p>{esc(roles.get(name, ""))}</p></div>'
    )


def render_the_99(issue, plan, cards, prose_doc, synergy):
    """Ranked roster — load-bearing cards lead, depth reads as depth.

    The plan's optional `roster` groups cards by role; anything it doesn't name
    falls into a final "Depth" group so no card silently vanishes.
    """
    dept = plan_dept(plan, "the-99")
    roles = (prose_doc or {}).get("card_roles", {})
    main = [c for c in cards if not c.get("is_sideboard")]
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
        tiles = "".join(_card_tile(c, roles, synergy) for c in group_cards)
        sections.append(f'{heading}<div class="card-grid">{tiles}</div>')
    tiles = "".join(sections)

    side = [c for c in cards if c.get("is_sideboard")]
    side_html = ""
    if side:
        side_tiles = []
        for card in side:
            accessory = card.get("type_line", "") == "Card"
            blurb = roles.get(card["name"]) or (
                "Table aid — no rules text. Use it to track game state on big turns."
                if accessory else "")
            image = (f'<img src="{esc(card["image"])}" alt="{esc(card["name"])}" loading="lazy">'
                     if card.get("image") else "")
            side_tiles.append(
                f'<div class="card-tile">{image}<h4>{esc(card["name"])}</h4>'
                f'<span class="chip">{"Table aid" if accessory else esc(card.get("type_line", ""))}</span>'
                f"<p>{esc(blurb)}</p></div>"
            )
        side_html = ("<h3>Sideboard &amp; table aids</h3>"
                     f'<div class="card-grid">{"".join(side_tiles)}</div>')
    return (
        dept_open("the-99", plan)
        + tiles + side_html
        + dept_close("the-99", issue["volume"])
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
  <p style="font-size:.9em;color:var(--ink-soft)">{esc(h.get("why", ""))}</p></div>"""
        for h in dept.get("hands", []))
    hands_html = f'<div class="branches">{hands}</div>' if hands else ""
    return (
        dept_open("keep-or-ship", plan)
        + meter + hands_html
        + f'<div class="body-copy">{prose(prose_doc, "mulligan")}</div>'
        + dept_captions(dept, cards_by_name)
        + dept_furniture(dept, cards_by_name)
        + dept_close("keep-or-ship", issue["volume"])
    )


def render_upgrade_watch(issue, plan, prose_doc, cards_by_name):
    dept = plan_dept(plan, "upgrade-watch")
    return (
        dept_open("upgrade-watch", plan)
        + f'<div class="body-copy">{prose(prose_doc, "upgrades")}</div>'
        + dept_furniture(dept, cards_by_name)
        + dept_close("upgrade-watch", issue["volume"])
    )


def render_judges_desk(issue, plan, stacks):
    """The proof. Every citation reproduced verbatim — never summarized."""
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
<div class="dossier" id="case-{esc(sid)}">
  <div class="file-tab">Case A-{esc(sid)}</div>
  <div class="dossier-head">
    <div><b>{esc(stack["title"])}</b><br>
      <span style="font-size:.85em;color:var(--ink-soft)">
        Rules version {esc(stack.get("rules_version", "—"))} ·
        status: cleared in {esc(checker.get("iterations", "?"))} review cycle(s)</span></div>
    <div class="stamp">Verified</div>
  </div>
  <ol>{"".join(steps)}</ol>
</div>""")
    return (
        dept_open("judges-desk", plan)
        + '<p class="dek">Every claim the magazine made, with the rule text that backs '
          "it. Nothing here is paraphrased.</p>"
        + ("".join(files) or TODO)
        + dept_close("judges-desk", issue["volume"])
    )


def render_back_page(issue, plan, deck_doc, stacks):
    sha = str(deck_doc.get("decklist_sha256", ""))[:12]
    rules_version = stacks[0].get("rules_version", "—") if stacks else "—"
    return f"""
<section class="dept" id="back-page" style="--accent:{ACCENT["back-page"]}">
  <div class="dept-head"><div><h2 class="dept-title">The Back Page</h2></div>
    <div class="dept-promise">What's in the next issue?</div></div>
  <div class="kicker">Next issue</div>
  <h1 class="feature">{esc(issue["next_issue"])}</h1>
  <p class="dek">Another commander, another 99, the same contract: verified lines,
    seeded numbers, and coaching that says when it's coaching.</p>
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
</section>""" + folio("The Back Page", issue["volume"])


# ── Assembly ────────────────────────────────────────────────────────────


def render_issue(slug, issue, plan, deck_doc, stacks, prose_doc, synergy,
                 goldfish=None, decisions=None):
    """Assemble a complete issue. Deterministic for fixed inputs."""
    cards = deck_doc["cards"]
    cards_by_name = {c["name"]: c for c in cards}
    commanders = [c for c in cards if c["is_commander"]]
    commander = commanders[0] if commanders else None
    decisions = decisions or []
    volume = issue["volume"]

    title = f"{issue['deck_name']} — Pilot's Manual Vol. {volume:03d}"
    og_image = commander.get("image") if commander else ""
    description = (
        f"{issue['deck_name']}: {len(stacks)} rules-verified lines, seeded goldfish "
        f"numbers, and table coaching. Pilot's Manual Vol. {volume:03d}."
    )

    body = "".join([
        render_cover(issue, plan, commander, stacks),
        render_contents(issue, plan, stacks, decisions),
        render_first_turns(issue, plan, prose_doc, cards_by_name),
        render_command_zone(issue, plan, commander, goldfish, cards_by_name),
        render_by_the_numbers(issue, plan, goldfish, cards_by_name),
        render_the_kill(issue, plan, stacks, prose_doc, cards_by_name),
        render_politics(issue, plan, prose_doc, cards_by_name),
        render_whats_your_play(issue, plan, decisions, cards_by_name),
        render_know_your_enemy(issue, plan, prose_doc, cards_by_name),
        render_the_99(issue, plan, cards, prose_doc, synergy),
        render_keep_or_ship(issue, plan, prose_doc, goldfish, cards_by_name),
        render_upgrade_watch(issue, plan, prose_doc, cards_by_name),
        render_judges_desk(issue, plan, stacks),
        render_back_page(issue, plan, deck_doc, stacks),
    ])

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
<style>{CSS}</style>
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
    synergy = load_json(SYNERGY_GRAPH_PATH, {})

    html_out = render_issue(slug, issue, plan, deck_doc, stacks, prose_doc, synergy,
                            goldfish, decisions)
    MANUALS_DIR.mkdir(parents=True, exist_ok=True)
    out = MANUALS_DIR / f"{slug}.html"
    out.write_text(html_out, encoding="utf-8")
    print(
        f"Wrote {out}: Vol. {issue['volume']:03d} · {len(stacks)} verified line(s), "
        f"{len(decisions)} decision spread(s), goldfish: {'yes' if goldfish else 'no'}, "
        f"plan: {'yes' if plan else 'defaults'}"
    )


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot build-manual <slug>`.")
