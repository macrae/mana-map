"""Pilot: render a deck's pilot's manual as standalone zine HTML.

Fully deterministic — no LLM calls. Inputs: cards.json, checker-passed stack
scenarios, manual_prose.json (agent-written, human-editable), and the
combo/synergy/obsolescence graphs. Only verified stacks appear; missing prose
renders as a visible [TODO] rather than failing the build.
"""

import html
import json

from manamap.config import MANUALS_DIR, SYNERGY_GRAPH_PATH
from manamap.pilot.common import deck_dir, load_deck_cards

# Dark theme matching viz/ (background #1a1a2e, panel #16213e, gold #c4a747).
CSS = """
:root { --bg:#1a1a2e; --panel:#16213e; --gold:#c4a747; --text:#e8e6e3; --dim:#9a97b0;
        --magenta:#E040FB; --green:#4ade80; --border:#3a3a5a; }
* { box-sizing:border-box; margin:0; }
body { background:var(--bg); color:var(--text); font:16px/1.6 Georgia, serif; }
.page { max-width:880px; margin:0 auto; padding:48px 32px; }
.spread { background:var(--panel); border:1px solid var(--border); border-radius:8px;
          padding:32px; margin:32px 0; page-break-inside:avoid; }
h1 { font-size:2.6em; color:var(--gold); line-height:1.15; }
h2 { color:var(--gold); font-size:1.6em; margin-bottom:12px; border-bottom:1px solid var(--border);
     padding-bottom:6px; }
h3 { color:var(--text); margin:16px 0 6px; }
.tagline { font-style:italic; color:var(--dim); font-size:1.2em; margin:8px 0 16px; }
.cover { text-align:center; }
.cover img { max-width:340px; border-radius:12px; margin:24px auto; display:block; }
.verified { display:inline-block; background:rgba(74,222,128,.12); color:var(--green);
            border:1px solid var(--green); border-radius:12px; padding:1px 10px;
            font:12px sans-serif; vertical-align:middle; margin-left:8px; }
.badge { display:inline-block; border-radius:12px; padding:1px 10px; font:11px sans-serif;
         vertical-align:middle; margin-left:10px; letter-spacing:.5px; }
.badge-rules { background:rgba(74,222,128,.12); color:var(--green); border:1px solid var(--green); }
.badge-data { background:rgba(96,165,250,.12); color:#60a5fa; border:1px solid #60a5fa; }
.badge-coach { background:rgba(196,167,71,.12); color:var(--gold); border:1px solid var(--gold); }
.legend { font:13px sans-serif; color:var(--dim); text-align:left; margin:20px auto 0;
          max-width:560px; line-height:1.9; }
table.metrics { border-collapse:collapse; width:100%; margin:12px 0; font-size:.92em; }
table.metrics th, table.metrics td { border:1px solid var(--border); padding:6px 10px; text-align:left; }
table.metrics th { color:var(--gold); font-family:sans-serif; font-size:.85em; }
.branch { background:rgba(0,0,0,.25); border:1px solid var(--border); border-radius:8px;
          padding:14px; margin:12px 0; }
.branch.recommended { border-color:var(--green); }
.branch h4 { color:var(--text); margin-bottom:6px; }
.branch .pick { color:var(--green); font:11px sans-serif; letter-spacing:.5px; }
.branch dl { font-size:.9em; margin-top:8px; }
.branch dt { color:var(--gold); font-family:sans-serif; font-size:.8em; margin-top:6px; }
.branch dd { color:var(--dim); margin-left:0; }
.todo { color:var(--magenta); font-family:monospace; }
ol.steps { padding-left:24px; }
ol.steps li { margin:10px 0; }
.effect { color:var(--dim); }
details { margin:4px 0 4px 8px; font-size:.9em; }
summary { color:var(--gold); cursor:pointer; font-family:monospace; }
blockquote { border-left:3px solid var(--gold); padding-left:12px; color:var(--dim);
             margin:6px 0; font-style:italic; }
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:16px; }
.card { background:rgba(0,0,0,.25); border:1px solid var(--border); border-radius:8px; padding:14px; }
.card img { width:100%; border-radius:6px; }
.card .labels { font:11px sans-serif; color:var(--magenta); margin:6px 0 2px; }
.card p { font-size:.85em; color:var(--dim); }
.scenario-box { background:rgba(0,0,0,.25); border-radius:6px; padding:14px; margin:12px 0;
                font-size:.92em; }
.footer { text-align:center; color:var(--dim); font-size:.8em; margin-top:48px; }
@media print { body { background:#fff; color:#111; } .spread { page-break-after:always; } }
"""


def esc(value):
    return html.escape(str(value)) if value is not None else ""


def prose(prose_doc, key, sub=None):
    """Fetch a prose string, or a visible TODO placeholder."""
    node = prose_doc.get(key, {})
    if sub is not None:
        node = node.get(sub, "") if isinstance(node, dict) else ""
    if not node or not isinstance(node, str):
        label = f"{key}.{sub}" if sub else key
        return f'<span class="todo">[TODO: {esc(label)} prose]</span>'
    return "".join(f"<p>{esc(p)}</p>" for p in node.split("\n\n"))


BADGES = {
    "rules": '<span class="badge badge-rules">✓ RULES-VERIFIED</span>',
    "data": '<span class="badge badge-data">◆ DATA-DERIVED</span>',
    "coach": '<span class="badge badge-coach">★ COACHING</span>',
}

LEGEND = """
<div class="legend">
  <strong>How to read this manual:</strong><br>
  <span class="badge badge-rules">✓ RULES-VERIFIED</span> machine-checked against the
  Comprehensive Rules — every step cites verbatim rule text, verified adversarially<br>
  <span class="badge badge-data">◆ DATA-DERIVED</span> reproducible from committed data
  and seeded simulations (assumptions stated)<br>
  <span class="badge badge-coach">★ COACHING</span> judgment from play patterns —
  grounded in the data, but a human call
</div>"""


def load_verified_stacks(slug):
    stacks_dir = deck_dir(slug) / "stacks"
    verified, skipped = [], []
    for path in sorted(stacks_dir.glob("*.json")):
        with open(path) as f:
            doc = json.load(f)
        if (doc.get("checker") or {}).get("verdict") == "pass":
            verified.append(doc)
        else:
            skipped.append(path.name)
    return verified, skipped


def load_decisions(slug):
    decisions_dir = deck_dir(slug) / "decisions"
    docs = []
    for path in sorted(decisions_dir.glob("*.json")):
        with open(path) as f:
            docs.append(json.load(f))
    return docs


def render_goldfish(metrics_doc):
    if not metrics_doc:
        return ""
    meta = metrics_doc["meta"]
    m = metrics_doc["metrics"]
    turns = sorted(m["land_drop_hit_rate_by_turn"], key=int)

    def row(label, values):
        cells = "".join(f"<td>{esc(v)}</td>" for v in values)
        return f"<tr><th>{esc(label)}</th>{cells}</tr>"

    turn_table = "<table class='metrics'>" + row("Turn", turns) + row(
        "Land drop hit", [f"{m['land_drop_hit_rate_by_turn'][t]:.0%}" for t in turns]
    ) + row(
        "Mean mana available", [m["mean_available_mana_by_turn"][t] for t in turns]
    ) + row(
        "Mean bodies on board", [m["mean_bodies_by_turn"][t] for t in turns]
    ) + "</table>"

    commander = m["commander"]
    target_rows = "".join(
        f"<tr><td>{esc(t['label'])}</td><td>{t['assembled_rate']:.0%}</td>"
        f"<td>{esc(t['mean_turn'] if t['mean_turn'] is not None else '—')}</td>"
        f"<td>{t['by_turn_6_rate']:.0%}</td></tr>"
        for t in m["targets"]
    )
    targets_table = (
        "<table class='metrics'><tr><th>Assembly target</th><th>Assembled</th>"
        "<th>Mean turn</th><th>By turn 6</th></tr>" + target_rows + "</table>"
        if m["targets"] else ""
    )
    assumptions = "".join(f"<li>{esc(a)}</li>" for a in meta["model_assumptions"])

    return f"""
<section class="spread"><h2>Goldfish Numbers {BADGES["data"]}</h2>
  <p>{esc(meta["commander"])} comes down on <strong>turn {commander["mean_cast_turn"]}</strong> on
  average (median turn {esc(commander["median_cast_turn"])}), and is out by turn 6 in
  <strong>{commander["cast_by_turn_6_rate"]:.0%}</strong> of games. First-seven keep rate:
  {m["opening_hand"]["keep_first_seven_rate"]:.0%}.</p>
  {turn_table}
  {targets_table}
  <details><summary>Simulation model &amp; assumptions (seed {meta["seed"]},
  {meta["iterations"]:,} iterations)</summary><ul>{assumptions}</ul></details>
</section>"""


def render_decision_spread(decision):
    scenario = decision["scenario"]
    board_bits = []
    board = scenario.get("board", {})
    if isinstance(board, dict):
        for key, value in board.items():
            board_bits.append(f"<strong>{esc(key)}:</strong> {esc(value)}")
    branches_html = []
    recommended = (decision.get("recommendation") or {}).get("choice")
    for branch in decision["branches"]:
        is_pick = branch["choice"] == recommended
        pick = '<div class="pick">★ RECOMMENDED LINE</div>' if is_pick else ""
        cites = "".join(
            f'<details><summary>{esc(c["rule"])}</summary>'
            f"<blockquote>{esc(c['quote'])}</blockquote></details>"
            for c in branch.get("citations", [])
        )
        branches_html.append(f"""
    <div class="branch{' recommended' if is_pick else ''}">{pick}
      <h4>{esc(branch["choice"])}</h4>
      <p class="effect">{esc(branch["line"])}</p>
      <dl>
        <dt>Signals sent</dt><dd>{esc(branch["signals"])}</dd>
        <dt>Coalition risk</dt><dd>{esc(branch["coalition_risk"])}</dd>
        <dt>Coaching</dt><dd>{esc(branch["coaching"])}</dd>
      </dl>{cites}
    </div>""")
    rationale = esc((decision.get("recommendation") or {}).get("rationale", ""))
    return f"""
<section class="spread"><h2>{esc(decision["title"])} {BADGES["coach"]}</h2>
  <div class="scenario-box">{"<br>".join(board_bits)}<br>
    <strong>Decision:</strong> {esc(scenario.get("question", ""))}</div>
  {"".join(branches_html)}
  <p><strong>Why the recommended line:</strong> {rationale}</p>
</section>"""


def render_stack_spread(stack, prose_doc):
    scenario = stack["scenario"]
    steps_html = []
    for step in stack["resolution"]["steps"]:
        cites = "".join(
            f'<details><summary>{esc(c["rule"])}</summary>'
            f"<blockquote>{esc(c['quote'])}</blockquote></details>"
            for c in step.get("citations", [])
        )
        steps_html.append(
            f"<li><strong>{esc(step['action'])}</strong>"
            f'<div class="effect">{esc(step.get("effect", ""))}</div>{cites}</li>'
        )
    checker = stack.get("checker", {})
    stack_lines = "".join(
        f"<li>{esc(item.get('object', '?'))} ({esc(item.get('controller', '?'))})</li>"
        for item in reversed(scenario.get("stack", []))
    )
    final = stack.get("resolution", {}).get("final_state", {})
    return f"""
<section class="spread">
  <h2>{esc(stack["title"])}
    <span class="verified">✓ verified · {checker.get("iterations", "?")} iteration(s)</span></h2>
  {prose(prose_doc, "combo_lines", stack["id"])}
  <div class="scenario-box">
    <strong>The stack (top first):</strong><ol>{stack_lines}</ol>
    <strong>Question:</strong> {esc(scenario.get("question", ""))}
  </div>
  <h3>Resolution</h3>
  <ol class="steps">{"".join(steps_html)}</ol>
  <h3>Where you end up</h3>
  <p>{esc(final.get("summary", ""))}</p>
</section>"""


def render_card_roles(cards, prose_doc, synergy):
    roles = prose_doc.get("card_roles", {})
    tiles = []
    for card in cards:
        if card["is_commander"]:
            continue
        name = card["name"]
        labels = sorted({
            label
            for entry in synergy.get(name, [])
            for label in entry.get("synergies", [])
        })[:4]
        blurb = roles.get(name)
        blurb_html = (
            f"<p>{esc(blurb)}</p>" if blurb
            else '<p class="todo">[TODO: role]</p>' if labels else ""
        )
        image = f'<img src="{esc(card["image"])}" alt="{esc(name)}" loading="lazy">' if card["image"] else ""
        tiles.append(
            f'<div class="card">{image}<h3>{esc(name)}</h3>'
            f'<div class="labels">{esc(" · ".join(labels))}</div>{blurb_html}</div>'
        )
    return "".join(tiles)


def render_manual(slug, deck_doc, stacks, prose_doc, synergy, goldfish=None, decisions=None):
    cards = deck_doc["cards"]
    commanders = [c for c in cards if c["is_commander"]]
    commander = commanders[0] if commanders else None
    commander_img = (
        f'<img src="{esc(commander["image"])}" alt="{esc(commander["name"])}">'
        if commander and commander.get("image") else ""
    )
    title = " & ".join(c["name"] for c in commanders) or slug
    stack_spreads = "".join(render_stack_spread(s, prose_doc) for s in stacks)
    if not stack_spreads:
        stack_spreads = '<section class="spread"><h2>Combo Lines</h2><p class="todo">' \
            "[TODO: no verified stack scenarios yet — run the resolve-stack skill]</p></section>"

    goldfish_section = render_goldfish(goldfish)
    decision_spreads = "".join(render_decision_spread(d) for d in (decisions or []))
    table_section = f"""
<section class="spread"><h2>Playing the Table {BADGES["coach"]}</h2>
  <p>Archetypal board states and the political reads that go with them.
  These are coaching calls — grounded in the numbers above, decided by judgment.</p>
  <h3>Threat assessment</h3>
  {prose(prose_doc, "threat_assessment")}
</section>""" + decision_spreads

    matchups_section = (
        f'<section class="spread"><h2>Matchups {BADGES["coach"]}</h2>'
        f'{prose(prose_doc, "matchups")}</section>'
    )

    tagline = prose_doc.get("cover", {}).get("tagline", "")
    og_image = (
        f'\n<meta property="og:image" content="{esc(commander["image"])}">'
        if commander and commander.get("image") else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} — Pilot's Manual</title>
<meta property="og:title" content="{esc(title)} — Pilot's Manual">
<meta property="og:description" content="{esc(tagline) or 'A machine-verified Commander pilot manual.'}">
<meta property="og:type" content="article">{og_image}
<meta name="twitter:card" content="summary_large_image">
<style>{CSS}</style></head><body><div class="page">

<section class="spread cover">
  <h1>{esc(title)}</h1>
  <div class="tagline">{esc(prose_doc.get("cover", {}).get("tagline", "")) or
                        '<span class="todo">[TODO: tagline]</span>'}</div>
  {commander_img}
  {prose(prose_doc, "cover", "identity")}
  {LEGEND}
  <p class="footer">A verified pilot's manual · {esc(slug)} · {len(stacks)} verified line(s)</p>
</section>

<section class="spread"><h2>How the Deck Wins</h2>{prose(prose_doc, "how_it_wins")}</section>

{goldfish_section}

{stack_spreads}

{table_section}

{matchups_section}

<section class="spread"><h2>Card Roles {BADGES["coach"]}</h2>
  <div class="cards">{render_card_roles(cards, prose_doc, synergy)}</div>
</section>

<section class="spread"><h2>Mulligan Guide {BADGES["coach"]}</h2>{prose(prose_doc, "mulligan")}</section>

<section class="spread"><h2>Upgrade Paths {BADGES["data"]}</h2>{prose(prose_doc, "upgrades")}</section>

<p class="footer">Every combo line above is machine-verified: each resolution step cites the
Magic Comprehensive Rules, and every citation was checked against the full rule text.</p>
</div></body></html>
"""


def main(args):
    deck_doc = load_deck_cards(args.slug)
    stacks, skipped = load_verified_stacks(args.slug)
    if skipped:
        print(f"  Skipping {len(skipped)} unverified stack(s): {', '.join(skipped)}")

    prose_path = deck_dir(args.slug) / "manual_prose.json"
    prose_doc = {}
    if prose_path.exists():
        with open(prose_path) as f:
            prose_doc = json.load(f)
    else:
        print(f"  {prose_path.name} not found — building with [TODO] placeholders.")

    synergy = {}
    if SYNERGY_GRAPH_PATH.exists():
        with open(SYNERGY_GRAPH_PATH) as f:
            synergy = json.load(f)

    goldfish = None
    goldfish_path = deck_dir(args.slug) / "goldfish_metrics.json"
    if goldfish_path.exists():
        with open(goldfish_path) as f:
            goldfish = json.load(f)

    decisions = load_decisions(args.slug)

    MANUALS_DIR.mkdir(exist_ok=True)
    out = MANUALS_DIR / f"{args.slug}.html"
    out.write_text(
        render_manual(args.slug, deck_doc, stacks, prose_doc, synergy, goldfish, decisions),
        encoding="utf-8",
    )
    print(
        f"Wrote {out} ({len(stacks)} verified line(s), "
        f"{len(decisions)} decision spread(s), goldfish: {'yes' if goldfish else 'no'})"
    )


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot build-manual <slug>`.")
