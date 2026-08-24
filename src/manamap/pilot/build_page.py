"""The compact deck page — the Pilot's Manual, rendered from the same artifacts.

WHAT THIS REPLACES. `build_manual.py` renders a magazine: seventeen departments,
a cover, a contents page, four columnists, ~70 screens. It was the product once;
it is frozen legacy now. This renders the thing a pilot actually opens before
game one — the plan, the roster, the mulligan, the lines, the table read, the
numbers, the proof — at a target of twelve to fifteen screens.

IT REUSES RATHER THAN REWRITES. Roughly seventy percent of what the sections need
already exists as pure functions in `design.py` and `build_manual.py`: the
constellation, the engine schematic, the CSS-only stack theatre, the board block,
the Judge's Desk case index, the card linker with its hover previews, the city
heads that share their ink with the map. Those components take an `esc_fn` and
read no module state, so this module imports them and `design.py` is untouched.
Rewriting them would have produced a second implementation of the theatre, which
is the single most intricate thing in the repo and the least worth duplicating.

NO `<script>`, EVER. Same rule the magazine kept: a page is a standalone file
that rebuilds byte-identically and prints. Everything that folds is `<details>`.

EVERY SECTION DEGRADES TO ABSENT. Not to `[TODO]` — that was a magazine
convention, needed because the Flight Plan indexed every department whether or
not it had copy. Nothing indexes these but the nav, which is generated from what
actually rendered. A deck with no stacks has no LINES section; a deck with no
`issue.json` still renders. That last one matters: `kianne` and `kinnan` have no
issue and no prose, and "a deck at day one" beside "a deck at day ninety" is a
better page than seven identical finished ones.
"""

import json
from datetime import date

from manamap.config import DECKS_DIR, MANUALS_DIR as MANUALS_DIR_PATH
from manamap.pilot import build_manual as bm
from manamap.pilot import design as dz
from manamap.pilot import scenario_facts as sf
from manamap.pilot.common import deck_dir, deck_lifecycle, load_json
from manamap.pilot import page_design as pd
from manamap.pilot.page_spec import SECTION_BY_ID, SECTIONS

MANUALS = "manuals"


# ── Loading ─────────────────────────────────────────────────────────────────

def load(slug):
    """Every input, each optional but `cards.json`.

    `issue.json` is optional here and authoritative when present — a deliberate
    divergence from the v5 spec, which assumed every deck has one. Two do not.
    """
    base = deck_dir(slug)
    doc = bm.load_deck_cards(slug)
    return {
        "slug": slug,
        "deck": doc,
        "issue": load_json(base / "issue.json", {}) or {},
        "prose": load_json(base / "manual_prose.json", {}) or {},
        "stacks": bm.load_verified_stacks(slug),
        "withheld": bm.load_withheld_stacks(slug),
        "decisions": bm.load_decisions(slug),
        "goldfish": load_json(base / "goldfish_metrics.json"),
        "mana": load_json(base / "mana_analysis.json"),
        "engine": load_json(base / "engine.json"),
        "deck_map": load_json(base / "deck_map.json"),
        "tutors": load_json(base / "tutor_guide.json"),
        "debrief": load_json(base / "log_annotations.json"),
        # The deck's own answer to "is this name a creature", so the board block
        # does not print a 4/4 under Permanents because the prose omitted a P/T.
        "creatures": sf.unconditional_creatures(slug),
    }


# ── Furniture ───────────────────────────────────────────────────────────────

def section(sid, body):
    """One section, or nothing at all.

    The `class="dept"` is load-bearing and must stay exactly that:
    `issue_length._SECTION_RE` requires a closing quote straight after `dept`, so
    `class="dept page-sec"` would make every section measure zero words and the
    length report would silently agree that the page got shorter.
    """
    if not body or not body.strip():
        return ""
    spec = SECTION_BY_ID[sid]
    badges = "".join(dz.badge(t) for t in spec["tiers"])
    return (f'<section class="dept" id="{sid}">'
            f'<h2 class="dept-title">{bm.esc(spec["title"])}</h2>'
            f'<p class="dept-promise">{bm.esc(spec["promise"])} {badges}</p>'
            f'{body}</section>')


def page_header(d):
    """Commander, identity, size, version, legend — one line, not a cover."""
    cards = d["deck"]["cards"]
    commander = next((c for c in cards if c.get("is_commander")), {})
    total = sum(int(c.get("quantity") or 1) for c in cards)
    issue = d["issue"]
    name = issue.get("deck_name") or commander.get("name") or d["slug"]
    identity = "".join(commander.get("color_identity") or []) or "C"
    bits = [bm.esc(commander.get("name", "—")), identity, f"{total} cards"]
    # NEVER a build date: `datetime.now()` would break the byte-identical rebuild
    # this file is supposed to guarantee. The authored issue date, or nothing.
    if issue.get("issue_date"):
        bits.append(bm.esc(issue["issue_date"]))
    legend = "".join(dz.badge(t) for t in ("verified", "data", "coach"))
    banner = dz.issue_status_banner(deck_lifecycle(d["slug"]))
    return (f'{banner}<header class="page-head">'
            f'<h1>{bm.esc(name)}</h1>'
            f'<p class="page-sub">{" · ".join(bits)}</p>'
            f'<p class="page-legend">{legend}</p></header>')


def page_nav(rendered_ids):
    """A sticky rail over what actually rendered. No prose, no promises."""
    links = "".join(
        f'<a href="#{sid}">{bm.esc(SECTION_BY_ID[sid]["title"])}</a>'
        for sid in rendered_ids)
    return f'<nav class="page-nav">{links}</nav>'


def page_footer(d):
    sha = (d["deck"].get("decklist_sha256") or "")[:12]
    return ('<footer class="page-foot">'
            f'<p>Built from decklist <code>{bm.esc(sha)}</code>. '
            'Every figure carries its interval, its N and its limits, or it is not '
            'here. ✓ rules-verified · ◆ data-derived, seeded · ★ coaching.</p>'
            '<p class="page-fan">Unofficial Fan Content permitted under the Wizards '
            'of the Coast Fan Content Policy. Not approved or endorsed by Wizards. '
            'Portions of the materials used are property of Wizards of the Coast. '
            '©Wizards of the Coast LLC.</p></footer>')


def _prose(d, key, sub=None):
    """Prose, or empty — never `[TODO]`.

    `build_manual.prose` returns a TODO marker for a missing key, which is right
    for a magazine department that must exist because the contents page indexes
    it. Nothing indexes these but the nav, and the nav is built from what
    rendered.
    """
    doc = d["prose"] or {}
    val = doc.get(key)
    if sub is not None:
        val = (val or {}).get(sub)
    if not val or not str(val).strip():
        return ""
    return bm.prose(doc, key, sub)


# ── 1 PLAN ──────────────────────────────────────────────────────────────────

def _tax_ladder(commander):
    """What the commander costs the second, third and fourth time. CR 903.8."""
    cmc = int(commander.get("cmc") or 0)
    if not cmc:
        return ""
    rows = "".join(
        f"<tr><td>cast {n + 1}</td><td>{cmc + 2 * n}</td></tr>" for n in range(4))
    return ('<table class="data tax-ladder"><caption>Commander tax (CR 903.8)'
            '</caption><thead><tr><th>which cast</th><th>mana value</th></tr>'
            f'</thead><tbody>{rows}</tbody></table>')


def render_plan(d):
    cards = d["deck"]["cards"]
    commander = next((c for c in cards if c.get("is_commander")), {})
    body = _prose(d, "how_it_wins")
    if d["engine"]:
        body += dz.engine_figure(
            d["engine"],
            "Solid where a checker-passed stack proves the step; dashed where it "
            "is a reading nobody has verified.")
    # The conditions the plan quietly assumes away. The renderer decides this,
    # not the author: a thesis stated as arithmetic on an empty table is a true
    # sum about a game nobody played.
    scope = None
    if d["goldfish"]:
        scope = ((d["goldfish"].get("meta") or {}).get("model_assumptions") or [None])[0]
    body += dz.not_modelled_rail(
        (), (d["engine"] or {}).get("open_questions") or (), scope)
    body += _tax_ladder(commander)
    return body


# ── 2 THE 99 ────────────────────────────────────────────────────────────────

def render_the_99(d):
    cards = d["deck"]["cards"]
    dmap = d["deck_map"]
    body = ""
    if dmap:
        body += dz.constellation_figure(
            dmap, "Positions are LOCAL to this deck — it is re-laid-out from its "
                  "own cards, so this is not the atlas.")
    by_name = {c["name"]: c for c in cards}
    stages = bm.engine_stage_of(d["engine"]) if d["engine"] else {}

    groups = []
    if dmap:
        cities = [r for r in (dmap.get("regions") or []) if r.get("level") == 0]
        cities.sort(key=lambda r: (-r.get("count", 0), r.get("id", "")))
        seen = set()
        for i, city in enumerate(cities):
            names = [n for n in (city.get("cards") or []) if n in by_name]
            seen.update(names)
            groups.append((i, city.get("label") or city.get("fallback") or city["id"],
                           city.get("gloss"), names, city.get("verified_count", 0)))
        stray = [c["name"] for c in cards if c["name"] not in seen]
        if stray:
            groups.append((len(groups), "Unmapped", None, stray, 0))
    else:
        groups.append((0, "The 99", None, [c["name"] for c in cards], 0))

    for index, label, gloss, names, verified in groups:
        if not names:
            continue
        body += dz.city_head(index, label, len(names), verified, gloss)
        rows = ""
        for n in sorted(names):
            c = by_name.get(n, {})
            cost = bm.esc(c.get("mana_cost") or "")
            roles = ", ".join((dmap and next(
                (x.get("roles") or [] for x in dmap.get("cards", []) if x["name"] == n),
                [])) or [])
            rows += (f"<tr><td>{bm.card_linkify(bm.esc(n))}</td>"
                     f"<td class='num'>{cost}</td>"
                     f"<td>{bm.esc(roles)}</td>"
                     f"<td>{bm.esc(stages.get(n, ''))}</td></tr>")
        body += ('<table class="data roster"><thead><tr><th>card</th><th>cost</th>'
                 '<th>roles</th><th>stage</th></tr></thead>'
                 f'<tbody>{rows}</tbody></table>')
    return body


# ── 3 KEEP OR SHIP ──────────────────────────────────────────────────────────

def land_histogram(hist):
    """Opening-hand land counts. Nothing rendered this before.

    `power_meter` is one rate; this is eight buckets, so it is its own component
    rather than a misuse of that one.
    """
    if not hist:
        return ""
    total = sum(hist.values()) or 1
    peak = max(hist.values()) or 1
    bars = ""
    for lands in range(8):
        n = hist.get(str(lands), hist.get(lands, 0))
        pct = 100 * n / total
        bars += (f'<div class="hist-row"><span class="hist-k">{lands}</span>'
                 f'<span class="hist-bar"><i style="width:{100 * n / peak:.1f}%"></i>'
                 f'</span><span class="hist-v">{pct:.1f}%</span></div>')
    return ('<figure class="hist"><figcaption>Lands in the opening seven, over '
            f'{total:,} seeded hands</figcaption>{bars}</figure>')


def render_keep_or_ship(d):
    body = _prose(d, "mulligan")
    g = ((d["goldfish"] or {}).get("metrics") or {}).get("opening_hand") or {}
    if g.get("keep_first_seven_rate") is not None:
        body += dz.power_meter("Keeps the first seven", g["keep_first_seven_rate"])
    body += land_histogram(g.get("first_seven_land_histogram"))
    return body


# ── 4 THE LINES ─────────────────────────────────────────────────────────────

def render_the_lines(d):
    body = ""
    for st in d["stacks"]:
        sid = st.get("id") or ""
        head, dek = bm.stack_headline(st.get("title", ""))
        body += f'<article class="line" id="line-{bm.esc(sid)}">'
        body += f'<h3>{bm.esc(head)}</h3>'
        if dek:
            body += f'<p class="line-dek">{bm.esc(dek)}</p>'
        intro = _prose(d, "combo_lines", sid)
        if intro:
            body += intro
        # THE ARGUMENT STAYS OPEN; THE EVIDENCE FOLDS. Board and theatre go into
        # ONE fold together, and that is what the spec's "~4 screens" estimate
        # for this section actually requires — measured, after everything else
        # matched or beat its per-section estimate and this section came in at
        # 8.2 against 4. The board block is 266px on radagast and the resolution
        # is taller still; a reader deciding whether to study a line does that
        # from the question, the intro and the result, and opens the fold when
        # the answer is yes.
        steps = (st.get("resolution") or {}).get("steps") or []
        board = bm.render_board_block(st.get("scenario") or {},
                                      creatures=d.get("creatures"))
        if board or steps:
            label = "The board and the walk-through"
            if steps:
                label += f" — {len(steps)} step(s)"
            body += (f'<details class="theatre-fold"><summary>{label}</summary>'
                     + board
                     + (dz.stack_theatre(sid, steps) if steps else "")
                     + '</details>')
        # THE RESULT IS A SENTENCE, NOT A SECOND BOARD. `render_after_block`
        # draws the whole post-resolution state, and measured on radagast that is
        # 266px — the same height as the board it follows, so the two together
        # were 57% of every line. It is ALSO rendered inside every case file in
        # THE RECORD, so the page was drawing the same board twice. The authored
        # summary says what happened in prose, which is what "result" meant.
        summary = ((st.get("resolution") or {}).get("final_state") or {}).get("summary")
        if summary:
            body += f'<p class="line-result">{bm.esc_x(summary)}</p>'
        body += '</article>'
    return body


# ── 5 AT THE TABLE ──────────────────────────────────────────────────────────

def tutor_table(guide):
    """Scenario -> fetch, two columns.

    The v5 spec asked for `default / behind / closing`. `tutor_guide.json` carries
    free-text scenarios, so deriving three labelled situations from them is
    invention — which is the thing `validate_tutor_guide` exists to prevent.
    """
    tutors = (guide or {}).get("tutors") or []
    if not tutors:
        return ""
    rows = ""
    for t in tutors:
        targets = t.get("targets") or []
        for i, tg in enumerate(targets):
            name = bm.card_linkify(bm.esc(t.get("card", ""))) if i == 0 else ""
            rows += (f"<tr><td>{name}</td><td>{bm.esc(tg.get('scenario', ''))}</td>"
                     f"<td>{bm.card_linkify(bm.esc(tg.get('fetch', '')))}</td></tr>")
    return ('<table class="data tutors"><thead><tr><th>tutor</th><th>when</th>'
            f'<th>get</th></tr></thead><tbody>{rows}</tbody></table>')


def render_at_the_table(d):
    body = _prose(d, "threat_assessment") + _prose(d, "matchups")
    body += tutor_table(d["tutors"])
    return body


# ── 6 PLAY ──────────────────────────────────────────────────────────────────

def render_play(d):
    body = ""
    for dec in d["decisions"]:
        sc = dec.get("scenario") or {}
        q = sc.get("question", "")
        branches = dec.get("branches") or []
        cards = "".join(
            f'<div class="branch"><h4>{bm.esc(b.get("choice", ""))}</h4>'
            f'{bm.esc_x_paras(b.get("signals", ""))}'
            f'{bm.esc_x_paras(b.get("coaching", ""))}</div>'
            for b in branches)
        rec = (dec.get("recommendation") or {}).get("choice", "")
        body += ('<details class="spread"><summary>'
                 f'{bm.esc(sc.get("spot") or q)}</summary>'
                 f'<p class="spread-q">{bm.esc(q)}</p>'
                 f'<div class="branches">{cards}</div>'
                 f'<p class="spread-rec"><b>Take:</b> {bm.esc(rec)}</p>'
                 '</details>')
    return body


# ── 7 THE DEBRIEF ───────────────────────────────────────────────────────────

def render_debrief(d):
    """Lessons from games played. NOT the log.

    The v5 spec forbids the manual growing "a log or a version panel". That
    forbids a FEED, and the mechanical form of the rule is that this section may
    print nothing keyed by a log entry id or a timestamp — a test asserts exactly
    that. What it prints is the synthesis: takeaways and card reads, aggregated,
    with the sample size stated, because a ★ claim drawn from three games that
    does not say "three" is the failure the not-modelled rail exists to prevent.
    """
    entries = ((d["debrief"] or {}).get("entries") or {})
    if not entries:
        return ""
    rows = list(entries.values())
    takeaways, reads = [], {}
    for e in rows:
        for t in e.get("takeaways") or []:
            if t not in takeaways:
                takeaways.append(t)
        for c in e.get("cards") or []:
            key = (c.get("card"), c.get("read"))
            reads[key] = reads.get(key, 0) + 1
    if not takeaways and not reads:
        return ""
    body = (f'<p class="ev">Drawn from {len(rows)} logged game'
            f'{"" if len(rows) == 1 else "s"}.</p>')
    if takeaways:
        body += "<ul class='takeaways'>" + "".join(
            f"<li>{bm.card_linkify(bm.esc(t))}</li>" for t in takeaways) + "</ul>"
    if reads:
        items = sorted(reads.items(), key=lambda kv: (-kv[1], kv[0][0] or ""))
        rows_html = "".join(
            f"<tr><td>{bm.card_linkify(bm.esc(card))}</td><td>{bm.esc(read)}</td>"
            f"<td class='num'>{n}</td></tr>" for (card, read), n in items)
        body += ('<table class="data reads"><thead><tr><th>card</th><th>read</th>'
                 f'<th>games</th></tr></thead><tbody>{rows_html}</tbody></table>')
    return body


# ── 8 THE NUMBERS ───────────────────────────────────────────────────────────

def render_the_numbers(d):
    body = ""
    g = (d["goldfish"] or {}).get("metrics") or {}
    if g:
        cmd = g.get("commander") or {}
        pairs = [("Commander cast, mean turn", cmd.get("mean_cast_turn")),
                 ("Cast by turn six", f"{100 * (cmd.get('cast_by_turn_6_rate') or 0):.0f}%"),
                 ("Iterations", f"{g.get('iterations', 0):,}")]
        body += dz.fast_facts("Goldfish", [(k, v) for k, v in pairs if v is not None])
        rows = "".join(
            f"<tr><td>{t.get('label','')}</td>"
            f"<td class='num'>{100 * (t.get('by_turn_6_rate') or 0):.0f}%</td>"
            f"<td class='num'>{t.get('mean_turn') or '—'}</td></tr>"
            for t in (g.get("targets") or []))
        if rows:
            body += ('<table class="data"><caption>Engine targets</caption><thead><tr>'
                     '<th>component</th><th>by turn 6</th><th>mean turn</th></tr></thead>'
                     f'<tbody>{rows}</tbody></table>')
    if d["mana"]:
        m = d["mana"]
        # `source_targets` is {colour: target}; the counts live under
        # `sources.total`, keyed by every WUBRG letter including the ones this
        # deck does not play. Only colours with a target belong in the table.
        have = (m.get("sources") or {}).get("total") or {}
        rows = ""
        for colour, target in sorted((m.get("source_targets") or {}).items()):
            got = have.get(colour, 0)
            short = got - target
            mark = "" if short >= 0 else f" <b class='short'>{short}</b>"
            rows += (f"<tr><td>{bm.esc(colour)}</td><td class='num'>{got}</td>"
                     f"<td class='num'>{target}{mark}</td></tr>")
        if rows:
            body += ('<table class="data"><caption>Colour sources against the '
                     'hypergeometric target — P(pips by the earliest castable turn) '
                     '≥ 90% over 99 cards</caption><thead><tr><th>colour</th>'
                     '<th>sources</th><th>target</th></tr></thead>'
                     f'<tbody>{rows}</tbody></table>')
    # The assumptions travel with the figures or the figures do not travel.
    assumptions = ((d["goldfish"] or {}).get("meta") or {}).get("model_assumptions") or []
    if assumptions:
        body += ("<details class='assumptions'><summary>What the goldfish does not "
                 f"model ({len(assumptions)})</summary><ul>"
                 + "".join(f"<li>{bm.esc(a)}</li>" for a in assumptions)
                 + "</ul></details>")
    return body


# ── 9 THE RECORD ────────────────────────────────────────────────────────────

def render_the_record(d):
    """The proof, one row per case, unabridged on click.

    A withheld case still gets a row. Nineteen published resolutions across the
    fleet say things like "exactly as in stack 001", and stack 001 is deliberately
    not printed — so without these rows the reader is sent hunting for a case that
    was held back on purpose. The obvious fix, editing the referring prose, is the
    wrong one: those artifacts passed the checker, so their step text is evidence.
    """
    if not d["stacks"] and not d["withheld"]:
        return ""
    body = ('<p class="dek">Every ✓ claim on this page, with the rule text behind '
            'it. Nothing is paraphrased. Open a case for its full record.</p>')
    body += bm.judges_desk_files(d["stacks"], back_label="The Lines",
                                 contents_link=False)
    body += bm.withheld_cases(d["withheld"])
    return body


RENDERERS = {
    "plan": render_plan, "the-99": render_the_99,
    "keep-or-ship": render_keep_or_ship, "the-lines": render_the_lines,
    "at-the-table": render_at_the_table, "play": render_play,
    "debrief": render_debrief, "the-numbers": render_the_numbers,
    "the-record": render_the_record,
}


# ── Assembly ────────────────────────────────────────────────────────────────

def render_page(d):
    """The whole page. Deterministic for fixed inputs."""
    cards = d["deck"]["cards"]
    commander = next((c for c in cards if c.get("is_commander")), None)
    bm.set_card_links(cards, commander["name"] if commander else None)
    try:
        bodies = []
        for sid, *_ in SECTIONS:
            html = section(sid, RENDERERS[sid](d))
            if html:
                bodies.append((sid, html))
        head = page_header(d)
        nav = page_nav([sid for sid, _ in bodies])
        body = "".join(h for _, h in bodies)
    finally:
        bm.clear_card_links()
    css = pd.stylesheet_link()
    title = bm.esc((d["issue"].get("deck_name")
                    or (commander or {}).get("name") or d["slug"]))
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{title} — Pilot's Manual</title>"
        f'<link rel="stylesheet" href="magazine.css?v={dz.stylesheet_version()}">'
        f'{css}</head><body class="page">'
        f'<div class="page-trim">{head}{nav}{body}{page_footer(d)}</div>'
        "</body></html>\n")


def main(args):
    slug = args.slug
    d = load(slug)
    html = render_page(d)
    out = getattr(args, "out", None)
    # `manuals/p/<slug>.html`, NOT `manuals/<slug>.html`. The compact page is
    # meant to replace the magazine and eventually will, but until the phase that
    # deletes `build_manual.py` they coexist — and writing to the magazine's path
    # would silently overwrite nine tracked, byte-compared files the moment
    # anyone ran this.
    path = (MANUALS_DIR_PATH / "p" / f"{slug}.html") if not out else None
    if out:
        from manamap.pilot.common import resolve_out_path
        path = resolve_out_path(out, slug, "build-page", ext=".html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    # The sheet is content-addressed, so writing it every time is a no-op unless
    # it changed — and a page whose stylesheet is missing measures three times
    # its real height, which is how the first measurement of this renderer came
    # in worse than the magazine it replaces.
    pd.write_stylesheet()
    # Both sheets sit beside the page wherever it lands. They are linked
    # relatively, and a page whose stylesheet 404s measures three times its real
    # height — which is how the first render of this module came in worse than
    # the magazine it replaces.
    if path.parent != MANUALS_DIR_PATH:
        (path.parent / "page.css").write_bytes(pd.stylesheet_bytes())
        import shutil
        mag = MANUALS_DIR_PATH / "magazine.css"
        if mag.exists():
            shutil.copy(mag, path.parent / "magazine.css")
    from manamap.pilot import issue_length
    rendered = issue_length.sections(html)
    print(f"Wrote {path} ({len(html):,} bytes, {len(rendered)} section(s))")
    for sid, fragment in rendered:
        words = issue_length.words(fragment)
        visible = issue_length.visible_words(fragment)
        print(f"  {sid:<14} {words:>6,} words  {visible:>6,} visible")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot build-page <slug>`.")
