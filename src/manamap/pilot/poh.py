"""The Pilot's Operating Handbook — the renderer.

The closest real object is the binder in an aircraft's side pocket, and its
ORDERING is what is being taken: emergencies before normal operation, a
procedure as a numbered checklist rather than prose, one procedure per sheet.
The dossier (`viz/deck.html`) is what is on the desk; this is what is in the
cockpit.

Phase 1 is the half that regenerates from tracked artifacts — front matter,
General with the schematic, Limitations, Performance, Systems. The authored half
(Emergency, Normal, Handling, Matchups) is drafted by an agent and edited by the
pilot, and renders when it exists.

THREE RULES INHERITED FROM THE PAGE THIS REPLACES, all load-bearing:

  NO `<script>`, EVER. Everything that folds is `<details>`. A handbook is a
  standalone file that prints.

  BYTE-IDENTICAL REBUILD. No `datetime.now()` anywhere — a build date would make
  every rebuild a diff. Dates come from `versions.json`, which is authored
  history, or from nowhere.

  A MISSING ARTIFACT RENDERS AS AN ABSENCE, never as `[TODO]` and never as zero.
  A section with nothing to say says so and names the command; a section that
  cannot exist for this deck is stated rather than dropped, because a
  disappearing section is indistinguishable from one nobody wrote.

AND ONE THAT IS NEW HERE: nothing on this page is computed. Every figure comes
from an artifact on disk, which is why `deck-audit --write` had to exist before
this file could — the sixteen axes were printed and never persisted, and a
handbook that recomputes its own Limitations is a handbook that can disagree
with the dossier.
"""

import html
import json

from manamap.config import DECKS_DIR, MANUALS_DIR
from manamap.pilot import poh_spec as spec
from manamap.pilot.common import expand_copies, load_json

ARTIFACT_DIR = "p"


def esc(v):
    return html.escape("" if v is None else str(v), quote=True)


def load(slug):
    """Every tracked artifact the handbook reads. Nothing is computed."""
    base = DECKS_DIR / slug
    return {
        "slug": slug,
        "base": base,
        "cards": load_json(base / "cards.json") or {},
        "engine": load_json(base / "engine.json"),
        "audit": load_json(base / "audit.json"),
        "goldfish": load_json(base / "goldfish_metrics.json"),
        "diagnostic": load_json(base / "diagnostic.json"),
        "mana": load_json(base / "mana_analysis.json"),
        "bracket": load_json(base / "bracket_report.json"),
        "versions": load_json(base / "versions.json") or {},
        "deck_versions": load_json(base / "deck_versions.json") or {},
        "revisions": load_json(base / "manual_revisions.json") or {},
        "info": load_json(base / "info.json") or {},
    }


# ── the furniture ────────────────────────────────────────────────────────

def callout(level, text):
    """One of three, and `validate_poh` caps them at two a page."""
    if level not in spec.CALLOUTS:
        raise ValueError(f"{level!r} is not a callout level")
    return (f'<aside class="poh-call {level}">'
            f'<span class="lbl">{level.upper()}</span>{esc(text)}</aside>')


def section(sid, body, changed=()):
    """A numbered section, or nothing.

    An empty body renders NOTHING rather than an empty shell — but a section
    that is absent for a REASON calls `absent()` instead and renders that. The
    difference matters: "this deck has no engine model" and "nobody has written
    section 6 yet" look identical to a reader when both are silence.
    """
    if not body or not body.strip():
        return ""
    s = spec.SECTION_BY_ID[sid]
    bar = " rev-bar" if s["number"] in (changed or ()) else ""
    return (f'<section class="poh-sec{bar}" id="s{s["number"]}">'
            f'<h2><span class="n">{s["number"]}</span>{esc(s["title"])}</h2>'
            f'<p class="promise">{esc(s["promise"])}</p>{body}</section>')


def sub(number, title, body):
    if not body or not body.strip():
        return ""
    return (f'<div class="poh-sub" id="s{number.replace(".", "-")}">'
            f'<h3><span class="n">{esc(number)}</span>{esc(title)}</h3>{body}</div>')


def absent(what, why, how=None):
    """A stated absence. ABSENT IS NOT ZERO and it is not silence."""
    return (f'<p class="ev"><b>{esc(what)} — not available.</b> {esc(why)}'
            + (f' <code>{esc(how)}</code>' if how else "") + "</p>")


# ── 0. front matter ──────────────────────────────────────────────────────

def render_front_matter(d):
    cards = d["cards"].get("cards") or []
    commander = next((c["name"] for c in cards if c.get("is_commander")), None)
    total = sum(int(c.get("quantity") or 1) for c in cards)
    paper = (d["deck_versions"].get("paper") or {})
    revs = d["revisions"].get("revisions") or []
    cur = revs[-1] if revs else None

    # THE BINDING. A handbook that does not say which list it applies to is a
    # handbook you cannot trust against the deck in your hand. `applies_to` is a
    # semver tag; `deck_versions.py` already polices that form.
    rev_line = ""
    if cur:
        rev_line = (f'Manual Rev {esc(cur.get("rev"))} — applies to '
                    f'{esc(cur.get("applies_to"))}')
    elif paper.get("decklist_sha256"):
        # THE SEMVER TAG, not `paper.version` — that is an ORDINAL (3), and
        # "applies to 3" tells a reader nothing they can match against a deck.
        # The tag is what every other artifact quotes, so it is what binds.
        tag = next((name for name, t in sorted((d["deck_versions"].get("tags") or {}).items())
                    if t.get("decklist_sha256") == paper["decklist_sha256"]), None)
        rev_line = (f'Applies to the sleeved list, {esc(tag)}' if tag
                    else f'Applies to the sleeved list (version '
                         f'{esc(paper.get("version"))}, untagged)')
    sha = (d["cards"].get("decklist_sha256") or "")[:12]

    head = (f'<div class="poh-title"><h1>{esc(commander or d["slug"])}</h1>'
            f'<p class="sub">Pilot\'s Operating Handbook · {total} cards</p>'
            f'<p class="rev">{rev_line or "UNBOUND — no version tag"}'
            + (f' · decklist {esc(sha)}' if sha else "") + "</p></div>")

    # The record of revisions, and the contents. Both are tables of fact.
    if revs:
        rows = "".join(
            f'<tr><td>{esc(r.get("rev"))}</td><td>{esc(r.get("applies_to"))}</td>'
            f'<td>{esc(r.get("date"))}</td>'
            f'<td>{esc(", ".join(r.get("changed") or []) or "—")}</td></tr>'
            for r in revs)
        record = ('<div class="poh-scroll"><table class="poh">'
                  "<tr><th>Rev</th><th>Applies to</th><th>Date</th>"
                  f"<th>Sections changed</th></tr>{rows}</table></div>")
    else:
        record = absent(
            "Record of revisions", "no manual_revisions.json — this handbook has "
            "not been revised, which is different from having been revised once.")
    return head, record


def render_contents(d, rendered):
    rows = "".join(
        f'<tr><td>{esc(spec.SECTION_BY_ID[sid]["number"])}</td>'
        f'<td>{spec.xref(spec.SECTION_BY_ID[sid]["number"], spec.SECTION_BY_ID[sid]["title"])}</td>'
        f'<td class="ev">{esc(spec.SECTION_BY_ID[sid]["source"])}</td></tr>'
        for sid in rendered)
    # THE CONTENTS LISTS WHAT RENDERED, never the full registry. A contents page
    # naming a section that is not in the book sends the reader to a page that
    # does not exist, which is exactly what a numbered cross-reference is for
    # avoiding.
    return ('<div class="poh-scroll"><table class="poh">'
            f"<tr><th>§</th><th>Section</th><th>Source</th></tr>{rows}</table></div>")


# ── 1. general ───────────────────────────────────────────────────────────

STAGE_LABEL = {
    "mana": "Mana", "ignition": "Ignition", "fuel": "Fuel", "fodder": "Fodder",
    "conversion": "Conversion", "output": "Output", "protection": "Protection",
    "wincon": "Win condition",
}


def schematic(engine):
    """Boxes and arrows, from `engine.json`. The spine the later sections cite.

    AN UNPROVEN ARROW IS DRAWN DASHED and the legend says what dashed means.
    `verified_by` is sparse — 4 of ur-dragon's 15 lines carry one — and
    `validate_engine`'s own docstring records two false-green arrows that
    shipped: a passing stack proves A BOARD RESOLVED THIS WAY, not that stage A
    feeds stage B. Drawing every arrow solid because the picture looks better is
    how that happens a third time.
    """
    if not engine:
        return ""
    stages = [s for s in (engine.get("stages") or [])]
    if not stages:
        return ""
    order = [s for s in spec.STAGE_ORDER if any(x["stage"] == s for x in stages)]
    idx = {s: i for i, s in enumerate(order)}
    by_stage = {s["stage"]: s for s in stages}

    W, H = 720, 90 + 62 * len(order)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="engine schematic">']
    y_of = lambda s: 46 + 62 * idx[s]
    for s in order:
        y = y_of(s)
        st = by_stage[s]
        n = len(st.get("cards") or [])
        parts.append(
            f'<rect x="12" y="{y - 20}" width="250" height="40" rx="3" '
            f'fill="none" stroke="#16161a" stroke-width="1.5"/>'
            f'<text x="26" y="{y - 2}" font-size="13" font-family="monospace">'
            f'{esc(STAGE_LABEL.get(s, s))}</text>'
            f'<text x="26" y="{y + 14}" font-size="10" fill="#6a665e">'
            f'{n} card(s)</text>')
    for line in (engine.get("lines") or []):
        a, b = line.get("from"), line.get("to")
        if a not in idx or b not in idx:
            continue
        ya, yb = y_of(a), y_of(b)
        proved = bool(line.get("verified_by"))
        dash = "" if proved else ' stroke-dasharray="5 4"'
        parts.append(
            f'<path d="M262 {ya} C 330 {ya}, 330 {yb}, 262 {yb}" fill="none" '
            f'stroke="#16161a" stroke-width="{1.6 if proved else 1}"{dash} '
            f'opacity="{0.9 if proved else 0.5}"/>')
        if proved:
            parts.append(
                f'<text x="336" y="{(ya + yb) / 2 + 4}" font-size="9" '
                f'font-family="monospace" fill="#6a665e">'
                f'✓ {esc(line["verified_by"])}</text>')
    parts.append("</svg>")

    n_proved = sum(1 for l in (engine.get("lines") or []) if l.get("verified_by"))
    n_lines = len(engine.get("lines") or [])
    legend = (f'<p class="poh-legend">Solid: proved by a checker-passed stack '
              f'({n_proved} of {n_lines}). Dashed: the analyst\'s reading, not '
              f'proved — a passing stack shows that a board resolved a certain '
              f'way, which is not the same as showing that one stage feeds '
              f'another.</p>')
    return f'<figure class="poh-schematic">{"".join(parts)}{legend}</figure>'


def render_general(d):
    e = d["engine"]
    if not e:
        return absent("General", "no engine.json — the schematic and the purpose "
                      "are read from the engine model.", f"/analyze-engine {d['slug']}")
    body = f'<p>{esc(e.get("thesis") or "")}</p>' + schematic(e)
    absent_stages = e.get("absent_stages") or []
    if absent_stages:
        body += sub("1.1", "Stages this deck does not have", "".join(
            f'<p><b>{esc(STAGE_LABEL.get(a["stage"], a["stage"]))}.</b> '
            f'{esc(a.get("why"))}</p>' for a in absent_stages))
    return body


# ── 2. limitations ───────────────────────────────────────────────────────

CLASS_NAME = {"creature": "creatures", "artifact": "artifacts",
              "enchantment": "enchantments", "land": "lands",
              "graveyard": "graveyards"}


def render_limitations(d):
    body = ""
    audit = d["audit"]
    if not audit:
        body += absent("What this deck cannot answer",
                       "no audit.json.", f"manamap pilot deck-audit {d['slug']} --write")
    else:
        axes = {a["axis"]: a for a in (audit.get("axes") or [])}
        ib = axes.get("interaction-breadth") or {}
        uncovered = (ib.get("measured") or {}).get("uncovered") or []
        if uncovered:
            names = ", ".join(CLASS_NAME.get(c, c) for c in uncovered)
            body += sub("2.1", "Permanent classes with no answer",
                        callout("warning",
                                f"This deck has no answer to {names}. "
                                f"Detected by oracle-text heuristic over the 99, "
                                f"with counterspells excluded — so this is a "
                                f"floor on breadth, not a ceiling.")
                        + f'<p class="ev">{esc((ib.get("measured") or {}).get("how") or "")}</p>')
        else:
            body += sub("2.1", "Permanent classes with no answer",
                        '<p>Every permanent class has at least one answer in the 99.</p>')

    # The operating envelope: bracket, clock, colour load. Three facts, each
    # from its own artifact, each with its definition.
    rows = []
    br = d["bracket"] or {}
    if br.get("floor") is not None:
        rows.append(("Bracket floor", f'{br["floor"]} — {br.get("floor_name", "")}',
                     "computed from the 99; the deck cannot be played below it"))
    gf = ((d["goldfish"] or {}).get("metrics") or {}).get("combat") or {}
    if gf.get("median_kill_turn") is not None:
        rows.append(("Goldfish kill turn", f'{gf["median_kill_turn"]} (median)',
                     "unopposed, no blockers — a floor on speed, never a verdict"))
    mana = (d["mana"] or {}).get("pips") or {}
    for colour in "WUBRG":
        p = mana.get(colour) or {}
        if p.get("total_pips"):
            rows.append((f"{colour} pips", str(p["total_pips"]),
                         f'earliest turn {p.get("earliest_turn", "—")}'))
    if rows:
        body += sub("2.2", "Operating envelope",
                    '<div class="poh-scroll"><table class="poh">'
                    "<tr><th>Limit</th><th>Value</th><th>Definition</th></tr>"
                    + "".join(f'<tr><td>{esc(a)}</td><td class="num">{esc(b)}</td>'
                              f'<td class="ev">{esc(c)}</td></tr>' for a, b, c in rows)
                    + "</table></div>")
    return body


# ── 5. performance ───────────────────────────────────────────────────────

def bar_chart(caption, series, unit, takeaway, ci=None):
    """One series, direct labels, units stated, no legend.

    Where a figure HAS an interval it is drawn. `diagnostic.engine.online_by_turn`
    carries ci95 at n=10,000, and a rate drawn without its interval is this
    repo's most-repeated mistake.
    """
    if not series:
        return ""
    keys = sorted(series, key=lambda k: (len(str(k)), str(k)))
    peak = max((v for v in series.values() if isinstance(v, (int, float))), default=0) or 1
    rows = ""
    for k in keys:
        v = series[k]
        if not isinstance(v, (int, float)):
            continue
        w = 100 * v / peak
        band = ""
        if ci and k in ci and ci[k]:
            lo, hi = ci[k]
            band = (f'<span class="ev"> [{lo:.2f}, {hi:.2f}]</span>')
        rows += (f'<tr><td>{esc(k)}</td>'
                 f'<td style="width:70%"><span style="display:inline-block;'
                 f'height:.55rem;background:#16161a;width:{w:.1f}%"></span></td>'
                 f'<td class="num">{v:.2f}{band}</td></tr>')
    return (f'<figure class="poh-chart"><figcaption>{esc(caption)} '
            f'<span class="ev">({esc(unit)})</span></figcaption>'
            f'<div class="poh-scroll"><table class="poh">{rows}</table></div>'
            f'<p class="take">{esc(takeaway)}</p></figure>')


def render_performance(d):
    gf = (d["goldfish"] or {}).get("metrics") or {}
    meta = (d["goldfish"] or {}).get("meta") or {}
    diag = d["diagnostic"] or {}
    if not gf:
        return absent("Performance", "no goldfish_metrics.json.",
                      f"manamap pilot goldfish {d['slug']}")
    n = gf.get("iterations")
    body = ""

    cmd = gf.get("commander") or {}
    if cmd.get("cast_turn_histogram"):
        body += sub("5.1", "Commander online", bar_chart(
            "Turn the commander is first cast", cmd["cast_turn_histogram"],
            f"count of {n:,} seeded games" if n else "count",
            f'Cast by turn six in {cmd.get("cast_by_turn_6_rate", 0) * 100:.0f}% of games. '
            f'Cast, not drawn.'))

    oh = gf.get("opening_hand") or {}
    if oh.get("first_seven_land_histogram"):
        body += sub("5.2", "Opening hands", bar_chart(
            "Lands in the opening seven", oh["first_seven_land_histogram"],
            "count", f'{oh.get("keep_first_seven_rate", 0) * 100:.0f}% of first '
                     f'sevens are keepable under the model rule: two to five '
                     f'lands, up to two redraws.'))

    if gf.get("mean_available_mana_by_turn"):
        body += sub("5.3", "Mana by turn", bar_chart(
            "Mean available mana", gf["mean_available_mana_by_turn"], "mana",
            "Unopposed. No land destruction, no taxes, no opponent."))

    eng = (diag.get("engine") or {}).get("online_by_turn") or {}
    if eng:
        series = {k: v.get("rate") for k, v in eng.items() if isinstance(v, dict)}
        ci = {k: v.get("ci95") for k, v in eng.items() if isinstance(v, dict)}
        body += sub("5.4", "Engine assembly", bar_chart(
            "Share of games with the engine online", series, "rate",
            "Intervals are 95% on the rate. Two overlapping intervals imply "
            "nothing about a difference.", ci=ci))

    combat = gf.get("combat") or {}
    if combat.get("kill_turn_histogram"):
        body += sub("5.5", "Time to kill", bar_chart(
            "Turn the goldfish kills", combat["kill_turn_histogram"], "count",
            f'Median turn {combat.get("median_kill_turn", "—")}. The median and '
            f'not the mean: a mean over a skewed sample is a true number '
            f'describing no game.'))
    elif combat == {}:
        body += sub("5.5", "Time to kill", absent(
            "Time to kill", "combat is opt-in and this deck has not been "
            "re-baselined for it."))

    if meta.get("model_assumptions"):
        body += ('<details><summary class="ev">What this model does not do '
                 f'({len(meta["model_assumptions"])} stated assumptions)</summary>'
                 "<ul>" + "".join(f"<li>{esc(a)}</li>"
                                  for a in meta["model_assumptions"]) + "</ul></details>")
    return body


def margin_figure(card, seen):
    """A thumbnail in the margin, on FIRST MENTION IN A SECTION and never again.

    The page this replaces carried 176-308 hidden full-card images — one per
    card mention — revealed on hover and explicitly hidden on mobile AND on
    paper. They were most of a 275 KB file and did nothing on either surface a
    handbook is read on.

    `art_crop` rather than the full card: it is a fraction of the bytes, it is
    the part a reader recognises, and the NAME is always present as text beside
    it, so the page degrades legibly with images off.
    """
    name = card.get("name")
    if not name or name in seen:
        return ""
    url = card.get("art_crop") or card.get("image")
    if not url:
        return ""
    seen.add(name)
    return (f'<figure class="poh-fig"><img src="{esc(url)}" alt="" loading="lazy">'
            f'<figcaption class="card">{esc(name)}</figcaption></figure>')


# ── 6. systems ───────────────────────────────────────────────────────────

def render_systems(d):
    e = d["engine"]
    if not e:
        return absent("Systems", "no engine.json.", f"/analyze-engine {d['slug']}")
    cards = {c["name"]: c for c in expand_copies(d["cards"].get("cards") or [])}
    seen_art = set()
    out = ""
    for i, st in enumerate(e.get("stages") or [], start=1):
        num = f"6.{i}"
        rows = ""
        for name in (st.get("cards") or []):
            c = cards.get(name) or {}
            rows += (f'<tr><td class="card">{esc(name)}</td>'
                     f'<td>{esc(c.get("mana_cost") or "")}</td>'
                     f'<td class="ev">{esc(c.get("type_line") or "")}</td></tr>')
        table = ('<div class="poh-scroll"><table class="poh">'
                 "<tr><th>Card</th><th>Cost</th><th>Type</th></tr>"
                 f"{rows}</table></div>") if rows else ""
        spof = st.get("single_point_of_failure")
        fail = (callout("caution", f"Single point of failure: {spof}")
                if spof else "")
        # ONE image per subsection, on the card the stage is named for. Not one
        # per mention — that is the failure this replaces.
        fig = ""
        for name in (st.get("cards") or []):
            fig = margin_figure(cards.get(name) or {}, seen_art)
            if fig:
                break
        out += sub(num, STAGE_LABEL.get(st["stage"], st["stage"]),
                   f'{fig}<p>{esc(st.get("what_it_does") or "")}</p>{fail}{table}')

    critic = e.get("critic") or {}
    if critic.get("verdict"):
        # BOXED, AND DATED BY THE VERSION IT WAS READ AGAINST — never by render
        # time, which would break the byte-identical rebuild.
        stamp = e.get("decklist_sha256_prefix")
        out += sub("6.0", "Assessment",
                   f'<div class="poh-call note"><span class="lbl">ASSESSMENT · '
                   f'ENGINE-CRITIC</span>Verdict: <b>{esc(critic["verdict"])}</b>. '
                   f'{len(critic.get("findings") or [])} finding(s), '
                   f'{len(critic.get("blind_spots") or [])} blind spot(s). '
                   f'Read against decklist {esc(stamp or "unstamped")}.</div>')
    return out


# ── 3. emergency procedures ──────────────────────────────────────────────

def _steps(items, ordered=True):
    """A checklist. NUMBERED where order matters, because in an emergency it
    always does — step three before step one is how a game is lost politely."""
    if not items:
        return ""
    tag = "ol" if ordered else "ul"
    return (f"<{tag}>" + "".join(f"<li>{esc(x)}</li>" for x in items)
            + f"</{tag}>")


def render_emergency(d):
    """One condition per page, one fixed template, tabbed in red.

    THE CONDITIONS ARE `deck_notes.CAUSES`, which is not decoration: a game
    logged `--cause wipe` and the page for a wipe are keyed the same, so a
    procedure can be read against the games that actually ended that way. Nine
    losses across seven causes exist on the fleet today.
    """
    proc = load_json(d["base"] / spec.PROCEDURES_ARTIFACT) or {}
    pages = proc.get("emergency") or []
    if not pages:
        return absent(
            "Emergency procedures",
            "not written yet. Drafted by an agent from the engine model, the "
            "diagnosis and the games that ended each way, then edited.",
            f"/poh-procedures {d['slug']}")
    out = ""
    for i, page in enumerate(pages, start=1):
        cond = page.get("condition")
        gloss = spec.EMERGENCY_CONDITIONS.get(cond, "")
        body = (f'<p class="ev">{esc(gloss)}</p>'
                f'<p><b>Condition.</b> {esc(page.get("condition_text") or gloss)}</p>')
        if page.get("indications"):
            body += ("<p><b>Indications.</b></p>"
                     + _steps(page["indications"], ordered=False))
        if page.get("immediate"):
            body += "<p><b>Immediate action.</b></p>" + _steps(page["immediate"])
        if page.get("subsequent"):
            body += "<p><b>Subsequent.</b></p>" + _steps(page["subsequent"])
        if page.get("notes"):
            body += callout("note", page["notes"])
        # WHICH GAMES THIS IS DRAWN FROM. A procedure grounded in a game the
        # pilot actually lost is worth more than one reasoned from the list, and
        # the reader should be able to tell which they are reading.
        seen = page.get("grounded_in") or []
        if seen:
            body += (f'<p class="ev">Drawn from {len(seen)} logged game(s): '
                     f'{esc(", ".join(seen))}.</p>')
        out += (f'<div class="poh-procedure">'
                + sub(f"3.{i}", (cond or "?").replace("-", " ").upper(), body)
                + "</div>")
    return out


# ── 4. normal procedures ─────────────────────────────────────────────────

def render_normal(d):
    proc = load_json(d["base"] / spec.PROCEDURES_ARTIFACT) or {}
    normal = proc.get("normal") or {}
    gf = ((d["goldfish"] or {}).get("metrics") or {}).get("opening_hand") or {}
    if not normal:
        return absent(
            "Normal procedures", "not written yet.", f"/poh-procedures {d['slug']}")
    out = ""
    for i, (key, title, gloss) in enumerate(spec.NORMAL_PHASES, start=1):
        block = normal.get(key)
        if not block:
            continue
        body = f'<p class="ev">{esc(gloss)}</p>'
        if isinstance(block, dict):
            if block.get("keep"):
                body += "<p><b>Keep if all of.</b></p>" + _steps(block["keep"], False)
            if block.get("ship"):
                body += "<p><b>Ship on sight.</b></p>" + _steps(block["ship"], False)
            if block.get("steps"):
                body += _steps(block["steps"])
            if block.get("note"):
                body += callout("note", block["note"])
        else:
            body += _steps(block)
        # THE MEASURED RATE BESIDE THE RULE, on the one phase that has one.
        if key == "preflight" and gf.get("keep_first_seven_rate") is not None:
            body += (f'<p class="ev">The model\'s own keep rule — two to five '
                     f'lands, up to two redraws — keeps '
                     f'{gf["keep_first_seven_rate"] * 100:.0f}% of first sevens. '
                     f'That is a floor: it does not read the cards.</p>')
        out += (f'<div class="poh-procedure">'
                + sub(f"4.{i}", title, body) + "</div>")
    return out


# ── 7. handling and rules of engagement ──────────────────────────────────

def render_handling(d):
    proc = load_json(d["base"] / spec.PROCEDURES_ARTIFACT) or {}
    h = proc.get("handling") or {}
    if not h:
        return absent("Handling", "not written yet.", f"/poh-procedures {d['slug']}")
    out = ""
    for i, (key, title) in enumerate((
            ("optics", "Threat optics — what the table sees"),
            ("reveal", "When to reveal"),
            ("alliances", "Alliances"),
            ("targets", "Who to hit first")), start=1):
        block = h.get(key)
        if not block:
            continue
        body = (_steps(block, ordered=False) if isinstance(block, list)
                else f"<p>{esc(block)}</p>")
        out += sub(f"7.{i}", title, body)

    # MEASURED, NOT ASSERTED. `interaction_received` is how much the pod aimed
    # at this seat per game — the sim's answer to "does the table take this deck
    # seriously", beside the pilot's reading of the same question.
    sim = _latest_sim(d)
    if sim:
        seat = (sim.get("analysis") or {}).get("seats", {}).get(d["slug"]) or {}
        recv = (seat.get("interaction_received") or {}).get("mean")
        elim = seat.get("eliminated_by") or {}
        if recv is not None:
            top = ", ".join(f"{k} {v}" for k, v in
                            sorted(elim.items(), key=lambda kv: -kv[1])[:3])
            out += sub("7.5", "What the pod actually did", (
                f'<p>Across {sim.get("games_completed", "?")} simulated games this '
                f'seat was aimed at {recv:.2f} times a game.'
                + (f' Eliminated by: {esc(top)}.' if top else "") + "</p>"
                f'<p class="ev">Forge\'s AI, not your table — opponent modelling, '
                f'never an equilibrium.</p>"'))
    return out


def _latest_sim(d):
    runs = sorted((d["base"] / "sim").glob("*.json")) if (d["base"] / "sim").is_dir() else []
    return load_json(runs[-1]) if runs else None


# ── assembly ─────────────────────────────────────────────────────────────

RENDERERS = {
    "general": render_general,
    "limitations": render_limitations,
    "emergency": render_emergency,
    "normal": render_normal,
    "handling": render_handling,
    "performance": render_performance,
    "systems": render_systems,
}


def render(slug):
    d = load(slug)
    changed = set()
    revs = d["revisions"].get("revisions") or []
    if revs:
        changed = set(revs[-1].get("changed") or [])

    head, record = render_front_matter(d)
    bodies, rendered = [], []
    for sid in spec.SECTION_IDS:
        fn = RENDERERS.get(sid)
        if not fn:
            continue
        html_body = section(sid, fn(d), changed)
        if html_body:
            bodies.append(html_body)
            rendered.append(sid)

    front = section("front-matter",
                    render_contents(d, rendered) + sub("0.1", "Record of revisions", record),
                    changed)
    title = f'{d["slug"]} — Pilot\'s Operating Handbook'
    from manamap.pilot import poh_design as pd
    return ("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{esc(title)}</title>"
            f'<link rel="stylesheet" href="poh.css?v={pd.stylesheet_version()}">'
            f'</head><body class="poh"><div class="poh-trim">{head}{front}'
            + "".join(bodies) + "</div></body></html>\n")


def main(args):
    from manamap.pilot import poh_design as pd

    slug = args.slug
    out_dir = MANUALS_DIR / ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.write_stylesheet(out_dir)
    path = out_dir / f"{slug}.html"
    path.write_text(render(slug), encoding="utf-8")
    size = path.stat().st_size
    print(f"Wrote {path}  ({size:,} bytes)")
    print(f"  next: manamap pilot validate-poh {slug}")
    return 0
