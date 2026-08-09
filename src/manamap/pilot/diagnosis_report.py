"""Pilot: render a deck diagnosis as something a human reads.

`diagnosis.json` runs to 77 KB on a deck with eight verified stacks, and until
this module existed there was no way to look at it. That is a real gap and not a
cosmetic one: the artifact's whole value is the argument it makes, and an
argument nobody can read is an argument nobody can disagree with.

Markdown, deliberately, because it serves three readers from one renderer — a
terminal, a published page, and a diff. Deterministic and free, like every other
renderer in this subsystem: it reads the committed artifact and nothing else. It
does **not** re-derive figures (`validate-diagnosis` does that, and doing it in
two places is how they drift), so a report is only as fresh as the artifact it
renders — which is why the header prints the decklist hash the diagnosis was
built against.

The ordering is the argument, not the schema: the verdict, then what binds, then
the engine, then what to do, then what is still unsettled, and the skeptic's
findings LAST and in full. Burying an adversary's open findings under the
recommendations they qualify would make this a sales document.
"""

from manamap.pilot.common import deck_dir, load_json, resolve_out_path

# Reading order for the axis table: the questions in the order the doctor's
# charter asks them, so the table narrates rather than lists. Anything not named
# here still prints, after these, in artifact order.
AXIS_ORDER = [
    "mana-base", "mana-sources", "colour-sources", "taplands", "consistency",
    "curve", "ramp", "card-advantage", "threat-density", "creatures",
    "interaction", "interaction-breadth", "sweepers", "protection",
    "tutors", "power",
]

VERDICT_MARK = {
    "strength": "++", "adequate": "ok", "weakness": "--", "liability": "!!",
}
DIFFICULTY_MARK = {"easy": "easy", "contested": "CONTESTED", "painful": "PAINFUL"}


def _cite(citations):
    """Citations as a compact trailing note. IDs only — the quote is in the JSON."""
    ids = [c.get("rule") for c in (citations or []) if c.get("rule")]
    return f"  \n  <sub>cites {', '.join(ids)}</sub>" if ids else ""


def _axis_rows(axes):
    order = {name: i for i, name in enumerate(AXIS_ORDER)}
    return sorted(axes, key=lambda a: (order.get(a["axis"], len(order)), a["axis"]))


def render(doc):
    slug = doc.get("slug", "?")
    out = [f"# {slug} — deck diagnosis", ""]

    skeptic = doc.get("skeptic") or {}
    if skeptic:
        statuses = [f.get("status") for f in skeptic.get("findings") or []]
        supported = sum(1 for s in statuses if s == "supported")
        out += [
            f"**Skeptic verdict: `{skeptic.get('verdict', '?')}`** — "
            f"{supported} supported, {len(statuses) - supported} open, after "
            f"{skeptic.get('iterations', '?')} iteration(s).",
            "",
        ]
        if skeptic.get("verdict") == "fail":
            out += [
                "> A `fail` diagnosis is saved on purpose: it documents what could "
                "not be grounded. Read the open findings at the bottom before "
                "acting on anything above them.",
                "",
            ]
    out += [
        f"<sub>Against decklist `{str(doc.get('as_of_decklist_sha256'))[:12]}`. "
        f"Measurements are ◆ deterministic and re-derived by `validate-diagnosis`; "
        f"every verdict, ranking and prescription is ★ judgment.</sub>",
        "",
        "## The verdict", "",
        doc.get("verdict", "—"), "",
    ]
    if doc.get("archetype"):
        out += ["**Archetype.** " + doc["archetype"], ""]

    # ── Axes ─────────────────────────────────────────────────────────
    axes = doc.get("axes") or []
    out += ["## The axes", "",
            "| | Axis | Measured | Reading |", "|---|---|---|---|"]
    for a in _axis_rows(axes):
        m = a.get("measured") or {}
        value = m.get("value")
        unit = m.get("unit", "")
        reading = str(a.get("reading", "")).replace("\n", " ").replace("|", "\\|")
        out.append(
            f"| `{VERDICT_MARK.get(a.get('verdict'), '??')}` | **{a['axis']}** | "
            f"{value} {unit} | {reading} |")
    out.append("")
    notable = [a for a in axes if a.get("verdict") in ("weakness", "liability")]
    if notable:
        out += ["**Called out as a weakness or liability:** "
                + ", ".join(f"`{a['axis']}`" for a in notable), ""]

    # ── Engine ───────────────────────────────────────────────────────
    engine = doc.get("engine") or {}
    if engine:
        out += ["## The engine", ""]
        if engine.get("declared"):
            out += [engine["declared"], ""]
        comps = engine.get("components") or []
        if comps:
            out += ["| Component | Have | Rate | Reading |", "|---|---|---|---|"]
            for c in comps:
                rate = c.get("measured_rate")
                rate = f"{rate:.1%}" if isinstance(rate, (int, float)) else "—"
                have = ", ".join((c.get("have") or [])[:6])
                if len(c.get("have") or []) > 6:
                    have += f" … (+{len(c['have']) - 6})"
                thin = " ⚠" if c.get("thinnest") else ""
                reading = str(c.get("reading", "")).replace("\n", " ").replace("|", "\\|")
                out.append(f"| **{c.get('role', '?')}**{thin} | {c.get('count', '?')} "
                           f"— {have} | {rate} | {reading} |")
            out.append("")
        spfs = engine.get("single_points_of_failure") or []
        if spfs:
            out += ["### Single points of failure", ""]
            for s in spfs:
                closers = s.get("closers") or []
                out += [f"**{s.get('component', '?')}** — {s.get('why', '')}"
                        + _cite(s.get("citations")), ""]
                out += [("  *Closers:* " + ", ".join(closers)) if closers
                        else "  *No closer found.*", ""]

    # ── Prescription ─────────────────────────────────────────────────
    lean = doc.get("lean_into") or []
    if lean:
        out += ["## Lean into", ""]
        for item in lean:
            out += [f"- **{item.get('what', '?')}** — {item.get('why', '')}"
                    + _cite(item.get("citations"))]
        out.append("")

    adds = doc.get("add_candidates") or []
    if adds:
        out += ["## Add", "",
                "| Card | Closes | Source | Natural cut | Bracket | Why |",
                "|---|---|---|---|---|---|"]
        for a in adds:
            bd = a.get("bracket_delta") or {}
            bracket = (f"{bd.get('before')}→{bd.get('after')}"
                       if bd else "—")
            why = str(a.get("why", "")).replace("\n", " ").replace("|", "\\|")
            out.append(
                f"| **{a.get('card')}** | {a.get('closes', '—')} | "
                f"{a.get('source', '—')} | {a.get('natural_cut') or '—'} | "
                f"{bracket} | {why} |")
        out.append("")

    cuts = doc.get("cut_candidates") or []
    if cuts:
        out += ["## Cut", "",
                "Ranked hardest-last. `orphans_stack` is **computed** by the "
                "validator, not claimed — a cut that would strand a "
                "checker-passed line cannot be filed quietly.", ""]
        rank = {"easy": 0, "contested": 1, "painful": 2}
        for c in sorted(cuts, key=lambda c: rank.get(c.get("difficulty"), 9)):
            orphans = c.get("orphans_stack") or []
            out += [
                f"### {c.get('card')} — `{DIFFICULTY_MARK.get(c.get('difficulty'), '?')}`",
                "",
                f"{c.get('why', '')}",
                "",
                f"**Cost of cutting.** {c.get('cost_of_cutting', '—')}",
            ]
            if orphans:
                out.append(f"**Touches verified stack(s):** {', '.join(orphans)}")
            out += [_cite(c.get("citations")).strip(), ""]

    # ── Unsettled ────────────────────────────────────────────────────
    questions = doc.get("open_questions") or []
    if questions:
        out += ["## Open questions", "",
                "Each routes to a skill. This is the work the diagnosis hands "
                "back rather than guessing at.", ""]
        for q in questions:
            out += [f"- **`{q.get('settled_by')}`** — {q.get('question')}  ",
                    f"  *Why it matters:* {q.get('why_it_matters', '')}"]
        out.append("")

    gaps = doc.get("gaps") or []
    if gaps:
        out += ["## Gaps", "",
                "What could not be grounded, in the doctor's own words.", ""]
        out += [f"- {g}" for g in gaps]
        out.append("")

    # ── The adversary, last and in full ──────────────────────────────
    findings = skeptic.get("findings") or []
    if findings:
        out += ["## Skeptic findings", "",
                "The adversarial pass, unabridged. `supported` means the skeptic "
                "checked the claim and it held.", ""]
        openers = [f for f in findings if f.get("status") != "supported"]
        held = [f for f in findings if f.get("status") == "supported"]
        for label, group in (("Open", openers), ("Confirmed", held)):
            if not group:
                continue
            out += [f"### {label}", ""]
            for f in group:
                out += [f"- **`{f.get('status')}`** at `{f.get('where')}` — "
                        f"{f.get('note', '')}"]
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def main(args):
    base = deck_dir(args.slug)
    path = base / "diagnosis.json"
    doc = load_json(path, default=None)
    if doc is None:
        raise SystemExit(
            f"{path} not found — run the diagnose-deck skill for {args.slug} first.")
    text = render(doc)
    out = getattr(args, "out", None)
    if out:
        target = resolve_out_path(out, args.slug, "diagnosis-report", ext=".md")
        with open(target, "w") as f:
            f.write(text)
        print(f"Wrote {target}")
    else:
        print(text)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot diagnosis-report <slug>`.")
