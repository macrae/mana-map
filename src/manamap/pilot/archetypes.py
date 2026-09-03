"""How a commander is actually built, and what each style needs. PRD-v1 §7.2-7.3.

    manamap pilot archetypes "Zur the Enchanter"
    manamap pilot archetypes "Zur the Enchanter" --theme voltron

**THE ARCHETYPES ARE DATA, NOT RESEARCH.** §7.2 describes an agent that goes and
reads about a commander and reports the distinct styles. EDHREC's `taglinks`
panel already is that list, with a deck count on each, and it answers Zur with
exactly the split PRD-v1 names as its worked example — Enchantress 1201, Auras
736, Stax 542, Control 529, Combo 380, Voltron 361. An agent would have
paraphrased this, more expensively and less precisely.

**STYLES ARE PRESENTED SIDE BY SIDE AND NOT RANKED.** §7.2 is explicit: win and
play rates may be shown as data; the platform does not tell the pilot which deck
to want. So EDHREC's own order is preserved — re-sorting would be inventing a
ranking — the count travels so the order means something, and nothing here calls
one style better than another. There is no "recommended".

**THE ROLE TEMPLATE IS DERIVED FROM THE STYLE'S OWN DECK.** §7.3 wants "the role
list that style needs". Rather than authoring one, this reads the role histogram
of that archetype's average deck through `card_roles.json` — the same taxonomy
the deck page and the builder use.

That fixes a defect this repo already had on the record. `DECK_ROLE_BUDGET` is
ONE FLAT BUDGET for every deck, its own comment calls it "PROVISIONAL", and
`upgrade_facts` printed its shortfalls as "Context, not evidence". A stax deck
and a voltron deck do not want the same eight numbers, and now they do not get
them: the template is measured from the decks people actually built in that
style, and it says how many decks that was.

**IT PROPOSES; IT DOES NOT BUILD.** §7.3: "The agent proposes; the user builds."
This emits a template and candidate roles. `build-deck` remains the one thing
that turns a brief into a 99, and it remains the one scorer.
"""

import json

from manamap import console
from manamap.config import CARD_ROLES_PATH
from manamap.ingest import edhrec

#: Roles that describe a card rather than a job, and would swamp any histogram.
#: `ROLE_BODY_FALLBACK` labels every creature — the same reason training excludes
#: it when mining positives.
NOISE_ROLES = frozenset({"threat:body"})

#: A style with fewer decks behind it than this is reported but not templated.
#: A role histogram over eleven decks is a description of eleven decks.
MIN_DECKS_FOR_TEMPLATE = 50


def _roles():
    doc = json.loads(CARD_ROLES_PATH.read_text(encoding="utf-8"))
    return doc.get("roles") or {}


def list_themes(commander, limit=None):
    """The styles this commander is built in, in EDHREC's order.

    `limit=None` means all of them — the caller decides what to SHOW, and the
    full list is what a theme is resolved against.
    """
    themes = edhrec.themes(commander)
    return themes if limit is None else themes[:limit]


def role_template(commander, theme, roles=None):
    """The role histogram of one style's average deck. §7.3.

    Counts COPIES, not entries, so a deck's thirty basics do not read as one
    land — the same rule `mana_analysis` keeps and the one that once published
    "18 lands" for a 33-land deck.

    A card carries several roles and every one of them is counted here, which is
    deliberate and different from the deck page's "paint it with one". A budget
    is about coverage — how much removal does this style want — and a card that
    both ramps and draws genuinely serves both, whereas a picture has to choose
    a colour.
    """
    roles = roles if roles is not None else _roles()
    deck = edhrec.average_deck(commander, theme=theme)

    histogram = {}
    unroled = 0
    for name, qty in deck["cards"]:
        card_roles = [r for r in (roles.get(name) or []) if r not in NOISE_ROLES]
        if not card_roles:
            unroled += int(qty)
            continue
        for r in card_roles:
            histogram[r] = histogram.get(r, 0) + int(qty)

    total = sum(int(q) for _, q in deck["cards"])
    return {
        "commander": deck["commander"] or commander,
        "theme": theme,
        "deck_size": total,
        "unroled": unroled,
        "roles": dict(sorted(histogram.items(), key=lambda kv: -kv[1])),
    }


def distinguishing(template, baseline):
    """What this style wants MORE and LESS of than the commander's baseline.

    MEASURED, and it is why this function exists. The raw role histograms of
    Zur's styles are **0.955 to 0.978 cosine similar** — voltron, stax and
    enchantress all run the same signets, the same lands, much the same removal,
    and that shared bulk swamps the part that differs. Printing the histogram
    alone shows a reader mostly what every Zur deck runs.

    The differences are real once the baseline is subtracted: voltron takes
    `buff:attached` from 8 to 14 while stax takes `stax` from 3 to 6. So the
    template is reported WITH its delta, and the delta is the half worth acting
    on — it is the answer to "what does this style need that the others do not".
    """
    keys = set(template) | set(baseline)
    delta = {k: template.get(k, 0) - baseline.get(k, 0) for k in keys}
    return dict(sorted((kv for kv in delta.items() if kv[1]),
                       key=lambda kv: -abs(kv[1])))


def report(commander, theme=None, limit=12):
    """Everything the command prints, as data."""
    # `limit` is a DISPLAY concern and must not decide what exists. Resolving the
    # requested theme against the truncated list made `--theme voltron --limit 3`
    # report that Zur has no voltron decks, which is both false and confidently
    # phrased — a display flag silently changing a lookup's answer.
    all_themes = list_themes(commander, limit=None)
    out = {"commander": commander, "themes": all_themes[:limit],
           "note": ("Deck counts are EDHREC play rates, shown as data. They are "
                    "not a ranking and nothing here recommends a style.")}
    if theme:
        match = next((t for t in all_themes if t["slug"] == theme), None)
        if match is None:
            raise SystemExit(
                f"{commander} has no EDHREC theme {theme!r} — "
                f"known: {', '.join(t['slug'] for t in all_themes[:8])}")
        if match["decks"] < MIN_DECKS_FOR_TEMPLATE:
            out["warning"] = (
                f"{match['name']} has {match['decks']} decks behind it; a role "
                f"template over that is a description of {match['decks']} decks, "
                f"not of a style")
        with console.task(f"Reading {match['name']} decks", total=None) as t:
            roles = _roles()
            t.state("fetching the style's average deck")
            out["template"] = role_template(commander, theme, roles)
            t.state("and the commander's baseline, to subtract it")
            base = role_template(commander, None, roles)
            out["baseline_deck_size"] = base["deck_size"]
            out["distinguishing"] = distinguishing(out["template"]["roles"],
                                                   base["roles"])
    return out


def format_report(doc):
    lines = [f"\nARCHETYPES — {doc['commander']}\n"]
    lines.append(f"  {'decks':>7}  style")
    lines.append("  " + "-" * 50)
    for t in doc["themes"]:
        lines.append(f"  {t['decks']:>7}  {t['name']}  ({t['slug']})")
    lines.append("")
    lines.append(f"  {doc['note']}")

    tpl = doc.get("template")
    if tpl:
        if doc.get("warning"):
            lines.append(f"\n  ⚠ {doc['warning']}")
        lines.append(f"\nROLE TEMPLATE — {tpl['theme']}, from a {tpl['deck_size']}-card "
                     f"average deck")
        lines.append(f"  {'copies':>7}  role")
        lines.append("  " + "-" * 50)
        for role, n in list(tpl["roles"].items())[:18]:
            lines.append(f"  {n:>7}  {role}")
        if tpl["unroled"]:
            lines.append(f"  {tpl['unroled']:>7}  (no role in card_roles.json)")
        d = doc.get("distinguishing") or {}
        if d:
            lines.append(f"\n  WHAT THIS STYLE WANTS THAT THE OTHERS DO NOT")
            lines.append(f"  (against this commander's overall average deck — the raw")
            lines.append(f"   histograms of its styles run 0.96+ cosine similar, so the")
            lines.append(f"   delta is the half worth acting on)")
            for role, n in list(d.items())[:10]:
                lines.append(f"  {n:>+7}  {role}")
        lines.append("")
        lines.append("  Candidates for any of these: "
                     "`manamap pilot card-search --identity <ID> --role <role>`")
    return "\n".join(lines)


def main(args):
    doc = report(args.commander, theme=getattr(args, "theme", None),
                 limit=getattr(args, "limit", 12))
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2))
    else:
        print(format_report(doc))
