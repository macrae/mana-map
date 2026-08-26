"""Pilot: a STARTING `goldfish_targets.json`, derived and clearly marked as one.

    manamap pilot scaffold-targets <slug>

`goldfish_targets.json` is the deck's engine DECLARATION — its `any_of` groups
are the components and a group's SIZE is that component's redundancy, priced
hypergeometrically by `deck_audit` and quoted by every diagnosis downstream. It
is authored, and it should stay authored: a component is a claim about what this
deck is trying to do, and nothing in this repo can read that off a card list.

WHAT THIS FIXES IS NOT THE AUTHORING, IT IS THE BLANK PAGE. A new deck's dossier
said `AUTHORED: data/decks/<slug>/goldfish_targets.json` and stopped there — a
path, a schema nobody has seen, and two panels (goldfish, mana) blocked behind
it. The pilot's first act was to go and invent a JSON shape from nothing. This
writes a real file with the right shape and honest starting content, so the work
is EDITING a draft rather than composing one.

THE DANGER IS EXACTLY THE ONE `DECK_ROLE_BUDGET` FELL INTO, and it is designed
around rather than hoped away. That constant was one flat budget for every deck,
its own comment called it PROVISIONAL, and it sat unfixed long enough that
`upgrade_facts` printed its shortfalls as "Context, not evidence". A scaffolded
declaration nobody edits is the same failure with worse consequences, because
the goldfish would then report assembly rates for generic role buckets while
every reader believes it is measuring the engine. So:

  * the file says `"scaffolded": true` and `validate-goldfish-targets` REPORTS
    that — an unedited draft is visible, not invisible;
  * every group carries `_from`, naming the signal it came from, so a reader can
    tell a measured group from a guessed one at a glance;
  * `_note` says in the file itself that the labels are placeholders.

WHAT IT DERIVES, in descending order of how much the evidence carries:

  1. CONTAINED COMBO LINES (`combo_details`, never the co-occurrence graph).
     A combo is a real interaction between named cards, and the schema
     represents it exactly: a combo needing A and B is one target with two
     `any_of` legs of one card each. This is the only part that is genuinely a
     claim about the deck rather than about the taxonomy.

  2. ROLE AXES — ramp, draw, removal, wincon, tutor. Coarse, and the real
     declarations do carry groups of this shape ("RAMP drawn", "A TUTOR
     drawn"). But `validate_goldfish_targets` records, from a fleet-wide
     prototype, that a role axis is NOT what a goldfish group is: it fired
     hardest on the most correct groups, because `ROLE_PATTERNS` answers "what
     job in a 99" while a component is a deck-specific functional set with no
     taxonomy axis. So these are offered as scaffolding to replace, and the
     `_from` field says `role:<axis>` so nobody mistakes one for a finding.

WHAT IT DELIBERATELY DOES NOT DO: name a win line. That is the defect this whole
area exists to catch — heliod's Hullbreaker Horror and ur-dragon's Aggravated
Assault are each named in two passing stacks and in no component, so the
simulator never measured how those decks actually win. A machine that guessed at
one would be manufacturing exactly the claim the validator is there to demand.
The note says so, and points at the command that checks.
"""

import json

from manamap.pilot.common import deck_dir, expand_copies, load_deck_cards, load_json
from manamap.pilot.deck_facts import combo_facts, role_facts

TARGETS_FILE = "goldfish_targets.json"

#: Role axes worth a starting group, and the label each gets. Ordered, because
#: the file is read top to bottom by a person. `wincon` is included as a GROUP
#: (what the deck's finishers are) and never as a LINE (how they combine) — the
#: second is the judgement, and it is the pilot's.
ROLE_AXES = [
    ("ramp", "RAMP drawn — an accelerant of any kind"),
    ("draw", "CARD ADVANTAGE drawn — a draw engine or a refill"),
    ("removal", "AN ANSWER drawn — interaction of any kind"),
    ("tutor", "A TUTOR drawn"),
    ("wincon", "A FINISHER drawn — the cards that end games"),
]

#: A group LARGER than every group any human has authored is a category, not a
#: component — priced as redundancy it comes back at ~100% and says nothing.
#:
#: MEASURED, and the first version of this constant was the opposite check and
#: was wrong. Across 113 authored groups on 10 tracked decks: median 5, p90 10,
#: **max 20**. So 21 is "bigger than anything anyone has declared", not a taste.
#:
#: What was deleted: a THIN_GROUP warning at 3. **31 of those 113 groups are
#: under 3, and 22 are exactly 1** — a size-1 group is a deliberate declaration
#: that a component has no backup, which radagast's own file spells out
#: ("SELVALA — the single best engine card ... no functional backup exists in
#: the 99"). Warning about them would have fired on a quarter of the correct
#: data, which is the failure this repo has already recorded twice and once
#: committed. Run a proposed check against the whole fleet before keeping it.
BROAD_GROUP = 21

_NOTE = (
    "SCAFFOLD — not a declaration. `manamap pilot scaffold-targets` derived these "
    "groups from contained combo lines and role axes; every label is a "
    "PLACEHOLDER and every group is a starting point to edit, not a finding. A "
    "goldfish group is a deck-specific functional set (\"the metronome bodies\", "
    "\"the flash traps\"), which is a different thing from a role: `ROLE_PATTERNS` "
    "answers what job a card does in any 99. Nothing here names a WIN LINE, "
    "because a machine guessing at one manufactures the exact claim "
    "`validate-goldfish-targets` exists to demand. Rewrite the labels, regroup the "
    "members, add the line this deck actually wins with, then run "
    "`manamap pilot validate-goldfish-targets <slug>` and delete \"scaffolded\"."
)


def _role_axis(role):
    """`ramp:rock` -> `ramp`. The axis is the part before the colon."""
    return role.split(":", 1)[0]


def derive(slug):
    """The proposed targets, with each group's provenance attached."""
    doc = load_deck_cards(slug)
    cards = doc.get("cards", [])
    names = [c["name"] for c in expand_copies(cards)]
    commanders = [c["name"] for c in cards if c.get("is_commander")]
    # A commander is on every board from turn one, so it says nothing about what
    # a draw assembles — the same exclusion `validate_goldfish_targets` makes.
    drawable = sorted({n for n in names if n not in commanders})

    targets, broad = [], []

    combos = combo_facts(drawable, commanders)
    for line in (combos.get("lines") or []):
        parts = [c for c in line["cards"] if c not in commanders]
        if len(parts) < 2:
            continue          # a one-card "combo" is the commander plus a card
        targets.append({
            "label": "COMBO LINE ASSEMBLED: " + " + ".join(parts),
            "_from": "combo_details",
            "need": [{"any_of": [c]} for c in parts],
        })

    roles = role_facts(drawable)
    if roles.get("available"):
        from manamap.pilot.common import load_card_roles

        table = load_card_roles()
        for axis, label in ROLE_AXES:
            members = sorted({n for n in drawable
                              if any(_role_axis(r) == axis for r in table.get(n, []))})
            if not members:
                continue
            if len(members) >= BROAD_GROUP:
                broad.append(f"{axis} ({len(members)})")
            targets.append({
                "label": label,
                "_from": f"role:{axis}",
                "need": [{"any_of": members}],
            })

    return {"targets": targets, "broad": broad,
            "combos": len([t for t in targets if t["_from"] == "combo_details"])}


def scaffold(slug, force=False):
    path = deck_dir(slug) / TARGETS_FILE
    if path.exists() and not force:
        existing = load_json(path) or {}
        # REFUSING IS THE POINT. This file is authored, and an author's version
        # is the one thing here that cannot be regenerated. `--force` exists for
        # overwriting a scaffold that was never edited.
        raise SystemExit(
            f"{path} already exists"
            + (" (still a scaffold — `--force` to redraw it)"
               if existing.get("scaffolded") else
               " and is AUTHORED — `--force` would overwrite work no command can rebuild"))

    got = derive(slug)
    doc = {
        "_note": _NOTE,
        "scaffolded": True,
        "targets": got["targets"],
    }
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return {"path": str(path), "targets": len(got["targets"]),
            "from_combos": got["combos"], "broad": got["broad"]}


def main(args):
    out = scaffold(args.slug, force=getattr(args, "force", False))
    print(f"Wrote {out['path']}")
    print(f"  {out['targets']} starting target(s), {out['from_combos']} from real "
          f"combo lines and the rest from role axes")
    if out["broad"]:
        # A role axis wider than any authored group is a CATEGORY, not a
        # component: priced as redundancy it returns ~100% and measures nothing.
        # Said out loud, because it is the group most in need of splitting.
        print(f"  BROAD — wider than any authored group ({BROAD_GROUP}+): "
              f"{', '.join(out['broad'])}; split these first")
    print("  Every label is a placeholder. Rewrite them, add the line this deck")
    print("  actually wins with, drop \"scaffolded\", then:")
    print(f"      manamap pilot validate-goldfish-targets {args.slug}")
