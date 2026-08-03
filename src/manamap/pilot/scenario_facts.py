"""Pilot: the deterministic brief for a stack scenario.

Every other deck agent starts from `deck-facts`. The two that did not are
`stack-resolver` and `rules-checker` — the most expensive pair per artifact in
the repo (one line reached ~600k tokens) and the only two that read `cards.json`
raw and re-derive everything from prose.

The Vol. 008 session made the cost of that gap legible. Five errors reached agent
briefs, every one a figure recalled instead of derived:

  * "28 from each opponent" when the verified stacks say **7 per opponent** and 28
    is the POD TOTAL across four. The decision-spread agent refused the premise
    rather than write to it.
  * stacks 002 and 003 described as sharing a board when 002 carries a fourth
    body (the Human Soldier that Bastion's own ETB makes, because an enchantment
    cannot be one). Two rounds of stack 008 went to untangling that.
  * a card named as in the 99 that had left the deck before the session began.

None of those is a judgement failure. All three are lookups, and this module does
them. It also answers the question the checkers kept having to answer in prose:
*which sibling scenarios are actually comparable to this one?*

**Computed on demand, never committed** — same rule as `deck_facts` and
`artist_credits`. The output is a view; a second copy of facts already in the
artifacts could only ever desync.

Everything here reads existing artifacts. No new analysis:

    deck membership   pilot/common.py     load_deck_cards, mainboard
    board parsing     this module         board_bodies (scenario shape only)
"""

import json
import re
from collections import Counter

from manamap.pilot.common import deck_dir, load_deck_cards, mainboard

# What a board entry IS, from how the scenario annotates it.
#
# The first version of this split treated tokens as furniture, on the reasoning
# that they are not cards. That is exactly backwards for these scenarios: in a
# sacrifice deck the tokens ARE the bodies, and the body count is what every
# engine here is bounded by. It reported yawgmoth's stack 002 and 003 as having
# the same three bodies — erasing the extra Human Soldier that is the entire
# reason their identical-looking totals are not comparable, which is the error
# this command exists to prevent. Corrected: a creature is anything the scenario
# gives a power/toughness, token or not.
_PT = re.compile(r"\b\d+\s*/\s*\d+\b")
_LAND_WORDS = re.compile(
    r"\b(plains|island|swamp|mountain|forest|wastes|land|lands)\b", re.I)
_NONCREATURE = re.compile(r"\b(enchantment|artifact|planeswalker|battle)\b", re.I)

# The house annotation for a permanent whose sacrifice already paid a cost. It is
# LISTED on the board and is NOT on the battlefield — the single most consequential
# reading in this repo's scenarios, and the one a resolver had to argue for twice
# inside resolution prose because nothing else recorded it.
_ALREADY_PAID = re.compile(r"already sacrificed|already paid|cost already", re.I)


def _strip_annotation(entry):
    """The card name a board entry names, before parentheticals and em-dashes."""
    name = str(entry or "")
    name = name.split("—")[0]
    name = re.sub(r"\s*\([^)]*\)", "", name)
    return name.strip()


def board_bodies(entries):
    """Split a board list into creature bodies, other permanents, lands, and spent.

    `spent` is the annotated cost payment: still listed, NOT on the battlefield.
    Getting that wrong changes the body count, and the body count is what every
    engine in this deck family is bounded by — so it is reported separately rather
    than folded into either side.
    """
    bodies, others, lands, spent = [], [], [], []
    for entry in entries or []:
        raw = str(entry)
        name = _strip_annotation(entry)
        if not name:
            continue
        if _ALREADY_PAID.search(raw):
            spent.append(name)
        elif _NONCREATURE.search(raw) and not _PT.search(raw):
            others.append(name)
        elif _PT.search(raw):
            bodies.append(name)          # tokens included: they are bodies
        elif _LAND_WORDS.search(name) and len(name.split()) <= 3:
            lands.append(name)
        else:
            others.append(name)
    return {"creature_bodies": bodies, "other_permanents": others,
            "lands": lands, "spent_paying_a_cost": spent}


def opponents_of(scenario):
    """Opponent life totals, whichever board shape the scenario uses.

    Two shapes exist in the corpus: `opponents: [{life, board}]` on seven decks
    and `opponent_a..d` bare lists on yawgmoth-swarm. Reading only one silently
    reports zero opponents on the other.
    """
    board = scenario.get("board") or {}
    extras = scenario.get("extras") or {}
    lives = (extras.get("life_totals") or {})

    out = []
    listed = board.get("opponents")
    if isinstance(listed, list):
        for i, opp in enumerate(listed):
            if isinstance(opp, dict):
                out.append({"seat": opp.get("name") or f"opponent_{i + 1}",
                            "life": opp.get("life")})
    # `opponents` itself starts with "opponent" — matching on the prefix alone
    # invented a phantom seat with no life total on every deck using the list shape.
    for key in sorted(k for k in board if k.startswith("opponent_")):
        out.append({"seat": key, "life": lives.get(key)})
    for key, life in sorted(lives.items()):
        if key.startswith("opponent") and not any(o["seat"] == key for o in out):
            out.append({"seat": key, "life": life})
    seen, unique = set(), []
    for o in out:
        if o["seat"] not in seen:
            seen.add(o["seat"])
            unique.append(o)
    return unique


def drain_arithmetic(opponents):
    """The per-opponent / pod-total distinction, stated so it cannot be conflated.

    This is the exact error that reached a brief in the Vol. 008 session: a drain
    that removes 7 from each of four opponents is 7 PER OPPONENT and 28 ACROSS THE
    POD. Quoting the pod total as a per-seat figure overstates a kill by 4x.
    """
    n = len(opponents)
    lives = [o["life"] for o in opponents if isinstance(o.get("life"), int)]
    return {
        "opponents": n,
        "opposing_life_total": sum(lives) if lives else None,
        "per_opponent_life": lives or None,
        "note": (
            f"{n} opponent(s). A drain of X 'each opponent' removes X per seat and "
            f"{n}*X across the pod. Quote the per-seat figure and the pod total "
            f"separately — they are never interchangeable."
        ),
    }


def membership(names, deck_names):
    """Which named cards are actually in this deck's 99 right now."""
    present = sorted(n for n in names if n in deck_names)
    absent = sorted(n for n in names if n not in deck_names)
    return {"in_the_deck": present, "NOT_IN_THE_DECK": absent}


def comparable_siblings(this_id, all_scenarios):
    """Sibling scenarios whose board is genuinely like-for-like with this one.

    Checkers kept reconciling this in prose because nothing computed it — one
    artifact spent ~400 words explaining that a sibling's headline figure was a
    different quantity on a different board. Two boards are comparable when their
    acting bodies match as a multiset; a differing count is exactly the "extra
    body" that made two stacks' identical-looking totals incomparable.
    """
    mine = all_scenarios.get(this_id)
    if not mine:
        return []
    my_bodies = Counter(board_bodies((mine.get("board") or {}).get("you"))["creature_bodies"])
    out = []
    for sid, sc in sorted(all_scenarios.items()):
        if sid == this_id:
            continue
        theirs = Counter(board_bodies((sc.get("board") or {}).get("you"))["creature_bodies"])
        shared = sum((my_bodies & theirs).values())
        same_count = sum(my_bodies.values()) == sum(theirs.values())
        only_theirs = sorted((theirs - my_bodies).elements())
        only_mine = sorted((my_bodies - theirs).elements())
        # Both directions, always. A one-sided diff hid the case this exists for:
        # yawgmoth 002 and 003 have the SAME body count, and 002 reaches it with a
        # Human Soldier token where 003 uses Zulaport — because Bastion is an
        # enchantment and cannot be a body itself. Reporting only what the sibling
        # adds makes two boards look interchangeable when their composition is the
        # whole reason their totals answer different questions.
        out.append({
            "stack": sid,
            "body_count": sum(theirs.values()),
            "same_body_count": same_count,
            "only_on_that_board": only_theirs or None,
            "only_on_this_board": only_mine or None,
            "note": ("same body count, but check the composition below before "
                     "quoting its totals" if same_count else
                     "DIFFERENT body count — do NOT quote its totals against this board"),
            "shared_bodies": shared,
        })
    return out


def analyze(slug, stack_id=None):
    base = deck_dir(slug)
    doc = load_deck_cards(slug)
    deck_names = {c["name"] for c in mainboard(doc["cards"])}

    scenarios = {}
    for path in sorted((base / "stacks").glob("*.json")):
        stack = json.loads(path.read_text())
        sid = str(stack.get("id") or path.name[:3])
        scenarios[sid] = stack.get("scenario") or {}

    out = {"slug": slug, "stacks": {}}
    for sid, sc in scenarios.items():
        if stack_id and sid != stack_id:
            continue
        you = board_bodies((sc.get("board") or {}).get("you"))
        opps = opponents_of(sc)
        named = you["creature_bodies"] + you["other_permanents"] + you["spent_paying_a_cost"]
        out["stacks"][sid] = {
            "your_board": you,
            "opponents": opps,
            "drain_arithmetic": drain_arithmetic(opps),
            "card_membership": membership(named, deck_names),
            "mana_available": sc.get("mana_available"),
            "hand": sc.get("hand"),
            "comparable_siblings": comparable_siblings(sid, scenarios),
        }
    out["notes"] = _notes(out)
    return out


def _notes(facts):
    notes = []
    absent = sorted({n for s in facts["stacks"].values()
                     for n in s["card_membership"]["NOT_IN_THE_DECK"]})
    if absent:
        notes.append(
            f"{len(absent)} card(s) named on a board are NOT in the maindeck: "
            f"{', '.join(absent)}. A scenario may legitimately name an opponent's "
            f"permanent, but a card of YOURS that left the deck makes the line "
            f"unreachable — check before citing it as playable.")
    spent = {sid: s["your_board"]["spent_paying_a_cost"]
             for sid, s in facts["stacks"].items() if s["your_board"]["spent_paying_a_cost"]}
    if spent:
        notes.append(
            "Boards annotated with an already-paid cost: "
            + "; ".join(f"{sid}: {', '.join(v)}" for sid, v in sorted(spent.items()))
            + ". Those permanents are LISTED but NOT on the battlefield. The body "
              "count entering the resolution excludes them, and every bound in this "
              "deck family is a body count.")
    incomparable = {sid: [c["stack"] for c in s["comparable_siblings"]
                          if not c["same_body_count"]]
                    for sid, s in facts["stacks"].items()}
    incomparable = {k: v for k, v in incomparable.items() if v}
    if incomparable:
        notes.append(
            "Sibling boards that are NOT like-for-like: "
            + "; ".join(f"{k} vs {', '.join(v)}" for k, v in sorted(incomparable.items()))
            + ". Their totals answer a different question and must never be quoted "
              "against each other without saying what differs.")
    notes.append(
        "Every figure here is derived from the scenario blocks on disk. Prefer it "
        "to recall: the five brief errors this command exists to prevent were all "
        "correct-sounding numbers remembered rather than looked up.")
    return notes


def main(args):
    facts = analyze(args.slug, getattr(args, "stack", None))
    text = json.dumps(facts, indent=2, sort_keys=True)
    print(text)
    out = getattr(args, "out", None)
    if out:
        from pathlib import Path
        Path(out).write_text(text + "\n")
        print(f"\nWrote {out}")
