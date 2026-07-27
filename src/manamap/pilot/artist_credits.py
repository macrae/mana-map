"""Pilot: who painted this deck — standout artists and art themes.

Reads the printing metadata `fetch-deck` stores (artist, set, collector number,
border, frame effects, finishes, foil) and finds the story in it: an artist who
painted a disproportionate share, clusters of two or three, whether a Secret
Lair drop was bought whole, and how the treatments line up.

Computed on demand, never committed. Goldfish earns an artifact because it is
an expensive seeded simulation; counting 82 cards is neither, and a second copy
of facts already in cards.json could only ever desync. The numbers still earn
the data-derived badge: they are a pure function of a committed, reproducible
artifact.

Counting is **per entry, never per copy**. "Painted a third of your deck" is
technically true of goblin-storm and rhetorically dishonest — 22 of those 37
copies are one basic-land art. Copies are reported as their own labeled fact.
"""

import json
from collections import Counter, defaultdict

from manamap.pilot.common import deck_dir, load_deck_cards

# An artist leads the department only if the lead is real, not noise.
STANDOUT_MIN_ENTRIES = 3
STANDOUT_RATIO = 2.0        # at least twice the runner-up...
STANDOUT_SHARE = 0.15       # ...or a clear share of the whole deck
CLUSTER_MIN_ENTRIES = 2
DROP_RUN_MIN = 3            # contiguous collector numbers worth calling a drop

# Secret Lair accessories (Red Mana, Storm Counter) have no rules text and no
# real art credit role; they are listed, never counted.
ACCESSORY_TYPE_LINE = "Card"


def is_accessory(card):
    return card.get("type_line", "") == ACCESSORY_TYPE_LINE


def countable(cards):
    """Cards that count toward artist tallies: real cards with an artist."""
    return [c for c in cards if c.get("artist") and not is_accessory(c)]


def _entry_sort_key(card):
    return str(card.get("name", ""))


def rank_artists(cards):
    """Artists by entry count, descending; ties broken by name for determinism."""
    by_artist = defaultdict(list)
    for card in countable(cards):
        by_artist[card["artist"]].append(card)
    ranking = []
    for artist, group in by_artist.items():
        group = sorted(group, key=_entry_sort_key)
        ranking.append({
            "artist": artist,
            "entries": len(group),
            "copies": sum(int(c.get("quantity", 1)) for c in group),
            "cards": [c["name"] for c in group],
        })
    ranking.sort(key=lambda r: (-r["entries"], r["artist"]))
    return ranking


def find_standout(ranking, total_entries):
    """The lead artist, if the lead is real. None means: tell a breadth story."""
    if not ranking or total_entries == 0:
        return None
    top = ranking[0]
    runner_up = ranking[1]["entries"] if len(ranking) > 1 else 0
    share = top["entries"] / total_entries
    if top["entries"] < STANDOUT_MIN_ENTRIES:
        return None
    if not (top["entries"] >= STANDOUT_RATIO * runner_up or share >= STANDOUT_SHARE):
        return None
    return {
        "artist": top["artist"],
        "entries": top["entries"],
        "copies": top["copies"],
        "share": round(share, 4),
        "runner_up": runner_up,
        "cards": top["cards"],
    }


def find_clusters(ranking, standout):
    """Artists with more than one card — the 'also worth noting' material."""
    lead = (standout or {}).get("artist")
    return [r for r in ranking
            if r["entries"] >= CLUSTER_MIN_ENTRIES and r["artist"] != lead]


def roster_overlap(cards, roster):
    """How each top artist maps onto the deck's load-bearing groups.

    `roster` is the issue plan's `the-99` grouping: [{"role", "cards": [...]}].
    The interesting finding is usually concentration in one group and absence
    from another.
    """
    if not roster:
        return []
    artist_of = {c["name"]: c.get("artist") for c in cards}
    out = []
    for group in roster:
        names = [n for n in group.get("cards", []) if n in artist_of]
        if not names:
            continue
        counts = Counter(artist_of[n] for n in names if artist_of[n])
        if not counts:
            continue
        top = max(counts.values())
        artist = sorted(a for a, n in counts.items() if n == top)[0]  # deterministic tie-break
        out.append({
            "group": group.get("role", "Roster"),
            "of": len(names),
            "artist": artist,
            "painted": top,
            "cards": sorted(n for n in names if artist_of[n] == artist),
            "distinct_artists": len(counts),
        })
    return out


def treatments(cards):
    """Border, foil and frame-effect distribution — the physical look."""
    real = [c for c in cards if not is_accessory(c)]
    frames = Counter()
    for card in real:
        for effect in card.get("frame_effects") or []:
            frames[effect] += 1
    return {
        "entries": len(real),
        "borderless": sum(1 for c in real if c.get("border_color") == "borderless"),
        "foil": sum(1 for c in real if c.get("foil")),
        "frame_effects": dict(sorted(frames.items())),
    }


def set_spread(cards):
    """Per set: how many cards, and how many distinct artists (dispersion)."""
    by_set = defaultdict(list)
    for card in countable(cards):
        by_set[(card.get("set"), card.get("set_name"))].append(card)
    out = [
        {
            "set": key[0], "set_name": key[1], "entries": len(group),
            "artists": len({c["artist"] for c in group}),
        }
        for key, group in by_set.items()
    ]
    out.sort(key=lambda s: (-s["entries"], str(s["set"])))
    return out


def drop_runs(cards):
    """Contiguous collector-number runs by one artist in one set.

    A run means a drop was bought whole rather than assembled card by card —
    which is exactly the kind of thing a magazine should say out loud instead
    of implying curation. Only all-digit collector numbers qualify (The List
    uses forms like 'LCI-132').
    """
    buckets = defaultdict(list)
    for card in countable(cards):
        number = str(card.get("collector_number") or "")
        if number.isdigit():
            buckets[(card.get("set"), card["artist"])].append((int(number), card["name"]))
    runs = []
    for (set_code, artist), items in buckets.items():
        items.sort()
        start = prev = None
        run = []
        for number, name in items:
            if prev is not None and number == prev + 1:
                run.append((number, name))
            else:
                if len(run) >= DROP_RUN_MIN:
                    runs.append((set_code, artist, start, run))
                start, run = number, [(number, name)]
            prev = number
        if len(run) >= DROP_RUN_MIN:
            runs.append((set_code, artist, start, run))
    out = [
        {"set": set_code, "artist": artist, "from": str(run[0][0]), "to": str(run[-1][0]),
         "entries": len(run), "cards": [name for _, name in run]}
        for set_code, artist, _start, run in runs
    ]
    out.sort(key=lambda r: (str(r["set"]), r["artist"], r["from"]))
    return out


def build_notes(cards, ranking, standout):
    """Honesty caveats that travel with the numbers into the editor's context."""
    notes = []
    multiples = [c for c in countable(cards) if int(c.get("quantity", 1)) > 1]
    for card in sorted(multiples, key=_entry_sort_key):
        notes.append(
            f'{card["name"]} is one entry but {card["quantity"]} copies — counted once. '
            f"Per-entry is the honest basis; per-copy would inflate "
            f'{card.get("artist")}.'
        )
    accessories = [c for c in cards if is_accessory(c) and c.get("artist")]
    if accessories:
        names = ", ".join(sorted(c["name"] for c in accessories))
        notes.append(f"Excluded from counts (no rules text, table aids): {names}.")
    if standout:
        spread = set_spread(cards)
        dispersed = [s for s in spread if s["entries"] and s["artists"] / s["entries"] > 0.7]
        if dispersed and len(spread) > 1:
            notes.append(
                "Concentration may be structural rather than curated: the rest of the "
                "deck is near-total artist dispersion, so check whether the lead comes "
                "from one product bought whole before implying taste."
            )
    if not standout and ranking:
        notes.append(
            f"No standout: the lead artist has {ranking[0]['entries']} card(s). "
            f"Tell a breadth story instead."
        )
    return notes


def analyze(cards, roster=None):
    """Full artist analysis for a deck. Deterministic for fixed input."""
    counted = countable(cards)
    ranking = rank_artists(cards)
    total_entries = len(counted)
    standout = find_standout(ranking, total_entries)
    return {
        "totals": {
            "entries": total_entries,
            "copies": sum(int(c.get("quantity", 1)) for c in counted),
            "artists": len(ranking),
            "accessories": sum(1 for c in cards if is_accessory(c)),
        },
        "ranking": ranking,
        "standout": standout,
        "clusters": find_clusters(ranking, standout),
        "roster_overlap": roster_overlap(cards, roster),
        "treatments": treatments(cards),
        "sets": set_spread(cards),
        "drop_runs": drop_runs(cards),
        "notes": build_notes(cards, ranking, standout),
    }


def load_roster(slug):
    """The issue plan's `the-99` roster, when one exists."""
    path = deck_dir(slug) / "issue_plan.json"
    if not path.exists():
        return None
    with open(path) as f:
        plan = json.load(f)
    for dept in plan.get("departments", []):
        if dept.get("id") == "the-99":
            return dept.get("roster")
    return None


def main(args):
    cards = load_deck_cards(args.slug)["cards"]
    result = analyze(cards, load_roster(args.slug))

    if getattr(args, "as_json", False):
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    totals = result["totals"]
    print(f"{totals['artists']} artists across {totals['entries']} cards "
          f"({totals['copies']} copies)")
    standout = result["standout"]
    if standout:
        print(f"\nSTANDOUT  {standout['artist']} — {standout['entries']} cards "
              f"({standout['share']:.1%}), runner-up {standout['runner_up']}")
        for name in standout["cards"]:
            print(f"    {name}")
    else:
        print("\nNo standout artist — tell a breadth story.")
    if result["clusters"]:
        print("\nAlso worth noting:")
        for cluster in result["clusters"][:6]:
            print(f"    {cluster['artist']} ({cluster['entries']}): "
                  f"{', '.join(cluster['cards'])}")
    if result["roster_overlap"]:
        print("\nAgainst the roster:")
        for row in result["roster_overlap"]:
            print(f"    {row['group']}: {row['artist']} painted {row['painted']} of "
                  f"{row['of']} ({row['distinct_artists']} artists in the group)")
    treat = result["treatments"]
    print(f"\nTreatments: {treat['borderless']} borderless, {treat['foil']} foil, "
          f"frame effects {treat['frame_effects'] or '(none)'}")
    for run in result["drop_runs"]:
        print(f"    Drop run: {run['set']} #{run['from']}–{run['to']} "
              f"({run['entries']} cards, {run['artist']})")
    for note in result["notes"]:
        print(f"  NOTE {note}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot artist-credits <slug>`.")
