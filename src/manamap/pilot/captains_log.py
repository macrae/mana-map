"""The captain's log — a LANGUAGE LAYER over the pilot's own notes.

The pilot writes 300-600 words after a game: venue and pod up front, what
happened, a lesson in capitals, a verdict. That note is the only thing on the
deck page written by somebody who was actually at the table, and it is authored,
append-only and never rewritten. This module does not touch it.

What it does is compute the SKELETON of a rendered log — which games make up a
night, what stardate that night carries, which version was sleeved, where the
night sat in an evening that spanned several decks — so that an agent has only
prose left to write. That split is the whole design:

    THE MORE OF THE ARTIFACT IS DETERMINISTIC, THE MORE OF IT A VALIDATOR CAN
    HOLD TO ACCOUNT WITHOUT JUDGING A WORD OF THE PROSE.

`validate_captains_log` recomputes everything here and compares. A stardate the
agent invented, a night it grouped by feel, a game it filed twice — all of that
is caught by arithmetic rather than by reading.

THIS IS THE ONE ARTIFACT IN THE REPO THAT DOES NOT GO STALE WHEN THE DECK
CHANGES. Every other agent output — `manual_prose.json`, `engine.json`,
`diagnosis.json` — describes the deck as it stands and rots the moment a card is
swapped. A log records a night that happened, and a swap on Tuesday does not make
Saturday's log wrong. Hence: no `deck_status.STAGES` row, no freshness stamp, and
`cards:semantic` deliberately absent from the cache routine's inputs.
"""

import json
from datetime import datetime, timedelta

from manamap.config import DECKS_DIR
from manamap.pilot.common import load_json
from manamap.pilot.deck_notes import causes, read_log

ARTIFACT = "captains_log.json"

#: The prose sections, in the order Picard dictates them. A log is not valid
#: with five of them (see `validate_captains_log`): a short log frozen as a cache
#: HIT renders short forever with every check still green.
SECTION_KEYS = ("header", "situation", "narrative", "assessment", "orders", "coda")

#: A night can hold a SHIP's log and, later, a PERSONAL one. Only `ship` is
#: minted today. The reserved key is why the deterministic facts sit OUTSIDE
#: `logs` — the two kinds share a stardate, and two copies of one fact is how
#: they come to disagree. Adding `personal` is one entry in this set: no
#: migration, no reader change.
LOG_KINDS = ("ship", "personal")

#: WHO IS RESPONSIBLE, in the order the captain assigns it: himself, then the
#: ship, then circumstance, and never reversed. Pushing this out of prose and
#: into structure is what turns the pilot's hardest style rule into a two-line
#: check with no judgment in it.
ATTRIBUTION_ORDER = ("self", "ship", "circumstance")

#: THE STATIONS. Officers are named by post, not by card type — the log refers to
#: Engineering's report rather than reproducing it, which is the whole reason the
#: jargon stays out of the prose.
#:
#: `helm` is the fourth because the other three leave the deck's WIN ROUTE with
#: nowhere to file, and half of these notes are about exactly that ("Edgar was
#: the only vampire I cast", "swung lethal into Alex"). A closed vocabulary must
#: be complete enough to use, or the agent misfiles under the nearest station and
#: the count silently means nothing — the same argument `deck_notes.CAUSES` makes.
#:
#: There is deliberately NO "Command" station for pilot error. `attribution:
#: "self"` already carries it, and an order to Command would have to read "I
#: will…", which breaks the rule that orders are stated as already issued.
STATIONS = {
    "engineering": "the mana base — lands, rocks, dorks, rituals, treasure",
    "tactical":    "interaction — removal, counterspells, protection, hate",
    "ops":         "card flow — draw, selection, tutors, recursion",
    "helm":        "the win route — the commander, the threats, the finishers",
}

#: Which `card_roles.json` role prefixes answer to which station.
#:
#: WRITTEN AGAINST THE ACTUAL VOCABULARY, which is 52 tags long and was read
#: before this was written down. The first cut was written against a GUESS at it
#: — inventing prefixes `mana` and `selection` that no card in the corpus
#: carries — and every station came back empty on every deck. A station nothing
#: answers to is a word the agent cannot use, and the test below exists because
#: that failure was silent.
#:
#: Measured across all ten decks with these prefixes: unassigned runs 0 to 12
#: cards out of ~99, and every deck fills every station. A card may answer to
#: SEVERAL stations — Edgar Markov carries four `buff:`/`payoff:`/`threat:` roles
#: — so membership is a set, never a value.
STATION_ROLES = {
    "engineering": ("ramp:", "land:"),
    "tactical":    ("removal:", "counterspell", "protection:", "stax", "hate:"),
    "ops":         ("draw:", "tutor:", "recursion", "value:etb"),
    "helm":        ("wincon:", "threat:", "buff:", "payoff:", "doubler:",
                    "sac-outlet"),
}

#: Roles that answer to NO station, listed so the omission is a decision rather
#: than an oversight. `utility:activated` sits on 4,514 cards and says only that
#: a permanent has an ability; `sac-cost` describes a cost, not a job. Filing
#: either would make its station mean nothing.
UNSTATIONED_ROLES = ("utility:activated", "sac-cost")

#: A game logged at 01:30 belongs to the night before. Commander runs late.
NIGHT_CUTOFF_HOUR = 4

STARDATE_EPOCH = 80000
STARDATE_EPOCH_YEAR = 2026


def _dt(at):
    """Parse a log entry's `at` AS LOCAL WALL-CLOCK TIME, and do not normalise.

    Five of the eleven entries on disk are naive (`2026-08-25T22:00`) and four
    carry an offset (`2026-09-01T21:30:00-07:00`). Converting to UTC would move
    edgar's 21:30-07:00 game to 2 September — shifting its night key and its
    stardate — while leaving the naive half exactly where it was. The fleet would
    then be split down the middle by a property of how the note was typed rather
    than by when the game was played.

    What the pilot means by "the night of the first" is the wall clock in the
    room, so that is what is read. `fromisoformat` gives it directly.
    """
    return datetime.fromisoformat(str(at))


def night_key(at):
    """The date a game belongs to. The night is keyed on DATE and nothing else.

    Tags cannot key anything and never could: the drift is already on disk —
    `pod3` against `pod-5`, `olivers-house` against `olivers`, with `alexs-house`
    and `orinda` both on the same four games. That is the failure
    `deck_notes.CAUSES` exists to prevent for causes, and nothing enforces it for
    tags.
    """
    dt = _dt(at)
    if dt.hour < NIGHT_CUTOFF_HOUR:
        dt -= timedelta(days=1)
    return dt.date().isoformat()


def stardate(at):
    """`at` -> a TNG-form stardate. Deterministic, local, and floored.

    `80000 + (year - 2026) * 1000 + day-of-year`, with the decimal being the
    fraction of the day elapsed. 1 September 2026 is day 244, which reproduces
    the integer part of the pilot's own example (`80244.6`) exactly; the decimal
    is the time of day, so a 21:30 game reads `.8`.

    FLOORED, never rounded: rounding 23:59 up would carry the decimal to `.10`
    or, worse, silently advance the day part of a number the header quotes.
    """
    dt = _dt(at)
    doy = dt.timetuple().tm_yday
    frac = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0
    whole = STARDATE_EPOCH + (dt.year - STARDATE_EPOCH_YEAR) * 1000 + doy
    return f"{whole}.{int(frac * 10)}"


def _ship(slug):
    """The deck's commander — the ship's name. Absent rather than guessed."""
    cards = load_json(DECKS_DIR / slug / "cards.json") or {}
    for c in cards.get("cards", []):
        if c.get("is_commander"):
            return c.get("name")
    return None


def _version_of(slug, sha):
    """Which release the deck was, at the sha the note was stamped with.

    Read from the TRACKED `deck_versions.json` tags rather than derived: the
    version string is a fact about the night, and an agent quoting one from
    memory is how prose comes to coach about a card that left the 99.
    """
    doc = load_json(DECKS_DIR / slug / "deck_versions.json") or {}
    for name, tag in sorted((doc.get("tags") or {}).items()):
        if tag.get("decklist_sha256") == sha or tag.get("sha") == sha:
            return name
    return None


def evening(at_date):
    """WHICH SHIPS FLEW THAT NIGHT, across the whole fleet, in order.

    The pilot flies a DIFFERENT DECK each game — 1 September at Oliver's is
    goblin-storm at 19:00, ur-dragon at 20:15, edgar at 21:30 — so a night is a
    fleet-wide event that each deck sees one slice of. The pilot already writes
    this by hand ("Game three of four on the night", heliod 001), which is the
    evidence that it belongs in the record.

    Without it, four logs from one evening each open by restating the same pod as
    though it were the only game played.
    """
    out = []
    for deck in sorted(DECKS_DIR.iterdir()):
        if not deck.is_dir() or not (deck / "log.jsonl").exists():
            continue
        for e in read_log(deck.name):
            if night_key(e["at"]) == at_date:
                out.append({"slug": deck.name, "at": e["at"], "id": e["id"],
                            "result": e.get("result")})
    return sorted(out, key=lambda g: _dt(g["at"]))


def nights(slug):
    """THE SKELETON: every fact about this deck's logged nights, computed.

    Returns `{night_key: {...}}` with no prose in it at all. `merge_captains_log`
    takes this wholesale and lets the agent fill only `logs[kind]`, so the agent
    cannot smuggle a stardate — or a grouping — past the merge.
    """
    entries = read_log(slug)
    if not entries:
        return {}
    filed = causes(slug)
    grouped = {}
    for e in entries:
        grouped.setdefault(night_key(e["at"]), []).append(e)

    out = {}
    for key in sorted(grouped):
        games = sorted(grouped[key], key=lambda e: _dt(e["at"]))
        fleet = evening(key)
        mine = next((i for i, g in enumerate(fleet)
                     if g["slug"] == slug and g["id"] == games[0]["id"]), None)
        first = games[0]
        out[key] = {
            "night": key,
            "stardate": stardate(first["at"]),
            "version": _version_of(slug, first.get("decklist_sha256")),
            "decklist_sha256": first.get("decklist_sha256"),
            "source_ids": [e["id"] for e in games],
            # WHERE THIS SAT IN THE EVENING. `after` names the ship flown
            # immediately before, which is what lets a Situation place itself
            # instead of four logs opening with the same sentence.
            "position_in_evening": None if mine is None else {
                "n": mine + 1, "of": len(fleet),
                "after": fleet[mine - 1]["slug"] if mine > 0 else None,
            },
            "games": [{
                "id": e["id"], "at": e["at"], "result": e.get("result"),
                "cause": (filed.get(e["id"]) or {}).get("cause"),
                "opponents": e.get("opponents"),
                # The index the SUPPLEMENTAL carries: 0 is the main log, 1+ are
                # appended mid-session. Zero exist on the fleet today — the pilot
                # has never played one deck twice in a night — so this path is
                # exercised by a fixture, never by real data.
                "supplemental_index": i,
            } for i, e in enumerate(games)],
            "logs": {},
        }
    return out


def stations_for_deck(slug):
    """Which cards answer to which station, from `card_roles.json`.

    THE AGENT MAPS NOTHING. It picks which stations a night's story involves and
    writes orders addressed to them; if it names a card while doing so, the
    validator holds it to this roster. Same move as `validate_debrief`'s "the
    debrief may not name a card the pilot did not", one turn further out.
    """
    from manamap.pilot.common import expand_copies
    from manamap.pilot.deck_facts import load_card_roles

    roles = load_card_roles()
    cards = load_json(DECKS_DIR / slug / "cards.json") or {}
    names = sorted({c.get("name") for c in expand_copies(cards.get("cards", []))
                    if c.get("name")})

    out = {k: [] for k in STATIONS}
    out["unassigned"] = []
    for name in names:
        card_roles = roles.get(name) or []
        hit = False
        for station, prefixes in STATION_ROLES.items():
            if any(str(r).startswith(p) for r in card_roles for p in prefixes):
                out[station].append(name)
                hit = True
        if not hit:
            out["unassigned"].append(name)
    return out


def read(slug):
    return load_json(DECKS_DIR / slug / ARTIFACT) or {}


def skeleton(slug):
    """The whole deterministic document, prose-free."""
    return {"slug": slug, "ship": _ship(slug), "nights": nights(slug)}


def main(args):
    slug = args.slug
    doc = skeleton(slug)
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2, ensure_ascii=False))
        return 0
    if not doc["nights"]:
        print(f"{slug}: nothing in the captain's log — "
              f'`manamap pilot deck-notes {slug} add "…"` first')
        return 0
    rendered = read(slug).get("nights") or {}
    print(f"CAPTAIN'S LOG — {slug} ({doc['ship'] or 'ship unknown'})")
    for key, night in doc["nights"].items():
        pos = night["position_in_evening"] or {}
        where = (f"  game {pos['n']} of {pos['of']} that night"
                 + (f", after {pos['after']}" if pos.get("after") else "")
                 if pos else "")
        done = "ship" in ((rendered.get(key) or {}).get("logs") or {})
        print(f"  stardate {night['stardate']}  {key}  "
              f"{night['version'] or 'unversioned'}  "
              f"{'rendered' if done else 'NOT YET RENDERED'}")
        print(f"    entries {', '.join(night['source_ids'])}{where}")
    missing = [k for k in doc["nights"]
               if "ship" not in ((rendered.get(k) or {}).get("logs") or {})]
    if missing:
        print(f"\n{len(missing)} night(s) not yet rendered — /captains-log {slug}")
    return 0
