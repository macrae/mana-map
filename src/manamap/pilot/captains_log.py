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

#: ONE SECTION PER NIGHT. It was six — header, situation, narrative, assessment,
#: orders, coda — dictated in the register of a starship captain, with a stardate
#: the pilot could not read and stations answering to an officer who does not
#: exist. What is actually happening is one person playing a deck of cards.
#:
#: A night now gets a short plain paragraph, and the synthesis that was missing
#: entirely moved to `READ_KEYS` below.
SECTION_KEYS = ("summary",)

#: THE READ — the deck-level roll-up, and the reason this artifact exists at all.
#:
#: Eleven rendered nights across five decks, and nothing had ever synthesised
#: ACROSS them: every entry described one evening and no artifact answered "what
#: is this deck, learned from playing it". Four other artifacts answer that
#: question from the DECKLIST (`engine.json:thesis`, `manual_prose.how_it_wins`,
#: `strategic_frame`, `diagnosis.verdict`). This one answers it from the GAMES,
#: and cites them — which is the whole of its justification for existing beside
#: them.
READ_KEYS = ("what_it_does", "how_it_plays", "mindful_of", "what_changed")

#: A night holds the PILOT's account. `personal` is reserved and nothing mints
#: it. The reserved key is why the deterministic facts sit OUTSIDE `logs` — two
#: kinds would share one version and two copies of a fact is how they come to
#: disagree.
#:
#: This said `ship` until the register was retired. There is no ship; there is a
#: person and a deck of cards, and every reader of `logs.ship` moves with it.
LOG_KINDS = ("pilot", "personal")

# THE STATIONS, ATTRIBUTION_ORDER AND STATION_ROLES WERE DELETED HERE.
#
# They named an engineering officer, a tactical officer and an order of blame —
# self, then ship, then circumstance — for a register that no longer exists.
# There is one person and a deck of cards.
#
# `stations_for_deck` went with them, and it was already dead: its docstring
# promised "the validator holds it to this roster" and NO CODE EVER CALLED IT.
# The equivalent guard is real in `validate_debrief`. Carrying a tested,
# documented, unreferenced function is worse than not having one — it reads as
# a working check.

#: A game logged at 01:30 belongs to the night before. Commander runs late.
NIGHT_CUTOFF_HOUR = 4

#: THE STARDATE IS NO LONGER RENDERED — the pilot could not read it, which was
#: the point of it going. `stardate()` and these constants stay because the
#: night grouping is derived from the same wall-clock parsing and the function
#: is exercised by tests that pin that behaviour. It is now an internal ordering
#: aid rather than something a header quotes.
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


def read(slug):
    return load_json(DECKS_DIR / slug / ARTIFACT) or {}


def read_meta(slug):
    """WHICH GAMES THE READ MAY REST ON, and which list each was played against.

    The read is supposed to describe the deck AS IT IS NOW, so it needs to know
    which of its games are about the current 99 and which are about a list that
    has been superseded. That join already exists — `deck_versions.report` maps
    every log entry's `decklist_sha256` to a version — and `deck_info.compose`
    computes it and throws it away. This keeps it.

    `current` is the flag the charter's history rule turns on: a game on the
    current list needs no caveat, and an older one earns its place only when the
    lesson still applies and says which version it came from.
    """
    from manamap.pilot import deck_notes, deck_versions

    try:
        report = deck_versions.report(slug)
    except Exception:                              # noqa: BLE001 - never block
        report = {}
    current = report.get("current_version")
    by_sha = {}
    for version in report.get("versions") or []:
        for sha in version.get("decklist_sha256s") or []:
            by_sha[sha] = version.get("version")
    games, spanned = [], []
    for entry in deck_notes.read_log(slug):
        version = by_sha.get(entry.get("decklist_sha256"))
        games.append({"id": entry["id"], "at": entry.get("at", "")[:10],
                      "version": version, "current": version == current})
        if version and version not in spanned:
            spanned.append(version)
    return {"as_of_version": current,
            "as_of_sha": report.get("working_decklist_sha256"),
            "games_considered": games,
            "versions_spanned": spanned,
            # A game whose sha matches no committed version. Named rather than
            # dropped: `deck-notes --at` stamps the CURRENT decklist on a
            # backfilled game, so an entry can be honestly logged and still
            # join to the wrong list.
            "unmatched": [g["id"] for g in games if g["version"] is None]}


def skeleton(slug):
    """The whole deterministic document, prose-free."""
    return {"slug": slug, "commander": _ship(slug),
            "read_meta": read_meta(slug), "nights": nights(slug)}


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
        done = "pilot" in ((rendered.get(key) or {}).get("logs") or {})
        print(f"  stardate {night['stardate']}  {key}  "
              f"{night['version'] or 'unversioned'}  "
              f"{'rendered' if done else 'NOT YET RENDERED'}")
        print(f"    entries {', '.join(night['source_ids'])}{where}")
    missing = [k for k in doc["nights"]
               if "pilot" not in ((rendered.get(k) or {}).get("logs") or {})]
    if missing:
        print(f"\n{len(missing)} night(s) not yet rendered — /captains-log {slug}")
    return 0
