"""A pod is a table you can name. PRD §6 B-2.

The standard pod has been a SENTENCE — `docs/simulation.md` says which three
decks it is — and three `--vs` flags typed by hand every time. `grep` for
`STANDARD_POD` finds `STANDARD_POD_PROFILE`, an AI profile, and no roster
anywhere. So the most load-bearing configuration in the simulation layer lived
in shell history, and the only way to know whether two runs faced the same table
was to read their run ids.

A pod file makes it an artifact: tracked, diffable, and carrying the two things
B-2 asks for that a slug list cannot — each seat's ARCHETYPE and BRACKET, so a
result can be reported by pod composition rather than only in aggregate.

    manamap pilot simulate <slug> --pod standard --games 100

**`--pod standard` MUST produce the same run id as the three `--vs` flags it
replaces.** The id is built from the ordered opponent slugs, so as long as a pod
expands to the same ordered list it is the same measurement — and the test
asserts exactly that, because a convenience that silently re-bases every record
would be worse than typing the flags.

## Per-seat AI profiles, and why the tag changes shape

B-2 asks for an AI strategy profile per opponent deck. The engine seam already
existed: Forge's `-a` is index-aligned per seat and `forge.command()` has always
taken a list. What did not exist was any way to reach it — `--vs-profile`
collapses to one name for every opponent.

A pod seat may now carry its own `profile`. When they all agree (the usual case,
and every pod that ships) nothing changes: the run id carries `-podExperimental`
exactly as before, and every record already on disk keeps its name. When they
differ the id carries `-podMixed<8hex>` over the sorted seat/profile pairs,
because two tables that play differently must not share a path — the same
silent-overwrite `profile_tag` was written for.
"""

import hashlib
import json
import pathlib

from manamap.config import DATA_DIR

#: Where pods live. Tracked: a pod is a claim about which table a result was
#: measured against, and that has to survive a fresh clone.
PODS_DIR = DATA_DIR / "pods"

#: A seat's keys. `slug` resolves through `forge.seat_dir`, so a pod may name an
#: opponent deck OR one of your own decks — which is what makes a mirror or a
#: "my three decks against each other" table expressible without new machinery.
SEAT_KEYS = frozenset({"slug", "archetype", "bracket", "profile", "note"})


class PodError(ValueError):
    """A pod that cannot be resolved, with a sentence saying what to fix."""


def path_for(name):
    return PODS_DIR / f"{name}.json"


def available():
    """Every pod on disk, by name."""
    if not PODS_DIR.exists():
        return []
    return sorted(p.stem for p in PODS_DIR.glob("*.json"))


def load(name):
    """One pod, form-checked. Raises `PodError` with the fix in the message."""
    path = path_for(name)
    if not path.exists():
        known = ", ".join(available()) or "none on disk"
        raise PodError(f"no pod named {name!r} — known pods: {known}. "
                       f"A pod is {PODS_DIR}/<name>.json")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PodError(f"{path} is not valid JSON: {exc}")

    seats = doc.get("seats")
    if not isinstance(seats, list) or not seats:
        raise PodError(f"{path} has no seats — a pod is a list of them")
    for i, seat in enumerate(seats):
        if not isinstance(seat, dict) or not seat.get("slug"):
            raise PodError(f"{path} seat {i} has no slug")
        unknown = set(seat) - SEAT_KEYS
        if unknown:
            raise PodError(f"{path} seat {seat['slug']} has unknown key(s): "
                           f"{', '.join(sorted(unknown))}")
    slugs = [s["slug"] for s in seats]
    if len(set(slugs)) != len(slugs):
        # Forge installs one .dck per slug, so the same deck twice is one deck
        # in two seats — a mirror nobody asked for, silently.
        raise PodError(f"{path} names a seat twice: "
                       f"{', '.join(sorted({s for s in slugs if slugs.count(s) > 1}))}")
    doc.setdefault("name", name)
    return doc


def seats(name):
    """The pod's opponent slugs, IN ORDER. The order is the run id."""
    return [s["slug"] for s in load(name)["seats"]]


def profiles(name):
    """`{slug: profile}` for seats that name one, or `None` if none do.

    `None` is not "everyone on Default" — it means the pod expresses no opinion,
    so `--vs-profile` and its default decide, which is what keeps a pod file
    orthogonal to the profile flags rather than fighting them.
    """
    out = {s["slug"]: s["profile"] for s in load(name)["seats"] if s.get("profile")}
    return out or None


def compose(name):
    """`{composition, brackets, seats}` — what a result should be reported by.

    B-2's last clause asks for results by POD COMPOSITION and not only in
    aggregate. Composition is the sorted archetype tags, so two pods with the
    same shape compare even when the decks differ.
    """
    doc = load(name)
    archetypes = sorted(s.get("archetype") or s["slug"] for s in doc["seats"])
    brackets = sorted(s["bracket"] for s in doc["seats"] if s.get("bracket"))
    return {"pod": doc["name"], "players": len(doc["seats"]) + 1,
            "composition": archetypes, "brackets": brackets,
            "seats": [dict(s) for s in doc["seats"]]}


def mixed_tag(profile_map, seat_slugs, default):
    """The run-id fragment for a table whose seats do not all play alike.

    Returns the shared profile name when every seat agrees — so a pod that sets
    one profile is byte-identical to `--vs-profile <that>` and no existing run id
    moves — and `Mixed<8hex>` when they do not, because two tables that play
    differently must not write the same path.
    """
    resolved = [(profile_map or {}).get(s) or default for s in seat_slugs]
    if len(set(resolved)) <= 1:
        return resolved[0] if resolved else default
    digest = hashlib.sha256(
        "\n".join(f"{s}:{p}" for s, p in sorted(zip(seat_slugs, resolved)))
        .encode()).hexdigest()[:8]
    return f"Mixed{digest}"


def match(slugs):
    """The pod whose seats ARE this ordered list, or None.

    DERIVED, never a stamp. Every record written before pods existed faced a
    table nobody named, and matching its opponents against today's files is a
    convenience for reading old runs — not a claim about what was configured.
    A pod edited later stops matching, which is correct: the record did not
    change, the file did. `record_for` marks the difference.
    """
    want = list(slugs)
    for name in available():
        try:
            if seats(name) == want:
                return name
        except PodError:
            continue
    return None


def record_for(name, opponents):
    """The block a run record carries about the table it faced.

    `named` is the load-bearing field: True when the pilot passed `--pod`, so
    the name is a FACT about the run; False when it was inferred from the
    opponent list, so it is a reading of today's files against an old record.
    """
    if name:
        info = compose(name)
        return {"name": name, "named": True, "players": info["players"],
                "composition": info["composition"], "brackets": info["brackets"]}
    inferred = match(opponents)
    if not inferred:
        return None
    info = compose(inferred)
    return {"name": inferred, "named": False, "players": info["players"],
            "composition": info["composition"], "brackets": info["brackets"]}


#: A seat taking more than this multiple of its fair share is DOMINATING the
#: table, and a seat under the reciprocal is a FLOOR. Not thresholds for a
#: verdict — a pod is allowed to be uneven — but for saying so out loud, because
#: a win rate read against 1/n when one seat takes half the games is a win rate
#: read against a null that does not exist.
DOMINANT = 2.0
FLOOR = 0.5


def calibration(name, records=None):
    """How this table actually divides its wins, from every run that faced it.

    THE POINT IS THE NULL. A four-player win rate reads against 0.25 unless
    somebody says otherwise, and nobody has: measured across the tracked
    records, `standard` gives giada-angels **0.572** and baylen-tokens
    **0.052**, and the subject seat **0.159**. So a deck scoring 0.16 there is
    at the table's typical subject rate, not two thirds of the way below a
    quarter — and the difference is the whole reading.

    TWO NULLS, AND THEY ARE NOT THE SAME. `subject` is what the decks WE have
    tried score in seat 0, pooled — it moves as the fleet's decks change, and it
    is a description of our own decks as much as of the table. A neutral control
    (`pod-control`: an opponent's own average deck in the subject chair) is the
    other one and is not computed here, because only one has ever been run.

    Pooling assumes runs are exchangeable, which they are not exactly: they
    differ in N, in clock, in AI profile and in which deck sat in seat 0, and
    games inside a JVM job are a Markov chain rather than independent draws.
    The interval is therefore optimistic and `limits` says so.
    """
    import collections

    from manamap.config import DECKS_DIR
    from manamap.sim import stats

    seats = collections.defaultdict(lambda: [0, 0])
    runs, decks, games = 0, set(), 0
    paths = records if records is not None else sorted(
        DECKS_DIR.glob("*/sim/*.json"))
    for path in paths:
        doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if (doc.get("pod") or {}).get("name") != name:
            continue
        runs += 1
        subject = doc["seats"][0]["slug"]
        decks.add(subject)
        decided = doc["summary"]["decided"]
        games += decided
        for slug, seat in doc["analysis"]["seats"].items():
            key = "SUBJECT" if slug == subject else slug
            seats[key][0] += seat.get("wins") or 0
            seats[key][1] += decided

    if not runs:
        return {"pod": name, "runs": 0, "seats": [], "measured": False,
                "note": "no tracked run has faced this table, so it has no null "
                        "and a win rate against it reads against 1/n by default"}

    fair = 1 / len(seats) if seats else None
    rows = []
    for key, (w, n) in seats.items():
        lo, hi = stats.wilson_bounds(w, n)
        rate = round(w / n, 3) if n else None
        rows.append({"seat": key, "wins": w, "games": n, "rate": rate,
                     "ci95": [round(lo, 3), round(hi, 3)] if lo is not None else None,
                     # `rate is not None`, NOT `rate`. A rate of 0.0 is FALSY
                     # and is also a measurement — the seat won nothing, which
                     # is the most informative reading a calibration produces.
                     # The first cut printed "Nonex fair" for a deck that went
                     # 0 for 39, which is the same absent-versus-zero confusion
                     # this repo keeps paying for, pointing the other way.
                     "share_of_fair": (round(rate / fair, 2)
                                       if rate is not None and fair else None)})
    rows.sort(key=lambda r: -(r["rate"] or 0))

    opponents = [r for r in rows if r["seat"] != "SUBJECT"]
    top = opponents[0] if opponents else None
    bottom = opponents[-1] if opponents else None
    subject_row = next((r for r in rows if r["seat"] == "SUBJECT"), None)
    return {
        "pod": name, "measured": True, "runs": runs, "games": games,
        "decks": sorted(decks), "fair_share": round(fair, 3) if fair else None,
        "seats": rows,
        "balance": {
            "dominant": [r["seat"] for r in opponents
                         if (r["share_of_fair"] or 0) >= DOMINANT],
            "floor": [r["seat"] for r in opponents
                      if (r["share_of_fair"] or 0) <= FLOOR],
            "spread": (round(top["rate"] - bottom["rate"], 3)
                       if top and bottom else None),
        },
        "subject_null": subject_row,
        "limits": [
            "Pooled across runs that differ in N, clock, AI profile and which "
            "deck sat in seat 0, so the interval assumes an exchangeability the "
            "runs do not have and is optimistic.",
            "Games inside one JVM job share a Match and are a Markov chain, not "
            "independent draws — the same caveat every win_rate_ci95 carries.",
            "SUBJECT pools OUR decks, so it describes the fleet as much as the "
            "table. A neutral control is `pod-control` and is not this figure.",
            "Truncated games have no winner and are excluded; the denominator "
            "is decided games.",
        ],
    }


def format_calibration(doc):
    if not doc.get("measured"):
        return f"\n{doc['pod']}: {doc['note']}"
    lines = [f"\n{doc['pod']} — CALIBRATION over {doc['runs']} run(s), "
             f"{doc['games']} decided games",
             f"  decks in seat 0: {', '.join(doc['decks'])}",
             f"  fair share {doc['fair_share']}\n"]
    for row in doc["seats"]:
        ci = f"[{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}]" if row["ci95"] else ""
        tag = ""
        if row["seat"] != "SUBJECT":
            if row["seat"] in doc["balance"]["dominant"]:
                tag = "  DOMINATES"
            elif row["seat"] in doc["balance"]["floor"]:
                tag = "  floor"
        lines.append(f"  {row['seat']:<18} {row['rate']:.3f} {ci:<18} "
                     f"{row['share_of_fair']}x fair{tag}")
    null = doc["subject_null"]
    if null:
        lines.append(f"\n  THE NULL IS {null['rate']:.3f}, NOT "
                     f"{doc['fair_share']:.3f} — that is what our decks have "
                     f"actually scored in seat 0 here.")
        # HOW MANY OF OUR DECKS, not how many games. A null pooled from ONE deck
        # is that deck's record wearing the table's name: it cannot separate an
        # uneven table from a bad deck, and 0.000 over 39 games is exactly the
        # reading a reader would take the wrong way.
        if len(doc["decks"]) < 2:
            lines.append(f"  ONE DECK ONLY ({doc['decks'][0]}), so this null is "
                         f"that deck's record as much as the table's. Run "
                         f"another deck here before reading it as a baseline.")
    if doc["balance"]["dominant"] or doc["balance"]["floor"]:
        lines.append("  This table is not even. A win rate against it is "
                     "relative to that unevenness, not to 1/n.")
    return "\n".join(lines)


def format_list():
    """Every pod, as the pilot reads it."""
    names = available()
    if not names:
        return f"no pods — a pod is {PODS_DIR}/<name>.json"
    lines = [f"\nPODS ({len(names)})\n"]
    for name in names:
        try:
            doc = load(name)
        except PodError as exc:
            lines.append(f"  {name:<14} BROKEN — {exc}")
            continue
        info = compose(name)
        lines.append(f"  {name:<14} {info['players']} players   "
                     f"{', '.join(s['slug'] for s in doc['seats'])}")
        if doc.get("note"):
            lines.append(f"                 {doc['note']}")
        tags = [f"{s['slug']} b{s['bracket']}" for s in doc["seats"] if s.get("bracket")]
        if tags:
            lines.append(f"                 {' · '.join(tags)}")
        odd = [s for s in doc["seats"] if s.get("profile")]
        if odd:
            lines.append("                 profiles: "
                         + ", ".join(f"{s['slug']}={s['profile']}" for s in odd))
    lines.append("\n  `simulate <slug> --pod <name>` expands to the same --vs "
                 "flags and the same run id.")
    return "\n".join(lines)


def main(args):
    name = getattr(args, "name", None)
    if getattr(args, "calibration", False):
        names = [name] if name else available()
        if getattr(args, "as_json", False):
            print(json.dumps([calibration(n) for n in names], indent=2,
                             ensure_ascii=False))
            return
        for n in names:
            print(format_calibration(calibration(n)))
        return
    if not name:
        print(format_list())
        return
    if getattr(args, "as_json", False):
        print(json.dumps(compose(name), indent=2, ensure_ascii=False))
        return
    info = compose(name)
    print(f"\n{info['pod']} — {info['players']} players")
    for seat in info["seats"]:
        bits = [seat.get("archetype") or "",
                f"bracket {seat['bracket']}" if seat.get("bracket") else "",
                f"profile {seat['profile']}" if seat.get("profile") else ""]
        print(f"  {seat['slug']:<18} {'  '.join(b for b in bits if b)}")
    print(f"\n  composition: {', '.join(info['composition'])}")
    print(f"  --vs " + " --vs ".join(s["slug"] for s in info["seats"]))


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot pods [<name>]`.")
