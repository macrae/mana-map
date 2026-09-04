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
