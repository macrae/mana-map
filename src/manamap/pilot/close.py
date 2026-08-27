"""A bottleneck becomes a pool: what would CLOSE the gap the diagnostic named.

THE HOLE THIS FILLS. `diagnose` names a bottleneck component. Nothing turned it
into candidates. `deck_audit._closers` is the only component-aware miner here and
it fires only for components flagged `thin`, so it never runs on the eight-member
group that is actually binding ur-dragon's treasure branch. The only remaining
route was the pilot hand-assembling a pile in the Atlas — which is how a
Dragon-typal combat pile came to be weighed against a treasure-drain deck, and
how `assess` came to refuse nineteen of twenty-one cards.

TWO ROUTES, REPORTED SEPARATELY, BECAUSE THEY DISAGREE AND THE DISAGREEMENT IS
INFORMATION. Measured on that branch's multiplier component:

  * FUNCTION SPACE — the centroid of the component's own members in
    `embeddings_ability.npy`, which is where "what does this card DO" lives.
    Returns Primal Vigor, Halving Season, Hosting Season, Song of the Worldsoul:
    real multipliers, and the hand-assembled pile of 21 contained ZERO.
  * SIGNATURE — the shared role/oracle features `deck_audit._closers` already
    uses, including its documented refusal to run off a MODAL role.

The oracle route alone is imprecise for a concept the words do not separate:
`assess._MULTIPLIER` matches 80 corpus names, and Bruvac doubles mill, Branching
Evolution doubles counters, Delney doubles small-creature triggers. The function
route alone is fuzzy the other way. Neither is good enough; both, with the
overlap marked, is.

IT RETRIEVES. IT DOES NOT SCORE. This repo deleted a six-factor card scorer for
good reasons and is not rebuilding one behind a new name — `close` proposes,
`assess` triages and `candidates` MEASURES by substitution. Nothing here ranks a
card by a heuristic over its properties, and nothing here edits a declaration.
"""

import json
import re

from manamap.pilot import card_pool
from manamap.pilot.common import deck_dir, deck_file, load_deck_cards, load_json

#: How many names each route offers before the list stops being a shortlist.
DEFAULT_LIMIT = 24

#: A SIGNATURE MUST BE SPECIFIC, and the obvious choice is not.
#: `_closers` takes the most COMMON shared role, which for any component holding
#: two creatures is `threat:body` — carried by **62.4% of the classified
#: corpus**. Asked to widen ur-dragon's multipliers it offered Birds of
#: Paradise. Measured across all 91 declared components on the fleet: the
#: most-common pick lands on a role more than 40% of the corpus carries for
#: **29 of them**; taking the RAREST shared role instead lands there for **7**,
#: and those 7 genuinely have no specific shared role. Rarest-first is also the
#: rule this repo already validated when mining training positives out of
#: `card_roles.json`.
SIGNATURE_MAX_SHARE = 0.40


def _component_of(slug, branch, wanted):
    """The declared component to close, and how we chose it.

    Defaults to the DIAGNOSTIC's own bottleneck rather than recomputing one —
    `diagnostic.engine()` owns that figure, the rule `deck-info` follows
    everywhere. `--component` matches a substring LONGEST-FIRST: `interaction`
    and `interaction-breadth` are both real axis names one module over, and a
    shortest-first scan silently checks the wrong one.
    """
    doc = load_json(deck_file(slug, "goldfish_targets.json", branch)) or {}
    targets = doc.get("targets") or []
    if not targets:
        raise SystemExit(
            f"{slug} has no engine declaration to close against. "
            f"`manamap pilot scaffold-targets {slug}` writes a draft to edit.")
    labels = [t.get("label", "") for t in targets]
    if wanted:
        hits = sorted((l for l in labels if wanted.lower() in l.lower()),
                      key=len)
        if not hits:
            raise SystemExit(
                f"no component on {slug} matches {wanted!r}. It declares:\n  "
                + "\n  ".join(labels))
        label, why = hits[0], f"named on the command line ({wanted!r})"
    else:
        diag = load_json(deck_file(slug, "diagnostic.json", branch)) or {}
        b = ((diag.get("engine") or {}).get("bottleneck") or {})
        if not b.get("label"):
            raise SystemExit(
                f"no bottleneck recorded for {slug} — run "
                f"`manamap pilot diagnose {slug}"
                + (f" --branch {branch}" if branch else "")
                + " --write` first, or name one with --component.")
        label, why = b["label"], "the diagnostic's own bottleneck"
    target = next(t for t in targets if t.get("label") == label)
    # A target holds several `need` groups; the one to widen is the SMALLEST,
    # which is the same rule `deck_audit` uses to pick a thinnest component.
    groups = target.get("need") or []
    group = min(groups, key=lambda g: len(g.get("any_of") or [])) if groups else {}
    return {"label": label, "why": why,
            "members": list(group.get("any_of") or []),
            "siblings": [n for g in groups if g is not group
                         for n in (g.get("any_of") or [])]}


def _by_function(members, identity, held, pool, limit):
    """Nearest to the component's centroid in the FUNCTION space.

    `embeddings_ability.npy` is gitignored, so this route is absent on a fresh
    clone and must DEGRADE rather than raise — and must say which route ran. A
    miner that silently returns half its answer is the `libraryNames` fallback
    bug: a plausible short list is worse than an honest refusal.
    """
    try:
        import numpy as np

        from manamap import config
        from manamap.analysis.commander_search import centroid
        path = config.ABILITY_EMBEDDINGS_PATH
        if not path.exists():
            return None, f"{path.name} is absent (it is gitignored) — run the pipeline"
        E = np.load(path)
    except Exception as exc:                       # pragma: no cover - env
        return None, f"unavailable: {type(exc).__name__}"

    frame = card_pool.load_frame()
    if frame is None or len(frame) != len(E):
        return None, "cards.csv and the embeddings disagree on length"
    row_of = {}
    for i, name in enumerate(frame["name"]):
        row_of.setdefault(name, i)
    rows = [row_of[n] for n in members if n in row_of]
    if not rows:
        return None, "none of the component's members resolved to a row"
    vec = centroid(E, rows)
    if vec is None:
        return None, "the component's centroid is degenerate"
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sim = (E / norms) @ vec

    out, seen = [], set()
    for i in np.argsort(-sim):
        name = frame["name"].iloc[int(i)]
        if name in seen or name in held or name in members:
            continue
        info = pool.get(name)
        if not info or not info["legal"] or not info["color_identity"] <= identity:
            continue
        seen.add(name)
        out.append({"name": name, "similarity": round(float(sim[int(i)]), 3),
                    "cmc": info["cmc"], "type_line": info["type_line"],
                    "edhrec_rank": info["edhrec_rank"]})
        if len(out) >= limit:
            break
    return out, None


def _role_frequency(roles):
    """What share of the classified corpus carries each role."""
    import collections
    counts = collections.Counter(r for rs in roles.values() for r in rs)
    total = len(roles) or 1
    return {r: n / total for r, n in counts.items()}


def _by_signature(component, held, identity, pool, roles, limit):
    """The role/oracle signature route, reusing `deck_audit._closers` verbatim.

    Its shared-role rule is load-bearing and documented there: a role held by
    ONE member describes that card, not the group's job.
    """
    from manamap.pilot import deck_audit
    from manamap.pilot.common import load_combo_details

    members = component["members"]
    # `_component` takes the raw `any_of` GROUP, not a name list — it is the
    # declaration's own shape and reusing it verbatim is the whole point.
    shaped = deck_audit._component({"any_of": members}, held, roles) if members else None
    if not shaped:
        return None, "the component has no members to take a signature from"
    # REORDER the shared roles rarest-first and let `_closers` take [0]. Done
    # here rather than in `deck_audit` on purpose: that function is also the
    # audit's and the doctor's, and changing what it picks would move
    # agent-facing output on every deck as a side effect of a new command.
    freq = _role_frequency(roles)
    shared = shaped.get("shared_roles") or []
    if shared:
        shaped = dict(shaped, shared_roles=sorted(shared, key=lambda r: freq.get(r, 0)))
        best = shaped["shared_roles"][0]
        if freq.get(best, 0) > SIGNATURE_MAX_SHARE:
            return None, (
                f"no specific signature: the rarest role two or more members "
                f"share is {best!r}, which {freq[best]:.0%} of the corpus "
                f"carries. Widening on it would return the corpus, so this "
                f"component is one the taxonomy has no word for — the function "
                f"route is the one to read.")
    siblings = [{"cards": component["siblings"]}] if component["siblings"] else []
    got = deck_audit._closers(shaped, siblings, held, identity, pool, roles,
                              load_combo_details())
    if not got.get("available"):
        return None, got.get("reason")
    rows = [{"name": e["name"], "signature": e["signature"], "cmc": e["cmc"],
             "type_line": e["type_line"], "edhrec_rank": e["edhrec_rank"]}
            for e in got["by_role"][:limit]]
    return {"cards": rows, "signature": got.get("role_signature"),
            "kind": got.get("role_signature_kind"),
            "total": got.get("by_role_total"),
            "by_combo_line": got.get("by_combo_line", [])[:limit],
            "note": got.get("note")}, None


def close(slug, branch=None, component=None, limit=DEFAULT_LIMIT, owned_only=False):
    comp = _component_of(slug, branch, component)
    doc = load_deck_cards(slug, branch)
    held = {c["name"] for c in doc["cards"]}
    identity = set()
    for c in doc["cards"]:
        if c.get("is_commander"):
            identity |= set(c.get("color_identity") or [])
    pool = card_pool.load_pool() or {}
    roles = {}
    try:
        from manamap.pilot.common import load_card_roles
        roles = load_card_roles()
    except Exception:                              # pragma: no cover - env
        pass

    fn, fn_why = _by_function(comp["members"], identity, held, pool, limit)
    sig, sig_why = _by_signature(comp, held, identity, pool, roles, limit)

    if owned_only:
        from manamap.pilot import collection
        own = collection.owned_names()
        if fn:
            fn = [r for r in fn if r["name"] in own]
        if sig:
            sig["cards"] = [r for r in sig["cards"] if r["name"] in own]

    fn_names = {r["name"] for r in (fn or [])}
    sig_names = {r["name"] for r in ((sig or {}).get("cards") or [])}
    both = sorted(fn_names & sig_names)
    return {
        "slug": slug, "branch": branch,
        "component": {k: comp[k] for k in ("label", "why", "members")},
        "routes": {
            "function": {"available": fn is not None, "why": fn_why,
                         "cards": fn or []},
            "signature": {"available": sig is not None, "why": sig_why,
                          **(sig or {})},
        },
        # THE OVERLAP IS THE STRONG SIGNAL and the disagreement is the
        # interesting one — two routes that agree everywhere would mean one of
        # them is redundant, and nothing here would tell you which.
        "both_routes": both,
        "pool": sorted(fn_names | sig_names),
    }


def main(args):
    branch = getattr(args, "branch", None)
    doc = close(args.slug, branch,
                component=getattr(args, "component", None),
                limit=getattr(args, "limit", None) or DEFAULT_LIMIT,
                owned_only=bool(getattr(args, "owned", False)))
    if getattr(args, "as_json", False) or getattr(args, "json", False):
        print(json.dumps(doc, indent=1)); return

    where = doc["slug"] + (f"/{doc['branch']}" if doc.get("branch") else "")
    c = doc["component"]
    print(f"\nCLOSE — {where}")
    print(f"  component: {c['label']}")
    print(f"  chosen as: {c['why']}")
    print(f"  it holds {len(c['members'])}: {', '.join(c['members'][:6])}"
          + (" …" if len(c["members"]) > 6 else ""))

    fnr = doc["routes"]["function"]
    print(f"\n  FUNCTION SPACE — nearest to what these cards DO")
    if not fnr["available"]:
        print(f"    unavailable: {fnr['why']}")
    else:
        for r in fnr["cards"][:12]:
            mark = "**" if r["name"] in doc["both_routes"] else "  "
            print(f"  {mark}{r['similarity']:.3f}  mv{int(r['cmc']):<2} {r['name']}")

    sg = doc["routes"]["signature"]
    print(f"\n  SIGNATURE — shares a role with two or more members")
    if not sg["available"]:
        print(f"    unavailable: {sg['why']}")
    else:
        if sg.get("note"):
            print(f"    {sg['note'][:96]}")
        if sg.get("signature"):
            print(f"    signature: {sg['signature']} ({sg['kind']}), "
                  f"{sg['total']} in the corpus")
        for r in sg["cards"][:12]:
            mark = "**" if r["name"] in doc["both_routes"] else "  "
            print(f"  {mark}       mv{int(r['cmc']):<2} {r['name']}")

    if doc["both_routes"]:
        print(f"\n  BOTH ROUTES ({len(doc['both_routes'])}): "
              f"{', '.join(doc['both_routes'])}")
    print(f"\n  {len(doc['pool'])} name(s) to triage:")
    print(f"    manamap pilot close {doc['slug']}"
          + (f" --branch {doc['branch']}" if doc.get("branch") else "")
          + " --write  &&  manamap pilot assess "
          + f"{doc['slug']}"
          + (f" --branch {doc['branch']}" if doc.get("branch") else "")
          + " --pool library")

    if getattr(args, "write", False):
        out = deck_dir(args.slug) / "pool.txt"
        out.write_text("\n".join(doc["pool"]) + "\n", encoding="utf-8")
        print(f"\n  Wrote {out} ({len(doc['pool'])} names)")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot close <slug>`.")
