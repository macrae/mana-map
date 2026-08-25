"""Commander search: a handful of cards in, the commanders worth building toward.

PRD §6. Give it ~20 cards — from a collection, from your library, hand-picked —
and it ranks real commanders by how close their decks sit to yours in embedding
space. Archetypes are not labelled anywhere; they emerge from what people
actually build.

    seed cards
      -> seed centroid            (basics excluded, utility lands kept)
      -> colour identity          (derived from the seed, never authored)
      -> candidate commanders     (EDHREC, live, for that identity)
      -> reference centroids      (same exclusions, same type control)
      -> ranked by cosine proximity

**This module owns the maths, and `eval_commander_search` imports it.** The
dependency runs that way deliberately: the eval measures the product, so if it
had its own copy of the centroid the two would drift and the reported accuracy
would belong to code nobody ships. One scorer, the same rule the deck builder
already lives under.

**It reads the TEXT embedding by default, and that is a measured decision rather
than a convenience.** `manamap eval-commander-search` over ten held-out draws
against 79 candidates:

    text baseline (frozen MiniLM)   top1 0.584 [0.52-0.67]   top5 0.962
    function (trained ability)      top1 0.410 [0.30-0.47]   top5 0.811

The ranges do not overlap. The trained space loses to the frozen text it was
built from — the same finding `eval-embeddings` reports at the card level, and
seventeen points of top-1 is a much louder version of it. `--space function`
exists so the choice can be re-measured rather than argued about, and the day
Track A2 lands, the default flips and the eval says so first.

**Proximity is a discovery aid, not a verdict** (§6.3). The embedding is built
on oracle text, and cards with similar phrasing can do meaningfully different
things. Top-5 is 0.96 and top-1 is 0.58: the right commander is nearly always in
the short list and is usually not first. The output is a ranked list to inspect,
which is why it prints scores and neighbours rather than an answer.
"""

import random

import numpy as np
import pandas as pd

from manamap import console
from manamap.config import (ABILITY_EMBEDDINGS_PATH, EMBEDDINGS_PATH,
                            OUTPUT_CSV_PATH, TEXT_EMBEDDINGS_PATH)

#: Basics carry no signal and would pull every centroid toward one point. §6.1
#: step 2 keeps SPECIALTY and utility lands, which carry real signal — so this
#: is a name list, not a type-line rule. Command Tower stays in.
BASIC_LANDS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
                         "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
                         "Snow-Covered Mountain", "Snow-Covered Forest"})

SEED_SIZE = 20                      # §6.1 step 1, advisory — any size is accepted

#: WUBRG, because EDHREC's identity paths are in that order and "ub" and "bu"
#: are not the same URL.
WUBRG = "wubrg"

SPACES = {
    "text": TEXT_EMBEDDINGS_PATH,           # the default — see the module docstring
    "function": ABILITY_EMBEDDINGS_PATH,
    "layout": EMBEDDINGS_PATH,
}


def normalized(path):
    array = np.load(path)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)


def primary_type(type_line):
    """The one type a card counts as, for composition control.

    The order is a judgement, not alphabetical: a body is a body first (so an
    artifact creature is a Creature), and a land that also does something is a
    Land.
    """
    t = str(type_line or "")
    for kind in ("Land", "Creature", "Planeswalker", "Battle",
                 "Instant", "Sorcery", "Enchantment", "Artifact"):
        if kind in t:
            return kind
    return "Other"


class Corpus:
    """One parse of `cards.csv`, and the four views this module needs from it."""

    def __init__(self, frame=None):
        self.frame = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False) if frame is None else frame
        self.names = self.frame["name"].tolist()
        self.by_name = {}
        for i, n in enumerate(self.names):
            self.by_name.setdefault(n, i)               # first printing wins, as everywhere
            if " // " in n:                             # decklists carry front faces
                self.by_name.setdefault(n.split(" // ")[0], i)
        self.types = [primary_type(t) for t in self.frame.get("type_line", [""] * len(self.names))]
        self.identities = [self._identity(v) for v in
                           self.frame.get("color_identity", [""] * len(self.names))]

    @staticmethod
    def _identity(value):
        """`"B, G"` -> `{"b", "g"}`. The comma form is what `cards.csv` stores,
        and reading it as a single token is a documented bug in this repo's own
        history (`--identity GU` once matched only colourless cards)."""
        if not isinstance(value, str) or not value.strip():
            return frozenset()
        return frozenset(p.strip().lower() for p in value.split(",") if p.strip())

    def rows(self, card_names):
        """Names -> corpus rows. Drops basics and anything unresolved, in order."""
        out, missing = [], []
        for name in card_names:
            if name in BASIC_LANDS:
                continue
            row = self.by_name.get(name)
            if row is None:
                missing.append(name)
            elif row not in out:
                out.append(row)
        return out, missing


def centroid(embeddings, rows):
    """Mean embedding of a card set, re-normalised. None when the set is empty."""
    if len(rows) == 0:
        return None
    v = embeddings[list(rows)].mean(axis=0)
    n = np.linalg.norm(v)
    return None if n < 1e-8 else v / n


def type_controlled_rows(rows, types, seed_composition, rng):
    """Reference rows resampled to the SEED's type composition (§6.1 step 6).

    The argument: a seed that is all instants and sorceries embeds nowhere near a
    creature-heavy deck centroid, so an uncontrolled comparison ranks decks by
    how creature-heavy they are — composition rather than identity.

    **Measured, and the honest summary is "does not hurt".** Over ten draws it
    moves the trained space by +0.000 and the text space by +0.037, with
    overlapping ranges; on a single draw it looked like +7.6 and +12.7 and that
    was the draw talking. The reasoning is sound and the effect is not
    established at this pool size, so it is on by default and cheap to turn off.
    """
    by_type = {}
    for r in rows:
        by_type.setdefault(types[r], []).append(r)
    picked = []
    for kind, share in seed_composition.items():
        pool = by_type.get(kind) or []
        if not pool:
            continue
        want = max(1, int(round(share * len(rows))))
        picked.extend(pool if want >= len(pool) else rng.sample(pool, want))
    # Never nothing: a deck with none of the seed's types must stay rankable —
    # badly, which is informative — rather than dropping out of the denominator.
    return picked or list(rows)


def composition(rows, types):
    """The seed's type mix as fractions, which is what the control matches against."""
    out = {}
    for r in rows:
        out[types[r]] = out.get(types[r], 0) + 1 / len(rows)
    return out


def seed_identity(rows, corpus):
    """Colour identity of a card set: the UNION of its members'.

    DERIVED, never authored — the same rule `Discovery.brief()` follows on the
    frontend, where identity rides along as information rather than as input. A
    seed containing one black card is a black-inclusive seed, because a deck
    containing it would be.
    """
    ident = set()
    for r in rows:
        ident |= corpus.identities[r]
    return frozenset(ident)


def identity_code(identity):
    """`{"u","w"}` -> `"wu"`. Colourless is EDHREC's `colorless`."""
    code = "".join(c for c in WUBRG if c in identity)
    return code or "colorless"


def search(seed_names, space="text", controlled=True, per_identity=25,
           limit=10, seed=0, corpus=None, fetch=True):
    """Rank commanders by proximity to a seed. Returns a result dict.

    `fetch=False` reads only what `ingest/edhrec.py` has already cached, so the
    whole thing runs offline once a colour identity has been seen before.
    """
    from manamap.ingest import edhrec

    corpus = corpus or Corpus()
    rows, missing = corpus.rows(seed_names)
    if not rows:
        raise SystemExit("no seed cards resolved against the corpus — "
                         "check the names, or run `manamap extract`")

    embeddings = normalized(SPACES[space])
    seed_vec = centroid(embeddings, rows)
    identity = seed_identity(rows, corpus)
    code = identity_code(identity)
    comp = composition(rows, corpus.types)
    rng = random.Random(seed)

    # §6.1 steps 3-4. The colour filter is a crude first cut and a cheap one:
    # EDHREC's identity path IS the filter, so no legendary pool is scanned here.
    # Rankings are pulled live and NOT frozen, per §6.1 step 4 — the eval freezes
    # its own copy precisely because a benchmark needs the opposite.
    with console.task(f"Commanders for {code}", total=None) as t:
        t.state("fetching the identity's top list")
        names = edhrec.top_commanders(code, limit=per_identity) if fetch else []
        if not names:
            raise SystemExit(f"no candidate commanders for identity {code!r} "
                             f"(cached only: try again with fetching enabled)")

    ranked = []
    with console.task("Scoring candidates", total=len(names), unit="commanders") as t:
        for name in names:
            t.state(name)
            try:
                deck = edhrec.average_deck(name) if fetch else None
            except Exception as exc:
                console.err(f"    no average deck for {name}: {exc}")
                t.advance()
                continue
            if not deck or not deck["cards"]:
                t.advance()
                continue
            ref, _ = corpus.rows([n for n, _ in deck["cards"]])
            if controlled:
                ref = type_controlled_rows(ref, corpus.types, comp, rng)
            vec = centroid(embeddings, ref)
            if vec is not None:
                # Cards the seed and this deck share — the reason a reader can
                # check the match instead of trusting the number (§6.3).
                overlap = [corpus.names[r] for r in rows if r in set(ref)]
                ranked.append({"commander": name, "score": float(seed_vec @ vec),
                               "deck_size": len(ref), "shared": overlap})
            t.advance()

    ranked.sort(key=lambda c: c["score"], reverse=True)
    return {
        "seed": {"resolved": len(rows), "missing": missing,
                 "identity": code, "composition": {k: round(v, 3) for k, v in comp.items()}},
        "space": space, "type_controlled": controlled,
        "candidates": len(ranked),
        "results": ranked[:limit],
        "caveat": ("Proximity is a discovery aid, not a verdict — the embedding "
                   "reads oracle text, and cards that read alike can play "
                   "differently. Measured on held-out seeds: the right commander "
                   "is in the top 5 about 96% of the time and first about 58%."),
    }


def format_report(result):
    lines = []
    s = result["seed"]
    lines.append(f"\nCOMMANDER SEARCH — {s['resolved']} seed cards, identity "
                 f"{s['identity']}, {result['candidates']} candidates "
                 f"({result['space']} space"
                 f"{', type-controlled' if result['type_controlled'] else ''})")
    if s["missing"]:
        lines.append(f"  unresolved: {', '.join(s['missing'][:6])}"
                     + (f" (+{len(s['missing']) - 6})" if len(s["missing"]) > 6 else ""))
    lines.append("")
    lines.append(f"  {'#':>2}  {'score':>6}  commander")
    lines.append("  " + "-" * 74)
    for i, c in enumerate(result["results"], 1):
        lines.append(f"  {i:>2}  {c['score']:6.3f}  {c['commander']}")
        if c["shared"]:
            shown = ", ".join(c["shared"][:4])
            more = f" +{len(c['shared']) - 4}" if len(c["shared"]) > 4 else ""
            lines.append(f"          shares: {shown}{more}")
    lines.append("")
    lines.append(f"  {result['caveat']}")
    return "\n".join(lines)
