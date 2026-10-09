"""A branch's purchases as one list Mana Pool's mass entry will take.

`manamap pilot buy-list <slug> --branch <name> [--exact] [--out F] [--json]`

WHAT IT IS. The bill on a branch (`net-change`, the branch page) already says
which cards nobody owns; what it could not do was hand you those cards in a form
a shop accepts. Mana Pool's mass entry (https://manapool.com/add-deck) takes a
pasted decklist — one `N Name` per line, or `N Name (SET) CN` to pin the
printing — and nothing else: no URL prefill, no API. So this prints exactly that
text, and the branch page copies it to the clipboard and opens the page.

WHERE THE ROWS COME FROM. `deck_branch.source` is the one place that decides
whether a card is in the deck, in a box, sleeved elsewhere or a purchase; this
reads its `buy` state and nothing else. It never opens the collection boxes
itself, and it never says which deck holds a card — a buy list is what to buy,
not an inventory (the pilot's rule: no paper inventory tracking).

WHAT A LINE LOOKS LIKE. `check_in.decklist_line` — the same formatter that
writes `decklist.txt` on a check-in — so the exact form here is byte-for-byte
the form the repo already reads back. The printing (`set`, `collector_number`,
`foil`) comes from the branch's `cards.json`, which is what `fetch-deck`
resolved; a branch without one is refused rather than guessed at.

WHAT IT COSTS. Prices are stripped from the corpus by design. If a
`prices.json` sits beside the branch (`{cards: {name: {nm_cents, …}}, as_of,
source}` — written by another area, not this one), the footer carries the sum;
if it does not, or any row is unpriced, there is no figure. A partial sum shown
as the bill is a figure nobody measured.
"""

import json

from manamap.pilot import check_in, deck_branch
from manamap.pilot.common import deck_dir, expand_faces, load_json, resolve_out_path

#: Paste-only. Mana Pool has no URL prefill, so the page opens this and the
#: text is already on the clipboard.
MANAPOOL_URL = "https://manapool.com/add-deck"

#: The price file another area writes beside a branch. Read if present; never
#: created here and never substituted for.
PRICES_FILE = "prices.json"

ROW_KEYS = ("name", "quantity", "set", "collector_number", "foil")


def rows(slug, branch):
    """The branch's cards whose source state is BUY, with their printings.

    One row per DISTINCT NAME, carrying the branch's own `quantity` — you buy
    Sol Ring once, and a basic the branch adds three of is one line reading `3`.
    Names come back in `source()`'s vocabulary (the `A // B` form `cards.json`
    keys), looked up by either face so a DFC resolves whichever way the
    decklist named it.
    """
    doc = load_json(deck_dir(slug, branch) / "cards.json")
    if not doc:
        raise SystemExit(
            f"{slug}@{branch} has no cards.json, so there is no printing to list — "
            f"run `manamap pilot fetch-deck {slug} --branch {branch}` first")
    by_face = {}
    for c in doc.get("cards") or []:
        for face in expand_faces(c.get("name")):
            by_face.setdefault(face, c)
    src = deck_branch.source(slug, branch)
    out = []
    for r in src["cards"]:
        if r["state"] != deck_branch.BUY:
            continue
        c = next((by_face[f] for f in expand_faces(r["name"]) if f in by_face), {})
        out.append({"name": r["name"],
                    "quantity": int(c.get("quantity") or 1),
                    "set": c.get("set") or None,
                    "collector_number": c.get("collector_number") or None,
                    "foil": bool(c.get("foil"))})
    return sorted(out, key=lambda r: r["name"])


def render(rows_, exact=False):
    """The paste: one line per row, `N Name` or — `exact` — `N Name (SET) CN`.

    Through `check_in.decklist_line`, so this is the form `check-in` writes and
    `fetch-deck` reads back; the foil marker rides through in both forms because
    the importer takes it in both and dropping it would re-resolve the finish.
    """
    lines = []
    for r in rows_:
        e = {"name": r["name"], "quantity": r.get("quantity") or 1,
             "foil": bool(r.get("foil"))}
        if exact:
            e["set"] = r.get("set")
            e["collector_number"] = r.get("collector_number")
        lines.append(check_in.decklist_line(e))
    return "\n".join(lines)


def prices(slug, branch):
    """The `prices.json` beside the branch, or None. Never invented."""
    return load_json(deck_dir(slug, branch) / PRICES_FILE)


def total_cents(rows_, prices_):
    """The sum of `nm_cents` × quantity over every row, or None.

    None when there is no price file, when it carries no `cards`, or when ANY
    row is unpriced — a partial sum presented as the total is a figure nobody
    measured. An empty buy list with a price file totals 0, which is a real
    number: there is nothing to pay.
    """
    cards = (prices_ or {}).get("cards")
    if not isinstance(cards, dict):
        return None
    total = 0
    for r in rows_:
        entry = next((cards[f] for f in expand_faces(r["name"]) if f in cards), None)
        cents = (entry or {}).get("nm_cents") if isinstance(entry, dict) else None
        if cents is None:
            return None
        total += int(cents) * int(r.get("quantity") or 1)
    return total


def payload(slug, branch, exact=False):
    """The dict `--json` prints and `serve`'s `branch/buy-list` returns."""
    rs = rows(slug, branch)
    p = prices(slug, branch)
    return {"text": render(rs, exact=bool(exact)),
            "count": len(rs),
            "buy_cents": total_cents(rs, p),
            "as_of": (p or {}).get("as_of") if p else None}


def footer(doc, prices_=None):
    """`N cards to buy`, and the figure only when there is one to carry."""
    n = doc["count"]
    line = f"{n} card{'' if n == 1 else 's'} to buy"
    if doc.get("buy_cents") is not None:
        line += (f"  ≈ ${doc['buy_cents'] / 100:.2f} "
                 f"({(prices_ or {}).get('source') or 'Manapool'}, {doc.get('as_of')})")
    return line


def main(args):
    slug, branch = args.slug, args.branch
    exact = bool(getattr(args, "exact", False))
    as_json = bool(getattr(args, "as_json", False))
    doc = payload(slug, branch, exact=exact)
    if as_json:
        body = json.dumps(doc, indent=2, ensure_ascii=False)
    else:
        body = (doc["text"] + "\n" if doc["text"] else "") + footer(doc, prices(slug, branch))
    out = getattr(args, "out", None)
    if out:
        # SLUG-SCOPED, like every per-deck view: a generic name in a shared
        # scratch directory is how one deck's list silently becomes another's.
        path = resolve_out_path(out, slug, "buy-list",
                                ext=".json" if as_json else ".txt")
        path.write_text(body + "\n", encoding="utf-8")
        print(f"wrote {path}")
        return
    print(body)
