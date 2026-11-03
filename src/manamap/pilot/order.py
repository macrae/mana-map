"""Pilot: what to paste into Mana Pool to buy exactly these decks — `manamap pilot order`.

`manamap pilot order <slug> [<slug> ...] [--check] [--plain] [--out F] [--json]`

ONE PASTE, ANY NUMBER OF DECKS. Mana Pool's mass entry (https://manapool.com/add-deck)
takes `N Name (SET) CN *F*`; a line without the printing lets it choose the cheapest,
which is how orders arrive with art the pilot does not like. So this merges every
list's lines: the same printing adds up, different printings stay separate lines,
non-lands then lands, each alphabetical (`check_in.render_line`, the one line form).
A line nobody pinned is said out loud, with the command that fixes it.

`--check` READS MANA POOL'S PUBLIC PRICE FEED, which is keyed by exact printing and
carries a price and a stock count for each. It says what each pinned line costs,
which lines cannot be filled, and the dated subtotal. It does not know shipping,
fees or how many packages an order becomes: those need the token-gated optimizer
(`POST /buyer/optimizer` answered 401 anonymously, 2026-11-03, whatever its spec
says), so checkout stays the final figure. The stock count covers a printing across
finishes, so a foil line can read in stock on the strength of non-foil copies.

It never places an order and never holds a token. It never says which deck holds a
card: an order is what to buy, not an inventory (the pilot's rule).
"""

import json
from pathlib import Path

from manamap import net
from manamap.pilot import check_in, common, prices
from manamap.pilot.common import deck_dir
from manamap.pilot.fetch_deck import parse_decklist


def _land(card):
    return "Land" in str((card or {}).get("type_line") or "")


def _ids(card):
    """`(set, cn, foil)` -> scryfall id, for every art of a cards.json row."""
    return {(a.get("set"), str(a.get("collector_number"))): a.get("scryfall_id")
            for a in (card.get("printings") or [card])}


def build(slugs):
    """Every line of `slugs` merged. Returns `{slugs, lines, unpinned, count}`.

    A deck with no cards.json still orders; it just cannot say which lines are lands.
    """
    merged = {}
    for slug in slugs:
        path = deck_dir(slug) / "decklist.txt"
        if not path.exists():
            raise SystemExit(f"{path} not found")
        doc = common.load_json(deck_dir(slug) / "cards.json") or {}
        cards = {c["name"].split(" // ")[0].lower(): c
                 for c in (doc.get("cards") or []) + (doc.get("sideboard") or [])}
        for e in parse_decklist(path.read_text(encoding="utf-8")):
            card = cards.get(e["name"].split(" // ")[0].lower()) or {}
            key = (e["name"], e.get("set"), e.get("collector_number"), bool(e.get("foil")))
            row = merged.setdefault(key, {
                "name": e["name"], "quantity": 0, "set": e.get("set"),
                "collector_number": e.get("collector_number"), "foil": bool(e.get("foil")),
                "pinned": bool(e.get("set")), "land": _land(card),
                "scryfall_id": _ids(card).get((e.get("set"), str(e.get("collector_number")))),
            })
            row["quantity"] += int(e["quantity"])
    lines = sorted(merged.values(), key=lambda r: (r["land"], r["name"].lower(),
                                                   str(r["set"]), str(r["collector_number"])))
    return {"slugs": list(slugs), "lines": lines,
            "unpinned": [r["name"] for r in lines if not r["pinned"]],
            "count": sum(r["quantity"] for r in lines)}


def paste(doc, plain=False):
    """The text Mana Pool's mass entry takes. `plain` drops every printing, gathering
    copies by name, for a pilot who does not mind the art."""
    if plain:
        by_name = {}
        for r in doc["lines"]:
            by_name.setdefault((r["land"], r["name"]), 0)
            by_name[(r["land"], r["name"])] += r["quantity"]
        rows = sorted(by_name.items(), key=lambda kv: (kv[0][0], kv[0][1].lower()))
        return "".join(f"{q} {k[1]}\n" for k, q in rows)
    return "".join(check_in.render_line(r) + "\n" for r in doc["lines"])


def check(doc, session=None):
    """Every line against the public feed. Returns `{as_of, lines, subtotal_cents,
    unfillable, unpriced}`; subtotal covers what can be filled and priced."""
    meta = {}
    feed = prices.manapool_feed(session=session, meta=meta)
    if feed is None:
        raise SystemExit("the Mana Pool feed did not answer; the paste above still works")
    cheapest = {}
    for row in feed.values():
        name, nm = str(row.get("name") or "").lower(), prices._row_nm(row)
        if name and nm is not None and (name not in cheapest
                                        or nm < prices._row_nm(cheapest[name])):
            cheapest[name] = row
    out, subtotal, unfillable, unpriced = [], 0, [], []
    for r in doc["lines"]:
        row = (feed.get(r["scryfall_id"]) if r["pinned"] and r["scryfall_id"]
               else cheapest.get(r["name"].lower()) if not r["pinned"] else None)
        line = {"name": r["name"], "quantity": r["quantity"], "pinned": r["pinned"],
                "line": check_in.render_line(r).strip()}
        if row is None:
            line["status"] = "not listed"
            unfillable.append(line)
            out.append(line)
            continue
        cents = (prices._cents(row.get("price_cents_nm_foil")) if r["foil"]
                 else prices._row_nm(row))
        stock = row.get("available_quantity")
        line.update(unit_cents=cents, stock=stock, url=row.get("url"))
        if stock is not None and stock < r["quantity"]:
            line["status"] = f"only {stock} in stock"
            unfillable.append(line)
        elif cents is None:
            line["status"] = "no price in this finish"
            unpriced.append(line)
        else:
            line["status"] = "ok"
            subtotal += cents * r["quantity"]
        out.append(line)
    return {"as_of": meta.get("as_of"), "source": "manapool", "lines": out,
            "subtotal_cents": subtotal, "unfillable": unfillable, "unpriced": unpriced}


def payload(slugs, run_check=False, plain=False):
    """What `--json` prints and serve's `order` returns, from the one function."""
    doc = build(slugs)
    result = None
    if run_check:
        try:
            result = check(doc)
        except (net.Offline, SystemExit) as exc:
            result = {"error": str(exc)}
    return {"text": paste(doc, plain=plain), "count": doc["count"], "lines": doc["lines"],
            "unpinned": doc["unpinned"], "check": result}


def main(args):
    slugs = args.slugs
    plain = getattr(args, "plain", False)
    got = payload(slugs, run_check=getattr(args, "check", False), plain=plain)
    doc, text, result = got, got["text"], got["check"]
    out = getattr(args, "out", None)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    if getattr(args, "as_json", False):
        print(json.dumps(got, indent=2))
        return
    print(text, end="")
    print(f"\n{doc['count']} cards on {len(doc['lines'])} lines from {len(slugs)} deck(s). "
          f"Paste at https://manapool.com/add-deck")
    if doc["unpinned"] and not plain:
        print(f"WARNING: {len(doc['unpinned'])} line(s) have no art pinned, so Mana Pool will "
              f"choose the cheapest: `manamap pilot art <slug> --pin-shown` fixes that")
    if out:
        print(f"wrote {out}")
    if result is None:
        return
    if "error" in result:
        print(f"CHECK did not run: {result['error']} — the paste above still works")
        return
    print(f"\nCHECK against Mana Pool's feed as of {result['as_of']}")
    print(f"  cards ${result['subtotal_cents'] / 100:.2f} at NM, before shipping and fees; "
          f"checkout has the final figure")
    for r in result["unfillable"]:
        print(f"  CANNOT FILL {r['line']}: {r['status']}")
    for r in result["unpriced"]:
        print(f"  UNPRICED    {r['line']}: {r['status']}")
    if not result["unfillable"] and not result["unpriced"]:
        print("  every line is listed in the quantity asked for")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot order <slug>`.")
