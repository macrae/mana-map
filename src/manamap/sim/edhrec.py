"""EDHREC's COMMANDER pages — per-card synergy and inclusion for one commander, optionally
under one of its THEMES — as a dated, ★-tier artifact (`edhrec_cards.json`).

`opponents.py` reads EDHREC's AVERAGE DECK to seat a pod; nothing read the commander
page, which is the only per-commander inclusion signal this repo has access to
(`docs/history/deck-builder-v2.md` names the hole). The page is
`https://json.edhrec.com/pages/commanders/<slug>[/<theme>].json`; the cards live in
`container.json_dict.cardlists[*].cardviews[*]` with `name`, `synergy` (−1..1, EDHREC's
own figure: inclusion here minus inclusion in the colour identity at large), `num_decks`
and `potential_decks`, grouped under tags (`highsynergycards`, `topcards`, `creatures`,
…). Each list truncates at about fifty cards per type, so "not listed" means below the
cut, not zero — `candidate_scan` records absence as absence.

This is EVIDENCE, never a measurement: a number about other people's decks, dated by
`as_of`, perishable like `deck_recon.json`, and the validator holds it to the same
form. It is never a model input and never drives a figure.
"""
import json
import re
import urllib.request
from datetime import date

from manamap.pilot.common import deck_dir, load_deck_cards

EDHREC_COMMANDER = "https://json.edhrec.com/pages/commanders/{slug}.json"
EDHREC_THEME = "https://json.edhrec.com/pages/commanders/{slug}/{theme}.json"
ARTIFACT = "edhrec_cards.json"
BASE = "base"


def edhrec_slug(commander):
    s = commander.lower().replace("'", "").replace(",", "")
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def url_for(slug, theme=None):
    return EDHREC_THEME.format(slug=slug, theme=theme) if theme else EDHREC_COMMANDER.format(slug=slug)


def flatten(doc, url):
    """The page's cards as rows — every cardlist, every tag — or ValueError when the
    page carries no `cardlists` (a theme that does not exist returns a page with none)."""
    jd = ((doc.get("container") or {}).get("json_dict") or {})
    lists = jd.get("cardlists")
    if not lists:
        raise ValueError(f"no cardlists in the EDHREC page at {url}")
    rows = []
    for cl in lists:
        tag = cl.get("tag") or cl.get("header") or "?"
        for cv in cl.get("cardviews") or []:
            if not cv.get("name"):
                continue
            rows.append({"name": cv["name"], "tag": tag, "synergy": cv.get("synergy"),
                         "num_decks": cv.get("num_decks"), "potential_decks": cv.get("potential_decks")})
    card = jd.get("card") or {}
    return {"url": url, "commander": card.get("name"), "num_decks": card.get("num_decks"),
            "potential_decks": card.get("potential_decks"), "rows": rows}


#: EDHREC drops a bare-client connection now and then ("Remote end closed connection
#: without response" on the second of three pages, 2026-09-30), so a fetch names itself
#: and retries with a pause. Three attempts; the last error is the one raised.
USER_AGENT = "mana-map/1.0 (+https://github.com/macrae/mana-map; a Commander deck workbench)"
ATTEMPTS = 3


def fetch(slug, theme=None):
    import time
    url = url_for(slug, theme)
    last = None
    for attempt in range(ATTEMPTS):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                doc = json.loads(r.read().decode("utf-8"))
            return flatten(doc, url)
        except (OSError, ValueError) as exc:          # RemoteDisconnected is an OSError
            last = exc
            if isinstance(exc, ValueError):
                raise
            time.sleep(2 * (attempt + 1))
    raise SystemExit(f"EDHREC did not answer {url} after {ATTEMPTS} attempts: {last}")


def merge(commander, slug, pages):
    """`{BASE|theme: page}` -> the artifact. A card keeps every theme's figures side by
    side (`by_theme`) and the lists it appeared in, so a reader can tell "high synergy
    on the aristocrats page" from "a staple on the base page"."""
    cards, themes = {}, {}
    for theme, page in pages.items():
        themes[theme] = {"url": page["url"], "num_decks": page["num_decks"],
                         "potential_decks": page["potential_decks"], "rows": len(page["rows"])}
        for r in page["rows"]:
            c = cards.setdefault(r["name"], {"lists": [], "by_theme": {}})
            c["lists"].append(f"{r['tag']}@{theme}")
            c["by_theme"].setdefault(theme, {"synergy": r["synergy"], "num_decks": r["num_decks"],
                                             "potential_decks": r["potential_decks"]})
    for c in cards.values():
        # THE HEADLINE FIGURES a scan row carries: the base page when it listed the card,
        # else the first theme that did — and `from` says which.
        src = BASE if BASE in c["by_theme"] else next(iter(c["by_theme"]))
        c.update({"synergy": c["by_theme"][src]["synergy"], "num_decks": c["by_theme"][src]["num_decks"],
                  "potential_decks": c["by_theme"][src]["potential_decks"], "from": src})
    return {"slug": slug, "commander": commander, "as_of": date.today().isoformat(),
            "source": "EDHREC commander page(s), json.edhrec.com", "themes": themes,
            "cards": dict(sorted(cards.items()))}


def partition(doc, names):
    """Move every card the corpus does not know out of `cards` into `not_in_corpus`.

    EDHREC lists cards the moment they are previewed; the corpus is a dated Scryfall
    dump (`docs/data-artifacts.md`), so three of Edgar's 302 rows on 2026-09-30 were
    newer than it (Edgar, Ancient Bloodlord; Kindred Judgment; Turbulent Crater). A
    validator that failed on them would be firing on correct data; listing them apart
    keeps the file honest about what it could and could not check."""
    if not names:
        return doc
    keep, gone = {}, []
    for name, row in doc["cards"].items():
        if name in names or name.split(" // ")[0] in names:
            keep[name] = row
        else:
            gone.append(name)
    doc["cards"] = keep
    doc["not_in_corpus"] = sorted(gone)
    return doc


def deck_commander(slug):
    cards = load_deck_cards(slug).get("cards") or []
    names = [c["name"] for c in cards if c.get("is_commander")]
    if not names:
        raise SystemExit(f"{slug}: cards.json names no commander")
    return names[0]


def main(args):
    slug = args.slug
    commander = deck_commander(slug)
    es = edhrec_slug(commander)
    themes = list(getattr(args, "theme", None) or [])
    pages = {BASE: fetch(es)}
    for th in themes:
        pages[th] = fetch(es, th)
    from manamap.pilot.card_pool import corpus_names
    doc = partition(merge(commander, slug, pages), corpus_names())
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2, ensure_ascii=False))
    else:
        print(f"EDHREC — {commander} ({es}) · {len(doc['cards'])} card(s) across {', '.join(pages)}")
        for th, meta in doc["themes"].items():
            print(f"  {th:14s} {meta['num_decks']} decks · {meta['rows']} rows · {meta['url']}")
        if doc.get("not_in_corpus"):
            print(f"  {len(doc['not_in_corpus'])} card(s) newer than the corpus, listed apart: "
                  + ", ".join(doc["not_in_corpus"]))
    p = deck_dir(slug) / ARTIFACT
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  wrote {p}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot fetch-edhrec <slug> [--theme T]`.")
