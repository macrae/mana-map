"""Pilot: render manuals/index.html — the newsstand.

Issues on a rack, not links in a list (STYLEv3 §12/R5). Deterministic.

**Run after `build-manual`**: the rack lists decks whose rendered HTML already
exists, so building the index first silently omits the issue you just made.
A deck without an `issue.json` sorts last under a sentinel volume of 999.
"""

import json
import re

from manamap.pilot.common import checker_passed, load_json
from manamap.config import DECKS_DIR, MANUALS_DIR
from manamap.pilot.design import stylesheet_link, write_stylesheet
from manamap.pilot.design import FONT_LINK, badge, barcode, esc
from manamap.pilot.issue_spec import MASTHEAD, SERIES_SLUG, STANDING_TAGLINE

EXTRA_CSS = """
.newsstand { padding:40px 34px 60px; }
.stand-head { text-align:center; border-bottom:5px solid var(--ink); padding-bottom:20px;
              margin-bottom:34px; }
.rack { display:grid; gap:30px; grid-template-columns:repeat(auto-fill,minmax(268px,1fr)); }
a.issue { display:block; text-decoration:none; color:inherit; background:#fff;
          border:4px solid var(--ink); box-shadow:9px 9px 0 rgba(0,0,0,.28);
          transition:transform .12s ease, box-shadow .12s ease; position:relative; }
a.issue:hover { transform:translate(-3px,-3px); box-shadow:13px 13px 0 rgba(0,0,0,.32); }
a.issue .vol { background:var(--ink); color:var(--paper); font-family:var(--condensed);
               text-transform:uppercase; letter-spacing:.2em; font-size:11px;
               padding:6px 11px; display:flex; justify-content:space-between; }
a.issue img { width:100%; border-bottom:3px solid var(--ink); }
a.issue .meta { padding:13px 14px 16px; }
a.issue h2 { font-family:var(--display); text-transform:uppercase; font-size:1.22em;
             margin:0 0 5px; line-height:1; }
a.issue .tag { color:var(--ink-soft); font-size:.9em; margin-bottom:10px; }
a.issue .stats { display:flex; flex-wrap:wrap; gap:5px; }
.stand-foot { text-align:center; margin-top:44px; font-size:.86em; color:var(--ink-soft); }
"""



# A stack scenario names its cards in STRUCTURED fields — the ordered stack, the hand, the
# graveyard — and also in prose all over `board`. The browser used to guess the line by
# substring-matching every deck card name against the whole scenario blob, which is how
# heliod's Approach-of-the-Second-Sun line drew "verified" edges to Ancient Tomb and
# Howling Mine (lands that happened to be on the board) while missing Swan Song, the actual
# interaction, because a 4-card cap truncated in NAME-LENGTH order.
#
# The line is what is on the stack and in hand. `board` is furniture: it is what the line
# happens in front of, not what it is made of. Deriving here rather than in the browser
# also means the tracked stack artifacts are untouched — rewriting them to add a field
# would change their digests and invalidate every agent-cache routine on the deck.
_STACK_TEXT_KEYS = ("object", "card", "name", "source")
# Generic board words that survive annotation-stripping and name nothing.
#
# NOTE the single backslashes. This was written `r"^(\\d+|...)\\s"`, which in a raw
# string is a LITERAL backslash followed by `d` — so the pattern only ever matched
# text containing a real backslash, i.e. nothing. The "a card name never opens with a
# number word" guard was dead from the day it was written; "2 Vampire tokens" sailed
# through it. Verified against the real scenarios before and after.
_COUNT_PHRASE = re.compile(
    r"^(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an)\s", re.I)
_NOT_A_CARD = {"lands", "untapped lands", "tapped lands", "creature", "creatures",
               "token", "tokens", "a land", "basic land", "basics"}

# A line is made of the cards that DO something in it. Two kinds of entry look like
# card names, pass every other filter, and are furniture:
#
#   Tokens — "Insect token X", "Snake token B". They are not cards, they have no row
#   in cards.csv, and a graph edge to one cannot be drawn.
#
#   Basic lands — a scenario states its mana, so "Swamp" and "Forest" appear on nearly
#   every board. Unlike a token, a basic IS in the deck, so it survives the browser's
#   `byName` check and draws a real edge to a real card that had nothing to do with the
#   line. That is the Ancient-Tomb-and-Howling-Mine failure this function exists to
#   prevent, recurring one level down: yawgmoth-swarm's line derived
#   [Fume Spitter, Blowfly Infestation, Nest of Scarabs, Zulaport Cutthroat,
#   "Insect token X", "Swamp"] and escaped only because a 4-card cap truncated first.
_BASIC_LANDS = {"plains", "island", "swamp", "mountain", "forest", "wastes",
                "snow-covered plains", "snow-covered island", "snow-covered swamp",
                "snow-covered mountain", "snow-covered forest"}
_TOKEN_WORD = re.compile(r"\btokens?\b", re.I)


def line_cards(scenario):
    """The cards a verified line is actually made of, from the scenario's structure.

    The ordered stack, the hand and the graveyard name the line directly. `board` is
    ambiguous: for heliod's Approach-of-the-Second-Sun scenario it is furniture (Ancient
    Tomb, Howling Mine, basics), but for edgar's combo loops the pieces are ALREADY
    RESOLVED PERMANENTS and the board is the only place they appear — the stack object
    there is a prose sentence about a trigger, not a card name.

    So: if the stack and hand name the line, the board is context and is skipped. If they
    name nothing, the line is on the board and the board is all there is. That rule gets
    both cases right, where "board is always furniture" got edgar wrong and "board is
    always cards" got heliod wrong.
    """
    def collect(sources):
        out, seen = [], set()

        def add(value):
            # Scenario entries carry annotations a card name never has: "Polyraptor
            # (creature spell)", "Island x5", "untapped lands x2". Strip them, then drop
            # anything that is plainly not a card.
            name = str(value or "").strip()
            name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
            name = re.sub(r"\s+x\s*\d+$", "", name, flags=re.I).strip()
            if not name or name in seen or name.lower() in _NOT_A_CARD:
                return
            # A stack object is often a sentence about a trigger rather than a card name.
            # Card names do not contain these.
            if len(name) > 60 or any(t in name for t in (";", " has just ", " trigger")):
                return
            # Board entries are sometimes a count rather than a card: "five lands, all
            # untapped", "2 Vampire tokens". A card name never opens with a number word.
            if _COUNT_PHRASE.match(name) or " lands" in name.lower():
                return
            # Furniture that survives every filter above — see _BASIC_LANDS.
            if name.lower() in _BASIC_LANDS or _TOKEN_WORD.search(name):
                return
            seen.add(name)
            out.append(name)

        for entry in sources:
            if isinstance(entry, dict):
                for key in _STACK_TEXT_KEYS:
                    if entry.get(key):
                        add(entry[key])
                        break
            else:
                add(entry)
        return out

    stack_objs = scenario.get("stack") or []
    named = collect(list(stack_objs)
                    + list(scenario.get("hand") or [])
                    + list(scenario.get("graveyard") or []))
    if len(named) >= 2:
        return named

    # Nothing on the stack or in hand — the line is already on the battlefield.
    board = (scenario.get("board") or {}).get("you") or []
    return collect(list(named) + list(board))


def gather_entries():
    """One scan, two questions — and they are not the same question.

    The newsstand asks "which decks have a published issue?"; the viz asks
    "which decks can the browser load?". Both used to be answered by the same
    gate (does `manuals/<slug>.html` exist), which meant a deck could be fully
    built, validated and rules-verified and still be invisible in the frontend
    until someone spent an agent budget on magazine prose for it.

    So the scan now admits every deck with a `cards.json` and records
    `published` per entry. `render_index` filters on it; `write_manifest` does
    not. Ordered by volume then slug.
    """
    entries = []
    if not DECKS_DIR.is_dir():
        return entries
    for deck_path in sorted(DECKS_DIR.iterdir()):
        if not deck_path.is_dir():
            continue
        slug = deck_path.name
        cards_path = deck_path / "cards.json"
        if not cards_path.exists():
            continue
        published = (MANUALS_DIR / f"{slug}.html").exists()
        doc = load_json(cards_path)
        commanders = [c for c in doc["cards"] if c.get("is_commander")]
        commander = commanders[0] if commanders else {}

        issue = load_json(deck_path / "issue.json", {})

        plan = load_json(deck_path / "issue_plan.json", {})
        coverline = (plan.get("cover") or {}).get("dominant_coverline", "")

        verified = sum(
            1 for stack_path in sorted((deck_path / "stacks").glob("*.json"))
            if checker_passed(load_json(stack_path, {})))
        decisions = len(list((deck_path / "decisions").glob("*.json")))
        mean_cast = None
        goldfish_path = deck_path / "goldfish_metrics.json"
        if goldfish_path.exists():
            with open(goldfish_path) as f:
                # A deck with no flagged commander has no commander block; one
                # bad artifact must not take down the whole newsstand.
                metrics = json.load(f).get("metrics") or {}
                mean_cast = (metrics.get("commander") or {}).get("mean_cast_turn")

        # The browser cannot list a directory, so the manifest carries the stack
        # filenames. Passing verdicts only — the dossier renders what publishes.
        stack_files = sorted(
            p.name for p in (deck_path / "stacks").glob("*.json")
            if checker_passed(load_json(p, {})))

        # What each passing line is MADE OF, so the browser can draw an edge that means
        # something instead of substring-matching prose. Keyed by filename, since that is
        # what `stack_files` hands the browser.
        stack_cards = {}
        for name in stack_files:
            doc_stack = load_json(deck_path / "stacks" / name, {})
            cards = line_cards(doc_stack.get("scenario") or {})
            if cards:
                stack_cards[name] = cards

        entries.append({
            "slug": slug,
            "published": published,
            "volume": issue.get("volume", 999),   # sentinel: un-numbered issues sort last
            "issue_date": issue.get("issue_date", ""),
            "deck_name": issue.get("deck_name") or commander.get("name", slug),
            "commander": commander.get("name", slug),
            "image": commander.get("art_crop") or commander.get("image"),
            "coverline": coverline or issue.get("cover_tagline", ""),
            "verified": verified,
            "decisions": decisions,
            "mean_cast": mean_cast,
            "stack_files": stack_files,
            "stack_cards": stack_cards,
        })
    return sorted(entries, key=lambda e: (e["volume"], e["slug"]))


def render_index(entries):
    # The rack shows what actually went to press.
    issues = []
    for e in [x for x in entries if x["published"]]:
        image = (f'<img src="{esc(e["image"])}" alt="{esc(e["commander"])}" loading="lazy">'
                 if e["image"] else "")
        stats = [f'{badge("verified")}' if e["verified"] else ""]
        if e["decisions"]:
            stats.append(badge("coach"))
        if e["mean_cast"] is not None:
            stats.append(badge("data"))
        issues.append(f"""
  <a class="issue" href="{esc(e["slug"])}.html">
    <div class="vol"><span>Vol. {e["volume"]:03d}</span><span>{esc(e["issue_date"])}</span></div>
    {image}
    <div class="meta">
      <h2>{esc(e["deck_name"])}</h2>
      <div class="tag">{esc(e["coverline"])}</div>
      <div class="tag" style="font-size:.82em">{esc(e["commander"])} ·
        {e["verified"]} verified line(s) · {e["decisions"]} decision spread(s)</div>
      <div class="stats">{"".join(stats)}</div>
    </div>
  </a>""")
    body = "".join(issues) or '<p class="dek">No issues on the rack yet.</p>'
    # The rack's social preview: the current lead issue's cover art (entries
    # arrive volume-sorted, so this is Vol. 001's commander crop).
    lead_art = next((e["image"] for e in entries if e.get("image")), "")
    og_image_tag = f'<meta property="og:image" content="{esc(lead_art)}">' if lead_art else ""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(SERIES_SLUG)} — {esc(MASTHEAD)}</title>
<meta name="description" content="Commander deck magazines: every combo line cites the Comprehensive Rules, every number is reproducible, and coaching says when it's coaching.">
<meta property="og:title" content="{esc(SERIES_SLUG)} — {esc(MASTHEAD)}">
<meta property="og:description" content="Commander deck magazines with a three-tier evidence contract.">
<meta property="og:type" content="website">
{og_image_tag}
{FONT_LINK}
{stylesheet_link()}
<style>{EXTRA_CSS}</style></head>
<body><div class="trim"><div class="newsstand">
  <div class="stand-head">
    <h1 class="masthead">{esc(MASTHEAD)}</h1>
    <div class="series-slug">{esc(SERIES_SLUG)}</div>
    <p class="dek" style="margin:16px auto 0">{esc(STANDING_TAGLINE)} — one deck per issue.
      Every combo line machine-checked against the Comprehensive Rules, every number
      reproducible, and coaching labeled as coaching.</p>
    {barcode("newsstand")}
  </div>
  <div class="rack">{body}</div>
  <p class="stand-foot">Built by the Mana Map pilot subsystem ·
    <a href="../viz/index.html">explore the card map</a><br>
    Unofficial fan content permitted under the Wizards of the Coast Fan Content Policy.</p>
</div></div></body></html>
"""


def write_manifest(entries):
    """A machine-readable rack for the deck dossier (`viz/deck.html`).

    The same scan that builds the newsstand, minus the presentation. It exists
    because a browser cannot list `stacks/` and should not hardcode a deck list:
    add a deck, run build-index, and the dossier picks it up.
    """
    manifest = {
        "decks": [
            {k: e[k] for k in ("slug", "volume", "deck_name", "commander",
                               "coverline", "verified", "decisions", "stack_files",
                               "stack_cards", "published")}
            for e in entries
        ]
    }
    out = DECKS_DIR / "index.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return out


def main(args=None):
    entries = gather_entries()
    MANUALS_DIR.mkdir(exist_ok=True)
    write_stylesheet(MANUALS_DIR)
    out = MANUALS_DIR / "index.html"
    out.write_text(render_index(entries), encoding="utf-8")
    manifest = write_manifest(entries)
    published = [e for e in entries if e["published"]]
    print(f"Wrote {out} ({len(published)} issue(s) on the rack)")
    print(f"Wrote {manifest} (deck manifest for viz/deck.html)")


if __name__ == "__main__":
    main()
