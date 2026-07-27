"""Pilot: render manuals/index.html — the newsstand.

Issues on a rack, not links in a list (STYLEv3 §12/R5). Deterministic.

**Run after `build-manual`**: the rack lists decks whose rendered HTML already
exists, so building the index first silently omits the issue you just made.
A deck without an `issue.json` sorts last under a sentinel volume of 999.
"""

import json

from manamap.pilot.common import checker_passed, load_json
from manamap.config import DECKS_DIR, MANUALS_DIR
from manamap.pilot.design import CSS as MAGAZINE_CSS
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


def gather_entries():
    """Gallery entries for every deck with a built issue, by volume then slug."""
    entries = []
    if not DECKS_DIR.is_dir():
        return entries
    for deck_path in sorted(DECKS_DIR.iterdir()):
        slug = deck_path.name
        if not (MANUALS_DIR / f"{slug}.html").exists():
            continue
        cards_path = deck_path / "cards.json"
        if not cards_path.exists():
            continue
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

        entries.append({
            "slug": slug,
            "volume": issue.get("volume", 999),   # sentinel: un-numbered issues sort last
            "issue_date": issue.get("issue_date", ""),
            "deck_name": issue.get("deck_name") or commander.get("name", slug),
            "commander": commander.get("name", slug),
            "image": commander.get("art_crop") or commander.get("image"),
            "coverline": coverline or issue.get("cover_tagline", ""),
            "verified": verified,
            "decisions": decisions,
            "mean_cast": mean_cast,
        })
    return sorted(entries, key=lambda e: (e["volume"], e["slug"]))


def render_index(entries):
    issues = []
    for e in entries:
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
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(SERIES_SLUG)} — {esc(MASTHEAD)}</title>
<meta name="description" content="Commander deck magazines: every combo line cites the Comprehensive Rules, every number is reproducible, and coaching says when it's coaching.">
<meta property="og:title" content="{esc(SERIES_SLUG)} — {esc(MASTHEAD)}">
<meta property="og:description" content="Commander deck magazines with a three-tier evidence contract.">
<meta property="og:type" content="website">
{FONT_LINK}
<style>{MAGAZINE_CSS}{EXTRA_CSS}</style></head>
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


def main(args=None):
    entries = gather_entries()
    MANUALS_DIR.mkdir(exist_ok=True)
    out = MANUALS_DIR / "index.html"
    out.write_text(render_index(entries), encoding="utf-8")
    print(f"Wrote {out} ({len(entries)} issue(s) on the rack)")


if __name__ == "__main__":
    main()
