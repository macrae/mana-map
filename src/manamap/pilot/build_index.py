"""Pilot: render manuals/index.html — the shareable gallery of pilot's manuals.

Deterministic: scans data/decks/ for decks whose manual exists, sorted by slug.
"""

import html
import json

from manamap.config import DECKS_DIR, MANUALS_DIR

CSS = """
:root { --bg:#1a1a2e; --panel:#16213e; --gold:#c4a747; --text:#e8e6e3; --dim:#9a97b0;
        --green:#4ade80; --border:#3a3a5a; }
* { box-sizing:border-box; margin:0; }
body { background:var(--bg); color:var(--text); font:16px/1.6 Georgia, serif; }
.page { max-width:960px; margin:0 auto; padding:48px 32px; text-align:center; }
h1 { color:var(--gold); font-size:2.4em; }
.sub { color:var(--dim); font-style:italic; margin:8px 0 32px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:24px; }
a.deck { display:block; background:var(--panel); border:1px solid var(--border);
         border-radius:10px; padding:20px; text-decoration:none; color:var(--text);
         transition:border-color .15s; }
a.deck:hover { border-color:var(--gold); }
a.deck img { width:100%; border-radius:8px; }
a.deck h2 { color:var(--gold); font-size:1.15em; margin:12px 0 4px; }
a.deck .tag { color:var(--dim); font-style:italic; font-size:.9em; }
a.deck .stats { font:12px sans-serif; color:var(--green); margin-top:10px; }
.footer { color:var(--dim); font-size:.8em; margin-top:48px; }
"""


def esc(value):
    return html.escape(str(value)) if value is not None else ""


def gather_entries():
    """Collect gallery entries for every deck with a built manual, sorted by slug."""
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
        with open(cards_path) as f:
            doc = json.load(f)
        commanders = [c for c in doc["cards"] if c.get("is_commander")]
        commander = commanders[0] if commanders else {}
        tagline = ""
        prose_path = deck_path / "manual_prose.json"
        if prose_path.exists():
            with open(prose_path) as f:
                tagline = json.load(f).get("cover", {}).get("tagline", "")
        verified = 0
        for stack_path in sorted((deck_path / "stacks").glob("*.json")):
            with open(stack_path) as f:
                if (json.load(f).get("checker") or {}).get("verdict") == "pass":
                    verified += 1
        decisions = len(list((deck_path / "decisions").glob("*.json")))
        mean_cast = None
        goldfish_path = deck_path / "goldfish_metrics.json"
        if goldfish_path.exists():
            with open(goldfish_path) as f:
                mean_cast = json.load(f)["metrics"]["commander"]["mean_cast_turn"]
        entries.append({
            "slug": slug,
            "commander": commander.get("name", slug),
            "image": commander.get("image"),
            "tagline": tagline,
            "verified": verified,
            "decisions": decisions,
            "mean_cast": mean_cast,
        })
    return entries


def render_index(entries):
    cards = []
    for e in entries:
        image = f'<img src="{esc(e["image"])}" alt="{esc(e["commander"])}" loading="lazy">' if e["image"] else ""
        stats = f"✓ {e['verified']} verified line(s)"
        if e["decisions"]:
            stats += f" · ★ {e['decisions']} decision spread(s)"
        if e["mean_cast"] is not None:
            stats += f" · ◆ commander turn {e['mean_cast']}"
        cards.append(f"""
  <a class="deck" href="{esc(e["slug"])}.html">{image}
    <h2>{esc(e["commander"])}</h2>
    <div class="tag">{esc(e["tagline"])}</div>
    <div class="stats">{stats}</div>
  </a>""")
    body = "".join(cards) or "<p class='sub'>No manuals built yet.</p>"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pilot's Manuals — Mana Map</title>
<meta property="og:title" content="Pilot's Manuals — Mana Map">
<meta property="og:description" content="Machine-verified Commander pilot manuals: every combo line cites the Comprehensive Rules.">
<style>{CSS}</style></head><body><div class="page">
<h1>Pilot's Manuals</h1>
<p class="sub">Every combo line machine-checked against the Comprehensive Rules.
Every number reproducible. Coaching labeled as coaching.</p>
<div class="grid">{body}</div>
<p class="footer">Built by the Mana Map pilot subsystem · <a style="color:inherit"
href="../viz/index.html">explore the card map</a></p>
</div></body></html>
"""


def main(args=None):
    entries = gather_entries()
    MANUALS_DIR.mkdir(exist_ok=True)
    out = MANUALS_DIR / "index.html"
    out.write_text(render_index(entries), encoding="utf-8")
    print(f"Wrote {out} ({len(entries)} manual(s))")


if __name__ == "__main__":
    main()
