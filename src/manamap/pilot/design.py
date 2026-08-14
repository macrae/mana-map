"""Pilot: the Pilot's Manual design system (STYLEv3 §8).

The visual layer of the magazine: design tokens, the stylesheet, and the fixed
component library the issue plan composes from. Kept separate from
build_manual.py so the renderer reads as editorial assembly and the costume
lives in one place.

Everything here is deterministic — no dates, no randomness. Where a "varied"
treatment is wanted (violator angles, tape rotation), it derives from a stable
hash of the content so the same input always renders the same page.
"""

import hashlib
import math
import html

# Google Fonts: Michroma (Eurostile-class display), Archivo Black + Oswald
# (condensed feature heads), Bangers (comic slugs), Special Elite (typewriter),
# Inter (body). All free/OFL.
FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    "family=Archivo+Black&family=Bangers&family=Inter:wght@400;600;800&"
    "family=Michroma&family=Oswald:wght@500;700&family=Special+Elite&"
    'display=swap" rel="stylesheet">'
)


def esc(value):
    """Escape for HTML text/attribute context."""
    return html.escape(str(value if value is not None else ""), quote=True)


def stable_angle(text, spread=5.0):
    """Deterministic small rotation from content — same text, same tilt."""
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()[0]
    return round((digest / 255.0) * 2 * spread - spread, 2)


CSS = """
/* ── Tokens ─────────────────────────────────────────────────────────── */
:root {
  --paper:#F4EFE4; --ink:#1A1714; --ink-soft:#4A4038; --rule:#1A1714;
  --power-red:#E4002B; --burst-yellow:#FFD800; --radical-purple:#7B2D8B;
  --slime-green:#3FBF3F; --hot-magenta:#E4007C;
  --chrome-hi:#E8ECF0; --chrome-lo:#7A8699; --y2k-blue:#1B4FD8; --y2k-violet:#5B2E9E;
  --tier-verified:#2E7D32; --tier-data:#1B4FD8; --tier-coach:#C8A03C;
  --manila:#E8D9A8; --stamp-red:#C41E1E;
  --trim:1080px;
  --display:"Archivo Black","Arial Black",sans-serif;
  --condensed:"Oswald","Arial Narrow",sans-serif;
  --techno:"Michroma","Arial Black",sans-serif;
  --comic:"Bangers",cursive;
  --type:"Special Elite","Courier New",monospace;
  --body:"Inter",system-ui,-apple-system,sans-serif;
}

/* ── Paper ──────────────────────────────────────────────────────────── */
* { box-sizing:border-box; }
body {
  margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--body); font-size:16px; line-height:1.55;
  /* cheap-stock tooth */
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='60' height='60'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3'/></filter><rect width='60' height='60' filter='url(%23n)' opacity='0.028'/></svg>");
}
.trim { max-width:var(--trim); margin:0 auto; background:var(--paper);
        box-shadow:0 0 60px rgba(0,0,0,.22); }
img { max-width:100%; display:block; }
a { color:inherit; }

/* ── Display type ───────────────────────────────────────────────────── */
.masthead {
  font-family:var(--techno); font-size:clamp(38px,8.5vw,86px); line-height:.92;
  letter-spacing:-.02em; text-transform:uppercase; color:var(--chrome-hi);
  background:linear-gradient(180deg,#fff 0%,#cfd8e2 42%,#7A8699 52%,#e9eef4 68%,#fff 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent;
  filter:drop-shadow(3px 3px 0 var(--power-red)) drop-shadow(5px 5px 0 rgba(0,0,0,.55));
  margin:0;
}
.series-slug {
  font-family:var(--condensed); font-weight:700; letter-spacing:.42em;
  text-transform:uppercase; font-size:13px; color:var(--paper);
  background:var(--ink); display:inline-block; padding:4px 12px 3px; margin-top:6px;
}
h1.feature {
  font-family:var(--display); font-size:clamp(30px,5.4vw,58px); line-height:.95;
  text-transform:uppercase; margin:.12em 0 .18em; letter-spacing:-.015em;
  text-shadow:3px 3px 0 rgba(0,0,0,.16);
}
.kicker {
  font-family:var(--condensed); font-weight:700; text-transform:uppercase;
  letter-spacing:.2em; font-size:12.5px; color:var(--power-red);
  display:inline-block; border-bottom:3px solid currentColor; padding-bottom:2px;
}
.dek { font-size:1.12em; line-height:1.45; color:var(--ink-soft); max-width:60ch;
       margin:0 0 1em; }
h2.dept-title {
  font-family:var(--display); text-transform:uppercase; font-size:clamp(22px,3.4vw,34px);
  margin:0; line-height:1; letter-spacing:-.01em;
}
h3 { font-family:var(--condensed); font-weight:700; text-transform:uppercase;
     letter-spacing:.06em; font-size:1.05em; margin:1.4em 0 .4em; }
p { margin:0 0 1em; }
.body-copy { max-width:68ch; }
.body-copy p:first-of-type::first-letter {
  font-family:var(--display); font-size:3.1em; line-height:.82; float:left;
  padding:.06em .12em 0 0; color:var(--power-red);
}

/* ── Department chrome ──────────────────────────────────────────────── */
.dept { padding:44px 34px 18px; border-top:6px solid var(--accent,var(--ink));
        position:relative; }
.dept-head { display:flex; align-items:flex-end; gap:14px; flex-wrap:wrap;
             border-bottom:3px solid var(--ink); padding-bottom:10px; margin-bottom:20px; }
.dept-promise { font-family:var(--condensed); text-transform:uppercase;
                letter-spacing:.12em; font-size:11.5px; color:var(--ink-soft);
                margin-left:auto; text-align:right; max-width:34ch; }
.byline { font-family:var(--condensed); text-transform:uppercase;
          letter-spacing:.14em; font-size:11px; color:var(--ink-soft);
          margin-top:3px; }
/* The Flight Plan: sections grouped by act — lean rows, no chart furniture. */
.toc-act { margin:18px 0 6px; }
.toc-act-title { font-family:var(--condensed); text-transform:uppercase;
                 letter-spacing:.22em; font-size:12.5px; color:var(--accent,var(--ink));
                 border-bottom:2px solid var(--ink); padding-bottom:4px;
                 margin:0 0 2px; }
table.toc { width:100%; border-collapse:collapse; font-size:.92em; }
table.toc td { padding:7px 10px 7px 0; border-bottom:1px solid #cdc5b5;
               vertical-align:baseline; }
table.toc tr:last-child td { border-bottom:none; }
td.toc-title { white-space:nowrap; width:1%; padding-right:18px; }
td.toc-title a { color:var(--ink); text-decoration:none; }
td.toc-title a:hover b { color:var(--accent,var(--ink)); }
td.toc-promise { color:var(--ink-soft); font-style:italic; }
.toc-byline { display:block; font-family:var(--condensed); text-transform:uppercase;
              letter-spacing:.1em; font-size:10px; color:var(--ink-soft);
              margin-top:3px; white-space:nowrap; }
/* ── Art break: the declared breather between dense spreads (§6) ────── */
.art-break { position:relative; border-top:6px solid var(--ink); }
.art-break img { width:100%; max-height:420px; object-fit:cover; }
.art-break .pull-quote { position:absolute; left:34px; bottom:26px; margin:0;
  background:rgba(26,23,20,.82); color:var(--paper); padding:12px 18px;
  border-left:6px solid var(--burst-yellow); max-width:32ch; }
.art-break .art-credit, .art-break .printing { position:absolute; right:12px;
  bottom:6px; color:rgba(244,239,228,.75); font-size:.72em; }
.art-break .printing { bottom:22px; }

/* ── Card links: tap → tile in The 99, hover → card preview ─────────── */
a.cardref { color:inherit; position:relative;
            text-decoration:underline solid; text-decoration-thickness:1px;
            text-decoration-color:var(--accent,var(--ink-soft));
            text-underline-offset:2px; }
a.cardref:hover { background:rgba(0,0,0,.05); }
.card-pop { display:none; position:absolute; left:0; bottom:1.5em;
            width:230px; max-width:60vw; z-index:70;
            border:3px solid var(--ink); border-radius:11px;
            box-shadow:6px 6px 0 rgba(0,0,0,.35); background:var(--paper); }
a.cardref:hover .card-pop, a.cardref:focus .card-pop { display:block; }
.folio { display:flex; justify-content:space-between; align-items:center;
         font-family:var(--condensed); text-transform:uppercase; letter-spacing:.18em;
         font-size:11px; border-top:2px solid var(--ink); margin-top:30px;
         padding:7px 34px 22px; color:var(--ink-soft); }
.folio strong { color:var(--accent,var(--ink)); }

/* ── Tier badges ────────────────────────────────────────────────────── */
.badge { font-family:var(--condensed); font-weight:700; text-transform:uppercase;
         letter-spacing:.13em; font-size:10.5px; padding:4px 9px; display:inline-block;
         white-space:nowrap; border:2px solid; }
.badge-verified { color:#fff; background:var(--tier-verified); border-color:var(--tier-verified); }
.badge-data     { color:#fff; background:var(--tier-data); border-color:var(--tier-data); }
.badge-coach    { color:var(--ink); background:rgba(200,160,60,.2); border-color:var(--tier-coach); }
/* Inline tier marks, for a section that genuinely mixes tiers inside one
   department — a computed bracket delta beside a judgment about a swap. The
   department badge still comes from the spec; these only label the sentence. */
.tier-data  { color:var(--tier-data); font-weight:800; }
.tier-coach { color:var(--tier-coach); font-weight:800; }
.swap-list { list-style:none; padding:0; margin:14px 0; }
.swap-list li { border-left:5px solid var(--y2k-blue); background:rgba(27,79,216,.05);
                padding:11px 14px; margin:9px 0; line-height:1.5; }
.legend { border:3px double var(--ink); padding:16px 18px; margin:20px 0;
          background:rgba(255,255,255,.5); }
.legend h3 { margin-top:0; }
.legend-row { display:flex; gap:12px; align-items:flex-start; margin:9px 0;
              font-size:.92em; }
.legend-row .badge { flex:0 0 auto; min-width:132px; text-align:center; }

/* ── Components ─────────────────────────────────────────────────────── */
.violator {
  position:absolute; font-family:var(--comic); background:var(--burst-yellow);
  color:var(--ink); border:3px solid var(--ink); padding:12px 16px; z-index:5;
  font-size:clamp(15px,2.4vw,22px); line-height:1; text-transform:uppercase;
  box-shadow:4px 4px 0 rgba(0,0,0,.5); letter-spacing:.02em;
  clip-path:polygon(50% 0,61% 12%,77% 6%,79% 23%,95% 27%,86% 41%,100% 52%,86% 62%,
                    95% 77%,79% 80%,77% 96%,61% 90%,50% 100%,39% 90%,23% 96%,21% 80%,
                    5% 77%,14% 62%,0 52%,14% 41%,5% 27%,21% 23%,23% 6%,39% 12%);
  padding:26px 24px; text-align:center; min-width:150px;
}
.violator.plain { clip-path:none; padding:9px 14px; }

.pilot-tip {
  display:grid; grid-template-columns:88px 1fr; gap:14px; align-items:center;
  border:3px solid var(--ink); background:linear-gradient(180deg,#fff,rgba(255,216,0,.22));
  padding:12px 14px; margin:18px 0; box-shadow:5px 5px 0 rgba(0,0,0,.18);
}
.pilot-tip img { border:2px solid var(--ink); }
.pilot-tip .slug { font-family:var(--comic); font-size:1.35em; color:var(--power-red);
                   letter-spacing:.03em; display:block; line-height:1; margin-bottom:3px; }

.fast-facts { border:3px solid var(--ink); background:#fff; padding:0; margin:18px 0; }
.fast-facts .ff-head { background:var(--ink); color:var(--paper); font-family:var(--techno);
                       font-size:11px; letter-spacing:.16em; padding:7px 12px;
                       text-transform:uppercase; }
.fast-facts dl { display:grid; grid-template-columns:auto 1fr; margin:0; }
.fast-facts dt { font-family:var(--condensed); text-transform:uppercase; font-size:11.5px;
                 letter-spacing:.09em; padding:7px 12px; border-bottom:1px solid #d9d2c4;
                 color:var(--ink-soft); }
.fast-facts dd { margin:0; padding:7px 12px; border-bottom:1px solid #d9d2c4;
                 font-variant-numeric:tabular-nums; font-weight:600; text-align:right; }

.meter { margin:11px 0; }
.meter-label { display:flex; justify-content:space-between; font-family:var(--condensed);
               text-transform:uppercase; letter-spacing:.09em; font-size:11.5px;
               margin-bottom:3px; }
.meter-label b { font-variant-numeric:tabular-nums; }
.meter-track { height:16px; border:2px solid var(--ink); background:#fff;
               display:flex; gap:2px; padding:2px; }
.meter-seg { flex:1; background:#e4ded1; }
.meter-seg.on { background:linear-gradient(180deg,var(--y2k-blue),var(--y2k-violet)); }
/* The signature number. Full-bleed within the column, and big enough that a reader
   flipping past stops — which is the whole job, and the reason it may appear once. */
.badge-none { display:inline-block; width:0; }
/* ── The Editor's Letter and The Pilot's Log ─────────────────────────── */
/* The letter is the only page in the book that is not a department, and it has
   to LOOK like it before it is read — a reader flipping past should register
   "this is the editor" from the shape alone. Hence the tinted stock, the heavy
   rule, the wordmark device and the two columns: four signals that all say the
   same thing, because one of them alone reads as a styling accident. */
.letterhead { background:var(--manila); border:3px solid var(--ink);
  border-top:12px solid var(--ink); box-shadow:7px 7px 0 rgba(0,0,0,.22);
  padding:26px 30px 22px; margin:0 0 22px; }
.letterhead .lh-top { display:flex; align-items:flex-end; gap:16px;
  flex-wrap:wrap; border-bottom:3px solid var(--ink); padding-bottom:12px; }
.letterhead .lh-mark { font-family:var(--techno); font-size:clamp(19px,3.1vw,29px);
  line-height:1; letter-spacing:-.01em; text-transform:uppercase;
  color:var(--ink); }
.letterhead .lh-slug { font-family:var(--condensed); font-weight:700;
  letter-spacing:.34em; text-transform:uppercase; font-size:11px;
  color:var(--paper); background:var(--stamp-red); padding:4px 10px 3px;
  display:inline-block; margin-top:7px; }
.letterhead .lh-vol { margin-left:auto; font-family:var(--type); font-size:.74rem;
  color:var(--ink-soft); text-align:right; letter-spacing:.04em; }
.letterhead .lh-body { display:grid; grid-template-columns:1fr 240px; gap:28px;
  margin-top:20px; align-items:start; }
/* Two columns of type, which is the actual format break — a letter set at body
   measure is indistinguishable from the article after it however it is tinted. */
.letterhead .lh-copy { column-count:2; column-gap:26px;
  column-rule:1px solid rgba(26,23,20,.28); font-size:.95em; line-height:1.5; }
.letterhead .lh-copy p { margin:0 0 .85em; }
.letterhead .lh-copy p:first-of-type::first-letter { font-family:var(--display);
  font-size:2.9em; line-height:.82; float:left; padding:.06em .1em 0 0;
  color:var(--stamp-red); }
.letterhead .lh-card { margin:0; }
.letterhead .lh-card img { border:3px solid var(--ink);
  box-shadow:5px 5px 0 rgba(0,0,0,.24); }
.letterhead .lh-card figcaption { font-family:var(--condensed);
  text-transform:uppercase; letter-spacing:.1em; font-size:.62rem;
  color:var(--ink-soft); margin-top:6px; }
.letterhead .lh-rail { border-top:3px solid var(--ink); margin-top:14px;
  padding-top:10px; }
.letterhead .lh-rail h4 { font-family:var(--condensed); font-weight:700;
  text-transform:uppercase; letter-spacing:.18em; font-size:.7rem; margin:0 0 .6em;
  color:var(--stamp-red); }
.letterhead .lh-rail ol { margin:0; padding-left:1.15em; }
.letterhead .lh-rail li { font-size:.79rem; line-height:1.36; margin-bottom:.6em; }
.letterhead .lh-rail b { font-family:var(--condensed); text-transform:uppercase;
  letter-spacing:.06em; display:block; font-size:.82rem; }
.letterhead .lh-sign { margin-top:18px; border-top:2px solid var(--ink);
  padding-top:10px; }
.letterhead .lh-sign .lh-hand { font-family:var(--type); font-size:1.28rem;
  color:var(--ink); }
.letterhead .lh-sign .lh-role { font-family:var(--condensed);
  text-transform:uppercase; letter-spacing:.14em; font-size:.72rem;
  color:var(--ink-soft); margin-top:2px; }
@media (max-width:760px) {
  .letterhead .lh-body { grid-template-columns:1fr; }
  .letterhead .lh-copy { column-count:1; }
}
@media print { .letterhead .lh-copy { column-count:2; } }
.letter { max-width:62ch; font-size:1.04em; }
.letter-sign { font-family:var(--condensed); text-transform:uppercase;
  letter-spacing:.14em; font-size:.78rem; color:var(--ink-soft); margin-top:1.4em;
  border-top:2px solid var(--ink); display:inline-block; padding-top:6px; }
/* ── The stack theatre ──────────────────────────────────────────────────
   A stack you move through. No script anywhere in this file, so the whole
   mechanism is radio inputs the labels drive and `:checked ~` selectors. The
   inputs are visually hidden but NOT display:none — a removed input is a
   removed tab stop, and the rail has to stay keyboard-reachable. */
.theatre { margin:22px 0 26px; border:3px solid var(--ink);
  box-shadow:7px 7px 0 rgba(0,0,0,.28); background:#0B0A14; overflow:hidden; }
.theatre .th-in { position:absolute; width:1px; height:1px; opacity:0;
  pointer-events:none; }
.th-stage { position:relative; height:340px; overflow:hidden;
  perspective:1000px; perspective-origin:50% 40%; }
.th-grid { position:absolute; inset:0; width:100%; height:100%; display:block; }
/* `pointer-events:none` on the deck is load-bearing, not tidiness. The deck is
   absolutely positioned over the whole stage, so it caught every pointer event
   before it reached a plate: exactly ONE plate of eight was hit-testable (the
   front one, whose transform is small enough that its layout box still sits
   under the cursor) and hover silently did nothing for the other seven. It looks
   correct in a screenshot and is dead to the hand, which is the failure mode
   this repo has written down about the canvas twice. Measured both ways —
   1 of 8 reachable before, 8 of 8 after. */
.th-deck { position:absolute; inset:0; transform-style:preserve-3d;
  pointer-events:none; }
.th-plate { pointer-events:auto; }
.th-plate { position:absolute; left:50%; top:50%; width:222px; margin:-84px 0 0 -111px;
  --off:calc(var(--i) - var(--n,6) / 2);
  background:linear-gradient(168deg,#F6F2E7,#D9D2C2); color:var(--ink);
  border:2px solid var(--ink); border-radius:5px; padding:8px 9px 9px;
  box-shadow:0 14px 30px rgba(0,0,0,.55); transform-style:preserve-3d;
  transform:translate3d(calc(var(--off) * 13px),calc(var(--off) * -8px),
            calc(var(--i) * -44px)) rotateY(-22deg);
  opacity:.62; transition:transform .42s cubic-bezier(.22,.75,.3,1),
            opacity .32s ease, filter .32s ease; }
.th-plate img { width:100%; height:74px; object-fit:cover; border:1px solid var(--ink);
  margin-bottom:6px; }
.th-plate .th-n { position:absolute; top:-11px; left:-11px; width:26px; height:26px;
  border-radius:50%; background:var(--power-red); color:#fff; border:2px solid var(--ink);
  font-family:var(--condensed); font-weight:700; font-size:13px; line-height:22px;
  text-align:center; }
.th-plate h5 { margin:0 0 3px; font-family:var(--condensed); font-weight:700;
  text-transform:uppercase; letter-spacing:.04em; font-size:.78rem; }
.th-plate p { margin:0; font-size:.66rem; line-height:1.3; color:var(--ink-soft);
  display:-webkit-box; -webkit-line-clamp:4; -webkit-box-orient:vertical;
  overflow:hidden; }
/* Hovering lifts a plate out of the deck — the one interaction that needs no
   click, and the reason the stack reads as a physical object. */
.th-plate:hover { opacity:1; transform:translate3d(calc(var(--off) * 13px),
  calc(var(--off) * -8px),calc(var(--i) * -44px + 92px)) rotateY(-10deg); z-index:98; }
.th-railwrap { display:flex; align-items:center; gap:10px; padding:9px 12px;
  background:var(--ink); border-top:3px solid var(--ink); }
.th-rail { display:flex; flex-wrap:wrap; align-items:center; gap:5px; }
.th-rail-lbl { font-family:var(--condensed); text-transform:uppercase;
  letter-spacing:.2em; font-size:.62rem; color:var(--burst-yellow);
  flex:0 0 auto; }
.th-tab { cursor:pointer; min-width:26px; text-align:center; padding:3px 7px;
  font-family:var(--condensed); font-weight:700; font-size:.76rem;
  background:rgba(244,239,228,.14); color:var(--paper); border:1px solid rgba(244,239,228,.4);
  transition:background .16s ease, transform .16s ease; }
.th-tab:hover { background:rgba(244,239,228,.34); }
.th-in:focus-visible ~ .th-railwrap .th-tab { outline:2px solid var(--burst-yellow); }
.th-body { background:var(--paper); padding:16px 18px 4px; }
.th-note { display:none; }
.th-note-n { font-family:var(--condensed); text-transform:uppercase;
  letter-spacing:.14em; font-size:.64rem; color:var(--power-red); margin-bottom:6px; }
.th-note > b { display:block; font-size:.95rem; line-height:1.4; margin-bottom:6px; }
.th-note .effect { font-size:.88rem; line-height:1.45; color:var(--ink-soft);
  margin-bottom:8px; }
.th-over { background:var(--paper); margin:0; padding:0 18px 12px;
  font-size:.76rem; font-style:italic; color:var(--ink-soft); }
@media (max-width:640px) { .th-stage { height:270px; } .th-plate { width:186px; } }
/* Motion is the affordance, not the content: with it off the plates still sort
   themselves by depth and every step is still one tab away. */
@media (prefers-reduced-motion:reduce) {
  .th-plate { transition:none; }
}
/* On paper there is no cursor, so the stage becomes an illustration and every
   step prints. A printed page that shows step 1 and hides twenty-three is a
   printed page missing the proof. */
@media print {
  .theatre { box-shadow:none; }
  .th-stage { height:230px; }
  .th-railwrap, .th-over { display:none; }
  .th-plate { opacity:1 !important; filter:none !important; }
  .th-note { display:block !important; border-top:1px solid rgba(26,23,20,.25);
    padding-top:10px; margin-bottom:10px; }
}
/* The take the rest of the department argues with. Loud on purpose: at body size
   it is an opinion, at this size it is a position somebody has to answer. */
.hot-take { margin:0 0 26px; padding:26px 26px 18px; position:relative;
  background:linear-gradient(155deg,var(--ink) 0%,#2A2036 58%,var(--radical-purple));
  border:3px solid var(--ink); box-shadow:8px 8px 0 var(--burst-yellow),
  8px 8px 0 3px var(--ink); color:var(--paper); }
.hot-take .ht-burst { font-family:var(--comic); font-size:1.5rem;
  letter-spacing:.06em; text-transform:uppercase; color:var(--ink);
  background:var(--burst-yellow); display:inline-block; padding:2px 14px 0;
  transform:rotate(-2.2deg); box-shadow:3px 3px 0 rgba(0,0,0,.45);
  margin-bottom:14px; }
.hot-take blockquote { margin:0; font-family:var(--display);
  font-size:clamp(1.25rem,3.1vw,1.85rem); line-height:1.16;
  letter-spacing:-.01em; text-shadow:2px 2px 0 rgba(0,0,0,.4); }
.hot-take blockquote p { margin:0 0 .45em; }
.hot-take blockquote p:last-child { margin-bottom:0; }
.hot-take .ht-by { margin-top:16px; padding-top:10px;
  border-top:2px solid rgba(244,239,228,.45); font-family:var(--condensed);
  text-transform:uppercase; letter-spacing:.16em; font-size:.72rem;
  color:var(--burst-yellow); }
/* A panel is people answering each other, so the turns are indented against a
   speaker rail rather than stacked as blocks — the eye tracks the exchange. */
.panel { max-width:70ch; }
.panel .turn { border-left:5px solid var(--turn,#8A93B5); padding:2px 0 2px 16px;
  margin:0 0 1.25em; }
.panel .turn-voice { font-family:var(--condensed); font-weight:700;
  text-transform:uppercase; letter-spacing:.12em; font-size:.72rem;
  color:var(--turn,var(--ink-soft)); margin-bottom:4px; }
.panel .turn-text p:last-child { margin-bottom:0; }
.panel .turn-coach { --turn:var(--tier-coach); }
.panel .turn-verified { --turn:var(--tier-verified); }
.panel .turn-data { --turn:var(--tier-data); }
.constellation-fig, .engine-fig { margin:26px 0; }
.engine-fig .engine-flow { width:100%; height:auto; display:block;
  border:3px solid var(--ink); box-shadow:6px 6px 0 rgba(0,0,0,.30); background:#0B0A14; }
.engine-fig figcaption { font-size:.78rem; color:var(--ink-soft); margin-top:8px; }
.ckeys .eline { width:24px; height:0; border-top:2px dashed #8A93B5; display:inline-block; }
.ckeys .eline.on { border-top:2px solid #4CAF50; }
/* A 99 group heading wearing its city's colour, so the grid reads as the map's
   legend rather than as a second taxonomy. */
.city-head { display:flex; align-items:center; gap:10px; margin:1.9em 0 .6em;
  padding-bottom:6px; border-bottom:3px solid var(--city,var(--ink));
  font-family:var(--display); text-transform:uppercase; letter-spacing:.02em;
  font-size:1.18em; color:var(--ink); }
.city-head .city-chip { width:16px; height:16px; flex:0 0 16px; border-radius:3px;
  background:var(--city); box-shadow:0 0 0 2px var(--ink), 3px 3px 0 var(--city-lt); }
.city-head .city-count { margin-left:auto; font-family:var(--condensed);
  font-weight:500; font-size:.62em; letter-spacing:.14em; color:var(--ink-soft); }
.city-gloss { margin:-.2em 0 1em; max-width:62ch; color:var(--ink-soft);
  font-size:.92em; }
.city-head .city-verified { font-family:var(--condensed); font-size:.6em;
  letter-spacing:.08em; color:var(--tier-verified); border:2px solid currentColor;
  padding:1px 5px; }
.constellation-fig .constellation { width:100%; height:auto; display:block;
  border:3px solid var(--ink); box-shadow:6px 6px 0 rgba(0,0,0,.30); background:#0B0A14; }
.ckeys { display:flex; flex-wrap:wrap; gap:14px; margin:10px 0 4px;
  font-family:var(--condensed); text-transform:uppercase; letter-spacing:.06em;
  font-size:.7rem; color:var(--ink-soft); }
.ckeys .ck { display:flex; align-items:center; gap:6px; }
.ckeys .dot { width:11px; height:11px; border-radius:50%; display:inline-block; }
.ckeys .dot.cmdr { background:#FFD800; box-shadow:0 0 0 2px #C8A03C; }
.ckeys .dot.ver { background:#E4007C; box-shadow:0 0 0 1.6px #fff, 0 0 0 2.6px var(--ink); }
.ckeys .dot.plain { background:#8A93B5; }
/* Cards the constellation could not fit a label on. Set quieter than the key it
   follows — it is a completeness note, not a fourth legend row. */
.cunplaced { margin:2px 0 0; font-size:.74rem; line-height:1.4;
  color:var(--ink-soft); font-style:italic; }
.ckeys .edge { width:22px; height:0; border-top:2px solid #8A93B5; display:inline-block; }
.constellation-fig figcaption { font-size:.78rem; color:var(--ink-soft); margin-top:8px; }
/* The Game Plan's conditions. Set as a rail beside the thesis, not under it:
   a caveat printed after the argument reads as a retraction, and one printed
   beside it reads as the terms the argument is offered on. */
.not-modelled { margin:24px 0; padding:16px 18px 12px; background:#F0E6CE;
  border:3px solid var(--ink); border-left:10px solid var(--stamp-red);
  box-shadow:5px 5px 0 rgba(0,0,0,.18); }
.not-modelled h4 { margin:0 0 .5em; font-family:var(--condensed); font-weight:700;
  text-transform:uppercase; letter-spacing:.14em; font-size:.78rem;
  color:var(--stamp-red); }
.not-modelled ul { margin:0; padding-left:1.1em; }
.not-modelled li { font-size:.86rem; line-height:1.45; margin-bottom:.5em;
  color:var(--ink); }
.not-modelled .nm-src { font-family:var(--condensed); text-transform:uppercase;
  letter-spacing:.1em; font-size:.62rem; background:var(--ink); color:var(--paper);
  padding:1px 5px; margin-right:6px; vertical-align:1px; }
.not-modelled .nm-more { margin:.4em 0 0; font-size:.78rem; font-style:italic;
  color:var(--ink-soft); }
.stat-slab { margin:26px 0; padding:22px 18px; text-align:center; color:#fff;
  background:linear-gradient(160deg,var(--ink),#2a2440 62%,var(--y2k-violet));
  border:3px solid var(--ink); box-shadow:6px 6px 0 rgba(0,0,0,.30); }
.stat-slab .ss-figure { font-family:var(--display); font-size:clamp(3.4rem,13vw,6.2rem);
  line-height:.86; letter-spacing:-.02em; text-shadow:4px 4px 0 rgba(0,0,0,.42); }
.stat-slab .ss-label { font-family:var(--condensed); text-transform:uppercase;
  letter-spacing:.16em; font-size:.9rem; margin-top:12px; color:var(--paper); }
.stat-slab .ss-note { font-size:.82rem; margin-top:9px; opacity:.82;
  max-width:48ch; margin-left:auto; margin-right:auto; }
/* The ★ gauge. Deliberately unlike .meter: a bar reads as measurement, and this
   is a judgment. Gold, not the data blue/violet ramp. */
.coach-gauge { display:flex; align-items:center; gap:10px; margin:11px 0; }
.coach-gauge .cg-label { font-family:var(--condensed); text-transform:uppercase;
  letter-spacing:.06em; font-size:.72rem; color:var(--ink); }
.coach-gauge .cg-stars { letter-spacing:.12em; font-size:1.05rem; line-height:1; }
.coach-gauge .cg-star { color:#d8d2c4; }
.coach-gauge .cg-star.on { color:var(--tier-coach); text-shadow:1px 1px 0 rgba(0,0,0,.28); }

.callout { display:grid; grid-template-columns:46px 1fr; gap:14px; margin:16px 0;
           align-items:start; }
.callout .n { font-family:var(--display); font-size:30px; line-height:.9; color:#fff;
              background:var(--power-red); border:3px solid var(--ink); text-align:center;
              padding:6px 0 4px; box-shadow:3px 3px 0 rgba(0,0,0,.35); }
.callout .t { font-family:var(--condensed); font-weight:700; text-transform:uppercase;
              letter-spacing:.07em; display:block; margin-bottom:2px; }

.threat-box { border:3px solid var(--ink); margin:18px 0; background:#fff;
              box-shadow:6px 6px 0 rgba(0,0,0,.16); }
.threat-box .tb-head { background:var(--radical-purple); color:#fff; padding:9px 14px;
                       font-family:var(--display); text-transform:uppercase;
                       display:flex; justify-content:space-between; align-items:center;
                       gap:10px; font-size:1.02em; }
.threat-box .tb-body { padding:14px; }

.scenario { border:2px dashed var(--ink); background:rgba(255,255,255,.6);
            padding:13px 15px; margin:16px 0; font-size:.94em; }
.scenario .lbl { font-family:var(--condensed); text-transform:uppercase;
                 letter-spacing:.1em; font-size:11px; color:var(--ink-soft); }

.branches { display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); }
.branch { border:3px solid var(--ink); background:#fff; padding:13px; }
.branch h4 { font-family:var(--condensed); text-transform:uppercase; margin:0 0 7px;
             font-size:1em; letter-spacing:.05em; border-bottom:2px solid var(--burst-yellow);
             padding-bottom:4px; }
.branch dt { font-family:var(--condensed); text-transform:uppercase; font-size:10.5px;
             letter-spacing:.1em; color:var(--ink-soft); margin-top:8px; }
.branch dd { margin:2px 0 0; font-size:.92em; }
.verdict { border:3px solid var(--tier-coach); background:rgba(200,160,60,.14);
           padding:13px 15px; margin-top:16px; }

.pull-quote { font-family:var(--display); font-size:clamp(20px,3.2vw,30px); line-height:1.12;
              text-transform:uppercase; color:var(--power-red); border-top:5px solid var(--ink);
              border-bottom:5px solid var(--ink); padding:18px 0; margin:26px 0;
              transform:skewY(-.6deg); }

.map-key { display:flex; flex-wrap:wrap; gap:16px; border:2px solid var(--ink);
           padding:9px 13px; margin:14px 0; font-family:var(--condensed);
           text-transform:uppercase; font-size:11.5px; letter-spacing:.08em;
           background:rgba(255,255,255,.6); }

.tax-ladder { width:100%; border-collapse:collapse; margin:16px 0; }
.tax-ladder th { background:var(--y2k-violet); color:#fff; font-family:var(--techno);
                 font-size:10.5px; letter-spacing:.1em; padding:8px; text-transform:uppercase; }
.tax-ladder td { border:2px solid var(--ink); padding:8px 10px; text-align:center;
                 font-variant-numeric:tabular-nums; font-weight:600; }

/* ── Cards ──────────────────────────────────────────────────────────── */
.card-grid { display:grid; gap:16px; grid-template-columns:repeat(auto-fill,minmax(158px,1fr)); }
.card-tile { border:3px solid var(--ink); background:#fff; padding:9px;
             box-shadow:4px 4px 0 rgba(0,0,0,.2); }
.card-tile img { border:1px solid var(--ink); margin-bottom:7px; }
.card-tile h4 { font-family:var(--condensed); text-transform:uppercase; font-size:.9em;
                margin:0 0 4px; letter-spacing:.03em; }
.card-tile p { font-size:.82em; line-height:1.4; margin:0; color:var(--ink-soft); }

/* Utility classes for the handful of styles the renderer repeated inline —
   promoted so the output stops carrying 40+ copies of the same declaration. */
.soft { color:var(--ink-soft); }
.small { font-size:.9em; }
.rule-top { border-top:3px solid var(--ink); padding-top:18px; margin-top:26px; }
/* A subhead INSIDE a department. Deliberately quieter than `.feature` and louder
   than an <h3>: it has to read as a turn in one argument, not as the start of a
   new department — which is the thing the Act III merge exists to stop. */
.act-sub { font-family:var(--display); font-size:1.6rem; line-height:1.05;
  text-transform:uppercase; letter-spacing:.01em; margin:34px 0 14px;
  padding-top:12px; border-top:2px solid var(--accent); color:var(--ink); }
.chip { font-family:var(--condensed); text-transform:uppercase; font-size:9.5px;
        letter-spacing:.1em; background:var(--ink); color:var(--paper);
        padding:2px 6px; display:inline-block; margin:0 3px 5px 0; }
/* The engine stage, wearing the same ink as its bay in the schematic. It leads
   the synergy chips because the job outranks the resemblance, and it is outlined
   rather than filled so a tile never reads as two competing black labels. */
.chip.stage { background:var(--paper); color:var(--st,var(--ink));
              box-shadow:inset 0 0 0 2px var(--st,var(--ink)); font-weight:700; }
figure.card-fig { margin:0 0 16px; }
figure.card-fig img { border:3px solid var(--ink); box-shadow:6px 6px 0 rgba(0,0,0,.22); }
figcaption { font-size:.87em; line-height:1.42; margin-top:7px; color:var(--ink-soft); }
figcaption b { color:var(--ink); font-family:var(--condensed); text-transform:uppercase;
               letter-spacing:.06em; }
.hero-art { position:relative; }
.hero-art img { border:4px solid var(--ink); box-shadow:10px 10px 0 rgba(0,0,0,.3); }
.art-credit { font-family:var(--condensed); text-transform:uppercase; font-size:10.5px;
              letter-spacing:.14em; color:var(--ink-soft); margin-top:6px; }
.printing { font-family:var(--condensed); text-transform:uppercase; font-size:10px;
            letter-spacing:.12em; color:var(--ink-soft); margin-top:2px; }

/* Foil: a real property of the physical card, not decoration. */
.foil { position:relative; display:inline-block; }
.foil::after {
  content:""; position:absolute; inset:0; pointer-events:none; mix-blend-mode:screen;
  background:linear-gradient(115deg, rgba(255,0,128,.34) 8%, rgba(255,214,0,.30) 24%,
    rgba(0,255,170,.30) 40%, rgba(0,170,255,.32) 56%, rgba(170,0,255,.30) 72%,
    rgba(255,0,128,.28) 88%);
  background-size:220% 220%; background-position:38% 42%;
}
.foil-tag { font-family:var(--comic); font-size:.95em; letter-spacing:.04em;
            background:linear-gradient(100deg,#ff4fa3,#ffd600,#00e0b0,#4aa8ff,#c04fff);
            -webkit-background-clip:text; background-clip:text; color:transparent;
            filter:drop-shadow(1px 1px 0 rgba(0,0,0,.45)); }

/* ── Cover ──────────────────────────────────────────────────────────── */
.cover { position:relative; padding:26px 30px 34px; overflow:hidden;
         background:radial-gradient(circle at 50% 22%,#fff 0%,var(--paper) 62%); }
.cover-top { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }
.cover-meta { font-family:var(--condensed); text-transform:uppercase; font-size:11px;
              letter-spacing:.14em; text-align:right; line-height:1.75; }
.cover-body { display:grid; grid-template-columns:1fr 300px; gap:26px; margin-top:22px;
              align-items:start; }
.coverline { font-family:var(--display); text-transform:uppercase; line-height:.92;
             font-size:clamp(34px,6.6vw,68px); margin:.12em 0;
             text-shadow:4px 4px 0 var(--burst-yellow), 7px 7px 0 rgba(0,0,0,.5); }
.teases { list-style:none; padding:0; margin:16px 0 0; }
.teases li { font-family:var(--condensed); font-weight:700; text-transform:uppercase;
             font-size:clamp(12px,1.6vw,15px); letter-spacing:.05em; padding:7px 0 7px 24px;
             border-top:2px solid var(--ink); position:relative; }
.teases li::before { content:"\\25B6"; position:absolute; left:0; color:var(--power-red); }
.barcode { display:flex; gap:1.5px; align-items:flex-end; height:38px; margin-top:14px; }
.barcode i { display:block; width:2px; background:var(--ink); }

/* ── Dossiers (Judge's Desk) ────────────────────────────────────────── */
.dossier { background:var(--manila); border:2px solid #b9a469; margin:20px 0;
           box-shadow:6px 6px 0 rgba(0,0,0,.18); position:relative; }
.dossier .file-tab { position:absolute; top:-19px; left:22px; background:var(--manila);
                     border:2px solid #b9a469; border-bottom:none; padding:3px 16px;
                     font-family:var(--condensed); text-transform:uppercase;
                     letter-spacing:.14em; font-size:11px; }
.dossier .dossier-head { padding:16px 18px 12px; border-bottom:2px solid #b9a469;
                         display:flex; justify-content:space-between; gap:14px;
                         align-items:center; flex-wrap:wrap; }
.stamp { font-family:var(--display); text-transform:uppercase; color:var(--stamp-red);
         border:3px solid var(--stamp-red); padding:5px 11px; font-size:.86em;
         letter-spacing:.08em; opacity:.85; }
.dossier ol { margin:0; padding:14px 18px 16px 40px; }
.dossier li { margin-bottom:13px; }
.dossier .effect { color:var(--ink-soft); }
.cite { font-family:var(--type); font-size:.85em; background:rgba(255,255,255,.62);
        border-left:4px solid var(--stamp-red); padding:6px 10px; margin:7px 0 0; }
.cite b { font-family:var(--condensed); letter-spacing:.06em; }
.dossier-pointer { font-family:var(--condensed); text-transform:uppercase;
                   letter-spacing:.12em; font-size:11.5px; border:2px solid var(--ink);
                   padding:7px 12px; display:inline-block; margin-top:12px;
                   background:var(--manila); text-decoration:none; }

/* Collapsible case files: the summary carries the tab, title and stamp; the
   record opens on tap. Print always shows the full record — proof does not
   collapse on paper. */
details.dossier > summary { cursor:pointer; list-style:none; position:relative;
                            display:block; }
details.dossier > summary::-webkit-details-marker { display:none; }
details.dossier > summary .dossier-head { display:flex; }
details.dossier > summary::after { content:"▸"; position:absolute; right:14px;
                                   top:16px; font-size:.9em; color:var(--ink-soft); }
details.dossier[open] > summary::after { content:"▾"; }
details.dossier > p.small { padding:0 18px 14px; margin:0; }
.dossier, .dept { scroll-margin-top:28px; }

/* ── Navigation (renderer-provided; STYLEv3 §8.4) ────────────────────── */
a.xref { color:inherit; text-decoration:underline dotted;
         text-decoration-color:var(--accent, var(--ink-soft));
         text-underline-offset:2px; }
a.xref:hover { text-decoration-style:solid; }
.toc-float { position:fixed; right:18px; bottom:18px; width:44px; height:44px;
             display:flex; align-items:center; justify-content:center;
             background:var(--ink); color:var(--paper); border:2px solid var(--paper);
             box-shadow:3px 3px 0 rgba(0,0,0,.3); font-size:18px;
             text-decoration:none; z-index:60; }
.toc-float:hover { background:var(--power-red); }
.masthead-block .legend-row b { font-family:var(--condensed);
                                letter-spacing:.05em; text-transform:uppercase; }

/* ── Tables ─────────────────────────────────────────────────────────── */
table.data { width:100%; border-collapse:collapse; margin:16px 0; font-size:.9em; }
table.data th { background:var(--ink); color:var(--paper); font-family:var(--condensed);
                text-transform:uppercase; letter-spacing:.08em; font-size:10.5px;
                padding:7px 9px; text-align:left; }
table.data td { border:1px solid #cdc5b5; padding:6px 9px; font-variant-numeric:tabular-nums; }
table.data tr:nth-child(even) td { background:rgba(255,255,255,.45); }

.assumptions { border-left:5px solid var(--tier-data); background:rgba(27,79,216,.06);
               padding:12px 15px; margin:16px 0; font-size:.9em; }
.assumptions ul { margin:6px 0 0; padding-left:20px; }
.todo { background:var(--power-red); color:#fff; font-family:var(--condensed);
        letter-spacing:.1em; padding:2px 8px; text-transform:uppercase; font-size:11px; }

/* ── Responsive & a11y ──────────────────────────────────────────────── */
@media (max-width:820px) {
  .cover-body { grid-template-columns:1fr; }
  .dept { padding:30px 18px 14px; }
  .folio { padding:7px 18px 18px; }
  .violator { position:static; margin:14px auto; }
  .body-copy p:first-of-type::first-letter { font-size:2.4em; }
  /* Touch: no hover — the tap follows the anchor to the tile instead. */
  .card-pop { display:none!important; }
  /* The Flight Plan stacks: title row, then promise + byline beneath. */
  table.toc, table.toc tr { display:block; }
  table.toc td { display:block; border-bottom:none; padding:1px 0; }
  td.toc-title { white-space:normal; width:auto; padding-top:8px; }
  table.toc tr { border-bottom:1px solid #cdc5b5; padding:2px 0 8px; }
  table.toc tr:last-child { border-bottom:none; }
}
@media (prefers-reduced-motion:reduce) { * { transition:none!important; animation:none!important; } }
@media print { .trim { box-shadow:none; } body { background:#fff; }
  .toc-float { display:none; }
  .card-pop { display:none!important; }
  details.dossier > *:not(summary) { display:block; }
  details.dossier > summary::after { content:""; } }
"""


# ── Component builders ──────────────────────────────────────────────────


def badge(tier):
    """Tier badge markup. Costume never earns the badge — STYLEv3 §10."""
    glyphs = {
        "verified": ("✓", "RULES-VERIFIED", "badge-verified"),
        "data": ("◆", "DATA-DERIVED", "badge-data"),
        "coach": ("★", "COACHING", "badge-coach"),
    }
    glyph, label, cls = glyphs[tier]
    return f'<span class="badge {cls}">{glyph} {label}</span>'


def violator(text, plain=False):
    """Starburst violator. Angle is content-derived, so renders identically."""
    angle = stable_angle(text, 9)
    cls = "violator plain" if plain else "violator"
    return (f'<div class="{cls}" style="transform:rotate({angle}deg)">{esc(text)}</div>')


def pilot_tip(card_name, text, image=None, esc_fn=esc):
    """GamePro ProTip formula: image + slug + one imperative sentence.
    `esc_fn` lets the renderer pass its link-aware escaper (esc_x)."""
    img = f'<img src="{esc(image)}" alt="{esc(card_name)}" loading="lazy">' if image else "<div></div>"
    return (
        f'<div class="pilot-tip">{img}<div>'
        f'<span class="slug">Pilot Tip</span>{esc_fn(text)}</div></div>'
    )


def fast_facts(title, pairs):
    """Spec-sheet box: label/value pairs with tabular figures."""
    rows = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in pairs)
    return (
        f'<div class="fast-facts"><div class="ff-head">{esc(title)}</div>'
        f"<dl>{rows}</dl></div>"
    )


def power_meter(label, rate, segments=20):
    """Segmented bar for a ◆ rate. `rate` is 0..1.

    The bar clamps; the printed percentage does not, so an out-of-range rate
    shows a full bar labelled with the real (wrong) number rather than hiding it.
    """
    filled = max(0, min(segments, round(rate * segments)))
    segs = "".join(
        f'<span class="meter-seg{" on" if i < filled else ""}"></span>'
        for i in range(segments)
    )
    return (
        f'<div class="meter"><div class="meter-label"><span>{esc(label)}</span>'
        f"<b>{rate:.0%}</b></div>"
        f'<div class="meter-track">{segs}</div></div>'
    )


def callout(n, title, text, esc_fn=esc):
    """Numbered play-sequence step with an all-caps mini-headline.
    `esc_fn` lets the renderer pass its link-aware escaper (esc_x)."""
    return (
        f'<div class="callout"><div class="n">{esc(n)}</div><div>'
        f'<span class="t">{esc(title)}</span>{esc_fn(text)}</div></div>'
    )



# ── The engine flow ─────────────────────────────────────────────────────
#
# The constellation answers "what shape is this deck". This answers the harder
# one: how does it RUN. They are different relations — a card is clustered by what
# it says and an engine is defined by what cards do to each other — so this is a
# second picture rather than a restyling of the first.
#
# THE SOLID/DASHED DISTINCTION IS THE WHOLE REASON TO DRAW IT. A line between two
# stages is solid when a checker-passed stack names its cards and dashed when it
# is the analyst's claim. That is the three-tier evidence contract rendered as
# geometry: a reader can see, without reading a badge, which parts of the engine
# are proven and which are argued. Everything else here is labelling.

# The engine's running order. Not alphabetical and not the model's array order —
# `validate_engine.STAGES` is the one place this sequence is decided.
ENGINE_STAGE_INK = {
    "mana": "#5B6B8C", "ignition": "#E4002B", "fuel": "#C8A03C",
    "fodder": "#7B5E3C", "conversion": "#7B2D8B", "output": "#1B4FD8",
    "protection": "#0FA3A3", "wincon": "#E4007C",
}

# What each stage HANDS ON — the noun that travels down an arrow leaving it.
#
# This is the difference between a schematic and a block diagram. The first
# version of this figure drew seven boxes and joined them with anonymous arcs,
# which says only "these two are related" — and an engine is not an adjacency
# list, it is a sequence of *conversions*. A reader should be able to follow one
# resource through the machine and watch it become another: mana buys windows,
# windows buy bodies, bodies buy cards, cards buy damage.
#
# An arrow's label is read from the line's own `carries` when the engineer wrote
# one, and derived from its SOURCE stage otherwise. Deriving costs no schema
# migration and no ~500k-token respawn of the engineer/critic loop — and the
# figure's key says which arrows are derived, because an inference wearing an
# authored label is exactly the thing the dashed-line contract exists to prevent.
STAGE_CARRIES = {
    "mana": "mana", "ignition": "windows", "fuel": "bodies",
    "fodder": "fodder", "conversion": "cards", "output": "answers",
    "protection": "insurance", "wincon": "damage",
}

# The plain-language job, printed inside each bay. Three of these are the triad
# every deck has whether or not it has ever named them — the thing that starts
# the machine, the thing that feeds it, and the thing that ends the game — and a
# reader who takes nothing else from the figure should take those.
STAGE_ROLE = {
    "mana": "PAYS FOR IT", "ignition": "STARTS IT", "fuel": "FEEDS IT",
    "fodder": "FEEDS IT", "conversion": "TURNS IT OVER",
    "output": "CLEARS THE WAY", "protection": "KEEPS IT RUNNING",
    "wincon": "ENDS IT",
}


def line_carries(line):
    """`(noun, authored)` — what an arrow moves, and whether anyone said so."""
    authored = (line or {}).get("carries")
    if authored:
        return str(authored), True
    return STAGE_CARRIES.get((line or {}).get("from"), ""), False


def _wrap(text, limit, max_lines):
    """Hand-wrap for SVG, which has no flow. A clipped name reads as a fault."""
    words, line, lines = str(text or "").split(), "", []
    for word in words:
        trial = (line + " " + word).strip()
        if len(trial) > limit and line:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines[:max_lines]


def engine_flow(doc, width=1060):
    """`engine.json` → an inline SVG SCHEMATIC of the engine, left to right.

    Bays in canonical stage order, and between them arrows that each carry a NAMED
    resource. Direction is the whole reading: an arrow running forward (down the
    stage order) arcs ABOVE the rail, and one running backward arcs BELOW it as a
    feedback loop — which is not decoration, because a loop is the thing that makes
    an engine an engine rather than a list of steps. Radagast's `wincon → conversion`
    is exactly that: the finisher pumps the board and the board draws more cards.

    Dashed still means unverified, unchanged and non-negotiable — it is the same
    line the panel is forbidden to assert.
    """
    stages = [s for s in (doc or {}).get("stages") or [] if s.get("stage")]
    if not stages:
        return ""
    from manamap.pilot.validate_engine import STAGES as ORDER
    stages.sort(key=lambda s: ORDER.index(s["stage"]) if s["stage"] in ORDER else 99)

    n = len(stages)
    gap, pad = 14, 18
    box_w = (width - 2 * pad - gap * (n - 1)) / n
    box_h = 146
    # Room above for forward arcs (deepest = widest span) and below for feedback.
    rail = 236
    height = rail + box_h + 128

    rank = {s["stage"]: i for i, s in enumerate(stages)}
    at = {}
    parts = [
        f'<svg class="engine-flow" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Schematic: what this deck converts, stage by stage" '
        f'xmlns="http://www.w3.org/2000/svg">',
        f'<defs><linearGradient id="ef-sky" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="#141329"/>'
        f'<stop offset="1" stop-color="#0B0A14"/></linearGradient></defs>',
        f'<rect width="{width}" height="{height}" fill="url(#ef-sky)"/>',
    ]

    thinnest = min(stages, key=lambda s: len(s.get("cards") or []))

    for i, stage in enumerate(stages):
        x = pad + i * (box_w + gap)
        name = stage["stage"]
        ink = ENGINE_STAGE_INK.get(name, "#8A93B5")
        at[name] = (x + box_w / 2, rail, rail + box_h)
        cards = stage.get("cards") or []
        thin = stage is thinnest
        parts.append(
            f'<rect x="{x:.1f}" y="{rail}" width="{box_w:.1f}" height="{box_h}" '
            f'rx="5" fill="{ink}" fill-opacity="0.17" stroke="{ink}" '
            f'stroke-width="{3 if thin else 1.6}"/>')
        # A tint bar along the top edge, so the bay reads as a labelled unit
        # rather than as a tinted rectangle that happens to have words in it.
        parts.append(
            f'<rect x="{x:.1f}" y="{rail}" width="{box_w:.1f}" height="5" '
            f'rx="2.5" fill="{ink}"/>')
        parts.append(
            f'<text x="{x + 9:.1f}" y="{rail + 26}" fill="{ink}" font-size="10.5" '
            f'font-family="Oswald,Arial Narrow,sans-serif" letter-spacing="1.6">'
            f'{esc(name.upper())}</text>')
        for j, text in enumerate(_wrap(stage.get("label"), max(8, int(box_w / 7.4)), 3)):
            parts.append(
                f'<text x="{x + 9:.1f}" y="{rail + 48 + j * 17}" fill="#F2F0F7" '
                f'font-size="14" font-weight="700" '
                f'font-family="Oswald,Arial Narrow,sans-serif">{esc(text)}</text>')
        role = STAGE_ROLE.get(name)
        if role:
            parts.append(
                f'<text x="{x + 9:.1f}" y="{rail + box_h - 40}" fill="{ink}" '
                f'font-size="10" font-family="Oswald,Arial Narrow,sans-serif" '
                f'letter-spacing="1.3" opacity=".95">{esc(role)}</text>')
        parts.append(
            f'<text x="{x + 9:.1f}" y="{rail + box_h - 23}" fill="#AEB4CC" '
            f'font-size="11.5" font-family="Inter,system-ui,sans-serif">'
            f'{len(cards)} card{"" if len(cards) == 1 else "s"}</text>')
        if stage.get("single_point_of_failure"):
            parts.append(
                f'<text x="{x + 9:.1f}" y="{rail + box_h - 10}" fill="#FFD800" '
                f'font-size="10" font-family="Oswald,Arial Narrow,sans-serif" '
                f'letter-spacing=".6">ONE CARD DEEP</text>')
        if thin:
            parts.append(
                f'<text x="{x + box_w - 9:.1f}" y="{rail + 26}" fill="{ink}" '
                f'text-anchor="end" font-size="10" '
                f'font-family="Oswald,Arial Narrow,sans-serif">THINNEST</text>')

    # ── The flows ────────────────────────────────────────────────────────────
    # Depth scales with span so a long arrow clears the short ones under it.
    #
    # `seen` separates arrows that share a pair AND a direction, which is not a
    # hypothetical: radagast declares `fuel → wincon` twice — once through Hornet
    # Nest and once without it, resting on two different stacks — and without the
    # offset they draw at identical coordinates. Two distinct proven lines would
    # render as one, and the figure's own count would then disagree with the
    # number of arrows a reader can find. Everything here is deterministic;
    # nothing is measured at view time.
    seen = {}
    for line in doc.get("lines") or []:
        a, b = at.get(line.get("from")), at.get(line.get("to"))
        if not a or not b or a[0] == b[0]:
            continue
        verified = bool(line.get("verified_by"))
        stroke = "#4CAF50" if verified else "#8A93B5"
        forward = rank.get(line.get("to"), 0) > rank.get(line.get("from"), 0)
        span = abs(rank.get(line.get("to"), 0) - rank.get(line.get("from"), 0))
        noun, authored = line_carries(line)

        key = (line.get("from"), line.get("to"))
        nth = seen.get(key, 0)
        seen[key] = nth + 1

        if forward:
            y0 = y1 = a[1]                       # leave and arrive at the bay tops
            apex = max(22, rail - 26 - span * 33 + nth * 21)
            head = f"{b[0]:.1f},{y1 - 1} {b[0] - 5:.1f},{y1 - 11} {b[0] + 5:.1f},{y1 - 11}"
        else:
            y0 = y1 = a[2]                       # …or at the bay bottoms
            apex = rail + box_h + 24 + (span + nth) % 3 * 27
            head = f"{b[0]:.1f},{y1 + 1} {b[0] - 5:.1f},{y1 + 11} {b[0] + 5:.1f},{y1 + 11}"

        parts.append(
            f'<path d="M {a[0]:.1f} {y0} C {a[0]:.1f} {apex:.1f}, '
            f'{b[0]:.1f} {apex:.1f}, {b[0]:.1f} {y1}" fill="none" '
            f'stroke="{stroke}" stroke-width="{2.2 if verified else 1.4}" '
            f'stroke-opacity="{0.9 if verified else 0.5}"'
            + ('' if verified else ' stroke-dasharray="5 4"') + '/>')
        parts.append(f'<polygon points="{head}" fill="{stroke}" '
                     f'fill-opacity="{0.9 if verified else 0.5}"/>')

        if noun:
            # The bezier's apex sits at 3/4 of the control offset, not at the
            # control point — putting the label on the control point floats it
            # clear of a line it is supposed to be labelling.
            mid_x = (a[0] + b[0]) / 2
            mid_y = y0 + 0.75 * (apex - y0)
            w = 7.0 * len(noun) + 14
            parts.append(
                f'<rect x="{mid_x - w / 2:.1f}" y="{mid_y - 10:.1f}" width="{w:.1f}" '
                f'height="17" rx="8.5" fill="#0B0A14" fill-opacity=".92" '
                f'stroke="{stroke}" stroke-width=".9" stroke-opacity=".55"/>')
            parts.append(
                f'<text x="{mid_x:.1f}" y="{mid_y + 2.5:.1f}" text-anchor="middle" '
                f'fill="{"#DDE3F2" if authored else "#A8B0C8"}" font-size="10.5" '
                f'font-family="Inter,system-ui,sans-serif"'
                + ('' if authored else ' font-style="italic"')
                + f'>{esc(noun)}</text>')

    parts.append("</svg>")
    return "".join(parts)


def engine_figure(doc, caption=""):
    """The schematic, its key, and the honesty line under it."""
    svg = engine_flow(doc)
    if not svg:
        return ""
    lines = doc.get("lines") or []
    proven = sum(1 for l in lines if l.get("verified_by"))
    derived = sum(1 for l in lines if not (l or {}).get("carries"))
    keys = ('<span class="ck"><i class="eline on"></i>Proven by a rules-verified line</span>'
            '<span class="ck"><i class="eline"></i>The analyst\'s reading, not yet verified</span>'
            '<span class="ck"><i class="dot cmdr"></i>One card deep</span>')
    # An arrow label the engineer did not write is set in italic and said out loud
    # here. It is a reasonable inference — an arrow leaving FUEL moves bodies — but
    # it is still an inference, and this magazine marks those.
    note = ("" if not derived else
            f' {derived} arrow label{"" if derived == 1 else "s"} '
            f'{"is" if derived == 1 else "are"} set in italic: the resource is '
            f"inferred from the stage it leaves, not stated by the analyst.")
    return (f'<figure class="engine-fig">{svg}'
            f'<div class="ckeys">{keys}</div>'
            f'<figcaption>{esc(caption or "")} '
            f'{proven} of {len(lines)} connections rest on a checker-passed stack.'
            f'{note}</figcaption></figure>')


# ── The deck constellation ──────────────────────────────────────────────
#
# One deck, re-laid-out from its own cards and clustered into cities. Drawn as
# inline SVG rather than as a canvas or a chart library, for three reasons that
# all matter here: the magazine must rebuild byte-identically (so no layout may be
# computed at view time), the page must print, and an issue is a standalone file
# with no scripts. Everything below is pure geometry over `deck_map.json`.
#
# The register is the poster, not the scatter plot. A card is a dot, a city is a
# lit blob with its name across it, and the eye is meant to land on the SHAPE
# before it reads anything — density first, structure second, labels third.

# A lobe whose RMS spread exceeds this multiple of the deck's MEDIAN lobe spread
# is drawn as points with no territory. See `deck_constellation`.
DIFFUSE_LOBE_RATIO = 2.5

# Seven, so the largest deck's cities each get one and no two neighbours collide.
# Ordered by how loud they are: city-0 is the biggest cluster and takes the
# strongest colour, which is also the order `deck_map` emits them in.
CITY_INK = [
    ("#E4007C", "#FF66B8"),   # magenta
    ("#1B4FD8", "#6E93F5"),   # blue
    ("#3FBF3F", "#8BE28B"),   # green
    ("#C8A03C", "#EBD08A"),   # gold
    ("#7B2D8B", "#B77BC4"),   # purple
    ("#E4002B", "#FF7A93"),   # red
    ("#0FA3A3", "#68DADA"),   # teal
]


def _hull(points):
    """Convex hull, monotone chain. Returns the hull in order, or the input.

    Hand-rolled rather than scipy: `design.py` renders and must import cheaply,
    and a hull over at most ~35 points is fifteen lines. scipy also raises on
    degenerate input (three collinear cards), which here is a normal deck.
    """
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _blob(points, pad):
    """A rounded, outward-padded hull as an SVG path.

    A raw convex hull reads as a chart annotation — straight edges, sharp corners,
    obviously computed. Pushing each vertex out from the centroid and joining them
    with quadratic curves through the edge midpoints gives a soft territory, which
    is what a map region looks like and what a reader recognises without a key.
    """
    hull = _hull(points)
    if len(hull) < 3:
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        r = pad + max((math.hypot(p[0] - cx, p[1] - cy) for p in points), default=0)
        return (f"M {cx - r:.1f} {cy:.1f} a {r:.1f} {r:.1f} 0 1 0 {2 * r:.1f} 0 "
                f"a {r:.1f} {r:.1f} 0 1 0 {-2 * r:.1f} 0 Z")
    cx = sum(p[0] for p in hull) / len(hull)
    cy = sum(p[1] for p in hull) / len(hull)
    out = []
    for x, y in hull:
        dx, dy = x - cx, y - cy
        length = math.hypot(dx, dy) or 1.0
        out.append((x + dx / length * pad, y + dy / length * pad))
    mids = [((out[i][0] + out[(i + 1) % len(out)][0]) / 2,
             (out[i][1] + out[(i + 1) % len(out)][1]) / 2) for i in range(len(out))]
    path = [f"M {mids[-1][0]:.1f} {mids[-1][1]:.1f}"]
    for i, (vx, vy) in enumerate(out):
        path.append(f"Q {vx:.1f} {vy:.1f} {mids[i][0]:.1f} {mids[i][1]:.1f}")
    return " ".join(path) + " Z"


def deck_constellation(doc, width=1060, height=760):
    """`deck_map.json` → one inline SVG.

    Layers, back to front, and the order is the argument: density (what the deck
    is made of), then structure (what sits beside what), then the cards, then the
    names. A reader who looks for one second should get the shape; a reader who
    looks for ten should get the cities; only then does a card name matter.
    """
    cards = doc.get("cards") or []
    if not cards:
        return ""
    regions = doc.get("regions") or []
    cities = [r for r in regions if r.get("level") == 0]

    pad = 74
    xs = [c["x"] for c in cards]
    ys = [c["y"] for c in cards]
    span_x = (max(xs) - min(xs)) or 1.0
    span_y = (max(ys) - min(ys)) or 1.0
    scale = min((width - 2 * pad) / span_x, (height - 2 * pad) / span_y)
    ox = (width - span_x * scale) / 2 - min(xs) * scale
    oy = (height - span_y * scale) / 2 - min(ys) * scale

    def place(card):
        return (card["x"] * scale + ox, card["y"] * scale + oy)

    pts = [place(c) for c in cards]
    by_city = {}
    for card, point in zip(cards, pts):
        by_city.setdefault(card["city"], []).append(point)

    # Territories are drawn per NEIGHBOURHOOD, not per city, and that is the whole
    # difference between a map and a smear. A city's convex hull is only tight when
    # its members happen to be adjacent in the projection; radagast's 23-card city
    # is spread right across the frame, so its hull covered every other city and
    # the picture read as one magenta continent with labels floating on it. Its
    # three neighbourhoods are compact, and a city drawn as a few lobes in one
    # colour reads as territory — which is also truer, because the neighbourhoods
    # are what the clustering actually found.
    lobes = {}
    for card, point in zip(cards, pts):
        lobes.setdefault((card["city"], card.get("hood", 0)), []).append(point)

    # A DIFFUSE lobe gets no territory, and that is a finding rather than a
    # rendering compromise. Radagast's largest lobe has an RMS spread of 0.761
    # against a 0.153 median — 5x — and hulling it drew a magenta continent down
    # the middle of the frame that contained three other cities and overlapped a
    # fourth. There is no territory there: those ten cards are genuinely spread
    # across the whole deck, and a hull asserting otherwise is the picture lying.
    #
    # Threshold is relative to the deck's own median lobe, not absolute, because
    # the spreads differ 3x across decks (sisay 0.084, yawgmoth 0.225) while the
    # RATIO separates cleanly everywhere. Measured on all nine: 1–4 diffuse lobes
    # per deck, never a majority.
    def _rms(points):
        cx = sum(q[0] for q in points) / len(points)
        cy = sum(q[1] for q in points) / len(points)
        return math.sqrt(sum((q[0]-cx)**2 + (q[1]-cy)**2
                             for q in points) / len(points))

    spreads = {k: _rms(v) for k, v in lobes.items()}
    ordered = sorted(spreads.values())
    median = ordered[len(ordered) // 2] if ordered else 0.0
    # Lobes under four points are exempt: with two or three members the RMS is
    # measuring a line segment, not a shape, and a pair that happens to straddle
    # the frame would lose a territory that is genuinely just "these two cards".
    solid = {k: v for k, v in lobes.items()
             if spreads[k] <= DIFFUSE_LOBE_RATIO * median or len(v) < 4}

    def trimmed(points, keep=0.85):
        """Drop the farthest few from the centroid before hulling.

        One card sitting out past the rest stretches a hull across empty space it
        does not occupy. The card is still DRAWN — it is just not allowed to claim
        territory on its own, which is the same judgment a cartographer makes about
        a lighthouse.
        """
        if len(points) < 5:
            return points
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        ranked = sorted(points, key=lambda p: math.hypot(p[0] - cx, p[1] - cy))
        return ranked[:max(3, int(len(ranked) * keep))]

    def ink(city, shade=0):
        return CITY_INK[city % len(CITY_INK)][shade]

    parts = [
        f'<svg class="constellation" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Cluster map of this deck\'s cards" '
        f'xmlns="http://www.w3.org/2000/svg">',
        '<defs><filter id="cbloom" x="-30%" y="-30%" width="160%" height="160%">'
        '<feGaussianBlur stdDeviation="16"/></filter></defs>',
        f'<rect width="{width}" height="{height}" fill="#0B0A14"/>',
    ]

    # 1. Density — the blurred territory of each city, additively lit.
    parts.append('<g filter="url(#cbloom)" opacity="0.5">')
    for (city, _hood), members in sorted(solid.items()):
        parts.append(f'<path d="{_blob(trimmed(members), 30)}" fill="{ink(city)}"/>')
    parts.append("</g>")

    # 2. The territory outline, so each lobe has an edge to read against.
    for (city, _hood), members in sorted(solid.items()):
        parts.append(f'<path d="{_blob(trimmed(members), 22)}" fill="{ink(city)}" '
                     f'fill-opacity="0.14" stroke="{ink(city, 1)}" '
                     f'stroke-opacity="0.42" stroke-width="1.4"/>')

    # 3. Structure — within-deck nearest neighbours. Same-city edges take the
    #    city's colour, cross-city edges stay pale: an edge that leaves its
    #    cluster is the interesting one and should read as a bridge, not as noise.
    for edge in doc.get("edges") or []:
        a, b = edge["a"], edge["b"]
        if a >= len(pts) or b >= len(pts):
            continue
        same = cards[a]["city"] == cards[b]["city"]
        colour = ink(cards[a]["city"], 1) if same else "#8A93B5"
        parts.append(
            f'<line x1="{pts[a][0]:.1f}" y1="{pts[a][1]:.1f}" '
            f'x2="{pts[b][0]:.1f}" y2="{pts[b][1]:.1f}" stroke="{colour}" '
            f'stroke-opacity="{0.38 if same else 0.22}" stroke-width="1"/>')

    # 4. The cards.
    for card, (x, y) in zip(cards, pts):
        colour = ink(card["city"], 1)
        if card.get("commander"):
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="none" '
                f'stroke="#FFD800" stroke-width="2.5"/>'
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#FFD800"/>')
        elif card.get("verified"):
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{colour}" '
                f'stroke="#FFFFFF" stroke-width="1.6"/>')
        else:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" '
                         f'fill="{colour}" fill-opacity="0.95"/>')

    # WHERE a city's name goes: the centroid of its tightest territory, not of all
    # its members. THE WIDE BOARD holds 23 cards, ten of which are spread across
    # the whole frame — averaging those pulls the name off every lobe it belongs to
    # and drops it into black space, which reads as a label for the emptiness. The
    # solid lobes are the parts of the city that ARE somewhere; name it there.
    def anchor_of(city):
        own = [(k, v) for k, v in solid.items() if k[0] == city]
        pool = max(own, key=lambda kv: len(kv[1]))[1] if own else by_city.get(city, [])
        if not pool:
            return None
        return (sum(q[0] for q in pool) / len(pool),
                sum(q[1] for q in pool) / len(pool))

    # 5. CARD NAMES, for the cards a reader would want to find.
    #
    # The web version has hover and the printed one does not, so before this the
    # magazine showed 71 anonymous dots and a reader could learn only that there
    # were seven groups — which the grid below already says, better. A map whose
    # points cannot be identified is decoration.
    #
    # Labelled, in priority order: the commander, then every card a checker-passed
    # stack names. That is ~14 of 71 here, which is the most a frame this size
    # holds. Greedy first-come placement with axis-aligned rejection, priority
    # first — the same rule `Stage.placer` uses on the atlas, because a label that
    # collides is worse than a label that is absent.
    taken = []

    def claim(box):
        for other in taken:
            if (box[0] < other[2] + 3 and box[2] > other[0] - 3
                    and box[1] < other[3] + 3 and box[3] > other[1] - 3):
                return False
        taken.append(box)
        return True

    # City names claim their space FIRST — they are the legend and outrank a card.
    for region in cities:
        at = anchor_of(int(region["id"].rsplit("-", 1)[-1]))
        if at:
            w = len(region.get("label") or region.get("fallback") or "") * 11.5
            claim((at[0] - w / 2, at[1] - 22, at[0] + w / 2, at[1] + 30))

    unplaced = []
    named = [(c, p) for c, p in zip(cards, pts)
             if c.get("commander") or c.get("verified")]
    named.sort(key=lambda cp: (not cp[0].get("commander"), cp[0]["name"]))
    for card, (x, y) in named:
        short = card["name"].split(" // ")[0].split(",")[0]
        if len(short) > 22:
            short = short[:21] + "…"
        w, h = len(short) * 5.6, 13
        # Eight candidate positions, nearest first. Two (below, then above) placed
        # only 6 of 13 on radagast — the load-bearing cards are load-bearing partly
        # BECAUSE they sit where the deck is dense, so the naive positions are
        # exactly the contested ones. Side placements are anchored start/end so the
        # text runs away from the dot rather than back across it.
        placed = False
        for dx, dy, anchor in ((0, 13, "middle"), (0, -17, "middle"),
                               (7, 4, "start"), (-7, 4, "end"),
                               (0, 25, "middle"), (0, -29, "middle"),
                               (7, -12, "start"), (-7, -12, "end")):
            left = x + dx if anchor == "start" else (
                x - w if anchor == "end" else x - w / 2)
            box = (left, y + dy - h, left + w, y + dy)
            if claim(box):
                parts.append(
                    f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}" '
                    f'font-family="Inter,system-ui,sans-serif" font-size="10.5" '
                    f'stroke="#0B0A14" stroke-width="3.5" paint-order="stroke" '
                    f'fill="#E8E6F0">{esc(short)}</text>')
                placed = True
                break
        if not placed:
            # The FULL name, not the shortened label. `short` exists because a
            # dot has ~120px beside it; a prose note has a line, and "Toski"
            # where the deck lists "Toski, Bearer of Secrets" reads as a
            # different card.
            unplaced.append(card["name"].split(" // ")[0])

    # 6. City names, across their own territory. Drawn last so nothing covers
    #    them, with a dark halo because they sit on top of the brightest ink.
    for region in cities:
        city = int(region["id"].rsplit("-", 1)[-1])
        at = anchor_of(city)
        if not at:
            continue
        cx, cy = at
        label = (region.get("label") or region.get("fallback") or "").upper()
        parts.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" '
            f'font-family="Oswald,Arial Narrow,sans-serif" font-size="21" '
            f'font-weight="700" letter-spacing="1.6" '
            f'stroke="#0B0A14" stroke-width="5" paint-order="stroke" '
            f'fill="{ink(city, 1)}">{esc(label)}</text>'
            f'<text x="{cx:.1f}" y="{cy + 17:.1f}" text-anchor="middle" '
            f'font-family="Inter,system-ui,sans-serif" font-size="11.5" '
            f'stroke="#0B0A14" stroke-width="4" paint-order="stroke" '
            f'fill="#CFD3E6">{region["count"]} cards</text>')

    parts.append("</svg>")
    # Render facts the caption has to account for, not decoration: a reader who
    # counts territories and compares against the caption's neighbourhood count
    # must not find a discrepancy nothing explains.
    diffuse = [k for k in lobes if k not in solid]
    return "".join(parts), {
        "unplaced": unplaced,
        "diffuse": len(diffuse),
        "diffuse_cards": sum(len(lobes[k]) for k in diffuse),
    }


def city_head(index, name, count, verified=0, gloss=None):
    """A 99 group heading that matches its territory on the map.

    The colour is the load-bearing part. A grid grouped by city under plain black
    headings is a second taxonomy the reader has to learn and reconcile; carrying
    the map's ink onto the heading makes the grid the map's LEGEND, which is what
    it is for. Same index, same colour, by construction — both read `CITY_INK`.
    """
    ink, light = CITY_INK[index % len(CITY_INK)]
    seal = (f'<span class="city-verified" title="named in a verified line">'
            f'✓{verified}</span>' if verified else "")
    head = (f'<h3 class="city-head" style="--city:{ink};--city-lt:{light}">'
            f'<span class="city-chip"></span>{esc(name)}'
            f'<span class="city-count">{count} cards</span>{seal}</h3>')
    # The gloss says what the city is FOR. The grid beneath already shows what is
    # in it, so a gloss that lists contents is a wasted line.
    return head + (f'<p class="city-gloss">{esc(gloss)}</p>' if gloss else "")


def constellation_figure(doc, caption):
    """The constellation with its legend and the honesty line under it."""
    svg, notes = deck_constellation(doc)
    if not svg:
        return ""
    keys = (
        '<span class="ck"><i class="dot cmdr"></i>Commander</span>'
        '<span class="ck"><i class="dot ver"></i>Named in a verified line</span>'
        '<span class="ck"><i class="dot plain"></i>Everything else</span>'
        '<span class="ck"><i class="edge"></i>Nearest neighbour in this deck</span>'
    )
    # Anything the frame could not hold is NAMED anyway. A load-bearing card that
    # lost a collision is still load-bearing, and silently dropping it makes the
    # map's own key a lie — it says verified cards are marked, and three of them
    # would not be. Placement is a constraint on the picture, not on the facts.
    tail = ""
    if notes["diffuse"]:
        n, cards = notes["diffuse"], notes["diffuse_cards"]
        tail += (
            f'<p class="cunplaced">'
            f'{"One neighbourhood is" if n == 1 else str(n) + " neighbourhoods are"}'
            f' drawn without a territory: {"its" if n == 1 else "their"} '
            f'{cards} cards are spread across the whole deck rather than gathered '
            f'anywhere, so an outline would claim ground that is not there.</p>')
    if notes["unplaced"]:
        tail += ('<p class="cunplaced">Also named in a verified line, too close '
                 f'to label here: {esc(", ".join(sorted(notes["unplaced"])))}.</p>')
    return (f'<figure class="constellation-fig">{svg}'
            f'<div class="ckeys">{keys}</div>{tail}'
            f'<figcaption>{esc(caption)}</figcaption></figure>')


# A named constant, not an inline literal: `coach_gauge` escapes its own label, so
# pre-escaping it here rendered "COACH&#x27;S READ" on the page — and the obvious
# fix (a backslash-escaped apostrophe inside the f-string) is a SyntaxError on
# Python 3.10, which is the version this project is pinned to.
def letterhead(masthead, slug, volume_line, body_html, card=None,
               teases=(), signature="", role=""):
    """The Editor's Letter's own furniture — and the only place it is used.

    It shipped as `.body-copy` with a sign-off, which was editorially right and
    visually invisible: an editor's page that looks like the article after it is
    not an editor's page, and a reader flipping through met one more column of
    type where a magazine puts its most recognisable furniture. Four devices do
    the work, and they are deliberately redundant — tinted stock, a heavy top
    rule, the wordmark, and two columns. Any one alone reads as a styling
    accident; together they say "this is the front of the book" before a word of
    it is read.

    `card` is a `cards.json` record — the letter opens on something you can see.
    `teases` are `(department title, line)` pairs for the IN THIS ISSUE rail,
    which is what makes the page a preview of the magazine rather than a note
    attached to it. Everything is optional; the letter degrades to the wordmark,
    the copy and the signature.

    `body_html` arrives already escaped and linkified from the caller — the same
    contract `card_figure` takes, for the same reason: card links and stack refs
    are the renderer's job and must survive into this component.
    """
    top = [f'<div><div class="lh-mark">{esc(masthead)}</div>'
           f'<div class="lh-slug">{esc(slug)}</div></div>']
    if volume_line:
        top.append(f'<div class="lh-vol">{esc(volume_line)}</div>')

    aside = []
    if card and (card.get("art_crop") or card.get("image")):
        image = card.get("art_crop") or card.get("image")
        img = (f'<img src="{esc(image)}" alt="{esc(card.get("name", ""))}" '
               f'loading="lazy">')
        if card.get("foil"):
            img = f'<span class="foil">{img}</span>'
        aside.append(f'<figure class="lh-card">{img}'
                     f'<figcaption>{esc(card.get("name", ""))}</figcaption>'
                     f'{printing_credit(card)}</figure>')
    rows = "".join(f"<li><b>{esc(title)}</b>{esc(line)}</li>"
                   for title, line in (teases or []) if title)
    if rows:
        aside.append(f'<div class="lh-rail"><h4>In this issue</h4>'
                     f"<ol>{rows}</ol></div>")

    sign = ""
    if signature:
        sign = (f'<div class="lh-sign"><div class="lh-hand">{esc(signature)}</div>'
                + (f'<div class="lh-role">{esc(role)}</div>' if role else "")
                + "</div>")

    return (f'<aside class="letterhead"><div class="lh-top">{"".join(top)}</div>'
            f'<div class="lh-body"><div><div class="lh-copy">{body_html}</div>'
            f'{sign}</div><div>{"".join(aside)}</div></div></aside>')


# The most steps any one resolution may be stepped through. The corpus tops out at
# 24 (yawgmoth-swarm's undying loop); the bound is the number of generic CSS rules
# the stylesheet carries, and a stack that exceeds it still renders every step —
# it simply stops offering tabs past this one. Measured, not guessed: raise it
# when a real artifact passes it, and `stack_theatre` says so on the page.
THEATRE_MAX_STEPS = 28


def _theatre_grid(seed):
    """The backdrop: a vanishing-point grid under a lit horizon.

    1980s vector geometry drawn with 2020s rendering — the lines converge and the
    horizon glows, but there is no dither, no bevel and no chrome. The period look
    is in the perspective, not in the artefacts of the hardware that used to draw
    it, which is the same argument STYLEv3 §L9 makes about every other device here.

    Deterministic: the accent rotates off a hash of the stack id, so a given case
    always renders identically and two adjacent cases do not look like reprints.
    """
    hue = int((stable_angle(seed, 180.0) + 180.0)) % 360
    vx, vy = 500.0, 168.0                      # the vanishing point
    parts = [
        '<svg class="th-grid" viewBox="0 0 1000 420" preserveAspectRatio="none" '
        'aria-hidden="true" xmlns="http://www.w3.org/2000/svg">',
        f'<defs><linearGradient id="thg-{esc(seed)}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="hsl({hue},58%,9%)"/>'
        f'<stop offset="0.40" stop-color="hsl({hue},64%,17%)"/>'
        f'<stop offset="0.42" stop-color="hsl({(hue + 40) % 360},72%,26%)"/>'
        f'<stop offset="1" stop-color="hsl({(hue + 200) % 360},70%,7%)"/>'
        f'</linearGradient>'
        f'<radialGradient id="thh-{esc(seed)}" cx="0.5" cy="0.40" r="0.42">'
        f'<stop offset="0" stop-color="hsl({(hue + 40) % 360},92%,64%)" '
        f'stop-opacity=".55"/>'
        f'<stop offset="1" stop-color="hsl({(hue + 40) % 360},92%,64%)" '
        f'stop-opacity="0"/></radialGradient></defs>',
        f'<rect width="1000" height="420" fill="url(#thg-{esc(seed)})"/>',
        f'<rect width="1000" height="420" fill="url(#thh-{esc(seed)})"/>',
    ]
    ink = f"hsl({(hue + 40) % 360},90%,68%)"
    # Radials: fanned about the vanishing point and run to the bottom edge.
    for i in range(-9, 10):
        x_end = vx + i * 132.0
        parts.append(f'<line x1="{vx}" y1="{vy}" x2="{x_end:.1f}" y2="420" '
                     f'stroke="{ink}" stroke-width="1.1" stroke-opacity=".30"/>')
    # Horizontals: spaced so the gaps compress toward the horizon, which is what
    # actually reads as depth — evenly spaced lines read as a ladder.
    for i in range(1, 13):
        t = (i / 12.0) ** 2.35
        y = vy + 4 + t * (420 - vy - 4)
        parts.append(f'<line x1="0" y1="{y:.1f}" x2="1000" y2="{y:.1f}" '
                     f'stroke="{ink}" stroke-width="1.1" '
                     f'stroke-opacity="{0.13 + 0.30 * t:.2f}"/>')
    parts.append(f'<line x1="0" y1="{vy}" x2="1000" y2="{vy}" stroke="{ink}" '
                 f'stroke-width="2" stroke-opacity=".85"/>')
    parts.append("</svg>")
    return "".join(parts)


def stack_theatre(sid, steps, cards=(), esc_fn=esc):
    """A stack you move through: the resolution, one plate per step.

    The Kill rendered its lines as a definition list, which is a strange thing to
    do to the most visual object in the game. Here the resolution IS the stack —
    plate `n` is step `n`, receding along the vanishing line, and the reader brings
    each one forward in turn to read what happened and which rules said so.

    **No JavaScript, deliberately.** An issue is a standalone printable file that
    rebuilds byte-identically (see the constellation's note), so the step-through
    is radio inputs and `:checked ~` selectors, and the depth is `preserve-3d`. The
    first step is `checked` in the markup, so a reader with CSS disabled, a printer,
    and a screen reader all get a valid view rather than a broken one. The generic
    per-index rules live in the stylesheet (`_theatre_rules`), because per-instance
    CSS would mean a `<style>` block inside every case and the same rules N times.

    `cards` is the deck's card list; a step whose action names one shows its art.
    That match is the plainest possible one — first probe found, longest name first
    — because a plate showing the wrong card is worse than a plate showing none.
    """
    steps = [s for s in (steps or []) if s]
    if not steps:
        return ""
    probes = sorted(((c.get("name") or "", c) for c in cards or []),
                    key=lambda pair: (-len(pair[0]), pair[0]))
    tabbed = steps[:THEATRE_MAX_STEPS]

    inputs, tabs, plates, notes = [], [], [], []
    for i, step in enumerate(steps, start=1):
        action = str(step.get("action") or "")
        card = next((c for name, c in probes if name and name in action), None)
        if i <= len(tabbed):
            rid = f"th-{sid}-{i}"
            inputs.append(f'<input class="th-in" type="radio" name="th-{sid}" '
                          f'id="{rid}"{" checked" if i == 1 else ""}>')
            tabs.append(f'<label class="th-tab" for="{rid}" '
                        f'title="Step {i}">{i}</label>')
            art = card.get("art_crop") or card.get("image") if card else None
            plates.append(
                f'<article class="th-plate" style="--i:{i - 1}">'
                + (f'<img src="{esc(art)}" alt="" loading="lazy">' if art else "")
                + f'<div class="th-n">{i}</div>'
                + f'<h5>{esc(card["name"] if card else f"Step {i}")}</h5>'
                + f'<p>{esc(action)}</p></article>')
        cites = "".join(
            f'<div class="cite"><b>CR {esc(c.get("rule", ""))}</b> — '
            f'“{esc(c.get("quote", ""))}”</div>'
            for c in step.get("citations") or [])
        notes.append(
            f'<div class="th-note"><div class="th-note-n">Step {i} of {len(steps)}'
            + (f' · {esc(card["name"])}' if card else "")
            + f'</div><b>{esc_fn(action)}</b>'
            f'<div class="effect">{esc_fn(str(step.get("effect") or ""))}</div>'
            f"{cites}</div>")

    over = len(steps) - len(tabbed)
    more = ("" if not over else
            f'<p class="th-over">{over} further step'
            f'{"" if over == 1 else "s"} follow{"s" if over == 1 else ""} and '
            f"{'is' if over == 1 else 'are'} printed in full below.</p>")
    return (
        f'<div class="theatre" id="th-{esc(sid)}" style="--n:{len(tabbed)}">'
        f'{"".join(inputs)}'
        f'<div class="th-stage">{_theatre_grid(sid)}'
        f'<div class="th-deck">{"".join(plates)}</div></div>'
        # The label is a SIBLING of the tab list, never a child of it. Inside the
        # nav it is child 1, so every `:nth-child(I)` rule lands one tab early —
        # which rendered as "step 4 is showing and tab 3 is lit", a bug that looks
        # like a rounding error and is actually a selector counting furniture.
        f'<div class="th-railwrap"><span class="th-rail-lbl">Resolve</span>'
        f'<nav class="th-rail" aria-label="Steps in this resolution">'
        f'{"".join(tabs)}</nav></div>'
        f'<div class="th-body">{"".join(notes)}</div>{more}</div>')


def _theatre_rules(n=THEATRE_MAX_STEPS):
    """The per-index `:checked ~` rules — generated, never hand-written.

    Three states and nothing arithmetic: CSS cannot compute "the plate before the
    checked one", so each index gets its own selectors. `:nth-child(-n+I)` is the
    run up to I and `:nth-child(I)` is singled out of it, which is how a plate can
    be *resolved* (behind you) rather than merely *not current*.
    """
    out = []
    for i in range(1, n + 1):
        at = f".th-in:nth-of-type({i}):checked"
        out.append(f"{at} ~ .th-body .th-note:nth-child({i}){{display:block}}")
        out.append(f"{at} ~ .th-railwrap .th-tab:nth-child({i}){{"
                   f"background:var(--burst-yellow);color:var(--ink);"
                   f"transform:translateY(-2px)}}")
        out.append(f"{at} ~ .th-stage .th-plate:nth-child({i}){{"
                   f"transform:translate3d(calc(var(--off) * 13px),"
                   f"calc(var(--off) * -8px),calc(var(--i) * -44px + 150px))"
                   f" rotateY(-4deg);opacity:1;z-index:99}}")
        out.append(f"{at} ~ .th-stage .th-plate:nth-child(-n+{i})"
                   f":not(:nth-child({i})){{opacity:.24;filter:saturate(.35)}}")
    return "\n".join(out)


# The stylesheet is finalised HERE, not where CSS is opened: the theatre's rules
# are generated and `_theatre_rules` has to exist first. Everything downstream
# (`stylesheet_version`, `write_stylesheet`) reads this module attribute at call
# time, so the content hash covers the generated block like any other rule.
CSS += "\n/* ── The stack theatre: generated step rules ── */\n" + _theatre_rules() + "\n"


def hot_take(text, byline, esc_fn=esc):
    """The Coach's opener — a counter-intuitive claim that happens to be true.

    The Pilot's Log needs a reason to be a conversation rather than three essays
    printed adjacently, and a disagreement is that reason. The hot take supplies
    one: Sunny states something about this deck that sounds wrong and is not, the
    Counselor tests whether it is actually on the record, Ledger prices it, and
    where the panel ends up is wherever that argument goes.

    Run full width and set large, because a claim printed at body size is an
    opinion and a claim printed at this size is a position someone has to answer.
    It carries ★ and only ★ — the take is a judgment however correct it turns out
    to be, and a dashed engine line is one Sunny may not assert here either. That
    rule is the panel's, not the renderer's, and the charter is where it is kept.
    """
    if not text:
        return ""
    return ('<aside class="hot-take"><div class="ht-burst">Hot take</div>'
            f'<blockquote>{esc_fn(text)}</blockquote>'
            f'<div class="ht-by">{badge("coach")} {esc(byline)}</div></aside>')


def not_modelled_rail(items, open_questions=(), scope=None, esc_fn=esc):
    """The Game Plan's conditions — what the thesis quietly assumes away.

    A deck's plan is stated as arithmetic ("five bodies is forty trample damage")
    and the arithmetic is correct, which is exactly what makes it dangerous: it is
    a number computed on an empty table. One chump blocker, one instant, one
    opponent who untaps and it is not a kill, and the reader has been handed a
    result dressed as a plan. The rail is where the thesis says what it assumed.

    Derived, not authored, so no deck can quietly skip it. `items` are the plan's
    own additions and lead; `open_questions` are the engine model's OWN admissions
    — the questions its analyst could not settle — and nothing else in the magazine
    prints them, so they cost no repetition. Capped at three with the true count
    stated: a rail long enough to skim past is a rail nobody reads, and a silent
    truncation would read as "that's all of them", which is the failure this whole
    component exists to correct.

    `scope` is the simulator's one-sentence limit. The nine full model assumptions
    stay in By the Numbers, under the byline that earned them — repeating the set
    here would sand down the best admission in the issue by saying it twice.
    """
    rows = [f"<li>{esc_fn(text)}</li>" for text in (items or []) if text]
    shown = [q for q in (open_questions or []) if (q or {}).get("question")][:3]
    for q in shown:
        rows.append(f'<li><span class="nm-src">Unsettled</span> '
                    f'{esc_fn(q["question"])}</li>')
    if scope:
        rows.append(f'<li><span class="nm-src">Simulation</span> {esc_fn(scope)}</li>')
    if not rows:
        return ""
    total = len([q for q in (open_questions or []) if (q or {}).get("question")])
    more = ("" if total <= len(shown) else
            f'<p class="nm-more">{total - len(shown)} further question'
            f'{"" if total - len(shown) == 1 else "s"} about this engine '
            f"{'is' if total - len(shown) == 1 else 'are'} open and unsettled.</p>")
    return ('<aside class="not-modelled"><h4>What this does not model</h4>'
            f'<ul>{"".join(rows)}</ul>{more}</aside>')


COACH_READ = "Coach's read"


def stat_slab(figure, label, note=""):
    """The issue's signature number, full width, once.

    Vol. 009's best finding — 36 lands is not 40 — appeared in six departments.
    Repetition does not emphasise a fact; it wears it down, and by the third
    restatement the reader is skipping the sentence that contains it. Print it
    once at the size it deserves and let every later mention refer back.

    `figure` is set as display type and is never escaped into a paragraph: it is a
    number or a very short phrase ("36", "42%", "4 WINDOWS"), not a sentence.
    """
    return (
        '<div class="stat-slab">'
        f'<div class="ss-figure">{esc(figure)}</div>'
        f'<div class="ss-label">{esc(label)}</div>'
        + (f'<div class="ss-note">{esc(note)}</div>' if note else "")
        + "</div>"
    )


def coach_gauge(label, level):
    """A ★ judgment on a five-point scale. NOT a rate, and it must not look like one.

    `power_meter` prints a percentage because it renders ◆ figures a simulation
    produced. Threat level is not one of those: it is the Coach's read, and Know
    Your Enemy says out loud in the same spread that zero games have been played.
    Rendering it through the data component published "Threat level 60%" beside
    that admission — a reader young enough to be the target audience does not parse
    the hedge, and one old enough sees a number nobody could have measured.

    Five stars, no decimals, and the label says whose opinion it is. If a figure
    here is ever genuinely derived, it belongs in a `power_meter` under Ledger's
    byline instead — the component is the tier claim (STYLEv3 §10).
    """
    n = max(1, min(5, int(round(level))))
    stars = "".join(f'<span class="cg-star{" on" if i < n else ""}">★</span>'
                    for i in range(5))
    return (
        f'<div class="coach-gauge"><div class="cg-label">{esc(label)}</div>'
        f'<div class="cg-stars" role="img" aria-label="{n} out of 5">{stars}</div>'
        f"</div>"
    )


def threat_box(name, meter_label, level, body_html):
    return (
        f'<div class="threat-box"><div class="tb-head"><span>{esc(name)}</span>'
        f"<span>{esc(meter_label)}</span></div>"
        # Plain apostrophe: `coach_gauge` escapes its label, so pre-escaping it here
        # produced a literal "COACH&#x27;S READ" on the page.
        f'<div class="tb-body">{coach_gauge(COACH_READ, level)}'
        f"{body_html}</div></div>"
    )


def stylesheet_version():
    """Content hash of the stylesheet — the cache-busting query string.

    Derived, never hand-bumped: hapatra shipped a stale inline stylesheet for
    two issues because resyncing depended on remembering to rebuild. A
    content-addressed href can't drift — same CSS, same URL; changed CSS,
    changed URL, and every page that links it re-fetches.
    """
    return hashlib.sha256(CSS.encode("utf-8")).hexdigest()[:8]


def stylesheet_link():
    """The <link> tag every magazine page carries."""
    return f'<link rel="stylesheet" href="magazine.css?v={stylesheet_version()}">'


def write_stylesheet(directory):
    """Materialise manuals/magazine.css. Idempotent — content-addressed writes
    only when the bytes differ, so repeat builds stay churn-free."""
    path = directory / "magazine.css"
    if not path.exists() or path.read_text(encoding="utf-8") != CSS:
        path.write_text(CSS, encoding="utf-8")
        return path, True
    return path, False


def card_tile(card, roles, synergy, printing=False, anchor_id=None, stage=None):
    """A card tile. `printing=True` adds the artist/set credit and foil sheen —
    used by the gallery, where the physical printing is the subject.
    `anchor_id` mints the card-link target — The 99 only, so a card that also
    appears in the Featured Artist gallery never duplicates an id.

    `stage` is the card's engine stage, inked from `ENGINE_STAGE_INK` so the chip
    and the schematic agree by construction rather than by anyone remembering to
    keep two palettes in step. It answers the question a roster exists for — what
    job does this card hold — which the cities deliberately cannot: a city says
    what a card is LIKE, a stage says what it DOES to the other cards. The repo
    measured that they disagree (4 of radagast's 10 components sit in one city),
    so the chip annotates the grid and never regroups it.

    Absent (no `engine.json`, or a card in the engine's `unassigned` list) the tile
    renders exactly as before. A missing chip means the engine model does not place
    the card, which is a finding, not a hole to paper over.

    Lives here beside card_figure: .card-tile's CSS is this module's, and a
    component whose markup and stylesheet sit in different files drifts.
    """
    name = card["name"]
    # Graph entries are {partner, score, synergies} — `synergies` is a list of rule
    # labels like "Tokens + Sacrifice". This used to read a `rule` key that has never
    # existed on this artifact, so every chip on every manual rendered empty.
    labels = sorted({
        label
        for entry in (synergy or {}).get(name, [])[:3]
        for label in entry.get("synergies", [])
    })
    chips = "".join(f'<span class="chip">{esc(l)}</span>' for l in labels[:2])
    if stage:
        chips = (f'<span class="chip stage" style="--st:'
                 f'{ENGINE_STAGE_INK.get(stage, "#8A93B5")}">{esc(stage)}</span>'
                 + chips)
    image = (f'<img src="{esc(card["image"])}" alt="{esc(name)}" loading="lazy">'
             if card.get("image") else "")
    if printing and card.get("foil") and image:
        image = f'<span class="foil">{image}</span>'
    credit = printing_credit(card) if printing else ""
    anchor = f' id="{esc(anchor_id)}"' if anchor_id else ""
    return (
        f'<div class="card-tile"{anchor}>{image}<h4>{esc(name)}</h4>{chips}'
        f'<p>{esc(roles.get(name, ""))}</p>{credit}</div>'
    )


def pull_quote(text):
    return f'<blockquote class="pull-quote">{esc(text)}</blockquote>'


def printing_credit(card):
    """Artist + printing line — the Duelist convention, and the reason the
    manual shows *your* cards rather than default reprints."""
    if not card:
        return ""
    bits = []
    if card.get("artist"):
        foil = ' <span class="foil-tag">foil</span>' if card.get("foil") else ""
        bits.append(f'<div class="art-credit">Art: {esc(card["artist"])}{foil}</div>')
    if card.get("set_name"):
        number = card.get("collector_number")
        suffix = f" #{esc(number)}" if number else ""
        bits.append(f'<div class="printing">{esc(card["set_name"])}{suffix}</div>')
    return "".join(bits)


def card_figure(name, image, caption_html, scryfall_uri=None, card=None):
    """Hero/feature card image with a teaching caption (STYLEv3 §7.4).

    `caption_html` is inserted **unescaped** — it arrives pre-escaped from
    build_manual.caption_html(), which injects the bold lead-in. Pass raw user
    text and you will emit broken markup.

    `card` is the full cards.json entry; without it the figure loses its artist
    and printing credit and its foil sheen, which is the whole point of the
    Featured Artist department.
    """
    if not image:
        return ""
    img = f'<img src="{esc(image)}" alt="{esc(name)}" loading="lazy">'
    if (card or {}).get("foil"):
        img = f'<span class="foil">{img}</span>'
    if scryfall_uri:
        img = f'<a href="{esc(scryfall_uri)}">{img}</a>'
    return (f'<figure class="card-fig">{img}<figcaption>{caption_html}'
            f"{printing_credit(card)}</figcaption></figure>")


def folio(department_title, volume, page_note=""):
    right = esc(page_note) if page_note else f"VOL. {volume:03d}"
    return (
        f'<div class="folio"><strong>{esc(department_title)}</strong>'
        f"<span>MANA MAP · {right}</span></div>"
    )


def barcode(seed):
    """Deterministic decorative barcode derived from the issue identity."""
    digest = hashlib.sha256(str(seed).encode("utf-8")).digest()
    bars = "".join(
        f'<i style="height:{22 + (b % 17)}px;width:{1 + (b % 3)}px"></i>'
        for b in digest[:28]
    )
    return f'<div class="barcode">{bars}</div>'
