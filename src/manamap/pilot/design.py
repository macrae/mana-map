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
.chip { font-family:var(--condensed); text-transform:uppercase; font-size:9.5px;
        letter-spacing:.1em; background:var(--ink); color:var(--paper);
        padding:2px 6px; display:inline-block; margin:0 3px 5px 0; }
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


def pilot_tip(card_name, text, image=None):
    """GamePro ProTip formula: image + slug + one imperative sentence."""
    img = f'<img src="{esc(image)}" alt="{esc(card_name)}" loading="lazy">' if image else "<div></div>"
    return (
        f'<div class="pilot-tip">{img}<div>'
        f'<span class="slug">Pilot Tip</span>{esc(text)}</div></div>'
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


def callout(n, title, text):
    """Numbered play-sequence step with an all-caps mini-headline."""
    return (
        f'<div class="callout"><div class="n">{esc(n)}</div><div>'
        f'<span class="t">{esc(title)}</span>{esc(text)}</div></div>'
    )


def threat_box(name, meter_label, rate, body_html):
    return (
        f'<div class="threat-box"><div class="tb-head"><span>{esc(name)}</span>'
        f"<span>{esc(meter_label)}</span></div>"
        f'<div class="tb-body">{power_meter("Threat level", rate)}{body_html}</div></div>'
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


def card_tile(card, roles, synergy, printing=False):
    """A card tile. `printing=True` adds the artist/set credit and foil sheen —
    used by the gallery, where the physical printing is the subject.

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
    image = (f'<img src="{esc(card["image"])}" alt="{esc(name)}" loading="lazy">'
             if card.get("image") else "")
    if printing and card.get("foil") and image:
        image = f'<span class="foil">{image}</span>'
    credit = printing_credit(card) if printing else ""
    return (
        f'<div class="card-tile">{image}<h4>{esc(name)}</h4>{chips}'
        f'<p>{esc(roles.get(name, ""))}</p>{credit}</div>'
    )


def pull_quote(text):
    return f'<blockquote class="pull-quote">{esc(text)}</blockquote>'


def map_key(entries):
    """Icon legend — one consistent set across every issue."""
    items = "".join(f"<span>{esc(icon)} {esc(label)}</span>" for icon, label in entries)
    return f'<div class="map-key">{items}</div>'


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
