# Manual v5 — the compact deck page (DRAFT for Sean to strike lines from)

*Branch `manual-v5`, 2026-08-19. The magazine (STYLEv3, 17 departments, ~70 screens) becomes
a **technical page per deck**: the evidence and the notes a pilot reads before game one and
between games, rendered deterministically from the same artifacts, ~15–20 screens. Strike,
reorder, or rename any row; the defaults below are a proposal.*

## The boundary first

| | the **manual** (this spec) | the **deck page** in the viz (next phase) |
|---|---|---|
| what | a self-contained, printable, JS-free HTML file under `manuals/<slug>.html` — evidence + notes, byte-identical on rebuild | the live workbench surface: notes/log, versions, prescriptions, `deck-info`'s `next` — and a link to the manual |
| reads | the tracked artifacts only | the same plus deploy-time JSON (version list) |
| agents | none at render time (unchanged) | none, ever (the frontend never calls an LLM) |

So the manual does **not** grow a log or a version panel. It stays the thing you could print.

## What survives, section by section

Cost = radagast today, words / screens (from `issue-length --rendered`; 71.3 screens total).

| # | today | renders from | cost | **verdict** | compact form |
|---|---|---|---|---|---|
| — | The Cover | `issue_plan.cover`, art | 1 scr | **DROP** | a one-line header: commander · identity · 100 cards · version/date · tiers legend |
| — | The Flight Plan (contents) | department list | 496 w | **DROP** | a sticky 8-item nav; no prose |
| 1 | The Editor's Letter | `editors_letter` (panel) | 390 w | **DROP** | — |
| 2 | The Command Zone | `command-zone` plan copy + tax ladder | 1,076 w | **FOLD → Plan** | the TAX LADDER table only, under the plan |
| 3 | The Game Plan | `how_it_wins` + engine schematic + `not_modelled` rail | 1,002 w | **KEEP** | `how_it_wins` · the **schematic** (keep — it's the deck's machine) · the not-modelled rail |
| 4 | The 99 | constellation + card grid by city + `card_roles` blurbs | 3,415 w / 9.4 scr | **KEEP-COMPACT** | the **constellation** (keep) + a dense roster table by city (name · cost · roles · stage), no blurbs |
| 5 | The Pilot's Log | `pilots_log` (panel) | 1,791 w | **DROP** | — (the hot take may survive as one line under the plan if the engineer writes it; not now) |
| 6 | Keep or Ship | `mulligan` + opening-hand histogram | 574 w | **KEEP** | as is, minus bylines |
| 7 | What's Your Play? | `decisions/*.json` branch cards | 3,296 w / 6.7 scr | **KEEP-COMPACT** | decisions render **collapsed** (spot + question visible; branches on click), since this is practice, not reading |
| 8 | The Kill | `combo_lines` intros + board block + **stack theatre** | 10,819 w / 14.6 scr | **KEEP-COMPACT** | the theatre stays (best object in the book) but **every** line renders as intro + board + result with the theatre collapsed by default; `features` goes (no plan to carry it) |
| 9 | At the Table | `threat_assessment` + `matchups` + tutor guide + threat boxes | 4,039 w / 9.7 scr | **KEEP-COMPACT** | `threat_assessment` + `matchups` at their 2,500 caps; tutor guide as a table (tutor → default/behind/closing fetch); threat boxes go |
| 10 | Sources Say | `mana_analysis.json` tables + `mana_base` prose | 648 w | **KEEP, data only** | the tables; `mana_base` prose is retired |
| 11 | By the Numbers | `goldfish_metrics.json` | 486 w | **KEEP** | commander cast curve, targets by turn, combat block when opted in, the stated assumptions |
| 12 | The Short List | `considering.json` + art sidecar | 1,366 w | **DROP** | replaced by prescriptions, which are workbench not manual |
| 13 | Judge's Desk | case index → verbatim stack records | 9,134 w (194 visible) | **KEEP** | unchanged: the proof, collapsed, one row per case incl. withheld |
| 14 | Featured Artist | `artist-credits` | 627 w | **DROP** | — |
| — | The Back Page | `issue.next_issue`, barcode | 131 w | **DROP** | a footer: built from `<decklist sha>`, date, evidence legend |

**Estimated result:** ~12–15 screens on radagast (Plan 1.5 · 99 3 · Keep 0.7 · Play 1.5 collapsed · Kill ~4 collapsed · Table 2.5 · Sources 0.7 · Numbers 0.7 · Desk 1.1) against 71.3 today. Yawgmoth's eleven lines collapse the same way, so its 88 screens land near the same number.

## New page order (eight blocks + header/footer)

```
header   commander · identity · 100 cards · V5 (2026-08-12) · ✓◆★ legend
1  PLAN          how_it_wins · engine schematic · not-modelled rail · tax ladder
2  THE 99        constellation · roster by city
3  KEEP OR SHIP  mulligan · hand histogram
4  THE LINES     per verified stack: intro · board · result · [theatre, collapsed]
5  AT THE TABLE  threats · matchups · tutor table
6  PLAY          decisions, collapsed
7  THE NUMBERS   goldfish · mana tables        (Sources Say + By the Numbers merged)
8  THE RECORD    Judge's Desk case index, collapsed (incl. withheld)
footer   decklist sha · built date · evidence legend
```

One technical voice throughout (the notes are `pilot-notes`' already). **Tiers ✓◆★ stay** on
every section and every figure — that is the product, not the magazine.

## What goes from the renderer and its gates

| remove | why |
|---|---|
| `issue_plan.json` as an input; `magazine-editor` plumbing; `the-kill.features`; `OPTIONAL_DEPARTMENTS`; `ACTS`; bylines/columnists/masthead; kickers/deks/violators/pull-quotes/pilot-tips/callouts; halftone/chrome period devices; newsstand-as-rack | packaging for a form that no longer exists |
| `validate_issue`: L10 lint, voice lint, dek-question lint, byline checks, plan validation | magazine-only; what stays is **budgets** (`PROSE_BUDGET`), **taxonomy-id leak**, card-name resolution |
| `issue_spec.DEPARTMENTS` (17) → `SECTIONS` (8) | the single source of truth shrinks; tests that count it follow |
| STYLEv3 | archived to `docs/history/`; a 1-page `docs/manual-v5-spec.md` (this file, finalised) replaces it |
| `issue.json` fields `volume`, `cover_price`, `cover_tagline`, `next_issue` | ignored, not deleted — `deck_name`, `commander`, `status` stay; **the file stays authored** |

## What gets unfrozen (one commit, after the renderer lands)

The frozen legacy on the nine decks becomes deletable the moment nothing reads it:
`issue_plan.json` ×9, the panel keys + `card_roles`/`mana_base`/`upgrades` in
`manual_prose.json` ×9, `considering.json` + `considering_art.json` ×9. Then rebuild the
**seven** live pages — goblin-storm, radagast, gishath, yawgmoth-swarm, heliod,
edgar-vampires, ur-dragon (hapatra is `broken-down`, sisay `retired`; both keep a page
that says so and nothing else) — and the index becomes a deck list, not a rack.

## Phases (each a commit; each leaves the suite green)

1. **Spec** — this file, after your strikes. Also decide: keep the stack theatre (yes) and
   the constellation/schematic (yes)?
2. **`SECTIONS` + the new renderer** alongside the old (`build_page.py`), rendering to
   `manuals/<slug>.html` behind a flag until it replaces `build_manual.py`; measure it with
   `issue-length`.
3. **Gates shrink** — `validate_issue` → `validate_page`; magazine lints deleted; tests follow.
4. **Unfreeze + rebuild seven + index**; delete `build_manual.py`, `design.py`'s magazine
   half, STYLEv3 → history.
5. **Viz deck page** (separate: notes, versions, prescriptions, link to the manual).

## Open questions for you

- **Theatre default:** collapsed on every line (proposed), or expanded on the top one?
- **The 99 roster:** table by city (proposed) or keep the card-tile grid? The grid is 9 screens.
- **Decisions:** collapsed by default (proposed) — is "practice" the right read, or do you
  want them open?
- **Hot take:** a single line under the plan, if the engineer is asked for one later — in or out?
- **Dark theme only**, or keep the printable light stock too?
