"""The Pilot's Operating Handbook, and the gate `manuals/p/` never had.

TWO CLAIMS THIS FILE MAKES GOOD ON.

`build_page.py:388` says "a test asserts exactly that" about the debrief
section printing nothing keyed by a log entry id or a timestamp. No such test
existed — nothing in `tests/` imported `build_page` at all. The claim was
repeated in `page_spec.py` and in the v5 spec, so three surfaces asserted the
existence of a check that was never written.

And `make manuals` rendered only the magazine, so CI's
`git diff --exit-code -- manuals/` could never fail on a compact page: nothing
in the gate rewrote one. A gate that compares a file it did not rebuild is not
a gate.
"""

import pathlib
import re
import subprocess
import sys

import pytest

from manamap.config import DECKS_DIR, MANUALS_DIR
from manamap.pilot import poh, poh_spec as spec, validate_poh
from manamap.pilot.deck_notes import read_log
from conftest import requires_deck

SLUG = "ur-dragon"


def _rendered(slug=SLUG):
    path = MANUALS_DIR / "p" / f"{slug}.html"
    if not path.exists():
        pytest.skip(f"no handbook rendered for {slug}")
    return path.read_text(encoding="utf-8")


# ── the registry ─────────────────────────────────────────────────────────

def test_the_stage_vocabulary_is_the_engines_and_not_a_second_one():
    """THE BRIEF NAMED FOUR STAGES AND THE REPO HAS EIGHT.

    fuel / ignition / payoff / conversion, with protection and answers "drawn as
    guards". Three of those are real; `payoff` is called `output`, there is no
    `answers` stage, and mana / fodder / wincon have no counterpart in the brief.

    Renaming a closed vocabulary that four validators already police, for the
    sake of one document, is how two vocabularies start disagreeing — so the
    handbook uses the engine's and this asserts they cannot drift apart.
    """
    from manamap.pilot.validate_engine import STAGES
    assert spec.STAGE_ORDER == tuple(STAGES)


def test_section_numbers_are_unique_and_emergencies_precede_normal_operation():
    """The ordering IS the product. A handbook that puts normal operation first
    is a manual; putting emergencies first is the whole reason to steal this
    form."""
    numbers = [s[0] for s in spec.SECTIONS]
    assert len(numbers) == len(set(numbers))
    order = {s[1]: i for i, s in enumerate(spec.SECTIONS)}
    assert order["emergency"] < order["normal"]
    assert order["limitations"] < order["emergency"], (
        "a reader meets what the deck cannot do before the procedures for when "
        "it goes wrong")


def test_a_section_is_data_or_authored_and_never_both():
    """The thing that made the magazine unmaintainable was prose and figures
    sharing a key: a regenerated figure would clobber an edited sentence."""
    for number, sid, _t, _p, _tiers, source in spec.SECTIONS:
        assert source in ("data", "authored"), (number, sid, source)
    assert set(spec.DATA_SECTIONS) & set(spec.AUTHORED_SECTIONS) == set()


# ── the render ───────────────────────────────────────────────────────────

@requires_deck
def test_the_handbook_carries_no_script_and_no_build_date():
    """THE CLAIM `build_page.py` MADE AND NOTHING CHECKED.

    No `<script>`, so the file is standalone and prints. And no build date:
    `datetime.now()` reaching the page would make every rebuild a diff and
    destroy the determinism the CI gate rests on.

    Prove it by re-introducing the bug — append `date.today()` to the output and
    this fails.
    """
    import datetime

    html = _rendered()
    assert "<script" not in html.lower()
    # SCOPED TO THE FURNITURE, for the reason `validate_poh` records: an
    # emergency page legitimately cites "the 2026-09-02 Forge run" as its
    # grounding, and dated evidence is exactly what the handbook should carry.
    # A build date lands in the title block; content does not.
    title = re.search(r'<div class="poh-title">.*?</div>', html, re.S)
    assert title, "no title block"
    assert datetime.date.today().isoformat() not in title.group(0)
    # Nothing keyed by a log entry id or a timestamp, either — the v5 rule that
    # three surfaces claimed was tested.
    assert not re.search(r'\b20\d\d-\d\d-\d\dT\d\d:', html), (
        "a timestamp reached the handbook — this is a printed book, not a feed")


@requires_deck
def test_a_rebuild_is_byte_identical():
    """The determinism claim, asserted on the artifact rather than about it."""
    before = _rendered()
    after = poh.render(SLUG)
    assert before == after, (
        "re-rendering changed the bytes — something in the page is not a pure "
        "function of the tracked artifacts")


@requires_deck
def test_every_cross_reference_resolves():
    """A handbook is read OUT OF ORDER, which is why it references by number and
    never by "see above". A number pointing at a page that did not render is the
    one mistake a reader cannot recover from by scrolling."""
    html = _rendered()
    ids = set(re.findall(r'id="(s[0-9-]+)"', html))
    refs = set(re.findall(r'href="#(s[0-9-]+)"', html))
    assert refs, "no cross-references at all — the contents should link"
    assert not (refs - ids), sorted(refs - ids)


@requires_deck
def test_an_unproven_arrow_is_drawn_dashed():
    """`verified_by` IS SPARSE and drawing every arrow solid would be a lie the
    reader cannot see through.

    `validate_engine`'s docstring records two false-green arrows that shipped —
    a passing stack proves A BOARD RESOLVED THIS WAY, not that stage A feeds
    stage B. On ur-dragon 4 of 15 lines carry proof, so 11 must be dashed.
    """
    from manamap.pilot.common import load_json

    engine = load_json(DECKS_DIR / SLUG / "engine.json")
    if not engine:
        pytest.skip("no engine model")
    lines = engine.get("lines") or []
    proved = sum(1 for l in lines if l.get("verified_by"))
    html = _rendered()
    dashed = html.count("stroke-dasharray")
    assert dashed == len(lines) - proved, (
        f"{dashed} dashed arrows against {len(lines) - proved} unproven lines")
    assert "not the same as showing that one stage feeds another" in html, (
        "the legend must say what dashed MEANS, or the distinction is decoration")


@requires_deck
def test_a_missing_artifact_renders_as_a_stated_absence():
    """ABSENT IS NOT ZERO AND IT IS NOT SILENCE.

    zur-enchantress has no engine.json, so sections 1 and 6 cannot exist for it.
    They must say so and name the command — a section that silently disappears
    is indistinguishable from one nobody wrote.
    """
    html = _rendered("zur-enchantress")
    assert "not available" in html
    assert "/analyze-engine" in html or "analyze-engine" in html


@requires_deck
def test_the_handbook_is_far_smaller_than_the_page_it_replaces():
    """A REDESIGN THAT ADDS WEIGHT TO A DOCUMENT ABOUT DENSITY HAS FAILED.

    manual-v5 carried 176-308 hidden full-card images per page — one per card
    mention, revealed on hover, and explicitly hidden on mobile AND on paper. On
    ur-dragon that was 275 KB. The handbook uses one `art_crop` thumbnail per
    subsection on first mention, and the card NAME is always text beside it so
    the page degrades legibly with images off.
    """
    html = _rendered()
    assert len(html) < 120_000, f"{len(html):,} bytes — the old page was 275,103"
    assert html.count("<img") < 20, (
        f"{html.count('<img')} images — the failure being replaced was one per "
        f"card mention")


# ── the gate ─────────────────────────────────────────────────────────────

@requires_deck
def test_every_rendered_handbook_passes_its_validator():
    checked = 0
    for path in sorted((MANUALS_DIR / "p").glob("*.html")):
        errors, _notes = validate_poh.validate(
            path.read_text(encoding="utf-8"), path.stem)
        checked += 1
        assert not errors, f"{path.name}: " + "; ".join(errors)
    assert checked >= 5


def test_the_callout_cap_counts_pages_and_not_sections():
    """THE FIRST CUT OF THIS CHECK FAILED ITS OWN FLEET MEASUREMENT.

    It counted per section and fired on four decks — every one for the same
    correct reason: Systems carries one CAUTION per engine stage with a single
    point of failure, and a deck with four fragile stages is telling the truth
    four times. A validator that fires on correct data is worse than none, so
    the unit became the subsection, which is what a page is in print.

    Two callouts inside ONE subsection is still an error; four spread across
    four subsections is not.
    """
    ok = ('<section class="poh-sec" id="s6"><h2>6 Systems</h2>'
          + "".join(f'<div class="poh-sub"><h3>6.{i}</h3>'
                    f'<aside class="poh-call caution"><span class="lbl">C</span>x</aside>'
                    f"</div>" for i in range(1, 5))
          + "</section>")
    errors, _ = validate_poh.validate(ok, "x")
    assert not [e for e in errors if "callout" in e], errors

    bad = ('<section class="poh-sec" id="s6"><h2>6 Systems</h2>'
           '<div class="poh-sub"><h3>6.1</h3>'
           + '<aside class="poh-call warning"><span class="lbl">W</span>x</aside>' * 3
           + "</div></section>")
    errors, _ = validate_poh.validate(bad, "x")
    assert any("callout" in e for e in errors), errors


def test_the_validator_catches_a_script_and_a_dangling_reference():
    errors, _ = validate_poh.validate(
        '<section class="poh-sec" id="s1"><h2>1</h2><script>x</script></section>', "x")
    assert any("script" in e for e in errors)

    errors, _ = validate_poh.validate(
        '<section class="poh-sec" id="s1"><h2>1</h2>'
        '<a class="xref" href="#s7">7</a></section>', "x")
    assert any("did not render" in e for e in errors)


def test_make_manuals_renders_the_handbook():
    """THE GATE THAT COULD NOT FAIL.

    CI runs `make manuals` then `git diff --exit-code -- manuals/`. The target
    rendered only the magazine, so a stale `manuals/p/` page could never turn CI
    red — nothing in the gate rewrote it. This asserts the target names the
    handbook renderer, from outside the Makefile.
    """
    from manamap.config import DECKS_DIR as _D

    makefile = (_D.parent.parent / "Makefile").read_text()
    target = makefile.split("\nmanuals:")[1].split("\n\n")[0]
    assert "build-poh" in target, (
        "`make manuals` does not render the handbook — the CI byte-diff gate "
        "covers manuals/ but nothing regenerates manuals/p/")


# ── the authored half ────────────────────────────────────────────────────

def test_the_emergency_conditions_mirror_the_causes_the_pilot_files_games_under():
    """THE JOIN IS THE POINT.

    `deck_notes.CAUSES` is the closed vocabulary the pilot files a finished game
    under; `poh_spec.EMERGENCY_CONDITIONS` is what the handbook writes a page
    for. Keyed the same, a procedure for a wipe can be read against the games
    that ended in one — nine losses across seven causes exist on the fleet.

    Let them drift and the pages stop joining to the evidence that grounds them,
    silently, because both files still look sensible on their own.
    """
    from manamap.pilot.deck_notes import CAUSES

    unknown = sorted(set(spec.EMERGENCY_CONDITIONS) - set(CAUSES))
    assert not unknown, (
        f"{unknown} are handbook conditions with no matching cause — a page "
        f"nobody can file a game under")
    # `won` is the one cause with no page, and that asymmetry is correct: a
    # handbook has no emergency procedure for winning.
    missing = sorted(set(CAUSES) - set(spec.EMERGENCY_CONDITIONS) - {"won"})
    assert not missing, (
        f"{missing} are causes games are filed under with no procedure page")


def test_immediate_action_must_be_ordered():
    """Step three before step one is how a game is lost politely."""
    unordered = ('<section class="poh-sec" id="s3"><h2>3 Emergency</h2>'
                 '<div class="poh-procedure"><h3>3.1 WIPE</h3>'
                 '<p><b>Immediate action.</b></p><ul><li>a</li><li>b</li></ul>'
                 "</div></section>")
    errors, _ = validate_poh.validate(unordered, "x")
    assert any("not a numbered list" in e for e in errors), errors

    ordered = unordered.replace("<ul>", "<ol>").replace("</ul>", "</ol>")
    errors, _ = validate_poh.validate(ordered, "x")
    assert not [e for e in errors if "numbered" in e], errors


@requires_deck
def test_a_procedure_page_names_only_cards_the_deck_runs():
    """A PROCEDURE THAT NAMES A CUT CARD FAILS AT THE TABLE.

    This is the handbook's whole claim — that it describes the deck in the
    pilot's hands. The engine model named 17 phantom cards until it was rebuilt,
    so the risk is demonstrated rather than hypothetical.

    IN THE 99 *OR* IN THE PILOT'S OWN LOG, which is the rule `validate_debrief`
    already uses for the same reason. A procedure legitimately names an
    OPPONENT'S card when describing what the table did: edgar's removal page
    lists "an opponent is pinging your 1/1 tokens instead of attacking your
    large bodies — log 004: Tom's Goblin Sharpshooter did this all game", and
    that indication is worth more than a generic one precisely because it
    happened. The first cut of this test called it an invented card.
    """
    import json as _json

    from manamap.pilot.common import expand_copies, load_json
    from manamap.pilot.card_pool import corpus_names

    corpus = {n.lower(): n for n in (corpus_names() or set())}
    if not corpus:
        pytest.skip("no corpus")
    checked = 0
    for base in sorted(DECKS_DIR.iterdir()):
        doc = load_json(base / spec.PROCEDURES_ARTIFACT)
        cards = load_json(base / "cards.json")
        if not doc or not cards:
            continue
        checked += 1
        have = {c["name"].lower() for c in expand_copies(cards.get("cards") or [])}
        # Cards the pilot met and wrote down are cards the handbook may name.
        # PARSED WITH THE PRODUCTION READER, then whitespace-normalised — the
        # same `_norm` `validate_debrief` uses, for the same reason.
        #
        # Reading log.jsonl as text does not work: the notes are hard-wrapped, so
        # log 004 stores "Tom had Goblin\\nSharpshooter pinging my tokens" with a
        # LITERAL backslash-n. A raw substring test called that correctly-grounded
        # reference an invented card, and normalising whitespace did not help
        # because there was no whitespace there to normalise.
        try:
            entries = read_log(base.name)
        except Exception:
            entries = []
        if entries:
            notes = re.sub(r"\s+", " ", " ".join(
                str(e.get("text") or "") for e in entries)).lower()
            have |= {k for k in corpus if len(k) > 10 and k in notes}
        text = _json.dumps(doc)
        # LONGEST MATCH WINS, and only on a word boundary.
        #
        # A bare substring scan flagged "Intervention" — a real card in the
        # corpus — inside "Heroic Intervention", which this deck runs. The naive
        # check called a correct reference an invented card, which is the
        # failure mode that trains a reader to ignore the test.
        named = set()
        for k, v in corpus.items():
            if len(k) <= 10 or k in have:
                continue
            if not re.search(r"\b" + re.escape(v) + r"\b", text):
                continue
            # Is it only there as part of a longer card name the deck DOES run?
            if any(v in real and v != real
                   for real in (corpus[h] for h in have if v.lower() in h)):
                continue
            named.add(v)
        assert not named, (
            f"{base.name}'s procedures name card(s) not in the 99: "
            f"{sorted(named)}")
    if not checked:
        pytest.skip("no deck has procedures yet")


def test_a_procedure_page_is_measured_to_its_own_end():
    """TWICE NOW, IN THIS FILE, A SPLIT MEASURED THE WRONG SPAN.

    First a lookahead matched zero procedure blocks, so the check silently did
    not run. Then the split left the LAST block running to the end of the
    document, absorbing every list item in every later section — it reported 53
    steps for a five-step page and would have taught its reader that the step
    count means nothing.

    A rendered handbook is one long line, so neither mistake is visible by
    eye. Two procedures with five steps each must read as five and five.
    """
    page = ('<div class="poh-procedure"><h3>3.1 A</h3>'
            "<ol>" + "<li>x</li>" * 5 + "</ol></div>")
    html = (f'<section class="poh-sec" id="s3"><h2>3 E</h2>{page}{page}</section>'
            + '<section class="poh-sec" id="s5"><h2>5 P</h2><ul>'
            + "<li>y</li>" * 40 + "</ul></section>")
    _errors, notes = validate_poh.validate(html, "x")
    assert not [n for n in notes if "steps on one page" in n], (
        "a later section's list items were counted against a procedure page: "
        + "; ".join(notes))

    # And a page that IS too long still says so.
    long_page = ('<div class="poh-procedure"><h3>3.1 A</h3><ol>'
                 + "<li>x</li>" * 25 + "</ol></div>")
    _errors, notes = validate_poh.validate(
        f'<section class="poh-sec" id="s3"><h2>3 E</h2>{long_page}</section>', "x")
    assert any("steps on one page" in n for n in notes), notes




# `test_only_the_handbook_writes_manuals_p` was deleted on 2026-09-13 with its
# subject. It asserted that `build_page` — the compact renderer that shared the
# handbook's output path and clobbered it from two callers — could no longer
# write into `manuals/p/`. `build_page` is gone, so the handbook is the only
# writer by construction rather than by assertion.
def test_no_live_caller_renders_the_compact_page_over_the_handbook():
    """The two callers that clobbered it, asserted gone from their own sources.

    A comment saying "do not call this" is a claim; reading the callers is a
    guard. `--out` remains legitimate everywhere, so the assertion is about the
    DEFAULT-path invocation only.
    """
    root = pathlib.Path(__file__).resolve().parent.parent

    branch = (root / "src/manamap/pilot/deck_branch.py").read_text(encoding="utf-8")
    assert '"manamap.pilot.build_page"' not in branch
    assert '"manamap.pilot.poh"' in branch, "the merge chain must rebuild the handbook"

    makefile = (root / "Makefile").read_text(encoding="utf-8")
    calls = re.findall(r"pilot build-page (\S+)", makefile)
    assert calls == [], f"the Makefile still renders the compact page: {calls}"
    assert "pilot build-poh" in makefile, "make manuals must still render handbooks"


def test_every_rendered_page_on_disk_is_a_handbook():
    """The state the collision left behind, asserted so a regression is visible."""
    root = pathlib.Path(__file__).resolve().parent.parent
    pages = sorted((root / "manuals" / "p").glob("*.html"))
    assert len(pages) >= 5, "the guard iterated almost nothing"
    for page in pages:
        head = page.read_text(encoding="utf-8", errors="replace")[:4096]
        assert "Pilot&#x27;s Operating Handbook" in head, page.name


# ── the card name: a preview on hover, a link into the atlas ─────────────
#
# The page this handbook replaced carried a hover preview and it was deleted:
# 176-308 hidden full-card images per page, one per card mention, most of a
# 275 KB file, hidden on mobile AND on paper. `poh.margin_figure`'s docstring
# and `test_the_handbook_is_far_smaller_than_the_page_it_replaces` both record
# it. The preview is back, and these hold it to the three things that got the
# old one deleted: no image elements, nothing fetched before the cursor lands,
# and nothing at all on paper or on touch.

@requires_deck
def test_the_preview_carries_a_committed_url_and_adds_no_image_element():
    """THE WHOLE DESIGN IN ONE ASSERTION.

    Every card name with an image gets the URL as a CSS custom property, and the
    `<img>` count does not move. Re-introduce the bug by dropping the `style`
    from `poh.card_ref` and the first assertion goes red; re-introduce the OLD
    bug by emitting an `<img>` per mention and the last one does.
    """
    html = _rendered()
    pops = re.findall(r"--poh-card-img:url\('([^']+)'\)", html)
    assert len(pops) >= 80, f"only {len(pops)} card names carry a preview URL"
    assert html.count('class="cardref pop"') == len(pops)
    for url in pops:
        assert url.startswith("https://cards.scryfall.io/"), url

    # The count that got the predecessor deleted. One thumbnail per §6
    # subsection, and NOT one image per card mention.
    assert html.count("<img") == html.count('<figure class="poh-fig">')
    assert html.count("<img") < 20, "the preview must not add image elements"


@requires_deck
def test_every_card_name_links_into_the_atlas_seeded_with_itself():
    """THE NAME IS THE DOOR. `?cards=` and not `?card=`: the singular form falls
    back to a RANDOM card when a name does not resolve, and one card in the
    fleet is newer than the corpus dump. Reporting `no match` beats showing the
    wrong card confidently."""
    html = _rendered()
    hrefs = re.findall(r'href="([^"]*viz/index\.html\?cards=[^"]*)"', html)
    assert len(hrefs) >= 80, f"only {len(hrefs)} atlas links"
    assert html.count("cardref") >= len(hrefs)
    for href in hrefs:
        assert href.startswith("../../viz/index.html?cards="), href
        assert href.rsplit("=", 1)[1], f"empty seed: {href}"
    assert "?card=" not in html, "the singular form shows a random card on a miss"


@requires_deck
def test_every_name_the_handbook_links_round_trips_through_the_parser():
    """A LINK THAT RESOLVES TO A DIFFERENT CARD IS WORSE THAN NO LINK. The atlas
    hands `?cards=` to the same reader the textarea uses, so a name carrying a
    comma or a ` // ` must survive encode, decode and parse unchanged. All 100
    sharknado names do; this is what says so after a rename or a re-fetch."""
    import json
    import urllib.parse

    from manamap.pilot.fetch_deck import parse_decklist

    checked = 0
    for path in sorted(DECKS_DIR.glob("*/cards.json")):
        for card in json.loads(path.read_text(encoding="utf-8"))["cards"]:
            name = card["name"]
            decoded = urllib.parse.unquote_plus(urllib.parse.quote_plus(name))
            entries = parse_decklist(decoded)
            assert len(entries) == 1, f"{name!r} parsed to {len(entries)} entries"
            assert entries[0]["name"] == name, f"{name!r} -> {entries[0]['name']!r}"
            checked += 1
    assert checked > 500, f"only {checked} names checked"


@requires_deck
def test_a_card_the_deck_does_not_carry_still_links_but_gets_no_preview():
    """TWO BEHAVIOURS, DELIBERATELY INDEPENDENT — and a missing artifact renders
    as an absence, never as an empty shell.

    ur-dragon's engine.json names Shivan Reef and Stormcarved Coast, which its
    decklist no longer runs, so there is no image. An ungated rule would pin an
    empty bordered box to the corner of the screen, indistinguishable from a
    broken image. The atlas link still works: the corpus knows the card even
    though this deck does not run it. Drop the `if url` arm of `poh.card_ref`
    and this fails.
    """
    html = _rendered()
    bare = re.findall(r'<a class="cardref" href="([^"]+)"[^>]*>([^<]+)</a>', html)
    assert bare, "expected at least one card with no image to prove the gate"
    for href, name in bare:
        assert "viz/index.html?cards=" in href, name
        assert f'class="cardref pop"' not in href
    assert "pop" not in poh.card_ref("Shivan Reef", {})
    assert "viz/index.html?cards=Shivan+Reef" in poh.card_ref("Shivan Reef", {})
    assert "pop" in poh.card_ref("X", {"image": "https://cards.scryfall.io/x.jpg"})


@requires_deck
def test_every_committed_image_url_is_safe_inside_a_css_url():
    """The URL goes into a stylesheet value inside an HTML attribute, so a
    character needing escaping in either language breaks it silently rather than
    loudly. True of all twelve decks today because `fetch_deck.stable_image_url`
    strips Scryfall's query string; asserted so a refetch that changes it fails
    here instead of in a browser."""
    import json

    checked = 0
    for path in sorted(DECKS_DIR.glob("*/cards.json")):
        for card in json.loads(path.read_text(encoding="utf-8"))["cards"]:
            url = card.get("image") or ""
            if not url:
                continue
            checked += 1
            assert not set(url) & set("\"'&<> "), f"{path.parent.name}: {url}"
    assert checked > 500, f"only {checked} URLs checked"


def test_the_preview_is_screen_only_pointer_only_and_wide_only():
    """PAPER AND TOUCH ARE THE TWO SURFACES THE OLD ONE WAS DEAD ON, and the
    third condition is what stops it covering the text: the trim is 46rem
    centred, so a 246px panel inset 1.5rem only clears the column from 1276px
    up. The LINK is not inside the guard — a link works everywhere."""
    from manamap.pilot import poh_design as pd

    css = pd.POH_CSS
    i = css.index("a.cardref.pop:hover::after")
    guard = css.rindex("@media", 0, i)
    header = css[guard:css.index("{", guard)]
    assert "screen" in header and "hover: hover" in header and "80rem" in header, header
    block = css[i:css.index("}", css.index("pointer-events", i))]
    assert "background-image: var(--poh-card-img, none)" in block
    assert "position: fixed" in block, "it would be clipped by .poh-scroll"
    assert "\n    background:" not in block, "use longhands, not the shorthand"
    # The link's own styling must sit OUTSIDE the media query.
    assert css.index("a.cardref {") < guard
