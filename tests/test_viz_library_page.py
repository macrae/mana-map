"""`viz/library.html` — Curate, the sixth surface.

The library drawer is good at keeping a card and useless for cutting forty: one
pile at a time behind tabs, 112-132px tiles, and hover already spent on the
remove ✕ and the move dropdown. This page is the other half — every pile at
once, a size you can read, and bulk actions.

Static assertions only; the behaviour lives in the browser suite. What can rot
here is the nav wiring, a cache-bust drifting out of step, a CSS class leaking
into the drawer's, and the two `shell.js` seams this page is the first caller of.
"""

import re

import pytest

from manamap.config import DATA_DIR

VIZ = DATA_DIR.parent / "viz"
PAGE = VIZ / "library.html"
SCRIPT = VIZ / "js" / "library-view.js"
SHEET = VIZ / "css" / "library.css"
SHELL = VIZ / "js" / "shell.js"
SESSION = VIZ / "js" / "session.js"


def test_the_page_its_script_and_its_sheet_exist():
    assert PAGE.exists() and SCRIPT.exists() and SHEET.exists()


def test_the_page_busts_all_of_its_own_assets_at_one_version():
    """`test_viz_cache_busts` checks this across the fleet; this is the local
    copy, so a failure names the page that drifted."""
    seen = {int(n) for n in re.findall(r"\?v=(\d+)", PAGE.read_text())}
    assert len(seen) == 1, f"library.html mixes bust versions: {sorted(seen)}"


def test_the_scripts_load_in_the_order_the_shell_requires():
    """`session.js` before `shell.js`: the strip reads the library through
    Session, and this page writes to it. Every non-Atlas page carries the same
    note, and `index.html` is the exception that made `libraryNames()` grow a
    localStorage fallback."""
    text = PAGE.read_text()
    order = [text.index(f"js/{n}.js") for n in ("api", "session", "shell", "library-view")]
    assert order == sorted(order), "api → session → shell → library-view"


def test_curate_is_in_the_shared_nav_and_the_surface_is_recognised():
    """Without the `currentSurface` branch the file falls through to `'atlas'`,
    so this page would highlight Atlas and render its own nav entry as a link
    back to itself. `test_viz_spaces_page` exists because of that trap; this is
    the same trap one surface later."""
    shell = SHELL.read_text()
    assert "href: 'library.html'" in shell, "the page is not in SURFACES"
    assert "if (file === 'library.html') return 'curate';" in shell, (
        "currentSurface cannot recognise the page")
    # A WORKING SURFACE, NOT A REFERENCE. `appendix: true` renders after a
    # divider in a quieter weight, which is right for Spaces and wrong here.
    entry = shell[shell.index("href: 'library.html'"):]
    assert "appendix" not in entry[:entry.index("},")], "Curate is not an appendix"


def test_the_nav_entry_does_not_reuse_the_word_the_library_button_owns():
    """The strip already carries a right-hand button reading "N in your
    library". Two controls saying the same noun on every page is how a reader
    learns the noun means nothing."""
    shell = SHELL.read_text()
    entry = shell[shell.index("href: 'library.html'"):]
    label = re.search(r"label: '([^']+)'", entry).group(1)
    assert label.lower() != "library", "the drawer button already owns that word"
    assert label == "Curate"


def test_no_other_surface_hand_links_the_page():
    """Two navs two lines apart is one of them being ignored."""
    for page in sorted(VIZ.glob("*.html")):
        if page.name == "library.html":
            continue
        assert 'href="library.html"' not in page.read_text(), (
            f"{page.name} hand-links Curate; shell.js already does")


def test_the_stylesheet_cannot_restyle_the_drawer():
    """THE PREFIX IS LOAD-BEARING TWICE. `test_viz_shell` asserts the `.lib-*`
    block is byte-identical between `mana-map.css` and `tokens.css` — but it
    does not scan THIS file, so a `.lib-` rule here would pass that test and
    still restyle the live drawer, which `Shell.mount` injects on this page too.

    One exception is allowed and named: `lib-tile-noart` is added by shell.js's
    shared `onImageError` and is not ours to rename."""
    for rule in re.findall(r"([^{}]+)\{", SHEET.read_text()):
        sel = rule.strip()
        if sel.startswith(("@", "/*")) or not sel:
            continue
        for bad in (".lib-", ".shell-lib"):
            if bad in sel:
                assert "lib-tile-noart" in sel, f"{sel!r} would restyle the drawer"


def test_the_selection_is_held_by_name_and_never_by_index():
    """`session.js`'s `storage` handler rebuilds `entries` GROUPED BY ZONE,
    which is a different order from the insertion order this page renders. An
    index-based selection therefore points at different cards the moment another
    tab keeps one: select rows 10-25, keep a card in the Atlas, shift-click,
    Remove, and sixteen cards you never looked at are gone."""
    src = SCRIPT.read_text()
    assert "picked = new Set()" in src, "the selection must be a Set of names"
    assert "shown.indexOf(anchor)" in src, (
        "the shift anchor must be a NAME whose index is recomputed at click time")


def test_the_bulk_paths_go_through_the_bulk_api():
    """`remove` and `move` each end in `commit()` — a full re-serialisation plus
    a shell rebuild plus every listener. Forty of those is forty localStorage
    writes and a storm in every other tab."""
    src = SCRIPT.read_text()
    assert "removeMany" in src and "moveMany" in src
    assert re.search(r"\bSession\.library\.remove\(", src) is None, (
        "a per-card remove in a loop is the thing removeMany exists to prevent")


def test_session_grew_the_bulk_operations_and_commits_once():
    """Re-introduce the bug by moving `commit()` inside either loop."""
    src = SESSION.read_text()
    for fn in ("removeMany", "moveMany"):
        body = src[src.index("function " + fn):]
        body = body[:body.index("\n  }")]
        assert body.count("commit()") == 1, f"{fn} must commit exactly once"
        assert "for (const name of" in body, f"{fn} must take a list"


def test_the_art_is_paced_rather_than_burst():
    """`branch-view` measured it: seventy images promoted at once and Scryfall
    answered THIRTY-FIVE — the rest errored and `onImageError` stripped them to
    empty boxes, permanently, because an <img> error carries no status. 190
    tiles is squarely in that regime."""
    src = SCRIPT.read_text()
    assert "data-src" in src, "tiles must defer their art to the queue"
    assert "Shell.queueArt" in src
    shell = SHELL.read_text()
    gap = int(re.search(r"var ART_GAP = (\d+);", shell).group(1))
    assert gap >= 100, (
        f"ART_GAP is {gap}ms — Scryfall asks for 50-100ms between requests and "
        f"190 cards at less than 100 is sustained over-rate")


def test_the_detail_pane_reuses_one_image_element():
    """Assigning `.src` aborts the load in flight, which per-tile elements
    cannot do. Without this, dragging across a row of eight queues eight
    full-size requests against the same limit the grid is spending."""
    src = SCRIPT.read_text()
    assert "paneImg" in src and "HOVER_MS" in src
    assert "cur-big-img" in src, "the pane's <img> must be a single reused node"


def test_the_shared_art_path_knows_about_this_pages_tiles():
    """`ART_HOST` is a hard-coded selector list and `onImageError` bails on
    anything outside it, so a DFC on this page would show a broken-image glyph
    forever instead of retrying its front face."""
    shell = SHELL.read_text()
    host = re.search(r"var ART_HOST = '([^']+)';", shell).group(1)
    assert ".cur-tile" in host, "this page's tiles are outside the retry path"
    for existing in (".lib-tile", ".rost-row"):
        assert existing in host, f"widening ART_HOST dropped {existing}"


def test_a_failed_normal_image_is_not_retried_at_small():
    """`cardImageUrl` defaults to `small`, so a retry written as
    `cardImageUrl(front)` silently downgrades the 340px pane to a 146px image
    stretched sixfold."""
    shell = SHELL.read_text()
    assert "function versionOf(" in shell
    assert "cardImageUrl(name.split(' // ')[0], version)" in shell, (
        "the DFC retry must carry the size that failed")


def test_the_page_probes_the_api_or_the_drawer_buttons_lie():
    """Nothing in shell.js probes; every page that wants a live `/api` calls it
    itself, and the one that forgot reports "needs a local server" with one
    running."""
    assert "Api.probe()" in SCRIPT.read_text()


def test_the_paced_queue_has_exactly_one_implementation():
    """THE COMMENT IN `shell.js` IS A CLAIM, AND THIS IS WHAT MAKES IT TRUE.

    `branch-view.js` grew the queue first and its comment carries the
    measurement the whole mechanism rests on — seventy images promoted at once,
    thirty-five answered. When this page became the second caller the queue
    moved to `shell.js`, and for a while the new comment there said both callers
    shared one implementation while `branch-view` still held its own copy: the
    identical `pump`/`hydrate` pair at `ART_GAP = 70`.

    Two copies of a paced fetch is how one copy stops matching the API's
    guidance — exactly the argument that kept `onImageError` shared rather than
    duplicated. Re-introduce the bug by pasting the old pair back into
    `branch-view.js`.
    """
    branch = (VIZ / "js" / "branch-view.js").read_text()
    assert "Shell.queueArt(sec)" in branch, (
        "branch-view must promote its gallery art through the shared queue")
    for gone in ("var ART_GAP =", "var artQueue", "function hydrate("):
        assert gone not in branch, (
            f"branch-view.js still carries its own {gone!r} — the queue is "
            f"shell.js's, and a second copy will drift from Scryfall's rate")
    assert SHELL.read_text().count("var ART_GAP = ") == 1


def test_the_card_index_is_fetched_after_the_first_render_never_before():
    """THE FACETS ARE AN UPGRADE, NOT A GATE.

    `viz_index.json` is 0.56 MB gzipped — cheap, but not free, and not instant
    on a bad connection. The page's contract is that the grid is on screen and
    usable before it is asked for, so `loadFacts()` must come AFTER the first
    `renderAll()` in `boot`. Written the other way round, or awaited, the page
    is blank for as long as the fetch takes and a failed fetch is a blank page
    forever. The browser suite proves the failure path end to end; this pins the
    ordering, which is the thing a later tidy-up would quietly reverse.
    """
    src = SCRIPT.read_text()
    boot = src[src.index("function boot()"):]
    boot = boot[:boot.index("\n  }")]
    assert boot.index("renderAll()") < boot.index("loadFacts()"), (
        "loadFacts() runs before the first render — the page would wait on a "
        "0.56 MB fetch before showing a library it already has in localStorage")


def test_the_page_does_not_keep_its_own_copy_of_data_version():
    """`DATA_VERSION` lives in `mana-map.js`, which is the atlas's and is not
    loaded here. A second copy is a constant that drifts — and unlike the
    embedding artifacts it would be versioning, `viz_index.json` changes only
    on a corpus refresh. The other non-atlas pages all fetch the same way:
    `cache: 'no-cache'`, which trades a conditional request for the guarantee
    that staleness is impossible."""
    src = SCRIPT.read_text()
    # A DEFINITION or a USE, not a mention: the comment in `loadFacts` names the
    # constant in order to explain why this file does not carry one, and a test
    # that cannot tell those apart forbids writing down the reason.
    code = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    assert "DATA_VERSION" not in code, "this page must not carry its own DATA_VERSION"
    assert "viz_index.json" in src
    fetched = re.search(r"fetch\((.{0,80}viz_index\.json.{0,60})\)", src, re.S)
    assert fetched and "no-cache" in fetched.group(1), (
        "the card index must be fetched with cache: 'no-cache' like every other "
        "non-atlas page, since it carries no version in its URL")


def test_a_missing_measurement_is_never_rendered_as_a_number():
    """ABSENT MEANS ABSENT, ON THIS PAGE TOO.

    Two places on this page could quietly turn "the corpus has never heard of
    this card" into a figure: the mana-value sort, where defaulting to 0 files
    it among the Sol Rings, and the facet filters, where dropping it makes it
    vanish with no way to find it again. Both are covered end to end by the
    browser suite; this is the cheap source-level tripwire beside them, because
    both bugs are one plausible edit away and neither looks wrong in a diff.
    """
    src = SCRIPT.read_text()
    assert "m === null ? Infinity : m" in src, (
        "a card with no mana value must sort LAST, not as zero")
    assert "var NOFACTS" in src, "the unresolvable bucket must be a real facet value"
    for fn in ("colorOf", "typeOf"):
        line = re.search(r"function " + fn + r"\(name\) \{[^}]*\}", src).group(0)
        assert "NOFACTS" in line, f"{fn} must bucket an unresolvable card, not blank it"
