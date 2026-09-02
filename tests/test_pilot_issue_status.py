"""A retired issue is MARKED, never edited or deleted.

LEGACY (2026-08-19): the magazine renderer. It still renders the nine frozen issues from
artifacts nothing regenerates any more (issue_plan.json, the panel keys,
card_roles/mana_base/upgrades, considering.json), and it is replaced by the compact deck
page in docs/manual-v5-spec.md. Do not extend it; internals below are accurate for what it
does.

An issue is a published record. When the deck it describes stops existing — the
first case being hapatra, broken down for parts so its aristocrats shell could be
sleeved into yawgmoth-swarm — the honest move is a banner, not a rewrite. Every
figure in Vol. 002 was true when it shipped, and the repo's own rule against
editing a passing artifact post-hoc applies to a whole magazine as much as to a
stack.

Three properties are pinned here:

1. **A live issue renders nothing.** `status` is optional and absent means live,
   so adding this mechanism could not shift a single byte on the eight issues
   that did not opt in. (The stylesheet hash moved, which is the documented cost
   of touching `magazine.css`; the rendered bodies did not.)
2. **A retired issue says so before its cover**, because a reader must learn the
   deck is gone before they start reading its numbers as current.
3. **An unknown status is tolerated by the renderer and reported by the
   validator.** A typo in an authored field must not be able to take a published
   magazine offline, but it must not silently read as live either.
"""

import json

import pytest

from manamap.config import DECKS_DIR
from manamap.pilot.common import deck_lifecycle
from manamap.pilot import validate_issue
from manamap.pilot.design import issue_status_banner
from manamap.pilot.issue_spec import ISSUE_STATUSES, issue_status

MANUALS = DECKS_DIR.parent.parent / "manuals"


# ── the status resolver ───────────────────────────────────────────────────

def test_an_issue_with_no_status_is_live():
    assert issue_status({"volume": 1}) is None
    assert issue_status({}) is None
    assert issue_status(None) is None


def test_a_known_status_resolves_to_a_headline_and_a_body():
    key, headline, body = issue_status({"status": "broken-down"})
    assert key == "broken-down"
    assert headline == "BROKEN DOWN FOR PARTS"
    assert "no longer exists physically" in body
    # The promise the whole mechanism rests on.
    assert "kept as published" in body


def test_an_unknown_status_is_tolerated_rather_than_raising():
    """A typo must not be able to crash a published magazine's build."""
    assert issue_status({"status": "brokendown"}) is None
    assert issue_status({"status": ""}) is None


# ── the banner ────────────────────────────────────────────────────────────

def test_a_live_issue_renders_no_banner_at_all():
    """Empty string, not an empty div — this is what kept the other eight
    issues' bodies byte-identical when the mechanism landed."""
    assert issue_status_banner(None) == ""


def test_a_retired_issue_renders_a_banner_carrying_its_key():
    html = issue_status_banner(issue_status({"status": "broken-down"}))
    assert 'class="issue-status"' in html
    assert 'data-status="broken-down"' in html
    assert "BROKEN DOWN FOR PARTS" in html


@pytest.mark.parametrize("key", sorted(ISSUE_STATUSES))
def test_every_declared_status_renders(key):
    """A status in the table that the renderer cannot draw is a trap: the
    validator would pass it and the page would show nothing."""
    html = issue_status_banner(issue_status({"status": key}))
    assert 'class="issue-status"' in html and key in html


# ── the validator ─────────────────────────────────────────────────────────

def test_validate_issue_reports_an_unknown_status():
    issue = {"volume": 1, "issue_date": "x", "cover_price": "x", "deck_name": "X",
             "commander": "X", "cover_tagline": "x", "next_issue": "X",
             "decklist_sha256": "abc", "status": "brokendown"}
    errors = validate_issue.validate_identity(issue, deck_sha256=None)
    assert any("status" in e and "brokendown" in e for e in errors), errors


def test_validate_issue_accepts_a_known_status_and_absence():
    base = {"volume": 1, "issue_date": "x", "cover_price": "x", "deck_name": "X",
            "commander": "X", "cover_tagline": "x", "next_issue": "X",
            "decklist_sha256": "abc"}
    assert not validate_issue.validate_identity(base, deck_sha256=None)
    # A `status` key here is now an ERROR whatever its value. The lifecycle moved
    # to `deck_versions.json` and NOTHING reads this one any more, so a hand edit
    # setting it would be obeyed by nobody while looking exactly like it worked —
    # quieter than the typo the old vocabulary check was written for.
    errs = validate_issue.validate_identity(
        dict(base, status="broken-down"), deck_sha256=None)
    assert any("deck_versions.json" in e for e in errs), errs


# ── the tracked artifacts ─────────────────────────────────────────────────

@pytest.mark.skipif(not (DECKS_DIR / "hapatra" / "issue.json").exists(),
                    reason="requires the tracked decks")
def test_hapatra_is_marked_broken_down_and_still_published():
    """The deck was broken down for parts; the issue stays on the rack.

    Deleting it would leave Vol. 002 as a hole in a run that goes 001-009, and
    would destroy a record that was accurate when it shipped.
    """
    # The STATUS moved to `deck_versions.json`; the VOLUME did not — the issue
    # is still a published record on the rack, which is the point of this test.
    assert deck_lifecycle("hapatra")[0] == "broken-down"
    issue = json.loads((DECKS_DIR / "hapatra" / "issue.json").read_text())
    assert issue["volume"] == 2
    assert "status" not in issue, "the lifecycle lives in deck_versions.json now"

    manual = MANUALS / "hapatra.html"
    if not manual.exists():
        pytest.skip("manuals not built")
    html = manual.read_text()
    assert 'data-status="broken-down"' in html
    # Before the cover, not buried mid-issue.
    assert html.index("issue-status") < html.index("masthead")

    index = MANUALS / "index.html"
    if index.exists():
        rack = index.read_text()
        assert "BROKEN DOWN FOR PARTS" in rack
        # Marked, and still a link — the record stays reachable.
        assert 'class="issue is-retired" href="hapatra.html"' in rack


@pytest.mark.skipif(not MANUALS.exists(), reason="manuals not built")
def test_only_the_decks_that_opted_in_carry_a_status_banner():
    """The mechanism is opt-in per issue; nothing may be swept up by accident.

    The list is derived from the LIFECYCLE PREDICATE rather than hardcoded, so
    retiring a deck updates this test's premise with it — a literal would fail
    on the next retirement and teach nothing. It used to derive from
    `issue.json`'s own `status` key, which stopped being the home of that fact
    on 2026-09-01; reading the predicate is what makes it survive the next move
    as well.
    """
    expected = sorted(
        f"{p.parent.name}.html" for p in DECKS_DIR.glob("*/issue.json")
        if deck_lifecycle(p.parent.name))
    marked = sorted(p.name for p in MANUALS.glob("*.html")
                    if p.name != "index.html" and "issue-status" in p.read_text())
    assert marked == expected, f"marked {marked}, issue.json says {expected}"
