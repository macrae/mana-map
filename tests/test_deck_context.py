"""The Deck Context (PRD v2, 2026-10-07): CONTEXT.md, one per deck.

Unit tests on inline text, no tracked data: the renderer's blocks are stubbed, the
deck's 99 and sha are monkeypatched. The fleet freshness test is at the bottom and
reads the tracked files (regression tier, by tests/conftest.py's rules).
"""
import pytest

from manamap.pilot import deck_context as dc

SHA = "ce0ceaa6adbe2d8c1e51eeef0b90f1697b2421e964cb686032427db861dfe129"
DECK = ["Shabraz, the Skyshark", "Brallin, Skyshark Rider", "Windfall",
        "Treasure Map // Treasure Cove", "Island"]
BLOCKS = {"summary": "- v1.2.0", "numbers": "- wheel by T6: **94%**",
          "record": "- No games logged yet.", "history": "- **v1.2.0 · V7**",
          "queue": "- Nothing queued for this deck yet."}


@pytest.fixture
def fake_deck(monkeypatch):
    names = {}
    for n in DECK:
        names[n.lower()] = n
        for face in n.split(" // "):
            names.setdefault(face.lower(), n)
    monkeypatch.setattr(dc, "_deck_names", lambda slug: dict(names))
    from manamap.pilot import common
    monkeypatch.setattr(common, "decklist_sha256", lambda slug, branch=None: SHA)
    monkeypatch.setattr(dc, "_title", lambda slug: "# Shabraz + Brallin — Deck Context")


def written(cards_section, stamp=f"version=v1.2.0 sha={SHA[:12]} at=2026-10-07",
            plays="Wheel into wheels: [[Windfall]] refills both hands and Brallin bills the table."):
    text = dc.scaffold_text("sharknado", BLOCKS)
    text = text.replace("<!-- ctx:written-for none -->", f"<!-- ctx:written-for {stamp} -->")
    text = text.replace("## Summary\n\n" + dc.PLACEHOLDER, "## Summary\n\nA shark that grows on every card.")
    text = text.replace("## How it plays\n\n" + dc.PLACEHOLDER, "## How it plays\n\n" + plays)
    text = text.replace("## Cards by role\n\n" + dc.PLACEHOLDER, "## Cards by role\n\n" + cards_section)
    text = text.replace(dc.PLACEHOLDER, "Nothing yet.")       # pilot notes, history, questions
    return dc.expand_links(text)


ALL_PLACED = ("### Be the engine\n- [[Shabraz, the Skyshark]]\n- [[Brallin, Skyshark Rider]]\n"
              "### Turn a wheel into damage\n- [[Windfall]]\n### Lands\n- [[Treasure Map]] (front face)\n- [[Island]]")


def test_a_scaffold_has_every_section_and_block_and_passes_the_gate(fake_deck):
    text = dc.scaffold_text("sharknado", BLOCKS)
    errors, warnings = dc.check_text("sharknado", text, BLOCKS)
    assert errors == []
    assert any("not written yet" in w for w in warnings)
    assert set(dc.sections(text)) >= set(dc.SECTION_KEYS)
    assert len(dc._GEN_RE.findall(text)) == len(dc.GEN_BLOCKS)


def test_refresh_rewrites_only_the_generated_blocks_byte_for_byte(fake_deck):
    text = written(ALL_PLACED)
    newer = dict(BLOCKS, numbers="- wheel by T6: **96%**", history="- **v1.3.0 · V8**")
    out = dc.replace_blocks(text, newer)
    assert "**96%**" in out and "**94%**" not in out
    assert dc.authored(out) == dc.authored(text)            # the Keeper's prose, untouched
    assert dc.replace_blocks(out, newer) == out             # idempotent


def test_a_block_the_template_gained_later_is_inserted_by_refresh(fake_deck):
    """The `queue` block arrived after eight files were written; refresh migrates them."""
    text = written(ALL_PLACED)
    old = dc._GEN_RE.sub(lambda m: "" if m.group(1) == "queue" else m.group(0), text)
    assert "ctx:gen queue" not in old
    out = dc.replace_blocks(old, BLOCKS)
    assert "<!-- ctx:gen queue -->\n- Nothing queued" in out
    assert out.index("## Open questions") < out.index("ctx:gen queue")
    assert dc.replace_blocks(out, BLOCKS) == out


def test_a_stale_generated_block_is_an_error(fake_deck):
    text = written(ALL_PLACED)
    errors, _ = dc.check_text("sharknado", text, dict(BLOCKS, numbers="- wheel by T6: **96%**"))
    assert any("generated block 'numbers' is out of date" in e for e in errors)


def test_a_cut_card_in_current_prose_is_an_error_naming_it(fake_deck):
    """THE BUG THIS EXISTS FOR: sharknado's handbook rendered Kefnet after v1.2.0 cut it."""
    text = written(ALL_PLACED + "\n- [[Kefnet the Mindful]]")
    errors, _ = dc.check_text("sharknado", text, BLOCKS)
    assert any("Kefnet the Mindful" in e and "does not run" in e for e in errors)
    # and without the card the same document is clean — the test is the card, not the text
    assert dc.check_text("sharknado", written(ALL_PLACED), BLOCKS)[0] == []


def test_once_the_list_moves_a_cut_card_is_a_named_stale_warning_not_silence(fake_deck):
    text = written(ALL_PLACED + "\n- [[Kefnet the Mindful]]",
                   stamp="version=v1.1.0 sha=7d623c5f9cf4 at=2026-10-01")
    errors, warnings = dc.check_text("sharknado", text, BLOCKS)
    assert errors == []
    assert any(w.startswith("STALE: prose written for v1.1.0") for w in warnings)
    assert any("STALE" in w and "Kefnet the Mindful" in w for w in warnings)


def test_history_and_questions_may_name_cut_cards_and_candidates(fake_deck):
    text = written(ALL_PLACED)
    text = text.replace("## Open questions\n\nNothing yet.",
                        "## Open questions\n\n" + dc.expand_links("Would [[Wheel of Fortune]] beat [[Windfall]]?"))
    assert dc.check_text("sharknado", text, BLOCKS)[0] == []


def test_the_install_gate_wants_every_card_placed_and_no_placeholder(fake_deck):
    text = written("### Turn a wheel into damage\n- [[Windfall]]")
    errors, _ = dc.check_text("sharknado", text, BLOCKS, strict=True)
    assert any("under no role" in e and "Island" in e for e in errors)
    assert dc.check_text("sharknado", written(ALL_PLACED), BLOCKS, strict=True)[0] == []


def test_the_strict_check_finds_an_unplaced_card_before_the_draft_is_stamped(fake_deck):
    """The Keeper self-checks an UNSTAMPED draft. Gating the unplaced-card check on
    the stamp let two drafts read clean that the install then refused."""
    draft = written("### Turn a wheel into damage\n- [[Windfall]]", stamp="none")
    errors, _ = dc.check_text("sharknado", draft, BLOCKS, strict=True)
    assert any("under no role" in e for e in errors)


def test_a_double_faced_card_answers_to_its_front_face(fake_deck):
    errors, _ = dc.check_text("sharknado", written(ALL_PLACED), BLOCKS, strict=True)
    assert not any("Treasure" in e for e in errors)


def test_slice_returns_the_header_and_only_the_sections_asked_for(fake_deck, tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "path", lambda slug: tmp_path / "CONTEXT.md")
    (tmp_path / "CONTEXT.md").write_text(written(ALL_PLACED))
    out = dc.slice_text("sharknado", ["numbers"])
    assert "## Numbers" in out and "**94%**" in out and "ctx:gen summary" in out
    assert "## How it plays" not in out and "## Cards by role" not in out
    with pytest.raises(SystemExit):
        dc.slice_text("sharknado", ["engine"])


def test_broken_markers_refuse_rather_than_rewrite():
    for bad, why in (("<!-- ctx:gen numbers -->\nx\n", "never closed"),
                     ("x\n<!-- /ctx:gen numbers -->\n", "closes without opening"),
                     ("<!-- ctx:gen engine -->\nx\n<!-- /ctx:gen engine -->\n", "unknown block")):
        assert any(why in e for e in dc.check_markers(bad)), bad
        with pytest.raises(ValueError):
            dc.replace_blocks(bad, BLOCKS)


def test_wiki_links_become_mana_map_links_and_read_back():
    text = dc.expand_links("[[Commit // Memory]] and [[Sol Ring]]")
    assert "?cards=Commit+%2F%2F+Memory" in text
    assert dc.named_cards(text) == ["Commit // Memory", "Sol Ring"]


def test_the_charters_declare_their_targets():
    """The band reads `sla_s:` from the frontmatter; a charter without one shows none."""
    from pathlib import Path

    from manamap import config

    agents = config.DATA_DIR.parent / ".claude" / "agents"
    for name, sla in (("context-keeper", 120), ("data-analyst", 30)):
        head = (agents / f"{name}.md").read_text().split("---")[1]
        assert f"sla_s: {sla}" in head, name
    assert Path(dc.__file__).exists()


@pytest.mark.regression
def test_every_tracked_deck_context_passes_its_gate():
    """Generated blocks equal a fresh render, and no current section names a card
    the 99 does not run. `regen` keeps the blocks current; a red here after a
    model change means `manamap pilot context --refresh --all`."""
    checked = 0
    for slug in dc.live_slugs():
        errors, _ = dc.check(slug)
        assert errors == [], f"{slug}: {errors}"
        checked += 1
    assert checked >= 1
