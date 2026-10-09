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


def _moved_deck(tmp_path, monkeypatch, written_sha):
    """A deck dir whose list just moved: `.txt.bak` is the old list, the context is
    stamped for `written_sha`, and the blocks render from BLOCKS."""
    monkeypatch.setattr(dc, "deck_dir", lambda slug, branch=None: tmp_path)
    monkeypatch.setattr(dc, "render_generated", lambda slug: dict(BLOCKS, numbers="- wheel by T6: **97%**"))
    (tmp_path / "decklist.txt.bak").write_text("1 Windfall\n2 Island\n1 Kefnet the Mindful\n")
    (tmp_path / "decklist.txt").write_text("1 Windfall\n1 Island\n1 Jeska's Will\n")
    stamp = f"version=v1.2.0 sha={written_sha[:12]} at=2026-10-07"
    (tmp_path / dc.ARTIFACT).write_text(written(ALL_PLACED, stamp=stamp))


def test_after_a_list_change_the_hook_refreshes_blocks_and_names_the_keeper_pass(
        fake_deck, tmp_path, monkeypatch, capsys):
    """THE MERGE/CHECK-IN HOOK. The prose is stamped for the old list, so it is
    STALE; the generated blocks are rewritten now, not at the next regen; and the
    Keeper's `deck-change` pass is named with the copies that moved."""
    _moved_deck(tmp_path, monkeypatch, written_sha="0" * 64)
    change = dc.list_change("sharknado")
    assert change["stale"] is True and change["written_for"] == "v1.2.0"
    assert change["outs"] == ["Island", "Kefnet the Mindful"] and change["ins"] == ["Jeska's Will"]
    assert "**97%**" in (tmp_path / dc.ARTIFACT).read_text()
    assert "MODE deck-change for sharknado" in change["keeper"]
    dc.print_list_change(change)
    printed = capsys.readouterr().out
    assert "CONTEXT STALE" in printed and "--install" in printed


def test_the_hook_says_current_when_the_prose_was_written_for_this_list(
        fake_deck, tmp_path, monkeypatch, capsys):
    _moved_deck(tmp_path, monkeypatch, written_sha=SHA)
    change = dc.list_change("sharknado")
    assert change["stale"] is False
    dc.print_list_change(change)
    assert "CONTEXT current" in capsys.readouterr().out


def test_no_context_no_hook(tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "deck_dir", lambda slug, branch=None: tmp_path)
    assert dc.list_change("sharknado") is None


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


def _info(**extra):
    base = {"slug": "sharknado", "commander": ["Shabraz, the Skyshark"], "stage": "bench",
            "version": {"current": 7, "tags": ["v1.2.0"]}, "colour_identity": ["W", "U", "R"],
            "lands": 36, "bracket": {"floor": 4, "floor_name": "Optimized"}}
    base.update(extra)
    return base


def test_the_summary_block_carries_the_combos_line_from_info():
    """One line Jarvis answers "what combos does it have" from: the counts out of
    `deck-combos --write`'s summary, the UNCAPPED near total, singular at one."""
    summary = {"included": 7, "infinite": 5, "two_card_infinite": 1, "near": 50, "near_total": 312}
    block = dc._block_summary("sharknado", _info(combos={"summary": summary, "top": []}))
    assert "- **Combos:** 7 known lines (5 infinite, 1 two-card) · 312 one card short" in block
    one = dict(summary, included=1, near_total=1)
    assert "1 known line (5 infinite" in dc._block_summary("sharknado", _info(combos={"summary": one}))


def test_the_combos_line_says_not_computed_and_names_the_command_when_absent():
    """`deck_info` writes `dm.absent(...)` for a deck with no combos.json — never a
    zero, which would read as "no combos". The same word the bracket floor uses."""
    for combos in (None, {"absent_because": "no combos.json — run it", "weight": "body"}):
        block = dc._block_summary("sharknado", _info(combos=combos))
        assert "- **Combos:** not computed (`manamap pilot deck-combos sharknado --write`)" in block
        assert "0 known" not in block
    assert "**bracket floor:** not checked" in dc._block_summary("sharknado", _info(bracket={}))


def test_the_charters_declare_their_targets():
    """The band reads `sla_s:` from the frontmatter; a charter without one shows none."""
    from pathlib import Path

    # The charters live in the repo, not under DATA_DIR (which the isolated tier moves).
    agents = Path(__file__).resolve().parent.parent / ".claude" / "agents"
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


# ── 60-card formats (2026-10-09) ────────────────────────────────────────────

def test_the_summary_block_names_the_format_when_the_deck_has_no_commander():
    """`- **Format:** Modern · G · 60 cards` in the slot the Commander line
    holds for the fleet — never an empty `**Commander:**`."""
    block = dc._block_summary("elves", _info(slug="elves", commander=[], format="modern",
                                             colour_identity=["G"], size=60, lands=16, bracket=None))
    assert block.splitlines()[0] == "- **Format:** Modern · G · 60 cards"
    assert "**Commander:**" not in block
    assert "**bracket floor:** not checked" in block
    # The control: a commander keeps its line, byte for byte.
    assert dc._block_summary("sharknado", _info()).startswith("- **Commander:** [Shabraz")


def test_the_numbers_block_says_the_goldfish_is_commander_only_for_a_60_card_deck(tmp_path, monkeypatch):
    base = tmp_path / "decks" / "elves"
    base.mkdir(parents=True)
    monkeypatch.setattr(dc, "deck_dir", lambda slug, branch=None: base)
    block = dc._block_numbers("elves", _info(slug="elves", commander=[], format="modern"), {})
    assert block.splitlines()[0] == ("- Goldfish: not modelled for Modern — the goldfish is "
                                     "Commander-only (docs/simulation.md).")
    assert "manamap pilot goldfish" not in block


def test_a_context_for_a_deck_with_no_commander_passes_the_gate(monkeypatch):
    """What `context --refresh` / `validate-context` would accept once the
    Keeper has seeded elves: every main card placed, no commander anywhere."""
    sixty = ["Llanowar Elves", "Craterhoof Behemoth", "Forest"]
    monkeypatch.setattr(dc, "_deck_names", lambda slug: {n.lower(): n for n in sixty})
    from manamap.pilot import common
    monkeypatch.setattr(common, "decklist_sha256", lambda slug, branch=None: SHA)
    monkeypatch.setattr(dc, "_title", lambda slug: "# elves — Deck Context")
    blocks = dict(BLOCKS, summary="- **Format:** Modern · G · 60 cards")
    text = dc.scaffold_text("elves", blocks)
    assert text.startswith("# elves — Deck Context")
    text = text.replace("<!-- ctx:written-for none -->",
                        f"<!-- ctx:written-for version=v1.0.0 sha={SHA[:12]} at=2026-10-09 -->")
    text = text.replace("## Summary\n\n" + dc.PLACEHOLDER, "## Summary\n\nMono-green Elves: dorks into [[Craterhoof Behemoth]].")
    text = text.replace("## How it plays\n\n" + dc.PLACEHOLDER, "## How it plays\n\nTurn one [[Llanowar Elves]], then go wide.")
    text = text.replace("## Cards by role\n\n" + dc.PLACEHOLDER,
                        "## Cards by role\n\n### Mana\n- [[Llanowar Elves]]\n- [[Forest]]\n### Finish\n- [[Craterhoof Behemoth]]")
    text = text.replace(dc.PLACEHOLDER, "Nothing yet.")
    text = dc.expand_links(text)
    errors, warnings = dc.check_text("elves", text, blocks, strict=True)
    assert errors == [] and warnings == [], (errors, warnings)
    assert dc.replace_blocks(text, blocks) == text
