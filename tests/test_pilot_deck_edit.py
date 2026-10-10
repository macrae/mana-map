"""In-place editing of a bench deck: one validator, one writer, a replayed journal.

Every case runs in a tmp deck tree (`config.DECKS_DIR` patched — `common.decks_root`
reads it at call time) with a stubbed corpus, so the unit tier needs no data and
no network. The rules are the repo's own (check_in.analyze, protected,
validate_deck); these tests drive `deck_edit.plan` / `edit` / `undo` / `redo` /
`rebuild` / `save_version` and `try_swap` through their production functions.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import manamap
from manamap import config
from manamap.pilot import card_pool, check_in, deck_edit, try_swap
from manamap.pilot.fetch_deck import parse_decklist

# ── fixtures ─────────────────────────────────────────────────────────────

POOL = {
    "Meren of Clan Nel Toth": ("B, G", "Legendary Creature — Human Shaman", 4),
    "Sol Ring": ("", "Artifact", 1),
    "Arcane Signet": ("", "Artifact", 2),
    "Blood Artist": ("B", "Creature — Vampire", 2),
    "Llanowar Elves": ("G", "Creature — Elf Druid", 1),
    "Animate Dead": ("B", "Enchantment — Aura", 2),
    "Necromancy": ("B", "Enchantment", 3),
    "Eternal Witness": ("G", "Creature — Human Shaman", 3),
    "Viscera Seer": ("B", "Creature — Vampire Wizard", 1),
    "Lightning Bolt": ("R", "Instant", 1),
    "Monastery Swiftspear": ("R", "Creature — Human Monk", 1),
    "Goblin Guide": ("R", "Creature — Goblin Scout", 1),
    "Smash to Smithereens": ("R", "Instant", 2),
    "Rending Volley": ("R", "Instant", 1),
    "Blood Moon": ("R", "Enchantment", 3),
    "Relic of Progenitus": ("", "Artifact", 1),
    "Swamp": ("", "Basic Land — Swamp", 0),
    "Forest": ("", "Basic Land — Forest", 0),
    "Mountain": ("", "Basic Land — Mountain", 0),
}

CMDR_NONBASIC = ["Sol Ring", "Blood Artist", "Llanowar Elves", "Animate Dead",
                 "Eternal Witness", "Viscera Seer"]
CMDR = ("Commander:\n1 Meren of Clan Nel Toth\n\nDeck:\n"
        + "".join(f"1 {n}\n" for n in CMDR_NONBASIC)
        + "47 Forest\n46 Swamp\n")

MODERN = ("4 Lightning Bolt\n4 Monastery Swiftspear\n52 Mountain\n"
          "Sideboard:\n4 Smash to Smithereens\n4 Rending Volley\n4 Blood Moon\n"
          "3 Relic of Progenitus\n")


def _pool():
    return {n: {"color_identity": [c.strip() for c in ci.split(",") if c.strip()],
                "type_line": t, "cmc": float(cmc), "legal": True, "mana_cost": ""}
            for n, (ci, t, cmc) in POOL.items()}


def _card(name, qty, cmdr=False):
    ci, t, cmc = POOL[name]
    return {"name": name, "quantity": qty, "is_commander": cmdr, "type_line": t,
            "color_identity": [c.strip() for c in ci.split(",") if c.strip()],
            "cmc": float(cmc), "oracle_text": "", "mana_cost": ""}


def _write_deck(decks, slug, text, fmt=None):
    d = decks / slug
    d.mkdir(parents=True)
    canon = check_in.render_decklist(parse_decklist(text))
    (d / "decklist.txt").write_text(canon, encoding="utf-8")
    entries = parse_decklist(canon)
    doc = {"deck": slug,
           "cards": [_card(e["name"], e["quantity"], e.get("is_commander"))
                     for e in entries if e["board"] == "main"],
           "decklist_sha256": "x"}
    side = [_card(e["name"], e["quantity"]) for e in entries if e["board"] == "side"]
    if side:
        doc["sideboard"] = side
    if fmt:
        doc["format"] = fmt
        (d / "brief.json").write_text(json.dumps({"slug": slug, "format": fmt}))
    (d / "cards.json").write_text(json.dumps(doc))
    return d


@pytest.fixture
def decks(tmp_path, monkeypatch):
    root = tmp_path / "repo" / "data" / "decks"
    root.mkdir(parents=True)
    monkeypatch.setattr(config, "DECKS_DIR", root)
    names = set(POOL)
    monkeypatch.setattr(check_in, "corpus_names", lambda: names)
    pool = _pool()
    monkeypatch.setattr(card_pool, "load_pool", lambda: pool)
    monkeypatch.setattr(card_pool, "legality",
                        lambda column: {n: "legal" for n in POOL})
    monkeypatch.setattr(deck_edit, "_NAME_MEMO", {})
    monkeypatch.setattr(deck_edit, "LOCK_TIMEOUT", 0.3)
    _write_deck(root, "cmdr", CMDR)
    _write_deck(root, "md", MODERN, fmt="modern")
    return root


def _text(decks, slug="cmdr"):
    return (decks / slug / "decklist.txt").read_text(encoding="utf-8")


def _set_versions(decks, slug, doc):
    (decks / slug / "deck_versions.json").write_text(json.dumps(doc))


def swap(o, i, board="main"):
    return {"op": "swap", "out": o, "in": i, "board": board}


def add(n, board="main", qty=1):
    return {"op": "add", "name": n, "board": board, "qty": qty}


def cut(n, board="main", qty=1):
    return {"op": "cut", "name": n, "board": board, "qty": qty}


def setq(n, q, board="main"):
    return {"op": "set", "name": n, "qty": q, "board": board}


def _blocked(p, needle):
    return any(needle in b for b in p["blocking"])


# ── per-format rules: Commander ──────────────────────────────────────────

def test_a_lone_add_to_a_commander_hundred_is_refused(decks):
    p = deck_edit.plan("cmdr", [add("Necromancy")])
    assert _blocked(p, "101 cards, expected exactly 100")


def test_an_add_and_a_cut_together_are_one_legal_edit(decks):
    p = deck_edit.plan("cmdr", [cut("Sol Ring"), add("Necromancy")])
    assert p["blocking"] == []
    assert p["diff"] == {"out": {"Sol Ring": 1}, "in": {"Necromancy": 1}}
    assert p["size"] == {"before": 100, "after": 100}
    assert "1 Necromancy\n" in p["text_after"] and "Sol Ring" not in p["text_after"]


def test_a_duplicate_is_refused_in_a_singleton_format(decks):
    p = deck_edit.plan("cmdr", [cut("Sol Ring"), add("Blood Artist")])
    assert _blocked(p, "already in cmdr")


def test_an_off_identity_add_is_refused(decks):
    p = deck_edit.plan("cmdr", [swap("Sol Ring", "Lightning Bolt")])
    assert _blocked(p, "Color identity violation: Lightning Bolt")


def test_the_commander_cannot_be_cut(decks):
    p = deck_edit.plan("cmdr", [swap("Meren of Clan Nel Toth", "Necromancy")])
    assert _blocked(p, "is the COMMANDER")
    p = deck_edit.plan("cmdr", [setq("Meren of Clan Nel Toth", 0), add("Necromancy")])
    assert _blocked(p, "is the COMMANDER")


def test_a_name_is_resolved_case_blind_to_the_corpus_spelling(decks):
    p = deck_edit.plan("cmdr", [swap("sol ring", "necromancy")])
    assert p["blocking"] == []
    assert p["diff"] == {"out": {"Sol Ring": 1}, "in": {"Necromancy": 1}}


def test_basics_carry_copies_on_a_commander_list(decks):
    p = deck_edit.plan("cmdr", [cut("Swamp"), add("Forest")])
    assert p["blocking"] == []
    assert "48 Forest" in p["text_after"] and "45 Swamp" in p["text_after"]


def test_the_sideboard_does_not_exist_in_commander(decks):
    p = deck_edit.plan("cmdr", [add("Necromancy", board="side")])
    assert _blocked(p, "Commander has no sideboard")


# ── per-format rules: Modern ─────────────────────────────────────────────

def test_a_fifth_copy_is_refused(decks):
    p = deck_edit.plan("md", [add("Lightning Bolt")])
    assert _blocked(p, "Lightning Bolt x5")


def test_adding_past_sixty_is_legal_in_constructed(decks):
    p = deck_edit.plan("md", [add("Goblin Guide", qty=3)])
    assert p["blocking"] == [] and p["size"]["after"] == 63


def test_cutting_below_sixty_is_refused(decks):
    p = deck_edit.plan("md", [cut("Monastery Swiftspear")])
    assert _blocked(p, "59 cards, expected at least 60")


def test_a_sixteenth_sideboard_card_is_refused(decks):
    p = deck_edit.plan("md", [add("Goblin Guide", board="side")])
    assert _blocked(p, "16 sideboard cards")


def test_basics_are_exempt_from_the_four_copy_limit(decks):
    p = deck_edit.plan("md", [setq("Mountain", 60)])
    assert p["blocking"] == [] and p["size"]["after"] == 68


def test_a_sideboard_set_moves_only_the_sideboard(decks):
    p = deck_edit.plan("md", [setq("Blood Moon", 3, board="side"),
                              add("Goblin Guide", board="side")])
    assert p["blocking"] == []
    assert p["diff"]["out"] == {} and p["diff"]["in"] == {}
    assert p["diff"]["side"] == {"out": {"Blood Moon": 1}, "in": {"Goblin Guide": 1}}


# ── the guard ────────────────────────────────────────────────────────────

def test_a_sleeved_deck_is_refused_with_the_branch_path(decks):
    _set_versions(decks, "cmdr", {"paper": {"version": 1, "decklist_sha256": "x"}})
    before = _text(decks)
    with pytest.raises(SystemExit, match="branch") as e:
        deck_edit.edit("cmdr", [swap("Sol Ring", "Necromancy")])
    assert "deck-branch cmdr new" in str(e.value)
    assert _text(decks) == before


@pytest.mark.parametrize("status", ["broken-down", "superseded", "retired"])
def test_every_lifecycle_status_is_refused_with_revive(decks, status):
    _set_versions(decks, "cmdr", {"lifecycle": {"status": status}})
    with pytest.raises(SystemExit, match=f"archived \\({status}\\).*deck-state cmdr revive"):
        deck_edit.edit("cmdr", [swap("Sol Ring", "Necromancy")])
    with pytest.raises(SystemExit, match="revive"):
        deck_edit.undo("cmdr")


def test_a_brewing_deck_is_editable(decks):
    _set_versions(decks, "cmdr", {"stage": {"name": "dev"}})
    assert deck_edit.guard("cmdr") == "dev"
    assert deck_edit.edit("cmdr", [swap("Sol Ring", "Necromancy")])["written"]


# ── the keep list ────────────────────────────────────────────────────────

@pytest.mark.parametrize("ops", [
    [cut("Blood Artist"), add("Necromancy")],
    [swap("Blood Artist", "Necromancy")],
    [setq("Blood Artist", 0), add("Necromancy")],
], ids=["cut", "swap", "set-0"])
def test_the_keep_list_refuses_every_way_out(decks, ops):
    (decks / "cmdr" / "protected.json").write_text(json.dumps(
        {"cards": [{"name": "Blood Artist", "why": "the drain"}]}))
    p = deck_edit.plan("cmdr", ops)
    assert p["keep_list_hits"] and "Blood Artist is PROTECTED on cmdr" in p["keep_list_hits"][0]
    with pytest.raises(SystemExit, match="PROTECTED"):
        deck_edit.edit("cmdr", ops)


# ── the journal: undo, redo, external, stale, lock ───────────────────────

def test_undo_and_redo_round_trip_the_bytes(decks):
    original = _text(decks)
    deck_edit.edit("cmdr", [swap("Sol Ring", "Necromancy")], note="try it")
    edited = _text(decks)
    assert edited != original
    rec = deck_edit.undo("cmdr")
    assert _text(decks) == original and rec["kind"] == "undo"
    deck_edit.redo("cmdr")
    assert _text(decks) == edited
    deck_edit.undo("cmdr")
    assert _text(decks) == original
    with pytest.raises(SystemExit, match="nothing to undo"):
        deck_edit.undo("cmdr")


def test_a_new_edit_clears_redo(decks):
    deck_edit.edit("cmdr", [swap("Sol Ring", "Necromancy")])
    deck_edit.undo("cmdr")
    deck_edit.edit("cmdr", [swap("Sol Ring", "Arcane Signet")])
    with pytest.raises(SystemExit, match="nothing to redo"):
        deck_edit.redo("cmdr")
    h = deck_edit.history("cmdr")
    assert (h["undo"], h["redo"]) == (1, 0)


def test_a_change_outside_the_editor_is_a_barrier(decks):
    deck_edit.edit("cmdr", [swap("Sol Ring", "Necromancy")])
    path = decks / "cmdr" / "decklist.txt"
    path.write_text(path.read_text().replace("1 Viscera Seer", "1 Arcane Signet"))
    hand = _text(decks)
    with pytest.raises(SystemExit, match="changed outside the editor"):
        deck_edit.undo("cmdr")
    assert _text(decks) == hand, "the hand edit must survive a refused undo"
    with pytest.raises(SystemExit, match="nothing to undo.*outside the editor"):
        deck_edit.undo("cmdr")
    kinds = [e["kind"] for e in deck_edit.read_journal(decks / "cmdr")]
    assert kinds == ["edit", "external"]
    # An edit after the barrier undoes back to the HAND-edited list, not past it.
    deck_edit.edit("cmdr", [swap("Eternal Witness", "Sol Ring")])
    deck_edit.undo("cmdr")
    assert _text(decks) == hand


def test_a_stale_page_is_refused(decks):
    with pytest.raises(SystemExit, match="moved since this page loaded"):
        deck_edit.edit("cmdr", [swap("Sol Ring", "Necromancy")], expect_sha="0" * 64)
    deck_edit.edit("cmdr", [swap("Sol Ring", "Necromancy")])
    with pytest.raises(SystemExit, match="moved since this page loaded"):
        deck_edit.undo("cmdr", expect_sha="0" * 64)


def test_the_other_writers_journal_and_can_be_undone(decks):
    """check-in writes through the one writer, so it lands in the journal."""
    original = _text(decks)
    new = original.replace("1 Sol Ring", "1 Arcane Signet")
    check_in.apply("cmdr", parse_decklist(new), run_chain=False)
    assert (decks / "cmdr" / "decklist.txt.bak").read_text() == original
    e = deck_edit.read_journal(decks / "cmdr")[-1]
    assert (e["kind"], e["source"]) == ("edit", "check-in")
    deck_edit.undo("cmdr")
    assert _text(decks) == original


def test_a_lock_held_by_another_process_blocks(decks):
    lock = decks / "cmdr" / deck_edit.LOCK
    child = subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl, os, sys, time\n"
         f"fd = os.open({str(lock)!r}, os.O_RDWR | os.O_CREAT)\n"
         "fcntl.flock(fd, fcntl.LOCK_EX)\n"
         "os.write(fd, b'pid child: a rebuild')\n"
         "print('held', flush=True)\n"
         "time.sleep(30)\n"],
        stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "held"
        before = _text(decks)
        with pytest.raises(SystemExit, match="busy"):
            deck_edit.edit("cmdr", [swap("Sol Ring", "Necromancy")])
        assert _text(decks) == before
    finally:
        child.kill()
        child.wait()
    assert deck_edit.edit("cmdr", [swap("Sol Ring", "Necromancy")])["written"]


def _cli(decks, *argv):
    env = dict(os.environ)
    env["MANAMAP_DATA_DIR"] = str(decks.parent)
    env["PYTHONPATH"] = str(Path(manamap.__file__).resolve().parents[1])
    return subprocess.run([sys.executable, "-m", "manamap.cli", "pilot", *argv],
                          env=env, capture_output=True, text=True, timeout=120)


def test_the_stack_is_shared_across_processes(decks):
    """Two CLI processes edit; this process undoes the second. No state but the file."""
    original = _text(decks)
    r1 = _cli(decks, "edit", "cmdr", "--swap", "Sol Ring=Necromancy", "--no-chain")
    assert r1.returncode == 0, r1.stderr
    after_first = _text(decks)
    r2 = _cli(decks, "edit", "cmdr", "--swap", "Viscera Seer=Arcane Signet", "--no-chain")
    assert r2.returncode == 0, r2.stderr
    assert _text(decks) != after_first
    deck_edit.undo("cmdr")
    assert _text(decks) == after_first
    r3 = _cli(decks, "edit", "cmdr", "undo", "--no-chain")
    assert r3.returncode == 0, r3.stderr
    assert _text(decks) == original
    h = json.loads(_cli(decks, "edit", "cmdr", "history", "--json").stdout)
    assert (h["undo"], h["redo"]) == (0, 2)
    assert [e["kind"] for e in h["entries"]] == ["undo", "undo", "edit", "edit"]


def test_rotation_keeps_the_stacks(decks, monkeypatch):
    monkeypatch.setattr(deck_edit, "ROTATE_AT", 6)
    original = _text(decks)
    for o, i in (("Sol Ring", "Necromancy"), ("Viscera Seer", "Arcane Signet"),
                 ("Necromancy", "Sol Ring"), ("Arcane Signet", "Viscera Seer")):
        deck_edit.edit("cmdr", [swap(o, i)])
    deck_edit.undo("cmdr")
    deck_edit.undo("cmdr")
    entries = deck_edit.read_journal(decks / "cmdr")
    assert len(entries) < 7 and (decks / "cmdr" / "edits.jsonl.1").exists()
    undo_s, redo_s = deck_edit.stacks(entries)
    assert (len(undo_s), len(redo_s)) == (2, 2)
    deck_edit.undo("cmdr")
    deck_edit.undo("cmdr")
    assert _text(decks) == original


# ── the rebuild ──────────────────────────────────────────────────────────

def _artifacts(d, names):
    for n in names:
        (d / n).write_text("{}")


@pytest.fixture
def recorder(monkeypatch, tmp_path):
    """Every producer the rebuild can reach, stubbed to record (stage, branch)."""
    import importlib
    from manamap import progress
    monkeypatch.setattr(progress, "DIR", tmp_path / ".progress")
    # In order, in this process: a stub set here does not reach a spawned worker.
    # The pool itself is `test_the_rebuild_pool_*` below.
    monkeypatch.setattr(deck_edit, "REBUILD_JOBS", 1)
    calls = []
    for stage, dotted in (("fetch-deck", "manamap.pilot.fetch_deck"),
                          ("goldfish", "manamap.pilot.goldfish"),
                          ("mana-analysis", "manamap.pilot.mana_analysis"),
                          ("deck-combos", "manamap.pilot.deck_combos"),
                          ("net-change", "manamap.pilot.net_change"),
                          ("diagnose", "manamap.pilot.diagnostic"),
                          ("benchmark", "manamap.pilot.benchmark"),
                          ("deck-info", "manamap.pilot.deck_info"),
                          ("context", "manamap.pilot.deck_context")):
        mod = importlib.import_module(dotted)
        monkeypatch.setattr(mod, "main",
                            lambda args, _s=stage: calls.append((_s, args.slug, args.branch)))
    monkeypatch.setattr(try_swap, "champion_reading",
                        lambda slug, branch, it, sd: calls.append(("warm", slug, branch)))
    from manamap.pilot import deck_context
    monkeypatch.setattr(deck_context, "list_change",
                        lambda slug, before_text=None: calls.append(
                            ("context-change", slug, before_text is not None)) or None)
    return calls


def test_the_rebuild_runs_in_order_goldfish_once_and_no_branch(decks, recorder):
    d = decks / "cmdr"
    _artifacts(d, ["goldfish_metrics.json", "mana_analysis.json", "combos.json",
                   "diagnostic.json", "benchmark.json", "info.json", "CONTEXT.md"])
    b = d / "branches" / "b1"
    b.mkdir(parents=True)
    (b / "decklist.txt").write_text(_text(decks))
    _artifacts(b, ["goldfish_metrics.json", "mana_analysis.json", "net_change.json",
                   "info.json", "combos.json"])
    r = deck_edit.rebuild("cmdr", before_text=_text(decks))
    # fetch-deck alone; then the independent middle (in parallel outside a test,
    # here in job order); then the context refresh, and deck-info ONCE, after it.
    assert [c[0] for c in recorder] == [
        "fetch-deck", "goldfish", "mana-analysis", "deck-combos", "diagnose",
        "benchmark", "warm", "context-change", "deck-info"]
    assert [c[0] for c in recorder].count("goldfish") == 1
    assert all(c[2] in (None, True) for c in recorder), "no branch may be touched"
    assert r["failures"] == [] and r["ran"] == ["fetch-deck", "goldfish", "mana-analysis"]


def test_a_sixty_card_rebuild_skips_the_goldfish_stages_and_says_why(decks, recorder):
    d = decks / "md"
    _artifacts(d, ["mana_analysis.json", "combos.json", "info.json"])
    r = deck_edit.rebuild("md")
    stages = [c[0] for c in recorder]
    # deck-info runs once, AFTER the context refresh: it validates the context.
    assert stages == ["fetch-deck", "mana-analysis", "deck-combos",
                      "context-change", "deck-info"]
    for s in ("goldfish", "diagnose", "benchmark"):
        assert "not modelled for Modern" in r["skipped"][s], s


def test_an_offline_fetch_leaves_the_edit_and_says_cards_json_is_behind(decks, recorder,
                                                                       monkeypatch):
    from manamap.pilot import fetch_deck

    def offline(args):
        raise ConnectionError("no network")
    monkeypatch.setattr(fetch_deck, "main", offline)
    deck_edit.edit("cmdr", [swap("Sol Ring", "Necromancy")])
    edited = _text(decks)
    r = deck_edit.rebuild("cmdr")
    assert "cards.json is behind the list" in r["behind"]
    assert "edit cmdr --rebuild" in r["behind"]
    assert _text(decks) == edited
    assert not any(c[0] == "goldfish" for c in recorder)


def test_regen_skip_and_include_branches_scope_the_plan(decks):
    from manamap.pilot import regen
    d = decks / "cmdr"
    _artifacts(d, ["goldfish_metrics.json", "info.json"])
    b = d / "branches" / "b1"
    b.mkdir(parents=True)
    (b / "decklist.txt").write_text(_text(decks))
    _artifacts(b, ["goldfish_metrics.json", "net_change.json"])
    full = {(s, t) for s, _m, _k, ts in regen.plan(slug="cmdr") for t in ts}
    assert ("goldfish", ("cmdr", "b1")) in full and ("net-change", ("cmdr", "b1")) in full
    scoped = regen.plan(slug="cmdr", include_branches=False, skip=("goldfish",))
    assert [(s, ts) for s, _m, _k, ts in scoped] == [("deck-info", [("cmdr", None)])]
    with pytest.raises(ValueError):
        regen.run(slug="cmdr", skip=("no-such-stage",))


# ── save-version ─────────────────────────────────────────────────────────

def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          check=True).stdout


@pytest.fixture
def repo(decks, monkeypatch):
    root = decks.parent.parent
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / "README").write_text("x\n")
    # The real repo's ignore lines for the editor's local state, verbatim.
    real = (Path(manamap.__file__).resolve().parents[2] / ".gitignore").read_text()
    (root / ".gitignore").write_text("\n".join(
        ln for ln in real.splitlines()
        if ln.startswith("data/decks/*/") and ("edits" in ln or "lock" in ln or ".bak" in ln))
        + "\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    from manamap.pilot import deck_context, deck_history
    monkeypatch.setattr(deck_history, "_REPO_ROOT", root)
    monkeypatch.setattr(deck_edit, "_consistency",
                        lambda slug, echo: {"ran": [], "failures": [], "invalid": []})
    monkeypatch.setattr(deck_context, "list_change", lambda slug, before_text=None: None)
    return root


def test_save_version_commits_only_the_decks_paths(repo, decks):
    (repo / "unrelated.txt").write_text("staged by the pilot\n")
    _git(repo, "add", "unrelated.txt")
    (repo / "README").write_text("an unstaged edit\n")
    deck_edit.edit("cmdr", [swap("Sol Ring", "Necromancy")])
    deck_edit.edit("cmdr", [swap("Viscera Seer", "Arcane Signet")])
    r = deck_edit.save_version("cmdr", "two value swaps")
    committed = _git(repo, "show", "--name-only", "--format=", "HEAD").split()
    assert committed and all(p.startswith("data/decks/cmdr/") for p in committed), committed
    assert "data/decks/cmdr/decklist.txt" in committed
    assert not any("edits.jsonl" in p or ".lock" in p or p.endswith(".bak")
                   for p in committed), committed
    staged = _git(repo, "diff", "--cached", "--name-only").split()
    assert staged == ["unrelated.txt"], "the pilot's staged file stays staged"
    assert "README" in _git(repo, "diff", "--name-only")
    msg = _git(repo, "log", "-1", "--format=%B")
    assert msg.startswith("Meren of Clan Nel Toth: two value swaps")
    assert "+in Arcane Signet, Necromancy" in msg and "−out Sol Ring, Viscera Seer" in msg
    assert "Co-Authored-By" not in msg
    assert r["commit"] == _git(repo, "rev-parse", "HEAD").strip()
    assert r["version"] == 2
    assert deck_edit.read_journal(decks / "cmdr")[-1]["kind"] == "save"
    # Undo may go past a save.
    deck_edit.undo("cmdr")
    assert "Viscera Seer" in _text(decks)


def test_save_version_refuses_while_rebuilding_and_when_git_is_busy(repo, decks):
    deck_edit.edit("cmdr", [swap("Sol Ring", "Necromancy")])
    with deck_edit.locked(decks / "cmdr", "a rebuild", name=deck_edit.REBUILD_LOCK):
        with pytest.raises(SystemExit, match="rebuilding"):
            deck_edit.save_version("cmdr", "x")
    head = _git(repo, "rev-parse", "HEAD").strip()
    (repo / ".git" / "MERGE_HEAD").write_text(head + "\n")
    with pytest.raises(SystemExit, match="mid-merge"):
        deck_edit.save_version("cmdr", "x")
    (repo / ".git" / "MERGE_HEAD").unlink()
    with pytest.raises(SystemExit, match="needs --note"):
        deck_edit.save_version("cmdr", "  ")


# ── try and the preview read the same validator ──────────────────────────

def _try_args(slug, **kw):
    base = dict(slug=slug, out=None, in_=None, add=None, cut=None, set=None, side=False,
                branch=None, each=False, stage=None, iterations=200, seed=1, json=True)
    base.update(kw)
    return argparse.Namespace(**base)


def test_try_refuses_exactly_what_plan_refuses(decks, monkeypatch):
    from manamap.pilot import diagnostic
    monkeypatch.setattr(diagnostic, "run_on",
                        lambda *a, **k: pytest.fail("a refused edit must measure nothing"))
    for ops, args in (
            ([add("Necromancy")], dict(add=["Necromancy"])),
            ([cut("Sol Ring")], dict(cut=["Sol Ring"])),
            ([swap("Sol Ring", "Lightning Bolt")], dict(out=["Sol Ring"], in_=["Lightning Bolt"])),
    ):
        want = deck_edit.plan("cmdr", ops)["blocking"]
        assert want
        with pytest.raises(SystemExit) as e:
            try_swap.main(_try_args("cmdr", **args))
        assert str(e.value) == "Refusing that swap set:\n  - " + "\n  - ".join(want)


def test_try_refuses_to_stage_a_size_change(decks):
    with pytest.raises(SystemExit, match="--stage writes SWAPS"):
        try_swap.main(_try_args("cmdr", add=["Necromancy"], cut=["Sol Ring"], stage="b"))


def test_the_preview_cache_key_moves_with_every_input(decks, monkeypatch):
    from manamap.pilot import diagnostic, goldfish
    d = decks / "cmdr"
    (d / "goldfish_targets.json").write_text(json.dumps({"targets": []}))
    p = deck_edit.plan("cmdr", [swap("Sol Ring", "Necromancy")])
    k = lambda pp=p, it=200, sd=1: try_swap.candidate_key("cmdr", None, it, sd, pp)  # noqa: E731
    base = k()
    assert base == k(), "the key is deterministic"
    seen = {base}

    def moved(label):
        new = k()
        assert new not in seen, f"the key did not move with {label}"
        seen.add(new)

    other = deck_edit.plan("cmdr", [swap("Sol Ring", "Arcane Signet")])
    assert k(pp=other) not in seen
    # Two op sequences that make the same list are ONE candidate.
    same = deck_edit.plan("cmdr", [cut("Sol Ring"), add("Necromancy")])
    assert k(pp=same) == base
    assert k(it=300) not in seen and k(sd=2) not in seen
    (d / "cards.json").write_text((d / "cards.json").read_text() + " ")
    moved("cards.json")
    (d / "goldfish_targets.json").write_text(json.dumps({"targets": [{"name": "x"}]}))
    moved("goldfish_targets.json")
    monkeypatch.setattr(goldfish, "model_version", lambda: "not-the-model")
    moved("the model version")
    monkeypatch.setitem(diagnostic.HARNESS, "max_turn", 11)
    moved("the harness")
    monkeypatch.setattr(try_swap, "_dump_stamp", lambda: "a-new-dump")
    moved("the Scryfall dump")


def test_a_sixty_card_preview_says_the_goldfish_is_absent(decks, monkeypatch):
    from manamap.pilot import diagnostic
    monkeypatch.setattr(diagnostic, "run_on", lambda *a, **k: pytest.fail("no goldfish"))
    pv = try_swap.preview("md", [cut("Lightning Bolt"), add("Goblin Guide")])
    assert pv["blocking"] == []
    assert pv["goldfish"] == {"absent": "not modelled for Modern"}
    assert pv["size"] == {"before": 60, "after": 60, "side_before": 15, "side_after": 15}
    assert pv["curve"]["before"]["1"] == 8 and pv["curve"]["after"]["1"] == 8
    json.dumps(pv)                                  # JSON-able, for the endpoint


def test_a_refused_preview_reports_and_measures_nothing(decks, monkeypatch):
    from manamap.pilot import diagnostic
    monkeypatch.setattr(diagnostic, "run_on", lambda *a, **k: pytest.fail("no goldfish"))
    pv = try_swap.preview("cmdr", [add("Necromancy")])
    assert any("101 cards" in b for b in pv["blocking"])
    assert pv["goldfish"] == {"absent": "the edit is refused — nothing to measure"}
    assert pv["price"]["absent"].startswith("no prices.json")


def test_the_preview_prices_from_the_dated_file(decks):
    (decks / "cmdr" / "prices.json").write_text(json.dumps({
        "as_of": "2026-10-01", "source": "scryfall",
        "cards": {"Sol Ring": {"nm_cents": 150, "quantity": 1}}}))
    pv = try_swap.preview("cmdr", [swap("Sol Ring", "Necromancy")], goldfish=False)
    assert pv["price"]["as_of"] == "2026-10-01"
    assert pv["price"]["out_cents"] == 150 and pv["price"]["unpriced"] == ["Necromancy"]
    assert pv["price"]["delta_cents"] == -150


def test_the_rebuild_pool_returns_every_result_in_job_order_and_matches_serial(
        tmp_path, monkeypatch):
    """The middle of `rebuild` runs in spawned processes. Its contract: results come
    back in JOB order whatever order they finish in (the last job here finishes
    first), a failure is a result rather than a raise, and the bytes written are the
    serial run's. Driven through `_run_tasks` with a real spawn pool and a producer
    importable by name (tests/rebuild_probe.py), since a stub cannot cross a spawn."""
    data = tmp_path / "data"
    for side in ("serial", "pool"):
        (data / side / "decks" / "d").mkdir(parents=True)
    jobs = [("regen", "d", ("rebuild_probe", {"n": n})) for n in (0, 1, -1, 3)]

    def run(side, workers):
        monkeypatch.setenv("MANAMAP_DATA_DIR", str(data / side))   # what a worker reads
        monkeypatch.setattr(config, "DECKS_DIR", data / side / "decks")
        got = deck_edit._run_tasks(jobs, workers)
        pids.append({r["pid"] for r in got})
        files = {p.name: p.read_bytes() for p in sorted((data / side / "decks" / "d").iterdir())}
        return [(r["kind"], r["error"]) for r in got], files

    pids = []
    serial = run("serial", 1)
    pool = run("pool", 4)
    assert pids[0] == {os.getpid()} and os.getpid() not in pids[1], \
        "the pool must really run elsewhere — a silent fallback would pass the rest"
    assert pool == serial
    assert [e for _k, e in pool[0]] == [None, None, "ValueError: probe -1 refuses", None]
    assert sorted(pool[1]) == ["probe-0.txt", "probe-1.txt", "probe-3.txt"]
