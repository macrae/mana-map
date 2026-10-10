"""Build's edit tray, server half: `deck/edit*`, `deck/save-version`, `branch/axes`.

Every guard lives in `deck_edit`; these tests hold that the API carries each
refusal through as a 400 WITH ITS SENTENCE, that the methods are gated (history
is a read, every other verb is POST-only), that `_oplist` is strict, and that
the two pieces of process state this layer owns — one coalesced rebuild per
deck, the newest preview per deck — behave.

The deck tree is `test_pilot_deck_edit`'s: a tmp `config.DECKS_DIR` with a
stubbed corpus, so nothing reads the real fleet. `deck_edit.rebuild` and
`try_swap.preview` are stubbed where a test is about the job machinery rather
than the measurement (the chain itself is tested in `test_pilot_deck_edit`).
"""
from __future__ import annotations

import http.client
import json
import threading
import time

import pytest

from manamap import serve
from manamap.pilot import common, deck_edit, try_swap
from test_pilot_deck_edit import (_set_versions, _text, add, cut, decks,  # noqa: F401
                                  swap)


def _wait(job_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = serve.call("job", {"id": job_id})
        if row["state"] != "running":
            return row
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} still running after {timeout}s")


def _ui(op):
    """deck_edit's op shape -> the page's (`name` is `card` on the wire)."""
    o = dict(op)
    if "name" in o:
        o["card"] = o.pop("name")
    return o


@pytest.fixture
def server():
    httpd = serve.serve(port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def _post(port, name, body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        blob = json.dumps(body).encode()
        conn.request("POST", f"/api/{name}", blob, {"Content-Type": "application/json",
                                                     "Content-Length": str(len(blob))})
        r = conn.getresponse()
        return r.status, json.loads(r.read() or b"{}")
    finally:
        conn.close()


def _get(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", path)
        r = conn.getresponse()
        return r.status, json.loads(r.read() or b"{}")
    finally:
        conn.close()


@pytest.fixture
def no_rebuild(monkeypatch):
    """`deck_edit.rebuild` stubbed to record, so an edit's job is instant."""
    calls = []
    monkeypatch.setattr(deck_edit, "rebuild",
                        lambda slug, **k: calls.append(slug) or {"slug": slug, "ran": []})
    monkeypatch.setattr(serve, "_REBUILDS", {})
    return calls


# ── refusals carry their sentence ────────────────────────────────────────

def test_a_sleeved_deck_is_a_400_with_the_branch_sentence(decks, server, no_rebuild):
    _set_versions(decks, "cmdr", {"paper": {"version": 1, "decklist_sha256": "x"}})
    before = _text(decks)
    status, doc = _post(server, "deck/edit", {
        "slug": "cmdr", "ops": [_ui(swap("Sol Ring", "Necromancy"))],
        "expect_sha": common.list_sha256(before)})
    assert status == 400
    assert "SLEEVED" in doc["error"] and "deck-branch cmdr new" in doc["error"]
    assert _text(decks) == before and no_rebuild == []


@pytest.mark.parametrize("verb", ["deck/edit", "deck/edit/undo", "deck/save-version"])
def test_an_archived_deck_is_a_400_with_revive(decks, server, no_rebuild, verb):
    _set_versions(decks, "cmdr", {"lifecycle": {"status": "retired"}})
    sha = common.list_sha256(_text(decks))
    body = {"slug": "cmdr", "expect_sha": sha, "confirm": "cmdr", "note": "n",
            "ops": [_ui(swap("Sol Ring", "Necromancy"))]}
    status, doc = _post(server, verb, body)
    assert status == 400, doc
    assert "archived (retired)" in doc["error"] and "deck-state cmdr revive" in doc["error"]


def test_a_stale_page_is_refused_and_nothing_is_written(decks, no_rebuild):
    before = _text(decks)
    with pytest.raises(SystemExit, match="moved since this page loaded"):
        serve.call("deck/edit", {"slug": "cmdr", "expect_sha": "0" * 64,
                                 "ops": [_ui(swap("Sol Ring", "Necromancy"))]})
    with pytest.raises(SystemExit, match="moved since this page loaded"):
        serve.call("deck/edit/undo", {"slug": "cmdr", "expect_sha": "0" * 64})
    assert _text(decks) == before and no_rebuild == []


def test_an_edit_needs_the_sha_the_page_loaded(decks, no_rebuild):
    with pytest.raises(ValueError, match="expect_sha"):
        serve.call("deck/edit", {"slug": "cmdr", "ops": [_ui(swap("Sol Ring", "Necromancy"))]})
    with pytest.raises(ValueError, match="expect_sha"):
        serve.call("deck/edit/redo", {"slug": "cmdr"})


def test_the_keep_list_refusal_reaches_the_page(decks, no_rebuild):
    (decks / "cmdr" / "protected.json").write_text(json.dumps(
        {"cards": [{"name": "Blood Artist", "why": "the drain"}]}))
    with pytest.raises(SystemExit, match="PROTECTED"):
        serve.call("deck/edit", {"slug": "cmdr", "expect_sha": common.list_sha256(_text(decks)),
                                 "ops": [_ui(swap("Blood Artist", "Necromancy"))]})


# ── the methods ──────────────────────────────────────────────────────────

def test_history_is_a_get_and_every_other_edit_verb_is_post_only(decks, server):
    assert "deck/edit/history" in serve.GETTABLE and "branch/axes" in serve.GETTABLE
    status, doc = _get(server, "/api/deck/edit/history?slug=cmdr")
    assert status == 200, doc
    h = doc["result"]
    assert h["undo"] == 0 and h["redo"] == 0
    assert h["decklist_sha256"] == common.list_sha256(_text(decks))
    # A tmp tree with no git and no save has no baseline: absent, not "nothing moved".
    assert h["since_save"] is None and h["rebuilding"] is False
    checked = 0
    for name in ("deck/edit", "deck/edit/preview", "deck/edit/undo", "deck/edit/redo",
                 "deck/save-version"):
        assert name in serve.ENDPOINTS and name not in serve.GETTABLE, name
        status, doc = _get(server, f"/api/{name}?slug=cmdr")
        assert status == 405, (name, status)
        assert "POST" in doc["error"]
        checked += 1
    assert checked == 5


def test_save_version_needs_the_slug_typed_back_and_a_note(decks):
    with pytest.raises(ValueError, match="confirm must be 'cmdr'"):
        serve.call("deck/save-version", {"slug": "cmdr", "note": "a note"})
    with pytest.raises(ValueError, match="confirm must be 'cmdr'"):
        serve.call("deck/save-version", {"slug": "cmdr", "note": "a note", "confirm": "cmd"})
    with pytest.raises(ValueError, match="needs a note"):
        serve.call("deck/save-version", {"slug": "cmdr", "note": "   ", "confirm": "cmdr"})


def test_save_version_runs_as_a_job(decks, monkeypatch):
    seen = {}
    monkeypatch.setattr(deck_edit, "save_version",
                        lambda slug, note: seen.update(slug=slug, note=note) or {
                            "slug": slug, "commit": "abc", "version": 3, "context": {"big": 1}})
    job = serve.call("deck/save-version", {"slug": "cmdr", "note": "  cut  the ring ",
                                           "confirm": "cmdr"})
    assert job["question"] == "save-version" and "git commit" in job["cost"]
    row = _wait(job["id"])
    assert row["state"] == "done", row
    assert row["result"] == {"slug": "cmdr", "commit": "abc", "version": 3}
    assert seen == {"slug": "cmdr", "note": "cut the ring"}


# ── _oplist ──────────────────────────────────────────────────────────────

def test_oplist_maps_card_to_name_and_coerces():
    got = serve._oplist([{"op": "cut", "card": " Sol Ring ", "qty": "2"},
                         {"op": "swap", "out": "A", "in": "B", "board": "main"}])
    assert got == [{"op": "cut", "name": "Sol Ring", "qty": 2},
                   {"op": "swap", "out": "A", "in": "B", "board": "main"}]


@pytest.mark.parametrize("bad", [
    None, [], "cut Sol Ring", {"op": "cut"}, ["cut"],
    [{"op": "cut", "card": "Sol Ring", "name": "Sol Ring"}],     # an unknown key
    [{"op": "cut", "card": "Sol Ring", "slug": "other"}],
    [{"op": "cut", "card": ["Sol Ring"]}],
    [{"op": "set", "card": "Forest", "qty": 1.5}],
    [{"op": "set", "card": "Forest", "qty": True}],
    [{"op": "cut", "card": "x"}] * (serve._OP_MAX + 1),
], ids=["none", "empty", "string", "object", "string-op", "name-key", "slug-key",
        "list-value", "half-copy", "bool-qty", "too-many"])
def test_oplist_rejects_anything_else(bad):
    with pytest.raises(ValueError):
        serve._oplist(bad)


def test_an_unknown_op_key_is_a_400_over_http(decks, server):
    status, doc = _post(server, "deck/edit/preview", {
        "slug": "cmdr", "ops": [{"op": "cut", "card": "Sol Ring", "evil": 1}]})
    assert status == 400 and "unknown: evil" in doc["error"]


# ── apply, undo, and the coalesced rebuild ───────────────────────────────

def test_an_edit_writes_returns_the_new_sha_and_undo_restores_the_bytes(decks, no_rebuild):
    before = _text(decks)
    got = serve.call("deck/edit", {"slug": "cmdr", "expect_sha": common.list_sha256(before),
                                   "ops": [_ui(swap("Sol Ring", "Necromancy"))]})
    assert got["written"] and got["diff"] == {"out": {"Sol Ring": 1}, "in": {"Necromancy": 1}}
    assert got["after_sha"] == common.list_sha256(_text(decks)) != common.list_sha256(before)
    assert "before" not in got["entry"] and got["entry"]["source"] == "ui"
    assert _wait(got["job"]["id"])["state"] == "done"
    h = serve.call("deck/edit/history", {"slug": "cmdr"})
    assert h["undo"] == 1 and h["redo"] == 0
    assert h["entries"][0]["diff"] == {"out": {"Sol Ring": 1}, "in": {"Necromancy": 1}}
    back = serve.call("deck/edit/undo", {"slug": "cmdr", "expect_sha": got["after_sha"]})
    assert _text(decks) == before and back["after_sha"] == common.list_sha256(before)
    _wait(back["job"]["id"])
    again = serve.call("deck/edit/redo", {"slug": "cmdr", "expect_sha": back["after_sha"]})
    assert again["after_sha"] == got["after_sha"]
    _wait(again["job"]["id"])
    assert no_rebuild == ["cmdr"] * 3


def test_an_edit_during_a_rebuild_joins_it_and_the_job_loops_once_more(decks, monkeypatch):
    gate, entered, calls = threading.Event(), threading.Event(), []

    def slow(slug, **k):
        calls.append(common.decklist_sha256(slug))
        entered.set()
        gate.wait(5)
        return {"slug": slug, "ran": ["fetch-deck"]}
    monkeypatch.setattr(deck_edit, "rebuild", slow)
    monkeypatch.setattr(serve, "_REBUILDS", {})
    sha0 = common.list_sha256(_text(decks))
    first = serve.call("deck/edit", {"slug": "cmdr", "expect_sha": sha0,
                                     "ops": [_ui(swap("Sol Ring", "Necromancy"))]})
    assert entered.wait(5)
    second = serve.call("deck/edit", {"slug": "cmdr", "expect_sha": first["after_sha"],
                                      "ops": [_ui(swap("Viscera Seer", "Arcane Signet"))]})
    assert second["job"]["id"] == first["job"]["id"], "one rebuild per deck"
    gate.set()
    row = _wait(first["job"]["id"])
    assert row["state"] == "done" and row["result"]["passes"] == 2
    assert calls == [first["after_sha"], second["after_sha"]], "the last pass reads the last list"
    assert row["result"]["decklist_sha256"] == second["after_sha"]
    # Once it has finished, the next edit starts a NEW job.
    third = serve.call("deck/edit/undo", {"slug": "cmdr", "expect_sha": second["after_sha"]})
    assert third["job"]["id"] != first["job"]["id"]
    _wait(third["job"]["id"])


# ── the preview ──────────────────────────────────────────────────────────

def test_only_the_newest_preview_per_deck_comes_back(decks, monkeypatch):
    gate, entered = threading.Event(), threading.Event()

    def fake(slug, ops, branch=None, goldfish=True, **k):
        if not goldfish:
            return {"slug": slug, "base_sha": "s", "blocking": [], "ops": ops,
                    "goldfish": {"absent": "not asked for (goldfish=False)"}}
        if ops[0]["name"] == "Sol Ring":
            entered.set()
            gate.wait(5)
        return {"base_sha": "s", "goldfish": {"table": [{"measure": "kill", "delta": 0.1}],
                                              "call": "better"}}
    monkeypatch.setattr(try_swap, "preview", fake)
    old = serve.call("deck/edit/preview", {"slug": "cmdr", "ops": [{"op": "cut", "card": "Sol Ring"}]})
    assert old["goldfish"] == {"pending": old["job"]["id"]} and old["token"]
    assert entered.wait(5)
    new = serve.call("deck/edit/preview", {"slug": "cmdr", "ops": [{"op": "cut", "card": "Blood Artist"}]})
    new_row = _wait(new["job"]["id"])
    gate.set()
    old_row = _wait(old["job"]["id"])
    assert old_row["result"] == {"superseded": True, "token": old["token"]}
    assert new_row["result"]["superseded"] is False
    assert new_row["result"]["goldfish"]["call"] == "better"


def test_a_preview_with_nothing_to_simulate_starts_no_job(decks, monkeypatch):
    monkeypatch.setattr(try_swap, "preview", lambda slug, ops, goldfish=True, **k: {
        "slug": slug, "goldfish": {"absent": "not modelled for Modern"}})
    got = serve.call("deck/edit/preview", {"slug": "md", "ops": [{"op": "cut", "card": "Mountain"}]})
    assert got["job"] is None and got["goldfish"] == {"absent": "not modelled for Modern"}


def test_branch_axes_are_the_objective_vocabulary_less_membership(decks):
    from manamap.pilot import candidates, deck_branch
    got = serve.call("branch/axes", {"slug": "cmdr"})
    names = [r["axis"] for r in got["axes"]]
    assert set(names) == set(candidates.OBJECTIVE_AXES) - set(deck_branch.MEMBERSHIP_AXES)
    assert "kill_by_8" in names and "engine_online_3" not in names
    # Every axis offered parses — the select cannot offer what branch/new refuses.
    checked = 0
    for r in got["axes"]:
        deck_branch.parse_objective(f"{r['axis']} {'<=' if r['lower_is_better'] else '>='} 0.5")
        checked += 1
    assert checked >= 10
