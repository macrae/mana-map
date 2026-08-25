"""The local bridge: what it will run, and what it refuses.

`manamap serve` is the one place this repo lets the browser reach the machine.
It reverses a decision recorded on 2026-08-01 ("there is no local-only bridge"),
on the PRD's authority (§5.2) and the owner's: he is the sole user, the static
build is hygiene rather than an audience, and the agents being reachable from
Build is the point of the software.

That makes these tests about the seam. A local server is still a server.
"""

import json

import pytest

from manamap import serve


def test_commands_are_an_allow_list_not_a_string_to_run():
    """The API takes a NAME and a dict, looks the name up, and calls a Python
    function. Nothing reaches a shell and no argument becomes a flag by string
    interpolation."""
    for name, (fn, spec) in serve.ENDPOINTS.items():
        assert callable(fn), name
        assert isinstance(spec, dict), name
        assert all(callable(c) for c in spec.values()), name
    with pytest.raises(KeyError):
        serve.call("rm -rf /", {})


def test_an_unknown_command_is_not_attempted():
    with pytest.raises(KeyError):
        serve.call("definitely-not-a-command", {})


def test_arguments_are_coerced_and_extras_ignored():
    """Coercion keeps an untyped value from reaching a function that expects a
    number; an unknown key is dropped rather than forwarded.

    Checked on `formats`, which touches nothing — a coercion test that needed
    the network would be testing EDHREC.
    """
    assert serve._int("4") == 4 and serve._int("") is None
    assert serve._strlist("Sol Ring") == ["Sol Ring"]
    assert serve._bool("false") is False and serve._bool("1") is True
    out = serve.call("formats", {"nonsense": "ignored", "limit": 3})
    assert "formats" in out


def test_a_missing_required_argument_is_the_callers_mistake():
    """It reached `edhrec_slug(None)` and surfaced as a 500 — a SERVER fault for
    what is a caller's error. Checked at the endpoint now, which is a 400."""
    with pytest.raises(ValueError):
        serve.call("archetypes", {})
    with pytest.raises(ValueError):
        serve.call("commander-search", {"cards": []})


def test_a_bad_argument_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        serve.call("card-search", {"cmc": "not-a-number"})


def test_it_binds_to_loopback_only():
    """A personal tool with no auth. The one-line change that would expose it to
    a network is a line somebody should have to write themselves."""
    assert serve.HOST == "127.0.0.1"


def test_health_is_cheap_and_says_yes():
    """The probe every agent affordance is gated on. It must not touch the
    corpus, the network or an agent — a page loads it before anything else."""
    assert serve.call("health", {}) == {"ok": True, "api": 1}


def test_the_agent_list_is_read_from_disk():
    """Read rather than transcribed, for the reason every registry in this repo
    is read: a hardcoded list is a second place to remember, and it is the one
    that goes stale."""
    names = [a["name"] for a in serve.call("agents", {})["agents"]]
    assert "deck-doctor" in names and "rules-checker" in names
    from pathlib import Path
    on_disk = sorted(p.stem for p in
                     (Path(serve.DATA_DIR).parent / ".claude" / "agents").glob("*.md"))
    assert names == on_disk


def test_every_agent_cost_is_stated_before_it_is_spent():
    """The cheapest measured routine is 54.5k tokens and `candidate-pool` is
    235k. A button that spends a quarter of a million tokens without saying so
    is the one thing this bridge must not become."""
    for entry in serve.call("agents", {})["agents"]:
        assert entry["cost"], entry
    assert "ask" in serve.AGENT_COSTS


def test_a_job_that_does_not_exist_is_an_error_not_an_empty_answer():
    with pytest.raises(ValueError):
        serve.call("job", {"id": "nope"})


def test_asking_nothing_is_refused():
    with pytest.raises(ValueError):
        serve.call("ask", {"question": ""})


def test_the_formats_endpoint_reports_all_five():
    keys = [f["key"] for f in serve.call("formats", {})["formats"]]
    assert set(keys) == {"commander", "standard", "modern", "pioneer", "pauper"}


# ── Drafts: work you can put down ──────────────────────────────────────────


def test_saving_a_draft_is_idempotent_and_preserves_what_it_does_not_mention(tmp_path,
                                                                             monkeypatch):
    """The page saves on every change, so a save must not be a replace.

    Sending `{"slug": …, "bracket": 4}` — a perfectly good update to a draft
    that already names a commander and holds a library — must not drop either.
    """
    import manamap.config as config

    monkeypatch.setattr(config, "DECKS_DIR", tmp_path)
    first = serve.call("build/save", {
        "slug": "zz", "commander": "Zur the Enchanter",
        "theme": "voltron", "library": ["Ethereal Armor"]})
    assert first["brief"]["must_include"] == ["Ethereal Armor"]

    second = serve.call("build/save", {"slug": "zz", "bracket": 4})
    assert second["brief"]["bracket"] == 4
    assert second["brief"]["commander"] == "Zur the Enchanter", "the commander was dropped"
    assert second["brief"]["theme"] == "voltron", "the style was dropped"
    assert second["brief"]["must_include"] == ["Ethereal Armor"], (
        "the library the pilot spent ten minutes gathering was dropped")


def test_a_required_field_is_checked_on_the_MERGED_brief(tmp_path, monkeypatch):
    """Requiring a commander on every save refused `{"slug", "bracket"}` — an
    update that was never going to send a field it already had on disk."""
    import manamap.config as config

    monkeypatch.setattr(config, "DECKS_DIR", tmp_path)
    with pytest.raises(ValueError):
        serve.call("build/save", {"slug": "zz"})          # nothing on disk yet
    serve.call("build/save", {"slug": "zz", "commander": "Zur the Enchanter"})
    serve.call("build/save", {"slug": "zz", "bracket": 4})   # must not raise


def test_a_sixty_card_format_needs_no_commander(tmp_path, monkeypatch):
    import manamap.config as config

    monkeypatch.setattr(config, "DECKS_DIR", tmp_path)
    out = serve.call("build/save", {"slug": "zz-modern", "fmt": "modern"})
    assert out["draft"] is True


def test_a_draft_writes_a_brief_and_nothing_else(tmp_path, monkeypatch):
    """No 99, no `cards.json`, and no `paper` block — a draft claims nothing."""
    import manamap.config as config

    monkeypatch.setattr(config, "DECKS_DIR", tmp_path)
    serve.call("build/save", {"slug": "zz", "commander": "Zur the Enchanter"})
    files = sorted(p.name for p in (tmp_path / "zz").iterdir())
    assert files == ["brief.json"], files
