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


def test_building_an_unbuildable_format_says_why(tmp_path, monkeypatch):
    """It came back "brief.json has no commander" — true, useless, and blaming
    the brief for a limitation of the BUILDER.

    Defence in depth: the picker should not offer it, and this says so if
    something else did.
    """
    import json as _json

    import manamap.config as config

    monkeypatch.setattr(config, "DECKS_DIR", tmp_path)
    (tmp_path / "zz").mkdir()
    (tmp_path / "zz" / "brief.json").write_text(_json.dumps(
        {"slug": "zz", "format": "standard"}))
    with pytest.raises(ValueError) as e:
        serve.call("build/run", {"slug": "zz"})
    msg = str(e.value)
    assert "cannot build Standard" in msg
    assert "anchored on a commander" in msg, "it did not say WHY"
    assert "validate-deck" in msg, "it did not say what CAN be done instead"


def test_the_formats_endpoint_reports_buildability():
    """The picker filters on it, so it has to travel."""
    fmts = {f["key"]: f for f in serve.call("formats", {})["formats"]}
    assert fmts["commander"]["buildable"] is True
    assert fmts["standard"]["buildable"] is False


# ── Four defects from one report: "I am trying to build a zur deck" ────────


def test_a_commander_is_matched_the_way_a_human_types_it():
    """The report was "zur, the enchanter" — lowercase, with a comma.

    That is not a user error. `Zur, Eternal Schemer` sits three rows away in the
    corpus WITH a comma; `Zur the Enchanter` has none. The lookup demanded more
    precision than the name itself carries, and EDHREC had already forgiven it —
    `archetypes` answered happily for the same string, so the styles panel
    worked and the build failed. The part that looks like progress succeeding
    while the part that does the work fails is the worst arrangement of the two.
    """
    exact, _ = serve._resolve_commander("zur, the enchanter")
    assert exact == "Zur the Enchanter"
    assert serve._resolve_commander("ZUR THE ENCHANTER")[0] == "Zur the Enchanter"
    assert serve._resolve_commander("Zur the Enchanter")[0] == "Zur the Enchanter"


def test_a_miss_suggests_commanders_rather_than_anything_that_starts_the_same():
    """"not in cards.csv" is accurate and useless; the corpus knows what you
    probably meant.

    Legendary creatures first, because this is a COMMANDER field — suggesting
    Zuran Orb to someone typing "zur" is a prefix match and no help. And the
    empty-named cards are skipped: the corpus holds cards literally called
    "_____", and `"".startswith("")` put them at the head of every list.
    """
    exact, near = serve._resolve_commander("zur")
    assert exact is None
    assert near, "a miss with no suggestions"
    assert "Zur the Enchanter" in near
    assert not any(n.strip("_") == "" for n in near), near
    assert "Zuran Orb" not in near[:3], f"an artifact outranked a commander: {near}"


def test_saving_stores_the_corpus_name_not_what_was_typed(tmp_path, monkeypatch):
    """Resolved at SAVE, so every later step — build, validate, the manual —
    agrees about which card this is."""
    import json as _json

    import manamap.config as config

    monkeypatch.setattr(config, "DECKS_DIR", tmp_path)
    serve.call("build/save", {"slug": "zz", "commander": "zur, the enchanter"})
    on_disk = _json.loads((tmp_path / "zz" / "brief.json").read_text())
    assert on_disk["commander"] == "Zur the Enchanter"


def test_an_unresolvable_commander_is_refused_with_suggestions(tmp_path, monkeypatch):
    import manamap.config as config

    monkeypatch.setattr(config, "DECKS_DIR", tmp_path)
    with pytest.raises(ValueError) as e:
        serve.call("build/save", {"slug": "zz", "commander": "zur"})
    assert "did you mean" in str(e.value)
    assert "Zur the Enchanter" in str(e.value)


@pytest.mark.skipif(not (serve.DATA_DIR / "decks").is_dir(), reason="no decks")
def test_build_run_WRITES_and_says_what_it_wrote():
    """THE WORST OF THE FOUR. `build_deck.build()` computes a plan and returns
    it; `build_deck.main()` is what persists one. Calling only the first
    reported "100 cards, bracket 3" and left NOTHING on disk — success for work
    it had thrown away.

    Routed through `main` rather than reimplementing the writes, so the page and
    the CLI produce byte-identical artifacts: `main` also merges the
    agent-authored keys an existing `build_plan.json` carries, and a second
    writer would drop them.
    """
    import inspect

    src = inspect.getsource(serve._build_run)
    assert "build_deck.main(" in src, (
        "the endpoint computes a plan without persisting it again")
    assert '"written"' in src, "it does not report what it wrote"
