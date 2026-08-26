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


def test_a_missing_brief_does_not_speak_in_CLI(tmp_path, monkeypatch):
    """`load_brief` says "author it first" and prints the JSON to write.

    Correct in a terminal, nonsense in a browser where nobody has a file open —
    and it is what the pilot actually saw after a commander failed to resolve
    and no draft was ever written. The endpoint owns the message its caller can
    act on.
    """
    import manamap.config as config

    monkeypatch.setattr(config, "DECKS_DIR", tmp_path)
    with pytest.raises(ValueError) as e:
        serve.call("build/run", {"slug": "never-saved"})
    msg = str(e.value)
    assert "author it first" not in msg
    assert "brief.json" not in msg, "it named an implementation file at the reader"
    assert "commander" in msg, f"it did not say what to do: {msg!r}"


def test_the_commander_picker_ranks_the_one_you_meant_first():
    """A text box that refuses was the wrong control. The picker has no invalid
    state to report — you type, real commanders appear, you choose one.

    Names that START with the query come before names that merely contain it,
    and shorter before longer, so "the ur" offers The Ur-Dragon before Scion of
    the Ur-Dragon.
    """
    out = serve.call("commanders", {"q": "the ur", "limit": 5})["commanders"]
    assert out[0] == "The Ur-Dragon", out
    assert "Zur the Enchanter" in serve.call("commanders", {"q": "zur"})["commanders"]


def test_the_picker_stays_quiet_until_there_is_something_to_match():
    """One letter matches a thousand commanders; that is a list nobody reads."""
    assert serve.call("commanders", {"q": "z"})["commanders"] == []
    assert serve.call("commanders", {"q": ""})["commanders"] == []


def test_the_picker_only_offers_things_that_can_be_commanders():
    """Suggesting Zuran Orb to someone filling a commander field is a prefix
    match and no help."""
    out = serve.call("commanders", {"q": "zur", "limit": 8})["commanders"]
    assert "Zuran Orb" not in out
    assert "Zur's Weirding" not in out


# ── Finishing a draft ──────────────────────────────────────────────────────


def test_finishing_needs_a_decklist_to_finish(tmp_path, monkeypatch):
    import json as _json

    import manamap.config as config

    monkeypatch.setattr(config, "DECKS_DIR", tmp_path)
    (tmp_path / "zz").mkdir()
    (tmp_path / "zz" / "brief.json").write_text(_json.dumps(
        {"slug": "zz", "commander": "Zur the Enchanter"}))
    with pytest.raises(ValueError) as e:
        serve.call("build/finish", {"slug": "zz"})
    assert "build it first" in str(e.value)


def test_finishing_does_not_commit_unless_asked():
    """`decklist.txt` is tracked, so the commit is what `deck-version` NUMBERS
    and what the captain's log stamps games against — "check a deck in without
    committing and tonight's games attach to no version at all".

    That makes it load-bearing rather than bookkeeping, and load-bearing enough
    that a button must not do it by surprise. The command is returned instead,
    which is also the right answer for anyone who wants their own message.
    """
    import inspect

    src = inspect.getsource(serve._build_finish)
    assert "if not commit:" in src, "the commit is not gated"
    assert "commit_command" in src, "it does not offer the command it declined to run"
    # The default must be off at the signature, not merely in the body.
    assert "commit=False" in src.split("\n")[0] or "commit=False" in src[:200]


# ── What a button may honestly run ────────────────────────────────────────
#
# The dossier prints, for each thing a deck does not have yet, the command that
# would produce it. Some of those a page may simply RUN; most it may not, and
# the difference is not a matter of taste — it is cost and determinism.


def test_the_measure_list_holds_nothing_expensive():
    """THE ALLOW-LIST IS THE FEATURE, and this is the guard on it.

    Everything in `MEASURES` is deterministic, makes no model call, and writes
    one artifact. What must never appear: `simulate` (45-62 minutes of Forge),
    any agent loop (the cheapest measured routine is 54.5k tokens and
    `candidate-pool` is 235k), anything that commits, and anything that writes
    a file a PERSON is supposed to author.

    Asserted against the module path rather than the key, because a key can be
    renamed into innocence.
    """
    allowed = {
        "manamap.pilot.bracket", "manamap.pilot.deck_map",
        "manamap.pilot.goldfish", "manamap.pilot.mana_analysis",
    }
    for stage, (module, artifact, needs, what) in serve.MEASURES.items():
        assert module in allowed, (
            f"{stage} runs {module}, which is not on the deterministic list. "
            f"If it belongs there, say why in the comment above MEASURES first.")
        assert artifact.endswith(".json"), stage
        assert isinstance(needs, tuple), stage
        assert what and what[0].islower(), f"{stage}: `what` is a phrase, not a sentence"
    for forbidden in ("sim", "simulate", "engine", "stacks", "experiment",
                      "prose", "frame", "targets", "issue"):
        assert forbidden not in serve.MEASURES, (
            f"{forbidden!r} is an agent loop, a 45-minute simulation, or a file "
            f"somebody has to write — it cannot be a button")


def test_a_stage_it_will_not_run_is_refused_with_the_reason():
    """A refusal names what it WILL do. `simulate` is the one a pilot is most
    likely to reach for from this page, and 'no' on its own would read as a
    bug."""
    with pytest.raises(ValueError) as exc:
        serve.call("deck/measure", {"slug": "radagast", "stage": "sim"})
    msg = str(exc.value)
    assert "sim" in msg
    assert "bracket" in msg and "map" in msg, f"it did not say what it runs: {msg}"


def test_a_measurement_states_its_dependency_rather_than_failing_inside_it():
    """`mana-analysis` embeds goldfish figures and `goldfish` reads an authored
    declaration. A button that fails because of an unstated dependency is worse
    than one that explains itself — and the dossier renders this string."""
    with pytest.raises(ValueError) as exc:
        serve.call("deck/measure", {"slug": "zur-enchantress", "stage": "goldfish"})
    assert "goldfish_targets.json" in str(exc.value)


def test_measuring_refreshes_the_dossier_it_will_be_read_from(tmp_path, monkeypatch):
    """The refresh is not a convenience.

    `info.json` is composed from every other artifact, so a measurement leaves
    it stale BY CONSTRUCTION — and the page renders `info.json`. Without the
    re-emit the pilot presses a button, the command succeeds, and the dossier
    goes on saying the thing they just measured is missing.

    The stale state is CONSTRUCTED rather than assumed: whether a given tracked
    deck happens to be missing a bracket report is not this test's subject, and
    a precondition that drifts with the fleet is a test that stops testing.
    """
    import shutil

    from manamap.config import DECKS_DIR

    src = DECKS_DIR / "zur-enchantress"
    if not (src / "cards.json").exists():
        pytest.skip("needs a fetched deck")

    dest = tmp_path / "decks"
    dest.mkdir()
    shutil.copytree(src, dest / "zur-enchantress")
    deck = dest / "zur-enchantress"
    (deck / "bracket_report.json").unlink(missing_ok=True)
    stale = json.loads((deck / "info.json").read_text())
    stale["bracket"] = None
    stale["status"]["todo"] = [{"stage": "bracket", "what": "x", "how": "y"}]
    (deck / "info.json").write_text(json.dumps(stale))

    monkeypatch.setattr("manamap.config.DECKS_DIR", dest)
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", dest, raising=False)

    result = serve.call("deck/measure",
                        {"slug": "zur-enchantress", "stage": "bracket"})

    assert (deck / "bracket_report.json").exists(), (
        "the measurement did not land in the redirected directory")
    # The returned dossier is the FRESH one, so the page never has to guess.
    assert not any(t["stage"] == "bracket" for t in result["info"]["status"]["todo"]), (
        "the artifact was written but info.json still says it is missing")
    assert result["info"]["bracket"]["floor"]
    on_disk = json.loads((deck / "info.json").read_text())
    assert on_disk["bracket"] == result["info"]["bracket"], (
        "the page was handed something the file does not say")
