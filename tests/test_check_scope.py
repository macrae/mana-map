"""`check_scope`: the suite a push must pass, decided from its diff.

The point is that a deck edit stops paying for code it cannot break — and the
risk is the opposite error, a code change slipping through on the cheap path. So
most of these hold the FULL side of the line.
"""
from manamap import check_scope as cs


def test_any_code_path_means_the_full_suite():
    for p in ("src/manamap/pilot/goldfish.py", "tests/test_x.py", "Makefile",
              "pyproject.toml", ".github/workflows/test.yml", ".claude/agents/deck-doctor.md",
              "viz/js/shell.js", "data/card_roles.json", "data/forge_overrides/unflag.txt"):
        plan = cs.classify(["data/decks/edgar-vampires/decklist.txt", p])
        assert plan["scope"] == "full", (p, plan)


def test_an_unrecognised_path_is_code_and_no_upstream_is_full():
    assert cs.classify(["something/new.bin"])["scope"] == "full"
    assert cs.classify(None)["scope"] == "full"


def test_deck_data_alone_is_scoped_to_those_decks():
    plan = cs.classify(["data/decks/edgar-vampires/branches/mardu-combo-v1/decklist.txt",
                        "data/decks/edgar-vampires/info.json", "manuals/p/heliod.html",
                        "data/decks/index.json", "docs/testing.md"])
    assert plan["scope"] == "decks" and plan["decks"] == ["edgar-vampires", "heliod"]


def test_the_manifest_alone_keeps_the_checks_that_name_no_deck():
    plan = cs.classify(["data/decks/index.json"])
    assert plan == {**plan, "scope": "decks", "decks": []}
    k = cs.deck_keyword([], every=["edgar-vampires", "heliod"])
    assert k == "not edgar-vampires and not heliod"


def test_docs_alone_run_the_doc_guards_and_nothing_else():
    plan = cs.classify(["docs/testing.md", "README.md"])
    assert plan["scope"] == "docs"
    assert cs.commands(plan) == [[cs.sys.executable, "-m", "pytest", "-o", "addopts=", "-n0",
                                  "-q", *cs.DOC_GUARDS]]


def test_the_deck_scope_drops_only_the_other_decks():
    k = cs.deck_keyword(["edgar-vampires"], every=["edgar-vampires", "heliod", "ur-dragon"])
    assert k == "not heliod and not ur-dragon"
    cmds = cs.commands({"scope": "decks", "decks": ["edgar-vampires"], "why": ""})
    pytest_cmd = cmds[0]
    assert "not regen" in pytest_cmd and "-k" in pytest_cmd
    assert set(cs.DECK_TESTS) <= set(pytest_cmd)
    assert all(__import__("pathlib").Path(cs.REPO / f).is_file() for f in cs.DECK_TESTS)
    assert "edgar-vampires" not in pytest_cmd[pytest_cmd.index("-k") + 1]


def test_full_scope_is_the_unconditional_target():
    assert cs.commands({"scope": "full", "decks": [], "why": ""}) == [["make", "prepush-full"]]
