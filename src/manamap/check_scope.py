"""How much of the suite a change has to pass before a push — decided from the diff.

A DECKLIST IS PACKAGING, NOT CODE (2026-10-06). `make prepush` ran both tiers on
every push, so a branch edit to one deck — no line of `src/` moved since the last
full green run — waited fifteen minutes for 2,500 unit tests and every other
deck's freshness checks, none of which it could break. The pilot named the cost:
the workbench had become slow and clunky. So the scope follows what changed:

    code    anything outside the paths below (src/, tests/, config, CI, viz/,
            .claude/, fleet-wide data)          -> the full suite, as before
    decks   only data/decks/<slug>/ and manuals/p/<slug>.html
                                                -> those decks' tests, plus every
                                                   check that names no deck
    docs    only docs, README, CLAUDE.md, PLAN  -> the doc guards

Never less safe than it looks: a path this file does not recognise is CODE, no
upstream to diff against means FULL, and CI runs the whole suite on every push
regardless — this only decides what blocks the pilot locally.

    python -m manamap.check_scope plan            what prepush would run, and why
    python -m manamap.check_scope prepush         run it
    python -m manamap.check_scope deck <slug>...  the deck-scoped check, no git
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DECKS = REPO / "data" / "decks"

_DECK_PATH = re.compile(r"^data/decks/([a-z0-9][a-z0-9-]*)/")
_MANUAL_PATH = re.compile(r"^manuals/p/([a-z0-9][a-z0-9-]*)\.html$")
#: Generated FROM the decks, so a deck edit legitimately moves them; their own
#: freshness tests name no single deck and are kept in the deck scope.
_DECK_DERIVED = {"data/decks/index.json", "manuals/page.css"}
_DOCS = re.compile(r"^(docs/.*\.md|README\.md|CLAUDE\.md|PLAN\.md|CONTRIBUTING\.md)$")

DOC_GUARDS = ["tests/test_docs_counts.py", "tests/test_docs_section_count.py"]

#: The tests ABOUT deck artifacts: every tracked file's validator and freshness,
#: branches, the manifest, version lists, the dossier, and the commit protocol
#: (a decklist and its version list never in one commit). A decklist edit can
#: break these and nothing else in `src/`-free territory — the fleet-wide
#: calibrations re-derive constants from code and wait for CI. Measured: the
#: whole regression tier minus other decks took 5:13 for Edgar; this set ~1:45.
DECK_TESTS = ["tests/test_pilot_tracked_artifacts_validate.py",
              "tests/test_pilot_artifact_freshness.py",
              "tests/test_pilot_deck_branch.py", "tests/test_pilot_validate_branch.py",
              "tests/test_pilot_deck_manifest.py", "tests/test_pilot_deck_versions.py",
              "tests/test_pilot_deck_status.py", "tests/test_pilot_deck_info.py",
              "tests/test_pilot_commit_protocol.py"]


def _git(*args):
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return r.returncode, r.stdout


def changed_paths():
    """Paths a push would carry: commits ahead of the upstream, plus the working
    tree. None when there is no upstream to compare against."""
    rc, _ = _git("rev-parse", "--abbrev-ref", "@{u}")
    if rc != 0:
        return None
    _, ahead = _git("diff", "--name-only", "@{u}...HEAD")
    _, dirty = _git("status", "--porcelain", "--untracked-files=all")
    paths = set(ahead.split())
    for line in dirty.splitlines():
        p = line[3:].split(" -> ")[-1].strip().strip('"')
        if p:
            paths.add(p)
    return sorted(paths)


def classify(paths):
    """`paths` -> {"scope": "full"|"decks"|"docs"|"none", "decks": [...], "why": str}."""
    if paths is None:
        return {"scope": "full", "decks": [], "why": "no upstream to diff against"}
    if not paths:
        return {"scope": "none", "decks": [], "why": "nothing to push"}
    decks, code, docs, derived = set(), [], [], False
    for p in paths:
        m = _DECK_PATH.match(p) or _MANUAL_PATH.match(p)
        if m:
            decks.add(m.group(1))
        elif p in _DECK_DERIVED:
            derived = True
        elif _DOCS.match(p):
            docs.append(p)
        else:
            code.append(p)
    if code:
        shown = ", ".join(code[:3]) + (f" (+{len(code) - 3} more)" if len(code) > 3 else "")
        return {"scope": "full", "decks": sorted(decks), "why": f"code changed: {shown}"}
    if decks or derived:
        if not decks:
            # The manifest alone: no deck moved, so keep only the checks that
            # name none (its own freshness test is one of them).
            return {"scope": "decks", "decks": [], "why": "only fleet-derived files changed"}
        return {"scope": "decks", "decks": sorted(decks),
                "why": f"only deck data changed ({', '.join(sorted(decks))})"
                       + (" and docs" if docs else "")}
    return {"scope": "docs", "decks": [], "why": "only docs changed"}


def all_slugs():
    return sorted(d.name for d in DECKS.iterdir() if d.is_dir())


def deck_keyword(decks, every=None):
    """A `-k` expression that drops the tests parametrized on OTHER decks and keeps
    everything else — the changed decks' cases and every check that names none."""
    others = [s for s in (every or all_slugs()) if s not in set(decks)]
    return " and ".join(f"not {s}" for s in others) if others else ""


def commands(plan):
    """The commands a plan runs, in order (lists, run from the repo root)."""
    py = sys.executable
    if plan["scope"] == "full":
        return [["make", "prepush-full"]]
    if plan["scope"] == "none":
        return []
    if plan["scope"] == "docs":
        return [[py, "-m", "pytest", "-o", "addopts=", "-n0", "-q", *DOC_GUARDS]]
    k = deck_keyword(plan["decks"])
    # `-m ""`: these files hold both tiers' deck tests; `not regen` keeps the
    # in-place fleet regen (~6 min, every deck) for the full suite and CI.
    return [[py, "-m", "pytest", "-m", "not regen", "-n", "auto", "-o", "addopts=", "-q",
             *(["-k", k] if k else []), *DECK_TESTS],
            [py, "-m", "pytest", "-o", "addopts=", "-n0", "-q", *DOC_GUARDS]]


def run(plan):
    print(f"CHECK SCOPE — {plan['scope'].upper()}: {plan['why']}")
    if plan["scope"] == "decks":
        print("  (CI runs the full suite on every push; this is what blocks locally)")
    for cmd in commands(plan):
        print("  $ " + " ".join(cmd if len(" ".join(cmd)) < 160 else cmd[:8] + ["…"]))
        rc = subprocess.run(cmd, cwd=REPO).returncode
        if rc not in (0, 5):     # 5: nothing selected, e.g. a deck with no tests
            print(f"CHECK SCOPE — FAILED (exit {rc}): fix it before pushing")
            return rc
    print(f"CHECK SCOPE — PASSED ({plan['scope']})")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "plan"
    if cmd == "deck":
        slugs = argv[1:]
        unknown = [s for s in slugs if s not in all_slugs()]
        if not slugs or unknown:
            raise SystemExit(f"check-deck needs deck slugs; unknown: {unknown or '(none given)'}")
        return run({"scope": "decks", "decks": slugs, "why": f"asked for {', '.join(slugs)}"})
    plan = classify(changed_paths())
    if cmd == "plan":
        print(json.dumps({**plan, "commands": [" ".join(c) for c in commands(plan)]}, indent=1))
        return 0
    if cmd == "prepush":
        return run(plan)
    raise SystemExit(f"unknown command {cmd!r}: plan | prepush | deck <slug>...")


if __name__ == "__main__":
    sys.exit(main())
