"""Orchestrator: ordered registry of pipeline steps and the full-pipeline runner.

Steps are imported lazily (inside run()) so that listing them — e.g. for
`manamap --help` — doesn't pay the multi-second torch/sentence-transformers
import cost.
"""

import importlib
import time

# (name, dotted module, description) — execution order matters.
STEPS = [
    ("download", "manamap.ingest.download", "Step 1: Download Scryfall bulk data"),
    ("extract", "manamap.ingest.extract", "Step 2: Extract cards to CSV"),
    ("preprocess", "manamap.ingest.preprocess", "Step 3: Features + text embeddings"),
    ("train", "manamap.training.train", "Step 4a: Train color/type model"),
    ("train-ability", "manamap.training.train_ability", "Step 4b: Train ability model"),
    ("embed", "manamap.training.embed", "Step 5: Generate embeddings (both models)"),
    ("reduce", "manamap.export.reduce", "Step 6: PaCMAP 2D projections (both)"),
    ("download-combos", "manamap.ingest.download_combos", "Step 7: Download Commander Spellbook combos"),
    ("process-combos", "manamap.ingest.process_combos", "Step 8: Build combo graph"),
    ("export", "manamap.export.export_embeddings", "Step 9: Export embeddings .bin (both)"),
    ("synergy", "manamap.analysis.synergy", "Step 10: Build synergy graph"),
    ("power-creep", "manamap.analysis.power_creep", "Step 11: Build obsolescence index"),
    ("cluster-regions", "manamap.analysis.cluster_regions", "Step 12: Cluster + name map regions"),
    ("card-roles", "manamap.analysis.card_roles", "Step 13: Classify deckbuilding roles"),
    # Depends on embeddings, synergy, obsolescence AND card-roles, so it has to come
    # after all of them — it is the last producer in the chain.
    ("viz-index", "manamap.export.viz_index", "Step 14: Discovery index + neighbour tables"),
    # The only step that writes no artifact — it reports embedding quality against a
    # hand-authored golden set. Last so a full run ends by saying whether the thing it
    # just spent an hour training actually represents similarity.
    ("eval-embeddings", "manamap.analysis.eval_embeddings", "Step 15: Report embedding quality"),
]

STEP_NAMES = [name for name, _, _ in STEPS]


#: Steps whose `main` takes a `space=` keyword. Named explicitly rather than
#: sniffed with `inspect`, so adding a space-aware step is a deliberate edit and
#: a typo fails loudly instead of silently falling back to the default space.
SPACE_AWARE = {"reduce", "export", "cluster-regions", "viz-index"}


def run_step(name, position=None, space=None):
    """Run a single pipeline step by registry name.

    The step banner stays on STDOUT. It is not theatre — it is the pipeline's
    own record, and `manamap run > pipeline.log` is a thing people do. What is
    new is the ELAPSED time, printed after the step rather than during it, and
    the optional `N/M` position when running the whole thing.

    Timing is the point: `--from` is chosen by remembering which step is
    expensive, and nothing has ever printed the numbers to remember.
    """
    for step_name, module_path, description in STEPS:
        if step_name == name:
            head, body = description.split(":")[0], description.split(": ", 1)[1]
            where = f" {position}" if position else ""
            print(f"\n[{head}{where}] {body}")
            started = time.monotonic()
            step_main = importlib.import_module(module_path).main
            if step_name in SPACE_AWARE:
                step_main(space=space)
            elif space is not None:
                raise SystemExit(
                    f"step {step_name!r} is not space-aware "
                    f"(space-aware: {', '.join(sorted(SPACE_AWARE))})")
            else:
                step_main()
            print(f"    ✓ {head.lower()} finished in {_duration(time.monotonic() - started)}")
            return
    raise ValueError(f"Unknown step: {name!r} (choose from {', '.join(STEP_NAMES)})")


def _duration(seconds):
    """Human time. A pipeline step runs from a second to an hour, and `3612.4s`
    is a number the reader has to convert before it means anything."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def run(start=None):
    """Run the full pipeline in order, optionally starting from a given step."""
    if start is not None and start not in STEP_NAMES:
        raise ValueError(f"Unknown step: {start!r} (choose from {', '.join(STEP_NAMES)})")

    print("=" * 50)
    print("Mana Map — Data Pipeline")
    print("=" * 50)

    started = start is None
    todo = []
    for name, _, _ in STEPS:
        if not started and name == start:
            started = True
        if started:
            todo.append(name)

    began = time.monotonic()
    timings = []
    for i, name in enumerate(todo, 1):
        step_began = time.monotonic()
        run_step(name, position=f" {i}/{len(todo)}")
        timings.append((name, time.monotonic() - step_began))

    print("\n" + "=" * 50)
    print(f"Pipeline complete — {len(todo)} step(s) in {_duration(time.monotonic() - began)}")
    print("=" * 50)
    # The slowest steps, named. This is what makes `--from` an evidence-based
    # choice on the next run instead of a guess about which step was the long one.
    slowest = sorted(timings, key=lambda kv: kv[1], reverse=True)[:3]
    if len(timings) > 3:
        print("  slowest: " + " · ".join(f"{n} {_duration(d)}" for n, d in slowest))


def main():
    run()


if __name__ == "__main__":
    main()
