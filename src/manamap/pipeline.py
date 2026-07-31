"""Orchestrator: ordered registry of pipeline steps and the full-pipeline runner.

Steps are imported lazily (inside run()) so that listing them — e.g. for
`manamap --help` — doesn't pay the multi-second torch/sentence-transformers
import cost.
"""

import importlib

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


def run_step(name):
    """Run a single pipeline step by registry name."""
    for step_name, module_path, description in STEPS:
        if step_name == name:
            print(f"\n[{description.split(':')[0]}] {description.split(': ', 1)[1]}")
            importlib.import_module(module_path).main()
            return
    raise ValueError(f"Unknown step: {name!r} (choose from {', '.join(STEP_NAMES)})")


def run(start=None):
    """Run the full pipeline in order, optionally starting from a given step."""
    if start is not None and start not in STEP_NAMES:
        raise ValueError(f"Unknown step: {start!r} (choose from {', '.join(STEP_NAMES)})")

    print("=" * 50)
    print("Mana Map — Data Pipeline")
    print("=" * 50)

    started = start is None
    for name, _, _ in STEPS:
        if not started and name == start:
            started = True
        if started:
            run_step(name)

    print("\n" + "=" * 50)
    print("Pipeline complete.")
    print("=" * 50)


def main():
    run()


if __name__ == "__main__":
    main()
