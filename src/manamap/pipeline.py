"""Orchestrator: runs the full data + embedding pipeline."""

from manamap.analysis import cluster_regions, power_creep, synergy
from manamap.export import export_embeddings, reduce
from manamap.ingest import download, download_combos, extract, preprocess, process_combos
from manamap.training import embed, train, train_ability


def main():
    print("=" * 50)
    print("Mana Map — Data Pipeline")
    print("=" * 50)

    print("\n[Step 1] Download")
    download.main()

    print("\n[Step 2] Extract")
    extract.main()

    print("\n[Step 3] Preprocess")
    preprocess.main()

    print("\n[Step 4a] Train (Color + Type)")
    train.main()

    print("\n[Step 4b] Train (Abilities)")
    train_ability.main()

    print("\n[Step 5] Embed (both models)")
    embed.main()

    print("\n[Step 6] Reduce (both projections)")
    reduce.main()

    print("\n[Step 7] Download Combos")
    download_combos.main()

    print("\n[Step 8] Process Combos")
    process_combos.main()

    print("\n[Step 9] Export Embeddings Binary (both)")
    export_embeddings.main()

    print("\n[Step 10] Build Synergy Graph")
    synergy.main()

    print("\n[Step 11] Build Obsolescence Index")
    power_creep.main()

    print("\n[Step 12] Cluster Regions")
    cluster_regions.main()

    print("\n" + "=" * 50)
    print("Pipeline complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
