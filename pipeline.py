"""Orchestrator: runs the full data + embedding pipeline."""

import download
import download_combos
import embed
import export_embeddings
import extract
import preprocess
import process_combos
import reduce
import train
import train_ability


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

    # Steps 10-11 run after combo/synergy data exists
    try:
        import synergy
        print("\n[Step 10] Build Synergy Graph")
        synergy.main()
    except ImportError:
        print("\n  [Step 10] Skipping synergy (module not yet created)")

    try:
        import power_creep
        print("\n[Step 11] Build Obsolescence Index")
        power_creep.main()
    except ImportError:
        print("\n  [Step 11] Skipping power creep (module not yet created)")

    print("\n" + "=" * 50)
    print("Pipeline complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
