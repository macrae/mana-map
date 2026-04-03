"""Orchestrator: runs the full data + embedding pipeline."""

import download
import embed
import extract
import preprocess
import reduce
import train


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

    print("\n[Step 4] Train")
    train.main()

    print("\n[Step 5] Embed")
    embed.main()

    print("\n[Step 6] Reduce")
    reduce.main()

    print("\n" + "=" * 50)
    print("Pipeline complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
