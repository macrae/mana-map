"""Step 9: Convert embeddings.npy to raw Float32 binary for JS consumption."""

import numpy as np

from config import (
    ABILITY_EMBEDDINGS_BIN_PATH,
    ABILITY_EMBEDDINGS_PATH,
    EMBEDDINGS_BIN_PATH,
    EMBEDDINGS_PATH,
)


def export_bin(npy_path, bin_path):
    """Convert a .npy embedding file to raw Float32 binary."""
    print(f"  Loading {npy_path}...")
    embeddings = np.load(npy_path)
    print(f"    Shape: {embeddings.shape}")

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    embeddings.tofile(bin_path)

    size_mb = bin_path.stat().st_size / (1024 * 1024)
    expected = embeddings.shape[0] * embeddings.shape[1] * 4
    actual = bin_path.stat().st_size
    assert actual == expected, f"Size mismatch: {actual} bytes vs expected {expected}"

    print(f"    Wrote {bin_path} ({size_mb:.1f} MB)")
    print(f"    {embeddings.shape[0]} cards x {embeddings.shape[1]} dims = {embeddings.size:,} floats")


def main():
    print("Exporting default embeddings binary...")
    export_bin(EMBEDDINGS_PATH, EMBEDDINGS_BIN_PATH)

    if ABILITY_EMBEDDINGS_PATH.exists():
        print("\nExporting ability embeddings binary...")
        export_bin(ABILITY_EMBEDDINGS_PATH, ABILITY_EMBEDDINGS_BIN_PATH)
    else:
        print(f"\n  Skipping ability binary ({ABILITY_EMBEDDINGS_PATH} not found)")


if __name__ == "__main__":
    main()
