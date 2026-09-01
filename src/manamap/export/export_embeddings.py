"""Step 9: Convert embeddings.npy to raw Float32 binary for JS consumption."""

import numpy as np

from manamap.config import (
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
    # UNIT ROWS ARE A CONTRACT, NOT A CONVENIENCE. `viz/js/mana-map.js:1417`
    # treats the dot product AS the cosine and never renormalises, so a space
    # exported unnormalised makes every similarity in the browser wrong without
    # erroring anywhere. Asserted rather than applied: the producer should be
    # normalising, and silently fixing it here would hide which one is not.
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), (
        f"{npy_path.name} rows are not unit norm "
        f"(min {norms.min():.4f}, max {norms.max():.4f}) — the browser reads a "
        f"dot product as a cosine and never renormalises")
    embeddings.tofile(bin_path)

    size_mb = bin_path.stat().st_size / (1024 * 1024)
    expected = embeddings.shape[0] * embeddings.shape[1] * 4
    actual = bin_path.stat().st_size
    assert actual == expected, f"Size mismatch: {actual} bytes vs expected {expected}"

    print(f"    Wrote {bin_path} ({size_mb:.1f} MB)")
    print(f"    {embeddings.shape[0]} cards x {embeddings.shape[1]} dims = {embeddings.size:,} floats")


def main(space=None):
    """Export the browser binaries.

    With no `space`, exports what the pipeline has always exported: the layout
    binary and the function binary. With one, exports only that space — which is
    how a second similarity space gets its `.bin` without re-running the rest.
    """
    from manamap import spaces as space_registry

    if space is not None:
        target = space_registry.get(space)
        if target.bin is None:
            raise SystemExit(
                f"the {target.slug!r} space exports no binary — "
                f"{target.note}")
        if not target.exists():
            raise SystemExit(f"{target.npy} not found — build it first")
        print(f"Exporting {target.label} binary...")
        export_bin(target.npy, target.bin)
        return

    print("Exporting default embeddings binary...")
    export_bin(EMBEDDINGS_PATH, EMBEDDINGS_BIN_PATH)

    if ABILITY_EMBEDDINGS_PATH.exists():
        print("\nExporting ability embeddings binary...")
        export_bin(ABILITY_EMBEDDINGS_PATH, ABILITY_EMBEDDINGS_BIN_PATH)
    else:
        print(f"\n  Skipping ability binary ({ABILITY_EMBEDDINGS_PATH} not found)")


if __name__ == "__main__":
    main()
