"""Shared pytest fixtures and markers for the mana-map test suite."""

import pytest

from manamap import config

# Marker for tests that need generated data/ artifacts (run the pipeline first).
# Gate on embeddings.npy — the last artifact of the train/embed stage — so a
# partially-populated data/ dir still skips cleanly.
requires_data = pytest.mark.skipif(
    not config.EMBEDDINGS_PATH.exists(),
    reason="requires generated data/ artifacts (run `manamap run` first)",
)


@pytest.fixture(scope="session")
def data_dir():
    """The resolved data directory (honors MANAMAP_DATA_DIR)."""
    return config.DATA_DIR
