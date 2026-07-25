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

# Pilot subsystem artifact gates (same pattern).
requires_rules = pytest.mark.skipif(
    not config.RULES_INDEX_PATH.exists(),
    reason="requires the rules DB (run `manamap pilot download-rules && manamap pilot build-rules-db`)",
)

requires_deck = pytest.mark.skipif(
    not (config.DECKS_DIR / "goblin-storm" / "cards.json").exists(),
    reason="requires a fetched deck (run `manamap pilot fetch-deck goblin-storm`)",
)

requires_strategy = pytest.mark.skipif(
    not config.STRATEGY_INDEX_PATH.exists(),
    reason="requires the strategy DB (run `manamap pilot build-strategy-db`)",
)


@pytest.fixture(scope="session")
def data_dir():
    """The resolved data directory (honors MANAMAP_DATA_DIR)."""
    return config.DATA_DIR
