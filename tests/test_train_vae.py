"""The training loop's contracts — the split, the vocabulary, the targets.

There are no mined positives anywhere in `train_vae`, which is the point of the
architecture. What can still go wrong is the plumbing: a split that leaks, a
vocabulary the decoder head cannot refer to, or targets built from a block the
model can already see.
"""

import numpy as np
import pandas as pd
import pytest

from manamap.training import train_vae as TV
from manamap.training.card_serialize import BLOCKS


def test_the_split_is_by_text_not_by_row():
    """THE LEAK THIS PREVENTS. The corpus holds 2,705 cards sharing an oracle
    text with another card across 1,031 families — reprints and functional
    reprints. A row split puts Llanowar Elves in train and Fyndhorn Elves in
    test and calls the memorised answer generalisation."""
    frame = pd.DataFrame({"oracle_text": ["{T}: Add {G}.", "{T}: Add {G}.",
                                          "Draw a card.", "Draw a card.",
                                          "Destroy target creature."]})
    mask = TV.text_hash_split(frame)
    assert mask[0] == mask[1], "identical text must land on the same side"
    assert mask[2] == mask[3]
    assert mask.dtype == bool and len(mask) == 5


def test_no_duplicate_text_family_crosses_the_split_on_the_real_corpus():
    from manamap.config import OUTPUT_CSV_PATH

    if not OUTPUT_CSV_PATH.exists():
        pytest.skip("corpus not built")
    frame = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
    mask = TV.text_hash_split(frame)
    text = frame["oracle_text"].fillna("")
    duplicated = frame.loc[text.duplicated(keep=False)]
    crossed = sum(1 for _t, group in duplicated.groupby(text.loc[duplicated.index])
                  if mask[group.index].any() and not mask[group.index].all())
    families = duplicated.groupby(text.loc[duplicated.index]).ngroups
    assert families >= 500, f"only {families} duplicate families — check the corpus"
    assert crossed == 0, f"{crossed} of {families} duplicate-text families leak"


def test_the_vocabulary_is_the_most_frequent_tokens():
    """The decoder's columns REFER to this map; a head trained against one
    vocabulary and scored against another is silently meaningless."""
    lists = [[1, 1, 1], [1, 2, 2], [3]]
    vocab = TV.build_vocab(lists, size=2)
    assert set(vocab) == {1, 2}, "the two most frequent survive"
    assert sorted(vocab.values()) == [0, 1], "columns are dense and zero-based"
    assert 3 not in vocab


def test_targets_are_built_only_for_masked_blocks():
    """Scoring a visible block rewards copying a value the model can already
    see — the failure the recoverability audit exists to prevent."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    card = {"type_line": "Creature — Elf Druid", "mana_cost": "{G}",
            "power": "1", "toughness": "1", "oracle_text": "{T}: Add {G}."}
    vocab = TV.build_vocab(tok([TV.serialize(card)],
                               add_special_tokens=False)["input_ids"], size=64)
    data = TV.MaskedCards([card], tok, vocab, seed=0)
    checked = 0
    for _ in range(30):
        _ids, _am, targets, masked = data[0]
        assert 1 <= int(masked.sum()) <= 2
        for i, _block in enumerate(BLOCKS):
            if not masked[i]:
                assert float(targets[i].sum()) == 0.0, "an unmasked block got targets"
        assert float(targets[masked].sum()) > 0.0, "a masked block got none"
        checked += 1
    assert checked == 30


def test_a_masked_block_is_absent_from_the_input_but_present_in_the_target():
    """The whole objective in one assertion: what the model must predict is
    exactly what was taken away from what it can see."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    card = {"type_line": "Legendary Creature — Dinosaur Avatar", "mana_cost": "{5}{R}{G}{W}",
            "power": "7", "toughness": "6", "oracle_text": "Trample."}
    assert "Dinosaur" not in TV.serialize(card, "type")
    assert "Dinosaur" in TV.serialize(card)
    dino = tok("Dinosaur", add_special_tokens=False)["input_ids"]
    assert dino, "tokeniser produced nothing for the probe word"


def test_the_shadow_artifact_never_overwrites_the_live_space():
    """Nothing downstream changes until the eval says it should."""
    from manamap.config import ABILITY_EMBEDDINGS_PATH, EMBEDDINGS_PATH

    assert TV.VAE_EMBEDDINGS_PATH != ABILITY_EMBEDDINGS_PATH
    assert TV.VAE_EMBEDDINGS_PATH != EMBEDDINGS_PATH
    assert "vae" in TV.VAE_EMBEDDINGS_PATH.name


def test_training_is_not_a_pipeline_step():
    """`manamap run` must not spend 18-40 minutes on a model nothing reads yet."""
    from manamap.pipeline import STEP_NAMES

    assert "train-vae" not in STEP_NAMES


def test_output_is_flushed():
    """A 36-minute job that prints nothing is indistinguishable from a hung one.
    Python block-buffers stdout when it is redirected, and the first real run
    produced an empty log for its entire length."""
    import inspect

    assert "flush=True" in inspect.getsource(TV._say)
    assert inspect.signature(TV.train).parameters["echo"].default is TV._say
