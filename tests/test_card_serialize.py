"""The masked card serialisation — what the imputation model actually sees.

The contrastive model needs labels and mines them from the repo's own regexes.
Masked imputation needs none: the label is the input. These tests guard the
three ways that argument can be quietly broken.
"""

import numpy as np
import pytest

from manamap.training import card_serialize as CS


def _card(**kw):
    base = {"type_line": "Creature — Elf Druid", "mana_cost": "{G}",
            "power": "1", "toughness": "1", "oracle_text": "{T}: Add {G}."}
    base.update(kw)
    return base


def test_absent_is_empty_never_the_string_nan():
    """RE-INTRODUCING A BUG THAT SHIPPED IN THE FIRST CUT. pandas hands every
    missing cell through as a float NaN and `str(nan)` is `"nan"`, so a Forest
    serialised as `[PT] nan/nan [COST] nan`. The model would have learned "nan"
    as a token meaning absent — a vocabulary item standing in for a fact the
    sentinel already carries."""
    land = _card(mana_cost=float("nan"), power=float("nan"),
                 toughness=float("nan"), type_line="Basic Land — Forest",
                 oracle_text="({T}: Add {G}.)")
    out = CS.serialize(land)
    assert "nan" not in out.lower().split(), out
    assert "[COST]" in out and "[PT]" in out, "the sentinels stay; only the body goes"
    assert CS.blocks_for(land)["stats"] == ""


def test_the_sentinel_survives_masking():
    """A masked block and an absent block are DIFFERENT FACTS. Dropping the
    sentinel collapses them, and a masked model then learns to predict "empty"."""
    out = CS.serialize(_card(), "type")
    assert "[TYPE] [MASK]" in out
    assert "[COST] {G}" in out, "only the masked block is hidden"


def test_keywords_are_not_a_block():
    """MEASURED, not assumed: 99.0% of a card's Scryfall keywords appear verbatim
    in its own oracle text and 98.1% of cards have every one there. Masking the
    block is a copy from [TEXT]; leaving it visible LEAKS the text when the text
    is what is hidden."""
    assert "keywords" not in CS.BLOCKS
    out = CS.serialize(_card(keywords="Flying,Trample"))
    assert "[KW]" not in out and "Trample" not in out


def test_tags_and_roles_can_never_be_a_target():
    """THE RULE THAT KEEPS THE BOOTSTRAPPING OUT. MECHANICAL_TAGS and
    ROLE_PATTERNS are regexes over the oracle text; training the model to predict
    them re-imports the exact supervision this architecture removes, one layer
    down. Asking for them must be an error, not a silent no-op."""
    from manamap.config import MECHANICAL_TAG_NAMES, ROLE_NAMES

    assert "tags" not in CS.BLOCKS and "roles" not in CS.BLOCKS
    for block in ("tags", "roles"):
        with pytest.raises(ValueError, match="not maskable"):
            CS.serialize(_card(), block)
    body = CS.serialize(_card(oracle_text="When this enters, draw a card."))
    for name in list(MECHANICAL_TAG_NAMES)[:5] + list(ROLE_NAMES)[:5]:
        assert f"[{name}]" not in body


def test_the_ability_line_boundary_is_restored():
    """`extract.py:157` flattens newlines for `embedding_text`, so two unrelated
    abilities become one run-on string. Fine for a pooled vector, wrong for a
    model that should learn what an ability IS."""
    card = _card(oracle_text="Flying\n{T}: Add {G}.\nSacrifice this: Draw a card.")
    out = CS.serialize(card)
    assert out.count(CS.LINE) == 2
    assert "Flying [LINE] {T}: Add {G}." in out


def test_an_empty_block_is_kept_rather_than_dropped():
    """409 cards have no oracle text. A model that never sees an empty [TEXT]
    cannot represent a vanilla creature."""
    out = CS.serialize(_card(oracle_text=""))
    assert "[TEXT]" in out
    assert CS.blocks_for(_card(oracle_text=""))["text"] == ""


def test_targets_are_the_hidden_content():
    card = _card()
    targets = CS.targets_for(card, ("type", "cost"))
    assert targets == {"type": "Creature — Elf Druid", "cost": "{G}"}
    assert "Creature — Elf Druid" not in CS.serialize(card, ("type", "cost"))


def test_masking_never_hides_everything():
    """A median card is 37 subword tokens. Hiding three blocks leaves nothing to
    condition on, and the model would be predicting from the sentinels alone."""
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(3000):
        mask = CS.sample_mask(rng)
        assert 1 <= len(mask) <= 2, mask
        seen.add(mask)
    assert {("text",), ("type",), ("cost",), ("stats",)} <= seen
    assert any(len(m) == 2 for m in seen), "the harder two-block case should occur"


def test_every_weighted_block_is_maskable():
    """A weight naming a block that `serialize` refuses would fail deep inside
    training, one batch at a time."""
    assert set(CS.MASK_WEIGHTS) <= set(CS.BLOCKS)
    assert abs(sum(CS.MASK_WEIGHTS.values()) - 1.0) < 1e-9
    for block in CS.MASK_WEIGHTS:
        CS.serialize(_card(), block)


def test_real_cards_serialise_without_leaking_pandas_sentinels():
    import pandas as pd
    import re

    from manamap.config import OUTPUT_CSV_PATH

    if not OUTPUT_CSV_PATH.exists():
        pytest.skip("corpus not built")
    frame = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False).sample(2000, random_state=0)
    checked = 0
    for card in frame.to_dict("records"):
        out = CS.serialize(card)
        assert not re.search(r"\bnan\b", out, re.IGNORECASE), card["name"]
        assert out.startswith("[TYPE]")
        checked += 1
    assert checked == 2000
