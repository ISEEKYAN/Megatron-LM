# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Models built from renumbered release layers keep the release Engram hash."""
from functools import partial

import pytest
import torch
from megatron.lite.primitive.modules import engram_lookup as memory
from megatron.lite.primitive.modules.row_memory_build import build_row_memories
from test_redo_parity import release_config

ORDER, HEADS, VOCABULARY, COMPRESSED = 3, 2, 10, 7


def _build(layer_ids, hash_layer_ids=None):
    primes = memory.prime_buckets((1, 14), ORDER, HEADS, VOCABULARY)
    return build_row_memories(
        layer_ids=layer_ids,
        row_counts=primes.flatten(1).sum(1).tolist(),
        order=ORDER,
        heads=HEADS,
        vocabulary=VOCABULARY,
        width=32,
        compressed_vocabulary=COMPRESSED,
        pad_id=0,
        token_map=list(range(COMPRESSED)),
        token_count=COMPRESSED,
        hidden_size=32,
        copies=4,
        eps=1e-6,
        trainable=False,
        group=None,
        group_size=1,
        local_range=(0, 0),
        constructors=(
            memory.prime_buckets,
            memory.hash_multipliers,
            memory.NgramHash,
            memory.RowLookup,
            memory.EngramTable,
            memory.ShardedEngramTable,
            memory.Engram,
        ),
        projection=partial(torch.nn.Linear, bias=False),
        hash_layer_ids=hash_layer_ids,
    )[0]


def test_hash_defaults_to_table_layer_ids():
    expected = memory.hash_multipliers((1, 14), ORDER, COMPRESSED)
    torch.testing.assert_close(_build((1, 14)).multipliers, expected, rtol=0, atol=0)


def test_renumbered_tables_keep_release_hash():
    release = _build((1, 14))
    renumbered = _build((0, 1), hash_layer_ids=(1, 14))
    for name in ('multipliers', 'primes', 'offsets'):
        torch.testing.assert_close(
            getattr(renumbered, name), getattr(release, name), rtol=0, atol=0
        )
    assert not torch.equal(_build((0, 1)).multipliers, release.multipliers)


@pytest.mark.parametrize('hash_ids', [[14], [1, -14], [1, 14.0]])
def test_config_rejects_malformed_hash_layer_ids(hash_ids):
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    release = release_config().to_hf_dict()
    release['text_config']['engram_hash_layer_ids'] = hash_ids
    with pytest.raises(ValueError, match='engram_hash_layer_ids'):
        DeepseekV41Config(release)


def test_config_round_trips_hash_layer_ids():
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    release = release_config().to_hf_dict()
    release['text_config']['engram_hash_layer_ids'] = [1, 14]
    config = DeepseekV41Config(release)
    assert config.to_hf_dict()['text_config']['engram_hash_layer_ids'] == [1, 14]
