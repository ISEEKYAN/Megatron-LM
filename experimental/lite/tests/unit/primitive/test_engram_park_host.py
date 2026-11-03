# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Frozen Engram host shards can be released to a file and read back byte for byte."""
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.primitive.modules.engram import ShardedFrozenTable


def test_park_releases_and_unpark_restores_exact_bytes(tmp_path):
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    table = ShardedFrozenTable(64, 32, group)
    weight = table.embedding.weight.view(torch.uint8).clone()
    scale = table.embedding.scale.view(torch.uint8).clone()
    path = str(tmp_path / 'shard.bin')

    table.park_host(path)
    assert table.embedding.weight.numel() == 0 and table.embedding.scale.numel() == 0
    with pytest.raises(RuntimeError, match='parked'):
        table(torch.zeros(2, dtype=torch.long))
    table.park_host(path)  # parking twice is a no-op

    table.unpark_host(path)
    assert torch.equal(table.embedding.weight.view(torch.uint8), weight)
    assert torch.equal(table.embedding.scale.view(torch.uint8), scale)
    table.unpark_host(path)  # unparking an unparked table is a no-op
