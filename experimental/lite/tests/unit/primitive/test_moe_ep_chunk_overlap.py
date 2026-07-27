# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import pytest
from megatron.lite.primitive.moe_ep_chunk_overlap_policy import (
    ep_chunk_ranges,
    parse_ep_chunk_spec,
    resolve_ep_chunk_overlap_chunks,
)


@pytest.mark.parametrize("value, expected", [(1, 1), ("2", 2), ("auto", "auto")])
def test_chunk_spec_is_explicit_and_validated(value, expected):
    assert parse_ep_chunk_spec(value) == expected


@pytest.mark.parametrize("value", [0, -1, "zero", "1.5"])
def test_chunk_spec_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_ep_chunk_spec(value)


def test_chunk_ranges_cover_each_token_exactly_once():
    ranges = ep_chunk_ranges(11, 4)

    assert ranges == [(0, 3), (3, 6), (6, 9), (9, 11)]
    assert [token for start, end in ranges for token in range(start, end)] == list(
        range(11)
    )


def test_auto_policy_keeps_small_inputs_whole_and_chunks_large_ep_inputs():
    assert (
        resolve_ep_chunk_overlap_chunks(
            8192, ep_size=8, hidden_size=4096, direction="forward"
        )
        == 1
    )
    assert (
        resolve_ep_chunk_overlap_chunks(
            32768, ep_size=8, hidden_size=4096, direction="forward"
        )
        > 1
    )


def test_production_chunked_ep_layer_is_importable():
    from megatron.lite.primitive.modules.moe_ep_chunk_overlap import (
        EPChunkOverlapMoELayer,
    )

    assert EPChunkOverlapMoELayer.__name__ == "EPChunkOverlapMoELayer"
