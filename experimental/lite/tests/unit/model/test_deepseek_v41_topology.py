# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from dataclasses import FrozenInstanceError, replace

import pytest
from megatron.lite.model.deepseek_v41.topology import (
    V41_TOPOLOGY,
    TopologySpec,
    build_topology,
    pipeline_stage_policies,
)


def test_release_layer_policies():
    assert len(V41_TOPOLOGY) == 40
    expected_kv = [None] * 2 + [2] * 6 + [8] * 6 + [14] * 6 + [20] * 20
    expected_index = expected_kv[:20] + [i for i in range(20, 40, 4) for _ in range(4)]
    for i, policy in enumerate(V41_TOPOLOGY):
        assert policy.index == i
        assert policy.compress_ratio == (0 if i < 2 else 2 if i < 20 else 1)
        assert policy.kv_owner == expected_kv[i]
        assert policy.index_owner == expected_index[i]
        assert policy.candidate_mode == (
            "none" if i < 20 else "build" if i == 20 else "reuse"
        )
        assert policy.engram_rows == {1: 384006168, 14: 384016682}.get(i, 0)
        assert policy.engram_slot == {1: 0, 14: 1}.get(i)
        assert policy.is_ced_boundary == (i == 19)
    with pytest.raises(FrozenInstanceError):
        V41_TOPOLOGY[0].compress_ratio = 1


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"kv_source_layer_ids": (2, 8, 8, 20)}, "strictly increasing"),
        ({"index_source_layer_ids": (2, 14, 8, 20)}, "strictly increasing"),
        ({"kv_source_layer_ids": (2, 8, 14, 40)}, "range"),
        ({"engram_layer_ids": (-1, 14)}, "range"),
        ({"engram_layer_ids": (1, 1)}, "strictly increasing"),
        ({"engram_num_embeddings": (10,)}, "one row count"),
        ({"engram_num_embeddings": (10, 0)}, "positive"),
        ({"compress_ratios": (0,) * 43}, "source.*uncompressed"),
        ({"compress_ratios": (0, 0) + (3,) * 38 + (0,) * 3}, "ratios"),
        ({"compress_ratios": (0, 0) + (2,) * 17 + (1,) * 21 + (0,) * 3}, "owner ratio"),
        ({"kv_source_layer_ids": (8, 14, 20)}, "no KV owner"),
        ({"index_source_layer_ids": (8, 14, 20)}, "no index owner"),
        ({"candidate_source_layer_id": 21}, "candidate source"),
        ({"compress_ratios": (0,) * 40}, "backbone and archival"),
        ({"compress_ratios": (0, 0) + (2,) * 18 + (1,) * 23}, "archival"),
        ({"num_hidden_layers": 0}, "positive"),
        ({"kv_source_layer_ids": (True, 8, 14, 20)}, "integer"),
    ],
)
def test_invalid_topology(changes, message):
    with pytest.raises(ValueError, match=message):
        build_topology(replace(TopologySpec(), **changes))


def test_policies_derive_from_sources_instead_of_release_magic_numbers():
    spec = TopologySpec(
        num_hidden_layers=8,
        compress_ratios=(0, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0),
        kv_source_layer_ids=(1, 4),
        index_source_layer_ids=(1, 4, 6),
        candidate_source_layer_id=4,
        engram_layer_ids=(0, 5),
        engram_num_embeddings=(11, 17),
    )
    policies = build_topology(spec)
    assert [p.kv_owner for p in policies] == [None, 1, 1, 1, 4, 4, 4, 4]
    assert [p.index_owner for p in policies] == [None, 1, 1, 1, 4, 4, 6, 6]
    assert [p.index for p in policies if p.is_ced_boundary] == [3]
    assert [p.engram_rows for p in policies] == [11, 0, 0, 0, 0, 17, 0, 0]
    assert [p.candidate_mode for p in policies] == ["none"] * 4 + ["build"] + [
        "reuse"
    ] * 3


def test_pp2_ownership_does_not_cross_natural_stage_boundary():
    stages = pipeline_stage_policies(V41_TOPOLOGY, 2)
    assert tuple(p.index for p in stages[0]) == tuple(range(20))
    assert tuple(p.index for p in stages[1]) == tuple(range(20, 40))
    for stage in stages:
        local = {p.index for p in stage}
        builder = next((p.index for p in stage if p.candidate_mode == "build"), None)
        for p in stage:
            assert p.kv_owner is None or p.kv_owner in local
            assert p.index_owner is None or p.index_owner in local
            assert p.candidate_mode != "reuse" or builder is not None
    assert pipeline_stage_policies(V41_TOPOLOGY, 1) == (V41_TOPOLOGY,)
    with pytest.raises(ValueError, match="crosses pipeline"):
        pipeline_stage_policies(V41_TOPOLOGY, 4)
    with pytest.raises(ValueError, match="evenly"):
        pipeline_stage_policies(V41_TOPOLOGY, 3)
