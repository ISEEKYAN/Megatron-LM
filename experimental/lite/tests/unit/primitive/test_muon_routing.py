# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.lite.primitive.optimizers.megatron_wrap import build_dist_opt_stack
from megatron.lite.primitive.optimizers.muon_routing import tag_muon_parameter_metadata


class VocabParallelEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)


class _OutputColumn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 8, bias=False)


class VocabParallelOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.col = _OutputColumn()


class _QKVProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 12, bias=False)


class GQAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads_local = 2
        self.num_kv_heads_local = 1
        self.head_dim = 2
        self._output_gate = True
        self.qkv = _QKVProjection()


class _SyntheticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = VocabParallelEmbedding()
        self.attn = GQAttention()
        self.experts = nn.ModuleList([nn.Linear(4, 4, bias=False)])
        self.dense = nn.Linear(4, 4, bias=False)
        self.head = VocabParallelOutput()


def test_central_routing_excludes_embedding_and_output_and_tags_qkv() -> None:
    model = _SyntheticModel()

    tag_muon_parameter_metadata(
        [model], is_expert_param=lambda name: name.startswith("experts.")
    )

    embedding = model.embed.embedding.weight
    output = model.head.col.linear.weight
    qkv = model.attn.qkv.linear.weight

    assert embedding.is_embedding_or_output_parameter is True
    assert embedding.is_managed_by_layer_wise_optimizer is False
    assert output.is_embedding_or_output_parameter is True
    assert output.is_managed_by_layer_wise_optimizer is False
    assert qkv.is_qkv is True
    assert qkv.qkv_split_shapes == [4, 4, 2, 2]
    assert qkv.is_managed_by_layer_wise_optimizer is True


def test_central_routing_preserves_expert_and_tensor_parallel_metadata() -> None:
    model = _SyntheticModel()
    expert = model.experts[0].weight
    dense = model.dense.weight
    expert.tensor_model_parallel = True
    expert.partition_dim = 0
    expert.partition_stride = 2
    expert.allreduce = False
    dense.tensor_model_parallel = False
    dense.sequence_parallel = True

    tag_muon_parameter_metadata(
        [model], is_expert_param=lambda name: name.startswith("experts.")
    )

    assert expert.expert_tp is True
    assert expert.tensor_model_parallel is True
    assert expert.partition_dim == 0
    assert expert.partition_stride == 2
    assert expert.allreduce is False
    assert dense.tensor_model_parallel is False
    assert dense.sequence_parallel is True


def test_qkv_metadata_is_not_set_when_shape_is_incompatible() -> None:
    model = _SyntheticModel()
    model.attn.qkv.linear.weight = nn.Parameter(torch.empty(15, 4))

    tag_muon_parameter_metadata([model], is_expert_param=lambda _name: False)

    assert not getattr(model.attn.qkv.linear.weight, "is_qkv", False)


def test_dist_opt_entry_tags_metadata_then_fails_before_owner_group_or_step_lowering() -> (
    None
):
    model = _SyntheticModel()
    parallel = SimpleNamespace(vpp=1, pp=1, tp=1, ep=1, etp=1, cp=1)
    engine_config = SimpleNamespace(
        parallel=parallel,
        optimizer=SimpleNamespace(optimizer="muon"),
        deterministic=False,
    )

    with pytest.raises(
        NotImplementedError,
        match="full-param layout.*pg_collection owner groups.*WORLD/singleton",
    ):
        build_dist_opt_stack(
            [model],
            model_cfg=SimpleNamespace(),
            engine_cfg=engine_config,
            ps=None,
            is_expert=lambda name: name.startswith("experts."),
        )

    assert model.embed.embedding.weight.is_managed_by_layer_wise_optimizer is False
    assert model.attn.qkv.linear.weight.qkv_split_shapes == [4, 4, 2, 2]
    assert model.experts[0].weight.expert_tp is True
