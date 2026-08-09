# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import sys
import types
from dataclasses import field as dataclass_field
from dataclasses import fields, make_dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.lite.primitive.optimizers.megatron_wrap import (
    build_dist_opt_optimizer_config,
    build_dist_opt_stack,
)
from megatron.lite.primitive.optimizers.muon_routing import tag_muon_parameter_metadata
from megatron.lite.runtime.contracts.config import OptimizerConfig


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


def test_native_fields_route_without_rewriting_and_legacy_offload_keeps_compat(
    monkeypatch,
) -> None:
    core_field_names = {
        "optimizer",
        "lr",
        "min_lr",
        "weight_decay",
        "clip_grad",
        "use_distributed_optimizer",
        "bf16",
        "params_dtype",
        *(item.name for item in fields(OptimizerConfig)),
    }
    fake_core_config = make_dataclass(
        "FakeCoreOptimizerConfig",
        [
            (name, object, dataclass_field(default=None))
            for name in sorted(core_field_names)
        ],
    )
    fake_module = types.ModuleType("megatron.core.optimizer.optimizer_config")
    fake_module.OptimizerConfig = fake_core_config
    monkeypatch.setitem(
        sys.modules, "megatron.core.optimizer.optimizer_config", fake_module
    )

    config = OptimizerConfig(
        optimizer="muon",
        muon_momentum=0.9,
        muon_split_qkv=False,
        muon_nesterov=True,
        muon_scale_mode="unit_rms_norm",
        muon_fp32_matmul_prec="high",
        muon_coefficient_type="simple",
        muon_num_ns_steps=7,
        muon_tp_mode="duplicated",
        muon_extra_scale_factor=0.2,
        use_layer_wise_param_layout=True,
        overlap_param_gather=True,
        optimizer_cpu_offload=True,
        optimizer_offload_fraction=0.25,
        use_torch_optimizer_for_cpu_offload=True,
        overlap_cpu_optimizer_d2h_h2d=False,
        pin_cpu_grads=False,
        pin_cpu_params=False,
        offload_optimizer_states=True,
    )

    core = build_dist_opt_optimizer_config(config)

    assert core.muon_momentum == 0.9
    assert core.muon_split_qkv is False
    assert core.muon_nesterov is True
    assert core.muon_scale_mode == "unit_rms_norm"
    assert core.muon_fp32_matmul_prec == "high"
    assert core.muon_coefficient_type == "simple"
    assert core.muon_num_ns_steps == 7
    assert core.muon_tp_mode == "duplicated"
    assert core.muon_extra_scale_factor == 0.2
    assert core.muon_scalar_optimizer == "adam"
    assert core.use_layer_wise_param_layout is True
    assert core.overlap_param_gather is True
    assert core.overlap_param_gather_with_optimizer_step is False
    assert core.optimizer_cpu_offload is True
    assert core.optimizer_offload_fraction == 0.25
    assert core.use_torch_optimizer_for_cpu_offload is True
    assert core.overlap_cpu_optimizer_d2h_h2d is False
    assert core.pin_cpu_grads is False
    assert core.pin_cpu_params is False
    assert core.offload_optimizer_states is True

    legacy_core = build_dist_opt_optimizer_config(
        OptimizerConfig(offload_fraction=0.5, overlap_cpu_optimizer_d2h_h2d=False)
    )

    assert legacy_core.optimizer_cpu_offload is True
    assert legacy_core.optimizer_offload_fraction == 0.5
    assert legacy_core.overlap_cpu_optimizer_d2h_h2d is True
