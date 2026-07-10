# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pinned, bitwise Muon x compact DistOpt correctness harness.

The reference path deliberately spells out the Megatron-Core lowering order.  The
MLite path enters through ``build_dist_opt_stack``.  Both paths use the same tiny
BF16 model, the same pinned Megatron-Core process groups, and two data-parallel
ranks with deterministic but rank-distinct inputs.

This file is also importable on a CPU-only host: Megatron and CUDA are imported
only by the ``run`` subcommand.  ``compare`` and ``selftest`` operate on CPU
tensors only.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


PINNED_MEGATRON_REVISION = "d64ba4ccb1e3e878c15171c9cc58d5d3b46bf4d5"
PINNED_EMERGING_REVISION = "b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16"
WORLD_SIZE = 2
TOTAL_STEPS = 4
SAVE_STEPS = 2
MARKER_NAME = "NON_SKIP_MUON_DISTOPT_COMPACT_BITWISE_PASSED"
ADAM_MARKER_NAME = "NON_SKIP_PINNED_ADAM_DISTOPT_GATE_PASSED"
SCHEMA_VERSION = 1

_ADAM_TEXT_ONLY_EXPECTED = {
    "backend": "mlite",
    "model_name": "qwen3_5",
    "seed": 42,
    "seq_len": 8,
    "num_microbatches": 1,
    "metadata": {
        "deterministic": True,
        "same_data_across_dp": True,
        "use_thd": False,
    },
    "eval_logits": {
        "shape": [1, 8],
        "sha256_as_bf16": "94c2d3527acaa8db8ba29d6a86d060d8039b70b5149ed27eeaeaec746f218010",
    },
    "steps": [
        {
            "step": 0,
            "loss": {"value": 13.027458190917969},
            "grad_norm": {"value": 120.75512734973202},
            "grad_fingerprint": {"tensor_count": 19},
            "post_step_weights": {
                "tensor_count": 21,
                "sha256": "618edfd90e5a2e10e6d42a436e129b3b22ff9625a2b2c6025cd925828cf41eee",
            },
            "update_successful": True,
        },
        {
            "step": 1,
            "loss": {"value": 14.698704719543457},
            "grad_norm": {"value": 96.15656991334498},
            "grad_fingerprint": {"tensor_count": 19},
            "post_step_weights": {
                "tensor_count": 21,
                "sha256": "3bb5841e92690d4b6864f40472875622e7a02e47cedc3f39fc54ba1a1c1ee635",
            },
            "update_successful": True,
        },
    ],
}


class VocabParallelEmbedding(nn.Module):
    """A dependency-free synthetic module with the native embedding type name."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.weight.is_embedding_or_output_parameter = True

    def forward(self, token_weights: torch.Tensor) -> torch.Tensor:
        return token_weights @ self.weight


class VocabParallelOutput(nn.Module):
    """A synthetic output head whose matrix must remain on Adam DistOpt."""

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(vocab_size))
        self.weight.is_embedding_or_output_parameter = True
        self.bias.is_embedding_or_output_parameter = True

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.weight, self.bias)


class SyntheticMixedModel(nn.Module):
    """Mixed model: embedding/output/bias/norm use Adam; four matrices use Muon."""

    vocab_size = 16
    hidden_size = 8

    def __init__(self) -> None:
        super().__init__()
        self.embedding = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=True) for _ in range(4)]
        )
        self.norm = nn.LayerNorm(self.hidden_size)
        self.output = VocabParallelOutput(self.hidden_size, self.vocab_size)

    def forward(self, token_weights: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_weights)
        for layer in self.layers:
            hidden = F.relu(layer(hidden))
        return self.output(self.norm(hidden))


@dataclass
class Stack:
    model: SyntheticMixedModel
    wrapped_chunks: list[nn.Module]
    optimizer: Any
    pg_collection: Any
    lowering_order: list[str]


@dataclass
class OptimizerIntrospection:
    layer_wise: Any
    dist_opt: Any
    raw_muon: Any
    muon_main_to_name: dict[torch.Tensor, str]
    adam_main_to_name: dict[torch.Tensor, str]
    adam_name_to_model_param: dict[str, torch.Tensor]


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _clone_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    return copy.deepcopy(value)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _stable_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _stable_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _initialize_model_parameters(model: nn.Module) -> None:
    with torch.no_grad():
        for index, (_name, parameter) in enumerate(model.named_parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32)
            values = ((values + index * 7) % 29 - 14) / 64.0
            parameter.copy_(values.reshape(parameter.shape))


def _make_input(step: int, rank: int, device: torch.device) -> torch.Tensor:
    rows = 4
    values = torch.arange(rows * SyntheticMixedModel.vocab_size, dtype=torch.int64)
    values = (values * 3 + step * 11 + rank * 17) % 23 - 11
    return (values.reshape(rows, SyntheticMixedModel.vocab_size).to(torch.bfloat16) / 8).to(
        device
    )


def _model_config_namespace() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=SyntheticMixedModel.hidden_size,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_experts=None,
        moe_intermediate_size=None,
        add_bias_linear=True,
    )


def _core_transformer_config():
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=1,
        hidden_size=SyntheticMixedModel.hidden_size,
        num_attention_heads=1,
        num_query_groups=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=True,
    )


def _core_optimizer_config():
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    required = {
        "muon_momentum",
        "muon_split_qkv",
        "muon_nesterov",
        "muon_scale_mode",
        "muon_fp32_matmul_prec",
        "muon_coefficient_type",
        "muon_num_ns_steps",
        "muon_tp_mode",
        "muon_extra_scale_factor",
        "muon_scalar_optimizer",
        "use_layer_wise_distributed_optimizer",
        "use_layer_wise_param_layout",
    }
    available = set(inspect.signature(OptimizerConfig).parameters)
    missing = sorted(required - available)
    _require(
        not missing,
        "Pinned Megatron Muon optimizer contract is unavailable; missing fields: "
        + ", ".join(missing),
    )
    return OptimizerConfig(
        optimizer="muon",
        lr=1.0e-2,
        min_lr=0.0,
        weight_decay=1.0e-2,
        clip_grad=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=False,
        use_layer_wise_distributed_optimizer=True,
        use_layer_wise_param_layout=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        muon_momentum=0.95,
        muon_split_qkv=True,
        muon_nesterov=False,
        muon_scale_mode="spectral",
        muon_fp32_matmul_prec="medium",
        muon_coefficient_type="quintic",
        muon_num_ns_steps=5,
        muon_tp_mode="blockwise",
        muon_extra_scale_factor=1.0,
        muon_scalar_optimizer="adam",
    )


def _initialize_distributed() -> tuple[int, torch.device]:
    import torch.distributed as dist

    _require(torch.cuda.is_available(), "The run subcommand requires CUDA")
    _require("LOCAL_RANK" in os.environ, "run must be launched by torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    _require(
        dist.get_world_size() == WORLD_SIZE,
        f"This acceptance requires world=DP={WORLD_SIZE}, got {dist.get_world_size()}",
    )

    torch.manual_seed(20260710)
    torch.cuda.manual_seed_all(20260710)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    return rank, torch.device("cuda", local_rank)


def _initialize_mpu() -> None:
    """Initialize the residual ambient d64 state after MLite's explicit groups."""

    from megatron.core import parallel_state as mpu

    _require(not mpu.is_initialized(), "Megatron model-parallel state was already initialized")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        num_distributed_optimizer_instances=1,
        create_gloo_process_groups=True,
    )


def _pg_collection_from_parallel_state(ps: Any):
    """Build the exact explicit group collection used by the MLite lowering."""

    from megatron.core.process_groups_config import ProcessGroupCollection

    return ProcessGroupCollection(
        tp=ps.tp_group,
        cp=ps.cp_group,
        pp=ps.pp_group,
        ep=ps.ep_group,
        mp=ps.tp_group,
        dp=ps.dp_group,
        dp_cp=ps.dp_cp_group,
        expt_dp=ps.ep_dp_group,
        expt_tp=ps.etp_group,
        tp_ep=ps.tp_ep_group,
        tp_ep_pp=ps.tp_ep_group,
        intra_dist_opt=ps.intra_dist_opt_group,
        embd=None,
        pos_embd=None,
    )


def _new_model(device: torch.device) -> SyntheticMixedModel:
    model = SyntheticMixedModel()
    _initialize_model_parameters(model)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.requires_grad_(True)
    return model


def _tag_upstream_metadata(model: nn.Module) -> None:
    """Native metadata pass, independent of all MLite optimizer helpers."""

    for module in model.modules():
        if type(module).__name__ in {"VocabParallelEmbedding", "VocabParallelOutput"}:
            for parameter in module.parameters(recurse=True):
                parameter.is_embedding_or_output_parameter = True
    for parameter in model.parameters():
        if parameter.requires_grad and not hasattr(parameter, "allreduce"):
            parameter.allreduce = True


def _build_upstream_stack(device: torch.device, ps: SimpleNamespace) -> Stack:
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.optimizer import get_megatron_optimizer
    from megatron.core.optimizer.layer_wise_optimizer import (
        LayerWiseDistributedOptimizer,
        tag_params_for_buffer_routing,
    )
    from megatron.core.transformer.enums import ModelType

    model = _new_model(device)
    transformer_config = _core_transformer_config()
    transformer_config.finalize_model_grads_func = finalize_model_grads
    model.config = transformer_config
    model.model_type = ModelType.encoder_or_decoder
    pg_collection = _pg_collection_from_parallel_state(ps)
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        use_layer_wise_param_layout=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        grad_reduce_in_fp32=True,
    )

    lowering_order: list[str] = []
    _tag_upstream_metadata(model)
    lowering_order.append("metadata")
    tag_params_for_buffer_routing([model])
    lowering_order.append("tag_params_for_buffer_routing")
    full_param_layout = LayerWiseDistributedOptimizer.compute_full_param_layout(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        None,
        WORLD_SIZE,
        ddp_config,
        expert_data_parallel_world_size=WORLD_SIZE,
    )
    lowering_order.append("compute_layer_wise_layout")
    wrapped = DistributedDataParallel(
        transformer_config,
        ddp_config,
        model,
        disable_bucketing=False,
        pg_collection=pg_collection,
        full_param_layout=full_param_layout,
    )
    lowering_order.append("ddp")
    optimizer = get_megatron_optimizer(
        config=_core_optimizer_config(),
        model_chunks=[wrapped],
        use_gloo_process_groups=False,
        pg_collection=pg_collection,
    )
    lowering_order.append("get_megatron_optimizer")
    optimizer._dist_opt_pg_collection = pg_collection
    return Stack(model, [wrapped], optimizer, pg_collection, lowering_order)


def _build_mlite_stack(device: torch.device, ps: SimpleNamespace) -> Stack:
    from megatron.lite.primitive.optimizers.megatron_wrap import build_dist_opt_stack
    from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig

    model = _new_model(device)
    parallel = ParallelConfig(tp=1, etp=1, ep=1, pp=1, vpp=1, cp=1)
    optimizer_config = OptimizerConfig(
        optimizer="muon",
        lr=1.0e-2,
        min_lr=0.0,
        weight_decay=1.0e-2,
        clip_grad=0.0,
        muon_momentum=0.95,
        muon_split_qkv=True,
        muon_nesterov=False,
        muon_scale_mode="spectral",
        muon_fp32_matmul_prec="medium",
        muon_coefficient_type="quintic",
        muon_num_ns_steps=5,
        muon_tp_mode="blockwise",
        muon_extra_scale_factor=1.0,
        muon_scalar_optimizer="adam",
        use_layer_wise_param_layout=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
    )
    engine_config = SimpleNamespace(
        model_name="synthetic_mixed_muon_distopt",
        parallel=parallel,
        optimizer=optimizer_config,
        deterministic=True,
    )
    wrapped_chunks, optimizer = build_dist_opt_stack(
        [model],
        model_cfg=_model_config_namespace(),
        engine_cfg=engine_config,
        ps=ps,
        is_expert=lambda _name: False,
    )
    pg_collection = getattr(optimizer, "_dist_opt_pg_collection", None)
    _require(pg_collection is not None, "MLite production lowering did not retain pg_collection")
    return Stack(
        model,
        wrapped_chunks,
        optimizer,
        pg_collection,
        ["build_dist_opt_stack"],
    )


def _walk_optimizer_nodes(root: Any) -> list[Any]:
    nodes: list[Any] = []
    seen: set[int] = set()

    def visit(node: Any) -> None:
        identity = id(node)
        if identity in seen:
            return
        seen.add(identity)
        nodes.append(node)
        children = getattr(node, "chained_optimizers", None)
        if children is not None:
            for child in children:
                visit(child)

    visit(root)
    return nodes


def _one_node(nodes: Iterable[Any], class_name: str) -> Any:
    matches = [node for node in nodes if type(node).__name__ == class_name]
    _require(
        len(matches) == 1,
        f"Expected exactly one {class_name}, found {[type(node).__name__ for node in matches]}",
    )
    return matches[0]


def _inspect_optimizer(stack: Stack) -> OptimizerIntrospection:
    name_by_model_param = {
        parameter: name
        for name, parameter in stack.model.named_parameters()
        if parameter.requires_grad
    }
    outer_children = getattr(stack.optimizer, "chained_optimizers", None)
    _require(
        isinstance(outer_children, list)
        and [type(child).__name__ for child in outer_children]
        == ["LayerWiseDistributedOptimizer", "DistributedOptimizer"],
        "Expected chained facade [LayerWiseDistributedOptimizer, DistributedOptimizer]",
    )
    nodes = _walk_optimizer_nodes(stack.optimizer)
    layer_wise = _one_node(nodes, "LayerWiseDistributedOptimizer")
    dist_opt = _one_node(nodes, "DistributedOptimizer")
    layer_nodes = _walk_optimizer_nodes(layer_wise)
    muon_wrapper = _one_node(layer_nodes, "Float16OptimizerWithFloat16Params")
    raw_muon = getattr(muon_wrapper, "optimizer", None)
    _require(raw_muon is not None, "Muon Float16 wrapper has no raw optimizer")
    _require(callable(getattr(raw_muon, "orthogonalize", None)), "Raw Muon lacks orthogonalize")

    muon_main_to_name: dict[torch.Tensor, str] = {}
    float16_groups = getattr(muon_wrapper, "float16_groups", None)
    main_groups = getattr(muon_wrapper, "fp32_from_float16_groups", None)
    _require(float16_groups is not None and main_groups is not None, "Muon master groups missing")
    _require(len(float16_groups) == len(main_groups), "Muon model/master group count mismatch")
    for model_group, main_group in zip(float16_groups, main_groups):
        _require(len(model_group) == len(main_group), "Muon model/master group length mismatch")
        for model_param, main_param in zip(model_group, main_group):
            _require(model_param in name_by_model_param, "Muon model parameter has no logical name")
            muon_main_to_name[main_param] = name_by_model_param[model_param]
    _require(muon_main_to_name, "This rank owns no Muon whole-matrix parameters")

    raw_adam = getattr(dist_opt, "optimizer", None)
    _require(raw_adam is not None, "DistributedOptimizer has no raw Adam optimizer")
    group_index_map = getattr(dist_opt, "model_param_group_index_map", None)
    _require(isinstance(group_index_map, dict), "DistributedOptimizer group index map missing")
    adam_main_to_name: dict[torch.Tensor, str] = {}
    adam_name_to_model_param: dict[str, torch.Tensor] = {}
    for model_param, (group_index, group_order) in group_index_map.items():
        _require(model_param in name_by_model_param, "Adam model parameter has no logical name")
        main_param = raw_adam.param_groups[group_index]["params"][group_order]
        name = name_by_model_param[model_param]
        adam_main_to_name[main_param] = name
        adam_name_to_model_param[name] = model_param
    _require(adam_main_to_name, "This rank owns no Adam DistOpt shard")
    _require(
        set(muon_main_to_name.values()).isdisjoint(adam_name_to_model_param),
        "A logical parameter is owned by both Muon and Adam",
    )
    return OptimizerIntrospection(
        layer_wise,
        dist_opt,
        raw_muon,
        muon_main_to_name,
        adam_main_to_name,
        adam_name_to_model_param,
    )


def _route_metadata(stack: Stack, view: OptimizerIntrospection, rank: int) -> dict[str, Any]:
    names = {
        parameter: name
        for name, parameter in stack.model.named_parameters()
        if parameter.requires_grad
    }
    route: dict[str, dict[str, Any]] = {}
    for parameter, name in names.items():
        managed = getattr(parameter, "is_managed_by_layer_wise_optimizer", None)
        _require(isinstance(managed, bool), f"Missing native route tag for {name}")
        route[name] = {
            "optimizer": "muon" if managed else "adam",
            "shape": list(parameter.shape),
            "dtype": _dtype_name(parameter.dtype),
            "embedding_or_output": bool(
                getattr(parameter, "is_embedding_or_output_parameter", False)
            ),
            "is_qkv": bool(getattr(parameter, "is_qkv", False)),
            "is_expert": not bool(getattr(parameter, "allreduce", True)),
        }

    owner_lists = getattr(view.layer_wise, "dp_cp_params_list", None)
    _require(isinstance(owner_lists, list), "LayerWise whole-matrix owner lists missing")
    _require(len(owner_lists) == WORLD_SIZE, "LayerWise owner list does not match DP=2")
    owners: dict[str, int | None] = {name: None for name in route}
    seen_owned: set[str] = set()
    for owner, parameters in enumerate(owner_lists):
        for parameter in parameters:
            _require(parameter in names, "LayerWise owner list contains an unnamed parameter")
            name = names[parameter]
            _require(route[name]["optimizer"] == "muon", f"Adam parameter {name} has Muon owner")
            _require(name not in seen_owned, f"Muon parameter {name} has multiple owners")
            seen_owned.add(name)
            owners[name] = owner
    muon_names = {name for name, item in route.items() if item["optimizer"] == "muon"}
    expected_muon_names = {f"layers.{index}.weight" for index in range(4)}
    _require(muon_names == expected_muon_names, "Native routing did not select exactly four matrices")
    for name in ("embedding.weight", "output.weight", "output.bias"):
        _require(route[name]["optimizer"] == "adam", f"{name} did not route to Adam")
        _require(route[name]["embedding_or_output"], f"{name} lacks embedding/output metadata")
    for name in ("norm.weight", "norm.bias"):
        _require(route[name]["optimizer"] == "adam", f"{name} did not route to Adam")
    _require(
        all(item["optimizer"] == "adam" for name, item in route.items() if name.endswith(".bias")),
        "At least one bias did not route to Adam",
    )
    _require(seen_owned == muon_names, "Whole-matrix ownership does not cover every Muon parameter")
    _require(
        {owner for name, owner in owners.items() if name in muon_names} == {0, 1},
        "Synthetic Muon matrices did not exercise both DP owners",
    )
    _require(
        set(view.muon_main_to_name.values())
        == {name for name, owner in owners.items() if owner == rank},
        "Local Muon optimizer parameters do not match the upstream whole-matrix owner list",
    )

    wrapped = stack.wrapped_chunks[0]
    full_layout = getattr(wrapped, "full_param_layout", None)
    _require(full_layout is not None, "DDP did not retain the explicit full_param_layout")
    layout_offsets: dict[str, dict[str, Any]] = {}
    for buffer_key, layout in full_layout.layouts.items():
        for parameter, (start, end, bucket_id) in layout.param_index_map.items():
            _require(parameter in names, "Buffer layout contains an unnamed parameter")
            name = names[parameter]
            _require(name not in layout_offsets, f"Duplicate buffer layout entry for {name}")
            bucket_start, bucket_end = layout.bucket_indices[bucket_id]
            bucket_numel_unpadded = int(layout.per_bucket_numel_unpadded[bucket_id])
            dp_size = WORLD_SIZE
            _require((bucket_end - bucket_start) % dp_size == 0, f"Unshardable bucket for {name}")
            if buffer_key.is_managed_by_layer_wise_optimizer:
                _require(
                    bucket_end - bucket_start == bucket_numel_unpadded,
                    f"Muon buffer for {name} is padded instead of compact",
                )
            shard_size = (bucket_end - bucket_start) // dp_size
            local_start = bucket_start + rank * shard_size
            local_end = local_start + shard_size
            overlap_start = max(start, local_start)
            overlap_end = min(end, local_end)
            layout_offsets[name] = {
                "buffer_key": {
                    "param_dtype": _dtype_name(buffer_key.param_dtype),
                    "grad_dtype": _dtype_name(buffer_key.grad_dtype),
                    "is_expert_parallel": bool(buffer_key.is_expert_parallel),
                    "is_managed_by_layer_wise_optimizer": bool(
                        buffer_key.is_managed_by_layer_wise_optimizer
                    ),
                },
                "param_start": int(start),
                "param_end": int(end),
                "bucket_id": int(bucket_id),
                "bucket_start": int(bucket_start),
                "bucket_end": int(bucket_end),
                "bucket_numel_unpadded": bucket_numel_unpadded,
                # Compact Muon ownership is recorded separately above.  These are
                # only the DDP layout's equal DP partitions, not Muon owner shards.
                "layout_dp_partition_start": int(local_start),
                "layout_dp_partition_end": int(local_end),
                "layout_dp_overlap_start": int(overlap_start),
                "layout_dp_overlap_end": int(max(overlap_start, overlap_end)),
            }
    _require(set(layout_offsets) == set(route), "Buffer layout does not cover all parameters")

    adam_signature = []
    for name, model_parameter in view.adam_name_to_model_param.items():
        group_index, group_order = view.dist_opt.model_param_group_index_map[model_parameter]
        gbuf_index, dtype_key, bucket_index = view.dist_opt.model_param_gbuf_map[model_parameter]
        range_map = view.dist_opt._get_model_param_range_map(model_parameter)
        adam_signature.append(
            {
                "name": name,
                "group_index": int(group_index),
                "group_order": int(group_order),
                "gbuf_index": int(gbuf_index),
                "bucket_index": int(bucket_index),
                "buffer_dtype": [_dtype_name(dtype) for dtype in dtype_key],
                "ranges": {
                    key: {"start": int(value.start), "end": int(value.end)}
                    for key, value in sorted(range_map.items())
                },
            }
        )
    return {
        "route": route,
        "whole_matrix_owner": owners,
        "buffer_layout_offsets": layout_offsets,
        "optimizer_signature": {
            "chain": [type(child).__name__ for child in stack.optimizer.chained_optimizers],
            "local_muon": [
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": _dtype_name(parameter.dtype),
                }
                for parameter, name in view.muon_main_to_name.items()
            ],
            "local_adam": adam_signature,
        },
    }


def _snapshot_reduced_grads(
    model: nn.Module, view: OptimizerIntrospection
) -> dict[str, dict[str, torch.Tensor]]:
    """Capture full Muon all-reduce results and local Adam reduce-scatter shards."""

    muon_full: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grad = getattr(parameter, "main_grad", None)
        _require(isinstance(grad, torch.Tensor), f"Reduced main_grad missing for {name}")
        _require(grad.dtype == torch.float32, f"Reduced main_grad for {name} is not fp32")
        if getattr(parameter, "is_managed_by_layer_wise_optimizer", False):
            muon_full[name] = grad.detach().cpu().clone()

    adam_shard: dict[str, torch.Tensor] = {}
    for name, model_parameter in view.adam_name_to_model_param.items():
        grad = getattr(model_parameter, "main_grad", None)
        _require(isinstance(grad, torch.Tensor), f"Adam main_grad missing for {name}")
        param_range = view.dist_opt._get_model_param_range_map(model_parameter)["param"]
        shard = grad.view(-1)[param_range.start : param_range.end]
        _require(shard.numel() > 0, f"Adam reduced shard is empty for {name}")
        adam_shard[name] = shard.detach().cpu().clone()
    return {
        "muon_all_reduced_full": dict(sorted(muon_full.items())),
        "adam_reduce_scattered_local_shard": dict(sorted(adam_shard.items())),
    }


def _snapshot_local_optimizer_grads(view: OptimizerIntrospection) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for main_param, name in {
        **view.muon_main_to_name,
        **view.adam_main_to_name,
    }.items():
        _require(main_param.grad is not None, f"Prepared local optimizer grad missing for {name}")
        result[name] = main_param.grad.detach().cpu().clone()
    return dict(sorted(result.items()))


def _snapshot_muon_state(view: OptimizerIntrospection) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for main_param, name in view.muon_main_to_name.items():
        raw_state = view.raw_muon.state.get(main_param)
        _require(isinstance(raw_state, dict), f"Muon state missing for {name}")
        _require("momentum_buffer" in raw_state, f"Muon momentum_buffer missing for {name}")
        momentum = raw_state["momentum_buffer"]
        _require(isinstance(momentum, torch.Tensor), f"Muon momentum for {name} is not a tensor")
        _require(momentum.shape == main_param.shape, f"Muon momentum shape mismatch for {name}")
        _require(momentum.dtype == torch.float32, f"Muon momentum for {name} is not fp32")
        _require(main_param.dtype == torch.float32, f"Muon master for {name} is not fp32")
        item = _clone_cpu(raw_state)
        item["fp32_master"] = main_param.detach().cpu().clone()
        state[name] = item
    return dict(sorted(state.items()))


def _adam_step_tensor(dist_opt: Any, main_param: torch.Tensor) -> torch.Tensor:
    raw = dist_opt.optimizer
    state = raw.state.get(main_param, {})
    if "step" in state:
        step = state["step"]
    else:
        group = next(
            (
                group
                for group in raw.param_groups
                if any(candidate is main_param for candidate in group["params"])
            ),
            None,
        )
        _require(group is not None, "Adam main parameter is absent from all parameter groups")
        _require("step" in group, "Adam step is absent from both state and parameter group")
        step = group["step"]
    if isinstance(step, torch.Tensor):
        return step.detach().cpu().clone()
    return torch.tensor(step, dtype=torch.int64)


def _snapshot_adam_state(view: OptimizerIntrospection) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    getter = getattr(view.dist_opt, "_get_main_param_and_optimizer_states", None)
    _require(callable(getter), "DistributedOptimizer state introspection API is unavailable")
    for name, model_param in view.adam_name_to_model_param.items():
        tensors = getter(model_param)
        required = {"param", "exp_avg", "exp_avg_sq"}
        _require(required <= set(tensors), f"Adam state for {name} lacks {sorted(required - set(tensors))}")
        item = {
            "fp32_master": tensors["param"].detach().cpu().clone(),
            "exp_avg": tensors["exp_avg"].detach().cpu().clone(),
            "exp_avg_sq": tensors["exp_avg_sq"].detach().cpu().clone(),
            "step": _adam_step_tensor(view.dist_opt, tensors["param"]),
        }
        _require(item["fp32_master"].dtype == torch.float32, f"Adam master for {name} is not fp32")
        _require(item["exp_avg"].dtype == torch.float32, f"Adam m for {name} is not fp32")
        _require(item["exp_avg_sq"].dtype == torch.float32, f"Adam v for {name} is not fp32")
        _require(
            item["fp32_master"].shape == item["exp_avg"].shape == item["exp_avg_sq"].shape,
            f"Adam state shape mismatch for {name}",
        )
        result[name] = item
    return dict(sorted(result.items()))


def _restore_adam_state(
    view: OptimizerIntrospection, saved: Mapping[str, Mapping[str, torch.Tensor]]
) -> None:
    current_names = set(view.adam_name_to_model_param)
    _require(current_names == set(saved), "Checkpoint Adam shard names differ from current layout")
    getter = view.dist_opt._get_main_param_and_optimizer_states
    for name, model_param in view.adam_name_to_model_param.items():
        current = getter(model_param)
        saved_item = saved[name]
        for current_key, saved_key in (
            ("param", "fp32_master"),
            ("exp_avg", "exp_avg"),
            ("exp_avg_sq", "exp_avg_sq"),
        ):
            _require(current_key in current, f"Current Adam state for {name} lacks {current_key}")
            current[current_key].copy_(saved_item[saved_key].to(current[current_key].device))

        main_param = current["param"]
        raw_state = view.dist_opt.optimizer.state[main_param]
        saved_step = saved_item["step"]
        if "step" in raw_state:
            if isinstance(raw_state["step"], torch.Tensor):
                raw_state["step"].copy_(saved_step.to(raw_state["step"].device))
            else:
                raw_state["step"] = int(saved_step.item())
        else:
            group = next(
                group
                for group in view.dist_opt.optimizer.param_groups
                if any(candidate is main_param for candidate in group["params"])
            )
            _require("step" in group, f"Current Adam group step missing for {name}")
            if isinstance(group["step"], torch.Tensor):
                group["step"].copy_(saved_step.to(group["step"].device))
            else:
                group["step"] = int(saved_step.item())


def _install_muon_capture(
    view: OptimizerIntrospection,
) -> tuple[dict[str, dict[str, torch.Tensor]], Callable[[], None]]:
    capture: dict[str, dict[str, torch.Tensor]] = {
        "pre_ns_momentum": {},
        "post_ns_update": {},
    }
    original = view.raw_muon.orthogonalize

    def wrapped(raw_self, parameter, grad, **kwargs):
        _require(parameter in view.muon_main_to_name, "Muon orthogonalize saw an unnamed master")
        name = view.muon_main_to_name[parameter]
        _require(name not in capture["pre_ns_momentum"], f"Muon {name} orthogonalized twice")
        capture["pre_ns_momentum"][name] = grad.detach().cpu().clone()
        update = original(parameter, grad, **kwargs)
        _require(isinstance(update, torch.Tensor), f"Muon orthogonalize returned non-tensor for {name}")
        _require(grad.shape == parameter.shape, f"Pre-NS shape mismatch for {name}")
        _require(update.shape == parameter.shape, f"Post-NS shape mismatch for {name}")
        _require(grad.dtype == torch.float32, f"Pre-NS momentum for {name} is not fp32")
        _require(update.dtype == torch.float32, f"Post-NS update for {name} is not fp32")
        capture["post_ns_update"][name] = update.detach().cpu().clone()
        return update

    view.raw_muon.orthogonalize = MethodType(wrapped, view.raw_muon)

    def reset() -> None:
        capture["pre_ns_momentum"].clear()
        capture["post_ns_update"].clear()

    return capture, reset


def _runtime_identity() -> dict[str, str]:
    return {
        "run_id": os.environ.get("RUN_ID", "CPU_SELFTEST"),
        "mlite_head": os.environ.get("MLITE_HEAD", "CPU_SELFTEST"),
        "mlite_tree": os.environ.get("MLITE_TREE", "CPU_SELFTEST"),
        "megatron_tree": os.environ.get("MEGATRON_TREE", "CPU_SELFTEST"),
        "emerging_tree": os.environ.get("EMERGING_TREE", "CPU_SELFTEST"),
    }


def _require_runtime_identity() -> None:
    identity = _runtime_identity()
    _require(
        re.fullmatch(r"[A-Za-z0-9_.-]+", identity["run_id"]) is not None,
        "RUN_ID is required and may contain only letters, digits, dot, underscore, and dash",
    )
    for name in ("mlite_head", "mlite_tree", "megatron_tree", "emerging_tree"):
        _require(
            re.fullmatch(r"[0-9a-f]{40}", identity[name]) is not None,
            f"{name.upper()} must be a full 40-character Git object id",
        )


def _contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pinned_megatron_revision": PINNED_MEGATRON_REVISION,
        "pinned_emerging_revision": PINNED_EMERGING_REVISION,
        "world_size": WORLD_SIZE,
        "data_parallel_size": WORLD_SIZE,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_parallel_size": 1,
        "parameter_dtype": "bfloat16",
        "compact_layer_wise_layout": True,
        "total_steps": TOTAL_STEPS,
        "save_steps": SAVE_STEPS,
        "runtime_identity": _runtime_identity(),
        "optimizer": {
            "name": "muon",
            "lr": 1.0e-2,
            "weight_decay": 1.0e-2,
            "clip_grad": 0.0,
            "muon_momentum": 0.95,
            "muon_nesterov": False,
            "muon_scale_mode": "spectral",
            "muon_fp32_matmul_prec": "medium",
            "muon_coefficient_type": "quintic",
            "muon_num_ns_steps": 5,
            "muon_tp_mode": "blockwise",
            "muon_extra_scale_factor": 1.0,
            "muon_scalar_optimizer": "adam",
        },
    }


def _artifact_path(output_dir: Path, trajectory: str, rank: int) -> Path:
    return output_dir / trajectory / f"rank_{rank:05d}.pt"


def _checkpoint_path(checkpoint_dir: Path, rank: int) -> Path:
    return checkpoint_dir / f"rank_{rank:05d}.pt"


def _save_checkpoint(
    path: Path,
    stack: Stack,
    view: OptimizerIntrospection,
    metadata: Mapping[str, Any],
) -> None:
    checkpoint = {
        "contract": _contract(),
        "completed_steps": SAVE_STEPS,
        "metadata": _clone_cpu(metadata),
        "model_state": _stable_model_state(stack.model),
        "optimizer_state": _clone_cpu(stack.optimizer.state_dict()),
        "parameter": _stable_parameters(stack.model),
        "muon_state": _snapshot_muon_state(view),
        "adam_state": _snapshot_adam_state(view),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _assert_tensor_tree_equal(lhs: Any, rhs: Any, label: str) -> None:
    mismatches: list[dict[str, Any]] = []
    counters = {"tensor_checks": 0, "torch_equal_checks": 0, "assert_close_checks": 0}
    _compare_node(lhs, rhs, label, mismatches, counters)
    _require(not mismatches, f"{label} mismatch after checkpoint load: {mismatches[:3]}")


def _load_checkpoint(
    path: Path,
    stack: Stack,
    view: OptimizerIntrospection,
    metadata: Mapping[str, Any],
) -> None:
    _require(path.is_file(), f"Resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _require(checkpoint.get("contract") == _contract(), "Checkpoint contract mismatch")
    _require(checkpoint.get("completed_steps") == SAVE_STEPS, "Checkpoint is not a step-2 save")
    _assert_tensor_tree_equal(checkpoint.get("metadata"), metadata, "checkpoint.metadata")
    stack.model.load_state_dict(checkpoint["model_state"], strict=True)
    stack.optimizer.load_state_dict(checkpoint["optimizer_state"])
    _restore_adam_state(view, checkpoint["adam_state"])
    _assert_tensor_tree_equal(_stable_parameters(stack.model), checkpoint["parameter"], "parameter")
    _assert_tensor_tree_equal(_snapshot_muon_state(view), checkpoint["muon_state"], "muon_state")
    _assert_tensor_tree_equal(_snapshot_adam_state(view), checkpoint["adam_state"], "adam_state")


def _run_steps(
    stack: Stack,
    view: OptimizerIntrospection,
    rank: int,
    device: torch.device,
    start: int,
    end: int,
) -> dict[int, dict[str, Any]]:
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads

    capture, reset_capture = _install_muon_capture(view)
    expected_local_muon = set(view.muon_main_to_name.values())
    steps: dict[int, dict[str, Any]] = {}
    wrapped = stack.wrapped_chunks[0]
    for step in range(start, end):
        stack.optimizer.zero_grad(set_to_none=True)
        wrapped.zero_grad_buffer()
        reset_capture()
        input_tensor = _make_input(step, rank, device)
        output = wrapped(input_tensor)
        loss = output.float().square().mean()
        loss.backward()
        finalize_model_grads(stack.wrapped_chunks, pg_collection=stack.pg_collection)
        reduced_grad = _snapshot_reduced_grads(stack.model, view)

        found_inf = stack.optimizer.prepare_grads()
        _require(not found_inf, f"Optimizer found inf/nan at step {step}")
        local_optimizer_grad = _snapshot_local_optimizer_grads(view)
        update_successful = stack.optimizer.step_with_ready_grads()
        _require(bool(update_successful), f"Optimizer update failed at step {step}")
        torch.cuda.synchronize(device)

        _require(
            set(capture["pre_ns_momentum"]) == expected_local_muon,
            f"Incomplete pre-NS capture at step {step}",
        )
        _require(
            set(capture["post_ns_update"]) == expected_local_muon,
            f"Incomplete post-NS capture at step {step}",
        )
        _require(
            any(
                torch.count_nonzero(value).item()
                for category in reduced_grad.values()
                for value in category.values()
            ),
            f"All reduced gradients are zero at step {step}",
        )
        _require(
            any(torch.count_nonzero(value).item() for value in capture["pre_ns_momentum"].values()),
            f"All pre-NS momentum tensors are zero at step {step}",
        )
        _require(
            any(torch.count_nonzero(value).item() for value in capture["post_ns_update"].values()),
            f"All post-NS updates are zero at step {step}",
        )
        muon_state = _snapshot_muon_state(view)
        adam_state = _snapshot_adam_state(view)
        for name, pre_ns in capture["pre_ns_momentum"].items():
            _require(
                torch.equal(pre_ns, muon_state[name]["momentum_buffer"]),
                f"Pre-NS tensor and stored Muon momentum differ for {name} at step {step}",
            )
        adam_steps = {int(item["step"].item()) for item in adam_state.values()}
        _require(adam_steps == {step + 1}, f"Adam step mismatch at training step {step}: {adam_steps}")
        steps[step] = {
            "input": input_tensor.detach().cpu().clone(),
            "loss": loss.detach().cpu().clone(),
            "reduced_grad": reduced_grad,
            "local_optimizer_grad": local_optimizer_grad,
            "pre_ns_momentum": dict(sorted(_clone_cpu(capture["pre_ns_momentum"]).items())),
            "post_ns_update": dict(sorted(_clone_cpu(capture["post_ns_update"]).items())),
            "parameter": _stable_parameters(stack.model),
            "muon_state": muon_state,
            "adam_state": adam_state,
        }
    return steps


def run_one(args: argparse.Namespace) -> int:
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu
    from megatron.lite.primitive.parallel.state import init_parallel
    from megatron.lite.runtime.contracts.config import ParallelConfig

    _require_runtime_identity()
    rank, device = _initialize_distributed()
    try:
        ps = init_parallel(ParallelConfig(tp=1, etp=1, ep=1, pp=1, vpp=1, cp=1))
        _require(ps.intra_dist_opt_group is not None, "MLite intra_dist_opt group is missing")
        _require(
            dist.get_world_size(ps.intra_dist_opt_group) == WORLD_SIZE,
            "MLite intra_dist_opt group does not span the optimizer instance",
        )
        if args.implementation == "upstream":
            _initialize_mpu()
            stack = _build_upstream_stack(device, ps)
        else:
            stack = _build_mlite_stack(device, ps)
            _require(mpu.is_initialized(), "MLite lowering did not initialize ambient MCore state")
        view = _inspect_optimizer(stack)
        metadata = _route_metadata(stack, view, rank)

        output_dir = Path(args.output_dir).resolve()
        checkpoint_dir = Path(args.checkpoint_dir).resolve()
        if args.trajectory == "continuous":
            start, end = 0, TOTAL_STEPS
        elif args.trajectory == "save":
            start, end = 0, SAVE_STEPS
        else:
            _load_checkpoint(
                _checkpoint_path(checkpoint_dir, rank), stack, view, metadata
            )
            start, end = SAVE_STEPS, TOTAL_STEPS

        steps = _run_steps(stack, view, rank, device, start, end)
        artifact = {
            "kind": "compact_muon_distopt_bitwise_rank_artifact",
            "implementation": args.implementation,
            "trajectory": args.trajectory,
            "rank": rank,
            "contract": _contract(),
            "metadata": metadata,
            "lowering_order": list(stack.lowering_order),
            "steps": steps,
        }
        output_path = _artifact_path(output_dir, args.trajectory, rank)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_clone_cpu(artifact), output_path)

        if args.trajectory == "save":
            _save_checkpoint(
                _checkpoint_path(checkpoint_dir, rank), stack, view, metadata
            )
        dist.barrier()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "implementation": args.implementation,
                        "trajectory": args.trajectory,
                        "world_size": WORLD_SIZE,
                        "steps": list(range(start, end)),
                        "output_dir": str(output_dir),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return 0
    finally:
        if mpu.is_initialized():
            mpu.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


def _tensor_mismatch_detail(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "lhs_shape": list(lhs.shape),
        "rhs_shape": list(rhs.shape),
        "lhs_dtype": str(lhs.dtype),
        "rhs_dtype": str(rhs.dtype),
    }
    if lhs.shape == rhs.shape and lhs.numel() and lhs.dtype != torch.bool and rhs.dtype != torch.bool:
        try:
            detail["max_abs_diff"] = float(
                (lhs.to(torch.float64) - rhs.to(torch.float64)).abs().max().item()
            )
        except (RuntimeError, TypeError, ValueError):
            pass
    return detail


def _compare_node(
    lhs: Any,
    rhs: Any,
    path: str,
    mismatches: list[dict[str, Any]],
    counters: dict[str, int],
) -> None:
    if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        counters["tensor_checks"] += 1
        equal = torch.equal(lhs, rhs)
        counters["torch_equal_checks"] += 1
        assert_close = True
        assert_close_error = ""
        try:
            torch.testing.assert_close(
                lhs,
                rhs,
                atol=0,
                rtol=0,
                check_dtype=True,
                check_device=True,
                equal_nan=False,
            )
        except AssertionError as error:
            assert_close = False
            assert_close_error = str(error).splitlines()[0]
        counters["assert_close_checks"] += 1
        if not (equal and assert_close):
            mismatches.append(
                {
                    "path": path,
                    "kind": "tensor",
                    "torch_equal": equal,
                    "assert_close_atol0_rtol0": assert_close,
                    "assert_close_error": assert_close_error,
                    **_tensor_mismatch_detail(lhs, rhs),
                }
            )
        return

    if isinstance(lhs, torch.Tensor) or isinstance(rhs, torch.Tensor):
        mismatches.append(
            {
                "path": path,
                "kind": "type",
                "lhs_type": type(lhs).__name__,
                "rhs_type": type(rhs).__name__,
            }
        )
        return
    if isinstance(lhs, Mapping) and isinstance(rhs, Mapping):
        lhs_keys, rhs_keys = set(lhs), set(rhs)
        if lhs_keys != rhs_keys:
            mismatches.append(
                {
                    "path": path,
                    "kind": "mapping_keys",
                    "lhs_only": sorted(map(str, lhs_keys - rhs_keys)),
                    "rhs_only": sorted(map(str, rhs_keys - lhs_keys)),
                }
            )
        for key in sorted(lhs_keys & rhs_keys, key=str):
            _compare_node(lhs[key], rhs[key], f"{path}.{key}", mismatches, counters)
        return
    if isinstance(lhs, (list, tuple)) and isinstance(rhs, (list, tuple)):
        if type(lhs) is not type(rhs) or len(lhs) != len(rhs):
            mismatches.append(
                {
                    "path": path,
                    "kind": "sequence",
                    "lhs_type": type(lhs).__name__,
                    "rhs_type": type(rhs).__name__,
                    "lhs_len": len(lhs),
                    "rhs_len": len(rhs),
                }
            )
        for index, (lhs_item, rhs_item) in enumerate(zip(lhs, rhs)):
            _compare_node(lhs_item, rhs_item, f"{path}[{index}]", mismatches, counters)
        return
    if type(lhs) is not type(rhs) or lhs != rhs:
        mismatches.append(
            {
                "path": path,
                "kind": "value",
                "lhs": repr(lhs),
                "rhs": repr(rhs),
            }
        )


def _load_artifact(root: Path, trajectory: str, rank: int, implementation: str) -> dict[str, Any]:
    path = _artifact_path(root, trajectory, rank)
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("implementation") != implementation:
        raise ValueError(
            f"{path}: expected implementation={implementation}, got {artifact.get('implementation')}"
        )
    if artifact.get("trajectory") != trajectory or artifact.get("rank") != rank:
        raise ValueError(f"{path}: trajectory/rank metadata mismatch")
    if artifact.get("contract") != _contract():
        raise ValueError(f"{path}: pinned contract mismatch")
    expected_lowering = {
        "upstream": [
            "metadata",
            "tag_params_for_buffer_routing",
            "compute_layer_wise_layout",
            "ddp",
            "get_megatron_optimizer",
        ],
        "mlite": ["build_dist_opt_stack"],
    }[implementation]
    if artifact.get("lowering_order") != expected_lowering:
        raise ValueError(f"{path}: lowering order mismatch")
    expected_steps = {
        "continuous": set(range(TOTAL_STEPS)),
        "save": set(range(SAVE_STEPS)),
        "resume": set(range(SAVE_STEPS, TOTAL_STEPS)),
    }[trajectory]
    if set(artifact.get("steps", {})) != expected_steps:
        raise ValueError(f"{path}: expected steps {sorted(expected_steps)}")
    return artifact


def _normalized(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract": artifact["contract"],
        "metadata": artifact["metadata"],
        "steps": artifact["steps"],
    }


def _compare_directories(upstream_dir: Path, mlite_dir: Path) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    counters = {"tensor_checks": 0, "torch_equal_checks": 0, "assert_close_checks": 0}
    checks: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, dict[int, dict[str, Any]]]] = {
        "upstream": {},
        "mlite": {},
    }

    try:
        for implementation, root in (("upstream", upstream_dir), ("mlite", mlite_dir)):
            for trajectory in ("continuous", "save", "resume"):
                artifacts[implementation][trajectory] = {
                    rank: _load_artifact(root, trajectory, rank, implementation)
                    for rank in range(WORLD_SIZE)
                }
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
        return {
            "passed": False,
            **counters,
            "checks": [],
            "mismatches": [{"path": "artifact_load", "kind": "load", "error": str(error)}],
        }

    def compare(check_name: str, lhs: Any, rhs: Any) -> None:
        before = len(mismatches)
        _compare_node(lhs, rhs, check_name, mismatches, counters)
        checks.append({"name": check_name, "passed": len(mismatches) == before})

    for implementation in ("upstream", "mlite"):
        for rank in range(WORLD_SIZE):
            continuous = artifacts[implementation]["continuous"][rank]
            saved = artifacts[implementation]["save"][rank]
            resumed = artifacts[implementation]["resume"][rank]
            reconstructed = {
                "contract": saved["contract"],
                "metadata": saved["metadata"],
                "steps": {**saved["steps"], **resumed["steps"]},
            }
            compare(
                f"{implementation}.rank{rank}.resume_vs_continuous",
                _normalized(continuous),
                reconstructed,
            )
            compare(
                f"{implementation}.rank{rank}.resume_metadata",
                saved["metadata"],
                resumed["metadata"],
            )

    for trajectory in ("continuous", "save", "resume"):
        for rank in range(WORLD_SIZE):
            compare(
                f"upstream_vs_mlite.{trajectory}.rank{rank}",
                _normalized(artifacts["upstream"][trajectory][rank]),
                _normalized(artifacts["mlite"][trajectory][rank]),
            )

    for implementation in ("upstream", "mlite"):
        for trajectory in ("continuous", "save", "resume"):
            rank_zero = artifacts[implementation][trajectory][0]
            rank_one = artifacts[implementation][trajectory][1]
            before = len(mismatches)
            for step in sorted(rank_zero["steps"]):
                input_zero = rank_zero["steps"][step]["input"]
                input_one = rank_one["steps"][step]["input"]
                if torch.equal(input_zero, input_one):
                    mismatches.append(
                        {
                            "path": f"{implementation}.{trajectory}.step{step}.rank_distinct_input",
                            "kind": "invariant",
                            "error": "rank inputs are bitwise equal",
                        }
                    )
                compare(
                    f"{implementation}.{trajectory}.step{step}.rank_parameter_sync",
                    rank_zero["steps"][step]["parameter"],
                    rank_one["steps"][step]["parameter"],
                )
            checks.append(
                {
                    "name": f"{implementation}.{trajectory}.rank_distinct_input",
                    "passed": len(mismatches) == before,
                }
            )

    return {
        "passed": not mismatches and all(check["passed"] for check in checks),
        **counters,
        "checks": checks,
        "mismatches": mismatches,
    }


def _write_comparison(
    report: dict[str, Any], output_json: Path, *, emit_stdout_marker: bool = True
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    marker = output_json.parent / MARKER_NAME
    marker.unlink(missing_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["passed"]:
        marker.write_text(
            f"{MARKER_NAME} world={WORLD_SIZE} steps={TOTAL_STEPS} tensor_checks={report['tensor_checks']}\n",
            encoding="utf-8",
        )
        if emit_stdout_marker:
            print(marker.read_text(encoding="utf-8").strip(), flush=True)


def compare_runs(args: argparse.Namespace) -> int:
    _require_runtime_identity()
    report = _compare_directories(
        Path(args.upstream_dir).resolve(), Path(args.mlite_dir).resolve()
    )
    report = {
        "kind": "compact_muon_distopt_bitwise_comparison",
        "contract": _contract(),
        "upstream_dir": str(Path(args.upstream_dir).resolve()),
        "mlite_dir": str(Path(args.mlite_dir).resolve()),
        **report,
    }
    _write_comparison(report, Path(args.output_json).resolve())
    if not report["passed"]:
        print(json.dumps(report, sort_keys=True), flush=True)
        return 1
    return 0


def _expected_subset_mismatches(
    actual: Any, expected: Any, path: str = "artifact"
) -> list[str]:
    mismatches: list[str] = []
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return [f"{path}: expected mapping, got {type(actual).__name__}"]
        for key, expected_value in expected.items():
            if key not in actual:
                mismatches.append(f"{path}.{key}: missing")
                continue
            mismatches.extend(
                _expected_subset_mismatches(actual[key], expected_value, f"{path}.{key}")
            )
        return mismatches
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path}: expected list, got {type(actual).__name__}"]
        if len(actual) != len(expected):
            return [f"{path}: expected length {len(expected)}, got {len(actual)}"]
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            mismatches.extend(
                _expected_subset_mismatches(
                    actual_value, expected_value, f"{path}[{index}]"
                )
            )
        return mismatches
    if type(actual) is not type(expected) or actual != expected:
        mismatches.append(f"{path}: expected {expected!r}, got {actual!r}")
    return mismatches


def validate_adam_text_only(input_json: Path, output_json: Path) -> int:
    """Validate the frozen compact text-only Qwen3.5 Adam DistOpt regression."""

    artifact = json.loads(input_json.read_text(encoding="utf-8"))
    mismatches = _expected_subset_mismatches(artifact, _ADAM_TEXT_ONLY_EXPECTED)
    verdict = {
        "kind": "pinned_adam_distopt_text_only_verdict",
        "model_contract": "compact_text_only_qwen35",
        "model_contract_evidence": "slurm-13635323",
        "numeric_contract_evidence": "slurm-13694402-mlite-subrun",
        "passed": not mismatches,
        "non_skip": not mismatches,
        "mismatches": mismatches,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    marker = output_json.parent / ADAM_MARKER_NAME
    marker.unlink(missing_ok=True)
    output_json.write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if mismatches:
        print(json.dumps(verdict, sort_keys=True), flush=True)
        return 1
    marker.write_text(
        f"{ADAM_MARKER_NAME} model_contract=compact_text_only_qwen35 steps=2\n",
        encoding="utf-8",
    )
    print(marker.read_text(encoding="utf-8").strip(), flush=True)
    return 0


def validate_adam_text_only_command(args: argparse.Namespace) -> int:
    return validate_adam_text_only(
        Path(args.input_json).resolve(), Path(args.output_json).resolve()
    )


def _fake_artifact(implementation: str, trajectory: str, rank: int) -> dict[str, Any]:
    ranges = {
        "continuous": range(TOTAL_STEPS),
        "save": range(SAVE_STEPS),
        "resume": range(SAVE_STEPS, TOTAL_STEPS),
    }
    route = {
        "embedding.weight": {"optimizer": "adam"},
        "layers.0.weight": {"optimizer": "muon"},
    }
    steps = {}
    for step in ranges[trajectory]:
        parameter = torch.tensor([step + 1.0], dtype=torch.bfloat16)
        steps[step] = {
            "input": torch.tensor([rank, step], dtype=torch.int64),
            "loss": torch.tensor(float(step), dtype=torch.float32),
            "reduced_grad": {"layers.0.weight": torch.tensor([step], dtype=torch.float32)},
            "local_optimizer_grad": {
                "layers.0.weight": torch.tensor([step], dtype=torch.float32)
            },
            "pre_ns_momentum": {
                "layers.0.weight": torch.tensor([step + 0.25], dtype=torch.float32)
            },
            "post_ns_update": {
                "layers.0.weight": torch.tensor([step + 0.5], dtype=torch.float32)
            },
            "parameter": {"shared": parameter},
            "muon_state": {
                "layers.0.weight": {
                    "momentum_buffer": torch.tensor([step + 0.25], dtype=torch.float32)
                }
            },
            "adam_state": {
                "embedding.weight": {
                    "fp32_master": torch.tensor([step + 1.0], dtype=torch.float32),
                    "exp_avg": torch.tensor([step + 0.1], dtype=torch.float32),
                    "exp_avg_sq": torch.tensor([step + 0.2], dtype=torch.float32),
                    "step": torch.tensor(step + 1, dtype=torch.int64),
                }
            },
        }
    return {
        "kind": "compact_muon_distopt_bitwise_rank_artifact",
        "implementation": implementation,
        "trajectory": trajectory,
        "rank": rank,
        "contract": _contract(),
        "metadata": {
            "route": route,
            "whole_matrix_owner": {"embedding.weight": None, "layers.0.weight": 0},
            "buffer_layout_offsets": {
                "embedding.weight": {"param_start": 0, "param_end": 1},
                "layers.0.weight": {"param_start": 0, "param_end": 1},
            },
        },
        "lowering_order": {
            "upstream": [
                "metadata",
                "tag_params_for_buffer_routing",
                "compute_layer_wise_layout",
                "ddp",
                "get_megatron_optimizer",
            ],
            "mlite": ["build_dist_opt_stack"],
        }[implementation],
        "steps": steps,
    }


def comparator_selftest(_args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="muon_distopt_compare_") as directory:
        root = Path(directory)
        upstream = root / "upstream"
        mlite = root / "mlite"
        for implementation, output in (("upstream", upstream), ("mlite", mlite)):
            for trajectory in ("continuous", "save", "resume"):
                for rank in range(WORLD_SIZE):
                    path = _artifact_path(output, trajectory, rank)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(_fake_artifact(implementation, trajectory, rank), path)

        passing = _compare_directories(upstream, mlite)
        _require(passing["passed"], f"Comparator rejected equal fixtures: {passing['mismatches']}")
        _require(
            passing["tensor_checks"]
            == passing["torch_equal_checks"]
            == passing["assert_close_checks"],
            "Comparator did not apply both exact checks to every tensor",
        )
        pass_json = root / "pass" / "comparison.json"
        _write_comparison(passing, pass_json, emit_stdout_marker=False)
        _require((pass_json.parent / MARKER_NAME).is_file(), "Passing compare did not write marker")

        corrupt_path = _artifact_path(mlite, "continuous", 0)
        corrupt = torch.load(corrupt_path, map_location="cpu", weights_only=True)
        corrupt["steps"][3]["adam_state"]["embedding.weight"]["exp_avg"].add_(1)
        torch.save(corrupt, corrupt_path)
        failing = _compare_directories(upstream, mlite)
        _require(not failing["passed"], "Comparator accepted a corrupted tensor")
        tensor_mismatches = [item for item in failing["mismatches"] if item["kind"] == "tensor"]
        _require(tensor_mismatches, "Comparator omitted tensor mismatch details")
        _require(
            all(
                not item["torch_equal"] and not item["assert_close_atol0_rtol0"]
                for item in tensor_mismatches
            ),
            "Comparator mismatch did not record both failed exact checks",
        )
        fail_json = root / "fail" / "comparison.json"
        _write_comparison(failing, fail_json, emit_stdout_marker=False)
        _require(
            not (fail_json.parent / MARKER_NAME).exists(),
            "Failing compare incorrectly wrote NON_SKIP marker",
        )
    print("CPU_COMPARATOR_SELFTEST_PASSED", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one implementation and trajectory")
    run_parser.add_argument("--implementation", required=True, choices=("upstream", "mlite"))
    run_parser.add_argument(
        "--trajectory", required=True, choices=("continuous", "save", "resume")
    )
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--checkpoint-dir", required=True)
    run_parser.set_defaults(handler=run_one)

    compare_parser = subparsers.add_parser("compare", help="strictly compare all rank artifacts")
    compare_parser.add_argument("--upstream-dir", required=True)
    compare_parser.add_argument("--mlite-dir", required=True)
    compare_parser.add_argument("--output-json", required=True)
    compare_parser.set_defaults(handler=compare_runs)

    adam_parser = subparsers.add_parser(
        "validate-adam", help="validate the frozen text-only Adam DistOpt regression"
    )
    adam_parser.add_argument("--input-json", required=True)
    adam_parser.add_argument("--output-json", required=True)
    adam_parser.set_defaults(handler=validate_adam_text_only_command)

    selftest_parser = subparsers.add_parser(
        "selftest", help="exercise the recursive comparator without CUDA or Megatron"
    )
    selftest_parser.set_defaults(handler=comparator_selftest)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
