# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Layered single-LoRA parity against Megatron-Bridge's production adapter.

This is an opt-in reference suite: it runs when ``megatron.bridge`` and its
runtime dependencies are installed, and otherwise skips at collection.  It
does not exercise an inference engine or an RL loop.  Every numerical check
loads identical base, A, B, input, and output-gradient tensors into both
implementations so initialization noise cannot hide a contract difference.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import megatron.lite.primitive.modules.lora as mlite_lora_module
import pytest
import torch
import torch.nn as nn
from megatron.lite.primitive.modules.lora import (
    LORA_DEFAULT_TARGET_MODULES,
    LinearLoRA,
    SharedGroupedLinearLoRA,
    freeze_non_lora_params,
    normalize_lora_spec,
)
from megatron.lite.primitive.modules.lora_apply import LoRAWrappedLinear


def _install_bridge_source_packages() -> None:
    """Load PEFT sources without triggering Bridge's unrelated model registry."""
    source_root = os.getenv("MEGATRON_BRIDGE_SRC")
    if not source_root:
        return
    bridge_root = Path(source_root) / "megatron" / "bridge"
    if not bridge_root.is_dir():
        raise RuntimeError(f"MEGATRON_BRIDGE_SRC has no bridge package: {bridge_root}")
    for name, path in (
        ("megatron.bridge", bridge_root),
        ("megatron.bridge.peft", bridge_root / "peft"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        sys.modules[name] = package


_install_bridge_source_packages()

bridge_lora = pytest.importorskip("megatron.bridge.peft.lora", exc_type=ImportError)
bridge_lora_layers = pytest.importorskip(
    "megatron.bridge.peft.lora_layers", exc_type=ImportError
)

BridgeLoRA = bridge_lora.LoRA
BridgeLinearAdapter = bridge_lora_layers.LinearAdapter
BridgeLoRAMerge = bridge_lora.LoRAMerge

pytestmark = [pytest.mark.mlite, pytest.mark.mbridge]

RANK = 4
ALPHA = 8
DTYPE = torch.float64
TARGETS = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")


@dataclass(frozen=True)
class Surface:
    name: str
    in_features: int
    out_features: int


SURFACES = (
    Surface("linear_qkv", 8, 12),
    Surface("linear_proj", 12, 8),
    Surface("linear_fc1", 8, 16),
    Surface("linear_fc2", 16, 8),
)


def _tensors(surface: Surface):
    generator = torch.Generator().manual_seed(20260808)
    base = torch.randn(
        surface.out_features, surface.in_features, generator=generator, dtype=DTYPE
    )
    a = torch.randn(RANK, surface.in_features, generator=generator, dtype=DTYPE)
    b = torch.randn(surface.out_features, RANK, generator=generator, dtype=DTYPE)
    x = torch.randn(3, 2, surface.in_features, generator=generator, dtype=DTYPE)
    grad_out = torch.randn(3, 2, surface.out_features, generator=generator, dtype=DTYPE)
    return base, a, b, x, grad_out


def _make_pair(surface: Surface):
    base, a, b, x, grad_out = _tensors(surface)

    mlite_base = nn.Linear(
        surface.in_features, surface.out_features, bias=False, dtype=DTYPE
    )
    mlite_adapter = LinearLoRA(
        surface.in_features, surface.out_features, RANK, alpha=ALPHA
    ).to(DTYPE)
    mlite = LoRAWrappedLinear(mlite_base, mlite_adapter)

    bridge_base = nn.Linear(
        surface.in_features, surface.out_features, bias=False, dtype=DTYPE
    )
    bridge = BridgeLinearAdapter(
        bridge_base,
        dim=RANK,
        alpha=ALPHA,
        dropout=0.0,
        lora_A_init_method="uniform",
        lora_dtype=DTYPE,
    )

    with torch.no_grad():
        mlite.base.weight.copy_(base)
        mlite.adapter.lora_a.copy_(a)
        mlite.adapter.lora_b.copy_(b)
        bridge.weight.copy_(base)
        bridge.linear_in.weight.copy_(a)
        bridge.linear_out.weight.copy_(b)

    return mlite, bridge, x, grad_out


def _adapter_inventory(
    module: nn.Module,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    aliases = {
        "adapter.lora_a": "A",
        "adapter.lora_b": "B",
        "linear_in.weight": "A",
        "linear_out.weight": "B",
    }
    return {
        aliases[name]: (tuple(parameter.shape), parameter.dtype)
        for name, parameter in module.named_parameters()
        if name in aliases
    }


def test_l0_explicit_recipe_resolves_to_the_same_contract():
    mlite = normalize_lora_spec(
        {
            "enabled": True,
            "rank": RANK,
            "alpha": ALPHA,
            "dropout": 0.0,
            "target_modules": TARGETS,
            "init": "default",
        }
    )
    bridge = BridgeLoRA(
        dim=RANK,
        alpha=ALPHA,
        dropout=0.0,
        target_modules=list(TARGETS),
        lora_A_init_method="kaiming",
        lora_B_init_method="zero",
    )

    assert mlite.enabled
    assert mlite.rank == bridge.dim == RANK
    assert mlite.alpha == bridge.alpha == ALPHA
    assert mlite.dropout == bridge.dropout == 0.0
    assert tuple(bridge.target_modules) == mlite.target_modules == TARGETS
    assert LORA_DEFAULT_TARGET_MODULES == TARGETS
    assert mlite.scale == bridge.alpha / bridge.dim
    assert bridge.lora_A_init_method == "kaiming"
    assert bridge.lora_B_init_method == "zero"


@pytest.mark.parametrize("surface", SURFACES, ids=lambda surface: surface.name)
def test_l0_adapter_inventory_shape_dtype_and_trainable_numel_match(surface):
    mlite, bridge, _, _ = _make_pair(surface)
    mlite_stats = freeze_non_lora_params(mlite)

    assert _adapter_inventory(mlite) == _adapter_inventory(bridge)
    assert mlite_stats["lora_tensors"] == 2
    assert mlite_stats["lora_numel"] == sum(
        parameter.numel()
        for name, parameter in bridge.named_parameters()
        if name in {"linear_in.weight", "linear_out.weight"}
    )
    assert not bridge.weight.requires_grad
    assert bridge.linear_in.weight.requires_grad
    assert bridge.linear_out.weight.requires_grad


@pytest.mark.parametrize("surface", SURFACES, ids=lambda surface: surface.name)
def test_l1_l2_forward_dx_da_db_match_on_all_megatron_surfaces(surface):
    mlite, bridge, x, grad_out = _make_pair(surface)
    x_mlite = x.clone().requires_grad_(True)
    x_bridge = x.clone().requires_grad_(True)

    out_mlite = mlite(x_mlite)
    out_bridge = bridge(x_bridge)
    (out_mlite * grad_out).sum().backward()
    (out_bridge * grad_out).sum().backward()

    torch.testing.assert_close(out_mlite, out_bridge, rtol=0, atol=1e-12)
    torch.testing.assert_close(x_mlite.grad, x_bridge.grad, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        mlite.adapter.lora_a.grad, bridge.linear_in.weight.grad, rtol=0, atol=1e-12
    )
    torch.testing.assert_close(
        mlite.adapter.lora_b.grad, bridge.linear_out.weight.grad, rtol=0, atol=1e-12
    )


def test_l1_scaling_negative_control_is_load_bearing():
    surface = SURFACES[0]
    mlite, bridge, x, _ = _make_pair(surface)
    with torch.no_grad():
        reference = bridge(x)
        mlite.adapter.scale *= 0.5
        wrong = mlite(x)

    assert not torch.allclose(wrong, reference)


@pytest.mark.parametrize("parallelism", ["column", "row"])
def test_l3_tp2_materialized_delta_shards_match_bridge_merge(parallelism, monkeypatch):
    """Compare the TP shard layouts without requiring CUDA collectives."""
    surface = Surface("linear_qkv", 8, 12)
    base, a, b, _, _ = _tensors(surface)
    tp_size = 2
    fake_group = object()

    if parallelism == "column":
        a_shards = a.chunk(tp_size, dim=0)
        b_shards = b.chunk(tp_size, dim=0)
        base_shards = base.chunk(tp_size, dim=0)

        inputs = zip(base_shards, a_shards, b_shards)
        gathered_shards = a_shards
    else:
        a_shards = a.chunk(tp_size, dim=1)
        b_shards = b.chunk(tp_size, dim=0)
        base_shards = base.chunk(tp_size, dim=1)

        inputs = zip(base_shards, a_shards, b_shards)
        gathered_shards = b_shards

    def fake_all_gather(outputs, _local, group):
        assert group is fake_group
        for output, shard in zip(outputs, gathered_shards):
            output.copy_(shard)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    for rank, (base_shard, a_shard, b_shard) in enumerate(inputs):
        if parallelism == "column":
            mlite = LinearLoRA(
                surface.in_features,
                surface.out_features // tp_size,
                RANK,
                alpha=ALPHA,
                rank_partition_size=tp_size,
                rank_partitioned_a=True,
            ).to(DTYPE)
            gathered = a
        else:
            mlite = LinearLoRA(
                surface.in_features // tp_size,
                surface.out_features,
                RANK,
                alpha=ALPHA,
                output_partition_size=tp_size,
                output_partitioned_b=True,
            ).to(DTYPE)
            gathered = b
        with torch.no_grad():
            mlite.lora_a.copy_(a_shard)
            mlite.lora_b.copy_(b_shard)
        monkeypatch.setattr(
            mlite_lora_module,
            "_all_gather_last_dim_forward",
            lambda tensor, _group, _size, full=gathered: full.t(),
        )
        mlite_merged = base_shard + mlite.materialized_delta_weight()
        bridge_merged = BridgeLoRAMerge().merge(
            base_shard,
            b_shard,
            a_shard,
            ALPHA,
            RANK,
            tp_group=fake_group,
            tp_size=tp_size,
        )
        torch.testing.assert_close(bridge_merged, mlite_merged, rtol=0, atol=1e-12)


def test_l3_tp_shard_order_negative_control_is_load_bearing(monkeypatch):
    surface = Surface("linear_qkv", 8, 12)
    base, a, b, _, _ = _tensors(surface)
    a_shards = a.chunk(2, dim=0)
    b_local = b.chunk(2, dim=0)[0]
    base_local = base.chunk(2, dim=0)[0]
    fake_group = object()

    def reversed_all_gather(outputs, _local, group):
        assert group is fake_group
        for output, shard in zip(outputs, reversed(a_shards)):
            output.copy_(shard)

    monkeypatch.setattr(torch.distributed, "all_gather", reversed_all_gather)
    wrong = BridgeLoRAMerge().merge(
        base_local,
        b_local,
        a_shards[0],
        ALPHA,
        RANK,
        tp_group=fake_group,
        tp_size=2,
    )
    correct = base_local + (b_local @ a) * (ALPHA / RANK)
    assert not torch.allclose(wrong, correct)


def test_l3_sequence_parallel_flag_is_a_tp1_noop_and_matches_bridge():
    mlite, bridge, x, grad_out = _make_pair(SURFACES[0])
    mlite.adapter.sequence_parallel_input = True
    x_mlite = x.clone().requires_grad_(True)
    x_bridge = x.clone().requires_grad_(True)

    out_mlite = mlite(x_mlite)
    out_bridge = bridge(x_bridge)
    (out_mlite * grad_out).sum().backward()
    (out_bridge * grad_out).sum().backward()

    torch.testing.assert_close(out_mlite, out_bridge, rtol=0, atol=1e-12)
    torch.testing.assert_close(x_mlite.grad, x_bridge.grad, rtol=0, atol=1e-12)


def test_l4_shared_grouped_expert_forward_and_gradients_match_with_zero_tokens():
    num_experts = 3
    splits = [2, 0, 3]
    surface = Surface("linear_fc1", 8, 16)
    _, a, b, x, grad_out = _tensors(surface)
    x = x.reshape(-1, surface.in_features)[: sum(splits)]
    grad_out = grad_out.reshape(-1, surface.out_features)[: sum(splits)]

    mlite = SharedGroupedLinearLoRA(
        num_experts, surface.in_features, surface.out_features, RANK, alpha=ALPHA
    ).to(DTYPE)
    bridge_base = nn.Linear(
        surface.in_features, surface.out_features, bias=False, dtype=DTYPE
    )
    bridge = BridgeLinearAdapter(
        bridge_base,
        dim=RANK,
        alpha=ALPHA,
        dropout=0.0,
        lora_A_init_method="uniform",
        lora_dtype=DTYPE,
    )
    with torch.no_grad():
        mlite.lora_a.copy_(a)
        mlite.lora_b.copy_(b)
        bridge.weight.zero_()
        bridge.linear_in.weight.copy_(a)
        bridge.linear_out.weight.copy_(b)

    x_mlite = x.clone().requires_grad_(True)
    x_bridge = x.clone().requires_grad_(True)
    out_mlite = mlite(x_mlite, splits)
    out_bridge = bridge(x_bridge)
    (out_mlite * grad_out).sum().backward()
    (out_bridge * grad_out).sum().backward()

    torch.testing.assert_close(out_mlite, out_bridge, rtol=0, atol=1e-12)
    torch.testing.assert_close(x_mlite.grad, x_bridge.grad, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        mlite.lora_a.grad, bridge.linear_in.weight.grad, rtol=0, atol=1e-12
    )
    torch.testing.assert_close(
        mlite.lora_b.grad, bridge.linear_out.weight.grad, rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("surface", SURFACES, ids=lambda surface: surface.name)
def test_l5_adapter_and_merged_export_tensors_match(surface):
    mlite, bridge, _, _ = _make_pair(surface)
    mlite_a, mlite_b = mlite.adapter.materialized_lora_factors()
    bridge_a = bridge.linear_in.weight
    bridge_b = bridge.linear_out.weight

    torch.testing.assert_close(mlite_a, bridge_a, rtol=0, atol=0)
    torch.testing.assert_close(mlite_b, bridge_b, rtol=0, atol=0)
    assert mlite_a.dtype == bridge_a.dtype
    assert mlite_b.dtype == bridge_b.dtype
    assert tuple(mlite_a.shape) == tuple(bridge_a.shape)
    assert tuple(mlite_b.shape) == tuple(bridge_b.shape)
    assert {name for name in bridge.state_dict() if name.startswith("linear_")} == {
        "linear_in.weight",
        "linear_out.weight",
    }
    assert {"linear_in.weight": mlite_a, "linear_out.weight": mlite_b}.keys() == {
        "linear_in.weight": bridge_a,
        "linear_out.weight": bridge_b,
    }.keys()

    bridge_merged = BridgeLoRAMerge().merge(
        bridge.weight,
        bridge_b,
        bridge_a,
        bridge.alpha,
        bridge.dim,
        tp_group=None,
        tp_size=1,
    )
    mlite_merged = mlite.base.weight + mlite.adapter.materialized_delta_weight()
    torch.testing.assert_close(mlite_merged, bridge_merged, rtol=0, atol=1e-12)
