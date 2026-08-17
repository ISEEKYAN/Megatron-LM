# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Matched-pair contracts for single LoRA and a one-slot dense LoRA bank."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.lite.primitive.ckpt.hf_weights import VLLM_LORA_NAME_PREFIX
from megatron.lite.primitive.modules import lora as lora_module
from megatron.lite.primitive.modules import multi_lora_bank, multi_lora_kernel

from megatron.lite.primitive.modules.lora import (  # isort: skip
    LinearLoRA,
    LoraSpec,
    SharedGroupedLinearLoRA,
)
from megatron.lite.primitive.modules.multi_lora_bank import (  # isort: skip
    DenseLoraBank,
    LoraBankPartition,
    MultiLoraTrainingState,
    NamedLoraBankRegistry,
    apply_batched_lora_delta,
)

pytestmark = [pytest.mark.mlite]


def _assert_single_slot_pair(
    adapter: LinearLoRA, x: torch.Tensor, grad_out: torch.Tensor, *, bank_kwargs=None
) -> None:
    bank_a = adapter.lora_a.detach().clone().unsqueeze(0).requires_grad_()
    bank_b = adapter.lora_b.detach().clone().unsqueeze(0).requires_grad_()
    bank_x = x.detach().clone().requires_grad_()
    single_x = x.detach().clone().requires_grad_()
    single = adapter(single_x)
    bank = apply_batched_lora_delta(
        DenseLoraBank(bank_a, bank_b),
        bank_x,
        torch.zeros(bank_x.numel() // bank_x.shape[-1], dtype=torch.long),
        scale=adapter.scale,
        **(bank_kwargs or {}),
    )
    torch.testing.assert_close(single, bank)
    (single * grad_out).sum().backward()
    (bank * grad_out).sum().backward()
    torch.testing.assert_close(single_x.grad, bank_x.grad)
    torch.testing.assert_close(adapter.lora_a.grad, bank_a.grad[0])
    torch.testing.assert_close(adapter.lora_b.grad, bank_b.grad[0])


@pytest.mark.parametrize(
    ("alpha", "use_rslora"), [(None, False), (8, False), (8, True)]
)
def test_linear_lora_matches_one_slot_bank_forward_and_all_gradients(alpha, use_rslora):
    torch.manual_seed(17)
    adapter = LinearLoRA(5, 7, 3, alpha=alpha, use_rslora=use_rslora).double()
    with torch.no_grad():
        adapter.lora_b.normal_()
    x = torch.randn(2, 3, 5, dtype=torch.float64)
    grad_out = torch.randn(2, 3, 7, dtype=torch.float64)
    _assert_single_slot_pair(adapter, x, grad_out)


def test_grouped_shared_lora_matches_one_slot_bank_and_preserves_split_validation():
    torch.manual_seed(23)
    adapter = SharedGroupedLinearLoRA(2, 5, 7, 3, alpha=8, use_rslora=True).double()
    with torch.no_grad():
        adapter.lora_b.normal_()
    x = torch.randn(6, 5, dtype=torch.float64)
    grad_out = torch.randn(6, 7, dtype=torch.float64)
    bank_a = adapter.lora_a.detach().clone().unsqueeze(0).requires_grad_()
    bank_b = adapter.lora_b.detach().clone().unsqueeze(0).requires_grad_()
    bank_x = x.detach().clone().requires_grad_()
    single_x = x.detach().clone().requires_grad_()
    single = adapter(single_x, [2, 4])
    bank = apply_batched_lora_delta(
        DenseLoraBank(bank_a, bank_b),
        bank_x,
        torch.zeros(6, dtype=torch.long),
        scale=adapter.scale,
    )
    torch.testing.assert_close(single, bank)
    (single * grad_out).sum().backward()
    (bank * grad_out).sum().backward()
    torch.testing.assert_close(single_x.grad, bank_x.grad)
    torch.testing.assert_close(adapter.lora_a.grad, bank_a.grad[0])
    torch.testing.assert_close(adapter.lora_b.grad, bank_b.grad[0])
    with pytest.raises(ValueError, match="expected 2 splits"):
        adapter(x, [6])


def test_single_and_one_slot_bank_share_the_bank_executor(monkeypatch):
    """Numerical parity alone must not leave two independent implementations."""
    calls = 0
    original = multi_lora_bank.apply_batched_lora_delta

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(multi_lora_bank, "apply_batched_lora_delta", counted)
    adapter = LinearLoRA(3, 4, 2)
    with torch.no_grad():
        adapter.lora_b.normal_()
    x = torch.randn(5, 3)
    adapter(x)
    multi_lora_bank.apply_batched_lora_delta(
        DenseLoraBank(adapter.lora_a.unsqueeze(0), adapter.lora_b.unsqueeze(0)),
        x,
        torch.zeros(5, dtype=torch.long),
        scale=adapter.scale,
    )
    assert calls == 2


def test_single_lora_forward_does_not_allocate_dense_slot_indices(monkeypatch):
    adapter = LinearLoRA(3, 4, 2)

    def unexpected_zeros(*args, **kwargs):
        raise AssertionError("single LoRA must not allocate per-token slot indices")

    monkeypatch.setattr(lora_module.torch, "zeros", unexpected_zeros)
    assert adapter(torch.randn(5, 3)).shape == (5, 4)


def test_nonzero_dropout_preserves_the_single_lora_compatibility_path(monkeypatch):
    def unexpected_bank_dispatch(*args, **kwargs):
        raise AssertionError("dropout LoRA must retain its established execution path")

    monkeypatch.setattr(
        multi_lora_kernel, "dense_batched_lora_forward", unexpected_bank_dispatch
    )
    adapter = LinearLoRA(3, 4, 2, dropout=0.25)
    assert adapter(torch.randn(5, 3)).shape == (5, 4)


def _tp_pair_worker(
    rank: int, world_size: int, init_file: str, kind: str, queue
) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(31)
        tokens, hidden, out_features, lora_rank = 4, 4, 6, 4
        x_full = torch.randn(tokens, hidden, dtype=torch.float64)
        a_full = torch.randn(lora_rank, hidden, dtype=torch.float64)
        b_full = torch.randn(out_features, lora_rank, dtype=torch.float64)
        if kind == "qkv":
            token_slice = slice(rank * 2, (rank + 1) * 2)
            rank_slice = slice(rank * 2, (rank + 1) * 2)
            out_slice = slice(rank * 3, (rank + 1) * 3)
            adapter = LinearLoRA(
                hidden,
                3,
                lora_rank,
                alpha=8,
                use_rslora=True,
                sequence_parallel_input=True,
                tp_group=dist.group.WORLD,
                tp_rank=rank,
                rank_partition_size=2,
                rank_partitioned_a=True,
            ).double()
            a_local, b_local = a_full[rank_slice], b_full[out_slice]
            x = x_full[token_slice]
            partition = LoraBankPartition(tp_size=2, rank_partitioned_a=True)
            kwargs = {"sequence_parallel_input": True}
        else:
            input_slice = slice(rank * 2, (rank + 1) * 2)
            output_slice = slice(rank * 2, (rank + 1) * 2)
            adapter = LinearLoRA(
                2,
                hidden,
                lora_rank,
                alpha=8,
                use_rslora=True,
                tp_group=dist.group.WORLD,
                tp_rank=rank,
                input_parallel_reduce=True,
                output_partition_size=2,
                output_partitioned_b=True,
                sequence_parallel_scatter_output=True,
            ).double()
            a_local, b_local = a_full[:, input_slice], b_full[output_slice]
            x = x_full[:, input_slice]
            partition = LoraBankPartition(tp_size=2, output_partitioned_b=True)
            kwargs = {
                "tp_rank": rank,
                "input_parallel_reduce": True,
                "sequence_parallel_scatter_output": True,
            }
        with torch.no_grad():
            adapter.lora_a.copy_(a_local)
            adapter.lora_b.copy_(b_local)
        single_x = x.clone().requires_grad_()
        bank_x = x.clone().requires_grad_()
        bank_a = a_local.clone().unsqueeze(0).requires_grad_()
        bank_b = b_local.clone().unsqueeze(0).requires_grad_()
        saved_shapes = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: saved_shapes.append(tuple(tensor.shape)) or tensor,
            lambda tensor: tensor,
        ):
            single = adapter(single_x)
        bank = apply_batched_lora_delta(
            DenseLoraBank(bank_a, bank_b, partition),
            bank_x,
            torch.zeros(x.shape[0], dtype=torch.long),
            scale=adapter.scale,
            tp_group=dist.group.WORLD,
            **kwargs,
        )
        grad = torch.randn_like(single)
        (single * grad).sum().backward()
        (bank * grad).sum().backward()
        queue.put(
            {
                "rank": rank,
                "error": max(
                    (single - bank).abs().max().item(),
                    (single_x.grad - bank_x.grad).abs().max().item(),
                    (adapter.lora_a.grad - bank_a.grad[0]).abs().max().item(),
                    (adapter.lora_b.grad - bank_b.grad[0]).abs().max().item(),
                ),
                "saved_shapes": saved_shapes,
            }
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.parametrize("kind", ["qkv", "proj"])
def test_tp2_sp_single_matches_one_slot_bank_forward_and_gradients(tmp_path, kind):
    init_file = tmp_path / f"single-bank-{kind}-init"
    queue = mp.get_context("spawn").SimpleQueue()
    mp.spawn(
        _tp_pair_worker, args=(2, str(init_file), kind, queue), nprocs=2, join=True
    )
    results = sorted((queue.get() for _ in range(2)), key=lambda item: item["rank"])
    assert [item["error"] for item in results] == pytest.approx([0.0, 0.0], abs=1e-12)
    if kind == "qkv":
        # The rank-partitioned SP fastpath may save local x and both factors,
        # but never the globally gathered activation rows retained by the
        # generic two-stage bank autograd path.
        for item in results:
            assert sorted(item["saved_shapes"]) == [(2, 4), (2, 4), (3, 4)]
    if init_file.exists():
        os.unlink(init_file)


def test_one_slot_checkpoint_and_export_preserve_linear_factors():
    adapter = LinearLoRA(3, 4, 2, alpha=8, use_rslora=True)
    with torch.no_grad():
        adapter.lora_a.copy_(torch.arange(6).reshape(2, 3))
        adapter.lora_b.copy_(torch.arange(8).reshape(4, 2))
    bank = DenseLoraBank(
        torch.nn.Parameter(adapter.lora_a.detach().clone().unsqueeze(0)),
        torch.nn.Parameter(adapter.lora_b.detach().clone().unsqueeze(0)),
    )
    registry = NamedLoraBankRegistry(
        banks={"layer.weight": bank},
        names={"default": 0},
        rank=2,
        alpha=8,
        base_model_identity={},
        lora_spec=LoraSpec(enabled=True, rank=2, alpha=8, use_rslora=True),
    )
    state = MultiLoraTrainingState(registry, {})
    restored_bank = DenseLoraBank(
        torch.nn.Parameter(torch.zeros_like(bank.a_bank)),
        torch.nn.Parameter(torch.zeros_like(bank.b_bank)),
    )
    restored_registry = NamedLoraBankRegistry(
        banks={"layer.weight": restored_bank},
        names={"default": 0},
        rank=2,
        alpha=8,
        base_model_identity={},
        lora_spec=registry.lora_spec,
    )
    restored = MultiLoraTrainingState(restored_registry, {})
    restored.load_state_dict(state.state_dict())
    selected_a, selected_b = restored_registry.select("default")["layer.weight"]
    linear_a, linear_b = adapter.materialized_lora_factors()
    torch.testing.assert_close(selected_a, linear_a)
    torch.testing.assert_close(selected_b, linear_b)

    class IdentitySpec:
        @staticmethod
        def tp_spec(name):
            assert name == "layer.weight"
            return None

        @staticmethod
        def is_expert(name):
            return False

        @staticmethod
        def native_to_hf(name, tensor):
            assert name == "layer.weight"
            return [(name, tensor)]

    ps = SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        etp_size=1,
        etp_rank=0,
        etp_group=None,
        ep_size=1,
        ep_rank=0,
        ep_group=None,
    )
    exported = restored_registry.export_hf_state("default", IdentitySpec(), ps)
    prefix = f"{VLLM_LORA_NAME_PREFIX}layer"
    torch.testing.assert_close(exported[f"{prefix}.lora_A.weight"], linear_a)
    torch.testing.assert_close(exported[f"{prefix}.lora_B.weight"], linear_b)
