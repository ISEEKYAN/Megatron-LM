# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""M-FSDP multi-forward full-parameter lifecycle tests. CPU-only."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from megatron.lite.primitive.optimizers.mfsdp import buffer as mfsdp_buffer
from megatron.lite.runtime.contracts.config import ParallelConfig
from torch.utils.checkpoint import checkpoint

from megatron.lite.primitive.optimizers.mfsdp import (  # isort: skip
    optimizer as mfsdp_optimizer,
)


class _NormUnit(nn.Module):
    """A unit whose LayerNorm contributes 1-D (retain-through-backward) params."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, value):
        return self.norm(torch.nn.functional.gelu(self.linear(value)))


class _NormModel(nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        self.unit = _NormUnit(dim)
        self.out = nn.Linear(dim, 2, bias=False)

    def forward(self, value):
        return self.out(self.unit(value))


def _build_chunk():
    ps = SimpleNamespace(
        dp_cp_group=None,
        dp_group=None,
        ep_dp_group=None,
        ep_group=None,
        etp_group=None,
        tp_group=None,
        pp_group=None,
        dp_cp_size=1,
        expert_dp_size=1,
    )
    opt = SimpleNamespace(
        optimizer="adam",
        lr=1.0e-3,
        min_lr=0.0,
        weight_decay=0.0,
        clip_grad=1000.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1.0e-8,
        override_optimizer_config={"mfsdp_sharding_strategy": "optim_grads_params"},
    )
    torch.manual_seed(0)
    chunks, optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [_NormModel()],
        engine_cfg=SimpleNamespace(
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1), optimizer=opt
        ),
        ps=ps,
        is_expert=lambda _name: False,
        fsdp_unit_modules=(_NormUnit,),
    )
    return chunks[0], optimizer


def _any_slot_busy(chunk) -> bool:
    seen: set[int] = set()
    for bucket in chunk.param_sync.buckets:
        allocator = bucket.allocator
        if id(allocator) in seen:
            continue
        seen.add(id(allocator))
        if any(getattr(allocator, "_busy", {}).values()):
            return True
        if getattr(bucket, "_full_lease", None) is not None:
            return True
    return False


def _release_cached_buffers(chunk) -> None:
    """Exercise the allocator guard used by wake/export without resync helpers."""
    seen: set[int] = set()
    for bucket in chunk.param_sync.buckets:
        allocator = bucket.allocator
        if id(allocator) not in seen:
            seen.add(id(allocator))
            allocator.release_cached()


def test_eval_root_storage_survives_next_train_forward_then_releases_after_backward():
    chunk, optimizer = _build_chunk()
    value = torch.randn(3, 4)

    chunk.eval()
    with torch.no_grad():
        eval_output = chunk(value)
    non_unit_buckets = [
        bucket for bucket in chunk.param_sync.buckets if not bucket.is_fsdp_unit
    ]
    assert non_unit_buckets
    assert all(bucket._full_ready for bucket in non_unit_buckets)

    chunk.train()
    optimizer.zero_grad()
    train_output = chunk(value)
    assert torch.equal(train_output, eval_output)
    assert all(bucket._full_ready for bucket in non_unit_buckets)

    train_output.square().mean().backward()
    optimizer.finish_grad_sync()
    assert optimizer.step()[0]
    assert not chunk.param_sync._preserve_non_fsdp_units_after_forward
    assert not _any_slot_busy(chunk)


def test_layernorm_params_follow_mcore_unit_reshard_lifecycle():
    chunk, _optimizer = _build_chunk()
    assert not any(
        bucket.retain_full_storage_through_backward
        for bucket in chunk.param_sync.buckets
    )


def test_train_step_leaves_no_active_buffers():
    chunk, optimizer = _build_chunk()
    value = torch.randn(3, 4)
    target = torch.randn(3, 2)

    optimizer.zero_grad()
    torch.nn.functional.mse_loss(chunk(value), target).backward()
    optimizer.finish_grad_sync()

    # A grad-enabled forward+backward releases the retain bucket via the
    # autograd graph, so the export/wake reclaim path is clean.
    assert not _any_slot_busy(chunk)
    _release_cached_buffers(chunk)  # must not raise


def test_no_grad_recompute_forward_releases_retain_bucket():
    """The reproducer: forward+backward, then a no_grad second forward.

    Before the fix the retain bucket's full-parameter lease stayed ``busy``
    after the grad-disabled recompute, so ``release_cached_buffers`` (the export/
    colocated-wake reclaim) raised the "active buffers" RuntimeError.
    """
    chunk, optimizer = _build_chunk()
    value = torch.randn(3, 4)
    target = torch.randn(3, 2)

    optimizer.zero_grad()
    torch.nn.functional.mse_loss(chunk(value), target).backward()
    optimizer.finish_grad_sync()

    # Rollout-correction logprob recompute: a second forward with no backward.
    with torch.no_grad():
        chunk(value)

    assert not _any_slot_busy(
        chunk
    ), "grad-disabled recompute forward left a full-parameter buffer active"
    # The reclaim a colocated vLLM wake / full-parameter export performs.
    _release_cached_buffers(chunk)  # must not raise


def test_inference_mode_recompute_forward_releases_retain_bucket():
    chunk, optimizer = _build_chunk()
    value = torch.randn(3, 4)
    target = torch.randn(3, 2)

    optimizer.zero_grad()
    torch.nn.functional.mse_loss(chunk(value), target).backward()
    optimizer.finish_grad_sync()

    with torch.inference_mode():
        chunk(value)

    assert not _any_slot_busy(chunk)
    _release_cached_buffers(chunk)  # must not raise


def test_grad_enabled_forward_reshards_before_backward_like_mcore():
    chunk, optimizer = _build_chunk()
    value = torch.randn(3, 4)
    target = torch.randn(3, 2)

    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(chunk(value), target)
    assert loss.requires_grad
    assert not _any_slot_busy(chunk)

    loss.backward()
    optimizer.finish_grad_sync()
    # Backward drains it.
    assert not _any_slot_busy(chunk)


def test_activation_recompute_uses_pre_backward_lazy_release(monkeypatch):
    chunk, optimizer = _build_chunk()
    original_forward = chunk.module.forward

    def checkpointed_forward(value):
        hidden = checkpoint(chunk.module.unit, value, use_reentrant=False)
        return chunk.module.out(hidden)

    chunk.module.forward = checkpointed_forward
    states = []
    acquire_forward_owner = chunk.param_sync.acquire_forward_owner

    def record_acquire(owner_id, bucket_ids):
        result = acquire_forward_owner(owner_id, bucket_ids)
        states.append(chunk.param_sync._training_states[owner_id])
        return result

    monkeypatch.setattr(chunk.param_sync, "acquire_forward_owner", record_acquire)
    optimizer.zero_grad()
    torch.nn.functional.mse_loss(chunk(torch.randn(3, 4)), torch.randn(3, 2)).backward()
    optimizer.finish_grad_sync()
    chunk.module.forward = original_forward

    assert mfsdp_buffer.TrainingState.FORWARD in states
    assert mfsdp_buffer.TrainingState.PRE_BACKWARD in states
    assert set(chunk.param_sync._training_states.values()) == {
        mfsdp_buffer.TrainingState.IDLE
    }
    assert not _any_slot_busy(chunk)
