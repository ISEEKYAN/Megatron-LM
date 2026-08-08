# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""M-FSDP multi-forward (recompute) full-parameter buffer lifecycle. CPU-only.

A ``retain_full_storage_through_backward`` bucket (1-D params -- LayerNorm
weight/bias) keeps its gathered full-parameter buffer resident past the forward
so the matching backward can reuse it without re-gathering; the actual release
is deferred to ``_ReleaseBackward`` in the autograd graph. That deferral is only
valid when a backward will run. A *grad-disabled* forward (``torch.no_grad`` /
``inference_mode`` -- e.g. the DAPO ``bypass_mode=False`` rollout-correction
logprob recompute, a second forward inside one logical step) has no backward, so
the deferred release never fires and the bucket's full-parameter lease stays
pinned (its allocator slot ``busy``). The next ``release_cached`` -- a colocated
vLLM wake, a full-parameter export, or a ``move_model_state`` offload -- then
tripped ``DoubleBufferAllocator.release_cached``'s busy guard with a spurious
"Cannot release active M-FSDP communication buffers." RuntimeError (buffer.py).

These tests build a real M-FSDP-wrapped model (world_size=1, gloo-free CPU path)
and exercise the multi-forward timing that produced the crash.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from megatron.lite.primitive.optimizers.mfsdp import optimizer as mfsdp_optimizer
from megatron.lite.runtime.contracts.config import ParallelConfig


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
        override_optimizer_config={
            "mfsdp_sharding_strategy": "optim_grads_params",
        },
    )
    torch.manual_seed(0)
    chunks, optimizer = mfsdp_optimizer.build_mfsdp_stack(
        [_NormModel()],
        engine_cfg=SimpleNamespace(
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1),
            optimizer=opt,
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


def test_retain_bucket_is_actually_the_layernorm_1d_params():
    # Guards the premise: LayerNorm weight/bias form a retain-through-backward
    # bucket. If bucketing changes so nothing retains, the regression below
    # would pass vacuously.
    chunk, _optimizer = _build_chunk()
    retain = [
        [spec.name for spec in bucket.specs]
        for bucket in chunk.param_sync.buckets
        if bucket.retain_full_storage_through_backward
    ]
    assert retain == [["unit.norm.weight", "unit.norm.bias"]]


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

    assert not _any_slot_busy(chunk), (
        "grad-disabled recompute forward left a full-parameter buffer active"
    )
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


def test_grad_enabled_forward_still_retains_until_backward():
    # The retain optimization must be preserved on the training path: with grad
    # enabled the 1-D bucket stays resident after the forward (so backward reuses
    # it) and is only released once backward runs.
    chunk, optimizer = _build_chunk()
    value = torch.randn(3, 4)
    target = torch.randn(3, 2)

    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(chunk(value), target)
    # The output still owns the complete autograd graph before backward.
    assert loss.requires_grad

    loss.backward()
    optimizer.finish_grad_sync()
    # Backward drains it.
    assert not _any_slot_busy(chunk)
