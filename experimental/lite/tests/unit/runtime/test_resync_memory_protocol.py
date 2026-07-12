# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit coverage for the rollout resync memory protocol primitives.

The rollout weight sync materialises full parameters on the GPU; these helpers
let the export path evict training-only state (optimizer moments + gradient
buffers) before the all-gather and restore whatever it found on entry.
"""

from __future__ import annotations

import types

import torch

from megatron.lite.runtime.megatron_utils import (
    cuda_mem_snapshot,
    free_grad_buffers,
    model_grads_resident,
    optimizer_states_on_gpu,
)

pytestmark = __import__("pytest").mark.mlite


class _FakeStorage:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size

    def resize_(self, size: int) -> None:
        self._size = size


class _FakeGradData:
    def __init__(self, size: int) -> None:
        self._storage = _FakeStorage(size)

    def storage(self) -> _FakeStorage:
        return self._storage


class _FakeBuffer:
    def __init__(self, grad_size: int) -> None:
        self.grad_data = _FakeGradData(grad_size)


class _FakeDDP:
    """Duck-types the pieces of Megatron DDP that the grad helpers touch."""

    def __init__(self, grad_sizes, expert_grad_sizes=()) -> None:
        self.buffers = [_FakeBuffer(s) for s in grad_sizes]
        self.expert_parallel_buffers = [_FakeBuffer(s) for s in expert_grad_sizes]
        self.module = torch.nn.Linear(1, 1)


def _patch_is_ddp(monkeypatch) -> None:
    monkeypatch.setattr(
        "megatron.lite.runtime.megatron_utils._is_megatron_ddp",
        lambda chunk: isinstance(chunk, _FakeDDP),
    )


def test_optimizer_states_on_gpu_detects_resident_moments() -> None:
    assert optimizer_states_on_gpu(None) is False

    cpu_state = {torch.nn.Parameter(torch.zeros(1)): {"exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)}}
    opt = types.SimpleNamespace(optimizer=types.SimpleNamespace(state=cpu_state))
    assert optimizer_states_on_gpu(opt) is False


def test_optimizer_states_on_gpu_true_when_moment_on_cuda(monkeypatch) -> None:
    class _CudaTensor:
        is_cuda = True

    # Bypass the ChainedOptimizer import path with a plain (non-chained) optimizer.
    state = {"p": {"exp_avg": _CudaTensor(), "exp_avg_sq": _CudaTensor()}}
    opt = types.SimpleNamespace(optimizer=types.SimpleNamespace(state=state))
    assert optimizer_states_on_gpu(opt) is True


def test_model_grads_resident_and_free_records_size_for_restore(monkeypatch) -> None:
    _patch_is_ddp(monkeypatch)
    ddp = _FakeDDP(grad_sizes=[128, 0], expert_grad_sizes=[64])

    assert model_grads_resident([ddp]) is True

    free_grad_buffers([ddp])

    # Resident buffers are released to zero, and the pre-free size is recorded
    # so load_model_to_gpu(..., load_grad=True) can re-allocate them.
    assert ddp.buffers[0].grad_data.storage().size() == 0
    assert ddp.buffers[0].grad_data_size == 128
    assert ddp.expert_parallel_buffers[0].grad_data.storage().size() == 0
    assert ddp.expert_parallel_buffers[0].grad_data_size == 64
    # An already-empty buffer keeps no bogus recorded size.
    assert not hasattr(ddp.buffers[1], "grad_data_size")

    assert model_grads_resident([ddp]) is False


def test_model_grads_resident_ignores_non_ddp(monkeypatch) -> None:
    _patch_is_ddp(monkeypatch)
    assert model_grads_resident([torch.nn.Linear(2, 2)]) is False


def test_cuda_mem_snapshot_is_safe_without_cuda() -> None:
    snap = cuda_mem_snapshot("resync/enter")
    assert snap["tag"] == "resync/enter"
    assert set(snap) >= {"allocated_gib", "reserved_gib", "max_allocated_gib"}
    for key in ("allocated_gib", "reserved_gib", "max_allocated_gib"):
        assert isinstance(snap[key], float)
