# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real eight-process CPU regressions; missing ranks never enter the guard."""

from __future__ import annotations

import multiprocessing
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.mlite
ABSENT = 2  # Also the largest input: absence is unrelated to emptiness.


def _worker(rank, rendezvous, results, release, mode):
    # Spawned interpreters reuse the shared CPU-only TE import fixture.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from conftest import transformer_engine_import_stub

    transformer_engine_import_stub.__wrapped__(pytest.MonkeyPatch())()
    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=8, timeout=timedelta(seconds=25)
    )
    from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
    from megatron.lite.primitive.modules.moe import _AllToAll
    from megatron.lite.primitive.parallel import ParallelState

    ps = ParallelState(ep_size=8, ep_group=dist.group.WORLD)
    dispatcher = TokenDispatcher(8, 2, ps, use_deepep=False, moe_permute_fusion=False)
    count = 16 if rank == ABSENT else rank + 1
    x = torch.full((count, 2), float(rank), requires_grad=True)
    indices = torch.full((count, 1), (rank + 1) % 8, dtype=torch.long)
    scores = torch.ones(count, 1, requires_grad=True)
    absent = rank == ABSENT
    a2a = lambda: _AllToAll.apply(x[:1].expand(8, 2), [1] * 8, [1] * 8, ps.ep_group)  # noqa: E731
    dist.barrier()
    start = time.monotonic()
    try:
        if absent and mode == "dispatch_missing":
            release.wait(20)
            return
        if mode == "phase_mismatch":
            # Rank 2 enters the backward rendezvous while its peers are still
            # in the forward one.
            if absent:
                ctx = SimpleNamespace(
                    group=ps.ep_group, input_splits=[1] * 8, output_splits=[1] * 8, ep_sequence=0
                )
                _AllToAll.backward(ctx, x[:1].expand(8, 2))
            else:
                a2a()
        elif mode == "a2a_missing":
            if absent:
                release.wait(20)
                return
            a2a()
        elif mode == "backward_identity":
            # Same shape, different collective: must not pass as participation.
            first, second = a2a(), a2a()
            (first if absent else second).sum().backward()
        elif mode == "backward_missing":
            y = a2a()
            if absent:
                release.wait(20)
                return
            y.sum().backward()
        else:
            if mode == "empty" and absent:
                x = torch.empty(0, 2, requires_grad=True)
                indices = torch.empty(0, 1, dtype=torch.long)
                scores = torch.empty(0, 1, requires_grad=True)
            for _ in range(3):
                y, _, routed = dispatcher.dispatch(x, scores, indices)
                restored = dispatcher.combine(y * routed.unsqueeze(-1))
                torch.testing.assert_close(restored, x, rtol=0, atol=0)
                restored.sum().backward()
                torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)
                x.grad = scores.grad = None
        results.put((rank, "ok", time.monotonic() - start))
    except Exception as exc:
        results.put((rank, str(exc), time.monotonic() - start))
    finally:
        dist.destroy_process_group()


def _run(tmp_path, mode):
    # The full suite may already have used autograd/CUDA: never fork that state.
    ctx = multiprocessing.get_context("spawn")
    results, release = ctx.Queue(), ctx.Event()
    args = (f"file://{tmp_path / 'rendezvous'}", results, release, mode)
    processes = [ctx.Process(target=_worker, args=(rank, *args)) for rank in range(8)]
    try:
        for process in processes:
            process.start()
        return [results.get(timeout=60) for _ in range(7 if mode.endswith("missing") else 8)]
    finally:
        release.set()
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join()


@pytest.mark.parametrize("mode", ["dispatch_missing", "a2a_missing", "backward_missing"])
def test_missing_rank_fails_before_collective(tmp_path, mode):
    rows = _run(tmp_path, mode)
    assert {rank for rank, _, _ in rows} == {0, 1, 3, 4, 5, 6, 7}
    expected, actual = (2, 1) if mode == "backward_missing" else (1, 0)
    for _, message, elapsed in rows:
        assert "EP participation" in message, message
        assert f"rank {ABSENT} expected={expected} actual={actual}" in message, message
        assert "expected participants=8 actual=7" in message, message
        assert elapsed < 15, (message, elapsed)


@pytest.mark.parametrize(
    "mode,phases",
    [
        ("phase_mismatch", ("alltoall.forward", "alltoall.backward")),
        ("backward_identity", ("alltoall.backward",)),
    ],
)
def test_out_of_order_participation_is_rejected(tmp_path, mode, phases):
    for _, message, elapsed in _run(tmp_path, mode):
        assert "EP participation" in message, message
        assert all(phase in message for phase in phases), message
        assert elapsed < 5, (message, elapsed)


@pytest.mark.parametrize("mode", ["ragged", "empty"])
def test_all_ranks_dispatch_and_backward(tmp_path, mode):
    rows = _run(tmp_path, mode)
    assert all(message == "ok" for _, message, _ in rows), rows
