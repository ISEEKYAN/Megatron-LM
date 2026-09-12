# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real eight-process CPU regressions; missing ranks never enter the guard."""

from __future__ import annotations

import multiprocessing
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.mlite


@pytest.fixture(autouse=True)
def _te_import_stub(transformer_engine_import_stub):
    transformer_engine_import_stub()


def _worker(rank, rendezvous, results, release, mode):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=25),
    )
    from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
    from megatron.lite.primitive.modules.moe import _AllToAll
    from megatron.lite.primitive.parallel import ParallelState

    ps = ParallelState(ep_size=8, ep_group=dist.group.WORLD)
    dispatcher = TokenDispatcher(8, 2, ps, use_deepep=False, moe_permute_fusion=False)
    # Rank 2 has the largest input. Missing participation is unrelated to emptiness.
    count = 16 if rank == 2 else rank + 1
    x = torch.full((count, 2), float(rank), requires_grad=True)
    indices = torch.full((count, 1), (rank + 1) % 8, dtype=torch.long)
    scores = torch.ones(count, 1, requires_grad=True)
    dist.barrier()
    start = time.monotonic()
    try:
        if mode == "dispatch_missing" and rank == 2:
            release.wait(20)
            return
        if mode == "phase_mismatch":
            if rank == 2:
                _AllToAll.backward(
                    SimpleNamespace(group=ps.ep_group, input_splits=[1] * 8, output_splits=[1] * 8),
                    x[:1].expand(8, 2),
                )
            else:
                _AllToAll.apply(x[:1].expand(8, 2), [1] * 8, [1] * 8, ps.ep_group)
        elif mode == "a2a_missing":
            if rank == 2:
                release.wait(20)
                return
            _AllToAll.apply(x[:1].expand(8, 2), [1] * 8, [1] * 8, ps.ep_group)
        elif mode == "backward_missing":
            y = _AllToAll.apply(x[:1].expand(8, 2), [1] * 8, [1] * 8, ps.ep_group)
            if rank == 2:
                release.wait(20)
                return
            y.sum().backward()
        else:
            if mode == "empty" and rank == 2:
                x = torch.empty(0, 2, requires_grad=True)
                indices = torch.empty(0, 1, dtype=torch.long)
                scores = torch.empty(0, 1, requires_grad=True)
            for _ in range(3):
                y, _, routed_scores = dispatcher.dispatch(x, scores, indices)
                restored = dispatcher.combine(y * routed_scores.unsqueeze(-1))
                torch.testing.assert_close(restored, x, rtol=0, atol=0)
                restored.sum().backward()
                torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)
                x.grad = None
                scores.grad = None
        results.put((rank, "ok", time.monotonic() - start))
    except Exception as exc:
        results.put((rank, str(exc), time.monotonic() - start))
    finally:
        dist.destroy_process_group()


def _run(tmp_path, mode):
    # fork avoids eight independent TE/CUDA imports; workers only execute CPU ops.
    ctx = multiprocessing.get_context("fork")
    results, release = ctx.Queue(), ctx.Event()
    processes = [
        ctx.Process(
            target=_worker,
            args=(
                rank,
                f"file://{tmp_path / 'rendezvous'}",
                results,
                release,
                mode,
            ),
        )
        for rank in range(8)
    ]
    try:
        for process in processes:
            process.start()
        expected = 7 if mode.endswith("missing") else 8
        rows = [results.get(timeout=18) for _ in range(expected)]
        return rows
    finally:
        release.set()
        for process in processes:
            process.join(timeout=2)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()


@pytest.mark.parametrize("mode", ["dispatch_missing", "a2a_missing", "backward_missing"])
def test_missing_rank_fails_before_collective(tmp_path, mode):
    rows = _run(tmp_path, mode)
    assert {rank for rank, _, _ in rows} == {0, 1, 3, 4, 5, 6, 7}
    for _, message, elapsed in rows:
        assert "EP participation" in message, message
        assert "rank 2" in message, message
        expected, actual = (2, 1) if mode == "backward_missing" else (1, 0)
        assert f"rank 2 expected={expected} actual={actual}" in message, message
        assert "expected participants=8 actual=7" in message, message
        assert elapsed < 15, (message, elapsed)


@pytest.mark.parametrize("mode", ["ragged", "empty"])
def test_all_ranks_dispatch_and_backward(tmp_path, mode):
    rows = _run(tmp_path, mode)
    assert all(message == "ok" for _, message, _ in rows), rows


def test_forward_backward_order_mismatch(tmp_path):
    rows = _run(tmp_path, "phase_mismatch")
    for _, message, elapsed in rows:
        assert "EP participation" in message, message
        assert "alltoall.forward" in message and "alltoall.backward" in message, message
        assert elapsed < 5, (message, elapsed)
