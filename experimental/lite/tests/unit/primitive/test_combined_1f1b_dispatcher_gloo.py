# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real two-rank gloo proof for the split all-to-all dispatcher path."""

import multiprocessing
import os
import sys
import types

import pytest
import torch
import torch.distributed as dist


def _install_te_import_stub():
    """Provide import-only TE symbols; this test exercises the unfused path."""

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("fused Transformer Engine path was unexpectedly used")

    root = types.ModuleType("transformer_engine")
    pytorch = types.ModuleType("transformer_engine.pytorch")
    cpp_extensions = types.ModuleType("transformer_engine.pytorch.cpp_extensions")
    module = types.ModuleType("transformer_engine.pytorch.module")
    module_base = types.ModuleType("transformer_engine.pytorch.module.base")
    permutation = types.ModuleType("transformer_engine.pytorch.permutation")
    router = types.ModuleType("transformer_engine.pytorch.router")
    cpp_extensions.general_gemm = unavailable
    module_base.get_workspace = unavailable
    module.base = module_base
    permutation.moe_permute = unavailable
    permutation.moe_permute_and_pad_with_probs = unavailable
    permutation.moe_permute_with_probs = unavailable
    permutation.moe_unpermute = unavailable
    router.fused_compute_score_for_moe_aux_loss = unavailable
    router.fused_moe_aux_loss = unavailable
    router.fused_topk_with_score_function = unavailable
    root.pytorch = pytorch
    for name, value in {
        "transformer_engine": root,
        "transformer_engine.pytorch": pytorch,
        "transformer_engine.pytorch.cpp_extensions": cpp_extensions,
        "transformer_engine.pytorch.module": module,
        "transformer_engine.pytorch.module.base": module_base,
        "transformer_engine.pytorch.permutation": permutation,
        "transformer_engine.pytorch.router": router,
    }.items():
        sys.modules[name] = value


def _dispatcher_worker(rank, world_size, port, results):
    _install_te_import_stub()
    from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
    from megatron.lite.primitive.parallel.state import ParallelState

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MEGATRON_LITE_MOE_PERMUTE_FUSION="0",
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        ps = ParallelState(
            ep_group=dist.group.WORLD, ep_size=world_size, ep_rank=rank
        )
        indices_by_rank = (
            torch.tensor([[0], [0], [2]]),
            torch.tensor([[1], [3], [3]]),
        )
        hidden = (
            torch.arange(6, dtype=torch.float32).reshape(3, 2) + rank * 10
        )
        scores = torch.tensor([[0.5], [0.75], [1.0]])

        baseline_hidden = hidden.clone().requires_grad_(True)
        baseline_scores = scores.clone().requires_grad_(True)
        baseline = TokenDispatcher(
            num_experts=4, hidden_size=2, ps=ps, use_deepep=False
        )
        dispatched, baseline_tpe, baseline_probs = baseline.dispatch(
            baseline_hidden, baseline_scores, indices_by_rank[rank]
        )
        baseline_output = baseline.combine(
            dispatched * baseline_probs.unsqueeze(-1)
        )
        baseline_output.sum().backward()

        split_hidden = hidden.clone().requires_grad_(True)
        split_scores = scores.clone().requires_grad_(True)
        split = TokenDispatcher(
            num_experts=4, hidden_size=2, ps=ps, use_deepep=False
        )
        permuted, permuted_probs, tokens_per_expert = (
            split.alltoall_dispatch_preprocess(
                split_hidden, split_scores, indices_by_rank[rank]
            )
        )
        dispatched, split_tpe, split_probs = split.alltoall_dispatch_communicate(
            permuted, permuted_probs, tokens_per_expert
        )
        split_output = split.alltoall_combine_communicate(
            dispatched * split_probs.unsqueeze(-1)
        )
        split_output.sum().backward()

        results.append(
            (
                rank,
                bool(torch.equal(split_tpe, baseline_tpe)),
                bool(torch.equal(split_output, baseline_output)),
                bool(torch.equal(split_hidden.grad, baseline_hidden.grad)),
                bool(torch.equal(split_scores.grad, baseline_scores.grad)),
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_split_alltoall_matches_existing_dispatcher_with_zero_expert_splits():
    import torch.multiprocessing as mp

    results = multiprocessing.Manager().list()
    mp.spawn(_dispatcher_worker, args=(2, 29675, results), nprocs=2, join=True)

    assert sorted(results) == [
        (0, True, True, True, True),
        (1, True, True, True, True),
    ]
