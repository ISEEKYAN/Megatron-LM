# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real EP2/EP4 x expert-DP2 transport and finalization against one global loss."""

from datetime import timedelta
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class _LinearExpert(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(width))
        self.weight.main_grad = None

    def forward(self, x, *, weights):
        return torch.nn.functional.linear(x, self.weight) * weights


def _ep_finalize_worker(rank, ep, directory):
    from megatron.lite.model.deepseek_v41.lite.moe import DeepseekV41MoE, ModalityRouter
    from megatron.lite.primitive.optimizers.headwise_muon import MixedOptimizer
    from megatron.lite.primitive.parallel.state import init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig

    torch.set_num_threads(1)
    world = ep * 2
    dist.init_process_group(
        'gloo',
        init_method=f'file://{directory}/store',
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=60),
    )
    ps = init_parallel(ParallelConfig(ep=ep, etp=1))
    cfg = NS(
        n_routed_experts=ep,
        num_experts_per_tok=ep,
        hidden_size=ep,
        norm_topk_prob=True,
        topk_method='noaux_tc',
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
    )
    router = ModalityRouter(cfg, ps)
    router.router.gate.weight.data.zero_()
    expert = _LinearExpert(ep)
    modules = [expert if i == ps.ep_rank else None for i in range(ep)]
    model = DeepseekV41MoE(router, modules, ps=ps, use_deepep=False)
    # Different source-rank data; every token visits every expert. All reductions
    # are small binary rationals so regrouping does not relax torch.equal.
    x = (torch.eye(ep) * (rank + 1)).requires_grad_()
    scores, indices, _ = router(x)
    assert torch.equal(scores, torch.full_like(scores, 1 / ep))
    assert torch.equal(indices, torch.arange(ep).expand(ep, ep))
    model(x).sum().div(ep).backward()
    # Before finalize this is a SUM of EP source losses, not their mean.
    source_sum = sum(
        range(ps.expert_dp_rank * ep + 1, (ps.expert_dp_rank + 1) * ep + 1)
    )
    assert torch.equal(
        expert.weight.grad, torch.full_like(expert.weight, source_sum / ep**2)
    )
    # Independent global objective: one unpartitioned linear expert bank over
    # all source tokens; no production routing, loss helper or finalizer.
    global_x = torch.cat([torch.eye(ep) * (r + 1) for r in range(world)])
    bank = torch.eye(ep).repeat(ep, 1, 1).requires_grad_()
    y = torch.einsum("eoi,ti->eto", bank, global_x).sum(0) / ep
    y.sum().div(global_x.shape[0]).backward()
    expected = bank.grad[ps.ep_rank]
    opt = MixedOptimizer.__new__(MixedOptimizer)
    opt.ps, opt.expert_parameters, opt.row_parameters = ps, [expert.weight], []
    calls, reduce = [], dist.all_reduce

    def traced(value, *args, **kwargs):
        calls.append(dist.get_process_group_ranks(kwargs['group']))
        return reduce(value, *args, **kwargs)

    with patch.object(dist, 'all_reduce', traced):
        opt.finalize_grads()
    actual = expert.weight.grad.clone()
    assert torch.equal(actual, expected), (rank, actual, expected)
    assert calls == [dist.get_process_group_ranks(ps.ep_dp_group)] * 2
    # EP dispatch accumulates local-loss contributions from every source rank;
    # a replica-only mean retains an extra EP factor, unlike the global batch.
    assert not torch.equal(actual * ep, expected)
    dist.destroy_process_group()


@pytest.mark.parametrize('ep', [2, 4])
def test_ep_finalize_matches_single_global_batch(v41_core_te, tmp_path, ep):
    # Import before fork: the CPU-only worker inherits the optional-TE fixture.
    from megatron.lite.model.deepseek_v41.lite import moe  # noqa: F401
    from megatron.lite.primitive.optimizers import headwise_muon  # noqa: F401

    torch.set_num_threads(1)
    mp.start_processes(
        _ep_finalize_worker,
        args=(ep, str(tmp_path)),
        nprocs=ep * 2,
        join=True,
        start_method='fork',
    )
