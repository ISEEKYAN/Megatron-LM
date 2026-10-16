# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real EP2/EP4 x expert-DP2 transport and finalization against one global loss."""

import runpy
from datetime import timedelta
from pathlib import Path
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


def _ep_finalize_worker(rank, ep, directory, cp=1, replicas=2, gathered_loss=False):
    # Spawn avoids inheriting a CUDA/autograd thread pool from earlier tests.
    # Each clean process installs the same optional-TE import fixture.
    fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
    import megatron.core.fp8_utils  # noqa: F401
    import megatron.core.transformer.experimental_attention_variant.csa  # noqa: F401
    import megatron.core.transformer.hyper_connection  # noqa: F401

    fixtures['transformer_engine_import_stub'].__wrapped__(pytest.MonkeyPatch())()
    from megatron.lite.model.deepseek_v41.lite import moe as moe_api
    from megatron.lite.primitive.optimizers import headwise_muon
    from megatron.lite.primitive.parallel.state import init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig

    torch.set_num_threads(1)
    world = ep * replicas
    dist.init_process_group(
        'gloo',
        init_method=f'file://{directory}/store',
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=60),
    )
    ps = init_parallel(ParallelConfig(ep=ep, etp=1, cp=cp))
    assert ps.dp_cp_size == world
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
    router = moe_api.ModalityRouter(cfg, ps)
    router.router.gate.weight.data.zero_()
    expert = _LinearExpert(ep)
    modules = [expert if i == ps.ep_rank else None for i in range(ep)]
    model = moe_api.DeepseekV41MoE(router, modules, ps=ps, use_deepep=False)
    # Different source-rank data; every token visits every expert. All reductions
    # are small binary rationals so regrouping does not relax torch.equal.
    x = (torch.eye(ep) * (rank + 1)).requires_grad_()
    scores, indices, _ = router(x)
    assert torch.equal(scores, torch.full_like(scores, 1 / ep))
    assert torch.equal(indices, torch.arange(ep).expand(ep, ep))
    output = model(x)
    if gathered_loss:
        from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence

        # External RL sees a full sequence on each CP peer. Its DP-only loss
        # multiplier is paired with differentiable CP gather's backward SUM.
        partition = ContiguousCPSequence(ep * cp, ps.cp_rank, cp, ps.cp_group)
        output = partition.gather(output, seq_dim=0)
    output.sum().div(ep * (cp if gathered_loss else 1)).backward()
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
    opt = headwise_muon.MixedOptimizer.__new__(headwise_muon.MixedOptimizer)
    opt.host_update = None
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
    # Compare the actual parameter and Adam state after one update. Checking
    # gradients too matters: Adam's first-step ratio can hide a scale error.
    reference = torch.nn.Parameter(torch.eye(ep))
    reference.grad = expected.clone()
    options = dict(lr=0.05, betas=(0.5, 0.75), eps=1e-3, weight_decay=0.1)
    optimizer = torch.optim.AdamW([expert.weight], foreach=False, **options)
    oracle = torch.optim.AdamW([reference], foreach=False, **options)
    optimizer.step()
    oracle.step()
    assert torch.equal(expert.weight, reference)
    for name, value in optimizer.state[expert.weight].items():
        assert torch.equal(value, oracle.state[reference][name]), (rank, name)
    dist.destroy_process_group()


@pytest.mark.parametrize('ep', [2, 4])
def test_ep_finalize_matches_single_global_batch(v41_core_te, tmp_path, ep):
    torch.set_num_threads(1)
    mp.start_processes(
        _ep_finalize_worker,
        args=(ep, str(tmp_path)),
        nprocs=ep * 2,
        join=True,
        start_method='spawn',
    )


@pytest.mark.parametrize(
    "ep,cp,replicas,gathered_loss",
    [(2, 2, 2, False), (8, 8, 1, False), (2, 4, 2, True)],
)
def test_ep_over_cp_update_matches_single_global_batch(
    v41_core_te, tmp_path, ep, cp, replicas, gathered_loss
):
    """Expert replicas share MCore's DP x CP denominator, including CP8/EP8."""
    torch.set_num_threads(1)
    mp.start_processes(
        _ep_finalize_worker,
        args=(ep, str(tmp_path), cp, replicas, gathered_loss),
        nprocs=ep * replicas,
        join=True,
        start_method="spawn",
    )
