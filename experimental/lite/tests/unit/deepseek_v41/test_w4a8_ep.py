# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real EP transport against the unpartitioned W4A8 bank, including VJPs."""
import copy
import runpy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class _Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Linear(128, 128, bias=False)
        self.w2 = torch.nn.Linear(128, 128, bias=False)
        self.w3 = torch.nn.Linear(128, 128, bias=False)
        self.swiglu_limit = 10.0


def _worker(rank, ep, directory):
    fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
    import megatron.core.fp8_utils  # noqa: F401
    import megatron.core.transformer.experimental_attention_variant.csa  # noqa: F401
    import megatron.core.transformer.hyper_connection  # noqa: F401

    fixtures['transformer_engine_import_stub'].__wrapped__(pytest.MonkeyPatch())()
    import megatron.lite.model.deepseek_v41.lite.moe as _imports_moe

    DeepseekV41MoE = _imports_moe.DeepseekV41MoE
    ModalityRouter = _imports_moe.ModalityRouter
    import megatron.lite.primitive.parallel.state as _imports_state

    ParallelState = _imports_state.ParallelState
    init_parallel = _imports_state.init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig

    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo',
        init_method=f'file://{directory}/store',
        rank=rank,
        world_size=ep,
        timeout=timedelta(seconds=90),
    )
    ps = init_parallel(ParallelConfig(ep=ep, etp=1))
    cfg = NS(
        n_routed_experts=8,
        num_experts_per_tok=3,
        hidden_size=128,
        norm_topk_prob=True,
        topk_method='noaux_tc',
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
    )
    torch.manual_seed(903)
    router = ModalityRouter(cfg, ps)
    # Route only to three experts: some ranks own zero-token experts. All
    # parameters remain live, and expert IDs stay global on every rank.
    router.router.gate.weight.data.zero_()
    router.router.gate.weight.data[:3].fill_(0.1)
    bank = [_Expert() for _ in range(8)]
    baseline = DeepseekV41MoE(
        copy.deepcopy(router), bank, ps=ParallelState(), w4a8=True
    )
    owned = [
        copy.deepcopy(e) if rank * (8 // ep) <= i < (rank + 1) * (8 // ep) else None
        for i, e in enumerate(bank)
    ]
    actual = DeepseekV41MoE(copy.deepcopy(router), owned, ps=ps, w4a8=True)
    torch.manual_seed(731)
    data = [torch.randn(r + 1, 128).bfloat16() for r in range(ep)]
    incoming = [torch.randn_like(x) for x in data]
    x = data[rank].clone().requires_grad_()
    local_x = data[rank].clone().requires_grad_()
    expected = baseline(local_x)
    result = actual(x)
    assert torch.equal(result, expected), (rank, 'forward')
    result.backward(incoming[rank])
    expected.backward(incoming[rank])
    assert torch.equal(x.grad, local_x.grad), (rank, 'input VJP')
    torch.testing.assert_close(
        actual.gate.router.gate.weight.grad,
        baseline.gate.router.gate.weight.grad,
        rtol=0,
        atol=0,
    )
    baseline.zero_grad(set_to_none=True)
    baseline(torch.cat(data)).backward(torch.cat(incoming))
    for index, expert in enumerate(actual.experts):
        if expert is None:
            continue
        reference = bank[index]
        for name in ('w1', 'w2', 'w3'):
            grad = getattr(expert, name).weight.grad
            target = getattr(reference, name).weight.grad
            assert grad is not None and grad.dtype == torch.float32
            torch.testing.assert_close(grad, target, rtol=2e-5, atol=2e-5)
    dist.destroy_process_group()


@pytest.mark.parametrize('ep', [4, 8])
def test_w4a8_ep_global_slots_forward_and_vjp(v41_core_te, tmp_path, monkeypatch, ep):
    monkeypatch.setenv('MEGATRON_LITE_MOE_PERMUTE_FUSION', '0')
    mp.start_processes(
        _worker, args=(ep, str(tmp_path)), nprocs=ep, join=True, start_method='spawn'
    )
