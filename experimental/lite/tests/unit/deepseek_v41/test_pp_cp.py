# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""PP x CP: pipeline stages over a contiguous CP slice on Gloo ranks.

The 4-layer prefix of test_cp_ep_parity is cut at its owner-safe layer 2. Each
CP rank of PP2 x CP2 (four ranks) runs the real 1F1B schedule on its slice of
the same packed batch; the last stage's logits must equal, bitwise, those of
the same CP rank under PP1 x CP2 (two ranks). PP changes neither the CP rows
nor their arithmetic, so the comparison is exact; backward must only run and
give finite gradients.
"""

import runpy
from dataclasses import replace
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_parity = runpy.run_path(str(Path(__file__).with_name('test_cp_ep_parity.py')))
_prefix_config, _impl, _inputs = (
    _parity['_prefix_config'],
    _parity['_impl'],
    _parity['_inputs'],
)
CP = 2


def _worker(rank, directory, pp, recompute):
    fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
    import megatron.core.fp8_utils  # noqa: F401
    import megatron.core.transformer.experimental_attention_variant.csa  # noqa: F401
    import megatron.core.transformer.hyper_connection  # noqa: F401

    fixtures['transformer_engine_import_stub'].__wrapped__(pytest.MonkeyPatch())()
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.parallel.pipeline import forward_backward_pipelining
    from megatron.lite.runtime.contracts import ParallelConfig
    from megatron.lite.runtime.contracts.data import PackedBatch

    torch.set_num_threads(1)
    _parity['_patch_gloo_all_gather_backward']()
    dist.init_process_group(
        'gloo',
        init_method=f'file://{directory}/store-{pp}-{recompute}',
        rank=rank,
        world_size=pp * CP,
        timeout=timedelta(seconds=300),
    )
    try:
        state = torch.load(f'{directory}/state.pt', weights_only=False)
        impl = _impl(ParallelConfig(pp=pp, cp=CP), recompute=recompute)
        if pp > 1:
            impl = replace(impl, pipeline_split_layer=2)
        bundle = protocol.build_model(_prefix_config(), impl_cfg=impl)
        model, ps = bundle.chunks[0], bundle.parallel_state
        assert (ps.pp_size, ps.cp_size) == (pp, CP)
        own = model.state_dict()
        model.load_state_dict({k: v for k, v in state.items() if k in own})
        ids, cu = _inputs()
        batch = PackedBatch(ids[0], labels=None, seq_lens=torch.diff(cu).to(torch.long))
        saved = {'cp_rank': ps.cp_rank, 'last': ps.pp_is_last, 'logits': {}}
        for forward_only in (True, False):
            model.zero_grad(set_to_none=True)
            logits = []

            def loss_fn(out, batch):
                logits.append(out['logits'].detach().clone())
                return out['logits'].square().mean(), {}

            context = torch.no_grad() if forward_only else torch.enable_grad()
            with context:
                forward_backward_pipelining(
                    partial(protocol._forward_step),
                    [model],
                    iter([batch]),
                    SimpleNamespace(num_microbatches=1, pipeline_dtype=torch.float32),
                    ps,
                    # Nominal only: 1F1B exchanges each carrier's true shape.
                    tensor_shape=(1, 1, 1),
                    loss_fn=loss_fn,
                    forward_only=forward_only,
                )
            assert len(logits) == (1 if ps.pp_is_last else 0)
            phase = 'forward-only' if forward_only else 'grad-enabled'
            if logits:
                saved['logits'][phase] = logits[0]
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        saved['grads'] = len(grads)
        saved['finite'] = all(torch.isfinite(g).all() for g in grads)
        torch.save(saved, f'{directory}/pp{pp}-{recompute}-{rank}.pt')
    finally:
        dist.destroy_process_group()


def _run(directory, pp, recompute=None):
    mp.start_processes(
        _worker,
        args=(str(directory), pp, recompute),
        nprocs=pp * CP,
        join=True,
        start_method='spawn',
    )
    return [
        torch.load(directory / f'pp{pp}-{recompute}-{rank}.pt', weights_only=False)
        for rank in range(pp * CP)
    ]


@pytest.mark.parametrize('recompute', [None, 'full', 'moe,attn'])
def test_pp2_over_cp2_matches_pp1_cp_rows(v41_core_te, tmp_path, recompute):
    from megatron.lite.model.deepseek_v41.lite import protocol

    torch.set_num_threads(1)
    torch.manual_seed(351)
    state = protocol.build_model(_prefix_config(), impl_cfg=_impl()).chunks[0]
    torch.save(
        {k: v.detach().clone() for k, v in state.state_dict().items()},
        tmp_path / 'state.pt',
    )
    reference = {r['cp_rank']: r for r in _run(tmp_path, 1)}
    staged = _run(tmp_path, 2, recompute)
    last = [r for r in staged if r['last']]
    assert sorted(r['cp_rank'] for r in last) == list(range(CP))
    for record in last:
        wanted = reference[record['cp_rank']]['logits']
        for phase in ('forward-only', 'grad-enabled'):
            assert torch.equal(record['logits'][phase], wanted[phase]), (
                record['cp_rank'],
                phase,
            )
    assert all(r['grads'] > 0 and r['finite'] for r in staged)
