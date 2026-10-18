# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CP with an external (VERL-style) loss_fn, as the RL engine runs it.

The engine passes its own loss for both the actor update and the log-prob pass,
so the runtime skips ``prepare_microbatches`` and marks the loss contexts
``external_normalization``; V4.1 then returns CP-sliced log-probs without its
native loss. The engine-shaped loss gathers them back to whole sequences with
the protocol's ``unpack_forward_output``.

PP1 runs the runtime's own microbatch loop; PP2 runs the real 1F1B and
forward-only schedules on contexts marked as the runtime's PP branch marks them.
The whole-sequence log-probs of CP2 (PP1 and PP2) must equal CP1's bitwise;
gradients must be finite.
"""

import json
import runpy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_parity = runpy.run_path(str(Path(__file__).with_name('test_cp_ep_parity.py')))
SEQ_LENS = (9, 7)
CP = 2


def _batch():
    from megatron.lite.runtime.contracts import PackedBatch

    generator = torch.Generator().manual_seed(7)
    ids = torch.randint(1, 64, (sum(SEQ_LENS),), generator=generator)
    mask = torch.ones(ids.shape, dtype=torch.float32)
    mask[0] = 0
    return PackedBatch(
        input_ids=ids,
        labels=ids.clone(),
        loss_mask=mask,
        seq_lens=torch.tensor(SEQ_LENS, dtype=torch.int64),
    )


def _engine_loss(model, gathered):
    from megatron.lite.model.deepseek_v41.lite import protocol

    def loss_fn(output, batch, context):
        # Under CP the model skips its native loss for an external loss_fn.
        assert context.external_normalization
        assert ('loss' in output) == (model.ps.cp_size == 1)
        log_probs = protocol.unpack_forward_output(model, batch, output['log_probs'])
        values = torch.cat(list(log_probs.unbind()))
        gathered.append(values.detach().clone())
        return -(values * batch.loss_mask).sum() / batch.loss_mask.sum(), {}

    return loss_fn


def _steps(bundle, pp):
    """Log-prob pass then actor update; returns {phase: whole-sequence LP}."""
    from megatron.lite.primitive.parallel.pipeline import forward_backward_pipelining
    from megatron.lite.primitive.train_step import run_microbatch_loop
    from megatron.lite.runtime.backends.mlite.runtime import _pipeline_callbacks
    from megatron.lite.runtime.contracts.loss import (
        LossContext,
        mark_external_normalization,
    )

    model, ps = bundle.chunks[0], bundle.parallel_state
    result = {}
    for phase, forward_only in (('infer', True), ('update', False)):
        model.zero_grad(set_to_none=True)
        gathered = []
        items = iter([(_batch(), LossContext())])
        with torch.no_grad() if forward_only else torch.enable_grad():
            if pp == 1:
                run_microbatch_loop(
                    model,
                    items,
                    1,
                    bundle.forward_step,
                    prepare_microbatches=bundle.extras['prepare_microbatches'],
                    loss_fn=_engine_loss(model, gathered),
                    forward_only=forward_only,
                )
            else:
                # The runtime's PP branch: marked contexts, wrapped callbacks.
                forward_step, loss_fn = _pipeline_callbacks(
                    bundle.forward_step, _engine_loss(model, gathered)
                )
                forward_backward_pipelining(
                    forward_step,
                    [model],
                    mark_external_normalization(items),
                    SimpleNamespace(num_microbatches=1, pipeline_dtype=torch.float32),
                    ps,
                    tensor_shape=(1, 1, 1),
                    loss_fn=loss_fn,
                    forward_only=forward_only,
                )
        assert len(gathered) == (1 if ps.pp_is_last else 0)
        if gathered:
            result[phase] = gathered[0]
    return result


def _worker(rank, directory, pp, cp):
    fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
    import megatron.core.fp8_utils  # noqa: F401
    import megatron.core.transformer.experimental_attention_variant.csa  # noqa: F401
    import megatron.core.transformer.hyper_connection  # noqa: F401

    fixtures['transformer_engine_import_stub'].__wrapped__(pytest.MonkeyPatch())()
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.contracts import ParallelConfig

    torch.set_num_threads(1)
    _parity['_patch_gloo_all_gather_backward']()
    dist.init_process_group(
        'gloo',
        init_method=f'file://{directory}/store-{pp}-{cp}',
        rank=rank,
        world_size=pp * cp,
        timeout=timedelta(seconds=300),
    )
    try:
        state = torch.load(f'{directory}/state.pt', weights_only=False)
        impl = _parity['_impl'](ParallelConfig(pp=pp, cp=cp))
        if pp > 1:
            impl = replace(impl, pipeline_split_layer=2)
        bundle = protocol.build_model(_parity['_prefix_config'](), impl_cfg=impl)
        model, ps = bundle.chunks[0], bundle.parallel_state
        own = model.state_dict()
        model.load_state_dict({k: v for k, v in state.items() if k in own})
        saved = {'last': ps.pp_is_last, 'log_probs': _steps(bundle, pp)}
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        saved['grads'] = len(grads)
        saved['finite'] = all(bool(torch.isfinite(g).all()) for g in grads)
        torch.save(saved, f'{directory}/pp{pp}cp{cp}-{rank}.pt')
    finally:
        dist.destroy_process_group()


def _run(directory, pp, cp):
    mp.start_processes(
        _worker,
        args=(str(directory), pp, cp),
        nprocs=pp * cp,
        join=True,
        start_method='spawn',
    )
    return [
        torch.load(directory / f'pp{pp}cp{cp}-{rank}.pt', weights_only=False)
        for rank in range(pp * cp)
    ]


def test_cp2_external_loss_log_probs(v41_core_te, tmp_path):
    from megatron.lite.model.deepseek_v41.lite import protocol

    torch.set_num_threads(1)
    torch.manual_seed(351)
    bundle = protocol.build_model(
        _parity['_prefix_config'](), impl_cfg=_parity['_impl']()
    )
    torch.save(
        {k: v.detach().clone() for k, v in bundle.chunks[0].state_dict().items()},
        tmp_path / 'state.pt',
    )
    cp1 = _steps(bundle, 1)
    pp1 = _run(tmp_path, 1, CP)
    staged = _run(tmp_path, 2, CP)
    pp2 = [r for r in staged if r['last']]
    assert len(pp1) == len(pp2) == CP
    report = {}
    for rank, (flat, last) in enumerate(zip(pp1, pp2)):
        for phase, wanted in cp1.items():
            got = flat['log_probs'][phase]
            assert got.shape == wanted.shape == (sum(SEQ_LENS),)
            report[f'{rank}/{phase}'] = dict(
                cp2_vs_cp1=float((got - wanted).abs().max()),
                cp2_equals_cp1=torch.equal(got, wanted),
                pp2_equals_pp1=torch.equal(last['log_probs'][phase], got),
            )
    print('CP_EXTERNAL_LOSS', json.dumps(report))
    assert all(
        r['cp2_equals_cp1'] and r['pp2_equals_pp1'] for r in report.values()
    ), report
    assert all(r['grads'] > 0 and r['finite'] for r in pp1 + staged)
