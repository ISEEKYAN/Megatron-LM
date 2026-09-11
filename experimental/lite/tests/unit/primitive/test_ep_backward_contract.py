# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real CPU/Gloo contract checks, including empty ranks and omitted backwards."""
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.lite.primitive.parallel import ep_contract
from megatron.lite.primitive.recompute import wrap_checkpoint

pytestmark = pytest.mark.mlite


def _worker(rank, init_file, result_dir):
    # Import the small primitive directly; no TE module is executed.
    import importlib.util

    source = (
        Path(__file__).resolve().parents[3] / 'megatron/lite/primitive/modules/moe.py'
    )
    spec = importlib.util.spec_from_file_location('ep_test_moe', source)
    moe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(moe)
    alltoall = moe._AllToAll
    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo',
        init_method=f'file://{init_file}',
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    group = dist.group.WORLD
    ps = SimpleNamespace(ep_size=2, ep_group=group)
    os.environ['MEGATRON_LITE_VALIDATE_EP_BACKWARD'] = '1'
    communication_calls = []
    original = dist.all_to_all_single

    def counted(*args, **kwargs):
        communication_calls.append('alltoall')
        return original(*args, **kwargs)

    dist.all_to_all_single = counted
    results = []
    try:
        for checkpoint in (False, True):
            for mode in (
                'connected',
                'empty_rank',
                'all_empty',
                'zero_loss',
                'missing_loss',
                'unrelated',
                'partial',
                'all_missing',
                'microbatch_mismatch',
            ):

                class Layer(torch.nn.Module):
                    def forward(self, x):
                        for i in range(3):
                            x = alltoall.apply(x, splits, splits, group)
                            if i == 1 and mode == 'partial' and rank == 1:
                                x = x.detach().requires_grad_()
                        return x

                # Under full recompute, a partial *internal* detach is opaque at
                # this boundary. It is not a supported contract predicate.
                if checkpoint and mode == 'partial':
                    continue
                rows = (
                    0
                    if mode == 'all_empty' or (mode == 'empty_rank' and rank == 1)
                    else 1
                )
                splits = [rows, 0] if rank == 0 else [0, rows]
                x = torch.ones(rows, 3, requires_grad=True)
                layer = Layer()
                if checkpoint:
                    wrap_checkpoint(layer, preserve_rng_state=False)
                communication_calls.clear()
                hidden = layer(x)
                loss = hidden.sum() * (0 if mode == 'zero_loss' else 1)
                if mode == 'all_missing' or (mode == 'missing_loss' and rank == 1):
                    loss = None
                if mode == 'unrelated' and rank == 1:
                    loss = torch.ones((), requires_grad=True)
                expected_error = mode in (
                    'missing_loss',
                    'unrelated',
                    'partial',
                    'all_missing',
                    'microbatch_mismatch',
                )
                sequence = rank if mode == 'microbatch_mismatch' else 0
                try:
                    ep_contract.validate_ep_backward_contract(
                        {'loss': loss}, ps, is_last_stage=True, microbatch=sequence
                    )
                except RuntimeError as error:
                    assert expected_error, str(error)
                    assert 'EP backward contract' in str(error)
                    assert len(communication_calls) == 3  # Before any replay/backward.
                    result = 'rejected_on_all_ranks'
                else:
                    assert not expected_error, (checkpoint, mode)
                    loss.backward()
                    assert len(communication_calls) == (9 if checkpoint else 6)
                    expected_grad = (
                        torch.zeros_like(x)
                        if mode == 'zero_loss'
                        else torch.ones_like(x)
                    )
                    assert torch.equal(x.grad, expected_grad)
                    result = 'backward_completed'
                results.append(dict(checkpoint=checkpoint, mode=mode, result=result))
                dist.barrier()

        for mode in ('nonlast_connected', 'nonlast_detached', 'virtual_stage_mismatch'):
            splits = [1, 0] if rank == 0 else [0, 1]
            x = torch.ones(1, 3, requires_grad=True)
            communication_calls.clear()
            hidden = alltoall.apply(x, splits, splits, group)
            if mode == 'nonlast_detached' and rank == 1:
                hidden = hidden.detach()
            try:
                ep_contract.validate_ep_backward_contract(
                    {'hidden_states': hidden},
                    ps,
                    is_last_stage=False,
                    virtual_stage=rank if mode == 'virtual_stage_mismatch' else 0,
                )
            except RuntimeError as error:
                assert mode != 'nonlast_connected' and 'EP backward contract' in str(
                    error
                )
                assert len(communication_calls) == 1
                result = 'rejected_on_all_ranks'
            else:
                assert mode == 'nonlast_connected'
                hidden.backward(torch.ones_like(hidden))
                assert torch.equal(x.grad, torch.ones_like(x))
                result = 'backward_completed_without_local_loss'
            results.append(dict(mode=mode, result=result))
            dist.barrier()

        # Actual training entrypoints must invoke the check before loss scaling.
        from megatron.lite.primitive.parallel import pipeline
        from megatron.lite.primitive.train_step import run_microbatch_loop

        ps.pp_size = 1
        for entry in ('pipeline', 'microbatch'):
            for loss_mode in ('normal', 'missing', 'unrelated', 'callback_none'):
                inputs = []
                splits = [1, 0] if rank == 0 else [0, 1]

                def forward_step(model, batch):
                    x = torch.ones(1, 3, requires_grad=True)
                    inputs.append(x)
                    hidden = x
                    for _ in range(3):
                        hidden = alltoall.apply(hidden, splits, splits, group)
                    out = {'hidden_states': hidden}
                    if not (rank == 1 and loss_mode == 'missing'):
                        out['loss'] = hidden.sum()
                    return out

                def external_loss(out, batch):
                    if rank == 1 and loss_mode == 'callback_none':
                        return None, {}
                    if rank == 1 and loss_mode == 'unrelated':
                        return torch.ones((), requires_grad=True), {}
                    return out['loss'], {}

                callback = None if loss_mode == 'missing' else external_loss
                communication_calls.clear()
                try:
                    if entry == 'pipeline':
                        pipeline.forward_backward_pipelining(
                            forward_step,
                            [None],
                            iter([None]),
                            SimpleNamespace(num_microbatches=1),
                            ps,
                            loss_fn=callback,
                        )
                    else:
                        run_microbatch_loop(
                            None, iter([None]), 1, forward_step, loss_fn=callback, ps=ps
                        )
                except RuntimeError as error:
                    assert loss_mode != 'normal' and 'EP backward contract' in str(
                        error
                    )
                    assert len(communication_calls) == 3
                    result = 'rejected_on_all_ranks'
                else:
                    assert loss_mode == 'normal'
                    assert len(communication_calls) == 6
                    assert torch.equal(inputs[0].grad, torch.ones_like(inputs[0]))
                    result = 'backward_completed'
                results.append(dict(entry=entry, loss_mode=loss_mode, result=result))
                dist.barrier()

        # Measure the opt-in check on a live CPU graph, separately from forward.
        x = torch.ones(4, requires_grad=True)
        loss = (x.sin() * x).sum()
        times = []
        for step in range(25):
            start = time.perf_counter()
            ep_contract.validate_ep_backward_contract(
                {'loss': loss}, ps, is_last_stage=True, microbatch=step
            )
            times.append((time.perf_counter() - start) * 1e6)
        Path(result_dir, f'{rank}.json').write_text(
            json.dumps(
                dict(rank=rank, cases=results, median_check_us=sorted(times[5:])[10])
            )
        )
    finally:
        dist.all_to_all_single = original
        dist.destroy_process_group()


def test_cpu_gloo_contract_and_real_backward(tmp_path):
    mp.start_processes(
        _worker,
        args=(str(tmp_path / 'init'), str(tmp_path)),
        nprocs=2,
        join=True,
        # The full suite may already have initialized autograd/CUDA threads.
        # Fork inherits that engine state and cannot safely execute backward.
        start_method='spawn',
    )
    rows = [json.loads((tmp_path / f'{rank}.json').read_text()) for rank in range(2)]
    assert rows[0]['cases'] == rows[1]['cases']
    assert len(rows[0]['cases']) == 28
    print('CPU/Gloo check latency (us):', [row['median_check_us'] for row in rows])


def test_disabled_and_single_rank_never_collect(monkeypatch):
    monkeypatch.setattr(
        dist, 'all_reduce', lambda *a, **k: pytest.fail('unexpected collective')
    )
    monkeypatch.delenv('MEGATRON_LITE_VALIDATE_EP_BACKWARD', raising=False)
    ep_contract.validate_ep_backward_contract(
        {}, SimpleNamespace(ep_size=2), is_last_stage=True
    )
    monkeypatch.setenv('MEGATRON_LITE_VALIDATE_EP_BACKWARD', '1')
    ep_contract.validate_ep_backward_contract(
        {}, SimpleNamespace(ep_size=1), is_last_stage=True
    )


def test_forward_only_entrypoints_bypass_contract(monkeypatch):
    from megatron.lite.primitive import train_step
    from megatron.lite.primitive.parallel import pipeline

    monkeypatch.setenv('MEGATRON_LITE_VALIDATE_EP_BACKWARD', '1')

    def unexpected(*args, **kwargs):
        pytest.fail('forward-only must not check backward participation')

    monkeypatch.setattr(ep_contract, 'validate_ep_backward_contract', unexpected)
    ps = SimpleNamespace(ep_size=2, pp_size=1)

    def forward(model, batch):
        return {'hidden_states': torch.ones(2)}

    pipeline.forward_backward_pipelining(
        forward,
        [None],
        iter([None]),
        SimpleNamespace(num_microbatches=1),
        ps,
        forward_only=True,
    )
    train_step.run_microbatch_loop(
        None, iter([None]), 1, forward, forward_only=True, ps=ps
    )
