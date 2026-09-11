# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""PP/VPP backward semantics against ordinary autograd (Core's non-deallocated path)."""
from types import SimpleNamespace

import pytest
import torch
from megatron.lite.primitive.parallel import pipeline

pytestmark = pytest.mark.mlite


@pytest.mark.parametrize('vpp', [False, True])
@pytest.mark.parametrize(
    'mode', ['loss', 'no_loss', 'constant_loss', 'detached', 'vector']
)
def test_backward_output_semantics(
    monkeypatch, transformer_engine_import_stub, vpp, mode
):
    transformer_engine_import_stub()
    ps = SimpleNamespace(pp_size=2, pp_rank=1, pp_is_first=False, pp_is_last=True)
    inputs, sent_grads = [], []

    class Model(torch.nn.Module):
        def set_input_tensor(self, value):
            self.input = value

    models = [Model() for _ in range(2 if vpp else 1)]

    def communicate(send_fwd, send_bwd, recv_fwd, recv_bwd, *args, **kwargs):
        if send_bwd is not None:
            sent_grads.append(send_bwd.clone())
        value = torch.ones(2, requires_grad=True) if recv_fwd else None
        if value is not None:
            inputs.append(value)
        return value, torch.ones(()) if recv_bwd else None

    def forward(model, batch):
        hidden = (model.input * 3).sum()
        if model is models[-1]:
            if mode == 'vector':
                hidden = model.input * 3
            elif mode == 'detached':
                hidden = hidden.detach()
        out = {'hidden_states': hidden}
        if model is models[-1] and mode in ('loss', 'constant_loss'):
            out['loss'] = hidden * 2 if mode == 'loss' else hidden.detach()
        return out

    monkeypatch.setattr(pipeline, '_send_recv_pipeline', communicate)
    monkeypatch.setattr(pipeline, '_pipeline_stage_barrier', lambda ps: None)
    monkeypatch.setattr(pipeline, '_set_virtual_pipeline_rank', lambda *args: None)
    monkeypatch.setattr(pipeline.dist, 'get_rank', lambda: 1)
    schedule = pipeline._interleaved_1f1b_schedule if vpp else pipeline._1f1b_schedule

    def run():
        return schedule(
            forward,
            models if vpp else models[0],
            iter([{}]),
            1,
            SimpleNamespace(),
            ps,
            (2,),
        )

    if mode == 'vector':
        # Core does not fabricate a gradient for a non-scalar terminal output.
        with pytest.raises(RuntimeError, match='implicitly created only for scalar'):
            run()
        return
    run()
    expected = (
        None
        if mode in ('constant_loss', 'detached')
        else torch.full((2,), 6.0 if mode == 'loss' else 3.0)
    )
    actual = inputs[-1].grad
    assert (actual is None and expected is None) or torch.equal(actual, expected)
    if vpp:
        assert torch.equal(inputs[0].grad, torch.full((2,), 3.0))
    if expected is not None:
        assert any(torch.equal(grad, expected) for grad in sent_grads)


def _ep_worker(rank, rendezvous):
    import importlib.util
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist
    from megatron.lite.primitive.recompute import wrap_checkpoint

    source = (
        Path(__file__).resolve().parents[4] / 'megatron/lite/primitive/modules/moe.py'
    )
    spec = importlib.util.spec_from_file_location('ep_backward_moe', source)
    moe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(moe)
    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo',
        init_method=f'file://{rendezvous}',
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        for checkpoint in (False, True):
            for empty in (False, True):
                splits = ([1, 0] if rank == 0 else [0, 0]) if empty else [1, 1]

                class Layer(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.weight = torch.nn.Parameter(torch.tensor(3.0))
                        self.calls = 0

                    def forward(self, *, hidden):
                        self.calls += 1
                        return moe._AllToAll.apply(
                            hidden * self.weight, splits, splits, dist.group.WORLD
                        ).sum()

                layer = Layer()
                if checkpoint:
                    wrap_checkpoint(layer, preserve_rng_state=False)
                x = torch.ones(sum(splits), requires_grad=True)

                def forward(model, batch):
                    output = model(hidden=x)
                    return {
                        'hidden_states': output,
                        **({'loss': output} if rank == 0 else {}),
                    }

                ps = SimpleNamespace(
                    pp_size=1, pp_rank=0, pp_is_first=True, pp_is_last=True
                )
                # Real EP collectives; this isolates the terminal PP backward boundary.
                original = pipeline._send_recv_pipeline
                pipeline._send_recv_pipeline = lambda *args, **kwargs: (None, None)
                try:
                    pipeline._1f1b_schedule(
                        forward, layer, iter([{}]), 1, SimpleNamespace(), ps, ()
                    )
                finally:
                    pipeline._send_recv_pipeline = original
                assert torch.equal(x.grad, torch.full_like(x, 3.0))
                assert layer.weight.grad.item() == sum(splits)
                assert layer.calls == (2 if checkpoint else 1)
                dist.barrier()
    finally:
        dist.destroy_process_group()


def test_ep_backward_with_missing_local_loss_and_keyword_recompute(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_ep_worker, args=(str(tmp_path / 'rendezvous'),), nprocs=2, join=True)
