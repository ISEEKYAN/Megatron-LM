# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real PP2 schedules preserve configured FP32 carriers and reverse gradients."""
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist


def _dtype_schedule_worker(rank, directory, explicit_fp32):
    from megatron.lite.primitive.parallel import pipeline as pl

    dist.init_process_group(
        'gloo', init_method=f'file://{directory}/init', rank=rank, world_size=2
    )
    try:
        # Do not override pl._PIPELINE_TENSOR_DTYPE: test the real BF16 default.
        dtype = torch.float32 if explicit_fp32 else torch.bfloat16
        ps = SimpleNamespace(
            pp_size=2,
            pp_rank=rank,
            pp_is_first=rank == 0,
            pp_is_last=rank == 1,
            pp_prev_rank=0,
            pp_next_rank=1,
            pp_group=dist.group.WORLD,
            pp_cpu_group=dist.group.WORLD,
        )
        cfg = SimpleNamespace(num_microbatches=3)
        if explicit_fp32:
            cfg.pipeline_dtype = dtype
        batches = []
        for i, size in enumerate([3, 7, 2]):
            gen = torch.Generator().manual_seed(179 + i)
            batches.append({'value': torch.randn(1, size, 10, generator=gen).to(dtype)})

        class Stage(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.tensor(2.0 if rank == 0 else 3.0, dtype=dtype)
                )
                self.input = None

            def set_input_tensor(self, value):
                self.input = value

        model = Stage()
        observed = []

        def forward(stage, batch):
            value = batch['value'] if rank == 0 else stage.input
            assert value.dtype == dtype
            if rank == 1:
                # Values deliberately include numbers not exactly representable in BF16.
                assert torch.equal(value.detach(), batch['value'] * 2)
            hidden = value * stage.weight
            if rank == 1:
                observed.append(hidden.detach().clone())
                return {'loss': hidden.float().square().sum()}
            return {'hidden_states': hidden}

        with torch.no_grad():
            pl.forward_backward_pipelining(
                forward,
                [model],
                iter(batches),
                cfg,
                ps,
                tensor_shape=(1, 1, 10),
                forward_only=True,
            )
        if rank == 1:
            assert len(observed) == 3
            for result, batch in zip(observed, batches, strict=True):
                assert torch.equal(result, batch['value'] * 2 * 3)
        observed.clear()
        pl.forward_backward_pipelining(
            forward,
            [model],
            iter(batches),
            cfg,
            ps,
            tensor_shape=(1, 1, 10),
            forward_only=False,
        )
        w0 = torch.tensor(2.0, dtype=dtype, requires_grad=True)
        w1 = torch.tensor(3.0, dtype=dtype, requires_grad=True)
        # Match the schedule's per-microbatch backward and accumulation order.
        for batch in batches:
            ((batch['value'] * w0 * w1).float().square().sum() / 3).backward()
        assert torch.equal(model.weight.grad, (w0 if rank == 0 else w1).grad)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('explicit_fp32', [False, True])
def test_real_pp2_schedules_carrier_and_vjp_dtype(tmp_path, explicit_fp32):
    torch.multiprocessing.spawn(
        _dtype_schedule_worker, args=(str(tmp_path), explicit_fp32), nprocs=2, join=True
    )
