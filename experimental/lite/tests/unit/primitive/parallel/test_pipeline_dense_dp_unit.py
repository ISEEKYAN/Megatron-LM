# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real DDP/PP schedules average every dense microbatch before global clipping."""
import math
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist


def _dense_dp_worker(rank, directory, pp_size, segmented):
    from megatron.lite.primitive.optimizers.headwise_muon import MixedOptimizer
    from megatron.lite.primitive.parallel import pipeline
    from megatron.lite.primitive.parallel.owned_ddp import wrap_owned_ddp

    dist.init_process_group(
        'gloo',
        init_method=f'file://{directory}/init',
        rank=rank,
        world_size=2 * pp_size,
    )
    try:
        dense_groups = [dist.new_group([2 * i, 2 * i + 1]) for i in range(pp_size)]
        pipeline_groups = [
            dist.new_group([i + 2 * s for s in range(pp_size)]) for i in range(2)
        ]
        stage, dp = rank // 2, rank % 2
        ps = SimpleNamespace(
            pp_size=pp_size,
            pp_rank=stage,
            pp_is_first=stage == 0,
            pp_is_last=stage == pp_size - 1,
            pp_prev_rank=dp,
            pp_next_rank=2 + dp,
            pp_group=pipeline_groups[dp],
            pp_cpu_group=pipeline_groups[dp],
            dp_size=2,
            dp_cp_size=2,
            dp_group=dense_groups[stage],
            cp_size=1,
            ep_size=1,
        )

        class Stage(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.tensor([[2.0 if stage == 0 else 3.0]])
                )
                self.input = None

            def set_input_tensor(self, value):
                self.input = value

            def forward(self, batch):
                value = batch['value'] if stage == 0 else self.input
                hidden = value * self.weight[0, 0]
                if pp_size == 1:
                    hidden = hidden * 3
                return (
                    {'loss': hidden.square().sum()}
                    if ps.pp_is_last
                    else {'hidden_states': hidden}
                )

        model = Stage()
        execution = wrap_owned_ddp(
            model,
            ps,
            optimizing=True,
            external_device=None,
            row_tables=[],
            shard_group=None,
            manual_dense_sync=segmented,
        )
        optimizer = MixedOptimizer(
            model,
            SimpleNamespace(clip_grad=0.25, lr=0.01, segmented_host=segmented),
            group_builder=lambda: [{'algorithm': 'adamw', 'params': [model.weight]}],
            owners=lambda: ([], [], [], None),
            stats_factory=None,
            dp_group=dense_groups[stage],
            ps=ps,
        )
        batches = [
            {'value': torch.full((1, 2, 3), float(i + 1 + dp))} for i in range(3)
        ]
        pipeline.forward_backward_pipelining(
            lambda module, batch: execution(batch),
            [model],
            iter(batches),
            SimpleNamespace(num_microbatches=3, pipeline_dtype=torch.float32),
            ps,
            tensor_shape=(1, 1, 3),
        )
        optimizer.finalize_grads()
        # Independent full-stage reference: average both replicas' distinct data.
        w0 = torch.tensor(2.0, requires_grad=True)
        w1 = torch.tensor(3.0, requires_grad=pp_size == 2)
        for replica in range(2):
            for microbatch in range(3):
                value = torch.full((1, 2, 3), float(microbatch + 1 + replica))
                ((value * w0 * w1).square().sum() / 6).backward()
        expected = w0.grad if stage == 0 else w1.grad
        assert torch.equal(model.weight.grad.flatten(), expected.reshape(1))
        norm = float(optimizer._grad_norm([model.weight], [model.weight.grad]))
        expected_norm = math.sqrt(
            float(w0.grad.double().square())
            + (float(w1.grad.double().square()) if pp_size == 2 else 0)
        )
        assert norm == expected_norm
        norms = [None] * (2 * pp_size)
        dist.all_gather_object(norms, norm)
        assert len(set(norms)) == 1
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('pp_size', [1, 2])
@pytest.mark.parametrize('segmented', [False, True])
def test_dense_dp_all_microbatches_and_global_clip(tmp_path, pp_size, segmented):
    torch.multiprocessing.spawn(
        _dense_dp_worker,
        args=(str(tmp_path), pp_size, segmented),
        nprocs=2 * pp_size,
        join=True,
    )
