# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DP must preserve the global token objective and complete optimizer updates."""


import parallel_test_utils as harness
import pytest
import torch
import torch.distributed as dist
from megatron.lite.model.deepseek_v41.lite import protocol
from parallel_test_utils import assert_exact


def _dp_worker(rank, config, trainable, directory):
    serial, impl = harness._init_parallel_worker(rank, config, trainable)
    with harness.world(rank, directory):
        parallel = protocol.build_model(config, impl_cfg=impl)
        parallel.chunks[0].load_state_dict(serial.chunks[0].state_dict())
        assert parallel.parallel_state.dp_size == 2
        errors = {'gradient_max_abs': 0.0, 'parameter_max_abs': 0.0}
        for step in range(2):
            harness._train_parallel_step(serial, parallel, rank, step)
            for name, p, q in harness.parameter_pairs(
                parallel.chunks[0], serial.chunks[0], strict=True
            ):
                assert (p.grad is None) == (q.grad is None), name
                if p.grad is not None:
                    harness.record_error(errors, 'gradient_max_abs', p.grad, q.grad)
                    # This fixture has an exact serial baseline, including unequal token counts.
                    assert_exact(p.grad, q.grad, msg=name)
            assert serial.optimizer.step()[0]
            assert parallel.optimizer.step()[0]
            for name, p, q in harness.parameter_pairs(
                parallel.chunks[0], serial.chunks[0], strict=True
            ):
                assert_exact(p, q, msg=name)
                harness.record_error(errors, 'parameter_max_abs', p, q)
                replicas = [torch.empty_like(p) for _ in range(2)]
                dist.all_gather(replicas, p)
                assert_exact(*replicas, msg=name)
            harness._check_parallel_buffers(parallel, serial, rank, trainable)
        harness.report_rank(directory, rank, errors)


@pytest.mark.gpus(2)
@pytest.mark.parametrize('trainable', [False, True])
def test_dp_matches_global_batch(model_config, trainable, tmp_path):
    assert torch.cuda.device_count() >= 2, 'DP comparison requires two allocated GPUs'
    harness.run_workers(_dp_worker, (model_config, trainable), tmp_path, report=True)
