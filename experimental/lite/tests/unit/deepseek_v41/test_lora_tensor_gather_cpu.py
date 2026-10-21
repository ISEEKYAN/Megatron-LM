# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real Gloo byte gathering: replicated factors, ragged owners and empty ranks."""
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.optional


def _worker(rank, world, init, mode):
    from megatron.lite.model.deepseek_v41.lite.export import gather_lora_tensors

    dist.init_process_group('gloo', init_method=init, rank=rank, world_size=world)
    try:
        shared = torch.tensor([0.0, -0.0, 1.0, -2.0], dtype=torch.bfloat16).reshape(2, 2)
        local = {'shared': shared.clone()}
        if rank == 0:
            local['owner0'] = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
        if mode == 'empty' and rank != 0:
            local = {}
        if mode == 'corrupt' and rank == 1:
            local['shared'][0, 1] = 0  # +0 and -0 are numerically equal but bitwise different
        if mode == 'corrupt':
            with pytest.raises(ValueError, match='Replicated LoRA factors differ'):
                dict(gather_lora_tensors(local))
        else:
            actual = dict(gather_lora_tensors(local))
            expected = {
                'shared': shared,
                'owner0': torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
            }
            assert actual.keys() == expected.keys()
            for key, value in expected.items():
                assert torch.equal(actual[key].view(torch.uint8), value.view(torch.uint8))
            for value in local.values():
                value.fill_(9)
            assert torch.equal(
                actual['shared'].view(torch.uint8), expected['shared'].view(torch.uint8)
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('mode', ['ragged', 'empty', 'corrupt'])
def test_real_gloo_lora_tensor_gather(tmp_path, mode):
    init = 'file://' + str(Path(tmp_path) / 'gloo')
    world = int(os.environ.get('W10_LORA_TEST_WORLD', '2'))
    mp.spawn(_worker, args=(world, init, mode), nprocs=world, join=True)
