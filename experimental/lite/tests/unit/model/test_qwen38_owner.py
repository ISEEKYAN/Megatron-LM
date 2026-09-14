"""Real two-process owner lookup against an independently built full table."""

from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.lite.model.qwen3_8_flash_next.engram import (
    Qwen3_8_FlashNextEngramTableConfig,
)


def owner_cases():
    # Uneven request counts, duplicate rows, both owners, and empty peers.
    return [([7, 0, 7, 4, 1], [2, 5, 2]), ([], [0, 0, 7]), ([], [])]


def _worker(rank, store):
    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo', init_method=f'file://{store}', rank=rank, world_size=2
    )
    try:
        for requests in owner_cases():
            table = Qwen3_8_FlashNextEngramTableConfig(8, 3).build(
                process_group=dist.group.WORLD, device='cpu', dtype=torch.float32
            )
            full = (torch.arange(24).reshape(8, 3).float() / 8).requires_grad_()
            with torch.no_grad():
                table.weight.copy_(full[rank * 4 : (rank + 1) * 4])
            ids = torch.tensor(requests[rank], dtype=torch.int64)
            output = table(ids)
            assert torch.equal(output, full[ids]), 'PLE_OWNER_FORWARD_BITWISE'
            gradients = [
                torch.arange(len(part) * 3).reshape(-1, 3).float() / 8 + r
                for r, part in enumerate(requests)
            ]
            output.backward(gradients[rank])
            serial = torch.nn.functional.embedding(
                torch.tensor(requests[0] + requests[1], dtype=torch.int64), full
            )
            serial.backward(torch.cat(gradients))
            assert torch.equal(
                table.weight.grad, full.grad[rank * 4 : (rank + 1) * 4]
            ), 'PLE_OWNER_GRAD_BITWISE'
            assert table.weight.shape == (4, 3), 'PLE_OWNER_LOCAL_ALLOCATION'
        # An invalid request on only one rank must reject on both before a2a.
        with pytest.raises(ValueError, match='PLE_OWNER_GLOBAL_IDS'):
            table(torch.tensor([-1] if rank else [0]))
    finally:
        dist.destroy_process_group()


def test_owner_distributed_forward_backward(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / 'store'),), nprocs=2, join=True)


def test_official_owner_allocation_is_local_without_allocating_release_table():
    with patch.object(dist, 'get_world_size', return_value=64), patch.object(
        dist, 'get_rank', return_value=63
    ):
        table = Qwen3_8_FlashNextEngramTableConfig(320001536, 160).build(
            process_group=object(), device='meta', dtype=torch.bfloat16
        )
    assert table.weight.shape == (5000024, 160), 'PLE_OWNER_RELEASE_LOCAL_SHAPE'
    assert table.global_row_end == 320001536, 'PLE_OWNER_RELEASE_LAST_BOUNDARY'


def test_owner_placement_is_explicit_and_uses_native_expert_axis():
    from megatron.lite.model.qwen3_8_flash_next.protocol import (
        is_expert_param,
        parameter_placements,
    )

    name = 'layers.1.ple.ple_embedding.ngram_embedding.weight'
    assert not is_expert_param(name)
    assert is_expert_param(name, ple_owner_sharding=True), 'PLE_OWNER_OPTIMIZER_GROUP'
    placement = parameter_placements(name, ple_owner_sharding=True)
    assert placement[2].is_shard(0), 'PLE_OWNER_CHECKPOINT_ROWS'
