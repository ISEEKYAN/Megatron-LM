"""CPU contract for the MCore ShardedTensor metadata used by the shared FC bank."""

from __future__ import annotations

import torch

from megatron.core.dist_checkpointing.mapping import ShardedTensor


def test_unsharded_sharded_tensor_surface_has_no_slice_metadata():
    data = torch.empty((2, 32, 64), dtype=torch.bfloat16)
    tensor = ShardedTensor.from_rank_offsets("shared_fc_bank", data, replica_id=(0, 0, 0))

    assert tensor.global_shape == tensor.local_shape == (2, 32, 64)
    assert tensor.global_offset == (0, 0, 0)
    assert tensor.axis_fragmentations == (1, 1, 1)
    assert tensor.flattened_range is None
