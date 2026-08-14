# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from megatron.lite.primitive.optimizers.fsdp2.optimizer import (
    _materialize_deferred_parameters,
    defer_large_parameters,
)


class MixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.large = nn.Parameter(torch.ones(8))
        self.small = nn.Parameter(torch.full((2,), 3.0))
        self.register_buffer("table", torch.full((2,), 5.0))


def test_defer_large_parameters_keeps_small_state_initialized() -> None:
    model = defer_large_parameters(
        MixedModel().to(torch.bfloat16), threshold=4, device="cpu"
    )

    assert model.large.is_meta
    assert model.large.dtype == torch.bfloat16
    assert model.small.device.type == "cpu"
    assert torch.equal(model.small, torch.full((2,), 3.0, dtype=torch.bfloat16))
    assert torch.equal(model.table, torch.full((2,), 5.0))


def test_defer_large_parameters_disabled_is_the_eager_path() -> None:
    model = defer_large_parameters(
        MixedModel().to(torch.bfloat16), threshold=None, device="cpu"
    )

    assert all(not param.is_meta for param in model.parameters())
    assert {param.dtype for param in model.parameters()} == {torch.bfloat16}


def test_materialize_changes_only_deferred_fsdp_shards(tmp_path) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        model = defer_large_parameters(
            MixedModel().to(torch.bfloat16), threshold=4, device="cpu"
        )
        fully_shard(model, mesh=init_device_mesh("cpu", (1,)))
        _materialize_deferred_parameters(model, device="cpu")

        assert model.large.to_local().device.type == "cpu"
        assert model.large._mlite_deferred is True
        assert torch.equal(model.small.full_tensor(), torch.full((2,), 3.0).bfloat16())
        assert torch.equal(model.table, torch.full((2,), 5.0))
    finally:
        dist.destroy_process_group()
