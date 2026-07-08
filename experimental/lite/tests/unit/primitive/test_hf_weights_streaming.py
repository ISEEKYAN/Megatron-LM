# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import importlib.util
import sys
import types

import torch
import torch.nn as nn
import pytest

if importlib.util.find_spec("safetensors") is None:
    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = None
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.save_file = None
    sys.modules["safetensors"] = safetensors
    sys.modules["safetensors.torch"] = safetensors_torch

from megatron.lite.primitive.ckpt.hf_weights import (
    _iter_bucketed_materialized_tensors,
    bucketed_all_gather_into_tensor,
    export_hf_weights,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_resident_export_is_bitwise_equal_to_legacy_cpu_export() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(
                torch.arange(12, dtype=torch.bfloat16, device="cuda").reshape(3, 4)
            )

    class Spec:
        num_experts = 0

        @staticmethod
        def is_expert(name):
            return False

        @staticmethod
        def tp_spec(name):
            return None

        @staticmethod
        def native_to_hf(name, tensor):
            return [(name, tensor)]

    ps = type(
        "ParallelState",
        (),
        {
            "pp_size": 1,
            "tp_size": 1,
            "tp_group": None,
            "ep_size": 1,
            "ep_group": None,
            "etp_size": 1,
            "etp_group": None,
        },
    )()
    model = Model()

    legacy = dict(export_hf_weights(model, Spec(), ps, cpu=True))
    resident = dict(export_hf_weights(model, Spec(), ps, cpu=False))

    assert legacy.keys() == resident.keys()
    assert legacy["weight"].device.type == "cpu"
    assert resident["weight"].device.type == "cuda"
    assert torch.equal(legacy["weight"], resident["weight"].cpu())


def test_bucketed_all_gather_uses_bounded_flat_buffers(monkeypatch) -> None:
    bucket = [
        ("first", torch.arange(6, dtype=torch.float32).reshape(2, 3)),
        ("second", torch.arange(5, dtype=torch.float32)),
    ]
    calls = []

    def fake_all_gather_into_tensor(output, tensor, group=None):
        assert group == "tp"
        calls.append((output.numel(), tensor.numel()))
        output[: tensor.numel()].copy_(tensor)
        output[tensor.numel() :].copy_(tensor + 100)

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", fake_all_gather_into_tensor)

    gathered = bucketed_all_gather_into_tensor(
        bucket,
        group="tp",
        group_size=2,
        buffer_max_size_bytes=32,
    )

    assert len(calls) == 3
    assert all(recv_numel * 4 <= 32 for recv_numel, _ in calls)
    assert all(recv_numel == 2 * send_numel for recv_numel, send_numel in calls)
    assert torch.equal(gathered[0][2][0], bucket[0][1])
    assert torch.equal(gathered[0][2][1], bucket[0][1] + 100)
    assert torch.equal(gathered[1][2][0], bucket[1][1])
    assert torch.equal(gathered[1][2][1], bucket[1][1] + 100)


def test_bucketed_all_gather_rejects_mixed_dtype_bucket() -> None:
    bucket = [
        ("float", torch.ones(2, dtype=torch.float32)),
        ("bf16", torch.ones(2, dtype=torch.bfloat16)),
    ]

    try:
        bucketed_all_gather_into_tensor(
            bucket,
            group="tp",
            group_size=2,
            buffer_max_size_bytes=32,
        )
    except ValueError as error:
        assert str(error) == "bucket tensors must share the same dtype"
    else:
        raise AssertionError("mixed dtype bucket was accepted")


def test_fsdp_dtensors_share_one_bounded_flat_collective(monkeypatch) -> None:
    class Shard:
        def __init__(self, dim: int) -> None:
            self.dim = dim

    class Mesh:
        @staticmethod
        def get_group(mesh_dim):
            assert mesh_dim == 0
            return "fsdp"

    class FakeDTensor:
        def __init__(self, local, shape, shard_dim):
            self._local = local
            self.shape = shape
            self.device = local.device
            self.dtype = local.dtype
            self.device_mesh = Mesh()
            self.placements = (Shard(shard_dim),)

        def to_local(self):
            return self._local

        def full_tensor(self):
            raise AssertionError("per-parameter full_tensor must not be used")

    first = FakeDTensor(torch.arange(6, dtype=torch.float32).reshape(2, 3), (4, 3), 0)
    second = FakeDTensor(torch.arange(4, dtype=torch.float32).reshape(2, 2), (2, 4), 1)
    calls = []

    def fake_all_gather_into_tensor(output, tensor, group=None):
        assert group == "fsdp"
        calls.append(tensor.clone())
        output[: tensor.numel()].copy_(tensor)
        output[tensor.numel() :].copy_(tensor + 100)

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.DTensor", FakeDTensor
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", fake_all_gather_into_tensor
    )

    outputs = list(
        _iter_bucketed_materialized_tensors(
            [
                ("first", first),
                ("plain", torch.tensor([7.0])),
                ("second", second),
            ],
            buffer_max_size_bytes=1024,
        )
    )
    materialized = dict(outputs)

    assert len(calls) == 1
    assert [name for name, _ in outputs] == ["plain", "first", "second"]
    assert torch.equal(
        materialized["first"],
        torch.cat([first.to_local(), first.to_local() + 100], dim=0),
    )
    assert torch.equal(
        materialized["second"],
        torch.cat([second.to_local(), second.to_local() + 100], dim=1),
    )


def test_export_batches_adjacent_tp_weights_into_one_flat_collective(
    monkeypatch,
) -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))
            self.second = nn.Parameter(torch.arange(3, dtype=torch.float32).reshape(1, 3))

    class Spec:
        num_experts = 0

        @staticmethod
        def is_expert(name):
            return False

        @staticmethod
        def tp_spec(name):
            return (0, 0)

        @staticmethod
        def native_to_hf(name, tensor):
            return [(name, tensor)]

    ps = type(
        "ParallelState",
        (),
        {
            "pp_size": 1,
            "tp_size": 2,
            "tp_group": "tp",
            "ep_size": 1,
            "ep_group": None,
            "etp_size": 1,
            "etp_group": None,
        },
    )()
    calls = []

    def fake_all_gather_into_tensor(output, tensor, group=None):
        assert group == "tp"
        calls.append(tensor.clone())
        output[: tensor.numel()].copy_(tensor)
        output[tensor.numel() :].copy_(tensor + 10)

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", fake_all_gather_into_tensor)

    exported = dict(export_hf_weights(Model(), Spec(), ps, cpu=False, buffer_max_size_bytes=1024))

    assert len(calls) == 1
    assert torch.equal(
        exported["first"],
        torch.cat([torch.arange(4).reshape(2, 2), torch.arange(4).reshape(2, 2) + 10]),
    )
    assert torch.equal(
        exported["second"],
        torch.cat([torch.arange(3).reshape(1, 3), torch.arange(3).reshape(1, 3) + 10]),
    )


def test_expert_export_yields_when_bounded_ep_bucket_fills(monkeypatch) -> None:
    class ExpertGroup(nn.Module):
        def __init__(self, offset: int) -> None:
            super().__init__()
            for idx in range(2):
                self.register_parameter(
                    f"weight{idx}",
                    nn.Parameter(torch.arange(4, dtype=torch.float32) + offset + idx * 10),
                )

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Module()
            self.first.experts = ExpertGroup(0)
            self.second = nn.Module()
            self.second.experts = ExpertGroup(1000)

        def named_parameters(self, *args, **kwargs):
            for item in super().named_parameters(*args, **kwargs):
                visited.append(item[0])
                yield item

    class Spec:
        num_experts = 4

        @staticmethod
        def is_expert(name):
            return ".experts." in name

        @staticmethod
        def tp_spec(name):
            return None

        @staticmethod
        def packed_expert_group_name(name):
            return name.rsplit(".weight", 1)[0] + ".packed"

        @staticmethod
        def native_to_hf(name, tensor):
            assert name.endswith(".packed")
            return [(name, tensor)]

    ps = type(
        "ParallelState",
        (),
        {
            "pp_size": 1,
            "tp_size": 1,
            "tp_group": None,
            "ep_size": 2,
            "ep_group": "ep",
            "etp_size": 1,
            "etp_group": None,
        },
    )()
    visited = []

    def fake_all_gather_into_tensor(output, tensor, group=None):
        assert group == "ep"
        output[: tensor.numel()].copy_(tensor)
        output[tensor.numel() :].copy_(tensor + 100)

    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", fake_all_gather_into_tensor
    )

    stream = export_hf_weights(
        Model(), Spec(), ps, cpu=False, buffer_max_size_bytes=64
    )
    name, tensor = next(stream)

    assert name == "first.experts.packed"
    assert len(visited) == 2
    expected_local = [
        torch.arange(4, dtype=torch.float32),
        torch.arange(4, dtype=torch.float32) + 10,
    ]
    assert torch.equal(
        tensor, torch.stack(expected_local + [value + 100 for value in expected_local])
    )
