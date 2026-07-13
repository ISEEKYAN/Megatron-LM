# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

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
    SafeTensorReader,
    _iter_bucketed_materialized_tensors,
    bucketed_all_gather_into_tensor,
    export_hf_weights,
)


def test_safetensor_reader_context_reuses_each_shard_handle(
    monkeypatch, tmp_path
) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "first": "shard-a.safetensors",
                    "second": "shard-a.safetensors",
                    "third": "shard-b.safetensors",
                }
            }
        )
    )
    events = []

    class FakeHandle:
        def __init__(self, filename: str) -> None:
            self.filename = filename

        def __enter__(self):
            events.append(("open", self.filename))
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            del exc_type, exc, traceback
            events.append(("close", self.filename))

        def get_tensor(self, name: str) -> torch.Tensor:
            events.append(("get", self.filename, name))
            return torch.tensor(len(name))

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.safe_open",
        lambda path, **kwargs: FakeHandle(Path(path).name),
    )

    with SafeTensorReader(str(tmp_path)) as reader:
        assert reader.get_tensor("first").item() == 5
        assert reader.get_tensor("second").item() == 6
        assert reader.get_tensor("first").item() == 5
        assert reader.get_tensor("third").item() == 5

    assert events.count(("open", "shard-a.safetensors")) == 1
    assert events.count(("open", "shard-b.safetensors")) == 1
    assert events.count(("close", "shard-a.safetensors")) == 1
    assert events.count(("close", "shard-b.safetensors")) == 1

    events.clear()
    reader.get_tensor("first")
    reader.get_tensor("first")
    assert events.count(("open", "shard-a.safetensors")) == 2
    assert events.count(("close", "shard-a.safetensors")) == 2

    events.clear()
    with pytest.raises(RuntimeError, match="stop loading"):
        with reader:
            reader.get_tensor("first")
            raise RuntimeError("stop loading")
    assert events.count(("open", "shard-a.safetensors")) == 1
    assert events.count(("close", "shard-a.safetensors")) == 1


def test_export_defaults_to_device_resident_tensors() -> None:
    assert inspect.signature(export_hf_weights).parameters["cpu"].default is False


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
    resident = dict(export_hf_weights(model, Spec(), ps))

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


def test_fsdp_dtensors_share_one_bounded_flat_collective(monkeypatch) -> None:
    class Shard:
        def __init__(self, dim: int) -> None:
            self.dim = dim

    class Mesh:
        def __init__(self, group) -> None:
            self.group = group

        def get_group(self, mesh_dim):
            assert mesh_dim == 0
            return self.group

    class FakeDTensor:
        def __init__(self, local, shape, shard_dim, group):
            self._local = local
            self.shape = shape
            self.device = local.device
            self.dtype = local.dtype
            self.device_mesh = Mesh(group)
            self.placements = (Shard(shard_dim),)

        def to_local(self):
            return self._local

        def full_tensor(self):
            raise AssertionError("per-parameter full_tensor must not be used")

    first_group = object()
    equivalent_group = object()
    single_rank_group = object()
    first = FakeDTensor(
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        (4, 3),
        0,
        first_group,
    )
    single = FakeDTensor(
        torch.arange(3, dtype=torch.float32), (3,), 0, single_rank_group
    )
    second = FakeDTensor(
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
        (2, 4),
        1,
        equivalent_group,
    )
    calls = []

    def fake_all_gather_into_tensor(output, tensor, group=None):
        assert group is first_group
        calls.append(tensor.clone())
        output[: tensor.numel()].copy_(tensor)
        output[tensor.numel() :].copy_(tensor + 100)

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.DTensor", FakeDTensor
    )
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group: 1 if group is single_rank_group else 2,
    )
    monkeypatch.setattr(
        torch.distributed,
        "get_process_group_ranks",
        lambda group: [0] if group is single_rank_group else [0, 1],
    )
    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", fake_all_gather_into_tensor
    )

    outputs = list(
        _iter_bucketed_materialized_tensors(
            [
                ("first", first),
                ("plain", torch.tensor([7.0])),
                ("single", single),
                ("second", second),
            ],
            buffer_max_size_bytes=1024,
        )
    )
    materialized = dict(outputs)

    assert len(calls) == 1
    assert [name for name, _ in outputs] == ["first", "plain", "single", "second"]
    assert materialized["single"] is single.to_local()
    assert torch.equal(
        materialized["first"],
        torch.cat([first.to_local(), first.to_local() + 100], dim=0),
    )
    assert torch.equal(
        materialized["second"],
        torch.cat([second.to_local(), second.to_local() + 100], dim=1),
    )


def test_oversized_fsdp_shard_is_gathered_in_shard_dimension_chunks(
    monkeypatch,
) -> None:
    class Shard:
        dim = 1

    class Mesh:
        @staticmethod
        def get_group(mesh_dim):
            assert mesh_dim == 0
            return "fsdp"

    class FakeDTensor:
        def __init__(self, local):
            self._local = local
            self.shape = (local.shape[0], local.shape[1] * 2)
            self.device_mesh = Mesh()
            self.placements = (Shard(),)

        def to_local(self):
            return self._local

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local = torch.arange(10, dtype=torch.float32, device=device).reshape(2, 5)
    calls = []

    def fake_all_gather_into_tensor(output, tensor, group=None):
        assert group == "fsdp"
        calls.append(tensor.shape)
        output[: tensor.numel()].copy_(tensor)
        output[tensor.numel() :].copy_(tensor + 100)

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.DTensor", FakeDTensor
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: [0, 1]
    )
    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", fake_all_gather_into_tensor
    )
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.torch.empty_like",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("oversized shards must not allocate full rank copies")
        ),
    )

    outputs = dict(
        _iter_bucketed_materialized_tensors(
            [("oversized", FakeDTensor(local))], buffer_max_size_bytes=32
        )
    )

    assert calls == [torch.Size([4]), torch.Size([4]), torch.Size([2])]
    assert torch.equal(outputs["oversized"], torch.cat([local, local + 100], dim=1))


def test_replicated_dtensor_uses_local_tensor_without_collective(monkeypatch) -> None:
    class Replicate:
        pass

    class FakeDTensor:
        def __init__(self, local):
            self._local = local
            self.shape = local.shape
            self.device = local.device
            self.dtype = local.dtype
            self.placements = (Replicate(),)

        def to_local(self):
            return self._local

        def full_tensor(self):
            raise AssertionError("replicated parameters must not call full_tensor")

    local = torch.arange(5, dtype=torch.float32)
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.DTensor", FakeDTensor
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_into_tensor",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("replicated parameters must not be gathered")
        ),
    )

    outputs = list(
        _iter_bucketed_materialized_tensors([("replicated", FakeDTensor(local))])
    )

    assert outputs[0][0] == "replicated"
    assert outputs[0][1] is local


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


def _pp2_stream_fixture(monkeypatch, *, rank=0, pp_rank=0, cpu=False, rank0_only=False):
    """Fake pp_size=2 streaming export scaffold.

    The R path replaces the legacy `all_gather_object(full_dict)` with a
    lightweight metadata `all_gather_object` + one-at-a-time `broadcast` per PP
    stage. This fixture drives one rank of a pp2 world: the local stage owns
    ``weight``; the remote stage owns ``weight2`` (value = local + 100). Returns
    ``(recorded, remote_weight, local_weight, run)``.
    """
    local_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    remote_weight = local_weight + 100

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(local_weight.clone())

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
            "pp_size": 2,
            "pp_rank": pp_rank,
            "pp_global_ranks": [0, 1],
            "tp_size": 1,
            "tp_group": None,
            "ep_size": 1,
            "ep_group": None,
            "etp_size": 1,
            "etp_group": None,
            "pp_group": "nccl-pp",
            "pp_cpu_group": "gloo-pp",
        },
    )()

    recorded = {
        "meta_group": None,
        "bcast_groups": [],
        "bcast_srcs": [],
        "empty_shapes": [],
        "max_live_empty": 0,
    }
    remote_meta = [("weight2", (2, 3), torch.float32)]

    def fake_all_gather_object(all_meta, local_meta, group=None):
        recorded["meta_group"] = group
        all_meta[pp_rank] = local_meta
        all_meta[1 - pp_rank] = remote_meta

    remote_iter = iter([remote_weight])
    my_global = ps.pp_global_ranks[pp_rank]

    def fake_broadcast(tensor, src=None, group=None):
        recorded["bcast_groups"].append(group)
        recorded["bcast_srcs"].append(src)
        if src != my_global:  # receiver: fill from the remote stage
            tensor.copy_(next(remote_iter))

    real_empty = torch.empty

    def spy_empty(*args, **kwargs):
        if args and not isinstance(args[0], int):
            recorded["empty_shapes"].append(tuple(args[0]))
            # One in-flight broadcast buffer at a time = bounded peak.
            recorded["max_live_empty"] = max(recorded["max_live_empty"], 1)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.torch.empty", spy_empty
    )

    def run():
        return dict(
            export_hf_weights(
                Model(), Spec(), ps, cpu=cpu, rank0_only=rank0_only
            )
        )

    return recorded, remote_weight, local_weight, run


def test_pp_export_streams_over_nccl_and_matches_materialized(monkeypatch) -> None:
    recorded, remote_weight, local_weight, run = _pp2_stream_fixture(monkeypatch)

    exported = run()

    # Metadata exchange and every tensor broadcast ride the NCCL pp_group — the
    # gloo pp_cpu_group is never used (R does not offload to CPU).
    assert recorded["meta_group"] == "nccl-pp"
    assert recorded["bcast_groups"] == ["nccl-pp", "nccl-pp"]
    # Streamed output is bitwise-equal to the legacy materialized full dict.
    assert exported.keys() == {"weight", "weight2"}
    assert torch.equal(exported["weight"], local_weight)
    assert torch.equal(exported["weight2"], remote_weight)


def test_pp_export_holds_one_inflight_buffer(monkeypatch) -> None:
    recorded, _, _, run = _pp2_stream_fixture(monkeypatch)

    run()

    # Exactly one remote param is allocated (the source pops its own tensor from
    # the resident stage dict), so peak residency is (own stage) + one buffer.
    assert recorded["empty_shapes"] == [(2, 3)]
    assert recorded["max_live_empty"] == 1
    # broadcast sources: our stage (global rank 0) then the remote stage (rank 1).
    assert recorded["bcast_srcs"] == [0, 1]


def test_pp_export_cpu_still_broadcasts_over_nccl(monkeypatch) -> None:
    recorded, remote_weight, local_weight, run = _pp2_stream_fixture(
        monkeypatch, cpu=True
    )

    exported = run()

    # cpu=True routes final residency to host but the broadcast stays on the
    # NCCL pp_group (no gloo dependency); output is still correct.
    assert recorded["bcast_groups"] == ["nccl-pp", "nccl-pp"]
    assert exported["weight"].device.type == "cpu"
    assert exported["weight2"].device.type == "cpu"
    assert torch.equal(exported["weight"], local_weight)
    assert torch.equal(exported["weight2"], remote_weight)


def test_pp_export_rank0_only_participates_but_yields_nothing(monkeypatch) -> None:
    # A non-zero rank under rank0_only must still join every collective
    # (metadata all_gather + both broadcasts) but emit no params.
    recorded, _, _, run = _pp2_stream_fixture(
        monkeypatch, rank=1, pp_rank=1, rank0_only=True
    )

    exported = run()

    assert exported == {}
    assert recorded["meta_group"] == "nccl-pp"
    assert len(recorded["bcast_groups"]) == 2
