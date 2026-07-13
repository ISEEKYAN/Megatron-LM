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

    The R path lazily gathers the own stage and streams the PP dimension in
    bounded buckets: per stage it broadcasts one bucket's (name, shape, dtype)
    header (``broadcast_object_list``, empty header = end-of-stage) then the
    tensors one at a time (``broadcast``). This fixture drives one rank of a pp2
    world: the local stage owns ``weight``; the remote stage owns ``weight2``
    (value = local + 100). Returns ``(recorded, remote_weight, local_weight,
    run)``.
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
        "obj_groups": [],
        "obj_srcs": [],
        "bcast_groups": [],
        "bcast_srcs": [],
        "empty_shapes": [],
        "max_live_empty": 0,
    }
    my_global = ps.pp_global_ranks[pp_rank]

    # The remote stage streams one non-empty bucket [(weight2, ...)] then an
    # empty header (end-of-stage). Consumed only during receiver turns.
    remote_headers = iter([[("weight2", (2, 3), torch.float32)], []])
    remote_iter = iter([remote_weight])

    def fake_broadcast_object_list(object_list, src=None, group=None, device=None):
        recorded["obj_groups"].append(group)
        recorded["obj_srcs"].append(src)
        if src != my_global:  # receiver: pull the remote stage's next header
            object_list[0] = next(remote_headers)
        # else: this rank is the source; object_list is already populated.

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
    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", fake_broadcast_object_list
    )
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

    # Every header exchange and tensor broadcast rides the NCCL pp_group — the
    # gloo pp_cpu_group is never used (R does not offload to CPU).
    assert set(recorded["obj_groups"]) == {"nccl-pp"}
    assert recorded["bcast_groups"] == ["nccl-pp", "nccl-pp"]
    # Streamed output is bitwise-equal to the legacy materialized full dict.
    assert exported.keys() == {"weight", "weight2"}
    assert torch.equal(exported["weight"], local_weight)
    assert torch.equal(exported["weight2"], remote_weight)


def test_pp_export_holds_one_inflight_buffer(monkeypatch) -> None:
    recorded, _, _, run = _pp2_stream_fixture(monkeypatch)

    run()

    # Exactly one remote param is allocated (the source broadcasts straight from
    # its bounded bucket), so peak residency is (one bucket) + one recv buffer.
    assert recorded["empty_shapes"] == [(2, 3)]
    assert recorded["max_live_empty"] == 1
    # tensor broadcast sources: our stage (global rank 0) then remote stage (1).
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
    # (header exchange + both tensor broadcasts) but emit no params.
    recorded, _, _, run = _pp2_stream_fixture(
        monkeypatch, rank=1, pp_rank=1, rank0_only=True
    )

    exported = run()

    assert exported == {}
    assert set(recorded["obj_groups"]) == {"nccl-pp"}
    assert len(recorded["bcast_groups"]) == 2


def test_pp_export_never_materializes_the_whole_stage(monkeypatch) -> None:
    """Residency guard: the own stage is gathered lazily, one bounded bucket at
    a time — the whole stage is never built before the first yield.

    With ``buffer_max_size_bytes`` forcing one param per bucket, pulling the very
    first streamed param must have visited exactly one of the stage's three
    parameters (had the legacy path pre-built the stage dict, all three would be
    visited before anything yielded).
    """
    visited: list[str] = []
    params = {
        "weight_a": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "weight_b": torch.arange(6, dtype=torch.float32).reshape(2, 3) + 10,
        "weight_c": torch.arange(6, dtype=torch.float32).reshape(2, 3) + 20,
    }

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for name, value in params.items():
                self.register_parameter(name, nn.Parameter(value.clone()))

        def named_parameters(self, *args, **kwargs):
            for item in super().named_parameters(*args, **kwargs):
                visited.append(item[0])
                yield item

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
            "pp_rank": 0,
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

    my_global = 0

    def fake_broadcast_object_list(object_list, src=None, group=None, device=None):
        if src != my_global:
            object_list[0] = []  # never reached: we stop after the first param

    def fake_broadcast(tensor, src=None, group=None):
        pass  # source (rank 0) broadcasts straight from its bucket

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", fake_broadcast_object_list
    )
    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)

    # buffer_max_size_bytes=1 caps every bucket at a single parameter.
    stream = export_hf_weights(Model(), Spec(), ps, buffer_max_size_bytes=1)
    first_name, first_tensor = next(stream)

    assert first_name == "weight_a"
    assert torch.equal(first_tensor, params["weight_a"])
    # Only the first parameter has been pulled from the lazy generator; the
    # remaining two stay untouched — the stage was never fully materialized.
    assert visited == ["weight_a"]


def test_expert_export_never_materializes_the_whole_expert_set(monkeypatch) -> None:
    """Residency guard: the EP (expert) export path streams bounded buckets and
    never materializes the whole expert set on any rank before it starts
    yielding.

    This mirrors DS4's production expert path: unpacked experts
    (``packed_expert_group_name`` returns ``None``) are gathered EP-rank by
    EP-rank in buckets capped at ``buffer_max_size_bytes // ep_size`` (and at
    most four local params per collective — hf_weights.py ``_flush_expert_bucket``
    / the ``len(expert_bucket) >= 4`` flush trigger).  A regression that reverted
    to whole-set materialization — collecting every local expert (and its EP
    all-gather output) into one dict before the first yield, the shape of both
    upstreams' in-memory EP gather (mbridge ``_flush_ep_bucket`` expands the
    gathered bucket into an ``etp_bucket`` sized to the full expert set;
    NVIDIA Megatron-Bridge ``gather_from_ep_ranks`` all-gathers each expert with
    no per-rank byte cap) — would make this test RED: it would fire every EP
    collective (one per local expert) before a single param streamed out, and
    the peak count of gathered-but-not-yet-yielded expert tensors would equal the
    whole local set rather than one bounded bucket.
    """
    num_local_experts = 16  # per EP rank; 2 EP ranks -> 32 experts total
    ep_size = 2
    num_experts_total = num_local_experts * ep_size

    class ExpertGroup(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for idx in range(num_local_experts):
                self.register_parameter(
                    f"weight{idx}",
                    nn.Parameter(torch.arange(8, dtype=torch.float32) + idx * 100),
                )

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.experts = ExpertGroup()

    class Spec:
        num_experts = num_experts_total

        @staticmethod
        def is_expert(name):
            return ".experts." in name

        @staticmethod
        def tp_spec(name):
            return None

        @staticmethod
        def packed_expert_group_name(name):
            return None  # DS4: unpacked, each expert emitted as its own key

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
            "ep_size": ep_size,
            "ep_group": "ep",
            "etp_size": 1,
            "etp_group": None,
        },
    )()

    # Track the number of EP collectives fired and the peak count of gathered
    # expert shards that are alive but not yet yielded.
    state = {"gathers": 0, "gathered_since_yield": 0, "peak_gathered": 0}

    def fake_all_gather_into_tensor(output, tensor, group=None):
        assert group == "ep"
        state["gathers"] += 1
        # Each collective gathers one bounded bucket (<=4 local params); count
        # how many local params rode this collective.
        per_param = tensor.numel() // 4  # 8 elements/param, fp32
        n_params_in_bucket = max(1, tensor.numel() // 8)
        state["gathered_since_yield"] += n_params_in_bucket
        state["peak_gathered"] = max(
            state["peak_gathered"], state["gathered_since_yield"]
        )
        del per_param
        output[: tensor.numel()].copy_(tensor)
        output[tensor.numel() :].copy_(tensor + 1000)

    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", fake_all_gather_into_tensor
    )

    # buffer_max_size_bytes // ep_size caps each bucket well below the whole
    # local set: 8 elements * 4 bytes = 32 B/param, cap 128 B/rank -> 4 params
    # per bucket (also hits the len>=4 flush).
    buffer_max_size_bytes = 256
    per_rank_cap_bytes = buffer_max_size_bytes // ep_size  # 128 B
    max_params_per_bucket = 4  # len(expert_bucket) >= 4 flush trigger

    stream = export_hf_weights(
        Model(), Spec(), ps, cpu=False, buffer_max_size_bytes=buffer_max_size_bytes
    )

    yielded = 0
    gathers_before_first_yield = None
    outputs = {}
    for name, tensor in stream:
        if gathers_before_first_yield is None:
            gathers_before_first_yield = state["gathers"]
        outputs[name] = tensor
        yielded += 1
        # After every yield the bucket has been flushed and its shards released,
        # so residency resets.
        state["gathered_since_yield"] = 0

    # (1) Streaming, not whole-set: the first param streamed out after only one
    # bounded bucket's worth of collectives — NOT after gathering every local
    # expert. Whole-set materialization would make this == num_local_experts.
    assert gathers_before_first_yield == 1, gathers_before_first_yield
    assert gathers_before_first_yield < num_local_experts

    # (2) Peak residency stays bounded by one bucket (<= 4 local params), never
    # the whole local expert set. This is the "整组物化必红" assertion.
    assert state["peak_gathered"] <= max_params_per_bucket, state["peak_gathered"]
    assert state["peak_gathered"] < num_local_experts

    # (3) Total collectives = ceil(local experts / bucket cap), i.e. the path
    # actually chunked instead of one giant gather.
    import math

    assert state["gathers"] == math.ceil(num_local_experts / max_params_per_bucket)

    # (4) Correctness: every global expert key is present and carries the right
    # value (local from rank 0, local+1000 from rank 1).
    assert len(outputs) == num_experts_total
    for local_idx in range(num_local_experts):
        base = torch.arange(8, dtype=torch.float32) + local_idx * 100
        # ep_rank 0 -> global == local_idx; ep_rank 1 -> global == num_local + local
        assert torch.equal(
            outputs[f"mlp.experts.weight{local_idx}"], base
        )
        assert torch.equal(
            outputs[f"mlp.experts.weight{num_local_experts + local_idx}"], base + 1000
        )
