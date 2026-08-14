# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.lite.primitive.ckpt.hf_weights import _copy_loaded_tensor_
from megatron.lite.primitive.init_device import (
    build_module_on_device,
    finalize_meta_materialization,
    materialize_meta_module,
    record_materialized_tensor,
    transformer_engine_init_device,
    use_fsdp2_meta_init,
)
from megatron.lite.primitive.optimizers.fsdp2 import FSDP2Config
from megatron.lite.primitive.optimizers.fsdp2.wrap import wrap_fsdp2
from megatron.lite.primitive.parallel.state import ParallelState


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 3)
        # Meta-capable model constructors keep non-checkpoint buffers real so
        # their explicit initializer survives delayed parameter materialization.
        self.register_buffer("scale", torch.full((1,), 2.0, device="cpu"))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.proj(value) * self.scale


class BadMetaBufferModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2))
        self.register_buffer("forgotten", torch.ones(1))


class ExplicitRealParameterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, device="cpu"))
        self.register_buffer("initialized", torch.full((1,), 3.0, device="cpu"))


def test_fsdp2_meta_default_is_scoped_to_fsdp2_only() -> None:
    assert use_fsdp2_meta_init(SimpleNamespace(optimizer="fsdp2")) is True
    assert (
        use_fsdp2_meta_init(SimpleNamespace(optimizer="fsdp2", fsdp2_meta_init=False))
        is False
    )
    assert use_fsdp2_meta_init(SimpleNamespace(optimizer="dist_opt")) is False


def test_transformer_engine_device_follows_meta_context_only() -> None:
    assert transformer_engine_init_device().type == "cuda"
    with torch.device("meta"):
        assert transformer_engine_init_device().type == "meta"


def test_build_module_on_meta_defers_only_parameter_storage() -> None:
    model = build_module_on_device(TinyModel, use_meta=True, dtype=torch.bfloat16)

    assert {param.device.type for param in model.parameters()} == {"meta"}
    assert {param.dtype for param in model.parameters()} == {torch.bfloat16}


def test_meta_build_canonicalizes_explicit_real_parameters_and_preserves_buffers() -> (
    None
):
    model = build_module_on_device(
        ExplicitRealParameterModel, use_meta=True, dtype=torch.bfloat16
    )

    assert model.weight.device.type == "meta"
    assert model.initialized.device.type == "cpu"
    assert model.initialized.item() == 3.0


def test_meta_materialization_matches_loaded_eager_state_bitwise() -> None:
    torch.manual_seed(17)
    eager = build_module_on_device(
        TinyModel, use_meta=False, dtype=torch.bfloat16, device="cpu"
    )
    expected = {
        name: tensor.detach().clone() for name, tensor in eager.state_dict().items()
    }

    meta = build_module_on_device(TinyModel, use_meta=True, dtype=torch.bfloat16)
    materialize_meta_module(meta, device="cpu")
    for name, param in meta.named_parameters():
        param.data.copy_(expected[name])
        record_materialized_tensor(meta, name)
    finalize_meta_materialization(meta)

    actual = meta.state_dict()
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name
    assert meta.scale.item() == 2.0


def test_meta_materialization_fails_loudly_for_unfilled_parameter() -> None:
    model = build_module_on_device(TinyModel, use_meta=True, dtype=torch.bfloat16)
    materialize_meta_module(model, device="cpu")
    model.proj.weight.data.zero_()
    record_materialized_tensor(model, "proj.weight")

    try:
        finalize_meta_materialization(model)
    except RuntimeError as exc:
        message = str(exc)
        assert "proj.bias" in message
        assert "checkpoint" in message.lower()
    else:  # pragma: no cover - assertion gives a clearer failure than pytest here
        raise AssertionError("missing meta-created parameters must fail loudly")


def test_meta_materialization_fails_loudly_for_uninitialized_buffer() -> None:
    model = build_module_on_device(
        BadMetaBufferModel, use_meta=True, dtype=torch.bfloat16
    )
    materialize_meta_module(model, device="cpu")
    model.weight.data.zero_()
    record_materialized_tensor(model, "weight")

    try:
        finalize_meta_materialization(model)
    except RuntimeError as exc:
        assert "forgotten" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("uninitialized meta buffers must fail loudly")


def _fsdp2_meta_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        model = build_module_on_device(TinyModel, use_meta=True, dtype=torch.float32)
        wrap_fsdp2(
            model,
            ParallelState(),
            FSDP2Config(
                unit_modules=(nn.Linear,),
                device_type="cpu",
                param_dtype=torch.float32,
            ),
            mesh=init_device_mesh("cpu", (world_size,)),
        )
        materialize_meta_module(model, device="cpu")
        expected = {
            "proj.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "proj.bias": torch.arange(3, dtype=torch.float32),
        }
        for name, param in model.named_parameters():
            _copy_loaded_tensor_(param, expected[name])
            record_materialized_tensor(model, name)
        finalize_meta_materialization(model)

        for name, param in model.named_parameters():
            assert torch.equal(param.full_tensor(), expected[name]), name
        model(torch.ones(2, 4)).sum().backward()
    finally:
        dist.destroy_process_group()


def test_fsdp2_meta_materializes_only_local_shards_and_runs_backward(
    free_tcp_port: int,
) -> None:
    import torch.multiprocessing as mp

    mp.spawn(_fsdp2_meta_worker, args=(2, free_tcp_port), nprocs=2, join=True)
