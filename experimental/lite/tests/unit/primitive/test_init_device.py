# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


def _build_with_load_setting(monkeypatch, transformer_engine_import_stub, load_hf_weights):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import protocol

    seen = []

    class Model(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            seen.append(self.weight.device.type)

    monkeypatch.setattr(protocol, "Qwen3MoEModel", Model)
    monkeypatch.setattr(protocol, "init_parallel", lambda _p: SimpleNamespace())
    monkeypatch.setattr(protocol, "normalize_lora_config", lambda _cfg: SimpleNamespace(enabled=False))
    monkeypatch.setattr(protocol, "parse_recompute_spec", lambda _cfg: [])
    monkeypatch.setattr(protocol, "set_cross_entropy_fusion", lambda *_args: None)
    monkeypatch.setattr(protocol, "apply_qat_to_chunks", lambda *_args: None)
    monkeypatch.setattr(nn.Module, "cuda", lambda self: self)
    cfg = SimpleNamespace(num_nextn_predict_layers=0)
    protocol.build_model(cfg, impl_cfg=protocol.ImplConfig(optimizer="fsdp2", load_hf_weights=load_hf_weights))
    return seen


@pytest.mark.parametrize(("load_hf_weights", "device"), [(True, "meta"), (False, "cpu")])
def test_fsdp2_init_device_follows_load_setting(
    monkeypatch, transformer_engine_import_stub, load_hf_weights, device
) -> None:
    assert _build_with_load_setting(
        monkeypatch, transformer_engine_import_stub, load_hf_weights
    ) == [device]


def test_fully_sharded_meta_model_supports_to_empty(tmp_path) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        with torch.device("meta"):
            model = nn.Linear(8, 2).to(torch.bfloat16)
        fully_shard(model, mesh=init_device_mesh("cpu", (1,)))
        model.to_empty(device="cpu")
        assert model.weight.to_local().device.type == "cpu"
        assert model.weight.dtype == torch.bfloat16
    finally:
        dist.destroy_process_group()


def test_dispatcher_metadata_stays_materialized_in_meta_context() -> None:
    from megatron.lite.primitive.modules.dispatcher import TokenDispatcher

    with torch.device("meta"):
        dispatcher = TokenDispatcher(
            num_experts=4,
            hidden_size=8,
            ps=SimpleNamespace(ep_size=2),
            use_deepep=False,
        )

    assert dispatcher._sort_by_experts == [0, 2, 1, 3]
    assert dispatcher._restore_by_ranks == [0, 2, 1, 3]
