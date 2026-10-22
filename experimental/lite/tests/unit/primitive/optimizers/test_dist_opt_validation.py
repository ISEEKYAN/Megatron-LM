# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from megatron.lite.primitive.optimizers.megatron_wrap import (
    build_dist_opt_optimizer_config,
    validate_dist_opt_config,
    validate_dist_opt_session,
)
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.contracts.config import ParallelConfig


def _engine_cfg(*, model_name: str, pp: int = 1, vpp: int = 1) -> MegatronLiteConfig:
    return MegatronLiteConfig(model_name=model_name, parallel=ParallelConfig(pp=pp, vpp=vpp))


def test_dist_opt_validation_accepts_model_agnostic_config():
    validate_dist_opt_config(_engine_cfg(model_name="synthetic_custom_model", pp=1, vpp=1))


def test_dist_opt_validation_keeps_vpp_parallel_constraint():
    with pytest.raises(ValueError, match="dist_opt requires pp>1 when vpp>1"):
        validate_dist_opt_config(_engine_cfg(model_name="synthetic_custom_model", pp=1, vpp=2))


def test_validate_dist_opt_session_alias_matches_config_validator():
    assert validate_dist_opt_session is validate_dist_opt_config
    validate_dist_opt_session(_engine_cfg(model_name="another_synthetic_model", pp=2, vpp=2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_chunked_state_offload_pins_host_buffers_at_exact_size(monkeypatch):
    from megatron.core.optimizer.cpu_offloading.chunked_optimizer_state_offload import (
        ChunkedOptimizerStateOffloader as Offloader,
    )

    monkeypatch.setattr(Offloader, "_new_cpu_buffer", Offloader.__dict__["_new_cpu_buffer"])
    opt = SimpleNamespace(
        optimizer="adam", lr=1e-6, weight_decay=0.1, clip_grad=1.0, chunked_state_offload=True
    )
    build_dist_opt_optimizer_config(opt)
    source = torch.randn(3, 1000, device="cuda")
    buffer = Offloader._new_cpu_buffer(source)
    assert buffer.is_pinned() and buffer.shape == source.shape and buffer.dtype == source.dtype
    assert buffer.untyped_storage().nbytes() == source.numel() * source.element_size()
    buffer.copy_(source, non_blocking=True)
    torch.cuda.synchronize()
    assert torch.equal(buffer, source.cpu())
