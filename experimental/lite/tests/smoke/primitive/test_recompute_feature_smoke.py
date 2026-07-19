# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CUDA semantic smoke for the observable recompute contract."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.primitive.recompute import apply_recompute
from megatron.lite.runtime.contracts.config import ParallelConfig


pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu]


class _CountingSquare(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x.square()


@pytest.fixture(scope="module", autouse=True)
def _single_gpu_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for recompute semantic smoke.")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29581")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created_pg = False
    if not dist.is_initialized():
        dist.init_process_group("nccl")
        created_pg = True
    yield
    if created_pg and dist.is_initialized():
        dist.destroy_process_group()


def test_recompute_replays_cuda_forward_and_preserves_gradient() -> None:
    layer = nn.Module().cuda()
    layer.inner = _CountingSquare().cuda()
    result = apply_recompute(
        nn.ModuleList([layer]), ["inner"], {"inner": lambda module: module.inner}
    )

    x = torch.tensor([2.0, -3.0], device="cuda", requires_grad=True)
    layer.inner(x).sum().backward()

    assert (result.units, result.matched, result.wrapped) == (1, 1, 1)
    assert layer.inner.calls == 2
    torch.testing.assert_close(x.grad, torch.tensor([4.0, -6.0], device="cuda"))
    print("CUDA_RECOMPUTE_SEMANTIC_SMOKE_PASSED units=1 matched=1 wrapped=1")


def test_deepseek_v4_build_model_applies_recompute_to_assembled_layers() -> None:
    """Exercise the real protocol assembly path, not its source text."""
    pytest.importorskip("cudnn", reason="deepseek_v4 fused DSA needs the cudnn DSA stack.")
    pytest.importorskip("transformer_engine.pytorch", reason="real Transformer Engine is required.")
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite import protocol

    cfg = DeepseekV4Config(
        vocab_size=64,
        hidden_size=128,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=64,
        qk_rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        max_position_embeddings=4096,
        compress_ratios=[4, 4],
        sliding_window=128,
        num_hash_layers=2,
        hc_mult=2,
        index_head_dim=64,
        index_n_heads=8,
        index_topk=512,
        num_nextn_predict_layers=0,
        rms_norm_eps=1e-6,
    )
    bundle = protocol.build_model(
        cfg,
        impl_cfg=protocol.ImplConfig(
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=1, vpp=1),
            optimizer=None,
            recompute=["core_attn"],
            mtp_enable=False,
        ),
    )

    units = protocol._iter_transformer_units(bundle.chunks[0])
    assert len(units) == cfg.num_hidden_layers
    assert all(
        getattr(unit.self_attn, "_megatron_lite_recompute_wrapped", False) for unit in units
    )
    print("DS4_BUILD_MODEL_RECOMPUTE_CONFORMANCE_PASSED layers=2 wrapped=2")


def test_deepseek_v4_assembled_weights_enumerate_through_resync_export() -> None:
    """Run the actual model enumeration and quantized resync export path."""
    pytest.importorskip("cudnn", reason="deepseek_v4 fused DSA needs the cudnn DSA stack.")
    pytest.importorskip("transformer_engine.pytorch", reason="real Transformer Engine is required.")
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite import protocol

    cfg = DeepseekV4Config(
        vocab_size=64,
        hidden_size=128,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=64,
        qk_rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        max_position_embeddings=4096,
        compress_ratios=[4, 4],
        sliding_window=128,
        num_hash_layers=2,
        hc_mult=2,
        index_head_dim=64,
        index_n_heads=8,
        index_topk=512,
        num_nextn_predict_layers=0,
        rms_norm_eps=1e-6,
        quantization_config={"weight_block_size": [128, 128]},
    )
    bundle = protocol.build_model(
        cfg,
        impl_cfg=protocol.ImplConfig(
            parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=1, vpp=1),
            optimizer=None,
            mtp_enable=False,
        ),
    )

    exported = dict(
        protocol.export_hf_weights(
            bundle.chunks,
            cfg,
            bundle.parallel_state,
            target="block_fp8",
            resync_config={"expert_dtype": "fp8"},
        )
    )
    expert_weights = [name for name in exported if ".experts." in name and name.endswith(".weight")]
    assert expert_weights
    assert all(f"{name[:-7]}.scale" in exported for name in expert_weights)
    print(f"DS4_RESYNC_ENUMERATION_PASSED expert_weights={len(expert_weights)}")
