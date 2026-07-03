# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4 indexer KL-loss wiring and scaling parity tests."""

from __future__ import annotations

import os

import pytest
import torch


def _tiny_ds4_config(*, loss_coeff: float = 0.0):
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    return DeepseekV4Config(
        vocab_size=64,
        hidden_size=128,
        moe_intermediate_size=16,
        num_hidden_layers=1,
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
        compress_ratios=[4],
        sliding_window=4,
        num_hash_layers=1,
        hc_mult=2,
        index_head_dim=64,
        index_n_heads=8,
        index_topk=2,
        num_nextn_predict_layers=0,
        dsa_indexer_loss_coeff=loss_coeff,
        rms_norm_eps=1e-6,
    )


def _init_cuda_dist_or_skip():
    import torch.distributed as dist

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for DeepSeek-V4 indexer-loss tests.")
    if "RANK" not in os.environ:
        pytest.skip("Run with torchrun so the real ModelBundle can initialize parallel state.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return torch.device("cuda", local_rank)


def test_deepseek_v4_config_loads_dsa_indexer_loss_coeff():
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    cfg = DeepseekV4Config._from_hf_dict({"dsa_indexer_loss_coeff": 0.125})

    assert cfg.dsa_indexer_loss_coeff == 0.125


@pytest.mark.gpu
def test_deepseek_v4_csa_attaches_configured_indexer_loss(monkeypatch):
    """The fused CSA path must pass the configured coeff and attach its loss."""
    device = _init_cuda_dist_or_skip()

    from megatron.lite.primitive.modules.attention import csa
    from megatron.lite.primitive.modules.attention.csa import CompressedSparseAttention
    from megatron.lite.primitive.modules.attention.dsa import DSAIndexerLossAutoScaler
    from megatron.lite.primitive.parallel.state import ParallelState

    calls = {}

    def fake_fused_indexer_sparse_attn(
        query,
        kv_full,
        attn_sink,
        window_idxs,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        ratio,
        softmax_scale,
        indexer_softmax_scale=1.0,
        loss_coeff=0.0,
        sparse_loss=False,
        kv_offset=0,
        calculate_per_token_loss=False,
    ):
        del (
            kv_full,
            attn_sink,
            window_idxs,
            k_indexer,
            weights,
            indexer_topk,
            ratio,
            softmax_scale,
            indexer_softmax_scale,
            kv_offset,
            calculate_per_token_loss,
        )
        calls.update(loss_coeff=loss_coeff, sparse_loss=sparse_loss)
        output = query.new_zeros(query.shape[0], query.shape[1], query.shape[2] * query.shape[3])
        indexer_loss = q_indexer.float().square().mean() * loss_coeff
        return output, indexer_loss

    monkeypatch.setattr(
        csa._load_dsa_kernels(),
        "fused_indexer_sparse_attn",
        fake_fused_indexer_sparse_attn,
    )

    cfg = _tiny_ds4_config(loss_coeff=0.25)
    attention = CompressedSparseAttention(cfg, layer_idx=0, ps=ParallelState()).to(
        device=device, dtype=torch.bfloat16
    )
    attention.attention_backend = "fused"
    attention.train()
    DSAIndexerLossAutoScaler.main_loss_backward_scale = None
    DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(0.5, device=device))

    x = torch.randn(1, 4, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    position_ids = torch.arange(4, device=device).unsqueeze(0)
    output = attention(x, position_ids=position_ids)
    output.float().sum().backward()

    assert calls == {"loss_coeff": 0.25, "sparse_loss": False}
    grad = attention.indexer.wq_b.weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad).item() > 0


@pytest.mark.gpu
def test_deepseek_v4_bundle_hook_matches_indexer_gradient_reference():
    """The registered hook must match the independently reduced gradient."""
    device = _init_cuda_dist_or_skip()

    from megatron.lite.model.deepseek_v4.lite import protocol
    from megatron.lite.primitive.modules.attention.dsa import (
        DSAIndexerLossAutoScaler as LiteDSAIndexerLossAutoScaler,
    )
    from megatron.lite.runtime.contracts.config import ParallelConfig

    cfg = _tiny_ds4_config(loss_coeff=0.25)
    impl_cfg = protocol.ImplConfig(
        parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=1),
        optimizer=None,
        mtp_enable=False,
        deterministic=True,
    )
    bundle = protocol.build_model(cfg, impl_cfg=impl_cfg)
    hook = bundle.extras["pre_forward_hook"]
    assert hook is not None

    num_microbatches = 4

    def accumulated_grad(scaler, *, scale_hook=None):
        scaler.main_loss_backward_scale = None
        parameter = torch.tensor(2.0, device=device, requires_grad=True)
        for _ in range(num_microbatches):
            if scale_hook is not None:
                scale_hook(torch.tensor(1.0 / num_microbatches, device=device))
            output = torch.zeros((), device=device, requires_grad=True)
            scaler.apply(output, parameter.square()).backward()
        return parameter.grad.detach().clone()

    lite_grad = accumulated_grad(LiteDSAIndexerLossAutoScaler, scale_hook=hook)
    unscaled_grad = accumulated_grad(LiteDSAIndexerLossAutoScaler)
    reference_parameter = torch.tensor(2.0, device=device, requires_grad=True)
    for _ in range(num_microbatches):
        (reference_parameter.square() / num_microbatches).backward()
    reference_grad = reference_parameter.grad

    torch.testing.assert_close(lite_grad, reference_grad)
    torch.testing.assert_close(unscaled_grad, lite_grad * num_microbatches)
    if torch.distributed.get_rank() == 0:
        print(
            "NON_SKIP_DS4_INDEXER_LOSS_SCALE_PARITY_PASSED "
            f"num_microbatches={num_microbatches} "
            f"lite_grad={lite_grad.item():.6e} "
            f"reference_grad={reference_grad.item():.6e} "
            f"unscaled_grad={unscaled_grad.item():.6e}"
        )
