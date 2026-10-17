# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The official-precision model (BF16 wo_a, group-32 experts) on the standard math.

The release contract -- wo_a unquantized, expert activations in group-32 E4M3
with the routing weight applied before FC2 quantization -- belongs to the
model, so the standard training path (deployment math off) must build and
train it, not only the rollout-aligned deployment path. The deployment-only
BF16 wo_a helper must stay off the standard path, the experts must take the
official MLP, and gradients must reach both. Agreement with the official
model itself is checked on real release weights by the θ0 gate
(examples/verl/ds41_rollout_tools); the deployment kernels need the release
attention geometry, so the tiny config cannot host that comparison.
"""

import os

import pytest
import torch


def _require_cuda(what):
    # Quantized FP8 linears have no CPU GEMM, so forward/backward need a GPU.
    if os.environ.get('MEGATRON_LITE_REQUIRE_CUDA_TESTS') == '1':
        assert torch.cuda.is_available(), what
    elif not torch.cuda.is_available():
        pytest.skip(what)


def _official_config():
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
    from test_w4a8_fp32 import tiny_config

    cfg = tiny_config().to_hf_dict()
    cfg['text_config'].update(
        wo_a_dtype='bf16',
        expert_activation_group_size=32,
        # The deployment rotary cache sizes itself from the HF text config.
        max_position_embeddings=4096,
    )
    return DeepseekV41Config(cfg)


def _build(device='cpu', **kwargs):
    from megatron.lite.model.deepseek_v41.lite import protocol

    impl = dict(device=device, dtype=torch.bfloat16, quantized=True, w4a8_experts=True)
    impl.update(kwargs)
    return protocol.build_model(
        _official_config(), impl_cfg=protocol.ImplConfig(**impl)
    ).chunks[0]


def _inputs(device='cuda'):
    generator = torch.Generator().manual_seed(5)
    ids = torch.randint(1, 64, (1, 24), generator=generator)
    return ids.to(device), torch.tensor([0, 24], dtype=torch.int32, device=device)


def test_standard_path_builds_the_official_contract(v41_core_te):
    model = _build()
    attentions = [layer.attn for layer in model.layers if layer is not None]
    assert attentions and all(not a.wo_a.fp8 and a.wq_a.fp8 for a in attentions)
    moes = [m for m in model.modules() if getattr(m, 'official_expert_contract', False)]
    assert moes and all(
        not m.deployment_math for m in moes if hasattr(m, 'deployment_math')
    )


def test_official_contract_still_requires_w4a8_experts(v41_core_te):
    with pytest.raises(ValueError, match='official expert contract requires W4A8'):
        _build(w4a8_experts=False)


def test_standard_path_runs_the_official_stages_and_trains_them(
    v41_core_te, monkeypatch
):
    from megatron.lite.primitive.modules.attention import csa
    from megatron.lite.primitive.quantization import w4a8_experts

    _require_cuda('quantized official-precision forward/backward')
    torch.manual_seed(351)
    model = _build('cuda')

    def deployment_only(*args, **kwargs):
        raise AssertionError('the standard path must not take the deployment BF16 wo_a')

    monkeypatch.setattr(csa, '_bf16_wo_a', deployment_only)
    calls = []
    official = w4a8_experts._official_expert_mlp

    def spy(*args):
        calls.append(args[-1])  # row_weights
        return official(*args)

    monkeypatch.setattr(w4a8_experts, '_official_expert_mlp', spy)
    ids, cu = _inputs()
    out = model(ids, cu_seqlens=cu, return_head_hidden=True)['head_hidden']
    out.float().square().mean().backward()
    assert calls and all(w is not None and torch.isfinite(w).all() for w in calls)
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    wo_a = [g for n, g in grads.items() if n.endswith('attn.wo_a.weight')]
    experts = [g for n, g in grads.items() if '.ffn.experts.' in n]
    assert wo_a and experts
    for g in wo_a + experts:
        assert torch.isfinite(g).all()
    assert any(g.abs().sum() > 0 for g in wo_a) and any(
        g.abs().sum() > 0 for g in experts
    )
