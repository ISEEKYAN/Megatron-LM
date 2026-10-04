# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Opt-in preserves the exact external 04c736eed CSA2 forward and VJP."""
import hashlib
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

BASELINE = '04c736eed0ff7c31dddec56e458e914cf067c716'
CSA_PATH = 'experimental/lite/megatron/lite/primitive/modules/attention/csa.py'


@pytest.fixture
def baseline_csa(v41_core_te, monkeypatch):
    from megatron.lite.primitive import transformer_engine as te
    from megatron.lite.primitive.modules.attention import csa

    monkeypatch.setattr(te, 'RMSNorm', torch.nn.RMSNorm)

    external = os.environ.get('CSA_R1_BASELINE')
    source = (
        Path(external).read_bytes()
        if external
        else subprocess.check_output(
            ['git', 'show', BASELINE + ':' + CSA_PATH],
            cwd=Path(__file__).resolve().parents[5],
        )
    )
    assert (
        hashlib.sha256(source).hexdigest()
        == '838373308fbdf7213a0fc6652738d322d7546a2202a36e4c10c7043ca54f46e1'
    )
    original = types.ModuleType('r1_baseline_csa')
    sys.modules[original.__name__] = original
    exec(compile(source, BASELINE + ':' + CSA_PATH, 'exec'), original.__dict__)
    return original, csa


@pytest.mark.parametrize(
    'quantized', [False, pytest.param(True, marks=pytest.mark.gpus(1))]
)
@pytest.mark.parametrize('optimizing', [False, True])
def test_off_csa2_is_bitwise_baseline_forward_and_all_gradients(
    baseline_csa, quantized, optimizing
):
    original, current = baseline_csa
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.environ.get('MEGATRON_LITE_REQUIRE_CUDA_TESTS') == '1':
        assert device == 'cuda', 'Quantized default-path proof requires CUDA'
    if quantized and device == 'cpu':
        pytest.skip('Real FP8 Linear has no CPU fallback')
    config = current.CrossLayerAttentionConfig(
        dim=32,
        heads=4,
        head_dim=32,
        rope_dim=4,
        q_rank=32,
        o_rank=32,
        groups=2,
        index_heads=1,
        index_dim=32,
        linear_fp8=quantized,
        main_qat=False,
        index_qat=False,
        swa_fp8=False,
    )
    kwargs = dict(
        ps=None,
        layer_idx=0,
        kv_owner=None,
        index_owner=None,
        candidate_mode='none',
        compress_ratio=0,
    )
    torch.manual_seed(817)
    left = original.CompressedSparseAttention(config, **kwargs).to(
        device=device, dtype=torch.bfloat16
    )
    right = current.CompressedSparseAttention(config, **kwargs).to(
        device=device, dtype=torch.bfloat16
    )
    # Muon keeps projection leaves in FP32 and selects native FP32 gradients.
    for model in (left, right):
        for module in model.modules():
            if hasattr(module, 'fp8_operator'):
                module.native_fp32 = optimizing
                if optimizing:
                    module.weight.data = module.weight.data.float()
    right.load_state_dict(left.state_dict())
    assert not right.deployment_math
    x = torch.randn(2, 5, 32, device=device).bfloat16().requires_grad_()
    z = x.detach().clone().requires_grad_()
    expected, _ = left(x, original.AttentionState())
    actual, _ = right(z, current.AttentionState())
    assert torch.equal(expected, actual)
    incoming = torch.randn_like(expected)
    expected.backward(incoming)
    actual.backward(incoming)
    assert torch.equal(x.grad, z.grad)
    for (name, a), (other, b) in zip(
        left.named_parameters(), right.named_parameters(), strict=True
    ):
        assert name == other
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad), name


@pytest.mark.parametrize(
    'quantized,w4a8', [(True, False), (False, True), (False, False)]
)
def test_deployment_math_rejects_unverified_expert_modes(v41_core_te, quantized, w4a8):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model
    from test_w4a8_fp32 import tiny_config

    config = tiny_config()
    with pytest.raises(ValueError, match='Deployment math requires quantized W4A8'):
        protocol.build_model(
            config,
            impl_cfg=protocol.ImplConfig(
                device='cpu',
                dtype=torch.bfloat16,
                quantized=quantized,
                w4a8_experts=w4a8,
                deployment_math=True,
            ),
        )
    with pytest.raises(ValueError, match='Deployment math requires quantized W4A8'):
        DeepseekV41Model(
            config, quantized=quantized, w4a8_experts=w4a8, deployment_math=True
        )
