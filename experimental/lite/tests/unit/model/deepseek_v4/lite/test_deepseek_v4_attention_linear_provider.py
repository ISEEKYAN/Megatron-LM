# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4 CSA as a consumer of the opt-in FP32-master Linear providers.

CPU cases construct real CSA / V4 layers. When Megatron Core's experimental CSA
kernels or Transformer Engine are unavailable, only the import-time symbols CSA
needs are stubbed; RMSNorm becomes ``torch.nn.RMSNorm``. No forward runs on CPU.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.lite.primitive.modules import native_fp32_linear as nfl

_PROJECTIONS = ("wq_a", "wq_b", "wkv", "wo_b")
_CORE_SYMBOLS = {
    "megatron.core.tensor_parallel.mappings": ["gather_from_sequence_parallel_region"],
    "megatron.core.transformer.experimental_attention_variant": [
        "csa_cp_layout_kernels",
        "csa_cp_utils",
    ],
    "megatron.core.transformer.experimental_attention_variant.csa": [
        "_unfused_indexer_sparse_attn_from_topk",
        "unfused_compressed_sparse_attn",
    ],
    "megatron.core.transformer.experimental_attention_variant.csa_kernels": [
        "FusedCSAIndexerSparseAttnFromTopkFunc",
        "csa_sparse_attn",
    ],
    "megatron.core.transformer.experimental_attention_variant.dsa": [
        "DSAIndexerLossAutoScaler",
        "DSAIndexerLossLoggingHelper",
    ],
}
_STUBBED_IMPORTERS = (
    "megatron.lite.primitive.modules.attention",
    "megatron.lite.model.deepseek_v4",
)


@pytest.fixture
def v4(transformer_engine_import_stub, monkeypatch):
    transformer_engine_import_stub()
    before, stubbed = set(sys.modules), False
    try:
        from megatron.lite.model.deepseek_v4.lite import model, protocol
    except ImportError:
        stubbed = True
        for name, symbols in _CORE_SYMBOLS.items():
            module = types.ModuleType(name)
            for symbol in symbols:
                setattr(module, symbol, object())
            monkeypatch.setitem(sys.modules, name, module)
        for name in set(sys.modules) - before:
            if name.startswith(_STUBBED_IMPORTERS):
                del sys.modules[name]
        from megatron.lite.model.deepseek_v4.lite import model, protocol
    from megatron.lite.primitive import transformer_engine as te

    monkeypatch.setattr(te, "RMSNorm", lambda size, eps: nn.RMSNorm(size, eps=eps))
    yield SimpleNamespace(model=model, protocol=protocol)
    # Modules imported against the stubs must not leak into later tests.
    for name in set(sys.modules) - before if stubbed else ():
        if name.startswith(_STUBBED_IMPORTERS):
            del sys.modules[name]


def _config():
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    return DeepseekV4Config(
        hidden_size=64,
        num_attention_heads=2,
        head_dim=32,
        qk_rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        compress_ratios=[4, 0],
        index_head_dim=32,
        index_n_heads=2,
        num_hidden_layers=2,
    )


def _ps():
    return SimpleNamespace(cp_size=1, cp_rank=0, cp_group=None)


def _csa(v4, seed=0, **kwargs):
    torch.manual_seed(seed)
    return v4.model.CompressedSparseAttention(
        _config(), layer_idx=0, ps=_ps(), **kwargs
    )


def test_default_csa_construction_is_unchanged(v4):
    omitted = _csa(v4)
    explicit = _csa(v4, linear_provider=None)
    default = _csa(v4, linear_provider=nfl.linear_provider("default"))
    for name in _PROJECTIONS:
        assert type(getattr(omitted, name)) is nn.Linear
    reference = omitted.state_dict()
    for other in (explicit, default):
        state = other.state_dict()
        assert list(state) == list(reference)
        assert all(torch.equal(state[key], reference[key]) for key in reference)


def test_block32_provider_keeps_names_and_fp32_masters(v4):
    default = _csa(v4).to(torch.bfloat16)
    provided = _csa(v4, linear_provider=nfl.linear_provider("block32_fp8"))
    nfl.restore_fp32_masters(provided.to(torch.bfloat16))
    default_state, state = default.state_dict(), provided.state_dict()
    assert list(state) == list(default_state)
    assert all(state[key].shape == default_state[key].shape for key in state)
    for key, tensor in state.items():
        projection = key.split(".")[0] in _PROJECTIONS
        assert tensor.dtype == (
            torch.float32 if projection else default_state[key].dtype
        )
    for name in _PROJECTIONS:
        assert getattr(provided, name).mode == "block32_fp8"
    # Only the four dense projections move; compressor / indexer / wo_a stay.
    assert type(provided.wo_a) is type(default.wo_a)
    assert type(provided.indexer.wq_b) is nn.Linear


def test_v4_layers_thread_provider_to_decoder_and_mtp_attention(v4, monkeypatch):
    class _Stub(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    for name in ("DeepseekV4MoE", "HyperConnection", "MultiHeadHyperConnectionHead"):
        monkeypatch.setattr(v4.model, name, _Stub)
    provider = nfl.linear_provider("native_fp32")
    layer = v4.model.DeepseekV4Layer(_config(), _ps(), 0, linear_provider=provider)
    mtp = v4.model.DeepseekV4MTPLayer(
        _config(),
        _ps(),
        2,
        embedding=_Stub(),
        use_deepep=False,
        detach_encoder=False,
        linear_provider=provider,
    )
    default = v4.model.DeepseekV4Layer(_config(), _ps(), 1)
    for name in _PROJECTIONS:
        assert getattr(layer.self_attn.self_attn, name).mode == "native_fp32"
        assert getattr(mtp.self_attn.self_attn, name).mode == "native_fp32"
        assert type(getattr(default.self_attn.self_attn, name)) is nn.Linear
    assert type(mtp.e_proj) is nn.Linear


def test_protocol_selects_provider_opt_in_and_rejects_unsupported(v4):
    ImplConfig = v4.protocol.ImplConfig
    select = v4.protocol._attention_linear_provider
    assert ImplConfig().attention_linear == "default"
    assert select(ImplConfig()) is None
    built = select(ImplConfig(attention_linear="block32_fp8"))(64, 32, bias=False)
    assert built.mode == "block32_fp8" and built.weight.dtype == torch.float32
    with pytest.raises(ValueError, match="unknown linear provider"):
        select(ImplConfig(attention_linear="mxfp4"))
    with pytest.raises(ValueError, match="QAT"):
        select(ImplConfig(attention_linear="native_fp32", qat={"enabled": True}))
    with pytest.raises(ValueError, match="fsdp2"):
        select(ImplConfig(attention_linear="block32_fp8", optimizer="fsdp2"))
    assert select(ImplConfig(qat={"enabled": True})) is None


@pytest.mark.gpus(1)
def test_csa_forward_backward_calls_fp8_gemm_with_fp32_wgrad(v4, monkeypatch):
    # Not run yet: fused BSHD CSA setup follows test_csa_thd_cp.py, which also
    # initializes a single-rank NCCL group before calling the fused kernels.
    config = _config()
    config.num_attention_heads, config.head_dim = 64, 512
    config.qk_rope_head_dim, config.index_head_dim, config.index_n_heads = 64, 128, 64
    calls = []
    scaled_mm = torch._scaled_mm

    def spy(*args, **kwargs):
        calls.append(args[0].shape)
        return scaled_mm(*args, **kwargs)

    monkeypatch.setattr(torch, "_scaled_mm", spy)
    provider = nfl.linear_provider("block32_fp8")
    module = v4.model.CompressedSparseAttention(
        config, layer_idx=0, ps=_ps(), linear_provider=provider
    )
    module = nfl.restore_fp32_masters(module.to(device="cuda", dtype=torch.bfloat16))
    module.attention_backend = "flash"
    x = torch.randn(1, 8, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    out = module(x, position_ids=torch.arange(8, device="cuda").unsqueeze(0))
    out.float().sum().backward()
    assert calls and torch.isfinite(out).all()
    for name in _PROJECTIONS:
        grad = getattr(module, name).weight.grad
        assert grad.dtype == torch.float32 and torch.isfinite(grad).all()
