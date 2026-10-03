# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3-MoE as the consumer of the opt-in W4A8 routed-expert forward (CPU)."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.quantization.qat import QATSpec

pytestmark = pytest.mark.mlite

LITE_ROOT = Path(__file__).resolve().parents[3]
W4A8_MODULE = "megatron.lite.primitive.quantization.w4a8_experts"
_CONFIG = SimpleNamespace(num_experts=3, hidden_size=128, moe_intermediate_size=128)


class CpuGroupedLinear(nn.Module):
    """TE GroupedLinear parameter layout (``weight0..weightN-1``) with a CPU forward."""

    def __init__(self, num_gemms, in_features, out_features, *, bias=False, **_):
        super().__init__()
        for index in range(num_gemms):
            weight = torch.randn(out_features, in_features) * 0.05
            self.register_parameter(f"weight{index}", nn.Parameter(weight))

    def forward(self, x, m_splits):
        outputs, start = [], 0
        for index, count in enumerate(m_splits):
            weight = getattr(self, f"weight{index}")
            outputs.append(x[start : start + count] @ weight.t())
            start += count
        return torch.cat(outputs)


@pytest.fixture
def experts_cls(transformer_engine_import_stub, monkeypatch):
    transformer_engine_import_stub()
    from megatron.lite.primitive import transformer_engine as lite_te

    # Patch the TE module the wrapper bound at import, which may be an earlier stub.
    monkeypatch.setattr(lite_te._TE, "GroupedLinear", CpuGroupedLinear, raising=False)
    # The default path's SwiGLU is torch.compile'd; eager is enough on CPU here.
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    from megatron.lite.primitive.modules.experts import Experts

    return Experts


def _experts(experts_cls, seed=0, **kwargs):
    torch.manual_seed(seed)
    return experts_cls(_CONFIG, ParallelState(), **kwargs).to(torch.bfloat16)


def _routed_tokens(seed=1):
    torch.manual_seed(seed)
    m_splits = [4, 0, 9]
    x = torch.randn(sum(m_splits), _CONFIG.hidden_size).to(torch.bfloat16)
    return x, torch.tensor(m_splits), torch.rand(sum(m_splits))


def _spec(**overrides):
    return QATSpec(enabled=True, format="mxfp4", activation_bits=8, **overrides)


def test_enabled_experts_run_the_w4a8_primitive_on_master_weights(experts_cls):
    from megatron.lite.primitive.modules.experts import enable_w4a8_experts
    from megatron.lite.primitive.quantization.w4a8_experts import w4a8_expert_mlp

    experts = _experts(experts_cls)
    x, tokens_per_expert, probs = _routed_tokens()
    default_out = experts(x, tokens_per_expert, probs)

    assert enable_w4a8_experts([experts], _spec()) == 1
    out = experts(x, tokens_per_expert, probs)

    fc1 = [getattr(experts.fc1, f"weight{i}") for i in range(3)]
    fc2 = [getattr(experts.fc2, f"weight{i}") for i in range(3)]
    expected = w4a8_expert_mlp(x, fc1, fc2, [4, 0, 9], probs.unsqueeze(-1))
    assert torch.equal(out, expected)
    assert not torch.equal(out, default_out)


def test_w4a8_experts_train_the_bf16_master_weights(experts_cls):
    from megatron.lite.primitive.modules.experts import enable_w4a8_experts

    experts = _experts(experts_cls)
    enable_w4a8_experts([experts], _spec())
    x, tokens_per_expert, probs = _routed_tokens()
    experts(x, tokens_per_expert, probs).float().square().sum().backward()

    for index, has_tokens in enumerate((True, False, True)):
        for linear in (experts.fc1, experts.fc2):
            weight = getattr(linear, f"weight{index}")
            assert weight.dtype == torch.bfloat16 and weight.grad is not None
            assert bool(torch.count_nonzero(weight.grad)) == has_tokens


def test_enable_leaves_state_dict_bytes_unchanged(experts_cls):
    from megatron.lite.primitive.modules.experts import enable_w4a8_experts

    baseline = _experts(experts_cls).state_dict()
    experts = _experts(experts_cls)
    enable_w4a8_experts([experts], _spec())
    state = experts.state_dict()

    assert list(state) == list(baseline)
    assert all(
        torch.equal(state[k].view(torch.uint8), baseline[k].view(torch.uint8))
        for k in state
    )


def test_enable_respects_ignore_patterns_and_rejects_unreproduced_modes(experts_cls):
    from megatron.lite.primitive.modules.experts import enable_w4a8_experts

    root = nn.Module()
    root.experts = _experts(experts_cls)
    assert enable_w4a8_experts([root], _spec(ignore_patterns=("experts",))) == 0
    assert root.experts.w4a8 is False

    for kwargs, match in (
        ({"fp8": True}, "fp8"),
        ({"moe_act_recompute": True}, "recompute"),
        ({"lora_config": {"enabled": True, "rank": 2}}, "LoRA"),
    ):
        with pytest.raises(ValueError, match=match):
            enable_w4a8_experts([_experts(experts_cls, **kwargs)], _spec())


@pytest.mark.parametrize(
    "hidden, intermediate, etp, gemm",
    [
        (192, 128, 1, "FC1"),  # hidden size not a multiple of 128
        (128, 192, 1, "FC2"),  # intermediate size not a multiple of 128
        (128, 128, 2, "FC2"),  # ETP shards FC2 K to 64 per rank
    ],
)
def test_enable_rejects_k_dims_the_a8_groups_cannot_tile(
    experts_cls, hidden, intermediate, etp, gemm
):
    from megatron.lite.primitive.modules.experts import enable_w4a8_experts

    config = SimpleNamespace(
        num_experts=2, hidden_size=hidden, moe_intermediate_size=intermediate
    )
    supported = _experts(experts_cls)
    unsupported = experts_cls(config, ParallelState(etp_size=etp))

    with pytest.raises(
        ValueError, match=f"{gemm} GEMM K dimension .* divisible by 128"
    ):
        enable_w4a8_experts([supported, unsupported], _spec())
    # Rejected at enable time, before any module is switched.
    assert supported.w4a8 is False and unsupported.w4a8 is False


def _build(monkeypatch, experts_cls, qat):
    from megatron.lite.model.qwen3_moe.lite import protocol

    class Model(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].moe = nn.Module()
            self.layers[0].moe.experts = experts_cls(_CONFIG, ParallelState())

    monkeypatch.setattr(protocol, "Qwen3MoEModel", Model)
    monkeypatch.setattr(protocol, "init_parallel", lambda _p: ParallelState())
    monkeypatch.setattr(nn.Module, "cuda", lambda self: self)
    impl_cfg = protocol.ImplConfig(optimizer=None, qat=qat)
    bundle = protocol.build_model(
        SimpleNamespace(num_nextn_predict_layers=0), impl_cfg=impl_cfg
    )
    return bundle.chunks[0].layers[0].moe.experts


def test_qwen3_moe_build_model_opts_experts_into_w4a8(monkeypatch, experts_cls):
    a8 = {"enabled": True, "format": "mxfp4", "activation_bits": 8}
    assert _build(monkeypatch, experts_cls, a8).w4a8 is True
    assert _build(monkeypatch, experts_cls, {**a8, "enabled": False}).w4a8 is False
    weight_only = {"enabled": True, "format": "mxfp4"}
    assert _build(monkeypatch, experts_cls, weight_only).w4a8 is False
    assert _build(monkeypatch, experts_cls, None).w4a8 is False

    with pytest.raises(ValueError, match="matched no routed-expert module"):
        _build(monkeypatch, experts_cls, {**a8, "ignore_patterns": ["experts"]})


# CPU stand-ins for the TE modules the real Qwen3MoEModel builds (TE is not
# installed for CPU unit tests): same parameter names, eager torch forwards.
def _rms_norm(x, weight, eps, zero_centered_gamma=False):
    weight = weight + 1 if zero_centered_gamma else weight
    x32 = x.float()
    out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    return out.to(x.dtype)


class CpuRMSNorm(nn.Module):
    def __init__(self, hidden, *, eps=1e-6, zero_centered_gamma=False, **_):
        super().__init__()
        self.eps, self.zero_centered_gamma = eps, zero_centered_gamma
        self.weight = nn.Parameter(
            torch.zeros(hidden) if zero_centered_gamma else torch.ones(hidden)
        )

    def forward(self, x):
        return _rms_norm(x, self.weight, self.eps, self.zero_centered_gamma)


class CpuLinear(nn.Module):
    def __init__(self, in_features, out_features, *, bias=True, return_bias=False, **_):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.return_bias = return_bias

    def forward(self, x):
        out = nn.functional.linear(
            x, self.weight, None if self.return_bias else self.bias
        )
        return (out, self.bias) if self.return_bias else out


class CpuLayerNormLinear(CpuLinear):
    def __init__(
        self,
        in_features,
        out_features,
        *,
        normalization="LayerNorm",
        eps=1e-5,
        zero_centered_gamma=False,
        return_layernorm_output=False,
        **kwargs,
    ):
        super().__init__(in_features, out_features, **kwargs)
        self.normalization, self.eps = normalization, eps
        self.zero_centered_gamma = zero_centered_gamma
        self.return_layernorm_output = return_layernorm_output
        self.layer_norm_weight = nn.Parameter(torch.ones(in_features))
        if normalization == "LayerNorm":
            self.layer_norm_bias = nn.Parameter(torch.zeros(in_features))

    def forward(self, x):
        if self.normalization == "RMSNorm":
            normed = _rms_norm(
                x, self.layer_norm_weight, self.eps, self.zero_centered_gamma
            )
        else:
            weight = self.layer_norm_weight + int(self.zero_centered_gamma)
            normed = nn.functional.layer_norm(
                x, x.shape[-1:], weight, self.layer_norm_bias, self.eps
            )
        out = super().forward(normed)
        if not self.return_layernorm_output:
            return out
        return (*out, normed) if self.return_bias else (out, normed)


class CpuDotProductAttention(nn.Module):
    """Causal GQA attention for the ``sbhd`` layout GQAttention passes (non-THD)."""

    def __init__(
        self,
        num_attention_heads,
        kv_channels,
        *,
        num_gqa_groups=None,
        attn_mask_type="causal",
        qkv_format="sbhd",
        **_,
    ):
        super().__init__()
        assert qkv_format == "sbhd" and attn_mask_type == "causal"
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_gqa_groups or num_attention_heads

    def forward(self, q, k, v, **_):
        # [s, b, h, d] -> [b, h, s, d]; each KV head serves a contiguous group of Q heads.
        q, k, v = (t.permute(1, 2, 0, 3) for t in (q, k, v))
        group = self.num_heads // self.num_kv_heads
        k, v = (t.repeat_interleave(group, dim=1) for t in (k, v))
        out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.permute(2, 0, 1, 3).flatten(2)  # [s, b, h * d]


def test_real_qwen3_moe_w4a8_build_model_trains_one_step_on_cpu(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    monkeypatch.setenv("MEGATRON_LITE_DISABLE_JIT_FUSER", "1")
    monkeypatch.setenv("MEGATRON_LITE_MOE_PERMUTE_FUSION", "0")  # unfused torch permute
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    from megatron.lite.primitive import transformer_engine as lite_te

    for name, cls in (
        ("Linear", CpuLinear),
        ("LayerNormLinear", CpuLayerNormLinear),
        ("RMSNorm", CpuRMSNorm),
        ("DotProductAttention", CpuDotProductAttention),
        ("GroupedLinear", CpuGroupedLinear),
    ):
        monkeypatch.setattr(lite_te._TE, name, cls, raising=False)

    import megatron.lite.primitive.quantization.w4a8_experts as w4a8
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
    from megatron.lite.model.qwen3_moe.lite import protocol
    from megatron.lite.model.qwen3_moe.lite.model import Qwen3MoEModel
    from megatron.lite.primitive.modules.experts import Experts
    from megatron.lite.primitive.quantization.qat import WeightFakeQuant
    from torch.nn.utils import parametrize

    assert protocol.Qwen3MoEModel is Qwen3MoEModel  # the real model, not a stub
    monkeypatch.setattr(protocol, "init_parallel", lambda _p: ParallelState())
    monkeypatch.setattr(nn.Module, "cuda", lambda self: self)
    w4a8_calls = []
    real_mlp = w4a8.w4a8_expert_mlp

    def counting_mlp(x, fc1, fc2, m_splits, *args, **kwargs):
        w4a8_calls.append(list(m_splits))
        return real_mlp(x, fc1, fc2, m_splits, *args, **kwargs)

    monkeypatch.setattr(w4a8, "w4a8_expert_mlp", counting_mlp)

    torch.manual_seed(0)
    config = Qwen3MoEConfig(
        num_hidden_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        max_position_embeddings=16,
        layer_types=["full_attention", "full_attention"],
    )
    # build_model's dist_opt needs CUDA/NCCL; the step below uses AdamW instead.
    impl_cfg = protocol.ImplConfig(optimizer=None, qat=_spec())
    model = protocol.build_model(config, impl_cfg=impl_cfg).chunks[0]
    assert type(model) is Qwen3MoEModel

    experts = [m for m in model.modules() if isinstance(m, Experts)]
    assert len(experts) == config.num_hidden_layers
    assert all(m.w4a8 is True for m in experts)
    # Dense attention linears are weight-only fake-quant; router / embedding /
    # lm_head stay unquantized; experts quantize inside the W4A8 GEMM instead.
    for layer in model.layers:
        for linear in (layer.attn.qkv.linear, layer.attn.proj.linear):
            assert isinstance(linear.parametrizations.weight[0], WeightFakeQuant)
        unquantized = (
            layer.moe.router.gate,
            layer.moe.experts.fc1,
            layer.moe.experts.fc2,
        )
        assert not any(parametrize.is_parametrized(m) for m in unquantized)
    assert not parametrize.is_parametrized(model.embed.embedding)
    assert not parametrize.is_parametrized(model.head.col.linear)

    expert_weights = [
        (e, i, getattr(linear, f"weight{i}"))
        for e, module in enumerate(experts)
        for linear in (module.fc1, module.fc2)
        for i in range(config.num_experts)
    ]
    before = [weight.detach().clone() for *_, weight in expert_weights]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)

    model.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.roll(input_ids, -1, dims=1)
    loss = model(input_ids=input_ids, labels=labels)["loss"]
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert len(w4a8_calls) == config.num_hidden_layers
    assert all(
        sum(m_splits) == 2 * 8 * config.num_experts_per_tok for m_splits in w4a8_calls
    )

    loss.backward()
    optimizer.step()
    for (layer, index, weight), old in zip(expert_weights, before):
        has_tokens = w4a8_calls[layer][index] > 0
        assert weight.dtype == torch.bfloat16 and weight.grad.dtype == torch.bfloat16
        assert bool(torch.count_nonzero(weight.grad)) == has_tokens
        assert (not torch.equal(weight.detach(), old)) == has_tokens


def _primitive_test_helpers():
    """The exact vLLM ``ep_gather`` transcription lives with the primitive tests."""
    import importlib.util

    path = LITE_ROOT / "tests/unit/primitive/quantization/test_w4a8_experts_unit.py"
    spec = importlib.util.spec_from_file_location("_w4a8_primitive_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _moe_layer(experts_cls, **overrides):
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
    from megatron.lite.model.qwen3_moe.lite.model import MoELayer

    # Qwen3-30B-A3B expert dims (hidden 2048, moe_intermediate 768, top-k 8);
    # 10 of its 128 experts to keep the CPU test small.
    fields = dict(
        hidden_size=2048,
        moe_intermediate_size=768,
        num_experts=10,
        num_experts_per_tok=8,
    )
    torch.manual_seed(13)
    config = Qwen3MoEConfig(**{**fields, **overrides})
    layer = MoELayer(config, ParallelState(), use_deepep=False).to(torch.bfloat16)
    return layer, config


def _capture(layer):
    seen = {}
    layer.router.register_forward_hook(
        lambda _m, _a, out: seen.update(scores=out[0], indices=out[1])
    )
    layer.experts.register_forward_hook(
        lambda _m, args, out: seen.update(dispatched=args[0], tpe=args[1], rows=out)
    )
    return seen


def test_w4a8_moe_layer_matches_vllm_combine_bitwise_qwen3_30b_topk8(
    monkeypatch, experts_cls
):
    """Full MoE layer output == rollout contract, including the top-k sum.

    Expert rows are the W4A8 FC2 outputs without router weights; vLLM's
    ``ep_gather`` then accumulates ``fp32(row) * w`` per token in top-k slot
    order with an FMA and rounds once to BF16. Transcribed exactly here.
    """
    monkeypatch.setenv("MEGATRON_LITE_MOE_PERMUTE_FUSION", "0")
    from megatron.lite.primitive.modules.experts import enable_w4a8_experts

    layer, config = _moe_layer(experts_cls)
    assert enable_w4a8_experts([layer], _spec()) == 1
    seen = _capture(layer)
    torch.manual_seed(14)
    x = torch.randn(4, config.hidden_size).to(torch.bfloat16)
    with torch.no_grad():
        out = layer(x)

    # The expert rows themselves are pinned bit for bit to the rollout's FC2
    # output elsewhere; here they must be unweighted and summed as vLLM does.
    indices, scores, rows = seen["indices"], seen["scores"], seen["rows"]
    m_splits = seen["tpe"].tolist()
    rows_tk = torch.empty(
        len(x), indices.shape[1], config.hidden_size, dtype=rows.dtype
    )
    start = 0
    for expert, count in enumerate(m_splits):
        tokens = (indices == expert).any(-1).nonzero()[:, 0]  # unfused order
        assert len(tokens) == count
        for row, token in zip(rows[start : start + count], tokens.tolist()):
            rows_tk[token, indices[token].tolist().index(expert)] = row
        start += count
    expected = _primitive_test_helpers()._vllm_ep_gather(rows_tk, scores)

    assert indices.shape[1] == 8
    assert torch.equal(out, expected)


def test_default_moe_layer_keeps_the_bf16_unpermute_combine(monkeypatch, experts_cls):
    monkeypatch.setenv("MEGATRON_LITE_MOE_PERMUTE_FUSION", "0")
    from megatron.lite.primitive.modules.dispatcher import TokenDispatcher

    def unexpected(*_args, **_kwargs):
        raise AssertionError("default path must not use the W4A8 combine")

    monkeypatch.setattr(TokenDispatcher, "combine_unreduced", unexpected, raising=False)
    layer, config = _moe_layer(
        experts_cls,
        hidden_size=128,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
    )
    seen = _capture(layer)
    torch.manual_seed(15)
    x = torch.randn(6, config.hidden_size).to(torch.bfloat16)
    with torch.no_grad():
        out = layer(x)
        # Experts apply the router weight to each BF16 row; unpermute sums in BF16.
        indices, scores = seen["indices"], seen["scores"]
        probs_2d = torch.zeros(len(x), config.num_experts, dtype=scores.dtype)
        probs_2d.scatter_add_(1, indices, scores)
        order = [
            (expert, token)
            for expert in range(config.num_experts)
            for token in (indices == expert).any(-1).nonzero()[:, 0].tolist()
        ]
        probs = torch.stack([probs_2d[t, e] for e, t in order])
        rows = layer.experts(seen["dispatched"], seen["tpe"], probs)
        expected = torch.zeros_like(x).index_add_(
            0, torch.tensor([t for _, t in order]), rows
        )
    assert torch.equal(out, expected)


def test_build_rejects_w4a8_with_fused_moe_permute(monkeypatch, experts_cls):
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
    from megatron.lite.model.qwen3_moe.lite import protocol

    monkeypatch.setenv("MEGATRON_LITE_MOE_PERMUTE_FUSION", "1")
    monkeypatch.setattr(protocol, "init_parallel", lambda _p: ParallelState())
    monkeypatch.setattr(nn.Module, "cuda", lambda self: self)
    from megatron.lite.primitive import transformer_engine as lite_te

    for name, cls in (
        ("Linear", CpuLinear),
        ("LayerNormLinear", CpuLayerNormLinear),
        ("RMSNorm", CpuRMSNorm),
        ("DotProductAttention", CpuDotProductAttention),
    ):
        monkeypatch.setattr(lite_te._TE, name, cls, raising=False)
    config = Qwen3MoEConfig(
        num_hidden_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        max_position_embeddings=16,
        layer_types=["full_attention"],
    )
    with pytest.raises(ValueError, match="unfused MoE permute"):
        protocol.build_model(
            config, impl_cfg=protocol.ImplConfig(optimizer=None, qat=_spec())
        )


def test_default_experts_forward_never_imports_w4a8_module():
    script = "\n".join(
        [
            "import sys, types",
            "import torch",
            "import torch.nn as nn",
            "te_root = types.ModuleType('transformer_engine')",
            "te_pt = types.ModuleType('transformer_engine.pytorch')",
            "te_root.pytorch = te_pt",
            "sys.modules['transformer_engine'] = te_root",
            "sys.modules['transformer_engine.pytorch'] = te_pt",
            inspect.getsource(CpuGroupedLinear),
            "te_pt.GroupedLinear = CpuGroupedLinear",
            "from types import SimpleNamespace",
            "import megatron.lite.primitive.quantization",
            "from megatron.lite.primitive.modules.experts import Experts",
            "from megatron.lite.primitive.parallel import ParallelState",
            "cfg = SimpleNamespace(num_experts=2, hidden_size=128, moe_intermediate_size=128)",
            "experts = Experts(cfg, ParallelState()).to(torch.bfloat16)",
            "x = torch.randn(5, 128).to(torch.bfloat16)",
            "experts(x, torch.tensor([2, 3]), torch.rand(5))",
            f"assert {W4A8_MODULE!r} not in sys.modules",
            "experts.w4a8 = True",
            "experts(x, torch.tensor([2, 3]), torch.rand(5))",
            f"assert {W4A8_MODULE!r} in sys.modules",
        ]
    )
    env = {**os.environ, "MEGATRON_LITE_DISABLE_JIT_FUSER": "1"}
    subprocess.run([sys.executable, "-c", script], cwd=LITE_ROOT, env=env, check=True)
