from __future__ import annotations

import importlib.util
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from megatron.lite.model.deepseek_v4.vllm import model as model_module
from megatron.lite.model.deepseek_v4.vllm.model import DeepseekV4Layer
from megatron.lite.primitive.kernels import vllm_ds4
from megatron.lite.primitive.kernels.vllm_ds4 import MHCKernel, MHCTileLangAdapter


def _post_inputs(device: str = "cpu") -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(4, 128, dtype=torch.bfloat16, device=device),
        torch.randn(4, 4, 128, dtype=torch.bfloat16, device=device),
        torch.randn(4, 4, 1, dtype=torch.float32, device=device),
        torch.randn(4, 4, 4, dtype=torch.float32, device=device),
    )


@pytest.mark.parametrize("kernel,entry", list(vllm_ds4._MHC_ENTRIES.items()))
def test_each_mhc_adapter_calls_the_official_entry(
    monkeypatch: pytest.MonkeyPatch, kernel: MHCKernel, entry: str
) -> None:
    result = torch.tensor([7])
    official = Mock(return_value=result)
    monkeypatch.setattr(
        vllm_ds4,
        "_symbol",
        lambda module, name: official
        if (module, name) == ("vllm.model_executor.kernels.mhc.tilelang", entry)
        else pytest.fail((module, name)),
    )
    if kernel is MHCKernel.POST:
        args = _post_inputs()
    elif kernel is MHCKernel.HEAD:
        args = (
            torch.zeros(2, 4, 128, dtype=torch.bfloat16),
            torch.zeros(4, 512),
            torch.zeros(1),
            torch.zeros(4),
            1e-6,
            1e-6,
        )
    else:
        residual = torch.zeros(2, 128, dtype=torch.bfloat16)
        if kernel is not MHCKernel.PRE_BROADCAST:
            residual = torch.zeros(2, 4, 128, dtype=torch.bfloat16)
        pre = (
            residual,
            torch.zeros(24, 512),
            torch.zeros(3),
            torch.zeros(24),
            1e-6,
            1e-6,
            1e-6,
            2.0,
            2,
        )
        args = _post_inputs() + pre[1:] if kernel is MHCKernel.POST_PRE else pre
    assert MHCTileLangAdapter(kernel)(*args) is result
    official.assert_called_once_with(*args)


def test_mhc_contract_rejects_bad_dtype_before_kernel_lookup(monkeypatch) -> None:
    lookup = Mock()
    monkeypatch.setattr(vllm_ds4, "_symbol", lookup)
    args = list(_post_inputs())
    args[0] = args[0].float()
    with pytest.raises(TypeError, match="dtype"):
        MHCTileLangAdapter("post")(*args)
    lookup.assert_not_called()


def test_cross_layer_state_runs_attention_post_pre_before_attention(monkeypatch) -> None:
    hidden_size, hc_mult, tokens = 8, 2, 3
    config = type(
        "Config",
        (),
        {
            "hidden_size": hidden_size,
            "hc_mult": hc_mult,
            "rms_norm_eps": 1e-6,
            "hc_eps": 1e-6,
            "hc_sinkhorn_iters": 2,
        },
    )()
    layer = DeepseekV4Layer.__new__(DeepseekV4Layer)
    nn.Module.__init__(layer)
    layer.config = config
    layer.selected_stages = frozenset({"mhc"})

    def hc_state():
        state = nn.Module()
        state.hc_fn = nn.Parameter(torch.zeros((2 + hc_mult) * hc_mult, hc_mult * hidden_size))
        state.hc_base = nn.Parameter(torch.zeros((2 + hc_mult) * hc_mult))
        state.hc_scale = nn.Parameter(torch.ones(3))
        return state

    layer.attn_hc = hc_state()
    layer.ffn_hc = hc_state()
    layer.input_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=True)
    layer.post_attention_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=True)

    attention_inputs = []

    class Attention(nn.Module):
        def forward(self, value, *, metadata):
            attention_inputs.append(value.clone())
            return value

    class MLP(nn.Module):
        def forward(self, value, *, input_ids, metadata):
            return value

    layer.self_attn = Attention()
    layer.mlp = MLP()
    calls = []

    class FakeAdapter:
        def __init__(self, kernel):
            calls.append(kernel)

        def __call__(self, hidden, residual, post_mix, res_mix, *args, **kwargs):
            return residual, post_mix, res_mix, hidden + 1

    monkeypatch.setattr(model_module, "MHCTileLangAdapter", FakeAdapter)
    hidden = torch.zeros(tokens, hidden_size, dtype=torch.bfloat16)
    residual = torch.zeros(tokens, hc_mult, hidden_size, dtype=torch.bfloat16)
    post_mix = torch.zeros(tokens, hc_mult, 1)
    res_mix = torch.zeros(tokens, hc_mult, hc_mult)

    layer(
        hidden,
        residual=residual,
        post_mix=post_mix,
        res_mix=res_mix,
        attention_metadata=object(),
        moe_metadata=object(),
    )

    assert calls == [MHCKernel.POST_PRE, MHCKernel.POST_PRE]
    torch.testing.assert_close(attention_inputs[0], hidden + 1)


@pytest.mark.gpus(1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="requires the official vLLM package and compiled TileLang kernels",
)
def test_mhc_post_official_kernel_is_bitwise_through_adapter() -> None:
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang

    args = _post_inputs("cuda")
    reference = mhc_post_tilelang(*(value.clone() for value in args))
    candidate = MHCTileLangAdapter("post")(*(value.clone() for value in args))
    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
