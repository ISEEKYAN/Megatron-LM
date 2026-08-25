from __future__ import annotations

import ast
import importlib.util
import inspect
import textwrap
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.model_executor.kernels.mhc import tilelang as mhc_tilelang

_MHC_ENTRY_NAMES = ("pre", "pre_broadcast", "post", "post_pre", "head")


def _mhc_kernel(name: str, *args, **kwargs):
    from megatron.lite.model.deepseek_v4.vllm.primitive.dense import mhc_kernel

    return mhc_kernel(name, *args, **kwargs)


def _post_inputs(
    device: str = "cpu", tokens: int = 4
) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(tokens, 128, dtype=torch.bfloat16, device=device),
        torch.randn(tokens, 4, 128, dtype=torch.bfloat16, device=device),
        torch.randn(tokens, 4, 1, dtype=torch.float32, device=device),
        torch.randn(tokens, 4, 4, dtype=torch.float32, device=device),
    )


@pytest.mark.parametrize("kernel", _MHC_ENTRY_NAMES)
@pytest.mark.parametrize("tokens", [1, 4, 32])
def test_each_mhc_call_uses_the_official_entry(
    monkeypatch: pytest.MonkeyPatch, kernel: str, tokens: int
) -> None:
    from megatron.lite.model.deepseek_v4.vllm.primitive import dense as vllm_ds4

    result = torch.tensor([7])
    official = Mock(return_value=result)
    monkeypatch.setitem(vllm_ds4._MHC_ENTRIES, kernel, official)
    if kernel == "post":
        args = _post_inputs(tokens=tokens)
    elif kernel == "head":
        args = (
            torch.zeros(tokens, 4, 128, dtype=torch.bfloat16),
            torch.zeros(4, 512),
            torch.zeros(1),
            torch.zeros(4),
            1e-6,
            1e-6,
        )
    else:
        residual = torch.zeros(tokens, 128, dtype=torch.bfloat16)
        if kernel != "pre_broadcast":
            residual = torch.zeros(tokens, 4, 128, dtype=torch.bfloat16)
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
        args = (
            _post_inputs(tokens=tokens) + pre[1:]
            if kernel == "post_pre"
            else pre
        )
    assert _mhc_kernel(kernel, *args) is result
    official.assert_called_once_with(*args)
    assert official.call_args.args[0].shape[0] == tokens


@pytest.mark.parametrize(
    "entry_name",
    ["mhc_pre_tilelang", "mhc_pre_broadcast_tilelang", "mhc_fused_post_pre_tilelang"],
)
def test_batched_tilelang_entries_have_no_recursive_singleton_fallback(
    entry_name: str,
) -> None:
    source = textwrap.dedent(inspect.getsource(getattr(mhc_tilelang, entry_name)))
    tree = ast.parse(source)
    recursive_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == entry_name
    ]
    assert not recursive_calls
    assert "VLLM_BATCH_INVARIANT" not in source


@pytest.mark.parametrize("tokens", [1, 4, 32])
def test_layer_matches_lite_unfused_pre_block_post_sequence(
    monkeypatch, tokens: int
) -> None:
    from megatron.lite.model.deepseek_v4.vllm import model as model_module
    from megatron.lite.model.deepseek_v4.vllm.model import DeepseekV4Layer

    hidden_size, hc_mult = 8, 2
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
    layer.layer_idx = 0

    def hc_state():
        state = nn.Module()
        state.fn = nn.Parameter(torch.zeros((2 + hc_mult) * hc_mult, hc_mult * hidden_size))
        state.base = nn.Parameter(torch.zeros((2 + hc_mult) * hc_mult))
        state.scale = nn.Parameter(torch.ones(3))
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
        def forward(self, value, *, input_ids):
            return value

    layer.self_attn = Attention()
    layer.mlp = MLP()
    calls: list[tuple[str, tuple[int, ...]]] = []

    def fake_kernel(kernel, *args, **kwargs):
        del kwargs
        calls.append((kernel, tuple(args[0].shape)))
        if kernel == "pre":
            streams = args[0]
            post = torch.zeros(tokens, hc_mult, 1)
            comb = torch.zeros(tokens, hc_mult, hc_mult)
            return post, comb, streams[:, 0] + 1
        if kernel == "post":
            return args[1]
        raise AssertionError(kernel)

    monkeypatch.setattr(model_module, "mhc_kernel", fake_kernel)
    streams = torch.zeros(tokens, hc_mult, hidden_size, dtype=torch.bfloat16)

    layer(
        streams,
        position_ids=torch.arange(tokens),
        attention_metadata=object(),
    )

    assert calls == [
        ("pre", (tokens, hc_mult, hidden_size)),
        ("post", (tokens, hidden_size)),
        ("pre", (tokens, hc_mult, hidden_size)),
        ("post", (tokens, hidden_size)),
    ]
    torch.testing.assert_close(attention_inputs[0], streams[:, 0] + 1)


@pytest.mark.gpus(1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="requires the official vLLM package and compiled TileLang kernels",
)
def test_mhc_post_official_kernel_is_bitwise() -> None:
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang

    args = _post_inputs("cuda")
    reference = mhc_post_tilelang(*(value.clone() for value in args))
    candidate = _mhc_kernel("post", *(value.clone() for value in args))
    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)


def _gpu_case(
    kernel: str, tokens: int
) -> tuple[tuple[torch.Tensor | float, ...], dict, int]:
    mult, hidden = 4, 128
    width = mult * hidden
    mixes = (2 + mult) * mult
    if kernel == "post":
        return _post_inputs("cuda", tokens), {}, 4
    if kernel == "head":
        return (
            (
                torch.randn(
                    tokens, mult, hidden, dtype=torch.bfloat16, device="cuda"
                ),
                torch.randn(mult, width, dtype=torch.float32, device="cuda"),
                torch.randn(1, dtype=torch.float32, device="cuda"),
                torch.randn(mult, dtype=torch.float32, device="cuda"),
                1e-6,
                1e-6,
            ),
            {},
            1,
        )

    residual_shape = (
        (tokens, hidden) if kernel == "pre_broadcast" else (tokens, mult, hidden)
    )
    fn = torch.randn(mixes, width, dtype=torch.float32, device="cuda")
    args = (
        torch.randn(*residual_shape, dtype=torch.bfloat16, device="cuda"),
        fn,
        torch.randn(3, dtype=torch.float32, device="cuda"),
        torch.randn(mixes, dtype=torch.float32, device="cuda"),
        1e-6,
        1e-6,
        1e-6,
        2.0,
        2,
    )
    kwargs = {
        "norm_weight": torch.randn(hidden, dtype=torch.bfloat16, device="cuda"),
        "norm_eps": 1e-6,
    }
    if kernel == "pre_broadcast":
        kwargs["fn_broadcast"] = (
            fn.view(-1, mult, hidden).sum(dim=1).contiguous()
        )
    return args, kwargs, 1


def _as_tuple(value) -> tuple[torch.Tensor, ...]:
    return value if isinstance(value, tuple) else (value,)


def _run_partitions(
    kernel: str,
    args: tuple[torch.Tensor | float, ...],
    kwargs: dict,
    batched_arg_count: int,
    sizes: list[int],
) -> tuple[torch.Tensor, ...]:
    outputs: list[tuple[torch.Tensor, ...]] = []
    start = 0
    for size in sizes:
        stop = start + size
        part_args = tuple(
            value[start:stop] if index < batched_arg_count else value
            for index, value in enumerate(args)
        )
        outputs.append(_as_tuple(_mhc_kernel(kernel, *part_args, **kwargs)))
        start = stop
    return tuple(torch.cat(parts, dim=0) for parts in zip(*outputs, strict=True))


@pytest.mark.gpus(1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="requires the official vLLM package and compiled TileLang kernels",
)
@pytest.mark.parametrize("kernel", ["pre", "pre_broadcast", "post", "head"])
@pytest.mark.parametrize("tokens", [1, 4, 32])
def test_mhc_is_bitwise_invariant_to_bs_composition(
    monkeypatch: pytest.MonkeyPatch, kernel: str, tokens: int
) -> None:
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.manual_seed(42)
    args, kwargs, batched_arg_count = _gpu_case(kernel, tokens)
    batched = _as_tuple(_mhc_kernel(kernel, *args, **kwargs))
    singleton = _run_partitions(
        kernel, args, kwargs, batched_arg_count, [1] * tokens
    )
    for batched_output, singleton_output in zip(batched, singleton, strict=True):
        assert torch.equal(batched_output, singleton_output)


@pytest.mark.gpus(1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="requires the official vLLM package and compiled TileLang kernels",
)
@pytest.mark.parametrize("kernel", ["pre", "pre_broadcast", "post", "head"])
def test_mhc_is_bitwise_invariant_to_ragged_token_composition(
    monkeypatch: pytest.MonkeyPatch, kernel: str
) -> None:
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.manual_seed(43)
    sizes = [3, 5, 9]
    args, kwargs, batched_arg_count = _gpu_case(kernel, sum(sizes))
    batched = _as_tuple(_mhc_kernel(kernel, *args, **kwargs))
    ragged = _run_partitions(kernel, args, kwargs, batched_arg_count, sizes)
    for batched_output, ragged_output in zip(batched, ragged, strict=True):
        assert torch.equal(batched_output, ragged_output)
