from __future__ import annotations

import importlib.util
import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.lite.model.deepseek_v4.vllm.primitive.moe import grouped as vllm_grouped_moe
from megatron.lite.primitive.modules.experts import swiglu_with_probs


def _reference(
    hidden: torch.Tensor,
    counts: tuple[int, ...],
    limit: float,
    w13: tuple[torch.Tensor, ...],
    w2: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    outputs = []
    offset = 0
    for count, fc1, fc2 in zip(counts, w13, w2, strict=True):
        selected = hidden[offset : offset + count]
        gate_up = F.linear(selected, fc1)
        outputs.append(F.linear(swiglu_with_probs(gate_up, None, limit), fc2))
        offset += count
    return torch.cat(outputs)


class _TorchGroupedAdapter:
    """CPU-only test double for the production TE grouped GEMM adapter."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @staticmethod
    def _parts(value, counts):
        return torch.split(value, counts, dim=0)

    def forward(self, value, weights, counts):
        self.calls.append("forward")
        return torch.cat(
            [
                F.linear(part, weight)
                for part, weight in zip(
                    self._parts(value, counts), weights, strict=True
                )
            ]
        )

    def dgrad(self, grad_output, weights, counts):
        self.calls.append("dgrad")
        return torch.cat(
            [
                F.linear(part, weight.t())
                for part, weight in zip(
                    self._parts(grad_output, counts), weights, strict=True
                )
            ]
        )

    def wgrad(self, value, grad_output, weights, counts):
        self.calls.append("wgrad")
        return tuple(
            grad_part.t() @ value_part
            for value_part, grad_part in zip(
                self._parts(value, counts),
                self._parts(grad_output, counts),
                strict=True,
            )
        )


def test_padding_layout_reuses_ragged_zero_expert_metadata(monkeypatch) -> None:
    vllm_grouped_moe._LAYOUT_CACHE.clear()
    monkeypatch.setattr(vllm_grouped_moe, "_m_alignment", lambda: 128)
    counts = (3, 0, 129)

    first = vllm_grouped_moe._get_forward_layout(counts, torch.device("cpu"))
    second = vllm_grouped_moe._get_forward_layout(counts, torch.device("cpu"))

    assert first is second
    assert first.padded_counts == (128, 0, 256)
    assert first.m_indices.data_ptr() == second.m_indices.data_ptr()
    assert first.valid_rows is not None
    assert first.valid_rows.data_ptr() == second.valid_rows.data_ptr()
    assert first.m_indices.bincount(minlength=3).tolist() == [128, 0, 256]


def test_grouped_weight_pack_cache_hits_and_invalidates_on_version(
    monkeypatch,
) -> None:
    calls = []

    def pack(weights):
        weights = tuple(weights)
        calls.append(tuple(weight._version for weight in weights))
        return SimpleNamespace(
            qweight=torch.empty(len(weights), 1),
            scales=torch.ones(len(weights), 1, dtype=torch.float32),
            cache_key=tuple(
                vllm_grouped_moe._weight_cache_key(weight) for weight in weights
            ),
        )

    monkeypatch.setattr(vllm_grouped_moe, "pack_grouped_block_fp8_weight", pack)
    cache = vllm_grouped_moe._GroupedWeightPackCache()
    weights = tuple(torch.nn.Parameter(torch.randn(2, 2)) for _ in range(3))

    first = cache.get(weights)
    assert cache.get(weights) is first
    assert len(calls) == 1

    with torch.no_grad():
        weights[1].add_(1)
    second = cache.get(weights)
    assert second is not first
    assert len(calls) == 2


def test_scale_validation_follows_runtime_deepgemm_mode(monkeypatch) -> None:
    non_power_of_two = torch.tensor([1.5], dtype=torch.float32)
    monkeypatch.setattr(vllm_grouped_moe, "_deep_gemm_uses_e8m0", lambda: False)
    vllm_grouped_moe._require_power_of_two_scales(
        "hopper float32", non_power_of_two
    )

    monkeypatch.setattr(vllm_grouped_moe, "_deep_gemm_uses_e8m0", lambda: True)
    vllm_grouped_moe._require_power_of_two_scales(
        "blackwell ue8m0", torch.ones(1, dtype=torch.float32)
    )


def test_packed_scale_validation_is_debug_only(monkeypatch) -> None:
    cache = vllm_grouped_moe._GroupedWeightPackCache()
    weights = (torch.nn.Parameter(torch.randn(2, 2)),)
    packed = SimpleNamespace(
        qweight=torch.empty(1),
        scales=torch.ones(1),
        cache_key=tuple(
            vllm_grouped_moe._weight_cache_key(weight) for weight in weights
        ),
    )
    monkeypatch.setattr(
        vllm_grouped_moe, "pack_grouped_block_fp8_weight", lambda _weights: packed
    )
    monkeypatch.setattr(
        vllm_grouped_moe,
        "_require_power_of_two_scales",
        lambda *_args: (_ for _ in ()).throw(AssertionError("debug validation ran")),
    )

    monkeypatch.delenv("MLITE_VALIDATE_PACKED_SCALES", raising=False)
    assert cache.get(weights) is packed

    cache.clear()
    monkeypatch.setenv("MLITE_VALIDATE_PACKED_SCALES", "1")
    with pytest.raises(AssertionError, match="debug validation ran"):
        cache.get(weights)


@pytest.mark.parametrize("num_experts", [1, 5, 9])
def test_forward_launch_contract_is_constant_across_experts(
    monkeypatch, num_experts: int
) -> None:
    from vllm.utils import deep_gemm

    counts = tuple(1 if expert % 2 == 0 else 0 for expert in range(num_experts))
    rows = sum(counts)
    hidden = torch.randn(rows, 128, dtype=torch.bfloat16)
    w13 = tuple(torch.randn(256, 128, dtype=torch.bfloat16) for _ in counts)
    w2 = tuple(torch.randn(128, 128, dtype=torch.bfloat16) for _ in counts)
    launches = []
    packed_expert_counts = []

    def grouped_gemm(_activation, weight, output, m_indices):
        launches.append(m_indices)
        assert weight[0].shape[0] == num_experts
        output.zero_()

    def packed(weights):
        packed_expert_counts.append(len(weights))
        return SimpleNamespace(
            qweight=torch.empty(len(weights), 1),
            scales=torch.empty(len(weights), 1),
        )

    monkeypatch.setattr(
        deep_gemm, "m_grouped_fp8_gemm_nt_contiguous", grouped_gemm
    )
    monkeypatch.setattr(
        vllm_grouped_moe,
        "_vllm_quantize_contiguous_input",
        lambda value: (value, torch.ones(1)),
    )
    monkeypatch.setattr(
        vllm_grouped_moe,
        "_vllm_silu_mul_quant",
        lambda _value, *, output, swiglu_limit: (output, torch.ones(1)),
    )
    monkeypatch.setattr(vllm_grouped_moe, "_m_alignment", lambda: 128)
    monkeypatch.setattr(vllm_grouped_moe._PACKED_WEIGHT_CACHE, "get", packed)

    output = vllm_grouped_moe._vllm_grouped_forward(
        hidden, counts, 0.0, w13, w2
    )

    assert output.shape == hidden.shape
    assert len(launches) == 2
    assert packed_expert_counts == [num_experts, num_experts]


def test_grouped_backward_adapter_is_fail_closed(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "transformer_engine.pytorch.cpp_extensions":
            raise ModuleNotFoundError("forced unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(RuntimeError, match="no per-expert fallback"):
        vllm_grouped_moe._TEGroupedGemmAdapter()


def test_te_adapter_layouts_match_ragged_reference() -> None:
    counts = (2, 0, 1)
    value = torch.randn(sum(counts), 4)
    weights = tuple(torch.randn(3, 4) for _ in counts)
    grad_output = torch.randn(sum(counts), 3)
    launches = []

    def grouped_gemm(
        a,
        b,
        output,
        _quantizers,
        _dtype,
        *,
        layout,
        m_splits=None,
        **_kwargs,
    ):
        launches.append(layout)
        if layout == "TN":
            output[0].copy_(
                torch.cat(
                    [
                        F.linear(part, weight)
                        for part, weight in zip(b, a, strict=True)
                    ]
                )
            )
        elif layout == "NN":
            output[0].copy_(
                torch.cat(
                    [
                        F.linear(part, weight.t())
                        for part, weight in zip(b, a, strict=True)
                    ]
                )
            )
        else:
            assert layout == "NT"
            for destination, input_part, grad_part in zip(
                output, a, b, strict=True
            ):
                destination.copy_(grad_part.t() @ input_part)
        assert m_splits == list(counts)

    adapter = vllm_grouped_moe._TEGroupedGemmAdapter.__new__(
        vllm_grouped_moe._TEGroupedGemmAdapter
    )
    adapter._gemm = grouped_gemm

    torch.testing.assert_close(
        adapter.forward(value, weights, counts),
        _TorchGroupedAdapter().forward(value, weights, counts),
    )
    torch.testing.assert_close(
        adapter.dgrad(grad_output, weights, counts),
        _TorchGroupedAdapter().dgrad(grad_output, weights, counts),
    )
    expected_wgrad = _TorchGroupedAdapter().wgrad(
        value, grad_output, weights, counts
    )
    for actual, expected in zip(
        adapter.wgrad(value, grad_output, weights, counts),
        expected_wgrad,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    assert launches == ["TN", "NN", "NT"]


def test_grouped_moe_source_has_no_expert_chunk_backward_fallback() -> None:
    source = inspect.getsource(vllm_grouped_moe)
    backward_source = inspect.getsource(
        vllm_grouped_moe.VLLMGroupedMoEWithBF16Backward.backward
    )
    forward_source = inspect.getsource(vllm_grouped_moe._vllm_grouped_forward)

    assert "_EXPERTS_PER_FORWARD_GROUP" not in source
    assert "_BACKWARD_CHUNK_ROWS" not in source
    assert "torch.autograd.grad" not in backward_source
    assert "F.linear" not in backward_source
    assert "for expert_start" not in forward_source
    assert ".item()" not in forward_source


def test_deepep_dispatch_keeps_expert_counts_on_host() -> None:
    from megatron.lite.primitive.modules.dispatcher import _DeepEPDispatch

    class _Buffer:
        @staticmethod
        def get_dispatch_layout(*_args, **_kwargs):
            return None, None, None, None, None

        @staticmethod
        def dispatch(hidden, *, topk_idx, topk_weights, **_kwargs):
            return hidden, topk_idx, topk_weights, [2, 1], object(), None

    hidden = torch.randn(3, 4)
    topk_indices = torch.tensor([[0], [0], [1]], dtype=torch.int64)
    topk_scores = torch.ones(3, 1)
    *_, counts, _handle = _DeepEPDispatch.apply(
        _Buffer(),
        hidden,
        topk_indices,
        topk_scores,
        2,
        False,
        False,
    )
    assert counts.device.type == "cpu"
    assert counts.tolist() == [2, 1]


def test_grouped_moe_requires_host_counts_without_cuda_roundtrip() -> None:
    from megatron.lite.model.deepseek_v4.vllm.primitive.moe.module import (
        _VLLMVisibleExperts,
    )
    from megatron.lite.primitive.modules.dispatcher import TokenDispatcher

    source = inspect.getsource(_VLLMVisibleExperts.forward)
    assert ".cpu()" not in source
    assert ".tolist()" not in source
    finish_source = inspect.getsource(TokenDispatcher._finish_deepep_dispatch)
    assert ".item()" not in finish_source
    assert ".cpu()" not in finish_source


@pytest.mark.parametrize(
    ("scale_format_name", "expected_quantizer", "expected_use_ue8m0"),
    [
        ("FLOAT32", "float32", False),
        ("FLOAT32_CEIL_UE8M0", "float32", True),
        ("UE8M0", "packed", True),
    ],
)
def test_contiguous_input_quant_matches_vllm_scale_format(
    monkeypatch,
    scale_format_name: str,
    expected_quantizer: str,
    expected_use_ue8m0: bool,
) -> None:
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    calls = []

    def float32_quant(value, group_size, **kwargs):
        calls.append(("float32", group_size, kwargs))
        return value, torch.ones(1)

    def packed_quant(value, group_size, **kwargs):
        calls.append(("packed", group_size, kwargs))
        return value, torch.ones(1)

    monkeypatch.setattr(
        DeepGemmQuantScaleFMT,
        "from_oracle",
        staticmethod(lambda: getattr(DeepGemmQuantScaleFMT, scale_format_name)),
    )
    monkeypatch.setattr(fp8_utils, "per_token_group_quant_fp8", float32_quant)
    monkeypatch.setattr(
        fp8_utils,
        "per_token_group_quant_fp8_packed_for_deepgemm",
        packed_quant,
    )

    value = torch.randn(2, 128, dtype=torch.bfloat16)
    vllm_grouped_moe._vllm_quantize_contiguous_input(value)

    assert len(calls) == 1
    quantizer, group_size, kwargs = calls[0]
    assert quantizer == expected_quantizer
    assert group_size == 128
    assert kwargs["use_ue8m0"] is expected_use_ue8m0


def test_vllm_visible_silu_quant_has_no_layout_dependent_fallback(monkeypatch):
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    calls = []

    def fused(value, **kwargs):
        calls.append((value, kwargs))
        return kwargs["output_q"], torch.ones(1)

    monkeypatch.setattr(
        DeepGemmQuantScaleFMT,
        "from_oracle",
        staticmethod(lambda: DeepGemmQuantScaleFMT.FLOAT32),
    )
    monkeypatch.setattr(fp8_utils, "fused_silu_mul_per_token_group_quant_fp8", fused)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT_KERNEL_LIB", raising=False)
    value = torch.randn(2, 256, dtype=torch.bfloat16)
    output = torch.empty(2, 128, dtype=torch.float8_e4m3fn)

    quantized, _scales = vllm_grouped_moe._vllm_silu_mul_quant(
        value, output=output, swiglu_limit=0.0
    )

    assert quantized is output
    assert len(calls) == 1
    assert calls[0][1]["masked_m"] is None


def test_vllm_visible_silu_quant_preserves_ds4_clamp(monkeypatch):
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    calls = []

    def fused(value, **kwargs):
        calls.append((value, kwargs))
        return kwargs["output_q"], torch.ones(1)

    monkeypatch.setattr(
        DeepGemmQuantScaleFMT,
        "from_oracle",
        staticmethod(lambda: DeepGemmQuantScaleFMT.UE8M0),
    )
    monkeypatch.setattr(fp8_utils, "fused_silu_mul_per_token_group_quant_fp8", fused)
    value = torch.randn(2, 256, dtype=torch.bfloat16)
    output = torch.empty(2, 128, dtype=torch.float8_e4m3fn)

    quantized, _scales = vllm_grouped_moe._vllm_silu_mul_quant(
        value, output=output, swiglu_limit=10.0
    )

    assert quantized is output
    assert len(calls) == 1
    assert calls[0][1]["clamp_limit"] == 10.0
    assert calls[0][1]["masked_m"] is None


def test_grouped_moe_preserves_clamped_forward_and_bf16_master_vjp(
    monkeypatch,
) -> None:
    torch.manual_seed(7)
    counts = (2, 1)
    limit = 10.0
    hidden = (torch.randn(3, 4) * 8).requires_grad_(True)
    w13 = tuple((torch.randn(6, 4) * 3).requires_grad_(True) for _ in counts)
    w2 = tuple(torch.randn(4, 3).requires_grad_(True) for _ in counts)
    adapter = _TorchGroupedAdapter()
    monkeypatch.setattr(vllm_grouped_moe, "_vllm_grouped_forward", _reference)
    monkeypatch.setattr(
        vllm_grouped_moe, "_get_grouped_backward_adapter", lambda: adapter
    )
    output = vllm_grouped_moe.VLLMGroupedMoEWithBF16Backward.apply(
        hidden,
        counts,
        limit,
        *w13,
        *w2,
    )
    expected = _reference(hidden, counts, limit, w13, w2)
    unclamped = _reference(hidden, counts, 0.0, w13, w2)
    assert torch.equal(output, expected)
    assert not torch.allclose(output, unclamped)

    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    actual_grads = (hidden.grad, *(weight.grad for weight in w13 + w2))

    ref_hidden = hidden.detach().requires_grad_(True)
    ref_w13 = tuple(weight.detach().requires_grad_(True) for weight in w13)
    ref_w2 = tuple(weight.detach().requires_grad_(True) for weight in w2)
    ref_output = _reference(ref_hidden, counts, limit, ref_w13, ref_w2)
    expected_grads = torch.autograd.grad(
        ref_output,
        (ref_hidden, *ref_w13, *ref_w2),
        grad_output,
    )
    for actual, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected_grad)
    assert adapter.calls == ["forward", "wgrad", "dgrad", "dgrad", "wgrad"]


@pytest.mark.parametrize("num_experts", [2, 8])
def test_backward_launch_contract_does_not_scale_with_experts(
    monkeypatch, num_experts: int
) -> None:
    counts = tuple(1 if expert % 2 == 0 else 0 for expert in range(num_experts))
    hidden = torch.randn(sum(counts), 4, requires_grad=True)
    w13 = tuple(torch.randn(6, 4, requires_grad=True) for _ in counts)
    w2 = tuple(torch.randn(4, 3, requires_grad=True) for _ in counts)
    adapter = _TorchGroupedAdapter()
    monkeypatch.setattr(vllm_grouped_moe, "_vllm_grouped_forward", _reference)
    monkeypatch.setattr(
        vllm_grouped_moe, "_get_grouped_backward_adapter", lambda: adapter
    )

    output = vllm_grouped_moe.VLLMGroupedMoEWithBF16Backward.apply(
        hidden, counts, 0.0, *w13, *w2
    )
    output.sum().backward()

    assert adapter.calls == ["forward", "wgrad", "dgrad", "dgrad", "wgrad"]


def test_grouped_backward_matches_ragged_zero_expert_reference(
    monkeypatch,
) -> None:
    torch.manual_seed(19)
    counts = (2, 0, 3)
    hidden = torch.randn(sum(counts), 4, requires_grad=True)
    w13 = tuple(torch.randn(6, 4, requires_grad=True) for _ in counts)
    w2 = tuple(torch.randn(4, 3, requires_grad=True) for _ in counts)
    adapter = _TorchGroupedAdapter()
    monkeypatch.setattr(vllm_grouped_moe, "_vllm_grouped_forward", _reference)
    monkeypatch.setattr(
        vllm_grouped_moe, "_get_grouped_backward_adapter", lambda: adapter
    )
    output = vllm_grouped_moe.VLLMGroupedMoEWithBF16Backward.apply(
        hidden, counts, 10.0, *w13, *w2
    )
    grad_output = torch.randn_like(output)
    actual = torch.autograd.grad(output, (hidden, *w13, *w2), grad_output)

    ref_hidden = hidden.detach().requires_grad_(True)
    ref_w13 = tuple(weight.detach().requires_grad_(True) for weight in w13)
    ref_w2 = tuple(weight.detach().requires_grad_(True) for weight in w2)
    expected = torch.autograd.grad(
        _reference(ref_hidden, counts, 10.0, ref_w13, ref_w2),
        (ref_hidden, *ref_w13, *ref_w2),
        grad_output,
    )

    for actual_grad, expected_grad in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_all_zero_experts_backward_needs_no_grouped_backend(monkeypatch) -> None:
    counts = (0, 0)
    hidden = torch.empty(0, 4, requires_grad=True)
    w13 = tuple(torch.randn(6, 4, requires_grad=True) for _ in counts)
    w2 = tuple(torch.randn(4, 3, requires_grad=True) for _ in counts)
    monkeypatch.setattr(vllm_grouped_moe, "_vllm_grouped_forward", _reference)
    monkeypatch.setattr(
        vllm_grouped_moe,
        "_get_grouped_backward_adapter",
        lambda: pytest.fail("zero-token backward must not create an adapter"),
    )

    output = vllm_grouped_moe.VLLMGroupedMoEWithBF16Backward.apply(
        hidden, counts, 0.0, *w13, *w2
    )
    output.sum().backward()

    assert hidden.grad is not None and hidden.grad.numel() == 0
    assert all(
        weight.grad is not None and not weight.grad.any() for weight in w13 + w2
    )


def test_visible_experts_forward_preserves_model_clamp(monkeypatch) -> None:
    from torch import nn

    from megatron.lite.model.deepseek_v4.vllm.primitive.moe import module as moe_module

    calls = []

    class _Grouped:
        @staticmethod
        def apply(
            hidden_states,
            tokens_per_expert,
            swiglu_limit,
            *weights,
        ):
            calls.append((tokens_per_expert, swiglu_limit, weights))
            return hidden_states

    class _Weights(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight0 = nn.Parameter(torch.ones(4, 4))

    experts = moe_module._VLLMVisibleExperts.__new__(
        moe_module._VLLMVisibleExperts
    )
    nn.Module.__init__(experts)
    experts.num_local_experts = 1
    experts.swiglu_limit = 10.0
    experts.fc1 = _Weights()
    experts.fc2 = _Weights()
    monkeypatch.setattr(moe_module, "VLLMGroupedMoEWithBF16Backward", _Grouped)
    monkeypatch.setattr(
        moe_module,
        "bind_source_scale_to_visible_weight",
        lambda _owner, _name, weight: weight,
    )

    hidden = torch.randn(2, 4)
    counts = torch.tensor([2], dtype=torch.int32)
    host_counts = [2]
    with pytest.raises(
        ValueError, match="dispatcher-provided host expert counts"
    ):
        experts(hidden, counts)
    assert (
        experts(hidden, counts, tokens_per_expert_list=host_counts) is hidden
    )
    assert len(calls) == 1
    assert calls[0][0] == (2,)
    assert calls[0][1] == 10.0


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("vllm") is None,
    reason="requires CUDA and vLLM grouped DeepGEMM",
)
def test_real_grouped_deepgemm_forward_has_bf16_master_vjp() -> None:
    from vllm.utils.deep_gemm import (
        DeepGemmQuantScaleFMT,
        is_deep_gemm_e8m0_used,
    )

    is_deep_gemm_e8m0_used()
    DeepGemmQuantScaleFMT.init_oracle_cache()
    torch.manual_seed(11)
    counts = (2, 1)
    limit = 10.0
    hidden = ((torch.randn(3, 128, device="cuda", dtype=torch.bfloat16) * 8)).requires_grad_(True)
    w13 = tuple(
        torch.nn.Parameter(
            torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) * 3
        )
        for _ in counts
    )
    w2 = tuple(
        torch.nn.Parameter(
            torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        )
        for _ in counts
    )
    padded, _layout = vllm_grouped_moe._pad_expert_rows(hidden.detach(), counts)
    _input_q, input_scale = vllm_grouped_moe._vllm_quantize_contiguous_input(
        padded
    )
    packed_w13 = vllm_grouped_moe._PACKED_WEIGHT_CACHE.get(w13)
    for name, scale in (
        ("input", input_scale),
        ("weight", packed_w13.scales),
    ):
        if scale.dtype == torch.float32 and vllm_grouped_moe._deep_gemm_uses_e8m0():
            invalid = (scale.contiguous().view(torch.int32) & 0x807FFFFF) != 0
            assert not bool(invalid.any().item()), (
                f"{name} scale is not UE8M0: "
                f"dtype={scale.dtype} shape={tuple(scale.shape)} stride={scale.stride()}"
            )
        elif scale.dtype != torch.float32:
            assert scale.dtype in (torch.int32, torch.uint8, torch.float8_e8m0fnu)
    output = vllm_grouped_moe.VLLMGroupedMoEWithBF16Backward.apply(
        hidden,
        counts,
        limit,
        *w13,
        *w2,
    )
    assert torch.isfinite(output).all()

    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    actual_grads = (hidden.grad, *(weight.grad for weight in w13 + w2))

    ref_hidden = hidden.detach().requires_grad_(True)
    ref_w13 = tuple(weight.detach().requires_grad_(True) for weight in w13)
    ref_w2 = tuple(weight.detach().requires_grad_(True) for weight in w2)
    ref_output = _reference(ref_hidden, counts, limit, ref_w13, ref_w2)
    expected_grads = torch.autograd.grad(
        ref_output,
        (ref_hidden, *ref_w13, *ref_w2),
        grad_output,
    )
    for actual, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected_grad, rtol=0, atol=0)
