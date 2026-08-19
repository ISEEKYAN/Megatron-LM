from __future__ import annotations

from types import SimpleNamespace

import torch

from megatron.lite.primitive.alignment.vllm_batched_prepare import (
    _compact_fused_expert_output,
    _quantize_batched_input,
)


def test_compact_output_accepts_token_major_expert_layout() -> None:
    output = torch.full((3, 2, 1), float("nan"))
    output[:2, 0, 0] = torch.tensor([10.0, 11.0])
    output[:1, 1, 0] = 20.0

    compact = _compact_fused_expert_output(output, (2, 1))

    torch.testing.assert_close(
        compact,
        torch.tensor([[10.0], [11.0], [20.0]]),
        rtol=0,
        atol=0,
    )


def test_batched_quantization_uses_exact_vllm_ll_entrypoint(
    monkeypatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import utils

    source = torch.randn(2, 5, 128, dtype=torch.bfloat16)
    expected_quantized = torch.empty(
        10, 128, dtype=torch.float8_e4m3fn
    )
    expected_scales = torch.ones(2, 5, 1)
    calls = []

    def quantize(*args):
        calls.append(args)
        return expected_quantized, expected_scales

    monkeypatch.setattr(utils, "moe_kernel_quantize_input", quantize)
    monkeypatch.setattr(
        utils,
        "normalize_batched_scales_shape",
        lambda scales, experts: scales,
    )
    config = SimpleNamespace(
        a1_scale=None,
        quant_dtype=torch.float8_e4m3fn,
        per_act_token_quant=True,
        block_shape=[128, 128],
    )
    quantized, scales = _quantize_batched_input(
        source,
        config,
        torch.tensor([5, 5], dtype=torch.int64),
    )

    assert quantized.shape == source.shape
    assert quantized.data_ptr() == expected_quantized.data_ptr()
    assert scales is expected_scales
    assert len(calls) == 1
    assert calls[0][0].shape == (10, 128)
    assert calls[0][0].data_ptr() == source.data_ptr()
    assert calls[0][1:] == (
        None,
        torch.float8_e4m3fn,
        True,
        [128, 128],
    )
