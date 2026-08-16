from __future__ import annotations

import pytest
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


@pytest.mark.gpus(1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_quantization_uses_2d_cuda_kernel_per_expert() -> None:
    torch.manual_seed(43)
    source = torch.randn(
        2,
        5,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    quantized, scales = _quantize_batched_input(
        source,
        torch.float8_e4m3fn,
        [128, 128],
        torch.tensor([5, 5], device=source.device, dtype=torch.int64),
    )

    assert quantized.shape == source.shape
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.shape == (2, 5, 1)
    assert scales.dtype == torch.float32
    assert scales.stride() == (5, 1, 5)
    reconstructed = quantized.float() * scales
    torch.testing.assert_close(
        reconstructed,
        source.float(),
        rtol=0.15,
        atol=0.15,
    )
