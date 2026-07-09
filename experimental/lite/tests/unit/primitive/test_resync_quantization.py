import pytest
import torch


def test_block_fp8_emits_checkpoint_scale_grid_and_roundtrips() -> None:
    from megatron.lite.primitive.quantization.block_fp8 import (
        dequantize_block_fp8,
        quantize_block_fp8,
    )

    source = torch.linspace(-2.0, 2.0, 256 * 128).reshape(256, 128)
    weight, scale = quantize_block_fp8(source)

    assert weight.dtype == torch.float8_e4m3fn
    assert weight.shape == source.shape
    assert scale.dtype == torch.float32
    assert scale.shape == (2, 1)
    restored = dequantize_block_fp8(weight, scale)
    assert (
        torch.linalg.vector_norm(restored - source) / torch.linalg.vector_norm(source)
        < 0.03
    )


def test_block_fp8_e8m0_scale_uses_power_of_two_checkpoint_dtype() -> None:
    from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8

    source = torch.full((128, 128), 448.0)
    _, scale = quantize_block_fp8(source, scale_format="e8m0")

    assert scale.dtype == torch.float8_e8m0fnu
    assert scale.float().item() == 1.0


def test_block_fp8_rejects_non_divisible_checkpoint_shape() -> None:
    from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8

    with pytest.raises(ValueError, match="divisible"):
        quantize_block_fp8(torch.ones(129, 128))


def test_mxfp4_uses_low_then_high_nibble_and_ue8m0_scale() -> None:
    from megatron.lite.primitive.quantization.mxfp4 import (
        dequantize_mxfp4,
        quantize_mxfp4,
    )

    values = torch.tensor([0.5, -1.0, 1.5, -6.0] * 8).reshape(1, 32)
    packed, scale = quantize_mxfp4(values)

    assert packed.dtype == torch.int8
    assert packed.shape == (1, 16)
    assert packed.view(torch.uint8)[0, :2].tolist() == [0xA1, 0xF3]
    assert scale.dtype == torch.float8_e8m0fnu
    assert scale.shape == (1, 1)
    assert scale.float().item() == 1.0
    torch.testing.assert_close(dequantize_mxfp4(packed, scale), values)


def test_mxfp4_scale_ceil_and_zero_floor_match_ds4_serialization() -> None:
    from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

    source = torch.zeros(2, 32)
    source[0, 0] = 6.01
    _, scale = quantize_mxfp4(source)

    assert scale.view(torch.uint8).flatten().tolist() == [128, 1]
    assert scale.float().flatten().tolist() == [2.0, 2.0**-126]


def test_mxfp4_midpoints_round_to_even_e2m1_encoding() -> None:
    from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

    source = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] + [0.0] * 25).reshape(
        1, 32
    )
    packed, _ = quantize_mxfp4(source)
    raw = packed.view(torch.uint8)
    nibbles = torch.stack((raw & 0x0F, raw >> 4), dim=-1).flatten()

    assert nibbles[:7].tolist() == [0, 2, 2, 4, 4, 6, 6]


def test_mxfp4_rejects_non_divisible_last_dimension() -> None:
    from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

    with pytest.raises(ValueError, match="divisible"):
        quantize_mxfp4(torch.ones(2, 33))
