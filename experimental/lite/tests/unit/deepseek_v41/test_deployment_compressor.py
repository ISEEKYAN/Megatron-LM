# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CR1/2 native deployment pool/normalization arithmetic, including odd tails."""
import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA deployment kernel oracle'
)
@pytest.mark.parametrize('ratio', [1, 2])
def test_dense_compressor_matches_native_group_program(ratio):
    import megatron.lite.primitive.kernels.deployment_compressor as _imports_deployment_compressor

    compress_norm = _imports_deployment_compressor.compress_norm
    import vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache as _imports_fused_compress_quant_cache

    fused_save_compress_norm = (
        _imports_fused_compress_quant_cache.fused_save_compress_norm
    )

    torch.manual_seed(714)
    for batch, length in [(1, 1), (1, 3), (2, 7), (2, 64)]:
        raw = torch.randn(batch, length, 512 * ratio, device='cuda')
        if ratio == 2:
            raw[:, :, 512:] *= 80
        gamma = (1 + torch.randn(512, device='cuda') * 0.01).bfloat16()
        actual = compress_norm(raw, gamma, ratio, 1e-20)
        positions = torch.arange(length, device='cuda').repeat(batch)
        requests = torch.arange(
            batch, device='cuda', dtype=torch.int32
        ).repeat_interleave(length)
        slots = requests.long() * 128 + positions
        starts = torch.arange(batch + 1, device='cuda', dtype=torch.int32) * length
        ring = torch.zeros(batch, 128, 1024, device='cuda') if ratio == 2 else None
        native = torch.empty(batch * length, 512, device='cuda', dtype=torch.bfloat16)
        fused_save_compress_norm(
            raw.reshape(-1, 512 * ratio),
            positions,
            ring,
            slots,
            starts if ratio == 2 else None,
            requests if ratio == 2 else None,
            gamma,
            1e-20,
            ratio,
            native,
        )
        expected = native.reshape(batch, length, 512)[:, ratio - 1 :: ratio]
        assert torch.equal(actual, expected), (ratio, batch, length)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires native NVFP4 paged codec'
)
@pytest.mark.parametrize('ratio', [1, 2])
def test_main_rope_codec_matches_native_bytes(ratio):
    import vllm.models.deepseek_v41.common.ops.cache_utils as _imports_cache_utils
    from megatron.lite.primitive.modules.attention import csa
    from megatron.lite.primitive.quantization.nvfp4 import quantize_main_kv

    dequantize_and_gather_k_cache = _imports_cache_utils.dequantize_and_gather_k_cache
    import vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache as _imports_fused_compress_quant_cache

    rope_quant_insert = _imports_fused_compress_quant_cache.rope_quant_insert

    torch.manual_seed(716)
    cfg = csa.CrossLayerAttentionConfig()
    latent = torch.randn(7, 512, device='cuda').bfloat16()
    positions = torch.arange(7, device='cuda')
    dims = torch.arange(0, 64, 2, device='cuda', dtype=torch.float32)
    freqs = 1 / cfg.compress_rope_theta ** (dims / 64)
    low, high = csa._yarn_find_correction_range(
        cfg.beta_fast, cfg.beta_slow, 64, cfg.compress_rope_theta, cfg.original_length
    )
    ramp = ((torch.arange(32, device='cuda') - low) / max(high - low, 1e-3)).clamp(0, 1)
    freqs = freqs / cfg.factor * ramp + freqs * (1 - ramp)
    angles = positions.float().unsqueeze(-1) * freqs
    cos_sin = torch.cat((angles.cos(), angles.sin()), -1).contiguous()
    cache = torch.zeros(1, 32, 288, dtype=torch.uint8, device='cuda')
    slots = torch.where((positions + 1) % ratio == 0, positions // ratio, -1)
    rope_quant_insert(latent, positions, cos_sin, cache, slots, ratio)
    valid = (positions + 1) % ratio == 0
    count = int(valid.sum().item())
    cp = positions[valid] // ratio * ratio
    quantized = quantize_main_kv(
        csa.rotate(
            latent[valid].unsqueeze(0), cp, cfg, ratio, output_dtype=torch.float32
        )
    )
    values = cache.flatten()[: 32 * 256].reshape(32, 256)[:count]
    scales = cache.flatten()[32 * 256 :].reshape(32, 32)[:count]
    assert torch.equal(values, quantized.packed.view(torch.uint8).reshape(count, 256))
    assert torch.equal(scales, quantized.scale.view(torch.uint8).reshape(count, 32))
    gathered = torch.empty(1, count, 512, device='cuda', dtype=torch.bfloat16)
    dequantize_and_gather_k_cache(
        gathered,
        cache,
        torch.tensor([count], device='cuda', dtype=torch.int32),
        None,
        torch.zeros(1, 1, device='cuda', dtype=torch.int32),
        32,
        0,
    )
    assert torch.equal(gathered, quantized.decoded.bfloat16())
