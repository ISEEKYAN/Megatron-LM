# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""wo_a_dtype="bf16": the reference-inference BF16 attention output stage."""
import os

import pytest
import torch


def _require_cuda(what):
    if os.environ.get("MEGATRON_LITE_REQUIRE_CUDA_TESTS") == "1":
        assert torch.cuda.is_available(), what
    elif not torch.cuda.is_available():
        pytest.skip(what)


def _bf16_wo_a_config():
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
    from test_w4a8_fp32 import tiny_config

    cfg = tiny_config().to_hf_dict()
    cfg['text_config']['wo_a_dtype'] = 'bf16'
    return DeepseekV41Config(cfg)


def _build(config, **kwargs):
    from megatron.lite.model.deepseek_v41.lite import protocol

    impl = dict(device='cpu', dtype=torch.bfloat16, quantized=True, w4a8_experts=True)
    impl.update(kwargs)
    return protocol.build_model(config, impl_cfg=protocol.ImplConfig(**impl)).chunks[0]


def test_bf16_wo_a_exports_plain_weight_and_loads_release_fp8(v41_core_te, tmp_path):
    from megatron.lite.model.deepseek_v41.lite import checkpoint
    from megatron.lite.primitive.ckpt.hf_weights import (
        SafeTensorReader,
        load_bound_weight,
    )
    from safetensors.torch import save_file

    model = _build(_bf16_wo_a_config(), deployment_math=True)
    layer = next(i for i, m in enumerate(model.layers) if m is not None)
    attention = model.layers[layer].attn
    assert not attention.wo_a.fp8 and attention.wq_a.fp8
    key = f'layers.{layer}.attn.wo_a.weight'
    assert model.tensor_bindings[key].encoding is None
    # The release scale stays bound so an FP8 release checkpoint still loads.
    assert model.tensor_bindings[key[:-6] + 'scale'].role == 'scale'

    exported = dict(
        checkpoint.export_checkpoint(model, cpu=True, include_archival=False)
    )
    assert exported[key].is_floating_point()
    assert key[:-6] + 'scale' not in exported
    assert f'layers.{layer}.attn.wq_a.scale' in exported

    weight = attention.wo_a.weight
    codes = (torch.randn(weight.shape) * 50).to(torch.float8_e4m3fn)
    exponents = torch.randint(115, 130, (weight.shape[0] // 32, weight.shape[1] // 32))
    scale = exponents.to(torch.uint8).view(torch.float8_e8m0fnu)
    save_file(
        {key: codes, key[:-6] + 'scale': scale}, str(tmp_path / 'model.safetensors')
    )
    with SafeTensorReader(str(tmp_path)) as reader:
        decoded = load_bound_weight(
            reader, key, checkpoint._SPEC, output_dtype=torch.float32
        )
    expected = codes.float() * torch.exp2(exponents.float() - 127).repeat_interleave(
        32, 0
    ).repeat_interleave(32, 1)
    assert torch.equal(decoded, expected)
    # Every decoded value is a BF16 value, so the rollout's BF16 copy is exact.
    assert torch.equal(decoded.bfloat16().float(), decoded)


def test_bf16_wo_a_reference_rounds_once_per_group():
    from megatron.lite.primitive.modules.attention import csa

    torch.manual_seed(3)
    config = csa.CrossLayerAttentionConfig(heads=8, groups=2, o_rank=64)
    b, length = 2, 5
    output = torch.randn(b, length, config.heads, config.head_dim).bfloat16()
    output.requires_grad_()
    positions = torch.arange(100, 100 + length)
    width = config.heads * config.head_dim // config.groups
    weight = (torch.randn(config.groups * config.o_rank, width) / 64).requires_grad_()

    actual = csa._bf16_wo_a(output, positions, config, 2, weight)
    rotated = csa.rotate(
        output,
        positions,
        config,
        2,
        inverse=True,
        output_dtype=torch.bfloat16,
        deployment_math=True,
    ).reshape(b, length, config.groups, width)
    groups = weight.reshape(config.groups, config.o_rank, width)
    expected = torch.cat(
        [
            (rotated[:, :, i].float() @ groups[i].bfloat16().float().T).bfloat16()
            for i in range(config.groups)
        ],
        -1,
    )
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)
    actual.float().square().sum().backward()
    assert torch.isfinite(output.grad).all() and output.grad.abs().sum() > 0
    assert torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("ratio", [0, 2])
def test_bf16_wo_a_matches_rollout_batch_invariant_o_proj(ratio, monkeypatch):
    _require_cuda("vLLM BF16 wo_a")
    from megatron.lite.primitive.modules.attention import csa
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.manual_seed(11 + ratio)
    config = csa.CrossLayerAttentionConfig(heads=16, groups=2, o_rank=1024)
    length, heads, dim = 300, config.heads, config.head_dim
    width = heads * dim // config.groups
    output = torch.randn(1, length, heads, dim, device="cuda").bfloat16()
    positions = torch.arange(7000, 7000 + length, device="cuda")
    master = torch.randn(config.groups * config.o_rank, width, device="cuda") / 64
    cache = csa.rope_cos_sin_cache(int(positions.max()) + 1, config, ratio, "cuda")
    wo_a = torch.nn.Module()
    wo_a.register_buffer("weight", master.bfloat16())

    def rollout(rows):
        return deep_gemm_fp8_o_proj(
            output[0, :rows].clone(),
            positions[:rows],
            cache,
            wo_a,
            torch.nn.Identity(),
            n_groups=config.groups,
            heads_per_group=heads // config.groups,
            nope_dim=dim - config.rope_dim,
            rope_dim=config.rope_dim,
            o_lora_rank=config.o_rank,
            einsum_recipe=(1, 1, 128),
            tma_aligned_scales=True,
        )

    leaf = output.clone().requires_grad_()
    weight = master.clone().requires_grad_()
    trained = csa._bf16_wo_a(leaf, positions, config, ratio, weight)
    for rows in (1, 7, length):
        expected = rollout(rows).reshape(rows, -1)
        assert torch.equal(trained[0, :rows], expected), rows
    # Gradients follow the deployment-order rotation and the FP32 GEMM VJP
    # (cuBLAS reduction order, so compared with a tolerance).
    trained.float().square().sum().backward()
    reference_leaf = output.clone().requires_grad_()
    rotated = csa.rotate(
        reference_leaf,
        positions,
        config,
        ratio,
        inverse=True,
        output_dtype=torch.bfloat16,
        deployment_math=True,
    ).reshape(1, length, config.groups, width)
    grad = (
        (2 * trained.detach().float()).bfloat16().reshape(1, length, config.groups, -1)
    )
    groups = master.bfloat16().reshape(config.groups, config.o_rank, width)
    d_rotated = torch.einsum('blgr,grk->blgk', grad.float(), groups.float()).bfloat16()
    rotated.backward(d_rotated)
    torch.testing.assert_close(
        leaf.grad.float(), reference_leaf.grad.float(), rtol=2e-2, atol=1e-3
    )
