# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TASK-1.1.12 拆桥数值验证 · block-FP8 量化-反量化往返保真度。

只验【数值保真】(quantize->dequantize 的往返误差在 E4M3 blockwise 量级内)，
不验布局/轴序/融合(那是静态布局对照表 + logit-similarity 的职责)。
零 GPU，纯 tensor。DS4 routed-expert 走 expert_dtype=fp8 -> quantize_block_fp8
(scale_format="float32")，故这里全部用 float32 scale 分支。
"""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.quantization.block_fp8 import (
    dequantize_block_fp8,
    quantize_block_fp8,
)

# E4M3 有 3 尾数位 -> 单值分辨率 ~2^-3=12.5%；blockwise per-128x128 absmax 缩放
# 下整体能量(Frobenius)相对误差应 ~2^-3/sqrt(3) 约几个百分点。以 6% 为通过阈，
# 逐元素 max-rel 不做判据(块内近零元素相对误差天然大，非布局/数学错)。
FROBENIUS_TOL = 0.06


def _frobenius_rel_err(restored: torch.Tensor, source: torch.Tensor) -> float:
    return (
        torch.linalg.vector_norm(restored.float() - source.float())
        / torch.linalg.vector_norm(source.float())
    ).item()


def _roundtrip(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight, scale = quantize_block_fp8(source, (128, 128), scale_format="float32")
    restored = dequantize_block_fp8(weight, scale, (128, 128))
    return weight, scale, restored


def test_roundtrip_extreme_negative_and_near_zero_values() -> None:
    """已知张量含 E4M3 极值/负值/接近 0/精确 0，往返能量误差在 fp8 量级内。"""
    torch.manual_seed(0)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    source = torch.empty(256, 256)
    # 混入：块级大幅值(正/负近 fp8_max)、极小值、精确 0、常规小权重尺度。
    source[:128, :128] = torch.linspace(-fp8_max, fp8_max, 128 * 128).reshape(128, 128)
    source[:128, 128:] = torch.full((128, 128), 1e-4)
    source[128:, :128] = torch.randn(128, 128) * 0.02
    block = torch.randn(128, 128) * 0.02
    block[0, 0] = fp8_max          # 块内一个大 outlier 定 scale
    block[1, 1] = -fp8_max
    block[2, 2] = 0.0              # 精确 0 必须往返回 0
    source[128:, 128:] = block

    weight, scale, restored = _roundtrip(source)

    assert weight.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    assert scale.shape == (2, 2)  # 256/128 x 256/128
    assert torch.isfinite(restored).all(), "dequant 出现 NaN/Inf"
    # 精确 0 的元素必须还原为 0（scale 有限、量化保 0）。
    assert restored[128 + 2, 128 + 2].abs().item() == 0.0
    rel = _frobenius_rel_err(restored, source)
    assert rel < FROBENIUS_TOL, f"Frobenius rel err {rel:.4f} >= {FROBENIUS_TOL}"


@pytest.mark.parametrize(
    "shape,label",
    [
        ((512, 1024), "w1_gate_up_I_by_H"),   # 专家 gate/up: [I, H]
        ((1024, 512), "w2_down_H_by_I"),      # 专家 down:    [H, I]
    ],
)
def test_roundtrip_ds4_expert_shapes(shape: tuple[int, int], label: str) -> None:
    """DS4 routed-expert 权重形状(w1[I,H]/w2[H,I])在真实权重尺度分布下往返保真。"""
    torch.manual_seed(1234)
    # 真实权重样分布：小尺度正态 + 少量 outlier（决定每块 scale）。
    source = torch.randn(*shape) * 0.02
    outlier_mask = torch.rand(*shape) < 0.001
    source = torch.where(outlier_mask, torch.sign(source) * 5.0, source)

    weight, scale, restored = _roundtrip(source)

    assert weight.shape == source.shape, f"{label}: 量化改了形状(疑轴序/布局错)"
    assert scale.shape == (shape[0] // 128, shape[1] // 128)
    assert torch.isfinite(restored).all()
    rel = _frobenius_rel_err(restored, source)
    assert rel < FROBENIUS_TOL, f"{label}: Frobenius rel err {rel:.4f} >= {FROBENIUS_TOL}"


def test_roundtrip_is_deterministic() -> None:
    """同输入两次量化-反量化逐位相同（无随机性，可作 CI 回归基线）。"""
    source = torch.randn(128, 128) * 0.05
    _, _, a = _roundtrip(source)
    _, _, b = _roundtrip(source)
    assert torch.equal(a, b)
