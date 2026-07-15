# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TASK-1.1.12 拆桥数值验证 step-3(CPU 版) · block-scale 逐块对位。

验证 mlite 导出的 fp8 权重 + block-scale 网格,被 vLLM 的 block-fp8 合同**逐块对位
正确消费**——即 mlite 摆在 ``scale[i,j]`` 的那块 descale,正是 vLLM 读 ``scale[i,j]``
去缩放权重块 ``[i*128:(i+1)*128, j*128:(j+1)*128]`` 的那一块(朝向/行列一致)。

这是前面 6 炮 async-GPU 探针撞架构墙后,把「block-scale 逐块对位」残留搬到纯 CPU 的
决定性证据(零 GPU,可作 CI 回归)。判据: 对位对 -> vLLM-合同 dequant ≈ 原权重(fp8 量级);
对位错(转置/错格) -> 坍塌或形状断言失败。

vLLM 合同来源(逐行核对): ``vllm/model_executor/layers/quantization/utils/fp8_utils.py``
``w8a8_block_fp8_matmul``: weight ``B[N,K]``, block-scale ``Bs[cdiv(N,block_n), cdiv(K,block_k)]``,
断言 ``cdiv(N,block_n)==Bs.shape[0]`` 且 ``cdiv(K,block_k)==Bs.shape[1]``; dequant
``B_deq[n,k] = B_fp8[n,k] * Bs[n//block_n, k//block_k]``。N=output(行), K=input(列)。
"""

from __future__ import annotations

import math

import pytest
import torch

from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8

FROBENIUS_TOL = 0.06  # fp8 e4m3 blockwise 量级


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _vllm_block_dequant(
    fp8: torch.Tensor, scale: torch.Tensor, block=(128, 128)
) -> torch.Tensor:
    """照 vLLM w8a8_block_fp8_matmul 合同逐块消费 block-scale(独立于 mlite dequant)。

    强制 vLLM 的 scale 网格形状断言 + 逐块 descale 索引 ``scale[n//bn, k//bk]``。
    """
    block_n, block_k = block
    N, K = fp8.shape
    # vLLM 合同硬断言(误配=转置网格会在此炸)。
    assert scale.shape[0] == _cdiv(N, block_n), (
        f"scale rows {scale.shape[0]} != cdiv(N={N},{block_n})={_cdiv(N, block_n)}"
    )
    assert scale.shape[1] == _cdiv(K, block_k), (
        f"scale cols {scale.shape[1]} != cdiv(K={K},{block_k})={_cdiv(K, block_k)}"
    )
    expanded = scale.float().repeat_interleave(block_n, dim=0).repeat_interleave(
        block_k, dim=1
    )[:N, :K]
    return fp8.float() * expanded


def _frob(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.linalg.vector_norm(a.float() - b.float())
            / torch.linalg.vector_norm(b.float())).item()


# DS4 routed-expert 真实形状: w13(gate/up 融合)=[2I,H], w2(down)=[H,I]; I≠H 使转置必被抓。
@pytest.mark.parametrize(
    "N,K,label",
    [
        (512, 1024, "w13_2I_by_H"),   # 融合 gate/up: 行=2I(output), 列=H(input)
        (1024, 512, "w2_H_by_I"),     # down:        行=H(output), 列=I(input)
        (256, 4096, "w13_small_I"),   # 非方,块网格 [2,32]
    ],
)
def test_mlite_scale_grid_paired_correctly_under_vllm_contract(N, K, label):
    """mlite 量化的 (fp8, scale) 过 vLLM 合同 dequant ≈ 原权重(fp8 量级)= 逐块对位对。"""
    torch.manual_seed(20260715)
    W = torch.randn(N, K) * 0.02
    W = torch.where(torch.rand(N, K) < 0.001, torch.sign(W) * 5.0, W)  # outlier 定 scale

    fp8, scale = quantize_block_fp8(W, (128, 128), scale_format="float32")

    # ① scale 网格形状 == vLLM 合同期望(与静态布局表一致)。
    assert scale.shape == (_cdiv(N, 128), _cdiv(K, 128)), (scale.shape, label)

    # ② 用 vLLM 合同逐块 dequant -> 对回原权重 fp8 容差内 = 对位对。
    deq = _vllm_block_dequant(fp8, scale, (128, 128))
    rel = _frob(deq, W)
    assert rel < FROBENIUS_TOL, f"{label}: vLLM-contract dequant rel {rel:.4f} >= {FROBENIUS_TOL}"


def test_mispaired_transposed_scale_grid_is_caught():
    """把 scale 网格转置(误配)-> vLLM 合同形状断言炸 = 测能抓错位(load-bearing)。"""
    torch.manual_seed(1)
    W = torch.randn(512, 1024) * 0.02          # 网格 [4,8]
    fp8, scale = quantize_block_fp8(W, (128, 128), scale_format="float32")
    assert scale.shape == (4, 8)
    with pytest.raises(AssertionError):        # 转置成 [8,4] 违反 cdiv 断言
        _vllm_block_dequant(fp8, scale.t().contiguous(), (128, 128))


def test_mispaired_rolled_scale_collapses_numerically():
    """方形网格下转置不改形状但错位每块 scale -> 数值坍塌(证不是靠形状侥幸)。"""
    torch.manual_seed(2)
    # 方形 weight 使转置 scale 形状仍合法(躲过形状断言),只能靠数值抓。
    N = K = 512                                  # 网格 [4,4] 方形
    W = torch.randn(N, K) * 0.02
    # 逐块注入【非对称】幅值(块(i,j)幅值随 i 强、随 j 弱)=> scale[i,j]!=scale[j,i],
    # 转置后离对角块的 descale 错位 -> 数值坍塌。
    for i in range(4):
        for j in range(4):
            W[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] *= (1.0 + 8.0 * i + 0.2 * j)
    fp8, scale = quantize_block_fp8(W, (128, 128), scale_format="float32")
    good = _frob(_vllm_block_dequant(fp8, scale, (128, 128)), W)
    bad = _frob(_vllm_block_dequant(fp8, scale.t().contiguous(), (128, 128)), W)
    assert good < FROBENIUS_TOL                  # 对位对
    assert bad > 0.2                             # 错位坍塌(区分度)
    assert bad > good * 5
