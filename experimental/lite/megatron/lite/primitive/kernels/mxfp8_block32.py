# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""One batched FP8 launch with fixed K32 scale and FP32 accumulation order."""

import torch
import triton
import triton.language as tl


@triton.jit
def _block32_mm(
    A,
    AS,
    W,
    WS,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    AM: tl.constexpr,
    AG: tl.constexpr,
    AK: tl.constexpr,
    ASM: tl.constexpr,
    ASG: tl.constexpr,
    ASK: tl.constexpr,
    WN: tl.constexpr,
    WG: tl.constexpr,
    WK: tl.constexpr,
    WSN: tl.constexpr,
    WSG: tl.constexpr,
    WSK: tl.constexpr,
    PACKED: tl.constexpr,
    SCALE_ROWS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    group = tl.program_id(2)
    kk = tl.arange(0, 32)
    result = tl.zeros((BM, BN), tl.float32)
    for block in range(K // 32):
        a = tl.load(
            A + row[:, None] * AM + group * AG + (block * 32 + kk[None, :]) * AK,
            mask=row[:, None] < M,
            other=0.0,
        )
        w = tl.load(
            W + col[None, :] * WN + group * WG + (block * 32 + kk[:, None]) * WK,
            mask=col[None, :] < N,
            other=0.0,
        )
        scale_block = block // 4 if PACKED else block
        sa = tl.load(
            AS + row * ASM + group * ASG + scale_block * ASK, mask=row < M, other=127
        ).to(tl.uint32)
        sw = tl.load(
            WS + (col // SCALE_ROWS) * WSN + group * WSG + scale_block * WSK,
            mask=col < N,
            other=127,
        ).to(tl.uint32)
        if PACKED:
            shift = (block % 4) * 8
            sa = (sa >> shift) & 255
            sw = (sw >> shift) & 255
        sa = tl.where(
            sa == 0, 0x00400000, tl.where(sa == 255, 0x7FC00000, sa << 23)
        ).to(tl.float32, bitcast=True)
        sw = tl.where(
            sw == 0, 0x00400000, tl.where(sw == 255, 0x7FC00000, sw << 23)
        ).to(tl.float32, bitcast=True)
        product = tl.dot(a, w, out_dtype=tl.float32)
        scaled = product * sa[:, None]
        scaled = scaled * sw[None, :]
        result = result + scaled
    tl.store(
        Output + (row[:, None] * G + group) * N + col[None, :],
        result,
        mask=(row[:, None] < M) & (col[None, :] < N),
    )


def batched_block32_mm(
    a,
    a_scale,
    weight,
    weight_scale,
    output_dtype,
    *,
    grouped=False,
    weight_scale_group_rows=1
):
    """Read strided FP8/E8M0 packets directly; never launch per row/group/K block."""
    if a_scale.dtype == torch.float8_e8m0fnu:
        a_scale = a_scale.view(torch.uint8)
    if weight_scale.dtype == torch.float8_e8m0fnu:
        weight_scale = weight_scale.view(torch.uint8)
    if grouped:
        m, groups, k = a.shape
        n = weight.shape[1]
        am, ag, ak = a.stride()
        asm, asg, ask = a_scale.stride()
        wg, wn, wk = weight.stride()
        wsg, wsn, wsk = weight_scale.stride()
    else:
        m, k = a.shape
        n, groups = weight.shape[0], 1
        am, ak = a.stride()
        asm, ask = a_scale.stride()
        wn, wk = weight.stride()
        wsn, wsk = weight_scale.stride()
        ag = asg = wg = wsg = 0
    if k % 32 or output_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError('K32 GEMM requires K divisible by 32 and floating output')
    if a.dtype != torch.float8_e4m3fn or weight.dtype != a.dtype:
        raise ValueError('K32 GEMM requires E4M3 operands')
    packed = a_scale.dtype == torch.int32
    if weight_scale.dtype != a_scale.dtype or a_scale.dtype not in (
        torch.uint8,
        torch.int32,
    ):
        raise ValueError('K32 GEMM requires paired E8M0 bytes or packed INT32 scales')
    output = torch.empty(
        (m, groups, n) if grouped else (m, n), device=a.device, dtype=output_dtype
    )
    if m and n:
        _block32_mm[(triton.cdiv(m, 16), triton.cdiv(n, 64), groups)](
            a,
            a_scale,
            weight,
            weight_scale,
            output,
            m,
            n,
            k,
            groups,
            am,
            ag,
            ak,
            asm,
            asg,
            ask,
            wn,
            wg,
            wk,
            wsn,
            wsg,
            wsk,
            packed,
            weight_scale_group_rows,
            16,
            64,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
