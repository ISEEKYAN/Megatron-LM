# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU operand/reduction reference for MXFP4 experts with clamped SwiGLU.

No vLLM or CUDA imports: training implementations can reuse these boundaries.
This is not a reference for the tensor-core GEMM accumulation order. FP32
transcendental implementations can differ across devices; compare the BF16
activation and quantized bytes explicitly rather than assuming equivalence.
"""

import ctypes
import ctypes.util

import torch


def swiglu_clamp(x: torch.Tensor, limit: float | None = 10.0) -> torch.Tensor:
    """Clamp gate above only and up on both sides, then round once to BF16."""
    assert x.device.type == "cpu" and x.dtype == torch.bfloat16
    gate, up = x.float().chunk(2, dim=-1)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    return (gate / (1.0 + torch.exp(-gate)) * up).to(torch.bfloat16)


def quantize_a8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E4M3 codes and FP32 power-of-two scales for per-token g128 A."""
    assert x.device.type == "cpu" and x.ndim == 2 and x.shape[1] % 128 == 0
    groups = x.float().reshape(x.shape[0], x.shape[1] // 128, 128)
    amax = groups.abs().amax(-1).clamp_min(1e-10)
    raw = (amax / 448.0).clamp_min(1e-10).contiguous().view(torch.int32)
    exponent = ((raw >> 23) & 255) + ((raw & 0x7FFFFF) != 0).int()
    scale = (exponent << 23).view(torch.float32)
    codes = (groups / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return codes.reshape(x.shape), scale


def topk_fma(rows: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Combine [token, slot, hidden] BF16 rows with one FP32 FMA per slot."""
    assert rows.device.type == weights.device.type == "cpu"
    assert rows.dtype == torch.bfloat16 and weights.dtype == torch.float32
    assert rows.shape[:2] == weights.shape
    libm = ctypes.CDLL(ctypes.util.find_library("m"))
    fma = libm.fmaf
    fma.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    fma.restype = ctypes.c_float
    values, probs = rows.float().tolist(), weights.tolist()
    result = torch.empty((rows.shape[0], rows.shape[2]), dtype=torch.float32)
    for token in range(rows.shape[0]):
        for col in range(rows.shape[2]):
            acc = 0.0
            for slot in range(rows.shape[1]):
                acc = fma(values[token][slot][col], probs[token][slot], acc)
            result[token, col] = acc
    return result.to(torch.bfloat16)
