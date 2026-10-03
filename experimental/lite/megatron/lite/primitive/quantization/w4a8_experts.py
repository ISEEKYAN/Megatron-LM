# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""W4A8 routed-expert forward: dynamic FP8 activations x MXFP4 weights.

Training-side counterpart of the rollout MoE contract served by vLLM's
``DeepGemmFP4Experts`` (``m_grouped_fp8_fp4_gemm_nt_contiguous``):

* **Activations** — dynamic, per token, one scale per 128 contiguous K values,
  E4M3 codes with a UE8M0 (power-of-two) scale; see :func:`quantize_fp8_act`.
  The scale is recomputed from the live tensor on every call, so there is no
  observer, calibration state or cross-rank amax to synchronize.
* **Weights** — MXFP4 (E2M1, one UE8M0 scale per 32 K values) produced by
  :func:`~megatron.lite.primitive.quantization.mxfp4.quantize_mxfp4`, the same
  function the QAT exporter uses, so the forward consumes exactly the bytes the
  rollout loads.
* **GEMM** — DeepGEMM on CUDA (no fallback: a missing kernel is an error). On
  CPU the reference multiplies the same dequantized operands in float32. The
  quantized operands and scales are bit-identical between the two; the
  accumulation order of the GEMM itself is the kernel's.
* **Expert MLP** (:func:`w4a8_expert_mlp`) — FC1 output in BF16, SwiGLU in
  float32 rounded once to BF16, A8 requantization, FC2 output in BF16, router
  probabilities applied after FC2 in float32.

Backward is a straight-through estimator on both operands: gradients use the
dequantized activations and weights the forward multiplied, and flow unmasked
to the BF16 input and the BF16 master weights. No saturation mask is needed:
the dynamic activation scale satisfies ``amax / scale <= 448`` and the MXFP4
scale rule satisfies ``amax / scale <= 6`` by construction.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence

import torch
from megatron.lite.primitive.quantization.mxfp4 import (
    MXFP4_BLOCK_SIZE,
    dequantize_mxfp4,
    quantize_mxfp4,
)

FP8_ACT_GROUP_SIZE = 128
_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0
# vLLM initializes the group absmax at eps=1e-10 and floors the scale at 1e-10.
_SCALE_FLOOR = 1e-10

__all__ = [
    "FP8_ACT_GROUP_SIZE",
    "dequantize_fp8_act",
    "quantize_fp8_act",
    "w4a8_expert_mlp",
    "w4a8_grouped_gemm",
]


def _ceil_ue8m0(scale: torch.Tensor) -> torch.Tensor:
    """Round a positive normal float32 scale up to a power of two, bit-exactly.

    Same integer rule as vLLM's packed per-token-group quant kernel and
    DeepGEMM's ``ceil_to_ue8m0``: keep the exponent field and add one if any
    mantissa bit is set. A float ``ceil(log2(s))`` can miss by one ulp.
    """
    bits = scale.float().contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return (exponent << 23).view(torch.float32)


def quantize_fp8_act(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token-group (128) E4M3 quantization with UE8M0 scales.

    Returns ``(codes, scale)``: ``codes`` is ``float8_e4m3fn`` with the shape of
    ``x``; ``scale`` is float32 ``[M, K // 128]`` holding exact powers of two, so
    ``codes * scale`` reconstructs the dequantized activation. Per group::

        amax  = max(|x|, 1e-10)
        scale = ceil_ue8m0(max(amax / 448, 1e-10))
        codes = e4m3_rne(clamp(x / scale, -448, 448))
    """
    if x.dim() != 2 or x.shape[-1] % FP8_ACT_GROUP_SIZE:
        raise ValueError(
            f"FP8 activation quantization expects [M, K] with K divisible by "
            f"{FP8_ACT_GROUP_SIZE}, got {tuple(x.shape)}."
        )
    rows, cols = x.shape
    groups = x.float().reshape(rows, cols // FP8_ACT_GROUP_SIZE, FP8_ACT_GROUP_SIZE)
    amax = groups.abs().amax(dim=-1).clamp_min(_SCALE_FLOOR)
    scale = _ceil_ue8m0((amax / _E4M3_MAX).clamp_min(_SCALE_FLOOR))
    codes = (groups / scale.unsqueeze(-1)).clamp(-_E4M3_MAX, _E4M3_MAX)
    return codes.to(torch.float8_e4m3fn).reshape(rows, cols), scale


def dequantize_fp8_act(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quantize_fp8_act` in float32."""
    rows, cols = codes.shape
    groups = codes.float().reshape(rows, cols // FP8_ACT_GROUP_SIZE, FP8_ACT_GROUP_SIZE)
    return (groups * scale.unsqueeze(-1)).reshape(rows, cols)


def _row_ranges(m_splits: Sequence[int]) -> list[tuple[int, int]]:
    ranges, start = [], 0
    for count in m_splits:
        ranges.append((start, start + count))
        start += count
    return ranges


def _reference_gemm(x_codes, x_scale, w_codes, w_scale, m_splits) -> torch.Tensor:
    x_hat = dequantize_fp8_act(x_codes, x_scale)
    out = x_hat.new_empty((x_hat.shape[0], w_codes.shape[1]))
    for expert, (start, end) in enumerate(_row_ranges(m_splits)):
        w_hat = dequantize_mxfp4(w_codes[expert], w_scale[expert])
        out[start:end] = x_hat[start:end] @ w_hat.t()
    return out.to(torch.bfloat16)


_DEEP_GEMM_MODULES = ("deep_gemm", "vllm.third_party.deep_gemm")


def _import_deep_gemm():
    """Resolve DeepGEMM in vLLM's order: an installed ``deep_gemm``, then the
    copy vendored in the vLLM wheel. Both are the same kernel library; nothing
    else is accepted."""
    for name in _DEEP_GEMM_MODULES:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise RuntimeError(
        "W4A8 experts on CUDA require DeepGEMM (m_grouped_fp8_fp4_gemm_nt_contiguous) "
        f"from one of {_DEEP_GEMM_MODULES}; refusing to fall back to a different GEMM."
    )


def _deep_gemm_weight_scales(deep_gemm, w_scale: torch.Tensor, k: int) -> torch.Tensor:
    """MXFP4 E8M0 scales ``[E, N, K // 32]`` in the layout the rollout hands DeepGEMM.

    Same steps as vLLM's MXFP4 weight loading for this backend
    (``deepgemm_post_process_weight_scale_block``): the exponent byte goes into
    the float32 exponent field (byte 0 becomes 0.0, not the denormal 2^-127 that
    ``float8_e8m0fnu.float()`` gives), then DeepGEMM packs it into int32 UE8M0.
    The CPU reference and the backward still dequantize byte 0 as 2^-127; that
    only matters for nonzero codes under byte 0 (|w| below ~3.5e-38), which
    the MXFP4 scale rule produces only for blocks that are entirely zero.
    """
    sf = (w_scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    return deep_gemm.transform_sf_into_required_layout(
        sf=sf,
        mn=w_scale.shape[1],
        k=k,
        recipe=(1, 1, MXFP4_BLOCK_SIZE),
        num_groups=w_scale.shape[0],
        is_sfa=False,
        disable_ue8m0_cast=False,
    )


def _deep_gemm_grouped(x_codes, x_scale, w_codes, w_scale, m_splits) -> torch.Tensor:
    deep_gemm = _import_deep_gemm()
    alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    device = x_codes.device
    real_rows, layout, padded = [], [], 0
    for expert, count in enumerate(m_splits):
        aligned = (count + alignment - 1) // alignment * alignment
        real_rows.append(torch.arange(padded, padded + count, device=device))
        layout.append(
            torch.full((aligned,), -1, dtype=torch.int32, device=device).index_fill_(
                0, torch.arange(count, device=device), expert
            )
        )
        padded += aligned
    rows = torch.cat(real_rows)
    grouped_layout = torch.cat(layout)
    a_codes = torch.zeros(
        (padded, x_codes.shape[1]), dtype=torch.uint8, device=device
    ).index_copy_(0, rows, x_codes.view(torch.uint8))
    a_scale = torch.ones(
        (padded, x_scale.shape[1]), dtype=torch.float32, device=device
    ).index_copy_(0, rows, x_scale)
    out = torch.empty((padded, w_codes.shape[1]), dtype=torch.bfloat16, device=device)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (a_codes.view(torch.float8_e4m3fn), a_scale),
        (w_codes, _deep_gemm_weight_scales(deep_gemm, w_scale, x_codes.shape[1])),
        out,
        grouped_layout,
        recipe_a=(1, FP8_ACT_GROUP_SIZE),
        recipe_b=(1, MXFP4_BLOCK_SIZE),
    )
    return out.index_select(0, rows)


_BACKENDS = {"deep_gemm": _deep_gemm_grouped, "reference": _reference_gemm}


class _W4A8GroupedGemm(torch.autograd.Function):
    """``out[rows_e] = Q_a8(x[rows_e]) @ Q_mxfp4(W_e).T`` with STE backward."""

    @staticmethod
    def forward(ctx, x, m_splits, backend, *weights):  # type: ignore[override]
        x_codes, x_scale = quantize_fp8_act(x)
        packed = [quantize_mxfp4(weight.detach()) for weight in weights]
        w_codes = torch.stack([codes for codes, _ in packed])
        w_scale = torch.stack([scale for _, scale in packed])
        ctx.save_for_backward(x_codes, x_scale, w_codes, w_scale)
        ctx.m_splits = m_splits
        ctx.weight_dtypes = [weight.dtype for weight in weights]
        return _BACKENDS[backend](x_codes, x_scale, w_codes, w_scale, m_splits)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        x_codes, x_scale, w_codes, w_scale = ctx.saved_tensors
        x_hat = dequantize_fp8_act(x_codes, x_scale).to(grad_out.dtype)
        grad_x = torch.empty_like(x_hat)
        grad_weights = []
        for expert, (start, end) in enumerate(_row_ranges(ctx.m_splits)):
            w_hat = dequantize_mxfp4(w_codes[expert], w_scale[expert])
            grad = grad_out[start:end]
            grad_x[start:end] = grad @ w_hat.to(grad.dtype)
            grad_w = grad.t() @ x_hat[start:end]
            grad_weights.append(grad_w.to(ctx.weight_dtypes[expert]))
        return (grad_x, None, None, *grad_weights)


def _default_backend(x: torch.Tensor) -> str:
    return "deep_gemm" if x.is_cuda else "reference"


def w4a8_grouped_gemm(
    x: torch.Tensor,
    weights: Sequence[torch.Tensor],
    m_splits: Sequence[int],
    *,
    backend: str | None = None,
) -> torch.Tensor:
    """Grouped ``[M, K] x [N, K]^T`` per expert with A8 activations and MXFP4 weights.

    ``x`` holds the expert-sorted BF16 tokens (``m_splits[e]`` rows for expert
    ``e``); ``weights`` are the per-expert BF16 master weights. ``backend`` is
    ``"deep_gemm"`` or ``"reference"``; ``None`` picks DeepGEMM on CUDA and the
    reference on CPU.
    """
    if x.dtype != torch.bfloat16:
        raise TypeError(f"W4A8 experts take BF16 activations, got {x.dtype}.")
    if len(weights) != len(m_splits) or sum(m_splits) != x.shape[0]:
        raise ValueError(
            f"m_splits {list(m_splits)} must give one row count per expert weight "
            f"({len(weights)}) and sum to the token count ({x.shape[0]})."
        )
    backend = _default_backend(x) if backend is None else backend
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown W4A8 backend {backend!r}; use one of {sorted(_BACKENDS)}."
        )
    return _W4A8GroupedGemm.apply(x, tuple(m_splits), backend, *weights)


def w4a8_expert_mlp(
    x: torch.Tensor,
    fc1_weights: Sequence[torch.Tensor],
    fc2_weights: Sequence[torch.Tensor],
    m_splits: Sequence[int],
    probs: torch.Tensor | None,
    swiglu_limit: float = 0.0,
    *,
    backend: str | None = None,
) -> torch.Tensor:
    """Routed-expert MLP in the rollout's W4A8 order of operations.

    FC1 (``[gate, up]`` halves) -> SwiGLU computed in float32 and rounded once
    to BF16 -> FC2 -> multiply by the router probability in float32. With
    ``swiglu_limit > 0`` the gate is clamped from above and ``up`` symmetrically
    before the activation, as in the rollout kernel.
    """
    fc1_out = w4a8_grouped_gemm(x, fc1_weights, m_splits, backend=backend)
    gate, up = fc1_out.float().chunk(2, dim=-1)
    if swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(-swiglu_limit, swiglu_limit)
    hidden = (torch.nn.functional.silu(gate) * up).to(fc1_out.dtype)
    out = w4a8_grouped_gemm(hidden, fc2_weights, m_splits, backend=backend)
    if probs is not None:
        out = (out.float() * probs.float()).to(out.dtype)
    return out
