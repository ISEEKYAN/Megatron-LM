# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Triton BGMV kernels ported from the Mint multi-LoRA training reference.

The dense bank contract is ``A[G, rank, in]``, ``B[G, out, rank]`` and
sorted ``int64`` resident-slot indices.  Each adapter owns a contiguous span
of tokens, allowing the transpose reductions in backward to avoid atomics.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU contract tests.
    _TRITON_AVAILABLE = False


__all__ = [
    "_TRITON_AVAILABLE",
    "bgmv_bwd",
    "bgmv_fwd",
    "compute_group_offsets",
    "use_fused_bgmv",
]

_BLOCK_N = 64
_FUSED_N_THRESHOLD = 256


def use_fused_bgmv(out_features: int, rank: int) -> bool:
    """Whether the production forward launches the fused BGMV kernel."""
    return out_features >= _FUSED_N_THRESHOLD and rank >= 16


_TUNING_CONFIGS = (
    [
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_K": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_K": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_K": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_K": 64, "BLOCK_N": 32}, num_warps=8, num_stages=4
        ),
    ]
    if _TRITON_AVAILABLE
    else []
)


def _l2_prefetch(*tensors: torch.Tensor) -> None:
    """Touch contiguous pool tensors before the Triton launches."""
    for tensor in tensors:
        if tensor.numel():
            flat = tensor.view(-1)
            _ = flat[torch.arange(0, flat.numel(), 64, device=tensor.device)]


def compute_group_offsets(
    lora_indices: torch.Tensor, num_loras: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return CUDA-graph-safe offsets and sizes for sorted resident slots."""
    if num_loras is None:
        num_loras = int(lora_indices.max().item()) + 1 if lora_indices.numel() else 0
    group_sizes = torch.zeros(num_loras, dtype=torch.int32, device=lora_indices.device)
    group_sizes.scatter_add_(
        0, lora_indices, torch.ones_like(lora_indices, dtype=torch.int32)
    )
    group_offsets = torch.zeros(
        num_loras, dtype=torch.int32, device=lora_indices.device
    )
    if num_loras > 1:
        group_offsets[1:] = torch.cumsum(group_sizes[:-1], dim=0)
    return group_offsets, group_sizes, num_loras


if _TRITON_AVAILABLE:

    @triton.autotune(configs=_TUNING_CONFIGS, key=["K", "N", "G"])
    @triton.jit
    def _bgmv_fused_fwd_kernel(
        x_ptr,
        a_ptr,
        b_ptr,
        delta_ptr,
        group_offsets_ptr,
        group_sizes_ptr,
        T,
        K: tl.constexpr,
        RANK: tl.constexpr,
        N: tl.constexpr,
        G: tl.constexpr,
        scale,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr = _BLOCK_N,
    ):
        pid_m, group = tl.program_id(0), tl.program_id(1)
        group_size = tl.load(group_sizes_ptr + group)
        if group_size == 0:
            return
        start = tl.load(group_offsets_ptr + group)
        offset = pid_m * BLOCK_M
        length = tl.minimum(BLOCK_M, group_size - offset)
        if length <= 0:
            return
        offsets_m, offsets_k, offsets_r = (
            tl.arange(0, BLOCK_M),
            tl.arange(0, BLOCK_K),
            tl.arange(0, RANK),
        )
        mask_m = offsets_m < length
        rows = start + offset + offsets_m
        x_ptrs = x_ptr + rows[:, None] * K + offsets_k[None, :]
        a_ptrs = (
            a_ptr + group * (RANK * K) + offsets_r[:, None] * K + offsets_k[None, :]
        )
        hidden = tl.zeros([BLOCK_M, RANK], dtype=tl.float32)
        for block_k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = block_k * BLOCK_K + offsets_k < K
            x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            hidden += tl.dot(x, a.T, input_precision="ieee")
            x_ptrs += BLOCK_K
            a_ptrs += BLOCK_K
        b_base = b_ptr + group * (N * RANK)
        for n_start in range(0, N, BLOCK_N):
            offsets_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offsets_n < N
            b = tl.load(
                b_base + offsets_r[:, None] + offsets_n[None, :] * RANK,
                mask=mask_n[None, :],
            )
            # Match the two-stage kernels exactly.  The default permits lower
            # precision FP32 dot modes on SM90, which makes the fused rank-32
            # path numerically diverge from its independent oracle.
            delta = tl.dot(hidden.to(b.dtype), b, input_precision="ieee") * scale
            tl.store(
                delta_ptr + rows[:, None] * N + offsets_n[None, :],
                delta.to(delta_ptr.dtype.element_ty),
                mask=mask_m[:, None] & mask_n[None, :],
            )

    @triton.autotune(configs=_TUNING_CONFIGS, key=["K", "N", "G"])
    @triton.jit
    def _bgmv_shrink_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        group_offsets_ptr,
        group_sizes_ptr,
        T,
        K: tl.constexpr,
        N: tl.constexpr,
        G: tl.constexpr,
        scale,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr = _BLOCK_N,
    ):
        pid_m, pid_n, group = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        group_size = tl.load(group_sizes_ptr + group)
        if group_size == 0:
            return
        start = tl.load(group_offsets_ptr + group)
        offset = pid_m * BLOCK_M
        length = tl.minimum(BLOCK_M, group_size - offset)
        if length <= 0:
            return
        offsets_m, offsets_n, offsets_k = (
            tl.arange(0, BLOCK_M),
            pid_n * BLOCK_N + tl.arange(0, BLOCK_N),
            tl.arange(0, BLOCK_K),
        )
        mask_m, mask_n = offsets_m < length, offsets_n < N
        rows = start + offset + offsets_m
        x_ptrs = x_ptr + rows[:, None] * K + offsets_k[None, :]
        w_ptrs = w_ptr + group * (N * K) + offsets_n[:, None] * K + offsets_k[None, :]
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for block_k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = block_k * BLOCK_K + offsets_k < K
            x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=mask_n[:, None] & k_mask[None, :], other=0.0)
            acc += tl.dot(x, w.T, input_precision="ieee")
            x_ptrs += BLOCK_K
            w_ptrs += BLOCK_K
        tl.store(
            out_ptr + rows[:, None] * N + offsets_n[None, :],
            (acc * scale).to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.autotune(configs=_TUNING_CONFIGS, key=["K", "N", "G"])
    @triton.jit
    def _bgmv_expand_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        group_offsets_ptr,
        group_sizes_ptr,
        T,
        K: tl.constexpr,
        N: tl.constexpr,
        G: tl.constexpr,
        scale,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr = _BLOCK_N,
    ):
        pid_m, pid_n, group = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        group_size = tl.load(group_sizes_ptr + group)
        if group_size == 0:
            return
        start = tl.load(group_offsets_ptr + group)
        offset = pid_m * BLOCK_M
        length = tl.minimum(BLOCK_M, group_size - offset)
        if length <= 0:
            return
        offsets_m, offsets_n, offsets_k = (
            tl.arange(0, BLOCK_M),
            pid_n * BLOCK_N + tl.arange(0, BLOCK_N),
            tl.arange(0, BLOCK_K),
        )
        mask_m, mask_n = offsets_m < length, offsets_n < N
        rows = start + offset + offsets_m
        x_ptrs = x_ptr + rows[:, None] * K + offsets_k[None, :]
        w_ptrs = w_ptr + group * (K * N) + offsets_k[:, None] * N + offsets_n[None, :]
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for block_k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = block_k * BLOCK_K + offsets_k < K
            x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(x, w, input_precision="ieee")
            x_ptrs += BLOCK_K
            w_ptrs += BLOCK_K * N
        tl.store(
            out_ptr + rows[:, None] * N + offsets_n[None, :],
            (acc * scale).to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.autotune(configs=_TUNING_CONFIGS, key=["M", "N", "G"])
    @triton.jit
    def _bgmv_shrink_transpose_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        group_offsets_ptr,
        group_sizes_ptr,
        T,
        M: tl.constexpr,
        N: tl.constexpr,
        G: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr = _BLOCK_N,
        scale: tl.constexpr = 1.0,
    ):
        pid_m, pid_n, group = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        group_size = tl.load(group_sizes_ptr + group)
        if group_size == 0:
            return
        start = tl.load(group_offsets_ptr + group)
        offsets_m, offsets_n, offsets_k = (
            pid_m * BLOCK_N + tl.arange(0, BLOCK_N),
            pid_n * BLOCK_N + tl.arange(0, BLOCK_N),
            tl.arange(0, BLOCK_K),
        )
        mask_m, mask_n = offsets_m < M, offsets_n < N
        x_ptrs = x_ptr + (start + offsets_k[None, :]) * M + offsets_m[:, None]
        w_ptrs = w_ptr + (start + offsets_k[:, None]) * N + offsets_n[None, :]
        acc = tl.zeros([BLOCK_N, BLOCK_N], dtype=tl.float32)
        for block_k in range(0, tl.cdiv(group_size, BLOCK_K)):
            k_mask = block_k * BLOCK_K + offsets_k < group_size
            x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(x, w, input_precision="ieee")
            x_ptrs += BLOCK_K * M
            w_ptrs += BLOCK_K * N
        tl.store(
            out_ptr + group * (M * N) + offsets_m[:, None] * N + offsets_n[None, :],
            (acc * scale).to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def bgmv_fwd(x, lora_a, lora_b, lora_indices, scale, max_g_size_hint=None):
    """Mint two-stage BGMV forward, saving hidden for the four-kernel backward."""
    tokens = x.shape[0]
    groups, rank, in_features = lora_a.shape
    out_features = lora_b.shape[1]
    offsets, sizes, groups = compute_group_offsets(lora_indices, num_loras=groups)
    max_group = (
        max_g_size_hint
        if max_g_size_hint is not None
        else (int(sizes.amax().item()) if groups else 1)
    )
    _l2_prefetch(lora_a, lora_b)
    # ``hidden`` is saved for grad_B.  Along with ``grad_hidden`` below, these
    # are the only T×R scratch buffers deliberately widened to FP32.  Relative
    # to the former BF16 storage, their combined peak increment is
    # 2 * tokens * rank * (4 - 2) = 4 * tokens * rank bytes.
    hidden = torch.empty(tokens, rank, device=x.device, dtype=torch.float32)

    def grid_hidden(meta):
        return (
            triton.cdiv(max_group, meta["BLOCK_M"]),
            triton.cdiv(rank, meta["BLOCK_N"]),
            groups,
        )

    _bgmv_shrink_kernel[grid_hidden](
        x,
        lora_a,
        hidden,
        offsets,
        sizes,
        tokens,
        K=in_features,
        N=rank,
        G=groups,
        scale=1.0,
    )
    delta = torch.empty(tokens, out_features, device=x.device, dtype=x.dtype)
    # Triton SM90 dot requires K >= 16.  Rank-8 banks retain the complete
    # two-stage path; the fused optimization is available once rank permits it.
    if use_fused_bgmv(out_features, rank):
        _bgmv_fused_fwd_kernel[
            lambda meta: (triton.cdiv(max_group, meta["BLOCK_M"]), groups)
        ](
            x,
            lora_a,
            lora_b,
            delta,
            offsets,
            sizes,
            tokens,
            K=in_features,
            RANK=rank,
            N=out_features,
            G=groups,
            scale=scale,
        )
    else:

        def grid_delta(meta):
            return (
                triton.cdiv(max_group, meta["BLOCK_M"]),
                triton.cdiv(out_features, meta["BLOCK_N"]),
                groups,
            )

        _bgmv_shrink_kernel[grid_delta](
            hidden,
            lora_b,
            delta,
            offsets,
            sizes,
            tokens,
            K=rank,
            N=out_features,
            G=groups,
            scale=scale,
        )
    return delta, hidden


def bgmv_bwd(
    x, grad_out, lora_a, lora_b, lora_indices, scale, hidden=None, max_g_size_hint=None
):
    """Mint four-kernel training backward: grad_hidden, grad_x, grad_A, grad_B."""
    tokens = x.shape[0]
    groups, rank, in_features = lora_a.shape
    out_features = lora_b.shape[1]
    offsets, sizes, groups = compute_group_offsets(lora_indices, num_loras=groups)
    max_group = (
        max_g_size_hint
        if max_g_size_hint is not None
        else (int(sizes.amax().item()) if groups else 1)
    )
    grad_out = grad_out.contiguous()

    def grid_hidden(meta):
        return (
            triton.cdiv(max_group, meta["BLOCK_M"]),
            triton.cdiv(rank, meta["BLOCK_N"]),
            groups,
        )

    # Keep the backward intermediate FP32 until the public gradients are stored.
    grad_hidden = torch.empty(tokens, rank, device=x.device, dtype=torch.float32)
    _bgmv_expand_kernel[grid_hidden](
        grad_out,
        lora_b,
        grad_hidden,
        offsets,
        sizes,
        tokens,
        K=out_features,
        N=rank,
        G=groups,
        scale=scale,
    )
    grad_x = torch.empty_like(x)

    def grid_x(meta):
        return (
            triton.cdiv(max_group, meta["BLOCK_M"]),
            triton.cdiv(in_features, meta["BLOCK_N"]),
            groups,
        )

    _bgmv_expand_kernel[grid_x](
        grad_hidden,
        lora_a,
        grad_x,
        offsets,
        sizes,
        tokens,
        K=rank,
        N=in_features,
        G=groups,
        scale=1.0,
    )
    grad_a = torch.zeros_like(lora_a)

    def grid_a(meta):
        return (
            triton.cdiv(rank, meta["BLOCK_N"]),
            triton.cdiv(in_features, meta["BLOCK_N"]),
            groups,
        )

    _bgmv_shrink_transpose_kernel[grid_a](
        grad_hidden, x, grad_a, offsets, sizes, tokens, M=rank, N=in_features, G=groups
    )
    if hidden is None:
        hidden = torch.empty(tokens, rank, device=x.device, dtype=torch.float32)
        _bgmv_shrink_kernel[grid_hidden](
            x,
            lora_a,
            hidden,
            offsets,
            sizes,
            tokens,
            K=in_features,
            N=rank,
            G=groups,
            scale=1.0,
        )
    grad_b = torch.zeros_like(lora_b)

    def grid_b(meta):
        return (
            triton.cdiv(out_features, meta["BLOCK_N"]),
            triton.cdiv(rank, meta["BLOCK_N"]),
            groups,
        )

    _bgmv_shrink_transpose_kernel[grid_b](
        grad_out,
        hidden,
        grad_b,
        offsets,
        sizes,
        tokens,
        M=out_features,
        N=rank,
        G=groups,
        scale=scale,
    )
    return grad_x, grad_a, grad_b
