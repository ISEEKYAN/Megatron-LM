"""BF16 DeepGEMM helpers used by the native mLite MoE backward."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def _should_log_deepgemm_summary() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _format_int_ranges(values: Iterable) -> str:
    sorted_values = sorted({int(value) for value in values})
    if not sorted_values:
        return "[]"
    ranges = []
    start = previous = sorted_values[0]
    for value in sorted_values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _deepgemm_bf16_gemm_nn(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[0]:
        raise RuntimeError(f"BF16 NN GEMM shape mismatch: {tuple(lhs.shape)} x {tuple(rhs.shape)}")
    if lhs.is_cuda and rhs.is_cuda and lhs.dtype == torch.bfloat16 and rhs.dtype == torch.bfloat16:
        import deep_gemm

        output = torch.empty((lhs.shape[0], rhs.shape[1]), device=lhs.device, dtype=torch.bfloat16)
        deep_gemm.bf16_gemm_nn(lhs.contiguous(), rhs.contiguous(), output)
        return output
    return lhs.matmul(rhs).to(lhs.dtype)


def _deepgemm_bf16_gemm_nt(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[1]:
        raise RuntimeError(f"BF16 NT GEMM shape mismatch: {tuple(lhs.shape)} x {tuple(rhs.shape)}")
    if lhs.is_cuda and rhs.is_cuda and lhs.dtype == torch.bfloat16 and rhs.dtype == torch.bfloat16:
        import deep_gemm

        output = torch.empty((lhs.shape[0], rhs.shape[0]), device=lhs.device, dtype=torch.bfloat16)
        deep_gemm.bf16_gemm_nt(lhs.contiguous(), rhs.contiguous(), output)
        return output
    return lhs.matmul(rhs.transpose(0, 1)).to(lhs.dtype)


def _deepgemm_bf16_gemm_tn(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[0] != rhs.shape[0]:
        raise RuntimeError(f"BF16 TN GEMM shape mismatch: {tuple(lhs.shape)} x {tuple(rhs.shape)}")
    if lhs.is_cuda and rhs.is_cuda and lhs.dtype == torch.bfloat16 and rhs.dtype == torch.bfloat16:
        import deep_gemm

        output = torch.empty((lhs.shape[1], rhs.shape[1]), device=lhs.device, dtype=torch.bfloat16)
        deep_gemm.bf16_gemm_tn(lhs.contiguous(), rhs.contiguous(), output)
        return output
    return lhs.transpose(0, 1).matmul(rhs).to(lhs.dtype)


def _sum_to_parameter_dtype(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return value.to(dtype=reference.dtype)
