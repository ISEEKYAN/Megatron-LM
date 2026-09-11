# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Numerical bindings of V4.1 checkpoint bytes; archival storage stays unchanged."""

import torch

from megatron.lite.primitive.quantization.block_fp8 import dequantize_block_fp8
from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4

_TORCH_DTYPES = {
    "I8": torch.int8,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
}


def _tensor(store, name):
    entry = store.entries[name]
    if entry.dtype not in _TORCH_DTYPES:
        raise TypeError(f"unsupported numerical dtype: {entry.dtype}")
    dtype = _TORCH_DTYPES[entry.dtype]
    if entry.byte_length == 0:
        return torch.empty(entry.shape, dtype=dtype)
    # Own the backing storage; neither the immutable entry nor a mapped file is mutated.
    return torch.frombuffer(bytearray(store.read(name)), dtype=dtype).reshape(
        entry.shape
    )


def load_weight(store, name, *, output_dtype=torch.bfloat16):
    """Decode one weight with its exact scale sibling, or reload a plain export.

    This is a CPU numerical binding, not the runtime FP8 activation/GEMM path.
    Engram tables use row-by-32 scales; ordinary matrices use 32-by-32 scales.
    """
    if output_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError("output dtype must be F32, BF16 or F16")
    if not name.endswith(".weight"):
        raise ValueError("load_weight requires an explicit .weight key")
    weight = _tensor(store, name)
    scale_name = name[:-6] + "scale"
    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
        if scale_name in store.entries:
            raise ValueError(f"unexpected scale for plain weight: {name}")
        result = weight.float()
    else:
        if weight.ndim != 2:
            raise ValueError("quantized weights must be matrices")
        if scale_name not in store.entries:
            raise ValueError(f"missing scale: {scale_name}")
        scale = _tensor(store, scale_name)
        if scale.dtype != torch.float8_e8m0fnu:
            raise TypeError("release scales must be E8M0")
        if not torch.isfinite(scale.float()).all():
            raise ValueError("nonfinite release scale")
        if weight.dtype == torch.int8:
            if weight.shape[-1] % 16:
                raise ValueError("packed FP4 width must be divisible by 16")
            result = dequantize_mxfp4(weight, scale)
        elif weight.dtype == torch.float8_e4m3fn:
            rows, columns = weight.shape
            row_block = 1 if name.endswith(".engram.embed.weight") else 32
            expected = ((rows + row_block - 1) // row_block, (columns + 31) // 32)
            if tuple(scale.shape) != expected:
                raise ValueError(
                    f"scale shape mismatch: {tuple(scale.shape)} != {expected}"
                )
            # Reuse the aligned primitive, allowing a final partially occupied block.
            padded = torch.zeros(
                expected[0] * row_block, expected[1] * 32, dtype=weight.dtype
            )
            padded[:rows, :columns] = weight
            result = dequantize_block_fp8(padded, scale, (row_block, 32))[
                :rows, :columns
            ]
        else:
            raise TypeError(f"unsupported weight dtype: {weight.dtype}")
    result = result.to(output_dtype)
    if not torch.isfinite(result).all():
        raise ValueError(f"nonfinite decoded weight: {name}")
    return result


def load_engram_rows(store, name, *, intervals, rank, device, chunk_rows=4096):
    """Load only the assigned FP8/E8M0 rows, with bounded host staging.

    CPU is supported for byte-level fixtures; runtime callers pass their CUDA
    device. No decoded full table or persistent host replica is created.
    """
    if not name.endswith('.engram.embed.weight'):
        raise ValueError('Require exact Engram weight family')
    weight = store.entries[name]
    scale_name = name[:-6] + 'scale'
    scale = store.entries[scale_name]
    if (weight.dtype != 'F8_E4M3' or len(weight.shape) != 2
            or weight.shape[1] % 32 or scale.dtype != 'F8_E8M0'
            or scale.shape != (weight.shape[0], weight.shape[1] // 32)):
        raise ValueError('Engram requires FP8 values and matching row/block32 E8M0 scales')
    intervals = tuple(intervals)
    cursor = 0
    for begin, end in intervals:
        if type(begin) is not int or type(end) is not int or begin != cursor or end < begin:
            raise ValueError('Row coverage has gaps, overlaps or invalid intervals')
        cursor = end
    if cursor != weight.shape[0] or not 0 <= rank < len(intervals):
        raise ValueError('Row coverage or rank does not match logical table')
    begin, end = intervals[rank]
    tensors = []
    for key, entry in ((name, weight), (scale_name, scale)):
        result = torch.empty((end - begin, entry.shape[1]), device=device, dtype=torch.uint8)
        for first, raw in store.iter_rows(key, begin, end, chunk_rows=chunk_rows):
            chunk = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(-1, entry.shape[1])
            result[first - begin:first - begin + chunk.shape[0]].copy_(chunk)
        tensors.append(result.view(_TORCH_DTYPES[entry.dtype]))
    return tuple(tensors)
